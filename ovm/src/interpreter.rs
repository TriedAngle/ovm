use bytecode::{Opcode, Operand, Scale};

use vm::{
    Array, CallableObject, HeapObject, HeapRef, InternedString, LocalHeap, Lookup, Map, NoGc,
    SlotName, SlotsObject, SlotsObjectInit, Smi, Tagged, Value, ValueRef,
};

use crate::{ContextState, Heap, NativeContext, NativeIndex, VM, VmError};

pub fn run<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    callable: Tagged<CallableObject>,
    args: &[Value],
) -> Result<Value, VmError> {
    let call_name = state.handle_scope(|scope| {
        SlotName::from(vm.interner().intern(heap, &scope, "call").as_tagged())
    });

    let stack = &state.stack;
    let cache = &state.cache;

    let result = stack.push_initial_frame(callable, args).and_then(|frame| {
        cache.enter(stack, frame);
        let result = dispatch(vm, heap, state, call_name);
        cache.deactivate();
        result
    });

    stack.set_top(0);
    stack.clear_frames();

    result
}

fn code_ref<'a, L: LocalHeap>(
    nogc: &'a NoGc<'a>,
    heap: &'a L,
    callable: Value,
) -> HeapRef<'a, CallableObject> {
    nogc.get_as::<CallableObject>(callable, heap.known().callable_map)
        .expect("frame code must be a callable object")
}

fn property_name<'a, L: LocalHeap>(
    nogc: &'a NoGc<'a>,
    heap: &'a L,
    constants: HeapRef<'a, Array>,
    idx: usize,
) -> SlotName {
    let v = constants.at(idx);
    let name = nogc
        .get_as::<InternedString>(v, heap.known().string_map)
        .expect("property name constant must be an interned string");
    SlotName::from(name.into_tagged())
}

fn dispatch<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    call_name: SlotName,
) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;
    let mut acc = heap.known().void.value();
    loop {
        let (op, operands, next_pc) =
            heap.no_gc(|nogc, _| decode(cache.code_ref(nogc).as_slice(), cache.pc()));
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        match op {
            Opcode::Return => match stack.pop_frame(meta.base) {
                Some(caller) => cache.load(stack, caller),
                None => return Ok(acc),
            },
            Opcode::Load => {
                acc = stack.reg(&meta, operands[0] as i32);
            }
            Opcode::Store => {
                stack.set_reg(&meta, operands[0] as i32, acc);
            }
            Opcode::Move => {
                let v = stack.reg(&meta, operands[1] as i32);
                stack.set_reg(&meta, operands[0] as i32, v);
            }
            Opcode::LoadSmi => {
                acc = Smi::new(operands[0] as i32 as i64).encode();
            }
            Opcode::LoadConstant => {
                let v = heap.no_gc(|nogc, _| cache.constants_ref(nogc).at(operands[0] as usize));
                acc = v;
            }
            Opcode::Add => {
                let a = Smi::decode(stack.reg(&meta, operands[0] as i32)).ok_or(VmError::Type)?;
                let b = Smi::decode(stack.reg(&meta, operands[1] as i32)).ok_or(VmError::Type)?;
                let r = a.value().checked_add(b.value()).ok_or(VmError::Overflow)?;
                if !Smi::in_range(r) {
                    return Err(VmError::Overflow);
                }
                acc = Smi::new(r).encode();
            }
            Opcode::CallNative => {
                let f = vm.native(NativeIndex(operands[0] as usize));
                let count = operands[2] as usize;
                let mut nctx = NativeContext::new(vm, heap, state);
                cache.spill_acc(acc);
                let result = f(&mut nctx, stack.args(&meta, operands[1] as i32, count));
                let _ = cache.take_acc();
                acc = result?;
            }
            Opcode::Call | Opcode::CallNoFeedback => {
                let count = operands[2] as usize;
                let target = heap.no_gc(|nogc, heap| {
                    match stack
                        .reg_slot(&meta, operands[0] as i32)
                        .lookup(nogc, heap, call_name)
                    {
                        Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => nogc
                            .get_as::<CallableObject>(slot.inner(), heap.known().callable_map)
                            .map(|r| r.into_tagged()),
                        _ => None,
                    }
                });
                let Some(target) = target else {
                    return Err(VmError::Type); // not callable
                };
                let callee = stack.push_frame(meta, target, operands[1] as i32, count)?;
                cache.load(stack, callee);
            }
            Opcode::LoadNamedProperty => {
                let v = heap.no_gc(|nogc, heap| {
                    let name =
                        property_name(nogc, heap, cache.constants_ref(nogc), operands[1] as usize);
                    match stack
                        .reg_slot(&meta, operands[0] as i32)
                        .lookup(nogc, heap, name)
                    {
                        Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => slot.inner(),
                        _ => unimplemented!("TODO: implement accessors"),
                    }
                });
                acc = v;
            }
            Opcode::StoreNamedProperty => {
                heap.no_gc(|nogc, heap| {
                    let name =
                        property_name(nogc, heap, cache.constants_ref(nogc), operands[1] as usize);
                    match stack
                        .reg_slot(&meta, operands[0] as i32)
                        .lookup(nogc, heap, name)
                    {
                        Lookup::Data { slot, holder, .. } => {
                            let ValueRef::Object(host) = holder else {
                                return Err(VmError::Type);
                            };
                            slot.set(heap, host.as_ref().erase(), Tagged::from_value(acc));
                            Ok(())
                        }
                        _ => unimplemented!("TODO: shape transitions (at least naive way)"),
                    }
                })?;
            }
            Opcode::CreateObjectFromMap => {
                let map = heap.no_gc(|nogc, heap| {
                    let v = cache.constants_ref(nogc).at(operands[0] as usize);
                    nogc.get_as::<Map>(v, heap.known().map_map)
                        .map(|r| r.into_tagged())
                        .ok_or(VmError::Type)
                })?;
                let count = operands[2] as usize;
                cache.spill_acc(acc);
                let args = stack.args(&meta, operands[1] as i32, count);
                let obj = heap.allocate_enter_nogc(
                    SlotsObjectInit { map, values: args },
                    |dst: HeapRef<'_, SlotsObject>, _nogc, _| dst.into_tagged().erase(),
                );
                let _ = cache.take_acc();
                acc = obj;
            }
            Opcode::CreateArrayLiteral => {
                let count = operands[1] as usize;
                cache.spill_acc(acc);
                let args = stack.args(&meta, operands[0] as i32, count);
                let array = heap.allocate_enter_nogc(args, |dst: HeapRef<'_, Array>, _nogc, _| {
                    dst.into_tagged().erase()
                });
                let _ = cache.take_acc();
                acc = array;
            }
            Opcode::LoadContextSlot => {
                let callable_value = stack.callable(&meta);
                let v = heap.no_gc(|nogc, heap| {
                    let code = code_ref(nogc, heap, callable_value);
                    let ValueRef::Object(context) = code.as_ref().context.value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    Ok(context.as_ref().slot(operands[0] as usize).inner())
                })?;
                acc = v;
            }
            Opcode::StoreContextSlot => {
                let callable_value = stack.callable(&meta);
                heap.no_gc(|nogc, heap| {
                    let code = code_ref(nogc, heap, callable_value);
                    let host = code.as_ref().context.inner();
                    let ValueRef::Object(context) = code.as_ref().context.value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    context.as_ref().slot(operands[0] as usize).set(
                        heap,
                        host,
                        Tagged::from_value(acc),
                    );
                    Ok(())
                })?;
            }
            Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
        }
    }
}

fn decode(code: &[u8], mut pc: usize) -> (Opcode, [u32; 4], usize) {
    let mut op = read_opcode(code, &mut pc);
    let mut scale = Scale::Byte1;
    if op == Opcode::Wide {
        scale = Scale::Byte2;
        op = read_opcode(code, &mut pc);
    }

    let mut operands = [0u32; 4];
    for (i, kind) in op.operands().iter().enumerate() {
        let size = kind.size_in_stream(scale);
        let bytes = code
            .get(pc..pc + size)
            .expect("truncated instruction stream");
        let mut buf = [0u8; 4];
        buf[..size].copy_from_slice(bytes);
        let raw = u32::from_le_bytes(buf);
        operands[i] = match (kind, size) {
            (Operand::Register | Operand::RegisterListStart | Operand::Immediate, 1) => {
                raw as u8 as i8 as u32
            }
            (Operand::Register | Operand::RegisterListStart | Operand::Immediate, 2) => {
                raw as u16 as i16 as u32
            }
            _ => raw,
        };
        pc += size;
    }
    (op, operands, pc)
}

fn read_opcode(code: &[u8], pc: &mut usize) -> Opcode {
    let byte = *code.get(*pc).expect("program counter out of bounds");
    *pc += 1;
    Opcode::from_byte(byte).expect("invalid opcode")
}
