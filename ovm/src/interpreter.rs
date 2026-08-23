use bytecode::{Opcode, Operand, Scale};

use vm::{
    FixedArray, Handle, HeapObject, HeapRef, InternedString, LocalHeap, Lookup, Map, NoGc, Object,
    ObjectSlotsInit, SlotName, Smi, Tagged, Value, ValueRef,
};

use crate::{ContextState, Heap, NativeContext, NativeIndex, VM, VmError};

pub fn run<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: &[Value],
) -> Result<Value, VmError> {
    // TODO: merge getting the register count into the frame allocation function (I think)
    let register_count = heap
        .no_gc(|nogc, heap| {
            callable
                .heap_ref(nogc)
                .as_ref()
                .callable_info(nogc, heap)
                .map(|info| info.register_count.to_smi().value() as usize)
        })
        .ok_or(VmError::Type)?;

    let stack = &state.stack;
    let cache = &state.cache;

    // TODO: consider having a frame scope?
    let result = stack
        .push_initial_frame(callable.as_tagged(), register_count, args)
        .and_then(|frame| {
            cache.enter(stack, frame, heap);
            let result = dispatch(vm, heap, state);
            cache.deactivate();
            result
        });

    stack.set_top(0);
    stack.clear_frames();

    result
}

fn property_name<'a, L: LocalHeap>(
    nogc: &'a NoGc<'a>,
    heap: &'a L,
    constants: HeapRef<'a, FixedArray>,
    idx: usize,
) -> SlotName {
    // TODO: this function must handle also non constants and non interned strings and symbols
    let v = constants.at(idx);
    let name = v
        .get_as::<InternedString>(nogc, heap.known().string_map)
        .expect("property name constant must be an interned string");
    SlotName::from(name.into_tagged())
}

// TODO: pass stack and cache directly, could be benificial for threading dispatch later
fn dispatch<H: Heap>(vm: &VM<H>, heap: &mut H::Local, state: &ContextState) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;
    let mut acc = heap.known().void.value();
    loop {
        let (op, operands, next_pc) =
            heap.no_gc(|nogc, _| decode(cache.code_ref(nogc).as_slice(), cache.pc()));
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        // TODO: nicer api for the operator casts
        // TODO: actually handle `Result` instead of just `?`
        match op {
            // TODO0: why return from value within cache/frame, shouldn't it always be in acc? 
            Opcode::Return => match stack.pop_frame(meta.base) {
                Some(caller) => cache.load(stack, caller, heap),
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
                // TODO: JS semantics
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
            // TODO: feedback vectors and separation once they are there
            Opcode::Call | Opcode::CallNoFeedback => {
                let count = operands[2] as usize;
                let target = heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) =
                        stack.reg_slot(&meta, operands[0] as i32).value_ref(nogc)
                    else {
                        return None;
                    };
                    let info = obj.as_ref().callable_info(nogc, heap)?;
                    let register_count = info.register_count.to_smi().value() as usize;
                    Some((obj.into_tagged(), register_count))
                });
                let Some((target, register_count)) = target else {
                    return Err(VmError::Type);
                };
                let callee = stack.push_frame(
                    meta,
                    target,
                    register_count,
                    operands[1] as i32,
                    count,
                )?;
                cache.load(stack, callee, heap);
            }
            // TODO: property lookup shoudln't solely depend on constant pool
            // runtime values and symbols must also be supported
            // probably separate bytecode of split load and lookup
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
                    v.get_as::<Map>(nogc, heap.known().map_map)
                        .map(|r| r.into_tagged())
                        .ok_or(VmError::Type)
                })?;
                let count = operands[2] as usize;
                cache.spill_acc(acc);
                let args = stack.args(&meta, operands[1] as i32, count);

                // TODO: maybe have a handlescope always accessible or a quickspill cache
                let obj = state.handle_scope(|scope| {
                    let map = scope
                        .create_handle(map)
                        .expect("map is a strong pointer");
                    heap.allocate_object(
                        &scope,
                        ObjectSlotsInit {
                            map,
                            values: args,
                            elements: heap.known().void.value(),
                            length: 0,
                        },
                    )
                });
                let _ = cache.take_acc();
                acc = obj.erase();
            }
            // TODO: create more array creation operations, this one is only for &[Value]
            Opcode::CreateArrayLiteral => {
                let count = operands[1] as usize;
                cache.spill_acc(acc);
                let args = stack.args(&meta, operands[0] as i32, count);
                let array = heap
                    .allocate_enter_nogc(args, |dst: HeapRef<'_, FixedArray>, _nogc, _| {
                        dst.into_tagged().erase()
                    });
                let _ = cache.take_acc();
                acc = array;
            }
            Opcode::LoadContextSlot => {
                let v = heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) = stack.callable_slot(&meta).value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    let info = obj
                        .as_ref()
                        .callable_info(nogc, heap)
                        .ok_or(VmError::Type)?;
                    let ValueRef::Object(context) = info.context.value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    Ok(context
                        .as_ref()
                        .slots
                        .heap_ref(nogc)
                        .as_ref()
                        .element_slot(operands[0] as usize)
                        .inner())
                })?;
                acc = v;
            }
            Opcode::StoreContextSlot => {
                heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) = stack.callable_slot(&meta).value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    let info = obj
                        .as_ref()
                        .callable_info(nogc, heap)
                        .ok_or(VmError::Type)?;
                    let host = info.context.inner();
                    let ValueRef::Object(context) = info.context.value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    context
                        .as_ref()
                        .slots
                        .heap_ref(nogc)
                        .element_slot(operands[0] as usize)
                        .set(heap, host, Tagged::from_value(acc));
                    Ok(())
                })?;
            }
            Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
        }
    }
}

// TODO: this decode function looks quite complex, I believe this can be simplified
// TODO: get rid of this much branching and .expect(), use `debug_assert!` instead
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

// TODO: see if force inlining matters
fn read_opcode(code: &[u8], pc: &mut usize) -> Opcode {
    let byte = *code.get(*pc).expect("program counter out of bounds");
    *pc += 1;
    Opcode::from_byte(byte).expect("invalid opcode")
}
