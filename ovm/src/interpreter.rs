use bytecode::{Opcode, decode};

use vm::{
    FixedArray, Handle, HeapRef, InternedString, LocalHeap, Lookup, Map, NoGc, Object,
    ObjectSlotsInit, SlotName, Smi, StoreOutcome, StoreSemantics, Symbol, Tagged, Value, ValueRef,
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

/// A runtime property key: a smi element index or a name (interned string / symbol).
enum Key {
    Element(usize),
    Name(SlotName),
}

fn classify_key<'a, L: LocalHeap>(nogc: &'a NoGc<'a>, heap: &'a L, key: Value) -> Result<Key, VmError> {
    if let Some(smi) = Smi::decode(key) {
        return usize::try_from(smi.value())
            .map(Key::Element)
            .map_err(|_| VmError::OutOfBounds);
    }
    if let Some(s) = key.get_as::<InternedString>(nogc, heap.known().string_map) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    if let Some(s) = key.get_as::<Symbol>(nogc, heap.known().symbol_map) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    Err(VmError::Type)
}

fn store_transition<L: LocalHeap>(
    heap: &mut L,
    state: &ContextState,
    receiver: Value,
    name: SlotName,
    value: Value,
) -> Result<(), VmError> {
    state.handle_scope(|scope| {
        let receiver = scope
            .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(receiver) })
            .expect("receiver must be strong");
        let name = scope
            .create_handle(name.tagged())
            .expect("name must be strong");
        let value = scope
            .create_handle(Tagged::from_value(value))
            .expect("value must be strong");
        Object::store_new_data_property(heap, receiver, name, value)
    })
}

fn element_array<'a, L: LocalHeap>(
    nogc: &'a NoGc<'a>,
    heap: &'a L,
    receiver: Value,
    i: usize,
) -> Result<HeapRef<'a, FixedArray>, VmError> {
    let arr = receiver
        .get_as::<FixedArray>(nogc, heap.known().array_map)
        .ok_or(VmError::Type)?;
    if i >= arr.len() {
        return Err(VmError::OutOfBounds);
    }
    Ok(arr)
}

// TODO: pass stack and cache directly, could be benificial for threading dispatch later
fn dispatch<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;
    let mut acc = heap.known().void.value();
    loop {
        let (op, ops, next_pc) =
            heap.no_gc(|nogc, _| decode(cache.code_ref(nogc).as_slice(), cache.pc()));
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        // TODO: actually handle `Result` instead of just `?`
        match op {
            Opcode::Return => match stack.pop_frame(meta.base) {
                Some(caller) => cache.load(stack, caller, heap),
                None => return Ok(acc),
            },
            Opcode::Load => {
                acc = stack.reg(&meta, ops.reg(0));
            }
            Opcode::Store => {
                stack.set_reg(&meta, ops.reg(0), acc);
            }
            Opcode::Move => {
                let v = stack.reg(&meta, ops.reg(1));
                stack.set_reg(&meta, ops.reg(0), v);
            }
            Opcode::LoadSmi => {
                acc = Smi::new(ops.imm(0) as i64).encode();
            }
            Opcode::LoadConstant => {
                let v = heap.no_gc(|nogc, _| cache.constants_ref(nogc).at(ops.idx(0)));
                acc = v;
            }
            Opcode::Add => {
                // TODO: JS semantics
                let a = Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)?;
                let b = Smi::decode(stack.reg(&meta, ops.reg(1))).ok_or(VmError::Type)?;
                let r = a.value().checked_add(b.value()).ok_or(VmError::Overflow)?;
                if !Smi::in_range(r) {
                    return Err(VmError::Overflow);
                }
                acc = Smi::new(r).encode();
            }
            Opcode::CallNative => {
                let f = vm.native(NativeIndex(ops.idx(0)));
                let count = ops.reg_count(2);
                let mut nctx = NativeContext::new(vm, heap, state);
                cache.spill_acc(acc);
                let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                let _ = cache.take_acc();
                acc = result?;
            }
            // TODO: feedback vectors and separation once they are there
            Opcode::Call | Opcode::CallNoFeedback => {
                let count = ops.reg_count(2);
                let target = heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) = stack.reg(&meta, ops.reg(0)).value_ref(nogc) else {
                        return None;
                    };
                    let info = obj.as_ref().callable_info(nogc, heap)?;
                    let register_count = info.register_count.to_smi().value() as usize;
                    Some((obj.into_tagged(), register_count))
                });
                let Some((target, register_count)) = target else {
                    return Err(VmError::Type);
                };
                let callee =
                    stack.push_frame(meta, target, register_count, ops.reg_list(1), count)?;
                cache.load(stack, callee, heap);
            }
            Opcode::LoadNamedProperty => {
                let v = heap.no_gc(|nogc, heap| {
                    let name = property_name(nogc, heap, cache.constants_ref(nogc), ops.idx(1));
                    match stack.reg(&meta, ops.reg(0)).lookup(nogc, heap, name) {
                        Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => slot.inner(),
                        _ => unimplemented!("TODO: implement accessors"),
                    }
                });
                acc = v;
            }
            Opcode::StoreNamedProperty | Opcode::StoreNamedPropertyShadow => {
                let semantics = match op {
                    Opcode::StoreNamedPropertyShadow => StoreSemantics::Shadow,
                    _ => StoreSemantics::WriteThrough,
                };
                let outcome = heap.no_gc(|nogc, heap| {
                    let name = property_name(nogc, heap, cache.constants_ref(nogc), ops.idx(1));
                    stack
                        .reg(&meta, ops.reg(0))
                        .store_lookup(nogc, heap, name, acc, semantics)
                })?;
                if let StoreOutcome::Transition { receiver, name } = outcome {
                    cache.spill_acc(acc);
                    let result = store_transition(heap, state, receiver, name, acc);
                    let _ = cache.take_acc();
                    result?;
                }
            }
            Opcode::LoadKeyedProperty => {
                let v = heap.no_gc(|nogc, heap| {
                    let receiver = stack.reg(&meta, ops.reg(0));
                    match classify_key(nogc, heap, acc)? {
                        Key::Element(i) => {
                            let arr = element_array(nogc, heap, receiver, i)?;
                            Ok(arr.at(i))
                        }
                        Key::Name(name) => match receiver.lookup(nogc, heap, name) {
                            Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => {
                                Ok(slot.inner())
                            }
                            _ => unimplemented!("TODO: implement accessors"),
                        },
                    }
                })?;
                acc = v;
            }
            Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyShadow => {
                let semantics = match op {
                    Opcode::StoreKeyedPropertyShadow => StoreSemantics::Shadow,
                    _ => StoreSemantics::WriteThrough,
                };
                let outcome = heap.no_gc(|nogc, heap| {
                    let receiver = stack.reg(&meta, ops.reg(0));
                    let key = stack.reg(&meta, ops.reg(1));
                    match classify_key(nogc, heap, key)? {
                        Key::Element(i) => {
                            let arr = element_array(nogc, heap, receiver, i)?;
                            arr.set(heap, i, acc);
                            Ok(StoreOutcome::Done)
                        }
                        Key::Name(name) => {
                            receiver.store_lookup(nogc, heap, name, acc, semantics)
                        }
                    }
                })?;
                if let StoreOutcome::Transition { receiver, name } = outcome {
                    cache.spill_acc(acc);
                    let result = store_transition(heap, state, receiver, name, acc);
                    let _ = cache.take_acc();
                    result?;
                }
            }
            Opcode::CreateObjectFromMap => {
                let map = heap.no_gc(|nogc, heap| {
                    let v = cache.constants_ref(nogc).at(ops.idx(0));
                    v.get_as::<Map>(nogc, heap.known().map_map)
                        .map(|r| r.into_tagged())
                        .ok_or(VmError::Type)
                })?;
                let count = ops.reg_count(2);
                cache.spill_acc(acc);
                let args = stack.args(&meta, ops.reg_list(1), count);

                // TODO: maybe have a handlescope always accessible or a quickspill cache
                let obj = state.handle_scope(|scope| {
                    let map = scope.create_handle(map).expect("map is a strong pointer");
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
                let count = ops.reg_count(1);
                cache.spill_acc(acc);
                let args = stack.args(&meta, ops.reg_list(0), count);
                let array = heap
                    .allocate_enter_nogc(args, |dst: HeapRef<'_, FixedArray>, _nogc, _| {
                        dst.into_tagged().erase()
                    });
                let _ = cache.take_acc();
                acc = array;
            }
            Opcode::LoadContextSlot => {
                let v = heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) = stack.callable_slot(&meta).inner().value_ref(nogc)
                    else {
                        return Err(VmError::Type);
                    };
                    let info = obj
                        .as_ref()
                        .callable_info(nogc, heap)
                        .ok_or(VmError::Type)?;
                    let ValueRef::Object(context) = info.context.inner().value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    Ok(context
                        .as_ref()
                        .slots
                        .heap_ref(nogc)
                        .as_ref()
                        .element_slot(ops.idx(0))
                        .inner())
                })?;
                acc = v;
            }
            Opcode::StoreContextSlot => {
                heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) = stack.callable_slot(&meta).inner().value_ref(nogc)
                    else {
                        return Err(VmError::Type);
                    };
                    let info = obj
                        .as_ref()
                        .callable_info(nogc, heap)
                        .ok_or(VmError::Type)?;
                    let host = info.context.inner();
                    let ValueRef::Object(context) = info.context.inner().value_ref(nogc) else {
                        return Err(VmError::Type);
                    };
                    context
                        .as_ref()
                        .slots
                        .heap_ref(nogc)
                        .element_slot(ops.idx(0))
                        .set(heap, host, Tagged::from_value(acc));
                    Ok(())
                })?;
            }
            Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
        }
    }
}
