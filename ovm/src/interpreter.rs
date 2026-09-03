use bytecode::{Opcode, decode};

use vm::{
    Context, FixedArray, Float, Handle, HeapRef, InternedString, LocalHeap, Lookup, Map, NoGc,
    Object, ObjectSlotsInit, SlotName, Smi, StoreOutcome, StoreSemantics, Symbol, Tagged, VMString,
    Value, ValueRef,
};

use crate::{
    ContextState, FrameMeta, Heap, NativeContext, NativeIndex, Stack, StackCache, VM, VmError,
    natives::EXCEPTION_SENTINEL,
};

pub fn execute<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: &[Value],
) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;

    let saved_top = stack.top();
    let was_active = cache.is_active();
    let was_acc_spilled = cache.is_acc_spilled();
    if was_active {
        stack.suspend_frame(cache.frame_meta());
        cache.reset_acc_spill();
    }
    let base_depth = stack.frame_depth();

    let result = start(vm, heap, state, callable, args, base_depth);

    stack.truncate_frames(base_depth);
    if was_active {
        let outer = stack.pop_frame(saved_top).expect("suspended caller frame");
        cache.load(stack, outer, heap);
        cache.restore_acc_spill(was_acc_spilled);
    } else {
        stack.set_top(saved_top);
        cache.deactivate();
    }
    result
}

fn start<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: &[Value],
    base_depth: usize,
) -> Result<Value, VmError> {
    match heap.no_gc(|nogc, heap| call_target(nogc, heap, callable.value())) {
        Some(CallTarget::Native(idx)) => {
            let f = vm.native(NativeIndex(idx));
            let void = heap.known().void.value();
            let mut nctx = NativeContext::new(vm, heap, state);
            let cache = &state.cache;
            cache.spill_acc(void);
            let result = f(&mut nctx, args);
            let _ = cache.take_acc();
            result
        }
        Some(CallTarget::Bytecode(target, register_count)) => {
            let stack = &state.stack;
            let frame = stack.push_initial_frame(target, register_count, args)?;
            state.cache.enter(stack, frame, heap);
            dispatch(vm, heap, state, base_depth)
        }
        None => Err(VmError::Type),
    }
}

enum CallTarget {
    Bytecode(Tagged<Object>, usize),
    Native(usize),
}

fn call_target<'a, L: LocalHeap>(nogc: &'a NoGc<'a>, heap: &'a L, f: Value) -> Option<CallTarget> {
    let ValueRef::Object(obj) = f.value_ref(nogc) else {
        return None;
    };
    let kind = obj.as_ref().header.map.heap_ref(nogc).kind();
    if !kind.is_callable() {
        return None;
    }
    if kind.is_native() {
        return Some(CallTarget::Native(obj.as_ref().native_index(nogc)?));
    }
    let info = obj.as_ref().callable_info(nogc, heap)?;
    let register_count = info.register_count.to_smi().value() as usize;
    Some(CallTarget::Bytecode(obj.into_tagged(), register_count))
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

fn classify_key<'a, L: LocalHeap>(
    nogc: &'a NoGc<'a>,
    heap: &'a L,
    key: Value,
) -> Result<Key, VmError> {
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

enum LoadOutcome {
    Value(Value),
    Getter(Value),
}

fn load_outcome<'a, L: LocalHeap>(
    nogc: &'a NoGc<'a>,
    heap: &'a L,
    receiver: Value,
    name: SlotName,
) -> LoadOutcome {
    match receiver.lookup(nogc, heap, name) {
        Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => LoadOutcome::Value(slot.inner()),
        Lookup::Accessor { pair, .. } => {
            let getter = pair.get.inner();
            // no getter (void sentinel): the load yields undefined
            if getter == heap.known().void.value() {
                LoadOutcome::Value(heap.known().undefined.value())
            } else {
                LoadOutcome::Getter(getter)
            }
        }
        Lookup::NotFound => LoadOutcome::Value(heap.known().undefined.value()),
    }
}

fn call_value<H: Heap>(
    heap: &mut H::Local,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    handler_pc: usize,
    f: Value,
    args: &[Value],
) -> Result<bool, VmError> {
    let target = heap.no_gc(|nogc, heap| call_target(nogc, heap, f));
    // TODO: native getters/setters invoke in place instead of pushing a frame
    let Some(CallTarget::Bytecode(target, register_count)) = target else {
        return Ok(false);
    };
    let callee = stack.push_frame_with_args(meta, handler_pc, target, register_count, args)?;
    cache.load(stack, callee, heap);
    Ok(true)
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

/// Materialize a VM error as an ECMAScript error object
pub fn error_from_vm_error<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    err: VmError,
) -> Result<Value, VmError> {
    state.handle_scope(|scope| {
        let name_string = vm.interner().intern(heap, &scope, "name");
        let message_string = vm.interner().intern(heap, &scope, "message");
        let name_value = vm.interner().intern(heap, &scope, err.name());
        let message_value = vm.interner().intern(heap, &scope, err.message());

        let map = scope
            .create_handle(heap.known().error_map.as_tagged())
            .expect("error map is strong");
        let obj = heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: heap.known().void.value(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let name = scope
            .create_handle(SlotName::from(name_string.as_tagged()).tagged())
            .expect("name is strong");
        let message = scope
            .create_handle(SlotName::from(message_string.as_tagged()).tagged())
            .expect("message is strong");
        let name_value = scope
            .create_handle(Tagged::from_value(name_value.value()))
            .expect("name value is strong");
        let message_value = scope
            .create_handle(Tagged::from_value(message_value.value()))
            .expect("message value is strong");
        Object::store_new_data_property(heap, obj, name, name_value)?;
        Object::store_new_data_property(heap, obj, message, message_value)?;
        Ok(obj.value())
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

/// ES ToBoolean. Falsey: `false`, `undefined`, `null`, the hole, 0, -0, NaN, everything else is truthy
fn is_truthy<'a, L: LocalHeap>(nogc: &'a NoGc<'a>, heap: &L, v: Value) -> bool {
    if let Some(smi) = Smi::decode(v) {
        return smi.value() != 0;
    }
    let known = heap.known();
    if v == known.false_object.value()
        || v == known.undefined.value()
        || v == known.null.value()
        || v == known.void.value()
    {
        return false;
    }
    if v == known.true_object.value() {
        return true;
    }
    if let Some(f) = v.get_as::<Float>(nogc, known.float_map) {
        let x = f.value.get();
        // -0.0 compares equal to 0.0; NaN compares unequal to everything
        return x != 0.0 && !x.is_nan();
    }
    if let Some(s) = v.get_as::<VMString>(nogc, known.string_map) {
        return s.len(nogc) != 0;
    }
    true
}

fn jump_target(pc: usize, offset: i32) -> usize {
    pc.wrapping_add_signed(offset as isize)
}

/// Result of an exception unwind.
enum Unwind {
    /// A handler was found: the accumulator must become the exception
    Caught(Value),
    /// No handler in this run: the exception escapes with the exception
    /// sentinel in the accumulator
    Escaped,
}

fn exception_dispatch<L: LocalHeap>(
    heap: &mut L,
    state: &ContextState,
    base_depth: usize,
    mut pc: usize,
) -> Unwind {
    let stack = &state.stack;
    let cache = &state.cache;
    loop {
        let handled = heap.no_gc(|nogc, heap| {
            let ValueRef::Object(obj) = stack
                .callable_slot(&cache.frame_meta())
                .inner()
                .value_ref(nogc)
            else {
                return None;
            };
            let info = obj.as_ref().callable_info(nogc, heap)?;
            info.handlers.heap_ref(nogc, heap)?.lookup(pc)
        });
        if let Some(handler_pc) = handled {
            let ex = state
                .take_pending_exception()
                .expect("pending exception must be set while unwinding");
            cache.set_pc(handler_pc);
            return Unwind::Caught(ex);
        }
        if stack.frame_depth() == base_depth {
            return Unwind::Escaped;
        }
        let base = cache.frame_meta().base;
        let caller = stack
            .pop_frame(base)
            .expect("suspended frame above base depth");
        cache.load(stack, caller, heap);
        pc = caller.handler_pc;
    }
}

fn raise<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    base_depth: usize,
    acc: Value,
    err: VmError,
    pc: usize,
) -> Unwind {
    let cache = &state.cache;
    cache.spill_acc(acc);
    let ex = error_from_vm_error(vm, heap, state, err)
        .expect("error materialization must not fail");
    let _ = cache.take_acc();
    state.set_pending_exception(ex);
    exception_dispatch(heap, state, base_depth, pc)
}

// TODO: pass stack and cache directly, could be benificial for threading dispatch later
// TODO: cleanup error handling here
fn dispatch<H: Heap>(
    vm: &VM<H>,
    heap: &mut H::Local,
    state: &ContextState,
    base_depth: usize,
) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;

    let mut acc = heap.known().undefined.value();
    loop {
        let pc = cache.pc();
        let (op, ops, next_pc) = heap.no_gc(|nogc, _| decode(cache.code_ref(nogc).as_slice(), pc));
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        match op {
            Opcode::Return => {
                // a nested run stops above the frame suspended on its entry
                if stack.frame_depth() == base_depth {
                    return Ok(acc);
                }
                let caller = stack
                    .pop_frame(meta.base)
                    .expect("suspended frame above base depth");
                cache.load(stack, caller, heap);
            }
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
                let Some(a) = Smi::decode(stack.reg(&meta, ops.reg(0))) else {
                    match raise(vm, heap, state, base_depth, acc, VmError::Type, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    }
                };
                let Some(b) = Smi::decode(stack.reg(&meta, ops.reg(1))) else {
                    match raise(vm, heap, state, base_depth, acc, VmError::Type, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    }
                };
                let Some(r) = a.value().checked_add(b.value()) else {
                    match raise(vm, heap, state, base_depth, acc, VmError::Overflow, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    }
                };
                if !Smi::in_range(r) {
                    match raise(vm, heap, state, base_depth, acc, VmError::Overflow, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    }
                }
                acc = Smi::new(r).encode();
            }
            Opcode::Jump => {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Opcode::JumpLoop => {
                cache.spill_acc(acc);
                heap.safepoint_poll();
                acc = cache.take_acc();
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Opcode::JumpIfTruthy => {
                if heap.no_gc(|nogc, heap| is_truthy(nogc, heap, acc)) {
                    cache.set_pc(jump_target(pc, ops.imm(0)));
                }
            }
            Opcode::JumpIfFalsy => {
                if !heap.no_gc(|nogc, heap| is_truthy(nogc, heap, acc)) {
                    cache.set_pc(jump_target(pc, ops.imm(0)));
                }
            }
            Opcode::TestReferenceEqual => {
                let other = stack.reg(&meta, ops.reg(0));
                let known = heap.known();
                acc = if other == acc {
                    known.true_object.value()
                } else {
                    known.false_object.value()
                };
            }
            Opcode::Throw | Opcode::ReThrow => {
                state.set_pending_exception(acc);
                match exception_dispatch(heap, state, base_depth, pc) {
                    Unwind::Caught(ex) => {
                        acc = ex;
                        continue;
                    }
                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                }
            }
            Opcode::CallNative => {
                let f = vm.native(NativeIndex(ops.idx(0)));
                let count = ops.reg_count(2);
                let mut nctx = NativeContext::new(vm, heap, state);
                cache.spill_acc(acc);
                let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                let saved = cache.take_acc();
                match result {
                    Ok(v) if v == EXCEPTION_SENTINEL => {
                        match exception_dispatch(heap, state, base_depth, pc) {
                            Unwind::Caught(ex) => {
                                acc = ex;
                                continue;
                            }
                            Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                        }
                    }
                    Ok(v) => acc = v,
                    Err(err) => {
                        match raise(vm, heap, state, base_depth, saved, err, pc) {
                            Unwind::Caught(ex) => {
                                acc = ex;
                                continue;
                            }
                            Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                        }
                    }
                }
            }
            // TODO: feedback vectors and separation once they are there
            Opcode::Call | Opcode::CallNoFeedback => {
                let count = ops.reg_count(2);
                let target =
                    heap.no_gc(|nogc, heap| call_target(nogc, heap, stack.reg(&meta, ops.reg(0))));
                let target = match target {
                    Some(target) => target,
                    None => match raise(vm, heap, state, base_depth, acc, VmError::Type, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    },
                };
                match target {
                    CallTarget::Native(idx) => {
                        let f = vm.native(NativeIndex(idx));
                        let mut nctx = NativeContext::new(vm, heap, state);
                        cache.spill_acc(acc);
                        let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                        let saved = cache.take_acc();
                        match result {
                            Ok(v) if v == EXCEPTION_SENTINEL => {
                                match exception_dispatch(heap, state, base_depth, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                            Ok(v) => acc = v,
                            Err(err) => {
                                match raise(vm, heap, state, base_depth, saved, err, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                        }
                    }
                    CallTarget::Bytecode(target, register_count) => {
                        let callee = stack
                            .push_frame(meta, pc, target, register_count, ops.reg_list(1), count);
                        let callee = match callee {
                            Ok(callee) => callee,
                            Err(err) => {
                                match raise(vm, heap, state, base_depth, acc, err, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                        };
                        cache.load(stack, callee, heap);
                    }
                }
            }
            Opcode::LoadNamedProperty => {
                let outcome = heap.no_gc(|nogc, heap| {
                    let name = property_name(nogc, heap, cache.constants_ref(nogc), ops.idx(1));
                    load_outcome(nogc, heap, stack.reg(&meta, ops.reg(0)), name)
                });
                match outcome {
                    LoadOutcome::Value(v) => acc = v,
                    LoadOutcome::Getter(getter) => {
                        let receiver = stack.reg(&meta, ops.reg(0));
                        let called =
                            call_value::<H>(heap, stack, cache, meta, pc, getter, &[receiver]);
                        let called = match called {
                            Ok(called) => called,
                            Err(err) => {
                                match raise(vm, heap, state, base_depth, acc, err, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                        };
                        if !called {
                            // non-callable getter: the load yields undefined
                            acc = heap.known().undefined.value();
                        }
                    }
                }
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
                });
                let outcome = match outcome {
                    Ok(outcome) => outcome,
                    Err(err) => match raise(vm, heap, state, base_depth, acc, err, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    },
                };
                match outcome {
                    StoreOutcome::Transition { receiver, name } => {
                        cache.spill_acc(acc);
                        let result = store_transition(heap, state, receiver, name, acc);
                        let _ = cache.take_acc();
                        if let Err(err) = result {
                            match raise(vm, heap, state, base_depth, acc, err, pc) {
                                Unwind::Caught(ex) => {
                                    acc = ex;
                                    continue;
                                }
                                Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                            }
                        }
                    }
                    StoreOutcome::CallSetter { setter } => {
                        let receiver = stack.reg(&meta, ops.reg(0));
                        let called =
                            call_value::<H>(heap, stack, cache, meta, pc, setter, &[receiver, acc]);
                        match called {
                            Ok(_) => {}
                            Err(err) => {
                                match raise(vm, heap, state, base_depth, acc, err, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                        }
                    }
                    StoreOutcome::Done => {}
                }
            }
            Opcode::LoadKeyedProperty => {
                let outcome = heap.no_gc(|nogc, heap| {
                    let receiver = stack.reg(&meta, ops.reg(0));
                    match classify_key(nogc, heap, acc)? {
                        Key::Element(i) => {
                            let arr = element_array(nogc, heap, receiver, i)?;
                            Ok(LoadOutcome::Value(arr.at(i)))
                        }
                        Key::Name(name) => Ok(load_outcome(nogc, heap, receiver, name)),
                    }
                });
                let outcome = match outcome {
                    Ok(outcome) => outcome,
                    Err(err) => match raise(vm, heap, state, base_depth, acc, err, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    },
                };
                match outcome {
                    LoadOutcome::Value(v) => acc = v,
                    LoadOutcome::Getter(getter) => {
                        let receiver = stack.reg(&meta, ops.reg(0));
                        let called =
                            call_value::<H>(heap, stack, cache, meta, pc, getter, &[receiver]);
                        let called = match called {
                            Ok(called) => called,
                            Err(err) => {
                                match raise(vm, heap, state, base_depth, acc, err, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                        };
                        if !called {
                            acc = heap.known().undefined.value();
                        }
                    }
                }
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
                        Key::Name(name) => receiver.store_lookup(nogc, heap, name, acc, semantics),
                    }
                });
                let outcome = match outcome {
                    Ok(outcome) => outcome,
                    Err(err) => match raise(vm, heap, state, base_depth, acc, err, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    },
                };
                match outcome {
                    StoreOutcome::Transition { receiver, name } => {
                        cache.spill_acc(acc);
                        let result = store_transition(heap, state, receiver, name, acc);
                        let _ = cache.take_acc();
                        if let Err(err) = result {
                            match raise(vm, heap, state, base_depth, acc, err, pc) {
                                Unwind::Caught(ex) => {
                                    acc = ex;
                                    continue;
                                }
                                Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                            }
                        }
                    }
                    StoreOutcome::CallSetter { setter } => {
                        let receiver = stack.reg(&meta, ops.reg(0));
                        let called =
                            call_value::<H>(heap, stack, cache, meta, pc, setter, &[receiver, acc]);
                        match called {
                            Ok(_) => {}
                            Err(err) => {
                                match raise(vm, heap, state, base_depth, acc, err, pc) {
                                    Unwind::Caught(ex) => {
                                        acc = ex;
                                        continue;
                                    }
                                    Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                                }
                            }
                        }
                    }
                    StoreOutcome::Done => {}
                }
            }
            Opcode::CreateObjectFromMap => {
                let map = heap.no_gc(|nogc, heap| {
                    let v = cache.constants_ref(nogc).at(ops.idx(0));
                    v.get_as::<Map>(nogc, heap.known().map_map)
                        .map(|r| r.into_tagged())
                        .ok_or(VmError::Type)
                });
                let map = match map {
                    Ok(map) => map,
                    Err(err) => match raise(vm, heap, state, base_depth, acc, err, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    },
                };
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
                    let context = info
                        .context
                        .inner()
                        .get_as::<Context>(nogc, heap.known().context_map)
                        .ok_or(VmError::Type)?;
                    Ok(context
                        .slots
                        .heap_ref(nogc)
                        .as_ref()
                        .element_slot(ops.idx(0))
                        .inner())
                });
                let v = match v {
                    Ok(v) => v,
                    Err(err) => match raise(vm, heap, state, base_depth, acc, err, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    },
                };
                acc = v;
            }
            Opcode::StoreContextSlot => {
                let result = heap.no_gc(|nogc, heap| {
                    let ValueRef::Object(obj) = stack.callable_slot(&meta).inner().value_ref(nogc)
                    else {
                        return Err(VmError::Type);
                    };
                    let info = obj
                        .as_ref()
                        .callable_info(nogc, heap)
                        .ok_or(VmError::Type)?;
                    let host = info.context.inner();
                    let context = info
                        .context
                        .inner()
                        .get_as::<Context>(nogc, heap.known().context_map)
                        .ok_or(VmError::Type)?;
                    context
                        .slots
                        .heap_ref(nogc)
                        .element_slot(ops.idx(0))
                        .set(heap, host, Tagged::from_value(acc));
                    Ok(())
                });
                if let Err(err) = result {
                    match raise(vm, heap, state, base_depth, acc, err, pc) {
                        Unwind::Caught(ex) => {
                            acc = ex;
                            continue;
                        }
                        Unwind::Escaped => return Ok(EXCEPTION_SENTINEL),
                    }
                }
            }
            Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
        }
    }
}
