use bytecode::{Opcode, Operands, decode};

use vm::{
    Context, FixedArray, Float, Handle, Heap, HeapRef, InternedString, Lookup, Map, NoGc, Object,
    ObjectSlotsInit, SlotName, Smi, StoreOutcome, StoreSemantics, Symbol, Tagged, VMString, Value,
    ValueRef,
};

use crate::{ContextState, FrameMeta, NativeContext, NativeIndex, Stack, StackCache, VM, VmError};

pub fn execute(
    vm: &VM,
    heap: &mut Heap,
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

fn start(
    vm: &VM,
    heap: &mut Heap,
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

fn call_target<'a>(nogc: &'a NoGc<'a>, heap: &'a Heap, f: Value) -> Option<CallTarget> {
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

fn property_name<'a>(
    nogc: &'a NoGc<'a>,
    heap: &'a Heap,
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

fn classify_key<'a>(nogc: &'a NoGc<'a>, heap: &'a Heap, key: Value) -> Result<Key, VmError> {
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

fn load_outcome<'a>(
    nogc: &'a NoGc<'a>,
    heap: &'a Heap,
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

fn call_value(
    heap: &mut Heap,
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

fn store_transition(
    heap: &mut Heap,
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
#[cold]
#[inline(never)]
pub fn error_from_vm_error(
    vm: &VM,
    heap: &mut Heap,
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
                    elements: heap.known().empty_fixed_array.erase(),
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

fn element_array<'a>(
    nogc: &'a NoGc<'a>,
    heap: &'a Heap,
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
fn is_truthy<'a>(nogc: &'a NoGc<'a>, heap: &Heap, v: Value) -> bool {
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

#[cold]
#[inline(never)]
fn exception_dispatch(
    heap: &mut Heap,
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

#[cold]
#[inline(never)]
fn raise(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    base_depth: usize,
    acc: Value,
    err: VmError,
    pc: usize,
) -> Unwind {
    let cache = &state.cache;
    cache.spill_acc(acc);
    let ex =
        error_from_vm_error(vm, heap, state, err).expect("error materialization must not fail");
    let _ = cache.take_acc();
    state.set_pending_exception(ex);
    exception_dispatch(heap, state, base_depth, pc)
}

// TODO: pass stack and cache directly, could be benificial for threading dispatch later
fn dispatch(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    base_depth: usize,
) -> Result<Value, VmError> {
    let cache = &state.cache;
    let mut acc = heap.known().undefined.value();
    loop {
        let pc = cache.pc();
        let (op, ops, next_pc) = heap.no_gc(|nogc, _| decode(cache.code_ref(nogc).as_slice(), pc));
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        let result = step(vm, heap, state, base_depth, &mut acc, op, ops, meta, pc);
        match result {
            Step::Next => {}
            Step::Return(v) => return Ok(v),
            Step::Throw(v) => {
                state.set_pending_exception(v);
                match exception_dispatch(heap, state, base_depth, pc) {
                    Unwind::Caught(ex) => acc = ex,
                    Unwind::Escaped => return Ok(heap.known().exception.value()),
                }
            }
            Step::PendingThrow => match exception_dispatch(heap, state, base_depth, pc) {
                Unwind::Caught(ex) => acc = ex,
                Unwind::Escaped => return Ok(heap.known().exception.value()),
            },
            Step::Error(err) => match raise(vm, heap, state, base_depth, acc, err, pc) {
                Unwind::Caught(ex) => acc = ex,
                Unwind::Escaped => return Ok(heap.known().exception.value()),
            },
        }
    }
}

enum Step {
    Next,
    Return(Value),
    Throw(Value),
    PendingThrow,
    Error(VmError),
}

macro_rules! step_try {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => return Step::Error(err),
        }
    };
}

fn apply_store_outcome(
    heap: &mut Heap,
    state: &ContextState,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    pc: usize,
    receiver: Value,
    acc: Value,
    outcome: StoreOutcome,
) -> Result<(), VmError> {
    match outcome {
        StoreOutcome::Transition { receiver, name } => {
            cache.spill_acc(acc);
            let result = store_transition(heap, state, receiver, name, acc);
            let _ = cache.take_acc();
            result
        }
        StoreOutcome::CallSetter { setter } => {
            call_value(heap, stack, cache, meta, pc, setter, &[receiver, acc])?;
            Ok(())
        }
        StoreOutcome::Done => Ok(()),
    }
}

fn step(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    base_depth: usize,
    acc: &mut Value,
    op: Opcode,
    ops: Operands,
    meta: FrameMeta,
    pc: usize,
) -> Step {
    let stack = &state.stack;
    let cache = &state.cache;

    match op {
        Opcode::Return => {
            // a nested run stops above the frame suspended on its entry
            if stack.frame_depth() == base_depth {
                return Step::Return(*acc);
            }
            let caller = stack
                .pop_frame(meta.base)
                .expect("suspended frame above base depth");
            cache.load(stack, caller, heap);
            Step::Next
        }
        Opcode::Load => {
            *acc = stack.reg(&meta, ops.reg(0));
            Step::Next
        }
        Opcode::Store => {
            stack.set_reg(&meta, ops.reg(0), *acc);
            Step::Next
        }
        Opcode::Move => {
            let v = stack.reg(&meta, ops.reg(1));
            stack.set_reg(&meta, ops.reg(0), v);
            Step::Next
        }
        Opcode::LoadSmi => {
            *acc = Smi::new(ops.imm(0) as i64).encode();
            Step::Next
        }
        Opcode::LoadConstant => {
            let v = heap.no_gc(|nogc, _| cache.constants_ref(nogc).at(ops.idx(0)));
            *acc = v;
            Step::Next
        }
        Opcode::Add => {
            // TODO: JS semantics
            let a = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type));
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(1))).ok_or(VmError::Type));
            let r = step_try!(a.value().checked_add(b.value()).ok_or(VmError::Overflow));
            if !Smi::in_range(r) {
                return Step::Error(VmError::Overflow);
            }
            *acc = Smi::new(r).encode();
            Step::Next
        }
        Opcode::Jump => {
            cache.set_pc(jump_target(pc, ops.imm(0)));
            Step::Next
        }
        Opcode::JumpLoop => {
            cache.spill_acc(*acc);
            heap.safepoint_poll();
            *acc = cache.take_acc();
            cache.set_pc(jump_target(pc, ops.imm(0)));
            Step::Next
        }
        Opcode::JumpIfTruthy => {
            if heap.no_gc(|nogc, heap| is_truthy(nogc, heap, *acc)) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfFalsy => {
            if !heap.no_gc(|nogc, heap| is_truthy(nogc, heap, *acc)) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::TestReferenceEqual => {
            let other = stack.reg(&meta, ops.reg(0));
            let known = heap.known();
            *acc = if other == *acc {
                known.true_object.value()
            } else {
                known.false_object.value()
            };
            Step::Next
        }
        Opcode::Throw | Opcode::ReThrow => Step::Throw(*acc),
        Opcode::CallNative => {
            let f = vm.native(NativeIndex(ops.idx(0)));
            let count = ops.reg_count(2);
            let mut nctx = NativeContext::new(vm, heap, state);
            cache.spill_acc(*acc);
            let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
            let _ = cache.take_acc();
            match result {
                Ok(v) if v == heap.known().exception.value() => Step::PendingThrow,
                Ok(v) => {
                    *acc = v;
                    Step::Next
                }
                Err(err) => Step::Error(err),
            }
        }
        // TODO: feedback vectors and separation once they are there
        Opcode::Call | Opcode::CallNoFeedback => {
            let count = ops.reg_count(2);
            let target =
                heap.no_gc(|nogc, heap| call_target(nogc, heap, stack.reg(&meta, ops.reg(0))));
            let Some(target) = target else {
                return Step::Error(VmError::Type);
            };
            match target {
                CallTarget::Native(idx) => {
                    let f = vm.native(NativeIndex(idx));
                    let mut nctx = NativeContext::new(vm, heap, state);
                    cache.spill_acc(*acc);
                    let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                    let _ = cache.take_acc();
                    match result {
                        Ok(v) if v == heap.known().exception.value() => Step::PendingThrow,
                        Ok(v) => {
                            *acc = v;
                            Step::Next
                        }
                        Err(err) => Step::Error(err),
                    }
                }
                CallTarget::Bytecode(target, register_count) => {
                    let callee = step_try!(stack.push_frame(
                        meta,
                        pc,
                        target,
                        register_count,
                        ops.reg_list(1),
                        count,
                    ));
                    cache.load(stack, callee, heap);
                    Step::Next
                }
            }
        }
        Opcode::LoadNamedProperty => {
            let outcome = heap.no_gc(|nogc, heap| {
                let name = property_name(nogc, heap, cache.constants_ref(nogc), ops.idx(1));
                load_outcome(nogc, heap, stack.reg(&meta, ops.reg(0)), name)
            });
            match outcome {
                LoadOutcome::Value(v) => *acc = v,
                LoadOutcome::Getter(getter) => {
                    let receiver = stack.reg(&meta, ops.reg(0));
                    let called =
                        step_try!(call_value(heap, stack, cache, meta, pc, getter, &[receiver]));
                    if !called {
                        // non-callable getter: the load yields undefined
                        *acc = heap.known().undefined.value();
                    }
                }
            }
            Step::Next
        }
        Opcode::StoreNamedProperty | Opcode::StoreNamedPropertyShadow => {
            let semantics = match op {
                Opcode::StoreNamedPropertyShadow => StoreSemantics::Shadow,
                _ => StoreSemantics::WriteThrough,
            };
            let receiver = stack.reg(&meta, ops.reg(0));
            let outcome = step_try!(heap.no_gc(|nogc, heap| {
                let name = property_name(nogc, heap, cache.constants_ref(nogc), ops.idx(1));
                receiver.store_lookup(nogc, heap, name, *acc, semantics)
            }));
            step_try!(apply_store_outcome(
                heap, state, stack, cache, meta, pc, receiver, *acc, outcome,
            ));
            Step::Next
        }
        Opcode::LoadKeyedProperty => {
            let receiver = stack.reg(&meta, ops.reg(0));
            let outcome = step_try!(heap.no_gc(|nogc, heap| {
                match classify_key(nogc, heap, *acc)? {
                    Key::Element(i) => {
                        let arr = element_array(nogc, heap, receiver, i)?;
                        Ok(LoadOutcome::Value(arr.at(i)))
                    }
                    Key::Name(name) => Ok(load_outcome(nogc, heap, receiver, name)),
                }
            }));
            match outcome {
                LoadOutcome::Value(v) => *acc = v,
                LoadOutcome::Getter(getter) => {
                    let called =
                        step_try!(call_value(heap, stack, cache, meta, pc, getter, &[receiver]));
                    if !called {
                        *acc = heap.known().undefined.value();
                    }
                }
            }
            Step::Next
        }
        Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyShadow => {
            let semantics = match op {
                Opcode::StoreKeyedPropertyShadow => StoreSemantics::Shadow,
                _ => StoreSemantics::WriteThrough,
            };
            let receiver = stack.reg(&meta, ops.reg(0));
            let key = stack.reg(&meta, ops.reg(1));
            let outcome = step_try!(heap.no_gc(|nogc, heap| {
                match classify_key(nogc, heap, key)? {
                    Key::Element(i) => {
                        let arr = element_array(nogc, heap, receiver, i)?;
                        arr.set(heap, i, *acc);
                        Ok(StoreOutcome::Done)
                    }
                    Key::Name(name) => receiver.store_lookup(nogc, heap, name, *acc, semantics),
                }
            }));
            step_try!(apply_store_outcome(
                heap, state, stack, cache, meta, pc, receiver, *acc, outcome,
            ));
            Step::Next
        }
        Opcode::CreateObjectFromMap => {
            let map = step_try!(heap.no_gc(|nogc, heap| {
                let v = cache.constants_ref(nogc).at(ops.idx(0));
                v.get_as::<Map>(nogc, heap.known().map_map)
                    .map(|r| r.into_tagged())
                    .ok_or(VmError::Type)
            }));
            let count = ops.reg_count(2);
            cache.spill_acc(*acc);
            let args = stack.args(&meta, ops.reg_list(1), count);

            // TODO: maybe have a handlescope always accessible or a quickspill cache
            let obj = state.handle_scope(|scope| {
                let map = scope.create_handle(map).expect("map is a strong pointer");
                heap.allocate_object(
                    &scope,
                    ObjectSlotsInit {
                        map,
                        values: args,
                        elements: heap.known().empty_fixed_array.erase(),
                        length: 0,
                    },
                )
            });
            let _ = cache.take_acc();
            *acc = obj.erase();
            Step::Next
        }
        // TODO: create more array creation operations, this one is only for &[Value]
        Opcode::CreateArrayLiteral => {
            let count = ops.reg_count(1);
            cache.spill_acc(*acc);
            let args = stack.args(&meta, ops.reg_list(0), count);
            let array = heap
                .allocate_enter_nogc(args, |dst: HeapRef<'_, FixedArray>, _nogc, _| {
                    dst.into_tagged().erase()
                });
            let _ = cache.take_acc();
            *acc = array;
            Step::Next
        }
        Opcode::LoadContextSlot => {
            let v = step_try!(heap.no_gc(|nogc, heap| {
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
            }));
            *acc = v;
            Step::Next
        }
        Opcode::StoreContextSlot => {
            step_try!(heap.no_gc(|nogc, heap| {
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
                context.slots.heap_ref(nogc).element_slot(ops.idx(0)).set(
                    heap,
                    host,
                    Tagged::from_value(*acc),
                );
                Ok(())
            }));
            Step::Next
        }
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}
