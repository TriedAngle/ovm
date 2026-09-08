use bytecode::{Opcode, Operands, PropertyFlags, decode, jump_target};

use vm::{
    AccessorPair, CallTarget, CallableInfoObject, Compare, Context, ContextInit, Convert,
    FixedArray, Float, GcSlice, Handle, Heap, HeapRef, Key, LoadOutcome, Lookup, NoGc, Object,
    ObjectSlotsInit, PropertyDescriptor, ScopeInfo, SlotName, Smi, StoreOutcome, StoreSemantics,
    Tagged, VMString, Value, ValueRef, call_target, classify_key, element_value, load_outcome,
    store_array_element,
};

use crate::{
    ContextState, FrameMeta, NativeContext, NativeIndex, Stack, StackCache, VM, VmError,
    error_from_vm_error,
    runtime::{Coercion, Hint, Runtime},
};

pub fn execute(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: GcSlice<'_>,
    new_target: Option<Handle<'_, Object>>,
) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;

    let saved_top = stack.top();
    let was_active = cache.is_active();
    // a nested run reuses the cache; the caller's accumulator is dead across
    // the call (the call result overwrites it), so it needs no save/restore
    if was_active {
        stack.suspend_frame(cache.frame_meta());
    }
    let base_depth = stack.frame_depth();

    let result = start(vm, heap, state, callable, args, new_target, base_depth);

    stack.truncate_frames(base_depth);
    if was_active {
        let outer = stack.pop_frame(saved_top).expect("suspended caller frame");
        cache.load(stack, outer, heap);
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
    args: GcSlice<'_>,
    new_target: Option<Handle<'_, Object>>,
    base_depth: usize,
) -> Result<Value, VmError> {
    match call_target(&heap.guard(), callable.value()) {
        Some(CallTarget::Native(idx)) => {
            let f = vm.native(NativeIndex(idx));
            let (saved_top, fargs) = state.stack.stage_args(args)?;
            let mut nctx = NativeContext::with_new_target(vm, heap, state, new_target);
            let result = f(&mut nctx, fargs);
            state.stack.set_top(saved_top);
            result
        }
        Some(CallTarget::Bytecode(target, register_count, kind)) => {
            if new_target.is_none() && kind.is_class_constructor() {
                return Err(VmError::Type);
            }
            // TODO: update once we have classes
            let stack = &state.stack;
            let context = closure_context(&heap.guard(), target);
            let frame = stack.push_initial_frame(target, register_count, context, args)?;
            state.cache.enter(stack, frame, heap);
            dispatch(vm, heap, state, base_depth)
        }
        None => Err(VmError::Type),
    }
}

fn callable_name<'a>(nogc: &'a NoGc<'a>, stack: &Stack, meta: &FrameMeta, idx: usize) -> SlotName {
    let callable = stack.callable_slot(meta).inner();
    let ValueRef::Object(callable) = callable.value_ref(nogc) else {
        panic!("frame callable must be an object");
    };
    let info = callable
        .as_ref()
        .callable_info(nogc)
        .expect("frame callable must have callable info");
    info.constant_slot_name(nogc, idx)
}

/// The frame's current context: the frame header's context slot (per-frame,
/// recursion-safe; the closure-context slot of the function object is the
/// immutable captured context used to initialize it).
fn frame_context(heap: &mut Heap, stack: &Stack, meta: &FrameMeta) -> Result<Value, VmError> {
    let _ = heap;
    Ok(stack.context_slot(meta).inner())
}

fn set_frame_context(
    heap: &mut Heap,
    stack: &Stack,
    meta: &FrameMeta,
    context: Value,
) -> Result<(), VmError> {
    heap.no_gc(|nogc| {
        context
            .get_as::<Context>(nogc, nogc.known().context_map)
            .ok_or(VmError::Type)?;
        stack.context_slot(meta).store(context);
        Ok(())
    })
}

/// The closure context a freshly pushed frame starts with: the callee's
/// immutable captured context (its closure slot).
fn closure_context<'a>(nogc: &'a NoGc<'a>, callable: Tagged<Object>) -> Value {
    let ValueRef::Object(obj) = callable.erase().value_ref(nogc) else {
        panic!("callable must be an object");
    };
    obj.as_ref()
        .closure_context(nogc)
        .expect("callable must have a closure context")
        .into_tagged()
        .erase()
}

/// Find the slot named `name` in `context`'s chain (direct eval). Returns
/// the slot cell, or Reference when no context in the chain has the name.
fn dynamic_slot<'a>(
    nogc: &'a NoGc<'a>,
    context: &mut HeapRef<'a, Context>,
    name: Value,
) -> Result<&'a vm::GcSlot, VmError> {
    let name_str = name
        .get_as::<VMString>(nogc, nogc.known().string_map)
        .ok_or(VmError::Type)?;
    let name_hash = name_str.hash();
    let name_bytes = name_str.as_slice(nogc);
    loop {
        let ctx = context.as_ref();
        let names = ctx.scope_info.heap_ref(nogc).as_ref().names.heap_ref(nogc);
        for i in 0..names.len() {
            let candidate = names.at(i);
            if let Some(s) = candidate.get_as::<VMString>(nogc, nogc.known().string_map) {
                if s.hash() == name_hash && s.as_slice(nogc) == name_bytes {
                    return Ok(ctx.slots.heap_ref(nogc).as_ref().element_slot(i));
                }
            }
        }
        match ctx.outer.heap_ref(nogc) {
            Some(outer) => *context = outer,
            None => return Err(VmError::Reference),
        }
    }
}

/// Walk the frame context chain looking for a slot named `name` (direct
/// eval: unresolved names resolve through the caller's chain).
fn dynamic_lookup<'a>(
    nogc: &'a NoGc<'a>,
    stack: &Stack,
    meta: &FrameMeta,
    name: Value,
) -> Result<Option<Value>, VmError> {
    let mut context = stack
        .context_slot(meta)
        .inner()
        .get_as::<Context>(nogc, nogc.known().context_map)
        .ok_or(VmError::Type)?;
    match dynamic_slot(nogc, &mut context, name) {
        Ok(slot) => Ok(Some(slot.inner())),
        Err(VmError::Reference) => Ok(None),
        Err(e) => Err(e),
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
    let target = call_target(&heap.guard(), f);
    // TODO: native getters/setters invoke in place instead of pushing a frame
    let Some(CallTarget::Bytecode(target, register_count, kind)) = target else {
        return Ok(false);
    };
    if kind.is_class_constructor() {
        return Err(VmError::Type);
    }
    let context = closure_context(&heap.guard(), target);
    let callee =
        stack.push_frame_with_args(meta, handler_pc, target, register_count, context, args)?;
    cache.load(stack, callee, heap);
    Ok(true)
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
        let handled = heap.no_gc(|nogc| {
            let ValueRef::Object(obj) = stack
                .callable_slot(&cache.frame_meta())
                .inner()
                .value_ref(nogc)
            else {
                return None;
            };
            let info = obj.as_ref().callable_info(nogc)?;
            info.handlers.heap_ref(nogc)?.lookup(pc)
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
    err: VmError,
    pc: usize,
) -> Unwind {
    let ex =
        error_from_vm_error(vm, heap, state, err).expect("error materialization must not fail");
    state.set_pending_exception(ex);
    exception_dispatch(heap, state, base_depth, pc)
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

fn dispatch(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    base_depth: usize,
) -> Result<Value, VmError> {
    let cache = &state.cache;
    let mut trace = 0;
    loop {
        let pc = cache.pc();
        if std::env::var("OVM_TRACE").is_ok() {
            trace += 1;
            if trace > 600 {
                panic!("trace limit");
            }
            eprintln!(
                "pc={pc} rc={} acc={:?}",
                cache.frame_meta().register_count,
                cache.acc(),
            );
        }
        let (op, ops, next_pc) = decode(cache.code_ref(&heap.guard()).as_slice(), pc);
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        let result = step(vm, heap, state, base_depth, op, ops, meta, pc);
        match result {
            Step::Next => {}
            Step::Return(v) => return Ok(v),
            Step::Throw(v) => {
                state.set_pending_exception(v);
                match exception_dispatch(heap, state, base_depth, pc) {
                    Unwind::Caught(ex) => cache.set_acc(ex),
                    Unwind::Escaped => return Ok(heap.known().exception.value()),
                }
            }
            Step::PendingThrow => match exception_dispatch(heap, state, base_depth, pc) {
                Unwind::Caught(ex) => cache.set_acc(ex),
                Unwind::Escaped => return Ok(heap.known().exception.value()),
            },
            Step::Error(err) => match raise(vm, heap, state, base_depth, err, pc) {
                Unwind::Caught(ex) => cache.set_acc(ex),
                Unwind::Escaped => return Ok(heap.known().exception.value()),
            },
        }
    }
}

fn apply_store_outcome(
    heap: &mut Heap,
    state: &ContextState,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    pc: usize,
    receiver: Value,
    outcome: StoreOutcome,
) -> Result<(), VmError> {
    match outcome {
        StoreOutcome::Transition { receiver, name } => {
            let result = state.handle_scope(|scope| {
                Object::add_own_property_values(
                    heap,
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(cache.acc()),
                )
                // TODO(strict-mode): a false result must throw in strict code;
                // the current store path preserves its existing sloppy result.
                .map(|_| ())
            });
            result
        }
        StoreOutcome::CallSetter { setter } => {
            call_value(
                heap,
                stack,
                cache,
                meta,
                pc,
                setter,
                &[receiver, cache.acc()],
            )?;
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
                return Step::Return(cache.acc());
            }
            let caller = stack
                .pop_frame(meta.base)
                .expect("suspended frame above base depth");
            cache.load(stack, caller, heap);
            Step::Next
        }
        Opcode::Load => {
            cache.set_acc(stack.reg(&meta, ops.reg(0)));
            Step::Next
        }
        Opcode::Store => {
            stack.set_reg(&meta, ops.reg(0), cache.acc());
            Step::Next
        }
        Opcode::Move => {
            let v = stack.reg(&meta, ops.reg(1));
            stack.set_reg(&meta, ops.reg(0), v);
            Step::Next
        }
        Opcode::LoadSmi => {
            cache.set_acc(Smi::new(ops.imm(0) as i64).encode());
            Step::Next
        }
        Opcode::LoadConstant => {
            let v = cache.constants_ref(&heap.guard()).at(ops.idx(0));
            cache.set_acc(v);
            Step::Next
        }
        Opcode::Add => {
            let other = stack.reg(&meta, ops.reg(0));
            let mut result = None;
            if let (Some(a), Some(b)) = (Smi::decode(cache.acc()), Smi::decode(other)) {
                if let Some(r) = a.value().checked_add(b.value()) {
                    if Smi::in_range(r) {
                        result = Some(Smi::new(r).encode());
                    }
                }
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let lhs = step_try!(Runtime::to_primitive(
                        vm,
                        heap,
                        state,
                        cache.acc(),
                        Hint::Default
                    ));
                    let lhs = match lhs {
                        Coercion::Threw => return Step::PendingThrow,
                        Coercion::Value(v) => v,
                    };
                    let rhs =
                        step_try!(Runtime::to_primitive(vm, heap, state, other, Hint::Default));
                    let rhs = match rhs {
                        Coercion::Threw => return Step::PendingThrow,
                        Coercion::Value(v) => v,
                    };
                    let is_string = heap.no_gc(|nogc| {
                        let known = nogc.known();
                        (
                            lhs.get_as::<VMString>(nogc, known.string_map).is_some(),
                            rhs.get_as::<VMString>(nogc, known.string_map).is_some(),
                        )
                    });
                    if is_string.0 || is_string.1 {
                        let s = step_try!(state.handle_scope(|scope| {
                            let a = Convert::to_string(heap, &scope, vm.interner(), lhs)?;
                            let b = Convert::to_string(heap, &scope, vm.interner(), rhs)?;
                            Ok::<_, VmError>(VMString::concat(heap, &scope, a, b).value())
                        }));
                        cache.set_acc(s);
                    } else {
                        let r = step_try!(heap.no_gc(|nogc| {
                            let a = Convert::to_number(nogc, lhs)?;
                            let b = Convert::to_number(nogc, rhs)?;
                            // IEEE `-0 + -0` yields +0; the spec demands -0
                            let r = a + b;
                            let r = if r == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                                -0.0
                            } else {
                                r
                            };
                            Ok::<_, VmError>(r)
                        }));
                        let v = state.handle_scope(|scope| Convert::to_value(heap, &scope, r));
                        cache.set_acc(v);
                    }
                }
            }
            Step::Next
        }
        Opcode::Sub => {
            let other = stack.reg(&meta, ops.reg(0));
            let mut result = None;
            if let (Some(a), Some(b)) = (Smi::decode(cache.acc()), Smi::decode(other)) {
                if let Some(r) = a.value().checked_sub(b.value()) {
                    if Smi::in_range(r) {
                        result = Some(Smi::new(r).encode());
                    }
                }
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        cache.acc(),
                        other,
                        |a, b| a - b
                    ));
                    let Some(v) = v else {
                        return Step::PendingThrow;
                    };
                    cache.set_acc(v);
                }
            }
            Step::Next
        }
        Opcode::Mul => {
            let other = stack.reg(&meta, ops.reg(0));
            let mut result = None;
            if let (Some(a), Some(b)) = (Smi::decode(cache.acc()), Smi::decode(other)) {
                if let Some(r) = a.value().checked_mul(b.value()) {
                    if Smi::in_range(r) {
                        result = Some(Smi::new(r).encode());
                    }
                }
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        cache.acc(),
                        other,
                        |a, b| a * b
                    ));
                    let Some(v) = v else {
                        return Step::PendingThrow;
                    };
                    cache.set_acc(v);
                }
            }
            Step::Next
        }
        Opcode::Div => {
            // JS division is IEEE double division: 7/2 = 3.5, x/0 = ±Infinity
            // or NaN, MIN/-1 overflows to a double.
            let other = stack.reg(&meta, ops.reg(0));
            let mut result = None;
            if let (Some(a), Some(b)) = (Smi::decode(cache.acc()), Smi::decode(other)) {
                if b.value() != 0 && a.value() % b.value() == 0 {
                    if let Some(r) = a.value().checked_div(b.value()) {
                        result = Some(Smi::new(r).encode());
                    }
                }
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        cache.acc(),
                        other,
                        |a, b| a / b
                    ));
                    let Some(v) = v else {
                        return Step::PendingThrow;
                    };
                    cache.set_acc(v);
                }
            }
            Step::Next
        }
        Opcode::Mod => {
            // JS remainder is IEEE fmod: x % 0 = NaN, signs follow the dividend.
            let other = stack.reg(&meta, ops.reg(0));
            let mut result = None;
            if let (Some(a), Some(b)) = (Smi::decode(cache.acc()), Smi::decode(other)) {
                if b.value() != 0 {
                    result = Some(Smi::new(a.value() % b.value()).encode());
                }
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        cache.acc(),
                        other,
                        |a, b| a % b
                    ));
                    let Some(v) = v else {
                        return Step::PendingThrow;
                    };
                    cache.set_acc(v);
                }
            }
            Step::Next
        }
        Opcode::Exp => {
            // JS exponentiation is always IEEE double math; the result only
            // needs a Smi tag when it is an in-range integer.
            let other = stack.reg(&meta, ops.reg(0));
            let v = step_try!(Runtime::numeric_op(
                vm,
                heap,
                state,
                cache.acc(),
                other,
                |a, b| a.powf(b)
            ));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            cache.set_acc(v);
            Step::Next
        }
        Opcode::BitwiseOr => {
            // ToInt32 semantics on the (integer) smi inputs
            let a = step_try!(Smi::decode(cache.acc()).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as i32;
            cache.set_acc(Smi::new((a | b) as i64).encode());
            Step::Next
        }
        Opcode::BitwiseXor => {
            let a = step_try!(Smi::decode(cache.acc()).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as i32;
            cache.set_acc(Smi::new((a ^ b) as i64).encode());
            Step::Next
        }
        Opcode::BitwiseAnd => {
            let a = step_try!(Smi::decode(cache.acc()).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as i32;
            cache.set_acc(Smi::new((a & b) as i64).encode());
            Step::Next
        }
        Opcode::ShiftLeft => {
            // ToInt32(lhs) << (ToUint32(rhs) & 31), truncated to int32
            let a = step_try!(Smi::decode(cache.acc()).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as u32;
            cache.set_acc(Smi::new((a.wrapping_shl(b & 31) as i32) as i64).encode());
            Step::Next
        }
        Opcode::ShiftRight => {
            // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending
            let a = step_try!(Smi::decode(cache.acc()).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as u32;
            cache.set_acc(Smi::new((a.wrapping_shr(b & 31) as i32) as i64).encode());
            Step::Next
        }
        Opcode::ShiftRightLogical => {
            // ToUint32(lhs) >>> (ToUint32(rhs) & 31): always non-negative
            let a = step_try!(Smi::decode(cache.acc()).ok_or(VmError::Type)).value() as u32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as u32;
            cache.set_acc(Smi::new(a.wrapping_shr(b & 31) as i64).encode());
            Step::Next
        }
        Opcode::Jump => {
            cache.set_pc(jump_target(pc, ops.imm(0)));
            Step::Next
        }
        Opcode::JumpLoop => {
            heap.safepoint_poll();
            cache.set_pc(jump_target(pc, ops.imm(0)));
            Step::Next
        }
        Opcode::JumpIfTruthy => {
            if Convert::is_truthy(&heap.guard(), cache.acc()) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfFalsy => {
            if !Convert::is_truthy(&heap.guard(), cache.acc()) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::TestReferenceEqual => {
            let other = stack.reg(&meta, ops.reg(0));
            let known = heap.known();
            cache.set_acc(if other == cache.acc() {
                known.true_object.value()
            } else {
                known.false_object.value()
            });
            Step::Next
        }
        Opcode::TestTypeof => {
            cache.set_acc(Runtime::type_of(vm, heap, state, cache.acc()));
            Step::Next
        }
        Opcode::Negate => {
            if let Some(smi) = Smi::decode(cache.acc()) {
                let v = smi.value();
                cache.set_acc(if v == 0 {
                    state.handle_scope(|scope| heap.allocate_handle::<Float>(-0.0, &scope).value())
                } else if v == Smi::MIN {
                    state.handle_scope(|scope| Convert::to_value(heap, &scope, -(v as f64)))
                } else {
                    Smi::new(-v).encode()
                });
            } else {
                let n = step_try!(Runtime::to_numeric(vm, heap, state, cache.acc()));
                let Some(n) = n else {
                    return Step::PendingThrow;
                };
                cache.set_acc(state.handle_scope(|scope| {
                    let r = -n;
                    // preserve -0.0: `-0` must not fold into Smi 0
                    if r == 0.0 && r.is_sign_negative() {
                        heap.allocate_handle::<Float>(-0.0, &scope).value()
                    } else {
                        Convert::to_value(heap, &scope, r)
                    }
                }));
            }
            Step::Next
        }
        Opcode::InstanceOf => {
            let other = stack.reg(&meta, ops.reg(0));
            let r = step_try!(Runtime::instance_of(vm, heap, state, cache.acc(), other));
            let Some(r) = r else {
                return Step::PendingThrow;
            };
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::Construct => {
            // ES 9.2.2 [[Construct]]: `new.target` (here the callee itself;
            // `super()`/Reflect.construct thread a different target later)
            // decides the receiver's prototype via GetPrototypeFromConstructor,
            // the callee runs with the receiver as `this`, and an object
            // result wins over the receiver.
            let constructible = heap.no_gc(|nogc| {
                let callee = stack.reg(&meta, ops.reg(0));
                let ValueRef::Object(obj) = callee.value_ref(nogc) else {
                    return false;
                };
                obj.as_ref()
                    .header
                    .map
                    .heap_ref(nogc)
                    .kind()
                    .is_constructor()
            });
            if !constructible {
                return Step::Error(VmError::Type);
            }
            state.handle_scope(|scope| {
                let callee = scope
                    .create_handle(unsafe {
                        Tagged::<Object>::from_value_unchecked(stack.reg(&meta, ops.reg(0)))
                    })
                    .expect("callee must be strong");
                let receiver = match Runtime::create_construct_receiver(vm, heap, state, callee) {
                    Ok(Some(r)) => r,
                    Ok(None) => return Step::PendingThrow,
                    Err(err) => return Step::Error(err),
                };
                let receiver = scope
                    .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(receiver) })
                    .expect("receiver is an object");
                // the register list holds only arguments; the receiver is
                // synthesized and prepended
                let count = ops.reg_count(2);
                let mut args = Vec::with_capacity(count + 1);
                args.push(receiver.value());
                args.extend_from_slice(stack.args(&meta, ops.reg_list(1), count).as_slice());
                let result = match NativeContext::new(vm, heap, state).call_construct(
                    callee.value(),
                    callee.value(),
                    unsafe { GcSlice::from_slice(&args) },
                ) {
                    Ok(r) => r,
                    Err(err) => return Step::Error(err),
                };
                if result == heap.known().exception.value() {
                    return Step::PendingThrow;
                }
                cache.set_acc(if Convert::is_primitive(&heap.guard(), result) {
                    receiver.value()
                } else {
                    result
                });
                Step::Next
            })
        }
        Opcode::EqualStrict => {
            let other = stack.reg(&meta, ops.reg(0));
            let r = Compare::strict_equal(&heap.guard(), cache.acc(), other);
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::Equal => {
            // IsLooselyEqual: objects are ToPrimitive'd (hint default) first
            let other = stack.reg(&meta, ops.reg(0));
            let x = step_try!(Runtime::to_primitive(
                vm,
                heap,
                state,
                cache.acc(),
                Hint::Default
            ));
            let x = match x {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let y = step_try!(Runtime::to_primitive(vm, heap, state, other, Hint::Default));
            let y = match y {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let r = step_try!(Compare::equal(&heap.guard(), x, y));
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::LessThan => {
            // Abstract Relational Comparison: objects ToPrimitive'd with hint Number
            let other = stack.reg(&meta, ops.reg(0));
            let x = step_try!(Runtime::to_primitive(
                vm,
                heap,
                state,
                cache.acc(),
                Hint::Number
            ));
            let x = match x {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let y = step_try!(Runtime::to_primitive(vm, heap, state, other, Hint::Number));
            let y = match y {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let r = step_try!(Compare::less_than(&heap.guard(), x, y));
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::LessThanOrEqual => {
            let other = stack.reg(&meta, ops.reg(0));
            let x = step_try!(Runtime::to_primitive(
                vm,
                heap,
                state,
                cache.acc(),
                Hint::Number
            ));
            let x = match x {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let y = step_try!(Runtime::to_primitive(vm, heap, state, other, Hint::Number));
            let y = match y {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let r = step_try!(Compare::less_than_or_equal(&heap.guard(), x, y));
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::GreaterThan => {
            let other = stack.reg(&meta, ops.reg(0));
            let x = step_try!(Runtime::to_primitive(
                vm,
                heap,
                state,
                cache.acc(),
                Hint::Number
            ));
            let x = match x {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let y = step_try!(Runtime::to_primitive(vm, heap, state, other, Hint::Number));
            let y = match y {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let r = step_try!(Compare::greater_than(&heap.guard(), x, y));
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::GreaterThanOrEqual => {
            let other = stack.reg(&meta, ops.reg(0));
            let x = step_try!(Runtime::to_primitive(
                vm,
                heap,
                state,
                cache.acc(),
                Hint::Number
            ));
            let x = match x {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let y = step_try!(Runtime::to_primitive(vm, heap, state, other, Hint::Number));
            let y = match y {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => v,
            };
            let r = step_try!(Compare::greater_than_or_equal(&heap.guard(), x, y));
            cache.set_acc(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::Throw | Opcode::ReThrow => Step::Throw(cache.acc()),
        Opcode::CallNative => {
            let f = vm.native(NativeIndex(ops.idx(0)));
            let count = ops.reg_count(2);
            let mut nctx = NativeContext::new(vm, heap, state);
            let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
            match result {
                Ok(v) if v == heap.known().exception.value() => Step::PendingThrow,
                Ok(v) => {
                    cache.set_acc(v);
                    Step::Next
                }
                Err(err) => Step::Error(err),
            }
        }
        // TODO: feedback vectors and separation once they are there
        Opcode::Call | Opcode::CallNoFeedback => {
            // TODO(strict-mode): ordinary sloppy functions still need nullish
            // receiver substitution and primitive receiver boxing.
            let count = ops.reg_count(2);
            let target = call_target(&heap.guard(), stack.reg(&meta, ops.reg(0)));
            let Some(target) = target else {
                return Step::Error(VmError::Type);
            };
            match target {
                CallTarget::Native(idx) => {
                    let f = vm.native(NativeIndex(idx));
                    let mut nctx = NativeContext::new(vm, heap, state);
                    let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                    match result {
                        Ok(v) if v == heap.known().exception.value() => Step::PendingThrow,
                        Ok(v) => {
                            cache.set_acc(v);
                            Step::Next
                        }
                        Err(err) => Step::Error(err),
                    }
                }
                CallTarget::Bytecode(target, register_count, kind) => {
                    if kind.is_class_constructor() {
                        return Step::Error(VmError::Type);
                    }
                    let context = closure_context(&heap.guard(), target);
                    let callee = step_try!(stack.push_frame(
                        meta,
                        pc,
                        target,
                        register_count,
                        context,
                        ops.reg_list(1),
                        count,
                    ));
                    cache.load(stack, callee, heap);
                    Step::Next
                }
            }
        }
        Opcode::LoadNamedProperty => {
            let outcome = step_try!(heap.no_gc(|nogc| {
                let name = callable_name(nogc, stack, &meta, ops.idx(1));
                load_outcome(nogc, stack.reg(&meta, ops.reg(0)), name)
            }));
            match outcome {
                LoadOutcome::Value(v) => cache.set_acc(v),
                LoadOutcome::Getter(getter) => {
                    let receiver = stack.reg(&meta, ops.reg(0));
                    let called = step_try!(call_value(
                        heap,
                        stack,
                        cache,
                        meta,
                        pc,
                        getter,
                        &[receiver]
                    ));
                    if !called {
                        // non-callable getter: the load yields undefined
                        cache.set_acc(heap.known().undefined.value());
                    }
                }
            }
            Step::Next
        }
        Opcode::StoreNamedProperty | Opcode::StoreNamedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreNamedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            let receiver = stack.reg(&meta, ops.reg(0));
            let outcome = step_try!(heap.no_gc(|nogc| {
                let name = callable_name(nogc, stack, &meta, ops.idx(1));
                receiver.store_lookup(nogc, name, cache.acc(), semantics)
            }));
            step_try!(apply_store_outcome(
                heap, state, stack, cache, meta, pc, receiver, outcome,
            ));
            Step::Next
        }
        Opcode::LoadKeyedProperty => {
            let receiver = stack.reg(&meta, ops.reg(0));
            let outcome = step_try!(heap.no_gc(|nogc| {
                match classify_key(nogc, cache.acc())? {
                    Key::Element(i) => match element_value(nogc, receiver, i) {
                        Some(v) => Ok(LoadOutcome::Value(v)),
                        // past the end, a hole, or a non-array receiver:
                        // fall back to an ordinary property lookup
                        None => load_outcome(
                            nogc,
                            receiver,
                            SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                        ),
                    },
                    Key::Name(name) => load_outcome(nogc, receiver, name),
                }
            }));
            match outcome {
                LoadOutcome::Value(v) => cache.set_acc(v),
                LoadOutcome::Getter(getter) => {
                    let called = step_try!(call_value(
                        heap,
                        stack,
                        cache,
                        meta,
                        pc,
                        getter,
                        &[receiver]
                    ));
                    if !called {
                        cache.set_acc(heap.known().undefined.value());
                    }
                }
            }
            Step::Next
        }
        Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreKeyedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            let receiver = stack.reg(&meta, ops.reg(0));
            let key = stack.reg(&meta, ops.reg(1));
            let key = step_try!(classify_key(&heap.guard(), key));
            match key {
                Key::Element(i) => {
                    let is_array = heap.no_gc(|nogc| {
                        let ValueRef::Object(obj) = receiver.value_ref(nogc) else {
                            return false;
                        };
                        obj.as_ref().is_array(nogc)
                    });
                    if is_array {
                        step_try!(state.handle_scope(|scope| {
                            store_array_element(heap, &scope, receiver, i, cache.acc())
                        }));
                    } else {
                        // numeric property on a non-array receiver
                        let outcome = step_try!(heap.no_gc(|nogc| {
                            receiver.store_lookup(
                                nogc,
                                SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                                cache.acc(),
                                semantics,
                            )
                        }));
                        step_try!(apply_store_outcome(
                            heap, state, stack, cache, meta, pc, receiver, outcome,
                        ));
                    }
                }
                Key::Name(name) => {
                    let outcome = step_try!(heap.no_gc(|nogc| {
                        receiver.store_lookup(nogc, name, cache.acc(), semantics)
                    }));
                    step_try!(apply_store_outcome(
                        heap, state, stack, cache, meta, pc, receiver, outcome,
                    ));
                }
            }
            Step::Next
        }
        Opcode::CreateAccessorPair => {
            let get = stack.reg(&meta, ops.reg(0));
            let set = stack.reg(&meta, ops.reg(1));
            let pair = state.handle_scope(|scope| {
                heap.allocate_handle::<AccessorPair>((get, set), &scope)
                    .value()
            });
            cache.set_acc(pair);
            Step::Next
        }
        Opcode::DefineNamedOwnProperty | Opcode::DefineKeyedOwnProperty => {
            let receiver = stack.reg(&meta, ops.reg(0));
            let (name, desc) = step_try!(heap.no_gc(|nogc| {
                if !matches!(receiver.value_ref(nogc), ValueRef::Object(_)) {
                    return Err(VmError::Type);
                }
                let name = match op {
                    Opcode::DefineNamedOwnProperty => callable_name(nogc, stack, &meta, ops.idx(1)),
                    _ => match classify_key(nogc, stack.reg(&meta, ops.reg(1)))? {
                        Key::Element(i) => SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                        Key::Name(name) => name,
                    },
                };
                let flags = ops.uimm(2);
                let enumerable = flags & PropertyFlags::DontEnum.bits() == 0;
                let configurable = flags & PropertyFlags::DontDelete.bits() == 0;
                let desc = if flags & PropertyFlags::Accessor.bits() != 0 {
                    let pair = cache
                        .acc()
                        .get_as::<AccessorPair>(nogc, nogc.known().accessor_pair_map)
                        .ok_or(VmError::Type)?;
                    let pair = pair.as_ref();
                    PropertyDescriptor::Accessor {
                        get: pair.get.inner(),
                        set: pair.set.inner(),
                        enumerable,
                        configurable,
                    }
                } else {
                    PropertyDescriptor::Data {
                        value: cache.acc(),
                        writable: flags & PropertyFlags::ReadOnly.bits() == 0,
                        enumerable,
                        configurable,
                    }
                };
                Ok((name, desc))
            }));
            let defined = state.handle_scope(|scope| {
                Object::define_own_property_values(heap, &scope, receiver, name, desc)
            });
            let defined = step_try!(defined);
            if !defined {
                return Step::Error(VmError::Type);
            }
            Step::Next
        }
        Opcode::CreateEmptyObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = scope
                    .create_handle(heap.known().object_initial_map.as_tagged())
                    .expect("object initial map is strong");
                heap.allocate_object(
                    &scope,
                    ObjectSlotsInit {
                        map,
                        values: &[],
                        elements: heap.known().empty_fixed_array.erase(),
                        length: 0,
                    },
                )
            });
            cache.set_acc(obj.erase());
            Step::Next
        }
        Opcode::CreateEmptyArrayLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = scope
                    .create_handle(heap.known().js_array_map.as_tagged())
                    .expect("array object map is strong");
                heap.allocate_object(
                    &scope,
                    ObjectSlotsInit {
                        map,
                        values: &[],
                        elements: heap.known().empty_fixed_array.erase(),
                        length: 0,
                    },
                )
                .into_tagged()
                .erase()
            });
            cache.set_acc(obj);
            Step::Next
        }
        Opcode::CreateClosure => {
            // constants[idx] is the shared callable-info template (SFI-like);
            // the closure captures the current frame's context
            let info = step_try!(heap.no_gc(|nogc| {
                let v = cache.constants_ref(nogc).at(ops.idx(0));
                v.get_as::<CallableInfoObject>(nogc, nogc.known().callable_map)
                    .map(|r| r.into_tagged().erase())
                    .ok_or(VmError::Type)
            }));
            let context = step_try!(frame_context(heap, stack, &meta));
            let obj = state.handle_scope(|scope| {
                let info = scope
                    .create_handle(unsafe {
                        Tagged::<CallableInfoObject>::from_value_unchecked(info)
                    })
                    .expect("callable info is strong");
                Runtime::create_closure(vm, heap, &scope, info, context)
            });
            let obj = step_try!(obj);
            cache.set_acc(obj);
            Step::Next
        }
        Opcode::LoadGlobal | Opcode::LoadGlobalNoThrow => {
            enum GlobalLoad {
                Data(Value),
                Getter(Value),
                Missing,
            }
            let global = heap.known().global_object.value();
            let lookup = heap.no_gc(|nogc| {
                let name = callable_name(nogc, stack, &meta, ops.idx(0));
                match global.lookup(nogc, name) {
                    Lookup::Data { slot, .. } => GlobalLoad::Data(slot.inner()),
                    Lookup::Accessor { pair, .. } => GlobalLoad::Getter(pair.get.inner()),
                    Lookup::NotFound => GlobalLoad::Missing,
                }
            });
            match lookup {
                GlobalLoad::Data(v) => cache.set_acc(v),
                GlobalLoad::Getter(getter) => {
                    if getter == heap.known().undefined.value() {
                        cache.set_acc(heap.known().undefined.value());
                    } else {
                        let called =
                            step_try!(call_value(heap, stack, cache, meta, pc, getter, &[global]));
                        if !called {
                            cache.set_acc(heap.known().undefined.value());
                        }
                    }
                }
                GlobalLoad::Missing => {
                    if op == Opcode::LoadGlobal {
                        // unresolvable reference: GetValue throws ReferenceError
                        return Step::Error(VmError::Reference);
                    }
                    cache.set_acc(heap.known().undefined.value());
                }
            }
            // TODO: full semantics: lookup the script-context table first
            // (lexical globals), throw ReferenceError on unresolved loads;
            // the global-object property path is the only one implemented
            Step::Next
        }
        Opcode::StoreGlobal => {
            let global = heap.known().global_object.value();
            let outcome = step_try!(heap.no_gc(|nogc| {
                let name = callable_name(nogc, stack, &meta, ops.idx(0));
                global.store_lookup(nogc, name, cache.acc(), StoreSemantics::WriteThrough)
            }));
            step_try!(apply_store_outcome(
                heap, state, stack, cache, meta, pc, global, outcome,
            ));
            Step::Next
        }
        Opcode::CreateFunctionContext => {
            // constants[idx] is the scope's shared ScopeInfo (its `names`
            // array is parallel to the context's slots)
            let scope_info = step_try!(heap.no_gc(|nogc| {
                cache
                    .constants_ref(nogc)
                    .at(ops.idx(0))
                    .get_as::<ScopeInfo>(nogc, nogc.known().scope_info_map)
                    .map(|r| r.into_tagged().erase())
                    .ok_or(VmError::Type)
            }));
            let count = step_try!(heap.no_gc(|nogc| {
                scope_info
                    .get_as::<ScopeInfo>(nogc, nogc.known().scope_info_map)
                    .map(|r| r.as_ref().names.heap_ref(nogc).len())
                    .ok_or(VmError::Type)
            }));
            let hole = heap.known().void.value();
            let values = vec![hole; count];
            let outer = step_try!(frame_context(heap, stack, &meta));
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .create_handle(unsafe { Tagged::<Context>::from_value_unchecked(outer) })
                    .expect("frame context is strong");
                let scope_info = scope
                    .create_handle(unsafe { Tagged::<ScopeInfo>::from_value_unchecked(scope_info) })
                    .expect("scope info is strong");
                let slots = heap.allocate_handle::<FixedArray>(&values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info,
                })
            });
            cache.set_acc(ctx.erase());
            Step::Next
        }
        Opcode::CreateBlockContext => {
            let count = ops.uimm(0) as usize;
            let hole = heap.known().void.value();
            let values = vec![hole; count];
            let outer = step_try!(frame_context(heap, stack, &meta));
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .create_handle(unsafe { Tagged::<Context>::from_value_unchecked(outer) })
                    .expect("frame context is strong");
                let slots = heap.allocate_handle::<FixedArray>(&values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info: heap.known().empty_scope_info,
                })
            });
            cache.set_acc(ctx.erase());
            Step::Next
        }
        Opcode::CreateCatchContext => {
            let exception = stack.reg(&meta, ops.reg(0));
            let outer = step_try!(frame_context(heap, stack, &meta));
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .create_handle(unsafe { Tagged::<Context>::from_value_unchecked(outer) })
                    .expect("frame context is strong");
                let slots = heap.allocate_handle::<FixedArray>(&[exception], &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info: heap.known().empty_scope_info,
                })
            });
            cache.set_acc(ctx.erase());
            Step::Next
        }
        Opcode::PushContext => {
            let old = step_try!(frame_context(heap, stack, &meta));
            stack.set_reg(&meta, ops.reg(0), old);
            step_try!(set_frame_context(heap, stack, &meta, cache.acc()));
            Step::Next
        }
        Opcode::PopContext => {
            let context = stack.reg(&meta, ops.reg(0));
            step_try!(set_frame_context(heap, stack, &meta, context));
            Step::Next
        }
        Opcode::SetPrototype => {
            let proto = stack.reg(&meta, ops.reg(0));
            let result =
                state.handle_scope(|scope| Object::set_prototype(heap, &scope, cache.acc(), proto));
            step_try!(result);
            Step::Next
        }
        Opcode::ThrowReferenceErrorIfHole => {
            if cache.acc() == heap.known().void.value() {
                return Step::Error(VmError::Reference);
            }
            Step::Next
        }
        Opcode::LoadContextSlot => {
            let depth = ops.uimm(1);
            let v = step_try!(heap.no_gc(|nogc| {
                let mut context = stack
                    .context_slot(&meta)
                    .inner()
                    .get_as::<Context>(nogc, nogc.known().context_map)
                    .ok_or(VmError::Type)?;
                for _ in 0..depth {
                    context = context.as_ref().outer.heap_ref(nogc).ok_or(VmError::Type)?;
                }
                Ok(context
                    .slots
                    .heap_ref(nogc)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .inner())
            }));
            cache.set_acc(v);
            Step::Next
        }
        Opcode::StoreContextSlot => {
            step_try!(heap.no_gc(|nogc| {
                let mut context = stack
                    .context_slot(&meta)
                    .inner()
                    .get_as::<Context>(nogc, nogc.known().context_map)
                    .ok_or(VmError::Type)?;
                for _ in 0..ops.uimm(1) {
                    context = context.as_ref().outer.heap_ref(nogc).ok_or(VmError::Type)?;
                }
                let host = context.clone().into_tagged().erase();
                context
                    .slots
                    .heap_ref(nogc)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .set(nogc, host, Tagged::from_value(cache.acc()));
                Ok(())
            }));
            Step::Next
        }
        Opcode::LoadDynamicName => {
            let name = cache.constants_ref(&heap.guard()).at(ops.idx(0));
            let found = step_try!(dynamic_lookup(&heap.guard(), stack, &meta, name));
            match found {
                Some(v) if v != heap.known().void.value() => cache.set_acc(v),
                Some(_) => return Step::Error(VmError::Reference),
                None => {
                    // unresolved: fall back to a global object property
                    let global = heap.known().global_object.value();
                    let outcome =
                        step_try!(heap.no_gc(|nogc| {
                            load_outcome(nogc, global, SlotName::from_value(name))
                        }));
                    match outcome {
                        LoadOutcome::Value(v) => cache.set_acc(v),
                        LoadOutcome::Getter(getter) => {
                            let called = step_try!(call_value(
                                heap,
                                stack,
                                cache,
                                meta,
                                pc,
                                getter,
                                &[global],
                            ));
                            if !called {
                                cache.set_acc(heap.known().undefined.value());
                            }
                        }
                    }
                }
            }
            Step::Next
        }
        Opcode::StoreDynamicName => {
            let name = cache.constants_ref(&heap.guard()).at(ops.idx(0));
            let found = step_try!(dynamic_lookup(&heap.guard(), stack, &meta, name));
            match found {
                Some(v) if v != heap.known().void.value() => {
                    // write through to the found slot
                    step_try!(heap.no_gc(|nogc| {
                        let mut context = stack
                            .context_slot(&meta)
                            .inner()
                            .get_as::<Context>(nogc, nogc.known().context_map)
                            .ok_or(VmError::Type)?;
                        let target = dynamic_slot(nogc, &mut context, name)?;
                        let host = context.into_tagged().erase();
                        target.set(nogc, host, Tagged::from_value(cache.acc()));
                        Ok(())
                    }));
                }
                Some(_) => return Step::Error(VmError::Reference),
                None => {
                    let global = heap.known().global_object.value();
                    let outcome = step_try!(heap.no_gc(|nogc| {
                        global.store_lookup(
                            nogc,
                            SlotName::from_value(name),
                            cache.acc(),
                            StoreSemantics::WriteThrough,
                        )
                    }));
                    step_try!(apply_store_outcome(
                        heap, state, stack, cache, meta, pc, global, outcome,
                    ));
                }
            }
            Step::Next
        }
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}
