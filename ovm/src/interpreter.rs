use bytecode::{Opcode, Operands, decode, jump_target};

use vm::{
    CallTarget, CallableInfoObject, Context, ContextInit, FixedArray, Handle, Heap, Key,
    LoadOutcome, NoGc, Object, ObjectSlotsInit, SlotName, Smi, StoreOutcome, StoreSemantics,
    Tagged, Value, ValueRef, call_target, classify_key, element_value, encode_smi, is_truthy,
    load_outcome, store_array_element, store_new_data_property_values,
};

use crate::{
    ContextState, FrameMeta, NativeContext, NativeIndex, Stack, StackCache, VM, VmError,
    error_from_vm_error,
};

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

fn callable_name<'a>(
    nogc: &'a NoGc<'a>,
    heap: &'a Heap,
    stack: &Stack,
    meta: &FrameMeta,
    idx: usize,
) -> SlotName {
    let callable = stack.callable_slot(meta).inner();
    let ValueRef::Object(callable) = callable.value_ref(nogc) else {
        panic!("frame callable must be an object");
    };
    let info = callable
        .as_ref()
        .callable_info(nogc, heap)
        .expect("frame callable must have callable info");
    info.constant_slot_name(nogc, heap, idx)
}

/// The frame's current context: the function object's closure context
/// (the interpreter frame's context slot).
fn frame_context(heap: &mut Heap, stack: &Stack, meta: &FrameMeta) -> Result<Value, VmError> {
    heap.no_gc(|nogc, heap| {
        let ValueRef::Object(obj) = stack.callable_slot(meta).inner().value_ref(nogc) else {
            return Err(VmError::Type);
        };
        obj.as_ref()
            .closure_context(nogc, heap)
            .map(|c| c.into_tagged().erase())
            .ok_or(VmError::Type)
    })
}

fn set_frame_context(
    heap: &mut Heap,
    stack: &Stack,
    meta: &FrameMeta,
    context: Value,
) -> Result<(), VmError> {
    heap.no_gc(|nogc, heap| {
        let ValueRef::Object(obj) = stack.callable_slot(meta).inner().value_ref(nogc) else {
            return Err(VmError::Type);
        };
        context
            .get_as::<Context>(nogc, heap.known().context_map)
            .ok_or(VmError::Type)?;
        let host = stack.callable_slot(meta).inner();
        obj.as_ref().slots.heap_ref(nogc).element_slot(1).set(
            heap,
            host,
            Tagged::from_value(context),
        );
        Ok(())
    })
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
            let result = state.handle_scope(|scope| {
                store_new_data_property_values(heap, &scope, receiver, name, acc)
            });
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
            // TODO: JS semantics (ToNumeric coercion, float results, string concat)
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value();
            let b =
                step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)).value();
            let r = step_try!(a.checked_add(b).ok_or(VmError::Overflow));
            *acc = step_try!(encode_smi(r));
            Step::Next
        }
        Opcode::Sub => {
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value();
            let b =
                step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)).value();
            let r = step_try!(a.checked_sub(b).ok_or(VmError::Overflow));
            *acc = step_try!(encode_smi(r));
            Step::Next
        }
        Opcode::Mul => {
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value();
            let b =
                step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)).value();
            let r = step_try!(a.checked_mul(b).ok_or(VmError::Overflow));
            *acc = step_try!(encode_smi(r));
            Step::Next
        }
        Opcode::Div => {
            // TODO: JS semantics: non-integral results and division by zero
            // yield doubles (Infinity/NaN), not errors
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value();
            let b =
                step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)).value();
            let r = step_try!(if b == 0 {
                Err(VmError::Overflow)
            } else {
                a.checked_div(b).ok_or(VmError::Overflow)
            });
            *acc = step_try!(encode_smi(r));
            Step::Next
        }
        Opcode::Mod => {
            // TODO: JS semantics: division by zero yields NaN
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value();
            let b =
                step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)).value();
            let r = step_try!(if b == 0 {
                Err(VmError::Overflow)
            } else {
                a.checked_rem(b).ok_or(VmError::Overflow)
            });
            *acc = step_try!(encode_smi(r));
            Step::Next
        }
        Opcode::Exp => {
            // TODO: JS semantics: fractional results yield doubles
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value();
            let b =
                step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type)).value();
            let r = (a as f64).powf(b as f64);
            let r = step_try!(if !r.is_finite() || r.fract() != 0.0 {
                Err(VmError::Overflow)
            } else {
                Ok(r as i64)
            });
            *acc = step_try!(encode_smi(r));
            Step::Next
        }
        Opcode::BitwiseOr => {
            // ToInt32 semantics on the (integer) smi inputs
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as i32;
            *acc = Smi::new((a | b) as i64).encode();
            Step::Next
        }
        Opcode::BitwiseXor => {
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as i32;
            *acc = Smi::new((a ^ b) as i64).encode();
            Step::Next
        }
        Opcode::BitwiseAnd => {
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as i32;
            *acc = Smi::new((a & b) as i64).encode();
            Step::Next
        }
        Opcode::ShiftLeft => {
            // ToInt32(lhs) << (ToUint32(rhs) & 31), truncated to int32
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as u32;
            *acc = Smi::new((a.wrapping_shl(b & 31) as i32) as i64).encode();
            Step::Next
        }
        Opcode::ShiftRight => {
            // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value() as i32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as u32;
            *acc = Smi::new((a.wrapping_shr(b & 31) as i32) as i64).encode();
            Step::Next
        }
        Opcode::ShiftRightLogical => {
            // ToUint32(lhs) >>> (ToUint32(rhs) & 31): always non-negative
            let a = step_try!(Smi::decode(*acc).ok_or(VmError::Type)).value() as u32;
            let b = step_try!(Smi::decode(stack.reg(&meta, ops.reg(0))).ok_or(VmError::Type))
                .value() as u32;
            *acc = Smi::new(a.wrapping_shr(b & 31) as i64).encode();
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
            let outcome = step_try!(heap.no_gc(|nogc, heap| {
                let name = callable_name(nogc, heap, stack, &meta, ops.idx(1));
                load_outcome(nogc, heap, stack.reg(&meta, ops.reg(0)), name)
            }));
            match outcome {
                LoadOutcome::Value(v) => *acc = v,
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
                let name = callable_name(nogc, heap, stack, &meta, ops.idx(1));
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
                    Key::Element(i) => match element_value(nogc, heap, receiver, i) {
                        Some(v) => Ok(LoadOutcome::Value(v)),
                        // past the end, a hole, or a non-array receiver:
                        // fall back to an ordinary property lookup
                        None => load_outcome(
                            nogc,
                            heap,
                            receiver,
                            SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                        ),
                    },
                    Key::Name(name) => load_outcome(nogc, heap, receiver, name),
                }
            }));
            match outcome {
                LoadOutcome::Value(v) => *acc = v,
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
            let key = step_try!(heap.no_gc(|nogc, heap| classify_key(nogc, heap, key)));
            match key {
                Key::Element(i) => {
                    let is_array = heap.no_gc(|nogc, _heap| {
                        let ValueRef::Object(obj) = receiver.value_ref(nogc) else {
                            return false;
                        };
                        obj.as_ref().is_array(nogc)
                    });
                    if is_array {
                        step_try!(state.handle_scope(|scope| {
                            store_array_element(heap, &scope, receiver, i, *acc)
                        }));
                    } else {
                        // numeric property on a non-array receiver
                        let outcome = step_try!(heap.no_gc(|nogc, heap| {
                            receiver.store_lookup(
                                nogc,
                                heap,
                                SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                                *acc,
                                semantics,
                            )
                        }));
                        step_try!(apply_store_outcome(
                            heap, state, stack, cache, meta, pc, receiver, *acc, outcome,
                        ));
                    }
                }
                Key::Name(name) => {
                    let outcome = step_try!(heap.no_gc(|nogc, heap| {
                        receiver.store_lookup(nogc, heap, name, *acc, semantics)
                    }));
                    step_try!(apply_store_outcome(
                        heap, state, stack, cache, meta, pc, receiver, *acc, outcome,
                    ));
                }
            }
            Step::Next
        }
        Opcode::CreateEmptyObjectLiteral => {
            cache.spill_acc(*acc);
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
            let _ = cache.take_acc();
            *acc = obj.erase();
            Step::Next
        }
        Opcode::CreateEmptyArrayLiteral => {
            cache.spill_acc(*acc);
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
            let _ = cache.take_acc();
            *acc = obj;
            Step::Next
        }
        Opcode::CreateClosure => {
            // constants[idx] is the shared callable-info template (SFI-like);
            // the closure inherits the current frame's context
            let info = step_try!(heap.no_gc(|nogc, heap| {
                let v = cache.constants_ref(nogc).at(ops.idx(0));
                v.get_as::<CallableInfoObject>(nogc, heap.known().callable_map)
                    .map(|r| r.into_tagged().erase())
                    .ok_or(VmError::Type)
            }));
            let context = step_try!(heap.no_gc(|nogc, heap| {
                let ValueRef::Object(obj) = stack.callable_slot(&meta).inner().value_ref(nogc)
                else {
                    return Err(VmError::Type);
                };
                obj.as_ref()
                    .closure_context(nogc, heap)
                    .map(|c| c.into_tagged().erase())
                    .ok_or(VmError::Type)
            }));
            cache.spill_acc(*acc);
            let obj = state.handle_scope(|scope| {
                let map = scope
                    .create_handle(heap.known().function_map.as_tagged())
                    .expect("function map is strong");
                heap.allocate_object(
                    &scope,
                    ObjectSlotsInit {
                        map,
                        values: &[info, context],
                        elements: heap.known().empty_fixed_array.erase(),
                        length: 0,
                    },
                )
            });
            let _ = cache.take_acc();
            *acc = obj.erase();
            Step::Next
        }
        Opcode::LoadGlobal => {
            let global = heap.known().global_object.value();
            let outcome = step_try!(heap.no_gc(|nogc, heap| {
                let name = callable_name(nogc, heap, stack, &meta, ops.idx(0));
                load_outcome(nogc, heap, global, name)
            }));
            match outcome {
                LoadOutcome::Value(v) => *acc = v,
                LoadOutcome::Getter(getter) => {
                    let called =
                        step_try!(call_value(heap, stack, cache, meta, pc, getter, &[global]));
                    if !called {
                        *acc = heap.known().undefined.value();
                    }
                }
            }
            // TODO: full semantics: lookup the script-context table first
            // (lexical globals), throw ReferenceError on unresolved loads;
            // the global-object property path is the only one implemented
            Step::Next
        }
        Opcode::StoreGlobal => {
            let global = heap.known().global_object.value();
            let outcome = step_try!(heap.no_gc(|nogc, heap| {
                let name = callable_name(nogc, heap, stack, &meta, ops.idx(0));
                global.store_lookup(nogc, heap, name, *acc, StoreSemantics::WriteThrough)
            }));
            step_try!(apply_store_outcome(
                heap, state, stack, cache, meta, pc, global, *acc, outcome,
            ));
            Step::Next
        }
        Opcode::CreateFunctionContext | Opcode::CreateBlockContext => {
            let count = ops.uimm(0) as usize;
            let hole = heap.known().void.value();
            let values = vec![hole; count];
            let outer = step_try!(frame_context(heap, stack, &meta));
            cache.spill_acc(*acc);
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .create_handle(unsafe { Tagged::<Context>::from_value_unchecked(outer) })
                    .expect("frame context is strong");
                let slots = heap.allocate_handle::<FixedArray>(&values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                })
            });
            let _ = cache.take_acc();
            *acc = ctx.erase();
            Step::Next
        }
        Opcode::CreateCatchContext => {
            let exception = stack.reg(&meta, ops.reg(0));
            let outer = step_try!(frame_context(heap, stack, &meta));
            cache.spill_acc(*acc);
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .create_handle(unsafe { Tagged::<Context>::from_value_unchecked(outer) })
                    .expect("frame context is strong");
                let slots = heap.allocate_handle::<FixedArray>(&[exception], &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                })
            });
            let _ = cache.take_acc();
            *acc = ctx.erase();
            Step::Next
        }
        Opcode::PushContext => {
            let old = step_try!(frame_context(heap, stack, &meta));
            stack.set_reg(&meta, ops.reg(0), old);
            step_try!(set_frame_context(heap, stack, &meta, *acc));
            Step::Next
        }
        Opcode::PopContext => {
            let context = stack.reg(&meta, ops.reg(0));
            step_try!(set_frame_context(heap, stack, &meta, context));
            Step::Next
        }
        Opcode::ThrowReferenceErrorIfHole => {
            if *acc == heap.known().void.value() {
                return Step::Error(VmError::Reference);
            }
            Step::Next
        }
        Opcode::LoadContextSlot => {
            let depth = ops.uimm(1);
            let v = step_try!(heap.no_gc(|nogc, heap| {
                let ValueRef::Object(obj) = stack.callable_slot(&meta).inner().value_ref(nogc)
                else {
                    return Err(VmError::Type);
                };
                let mut context = obj
                    .as_ref()
                    .closure_context(nogc, heap)
                    .ok_or(VmError::Type)?;
                for _ in 0..depth {
                    context = context
                        .as_ref()
                        .outer
                        .heap_ref(nogc, heap)
                        .ok_or(VmError::Type)?;
                }
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
                let mut context = obj
                    .as_ref()
                    .closure_context(nogc, heap)
                    .ok_or(VmError::Type)?;
                for _ in 0..ops.uimm(1) {
                    context = context
                        .as_ref()
                        .outer
                        .heap_ref(nogc, heap)
                        .ok_or(VmError::Type)?;
                }
                let host = context.clone().into_tagged().erase();
                context
                    .slots
                    .heap_ref(nogc)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .set(heap, host, Tagged::from_value(*acc));
                Ok(())
            }));
            Step::Next
        }
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}
