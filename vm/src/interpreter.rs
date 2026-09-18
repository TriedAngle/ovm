use crate::proxy::Proxy;
use bytecode::{Opcode, Operands, decode, jump_target};

use crate::{
    CallTarget, CallableInfoObject, Coercion, Compare, Context, ContextInit, ContextState, Convert,
    DenseString, Errors, FixedArray, FrameMeta, Handle, HandleSlice, Heap, Hint, Key, LoadOutcome,
    Lookup, NativeContext, NativeIndex, Object, PropertyDescriptor, Runtime, ScopeInfo, SlotName,
    Smi, Stack, StackCache, StoreOutcome, StoreSemantics, Tagged, VM, Value, VmError,
};

enum Called<'a> {
    /// A bytecode frame was pushed; its `Return` produces the value.
    Frame,
    /// The callee is not callable.
    NotCallable,
    /// The call ran to completion eagerly (proxy `apply`): the value.
    Immediate(Tagged<'a, Value>),
    /// The call threw; the pending exception is set.
    Threw,
}

enum Unwind<'a> {
    /// A handler was found: the accumulator must become the exception.
    Caught(Tagged<'a, Value>),
    /// No handler in this run: the exception escapes with the exception
    /// sentinel in the accumulator.
    Escaped,
}

enum Step<'a> {
    Next,
    Return,
    Throw(Tagged<'a, Value>),
    PendingThrow,
    Error(VmError),
}

pub fn execute<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'_, Value>>,
) -> Result<Tagged<'a, Value>, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;

    let saved_top = stack.top();
    let was_active = cache.is_active();
    if was_active {
        stack.suspend_frame(cache.frame_meta());
    }
    let base_depth = stack.frame_depth();

    state.handle_scope(|scope| {
        let result = start(vm, heap, state, callable, args, new_target, base_depth)?;
        let rooted = scope.handle(result);

        stack.truncate_frames(base_depth);
        if was_active {
            let outer = stack.pop_frame(saved_top).expect("suspended caller frame");
            cache.load(stack, outer, heap);
        } else {
            stack.set_top(saved_top);
            cache.deactivate();
        }
        Ok(rooted.as_tagged(heap))
    })
}

fn start<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'_, Value>>,
    base_depth: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    if Proxy::is_proxy(heap, callable.as_tagged(heap).erase()) {
        return state.handle_scope(|scope| {
            let result = match new_target {
                None => Proxy::apply(vm, heap, state, callable.erase(), args),
                Some(nt) => {
                    let real: Vec<Tagged<'_, Value>> =
                        args.iter().map(|h| h.as_tagged(heap)).skip(1).collect();
                    let staged = scope.stage(&real);
                    Proxy::construct(vm, heap, state, callable.erase(), staged, nt)
                }
            };
            match result {
                Ok(Coercion::Value(v)) => Ok(scope.handle(v).as_tagged(heap)),
                Ok(Coercion::Threw) => Ok(heap.known().exception.as_tagged(heap).erase()),
                Err(e) => Err(e),
            }
        });
    }

    match Object::call_target(heap, callable.as_tagged(heap).erase()) {
        None => Err(VmError::Type),
        Some(CallTarget::Native(idx)) => {
            let f = vm.native(NativeIndex(idx));
            let (saved_top, fargs) = state.stack.stage_args(args)?;
            let mut nctx = NativeContext::with_new_target(vm, heap, state, new_target);
            let result = f(&mut nctx, fargs);
            state.stack.set_top(saved_top);

            result.map(|v| unsafe { Tagged::from_value_unchecked(v) })
        }
        Some(CallTarget::Bytecode(target, register_count, kind)) => {
            if new_target.is_none() && kind.is_class_constructor() {
                return Err(VmError::Type);
            }
            // derived constructors receive TheHole as their receiver: `this`
            // stays uninitialized until super() binds it (ES 10.2.2)
            let stack = &state.stack;
            // one shared anchor covers every read feeding the frame push;
            // the push itself never allocates, so the anchor may span it
            let frame = {
                let heap: &Heap = heap;
                let context = target
                    .as_ref()
                    .closure_context(heap)
                    .expect("callable must have a closure context")
                    .erase();
                let formal_min = target
                    .as_ref()
                    .callable_info(heap)
                    .map(|info| info.formal_parameter_count() + 1)
                    .unwrap_or(1);
                let new_target_value = match &new_target {
                    Some(nt) => nt.as_tagged(heap).erase(),
                    None => heap.known().undefined.as_tagged(heap).erase(),
                };
                stack.push_initial_frame(
                    target.erase(),
                    register_count,
                    context,
                    new_target_value,
                    args,
                    formal_min,
                )?
            };
            state.cache.enter(stack, frame, heap);
            dispatch(vm, heap, state, base_depth)
        }
    }
}

fn call_value<'a>(
    vm: &VM,
    state: &ContextState,
    heap: &'a mut Heap,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    handler_pc: usize,
    f: Handle<'_, Value>,
    args: HandleSlice<'_>,
) -> Result<Called<'a>, VmError> {
    if Proxy::is_proxy(heap, f.as_tagged(heap)) {
        return match Proxy::apply(vm, heap, state, f, args)? {
            Coercion::Threw => Ok(Called::Threw),
            Coercion::Value(v) => Ok(Called::Immediate(v)),
        };
    }
    // TODO: native getters/setters invoke in place instead of pushing a frame
    let Some(CallTarget::Bytecode(target, register_count, kind)) =
        Object::call_target(heap, f.as_tagged(heap))
    else {
        return Ok(Called::NotCallable);
    };
    if kind.is_class_constructor() {
        return Err(VmError::Type);
    }

    let context = target
        .as_ref()
        .closure_context(heap)
        .expect("callable must have a closure context")
        .erase();
    let formal_min = target
        .as_ref()
        .callable_info(heap)
        .map(|info| info.formal_parameter_count() + 1)
        .unwrap_or(1);
    let target = target.erase();
    let undefined = heap.known().undefined.as_tagged(heap).erase();
    let callee = stack.push_frame_with_args(
        meta,
        handler_pc,
        target,
        register_count,
        context,
        args,
        undefined,
        formal_min,
    )?;
    cache.load(stack, callee, heap);
    Ok(Called::Frame)
}

#[cold]
#[inline(never)]
fn exception_dispatch<'a>(
    heap: &'a mut Heap,
    state: &ContextState,
    base_depth: usize,
    mut pc: usize,
) -> Unwind<'a> {
    let stack = &state.stack;
    let cache = &state.cache;
    loop {
        let handled = 'handled: {
            let Some(obj) = stack
                .callable_slot(&cache.frame_meta())
                .read(heap)
                .as_heap_object()
            else {
                break 'handled None;
            };
            let Some(info) = obj.as_ref().callable_info(heap) else {
                break 'handled None;
            };
            let Some(handlers) = info.handlers.heap_ref(heap) else {
                break 'handled None;
            };
            handlers.lookup(pc)
        };
        if let Some(handler_pc) = handled {
            let ex = state
                .take_pending_exception_tagged(heap)
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
fn raise<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    base_depth: usize,
    err: VmError,
    pc: usize,
) -> Unwind<'a> {
    let ex =
        Errors::from_vm_error(vm, heap, state, err).expect("error materialization must not fail");
    state.set_pending_exception(ex);
    exception_dispatch(heap, state, base_depth, pc)
}

macro_rules! step_try {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => return Step::Error(err),
        }
    };
}

fn dispatch<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    base_depth: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let cache = &state.cache;
    let acc = cache.acc_mut();
    let mut trace = 0;
    loop {
        let pc = cache.pc();
        let (op, ops, next_pc) = decode(cache.code_ref(heap).as_slice(), pc);
        if std::env::var("OVM_TRACE").is_ok() {
            trace += 1;
            if trace > 200000 {
                panic!("trace limit");
            }
            eprintln!(
                "pc={pc} base={} rc={} op={op:?} acc={:?}",
                cache.frame_meta().base,
                cache.frame_meta().register_count,
                *acc,
            );
        }
        cache.set_pc(next_pc);
        let meta = cache.frame_meta();

        let result = step(vm, heap, state, base_depth, op, ops, meta, pc);
        match result {
            Step::Next => {}
            Step::Return => return Ok(acc.read(heap)),
            Step::Throw(v) => {
                state.set_pending_exception(v);
                match exception_dispatch(heap, state, base_depth, pc) {
                    Unwind::Caught(ex) => acc.store(ex),
                    Unwind::Escaped => {
                        return Ok(heap.known().exception.as_tagged(heap).erase());
                    }
                }
            }
            Step::PendingThrow => match exception_dispatch(heap, state, base_depth, pc) {
                Unwind::Caught(ex) => acc.store(ex),
                Unwind::Escaped => return Ok(heap.known().exception.as_tagged(heap).erase()),
            },
            Step::Error(err) => match raise(vm, heap, state, base_depth, err, pc) {
                Unwind::Caught(ex) => acc.store(ex),
                Unwind::Escaped => return Ok(heap.known().exception.as_tagged(heap).erase()),
            },
        }
    }
}

fn apply_store_outcome(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    pc: usize,
    receiver: Handle<'_, Value>,
    outcome: StoreOutcome<'_>,
) -> Result<(), VmError> {
    match outcome {
        StoreOutcome::Transition {
            receiver: recv,
            name,
        } => {
            state.handle_scope(|scope| {
                let value = scope.handle(cache.acc(heap));
                Object::add_own_property(heap, &scope, recv, name, PropertyDescriptor::data(value))
                    // TODO(strict-mode): a false result must throw in strict code;
                    // the current store path preserves its existing sloppy result.
                    .map(|_| ())
            })
        }
        StoreOutcome::CallSetter { setter } => state.handle_scope(|scope| {
            let value = cache.acc(heap);
            let args = scope.stage(&[receiver.as_tagged(heap).erase(), value]);
            call_value(vm, state, heap, stack, cache, meta, pc, setter, args)?;
            Ok(())
        }),
        StoreOutcome::Done => Ok(()),
    }
}

fn step<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    base_depth: usize,
    op: Opcode,
    ops: Operands,
    meta: FrameMeta,
    pc: usize,
) -> Step<'a> {
    let stack = &state.stack;
    let cache = &state.cache;
    let mut acc = cache.acc_mut();

    match op {
        Opcode::Return => {
            if stack.frame_depth() == base_depth {
                return Step::Return;
            }
            let caller = stack
                .pop_frame(meta.base)
                .expect("suspended frame above base depth");
            cache.load(stack, caller, heap);
            Step::Next
        }
        Opcode::Load => {
            acc.store(stack.reg(heap, &meta, ops.reg(0)));
            Step::Next
        }
        Opcode::LoadSmi => {
            *acc = Smi::new(ops.imm(0) as i64).encode();
            Step::Next
        }
        Opcode::LoadConstant => {
            acc.store(cache.constants_ref(heap).at(heap, ops.idx(0)));
            Step::Next
        }
        Opcode::LoadZero => {
            *acc = Smi::new(0).encode();
            Step::Next
        }
        Opcode::LoadUndefined => {
            acc.store(heap.known().undefined.as_tagged(heap));
            Step::Next
        }
        Opcode::LoadNull => {
            acc.store(heap.known().null.as_tagged(heap));
            Step::Next
        }
        Opcode::LoadTrue => {
            acc.store(heap.known().true_object.as_tagged(heap));
            Step::Next
        }
        Opcode::LoadFalse => {
            acc.store(heap.known().false_object.as_tagged(heap));
            Step::Next
        }
        Opcode::LoadHole => {
            acc.store(heap.known().the_hole.as_tagged(heap));
            Step::Next
        }
        Opcode::LoadGlobal | Opcode::LoadGlobalNoThrow => {
            state.handle_scope(|scope| -> Step<'_> {
                let global = heap.known().global_object;
                let name = stack
                    .callable(heap, &meta)
                    .as_ref()
                    .constant_slot_name(heap, ops.idx(0));

                match global.lookup(heap, name) {
                    Lookup::Data { slot, .. } => acc.store(slot.get(heap)),
                    Lookup::Accessor { pair, .. } => {
                        let getter = scope.handle(pair.get.get(heap));
                        if getter.as_tagged(heap) == heap.known().undefined.as_tagged(heap) {
                            acc.store(heap.known().undefined.as_tagged(heap));
                        } else {
                            let args = scope.stage(&[global.as_tagged(heap).erase()]);
                            match step_try!(call_value(
                                vm, state, heap, stack, cache, meta, pc, getter, args
                            )) {
                                Called::Frame => {}
                                Called::NotCallable => {
                                    acc.store(heap.known().undefined.as_tagged(heap));
                                }
                                Called::Immediate(v) => acc.store(v),
                                Called::Threw => return Step::PendingThrow,
                            }
                        }
                    }
                    Lookup::NotFound => {
                        if op == Opcode::LoadGlobal {
                            // unresolvable reference: GetValue throws ReferenceError
                            return Step::Error(VmError::Reference);
                        }
                        acc.store(heap.known().undefined.as_tagged(heap));
                    }
                }
                // TODO: full semantics: lookup the script-context table first
                // (lexical globals), throw ReferenceError on unresolved loads;
                // the global-object property path is the only one implemented
                Step::Next
            })
        }
        Opcode::LoadNamedProperty => state.handle_scope(|scope| -> Step<'_> {
            let receiver = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let name = stack
                .callable(heap, &meta)
                .as_ref()
                .constant_slot_name(heap, ops.idx(1));

            if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
                let name = scope.handle(name.erase());
                return match step_try!(Proxy::get(vm, heap, state, receiver, receiver, name)) {
                    Coercion::Threw => Step::PendingThrow,
                    Coercion::Value(v) => {
                        acc.store(v);
                        Step::Next
                    }
                };
            }

            let outcome = step_try!(Lookup::load_outcome(heap, receiver.as_tagged(heap), name));
            match outcome {
                LoadOutcome::Value(v) => acc.store(v),
                LoadOutcome::Getter(getter) => {
                    let args = scope.stage(&[receiver.as_tagged(heap).erase()]);
                    let getter = scope.handle(getter);
                    match step_try!(call_value(
                        vm, state, heap, stack, cache, meta, pc, getter, args
                    )) {
                        Called::Frame => {}
                        Called::NotCallable => {
                            acc.store(heap.known().undefined.as_tagged(heap));
                        }
                        Called::Immediate(v) => acc.store(v),
                        Called::Threw => return Step::PendingThrow,
                    }
                }
            }
            Step::Next
        }),
        Opcode::LoadKeyedProperty => state.handle_scope(|scope| -> Step<'_> {
            let receiver = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let raw_key = scope.handle(acc.read(heap));
            let Some(key) = step_try!(Runtime::to_property_key(vm, heap, state, raw_key)) else {
                return Step::PendingThrow;
            };
            let key = scope.handle(key);

            if let Some(unit) = DenseString::index_element(heap, &scope, receiver, key) {
                acc.store(unit);
                return Step::Next;
            }

            if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
                return match step_try!(Proxy::get(vm, heap, state, receiver, receiver, key.erase()))
                {
                    Coercion::Threw => Step::PendingThrow,
                    Coercion::Value(v) => {
                        acc.store(v);
                        Step::Next
                    }
                };
            }

            let outcome = step_try!(Lookup::load_outcome_keyed(
                heap,
                receiver.as_tagged(heap),
                key.as_tagged(heap)
            ));
            match outcome {
                LoadOutcome::Value(v) => acc.store(v),
                LoadOutcome::Getter(getter) => {
                    let getter = scope.handle(getter);
                    let args = scope.stage(&[receiver.as_tagged(heap).erase()]);
                    match step_try!(call_value(
                        vm, state, heap, stack, cache, meta, pc, getter, args
                    )) {
                        Called::Frame => {}
                        Called::NotCallable => acc.store(heap.known().undefined.as_tagged(heap)),
                        Called::Immediate(v) => acc.store(v),
                        Called::Threw => return Step::PendingThrow,
                    }
                }
            }
            Step::Next
        }),
        Opcode::LoadNewTarget => {
            acc.store(stack.new_target_slot(&meta).read(heap));
            Step::Next
        }
        Opcode::Store => {
            stack.set_reg(&meta, ops.reg(0), acc.read(heap));
            Step::Next
        }
        Opcode::StoreGlobal => state.handle_scope(|scope| -> Step<'_> {
            let outcome = step_try!({
                let name = stack
                    .callable(heap, &meta)
                    .as_ref()
                    .constant_slot_name(heap, ops.idx(0));
                heap.known().global_object.store_lookup(
                    heap,
                    &scope,
                    name,
                    acc.read(heap),
                    StoreSemantics::WriteThrough,
                )
            });
            step_try!(apply_store_outcome(
                vm,
                heap,
                state,
                stack,
                cache,
                meta,
                pc,
                heap.known().global_object.erase(),
                outcome,
            ));
            Step::Next
        }),
        Opcode::StoreNamedProperty | Opcode::StoreNamedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreNamedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            state.handle_scope(|scope| -> Step<'_> {
                let receiver = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
                let name = stack
                    .callable(heap, &meta)
                    .as_ref()
                    .constant_slot_name(heap, ops.idx(1));

                if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
                    let name = scope.handle(name.erase());
                    let value = scope.handle(acc.read(heap));
                    return match step_try!(Proxy::set(
                        vm, heap, state, receiver, name, value, receiver
                    )) {
                        Coercion::Threw => Step::PendingThrow,
                        Coercion::Value(_) => Step::Next,
                    };
                }

                let outcome =
                    step_try!(receiver.store_lookup(heap, &scope, name, acc.read(heap), semantics));
                let receiver = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
                step_try!(apply_store_outcome(
                    vm, heap, state, stack, cache, meta, pc, receiver, outcome,
                ));
                Step::Next
            })
        }
        Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreKeyedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            state.handle_scope(|scope| -> Step<'_> {
                let receiver = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
                let raw = scope.handle(stack.reg(heap, &meta, ops.reg(1)));
                let Some(key) = step_try!(Runtime::to_property_key(vm, heap, state, raw)) else {
                    return Step::PendingThrow;
                };
                let key = scope.handle(key);

                if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
                    return match step_try!(Proxy::set(
                        vm,
                        heap,
                        state,
                        receiver,
                        key.erase(),
                        scope.handle(acc.read(heap)),
                        receiver,
                    )) {
                        Coercion::Threw => Step::PendingThrow,
                        Coercion::Value(_) => Step::Next,
                    };
                }

                let name: Handle<'_, SlotName> =
                    match step_try!(Lookup::classify_key(heap, key.as_tagged(heap).erase())) {
                        Key::Element(i) => {
                            // arrays store into the backing elements
                            if receiver
                                .as_tagged(heap)
                                .as_heap_object()
                                .is_some_and(|obj| obj.as_ref().is_array(heap))
                            {
                                let receiver = scope
                                    .cast::<Object>(receiver.as_tagged(heap))
                                    .expect("array receiver is an object");
                                let value = scope.handle(acc.read(heap));
                                step_try!(Object::store_array_element(
                                    heap, &scope, &receiver, i, &value
                                ));
                                return Step::Next;
                            }
                            // numeric property on a non-array receiver
                            scope.handle(Smi::new(i as i64))
                        }
                        Key::Name(key) => scope.handle(key),
                    };

                let outcome = step_try!(receiver.as_tagged(heap).store_lookup(
                    heap,
                    &scope,
                    name.as_tagged(heap),
                    acc.read(heap),
                    semantics,
                ));
                let receiver = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
                step_try!(apply_store_outcome(
                    vm, heap, state, stack, cache, meta, pc, receiver, outcome,
                ));
                Step::Next
            })
        }
        Opcode::Move => {
            stack.set_reg(&meta, ops.reg(0), stack.reg(heap, &meta, ops.reg(1)));
            Step::Next
        }
        Opcode::LoadContextSlot => {
            let depth = ops.uimm(1);
            let v = {
                let mut context = match stack.context_slot(&meta).read(heap).get_as::<Context>() {
                    Some(context) => context,
                    None => return Step::Error(VmError::Type),
                };
                for _ in 0..depth {
                    context = match context.as_ref().outer.heap_ref(heap) {
                        Some(context) => context,
                        None => return Step::Error(VmError::Type),
                    };
                }
                context
                    .slots
                    .heap_ref(heap)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .inner()
            };
            *acc = v;
            Step::Next
        }
        Opcode::StoreContextSlot => {
            let mut context = match stack.context_slot(&meta).read(heap).get_as::<Context>() {
                Some(context) => context,
                None => return Step::Error(VmError::Type),
            };
            for _ in 0..ops.uimm(1) {
                context = match context.as_ref().outer.heap_ref(heap) {
                    Some(context) => context,
                    None => return Step::Error(VmError::Type),
                };
            }
            // Safety: fresh anchored slot read.
            let host = context.clone().into_tagged().raw();
            context
                .slots
                .heap_ref(heap)
                .as_ref()
                .element_slot(ops.idx(0))
                .set(heap, host, acc.read(heap));
            Step::Next
        }
        Opcode::CreateFunctionContext => {
            // constants[idx] is the scope's shared ScopeInfo (its `names`
            // array is parallel to the context's slots)
            let count = step_try!({
                cache
                    .constants_ref(heap)
                    .at(heap, ops.idx(0))
                    .get_as::<ScopeInfo>()
                    .map(|r| r.as_ref().names.heap_ref(heap).len())
                    .ok_or(VmError::Type)
            });
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .cast::<Context>(stack.context_slot(&meta).read(heap))
                    .expect("frame context slot holds a Context");
                let values =
                    scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
                let scope_info = scope
                    .cast::<ScopeInfo>(cache.constants_ref(heap).at(heap, ops.idx(0)))
                    .expect("constants slot holds a ScopeInfo");
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info,
                })
            });
            acc.store(ctx);
            Step::Next
        }
        Opcode::CreateBlockContext => {
            let count = ops.uimm(0) as usize;
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .cast::<Context>(stack.context_slot(&meta).read(heap))
                    .expect("frame context slot holds a Context");
                let values =
                    scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info: heap.known().empty_scope_info,
                })
                .raw()
            });
            *acc = ctx;
            Step::Next
        }
        Opcode::PushContext => {
            let old = stack.context_slot(&meta).read(heap);
            stack.set_reg(&meta, ops.reg(0), old);
            let context = acc.read(heap);
            if context.get_as::<Context>().is_none() {
                return Step::Error(VmError::Type);
            }
            stack.context_slot(&meta).store(context);
            Step::Next
        }
        Opcode::PopContext => {
            let context = stack.reg(heap, &meta, ops.reg(0));
            if context.get_as::<Context>().is_none() {
                return Step::Error(VmError::Type);
            }
            stack.context_slot(&meta).store(context);
            Step::Next
        }
        Opcode::ThrowReferenceErrorIfHole => {
            if *acc == heap.known().the_hole.as_tagged(heap) {
                return Step::Error(VmError::Reference);
            }
            Step::Next
        }
        Opcode::LoadContext => {
            acc.store(stack.context_slot(&meta).read(heap));
            Step::Next
        }
        // TODO: feedback vectors and separation once they are there
        Opcode::Call | Opcode::CallNoFeedback => {
            // TODO(strict-mode): ordinary sloppy functions still need nullish
            // receiver substitution and primitive receiver boxing.
            let count = ops.reg_count(2);
            // a callable proxy dispatches through its `apply` trap (or
            // a nested call of the target)
            if Proxy::is_proxy(heap, stack.reg(heap, &meta, ops.reg(0))) {
                return match state.handle_scope(|scope| {
                    // staged: trap lookups may run user getters
                    let callee = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
                    let staged = stack.args(&meta, ops.reg_list(1), count);
                    Proxy::apply(vm, heap, state, callee, staged)
                }) {
                    Ok(Coercion::Threw) => Step::PendingThrow,
                    Ok(Coercion::Value(v)) => {
                        acc.store(v);
                        Step::Next
                    }
                    Err(err) => Step::Error(err),
                };
            }
            match Object::call_target(heap, stack.reg(heap, &meta, ops.reg(0))) {
                None => Step::Error(VmError::Type),
                Some(CallTarget::Native(idx)) => {
                    let f = vm.native(NativeIndex(idx));
                    let mut nctx = NativeContext::new(vm, heap, state);
                    let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                    match result {
                        // Safety: old-gen singleton word.
                        Ok(v) if v == heap.known().exception.as_tagged(heap) => Step::PendingThrow,
                        Ok(v) => {
                            *acc = v;
                            Step::Next
                        }
                        Err(err) => Step::Error(err),
                    }
                }
                Some(CallTarget::Bytecode(callee, register_count, kind)) => {
                    if kind.is_class_constructor() {
                        return Step::Error(VmError::Type);
                    }
                    // one shared anchor covers every read feeding the push
                    let frame = {
                        let context = callee
                            .as_ref()
                            .closure_context(heap)
                            .expect("callable must have a closure context")
                            .erase();
                        let formal_min = callee
                            .as_ref()
                            .callable_info(heap)
                            .map(|info| info.formal_parameter_count() + 1)
                            .unwrap_or(1);
                        let callee = callee.erase();
                        let undefined = heap.known().undefined.as_tagged(heap).erase();
                        step_try!(stack.push_frame(
                            meta,
                            pc,
                            callee,
                            register_count,
                            context,
                            ops.reg_list(1),
                            count,
                            undefined,
                            formal_min,
                        ))
                    };
                    cache.load(stack, frame, heap);
                    Step::Next
                }
            }
        }
        Opcode::CallRuntime => {
            // operand 0 is a RuntimeFn discriminant: the fixed
            // runtime-helper table (vm::natives::runtime_fn) registered at
            // indices 0..RuntimeFn::COUNT
            let f = vm.native(NativeIndex(ops.idx(0)));
            let count = ops.reg_count(2);
            let mut nctx = NativeContext::new(vm, heap, state);
            let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
            match result {
                // Safety: old-gen singleton word.
                Ok(v) if v == heap.known().exception.as_tagged(heap) => Step::PendingThrow,
                Ok(v) => {
                    *acc = v;
                    Step::Next
                }
                Err(err) => Step::Error(err),
            }
        }
        Opcode::Construct => state.handle_scope(|scope| -> Step<'_> {
            let callee = scope.handle(stack.reg(heap, &meta, ops.reg(0)));

            // ES 9.2.2 [[Construct]]: the callee's map decides whether it is
            // constructible and whether it is a derived class (whose `this`
            // is bound by super() and starts as TheHole).
            let Some(obj) = callee.as_tagged(heap).as_heap_object() else {
                return Step::Error(VmError::Type);
            };
            let kind = obj.as_ref().header.map.heap_ref(heap).kind();
            if !kind.is_constructor() {
                return Step::Error(VmError::Type);
            }
            let derived = kind.is_class_constructor()
                && obj
                    .as_ref()
                    .callable_info(heap)
                    .is_some_and(|info| info.function_kind().is_derived_class_constructor());
            let count = ops.reg_count(2);

            // a constructor proxy dispatches through its `construct` trap
            if Proxy::is_proxy(heap, callee.as_tagged(heap)) {
                let staged = stack.args(&meta, ops.reg_list(1), count);
                return match step_try!(Proxy::construct(vm, heap, state, callee, staged, callee)) {
                    Coercion::Threw => Step::PendingThrow,
                    Coercion::Value(v) => {
                        acc.store(v);
                        Step::Next
                    }
                };
            }

            let callee = scope
                .cast::<Object>(callee.as_tagged(heap))
                .expect("constructible callee is an object");
            let receiver = if derived {
                scope.handle(heap.known().the_hole.as_tagged(heap).erase())
            } else {
                match Runtime::create_construct_receiver(vm, heap, state, callee) {
                    Ok(Some(r)) => scope.handle(r),
                    Ok(None) => return Step::PendingThrow,
                    Err(err) => return Step::Error(err),
                }
            };

            // the register list holds only arguments; the synthesized
            // receiver is prepended
            let mut args: Vec<Tagged<'_, Value>> = Vec::with_capacity(count + 1);
            args.push(receiver.as_tagged(heap).erase());
            args.extend(
                stack
                    .args(&meta, ops.reg_list(1), count)
                    .iter()
                    .map(|h| h.as_tagged(heap)),
            );
            let staged = scope.stage(&args);

            let callee = callee.erase();
            let result = step_try!(
                NativeContext::new(vm, heap, state).call_construct_rooted(callee, callee, staged)
            );
            if result == heap.known().exception.as_tagged(heap) {
                return Step::PendingThrow;
            }
            *acc = if Convert::is_primitive(heap, unsafe { result.assume_valid(heap) }) {
                if derived {
                    // a derived constructor returned a primitive (ES 9.2.2.1)
                    return Step::Error(VmError::Type);
                }
                receiver.as_tagged(heap).raw()
            } else {
                result
            };
            Step::Next
        }),
        Opcode::CreateEmptyObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().object_initial_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY).raw()
            });
            *acc = obj;
            Step::Next
        }
        Opcode::CreateEmptyArrayLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().js_array_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY).raw()
            });
            *acc = obj;
            Step::Next
        }
        Opcode::CreateClosure => state.handle_scope(|scope| -> Step<'_> {
            let Some(info) =
                scope.cast::<CallableInfoObject>(cache.constants_ref(heap).at(heap, ops.idx(0)))
            else {
                return Step::Error(VmError::Type);
            };
            let context = scope
                .cast::<Context>(stack.context_slot(&meta).read(heap))
                .expect("frame context slot holds a Context");
            let obj = step_try!(Runtime::create_closure(heap, &scope, info, context));
            acc.store(obj);
            Step::Next
        }),
        Opcode::LoadCurrentClosure => {
            acc.store(stack.callable_slot(&meta).read(heap));
            Step::Next
        }
        Opcode::Add => {
            let other_reg = ops.reg(0);
            let other = stack.reg(heap, &meta, other_reg);
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && let Some(r) = a.checked_add(b)
                && Smi::in_range(r)
            {
                *acc = Smi::new(r).encode();
                return Step::Next;
            }
            state.handle_scope(|scope| {
                let lhs = scope.handle(acc.read(heap));
                let lhs =
                    match step_try!(Runtime::to_primitive(vm, heap, state, lhs, Hint::Default)) {
                        Coercion::Threw => return Step::PendingThrow,
                        Coercion::Value(v) => scope.handle(v),
                    };
                let rhs = scope.handle(stack.reg(heap, &meta, other_reg));
                let rhs =
                    match step_try!(Runtime::to_primitive(vm, heap, state, rhs, Hint::Default)) {
                        Coercion::Threw => return Step::PendingThrow,
                        Coercion::Value(v) => scope.handle(v),
                    };
                let is_string = (
                    lhs.as_tagged(heap).get_as::<DenseString>().is_some(),
                    rhs.as_tagged(heap).get_as::<DenseString>().is_some(),
                );
                if is_string.0 || is_string.1 {
                    let s = step_try!((|| -> Result<Value, VmError> {
                        let a = scope.handle(Convert::to_string(heap, &scope, lhs)?);
                        let b = scope.handle(Convert::to_string(heap, &scope, rhs)?);
                        Ok(DenseString::concat(heap, &scope, a, b)
                            .as_tagged(heap)
                            .raw())
                    })());
                    *acc = s;
                } else {
                    let a = step_try!(Convert::to_number(heap, lhs.as_tagged(heap)));
                    let b = step_try!(Convert::to_number(heap, rhs.as_tagged(heap)));
                    // IEEE `-0 + -0` yields +0; the spec demands -0
                    let r = a + b;
                    let r = if r == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                        -0.0
                    } else {
                        r
                    };
                    acc.store(heap.new_number(&scope, r));
                }
                Step::Next
            })
        }
        Opcode::Sub => state.handle_scope(|scope| -> Step<'_> {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && let Some(r) = a.checked_sub(b)
                && Smi::in_range(r)
            {
                *acc = Smi::new(r).encode();
                return Step::Next;
            }
            let a = scope.handle(acc.read(heap));
            let b = scope.handle(other);
            let v = step_try!(Runtime::numeric_op(vm, heap, state, a, b, |a, b| a - b));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            *acc = v;
            Step::Next
        }),
        Opcode::Mul => state.handle_scope(|scope| -> Step<'_> {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && let Some(r) = a.checked_mul(b)
                && Smi::in_range(r)
            {
                *acc = Smi::new(r).encode();
                return Step::Next;
            }
            let a = scope.handle(acc.read(heap));
            let b = scope.handle(other);
            let v = step_try!(Runtime::numeric_op(vm, heap, state, a, b, |a, b| a * b));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            *acc = v;
            Step::Next
        }),
        Opcode::Div => state.handle_scope(|scope| -> Step<'_> {
            // JS division is IEEE double division: 7/2 = 3.5, x/0 = ±Infinity
            // or NaN, MIN/-1 overflows to a double.
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && b != 0
                && a % b == 0
                && let Some(r) = a.checked_div(b)
            {
                *acc = Smi::new(r).encode();
                return Step::Next;
            }
            let a = scope.handle(acc.read(heap));
            let b = scope.handle(other);
            let v = step_try!(Runtime::numeric_op(vm, heap, state, a, b, |a, b| a / b));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            *acc = v;
            Step::Next
        }),
        Opcode::Mod => state.handle_scope(|scope| -> Step<'_> {
            // JS remainder is IEEE fmod: x % 0 = NaN, signs follow the dividend.
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && b != 0
            {
                *acc = Smi::new(a % b).encode();
                return Step::Next;
            }
            let a = scope.handle(acc.read(heap));
            let b = scope.handle(other);
            let v = step_try!(Runtime::numeric_op(vm, heap, state, a, b, |a, b| a % b));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            *acc = v;
            Step::Next
        }),
        Opcode::Exp => state.handle_scope(|scope| -> Step<'_> {
            // JS exponentiation is always IEEE double math; the result only
            // needs a Smi tag when it is an in-range integer.
            let a = scope.handle(acc.read(heap));
            let b = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let v = step_try!(Runtime::numeric_op(vm, heap, state, a, b, |a, b| a.powf(b)));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            *acc = v;
            Step::Next
        }),
        Opcode::BitwiseOr => {
            // ToInt32 semantics on the (integer) smi inputs
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            *acc = Smi::new((a | b) as i64).encode();
            Step::Next
        }
        Opcode::BitwiseXor => {
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            *acc = Smi::new((a ^ b) as i64).encode();
            Step::Next
        }
        Opcode::BitwiseAnd => {
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            *acc = Smi::new((a & b) as i64).encode();
            Step::Next
        }
        Opcode::ShiftLeft => {
            // ToInt32(lhs) << (ToUint32(rhs) & 31), truncated to int32
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as u32;
            *acc = Smi::new(a.wrapping_shl(b & 31) as i64).encode();
            Step::Next
        }
        Opcode::ShiftRight => {
            // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as u32;
            *acc = Smi::new(a.wrapping_shr(b & 31) as i64).encode();
            Step::Next
        }
        Opcode::ShiftRightLogical => {
            // ToUint32(lhs) >>> (ToUint32(rhs) & 31): always non-negative
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as u32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as u32;
            *acc = Smi::new(a.wrapping_shr(b & 31) as i64).encode();
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
            if Convert::is_truthy(heap, acc.read(heap)) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfFalsy => {
            if !Convert::is_truthy(heap, acc.read(heap)) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfNotUndefined => {
            if *acc != heap.known().undefined.as_tagged(heap) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::TestReferenceEqual => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            let known = heap.known();
            acc.store(if other == *acc {
                known.true_object.as_tagged(heap)
            } else {
                known.false_object.as_tagged(heap)
            });
            Step::Next
        }
        Opcode::TestTypeof => {
            acc.store(Runtime::type_of(heap, acc.read(heap)));
            Step::Next
        }
        Opcode::Negate => {
            let acc_word = *acc;
            if let Some(v) = acc_word.to_i64() {
                *acc = if v == 0 {
                    state.handle_scope(|scope| heap.new_number(&scope, -0.0).raw())
                } else if v == Smi::MIN {
                    state.handle_scope(|scope| heap.new_number(&scope, -(v as f64)).raw())
                } else {
                    Smi::new(-v).encode()
                };
            } else {
                let n = state.handle_scope(|scope| {
                    let acc = scope.handle(acc.read(heap));
                    Runtime::to_numeric(vm, heap, state, acc)
                });
                let n = step_try!(n);
                let Some(n) = n else {
                    return Step::PendingThrow;
                };
                *acc = state.handle_scope(|scope| {
                    let r = -n;
                    // preserve -0.0: `-0` must not fold into Smi 0
                    if r == 0.0 && r.is_sign_negative() {
                        heap.new_number(&scope, -0.0).raw()
                    } else {
                        heap.new_number(&scope, r).raw()
                    }
                });
            }
            Step::Next
        }
        Opcode::InstanceOf => {
            let acc_word = *acc;
            let other = stack.reg(heap, &meta, ops.reg(0)).raw();
            let r = step_try!(Runtime::instance_of(vm, heap, state, acc_word, other));
            let Some(r) = r else {
                return Step::PendingThrow;
            };
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::EqualStrict => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            let r = Compare::strict_equal(heap, acc.read(heap), other);
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::Equal => state.handle_scope(|scope| {
            let x = scope.handle(acc.read(heap));
            let x = match step_try!(Runtime::to_primitive(vm, heap, state, x, Hint::Default)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let y = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let y = match step_try!(Runtime::to_primitive(vm, heap, state, y, Hint::Default)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let r = step_try!(Compare::equal(heap, x.as_tagged(heap), y.as_tagged(heap)));
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }),
        Opcode::LessThan => state.handle_scope(|scope| {
            let x = scope.handle(acc.read(heap));
            let x = match step_try!(Runtime::to_primitive(vm, heap, state, x, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let y = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let y = match step_try!(Runtime::to_primitive(vm, heap, state, y, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let r = step_try!(Compare::less_than(
                heap,
                x.as_tagged(heap),
                y.as_tagged(heap)
            ));
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }),
        Opcode::LessThanOrEqual => state.handle_scope(|scope| {
            let x = scope.handle(acc.read(heap));
            let x = match step_try!(Runtime::to_primitive(vm, heap, state, x, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let y = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let y = match step_try!(Runtime::to_primitive(vm, heap, state, y, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let r = step_try!(Compare::less_than_or_equal(
                heap,
                x.as_tagged(heap),
                y.as_tagged(heap)
            ));
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }),
        Opcode::GreaterThan => state.handle_scope(|scope| {
            let x = scope.handle(acc.read(heap));
            let x = match step_try!(Runtime::to_primitive(vm, heap, state, x, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let y = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let y = match step_try!(Runtime::to_primitive(vm, heap, state, y, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let r = step_try!(Compare::greater_than(
                heap,
                x.as_tagged(heap),
                y.as_tagged(heap)
            ));
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }),
        Opcode::GreaterThanOrEqual => state.handle_scope(|scope| {
            let x = scope.handle(acc.read(heap));
            let x = match step_try!(Runtime::to_primitive(vm, heap, state, x, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let y = scope.handle(stack.reg(heap, &meta, ops.reg(0)));
            let y = match step_try!(Runtime::to_primitive(vm, heap, state, y, Hint::Number)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let r = step_try!(Compare::greater_than_or_equal(
                heap,
                x.as_tagged(heap),
                y.as_tagged(heap)
            ));
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }),
        Opcode::Throw | Opcode::ReThrow => Step::Throw(acc.read(heap)),
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}
