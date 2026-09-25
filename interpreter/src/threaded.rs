use bytecode::{Opcode, Operands, decode, jump_target};
use vm_core::proxy::Proxy;

use vm_core::ic::{Hit, InlineCache, StoreHit, StoreOutcomeKind};
use vm_core::{
    Acc, CallTarget, CallableInfoObject, Coercion, Compare, Context, ContextInit, ContextState,
    Convert, DenseString, Errors, FixedArray, FrameMeta, Handle, HandleSlice, Heap, Hint, Key,
    LoadOutcome, Lookup, Object, PropertyDescriptor, RuntimeContext, RuntimeIndex, ScopeInfo,
    SlotName, Smi, Stack, StackCache, StoreOutcome, StoreSemantics, Tagged, Termination, VM, Value,
    VmError,
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
    /// transfer to a new pc in the same frame (jump arms)
    Jump(usize),
    /// the current frame changed
    Reframe,
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
    let stack = &state.stack();
    let cache = &state.cache();

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
            cache.deactivate(heap);
        }
        Ok(rooted.as_tagged(heap))
    })
}

fn start<'b>(
    vm: &'b VM,
    heap: &'b mut Heap,
    state: &'b ContextState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'b, Value>>,
    base_depth: usize,
) -> Result<Tagged<'b, Value>, VmError> {
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
        Some(CallTarget::Runtime(idx)) => {
            let f = vm.runtime(RuntimeIndex(idx));
            let (saved_top, fargs) = state.stack().stage_args(heap, args)?;
            let nctx = RuntimeContext::with_new_target(vm, heap, state, new_target);
            let result = f(nctx, fargs);
            state.stack().set_top(saved_top);
            result
        }
        Some(CallTarget::Bytecode(target, register_count, kind)) => {
            if new_target.is_none() && kind.is_class_constructor() {
                return Err(VmError::Type);
            }
            // derived constructors receive TheHole as their receiver: `this`
            // stays uninitialized until super() binds it (ES 10.2.2)
            let stack = &state.stack();
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
                    heap,
                    target.erase(),
                    register_count,
                    context,
                    new_target_value,
                    args,
                    formal_min,
                )?
            };
            state.cache().enter(stack, frame, heap);
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
    // TODO: runtime getters/setters invoke in place instead of pushing a frame
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
        heap,
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
    let stack = &state.stack();
    let cache = &state.cache();
    // A termination is not an exception: no handler (catch or finally in
    // any frame) may observe or intercept it — unwind straight out.
    let terminated = state.termination().is_some();
    loop {
        let handled = if terminated {
            None
        } else {
            'handled: {
                let Some(obj) = stack
                    .callable_slot(&cache.frame_meta())
                    .get(heap)
                    .as_heap_object()
                else {
                    break 'handled None;
                };
                let Some(info) = obj.as_ref().callable_info(heap) else {
                    break 'handled None;
                };
                let Some(handlers) = info.handlers.get(heap) else {
                    break 'handled None;
                };
                handlers.as_ref().lookup(pc)
            }
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

/// The text of an interned `SlotName` (empty for non-string names).
fn slot_name_text(heap: &Heap, name: Handle<'_, SlotName>) -> String {
    name.as_tagged(heap)
        .erase()
        .get_as::<DenseString>()
        .map(|s| s.to_rust_string(heap))
        .unwrap_or_default()
}

fn begin_termination(heap: &Heap, state: &ContextState) -> Step<'static> {
    state.set_termination(Termination::Shutdown);
    let undefined = heap.known().undefined.as_tagged(heap);
    state.set_pending_exception(undefined);
    Step::PendingThrow
}

macro_rules! step_try {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => return Step::Error(err),
        }
    };
}

/// Which loose comparison a compare arm performs.
enum Cmp {
    Eq,
    EqStrict,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Result of a cold comparison: the boolean, or a pending throw.
enum CmpOutcome {
    Bool(bool),
    Threw,
}

impl Cmp {
    /// The comparison index encoded into `CompareJump`'s kind operand.
    fn from_kind(kind: u32) -> Self {
        match kind {
            0 => Self::Eq,
            1 => Self::EqStrict,
            2 => Self::Lt,
            3 => Self::Le,
            4 => Self::Gt,
            _ => Self::Ge,
        }
    }
}

struct Slow;

impl Slow {
    fn numeric_op<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        meta: FrameMeta,
        acc: &Acc<'_>,
        reg: i32,
        f: fn(f64, f64) -> f64,
    ) -> Step<'a> {
        let stack = &state.stack();
        state.handle_scope(|scope| {
            let a = scope.handle(acc.get(heap));
            let b = scope.handle(stack.reg(heap, &meta, reg));
            let v = step_try!(Object::numeric_op(vm, heap, state, a, b, f));
            let Some(v) = v else {
                return Step::PendingThrow;
            };
            acc.store(v);
            Step::Next
        })
    }

    /// Shared cold body of the loose compares: to_primitive, then the
    /// full `Compare` implementation. `Threw` means a coercion threw (the
    /// pending exception is set).
    fn compare_bool(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        meta: FrameMeta,
        acc: &Acc<'_>,
        reg: i32,
        cmp: Cmp,
    ) -> Result<CmpOutcome, VmError> {
        let stack = &state.stack();
        state.handle_scope(|scope| {
            let hint = match cmp {
                Cmp::Eq => Hint::Default,
                Cmp::EqStrict | Cmp::Lt | Cmp::Le | Cmp::Gt | Cmp::Ge => Hint::Number,
            };
            let x = scope.handle(acc.get(heap));
            let x = match Object::to_primitive(vm, heap, state, x, hint)? {
                Coercion::Threw => return Ok(CmpOutcome::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let y = scope.handle(stack.reg(heap, &meta, reg));
            let y = match Object::to_primitive(vm, heap, state, y, hint)? {
                Coercion::Threw => return Ok(CmpOutcome::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let b = match cmp {
                Cmp::Eq => Compare::equal(heap, x.as_tagged(heap), y.as_tagged(heap))?,
                Cmp::EqStrict => Compare::strict_equal(heap, x.as_tagged(heap), y.as_tagged(heap)),
                Cmp::Lt => Compare::less_than(heap, x.as_tagged(heap), y.as_tagged(heap))?,
                Cmp::Le => Compare::less_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap))?,
                Cmp::Gt => Compare::greater_than(heap, x.as_tagged(heap), y.as_tagged(heap))?,
                Cmp::Ge => {
                    Compare::greater_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap))?
                }
            };
            Ok(CmpOutcome::Bool(b))
        })
    }

    /// The compare arms' cold body: the boolean lands in the accumulator.
    fn compare<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        meta: FrameMeta,
        acc: &Acc<'_>,
        reg: i32,
        cmp: Cmp,
    ) -> Step<'a> {
        match Self::compare_bool(vm, heap, state, meta, acc, reg, cmp) {
            Ok(CmpOutcome::Bool(b)) => {
                acc.store(Convert::boolean(heap, b));
                Step::Next
            }
            Ok(CmpOutcome::Threw) => Step::PendingThrow,
            Err(err) => Step::Error(err),
        }
    }

    fn global_load<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        stack: &Stack,
        cache: &StackCache,
        meta: FrameMeta,
        pc: usize,
        name_idx: usize,
        feedback_slot: usize,
        acc: &Acc<'_>,
        op: Opcode,
    ) -> Step<'a> {
        state.handle_scope(|scope| -> Step<'_> {
            let global = heap.known().global_object;
            let name = scope.handle(
                stack
                    .callable(heap, &meta)
                    .as_ref()
                    .constant_slot_name(heap, name_idx),
            );

            if let Some(hit) = InlineCache::try_load(
                heap,
                cache.feedback_ref(heap),
                feedback_slot,
                global.as_tagged(heap).erase(),
            ) {
                match hit {
                    // unreachable in practice: the arm's scope-free fast
                    // path already consumed plain value hits
                    Hit::Value(v) => acc.store(v),
                    Hit::Getter(getter) => {
                        let args = scope.stage(&[global.as_tagged(heap).erase()]);
                        let getter = scope.handle(getter);
                        match step_try!(call_value(
                            vm, state, heap, stack, cache, meta, pc, getter, args
                        )) {
                            Called::Frame => return Step::Reframe,
                            Called::NotCallable => {
                                acc.store(heap.known().undefined.as_tagged(heap));
                            }
                            Called::Immediate(v) => acc.store(v),
                            Called::Threw => return Step::PendingThrow,
                        }
                    }
                    // absent globals are uncached (transient: hoisting)
                    Hit::NotFound => {
                        if op == Opcode::LoadGlobal {
                            // unresolvable reference: GetValue throws
                            // a ReferenceError naming the binding
                            let text = slot_name_text(heap, name);
                            let ex = Errors::not_defined(vm, heap, state, &text)
                                .expect("error materialization must not fail");
                            state.set_pending_exception(ex);
                            return Step::PendingThrow;
                        }
                        acc.store(heap.known().undefined.as_tagged(heap));
                    }
                }
                return Step::Next;
            }

            match global.lookup(heap, name.as_tagged(heap)) {
                Lookup::Data { slot, .. } => {
                    acc.store(slot.get(heap));
                    InlineCache::update_load(
                        heap,
                        &scope,
                        cache.feedback_ref(heap).map(|v| scope.handle(v)),
                        feedback_slot,
                        Some(scope.handle(global.as_tagged(heap))),
                        name,
                        false,
                    );
                }
                Lookup::Accessor { pair, .. } => {
                    let getter = scope.handle(pair.get.get(heap));
                    InlineCache::update_load(
                        heap,
                        &scope,
                        cache.feedback_ref(heap).map(|v| scope.handle(v)),
                        feedback_slot,
                        Some(scope.handle(global.as_tagged(heap))),
                        name,
                        false,
                    );
                    if getter.as_tagged(heap) == heap.known().undefined.as_tagged(heap) {
                        acc.store(heap.known().undefined.as_tagged(heap));
                    } else {
                        let args = scope.stage(&[global.as_tagged(heap).erase()]);
                        match step_try!(call_value(
                            vm, state, heap, stack, cache, meta, pc, getter, args
                        )) {
                            Called::Frame => return Step::Reframe,
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
                        // unresolvable reference: GetValue throws a
                        // ReferenceError naming the binding
                        let text = slot_name_text(heap, name);
                        let ex = Errors::not_defined(vm, heap, state, &text)
                            .expect("error materialization must not fail");
                        state.set_pending_exception(ex);
                        return Step::PendingThrow;
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

    fn named_load<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        stack: &Stack,
        cache: &StackCache,
        meta: FrameMeta,
        pc: usize,
        reg: i32,
        name_idx: usize,
        feedback_slot: usize,
        acc: &Acc<'_>,
    ) -> Step<'a> {
        state.handle_scope(|scope| -> Step<'_> {
            let receiver = scope.handle(stack.reg(heap, &meta, reg));
            let name = scope.handle(
                stack
                    .callable(heap, &meta)
                    .as_ref()
                    .constant_slot_name(heap, name_idx),
            );

            if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
                let name = scope.handle(name.as_tagged(heap).erase());
                return match step_try!(Proxy::get(vm, heap, state, receiver, receiver, name)) {
                    Coercion::Threw => Step::PendingThrow,
                    Coercion::Value(v) => {
                        acc.store(v);
                        Step::Next
                    }
                };
            }

            if let Some(hit) = InlineCache::try_load(
                heap,
                cache.feedback_ref(heap),
                feedback_slot,
                receiver.as_tagged(heap),
            ) {
                match hit {
                    Hit::Value(v) => acc.store(v),
                    Hit::Getter(getter) => {
                        let args = scope.stage(&[receiver.as_tagged(heap).erase()]);
                        let getter = scope.handle(getter);
                        match step_try!(call_value(
                            vm, state, heap, stack, cache, meta, pc, getter, args
                        )) {
                            Called::Frame => return Step::Reframe,
                            Called::NotCallable => {
                                acc.store(heap.known().undefined.as_tagged(heap));
                            }
                            Called::Immediate(v) => acc.store(v),
                            Called::Threw => return Step::PendingThrow,
                        }
                    }
                    Hit::NotFound => {
                        acc.store(heap.known().undefined.as_tagged(heap));
                    }
                }
                return Step::Next;
            }

            let outcome = step_try!(Lookup::load_outcome(
                heap,
                receiver.as_tagged(heap),
                name.as_tagged(heap)
            ));
            match outcome {
                LoadOutcome::Value(v) => {
                    acc.store(v);
                    InlineCache::update_load(
                        heap,
                        &scope,
                        cache.feedback_ref(heap).map(|v| scope.handle(v)),
                        feedback_slot,
                        receiver
                            .as_tagged(heap)
                            .as_heap_object()
                            .map(|o| scope.handle(o)),
                        name,
                        true,
                    );
                }
                LoadOutcome::Getter(getter) => {
                    let getter = scope.handle(getter);
                    InlineCache::update_load(
                        heap,
                        &scope,
                        cache.feedback_ref(heap).map(|v| scope.handle(v)),
                        feedback_slot,
                        receiver
                            .as_tagged(heap)
                            .as_heap_object()
                            .map(|o| scope.handle(o)),
                        name,
                        true,
                    );
                    let args = scope.stage(&[receiver.as_tagged(heap).erase()]);
                    match step_try!(call_value(
                        vm, state, heap, stack, cache, meta, pc, getter, args
                    )) {
                        Called::Frame => return Step::Reframe,
                        Called::NotCallable => {
                            acc.store(heap.known().undefined.as_tagged(heap));
                        }
                        Called::Immediate(v) => acc.store(v),
                        Called::Threw => return Step::PendingThrow,
                    }
                }
            }
            Step::Next
        })
    }

    /// `AddImmediate` cold body: the full `Add` semantics with the
    /// constant as the left operand (string concat included).
    fn add_imm<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        meta: FrameMeta,
        acc: &Acc<'_>,
        reg: i32,
        imm: i32,
    ) -> Step<'a> {
        let stack = &state.stack();
        state.handle_scope(|scope| {
            let lhs = scope.handle(Smi::new(imm as i64).into_tagged());
            let rhs = scope.handle(stack.reg(heap, &meta, reg));
            let rhs = match step_try!(Object::to_primitive(vm, heap, state, rhs, Hint::Default)) {
                Coercion::Threw => return Step::PendingThrow,
                Coercion::Value(v) => scope.handle(v),
            };
            let is_string = rhs.as_tagged(heap).get_as::<DenseString>().is_some();
            if is_string {
                let a = scope.handle(step_try!(Convert::to_string(heap, &scope, lhs)));
                let b = scope.handle(step_try!(Convert::to_string(heap, &scope, rhs)));
                let concat = DenseString::concat(heap, &scope, a, b).as_tagged(heap);
                acc.store(concat);
            } else {
                let a = Convert::as_number(lhs.as_tagged(heap)).unwrap_or(f64::NAN);
                let b = step_try!(Convert::to_number(heap, rhs.as_tagged(heap)));
                // IEEE `-0 + -0` yields +0; the spec demands -0
                let r = a + b;
                let r = if r == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                    -0.0
                } else {
                    r
                };
                acc.store(heap.new_number(r));
            }
            Step::Next
        })
    }

    /// `LoadElementImm` cold body: the keyed-load tail with the constant
    /// index as the key (string receivers, proxies, named fallback).
    fn keyed_load_imm<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        stack: &Stack,
        cache: &StackCache,
        meta: FrameMeta,
        pc: usize,
        reg: i32,
        idx: usize,
        acc: &Acc<'_>,
    ) -> Step<'a> {
        state.handle_scope(|scope| -> Step<'_> {
            let receiver = scope.handle(stack.reg(heap, &meta, reg));
            let smi = Smi::new(idx as i64).into_tagged();
            let key: Handle<'_, SlotName> = scope.handle(smi.as_name());

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
                        Called::Frame => return Step::Reframe,
                        Called::NotCallable => acc.store(heap.known().undefined.as_tagged(heap)),
                        Called::Immediate(v) => acc.store(v),
                        Called::Threw => return Step::PendingThrow,
                    }
                }
            }
            Step::Next
        })
    }

    /// `LoadKeyedProperty` cold body: property-key coercion, string and
    /// proxy receivers, getters.
    fn keyed_load<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        stack: &Stack,
        cache: &StackCache,
        meta: FrameMeta,
        pc: usize,
        reg: i32,
        acc: &Acc<'_>,
    ) -> Step<'a> {
        state.handle_scope(|scope| -> Step<'_> {
            let receiver = scope.handle(stack.reg(heap, &meta, reg));
            let raw_key = scope.handle(acc.get(heap));
            let Some(key) = step_try!(Object::to_property_key(vm, heap, state, raw_key)) else {
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
                        Called::Frame => return Step::Reframe,
                        Called::NotCallable => acc.store(heap.known().undefined.as_tagged(heap)),
                        Called::Immediate(v) => acc.store(v),
                        Called::Threw => return Step::PendingThrow,
                    }
                }
            }
            Step::Next
        })
    }

    /// `StoreKeyedProperty` cold body: property-key coercion, proxies,
    /// element growth and named-property transitions.
    fn keyed_store<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        stack: &Stack,
        cache: &StackCache,
        meta: FrameMeta,
        pc: usize,
        recv: i32,
        key_reg: i32,
        acc: &Acc<'_>,
        semantics: StoreSemantics,
    ) -> Step<'a> {
        state.handle_scope(|scope| -> Step<'_> {
            let receiver = scope.handle(stack.reg(heap, &meta, recv));
            let raw = scope.handle(stack.reg(heap, &meta, key_reg));
            let Some(key) = step_try!(Object::to_property_key(vm, heap, state, raw)) else {
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
                    scope.handle(acc.get(heap)),
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
                            let value = scope.handle(acc.get(heap));
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
                acc.get(heap),
                semantics,
            ));
            let receiver = scope.handle(stack.reg(heap, &meta, recv));
            if step_try!(apply_store_outcome(
                vm, heap, state, stack, cache, meta, pc, receiver, outcome,
            )) {
                return Step::Reframe;
            }
            Step::Next
        })
    }
}
/// `Construct` cold body: [[Construct]] with receiver synthesis,
/// derived-class handling and proxy traps.
#[cold]
#[inline(never)]
fn construct<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    stack: &Stack,
    meta: FrameMeta,
    acc: &Acc<'_>,
    callee_reg: i32,
    args_base: i32,
    count: usize,
) -> Step<'a> {
    state.handle_scope(|scope| -> Step<'_> {
        let callee = scope.handle(stack.reg(heap, &meta, callee_reg));

        // ES 9.2.2 [[Construct]]: the callee's map decides whether it is
        // constructible and whether it is a derived class (whose `this`
        // is bound by super() and starts as TheHole).
        let Some(obj) = callee.as_tagged(heap).as_heap_object() else {
            return Step::Error(VmError::Type);
        };
        let kind = obj.as_ref().header.map.get(heap).kind();
        if !kind.is_constructor() {
            return Step::Error(VmError::Type);
        }
        let derived = kind.is_class_constructor()
            && obj
                .as_ref()
                .callable_info(heap)
                .is_some_and(|info| info.function_kind().is_derived_class_constructor());
        // a constructor proxy dispatches through its `construct` trap
        if Proxy::is_proxy(heap, callee.as_tagged(heap)) {
            let staged = stack.args(&meta, args_base, count);
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
            match Object::create_construct_receiver_value(vm, heap, state, callee.erase()) {
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
                .args(&meta, args_base, count)
                .iter()
                .map(|h| h.as_tagged(heap)),
        );
        let staged = scope.stage(&args);

        let callee = callee.erase();
        let result = scope.handle(step_try!(RuntimeContext::call(
            vm,
            heap,
            state,
            callee,
            staged,
            Some(callee)
        )));
        if result.as_tagged(heap) == heap.known().exception.as_tagged(heap) {
            return Step::PendingThrow;
        }
        acc.store(if Convert::is_primitive(heap, result.as_tagged(heap)) {
            if derived {
                // a derived constructor returned a primitive (ES 9.2.2.1)
                return Step::Error(VmError::Type);
            }
            receiver.as_tagged(heap).erase()
        } else {
            result.as_tagged(heap).erase()
        });
        Step::Next
    })
}

/// `StoreNamedProperty`/`NoShadow` cold body: proxies, the store IC,
/// transitions and setter invocation.
#[cold]
#[inline(never)]
fn store_named<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    pc: usize,
    acc: &Acc<'_>,
    recv_reg: i32,
    name_idx: usize,
    feedback_slot: usize,
    semantics: StoreSemantics,
    is_shadow: bool,
) -> Step<'a> {
    let op = if is_shadow {
        Opcode::StoreNamedProperty
    } else {
        Opcode::StoreNamedPropertyNoShadow
    };
    state.handle_scope(|scope| -> Step<'_> {
        let receiver = scope.handle(stack.reg(heap, &meta, recv_reg));
        let name = scope.handle(
            stack
                .callable(heap, &meta)
                .as_ref()
                .constant_slot_name(heap, name_idx),
        );
        let vector = cache.feedback_ref(heap).map(|v| scope.handle(v));
        // the stored value stays in the accumulator (a root); the
        // IC re-reads it fresh at each use

        if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
            let name = scope.handle(name.as_tagged(heap).erase());
            let value = scope.handle(acc.get(heap));
            return match step_try!(Proxy::set(vm, heap, state, receiver, name, value, receiver)) {
                Coercion::Threw => Step::PendingThrow,
                Coercion::Value(_) => Step::Next,
            };
        }

        // inline cache (Shadow semantics only: WriteThrough stores
        // into parent-pair arrays that no map describes)
        if op == Opcode::StoreNamedProperty
            && let Some(hit) =
                InlineCache::try_store(heap, &scope, vector, feedback_slot, receiver, name, &acc)
        {
            match hit {
                StoreHit::Done => {}
                StoreHit::Setter(setter) => {
                    let setter = scope.handle(setter);
                    let args = scope.stage(&[receiver.as_tagged(heap).erase(), acc.get(heap)]);
                    match step_try!(call_value(
                        vm, state, heap, stack, cache, meta, pc, setter, args
                    )) {
                        Called::Frame => return Step::Reframe,
                        Called::NotCallable | Called::Immediate(_) => {}
                        Called::Threw => return Step::PendingThrow,
                    }
                }
            }
            return Step::Next;
        }

        // the receiver's map before the store: the state key when
        // the outcome transitions it
        let prev = receiver
            .as_tagged(heap)
            .as_heap_object()
            .map(|o| scope.handle(o.as_ref().map_ref(heap)));
        let outcome = step_try!(receiver.store_lookup(
            heap,
            &scope,
            name.as_tagged(heap),
            acc.get(heap),
            semantics
        ));
        // update the IC before invoking a setter (mirrors loads)
        let kind = match outcome {
            StoreOutcome::Done => StoreOutcomeKind::Done,
            StoreOutcome::Transition { .. } => StoreOutcomeKind::Transition,
            StoreOutcome::CallSetter { .. } => StoreOutcomeKind::CallSetter,
        };
        if op == Opcode::StoreNamedProperty
            && let Some(prev) = prev
            && kind != StoreOutcomeKind::Transition
        {
            InlineCache::update_store(
                heap,
                &scope,
                vector,
                feedback_slot,
                receiver,
                name,
                prev,
                kind,
            );
        }
        if step_try!(apply_store_outcome(
            vm, heap, state, stack, cache, meta, pc, receiver, outcome,
        )) {
            return Step::Reframe;
        }
        // a transitioned store: cache prev -> the produced map
        if op == Opcode::StoreNamedProperty
            && let Some(prev) = prev
            && kind == StoreOutcomeKind::Transition
        {
            InlineCache::update_store(
                heap,
                &scope,
                vector,
                feedback_slot,
                receiver,
                name,
                prev,
                kind,
            );
        }
        Step::Next
    })
}

fn dispatch<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    base_depth: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let cache = &state.cache();
    let acc = cache.acc_mut();

    // Frame state cached across instructions (the quickjs shape: locals
    // in registers, not re-materialized per op). Two words detect every
    // invalidation: the code register (the GC rewrites it when the code
    // object moves; frame transitions reload it) and the frame base
    // (push/pop/unwind). Refreshed only on change — never per step.
    let mut frame_base = cache.base();
    let mut frame_regs = cache.register_count();
    let mut code_word = cache.code_raw();
    // Safety: the slice is re-derived on every code-register change, and
    // the GC only runs inside steps — between refreshes no read can see
    // a moved code object. Reads themselves happen only before an arm's
    // first allocation (the operand-cursor safepoint discipline).
    let mut code = code_bytes(cache, heap);
    let mut pc = cache.pc();

    loop {
        let (op, ops, next_pc) = decode(code, pc);
        let result = step(
            vm, heap, state, base_depth, pc, next_pc, frame_base, frame_regs, ops, op,
        );
        match result {
            Step::Next | Step::Jump(_) => {
                if let Step::Jump(target) = result {
                    pc = target;
                } else {
                    pc = next_pc;
                }
            }
            Step::Reframe => {
                frame_base = cache.base();
                frame_regs = cache.register_count();
                code_word = cache.code_raw();
                code = code_bytes(cache, heap);
                pc = cache.pc();
            }
            Step::Return => return Ok(acc.get(heap)),
            Step::Throw(v) => {
                state.set_pending_exception(v);
                match exception_dispatch(heap, state, base_depth, pc) {
                    Unwind::Caught(ex) => {
                        acc.store(ex);
                        frame_base = cache.base();
                        frame_regs = cache.register_count();
                        code_word = cache.code_raw();
                        code = code_bytes(cache, heap);
                        pc = cache.pc();
                    }
                    Unwind::Escaped => {
                        return Ok(heap.known().exception.as_tagged(heap).erase());
                    }
                }
            }
            Step::PendingThrow => match exception_dispatch(heap, state, base_depth, pc) {
                Unwind::Caught(ex) => {
                    acc.store(ex);
                    frame_base = cache.base();
                    frame_regs = cache.register_count();
                    code_word = cache.code_raw();
                    code = code_bytes(cache, heap);
                    pc = cache.pc();
                }
                Unwind::Escaped => return Ok(heap.known().exception.as_tagged(heap).erase()),
            },
            Step::Error(err) => match raise(vm, heap, state, base_depth, err, pc) {
                Unwind::Caught(ex) => {
                    acc.store(ex);
                    frame_base = cache.base();
                    frame_regs = cache.register_count();
                    code_word = cache.code_raw();
                    code = code_bytes(cache, heap);
                    pc = cache.pc();
                }
                Unwind::Escaped => return Ok(heap.known().exception.as_tagged(heap).erase()),
            },
        }
        // the code-register word doubles as the GC epoch: a collection
        // rewrites it in place when the code object moves, and nested
        // executions restore it — both flip the word, triggering one
        // cheap re-derivation of the raw slice
        if cache.code_raw() != code_word {
            code_word = cache.code_raw();
            code = code_bytes(cache, heap);
        }
    }
}

fn code_bytes(cache: &StackCache, heap: &Heap) -> &'static [u8] {
    let (ptr, len) = {
        let arr = cache.code_ref(heap);
        let bytes = arr.as_slice();
        (bytes.as_ptr(), bytes.len())
    };

    unsafe { core::slice::from_raw_parts(ptr, len) }
}

/// Returns `true` when a setter invocation pushed a frame (the caller
/// must yield `Step::Reframe` so the loop re-derives its cached state).
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
) -> Result<bool, VmError> {
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
                    .map(|_| false)
            })
        }
        StoreOutcome::CallSetter { setter } => state.handle_scope(|scope| {
            let value = cache.acc(heap);
            let args = scope.stage(&[receiver.as_tagged(heap).erase(), value]);
            let reframed = matches!(
                call_value(vm, state, heap, stack, cache, meta, pc, setter, args)?,
                Called::Frame
            );
            Ok(reframed)
        }),
        StoreOutcome::Done => Ok(false),
    }
}

#[inline(always)]
fn step<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    base_depth: usize,
    pc: usize,
    next_pc: usize,
    frame_base: usize,
    frame_regs: usize,
    ops: Operands<'_>,
    op: Opcode,
) -> Step<'a> {
    let stack = &state.stack();
    let cache = &state.cache();
    let acc = cache.acc_mut();

    let meta = FrameMeta {
        base: frame_base,
        pc: next_pc,
        register_count: frame_regs,
        handler_pc: 0,
    };

    match op {
        Opcode::Return => {
            if stack.frame_depth() == base_depth {
                return Step::Return;
            }
            let caller = stack
                .pop_frame(meta.base)
                .expect("suspended frame above base depth");
            cache.load(stack, caller, heap);
            Step::Reframe
        }
        Opcode::Load => {
            acc.store(stack.reg(heap, &meta, ops.reg(0)));
            Step::Next
        }
        Opcode::LoadSmi => {
            acc.store(Smi::new(ops.imm(0) as i64).into_tagged());
            Step::Next
        }
        Opcode::LoadConstant => {
            acc.store(cache.constants_ref(heap).at(heap, ops.idx(0)));
            Step::Next
        }
        Opcode::LoadZero => {
            acc.store(Smi::new(0).into_tagged());
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
            let global = heap.known().global_object.as_tagged(heap).erase();
            if let Some(Hit::Value(v)) =
                InlineCache::try_load(heap, cache.feedback_ref(heap), ops.idx(1), global)
            {
                acc.store(v);
                return Step::Next;
            }
            Slow::global_load(
                vm,
                heap,
                state,
                stack,
                cache,
                meta,
                pc,
                ops.idx(0),
                ops.idx(1),
                &acc,
                op,
            )
        }
        Opcode::LoadNamedProperty => {
            let receiver = stack.reg(heap, &meta, ops.reg(0));
            if let Some(Hit::Value(v)) =
                InlineCache::try_load(heap, cache.feedback_ref(heap), ops.idx(2), receiver)
            {
                acc.store(v);
                return Step::Next;
            }
            Slow::named_load(
                vm,
                heap,
                state,
                stack,
                cache,
                meta,
                pc,
                ops.reg(0),
                ops.idx(1),
                ops.idx(2),
                &acc,
            )
        }
        Opcode::LoadKeyedProperty => {
            if let Some(idx) = Smi::decode(acc.get(heap).raw())
                && idx.value() >= 0
                && let Some(recv) = stack.reg(heap, &meta, ops.reg(0)).as_heap_object()
                && let Some(v) = recv.as_ref().element_value(heap, idx.value() as usize)
            {
                acc.store(v);
                return Step::Next;
            }
            Slow::keyed_load(vm, heap, state, stack, cache, meta, pc, ops.reg(0), &acc)
        }
        Opcode::LoadNewTarget => {
            acc.store(stack.new_target_slot(&meta).get(heap));
            Step::Next
        }
        Opcode::Store => {
            stack.set_reg(&meta, ops.reg(0), acc.get(heap));
            Step::Next
        }
        Opcode::StoreGlobal => {
            let name_idx = ops.idx(0);
            state.handle_scope(|scope| -> Step<'_> {
                let outcome = step_try!({
                    let name = stack
                        .callable(heap, &meta)
                        .as_ref()
                        .constant_slot_name(heap, name_idx);
                    heap.known().global_object.store_lookup(
                        heap,
                        &scope,
                        name,
                        acc.get(heap),
                        StoreSemantics::WriteThrough,
                    )
                });
                if step_try!(apply_store_outcome(
                    vm,
                    heap,
                    state,
                    stack,
                    cache,
                    meta,
                    pc,
                    heap.known().global_object.erase(),
                    outcome,
                )) {
                    return Step::Reframe;
                }
                Step::Next
            })
        }
        Opcode::StoreNamedProperty | Opcode::StoreNamedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreNamedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            store_named(
                vm,
                heap,
                state,
                stack,
                cache,
                meta,
                pc,
                &acc,
                ops.reg(0),
                ops.idx(1),
                ops.idx(2),
                semantics,
                op == Opcode::StoreNamedProperty,
            )
        }
        Opcode::AddParent => {
            let recv_reg = ops.reg(0);
            let name_idx = ops.idx(1);
            state.handle_scope(|scope| -> Step<'_> {
                let receiver = scope.handle(stack.reg(heap, &meta, recv_reg));
                let name = stack
                    .callable(heap, &meta)
                    .as_ref()
                    .constant_slot_name(heap, name_idx);
                let name = scope.handle(name);
                let value = scope.handle(acc.get(heap));
                let Some(receiver) = scope.cast::<Object>(receiver.as_tagged(heap)) else {
                    return Step::Error(VmError::Type);
                };
                step_try!(Object::add_parent(heap, &scope, receiver, name, value));
                Step::Next
            })
        }
        Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreKeyedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            if let Some(idx) = Smi::decode(stack.reg(heap, &meta, ops.reg(1)).raw())
                && idx.value() >= 0
                && let Some(recv) = stack.reg(heap, &meta, ops.reg(0)).as_heap_object()
                && recv
                    .as_ref()
                    .element_value(heap, idx.value() as usize)
                    .is_some()
                && Object::store_array_element_in_place(
                    heap,
                    stack.reg(heap, &meta, ops.reg(0)),
                    idx.value() as usize,
                    acc.get(heap),
                )
                .is_ok()
            {
                return Step::Next;
            }
            Slow::keyed_store(
                vm,
                heap,
                state,
                stack,
                cache,
                meta,
                pc,
                ops.reg(0),
                ops.reg(1),
                &acc,
                semantics,
            )
        }
        Opcode::StoreKeyedSlot => {
            let recv_reg = ops.reg(0);
            let key_reg = ops.reg(1);
            state.handle_scope(|scope| -> Step<'_> {
                let receiver = scope.handle(stack.reg(heap, &meta, recv_reg));
                let raw = scope.handle(stack.reg(heap, &meta, key_reg));
                let Some(key) = step_try!(Object::to_property_key(vm, heap, state, raw)) else {
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
                        scope.handle(acc.get(heap)),
                        receiver,
                    )) {
                        Coercion::Threw => Step::PendingThrow,
                        Coercion::Value(_) => Step::Next,
                    };
                }

                let name: Handle<'_, SlotName> =
                    match step_try!(Lookup::classify_key(heap, key.as_tagged(heap).erase())) {
                        Key::Element(i) => {
                            let Some(obj) = scope.cast::<Object>(receiver.as_tagged(heap)) else {
                                return Step::Error(VmError::Type);
                            };
                            if obj.as_tagged(heap).as_ref().is_array(heap) {
                                let value = scope.handle(acc.get(heap));
                                step_try!(Object::store_array_element_in_place(
                                    heap,
                                    obj.as_tagged(heap).erase(),
                                    i,
                                    value.as_tagged(heap)
                                ));
                                return Step::Next;
                            }
                            let smi: Handle<'_, Smi> = scope.handle(Smi::new(i as i64));
                            let name = smi.as_tagged(heap).erase().as_name();
                            if matches!(
                                receiver.as_tagged(heap).lookup(heap, name),
                                Lookup::NotFound
                            ) {
                                return Step::Error(VmError::OutOfBounds);
                            }
                            scope.handle(name)
                        }
                        Key::Name(key) => scope.handle(key),
                    };

                let outcome = step_try!(receiver.as_tagged(heap).store_lookup(
                    heap,
                    &scope,
                    name.as_tagged(heap),
                    acc.get(heap),
                    StoreSemantics::WriteThrough,
                ));
                let receiver = scope.handle(stack.reg(heap, &meta, recv_reg));
                if step_try!(apply_store_outcome(
                    vm, heap, state, stack, cache, meta, pc, receiver, outcome,
                )) {
                    return Step::Reframe;
                }
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
                let mut context = match stack.context_slot(&meta).get(heap).get_as::<Context>() {
                    Some(context) => context,
                    None => return Step::Error(VmError::Type),
                };
                for _ in 0..depth {
                    context = match context.as_ref().outer.get(heap) {
                        Some(context) => context,
                        None => return Step::Error(VmError::Type),
                    };
                }
                context
                    .slots
                    .get(heap)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .get(heap)
            };
            acc.store(v);
            Step::Next
        }
        Opcode::StoreContextSlot => {
            let mut context = match stack.context_slot(&meta).get(heap).get_as::<Context>() {
                Some(context) => context,
                None => return Step::Error(VmError::Type),
            };
            for _ in 0..ops.uimm(1) {
                context = match context.as_ref().outer.get(heap) {
                    Some(context) => context,
                    None => return Step::Error(VmError::Type),
                };
            }
            // Safety: fresh anchored slot read.
            let host = context.erase();
            context
                .slots
                .get(heap)
                .as_ref()
                .element_slot(ops.idx(0))
                .set(heap, host, acc.get(heap));
            Step::Next
        }
        Opcode::CreateFunctionContext => {
            // constants[idx] is the scope's shared ScopeInfo (its `names`
            // array is parallel to the context's slots)
            let scope_idx = ops.idx(0);
            let count = step_try!({
                cache
                    .constants_ref(heap)
                    .at(heap, scope_idx)
                    .get_as::<ScopeInfo>()
                    .map(|r| r.as_ref().names.get(heap).len())
                    .ok_or(VmError::Type)
            });
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .cast::<Context>(stack.context_slot(&meta).get(heap))
                    .expect("frame context slot holds a Context");
                let values =
                    scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
                let scope_info = scope
                    .cast::<ScopeInfo>(cache.constants_ref(heap).at(heap, scope_idx))
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
                    .cast::<Context>(stack.context_slot(&meta).get(heap))
                    .expect("frame context slot holds a Context");
                let values =
                    scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info: heap.known().empty_scope_info,
                })
            });
            acc.store(ctx);
            Step::Next
        }
        Opcode::PushContext => {
            let old = stack.context_slot(&meta).get(heap);
            stack.set_reg(&meta, ops.reg(0), old);
            let context = acc.get(heap);
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
            acc.store(stack.context_slot(&meta).get(heap));
            Step::Next
        }
        // TODO: feedback vectors and separation once they are there
        Opcode::Call | Opcode::CallNoFeedback => {
            // TODO(strict-mode): ordinary sloppy functions still need nullish
            // receiver substitution and primitive receiver boxing.
            let callee_reg = ops.reg(0);
            let args_base = ops.reg_list(1);
            let count = ops.reg_count(2);
            // a callable proxy dispatches through its `apply` trap (or
            // a nested call of the target)
            if Proxy::is_proxy(heap, stack.reg(heap, &meta, callee_reg)) {
                return match state.handle_scope(|scope| {
                    // staged: trap lookups may run user getters
                    let callee = scope.handle(stack.reg(heap, &meta, callee_reg));
                    let staged = stack.args(&meta, args_base, count);
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
            match Object::call_target(heap, stack.reg(heap, &meta, callee_reg)) {
                None => Step::Error(VmError::Type),
                Some(CallTarget::Runtime(idx)) => {
                    let f = vm.runtime(RuntimeIndex(idx));
                    let exception = heap.known().exception.as_tagged(heap).raw();
                    let nctx = RuntimeContext::new(vm, heap, state);
                    let result = f(nctx, stack.args(&meta, args_base, count));
                    match result {
                        // Safety: old-gen singleton word.
                        Ok(v) if v.raw() == exception => Step::PendingThrow,
                        Ok(v) => {
                            acc.store(v);
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
                            heap,
                            meta,
                            pc,
                            callee,
                            register_count,
                            context,
                            args_base,
                            count,
                            undefined,
                            formal_min,
                        ))
                    };
                    cache.load(stack, frame, heap);
                    Step::Reframe
                }
            }
        }
        Opcode::CallRuntime => {
            // operand 0 is a RuntimeFn discriminant: the fixed
            // runtime-helper table (vm::runtime_fn) registered at
            // indices 0..RuntimeFn::COUNT
            let f = vm.runtime(RuntimeIndex(ops.idx(0)));
            let args_base = ops.reg_list(1);
            let count = ops.reg_count(2);
            let exception = heap.known().exception.as_tagged(heap).raw();
            let nctx = RuntimeContext::new(vm, heap, state);
            let result = f(nctx, stack.args(&meta, args_base, count));
            match result {
                // Safety: old-gen singleton word.
                Ok(v) if v.raw() == exception => Step::PendingThrow,
                Ok(v) => {
                    acc.store(v);
                    Step::Next
                }
                Err(err) => Step::Error(err),
            }
        }
        Opcode::Construct => construct(
            vm,
            heap,
            state,
            stack,
            meta,
            &acc,
            ops.reg(0),
            ops.reg_list(1),
            ops.reg_count(2),
        ),
        Opcode::CreateEmptyObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().object_initial_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY)
            });
            acc.store(obj);
            Step::Next
        }
        Opcode::CreateEmptyArrayLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().js_array_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY)
            });
            acc.store(obj);
            Step::Next
        }
        Opcode::CreateBareObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().plain_object_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY)
            });
            acc.store(obj);
            Step::Next
        }
        Opcode::CreateClosure => {
            let info_idx = ops.idx(0);
            state.handle_scope(|scope| -> Step<'_> {
                let Some(info) =
                    scope.cast::<CallableInfoObject>(cache.constants_ref(heap).at(heap, info_idx))
                else {
                    return Step::Error(VmError::Type);
                };
                let context = scope
                    .cast::<Context>(stack.context_slot(&meta).get(heap))
                    .expect("frame context slot holds a Context");
                let obj = step_try!(Object::create_closure(heap, &scope, info, context));
                acc.store(obj);
                Step::Next
            })
        }
        Opcode::LoadCurrentClosure => {
            acc.store(stack.callable_slot(&meta).get(heap));
            Step::Next
        }
        Opcode::Add => {
            let other_reg = ops.reg(0);
            let other = stack.reg(heap, &meta, other_reg);
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && let Some(r) = a.checked_add(b)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Step::Next;
            }
            state.handle_scope(|scope| {
                let lhs = scope.handle(acc.get(heap));
                let lhs = match step_try!(Object::to_primitive(vm, heap, state, lhs, Hint::Default))
                {
                    Coercion::Threw => return Step::PendingThrow,
                    Coercion::Value(v) => scope.handle(v),
                };
                let rhs = scope.handle(stack.reg(heap, &meta, other_reg));
                let rhs = match step_try!(Object::to_primitive(vm, heap, state, rhs, Hint::Default))
                {
                    Coercion::Threw => return Step::PendingThrow,
                    Coercion::Value(v) => scope.handle(v),
                };
                let is_string = (
                    lhs.as_tagged(heap).get_as::<DenseString>().is_some(),
                    rhs.as_tagged(heap).get_as::<DenseString>().is_some(),
                );
                if is_string.0 || is_string.1 {
                    let a = scope.handle(step_try!(Convert::to_string(heap, &scope, lhs)));
                    let b = scope.handle(step_try!(Convert::to_string(heap, &scope, rhs)));
                    let s = DenseString::concat(heap, &scope, a, b).as_tagged(heap);
                    acc.store(s);
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
                    acc.store(heap.new_number(r));
                }
                Step::Next
            })
        }
        Opcode::Sub => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && let Some(r) = a.checked_sub(b)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Step::Next;
            }
            Slow::numeric_op(vm, heap, state, meta, &acc, ops.reg(0), |a, b| a - b)
        }
        Opcode::Mul => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && let Some(r) = a.checked_mul(b)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Step::Next;
            }
            Slow::numeric_op(vm, heap, state, meta, &acc, ops.reg(0), |a, b| a * b)
        }
        Opcode::Div => {
            // JS division is IEEE double division: 7/2 = 3.5, x/0 = ±Infinity
            // or NaN, MIN/-1 overflows to a double.
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && b != 0
                && a % b == 0
                && let Some(r) = a.checked_div(b)
            {
                acc.store(Smi::new(r).into_tagged());
                return Step::Next;
            }
            Slow::numeric_op(vm, heap, state, meta, &acc, ops.reg(0), |a, b| a / b)
        }
        Opcode::Mod => {
            // JS remainder is IEEE fmod: x % 0 = NaN, signs follow the dividend.
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.to_i64(), other.to_i64())
                && b != 0
            {
                acc.store(Smi::new(a % b).into_tagged());
                return Step::Next;
            }
            Slow::numeric_op(vm, heap, state, meta, &acc, ops.reg(0), |a, b| a % b)
        }
        Opcode::Exp => {
            let reg = ops.reg(0);
            state.handle_scope(|scope| -> Step<'_> {
                // JS exponentiation is always IEEE double math; the result only
                // needs a Smi tag when it is an in-range integer.
                let a = scope.handle(acc.get(heap));
                let b = scope.handle(stack.reg(heap, &meta, reg));
                let v = step_try!(Object::numeric_op(vm, heap, state, a, b, |a, b| a.powf(b)));
                let Some(v) = v else {
                    return Step::PendingThrow;
                };
                acc.store(v);
                Step::Next
            })
        }
        Opcode::BitwiseOr => {
            // ToInt32 semantics on the (integer) smi inputs
            let a = step_try!(acc.to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(heap, &meta, ops.reg(0))
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            acc.store(Smi::new((a | b) as i64).into_tagged());
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
            acc.store(Smi::new((a ^ b) as i64).into_tagged());
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
            acc.store(Smi::new((a & b) as i64).into_tagged());
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
            acc.store(Smi::new(a.wrapping_shl(b & 31) as i64).into_tagged());
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
            acc.store(Smi::new(a.wrapping_shr(b & 31) as i64).into_tagged());
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
            acc.store(Smi::new(a.wrapping_shr(b & 31) as i64).into_tagged());
            Step::Next
        }
        Opcode::Jump => Step::Jump(jump_target(pc, ops.imm(0))),
        Opcode::JumpLoop => {
            // read the offset before the safepoint poll: a collection
            // invalidates the operand cursor
            let offset = ops.imm(0);
            if heap.safepoint_poll() {
                return begin_termination(heap, state);
            }
            Step::Jump(jump_target(pc, offset))
        }
        Opcode::JumpIfTruthy => {
            if Convert::is_truthy(heap, acc.get(heap)) {
                return Step::Jump(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfFalsy => {
            if !Convert::is_truthy(heap, acc.get(heap)) {
                return Step::Jump(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfNotUndefined => {
            if *acc != heap.known().undefined.as_tagged(heap) {
                return Step::Jump(jump_target(pc, ops.imm(0)));
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
            acc.store(Object::type_of(heap, acc.get(heap)));
            Step::Next
        }
        Opcode::Negate => {
            let acc_word = *acc;
            if let Some(v) = acc_word.to_i64() {
                if v == 0 {
                    // preserve -0.0: `-0` must not fold into Smi 0
                    acc.store(heap.new_number(-0.0));
                } else if v == Smi::MIN {
                    // `-Smi::MIN` overflows i64
                    acc.store(heap.new_number(-(v as f64)));
                } else {
                    acc.store(Smi::new(-v).into_tagged());
                }
            } else {
                let n = state.handle_scope(|scope| {
                    let acc = scope.handle(acc.get(heap));
                    Object::to_numeric(vm, heap, state, acc)
                });
                let n = step_try!(n);
                let Some(n) = n else {
                    return Step::PendingThrow;
                };
                // new_number boxes -0.0 itself
                acc.store(heap.new_number(-n));
            }
            Step::Next
        }
        Opcode::InstanceOf => {
            let callable_reg = ops.reg(0);
            state.handle_scope(|scope| -> Step<'_> {
                let object = scope.handle(acc.get(heap));
                let callable = scope.handle(stack.reg(heap, &meta, callable_reg));
                let r = step_try!(Object::instance_of(vm, heap, state, object, callable));
                let Some(r) = r else {
                    return Step::PendingThrow;
                };
                acc.store(Convert::boolean(heap, r));
                Step::Next
            })
        }
        Opcode::EqualStrict => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            let r = Compare::strict_equal(heap, acc.get(heap), other);
            acc.store(Convert::boolean(heap, r));
            Step::Next
        }
        Opcode::Equal => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) =
                (Convert::as_number(acc.get(heap)), Convert::as_number(other))
            {
                acc.store(Convert::boolean(heap, a == b));
                return Step::Next;
            }
            Slow::compare(vm, heap, state, meta, &acc, ops.reg(0), Cmp::Eq)
        }
        Opcode::LessThan => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) =
                (Convert::as_number(acc.get(heap)), Convert::as_number(other))
            {
                acc.store(Convert::boolean(heap, a < b));
                return Step::Next;
            }
            Slow::compare(vm, heap, state, meta, &acc, ops.reg(0), Cmp::Lt)
        }
        Opcode::LessThanOrEqual => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) =
                (Convert::as_number(acc.get(heap)), Convert::as_number(other))
            {
                acc.store(Convert::boolean(heap, a <= b));
                return Step::Next;
            }
            Slow::compare(vm, heap, state, meta, &acc, ops.reg(0), Cmp::Le)
        }
        Opcode::GreaterThan => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) =
                (Convert::as_number(acc.get(heap)), Convert::as_number(other))
            {
                acc.store(Convert::boolean(heap, a > b));
                return Step::Next;
            }
            Slow::compare(vm, heap, state, meta, &acc, ops.reg(0), Cmp::Gt)
        }
        Opcode::GreaterThanOrEqual => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            if let (Some(a), Some(b)) =
                (Convert::as_number(acc.get(heap)), Convert::as_number(other))
            {
                acc.store(Convert::boolean(heap, a >= b));
                return Step::Next;
            }
            Slow::compare(vm, heap, state, meta, &acc, ops.reg(0), Cmp::Ge)
        }
        Opcode::CompareJump => {
            let other = stack.reg(heap, &meta, ops.reg(0));
            // kind = cmp index * 2 + jump_if_falsy
            let cmp = Cmp::from_kind(ops.uimm(1) / 2);
            let falsy_jump = ops.uimm(1) % 2 == 1;
            let offset = ops.imm(2);
            let b = if let (Some(a), Some(b)) =
                (Convert::as_number(acc.get(heap)), Convert::as_number(other))
            {
                match cmp {
                    Cmp::Eq | Cmp::EqStrict => a == b,
                    Cmp::Lt => a < b,
                    Cmp::Le => a <= b,
                    Cmp::Gt => a > b,
                    Cmp::Ge => a >= b,
                }
            } else {
                match Slow::compare_bool(vm, heap, state, meta, &acc, ops.reg(0), cmp) {
                    Ok(CmpOutcome::Bool(b)) => b,
                    Ok(CmpOutcome::Threw) => return Step::PendingThrow,
                    Err(err) => return Step::Error(err),
                }
            };
            acc.store(Convert::boolean(heap, b));
            if b != falsy_jump {
                return Step::Jump(jump_target(pc, offset));
            }
            Step::Next
        }
        Opcode::AddImmediate => {
            let imm = ops.imm(1);
            let rhs = stack.reg(heap, &meta, ops.reg(0));
            if let Some(a) = Smi::decode(rhs.raw()).map(|s| s.value())
                && let Some(r) = a.checked_add(imm as i64)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Step::Next;
            }
            Slow::add_imm(vm, heap, state, meta, &acc, ops.reg(0), imm)
        }
        Opcode::LoadElementImm => {
            let idx = ops.uimm(1) as usize;
            if let Some(recv) = stack.reg(heap, &meta, ops.reg(0)).as_heap_object()
                && let Some(v) = recv.as_ref().element_value(heap, idx)
            {
                acc.store(v);
                return Step::Next;
            }
            Slow::keyed_load_imm(
                vm,
                heap,
                state,
                stack,
                cache,
                meta,
                pc,
                ops.reg(0),
                idx,
                &acc,
            )
        }
        Opcode::Throw | Opcode::ReThrow => Step::Throw(acc.get(heap)),
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}
