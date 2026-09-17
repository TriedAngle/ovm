use bytecode::{Opcode, Operands, decode, jump_target};

use crate::{
    CallTarget, CallableInfoObject, Compare, Context, ContextInit, Convert, DenseString,
    FixedArray, FunctionKind, GcSlice, Handle, Heap, Key, LoadOutcome, Lookup, Object,
    PropertyDescriptor, ScopeInfo, SlotName, Smi, StoreOutcome, StoreSemantics, Tagged, Value,
    call_target, classify_key, function_kind_of, load_outcome, store_array_element,
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
    new_target: Option<Handle<'_, Value>>,
) -> Result<Value, VmError> {
    let stack = &state.stack;
    let cache = &state.cache;

    let saved_top = stack.top();
    let was_active = cache.is_active();
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
    new_target: Option<Handle<'_, Value>>,
    base_depth: usize,
) -> Result<Value, VmError> {
    // Safety: entry handle word, consumed before any allocation.
    let callee_word = unsafe { callable.read_unchecked() };
    if heap.no_gc(|heap| crate::proxy::is_proxy(heap, unsafe { callee_word.assume_valid(heap) })) {
        // no raw copies: stage the args into GC-visited stack memory —
        // trap lookups may run user getters, so raw Vecs would go stale
        return state.handle_scope(|scope| {
            let staged = scope.stage_words(args.words());
            let argv: Vec<_> = staged
                .words()
                .iter()
                // Safety: fresh rooted-slot words, rooted by the callee
                // before its first allocation.
                .map(|w| unsafe { Tagged::from_value_unchecked(*w) })
                .collect();
            let result = match new_target {
                None => crate::proxy::apply(
                    vm,
                    heap,
                    state,
                    // Safety: entry word, no GC since the is_proxy check.
                    unsafe { Tagged::from_value_unchecked(callee_word) },
                    &argv,
                ),
                Some(nt) => {
                    let real: Vec<Value> = staged.words().iter().skip(1).copied().collect();
                    let staged = scope.stage_words(&real);
                    let words: Vec<_> = staged
                        .words()
                        .iter()
                        // Safety: fresh rooted-slot words.
                        .map(|w| unsafe { Tagged::from_value_unchecked(*w) })
                        .collect();
                    crate::proxy::construct(
                        vm,
                        heap,
                        state,
                        // Safety: entry word, no GC since the is_proxy check.
                        unsafe { Tagged::from_value_unchecked(callee_word) },
                        &words,
                        // Safety: entry handle word.
                        unsafe { Tagged::from_value_unchecked(nt.read_unchecked()) },
                    )
                }
            };
            match result {
                // Safety: old-gen singleton word, returned for an
                // immediate store/compare by the caller.
                Ok(Coercion::Threw) => Ok(unsafe { heap.known().exception.read_unchecked() }),
                Ok(Coercion::Value(v)) => Ok(v),
                Err(e) => Err(e),
            }
        });
    }
    match classify_callee(&*heap, callee_word) {
        Callee::NotCallable => Err(VmError::Type),
        Callee::Native(idx) => {
            let f = vm.native(NativeIndex(idx));
            let (saved_top, fargs) = state.stack.stage_args(args)?;
            let mut nctx = NativeContext::with_new_target(vm, heap, state, new_target);
            let result = f(&mut nctx, fargs);
            state.stack.set_top(saved_top);
            result
        }
        Callee::Bytecode(register_count, kind) => {
            if new_target.is_none() && kind.is_class_constructor() {
                return Err(VmError::Type);
            }
            // derived constructors receive TheHole as their receiver: `this`
            // stays uninitialized until super() binds it (ES 10.2.2)
            let stack = &state.stack;
            // one shared anchor covers every read feeding the frame push;
            // the push itself never allocates, so the anchor may span it
            let frame = {
                let heap = &*heap;
                let target = callable.as_tagged(heap);
                let context = closure_context(heap, target);
                let formal_min = target
                    .erase_type()
                    .as_heap_object()
                    .and_then(|obj| obj.as_ref().callable_info(heap))
                    .map(|info| info.formal_parameter_count() + 1)
                    .unwrap_or(1);
                let new_target_value = match &new_target {
                    Some(nt) => nt.as_tagged(heap).erase_type(),
                    None => heap.known().undefined.as_tagged(heap).erase_type(),
                };
                stack.push_initial_frame(
                    target,
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

/// Callee classification for the call paths: the anchored `CallTarget`
/// payloads erased to plain data so no anchor crosses an allocation.
enum Callee {
    NotCallable,
    Native(usize),
    Bytecode(usize, FunctionKind),
}

fn classify_callee(heap: &Heap, f: Value) -> Callee {
    // Safety: caller-supplied word, fresh at entry.
    match call_target(heap, unsafe { f.assume_valid(heap) }) {
        Some(CallTarget::Native(idx)) => Callee::Native(idx),
        Some(CallTarget::Bytecode(_, register_count, kind)) => {
            Callee::Bytecode(register_count, kind)
        }
        None => Callee::NotCallable,
    }
}

fn callable_name<'a>(
    heap: &'a Heap,
    stack: &Stack,
    meta: &FrameMeta,
    idx: usize,
) -> Tagged<'a, SlotName> {
    let callable = stack.callable_slot(meta).read(heap);
    let Some(callable) = callable.as_heap_object() else {
        panic!("frame callable must be an object");
    };
    let info = callable
        .as_ref()
        .callable_info(heap)
        .expect("frame callable must have callable info");
    info.constant_slot_name(heap, idx)
}

fn frame_context(stack: &Stack, meta: &FrameMeta) -> Result<Value, VmError> {
    // rooted memory: the word is current and stays valid until the next GC
    Ok(stack.context_slot(meta).inner())
}

fn set_frame_context(
    stack: &Stack,
    meta: &FrameMeta,
    context: Tagged<'_, Value>,
) -> Result<(), VmError> {
    context.get_as::<Context>().ok_or(VmError::Type)?;
    stack.context_slot(meta).store(context);
    Ok(())
}

/// The closure context a freshly pushed frame starts with: the callee's
/// immutable captured context (its closure slot).
fn closure_context<'a>(heap: &'a Heap, callable: Tagged<'a, Object>) -> Tagged<'a, Value> {
    let Some(obj) = callable.erase_type().as_heap_object() else {
        panic!("callable must be an object");
    };
    obj.as_ref()
        .closure_context(heap)
        .expect("callable must have a closure context")
        .into_tagged()
        .erase_type()
}

/// Result of an inline call attempt (`call_value`).
enum Called {
    /// A bytecode frame was pushed; its `Return` produces the value.
    Frame,
    /// The callee is not callable.
    NotCallable,
    /// The call ran to completion eagerly (proxy `apply`): the value.
    Immediate(Value),
    /// The call threw; the pending exception is set.
    Threw,
}

fn call_value(
    vm: &VM,
    state: &ContextState,
    heap: &mut Heap,
    stack: &Stack,
    cache: &StackCache,
    meta: FrameMeta,
    handler_pc: usize,
    f: Value,
    args: &[Value],
) -> Result<Called, VmError> {
    // proxies dispatch through their `apply` trap (or the target call)
    // via a nested run — their callable state lives on a ProxyObject,
    // not in function slots
    if heap.no_gc(|heap| crate::proxy::is_proxy(heap, unsafe { f.assume_valid(heap) })) {
        return state.handle_scope(|scope| {
            // staged: trap lookups may run user getters
            let staged = scope.stage_words(args);
            let argv: Vec<_> = staged
                .words()
                .iter()
                // Safety: fresh rooted-slot words.
                .map(|w| unsafe { Tagged::from_value_unchecked(*w) })
                .collect();
            match crate::proxy::apply(
                vm,
                heap,
                state,
                // Safety: entry word, no GC since the is_proxy check.
                unsafe { Tagged::from_value_unchecked(f) },
                &argv,
            )? {
                Coercion::Threw => Ok(Called::Threw),
                Coercion::Value(v) => Ok(Called::Immediate(v)),
            }
        });
    }
    // TODO: native getters/setters invoke in place instead of pushing a frame
    let Callee::Bytecode(register_count, kind) = classify_callee(&*heap, f) else {
        return Ok(Called::NotCallable);
    };
    if kind.is_class_constructor() {
        return Err(VmError::Type);
    }
    let context = closure_context(&*heap, unsafe { Tagged::from_value_unchecked(f) }).erase();
    // Safety: old-gen singleton word.
    let undefined = unsafe { heap.known().undefined.read_unchecked() };
    let formal_min = heap.no_gc(|heap| {
        unsafe { f.assume_valid(heap) }
            .as_heap_object()
            .and_then(|obj| obj.as_ref().callable_info(heap))
            .map(|info| info.formal_parameter_count() + 1)
            .unwrap_or(1)
    });
    let callee = stack.push_frame_with_args(
        meta,
        handler_pc,
        // Safety: entry word, no GC since the classify reads above.
        unsafe { Tagged::from_value_unchecked(f) },
        register_count,
        // Safety: closure-context word read above, no GC since.
        unsafe { Tagged::from_value_unchecked(context) },
        args,
        // Safety: old-gen singleton word.
        unsafe { Tagged::from_value_unchecked(undefined) },
        formal_min,
    )?;
    cache.load(stack, callee, heap);
    Ok(Called::Frame)
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
        let handled = heap.no_gc(|heap| {
            let Some(obj) = stack
                .callable_slot(&cache.frame_meta())
                .read(heap)
                .as_heap_object()
            else {
                return None;
            };
            let info = obj.as_ref().callable_info(heap)?;
            info.handlers.heap_ref(heap)?.lookup(pc)
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
        let (op, ops, next_pc) = decode(cache.code_ref(&*heap).as_slice(), pc);
        if std::env::var("OVM_TRACE").is_ok() {
            trace += 1;
            if trace > 200000 {
                panic!("trace limit");
            }
            eprintln!(
                "pc={pc} base={} rc={} op={op:?} acc={:?}",
                cache.frame_meta().base,
                cache.frame_meta().register_count,
                cache.acc(&*heap),
            );
        }
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
                    // Safety: old-gen singleton word.
                    Unwind::Escaped => {
                        return Ok(unsafe { heap.known().exception.read_unchecked() });
                    }
                }
            }
            Step::PendingThrow => match exception_dispatch(heap, state, base_depth, pc) {
                Unwind::Caught(ex) => cache.set_acc(ex),
                // Safety: old-gen singleton word.
                Unwind::Escaped => return Ok(unsafe { heap.known().exception.read_unchecked() }),
            },
            Step::Error(err) => match raise(vm, heap, state, base_depth, err, pc) {
                Unwind::Caught(ex) => cache.set_acc(ex),
                // Safety: old-gen singleton word.
                Unwind::Escaped => return Ok(unsafe { heap.known().exception.read_unchecked() }),
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
    receiver: Value,
    outcome: StoreOutcome<'_>,
) -> Result<(), VmError> {
    match outcome {
        StoreOutcome::Transition {
            receiver: recv,
            name,
        } => {
            state.handle_scope(|scope| {
                // fresh word for the descriptor, read at the call
                let value = scope.handle(cache.acc(&*heap));
                Object::add_own_property(heap, &scope, recv, name, PropertyDescriptor::data(value))
                    // TODO(strict-mode): a false result must throw in strict code;
                    // the current store path preserves its existing sloppy result.
                    .map(|_| ())
            })
        }
        StoreOutcome::CallSetter { setter } => {
            // fresh word for the argument, read at the call
            let value = cache.acc(&*heap).erase();
            call_value(
                vm,
                state,
                heap,
                stack,
                cache,
                meta,
                pc,
                // Safety: rooted-slot word, consumed by the call.
                unsafe { setter.read_unchecked() },
                &[receiver, value],
            )?;
            Ok(())
        }
        StoreOutcome::Done => Ok(()),
    }
}

/// A `LoadOutcome` with its anchored payloads erased to raw words, so it
/// can leave the `no_gc` region that produced it. Each word must be
/// consumed (stored or rooted) before the next allocation.
enum LoadResult {
    Value(Value),
    Getter(Value),
}

impl LoadResult {
    fn of(outcome: LoadOutcome<'_>) -> Self {
        match outcome {
            LoadOutcome::Value(v) => LoadResult::Value(v.erase()),
            LoadOutcome::Getter(g) => LoadResult::Getter(g.erase()),
        }
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
                return Step::Return(cache.acc(&*heap).erase());
            }
            let caller = stack
                .pop_frame(meta.base)
                .expect("suspended frame above base depth");
            cache.load(stack, caller, heap);
            Step::Next
        }
        Opcode::Load => {
            cache.set_acc(stack.reg(&*heap, &meta, ops.reg(0)));
            Step::Next
        }
        Opcode::Store => {
            stack.set_reg(&meta, ops.reg(0), cache.acc(&*heap));
            Step::Next
        }
        Opcode::Move => {
            stack.set_reg(&meta, ops.reg(0), stack.reg(&*heap, &meta, ops.reg(1)));
            Step::Next
        }
        Opcode::LoadSmi => {
            cache.set_acc(Smi::new(ops.imm(0) as i64).encode());
            Step::Next
        }
        Opcode::LoadConstant => {
            let v = heap.no_gc(|heap| cache.constants_ref(heap).at(heap, ops.idx(0)).erase());
            cache.set_acc(v);
            Step::Next
        }
        Opcode::LdaZero => {
            cache.set_acc(Smi::new(0).encode());
            Step::Next
        }
        Opcode::LdaUndefined => {
            // Safety: old-gen singleton word.
            cache.set_acc(unsafe { heap.known().undefined.read_unchecked() });
            Step::Next
        }
        Opcode::LdaNull => {
            // Safety: old-gen singleton word.
            cache.set_acc(unsafe { heap.known().null.read_unchecked() });
            Step::Next
        }
        Opcode::LdaTrue => {
            // Safety: old-gen singleton word.
            cache.set_acc(unsafe { heap.known().true_object.read_unchecked() });
            Step::Next
        }
        Opcode::LdaFalse => {
            // Safety: old-gen singleton word.
            cache.set_acc(unsafe { heap.known().false_object.read_unchecked() });
            Step::Next
        }
        Opcode::Add => {
            // the register index, not a raw copy: the first to_primitive
            // below may run user code, and a held raw value would go
            // stale — registers are GC-visited and always read fresh
            let other_reg = ops.reg(0);
            let acc_word = cache.acc(&*heap).erase();
            let other_word = stack.reg(&*heap, &meta, other_reg).erase();
            let mut result = None;
            if let (Some(a), Some(b)) = (acc_word.to_i64(), other_word.to_i64())
                && let Some(r) = a.checked_add(b)
                && Smi::in_range(r)
            {
                result = Some(Smi::new(r).encode());
            }
            match result {
                Some(v) => {
                    cache.set_acc(v);
                    Step::Next
                }
                None => {
                    state.handle_scope(|scope| {
                        // to_primitive may run user code and to_string
                        // allocates: the coerced operands must stay rooted
                        // across the sibling coercion
                        let lhs = step_try!(Runtime::to_primitive(
                            vm,
                            heap,
                            state,
                            acc_word,
                            Hint::Default
                        ));
                        let lhs = match lhs {
                            Coercion::Threw => return Step::PendingThrow,
                            // Safety: fresh word from the coercion.
                            Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                        };
                        // re-read the register: the coercion above may
                        // have run user code
                        let other_word = stack.reg(&*heap, &meta, other_reg).erase();
                        let rhs = step_try!(Runtime::to_primitive(
                            vm,
                            heap,
                            state,
                            other_word,
                            Hint::Default
                        ));
                        let rhs = match rhs {
                            Coercion::Threw => return Step::PendingThrow,
                            // Safety: fresh word from the coercion.
                            Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                        };
                        let is_string = heap.no_gc(|heap| {
                            (
                                lhs.as_tagged(heap).get_as::<DenseString>().is_some(),
                                rhs.as_tagged(heap).get_as::<DenseString>().is_some(),
                            )
                        });
                        if is_string.0 || is_string.1 {
                            let s = step_try!((|| -> Result<Value, VmError> {
                                // Safety: fresh rooted-slot word.
                                let a = Convert::to_string(heap, &scope, unsafe {
                                    Tagged::from_value_unchecked(lhs.read_unchecked())
                                })?
                                .erase();
                                // to_string of the sibling allocates: keep
                                // this side rooted until the concat reads it
                                // Safety: fresh word from to_string.
                                let a = scope.handle(unsafe { a.assume_valid(&*heap) });
                                // Safety: fresh rooted-slot word.
                                let b = Convert::to_string(heap, &scope, unsafe {
                                    Tagged::from_value_unchecked(rhs.read_unchecked())
                                })?
                                .erase();
                                // Safety: fresh word from to_string.
                                let b = scope.handle(unsafe { b.assume_valid(&*heap) });
                                // Safety: fresh rooted-slot word, returned
                                // for an immediate store.
                                Ok(unsafe {
                                    DenseString::concat(heap, &scope, a, b).read_unchecked()
                                })
                            })());
                            cache.set_acc(s);
                        } else {
                            let r = step_try!(heap.no_gc(|heap| {
                                let a = Convert::to_number(heap, lhs.as_tagged(heap))?;
                                let b = Convert::to_number(heap, rhs.as_tagged(heap))?;
                                // IEEE `-0 + -0` yields +0; the spec demands -0
                                let r = a + b;
                                let r = if r == 0.0 && a.is_sign_negative() && b.is_sign_negative()
                                {
                                    -0.0
                                } else {
                                    r
                                };
                                Ok::<_, VmError>(r)
                            }));
                            cache.set_acc(heap.new_number(&scope, r));
                        }
                        Step::Next
                    })
                }
            }
        }
        Opcode::Sub => {
            let acc_word = cache.acc(&*heap).erase();
            let other = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            let mut result = None;
            if let (Some(a), Some(b)) = (acc_word.to_i64(), other.to_i64())
                && let Some(r) = a.checked_sub(b)
                && Smi::in_range(r)
            {
                result = Some(Smi::new(r).encode());
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        acc_word,
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
            let acc_word = cache.acc(&*heap).erase();
            let other = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            let mut result = None;
            if let (Some(a), Some(b)) = (acc_word.to_i64(), other.to_i64())
                && let Some(r) = a.checked_mul(b)
                && Smi::in_range(r)
            {
                result = Some(Smi::new(r).encode());
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        acc_word,
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
            let acc_word = cache.acc(&*heap).erase();
            let other = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            let mut result = None;
            if let (Some(a), Some(b)) = (acc_word.to_i64(), other.to_i64())
                && b != 0
                && a % b == 0
                && let Some(r) = a.checked_div(b)
            {
                result = Some(Smi::new(r).encode());
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        acc_word,
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
            let acc_word = cache.acc(&*heap).erase();
            let other = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            let mut result = None;
            if let (Some(a), Some(b)) = (acc_word.to_i64(), other.to_i64())
                && b != 0
            {
                result = Some(Smi::new(a % b).encode());
            }
            match result {
                Some(v) => cache.set_acc(v),
                None => {
                    let v = step_try!(Runtime::numeric_op(
                        vm,
                        heap,
                        state,
                        acc_word,
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
            let acc_word = cache.acc(&*heap).erase();
            let other = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            let v = step_try!(Runtime::numeric_op(
                vm,
                heap,
                state,
                acc_word,
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
            let a = step_try!(cache.acc(&*heap).erase().to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(&*heap, &meta, ops.reg(0))
                    .erase()
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            cache.set_acc(Smi::new((a | b) as i64).encode());
            Step::Next
        }
        Opcode::BitwiseXor => {
            let a = step_try!(cache.acc(&*heap).erase().to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(&*heap, &meta, ops.reg(0))
                    .erase()
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            cache.set_acc(Smi::new((a ^ b) as i64).encode());
            Step::Next
        }
        Opcode::BitwiseAnd => {
            let a = step_try!(cache.acc(&*heap).erase().to_i64().ok_or(VmError::Type)) as i32;
            let b = step_try!(
                stack
                    .reg(&*heap, &meta, ops.reg(0))
                    .erase()
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as i32;
            cache.set_acc(Smi::new((a & b) as i64).encode());
            Step::Next
        }
        Opcode::ShiftLeft => {
            // ToInt32(lhs) << (ToUint32(rhs) & 31), truncated to int32
            let a = step_try!(Smi::decode(cache.acc(&*heap).erase()).ok_or(VmError::Type)).value()
                as i32;
            let b = step_try!(
                Smi::decode(stack.reg(&*heap, &meta, ops.reg(0)).erase()).ok_or(VmError::Type)
            )
            .value() as u32;
            cache.set_acc(Smi::new(a.wrapping_shl(b & 31) as i64).encode());
            Step::Next
        }
        Opcode::ShiftRight => {
            // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending
            let a = step_try!(Smi::decode(cache.acc(&*heap).erase()).ok_or(VmError::Type)).value()
                as i32;
            let b = step_try!(
                Smi::decode(stack.reg(&*heap, &meta, ops.reg(0)).erase()).ok_or(VmError::Type)
            )
            .value() as u32;
            cache.set_acc(Smi::new(a.wrapping_shr(b & 31) as i64).encode());
            Step::Next
        }
        Opcode::ShiftRightLogical => {
            // ToUint32(lhs) >>> (ToUint32(rhs) & 31): always non-negative
            let a = step_try!(cache.acc(&*heap).erase().to_i64().ok_or(VmError::Type)) as u32;
            let b = step_try!(
                stack
                    .reg(&*heap, &meta, ops.reg(0))
                    .erase()
                    .to_i64()
                    .ok_or(VmError::Type)
            ) as u32;
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
            if Convert::is_truthy(&*heap, cache.acc(&*heap)) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::JumpIfFalsy => {
            if !Convert::is_truthy(&*heap, cache.acc(&*heap)) {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::TestReferenceEqual => {
            let other = stack.reg(&*heap, &meta, ops.reg(0));
            let known = heap.known();
            cache.set_acc(if other.erase() == cache.acc(&*heap).erase() {
                // Safety: old-gen singleton word.
                unsafe { known.true_object.read_unchecked() }
            } else {
                // Safety: old-gen singleton word.
                unsafe { known.false_object.read_unchecked() }
            });
            Step::Next
        }
        Opcode::TestTypeof => {
            cache.set_acc(Runtime::type_of(&*heap, cache.acc(&*heap)));
            Step::Next
        }
        Opcode::Negate => {
            let acc_word = cache.acc(&*heap).erase();
            if let Some(v) = acc_word.to_i64() {
                cache.set_acc(if v == 0 {
                    state.handle_scope(|scope| heap.new_number(&scope, -0.0).erase())
                } else if v == Smi::MIN {
                    state.handle_scope(|scope| heap.new_number(&scope, -(v as f64)).erase())
                } else {
                    Smi::new(-v).encode()
                });
            } else {
                let n = step_try!(Runtime::to_numeric(vm, heap, state, acc_word));
                let Some(n) = n else {
                    return Step::PendingThrow;
                };
                cache.set_acc(state.handle_scope(|scope| {
                    let r = -n;
                    // preserve -0.0: `-0` must not fold into Smi 0
                    if r == 0.0 && r.is_sign_negative() {
                        heap.new_number(&scope, -0.0).erase()
                    } else {
                        heap.new_number(&scope, r).erase()
                    }
                }));
            }
            Step::Next
        }
        Opcode::InstanceOf => {
            let acc_word = cache.acc(&*heap).erase();
            let other = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            let r = step_try!(Runtime::instance_of(vm, heap, state, acc_word, other));
            let Some(r) = r else {
                return Step::PendingThrow;
            };
            cache.set_acc(Convert::boolean(&*heap, r));
            Step::Next
        }
        Opcode::Construct => {
            // ES 9.2.2 [[Construct]]: `new.target` (here the callee itself;
            // `super()`/Reflect.construct thread a different target later)
            // decides the receiver's prototype via GetPrototypeFromConstructor,
            // the callee runs with the receiver as `this`, and an object
            // result wins over the receiver. Derived class constructors get
            // TheHole instead: `this` is bound by their super() call.
            let (constructible, derived) = heap.no_gc(|heap| {
                let callee = stack.reg(heap, &meta, ops.reg(0));
                let Some(obj) = callee.as_heap_object() else {
                    return (false, false);
                };
                let kind = obj.as_ref().header.map.heap_ref(heap).kind();
                let derived = kind.is_class_constructor()
                    && function_kind_of(heap, callee)
                        .is_some_and(|k| k.is_derived_class_constructor());
                (kind.is_constructor(), derived)
            });
            if !constructible {
                return Step::Error(VmError::Type);
            }
            // a constructor proxy dispatches through its `construct`
            // trap (the target construct synthesizes its own receiver)
            if heap.no_gc(|heap| crate::proxy::is_proxy(heap, stack.reg(heap, &meta, ops.reg(0)))) {
                // Safety: fresh register word, no GC since.
                let callee_word = stack.reg(&*heap, &meta, ops.reg(0)).erase();
                let count = ops.reg_count(2);
                return match state.handle_scope(|scope| {
                    // staged: trap lookups may run user getters
                    let callee = scope.handle(unsafe { callee_word.assume_valid(&*heap) });
                    let argv: Vec<Value> =
                        stack.args(&meta, ops.reg_list(1), count).words().to_vec();
                    let staged = scope.stage_words(&argv);
                    let words: Vec<_> = staged
                        .words()
                        .iter()
                        // Safety: fresh rooted-slot words.
                        .map(|w| unsafe { Tagged::from_value_unchecked(*w) })
                        .collect();
                    crate::proxy::construct(
                        vm,
                        heap,
                        state,
                        // Safety: fresh rooted-slot word.
                        unsafe { Tagged::from_value_unchecked(callee.read_unchecked()) },
                        &words,
                        // Safety: fresh rooted-slot word.
                        unsafe { Tagged::from_value_unchecked(callee.read_unchecked()) },
                    )
                }) {
                    Ok(Coercion::Threw) => Step::PendingThrow,
                    Ok(Coercion::Value(v)) => {
                        cache.set_acc(v);
                        Step::Next
                    }
                    Err(err) => Step::Error(err),
                };
            }
            state.handle_scope(|scope| {
                let Some(callee) = scope.cast::<Object>(stack.reg(&*heap, &meta, ops.reg(0)))
                else {
                    return Step::Error(VmError::Type);
                };
                let (receiver, allocated) = if derived {
                    (
                        scope.handle(heap.known().the_hole.as_tagged(&*heap).erase_type()),
                        false,
                    )
                } else {
                    match Runtime::create_construct_receiver(vm, heap, state, callee) {
                        // Safety: fresh word from the receiver synthesis.
                        Ok(Some(r)) => (scope.handle(unsafe { r.assume_valid(&*heap) }), true),
                        Ok(None) => return Step::PendingThrow,
                        Err(err) => return Step::Error(err),
                    }
                };
                // the register list holds only arguments; the receiver is
                // synthesized and prepended
                let count = ops.reg_count(2);
                let mut args = Vec::with_capacity(count + 1);
                // Safety: fresh rooted-slot word.
                args.push(unsafe { receiver.read_unchecked() });
                args.extend_from_slice(stack.args(&meta, ops.reg_list(1), count).words());
                let staged = scope.stage_words(&args);
                // Safety: fresh rooted-slot words, consumed by the call.
                let callee_word = unsafe { callee.read_unchecked() };
                let result = match NativeContext::new(vm, heap, state).call_construct(
                    unsafe { Tagged::from_value_unchecked(callee_word) },
                    unsafe { Tagged::from_value_unchecked(callee_word) },
                    staged,
                ) {
                    Ok(r) => r,
                    Err(err) => return Step::Error(err),
                };
                if result == unsafe { heap.known().exception.read_unchecked() } {
                    return Step::PendingThrow;
                }
                cache.set_acc(
                    if Convert::is_primitive(&*heap, unsafe { result.assume_valid(&*heap) }) {
                        if allocated {
                            // Safety: fresh rooted-slot word.
                            unsafe { receiver.read_unchecked() }
                        } else {
                            // a derived constructor returned a primitive: only
                            // reachable via `return <primitive>` (ES 9.2.2.1)
                            return Step::Error(VmError::Type);
                        }
                    } else {
                        result
                    },
                );
                Step::Next
            })
        }
        Opcode::EqualStrict => {
            let other = stack.reg(&*heap, &meta, ops.reg(0));
            let r = Compare::strict_equal(&*heap, cache.acc(&*heap), other);
            cache.set_acc(Convert::boolean(&*heap, r));
            Step::Next
        }
        Opcode::Equal => {
            // IsLooselyEqual: objects are ToPrimitive'd (hint default)
            // first; the coercions run user code (valueOf/toString), so
            // the left result stays rooted and the right operand is
            // re-read from its register instead of a raw copy
            state.handle_scope(|scope| {
                let other_reg = ops.reg(0);
                let acc_word = cache.acc(&*heap).erase();
                let x = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    acc_word,
                    Hint::Default
                ));
                let x = match x {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let other_word = stack.reg(&*heap, &meta, other_reg).erase();
                let y = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    other_word,
                    Hint::Default
                ));
                let y = match y {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let r = step_try!(Compare::equal(
                    &*heap,
                    x.as_tagged(&*heap),
                    y.as_tagged(&*heap)
                ));
                cache.set_acc(Convert::boolean(&*heap, r));
                Step::Next
            })
        }
        Opcode::LessThan => {
            // Abstract Relational Comparison: objects ToPrimitive'd with
            // hint Number; the coercions run user code (valueOf), so the
            // left result stays rooted and the right operand is re-read
            // from its register instead of a raw copy
            state.handle_scope(|scope| {
                let other_reg = ops.reg(0);
                let acc_word = cache.acc(&*heap).erase();
                let x = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    acc_word,
                    Hint::Number
                ));
                let x = match x {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let other_word = stack.reg(&*heap, &meta, other_reg).erase();
                let y = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    other_word,
                    Hint::Number
                ));
                let y = match y {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let r = step_try!(Compare::less_than(
                    &*heap,
                    x.as_tagged(&*heap),
                    y.as_tagged(&*heap)
                ));
                cache.set_acc(Convert::boolean(&*heap, r));
                Step::Next
            })
        }
        Opcode::LessThanOrEqual => {
            // Abstract Relational Comparison: objects ToPrimitive'd with
            // hint Number; the coercions run user code (valueOf), so the
            // left result stays rooted and the right operand is re-read
            // from its register instead of a raw copy
            state.handle_scope(|scope| {
                let other_reg = ops.reg(0);
                let acc_word = cache.acc(&*heap).erase();
                let x = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    acc_word,
                    Hint::Number
                ));
                let x = match x {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let other_word = stack.reg(&*heap, &meta, other_reg).erase();
                let y = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    other_word,
                    Hint::Number
                ));
                let y = match y {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let r = step_try!(Compare::less_than_or_equal(
                    &*heap,
                    x.as_tagged(&*heap),
                    y.as_tagged(&*heap)
                ));
                cache.set_acc(Convert::boolean(&*heap, r));
                Step::Next
            })
        }
        Opcode::GreaterThan => {
            // Abstract Relational Comparison: objects ToPrimitive'd with
            // hint Number; the coercions run user code (valueOf), so the
            // left result stays rooted and the right operand is re-read
            // from its register instead of a raw copy
            state.handle_scope(|scope| {
                let other_reg = ops.reg(0);
                let acc_word = cache.acc(&*heap).erase();
                let x = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    acc_word,
                    Hint::Number
                ));
                let x = match x {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let other_word = stack.reg(&*heap, &meta, other_reg).erase();
                let y = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    other_word,
                    Hint::Number
                ));
                let y = match y {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let r = step_try!(Compare::greater_than(
                    &*heap,
                    x.as_tagged(&*heap),
                    y.as_tagged(&*heap)
                ));
                cache.set_acc(Convert::boolean(&*heap, r));
                Step::Next
            })
        }
        Opcode::GreaterThanOrEqual => {
            // Abstract Relational Comparison: objects ToPrimitive'd with
            // hint Number; the coercions run user code (valueOf), so the
            // left result stays rooted and the right operand is re-read
            // from its register instead of a raw copy
            state.handle_scope(|scope| {
                let other_reg = ops.reg(0);
                let acc_word = cache.acc(&*heap).erase();
                let x = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    acc_word,
                    Hint::Number
                ));
                let x = match x {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let other_word = stack.reg(&*heap, &meta, other_reg).erase();
                let y = step_try!(Runtime::to_primitive(
                    vm,
                    heap,
                    state,
                    other_word,
                    Hint::Number
                ));
                let y = match y {
                    Coercion::Threw => return Step::PendingThrow,
                    // Safety: fresh word from the coercion.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let r = step_try!(Compare::greater_than_or_equal(
                    &*heap,
                    x.as_tagged(&*heap),
                    y.as_tagged(&*heap)
                ));
                cache.set_acc(Convert::boolean(&*heap, r));
                Step::Next
            })
        }
        Opcode::Throw | Opcode::ReThrow => Step::Throw(cache.acc(&*heap).erase()),
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
                Ok(v) if v == unsafe { heap.known().exception.read_unchecked() } => {
                    Step::PendingThrow
                }
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
            // Safety: fresh register word, no GC since.
            let callee_word = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            // a callable proxy dispatches through its `apply` trap (or
            // a nested call of the target)
            if heap.no_gc(|heap| {
                crate::proxy::is_proxy(heap, unsafe { callee_word.assume_valid(heap) })
            }) {
                return match state.handle_scope(|scope| {
                    // staged: trap lookups may run user getters
                    let argv: Vec<Value> =
                        stack.args(&meta, ops.reg_list(1), count).words().to_vec();
                    let staged = scope.stage_words(&argv);
                    let words: Vec<_> = staged
                        .words()
                        .iter()
                        // Safety: fresh rooted-slot words.
                        .map(|w| unsafe { Tagged::from_value_unchecked(*w) })
                        .collect();
                    crate::proxy::apply(
                        vm,
                        heap,
                        state,
                        // Safety: register word read above; the callee
                        // roots it before its first allocation.
                        unsafe { Tagged::from_value_unchecked(callee_word) },
                        &words,
                    )
                }) {
                    Ok(Coercion::Threw) => Step::PendingThrow,
                    Ok(Coercion::Value(v)) => {
                        cache.set_acc(v);
                        Step::Next
                    }
                    Err(err) => Step::Error(err),
                };
            }
            match classify_callee(&*heap, callee_word) {
                Callee::NotCallable => Step::Error(VmError::Type),
                Callee::Native(idx) => {
                    let f = vm.native(NativeIndex(idx));
                    let mut nctx = NativeContext::new(vm, heap, state);
                    let result = f(&mut nctx, stack.args(&meta, ops.reg_list(1), count));
                    match result {
                        // Safety: old-gen singleton word.
                        Ok(v) if v == unsafe { heap.known().exception.read_unchecked() } => {
                            Step::PendingThrow
                        }
                        Ok(v) => {
                            cache.set_acc(v);
                            Step::Next
                        }
                        Err(err) => Step::Error(err),
                    }
                }
                Callee::Bytecode(register_count, kind) => {
                    if kind.is_class_constructor() {
                        return Step::Error(VmError::Type);
                    }
                    let context = closure_context(
                        &*heap,
                        // Safety: register word read above, no GC since.
                        unsafe { Tagged::from_value_unchecked(callee_word) },
                    )
                    .erase();
                    // Safety: old-gen singleton word.
                    let undefined = unsafe { heap.known().undefined.read_unchecked() };
                    let formal_min = heap.no_gc(|heap| {
                        unsafe { callee_word.assume_valid(heap) }
                            .as_heap_object()
                            .and_then(|obj| obj.as_ref().callable_info(heap))
                            .map(|info| info.formal_parameter_count() + 1)
                            .unwrap_or(1)
                    });
                    let callee = step_try!(stack.push_frame(
                        meta,
                        pc,
                        // Safety: register word read above, no GC since.
                        unsafe { Tagged::from_value_unchecked(callee_word) },
                        register_count,
                        // Safety: closure-context word read above, no GC
                        // since.
                        unsafe { Tagged::from_value_unchecked(context) },
                        ops.reg_list(1),
                        count,
                        // Safety: old-gen singleton word.
                        unsafe { Tagged::from_value_unchecked(undefined) },
                        formal_min,
                    ));
                    cache.load(stack, callee, heap);
                    Step::Next
                }
            }
        }
        Opcode::LoadNamedProperty => {
            // Safety: fresh register word, no GC since.
            let receiver_word = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            // proxies run their `get` trap outside any no-GC scope
            if heap.no_gc(|heap| {
                crate::proxy::is_proxy(heap, unsafe { receiver_word.assume_valid(heap) })
            }) {
                // Safety: name held rooted by the owning constants pool.
                let name_word = callable_name(&*heap, stack, &meta, ops.idx(1)).erase();
                return match step_try!(crate::proxy::get(
                    vm,
                    heap,
                    state,
                    // Safety: fresh register word, no GC since.
                    unsafe { Tagged::from_value_unchecked(receiver_word) },
                    unsafe { Tagged::from_value_unchecked(receiver_word) },
                    // Safety: constants-pool word, rooted by the callable.
                    unsafe { Tagged::from_value_unchecked(name_word) },
                )) {
                    Coercion::Threw => Step::PendingThrow,
                    Coercion::Value(v) => {
                        cache.set_acc(v);
                        Step::Next
                    }
                };
            }
            let outcome = step_try!(heap.no_gc(|heap| {
                let name = callable_name(heap, stack, &meta, ops.idx(1));
                load_outcome(heap, unsafe { receiver_word.assume_valid(heap) }, name)
                    .map(LoadResult::of)
            }));
            match outcome {
                LoadResult::Value(v) => cache.set_acc(v),
                LoadResult::Getter(getter) => {
                    let receiver = stack.reg(&*heap, &meta, ops.reg(0)).erase();
                    match step_try!(call_value(
                        vm,
                        state,
                        heap,
                        stack,
                        cache,
                        meta,
                        pc,
                        getter,
                        &[receiver]
                    )) {
                        Called::Frame => {}
                        // non-callable getter: the load yields undefined
                        Called::NotCallable => {
                            // Safety: old-gen singleton word.
                            cache.set_acc(unsafe { heap.known().undefined.read_unchecked() });
                        }
                        Called::Immediate(v) => cache.set_acc(v),
                        Called::Threw => return Step::PendingThrow,
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
            // Safety: fresh register word, no GC since.
            let receiver_word = stack.reg(&*heap, &meta, ops.reg(0)).erase();
            // proxies run their `set` trap outside any no-GC scope
            if heap.no_gc(|heap| {
                crate::proxy::is_proxy(heap, unsafe { receiver_word.assume_valid(heap) })
            }) {
                // fresh acc word read for the trap call
                let value_word = cache.acc(&*heap).erase();
                // Safety: name held rooted by the owning constants pool.
                let name_word = callable_name(&*heap, stack, &meta, ops.idx(1)).erase();
                return match step_try!(crate::proxy::set(
                    vm,
                    heap,
                    state,
                    // Safety: fresh register word, no GC since.
                    unsafe { Tagged::from_value_unchecked(receiver_word) },
                    // Safety: constants-pool word, rooted by the callable.
                    unsafe { Tagged::from_value_unchecked(name_word) },
                    // Safety: fresh acc word, no GC since.
                    unsafe { Tagged::from_value_unchecked(value_word) },
                    unsafe { Tagged::from_value_unchecked(receiver_word) },
                )) {
                    Coercion::Threw => Step::PendingThrow,
                    Coercion::Value(_) => Step::Next,
                };
            }
            state.handle_scope(|scope| -> Step {
                let outcome = step_try!(heap.no_gc(|heap| {
                    let name = callable_name(heap, stack, &meta, ops.idx(1));
                    unsafe { receiver_word.assume_valid(heap) }.store_lookup(
                        heap,
                        &scope,
                        name,
                        cache.acc(heap),
                        semantics,
                    )
                }));
                step_try!(apply_store_outcome(
                    vm,
                    heap,
                    state,
                    stack,
                    cache,
                    meta,
                    pc,
                    receiver_word,
                    outcome,
                ));
                Step::Next
            })
        }
        Opcode::LoadKeyedProperty => {
            // the key coercion allocates (to_property_key interns): the
            // receiver must stay rooted across it
            state.handle_scope(|scope| {
                let receiver = scope.handle(stack.reg(&*heap, &meta, ops.reg(0)));
                let raw_key = cache.acc(&*heap).erase();
                let Some(key) = step_try!(Runtime::to_property_key(
                    vm,
                    heap,
                    state,
                    // Safety: fresh acc word, no GC since.
                    unsafe { Tagged::from_value_unchecked(raw_key) },
                )) else {
                    return Step::PendingThrow;
                };
                // root the word in the acc register: the tagged result
                // anchors the `&mut` borrow
                let key = key.erase();
                cache.set_acc(key);
                // re-read through the handle: the coercion above allocated
                // Safety: fresh rooted-slot word.
                let receiver_word = unsafe { receiver.read_unchecked() };
                // string primitives expose their code units as index
                // properties (ES 5.4.3.1): `"ab"[1]` is "b". The one-unit
                // string is allocated fresh — string comparison is by
                // content, so identity is unobservable. Out-of-range and
                // non-string receivers fall through to the ordinary path.
                let string_index = heap.no_gc(|heap| {
                    match classify_key(heap, unsafe { key.assume_valid(heap) }) {
                        Ok(Key::Element(i))
                            if unsafe { receiver_word.assume_valid(heap) }
                                .get_as::<DenseString>()
                                .is_some() =>
                        {
                            Some(i)
                        }
                        _ => None,
                    }
                });
                if let Some(i) = string_index {
                    let unit = state.handle_scope(|scope| {
                        crate::builtins::intrinsics::string_char_at(heap, &scope, receiver_word, i)
                    });
                    if let Some(unit) = unit {
                        cache.set_acc(unit);
                        return Step::Next;
                    }
                }
                // proxies run their `get` trap outside any no-GC scope
                if heap.no_gc(|heap| {
                    crate::proxy::is_proxy(heap, unsafe { receiver_word.assume_valid(heap) })
                }) {
                    // fresh acc word (the interned key) read for the trap call
                    let key_word = cache.acc(&*heap).erase();
                    return match step_try!(crate::proxy::get(
                        vm,
                        heap,
                        state,
                        // Safety: fresh rooted-slot word.
                        unsafe { Tagged::from_value_unchecked(receiver_word) },
                        unsafe { Tagged::from_value_unchecked(receiver_word) },
                        // Safety: fresh acc word, no GC since.
                        unsafe { Tagged::from_value_unchecked(key_word) },
                    )) {
                        Coercion::Threw => Step::PendingThrow,
                        Coercion::Value(v) => {
                            cache.set_acc(v);
                            Step::Next
                        }
                    };
                }
                let outcome = step_try!(heap.no_gc(|heap| {
                    let receiver = unsafe { receiver_word.assume_valid(heap) };
                    match classify_key(heap, cache.acc(heap))? {
                        Key::Element(i) => {
                            match receiver
                                .as_heap_object()
                                .and_then(|obj| obj.as_ref().element_value(heap, i))
                            {
                                Some(v) => Ok(LoadResult::Value(v.erase())),
                                // past the end, a hole, or a non-array receiver:
                                // fall back to an ordinary property lookup
                                None => {
                                    load_outcome(heap, receiver, Tagged::from(Smi::new(i as i64)))
                                        .map(LoadResult::of)
                                }
                            }
                        }
                        Key::Name(name) => load_outcome(heap, receiver, name).map(LoadResult::of),
                    }
                }));
                match outcome {
                    LoadResult::Value(v) => cache.set_acc(v),
                    LoadResult::Getter(getter) => {
                        // Safety: fresh rooted-slot word.
                        let receiver = unsafe { receiver.read_unchecked() };
                        match step_try!(call_value(
                            vm,
                            state,
                            heap,
                            stack,
                            cache,
                            meta,
                            pc,
                            getter,
                            &[receiver]
                        )) {
                            Called::Frame => {}
                            Called::NotCallable => {
                                // Safety: old-gen singleton word.
                                cache.set_acc(unsafe { heap.known().undefined.read_unchecked() });
                            }
                            Called::Immediate(v) => cache.set_acc(v),
                            Called::Threw => return Step::PendingThrow,
                        }
                    }
                }
                Step::Next
            })
        }
        Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyNoShadow => {
            // the key coercion allocates (to_property_key interns): the
            // receiver must stay rooted across it
            state.handle_scope(|scope| {
                let semantics = match op {
                    Opcode::StoreKeyedPropertyNoShadow => StoreSemantics::WriteThrough,
                    _ => StoreSemantics::Shadow,
                };
                let receiver = scope.handle(stack.reg(&*heap, &meta, ops.reg(0)));
                let raw = stack.reg(&*heap, &meta, ops.reg(1)).erase();
                let Some(key) = step_try!(Runtime::to_property_key(
                    vm,
                    heap,
                    state,
                    // Safety: fresh register word, no GC since.
                    unsafe { Tagged::from_value_unchecked(raw) },
                )) else {
                    return Step::PendingThrow;
                };
                // root the word in the key register: the tagged result
                // anchors the `&mut` borrow
                let key = key.erase();
                stack.set_reg(&meta, ops.reg(1), key);
                // the receiver is re-read through its handle at every
                // use: the key coercion above allocated and may have moved
                // it (a raw snapshot would go stale)
                // Safety: fresh rooted-slot word.
                let receiver_word = unsafe { receiver.read_unchecked() };
                // proxies run their `set` trap outside any no-GC scope
                // (including array-element stores)
                if heap.no_gc(|heap| {
                    crate::proxy::is_proxy(heap, unsafe { receiver_word.assume_valid(heap) })
                }) {
                    // fresh acc word read for the trap call
                    let value_word = cache.acc(&*heap).erase();
                    return match step_try!(crate::proxy::set(
                        vm,
                        heap,
                        state,
                        // Safety: fresh rooted-slot word.
                        unsafe { Tagged::from_value_unchecked(receiver_word) },
                        // Safety: key fresh from the coercion, rooted by
                        // the callee before its first allocation.
                        unsafe { Tagged::from_value_unchecked(key) },
                        // Safety: fresh acc word, no GC since.
                        unsafe { Tagged::from_value_unchecked(value_word) },
                        unsafe { Tagged::from_value_unchecked(receiver_word) },
                    )) {
                        Coercion::Threw => Step::PendingThrow,
                        Coercion::Value(_) => Step::Next,
                    };
                }
                // classify the coerced key inside a no-GC scope: the
                // Key borrows the heap, so the name arm is rooted
                // before it escapes
                let mut name_slot: Option<Handle<'_, SlotName>> = None;
                let element = step_try!(heap.no_gc(|heap| -> Result<Option<usize>, VmError> {
                    // Safety: register word, fresh at entry.
                    Ok(
                        match classify_key(heap, unsafe { key.assume_valid(heap) })? {
                            Key::Element(i) => Some(i),
                            Key::Name(name) => {
                                name_slot = Some(scope.handle(name));
                                None
                            }
                        },
                    )
                }));
                match element {
                    Some(i) => {
                        let is_array = heap.no_gc(|heap| {
                            let Some(obj) =
                                unsafe { receiver_word.assume_valid(heap) }.as_heap_object()
                            else {
                                return false;
                            };
                            obj.as_ref().is_array(heap)
                        });
                        if is_array {
                            // fresh acc word read for the store
                            let value_word = cache.acc(&*heap).erase();
                            step_try!(state.handle_scope(|scope| {
                                store_array_element(
                                    heap,
                                    &scope,
                                    // Safety: fresh rooted-slot word.
                                    unsafe { Tagged::from_value_unchecked(receiver_word) },
                                    i,
                                    // Safety: fresh acc word, no GC since.
                                    unsafe { Tagged::from_value_unchecked(value_word) },
                                )
                            }));
                        } else {
                            // numeric property on a non-array receiver
                            let outcome = step_try!(heap.no_gc(|heap| {
                                unsafe { receiver_word.assume_valid(heap) }.store_lookup(
                                    heap,
                                    &scope,
                                    Tagged::from(Smi::new(i as i64)),
                                    cache.acc(heap),
                                    semantics,
                                )
                            }));
                            step_try!(apply_store_outcome(
                                vm,
                                heap,
                                state,
                                stack,
                                cache,
                                meta,
                                pc,
                                receiver_word,
                                outcome,
                            ));
                        }
                    }
                    None => {
                        let name = name_slot.expect("name classified above");
                        let outcome = step_try!(heap.no_gc(|heap| {
                            unsafe { receiver_word.assume_valid(heap) }.store_lookup(
                                heap,
                                &scope,
                                name.as_tagged(heap),
                                cache.acc(heap),
                                semantics,
                            )
                        }));
                        step_try!(apply_store_outcome(
                            vm,
                            heap,
                            state,
                            stack,
                            cache,
                            meta,
                            pc,
                            receiver_word,
                            outcome,
                        ));
                    }
                }
                Step::Next
            })
        }
        Opcode::CreateEmptyObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().object_initial_map;
                heap.new_object(&scope, map, GcSlice::EMPTY).erase()
            });
            cache.set_acc(obj);
            Step::Next
        }
        Opcode::CreateEmptyArrayLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().js_array_map;
                heap.new_object(&scope, map, GcSlice::EMPTY).erase()
            });
            cache.set_acc(obj);
            Step::Next
        }
        Opcode::CreateClosure => {
            // constants[idx] is the shared callable-info template (SFI-like);
            // the closure captures the current frame's context
            let info = step_try!(heap.no_gc(|heap| {
                cache
                    .constants_ref(heap)
                    .at(heap, ops.idx(0))
                    .get_as::<CallableInfoObject>()
                    .map(|r| r.into_tagged().erase())
                    .ok_or(VmError::Type)
            }));
            let context = step_try!(frame_context(stack, &meta));
            let obj = state.handle_scope(|scope| {
                // Safety: fresh anchored constant read, no GC since.
                let Some(info) =
                    scope.cast::<CallableInfoObject>(unsafe { info.assume_valid(&*heap) })
                else {
                    return Err(VmError::Type);
                };
                Runtime::create_closure(
                    heap,
                    &scope,
                    info,
                    // Safety: fresh register-slot word, rooted by the
                    // callee before its first allocation.
                    unsafe { Tagged::from_value_unchecked(context) },
                )
                .map(|r| r.erase())
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
            // Safety: old-gen root-slot word.
            let global = unsafe { heap.known().global_object.read_unchecked() };
            let lookup = heap.no_gc(|heap| {
                let name = callable_name(heap, stack, &meta, ops.idx(0));
                match unsafe { global.assume_valid(heap) }.lookup(heap, name) {
                    Lookup::Data { slot, .. } => GlobalLoad::Data(slot.inner()),
                    Lookup::Accessor { pair, .. } => GlobalLoad::Getter(pair.get.inner()),
                    Lookup::NotFound => GlobalLoad::Missing,
                }
            });
            match lookup {
                GlobalLoad::Data(v) => cache.set_acc(v),
                GlobalLoad::Getter(getter) => {
                    // Safety: old-gen singleton word.
                    if getter == unsafe { heap.known().undefined.read_unchecked() } {
                        // Safety: old-gen singleton word.
                        cache.set_acc(unsafe { heap.known().undefined.read_unchecked() });
                    } else {
                        match step_try!(call_value(
                            vm,
                            state,
                            heap,
                            stack,
                            cache,
                            meta,
                            pc,
                            getter,
                            &[global]
                        )) {
                            Called::Frame => {}
                            Called::NotCallable => {
                                // Safety: old-gen singleton word.
                                cache.set_acc(unsafe { heap.known().undefined.read_unchecked() });
                            }
                            Called::Immediate(v) => cache.set_acc(v),
                            Called::Threw => return Step::PendingThrow,
                        }
                    }
                }
                GlobalLoad::Missing => {
                    if op == Opcode::LoadGlobal {
                        // unresolvable reference: GetValue throws ReferenceError
                        return Step::Error(VmError::Reference);
                    }
                    // Safety: old-gen singleton word.
                    cache.set_acc(unsafe { heap.known().undefined.read_unchecked() });
                }
            }
            // TODO: full semantics: lookup the script-context table first
            // (lexical globals), throw ReferenceError on unresolved loads;
            // the global-object property path is the only one implemented
            Step::Next
        }
        Opcode::StoreGlobal => {
            // Safety: old-gen root-slot word.
            let global = unsafe { heap.known().global_object.read_unchecked() };
            state.handle_scope(|scope| -> Step {
                let outcome = step_try!(heap.no_gc(|heap| {
                    let name = callable_name(heap, stack, &meta, ops.idx(0));
                    unsafe { global.assume_valid(heap) }.store_lookup(
                        heap,
                        &scope,
                        name,
                        cache.acc(heap),
                        StoreSemantics::WriteThrough,
                    )
                }));
                step_try!(apply_store_outcome(
                    vm, heap, state, stack, cache, meta, pc, global, outcome,
                ));
                Step::Next
            })
        }
        Opcode::CreateFunctionContext => {
            // constants[idx] is the scope's shared ScopeInfo (its `names`
            // array is parallel to the context's slots)
            let scope_info = step_try!(heap.no_gc(|heap| {
                cache
                    .constants_ref(heap)
                    .at(heap, ops.idx(0))
                    .get_as::<ScopeInfo>()
                    .map(|r| r.into_tagged().erase())
                    .ok_or(VmError::Type)
            }));
            let count = step_try!(heap.no_gc(|heap| {
                unsafe { scope_info.assume_valid(heap) }
                    .get_as::<ScopeInfo>()
                    .map(|r| r.as_ref().names.heap_ref(heap).len())
                    .ok_or(VmError::Type)
            }));
            // Safety: old-gen singleton word.
            let hole = unsafe { heap.known().the_hole.read_unchecked() };
            let values = vec![hole; count];
            let outer = step_try!(frame_context(stack, &meta));
            let ctx = state.handle_scope(|scope| {
                // Safety: fresh register-slot word, no GC since.
                let outer = scope
                    .cast::<Context>(unsafe { outer.assume_valid(&*heap) })
                    .expect("frame context slot holds a Context");
                // Safety: fresh anchored constant read, no GC since.
                let scope_info = scope
                    .cast::<ScopeInfo>(unsafe { scope_info.assume_valid(&*heap) })
                    .expect("constants slot holds a ScopeInfo");
                let values = scope.stage_words(&values);
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info,
                })
                .erase()
            });
            cache.set_acc(ctx);
            Step::Next
        }
        Opcode::CreateBlockContext => {
            let count = ops.uimm(0) as usize;
            // Safety: old-gen singleton word.
            let hole = unsafe { heap.known().the_hole.read_unchecked() };
            let values = vec![hole; count];
            let outer = step_try!(frame_context(stack, &meta));
            let ctx = state.handle_scope(|scope| {
                // Safety: fresh register-slot word, no GC since.
                let outer = scope
                    .cast::<Context>(unsafe { outer.assume_valid(&*heap) })
                    .expect("frame context slot holds a Context");
                let values = scope.stage_words(&values);
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<Context>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info: heap.known().empty_scope_info,
                })
                .erase()
            });
            cache.set_acc(ctx);
            Step::Next
        }
        Opcode::PushContext => {
            let old = step_try!(frame_context(stack, &meta));
            stack.set_reg(&meta, ops.reg(0), old);
            step_try!(set_frame_context(stack, &meta, cache.acc(&*heap)));
            Step::Next
        }
        Opcode::PopContext => {
            let context = stack.reg(&*heap, &meta, ops.reg(0));
            step_try!(set_frame_context(stack, &meta, context));
            Step::Next
        }
        Opcode::ThrowReferenceErrorIfHole => {
            if cache.acc(&*heap).erase() == unsafe { heap.known().the_hole.read_unchecked() } {
                return Step::Error(VmError::Reference);
            }
            Step::Next
        }
        Opcode::LoadContextSlot => {
            let depth = ops.uimm(1);
            let v = step_try!(heap.no_gc(|heap| {
                let mut context = stack
                    .context_slot(&meta)
                    .read(heap)
                    .get_as::<Context>()
                    .ok_or(VmError::Type)?;
                for _ in 0..depth {
                    context = context.as_ref().outer.heap_ref(heap).ok_or(VmError::Type)?;
                }
                Ok(context
                    .slots
                    .heap_ref(heap)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .inner())
            }));
            cache.set_acc(v);
            Step::Next
        }
        Opcode::StoreContextSlot => {
            step_try!(heap.no_gc(|heap| {
                let mut context = stack
                    .context_slot(&meta)
                    .read(heap)
                    .get_as::<Context>()
                    .ok_or(VmError::Type)?;
                for _ in 0..ops.uimm(1) {
                    context = context.as_ref().outer.heap_ref(heap).ok_or(VmError::Type)?;
                }
                // Safety: fresh anchored slot read.
                let host = context.clone().into_tagged().erase();
                context
                    .slots
                    .heap_ref(heap)
                    .as_ref()
                    .element_slot(ops.idx(0))
                    .set(heap, host, cache.acc(heap));
                Ok(())
            }));
            Step::Next
        }
        Opcode::LdaNewTarget => {
            cache.set_acc(stack.new_target_slot(&meta).inner());
            Step::Next
        }
        Opcode::LdaCurrentClosure => {
            cache.set_acc(stack.callable_slot(&meta).inner());
            Step::Next
        }
        Opcode::LdaContext => {
            let ctx = step_try!(frame_context(stack, &meta));
            cache.set_acc(ctx);
            Step::Next
        }
        Opcode::LdaHole => {
            // Safety: old-gen singleton word.
            cache.set_acc(unsafe { heap.known().the_hole.read_unchecked() });
            Step::Next
        }
        Opcode::JumpIfNotUndefined => {
            if cache.acc(&*heap).erase() != unsafe { heap.known().undefined.read_unchecked() } {
                cache.set_pc(jump_target(pc, ops.imm(0)));
            }
            Step::Next
        }
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}
