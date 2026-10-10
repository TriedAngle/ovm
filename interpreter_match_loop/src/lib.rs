use bytecode::{Opcode, Operands, decode, jump_target};
use vm_core::proxy::Proxy;

use vm_core::cold::{
    add, apply_store_outcome, bitwise, compare, construct, global_load, keyed_load, keyed_load_imm,
    keyed_store, named_load, numeric, store_global, store_named, store_named_no_shadow,
};
use vm_core::ic::StoreOutcomeKind;
use vm_core::ic::{CallProbe, CallRecord, ElementHit, Hit, InlineCache, MonoProbe, StoreHit};
use vm_core::interpreter::{Ctx, Unwind, unwind};
use vm_core::{
    CallTarget, CallableInfoObject, Callee, Coercion, Compare, ContextInit, ContextObject, Convert,
    FixedArray, FrameMeta, Handle, HandleSlice, Heap, HostCtx, Key, LoadOutcome, Lookup,
    NativeIndex, Object, Recv, Register, ScopeInfo, SlotName, Smi, StoreSemantics, Tagged,
    Termination, ThreadState, VM, Value, VmError,
};

/// The per-instruction control flow of the dispatch loop (the error
/// channel is the exception sentinel, never a variant).
enum Flow {
    /// advance to `next_pc`; the arm cannot have moved the bytecode
    /// object (no allocation, no call): the cached code stays valid
    Next,
    /// advance to `next_pc` after re-checking the code pointer (the arm
    /// may have allocated or run nested JS)
    Sync,
    /// transfer to a new pc in the same frame (jump arms)
    Jump(usize),
    /// the current frame changed; resume at the carried pc
    Reframe(usize),
    Return,
    /// the exception sentinel was produced (pending set): unwind
    Threw,
}

/// Fold a `Result`-returning primitive helper into the sentinel channel:
/// materialize the error and yield `Flow::Threw`.
macro_rules! fold {
    ($ctx:expr, $e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => {
                $ctx.raise_tag(e);
                return Flow::Threw;
            }
        }
    };
}

macro_rules! throw_err {
    ($ctx:expr, $e:expr) => {{
        $ctx.raise_tag($e);
        return Flow::Threw;
    }};
}

pub fn execute<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ThreadState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'_, Value>>,
) -> Result<Tagged<'a, Value>, VmError> {
    let stack = state.stack();

    let saved_top = stack.top();
    let was_active = state.is_frame_active();
    let outer = was_active.then(|| state.frame_base());

    // one machine-stack budget per execution entry: blocking calls and
    // nested `execute`s only deepen the SP against it (the become
    // interpreter recomputes its limit the same way at each `enter`)
    let probe = 0u8;
    let stack_limit = (&probe as *const u8 as usize).saturating_sub(6 * 1024 * 1024);

    state.handle_scope(|scope| {
        let result = start(vm, heap, state, callable, args, new_target, stack_limit)?;
        let rooted = scope.handle(result);

        stack.set_top(saved_top);
        if let Some(outer) = outer {
            state.set_frame_base(outer);
        } else {
            state.set_frame_active(false);
        }
        Ok(rooted.as_tagged(heap))
    })
}

fn start<'b>(
    vm: &'b VM,
    heap: &'b mut Heap,
    state: &'b ThreadState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'b, Value>>,
    stack_limit: usize,
) -> Result<Tagged<'b, Value>, VmError> {
    if Proxy::is_proxy(heap, callable.as_tagged(heap).erase()) {
        return state.handle_scope(|scope| {
            let args = args.as_args();
            let result = match new_target {
                None => Proxy::apply(vm, heap, state, callable.erase(), args),
                Some(nt) => {
                    Proxy::construct(vm, heap, state, callable.erase(), args.slice_from(1), nt)
                }
            };
            match result {
                Ok(Coercion::Value(v)) => Ok(scope.handle(v).as_tagged(heap)),
                Ok(Coercion::Threw) => Ok(heap.known().exception.as_tagged(heap).erase()),
                Err(e) => Err(e),
            }
        });
    }

    match callable.as_tagged(heap).erase().call_target(heap) {
        None => Err(VmError::Type),
        // the proxy dispatch above already intercepted these
        Some(CallTarget::Proxy(_)) => Err(VmError::Type),
        Some(CallTarget::Native(rt)) => {
            let f = vm.runtime(rt);
            let nctx = HostCtx::new(vm, heap, state);
            Ok(f(nctx, new_target, args.as_args()))
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            kind,
            ..
        }) => {
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
                let new_target_value = match &new_target {
                    Some(nt) => nt.as_tagged(heap).erase(),
                    None => heap.known().undefined.as_tagged(heap).erase(),
                };
                stack.push_frame(
                    heap,
                    FrameMeta::ROOT,
                    Callee {
                        callable: target.erase(),
                        info,
                        context: context.erase(),
                    },
                    new_target_value,
                    stack.stage_slice(args.as_tagged())?,
                )?
            };
            state.set_frame_base(frame.base);
            state.set_frame_active(true);
            // the interpreter context: raw pointers + anchors shared by
            // every cold body in vm_core::cold
            let ctx = unsafe { Ctx::new(vm, heap, state, frame.base, stack_limit) };
            dispatch(&ctx)
        }
    }
}

/// Run a just-pushed callee frame to completion: the blocking-call
/// primitive mirroring the become interpreter's call trampoline. The
/// callee's value is returned (the exception sentinel when it threw);
/// the caller's current frame is restored on every path.
fn run_callee<'a>(ctx: &Ctx<'a>) -> Result<Tagged<'a, Value>, VmError> {
    let stack = ctx.stack();
    // the machine-stack guard: one Rust frame per JS call
    if ctx.stack_overflowed() {
        let caller = stack.pop_frame(ctx.frame_base());
        ctx.set_frame_base(caller.base);
        return Err(VmError::StackOverflow);
    }
    // a fresh context per callee (its accumulator starts undefined: the
    // push seeded the callee's slot)
    let value = {
        let callee_ctx = unsafe { ctx.child(ctx.frame_base()) };
        dispatch(&callee_ctx)?
    };
    // the callee's anchor `Return` left its frame pushed with the value as
    // the dispatch result: pop back into the caller
    let caller = stack.pop_frame(ctx.frame_base());
    ctx.set_frame_base(caller.base);
    Ok(value)
}

/// The text of an interned `SlotName` (empty for non-string names).
fn begin_termination(heap: &Heap, state: &ThreadState) -> Flow {
    state.set_termination(Termination::Shutdown);
    let undefined = heap.known().undefined.as_tagged(heap);
    state.set_pending_exception(undefined);
    Flow::Threw
}

fn dispatch<'a>(ctx: &Ctx<'a>) -> Result<Tagged<'a, Value>, VmError> {
    let mut frame_base = ctx.frame_base();
    let acc = ctx.acc_slot();
    acc.store(ctx.undefined_word());

    let mut code = code_bytes(ctx);
    let mut pc = 0usize;

    loop {
        let (op, ops, next_pc) = decode(code, pc);
        let result = step(ctx, pc, next_pc, frame_base, acc, ops, op);
        match result {
            Flow::Next => pc = next_pc,
            Flow::Jump(target) => pc = target,
            Flow::Sync => {
                pc = next_pc;
                code = code_bytes(ctx);
            }
            Flow::Reframe(resume) => {
                frame_base = ctx.frame_base();
                code = code_bytes(ctx);
                pc = resume;
                acc.store(ctx.undefined_word());
            }
            Flow::Return => return Ok(acc.get(ctx.heap())),
            Flow::Threw => match unwind(ctx, pc) {
                Unwind::Caught { pc: handler, ex } => {
                    acc.store(ex);
                    frame_base = ctx.frame_base();
                    code = code_bytes(ctx);
                    pc = handler;
                }
                Unwind::Escaped => return Ok(ctx.exception_word()),
            },
        }
    }
}

fn code_bytes(ctx: &Ctx<'_>) -> &'static [u8] {
    let (ptr, len) = {
        // the slice is re-derived after every event that can move it
        // (allocation, nested run)
        let arr = ctx.stack().frame_header(ctx.frame_base()).code(ctx.heap());
        let bytes = arr.as_slice();
        (bytes.as_ptr(), bytes.len())
    };

    unsafe { core::slice::from_raw_parts(ptr, len) }
}

#[inline(always)]
fn step<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    next_pc: usize,
    frame_base: usize,
    acc: &Register,
    ops: Operands<'_>,
    op: Opcode,
) -> Flow {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let stack = ctx.stack();

    match op {
        Opcode::Return => {
            if frame_base == ctx.base_anchor() {
                return Flow::Return;
            }
            let caller = stack.pop_frame(frame_base);
            ctx.set_frame_base(caller.base);
            Flow::Reframe(caller.pc)
        }
        Opcode::Load => {
            acc.store(stack.reg(heap, frame_base, ops.reg(0)));
            Flow::Next
        }
        Opcode::LoadSmi => {
            acc.store(Smi::new(ops.imm(0) as i64).into_tagged());
            Flow::Next
        }
        Opcode::LoadConstant => {
            acc.store(ctx.constants_ref(heap).at(heap, ops.idx(0)));
            Flow::Next
        }
        Opcode::LoadZero => {
            acc.store(Smi::new(0).into_tagged());
            Flow::Next
        }
        Opcode::LoadUndefined => {
            acc.store(heap.known().undefined.as_tagged(heap).erase());
            Flow::Next
        }
        Opcode::LoadNull => {
            acc.store(heap.known().null.as_tagged(heap).erase());
            Flow::Next
        }
        Opcode::LoadTrue => {
            acc.store(heap.known().true_object.as_tagged(heap).erase());
            Flow::Next
        }
        Opcode::LoadFalse => {
            acc.store(heap.known().false_object.as_tagged(heap).erase());
            Flow::Next
        }
        Opcode::LoadHole => {
            acc.store(heap.known().the_hole.as_tagged(heap).erase());
            Flow::Next
        }
        Opcode::LoadGlobal | Opcode::LoadGlobalFast => {
            let global = heap.known().global_object.as_tagged(heap).erase();
            if let Some(Hit::Value(v)) =
                InlineCache::try_load(heap, ctx.feedback_ref(heap), ops.idx(1), global)
            {
                acc.store(v);
                return Flow::Next;
            }
            let v = global_load(ctx, ops.idx(0), ops.idx(1), true);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LoadGlobalNoThrow => {
            let global = heap.known().global_object.as_tagged(heap).erase();
            if let Some(Hit::Value(v)) =
                InlineCache::try_load(heap, ctx.feedback_ref(heap), ops.idx(1), global)
            {
                acc.store(v);
                return Flow::Next;
            }
            let v = global_load(ctx, ops.idx(0), ops.idx(1), false);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LoadNamedProperty | Opcode::LoadNamedPropertyFast => {
            let receiver = stack.reg(heap, frame_base, ops.reg(0));
            // the become interpreter's three-tier probe: inline value hit,
            // out-of-line monomorphic handler, polymorphic resume
            match InlineCache::probe_mono(heap, ctx.feedback_ref(heap), ops.idx(2), receiver) {
                MonoProbe::Value(v) => {
                    acc.store(v);
                    return Flow::Next;
                }
                MonoProbe::Handler(obj, handler) => {
                    if let Some(Hit::Value(v)) = InlineCache::apply_mono(heap, obj, handler) {
                        acc.store(v);
                        return Flow::Next;
                    }
                }
                MonoProbe::Poly {
                    obj,
                    map,
                    pairs,
                    start,
                } => {
                    if let Some(Hit::Value(v)) =
                        InlineCache::try_load_resume(heap, obj, map, pairs, start)
                    {
                        acc.store(v);
                        return Flow::Next;
                    }
                }
                MonoProbe::Miss | MonoProbe::NotReceiver => {}
            }
            let v = named_load(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                ops.idx(1),
                ops.idx(2),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LoadKeyedProperty => {
            let fb = ops.idx(1);
            if let Some(idx) = acc.get(heap).to_i64()
                && idx >= 0
            {
                let recv = stack.reg(heap, frame_base, ops.reg(0));
                if let Some(obj) = recv.as_heap_object()
                    && let Some(v) = obj.as_ref().element_value_dense(heap, idx as usize)
                {
                    acc.store(v);
                    return Flow::Next;
                }
                if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
                    heap,
                    ctx.feedback_ref(heap),
                    fb,
                    recv,
                    idx as usize,
                ) {
                    acc.store(v);
                    return Flow::Next;
                }
            }
            let v = keyed_load(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                acc.get(heap),
                Some(fb),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LoadKeyedPropertyReg => {
            let key_reg = ops.reg(1);
            let fb = ops.idx(2);
            if let Some(idx) = stack.reg(heap, frame_base, key_reg).to_i64()
                && idx >= 0
            {
                let recv = stack.reg(heap, frame_base, ops.reg(0));
                if let Some(obj) = recv.as_heap_object()
                    && let Some(v) = obj.as_ref().element_value_dense(heap, idx as usize)
                {
                    acc.store(v);
                    return Flow::Next;
                }
                if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
                    heap,
                    ctx.feedback_ref(heap),
                    fb,
                    recv,
                    idx as usize,
                ) {
                    acc.store(v);
                    return Flow::Next;
                }
            }
            acc.store(stack.reg(heap, frame_base, key_reg));
            let v = keyed_load(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                stack.reg(heap, frame_base, key_reg),
                Some(fb),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LoadNewTarget => {
            acc.store(stack.frame_header(frame_base).new_target_slot().get(heap));
            Flow::Next
        }
        Opcode::Store => {
            stack.set_reg(frame_base, ops.reg(0), acc.get(heap));
            Flow::Next
        }
        Opcode::StoreGlobal => {
            let v = store_global(ctx, ops.idx(0), acc.get(heap));
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::StoreNamedProperty => {
            let v = store_named(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                ops.idx(1),
                ops.idx(2),
                acc.get(heap),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::StoreNamedPropertyNoShadow => {
            let v = store_named_no_shadow(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                ops.idx(1),
                acc.get(heap),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::AddParent => {
            let recv_reg = ops.reg(0);
            let name_idx = ops.idx(1);
            state.handle_scope(|scope| -> Flow {
                let receiver = scope.handle(stack.reg(heap, frame_base, recv_reg));
                let name = stack
                    .callable(heap, frame_base)
                    .as_ref()
                    .constant_slot_name(heap, name_idx);
                let name = scope.handle(name);
                let value = scope.handle(acc.get(heap));
                let Some(receiver) = scope.cast::<Object>(heap, receiver.as_tagged(heap)) else {
                    throw_err!(ctx, VmError::Type);
                };
                fold!(ctx, Object::add_parent(heap, &scope, receiver, name, value));
                Flow::Sync
            })
        }
        Opcode::StoreKeyedProperty | Opcode::StoreKeyedPropertyNoShadow => {
            let semantics = match op {
                Opcode::StoreKeyedPropertyNoShadow => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            let fb = ops.idx(2);
            if let Some(idx) = stack.reg(heap, frame_base, ops.reg(1)).to_i64()
                && idx >= 0
            {
                let recv = stack.reg(heap, frame_base, ops.reg(0));
                if recv
                    .store_array_element_in_place(heap, idx as usize, acc.get(heap))
                    .is_ok()
                {
                    return Flow::Next;
                }
                if let Some(v) = InlineCache::try_store_element(
                    heap,
                    ctx.feedback_ref(heap),
                    fb,
                    recv,
                    idx as usize,
                    acc.get(heap),
                ) {
                    acc.store(v);
                    return Flow::Next;
                }
            }
            let v = keyed_store(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                stack.reg(heap, frame_base, ops.reg(1)),
                acc.get(heap),
                Some(fb),
                semantics,
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::StoreKeyedSlot => {
            let recv_reg = ops.reg(0);
            let key_reg = ops.reg(1);
            state.handle_scope(|scope| -> Flow {
                let receiver = scope.handle(stack.reg(heap, frame_base, recv_reg));
                let raw = scope.handle(stack.reg(heap, frame_base, key_reg));
                let Some(key) = fold!(ctx, Object::to_property_key(vm, heap, state, raw)) else {
                    return Flow::Threw;
                };
                let key = scope.handle(key);

                if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
                    return match fold!(
                        ctx,
                        Proxy::set(
                            vm,
                            heap,
                            state,
                            receiver,
                            key.erase(),
                            scope.handle(acc.get(heap)),
                            receiver,
                        )
                    ) {
                        Coercion::Threw => Flow::Threw,
                        Coercion::Value(_) => Flow::Sync,
                    };
                }

                let name: Handle<'_, SlotName> =
                    match fold!(ctx, key.as_tagged(heap).erase().classify_key(heap)) {
                        Key::Element(i) => {
                            let Some(obj) = scope.cast::<Object>(heap, receiver.as_tagged(heap))
                            else {
                                throw_err!(ctx, VmError::Type);
                            };
                            if obj.as_tagged(heap).is_array(heap) {
                                let value = scope.handle(acc.get(heap));
                                fold!(
                                    ctx,
                                    obj.as_tagged(heap).erase().store_array_element_in_place(
                                        heap,
                                        i,
                                        value.as_tagged(heap)
                                    )
                                );
                                return Flow::Next;
                            }
                            let smi: Handle<'_, Smi> = scope.handle(Smi::new(i as i64));
                            let name = smi.as_tagged(heap).erase().as_name();
                            if matches!(
                                receiver.as_tagged(heap).lookup(heap, name),
                                Lookup::NotFound
                            ) {
                                throw_err!(ctx, VmError::OutOfBounds);
                            }
                            scope.handle(name)
                        }
                        Key::Name(key) => scope.handle(key),
                    };

                let value = scope.handle(acc.get(heap));
                let outcome = fold!(
                    ctx,
                    receiver.as_tagged(heap).store_lookup(
                        heap,
                        &scope,
                        name.as_tagged(heap),
                        value.as_tagged(heap),
                        StoreSemantics::WriteThrough,
                    )
                );
                let receiver = scope.handle(stack.reg(heap, frame_base, recv_reg));
                // `Err(())` = threw (the pending exception is set)
                if apply_store_outcome(ctx, receiver, outcome, value).is_err() {
                    return Flow::Threw;
                }
                Flow::Sync
            })
        }
        Opcode::StoreGlobalFast => {
            let name = stack
                .callable(heap, frame_base)
                .as_ref()
                .constant_slot_name(heap, ops.idx(0));
            match fold!(
                ctx,
                heap.known()
                    .global_object
                    .as_tagged(heap)
                    .erase()
                    .store_lookup_existing(heap, name, acc.get(heap), StoreSemantics::WriteThrough)
            ) {
                true => Flow::Sync,
                false => throw_err!(ctx, VmError::OutOfBounds),
            }
        }
        Opcode::LoadKeyedPropertyFast => {
            if let Some(idx) = Smi::decode(acc.get(heap).raw())
                && idx.value() >= 0
                && let Some(recv) = stack.reg(heap, frame_base, ops.reg(0)).as_heap_object()
                && let Some(v) = recv
                    .as_ref()
                    .element_value_dense(heap, idx.value() as usize)
            {
                acc.store(v);
                return Flow::Next;
            }
            debug_assert!(
                !Proxy::is_proxy(heap, stack.reg(heap, frame_base, ops.reg(0))),
                "fast load on a proxy receiver"
            );
            let receiver = stack.reg(heap, frame_base, ops.reg(0));
            let key = acc.get(heap).as_name();
            match fold!(ctx, receiver.load_outcome_keyed(heap, key)) {
                LoadOutcome::Value(v) => acc.store(v),
                LoadOutcome::Getter(_) => {
                    debug_assert!(false, "fast load on an accessor property");
                    acc.store(heap.known().undefined.as_tagged(heap).erase());
                }
            }
            Flow::Sync
        }
        Opcode::StoreNamedPropertyFast | Opcode::StoreNamedPropertyNoShadowFast => {
            let semantics = match op {
                Opcode::StoreNamedPropertyNoShadowFast => StoreSemantics::WriteThrough,
                _ => StoreSemantics::Shadow,
            };
            state.handle_scope(|scope| -> Flow {
                let receiver = scope.handle(stack.reg(heap, frame_base, ops.reg(0)));
                let name = scope.handle(
                    stack
                        .callable(heap, frame_base)
                        .as_ref()
                        .constant_slot_name(heap, ops.idx(1)),
                );
                let vector = ctx.feedback_ref(heap).map(|v| scope.handle(v));
                let value = scope.handle(acc.get(heap));

                // store IC (Shadow semantics only: WriteThrough stores
                // into parent-pair arrays that no map describes)
                if op == Opcode::StoreNamedPropertyFast
                    && let Some(hit) = InlineCache::try_store(
                        heap,
                        &scope,
                        vector,
                        ops.idx(2),
                        receiver,
                        name,
                        value,
                    )
                {
                    debug_assert!(
                        !matches!(hit, StoreHit::Setter(_)),
                        "fast store on a setter"
                    );
                    return Flow::Next;
                }

                let prev = receiver.as_tagged(heap).as_heap_object().map(|o| {
                    let map = o.as_ref().map(heap);
                    scope.handle(map)
                });
                let written = fold!(
                    ctx,
                    receiver.as_tagged(heap).erase().store_lookup_existing(
                        heap,
                        name.as_tagged(heap),
                        acc.get(heap),
                        semantics,
                    )
                );
                if !written {
                    throw_err!(ctx, VmError::OutOfBounds);
                }
                if op == Opcode::StoreNamedPropertyFast
                    && let Some(prev) = prev
                {
                    InlineCache::update_store(
                        heap,
                        &scope,
                        vector,
                        ops.idx(2),
                        receiver,
                        name,
                        prev,
                        StoreOutcomeKind::Done,
                    );
                }
                Flow::Sync
            })
        }
        Opcode::StoreKeyedPropertyFast => {
            if let Some(idx) = Smi::decode(stack.reg(heap, frame_base, ops.reg(1)).raw())
                && idx.value() >= 0
                && stack
                    .reg(heap, frame_base, ops.reg(0))
                    .store_array_element_in_place(heap, idx.value() as usize, acc.get(heap))
                    .is_ok()
            {
                return Flow::Next;
            }
            state.handle_scope(|scope| -> Flow {
                let receiver = scope.handle(stack.reg(heap, frame_base, ops.reg(0)));
                let key = scope.handle(stack.reg(heap, frame_base, ops.reg(1)));
                match fold!(ctx, key.as_tagged(heap).erase().classify_key(heap)) {
                    Key::Element(i) => {
                        if receiver
                            .as_tagged(heap)
                            .as_heap_object()
                            .is_some_and(|obj| obj.as_ref().is_array(heap))
                        {
                            let receiver = scope
                                .cast::<Object>(heap, receiver.as_tagged(heap))
                                .expect("array receiver is an object");
                            let value = scope.handle(acc.get(heap));
                            fold!(
                                ctx,
                                Object::store_array_element(heap, &scope, &receiver, i, &value)
                            );
                            return Flow::Sync;
                        }
                        let name = Tagged::from(Smi::new(i as i64));
                        match fold!(
                            ctx,
                            receiver.as_tagged(heap).erase().store_lookup_existing(
                                heap,
                                name,
                                acc.get(heap),
                                StoreSemantics::Shadow,
                            )
                        ) {
                            true => Flow::Sync,
                            false => throw_err!(ctx, VmError::OutOfBounds),
                        }
                    }
                    Key::Name(name) => {
                        match fold!(
                            ctx,
                            receiver.as_tagged(heap).erase().store_lookup_existing(
                                heap,
                                name,
                                acc.get(heap),
                                StoreSemantics::Shadow,
                            )
                        ) {
                            true => Flow::Sync,
                            false => throw_err!(ctx, VmError::OutOfBounds),
                        }
                    }
                }
            })
        }
        Opcode::Move => {
            stack.set_reg(
                frame_base,
                ops.reg(0),
                stack.reg(heap, frame_base, ops.reg(1)),
            );
            Flow::Next
        }
        Opcode::LoadContextSlot => {
            let depth = ops.uimm(1);
            let v = {
                let mut context = match stack
                    .frame_header(frame_base)
                    .context_slot()
                    .get(heap)
                    .get_as::<ContextObject>(heap)
                {
                    Some(context) => context,
                    None => throw_err!(ctx, VmError::Type),
                };
                for _ in 0..depth {
                    context = match context.as_ref().outer.get(heap) {
                        Some(context) => context,
                        None => throw_err!(ctx, VmError::Type),
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
            Flow::Next
        }
        Opcode::StoreContextSlot => {
            let mut context = match stack
                .frame_header(frame_base)
                .context_slot()
                .get(heap)
                .get_as::<ContextObject>(heap)
            {
                Some(context) => context,
                None => throw_err!(ctx, VmError::Type),
            };
            for _ in 0..ops.uimm(1) {
                context = match context.as_ref().outer.get(heap) {
                    Some(context) => context,
                    None => throw_err!(ctx, VmError::Type),
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
            Flow::Next
        }
        Opcode::CreateFunctionContext => {
            // constants[idx] is the scope's shared ScopeInfo (its `names`
            // array is parallel to the context's slots)
            let scope_idx = ops.idx(0);
            let count = fold!(ctx, {
                ctx.constants_ref(heap)
                    .at(heap, scope_idx)
                    .get_as::<ScopeInfo>(heap)
                    .map(|r| r.as_ref().names.get(heap).len())
                    .ok_or(VmError::Type)
            });
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .cast::<ContextObject>(
                        heap,
                        stack.frame_header(frame_base).context_slot().get(heap),
                    )
                    .expect("frame context slot holds a ContextObject");
                let values =
                    scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
                let scope_info = scope
                    .cast::<ScopeInfo>(heap, ctx.constants_ref(heap).at(heap, scope_idx))
                    .expect("constants slot holds a ScopeInfo");
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<ContextObject>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info,
                })
            });
            acc.store(ctx.erase());
            Flow::Sync
        }
        Opcode::CreateBlockContext => {
            let count = ops.uimm(0) as usize;
            let ctx = state.handle_scope(|scope| {
                let outer = scope
                    .cast::<ContextObject>(
                        heap,
                        stack.frame_header(frame_base).context_slot().get(heap),
                    )
                    .expect("frame context slot holds a ContextObject");
                let values =
                    scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
                let slots = heap.allocate_handle::<FixedArray>(values, &scope);
                heap.allocate::<ContextObject>(ContextInit {
                    outer: Some(outer),
                    slots,
                    scope_info: heap.known().empty_scope_info,
                })
            });
            acc.store(ctx.erase());
            Flow::Sync
        }
        Opcode::PushContext => {
            let old = stack.frame_header(frame_base).context_slot().get(heap);
            stack.set_reg(frame_base, ops.reg(0), old);
            let context = acc.get(heap);
            if context.get_as::<ContextObject>(heap).is_none() {
                throw_err!(ctx, VmError::Type);
            }
            stack.frame_header(frame_base).context_slot().store(context);
            Flow::Next
        }
        Opcode::PopContext => {
            let context = stack.reg(heap, frame_base, ops.reg(0));
            if context.get_as::<ContextObject>(heap).is_none() {
                throw_err!(ctx, VmError::Type);
            }
            stack.frame_header(frame_base).context_slot().store(context);
            Flow::Next
        }
        Opcode::ThrowReferenceErrorIfHole => {
            if acc.get(heap) == heap.known().the_hole.as_tagged(heap) {
                throw_err!(ctx, VmError::Reference);
            }
            Flow::Sync
        }
        Opcode::LoadContext => {
            acc.store(stack.frame_header(frame_base).context_slot().get(heap));
            Flow::Sync
        }
        // TODO: feedback vectors and separation once they are there
        Opcode::Call => {
            // TODO(strict-mode): ordinary sloppy functions still need nullish
            // receiver substitution and primitive receiver boxing.
            let callee_reg = ops.reg(0);
            let args_base = ops.reg_list(1);
            let count = ops.reg_count(2);
            let fb = ops.idx(3);

            // the call IC: a monomorphic site skips call-target
            // classification entirely (the become interpreter's fast path)
            let callee_word = stack.reg(heap, frame_base, callee_reg);
            if !Proxy::is_proxy(heap, callee_word) {
                match InlineCache::call_probe(heap, ctx.feedback_ref(heap), fb, callee_word) {
                    CallProbe::Bytecode(hit) => {
                        if hit.kind.is_class_constructor() {
                            throw_err!(ctx, VmError::Type);
                        }
                        // one shared anchor covers every read feeding the
                        // push; the probe never allocates, so the hit's
                        // borrow is still valid here
                        let frame = fold!(
                            ctx,
                            stack.push_frame(
                                heap,
                                ctx.caller_meta(next_pc, pc),
                                Callee {
                                    callable: hit.target.erase(),
                                    info: hit.info,
                                    context: hit.context.erase(),
                                },
                                heap.known().undefined.as_tagged(heap).erase(),
                                stack.window(frame_base, args_base, count),
                            )
                        );
                        ctx.set_frame_base(frame.base);
                        acc.store(fold!(ctx, run_callee(ctx)));
                        if ctx.is_throw(acc.get(heap)) {
                            return Flow::Threw;
                        }
                        return Flow::Sync;
                    }
                    CallProbe::Native(rt) => {
                        let exception = heap.known().exception.as_tagged(heap).raw();
                        let v = ctx.call_runtime(rt, stack.window(frame_base, args_base, count));
                        // Safety: old-gen singleton word.
                        return if v.raw() == exception {
                            Flow::Threw
                        } else {
                            acc.store(v);
                            Flow::Sync
                        };
                    }
                    CallProbe::Miss => {}
                }
            }

            // a callable proxy dispatches through its `apply` trap (or
            // a nested call of the target)
            if Proxy::is_proxy(heap, stack.reg(heap, frame_base, callee_reg)) {
                return match state.handle_scope(|scope| {
                    // staged: trap lookups may run user getters
                    let callee = scope.handle(stack.reg(heap, frame_base, callee_reg));
                    let staged = stack.args(frame_base, args_base, count);
                    Proxy::apply(vm, heap, state, callee, staged)
                }) {
                    Ok(Coercion::Threw) => Flow::Threw,
                    Ok(Coercion::Value(v)) => {
                        acc.store(v);
                        Flow::Sync
                    }
                    Err(err) => throw_err!(ctx, err),
                };
            }
            match stack.reg(heap, frame_base, callee_reg).call_target(heap) {
                None => throw_err!(ctx, VmError::Type),
                // the proxy dispatch above already intercepted these
                Some(CallTarget::Proxy(_)) => throw_err!(ctx, VmError::Type),
                Some(CallTarget::Native(rt)) => {
                    let exception = heap.known().exception.as_tagged(heap).raw();
                    let v = ctx.call_runtime(rt, stack.window(frame_base, args_base, count));
                    // Safety: old-gen singleton word.
                    if v.raw() == exception {
                        return Flow::Threw;
                    }
                    acc.store(v);
                    InlineCache::call_update(
                        heap,
                        ctx.feedback_ref(heap),
                        fb,
                        stack.reg(heap, frame_base, callee_reg),
                        CallRecord::Native(rt),
                    );
                    Flow::Sync
                }
                Some(CallTarget::Bytecode {
                    target: callee,
                    info,
                    context,
                    kind,
                    ..
                }) => {
                    if kind.is_class_constructor() {
                        throw_err!(ctx, VmError::Type);
                    }
                    // one shared anchor covers every read feeding the push
                    let frame = {
                        let context = context.erase();
                        let callee = callee.erase();
                        let undefined = heap.known().undefined.as_tagged(heap).erase();
                        fold!(
                            ctx,
                            stack.push_frame(
                                heap,
                                ctx.caller_meta(next_pc, pc),
                                Callee {
                                    callable: callee,
                                    info,
                                    context,
                                },
                                undefined,
                                stack.window(frame_base, args_base, count),
                            )
                        )
                    };
                    // record the site before the callee runs (the
                    // become interpreter's miss-path ordering)
                    InlineCache::call_update(
                        heap,
                        ctx.feedback_ref(heap),
                        fb,
                        stack.reg(heap, frame_base, callee_reg),
                        CallRecord::Bytecode(info),
                    );
                    ctx.set_frame_base(frame.base);
                    // blocking call: run the callee to completion; its
                    // value becomes the caller's accumulator
                    acc.store(fold!(ctx, run_callee(ctx)));
                    if ctx.is_throw(acc.get(heap)) {
                        return Flow::Threw;
                    }
                    Flow::Sync
                }
            }
        }
        Opcode::CallNoFeedback => {
            // TODO(strict-mode): ordinary sloppy functions still need nullish
            // receiver substitution and primitive receiver boxing.
            let callee_reg = ops.reg(0);
            let args_base = ops.reg_list(1);
            let count = ops.reg_count(2);
            // a callable proxy dispatches through its `apply` trap (or
            // a nested call of the target)
            if Proxy::is_proxy(heap, stack.reg(heap, frame_base, callee_reg)) {
                return match state.handle_scope(|scope| {
                    // staged: trap lookups may run user getters
                    let callee = scope.handle(stack.reg(heap, frame_base, callee_reg));
                    let staged = stack.args(frame_base, args_base, count);
                    Proxy::apply(vm, heap, state, callee, staged)
                }) {
                    Ok(Coercion::Threw) => Flow::Threw,
                    Ok(Coercion::Value(v)) => {
                        acc.store(v);
                        Flow::Sync
                    }
                    Err(err) => throw_err!(ctx, err),
                };
            }
            match stack.reg(heap, frame_base, callee_reg).call_target(heap) {
                None => throw_err!(ctx, VmError::Type),
                // the proxy dispatch above already intercepted these
                Some(CallTarget::Proxy(_)) => throw_err!(ctx, VmError::Type),
                Some(CallTarget::Native(rt)) => {
                    let exception = heap.known().exception.as_tagged(heap).raw();
                    let v = ctx.call_runtime(rt, stack.window(frame_base, args_base, count));
                    // Safety: old-gen singleton word.
                    if v.raw() == exception {
                        Flow::Threw
                    } else {
                        acc.store(v);
                        Flow::Sync
                    }
                }
                Some(CallTarget::Bytecode {
                    target: callee,
                    info,
                    context,
                    kind,
                    ..
                }) => {
                    if kind.is_class_constructor() {
                        throw_err!(ctx, VmError::Type);
                    }
                    // one shared anchor covers every read feeding the push
                    let frame = {
                        let context = context.erase();
                        let callee = callee.erase();
                        let undefined = heap.known().undefined.as_tagged(heap).erase();
                        fold!(
                            ctx,
                            stack.push_frame(
                                heap,
                                ctx.caller_meta(next_pc, pc),
                                Callee {
                                    callable: callee,
                                    info,
                                    context,
                                },
                                undefined,
                                stack.window(frame_base, args_base, count),
                            )
                        )
                    };
                    ctx.set_frame_base(frame.base);
                    // blocking call: run the callee to completion; its
                    // value becomes the caller's accumulator
                    acc.store(fold!(ctx, run_callee(ctx)));
                    if ctx.is_throw(acc.get(heap)) {
                        return Flow::Threw;
                    }
                    Flow::Sync
                }
            }
        }
        Opcode::CallMethod0 | Opcode::CallMethod1 | Opcode::CallMethod2 => {
            let callee_reg = ops.reg(0);
            let argc = match op {
                Opcode::CallMethod0 => 0usize,
                Opcode::CallMethod1 => 1,
                _ => 2,
            };
            let recv = ops.reg(1);
            let mut margs = [0i32; 2];
            for (i, arg) in margs.iter_mut().enumerate().take(argc) {
                *arg = ops.reg(2 + i);
            }
            let srcs = (recv, margs, argc);
            let fb = ops.idx(argc as usize + 2);
            // the call IC: a monomorphic site skips call-target
            // classification entirely (the become interpreter's fast path)
            let callee_word = stack.reg(heap, frame_base, callee_reg);
            if !Proxy::is_proxy(heap, callee_word) {
                match InlineCache::call_probe(heap, ctx.feedback_ref(heap), fb, callee_word) {
                    CallProbe::Bytecode(hit) => {
                        if hit.kind.is_class_constructor() {
                            throw_err!(ctx, VmError::Type);
                        }
                        let frame = fold!(
                            ctx,
                            stack.push_scattered_frame(
                                heap,
                                ctx.caller_meta(next_pc, pc),
                                Callee {
                                    callable: hit.target.erase(),
                                    info: hit.info,
                                    context: hit.context.erase(),
                                },
                                heap.known().undefined.as_tagged(heap).erase(),
                                Recv::Reg(srcs.0),
                                srcs.1,
                                srcs.2
                            )
                        );
                        ctx.set_frame_base(frame.base);
                        acc.store(fold!(ctx, run_callee(ctx)));
                        if ctx.is_throw(acc.get(heap)) {
                            return Flow::Threw;
                        }
                        return Flow::Sync;
                    }
                    CallProbe::Native(rt) => {
                        let exception = heap.known().exception.as_tagged(heap).raw();
                        let staged = fold!(
                            ctx,
                            stack.stage_scattered(
                                heap,
                                frame_base,
                                Recv::Reg(srcs.0),
                                srcs.1,
                                srcs.2
                            )
                        );
                        let v = ctx.call_runtime(rt, staged);
                        // Safety: old-gen singleton word.
                        return if v.raw() == exception {
                            Flow::Threw
                        } else {
                            acc.store(v);
                            Flow::Sync
                        };
                    }
                    CallProbe::Miss => {}
                }
            }

            if Proxy::is_proxy(heap, stack.reg(heap, frame_base, callee_reg)) {
                let staged = fold!(
                    ctx,
                    stack.stage_scattered(heap, frame_base, Recv::Reg(srcs.0), srcs.1, srcs.2)
                );
                let saved_top = stack.top();
                if let Some(top) = stack.rooting_top(staged) {
                    stack.set_top(top);
                }
                let result = state.handle_scope(|scope| {
                    let callee = scope.handle(stack.reg(heap, frame_base, callee_reg));
                    Proxy::apply(vm, heap, state, callee, staged)
                });
                stack.set_top(saved_top);
                return match result {
                    Ok(Coercion::Threw) => Flow::Threw,
                    Ok(Coercion::Value(v)) => {
                        acc.store(v);
                        Flow::Sync
                    }
                    Err(err) => throw_err!(ctx, err),
                };
            }
            match stack.reg(heap, frame_base, callee_reg).call_target(heap) {
                None => throw_err!(ctx, VmError::Type),
                Some(CallTarget::Proxy(_)) => throw_err!(ctx, VmError::Type),
                Some(CallTarget::Native(rt)) => {
                    let exception = heap.known().exception.as_tagged(heap).raw();
                    let staged = fold!(
                        ctx,
                        stack.stage_scattered(heap, frame_base, Recv::Reg(srcs.0), srcs.1, srcs.2)
                    );
                    let v = ctx.call_runtime(rt, staged);
                    InlineCache::call_update(
                        heap,
                        ctx.feedback_ref(heap),
                        fb,
                        stack.reg(heap, frame_base, callee_reg),
                        CallRecord::Native(rt),
                    );
                    if v.raw() == exception {
                        Flow::Threw
                    } else {
                        acc.store(v);
                        Flow::Sync
                    }
                }
                Some(CallTarget::Bytecode {
                    target: callee,
                    info,
                    context,
                    kind,
                    ..
                }) => {
                    if kind.is_class_constructor() {
                        throw_err!(ctx, VmError::Type);
                    }
                    let frame = fold!(
                        ctx,
                        stack.push_scattered_frame(
                            heap,
                            ctx.caller_meta(next_pc, pc),
                            Callee {
                                callable: callee.erase(),
                                info,
                                context: context.erase(),
                            },
                            heap.known().undefined.as_tagged(heap).erase(),
                            Recv::Reg(srcs.0),
                            srcs.1,
                            srcs.2
                        )
                    );
                    InlineCache::call_update(
                        heap,
                        ctx.feedback_ref(heap),
                        fb,
                        stack.reg(heap, frame_base, callee_reg),
                        CallRecord::Bytecode(info),
                    );
                    ctx.set_frame_base(frame.base);
                    // blocking call: run the callee to completion; its
                    // value becomes the caller's accumulator
                    acc.store(fold!(ctx, run_callee(ctx)));
                    if ctx.is_throw(acc.get(heap)) {
                        return Flow::Threw;
                    }
                    Flow::Sync
                }
            }
        }
        Opcode::CallFunction0 | Opcode::CallFunction1 | Opcode::CallFunction2 => {
            let callee_reg = ops.reg(0);
            let argc = match op {
                Opcode::CallFunction0 => 0usize,
                Opcode::CallFunction1 => 1,
                _ => 2,
            };
            let mut fargs = [0i32; 2];
            for (i, arg) in fargs.iter_mut().enumerate().take(argc) {
                *arg = ops.reg(1 + i);
            }
            let args = (fargs, argc);
            let fb = ops.idx(argc as usize + 1);

            // the call IC: a monomorphic site skips call-target
            // classification entirely (the become interpreter's fast path)
            let callee_word = stack.reg(heap, frame_base, callee_reg);
            if !Proxy::is_proxy(heap, callee_word) {
                match InlineCache::call_probe(heap, ctx.feedback_ref(heap), fb, callee_word) {
                    CallProbe::Bytecode(hit) => {
                        if hit.kind.is_class_constructor() {
                            throw_err!(ctx, VmError::Type);
                        }
                        let frame = fold!(
                            ctx,
                            stack.push_scattered_frame(
                                heap,
                                ctx.caller_meta(next_pc, pc),
                                Callee {
                                    callable: hit.target.erase(),
                                    info: hit.info,
                                    context: hit.context.erase(),
                                },
                                heap.known().undefined.as_tagged(heap).erase(),
                                Recv::Undefined,
                                args.0,
                                args.1
                            )
                        );
                        ctx.set_frame_base(frame.base);
                        acc.store(fold!(ctx, run_callee(ctx)));
                        if ctx.is_throw(acc.get(heap)) {
                            return Flow::Threw;
                        }
                        return Flow::Sync;
                    }
                    CallProbe::Native(rt) => {
                        let exception = heap.known().exception.as_tagged(heap).raw();
                        let staged = fold!(
                            ctx,
                            stack.stage_scattered(
                                heap,
                                frame_base,
                                Recv::Undefined,
                                args.0,
                                args.1
                            )
                        );
                        let v = ctx.call_runtime(rt, staged);
                        // Safety: old-gen singleton word.
                        return if v.raw() == exception {
                            Flow::Threw
                        } else {
                            acc.store(v);
                            Flow::Sync
                        };
                    }
                    CallProbe::Miss => {}
                }
            }

            if Proxy::is_proxy(heap, stack.reg(heap, frame_base, callee_reg)) {
                let staged = fold!(
                    ctx,
                    stack.stage_scattered(heap, frame_base, Recv::Undefined, args.0, args.1)
                );
                let saved_top = stack.top();
                if let Some(top) = stack.rooting_top(staged) {
                    stack.set_top(top);
                }
                let result = state.handle_scope(|scope| {
                    let callee = scope.handle(stack.reg(heap, frame_base, callee_reg));
                    Proxy::apply(vm, heap, state, callee, staged)
                });
                stack.set_top(saved_top);
                return match result {
                    Ok(Coercion::Threw) => Flow::Threw,
                    Ok(Coercion::Value(v)) => {
                        acc.store(v);
                        Flow::Sync
                    }
                    Err(err) => throw_err!(ctx, err),
                };
            }
            match stack.reg(heap, frame_base, callee_reg).call_target(heap) {
                None => throw_err!(ctx, VmError::Type),
                Some(CallTarget::Proxy(_)) => throw_err!(ctx, VmError::Type),
                Some(CallTarget::Native(rt)) => {
                    let exception = heap.known().exception.as_tagged(heap).raw();
                    let staged = fold!(
                        ctx,
                        stack.stage_scattered(heap, frame_base, Recv::Undefined, args.0, args.1)
                    );
                    let v = ctx.call_runtime(rt, staged);
                    InlineCache::call_update(
                        heap,
                        ctx.feedback_ref(heap),
                        fb,
                        stack.reg(heap, frame_base, callee_reg),
                        CallRecord::Native(rt),
                    );
                    if v.raw() == exception {
                        Flow::Threw
                    } else {
                        acc.store(v);
                        Flow::Sync
                    }
                }
                Some(CallTarget::Bytecode {
                    target: callee,
                    info,
                    context,
                    kind,
                    ..
                }) => {
                    if kind.is_class_constructor() {
                        throw_err!(ctx, VmError::Type);
                    }
                    let frame = fold!(
                        ctx,
                        stack.push_scattered_frame(
                            heap,
                            ctx.caller_meta(next_pc, pc),
                            Callee {
                                callable: callee.erase(),
                                info,
                                context: context.erase(),
                            },
                            heap.known().undefined.as_tagged(heap).erase(),
                            Recv::Undefined,
                            args.0,
                            args.1
                        )
                    );
                    InlineCache::call_update(
                        heap,
                        ctx.feedback_ref(heap),
                        fb,
                        stack.reg(heap, frame_base, callee_reg),
                        CallRecord::Bytecode(info),
                    );
                    ctx.set_frame_base(frame.base);
                    // blocking call: run the callee to completion; its
                    // value becomes the caller's accumulator
                    acc.store(fold!(ctx, run_callee(ctx)));
                    if ctx.is_throw(acc.get(heap)) {
                        return Flow::Threw;
                    }
                    Flow::Sync
                }
            }
        }
        Opcode::CallRuntime => {
            // operand 0 is a BuiltinFn discriminant: the fixed
            // runtime-helper table (vm::runtime_fn) registered at
            // indices 0..BuiltinFn::COUNT
            let f = vm.runtime(NativeIndex(ops.idx(0)));
            let args_base = ops.reg_list(1);
            let count = ops.reg_count(2);
            let exception = heap.known().exception.as_tagged(heap).raw();
            let nctx = HostCtx::new(vm, heap, state);
            let v = f(nctx, None, stack.window(frame_base, args_base, count));
            // Safety: old-gen singleton word.
            if v.raw() == exception {
                Flow::Threw
            } else {
                acc.store(v);
                Flow::Sync
            }
        }
        Opcode::Construct => {
            let v = construct(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                ops.reg_list(1),
                ops.reg_count(2),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::CreateEmptyObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().object_initial_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY)
            });
            acc.store(obj.erase());
            Flow::Sync
        }
        Opcode::CreateEmptyArrayLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().js_array_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY)
            });
            acc.store(obj.erase());
            Flow::Sync
        }
        Opcode::CreateSizedArrayLiteral => {
            let capacity = ops.uimm(0) as usize;
            let obj = state.handle_scope(|scope| heap.new_array_with_capacity(&scope, capacity));
            acc.store(obj.erase());
            Flow::Sync
        }
        Opcode::CreateBareObjectLiteral => {
            let obj = state.handle_scope(|scope| {
                let map = heap.known().plain_object_map;
                heap.new_object(&scope, map, HandleSlice::EMPTY)
            });
            acc.store(obj.erase());
            Flow::Sync
        }
        Opcode::CreateClosure => {
            let info_idx = ops.idx(0);
            state.handle_scope(|scope| -> Flow {
                let Some(info) = scope
                    .cast::<CallableInfoObject>(heap, ctx.constants_ref(heap).at(heap, info_idx))
                else {
                    throw_err!(ctx, VmError::Type);
                };
                let context = scope
                    .cast::<ContextObject>(
                        heap,
                        stack.frame_header(frame_base).context_slot().get(heap),
                    )
                    .expect("frame context slot holds a ContextObject");
                let obj = fold!(ctx, Object::create_closure(heap, &scope, info, context));
                acc.store(obj.erase());
                Flow::Sync
            })
        }
        Opcode::LoadCurrentClosure => {
            acc.store(stack.frame_header(frame_base).callable_slot().get(heap));
            Flow::Next
        }
        Opcode::Add => {
            let lhs = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (lhs.to_i64(), acc.get(heap).to_i64())
                && let Some(r) = a.checked_add(b)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Flow::Next;
            }
            if let (Some(a), Some(b)) = (lhs.as_number(heap), acc.get(heap).as_number(heap)) {
                acc.store(heap.new_number(a + b));
                return Flow::Next;
            }
            let v = add(ctx, stack.reg(heap, frame_base, ops.reg(0)), acc.get(heap));
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::AddLoc | Opcode::SubLoc => {
            let dst = ops.reg(0);
            let src = ops.reg(1);
            let lhs = stack.reg(heap, frame_base, dst);
            let rhs = stack.reg(heap, frame_base, src);
            if let (Some(a), Some(b)) = (lhs.smi_bits(), rhs.smi_bits()) {
                let stepped = if op == Opcode::AddLoc {
                    a.checked_add(b)
                } else {
                    a.checked_sub(b)
                };
                if let Some(r) = stepped {
                    let v = Tagged::from_smi_bits(r);
                    stack.set_reg(frame_base, dst, v);
                    acc.store(v);
                    return Flow::Next;
                }
            }
            if let (Some(a), Some(b)) = (lhs.as_number(heap), rhs.as_number(heap)) {
                let r = if op == Opcode::AddLoc { a + b } else { a - b };
                let v = heap.new_number(r);
                stack.set_reg(frame_base, dst, v);
                acc.store(v);
                return Flow::Sync;
            }
            let v = if op == Opcode::AddLoc {
                add(
                    ctx,
                    stack.reg(heap, frame_base, dst),
                    stack.reg(heap, frame_base, src),
                )
            } else {
                numeric(
                    ctx,
                    stack.reg(heap, frame_base, dst),
                    stack.reg(heap, frame_base, src),
                    |a, b| a - b,
                )
            };
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            stack.set_reg(frame_base, dst, v);
            acc.store(v);
            Flow::Sync
        }
        Opcode::Sub => {
            let lhs = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (lhs.to_i64(), acc.get(heap).to_i64())
                && let Some(r) = a.checked_sub(b)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Flow::Next;
            }
            if let (Some(a), Some(b)) = (lhs.as_number(heap), acc.get(heap).as_number(heap)) {
                acc.store(heap.new_number(a - b));
                return Flow::Next;
            }
            let v = numeric(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                acc.get(heap),
                |a, b| a - b,
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::IncLoc | Opcode::DecLoc => {
            let reg = ops.reg(0);
            let delta: i64 = if op == Opcode::IncLoc { 1 } else { -1 };
            let bits = stack.reg(heap, frame_base, reg).smi_bits();
            if let Some(bits) = bits {
                let stepped = if delta > 0 {
                    bits.checked_add(2)
                } else {
                    bits.checked_sub(2)
                };
                if let Some(new) = stepped {
                    stack.set_reg(frame_base, reg, Tagged::from_smi_bits(new));
                    // the postfix value is the numeric old value
                    acc.store(Tagged::from_smi_bits(bits));
                    return Flow::Next;
                }
            }
            state.handle_scope(|scope| -> Flow {
                let old = scope.handle(stack.reg(heap, frame_base, reg));
                let n = fold!(ctx, Object::to_numeric(vm, heap, state, old));
                let Some(n) = n else {
                    return Flow::Threw;
                };
                let old_num = if n.fract() == 0.0
                    && Smi::in_range(n as i64)
                    && !(n == 0.0 && n.is_sign_negative())
                {
                    scope.handle(Smi::new(n as i64).into_tagged())
                } else {
                    scope.handle(heap.new_number(n))
                };
                let new = heap.new_number(n + delta as f64);
                stack.set_reg(frame_base, reg, new);
                acc.store(old_num.as_tagged(heap));
                Flow::Sync
            })
        }
        Opcode::Mul => {
            let lhs = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (lhs.to_i64(), acc.get(heap).to_i64())
                && let Some(r) = a.checked_mul(b)
                && Smi::in_range(r)
            {
                acc.store(Smi::new(r).into_tagged());
                return Flow::Next;
            }
            if let (Some(a), Some(b)) = (lhs.as_number(heap), acc.get(heap).as_number(heap)) {
                acc.store(heap.new_number(a * b));
                return Flow::Next;
            }
            let v = numeric(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                acc.get(heap),
                |a, b| a * b,
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::Div => {
            // JS division is IEEE double division: 7/2 = 3.5, x/0 = ±Infinity
            // or NaN, MIN/-1 overflows to a double.
            let lhs = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (lhs.to_i64(), acc.get(heap).to_i64())
                && b != 0
                && a % b == 0
                && let Some(r) = a.checked_div(b)
            {
                acc.store(Smi::new(r).into_tagged());
                return Flow::Next;
            }
            if let (Some(a), Some(b)) = (lhs.as_number(heap), acc.get(heap).as_number(heap)) {
                acc.store(heap.new_number(a / b));
                return Flow::Next;
            }
            let v = numeric(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                acc.get(heap),
                |a, b| a / b,
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::Mod => {
            // JS remainder is IEEE fmod: x % 0 = NaN, signs follow the dividend.
            let lhs = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (lhs.to_i64(), acc.get(heap).to_i64())
                && b != 0
            {
                acc.store(Smi::new(a % b).into_tagged());
                return Flow::Next;
            }
            let v = numeric(
                ctx,
                stack.reg(heap, frame_base, ops.reg(0)),
                acc.get(heap),
                |a, b| a % b,
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::Exp => {
            let reg = ops.reg(0);
            state.handle_scope(|scope| -> Flow {
                // JS exponentiation is always IEEE double math; the result only
                // needs a Smi tag when it is an in-range integer.
                let a = scope.handle(stack.reg(heap, frame_base, reg));
                let b = scope.handle(acc.get(heap));
                let v = fold!(
                    ctx,
                    Object::numeric_op(vm, heap, state, a, b, |a, b| a.powf(b))
                );
                let Some(v) = v else {
                    return Flow::Threw;
                };
                acc.store(v);
                Flow::Sync
            })
        }
        Opcode::BitwiseOr => {
            let a = stack.reg(heap, frame_base, ops.reg(0));
            let b = acc.get(heap);
            if let (Some(ai), Some(bi)) = (a.to_i64(), b.to_i64()) {
                acc.store(Smi::new(((ai as i32) | (bi as i32)) as i64).into_tagged());
                return Flow::Sync;
            }
            let v = bitwise(ctx, 0, a, b);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::BitwiseXor => {
            let a = stack.reg(heap, frame_base, ops.reg(0));
            let b = acc.get(heap);
            if let (Some(ai), Some(bi)) = (a.to_i64(), b.to_i64()) {
                acc.store(Smi::new(((ai as i32) ^ (bi as i32)) as i64).into_tagged());
                return Flow::Sync;
            }
            let v = bitwise(ctx, 1, a, b);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::BitwiseAnd => {
            let a = stack.reg(heap, frame_base, ops.reg(0));
            let b = acc.get(heap);
            if let (Some(ai), Some(bi)) = (a.to_i64(), b.to_i64()) {
                acc.store(Smi::new(((ai as i32) & (bi as i32)) as i64).into_tagged());
                return Flow::Sync;
            }
            let v = bitwise(ctx, 2, a, b);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::ShiftLeft => {
            // ToInt32(lhs) << (ToUint32(rhs) & 31), truncated to int32
            let a = stack.reg(heap, frame_base, ops.reg(0));
            let b = acc.get(heap);
            if let (Some(ai), Some(bi)) = (a.to_i64(), b.to_i64()) {
                acc.store(Smi::new((ai as i32).wrapping_shl(bi as u32 & 31) as i64).into_tagged());
                return Flow::Sync;
            }
            let v = bitwise(ctx, 3, a, b);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::ShiftRight => {
            // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending
            let a = stack.reg(heap, frame_base, ops.reg(0));
            let b = acc.get(heap);
            if let (Some(ai), Some(bi)) = (a.to_i64(), b.to_i64()) {
                acc.store(Smi::new((ai as i32).wrapping_shr(bi as u32 & 31) as i64).into_tagged());
                return Flow::Sync;
            }
            let v = bitwise(ctx, 4, a, b);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::ShiftRightLogical => {
            // ToUint32(lhs) >>> (ToUint32(rhs) & 31): always non-negative
            let a = stack.reg(heap, frame_base, ops.reg(0));
            let b = acc.get(heap);
            if let (Some(ai), Some(bi)) = (a.to_i64(), b.to_i64()) {
                acc.store(Smi::new((ai as u32).wrapping_shr(bi as u32 & 31) as i64).into_tagged());
                return Flow::Sync;
            }
            let v = bitwise(ctx, 5, a, b);
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::Jump => Flow::Jump(jump_target(pc, ops.imm(0))),
        Opcode::JumpLoop => {
            // read the offset before the safepoint poll: a collection
            // invalidates the operand cursor
            let offset = ops.imm(0);
            if ctx.safepoint_tick() && heap.safepoint_poll() {
                return begin_termination(heap, state);
            }
            Flow::Jump(jump_target(pc, offset))
        }
        Opcode::JumpIfTruthy => {
            if acc.get(heap).is_truthy(heap) {
                return Flow::Jump(jump_target(pc, ops.imm(0)));
            }
            Flow::Next
        }
        Opcode::JumpIfFalsy => {
            if !acc.get(heap).is_truthy(heap) {
                return Flow::Jump(jump_target(pc, ops.imm(0)));
            }
            Flow::Next
        }
        Opcode::JumpIfNotUndefined => {
            if acc.get(heap) != heap.known().undefined.as_tagged(heap) {
                return Flow::Jump(jump_target(pc, ops.imm(0)));
            }
            Flow::Next
        }
        Opcode::TestReferenceEqual => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            let known = heap.known();
            acc.store(if other == acc.get(heap) {
                known.true_object.as_tagged(heap).erase()
            } else {
                known.false_object.as_tagged(heap).erase()
            });
            Flow::Next
        }
        Opcode::TestTypeof => {
            acc.store(Object::type_of(heap, acc.get(heap)));
            Flow::Next
        }
        Opcode::Negate => {
            let acc_word = acc.get(heap);
            if let Some(n) = acc.get(heap).as_number(heap) {
                // `new_number` keeps -0.0 boxed and folds everything else
                acc.store(heap.new_number(-n));
            } else if let Some(v) = acc_word.to_i64() {
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
                let n = fold!(ctx, n);
                let Some(n) = n else {
                    return Flow::Threw;
                };
                // new_number boxes -0.0 itself
                acc.store(heap.new_number(-n));
            }
            Flow::Sync
        }
        Opcode::InstanceOf => {
            let callable_reg = ops.reg(0);
            state.handle_scope(|scope| -> Flow {
                let object = scope.handle(acc.get(heap));
                let callable = scope.handle(stack.reg(heap, frame_base, callable_reg));
                let r = fold!(ctx, Object::instance_of(vm, heap, state, object, callable));
                let Some(r) = r else {
                    return Flow::Threw;
                };
                acc.store(Convert::boolean(heap, r));
                Flow::Sync
            })
        }
        Opcode::EqualStrict => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            let r = Compare::strict_equal(heap, acc.get(heap), other);
            acc.store(Convert::boolean(heap, r));
            Flow::Next
        }
        Opcode::Equal => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.get(heap).as_number(heap), other.as_number(heap)) {
                acc.store(Convert::boolean(heap, a == b));
                return Flow::Next;
            }
            let v = compare(
                ctx,
                0,
                acc.get(heap),
                stack.reg(heap, frame_base, ops.reg(0)),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LessThan => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.get(heap).as_number(heap), other.as_number(heap)) {
                acc.store(Convert::boolean(heap, a < b));
                return Flow::Next;
            }
            let v = compare(
                ctx,
                2,
                acc.get(heap),
                stack.reg(heap, frame_base, ops.reg(0)),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::LessThanOrEqual => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.get(heap).as_number(heap), other.as_number(heap)) {
                acc.store(Convert::boolean(heap, a <= b));
                return Flow::Next;
            }
            let v = compare(
                ctx,
                3,
                acc.get(heap),
                stack.reg(heap, frame_base, ops.reg(0)),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::GreaterThan => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.get(heap).as_number(heap), other.as_number(heap)) {
                acc.store(Convert::boolean(heap, a > b));
                return Flow::Next;
            }
            let v = compare(
                ctx,
                4,
                acc.get(heap),
                stack.reg(heap, frame_base, ops.reg(0)),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::GreaterThanOrEqual => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            if let (Some(a), Some(b)) = (acc.get(heap).as_number(heap), other.as_number(heap)) {
                acc.store(Convert::boolean(heap, a >= b));
                return Flow::Next;
            }
            let v = compare(
                ctx,
                5,
                acc.get(heap),
                stack.reg(heap, frame_base, ops.reg(0)),
            );
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::CompareJump => {
            let other = stack.reg(heap, frame_base, ops.reg(0));
            // kind = relation index * 2 + jump_if_falsy (0 = `==`,
            // 1 = `===`, 2 = `<`, 3 = `<=`, 4 = `>`, 5 = `>=`)
            let cmp = (ops.uimm(1) / 2) as u8;
            let falsy_jump = ops.uimm(1) % 2 == 1;
            let offset = ops.imm(2);
            let b = if let (Some(a), Some(b)) =
                (acc.get(heap).as_number(heap), other.as_number(heap))
            {
                match cmp {
                    0 | 1 => a == b,
                    2 => a < b,
                    3 => a <= b,
                    4 => a > b,
                    _ => a >= b,
                }
            } else {
                let v = compare(ctx, cmp, acc.get(heap), other);
                if ctx.is_throw(v) {
                    return Flow::Threw;
                }
                v.is_truthy(heap)
            };
            acc.store(Convert::boolean(heap, b));
            if b != falsy_jump {
                return Flow::Jump(jump_target(pc, offset));
            }
            Flow::Next
        }
        Opcode::AddImmediate
        | Opcode::SubImmediate
        | Opcode::MulImmediate
        | Opcode::DivImmediate
        | Opcode::ModImmediate
        | Opcode::ExpImmediate
        | Opcode::BitwiseOrImmediate
        | Opcode::BitwiseXorImmediate
        | Opcode::BitwiseAndImmediate
        | Opcode::ShiftLeftImmediate
        | Opcode::ShiftRightImmediate
        | Opcode::ShiftRightLogicalImmediate => {
            let reg = ops.reg(0);
            let imm = ops.imm(1);
            let lhs = stack.reg(heap, frame_base, reg);
            if let Some(a) = lhs.to_i64() {
                let fast: Option<i64> = match op {
                    Opcode::AddImmediate => a.checked_add(imm as i64),
                    Opcode::SubImmediate => a.checked_sub(imm as i64),
                    Opcode::MulImmediate => a.checked_mul(imm as i64),
                    Opcode::DivImmediate => {
                        (imm != 0 && a % imm as i64 == 0).then(|| a / imm as i64)
                    }
                    Opcode::ModImmediate => (imm != 0).then(|| a % imm as i64),
                    Opcode::BitwiseOrImmediate => Some((a as i32 | imm) as i64),
                    Opcode::BitwiseXorImmediate => Some((a as i32 ^ imm) as i64),
                    Opcode::BitwiseAndImmediate => Some((a as i32 & imm) as i64),
                    Opcode::ShiftLeftImmediate => {
                        Some((a as i32).wrapping_shl(imm as u32 & 31) as i64)
                    }
                    Opcode::ShiftRightImmediate => {
                        Some((a as i32).wrapping_shr(imm as u32 & 31) as i64)
                    }
                    Opcode::ShiftRightLogicalImmediate => {
                        Some((a as u32).wrapping_shr(imm as u32 & 31) as i64)
                    }
                    _ => None,
                };
                if let Some(r) = fast
                    && Smi::in_range(r)
                {
                    acc.store(Smi::new(r).into_tagged());
                    return Flow::Next;
                }
            }
            match op {
                Opcode::AddImmediate => {
                    let v = add(
                        ctx,
                        stack.reg(heap, frame_base, reg),
                        Smi::new(imm as i64).into_tagged(),
                    );
                    if ctx.is_throw(v) {
                        return Flow::Threw;
                    }
                    acc.store(v);
                    Flow::Sync
                }
                Opcode::SubImmediate
                | Opcode::MulImmediate
                | Opcode::DivImmediate
                | Opcode::ModImmediate
                | Opcode::ExpImmediate => {
                    let f: fn(f64, f64) -> f64 = match op {
                        Opcode::SubImmediate => |a, b| a - b,
                        Opcode::MulImmediate => |a, b| a * b,
                        Opcode::DivImmediate => |a, b| a / b,
                        Opcode::ModImmediate => |a, b| a % b,
                        _ => |a, b| a.powf(b),
                    };
                    let v = numeric(
                        ctx,
                        stack.reg(heap, frame_base, reg),
                        Smi::new(imm as i64).into_tagged(),
                        f,
                    );
                    if ctx.is_throw(v) {
                        return Flow::Threw;
                    }
                    acc.store(v);
                    Flow::Sync
                }
                // bitwise/shift: ToInt32/ToUint32 coercion of the lhs
                _ => {
                    let kind = match op {
                        Opcode::BitwiseOrImmediate => 0,
                        Opcode::BitwiseXorImmediate => 1,
                        Opcode::BitwiseAndImmediate => 2,
                        Opcode::ShiftLeftImmediate => 3,
                        Opcode::ShiftRightImmediate => 4,
                        _ => 5,
                    };
                    let v = bitwise(
                        ctx,
                        kind,
                        stack.reg(heap, frame_base, reg),
                        Smi::new(imm as i64).into_tagged(),
                    );
                    if ctx.is_throw(v) {
                        return Flow::Threw;
                    }
                    acc.store(v);
                    Flow::Sync
                }
            }
        }
        Opcode::LoadElementImm => {
            let idx = ops.uimm(1) as usize;
            let fb = ops.idx(2);
            if let Some(recv) = stack.reg(heap, frame_base, ops.reg(0)).as_heap_object()
                && let Some(v) = recv.as_ref().element_value_dense(heap, idx)
            {
                acc.store(v);
                return Flow::Sync;
            }
            if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
                heap,
                ctx.feedback_ref(heap),
                fb,
                stack.reg(heap, frame_base, ops.reg(0)),
                idx,
            ) {
                acc.store(v);
                return Flow::Sync;
            }
            let v = keyed_load_imm(ctx, stack.reg(heap, frame_base, ops.reg(0)), idx, Some(fb));
            if ctx.is_throw(v) {
                return Flow::Threw;
            }
            acc.store(v);
            Flow::Sync
        }
        Opcode::Throw | Opcode::ReThrow => {
            state.set_pending_exception(acc.get(heap));
            Flow::Threw
        }
        Opcode::Wide => unreachable!("wide prefix is consumed by the decoder"),
    }
}

pub use vm_core::{EntryFn, Interpreter};

pub struct MatchLoopInterpreter;

impl Interpreter for MatchLoopInterpreter {
    const EXECUTE: EntryFn = execute;
}
