//! Slow paths of the become interpreter: the handlers a fast path
//! tails into when its inline check fails (numeric fallbacks, IC
//! misses, allocation, proxies, construct) plus the shared exception
//! dispatch. Operand decoding falls back to `slow_layout` and the
//! `read_*` helpers (the fast paths' const-generic `Ops` cursor is
//! monomorphized per handler), and anything that can allocate
//! re-derive `ip`/`regs` from the frame header before resuming dispatch.

use bytecode::Opcode;
use vm_core::proxy::Proxy;
use vm_core::{
    CallTarget, CallableInfoObject, Callee, Coercion, Context, ContextInit, ContextState, Convert,
    Ctx, FixedArray, HandleSlice, Heap, Intrinsic, Object, Params, RuntimeContext, RuntimeIndex,
    ScopeInfo, Smi, StoreSemantics, Tagged, Value, VmError, spread_apply_args,
};

use crate::{
    ApplyOut, FloatReg, MethodCall, Regs, RootsArg, TableArg, dispatch_runtime_contiguous,
    dispatch_runtime_function, dispatch_runtime_method, push_callee_frame, read_signed,
    read_unsigned, resume, slow_layout, slow_next_pc,
};

macro_rules! slow_try {
    ($ctx:ident, $e:expr) => {
        match $e {
            Ok(v) => v,
            // the sentinel VALUE flows on to the caller's `become resume`,
            // which routes it into `throw_dispatch`
            Err(err) => unsafe { $ctx.raise_tag(err) },
        }
    };
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_box_number<'a>(
    ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    // the allocation may have moved the code object: re-derive the
    // pointer and the resume point from the (updated) frame header
    let code = ctx.code_ptr();
    let next = unsafe { slow_next_pc(ip) } as usize - ip as usize;
    let pc = ip as usize - code as usize + next;
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let h = table.get(unsafe { *code.add(pc) });
    become h(unsafe { code.add(pc) }, regs, v, ctx, table, roots, float)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_box_add_loc<'a>(
    ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let dst = read_signed(ip, base, stride);
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    regs.write(dst, v);
    let code = ctx.code_ptr();
    let next = unsafe { slow_next_pc(ip) } as usize - ip as usize;
    let pc = ip as usize - code as usize + next;
    let h = table.get(unsafe { *code.add(pc) });
    become h(unsafe { code.add(pc) }, regs, v, ctx, table, roots, float)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_box_sub_loc<'a>(
    ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let dst = read_signed(ip, base, stride);
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    regs.write(dst, v);
    let code = ctx.code_ptr();
    let next = unsafe { slow_next_pc(ip) } as usize - ip as usize;
    let pc = ip as usize - code as usize + next;
    let h = table.get(unsafe { *code.add(pc) });
    become h(unsafe { code.add(pc) }, regs, v, ctx, table, roots, float)
}

/// Best-effort cache fill: record the synthesized object's initial map
/// keyed on the closure, validated by (function map, `.prototype` slot
/// word) compares. Skips anything unusual (no data `prototype` slot,
/// non-object prototype) — those just stay uncached.
#[cold]
#[inline(never)]
pub fn construct_cache_record(
    heap: &mut Heap,
    state: &ContextState,
    scope: &vm_core::HandleScope,
    callee: vm_core::Handle<'_, Object>,
    obj: vm_core::Handle<'_, Object>,
) {
    // Safety: fresh synthesized-receiver word; the insert's allocation
    // runs before it reads parameters, so the raw word stays stable.
    let map: Tagged<vm_core::Map> = unsafe {
        Tagged::from_value_unchecked(obj.as_tagged(heap).as_ref().header.map.get(heap).raw())
    };
    // find the closure's `.prototype` data-property slot
    let proto_name = heap.known().strings.prototype.as_tagged(heap);
    let closure_map = callee.as_tagged(heap).as_ref().header.map.get(heap);
    let mut found = None;
    for d in closure_map.as_ref().descriptors() {
        if d.name(heap).raw() == proto_name.raw() && !d.flags().is_accessor() {
            found = Some(d.offset());
            break;
        }
    }
    let Some(offset) = found else { return };
    let proto_word = callee
        .as_tagged(heap)
        .as_ref()
        .slot(heap, offset)
        .get(heap)
        .raw();
    if !proto_word.is_strong_ptr() {
        return; // non-object prototype: ES falls back, stays uncached
    }
    // Safety: fresh rooted-slot words; the insert's only allocation runs
    // before it reads parameters, so raw words stay stable.
    let proto: Tagged<Value> = unsafe { Tagged::from_value_unchecked(proto_word) };
    let _ = scope;
    // Safety: the insert's only allocation (the first-use table) runs
    // before it reads these words — they stay stable throughout.
    let callee_obj: Tagged<Object> =
        unsafe { Tagged::from_value_unchecked(callee.as_tagged(heap).raw()) };
    state.construct_cache_insert(heap, callee_obj, proto, offset, map);
}

#[cold]
#[inline(never)]
fn create_closure_slow<'a>(ctx: &Ctx<'a>, info_idx: usize) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let base = ctx.frame_base();
        let Some(info) =
            scope.cast::<CallableInfoObject>(ctx.constants_ref(heap).at(heap, info_idx))
        else {
            return Err(VmError::Type);
        };
        let context = scope
            .cast::<Context>(ctx.stack().context_slot(base).get(heap))
            .expect("frame context slot holds a Context");
        let obj = Object::create_closure(heap, &scope, info, context)?;
        Ok(obj.erase())
    })
}

#[cold]
#[inline(never)]
fn create_function_context_slow<'a>(
    ctx: &Ctx<'a>,
    scope_idx: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let base = ctx.frame_base();
        let outer = scope
            .cast::<Context>(ctx.stack().context_slot(base).get(heap))
            .expect("frame context slot holds a Context");
        let count = ctx
            .constants_ref(heap)
            .at(heap, scope_idx)
            .get_as::<ScopeInfo>()
            .map(|r| r.as_ref().names.get(heap).len())
            .ok_or(VmError::Type)?;
        let scope_info = scope
            .cast::<ScopeInfo>(ctx.constants_ref(heap).at(heap, scope_idx))
            .expect("constants slot holds a ScopeInfo");
        let slots = if count == 0 {
            heap.known().empty_fixed_array
        } else {
            let values = scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
            heap.allocate_handle::<FixedArray>(values, &scope)
        };
        let ctx_obj = heap.allocate::<Context>(ContextInit {
            outer: Some(outer),
            slots,
            scope_info,
        });
        Ok(ctx_obj.erase())
    })
}

#[cold]
#[inline(never)]
pub fn proxy_apply_slow<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args_base: i32,
    count: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let callee = scope.handle(callee);
        let base = ctx.frame_base();
        let staged = ctx.stack().args(base, args_base, count);
        match Proxy::apply(vm, heap, state, callee, staged)? {
            Coercion::Threw => Ok(ctx.exception_word()),
            Coercion::Value(v) => Ok(v),
        }
    })
}

#[cold]
#[inline(never)]
pub fn proxy_apply_regs_slow<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    srcs: &[i32],
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let base = ctx.frame_base();
    let heap = unsafe { ctx.heap_mut() };
    let (saved_top, staged) = ctx.stack().stage_args_regs(heap, base, srcs)?;
    let result = state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let callee = scope.handle(callee);
        match Proxy::apply(vm, heap, state, callee, staged)? {
            Coercion::Threw => Ok(ctx.exception_word()),
            Coercion::Value(v) => Ok(v),
        }
    });
    ctx.stack().set_top(saved_top);
    result
}

/// Resolve the (already reshaped) call and push a frame, dispatch a
/// runtime callee, or report an intrinsic/proxy for the caller to tail
/// into. `args[0]` is the receiver of the call being made.
#[cold]
#[inline(never)]
pub fn intrinsic_call_scattered<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    f: Tagged<'a, Value>,
    srcs: &[i32],
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), f) {
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
            ..
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            push_callee_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info: info,
                    context: context.erase(),
                    register_count: register_count,
                    formal_min: formal_min,
                },
                Params::Scattered(srcs),
            )
        }
        Some(CallTarget::Runtime(idx)) => {
            Ok(MethodCall::Value(dispatch_runtime_method(ctx, idx, srcs)))
        }
        Some(CallTarget::Intrinsic(_)) => {
            // a nested intrinsic (`f.call.call(g, x)`): unwrap it against
            // the reshaped window, never against the original operands
            let base = ctx.frame_base();
            let (saved_top, staged) = ctx.stack().stage_args_regs(ctx.heap(), base, srcs)?;
            let out = intrinsic_apply_call(ctx, pc, size, f, staged);
            ctx.stack().set_top(saved_top);
            match out? {
                ApplyOut::Frame(frame) => Ok(MethodCall::Frame(frame)),
                ApplyOut::Value(v) => Ok(MethodCall::Value(v)),
            }
        }
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        None => Err(VmError::Type),
    }
}

/// `intrinsic_call_scattered` for the contiguous call window.
#[cold]
#[inline(never)]
pub fn intrinsic_call_contiguous<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    f: Tagged<'a, Value>,
    base: i32,
    count: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), f) {
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
            ..
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            push_callee_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info: info,
                    context: context.erase(),
                    register_count: register_count,
                    formal_min: formal_min,
                },
                Params::Window {
                    base: base,
                    count: count,
                },
            )
        }
        Some(CallTarget::Runtime(idx)) => Ok(MethodCall::Value(dispatch_runtime_contiguous(
            ctx, idx, base, count,
        ))),
        Some(CallTarget::Intrinsic(_)) => {
            // a nested intrinsic (`f.call.call(g, x)`): unwrap it against
            // the reshaped window, never against the original operands
            let staged = ctx.stack().args(ctx.frame_base(), base, count);
            match intrinsic_apply_call(ctx, pc, size, f, staged)? {
                ApplyOut::Frame(frame) => Ok(MethodCall::Frame(frame)),
                ApplyOut::Value(v) => Ok(MethodCall::Value(v)),
            }
        }
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        None => Err(VmError::Type),
    }
}

/// Resolve a call on `target` with a staged argument window
/// (`args[0]` = receiver) and push the callee frame when it is bytecode.
/// Nested intrinsics (`f.call.apply(...)`) are unwrapped recursively.
#[cold]
#[inline(never)]
pub fn intrinsic_apply_call<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    target: Tagged<'a, Value>,
    args: HandleSlice<'_>,
) -> Result<ApplyOut<'a>, VmError> {
    match Object::call_target(ctx.heap(), target) {
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
            ..
        }) => {
            if kind.is_class_constructor() {
                return Ok(ApplyOut::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            let heap = unsafe { ctx.heap_mut() };
            let frame = ctx.stack().push_frame(
                heap,
                ctx.caller_meta(pc + size, pc),
                Callee {
                    callable: target.erase(),
                    info,
                    context: context.erase(),
                    register_count,
                    formal_min,
                },
                heap.known().undefined.as_tagged(heap).erase(),
                Params::Slice(args),
            )?;
            ctx.set_frame_base(frame.base);
            Ok(ApplyOut::Frame(frame))
        }
        Some(CallTarget::Runtime(idx)) => {
            // the spread is rooted by the caller's scope staging
            let f = ctx.vm().runtime(RuntimeIndex(idx));
            let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
            Ok(ApplyOut::Value(f(nctx, args)))
        }
        Some(CallTarget::Intrinsic(intrinsic)) => {
            // unwrap the nested intrinsic and retry on the reshaped args
            match intrinsic {
                Intrinsic::FunctionCall => {
                    let f = args
                        .get(0)
                        .map(|h| h.as_tagged(ctx.heap()))
                        .ok_or(VmError::Arity)?;
                    let rest = args.slice_from(1);
                    intrinsic_apply_call(ctx, pc, size, f, rest)
                }
                Intrinsic::FunctionApply => {
                    let f = args
                        .get(0)
                        .map(|h| h.as_tagged(ctx.heap()))
                        .ok_or(VmError::Arity)?;
                    let this_arg = args
                        .get(1)
                        .map(|h| h.as_tagged(ctx.heap()))
                        .unwrap_or_else(|| ctx.undefined_word());
                    let array = args.get(2).map(|h| h.as_tagged(ctx.heap()));
                    ctx.state()
                        .handle_scope(|scope| -> Result<ApplyOut<'a>, VmError> {
                            // Safety: fresh reads of rooted slots, staged before
                            // the frame push below can move anything.
                            let staged =
                                scope.stage(&spread_apply_args(ctx.heap(), this_arg, array));
                            intrinsic_apply_call(ctx, pc, size, f, staged)
                        })
                }
            }
        }
        Some(CallTarget::Proxy(_)) => {
            let vm = ctx.vm();
            let heap = unsafe { ctx.heap_mut() };
            let state = ctx.state();
            let result = state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
                let target = scope.handle(target);
                match Proxy::apply(vm, heap, state, target, args)? {
                    Coercion::Threw => Ok(ctx.exception_word()),
                    Coercion::Value(v) => Ok(v),
                }
            });
            Ok(ApplyOut::Value(result?))
        }
        None => Err(VmError::Type),
    }
}

/// Exception dispatch: walk the ovm frames from the faulting one, consulting
/// each frame's static handler table for a try range covering
/// `fault_pc`; on a hit, take the pending exception and tail-dispatch
/// into the handler with it in the accumulator. Pops frames until the
/// anchor; then the sentinel escapes to the `execute` caller with the
/// pending exception left set. A termination is uncatchable: no
/// handler may observe it, so it unwinds straight out.
#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn throw_dispatch<'a>(
    fault_ip: *const u8,
    _regs: Regs,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let fault_pc = fault_ip as usize - ctx.code_ptr() as usize;
    match unsafe { vm_core::interp::unwind(ctx, fault_pc) } {
        vm_core::interp::Unwind::Caught { pc: handler_pc, ex } => {
            // the walk itself never allocates, so the re-derived
            // pointers are stable
            let code = ctx.code_ptr();
            let regs = unsafe { Regs::new(ctx.regs_ptr()) };
            let ip = unsafe { code.add(handler_pc) };
            let h = table.get(unsafe { *ip });
            become h(ip, regs, ex, ctx, table, roots, float)
        }
        vm_core::interp::Unwind::Escaped => ctx.exception_word(),
    }
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_add<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let lhs = regs.read(read_signed(ip, base, stride), ctx);
    let v = unsafe { vm_core::cold::add(ctx, lhs, acc) };
    let pc = ip as usize - ctx.code_ptr() as usize;
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_numeric<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let lhs = regs.read(read_signed(ip, base, stride), ctx);
    let op = unsafe { Opcode::from_byte_unchecked(*ip.add((stride == 2) as usize)) };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::Sub => |a, b| a - b,
        Opcode::Mul => |a, b| a * b,
        Opcode::Mod => |a, b| a % b,
        Opcode::Exp => |a, b| a.powf(b),
        _ => |a, b| a / b,
    };
    let v = unsafe { vm_core::cold::numeric(ctx, lhs, acc, f) };
    let pc = ip as usize - ctx.code_ptr() as usize;
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_add_immediate<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let lhs = regs.read(read_signed(ip, base, stride), ctx);
    let imm = Smi::new(read_signed(ip, base + stride, stride) as i64).into_tagged();
    let v = unsafe { vm_core::cold::add(ctx, lhs, imm) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_numeric_immediate<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let lhs = regs.read(read_signed(ip, base, stride), ctx);
    let imm = Smi::new(read_signed(ip, base + stride, stride) as i64).into_tagged();
    let op = unsafe { Opcode::from_byte_unchecked(*ip.add((stride == 2) as usize)) };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::SubImmediate => |a, b| a - b,
        Opcode::MulImmediate => |a, b| a * b,
        Opcode::ModImmediate => |a, b| a % b,
        Opcode::ExpImmediate => |a, b| a.powf(b),
        _ => |a, b| a / b,
    };
    let v = unsafe { vm_core::cold::numeric(ctx, lhs, imm, f) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_negate<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = unsafe { vm_core::cold::negate(ctx, acc) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
fn incdec_slow<'a>(
    ip: *const u8,
    ctx: &Ctx<'a>,
    regs: Regs,
    delta: f64,
) -> Result<Tagged<'a, Value>, VmError> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let r = read_signed(ip, base, stride);
    let old = regs.read(r, ctx);
    let heap = unsafe { ctx.heap_mut() };
    if let Some(bits) = old.smi_bits() {
        let new = heap.new_number((bits >> 1) as f64 + delta);
        regs.write(r, new);
        return Ok(old);
    }
    let vm = ctx.vm();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let old = scope.handle(old);
        let Some(n) = Object::to_numeric(vm, heap, state, old)? else {
            return Ok(ctx.exception_word());
        };
        let old_num =
            if n.fract() == 0.0 && Smi::in_range(n as i64) && !(n == 0.0 && n.is_sign_negative()) {
                scope.handle(Smi::new(n as i64).into_tagged())
            } else {
                scope.handle(heap.new_number(n))
            };
        let new = heap.new_number(n + delta);
        regs.write(r, new);
        Ok(old_num.as_tagged(heap))
    })
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_inc_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = slow_try!(ctx, incdec_slow(ip, ctx, regs, 1.0));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_dec_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = slow_try!(ctx, incdec_slow(ip, ctx, regs, -1.0));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

fn loc_op_slow<'a>(
    ip: *const u8,
    ctx: &Ctx<'a>,
    regs: Regs,
    sub: bool,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = slow_layout(ip);
    let dst = read_signed(ip, base, stride);
    let src = read_signed(ip, base + stride, stride);
    let lhs = regs.read(dst, ctx);
    let rhs = regs.read(src, ctx);
    let v = if sub {
        unsafe { vm_core::cold::numeric(ctx, lhs, rhs, |a, b| a - b) }
    } else {
        unsafe { vm_core::cold::add(ctx, lhs, rhs) }
    };
    if !ctx.is_throw(v) {
        regs.write(dst, v);
    }
    Ok(v)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_add_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let v = slow_try!(ctx, loc_op_slow(ip, ctx, regs, false));
    let pc = ip as usize - ctx.code_ptr() as usize;
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_sub_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let v = slow_try!(ctx, loc_op_slow(ip, ctx, regs, true));
    let pc = ip as usize - ctx.code_ptr() as usize;
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_keyed_load_reg<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let key = regs.read(read_signed(ip, base + stride, stride), ctx);
    let fb = read_unsigned(ip, base + 2 * stride, stride);
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = unsafe { vm_core::cold::keyed_load(ctx, recv, key, Some(fb)) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_equal<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let other = regs.read(read_signed(ip, base, stride), ctx);
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = unsafe { vm_core::cold::compare(ctx, 0, acc, other) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_less_than<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let other = regs.read(read_signed(ip, base, stride), ctx);
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = unsafe { vm_core::cold::compare(ctx, 2, acc, other) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_greater_than<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let other = regs.read(read_signed(ip, base, stride), ctx);
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = unsafe { vm_core::cold::compare(ctx, 4, acc, other) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_compare_jump<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let next = slow_next_pc(ip);
    let r = read_signed(ip, base, stride);
    let kind = read_unsigned(ip, base + stride, stride);
    let off = read_signed(ip, base + 2 * stride, stride);
    let other = regs.read(r, ctx);
    let cmp = (kind / 2) as u8;
    let b = match (Convert::as_number(acc), Convert::as_number(other)) {
        (Some(a), Some(b)) => match cmp {
            0 | 1 => a == b,
            2 => a < b,
            3 => a <= b,
            4 => a > b,
            _ => a >= b,
        },
        _ => {
            let pc = ip as usize - ctx.code_ptr() as usize;
            let v = unsafe { vm_core::cold::compare(ctx, cmp, acc, other) };
            if ctx.is_throw(v) {
                become resume(
                    unsafe { ctx.code_ptr().add(pc) },
                    regs,
                    v,
                    ctx,
                    table,
                    roots,
                    float,
                )
            }
            Convert::is_truthy(ctx.heap(), v)
        }
    };
    let boolean = Convert::boolean(ctx.heap(), b);
    let dest = if b != (kind % 2 == 1) {
        ip.wrapping_offset(off as isize)
    } else {
        next
    };
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let h = table.get(unsafe { *dest });
    become h(dest, regs, boolean, ctx, table, roots, float)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_named_load<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let name_idx = read_unsigned(ip, base + stride, stride);
    let fb_slot = read_unsigned(ip, base + 2 * stride, stride);
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = unsafe { vm_core::cold::named_load(ctx, recv, name_idx, fb_slot) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_keyed_load<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let fb = read_unsigned(ip, base + stride, stride);
    let v = unsafe { vm_core::cold::keyed_load(ctx, recv, acc, Some(fb)) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_keyed_load_imm<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let idx = read_unsigned(ip, base + stride, stride);
    let v = unsafe { vm_core::cold::keyed_load_imm(ctx, recv, idx) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_keyed_store<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let key = regs.read(read_signed(ip, base + stride, stride), ctx);
    let fb = read_unsigned(ip, base + 2 * stride, stride);
    let v = unsafe {
        vm_core::cold::keyed_store(ctx, recv, key, acc, Some(fb), StoreSemantics::Shadow)
    };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_keyed_store_no_shadow<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let key = regs.read(read_signed(ip, base + stride, stride), ctx);
    let fb = read_unsigned(ip, base + 2 * stride, stride);
    let v = unsafe {
        vm_core::cold::keyed_store(ctx, recv, key, acc, Some(fb), StoreSemantics::WriteThrough)
    };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_global_load<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let name_idx = read_unsigned(ip, base, stride);
    let fb_slot = read_unsigned(ip, base + stride, stride);
    let v = unsafe { vm_core::cold::global_load(ctx, name_idx, fb_slot, true) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_store_named<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let name_idx = read_unsigned(ip, base + stride, stride);
    let fb_slot = read_unsigned(ip, base + 2 * stride, stride);
    let v = unsafe { vm_core::cold::store_named(ctx, recv, name_idx, fb_slot, acc) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
pub fn slow_call_method_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    srcs: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) })),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    None,
                )
            };
            Ok(MethodCall::Value(dispatch_runtime_method(ctx, idx, srcs)))
        }
        Some(CallTarget::Intrinsic(i)) => {
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    None,
                )
            };
            Ok(MethodCall::Intrinsic(i))
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    Some(info),
                )
            };
            push_callee_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info: info,
                    context: context.erase(),
                    register_count: register_count,
                    formal_min: formal_min,
                },
                Params::Scattered(srcs),
            )
        }
    }
}

#[cold]
#[inline(never)]
pub fn slow_call_function_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    args: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) })),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    None,
                )
            };
            Ok(MethodCall::Value(dispatch_runtime_function(ctx, idx, args)))
        }
        Some(CallTarget::Intrinsic(i)) => {
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    None,
                )
            };
            Ok(MethodCall::Intrinsic(i))
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    Some(info),
                )
            };
            push_callee_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info: info,
                    context: context.erase(),
                    register_count: register_count,
                    formal_min: formal_min,
                },
                Params::Function(args),
            )
        }
    }
}

#[cold]
#[inline(never)]
pub fn slow_call_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    base: i32,
    count: usize,
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) })),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    None,
                )
            };
            Ok(MethodCall::Value(dispatch_runtime_contiguous(
                ctx, idx, base, count,
            )))
        }
        Some(CallTarget::Intrinsic(i)) => {
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    None,
                )
            };
            Ok(MethodCall::Intrinsic(i))
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            unsafe {
                vm_core::ic::call_update(
                    ctx.heap(),
                    ctx.feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    Some(info),
                )
            };
            push_callee_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info: info,
                    context: context.erase(),
                    register_count: register_count,
                    formal_min: formal_min,
                },
                Params::Window {
                    base: base,
                    count: count,
                },
            )
        }
    }
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_construct<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let callee = regs.read(read_signed(ip, base, stride), ctx);
    let args_base = read_signed(ip, base + stride, stride);
    let count = read_unsigned(ip, base + 2 * stride, stride);
    let v = unsafe { vm_core::cold::construct(ctx, callee, args_base, count) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_create_closure<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let info_idx = read_unsigned(ip, base, stride);
    let v = slow_try!(ctx, create_closure_slow(ctx, info_idx));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_create_empty_array<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().js_array_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        obj,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_create_empty_object<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().object_initial_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        obj,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_create_bare_object<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().plain_object_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        obj,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
fn create_block_context_slow<'a>(
    ctx: &Ctx<'a>,
    count: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let base = ctx.frame_base();
        let outer = scope
            .cast::<Context>(ctx.stack().context_slot(base).get(heap))
            .expect("frame context slot holds a Context");
        let slots = if count == 0 {
            heap.known().empty_fixed_array
        } else {
            let values = scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
            heap.allocate_handle::<FixedArray>(values, &scope)
        };
        let ctx_obj = heap.allocate::<Context>(ContextInit {
            outer: Some(outer),
            slots,
            scope_info: heap.known().empty_scope_info,
        });
        Ok(ctx_obj.erase())
    })
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_create_block_context<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let count = read_unsigned(ip, base, stride);
    let v = slow_try!(ctx, create_block_context_slow(ctx, count));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_less_than_or_equal<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let other = regs.read(read_signed(ip, base, stride), ctx);
    let v = unsafe { vm_core::cold::compare(ctx, 3, acc, other) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_global_load_nothrow<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let name_idx = read_unsigned(ip, base, stride);
    let fb_slot = read_unsigned(ip, base + stride, stride);
    let v = unsafe { vm_core::cold::global_load(ctx, name_idx, fb_slot, false) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_store_global<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let name_idx = read_unsigned(ip, base, stride);
    let v = unsafe { vm_core::cold::store_global(ctx, name_idx, acc) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_store_named_no_shadow<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let name_idx = read_unsigned(ip, base + stride, stride);
    let v = unsafe { vm_core::cold::store_named_no_shadow(ctx, recv, name_idx, acc) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
fn instance_of_slow<'a>(
    ctx: &Ctx<'a>,
    object: Tagged<'_, Value>,
    callable: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let object = scope.handle(object);
        let callable = scope.handle(callable);
        match Object::instance_of(vm, heap, state, object, callable)? {
            None => Ok(ctx.exception_word()),
            Some(r) => Ok(Convert::boolean(heap, r)),
        }
    })
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_instance_of<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let callable = regs.read(read_signed(ip, base, stride), ctx);
    let v = slow_try!(ctx, instance_of_slow(ctx, acc, callable));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_greater_than_or_equal<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let other = regs.read(read_signed(ip, base, stride), ctx);
    let v = unsafe { vm_core::cold::compare(ctx, 5, acc, other) };
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
fn add_parent_slow<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let receiver = scope.handle(recv);
        let name = scope.handle(ctx.constants_ref(heap).at(heap, name_idx).erase().as_name());
        let value = scope.handle(value);
        let Some(receiver) = scope.cast::<Object>(receiver.as_tagged(heap)) else {
            return Err(VmError::Type);
        };
        Object::add_parent(heap, &scope, receiver, name, value)?;
        Ok(value.as_tagged(heap).erase())
    })
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_add_parent<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let name_idx = read_unsigned(ip, base + stride, stride);
    let v = slow_try!(ctx, add_parent_slow(ctx, recv, name_idx, acc));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_create_function_context<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let scope_idx = read_unsigned(ip, base, stride);
    let v = slow_try!(ctx, create_function_context_slow(ctx, scope_idx));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_proxy_apply<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let callee = regs.read(read_signed(ip, base, stride), ctx);
    let args_base = read_signed(ip, base + stride, stride);
    let count = read_unsigned(ip, base + 2 * stride, stride);
    let v = slow_try!(ctx, proxy_apply_slow(ctx, callee, args_base, count));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_call_method_proxy<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let wide = unsafe { *ip } == Opcode::Wide as u8;
    let argc = match unsafe { Opcode::from_byte_unchecked(unsafe { *ip.add(wide as usize) }) } {
        Opcode::CallMethod0 => 0usize,
        Opcode::CallMethod1 => 1,
        _ => 2,
    };
    let callee = regs.read(read_signed(ip, base, stride), ctx);
    let mut srcs = [0i32; 3];
    for (i, src) in srcs.iter_mut().enumerate().take(argc + 1) {
        *src = read_signed(ip, base + (i + 1) * stride, stride);
    }
    let v = slow_try!(ctx, proxy_apply_regs_slow(ctx, callee, &srcs[..argc + 1]));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}

#[cold]
#[inline(never)]
fn proxy_apply_function_slow<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args: &[i32],
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let base = ctx.frame_base();
    let heap = unsafe { ctx.heap_mut() };
    let (saved_top, staged) = ctx.stack().stage_function_args(heap, base, args)?;
    let result = state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let callee = scope.handle(callee);
        match Proxy::apply(vm, heap, state, callee, staged)? {
            Coercion::Threw => Ok(ctx.exception_word()),
            Coercion::Value(v) => Ok(v),
        }
    });
    ctx.stack().set_top(saved_top);
    result
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_call_function_proxy<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = slow_layout(ip);
    let wide = unsafe { *ip } == Opcode::Wide as u8;
    let argc = match unsafe { Opcode::from_byte_unchecked(unsafe { *ip.add(wide as usize) }) } {
        Opcode::CallFunction0 => 0usize,
        Opcode::CallFunction1 => 1,
        _ => 2,
    };
    let callee = regs.read(read_signed(ip, base, stride), ctx);
    let mut args = [0i32; 2];
    for (i, arg) in args.iter_mut().enumerate().take(argc) {
        *arg = read_signed(ip, base + (i + 1) * stride, stride);
    }
    let v = slow_try!(ctx, proxy_apply_function_slow(ctx, callee, &args[..argc]));
    become resume(
        unsafe { ctx.code_ptr().add(pc) },
        regs,
        v,
        ctx,
        table,
        roots,
        float,
    )
}
