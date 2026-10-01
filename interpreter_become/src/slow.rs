//! Slow paths of the become interpreter: the handlers a fast path
//! tails into when its inline check fails (numeric fallbacks, IC
//! misses, allocation, proxies, construct) plus the shared exception
//! dispatch. Each handler is monomorphized on the operand `STRIDE` and
//! decodes through the same const-generic `Ops` cursor as the fast
//! paths; anything that can allocate re-derives `ip`/`regs` from the
//! frame header before resuming dispatch.

use bytecode::Opcode;
use vm_core::proxy::Proxy;
use vm_core::{
    CallTarget, CallableInfoObject, Callee, Coercion, Context, ContextInit, ContextState, Convert,
    Ctx, FixedArray, HandleSlice, Heap, Object, Params, ScopeInfo, Smi, StoreSemantics, Tagged,
    Value, VmError,
};

use crate::{
    FloatReg, MethodCall, Ops, Regs, RootsArg, TableArg, base, dispatch_runtime_contiguous,
    dispatch_runtime_function, dispatch_runtime_method, push_callee_frame, read_signed, resume,
    slow_next_pc,
};

/// Define a become-interpreter slow-path handler (always `pub`). The
/// expansion snapshots the faulting instruction's code-relative pc
/// BEFORE the body runs and resumes from the post-body code base: the
/// body may allocate and move the code object, after which the old `ip`
/// no longer pairs with the frame-header base. Bodies decode their
/// operands from `ip` first; anything else derived from `ip` that must
/// survive an allocation has to be computed code-relative up front
/// (like `slow_compare_jump` does for its jump targets).
macro_rules! slow_handler {
    ($name:ident |$ip:ident, $ops:ident, $regs:ident, $acc:ident, $ctx:ident, $float:ident| $body:block) => {
        #[cold]
        #[inline(never)]
        #[rustc_align(32)]
        pub extern "rust-preserve-none" fn $name<'a, const STRIDE: usize>(
            $ip: *const u8,
            $regs: Regs,
            $acc: Tagged<'a, Value>,
            $ctx: &Ctx<'a>,
            table: TableArg<'a>,
            roots: RootsArg<'a>,
            $float: FloatReg,
        ) -> Tagged<'a, Value> {
            // snapshot while `ip` still pairs with the current base
            let pc = $ip as usize - $ctx.code_ptr() as usize;
            let $ops = Ops::<STRIDE>::from_ip($ip);
            let v = $body;
            become resume(
                unsafe { $ctx.code_ptr().add(pc) },
                $regs,
                v,
                $ctx,
                table,
                roots,
                $float,
            )
        }
    };
}

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

slow_handler!(slow_box_number |ip, ops, regs, acc, ctx, float| {
    unsafe { ctx.heap_mut() }.new_float(float.get())
});

slow_handler!(slow_box_add_loc |ip, ops, regs, acc, ctx, float| {
    let dst = ops.signed::<0>();
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    regs.write(dst, v);
    v
});

slow_handler!(slow_box_sub_loc |ip, ops, regs, acc, ctx, float| {
    let dst = ops.signed::<0>();
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    regs.write(dst, v);
    v
});

slow_handler!(slow_add |ip, ops, regs, acc, ctx, float| {
    let lhs = regs.read(ops.signed::<0>(), ctx);
    unsafe { vm_core::cold::add(ctx, lhs, acc) }
});

slow_handler!(slow_numeric |ip, ops, regs, acc, ctx, float| {
    let lhs = regs.read(ops.signed::<0>(), ctx);
    let op = unsafe { ops.op() };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::Sub => |a, b| a - b,
        Opcode::Mul => |a, b| a * b,
        Opcode::Mod => |a, b| a % b,
        Opcode::Exp => |a, b| a.powf(b),
        _ => |a, b| a / b,
    };
    let v = unsafe { vm_core::cold::numeric(ctx, lhs, acc, f) };
    v
});

slow_handler!(slow_add_immediate |ip, ops, regs, acc, ctx, float| {
    let lhs = regs.read(ops.signed::<0>(), ctx);
    let imm = Smi::new(ops.signed::<1>() as i64).into_tagged();
    unsafe { vm_core::cold::add(ctx, lhs, imm) }
});

slow_handler!(slow_numeric_immediate |ip, ops, regs, acc, ctx, float| {
    let lhs = regs.read(ops.signed::<0>(), ctx);
    let imm = Smi::new(ops.signed::<1>() as i64).into_tagged();
    let op = unsafe { ops.op() };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::SubImmediate => |a, b| a - b,
        Opcode::MulImmediate => |a, b| a * b,
        Opcode::ModImmediate => |a, b| a % b,
        Opcode::ExpImmediate => |a, b| a.powf(b),
        _ => |a, b| a / b,
    };
    unsafe { vm_core::cold::numeric(ctx, lhs, imm, f) }
});

slow_handler!(slow_negate |ip, ops, regs, acc, ctx, float| {
    unsafe { vm_core::cold::negate(ctx, acc) }
});

slow_handler!(slow_inc_loc |ip, ops, regs, acc, ctx, float| {
    slow_try!(ctx, incdec_slow::<STRIDE>(ip, ctx, regs, 1.0))
});

slow_handler!(slow_dec_loc |ip, ops, regs, acc, ctx, float| {
    slow_try!(ctx, incdec_slow::<STRIDE>(ip, ctx, regs, -1.0))
});

slow_handler!(slow_add_loc |ip, ops, regs, acc, ctx, float| {
    let v = slow_try!(ctx, loc_op_slow::<STRIDE>(ip, ctx, regs, false));
    v
});

slow_handler!(slow_sub_loc |ip, ops, regs, acc, ctx, float| {
    let v = slow_try!(ctx, loc_op_slow::<STRIDE>(ip, ctx, regs, true));
    v
});

slow_handler!(slow_keyed_load_reg |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let key = regs.read(ops.signed::<1>(), ctx);
    let fb = ops.unsigned::<2>();
    unsafe { vm_core::cold::keyed_load(ctx, recv, key, Some(fb)) }
});

slow_handler!(slow_equal |ip, ops, regs, acc, ctx, float| {
    let other = regs.read(ops.signed::<0>(), ctx);
    unsafe { vm_core::cold::compare(ctx, 0, acc, other) }
});

slow_handler!(slow_less_than |ip, ops, regs, acc, ctx, float| {
    let other = regs.read(ops.signed::<0>(), ctx);
    unsafe { vm_core::cold::compare(ctx, 2, acc, other) }
});

slow_handler!(slow_greater_than |ip, ops, regs, acc, ctx, float| {
    let other = regs.read(ops.signed::<0>(), ctx);
    unsafe { vm_core::cold::compare(ctx, 4, acc, other) }
});

#[cold]
#[inline(never)]
#[rustc_align(32)]
pub extern "rust-preserve-none" fn slow_compare_jump<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::from_ip(ip);
    // every target is captured code-relative BEFORE the compare below:
    // it may allocate and move the code object, after which the old `ip`
    // no longer pairs with the frame-header base
    let pc = ip as usize - ctx.code_ptr() as usize;
    let next_pc = slow_next_pc(ip) as usize - ctx.code_ptr() as usize;
    let r = ops.signed::<0>();
    let kind = ops.unsigned::<1>();
    let off = ops.signed::<2>();
    let jump_pc = (pc as isize + off as isize) as usize;
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
    let code = ctx.code_ptr();
    let dest = unsafe {
        if b != (kind % 2 == 1) {
            code.add(jump_pc)
        } else {
            code.add(next_pc)
        }
    };
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let h = table.get(unsafe { *dest });
    become h(dest, regs, boolean, ctx, table, roots, float)
}

slow_handler!(slow_named_load |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let name_idx = ops.unsigned::<1>();
    let fb_slot = ops.unsigned::<2>();
    unsafe { vm_core::cold::named_load(ctx, recv, name_idx, fb_slot) }
});

slow_handler!(slow_keyed_load |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let fb = ops.unsigned::<1>();
    unsafe { vm_core::cold::keyed_load(ctx, recv, acc, Some(fb)) }
});

slow_handler!(slow_keyed_load_imm |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let idx = ops.unsigned::<1>();
    unsafe { vm_core::cold::keyed_load_imm(ctx, recv, idx) }
});

slow_handler!(slow_keyed_store |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let key = regs.read(ops.signed::<1>(), ctx);
    let fb = ops.unsigned::<2>();
    unsafe { vm_core::cold::keyed_store(ctx, recv, key, acc, Some(fb), StoreSemantics::Shadow) }
});

slow_handler!(slow_keyed_store_no_shadow |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let key = regs.read(ops.signed::<1>(), ctx);
    let fb = ops.unsigned::<2>();
    unsafe {
        vm_core::cold::keyed_store(ctx, recv, key, acc, Some(fb), StoreSemantics::WriteThrough)
    }
});

slow_handler!(slow_global_load |ip, ops, regs, acc, ctx, float| {
    let name_idx = ops.unsigned::<0>();
    let fb_slot = ops.unsigned::<1>();
    unsafe { vm_core::cold::global_load(ctx, name_idx, fb_slot, true) }
});

slow_handler!(slow_store_named |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let name_idx = ops.unsigned::<1>();
    let fb_slot = ops.unsigned::<2>();
    unsafe { vm_core::cold::store_named(ctx, recv, name_idx, fb_slot, acc) }
});

slow_handler!(slow_construct |ip, ops, regs, acc, ctx, float| {
    let callee = regs.read(ops.signed::<0>(), ctx);
    let args_base = ops.signed::<1>();
    let count = ops.unsigned::<2>();
    unsafe { vm_core::cold::construct(ctx, callee, args_base, count) }
});

slow_handler!(slow_create_closure |ip, ops, regs, acc, ctx, float| {
    let info_idx = ops.unsigned::<0>();
    slow_try!(ctx, create_closure_slow(ctx, info_idx))
});

slow_handler!(slow_create_empty_array |ip, ops, regs, acc, ctx, float| {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().js_array_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    obj
});

slow_handler!(slow_create_empty_object |ip, ops, regs, acc, ctx, float| {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().object_initial_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    obj
});

slow_handler!(slow_create_bare_object |ip, ops, regs, acc, ctx, float| {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().plain_object_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    obj
});

slow_handler!(slow_create_block_context |ip, ops, regs, acc, ctx, float| {
    let count = ops.unsigned::<0>();
    slow_try!(ctx, create_block_context_slow(ctx, count))
});

slow_handler!(slow_less_than_or_equal |ip, ops, regs, acc, ctx, float| {
    let other = regs.read(ops.signed::<0>(), ctx);
    unsafe { vm_core::cold::compare(ctx, 3, acc, other) }
});

slow_handler!(slow_global_load_nothrow |ip, ops, regs, acc, ctx, float| {
    let name_idx = ops.unsigned::<0>();
    let fb_slot = ops.unsigned::<1>();
    unsafe { vm_core::cold::global_load(ctx, name_idx, fb_slot, false) }
});

slow_handler!(slow_store_global |ip, ops, regs, acc, ctx, float| {
    let name_idx = ops.unsigned::<0>();
    unsafe { vm_core::cold::store_global(ctx, name_idx, acc) }
});

slow_handler!(slow_store_named_no_shadow |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let name_idx = ops.unsigned::<1>();
    unsafe { vm_core::cold::store_named_no_shadow(ctx, recv, name_idx, acc) }
});

slow_handler!(slow_instance_of |ip, ops, regs, acc, ctx, float| {
    let callable = regs.read(ops.signed::<0>(), ctx);
    slow_try!(ctx, instance_of_slow(ctx, acc, callable))
});

slow_handler!(slow_greater_than_or_equal |ip, ops, regs, acc, ctx, float| {
    let other = regs.read(ops.signed::<0>(), ctx);
    unsafe { vm_core::cold::compare(ctx, 5, acc, other) }
});

slow_handler!(slow_add_parent |ip, ops, regs, acc, ctx, float| {
    let recv = regs.read(ops.signed::<0>(), ctx);
    let name_idx = ops.unsigned::<1>();
    slow_try!(ctx, add_parent_slow(ctx, recv, name_idx, acc))
});

slow_handler!(slow_create_function_context |ip, ops, regs, acc, ctx, float| {
    let scope_idx = ops.unsigned::<0>();
    slow_try!(ctx, create_function_context_slow(ctx, scope_idx))
});

slow_handler!(slow_proxy_apply |ip, ops, regs, acc, ctx, float| {
    let callee = regs.read(ops.signed::<0>(), ctx);
    let args_base = ops.signed::<1>();
    let count = ops.unsigned::<2>();
    slow_try!(ctx, proxy_apply_slow(ctx, callee, args_base, count))
});

slow_handler!(slow_call_method_proxy |ip, ops, regs, acc, ctx, float| {
    let base = base::<STRIDE>();
    let wide = unsafe { *ip } == Opcode::Wide as u8;
    let argc = match unsafe { Opcode::from_byte_unchecked(unsafe { *ip.add(wide as usize) }) } {
        Opcode::CallMethod0 => 0usize,
        Opcode::CallMethod1 => 1,
        _ => 2,
    };
    let callee = regs.read(ops.signed::<0>(), ctx);
    let recv = ops.signed::<1>();
    let mut pargs = [0i32; 2];
    for (i, arg) in pargs.iter_mut().enumerate().take(argc) {
        *arg = read_signed(ip, base + (2 + i) * STRIDE, STRIDE);
    }
    slow_try!(ctx, proxy_apply_regs_slow(ctx, callee, recv, pargs, argc))
});

slow_handler!(slow_call_function_proxy |ip, ops, regs, acc, ctx, float| {
    let base = base::<STRIDE>();
    let wide = unsafe { *ip } == Opcode::Wide as u8;
    let argc = match unsafe { Opcode::from_byte_unchecked(unsafe { *ip.add(wide as usize) }) } {
        Opcode::CallFunction0 => 0usize,
        Opcode::CallFunction1 => 1,
        _ => 2,
    };
    let callee = regs.read(ops.signed::<0>(), ctx);
    let mut pargs = [0i32; 2];
    for (i, arg) in pargs.iter_mut().enumerate().take(argc) {
        *arg = read_signed(ip, base + (i + 1) * STRIDE, STRIDE);
    }
    slow_try!(ctx, proxy_apply_function_slow(ctx, callee, pargs, argc))
});

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
    recv: i32,
    args: [i32; 2],
    argc: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let base = ctx.frame_base();
    let heap = unsafe { ctx.heap_mut() };
    let (saved_top, staged) = ctx.stack().stage_args_regs(heap, base, recv, args, argc)?;
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
fn incdec_slow<'a, const STRIDE: usize>(
    ip: *const u8,
    ctx: &Ctx<'a>,
    regs: Regs,
    delta: f64,
) -> Result<Tagged<'a, Value>, VmError> {
    let ops = Ops::<STRIDE>::from_ip(ip);
    let r = ops.signed::<0>();
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

fn loc_op_slow<'a, const STRIDE: usize>(
    ip: *const u8,
    ctx: &Ctx<'a>,
    regs: Regs,
    sub: bool,
) -> Result<Tagged<'a, Value>, VmError> {
    let ops = Ops::<STRIDE>::from_ip(ip);
    let dst = ops.signed::<0>();
    let src = ops.signed::<1>();
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
pub fn slow_call_method_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    recv: i32,
    args: [i32; 2],
    argc: usize,
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
            Ok(MethodCall::Value(dispatch_runtime_method(
                ctx, idx, recv, args, argc,
            )))
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
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
                },
                Params::MethodFast { recv, args, argc },
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
    args: [i32; 2],
    argc: usize,
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
            Ok(MethodCall::Value(dispatch_runtime_function(
                ctx, idx, args, argc,
            )))
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
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
                },
                Params::FunctionFast { args, argc },
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
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
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
pub fn proxy_apply_function_slow<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args: [i32; 2],
    argc: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let base = ctx.frame_base();
    let heap = unsafe { ctx.heap_mut() };
    let (saved_top, staged) = ctx.stack().stage_function_args(heap, base, args, argc)?;
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
