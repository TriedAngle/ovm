#![feature(explicit_tail_calls)]
#![feature(rust_preserve_none_cc)]
#![feature(fn_align)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_macros, unused_unsafe, unused_variables)]

use core::marker::PhantomData;

use bytecode::{Opcode, jump_target};
use vm_core::ic::{Hit, InlineCache, StoreHit, StoreOutcomeKind};
use vm_core::proxy::Proxy;
use vm_core::{
    CallTarget, Coercion, Compare, Context, ContextInit, ContextState, Convert, Key, SlotName,
    CallableInfoObject, DenseString, Errors, ExecuteFn, FixedArray, FrameMeta, Handle, HandleSlice,
    Heap, Hint, Interpreter, LoadOutcome, Lookup, Object, PropertyDescriptor,
    Register, RuntimeContext, RuntimeIndex, ScopeInfo, Smi, Stack, StackCache, StoreOutcome,
    StoreSemantics, Tagged, VM, Value, VmError,
};

pub struct BecomeInterpreter;

pub struct Ctx<'a> {
    vm: *const VM,
    heap: *mut Heap,
    state: *const ContextState,
    base_depth: usize,
    _heap: PhantomData<&'a mut Heap>,
}

impl<'a> Ctx<'a> {
    #[inline(always)]
    fn vm(&self) -> &'a VM {
        unsafe { &*self.vm }
    }
    #[inline(always)]
    fn heap(&self) -> &'a Heap {
        unsafe { &*self.heap }
    }
    #[inline(always)]
    unsafe fn heap_mut(&self) -> &'a mut Heap {
        unsafe { &mut *self.heap }
    }
    #[inline(always)]
    fn state(&self) -> &'a ContextState {
        unsafe { &*self.state }
    }
    #[inline(always)]
    fn stack(&self) -> &'a Stack {
        self.state().stack()
    }
    #[inline(always)]
    fn cache(&self) -> &'a StackCache {
        self.state().cache()
    }

    #[inline(always)]
    fn meta(&self, pc: usize) -> FrameMeta {
        let cache = self.cache();
        FrameMeta {
            base: cache.base(),
            pc,
            register_count: cache.register_count(),
            handler_pc: 0,
        }
    }

    #[inline(always)]
    fn code_ptr(&self) -> *const u8 {
        self.cache().code_ref(self.heap()).as_slice().as_ptr()
    }

    #[inline(always)]
    fn regs_ptr(&self) -> *mut Register {
        unsafe { self.stack().slots_ptr().add(self.cache().base()) }
    }

    #[inline(always)]
    fn exception_word(&self) -> Tagged<'a, Value> {
        let heap = self.heap();
        heap.known().exception.as_tagged(heap).erase()
    }

    #[inline(always)]
    fn is_throw(&self, v: Tagged<'_, Value>) -> bool {
        v == self.exception_word()
    }

    /// Materialize a VM error as the pending exception and return the
    /// exception sentinel (the `execute`-caller convention for throws).
    #[cold]
    #[inline(never)]
    unsafe fn raise(&self, err: VmError) -> Result<Tagged<'a, Value>, VmError> {
        let heap = self.heap_mut();
        let state = self.state();
        let ex = Errors::from_vm_error(self.vm(), heap, state, err)
            .expect("error materialization must not fail");
        state.set_pending_exception(ex);
        Ok(self.exception_word())
    }

    #[inline(always)]
    fn threw(&self) -> Result<Tagged<'a, Value>, VmError> {
        Ok(self.exception_word())
    }
}

#[inline(always)]
unsafe fn reg_read<'a>(regs: *mut Register, heap: &'a Heap, i: i32) -> Tagged<'a, Value> {
    (*regs.offset(i as isize)).get(heap)
}

#[inline(always)]
unsafe fn reg_write<'x, T: 'x>(regs: *mut Register, i: i32, v: Tagged<'x, T>) {
    (*regs.offset(i as isize)).store(v);
}


pub type Handler = for<'a> unsafe extern "rust-preserve-none" fn(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError>;

pub struct HandlerTable([Handler; 256]);

macro_rules! helpers {
    ($pc:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident, $stride:literal, $($arg:ident => $kind:ident),* $(,)?) => {
        const BASE: usize = if $stride == 2 { 2 } else { 1 };
        let mut _oi = BASE;
        $( let $arg = unsafe { $kind::read($pc, $code, _oi, $stride) }; _oi += $stride; )*
        #[allow(dead_code)]
        const SIZE: usize = BASE + $stride * <[()]>::len(&[$( { let _ = stringify!($arg); () } ),*]);

        #[allow(unused_variables, unused_macros)]
        let ($code, $regs) = ($code, $regs);
        macro_rules! reg { ($i:expr) => { unsafe { reg_read($regs, $ctx.heap(), $i) } } }
        macro_rules! set_reg { ($i:expr, $v:expr) => { unsafe { reg_write($regs, $i, $v) } } }
        macro_rules! dispatch {
            ($p:expr, $a:expr, $r:expr) => {
                unsafe {
                    let op = *$code.add($p) as usize;
                    let h = TABLE_NARROW.0[op];
                    become h($p, $code, $r, $a, $ctx)
                }
            };
            ($p:expr, $a:expr, $r:expr, $c:expr) => {
                unsafe {
                    let op = *$c.add($p) as usize;
                    let h = TABLE_NARROW.0[op];
                    become h($p, $c, $r, $a, $ctx)
                }
            };
        }
        macro_rules! next { ($a:expr) => { dispatch!($pc + SIZE, $a, $regs) } }
        macro_rules! jump { ($off:expr, $a:expr) => { dispatch!(jump_target($pc, $off), $a, $regs) } }
        /// Re-entry after a cold call: the code object may have moved, so
        /// the pc is re-derived from the (GC-updated) cache.
        macro_rules! reenter {
            ($pcrel:expr, $delta:expr, $a:expr) => {{
                let p = $pcrel + $delta;
                let c = $ctx.code_ptr();
                let r = $ctx.regs_ptr();
                dispatch!(p, $a, r, c)
            }};
        }
        macro_rules! bail { ($e:expr) => { return $ctx.raise($e) } }
        macro_rules! threw { () => { return $ctx.threw() } }
        macro_rules! cold {
            ($e:expr) => {{
                let v = $e?;
                if $ctx.is_throw(v) {
                    return Ok(v);
                }
                v
            }};
        }
    };
}

macro_rules! handlers {
    ($pc:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident; $( $op:ident : $n:ident / $w:ident ($($arg:ident => $kind:ident),*) $body:block )*) => {
        $(
            #[inline(never)]
            #[rustc_align(32)]
            unsafe extern "rust-preserve-none" fn $n<'a>(
                $pc: usize, $code: *const u8, $regs: *mut Register, $acc: Tagged<'a, Value>, $ctx: &Ctx<'a>,
            ) -> Result<Tagged<'a, Value>, VmError> {
                helpers!($pc, $code, $regs, $acc, $ctx, 1, $($arg => $kind),*);
                $body
            }
            #[inline(never)]
            #[rustc_align(32)]
            unsafe extern "rust-preserve-none" fn $w<'a>(
                $pc: usize, $code: *const u8, $regs: *mut Register, $acc: Tagged<'a, Value>, $ctx: &Ctx<'a>,
            ) -> Result<Tagged<'a, Value>, VmError> {
                helpers!($pc, $code, $regs, $acc, $ctx, 2, $($arg => $kind),*);
                $body
            }
        )*
    };
}


#[inline(never)]
#[rustc_align(32)]
unsafe extern "rust-preserve-none" fn op_wide<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    unsafe {
        let op = *code.add(pc + 1) as usize;
        let h = TABLE_WIDE.0[op];
        become h(pc, code, regs, acc, ctx)
    }
}

#[inline(never)]
#[rustc_align(32)]
unsafe extern "rust-preserve-none" fn op_trap<'a>(
    pc: usize,
    code: *const u8,
    _regs: *mut Register,
    _acc: Tagged<'a, Value>,
    _ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let op = unsafe { *code.add(pc) };
    panic!("become interpreter: opcode {op} (at +{pc}) not in the subset")
}

handlers!(pc, code, regs, acc, ctx;
    Load : op_load_n / op_load_w (r => signed) {
        next!(reg!(r))
    }

    Move : op_move_n / op_move_w (dst => signed, src => signed) {
        set_reg!(dst, reg!(src));
        next!(acc)
    }

    Store : op_store_n / op_store_w (r => signed) {
        set_reg!(r, acc);
        next!(acc)
    }

    LoadSmi : op_load_smi_n / op_load_smi_w (imm => signed) {
        next!(Smi::new(imm as i64).into_tagged())
    }

    LoadConstant : op_load_constant_n / op_load_constant_w (idx => unsigned) {
        let v = ctx.cache().constants_ref(ctx.heap()).at(ctx.heap(), idx);
        next!(v)
    }

    LoadZero : op_load_zero_n / op_load_zero_w () {
        next!(Smi::new(0).into_tagged())
    }

    LoadUndefined : op_load_undefined_n / op_load_undefined_w () {
        next!(ctx.heap().known().undefined.as_tagged(ctx.heap()).erase())
    }

    LoadTrue : op_load_true_n / op_load_true_w () {
        next!(ctx.heap().known().true_object.as_tagged(ctx.heap()).erase())
    }

    LoadFalse : op_load_false_n / op_load_false_w () {
        next!(ctx.heap().known().false_object.as_tagged(ctx.heap()).erase())
    }

    Add : op_add_n / op_add_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && let Some(sum) = a.checked_add(b)
        {
            next!(Tagged::from_smi_bits(sum))
        }
        let v = cold!(add_cold(ctx, acc, other));
        reenter!(pc, SIZE, v)
    }

    Sub : op_sub_n / op_sub_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && let Some(diff) = a.checked_sub(b)
        {
            next!(Tagged::from_smi_bits(diff))
        }
        let v = cold!(numeric_cold(ctx, acc, other, |a, b| a - b));
        reenter!(pc, SIZE, v)
    }

    Mul : op_mul_n / op_mul_w (r => signed) {
        let other = reg!(r);
        // untag one side only: `a * other` then already is the encoded
        // product, and checked_mul is the range check
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && let Some(product) = (a >> 1).checked_mul(b)
        {
            next!(Tagged::from_smi_bits(product))
        }
        let v = cold!(numeric_cold(ctx, acc, other, |a, b| a * b));
        reenter!(pc, SIZE, v)
    }

    Div : op_div_n / op_div_w (r => signed) {
        let other = reg!(r);
        // the operand shifts cancel in `a / b`; the quotient is untagged
        // and needs a checked re-tag
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && b != 0
            && a % b == 0
            && let Some(quotient) = a.checked_div(b)
            && let Some(encoded) = quotient.checked_mul(2)
        {
            next!(Tagged::from_smi_bits(encoded))
        }
        let v = cold!(numeric_cold(ctx, acc, other, |a, b| a / b));
        reenter!(pc, SIZE, v)
    }

    Equal : op_equal_n / op_equal_w (r => signed) {
        let other = reg!(r);
        // Smi == Smi is encoded-word equality; only mixed/float pairs need
        // the numeric conversion
        let b = match (acc.smi_bits(), other.smi_bits()) {
            (Some(a), Some(b)) => a == b,
            _ => match (Convert::as_number(acc), Convert::as_number(other)) {
                (Some(a), Some(b)) => a == b,
                _ => match compare_cold(ctx, 0, acc, other) {
                    Ok(Some(b)) => b,
                    Ok(None) => threw!(),
                    Err(err) => bail!(err),
                },
            },
        };
        next!(Convert::boolean(ctx.heap(), b))
    }

    ShiftRight : op_shift_right_n / op_shift_right_w (r => signed) {
        // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending; Smis only
        let (Some(a), Some(b)) = (acc.to_i64(), reg!(r).to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32).wrapping_shr(b as u32 & 31) as i64).into_tagged())
    }

    StoreKeyedProperty : op_store_keyed_n / op_store_keyed_w (recv => signed, key => signed, fb => unsigned) {
        let recv_w = reg!(recv);
        let key_w = reg!(key);
        if let Some(idx) = key_w.to_i64()
            && idx >= 0
            && let Some(obj) = recv_w.as_heap_object()
            && obj.as_ref().element_value(ctx.heap(), idx as usize).is_some()
            && Object::store_array_element_in_place(
                ctx.heap_mut(),
                recv_w,
                idx as usize,
                acc,
            )
            .is_ok()
        {
            next!(acc)
        }
        let v = cold!(keyed_store_cold(ctx, recv_w, key_w, acc));
        reenter!(pc, SIZE, v)
    }

    Negate : op_negate_n / op_negate_w () {
        if let Some(bits) = acc.smi_bits() {
            if bits == 0 {
                // preserve -0.0: `-0` must not fold into Smi 0
                let n = ctx.heap_mut().new_number(-0.0);
                reenter!(pc, SIZE, n)
            } else if let Some(neg) = bits.checked_neg() {
                // encoded negation: i64::MIN (Smi::MIN) overflows to a float
                next!(Tagged::from_smi_bits(neg))
            } else {
                let v = cold!(negate_cold(ctx, acc));
                reenter!(pc, SIZE, v)
            }
        } else {
            let v = cold!(negate_cold(ctx, acc));
            reenter!(pc, SIZE, v)
        }
    }

    CompareJump : op_compare_jump_n / op_compare_jump_w (r => signed, kind => unsigned, off => signed) {
        let other = reg!(r);
        let cmp = (kind / 2) as u8;
        let falsy_jump = kind % 2 == 1;
        // encoded Smis compare in the same order as their values
        let b = match (acc.smi_bits(), other.smi_bits()) {
            (Some(a), Some(b)) => match cmp {
                0 | 1 => a == b,
                2 => a < b,
                3 => a <= b,
                4 => a > b,
                _ => a >= b,
            },
            _ => match (Convert::as_number(acc), Convert::as_number(other)) {
                (Some(a), Some(b)) => match cmp {
                    0 | 1 => a == b,
                    2 => a < b,
                    3 => a <= b,
                    4 => a > b,
                    _ => a >= b,
                },
                _ => match compare_cold(ctx, cmp, acc, other) {
                    Ok(Some(b)) => b,
                    Ok(None) => threw!(),
                    Err(err) => bail!(err),
                },
            },
        };
        let boolean = Convert::boolean(ctx.heap(), b);
        if b != falsy_jump {
            jump!(off, boolean)
        }
        reenter!(pc, SIZE, boolean)
    }

    Jump : op_jump_n / op_jump_w (off => signed) {
        jump!(off, acc)
    }

    JumpIfTruthy : op_jump_if_truthy_n / op_jump_if_truthy_w (off => signed) {
        if Convert::is_truthy(ctx.heap(), acc) {
            jump!(off, acc)
        }
        next!(acc)
    }

    JumpIfFalsy : op_jump_if_falsy_n / op_jump_if_falsy_w (off => signed) {
        if !Convert::is_truthy(ctx.heap(), acc) {
            jump!(off, acc)
        }
        next!(acc)
    }

    JumpLoop : op_jump_loop_n / op_jump_loop_w (off => signed) {
        // the acc word must be GC-visible while parked: sync to the cache
        ctx.cache().acc_mut().store(acc);
        if ctx.heap_mut().safepoint_poll() {
            let state = ctx.state();
            state.set_termination(vm_core::Termination::Shutdown);
            let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap());
            state.set_pending_exception(undefined);
            threw!()
        }
        let acc = ctx.cache().acc(ctx.heap());
        let target = jump_target(pc, off);
        reenter!(target, 0, acc)
    }

    Throw : op_throw_n / op_throw_w () {
        ctx.state().set_pending_exception(acc);
        threw!()
    }

    Return : op_return_n / op_return_w () {
        if ctx.stack().frame_depth() == ctx.base_depth {
            return Ok(acc);
        }
        let low = Stack::frame_low(&ctx.meta(pc));
        let Some(caller) = ctx.stack().pop_frame(low) else {
            bail!(VmError::Type);
        };
        ctx.cache().load(ctx.stack(), caller, ctx.heap_mut());
        let rel = ctx.cache().pc();
        let base = ctx.code_ptr();
        dispatch!(rel, acc, ctx.regs_ptr(), base)
    }

    LoadNamedProperty : op_load_named_n / op_load_named_w (r => signed, name => unsigned, fb => unsigned) {
        let recv = reg!(r);
        if let Some(Hit::Value(v)) = InlineCache::try_load(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv,
        ) {
            next!(v)
        }
        let v = cold!(named_load_cold(ctx, recv, name, fb));
        reenter!(pc, SIZE, v)
    }

    LoadKeyedProperty : op_load_keyed_n / op_load_keyed_w (r => signed, _fb => unsigned) {
        let recv = reg!(r);
        if let Some(idx) = acc.to_i64()
            && idx >= 0
            && let Some(obj) = recv.as_heap_object()
            && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx as usize)
        {
            next!(v)
        }
        let v = cold!(keyed_load_cold(ctx, recv, acc));
        reenter!(pc, SIZE, v)
    }

    LoadElementImm : op_load_element_imm_n / op_load_element_imm_w (r => signed, idx => unsigned) {
        let recv = reg!(r);
        if let Some(obj) = recv.as_heap_object()
            && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx)
        {
            next!(v)
        }
        let v = cold!(keyed_load_imm_cold(ctx, recv, idx));
        reenter!(pc, SIZE, v)
    }

    LoadGlobal : op_load_global_n / op_load_global_w (name => unsigned, fb => unsigned) {
        let global = ctx.heap().known().global_object.as_tagged(ctx.heap()).erase();
        if let Some(Hit::Value(v)) = InlineCache::try_load(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            global,
        ) {
            next!(v)
        }
        let v = cold!(global_load_cold(ctx, name, fb));
        reenter!(pc, SIZE, v)
    }

    StoreNamedProperty : op_store_named_n / op_store_named_w (r => signed, name => unsigned, fb => unsigned) {
        let recv = reg!(r);
        let v = cold!(store_named_cold(ctx, recv, name, fb, acc));
        reenter!(pc, SIZE, v)
    }

    LoadContextSlot : op_load_context_slot_n / op_load_context_slot_w (slot => unsigned, depth => unsigned) {
        let meta = ctx.meta(0);
        let v = {
            let heap = ctx.heap();
            let Some(mut context) = ctx
                .stack()
                .context_slot(&meta)
                .get(heap)
                .get_as::<Context>()
            else {
                bail!(VmError::Type);
            };
            for _ in 0..depth {
                context = match context.as_ref().outer.get(heap) {
                    Some(context) => context,
                    None => bail!(VmError::Type),
                };
            }
            context.slots.get(heap).as_ref().element_slot(slot).get(heap)
        };
        next!(v)
    }

    StoreContextSlot : op_store_context_slot_n / op_store_context_slot_w (slot => unsigned, depth => unsigned) {
        let meta = ctx.meta(0);
        let heap = ctx.heap();
        let Some(mut context) = ctx
            .stack()
            .context_slot(&meta)
            .get(heap)
            .get_as::<Context>()
        else {
            bail!(VmError::Type);
        };
        for _ in 0..depth {
            context = match context.as_ref().outer.get(heap) {
                Some(context) => context,
                None => bail!(VmError::Type),
            };
        }
        let host = context.erase();
        context
            .slots
            .get(heap)
            .as_ref()
            .element_slot(slot)
            .set(heap, host, acc);
        next!(acc)
    }

    PushContext : op_push_context_n / op_push_context_w (r => signed) {
        let meta = ctx.meta(0);
        let old = ctx.stack().context_slot(&meta).get(ctx.heap());
        set_reg!(r, old);
        if acc.get_as::<Context>().is_none() {
            bail!(VmError::Type);
        }
        ctx.stack().context_slot(&meta).store(acc);
        next!(acc)
    }

    PopContext : op_pop_context_n / op_pop_context_w (r => signed) {
        let meta = ctx.meta(0);
        let context = reg!(r);
        if context.get_as::<Context>().is_none() {
            bail!(VmError::Type);
        }
        ctx.stack().context_slot(&meta).store(context);
        next!(acc)
    }

    CreateFunctionContext : op_create_function_context_n / op_create_function_context_w (scope => unsigned) {
        let v = cold!(create_function_context_cold(ctx, scope));
        reenter!(pc, SIZE, v)
    }

    CreateClosure : op_create_closure_n / op_create_closure_w (info => unsigned) {
        let v = cold!(create_closure_cold(ctx, info));
        reenter!(pc, SIZE, v)
    }

    CallRuntime : op_call_runtime_n / op_call_runtime_w (rt => unsigned, base => signed, count => unsigned) {
        let f = ctx.vm().runtime(RuntimeIndex(rt));
        let meta = ctx.meta(0);
        let args = ctx.stack().args(&meta, base, count);
        let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
        let result = f(nctx, args);
        match result {
            Ok(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            Err(err) => ctx.raise(err),
        }
    }

    CallNoFeedback : op_call_n / op_call_w (callee => signed, base => signed, count => unsigned) {
        let callee_word = reg!(callee);
        if Proxy::is_proxy(ctx.heap(), callee_word) {
            let v = cold!(proxy_apply_cold(ctx, callee_word, base, count));
            reenter!(pc, SIZE, v)
        }
        match Object::call_target(ctx.heap(), callee_word) {
            None => bail!(VmError::Type),
            Some(CallTarget::Runtime(idx)) => {
                let f = ctx.vm().runtime(RuntimeIndex(idx));
                let meta = ctx.meta(0);
                let args = ctx.stack().args(&meta, base, count);
                let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
                let result = f(nctx, args);
                match result {
                    Ok(v) => {
                        if ctx.is_throw(v) {
                            return Ok(v);
                        }
                        reenter!(pc, SIZE, v)
                    }
                    Err(err) => ctx.raise(err),
                }
            }
            Some(CallTarget::Bytecode(target, register_count, kind)) => {
                if kind.is_class_constructor() {
                    bail!(VmError::Type);
                }
                let heap = ctx.heap_mut();
                let meta = ctx.meta(pc + SIZE);
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
                let undefined = heap.known().undefined.as_tagged(heap).erase();
                let frame = ctx.stack().push_frame(
                    heap,
                    meta,
                    pc,
                    target.erase(),
                    register_count,
                    context,
                    base,
                    count,
                    undefined,
                    formal_min,
                )?;
                ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
                let callee_pc = ctx.cache().pc();
                let callee_base = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(callee_pc, undefined, r, callee_base)
            }
        }
    }

    Construct : op_construct_n / op_construct_w (callee => signed, base => signed, count => unsigned) {
        let v = cold!(construct_cold(ctx, reg!(callee), base, count));
        reenter!(pc, SIZE, v)
    }
);

#[inline(always)]
unsafe fn r8s(pc: usize, code: *const u8, off: usize) -> i32 {
    *code.add(pc + off) as i8 as i32
}

#[inline(always)]
unsafe fn r8u(pc: usize, code: *const u8, off: usize) -> usize {
    *code.add(pc + off) as usize
}

#[inline(always)]
unsafe fn r16s(pc: usize, code: *const u8, off: usize) -> i32 {
    i16::from_le_bytes([*code.add(pc + off), *code.add(pc + off + 1)]) as i32
}

#[inline(always)]
unsafe fn r16u(pc: usize, code: *const u8, off: usize) -> usize {
    u16::from_le_bytes([*code.add(pc + off), *code.add(pc + off + 1)]) as usize
}

mod signed {
    #[inline(always)]
    pub(super) unsafe fn read(pc: usize, code: *const u8, off: usize, stride: usize) -> i32 {
        unsafe {
            if stride == 1 {
                super::r8s(pc, code, off)
            } else {
                super::r16s(pc, code, off)
            }
        }
    }
}

mod unsigned {
    #[inline(always)]
    pub(super) unsafe fn read(pc: usize, code: *const u8, off: usize, stride: usize) -> usize {
        unsafe {
            if stride == 1 {
                super::r8u(pc, code, off)
            } else {
                super::r16u(pc, code, off)
            }
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn add_cold<'a>(
    ctx: &Ctx<'a>,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let lhs = scope.handle(lhs);
        let lhs = match Object::to_primitive(vm, heap, state, lhs, Hint::Default)? {
            Coercion::Threw => return Ok(ctx.exception_word()),
            Coercion::Value(v) => scope.handle(v),
        };
        let rhs = scope.handle(rhs);
        let rhs = match Object::to_primitive(vm, heap, state, rhs, Hint::Default)? {
            Coercion::Threw => return Ok(ctx.exception_word()),
            Coercion::Value(v) => scope.handle(v),
        };
        let is_string = (
            lhs.as_tagged(heap).get_as::<DenseString>().is_some(),
            rhs.as_tagged(heap).get_as::<DenseString>().is_some(),
        );
        if is_string.0 || is_string.1 {
            let a = scope.handle(Convert::to_string(heap, &scope, lhs)?);
            let b = scope.handle(Convert::to_string(heap, &scope, rhs)?);
            Ok(DenseString::concat(heap, &scope, a, b).as_tagged(heap).erase())
        } else {
            let a = Convert::to_number(heap, lhs.as_tagged(heap))?;
            let b = Convert::to_number(heap, rhs.as_tagged(heap))?;
            let r = a + b;
            let r = if r == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                -0.0
            } else {
                r
            };
            Ok(heap.new_number(r))
        }
    })
}

#[cold]
#[inline(never)]
unsafe fn numeric_cold<'a>(
    ctx: &Ctx<'a>,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
    f: fn(f64, f64) -> f64,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let a = scope.handle(lhs);
        let b = scope.handle(rhs);
        match Object::numeric_op(vm, heap, state, a, b, f)? {
            Some(v) => Ok(v),
            None => Ok(ctx.exception_word()),
        }
    })
}

#[cold]
#[inline(never)]
unsafe fn negate_cold<'a>(
    ctx: &Ctx<'a>,
    v: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let v = scope.handle(v);
        match Object::to_numeric(vm, heap, state, v)? {
            Some(n) => Ok(heap.new_number(-n)),
            None => Ok(ctx.exception_word()),
        }
    })
}

/// The loose-compare cold body: `Ok(Some(bool))`, `Ok(None)` = threw.
#[cold]
#[inline(never)]
unsafe fn compare_cold<'a>(
    ctx: &Ctx<'a>,
    cmp: u8,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
) -> Result<Option<bool>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Option<bool>, VmError> {
        let hint = match cmp {
            0 => Hint::Default,
            _ => Hint::Number,
        };
        let x = scope.handle(lhs);
        let x = match Object::to_primitive(vm, heap, state, x, hint)? {
            Coercion::Threw => return Ok(None),
            Coercion::Value(v) => scope.handle(v),
        };
        let y = scope.handle(rhs);
        let y = match Object::to_primitive(vm, heap, state, y, hint)? {
            Coercion::Threw => return Ok(None),
            Coercion::Value(v) => scope.handle(v),
        };
        let b = match cmp {
            0 => Compare::equal(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            1 => Compare::strict_equal(heap, x.as_tagged(heap), y.as_tagged(heap)),
            2 => Compare::less_than(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            3 => Compare::less_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            4 => Compare::greater_than(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            _ => {
                Compare::greater_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap))?
            }
        };
        Ok(Some(b))
    })
}

#[cold]
#[inline(never)]
unsafe fn named_load_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    fb_slot: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    let meta = ctx.meta(0);
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let recv = scope.handle(recv);
        let name = scope.handle(
            ctx.stack()
                .callable(heap, &meta)
                .as_ref()
                .constant_slot_name(heap, name_idx),
        );
        enum Res<'s> {
            Word(Handle<'s, Value>),
            Getter(Handle<'s, Value>),
        }
        let res = match Lookup::load_outcome(heap, recv.as_tagged(heap), name.as_tagged(heap))? {
            LoadOutcome::Value(v) => Res::Word(scope.handle(v)),
            LoadOutcome::Getter(g) => Res::Getter(scope.handle(g)),
        };
        match res {
            Res::Word(v) => {
                InlineCache::update_load(
                    heap,
                    &scope,
                    ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                    fb_slot,
                    recv.as_tagged(heap).as_heap_object().map(|o| scope.handle(o)),
                    scope.handle(name.as_tagged(heap).erase().as_name()),
                    true,
                );
                Ok(v.as_tagged(heap).erase())
            }
            Res::Getter(getter) => {
                let args = scope.stage(&[recv.as_tagged(heap).erase()]);
                RuntimeContext::call(vm, heap, state, getter, args, None)
            }
        }
    })
}

#[cold]
#[inline(never)]
unsafe fn keyed_load_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let recv = scope.handle(recv);
        let raw_key = scope.handle(key);
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            let staged = scope.stage(&[raw_key.as_tagged(heap).erase()]);
            return match Proxy::apply(vm, heap, state, recv.erase(), staged)? {
                Coercion::Threw => Ok(ctx.exception_word()),
                Coercion::Value(v) => Ok(v),
            };
        }
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(ctx.exception_word());
        };
        let key = scope.handle(key);
        match Lookup::load_outcome_keyed(heap, recv.as_tagged(heap), key.as_tagged(heap))? {
            LoadOutcome::Value(v) => Ok(v),
            LoadOutcome::Getter(getter) => {
                let getter = scope.handle(getter);
                let args = scope.stage(&[recv.as_tagged(heap).erase()]);
                RuntimeContext::call(vm, heap, state, getter, args, None)
            }
        }
    })
}

#[cold]
#[inline(never)]
unsafe fn keyed_load_imm_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    idx: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let recv = scope.handle(recv);
        let key = scope.handle(Smi::new(idx as i64).into_tagged());
        match Lookup::load_outcome_keyed(heap, recv.as_tagged(heap), key.as_tagged(heap).as_name())? {
            LoadOutcome::Value(v) => Ok(v),
            LoadOutcome::Getter(getter) => {
                let getter = scope.handle(getter);
                let args = scope.stage(&[recv.as_tagged(heap).erase()]);
                RuntimeContext::call(vm, heap, state, getter, args, None)
            }
        }
    })
}

#[cold]
#[inline(never)]
unsafe fn keyed_store_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let recv = scope.handle(recv);
        let raw_key = scope.handle(key);
        let value = scope.handle(value);
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            let name = scope.handle(raw_key.as_tagged(heap).erase());
            return match Proxy::set(
                vm,
                heap,
                state,
                recv,
                name,
                value,
                recv,
            )? {
                Coercion::Threw => Ok(ctx.exception_word()),
                Coercion::Value(_) => Ok(value.as_tagged(heap).erase()),
            };
        }
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(ctx.exception_word());
        };
        let key = scope.handle(key);
        let name: Handle<'_, vm_core::SlotName> = match Lookup::classify_key(
            heap,
            key.as_tagged(heap).erase(),
        )? {
            Key::Element(i) => {
                if recv
                    .as_tagged(heap)
                    .as_heap_object()
                    .is_some_and(|obj| obj.as_ref().is_array(heap))
                {
                    let recv = scope
                        .cast::<Object>(recv.as_tagged(heap))
                        .expect("array receiver is an object");
                    Object::store_array_element(heap, &scope, &recv, i, &value)?;
                    return Ok(value.as_tagged(heap).erase());
                }
                scope.handle(Tagged::<SlotName>::from(Smi::new(i as i64)))
            }
            Key::Name(key) => scope.handle(key),
        };
        let outcome = recv.as_tagged(heap).erase().store_lookup(
            heap,
            &scope,
            name.as_tagged(heap),
            value.as_tagged(heap),
            StoreSemantics::Shadow,
        )?;
        match outcome {
            StoreOutcome::Transition { receiver, name } => {
                Object::add_own_property(
                    heap,
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(scope.handle(value.as_tagged(heap))),
                )?;
            }
            StoreOutcome::CallSetter { setter } => {
                let args = scope.stage(&[recv.as_tagged(heap).erase(), value.as_tagged(heap)]);
                let _ = RuntimeContext::call(vm, heap, state, setter, args, None)?;
            }
            StoreOutcome::Done => {}
        }
        Ok(value.as_tagged(heap).erase())
    })
}

#[cold]
#[inline(never)]
unsafe fn global_load_cold<'a>(
    ctx: &Ctx<'a>,
    name_idx: usize,
    fb_slot: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let global = heap.known().global_object;
        let name = scope.handle(
            ctx.cache()
                .constants_ref(heap)
                .at(heap, name_idx)
                .erase()
                .as_name(),
        );
        match global.as_tagged(heap).lookup(heap, name.as_tagged(heap)) {
            Lookup::Data { slot, .. } => {
                let v = scope.handle(slot.get(heap));
                InlineCache::update_load(
                    heap,
                    &scope,
                    ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                    fb_slot,
                    Some(scope.handle(global.as_tagged(heap))),
                    scope.handle(name.as_tagged(heap).erase().as_name()),
                    false,
                );
                Ok(v.as_tagged(heap).erase())
            }
            Lookup::Accessor { pair, .. } => {
                let getter = scope.handle(pair.get.get(heap));
                let args = scope.stage(&[global.as_tagged(heap).erase()]);
                RuntimeContext::call(vm, heap, state, getter, args, None)
            }
            Lookup::NotFound => {
                let text = name
                    .as_tagged(heap)
                    .erase()
                    .get_as::<DenseString>()
                    .map(|s| s.to_rust_string(heap))
                    .unwrap_or_default();
                let ex = Errors::not_defined(vm, heap, state, &text)
                    .expect("error materialization must not fail");
                state.set_pending_exception(ex);
                Ok(ctx.exception_word())
            }
        }
    })
}

#[cold]
#[inline(never)]
unsafe fn store_named_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    fb_slot: usize,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    // the store IC reads the accumulator from the cache slot
    ctx.cache().acc_mut().store(value);
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let recv = scope.handle(recv);
        let value = scope.handle(value);
        let name = scope.handle(
            ctx.cache()
                .constants_ref(heap)
                .at(heap, name_idx)
                .erase()
                .as_name(),
        );

        if let Some(hit) = InlineCache::try_store(
            heap,
            &scope,
            ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
            fb_slot,
            recv.erase(),
            name,
            &ctx.cache().acc_mut(),
        ) {
            return match hit {
                StoreHit::Done => Ok(value.as_tagged(heap).erase()),
                StoreHit::Setter(setter) => {
                    let setter = scope.handle(setter);
                    let args =
                        scope.stage(&[recv.as_tagged(heap).erase(), value.as_tagged(heap)]);
                    let _ = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                    Ok(value.as_tagged(heap).erase())
                }
            };
        }

        let prev = recv
            .as_tagged(heap)
            .as_heap_object()
            .map(|o| scope.handle(o.as_ref().map_ref(heap)));
        let outcome = recv
            .as_tagged(heap)
            .erase()
            .store_lookup(heap, &scope, name.as_tagged(heap), value.as_tagged(heap), StoreSemantics::Shadow)?;
        let kind = match outcome {
            StoreOutcome::Done => StoreOutcomeKind::Done,
            StoreOutcome::Transition { .. } => StoreOutcomeKind::Transition,
            StoreOutcome::CallSetter { .. } => StoreOutcomeKind::CallSetter,
        };
        match outcome {
            StoreOutcome::Transition { receiver, name } => {
                Object::add_own_property(
                    heap,
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(scope.handle(value.as_tagged(heap))),
                )?;
                if let Some(prev) = prev {
                    InlineCache::update_store(
                        heap,
                        &scope,
                        ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                        fb_slot,
                        recv.erase(),
                        name,
                        prev,
                        kind,
                    );
                }
            }
            StoreOutcome::CallSetter { setter } => {
                let args =
                    scope.stage(&[recv.as_tagged(heap).erase(), value.as_tagged(heap)]);
                let _ = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                if let Some(prev) = prev {
                    InlineCache::update_store(
                        heap,
                        &scope,
                        ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                        fb_slot,
                        recv.erase(),
                        name,
                        prev,
                        kind,
                    );
                }
            }
            StoreOutcome::Done => {
                if let Some(prev) = prev {
                    InlineCache::update_store(
                        heap,
                        &scope,
                        ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                        fb_slot,
                        recv.erase(),
                        name,
                        prev,
                        kind,
                    );
                }
            }
        }
        Ok(value.as_tagged(heap).erase())
    })
}

#[cold]
#[inline(never)]
unsafe fn construct_cold<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args_base: i32,
    count: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let callee = scope.handle(callee);
        let Some(obj) = callee.as_tagged(heap).as_heap_object() else {
            return Err(VmError::Type);
        };
        if !obj.as_ref().header.map.get(heap).kind().is_constructor() {
            return Err(VmError::Type);
        }
        let meta = ctx.meta(0);
        let args = ctx.stack().args(&meta, args_base, count);
        if Proxy::is_proxy(heap, callee.as_tagged(heap)) {
            return match Proxy::construct(vm, heap, state, callee, args, callee)? {
                Coercion::Threw => Ok(ctx.exception_word()),
                Coercion::Value(v) => Ok(v),
            };
        }
        let callee = scope
            .cast::<Object>(callee.as_tagged(heap))
            .expect("constructible callee is an object");
        let receiver = match Object::create_construct_receiver_value(
            vm, heap, state, callee.erase(),
        ) {
            Ok(Some(r)) => scope.handle(r),
            Ok(None) => return Ok(ctx.exception_word()),
            Err(err) => return Err(err),
        };
        let mut staged: Vec<Tagged<'_, Value>> = Vec::with_capacity(count + 1);
        staged.push(receiver.as_tagged(heap).erase());
        staged.extend(args.iter().map(|h| h.as_tagged(heap)));
        let staged = scope.stage(&staged);
        let result = scope.handle(RuntimeContext::call(
            vm,
            heap,
            state,
            callee.erase(),
            staged,
            Some(callee.erase()),
        )?);
        if result.as_tagged(heap) == ctx.exception_word() {
            return Ok(ctx.exception_word());
        }
        let result = if Convert::is_primitive(heap, result.as_tagged(heap)) {
            receiver.as_tagged(heap).erase()
        } else {
            result.as_tagged(heap).erase()
        };
        Ok(result)
    })
}

#[cold]
#[inline(never)]
unsafe fn create_closure_cold<'a>(
    ctx: &Ctx<'a>,
    info_idx: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let meta = ctx.meta(0);
        let Some(info) = scope.cast::<CallableInfoObject>(
            ctx.cache().constants_ref(heap).at(heap, info_idx),
        ) else {
            return Err(VmError::Type);
        };
        let context = scope
            .cast::<Context>(ctx.stack().context_slot(&meta).get(heap))
            .expect("frame context slot holds a Context");
        let obj = Object::create_closure(heap, &scope, info, context)?;
        Ok(obj.erase())
    })
}

#[cold]
#[inline(never)]
unsafe fn create_function_context_cold<'a>(
    ctx: &Ctx<'a>,
    scope_idx: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let meta = ctx.meta(0);
        let outer = scope
            .cast::<Context>(ctx.stack().context_slot(&meta).get(heap))
            .expect("frame context slot holds a Context");
        let count = ctx
            .cache()
            .constants_ref(heap)
            .at(heap, scope_idx)
            .get_as::<ScopeInfo>()
            .map(|r| r.as_ref().names.get(heap).len())
            .ok_or(VmError::Type)?;
        let values =
            scope.stage(&vec![heap.known().the_hole.as_tagged(heap).erase(); count]);
        let scope_info = scope
            .cast::<ScopeInfo>(ctx.cache().constants_ref(heap).at(heap, scope_idx))
            .expect("constants slot holds a ScopeInfo");
        let slots = heap.allocate_handle::<FixedArray>(values, &scope);
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
unsafe fn proxy_apply_cold<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args_base: i32,
    count: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let callee = scope.handle(callee);
        let meta = ctx.meta(0);
        let staged = ctx.stack().args(&meta, args_base, count);
        match Proxy::apply(vm, heap, state, callee, staged)? {
            Coercion::Threw => Ok(ctx.exception_word()),
            Coercion::Value(v) => Ok(v),
        }
    })
}

const fn table_narrow() -> [Handler; 256] {
    let mut t = [op_trap as Handler; 256];
    t[Opcode::Load as usize] = op_load_n as Handler;
    t[Opcode::Move as usize] = op_move_n as Handler;
    t[Opcode::Store as usize] = op_store_n as Handler;
    t[Opcode::LoadSmi as usize] = op_load_smi_n as Handler;
    t[Opcode::LoadConstant as usize] = op_load_constant_n as Handler;
    t[Opcode::LoadZero as usize] = op_load_zero_n as Handler;
    t[Opcode::LoadUndefined as usize] = op_load_undefined_n as Handler;
    t[Opcode::LoadTrue as usize] = op_load_true_n as Handler;
    t[Opcode::LoadFalse as usize] = op_load_false_n as Handler;
    t[Opcode::Add as usize] = op_add_n as Handler;
    t[Opcode::Sub as usize] = op_sub_n as Handler;
    t[Opcode::Mul as usize] = op_mul_n as Handler;
    t[Opcode::Div as usize] = op_div_n as Handler;
    t[Opcode::Equal as usize] = op_equal_n as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right_n as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed_n as Handler;
    t[Opcode::Negate as usize] = op_negate_n as Handler;
    t[Opcode::CompareJump as usize] = op_compare_jump_n as Handler;
    t[Opcode::Jump as usize] = op_jump_n as Handler;
    t[Opcode::JumpIfTruthy as usize] = op_jump_if_truthy_n as Handler;
    t[Opcode::JumpIfFalsy as usize] = op_jump_if_falsy_n as Handler;
    t[Opcode::JumpLoop as usize] = op_jump_loop_n as Handler;
    t[Opcode::Throw as usize] = op_throw_n as Handler;
    t[Opcode::Return as usize] = op_return_n as Handler;
    t[Opcode::LoadNamedProperty as usize] = op_load_named_n as Handler;
    t[Opcode::LoadKeyedProperty as usize] = op_load_keyed_n as Handler;
    t[Opcode::LoadElementImm as usize] = op_load_element_imm_n as Handler;
    t[Opcode::LoadGlobal as usize] = op_load_global_n as Handler;
    t[Opcode::StoreNamedProperty as usize] = op_store_named_n as Handler;
    t[Opcode::LoadContextSlot as usize] = op_load_context_slot_n as Handler;
    t[Opcode::StoreContextSlot as usize] = op_store_context_slot_n as Handler;
    t[Opcode::PushContext as usize] = op_push_context_n as Handler;
    t[Opcode::PopContext as usize] = op_pop_context_n as Handler;
    t[Opcode::CreateFunctionContext as usize] = op_create_function_context_n as Handler;
    t[Opcode::CreateClosure as usize] = op_create_closure_n as Handler;
    t[Opcode::CallRuntime as usize] = op_call_runtime_n as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call_n as Handler;
    t[Opcode::Construct as usize] = op_construct_n as Handler;
    t[Opcode::Wide as usize] = op_wide as Handler;
    t
}

const fn table_wide() -> [Handler; 256] {
    let mut t = [op_trap as Handler; 256];
    t[Opcode::Load as usize] = op_load_w as Handler;
    t[Opcode::Move as usize] = op_move_w as Handler;
    t[Opcode::Store as usize] = op_store_w as Handler;
    t[Opcode::LoadSmi as usize] = op_load_smi_w as Handler;
    t[Opcode::LoadConstant as usize] = op_load_constant_w as Handler;
    t[Opcode::LoadZero as usize] = op_load_zero_w as Handler;
    t[Opcode::LoadUndefined as usize] = op_load_undefined_w as Handler;
    t[Opcode::LoadTrue as usize] = op_load_true_w as Handler;
    t[Opcode::LoadFalse as usize] = op_load_false_w as Handler;
    t[Opcode::Add as usize] = op_add_w as Handler;
    t[Opcode::Sub as usize] = op_sub_w as Handler;
    t[Opcode::Mul as usize] = op_mul_w as Handler;
    t[Opcode::Div as usize] = op_div_w as Handler;
    t[Opcode::Equal as usize] = op_equal_w as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right_w as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed_w as Handler;
    t[Opcode::Negate as usize] = op_negate_w as Handler;
    t[Opcode::CompareJump as usize] = op_compare_jump_w as Handler;
    t[Opcode::Jump as usize] = op_jump_w as Handler;
    t[Opcode::JumpIfTruthy as usize] = op_jump_if_truthy_w as Handler;
    t[Opcode::JumpIfFalsy as usize] = op_jump_if_falsy_w as Handler;
    t[Opcode::JumpLoop as usize] = op_jump_loop_w as Handler;
    t[Opcode::Throw as usize] = op_throw_w as Handler;
    t[Opcode::Return as usize] = op_return_w as Handler;
    t[Opcode::LoadNamedProperty as usize] = op_load_named_w as Handler;
    t[Opcode::LoadKeyedProperty as usize] = op_load_keyed_w as Handler;
    t[Opcode::LoadElementImm as usize] = op_load_element_imm_w as Handler;
    t[Opcode::LoadGlobal as usize] = op_load_global_w as Handler;
    t[Opcode::StoreNamedProperty as usize] = op_store_named_w as Handler;
    t[Opcode::LoadContextSlot as usize] = op_load_context_slot_w as Handler;
    t[Opcode::StoreContextSlot as usize] = op_store_context_slot_w as Handler;
    t[Opcode::PushContext as usize] = op_push_context_w as Handler;
    t[Opcode::PopContext as usize] = op_pop_context_w as Handler;
    t[Opcode::CreateFunctionContext as usize] = op_create_function_context_w as Handler;
    t[Opcode::CreateClosure as usize] = op_create_closure_w as Handler;
    t[Opcode::CallRuntime as usize] = op_call_runtime_w as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call_w as Handler;
    t[Opcode::Construct as usize] = op_construct_w as Handler;
    t
}

static TABLE_NARROW: HandlerTable = HandlerTable(table_narrow());
static TABLE_WIDE: HandlerTable = HandlerTable(table_wide());

// ---------------------------------------------------------------------------
// entry
// ---------------------------------------------------------------------------

fn enter<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'a, Value>>,
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
                Ok(Coercion::Value(v)) => Ok(scope.handle(v).as_tagged(heap).erase()),
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
            let stack = state.stack();
            let cache = state.cache();
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
            cache.enter(stack, frame, heap);

            let ctx: Ctx<'a> = Ctx {
                vm: vm as *const VM,
                heap: heap as *mut Heap,
                state: state as *const ContextState,
                base_depth,
                _heap: PhantomData,
            };
            let base = ctx.code_ptr();
            let pc = cache.pc();
            let regs = ctx.regs_ptr();
            let acc = cache.acc(ctx.heap());
            unsafe { TABLE_NARROW.0[*base.add(pc) as usize](pc, base, regs, acc, &ctx) }
        }
    }
}

pub fn execute<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'_, Value>>,
) -> Result<Tagged<'a, Value>, VmError> {
    let stack = state.stack();
    let cache = state.cache();

    let saved_top = stack.top();
    let was_active = cache.is_active();
    if was_active {
        stack.suspend_frame(cache.frame_meta());
    }
    let base_depth = stack.frame_depth();

    state.handle_scope(|scope| {
        let result = enter(vm, heap, state, callable, args, new_target, base_depth)?;
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

impl Interpreter for BecomeInterpreter {
    const EXECUTE: ExecuteFn = execute;
}
