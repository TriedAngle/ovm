#![feature(explicit_tail_calls)]
#![feature(rust_preserve_none_cc)]
#![feature(fn_align)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_macros, unused_unsafe, unused_variables)]

use core::cell::Cell;
use core::marker::PhantomData;

use bytecode::{OPERAND_SIZES_NARROW, OPERAND_SIZES_WIDE, Opcode, jump_target};
use vm_core::ic::{ElementHit, Hit, InlineCache, MonoProbe, StoreHit, StoreOutcomeKind};
use vm_core::proxy::Proxy;
use vm_core::{
    CallTarget, CallableInfoObject, Coercion, Compare, Context, ContextInit, ContextState, Convert,
    DenseString, Errors, ExecuteFn, FixedArray, FrameMeta, Handle, HandleSlice, Heap, Hint,
    Interpreter, Key, LoadOutcome, Lookup, Object, PropertyDescriptor, Register, RuntimeContext,
    RuntimeIndex, ScopeInfo, SlotName, Smi, Stack, StackCache, StoreOutcome, StoreSemantics,
    Tagged, VM, Value, VmError,
};

pub struct BecomeInterpreter;

const SAFEPOINT_INTERVAL: u32 = 1 << 12;

pub struct Ctx<'a> {
    vm: *const VM,
    heap: *mut Heap,
    state: *const ContextState,
    /// Anchor of the frame `execute` entered: a return at this anchor ends
    /// the execution instead of unwinding into a caller.
    base_anchor: usize,
    safepoints: Cell<u32>,
    /// numeric result parked between an arithmetic fast path and the
    /// boxing continuation (which must allocate, so it is out of line)
    num: Cell<f64>,
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
    fn set_num(&self, f: f64) {
        self.num.set(f);
    }

    #[inline(always)]
    fn num(&self) -> f64 {
        self.num.get()
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
        self.cache().code_ref(self.heap()).as_ref().as_ptr()
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

    /// One loop back-edge: `true` every `SAFEPOINT_INTERVAL` ticks or when
    /// a collection has been requested since the last reset.
    #[inline(always)]
    fn safepoint_tick(&self) -> bool {
        let n = self.safepoints.get();
        if n == 0 {
            self.safepoints.set(SAFEPOINT_INTERVAL);
            true
        } else {
            self.safepoints.set(n - 1);
            false
        }
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

pub type Handler =
    for<'a> unsafe extern "rust-preserve-none" fn(
        pc: usize,
        code: *const u8,
        regs: *mut Register,
        acc: Tagged<'a, Value>,
        ctx: &Ctx<'a>,
    ) -> Result<Tagged<'a, Value>, VmError>;

pub struct HandlerTable([Handler; 256]);

macro_rules! helpers {
    ($pc:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident, $op:ident, $stride:literal, $($arg:ident => $kind:ident),* $(,)?) => {
        const BASE: usize = if $stride == 2 { 2 } else { 1 };
        const STAR: bool = star_lookahead(Opcode::$op);
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
        macro_rules! next {
            ($a:expr) => {{
                let p = $pc + SIZE;
                if STAR && unsafe { *$code.add(p) } == Opcode::Store as u8 {
                    let r = unsafe { *$code.add(p + 1) } as i8 as i32;
                    unsafe { reg_write($regs, r, $a) };
                    dispatch!(p + 2, $a, $regs)
                } else {
                    dispatch!(p, $a, $regs)
                }
            }};
        }
        macro_rules! jump { ($off:expr, $a:expr) => { dispatch!(jump_target($pc, $off), $a, $regs) } }

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
    };
}

const fn star_lookahead(op: Opcode) -> bool {
    matches!(
        op,
        Opcode::Load
            | Opcode::LoadSmi
            | Opcode::LoadConstant
            | Opcode::LoadZero
            | Opcode::LoadUndefined
            | Opcode::LoadNull
            | Opcode::LoadTrue
            | Opcode::LoadFalse
            | Opcode::LoadHole
            | Opcode::LoadGlobal
            | Opcode::LoadNamedProperty
            | Opcode::LoadKeyedProperty
            | Opcode::LoadElementImm
            | Opcode::LoadNewTarget
            | Opcode::LoadContextSlot
            | Opcode::AddRight
            | Opcode::SubRight
            | Opcode::MulRight
            | Opcode::DivRight
            | Opcode::AddLeft
            | Opcode::SubLeft
            | Opcode::MulLeft
            | Opcode::DivLeft
            | Opcode::AddImmediate
            | Opcode::AddLoc
            | Opcode::SubLoc
    )
}

macro_rules! handlers {
    ($pc:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident; $( $op:ident : $n:ident / $w:ident ($($arg:ident => $kind:ident),*) $body:block )*) => {
        $(
            #[inline(never)]
            #[rustc_align(32)]
            unsafe extern "rust-preserve-none" fn $n<'a>(
                $pc: usize, $code: *const u8, $regs: *mut Register, $acc: Tagged<'a, Value>, $ctx: &Ctx<'a>,
            ) -> Result<Tagged<'a, Value>, VmError> {
                helpers!($pc, $code, $regs, $acc, $ctx, $op, 1, $($arg => $kind),*);
                $body
            }
            #[inline(never)]
            #[rustc_align(32)]
            unsafe extern "rust-preserve-none" fn $w<'a>(
                $pc: usize, $code: *const u8, $regs: *mut Register, $acc: Tagged<'a, Value>, $ctx: &Ctx<'a>,
            ) -> Result<Tagged<'a, Value>, VmError> {
                helpers!($pc, $code, $regs, $acc, $ctx, $op, 2, $($arg => $kind),*);
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

    LoadNull : op_load_null_n / op_load_null_w () {
        next!(ctx.heap().known().null.as_tagged(ctx.heap()).erase())
    }

    LoadTrue : op_load_true_n / op_load_true_w () {
        next!(ctx.heap().known().true_object.as_tagged(ctx.heap()).erase())
    }

    LoadFalse : op_load_false_n / op_load_false_w () {
        next!(ctx.heap().known().false_object.as_tagged(ctx.heap()).erase())
    }

    AddRight : op_add_right_n / op_add_right_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && let Some(sum) = a.checked_add(b)
        {
            next!(Tagged::from_smi_bits(sum))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(acc), Convert::as_number(other)) {
            let sum = a + b;
            if let Some(s) = Smi::from_f64(sum) {
                next!(s.into_tagged())
            }
            ctx.set_num(sum);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_add(pc, code, regs, acc, ctx)
    }

    IncLoc : op_inc_loc_n / op_inc_loc_w (r => signed) {
        let v = reg!(r);
        if let Some(bits) = v.smi_bits()
            && let Some(new) = bits.checked_add(2)
        {
            set_reg!(r, Tagged::from_smi_bits(new));
            next!(v)
        }
        become cold_inc_loc(pc, code, regs, acc, ctx)
    }

    DecLoc : op_dec_loc_n / op_dec_loc_w (r => signed) {
        let v = reg!(r);
        if let Some(bits) = v.smi_bits()
            && let Some(new) = bits.checked_sub(2)
        {
            set_reg!(r, Tagged::from_smi_bits(new));
            next!(v)
        }
        become cold_dec_loc(pc, code, regs, acc, ctx)
    }

    AddLoc : op_add_loc_n / op_add_loc_w (dst => signed, src => signed) {
        let lhs = reg!(dst);
        let rhs = reg!(src);
        if let (Some(a), Some(b)) = (lhs.smi_bits(), rhs.smi_bits())
            && let Some(sum) = a.checked_add(b)
        {
            let v = Tagged::from_smi_bits(sum);
            set_reg!(dst, v);
            next!(v)
        }
        if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(rhs)) {
            let sum = a + b;
            if let Some(s) = Smi::from_f64(sum) {
                let v = s.into_tagged();
                set_reg!(dst, v);
                next!(v)
            }
            ctx.set_num(sum);
            become cold_box_add_loc(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_add_loc(pc, code, regs, acc, ctx)
    }

    SubLoc : op_sub_loc_n / op_sub_loc_w (dst => signed, src => signed) {
        let lhs = reg!(dst);
        let rhs = reg!(src);
        if let (Some(a), Some(b)) = (lhs.smi_bits(), rhs.smi_bits())
            && let Some(diff) = a.checked_sub(b)
        {
            let v = Tagged::from_smi_bits(diff);
            set_reg!(dst, v);
            next!(v)
        }
        if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(rhs)) {
            let diff = a - b;
            if let Some(s) = Smi::from_f64(diff) {
                let v = s.into_tagged();
                set_reg!(dst, v);
                next!(v)
            }
            ctx.set_num(diff);
            become cold_box_sub_loc(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_sub_loc(pc, code, regs, acc, ctx)
    }

    LoadKeyedPropertyReg : op_load_keyed_reg_n / op_load_keyed_reg_w (recv => signed, key => signed, fb => unsigned) {
        let recv = reg!(recv);
        let key = reg!(key);
        if let Some(idx) = key.to_i64()
            && idx >= 0
        {
            if let Some(obj) = recv.as_heap_object()
                && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx as usize)
            {
                next!(v)
            }
            if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
                ctx.heap(),
                ctx.cache().feedback_ref(ctx.heap()),
                fb,
                recv,
                idx as usize,
            ) {
                next!(v)
            }
        }
        become cold_keyed_load_reg(pc, code, regs, acc, ctx)
    }

    SubRight : op_sub_right_n / op_sub_right_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && let Some(diff) = a.checked_sub(b)
        {
            next!(Tagged::from_smi_bits(diff))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(acc), Convert::as_number(other)) {
            let diff = a - b;
            if let Some(s) = Smi::from_f64(diff) {
                next!(s.into_tagged())
            }
            ctx.set_num(diff);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    MulRight : op_mul_right_n / op_mul_right_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits())
            && let Some(product) = (a >> 1).checked_mul(b)
        {
            next!(Tagged::from_smi_bits(product))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(acc), Convert::as_number(other)) {
            let product = a * b;
            if let Some(s) = Smi::from_f64(product) {
                next!(s.into_tagged())
            }
            ctx.set_num(product);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    DivRight : op_div_right_n / op_div_right_w (r => signed) {
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
        if let (Some(a), Some(b)) = (Convert::as_number(acc), Convert::as_number(other)) {
            let quotient = a / b;
            if let Some(s) = Smi::from_f64(quotient) {
                next!(s.into_tagged())
            }
            ctx.set_num(quotient);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    AddLeft : op_add_left_n / op_add_left_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (other.smi_bits(), acc.smi_bits())
            && let Some(sum) = a.checked_add(b)
        {
            next!(Tagged::from_smi_bits(sum))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(other), Convert::as_number(acc)) {
            let sum = a + b;
            if let Some(s) = Smi::from_f64(sum) {
                next!(s.into_tagged())
            }
            ctx.set_num(sum);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_add_left(pc, code, regs, acc, ctx)
    }

    SubLeft : op_sub_left_n / op_sub_left_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (other.smi_bits(), acc.smi_bits())
            && let Some(diff) = a.checked_sub(b)
        {
            next!(Tagged::from_smi_bits(diff))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(other), Convert::as_number(acc)) {
            let diff = a - b;
            if let Some(s) = Smi::from_f64(diff) {
                next!(s.into_tagged())
            }
            ctx.set_num(diff);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric_left(pc, code, regs, acc, ctx)
    }

    MulLeft : op_mul_left_n / op_mul_left_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (other.smi_bits(), acc.smi_bits())
            && let Some(product) = (a >> 1).checked_mul(b)
        {
            next!(Tagged::from_smi_bits(product))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(other), Convert::as_number(acc)) {
            let product = a * b;
            if let Some(s) = Smi::from_f64(product) {
                next!(s.into_tagged())
            }
            ctx.set_num(product);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric_left(pc, code, regs, acc, ctx)
    }

    DivLeft : op_div_left_n / op_div_left_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (other.smi_bits(), acc.smi_bits())
            && b != 0
            && a % b == 0
            && let Some(quotient) = a.checked_div(b)
            && let Some(encoded) = quotient.checked_mul(2)
        {
            next!(Tagged::from_smi_bits(encoded))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(other), Convert::as_number(acc)) {
            let quotient = a / b;
            if let Some(s) = Smi::from_f64(quotient) {
                next!(s.into_tagged())
            }
            ctx.set_num(quotient);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric_left(pc, code, regs, acc, ctx)
    }

    Equal : op_equal_n / op_equal_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
            next!(Convert::boolean(ctx.heap(), a == b))
        }
        become cold_equal(pc, code, regs, acc, ctx)
    }

    LessThan : op_less_than_n / op_less_than_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
            next!(Convert::boolean(ctx.heap(), a < b))
        }
        become cold_less_than(pc, code, regs, acc, ctx)
    }

    GreaterThan : op_greater_than_n / op_greater_than_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
            next!(Convert::boolean(ctx.heap(), a > b))
        }
        become cold_greater_than(pc, code, regs, acc, ctx)
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
        {
            if Object::store_array_element_in_place(ctx.heap(), recv_w, idx as usize, acc).is_ok()
            {
                next!(acc)
            }
            if let Some(v) = InlineCache::try_store_element(
                ctx.heap(),
                ctx.cache().feedback_ref(ctx.heap()),
                fb,
                recv_w,
                idx as usize,
                acc,
            ) {
                next!(v)
            }
        }
        become cold_keyed_store(pc, code, regs, acc, ctx)
    }

    StoreKeyedPropertyNoShadow : op_store_keyed_no_shadow_n / op_store_keyed_no_shadow_w (recv => signed, key => signed, fb => unsigned) {
        let recv_w = reg!(recv);
        let key_w = reg!(key);
        if let Some(idx) = key_w.to_i64()
            && idx >= 0
        {
            if Object::store_array_element_in_place(ctx.heap(), recv_w, idx as usize, acc).is_ok()
            {
                next!(acc)
            }
            if let Some(v) = InlineCache::try_store_element(
                ctx.heap(),
                ctx.cache().feedback_ref(ctx.heap()),
                fb,
                recv_w,
                idx as usize,
                acc,
            ) {
                next!(v)
            }
        }
        become cold_keyed_store_no_shadow(pc, code, regs, acc, ctx)
    }

    Negate : op_negate_n / op_negate_w () {
        if let Some(bits) = acc.smi_bits()
            && bits != 0
            && let Some(neg) = bits.checked_neg()
        {
            next!(Tagged::from_smi_bits(neg))
        }
        if let Some(a) = Convert::as_number(acc) {
            let neg = -a;
            if let Some(s) = Smi::from_f64(neg) {
                next!(s.into_tagged())
            }
            ctx.set_num(neg);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_negate(pc, code, regs, acc, ctx)
    }

    CompareJump : op_compare_jump_n / op_compare_jump_w (r => signed, kind => unsigned, off => signed) {
        let other = reg!(r);
        let cmp = (kind / 2) as u8;
        let falsy_jump = kind % 2 == 1;
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
            let b = match cmp {
                0 | 1 => a == b,
                2 => a < b,
                3 => a <= b,
                4 => a > b,
                _ => a >= b,
            };
            let boolean = Convert::boolean(ctx.heap(), b);
            if b != falsy_jump {
                jump!(off, boolean)
            } else {
                next!(boolean)
            }
        }
        become cold_compare_jump(pc, code, regs, acc, ctx)
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
        if ctx.safepoint_tick() {
            // the acc word must be GC-visible while parked: sync to the cache
            ctx.cache().acc_mut().store(acc);
            if ctx.heap_mut().safepoint_poll() {
                let state = ctx.state();
                state.set_termination(vm_core::Termination::Shutdown);
                let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap());
                state.set_pending_exception(undefined);
                threw!()
            }
            let target = jump_target(pc, off);
            let acc = ctx.cache().acc(ctx.heap());
            reenter!(target, 0, acc)
        } else {
            jump!(off, acc)
        }
    }

    Throw : op_throw_n / op_throw_w () {
        ctx.state().set_pending_exception(acc);
        threw!()
    }

    Return : op_return_n / op_return_w () {
        if ctx.cache().base() == ctx.base_anchor {
            return Ok(acc);
        }
        let meta = ctx.meta(pc);
        let caller = ctx.stack().pop_frame(&meta);
        ctx.cache().load(ctx.stack(), caller, ctx.heap_mut());
        let rel = ctx.cache().pc();
        let base = ctx.code_ptr();
        dispatch!(rel, acc, ctx.regs_ptr(), base)
    }

    LoadNamedProperty : op_load_named_n / op_load_named_w (r => signed, name => unsigned, fb => unsigned) {
        let recv = reg!(r);
        match InlineCache::probe_mono(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv,
        ) {
            MonoProbe::Value(v) => next!(v),
            MonoProbe::Handler(obj, handler) => {
                if let Some(Hit::Value(v)) = InlineCache::apply_mono(ctx.heap(), obj, handler) {
                    next!(v)
                }
            }
            MonoProbe::Poly {
                obj,
                map,
                pairs,
                start,
            } => {
                if let Some(Hit::Value(v)) =
                    InlineCache::try_load_resume(ctx.heap(), obj, map, pairs, start)
                {
                    next!(v)
                }
            }
            MonoProbe::Miss | MonoProbe::NotReceiver => {}
        }
        become cold_named_load(pc, code, regs, acc, ctx)
    }

    LoadKeyedProperty : op_load_keyed_n / op_load_keyed_w (r => signed, fb => unsigned) {
        let recv = reg!(r);
        if let Some(idx) = acc.to_i64()
            && idx >= 0
        {
            if let Some(obj) = recv.as_heap_object()
                && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx as usize)
            {
                next!(v)
            }
            if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
                ctx.heap(),
                ctx.cache().feedback_ref(ctx.heap()),
                fb,
                recv,
                idx as usize,
            ) {
                next!(v)
            }
        }
        become cold_keyed_load(pc, code, regs, acc, ctx)
    }

    LoadElementImm : op_load_element_imm_n / op_load_element_imm_w (r => signed, idx => unsigned) {
        let recv = reg!(r);
        if let Some(obj) = recv.as_heap_object()
            && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx)
        {
            next!(v)
        }
        become cold_keyed_load_imm(pc, code, regs, acc, ctx)
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
        become cold_global_load(pc, code, regs, acc, ctx)
    }

    StoreNamedProperty : op_store_named_n / op_store_named_w (r => signed, name => unsigned, fb => unsigned) {
        let recv = reg!(r);
        if InlineCache::try_store_fast(
            ctx.heap_mut(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv,
            acc,
        ) {
            next!(acc)
        }
        become cold_store_named(pc, code, regs, acc, ctx)
    }

    LoadContextSlot : op_load_context_slot_n / op_load_context_slot_w (slot => unsigned, depth => unsigned) {
        let meta = ctx.meta(0);
        if depth == 0 {
            let heap = ctx.heap();
            let context_word = ctx.stack().context_slot(&meta).get(heap);
            let context = unsafe { context_word.cast::<Context>() };
            let slots = context.as_ref().slots.get(heap);
            let slots = unsafe { slots.cast::<FixedArray>() };
            let v = slots.as_ref().element_slot(slot).get(heap);
            next!(v)
        }
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
        become cold_create_function_context(pc, code, regs, acc, ctx)
    }

    CreateClosure : op_create_closure_n / op_create_closure_w (info => unsigned) {
        become cold_create_closure(pc, code, regs, acc, ctx)
    }

    CreateEmptyArrayLiteral : op_create_empty_array_n / op_create_empty_array_w () {
        become cold_create_empty_array(pc, code, regs, acc, ctx)
    }

    CreateEmptyObjectLiteral : op_create_empty_object_n / op_create_empty_object_w () {
        become cold_create_empty_object(pc, code, regs, acc, ctx)
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
        match Object::call_target(ctx.heap(), callee_word) {
            None => bail!(VmError::Type),
            Some(CallTarget::Proxy(_)) => become cold_proxy_apply(pc, code, regs, acc, ctx),
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
                    bail!(VmError::Type);
                }
                let heap = ctx.heap_mut();
                let meta = ctx.meta(pc + SIZE);
                let undefined = heap.known().undefined.as_tagged(heap).erase();
                let frame = ctx.stack().push_frame(
                    heap,
                    meta,
                    pc,
                    target.erase(),
                    info,
                    register_count,
                    context.erase(),
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

    CallMethod0 : op_call_method0_n / op_call_method0_w (callee => signed, recv => signed) {
        let callee_word = reg!(callee);
        match call_method_start(ctx, pc, SIZE, callee_word, &[recv])? {
            MethodCall::Value(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            MethodCall::Frame => {
                let p = ctx.cache().pc();
                let b = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let u = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(p, u, r, b)
            }
            MethodCall::Proxy => become cold_call_method_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallMethod1 : op_call_method1_n / op_call_method1_w (callee => signed, recv => signed, arg0 => signed) {
        let callee_word = reg!(callee);
        match call_method_start(ctx, pc, SIZE, callee_word, &[recv, arg0])? {
            MethodCall::Value(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            MethodCall::Frame => {
                let p = ctx.cache().pc();
                let b = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let u = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(p, u, r, b)
            }
            MethodCall::Proxy => become cold_call_method_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallMethod2 : op_call_method2_n / op_call_method2_w (callee => signed, recv => signed, arg0 => signed, arg1 => signed) {
        let callee_word = reg!(callee);
        match call_method_start(ctx, pc, SIZE, callee_word, &[recv, arg0, arg1])? {
            MethodCall::Value(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            MethodCall::Frame => {
                let p = ctx.cache().pc();
                let b = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let u = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(p, u, r, b)
            }
            MethodCall::Proxy => become cold_call_method_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallFunction0 : op_call_function0_n / op_call_function0_w (callee => signed) {
        let callee_word = reg!(callee);
        match call_function_start(ctx, pc, SIZE, callee_word, &[])? {
            MethodCall::Value(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            MethodCall::Frame => {
                let p = ctx.cache().pc();
                let b = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let u = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(p, u, r, b)
            }
            MethodCall::Proxy => become cold_call_function_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallFunction1 : op_call_function1_n / op_call_function1_w (callee => signed, arg0 => signed) {
        let callee_word = reg!(callee);
        match call_function_start(ctx, pc, SIZE, callee_word, &[arg0])? {
            MethodCall::Value(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            MethodCall::Frame => {
                let p = ctx.cache().pc();
                let b = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let u = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(p, u, r, b)
            }
            MethodCall::Proxy => become cold_call_function_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallFunction2 : op_call_function2_n / op_call_function2_w (callee => signed, arg0 => signed, arg1 => signed) {
        let callee_word = reg!(callee);
        match call_function_start(ctx, pc, SIZE, callee_word, &[arg0, arg1])? {
            MethodCall::Value(v) => {
                if ctx.is_throw(v) {
                    return Ok(v);
                }
                reenter!(pc, SIZE, v)
            }
            MethodCall::Frame => {
                let p = ctx.cache().pc();
                let b = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let u = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(p, u, r, b)
            }
            MethodCall::Proxy => become cold_call_function_proxy(pc, code, regs, acc, ctx),
        }
    }

    Construct : op_construct_n / op_construct_w (callee => signed, base => signed, count => unsigned) {
        become cold_construct(pc, code, regs, acc, ctx)
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
            Ok(DenseString::concat(heap, &scope, a, b)
                .as_tagged(heap)
                .erase())
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
unsafe extern "rust-preserve-none" fn cold_box_number<'a>(
    pc: usize,
    _code: *const u8,
    _regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = pc + packed_size(acc);
    let v = ctx.heap_mut().new_float(ctx.num());
    // the allocation may have moved the code object: re-derive both
    // pointers from the cache/stack before dispatching
    let code = ctx.code_ptr();
    let regs = ctx.regs_ptr();
    let h = TABLE_NARROW.0[*code.add(next) as usize];
    become h(next, code, regs, v, ctx)
}

/// The instruction size a boxing continuation was entered with: the
/// handlers pack `SIZE` into the (otherwise dead) accumulator argument.
#[inline(always)]
fn packed_size(acc: Tagged<'_, Value>) -> usize {
    debug_assert!(acc.to_i64().is_some(), "boxing continuation lost its size");
    acc.to_i64().unwrap_or(0) as usize
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_box_add_loc<'a>(
    pc: usize,
    code: *const u8,
    _regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = pc + packed_size(acc);
    let dst = signed::read(pc, code, base, stride);
    let v = ctx.heap_mut().new_float(ctx.num());
    let regs = ctx.regs_ptr();
    reg_write(regs, dst, v);
    let code = ctx.code_ptr();
    let h = TABLE_NARROW.0[*code.add(next) as usize];
    become h(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_box_sub_loc<'a>(
    pc: usize,
    code: *const u8,
    _regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = pc + packed_size(acc);
    let dst = signed::read(pc, code, base, stride);
    let v = ctx.heap_mut().new_float(ctx.num());
    let regs = ctx.regs_ptr();
    reg_write(regs, dst, v);
    let code = ctx.code_ptr();
    let h = TABLE_NARROW.0[*code.add(next) as usize];
    become h(next, code, regs, v, ctx)
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
        let x = scope.handle(lhs);
        let y = scope.handle(rhs);
        match cmp {
            1 => {
                // IsStrictEqual (ES 7.2.15): no coercion, ever
                let b = Compare::strict_equal(heap, x.as_tagged(heap), y.as_tagged(heap));
                return Ok(Some(b));
            }
            0 => {
                // IsLooselyEqual: object ↔ object compares by identity;
                // object ↔ primitive coerces the object
                let x_obj = Compare::is_object_operand(heap, x.as_tagged(heap));
                let y_obj = Compare::is_object_operand(heap, y.as_tagged(heap));
                if x_obj && y_obj {
                    let b = Compare::strict_equal(heap, x.as_tagged(heap), y.as_tagged(heap));
                    return Ok(Some(b));
                }
                let x = if x_obj {
                    match Object::to_primitive(vm, heap, state, x, Hint::Default)? {
                        Coercion::Threw => return Ok(None),
                        Coercion::Value(v) => scope.handle(v),
                    }
                } else {
                    x
                };
                let y = if y_obj {
                    match Object::to_primitive(vm, heap, state, y, Hint::Default)? {
                        Coercion::Threw => return Ok(None),
                        Coercion::Value(v) => scope.handle(v),
                    }
                } else {
                    y
                };
                let b = Compare::equal(heap, x.as_tagged(heap), y.as_tagged(heap))?;
                return Ok(Some(b));
            }
            _ => {}
        }
        let x = match Object::to_primitive(vm, heap, state, x, Hint::Number)? {
            Coercion::Threw => return Ok(None),
            Coercion::Value(v) => scope.handle(v),
        };
        let y = match Object::to_primitive(vm, heap, state, y, Hint::Number)? {
            Coercion::Threw => return Ok(None),
            Coercion::Value(v) => scope.handle(v),
        };
        let b = match cmp {
            2 => Compare::less_than(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            3 => Compare::less_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            4 => Compare::greater_than(heap, x.as_tagged(heap), y.as_tagged(heap))?,
            _ => Compare::greater_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap))?,
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
                    recv.as_tagged(heap)
                        .as_heap_object()
                        .map(|o| scope.handle(o)),
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
    fb_slot: Option<usize>,
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
        let element_index = match Lookup::classify_key(heap, key.as_tagged(heap).erase()) {
            Ok(Key::Element(i)) => Some(i),
            _ => None,
        };
        if let Some(i) = element_index {
            if let Some(fb) = fb_slot {
                InlineCache::update_load_element(
                    heap,
                    &scope,
                    ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                    fb,
                    Some(recv),
                    i,
                );
            }
            if let Some(unit) = DenseString::index_element(heap, &scope, recv, key) {
                return Ok(unit);
            }
        }
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
        let smi = Smi::new(idx as i64).into_tagged();
        let key: Handle<'_, SlotName> = scope.handle(smi.as_name());
        if let Some(unit) = DenseString::index_element(heap, &scope, recv, key) {
            return Ok(unit);
        }
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
unsafe fn keyed_store_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
    value: Tagged<'_, Value>,
    fb_slot: Option<usize>,
    semantics: StoreSemantics,
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
            return match Proxy::set(vm, heap, state, recv, name, value, recv)? {
                Coercion::Threw => Ok(ctx.exception_word()),
                Coercion::Value(_) => Ok(value.as_tagged(heap).erase()),
            };
        }
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(ctx.exception_word());
        };
        let key = scope.handle(key);
        let name: Handle<'_, vm_core::SlotName> =
            match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => {
                    if recv
                        .as_tagged(heap)
                        .as_heap_object()
                        .is_some_and(|obj| obj.as_ref().is_array(heap))
                    {
                        let obj = scope
                            .cast::<Object>(recv.as_tagged(heap))
                            .expect("array receiver is an object");
                        let grew = i >= obj.as_tagged(heap).as_ref().length();
                        Object::store_array_element(heap, &scope, &obj, i, &value)?;
                        if let Some(fb) = fb_slot {
                            InlineCache::update_store_element(
                                heap,
                                &scope,
                                ctx.cache().feedback_ref(heap).map(|v| scope.handle(v)),
                                fb,
                                Some(recv),
                                grew,
                            );
                        }
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
            semantics,
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
                    let args = scope.stage(&[recv.as_tagged(heap).erase(), value.as_tagged(heap)]);
                    let _ = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                    Ok(value.as_tagged(heap).erase())
                }
            };
        }

        let prev = recv
            .as_tagged(heap)
            .as_heap_object()
            .map(|o| scope.handle(o.as_ref().map_ref(heap)));
        let outcome = recv.as_tagged(heap).erase().store_lookup(
            heap,
            &scope,
            name.as_tagged(heap),
            value.as_tagged(heap),
            StoreSemantics::Shadow,
        )?;
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
                let args = scope.stage(&[recv.as_tagged(heap).erase(), value.as_tagged(heap)]);
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
        let receiver =
            match Object::create_construct_receiver_value(vm, heap, state, callee.erase()) {
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
        let Some(info) =
            scope.cast::<CallableInfoObject>(ctx.cache().constants_ref(heap).at(heap, info_idx))
        else {
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
        let scope_info = scope
            .cast::<ScopeInfo>(ctx.cache().constants_ref(heap).at(heap, scope_idx))
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


#[cold]
#[inline(never)]
unsafe fn proxy_apply_regs_cold<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    srcs: &[i32],
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let meta = ctx.meta(0);
    let heap = ctx.heap_mut();
    let (saved_top, staged) = ctx.stack().stage_args_regs(heap, &meta, srcs)?;
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

#[inline(always)]
unsafe fn cold_layout(pc: usize, code: *const u8) -> (usize, usize) {
    if *code.add(pc) == Opcode::Wide as u8 {
        (2, 2)
    } else {
        (1, 1)
    }
}

#[inline(always)]
unsafe fn cold_next_pc(pc: usize, code: *const u8) -> usize {
    let wide = *code.add(pc) == Opcode::Wide as u8;
    let op = *code.add(pc + wide as usize) as usize;
    let size = if wide {
        *OPERAND_SIZES_WIDE.get_unchecked(op)
    } else {
        *OPERAND_SIZES_NARROW.get_unchecked(op)
    } as usize;
    pc + wide as usize + 1 + size
}


#[inline(always)]
unsafe extern "rust-preserve-none" fn resume<'a>(
    pc: usize,
    _code: *const u8,
    _regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    if ctx.is_throw(acc) {
        return Ok(acc);
    }
    let code = ctx.code_ptr();
    let regs = ctx.regs_ptr();
    let op = *code.add(pc) as usize;
    let h = TABLE_NARROW.0[op];
    become h(pc, code, regs, acc, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_add<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = add_cold(ctx, acc, other)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_add_left<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = add_cold(ctx, other, acc)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_numeric_left<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let op = unsafe { Opcode::from_byte_unchecked(*code.add(pc + (stride == 2) as usize)) };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::SubLeft => |a, b| a - b,
        Opcode::MulLeft => |a, b| a * b,
        _ => |a, b| a / b,
    };
    let v = numeric_cold(ctx, other, acc, f)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_numeric<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let op = unsafe { Opcode::from_byte_unchecked(*code.add(pc + (stride == 2) as usize)) };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::SubRight => |a, b| a - b,
        Opcode::MulRight => |a, b| a * b,
        _ => |a, b| a / b,
    };
    let v = numeric_cold(ctx, acc, other, f)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_negate<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let v = negate_cold(ctx, acc)?;
    become resume(next, code, regs, v, ctx)
}


#[cold]
#[inline(never)]
unsafe fn incdec_cold<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    ctx: &Ctx<'a>,
    delta: f64,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let r = signed::read(pc, code, base, stride);
    let old = reg_read(regs, ctx.heap(), r);
    let heap = ctx.heap_mut();
    if let Some(bits) = old.smi_bits() {
        let new = heap.new_number((bits >> 1) as f64 + delta);
        reg_write(regs, r, new);
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
        reg_write(regs, r, new);
        Ok(old_num.as_tagged(heap))
    })
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_inc_loc<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let v = incdec_cold(pc, code, regs, ctx, 1.0)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_dec_loc<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let v = incdec_cold(pc, code, regs, ctx, -1.0)?;
    become resume(next, code, regs, v, ctx)
}

unsafe fn loc_op_cold<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    ctx: &Ctx<'a>,
    sub: bool,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let dst = signed::read(pc, code, base, stride);
    let src = signed::read(pc, code, base + stride, stride);
    let lhs = reg_read(regs, ctx.heap(), dst);
    let rhs = reg_read(regs, ctx.heap(), src);
    let v = if sub {
        numeric_cold(ctx, lhs, rhs, |a, b| a - b)?
    } else {
        add_cold(ctx, lhs, rhs)?
    };
    if !ctx.is_throw(v) {
        reg_write(regs, dst, v);
    }
    Ok(v)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_add_loc<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let v = loc_op_cold(pc, code, regs, ctx, false)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_sub_loc<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let v = loc_op_cold(pc, code, regs, ctx, true)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_load_reg<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let key = reg_read(
        regs,
        ctx.heap(),
        signed::read(pc, code, base + stride, stride),
    );
    let fb = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = keyed_load_cold(ctx, recv, key, Some(fb))?;
    become resume(next, code, regs, v, ctx)
}


#[cold]
#[inline(never)]
unsafe fn compare_op_cold<'a>(
    ctx: &Ctx<'a>,
    cmp: u8,
    acc: Tagged<'_, Value>,
    other: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let b = match (Convert::as_number(acc), Convert::as_number(other)) {
        (Some(a), Some(b)) => Some(match cmp {
            0 => a == b,
            2 => a < b,
            3 => a <= b,
            4 => a > b,
            _ => a >= b,
        }),
        _ => compare_cold(ctx, cmp, acc, other)?,
    };
    let Some(b) = b else {
        return Ok(ctx.exception_word());
    };
    Ok(Convert::boolean(ctx.heap(), b))
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_equal<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = compare_op_cold(ctx, 0, acc, other)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_less_than<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = compare_op_cold(ctx, 2, acc, other)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_greater_than<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = compare_op_cold(ctx, 4, acc, other)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_compare_jump<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let r = signed::read(pc, code, base, stride);
    let kind = unsigned::read(pc, code, base + stride, stride);
    let off = signed::read(pc, code, base + 2 * stride, stride);
    let other = reg_read(regs, ctx.heap(), r);
    let cmp = (kind / 2) as u8;
    let b = match (Convert::as_number(acc), Convert::as_number(other)) {
        (Some(a), Some(b)) => Some(match cmp {
            0 | 1 => a == b,
            2 => a < b,
            3 => a <= b,
            4 => a > b,
            _ => a >= b,
        }),
        _ => compare_cold(ctx, cmp, acc, other)?,
    };
    let Some(b) = b else {
        return Ok(ctx.exception_word());
    };
    let boolean = Convert::boolean(ctx.heap(), b);
    let dest = if b != (kind % 2 == 1) {
        jump_target(pc, off)
    } else {
        next
    };
    become resume(dest, code, regs, boolean, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_named_load<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let name_idx = unsigned::read(pc, code, base + stride, stride);
    let fb_slot = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = named_load_cold(ctx, recv, name_idx, fb_slot)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_load<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let fb = unsigned::read(pc, code, base + stride, stride);
    let v = keyed_load_cold(ctx, recv, acc, Some(fb))?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_load_imm<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let idx = unsigned::read(pc, code, base + stride, stride);
    let v = keyed_load_imm_cold(ctx, recv, idx)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_store<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let key = reg_read(
        regs,
        ctx.heap(),
        signed::read(pc, code, base + stride, stride),
    );
    let fb = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = keyed_store_cold(ctx, recv, key, acc, Some(fb), StoreSemantics::Shadow)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_store_no_shadow<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let key = reg_read(
        regs,
        ctx.heap(),
        signed::read(pc, code, base + stride, stride),
    );
    let fb = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = keyed_store_cold(ctx, recv, key, acc, Some(fb), StoreSemantics::WriteThrough)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_global_load<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let name_idx = unsigned::read(pc, code, base, stride);
    let fb_slot = unsigned::read(pc, code, base + stride, stride);
    let v = global_load_cold(ctx, name_idx, fb_slot)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_store_named<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let name_idx = unsigned::read(pc, code, base + stride, stride);
    let fb_slot = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = store_named_cold(ctx, recv, name_idx, fb_slot, acc)?;
    become resume(next, code, regs, v, ctx)
}

enum MethodCall<'a> {
    /// the call completed (runtime callee): the value or the exception
    /// sentinel
    Value(Tagged<'a, Value>),
    /// a bytecode frame was pushed and the cache switched to it
    Frame,
    /// a callable proxy: the caller tail-calls the cold trap dispatch
    Proxy,
}

#[inline(always)]
unsafe fn call_method_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'_, Value>,
    srcs: &[i32],
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(ctx.raise(VmError::Type)?)),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            let f = ctx.vm().runtime(RuntimeIndex(idx));
            let meta = ctx.meta(0);
            let (saved_top, args) = ctx.stack().stage_args_regs(ctx.heap(), &meta, srcs)?;
            let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
            let result = f(nctx, args);
            ctx.stack().set_top(saved_top);
            match result {
                Ok(v) => Ok(MethodCall::Value(v)),
                Err(err) => Err(err),
            }
        }
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
                return Ok(MethodCall::Value(ctx.raise(VmError::Type)?));
            }
            let heap = ctx.heap_mut();
            let meta = ctx.meta(pc + size);
            let undefined = heap.known().undefined.as_tagged(heap).erase();
            let frame = ctx.stack().push_frame_scattered(
                heap,
                meta,
                pc,
                target.erase(),
                info,
                register_count,
                context.erase(),
                srcs,
                undefined,
                formal_min,
            )?;
            ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
            Ok(MethodCall::Frame)
        }
    }
}

#[inline(always)]
unsafe fn call_function_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'_, Value>,
    args: &[i32],
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(ctx.raise(VmError::Type)?)),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            let f = ctx.vm().runtime(RuntimeIndex(idx));
            let meta = ctx.meta(0);
            let (saved_top, staged) = ctx.stack().stage_function_args(ctx.heap(), &meta, args)?;
            let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
            let result = f(nctx, staged);
            ctx.stack().set_top(saved_top);
            match result {
                Ok(v) => Ok(MethodCall::Value(v)),
                Err(err) => Err(err),
            }
        }
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
                return Ok(MethodCall::Value(ctx.raise(VmError::Type)?));
            }
            let heap = ctx.heap_mut();
            let meta = ctx.meta(pc + size);
            let undefined = heap.known().undefined.as_tagged(heap).erase();
            let frame = ctx.stack().push_frame_function(
                heap,
                meta,
                pc,
                target.erase(),
                info,
                register_count,
                context.erase(),
                args,
                undefined,
                formal_min,
            )?;
            ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
            Ok(MethodCall::Frame)
        }
    }
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_construct<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let callee = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let args_base = signed::read(pc, code, base + stride, stride);
    let count = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = construct_cold(ctx, callee, args_base, count)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_closure<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let info_idx = unsigned::read(pc, code, base, stride);
    let v = create_closure_cold(ctx, info_idx)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_empty_array<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let heap = ctx.heap_mut();
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().js_array_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(next, code, regs, obj, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_empty_object<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let next = cold_next_pc(pc, code);
    let heap = ctx.heap_mut();
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().object_initial_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(next, code, regs, obj, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_function_context<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let scope_idx = unsigned::read(pc, code, base, stride);
    let v = create_function_context_cold(ctx, scope_idx)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_proxy_apply<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let callee = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let args_base = signed::read(pc, code, base + stride, stride);
    let count = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = proxy_apply_cold(ctx, callee, args_base, count)?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_call_method_proxy<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let wide = *code.add(pc) == Opcode::Wide as u8;
    let argc = match Opcode::from_byte_unchecked(*code.add(pc + wide as usize)) {
        Opcode::CallMethod0 => 0usize,
        Opcode::CallMethod1 => 1,
        _ => 2,
    };
    let callee = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let mut srcs = [0i32; 3];
    for (i, src) in srcs.iter_mut().enumerate().take(argc + 1) {
        *src = signed::read(pc, code, base + (i + 1) * stride, stride);
    }
    let v = proxy_apply_regs_cold(ctx, callee, &srcs[..argc + 1])?;
    become resume(next, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe fn proxy_apply_function_cold<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args: &[i32],
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let meta = ctx.meta(0);
    let heap = ctx.heap_mut();
    let (saved_top, staged) = ctx.stack().stage_function_args(heap, &meta, args)?;
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
unsafe extern "rust-preserve-none" fn cold_call_function_proxy<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let wide = *code.add(pc) == Opcode::Wide as u8;
    let argc = match Opcode::from_byte_unchecked(*code.add(pc + wide as usize)) {
        Opcode::CallFunction0 => 0usize,
        Opcode::CallFunction1 => 1,
        _ => 2,
    };
    let callee = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let mut args = [0i32; 2];
    for (i, arg) in args.iter_mut().enumerate().take(argc) {
        *arg = signed::read(pc, code, base + (i + 1) * stride, stride);
    }
    let v = proxy_apply_function_cold(ctx, callee, &args[..argc])?;
    become resume(next, code, regs, v, ctx)
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
    t[Opcode::LoadNull as usize] = op_load_null_n as Handler;
    t[Opcode::LoadTrue as usize] = op_load_true_n as Handler;
    t[Opcode::LoadFalse as usize] = op_load_false_n as Handler;
    t[Opcode::AddRight as usize] = op_add_right_n as Handler;
    t[Opcode::IncLoc as usize] = op_inc_loc_n as Handler;
    t[Opcode::DecLoc as usize] = op_dec_loc_n as Handler;
    t[Opcode::AddLoc as usize] = op_add_loc_n as Handler;
    t[Opcode::SubLoc as usize] = op_sub_loc_n as Handler;
    t[Opcode::LoadKeyedPropertyReg as usize] = op_load_keyed_reg_n as Handler;
    t[Opcode::SubRight as usize] = op_sub_right_n as Handler;
    t[Opcode::MulRight as usize] = op_mul_right_n as Handler;
    t[Opcode::DivRight as usize] = op_div_right_n as Handler;
    t[Opcode::AddLeft as usize] = op_add_left_n as Handler;
    t[Opcode::SubLeft as usize] = op_sub_left_n as Handler;
    t[Opcode::MulLeft as usize] = op_mul_left_n as Handler;
    t[Opcode::DivLeft as usize] = op_div_left_n as Handler;
    t[Opcode::Equal as usize] = op_equal_n as Handler;
    t[Opcode::LessThan as usize] = op_less_than_n as Handler;
    t[Opcode::GreaterThan as usize] = op_greater_than_n as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right_n as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed_n as Handler;
    t[Opcode::StoreKeyedPropertyNoShadow as usize] = op_store_keyed_no_shadow_n as Handler;
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
    t[Opcode::CreateEmptyArrayLiteral as usize] = op_create_empty_array_n as Handler;
    t[Opcode::CreateEmptyObjectLiteral as usize] = op_create_empty_object_n as Handler;
    t[Opcode::CallRuntime as usize] = op_call_runtime_n as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call_n as Handler;
    t[Opcode::CallMethod0 as usize] = op_call_method0_n as Handler;
    t[Opcode::CallFunction0 as usize] = op_call_function0_n as Handler;
    t[Opcode::CallFunction1 as usize] = op_call_function1_n as Handler;
    t[Opcode::CallFunction2 as usize] = op_call_function2_n as Handler;
    t[Opcode::CallMethod1 as usize] = op_call_method1_n as Handler;
    t[Opcode::CallMethod2 as usize] = op_call_method2_n as Handler;
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
    t[Opcode::LoadNull as usize] = op_load_null_w as Handler;
    t[Opcode::LoadTrue as usize] = op_load_true_w as Handler;
    t[Opcode::LoadFalse as usize] = op_load_false_w as Handler;
    t[Opcode::AddRight as usize] = op_add_right_w as Handler;
    t[Opcode::IncLoc as usize] = op_inc_loc_w as Handler;
    t[Opcode::DecLoc as usize] = op_dec_loc_w as Handler;
    t[Opcode::AddLoc as usize] = op_add_loc_w as Handler;
    t[Opcode::SubLoc as usize] = op_sub_loc_w as Handler;
    t[Opcode::LoadKeyedPropertyReg as usize] = op_load_keyed_reg_w as Handler;
    t[Opcode::SubRight as usize] = op_sub_right_w as Handler;
    t[Opcode::MulRight as usize] = op_mul_right_w as Handler;
    t[Opcode::DivRight as usize] = op_div_right_w as Handler;
    t[Opcode::AddLeft as usize] = op_add_left_w as Handler;
    t[Opcode::SubLeft as usize] = op_sub_left_w as Handler;
    t[Opcode::MulLeft as usize] = op_mul_left_w as Handler;
    t[Opcode::DivLeft as usize] = op_div_left_w as Handler;
    t[Opcode::Equal as usize] = op_equal_w as Handler;
    t[Opcode::LessThan as usize] = op_less_than_w as Handler;
    t[Opcode::GreaterThan as usize] = op_greater_than_w as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right_w as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed_w as Handler;
    t[Opcode::StoreKeyedPropertyNoShadow as usize] = op_store_keyed_no_shadow_w as Handler;
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
    t[Opcode::CreateEmptyArrayLiteral as usize] = op_create_empty_array_w as Handler;
    t[Opcode::CreateEmptyObjectLiteral as usize] = op_create_empty_object_w as Handler;
    t[Opcode::CallRuntime as usize] = op_call_runtime_w as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call_w as Handler;
    t[Opcode::CallMethod0 as usize] = op_call_method0_w as Handler;
    t[Opcode::CallFunction0 as usize] = op_call_function0_w as Handler;
    t[Opcode::CallFunction1 as usize] = op_call_function1_w as Handler;
    t[Opcode::CallFunction2 as usize] = op_call_function2_w as Handler;
    t[Opcode::CallMethod1 as usize] = op_call_method1_w as Handler;
    t[Opcode::CallMethod2 as usize] = op_call_method2_w as Handler;
    t[Opcode::Construct as usize] = op_construct_w as Handler;
    t
}

static TABLE_NARROW: HandlerTable = HandlerTable(table_narrow());
static TABLE_WIDE: HandlerTable = HandlerTable(table_wide());


fn enter<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    callable: Handle<'_, Object>,
    args: HandleSlice<'_>,
    new_target: Option<Handle<'a, Value>>,
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
        // the proxy dispatch above already intercepted these
        Some(CallTarget::Proxy(_)) => Err(VmError::Type),
        Some(CallTarget::Runtime(idx)) => {
            let f = vm.runtime(RuntimeIndex(idx));
            let (saved_top, fargs) = state.stack().stage_args(heap, args)?;
            let nctx = RuntimeContext::with_new_target(vm, heap, state, new_target);
            let result = f(nctx, fargs);
            state.stack().set_top(saved_top);
            result
        }
        Some(CallTarget::Bytecode {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
            ..
        }) => {
            if new_target.is_none() && kind.is_class_constructor() {
                return Err(VmError::Type);
            }
            let stack = state.stack();
            let cache = state.cache();
            let frame = {
                let heap: &Heap = heap;
                let new_target_value = match &new_target {
                    Some(nt) => nt.as_tagged(heap).erase(),
                    None => heap.known().undefined.as_tagged(heap).erase(),
                };
                stack.push_initial_frame(
                    heap,
                    target.erase(),
                    info,
                    register_count,
                    context.erase(),
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
                base_anchor: frame.base,
                safepoints: Cell::new(SAFEPOINT_INTERVAL),
                num: Cell::new(0.0),
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
    let outer = was_active.then(|| cache.frame_meta());

    state.handle_scope(|scope| {
        let result = enter(vm, heap, state, callable, args, new_target)?;
        let rooted = scope.handle(result);

        stack.set_top(saved_top);
        if let Some(outer) = outer {
            cache.load(stack, outer, heap);
        } else {
            cache.deactivate(heap);
        }
        Ok(rooted.as_tagged(heap))
    })
}

impl Interpreter for BecomeInterpreter {
    const EXECUTE: ExecuteFn = execute;
}
