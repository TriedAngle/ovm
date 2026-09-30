#![feature(explicit_tail_calls)]
#![feature(rust_preserve_none_cc)]
#![allow(incomplete_features)]
#![feature(fn_align)]
#![allow(unused_macros, unused_unsafe, unused_variables)]

use bytecode::{OPERAND_SIZES_NARROW, OPERAND_SIZES_WIDE, Opcode};
use vm_core::StoreSemantics;
use vm_core::ic::{ElementHit, Hit, InlineCache, MonoProbe};
use vm_core::proxy::Proxy;
use vm_core::{
    CallTarget, CallableInfoObject, Coercion, Compare, Context, ContextInit, ContextState, Convert,
    Ctx, ExecuteFn, FixedArray, FrameMeta, FunctionKind, Handle, HandleSlice, Heap, Interpreter,
    Intrinsic, Object, Register, RuntimeContext, RuntimeIndex, ScopeInfo, Smi, Tagged, VM, Value,
    VmError, spread_apply_args,
};

pub struct BecomeInterpreter;

/// The frame's register window: signed indices (negative = parameters).
#[derive(Clone, Copy)]
pub struct Regs(*mut Register);

impl Regs {
    /// # Safety
    /// `base` is a rooted window of the running frame; no GC before the
    /// last use of this handle or anything loaded through it.
    #[inline(always)]
    pub unsafe fn new(base: *mut Register) -> Regs {
        Regs(base)
    }

    /// Safe by the register-file invariant: slots hold strong, valid
    /// values from frame setup until the frame pops (the collector
    /// updates them in place). The heap argument anchors the returned
    /// value to the GC epoch, like `Register::get(heap)`.
    #[inline(always)]
    pub fn read<'h, H>(&self, i: i32, _heap: H) -> Tagged<'h, Value>
    where
        H: AsHeap<'h>,
    {
        // Safety: the invariant above.
        unsafe { Tagged::from_value_unchecked((*self.0.offset(i as isize)).raw()) }
    }

    #[inline(always)]
    pub fn write(&self, i: i32, v: Tagged<'_, Value>) {
        // Safety: the window invariant covers this slot.
        unsafe {
            (*self.0.offset(i as isize)).store(v);
        }
    }
}

/// Witness conversion for epoch-anchored register reads: anything that
/// can lend the current heap.
pub trait AsHeap<'h> {
    fn as_heap(self) -> &'h Heap;
}

impl<'h> AsHeap<'h> for &'h Heap {
    #[inline(always)]
    fn as_heap(self) -> &'h Heap {
        self
    }
}

impl<'h, 'x> AsHeap<'h> for &'x Ctx<'h> {
    #[inline(always)]
    fn as_heap(self) -> &'h Heap {
        self.heap()
    }
}

/// A pinned FP-register argument. Rides the float register pool (d0
/// under preserve-none), so it costs no integer argument registers;
/// forwarding it through `become` is free. Numeric slow paths hand
/// their payload through it instead of parking it in a Ctx cell.
#[derive(Clone, Copy)]
pub struct FloatReg(f64);

impl FloatReg {
    #[inline(always)]
    pub fn new(v: f64) -> FloatReg {
        FloatReg(v)
    }

    #[inline(always)]
    pub fn get(self) -> f64 {
        black_box(self.0)
    }
}

pub type Handler = for<'a> extern "rust-preserve-none" fn(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value>;

/// Fused instruction pointer + pinned dispatch table. The trailing args
/// are Scalar newtypes riding their own argument registers on aarch64
/// and ZSTs the rustic ABI omits elsewhere, so handler signatures are
/// one cfg-free text.
#[derive(Clone, Copy)]
pub struct TableArg<'a>(&'static HandlerTable, PhantomData<&'a ()>);

impl<'a> TableArg<'a> {
    #[inline(always)]
    pub fn new(table: &'static HandlerTable) -> TableArg<'a> {
        TableArg(table, PhantomData)
    }

    #[inline(always)]
    pub fn get(&self, op: u8) -> Handler {
        self.0.0[op as usize]
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[derive(Clone, Copy)]
pub struct RootsArg<'a>(PhantomData<&'a ()>);

#[cfg(not(target_arch = "aarch64"))]
impl<'a> RootsArg<'a> {
    #[inline(always)]
    pub fn new() -> RootsArg<'a> {
        RootsArg(PhantomData)
    }

    #[inline(always)]
    pub fn known<'h>(&self, heap: &'h Heap) -> &'static WellKnown {
        heap.known()
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct RootsArg<'a>(&'static WellKnown, PhantomData<&'a ()>);

#[cfg(target_arch = "aarch64")]
impl<'a> RootsArg<'a> {
    #[inline(always)]
    pub fn new(known: &'static WellKnown) -> RootsArg<'a> {
        RootsArg(known, PhantomData)
    }

    #[inline(always)]
    pub fn known<'h>(&self, _heap: &'h Heap) -> &'static WellKnown {
        self.0
    }
}

use core::hint::black_box;
use std::marker::PhantomData;

use vm_core::bootstrap::WellKnown;

pub struct HandlerTable([Handler; 256]);

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
            | Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Mod
            | Opcode::Exp
            | Opcode::BitwiseOr
            | Opcode::BitwiseXor
            | Opcode::BitwiseAnd
            | Opcode::ShiftLeft
            | Opcode::ShiftRight
            | Opcode::ShiftRightLogical
            | Opcode::AddImmediate
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
            | Opcode::ShiftRightLogicalImmediate
            | Opcode::AddLoc
            | Opcode::SubLoc
    )
}

macro_rules! cold_try {
    ($ctx:ident, $e:expr) => {
        match $e {
            Ok(v) => v,
            // the sentinel VALUE flows on to the caller's `become resume`,
            // which routes it into `throw_dispatch`
            Err(err) => unsafe { $ctx.raise_tag(err) },
        }
    };
}

// Plain-fn handlers using a const-indexed operand cursor and canonical

struct Ops<const STRIDE: usize> {
    ip: *const u8,
    #[cfg(debug_assertions)]
    op: Opcode,
}

impl<const STRIDE: usize> Ops<STRIDE> {
    #[inline(always)]
    fn new(ip: *const u8, op: Opcode) -> Ops<STRIDE> {
        debug_assert_eq!(
            if STRIDE == 2 {
                unsafe { *ip.add(1) }
            } else {
                unsafe { *ip }
            },
            op as u8,
            "handler dispatched for the wrong opcode",
        );
        Ops {
            ip,
            #[cfg(debug_assertions)]
            op,
        }
    }

    #[inline(always)]
    fn signed<const I: usize>(&self) -> i32 {
        #[cfg(debug_assertions)]
        debug_assert!(
            I < self.op.operands().len() && self.op.operands()[I].is_signed(),
            "operand {I} of {:?} is not a signed operand",
            self.op
        );
        let off = base::<STRIDE>() + I * STRIDE;
        unsafe {
            if STRIDE == 1 {
                *self.ip.add(off) as i8 as i32
            } else {
                i16::from_le_bytes([*self.ip.add(off), *self.ip.add(off + 1)]) as i32
            }
        }
    }

    #[inline(always)]
    fn unsigned<const I: usize>(&self) -> usize {
        #[cfg(debug_assertions)]
        debug_assert!(
            I < self.op.operands().len() && !self.op.operands()[I].is_signed(),
            "operand {I} of {:?} is not an unsigned operand",
            self.op
        );
        let off = base::<STRIDE>() + I * STRIDE;
        unsafe {
            if STRIDE == 1 {
                *self.ip.add(off) as usize
            } else {
                u16::from_le_bytes([*self.ip.add(off), *self.ip.add(off + 1)]) as usize
            }
        }
    }
}

const fn base<const STRIDE: usize>() -> usize {
    if STRIDE == 2 { 2 } else { 1 }
}

macro_rules! dispatch {
    ($p:expr, $a:expr, $r:expr, $x:expr, $t:ident, $k:ident, $f:ident) => {{
        let h = $t.get(unsafe { *$p });
        become h($p, $r, $a, $x, $t, $k, $f)
    }};
}

macro_rules! jump {
    ($ip:expr, $off:expr, $a:expr, $r:expr, $x:expr, $t:ident, $k:ident, $f:ident) => {
        dispatch!($ip.wrapping_offset($off as isize), $a, $r, $x, $t, $k, $f)
    };
}

macro_rules! reenter {
    ($x:expr, $pcrel:expr, $delta:expr, $a:expr, $t:ident, $k:ident, $f:ident) => {{
        let p = $pcrel.wrapping_add($delta);
        let c = $x.code_ptr();
        let r = unsafe { Regs::new($x.regs_ptr()) };
        dispatch!(p, $a, r, $x, $t, $k, $f)
    }};
}

macro_rules! bail {
    ($acc:expr, $ip:expr, $r:expr, $x:expr, $t:ident, $k:ident, $f:ident, $e:expr) => {{
        unsafe {
            let _ = $x.raise_tag($e);
        }
        become throw_dispatch($ip, $r, $acc, $x, $t, $k, $f)
    }};
}

macro_rules! threw {
    ($acc:expr, $ip:expr, $r:expr, $x:expr, $t:ident, $k:ident, $f:ident) => {
        become throw_dispatch($ip, $r, $acc, $x, $t, $k, $f)
    };
}

macro_rules! next {
    ($op:ident, $ip:expr, $r:expr, $x:expr, $t:ident, $k:ident, $f:ident, $a:expr) => {{
        let p = $ip.wrapping_add(
            (if STRIDE == 2 { 2 } else { 1 }) + STRIDE * const { Opcode::$op.operands().len() },
        );
        #[cfg(feature = "star_fusion")]
        {
            if star_lookahead(Opcode::$op) && unsafe { *p } == Opcode::Store as u8 {
                let dst = unsafe { *p.add(1) } as i8 as i32;
                $r.write(dst, $a);
                dispatch!(unsafe { p.add(2) }, $a, $r, $x, $t, $k, $f)
            } else {
                dispatch!(p, $a, $r, $x, $t, $k, $f)
            }
        }
        #[cfg(not(feature = "star_fusion"))]
        {
            dispatch!(p, $a, $r, $x, $t, $k, $f)
        }
    }};
}

macro_rules! cold_start {
    ($ip:ident, $regs:ident, $acc:ident, $ctx:ident, $t:ident, $k:ident, $f:ident, $e:expr) => {
        match $e {
            Ok(m) => m,
            Err(err) => {
                unsafe {
                    let _ = $ctx.raise_tag(err);
                }
                become throw_dispatch($ip, $regs, $acc, $ctx, $t, $k, $f)
            }
        }
    };
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_wide<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let h = TABLE_WIDE.0[unsafe { *ip.add(1) } as usize];
    become h(ip, regs, acc, ctx, table, roots, float)
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_trap<'a>(
    ip: *const u8,
    _regs: Regs,
    _acc: Tagged<'a, Value>,
    _ctx: &Ctx<'a>,
    _table: TableArg<'a>,
    _roots: RootsArg<'a>,
    _float: FloatReg,
) -> Tagged<'a, Value> {
    let op = unsafe { *ip };
    panic!("become interpreter: opcode {op} (at +{})", ip as usize)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Load);
    let r = ops.signed::<0>();
    next!(Load, ip, regs, ctx, table, roots, float, regs.read(r, ctx))
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_move<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Move);
    let dst = ops.signed::<0>();
    let src = ops.signed::<1>();
    let v = regs.read(src, ctx);
    regs.write(dst, v);
    next!(Move, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Store);
    let r = ops.signed::<0>();
    regs.write(r, acc);
    next!(Store, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_smi<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadSmi);
    let imm = ops.signed::<0>();
    next!(
        LoadSmi,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new(imm as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_constant<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadConstant);
    let idx = ops.unsigned::<0>();
    let v = ctx.cache().constants_ref(ctx.heap()).at(ctx.heap(), idx);
    next!(LoadConstant, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_zero<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadZero);
    next!(
        LoadZero,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new(0).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_undefined<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadUndefined);
    let v = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
    next!(LoadUndefined, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_null<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadNull);
    let v = ctx.heap().known().null.as_tagged(ctx.heap()).erase();
    next!(LoadNull, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_true<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadTrue);
    let v = ctx.heap().known().true_object.as_tagged(ctx.heap()).erase();
    next!(LoadTrue, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_false<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadFalse);
    let v = ctx
        .heap()
        .known()
        .false_object
        .as_tagged(ctx.heap())
        .erase();
    next!(LoadFalse, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_add<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Add);
    let r = ops.signed::<0>();
    let lhs = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
        && let Some(sum) = a.checked_add(b)
    {
        next!(
            Add,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(sum)
        )
    }
    if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
        let sum = a + b;
        if let Some(s) = Smi::from_f64(sum) {
            next!(Add, ip, regs, ctx, table, roots, float, s.into_tagged())
        }
        become cold_box_number(ip, regs, acc, ctx, table, roots, FloatReg::new(sum))
    }
    become cold_add(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_sub<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Sub);
    let r = ops.signed::<0>();
    let lhs = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
        && let Some(diff) = a.checked_sub(b)
    {
        next!(
            Sub,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(diff)
        )
    }
    if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
        let diff = a - b;
        if let Some(s) = Smi::from_f64(diff) {
            next!(Sub, ip, regs, ctx, table, roots, float, s.into_tagged())
        }
        become cold_box_number(ip, regs, acc, ctx, table, roots, FloatReg::new(diff))
    }
    become cold_numeric(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_mul<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Mul);
    let r = ops.signed::<0>();
    let lhs = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
        && let Some(product) = (a >> 1).checked_mul(b)
    {
        next!(
            Mul,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(product)
        )
    }
    if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
        let product = a * b;
        if let Some(s) = Smi::from_f64(product) {
            next!(Mul, ip, regs, ctx, table, roots, float, s.into_tagged())
        }
        become cold_box_number(ip, regs, acc, ctx, table, roots, FloatReg::new(product))
    }
    become cold_numeric(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_add_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::AddImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let lhs = regs.read(r, ctx);
    if let Some(bits) = lhs.smi_bits()
        && let Some(sum) = bits.checked_add((imm as i64) << 1)
    {
        next!(
            AddImmediate,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(sum)
        )
    }
    become cold_add_immediate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_sub_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::SubImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let lhs = regs.read(r, ctx);
    if let Some(bits) = lhs.smi_bits()
        && let Some(diff) = bits.checked_sub((imm as i64) << 1)
    {
        next!(
            SubImmediate,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(diff)
        )
    }
    become cold_numeric_immediate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_inc_loc<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::IncLoc);
    let r = ops.signed::<0>();
    let v = regs.read(r, ctx);
    if let Some(bits) = v.smi_bits()
        && let Some(new) = bits.checked_add(2)
    {
        regs.write(r, Tagged::from_smi_bits(new));
        next!(IncLoc, ip, regs, ctx, table, roots, float, v)
    }
    become cold_inc_loc(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_dec_loc<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::DecLoc);
    let r = ops.signed::<0>();
    let v = regs.read(r, ctx);
    if let Some(bits) = v.smi_bits()
        && let Some(new) = bits.checked_sub(2)
    {
        regs.write(r, Tagged::from_smi_bits(new));
        next!(DecLoc, ip, regs, ctx, table, roots, float, v)
    }
    become cold_dec_loc(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_equal<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Equal);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
        next!(
            Equal,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Convert::boolean(ctx.heap(), a == b)
        )
    }
    become cold_equal(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_less_than<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LessThan);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
        next!(
            LessThan,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Convert::boolean(ctx.heap(), a < b)
        )
    }
    become cold_less_than(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_greater_than<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::GreaterThan);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
        next!(
            GreaterThan,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Convert::boolean(ctx.heap(), a > b)
        )
    }
    become cold_greater_than(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_jump<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Jump);
    let off = ops.signed::<0>();
    jump!(ip, off, acc, regs, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_jump_if_truthy<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::JumpIfTruthy);
    let off = ops.signed::<0>();
    if Convert::is_truthy(ctx.heap(), acc) {
        jump!(ip, off, acc, regs, ctx, table, roots, float)
    }
    next!(JumpIfTruthy, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_jump_if_falsy<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::JumpIfFalsy);
    let off = ops.signed::<0>();
    if !Convert::is_truthy(ctx.heap(), acc) {
        jump!(ip, off, acc, regs, ctx, table, roots, float)
    }
    next!(JumpIfFalsy, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_return<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::Return);
    if ctx.cache().base() == ctx.base_anchor() {
        return acc;
    }
    let pc = ip as usize - ctx.code_ptr() as usize;
    let meta = ctx.meta(pc);
    let caller = ctx.stack().pop_frame(&meta);
    ctx.cache()
        .load(ctx.stack(), caller, unsafe { ctx.heap_mut() });
    let rel = ctx.cache().pc();
    let base = ctx.code_ptr();
    dispatch!(
        unsafe { base.add(rel) },
        acc,
        unsafe { Regs::new(ctx.regs_ptr()) },
        ctx,
        table,
        roots,
        float
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_bitwise_or<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::BitwiseOr);
    let r = ops.signed::<0>();
    let (Some(a), Some(b)) = (regs.read(r, ctx).to_i64(), acc.to_i64()) else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        BitwiseOr,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32 | b as i32) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_bitwise_xor<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::BitwiseXor);
    let r = ops.signed::<0>();
    let (Some(a), Some(b)) = (regs.read(r, ctx).to_i64(), acc.to_i64()) else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        BitwiseXor,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32 ^ b as i32) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_bitwise_and<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::BitwiseAnd);
    let r = ops.signed::<0>();
    let (Some(a), Some(b)) = (regs.read(r, ctx).to_i64(), acc.to_i64()) else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        BitwiseAnd,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32 & b as i32) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_shift_left<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ShiftLeft);
    let r = ops.signed::<0>();
    let (Some(a), Some(b)) = (regs.read(r, ctx).to_i64(), acc.to_i64()) else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        ShiftLeft,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32).wrapping_shl(b as u32 & 31) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_shift_right<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ShiftRight);
    let r = ops.signed::<0>();
    let (Some(a), Some(b)) = (regs.read(r, ctx).to_i64(), acc.to_i64()) else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        ShiftRight,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32).wrapping_shr(b as u32 & 31) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_shift_right_logical<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ShiftRightLogical);
    let r = ops.signed::<0>();
    let (Some(a), Some(b)) = (regs.read(r, ctx).to_i64(), acc.to_i64()) else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        ShiftRightLogical,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as u32).wrapping_shr(b as u32 & 31) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_bitwise_or_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::BitwiseOrImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let Some(a) = regs.read(r, ctx).to_i64() else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        BitwiseOrImmediate,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32 | imm) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_bitwise_xor_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::BitwiseXorImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let Some(a) = regs.read(r, ctx).to_i64() else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        BitwiseXorImmediate,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32 ^ imm) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_bitwise_and_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::BitwiseAndImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let Some(a) = regs.read(r, ctx).to_i64() else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        BitwiseAndImmediate,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32 & imm) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_shift_left_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ShiftLeftImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let Some(a) = regs.read(r, ctx).to_i64() else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        ShiftLeftImmediate,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32).wrapping_shl(imm as u32 & 31) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_shift_right_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ShiftRightImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let Some(a) = regs.read(r, ctx).to_i64() else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        ShiftRightImmediate,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as i32).wrapping_shr(imm as u32 & 31) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_shift_right_logical_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ShiftRightLogicalImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let Some(a) = regs.read(r, ctx).to_i64() else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    next!(
        ShiftRightLogicalImmediate,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Smi::new((a as u32).wrapping_shr(imm as u32 & 31) as i64).into_tagged()
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_exp<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Exp);
    let _ = ops.signed::<0>();
    become cold_numeric(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_div<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Div);
    let r = ops.signed::<0>();
    let lhs = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
        && b != 0
        && a % b == 0
        && let Some(quotient) = a.checked_div(b)
        && let Some(encoded) = quotient.checked_mul(2)
    {
        next!(
            Div,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(encoded)
        )
    }
    if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
        let quotient = a / b;
        if let Some(s) = Smi::from_f64(quotient) {
            next!(Div, ip, regs, ctx, table, roots, float, s.into_tagged())
        }
        become cold_box_number(ip, regs, acc, ctx, table, roots, FloatReg::new(quotient))
    }
    become cold_numeric(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_mod<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Mod);
    let r = ops.signed::<0>();
    let lhs = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (lhs.to_i64(), acc.to_i64())
        && b != 0
    {
        next!(
            Mod,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Smi::new(a % b).into_tagged()
        )
    }
    become cold_numeric(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_mul_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::MulImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let lhs = regs.read(r, ctx);
    if let Some(bits) = lhs.smi_bits()
        && let Some(product) = (bits >> 1).checked_mul(imm as i64)
        && let Some(encoded) = product.checked_mul(2)
    {
        next!(
            MulImmediate,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(encoded)
        )
    }
    become cold_numeric_immediate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_div_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::DivImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let lhs = regs.read(r, ctx);
    if let Some(bits) = lhs.smi_bits() {
        let value = bits >> 1;
        if imm != 0
            && value % imm as i64 == 0
            && let Some(quotient) = value.checked_div(imm as i64)
            && let Some(encoded) = quotient.checked_mul(2)
        {
            next!(
                DivImmediate,
                ip,
                regs,
                ctx,
                table,
                roots,
                float,
                Tagged::from_smi_bits(encoded)
            )
        }
    }
    become cold_numeric_immediate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_mod_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ModImmediate);
    let r = ops.signed::<0>();
    let imm = ops.signed::<1>();
    let lhs = regs.read(r, ctx);
    if imm != 0
        && let Some(bits) = lhs.smi_bits()
    {
        next!(
            ModImmediate,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(((bits >> 1) % imm as i64) << 1)
        )
    }
    become cold_numeric_immediate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_exp_immediate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ExpImmediate);
    let _ = (ops.signed::<0>(), ops.signed::<1>());
    become cold_numeric_immediate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_negate<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::Negate);
    if let Some(bits) = acc.smi_bits()
        && bits != 0
        && let Some(neg) = bits.checked_neg()
    {
        next!(
            Negate,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Tagged::from_smi_bits(neg)
        )
    }
    if let Some(a) = Convert::as_number(acc) {
        let neg = -a;
        if let Some(s) = Smi::from_f64(neg) {
            next!(Negate, ip, regs, ctx, table, roots, float, s.into_tagged())
        }
        become cold_box_number(ip, regs, acc, ctx, table, roots, FloatReg::new(neg))
    }
    become cold_negate(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_compare_jump<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CompareJump);
    let r = ops.signed::<0>();
    let kind = ops.unsigned::<1>();
    let off = ops.signed::<2>();
    let other = regs.read(r, ctx);
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
            jump!(ip, off, boolean, regs, ctx, table, roots, float)
        } else {
            next!(CompareJump, ip, regs, ctx, table, roots, float, boolean)
        }
    }
    become cold_compare_jump(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_throw<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::Throw);
    ctx.state().set_pending_exception(acc);
    threw!(acc, ip, regs, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_rethrow<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::ReThrow);
    ctx.state().set_pending_exception(acc);
    threw!(acc, ip, regs, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_hole<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadHole);
    let v = ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase();
    next!(LoadHole, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_add_loc<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::AddLoc);
    let dst = ops.signed::<0>();
    let src = ops.signed::<1>();
    let lhs = regs.read(dst, ctx);
    let rhs = regs.read(src, ctx);
    if let (Some(a), Some(b)) = (lhs.smi_bits(), rhs.smi_bits())
        && let Some(sum) = a.checked_add(b)
    {
        let v = Tagged::from_smi_bits(sum);
        regs.write(dst, v);
        next!(AddLoc, ip, regs, ctx, table, roots, float, v)
    }
    if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(rhs)) {
        let sum = a + b;
        if let Some(s) = Smi::from_f64(sum) {
            let v = s.into_tagged();
            regs.write(dst, v);
            next!(AddLoc, ip, regs, ctx, table, roots, float, v)
        }
        become cold_box_add_loc(ip, regs, acc, ctx, table, roots, FloatReg::new(sum))
    }
    become cold_add_loc(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_sub_loc<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::SubLoc);
    let dst = ops.signed::<0>();
    let src = ops.signed::<1>();
    let lhs = regs.read(dst, ctx);
    let rhs = regs.read(src, ctx);
    if let (Some(a), Some(b)) = (lhs.smi_bits(), rhs.smi_bits())
        && let Some(diff) = a.checked_sub(b)
    {
        let v = Tagged::from_smi_bits(diff);
        regs.write(dst, v);
        next!(SubLoc, ip, regs, ctx, table, roots, float, v)
    }
    if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(rhs)) {
        let diff = a - b;
        if let Some(s) = Smi::from_f64(diff) {
            let v = s.into_tagged();
            regs.write(dst, v);
            next!(SubLoc, ip, regs, ctx, table, roots, float, v)
        }
        become cold_box_sub_loc(ip, regs, acc, ctx, table, roots, FloatReg::new(diff))
    }
    become cold_sub_loc(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_jump_loop<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::JumpLoop);
    let off = ops.signed::<0>();
    if ctx.safepoint_tick() {
        ctx.cache().acc_mut().store(acc);
        if unsafe { ctx.heap_mut() }.safepoint_poll() {
            let state = ctx.state();
            state.set_termination(vm_core::Termination::Shutdown);
            let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap());
            state.set_pending_exception(undefined);
            threw!(acc, ip, regs, ctx, table, roots, float)
        }
        let pc = ip as usize - ctx.code_ptr() as usize + off as usize;
        let acc = ctx.cache().acc(ctx.heap());
        reenter!(
            ctx,
            unsafe { ctx.code_ptr().add(pc) },
            0,
            acc,
            table,
            roots,
            float
        )
    } else {
        jump!(ip, off, acc, regs, ctx, table, roots, float)
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_keyed_reg<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadKeyedPropertyReg);
    let recv = ops.signed::<0>();
    let key = ops.signed::<1>();
    let fb = ops.unsigned::<2>();
    let recv = regs.read(recv, ctx);
    let key = regs.read(key, ctx);
    if let Some(idx) = key.to_i64()
        && idx >= 0
    {
        if let Some(obj) = recv.as_heap_object()
            && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx as usize)
        {
            next!(LoadKeyedPropertyReg, ip, regs, ctx, table, roots, float, v)
        }
        if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv,
            idx as usize,
        ) {
            next!(LoadKeyedPropertyReg, ip, regs, ctx, table, roots, float, v)
        }
    }
    become cold_keyed_load_reg(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_keyed<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreKeyedProperty);
    let recv = ops.signed::<0>();
    let key = ops.signed::<1>();
    let fb = ops.unsigned::<2>();
    let recv_w = regs.read(recv, ctx);
    let key_w = regs.read(key, ctx);
    if let Some(idx) = key_w.to_i64()
        && idx >= 0
    {
        if Object::store_array_element_in_place(ctx.heap(), recv_w, idx as usize, acc).is_ok() {
            next!(StoreKeyedProperty, ip, regs, ctx, table, roots, float, acc)
        }
        if let Some(v) = InlineCache::try_store_element(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv_w,
            idx as usize,
            acc,
        ) {
            next!(StoreKeyedProperty, ip, regs, ctx, table, roots, float, v)
        }
    }
    become cold_keyed_store(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_keyed_no_shadow<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreKeyedPropertyNoShadow);
    let recv = ops.signed::<0>();
    let key = ops.signed::<1>();
    let fb = ops.unsigned::<2>();
    let recv_w = regs.read(recv, ctx);
    let key_w = regs.read(key, ctx);
    if let Some(idx) = key_w.to_i64()
        && idx >= 0
    {
        if Object::store_array_element_in_place(ctx.heap(), recv_w, idx as usize, acc).is_ok() {
            next!(
                StoreKeyedPropertyNoShadow,
                ip,
                regs,
                ctx,
                table,
                roots,
                float,
                acc
            )
        }
        if let Some(v) = InlineCache::try_store_element(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv_w,
            idx as usize,
            acc,
        ) {
            next!(
                StoreKeyedPropertyNoShadow,
                ip,
                regs,
                ctx,
                table,
                roots,
                float,
                v
            )
        }
    }
    become cold_keyed_store_no_shadow(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_named<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadNamedProperty);
    let r = ops.signed::<0>();
    let fb = ops.unsigned::<2>();
    let recv = regs.read(r, ctx);
    match InlineCache::probe_mono(ctx.heap(), ctx.cache().feedback_ref(ctx.heap()), fb, recv) {
        MonoProbe::Value(v) => next!(LoadNamedProperty, ip, regs, ctx, table, roots, float, v),
        MonoProbe::Handler(obj, handler) => {
            if let Some(Hit::Value(v)) = InlineCache::apply_mono(ctx.heap(), obj, handler) {
                next!(LoadNamedProperty, ip, regs, ctx, table, roots, float, v)
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
                next!(LoadNamedProperty, ip, regs, ctx, table, roots, float, v)
            }
        }
        MonoProbe::Miss | MonoProbe::NotReceiver => {}
    }
    become cold_named_load(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_keyed<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadKeyedProperty);
    let r = ops.signed::<0>();
    let fb = ops.unsigned::<1>();
    let recv = regs.read(r, ctx);
    if let Some(idx) = acc.to_i64()
        && idx >= 0
    {
        if let Some(obj) = recv.as_heap_object()
            && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx as usize)
        {
            next!(LoadKeyedProperty, ip, regs, ctx, table, roots, float, v)
        }
        if let Some(ElementHit::Value(v)) = InlineCache::try_load_element(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            recv,
            idx as usize,
        ) {
            next!(LoadKeyedProperty, ip, regs, ctx, table, roots, float, v)
        }
    }
    become cold_keyed_load(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_element_imm<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadElementImm);
    let r = ops.signed::<0>();
    let idx = ops.unsigned::<1>();
    let recv = regs.read(r, ctx);
    if let Some(obj) = recv.as_heap_object()
        && let Some(v) = obj.as_ref().element_value(ctx.heap(), idx)
    {
        next!(LoadElementImm, ip, regs, ctx, table, roots, float, v)
    }
    become cold_keyed_load_imm(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_global<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadGlobal);
    let fb = ops.unsigned::<1>();
    let global = ctx
        .heap()
        .known()
        .global_object
        .as_tagged(ctx.heap())
        .erase();
    if let Some(Hit::Value(v)) =
        InlineCache::try_load(ctx.heap(), ctx.cache().feedback_ref(ctx.heap()), fb, global)
    {
        next!(LoadGlobal, ip, regs, ctx, table, roots, float, v)
    }
    become cold_global_load(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_named<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreNamedProperty);
    let r = ops.signed::<0>();
    let fb = ops.unsigned::<2>();
    let recv = regs.read(r, ctx);
    if InlineCache::try_store_fast(
        unsafe { ctx.heap_mut() },
        ctx.cache().feedback_ref(ctx.heap()),
        fb,
        recv,
        acc,
    ) {
        next!(StoreNamedProperty, ip, regs, ctx, table, roots, float, acc)
    }
    become cold_store_named(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_context_slot<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadContextSlot);
    let slot = ops.unsigned::<0>();
    let depth = ops.unsigned::<1>();
    let meta = ctx.meta(0);
    if depth == 0 {
        let heap = ctx.heap();
        let context_word = ctx.stack().context_slot(&meta).get(heap);
        let context = unsafe { context_word.cast::<Context>() };
        let slots = context.as_ref().slots.get(heap);
        let slots = unsafe { slots.cast::<FixedArray>() };
        let v = slots.as_ref().element_slot(slot).get(heap);
        next!(LoadContextSlot, ip, regs, ctx, table, roots, float, v)
    }
    let v = {
        let heap = ctx.heap();
        let Some(mut context) = ctx
            .stack()
            .context_slot(&meta)
            .get(heap)
            .get_as::<Context>()
        else {
            bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
        };
        for _ in 0..depth {
            context = match context.as_ref().outer.get(heap) {
                Some(context) => context,
                None => bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type),
            };
        }
        context
            .slots
            .get(heap)
            .as_ref()
            .element_slot(slot)
            .get(heap)
    };
    next!(LoadContextSlot, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_context_slot<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreContextSlot);
    let slot = ops.unsigned::<0>();
    let depth = ops.unsigned::<1>();
    let meta = ctx.meta(0);
    let heap = ctx.heap();
    let Some(mut context) = ctx
        .stack()
        .context_slot(&meta)
        .get(heap)
        .get_as::<Context>()
    else {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    };
    for _ in 0..depth {
        context = match context.as_ref().outer.get(heap) {
            Some(context) => context,
            None => bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type),
        };
    }
    let host = context.erase();
    context
        .slots
        .get(heap)
        .as_ref()
        .element_slot(slot)
        .set(heap, host, acc);
    next!(StoreContextSlot, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_push_context<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::PushContext);
    let r = ops.signed::<0>();
    let meta = ctx.meta(0);
    let old = ctx.stack().context_slot(&meta).get(ctx.heap());
    regs.write(r, old);
    if acc.get_as::<Context>().is_none() {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    }
    ctx.stack().context_slot(&meta).store(acc);
    next!(PushContext, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_pop_context<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::PopContext);
    let r = ops.signed::<0>();
    let meta = ctx.meta(0);
    let context = regs.read(r, ctx);
    if context.get_as::<Context>().is_none() {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    }
    ctx.stack().context_slot(&meta).store(context);
    next!(PopContext, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_create_function_context<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CreateFunctionContext);
    let _ = ops.unsigned::<0>();
    become cold_create_function_context(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_create_closure<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CreateClosure);
    let _ = ops.unsigned::<0>();
    become cold_create_closure(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_create_empty_array<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::CreateEmptyArrayLiteral);
    become cold_create_empty_array(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_create_empty_object<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::CreateEmptyObjectLiteral);
    become cold_create_empty_object(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_new_target<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadNewTarget);
    let v = ctx.stack().new_target_slot(&ctx.meta(0)).get(ctx.heap());
    next!(LoadNewTarget, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_context<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadContext);
    let v = ctx.stack().context_slot(&ctx.meta(0)).get(ctx.heap());
    next!(LoadContext, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_throw_reference_error_if_hole<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::ThrowReferenceErrorIfHole);
    if acc == ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase() {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Reference)
    }
    next!(
        ThrowReferenceErrorIfHole,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        acc
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_test_reference_equal<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::TestReferenceEqual);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    next!(
        TestReferenceEqual,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Convert::boolean(ctx.heap(), other == acc)
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_test_typeof<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::TestTypeof);
    next!(
        TestTypeof,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Object::type_of(ctx.heap(), acc)
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_equal_strict<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::EqualStrict);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    next!(
        EqualStrict,
        ip,
        regs,
        ctx,
        table,
        roots,
        float,
        Convert::boolean(ctx.heap(), Compare::strict_equal(ctx.heap(), acc, other))
    )
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_jump_if_not_undefined<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::JumpIfNotUndefined);
    let off = ops.signed::<0>();
    if acc != ctx.heap().known().undefined.as_tagged(ctx.heap()).erase() {
        jump!(ip, off, acc, regs, ctx, table, roots, float)
    }
    next!(JumpIfNotUndefined, ip, regs, ctx, table, roots, float, acc)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_less_than_or_equal<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LessThanOrEqual);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
        next!(
            LessThanOrEqual,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Convert::boolean(ctx.heap(), a <= b)
        )
    }
    become cold_less_than_or_equal(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_greater_than_or_equal<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::GreaterThanOrEqual);
    let r = ops.signed::<0>();
    let other = regs.read(r, ctx);
    if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
        next!(
            GreaterThanOrEqual,
            ip,
            regs,
            ctx,
            table,
            roots,
            float,
            Convert::boolean(ctx.heap(), a >= b)
        )
    }
    become cold_greater_than_or_equal(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_create_bare_object<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::CreateBareObjectLiteral);
    become cold_create_bare_object(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_create_block_context<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CreateBlockContext);
    let _ = ops.unsigned::<0>();
    become cold_create_block_context(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_global_fast<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadGlobalFast);
    let fb = ops.unsigned::<1>();
    let global = ctx
        .heap()
        .known()
        .global_object
        .as_tagged(ctx.heap())
        .erase();
    if let Some(Hit::Value(v)) =
        InlineCache::try_load(ctx.heap(), ctx.cache().feedback_ref(ctx.heap()), fb, global)
    {
        next!(LoadGlobalFast, ip, regs, ctx, table, roots, float, v)
    }
    become cold_global_load(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_global_nothrow<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadGlobalNoThrow);
    let _ = (ops.unsigned::<0>(), ops.unsigned::<1>());
    become cold_global_load_nothrow(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_global<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreGlobal);
    let _ = (ops.unsigned::<0>(), ops.unsigned::<1>());
    become cold_store_global(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_named_no_shadow<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreNamedPropertyNoShadow);
    let _ = (ops.signed::<0>(), ops.unsigned::<1>(), ops.unsigned::<2>());
    become cold_store_named_no_shadow(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_store_named_no_shadow_fast<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::StoreNamedPropertyNoShadowFast);
    let _ = (ops.signed::<0>(), ops.unsigned::<1>(), ops.unsigned::<2>());
    become cold_store_named_no_shadow(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_instance_of<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::InstanceOf);
    let _ = ops.signed::<0>();
    become cold_instance_of(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_current_closure<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    Ops::<STRIDE>::new(ip, Opcode::LoadCurrentClosure);
    let v = ctx.stack().callable_slot(&ctx.meta(0)).get(ctx.heap());
    next!(LoadCurrentClosure, ip, regs, ctx, table, roots, float, v)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_add_parent<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::AddParent);
    let _ = (ops.signed::<0>(), ops.unsigned::<1>());
    become cold_add_parent(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_named_fast<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadNamedPropertyFast);
    let r = ops.signed::<0>();
    let fb = ops.unsigned::<2>();
    let recv = regs.read(r, ctx);
    match InlineCache::probe_mono(ctx.heap(), ctx.cache().feedback_ref(ctx.heap()), fb, recv) {
        MonoProbe::Value(v) => next!(LoadNamedPropertyFast, ip, regs, ctx, table, roots, float, v),
        MonoProbe::Handler(obj, handler) => {
            if let Some(Hit::Value(v)) = InlineCache::apply_mono(ctx.heap(), obj, handler) {
                next!(LoadNamedPropertyFast, ip, regs, ctx, table, roots, float, v)
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
                next!(LoadNamedPropertyFast, ip, regs, ctx, table, roots, float, v)
            }
        }
        MonoProbe::Miss | MonoProbe::NotReceiver => {}
    }
    become cold_named_load(ip, regs, acc, ctx, table, roots, float)
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_runtime<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallRuntime);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallRuntime.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let rt = ops.unsigned::<0>();
    let base = ops.signed::<1>();
    let count = ops.unsigned::<2>();
    let f = ctx.vm().runtime(RuntimeIndex(rt));
    let meta = ctx.meta(0);
    let args = ctx.stack().args(&meta, base, count);
    let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
    let v = f(nctx, args);
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

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_method0<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallMethod0);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallMethod0.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let recv = ops.signed::<1>();
    let fb = ops.unsigned::<2>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_method_start(ctx, pc, size, callee_word, &[recv], fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => become cold_call_method_proxy(ip, regs, acc, ctx, table, roots, float),
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_method1<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallMethod1);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallMethod1.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let recv = ops.signed::<1>();
    let arg0 = ops.signed::<2>();
    let fb = ops.unsigned::<3>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_method_start(ctx, pc, size, callee_word, &[recv, arg0], fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => become cold_call_method_proxy(ip, regs, acc, ctx, table, roots, float),
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_method2<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallMethod2);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallMethod2.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let recv = ops.signed::<1>();
    let arg0 = ops.signed::<2>();
    let arg1 = ops.signed::<3>();
    let fb = ops.unsigned::<4>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_method_start(ctx, pc, size, callee_word, &[recv, arg0, arg1], fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => become cold_call_method_proxy(ip, regs, acc, ctx, table, roots, float),
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_function0<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallFunction0);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallFunction0.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let fb = ops.unsigned::<1>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_function_start(ctx, pc, size, callee_word, &[], fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => {
            become cold_call_function_proxy(ip, regs, acc, ctx, table, roots, float)
        }
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_function1<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallFunction1);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallFunction1.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let arg0 = ops.signed::<1>();
    let fb = ops.unsigned::<2>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_function_start(ctx, pc, size, callee_word, &[arg0], fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => {
            become cold_call_function_proxy(ip, regs, acc, ctx, table, roots, float)
        }
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_function2<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallFunction2);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallFunction2.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let arg0 = ops.signed::<1>();
    let arg1 = ops.signed::<2>();
    let fb = ops.unsigned::<3>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_function_start(ctx, pc, size, callee_word, &[arg0, arg1], fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => {
            become cold_call_function_proxy(ip, regs, acc, ctx, table, roots, float)
        }
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call_ic<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Call);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::Call.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let base_r = ops.signed::<1>();
    let count = ops.unsigned::<2>();
    let fb = ops.unsigned::<3>();
    let callee_word = regs.read(callee, ctx);
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_start(ctx, pc, size, callee_word, base_r, count, fb)
    );
    match __mc {
        MethodCall::Value(v) => {
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
        MethodCall::Frame(_frame) => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }

        MethodCall::Intrinsic(i) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        MethodCall::Proxy => become cold_proxy_apply(ip, regs, acc, ctx, table, roots, float),
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_call<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::CallNoFeedback);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::CallNoFeedback.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let base_r = ops.signed::<1>();
    let count = ops.unsigned::<2>();
    let callee_word = regs.read(callee, ctx);
    match Object::call_target(ctx.heap(), callee_word) {
        None => bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type),
        Some(CallTarget::Proxy(_)) => {
            become cold_proxy_apply(ip, regs, acc, ctx, table, roots, float)
        }
        Some(CallTarget::Intrinsic(i)) => {
            become INTRINSICS[i.id()](ip, regs, acc, ctx, table, roots, float)
        }
        Some(CallTarget::Runtime(idx)) => {
            let f = ctx.vm().runtime(RuntimeIndex(idx));
            let meta = ctx.meta(0);
            let args = ctx.stack().args(&meta, base_r, count);
            let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
            let v = f(nctx, args);
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
                bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
            }
            let heap = unsafe { ctx.heap_mut() };
            let meta = ctx.meta(pc + size);
            let undefined = heap.known().undefined.as_tagged(heap).erase();
            let frame = match ctx.stack().push_frame(
                heap,
                meta,
                pc,
                target.erase(),
                info,
                register_count,
                context.erase(),
                base_r,
                count,
                undefined,
                formal_min,
            ) {
                Ok(frame) => frame,
                Err(err) => return unsafe { ctx.raise_tag(err) },
            };
            ctx.cache()
                .load(ctx.stack(), frame, unsafe { ctx.heap_mut() });
            let callee_pc = ctx.cache().pc();
            let callee_base = ctx.code_ptr();
            let r = unsafe { Regs::new(ctx.regs_ptr()) };
            let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
            dispatch!(
                unsafe { callee_base.add(callee_pc) },
                undefined,
                r,
                ctx,
                table,
                roots,
                float
            )
        }
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_construct<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::Construct);
    let size = base::<STRIDE>() + STRIDE * const { Opcode::Construct.operands().len() };
    let pc = ip as usize - ctx.code_ptr() as usize;
    let callee = ops.signed::<0>();
    let base_r = ops.signed::<1>();
    let count = ops.unsigned::<2>();
    let out_r = ops.signed::<3>();
    let __mc = cold_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        construct_start(ctx, pc, size, regs, callee, base_r, count, out_r)
    );
    match __mc {
        ConstructStart::Frame => {
            become call_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
        ConstructStart::Runtime(idx) => {
            regs.write(out_r, ctx.undefined_word());
            let v = dispatch_runtime_construct(ctx, idx, callee, base_r, count);
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
        ConstructStart::Threw(()) => {
            become throw_dispatch(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
        ConstructStart::Cold => {
            regs.write(out_r, ctx.undefined_word());
            become cold_construct(ip, regs, acc, ctx, table, roots, float)
        }
    }
}

#[rustc_align(32)]
extern "rust-preserve-none" fn op_construct_check<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::ConstructCheck);
    let out_r = ops.signed::<0>();
    let v = if Convert::is_primitive(ctx.heap(), acc) {
        let recv = regs.read(out_r, ctx);
        if recv == ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase() {
            bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
        }
        recv
    } else {
        acc
    };
    next!(ConstructCheck, ip, regs, ctx, table, roots, float, v)
}

fn read_signed(ip: *const u8, off: usize, stride: usize) -> i32 {
    unsafe {
        if stride == 1 {
            *ip.add(off) as i8 as i32
        } else {
            i16::from_le_bytes([*ip.add(off), *ip.add(off + 1)]) as i32
        }
    }
}

fn read_unsigned(ip: *const u8, off: usize, stride: usize) -> usize {
    unsafe {
        if stride == 1 {
            *ip.add(off) as usize
        } else {
            u16::from_le_bytes([*ip.add(off), *ip.add(off + 1)]) as usize
        }
    }
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cold_box_number<'a>(
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
    // pointer and the resume point from the (updated) cache
    let code = ctx.code_ptr();
    let next = unsafe { cold_next_pc(ip) } as usize - ip as usize;
    let pc = ip as usize - code as usize + next;
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let h = table.get(unsafe { *code.add(pc) });
    become h(unsafe { code.add(pc) }, regs, v, ctx, table, roots, float)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cold_box_add_loc<'a>(
    ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
    let dst = read_signed(ip, base, stride);
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    regs.write(dst, v);
    let code = ctx.code_ptr();
    let next = unsafe { cold_next_pc(ip) } as usize - ip as usize;
    let pc = ip as usize - code as usize + next;
    let h = table.get(unsafe { *code.add(pc) });
    become h(unsafe { code.add(pc) }, regs, v, ctx, table, roots, float)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cold_box_sub_loc<'a>(
    ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
    let dst = read_signed(ip, base, stride);
    let v = unsafe { ctx.heap_mut() }.new_float(float.get());
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    regs.write(dst, v);
    let code = ctx.code_ptr();
    let next = unsafe { cold_next_pc(ip) } as usize - ip as usize;
    let pc = ip as usize - code as usize + next;
    let h = table.get(unsafe { *code.add(pc) });
    become h(unsafe { code.add(pc) }, regs, v, ctx, table, roots, float)
}

/// Outcome of a `Construct` fast-path attempt.
enum ConstructStart {
    /// a constructor frame was pushed (the cache already points at it):
    /// enter it through the trampoline
    Frame,
    /// receiver synthesis threw; the pending exception is set
    Threw(()),
    /// a runtime constructor: call it directly with `new_target` set
    Runtime(usize),
    /// not an ordinary function constructor: fall back to `cold_construct`
    Cold,
}

fn construct_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    regs: Regs,
    callee: i32,
    base: i32,
    count: usize,
    out: i32,
) -> Result<ConstructStart, VmError> {
    let callee_word = regs.read(callee, ctx);
    let kind = match Object::call_target(ctx.heap(), callee_word) {
        // runtime constructors (`new Array`, `new Object`, …) run as a
        // plain tier-1 call with `new_target` set — the callee allocates
        // the instance itself, no frame and no nested execute
        Some(CallTarget::Runtime(idx)) => return Ok(ConstructStart::Runtime(idx)),
        Some(CallTarget::Bytecode { kind, .. }) => kind,
        _ => return Ok(ConstructStart::Cold),
    };
    // ordinary and base-class constructors synthesize the receiver; a
    // derived constructor's `this` is the hole until `super()` binds it
    // and the hole doubles
    // as the ConstructCheck's derived marker
    let derived = matches!(
        kind,
        FunctionKind::DerivedClassConstructor | FunctionKind::DefaultDerivedConstructor
    );
    let receiver = if derived {
        ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase()
    } else {
        if kind != FunctionKind::Normal && kind != FunctionKind::BaseClassConstructor {
            return Ok(ConstructStart::Cold);
        }
        match construct_receiver_fast(ctx, callee_word)? {
            Some(r) => r,
            None => return Ok(ConstructStart::Threw(())),
        }
    };
    // park the receiver in the caller's `out` register: it stays
    // GC-rooted for the whole callee run and the ConstructCheck that
    // follows the callee's return reads it (register-file slots are a
    // fixed-capacity arena, so `regs` survives the allocation above)
    regs.write(out, receiver);
    let callee_word = regs.read(callee, ctx);
    let Some(CallTarget::Bytecode {
        target,
        info,
        context,
        register_count,
        formal_min,
        kind,
        ..
    }) = Object::call_target(ctx.heap(), callee_word)
    else {
        return Ok(ConstructStart::Cold);
    };
    let derived2 = matches!(
        kind,
        FunctionKind::DerivedClassConstructor | FunctionKind::DefaultDerivedConstructor
    );
    if !derived2 && kind != FunctionKind::Normal && kind != FunctionKind::BaseClassConstructor {
        return Ok(ConstructStart::Cold);
    }
    let heap = unsafe { ctx.heap_mut() };
    let meta = ctx.meta(pc + size);
    let frame = ctx.stack().push_construct_frame(
        heap,
        meta,
        pc,
        target.erase(),
        info,
        register_count,
        context.erase(),
        base,
        count,
        callee_word,
        receiver,
        formal_min,
    )?;
    ctx.cache()
        .load(ctx.stack(), frame, unsafe { ctx.heap_mut() });
    Ok(ConstructStart::Frame)
}

#[inline(never)]
fn construct_receiver_fast<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
) -> Result<Option<Tagged<'a, Value>>, VmError> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let callee = scope
            .cast::<Object>(callee)
            .expect("constructible callee is an object");
        // identity-keyed initial-map cache hit: allocate directly
        if let Some(map) = state.construct_initial_map(heap, callee.as_tagged(heap)) {
            let map = scope.handle(map);
            let obj = heap.new_object(&scope, map, HandleSlice::EMPTY);
            return Ok(Some(obj.erase()));
        }
        // miss: full synthesis, then record `[closure, fn_map, proto,
        // proto_slot] -> initial_map` for next time
        let Some(obj) = Object::create_construct_receiver_value(vm, heap, state, callee.erase())?
        else {
            return Ok(None);
        };
        let obj: vm_core::Handle<Object> =
            scope.cast::<Object>(obj).expect("receiver is an object");
        construct_cache_record(heap, state, &scope, callee, obj);
        Ok(Some(obj.as_tagged(heap).erase()))
    })
}

/// Best-effort cache fill: record the synthesized object's initial map
/// keyed on the closure, validated by (function map, `.prototype` slot
/// word) compares. Skips anything unusual (no data `prototype` slot,
/// non-object prototype) — those just stay uncached.
#[cold]
#[inline(never)]
fn construct_cache_record(
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
fn create_closure_cold<'a>(ctx: &Ctx<'a>, info_idx: usize) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
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
fn create_function_context_cold<'a>(
    ctx: &Ctx<'a>,
    scope_idx: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
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
fn proxy_apply_cold<'a>(
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
fn proxy_apply_regs_cold<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    srcs: &[i32],
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let meta = ctx.meta(0);
    let heap = unsafe { ctx.heap_mut() };
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

// -- interpreter intrinsics ----------------------------------------------
//
// Handler-shaped builtins (`Function.prototype.call` / `apply` and
// future trampolines): entered through the call IC exactly like
// bytecode handlers — via a guaranteed tail call — so a call-through
// builtin pushes the callee frame and tail-dispatches into it instead
// of re-entering `execute` through nested Rust frames. The intrinsic
// re-decodes the invoking call opcode's operands from `pc`/`code`, so
// it knows the argument shape (scattered or contiguous window) it was
// invoked with.

/// Continue in the frame the helper just pushed (the cache already
/// switched to the callee). Become-compatible tail entry.
#[inline(always)]
#[rustc_align(32)]
extern "rust-preserve-none" fn enter_fresh_frame<'a>(
    _ip: *const u8,
    _regs: Regs,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let code = ctx.code_ptr();
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let acc = ctx.undefined_word();
    let pc = ctx.cache().pc();
    let ip = unsafe { code.add(pc) };
    let h = table.get(unsafe { *ip });
    become h(ip, regs, acc, ctx, table, roots, float)
}

/// Resolve the (already reshaped) call and push a frame, dispatch a
/// runtime callee, or report an intrinsic/proxy for the caller to tail
/// into. `args[0]` is the receiver of the call being made.
#[cold]
#[inline(never)]
fn intrinsic_call_scattered<'a>(
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
            push_scattered_method_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                srcs,
            )
        }
        Some(CallTarget::Runtime(idx)) => {
            Ok(MethodCall::Value(dispatch_runtime_method(ctx, idx, srcs)))
        }
        Some(CallTarget::Intrinsic(_)) => {
            // a nested intrinsic (`f.call.call(g, x)`): unwrap it against
            // the reshaped window, never against the original operands
            let meta = ctx.meta(0);
            let (saved_top, staged) = ctx.stack().stage_args_regs(ctx.heap(), &meta, srcs)?;
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
fn intrinsic_call_contiguous<'a>(
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
            push_contiguous_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                base,
                count,
            )
        }
        Some(CallTarget::Runtime(idx)) => Ok(MethodCall::Value(dispatch_runtime_contiguous(
            ctx, idx, base, count,
        ))),
        Some(CallTarget::Intrinsic(_)) => {
            // a nested intrinsic (`f.call.call(g, x)`): unwrap it against
            // the reshaped window, never against the original operands
            let meta = ctx.meta(0);
            let staged = ctx.stack().args(&meta, base, count);
            match intrinsic_apply_call(ctx, pc, size, f, staged)? {
                ApplyOut::Frame(frame) => Ok(MethodCall::Frame(frame)),
                ApplyOut::Value(v) => Ok(MethodCall::Value(v)),
            }
        }
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        None => Err(VmError::Type),
    }
}

/// Shared tail of the intrinsics: run the resolved `MethodCall`.
/// `MethodCall::Intrinsic` never appears here — nested intrinsics are
/// unwrapped inside the call helpers against the reshaped window.
macro_rules! finish_intrinsic {
    ($ip:ident, $next:ident, $regs:ident, $acc:ident, $ctx:ident, $t:ident, $k:ident, $f:ident, { $out:expr }, { $proxy:expr }) => {
        match $out {
            Ok(mc) => match mc {
                MethodCall::Frame(_) => {
                    become enter_fresh_frame($ip, $regs, $acc, $ctx, $t, $k, $f)
                }
                MethodCall::Value(v) => {
                    become resume(
                        unsafe { $ctx.code_ptr().add($next as usize - $ip as usize) },
                        $regs,
                        v,
                        $ctx,
                        $t,
                        $k,
                        $f,
                    )
                }
                MethodCall::Proxy => {
                    let v = match $proxy {
                        Ok(v) => v,
                        Err(err) => unsafe { $ctx.raise_tag(err) },
                    };
                    become resume(
                        unsafe { $ctx.code_ptr().add($next as usize - $ip as usize) },
                        $regs,
                        v,
                        $ctx,
                        $t,
                        $k,
                        $f,
                    )
                }
                MethodCall::Intrinsic(i) => {
                    become INTRINSICS[i.id()]($ip, $regs, $acc, $ctx, $t, $k, $f)
                }
            },
            Err(err) => unsafe { $ctx.raise_tag(err) },
        }
    };
}

#[rustc_align(32)]
extern "rust-preserve-none" fn intrinsic_function_call<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let code = ctx.code_ptr();
    let (base, stride) = cold_layout(ip);
    let next = cold_next_pc(ip);
    let op = unsafe { Opcode::from_byte_unchecked(*ip.add(base / 2)) };
    match op {
        // contiguous window `[f, thisArg, args...]`: element 0 rides the
        // window bottom, so dropping it is `count - 1` at the same base
        Opcode::Call | Opcode::CallNoFeedback => {
            let args_base = read_signed(ip, base + stride, stride);
            let count = read_unsigned(ip, base + 2 * stride, stride);
            if count == 0 {
                let _ = unsafe { ctx.raise_tag(VmError::Type) };
                let pc = ip as usize - code as usize;
                become throw_dispatch(
                    unsafe { ctx.code_ptr().add(pc) },
                    regs,
                    acc,
                    ctx,
                    table,
                    roots,
                    float,
                )
            }
            let f = regs.read(args_base - count as i32 + 1, ctx);
            let size = next as usize - ip as usize;
            finish_intrinsic!(
                ip,
                next,
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
                {
                    intrinsic_call_contiguous(
                        ctx,
                        ip as usize - code as usize,
                        size,
                        f,
                        args_base,
                        count - 1,
                    )
                },
                { proxy_apply_cold(ctx, f, args_base, count - 1) }
            )
        }
        // scattered: the receiver register holds the real target
        Opcode::CallMethod0 => {
            let recv = read_signed(ip, base + stride, stride);
            let f = regs.read(recv, ctx);
            let size = next as usize - ip as usize;
            finish_intrinsic!(
                ip,
                next,
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
                { intrinsic_call_scattered(ctx, ip as usize - code as usize, size, f, &[]) },
                { proxy_apply_regs_cold(ctx, f, &[]) }
            )
        }
        Opcode::CallMethod1 => {
            let recv = read_signed(ip, base + stride, stride);
            let arg0 = read_signed(ip, base + 2 * stride, stride);
            let f = regs.read(recv, ctx);
            let size = next as usize - ip as usize;
            finish_intrinsic!(
                ip,
                next,
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
                { intrinsic_call_scattered(ctx, ip as usize - code as usize, size, f, &[arg0]) },
                { proxy_apply_regs_cold(ctx, f, &[arg0]) }
            )
        }
        Opcode::CallMethod2 => {
            let recv = read_signed(ip, base + stride, stride);
            let arg0 = read_signed(ip, base + 2 * stride, stride);
            let arg1 = read_signed(ip, base + 3 * stride, stride);
            let f = regs.read(recv, ctx);
            let size = next as usize - ip as usize;
            finish_intrinsic!(
                ip,
                next,
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
                {
                    intrinsic_call_scattered(
                        ctx,
                        ip as usize - code as usize,
                        size,
                        f,
                        &[arg0, arg1],
                    )
                },
                { proxy_apply_regs_cold(ctx, f, &[arg0, arg1]) }
            )
        }
        // `call` invoked with an undefined receiver (unbound): `this` is
        // not callable
        Opcode::CallFunction0 | Opcode::CallFunction1 | Opcode::CallFunction2 => {
            let _ = unsafe { ctx.raise_tag(VmError::Type) };
            let pc = ip as usize - code as usize;
            become throw_dispatch(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
        _ => {
            let _ = unsafe { ctx.raise_tag(VmError::Type) };
            let pc = ip as usize - code as usize;
            become throw_dispatch(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
    }
}

/// Outcome of a resolved apply: a pushed frame or a completed value.
enum ApplyOut<'a> {
    Frame(FrameMeta),
    Value(Tagged<'a, Value>),
}

/// Resolve a call on `target` with a staged argument window
/// (`args[0]` = receiver) and push the callee frame when it is bytecode.
/// Nested intrinsics (`f.call.apply(...)`) are unwrapped recursively.
#[cold]
#[inline(never)]
fn intrinsic_apply_call<'a>(
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
            let meta = ctx.meta(pc + size);
            let undefined = heap.known().undefined.as_tagged(heap).erase();
            let frame = ctx.stack().push_frame_with_args(
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
            ctx.cache()
                .load(ctx.stack(), frame, unsafe { ctx.heap_mut() });
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

#[rustc_align(32)]
extern "rust-preserve-none" fn intrinsic_function_apply<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let code = ctx.code_ptr();
    let (base, stride) = cold_layout(ip);
    let next = cold_next_pc(ip);
    let op = unsafe { Opcode::from_byte_unchecked(*ip.add(base / 2)) };
    // recover (f, thisArg, argsArray) from the invoking call shape
    let (f, this_arg, array) = match op {
        Opcode::Call | Opcode::CallNoFeedback => {
            let args_base = read_signed(ip, base + stride, stride);
            let count = read_unsigned(ip, base + 2 * stride, stride);
            if count < 2 {
                // window must at least hold [f, thisArg]
                let _ = unsafe { ctx.raise_tag(VmError::Type) };
                let pc = ip as usize - code as usize;
                become throw_dispatch(
                    unsafe { ctx.code_ptr().add(pc) },
                    regs,
                    acc,
                    ctx,
                    table,
                    roots,
                    float,
                )
            }
            let f = regs.read(args_base - count as i32 + 1, ctx);
            let this_arg = regs.read(args_base - count as i32 + 2, ctx);
            let array = if count >= 3 {
                Some(regs.read(args_base - count as i32 + 3, ctx))
            } else {
                None
            };
            (f, this_arg, array)
        }
        Opcode::CallMethod0 => {
            let recv = read_signed(ip, base + stride, stride);
            let f = regs.read(recv, ctx);
            (f, ctx.undefined_word(), None)
        }
        Opcode::CallMethod1 => {
            let recv = read_signed(ip, base + stride, stride);
            let arg0 = read_signed(ip, base + 2 * stride, stride);
            let f = regs.read(recv, ctx);
            (f, regs.read(arg0, ctx), None)
        }
        Opcode::CallMethod2 => {
            let recv = read_signed(ip, base + stride, stride);
            let arg0 = read_signed(ip, base + 2 * stride, stride);
            let arg1 = read_signed(ip, base + 3 * stride, stride);
            let f = regs.read(recv, ctx);
            (f, regs.read(arg0, ctx), Some(regs.read(arg1, ctx)))
        }
        Opcode::CallFunction0 | Opcode::CallFunction1 | Opcode::CallFunction2 => {
            let _ = unsafe { ctx.raise_tag(VmError::Type) };
            let pc = ip as usize - code as usize;
            become throw_dispatch(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
        _ => {
            let _ = unsafe { ctx.raise_tag(VmError::Type) };
            let pc = ip as usize - code as usize;
            become throw_dispatch(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
    };
    let size = next as usize - ip as usize;
    match ctx
        .state()
        .handle_scope(|scope| -> Result<ApplyOut<'a>, VmError> {
            // Safety: fresh register reads, staged before the frame push
            // below can move anything.
            let staged = scope.stage(&spread_apply_args(ctx.heap(), this_arg, array));
            intrinsic_apply_call(ctx, ip as usize - code as usize, size, f, staged)
        }) {
        Ok(ApplyOut::Frame(_)) => become enter_fresh_frame(ip, regs, acc, ctx, table, roots, float),
        Ok(ApplyOut::Value(v)) => {
            become resume(
                unsafe { ctx.code_ptr().add(ip as usize - code as usize) },
                regs,
                v,
                ctx,
                table,
                roots,
                float,
            )
        }
        Err(err) => unsafe { ctx.raise_tag(err) },
    }
}

/// The intrinsic table: builtins entered like bytecode handlers, by id.
static INTRINSICS: [Handler; Intrinsic::COUNT] = [
    intrinsic_function_call as Handler,
    intrinsic_function_apply as Handler,
];

#[inline(always)]
fn cold_layout(ip: *const u8) -> (usize, usize) {
    if unsafe { *ip } == Opcode::Wide as u8 {
        (2, 2)
    } else {
        (1, 1)
    }
}

#[inline(always)]
fn cold_next_pc(ip: *const u8) -> *const u8 {
    let wide = unsafe { *ip } == Opcode::Wide as u8;
    let op = unsafe { *ip.add(wide as usize) } as usize;
    let size = if wide {
        *unsafe { OPERAND_SIZES_WIDE.get_unchecked(op) }
    } else {
        *unsafe { OPERAND_SIZES_NARROW.get_unchecked(op) }
    } as usize;
    unsafe { ip.add(wide as usize + 1 + size) }
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
extern "rust-preserve-none" fn throw_dispatch<'a>(
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
        vm_core::interp::Unwind::Caught(ex) => {
            // `unwind` parked the handler's entry in the cache pc; the
            // walk itself never allocates, so the re-derived pointers
            // are stable
            let handler_pc = ctx.cache().pc();
            let code = ctx.code_ptr();
            let regs = unsafe { Regs::new(ctx.regs_ptr()) };
            let ip = unsafe { code.add(handler_pc) };
            let h = table.get(unsafe { *ip });
            become h(ip, regs, ex, ctx, table, roots, float)
        }
        vm_core::interp::Unwind::Escaped => ctx.exception_word(),
    }
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn call_trampoline<'a>(
    fault_ip: *const u8,
    _regs: Regs,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    // the pushed callee frame is the cache's current frame
    let fault_pc = fault_ip as usize - ctx.code_ptr() as usize;
    let frame = ctx.cache().frame_meta();
    let probe = 0u8;
    if ctx.stack_overflowed() {
        // drop the half-built callee frame and surface the overflow
        // through the sentinel channel
        let caller = ctx.stack().pop_frame(&frame);
        ctx.cache()
            .load(ctx.stack(), caller, unsafe { ctx.heap_mut() });
        let _ = unsafe { ctx.raise_tag(VmError::StackOverflow) };
        let v = ctx.exception_word();
        let code = ctx.code_ptr();
        become resume(
            unsafe { code.add(fault_pc) },
            unsafe { Regs::new(core::ptr::null_mut()) },
            v,
            ctx,
            table,
            roots,
            float,
        );
    }
    // the cache already points at the callee (the push helper loaded it)
    let code = ctx.code_ptr();
    let pc0 = ctx.cache().pc();
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let acc0 = ctx.stack().undefined_word(ctx.heap());
    let callee_ctx = unsafe { Ctx::child(ctx, frame.base) };
    let ip0 = unsafe { code.add(pc0) };
    let op = unsafe { *ip0 } as usize;
    let acc = unsafe { table.get(op as u8)(ip0, regs, acc0, &callee_ctx, table, roots, float) };
    let caller = ctx.stack().pop_frame(&frame);
    ctx.cache()
        .load(ctx.stack(), caller, unsafe { ctx.heap_mut() });
    if ctx.is_throw(acc) {
        become throw_dispatch(
            unsafe { code.add(fault_pc) },
            unsafe { Regs::new(core::ptr::null_mut()) },
            acc,
            ctx,
            table,
            roots,
            float,
        )
    }
    let code = ctx.code_ptr();
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let next = ctx.cache().pc();
    let ip = unsafe { code.add(next) };
    let h = table.get(unsafe { *ip });
    become h(ip, regs, acc, ctx, table, roots, float)
}

#[inline(always)]
#[rustc_align(32)]
extern "rust-preserve-none" fn resume<'a>(
    fault_ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let fault_pc = fault_ip as usize - ctx.code_ptr() as usize;
    if ctx.is_throw(acc) {
        become throw_dispatch(fault_ip, _regs, acc, ctx, table, roots, float)
    }
    let ip = unsafe { cold_next_pc(ctx.code_ptr().add(fault_pc)) };
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let h = table.get(unsafe { *ip });
    become h(ip, regs, acc, ctx, table, roots, float)
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cold_add<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_numeric<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_add_immediate<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_numeric_immediate<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_negate<'a>(
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
fn incdec_cold<'a>(
    ip: *const u8,
    ctx: &Ctx<'a>,
    regs: Regs,
    delta: f64,
) -> Result<Tagged<'a, Value>, VmError> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_inc_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = cold_try!(ctx, incdec_cold(ip, ctx, regs, 1.0));
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
extern "rust-preserve-none" fn cold_dec_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let v = cold_try!(ctx, incdec_cold(ip, ctx, regs, -1.0));
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

fn loc_op_cold<'a>(
    ip: *const u8,
    ctx: &Ctx<'a>,
    regs: Regs,
    sub: bool,
) -> Result<Tagged<'a, Value>, VmError> {
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_add_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, loc_op_cold(ip, ctx, regs, false));
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
extern "rust-preserve-none" fn cold_sub_loc<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, loc_op_cold(ip, ctx, regs, true));
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
extern "rust-preserve-none" fn cold_keyed_load_reg<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_equal<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_less_than<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_greater_than<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_compare_jump<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
    let next = cold_next_pc(ip);
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
extern "rust-preserve-none" fn cold_named_load<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_keyed_load<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_keyed_load_imm<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_keyed_store<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_keyed_store_no_shadow<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_global_load<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_store_named<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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

enum MethodCall<'a> {
    /// the call completed (runtime callee): the value or the exception
    /// sentinel
    Value(Tagged<'a, Value>),
    /// a bytecode frame was pushed and the cache switched to it: enter it
    /// through the machine-call trampoline
    Frame(FrameMeta),
    /// a callable proxy: the caller tail-calls the cold trap dispatch
    Proxy,
    /// an interpreter intrinsic: the caller tail-calls it like a handler
    Intrinsic(vm_core::Intrinsic),
}

#[inline(always)]
fn dispatch_runtime_method<'a>(ctx: &Ctx<'a>, idx: usize, srcs: &[i32]) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let (saved_top, args) = match ctx.stack().stage_args_regs(ctx.heap(), &meta, srcs) {
        Ok(staged) => staged,
        Err(err) => {
            let _ = unsafe { ctx.raise(err) };
            return ctx.exception_word();
        }
    };
    let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
    let v = f(nctx, args);
    ctx.stack().set_top(saved_top);
    v
}

#[inline(always)]
fn dispatch_runtime_function<'a>(ctx: &Ctx<'a>, idx: usize, args: &[i32]) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let (saved_top, staged) = match ctx.stack().stage_function_args(ctx.heap(), &meta, args) {
        Ok(staged) => staged,
        Err(err) => {
            let _ = unsafe { ctx.raise(err) };
            return ctx.exception_word();
        }
    };
    let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
    let v = f(nctx, staged);
    ctx.stack().set_top(saved_top);
    v
}

#[inline(always)]
fn dispatch_runtime_construct<'a>(
    ctx: &Ctx<'a>,
    idx: usize,
    callee_reg: i32,
    base: i32,
    count: usize,
) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let args = ctx.stack().args(&meta, base, count);
    let (saved_top, staged) = match ctx.stack().stage_construct_args(ctx.heap(), args) {
        Ok(staged) => staged,
        Err(err) => {
            let _ = unsafe { ctx.raise(err) };
            return ctx.exception_word();
        }
    };
    // Safety: fresh register word, no allocation since the read.
    let callee = ctx.stack().reg(ctx.heap(), &meta, callee_reg);
    let raw = ctx.state().handle_scope(|scope| -> Value {
        let new_target = scope.handle(callee);
        let nctx = RuntimeContext::with_new_target(
            ctx.vm(),
            unsafe { ctx.heap_mut() },
            ctx.state(),
            Some(new_target),
        );
        f(nctx, staged).raw()
    });
    ctx.stack().set_top(saved_top);
    // Safety: fresh result word, consumed before any allocation.
    unsafe { Tagged::<Value>::from_value_unchecked(raw) }
}

fn dispatch_runtime_contiguous<'a>(
    ctx: &Ctx<'a>,
    idx: usize,
    base: i32,
    count: usize,
) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let args = ctx.stack().args(&meta, base, count);
    let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
    f(nctx, args)
}

#[inline(always)]
fn push_scattered_method_frame<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    target: Tagged<'_, Object>,
    info: Tagged<'_, CallableInfoObject>,
    context: Tagged<'_, Context>,
    register_count: usize,
    formal_min: usize,
    srcs: &[i32],
) -> Result<MethodCall<'a>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
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
    ctx.cache()
        .load(ctx.stack(), frame, unsafe { ctx.heap_mut() });
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
fn push_function_frame<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    target: Tagged<'_, Object>,
    info: Tagged<'_, CallableInfoObject>,
    context: Tagged<'_, Context>,
    register_count: usize,
    formal_min: usize,
    args: &[i32],
) -> Result<MethodCall<'a>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
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
    ctx.cache()
        .load(ctx.stack(), frame, unsafe { ctx.heap_mut() });
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
fn push_contiguous_frame<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    target: Tagged<'_, Object>,
    info: Tagged<'_, CallableInfoObject>,
    context: Tagged<'_, Context>,
    register_count: usize,
    formal_min: usize,
    base: i32,
    count: usize,
) -> Result<MethodCall<'a>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let meta = ctx.meta(pc + size);
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
    ctx.cache()
        .load(ctx.stack(), frame, unsafe { ctx.heap_mut() });
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
fn call_method_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    srcs: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match unsafe {
        vm_core::ic::call_probe(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            callee_word,
        )
    } {
        vm_core::ic::CallProbe::Bytecode(vm_core::ic::CallHit {
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
            push_scattered_method_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                srcs,
            )
        }
        vm_core::ic::CallProbe::Runtime(idx) => {
            Ok(MethodCall::Value(dispatch_runtime_method(ctx, idx, srcs)))
        }
        vm_core::ic::CallProbe::Intrinsic(i) => Ok(MethodCall::Intrinsic(i)),
        vm_core::ic::CallProbe::Miss => cold_call_method_miss(ctx, pc, size, callee_word, srcs, fb),
    }
}

#[cold]
#[inline(never)]
fn cold_call_method_miss<'a>(
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
                    ctx.cache().feedback_ref(ctx.heap()),
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
                    ctx.cache().feedback_ref(ctx.heap()),
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
                    ctx.cache().feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    Some(info),
                )
            };
            push_scattered_method_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                srcs,
            )
        }
    }
}

#[inline(always)]
fn call_function_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    args: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match unsafe {
        vm_core::ic::call_probe(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            callee_word,
        )
    } {
        vm_core::ic::CallProbe::Bytecode(vm_core::ic::CallHit {
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
            push_function_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                args,
            )
        }
        vm_core::ic::CallProbe::Runtime(idx) => {
            Ok(MethodCall::Value(dispatch_runtime_function(ctx, idx, args)))
        }
        vm_core::ic::CallProbe::Intrinsic(i) => Ok(MethodCall::Intrinsic(i)),
        vm_core::ic::CallProbe::Miss => {
            cold_call_function_miss(ctx, pc, size, callee_word, args, fb)
        }
    }
}

#[cold]
#[inline(never)]
fn cold_call_function_miss<'a>(
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
                    ctx.cache().feedback_ref(ctx.heap()),
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
                    ctx.cache().feedback_ref(ctx.heap()),
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
                    ctx.cache().feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    Some(info),
                )
            };
            push_function_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                args,
            )
        }
    }
}

/// `Call` (contiguous `[receiver, args...]` window) with the same call IC.
#[inline(always)]
fn call_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    base: i32,
    count: usize,
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match unsafe {
        vm_core::ic::call_probe(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            callee_word,
        )
    } {
        vm_core::ic::CallProbe::Bytecode(vm_core::ic::CallHit {
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
            push_contiguous_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                base,
                count,
            )
        }
        vm_core::ic::CallProbe::Runtime(idx) => Ok(MethodCall::Value(dispatch_runtime_contiguous(
            ctx, idx, base, count,
        ))),
        vm_core::ic::CallProbe::Intrinsic(i) => Ok(MethodCall::Intrinsic(i)),
        vm_core::ic::CallProbe::Miss => cold_call_miss(ctx, pc, size, callee_word, base, count, fb),
    }
}

#[cold]
#[inline(never)]
fn cold_call_miss<'a>(
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
                    ctx.cache().feedback_ref(ctx.heap()),
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
                    ctx.cache().feedback_ref(ctx.heap()),
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
                    ctx.cache().feedback_ref(ctx.heap()),
                    fb,
                    callee_word,
                    Some(info),
                )
            };
            push_contiguous_frame(
                ctx,
                pc,
                size,
                target,
                info,
                context,
                register_count,
                formal_min,
                base,
                count,
            )
        }
    }
}

#[cold]
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn cold_construct<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_create_closure<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
    let info_idx = read_unsigned(ip, base, stride);
    let v = cold_try!(ctx, create_closure_cold(ctx, info_idx));
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
extern "rust-preserve-none" fn cold_create_empty_array<'a>(
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
extern "rust-preserve-none" fn cold_create_empty_object<'a>(
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
extern "rust-preserve-none" fn cold_create_bare_object<'a>(
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
fn create_block_context_cold<'a>(
    ctx: &Ctx<'a>,
    count: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let meta = ctx.meta(0);
        let outer = scope
            .cast::<Context>(ctx.stack().context_slot(&meta).get(heap))
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
extern "rust-preserve-none" fn cold_create_block_context<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
    let count = read_unsigned(ip, base, stride);
    let v = cold_try!(ctx, create_block_context_cold(ctx, count));
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
extern "rust-preserve-none" fn cold_less_than_or_equal<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_global_load_nothrow<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_store_global<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
extern "rust-preserve-none" fn cold_store_named_no_shadow<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
fn instance_of_cold<'a>(
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
extern "rust-preserve-none" fn cold_instance_of<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
    let callable = regs.read(read_signed(ip, base, stride), ctx);
    let v = cold_try!(ctx, instance_of_cold(ctx, acc, callable));
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
extern "rust-preserve-none" fn cold_greater_than_or_equal<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
fn add_parent_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let receiver = scope.handle(recv);
        let name = scope.handle(
            ctx.cache()
                .constants_ref(heap)
                .at(heap, name_idx)
                .erase()
                .as_name(),
        );
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
extern "rust-preserve-none" fn cold_add_parent<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
    let recv = regs.read(read_signed(ip, base, stride), ctx);
    let name_idx = read_unsigned(ip, base + stride, stride);
    let v = cold_try!(ctx, add_parent_cold(ctx, recv, name_idx, acc));
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
extern "rust-preserve-none" fn cold_create_function_context<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
    let scope_idx = read_unsigned(ip, base, stride);
    let v = cold_try!(ctx, create_function_context_cold(ctx, scope_idx));
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
extern "rust-preserve-none" fn cold_proxy_apply<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
    let callee = regs.read(read_signed(ip, base, stride), ctx);
    let args_base = read_signed(ip, base + stride, stride);
    let count = read_unsigned(ip, base + 2 * stride, stride);
    let v = cold_try!(ctx, proxy_apply_cold(ctx, callee, args_base, count));
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
extern "rust-preserve-none" fn cold_call_method_proxy<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
    let v = cold_try!(ctx, proxy_apply_regs_cold(ctx, callee, &srcs[..argc + 1]));
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
fn proxy_apply_function_cold<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args: &[i32],
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let state = ctx.state();
    let meta = ctx.meta(0);
    let heap = unsafe { ctx.heap_mut() };
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
#[rustc_align(32)]
extern "rust-preserve-none" fn cold_call_function_proxy<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let pc = ip as usize - ctx.code_ptr() as usize;
    let (base, stride) = cold_layout(ip);
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
    let v = cold_try!(ctx, proxy_apply_function_cold(ctx, callee, &args[..argc]));
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

const fn table_narrow() -> [Handler; 256] {
    let mut t = [op_trap as Handler; 256];
    t[Opcode::Load as usize] = op_load::<1> as Handler;
    t[Opcode::Move as usize] = op_move::<1> as Handler;
    t[Opcode::Store as usize] = op_store::<1> as Handler;
    t[Opcode::LoadSmi as usize] = op_load_smi::<1> as Handler;
    t[Opcode::LoadConstant as usize] = op_load_constant::<1> as Handler;
    t[Opcode::LoadZero as usize] = op_load_zero::<1> as Handler;
    t[Opcode::LoadUndefined as usize] = op_load_undefined::<1> as Handler;
    t[Opcode::LoadNull as usize] = op_load_null::<1> as Handler;
    t[Opcode::LoadTrue as usize] = op_load_true::<1> as Handler;
    t[Opcode::LoadFalse as usize] = op_load_false::<1> as Handler;
    t[Opcode::Add as usize] = op_add::<1> as Handler;
    t[Opcode::Sub as usize] = op_sub::<1> as Handler;
    t[Opcode::Mul as usize] = op_mul::<1> as Handler;
    t[Opcode::Div as usize] = op_div::<1> as Handler;
    t[Opcode::Mod as usize] = op_mod::<1> as Handler;
    t[Opcode::Exp as usize] = op_exp::<1> as Handler;
    t[Opcode::BitwiseOr as usize] = op_bitwise_or::<1> as Handler;
    t[Opcode::BitwiseXor as usize] = op_bitwise_xor::<1> as Handler;
    t[Opcode::BitwiseAnd as usize] = op_bitwise_and::<1> as Handler;
    t[Opcode::ShiftLeft as usize] = op_shift_left::<1> as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right::<1> as Handler;
    t[Opcode::ShiftRightLogical as usize] = op_shift_right_logical::<1> as Handler;
    t[Opcode::AddImmediate as usize] = op_add_immediate::<1> as Handler;
    t[Opcode::SubImmediate as usize] = op_sub_immediate::<1> as Handler;
    t[Opcode::MulImmediate as usize] = op_mul_immediate::<1> as Handler;
    t[Opcode::DivImmediate as usize] = op_div_immediate::<1> as Handler;
    t[Opcode::ModImmediate as usize] = op_mod_immediate::<1> as Handler;
    t[Opcode::ExpImmediate as usize] = op_exp_immediate::<1> as Handler;
    t[Opcode::BitwiseOrImmediate as usize] = op_bitwise_or_immediate::<1> as Handler;
    t[Opcode::BitwiseXorImmediate as usize] = op_bitwise_xor_immediate::<1> as Handler;
    t[Opcode::BitwiseAndImmediate as usize] = op_bitwise_and_immediate::<1> as Handler;
    t[Opcode::ShiftLeftImmediate as usize] = op_shift_left_immediate::<1> as Handler;
    t[Opcode::ShiftRightImmediate as usize] = op_shift_right_immediate::<1> as Handler;
    t[Opcode::ShiftRightLogicalImmediate as usize] =
        op_shift_right_logical_immediate::<1> as Handler;
    t[Opcode::IncLoc as usize] = op_inc_loc::<1> as Handler;
    t[Opcode::DecLoc as usize] = op_dec_loc::<1> as Handler;
    t[Opcode::AddLoc as usize] = op_add_loc::<1> as Handler;
    t[Opcode::SubLoc as usize] = op_sub_loc::<1> as Handler;
    t[Opcode::LoadKeyedPropertyReg as usize] = op_load_keyed_reg::<1> as Handler;
    t[Opcode::Equal as usize] = op_equal::<1> as Handler;
    t[Opcode::LessThan as usize] = op_less_than::<1> as Handler;
    t[Opcode::GreaterThan as usize] = op_greater_than::<1> as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed::<1> as Handler;
    t[Opcode::StoreKeyedPropertyNoShadow as usize] = op_store_keyed_no_shadow::<1> as Handler;
    t[Opcode::Negate as usize] = op_negate::<1> as Handler;
    t[Opcode::CompareJump as usize] = op_compare_jump::<1> as Handler;
    t[Opcode::Jump as usize] = op_jump::<1> as Handler;
    t[Opcode::JumpIfTruthy as usize] = op_jump_if_truthy::<1> as Handler;
    t[Opcode::JumpIfFalsy as usize] = op_jump_if_falsy::<1> as Handler;
    t[Opcode::JumpLoop as usize] = op_jump_loop::<1> as Handler;
    t[Opcode::Throw as usize] = op_throw::<1> as Handler;
    t[Opcode::ReThrow as usize] = op_rethrow::<1> as Handler;
    t[Opcode::Return as usize] = op_return::<1> as Handler;
    t[Opcode::LoadNamedProperty as usize] = op_load_named::<1> as Handler;
    t[Opcode::LoadKeyedProperty as usize] = op_load_keyed::<1> as Handler;
    t[Opcode::LoadElementImm as usize] = op_load_element_imm::<1> as Handler;
    t[Opcode::LoadGlobal as usize] = op_load_global::<1> as Handler;
    t[Opcode::StoreNamedProperty as usize] = op_store_named::<1> as Handler;
    t[Opcode::LoadContextSlot as usize] = op_load_context_slot::<1> as Handler;
    t[Opcode::StoreContextSlot as usize] = op_store_context_slot::<1> as Handler;
    t[Opcode::PushContext as usize] = op_push_context::<1> as Handler;
    t[Opcode::PopContext as usize] = op_pop_context::<1> as Handler;
    t[Opcode::CreateFunctionContext as usize] = op_create_function_context::<1> as Handler;
    t[Opcode::CreateClosure as usize] = op_create_closure::<1> as Handler;
    t[Opcode::CreateEmptyArrayLiteral as usize] = op_create_empty_array::<1> as Handler;
    t[Opcode::CreateEmptyObjectLiteral as usize] = op_create_empty_object::<1> as Handler;
    t[Opcode::CallRuntime as usize] = op_call_runtime::<1> as Handler;
    t[Opcode::Call as usize] = op_call_ic::<1> as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call::<1> as Handler;
    t[Opcode::CallMethod0 as usize] = op_call_method0::<1> as Handler;
    t[Opcode::CallFunction0 as usize] = op_call_function0::<1> as Handler;
    t[Opcode::CallFunction1 as usize] = op_call_function1::<1> as Handler;
    t[Opcode::CallFunction2 as usize] = op_call_function2::<1> as Handler;
    t[Opcode::CallMethod1 as usize] = op_call_method1::<1> as Handler;
    t[Opcode::CallMethod2 as usize] = op_call_method2::<1> as Handler;
    t[Opcode::Construct as usize] = op_construct::<1> as Handler;
    t[Opcode::ConstructCheck as usize] = op_construct_check::<1> as Handler;
    t[Opcode::LoadHole as usize] = op_load_hole::<1> as Handler;
    t[Opcode::LoadNewTarget as usize] = op_load_new_target::<1> as Handler;
    t[Opcode::LoadContext as usize] = op_load_context::<1> as Handler;
    t[Opcode::ThrowReferenceErrorIfHole as usize] =
        op_throw_reference_error_if_hole::<1> as Handler;
    t[Opcode::TestReferenceEqual as usize] = op_test_reference_equal::<1> as Handler;
    t[Opcode::TestTypeof as usize] = op_test_typeof::<1> as Handler;
    t[Opcode::EqualStrict as usize] = op_equal_strict::<1> as Handler;
    t[Opcode::JumpIfNotUndefined as usize] = op_jump_if_not_undefined::<1> as Handler;
    t[Opcode::LessThanOrEqual as usize] = op_less_than_or_equal::<1> as Handler;
    t[Opcode::CreateBareObjectLiteral as usize] = op_create_bare_object::<1> as Handler;
    t[Opcode::CreateBlockContext as usize] = op_create_block_context::<1> as Handler;
    t[Opcode::LoadGlobalFast as usize] = op_load_global_fast::<1> as Handler;
    t[Opcode::LoadGlobalNoThrow as usize] = op_load_global_nothrow::<1> as Handler;
    t[Opcode::StoreGlobal as usize] = op_store_global::<1> as Handler;
    t[Opcode::StoreNamedPropertyNoShadow as usize] = op_store_named_no_shadow::<1> as Handler;
    t[Opcode::InstanceOf as usize] = op_instance_of::<1> as Handler;
    t[Opcode::LoadCurrentClosure as usize] = op_load_current_closure::<1> as Handler;
    t[Opcode::GreaterThanOrEqual as usize] = op_greater_than_or_equal::<1> as Handler;
    t[Opcode::AddParent as usize] = op_add_parent::<1> as Handler;
    t[Opcode::LoadNamedPropertyFast as usize] = op_load_named_fast::<1> as Handler;
    t[Opcode::StoreNamedPropertyNoShadowFast as usize] =
        op_store_named_no_shadow_fast::<1> as Handler;
    t[Opcode::Wide as usize] = op_wide as Handler;
    t
}

const fn table_wide() -> [Handler; 256] {
    let mut t = [op_trap as Handler; 256];
    t[Opcode::Load as usize] = op_load::<2> as Handler;
    t[Opcode::Move as usize] = op_move::<2> as Handler;
    t[Opcode::Store as usize] = op_store::<2> as Handler;
    t[Opcode::LoadSmi as usize] = op_load_smi::<2> as Handler;
    t[Opcode::LoadConstant as usize] = op_load_constant::<2> as Handler;
    t[Opcode::LoadZero as usize] = op_load_zero::<2> as Handler;
    t[Opcode::LoadUndefined as usize] = op_load_undefined::<2> as Handler;
    t[Opcode::LoadNull as usize] = op_load_null::<2> as Handler;
    t[Opcode::LoadTrue as usize] = op_load_true::<2> as Handler;
    t[Opcode::LoadFalse as usize] = op_load_false::<2> as Handler;
    t[Opcode::Add as usize] = op_add::<2> as Handler;
    t[Opcode::Sub as usize] = op_sub::<2> as Handler;
    t[Opcode::Mul as usize] = op_mul::<2> as Handler;
    t[Opcode::Div as usize] = op_div::<2> as Handler;
    t[Opcode::Mod as usize] = op_mod::<2> as Handler;
    t[Opcode::Exp as usize] = op_exp::<2> as Handler;
    t[Opcode::BitwiseOr as usize] = op_bitwise_or::<2> as Handler;
    t[Opcode::BitwiseXor as usize] = op_bitwise_xor::<2> as Handler;
    t[Opcode::BitwiseAnd as usize] = op_bitwise_and::<2> as Handler;
    t[Opcode::ShiftLeft as usize] = op_shift_left::<2> as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right::<2> as Handler;
    t[Opcode::ShiftRightLogical as usize] = op_shift_right_logical::<2> as Handler;
    t[Opcode::AddImmediate as usize] = op_add_immediate::<2> as Handler;
    t[Opcode::SubImmediate as usize] = op_sub_immediate::<2> as Handler;
    t[Opcode::MulImmediate as usize] = op_mul_immediate::<2> as Handler;
    t[Opcode::DivImmediate as usize] = op_div_immediate::<2> as Handler;
    t[Opcode::ModImmediate as usize] = op_mod_immediate::<2> as Handler;
    t[Opcode::ExpImmediate as usize] = op_exp_immediate::<2> as Handler;
    t[Opcode::BitwiseOrImmediate as usize] = op_bitwise_or_immediate::<2> as Handler;
    t[Opcode::BitwiseXorImmediate as usize] = op_bitwise_xor_immediate::<2> as Handler;
    t[Opcode::BitwiseAndImmediate as usize] = op_bitwise_and_immediate::<2> as Handler;
    t[Opcode::ShiftLeftImmediate as usize] = op_shift_left_immediate::<2> as Handler;
    t[Opcode::ShiftRightImmediate as usize] = op_shift_right_immediate::<2> as Handler;
    t[Opcode::ShiftRightLogicalImmediate as usize] =
        op_shift_right_logical_immediate::<2> as Handler;
    t[Opcode::IncLoc as usize] = op_inc_loc::<2> as Handler;
    t[Opcode::DecLoc as usize] = op_dec_loc::<2> as Handler;
    t[Opcode::AddLoc as usize] = op_add_loc::<2> as Handler;
    t[Opcode::SubLoc as usize] = op_sub_loc::<2> as Handler;
    t[Opcode::LoadKeyedPropertyReg as usize] = op_load_keyed_reg::<2> as Handler;
    t[Opcode::Equal as usize] = op_equal::<2> as Handler;
    t[Opcode::LessThan as usize] = op_less_than::<2> as Handler;
    t[Opcode::GreaterThan as usize] = op_greater_than::<2> as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed::<2> as Handler;
    t[Opcode::StoreKeyedPropertyNoShadow as usize] = op_store_keyed_no_shadow::<2> as Handler;
    t[Opcode::Negate as usize] = op_negate::<2> as Handler;
    t[Opcode::CompareJump as usize] = op_compare_jump::<2> as Handler;
    t[Opcode::Jump as usize] = op_jump::<2> as Handler;
    t[Opcode::JumpIfTruthy as usize] = op_jump_if_truthy::<2> as Handler;
    t[Opcode::JumpIfFalsy as usize] = op_jump_if_falsy::<2> as Handler;
    t[Opcode::JumpLoop as usize] = op_jump_loop::<2> as Handler;
    t[Opcode::Throw as usize] = op_throw::<2> as Handler;
    t[Opcode::ReThrow as usize] = op_rethrow::<2> as Handler;
    t[Opcode::Return as usize] = op_return::<2> as Handler;
    t[Opcode::LoadNamedProperty as usize] = op_load_named::<2> as Handler;
    t[Opcode::LoadKeyedProperty as usize] = op_load_keyed::<2> as Handler;
    t[Opcode::LoadElementImm as usize] = op_load_element_imm::<2> as Handler;
    t[Opcode::LoadGlobal as usize] = op_load_global::<2> as Handler;
    t[Opcode::StoreNamedProperty as usize] = op_store_named::<2> as Handler;
    t[Opcode::LoadContextSlot as usize] = op_load_context_slot::<2> as Handler;
    t[Opcode::StoreContextSlot as usize] = op_store_context_slot::<2> as Handler;
    t[Opcode::PushContext as usize] = op_push_context::<2> as Handler;
    t[Opcode::PopContext as usize] = op_pop_context::<2> as Handler;
    t[Opcode::CreateFunctionContext as usize] = op_create_function_context::<2> as Handler;
    t[Opcode::CreateClosure as usize] = op_create_closure::<2> as Handler;
    t[Opcode::CreateEmptyArrayLiteral as usize] = op_create_empty_array::<2> as Handler;
    t[Opcode::CreateEmptyObjectLiteral as usize] = op_create_empty_object::<2> as Handler;
    t[Opcode::CallRuntime as usize] = op_call_runtime::<2> as Handler;
    t[Opcode::Call as usize] = op_call_ic::<2> as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call::<2> as Handler;
    t[Opcode::CallMethod0 as usize] = op_call_method0::<2> as Handler;
    t[Opcode::CallFunction0 as usize] = op_call_function0::<2> as Handler;
    t[Opcode::CallFunction1 as usize] = op_call_function1::<2> as Handler;
    t[Opcode::CallFunction2 as usize] = op_call_function2::<2> as Handler;
    t[Opcode::CallMethod1 as usize] = op_call_method1::<2> as Handler;
    t[Opcode::CallMethod2 as usize] = op_call_method2::<2> as Handler;
    t[Opcode::Construct as usize] = op_construct::<2> as Handler;
    t[Opcode::ConstructCheck as usize] = op_construct_check::<2> as Handler;
    t[Opcode::LoadHole as usize] = op_load_hole::<2> as Handler;
    t[Opcode::LoadNewTarget as usize] = op_load_new_target::<2> as Handler;
    t[Opcode::LoadContext as usize] = op_load_context::<2> as Handler;
    t[Opcode::ThrowReferenceErrorIfHole as usize] =
        op_throw_reference_error_if_hole::<2> as Handler;
    t[Opcode::TestReferenceEqual as usize] = op_test_reference_equal::<2> as Handler;
    t[Opcode::TestTypeof as usize] = op_test_typeof::<2> as Handler;
    t[Opcode::EqualStrict as usize] = op_equal_strict::<2> as Handler;
    t[Opcode::JumpIfNotUndefined as usize] = op_jump_if_not_undefined::<2> as Handler;
    t[Opcode::LessThanOrEqual as usize] = op_less_than_or_equal::<2> as Handler;
    t[Opcode::CreateBareObjectLiteral as usize] = op_create_bare_object::<2> as Handler;
    t[Opcode::CreateBlockContext as usize] = op_create_block_context::<2> as Handler;
    t[Opcode::LoadGlobalFast as usize] = op_load_global_fast::<2> as Handler;
    t[Opcode::LoadGlobalNoThrow as usize] = op_load_global_nothrow::<2> as Handler;
    t[Opcode::StoreGlobal as usize] = op_store_global::<2> as Handler;
    t[Opcode::StoreNamedPropertyNoShadow as usize] = op_store_named_no_shadow::<2> as Handler;
    t[Opcode::InstanceOf as usize] = op_instance_of::<2> as Handler;
    t[Opcode::LoadCurrentClosure as usize] = op_load_current_closure::<2> as Handler;
    t[Opcode::GreaterThanOrEqual as usize] = op_greater_than_or_equal::<2> as Handler;
    t[Opcode::AddParent as usize] = op_add_parent::<2> as Handler;
    t[Opcode::LoadNamedPropertyFast as usize] = op_load_named_fast::<2> as Handler;
    t[Opcode::StoreNamedPropertyNoShadowFast as usize] =
        op_store_named_no_shadow_fast::<2> as Handler;
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
        Some(CallTarget::Intrinsic(intrinsic)) => {
            // Rust-world entry of an intrinsic: unwrap it and re-enter.
            // This is the cold path — bytecode call sites reach intrinsics
            // through the handler-shaped table instead.
            state.handle_scope(|scope| match intrinsic {
                Intrinsic::FunctionCall => {
                    let f = args
                        .get(0)
                        .map(|h| h.as_tagged(heap))
                        .ok_or(VmError::Arity)?;
                    if !Object::is_callable(heap, f) {
                        return Err(VmError::Type);
                    }
                    let f = scope.cast::<Object>(f).ok_or(VmError::Type)?;
                    enter(vm, heap, state, f, args.slice_from(1), None)
                }
                Intrinsic::FunctionApply => {
                    let f = args
                        .get(0)
                        .map(|h| h.as_tagged(heap))
                        .ok_or(VmError::Arity)?;
                    if !Object::is_callable(heap, f) {
                        return Err(VmError::Type);
                    }
                    let this_arg = args
                        .get(1)
                        .map(|h| h.as_tagged(heap))
                        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
                    let array = args.get(2).map(|h| h.as_tagged(heap));
                    // Safety: fresh rooted-slot reads, staged before `enter`
                    // below can allocate.
                    let staged = scope.stage(&unsafe { spread_apply_args(heap, this_arg, array) });
                    let f = scope.cast::<Object>(f).ok_or(VmError::Type)?;
                    enter(vm, heap, state, f, staged, None)
                }
            })
        }
        Some(CallTarget::Runtime(idx)) => {
            let f = vm.runtime(RuntimeIndex(idx));
            let (saved_top, fargs) = state.stack().stage_args(heap, args)?;
            let nctx = RuntimeContext::with_new_target(vm, heap, state, new_target);
            let v = f(nctx, fargs);
            state.stack().set_top(saved_top);
            Ok(v)
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

            let probe = 0u8;
            let stack_limit = (&probe as *const u8 as usize).saturating_sub(6 * 1024 * 1024);
            let ctx: Ctx<'a> = unsafe { Ctx::new(vm, heap, state, frame.base, stack_limit) };
            let base = ctx.code_ptr();
            let pc = cache.pc();
            let regs = unsafe { Regs::new(ctx.regs_ptr()) };
            let acc = cache.acc(ctx.heap());
            let table = TableArg::new(&TABLE_NARROW);
            let roots = RootsArg::new(ctx.heap().known());
            let ip = unsafe { base.add(pc) };
            Ok(unsafe {
                table.get(*ip)(ip, regs, acc, &ctx, table, roots, FloatReg::new(f64::NAN))
            })
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
