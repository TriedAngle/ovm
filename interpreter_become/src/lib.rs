#![feature(explicit_tail_calls)]
#![feature(rust_preserve_none_cc)]
#![allow(incomplete_features)]
#![feature(fn_align)]
#![allow(unused_macros, unused_unsafe, unused_variables)]

use bytecode::{OPERAND_SIZES_NARROW, OPERAND_SIZES_WIDE, Opcode};
use vm_core::ic::{CallHit, CallProbe, ElementHit, Hit, InlineCache, MonoProbe};
use vm_core::proxy::Proxy;
use vm_core::{
    Args, CallTarget, Callee, Coercion, Compare, Context, ContextState, Convert, Ctx, ExecuteFn,
    FixedArray, FrameMeta, FunctionKind, Handle, HandleSlice, Heap, Interpreter, Object, Recv,
    Register, RuntimeContext, RuntimeIndex, Smi, Tagged, VM, Value, VmError,
};

pub struct BecomeInterpreter;

/// The frame's register window: signed indices are anchor-relative
/// slot offsets (negative = locals below the frame header, positive =
/// parameters above it, receiver = 0).
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
    pub fn from_heap(_heap: &Heap) -> RootsArg<'a> {
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
    pub fn from_heap(heap: &Heap) -> RootsArg<'a> {
        RootsArg(heap.known(), PhantomData)
    }

    #[inline(always)]
    pub fn known<'h>(&self, _heap: &'h Heap) -> &'static WellKnown {
        self.0
    }
}

use core::hint::black_box;
use std::marker::PhantomData;

use vm_core::bootstrap::WellKnown;

mod slow;

use slow::*;

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

    /// Build a cursor for a handler that serves several opcodes (or is
    /// entered generically): the opcode is read from the instruction
    /// rather than asserted against a literal, keeping the operand-kind
    /// debug checks.
    #[inline(always)]
    fn from_ip(ip: *const u8) -> Ops<STRIDE> {
        Ops {
            ip,
            #[cfg(debug_assertions)]
            op: unsafe { Opcode::from_byte_unchecked(*ip.add(if STRIDE == 2 { 1 } else { 0 })) },
        }
    }

    /// The instruction's opcode (skipping the `Wide` prefix).
    #[inline(always)]
    fn op(&self) -> Opcode {
        unsafe { Opcode::from_byte_unchecked(*self.ip.add(if STRIDE == 2 { 1 } else { 0 })) }
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

macro_rules! slow_start {
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
    let v = ctx.constants_ref(ctx.heap()).at(ctx.heap(), idx);
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(sum) {
            next!(Add, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_number::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(sum))
    }
    become slow_add::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(diff) {
            next!(Sub, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_number::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(diff))
    }
    become slow_numeric::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(product) {
            next!(Mul, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_number::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(product))
    }
    become slow_numeric::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_add_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    if let Some(n) = Convert::as_number(v) {
        let new = n + 1.0;
        if let Some(s) = Smi::from_f64(new) {
            regs.write(r, s.into_tagged());
            next!(IncLoc, ip, regs, ctx, table, roots, float, v)
        }
        if let Some(t) = unsafe { ctx.heap_mut() }.try_new_float(new) {
            regs.write(r, t);
            next!(IncLoc, ip, regs, ctx, table, roots, float, v)
        }
    }
    become slow_inc_loc::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    if let Some(n) = Convert::as_number(v) {
        let new = n - 1.0;
        if let Some(s) = Smi::from_f64(new) {
            regs.write(r, s.into_tagged());
            next!(DecLoc, ip, regs, ctx, table, roots, float, v)
        }
        if let Some(t) = unsafe { ctx.heap_mut() }.try_new_float(new) {
            regs.write(r, t);
            next!(DecLoc, ip, regs, ctx, table, roots, float, v)
        }
    }
    become slow_dec_loc::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_equal::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_less_than::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_greater_than::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    if ctx.frame_base() == ctx.base_anchor() {
        return acc;
    }
    let caller = ctx.stack().pop_frame(ctx.frame_base());
    ctx.set_frame_base(caller.base);
    let base = ctx.code_ptr();
    dispatch!(
        unsafe { base.add(caller.pc) },
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
        become slow_bitwise::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        become slow_bitwise_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(quotient) {
            next!(Div, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_number::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(quotient))
    }
    become slow_numeric::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_numeric_immediate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(neg) {
            next!(Negate, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_number::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(neg))
    }
    become slow_negate::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_compare_jump::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(sum) {
            regs.write(dst, v);
            next!(AddLoc, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_add_loc::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(sum))
    }
    become slow_add_loc::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        if let Some(v) = unsafe { ctx.heap_mut() }.try_new_float(diff) {
            regs.write(dst, v);
            next!(SubLoc, ip, regs, ctx, table, roots, float, v)
        }
        become slow_box_sub_loc::<STRIDE>(ip, regs, acc, ctx, table, roots, FloatReg::new(diff))
    }
    become slow_sub_loc::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        ctx.acc_slot().store(acc);
        if unsafe { ctx.heap_mut() }.safepoint_poll() {
            let state = ctx.state();
            state.set_termination(vm_core::Termination::Shutdown);
            let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap());
            state.set_pending_exception(undefined);
            threw!(acc, ip, regs, ctx, table, roots, float)
        }
        let target = ip.wrapping_offset(off as isize);
        let pc = target as usize - ctx.code_ptr() as usize;
        let acc = ctx.acc_slot().get(ctx.heap());
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
            ctx.feedback_ref(ctx.heap()),
            fb,
            recv,
            idx as usize,
        ) {
            next!(LoadKeyedPropertyReg, ip, regs, ctx, table, roots, float, v)
        }
    }
    become slow_keyed_load_reg::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
            ctx.feedback_ref(ctx.heap()),
            fb,
            recv_w,
            idx as usize,
            acc,
        ) {
            next!(StoreKeyedProperty, ip, regs, ctx, table, roots, float, v)
        }
    }
    become slow_keyed_store::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
            ctx.feedback_ref(ctx.heap()),
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
    become slow_keyed_store_no_shadow::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    match InlineCache::probe_mono(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, recv) {
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
    become slow_named_load::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
            ctx.feedback_ref(ctx.heap()),
            fb,
            recv,
            idx as usize,
        ) {
            next!(LoadKeyedProperty, ip, regs, ctx, table, roots, float, v)
        }
    }
    become slow_keyed_load::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_keyed_load_imm::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        InlineCache::try_load(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, global)
    {
        next!(LoadGlobal, ip, regs, ctx, table, roots, float, v)
    }
    become slow_global_load::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let name_idx = ops.unsigned::<1>();
    let fb = ops.unsigned::<2>();
    let recv = regs.read(r, ctx);
    // The constant-pool name is a stable interned string; the word stays
    // valid for the allocation-free fast path.
    let name_word = ctx
        .constants_ref(ctx.heap())
        .as_ref()
        .at(ctx.heap(), name_idx)
        .raw();
    let name: Tagged<'_, vm_core::SlotName> =
        unsafe { Tagged::from_value_unchecked(name_word) };
    if InlineCache::try_store_fast(
        unsafe { ctx.heap_mut() },
        ctx.feedback_ref(ctx.heap()),
        fb,
        recv,
        name,
        acc,
    ) {
        next!(StoreNamedProperty, ip, regs, ctx, table, roots, float, acc)
    }
    become slow_store_named::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let base = ctx.frame_base();
    if depth == 0 {
        let heap = ctx.heap();
        let context_word = ctx.stack().context_slot(base).get(heap);
        let context = unsafe { context_word.cast::<Context>() };
        let slots = context.as_ref().slots.get(heap);
        let slots = unsafe { slots.cast::<FixedArray>() };
        let v = slots.as_ref().element_slot(slot).get(heap);
        next!(LoadContextSlot, ip, regs, ctx, table, roots, float, v)
    }
    let v = {
        let heap = ctx.heap();
        let Some(mut context) = ctx.stack().context_slot(base).get(heap).get_as::<Context>() else {
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
    let base = ctx.frame_base();
    let heap = ctx.heap();
    let Some(mut context) = ctx.stack().context_slot(base).get(heap).get_as::<Context>() else {
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
    let base = ctx.frame_base();
    let old = ctx.stack().context_slot(base).get(ctx.heap());
    regs.write(r, old);
    if acc.get_as::<Context>().is_none() {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    }
    ctx.stack().context_slot(base).store(acc);
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
    let base = ctx.frame_base();
    let context = regs.read(r, ctx);
    if context.get_as::<Context>().is_none() {
        bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
    }
    ctx.stack().context_slot(base).store(context);
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
    become slow_create_function_context::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_create_closure::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_create_empty_array::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_create_empty_object::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let v = ctx
        .stack()
        .new_target_slot(ctx.frame_base())
        .get(ctx.heap());
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
    let v = ctx.stack().context_slot(ctx.frame_base()).get(ctx.heap());
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
    become slow_less_than_or_equal::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_greater_than_or_equal::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_create_bare_object::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_create_block_context::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
        InlineCache::try_load(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, global)
    {
        next!(LoadGlobalFast, ip, regs, ctx, table, roots, float, v)
    }
    become slow_global_load::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_global_load_nothrow::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_store_global::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_store_named_no_shadow::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_store_named_no_shadow::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    become slow_instance_of::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let v = ctx.stack().callable_slot(ctx.frame_base()).get(ctx.heap());
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
    become slow_add_parent::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    match InlineCache::probe_mono(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, recv) {
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
    become slow_named_load::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let args = ctx.stack().window(ctx.frame_base(), base, count);
    let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
    let v = f(nctx, None, args);
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<2>();
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_method_start(ctx, pc, size, callee_word, recv, [0, 0], 0, fb)
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_call_method_proxy::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
        }
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<3>();
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_method_start(ctx, pc, size, callee_word, recv, [arg0, 0], 1, fb)
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_call_method_proxy::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
        }
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<4>();
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_method_start(ctx, pc, size, callee_word, recv, [arg0, arg1], 2, fb)
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_call_method_proxy::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
        }
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<1>();
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_function_start(ctx, pc, size, callee_word, [0, 0], 0, fb)
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_call_function_proxy::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<2>();
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_function_start(ctx, pc, size, callee_word, [arg0, 0], 1, fb)
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_call_function_proxy::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<3>();
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        call_function_start(ctx, pc, size, callee_word, [arg0, arg1], 2, fb)
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_call_function_proxy::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
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
    let callee_word = regs.read(callee, ctx);
    let fb = ops.unsigned::<3>();
    let __mc = slow_start!(
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
        MethodCall::Frame(_frame) => dispatch!(
            ctx.code_ptr(),
            ctx.undefined_word(),
            unsafe { Regs::new(ctx.regs_ptr()) },
            ctx,
            table,
            roots,
            float
        ),

        MethodCall::Proxy => {
            become slow_proxy_apply::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
        }
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
            become slow_proxy_apply::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
        }
        Some(CallTarget::Runtime(rt)) => {
            let v = ctx.call_runtime(rt, ctx.stack().window(ctx.frame_base(), base_r, count));
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
            kind,
            ..
        }) => {
            if kind.is_class_constructor() {
                bail!(acc, ip, regs, ctx, table, roots, float, VmError::Type);
            }
            let heap = unsafe { ctx.heap_mut() };
            let frame = match ctx.stack().push_frame(
                heap,
                ctx.caller_meta(pc + size, pc),
                Callee {
                    callable: target.erase(),
                    info,
                    context: context.erase(),
                },
                heap.known().undefined.as_tagged(heap).erase(),
                ctx.stack().window(ctx.frame_base(), base_r, count),
            ) {
                Ok(frame) => frame,
                Err(err) => return unsafe { ctx.raise_tag(err) },
            };
            ctx.set_frame_base(frame.base);
            // a pushed frame is entered at pc 0 with the accumulator
            // seeded undefined (the header init wrote both)
            dispatch!(
                ctx.code_ptr(),
                ctx.undefined_word(),
                unsafe { Regs::new(ctx.regs_ptr()) },
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
    let __mc = slow_start!(
        ip,
        regs,
        acc,
        ctx,
        table,
        roots,
        float,
        construct_start(ctx, pc, size, regs, callee, base_r, count)
    );
    match __mc {
        ConstructStart::Frame => {
            become construct_trampoline(
                unsafe { ctx.code_ptr().add(pc) },
                regs,
                acc,
                ctx,
                table,
                roots,
                float,
            )
        }
        ConstructStart::Runtime(rt) => {
            let v = dispatch_runtime_construct(ctx, rt, callee, base_r, count);
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
        ConstructStart::Slow => {
            become slow_construct::<STRIDE>(ip, regs, acc, ctx, table, roots, float)
        }
    }
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

/// Outcome of a `Construct` fast-path attempt.
enum ConstructStart {
    /// a constructor frame was pushed (the cache already points at it):
    /// enter it at this execution's anchor; its `Return` applies the
    /// receiver fixup
    Frame,
    /// receiver synthesis threw; the pending exception is set
    Threw(()),
    /// a runtime constructor: call it directly with `new_target` set
    Runtime(RuntimeIndex),
    /// not an ordinary function constructor: fall back to `slow_construct`
    Slow,
}

fn construct_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    regs: Regs,
    callee: i32,
    base: i32,
    count: usize,
) -> Result<ConstructStart, VmError> {
    let callee_word = regs.read(callee, ctx);
    let constructible = callee_word.as_heap_object().is_some_and(|o| {
        o.as_ref()
            .header
            .map
            .get(ctx.heap())
            .kind()
            .is_constructor()
    });
    if !constructible {
        return Ok(ConstructStart::Slow);
    }
    let kind = match Object::call_target(ctx.heap(), callee_word) {
        Some(CallTarget::Runtime(rt)) => return Ok(ConstructStart::Runtime(rt)),
        Some(CallTarget::Bytecode { kind, .. }) => kind,
        _ => return Ok(ConstructStart::Slow),
    };
    let derived = matches!(
        kind,
        FunctionKind::DerivedClassConstructor | FunctionKind::DefaultDerivedConstructor
    );
    let receiver = if derived {
        ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase()
    } else {
        if kind != FunctionKind::Normal && kind != FunctionKind::BaseClassConstructor {
            return Ok(ConstructStart::Slow);
        }
        match construct_receiver_fast(ctx, callee_word)? {
            Some(r) => r,
            None => return Ok(ConstructStart::Threw(())),
        }
    };
    let callee_word = regs.read(callee, ctx);
    let Some(CallTarget::Bytecode {
        target,
        info,
        context,
        kind,
        ..
    }) = Object::call_target(ctx.heap(), callee_word)
    else {
        return Ok(ConstructStart::Slow);
    };
    let derived2 = matches!(
        kind,
        FunctionKind::DerivedClassConstructor | FunctionKind::DefaultDerivedConstructor
    );
    if !derived2 && kind != FunctionKind::Normal && kind != FunctionKind::BaseClassConstructor {
        return Ok(ConstructStart::Slow);
    }
    let heap = unsafe { ctx.heap_mut() };
    let staged =
        ctx.stack()
            .stage_construct(ctx.frame_base(), receiver, base, count)?;
    let frame = ctx.stack().push_frame(
        heap,
        ctx.caller_meta(pc + size, pc),
        Callee {
            callable: target.erase(),
            info,
            context: context.erase(),
        },
        callee_word,
        staged,
    )?;
    ctx.set_frame_base(frame.base);
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
            // Parser-estimated slots plus margin: the first property adds
            // land in the preallocated array instead of growing it.
            let expected = callee
                .as_tagged(heap)
                .as_ref()
                .callable_info(heap)
                .map_or(0, |info| info.as_ref().expected_slots());
            let capacity = if expected == 0 {
                0
            } else {
                expected.max(2) + vm_core::Map::SLACK_MARGIN
            };
            let obj = heap.new_object_prealloc(&scope, map, capacity);
            return Ok(Some(obj.erase()));
        }
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

#[inline(always)]
fn slow_next_pc(ip: *const u8) -> *const u8 {
    let wide = unsafe { *ip } == Opcode::Wide as u8;
    let op = unsafe { *ip.add(wide as usize) } as usize;
    let size = if wide {
        *unsafe { OPERAND_SIZES_WIDE.get_unchecked(op) }
    } else {
        *unsafe { OPERAND_SIZES_NARROW.get_unchecked(op) }
    } as usize;
    unsafe { ip.add(wide as usize + 1 + size) }
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn construct_trampoline<'a>(
    fault_ip: *const u8,
    _regs: Regs,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
    float: FloatReg,
) -> Tagged<'a, Value> {
    let call_pc = fault_ip as usize - ctx.code_ptr() as usize;
    let frame_base = ctx.frame_base();
    let probe = 0u8;
    if ctx.stack_overflowed() {
        let caller = ctx.stack().pop_frame(frame_base);
        ctx.set_frame_base(caller.base);
        let _ = unsafe { ctx.raise_tag(VmError::StackOverflow) };
        let v = ctx.exception_word();
        let code = ctx.code_ptr();
        become resume(
            unsafe { code.add(call_pc) },
            unsafe { Regs::new(core::ptr::null_mut()) },
            v,
            ctx,
            table,
            roots,
            float,
        );
    }
    let code = ctx.code_ptr();
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let acc0 = ctx.stack().undefined_word(ctx.heap());
    let callee_ctx = unsafe { Ctx::child(ctx, frame_base) };
    let ip0 = code;
    let op = unsafe { *ip0 } as usize;
    let acc = unsafe { table.get(op as u8)(ip0, regs, acc0, &callee_ctx, table, roots, float) };
    if ctx.is_throw(acc) {
        let caller = ctx.stack().pop_frame(frame_base);
        ctx.set_frame_base(caller.base);
        let caller_code = ctx.code_ptr();
        become throw_dispatch(
            unsafe { caller_code.add(call_pc) },
            unsafe { Regs::new(core::ptr::null_mut()) },
            acc,
            ctx,
            table,
            roots,
            float,
        )
    }
    // `this` lives in parameter register 0 of the callee window
    let this_val = regs.read(0, ctx.heap());
    let caller = ctx.stack().pop_frame(frame_base);
    ctx.set_frame_base(caller.base);
    let v = if Convert::is_primitive(ctx.heap(), acc) {
        if this_val == ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase() {
            let _ = unsafe { ctx.raise_tag(VmError::Type) };
            let caller_code = ctx.code_ptr();
            become throw_dispatch(
                unsafe { caller_code.add(caller.pc) },
                unsafe { Regs::new(core::ptr::null_mut()) },
                ctx.exception_word(),
                ctx,
                table,
                roots,
                float,
            )
        }
        this_val
    } else {
        acc
    };
    let code = ctx.code_ptr();
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let ip = unsafe { code.add(caller.pc) };
    let h = table.get(unsafe { *ip });
    become h(ip, regs, v, ctx, table, roots, float)
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
    let ip = unsafe { slow_next_pc(ctx.code_ptr().add(fault_pc)) };
    let regs = unsafe { Regs::new(ctx.regs_ptr()) };
    let h = table.get(unsafe { *ip });
    become h(ip, regs, acc, ctx, table, roots, float)
}

enum MethodCall<'a> {
    /// the call completed (runtime callee): the value or the exception
    /// sentinel
    Value(Tagged<'a, Value>),
    /// a bytecode frame was pushed and made current: enter it
    /// through the machine-call trampoline
    Frame(FrameMeta),
    /// a callable proxy: the caller tail-calls the slow trap dispatch
    Proxy,
}

/// Invoke a runtime callee over scattered call-site registers.
#[inline(always)]
fn dispatch_runtime_scattered<'a>(
    ctx: &Ctx<'a>,
    rt: RuntimeIndex,
    recv: Recv,
    args: [i32; 2],
    argc: usize,
) -> Tagged<'a, Value> {
    match ctx
        .stack()
        .stage_scattered(ctx.heap(), ctx.frame_base(), recv, args, argc)
    {
        Ok(a) => ctx.call_runtime(rt, a),
        Err(err) => {
            let _ = unsafe { ctx.raise(err) };
            ctx.exception_word()
        }
    }
}

/// Invoke a runtime constructor: undefined receiver at element 0,
/// `new_target` rooted over the callee register.
#[inline(always)]
fn dispatch_runtime_construct<'a>(
    ctx: &Ctx<'a>,
    rt: RuntimeIndex,
    callee_reg: i32,
    base: i32,
    count: usize,
) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(rt);
    let stack = ctx.stack();
    let undefined = stack.undefined_word(ctx.heap());
    let args = match stack.stage_construct(ctx.frame_base(), undefined, base, count) {
        Ok(a) => a,
        Err(err) => {
            let _ = unsafe { ctx.raise(err) };
            return ctx.exception_word();
        }
    };
    let saved = stack.top();
    let bump = stack.rooting_top(args);
    if let Some(top) = bump {
        stack.set_top(top);
    }
    // new.target for the Construct opcode is the callee itself
    let new_target = Some(stack.reg_handle(ctx.frame_base(), callee_reg));
    let nctx = RuntimeContext::new(ctx.vm(), unsafe { ctx.heap_mut() }, ctx.state());
    let v = f(nctx, new_target, args);
    if bump.is_some() {
        stack.set_top(saved);
    }
    v
}

/// Push a callee frame (facts from a `CallTarget::Bytecode` destructure
/// or a call-IC hit; the caller is the current frame) and switch to it.
/// Entry is at pc 0 with the accumulator seeded undefined.
#[inline(always)]
fn push_callee_frame<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee: Callee<'_>,
    args: Args,
) -> Result<MethodCall<'a>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let frame = ctx.stack().push_frame(
        heap,
        ctx.caller_meta(pc + size, pc),
        callee,
        heap.known().undefined.as_tagged(heap).erase(),
        args,
    )?;
    ctx.set_frame_base(frame.base);
    Ok(MethodCall::Frame(frame))
}

/// Push a callee frame from scattered call-site registers (one copy
/// per word).
#[inline(always)]
fn push_scattered_frame<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee: Callee<'_>,
    recv: Recv,
    args: [i32; 2],
    argc: usize,
) -> Result<MethodCall<'a>, VmError> {
    let heap = unsafe { ctx.heap_mut() };
    let frame = ctx.stack().push_scattered_frame(
        heap,
        ctx.caller_meta(pc + size, pc),
        callee,
        heap.known().undefined.as_tagged(heap).erase(),
        recv,
        args,
        argc,
    )?;
    ctx.set_frame_base(frame.base);
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
fn call_method_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    recv: i32,
    args: [i32; 2],
    argc: usize,
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match InlineCache::call_probe(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, callee_word) {
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            push_scattered_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info,
                    context: context.erase(),
                },
                Recv::Reg(recv),
                args,
                argc,
            )
        }
        CallProbe::Runtime(rt) => Ok(MethodCall::Value(dispatch_runtime_scattered(
            ctx,
            rt,
            Recv::Reg(recv),
            args,
            argc,
        ))),
        CallProbe::Miss => {
            slow_call_method_miss(ctx, pc, size, callee_word, recv, args, argc, fb)
        }
    }
}

#[inline(always)]
fn call_function_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    args: [i32; 2],
    argc: usize,
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match InlineCache::call_probe(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, callee_word) {
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(unsafe { ctx.raise_tag(VmError::Type) }));
            }
            push_scattered_frame(
                ctx,
                pc,
                size,
                Callee {
                    callable: target.erase(),
                    info,
                    context: context.erase(),
                },
                Recv::Undefined,
                args,
                argc,
            )
        }
        CallProbe::Runtime(rt) => Ok(MethodCall::Value(dispatch_runtime_scattered(
            ctx,
            rt,
            Recv::Undefined,
            args,
            argc,
        ))),
        CallProbe::Miss => {
            slow_call_function_miss(ctx, pc, size, callee_word, args, argc, fb)
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
    match InlineCache::call_probe(ctx.heap(), ctx.feedback_ref(ctx.heap()), fb, callee_word) {
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            kind,
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
                    info,
                    context: context.erase(),
                },
                ctx.stack().window(ctx.frame_base(), base, count),
            )
        }
        CallProbe::Runtime(rt) => Ok(MethodCall::Value(
            ctx.call_runtime(rt, ctx.stack().window(ctx.frame_base(), base, count)),
        )),
        CallProbe::Miss => slow_call_miss(ctx, pc, size, callee_word, base, count, fb),
    }
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
        Some(CallTarget::Runtime(rt)) => {
            let f = vm.runtime(rt);
            let nctx = RuntimeContext::new(vm, heap, state);
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
            let stack = state.stack();
            let frame = {
                let heap: &Heap = heap;
                let new_target_value = match &new_target {
                    Some(nt) => nt.as_tagged(heap).erase(),
                    None => heap.known().undefined.as_tagged(heap).erase(),
                };
                let staged = stack.stage_slice(args.as_tagged())?;
                stack.push_frame(
                    heap,
                    FrameMeta::ROOT,
                    Callee {
                        callable: target.erase(),
                        info,
                        context: context.erase(),
                    },
                    new_target_value,
                    staged,
                )?
            };
            state.set_frame_base(frame.base);
            state.set_frame_active(true);

            let probe = 0u8;
            let stack_limit = (&probe as *const u8 as usize).saturating_sub(6 * 1024 * 1024);
            let ctx: Ctx<'a> = unsafe { Ctx::new(vm, heap, state, frame.base, stack_limit) };
            // the fresh frame is entered at pc 0 with the accumulator
            // seeded undefined (the header init wrote both)
            let ip = ctx.code_ptr();
            let regs = unsafe { Regs::new(ctx.regs_ptr()) };
            let acc = ctx.undefined_word();
            let table = TableArg::new(&TABLE_NARROW);
            let roots = RootsArg::from_heap(ctx.heap());
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

    let saved_top = stack.top();
    let was_active = state.is_frame_active();
    let outer = was_active.then(|| state.frame_base());

    state.handle_scope(|scope| {
        let result = enter(vm, heap, state, callable, args, new_target)?;
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

impl Interpreter for BecomeInterpreter {
    const EXECUTE: ExecuteFn = execute;
}
