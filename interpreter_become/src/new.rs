//! Miniature become interpreter sketching the macro-light redesign:
//! one plain-Rust handler body per opcode (const-generic over stride),
//! names-only table macro, per-arch ABI args, ip-fused dispatch.
//! Everything is verified against disassembly; run
//! `cargo test --lib new --release --no-run` and objdump the test
//! binary to inspect codegen.

use std::marker::PhantomData;

use bytecode::{Opcode, Scale, jump_target};
use vm_core::{Register, Smi, Tagged, Value};

pub struct MiniHeap {
    pub roots: Roots,
}

/// `booleans`: `false` at +0, `true` at +8 — selection by bit.
#[repr(C)]
pub struct Roots {
    pub undefined: Register,
    pub null: Register,
    pub booleans: [Register; 2],
}

impl Roots {
    #[inline(always)]
    pub fn undefined<'h>(&self, _heap: &'h MiniHeap) -> Tagged<'h, Value> {
        unsafe { Tagged::from_value_unchecked(self.undefined.raw()) }
    }

    #[inline(always)]
    pub fn boolean<'h>(&self, _heap: &'h MiniHeap, which: bool) -> Tagged<'h, Value> {
        unsafe { Tagged::from_value_unchecked(self.booleans[which as usize].raw()) }
    }
}

pub struct MiniCtx<'h> {
    heap: *mut MiniHeap,
    _anchor: PhantomData<&'h mut MiniHeap>,
}

impl<'h> MiniCtx<'h> {
    #[inline(always)]
    pub fn new(heap: &'h mut MiniHeap) -> MiniCtx<'h> {
        MiniCtx {
            heap: heap as *mut MiniHeap,
            _anchor: PhantomData,
        }
    }

    #[inline(always)]
    pub fn heap(&self) -> &'h MiniHeap {
        unsafe { &*self.heap }
    }

    #[inline(always)]
    pub fn roots(&self) -> &'h Roots {
        &self.heap().roots
    }
}

/// Table arg: Scalar newtype (own register) on aarch64, ZST the
/// rustic ABI omits entirely elsewhere.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct TableArg<'a>(&'a MiniTable);

#[cfg(target_arch = "aarch64")]
impl<'a> TableArg<'a> {
    #[inline(always)]
    pub fn new(table: &'a MiniTable) -> TableArg<'a> {
        TableArg(table)
    }

    #[inline(always)]
    pub fn get(&self, op: u8) -> MiniHandler {
        self.0.0[op as usize]
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[derive(Clone, Copy)]
pub struct TableArg<'a>(PhantomData<&'a ()>);

#[cfg(not(target_arch = "aarch64"))]
impl<'a> TableArg<'a> {
    #[inline(always)]
    pub fn new() -> TableArg<'a> {
        TableArg(PhantomData)
    }

    #[inline(always)]
    pub fn get(&self, op: u8) -> MiniHandler {
        MINI_TABLE_NARROW.0[op as usize]
    }
}

/// Roots arg: pinned interior reference on aarch64, ZST elsewhere
/// (accessors reach the heap-inlined roots).
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct RootsArg<'a>(&'a Roots);

#[cfg(target_arch = "aarch64")]
impl<'a> RootsArg<'a> {
    #[inline(always)]
    pub fn new(roots: &'a Roots) -> RootsArg<'a> {
        RootsArg(roots)
    }

    #[inline(always)]
    pub fn undefined<'h, W>(&self, heap: W) -> Tagged<'h, Value>
    where
        W: Into<&'h MiniHeap>,
    {
        self.0.undefined(heap.into())
    }

    #[inline(always)]
    pub fn boolean<'h, W>(&self, heap: W, which: bool) -> Tagged<'h, Value>
    where
        W: Into<&'h MiniHeap>,
    {
        self.0.boolean(heap.into(), which)
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
    pub fn undefined<'h, W>(&self, heap: W) -> Tagged<'h, Value>
    where
        W: Into<&'h MiniHeap>,
    {
        let heap = heap.into();
        heap.roots.undefined(heap)
    }

    #[inline(always)]
    pub fn boolean<'h, W>(&self, heap: W, which: bool) -> Tagged<'h, Value>
    where
        W: Into<&'h MiniHeap>,
    {
        let heap = heap.into();
        heap.roots.boolean(heap, which)
    }
}

#[cfg(target_arch = "aarch64")]
mod regbank {
    use vm_core::{Register, Value};

    pub type Reg = Register;

    #[inline(always)]
    pub unsafe fn init(v: Value) -> Reg {
        Register::from_value(v)
    }
}

/// x86_64 stand-in bank: same `raw`/`store` surface as `Register`.
#[cfg(not(target_arch = "aarch64"))]
mod regbank {
    use std::cell::Cell;

    use vm_core::{Tagged, Value};

    pub struct Reg(Cell<u64>);

    #[inline(always)]
    pub unsafe fn init(v: Value) -> Reg {
        Reg(Cell::new(v.to_bits()))
    }

    impl Reg {
        #[inline(always)]
        pub fn raw(&self) -> Value {
            Value::from_bits(self.0.get())
        }

        #[inline(always)]
        pub fn store(&self, v: Tagged<'_, Value>) {
            self.0.set(v.raw().to_bits())
        }
    }
}

use regbank::Reg;

/// Frame register window: signed indices (negative = parameters),
/// re-acquire after any safepoint. Reads take a heap witness that
/// anchors the returned value to the GC epoch; writes root.
#[derive(Clone, Copy)]
pub struct Regs(*mut Reg);

impl Regs {
    /// # Safety
    /// `base` is a rooted window of the running frame; no GC before the
    /// last use of this handle or anything loaded through it.
    #[inline(always)]
    pub unsafe fn new(base: *mut Reg) -> Regs {
        Regs(base)
    }

    /// # Safety
    /// `i` addresses a live slot (negative = parameters) holding a
    /// strong `Value`; the window is still valid for `'h`.
    #[inline(always)]
    pub unsafe fn read<'h, W>(self, i: i32, _heap: W) -> Tagged<'h, Value>
    where
        W: Into<&'h MiniHeap>,
    {
        Tagged::from_value_unchecked((*self.0.offset(i as isize)).raw())
    }

    /// # Safety
    /// `i` must address a live slot in this window.
    #[inline(always)]
    pub unsafe fn write(self, i: i32, v: Tagged<'_, Value>) {
        (*self.0.offset(i as isize)).store(v)
    }
}

impl<'h> From<&MiniCtx<'h>> for &'h MiniHeap {
    #[inline(always)]
    fn from(ctx: &MiniCtx<'h>) -> &'h MiniHeap {
        ctx.heap()
    }
}

pub struct MiniTable([MiniHandler; 256]);

/// One definition for all arches: the trailing arg types carry every
/// per-arch difference (Scalar newtypes on aarch64, ZSTs the rustic
/// ABI omits elsewhere).
pub type MiniHandler = for<'a> extern "rust-preserve-none" fn(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value>;

/// Operand cursor; debug builds cross-check against the canonical
/// operand tables (dispatched byte in `new`, operand count in `size`),
/// which compile away entirely in release.
struct Ops<const STRIDE: usize> {
    ip: *const u8,
    #[cfg(debug_assertions)]
    op: Opcode,
}

impl<const STRIDE: usize> Ops<STRIDE> {
    #[inline(always)]
    fn new(ip: *const u8, op: Opcode) -> Ops<STRIDE> {
        debug_assert_eq!(
            unsafe { if STRIDE == 2 { *ip.add(1) } else { *ip } },
            op as u8,
            "handler dispatched for the wrong opcode"
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
        let off = (if STRIDE == 2 { 2 } else { 1 }) + I * STRIDE;
        unsafe {
            if STRIDE == 1 {
                *self.ip.add(off) as i8 as i32
            } else {
                i16::from_le_bytes([*self.ip.add(off), *self.ip.add(off + 1)]) as i32
            }
        }
    }

    #[inline(always)]
    fn advance(&mut self, by: usize) {
        self.ip = self.ip.wrapping_add(by);
    }
}

#[cfg(feature = "star_fusion")]
const fn mini_star(op: Opcode) -> bool {
    matches!(op, Opcode::Load | Opcode::LoadSmi)
}

unsafe fn size_at(ip: *const u8) -> usize {
    let b = *ip;
    if b == Opcode::Wide as u8 {
        let op = Opcode::from_byte_unchecked(*ip.add(1));
        1 + op.size(Scale::Byte2)
    } else {
        Opcode::from_byte_unchecked(b).size(Scale::Byte1)
    }
}

macro_rules! mini_dispatch {
    ($p:expr, $a:expr, $r:expr, $x:expr, $t:ident, $k:ident) => {{
        let h = unsafe { $t.get(*$p) };
        become h($p, $r, $a, $x, $t, $k)
    }};
}

macro_rules! mini_jump {
    ($ops:expr, $a:expr, $r:expr, $x:expr, $t:ident, $k:ident) => {{
        let off = $ops.signed::<0>();
        mini_dispatch!($ops.ip.wrapping_offset(off as isize), $a, $r, $x, $t, $k)
    }};
}

/// Advance to the next instruction; `mini_next!(ops, regs, ctx,
/// value, Op, table, roots)`.
macro_rules! mini_next {
    ($ops:expr, $r:expr, $x:expr, $a:expr, $op:ident, $t:ident, $k:ident) => {{
        let mut ops = $ops;
        let size =
            (if STRIDE == 2 { 2 } else { 1 }) + STRIDE * const { Opcode::$op.operands().len() };
        ops.advance(size);
        #[cfg(feature = "star_fusion")]
        {
            {
                if mini_star(Opcode::$op) && unsafe { *ops.ip } == Opcode::Store as u8 {
                    let dst = unsafe { *ops.ip.add(1) } as i8 as i32;
                    unsafe { $r.write(dst, $a) };
                    mini_dispatch!(unsafe { ops.ip.add(2) }, $a, $r, $x, $t, $k)
                } else {
                    mini_dispatch!(ops.ip, $a, $r, $x, $t, $k)
                }
            }
        }
        #[cfg(not(feature = "star_fusion"))]
        {
            { mini_dispatch!(ops.ip, $a, $r, $x, $t, $k) }
        }
    }};
}

macro_rules! mini_reenter {
    ($ip:expr, $r:expr, $x:expr, $a:expr, $t:ident, $k:ident) => {{ become mini_next_fn($ip, $r, $a, $x, $t, $k) }};
}

/// Handler-shaped dispatch trampoline (`become` needs matching
/// signatures); re-derives the instruction size from the opcode.
extern "rust-preserve-none" fn mini_next_fn<'a>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let p = ip.wrapping_add(unsafe { size_at(ip) });
    #[cfg(feature = "star_fusion")]
    {
        let op = unsafe { Opcode::from_byte_unchecked(*ip) };
        if mini_star(op) && unsafe { *p } == Opcode::Store as u8 {
            let dst = unsafe { *p.add(1) } as i8 as i32;
            unsafe { regs.write(dst, acc) };
            mini_dispatch!(unsafe { p.add(2) }, acc, regs, ctx, table, roots)
        } else {
            mini_dispatch!(p, acc, regs, ctx, table, roots)
        }
    }
    #[cfg(not(feature = "star_fusion"))]
    {
        mini_dispatch!(p, acc, regs, ctx, table, roots)
    }
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_load<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let mut ops = Ops::<STRIDE>::new(ip, Opcode::Load);
    let r = ops.signed::<0>();
    let v = unsafe { regs.read(r, ctx) };
    mini_next!(ops, regs, ctx, v, Load, table, roots)
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_smi<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let mut ops = Ops::<STRIDE>::new(ip, Opcode::LoadSmi);
    let imm = ops.signed::<0>();
    let v = Smi::new(imm as i64).into_tagged();
    mini_next!(ops, regs, ctx, v, LoadSmi, table, roots)
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_load_undefined<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let ops = Ops::<STRIDE>::new(ip, Opcode::LoadUndefined);
    let v = roots.undefined(ctx);
    mini_next!(ops, regs, ctx, v, LoadUndefined, table, roots)
}

/// Smi equality; non-smi operands conservatively compare unequal.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_equal<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let mut ops = Ops::<STRIDE>::new(ip, Opcode::Equal);
    let r = ops.signed::<0>();
    let lhs = unsafe { regs.read(r, ctx) };
    let eq = match (lhs.smi_bits(), acc.smi_bits()) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };
    let v = roots.boolean(ctx, eq);
    mini_next!(ops, regs, ctx, v, Equal, table, roots)
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_store<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let mut ops = Ops::<STRIDE>::new(ip, Opcode::Store);
    let r = ops.signed::<0>();
    unsafe { regs.write(r, acc) };
    mini_next!(ops, regs, ctx, acc, Store, table, roots)
}

/// Uses the fn-based trampoline instead of `mini_next!`.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_add<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let mut ops = Ops::<STRIDE>::new(ip, Opcode::Add);
    let r = ops.signed::<0>();
    let lhs = unsafe { regs.read(r, ctx) };
    if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
        && let Some(sum) = a.checked_add(b)
    {
        mini_reenter!(ip, regs, ctx, Tagged::from_smi_bits(sum), table, roots)
    } else {
        mini_reenter!(ip, regs, ctx, acc, table, roots)
    }
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_jump<'a, const STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let mut ops = Ops::<STRIDE>::new(ip, Opcode::Jump);
    mini_jump!(ops, acc, regs, ctx, table, roots)
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_return<'a, const STRIDE: usize>(
    _ip: *const u8,
    _regs: Regs,
    acc: Tagged<'a, Value>,
    _ctx: &MiniCtx<'a>,
    _table: TableArg<'a>,
    _roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    acc
}

#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_wide<'a, const _STRIDE: usize>(
    ip: *const u8,
    regs: Regs,
    acc: Tagged<'a, Value>,
    ctx: &MiniCtx<'a>,
    table: TableArg<'a>,
    roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    let h = MINI_TABLE_WIDE.0[unsafe { *ip.add(1) } as usize];
    become h(ip, regs, acc, ctx, table, roots)
}

/// A safe fn coercing into the handler-pointer type.
#[inline(never)]
#[rustc_align(32)]
extern "rust-preserve-none" fn op_trap<'a>(
    _ip: *const u8,
    _regs: Regs,
    _acc: Tagged<'a, Value>,
    _ctx: &MiniCtx<'a>,
    _table: TableArg<'a>,
    _roots: RootsArg<'a>,
) -> Tagged<'a, Value> {
    panic!("mini become interpreter: opcode stream reached a trap slot");
}

/// Names-only opcode -> handler list; no bodies live here.
macro_rules! mini_tables {
    ($($op:ident => $f:ident),* $(,)?) => {
        const fn mini_table_narrow() -> [MiniHandler; 256] {
            let mut t = [op_trap as MiniHandler; 256];
            $( t[Opcode::$op as usize] = $f::<1> as MiniHandler; )*
            t
        }

        const fn mini_table_wide() -> [MiniHandler; 256] {
            let mut t = [op_trap as MiniHandler; 256];
            $( t[Opcode::$op as usize] = $f::<2> as MiniHandler; )*
            t
        }
    };
}

mini_tables! {
    Load => op_load,
    LoadSmi => op_load_smi,
    LoadUndefined => op_load_undefined,
    Store => op_store,
    Add => op_add,
    Equal => op_equal,
    Jump => op_jump,
    Return => op_return,
    Wide => op_wide,
}

static MINI_TABLE_NARROW: MiniTable = MiniTable(mini_table_narrow());
static MINI_TABLE_WIDE: MiniTable = MiniTable(mini_table_wide());

#[cfg(test)]
mod tests {
    use super::*;
    use bytecode::emit;

    fn heap() -> MiniHeap {
        unsafe {
            MiniHeap {
                roots: Roots {
                    undefined: Register::from_value(Smi::new(42).into_tagged().raw()),
                    null: Register::from_value(Smi::new(1).into_tagged().raw()),
                    booleans: [
                        Register::from_value(Smi::new(0).into_tagged().raw()),
                        Register::from_value(Smi::new(1).into_tagged().raw()),
                    ],
                },
            }
        }
    }

    fn run<'a>(code: &[u8], regs: &mut [Reg], ctx: &'a MiniCtx<'a>) -> Tagged<'a, Value> {
        let h = MINI_TABLE_NARROW.0[code[0] as usize];
        unsafe {
            h(
                code.as_ptr(),
                Regs::new(regs.as_mut_ptr()),
                Smi::new(0).into_tagged(),
                ctx,
                TableArg::new(&MINI_TABLE_NARROW),
                RootsArg::new(ctx.roots()),
            )
        }
    }

    fn fresh_regs<const N: usize>() -> [Reg; N] {
        [(); N].map(|()| unsafe { regbank::init(Smi::new(0).into_tagged().raw()) })
    }

    #[test]
    fn narrow_wide_jump_and_fusion() {
        let mut code = Vec::new();
        emit(&mut code, Opcode::LoadSmi, &[5]);
        emit(&mut code, Opcode::Store, &[0]);
        emit(&mut code, Opcode::LoadSmi, &[300]);
        emit(&mut code, Opcode::Add, &[0]);
        emit(&mut code, Opcode::Store, &[1]);
        emit(&mut code, Opcode::LoadUndefined, &[]);
        emit(&mut code, Opcode::Store, &[2]);
        emit(&mut code, Opcode::Load, &[1]);
        emit(&mut code, Opcode::Equal, &[1]);
        emit(&mut code, Opcode::Jump, &[4]);
        emit(&mut code, Opcode::LoadSmi, &[99]);
        emit(&mut code, Opcode::Return, &[]);
        let mut heap = heap();
        let mut regs = fresh_regs::<4>();
        let ctx = MiniCtx::new(&mut heap);
        let out = run(&code, &mut regs, &ctx);
        let rs = unsafe { Regs::new(regs.as_mut_ptr()) };
        let r0 = unsafe { rs.read(0, &ctx).to_i64() };
        let r1 = unsafe { rs.read(1, &ctx).to_i64() };
        let r2 = unsafe { rs.read(2, &ctx).to_i64() };
        assert_eq!(out.to_i64(), Some(1));
        assert_eq!(r0, Some(5));
        assert_eq!(r1, Some(305));
        assert_eq!(r2, Some(42));
    }

    #[test]
    fn long_tail_chain_does_not_overflow() {
        let mut code = Vec::with_capacity(2 * 1_000_000 + 2);
        for _ in 0..1_000_000 {
            emit(&mut code, Opcode::LoadSmi, &[7]);
        }
        emit(&mut code, Opcode::Return, &[]);
        let mut heap = heap();
        let mut regs = fresh_regs::<4>();
        let ctx = MiniCtx::new(&mut heap);
        let out = run(&code, &mut regs, &ctx);
        assert_eq!(out.to_i64(), Some(7));
    }

    #[test]
    fn table_instantiations_are_32_byte_aligned() {
        let load_n = MINI_TABLE_NARROW.0[Opcode::Load as usize] as usize;
        assert_eq!(
            load_n % 32,
            0,
            "narrow instantiation lost #[rustc_align(32)]"
        );
        let load_w = MINI_TABLE_WIDE.0[Opcode::Load as usize] as usize;
        assert_eq!(load_w % 32, 0, "wide instantiation lost #[rustc_align(32)]");
    }
}
