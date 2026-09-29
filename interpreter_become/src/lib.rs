#![feature(explicit_tail_calls)]
#![feature(rust_preserve_none_cc)]
#![allow(incomplete_features)]
#![feature(fn_align)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_macros, unused_unsafe, unused_variables)]

use bytecode::{OPERAND_SIZES_NARROW, OPERAND_SIZES_WIDE, Opcode, jump_target};
use vm_core::ic::{ElementHit, Hit, InlineCache, MonoProbe, StoreHit, StoreOutcomeKind};
use vm_core::proxy::Proxy;
use vm_core::{
    CallTarget, CallableInfoObject, Coercion, Compare, Context, ContextInit, ContextState, Convert,
    Ctx, DenseString, Errors, ExecuteFn, FixedArray, FrameMeta, FunctionKind, Handle, HandleSlice,
    Heap, Hint, Interpreter, Intrinsic, Key, LoadOutcome, Lookup, Object, PropertyDescriptor,
    Register, RuntimeContext, RuntimeIndex, ScopeInfo, SlotName, Smi, StoreOutcome, StoreSemantics,
    Tagged, VM, Value, VmError, spread_apply_args,
};

pub struct BecomeInterpreter;

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
) -> Tagged<'a, Value>;

pub struct HandlerTable([Handler; 256]);

macro_rules! helpers {
    ($pc:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident, $op:ident, $stride:literal, $($arg:ident => $kind:ident),* $(,)?) => {
        const BASE: usize = if $stride == 2 { 2 } else { 1 };
        #[allow(dead_code)]
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
        macro_rules! bail {
            ($e:expr) => {{
                let _ = $ctx.raise_tag($e);
                become throw_dispatch($pc, $code, $regs, $acc, $ctx)
            }}
        }
        macro_rules! threw { () => { become throw_dispatch($pc, $code, $regs, $acc, $ctx) } }
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
            Err(err) => $ctx.raise_tag(err),
        }
    };
}

macro_rules! cold_start {
    ($pc:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident, $e:expr) => {
        match $e {
            Ok(m) => m,
            Err(err) => {
                let _ = $ctx.raise_tag(err);
                become throw_dispatch($pc, $code, $regs, $acc, $ctx)
            }
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
            ) -> Tagged<'a, Value> {
                helpers!($pc, $code, $regs, $acc, $ctx, $op, 1, $($arg => $kind),*);
                $body
            }
            #[inline(never)]
            #[rustc_align(32)]
            unsafe extern "rust-preserve-none" fn $w<'a>(
                $pc: usize, $code: *const u8, $regs: *mut Register, $acc: Tagged<'a, Value>, $ctx: &Ctx<'a>,
            ) -> Tagged<'a, Value> {
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
) -> Tagged<'a, Value> {
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
) -> Tagged<'a, Value> {
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

    Add : op_add_n / op_add_w (r => signed) {
        let lhs = reg!(r);
        if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
            && let Some(sum) = a.checked_add(b)
        {
            next!(Tagged::from_smi_bits(sum))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
            let sum = a + b;
            if let Some(s) = Smi::from_f64(sum) {
                next!(s.into_tagged())
            }
            ctx.set_num(sum);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_add(pc, code, regs, acc, ctx)
    }

    Sub : op_sub_n / op_sub_w (r => signed) {
        let lhs = reg!(r);
        if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
            && let Some(diff) = a.checked_sub(b)
        {
            next!(Tagged::from_smi_bits(diff))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
            let diff = a - b;
            if let Some(s) = Smi::from_f64(diff) {
                next!(s.into_tagged())
            }
            ctx.set_num(diff);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    Mul : op_mul_n / op_mul_w (r => signed) {
        let lhs = reg!(r);
        if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
            && let Some(product) = (a >> 1).checked_mul(b)
        {
            next!(Tagged::from_smi_bits(product))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
            let product = a * b;
            if let Some(s) = Smi::from_f64(product) {
                next!(s.into_tagged())
            }
            ctx.set_num(product);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    Div : op_div_n / op_div_w (r => signed) {
        let lhs = reg!(r);
        if let (Some(a), Some(b)) = (lhs.smi_bits(), acc.smi_bits())
            && b != 0
            && a % b == 0
            && let Some(quotient) = a.checked_div(b)
            && let Some(encoded) = quotient.checked_mul(2)
        {
            next!(Tagged::from_smi_bits(encoded))
        }
        if let (Some(a), Some(b)) = (Convert::as_number(lhs), Convert::as_number(acc)) {
            let quotient = a / b;
            if let Some(s) = Smi::from_f64(quotient) {
                next!(s.into_tagged())
            }
            ctx.set_num(quotient);
            become cold_box_number(pc, code, regs, Smi::new(SIZE as i64).into_tagged(), ctx)
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    Mod : op_mod_n / op_mod_w (r => signed) {
        let lhs = reg!(r);
        if let (Some(a), Some(b)) = (lhs.to_i64(), acc.to_i64())
            && b != 0
        {
            next!(Smi::new(a % b).into_tagged())
        }
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    Exp : op_exp_n / op_exp_w (r => signed) {
        become cold_numeric(pc, code, regs, acc, ctx)
    }

    BitwiseOr : op_bitwise_or_n / op_bitwise_or_w (r => signed) {
        let (Some(a), Some(b)) = (reg!(r).to_i64(), acc.to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32 | b as i32) as i64).into_tagged())
    }

    BitwiseXor : op_bitwise_xor_n / op_bitwise_xor_w (r => signed) {
        let (Some(a), Some(b)) = (reg!(r).to_i64(), acc.to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32 ^ b as i32) as i64).into_tagged())
    }

    BitwiseAnd : op_bitwise_and_n / op_bitwise_and_w (r => signed) {
        let (Some(a), Some(b)) = (reg!(r).to_i64(), acc.to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32 & b as i32) as i64).into_tagged())
    }

    ShiftLeft : op_shift_left_n / op_shift_left_w (r => signed) {
        let (Some(a), Some(b)) = (reg!(r).to_i64(), acc.to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32).wrapping_shl(b as u32 & 31) as i64).into_tagged())
    }

    ShiftRight : op_shift_right_n / op_shift_right_w (r => signed) {
        // ToInt32(lhs) >> (ToUint32(rhs) & 31), sign-extending; Smis only
        let (Some(a), Some(b)) = (reg!(r).to_i64(), acc.to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32).wrapping_shr(b as u32 & 31) as i64).into_tagged())
    }

    ShiftRightLogical : op_shift_right_logical_n / op_shift_right_logical_w (r => signed) {
        let (Some(a), Some(b)) = (reg!(r).to_i64(), acc.to_i64()) else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as u32).wrapping_shr(b as u32 & 31) as i64).into_tagged())
    }

    AddImmediate : op_add_immediate_n / op_add_immediate_w (r => signed, imm => signed) {
        let lhs = reg!(r);
        if let Some(bits) = lhs.smi_bits()
            && let Some(sum) = bits.checked_add((imm as i64) << 1)
        {
            next!(Tagged::from_smi_bits(sum))
        }
        become cold_add_immediate(pc, code, regs, acc, ctx)
    }

    SubImmediate : op_sub_immediate_n / op_sub_immediate_w (r => signed, imm => signed) {
        let lhs = reg!(r);
        if let Some(bits) = lhs.smi_bits()
            && let Some(diff) = bits.checked_sub((imm as i64) << 1)
        {
            next!(Tagged::from_smi_bits(diff))
        }
        become cold_numeric_immediate(pc, code, regs, acc, ctx)
    }

    MulImmediate : op_mul_immediate_n / op_mul_immediate_w (r => signed, imm => signed) {
        let lhs = reg!(r);
        if let Some(bits) = lhs.smi_bits()
            && let Some(product) = (bits >> 1).checked_mul(imm as i64)
            && let Some(encoded) = product.checked_mul(2)
        {
            next!(Tagged::from_smi_bits(encoded))
        }
        become cold_numeric_immediate(pc, code, regs, acc, ctx)
    }

    DivImmediate : op_div_immediate_n / op_div_immediate_w (r => signed, imm => signed) {
        let lhs = reg!(r);
        if let Some(bits) = lhs.smi_bits() {
            let value = bits >> 1;
            if imm != 0
                && value % imm as i64 == 0
                && let Some(quotient) = value.checked_div(imm as i64)
                && let Some(encoded) = quotient.checked_mul(2)
            {
                next!(Tagged::from_smi_bits(encoded))
            }
        }
        become cold_numeric_immediate(pc, code, regs, acc, ctx)
    }

    ModImmediate : op_mod_immediate_n / op_mod_immediate_w (r => signed, imm => signed) {
        let lhs = reg!(r);
        if imm != 0
            && let Some(bits) = lhs.smi_bits()
        {
            next!(Tagged::from_smi_bits(((bits >> 1) % imm as i64) << 1))
        }
        become cold_numeric_immediate(pc, code, regs, acc, ctx)
    }

    ExpImmediate : op_exp_immediate_n / op_exp_immediate_w (r => signed, imm => signed) {
        become cold_numeric_immediate(pc, code, regs, acc, ctx)
    }

    BitwiseOrImmediate : op_bitwise_or_immediate_n / op_bitwise_or_immediate_w (r => signed, imm => signed) {
        let Some(a) = reg!(r).to_i64() else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32 | imm) as i64).into_tagged())
    }

    BitwiseXorImmediate : op_bitwise_xor_immediate_n / op_bitwise_xor_immediate_w (r => signed, imm => signed) {
        let Some(a) = reg!(r).to_i64() else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32 ^ imm) as i64).into_tagged())
    }

    BitwiseAndImmediate : op_bitwise_and_immediate_n / op_bitwise_and_immediate_w (r => signed, imm => signed) {
        let Some(a) = reg!(r).to_i64() else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32 & imm) as i64).into_tagged())
    }

    ShiftLeftImmediate : op_shift_left_immediate_n / op_shift_left_immediate_w (r => signed, imm => signed) {
        let Some(a) = reg!(r).to_i64() else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32).wrapping_shl(imm as u32 & 31) as i64).into_tagged())
    }

    ShiftRightImmediate : op_shift_right_immediate_n / op_shift_right_immediate_w (r => signed, imm => signed) {
        let Some(a) = reg!(r).to_i64() else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as i32).wrapping_shr(imm as u32 & 31) as i64).into_tagged())
    }

    ShiftRightLogicalImmediate : op_shift_right_logical_immediate_n / op_shift_right_logical_immediate_w (r => signed, imm => signed) {
        let Some(a) = reg!(r).to_i64() else {
            bail!(VmError::Type);
        };
        next!(Smi::new((a as u32).wrapping_shr(imm as u32 & 31) as i64).into_tagged())
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

    ReThrow : op_rethrow_n / op_rethrow_w () {
        // a finally handler re-emits the in-flight exception: same flow
        // as Throw (the match-loop treats them identically)
        ctx.state().set_pending_exception(acc);
        threw!()
    }

    Return : op_return_n / op_return_w () {
        if ctx.cache().base() == ctx.base_anchor() {
            return acc;
        }
        let meta = ctx.meta(pc);
        // Construct-result fixups are caller-side (`ConstructCheck`) and
        // cold paths apply their own, so the callee's return is
        // unconditional: pop, restore the caller, resume.
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
        let v = f(nctx, args);
        become resume(pc, code, regs, v, ctx)
    }

    Call : op_call_ic_n / op_call_ic_w (callee => signed, base => signed, count => unsigned, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_start(ctx, pc, SIZE, callee_word, base, count, fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_proxy_apply(pc, code, regs, acc, ctx),
        }
    }

    CallNoFeedback : op_call_n / op_call_w (callee => signed, base => signed, count => unsigned) {
        let callee_word = reg!(callee);
        match Object::call_target(ctx.heap(), callee_word) {
            None => bail!(VmError::Type),
            Some(CallTarget::Proxy(_)) => become cold_proxy_apply(pc, code, regs, acc, ctx),
            Some(CallTarget::Intrinsic(i)) => {
                become INTRINSICS[i.id()](pc, code, regs, acc, ctx)
            }
            Some(CallTarget::Runtime(idx)) => {
                let f = ctx.vm().runtime(RuntimeIndex(idx));
                let meta = ctx.meta(0);
                let args = ctx.stack().args(&meta, base, count);
                let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
                let v = f(nctx, args);
                become resume(pc, code, regs, v, ctx)
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
                let frame = match ctx.stack().push_frame(
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
                ) {
                    Ok(frame) => frame,
                    Err(err) => return ctx.raise_tag(err),
                };
                ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
                let callee_pc = ctx.cache().pc();
                let callee_base = ctx.code_ptr();
                let r = ctx.regs_ptr();
                let undefined = ctx.heap().known().undefined.as_tagged(ctx.heap()).erase();
                dispatch!(callee_pc, undefined, r, callee_base)
            }
        }
    }

    CallMethod0 : op_call_method0_n / op_call_method0_w (callee => signed, recv => signed, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_method_start(ctx, pc, SIZE, callee_word, &[recv], fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_call_method_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallMethod1 : op_call_method1_n / op_call_method1_w (callee => signed, recv => signed, arg0 => signed, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_method_start(ctx, pc, SIZE, callee_word, &[recv, arg0], fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_call_method_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallMethod2 : op_call_method2_n / op_call_method2_w (callee => signed, recv => signed, arg0 => signed, arg1 => signed, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_method_start(ctx, pc, SIZE, callee_word, &[recv, arg0, arg1], fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_call_method_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallFunction0 : op_call_function0_n / op_call_function0_w (callee => signed, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_function_start(ctx, pc, SIZE, callee_word, &[], fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_call_function_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallFunction1 : op_call_function1_n / op_call_function1_w (callee => signed, arg0 => signed, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_function_start(ctx, pc, SIZE, callee_word, &[arg0], fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_call_function_proxy(pc, code, regs, acc, ctx),
        }
    }

    CallFunction2 : op_call_function2_n / op_call_function2_w (callee => signed, arg0 => signed, arg1 => signed, fb => unsigned) {
        let callee_word = reg!(callee);
        let __mc = cold_start!(pc, code, regs, acc, ctx, call_function_start(ctx, pc, SIZE, callee_word, &[arg0, arg1], fb));
        match __mc {
            MethodCall::Value(v) => {
                become resume(pc, code, regs, v, ctx)
            }
            MethodCall::Frame(frame) => become call_trampoline(pc, code, regs, acc, ctx),

            MethodCall::Intrinsic(i) => become INTRINSICS[i.id()](pc, code, regs, acc, ctx),
            MethodCall::Proxy => become cold_call_function_proxy(pc, code, regs, acc, ctx),
        }
    }

    Construct : op_construct_n / op_construct_w (callee => signed, base => signed, count => unsigned, out => signed) {
        let __mc = cold_start!(pc, code, regs, acc, ctx, construct_start(ctx, pc, SIZE, regs, callee, base, count, out));
        match __mc {
            ConstructStart::Frame => become call_trampoline(pc, code, regs, acc, ctx),
            ConstructStart::Runtime(idx) => {
                // runtime ctor: tier-1 call with `new_target` = callee
                set_reg!(out, ctx.undefined_word());
                let v = dispatch_runtime_construct(ctx, idx, callee, base, count);
                become resume(pc, code, regs, v, ctx)
            }
            ConstructStart::Threw(()) => become throw_dispatch(pc, code, regs, acc, ctx),
            ConstructStart::Cold => {
                // the cold path applies its own result fixup (or is a
                // proxy trap); park undefined so the ConstructCheck that
                // follows never reads an uninitialized slot
                set_reg!(out, ctx.undefined_word());
                become cold_construct(pc, code, regs, acc, ctx)
            }
        }
    }

    // The hole in `out` marks a derived constructor (no synthesized
    // receiver): a primitive result is a TypeError; base constructors
    // fall back to the parked receiver.
    ConstructCheck : op_construct_check_n / op_construct_check_w (out => signed) {
        let v = if Convert::is_primitive(ctx.heap(), acc) {
            let recv = reg!(out);
            if recv == ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase() {
                bail!(VmError::Type);
            }
            recv
        } else {
            acc
        };
        next!(v)
    }

    LoadHole : op_load_hole_n / op_load_hole_w () {
        next!(ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase())
    }

    LoadNewTarget : op_load_new_target_n / op_load_new_target_w () {
        next!(ctx.stack().new_target_slot(&ctx.meta(0)).get(ctx.heap()))
    }

    LoadContext : op_load_context_n / op_load_context_w () {
        next!(ctx.stack().context_slot(&ctx.meta(0)).get(ctx.heap()))
    }

    ThrowReferenceErrorIfHole : op_throw_reference_error_if_hole_n / op_throw_reference_error_if_hole_w () {
        if acc == ctx.heap().known().the_hole.as_tagged(ctx.heap()).erase() {
            bail!(VmError::Reference)
        }
        next!(acc)
    }

    TestReferenceEqual : op_test_reference_equal_n / op_test_reference_equal_w (r => signed) {
        let other = reg!(r);
        next!(Convert::boolean(ctx.heap(), other == acc))
    }

    TestTypeof : op_test_typeof_n / op_test_typeof_w () {
        next!(Object::type_of(ctx.heap(), acc))
    }

    EqualStrict : op_equal_strict_n / op_equal_strict_w (r => signed) {
        let other = reg!(r);
        next!(Convert::boolean(ctx.heap(), Compare::strict_equal(ctx.heap(), acc, other)))
    }

    JumpIfNotUndefined : op_jump_if_not_undefined_n / op_jump_if_not_undefined_w (off => signed) {
        if acc != ctx.heap().known().undefined.as_tagged(ctx.heap()).erase() {
            jump!(off, acc)
        }
        next!(acc)
    }

    LessThanOrEqual : op_less_than_or_equal_n / op_less_than_or_equal_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
            next!(Convert::boolean(ctx.heap(), a <= b))
        }
        become cold_less_than_or_equal(pc, code, regs, acc, ctx)
    }

    CreateBareObjectLiteral : op_create_bare_object_n / op_create_bare_object_w () {
        become cold_create_bare_object(pc, code, regs, acc, ctx)
    }

    CreateBlockContext : op_create_block_context_n / op_create_block_context_w (count => unsigned) {
        become cold_create_block_context(pc, code, regs, acc, ctx)
    }

    LoadGlobalFast : op_load_global_fast_n / op_load_global_fast_w (name => unsigned, fb => unsigned) {
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

    LoadGlobalNoThrow : op_load_global_nothrow_n / op_load_global_nothrow_w (name => unsigned, fb => unsigned) {
        let global = ctx.heap().known().global_object.as_tagged(ctx.heap()).erase();
        if let Some(Hit::Value(v)) = InlineCache::try_load(
            ctx.heap(),
            ctx.cache().feedback_ref(ctx.heap()),
            fb,
            global,
        ) {
            next!(v)
        }
        become cold_global_load_nothrow(pc, code, regs, acc, ctx)
    }

    StoreGlobal : op_store_global_n / op_store_global_w (name => unsigned, fb => unsigned) {
        become cold_store_global(pc, code, regs, acc, ctx)
    }

    StoreNamedPropertyNoShadow : op_store_named_no_shadow_n / op_store_named_no_shadow_w (r => signed, name => unsigned, fb => unsigned) {
        become cold_store_named_no_shadow(pc, code, regs, acc, ctx)
    }

    StoreNamedPropertyNoShadowFast : op_store_named_no_shadow_fast_n / op_store_named_no_shadow_fast_w (r => signed, name => unsigned, fb => unsigned) {
        become cold_store_named_no_shadow(pc, code, regs, acc, ctx)
    }

    InstanceOf : op_instance_of_n / op_instance_of_w (r => signed) {
        become cold_instance_of(pc, code, regs, acc, ctx)
    }

    LoadCurrentClosure : op_load_current_closure_n / op_load_current_closure_w () {
        next!(ctx.stack().callable_slot(&ctx.meta(0)).get(ctx.heap()))
    }

    GreaterThanOrEqual : op_greater_than_or_equal_n / op_greater_than_or_equal_w (r => signed) {
        let other = reg!(r);
        if let (Some(a), Some(b)) = (acc.smi_bits(), other.smi_bits()) {
            next!(Convert::boolean(ctx.heap(), a >= b))
        }
        become cold_greater_than_or_equal(pc, code, regs, acc, ctx)
    }

    AddParent : op_add_parent_n / op_add_parent_w (r => signed, name => unsigned) {
        become cold_add_parent(pc, code, regs, acc, ctx)
    }

    LoadNamedPropertyFast : op_load_named_fast_n / op_load_named_fast_w (r => signed, name => unsigned, fb => unsigned) {
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
) -> Tagged<'a, Value> {
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
) -> Tagged<'a, Value> {
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
) -> Tagged<'a, Value> {
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
        // ES 20.2.5.8: a proxy receiver runs its `get` trap (with the
        // proxy as `this`) instead of a descriptor walk. No IC update:
        // traps are dynamic, a proxy map must never be cached as a field
        // handler.
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::get(vm, heap, state, recv, recv, name.erase()) {
                Ok(Coercion::Threw) => Ok(ctx.exception_word()),
                Ok(Coercion::Value(v)) => Ok(v),
                Err(err) => Err(err),
            };
        }
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
        let key = match Object::to_property_key(vm, heap, state, raw_key) {
            Ok(k) => k,
            Err(err) => return Err(err),
        };
        let Some(key) = key else {
            return Ok(ctx.exception_word());
        };
        let key = scope.handle(key);
        // ES 20.2.5.8 (keyed form): the `get` trap with the coerced key.
        // No IC update: traps are dynamic.
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::get(vm, heap, state, recv, recv, key.erase()) {
                Ok(Coercion::Threw) => Ok(ctx.exception_word()),
                Ok(Coercion::Value(v)) => Ok(v),
                Err(err) => Err(err),
            };
        }
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
        let key = match Object::to_property_key(vm, heap, state, raw_key) {
            Ok(k) => k,
            Err(err) => return Err(err),
        };
        let Some(key) = key else {
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
                // a throwing setter routes the sentinel into `resume` ->
                // `throw_dispatch`; the store never completes
                let v = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                if ctx.is_throw(v) {
                    return Ok(v);
                }
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
                if getter.as_tagged(heap) == heap.known().undefined.as_tagged(heap) {
                    // a setter-only accessor: the read yields undefined
                    return Ok(ctx.undefined_word());
                }
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

        // ES 20.2.5.10: a proxy receiver runs its `set` trap; no IC
        // update (dynamic traps must not be cached), and the probe is
        // skipped entirely so an armed site never matches a proxy map.
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::set(vm, heap, state, recv, name.erase(), value, recv)? {
                Coercion::Threw => Ok(ctx.exception_word()),
                Coercion::Value(_) => Ok(value.as_tagged(heap).erase()),
            };
        }

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
                    let v = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                    if ctx.is_throw(v) {
                        return Ok(v);
                    }
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
                let v = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                if ctx.is_throw(v) {
                    return Ok(v);
                }
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

unsafe fn construct_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    regs: *mut Register,
    callee: i32,
    base: i32,
    count: usize,
    out: i32,
) -> Result<ConstructStart, VmError> {
    let callee_word = reg_read(regs, ctx.heap(), callee);
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
    reg_write(regs, out, receiver);
    let callee_word = reg_read(regs, ctx.heap(), callee);
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
    let heap = ctx.heap_mut();
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
    ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
    Ok(ConstructStart::Frame)
}

#[inline(never)]
unsafe fn construct_receiver_fast<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
) -> Result<Option<Tagged<'a, Value>>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
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
unsafe extern "rust-preserve-none" fn enter_fresh_frame<'a>(
    _pc: usize,
    _code: *const u8,
    _regs: *mut Register,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let code = ctx.code_ptr();
    let regs = ctx.regs_ptr();
    let acc = ctx.undefined_word();
    let pc = ctx.cache().pc();
    let op = *code.add(pc) as usize;
    let h = TABLE_NARROW.0[op];
    become h(pc, code, regs, acc, ctx)
}

/// Resolve the (already reshaped) call and push a frame, dispatch a
/// runtime callee, or report an intrinsic/proxy for the caller to tail
/// into. `args[0]` is the receiver of the call being made.
#[cold]
#[inline(never)]
unsafe fn intrinsic_call_scattered<'a>(
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
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
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
unsafe fn intrinsic_call_contiguous<'a>(
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
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
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
    ($pc:ident, $next:ident, $code:ident, $regs:ident, $acc:ident, $ctx:ident, $out:expr, $proxy:expr) => {
        match $out {
            Ok(mc) => match mc {
                MethodCall::Frame(_) => become enter_fresh_frame($pc, $code, $regs, $acc, $ctx),
                MethodCall::Value(v) => become resume($pc, $code, $regs, v, $ctx),
                MethodCall::Proxy => {
                    let v = match $proxy {
                        Ok(v) => v,
                        Err(err) => $ctx.raise_tag(err),
                    };
                    become resume($pc, $code, $regs, v, $ctx)
                }
                MethodCall::Intrinsic(i) => {
                    become INTRINSICS[i.id()]($pc, $code, $regs, $acc, $ctx)
                }
            },
            Err(err) => $ctx.raise_tag(err),
        }
    };
}

#[rustc_align(32)]
unsafe extern "rust-preserve-none" fn intrinsic_function_call<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let op = unsafe { Opcode::from_byte_unchecked(*code.add(pc + base / 2)) };
    match op {
        // contiguous window `[f, thisArg, args...]`: element 0 rides the
        // window bottom, so dropping it is `count - 1` at the same base
        Opcode::Call | Opcode::CallNoFeedback => {
            let args_base = signed::read(pc, code, base + stride, stride);
            let count = unsigned::read(pc, code, base + 2 * stride, stride);
            if count == 0 {
                let _ = ctx.raise_tag(VmError::Type);
                become throw_dispatch(pc, code, regs, acc, ctx)
            }
            let f = reg_read(regs, ctx.heap(), args_base - count as i32 + 1);
            let size = next - pc;
            finish_intrinsic!(
                pc,
                next,
                code,
                regs,
                acc,
                ctx,
                intrinsic_call_contiguous(ctx, pc, size, f, args_base, count - 1),
                proxy_apply_cold(ctx, f, args_base, count - 1)
            )
        }
        // scattered: the receiver register holds the real target
        Opcode::CallMethod0 => {
            let recv = signed::read(pc, code, base + stride, stride);
            let f = reg_read(regs, ctx.heap(), recv);
            let size = next - pc;
            finish_intrinsic!(
                pc,
                next,
                code,
                regs,
                acc,
                ctx,
                intrinsic_call_scattered(ctx, pc, size, f, &[]),
                proxy_apply_regs_cold(ctx, f, &[])
            )
        }
        Opcode::CallMethod1 => {
            let recv = signed::read(pc, code, base + stride, stride);
            let arg0 = signed::read(pc, code, base + 2 * stride, stride);
            let f = reg_read(regs, ctx.heap(), recv);
            let size = next - pc;
            finish_intrinsic!(
                pc,
                next,
                code,
                regs,
                acc,
                ctx,
                intrinsic_call_scattered(ctx, pc, size, f, &[arg0]),
                proxy_apply_regs_cold(ctx, f, &[arg0])
            )
        }
        Opcode::CallMethod2 => {
            let recv = signed::read(pc, code, base + stride, stride);
            let arg0 = signed::read(pc, code, base + 2 * stride, stride);
            let arg1 = signed::read(pc, code, base + 3 * stride, stride);
            let f = reg_read(regs, ctx.heap(), recv);
            let size = next - pc;
            finish_intrinsic!(
                pc,
                next,
                code,
                regs,
                acc,
                ctx,
                intrinsic_call_scattered(ctx, pc, size, f, &[arg0, arg1]),
                proxy_apply_regs_cold(ctx, f, &[arg0, arg1])
            )
        }
        // `call` invoked with an undefined receiver (unbound): `this` is
        // not callable
        Opcode::CallFunction0 | Opcode::CallFunction1 | Opcode::CallFunction2 => {
            let _ = ctx.raise_tag(VmError::Type);
            become throw_dispatch(pc, code, regs, acc, ctx)
        }
        _ => {
            let _ = ctx.raise_tag(VmError::Type);
            become throw_dispatch(pc, code, regs, acc, ctx)
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
unsafe fn intrinsic_apply_call<'a>(
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
                return Ok(ApplyOut::Value(ctx.raise_tag(VmError::Type)));
            }
            let heap = ctx.heap_mut();
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
            ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
            Ok(ApplyOut::Frame(frame))
        }
        Some(CallTarget::Runtime(idx)) => {
            // the spread is rooted by the caller's scope staging
            let f = ctx.vm().runtime(RuntimeIndex(idx));
            let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
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
            let heap = ctx.heap_mut();
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
unsafe extern "rust-preserve-none" fn intrinsic_function_apply<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let next = cold_next_pc(pc, code);
    let op = unsafe { Opcode::from_byte_unchecked(*code.add(pc + base / 2)) };
    // recover (f, thisArg, argsArray) from the invoking call shape
    let (f, this_arg, array) = match op {
        Opcode::Call | Opcode::CallNoFeedback => {
            let args_base = signed::read(pc, code, base + stride, stride);
            let count = unsigned::read(pc, code, base + 2 * stride, stride);
            if count < 2 {
                // window must at least hold [f, thisArg]
                let _ = ctx.raise_tag(VmError::Type);
                become throw_dispatch(pc, code, regs, acc, ctx)
            }
            let f = reg_read(regs, ctx.heap(), args_base - count as i32 + 1);
            let this_arg = reg_read(regs, ctx.heap(), args_base - count as i32 + 2);
            let array = if count >= 3 {
                Some(reg_read(regs, ctx.heap(), args_base - count as i32 + 3))
            } else {
                None
            };
            (f, this_arg, array)
        }
        Opcode::CallMethod0 => {
            let recv = signed::read(pc, code, base + stride, stride);
            let f = reg_read(regs, ctx.heap(), recv);
            (f, ctx.undefined_word(), None)
        }
        Opcode::CallMethod1 => {
            let recv = signed::read(pc, code, base + stride, stride);
            let arg0 = signed::read(pc, code, base + 2 * stride, stride);
            let f = reg_read(regs, ctx.heap(), recv);
            (f, reg_read(regs, ctx.heap(), arg0), None)
        }
        Opcode::CallMethod2 => {
            let recv = signed::read(pc, code, base + stride, stride);
            let arg0 = signed::read(pc, code, base + 2 * stride, stride);
            let arg1 = signed::read(pc, code, base + 3 * stride, stride);
            let f = reg_read(regs, ctx.heap(), recv);
            (
                f,
                reg_read(regs, ctx.heap(), arg0),
                Some(reg_read(regs, ctx.heap(), arg1)),
            )
        }
        Opcode::CallFunction0 | Opcode::CallFunction1 | Opcode::CallFunction2 => {
            let _ = ctx.raise_tag(VmError::Type);
            become throw_dispatch(pc, code, regs, acc, ctx)
        }
        _ => {
            let _ = ctx.raise_tag(VmError::Type);
            become throw_dispatch(pc, code, regs, acc, ctx)
        }
    };
    let size = next - pc;
    match ctx
        .state()
        .handle_scope(|scope| -> Result<ApplyOut<'a>, VmError> {
            // Safety: fresh register reads, staged before the frame push
            // below can move anything.
            let staged = scope.stage(&spread_apply_args(ctx.heap(), this_arg, array));
            intrinsic_apply_call(ctx, pc, size, f, staged)
        }) {
        Ok(ApplyOut::Frame(_)) => become enter_fresh_frame(pc, code, regs, acc, ctx),
        Ok(ApplyOut::Value(v)) => become resume(pc, code, regs, v, ctx),
        Err(err) => ctx.raise_tag(err),
    }
}

/// The intrinsic table: builtins entered like bytecode handlers, by id.
static INTRINSICS: [Handler; Intrinsic::COUNT] = [
    intrinsic_function_call as Handler,
    intrinsic_function_apply as Handler,
];

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

/// Exception dispatch: walk the ovm frames from the faulting one, consulting
/// each frame's static handler table for a try range covering
/// `fault_pc`; on a hit, take the pending exception and tail-dispatch
/// into the handler with it in the accumulator. Pops frames until the
/// anchor; then the sentinel escapes to the `execute` caller with the
/// pending exception left set. A termination is uncatchable: no
/// handler may observe it, so it unwinds straight out.
#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn throw_dispatch<'a>(
    fault_pc: usize,
    _code: *const u8,
    _regs: *mut Register,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let mut pc = fault_pc;
    loop {
        let handled = if ctx.state().termination().is_some() {
            None
        } else {
            let meta = ctx.cache().frame_meta();
            let callable = ctx.stack().callable_slot(&meta).get(ctx.heap());
            callable
                .as_heap_object()
                .and_then(|obj| obj.as_ref().callable_info(ctx.heap()))
                .and_then(|info| info.handlers.get(ctx.heap()))
                .and_then(|handlers| handlers.as_ref().lookup(pc))
        };
        if let Some(handler_pc) = handled {
            let ex = ctx
                .state()
                .take_pending_exception_tagged(ctx.heap())
                .expect("pending exception must be set while unwinding");
            let code = ctx.code_ptr();
            let regs = ctx.regs_ptr();
            let op = unsafe { *code.add(handler_pc) } as usize;
            let h = TABLE_NARROW.0[op];
            become h(handler_pc, code, regs, ex, ctx)
        }
        if ctx.cache().base() == ctx.base_anchor() {
            return ctx.exception_word();
        }
        let meta = ctx.cache().frame_meta();
        let caller = ctx.stack().pop_frame(&meta);
        ctx.cache().load(ctx.stack(), caller, ctx.heap_mut());
        pc = caller.handler_pc;
    }
}

#[inline(never)]
unsafe extern "rust-preserve-none" fn call_trampoline<'a>(
    fault_pc: usize,
    _code: *const u8,
    _regs: *mut Register,
    _acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    // the pushed callee frame is the cache's current frame
    let frame = ctx.cache().frame_meta();
    let probe = 0u8;
    if ctx.stack_overflowed() {
        // drop the half-built callee frame and surface the overflow
        // through the sentinel channel
        let caller = ctx.stack().pop_frame(&frame);
        ctx.cache().load(ctx.stack(), caller, ctx.heap_mut());
        let _ = ctx.raise_tag(VmError::StackOverflow);
        let v = ctx.exception_word();
        become resume(fault_pc, core::ptr::null(), core::ptr::null_mut(), v, ctx);
    }
    // the cache already points at the callee (the push helper loaded it)
    let code = ctx.code_ptr();
    let pc0 = ctx.cache().pc();
    let regs = ctx.regs_ptr();
    let acc0 = ctx.stack().undefined_word(ctx.heap());
    let callee_ctx = unsafe { Ctx::child(ctx, frame.base) };
    let op = unsafe { *code.add(pc0) } as usize;
    let acc = unsafe { TABLE_NARROW.0[op](pc0, code, regs, acc0, &callee_ctx) };
    let caller = ctx.stack().pop_frame(&frame);
    ctx.cache().load(ctx.stack(), caller, ctx.heap_mut());
    if ctx.is_throw(acc) {
        become throw_dispatch(fault_pc, core::ptr::null(), core::ptr::null_mut(), acc, ctx)
    }
    let code = ctx.code_ptr();
    let regs = ctx.regs_ptr();
    let next = ctx.cache().pc();
    let op = unsafe { *code.add(next) } as usize;
    let h = TABLE_NARROW.0[op];
    become h(next, code, regs, acc, ctx)
}

#[inline(always)]
unsafe extern "rust-preserve-none" fn resume<'a>(
    fault_pc: usize,
    _code: *const u8,
    _regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    if ctx.is_throw(acc) {
        become throw_dispatch(fault_pc, _code, _regs, acc, ctx)
    }
    let code = ctx.code_ptr();
    let next = unsafe { cold_next_pc(fault_pc, code) };
    let regs = ctx.regs_ptr();
    let op = unsafe { *code.add(next) } as usize;
    let h = TABLE_NARROW.0[op];
    become h(next, code, regs, acc, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_add<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let lhs = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, add_cold(ctx, lhs, acc));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_numeric<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let lhs = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let op = unsafe { Opcode::from_byte_unchecked(*code.add(pc + (stride == 2) as usize)) };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::Sub => |a, b| a - b,
        Opcode::Mul => |a, b| a * b,
        Opcode::Mod => |a, b| a % b,
        Opcode::Exp => |a, b| a.powf(b),
        _ => |a, b| a / b,
    };
    let v = cold_try!(ctx, numeric_cold(ctx, lhs, acc, f));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_add_immediate<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let lhs = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let imm = Smi::new(signed::read(pc, code, base + stride, stride) as i64).into_tagged();
    let v = cold_try!(ctx, add_cold(ctx, lhs, imm));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_numeric_immediate<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let lhs = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let imm = Smi::new(signed::read(pc, code, base + stride, stride) as i64).into_tagged();
    let op = unsafe { Opcode::from_byte_unchecked(*code.add(pc + (stride == 2) as usize)) };
    let f: fn(f64, f64) -> f64 = match op {
        Opcode::SubImmediate => |a, b| a - b,
        Opcode::MulImmediate => |a, b| a * b,
        Opcode::ModImmediate => |a, b| a % b,
        Opcode::ExpImmediate => |a, b| a.powf(b),
        _ => |a, b| a / b,
    };
    let v = cold_try!(ctx, numeric_cold(ctx, lhs, imm, f));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_negate<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, negate_cold(ctx, acc));
    become resume(pc, code, regs, v, ctx)
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
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, incdec_cold(pc, code, regs, ctx, 1.0));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_dec_loc<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, incdec_cold(pc, code, regs, ctx, -1.0));
    become resume(pc, code, regs, v, ctx)
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
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, loc_op_cold(pc, code, regs, ctx, false));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_sub_loc<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let v = cold_try!(ctx, loc_op_cold(pc, code, regs, ctx, true));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_load_reg<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let key = reg_read(
        regs,
        ctx.heap(),
        signed::read(pc, code, base + stride, stride),
    );
    let fb = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(ctx, keyed_load_cold(ctx, recv, key, Some(fb)));
    become resume(pc, code, regs, v, ctx)
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
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, compare_op_cold(ctx, 0, acc, other));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_less_than<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, compare_op_cold(ctx, 2, acc, other));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_greater_than<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, compare_op_cold(ctx, 4, acc, other));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_compare_jump<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
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
        _ => match compare_cold(ctx, cmp, acc, other) {
            Ok(v) => v,
            Err(err) => return ctx.raise_tag(err),
        },
    };
    let Some(b) = b else {
        become resume(pc, code, regs, ctx.exception_word(), ctx)
    };
    let boolean = Convert::boolean(ctx.heap(), b);
    let dest = if b != (kind % 2 == 1) {
        jump_target(pc, off)
    } else {
        next
    };
    let code = ctx.code_ptr();
    let regs = ctx.regs_ptr();
    let op = unsafe { *code.add(dest) } as usize;
    let h = TABLE_NARROW.0[op];
    become h(dest, code, regs, boolean, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_named_load<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let name_idx = unsigned::read(pc, code, base + stride, stride);
    let fb_slot = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(ctx, named_load_cold(ctx, recv, name_idx, fb_slot));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_load<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let fb = unsigned::read(pc, code, base + stride, stride);
    let v = cold_try!(ctx, keyed_load_cold(ctx, recv, acc, Some(fb)));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_load_imm<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let idx = unsigned::read(pc, code, base + stride, stride);
    let v = cold_try!(ctx, keyed_load_imm_cold(ctx, recv, idx));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_store<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let key = reg_read(
        regs,
        ctx.heap(),
        signed::read(pc, code, base + stride, stride),
    );
    let fb = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(
        ctx,
        keyed_store_cold(ctx, recv, key, acc, Some(fb), StoreSemantics::Shadow)
    );
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_keyed_store_no_shadow<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let key = reg_read(
        regs,
        ctx.heap(),
        signed::read(pc, code, base + stride, stride),
    );
    let fb = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(
        ctx,
        keyed_store_cold(ctx, recv, key, acc, Some(fb), StoreSemantics::WriteThrough)
    );
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_global_load<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let name_idx = unsigned::read(pc, code, base, stride);
    let fb_slot = unsigned::read(pc, code, base + stride, stride);
    let v = cold_try!(ctx, global_load_cold(ctx, name_idx, fb_slot));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_store_named<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let name_idx = unsigned::read(pc, code, base + stride, stride);
    let fb_slot = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(ctx, store_named_cold(ctx, recv, name_idx, fb_slot, acc));
    become resume(pc, code, regs, v, ctx)
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

struct CallHit<'a> {
    target: Tagged<'a, Object>,
    info: Tagged<'a, CallableInfoObject>,
    context: Tagged<'a, Context>,
    register_count: usize,
    formal_min: usize,
    kind: FunctionKind,
}

enum CallProbe<'a> {
    Bytecode(CallHit<'a>),
    Runtime(usize),
    Intrinsic(vm_core::Intrinsic),
    Miss,
}

const CALL_TAG_RUNTIME: i64 = 1;

/// Decode the packed callable descriptor: `register_count | formal_min<<16
/// | kind<<32` (mirrors `Object::call_target`).
#[inline(always)]
fn decode_descriptor(desc: i64) -> (usize, usize, FunctionKind) {
    let desc = desc as u64;
    (
        (desc & 0xffff) as usize,
        ((desc >> 16) & 0xffff) as usize,
        FunctionKind::decode(((desc >> 32) & 0xf) as i64),
    )
}

#[inline(always)]
unsafe fn call_ic_probe<'a>(ctx: &Ctx<'a>, fb: usize, callee: Tagged<'a, Value>) -> CallProbe<'a> {
    let heap = ctx.heap();
    let Some(vector) = ctx.cache().feedback_ref(heap) else {
        return CallProbe::Miss;
    };
    let state = vector.as_ref().slot(fb).get(heap);
    if !state.raw().is_weak_ptr() || !state.ptr_eq(callee) {
        return CallProbe::Miss;
    }
    let payload = vector.as_ref().slot(fb + 1).get(heap);
    let raw = payload.raw();
    if let Some(tag) = Smi::decode(raw) {
        // runtime/intrinsic callee: decode the index off the object
        if tag.value() != CALL_TAG_RUNTIME {
            return CallProbe::Miss;
        }
        let Some(obj) = callee.as_heap_object() else {
            return CallProbe::Miss;
        };
        return match obj.as_ref().runtime_call_target(heap) {
            Some(CallTarget::Runtime(idx)) => CallProbe::Runtime(idx),
            Some(CallTarget::Intrinsic(i)) => CallProbe::Intrinsic(i),
            _ => CallProbe::Miss,
        };
    }

    let info: Tagged<'a, CallableInfoObject> = unsafe { core::mem::transmute(raw) };
    let (register_count, formal_min, kind) =
        decode_descriptor(info.as_ref().descriptor.to_smi_unchecked().value());
    // Safety: a bytecode callable's slots are `[info, context]` by layout.
    let obj: Tagged<'a, Object> = unsafe { core::mem::transmute(callee.raw()) };
    let context = obj.as_ref().slot(heap, 1).get(heap);
    let context: Tagged<'a, Context> = unsafe { core::mem::transmute(context.raw()) };
    CallProbe::Bytecode(CallHit {
        target: unsafe { core::mem::transmute(callee.raw()) },
        info,
        context,
        register_count,
        formal_min,
        kind,
    })
}

#[inline(always)]
unsafe fn call_ic_update<'a>(
    ctx: &Ctx<'a>,
    fb: usize,
    callee: Tagged<'a, Value>,
    info: Option<Tagged<'a, CallableInfoObject>>,
) {
    let heap = ctx.heap();
    let Some(vector) = ctx.cache().feedback_ref(heap) else {
        return;
    };
    let Some((state_slot, tag_slot)) = vector.as_ref().site(fb) else {
        return;
    };
    let host = vector.erase();
    if state_slot.is_cleared() {
        // a collected weak entry leaves the site free to become monomorphic
        // again
    } else {
        let state = state_slot.get(heap);
        if state.raw().is_strong_ptr() {
            let hole = heap.known().the_hole.as_tagged(heap).erase();
            if !state.ptr_eq(hole) {
                // megamorphic: leave it alone
                return;
            }
        } else if !state.ptr_eq(callee) {
            let same_code = match (state.as_strong().and_then(|c| c.as_heap_object()), info) {
                (Some(old), Some(info)) => old
                    .as_ref()
                    .callable_info(heap)
                    .is_some_and(|old_info| old_info.ptr_eq(info)),
                _ => false,
            };
            if !same_code {
                vector.as_ref().set_megamorphic(heap, fb);
                return;
            }
        }
    }
    state_slot.set_weak(heap, host, callee);
    match info {
        // bytecode: the payload IS the resolved info
        Some(info) => tag_slot.set_strong(heap, host, info.erase()),
        // runtime/intrinsic callee
        None => tag_slot.set(
            heap,
            host,
            Smi::new(CALL_TAG_RUNTIME).into_tagged().as_maybe_weak(),
        ),
    }
}

#[inline(always)]
unsafe fn dispatch_runtime_method<'a>(
    ctx: &Ctx<'a>,
    idx: usize,
    srcs: &[i32],
) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let (saved_top, args) = match ctx.stack().stage_args_regs(ctx.heap(), &meta, srcs) {
        Ok(staged) => staged,
        Err(err) => {
            let _ = ctx.raise(err);
            return ctx.exception_word();
        }
    };
    let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
    let v = f(nctx, args);
    ctx.stack().set_top(saved_top);
    v
}

#[inline(always)]
unsafe fn dispatch_runtime_function<'a>(
    ctx: &Ctx<'a>,
    idx: usize,
    args: &[i32],
) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let (saved_top, staged) = match ctx.stack().stage_function_args(ctx.heap(), &meta, args) {
        Ok(staged) => staged,
        Err(err) => {
            let _ = ctx.raise(err);
            return ctx.exception_word();
        }
    };
    let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
    let v = f(nctx, staged);
    ctx.stack().set_top(saved_top);
    v
}

#[inline(always)]
unsafe fn dispatch_runtime_construct<'a>(
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
            let _ = ctx.raise(err);
            return ctx.exception_word();
        }
    };
    // Safety: fresh register word, no allocation since the read.
    let callee = ctx.stack().reg(ctx.heap(), &meta, callee_reg);
    let raw = ctx.state().handle_scope(|scope| -> Value {
        let new_target = scope.handle(callee);
        let nctx = RuntimeContext::with_new_target(
            ctx.vm(),
            ctx.heap_mut(),
            ctx.state(),
            Some(new_target),
        );
        f(nctx, staged).raw()
    });
    ctx.stack().set_top(saved_top);
    // Safety: fresh result word, consumed before any allocation.
    unsafe { Tagged::<Value>::from_value_unchecked(raw) }
}

unsafe fn dispatch_runtime_contiguous<'a>(
    ctx: &Ctx<'a>,
    idx: usize,
    base: i32,
    count: usize,
) -> Tagged<'a, Value> {
    let f = ctx.vm().runtime(RuntimeIndex(idx));
    let meta = ctx.meta(0);
    let args = ctx.stack().args(&meta, base, count);
    let nctx = RuntimeContext::new(ctx.vm(), ctx.heap_mut(), ctx.state());
    f(nctx, args)
}

#[inline(always)]
unsafe fn push_scattered_method_frame<'a>(
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
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
unsafe fn push_function_frame<'a>(
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
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
unsafe fn push_contiguous_frame<'a>(
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
    let heap = ctx.heap_mut();
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
    ctx.cache().load(ctx.stack(), frame, ctx.heap_mut());
    Ok(MethodCall::Frame(frame))
}

#[inline(always)]
unsafe fn call_method_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    srcs: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match call_ic_probe(ctx, fb, callee_word) {
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
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
        CallProbe::Runtime(idx) => Ok(MethodCall::Value(dispatch_runtime_method(ctx, idx, srcs))),
        CallProbe::Intrinsic(i) => Ok(MethodCall::Intrinsic(i)),
        CallProbe::Miss => cold_call_method_miss(ctx, pc, size, callee_word, srcs, fb),
    }
}

#[cold]
#[inline(never)]
unsafe fn cold_call_method_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    srcs: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(ctx.raise_tag(VmError::Type))),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            call_ic_update(ctx, fb, callee_word, None);
            Ok(MethodCall::Value(dispatch_runtime_method(ctx, idx, srcs)))
        }
        Some(CallTarget::Intrinsic(i)) => {
            call_ic_update(ctx, fb, callee_word, None);
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
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
            }
            call_ic_update(ctx, fb, callee_word, Some(info));
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
unsafe fn call_function_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    args: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match call_ic_probe(ctx, fb, callee_word) {
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
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
        CallProbe::Runtime(idx) => Ok(MethodCall::Value(dispatch_runtime_function(ctx, idx, args))),
        CallProbe::Intrinsic(i) => Ok(MethodCall::Intrinsic(i)),
        CallProbe::Miss => cold_call_function_miss(ctx, pc, size, callee_word, args, fb),
    }
}

#[cold]
#[inline(never)]
unsafe fn cold_call_function_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    args: &[i32],
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(ctx.raise_tag(VmError::Type))),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            call_ic_update(ctx, fb, callee_word, None);
            Ok(MethodCall::Value(dispatch_runtime_function(ctx, idx, args)))
        }
        Some(CallTarget::Intrinsic(i)) => {
            call_ic_update(ctx, fb, callee_word, None);
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
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
            }
            call_ic_update(ctx, fb, callee_word, Some(info));
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
unsafe fn call_start<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    base: i32,
    count: usize,
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match call_ic_probe(ctx, fb, callee_word) {
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            register_count,
            formal_min,
            kind,
        }) => {
            if kind.is_class_constructor() {
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
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
        CallProbe::Runtime(idx) => Ok(MethodCall::Value(dispatch_runtime_contiguous(
            ctx, idx, base, count,
        ))),
        CallProbe::Intrinsic(i) => Ok(MethodCall::Intrinsic(i)),
        CallProbe::Miss => cold_call_miss(ctx, pc, size, callee_word, base, count, fb),
    }
}

#[cold]
#[inline(never)]
unsafe fn cold_call_miss<'a>(
    ctx: &Ctx<'a>,
    pc: usize,
    size: usize,
    callee_word: Tagged<'a, Value>,
    base: i32,
    count: usize,
    fb: usize,
) -> Result<MethodCall<'a>, VmError> {
    match Object::call_target(ctx.heap(), callee_word) {
        None => Ok(MethodCall::Value(ctx.raise_tag(VmError::Type))),
        Some(CallTarget::Proxy(_)) => Ok(MethodCall::Proxy),
        Some(CallTarget::Runtime(idx)) => {
            call_ic_update(ctx, fb, callee_word, None);
            Ok(MethodCall::Value(dispatch_runtime_contiguous(
                ctx, idx, base, count,
            )))
        }
        Some(CallTarget::Intrinsic(i)) => {
            call_ic_update(ctx, fb, callee_word, None);
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
                return Ok(MethodCall::Value(ctx.raise_tag(VmError::Type)));
            }
            call_ic_update(ctx, fb, callee_word, Some(info));
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
unsafe extern "rust-preserve-none" fn cold_construct<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let callee = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let args_base = signed::read(pc, code, base + stride, stride);
    let count = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(ctx, construct_cold(ctx, callee, args_base, count));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_closure<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let info_idx = unsigned::read(pc, code, base, stride);
    let v = cold_try!(ctx, create_closure_cold(ctx, info_idx));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_empty_array<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let heap = ctx.heap_mut();
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().js_array_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(pc, code, regs, obj, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_empty_object<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let heap = ctx.heap_mut();
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().object_initial_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(pc, code, regs, obj, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_bare_object<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let heap = ctx.heap_mut();
    let state = ctx.state();
    let obj = state.handle_scope(|scope| {
        let map = heap.known().plain_object_map;
        heap.new_object(&scope, map, HandleSlice::EMPTY).erase()
    });
    become resume(pc, code, regs, obj, ctx)
}

#[cold]
#[inline(never)]
unsafe fn create_block_context_cold<'a>(
    ctx: &Ctx<'a>,
    count: usize,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = ctx.heap_mut();
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
unsafe extern "rust-preserve-none" fn cold_create_block_context<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let count = unsigned::read(pc, code, base, stride);
    let v = cold_try!(ctx, create_block_context_cold(ctx, count));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_less_than_or_equal<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, compare_op_cold(ctx, 3, acc, other));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe fn global_load_nothrow_cold<'a>(
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
            Lookup::NotFound => Ok(heap.known().undefined.as_tagged(heap).erase()),
        }
    })
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_global_load_nothrow<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let name_idx = unsigned::read(pc, code, base, stride);
    let fb_slot = unsigned::read(pc, code, base + stride, stride);
    let v = cold_try!(ctx, global_load_nothrow_cold(ctx, name_idx, fb_slot));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe fn store_global_cold<'a>(
    ctx: &Ctx<'a>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let value = scope.handle(value);
        let name = scope.handle(
            ctx.cache()
                .constants_ref(heap)
                .at(heap, name_idx)
                .erase()
                .as_name(),
        );
        let global = heap.known().global_object;
        let outcome = global.as_tagged(heap).erase().store_lookup(
            heap,
            &scope,
            name.as_tagged(heap),
            value.as_tagged(heap),
            StoreSemantics::WriteThrough,
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
                let args = scope.stage(&[global.as_tagged(heap).erase(), value.as_tagged(heap)]);
                let v = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                if ctx.is_throw(v) {
                    return Ok(v);
                }
            }
            StoreOutcome::Done => {}
        }
        Ok(value.as_tagged(heap).erase())
    })
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_store_global<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let name_idx = unsigned::read(pc, code, base, stride);
    let v = cold_try!(ctx, store_global_cold(ctx, name_idx, acc));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe fn store_named_no_shadow_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
    let state = ctx.state();
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
        let outcome = recv.as_tagged(heap).erase().store_lookup(
            heap,
            &scope,
            name.as_tagged(heap),
            value.as_tagged(heap),
            StoreSemantics::WriteThrough,
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
                // a throwing setter routes the sentinel into `resume` ->
                // `throw_dispatch`; the store never completes
                let v = RuntimeContext::call(vm, heap, state, setter, args, None)?;
                if ctx.is_throw(v) {
                    return Ok(v);
                }
            }
            StoreOutcome::Done => {}
        }
        Ok(value.as_tagged(heap).erase())
    })
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_store_named_no_shadow<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let name_idx = unsigned::read(pc, code, base + stride, stride);
    let v = cold_try!(ctx, store_named_no_shadow_cold(ctx, recv, name_idx, acc));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe fn instance_of_cold<'a>(
    ctx: &Ctx<'a>,
    object: Tagged<'_, Value>,
    callable: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let vm = ctx.vm();
    let heap = ctx.heap_mut();
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
unsafe extern "rust-preserve-none" fn cold_instance_of<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let callable = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, instance_of_cold(ctx, acc, callable));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_greater_than_or_equal<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let other = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let v = cold_try!(ctx, compare_op_cold(ctx, 5, acc, other));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe fn add_parent_cold<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let heap = ctx.heap_mut();
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
unsafe extern "rust-preserve-none" fn cold_add_parent<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let recv = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let name_idx = unsigned::read(pc, code, base + stride, stride);
    let v = cold_try!(ctx, add_parent_cold(ctx, recv, name_idx, acc));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_create_function_context<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let scope_idx = unsigned::read(pc, code, base, stride);
    let v = cold_try!(ctx, create_function_context_cold(ctx, scope_idx));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_proxy_apply<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
    let callee = reg_read(regs, ctx.heap(), signed::read(pc, code, base, stride));
    let args_base = signed::read(pc, code, base + stride, stride);
    let count = unsigned::read(pc, code, base + 2 * stride, stride);
    let v = cold_try!(ctx, proxy_apply_cold(ctx, callee, args_base, count));
    become resume(pc, code, regs, v, ctx)
}

#[cold]
#[inline(never)]
unsafe extern "rust-preserve-none" fn cold_call_method_proxy<'a>(
    pc: usize,
    code: *const u8,
    regs: *mut Register,
    acc: Tagged<'a, Value>,
    ctx: &Ctx<'a>,
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
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
    let v = cold_try!(ctx, proxy_apply_regs_cold(ctx, callee, &srcs[..argc + 1]));
    become resume(pc, code, regs, v, ctx)
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
) -> Tagged<'a, Value> {
    let (base, stride) = cold_layout(pc, code);
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
    let v = cold_try!(ctx, proxy_apply_function_cold(ctx, callee, &args[..argc]));
    become resume(pc, code, regs, v, ctx)
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
    t[Opcode::Add as usize] = op_add_n as Handler;
    t[Opcode::Sub as usize] = op_sub_n as Handler;
    t[Opcode::Mul as usize] = op_mul_n as Handler;
    t[Opcode::Div as usize] = op_div_n as Handler;
    t[Opcode::Mod as usize] = op_mod_n as Handler;
    t[Opcode::Exp as usize] = op_exp_n as Handler;
    t[Opcode::BitwiseOr as usize] = op_bitwise_or_n as Handler;
    t[Opcode::BitwiseXor as usize] = op_bitwise_xor_n as Handler;
    t[Opcode::BitwiseAnd as usize] = op_bitwise_and_n as Handler;
    t[Opcode::ShiftLeft as usize] = op_shift_left_n as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right_n as Handler;
    t[Opcode::ShiftRightLogical as usize] = op_shift_right_logical_n as Handler;
    t[Opcode::AddImmediate as usize] = op_add_immediate_n as Handler;
    t[Opcode::SubImmediate as usize] = op_sub_immediate_n as Handler;
    t[Opcode::MulImmediate as usize] = op_mul_immediate_n as Handler;
    t[Opcode::DivImmediate as usize] = op_div_immediate_n as Handler;
    t[Opcode::ModImmediate as usize] = op_mod_immediate_n as Handler;
    t[Opcode::ExpImmediate as usize] = op_exp_immediate_n as Handler;
    t[Opcode::BitwiseOrImmediate as usize] = op_bitwise_or_immediate_n as Handler;
    t[Opcode::BitwiseXorImmediate as usize] = op_bitwise_xor_immediate_n as Handler;
    t[Opcode::BitwiseAndImmediate as usize] = op_bitwise_and_immediate_n as Handler;
    t[Opcode::ShiftLeftImmediate as usize] = op_shift_left_immediate_n as Handler;
    t[Opcode::ShiftRightImmediate as usize] = op_shift_right_immediate_n as Handler;
    t[Opcode::ShiftRightLogicalImmediate as usize] = op_shift_right_logical_immediate_n as Handler;
    t[Opcode::IncLoc as usize] = op_inc_loc_n as Handler;
    t[Opcode::DecLoc as usize] = op_dec_loc_n as Handler;
    t[Opcode::AddLoc as usize] = op_add_loc_n as Handler;
    t[Opcode::SubLoc as usize] = op_sub_loc_n as Handler;
    t[Opcode::LoadKeyedPropertyReg as usize] = op_load_keyed_reg_n as Handler;
    t[Opcode::Equal as usize] = op_equal_n as Handler;
    t[Opcode::LessThan as usize] = op_less_than_n as Handler;
    t[Opcode::GreaterThan as usize] = op_greater_than_n as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed_n as Handler;
    t[Opcode::StoreKeyedPropertyNoShadow as usize] = op_store_keyed_no_shadow_n as Handler;
    t[Opcode::Negate as usize] = op_negate_n as Handler;
    t[Opcode::CompareJump as usize] = op_compare_jump_n as Handler;
    t[Opcode::Jump as usize] = op_jump_n as Handler;
    t[Opcode::JumpIfTruthy as usize] = op_jump_if_truthy_n as Handler;
    t[Opcode::JumpIfFalsy as usize] = op_jump_if_falsy_n as Handler;
    t[Opcode::JumpLoop as usize] = op_jump_loop_n as Handler;
    t[Opcode::Throw as usize] = op_throw_n as Handler;
    t[Opcode::ReThrow as usize] = op_rethrow_n as Handler;
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
    t[Opcode::Call as usize] = op_call_ic_n as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call_n as Handler;
    t[Opcode::CallMethod0 as usize] = op_call_method0_n as Handler;
    t[Opcode::CallFunction0 as usize] = op_call_function0_n as Handler;
    t[Opcode::CallFunction1 as usize] = op_call_function1_n as Handler;
    t[Opcode::CallFunction2 as usize] = op_call_function2_n as Handler;
    t[Opcode::CallMethod1 as usize] = op_call_method1_n as Handler;
    t[Opcode::CallMethod2 as usize] = op_call_method2_n as Handler;
    t[Opcode::Construct as usize] = op_construct_n as Handler;
    t[Opcode::ConstructCheck as usize] = op_construct_check_n as Handler;
    t[Opcode::LoadHole as usize] = op_load_hole_n as Handler;
    t[Opcode::LoadNewTarget as usize] = op_load_new_target_n as Handler;
    t[Opcode::LoadContext as usize] = op_load_context_n as Handler;
    t[Opcode::ThrowReferenceErrorIfHole as usize] = op_throw_reference_error_if_hole_n as Handler;
    t[Opcode::TestReferenceEqual as usize] = op_test_reference_equal_n as Handler;
    t[Opcode::TestTypeof as usize] = op_test_typeof_n as Handler;
    t[Opcode::EqualStrict as usize] = op_equal_strict_n as Handler;
    t[Opcode::JumpIfNotUndefined as usize] = op_jump_if_not_undefined_n as Handler;
    t[Opcode::LessThanOrEqual as usize] = op_less_than_or_equal_n as Handler;
    t[Opcode::CreateBareObjectLiteral as usize] = op_create_bare_object_n as Handler;
    t[Opcode::CreateBlockContext as usize] = op_create_block_context_n as Handler;
    t[Opcode::LoadGlobalFast as usize] = op_load_global_fast_n as Handler;
    t[Opcode::LoadGlobalNoThrow as usize] = op_load_global_nothrow_n as Handler;
    t[Opcode::StoreGlobal as usize] = op_store_global_n as Handler;
    t[Opcode::StoreNamedPropertyNoShadow as usize] = op_store_named_no_shadow_n as Handler;
    t[Opcode::InstanceOf as usize] = op_instance_of_n as Handler;
    t[Opcode::LoadCurrentClosure as usize] = op_load_current_closure_n as Handler;
    t[Opcode::GreaterThanOrEqual as usize] = op_greater_than_or_equal_n as Handler;
    t[Opcode::AddParent as usize] = op_add_parent_n as Handler;
    t[Opcode::LoadNamedPropertyFast as usize] = op_load_named_fast_n as Handler;
    t[Opcode::StoreNamedPropertyNoShadowFast as usize] = op_store_named_no_shadow_fast_n as Handler;
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
    t[Opcode::Add as usize] = op_add_w as Handler;
    t[Opcode::Sub as usize] = op_sub_w as Handler;
    t[Opcode::Mul as usize] = op_mul_w as Handler;
    t[Opcode::Div as usize] = op_div_w as Handler;
    t[Opcode::Mod as usize] = op_mod_w as Handler;
    t[Opcode::Exp as usize] = op_exp_w as Handler;
    t[Opcode::BitwiseOr as usize] = op_bitwise_or_w as Handler;
    t[Opcode::BitwiseXor as usize] = op_bitwise_xor_w as Handler;
    t[Opcode::BitwiseAnd as usize] = op_bitwise_and_w as Handler;
    t[Opcode::ShiftLeft as usize] = op_shift_left_w as Handler;
    t[Opcode::ShiftRight as usize] = op_shift_right_w as Handler;
    t[Opcode::ShiftRightLogical as usize] = op_shift_right_logical_w as Handler;
    t[Opcode::AddImmediate as usize] = op_add_immediate_w as Handler;
    t[Opcode::SubImmediate as usize] = op_sub_immediate_w as Handler;
    t[Opcode::MulImmediate as usize] = op_mul_immediate_w as Handler;
    t[Opcode::DivImmediate as usize] = op_div_immediate_w as Handler;
    t[Opcode::ModImmediate as usize] = op_mod_immediate_w as Handler;
    t[Opcode::ExpImmediate as usize] = op_exp_immediate_w as Handler;
    t[Opcode::BitwiseOrImmediate as usize] = op_bitwise_or_immediate_w as Handler;
    t[Opcode::BitwiseXorImmediate as usize] = op_bitwise_xor_immediate_w as Handler;
    t[Opcode::BitwiseAndImmediate as usize] = op_bitwise_and_immediate_w as Handler;
    t[Opcode::ShiftLeftImmediate as usize] = op_shift_left_immediate_w as Handler;
    t[Opcode::ShiftRightImmediate as usize] = op_shift_right_immediate_w as Handler;
    t[Opcode::ShiftRightLogicalImmediate as usize] = op_shift_right_logical_immediate_w as Handler;
    t[Opcode::IncLoc as usize] = op_inc_loc_w as Handler;
    t[Opcode::DecLoc as usize] = op_dec_loc_w as Handler;
    t[Opcode::AddLoc as usize] = op_add_loc_w as Handler;
    t[Opcode::SubLoc as usize] = op_sub_loc_w as Handler;
    t[Opcode::LoadKeyedPropertyReg as usize] = op_load_keyed_reg_w as Handler;
    t[Opcode::Equal as usize] = op_equal_w as Handler;
    t[Opcode::LessThan as usize] = op_less_than_w as Handler;
    t[Opcode::GreaterThan as usize] = op_greater_than_w as Handler;
    t[Opcode::StoreKeyedProperty as usize] = op_store_keyed_w as Handler;
    t[Opcode::StoreKeyedPropertyNoShadow as usize] = op_store_keyed_no_shadow_w as Handler;
    t[Opcode::Negate as usize] = op_negate_w as Handler;
    t[Opcode::CompareJump as usize] = op_compare_jump_w as Handler;
    t[Opcode::Jump as usize] = op_jump_w as Handler;
    t[Opcode::JumpIfTruthy as usize] = op_jump_if_truthy_w as Handler;
    t[Opcode::JumpIfFalsy as usize] = op_jump_if_falsy_w as Handler;
    t[Opcode::JumpLoop as usize] = op_jump_loop_w as Handler;
    t[Opcode::Throw as usize] = op_throw_w as Handler;
    t[Opcode::ReThrow as usize] = op_rethrow_w as Handler;
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
    t[Opcode::Call as usize] = op_call_ic_w as Handler;
    t[Opcode::CallNoFeedback as usize] = op_call_w as Handler;
    t[Opcode::CallMethod0 as usize] = op_call_method0_w as Handler;
    t[Opcode::CallFunction0 as usize] = op_call_function0_w as Handler;
    t[Opcode::CallFunction1 as usize] = op_call_function1_w as Handler;
    t[Opcode::CallFunction2 as usize] = op_call_function2_w as Handler;
    t[Opcode::CallMethod1 as usize] = op_call_method1_w as Handler;
    t[Opcode::CallMethod2 as usize] = op_call_method2_w as Handler;
    t[Opcode::Construct as usize] = op_construct_w as Handler;
    t[Opcode::ConstructCheck as usize] = op_construct_check_w as Handler;
    t[Opcode::LoadHole as usize] = op_load_hole_w as Handler;
    t[Opcode::LoadNewTarget as usize] = op_load_new_target_w as Handler;
    t[Opcode::LoadContext as usize] = op_load_context_w as Handler;
    t[Opcode::ThrowReferenceErrorIfHole as usize] = op_throw_reference_error_if_hole_w as Handler;
    t[Opcode::TestReferenceEqual as usize] = op_test_reference_equal_w as Handler;
    t[Opcode::TestTypeof as usize] = op_test_typeof_w as Handler;
    t[Opcode::EqualStrict as usize] = op_equal_strict_w as Handler;
    t[Opcode::JumpIfNotUndefined as usize] = op_jump_if_not_undefined_w as Handler;
    t[Opcode::LessThanOrEqual as usize] = op_less_than_or_equal_w as Handler;
    t[Opcode::CreateBareObjectLiteral as usize] = op_create_bare_object_w as Handler;
    t[Opcode::CreateBlockContext as usize] = op_create_block_context_w as Handler;
    t[Opcode::LoadGlobalFast as usize] = op_load_global_fast_w as Handler;
    t[Opcode::LoadGlobalNoThrow as usize] = op_load_global_nothrow_w as Handler;
    t[Opcode::StoreGlobal as usize] = op_store_global_w as Handler;
    t[Opcode::StoreNamedPropertyNoShadow as usize] = op_store_named_no_shadow_w as Handler;
    t[Opcode::InstanceOf as usize] = op_instance_of_w as Handler;
    t[Opcode::LoadCurrentClosure as usize] = op_load_current_closure_w as Handler;
    t[Opcode::GreaterThanOrEqual as usize] = op_greater_than_or_equal_w as Handler;
    t[Opcode::AddParent as usize] = op_add_parent_w as Handler;
    t[Opcode::LoadNamedPropertyFast as usize] = op_load_named_fast_w as Handler;
    t[Opcode::StoreNamedPropertyNoShadowFast as usize] = op_store_named_no_shadow_fast_w as Handler;
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
            let regs = ctx.regs_ptr();
            let acc = cache.acc(ctx.heap());
            Ok(unsafe { TABLE_NARROW.0[*base.add(pc) as usize](pc, base, regs, acc, &ctx) })
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
