//! ES 22.1 (Math): function properties of the Math namespace object.

use std::sync::atomic::{AtomicU64, Ordering};

use vm_core::Object;
use vm_core::RuntimeContext;
use vm_core::rt_try;
use vm_core::{Args, ContextState, Convert, Handle, Heap, Tagged, VM, Value, VmError};

/// TODO: better distribution
/// xorshift64
static RANDOM_STATE: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);

fn random_bits() -> u64 {
    let mut x = RANDOM_STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    RANDOM_STATE.store(x, Ordering::Relaxed);
    x
}

/// The inline half of an argument read: a value that already is a number
/// (a Smi, or a boxed `Float`) needs no ToPrimitive/ToNumber round-trip.
#[inline]
fn number_arg(heap: &Heap, args: Args, i: usize) -> Option<f64> {
    Convert::as_number(heap, args.get(heap, i))
}

/// The cold half of an argument read: full ToNumeric for strings, objects,
/// `undefined`, etc. `None` means user code threw (the pending exception
/// holds the cause).
#[cold]
#[inline(never)]
fn numeric_arg_cold(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    args: Args,
    i: usize,
) -> Result<Option<f64>, VmError> {
    state.handle_scope(|scope| {
        let arg = scope.handle(args.get(heap, i));
        Object::to_numeric(vm, heap, state, arg)
    })
}

/// An argument as an f64: the inlinable number fast path, falling back to
/// the cold ToNumeric body. `Ok(None)` is a pending exception.
#[inline]
fn arg_as_f64(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    args: Args,
    i: usize,
) -> Result<Option<f64>, VmError> {
    if let Some(x) = number_arg(heap, args, i) {
        return Ok(Some(x));
    }
    numeric_arg_cold(vm, heap, state, args, i)
}

/// Shared body of the unary Math functions: coerce the sole argument and
/// apply `f` (`Math.abs`/`sqrt`/`log`).
#[inline]
fn math_unary<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    args: Args,
    f: impl Fn(f64) -> f64,
) -> Tagged<'a, Value> {
    let Some(x) = rt_try!(vm, heap, state, arg_as_f64(vm, heap, state, args, 1)) else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    heap.new_number(f(x))
}

/// Shared body of the integer-preserving unaries (`floor`/`ceil`/`trunc`/
/// `round`): a Smi is already integral, so it comes straight back out with
/// no f64 round-trip and no heap traffic.
#[inline]
fn math_integral<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    args: Args,
    f: impl Fn(f64) -> f64,
) -> Tagged<'a, Value> {
    if let Some(bits) = args.get(heap, 1).smi_bits() {
        return Tagged::from_smi_bits(bits).erase();
    }
    math_unary(vm, heap, state, args, f)
}

/// `Math.sqrt(x)` (ES 22.1.2.29): ToNumber, then the IEEE-754 square root
/// (NaN/negative input → NaN, ±0 → ±0).
pub fn math_sqrt<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_unary(vm, heap, state, args, f64::sqrt)
}

/// `Math.log(x)` (ES 22.1.2.15): natural logarithm.
pub fn math_log<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_unary(vm, heap, state, args, f64::ln)
}

/// `Math.pow(base, exponent)` (ES 22.1.2.20).
pub fn math_pow<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(base) = rt_try!(vm, heap, state, arg_as_f64(vm, heap, state, args, 1)) else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    let Some(exp) = rt_try!(vm, heap, state, arg_as_f64(vm, heap, state, args, 2)) else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    heap.new_number(base.powf(exp))
}

/// `Math.round(x)` (ES 22.1.2.24): ties round toward +∞ (`-0.5` → `-0`),
/// so `f64::round`'s half-away-from-zero is not usable directly.
fn js_round(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    if (-0.5..0.0).contains(&x) {
        return -0.0;
    }
    (x + 0.5).floor()
}

/// `Math.abs(x)` (ES 22.1.2.1).
pub fn math_abs<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_unary(vm, heap, state, args, f64::abs)
}

/// `Math.floor(x)` (ES 22.1.2.11).
pub fn math_floor<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_integral(vm, heap, state, args, f64::floor)
}

/// `Math.ceil(x)` (ES 22.1.2.2).
pub fn math_ceil<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_integral(vm, heap, state, args, f64::ceil)
}

/// `Math.trunc(x)` (ES 22.1.2.31).
pub fn math_trunc<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_integral(vm, heap, state, args, f64::trunc)
}

/// `Math.round(x)` (ES 22.1.2.24).
pub fn math_round<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    math_integral(vm, heap, state, args, js_round)
}

/// Shared body of `Math.min`/`Math.max`: ToNumber every argument; any NaN
/// argument (or no argument at all) decides the result per spec.
#[inline]
fn math_minmax<'a>(nctx: RuntimeContext<'a>, args: Args, min: bool) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let mut result = if min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let mut i = 1;
    while i < args.len() {
        let Some(x) = rt_try!(vm, heap, state, arg_as_f64(vm, heap, state, args, i)) else {
            return heap.known().exception.as_tagged(heap).erase();
        };
        if x.is_nan() {
            result = f64::NAN;
        } else if min {
            if x < result || (x == 0.0 && result == 0.0 && x.is_sign_negative()) {
                result = x;
            }
        } else if x > result || (x == 0.0 && result == 0.0 && x.is_sign_positive()) {
            result = x;
        }
        i += 1;
    }
    heap.new_number(result)
}

/// `Math.min(...args)` (ES 22.1.2.14).
pub fn math_min<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    math_minmax(nctx, args, true)
}

/// `Math.max(...args)` (ES 22.1.2.13).
pub fn math_max<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    math_minmax(nctx, args, false)
}

/// `Math.random()` (ES 22.1.2.22): a number in `[0, 1)` built from the top
/// 53 random bits.
pub fn math_random<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    _args: Args,
) -> Tagged<'a, Value> {
    let bits = random_bits() >> 11;
    nctx.heap
        .new_number((bits as f64) * (1.0 / 9_007_199_254_740_992.0))
}
