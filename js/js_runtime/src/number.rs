//! ES 21.1: the Number constructor and prototype methods.

use vm_core::HostCtx;
use vm_core::Object;
use vm_core::{Args, Convert, DenseString, Handle, Heap, Smi, Tagged, Value, VmError, WrapperKind};
use vm_core::{raise_runtime, rt_try};

pub fn number_constructor<'a>(
    nctx: HostCtx<'a>,
    new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let is_construct = new_target.is_some();
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let n = rt_try!(
        vm,
        heap,
        state,
        state.handle_scope(|scope| {
            let arg = if args.len() > 1 {
                scope.handle(args.get(heap, 1))
            } else {
                scope.handle(Smi::new(0).into_tagged())
            };
            Object::to_numeric(vm, heap, state, arg)
        })
    );
    let Some(n) = n else {
        return heap.known().exception.as_tagged(heap).erase();
    };

    if !is_construct {
        return heap.new_number(n);
    }
    state.handle_scope(|scope| {
        let map = heap.known().number_wrapper_map;
        let value = scope.handle(heap.new_number(n));
        heap.new_object(&scope, map, scope.stage(&[value.as_tagged(heap)]))
            .erase()
    })
}

pub fn number_value_of<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    // The result crosses this call boundary, so root it: `wrapper_value`
    // borrows the heap immutably, and a handle releases that borrow before
    // the `TypeError` path needs the heap mutably.
    nctx.handle_scope(|vm, heap, state, scope| {
        let v = match Object::wrapper_value(heap, args.get(heap, 0), WrapperKind::Number) {
            Ok(v) => scope.handle(v),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        v.as_tagged(heap).erase()
    })
}

pub fn number_to_string<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    nctx.handle_scope(|vm, heap, state, scope| {
        // root the receiver: `Convert::to_string` allocates a fresh string
        let v = match Object::wrapper_value(heap, args.get(heap, 0), WrapperKind::Number) {
            Ok(v) => scope.handle(v),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        let s = match Convert::to_string(heap, &scope, v) {
            Ok(v) => scope.handle(v),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        s.as_tagged(heap).erase()
    })
}

/// The numeric `this` of a Number.prototype method (receiver or wrapper).
fn number_receiver(heap: &Heap, args: Args) -> Result<f64, VmError> {
    let v = Object::wrapper_value(heap, args.get(heap, 0), WrapperKind::Number)?;
    Convert::to_number(heap, v)
}

/// `Number.prototype.toFixed(fractionDigits?)` (ES 21.1.3.3): fixed-point
/// notation with `fractionDigits` digits after the decimal point.
pub fn number_to_fixed<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    nctx.handle_scope(|vm, heap, state, scope| {
        let x = rt_try!(vm, heap, state, number_receiver(heap, args));
        let digits = match (args.len() > 1).then(|| args.get(heap, 1)) {
            Some(d) if d != heap.known().undefined.as_tagged(heap) => {
                scope.handle(d);
                rt_try!(vm, heap, state, Convert::to_number(heap, d)) as i64
            }
            _ => 0,
        };        if !(0..=100).contains(&digits) {
            return raise_runtime(vm, heap, state, VmError::OutOfBounds);
        }
        let text = if x.is_nan() {
            "NaN".into()
        } else if x == f64::INFINITY {
            "Infinity".into()
        } else if x == f64::NEG_INFINITY {
            "-Infinity".into()
        } else {
            format!("{:.*}", digits as usize, x)
        };
        let s = DenseString::from_utf8(heap, &scope, &text);
        s.as_tagged(heap).erase()
    })
}

/// `Number.prototype.toPrecision(precision?)` (ES 21.1.3.6): `precision`
/// significant digits, fixed or exponential per the magnitude.
pub fn number_to_precision<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    nctx.handle_scope(|vm, heap, state, scope| {
        let x = rt_try!(vm, heap, state, number_receiver(heap, args));
        let arg = args.get(heap, 1);
        if arg == heap.known().undefined.as_tagged(heap) {
            let v = scope.handle(heap.new_number(x));
            let s = match Convert::to_string(heap, &scope, v) {
                Ok(v) => scope.handle(v),
                Err(err) => return raise_runtime(vm, heap, state, err),
            };
            return s.as_tagged(heap).erase();
        }
        let p = rt_try!(vm, heap, state, Convert::to_number(heap, arg)) as i64;
        if !(1..=100).contains(&p) {
            return raise_runtime(vm, heap, state, VmError::OutOfBounds);
        }
        let text = if x.is_nan() {
            "NaN".into()
        } else if x == f64::INFINITY {
            "Infinity".into()
        } else if x == f64::NEG_INFINITY {
            "-Infinity".into()
        } else if x == 0.0 {
            format!("{:.*}", p as usize - 1, 0.0)
        } else {
            let e = x.abs().log10().floor() as i64;
            if e < -6 || e >= p {
                // exponential: Rust writes 1.23e3; JS wants 1.23e+3
                let s = format!("{:.*e}", p as usize - 1, x);
                let (m, exp) = s.split_once('e').expect("exponent formatting");
                let exp: i64 = exp.parse().expect("decimal exponent");
                format!("{m}e{exp:+}")
            } else {
                format!("{:.*}", (p - 1 - e) as usize, x)
            }
        };
        let s = DenseString::from_utf8(heap, &scope, &text);
        s.as_tagged(heap).erase()
    })
}
