//! ES 21.1: the Number constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::Heap;
use crate::Object;
use crate::RuntimeContext;
use crate::{Convert, DenseString, HandleSlice, Smi, Tagged, Value, VmError};

pub fn number_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let is_construct = nctx.is_construct();
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = state.handle_scope(|scope| {
        let arg = match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(v) => scope.handle(v),
            None => scope.handle(Smi::new(0).into_tagged()),
        };
        Object::to_numeric(vm, heap, state, arg)
    })?;
    let Some(n) = n else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };

    if !is_construct {
        return Ok(heap.new_number(n));
    }
    state.handle_scope(|scope| {
        let map = heap.known().number_wrapper_map;
        let value = scope.handle(heap.new_number(n));
        Ok(heap
            .new_object(&scope, map, scope.stage(&[value.as_tagged(heap)]))
            .erase())
    })
}

pub fn number_value_of<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    wrapper_value(
        heap,
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?,
    )
}

pub fn number_to_string<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let _ = vm;
    state.handle_scope(|scope| {
        let v = wrapper_value(
            heap,
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        )?;
        let v = scope.handle(v);
        Convert::to_string(heap, &scope, v)
    })
}

/// The numeric `this` of a Number.prototype method (receiver or wrapper).
fn number_receiver(heap: &Heap, args: &HandleSlice<'_>) -> Result<f64, VmError> {
    let v = wrapper_value(
        heap,
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?,
    )?;
    Convert::to_number(heap, v)
}

/// `Number.prototype.toFixed(fractionDigits?)` (ES 21.1.3.3): fixed-point
/// notation with `fractionDigits` digits after the decimal point.
pub fn number_to_fixed<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let x = number_receiver(heap, &args)?;
        let digits = match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(d) if d != heap.known().undefined.as_tagged(heap) => {
                scope.handle(d);
                Convert::to_number(heap, d)? as i64
            }
            _ => 0,
        };
        if !(0..=100).contains(&digits) {
            return Err(VmError::OutOfBounds);
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
        Ok(s.as_tagged(heap).erase())
    })
}

/// `Number.prototype.toPrecision(precision?)` (ES 21.1.3.6): `precision`
/// significant digits, fixed or exponential per the magnitude.
pub fn number_to_precision<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let x = number_receiver(heap, &args)?;
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        if arg == heap.known().undefined.as_tagged(heap) {
            let v = scope.handle(heap.new_number(x));
            return Convert::to_string(heap, &scope, v);
        }
        let p = Convert::to_number(heap, arg)? as i64;
        if !(1..=100).contains(&p) {
            return Err(VmError::OutOfBounds);
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
        Ok(s.as_tagged(heap).erase())
    })
}
