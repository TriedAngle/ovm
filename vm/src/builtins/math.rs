//! ES 22.1 (Math): function properties of the Math namespace object.

use crate::ContextState;
use crate::Heap;
use crate::Object;
use crate::RuntimeContext;
use crate::{HandleSlice, Tagged, Value, VM, VmError};

/// `Math.sqrt(x)` (ES 22.1.2.29): ToNumber, then the IEEE-754 square root
/// (NaN/negative input → NaN, ±0 → ±0).
pub fn math_sqrt<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = state.handle_scope(|scope| {
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let arg = scope.handle(arg);
        Object::to_numeric(vm, heap, state, arg)
    })?;
    let Some(n) = n else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };
    Ok(heap.new_number(n.sqrt()))
}

/// ToNumeric the argument at `i` (defaulting to NaN), or `None` when user
/// code threw (a pending exception holds the cause).
fn numeric_arg(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    args: &HandleSlice<'_>,
    i: usize,
) -> Result<Option<f64>, VmError> {
    state.handle_scope(|scope| {
        let arg = args
            .get(i)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let arg = scope.handle(arg);
        Object::to_numeric(vm, heap, state, arg)
    })
}

/// `Math.log(x)` (ES 22.1.2.15): natural logarithm.
pub fn math_log<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(x) = numeric_arg(vm, heap, state, &args, 1)? else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };
    Ok(heap.new_number(x.ln()))
}

/// `Math.pow(base, exponent)` (ES 22.1.2.20).
pub fn math_pow<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(base) = numeric_arg(vm, heap, state, &args, 1)? else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };
    let Some(exp) = numeric_arg(vm, heap, state, &args, 2)? else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };
    Ok(heap.new_number(base.powf(exp)))
}
