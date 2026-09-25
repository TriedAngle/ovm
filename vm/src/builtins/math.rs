//! ES 22.1 (Math): function properties of the Math namespace object.

use crate::Object;
use crate::RuntimeContext;
use crate::{HandleSlice, Tagged, Value, VmError};

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
