//! ES 20.3: the Boolean constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::{Convert, GcSlice, Tagged, Value, VmError};

pub fn boolean_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let value = nctx.heap().no_gc(|heap| {
        let arg = args
            .get(heap, 1)
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase_type());
        Convert::boolean(heap, Convert::is_truthy(heap, arg)).erase()
    });
    if !nctx.is_construct() {
        return Ok(value);
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().boolean_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage_words(&[value]))
            .erase_type()
            .erase())
    })
}

pub fn boolean_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.heap().no_gc(|heap| {
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        wrapper_value(heap, receiver)
    })
}

pub fn boolean_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let v = nctx.heap().no_gc(|heap| {
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        wrapper_value(heap, receiver)
    })?;
    nctx.handle_scope(|nctx, scope| {
        let (_vm, heap, _) = nctx.split();
        // Safety: fresh word read above, consumed before any allocation.
        let v = unsafe { Tagged::<Value>::from_value_unchecked(v) };
        Convert::to_string(heap, &scope, v).map(|s| s.erase())
    })
}
