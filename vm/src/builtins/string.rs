//! ES 22.1: the String constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::natives::NativeContext;
use crate::{Convert, HandleSlice, Tagged, Value, VmError};

pub fn string_constructor(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let construct = nctx.is_construct();
        let (_vm, heap, _) = nctx.split();
        // Safety: fresh argument word, consumed before any allocation.
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .map(|v| v.raw())
            // Safety: fresh root-slot word read for the immediate use.
            .unwrap_or_else(|| unsafe { heap.known().undefined.read_unchecked() });
        let arg = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(arg) });
        let s = scope.handle(Convert::to_string(heap, &scope, arg)?);
        if !construct {
            return Ok(s.as_tagged(heap).raw());
        }
        let map = heap.known().string_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage(&[s.as_tagged(heap).erase()]))
            .erase()
            .raw())
    })
}

pub fn string_value_of(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let heap = &*nctx.heap();
    let receiver = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    wrapper_value(heap, receiver)
}

pub fn string_to_string(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let heap = &*nctx.heap();
    let receiver = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    wrapper_value(heap, receiver)
}
