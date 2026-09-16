//! ES 22.1: the String constructor and prototype methods.

use crate::{Convert, GcSlice, Value, VmError};
use super::helpers::wrapper_value;

pub(crate) fn string_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    nctx.handle_scope(|nctx, scope| {
        let (_vm, heap, _) = nctx.split();
        let s = Convert::to_string(heap, &scope, arg)?;
        if !nctx.is_construct() {
            return Ok(s);
        }
        let (_, heap, _) = nctx.split();
        let map = heap.known().string_wrapper_map;
        Ok(heap.new_object(&scope, map, &[s]).into_tagged().erase())
    })
}

pub(crate) fn string_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

pub(crate) fn string_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}
