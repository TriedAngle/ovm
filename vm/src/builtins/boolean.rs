//! ES 20.3: the Boolean constructor and prototype methods.

use crate::{Convert, GcSlice, Value, VmError};
use super::helpers::wrapper_value;

pub(crate) fn boolean_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    let b = nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, arg));
    let value = Convert::boolean(nctx.heap(), b);
    if !nctx.is_construct() {
        return Ok(value);
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().boolean_wrapper_map;
        Ok(heap.new_object(&scope, map, &[value]).into_tagged().erase())
    })
}

pub(crate) fn boolean_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

pub(crate) fn boolean_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let v = wrapper_value(nctx.heap(), receiver)?;
    nctx.handle_scope(|nctx, scope| {
        let (_vm, heap, _) = nctx.split();
        Convert::to_string(heap, &scope, v)
    })
}
