//! ES 21.1: the Number constructor and prototype methods.

use crate::{Convert, GcSlice, Smi, Value, VmError, runtime::Runtime};
use super::helpers::wrapper_value;

pub(crate) fn number_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(Smi::new(0).encode());
    let (vm, heap, state) = nctx.split();
    let n = match Runtime::to_numeric(vm, heap, state, arg)? {
        Some(n) => n,
        None => return Ok(nctx.heap().known().exception.value()),
    };
    if !nctx.is_construct() {
        return nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, n)));
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let value = heap.new_number(&scope, n);
        let map = heap.known().number_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage(&[value]))
            .into_tagged()
            .erase())
    })
}

pub(crate) fn number_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

pub(crate) fn number_to_string(
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
