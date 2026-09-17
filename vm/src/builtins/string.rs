//! ES 22.1: the String constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::{Convert, GcSlice, Tagged, Value, VmError};

pub fn string_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let construct = nctx.is_construct();
        let (_vm, heap, _) = nctx.split();
        // Safety: fresh argument word, consumed before any allocation.
        let arg = heap
            .no_gc(|heap| args.get(heap, 1).map(|v| v.erase()))
            // Safety: fresh root-slot word read for the immediate use.
            .unwrap_or_else(|| unsafe { heap.known().undefined.read_unchecked() });
        let arg = unsafe { Tagged::<Value>::from_value_unchecked(arg) };
        let s = Convert::to_string(heap, &scope, arg)?;
        let s = s.erase();
        if !construct {
            return Ok(s);
        }
        let map = heap.known().string_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage_words(&[s]))
            .erase_type()
            .erase())
    })
}

pub fn string_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.heap().no_gc(|heap| {
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        wrapper_value(heap, receiver)
    })
}

pub fn string_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.heap().no_gc(|heap| {
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        wrapper_value(heap, receiver)
    })
}
