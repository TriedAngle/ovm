//! ES 21.1: the Number constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::{Convert, GcSlice, Smi, Tagged, Value, VmError, runtime::Runtime};

pub fn number_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = nctx
        .heap()
        .no_gc(|heap| args.get(heap, 1).map(|v| v.erase()))
        .unwrap_or_else(|| Smi::new(0).encode());
    let (vm, heap, state) = nctx.split();
    let n = match Runtime::to_numeric(vm, heap, state, arg)? {
        Some(n) => n,
        // Safety: fresh root-slot word read for the immediate return.
        None => return Ok(unsafe { nctx.heap().known().exception.read_unchecked() }),
    };
    if !nctx.is_construct() {
        return nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, n).erase()));
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let value = heap.new_number(&scope, n).erase();
        let map = heap.known().number_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage_words(&[value]))
            .erase_type()
            .erase())
    })
}

pub fn number_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.heap().no_gc(|heap| {
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        wrapper_value(heap, receiver)
    })
}

pub fn number_to_string(
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
