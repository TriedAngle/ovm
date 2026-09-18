//! ES 21.1: the Number constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::Object;
use crate::RuntimeContext;
use crate::{Convert, HandleSlice, Smi, Tagged, Value, VmError};

pub fn number_constructor(
    nctx: &mut RuntimeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let n = nctx.handle_scope(|nctx, scope| {
        let (vm, heap, state) = nctx.split();
        let arg = match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(v) => scope.handle(v),
            None => scope.handle(Smi::new(0).into_tagged()),
        };
        Object::to_numeric(vm, heap, state, arg)
    })?;
    let Some(n) = n else {
        // Safety: fresh root-slot word read for the immediate return.
        return Ok(unsafe { nctx.heap().known().exception.read_unchecked() });
    };

    if !nctx.is_construct() {
        return nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, n).raw()));
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let value = heap.new_number(&scope, n).raw();
        let map = heap.known().number_wrapper_map;
        Ok(heap
            .new_object(
                &scope,
                map,
                scope.stage(&[unsafe { Tagged::<Value>::from_value_unchecked(value) }]),
            )
            .erase()
            .raw())
    })
}

pub fn number_value_of(
    nctx: &mut RuntimeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let heap = &*nctx.heap();
    let receiver = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    wrapper_value(heap, receiver)
}

pub fn number_to_string(
    nctx: &mut RuntimeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let v = {
        let heap = &*nctx.heap();
        let receiver = args
            .get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        wrapper_value(heap, receiver)
    }?;
    nctx.handle_scope(|nctx, scope| {
        let (_vm, heap, _) = nctx.split();
        // Safety: fresh word read above, consumed before any allocation.
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(v) });
        Convert::to_string(heap, &scope, v).map(|s| s.raw())
    })
}
