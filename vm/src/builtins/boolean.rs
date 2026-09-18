//! ES 20.3: the Boolean constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::RuntimeContext;
use crate::{Convert, HandleSlice, Tagged, Value, VmError};

pub fn boolean_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let is_construct = nctx.is_construct();
    let RuntimeContext { heap, state, .. } = nctx;
    if !is_construct {
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        return Ok(Convert::boolean(heap, Convert::is_truthy(heap, arg)));
    }
    state.handle_scope(|scope| {
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let value = scope.handle(Convert::boolean(heap, Convert::is_truthy(heap, arg)));
        let map = heap.known().boolean_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage(&[value.as_tagged(heap).erase()]))
            .erase())
    })
}

pub fn boolean_value_of<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    wrapper_value(
        heap,
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?,
    )
}

pub fn boolean_to_string<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let v = wrapper_value(
            heap,
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        )?;
        let v = scope.handle(v);
        Convert::to_string(heap, &scope, v)
    })
}
