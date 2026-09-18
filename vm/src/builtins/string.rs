//! ES 22.1: the String constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::RuntimeContext;
use crate::{Convert, HandleSlice, Tagged, Value, VmError};

pub fn string_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let is_construct = nctx.is_construct();
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let arg = match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(v) => scope.handle(v),
            None => scope.handle(heap.known().undefined.as_tagged(heap).erase()),
        };
        let s = scope.handle(Convert::to_string(heap, &scope, arg)?);
        if !is_construct {
            return Ok(s.as_tagged(heap).erase());
        }
        let map = heap.known().string_wrapper_map;
        Ok(heap
            .new_object(&scope, map, scope.stage(&[s.as_tagged(heap).erase()]))
            .erase())
    })
}

pub fn string_value_of<'a>(
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

pub fn string_to_string<'a>(
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
