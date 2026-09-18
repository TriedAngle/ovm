//! ES 21.1: the Number constructor and prototype methods.

use super::helpers::wrapper_value;
use crate::Object;
use crate::RuntimeContext;
use crate::{Convert, HandleSlice, Smi, Tagged, Value, VmError};

pub fn number_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let is_construct = nctx.is_construct();
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = state.handle_scope(|scope| {
        let arg = match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(v) => scope.handle(v),
            None => scope.handle(Smi::new(0).into_tagged()),
        };
        Object::to_numeric(vm, heap, state, arg)
    })?;
    let Some(n) = n else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };

    if !is_construct {
        return Ok(heap.new_number(n));
    }
    state.handle_scope(|scope| {
        let map = heap.known().number_wrapper_map;
        let value = scope.handle(heap.new_number(n));
        Ok(heap
            .new_object(&scope, map, scope.stage(&[value.as_tagged(heap)]))
            .erase())
    })
}

pub fn number_value_of<'a>(
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

pub fn number_to_string<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let _ = vm;
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
