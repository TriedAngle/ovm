//! ES 20.3: the Boolean constructor and prototype methods.

use vm_core::RuntimeContext;
use vm_core::raise_runtime;
use vm_core::runtime_api::wrapper_value;
use vm_core::{Convert, HandleSlice, Tagged, Value, VmError};

pub fn boolean_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let is_construct = nctx.is_construct();
    let RuntimeContext {
        vm: _, heap, state, ..
    } = nctx;
    if !is_construct {
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        return Convert::boolean(heap, Convert::is_truthy(heap, arg));
    }
    state.handle_scope(|scope| {
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let value = scope.handle(Convert::boolean(heap, Convert::is_truthy(heap, arg)));
        let map = heap.known().boolean_wrapper_map;
        heap.new_object(&scope, map, scope.stage(&[value.as_tagged(heap).erase()]))
            .erase()
    })
}

pub fn boolean_value_of<'a>(nctx: RuntimeContext<'a>, args: HandleSlice<'_>) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(arg) = args.get(0) else {
        return raise_runtime(vm, heap, state, VmError::Arity);
    };
    match wrapper_value(
        heap,
        // Safety: fresh rooted-slot word, no allocation since the read.
        unsafe { Tagged::<Value>::from_value_unchecked(arg.raw()) },
    )
    .map(|v| v.raw())
    {
        Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
        Err(err) => return raise_runtime(vm, heap, state, err),
    }
}

pub fn boolean_to_string<'a>(nctx: RuntimeContext<'a>, args: HandleSlice<'_>) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let Some(arg) = args.get(0) else {
            return raise_runtime(vm, heap, state, VmError::Arity);
        };
        let v = match wrapper_value(
            heap,
            // Safety: fresh rooted-slot word, no allocation since the read.
            unsafe { Tagged::<Value>::from_value_unchecked(arg.raw()) },
        )
        .map(|v| v.raw())
        {
            Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        let v = scope.handle(v);
        match Convert::to_string(heap, &scope, v).map(|v| v.raw()) {
            Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
            Err(err) => return raise_runtime(vm, heap, state, err),
        }
    })
}
