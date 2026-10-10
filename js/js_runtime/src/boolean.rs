//! ES 20.3: the Boolean constructor and prototype methods.

use vm_core::HostCtx;
use vm_core::raise_runtime;
use vm_core::{Args, Convert, Handle, Tagged, Value, WrapperKind};

pub fn boolean_constructor<'a>(
    nctx: HostCtx<'a>,
    new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let is_construct = new_target.is_some();
    let HostCtx {
        vm: _, heap, state, ..
    } = nctx;
    if !is_construct {
        let arg = args.get(heap, 1);
        return Convert::boolean(heap, arg.is_truthy(heap));
    }
    state.handle_scope(|scope| {
        let arg = args.get(heap, 1);
        let value = scope.handle(Convert::boolean(heap, arg.is_truthy(heap)));
        let map = heap.known().boolean_wrapper_map;
        heap.new_object(&scope, map, scope.stage(&[value.as_tagged(heap).erase()]))
            .erase()
    })
}

pub fn boolean_value_of<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let arg = args.get(heap, 0);
    match unsafe { Tagged::<Value>::from_value_unchecked(arg.raw()) }
        .wrapper_value(heap, WrapperKind::Boolean)
        .map(|v| v.raw())
    {
        Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
        Err(err) => return raise_runtime(vm, heap, state, err),
    }
}

pub fn boolean_to_string<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let arg = args.get(heap, 0);
        let v = match unsafe { Tagged::<Value>::from_value_unchecked(arg.raw()) }
            .wrapper_value(heap, WrapperKind::Boolean)
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
