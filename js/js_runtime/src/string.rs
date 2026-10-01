//! ES 22.1: the String constructor and prototype methods.

use vm_core::Object;
use vm_core::RuntimeContext;
use vm_core::raise_runtime;
use vm_core::runtime_api::wrapper_value;
use vm_core::{DenseString, HandleSlice, Symbol, Tagged, Value, VmError};

pub fn string_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let is_construct = nctx.is_construct();
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let arg = match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(v) => scope.handle(v),
            None => scope.handle(heap.known().undefined.as_tagged(heap).erase()),
        };
        // `String(sym)` (call, not NewTarget) is SymbolDescriptiveString
        // (ES 22.1.1.1 step 2.a): "Symbol(" + [[Description]] + ")".
        if !is_construct {
            if let Some(sym) = arg.as_tagged(heap).get_as::<Symbol>() {
                let mut text = String::from("Symbol(");
                text.push_str(&String::from_utf8_lossy(sym.as_ref().description(heap)));
                text.push(')');
                let s = DenseString::from_utf8(heap, &scope, &text);
                return s.as_tagged(heap).erase();
            }
        }
        let s = match Object::to_string(vm, heap, state, arg) {
            Ok(Some(s)) => scope.handle(s),
            Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        if !is_construct {
            return s.as_tagged(heap).erase();
        }
        let map = heap.known().string_wrapper_map;
        heap.new_object(&scope, map, scope.stage(&[s.as_tagged(heap).erase()]))
            .erase()
    })
}

pub fn string_value_of<'a>(nctx: RuntimeContext<'a>, args: HandleSlice<'_>) -> Tagged<'a, Value> {
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

pub fn string_to_string<'a>(nctx: RuntimeContext<'a>, args: HandleSlice<'_>) -> Tagged<'a, Value> {
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
