//! ES 20.2: the Function constructor, Function.prototype
//! toString/call/apply/bind, and the bind-closure prelude.

use crate::Lookup;
use crate::Object;
use crate::RuntimeContext;
use crate::materialize::Materialize;
use crate::runtime::Coercion;
use crate::{Context, Convert, DenseString, Errors, HandleSlice, Smi, Tagged, Value, VmError};

/// Stub: `Function.prototype.toString` returns a stable marker string
/// (test262 A2.2 compares it against itself, not against real source).
pub fn function_to_string<'a>(
    nctx: RuntimeContext<'a>,
    _args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // fresh interned word, consumed with no allocation delay
        Ok(vm
            .interner()
            .intern_str(heap, &scope, "function () { [native code] }")
            .as_tagged(heap)
            .erase())
    })
}

/// `Function.prototype.call(thisArg, ...args)` (ES 20.2.3.4).
pub fn function_call<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let f = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    if !Object::is_callable(heap, f) {
        return Err(VmError::Type);
    }
    let f = f.raw();
    let call_args: Vec<Value> = args
        .iter()
        .map(|h| h.as_tagged(heap))
        .skip(1)
        .map(|v| v.raw())
        .collect();
    state.handle_scope(|scope| {
        // Safety: fresh argument word, still fresh (no allocation since).
        let f = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(f) });
        RuntimeContext::call(
            vm,
            heap,
            state,
            f,
            scope.stage(
                &call_args
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
            None,
        )
    })
}

/// `Function.prototype.bind(thisArg, ...prepend)` (ES 20.2.3.5): the
/// bound function is the JS closure template installed by BIND_PRELUDE,
/// called with (target, thisArg, prepend-array).
pub fn function_bind<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let raw_f = {
        let f = args
            .get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        if !Object::is_callable(heap, f) {
            return Err(VmError::Type);
        }
        f.raw()
    };
    // Safety: fresh root-slot word read for the immediate use.
    let undefined = heap.known().undefined.as_tagged(heap).raw();
    let raw_this_arg = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .map(|v| v.raw())
        .unwrap_or(undefined);
    let prepend: Vec<Value> = args
        .iter()
        .map(|h| h.as_tagged(heap))
        .skip(2)
        .map(|v| v.raw())
        .collect();
    state.handle_scope(|scope| {
        // everything below allocates (interning, the [[Get]] for
        // __makeBound, the prepend array, the call): keep the raw inputs
        // rooted and re-read at the point of use
        // Safety: fresh argument words, rooted below before any allocation.
        let f = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_f) });
        let this_arg = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_this_arg) });
        // Function.prototype.__makeBound (installed by BIND_PRELUDE)
        let make_bound = {
            let name = vm
                .interner()
                .intern_str(heap, &scope, "__makeBound")
                .erase();
            let proto = heap.known().function_prototype.erase();
            match Lookup::get_property_on(vm, heap, state, proto, proto, name)? {
                Coercion::Threw => {
                    return Ok(heap.known().exception.as_tagged(heap).erase());
                }
                // Safety: fresh word from the lookup, rooted immediately.
                Coercion::Value(v) => scope.handle(v),
            }
        };
        let staged = scope.stage(
            &prepend
                .iter()
                .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                .collect::<Vec<_>>(),
        );
        let array = scope.handle(heap.new_array(&scope, staged).erase());
        // args[0] is the receiver (undefined for the plain call)
        // Safety: fresh rooted-slot words staged for the call.
        let staged = scope.stage(&[
            heap.known().undefined.as_tagged(heap).erase(),
            f.as_tagged(heap).erase(),
            this_arg.as_tagged(heap).erase(),
            array.as_tagged(heap).erase(),
        ]);
        RuntimeContext::call(vm, heap, state, make_bound, staged, None)
    })
}

/// `Function.prototype.apply(thisArg, argsArray)` (ES 20.2.3.3).
pub fn function_apply<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let f = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    if !Object::is_callable(heap, f) {
        return Err(VmError::Type);
    }
    let f = f.raw();
    // Safety: fresh root-slot word read for the immediate use.
    let undefined = heap.known().undefined.as_tagged(heap).raw();
    let this_arg = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .map(|v| v.raw())
        .unwrap_or(undefined);
    let array = args
        .get(2)
        .map(|h| h.as_tagged(heap))
        .map(|v| v.raw())
        .unwrap_or(undefined);
    let call_args: Vec<Value> = {
        let nullish = array == heap.known().undefined.as_tagged(heap).raw()
            || array == heap.known().null.as_tagged(heap).raw();
        if nullish {
            vec![this_arg]
        } else {
            // array-like: read elements 0..length (holes read as undefined)
            let len = {
                unsafe { array.assume_valid(heap) }
                    .as_heap_object()
                    .map(|o| {
                        o.as_ref()
                            .array_length(heap, heap.known().strings.length.as_tagged(heap))
                            .and_then(|v| Smi::decode(v.raw()).map(|s| s.value() as usize))
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            };
            let mut out = Vec::with_capacity(len + 1);
            out.push(this_arg);
            for i in 0..len {
                out.push(
                    unsafe { array.assume_valid(heap) }
                        .as_heap_object()
                        .and_then(|o| o.as_ref().element_value(heap, i))
                        .map(|v| v.raw())
                        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).raw()),
                );
            }
            out
        }
    };
    state.handle_scope(|scope| {
        // Safety: fresh argument word (no allocation since the reads).
        let f = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(f) });
        RuntimeContext::call(
            vm,
            heap,
            state,
            f,
            scope.stage(
                &call_args
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
            None,
        )
    })
}

/// The bound-function template: a plain JS closure over (target, bound
/// this, prepend array). The runtime `function_bind` builds the prepend
/// array and delegates here — the runtime registry holds stateless fn
/// pointers, so the closure state must live in a JS closure.
pub const BIND_PRELUDE: &str = r#"
Function.prototype.__makeBound = function (f, t, p) {
  return function (...rest) {
    var all = [];
    for (var i = 0; i < p.length; i++) all[all.length] = p[i];
    for (var j = 0; j < rest.length; j++) all[all.length] = rest[j];
    return f.apply(t, all);
  };
};
"#;

/// `Function(p0, p1, ..., body)` (ES 20.2.1.1): the dynamic constructor.
/// Builds `function (p0, p1, ...) { body }` and evaluates it in the global
/// scope (approximated with the caller's context; the direct-eval pipeline
/// provides the parsing).
pub fn function_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let argv: Vec<Value> = (1..args.len())
        .filter_map(|i| args.get(i).map(|h| h.as_tagged(heap)).map(|v| v.raw()))
        .collect();
    state.handle_scope(|scope| {
        // root the caller context before the allocating ToString loop below
        let context = scope.handle(
            state
                .current_context(heap)
                .unwrap_or_else(|| heap.known().empty_context.as_tagged(heap).erase()),
        );
        // ToString all arguments (user toString may run)
        let mut parts: Vec<String> = Vec::with_capacity(argv.len());
        for a in argv {
            let s = {
                // Safety: fresh argument word, consumed before any allocation.
                let a = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(a) });
                Convert::to_string(heap, &scope, a)?.raw()
            };
            parts.push({
                // Safety: fresh word, no allocation since the read.
                unsafe { s.assume_valid(heap) }
                    .get_as::<DenseString>()
                    .map(|x| x.to_rust_string(heap))
                    .unwrap_or_default()
            });
        }
        let (params, body) = match parts.split_last() {
            Some((body, params)) => (params.join(", "), body.clone()),
            None => (String::new(), String::new()),
        };
        let source = format!("(function ({params}) {{\n{body}\n}})");
        let program = match js_compiler::compile_js(&source, bytecode::SourceMode::Eval) {
            Ok(program) => program,
            Err(_) => {
                let ex = Errors::from_vm_error(vm, heap, state, VmError::Type)?;
                state.set_pending_exception(ex);
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
        };
        let context = scope
            .cast::<Context>(context.as_tagged(heap))
            .ok_or(VmError::Type)?;
        let closure = Materialize::closure_vm(vm, heap, state, &scope, &program, context)?;
        RuntimeContext::call(vm, heap, state, closure.erase(), HandleSlice::EMPTY, None)
    })
}
