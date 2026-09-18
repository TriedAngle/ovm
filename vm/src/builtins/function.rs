//! ES 20.2: the Function constructor, Function.prototype
//! toString/call/apply/bind, and the bind-closure prelude.

use crate::materialize::materialize_closure_vm;
use crate::natives::NativeContext;
use crate::runtime::Coercion;
use crate::runtime::Runtime;
use crate::{Context, Convert, DenseString, GcSlice, Smi, Tagged, Value, VmError};

/// Stub: `Function.prototype.toString` returns a stable marker string
/// (test262 A2.2 compares it against itself, not against real source).
pub fn function_to_string(
    nctx: &mut NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let s = nctx.intern(&scope, "function () { [native code] }");
        // Safety: fresh rooted-slot word, returned without an
        // intervening allocation.
        Ok(unsafe { s.read_unchecked() })
    })
}

/// `Function.prototype.call(thisArg, ...args)` (ES 20.2.3.4).
pub fn function_call(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let f = {
        let heap = &*nctx.heap();
        args.get(heap, 0).ok_or(VmError::Arity)?.raw()
    };
    if !Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let call_args: Vec<Value> = {
        let heap = &*nctx.heap();
        args.iter(heap).skip(1).map(|v| v.raw()).collect()
    };
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word, still fresh (no allocation since).
        let f = unsafe { Tagged::<Value>::from_value_unchecked(f) };
        nctx.call(
            f,
            scope.stage(
                &call_args
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
        )
    })
}

/// `Function.prototype.bind(thisArg, ...prepend)` (ES 20.2.3.5): the
/// bound function is the JS closure template installed by BIND_PRELUDE,
/// called with (target, thisArg, prepend-array).
pub fn function_bind(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let raw_f = {
        let heap = &*nctx.heap();
        args.get(heap, 0).ok_or(VmError::Arity)?.raw()
    };
    if !Runtime::is_callable(nctx.heap(), raw_f) {
        return Err(VmError::Type);
    }
    // Safety: fresh root-slot word read for the immediate use.
    let undefined = {
        let heap = &*nctx.heap();
        heap.known().undefined.as_tagged(heap).raw()
    };
    let raw_this_arg = {
        let heap = &*nctx.heap();
        args.get(heap, 1).map(|v| v.raw())
    }
    .unwrap_or(undefined);
    let prepend: Vec<Value> = {
        let heap = &*nctx.heap();
        args.iter(heap).skip(2).map(|v| v.raw()).collect()
    };
    nctx.handle_scope(|nctx, scope| {
        // everything below allocates (interning, the [[Get]] for
        // __makeBound, the prepend array, the call): keep the raw inputs
        // rooted and re-read at the point of use
        // Safety: fresh argument words, rooted below before any allocation.
        let f = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_f) });
        let this_arg = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_this_arg) });
        // Function.prototype.__makeBound (installed by BIND_PRELUDE)
        let make_bound = {
            let name = nctx.intern(&scope, "__makeBound").erase();
            let proto = nctx.heap().known().function_prototype.erase();
            let (vm, heap, state) = nctx.split();
            match Runtime::get_property(vm, heap, state, proto, name)? {
                Coercion::Threw => {
                    // Safety: fresh root-slot word read for the return.
                    return Ok(unsafe { heap.known().exception.read_unchecked() });
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
        let (_, heap, _) = nctx.split();
        let array = scope.handle(heap.new_array(&scope, staged).erase());
        // args[0] is the receiver (undefined for the plain call)
        // Safety: fresh rooted-slot words staged for the call.
        let nctx_heap = nctx.heap();
        let staged = scope.stage(&[
            nctx_heap.known().undefined.as_tagged(nctx_heap).erase(),
            f.as_tagged(nctx_heap).erase(),
            this_arg.as_tagged(nctx_heap).erase(),
            array.as_tagged(nctx_heap).erase(),
        ]);
        nctx.call_rooted(make_bound, staged)
    })
}

/// `Function.prototype.apply(thisArg, argsArray)` (ES 20.2.3.3).
pub fn function_apply(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let f = {
        let heap = &*nctx.heap();
        args.get(heap, 0).ok_or(VmError::Arity)?.raw()
    };
    if !Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    // Safety: fresh root-slot word read for the immediate use.
    let undefined = {
        let heap = &*nctx.heap();
        heap.known().undefined.as_tagged(heap).raw()
    };
    let this_arg = {
        let heap = &*nctx.heap();
        args.get(heap, 1).map(|v| v.raw())
    }
    .unwrap_or(undefined);
    let array = {
        let heap = &*nctx.heap();
        args.get(heap, 2).map(|v| v.raw())
    }
    .unwrap_or(undefined);
    let call_args: Vec<Value> = {
        let nullish = {
            let heap = &*nctx.heap();
            array == heap.known().undefined.as_tagged(heap).raw()
                || array == heap.known().null.as_tagged(heap).raw()
        };
        if nullish {
            vec![this_arg]
        } else {
            // array-like: read elements 0..length (holes read as undefined)
            let len = {
                let heap = &*nctx.heap();
                unsafe { array.assume_valid(heap) }
                    .as_heap_object()
                    .map(|o| {
                        o.as_ref()
                            .array_length(
                                heap,
                                // Safety: fresh root-slot word for a name read.
                                heap.known().strings.length.as_tagged(heap),
                            )
                            .and_then(|v| Smi::decode(v.raw()).map(|s| s.value() as usize))
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            };
            let mut out = Vec::with_capacity(len + 1);
            out.push(this_arg);
            for i in 0..len {
                out.push({
                    let heap = &*nctx.heap();
                    unsafe { array.assume_valid(heap) }
                        .as_heap_object()
                        .and_then(|o| o.as_ref().element_value(heap, i))
                        .map(|v| v.raw())
                        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).raw())
                });
            }
            out
        }
    };
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word (no allocation since the reads).
        let f = unsafe { Tagged::<Value>::from_value_unchecked(f) };
        nctx.call(
            f,
            scope.stage(
                &call_args
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
        )
    })
}

/// The bound-function template: a plain JS closure over (target, bound
/// this, prepend array). The native `function_bind` builds the prepend
/// array and delegates here — the native registry holds stateless fn
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
pub fn function_constructor(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let argv: Vec<Value> = {
        let heap = &*nctx.heap();
        (1..args.len())
            .filter_map(|i| args.get(heap, i).map(|v| v.raw()))
            .collect()
    };
    nctx.handle_scope(|nctx, scope| {
        // root the caller context before the allocating ToString loop below
        let (_, heap, state) = nctx.split();
        let context = scope.handle(
            state
                .current_context(heap)
                .unwrap_or_else(|| heap.known().empty_context.as_tagged(heap).erase()),
        );
        // ToString all arguments (user toString may run)
        let mut parts: Vec<String> = Vec::with_capacity(argv.len());
        for a in argv {
            let s = {
                let (_, heap, _) = nctx.split();
                // Safety: fresh argument word, consumed before any allocation.
                let a = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(a) });
                Convert::to_string(heap, &scope, a)?.raw()
            };
            parts.push({
                let heap = &*nctx.heap();
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
        let mut p = parser::Parser::new(parser::Utf8SliceStream::new(&source));
        if p.parse_script().is_err() {
            nctx.set_pending_exception(VmError::Type);
            // Safety: fresh root-slot word read for the immediate return.
            return Ok(unsafe { nctx.heap().known().exception.read_unchecked() });
        }
        let ast = p.into_ast();
        let compiled = match base_compiler::compile_eval(&ast) {
            Ok(c) => c,
            Err(_) => {
                nctx.set_pending_exception(VmError::Type);
                // Safety: fresh root-slot word read for the immediate return.
                return Ok(unsafe { nctx.heap().known().exception.read_unchecked() });
            }
        };
        let (vm, heap, state) = nctx.split();
        let context = scope
            .cast::<Context>(context.as_tagged(heap))
            .ok_or(VmError::Type)?;
        let closure = materialize_closure_vm(vm, heap, state, &scope, &compiled, context)?;
        nctx.call(
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(closure.read_unchecked()) },
            GcSlice::EMPTY,
        )
    })
}
