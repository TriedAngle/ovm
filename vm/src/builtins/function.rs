//! ES 20.2: the Function constructor, Function.prototype
//! toString/call/apply/bind, and the bind-closure prelude.

use crate::{Context, Convert, DenseString, GcSlice, Smi, Value, VmError};
use crate::materialize::materialize_closure_vm;

/// Stub: `Function.prototype.toString` returns a stable marker string
/// (test262 A2.2 compares it against itself, not against real source).
pub(crate) fn function_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        Ok(nctx
            .intern(&scope, "function () { [native code] }")
            .as_tagged()
            .erase())
    })
}

/// `Function.prototype.call(thisArg, ...args)` (ES 20.2.3.4).
pub(crate) fn function_call(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = args.get(0).ok_or(VmError::Arity)?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let call_args: Vec<Value> = args.as_slice().iter().skip(1).copied().collect();
    nctx.handle_scope(|nctx, scope| nctx.call(f, scope.stage(&call_args)))
}

/// `Function.prototype.bind(thisArg, ...prepend)` (ES 20.2.3.5): the
/// bound function is the JS closure template installed by BIND_PRELUDE,
/// called with (target, thisArg, prepend-array).
pub(crate) fn function_bind(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = args.get(0).ok_or(VmError::Arity)?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let this_arg = args
        .get(1)
        .unwrap_or_else(|| nctx.heap().known().undefined.value());
    let prepend: Vec<Value> = args.as_slice().iter().skip(2).copied().collect();
    // Function.prototype.__makeBound (installed by BIND_PRELUDE)
    let make_bound = {
        let name = nctx.handle_scope(|nctx, scope| nctx.intern(&scope, "__makeBound").value());
        let (vm, heap, state) = nctx.split();
        let proto = heap.known().function_prototype.value();
        match crate::runtime::Runtime::get_property(vm, heap, state, proto, name)? {
            crate::runtime::Coercion::Threw => return Ok(heap.known().exception.value()),
            crate::runtime::Coercion::Value(v) => v,
        }
    };
    let array = nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        heap.new_array(&scope, &prepend).into_tagged().erase()
    });
    nctx.handle_scope(|nctx, scope| {
        // args[0] is the receiver (undefined for the plain call)
        let recv = nctx.heap().known().undefined.value();
        nctx.call(make_bound, scope.stage(&[recv, f, this_arg, array]))
    })
}

/// `Function.prototype.apply(thisArg, argsArray)` (ES 20.2.3.3).
pub(crate) fn function_apply(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = args.get(0).ok_or(VmError::Arity)?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let this_arg = args
        .get(1)
        .unwrap_or_else(|| nctx.heap().known().undefined.value());
    let array = args
        .get(2)
        .unwrap_or_else(|| nctx.heap().known().undefined.value());
    let call_args: Vec<Value> = if array == nctx.heap().known().undefined.value()
        || array == nctx.heap().known().null.value()
    {
        vec![this_arg]
    } else {
        // array-like: read elements 0..length (holes read as undefined)
        let len = nctx.heap().no_gc(|nogc| {
            array
                .as_heap_object(nogc)
                .map(|o| {
                    o.as_ref()
                        .array_length(
                            nogc,
                            crate::SlotName::from_value(nogc.known().strings.length.value()),
                        )
                        .and_then(|v| Smi::decode(v).map(|s| s.value() as usize))
                        .unwrap_or(0)
                })
                .unwrap_or(0)
        });
        let mut out = Vec::with_capacity(len + 1);
        out.push(this_arg);
        for i in 0..len {
            out.push(nctx.heap().no_gc(|nogc| {
                array
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, i))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            }));
        }
        out
    };
    nctx.handle_scope(|nctx, scope| nctx.call(f, scope.stage(&call_args)))
}

/// The bound-function template: a plain JS closure over (target, bound
/// this, prepend array). The native `function_bind` builds the prepend
/// array and delegates here — the native registry holds stateless fn
/// pointers, so the closure state must live in a JS closure.
pub(crate) const BIND_PRELUDE: &str = r#"
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
pub(crate) fn function_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let argv: Vec<Value> = (1..args.len()).filter_map(|i| args.get(i)).collect();
    // ToString all arguments (user toString may run)
    let mut parts: Vec<String> = Vec::with_capacity(argv.len());
    for a in argv {
        let s = nctx.handle_scope(|nctx, scope| {
            let (_, heap, _) = nctx.split();
            let _ = heap;
            Convert::to_string(nctx.heap(), &scope, a)
        })?;
        parts.push(nctx.heap().no_gc(|nogc| {
            s.get_as::<DenseString>(nogc)
                .map(|x| x.to_rust_string(nogc))
                .unwrap_or_default()
        }));
    }
    let (params, body) = match parts.split_last() {
        Some((body, params)) => (params.join(", "), body.clone()),
        None => (String::new(), String::new()),
    };
    let source = format!("(function ({params}) {{\n{body}\n}})");
    let context = nctx
        .current_context()
        .unwrap_or_else(|| nctx.heap().known().empty_context.value());
    let mut p = parser::Parser::new(parser::Utf8SliceStream::new(&source));
    if p.parse_script().is_err() {
        nctx.set_pending_exception(VmError::Type);
        return Ok(nctx.heap().known().exception.value());
    }
    let ast = p.into_ast();
    let compiled = match base_compiler::compile_eval(&ast) {
        Ok(c) => c,
        Err(_) => {
            nctx.set_pending_exception(VmError::Type);
            return Ok(nctx.heap().known().exception.value());
        }
    };
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, state) = nctx.split();
        let context = scope.cast::<Context>(context).ok_or(VmError::Type)?;
        let closure = materialize_closure_vm(vm, heap, state, &scope, &compiled, context)?;
        nctx.call(closure.value(), GcSlice::EMPTY)
    })
}
