//! ES 20.2: the Function constructor, Function.prototype
//! toString/call/apply/bind, and the bind-closure prelude.

use crate::materialize::materialize_closure_vm;
use crate::{Context, Convert, DenseString, GcSlice, Smi, Tagged, Value, VmError};

/// Stub: `Function.prototype.toString` returns a stable marker string
/// (test262 A2.2 compares it against itself, not against real source).
pub(crate) fn function_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
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
pub(crate) fn function_call(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = nctx
        .heap()
        .no_gc(|heap| Ok(args.get(heap, 0).ok_or(VmError::Arity)?.erase()))?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let call_args: Vec<Value> = nctx
        .heap()
        .no_gc(|heap| args.iter(heap).skip(1).map(|v| v.erase()).collect());
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word, still fresh (no allocation since).
        let f = unsafe { Tagged::<Value>::from_value_unchecked(f) };
        nctx.call(f, scope.stage_words(&call_args))
    })
}

/// `Function.prototype.bind(thisArg, ...prepend)` (ES 20.2.3.5): the
/// bound function is the JS closure template installed by BIND_PRELUDE,
/// called with (target, thisArg, prepend-array).
pub(crate) fn function_bind(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let raw_f = nctx
        .heap()
        .no_gc(|heap| Ok(args.get(heap, 0).ok_or(VmError::Arity)?.erase()))?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), raw_f) {
        return Err(VmError::Type);
    }
    // Safety: fresh root-slot word read for the immediate use.
    let undefined = nctx
        .heap()
        .no_gc(|heap| heap.known().undefined.as_tagged(heap).erase());
    let raw_this_arg = nctx
        .heap()
        .no_gc(|heap| args.get(heap, 1).map(|v| v.erase()))
        .unwrap_or(undefined);
    let prepend: Vec<Value> = nctx
        .heap()
        .no_gc(|heap| args.iter(heap).skip(2).map(|v| v.erase()).collect());
    nctx.handle_scope(|nctx, scope| {
        // everything below allocates (interning, the [[Get]] for
        // __makeBound, the prepend array, the call): keep the raw inputs
        // rooted and re-read at the point of use
        // Safety: fresh argument words, rooted below before any allocation.
        let f = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_f) });
        let this_arg = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_this_arg) });
        // Function.prototype.__makeBound (installed by BIND_PRELUDE)
        let make_bound = {
            let name = nctx.intern(&scope, "__makeBound");
            let (vm, heap, state) = nctx.split();
            // Safety: fresh rooted-slot words, consumed by the lookup.
            let name = unsafe { name.read_unchecked() };
            let proto = unsafe { heap.known().function_prototype.read_unchecked() };
            match crate::runtime::Runtime::get_property(vm, heap, state, proto, name)? {
                crate::runtime::Coercion::Threw => {
                    // Safety: fresh root-slot word read for the return.
                    return Ok(unsafe { heap.known().exception.read_unchecked() });
                }
                // Safety: fresh word from the lookup, rooted immediately.
                crate::runtime::Coercion::Value(v) => {
                    scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(v) })
                }
            }
        };
        let staged = scope.stage_words(&prepend);
        let (_, heap, _) = nctx.split();
        let array = scope.handle(heap.new_array(&scope, staged).erase_type());
        // args[0] is the receiver (undefined for the plain call)
        // Safety: fresh rooted-slot words staged for the call.
        let recv = unsafe { nctx.heap().known().undefined.read_unchecked() };
        let f_word = unsafe { f.read_unchecked() };
        let this_word = unsafe { this_arg.read_unchecked() };
        let array_word = unsafe { array.read_unchecked() };
        nctx.call(
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(make_bound.read_unchecked()) },
            scope.stage_words(&[recv, f_word, this_word, array_word]),
        )
    })
}

/// `Function.prototype.apply(thisArg, argsArray)` (ES 20.2.3.3).
pub(crate) fn function_apply(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = nctx
        .heap()
        .no_gc(|heap| Ok(args.get(heap, 0).ok_or(VmError::Arity)?.erase()))?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    // Safety: fresh root-slot word read for the immediate use.
    let undefined = nctx
        .heap()
        .no_gc(|heap| heap.known().undefined.as_tagged(heap).erase());
    let this_arg = nctx
        .heap()
        .no_gc(|heap| args.get(heap, 1).map(|v| v.erase()))
        .unwrap_or(undefined);
    let array = nctx
        .heap()
        .no_gc(|heap| args.get(heap, 2).map(|v| v.erase()))
        .unwrap_or(undefined);
    let call_args: Vec<Value> = {
        let nullish = nctx.heap().no_gc(|heap| {
            array == heap.known().undefined.as_tagged(heap).erase()
                || array == heap.known().null.as_tagged(heap).erase()
        });
        if nullish {
            vec![this_arg]
        } else {
            // array-like: read elements 0..length (holes read as undefined)
            let len = nctx.heap().no_gc(|heap| {
                unsafe { array.assume_valid(heap) }
                    .as_heap_object()
                    .map(|o| {
                        o.as_ref()
                            .array_length(
                                heap,
                                // Safety: fresh root-slot word for a name read.
                                heap.known().strings.length.as_tagged(heap),
                            )
                            .and_then(|v| Smi::decode(v.erase()).map(|s| s.value() as usize))
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            });
            let mut out = Vec::with_capacity(len + 1);
            out.push(this_arg);
            for i in 0..len {
                out.push(nctx.heap().no_gc(|heap| {
                    unsafe { array.assume_valid(heap) }
                        .as_heap_object()
                        .and_then(|o| o.as_ref().element_value(heap, i))
                        .map(|v| v.erase())
                        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase())
                }));
            }
            out
        }
    };
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word (no allocation since the reads).
        let f = unsafe { Tagged::<Value>::from_value_unchecked(f) };
        nctx.call(f, scope.stage_words(&call_args))
    })
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
    let argv: Vec<Value> = nctx.heap().no_gc(|heap| {
        (1..args.len())
            .filter_map(|i| args.get(heap, i).map(|v| v.erase()))
            .collect()
    });
    nctx.handle_scope(|nctx, scope| {
        // root the caller context before the allocating ToString loop below
        // Safety: register/root-slot words, fresh at entry, rooted below.
        let context = scope.handle(unsafe {
            Tagged::<Value>::from_value_unchecked(nctx.current_context().unwrap_or_else(|| {
                // Safety (outer block): fresh root-slot word read for
                // the rooting.
                nctx.heap().known().empty_context.read_unchecked()
            }))
        });
        // ToString all arguments (user toString may run)
        let mut parts: Vec<String> = Vec::with_capacity(argv.len());
        for a in argv {
            let s = {
                let (_, heap, _) = nctx.split();
                // Safety: fresh argument word, consumed before any allocation.
                let a = unsafe { Tagged::<Value>::from_value_unchecked(a) };
                Convert::to_string(heap, &scope, a)?.erase()
            };
            parts.push(nctx.heap().no_gc(|heap| {
                // Safety: fresh word, no allocation since the read.
                unsafe { s.assume_valid(heap) }
                    .get_as::<DenseString>()
                    .map(|x| x.to_rust_string(heap))
                    .unwrap_or_default()
            }));
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
            .cast::<Context>(context.as_tagged(&*heap))
            .ok_or(VmError::Type)?;
        let closure = materialize_closure_vm(vm, heap, state, &scope, &compiled, context)?;
        nctx.call(
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(closure.read_unchecked()) },
            GcSlice::EMPTY,
        )
    })
}
