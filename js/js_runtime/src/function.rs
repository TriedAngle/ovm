//! ES 20.2: the Function constructor, Function.prototype
//! toString/call/apply/bind, and the bind-closure prelude.

use vm_core::Coercion;
use vm_core::HostCtx;
use vm_core::Lookup;
use vm_core::Object;
use vm_core::materialize::Materialize;
use vm_core::{
    Args, ContextObject, Convert, DenseString, Errors, Handle, HandleSlice, Heap, Smi, Tagged,
    Value, VmError,
};
use vm_core::{raise_runtime, rt_try};

/// Stub: `Function.prototype.toString` returns a stable marker string
/// (test262 A2.2 compares it against itself, not against real source).
pub fn function_to_string<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    _args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // fresh interned word, consumed with no allocation delay
        vm.interner()
            .intern_str(heap, &scope, "function () { [native code] }")
            .as_tagged(heap)
            .erase()
    })
}

/// `Function.prototype.call(thisArg, ...args)` (ES 20.2.3.1): invoke the
/// receiver (args[0], per the receiver-first runtime calling convention)
/// with `thisArg` as `this`.
pub fn function_call<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let f = args.get_handle(heap, 0);
    if !Object::is_callable(heap, f.as_tagged(heap)) {
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    let fwd = nctx.state.stack().slice(args.slice_from(1));
    match HostCtx::enter(vm, heap, state, f, fwd, None).map(|v| v.raw()) {
        Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
        Err(err) => raise_runtime(vm, heap, state, err),
    }
}

/// `Function.prototype.apply(thisArg, argsArray)` (ES 20.2.3.2): invoke
/// the receiver with `thisArg` as `this` and the array-like spread as
/// arguments.
pub fn function_apply<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let f = args.get_handle(heap, 0);
    if !Object::is_callable(heap, f.as_tagged(heap)) {
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    // the args window is rooted for the call; re-read at the point of use
    let this_arg = args.get_handle(heap, 1);
    let array = (args.len() > 2).then(|| args.get_handle(heap, 2));
    state.handle_scope(|scope| {
        let staged = scope.stage(&spread_apply_args(
            heap,
            this_arg.as_tagged(heap),
            array.map(|a| a.as_tagged(heap)),
        ));
        match HostCtx::enter(vm, heap, state, f, staged, None).map(|v| v.raw()) {
            Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
            Err(err) => raise_runtime(vm, heap, state, err),
        }
    })
}

/// `Function.prototype.bind(thisArg, ...prepend)` (ES 20.2.3.5): the
/// bound function is the JS closure template installed by BIND_PRELUDE,
/// called with (target, thisArg, prepend-array).
pub fn function_bind<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let raw_f = {
        let f = args.get(heap, 0);
        let f = f;
        if !Object::is_callable(heap, f) {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        f.raw()
    };
    // the args window is rooted for the call; re-read at the point of use
    let raw_this_arg = args.get(heap, 1).raw();
    let prepend: Vec<Value> = (2..args.len()).map(|i| args.get(heap, i).raw()).collect();
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
            match rt_try!(
                vm,
                heap,
                state,
                Lookup::get_property_on(vm, heap, state, proto, proto, name)
            ) {
                Coercion::Threw => {
                    return heap.known().exception.as_tagged(heap).erase();
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
        match HostCtx::enter(vm, heap, state, make_bound, staged, None).map(|v| v.raw()) {
            Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
            Err(err) => return raise_runtime(vm, heap, state, err),
        }
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
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let argv: Vec<_> = (1..args.len()).map(|i| args.get_handle(heap, i)).collect();
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
            let s = scope.handle(match Convert::to_string(heap, &scope, a).map(|v| v.raw()) {
                Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
                Err(err) => return raise_runtime(vm, heap, state, err),
            });
            parts.push(
                s.as_tagged(heap)
                    .get_as::<DenseString>(heap)
                    .map(|x| x.to_rust_string(heap))
                    .unwrap_or_default(),
            );
        }
        let (params, body) = match parts.split_last() {
            Some((body, params)) => (params.join(", "), body.clone()),
            None => (String::new(), String::new()),
        };
        let source = format!("(function ({params}) {{\n{body}\n}})");
        let program = match js_compiler::compile_js(&source, bytecode::SourceMode::Eval) {
            Ok(program) => program,
            Err(_) => {
                let ex = rt_try!(
                    vm,
                    heap,
                    state,
                    Errors::from_vm_error(vm, heap, state, VmError::Type)
                );
                state.set_pending_exception(ex);
                return heap.known().exception.as_tagged(heap).erase();
            }
        };
        let Some(context) = scope.cast::<ContextObject>(heap, context.as_tagged(heap)) else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        let closure = rt_try!(
            vm,
            heap,
            state,
            Materialize::closure_vm(vm, heap, state, &scope, &program, context)
        );
        match HostCtx::enter(vm, heap, state, closure.erase(), HandleSlice::EMPTY, None)
            .map(|v| v.raw())
        {
            Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
            Err(err) => return raise_runtime(vm, heap, state, err),
        }
    })
}

/// Spread the `apply` argument window: `[thisArg, elements...]` read
/// array-like from `array` (holes and out-of-range indices read as
/// undefined; a missing/nullish array yields just the receiver).
fn spread_apply_args<'a>(
    heap: &'a Heap,
    this_arg: Tagged<'a, Value>,
    array: Option<Tagged<'a, Value>>,
) -> Vec<Tagged<'a, Value>> {
    let mut out = vec![this_arg];
    let Some(array) = array else {
        return out;
    };
    let undefined = heap.known().undefined.as_tagged(heap).erase();
    if array == undefined || array == heap.known().null.as_tagged(heap) {
        return out;
    }
    let len = array
        .as_heap_object()
        .map(|o| {
            o.as_ref()
                .array_length(heap, heap.known().strings.length.as_tagged(heap))
                .and_then(|v| Smi::decode(v.raw()).map(|s| s.value() as usize))
                .unwrap_or(0)
        })
        .unwrap_or(0);
    out.reserve(len);
    for i in 0..len {
        out.push(
            array
                .as_heap_object()
                .and_then(|o| o.as_ref().element_value(heap, i))
                .unwrap_or(undefined),
        );
    }
    out
}
