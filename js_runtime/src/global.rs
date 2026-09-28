//! ES 19: function properties of the global object (eval, isNaN).

use vm_core::Object;
use vm_core::RuntimeContext;
use vm_core::materialize::Materialize;
use vm_core::{Context, Convert, DenseString, Errors, HandleSlice, Tagged, Value, VmError};
use vm_core::{raise_runtime, rt_try};

pub fn eval_runtime<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // root the caller context before the allocating ToString below
        let Some(context) = state.current_context(heap) else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        let context = scope.handle(context);
        let src = rt_try!(vm, heap, state, args.get(1).ok_or(VmError::Arity));
        let s = scope.handle(match Convert::to_string(heap, &scope, src).map(|v| v.raw()) {
        Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
        Err(err) => return raise_runtime(vm, heap, state, err),
    });
        let Some(text) = s
            .as_tagged(heap)
            .get_as::<DenseString>()
            .map(|s| s.to_rust_string(heap))
        else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };

        let program = match js_compiler::compile_js(&text, bytecode::SourceMode::Eval) {
            Ok(program) => program,
            // TODO: a SyntaxError class; approximate with TypeError for now
            Err(_) => {
                let ex = rt_try!(vm, heap, state, Errors::from_vm_error(vm, heap, state, VmError::Type));
                state.set_pending_exception(ex);
                return heap.known().exception.as_tagged(heap).erase();
            }
        };

        let Some(context) = scope.cast::<Context>(context.as_tagged(heap)) else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        let closure = rt_try!(vm, heap, state, Materialize::closure_vm(vm, heap, state, &scope, &program, context));
        match RuntimeContext::call(vm, heap, state, closure.erase(), HandleSlice::EMPTY, None).map(|v| v.raw()) {
            Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
            Err(err) => return raise_runtime(vm, heap, state, err),
        }
    })
}

/// `isNaN(x)`: ToNumber(x) is NaN.
pub fn is_nan<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = rt_try!(vm, heap, state, state.handle_scope(|scope| {
        let arg = {
            let v = args
                .get(1)
                .map(|h| h.as_tagged(heap))
                .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
            scope.handle(v)
        };
        Object::to_numeric(vm, heap, state, arg)
    }));
    let Some(n) = n else {
        return heap.known().exception.as_tagged(heap).erase();
    };

    Convert::boolean(heap, n.is_nan())
}

/// `print(x)`: ToString(x) to stdout followed by a newline (a shell
/// convenience, not an ES builtin; the Octane runner reports through it).
pub fn print<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        if let Some(arg) = args.get(1) {
            let s = match Convert::to_string(heap, &scope, arg).map(|v| v.raw()) {
        Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
        Err(err) => return raise_runtime(vm, heap, state, err),
    };
            let s = s.raw();
            // Safety: fresh string word, no allocation since the read.
            let text = unsafe { s.assume_valid(heap) }
                .get_as::<DenseString>()
                .map(|s| s.to_rust_string(heap))
                .unwrap_or_default();
            println!("{text}");
        } else {
            println!();
        }
        heap.known().undefined.as_tagged(heap).erase()
    })
}

/// `performance.now()`: fractional milliseconds since the Unix epoch
/// (shell timing convenience mirroring the browser API).
pub fn performance_now<'a>(
    nctx: RuntimeContext<'a>,
    _args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    nctx.heap.new_number(ms)
}
