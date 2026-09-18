//! ES 19: function properties of the global object (eval, isNaN).

use crate::Object;
use crate::RuntimeContext;
use crate::materialize::materialize_closure_vm;
use crate::{Context, Convert, DenseString, Errors, HandleSlice, Tagged, Value, VmError};
use base_compiler::compile_eval;

pub fn eval_runtime<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // root the caller context before the allocating ToString below
        let context = scope.handle(state.current_context(heap).ok_or(VmError::Type)?);
        let src = Ok(args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw())?;
        // Safety: fresh argument word, consumed before any allocation.
        let src = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(src) });
        let s = Convert::to_string(heap, &scope, src)?;
        let s = s.raw();
        let text = // Safety: fresh word, no allocation since the read.
            unsafe { s.assume_valid(heap) }
                .get_as::<DenseString>()
                .map(|s| s.to_rust_string(heap))
                .ok_or(VmError::Type)?;

        let mut p = parser::Parser::new(parser::Utf8SliceStream::new(&text));
        if let Err(e) = p.parse_script() {
            // TODO: a SyntaxError class; approximate with TypeError for now
            let _ = e;
            let ex = Errors::from_vm_error(vm, heap, state, VmError::Type)?;
            state.set_pending_exception(ex);
            return Ok(heap.known().exception.as_tagged(heap).erase());
        }
        let ast = p.into_ast();
        let compiled = match compile_eval(&ast) {
            Ok(c) => c,
            Err(_) => {
                let ex = Errors::from_vm_error(vm, heap, state, VmError::Type)?;
                state.set_pending_exception(ex);
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
        };

        let context = scope
            .cast::<Context>(context.as_tagged(heap))
            .ok_or(VmError::Type)?;
        let closure = materialize_closure_vm(vm, heap, state, &scope, &compiled, context)?;
        RuntimeContext::call(vm, heap, state, closure.erase(), HandleSlice::EMPTY, None)
    })
}

/// `isNaN(x)`: ToNumber(x) is NaN.
pub fn is_nan<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = state.handle_scope(|scope| {
        let arg = {
            let v = args
                .get(1)
                .map(|h| h.as_tagged(heap))
                .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
            scope.handle(v)
        };
        Object::to_numeric(vm, heap, state, arg)
    })?;
    let Some(n) = n else {
        return Ok(heap.known().exception.as_tagged(heap).erase());
    };

    Ok(Convert::boolean(heap, n.is_nan()))
}
