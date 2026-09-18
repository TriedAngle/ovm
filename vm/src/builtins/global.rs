//! ES 19: function properties of the global object (eval, isNaN).

use crate::materialize::materialize_closure_vm;
use crate::natives::NativeContext;
use crate::{Context, Convert, DenseString, HandleSlice, Tagged, Value, VmError, runtime::Runtime};
use base_compiler::compile_eval;

pub fn eval_native(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        // root the caller context before the allocating ToString below
        let (_vm, heap, state) = nctx.split();
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
            nctx.set_pending_exception(VmError::Type);
            // Safety: fresh root-slot word read for the immediate return.
            return Ok(unsafe { nctx.heap().known().exception.read_unchecked() });
        }
        let ast = p.into_ast();
        let compiled = match compile_eval(&ast) {
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
            HandleSlice::EMPTY,
        )
    })
}

/// `isNaN(x)`: ToNumber(x) is NaN.
pub fn is_nan(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
    let n = nctx.handle_scope(|nctx, scope| {
        let (vm, heap, state) = nctx.split();
        let arg = {
            let v = args
                .get(1)
                .map(|h| h.as_tagged(heap))
                .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
            scope.handle(v)
        };
        Runtime::to_numeric(vm, heap, state, arg)
    })?;
    let Some(n) = n else {
        // Safety: fresh root-slot word read for the immediate return.
        return Ok(unsafe { nctx.heap().known().exception.read_unchecked() });
    };

    Ok({
        let heap = &*nctx.heap();
        Convert::boolean(heap, n.is_nan()).raw()
    })
}
