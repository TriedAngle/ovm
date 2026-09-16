//! ES 19: function properties of the global object (eval, isNaN).

use crate::{Context, Convert, DenseString, GcSlice, Value, VmError, runtime::Runtime};
use base_compiler::compile_eval;
use crate::materialize::materialize_closure_vm;

pub(crate) fn eval_native(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let src = args.get(1).ok_or(VmError::Arity)?;
    let context = nctx.current_context().ok_or(VmError::Type)?;

    let text = nctx.handle_scope(|nctx, scope| {
        let (_vm, heap, _) = nctx.split();
        let s = Convert::to_string(heap, &scope, src)?;
        heap.no_gc(|nogc| {
            s.get_as::<DenseString>(nogc)
                .map(|s| s.to_rust_string(nogc))
                .ok_or(VmError::Type)
        })
    })?;

    let mut p = parser::Parser::new(parser::Utf8SliceStream::new(&text));
    if let Err(e) = p.parse_script() {
        // TODO: a SyntaxError class; approximate with TypeError for now
        let _ = e;
        nctx.set_pending_exception(VmError::Type);
        return Ok(nctx.heap().known().exception.value());
    }
    let ast = p.into_ast();
    let compiled = match compile_eval(&ast) {
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

/// `isNaN(x)`: ToNumber(x) is NaN.
pub(crate) fn is_nan(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    let (vm, heap, state) = nctx.split();
    let n = match Runtime::to_numeric(vm, heap, state, arg)? {
        Some(n) => n,
        None => return Ok(heap.known().exception.value()),
    };
    Ok(Convert::boolean(heap, n.is_nan()))
}
