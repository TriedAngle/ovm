//! ES 20.4: the Symbol constructor (minimal surface).

use vm_core::RuntimeContext;
use vm_core::{DenseString, HandleSlice, Symbol, Tagged, Value, VmError};

/// `Symbol(desc)`: a fresh Symbol primitive (ES 20.4.1.1). This minimal
/// surface exists so user code can author iterables
/// (`obj[Symbol.iterator] = ...`); `Symbol.iterator` is the well-known one.
pub fn symbol_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let desc_text = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .and_then(|d| d.get_as::<DenseString>())
            .map(|s| s.to_rust_string(heap));
        let mut text = String::from("Symbol(");
        if let Some(d) = &desc_text {
            text.push_str(d);
        }
        text.push(')');
        let sym = Symbol::new(heap, &scope, text.as_bytes());
        Ok(sym.as_tagged(heap).erase())
    })
}
