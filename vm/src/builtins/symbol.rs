//! ES 20.4: the Symbol constructor (minimal surface).

use crate::natives::NativeContext;
use crate::{DenseString, HandleSlice, Symbol, Value, VmError};

/// `Symbol(desc)`: a fresh Symbol primitive (ES 20.4.1.1). This minimal
/// surface exists so user code can author iterables
/// (`obj[Symbol.iterator] = ...`); `Symbol.iterator` is the well-known one.
pub fn symbol_constructor(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let desc_text = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .and_then(|d| d.get_as::<DenseString>())
            .map(|s| s.to_rust_string(heap))
    };
    nctx.handle_scope(|nctx, scope| {
        let mut text = String::from("Symbol(");
        if let Some(d) = &desc_text {
            text.push_str(d);
        }
        text.push(')');
        let sym = Symbol::new(nctx.heap(), &scope, text.as_bytes());
        // Safety: fresh rooted-slot word, returned without an
        // intervening allocation.
        Ok(unsafe { sym.read_unchecked() })
    })
}
