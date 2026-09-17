//! ES 20.4: the Symbol constructor (minimal surface).

use crate::{DenseString, GcSlice, Symbol, Value, VmError};

/// `Symbol(desc)`: a fresh Symbol primitive (ES 20.4.1.1). This minimal
/// surface exists so user code can author iterables
/// (`obj[Symbol.iterator] = ...`); `Symbol.iterator` is the well-known one.
pub fn symbol_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let desc_text = nctx.heap().no_gc(|heap| {
        args.get(heap, 1)
            .and_then(|d| d.get_as::<DenseString>())
            .map(|s| s.to_rust_string(heap))
    });
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
