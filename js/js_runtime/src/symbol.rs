//! ES 20.4: the Symbol constructor (minimal surface).

use vm_core::RuntimeContext;
use vm_core::raise_runtime;
use vm_core::{DenseString, HandleSlice, Object, Symbol, Tagged, Value};

/// `Symbol(desc)`: a fresh Symbol primitive (ES 20.4.1.1). The description
/// is ToString'd and stored raw as `[[Description]]`; the `"Symbol(…)"`
/// wording is applied by `SymbolDescriptiveString` (see the `String`
/// constructor). This minimal surface exists so user code can author
/// iterables (`obj[Symbol.iterator] = ...`); `Symbol.iterator` is the
/// well-known one.
pub fn symbol_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // absent/undefined description is the empty string
        let desc = match args.get(1) {
            Some(h) if h.as_tagged(heap) != heap.known().undefined.as_tagged(heap) => {
                match Object::to_string(vm, heap, state, h) {
                    Ok(Some(s)) => {
                        let word = s.raw();
                        // Safety: fresh string word, no allocation since the read.
                        unsafe { word.assume_valid(heap) }
                            .get_as::<DenseString>()
                            .map(|d| d.to_rust_string(heap))
                            .unwrap_or_default()
                    }
                    Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
                    Err(err) => return raise_runtime(vm, heap, state, err),
                }
            }
            _ => String::new(),
        };
        let sym = Symbol::new(heap, &scope, desc.as_bytes());
        sym.as_tagged(heap).erase()
    })
}
