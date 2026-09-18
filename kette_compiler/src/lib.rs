//! Kette AST → shared IR lowering.

use ir::{FrontendError, Program, SourceMode};

/// Kette frontend entry point. The parser/compiler do not exist yet, so
/// every source unit is rejected with a compile error.
pub fn compile_kette(source: &str, _mode: SourceMode) -> Result<Program, FrontendError> {
    let _ = source;
    Err(FrontendError::compile(
        "kette: compiler not implemented yet",
    ))
}
