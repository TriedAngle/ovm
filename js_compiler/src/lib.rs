//! AST → shared IR lowering.
//!
//! A single naive pass over a resolved parser AST, in the Ignition style:
//! one walk, an implicit register file, temporaries stacked above the
//! resolver's per-function layout. Output is a heap-free [`ir::Program`];
//! the `vm` crate owns converting that to VM objects.

pub mod codegen;
mod label;

use js_parser::Ast;

pub use ir::{
    CallableKind, Constant, Function, FunctionBuilder, FunctionId, HandlerEntry, Program,
};
use ir::{FrontendError, SourceMode};

/// A construct the materializer does not support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub span: js_parser::Span,
    pub feature: &'static str,
}

impl CompileError {
    fn new(span: js_parser::Span, feature: &'static str) -> Self {
        Self { span, feature }
    }
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unsupported: {} at {}..{}",
            self.feature, self.span.start, self.span.end
        )
    }
}

/// Resolve + compile a parsed script. `functions[0]` is the script body.
pub fn compile_script(ast: &Ast) -> Result<Program, CompileError> {
    let resolved = js_parser::resolver::resolve(ast);
    codegen::generate(ast, &resolved)
}

/// Resolve + compile direct eval source: unresolved names are compiled as
/// runtime lookups through the caller's context chain (`LoadDynamicName`).
pub fn compile_eval(ast: &Ast) -> Result<Program, CompileError> {
    let resolved = js_parser::resolver::resolve_for_eval(ast);
    codegen::generate(ast, &resolved)
}

/// Resolve + compile a REPL entry: top-level declarations become global
/// object properties so they persist across entries
pub fn compile_repl(ast: &Ast) -> Result<Program, CompileError> {
    let resolved = js_parser::resolver::resolve_repl(ast);
    codegen::generate(ast, &resolved)
}

/// JavaScript frontend entry point: parse `source` and lower it to the
/// shared IR, mapping failures into the language-neutral [`FrontendError`].
pub fn compile_js(source: &str, mode: SourceMode) -> Result<Program, FrontendError> {
    let mut parser = js_parser::Parser::new(js_parser::Utf8SliceStream::new(source));
    parser
        .parse_script()
        .map_err(|e| FrontendError::syntax(e.to_string()))?;
    let ast = parser.into_ast();
    let result = match mode {
        SourceMode::Script => compile_script(&ast),
        SourceMode::Eval => compile_eval(&ast),
        SourceMode::Repl => compile_repl(&ast),
    };
    result.map_err(|e| FrontendError::compile(e.to_string()))
}
