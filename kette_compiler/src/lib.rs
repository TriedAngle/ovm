//! Kette AST → bytecode lowering.
//!
//! Pipeline: parse → [`desugar`] (control flow / operators become sends,
//! in place) → resolve → [`codegen`]. The output is a heap-free
//! [`bytecode::Program`]; the `vm` crate owns converting it to VM objects.

pub mod codegen;
pub mod desugar;

use bytecode::{FrontendError, Program, SourceMode};
use kette_parser::{Ast, ByteSpan, Parser, Utf8SliceStream, resolve};
/// A construct the lowering does not support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub span: ByteSpan,
    pub feature: &'static str,
}

impl CompileError {
    pub fn new(span: ByteSpan, feature: &'static str) -> Self {
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

/// Desugar + resolve + lower a parsed unit. `functions[0]` is the script.
pub fn compile_ast(ast: &mut Ast) -> Result<Program, CompileError> {
    let _pipeline = trace::info_span!("kette::compile");
    {
        let _span = trace::info_span!("kette::desugar").entered();
        desugar::desugar(ast);
    }
    let resolved = {
        let _span = trace::info_span!("kette::resolve").entered();
        resolve(ast)
    };
    {
        let _span = trace::info_span!("kette::codegen").entered();
        codegen::generate(ast, &resolved)
    }
}

/// Kette frontend entry point: parse `source` and lower it to bytecode,
/// mapping failures into the language-neutral [`FrontendError`].
pub fn compile_kette(source: &str, mode: SourceMode) -> Result<Program, FrontendError> {
    if mode != SourceMode::Script {
        return Err(FrontendError::compile(
            "kette: only script mode is implemented",
        ));
    }
    let mut parser = Parser::new(Utf8SliceStream::new(source));
    parser
        .parse_script()
        .map_err(|e| FrontendError::syntax(e.to_string()))?;
    let mut ast = parser.into_ast();
    compile_ast(&mut ast).map_err(|e| FrontendError::compile(e.to_string()))
}
