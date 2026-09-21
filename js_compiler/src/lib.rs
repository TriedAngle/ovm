//! JavaScript frontend: oxc parse + semantic analysis → shared IR.
//!
//! [`compile_js`] is the whole pipeline. oxc provides parsing, name
//! resolution, the scope tree, var hoisting, strict-mode propagation and
//! direct-eval detection; the [`analysis`] module derives the VM's layout
//! facts oxc does not compute (IR function ids incl. synthesized
//! constructors/field initializers, the captured-symbol set, `super` /
//! `this` / `new.target` capture info, hoisting lists); [`codegen`] walks
//! the oxc AST and emits the Ignition-style IR, assigning each
//! function's registers and context slots at its prologue.

pub mod analysis;
pub mod codegen;
mod label;

pub use codegen::CompileError;

use oxc_allocator::Allocator;
use oxc_span::SourceType;
use oxc_parser::{ParseOptions, Parser};
use oxc_semantic::SemanticBuilder;

pub use ir::{FrontendError, Program, SourceMode};

/// Parse + analyze + lower `source` in the given mode.
pub fn compile_js(source: &str, mode: SourceMode) -> Result<Program, FrontendError> {
    let allocator = Allocator::default();
    let parser = Parser::new(&allocator, source, SourceType::script()).with_options(
        ParseOptions {
            // the old parser dropped parens; also keeps NamedEvaluation
            // semantics uniform
            preserve_parens: false,
            // scripts and REPL entries may `return` at the top level
            allow_return_outside_function: true,
            ..ParseOptions::default()
        },
    );
    let ret = parser.parse();
    if let Some(err) = ret.diagnostics.first() {
        return Err(FrontendError::syntax(err.to_string()));
    }
    let semantic = SemanticBuilder::new()
        .with_build_nodes(true)
        .with_check_syntax_error(true)
        .build(&ret.program);
    if let Some(err) = semantic.diagnostics.first() {
        return Err(FrontendError::syntax(err.to_string()));
    }
    let facts = analysis::analyze(
        &ret.program,
        &semantic.semantic,
        match mode {
            SourceMode::Script => analysis::Mode::Script,
            SourceMode::Eval => analysis::Mode::Eval,
            SourceMode::Repl => analysis::Mode::Repl,
        },
    );
    codegen::generate(semantic.semantic.scoping(), &facts)
        .map_err(|e| FrontendError::compile(e.to_string()))
}
