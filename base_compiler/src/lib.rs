//! AST → bytecode materializer.
//!
//! A single naive pass over a resolved parser AST, in the Ignition style:
//! one walk, an implicit register file, temporaries stacked above the
//! resolver's per-function layout. Output is a heap-free
//! [`CompiledScript`]; the `vm` crate owns converting that to VM objects.

pub mod codegen;
mod label;

use parser::{Ast, FunctionId};

/// Value table entries. Materialization (in `vm`) converts these into
/// heap objects: interned strings, `Float`s, oddball singletons, and
/// shared `CallableInfoObject` templates for closures.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// Interned string (property names, string literals)
    String(Vec<u8>),
    /// Heap float (non-Smi number literals)
    Float(f64),
    /// Smi-range integer literals too big for the (at most 2-byte signed)
    /// `LoadSmi` operand: they ride the constant pool instead
    Smi(i64),
    /// The true/false singletons
    Boolean(bool),
    /// `undefined` / `null` singletons
    Undefined,
    Null,
    /// `CreateClosure` template: the shared callable info of a nested function
    Callable(FunctionId),
    /// The function context's slot names (parallel to its slots, for
    /// dynamic name resolution); materialized into a shared `ScopeInfo`
    /// referenced by `CreateFunctionContext`
    ContextNames(Vec<Vec<u8>>),
    /// %Object.prototype% (base-class prototype parent)
    ObjectPrototype,
    /// %Function.prototype% (base-class constructor parent)
    FunctionPrototype,
}

/// `layout [try_start, try_end, handler_pc]`: a half-open bytecode region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandlerEntry {
    pub try_start: usize,
    pub try_end: usize,
    pub handler_pc: usize,
}

#[derive(Debug, Default)]
pub struct CompiledFunction {
    pub bytecode: Vec<u8>,
    pub constants: Vec<Constant>,
    pub name: Option<Vec<u8>>,
    /// frame layout: one register per formal parameter (patterns count one)
    pub formal_parameter_count: u32,
    /// JS-visible `length`: parameters before the first default/rest/pattern
    pub formal_length: u32,
    pub kind: parser::FunctionKind,
    /// Preserved for strict-sensitive runtime operations. Enforcement is
    /// intentionally deferred until the VM has language-mode-aware stores.
    pub strict: bool,
    /// Frame size in stack slots: resolver locals + context-save slot +
    /// temporaries.
    pub register_count: u32,
    pub handlers: Vec<HandlerEntry>,
}

/// Parallel to `Ast`'s function table: `functions[FunctionId]`.
#[derive(Debug, Default)]
pub struct CompiledScript {
    pub functions: Vec<CompiledFunction>,
}

/// A construct the materializer does not support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub span: parser::Span,
    pub feature: &'static str,
}

impl CompileError {
    fn new(span: parser::Span, feature: &'static str) -> Self {
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
pub fn compile_script(ast: &Ast) -> Result<CompiledScript, CompileError> {
    let resolved = parser::resolver::resolve(ast);
    codegen::generate(ast, &resolved)
}

/// Resolve + compile direct eval source: unresolved names are compiled as
/// runtime lookups through the caller's context chain (`LoadDynamicName`).
pub fn compile_eval(ast: &Ast) -> Result<CompiledScript, CompileError> {
    let resolved = parser::resolver::resolve_for_eval(ast);
    codegen::generate(ast, &resolved)
}

/// Resolve + compile a REPL entry: top-level declarations become global
/// object properties so they persist across entries
pub fn compile_repl(ast: &Ast) -> Result<CompiledScript, CompileError> {
    let resolved = parser::resolver::resolve_repl(ast);
    codegen::generate(ast, &resolved)
}
