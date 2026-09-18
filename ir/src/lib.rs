//! Shared compiler → VM program IR.
//!
//! Frontends (`js_compiler`, `kette_compiler`, ...) lower their ASTs into a
//! [`Program`]; the VM's materializer turns that into heap objects without
//! knowing which language produced it. The IR is deliberately language
//! neutral apart from the JS-centric [`CallableKind`] and well-known
//! [`Constant`] singletons, which other prototype-based languages reuse.
//!
//! Storage is arena/pool based: functions live in an [`Arena`], and their
//! bytecode, constants, handler table and name are runs inside shared
//! [`Pool`]s addressed by [`Span`]. The program is append-only.

mod arena;

pub use arena::{Arena, Pool, Span};

/// Typed index of a function in the program's function arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionId(pub u32);

impl FunctionId {
    /// The script body / entry function.
    pub const SCRIPT: FunctionId = FunctionId(0);

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What kind of source unit a frontend is asked to compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceMode {
    /// A top-level program: declarations resolve normally.
    Script,
    /// Direct eval: unresolved names become dynamic lookups through the
    /// caller's context chain.
    Eval,
    /// A REPL entry: top-level declarations become global object properties
    /// so they persist across entries.
    Repl,
}

/// Which stage rejected the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontendErrorKind {
    /// The source did not parse.
    Syntax,
    /// The source parsed but used an unsupported construct.
    Compile,
}

/// A frontend (parser/compiler) failure, in a form the VM can turn into a
/// thrown exception without knowing the language.
#[derive(Clone, Debug)]
pub struct FrontendError {
    pub kind: FrontendErrorKind,
    pub message: String,
}

impl FrontendError {
    pub fn syntax(message: impl Into<String>) -> Self {
        Self {
            kind: FrontendErrorKind::Syntax,
            message: message.into(),
        }
    }

    pub fn compile(message: impl Into<String>) -> Self {
        Self {
            kind: FrontendErrorKind::Compile,
            message: message.into(),
        }
    }
}

impl core::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FrontendError {}

/// The signature every frontend exposes for source → [`Program`]. Passed
/// around as a plain function pointer: frontends are selected explicitly at
/// the call site, so no trait objects or registries are needed yet.
pub type CompileFn = fn(&str, SourceMode) -> Result<Program, FrontendError>;

/// Callable execution metadata consumed by the interpreter.
///
/// The taxonomy is JS-centric; prototype-based frontends map their own
/// function/block forms onto the neutral subset (`Normal`, `Method`, ...).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CallableKind {
    #[default]
    Normal,
    Generator,
    Arrow,
    Method,
    Getter,
    Setter,
    BaseClassConstructor,
    DerivedClassConstructor,
    /// synthesized `constructor(...args) { super(...args) }`
    DefaultDerivedConstructor,
}

/// Value table entries. The VM materializes these into heap objects:
/// interned strings, `Float`s, and shared callable templates for closures.
/// The well-known singletons (undefined/null/true/false/0) load through
/// their dedicated `Load*` opcodes instead of the constant pool.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// Interned string (property names, string literals)
    String(Vec<u8>),
    /// Heap float (non-Smi number literals)
    Float(f64),
    /// Smi-range integer literals too big for the (at most 2-byte signed)
    /// `LoadSmi` operand: they ride the constant pool instead
    Smi(i64),
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

/// A stored function: metadata plus spans into the enclosing program's pools.
///
/// Pools are private: read the runs through the [`Program`] accessors.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    code: Span,
    constants: Span,
    handlers: Span,
    name: Option<Span>,
    /// Frame size in stack slots.
    pub register_count: u32,
    pub kind: CallableKind,
    /// Frame layout: one register per formal parameter (patterns count one).
    pub arity: u32,
    /// JS-visible `length`: parameters before the first default/rest/pattern.
    pub length: u32,
    /// Preserved for strict-sensitive runtime operations; enforcement is
    /// intentionally deferred until the VM has language-mode-aware stores.
    pub strict: bool,
}

/// Owned, not-yet-interned function produced by a frontend compiler.
///
/// The compiler fills this incrementally (code and constants are naturally
/// grown as `Vec`s); [`Program::add_function`] freezes it into the pools.
#[derive(Debug, Default)]
pub struct FunctionBuilder {
    pub bytecode: Vec<u8>,
    pub constants: Vec<Constant>,
    pub handlers: Vec<HandlerEntry>,
    pub name: Option<Vec<u8>>,
    pub register_count: u32,
    pub kind: CallableKind,
    pub arity: u32,
    pub length: u32,
    pub strict: bool,
}

impl FunctionBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

/// An arena-backed compiled program: `functions[0]` is the entry body.
#[derive(Debug, Default)]
pub struct Program {
    functions: Arena<Function>,
    code: Pool<u8>,
    constants: Pool<Constant>,
    handlers: Pool<HandlerEntry>,
    names: Pool<u8>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(functions: usize) -> Self {
        Self {
            functions: Arena::with_capacity(functions),
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.functions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Freeze `builder` into the shared pools and return its handle.
    pub fn add_function(&mut self, builder: FunctionBuilder) -> FunctionId {
        let code = self.code.alloc(&builder.bytecode);
        let constants = self.constants.alloc(&builder.constants);
        let handlers = self.handlers.alloc(&builder.handlers);
        let name = builder.name.map(|name| self.names.alloc(&name));
        let function = Function {
            code,
            constants,
            handlers,
            name,
            register_count: builder.register_count,
            kind: builder.kind,
            arity: builder.arity,
            length: builder.length,
            strict: builder.strict,
        };
        FunctionId(self.functions.insert(function))
    }

    pub fn function(&self, id: FunctionId) -> &Function {
        self.functions.get(id.0)
    }

    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.iter()
    }

    pub fn function_ids(&self) -> impl Iterator<Item = FunctionId> {
        (0..self.functions.len() as u32).map(FunctionId)
    }

    pub fn code(&self, function: &Function) -> &[u8] {
        self.code.get(function.code)
    }

    pub fn constants(&self, function: &Function) -> &[Constant] {
        self.constants.get(function.constants)
    }

    pub fn constant(&self, function: &Function, index: u32) -> &Constant {
        &self.constants(function)[index as usize]
    }

    pub fn handlers(&self, function: &Function) -> &[HandlerEntry] {
        self.handlers.get(function.handlers)
    }

    pub fn name(&self, function: &Function) -> Option<&[u8]> {
        function.name.map(|span| self.names.get(span))
    }
}
