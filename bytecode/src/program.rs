#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionId(pub u32);

impl FunctionId {
    /// The script body / entry function.
    pub const SCRIPT: FunctionId = FunctionId(0);

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// How a frontend was invoked; shapes top-level declaration handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceMode {
    Script,
    /// unresolved names become dynamic lookups through the caller's context chain.
    Eval,
    /// top-level declarations become global object properties
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

/// Constant-pool entry. The VM materializes these into heap objects:
/// interned strings, `Float`s, and shared callable templates for closures.
/// The well-known singletons (undefined/null/true/false/0) load through
/// their dedicated `Load*` opcodes instead of the constant pool.
///
/// `Eq`/`Hash` compare [`Float`] by bit pattern, so deduplication never
/// merges distinct values (`0.0` / `-0.0`); NaN re-dedups with itself.
#[derive(Debug, Clone)]
pub enum Constant {
    /// Interned string (property names, string literals)
    String(Box<[u8]>),
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
    ContextNames(Vec<Box<[u8]>>),
    /// %Object.prototype% (base-class prototype parent)
    ObjectPrototype,
    /// %Function.prototype% (base-class constructor parent)
    FunctionPrototype,
}

impl Constant {
    /// Key identity for pool deduplication: bit-exact for floats.
    fn key(&self) -> ConstantKey<'_> {
        match self {
            Self::String(bytes) => ConstantKey::String(bytes),
            Self::Float(f) => ConstantKey::Float(f.to_bits()),
            Self::Smi(v) => ConstantKey::Smi(*v),
            Self::Callable(id) => ConstantKey::Callable(*id),
            Self::ContextNames(names) => ConstantKey::ContextNames(names),
            Self::ObjectPrototype => ConstantKey::ObjectPrototype,
            Self::FunctionPrototype => ConstantKey::FunctionPrototype,
        }
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Constant {}

impl core::hash::Hash for Constant {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

/// Borrowed, hashable view of a [`Constant`] (floats by bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstantKey<'a> {
    String(&'a [u8]),
    Float(u64),
    Smi(i64),
    Callable(FunctionId),
    ContextNames(&'a [Box<[u8]>]),
    ObjectPrototype,
    FunctionPrototype,
}

/// `layout [try_start, try_end, handler_pc]`: a half-open bytecode region.
/// An exception raised at a pc inside `[try_start, try_end)` transfers
/// control to `handler_pc` with the exception in the accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandlerEntry {
    pub try_start: usize,
    pub try_end: usize,
    pub handler_pc: usize,
}

/// A finished, frozen function: emitted code plus its pools and frame
/// metadata. Produced by [`FnBuilder::finish`](crate::FnBuilder::finish).
#[derive(Debug, Clone, Default)]
pub struct Function {
    pub code: Vec<u8>,
    pub constants: Vec<Constant>,
    pub handlers: Vec<HandlerEntry>,
    pub name: Option<Box<[u8]>>,
    /// Frame size in stack slots.
    pub register_count: u32,
    pub kind: CallableKind,
    /// Frame layout: one register per formal parameter (patterns count one).
    pub arity: u32,
    /// JS-visible `length`: parameters before the first default/rest/pattern.
    pub length: u32,
    /// Feedback-vector slots (inline-cache state) the function needs.
    /// Property-access sites index into it via their feedback operand.
    pub feedback_count: u32,
    pub strict: bool,
}

/// An ordered function table: `functions[0]` is the entry body.
///
/// Functions are added in [`FunctionId`] order; constant-pool
/// [`Constant::Callable`] references are resolved against the same table
/// when the program is materialized.
#[derive(Debug, Default)]
pub struct Program {
    functions: Vec<Function>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(functions: usize) -> Self {
        Self {
            functions: Vec::with_capacity(functions),
        }
    }

    pub fn len(&self) -> usize {
        self.functions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Freeze a finished function into the table and return its handle.
    pub fn add_function(&mut self, function: Function) -> FunctionId {
        let id = FunctionId(self.functions.len() as u32);
        self.functions.push(function);
        id
    }

    pub fn function(&self, id: FunctionId) -> &Function {
        &self.functions[id.index()]
    }

    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.iter()
    }

    pub fn function_ids(&self) -> impl Iterator<Item = FunctionId> {
        (0..self.functions.len() as u32).map(FunctionId)
    }

    pub fn code<'a>(&self, function: &'a Function) -> &'a [u8] {
        &function.code
    }

    pub fn constants<'a>(&self, function: &'a Function) -> &'a [Constant] {
        &function.constants
    }

    pub fn constant<'a>(&self, function: &'a Function, index: u32) -> &'a Constant {
        &function.constants[index as usize]
    }

    pub fn handlers<'a>(&self, function: &'a Function) -> &'a [HandlerEntry] {
        &function.handlers
    }

    pub fn name<'a>(&self, function: &'a Function) -> Option<&'a [u8]> {
        function.name.as_deref()
    }
}
