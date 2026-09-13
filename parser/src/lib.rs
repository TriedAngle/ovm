pub mod parser;
pub mod resolver;
pub mod scanner;
pub mod token;

use std::collections::HashMap;

pub use parser::{ParseError, Parser};
pub use resolver::{FunctionLayout, Resolution, Resolved, resolve, resolve_for_eval};
pub use scanner::{Bookmark, ScanResult, Scanner};
pub use token::{Span, Token, TokenInfo, TokenKind, TokenValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopeKind {
    Script,
    Function,
    Block,
    /// for-loop head scope: `for (let i = ...)` binds there, not outside
    For,
    /// catch block scope; the param lives in it
    Catch,
    /// class inner scope (ES 15.7.14 ClassDefinitionEvaluation): holds the
    /// immutable class-name binding and the super home-object slots; owns a
    /// dedicated block context created per class evaluation
    Class,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclKind {
    /// function parameter; `Declaration.param_index` is its positional slot
    /// (none for the names bound out of a parameter pattern)
    Param,
    Var,
    Let,
    Const,
    /// function declaration (var-like in a function scope, lexical in a block)
    Function,
    CatchParam,
    /// a name bound by a parameter destructuring pattern: initialized by the
    /// prologue's destructure step (Let-like TDZ before that, var-like
    /// redeclaration rules)
    PatternParam,
    Class,
}

pub struct Declaration {
    pub name: Symbol,
    pub kind: DeclKind,
    pub span: Span,
    /// positional register index of a plain `DeclKind::Param` declaration
    /// (`None` for pattern-bound param names and every other kind)
    pub param_index: Option<u32>,
}

pub struct ScopeInfo {
    pub kind: ScopeKind,
    pub parent: Option<ScopeId>,
    pub decls: Vec<Declaration>,
    /// set on Script/Function scopes
    pub function: Option<FunctionId>,
    pub strict: bool,
    /// direct eval call somewhere in this function's code: bindings in the
    /// whole visible chain must stay dynamic-safe (context-allocated)
    pub calls_eval: bool,
    /// this function contains a nested function/class/eval anywhere —
    /// gates the for-loop per-iteration-environment desugar
    pub contains_function_or_eval: bool,
}

impl ScopeInfo {
    fn new(kind: ScopeKind, parent: Option<ScopeId>) -> Self {
        Self {
            kind,
            parent,
            decls: Vec::new(),
            function: None,
            strict: false,
            calls_eval: false,
            contains_function_or_eval: false,
        }
    }
}

/// (start, len) slice of the arena's `lists` pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeList {
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarKind {
    Var,
    Let,
    Const,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropKind {
    Init,
    /// method shorthand `{ foo() {} }`
    Method,
    Get,
    Set,
    /// class field definition `x = init` (instance or static)
    Field,
}

#[derive(Clone, Copy, Debug)]
pub enum Node {
    NumberLiteral(f64),
    StringLiteral(Symbol),
    /// raw literal text without the trailing `n`; radix via `0x`/`0b`/`0o` prefix
    BigIntLiteral(Symbol),
    BoolLiteral(bool),
    NullLiteral,
    Identifier {
        sym: Symbol,
    },
    This,
    Unary {
        op: TokenKind,
        expr: NodeId,
    },
    Update {
        op: TokenKind,
        prefix: bool,
        target: NodeId,
    },
    Binary {
        op: TokenKind,
        lhs: NodeId,
        rhs: NodeId,
    },
    Assign {
        op: TokenKind,
        target: NodeId,
        value: NodeId,
    },
    Conditional {
        cond: NodeId,
        then: NodeId,
        else_: NodeId,
    },
    Call {
        callee: NodeId,
        args: NodeList,
    },
    /// args: None for `new f` (no parens)
    New {
        callee: NodeId,
        args: Option<NodeList>,
    },
    /// a.b (computed = false) or a[b]
    Property {
        object: NodeId,
        key: NodeId,
        computed: bool,
    },
    ArrayLiteral {
        elements: NodeList,
    },
    /// array elision in `[1, , 2]`
    Hole,
    /// `...expr` in array literals, call args, and object literals
    Spread {
        expr: NodeId,
    },
    ObjectLiteral {
        props: NodeList,
    },
    ObjectProperty {
        key: NodeId,
        value: NodeId,
        kind: PropKind,
        computed: bool,
    },
    FunctionExpr {
        function: FunctionId,
    },
    ClassExpr {
        class: ClassId,
    },
    /// `super.x` / `super[key]` (ES 15.4.2). The home object is resolved
    /// lexically from the enclosing class scope; `is_static` records which
    /// home slot (prototype vs constructor) the enclosing member uses.
    SuperProperty {
        key: NodeId,
        computed: bool,
        is_static: bool,
    },
    /// `super(...)` (ES 15.4.3): construct the superclass with the current
    /// frame's new.target and bind the result to `this`
    SuperCall {
        args: NodeList,
    },
    /// `new.target` (ES 13.3.11): the active [[Construct]] target, or
    /// undefined outside construction; arrows delegate to the enclosing
    /// non-arrow function
    NewTarget,
    /// `#name` reference: `this.#x`, `#x in obj`. The symbol is the hidden
    /// class-scope declaration (`.priv.#x`), resolved against the enclosing
    /// class's private environment
    PrivateName {
        sym: Symbol,
    },

    // binding & assignment patterns (ES 14.13)
    /// `[a, b = 1, ...rest]`; `elements` are PatternElement | Hole (elision)
    /// | PatternRest
    ArrayPattern {
        elements: NodeList,
    },
    /// `{a, b: c = 1, ...rest}`; `props` are PatternProperty | PatternRest
    ObjectPattern {
        props: NodeList,
    },
    /// one array-pattern element or the value of a property/parameter:
    /// the binding or nested pattern target plus an optional default
    PatternElement {
        target: NodeId,
        default: Option<NodeId>,
    },
    /// `{ key: target = default }` (shorthand keys carry an Identifier
    /// target); `computed` marks `[expr]` keys
    PatternProperty {
        key: NodeId,
        value: NodeId,
        computed: bool,
    },
    /// `...target` in an array or object pattern (always last)
    PatternRest {
        target: NodeId,
    },

    // statements & declarations
    ExprStmt {
        expr: NodeId,
    },
    VarDecl {
        kind: VarKind,
        /// VarDeclarator nodes
        decls: NodeList,
    },
    VarDeclarator {
        /// Identifier node or a binding pattern
        target: NodeId,
        init: Option<NodeId>,
    },
    Block {
        stmts: NodeList,
    },
    If {
        cond: NodeId,
        then: NodeId,
        else_: Option<NodeId>,
    },
    While {
        cond: NodeId,
        body: NodeId,
    },
    For {
        init: Option<NodeId>,
        cond: Option<NodeId>,
        next: Option<NodeId>,
        body: NodeId,
    },
    Return {
        value: Option<NodeId>,
    },
    Break {
        label: Option<Symbol>,
    },
    Continue {
        label: Option<Symbol>,
    },
    Throw {
        expr: NodeId,
    },
    TryCatch {
        try_block: NodeId,
        /// Identifier node or a binding pattern
        catch_param: Option<NodeId>,
        catch_block: Option<NodeId>,
        finally_block: Option<NodeId>,
    },
    Switch {
        disc: NodeId,
        cases: NodeList,
    },
    /// `test: None` marks the `default:` clause
    SwitchCase {
        test: Option<NodeId>,
        stmts: NodeList,
    },
    Labeled {
        label: Symbol,
        body: NodeId,
    },
    FunctionDecl {
        function: FunctionId,
    },
    ClassDecl {
        class: ClassId,
    },
    Empty,
}

/// One formal parameter: a binding target (Identifier node or pattern),
/// an optional default initializer, and the rest flag (ES 15.1).
#[derive(Clone, Copy, Debug)]
pub struct Param {
    /// Identifier node or an ArrayPattern/ObjectPattern node
    pub target: NodeId,
    pub default: Option<NodeId>,
    pub rest: bool,
}

impl Param {
    /// Parameter lists containing any of these make the function's parameter
    /// list "non-simple" (ES 15.1.2): names become unique, TDZ-initialized
    /// left-to-right, and `length` truncates.
    pub fn is_non_simple(&self, ast: &Ast) -> bool {
        self.rest
            || self.default.is_some()
            || !matches!(ast.node(self.target), Node::Identifier { .. })
    }
}

pub struct FunctionInfo {
    pub span: Span,
    pub name: Option<Symbol>,
    pub params: Vec<Param>,
    /// JS-visible `length`: the number of parameters preceding the first
    /// default / rest / pattern parameter (ES 20.2.3)
    pub formal_length: u32,
    /// body block root; `None` once lazy parsing can skip bodies
    pub body: Option<NodeId>,
    /// stable across a skipping (pre)parse and a later full re-parse
    pub literal_id: u32,
    pub is_declaration: bool,
    pub kind: FunctionKind,
    pub strict: bool,
    /// synthesized field-initializer functions only: the field's key node.
    /// When the initializer returns an anonymous function definition, the
    /// value is NamedEvaluation'd after this key (ES 15.7.19).
    pub field_key: Option<NodeId>,
    /// preparse data slot for lazy body skipping
    pub lazy_data: Option<Box<[u8]>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FunctionKind {
    #[default]
    Normal,
    Generator,
    Arrow,
    Method,
    Getter,
    Setter,
    BaseClassConstructor,
    DerivedClassConstructor,
    /// synthesized `constructor(...args) { super(...args) }` for derived
    /// classes without an explicit constructor; compiles to an implicit
    /// forward-all-arguments super call
    DefaultDerivedConstructor,
}

impl FunctionKind {
    pub const fn is_generator(self) -> bool {
        matches!(self, Self::Generator)
    }

    pub const fn is_arrow(self) -> bool {
        matches!(self, Self::Arrow)
    }

    /// class-member function kinds: `super.x` is allowed inside these
    pub const fn is_class_member(self) -> bool {
        matches!(
            self,
            Self::Method
                | Self::Getter
                | Self::Setter
                | Self::BaseClassConstructor
                | Self::DerivedClassConstructor
        )
    }

    pub const fn is_derived_class_constructor(self) -> bool {
        matches!(
            self,
            Self::DerivedClassConstructor | Self::DefaultDerivedConstructor
        )
    }
}

pub struct ClassMember {
    pub key: NodeId,
    /// FunctionExpr node: the method, or the synthesized field-initializer
    /// function for `PropKind::Field` members (body `return <init>;`)
    pub value: NodeId,
    pub kind: PropKind,
    pub is_static: bool,
    pub is_constructor: bool,
    pub computed: bool,
    /// private name fields (`#x`): key is a PrivateName node
    pub is_private: bool,
}

pub struct ClassInfo {
    pub span: Span,
    pub name: Option<Symbol>,
    pub superclass: Option<NodeId>,
    pub members: Vec<ClassMember>,
    /// the constructor function: the explicit one, or the synthesized
    /// default (`constructor() {}` / `constructor(...args){super(...args)}`)
    pub ctor: FunctionId,
    /// some member body (or nested computed key resolving here) uses
    /// `super.x`: the class context carries home-object slots
    pub uses_super: bool,
    /// hidden class-scope declarations holding the home objects for
    /// `super.x` resolution (the prototype and the constructor)
    pub home: Option<Symbol>,
    pub static_home: Option<Symbol>,
    /// private names declared by this class in declaration order (hidden
    /// class-scope const declarations `.priv.#x`, holding a fresh private
    /// Symbol per class evaluation)
    pub privates: Vec<Symbol>,
}

/// parse-local byte-slice interner; heap internalization at materialization
#[derive(Default)]
pub struct SymbolTable {
    map: HashMap<Vec<u8>, Symbol>,
    strings: Vec<Vec<u8>>,
}

impl SymbolTable {
    pub fn intern(&mut self, s: &[u8]) -> Symbol {
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let sym = Symbol(self.strings.len() as u32);
        let owned = s.to_vec();
        self.strings.push(owned.clone());
        self.map.insert(owned, sym);
        sym
    }

    pub fn get(&self, sym: Symbol) -> &[u8] {
        &self.strings[sym.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

pub struct Ast {
    nodes: Vec<Node>,
    spans: Vec<Span>,
    /// shared pool backing every NodeList
    lists: Vec<NodeId>,
    strings: SymbolTable,
    functions: Vec<FunctionInfo>,
    classes: Vec<ClassInfo>,
    scopes: Vec<ScopeInfo>,
    /// parallel to `nodes`: the scope a scope-introducing node owns
    /// (Block, For, function bodies)
    node_scope: Vec<Option<ScopeId>>,
}

impl Default for Ast {
    fn default() -> Self {
        Self::new()
    }
}

impl Ast {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            spans: Vec::new(),
            lists: Vec::new(),
            strings: SymbolTable::default(),
            functions: Vec::new(),
            classes: Vec::new(),
            scopes: Vec::new(),
            node_scope: Vec::new(),
        }
    }

    pub fn add(&mut self, node: Node, span: Span) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.spans.push(span);
        self.node_scope.push(None);
        id
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn span(&self, id: NodeId) -> Span {
        self.spans[id.0 as usize]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn list(&mut self, items: &[NodeId]) -> NodeList {
        let start = self.lists.len() as u32;
        self.lists.extend_from_slice(items);
        NodeList {
            start,
            len: items.len() as u32,
        }
    }

    pub fn list_items(&self, list: NodeList) -> &[NodeId] {
        &self.lists[list.start as usize..(list.start + list.len) as usize]
    }

    pub fn intern(&mut self, s: &[u8]) -> Symbol {
        self.strings.intern(s)
    }

    pub fn symbol(&self, sym: Symbol) -> &[u8] {
        self.strings.get(sym)
    }

    pub fn symbol_count(&self) -> usize {
        self.strings.len()
    }

    pub fn add_function(&mut self, info: FunctionInfo) -> FunctionId {
        let id = FunctionId(self.functions.len() as u32);
        self.functions.push(info);
        id
    }

    pub fn function(&self, id: FunctionId) -> &FunctionInfo {
        &self.functions[id.0 as usize]
    }

    pub fn function_mut(&mut self, id: FunctionId) -> &mut FunctionInfo {
        &mut self.functions[id.0 as usize]
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    pub fn add_class(&mut self, info: ClassInfo) -> ClassId {
        let id = ClassId(self.classes.len() as u32);
        self.classes.push(info);
        id
    }

    pub fn class(&self, id: ClassId) -> &ClassInfo {
        &self.classes[id.0 as usize]
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    // ---- scopes ----

    pub fn add_scope(&mut self, kind: ScopeKind, parent: Option<ScopeId>) -> ScopeId {
        let id = ScopeId(self.scopes.len() as u32);
        self.scopes.push(ScopeInfo::new(kind, parent));
        id
    }

    pub fn scope(&self, id: ScopeId) -> &ScopeInfo {
        &self.scopes[id.0 as usize]
    }

    pub fn scope_mut(&mut self, id: ScopeId) -> &mut ScopeInfo {
        &mut self.scopes[id.0 as usize]
    }

    pub fn scope_count(&self) -> usize {
        self.scopes.len()
    }

    pub fn declare(&mut self, scope: ScopeId, name: Symbol, kind: DeclKind, span: Span) {
        self.scopes[scope.0 as usize].decls.push(Declaration {
            name,
            kind,
            span,
            param_index: None,
        });
    }

    /// Declare a positional formal parameter (register-backed).
    pub fn declare_param(
        &mut self,
        scope: ScopeId,
        name: Symbol,
        kind: DeclKind,
        span: Span,
        param_index: u32,
    ) {
        self.scopes[scope.0 as usize].decls.push(Declaration {
            name,
            kind,
            span,
            param_index: Some(param_index),
        });
    }

    /// Replace the node stored at `id` (pattern cover-grammar rewrites keep
    /// node ids stable so parent references survive).
    pub fn replace(&mut self, id: NodeId, node: Node, span: Span) {
        self.nodes[id.0 as usize] = node;
        self.spans[id.0 as usize] = span;
    }

    pub fn set_node_scope(&mut self, node: NodeId, scope: ScopeId) {
        self.node_scope[node.0 as usize] = Some(scope);
    }

    pub fn node_scope(&self, node: NodeId) -> Option<ScopeId> {
        self.node_scope[node.0 as usize]
    }

    pub(crate) fn set_symbol_table(&mut self, strings: SymbolTable) {
        self.strings = strings;
    }
}

/// Pull-based stream of decoded code points; the only encoding-specific
/// layer. UTF-16 streams may yield lone surrogates (0xD800..=0xDFFF).
pub trait CharStream {
    /// &mut self: buffered/chunked streams may need to fetch data to answer
    fn peek(&mut self) -> Option<u32>;
    fn advance(&mut self);
    /// byte offset for UTF-8, code units for UTF-16
    fn pos(&self) -> u32;
    /// backwards-only, over already-consumed data
    fn seek(&mut self, pos: u32);
    /// zero-copy access to a consumed range; `None` if the bytes are not
    /// contiguous in memory (chunked streams), callers then fall back to
    /// copying via seek/read
    fn slice(&self, _start: u32, _end: u32) -> Option<&[u8]> {
        None
    }
}

pub struct Utf8SliceStream<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Utf8SliceStream<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }
}

impl CharStream for Utf8SliceStream<'_> {
    fn peek(&mut self) -> Option<u32> {
        self.src[self.pos..].chars().next().map(|c| c as u32)
    }

    fn advance(&mut self) {
        if let Some(c) = self.src[self.pos..].chars().next() {
            self.pos += c.len_utf8();
        }
    }

    fn pos(&self) -> u32 {
        self.pos as u32
    }

    fn seek(&mut self, pos: u32) {
        let pos = pos as usize;
        assert!(pos <= self.pos, "streams only seek backwards");
        assert!(self.src.is_char_boundary(pos), "seek to non-boundary {pos}");
        self.pos = pos;
    }

    fn slice(&self, start: u32, end: u32) -> Option<&[u8]> {
        Some(&self.src.as_bytes()[start as usize..end as usize])
    }
}
