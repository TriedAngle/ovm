use std::collections::HashMap;

use crate::token::{Span, TokenKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

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
    Get,
    Set,
}

#[derive(Clone, Copy, Debug)]
pub enum Node {
    NumberLiteral(f64),
    StringLiteral(Symbol),
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
    ObjectLiteral {
        props: NodeList,
    },
    ObjectProperty {
        key: NodeId,
        value: NodeId,
        kind: PropKind,
    },
    FunctionExpr {
        function: FunctionId,
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
        name: Symbol,
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
    FunctionDecl {
        function: FunctionId,
    },
    Empty,
}

pub struct FunctionInfo {
    pub span: Span,
    pub name: Option<Symbol>,
    pub params: Vec<Symbol>,
    /// body block root; `None` once lazy parsing can skip bodies
    pub body: Option<NodeId>,
    /// stable across a skipping (pre)parse and a later full re-parse
    pub literal_id: u32,
    pub is_declaration: bool,
    pub strict: bool,
    /// preparse data slot for lazy body skipping
    pub lazy_data: Option<Box<[u8]>>,
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
        }
    }

    pub fn add(&mut self, node: Node, span: Span) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.spans.push(span);
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
}

#[derive(Debug)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl ParseError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}..{}", self.message, self.span.start, self.span.end)
    }
}

impl std::error::Error for ParseError {}

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
}

#[derive(Clone, Copy, Debug)]
pub struct Bookmark {
    pos: u32,
}

pub struct Parser<S: CharStream> {
    stream: S,
    ast: Ast,
    errors: Vec<ParseError>,
}

impl<S: CharStream> Parser<S> {
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            ast: Ast::new(),
            errors: Vec::new(),
        }
    }

    pub fn ast(&self) -> &Ast {
        &self.ast
    }

    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    pub fn into_ast(self) -> Ast {
        self.ast
    }

    pub fn bookmark(&self) -> Bookmark {
        Bookmark {
            pos: self.stream.pos(),
        }
    }

    pub fn restore(&mut self, bookmark: Bookmark) {
        self.stream.seek(bookmark.pos);
    }

    /// The top level parses as an implicit function.
    pub fn parse_script(&mut self) -> Result<FunctionId, ParseError> {
        todo!("scanner + grammar land next")
    }
}
