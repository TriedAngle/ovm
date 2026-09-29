use crate::plumbing::{ByteSpan, Symbol, SymbolTable};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeList {
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    /// `name: value`
    Named,
    /// `name*: value` (a parent / trait)
    Parent,
    /// `[i]: value`
    Element,
}

/// Prefix operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `!x`
    Not,
}

/// Infix operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// `||`
    Or,
    /// `&&`
    And,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Mod,
}

/// The single structured literal: `{ slots | params | body }`.
///
/// Every part is optional. Slots-only is a plain object; a `|params|`
/// marker or a non-empty body makes it callable (its body is compiled
/// as a function whose receiver is the object itself). Slot values are
/// evaluated in the enclosing scope when the object is created.
#[derive(Clone, Debug)]
pub struct ObjectParts {
    pub slots: NodeList,
    /// `None` when there is no `|...|` marker (even `{ || body }` keeps
    /// `Some` with an empty list: it is callable)
    pub params: Option<NodeList>,
    /// always a `StmtList` node, possibly empty
    pub body: NodeId,
}

#[derive(Clone, Debug)]
pub enum Node {
    Number(f64),
    String(Symbol),
    Bool(bool),
    Null,
    Self_,
    Ident(Symbol),

    Object(ObjectParts),
    Slot {
        kind: SlotKind,
        /// `Ident` for named/parent slots, an arbitrary expression for elements
        key: NodeId,
        value: NodeId,
    },
    /// `[a, b, c]` — sugar for element slots
    Array {
        elements: NodeList,
    },
    StmtList {
        stmts: NodeList,
    },

    /// `recv.name` slot read
    Get {
        recv: NodeId,
        name: Symbol,
    },
    /// `recv[key]` / `recv.0` element read
    Index {
        recv: NodeId,
        key: NodeId,
    },
    /// `recv.name(args)`
    Send {
        recv: NodeId,
        name: Symbol,
        args: NodeList,
    },
    /// `callee(args)` — the callee itself is the receiver (`self`)
    Call {
        callee: NodeId,
        args: NodeList,
    },

    Assign {
        target: NodeId,
        value: NodeId,
    },
    /// prefix `-` / `!`
    Unary {
        op: UnaryOp,
        expr: NodeId,
    },
    Binary {
        op: BinaryOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    /// `return expr` returns from the enclosing block
    Return {
        value: NodeId,
    },

    Let {
        name: Symbol,
        init: NodeId,
    },
    ExprStmt {
        expr: NodeId,
    },

    /// `if cond { A } else { B }`; the branches are `StmtList`s compiled
    /// inline in the enclosing function
    If {
        cond: NodeId,
        then: NodeId,
        else_: Option<NodeId>,
    },
}

impl Node {
    /// The statement list of an object body, if it is one.
    pub fn body_stmts<'a>(&self, ast: &'a Ast) -> &'a [NodeId] {
        match self {
            Node::Object(parts) => ast.object_body_stmts(parts),
            Node::StmtList { stmts } => ast.list_items(*stmts),
            _ => &[],
        }
    }
}

pub struct Ast {
    nodes: Vec<Node>,
    spans: Vec<ByteSpan>,
    /// shared pool backing every NodeList
    lists: Vec<NodeId>,
    strings: SymbolTable,
    /// root `StmtList` of the parsed unit, set by the parser
    root: Option<NodeId>,
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
            root: None,
        }
    }

    pub fn set_root(&mut self, root: NodeId) {
        self.root = Some(root);
    }

    pub fn root(&self) -> Option<NodeId> {
        self.root
    }

    pub fn add(&mut self, node: Node, span: ByteSpan) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.spans.push(span);
        id
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn span(&self, id: NodeId) -> ByteSpan {
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

    /// The statement list of an object body (an empty slice for
    /// non-`StmtList` bodies, which do not occur).
    pub fn object_body_stmts(&self, parts: &ObjectParts) -> &[NodeId] {
        match self.node(parts.body) {
            Node::StmtList { stmts } => self.list_items(*stmts),
            _ => &[],
        }
    }

    /// Whether a `{ ... }` literal is compiled as a function: it carries a
    /// `|params|` marker or a non-empty body.
    pub fn is_closure(&self, id: NodeId) -> bool {
        match self.node(id) {
            Node::Object(parts) => {
                parts.params.is_some() || !self.object_body_stmts(parts).is_empty()
            }
            _ => false,
        }
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

    pub fn set_symbol_table(&mut self, strings: SymbolTable) {
        self.strings = strings;
    }
}
