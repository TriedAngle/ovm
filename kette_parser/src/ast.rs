use parser_utils::{ByteSpan, Symbol, SymbolTable};

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

#[derive(Clone, Debug)]
pub enum Node {
    Number(f64),
    String(Symbol),
    Bool(bool),
    Null,
    Self_,
    Ident(Symbol),

    Object {
        slots: NodeList,
    },
    Slot {
        kind: SlotKind,
        /// `Ident` for named/parent slots, an arbitrary expression for elements
        key: NodeId,
        value: NodeId,
    },
    Array {
        elements: NodeList,
    },
    /// `{ |params| body }` or a zero-arg `{ body }` block.
    Block {
        params: NodeList,
        body: NodeId,
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

    /// `try { body } catch name { handler }`; both arms are `Block`s and
    /// the catch binding is the handler's first parameter
    Try {
        body: NodeId,
        handler: NodeId,
    },

    Let {
        name: Symbol,
        init: NodeId,
    },
    ExprStmt {
        expr: NodeId,
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
    ForIn {
        name: Symbol,
        iter: NodeId,
        body: NodeId,
    },
    Match {
        scrut: NodeId,
        arms: NodeList,
    },
    /// one `target -> handler` arm; `target: None` is the `else` arm
    MatchArm {
        target: Option<NodeId>,
        handler: NodeId,
    },
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

    /// Overwrite the node stored at `id` in place. Children keep their
    /// ids, so parents referencing `id` observe the replacement. Used by
    /// the lowering pass to rewrite nodes without rebuilding the tree.
    pub fn replace(&mut self, id: NodeId, node: Node) {
        self.nodes[id.0 as usize] = node;
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
