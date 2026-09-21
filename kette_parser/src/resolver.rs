use crate::plumbing::{ByteSpan, Symbol};

use crate::{Ast, Node, NodeId, SlotKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopeKind {
    /// the top-level program body
    Script,
    /// a block / lambda body
    Block,
}

#[derive(Clone, Debug)]
pub struct Declaration {
    pub name: Symbol,
    pub span: ByteSpan,
}

pub struct ScopeInfo {
    pub kind: ScopeKind,
    pub parent: Option<ScopeId>,
    pub decls: Vec<Declaration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// declared in the current block scope
    Local { scope: ScopeId, decl: u32 },
    /// declared in an enclosing block scope, `depth` blocks up
    Capture {
        scope: ScopeId,
        decl: u32,
        depth: u32,
    },
    /// unbound: a runtime lookup
    Global,
}

pub struct Resolved {
    /// parallel to the node arena; `Some` on identifier uses only
    resolutions: Vec<Option<Resolution>>,
    /// scope owned by a `Block` / the root `StmtList` node
    node_scopes: Vec<Option<ScopeId>>,
    scopes: Vec<ScopeInfo>,
}

impl Resolved {
    pub fn resolution(&self, node: NodeId) -> Option<Resolution> {
        self.resolutions[node.0 as usize]
    }

    /// The scope a block node (or the root statement list) owns.
    pub fn scope_of(&self, node: NodeId) -> Option<ScopeId> {
        self.node_scopes[node.0 as usize]
    }

    pub fn scope(&self, id: ScopeId) -> &ScopeInfo {
        &self.scopes[id.0 as usize]
    }

    pub fn scope_count(&self) -> usize {
        self.scopes.len()
    }
}

pub fn resolve(ast: &Ast) -> Resolved {
    let mut resolver = Resolver {
        ast,
        resolutions: vec![None; ast.node_count()],
        node_scopes: vec![None; ast.node_count()],
        scopes: Vec::new(),
        stack: Vec::new(),
    };
    if let Some(root) = ast.root() {
        resolver.resolve_root(root);
    }
    Resolved {
        resolutions: resolver.resolutions,
        node_scopes: resolver.node_scopes,
        scopes: resolver.scopes,
    }
}

struct Resolver<'a> {
    ast: &'a Ast,
    resolutions: Vec<Option<Resolution>>,
    node_scopes: Vec<Option<ScopeId>>,
    scopes: Vec<ScopeInfo>,
    /// innermost scope last
    stack: Vec<ScopeId>,
}

impl Resolver<'_> {
    fn push_scope(&mut self, kind: ScopeKind) -> ScopeId {
        let parent = self.stack.last().copied();
        let id = ScopeId(self.scopes.len() as u32);
        self.scopes.push(ScopeInfo {
            kind,
            parent,
            decls: Vec::new(),
        });
        self.stack.push(id);
        id
    }

    fn pop_scope(&mut self) {
        self.stack.pop();
    }

    fn current_scope(&self) -> ScopeId {
        *self.stack.last().expect("a scope is always active")
    }

    fn declare(&mut self, name: Symbol, span: ByteSpan) {
        let scope = self.current_scope();
        self.scopes[scope.0 as usize]
            .decls
            .push(Declaration { name, span });
    }

    fn resolve_root(&mut self, root: NodeId) {
        let scope = self.push_scope(ScopeKind::Script);
        self.node_scopes[root.0 as usize] = Some(scope);
        let ast = self.ast;
        if let Node::StmtList { stmts } = ast.node(root) {
            let stmts = *stmts;
            self.collect_decls(ast.list_items(stmts));
            for &stmt in ast.list_items(stmts) {
                self.resolve_node(stmt);
            }
        }
        self.pop_scope();
    }

    /// Declare the direct `let` bindings of a statement list. Nested blocks
    /// are not descended into: they own their own scope.
    fn collect_decls(&mut self, stmts: &[NodeId]) {
        let ast = self.ast;
        for &stmt in stmts {
            if let Node::Let { name, .. } = ast.node(stmt) {
                let span = ast.span(stmt);
                self.declare(*name, span);
            }
        }
    }

    /// Enter a block's scope: params, then any `extra` bindings (a `for`
    /// loop variable), then the block's own `let`s; then walk the body.
    fn resolve_block(&mut self, block: NodeId, extra: &[Symbol]) {
        let scope = self.push_scope(ScopeKind::Block);
        self.node_scopes[block.0 as usize] = Some(scope);
        let ast = self.ast;
        if let Node::Block { params, body } = ast.node(block) {
            let (params, body) = (*params, *body);
            for &param in ast.list_items(params) {
                if let Node::Ident(sym) = ast.node(param) {
                    let (sym, span) = (*sym, ast.span(param));
                    self.declare(sym, span);
                }
            }
            for &sym in extra {
                let span = ast.span(block);
                self.declare(sym, span);
            }
            if let Node::StmtList { stmts } = ast.node(body) {
                let stmts = *stmts;
                self.collect_decls(ast.list_items(stmts));
                for &stmt in ast.list_items(stmts) {
                    self.resolve_node(stmt);
                }
            }
        }
        self.pop_scope();
    }

    fn resolve_node(&mut self, node: NodeId) {
        let ast = self.ast;
        match ast.node(node) {
            Node::StmtList { stmts } => {
                for &stmt in ast.list_items(*stmts) {
                    self.resolve_node(stmt);
                }
            }
            Node::Let { init, .. } => self.resolve_node(*init),
            Node::ExprStmt { expr } => self.resolve_node(*expr),
            Node::Ident(sym) => self.resolve_ident(node, *sym),
            Node::Block { .. } => self.resolve_block(node, &[]),
            Node::ForIn { name, iter, body } => {
                let (name, iter, body) = (*name, *iter, *body);
                self.resolve_node(iter);
                self.resolve_block(body, &[name]);
            }
            Node::If { cond, then, else_ } => {
                let (cond, then, else_) = (*cond, *then, *else_);
                self.resolve_node(cond);
                self.resolve_node(then);
                if let Some(else_) = else_ {
                    self.resolve_node(else_);
                }
            }
            Node::While { cond, body } => {
                let (cond, body) = (*cond, *body);
                self.resolve_node(cond);
                self.resolve_node(body);
            }
            Node::Match { scrut, arms } => {
                let (scrut, arms) = (*scrut, *arms);
                self.resolve_node(scrut);
                for &arm in ast.list_items(arms) {
                    if let Node::MatchArm { handler, .. } = ast.node(arm) {
                        // the arm target is a pattern label, not a value
                        self.resolve_node(*handler);
                    }
                }
            }
            Node::Object { slots } => {
                for &slot in ast.list_items(*slots) {
                    if let Node::Slot { kind, key, value } = ast.node(slot) {
                        let (kind, key, value) = (*kind, *key, *value);
                        if kind == SlotKind::Element {
                            self.resolve_node(key);
                        }
                        self.resolve_node(value);
                    }
                }
            }
            Node::Array { elements } => {
                for &element in ast.list_items(*elements) {
                    self.resolve_node(element);
                }
            }
            Node::Get { recv, .. } => self.resolve_node(*recv),
            Node::Index { recv, key } => {
                let (recv, key) = (*recv, *key);
                self.resolve_node(recv);
                self.resolve_node(key);
            }
            Node::Send { recv, args, .. } => {
                let (recv, args) = (*recv, *args);
                self.resolve_node(recv);
                for &arg in ast.list_items(args) {
                    self.resolve_node(arg);
                }
            }
            Node::Call { callee, args } => {
                let (callee, args) = (*callee, *args);
                self.resolve_node(callee);
                for &arg in ast.list_items(args) {
                    self.resolve_node(arg);
                }
            }
            Node::Assign { target, value } => {
                let (target, value) = (*target, *value);
                self.resolve_node(target);
                self.resolve_node(value);
            }
            Node::Unary { expr, .. } => self.resolve_node(*expr),
            Node::Binary { lhs, rhs, .. } => {
                let (lhs, rhs) = (*lhs, *rhs);
                self.resolve_node(lhs);
                self.resolve_node(rhs);
            }
            Node::Return { value } => self.resolve_node(*value),
            Node::Try { body, handler } => {
                let (body, handler) = (*body, *handler);
                self.resolve_node(body);
                self.resolve_node(handler);
            }
            Node::Slot { .. } | Node::MatchArm { .. } => {}
            Node::Number(_) | Node::String(_) | Node::Bool(_) | Node::Null | Node::Self_ => {}
        }
    }

    fn resolve_ident(&mut self, node: NodeId, sym: Symbol) {
        let top = self.stack.len() - 1;
        let mut found = None;
        for (depth_from_root, &scope) in self.stack.iter().enumerate().rev() {
            let decl = self.scopes[scope.0 as usize]
                .decls
                .iter()
                .rposition(|d| d.name == sym);
            if let Some(decl) = decl {
                found = Some((scope, decl as u32, (top - depth_from_root) as u32));
                break;
            }
        }
        self.resolutions[node.0 as usize] = Some(match found {
            Some((scope, decl, 0)) => Resolution::Local { scope, decl },
            Some((scope, decl, depth)) => Resolution::Capture { scope, decl, depth },
            None => Resolution::Global,
        });
    }
}
