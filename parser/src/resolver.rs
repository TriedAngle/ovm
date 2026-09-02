use std::collections::{HashMap, HashSet};

use crate::{Ast, DeclKind, FunctionId, Node, NodeId, ScopeId, Symbol};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// function parameter i → negative register -(i+2) (slot 0 = receiver)
    Param(u32),
    /// local register
    Local { reg: u32, hole_check: bool },
    /// captured: context slot in the context owned by `scope`'s function
    /// (the materializer derives the chain depth from `scope`)
    Context {
        slot: u32,
        scope: ScopeId,
        hole_check: bool,
    },
    /// no binding found in any scope: global object property
    GlobalObject,
    /// a direct eval in the chain forced dynamic handling
    Dynamic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FunctionLayout {
    /// local registers (params live in the negative-index param region)
    pub register_count: u32,
    pub context_slots: u32,
}

pub struct Resolved {
    /// parallel to the node arena; `Some` on Identifier nodes only
    pub resolutions: Vec<Option<Resolution>>,
    /// parallel to the function table
    pub layouts: Vec<FunctionLayout>,
    /// For nodes whose let/const head bindings are captured: the
    /// materializer must create a fresh environment per iteration
    /// (ECMA-262 13.7.4.8, CreatePerIterationEnvironment)
    pub per_iteration_loops: Vec<NodeId>,
}

impl Resolved {
    pub fn resolution(&self, node: NodeId) -> Option<Resolution> {
        self.resolutions[node.0 as usize]
    }

    pub fn layout(&self, f: FunctionId) -> FunctionLayout {
        self.layouts[f.0 as usize]
    }
}

#[derive(Clone, Copy)]
enum Pending {
    GlobalObject,
    Decl { scope: ScopeId, decl: u32 },
}

struct Resolver<'a> {
    ast: &'a Ast,
    /// per scope: name → index into ScopeInfo.decls
    decl_maps: Vec<HashMap<Symbol, u32>>,
    /// per scope: owning function
    fn_owner: Vec<FunctionId>,
    /// function/script scope per function
    fn_scope: Vec<ScopeId>,
    captured: HashSet<(ScopeId, u32)>,
    pending: Vec<Option<Pending>>,
    scope_stack: Vec<ScopeId>,
    fn_stack: Vec<FunctionId>,
}

pub fn resolve(ast: &Ast) -> Resolved {
    let n_scopes = ast.scope_count();
    let mut decl_maps = vec![HashMap::new(); n_scopes];
    let mut fn_owner = vec![FunctionId(0); n_scopes];
    let mut fn_scope = vec![ScopeId(0); ast.function_count()];
    for i in 0..n_scopes {
        let id = ScopeId(i as u32);
        let scope = ast.scope(id);
        for (j, d) in scope.decls.iter().enumerate() {
            decl_maps[i].insert(d.name, j as u32);
        }
        // owning function: nearest enclosing Script/Function scope
        let mut s = id;
        loop {
            let info = ast.scope(s);
            if let Some(f) = info.function {
                fn_owner[i] = f;
                break;
            }
            s = info.parent.expect("scope chain ends at script scope");
        }
        if let Some(f) = scope.function {
            fn_scope[f.0 as usize] = id;
        }
    }

    let mut r = Resolver {
        ast,
        decl_maps,
        fn_owner,
        fn_scope,
        captured: HashSet::new(),
        pending: vec![None; ast.node_count()],
        scope_stack: Vec::new(),
        fn_stack: Vec::new(),
    };
    r.walk_function(FunctionId(0));

    r.allocate_all()
}

impl<'a> Resolver<'a> {
    fn walk_function(&mut self, fid: FunctionId) {
        let body = self.ast.function(fid).body.expect("function must be parsed");
        self.fn_stack.push(fid);
        self.walk_node(body);
        self.fn_stack.pop();
    }

    fn resolve_reference(&mut self, node: NodeId, name: Symbol) {
        let current_fn = *self.fn_stack.last().unwrap();
        let mut pending = Pending::GlobalObject;
        for &scope in self.scope_stack.iter().rev() {
            if let Some(&decl) = self.decl_maps[scope.0 as usize].get(&name) {
                if self.fn_owner[scope.0 as usize] != current_fn {
                    // captured across a function boundary → context-allocated
                    self.captured.insert((scope, decl));
                }
                pending = Pending::Decl { scope, decl };
                break;
            }
        }
        self.pending[node.0 as usize] = Some(pending);
    }

    fn walk_list(&mut self, list: crate::NodeList) {
        for &item in self.ast.list_items(list) {
            self.walk_node(item);
        }
    }

    fn walk_node(&mut self, id: NodeId) {
        // enter the scope this node introduces (if any)
        let entered = self.ast.node_scope(id);
        if let Some(scope) = entered {
            self.scope_stack.push(scope);
        }
        match *self.ast.node(id) {
            Node::Identifier { sym } => self.resolve_reference(id, sym),
            Node::Unary { expr, .. } => self.walk_node(expr),
            Node::Update { target, .. } => self.walk_node(target),
            Node::Binary { lhs, rhs, .. } => {
                self.walk_node(lhs);
                self.walk_node(rhs);
            }
            Node::Assign { target, value, .. } => {
                self.walk_node(target);
                self.walk_node(value);
            }
            Node::Conditional { cond, then, else_ } => {
                self.walk_node(cond);
                self.walk_node(then);
                self.walk_node(else_);
            }
            Node::Call { callee, args } => {
                self.walk_node(callee);
                self.walk_list(args);
            }
            Node::New { callee, args } => {
                self.walk_node(callee);
                if let Some(args) = args {
                    self.walk_list(args);
                }
            }
            Node::Property { object, key, computed } => {
                self.walk_node(object);
                if computed {
                    self.walk_node(key);
                }
            }
            Node::ArrayLiteral { elements } => self.walk_list(elements),
            Node::Spread { expr } => self.walk_node(expr),
            Node::ObjectLiteral { props } => self.walk_list(props),
            Node::ObjectProperty { key, value, computed, .. } => {
                if computed {
                    self.walk_node(key);
                }
                self.walk_node(value);
            }
            Node::FunctionExpr { function } | Node::FunctionDecl { function } => {
                self.walk_function(function)
            }
            Node::ClassExpr { class } | Node::ClassDecl { class } => {
                let info = self.ast.class(class);
                if let Some(sup) = info.superclass {
                    self.walk_node(sup);
                }
                for m in &info.members {
                    if m.computed {
                        self.walk_node(m.key);
                    }
                    self.walk_node(m.value); // FunctionExpr
                }
            }
            Node::ExprStmt { expr } => self.walk_node(expr),
            Node::VarDecl { decls, .. } => self.walk_list(decls),
            Node::VarDeclarator { init, .. } => {
                if let Some(init) = init {
                    self.walk_node(init);
                }
            }
            Node::Block { stmts } => self.walk_list(stmts),
            Node::If { cond, then, else_ } => {
                self.walk_node(cond);
                self.walk_node(then);
                if let Some(e) = else_ {
                    self.walk_node(e);
                }
            }
            Node::While { cond, body } => {
                self.walk_node(cond);
                self.walk_node(body);
            }
            Node::For {
                init,
                cond,
                next,
                body,
            } => {
                if let Some(n) = init {
                    self.walk_node(n);
                }
                if let Some(c) = cond {
                    self.walk_node(c);
                }
                if let Some(n) = next {
                    self.walk_node(n);
                }
                self.walk_node(body);
            }
            Node::Return { value } => {
                if let Some(v) = value {
                    self.walk_node(v);
                }
            }
            Node::Throw { expr } => self.walk_node(expr),
            Node::TryCatch {
                try_block,
                catch_block,
                finally_block,
                ..
            } => {
                self.walk_node(try_block);
                if let Some(c) = catch_block {
                    self.walk_node(c);
                }
                if let Some(f) = finally_block {
                    self.walk_node(f);
                }
            }
            Node::NumberLiteral(_)
            | Node::StringLiteral(_)
            | Node::BigIntLiteral(_)
            | Node::BoolLiteral(_)
            | Node::NullLiteral
            | Node::This
            | Node::Hole
            | Node::Empty
            | Node::Break { .. }
            | Node::Continue { .. } => {}
        }
        if entered.is_some() {
            self.scope_stack.pop();
        }
    }

    // -- slot allocation -------------------------------------------------------

    fn allocate_all(self) -> Resolved {
        let ast = self.ast;
        let mut layouts = vec![FunctionLayout::default(); ast.function_count()];
        // (scope, decl) → concrete slot kind
        let mut slots: HashMap<(ScopeId, u32), Resolution> = HashMap::new();

        for fid in 0..ast.function_count() {
            let fid = FunctionId(fid as u32);
            let fscope = self.fn_scope[fid.0 as usize];
            let calls_eval = ast.scope(fscope).calls_eval;
            let mut next_reg = 0u32;
            let mut next_ctx = 0u32;
            for s in 0..ast.scope_count() {
                let scope = ScopeId(s as u32);
                if self.fn_owner[s] != fid {
                    continue;
                }
                for (d, decl) in ast.scope(scope).decls.iter().enumerate() {
                    let key = (scope, d as u32);
                    let hole_check =
                        matches!(decl.kind, DeclKind::Let | DeclKind::Const | DeclKind::Class);
                    let forced = calls_eval || self.captured.contains(&key);
                    let res = match decl.kind {
                        DeclKind::Param if !forced => Resolution::Param(d as u32),
                        _ if forced => {
                            let slot = next_ctx;
                            next_ctx += 1;
                            Resolution::Context {
                                slot,
                                scope,
                                hole_check,
                            }
                        }
                        _ => {
                            let reg = next_reg;
                            next_reg += 1;
                            Resolution::Local {
                                reg,
                                hole_check,
                            }
                        }
                    };
                    slots.insert(key, res);
                }
            }
            layouts[fid.0 as usize] = FunctionLayout {
                register_count: next_reg,
                context_slots: next_ctx,
            };
        }

        let mut resolutions = vec![None; ast.node_count()];
        for (node, pending) in self.pending.iter().enumerate() {
            resolutions[node] = pending.as_ref().map(|p| match *p {
                Pending::GlobalObject => Resolution::GlobalObject,
                Pending::Decl { scope, decl } => slots[&(scope, decl)],
            });
        }

        // per-iteration environments: for-heads whose let/const bindings are
        // captured need a fresh copy per iteration (ECMA-262 13.7.4.8)
        let mut per_iteration_loops = Vec::new();
        for n in 0..ast.node_count() {
            let id = NodeId(n as u32);
            if !matches!(ast.node(id), Node::For { .. }) {
                continue;
            }
            let Some(scope) = ast.node_scope(id) else {
                continue;
            };
            let needs = ast.scope(scope).decls.iter().enumerate().any(|(d, decl)| {
                matches!(decl.kind, DeclKind::Let | DeclKind::Const)
                    && self.captured.contains(&(scope, d as u32))
            });
            if needs {
                per_iteration_loops.push(id);
            }
        }

        Resolved {
            resolutions,
            layouts,
            per_iteration_loops,
        }
    }
}
