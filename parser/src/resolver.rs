use std::collections::{HashMap, HashSet};

use crate::{Ast, DeclKind, FunctionId, Node, NodeId, ScopeId, ScopeKind, Symbol};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// function parameter i → negative register -(i+2) (slot 0 = receiver).
    /// `hole_check`: non-simple parameter lists initialize left-to-right
    /// with TDZ (references to later parameters from initializers throw)
    Param { index: u32, hole_check: bool },
    /// local register
    Local { reg: u32, hole_check: bool },
    /// captured: context slot, at the precomputed chain depth from the use
    /// site (each function and each context-creating class scope between
    /// the use and the hosting scope is one hop)
    Context {
        slot: u32,
        depth: u32,
        hole_check: bool,
    },
    /// no binding found in any scope: global object property
    GlobalObject,
    /// a direct eval in the chain forced dynamic handling
    Dynamic,
    /// `this` in an arrow function: the enclosing non-arrow function's
    /// receiver, stored in that function's hidden this-slot, `depth` hops
    /// from the arrow's own context
    This { owner: FunctionId, depth: u32 },
    /// `new.target` (ES 13.3.11): the frame's own new.target when
    /// `owner == use function`, otherwise the owner's context slot
    NewTarget { owner: FunctionId, depth: u32 },
    /// `super(...)`: the owning derived constructor (whose closure and
    /// new.target are needed when the call is delegated through arrows)
    SuperCall { owner: FunctionId, depth: u32 },
    /// `super.x`: the home-object slot in the enclosing class's context,
    /// `depth` hops away. `this_owner` is the nearest enclosing non-arrow
    /// function (the receiver), whose this-slot is `this_depth` hops away.
    Super {
        home_slot: u32,
        depth: u32,
        this_owner: FunctionId,
        this_depth: u32,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FunctionLayout {
    /// local registers (params live in the negative-index param region)
    pub register_count: u32,
    pub context_slots: u32,
    /// hidden context slot holding the receiver, when a nested arrow
    /// function uses `this` (allocated first among the hidden slots)
    pub this_slot: Option<u32>,
    /// hidden context slot holding new.target, when a nested arrow reads
    /// it or delegates super()
    pub new_target_slot: Option<u32>,
    /// hidden context slot holding the running closure, for derived
    /// constructors with an arrow-delegated super()
    pub this_function_slot: Option<u32>,
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
    /// per scope: name → decl index into `ScopeInfo.decls`
    decl_maps: Vec<HashMap<Symbol, u32>>,
    /// (scope, decl index) → concrete slot kind, for declarations whose
    /// name never appears as an Identifier node (var/function decls)
    slots: HashMap<(ScopeId, u32), Resolution>,
}

impl Resolved {
    pub fn resolution(&self, node: NodeId) -> Option<Resolution> {
        self.resolutions[node.0 as usize]
    }

    pub fn layout(&self, f: FunctionId) -> FunctionLayout {
        self.layouts[f.0 as usize]
    }

    /// The concrete slot kind of a declaration `name` in `scope`, for
    /// declarations the materializer stores into without an Identifier
    /// node (var declarators, function decls, catch params).
    pub fn resolution_for_decl(&self, scope: ScopeId, name: Symbol) -> Option<Resolution> {
        let idx = *self.decl_maps.get(scope.0 as usize)?.get(&name)?;
        self.slots.get(&(scope, idx)).copied()
    }
}

#[derive(Clone, Copy)]
enum Pending {
    GlobalObject,
    /// unresolved name compiled as a runtime chain walk (direct eval)
    Dynamic,
    This {
        owner: FunctionId,
        depth: u32,
    },
    NewTarget {
        owner: FunctionId,
        depth: u32,
    },
    SuperCall {
        owner: FunctionId,
        depth: u32,
    },
    Super {
        scope: ScopeId,
        decl: u32,
        depth: u32,
        this_owner: FunctionId,
        this_depth: u32,
    },
    Decl {
        scope: ScopeId,
        decl: u32,
        /// precomputed context hops from this use site to the hosting
        /// scope (used when the decl resolves to a Context slot)
        depth: u32,
    },
}

struct Resolver<'a> {
    ast: &'a Ast,
    mode: ResolveMode,
    /// the scope owned by the script function: in REPL mode its bindings
    /// are global object properties
    script_scope: ScopeId,
    /// per scope: name → index into ScopeInfo.decls
    decl_maps: Vec<HashMap<Symbol, u32>>,
    /// per scope: owning function
    fn_owner: Vec<FunctionId>,
    /// function/script scope per function
    fn_scope: Vec<ScopeId>,
    captured: HashSet<(ScopeId, u32)>,
    /// functions whose receiver a nested arrow captures
    captures_this: HashSet<FunctionId>,
    /// functions whose new.target a nested arrow reads (or whose arrow
    /// super() needs it)
    captures_new_target: HashSet<FunctionId>,
    /// derived constructors with an arrow-delegated super(): their closure
    /// is stored into a context slot (.this_function)
    needs_this_function: HashSet<FunctionId>,
    /// class scopes that create a context (named classes and classes
    /// whose members use super): each adds one hop to context depths
    ctx_classes: HashSet<ScopeId>,
    pending: Vec<Option<Pending>>,
    scope_stack: Vec<ScopeId>,
    fn_stack: Vec<FunctionId>,
}

pub fn resolve(ast: &Ast) -> Resolved {
    resolve_with_mode(ast, ResolveMode::Script)
}

/// Resolve with all unresolved names marked `Dynamic` (direct eval source:
/// free names must resolve through the caller's context chain at runtime).
pub fn resolve_for_eval(ast: &Ast) -> Resolved {
    resolve_with_mode(ast, ResolveMode::Eval)
}

/// Resolve a REPL entry: script-scope (top-level) bindings live on the
/// global object, so they persist across entries and can be redeclared.
pub fn resolve_repl(ast: &Ast) -> Resolved {
    resolve_with_mode(ast, ResolveMode::Repl)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResolveMode {
    Script,
    Eval,
    Repl,
}

fn resolve_with_mode(ast: &Ast, mode: ResolveMode) -> Resolved {
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
        mode,
        script_scope: fn_scope[0],
        decl_maps,
        fn_owner,
        fn_scope,
        captured: HashSet::new(),
        captures_this: HashSet::new(),
        captures_new_target: HashSet::new(),
        needs_this_function: HashSet::new(),
        ctx_classes: HashSet::new(),
        pending: vec![None; ast.node_count()],
        scope_stack: Vec::new(),
        fn_stack: Vec::new(),
    };
    // class-kind scopes that create a context: named classes (the inner
    // name binding), classes whose members use super (home objects), and
    // object literals with super-using methods — uniformly: every
    // class-kind scope with at least one declaration
    for s in 0..ast.scope_count() {
        let scope = ScopeId(s as u32);
        if ast.scope(scope).kind == ScopeKind::Class && !ast.scope(scope).decls.is_empty() {
            r.ctx_classes.insert(scope);
        }
    }
    r.walk_function(FunctionId(0));

    r.allocate_all()
}

impl<'a> Resolver<'a> {
    fn walk_function(&mut self, fid: FunctionId) {
        let info = self.ast.function(fid);
        let body = info.body.expect("function must be parsed");
        self.fn_stack.push(fid);
        // parameter initializers and pattern leaves resolve within the
        // function scope (they are initialized before the body runs)
        let fscope = self.fn_scope[fid.0 as usize];
        self.scope_stack.push(fscope);
        for p in &info.params {
            if let Some(d) = p.default {
                self.walk_node(d);
            }
            if !matches!(self.ast.node(p.target), Node::Identifier { .. }) {
                self.walk_node(p.target);
            }
        }
        self.scope_stack.pop();
        self.walk_node(body);
        self.fn_stack.pop();
    }

    /// Whether a scope introduces a context at runtime: every function
    /// does, plus the context-creating class scopes and for-head scopes
    /// with lexical bindings (per-iteration environments, ES 14.7.5).
    fn creates_ctx(&self, scope: ScopeId) -> bool {
        match self.ast.scope(scope).kind {
            ScopeKind::Script | ScopeKind::Function => true,
            ScopeKind::Class => self.ctx_classes.contains(&scope),
            // `for (let/const …; …)` and `for (let/const k in …)`: the
            // head bindings need a per-evaluation context. Uncaptured
            // heads use a single one; captured heads get a fresh copy
            // per iteration (per_iteration_loops)
            ScopeKind::For => !self.ast.scope(scope).decls.is_empty(),
            _ => false,
        }
    }

    /// The innermost scope at-or-outside `scope` that introduces a context:
    /// the one hosting any slot declared in `scope` (block scopes share
    /// their function's context).
    fn hosting_ctx(&self, scope: ScopeId) -> ScopeId {
        let mut scope = scope;
        loop {
            if self.creates_ctx(scope) {
                return scope;
            }
            scope = self
                .ast
                .scope(scope)
                .parent
                .expect("scope chain ends at script scope");
        }
    }

    /// Context hops from the use site (the innermost context-creating
    /// scope on the stack) to `hosting`, inclusive of `hosting`: the
    /// runtime `LoadContextSlot` depth. Class scopes between the use and
    /// the hosting scope each contribute a hop even though they do not
    /// correspond to a function boundary.
    fn context_hops(&self, hosting: ScopeId) -> u32 {
        let mut depth = 0u32;
        let mut passed_use = false;
        for &scope in self.scope_stack.iter().rev() {
            if !self.creates_ctx(scope) {
                continue;
            }
            if !passed_use {
                // the use site's own context: zero hops away from itself
                passed_use = true;
                if scope == hosting {
                    return 0;
                }
                continue;
            }
            depth += 1;
            if scope == hosting {
                return depth;
            }
        }
        unreachable!("hosting scope is on the resolver's scope stack")
    }

    fn resolve_reference(&mut self, node: NodeId, name: Symbol) {
        let current_fn = *self.fn_stack.last().unwrap();
        let mut pending = if self.mode == ResolveMode::Eval {
            Pending::Dynamic
        } else {
            Pending::GlobalObject
        };
        for &scope in self.scope_stack.iter().rev() {
            if let Some(&decl) = self.decl_maps[scope.0 as usize].get(&name) {
                if self.mode == ResolveMode::Repl && scope == self.script_scope {
                    // REPL mode: script-scope bindings live on the global
                    // object; never captured into a context
                    break;
                }
                if self.fn_owner[scope.0 as usize] != current_fn {
                    // captured across a function boundary → context-allocated
                    self.captured.insert((scope, decl));
                }
                let depth = self.context_hops(self.hosting_ctx(scope));
                pending = Pending::Decl { scope, decl, depth };
                break;
            }
        }
        self.pending[node.0 as usize] = Some(pending);
    }

    /// Resolve a `super.x` use: find the hidden home-object binding in the
    /// enclosing class scope on the scope stack.
    fn resolve_super(&mut self, node: NodeId, owner: FunctionId) -> Pending {
        let is_static = match *self.ast.node(node) {
            Node::SuperProperty { is_static, .. } => is_static,
            _ => unreachable!("super property node"),
        };
        let hidden: &[u8] = if is_static {
            b".static_home_object"
        } else {
            b".home_object"
        };
        for &scope in self.scope_stack.iter().rev() {
            if self.ast.scope(scope).kind != ScopeKind::Class {
                continue;
            }
            for (decl, d) in self.ast.scope(scope).decls.iter().enumerate() {
                if self.ast.symbol(d.name) == hidden {
                    let depth = self.context_hops(scope);
                    let this_depth = self.context_hops(self.fn_scope[owner.0 as usize]);
                    return Pending::Super {
                        scope,
                        decl: decl as u32,
                        depth,
                        this_owner: owner,
                        this_depth,
                    };
                }
            }
        }
        unreachable!("parser guarantees a class scope with home slots around super")
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
            Node::PrivateName { sym } => self.resolve_reference(id, sym),
            Node::This => {
                // nearest enclosing non-arrow function owns `this`
                let owner = self
                    .fn_stack
                    .iter()
                    .rev()
                    .copied()
                    .find(|&f| !self.ast.function(f).kind.is_arrow())
                    .expect("script function is never an arrow");
                self.captures_this.insert(owner);
                let depth = self.context_hops(self.fn_scope[owner.0 as usize]);
                self.pending[id.0 as usize] = Some(Pending::This { owner, depth });
            }
            Node::SuperProperty { key, computed, .. } => {
                if computed {
                    self.walk_node(key);
                }
                // receiver: like `this` — owned by the nearest non-arrow
                let owner = self
                    .fn_stack
                    .iter()
                    .rev()
                    .copied()
                    .find(|&f| !self.ast.function(f).kind.is_arrow())
                    .expect("script function is never an arrow");
                self.captures_this.insert(owner);
                self.pending[id.0 as usize] = Some(self.resolve_super(id, owner));
            }
            Node::SuperCall { args } => {
                self.walk_list(args);
                let owner = self
                    .fn_stack
                    .iter()
                    .rev()
                    .copied()
                    .find(|&f| !self.ast.function(f).kind.is_arrow())
                    .expect("script function is never an arrow");
                if owner != *self.fn_stack.last().expect("fn stack") {
                    // arrow-delegated super(): the constructor's closure
                    // and new.target must be context-visible
                    self.needs_this_function.insert(owner);
                    self.captures_new_target.insert(owner);
                    // `this` binding happens through the owner's slot too
                    self.captures_this.insert(owner);
                }
                let depth = self.context_hops(self.fn_scope[owner.0 as usize]);
                self.pending[id.0 as usize] = Some(Pending::SuperCall { owner, depth });
            }
            Node::NewTarget => {
                let owner = self
                    .fn_stack
                    .iter()
                    .rev()
                    .copied()
                    .find(|&f| !self.ast.function(f).kind.is_arrow())
                    .expect("script function is never an arrow");
                if owner != *self.fn_stack.last().expect("fn stack") {
                    // arrow-delegated new.target: read the owner's slot
                    self.captures_new_target.insert(owner);
                }
                let depth = self.context_hops(self.fn_scope[owner.0 as usize]);
                self.pending[id.0 as usize] = Some(Pending::NewTarget { owner, depth });
            }
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
            Node::Property {
                object,
                key,
                computed,
            } => {
                self.walk_node(object);
                // private-name keys (`obj.#x`) resolve against the class
                // scope even though they are never computed
                if computed || matches!(self.ast.node(key), Node::PrivateName { .. }) {
                    self.walk_node(key);
                }
            }
            Node::ArrayLiteral { elements } => self.walk_list(elements),
            Node::Spread { expr } => self.walk_node(expr),
            Node::ObjectLiteral { props } => self.walk_list(props),
            Node::ObjectProperty {
                key,
                value,
                computed,
                ..
            } => {
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
                    if m.computed || matches!(self.ast.node(m.key), Node::PrivateName { .. }) {
                        self.walk_node(m.key);
                    }
                    self.walk_node(m.value); // FunctionExpr
                }
            }
            Node::ExprStmt { expr } => self.walk_node(expr),
            Node::VarDecl { decls, .. } => self.walk_list(decls),
            Node::VarDeclarator { target, init } => {
                // targets always resolve: pattern leaves are stored
                // through by destructuring codegen, and a plain
                // identifier target is the per-iteration assignment
                // target of a for-in declaration head (its resolution
                // is keyed by node id like any reference)
                self.walk_node(target);
                if let Some(init) = init {
                    self.walk_node(init);
                }
            }
            // binding patterns: keys, defaults and leaf targets (defaults
            // may reference outer bindings; leaf Identifier targets carry
            // the store resolutions)
            Node::ArrayPattern { elements } => self.walk_list(elements),
            Node::ObjectPattern { props } => self.walk_list(props),
            Node::PatternElement { target, default } => {
                self.walk_node(target);
                if let Some(d) = default {
                    self.walk_node(d);
                }
            }
            Node::PatternProperty {
                key,
                value,
                computed,
            } => {
                if computed {
                    self.walk_node(key);
                }
                self.walk_node(value);
            }
            Node::PatternRest { target } => self.walk_node(target),
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
            Node::ForIn { left, object, body } => {
                self.walk_node(left);
                self.walk_node(object);
                self.walk_node(body);
            }
            Node::Return { value } => {
                if let Some(v) = value {
                    self.walk_node(v);
                }
            }
            Node::Throw { expr } => self.walk_node(expr),
            Node::Switch { disc, cases } => {
                self.walk_node(disc);
                for &case in self.ast.list_items(cases) {
                    if let Node::SwitchCase { test, stmts } = *self.ast.node(case) {
                        if let Some(test) = test {
                            self.walk_node(test);
                        }
                        self.walk_list(stmts);
                    }
                }
            }
            Node::SwitchCase { .. } => unreachable!("switch cases handled by the switch walk"),
            Node::Labeled { body, .. } => self.walk_node(body),
            Node::TryCatch {
                try_block,
                catch_param,
                catch_block,
                finally_block,
                ..
            } => {
                // the catch param scope covers only the catch block, not
                // the try block (a name in the try body must resolve
                // outside the catch scope)
                if entered.is_some() {
                    self.scope_stack.pop();
                }
                self.walk_node(try_block);
                if let Some(scope) = entered {
                    self.scope_stack.push(scope);
                }
                if let (Some(param), Some(c)) = (catch_param, catch_block) {
                    // the param pattern's leaves resolve within the catch
                    // scope (the store targets of the handler prologue);
                    // plain identifier params store by declaration
                    if !matches!(self.ast.node(*&param), Node::Identifier { .. }) {
                        self.walk_node(param);
                    }
                    self.walk_node(c);
                }
                if entered.is_some() {
                    self.scope_stack.pop();
                }
                if let Some(f) = finally_block {
                    self.walk_node(f);
                }
                // restore so the walk_node epilogue pops exactly once
                if let Some(scope) = entered {
                    self.scope_stack.push(scope);
                }
            }
            Node::NumberLiteral(_)
            | Node::StringLiteral(_)
            | Node::BigIntLiteral(_)
            | Node::BoolLiteral(_)
            | Node::NullLiteral
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
            // non-simple parameter lists (any default / pattern / rest)
            // initialize parameters left-to-right with TDZ: every parameter
            // read inside initializers carries a hole check
            let param_hole_check = ast
                .function(fid)
                .params
                .iter()
                .any(|p| p.is_non_simple(ast));
            let mut next_reg = 0u32;
            let mut next_ctx = 0u32;
            let mut this_slot = None;
            if self.captures_this.contains(&fid) {
                this_slot = Some(next_ctx);
                next_ctx += 1;
            }
            let mut new_target_slot = None;
            if self.captures_new_target.contains(&fid) {
                new_target_slot = Some(next_ctx);
                next_ctx += 1;
            }
            let mut this_function_slot = None;
            if self.needs_this_function.contains(&fid) {
                this_function_slot = Some(next_ctx);
                next_ctx += 1;
            }
            for s in 0..ast.scope_count() {
                let scope = ScopeId(s as u32);
                if self.fn_owner[s] != fid {
                    continue;
                }
                // class scopes and lexical for-head scopes own a
                // dedicated per-evaluation context with its own slot
                // space (created by CreateBlockContext)
                if matches!(ast.scope(scope).kind, ScopeKind::Class)
                    || (ast.scope(scope).kind == ScopeKind::For
                        && !ast.scope(scope).decls.is_empty())
                {
                    continue;
                }
                for (d, decl) in ast.scope(scope).decls.iter().enumerate() {
                    let key = (scope, d as u32);
                    if self.mode == ResolveMode::Repl && scope == self.script_scope {
                        // REPL mode: top-level bindings are global object
                        // properties, not locals or context slots
                        slots.insert(key, Resolution::GlobalObject);
                        continue;
                    }
                    let hole_check = matches!(
                        decl.kind,
                        DeclKind::Let | DeclKind::Const | DeclKind::Class | DeclKind::PatternParam
                    );
                    let forced = calls_eval || self.captured.contains(&key);
                    let res = match decl.kind {
                        DeclKind::Param if !forced => Resolution::Param {
                            index: decl.param_index.unwrap_or(0),
                            hole_check: param_hole_check,
                        },
                        _ if forced => {
                            let slot = next_ctx;
                            next_ctx += 1;
                            // declaration stores run with the hosting
                            // context as the frame context: zero hops
                            Resolution::Context {
                                slot,
                                depth: 0,
                                hole_check,
                            }
                        }
                        _ => {
                            let reg = next_reg;
                            next_reg += 1;
                            Resolution::Local { reg, hole_check }
                        }
                    };
                    slots.insert(key, res);
                }
            }
            layouts[fid.0 as usize] = FunctionLayout {
                register_count: next_reg,
                context_slots: next_ctx,
                this_slot,
                new_target_slot,
                this_function_slot,
            };
        }

        // lexical for-head scopes: like class inner scopes, every
        // declaration is a context slot in the loop's dedicated block
        // context, numbered in declaration order (the context is
        // re-created per iteration for captured heads)
        for s in 0..ast.scope_count() {
            let scope = ScopeId(s as u32);
            if ast.scope(scope).kind != ScopeKind::For
                || ast.scope(scope).decls.is_empty()
            {
                continue;
            }
            for (d, decl) in ast.scope(scope).decls.iter().enumerate() {
                let hole_check =
                    matches!(decl.kind, DeclKind::Let | DeclKind::Const);
                slots.insert(
                    (scope, d as u32),
                    Resolution::Context {
                        slot: d as u32,
                        depth: 0,
                        hole_check,
                    },
                );
            }
        }

        // class inner scopes: every declaration is a context slot in the
        // class's dedicated block context (created per class evaluation),
        // numbered in declaration order regardless of capture analysis
        for s in 0..ast.scope_count() {
            let scope = ScopeId(s as u32);
            if ast.scope(scope).kind != ScopeKind::Class {
                continue;
            }
            for (d, decl) in ast.scope(scope).decls.iter().enumerate() {
                let hole_check =
                    matches!(decl.kind, DeclKind::Let | DeclKind::Const | DeclKind::Class);
                slots.insert(
                    (scope, d as u32),
                    Resolution::Context {
                        slot: d as u32,
                        depth: 0,
                        hole_check,
                    },
                );
            }
        }

        let mut resolutions = vec![None; ast.node_count()];
        for (node, pending) in self.pending.iter().enumerate() {
            resolutions[node] = pending.as_ref().map(|p| match *p {
                Pending::GlobalObject => Resolution::GlobalObject,
                Pending::Dynamic => Resolution::Dynamic,
                Pending::This { owner, depth } => Resolution::This { owner, depth },
                Pending::NewTarget { owner, depth } => Resolution::NewTarget { owner, depth },
                Pending::SuperCall { owner, depth } => Resolution::SuperCall { owner, depth },
                Pending::Super {
                    scope,
                    decl,
                    depth,
                    this_owner: owner,
                    this_depth,
                } => {
                    let Resolution::Context { slot, .. } = slots[&(scope, decl)] else {
                        unreachable!("class-scope slots are context slots");
                    };
                    Resolution::Super {
                        home_slot: slot,
                        depth,
                        this_owner: owner,
                        this_depth,
                    }
                }
                // keep the per-use depth: the shared slots entry carries
                // the declaration-store depth (0), not this use site's
                Pending::Decl { scope, decl, depth } => match slots[&(scope, decl)] {
                    Resolution::Context {
                        slot, hole_check, ..
                    } => Resolution::Context {
                        slot,
                        depth,
                        hole_check,
                    },
                    other => other,
                },
            });
        }

        // per-iteration environments: for-heads whose let/const bindings are
        // captured need a fresh copy per iteration (ECMA-262 13.7.4.8)
        let mut per_iteration_loops = Vec::new();
        for n in 0..ast.node_count() {
            let id = NodeId(n as u32);
            if !matches!(ast.node(id), Node::For { .. } | Node::ForIn { .. }) {
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
            decl_maps: self.decl_maps,
            slots,
        }
    }
}
