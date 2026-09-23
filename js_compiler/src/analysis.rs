use std::collections::{HashMap, HashSet};

use oxc_ast::AstKind;
use oxc_ast::ast::*;
use oxc_ast_visit::{
    Visit,
    walk::{walk_for_in_statement, walk_for_of_statement, walk_for_statement},
};
use oxc_semantic::{AstNodes, Scoping, Semantic};
use oxc_span::{GetSpan, Span};
use oxc_syntax::node::NodeId;
use oxc_syntax::reference::ReferenceId;
use oxc_syntax::scope::{ScopeFlags, ScopeId};
use oxc_syntax::symbol::{SymbolFlags, SymbolId};

/// Index into [`Facts::functions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fid(pub u32);

/// Index into [`Facts::classes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassIdx(pub u32);

/// Resolutions that need parent-chain walks, precomputed per node:
/// `this`, `new.target`, `super(...)`, `super.x`, private names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special {
    This {
        owner: Fid,
        depth: u32,
    },
    NewTarget {
        owner: Fid,
        depth: u32,
    },
    SuperCall {
        owner: Fid,
        depth: u32,
    },
    /// `super.x`: home-object context slot + receiver owner
    Super {
        home: Home,
        depth: u32,
        this_owner: Fid,
        this_depth: u32,
    },
    /// private name: class-context slot
    Private {
        slot: u32,
        depth: u32,
    },
}

/// Where a `super.x` home object lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    /// class context slot
    Class(ClassIdx, u32),
    /// object literal with super-using methods: its 1-slot context
    Object(NodeId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FnKind {
    Script,
    Normal,
    Arrow,
    Method,
    Getter,
    Setter,
    BaseClassCtor,
    DerivedClassCtor,
    DefaultDerivedCtor,
}

impl FnKind {
    pub const fn is_arrow(self) -> bool {
        matches!(self, Self::Arrow)
    }

    pub const fn is_derived_class_constructor(self) -> bool {
        matches!(self, Self::DerivedClassCtor | Self::DefaultDerivedCtor)
    }
}

/// One formal parameter as the codegen needs it.
pub struct ParamData<'a> {
    pub pattern: &'a BindingPattern<'a>,
    pub default: Option<&'a Expression<'a>>,
    pub rest: bool,
}

pub enum FnBody<'a> {
    Script(&'a Program<'a>),
    Function(&'a FunctionBody<'a>),
    /// concise arrow: `=> expr`
    ArrowExpr(&'a Expression<'a>),
    /// synthesized field initializer: `return <init>;`
    FieldInit(&'a Expression<'a>),
    /// synthesized empty body
    Empty,
}

pub struct FuncInfo<'a> {
    pub span: Span,
    pub name: Option<String>,
    pub params: Vec<ParamData<'a>>,
    pub formal_length: u32,
    pub kind: FnKind,
    pub strict: bool,
    pub body: FnBody<'a>,
    /// synthesized field-initializer functions: the field's key node
    pub field_key: Option<&'a PropertyKey<'a>>,
    /// owning class of a constructor
    pub class: Option<ClassIdx>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Method,
    Get,
    Set,
    Field,
}

pub struct MemberInfo<'a> {
    pub key: &'a PropertyKey<'a>,
    /// the method function (`Method`/`Get`/`Set`)
    pub value: Option<&'a Function<'a>>,
    /// field initializer expression (`Field`)
    pub init: Option<&'a Expression<'a>>,
    pub kind: MemberKind,
    pub is_static: bool,
    pub computed: bool,
    pub is_private: bool,
    /// the member's function id (method or synthesized field initializer)
    pub fid: Fid,
}

pub struct ClassInfo<'a> {
    pub span: Span,
    pub node: NodeId,
    pub name: Option<String>,
    /// class-expression inner-binding slot (declarations bind the name in
    /// the outer scope; oxc resolves inner references there)
    pub name_slot: Option<u32>,
    pub superclass: Option<&'a Expression<'a>>,
    pub members: Vec<MemberInfo<'a>>,
    pub ctor: Fid,
    pub is_decl: bool,
    pub uses_super: bool,
    pub home_slot: Option<u32>,
    pub static_home_slot: Option<u32>,
    /// private names in declaration order (slots parallel in
    /// `private_slots`, assigned once `uses_super` is known)
    pub privates: Vec<String>,
    pub private_slots: Vec<u32>,
    pub slot_count: u32,
    pub has_instance_fields: bool,
    /// class declarations: the outer binding to store the ctor into
    pub decl_symbol: Option<(SymbolId, String)>,
}

/// Everything the codegen needs besides the AST and oxc's own tables.
pub struct Facts<'a> {
    pub mode: Mode,
    pub root: ScopeId,
    pub functions: Vec<FuncInfo<'a>>,
    pub classes: Vec<ClassInfo<'a>>,
    /// function node id (Function / ArrowFunctionExpression /
    /// PropertyDefinition field-init) → fid
    pub fn_of_node: HashMap<NodeId, Fid>,
    /// class node id → index
    pub class_of_node: HashMap<NodeId, ClassIdx>,
    /// oxc scope → owning function (class / for contexts excluded: they
    /// have their own slot spaces)
    pub scope_owner: HashMap<ScopeId, Fid>,
    /// oxc function scope → fid
    pub fn_scope_to_fid: HashMap<ScopeId, Fid>,
    /// oxc class scope → class index
    pub class_of_scope: HashMap<ScopeId, ClassIdx>,
    /// precomputed this / new.target / super / private resolutions
    pub special: HashMap<NodeId, Special>,
    /// per resolved reference: context hops from the use site to the
    /// hosting context (node-ancestry based, so synthesized field-init
    /// frames count)
    pub ref_depth: HashMap<ReferenceId, u32>,
    /// symbols referenced from a function other than their owner
    pub captured: HashSet<SymbolId>,
    /// functions whose receiver / new.target a nested arrow captures
    pub captures_this: HashSet<Fid>,
    pub captures_new_target: HashSet<Fid>,
    /// derived constructors with an arrow-delegated super()
    pub needs_this_function: HashSet<Fid>,
    /// for/for-in loop nodes → number of lexical head bindings
    pub for_slots: HashMap<NodeId, u32>,
    /// lexical-head for scope → loop node
    pub for_of_scope: HashMap<ScopeId, NodeId>,
    /// loop nodes needing a fresh context per iteration
    pub per_iteration: HashSet<NodeId>,
    /// object literals owning a home-object context
    pub obj_lit_home: HashSet<NodeId>,
    /// per function: `var` symbols to pre-initialize with `undefined`
    pub hoist_vars: HashMap<Fid, Vec<SymbolId>>,
    /// per function: top-level function declarations to hoist
    pub hoist_fns: HashMap<Fid, Vec<(SymbolId, Fid)>>,
    /// param symbol → positional index
    pub param_symbols: HashMap<SymbolId, u32>,
    /// names bound by parameter patterns (TDZ locals, not param registers)
    pub pattern_params: HashSet<SymbolId>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Script,
    Eval,
    Repl,
}

pub fn analyze<'a>(program: &'a Program<'a>, semantic: &Semantic<'a>, mode: Mode) -> Facts<'a> {
    Collector::new(program, semantic.scoping(), mode).run(program, semantic.nodes())
}

// ---------------------------------------------------------------------------
// pass 1: structure collection via Visit
// ---------------------------------------------------------------------------

struct Collector<'a, 'p> {
    scoping: &'p Scoping,
    mode: Mode,
    functions: Vec<FuncInfo<'a>>,
    fn_of_node: HashMap<NodeId, Fid>,
    fn_scope_to_fid: HashMap<ScopeId, Fid>,
    fn_stack: Vec<Fid>,
    /// fid → its AST node (program / function / property definition)
    fid_node: HashMap<Fid, NodeId>,
    classes: Vec<ClassInfo<'a>>,
    class_of_node: HashMap<NodeId, ClassIdx>,
    class_of_scope: HashMap<ScopeId, ClassIdx>,
    for_slots: HashMap<NodeId, u32>,
    for_of_scope: HashMap<ScopeId, NodeId>,
    hoist_fns: HashMap<Fid, Vec<(SymbolId, Fid)>>,
    param_symbols: HashMap<SymbolId, u32>,
    pattern_params: HashSet<SymbolId>,
}

impl<'a, 'p> Collector<'a, 'p> {
    fn new(program: &'a Program<'a>, scoping: &'p Scoping, mode: Mode) -> Self {
        Self {
            scoping,
            mode,
            functions: vec![FuncInfo {
                span: program.span,
                name: None,
                params: Vec::new(),
                formal_length: 0,
                kind: FnKind::Script,
                strict: scoping.root_scope_flags().contains(ScopeFlags::StrictMode),
                body: FnBody::Script(program),
                field_key: None,
                class: None,
            }],
            fn_of_node: HashMap::from([(program.node_id.get(), Fid(0))]),
            fn_scope_to_fid: HashMap::from([(scoping.root_scope_id(), Fid(0))]),
            fn_stack: vec![Fid(0)],
            fid_node: HashMap::from([(Fid(0), program.node_id.get())]),
            classes: Vec::new(),
            class_of_node: HashMap::new(),
            class_of_scope: HashMap::new(),
            for_slots: HashMap::new(),
            for_of_scope: HashMap::new(),
            hoist_fns: HashMap::new(),
            param_symbols: HashMap::new(),
            pattern_params: HashSet::new(),
        }
    }

    fn run(mut self, program: &'a Program<'a>, nodes: &AstNodes<'a>) -> Facts<'a> {
        self.visit_program(program);
        let mut d = Deriver {
            scoping: self.scoping,
            nodes,
            functions: &mut self.functions,
            fn_of_node: &self.fn_of_node,
            classes: &mut self.classes,
            class_of_node: &self.class_of_node,
            class_of_scope: &self.class_of_scope,
            for_slots: &self.for_slots,
            for_of_scope: &self.for_of_scope,
            fn_scope_to_fid: &self.fn_scope_to_fid,
            captured: HashSet::new(),
            captures_this: HashSet::new(),
            captures_new_target: HashSet::new(),
            needs_this_function: HashSet::new(),
            obj_lit_home: HashSet::new(),
            special: HashMap::new(),
            ref_depth: HashMap::new(),
            fid_node: &self.fid_node,
        };
        d.mark_super_uses();
        d.finalize_class_slots();
        d.patch_object_methods();
        d.scan_special_nodes();
        d.scan_captures();
        d.compute_reference_depths();
        let Deriver {
            special,
            ref_depth,
            captured,
            captures_this,
            captures_new_target,
            needs_this_function,
            obj_lit_home,
            ..
        } = d;

        // scope ownership: which function's register/context space a
        // scope's bindings allocate from
        let mut scope_owner: HashMap<ScopeId, Fid> = HashMap::new();
        for sid in 0..self.scoping.scopes_len() {
            let sid = ScopeId::from_usize(sid);
            let mut cur = sid;
            let owner = loop {
                if let Some(&fid) = self.fn_scope_to_fid.get(&cur) {
                    break fid;
                }
                if self.class_of_scope.contains_key(&cur) || self.for_of_scope.contains_key(&cur) {
                    // class / for contexts have their own slot spaces
                    break Fid(u32::MAX);
                }
                cur = self
                    .scoping
                    .scope_parent_id(cur)
                    .expect("scope chain ends at the root");
            };
            scope_owner.insert(sid, owner);
        }

        // per-iteration loops: for heads whose lexical bindings are captured
        let mut per_iteration = HashSet::new();
        for (sid, &node) in &self.for_of_scope {
            if self
                .scoping
                .iter_bindings_in(*sid)
                .any(|sym| captured.contains(&sym))
            {
                per_iteration.insert(node);
            }
        }

        // `var` hoisting lists per function (deterministic: scope-id order)
        let mut hoist_vars: HashMap<Fid, Vec<SymbolId>> = HashMap::new();
        for s in 0..self.scoping.scopes_len() {
            let sid = ScopeId::from_usize(s);
            let Some(&owner) = scope_owner.get(&sid) else {
                continue;
            };
            if owner == Fid(u32::MAX) {
                continue;
            }
            for sym in self.scoping.iter_bindings_in(sid) {
                let flags = self.scoping.symbol_flags(sym);
                let is_var = !self.param_symbols.contains_key(&sym)
                    && !self.pattern_params.contains(&sym)
                    && flags.contains(SymbolFlags::FunctionScopedVariable)
                    && !flags.contains(SymbolFlags::Function);
                if is_var {
                    hoist_vars.entry(owner).or_default().push(sym);
                }
            }
        }

        Facts {
            mode: self.mode,
            root: self.scoping.root_scope_id(),
            functions: self.functions,
            classes: self.classes,
            fn_of_node: self.fn_of_node,
            class_of_node: self.class_of_node,
            scope_owner,
            fn_scope_to_fid: self.fn_scope_to_fid,
            class_of_scope: self.class_of_scope,
            special,
            ref_depth,
            captured,
            captures_this,
            captures_new_target,
            needs_this_function,
            for_slots: self.for_slots,
            for_of_scope: self.for_of_scope,
            per_iteration,
            obj_lit_home,
            hoist_vars,
            hoist_fns: self.hoist_fns,
            param_symbols: self.param_symbols,
            pattern_params: self.pattern_params,
        }
    }

    fn push_function(
        &mut self,
        node: NodeId,
        scope: Option<ScopeId>,
        span: Span,
        name: Option<String>,
        params: Vec<ParamData<'a>>,
        formal_length: u32,
        kind: FnKind,
        strict: bool,
        body: FnBody<'a>,
        field_key: Option<&'a PropertyKey<'a>>,
        class: Option<ClassIdx>,
    ) -> Fid {
        let fid = Fid(self.functions.len() as u32);
        self.functions.push(FuncInfo {
            span,
            name,
            params,
            formal_length,
            kind,
            strict,
            body,
            field_key,
            class,
        });
        self.fn_of_node.insert(node, fid);
        self.fid_node.insert(fid, node);
        if let Some(scope) = scope {
            self.fn_scope_to_fid.insert(scope, fid);
        }
        fid
    }

    fn record_param_symbols(&mut self, ps: &FormalParameters<'a>) {
        let mut index = 0u32;
        for p in &ps.items {
            self.record_pattern_symbols(&p.pattern, index);
            index += 1;
        }
        if let Some(rest) = &ps.rest {
            self.record_pattern_symbols(&rest.rest.argument, index);
        }
    }

    fn record_pattern_symbols(&mut self, p: &BindingPattern<'a>, index: u32) {
        match p {
            BindingPattern::BindingIdentifier(b) => {
                if let Some(sym) = b.symbol_id.get() {
                    self.param_symbols.insert(sym, index);
                }
            }
            BindingPattern::ObjectPattern(o) => {
                for prop in &o.properties {
                    self.record_pattern_symbols(&prop.value, index);
                }
                if let Some(rest) = &o.rest {
                    self.record_pattern_symbols(&rest.argument, index);
                }
            }
            BindingPattern::ArrayPattern(a) => {
                for el in a.elements.iter().flatten() {
                    self.record_pattern_symbols(el, index);
                }
                if let Some(rest) = &a.rest {
                    self.record_pattern_symbols(&rest.argument, index);
                }
            }
            BindingPattern::AssignmentPattern(a) => {
                self.record_pattern_symbols(&a.left, index);
            }
        }
        // names bound by non-identifier parameter patterns are TDZ locals
        // stored by the destructuring prologue, not param registers
        if !matches!(p, BindingPattern::BindingIdentifier(_)) {
            collect_binding_symbols(p).into_iter().for_each(|sym| {
                self.pattern_params.insert(sym);
            });
        }
    }

    /// Direct-body function declarations become hoisted closures. The
    /// walk has already created their fids.
    fn collect_hoisted_fns(&mut self, stmts: &'a [Statement<'a>]) {
        let owner = *self.fn_stack.last().expect("fn stack");
        for stmt in stmts {
            if let Statement::FunctionDeclaration(f) = stmt
                && let Some(id) = &f.id
                && let Some(sym) = id.symbol_id.get()
                && let Some(&fid) = self.fn_of_node.get(&f.node_id.get())
            {
                self.hoist_fns.entry(owner).or_default().push((sym, fid));
            }
        }
    }

    fn record_for(&mut self, scope: Option<ScopeId>, node: NodeId, lexical_head: bool) {
        let Some(scope) = scope else { return };
        if !lexical_head {
            return;
        }
        let count = self.scoping.iter_bindings_in(scope).len() as u32;
        if count > 0 {
            self.for_slots.insert(node, count);
            self.for_of_scope.insert(scope, node);
        }
    }

    fn strict_scope(&self, scope: Option<ScopeId>) -> bool {
        scope
            .map(|s| self.scoping.scope_flags(s).contains(ScopeFlags::StrictMode))
            .unwrap_or(true)
    }
}

/// Extend a visit-time borrow to the arena lifetime. Sound inside
/// [`analyze`]: the arena outlives `'a` there and the visitor only ever
/// receives borrows derived from it (the same trick oxc's own `AstKind`
/// pointers use).
fn extend<'a, T: ?Sized>(r: &T) -> &'a T {
    unsafe { &*(r as *const T) }
}

impl<'a, 'p> Visit<'a> for Collector<'a, 'p> {
    fn visit_program(&mut self, program: &Program<'a>) {
        for stmt in &program.body {
            self.visit_statement(stmt);
        }
        // hoistable script-level declarations: collected after the walk,
        // once their fids exist
        let stmts: &'a [Statement<'a>] = extend(&program.body);
        self.collect_hoisted_fns(stmts);
    }

    fn visit_function(&mut self, it: &Function<'a>, flags: ScopeFlags) {
        let it: &'a Function<'a> = extend(it);
        // class members and ctors are pre-created by the class scan
        if !self.fn_of_node.contains_key(&it.node_id.get()) {
            let (params, formal_length) = collect_params(&it.params);
            // semantic scope flags carry the propagated strictness; the
            // visitor's `flags` argument is per-node-kind only
            let strict = self.strict_scope(it.scope_id.get());
            let _ = flags;
            self.push_function(
                it.node_id.get(),
                it.scope_id.get(),
                it.span,
                it.id.as_ref().map(|i| i.name.to_string()),
                params,
                formal_length,
                FnKind::Normal,
                strict,
                FnBody::Function(it.body.as_deref().expect("function body")),
                None,
                None,
            );
        }
        self.record_param_symbols(&it.params);
        let fid = self.fn_of_node[&it.node_id.get()];
        self.fn_stack.push(fid);
        self.visit_formal_parameters(&it.params);
        if let Some(body) = &it.body {
            for stmt in &body.statements {
                self.visit_statement(stmt);
            }
            self.collect_hoisted_fns(&body.statements);
        }
        self.fn_stack.pop();
    }

    fn visit_arrow_function_expression(&mut self, it: &ArrowFunctionExpression<'a>) {
        let it: &'a ArrowFunctionExpression<'a> = extend(it);
        let (params, formal_length) = collect_params(&it.params);
        let strict = self.strict_scope(it.scope_id.get());
        let body = match &it.body {
            ArrowFunctionBody::FunctionBody(b) => FnBody::Function(b),
            other => FnBody::ArrowExpr(other.as_expression().expect("expression body")),
        };
        let fid = self.push_function(
            it.node_id.get(),
            it.scope_id.get(),
            it.span,
            None,
            params,
            formal_length,
            FnKind::Arrow,
            strict,
            body,
            None,
            None,
        );
        self.record_param_symbols(&it.params);
        self.fn_stack.push(fid);
        self.visit_formal_parameters(&it.params);
        match &it.body {
            ArrowFunctionBody::FunctionBody(b) => {
                for stmt in &b.statements {
                    self.visit_statement(stmt);
                }
                self.collect_hoisted_fns(&b.statements);
            }
            other => self.visit_expression(other.as_expression().expect("expression body")),
        }
        self.fn_stack.pop();
    }

    fn visit_class(&mut self, it: &Class<'a>) {
        let it: &'a Class<'a> = extend(it);
        let idx = ClassIdx(self.classes.len() as u32);
        let is_decl = it.is_declaration();
        let derived = it.heritage.is_some();
        self.class_of_node.insert(it.node_id.get(), idx);
        if let Some(scope) = it.scope_id.get() {
            self.class_of_scope.insert(scope, idx);
        }

        let mut privates: Vec<String> = Vec::new();
        let mut members: Vec<MemberInfo<'a>> = Vec::new();
        let mut has_instance_fields = false;
        let mut explicit_ctor: Option<&'a MethodDefinition<'a>> = None;
        for el in &it.body.body {
            match el {
                ClassElement::MethodDefinition(m) => {
                    if m.kind == MethodDefinitionKind::Constructor && explicit_ctor.is_none() {
                        explicit_ctor = Some(m);
                        continue;
                    }
                    let is_private = matches!(m.key, PropertyKey::PrivateIdentifier(_));
                    if is_private
                        && let PropertyKey::PrivateIdentifier(p) = &m.key
                        && !privates.contains(&p.name.to_string())
                    {
                        privates.push(p.name.to_string());
                    }
                    let kind = match m.kind {
                        MethodDefinitionKind::Get => MemberKind::Get,
                        MethodDefinitionKind::Set => MemberKind::Set,
                        _ => MemberKind::Method,
                    };
                    let fn_kind = match m.kind {
                        MethodDefinitionKind::Get => FnKind::Getter,
                        MethodDefinitionKind::Set => FnKind::Setter,
                        _ => FnKind::Method,
                    };
                    let strict = self.strict_scope(m.value.scope_id.get());
                    let (params, formal_length) = collect_params(&m.value.params);
                    let fid = self.push_function(
                        m.value.node_id.get(),
                        m.value.scope_id.get(),
                        m.value.span,
                        member_display_name(&m.key, m.kind),
                        params,
                        formal_length,
                        fn_kind,
                        strict,
                        FnBody::Function(m.value.body.as_deref().expect("method body")),
                        None,
                        None,
                    );
                    members.push(MemberInfo {
                        key: &m.key,
                        value: Some(&m.value),
                        init: None,
                        kind,
                        is_static: m.r#static,
                        computed: m.computed,
                        is_private,
                        fid,
                    });
                }
                ClassElement::PropertyDefinition(p) => {
                    let is_private = matches!(p.key, PropertyKey::PrivateIdentifier(_));
                    if !p.r#static {
                        has_instance_fields = true;
                    }
                    if is_private
                        && let PropertyKey::PrivateIdentifier(name) = &p.key
                        && !privates.contains(&name.name.to_string())
                    {
                        privates.push(name.name.to_string());
                    }
                    let fid = self.push_function(
                        p.node_id.get(),
                        None,
                        p.span,
                        None,
                        Vec::new(),
                        0,
                        FnKind::Normal,
                        true,
                        match p.value.as_ref() {
                            Some(v) => FnBody::FieldInit(v),
                            None => FnBody::Empty,
                        },
                        Some(&p.key),
                        None,
                    );
                    members.push(MemberInfo {
                        key: &p.key,
                        value: None,
                        init: p.value.as_ref(),
                        kind: MemberKind::Field,
                        is_static: p.r#static,
                        computed: p.computed,
                        is_private,
                        fid,
                    });
                }
                _ => {}
            }
        }

        // constructor: explicit or synthesized
        let ctor = match explicit_ctor {
            Some(m) => {
                let kind = if derived {
                    FnKind::DerivedClassCtor
                } else {
                    FnKind::BaseClassCtor
                };
                let strict = self.strict_scope(m.value.scope_id.get());
                let (params, formal_length) = collect_params(&m.value.params);
                self.push_function(
                    m.value.node_id.get(),
                    m.value.scope_id.get(),
                    m.value.span,
                    it.id.as_ref().map(|i| i.name.to_string()),
                    params,
                    formal_length,
                    kind,
                    strict,
                    FnBody::Function(m.value.body.as_deref().expect("ctor body")),
                    None,
                    Some(idx),
                )
            }
            None => {
                let kind = if derived {
                    FnKind::DefaultDerivedCtor
                } else {
                    FnKind::BaseClassCtor
                };
                let fid = Fid(self.functions.len() as u32);
                self.functions.push(FuncInfo {
                    span: it.span,
                    name: it.id.as_ref().map(|i| i.name.to_string()),
                    params: Vec::new(),
                    formal_length: 0,
                    kind,
                    strict: true,
                    body: FnBody::Empty,
                    field_key: None,
                    class: Some(idx),
                });
                fid
            }
        };

        self.classes.push(ClassInfo {
            span: it.span,
            node: it.node_id.get(),
            name: it.id.as_ref().map(|i| i.name.to_string()),
            name_slot: None,
            superclass: it.heritage.as_ref().map(|h| &h.expression),
            members,
            ctor,
            is_decl,
            uses_super: false,
            home_slot: None,
            static_home_slot: None,
            privates,
            private_slots: Vec::new(),
            slot_count: 0,
            has_instance_fields,
            decl_symbol: it
                .id
                .as_ref()
                .and_then(|id| id.symbol_id.get().map(|sym| (sym, id.name.to_string()))),
        });

        // walk: heritage evaluates inside the class context
        if let Some(heritage) = &it.heritage {
            self.visit_expression(&heritage.expression);
        }
        for el in &it.body.body {
            match el {
                ClassElement::MethodDefinition(m) => {
                    if m.computed {
                        self.visit_property_key(&m.key);
                    }
                    self.visit_function(&m.value, ScopeFlags::empty());
                }
                ClassElement::PropertyDefinition(p) => {
                    if p.computed {
                        self.visit_property_key(&p.key);
                    }
                    // the initializer walks as the field-init function body
                    if let Some(v) = &p.value {
                        let fid = self.fn_of_node[&p.node_id.get()];
                        self.fn_stack.push(fid);
                        self.visit_expression(v);
                        self.fn_stack.pop();
                    }
                }
                ClassElement::StaticBlock(b) => {
                    for stmt in &b.body {
                        self.visit_statement(stmt);
                    }
                }
                _ => {}
            }
        }
    }

    fn visit_for_statement(&mut self, it: &ForStatement<'a>) {
        let it: &'a ForStatement<'a> = extend(it);
        let lexical_head = matches!(
            &it.init,
            Some(ForStatementInit::VariableDeclaration(d))
                if d.kind != VariableDeclarationKind::Var
        );
        self.record_for(it.scope_id.get(), it.node_id.get(), lexical_head);
        walk_for_statement(self, it);
    }

    fn visit_for_in_statement(&mut self, it: &ForInStatement<'a>) {
        let it: &'a ForInStatement<'a> = extend(it);
        let lexical_head = matches!(
            &it.left,
            ForStatementLeft::VariableDeclaration(d) if d.kind != VariableDeclarationKind::Var
        );
        self.record_for(it.scope_id.get(), it.node_id.get(), lexical_head);
        walk_for_in_statement(self, it);
    }

    fn visit_for_of_statement(&mut self, it: &ForOfStatement<'a>) {
        let it: &'a ForOfStatement<'a> = extend(it);
        let lexical_head = matches!(
            &it.left,
            ForStatementLeft::VariableDeclaration(d) if d.kind != VariableDeclarationKind::Var
        );
        self.record_for(it.scope_id.get(), it.node_id.get(), lexical_head);
        walk_for_of_statement(self, it);
    }
}

// ---------------------------------------------------------------------------
// pass 2: table-driven facts from node/reference tables + parent chains
// ---------------------------------------------------------------------------

struct Deriver<'a, 'p, 'f> {
    scoping: &'p Scoping,
    nodes: &'f AstNodes<'a>,
    functions: &'f mut Vec<FuncInfo<'a>>,
    fn_of_node: &'f HashMap<NodeId, Fid>,
    classes: &'f mut Vec<ClassInfo<'a>>,
    class_of_node: &'f HashMap<NodeId, ClassIdx>,
    class_of_scope: &'f HashMap<ScopeId, ClassIdx>,
    for_slots: &'f HashMap<NodeId, u32>,
    for_of_scope: &'f HashMap<ScopeId, NodeId>,
    fn_scope_to_fid: &'f HashMap<ScopeId, Fid>,
    captured: HashSet<SymbolId>,
    captures_this: HashSet<Fid>,
    captures_new_target: HashSet<Fid>,
    needs_this_function: HashSet<Fid>,
    obj_lit_home: HashSet<NodeId>,
    special: HashMap<NodeId, Special>,
    ref_depth: HashMap<ReferenceId, u32>,
    /// fid → its AST node (program, function, property definition)
    fid_node: &'f HashMap<Fid, NodeId>,
}

impl<'a, 'p, 'f> Deriver<'a, 'p, 'f> {
    /// Mark `super.x`/`super[k]` uses on their owning class / object
    /// literal (decides context ownership).
    fn mark_super_uses(&mut self) {
        for node in self.nodes.iter() {
            let object = match node.kind() {
                AstKind::StaticMemberExpression(e) => &e.object,
                AstKind::ComputedMemberExpression(e) => &e.object,
                _ => continue,
            };
            if !matches!(object, Expression::Super(_)) {
                continue;
            }
            // attribute to the innermost enclosing class or object literal
            for anc in self.nodes.ancestors(node.id()) {
                match anc.kind() {
                    AstKind::ObjectExpression(o) => {
                        self.obj_lit_home.insert(o.node_id.get());
                        break;
                    }
                    AstKind::Class(c) => {
                        let idx = self.class_of_node[&c.node_id.get()];
                        self.classes[idx.0 as usize].uses_super = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Assign class context slot layout once `uses_super` is known:
    /// [name?, home?, static_home?, privates...].
    fn finalize_class_slots(&mut self) {
        for i in 0..self.classes.len() {
            let mut slot = 0u32;
            if !self.classes[i].is_decl && self.classes[i].name.is_some() {
                self.classes[i].name_slot = Some(slot);
                slot += 1;
            }
            if self.classes[i].uses_super {
                self.classes[i].home_slot = Some(slot);
                self.classes[i].static_home_slot = Some(slot + 1);
                slot += 2;
            }
            let n_privates = self.classes[i].privates.len() as u32;
            self.classes[i].private_slots = (0..n_privates).map(|n| slot + n).collect();
            self.classes[i].slot_count = slot + n_privates;
        }
    }

    /// Object-literal methods and accessors: patch IR kind and the
    /// display name (the Collector sees them as plain functions).
    fn patch_object_methods(&mut self) {
        for node in self.nodes.iter() {
            let AstKind::ObjectProperty(p) = node.kind() else {
                continue;
            };
            let Expression::FunctionExpression(f) = &p.value else {
                continue;
            };
            let Some(&fid) = self.fn_of_node.get(&f.node_id.get()) else {
                continue;
            };
            let name = |prefix: Option<&str>, key: &PropertyKey| match key {
                PropertyKey::StaticIdentifier(i) => match prefix {
                    Some(p) => Some(format!("{p}{}", i.name)),
                    None => Some(i.name.to_string()),
                },
                PropertyKey::StringLiteral(s) => match prefix {
                    Some(p) => Some(format!("{p}{}", s.value)),
                    None => Some(s.value.to_string()),
                },
                PropertyKey::NumericLiteral(n) => match prefix {
                    Some(p) => Some(format!("{p}{}", n.value)),
                    None => Some(n.value.to_string()),
                },
                _ => None,
            };
            let info = &mut self.functions[fid.0 as usize];
            match p.kind {
                PropertyKind::Init => {
                    if p.method {
                        info.kind = FnKind::Method;
                        info.name = name(None, &p.key);
                    }
                }
                PropertyKind::Get => {
                    info.kind = FnKind::Getter;
                    info.name = name(Some("get "), &p.key);
                }
                PropertyKind::Set => {
                    info.kind = FnKind::Setter;
                    info.name = name(Some("set "), &p.key);
                }
            }
        }
    }

    /// Resolve this / new.target / super(...) / super.x / private names by
    /// walking parent chains.
    fn scan_special_nodes(&mut self) {
        for node in self.nodes.iter() {
            let id = node.id();
            match node.kind() {
                AstKind::ThisExpression(t) => {
                    let Some((owner, depth)) = self.fn_owner_and_depth(id) else {
                        continue;
                    };
                    self.captures_this.insert(owner);
                    self.special
                        .insert(t.node_id.get(), Special::This { owner, depth });
                }
                AstKind::NewTarget(m) => {
                    let Some((owner, depth)) = self.fn_owner_and_depth(id) else {
                        continue;
                    };
                    if owner != self.current_fn_of(id) {
                        self.captures_new_target.insert(owner);
                    }
                    self.special
                        .insert(m.node_id.get(), Special::NewTarget { owner, depth });
                }
                AstKind::CallExpression(c) if matches!(c.callee, Expression::Super(_)) => {
                    let Some((owner, depth)) = self.fn_owner_and_depth(id) else {
                        continue;
                    };
                    if owner != self.current_fn_of(id) {
                        self.needs_this_function.insert(owner);
                        self.captures_new_target.insert(owner);
                        self.captures_this.insert(owner);
                    }
                    self.special.insert(id, Special::SuperCall { owner, depth });
                }
                AstKind::StaticMemberExpression(e) if matches!(e.object, Expression::Super(_)) => {
                    let is_static = self.enclosing_member_is_static(id);
                    self.resolve_super(id, is_static);
                }
                AstKind::ComputedMemberExpression(e)
                    if matches!(e.object, Expression::Super(_)) =>
                {
                    let is_static = self.enclosing_member_is_static(id);
                    self.resolve_super(id, is_static);
                }
                AstKind::PrivateFieldExpression(e) => {
                    let name = e.field.name.to_string();
                    self.resolve_private(&name, e.field.node_id.get(), id);
                }
                AstKind::PrivateInExpression(e) => {
                    let name = e.left.name.to_string();
                    self.resolve_private(&name, e.left.node_id.get(), id);
                }
                // private field/method *declaration* keys: distinct nodes
                // from their uses, resolved against the owning class
                AstKind::PropertyDefinition(p) => {
                    if let PropertyKey::PrivateIdentifier(ident) = &p.key {
                        let name = ident.name.to_string();
                        self.resolve_private(&name, ident.node_id.get(), id);
                    }
                }
                AstKind::MethodDefinition(m) => {
                    if let PropertyKey::PrivateIdentifier(ident) = &m.key {
                        let name = ident.name.to_string();
                        self.resolve_private(&name, ident.node_id.get(), id);
                    }
                }
                _ => {}
            }
        }
    }

    /// The nearest enclosing non-arrow function of `node` and the context
    /// hops to its frame context.
    fn fn_owner_and_depth(&self, node: NodeId) -> Option<(Fid, u32)> {
        let owner = self.nearest_function(node, false)?;
        let fid = *self.fn_of_node.get(&owner)?;
        Some((fid, self.ctx_hops_until(node, owner)))
    }

    /// The function directly containing `node` (any kind).
    fn current_fn_of(&self, node: NodeId) -> Fid {
        let fn_node = self
            .nearest_function(node, true)
            .expect("code always sits inside a function");
        self.fn_of_node[&fn_node]
    }

    /// Walk parents to the nearest enclosing function node. With
    /// `any = false`, arrow functions are transparent (their receiver /
    /// new.target / super belong to the enclosing non-arrow function).
    fn nearest_function(&self, node: NodeId, any: bool) -> Option<NodeId> {
        for anc in self.nodes.ancestors(node) {
            if !any && matches!(anc.kind(), AstKind::ArrowFunctionExpression(_)) {
                continue;
            }
            if self.fn_of_node.contains_key(&anc.id()) {
                return Some(anc.id());
            }
        }
        None
    }

    /// Count context-creating scopes strictly between `node` and the
    /// ancestor `target` (the use site's own context is depth 0).
    fn ctx_hops_until(&self, node: NodeId, target: NodeId) -> u32 {
        // depth = index of the target context in the use site's context
        // chain (innermost = 0)
        let mut passed_use = false;
        let mut hops = 0u32;
        for anc in self.nodes.ancestors(node) {
            if !self.node_creates_ctx(anc.id()) {
                continue;
            }
            if !passed_use {
                passed_use = true;
            } else {
                hops += 1;
            }
            if anc.id() == target {
                return hops;
            }
        }
        unreachable!("target is an ancestor of the use site")
    }

    /// Whether an AST node creates a runtime context in our model:
    /// functions (incl. arrows and field initializers), context-owning
    /// classes, home-owning object literals, lexical `for` heads.
    fn node_creates_ctx(&self, node: NodeId) -> bool {
        if self.fn_of_node.contains_key(&node) {
            return true;
        }
        match self.nodes.get_node(node).kind() {
            AstKind::Class(c) => {
                let idx = self.class_of_node[&c.node_id.get()];
                self.classes[idx.0 as usize].slot_count > 0
            }
            AstKind::ObjectExpression(o) => self.obj_lit_home.contains(&o.node_id.get()),
            AstKind::ForStatement(f) => self.for_slots.contains_key(&f.node_id.get()),
            AstKind::ForInStatement(f) => self.for_slots.contains_key(&f.node_id.get()),
            AstKind::ForOfStatement(f) => self.for_slots.contains_key(&f.node_id.get()),
            _ => false,
        }
    }

    /// A home owner for `super.x`: a class with `uses_super` or a
    /// home-owning object literal.
    fn is_home_owner(&self, node: NodeId) -> bool {
        match self.nodes.get_node(node).kind() {
            AstKind::Class(c) => {
                let idx = self.class_of_node[&c.node_id.get()];
                self.classes[idx.0 as usize].uses_super
            }
            AstKind::ObjectExpression(o) => self.obj_lit_home.contains(&o.node_id.get()),
            _ => false,
        }
    }

    /// Whether the nearest enclosing class member is `static` (object
    /// literal methods are never static).
    fn enclosing_member_is_static(&self, node: NodeId) -> bool {
        for anc in self.nodes.ancestors(node) {
            match anc.kind() {
                AstKind::MethodDefinition(m) => return m.r#static,
                AstKind::ObjectExpression(_) => return false,
                _ => {}
            }
        }
        false
    }

    fn resolve_super(&mut self, node: NodeId, is_static: bool) {
        let Some((this_owner, this_depth)) = self.fn_owner_and_depth(node) else {
            return;
        };
        self.captures_this.insert(this_owner);
        let mut home = None;
        let mut depth = 0u32;
        let mut passed_use = false;
        let mut hops = 0u32;
        for anc in self.nodes.ancestors(node) {
            if !self.node_creates_ctx(anc.id()) {
                continue;
            }
            if !passed_use {
                passed_use = true;
            } else {
                hops += 1;
            }
            if self.is_home_owner(anc.id()) {
                home = Some(match anc.kind() {
                    AstKind::Class(c) => {
                        let idx = self.class_of_node[&c.node_id.get()];
                        let slot = if is_static {
                            self.classes[idx.0 as usize].static_home_slot
                        } else {
                            self.classes[idx.0 as usize].home_slot
                        }
                        .expect("uses_super classes declare home slots");
                        Home::Class(idx, slot)
                    }
                    AstKind::ObjectExpression(o) => Home::Object(o.node_id.get()),
                    _ => unreachable!(),
                });
                depth = hops;
                break;
            }
        }
        if let Some(home) = home {
            self.special.insert(
                node,
                Special::Super {
                    home,
                    depth,
                    this_owner,
                    this_depth,
                },
            );
        }
    }

    fn resolve_private(&mut self, name: &str, ident_node: NodeId, use_node: NodeId) {
        for anc in self.nodes.ancestors(use_node) {
            if let AstKind::Class(c) = anc.kind() {
                let idx = self.class_of_node[&c.node_id.get()];
                let info = &self.classes[idx.0 as usize];
                if let Some(i) = info.privates.iter().position(|n| n == name) {
                    let slot = info.private_slots[i];
                    let depth = self.ctx_hops_until(use_node, anc.id());
                    self.special
                        .insert(ident_node, Special::Private { slot, depth });
                    return;
                }
            }
        }
        // an unresolved private name is a syntax error oxc reports
    }

    /// Captured-ness: a symbol referenced from a function other than its
    /// owning one is context-allocated. Attribution uses node ancestry —
    /// synthesized field-initializer frames count as functions, even
    /// though oxc scopes their references to the class.
    fn scan_captures(&mut self) {
        for rid in 0..self.scoping.references_len() {
            let rid = ReferenceId::from_usize(rid);
            let reference = self.scoping.get_reference(rid);
            let Some(symbol) = reference.symbol_id() else {
                continue;
            };
            let decl_owner = self.scope_owner(self.scoping.symbol_scope_id(symbol));
            let use_owner = self.use_function_of(reference.node_id());
            if decl_owner != use_owner {
                self.captured.insert(symbol);
            }
        }
    }

    /// A field's computed KEY evaluates in the enclosing function; only
    /// its initializer runs in the synthesized frame. The property
    /// definition is transparent for key references.
    fn transparent_field_key(&self, anc_kind: AstKind<'_>, ref_span: Span) -> bool {
        match anc_kind {
            AstKind::PropertyDefinition(p) => {
                let key_span = GetSpan::span(&p.key);
                p.computed && ref_span.start >= key_span.start && ref_span.end <= key_span.end
            }
            _ => false,
        }
    }

    /// The runtime function a reference executes in, node-ancestry based
    /// (synthesized field-initializer frames count; computed keys are
    /// transparent to them).
    fn use_function_of(&self, node: NodeId) -> Fid {
        let ref_span = GetSpan::span(self.nodes.get_node(node));
        for anc in self.nodes.ancestors(node) {
            if self.transparent_field_key(anc.kind(), ref_span) {
                continue;
            }
            if let Some(&fid) = self.fn_of_node.get(&anc.id()) {
                return fid;
            }
        }
        Fid(0)
    }

    /// Context hops for every resolved reference: the hosting context's
    /// node (owning function, class, or lexical for-head) found by scope
    /// walk, the depth counted over the use site's node ancestors.
    fn compute_reference_depths(&mut self) {
        for rid in 0..self.scoping.references_len() {
            let rid = ReferenceId::from_usize(rid);
            let reference = self.scoping.get_reference(rid);
            let Some(symbol) = reference.symbol_id() else {
                continue;
            };
            // only context loads need a depth: captured symbols, and
            // class / for-head slots (always context-allocated). Local
            // and param references never consult the table.
            let decl_scope = self.scoping.symbol_scope_id(symbol);
            if !self.captured.contains(&symbol) {
                let mut cur = decl_scope;
                let fn_hosted = loop {
                    if self.fn_scope_to_fid.contains_key(&cur) {
                        break true;
                    }
                    if self.class_of_scope.contains_key(&cur)
                        || self.for_of_scope.contains_key(&cur)
                    {
                        break false;
                    }
                    match self.scoping.scope_parent_id(cur) {
                        Some(p) => cur = p,
                        None => break true,
                    }
                };
                if fn_hosted {
                    continue;
                }
            }
            let Some(host) = self.hosting_node(decl_scope) else {
                continue;
            };
            let depth = self.ctx_hops_from_ref(reference.node_id(), host);
            self.ref_depth.insert(rid, depth);
        }
    }

    /// The AST node of the context hosting a declaration scope's slots.
    /// Like [`ctx_hops_until`], but field computed keys skip the
    /// synthesized initializer frame.
    fn ctx_hops_from_ref(&self, node: NodeId, target: NodeId) -> u32 {
        let ref_span = GetSpan::span(self.nodes.get_node(node));
        let mut passed_use = false;
        let mut hops = 0u32;
        for anc in self.nodes.ancestors(node) {
            if self.transparent_field_key(anc.kind(), ref_span) {
                continue;
            }
            if !self.node_creates_ctx(anc.id()) {
                continue;
            }
            if !passed_use {
                passed_use = true;
            } else {
                hops += 1;
            }
            if anc.id() == target {
                return hops;
            }
        }
        unreachable!("target is an ancestor of the use site")
    }

    fn hosting_node(&self, decl_scope: ScopeId) -> Option<NodeId> {
        let mut cur = decl_scope;
        loop {
            if let Some(&fid) = self.fn_scope_to_fid.get(&cur) {
                return Some(self.f_node(fid));
            }
            if let Some(&idx) = self.class_of_scope.get(&cur) {
                if self.classes[idx.0 as usize].slot_count > 0 {
                    return Some(self.classes[idx.0 as usize].node);
                }
            }
            if let Some(&node) = self.for_of_scope.get(&cur) {
                return Some(node);
            }
            cur = self.scoping.scope_parent_id(cur)?;
        }
    }

    fn f_node(&self, fid: Fid) -> NodeId {
        self.fid_node[&fid]
    }

    /// The function owning slots declared in oxc `scope`.
    fn scope_owner(&self, scope: ScopeId) -> Fid {
        let mut cur = scope;
        loop {
            if let Some(&fid) = self.fn_scope_to_fid.get(&cur) {
                return fid;
            }
            cur = self
                .scoping
                .scope_parent_id(cur)
                .expect("scope chain ends at the root");
        }
    }
}

fn collect_binding_symbols(p: &BindingPattern<'_>) -> Vec<SymbolId> {
    fn collect(p: &BindingPattern<'_>, out: &mut Vec<SymbolId>) {
        match p {
            BindingPattern::BindingIdentifier(b) => {
                if let Some(sym) = b.symbol_id.get() {
                    out.push(sym);
                }
            }
            BindingPattern::ObjectPattern(o) => {
                for prop in &o.properties {
                    collect(&prop.value, out);
                }
                if let Some(rest) = &o.rest {
                    collect(&rest.argument, out);
                }
            }
            BindingPattern::ArrayPattern(a) => {
                for el in a.elements.iter().flatten() {
                    collect(el, out);
                }
                if let Some(rest) = &a.rest {
                    collect(&rest.argument, out);
                }
            }
            BindingPattern::AssignmentPattern(a) => collect(&a.left, out),
        }
    }
    let mut out = Vec::new();
    collect(p, &mut out);
    out
}

fn collect_params<'a>(ps: &'a FormalParameters<'a>) -> (Vec<ParamData<'a>>, u32) {
    let mut params = Vec::new();
    let mut length = 0u32;
    let mut simple = true;
    for p in &ps.items {
        if simple {
            if p.initializer.is_some() || !matches!(p.pattern, BindingPattern::BindingIdentifier(_))
            {
                simple = false;
            } else {
                length += 1;
            }
        }
        params.push(ParamData {
            pattern: &p.pattern,
            default: p.initializer.as_deref(),
            rest: false,
        });
    }
    if let Some(rest) = &ps.rest {
        params.push(ParamData {
            pattern: &rest.rest.argument,
            default: None,
            rest: true,
        });
    }
    (params, length)
}

fn member_display_name(key: &PropertyKey, kind: MethodDefinitionKind) -> Option<String> {
    let base = match key {
        PropertyKey::StaticIdentifier(i) => Some(i.name.to_string()),
        PropertyKey::StringLiteral(s) => Some(s.value.to_string()),
        PropertyKey::NumericLiteral(n) => Some(n.value.to_string()),
        _ => None,
    };
    match kind {
        MethodDefinitionKind::Get => base.map(|b| format!("get {b}")),
        MethodDefinitionKind::Set => base.map(|b| format!("set {b}")),
        _ => base,
    }
}
