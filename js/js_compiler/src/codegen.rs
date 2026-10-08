use std::collections::HashMap;

use oxc_ast::ast::*;
use oxc_semantic::Scoping;
use oxc_span::{GetSpan, Span};
use oxc_syntax::node::NodeId;
use oxc_syntax::scope::{ScopeFlags, ScopeId};
use oxc_syntax::symbol::{SymbolFlags, SymbolId};

use bytecode::{
    BuiltinFn, CallableKind, ConstIdx, Constant, FnBuilder, FunctionId, FunctionMeta, Label,
    Opcode, Program, PropertyFlags, Reg, RegList, RtArg,
};

use crate::analysis::{ClassIdx, Facts, Fid, FnBody, FnKind, Home, MemberKind, Mode, Special};

/// A construct the materializer does not support yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub span: Span,
    pub feature: &'static str,
}

impl CompileError {
    fn new(span: Span, feature: &'static str) -> Self {
        Self { span, feature }
    }
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unsupported: {} at {}..{}",
            self.feature, self.span.start, self.span.end
        )
    }
}

/// A breakable statement: loops add a continue target, switches don't.
struct Breakable {
    labels: Vec<String>,
    breaks: Label,
    continues: Option<Label>,
    /// loops owning a head context: jumps out must unwind past it
    unwind_ctx: Option<Reg>,
}

/// A store target ready for the final store.
enum StoreTarget {
    Named {
        obj: Reg,
        name_idx: ConstIdx,
        /// whether `obj` was staged as a temp that must be dropped at the
        /// end (false when `obj` is the frame receiver, operand 0)
        obj_staged: bool,
    },
    Keyed {
        obj: Reg,
        key: Reg,
    },
    /// `this.#x = v`
    PrivateKeyed {
        obj: Reg,
        key: Reg,
    },
    SuperNamed {
        recv: Reg,
        home: Reg,
        name_idx: ConstIdx,
    },
    SuperKeyed {
        recv: Reg,
        home: Reg,
        key: Reg,
    },
}

/// Name hint for NamedEvaluation in pattern defaults (ES 8.4.3).
enum NameHint {
    Const(ConstIdx),
    Reg(Reg),
    None,
}

/// The slot decision for a declared symbol, shared between its store and
/// load sites (assigned in the owning function's prologue, or lazily for
/// class / for-head context slots).
#[derive(Clone, Copy, Debug)]
enum Slot {
    Param {
        index: u32,
        hole_check: bool,
    },
    Local {
        reg: u32,
        hole_check: bool,
    },
    /// context slot in the owning function's frame context; the depth is
    /// a property of each use site
    Ctx {
        slot: u32,
        hole_check: bool,
    },
    /// context slot in a class / for-head block context (the depth comes
    /// from the precomputed per-reference table)
    CtxAt {
        slot: u32,
        hole_check: bool,
    },
    /// REPL mode: script-level bindings are global object properties
    Global,
}

/// Per-function frame layout, built at the prologue.
struct Layout {
    register_count: u32,
    this_slot: Option<u32>,
    new_target_slot: Option<u32>,
    this_function_slot: Option<u32>,
    slot_names: Vec<Vec<u8>>,
    /// whether the prologue must create/push a function context: any
    /// context-allocated binding, or a direct `eval` that may add
    /// bindings at runtime
    creates_context: bool,
}

/// How an identifier reference resolves at its use site.
enum IdRes {
    Global,
    Dynamic,
    Slot(Slot, u32),
}

/// A destructuring view over either pattern flavor.
#[derive(Clone, Copy)]
enum PatTarget<'r, 'a> {
    Binding(&'r BindingPattern<'a>),
    Assign(&'r AssignmentTarget<'a>),
    /// shorthand assignment-property leaf: `{a}` / `{a = 1}`
    AssignIdent(&'r IdentifierReference<'a>),
}

impl PatTarget<'_, '_> {
    fn span(&self) -> Span {
        match self {
            PatTarget::Binding(p) => p.span(),
            PatTarget::Assign(t) => t.span(),
            PatTarget::AssignIdent(i) => i.span,
        }
    }
}

enum PatElement<'r, 'a> {
    Hole,
    Item {
        target: PatTarget<'r, 'a>,
        default: Option<&'r Expression<'a>>,
    },
    Rest(PatTarget<'r, 'a>),
}

enum PatProperty<'r, 'a> {
    Prop {
        key: &'r PropertyKey<'a>,
        computed: bool,
        target: PatTarget<'r, 'a>,
        default: Option<&'r Expression<'a>>,
    },
    /// shorthand `{a}` assignment property: the key is the binding name
    Named {
        name: Vec<u8>,
        target: PatTarget<'r, 'a>,
        default: Option<&'r Expression<'a>>,
    },
    Rest(PatTarget<'r, 'a>),
}

fn array_binding_elements<'r, 'a>(a: &'r ArrayPattern<'a>) -> Vec<PatElement<'r, 'a>> {
    let mut els = Vec::new();
    for el in a.elements.iter() {
        match el {
            None => els.push(PatElement::Hole),
            Some(p) => els.push(binding_item(p)),
        }
    }
    if let Some(rest) = &a.rest {
        els.push(PatElement::Rest(PatTarget::Binding(&rest.argument)));
    }
    els
}

fn object_binding_props<'r, 'a>(o: &'r ObjectPattern<'a>) -> Vec<PatProperty<'r, 'a>> {
    let mut props = Vec::new();
    for prop in &o.properties {
        let (target, default) = split_binding_default(&prop.value);
        props.push(PatProperty::Prop {
            key: &prop.key,
            computed: prop.computed,
            target,
            default,
        });
    }
    if let Some(rest) = &o.rest {
        props.push(PatProperty::Rest(PatTarget::Binding(&rest.argument)));
    }
    props
}

fn array_assign_elements<'r, 'a>(a: &'r ArrayAssignmentTarget<'a>) -> Vec<PatElement<'r, 'a>> {
    let mut els = Vec::new();
    for el in a.elements.iter() {
        match el {
            None => els.push(PatElement::Hole),
            Some(AssignmentTargetMaybeDefault::AssignmentTargetWithDefault(d)) => {
                els.push(PatElement::Item {
                    target: PatTarget::Assign(&d.binding),
                    default: Some(&d.init),
                });
            }
            Some(other) => {
                els.push(PatElement::Item {
                    target: PatTarget::Assign(other.as_assignment_target().expect("plain target")),
                    default: None,
                });
            }
        }
    }
    if let Some(rest) = &a.rest {
        els.push(PatElement::Rest(PatTarget::Assign(&rest.target)));
    }
    els
}

fn object_assign_props<'r, 'a>(o: &'r ObjectAssignmentTarget<'a>) -> Vec<PatProperty<'r, 'a>> {
    let mut props = Vec::new();
    for prop in &o.properties {
        match prop {
            AssignmentTargetProperty::AssignmentTargetPropertyIdentifier(p) => {
                let p = &**p;
                props.push(PatProperty::Named {
                    name: p.binding.name.as_bytes().to_vec(),
                    target: PatTarget::AssignIdent(&p.binding),
                    default: p.init.as_ref(),
                });
            }
            AssignmentTargetProperty::AssignmentTargetPropertyProperty(p) => {
                let p = &**p;
                let (target, default) = match &p.binding {
                    AssignmentTargetMaybeDefault::AssignmentTargetWithDefault(d) => {
                        (PatTarget::Assign(&d.binding), Some(&d.init))
                    }
                    other => (
                        PatTarget::Assign(other.as_assignment_target().expect("plain target")),
                        None,
                    ),
                };
                props.push(PatProperty::Prop {
                    key: &p.name,
                    computed: p.computed,
                    target,
                    default,
                });
            }
        }
    }
    if let Some(rest) = &o.rest {
        props.push(PatProperty::Rest(PatTarget::Assign(&rest.target)));
    }
    props
}

/// One binding-pattern array element, defaults unwrapped.
fn binding_item<'r, 'a>(p: &'r BindingPattern<'a>) -> PatElement<'r, 'a> {
    match p {
        BindingPattern::AssignmentPattern(a) => PatElement::Item {
            target: PatTarget::Binding(&a.left),
            default: Some(&a.right),
        },
        other => PatElement::Item {
            target: PatTarget::Binding(other),
            default: None,
        },
    }
}

fn is_simple(p: &BindingPattern<'_>) -> bool {
    matches!(p, BindingPattern::BindingIdentifier(_))
}

/// The immediate form of a register arithmetic opcode (`Add` -> `AddImmediate`).
fn immediate_variant(op: Opcode) -> Option<Opcode> {
    Some(match op {
        Opcode::Add => Opcode::AddImmediate,
        Opcode::Sub => Opcode::SubImmediate,
        Opcode::Mul => Opcode::MulImmediate,
        Opcode::Div => Opcode::DivImmediate,
        Opcode::Mod => Opcode::ModImmediate,
        Opcode::Exp => Opcode::ExpImmediate,
        Opcode::BitwiseOr => Opcode::BitwiseOrImmediate,
        Opcode::BitwiseXor => Opcode::BitwiseXorImmediate,
        Opcode::BitwiseAnd => Opcode::BitwiseAndImmediate,
        Opcode::ShiftLeft => Opcode::ShiftLeftImmediate,
        Opcode::ShiftRight => Opcode::ShiftRightImmediate,
        Opcode::ShiftRightLogical => Opcode::ShiftRightLogicalImmediate,
        _ => return None,
    })
}

/// Whether `op` is commutative (`imm OP x` == `x OP imm`).
fn is_commutative(op: Opcode) -> bool {
    matches!(
        op,
        Opcode::Add | Opcode::Mul | Opcode::BitwiseOr | Opcode::BitwiseXor | Opcode::BitwiseAnd
    )
}

/// Whether evaluating `e` can write a non-captured local (and thus whether
/// a register left operand may be read after it). A call runs in its own
/// frame and cannot reach our locals, but the callee/argument expressions
/// can contain assignments: any assignment or update in the subtree makes
/// this conservative.
fn rhs_cannot_clobber_locals(e: &Expression<'_>) -> bool {
    struct Scan {
        assigns: bool,
    }
    impl<'a> oxc_ast_visit::Visit<'a> for Scan {
        fn visit_assignment_expression(&mut self, _it: &oxc_ast::ast::AssignmentExpression<'a>) {
            self.assigns = true;
        }
        fn visit_update_expression(&mut self, _it: &oxc_ast::ast::UpdateExpression<'a>) {
            self.assigns = true;
        }
    }
    use oxc_ast_visit::Visit as _;
    let mut scan = Scan { assigns: false };
    scan.visit_expression(e);
    !scan.assigns
}

/// The integer value of a numeric literal that fits a 2-byte immediate
/// operand, if `e` is one.
fn int_literal(e: &Expression<'_>) -> Option<i32> {
    let Expression::NumericLiteral(n) = e else {
        return None;
    };
    let v = n.value;
    (v.fract() == 0.0 && v >= i16::MIN as f64 && v <= i16::MAX as f64).then_some(v as i32)
}

fn arith_opcode(op: BinaryOperator) -> Option<Opcode> {
    use BinaryOperator as Op;
    Some(match op {
        Op::Addition => Opcode::Add,
        Op::Subtraction => Opcode::Sub,
        Op::Multiplication => Opcode::Mul,
        Op::Division => Opcode::Div,
        Op::Remainder => Opcode::Mod,
        Op::Exponential => Opcode::Exp,
        Op::ShiftLeft => Opcode::ShiftLeft,
        Op::ShiftRight => Opcode::ShiftRight,
        Op::ShiftRightZeroFill => Opcode::ShiftRightLogical,
        Op::BitwiseOR => Opcode::BitwiseOr,
        Op::BitwiseXOR => Opcode::BitwiseXor,
        Op::BitwiseAnd => Opcode::BitwiseAnd,
        _ => return None,
    })
}

/// A borrowed member expression: `Expression` flattens the member kinds,
/// so helpers consume this view instead.
#[derive(Clone, Copy)]
enum MemberRef<'a> {
    Static(&'a StaticMemberExpression<'a>),
    Computed(&'a ComputedMemberExpression<'a>),
    Private(&'a PrivateFieldExpression<'a>),
}

impl<'a> MemberRef<'a> {
    fn of(e: &'a Expression<'a>) -> Option<Self> {
        Some(match e {
            Expression::StaticMemberExpression(m) => Self::Static(m),
            Expression::ComputedMemberExpression(m) => Self::Computed(m),
            Expression::PrivateFieldExpression(m) => Self::Private(m),
            _ => return None,
        })
    }

    fn span(&self) -> Span {
        match self {
            Self::Static(m) => m.span,
            Self::Computed(m) => m.span,
            Self::Private(m) => m.span,
        }
    }

    fn node_id(&self) -> NodeId {
        match self {
            Self::Static(m) => m.node_id.get(),
            Self::Computed(m) => m.node_id.get(),
            Self::Private(m) => m.node_id.get(),
        }
    }
}

fn is_super_member(m: MemberRef<'_>) -> bool {
    match m {
        MemberRef::Static(e) => matches!(e.object, Expression::Super(_)),
        MemberRef::Computed(e) => matches!(e.object, Expression::Super(_)),
        MemberRef::Private(_) => false,
    }
}

/// `this.<name> = ...` stores: V8 counts them to presize constructor
/// instances. Only plain assignments count (compound/update stores are
/// absorbed by slack).
fn is_this_member(m: MemberRef<'_>) -> bool {
    match m {
        MemberRef::Static(e) => matches!(e.object, Expression::ThisExpression(_)),
        MemberRef::Computed(e) => matches!(e.object, Expression::ThisExpression(_)),
        MemberRef::Private(_) => false,
    }
}

/// A member expression inside a simple assignment target.
fn assign_member_ref_for_left<'a>(t: &'a ForStatementLeft<'a>) -> Option<MemberRef<'a>> {
    match t {
        ForStatementLeft::StaticMemberExpression(m) => Some(MemberRef::Static(m)),
        ForStatementLeft::ComputedMemberExpression(m) => Some(MemberRef::Computed(m)),
        ForStatementLeft::PrivateFieldExpression(m) => Some(MemberRef::Private(m)),
        _ => None,
    }
}

fn assign_member_ref<'a>(t: &'a AssignmentTarget<'a>) -> Option<MemberRef<'a>> {
    match t {
        AssignmentTarget::StaticMemberExpression(m) => Some(MemberRef::Static(m)),
        AssignmentTarget::ComputedMemberExpression(m) => Some(MemberRef::Computed(m)),
        AssignmentTarget::PrivateFieldExpression(m) => Some(MemberRef::Private(m)),
        _ => None,
    }
}

fn simple_member_ref<'a>(t: &'a SimpleAssignmentTarget<'a>) -> Option<MemberRef<'a>> {
    match t {
        SimpleAssignmentTarget::StaticMemberExpression(m) => Some(MemberRef::Static(m)),
        SimpleAssignmentTarget::ComputedMemberExpression(m) => Some(MemberRef::Computed(m)),
        SimpleAssignmentTarget::PrivateFieldExpression(m) => Some(MemberRef::Private(m)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// shared compiler state: slot memo + scope queries
// ---------------------------------------------------------------------------

pub struct Compiler<'a, 'p> {
    scoping: &'p Scoping,
    facts: &'p Facts<'a>,
    /// scopes allocating from each function's register/context space
    /// (class / for contexts excluded), in scope-id order
    scopes_by_fid: Vec<Vec<ScopeId>>,
    /// context-free for-head scopes allocating function registers
    for_scopes_by_fid: Vec<Vec<ScopeId>>,
    /// scopes nested under a for scope that the function hosts
    nested_scopes_by_fid: Vec<Vec<ScopeId>>,
    /// symbol → slot, the single source of agreement between stores/loads
    slots: HashMap<SymbolId, Slot>,
    /// per-fid layouts, filled as functions are compiled (reads only ever
    /// target already-compiled ancestors)
    layouts: Vec<Layout>,
    /// fid → its oxc function scope (eval detection)
    fid_scope: HashMap<Fid, ScopeId>,
    /// for-head scopes: next slot index (declaration-order assignment)
    for_counters: HashMap<ScopeId, u32>,
}

pub fn generate<'a>(scoping: &Scoping, facts: &Facts<'a>) -> Result<Program, CompileError> {
    let n = facts.functions.len();
    let mut scopes_by_fid: Vec<Vec<ScopeId>> = (0..n).map(|_| Vec::new()).collect();
    for sid in 0..scoping.scopes_len() {
        let sid = ScopeId::from_usize(sid);
        if let Some(&fid) = facts.scope_owner.get(&sid) {
            if fid != Fid(u32::MAX) {
                scopes_by_fid[fid.0 as usize].push(sid);
            }
        }
    }
    // register-allocated for heads (no context) and bindings nested under
    // a for scope that the function hosts
    let mut for_scopes_by_fid: Vec<Vec<ScopeId>> = (0..n).map(|_| Vec::new()).collect();
    let mut nested_scopes_by_fid: Vec<Vec<ScopeId>> = (0..n).map(|_| Vec::new()).collect();
    for sid in 0..scoping.scopes_len() {
        let sid = ScopeId::from_usize(sid);
        if facts.class_of_scope.contains_key(&sid) || facts.fn_scope_to_fid.contains_key(&sid) {
            continue;
        }
        if let Some(&node) = facts.for_of_scope.get(&sid) {
            if !facts.for_slots.contains_key(&node) {
                let mut cur = scoping.scope_parent_id(sid);
                while let Some(s) = cur {
                    if let Some(&fid) = facts.fn_scope_to_fid.get(&s) {
                        for_scopes_by_fid[fid.0 as usize].push(sid);
                        break;
                    }
                    cur = scoping.scope_parent_id(s);
                }
            }
            continue;
        }
        let mut crossed_for = false;
        let mut owner = None;
        let mut cur = scoping.scope_parent_id(sid);
        while let Some(s) = cur {
            if let Some(&fid) = facts.fn_scope_to_fid.get(&s) {
                owner = Some(fid);
                break;
            }
            if facts.class_of_scope.contains_key(&s) {
                break;
            }
            if facts.for_of_scope.contains_key(&s) {
                crossed_for = true;
            }
            cur = scoping.scope_parent_id(s);
        }
        if crossed_for && let Some(fid) = owner {
            nested_scopes_by_fid[fid.0 as usize].push(sid);
        }
    }
    let mut compiler = Compiler {
        scoping,
        facts,
        scopes_by_fid,
        for_scopes_by_fid,
        nested_scopes_by_fid,
        slots: HashMap::new(),
        layouts: (0..n)
            .map(|_| Layout {
                register_count: 0,
                this_slot: None,
                new_target_slot: None,
                this_function_slot: None,
                slot_names: Vec::new(),
                creates_context: true,
            })
            .collect(),
        fid_scope: facts
            .fn_scope_to_fid
            .iter()
            .map(|(&s, &f)| (f, s))
            .collect(),
        for_counters: HashMap::new(),
    };
    // Frame layouts decide both register counts and whether a function
    // pushes a context; context depths are baked into the bytecode, so
    // they must be settled before any body (including nested depth
    // calculations) is emitted.
    for fid in 0..n as u32 {
        let layout = compiler.assign_function_slots(Fid(fid));
        compiler.layouts[fid as usize] = layout;
    }
    let mut program = Program::with_capacity(n);
    for fid in 0..n as u32 {
        let _span = trace::debug_span!("js::function", fid).entered();
        let mut fgen = FunctionGen::new(&mut compiler, Fid(fid));
        fgen.emit_function_body()?;
        program.add_function(fgen.finish());
    }
    #[cfg(debug_assertions)]
    if let Err(err) = bytecode::validate(&program) {
        panic!("invalid bytecode: {err:?}");
    }
    Ok(program)
}

impl<'a, 'p> Compiler<'a, 'p> {
    fn scope_creates_ctx(&self, scope: ScopeId) -> bool {
        if let Some(&fid) = self.facts.fn_scope_to_fid.get(&scope) {
            return self.layouts[fid.0 as usize].creates_context;
        }
        if let Some(&idx) = self.facts.class_of_scope.get(&scope) {
            return self.facts.classes[idx.0 as usize].slot_count > 0;
        }
        self.facts
            .for_of_scope
            .get(&scope)
            .is_some_and(|node| self.facts.for_slots.contains_key(node))
    }

    /// Whether `host` is the context-creating scope at or above
    /// `decl_scope` that hosts its slots.
    fn hosts(&self, host: ScopeId, decl_scope: ScopeId) -> bool {
        let mut cur = decl_scope;
        loop {
            if cur == host && self.scope_creates_ctx(cur) {
                return true;
            }
            if self.scope_creates_ctx(cur) {
                return false;
            }
            cur = self
                .scoping
                .scope_parent_id(cur)
                .expect("scope chain ends at the root");
        }
    }

    /// ContextObject hops from the use site's scope to the context hosting
    /// `decl_scope`'s slots (the use site's own context-creating
    /// ancestor is depth 0).
    fn depth_to(&self, use_scope: ScopeId, decl_scope: ScopeId) -> u32 {
        // depth = index of the hosting context in the use site's context
        // chain, where the innermost (use site's own) context is index 0
        let mut passed_use = false;
        let mut hops = 0u32;
        let mut cur = use_scope;
        loop {
            if self.scope_creates_ctx(cur) {
                if !passed_use {
                    passed_use = true;
                } else {
                    hops += 1;
                }
                if self.hosts(cur, decl_scope) {
                    return hops;
                }
            }
            cur = self
                .scoping
                .scope_parent_id(cur)
                .expect("scope chain ends at the root");
        }
    }

    /// The memoized slot decision for a symbol.
    fn slot_of(&mut self, sym: SymbolId) -> Slot {
        if let Some(&slot) = self.slots.get(&sym) {
            return slot;
        }
        let scope = self.scoping.symbol_scope_id(sym);
        // only class / for-head symbols reach the lazy path:
        // function-owned symbols are assigned eagerly at their
        // function's prologue
        let slot = if let Some(&idx) = self.facts.class_of_scope.get(&scope) {
            Slot::CtxAt {
                slot: self.facts.classes[idx.0 as usize].name_slot.unwrap_or(0),
                hole_check: true,
            }
        } else {
            let slot = self.for_next_entry(scope);
            Slot::CtxAt {
                slot,
                hole_check: true,
            }
        };
        self.slots.insert(sym, slot);
        slot
    }

    /// Next context-slot index for a for-head scope (declaration order).
    fn for_next_entry(&mut self, scope: ScopeId) -> u32 {
        let n = self.scoping.iter_bindings_in(scope).count() as u32;
        let next = self.for_counters.get(&scope).copied().unwrap_or(0);
        self.for_counters.insert(scope, next + 1);
        debug_assert!(next < n, "for-head slot overflow");
        next
    }

    /// Assign every slot of `fid`'s scopes (its prologue's layout).
    fn assign_function_slots(&mut self, fid: Fid) -> Layout {
        let repl_root = matches!(self.facts.mode, Mode::Repl)
            && self.facts.functions[fid.0 as usize].kind == FnKind::Script;
        let non_simple = self.facts.functions[fid.0 as usize]
            .params
            .iter()
            .any(|p| p.rest || p.default.is_some() || !is_simple(p.pattern));
        // oxc propagates DirectEval to every ancestor scope, so the
        // function scope flag covers calls anywhere inside it
        let calls_eval = self
            .fid_scope
            .get(&fid)
            .is_some_and(|&s| self.scoping.scope_flags(s).contains(ScopeFlags::DirectEval));

        let mut next_reg = 0u32;
        let mut next_ctx = 0u32;
        let mut slot_names: Vec<Vec<u8>> = Vec::new();
        let mut this_slot = None;
        if self.facts.captures_this.contains(&fid) {
            this_slot = Some(next_ctx);
            slot_names.push(b"this".to_vec());
            next_ctx += 1;
        }
        let mut new_target_slot = None;
        if self.facts.captures_new_target.contains(&fid) {
            new_target_slot = Some(next_ctx);
            slot_names.push(b".new.target".to_vec());
            next_ctx += 1;
        }
        let mut this_function_slot = None;
        if self.facts.needs_this_function.contains(&fid) {
            this_function_slot = Some(next_ctx);
            slot_names.push(b".this_function".to_vec());
            next_ctx += 1;
        }

        for &sid in &self.scopes_by_fid[fid.0 as usize] {
            for sym in self.scoping.iter_bindings_in(sid) {
                if repl_root && sid == self.facts.root {
                    self.slots.insert(sym, Slot::Global);
                    continue;
                }
                let is_param = self.facts.param_symbols.contains_key(&sym);
                let is_pattern_param = self.facts.pattern_params.contains(&sym);
                let hole_check = if is_pattern_param {
                    true
                } else if is_param {
                    non_simple
                } else {
                    self.scoping
                        .symbol_flags(sym)
                        .intersects(SymbolFlags::BlockScopedVariable | SymbolFlags::Class)
                };
                let forced = calls_eval || self.facts.captured.contains(&sym);
                let slot = if is_param && !is_pattern_param && !forced {
                    Slot::Param {
                        index: self.facts.param_symbols[&sym],
                        hole_check,
                    }
                } else if forced {
                    let slot = next_ctx;
                    next_ctx += 1;
                    slot_names.push(self.scoping.symbol_name(sym).as_bytes().to_vec());
                    Slot::Ctx { slot, hole_check }
                } else {
                    let reg = next_reg;
                    next_reg += 1;
                    Slot::Local { reg, hole_check }
                };
                self.slots.insert(sym, slot);
            }
        }

        for i in 0..self.for_scopes_by_fid[fid.0 as usize].len() {
            let sid = self.for_scopes_by_fid[fid.0 as usize][i];
            for sym in self.scoping.iter_bindings_in(sid) {
                let reg = next_reg;
                next_reg += 1;
                self.slots.insert(
                    sym,
                    Slot::Local {
                        reg,
                        hole_check: true,
                    },
                );
            }
        }
        for i in 0..self.nested_scopes_by_fid[fid.0 as usize].len() {
            let sid = self.nested_scopes_by_fid[fid.0 as usize][i];
            for sym in self.scoping.iter_bindings_in(sid) {
                if calls_eval || self.facts.captured.contains(&sym) {
                    continue;
                }
                let hole_check = self
                    .scoping
                    .symbol_flags(sym)
                    .intersects(SymbolFlags::BlockScopedVariable | SymbolFlags::Class);
                let reg = next_reg;
                next_reg += 1;
                self.slots.insert(sym, Slot::Local { reg, hole_check });
            }
        }

        let creates_context = !slot_names.is_empty() || calls_eval;
        debug_assert_eq!(
            creates_context,
            self.facts.creates_context.contains(&fid),
            "context predicate desynced from the slot layout"
        );
        Layout {
            register_count: next_reg,
            this_slot,
            new_target_slot,
            this_function_slot,
            creates_context,
            slot_names,
        }
    }
}

// ---------------------------------------------------------------------------
// per-function emission
// ---------------------------------------------------------------------------

struct FunctionGen<'c, 'a, 'p> {
    c: &'c mut Compiler<'a, 'p>,
    fid: Fid,
    b: FnBuilder,
    /// register holding the pushed-over context for prologue/epilogue
    ctx_save: Reg,
    /// whether the prologue created/pushed a function context
    frame_context: bool,
    /// script completion-value register (scripts/eval only)
    completion: Option<Reg>,
    breakables: Vec<Breakable>,
    /// labels forwarded into an enclosing loop/switch by LabeledStatements
    nested_labels: Vec<String>,
}

impl<'c, 'a, 'p> FunctionGen<'c, 'a, 'p> {
    fn new(c: &'c mut Compiler<'a, 'p>, fid: Fid) -> Self {
        let arity = c.facts.functions[fid.0 as usize].params.len() as u32;
        Self {
            c,
            fid,
            b: FnBuilder::new(arity),
            ctx_save: Reg::new(0),
            frame_context: false,
            completion: None,
            breakables: Vec::new(),
            nested_labels: Vec::new(),
        }
    }

    fn finish(self) -> bytecode::Function {
        let info = &self.c.facts.functions[self.fid.0 as usize];
        let meta = FunctionMeta {
            name: info.name.clone().map(|n| n.into_bytes().into()),
            kind: callable_kind(info.kind),
            length: info.formal_length,
            strict: info.strict,
        };
        // build failures (unbalanced temps, unbound labels) are compiler
        // bugs: every emit path pairs its temp marks and binds its labels
        self.b.finish(meta).expect("js function builder invariants")
    }

    fn err<T>(&self, span: Span, feature: &'static str) -> Result<T, CompileError> {
        Err(CompileError::new(span, feature))
    }

    // -- staging helpers ----------------------------------------------------

    fn with_temps<F, T>(&mut self, f: F) -> Result<T, CompileError>
    where
        F: FnOnce(&mut Self) -> Result<T, CompileError>,
    {
        let mark = self.b.temp_depth();
        let result = f(self);
        self.b.drop_temps(mark);
        result
    }

    /// Leave the function-context frame, if the prologue pushed one.
    fn pop_frame_context(&mut self) {
        if self.frame_context {
            self.b.pop_context(self.ctx_save);
        }
    }

    fn emit_this_initialized_check(&mut self) {
        self.b
            .call_runtime_staged(BuiltinFn::ThrowSuperNotCalledIfHole, &[RtArg::Acc]);
    }

    fn name_constant(&mut self, key: &PropertyKey<'_>) -> Result<ConstIdx, CompileError> {
        match key {
            PropertyKey::StaticIdentifier(i) => Ok(self.b.name(i.name.as_bytes())),
            PropertyKey::StringLiteral(s) => Ok(self.b.name(s.value.as_bytes())),
            _ => self.err(key.span(), "non-string property names"),
        }
    }

    /// ES 8.4.2 IsAnonymousFunctionDefinition.
    fn is_anon_function(&self, e: &Expression<'_>) -> bool {
        match e {
            Expression::FunctionExpression(f) => f.id.is_none(),
            Expression::ClassExpression(c) => c.id.is_none(),
            _ => false,
        }
    }

    fn emit_set_name_const(&mut self, bytes: &[u8]) {
        let idx = self.b.name(bytes);
        self.emit_set_name_by_const(idx);
    }

    fn emit_set_name_by_const(&mut self, idx: ConstIdx) {
        self.b.call_runtime_staged(
            BuiltinFn::SetFunctionName,
            &[RtArg::Acc, RtArg::Const(idx), RtArg::Smi(0)],
        );
    }

    fn emit_set_name_by_reg(&mut self, key: Reg, prefix: u32) {
        self.b.call_runtime_staged(
            BuiltinFn::SetFunctionName,
            &[RtArg::Acc, RtArg::Reg(key), RtArg::Smi(prefix)],
        );
    }

    fn emit_set_name_for_key_node(&mut self, key: &PropertyKey<'_>) {
        match key {
            PropertyKey::StaticIdentifier(i) => {
                self.emit_set_name_const(i.name.as_bytes());
            }
            PropertyKey::StringLiteral(s) => {
                self.emit_set_name_const(s.value.as_bytes());
            }
            _ => {}
        }
    }

    // -- variable access -------------------------------------------------------

    /// Store the accumulator into a symbol's slot (declaration stores run
    /// with the hosting context as the frame context: zero hops).
    fn store_symbol(&mut self, sym: SymbolId, name: &str) -> Result<(), CompileError> {
        match self.c.slot_of(sym) {
            Slot::Global => {
                let idx = self.b.name(name.as_bytes());
                let feedback = self.b.new_feedback();
                self.b.store_global(idx, feedback);
            }
            Slot::Param { index, .. } => {
                let reg = self.b.param(index);
                self.b.store(reg);
            }
            Slot::Local { reg, .. } => self.b.store(Reg::new(reg as i32)),
            Slot::Ctx { slot, .. } | Slot::CtxAt { slot, .. } => {
                self.b.store_context_slot(slot, 0);
            }
        }
        Ok(())
    }

    /// Store the accumulator into an identifier assignment target.
    fn store_name(&mut self, ident: &IdentifierReference<'_>) -> Result<(), CompileError> {
        match self.identifier_resolution(ident)? {
            IdRes::Global => {
                let idx = self.b.name(ident.name.as_bytes());
                let feedback = self.b.new_feedback();
                self.b.store_global(idx, feedback);
            }
            IdRes::Dynamic => {
                let idx = self.b.name(ident.name.as_bytes());
                self.b.call_runtime_staged(
                    BuiltinFn::StoreDynamicName,
                    &[RtArg::Acc, RtArg::Const(idx)],
                );
            }
            IdRes::Slot(slot, depth) => match slot {
                Slot::Param { index, .. } => {
                    let reg = self.b.param(index);
                    self.b.store(reg);
                }
                Slot::Local { reg, .. } => self.b.store(Reg::new(reg as i32)),
                Slot::Ctx { slot, .. } | Slot::CtxAt { slot, .. } => {
                    self.b.store_context_slot(slot, depth);
                }
                Slot::Global => unreachable!(),
            },
        }
        Ok(())
    }

    fn emit_identifier(&mut self, ident: &IdentifierReference<'_>) -> Result<(), CompileError> {
        match self.identifier_resolution(ident)? {
            IdRes::Global => {
                let idx = self.b.name(ident.name.as_bytes());
                let feedback = self.b.new_feedback();
                self.b.load_global(idx, feedback);
            }
            IdRes::Dynamic => {
                let idx = self.b.name(ident.name.as_bytes());
                self.b
                    .call_runtime_staged(BuiltinFn::LoadDynamicName, &[RtArg::Const(idx)]);
            }
            IdRes::Slot(slot, depth) => match slot {
                Slot::Param { index, hole_check } => {
                    let reg = self.b.param(index);
                    self.b.load(reg);
                    if hole_check {
                        self.b.throw_reference_error_if_hole();
                    }
                }
                Slot::Local { reg, hole_check } => {
                    self.b.load(Reg::new(reg as i32));
                    if hole_check {
                        self.b.throw_reference_error_if_hole();
                    }
                }
                Slot::Ctx { slot, hole_check } | Slot::CtxAt { slot, hole_check } => {
                    self.b.load_context_slot(slot, depth);
                    if hole_check {
                        self.b.throw_reference_error_if_hole();
                    }
                }
                Slot::Global => unreachable!(),
            },
        }
        Ok(())
    }

    /// Whether an identifier resolves to a plain global-object reference
    /// (the `typeof` no-throw path).
    fn resolves_global(&mut self, ident: &IdentifierReference<'_>) -> bool {
        matches!(self.identifier_resolution(ident), Ok(IdRes::Global))
    }

    fn identifier_resolution(
        &mut self,
        ident: &IdentifierReference<'_>,
    ) -> Result<IdRes, CompileError> {
        let Some(rid) = ident.reference_id.get() else {
            return Err(CompileError::new(ident.span, "unresolved identifier"));
        };
        let reference = self.c.scoping.get_reference(rid);
        match reference.symbol_id() {
            None => Ok(match self.c.facts.mode {
                Mode::Eval => IdRes::Dynamic,
                _ => IdRes::Global,
            }),
            Some(sym) => {
                let skip_hole = self.c.facts.ref_skip_hole.contains(&rid);
                let check = |hole: bool| hole && !skip_hole;
                let slot = self.c.slot_of(sym);
                match slot {
                    Slot::Global => Ok(IdRes::Global),
                    // registers carry no depth
                    Slot::Param { index, hole_check } => Ok(IdRes::Slot(
                        Slot::Param {
                            index,
                            hole_check: check(hole_check),
                        },
                        0,
                    )),
                    Slot::Local { reg, hole_check } => Ok(IdRes::Slot(
                        Slot::Local {
                            reg,
                            hole_check: check(hole_check),
                        },
                        0,
                    )),
                    Slot::CtxAt { slot, hole_check } | Slot::Ctx { slot, hole_check } => {
                        // depths are precomputed over node ancestry
                        // (synthesized field-initializer frames count);
                        // the scope-tree walk is only a fallback
                        let depth =
                            self.c
                                .facts
                                .ref_depth
                                .get(&rid)
                                .copied()
                                .unwrap_or_else(|| {
                                    let decl_scope = self.c.scoping.symbol_scope_id(sym);
                                    self.c.depth_to(reference.scope_id(), decl_scope)
                                });
                        Ok(IdRes::Slot(
                            Slot::Ctx {
                                slot,
                                hole_check: check(hole_check),
                            },
                            depth,
                        ))
                    }
                }
            }
        }
    }

    fn emit_not(&mut self) {
        let is_truthy = self.b.new_label();
        self.b.jump_if_truthy(is_truthy);
        self.b.load_true();
        let end = self.b.new_label();
        self.b.jump(end);
        self.b.bind(is_truthy);
        self.b.load_false();
        self.b.bind(end);
    }

    /// Load a private-name symbol from its class-context slot.
    fn emit_private_key_load(&mut self, node: NodeId, span: Span) -> Result<(), CompileError> {
        match self.c.facts.special.get(&node) {
            Some(Special::Private { slot, depth }) => {
                self.b.load_context_slot(*slot, *depth);
                Ok(())
            }
            _ => self.err(span, "private name outside its class"),
        }
    }

    /// Load `this` of `owner`. Inside a derived constructor the value may
    /// still be the hole: every access must throw "super not called".
    fn emit_this_for(&mut self, owner: Fid, depth: u32) {
        if owner != self.fid {
            let slot = self.c.layouts[owner.0 as usize]
                .this_slot
                .expect("owner captures this");
            self.b.load_context_slot(slot, depth);
        } else if let Some(slot) = self.c.layouts[self.fid.0 as usize].this_slot {
            self.b.load_context_slot(slot, 0);
        } else {
            self.b.load(self.b.this_reg());
        }
        if self.c.facts.functions[owner.0 as usize]
            .kind
            .is_derived_class_constructor()
        {
            self.emit_this_initialized_check();
        }
    }

    // -- expressions ------------------------------------------------------------

    fn expr(&mut self, e: &Expression<'_>) -> Result<(), CompileError> {
        match e {
            Expression::NumericLiteral(n) => {
                // (2^53 − 1: largest exactly-representable integer)
                const MAX_EXACT_INT: f64 = 9007199254740991.0;
                let f = n.value;
                let is_int = f.fract() == 0.0 && !(f == 0.0 && f.is_sign_negative());
                if is_int && f >= i16::MIN as f64 && f <= i16::MAX as f64 {
                    self.b.load_smi(f as i32);
                } else if is_int && f.abs() <= MAX_EXACT_INT {
                    let c = self.b.constant(Constant::Smi(f as i64));
                    self.b.load_constant(c);
                } else {
                    let c = self.b.constant(Constant::Float(f));
                    self.b.load_constant(c);
                }
                Ok(())
            }
            Expression::StringLiteral(s) => {
                let c = self.b.constant(Constant::String(s.value.as_bytes().into()));
                self.b.load_constant(c);
                Ok(())
            }
            Expression::BigIntLiteral(_) => self.err(e.span(), "BigInt literals"),
            Expression::RegExpLiteral(_) => self.err(e.span(), "regular expressions"),
            Expression::BooleanLiteral(b) => {
                if b.value {
                    self.b.load_true();
                } else {
                    self.b.load_false();
                }
                Ok(())
            }
            Expression::NullLiteral(_) => {
                self.b.load_null();
                Ok(())
            }
            Expression::Identifier(i) => self.emit_identifier(i),
            Expression::ThisExpression(t) => {
                match self.c.facts.special.get(&t.node_id.get()) {
                    Some(Special::This { owner, depth }) => self.emit_this_for(*owner, *depth),
                    _ => self.emit_this_for(self.fid, 0),
                }
                Ok(())
            }
            Expression::UnaryExpression(u) => self.emit_unary(u),
            Expression::UpdateExpression(u) => self.emit_update(u),
            Expression::BinaryExpression(b) => self.emit_binary(b),
            Expression::LogicalExpression(l) => self.emit_logical(l),
            Expression::ConditionalExpression(c) => {
                self.expr(&c.test)?;
                let else_l = self.b.new_label();
                self.b.jump_if_falsy(else_l);
                self.expr(&c.consequent)?;
                let end = self.b.new_label();
                self.b.jump(end);
                self.b.bind(else_l);
                self.expr(&c.alternate)?;
                self.b.bind(end);
                Ok(())
            }
            Expression::AssignmentExpression(a) => self.emit_assign(a),
            Expression::SequenceExpression(s) => {
                let [rest @ .., last] = s.expressions.as_slice() else {
                    return self.err(s.span, "empty sequence");
                };
                for x in rest {
                    self.expr(x)?;
                }
                self.expr(last)
            }
            Expression::CallExpression(c) => self.emit_call(c),
            Expression::NewExpression(n) => self.emit_new(n),
            Expression::StaticMemberExpression(m) => self.emit_property_load(MemberRef::Static(m)),
            Expression::ComputedMemberExpression(m) => {
                self.emit_property_load(MemberRef::Computed(m))
            }
            Expression::PrivateFieldExpression(m) => self.emit_property_load(MemberRef::Private(m)),
            Expression::ArrayExpression(a) => self.emit_array_literal(a),
            Expression::ObjectExpression(o) => self.emit_object_literal(o),
            Expression::FunctionExpression(f) => {
                let fid = self.c.facts.fn_of_node[&f.node_id.get()];
                let idx = self.b.constant(Constant::Callable(FunctionId(fid.0)));
                self.b.create_closure(idx);
                Ok(())
            }
            Expression::ArrowFunctionExpression(f) => {
                let fid = self.c.facts.fn_of_node[&f.node_id.get()];
                let idx = self.b.constant(Constant::Callable(FunctionId(fid.0)));
                self.b.create_closure(idx);
                Ok(())
            }
            Expression::ClassExpression(c) => {
                let idx = self.c.facts.class_of_node[&c.node_id.get()];
                self.emit_class(idx)
            }
            Expression::NewTarget(m) => {
                match self.c.facts.special.get(&m.node_id.get()) {
                    Some(Special::NewTarget { owner, depth }) if *owner != self.fid => {
                        let slot = self.c.layouts[owner.0 as usize]
                            .new_target_slot
                            .expect("arrow new.target forces the slot");
                        self.b.load_context_slot(slot, *depth);
                    }
                    _ => {
                        self.b.load_new_target();
                    }
                }
                Ok(())
            }
            Expression::ImportMeta(_) => self.err(e.span(), "import.meta"),
            Expression::PrivateInExpression(p) => {
                // `#x in obj`: (key, obj) -> bool — key rides the top slot
                let mark = self.b.temp_depth();
                let base = self.b.reserve_temps(2);
                self.emit_private_key_load(p.left.node_id.get(), p.left.span)?;
                self.b.store(Reg::new(base.index() + 1));
                self.expr(&p.right)?;
                self.b.store(base);
                self.b
                    .call_runtime(BuiltinFn::PrivateIn, RegList::new(base, 2));
                self.b.drop_temps(mark);
                Ok(())
            }
            Expression::ParenthesizedExpression(_) => {
                // parse options drop parens; defensive
                self.err(e.span(), "parenthesized expressions")
            }
            Expression::ChainExpression(_) => self.err(e.span(), "optional chaining"),
            Expression::TemplateLiteral(_) => self.err(e.span(), "template literals"),
            Expression::TaggedTemplateExpression(_) => self.err(e.span(), "template literals"),
            Expression::AwaitExpression(_) => self.err(e.span(), "await expressions"),
            Expression::YieldExpression(_) => self.err(e.span(), "generator functions"),
            Expression::ImportExpression(_) => self.err(e.span(), "dynamic import"),
            Expression::Super(_) => self.err(e.span(), "super outside a call or member"),
            _ => self.err(e.span(), "expression"),
        }
    }

    fn emit_unary(&mut self, u: &UnaryExpression<'_>) -> Result<(), CompileError> {
        match u.operator {
            UnaryOperator::LogicalNot => {
                self.expr(&u.argument)?;
                self.emit_not();
                Ok(())
            }
            UnaryOperator::UnaryNegation => {
                self.expr(&u.argument)?;
                self.b.negate();
                Ok(())
            }
            UnaryOperator::UnaryPlus => {
                // ToNumber(arg) via `arg - 0`
                self.expr(&u.argument)?;
                let arg = self.b.stage_acc();
                self.b.load_zero();
                self.b.raw(Opcode::Sub, &[arg.operand()]);
                self.b.drop_temp();
                Ok(())
            }
            UnaryOperator::Typeof => {
                // `typeof` on an unresolved global yields "undefined"
                // instead of throwing
                if let Expression::Identifier(i) = &u.argument
                    && self.resolves_global(i)
                {
                    let idx = self.b.name(i.name.as_bytes());
                    let feedback = self.b.new_feedback();
                    self.b.load_global_no_throw(idx, feedback);
                    self.b.test_typeof();
                    return Ok(());
                }
                self.expr(&u.argument)?;
                self.b.test_typeof();
                Ok(())
            }
            UnaryOperator::Void => {
                self.expr(&u.argument)?;
                self.b.load_undefined();
                Ok(())
            }
            UnaryOperator::BitwiseNot => self.err(u.span, "bitwise not"),
            UnaryOperator::Delete => self.emit_delete(u),
        }
    }

    fn emit_delete(&mut self, u: &UnaryExpression<'_>) -> Result<(), CompileError> {
        match &u.argument {
            e if MemberRef::of(e).is_some_and(is_super_member) => {
                let m = MemberRef::of(e).unwrap();
                // ReferenceError in both language modes (ES 13.5.1.2 step
                // 4.c); reference evaluation runs first
                let Some(store) = self.prepare_super_parts(m)? else {
                    return self.err(u.span, "super property");
                };
                self.release_store(&store);
                self.b.call_runtime(
                    BuiltinFn::DeleteSuperProperty,
                    RegList::new(self.b.this_reg(), 0),
                );
                Ok(())
            }
            e if MemberRef::of(e).is_some() => {
                let m = MemberRef::of(e).unwrap();
                if let MemberRef::Private(p) = m {
                    return self.err(p.span, "delete of a private name");
                }
                let object = match m {
                    MemberRef::Static(s) => &s.object,
                    MemberRef::Computed(c) => &c.object,
                    MemberRef::Private(_) => unreachable!(),
                };
                self.expr(object)?;
                let mark = self.b.temp_depth();
                let base = self.b.reserve_temps(2);
                self.b.store(Reg::new(base.index() + 1));
                match m {
                    MemberRef::Static(s) => {
                        let idx = self.b.name(s.property.name.as_bytes());
                        self.b.load_constant(idx);
                    }
                    MemberRef::Computed(c) => {
                        self.expr(&c.expression)?;
                    }
                    MemberRef::Private(_) => unreachable!(),
                }
                self.b.store(base);
                let runtime_fn = if self.c.facts.functions[self.fid.0 as usize].strict {
                    BuiltinFn::DeletePropertyStrict
                } else {
                    BuiltinFn::DeletePropertySloppy
                };
                self.b.call_runtime(runtime_fn, RegList::new(base, 2));
                self.b.drop_temps(mark);
                Ok(())
            }
            Expression::Identifier(i) => {
                // strict sites are rejected at parse time; sloppy
                // declarative bindings cannot be deleted (false), free
                // names are global-object properties
                match self.identifier_resolution(i)? {
                    IdRes::Global | IdRes::Dynamic => {
                        let idx = self.b.name(i.name.as_bytes());
                        self.b.load_constant(idx);
                        let name = self.b.stage_acc();
                        self.b
                            .call_runtime(BuiltinFn::DeleteIdentifierSloppy, RegList::new(name, 1));
                        self.b.drop_temp();
                    }
                    _ => self.b.load_false(),
                }
                Ok(())
            }
            _ => {
                // not a reference: side effects only, the result is true
                self.expr(&u.argument)?;
                self.b.load_true();
                Ok(())
            }
        }
    }

    fn emit_binary(&mut self, b: &BinaryExpression<'_>) -> Result<(), CompileError> {
        use BinaryOperator as Op;
        match b.operator {
            Op::StrictInequality => {
                self.binary_arith(&b.left, &b.right, Opcode::EqualStrict)?;
                self.emit_not();
                Ok(())
            }
            Op::Inequality => {
                self.binary_arith(&b.left, &b.right, Opcode::Equal)?;
                self.emit_not();
                Ok(())
            }
            Op::Instanceof => {
                // acc = lhs instanceof rhs: evaluate rhs first
                self.expr(&b.right)?;
                let r = self.b.stage_acc();
                self.expr(&b.left)?;
                self.b.instance_of(r);
                self.b.drop_temp();
                Ok(())
            }
            Op::In => {
                // `key in obj`: (key, obj) -> bool — key rides the top slot
                let mark = self.b.temp_depth();
                let base = self.b.reserve_temps(2);
                self.expr(&b.left)?;
                self.b.store(Reg::new(base.index() + 1));
                self.expr(&b.right)?;
                self.b.store(base);
                self.b
                    .call_runtime(BuiltinFn::HasProperty, RegList::new(base, 2));
                self.b.drop_temps(mark);
                Ok(())
            }
            _ => {
                let op = match b.operator {
                    Op::Addition => Opcode::Add,
                    Op::Subtraction => Opcode::Sub,
                    Op::Multiplication => Opcode::Mul,
                    Op::Division => Opcode::Div,
                    Op::Remainder => Opcode::Mod,
                    Op::Exponential => Opcode::Exp,
                    Op::BitwiseOR => Opcode::BitwiseOr,
                    Op::BitwiseXOR => Opcode::BitwiseXor,
                    Op::BitwiseAnd => Opcode::BitwiseAnd,
                    Op::ShiftLeft => Opcode::ShiftLeft,
                    Op::ShiftRight => Opcode::ShiftRight,
                    Op::ShiftRightZeroFill => Opcode::ShiftRightLogical,
                    Op::StrictEquality => Opcode::EqualStrict,
                    Op::Equality => Opcode::Equal,
                    Op::LessThan => Opcode::LessThan,
                    Op::GreaterThan => Opcode::GreaterThan,
                    Op::LessEqualThan => Opcode::LessThanOrEqual,
                    Op::GreaterEqualThan => Opcode::GreaterThanOrEqual,
                    _ => return self.err(b.span, "binary operator"),
                };
                self.binary_arith(&b.left, &b.right, op)
            }
        }
    }

    fn emit_logical(&mut self, l: &LogicalExpression<'_>) -> Result<(), CompileError> {
        let truthy = match l.operator {
            LogicalOperator::And => false,
            LogicalOperator::Or => true,
            LogicalOperator::Coalesce => return self.err(l.span, "nullish coalescing"),
        };
        self.expr(&l.left)?;
        let t = self.b.stage_acc();
        let short = self.b.new_label();
        if truthy {
            self.b.jump_if_truthy(short);
        } else {
            self.b.jump_if_falsy(short);
        }
        self.expr(&l.right)?;
        let end = self.b.new_label();
        self.b.jump(end);
        self.b.bind(short);
        self.b.load(t);
        self.b.bind(end);
        self.b.drop_temp();
        Ok(())
    }

    /// The register behind a plain local/parameter operand whose read is
    /// side-effect free (no TDZ check), if `e` is one.
    fn simple_register(&mut self, e: &Expression<'_>) -> Option<Reg> {
        let Expression::Identifier(ident) = e else {
            return None;
        };
        self.simple_identifier_register(ident)
    }

    /// The receiver operand of the current frame (`this`, operand 0) when
    /// `e` is a `this` expression that resolves to this frame's receiver.
    /// `None` for arrows (lexical `this`), `super`-captured `this`, and
    /// derived constructors (whose `this` load carries an initialization
    /// check that must not be elided).
    fn this_operand_reg(&self, e: &Expression<'_>) -> Option<Reg> {
        let Expression::ThisExpression(t) = e else {
            return None;
        };
        if let Some(Special::This { owner, .. }) = self.c.facts.special.get(&t.node_id.get())
            && *owner != self.fid
        {
            return None;
        }
        if self.c.facts.functions[self.fid.0 as usize]
            .kind
            .is_derived_class_constructor()
        {
            return None;
        }
        Some(self.b.this_reg())
    }

    /// A register whose value is stable while sibling expressions (call
    /// arguments, an assignment's right-hand side) evaluate: the frame
    /// receiver, or a local/parameter this function never rebinds.
    /// Captured bindings and eval-scoped functions live in contexts, so a
    /// callee can only clobber such a binding through a direct assignment
    /// in this function — exactly what `symbol_is_mutated` detects.
    fn stable_register(&mut self, e: &Expression<'_>) -> Option<Reg> {
        if let Some(reg) = self.this_operand_reg(e) {
            return Some(reg);
        }
        let Expression::Identifier(ident) = e else {
            return None;
        };
        let rid = ident.reference_id.get()?;
        let sym = self.c.scoping.get_reference(rid).symbol_id()?;
        if self.c.scoping.symbol_is_mutated(sym) {
            return None;
        }
        self.simple_identifier_register(ident)
    }

    fn simple_identifier_register(&mut self, ident: &IdentifierReference<'_>) -> Option<Reg> {
        match self.identifier_resolution(ident).ok()? {
            IdRes::Slot(
                Slot::Param {
                    index,
                    hole_check: false,
                },
                0,
            ) => Some(self.b.param(index)),
            IdRes::Slot(
                Slot::Local {
                    reg,
                    hole_check: false,
                },
                0,
            ) => Some(Reg::new(reg as i32)),
            _ => None,
        }
    }

    fn binary_arith(
        &mut self,
        lhs: &Expression<'_>,
        rhs: &Expression<'_>,
        op: Opcode,
    ) -> Result<(), CompileError> {
        if let Some(imm_op) = immediate_variant(op) {
            // `reg OP imm`
            if let Some(l) = self.simple_register(lhs)
                && let Some(imm) = int_literal(rhs)
            {
                self.b.raw(imm_op, &[l.operand(), imm as u32]);
                return Ok(());
            }
            // `lhs` is already a register: keep it as the register operand
            // and evaluate `rhs` into the accumulator. Evaluation order
            // (lhs then rhs) is preserved because a simple register has no
            // side effects.
            if let Some(l) = self.simple_register(lhs)
                && rhs_cannot_clobber_locals(rhs)
            {
                self.expr(rhs)?;
                self.b.raw(op, &[l.operand()]);
                return Ok(());
            }
            // `imm OP reg` for a commutative operator == `reg OP imm`
            if is_commutative(op)
                && let Some(imm) = int_literal(lhs)
                && let Some(r) = self.simple_register(rhs)
            {
                self.b.raw(imm_op, &[r.operand(), imm as u32]);
                return Ok(());
            }
            // complex `lhs`, immediate `rhs`: stage lhs, then `reg OP imm`
            if let Some(imm) = int_literal(rhs) {
                self.expr(lhs)?;
                let t = self.b.stage_acc();
                self.b.raw(imm_op, &[t.operand(), imm as u32]);
                self.b.drop_temp();
                return Ok(());
            }
            // general: `lhs` -> temp register, `rhs` -> accumulator
            self.expr(lhs)?;
            let t = self.b.stage_acc();
            self.expr(rhs)?;
            self.b.raw(op, &[t.operand()]);
            self.b.drop_temp();
            return Ok(());
        }
        // non-arithmetic operator (`acc OP reg`): comparisons/tests. When
        // the right operand is a plain register there is nothing to stage.
        if let Some(r) = self.simple_register(rhs) {
            self.expr(lhs)?;
            self.b.raw(op, &[r.operand()]);
            return Ok(());
        }
        self.expr(lhs)?;
        let a = self.b.stage_acc();
        self.expr(rhs)?;
        let b = self.b.stage_acc();
        self.b.load(a);
        self.b.raw(op, &[b.operand()]);
        self.b.drop_temp();
        self.b.drop_temp();
        Ok(())
    }

    fn emit_assign(&mut self, a: &AssignmentExpression<'_>) -> Result<(), CompileError> {
        if a.operator.is_assign()
            && matches!(
                &a.left,
                AssignmentTarget::ArrayAssignmentTarget(_)
                    | AssignmentTarget::ObjectAssignmentTarget(_)
            )
        {
            // destructuring assignment (ES 14.13.3): the RHS value, then
            // the pattern against it; the expression evaluates to the value
            self.expr(&a.right)?;
            let v = self.b.stage_acc();
            self.emit_assign_target_pattern(&a.left, v)?;
            self.b.load(v);
            self.b.drop_temp();
            return Ok(());
        }
        if a.operator.is_logical() {
            return self.err(a.span, "logical assignment");
        }
        if a.operator.is_assign() {
            self.emit_simple_assign(&a.left, &a.right)
        } else {
            let op = a
                .operator
                .to_binary_operator()
                .and_then(arith_opcode)
                .ok_or_else(|| CompileError::new(a.span, "assignment operator"))?;
            self.emit_compound_assign(&a.left, &a.right, op)
        }
    }

    fn emit_simple_assign(
        &mut self,
        target: &AssignmentTarget<'_>,
        value: &Expression<'_>,
    ) -> Result<(), CompileError> {
        match target {
            AssignmentTarget::AssignmentTargetIdentifier(i) => {
                if let Some(dst) = self.simple_identifier_register(i)
                    && let Expression::BinaryExpression(b) = value
                    && matches!(
                        b.operator,
                        BinaryOperator::Addition | BinaryOperator::Subtraction
                    )
                    && let Expression::Identifier(lhs) = &b.left
                    && self.simple_identifier_register(lhs).map(|r| r.operand())
                        == Some(dst.operand())
                    && let Some(src) = self.simple_register(&b.right)
                {
                    if b.operator == BinaryOperator::Addition {
                        self.b.add_loc(dst, src);
                    } else {
                        self.b.sub_loc(dst, src);
                    }
                    return Ok(());
                }
                self.expr(value)?;
                if self.is_anon_function(value) {
                    self.emit_set_name_const(i.name.as_bytes());
                }
                self.store_name(i)
            }
            t if assign_member_ref(t).is_some() => {
                let m = assign_member_ref(t).unwrap();
                if is_this_member(m) {
                    self.b.note_this_property();
                }
                let store = if is_super_member(m) {
                    self.prepare_super_store(m)?
                } else {
                    self.prepare_property_store(m)?
                };
                self.expr(value)?;
                // NamedEvaluation: `a.b = function () {}` names the closure "b"
                if let StoreTarget::Named { name_idx, .. } = &store
                    && self.is_anon_function(value)
                {
                    let name_idx = *name_idx;
                    self.emit_set_name_by_const(name_idx);
                }
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(target.span(), "assignment target"),
        }
    }

    fn emit_compound_assign(
        &mut self,
        target: &AssignmentTarget<'_>,
        value: &Expression<'_>,
        op: Opcode,
    ) -> Result<(), CompileError> {
        match target {
            AssignmentTarget::AssignmentTargetIdentifier(i) => {
                if let (Some(dst), Some(src)) = (
                    self.simple_identifier_register(i),
                    self.simple_register(value),
                ) {
                    match op {
                        Opcode::Add => {
                            self.b.add_loc(dst, src);
                            return Ok(());
                        }
                        Opcode::Sub => {
                            self.b.sub_loc(dst, src);
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                self.expr(value)?;
                let v = self.b.stage_acc();
                self.emit_identifier(i)?;
                let orig = self.b.stage_acc();
                self.b.load(v);
                self.b.raw(op, &[orig.operand()]);
                self.b.drop_temp(); // orig
                self.b.drop_temp(); // v
                self.store_name(i)?;
                Ok(())
            }
            t if assign_member_ref(t).is_some() => {
                let m = assign_member_ref(t).unwrap();
                let store = if is_super_member(m) {
                    self.prepare_super_store(m)?
                } else {
                    self.prepare_property_store(m)?
                };
                self.expr(value)?;
                let v = self.b.stage_acc();
                self.emit_property_load_of(&store);
                let orig = self.b.stage_acc();
                self.b.load(v);
                self.b.raw(op, &[orig.operand()]);
                self.b.drop_temp(); // orig
                self.b.drop_temp(); // v
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(target.span(), "assignment target"),
        }
    }

    /// With the old value staged in `orig` and the step in `d`: replace
    /// the accumulator with `ToNumber(orig) ± d` (`orig - 0` performs the
    /// ToNumber, ES 13.5.6.2).
    fn emit_add_delta(&mut self, arith: Opcode, orig: Reg, d: Reg) {
        // acc = ToNumber(orig) OP d
        self.b.load(orig);
        let t = self.b.stage_acc();
        self.b.load_zero();
        self.b.raw(Opcode::Sub, &[t.operand()]);
        self.b.drop_temp(); // t
        let lhs = self.b.stage_acc();
        self.b.load(d);
        self.b.raw(arith, &[lhs.operand()]);
        self.b.drop_temp(); // lhs
        self.b.drop_temp(); // d
    }

    fn emit_update(&mut self, u: &UpdateExpression<'_>) -> Result<(), CompileError> {
        let delta: i32 = match u.operator {
            UpdateOperator::Increment => 1,
            UpdateOperator::Decrement => -1,
        };
        let arith = if delta > 0 { Opcode::Add } else { Opcode::Sub };
        let delta = delta.unsigned_abs();

        if let SimpleAssignmentTarget::AssignmentTargetIdentifier(i) = &u.argument {
            // a plain local updates in place: `IncLoc`/`DecLoc` do the
            // ToNumeric + step and leave the numeric old value in acc
            if let Some(reg) = self.simple_identifier_register(i) {
                let op = match u.operator {
                    UpdateOperator::Increment => Opcode::IncLoc,
                    UpdateOperator::Decrement => Opcode::DecLoc,
                };
                self.b.raw(op, &[reg.operand()]);
                if u.prefix {
                    self.b.load(reg);
                }
                return Ok(());
            }
            self.emit_identifier(i)?;
            let mark = self.b.temp_depth();
            let orig = self.b.stage_acc();
            self.b.load_smi(delta as i32);
            let d = self.b.stage_acc();
            self.emit_add_delta(arith, orig, d);
            self.store_name(i)?;
            if !u.prefix {
                self.b.load(orig);
            }
            self.b.drop_temp(); // orig
            self.b.drop_temps(mark);
            return Ok(());
        }
        if let Some(m) = simple_member_ref(&u.argument) {
            let store = if is_super_member(m) {
                self.prepare_super_store(m)?
            } else {
                self.prepare_property_store(m)?
            };
            self.emit_property_load_of(&store);
            let mark = self.b.temp_depth();
            let orig = self.b.stage_acc();
            self.b.load_smi(delta as i32);
            let d = self.b.stage_acc();
            self.emit_add_delta(arith, orig, d);
            self.emit_property_store(&store);
            if !u.prefix {
                self.b.load(orig);
            }
            self.b.drop_temp(); // orig
            self.b.drop_temps(mark);
            self.release_store(&store);
            return Ok(());
        }
        self.err(u.span, "update target")
    }
}

// ---------------------------------------------------------------------------
// destructuring, property stores, calls, classes, literals
// ---------------------------------------------------------------------------

impl<'c, 'a, 'p> FunctionGen<'c, 'a, 'p> {
    fn emit_binding_pattern(&mut self, p: &BindingPattern<'_>, v: Reg) -> Result<(), CompileError> {
        match p {
            BindingPattern::ObjectPattern(o) => {
                self.emit_object_pattern(object_binding_props(o), v, true)
            }
            BindingPattern::ArrayPattern(a) => {
                self.emit_array_pattern(array_binding_elements(a), v, true)
            }
            _ => self.err(p.span(), "destructuring pattern"),
        }
    }

    fn emit_assign_target_pattern(
        &mut self,
        t: &AssignmentTarget<'_>,
        v: Reg,
    ) -> Result<(), CompileError> {
        match t {
            AssignmentTarget::ArrayAssignmentTarget(a) => {
                self.emit_array_pattern(array_assign_elements(a), v, false)
            }
            AssignmentTarget::ObjectAssignmentTarget(o) => {
                self.emit_object_pattern(object_assign_props(o), v, false)
            }
            _ => self.err(t.span(), "destructuring pattern"),
        }
    }

    /// `{a, b: c = 1, ...rest}` (ES 14.13).
    fn emit_object_pattern(
        &mut self,
        props: Vec<PatProperty<'_, '_>>,
        value_reg: Reg,
        binding: bool,
    ) -> Result<(), CompileError> {
        // RequireObjectCoercible runs even for the empty pattern
        self.b.call_runtime(
            BuiltinFn::RequireObjectCoercible,
            RegList::new(value_reg, 1),
        );
        self.b.store(value_reg);
        let has_rest = matches!(props.last(), Some(PatProperty::Rest(_)));
        // with a rest property, every earlier key stays live in contiguous
        // registers as the CopyDataProperties exclusion set
        let mut excl_base: Option<Reg> = None;
        let mut excluded = 0u32;
        for prop in &props {
            match prop {
                PatProperty::Named {
                    name,
                    target,
                    default,
                } => {
                    let name_idx = self.b.name(name);
                    if has_rest {
                        self.b.load_constant(name_idx);
                        let r = self.b.stage_acc();
                        excl_base.get_or_insert(r);
                        excluded += 1;
                    }
                    let feedback = self.b.new_feedback();
                    self.b.load_named_property(value_reg, name_idx, feedback);
                    self.emit_pattern_element(
                        *target,
                        *default,
                        binding,
                        NameHint::Const(name_idx),
                    )?;
                }
                PatProperty::Rest(target) => {
                    // native layout: (excluded..., target, source) — element
                    // 0 (the first excluded key) rides the top slot
                    self.b.create_empty_object_literal();
                    let rest_obj = self.b.stage_acc();
                    self.b.load(value_reg);
                    self.b.stage_acc();
                    let base = excl_base.unwrap_or(rest_obj);
                    let count = excluded + 2;
                    // restage the window in element order: excluded keys
                    // ride the top (excluded[0] highest), then the target,
                    // the source at the bottom slot
                    let mark2 = self.b.temp_depth();
                    let w = self.b.reserve_temps(count);
                    let top = w.index() + (count - 1) as i32;
                    for k in 0..excluded {
                        self.b.load(Reg::new(base.index() + k as i32));
                        self.b.store(Reg::new(top - k as i32));
                    }
                    self.b.load(rest_obj);
                    self.b.store(Reg::new(w.index() + 1));
                    self.b.load(Reg::new(base.index() + excluded as i32 + 1));
                    self.b.store(w);
                    self.b
                        .call_runtime(BuiltinFn::CopyDataProperties, RegList::new(w, count));
                    self.b.drop_temps(mark2);
                    self.b.store(rest_obj);
                    self.emit_pattern_leaf(*target, rest_obj, binding)?;
                    self.b.drop_temp(); // source
                    self.b.drop_temp(); // rest object
                }
                PatProperty::Prop {
                    key,
                    computed,
                    target,
                    default,
                } => {
                    let key_reg = self.stage_key(key, *computed)?;
                    let name_hint = match key_reg {
                        Some(r) => NameHint::Reg(r),
                        None => NameHint::Const(self.name_constant(key)?),
                    };
                    if has_rest {
                        match key_reg {
                            Some(r) => {
                                excl_base.get_or_insert(r);
                                excluded += 1;
                            }
                            None => {
                                // materialize the constant key for the
                                // exclusion set
                                if let NameHint::Const(idx) = name_hint {
                                    self.b.load_constant(idx);
                                    let r = self.b.stage_acc();
                                    excl_base.get_or_insert(r);
                                    excluded += 1;
                                }
                            }
                        }
                    }
                    // v = GetV(value, P); computed keys read the value
                    // straight from their register via the accumulator
                    match key_reg {
                        Some(k) => {
                            self.b.load(k);
                            let feedback = self.b.new_feedback();
                            self.b.load_keyed_property(value_reg, feedback);
                        }
                        None => {
                            let NameHint::Const(name_idx) = name_hint else {
                                unreachable!("constant keys have a constant hint")
                            };
                            let feedback = self.b.new_feedback();
                            self.b.load_named_property(value_reg, name_idx, feedback);
                        }
                    }
                    self.emit_pattern_element(*target, *default, binding, name_hint)?;
                    if !has_rest && key_reg.is_some() {
                        self.b.drop_temp();
                    }
                }
            }
        }
        for _ in 0..excluded {
            self.b.drop_temp();
        }
        Ok(())
    }

    /// `[a, b = 1, , ...rest]` (ES 14.13).
    fn emit_array_pattern(
        &mut self,
        elements: Vec<PatElement<'_, '_>>,
        value_reg: Reg,
        binding: bool,
    ) -> Result<(), CompileError> {
        // iterator = GetIterator(value)
        self.b
            .call_runtime(BuiltinFn::GetIterator, RegList::new(value_reg, 1));
        let iter = self.b.stage_acc();
        // done flag (ES 8.5.9: once done, later elements read undefined
        // without calling next again)
        self.b.load_zero();
        let done = self.b.stage_acc();
        let mut rest: Option<PatTarget<'_, '_>> = None;
        for el in &elements {
            match el {
                PatElement::Hole => {
                    // elision still consumes one iterator step
                    self.b.load(done);
                    let skip = self.b.new_label();
                    self.b.jump_if_truthy(skip);
                    self.b
                        .call_runtime(BuiltinFn::IteratorNext, RegList::new(iter, 1));
                    self.b.bind(skip);
                }
                PatElement::Rest(target) => {
                    rest = Some(*target);
                }
                PatElement::Item { target, default } => {
                    self.emit_iterator_element(*target, *default, iter, done, binding)?;
                }
            }
        }
        if let Some(target) = rest {
            // array ← remaining values (loop while !done)
            self.b.create_empty_array_literal();
            let arr = self.b.stage_acc();
            self.b.load_zero();
            let idx = self.b.stage_acc();
            let back = self.b.new_label();
            self.b.bind(back); // loop head
            self.b.load(done);
            let exit = self.b.new_label();
            self.b.jump_if_truthy(exit);
            self.b
                .call_runtime(BuiltinFn::IteratorNext, RegList::new(iter, 1));
            let result = self.b.stage_acc();
            self.b
                .call_runtime(BuiltinFn::IteratorDone, RegList::new(result, 1));
            let have = self.b.new_label();
            self.b.jump_if_falsy(have);
            self.b.load_smi(1);
            self.b.store(done);
            let after = self.b.new_label();
            self.b.jump(after);
            self.b.bind(have);
            self.b
                .call_runtime(BuiltinFn::IteratorValue, RegList::new(result, 1));
            let feedback = self.b.new_feedback();
            self.b.store_keyed_property_no_shadow(arr, idx, feedback);
            self.b.raw(Opcode::AddImmediate, &[idx.operand(), 1]); // acc = idx + 1
            self.b.store(idx);
            self.b.jump_loop(back);
            self.b.bind(after);
            self.b.drop_temp(); // result
            self.b.drop_temp(); // idx
            // both loop exits converge here
            self.b.bind(exit);
            self.b.load(arr);
            self.emit_pattern_leaf(target, arr, binding)?;
            self.b.drop_temp(); // arr
        }
        self.b.drop_temp(); // done
        self.b.drop_temp(); // iter
        Ok(())
    }

    /// Load a numeric literal key (small ints inline, floats via the pool).
    fn emit_number_key(&mut self, key: &PropertyKey<'_>) {
        if let PropertyKey::NumericLiteral(n) = key {
            let f = n.value;
            if f.fract() == 0.0 && f.is_sign_positive() && f <= i16::MAX as f64 {
                self.b.load_smi(f as i32);
            } else {
                let c = self.b.constant(Constant::Float(f));
                self.b.load_constant(c);
            }
        }
    }

    /// Stage a property key for the keyed paths: computed keys evaluate
    /// into a register, numeric literal keys load their literal; string
    /// keys stay pooled (`None`, addressed by name).
    fn stage_key(
        &mut self,
        key: &PropertyKey<'_>,
        computed: bool,
    ) -> Result<Option<Reg>, CompileError> {
        if computed {
            if let Some(e) = key.as_expression() {
                self.expr(e)?;
            }
            Ok(Some(self.b.stage_acc()))
        } else if matches!(key, PropertyKey::NumericLiteral(_)) {
            self.emit_number_key(key);
            Ok(Some(self.b.stage_acc()))
        } else {
            Ok(None)
        }
    }

    /// One array-pattern element: v = done ? undefined : IteratorValue;
    /// then the shared default handling.
    fn emit_iterator_element(
        &mut self,
        target: PatTarget<'_, '_>,
        default: Option<&Expression<'_>>,
        iter: Reg,
        done: Reg,
        binding: bool,
    ) -> Result<(), CompileError> {
        self.b.load_undefined();
        let v = self.b.stage_acc();
        self.b.load(done);
        let skip_next = self.b.new_label();
        self.b.jump_if_truthy(skip_next);
        self.b
            .call_runtime(BuiltinFn::IteratorNext, RegList::new(iter, 1));
        let result = self.b.stage_acc();
        self.b
            .call_runtime(BuiltinFn::IteratorDone, RegList::new(result, 1));
        let have = self.b.new_label();
        self.b.jump_if_falsy(have);
        self.b.load_smi(1);
        self.b.store(done);
        let after = self.b.new_label();
        self.b.jump(after);
        self.b.bind(have);
        self.b
            .call_runtime(BuiltinFn::IteratorValue, RegList::new(result, 1));
        self.b.store(v);
        self.b.bind(after);
        self.b.drop_temp(); // result
        self.b.bind(skip_next);
        // array-element defaults name anonymous functions after the
        // binding identifier (ES 8.6.2: NamedEvaluation with bindingName)
        let bind_name: Option<Vec<u8>> = match target {
            PatTarget::Binding(BindingPattern::BindingIdentifier(b)) if binding => {
                Some(b.name.as_bytes().to_vec())
            }
            PatTarget::Assign(AssignmentTarget::AssignmentTargetIdentifier(i)) => {
                Some(i.name.as_bytes().to_vec())
            }
            _ => None,
        };
        let hint = if default.is_some_and(|d| self.is_anon_function(d)) && bind_name.is_some() {
            let idx = self.b.name(&bind_name.unwrap());
            NameHint::Const(idx)
        } else {
            NameHint::None
        };
        self.emit_pattern_element_with(target, default, v, binding, hint)?;
        self.b.drop_temp(); // v
        Ok(())
    }

    /// Default application + target binding with the current value in
    /// `v` (the accumulator is not used).
    fn emit_pattern_element_with(
        &mut self,
        target: PatTarget<'_, '_>,
        default: Option<&Expression<'_>>,
        v: Reg,
        binding: bool,
        name_hint: NameHint,
    ) -> Result<(), CompileError> {
        if let Some(default) = default {
            self.b.load(v);
            let skip = self.b.new_label();
            self.b.jump_if_not_undefined(skip);
            self.expr(default)?;
            // NamedEvaluation: `{a = function(){}} = x` names the closure
            // after the property key / array index (ES 8.4.3). Binding
            // patterns always name; assignment patterns only when the
            // target is an identifier reference
            if self.is_anon_function(default) && {
                let is_ident = matches!(
                    target,
                    PatTarget::Assign(AssignmentTarget::AssignmentTargetIdentifier(_))
                        | PatTarget::AssignIdent(_)
                        | PatTarget::Binding(BindingPattern::BindingIdentifier(_))
                );
                binding || is_ident
            } {
                match name_hint {
                    NameHint::Const(idx) => self.emit_set_name_by_const(idx),
                    NameHint::Reg(r) => self.emit_set_name_by_reg(r, 0),
                    NameHint::None => {}
                }
            }
            self.b.store(v);
            self.b.bind(skip);
        }
        self.b.load(v);
        self.emit_pattern_leaf(target, v, binding)
    }

    /// Value in the accumulator: apply the default and bind into a fresh
    /// register.
    fn emit_pattern_element(
        &mut self,
        target: PatTarget<'_, '_>,
        default: Option<&Expression<'_>>,
        binding: bool,
        name_hint: NameHint,
    ) -> Result<(), CompileError> {
        let v = self.b.stage_acc();
        self.emit_pattern_element_with(target, default, v, binding, name_hint)?;
        self.b.drop_temp();
        Ok(())
    }

    /// Store the value in the accumulator into one pattern leaf: a
    /// declaration name, an assignment target, or a nested pattern.
    fn emit_pattern_leaf(
        &mut self,
        target: PatTarget<'_, '_>,
        value_reg: Reg,
        binding: bool,
    ) -> Result<(), CompileError> {
        match target {
            PatTarget::Binding(BindingPattern::BindingIdentifier(b)) => {
                let Some(sym) = b.symbol_id.get() else {
                    return self.err(b.span, "unresolved binding");
                };
                self.store_symbol(sym, b.name.as_ref())
            }
            PatTarget::Assign(AssignmentTarget::AssignmentTargetIdentifier(i)) => {
                self.store_name(i)
            }
            PatTarget::AssignIdent(i) => self.store_name(i),
            PatTarget::Assign(t) if assign_member_ref(t).is_some() => {
                let m = assign_member_ref(t).unwrap();
                let value = self.b.stage_acc();
                let store = if is_super_member(m) {
                    self.prepare_super_store(m)?
                } else {
                    self.prepare_property_store(m)?
                };
                self.b.load(value);
                self.emit_property_store(&store);
                self.release_store(&store);
                self.b.drop_temp(); // value
                Ok(())
            }
            PatTarget::Binding(BindingPattern::ObjectPattern(o)) => {
                self.emit_object_pattern(object_binding_props(o), value_reg, binding)
            }
            PatTarget::Binding(BindingPattern::ArrayPattern(arr)) => {
                self.emit_array_pattern(array_binding_elements(arr), value_reg, binding)
            }
            PatTarget::Assign(AssignmentTarget::ArrayAssignmentTarget(arr)) => {
                self.emit_array_pattern(array_assign_elements(arr), value_reg, binding)
            }
            PatTarget::Assign(AssignmentTarget::ObjectAssignmentTarget(o)) => {
                self.emit_object_pattern(object_assign_props(o), value_reg, binding)
            }
            _ => self.err(target.span(), "destructuring target"),
        }
    }

    // -- property stores ------------------------------------------------------

    fn emit_property_store(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { obj, name_idx, .. } => {
                let feedback = self.b.new_feedback();
                self.b.store_named_property(*obj, *name_idx, feedback);
            }
            StoreTarget::Keyed { obj, key } => {
                let feedback = self.b.new_feedback();
                self.b.store_keyed_property(*obj, *key, feedback);
            }
            StoreTarget::PrivateKeyed { obj, key: _ } => {
                // (obj, key, value): obj rides the top slot, the key one
                // above the pre-staged obj was staged by the target
                // preparation, the value is the accumulator
                let key = Reg::new(obj.index() + 1);
                let mark = self.b.temp_depth();
                let base = self.b.reserve_temps(3);
                self.b.store(base);
                self.b.load(key);
                self.b.store(Reg::new(base.index() + 1));
                self.b.load(*obj);
                self.b.store(Reg::new(base.index() + 2));
                self.b
                    .call_runtime(BuiltinFn::PrivateSet, RegList::new(base, 3));
                self.b.drop_temps(mark);
            }
            // runtime(home, recv, key, value = acc, semantics: shadow)
            StoreTarget::SuperNamed {
                recv,
                home,
                name_idx,
            } => {
                self.b.call_runtime_staged(
                    BuiltinFn::SuperSetProperty,
                    &[
                        RtArg::Reg(*home),
                        RtArg::Reg(*recv),
                        RtArg::Const(*name_idx),
                        RtArg::Acc,
                        RtArg::Smi(0),
                    ],
                );
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                self.b.call_runtime_staged(
                    BuiltinFn::SuperSetProperty,
                    &[
                        RtArg::Reg(*home),
                        RtArg::Reg(*recv),
                        RtArg::Reg(*key),
                        RtArg::Acc,
                        RtArg::Smi(0),
                    ],
                );
            }
        }
    }

    fn emit_property_load_of(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { obj, name_idx, .. } => {
                let feedback = self.b.new_feedback();
                self.b.load_named_property(*obj, *name_idx, feedback);
            }
            StoreTarget::Keyed { obj, key } => {
                let feedback = self.b.new_feedback();
                self.b.load_keyed_property_reg(*obj, *key, feedback);
            }
            StoreTarget::PrivateKeyed { obj, key } => {
                // (obj, key): obj rides the top slot — restaged into a
                // fresh window (the prep stages obj below the key, the
                // old layout's order)
                let mark = self.b.temp_depth();
                let base = self.b.reserve_temps(2);
                self.b.load(*key);
                self.b.store(base);
                self.b.load(*obj);
                self.b.store(Reg::new(base.index() + 1));
                self.b
                    .call_runtime(BuiltinFn::PrivateGet, RegList::new(base, 2));
                self.b.drop_temps(mark);
            }
            // runtime(home, recv, key) -> value
            StoreTarget::SuperNamed {
                recv,
                home,
                name_idx,
            } => {
                self.b.call_runtime_staged(
                    BuiltinFn::SuperGetProperty,
                    &[
                        RtArg::Reg(*home),
                        RtArg::Reg(*recv),
                        RtArg::Const(*name_idx),
                    ],
                );
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                self.b.call_runtime_staged(
                    BuiltinFn::SuperGetProperty,
                    &[RtArg::Reg(*home), RtArg::Reg(*recv), RtArg::Reg(*key)],
                );
            }
        }
    }

    fn release_store(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { obj_staged, .. } => {
                if *obj_staged {
                    self.b.drop_temp(); // obj
                }
            }
            StoreTarget::Keyed { .. } | StoreTarget::PrivateKeyed { .. } => {
                self.b.drop_temp(); // key
                self.b.drop_temp(); // obj
            }
            StoreTarget::SuperNamed { .. } => {
                self.b.drop_temp(); // home
                self.b.drop_temp(); // recv
            }
            StoreTarget::SuperKeyed { .. } => {
                self.b.drop_temp(); // home
                self.b.drop_temp(); // key
                self.b.drop_temp(); // recv
            }
        }
    }

    fn prepare_property_store(&mut self, m: MemberRef<'_>) -> Result<StoreTarget, CompileError> {
        match m {
            MemberRef::Private(p) => {
                self.expr(&p.object)?;
                let obj = self.b.stage_acc();
                self.emit_private_key_load(p.field.node_id.get(), p.field.span)?;
                let k = self.b.stage_acc();
                Ok(StoreTarget::PrivateKeyed { obj, key: k })
            }
            MemberRef::Static(s) => {
                let name_idx = self.b.name(s.property.name.as_bytes());
                if let Some(obj) = self.this_operand_reg(&s.object) {
                    return Ok(StoreTarget::Named {
                        obj,
                        name_idx,
                        obj_staged: false,
                    });
                }
                self.expr(&s.object)?;
                let obj = self.b.stage_acc();
                Ok(StoreTarget::Named {
                    obj,
                    name_idx,
                    obj_staged: true,
                })
            }
            MemberRef::Computed(c) => {
                self.expr(&c.object)?;
                let obj = self.b.stage_acc();
                self.expr(&c.expression)?;
                let k = self.b.stage_acc();
                Ok(StoreTarget::Keyed { obj, key: k })
            }
        }
    }

    /// The receiver and home-object registers of a `super.x` reference
    /// (assignment target or compound-access base).
    fn prepare_super_parts(
        &mut self,
        m: MemberRef<'_>,
    ) -> Result<Option<StoreTarget>, CompileError> {
        let node = m.node_id();
        let Some(Special::Super {
            home,
            depth,
            this_owner,
            this_depth,
        }) = self.c.facts.special.get(&node).copied()
        else {
            return Ok(None);
        };
        self.emit_this_for(this_owner, this_depth);
        let recv = self.b.stage_acc();
        let (home_slot, home_depth) = match home {
            Home::Class(_, slot) => (slot, depth),
            Home::Object(_) => (0, depth),
        };
        match m {
            MemberRef::Computed(c) => {
                self.expr(&c.expression)?;
                let k = self.b.stage_acc();
                self.b.load_context_slot(home_slot, home_depth);
                let home = self.b.stage_acc();
                Ok(Some(StoreTarget::SuperKeyed { recv, home, key: k }))
            }
            MemberRef::Static(s) => {
                let name_idx = self.b.name(s.property.name.as_bytes());
                self.b.load_context_slot(home_slot, home_depth);
                let home = self.b.stage_acc();
                Ok(Some(StoreTarget::SuperNamed {
                    recv,
                    home,
                    name_idx,
                }))
            }
            MemberRef::Private(p) => {
                self.err(p.span, "private fields may not be accessed on 'super'")
            }
        }
    }

    fn prepare_super_store(&mut self, m: MemberRef<'_>) -> Result<StoreTarget, CompileError> {
        self.prepare_super_parts(m)?
            .ok_or_else(|| CompileError::new(m.span(), "super assignment target"))
    }

    fn emit_property_load(&mut self, m: MemberRef<'_>) -> Result<(), CompileError> {
        if is_super_member(m) {
            let Some(store) = self.prepare_super_parts(m)? else {
                return self.err(m.span(), "super property");
            };
            self.emit_property_load_of(&store);
            self.release_store(&store);
            return Ok(());
        }
        match m {
            MemberRef::Private(p) => {
                // PrivateGet: `obj.#x` — own private field or TypeError
                let mark = self.b.temp_depth();
                let base = self.b.reserve_temps(2);
                self.emit_private_key_load(p.field.node_id.get(), p.field.span)?;
                self.b.store(base);
                self.expr(&p.object)?;
                self.b.store(Reg::new(base.index() + 1));
                self.b
                    .call_runtime(BuiltinFn::PrivateGet, RegList::new(base, 2));
                self.b.drop_temps(mark);
                Ok(())
            }
            MemberRef::Computed(c) => {
                let feedback = self.b.new_feedback();
                if let (Some(obj), Some(key)) = (
                    self.simple_register(&c.object),
                    self.simple_register(&c.expression),
                ) {
                    self.b.load_keyed_property_reg(obj, key, feedback);
                    return Ok(());
                }
                self.expr(&c.object)?;
                let obj = self.b.stage_acc();
                if let Some(key) = self.simple_register(&c.expression) {
                    self.b.load_keyed_property_reg(obj, key, feedback);
                } else {
                    self.expr(&c.expression)?;
                    self.b.load_keyed_property(obj, feedback);
                }
                self.b.drop_temp();
                Ok(())
            }
            MemberRef::Static(s) => {
                let name_idx = self.b.name(s.property.name.as_bytes());
                let feedback = self.b.new_feedback();
                if let Some(obj) = self.this_operand_reg(&s.object) {
                    self.b.load_named_property(obj, name_idx, feedback);
                    return Ok(());
                }
                if let Some(obj) = self.simple_register(&s.object) {
                    self.b.load_named_property(obj, name_idx, feedback);
                    return Ok(());
                }
                self.expr(&s.object)?;
                let obj = self.b.stage_acc();
                self.b.load_named_property(obj, name_idx, feedback);
                self.b.drop_temp();
                Ok(())
            }
        }
    }

    // -- calls -----------------------------------------------------------------

    fn call_argument(&mut self, arg: &Argument<'_>) -> Result<(), CompileError> {
        match arg {
            Argument::SpreadElement(_) => self.err(arg.span(), "spread in calls"),
            other => self.expr(other.as_expression().expect("spread handled")),
        }
    }

    /// Stage a method-call window: `[recv, args..., callee]` with the
    /// receiver already staged and the callee in the accumulator.
    /// Returns `(callee, args_base, argc + 1, base_mark)` — callers drop
    /// back to `base_mark` (the depth before the window) after the call.
    /// A method call `recv.<method>(args...)`: `load_method` evaluates and
    /// stages the receiver, loads the method into the accumulator, and
    /// returns the receiver's register. The callee then sits in a fixed
    /// slot above the contiguous `[recv, args...]` window.
    fn emit_method_call(
        &mut self,
        c: &CallExpression<'_>,
        load_method: impl FnOnce(&mut Self) -> Result<Reg, CompileError>,
    ) -> Result<(), CompileError> {
        let argc = c.arguments.len();
        // zero to two arguments: the call opcode takes callee, receiver and
        // arguments as independent registers, so no window is built and the
        // receiver is never copied
        if argc <= 2 {
            let mark = self.b.temp_depth();
            let recv = load_method(self)?;
            let callee = self.b.stage_acc();
            let mut args = [Reg::new(0); 2];
            for (i, arg) in c.arguments.iter().enumerate() {
                // a plain local/parameter is already in a register: read it
                // in place instead of `Load`+`Store` through the
                // accumulator. The last argument needs no stability (nothing
                // evaluates after it); earlier ones must survive sibling
                // evaluation, which `stable_register` guarantees.
                let direct = match arg.as_expression() {
                    Some(inner) if i + 1 == argc => self.simple_register(inner),
                    Some(inner) => self.stable_register(inner),
                    None => None,
                };
                if let Some(reg) = direct {
                    args[i] = reg;
                } else {
                    self.call_argument(arg)?;
                    args[i] = self.b.stage_acc();
                }
            }
            match argc {
                0 => {
                    let fb = self.b.new_feedback();
                    self.b.call_method0(callee, recv, fb)
                }
                1 => {
                    let fb = self.b.new_feedback();
                    self.b.call_method1(callee, recv, args[0], fb)
                }
                _ => {
                    let fb = self.b.new_feedback();
                    self.b.call_method2(callee, recv, args[0], args[1], fb)
                }
            }
            self.b.drop_temps(mark);
            return Ok(());
        }
        let argc = argc as u32;
        let mark = self.b.temp_depth();
        let recv = load_method(self)?;
        // the window is [args_base .. args_base+argc]: element 0 (the
        // receiver) rides the top slot, argument i sits argc-1-i slots in,
        // the callee rides above
        let args_base = self.b.reserve_temps(argc + 2);
        let top = args_base.index() + argc as i32;
        let callee = Reg::new(top + 1);
        self.b.store(callee);
        for (i, arg) in c.arguments.iter().enumerate() {
            let direct = match arg.as_expression() {
                Some(inner) if i + 1 == argc as usize => self.simple_register(inner),
                Some(inner) => self.stable_register(inner),
                None => None,
            };
            if let Some(reg) = direct {
                self.b.move_reg(Reg::new(top - 1 - i as i32), reg);
            } else {
                self.call_argument(arg)?;
                self.b.store(Reg::new(top - 1 - i as i32));
            }
        }
        self.b.move_reg(Reg::new(top), recv);
        let fb = self.b.new_feedback();
        self.b.call(callee, RegList::new(args_base, argc + 1), fb);
        self.b.drop_temps(mark);
        Ok(())
    }

    fn emit_call(&mut self, c: &CallExpression<'_>) -> Result<(), CompileError> {
        // super.m(...): the method comes from the home-object chain and
        // runs with the current `this`
        if let Some(m) = MemberRef::of(&c.callee)
            && is_super_member(m)
        {
            return self.emit_super_method_call(c, m);
        }
        if let Expression::Super(_) = &c.callee {
            return self.emit_super_call(c);
        }
        // method calls: the receiver is the first register of the argument
        // list (evaluation order: receiver, property get, then arguments)
        if let Expression::StaticMemberExpression(s) = &c.callee {
            let name_idx = self.b.name(s.property.name.as_bytes());
            return self.emit_method_call(c, |g| {
                let recv = match g.stable_register(&s.object) {
                    Some(r) => r,
                    None => {
                        g.expr(&s.object)?;
                        g.b.stage_acc()
                    }
                };
                let feedback = g.b.new_feedback();
                g.b.load_named_property(recv, name_idx, feedback);
                Ok(recv)
            });
        }
        if let Expression::ComputedMemberExpression(cm) = &c.callee {
            return self.emit_method_call(c, |g| {
                let recv = match g.stable_register(&cm.object) {
                    Some(r) => r,
                    None => {
                        g.expr(&cm.object)?;
                        g.b.stage_acc()
                    }
                };
                g.expr(&cm.expression)?;
                let feedback = g.b.new_feedback();
                g.b.load_keyed_property(recv, feedback);
                Ok(recv)
            });
        }
        if let Expression::PrivateFieldExpression(p) = &c.callee {
            return self.err(p.span, "calling a private method");
        }
        // plain call: receiver = undefined (slot 0), callee evaluated
        // first, then arguments
        let argc = c.arguments.len();
        if argc <= 2 {
            let mark = self.b.temp_depth();
            self.expr(&c.callee)?;
            let callee = self.b.stage_acc();
            let mut args = [Reg::new(0); 2];
            for (i, arg) in c.arguments.iter().enumerate() {
                // see `emit_method_call`: read a plain local/parameter
                // argument in place rather than copying through the
                // accumulator.
                let direct = match arg.as_expression() {
                    Some(inner) if i + 1 == argc => self.simple_register(inner),
                    Some(inner) => self.stable_register(inner),
                    None => None,
                };
                if let Some(reg) = direct {
                    args[i] = reg;
                } else {
                    self.call_argument(arg)?;
                    args[i] = self.b.stage_acc();
                }
            }
            match argc {
                0 => {
                    let fb = self.b.new_feedback();
                    self.b.call_function0(callee, fb)
                }
                1 => {
                    let fb = self.b.new_feedback();
                    self.b.call_function1(callee, args[0], fb)
                }
                _ => {
                    let fb = self.b.new_feedback();
                    self.b.call_function2(callee, args[0], args[1], fb)
                }
            }
            self.b.drop_temps(mark);
            return Ok(());
        }
        let argc = argc as u32;
        let mark = self.b.temp_depth();
        self.expr(&c.callee)?;
        let args_base = self.b.reserve_temps(argc + 2);
        let top = args_base.index() + argc as i32;
        let callee = Reg::new(top + 1);
        self.b.store(callee);
        self.b.load_undefined();
        self.b.store(Reg::new(top));
        for (i, arg) in c.arguments.iter().enumerate() {
            let direct = match arg.as_expression() {
                Some(inner) if i + 1 == argc as usize => self.simple_register(inner),
                Some(inner) => self.stable_register(inner),
                None => None,
            };
            if let Some(reg) = direct {
                self.b.move_reg(Reg::new(top - 1 - i as i32), reg);
            } else {
                self.call_argument(arg)?;
                self.b.store(Reg::new(top - 1 - i as i32));
            }
        }
        let fb = self.b.new_feedback();
        self.b.call(callee, RegList::new(args_base, argc + 1), fb);
        self.b.drop_temps(mark);
        Ok(())
    }

    fn emit_super_method_call(
        &mut self,
        c: &CallExpression<'_>,
        m: MemberRef<'_>,
    ) -> Result<(), CompileError> {
        self.emit_method_call(c, |g| {
            let Some(store) = g.prepare_super_parts(m)? else {
                return Err(CompileError::new(c.span, "super method call"));
            };
            g.emit_property_load_of(&store);
            let (recv, dead_parts) = match &store {
                StoreTarget::SuperNamed { recv, .. } => (*recv, 1),
                StoreTarget::SuperKeyed { recv, .. } => (*recv, 2),
                _ => unreachable!("super store parts"),
            };
            // the home/key parts above the receiver are dead once the
            // method is loaded; dropping them keeps the argument window
            // contiguous with the receiver
            for _ in 0..dead_parts {
                g.b.drop_temp();
            }
            Ok(recv)
        })
    }

    fn emit_new(&mut self, n: &NewExpression<'_>) -> Result<(), CompileError> {
        let argc = n.arguments.len() as u32;
        let mark = self.b.temp_depth();
        self.expr(&n.callee)?;
        // window [args...] with argument 0 at the top slot, callee above
        let args_base = self.b.reserve_temps(argc + 1);
        let top = args_base.index() + argc as i32 - 1;
        let callee = Reg::new(top + 1);
        self.b.store(callee);
        for (i, arg) in n.arguments.iter().enumerate() {
            // like the call paths: a plain local/parameter is already in a
            // register, so move it directly instead of routing through the
            // accumulator; earlier arguments must survive sibling
            // evaluation (`stable_register`), the last needs no stability.
            let direct = match arg.as_expression() {
                Some(inner) if i + 1 == argc as usize => self.simple_register(inner),
                Some(inner) => self.stable_register(inner),
                None => None,
            };
            if let Some(reg) = direct {
                self.b.move_reg(Reg::new(top - i as i32), reg);
            } else {
                self.call_argument(arg)?;
                self.b.store(Reg::new(top - i as i32));
            }
        }
        self.b.construct(callee, RegList::new(args_base, argc));
        self.b.drop_temps(mark);
        Ok(())
    }

    /// `super(...)`: construct the superclass with the constructor's
    /// new.target and initialize `this` with the result (ES 15.4.3).
    fn emit_super_call(&mut self, c: &CallExpression<'_>) -> Result<(), CompileError> {
        let argc = c.arguments.len() as u32;
        let (owner, owner_depth) = match self.c.facts.special.get(&c.node_id.get()) {
            Some(Special::SuperCall { owner, depth }) => (*owner, *depth),
            _ => (self.fid, 0),
        };
        let direct = owner == self.fid;
        let this_slot = if direct {
            self.c.layouts[self.fid.0 as usize].this_slot
        } else {
            Some(
                self.c.layouts[owner.0 as usize]
                    .this_slot
                    .expect("delegated super() forces the this slot"),
            )
        };

        let mark = self.b.temp_depth();
        // reserve arguments + result (+ closure/new.target registers for
        // the delegated variant) so nested temps land above
        let reserved = argc + 1 + u32::from(!direct) * 2;
        let arg_base = self.b.reserve_temps(reserved);
        // the delegated window is (args..., closure, new_target): args
        // ride the top, closure/new.target the two bottom slots
        let arg_top = arg_base.index() + argc as i32 + i32::from(!direct) * 2 - 1;
        for (i, arg) in c.arguments.iter().enumerate() {
            let direct_reg = match arg.as_expression() {
                Some(inner) if i + 1 == argc as usize => self.simple_register(inner),
                Some(inner) => self.stable_register(inner),
                None => None,
            };
            if let Some(reg) = direct_reg {
                self.b.move_reg(Reg::new(arg_top - i as i32), reg);
            } else {
                self.call_argument(arg)?;
                self.b.store(Reg::new(arg_top - i as i32));
            }
        }
        if direct {
            self.b
                .call_runtime(BuiltinFn::ConstructSuper, RegList::new(arg_base, argc));
        } else {
            // .this_function and .new.target of the owning constructor
            // (runtime ABI: (args..., closure, new_target))
            let (this_function_slot, new_target_slot) = {
                let l = &self.c.layouts[owner.0 as usize];
                (
                    l.this_function_slot
                        .expect("delegated super() forces the closure slot"),
                    l.new_target_slot
                        .expect("delegated super() forces the new.target slot"),
                )
            };
            let new_target_reg = arg_base;
            let closure_reg = Reg::new(arg_base.index() + 1);
            self.b.load_context_slot(this_function_slot, owner_depth);
            self.b.store(closure_reg);
            self.b.load_context_slot(new_target_slot, owner_depth);
            self.b.store(new_target_reg);
            self.b.call_runtime(
                BuiltinFn::ConstructSuperVia,
                RegList::new(arg_base, argc + 2),
            );
        }
        // the constructed instance lands above the (contiguous) runtime
        // argument window
        let result = Reg::new(arg_base.index() + argc as i32 + i32::from(!direct) * 2);
        self.b.store(result);
        // InitializeThisBinding: this must still be uninitialized
        let super_once_check = |g: &mut Self| {
            let mark = g.b.temp_depth();
            let t = g.b.stage_acc();
            g.b.call_runtime(
                BuiltinFn::ThrowSuperAlreadyCalledIfNotHole,
                RegList::new(t, 1),
            );
            g.b.drop_temps(mark);
        };
        match this_slot {
            Some(slot) => {
                let depth = if direct { 0 } else { owner_depth };
                self.b.load_context_slot(slot, depth);
                super_once_check(self);
                self.b.load(result);
                self.b.store_context_slot(slot, depth);
            }
            None => {
                let this = self.b.this_reg();
                self.b.load(this);
                super_once_check(self);
                self.b.load(result);
                self.b.store(this);
            }
        }
        // InitializeInstanceElements (ES 7.3.33): the derived
        // constructor's own fields are defined on the freshly bound
        // instance (for arrow-delegated super(), the owner is the ctor)
        let field_owner = if direct { self.fid } else { owner };
        if self.fn_has_instance_fields(field_owner) {
            // (ctor, instance): ctor rides the top slot
            self.b.load(result);
            let instance = self.b.stage_acc();
            let ctor = self.b.temp();
            if direct {
                self.b.load_current_closure();
            } else {
                let slot = self.c.layouts[owner.0 as usize]
                    .this_function_slot
                    .expect("delegated super() forces the closure slot");
                self.b.load_context_slot(slot, owner_depth);
            }
            self.b.store(ctor);
            self.b
                .call_runtime(BuiltinFn::InitInstanceFields, RegList::new(instance, 2));
            self.b.drop_temp(); // ctor
            self.b.drop_temp(); // instance
        }
        self.b.load(result);
        self.b.drop_temps(mark);
        Ok(())
    }

    // -- classes ------------------------------------------------------------------

    /// Class definitions (ES 15.7.14 ClassDefinitionEvaluation) as inline
    /// per-member emission. Leaves the class constructor in the
    /// accumulator.
    fn emit_class(&mut self, idx: ClassIdx) -> Result<(), CompileError> {
        let slot_count = self.c.facts.classes[idx.0 as usize].slot_count;
        let ctor = self.c.facts.classes[idx.0 as usize].ctor;
        let uses_super = self.c.facts.classes[idx.0 as usize].uses_super;
        let name_slot = self.c.facts.classes[idx.0 as usize].name_slot;
        let is_decl = self.c.facts.classes[idx.0 as usize].is_decl;
        let superclass = self.c.facts.classes[idx.0 as usize].superclass;
        let has_instance_fields = self.c.facts.classes[idx.0 as usize].has_instance_fields;
        let member_count = self.c.facts.classes[idx.0 as usize].members.len();
        let decl_symbol = self.c.facts.classes[idx.0 as usize].decl_symbol.clone();

        // class inner context: the name binding (TDZ until the class value
        // exists) and the super home objects; captured by member closures.
        // Pushed before the superclass evaluation (ES 15.7.14 step 8).
        let ctx_save = if slot_count > 0 {
            self.b.create_block_context(slot_count);
            let save = self.b.temp();
            self.b.push_context(save);
            Some(save)
        } else {
            None
        };

        // private names: one fresh Symbol per private name per class
        // evaluation, stored into the class context before any element
        // evaluates (methods and initializers reference them by slot)
        let privates = self.c.facts.classes[idx.0 as usize].privates.clone();
        let private_slots = self.c.facts.classes[idx.0 as usize].private_slots.clone();
        for (name, slot) in privates.iter().zip(&private_slots) {
            let desc = self.b.name(name.as_bytes());
            self.b.load_constant(desc);
            let desc_reg = self.b.stage_acc();
            self.b
                .call_runtime(BuiltinFn::CreatePrivateName, RegList::new(desc_reg, 1));
            self.b.drop_temp();
            self.b.store_context_slot(*slot, 0);
        }

        // superclass: must be null or a constructor
        let sup = if let Some(sup) = superclass {
            self.expr(sup)?;
            let sup = self.b.stage_acc();
            self.b
                .call_runtime_staged(BuiltinFn::ThrowIfNotConstructorOrNull, &[RtArg::Reg(sup)]);
            Some(sup)
        } else {
            None
        };

        // prototype/constructor parents; defaults are the no-extends case
        let obj_proto = self.b.constant(Constant::ObjectPrototype);
        self.b.load_constant(obj_proto);
        let pp = self.b.stage_acc();
        let fn_proto = self.b.constant(Constant::FunctionPrototype);
        self.b.load_constant(fn_proto);
        let cp = self.b.stage_acc();
        if let Some(sup) = sup {
            // null superclass → null-proto prototype; ctor parent stays
            // %Function.prototype% (objects are always truthy, so the
            // falsy test identifies null)
            self.b.load(sup);
            let null_extends = self.b.new_label();
            self.b.jump_if_falsy(null_extends);
            // protoParent = Get(superCtor, "prototype") (full [[Get]])
            let proto_name = self.b.name(b"prototype");
            let feedback = self.b.new_feedback();
            self.b.load_named_property(sup, proto_name, feedback);
            self.b.store(pp);
            self.b
                .call_runtime_staged(BuiltinFn::ThrowIfNotObjectOrNull, &[RtArg::Reg(pp)]);
            self.b.load(sup);
            self.b.store(cp);
            let done = self.b.new_label();
            self.b.jump(done);
            self.b.bind(null_extends);
            self.b.load_null();
            self.b.store(pp);
            self.b.bind(done);
        }

        // prototype: a fresh ordinary object with protoParent
        self.b.create_empty_object_literal();
        let proto = self.b.stage_acc();
        self.b.call_runtime_staged(
            BuiltinFn::SetPrototype,
            &[RtArg::Reg(proto), RtArg::Reg(pp)],
        );

        // constructor closure
        let ctor_idx = self.b.constant(Constant::Callable(FunctionId(ctor.0)));
        self.b.create_closure(ctor_idx);
        let ctor = self.b.stage_acc();

        // wiring before member installation (ES 15.7.14 steps 17–18
        // precede the element loop): computed `['constructor']` members
        // overwrite proto.constructor, computed static `['prototype']`
        // defines fail against the non-configurable ctor.prototype
        // proto.constructor → the class {w+, e−, c+}
        let ctor_name = self.b.name(b"constructor");
        self.b.call_runtime_staged(
            BuiltinFn::DefineOwnProperty,
            &[
                RtArg::Reg(proto),
                RtArg::Const(ctor_name),
                RtArg::Reg(ctor),
                RtArg::Smi(PropertyFlags::DontEnum.bits()),
            ],
        );
        // ctor.prototype → the prototype {w+, e−, c−}
        let proto_name = self.b.name(b"prototype");
        self.b.call_runtime_staged(
            BuiltinFn::DefineOwnProperty,
            &[
                RtArg::Reg(ctor),
                RtArg::Const(proto_name),
                RtArg::Reg(proto),
                RtArg::Smi(PropertyFlags::DontEnum.bits() | PropertyFlags::DontDelete.bits()),
            ],
        );
        // the class itself inherits from the superclass constructor
        self.b
            .call_runtime_staged(BuiltinFn::SetPrototype, &[RtArg::Reg(ctor), RtArg::Reg(cp)]);

        // instance field list: a JS array [key0, init0, key1, init1, ...]
        // attached to the constructor (its hidden fields slot)
        let fields_arr = if has_instance_fields {
            self.b.create_empty_array_literal();
            Some(self.b.stage_acc())
        } else {
            None
        };
        let mut field_index = 0u32;
        // static field keys (evaluated in element order) stay live until
        // the deferred initializer calls after the class is complete
        // (closure register, key register, constant-key name index)
        let mut static_fields: Vec<(Reg, Option<Reg>, Option<ConstIdx>)> = Vec::new();

        // members in declaration order; the constructor is already installed
        for mi in 0..member_count {
            let (kind, is_static, computed, fid, key) = {
                let m = &self.c.facts.classes[idx.0 as usize].members[mi];
                (m.kind, m.is_static, m.computed, m.fid, m.key)
            };
            if kind == MemberKind::Field {
                // field key: evaluated in element order (ES 15.7.14 step 27)
                let key_reg = if matches!(key, PropertyKey::PrivateIdentifier(_)) {
                    if let PropertyKey::PrivateIdentifier(p) = key {
                        self.emit_private_key_load(p.node_id.get(), p.span)?;
                    }
                    Some(self.b.stage_acc())
                } else {
                    self.stage_key(key, computed)?
                };
                let fn_idx = self.b.constant(Constant::Callable(FunctionId(fid.0)));
                self.b.create_closure(fn_idx);
                if is_static {
                    let closure = self.b.stage_acc();
                    let name_idx = match key_reg {
                        Some(_) => None,
                        None => Some(self.name_constant(key)?),
                    };
                    static_fields.push((closure, key_reg, name_idx));
                } else {
                    // append [key, initializer] to the instance field list
                    let arr = fields_arr.expect("instance fields allocate the list");
                    let closure = self.b.stage_acc();
                    self.b.load_smi(field_index as i32);
                    let i = self.b.stage_acc();
                    match key_reg {
                        Some(k) => {
                            self.b.load(k);
                            let feedback = self.b.new_feedback();
                            self.b.store_keyed_property_no_shadow(arr, i, feedback);
                        }
                        None => {
                            let name_idx = self.name_constant(key)?;
                            self.b.load_constant(name_idx);
                            let feedback = self.b.new_feedback();
                            self.b.store_keyed_property_no_shadow(arr, i, feedback);
                        }
                    }
                    // initializer at the next slot
                    self.b.load_smi(field_index as i32 + 1);
                    self.b.store(i);
                    self.b.load(closure);
                    let feedback = self.b.new_feedback();
                    self.b.store_keyed_property_no_shadow(arr, i, feedback);
                    self.b.drop_temp(); // i
                    self.b.drop_temp(); // closure
                    field_index += 2;
                    if key_reg.is_some() {
                        self.b.drop_temp();
                    }
                }
                continue;
            }
            let target = if is_static { ctor } else { proto };
            // non-computed keys are strings or numbers; numbers go through
            // the keyed define with the literal loaded into a register
            let key_reg = self.stage_key(key, computed)?;
            let fn_idx = self.b.constant(Constant::Callable(FunctionId(fid.0)));
            self.b.create_closure(fn_idx);
            // computed keys name the member after their ToPropertyKey value
            if let Some(k) = key_reg {
                let prefix = match kind {
                    MemberKind::Get => 1,
                    MemberKind::Set => 2,
                    _ => 0,
                };
                self.emit_set_name_by_reg(k, prefix);
            }
            let name_idx = match key_reg {
                Some(_) => None,
                None => Some(self.name_constant(key)?),
            };
            match kind {
                MemberKind::Method => {
                    // {w+, e−, c+}: runtime(obj, key, value = acc, flags)
                    let key = match key_reg {
                        Some(k) => RtArg::Reg(k),
                        None => RtArg::Const(name_idx.unwrap()),
                    };
                    self.b.call_runtime_staged(
                        BuiltinFn::DefineOwnProperty,
                        &[
                            RtArg::Reg(target),
                            key,
                            RtArg::Acc,
                            RtArg::Smi(PropertyFlags::DontEnum.bits()),
                        ],
                    );
                }
                MemberKind::Get | MemberKind::Set => {
                    // one accessor half; merges with an existing pair:
                    // runtime(target, key, closure = acc, flags)
                    let mut flags = PropertyFlags::DontEnum.bits();
                    if kind == MemberKind::Get {
                        flags |= 1;
                    }
                    let key = match key_reg {
                        Some(k) => RtArg::Reg(k),
                        None => RtArg::Const(name_idx.unwrap()),
                    };
                    self.b.call_runtime_staged(
                        BuiltinFn::InstallAccessor,
                        &[RtArg::Reg(target), key, RtArg::Acc, RtArg::Smi(flags)],
                    );
                }
                MemberKind::Field => unreachable!("fields handled above"),
            }
            if key_reg.is_some() {
                self.b.drop_temp();
            }
        }

        // home objects and the inner class-name binding
        if uses_super {
            let (home_slot, static_home_slot) = {
                let c = &self.c.facts.classes[idx.0 as usize];
                (c.home_slot.unwrap(), c.static_home_slot.unwrap())
            };
            // the class context is pushed here: zero hops
            self.b.load(proto);
            self.b.store_context_slot(home_slot, 0);
            self.b.load(ctor);
            self.b.store_context_slot(static_home_slot, 0);
        }
        if let Some(slot) = name_slot {
            self.b.load(ctor);
            self.b.store_context_slot(slot, 0);
        }

        // static fields: each initializer runs with the constructor as
        // `this` and its result is [[DefineOwnProperty]]'d on it, in
        // declaration order (ES 15.7.14 step 33)
        for (closure, key_reg, name_idx) in &static_fields {
            self.b.load(*closure);
            let fb = self.b.new_feedback();
            self.b.call(*closure, RegList::new(ctor, 1), fb);
            // runtime(obj = ctor, key, value = acc, flags 0)
            let key = match (key_reg, name_idx) {
                (Some(k), _) => RtArg::Reg(*k),
                (None, Some(name_idx)) => RtArg::Const(*name_idx),
                (None, None) => unreachable!("static field keys are reg or const"),
            };
            self.b.call_runtime_staged(
                BuiltinFn::DefineOwnProperty,
                &[RtArg::Reg(ctor), key, RtArg::Acc, RtArg::Smi(0)],
            );
        }
        // LIFO pops for the static field registers (key under closure)
        for (_closure, key_reg, _) in static_fields.iter().rev() {
            self.b.drop_temp(); // closure
            if key_reg.is_some() {
                self.b.drop_temp(); // key
            }
        }

        // attach the instance field list to the constructor
        if let Some(arr) = fields_arr {
            // (ctor, fields): ctor rides the top slot
            self.b.load(arr);
            let arr_reg = self.b.stage_acc();
            let ctor_reg = self.b.temp();
            self.b.load(ctor);
            self.b.store(ctor_reg);
            self.b
                .call_runtime(BuiltinFn::SetClassFields, RegList::new(arr_reg, 2));
            self.b.drop_temp(); // ctor copy
            self.b.drop_temp(); // arr copy
            self.b.drop_temp(); // the fields array itself
        }

        // pop the class context; member closures already captured it
        if let Some(save) = ctx_save {
            self.b.pop_context(save);
            self.b.drop_temp(); // save
        }

        // outer binding for declarations (the class value in acc)
        self.b.load(ctor);
        if is_decl && let Some((sym, name)) = decl_symbol {
            self.store_symbol(sym, &name)?;
        }

        // LIFO: ctor, proto, cp, pp, superclass
        self.b.drop_temp(); // ctor
        self.b.drop_temp(); // proto
        self.b.drop_temp(); // cp
        self.b.drop_temp(); // pp
        if sup.is_some() {
            self.b.drop_temp();
        }
        self.b.load(ctor);
        Ok(())
    }

    // -- literals -----------------------------------------------------------------

    fn emit_array_literal(&mut self, a: &ArrayExpression<'_>) -> Result<(), CompileError> {
        if a.elements.is_empty() {
            self.b.create_empty_array_literal();
        } else {
            // pre-size the backing store with the store-growth policy's
            // slack: element stores then append within headroom
            // (IC-fast, no reallocation)
            let len = a.elements.len();
            let capacity = len + (len >> 1) + 8;
            self.b.create_sized_array_literal(capacity as u32);
        }
        let arr = self.b.stage_acc();
        let mut i = 0u32;
        for el in &a.elements {
            match el {
                // an elision is a hole: it consumes an index but stores
                // nothing (the gap grow/holes below fill it in)
                ArrayExpressionElement::Elision(_) => i += 1,
                ArrayExpressionElement::SpreadElement(_) => {
                    return self.err(a.span, "spread in array literals");
                }
                other => {
                    let x = other.as_expression().expect("elision/spread handled");
                    self.b.load_smi(i as i32);
                    let idx = self.b.stage_acc();
                    self.expr(x)?;
                    let feedback = self.b.new_feedback();
                    self.b.store_keyed_property_no_shadow(arr, idx, feedback);
                    self.b.drop_temp();
                    i += 1;
                }
            }
        }
        // trailing elisions raise the length past the last store: the
        // elements store cannot do it (nothing is stored), so set it
        // explicitly (`length` is the array accessor)
        if matches!(a.elements.last(), Some(ArrayExpressionElement::Elision(_))) {
            self.b.load_smi(a.elements.len() as i32);
            let name = self.b.name(b"length");
            let feedback = self.b.new_feedback();
            self.b.store_named_property(arr, name, feedback);
        }
        self.b.load(arr);
        self.b.drop_temp();
        Ok(())
    }

    fn emit_object_literal(&mut self, o: &ObjectExpression<'_>) -> Result<(), CompileError> {
        // methods using `super` capture a per-literal home-object context
        // (the literal itself), mirroring class scopes
        let node = o.node_id.get();
        let needs_home = self.c.facts.obj_lit_home.contains(&node);
        let ctx_save = if needs_home {
            self.b.create_block_context(1);
            let save = self.b.temp();
            self.b.push_context(save);
            Some(save)
        } else {
            None
        };

        self.b.create_empty_object_literal();
        let obj = self.b.stage_acc();
        for p in &o.properties {
            let ObjectPropertyKind::ObjectProperty(p) = p else {
                return self.err(o.span, "spread in object literals");
            };
            // numeric literal keys ({ 1: x }) use the keyed path
            let key_reg = self.stage_key(&p.key, p.computed)?;
            let is_method = p.method || matches!(p.value, Expression::FunctionExpression(_));
            match p.kind {
                PropertyKind::Init if !is_method => {
                    self.expr(&p.value)?;
                    match key_reg {
                        None => {
                            // NamedEvaluation: { m: function () {} }
                            let name_idx = self.name_constant(&p.key)?;
                            if self.is_anon_function(&p.value) {
                                self.emit_set_name_by_const(name_idx);
                            }
                            let feedback = self.b.new_feedback();
                            self.b.store_named_property(obj, name_idx, feedback);
                        }
                        Some(k) => {
                            if self.is_anon_function(&p.value) {
                                self.emit_set_name_by_reg(k, 0);
                            }
                            let feedback = self.b.new_feedback();
                            self.b.store_keyed_property(obj, k, feedback);
                        }
                    }
                }
                PropertyKind::Init => {
                    // method shorthand `{ foo() {} }`: the closure is named
                    // after its key
                    self.expr(&p.value)?;
                    if let Some(k) = key_reg {
                        self.emit_set_name_by_reg(k, 0);
                    }
                    if let Some(k) = key_reg {
                        let feedback = self.b.new_feedback();
                        self.b.store_keyed_property(obj, k, feedback);
                    } else {
                        let name_idx = self.name_constant(&p.key)?;
                        let feedback = self.b.new_feedback();
                        self.b.store_named_property(obj, name_idx, feedback);
                    }
                }
                PropertyKind::Get | PropertyKind::Set => {
                    // enumerable accessor halves, merged pairs; bit 0 marks
                    // the getter half
                    let prefix = match p.kind {
                        PropertyKind::Get => 1u32,
                        _ => 2,
                    };
                    self.expr(&p.value)?;
                    if let Some(k) = key_reg {
                        self.emit_set_name_by_reg(k, prefix);
                    }
                    let flags: u32 = match p.kind {
                        PropertyKind::Get => 1,
                        _ => 0,
                    };
                    let name_idx = match key_reg {
                        Some(_) => None,
                        None => Some(self.name_constant(&p.key)?),
                    };
                    let key = match key_reg {
                        Some(k) => RtArg::Reg(k),
                        None => RtArg::Const(name_idx.unwrap()),
                    };
                    self.b.call_runtime_staged(
                        BuiltinFn::InstallAccessor,
                        &[RtArg::Reg(obj), key, RtArg::Acc, RtArg::Smi(flags)],
                    );
                }
            }
            if key_reg.is_some() {
                self.b.drop_temp();
            }
        }
        // the home object is the literal itself; methods already captured
        // the context, so the store is visible to them
        if needs_home {
            self.b.load(obj);
            self.b.store_context_slot(0, 0);
        }
        if let Some(save) = ctx_save {
            self.b.pop_context(save);
            self.b.drop_temp();
        }
        self.b.load(obj);
        self.b.drop_temp();
        Ok(())
    }
}

/// Split a binding pattern's optional default: `{a = 1}` wraps the target
/// in an AssignmentPattern.
fn split_binding_default<'r, 'a>(
    p: &'r BindingPattern<'a>,
) -> (PatTarget<'r, 'a>, Option<&'r Expression<'a>>) {
    match p {
        BindingPattern::AssignmentPattern(a) => (PatTarget::Binding(&a.left), Some(&a.right)),
        other => (PatTarget::Binding(other), None),
    }
}

// ---------------------------------------------------------------------------
// statements + function bodies
// ---------------------------------------------------------------------------

fn callable_kind(kind: FnKind) -> CallableKind {
    match kind {
        FnKind::Script | FnKind::Normal => CallableKind::Normal,
        FnKind::Arrow => CallableKind::Arrow,
        FnKind::Method => CallableKind::Method,
        FnKind::Getter => CallableKind::Getter,
        FnKind::Setter => CallableKind::Setter,
        FnKind::BaseClassCtor => CallableKind::BaseClassConstructor,
        FnKind::DerivedClassCtor => CallableKind::DerivedClassConstructor,
        FnKind::DefaultDerivedCtor => CallableKind::DefaultDerivedConstructor,
    }
}

impl<'c, 'a, 'p> FunctionGen<'c, 'a, 'p> {
    fn stmt(&mut self, stmt: &Statement<'_>) -> Result<(), CompileError> {
        match stmt {
            Statement::ExpressionStatement(s) => {
                self.expr(&s.expression)?;
                // a script-level value-producing statement records the
                // completion value (non-value statements leave it)
                if let Some(completion) = self.completion {
                    self.b.store(completion);
                }
                Ok(())
            }
            Statement::VariableDeclaration(d) => self.emit_var_decl(d),
            Statement::BlockStatement(b) => {
                for s in &b.body {
                    self.stmt(s)?;
                }
                Ok(())
            }
            Statement::IfStatement(s) => {
                self.expr(&s.test)?;
                let else_l = self.b.new_label();
                self.b.jump_if_falsy(else_l);
                self.stmt(&s.consequent)?;
                match &s.alternate {
                    Some(e) => {
                        let end = self.b.new_label();
                        self.b.jump(end);
                        self.b.bind(else_l);
                        self.stmt(e)?;
                        self.b.bind(end);
                    }
                    None => {
                        self.b.bind(else_l);
                    }
                }
                Ok(())
            }
            Statement::WhileStatement(s) => {
                let labels = self.take_labels();
                self.emit_while(&s.test, &s.body, labels)
            }
            Statement::DoWhileStatement(s) => {
                let labels = self.take_labels();
                self.emit_do_while(&s.test, &s.body, labels)
            }
            Statement::ForStatement(s) => self.emit_for(s),
            Statement::ForInStatement(s) => self.emit_for_in(s),
            Statement::ForOfStatement(s) => self.err(s.span, "for-of loops"),
            Statement::ReturnStatement(s) => {
                // derived constructors: `return v` returns v only when it
                // is an object; `undefined` (and fallthrough) return `this`
                // (ES 9.2.2.1)
                if self.is_derived_ctor() {
                    return self.emit_derived_return(s.argument.as_ref());
                }
                match &s.argument {
                    Some(v) => self.expr(v)?,
                    None => self.b.load_undefined(),
                }
                self.pop_frame_context();
                self.b.ret();
                Ok(())
            }
            Statement::ThrowStatement(s) => {
                self.expr(&s.argument)?;
                self.b.throw();
                Ok(())
            }
            Statement::TryStatement(s) => self.emit_try_catch(s),
            Statement::SwitchStatement(s) => self.emit_switch(s),
            Statement::FunctionDeclaration(f) => {
                // direct-body declarations were hoisted into the prologue;
                // block-level ones initialize at their position
                let id = f.id.as_ref().expect("function declaration");
                let Some(sym) = id.symbol_id.get() else {
                    return self.err(f.span, "function declaration without a binding");
                };
                let hoisted = self
                    .c
                    .facts
                    .hoist_fns
                    .get(&self.fid)
                    .is_some_and(|v| v.iter().any(|(s, _)| *s == sym));
                if !hoisted {
                    let fid = self.c.facts.fn_of_node[&f.node_id.get()];
                    let idx = self.b.constant(Constant::Callable(FunctionId(fid.0)));
                    self.b.create_closure(idx);
                    self.store_symbol(sym, id.name.as_ref())?;
                }
                Ok(())
            }
            Statement::LabeledStatement(s) => self.emit_labeled(s),
            Statement::BreakStatement(s) => {
                self.emit_break_continue(s.span, s.label.as_ref().map(|l| l.name.as_ref()), true)
            }
            Statement::ContinueStatement(s) => {
                self.emit_break_continue(s.span, s.label.as_ref().map(|l| l.name.as_ref()), false)
            }
            Statement::EmptyStatement(_) => Ok(()),
            Statement::ClassDeclaration(c) => {
                let idx = self.c.facts.class_of_node[&c.node_id.get()];
                self.emit_class(idx)
            }
            Statement::DebuggerStatement(s) => self.err(s.span, "debugger statements"),
            Statement::WithStatement(s) => self.err(s.span, "with statements"),
            _ => self.err(stmt.span(), "statement"),
        }
    }

    /// Outer labels (from enclosing LabeledStatements) forwarded to the
    /// innermost loop/switch.
    fn take_labels(&mut self) -> Vec<String> {
        let mut labels = std::mem::take(&mut self.nested_labels);
        labels.reverse(); // outermost first
        labels
    }

    /// Labeled statements around loops forward their label set into the
    /// loop emitter; anything else is a break-only breakable (ES 14.13).
    fn emit_labeled(&mut self, s: &LabeledStatement<'_>) -> Result<(), CompileError> {
        let label = s.label.name.to_string();
        match &s.body {
            Statement::WhileStatement(_)
            | Statement::DoWhileStatement(_)
            | Statement::ForStatement(_)
            | Statement::ForInStatement(_)
            | Statement::SwitchStatement(_)
            | Statement::LabeledStatement(_) => {
                self.nested_labels.push(label);
                let r = self.stmt(&s.body);
                self.nested_labels.pop();
                r
            }
            _ => {
                let breaks = self.b.new_label();
                self.breakables.push(Breakable {
                    labels: vec![label],
                    breaks,
                    continues: None,
                    unwind_ctx: None,
                });
                let result = self.stmt(&s.body);
                let (breaks, _) = self.end_breakable();
                self.b.bind(breaks);
                result
            }
        }
    }

    fn emit_var_decl(&mut self, d: &VariableDeclaration<'_>) -> Result<(), CompileError> {
        for decl in &d.declarations {
            match &decl.id {
                BindingPattern::ObjectPattern(_) | BindingPattern::ArrayPattern(_) => {
                    let Some(init) = &decl.init else {
                        return self
                            .err(decl.span, "destructuring declaration needs an initializer");
                    };
                    self.expr(init)?;
                    let value = self.b.stage_acc();
                    self.emit_binding_pattern(&decl.id, value)?;
                    self.b.drop_temp();
                }
                BindingPattern::BindingIdentifier(b) => {
                    let Some(sym) = b.symbol_id.get() else {
                        return self.err(b.span, "declaration without a binding");
                    };
                    if let Some(init) = &decl.init {
                        self.expr(init)?;
                        if self.is_anon_function(init) {
                            self.emit_set_name_const(b.name.as_bytes());
                        }
                        self.store_symbol(sym, b.name.as_ref())?;
                    }
                    // no initializer: `var` was pre-initialized to
                    // `undefined` by the hoisting prologue; let/const stay
                    // the hole (TDZ)
                }
                _ => return self.err(decl.span, "var declarator target"),
            }
        }
        Ok(())
    }

    /// `for (left in object) body` (ES 14.7.5): the subject evaluates in
    /// the head scope (lexical bindings are in TDZ there), null/undefined
    /// enumerate nothing, and the loop pulls one key per iteration from
    /// the hidden enumerator. The assignment target re-evaluates per
    /// iteration.
    fn emit_for_in(&mut self, s: &ForInStatement<'_>) -> Result<(), CompileError> {
        let node = s.node_id.get();
        let per_iteration = self.c.facts.per_iteration.contains(&node);
        let slots = self.c.facts.for_slots.get(&node).copied().unwrap_or(0);
        let labels = self.take_labels();
        self.with_temps(|g| {
            // lexical heads (`for (let/const k in …)`) own a block
            // context; captured heads get a fresh copy per iteration so
            // closures in the body capture per-iteration bindings
            let mut loop_ctx: Option<Reg> = None;
            let ctx_save = if slots > 0 {
                g.b.create_block_context(slots);
                let save = g.b.temp();
                let lc = g.b.temp();
                g.b.store(lc);
                g.b.push_context(save);
                loop_ctx = Some(lc);
                Some(save)
            } else {
                None
            };
            // head: subject → enumerator (undefined for nullish subjects;
            // ForInNext(undefined) is immediately done)
            g.expr(&s.right)?;
            let subject = g.b.stage_acc();
            g.b.call_runtime(BuiltinFn::ForInEnumerate, RegList::new(subject, 1));
            g.b.drop_temp(); // the call consumed the subject; acc = enumerator
            let enumerator = g.b.stage_acc();

            // loop: next key or undefined
            let back = g.b.new_label();
            g.b.bind(back); // loop head
            let breaks = g.b.new_label();
            let continues = g.b.new_label();
            g.breakables.push(Breakable {
                labels,
                breaks,
                continues: Some(continues),
                unwind_ctx: ctx_save,
            });
            g.b.call_runtime(BuiltinFn::ForInNext, RegList::new(enumerator, 1));
            let have_key = g.b.new_label();
            g.b.jump_if_not_undefined(have_key);
            g.b.jump(breaks);
            g.b.bind(have_key);
            let key = g.b.stage_acc();

            // per iteration with a lexical head: replace the context with
            // a fresh sibling (same outer) and initialize the binding
            // there — a fresh binding per iteration
            if per_iteration {
                let save = ctx_save.expect("per-iteration loops own a context");
                g.b.pop_context(save);
                g.b.create_block_context(slots);
                g.b.push_context(save);
            }

            // assign the key to the target (per iteration), then the body
            g.emit_for_in_assign(&s.left, key)?;
            g.stmt(&s.body)?;

            let continues = g.breakables.last().unwrap().continues.unwrap();
            g.b.bind(continues);
            // absolute restore: a labelled continue may arrive from a
            // nested construct still holding its context (per-iteration
            // heads self-heal at the top of the next iteration)
            if !per_iteration && let Some(lc) = loop_ctx {
                g.b.pop_context(lc);
            }
            g.b.jump_loop(back);
            // absolute restore: break may arrive from arbitrary context
            // depth (labelled jumps past nested loop pops)
            let (breaks, _) = g.end_breakable();
            g.b.bind(breaks);
            if let Some(save) = ctx_save {
                g.b.pop_context(save);
            }

            g.b.drop_temp(); // key
            g.b.drop_temp(); // enumerator
            Ok(())
        })
    }

    /// Store the enumeration key (in register `key`) into the for-in
    /// assignment target. Declaration heads assign their single binding
    /// (patterns destructure); expression targets are stored through the
    /// normal assignment machinery.
    fn emit_for_in_assign(
        &mut self,
        left: &ForStatementLeft<'_>,
        key: Reg,
    ) -> Result<(), CompileError> {
        match left {
            ForStatementLeft::VariableDeclaration(d) => {
                let [declarator] = d.declarations.as_slice() else {
                    return self.err(d.span, "for-in declarator");
                };
                match &declarator.id {
                    BindingPattern::BindingIdentifier(b) => {
                        self.b.load(key);
                        let Some(sym) = b.symbol_id.get() else {
                            return self.err(b.span, "for-in binding");
                        };
                        self.store_symbol(sym, b.name.as_ref())
                    }
                    BindingPattern::ObjectPattern(_) | BindingPattern::ArrayPattern(_) => {
                        self.emit_binding_pattern(&declarator.id, key)
                    }
                    _ => self.err(d.span, "for-in binding pattern"),
                }
            }
            ForStatementLeft::AssignmentTargetIdentifier(i) => {
                self.b.load(key);
                self.store_name(i)
            }
            t if assign_member_ref_for_left(t).is_some() => {
                let m = assign_member_ref_for_left(t).unwrap();
                let store = if is_super_member(m) {
                    self.prepare_super_store(m)?
                } else {
                    self.prepare_property_store(m)?
                };
                self.b.load(key);
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(left.span(), "for-in assignment target"),
        }
    }

    fn emit_while(
        &mut self,
        test: &Expression<'_>,
        body: &Statement<'_>,
        labels: Vec<String>,
    ) -> Result<(), CompileError> {
        // head: cond; JumpIfFalsy breaks; body; continues; back-edge
        let back = self.b.new_label();
        self.b.bind(back); // loop head
        self.expr(test)?;
        let breaks = self.b.new_label();
        let continues = self.b.new_label();
        self.breakables.push(Breakable {
            labels,
            breaks,
            continues: Some(continues),
            unwind_ctx: None,
        });
        self.b.jump_if_falsy(breaks);
        self.stmt(body)?;
        let continues = self.breakables.last().unwrap().continues.unwrap();
        self.b.bind(continues);
        self.b.jump_loop(back);
        let (breaks, _) = self.end_breakable();
        self.b.bind(breaks);
        Ok(())
    }

    /// `do body while (test)`: the body runs once before the first test;
    /// `continue` targets the test, and the back-edge stays a `JumpLoop`
    /// so a long loop still reaches safepoints.
    fn emit_do_while(
        &mut self,
        test: &Expression<'_>,
        body: &Statement<'_>,
        labels: Vec<String>,
    ) -> Result<(), CompileError> {
        let back = self.b.new_label();
        self.b.bind(back); // loop head, entered without a test
        let breaks = self.b.new_label();
        let continues = self.b.new_label();
        self.breakables.push(Breakable {
            labels,
            breaks,
            continues: Some(continues),
            unwind_ctx: None,
        });
        self.stmt(body)?;
        let continues = self.breakables.last().unwrap().continues.unwrap();
        self.b.bind(continues);
        self.expr(test)?;
        self.b.jump_if_falsy(breaks);
        self.b.jump_loop(back);
        let (breaks, _) = self.end_breakable();
        self.b.bind(breaks);
        Ok(())
    }

    /// `for (init; cond; next) body`: lexical heads own a block context;
    /// captured bindings require a fresh environment per iteration with
    /// the values copied forward (ES 14.7.5.4).
    fn emit_for(&mut self, s: &ForStatement<'_>) -> Result<(), CompileError> {
        let node = s.node_id.get();
        let per_iteration = self.c.facts.per_iteration.contains(&node);
        let slots = self.c.facts.for_slots.get(&node).copied().unwrap_or(0);
        let labels = self.take_labels();
        self.with_temps(|g| {
            let mut loop_ctx: Option<Reg> = None;
            let ctx_save = if slots > 0 {
                g.b.create_block_context(slots);
                let save = g.b.temp();
                let lc = g.b.temp();
                g.b.store(lc);
                g.b.push_context(save);
                loop_ctx = Some(lc);
                Some(save)
            } else {
                None
            };
            match &s.init {
                Some(ForStatementInit::VariableDeclaration(d)) => g.emit_var_decl(d)?,
                Some(other) => {
                    g.expr(other.as_expression().expect("init expression"))?;
                }
                None => {}
            }
            // per-iteration state: copy registers for the head bindings
            // and the current iteration's context (merge points restore
            // absolutely — a labelled continue may bypass the pops of
            // nested loops still holding contexts)
            let copies: Vec<Reg>;
            let mut iter_ctx: Option<Reg> = None;
            if per_iteration {
                copies = (0..slots).map(|_| g.b.temp()).collect();
                let iter_ctx_reg = g.b.temp();
                // the first iteration starts from a copy of the head
                // context: values copied out, fresh sibling pushed
                g.emit_iteration_context_copy(Some(iter_ctx_reg), slots, &copies, ctx_save);
                iter_ctx = Some(iter_ctx_reg);
            } else {
                copies = Vec::new();
            }
            // head: cond?; JumpIfFalsy breaks; body; continues; update; back-edge
            let back = g.b.new_label();
            g.b.bind(back); // loop head
            let breaks = g.b.new_label();
            let continues = g.b.new_label();
            g.breakables.push(Breakable {
                labels,
                breaks,
                continues: Some(continues),
                unwind_ctx: ctx_save,
            });
            if let Some(cond) = &s.test {
                g.expr(cond)?;
                g.b.jump_if_falsy(breaks);
            }
            g.stmt(&s.body)?;
            let continues = g.breakables.last().unwrap().continues.unwrap();
            g.b.bind(continues);
            if let Some(iter_ctx) = iter_ctx {
                // absolute restore to this iteration's context, then copy
                // its values into a fresh sibling for the next iteration
                // (ES 14.7.5.4: the copy precedes the update)
                g.b.pop_context(iter_ctx);
                g.emit_iteration_context_copy(Some(iter_ctx), slots, &copies, ctx_save);
            } else if let Some(lc) = loop_ctx {
                g.b.pop_context(lc);
            }
            if let Some(next) = &s.update {
                g.expr(next)?;
            }
            g.b.jump_loop(back);
            // absolute restore: break may arrive from arbitrary context
            // depth (labelled jumps past nested loop pops)
            let (breaks, _) = g.end_breakable();
            g.b.bind(breaks);
            if let Some(save) = ctx_save {
                g.b.pop_context(save);
            }
            Ok(())
        })
    }

    /// Copy a loop head's bindings into a fresh sibling context: read the
    /// slots out of the current context, pop to the shared outer, create
    /// a fresh context (holes) and write the values back in.
    fn emit_iteration_context_copy(
        &mut self,
        iter_ctx: Option<Reg>,
        count: u32,
        copies: &[Reg],
        ctx_save: Option<Reg>,
    ) {
        for slot in 0..count {
            self.b.load_context_slot(slot, 0);
            self.b.store(copies[slot as usize]);
        }
        let save = ctx_save.expect("per-iteration loops own a context");
        self.b.pop_context(save);
        self.b.create_block_context(count);
        if let Some(iter_ctx) = iter_ctx {
            self.b.store(iter_ctx);
        }
        self.b.push_context(save);
        for slot in 0..count {
            self.b.load(copies[slot as usize]);
            self.b.store_context_slot(slot, 0);
        }
    }

    fn end_breakable(&mut self) -> (Label, Option<Label>) {
        let state = self.breakables.pop().expect("breakable state");
        (state.breaks, state.continues)
    }

    fn emit_break_continue(
        &mut self,
        span: Span,
        label: Option<&str>,
        is_break: bool,
    ) -> Result<(), CompileError> {
        if is_break {
            let idx = match label {
                None => self.breakables.len().checked_sub(1),
                Some(name) => self
                    .breakables
                    .iter()
                    .rposition(|b| b.labels.iter().any(|l| l == name)),
            };
            let Some(idx) = idx else {
                return self.err(span, "break outside a breakable statement");
            };
            // unwind the context-owning breakables this jump crosses: each
            // holds the pre-statement context in its save register, so
            // popping them innermost-first lands at the target's context
            for b in self.breakables[idx + 1..].iter().rev() {
                if let Some(reg) = b.unwind_ctx {
                    self.b.pop_context(reg);
                }
            }
            let target = self.breakables[idx].breaks;
            self.b.jump(target);
            return Ok(());
        }
        let idx = match label {
            None => self.breakables.iter().rposition(|b| b.continues.is_some()),
            Some(name) => self
                .breakables
                .iter()
                .rposition(|b| b.labels.iter().any(|l| l == name) && b.continues.is_some()),
        };
        let Some(idx) = idx else {
            return self.err(span, "continue outside a loop");
        };
        for b in self.breakables[idx + 1..].iter().rev() {
            if let Some(reg) = b.unwind_ctx {
                self.b.pop_context(reg);
            }
        }
        let target = self.breakables[idx].continues.unwrap();
        self.b.jump(target);
        Ok(())
    }

    fn emit_switch(&mut self, s: &SwitchStatement<'_>) -> Result<(), CompileError> {
        let labels = self.take_labels();
        // evaluate the discriminant once into a temp
        self.expr(&s.discriminant)?;
        let d = self.b.stage_acc();
        let breaks = self.b.new_label();
        self.breakables.push(Breakable {
            labels,
            breaks,
            continues: None,
            unwind_ctx: None,
        });

        let cases = &s.cases;
        let bodies: Vec<Label> = (0..cases.len()).map(|_| self.b.new_label()).collect();
        let mut default_idx = None;

        for (i, case) in cases.iter().enumerate() {
            match &case.test {
                Some(test) => {
                    self.expr(test)?;
                    let t = self.b.stage_acc();
                    self.b.load(d);
                    self.b.equal_strict(t);
                    self.b.drop_temp();
                    self.b.jump_if_truthy(bodies[i]);
                }
                None => default_idx = Some(i),
            }
        }

        // no case matched: the default body, or past the switch
        let end = self.b.new_label();
        match default_idx {
            Some(i) => self.b.jump(bodies[i]),
            None => self.b.jump(end),
        }

        // bodies execute in order; fallthrough is just sequential layout
        for (i, case) in cases.iter().enumerate() {
            self.b.bind(bodies[i]);
            for st in &case.consequent {
                self.stmt(st)?;
            }
        }

        let (breaks, _) = self.end_breakable();
        self.b.bind(breaks);
        self.b.bind(end);
        self.b.drop_temp(); // discriminant
        Ok(())
    }

    fn emit_try_catch(&mut self, s: &TryStatement<'_>) -> Result<(), CompileError> {
        if s.finalizer.is_some() {
            return self.err(s.span, "finally blocks");
        }
        self.with_temps(|g| {
            // snapshot the current context: an exception may unwind out of
            // context-owning constructs (lexical loop heads, class
            // evaluation) whose PushContext the handler entry bypasses;
            // the handler restores absolutely so the catch block's
            // context-slot accesses see the context of the enclosing
            // statement
            let try_ctx = g.b.temp();
            g.b.load_context();
            g.b.store(try_ctx);
            let t = g.b.begin_try();
            for st in &s.block.body {
                g.stmt(st)?;
            }
            g.b.end_try(t);

            let end = g.b.new_label();
            g.b.jump(end);

            // handler entry: the exception arrives in the accumulator
            g.b.handler_entry(t);
            g.b.pop_context(try_ctx);
            if let Some(h) = &s.handler {
                if let Some(param) = &h.param {
                    match &param.pattern {
                        BindingPattern::ObjectPattern(_) | BindingPattern::ArrayPattern(_) => {
                            let value = g.b.stage_acc();
                            g.emit_binding_pattern(&param.pattern, value)?;
                            g.b.drop_temp();
                        }
                        BindingPattern::BindingIdentifier(b) => {
                            let Some(sym) = b.symbol_id.get() else {
                                return Err(CompileError::new(b.span, "catch param declared"));
                            };
                            g.store_symbol(sym, b.name.as_ref())?;
                        }
                        _ => return g.err(param.span, "catch parameter"),
                    }
                }
                for st in &h.body.body {
                    g.stmt(st)?;
                }
            }

            g.b.bind(end);
            Ok(())
        })
    }

    // -- function body ---------------------------------------------------------

    fn is_derived_ctor(&self) -> bool {
        self.c.facts.functions[self.fid.0 as usize]
            .kind
            .is_derived_class_constructor()
    }

    fn fn_has_instance_fields(&self, fid: Fid) -> bool {
        self.c.facts.functions[fid.0 as usize]
            .class
            .is_some_and(|c| self.c.facts.classes[c.0 as usize].has_instance_fields)
    }

    /// Load this function's own `this`: the context slot when captured
    /// (the bind target of super(), shared with nested arrows), else the
    /// receiver register.
    fn emit_this_load_own(&mut self) {
        match self.c.layouts[self.fid.0 as usize].this_slot {
            Some(slot) => self.b.load_context_slot(slot, 0),
            None => self.b.load(self.b.this_reg()),
        }
    }

    /// `return [expr]` inside a derived constructor: an object result
    /// wins, `undefined` (and missing) return `this` (initialization
    /// checked), other primitives make the [[Construct]] throw (ES
    /// 9.2.2.1 — the primitive escapes and the Construct opcode rejects
    /// it).
    fn emit_derived_return(&mut self, value: Option<&Expression<'_>>) -> Result<(), CompileError> {
        let Some(v) = value else {
            // `return;` → return this (loaded while the frame context is
            // still pushed: the captured-this slot lives in it)
            self.emit_this_load_own();
            self.emit_this_initialized_check();
            self.pop_frame_context();
            self.b.ret();
            return Ok(());
        };
        self.expr(v)?;
        let t = self.b.stage_acc();
        // acc === undefined → return this
        self.b.load_undefined();
        let u = self.b.stage_acc();
        self.b.load(t);
        self.b.equal_strict(u);
        let is_obj = self.b.new_label();
        self.b.jump_if_falsy(is_obj);
        // undefined → return this (context still pushed for the slot read)
        self.emit_this_load_own();
        self.emit_this_initialized_check();
        self.pop_frame_context();
        self.b.ret();
        self.b.bind(is_obj);
        self.b.load(t);
        self.pop_frame_context();
        self.b.ret();
        self.b.drop_temp(); // u
        self.b.drop_temp(); // t
        Ok(())
    }

    /// Fill this function's register-resident lexical bindings with the hole
    /// at prologue time. The frame itself is filled with `undefined`, so
    /// without this a `let`/`const`/class binding read before its
    /// declaration would yield `undefined` instead of throwing.
    fn emit_lexical_hole_fills(&mut self) -> Result<(), CompileError> {
        let scopes = self.c.scopes_by_fid[self.fid.0 as usize].clone();
        for sid in scopes {
            for sym in self.c.scoping.iter_bindings_in(sid) {
                if self.c.facts.param_symbols.contains_key(&sym)
                    || self.c.facts.pattern_params.contains(&sym)
                {
                    continue;
                }
                let flags = self.c.scoping.symbol_flags(sym);
                if !flags.intersects(SymbolFlags::BlockScopedVariable | SymbolFlags::Class) {
                    continue;
                }
                if let Some(&Slot::Local {
                    reg,
                    hole_check: true,
                }) = self.c.slots.get(&sym)
                {
                    self.b.load_hole();
                    self.b.store(Reg::new(reg as i32));
                }
            }
        }
        Ok(())
    }

    fn emit_function_body(&mut self) -> Result<(), CompileError> {
        let layout = &self.c.layouts[self.fid.0 as usize];
        // the script (and each eval compilation) tracks its completion
        // value in a dedicated register between ctx_save and the temps
        self.completion = (self.fid.0 == 0).then(|| Reg::new(layout.register_count as i32 + 1));
        self.ctx_save = Reg::new(layout.register_count as i32);
        self.b
            .set_temp_base(layout.register_count + 1 + u32::from(self.completion.is_some()));

        // prologue: functions with context-allocated bindings get a fresh
        // context pushed onto the frame context; locals below the temps are
        // born as the hole. Functions with none reuse the closure's context
        // (outer slots resolve through it), and the depth baked into their
        // nested accesses already skips this frame.
        self.frame_context = layout.creates_context;
        if self.frame_context {
            let names: Vec<Box<[u8]>> = layout
                .slot_names
                .iter()
                .map(|n| n.as_slice().into())
                .collect();
            let ctx_info = self.b.constant(Constant::ContextNames(names));
            self.b.create_function_context(ctx_info);
            self.b.push_context(self.ctx_save);
        }

        // parameters: non-simple lists (any default / pattern / rest)
        // stage the incoming arguments, hole-fill the parameter registers,
        // and initialize each binding in order (TDZ until its turn, ES
        // 10.2.11); simple lists only copy context-allocated (captured)
        // parameters into their slots
        let params: Vec<(PatTarget<'_, '_>, Option<&Expression<'_>>, bool)> =
            self.c.facts.functions[self.fid.0 as usize]
                .params
                .iter()
                .map(|p| (PatTarget::Binding(p.pattern), p.default, p.rest))
                .collect();
        let non_simple = params.iter().any(|(t, d, rest)| {
            *rest
                || d.is_some()
                || !matches!(t, PatTarget::Binding(BindingPattern::BindingIdentifier(_)))
        });
        if non_simple {
            let n = params.len() as u32;
            let staged_base = self.b.reserve_temps(n);
            // stage the incoming arguments (missing ones arrive as
            // undefined through frame padding)
            for i in 0..n {
                let param = self.b.param(i);
                self.b.load(param);
                self.b.store(Reg::new(staged_base.index() + i as i32));
            }
            // rest arrays capture the frame's raw argument list — build
            // them before the registers are hole-filled
            for (i, (_, _, rest)) in params.iter().enumerate() {
                if *rest {
                    let i = i as u32;
                    self.b
                        .call_runtime_staged(BuiltinFn::CreateRestParameter, &[RtArg::Smi(i)]);
                    self.b.store(Reg::new(staged_base.index() + i as i32));
                }
            }
            // parameters start in their TDZ
            for i in 0..n {
                let param = self.b.param(i);
                self.b.load_hole();
                self.b.store(param);
            }
            // left-to-right initialization
            for (i, (target, default, _)) in params.iter().enumerate() {
                let reg = self.b.param(i as u32);
                let staged = Reg::new(staged_base.index() + i as i32);
                if let Some(default) = default {
                    self.b.load(staged);
                    let skip = self.b.new_label();
                    self.b.jump_if_not_undefined(skip);
                    self.expr(default)?;
                    self.b.store(staged);
                    self.b.bind(skip);
                }
                // InitializeBinding: the register (and, when captured, the
                // context slot) receives the value
                self.b.load(staged);
                self.b.store(reg);
                if let PatTarget::Binding(BindingPattern::BindingIdentifier(b)) = target
                    && let Some(sym) = b.symbol_id.get()
                    && let Some(Slot::Ctx { slot, .. }) = self.c.slots.get(&sym)
                {
                    let slot = *slot;
                    self.b.load(reg);
                    self.b.store_context_slot(slot, 0);
                }
                // pattern parameters destructure the bound value
                if matches!(
                    target,
                    PatTarget::Binding(BindingPattern::ArrayPattern(_))
                        | PatTarget::Binding(BindingPattern::ObjectPattern(_))
                ) {
                    self.emit_binding_pattern(
                        match target {
                            PatTarget::Binding(p) => p,
                            _ => unreachable!(),
                        },
                        staged,
                    )?;
                }
            }
            self.b.drop_temps(self.b.temp_depth() - n); // staged parameter window
        } else {
            // context-allocated parameters: copy the argument into its
            // slot (captured params and direct-eval scopes force params
            // to contexts)
            for (target, _, _) in &params {
                let PatTarget::Binding(BindingPattern::BindingIdentifier(b)) = target else {
                    unreachable!("simple lists have identifier params")
                };
                let Some(sym) = b.symbol_id.get() else {
                    continue;
                };
                if let Some(Slot::Ctx { slot, .. }) = self.c.slots.get(&sym) {
                    let slot = *slot;
                    let index = self.c.facts.param_symbols[&sym];
                    let reg = self.b.param(index);
                    self.b.load(reg);
                    self.b.store_context_slot(slot, 0);
                }
            }
        }

        // store the receiver into the hidden this-slot when a nested
        // arrow captures it
        if let Some(slot) = self.c.layouts[self.fid.0 as usize].this_slot {
            self.b.load(self.b.this_reg());
            self.b.store_context_slot(slot, 0);
        }
        // expose new.target / the running closure to nested arrows
        // (arrow-delegated super() and arrow new.target reads)
        if let Some(slot) = self.c.layouts[self.fid.0 as usize].new_target_slot {
            self.b.load_new_target();
            self.b.store_context_slot(slot, 0);
        }
        if let Some(slot) = self.c.layouts[self.fid.0 as usize].this_function_slot {
            self.b.load_current_closure();
            self.b.store_context_slot(slot, 0);
        }

        let kind = self.c.facts.functions[self.fid.0 as usize].kind;

        // the synthesized default derived constructor forwards every
        // argument to super() and returns the bound this (ES 15.7.13)
        if kind == FnKind::DefaultDerivedCtor {
            self.b.call_runtime(
                BuiltinFn::ConstructSuperAllArgs,
                RegList::new(self.b.this_reg(), 0),
            );
            self.b.store(self.b.this_reg());
            if self.fn_has_instance_fields(self.fid) {
                // InitializeInstanceElements on the bound this:
                // native(ctor, instance)
                self.with_temps(|g| {
                    // (ctor, instance): ctor rides the top slot
                    g.emit_this_load_own();
                    let instance = g.b.stage_acc();
                    g.b.load_current_closure();
                    let ctor = g.b.temp();
                    g.b.store(ctor);
                    g.b.call_runtime(BuiltinFn::InitInstanceFields, RegList::new(instance, 2));
                    Ok(())
                })?;
            }
            self.pop_frame_context();
            self.b.load(self.b.this_reg());
            self.b.ret();
            return Ok(());
        }

        // base class constructors run their instance field initializers
        // right after the receiver exists (ES 7.3.33, before the body)
        if kind == FnKind::BaseClassCtor && self.fn_has_instance_fields(self.fid) {
            self.with_temps(|g| {
                // (ctor, instance): ctor rides the top slot
                g.emit_this_load_own();
                let instance = g.b.stage_acc();
                g.b.load_current_closure();
                let ctor = g.b.temp();
                g.b.store(ctor);
                g.b.call_runtime(BuiltinFn::InitInstanceFields, RegList::new(instance, 2));
                Ok(())
            })?;
        }

        // hoisting: `var`s initialize to undefined and top-level function
        // declarations become closures before any body code runs
        if let Some(vars) = self.c.facts.hoist_vars.get(&self.fid).cloned() {
            for sym in vars {
                let name = self.c.scoping.symbol_name(sym).to_string();
                match self.c.slot_of(sym) {
                    Slot::Global => {
                        let idx = self.b.name(name.as_bytes());
                        let feedback = self.b.new_feedback();
                        self.b.load_undefined();
                        self.b.store_global(idx, feedback);
                    }
                    Slot::Param { .. } => {
                        // simple-list params are already undefined when
                        // omitted (frame padding); a `var` bound to a
                        // parameter keeps its incoming value
                    }
                    Slot::Local { .. } => {
                        // register locals are pre-filled with undefined by
                        // the frame (`Stack::reserve`)
                    }
                    Slot::Ctx { slot, .. } => {
                        self.b.load_undefined();
                        self.b.store_context_slot(slot, 0);
                    }
                    Slot::CtxAt { .. } => unreachable!("vars never live in class/for contexts"),
                }
            }
        }
        if let Some(fns) = self.c.facts.hoist_fns.get(&self.fid).cloned() {
            for (sym, fid) in fns {
                let name = self.c.scoping.symbol_name(sym).to_string();
                let idx = self.b.constant(Constant::Callable(FunctionId(fid.0)));
                self.b.create_closure(idx);
                self.store_symbol(sym, &name)?;
            }
        }

        // lexical declarations start in their TDZ: the frame is filled with
        // `undefined`, so register-resident `let`/`const`/class
        // bindings get an explicit hole. ContextObject-allocated lexicals are born
        // the hole when their context is materialized.
        self.emit_lexical_hole_fills()?;

        // the script's completion value starts as undefined; only
        // value-producing statements overwrite it (see `stmt`)
        if let Some(completion) = self.completion {
            self.b.load_undefined();
            self.b.store(completion);
        }

        // field-initializer functions (`return <init>;`): an anonymous
        // function value is named after the field key (ES 15.7.19)
        let field_key = self.c.facts.functions[self.fid.0 as usize].field_key;
        if field_key.is_some() {
            match &self.c.facts.functions[self.fid.0 as usize].body {
                FnBody::FieldInit(e) => {
                    self.expr(e)?;
                    if self.is_anon_function(e) {
                        self.emit_set_name_for_key_node(field_key.unwrap());
                    }
                    self.pop_frame_context();
                    self.b.ret();
                    return Ok(());
                }
                FnBody::Empty => {
                    self.b.load_undefined();
                    self.pop_frame_context();
                    self.b.ret();
                    return Ok(());
                }
                _ => {}
            }
        }

        match &self.c.facts.functions[self.fid.0 as usize].body {
            FnBody::Script(p) => {
                for stmt in &p.body {
                    self.stmt(stmt)?;
                }
            }
            FnBody::Function(b) => {
                for stmt in &b.statements {
                    self.stmt(stmt)?;
                }
            }
            FnBody::ArrowExpr(e) => {
                self.expr(e)?;
                self.pop_frame_context();
                self.b.ret();
                return Ok(());
            }
            FnBody::FieldInit(_) | FnBody::Empty => {}
        }

        // fallthrough: the script yields its completion value; ordinary
        // functions return undefined; derived constructors return `this`
        // (initialization checked — super() must have run)
        match self.completion {
            Some(completion) => {
                self.pop_frame_context();
                self.b.load(completion);
            }
            None if self.is_derived_ctor() => {
                self.emit_this_load_own();
                self.emit_this_initialized_check();
                self.pop_frame_context();
            }
            None => {
                self.pop_frame_context();
                self.b.load_undefined();
            }
        }
        self.b.ret();
        Ok(())
    }
}
