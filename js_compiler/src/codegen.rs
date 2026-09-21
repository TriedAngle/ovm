use std::collections::HashMap;

use oxc_ast::ast::*;
use oxc_semantic::Scoping;
use oxc_span::{GetSpan, Span};
use oxc_syntax::node::NodeId;
use oxc_syntax::scope::{ScopeFlags, ScopeId};
use oxc_syntax::symbol::{SymbolFlags, SymbolId};

use bytecode::{Opcode, PropertyFlags, emit};
use ir::{CallableKind, Constant, FunctionBuilder, FunctionId as IrFunctionId, Program};

use crate::analysis::{ClassIdx, Fid, Facts, FnBody, FnKind, Home, MemberKind, Mode, Special};
use crate::label::Label;

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

/// Emit a forced-wide relative jump to `label`.
fn emit_jump(code: &mut Vec<u8>, op: Opcode, label: &mut Label) {
    let pc = code.len();
    code.push(Opcode::Wide as u8);
    code.push(op as u8);
    code.extend_from_slice(&[0, 0]);
    label.patch_here(pc);
}

/// A breakable statement: loops add a continue target, switches don't.
struct Breakable {
    labels: Vec<String>,
    breaks: Label,
    continues: Option<Label>,
    /// loops owning a head context: jumps out must unwind past it
    unwind_ctx: Option<u32>,
}

/// A store target ready for the final store.
enum StoreTarget {
    Named { obj: u32, name_idx: u32 },
    Keyed { obj: u32, key: u32 },
    /// `this.#x = v`
    PrivateKeyed { obj: u32, key: u32 },
    SuperNamed { recv: u32, home: u32, name_idx: u32 },
    SuperKeyed { recv: u32, home: u32, key: u32 },
}

/// Name hint for NamedEvaluation in pattern defaults (ES 8.4.3).
enum NameHint {
    Const(u32),
    Reg(u32),
    None,
}

/// The slot decision for a declared symbol, shared between its store and
/// load sites (assigned in the owning function's prologue, or lazily for
/// class / for-head context slots).
#[derive(Clone, Copy, Debug)]
enum Slot {
    Param { index: u32, hole_check: bool },
    Local { reg: u32, hole_check: bool },
    /// context slot in the owning function's frame context; the depth is
    /// a property of each use site
    Ctx { slot: u32, hole_check: bool },
    /// context slot in a class / for-head block context (the depth comes
    /// from the precomputed per-reference table)
    CtxAt { slot: u32, hole_check: bool },
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
    Item { target: PatTarget<'r, 'a>, default: Option<&'r Expression<'a>> },
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
                    other => (PatTarget::Assign(other.as_assignment_target().expect("plain target")), None),
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
        other => PatElement::Item { target: PatTarget::Binding(other), default: None },
    }
}

fn is_simple(p: &BindingPattern<'_>) -> bool {
    matches!(p, BindingPattern::BindingIdentifier(_))
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
    let mut compiler = Compiler {
        scoping,
        facts,
        scopes_by_fid,
        slots: HashMap::new(),
        layouts: (0..n)
            .map(|_| Layout {
                register_count: 0,
                this_slot: None,
                new_target_slot: None,
                this_function_slot: None,
                slot_names: Vec::new(),
            })
            .collect(),
        fid_scope: facts
            .fn_scope_to_fid
            .iter()
            .map(|(&s, &f)| (f, s))
            .collect(),
        for_counters: HashMap::new(),
    };
    let mut program = Program::with_capacity(n);
    for fid in 0..n as u32 {
        let mut fgen = FunctionGen::new(&mut compiler, Fid(fid));
        fgen.emit_function_body()?;
        debug_assert!(fgen.next_temp == 0, "unbalanced temp allocation in {fid}");
        program.add_function(fgen.finish());
    }
    validate_ir(&program);
    Ok(program)
}

/// Debug-mode IR contract check: every register operand must land inside
/// the frame (params at `-(arity+1)..-1`, locals/temps in
/// `0..register_count`), every feedback operand inside the feedback
/// vector, every jump inside the code. Violations are frontend bugs; a
/// clean pass shifts suspicion to the VM.
#[cfg(debug_assertions)]
fn validate_ir(program: &Program) {
    use bytecode::Operand;
    for (fi, f) in program.functions().enumerate() {
        let arity = f.arity as i32;
        let code = program.code(f);
        let mut pc = 0usize;
        while pc < code.len() {
            let (op, ops, next) = bytecode::decode(code, pc);
            let mut reg_window: Option<(i32, u32)> = None;
            for (i, kind) in op.operands().iter().enumerate() {
                match kind {
                    Operand::Register => {
                        let r = ops.reg(i);
                        assert!(
                            r >= -(arity + 1) && r < f.register_count as i32,
                            "function {fi} pc {pc}: {op:?} register {r} outside frame                              (arity {arity}, register_count {})",
                            f.register_count
                        );
                    }
                    Operand::RegisterListStart => {
                        let base = ops.reg_list(i);
                        if let Some(count_operand) = op
                            .operands()
                            .iter()
                            .position(|k| matches!(k, Operand::RegisterCount))
                        {
                            reg_window = Some((base, ops.reg_count(count_operand) as u32));
                        }
                    }
                    Operand::RegisterCount | Operand::Immediate => {}
                    Operand::UImmediate => {
                        let v = ops.uimm(i);
                        if matches!(op, Opcode::LoadGlobal | Opcode::LoadGlobalNoThrow | Opcode::StoreGlobal) && i == 1
                            || matches!(
                                op,
                                Opcode::LoadNamedProperty
                                    | Opcode::StoreNamedProperty
                                    | Opcode::LoadKeyedProperty
                                    | Opcode::StoreKeyedProperty
                                    | Opcode::StoreKeyedPropertyNoShadow
                            ) && i == op.operands().len() - 1
                        {
                            assert!(
                                v < f.feedback_count,
                                "function {fi} pc {pc}: {op:?} feedback {v} >= {}",
                                f.feedback_count
                            );
                        }
                    }
                    Operand::Index => {}
                }
            }
            if let Some((base, count)) = reg_window {
                assert!(
                    base + count as i32 <= f.register_count as i32 && base >= -(arity + 1),
                    "function {fi} pc {pc}: {op:?} register window {base}..{} outside frame                      (register_count {})",
                    base + count as i32,
                    f.register_count
                );
            }
            if matches!(
                op,
                Opcode::Jump
                    | Opcode::JumpLoop
                    | Opcode::JumpIfTruthy
                    | Opcode::JumpIfFalsy
                    | Opcode::JumpIfNotUndefined
            ) {
                let target = pc as i64 + ops.imm(0) as i64;
                assert!(
                    (0..code.len() as i64).contains(&target),
                    "function {fi} pc {pc}: {op:?} jumps to {target}, code is {} bytes",
                    code.len()
                );
            }
            pc = next;
        }
    }
}

#[cfg(not(debug_assertions))]
fn validate_ir(_program: &Program) {}


impl<'a, 'p> Compiler<'a, 'p> {
    fn scope_creates_ctx(&self, scope: ScopeId) -> bool {
        if self.facts.fn_scope_to_fid.contains_key(&scope) {
            return true;
        }
        if let Some(&idx) = self.facts.class_of_scope.get(&scope) {
            return self.facts.classes[idx.0 as usize].slot_count > 0;
        }
        self.facts.for_of_scope.contains_key(&scope)
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

    /// Context hops from the use site's scope to the context hosting
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
            Slot::CtxAt { slot, hole_check: true }
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
        let repl_root =
            matches!(self.facts.mode, Mode::Repl) && self.facts.functions[fid.0 as usize].kind == FnKind::Script;
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
                    self.scoping.symbol_flags(sym).intersects(
                        SymbolFlags::BlockScopedVariable | SymbolFlags::Class,
                    )
                };
                let forced = calls_eval || self.facts.captured.contains(&sym);
                let slot = if is_param && !is_pattern_param && !forced {
                    Slot::Param { index: self.facts.param_symbols[&sym], hole_check }
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

        Layout {
            register_count: next_reg,
            this_slot,
            new_target_slot,
            this_function_slot,
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
    code: Vec<u8>,
    constants: Vec<Constant>,
    handlers: Vec<ir::HandlerEntry>,
    /// first temp register: locals + 1 (context-save slot)
    reg_base: u32,
    /// register holding the pushed-over context for prologue/epilogue
    ctx_save: i32,
    /// script completion-value register (scripts/eval only)
    completion: Option<u32>,
    next_temp: u32,
    max_temps: u32,
    feedback_slots: u32,
    breakables: Vec<Breakable>,
    /// labels forwarded into an enclosing loop/switch by LabeledStatements
    nested_labels: Vec<String>,
}

impl<'c, 'a, 'p> FunctionGen<'c, 'a, 'p> {
    fn new(c: &'c mut Compiler<'a, 'p>, fid: Fid) -> Self {
        Self {
            c,
            fid,
            code: Vec::new(),
            constants: Vec::new(),
            handlers: Vec::new(),
            reg_base: 0,
            ctx_save: 0,
            completion: None,
            next_temp: 0,
            max_temps: 0,
            feedback_slots: 0,
            breakables: Vec::new(),
            nested_labels: Vec::new(),
        }
    }

    fn finish(self) -> FunctionBuilder {
        let info = &self.c.facts.functions[self.fid.0 as usize];
        let layout = &self.c.layouts[self.fid.0 as usize];
        FunctionBuilder {
            bytecode: self.code,
            constants: self.constants,
            name: info.name.clone().map(String::into_bytes),
            arity: info.params.len() as u32,
            length: info.formal_length,
            kind: callable_kind(info.kind),
            strict: info.strict,
            register_count: layout.register_count
                + 1
                + u32::from(self.completion.is_some())
                + self.max_temps,
            handlers: self.handlers,
            feedback_count: self.feedback_slots,
        }
    }

    fn err<T>(&self, span: Span, feature: &'static str) -> Result<T, CompileError> {
        Err(CompileError::new(span, feature))
    }

    // -- temps ------------------------------------------------------------

    fn push_value(&mut self) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += 1;
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(&mut self.code, Opcode::Store, &[r]);
        r
    }

    fn pop_value(&mut self) {
        debug_assert!(self.next_temp > 0, "temp underflow");
        self.next_temp -= 1;
    }

    fn reserve_temp(&mut self) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += 1;
        self.max_temps = self.max_temps.max(self.next_temp);
        r
    }

    fn reserve_temps(&mut self, n: u32) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += n;
        self.max_temps = self.max_temps.max(self.next_temp);
        r
    }

    fn with_temps<F, T>(&mut self, f: F) -> Result<T, CompileError>
    where
        F: FnOnce(&mut Self) -> Result<T, CompileError>,
    {
        let mark = self.next_temp;
        let result = f(self);
        self.next_temp = mark;
        result
    }

    // -- feedback / constants ------------------------------------------------

    fn feedback_slot(&mut self) -> u32 {
        let slot = self.feedback_slots;
        self.feedback_slots += 2;
        slot
    }

    fn add_constant(&mut self, c: Constant) -> u32 {
        self.constants.push(c);
        (self.constants.len() - 1) as u32
    }

    fn emit_load_constant(&mut self, c: Constant) {
        let idx = self.add_constant(c);
        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
    }

    fn emit_load_undefined(&mut self) {
        emit(&mut self.code, Opcode::LoadUndefined, &[]);
    }

    fn emit_runtime_call<F>(&mut self, f: bytecode::RuntimeFn, count: u32, fill: F)
    where
        F: FnOnce(&mut Self, u32),
    {
        let base = self.reserve_temps(count);
        fill(self, base);
        emit(&mut self.code, Opcode::CallRuntime, &[f as u32, base, count]);
        self.next_temp -= count;
    }

    fn stage_reg(&mut self, src: u32, dst: u32) {
        emit(&mut self.code, Opcode::Load, &[src]);
        emit(&mut self.code, Opcode::Store, &[dst]);
    }

    fn stage_constant(&mut self, idx: u32, dst: u32) {
        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
        emit(&mut self.code, Opcode::Store, &[dst]);
    }

    fn stage_smi(&mut self, v: u32, dst: u32) {
        emit(&mut self.code, Opcode::LoadSmi, &[v]);
        emit(&mut self.code, Opcode::Store, &[dst]);
    }

    fn stage_acc(&mut self, dst: u32) {
        emit(&mut self.code, Opcode::Store, &[dst]);
    }

    fn emit_this_initialized_check(&mut self) {
        let t = self.push_value();
        emit(
            &mut self.code,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::ThrowSuperNotCalledIfHole as u32, t, 1],
        );
        self.pop_value();
    }

    fn name_constant(&mut self, key: &PropertyKey<'_>) -> Result<u32, CompileError> {
        let bytes: Vec<u8> = match key {
            PropertyKey::StaticIdentifier(i) => i.name.as_bytes().to_vec(),
            PropertyKey::StringLiteral(s) => s.value.as_bytes().to_vec(),
            _ => return self.err(key.span(), "non-string property names"),
        };
        Ok(self.add_constant(Constant::String(bytes)))
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
        let idx = self.add_constant(Constant::String(bytes.to_vec()));
        self.emit_set_name_by_const(idx);
    }

    fn emit_set_name_by_const(&mut self, idx: u32) {
        self.emit_runtime_call(bytecode::RuntimeFn::SetFunctionName, 3, |g, b| {
            g.stage_acc(b);
            g.stage_constant(idx, b + 1);
            g.stage_smi(0, b + 2);
        });
    }

    fn emit_set_name_by_reg(&mut self, key: u32, prefix: u32) {
        self.emit_runtime_call(bytecode::RuntimeFn::SetFunctionName, 3, |g, b| {
            g.stage_acc(b);
            g.stage_reg(key, b + 1);
            g.stage_smi(prefix, b + 2);
        });
    }

    fn emit_set_name_for_key_node(&mut self, key: &PropertyKey<'_>) {
        match key {
            PropertyKey::StaticIdentifier(i) => {
                let bytes = i.name.as_bytes().to_vec();
                self.emit_set_name_const(&bytes);
            }
            PropertyKey::StringLiteral(s) => {
                let bytes = s.value.as_bytes().to_vec();
                self.emit_set_name_const(&bytes);
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
                let idx = self.add_constant(Constant::String(name.as_bytes().to_vec()));
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::StoreGlobal, &[idx, feedback]);
            }
            Slot::Param { index, .. } => {
                emit(&mut self.code, Opcode::Store, &[(-(index as i32 + 2)) as u32]);
            }
            Slot::Local { reg, .. } => emit(&mut self.code, Opcode::Store, &[reg]),
            Slot::Ctx { slot, .. } => {
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
            Slot::CtxAt { slot, .. } => {
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
        }
        Ok(())
    }

    /// Store the accumulator into an identifier assignment target.
    fn store_name(&mut self, ident: &IdentifierReference<'_>) -> Result<(), CompileError> {
        match self.identifier_resolution(ident)? {
            IdRes::Global => {
                let idx = self.add_constant(Constant::String(ident.name.as_bytes().to_vec()));
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::StoreGlobal, &[idx, feedback]);
            }
            IdRes::Dynamic => {
                let idx = self.add_constant(Constant::String(ident.name.as_bytes().to_vec()));
                self.emit_runtime_call(bytecode::RuntimeFn::StoreDynamicName, 2, |g, b| {
                    g.stage_acc(b);
                    g.stage_constant(idx, b + 1);
                });
            }
            IdRes::Slot(slot, depth) => match slot {
                Slot::Param { index, .. } => {
                    emit(&mut self.code, Opcode::Store, &[(-(index as i32 + 2)) as u32]);
                    let _ = depth;
                }
                Slot::Local { reg, .. } => emit(&mut self.code, Opcode::Store, &[reg]),
                Slot::Ctx { slot, .. } => {
                    emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth]);
                }
                Slot::CtxAt { slot, .. } => {
                    emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth]);
                }
                Slot::Global => unreachable!(),
            },
        }
        Ok(())
    }

    fn emit_identifier(&mut self, ident: &IdentifierReference<'_>) -> Result<(), CompileError> {
        match self.identifier_resolution(ident)? {
            IdRes::Global => {
                let idx = self.add_constant(Constant::String(ident.name.as_bytes().to_vec()));
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::LoadGlobal, &[idx, feedback]);
            }
            IdRes::Dynamic => {
                let idx = self.add_constant(Constant::String(ident.name.as_bytes().to_vec()));
                self.emit_runtime_call(bytecode::RuntimeFn::LoadDynamicName, 1, |g, b| {
                    g.stage_constant(idx, b);
                });
            }
            IdRes::Slot(Slot::Param { index, hole_check }, _) => {
                emit(&mut self.code, Opcode::Load, &[(-(index as i32 + 2)) as u32]);
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
            }
            IdRes::Slot(Slot::Local { reg, hole_check }, _) => {
                emit(&mut self.code, Opcode::Load, &[reg]);
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
            }
            IdRes::Slot(Slot::Ctx { slot, hole_check }, depth) => {
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
            }
            IdRes::Slot(Slot::CtxAt { slot, hole_check }, depth) => {
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
            }
            IdRes::Slot(Slot::Global, _) => unreachable!(),
        }
        Ok(())
    }

    /// Whether an identifier resolves to a plain global-object reference
    /// (the `typeof` no-throw path).
    fn resolves_global(&mut self, ident: &IdentifierReference<'_>) -> bool {
        matches!(self.identifier_resolution(ident), Ok(IdRes::Global))
    }

    fn identifier_resolution(&mut self, ident: &IdentifierReference<'_>) -> Result<IdRes, CompileError> {
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
                let slot = self.c.slot_of(sym);
                match slot {
                    Slot::Global => Ok(IdRes::Global),
                    // registers carry no depth
                    Slot::Param { index, hole_check } => {
                        Ok(IdRes::Slot(Slot::Param { index, hole_check }, 0))
                    }
                    Slot::Local { reg, hole_check } => {
                        Ok(IdRes::Slot(Slot::Local { reg, hole_check }, 0))
                    }
                    Slot::CtxAt { slot, hole_check } | Slot::Ctx { slot, hole_check } => {
                        // depths are precomputed over node ancestry
                        // (synthesized field-initializer frames count);
                        // the scope-tree walk is only a fallback
                        let depth = self
                            .c
                            .facts
                            .ref_depth
                            .get(&rid)
                            .copied()
                            .unwrap_or_else(|| {
                                let decl_scope = self.c.scoping.symbol_scope_id(sym);
                                self.c.depth_to(reference.scope_id(), decl_scope)
                            });
                        Ok(IdRes::Slot(Slot::Ctx { slot, hole_check }, depth))
                    }
                }
            }
        }
    }

    fn emit_not(&mut self) {
        let mut is_truthy = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut is_truthy);
        emit(&mut self.code, Opcode::LoadTrue, &[]);
        let mut end = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut end);
        is_truthy.bind(&self.code);
        is_truthy.patch_all(&mut self.code);
        emit(&mut self.code, Opcode::LoadFalse, &[]);
        end.bind(&self.code);
        end.patch_all(&mut self.code);
    }

    /// Load a private-name symbol from its class-context slot.
    fn emit_private_key_load(&mut self, node: NodeId, span: Span) -> Result<(), CompileError> {
        match self.c.facts.special.get(&node) {
            Some(Special::Private { slot, depth }) => {
                emit(&mut self.code, Opcode::LoadContextSlot, &[*slot, *depth]);
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
            emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
        } else if let Some(slot) = self.c.layouts[self.fid.0 as usize].this_slot {
            emit(&mut self.code, Opcode::LoadContextSlot, &[slot, 0]);
        } else {
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
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
                    emit(&mut self.code, Opcode::LoadSmi, &[f as i32 as u32]);
                } else if is_int && f.abs() <= MAX_EXACT_INT {
                    self.emit_load_constant(Constant::Smi(f as i64));
                } else {
                    self.emit_load_constant(Constant::Float(f));
                }
                Ok(())
            }
            Expression::StringLiteral(s) => {
                self.emit_load_constant(Constant::String(s.value.as_bytes().to_vec()));
                Ok(())
            }
            Expression::BigIntLiteral(_) => self.err(e.span(), "BigInt literals"),
            Expression::RegExpLiteral(_) => self.err(e.span(), "regular expressions"),
            Expression::BooleanLiteral(b) => {
                let op = if b.value { Opcode::LoadTrue } else { Opcode::LoadFalse };
                emit(&mut self.code, op, &[]);
                Ok(())
            }
            Expression::NullLiteral(_) => {
                emit(&mut self.code, Opcode::LoadNull, &[]);
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
                let mut else_l = Label::new();
                emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut else_l);
                self.expr(&c.consequent)?;
                let mut end = Label::new();
                emit_jump(&mut self.code, Opcode::Jump, &mut end);
                else_l.bind(&self.code);
                else_l.patch_all(&mut self.code);
                self.expr(&c.alternate)?;
                end.bind(&self.code);
                end.patch_all(&mut self.code);
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
            Expression::StaticMemberExpression(m) => {
                self.emit_property_load(MemberRef::Static(m))
            }
            Expression::ComputedMemberExpression(m) => {
                self.emit_property_load(MemberRef::Computed(m))
            }
            Expression::PrivateFieldExpression(m) => {
                self.emit_property_load(MemberRef::Private(m))
            }
            Expression::ArrayExpression(a) => self.emit_array_literal(a),
            Expression::ObjectExpression(o) => self.emit_object_literal(o),
            Expression::FunctionExpression(f) => {
                let fid = self.c.facts.fn_of_node[&f.node_id.get()];
                let idx = self.add_constant(Constant::Callable(IrFunctionId(fid.0)));
                emit(&mut self.code, Opcode::CreateClosure, &[idx]);
                Ok(())
            }
            Expression::ArrowFunctionExpression(f) => {
                let fid = self.c.facts.fn_of_node[&f.node_id.get()];
                let idx = self.add_constant(Constant::Callable(IrFunctionId(fid.0)));
                emit(&mut self.code, Opcode::CreateClosure, &[idx]);
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
                        emit(&mut self.code, Opcode::LoadContextSlot, &[slot, *depth]);
                    }
                    _ => emit(&mut self.code, Opcode::LoadNewTarget, &[]),
                }
                Ok(())
            }
            Expression::ImportMeta(_) => self.err(e.span(), "import.meta"),
            Expression::PrivateInExpression(p) => {
                // `#x in obj`: (key, obj) -> bool
                self.emit_private_key_load(p.left.node_id.get(), p.left.span)?;
                let base = self.push_value();
                self.expr(&p.right)?;
                self.push_value();
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::PrivateIn as u32, base, 2],
                );
                self.pop_value();
                self.pop_value();
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
                emit(&mut self.code, Opcode::Negate, &[]);
                Ok(())
            }
            UnaryOperator::UnaryPlus => {
                emit(&mut self.code, Opcode::LoadZero, &[]);
                let zero = self.push_value();
                self.expr(&u.argument)?;
                emit(&mut self.code, Opcode::Sub, &[zero]);
                self.pop_value();
                Ok(())
            }
            UnaryOperator::Typeof => {
                // `typeof` on an unresolved global yields "undefined"
                // instead of throwing
                if let Expression::Identifier(i) = &u.argument
                    && self.resolves_global(i)
                {
                    let idx = self.add_constant(Constant::String(i.name.as_bytes().to_vec()));
                    let feedback = self.feedback_slot();
                    emit(&mut self.code, Opcode::LoadGlobalNoThrow, &[idx, feedback]);
                    emit(&mut self.code, Opcode::TestTypeof, &[]);
                    return Ok(());
                }
                self.expr(&u.argument)?;
                emit(&mut self.code, Opcode::TestTypeof, &[]);
                Ok(())
            }
            UnaryOperator::Void => {
                self.expr(&u.argument)?;
                self.emit_load_undefined();
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
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::DeleteSuperProperty as u32, 0, 0],
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
                let base = self.push_value();
                match m {
                    MemberRef::Static(s) => {
                        let idx = self
                            .add_constant(Constant::String(s.property.name.as_bytes().to_vec()));
                        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
                    }
                    MemberRef::Computed(c) => {
                        self.expr(&c.expression)?;
                    }
                    MemberRef::Private(_) => unreachable!(),
                }
                self.push_value();
                let runtime_fn = if self.c.facts.functions[self.fid.0 as usize].strict {
                    bytecode::RuntimeFn::DeletePropertyStrict
                } else {
                    bytecode::RuntimeFn::DeletePropertySloppy
                };
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[runtime_fn as u32, base, 2],
                );
                self.pop_value();
                self.pop_value();
                Ok(())
            }
            Expression::Identifier(i) => {
                // strict sites are rejected at parse time; sloppy
                // declarative bindings cannot be deleted (false), free
                // names are global-object properties
                match self.identifier_resolution(i)? {
                    IdRes::Global | IdRes::Dynamic => {
                        let idx = self.add_constant(Constant::String(i.name.as_bytes().to_vec()));
                        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
                        let name = self.push_value();
                        emit(
                            &mut self.code,
                            Opcode::CallRuntime,
                            &[bytecode::RuntimeFn::DeleteIdentifierSloppy as u32, name, 1],
                        );
                        self.pop_value();
                    }
                    _ => emit(&mut self.code, Opcode::LoadFalse, &[]),
                }
                Ok(())
            }
            _ => {
                // not a reference: side effects only, the result is true
                self.expr(&u.argument)?;
                emit(&mut self.code, Opcode::LoadTrue, &[]);
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
                let r = self.push_value();
                self.expr(&b.left)?;
                emit(&mut self.code, Opcode::InstanceOf, &[r]);
                self.pop_value();
                Ok(())
            }
            Op::In => {
                // `key in obj`: (key, obj) -> bool
                self.expr(&b.left)?;
                let base = self.push_value();
                self.expr(&b.right)?;
                self.push_value();
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::HasProperty as u32, base, 2],
                );
                self.pop_value();
                self.pop_value();
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
        let op = match l.operator {
            LogicalOperator::And => Opcode::JumpIfFalsy,
            LogicalOperator::Or => Opcode::JumpIfTruthy,
            LogicalOperator::Coalesce => return self.err(l.span, "nullish coalescing"),
        };
        self.expr(&l.left)?;
        let t = self.push_value();
        let mut short = Label::new();
        emit_jump(&mut self.code, op, &mut short);
        self.expr(&l.right)?;
        let mut end = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut end);
        short.bind(&self.code);
        short.patch_all(&mut self.code);
        emit(&mut self.code, Opcode::Load, &[t]);
        end.bind(&self.code);
        end.patch_all(&mut self.code);
        self.pop_value();
        Ok(())
    }

    fn binary_arith(
        &mut self,
        lhs: &Expression<'_>,
        rhs: &Expression<'_>,
        op: Opcode,
    ) -> Result<(), CompileError> {
        self.expr(lhs)?;
        let a = self.push_value();
        self.expr(rhs)?;
        let b = self.push_value();
        self.pop_value();
        self.pop_value();
        emit(&mut self.code, Opcode::Load, &[a]);
        emit(&mut self.code, op, &[b]);
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
            let v = self.push_value();
            self.emit_assign_target_pattern(&a.left, v)?;
            emit(&mut self.code, Opcode::Load, &[v]);
            self.pop_value();
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
                self.expr(value)?;
                if self.is_anon_function(value) {
                    self.emit_set_name_const(i.name.as_bytes());
                }
                self.store_name(i)
            }
            t if assign_member_ref(t).is_some() => {
                let m = assign_member_ref(t).unwrap();
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
                self.expr(value)?;
                let v = self.push_value();
                self.emit_identifier(i)?;
                emit(&mut self.code, op, &[v]);
                self.pop_value();
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
                let v = self.push_value();
                self.emit_property_load_of(&store);
                emit(&mut self.code, op, &[v]);
                self.pop_value();
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(target.span(), "assignment target"),
        }
    }

    fn emit_update(&mut self, u: &UpdateExpression<'_>) -> Result<(), CompileError> {
        let delta = match u.operator {
            UpdateOperator::Increment => 1i32,
            UpdateOperator::Decrement => -1i32,
        };
        let arith = if delta > 0 { Opcode::Add } else { Opcode::Sub };
        let delta = delta.unsigned_abs();

        if let SimpleAssignmentTarget::AssignmentTargetIdentifier(i) = &u.argument {
            self.emit_identifier(i)?;
            let orig = self.push_value();
            emit(&mut self.code, Opcode::LoadSmi, &[delta]);
            let d = self.push_value();
            emit(&mut self.code, Opcode::LoadZero, &[]);
            let zero = self.push_value();
            emit(&mut self.code, Opcode::Load, &[orig]);
            emit(&mut self.code, Opcode::Sub, &[zero]);
            self.pop_value();
            emit(&mut self.code, arith, &[d]);
            self.pop_value();
            self.store_name(i)?;
            if !u.prefix {
                emit(&mut self.code, Opcode::Load, &[orig]);
            }
            self.pop_value();
            return Ok(());
        }
        if let Some(m) = simple_member_ref(&u.argument) {
            let store = if is_super_member(m) {
                self.prepare_super_store(m)?
            } else {
                self.prepare_property_store(m)?
            };
            self.emit_property_load_of(&store);
            let orig = self.push_value();
            emit(&mut self.code, Opcode::LoadSmi, &[delta]);
            let d = self.push_value();
            emit(&mut self.code, Opcode::LoadZero, &[]);
            let zero = self.push_value();
            emit(&mut self.code, Opcode::Load, &[orig]);
            emit(&mut self.code, Opcode::Sub, &[zero]);
            self.pop_value();
            emit(&mut self.code, arith, &[d]);
            self.pop_value();
            self.emit_property_store(&store);
            if !u.prefix {
                emit(&mut self.code, Opcode::Load, &[orig]);
            }
            self.pop_value();
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
    fn emit_binding_pattern(&mut self, p: &BindingPattern<'_>, v: u32) -> Result<(), CompileError> {
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
        v: u32,
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
        value_reg: u32,
        binding: bool,
    ) -> Result<(), CompileError> {
        // RequireObjectCoercible runs even for the empty pattern
        emit(
            &mut self.code,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::RequireObjectCoercible as u32, value_reg, 1],
        );
        emit(&mut self.code, Opcode::Store, &[value_reg]);
        let has_rest = matches!(props.last(), Some(PatProperty::Rest(_)));
        // with a rest property, every earlier key stays live in contiguous
        // registers as the CopyDataProperties exclusion set
        let excl_base = self.reg_base + self.next_temp;
        let mut excluded = 0u32;
        for prop in &props {
            match prop {
                PatProperty::Named { name, target, default } => {
                    let name_idx = self.add_constant(Constant::String(name.clone()));
                    if has_rest {
                        emit(&mut self.code, Opcode::LoadConstant, &[name_idx]);
                        self.push_value();
                        excluded += 1;
                    }
                    let feedback = self.feedback_slot();
                    emit(
                        &mut self.code,
                        Opcode::LoadNamedProperty,
                        &[value_reg, name_idx, feedback],
                    );
                    self.emit_pattern_element(
                        *target,
                        *default,
                        binding,
                        NameHint::Const(name_idx),
                    )?;
                }
                PatProperty::Rest(target) => {
                    // native layout: (excluded..., target, source)
                    emit(&mut self.code, Opcode::CreateEmptyObjectLiteral, &[]);
                    let rest_obj = self.push_value();
                    emit(&mut self.code, Opcode::Load, &[value_reg]);
                    self.push_value();
                    emit(
                        &mut self.code,
                        Opcode::CallRuntime,
                        &[
                            bytecode::RuntimeFn::CopyDataProperties as u32,
                            excl_base,
                            excluded + 2,
                        ],
                    );
                    emit(&mut self.code, Opcode::Store, &[rest_obj]);
                    self.emit_pattern_leaf(*target, rest_obj, binding)?;
                    self.pop_value();
                    self.pop_value();
                }
                PatProperty::Prop { key, computed, target, default } => {
                    let key_reg = if *computed {
                        if let Some(e) = key.as_expression() {
                            self.expr(e)?;
                        }
                        Some(self.push_value())
                    } else if matches!(key, PropertyKey::NumericLiteral(_)) {
                        self.emit_number_key(key);
                        Some(self.push_value())
                    } else {
                        None
                    };
                    let name_hint = match key_reg {
                        Some(r) => NameHint::Reg(r),
                        None => NameHint::Const(self.name_constant(key)?),
                    };
                    if has_rest {
                        match key_reg {
                            Some(_) => excluded += 1,
                            None => {
                                // materialize the constant key for the
                                // exclusion set
                                if let NameHint::Const(idx) = name_hint {
                                    emit(&mut self.code, Opcode::LoadConstant, &[idx]);
                                    self.push_value();
                                    excluded += 1;
                                }
                            }
                        }
                    }
                    // v = GetV(value, P)
                    match key_reg {
                        Some(k) => {
                            emit(&mut self.code, Opcode::Load, &[k]);
                            let feedback = self.feedback_slot();
                            emit(
                                &mut self.code,
                                Opcode::LoadKeyedProperty,
                                &[value_reg, feedback],
                            );
                        }
                        None => {
                            let NameHint::Const(name_idx) = name_hint else {
                                unreachable!("constant keys have a constant hint")
                            };
                            let feedback = self.feedback_slot();
                            emit(
                                &mut self.code,
                                Opcode::LoadNamedProperty,
                                &[value_reg, name_idx, feedback],
                            );
                        }
                    }
                    self.emit_pattern_element(*target, *default, binding, name_hint)?;
                    if !has_rest && key_reg.is_some() {
                        self.pop_value();
                    }
                }
            }
        }
        for _ in 0..excluded {
            self.pop_value();
        }
        Ok(())
    }

    /// `[a, b = 1, , ...rest]` (ES 14.13).
    fn emit_array_pattern(
        &mut self,
        elements: Vec<PatElement<'_, '_>>,
        value_reg: u32,
        binding: bool,
    ) -> Result<(), CompileError> {
        // iterator = GetIterator(value)
        emit(
            &mut self.code,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::GetIterator as u32, value_reg, 1],
        );
        let iter = self.push_value();
        // done flag (ES 8.5.9: once done, later elements read undefined
        // without calling next again)
        emit(&mut self.code, Opcode::LoadZero, &[]);
        let done = self.push_value();
        let mut rest: Option<PatTarget<'_, '_>> = None;
        for el in &elements {
            match el {
                PatElement::Hole => {
                    // elision still consumes one iterator step
                    emit(&mut self.code, Opcode::Load, &[done]);
                    let mut skip = Label::new();
                    emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut skip);
                    emit(
                        &mut self.code,
                        Opcode::CallRuntime,
                        &[bytecode::RuntimeFn::IteratorNext as u32, iter, 1],
                    );
                    skip.bind(&self.code);
                    skip.patch_all(&mut self.code);
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
            emit(&mut self.code, Opcode::CreateEmptyArrayLiteral, &[]);
            let arr = self.push_value();
            emit(&mut self.code, Opcode::LoadZero, &[]);
            let idx = self.push_value();
            let head = self.code.len();
            emit(&mut self.code, Opcode::Load, &[done]);
            let mut exit = Label::new();
            emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut exit);
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::IteratorNext as u32, iter, 1],
            );
            let result = self.push_value();
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::IteratorDone as u32, result, 1],
            );
            let mut have = Label::new();
            emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut have);
            emit(&mut self.code, Opcode::LoadSmi, &[1]);
            emit(&mut self.code, Opcode::Store, &[done]);
            let mut after = Label::new();
            emit_jump(&mut self.code, Opcode::Jump, &mut after);
            have.bind(&self.code);
            have.patch_all(&mut self.code);
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::IteratorValue as u32, result, 1],
            );
            let feedback = self.feedback_slot();
            emit(
                &mut self.code,
                Opcode::StoreKeyedPropertyNoShadow,
                &[arr, idx, feedback],
            );
            emit(&mut self.code, Opcode::Load, &[idx]);
            emit(&mut self.code, Opcode::LoadSmi, &[1]);
            emit(&mut self.code, Opcode::Add, &[idx]);
            emit(&mut self.code, Opcode::Store, &[idx]);
            let mut back = Label::new();
            emit_jump(&mut self.code, Opcode::JumpLoop, &mut back);
            back.bind_at(head);
            back.patch_all(&mut self.code);
            after.bind(&self.code);
            after.patch_all(&mut self.code);
            self.pop_value(); // result
            self.pop_value(); // idx
            // both loop exits converge here
            exit.bind(&self.code);
            exit.patch_all(&mut self.code);
            emit(&mut self.code, Opcode::Load, &[arr]);
            self.emit_pattern_leaf(target, arr, binding)?;
            self.pop_value(); // arr
        }
        self.pop_value(); // done
        self.pop_value(); // iter
        Ok(())
    }

    /// Load a numeric literal key (small ints inline, floats via the pool).
    fn emit_number_key(&mut self, key: &PropertyKey<'_>) {
        if let PropertyKey::NumericLiteral(n) = key {
            let f = n.value;
            if f.fract() == 0.0 && f.is_sign_positive() && f <= i16::MAX as f64 {
                emit(&mut self.code, Opcode::LoadSmi, &[f as i32 as u32]);
            } else {
                self.emit_load_constant(Constant::Float(f));
            }
        }
    }

    /// One array-pattern element: v = done ? undefined : IteratorValue;
    /// then the shared default handling.
    fn emit_iterator_element(
        &mut self,
        target: PatTarget<'_, '_>,
        default: Option<&Expression<'_>>,
        iter: u32,
        done: u32,
        binding: bool,
    ) -> Result<(), CompileError> {
        self.emit_load_undefined();
        let v = self.push_value();
        emit(&mut self.code, Opcode::Load, &[done]);
        let mut skip_next = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut skip_next);
        emit(
            &mut self.code,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::IteratorNext as u32, iter, 1],
        );
        let result = self.push_value();
        emit(
            &mut self.code,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::IteratorDone as u32, result, 1],
        );
        let mut have = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut have);
        emit(&mut self.code, Opcode::LoadSmi, &[1]);
        emit(&mut self.code, Opcode::Store, &[done]);
        let mut after = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut after);
        have.bind(&self.code);
        have.patch_all(&mut self.code);
        emit(
            &mut self.code,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::IteratorValue as u32, result, 1],
        );
        emit(&mut self.code, Opcode::Store, &[v]);
        after.bind(&self.code);
        after.patch_all(&mut self.code);
        self.pop_value(); // result
        skip_next.bind(&self.code);
        skip_next.patch_all(&mut self.code);
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
            let idx = self.add_constant(Constant::String(bind_name.unwrap()));
            NameHint::Const(idx)
        } else {
            NameHint::None
        };
        self.emit_pattern_element_with(target, default, v, binding, hint)?;
        self.pop_value(); // v
        Ok(())
    }

    /// Default application + target binding with the current value in
    /// `v` (the accumulator is not used).
    fn emit_pattern_element_with(
        &mut self,
        target: PatTarget<'_, '_>,
        default: Option<&Expression<'_>>,
        v: u32,
        binding: bool,
        name_hint: NameHint,
    ) -> Result<(), CompileError> {
        if let Some(default) = default {
            emit(&mut self.code, Opcode::Load, &[v]);
            let mut skip = Label::new();
            emit_jump(&mut self.code, Opcode::JumpIfNotUndefined, &mut skip);
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
            emit(&mut self.code, Opcode::Store, &[v]);
            skip.bind(&self.code);
            skip.patch_all(&mut self.code);
        }
        emit(&mut self.code, Opcode::Load, &[v]);
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
        let v = self.push_value();
        self.emit_pattern_element_with(target, default, v, binding, name_hint)?;
        self.pop_value();
        Ok(())
    }

    /// Store the value in the accumulator into one pattern leaf: a
    /// declaration name, an assignment target, or a nested pattern.
    fn emit_pattern_leaf(
        &mut self,
        target: PatTarget<'_, '_>,
        value_reg: u32,
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
                let value = self.push_value();
                let store = if is_super_member(m) {
                    self.prepare_super_store(m)?
                } else {
                    self.prepare_property_store(m)?
                };
                emit(&mut self.code, Opcode::Load, &[value]);
                self.emit_property_store(&store);
                self.release_store(&store);
                self.pop_value(); // value
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
            StoreTarget::Named { obj, name_idx } => {
                let feedback = self.feedback_slot();
                emit(
                    &mut self.code,
                    Opcode::StoreNamedProperty,
                    &[*obj, *name_idx, feedback],
                );
            }
            StoreTarget::Keyed { obj, key } => {
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::StoreKeyedProperty, &[*obj, *key, feedback]);
            }
            StoreTarget::PrivateKeyed { obj, key: _ } => {
                // (obj, key, value): the value is in the accumulator
                let _ = self.push_value();
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::PrivateSet as u32, *obj, 3],
                );
                self.pop_value();
            }
            StoreTarget::SuperNamed { recv, home, name_idx } => {
                // runtime(home, recv, key, value = acc, semantics: shadow)
                self.emit_runtime_call(bytecode::RuntimeFn::SuperSetProperty, 5, |g, b| {
                    // the value is in the accumulator: stage it first
                    g.stage_acc(b + 3);
                    g.stage_reg(*home, b);
                    g.stage_reg(*recv, b + 1);
                    g.stage_constant(*name_idx, b + 2);
                    g.stage_smi(0, b + 4);
                });
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                self.emit_runtime_call(bytecode::RuntimeFn::SuperSetProperty, 5, |g, b| {
                    g.stage_acc(b + 3);
                    g.stage_reg(*home, b);
                    g.stage_reg(*recv, b + 1);
                    g.stage_reg(*key, b + 2);
                    g.stage_smi(0, b + 4);
                });
            }
        }
    }

    fn emit_property_load_of(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { obj, name_idx } => {
                let feedback = self.feedback_slot();
                emit(
                    &mut self.code,
                    Opcode::LoadNamedProperty,
                    &[*obj, *name_idx, feedback],
                );
            }
            StoreTarget::Keyed { obj, key } => {
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::Load, &[*key]);
                emit(&mut self.code, Opcode::LoadKeyedProperty, &[*obj, feedback]);
            }
            StoreTarget::PrivateKeyed { obj, key } => {
                emit(&mut self.code, Opcode::Load, &[*key]);
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::PrivateGet as u32, *obj, 2],
                );
            }
            StoreTarget::SuperNamed { recv, home, name_idx } => {
                // runtime(home, recv, key) -> value
                self.emit_runtime_call(bytecode::RuntimeFn::SuperGetProperty, 3, |g, b| {
                    g.stage_reg(*home, b);
                    g.stage_reg(*recv, b + 1);
                    g.stage_constant(*name_idx, b + 2);
                });
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                self.emit_runtime_call(bytecode::RuntimeFn::SuperGetProperty, 3, |g, b| {
                    g.stage_reg(*home, b);
                    g.stage_reg(*recv, b + 1);
                    g.stage_reg(*key, b + 2);
                });
            }
        }
    }

    fn release_store(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { .. } => self.pop_value(), // obj
            StoreTarget::Keyed { .. } | StoreTarget::PrivateKeyed { .. } => {
                self.pop_value(); // key
                self.pop_value(); // obj
            }
            StoreTarget::SuperNamed { .. } => {
                self.pop_value(); // home
                self.pop_value(); // recv
            }
            StoreTarget::SuperKeyed { .. } => {
                self.pop_value(); // home
                self.pop_value(); // key
                self.pop_value(); // recv
            }
        }
    }

    fn prepare_property_store(
        &mut self,
        m: MemberRef<'_>,
    ) -> Result<StoreTarget, CompileError> {
        match m {
            MemberRef::Private(p) => {
                self.expr(&p.object)?;
                let obj = self.push_value();
                self.emit_private_key_load(p.field.node_id.get(), p.field.span)?;
                let k = self.push_value();
                Ok(StoreTarget::PrivateKeyed { obj, key: k })
            }
            MemberRef::Static(s) => {
                self.expr(&s.object)?;
                let obj = self.push_value();
                let name_idx = self
                    .add_constant(Constant::String(s.property.name.as_bytes().to_vec()));
                Ok(StoreTarget::Named { obj, name_idx })
            }
            MemberRef::Computed(c) => {
                self.expr(&c.object)?;
                let obj = self.push_value();
                self.expr(&c.expression)?;
                let k = self.push_value();
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
        let Some(Special::Super { home, depth, this_owner, this_depth }) =
            self.c.facts.special.get(&node).copied()
        else {
            return Ok(None);
        };
        self.emit_this_for(this_owner, this_depth);
        let recv = self.push_value();
        let (home_slot, home_depth) = match home {
            Home::Class(_, slot) => (slot, depth),
            Home::Object(_) => (0, depth),
        };
        match m {
            MemberRef::Computed(c) => {
                self.expr(&c.expression)?;
                let k = self.push_value();
                emit(&mut self.code, Opcode::LoadContextSlot, &[home_slot, home_depth]);
                let home = self.push_value();
                Ok(Some(StoreTarget::SuperKeyed { recv, home, key: k }))
            }
            MemberRef::Static(s) => {
                let name_idx = self
                    .add_constant(Constant::String(s.property.name.as_bytes().to_vec()));
                emit(&mut self.code, Opcode::LoadContextSlot, &[home_slot, home_depth]);
                let home = self.push_value();
                Ok(Some(StoreTarget::SuperNamed { recv, home, name_idx }))
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
        match m {
            MemberRef::Private(p) => {
                // PrivateGet: `obj.#x` — own private field or TypeError
                self.expr(&p.object)?;
                let obj = self.push_value();
                self.emit_private_key_load(p.field.node_id.get(), p.field.span)?;
                self.push_value();
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::PrivateGet as u32, obj, 2],
                );
                self.pop_value();
                self.pop_value();
                Ok(())
            }
            MemberRef::Computed(c) => {
                if is_super_member(m) {
                    let Some(store) = self.prepare_super_parts(m)? else {
                        return self.err(c.span, "super property");
                    };
                    self.emit_property_load_of(&store);
                    self.release_store(&store);
                    return Ok(());
                }
                self.expr(&c.object)?;
                let obj = self.push_value();
                self.expr(&c.expression)?;
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::LoadKeyedProperty, &[obj, feedback]);
                self.pop_value();
                Ok(())
            }
            MemberRef::Static(s) => {
                if is_super_member(m) {
                    let Some(store) = self.prepare_super_parts(m)? else {
                        return self.err(s.span, "super property");
                    };
                    self.emit_property_load_of(&store);
                    self.release_store(&store);
                    return Ok(());
                }
                self.expr(&s.object)?;
                let obj = self.push_value();
                let name_idx = self
                    .add_constant(Constant::String(s.property.name.as_bytes().to_vec()));
                let feedback = self.feedback_slot();
                emit(
                    &mut self.code,
                    Opcode::LoadNamedProperty,
                    &[obj, name_idx, feedback],
                );
                self.pop_value();
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
        // list; the callee sits in a fixed slot above the args (evaluation
        // order: receiver, property get, then arguments)
        if let Expression::StaticMemberExpression(s) = &c.callee {
            let argc = c.arguments.len();
            self.expr(&s.object)?;
            let recv = self.push_value();
            let name_idx = self
                .add_constant(Constant::String(s.property.name.as_bytes().to_vec()));
            let feedback = self.feedback_slot();
            emit(
                &mut self.code,
                Opcode::LoadNamedProperty,
                &[recv, name_idx, feedback],
            );
            let callee_reg = self.reg_base + self.next_temp + argc as u32;
            emit(&mut self.code, Opcode::Store, &[callee_reg]);
            // reserve args + callee so nested argument temps land above
            self.next_temp += argc as u32 + 1;
            for (i, arg) in c.arguments.iter().enumerate() {
                self.call_argument(arg)?;
                emit(&mut self.code, Opcode::Store, &[recv + 1 + i as u32]);
            }
            self.max_temps = self.max_temps.max(self.next_temp);
            emit(
                &mut self.code,
                Opcode::CallNoFeedback,
                &[callee_reg, recv, (argc + 1) as u32],
            );
            self.next_temp -= argc as u32 + 1;
            self.pop_value(); // recv
            return Ok(());
        }
        if let Expression::ComputedMemberExpression(cm) = &c.callee {
            let argc = c.arguments.len();
            self.expr(&cm.object)?;
            let recv = self.push_value();
            self.expr(&cm.expression)?;
            let feedback = self.feedback_slot();
            emit(&mut self.code, Opcode::LoadKeyedProperty, &[recv, feedback]);
            let callee_reg = self.reg_base + self.next_temp + argc as u32;
            emit(&mut self.code, Opcode::Store, &[callee_reg]);
            self.next_temp += argc as u32 + 1;
            for (i, arg) in c.arguments.iter().enumerate() {
                self.call_argument(arg)?;
                emit(&mut self.code, Opcode::Store, &[recv + 1 + i as u32]);
            }
            self.max_temps = self.max_temps.max(self.next_temp);
            emit(
                &mut self.code,
                Opcode::CallNoFeedback,
                &[callee_reg, recv, (argc + 1) as u32],
            );
            self.next_temp -= argc as u32 + 1;
            self.pop_value(); // recv
            return Ok(());
        }
        if let Expression::PrivateFieldExpression(p) = &c.callee {
            return self.err(p.span, "calling a private method");
        }
        // plain call: receiver = undefined (slot 0), callee evaluated
        // first, then arguments
        let argc = c.arguments.len();
        self.expr(&c.callee)?;
        let callee_reg = self.reg_base + self.next_temp + 1 + argc as u32;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        self.emit_load_undefined();
        let recv = self.push_value();
        self.next_temp += argc as u32 + 1;
        for (i, arg) in c.arguments.iter().enumerate() {
            self.call_argument(arg)?;
            emit(&mut self.code, Opcode::Store, &[recv + 1 + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(
            &mut self.code,
            Opcode::CallNoFeedback,
            &[callee_reg, recv, (argc + 1) as u32],
        );
        self.next_temp -= argc as u32 + 1;
        self.pop_value(); // recv
        Ok(())
    }

    fn emit_super_method_call(
        &mut self,
        c: &CallExpression<'_>,
        m: MemberRef<'_>,
    ) -> Result<(), CompileError> {
        let argc = c.arguments.len();
        let Some(store) = self.prepare_super_parts(m)? else {
            return self.err(c.span, "super method call");
        };
        self.emit_property_load_of(&store);
        let recv = match &store {
            StoreTarget::SuperNamed { recv, .. } => *recv,
            StoreTarget::SuperKeyed { recv, .. } => *recv,
            _ => unreachable!("super store parts"),
        };
        let callee_reg = self.reg_base + self.next_temp + argc as u32;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        self.next_temp += argc as u32 + 1;
        for (i, arg) in c.arguments.iter().enumerate() {
            self.call_argument(arg)?;
            emit(&mut self.code, Opcode::Store, &[recv + 1 + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(
            &mut self.code,
            Opcode::CallNoFeedback,
            &[callee_reg, recv, (argc + 1) as u32],
        );
        self.next_temp -= argc as u32 + 1;
        self.release_store(&store);
        Ok(())
    }

    fn emit_new(&mut self, n: &NewExpression<'_>) -> Result<(), CompileError> {
        let argc = n.arguments.len();
        self.expr(&n.callee)?;
        let callee_reg = self.reg_base + self.next_temp + argc as u32;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        let arg_base = self.reg_base + self.next_temp;
        self.next_temp += argc as u32 + 1;
        for (i, arg) in n.arguments.iter().enumerate() {
            self.call_argument(arg)?;
            emit(&mut self.code, Opcode::Store, &[arg_base + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(
            &mut self.code,
            Opcode::Construct,
            &[callee_reg, arg_base, argc as u32],
        );
        self.next_temp -= argc as u32 + 1;
        Ok(())
    }

    /// `super(...)`: construct the superclass with the constructor's
    /// new.target and initialize `this` with the result (ES 15.4.3).
    fn emit_super_call(&mut self, c: &CallExpression<'_>) -> Result<(), CompileError> {
        let argc = c.arguments.len();
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

        let arg_base = self.reg_base + self.next_temp;
        // reserve arguments + result (+ closure/new.target registers for
        // the delegated variant) so nested temps land above
        let reserved = argc as u32 + 1 + u32::from(!direct) * 2;
        self.next_temp += reserved;
        for (i, arg) in c.arguments.iter().enumerate() {
            self.call_argument(arg)?;
            emit(&mut self.code, Opcode::Store, &[arg_base + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        if direct {
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::ConstructSuper as u32, arg_base, argc as u32],
            );
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
            let closure_reg = arg_base + argc as u32;
            let new_target_reg = closure_reg + 1;
            emit(
                &mut self.code,
                Opcode::LoadContextSlot,
                &[this_function_slot, owner_depth],
            );
            emit(&mut self.code, Opcode::Store, &[closure_reg]);
            emit(
                &mut self.code,
                Opcode::LoadContextSlot,
                &[new_target_slot, owner_depth],
            );
            emit(&mut self.code, Opcode::Store, &[new_target_reg]);
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[
                    bytecode::RuntimeFn::ConstructSuperVia as u32,
                    arg_base,
                    (argc + 2) as u32,
                ],
            );
        }
        // the constructed instance lands above the (contiguous) runtime
        // argument window
        let result = arg_base + argc as u32 + u32::from(!direct) * 2;
        emit(&mut self.code, Opcode::Store, &[result]);
        // InitializeThisBinding: this must still be uninitialized
        let super_once_check = |g: &mut Self| {
            let t = g.push_value();
            emit(
                &mut g.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::ThrowSuperAlreadyCalledIfNotHole as u32, t, 1],
            );
            g.pop_value();
        };
        match this_slot {
            Some(slot) => {
                let depth = if direct { 0 } else { owner_depth };
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                super_once_check(self);
                emit(&mut self.code, Opcode::Load, &[result]);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth]);
            }
            None => {
                emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
                super_once_check(self);
                emit(&mut self.code, Opcode::Load, &[result]);
                emit(&mut self.code, Opcode::Store, &[(-1i32) as u32]);
            }
        }
        // InitializeInstanceElements (ES 7.3.33): the derived
        // constructor's own fields are defined on the freshly bound
        // instance (for arrow-delegated super(), the owner is the ctor)
        let field_owner = if direct { self.fid } else { owner };
        if self.fn_has_instance_fields(field_owner) {
            let ctor = self.reserve_temp();
            if direct {
                emit(&mut self.code, Opcode::LoadCurrentClosure, &[]);
            } else {
                let slot = self.c.layouts[owner.0 as usize]
                    .this_function_slot
                    .expect("delegated super() forces the closure slot");
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, owner_depth]);
            }
            emit(&mut self.code, Opcode::Store, &[ctor]);
            emit(&mut self.code, Opcode::Load, &[result]);
            self.push_value();
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::InitInstanceFields as u32, ctor, 2],
            );
            self.next_temp -= 2;
        }
        emit(&mut self.code, Opcode::Load, &[result]);
        self.next_temp -= reserved;
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
            emit(&mut self.code, Opcode::CreateBlockContext, &[slot_count]);
            let save = self.reserve_temp();
            emit(&mut self.code, Opcode::PushContext, &[save]);
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
            let desc = self.add_constant(Constant::String(name.as_bytes().to_vec()));
            emit(&mut self.code, Opcode::LoadConstant, &[desc]);
            let desc_reg = self.push_value();
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::CreatePrivateName as u32, desc_reg, 1],
            );
            self.pop_value();
            emit(&mut self.code, Opcode::StoreContextSlot, &[*slot, 0]);
        }

        // superclass: must be null or a constructor
        let sup = if let Some(sup) = superclass {
            self.expr(sup)?;
            let sup = self.push_value();
            self.emit_runtime_call(bytecode::RuntimeFn::ThrowIfNotConstructorOrNull, 1, |g, b| {
                g.stage_reg(sup, b);
            });
            Some(sup)
        } else {
            None
        };

        // prototype/constructor parents; defaults are the no-extends case
        self.emit_load_constant(Constant::ObjectPrototype);
        let pp = self.push_value();
        self.emit_load_constant(Constant::FunctionPrototype);
        let cp = self.push_value();
        if let Some(sup) = sup {
            // null superclass → null-proto prototype; ctor parent stays
            // %Function.prototype% (objects are always truthy, so the
            // falsy test identifies null)
            emit(&mut self.code, Opcode::Load, &[sup]);
            let mut null_extends = Label::new();
            emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut null_extends);
            // protoParent = Get(superCtor, "prototype") (full [[Get]])
            let proto_name = self.add_constant(Constant::String(b"prototype".to_vec()));
            let feedback = self.feedback_slot();
            emit(
                &mut self.code,
                Opcode::LoadNamedProperty,
                &[sup, proto_name, feedback],
            );
            emit(&mut self.code, Opcode::Store, &[pp]);
            self.emit_runtime_call(bytecode::RuntimeFn::ThrowIfNotObjectOrNull, 1, |g, b| {
                g.stage_reg(pp, b);
            });
            emit(&mut self.code, Opcode::Load, &[sup]);
            emit(&mut self.code, Opcode::Store, &[cp]);
            let mut done = Label::new();
            emit_jump(&mut self.code, Opcode::Jump, &mut done);
            null_extends.bind(&self.code);
            null_extends.patch_all(&mut self.code);
            emit(&mut self.code, Opcode::LoadNull, &[]);
            emit(&mut self.code, Opcode::Store, &[pp]);
            done.bind(&self.code);
            done.patch_all(&mut self.code);
        }

        // prototype: a fresh ordinary object with protoParent
        emit(&mut self.code, Opcode::CreateEmptyObjectLiteral, &[]);
        let proto = self.push_value();
        self.emit_runtime_call(bytecode::RuntimeFn::SetPrototype, 2, |g, b| {
            g.stage_reg(proto, b);
            g.stage_reg(pp, b + 1);
        });

        // constructor closure
        let ctor_idx = self.add_constant(Constant::Callable(IrFunctionId(ctor.0)));
        emit(&mut self.code, Opcode::CreateClosure, &[ctor_idx]);
        let ctor = self.push_value();

        // wiring before member installation (ES 15.7.14 steps 17–18
        // precede the element loop): computed `['constructor']` members
        // overwrite proto.constructor, computed static `['prototype']`
        // defines fail against the non-configurable ctor.prototype
        // proto.constructor → the class {w+, e−, c+}
        let ctor_name = self.add_constant(Constant::String(b"constructor".to_vec()));
        let (proto_reg, ctor_reg) = (proto, ctor);
        self.emit_runtime_call(bytecode::RuntimeFn::DefineOwnProperty, 4, |g, b| {
            g.stage_reg(proto_reg, b);
            g.stage_constant(ctor_name, b + 1);
            g.stage_reg(ctor_reg, b + 2);
            g.stage_smi(PropertyFlags::DontEnum.bits(), b + 3);
        });
        // ctor.prototype → the prototype {w+, e−, c−}
        let proto_name = self.add_constant(Constant::String(b"prototype".to_vec()));
        self.emit_runtime_call(bytecode::RuntimeFn::DefineOwnProperty, 4, |g, b| {
            g.stage_reg(ctor_reg, b);
            g.stage_constant(proto_name, b + 1);
            g.stage_reg(proto_reg, b + 2);
            g.stage_smi(
                PropertyFlags::DontEnum.bits() | PropertyFlags::DontDelete.bits(),
                b + 3,
            );
        });
        // the class itself inherits from the superclass constructor
        self.emit_runtime_call(bytecode::RuntimeFn::SetPrototype, 2, |g, b| {
            g.stage_reg(ctor_reg, b);
            g.stage_reg(cp, b + 1);
        });

        // instance field list: a JS array [key0, init0, key1, init1, ...]
        // attached to the constructor (its hidden fields slot)
        let fields_arr = if has_instance_fields {
            emit(&mut self.code, Opcode::CreateEmptyArrayLiteral, &[]);
            Some(self.push_value())
        } else {
            None
        };
        let mut field_index = 0u32;
        // static field keys (evaluated in element order) stay live until
        // the deferred initializer calls after the class is complete
        // (closure register, key register, constant-key name index)
        let mut static_fields: Vec<(u32, Option<u32>, Option<u32>)> = Vec::new();

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
                    Some(self.push_value())
                } else if computed {
                    if let Some(e) = key.as_expression() {
                        self.expr(e)?;
                    }
                    Some(self.push_value())
                } else if matches!(key, PropertyKey::NumericLiteral(_)) {
                    self.emit_number_key(key);
                    Some(self.push_value())
                } else {
                    None
                };
                let fn_idx = self.add_constant(Constant::Callable(IrFunctionId(fid.0)));
                emit(&mut self.code, Opcode::CreateClosure, &[fn_idx]);
                if is_static {
                    let closure = self.push_value();
                    let name_idx = match key_reg {
                        Some(_) => None,
                        None => Some(self.name_constant(key)?),
                    };
                    static_fields.push((closure, key_reg, name_idx));
                } else {
                    // append [key, initializer] to the instance field list
                    let arr = fields_arr.expect("instance fields allocate the list");
                    let closure = self.push_value();
                    emit(&mut self.code, Opcode::LoadSmi, &[field_index]);
                    let i = self.push_value();
                    match key_reg {
                        Some(k) => {
                            emit(&mut self.code, Opcode::Load, &[k]);
                            let feedback = self.feedback_slot();
                            emit(
                                &mut self.code,
                                Opcode::StoreKeyedPropertyNoShadow,
                                &[arr, i, feedback],
                            );
                        }
                        None => {
                            let name_idx = self.name_constant(key)?;
                            emit(&mut self.code, Opcode::LoadConstant, &[name_idx]);
                            let feedback = self.feedback_slot();
                            emit(
                                &mut self.code,
                                Opcode::StoreKeyedPropertyNoShadow,
                                &[arr, i, feedback],
                            );
                        }
                    }
                    // initializer at the next slot
                    emit(&mut self.code, Opcode::LoadSmi, &[field_index + 1]);
                    emit(&mut self.code, Opcode::Store, &[i]);
                    emit(&mut self.code, Opcode::Load, &[closure]);
                    let feedback = self.feedback_slot();
                    emit(
                        &mut self.code,
                        Opcode::StoreKeyedPropertyNoShadow,
                        &[arr, i, feedback],
                    );
                    self.pop_value(); // i
                    self.pop_value(); // closure
                    field_index += 2;
                    if key_reg.is_some() {
                        self.pop_value();
                    }
                }
                continue;
            }
            let target = if is_static { ctor } else { proto };
            // non-computed keys are strings or numbers; numbers go through
            // the keyed define with the literal loaded into a register
            let numeric_key = !computed && matches!(key, PropertyKey::NumericLiteral(_));
            let key_reg = if computed || numeric_key {
                if computed {
                    if let Some(e) = key.as_expression() {
                        self.expr(e)?;
                    }
                } else {
                    self.emit_number_key(key);
                }
                Some(self.push_value())
            } else {
                None
            };
            let fn_idx = self.add_constant(Constant::Callable(IrFunctionId(fid.0)));
            emit(&mut self.code, Opcode::CreateClosure, &[fn_idx]);
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
                    self.emit_runtime_call(bytecode::RuntimeFn::DefineOwnProperty, 4, |g, b| {
                        // the member value is in the accumulator: stage it
                        // before the loads below clobber it
                        g.stage_acc(b + 2);
                        g.stage_reg(target, b);
                        match key_reg {
                            Some(k) => g.stage_reg(k, b + 1),
                            None => g.stage_constant(name_idx.unwrap(), b + 1),
                        }
                        g.stage_smi(PropertyFlags::DontEnum.bits(), b + 3);
                    });
                }
                MemberKind::Get | MemberKind::Set => {
                    // one accessor half; merges with an existing pair:
                    // runtime(target, key, closure = acc, flags)
                    let mut flags = PropertyFlags::DontEnum.bits();
                    if kind == MemberKind::Get {
                        flags |= 1;
                    }
                    self.emit_runtime_call(bytecode::RuntimeFn::InstallAccessor, 4, |g, b| {
                        g.stage_acc(b + 2);
                        g.stage_reg(target, b);
                        match key_reg {
                            Some(k) => g.stage_reg(k, b + 1),
                            None => g.stage_constant(name_idx.unwrap(), b + 1),
                        }
                        g.stage_smi(flags, b + 3);
                    });
                }
                MemberKind::Field => unreachable!("fields handled above"),
            }
            if key_reg.is_some() {
                self.pop_value();
            }
        }

        // home objects and the inner class-name binding
        if uses_super {
            let (home_slot, static_home_slot) = {
                let c = &self.c.facts.classes[idx.0 as usize];
                (c.home_slot.unwrap(), c.static_home_slot.unwrap())
            };
            // the class context is pushed here: zero hops
            emit(&mut self.code, Opcode::Load, &[proto]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[home_slot, 0]);
            emit(&mut self.code, Opcode::Load, &[ctor]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[static_home_slot, 0]);
        }
        if let Some(slot) = name_slot {
            emit(&mut self.code, Opcode::Load, &[ctor]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }

        // static fields: each initializer runs with the constructor as
        // `this` and its result is [[DefineOwnProperty]]'d on it, in
        // declaration order (ES 15.7.14 step 33)
        for (closure, key_reg, name_idx) in &static_fields {
            emit(&mut self.code, Opcode::Load, &[*closure]);
            emit(&mut self.code, Opcode::CallNoFeedback, &[*closure, ctor, 1]);
            // runtime(obj = ctor, key, value = acc, flags 0)
            self.emit_runtime_call(bytecode::RuntimeFn::DefineOwnProperty, 4, |g, b| {
                g.stage_acc(b + 2);
                g.stage_reg(ctor, b);
                match (key_reg, name_idx) {
                    (Some(k), _) => g.stage_reg(*k, b + 1),
                    (None, Some(name_idx)) => g.stage_constant(*name_idx, b + 1),
                    (None, None) => unreachable!("static field keys are reg or const"),
                }
                g.stage_smi(0, b + 3);
            });
        }
        // LIFO pops for the static field registers (key under closure)
        for (_closure, key_reg, _) in static_fields.iter().rev() {
            self.pop_value(); // closure
            if key_reg.is_some() {
                self.pop_value(); // key
            }
        }

        // attach the instance field list to the constructor
        if let Some(arr) = fields_arr {
            emit(&mut self.code, Opcode::Load, &[ctor]);
            let ctor_reg = self.push_value();
            emit(&mut self.code, Opcode::Load, &[arr]);
            self.push_value();
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::SetClassFields as u32, ctor_reg, 2],
            );
            self.pop_value(); // arr copy
            self.pop_value(); // ctor_reg
            self.pop_value(); // the fields array itself
        }

        // pop the class context; member closures already captured it
        if let Some(save) = ctx_save {
            emit(&mut self.code, Opcode::PopContext, &[save]);
            self.next_temp -= 1; // save
        }

        // outer binding for declarations (the class value in acc)
        emit(&mut self.code, Opcode::Load, &[ctor]);
        if is_decl
            && let Some((sym, name)) = decl_symbol
        {
            self.store_symbol(sym, &name)?;
        }

        // LIFO: ctor, proto, cp, pp, superclass
        self.pop_value(); // ctor
        self.pop_value(); // proto
        self.pop_value(); // cp
        self.pop_value(); // pp
        if sup.is_some() {
            self.pop_value();
        }
        emit(&mut self.code, Opcode::Load, &[ctor]);
        Ok(())
    }

    // -- literals -----------------------------------------------------------------

    fn emit_array_literal(&mut self, a: &ArrayExpression<'_>) -> Result<(), CompileError> {
        emit(&mut self.code, Opcode::CreateEmptyArrayLiteral, &[]);
        let arr = self.push_value();
        let mut i = 0u32;
        for el in &a.elements {
            match el {
                ArrayExpressionElement::Elision(_) => {}
                ArrayExpressionElement::SpreadElement(_) => {
                    return self.err(a.span, "spread in array literals");
                }
                other => {
                    let x = other.as_expression().expect("elision/spread handled");
                    emit(&mut self.code, Opcode::LoadSmi, &[i]);
                    let idx = self.push_value();
                    self.expr(x)?;
                    let feedback = self.feedback_slot();
                    emit(
                        &mut self.code,
                        Opcode::StoreKeyedPropertyNoShadow,
                        &[arr, idx, feedback],
                    );
                    self.pop_value();
                    i += 1;
                }
            }
        }
        emit(&mut self.code, Opcode::Load, &[arr]);
        self.pop_value();
        Ok(())
    }

    fn emit_object_literal(&mut self, o: &ObjectExpression<'_>) -> Result<(), CompileError> {
        // methods using `super` capture a per-literal home-object context
        // (the literal itself), mirroring class scopes
        let node = o.node_id.get();
        let needs_home = self.c.facts.obj_lit_home.contains(&node);
        let ctx_save = if needs_home {
            emit(&mut self.code, Opcode::CreateBlockContext, &[1]);
            let save = self.reserve_temp();
            emit(&mut self.code, Opcode::PushContext, &[save]);
            Some(save)
        } else {
            None
        };

        emit(&mut self.code, Opcode::CreateEmptyObjectLiteral, &[]);
        let obj = self.push_value();
        for p in &o.properties {
            let ObjectPropertyKind::ObjectProperty(p) = p else {
                return self.err(o.span, "spread in object literals");
            };
            let key_reg = if p.computed {
                if let Some(e) = p.key.as_expression() {
                    self.expr(e)?;
                }
                Some(self.push_value())
            } else if matches!(p.key, PropertyKey::NumericLiteral(_)) {
                // numeric literal keys ({ 1: x }) use the keyed path
                self.emit_number_key(&p.key);
                Some(self.push_value())
            } else {
                None
            };
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
                            let feedback = self.feedback_slot();
                            emit(
                                &mut self.code,
                                Opcode::StoreNamedProperty,
                                &[obj, name_idx, feedback],
                            );
                        }
                        Some(k) => {
                            if self.is_anon_function(&p.value) {
                                self.emit_set_name_by_reg(k, 0);
                            }
                            let feedback = self.feedback_slot();
                            emit(
                                &mut self.code,
                                Opcode::StoreKeyedProperty,
                                &[obj, k, feedback],
                            );
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
                        let feedback = self.feedback_slot();
                        emit(
                            &mut self.code,
                            Opcode::StoreKeyedProperty,
                            &[obj, k, feedback],
                        );
                    } else {
                        let name_idx = self.name_constant(&p.key)?;
                        let feedback = self.feedback_slot();
                        emit(
                            &mut self.code,
                            Opcode::StoreNamedProperty,
                            &[obj, name_idx, feedback],
                        );
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
                    self.emit_runtime_call(bytecode::RuntimeFn::InstallAccessor, 4, |g, b| {
                        // the closure is in the accumulator: stage it first
                        g.stage_acc(b + 2);
                        g.stage_reg(obj, b);
                        match key_reg {
                            Some(k) => g.stage_reg(k, b + 1),
                            None => g.stage_constant(name_idx.unwrap(), b + 1),
                        }
                        g.stage_smi(flags, b + 3);
                    });
                }
            }
            if key_reg.is_some() {
                self.pop_value();
            }
        }
        // the home object is the literal itself; methods already captured
        // the context, so the store is visible to them
        if needs_home {
            emit(&mut self.code, Opcode::Load, &[obj]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[0, 0]);
        }
        if let Some(save) = ctx_save {
            emit(&mut self.code, Opcode::PopContext, &[save]);
            self.next_temp -= 1;
        }
        emit(&mut self.code, Opcode::Load, &[obj]);
        self.pop_value();
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
                    emit(&mut self.code, Opcode::Store, &[completion]);
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
                let mut else_l = Label::new();
                emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut else_l);
                self.stmt(&s.consequent)?;
                match &s.alternate {
                    Some(e) => {
                        let mut end = Label::new();
                        emit_jump(&mut self.code, Opcode::Jump, &mut end);
                        else_l.bind(&self.code);
                        else_l.patch_all(&mut self.code);
                        self.stmt(e)?;
                        end.bind(&self.code);
                        end.patch_all(&mut self.code);
                    }
                    None => {
                        else_l.bind(&self.code);
                        else_l.patch_all(&mut self.code);
                    }
                }
                Ok(())
            }
            Statement::WhileStatement(s) => {
                let labels = self.take_labels();
                self.emit_while(&s.test, &s.body, labels)
            }
            Statement::DoWhileStatement(s) => self.err(s.span, "do-while loops"),
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
                    None => self.emit_load_undefined(),
                }
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                emit(&mut self.code, Opcode::Return, &[]);
                Ok(())
            }
            Statement::ThrowStatement(s) => {
                self.expr(&s.argument)?;
                emit(&mut self.code, Opcode::Throw, &[]);
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
                    let idx = self.add_constant(Constant::Callable(IrFunctionId(fid.0)));
                    emit(&mut self.code, Opcode::CreateClosure, &[idx]);
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
                self.breakables.push(Breakable {
                    labels: vec![label],
                    breaks: Label::new(),
                    continues: None,
                    unwind_ctx: None,
                });
                let result = self.stmt(&s.body);
                let (mut breaks, _) = self.end_breakable();
                breaks.bind(&self.code);
                breaks.patch_all(&mut self.code);
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
                    let value = self.push_value();
                    self.emit_binding_pattern(&decl.id, value)?;
                    self.pop_value();
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
            let mut loop_ctx: Option<u32> = None;
            let ctx_save = if slots > 0 {
                emit(&mut g.code, Opcode::CreateBlockContext, &[slots]);
                let save = g.reserve_temp();
                let lc = g.reserve_temp();
                emit(&mut g.code, Opcode::Store, &[lc]);
                emit(&mut g.code, Opcode::PushContext, &[save]);
                loop_ctx = Some(lc);
                Some(save)
            } else {
                None
            };
            // head: subject → enumerator (undefined for nullish subjects;
            // ForInNext(undefined) is immediately done)
            g.expr(&s.right)?;
            let subject = g.push_value();
            emit(
                &mut g.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::ForInEnumerate as u32, subject, 1],
            );
            g.pop_value(); // the call consumed the subject; acc = enumerator
            let enumerator = g.push_value();

            // loop: next key or undefined
            let head = g.code.len();
            g.breakables.push(Breakable {
                labels,
                breaks: Label::new(),
                continues: Some(Label::new()),
                unwind_ctx: ctx_save,
            });
            emit(
                &mut g.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::ForInNext as u32, enumerator, 1],
            );
            let mut have_key = Label::new();
            emit_jump(&mut g.code, Opcode::JumpIfNotUndefined, &mut have_key);
            emit_jump(
                &mut g.code,
                Opcode::Jump,
                &mut g.breakables.last_mut().unwrap().breaks,
            );
            have_key.bind(&g.code);
            have_key.patch_all(&mut g.code);
            let key = g.push_value();

            // per iteration with a lexical head: replace the context with
            // a fresh sibling (same outer) and initialize the binding
            // there — a fresh binding per iteration
            if per_iteration {
                let save = ctx_save.expect("per-iteration loops own a context");
                emit(&mut g.code, Opcode::PopContext, &[save]);
                emit(&mut g.code, Opcode::CreateBlockContext, &[slots]);
                emit(&mut g.code, Opcode::PushContext, &[save]);
            }

            // assign the key to the target (per iteration), then the body
            g.emit_for_in_assign(&s.left, key)?;
            g.stmt(&s.body)?;

            g.breakables
                .last_mut()
                .unwrap()
                .continues
                .as_mut()
                .unwrap()
                .bind(&g.code);
            // absolute restore: a labelled continue may arrive from a
            // nested construct still holding its context (per-iteration
            // heads self-heal at the top of the next iteration)
            if !per_iteration && let Some(lc) = loop_ctx {
                emit(&mut g.code, Opcode::PopContext, &[lc]);
            }
            let mut back = Label::new();
            emit_jump(&mut g.code, Opcode::JumpLoop, &mut back);
            g.bind_loop_end(back, head);
            // absolute restore: break may arrive from arbitrary context
            // depth (labelled jumps past nested loop pops)
            if let Some(save) = ctx_save {
                emit(&mut g.code, Opcode::PopContext, &[save]);
            }

            g.pop_value(); // key
            g.pop_value(); // enumerator
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
        key: u32,
    ) -> Result<(), CompileError> {
        match left {
            ForStatementLeft::VariableDeclaration(d) => {
                let [declarator] = d.declarations.as_slice() else {
                    return self.err(d.span, "for-in declarator");
                };
                match &declarator.id {
                    BindingPattern::BindingIdentifier(b) => {
                        emit(&mut self.code, Opcode::Load, &[key]);
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
                emit(&mut self.code, Opcode::Load, &[key]);
                self.store_name(i)
            }
            t if assign_member_ref_for_left(t).is_some() => {
                let m = assign_member_ref_for_left(t).unwrap();
                let store = if is_super_member(m) {
                    self.prepare_super_store(m)?
                } else {
                    self.prepare_property_store(m)?
                };
                emit(&mut self.code, Opcode::Load, &[key]);
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(left.span(), "for-in assignment target"),
        }
    }

    fn bind_loop_end(&mut self, mut back: Label, head: usize) {
        let (mut breaks, continues) = self.end_breakable();
        breaks.bind(&self.code);
        breaks.patch_all(&mut self.code);
        continues.unwrap().patch_all(&mut self.code);
        back.bind_at(head);
        back.patch_all(&mut self.code);
    }

    fn emit_while(
        &mut self,
        test: &Expression<'_>,
        body: &Statement<'_>,
        labels: Vec<String>,
    ) -> Result<(), CompileError> {
        // head: cond; JumpIfFalsy breaks; body; continues; back-edge
        let head = self.code.len();
        self.expr(test)?;
        self.breakables.push(Breakable {
            labels,
            breaks: Label::new(),
            continues: Some(Label::new()),
            unwind_ctx: None,
        });
        emit_jump(
            &mut self.code,
            Opcode::JumpIfFalsy,
            &mut self.breakables.last_mut().unwrap().breaks,
        );
        self.stmt(body)?;
        self.breakables
            .last_mut()
            .unwrap()
            .continues
            .as_mut()
            .unwrap()
            .bind(&self.code);
        let mut back = Label::new();
        emit_jump(&mut self.code, Opcode::JumpLoop, &mut back);
        self.bind_loop_end(back, head);
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
            let mut loop_ctx: Option<u32> = None;
            let ctx_save = if slots > 0 {
                emit(&mut g.code, Opcode::CreateBlockContext, &[slots]);
                let save = g.reserve_temp();
                let lc = g.reserve_temp();
                emit(&mut g.code, Opcode::Store, &[lc]);
                emit(&mut g.code, Opcode::PushContext, &[save]);
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
            let copies: Vec<u32>;
            let mut iter_ctx: Option<u32> = None;
            if per_iteration {
                copies = (0..slots).map(|_| g.reserve_temp()).collect();
                let iter_ctx_reg = g.reserve_temp();
                // the first iteration starts from a copy of the head
                // context: values copied out, fresh sibling pushed
                g.emit_iteration_context_copy(Some(iter_ctx_reg), slots, &copies, ctx_save);
                iter_ctx = Some(iter_ctx_reg);
            } else {
                copies = Vec::new();
            }
            // head: cond?; JumpIfFalsy breaks; body; continues; update; back-edge
            let head = g.code.len();
            g.breakables.push(Breakable {
                labels,
                breaks: Label::new(),
                continues: Some(Label::new()),
                unwind_ctx: ctx_save,
            });
            if let Some(cond) = &s.test {
                g.expr(cond)?;
                emit_jump(
                    &mut g.code,
                    Opcode::JumpIfFalsy,
                    &mut g.breakables.last_mut().unwrap().breaks,
                );
            }
            g.stmt(&s.body)?;
            g.breakables
                .last_mut()
                .unwrap()
                .continues
                .as_mut()
                .unwrap()
                .bind(&g.code);
            if let Some(iter_ctx) = iter_ctx {
                // absolute restore to this iteration's context, then copy
                // its values into a fresh sibling for the next iteration
                // (ES 14.7.5.4: the copy precedes the update)
                emit(&mut g.code, Opcode::PopContext, &[iter_ctx]);
                g.emit_iteration_context_copy(Some(iter_ctx), slots, &copies, ctx_save);
            } else if let Some(lc) = loop_ctx {
                emit(&mut g.code, Opcode::PopContext, &[lc]);
            }
            if let Some(next) = &s.update {
                g.expr(next)?;
            }
            let mut back = Label::new();
            emit_jump(&mut g.code, Opcode::JumpLoop, &mut back);
            g.bind_loop_end(back, head);
            // absolute restore: break may arrive from arbitrary context
            // depth (labelled jumps past nested loop pops)
            if let Some(save) = ctx_save {
                emit(&mut g.code, Opcode::PopContext, &[save]);
            }
            Ok(())
        })
    }

    /// Copy a loop head's bindings into a fresh sibling context: read the
    /// slots out of the current context, pop to the shared outer, create
    /// a fresh context (holes) and write the values back in.
    fn emit_iteration_context_copy(
        &mut self,
        iter_ctx: Option<u32>,
        count: u32,
        copies: &[u32],
        ctx_save: Option<u32>,
    ) {
        for slot in 0..count {
            emit(&mut self.code, Opcode::LoadContextSlot, &[slot, 0]);
            emit(&mut self.code, Opcode::Store, &[copies[slot as usize]]);
        }
        let save = ctx_save.expect("per-iteration loops own a context");
        emit(&mut self.code, Opcode::PopContext, &[save]);
        emit(&mut self.code, Opcode::CreateBlockContext, &[count]);
        if let Some(iter_ctx) = iter_ctx {
            emit(&mut self.code, Opcode::Store, &[iter_ctx]);
        }
        emit(&mut self.code, Opcode::PushContext, &[save]);
        for slot in 0..count {
            emit(&mut self.code, Opcode::Load, &[copies[slot as usize]]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
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
                    emit(&mut self.code, Opcode::PopContext, &[reg]);
                }
            }
            let state = &mut self.breakables[idx];
            emit_jump(&mut self.code, Opcode::Jump, &mut state.breaks);
            return Ok(());
        }
        let idx = match label {
            None => self.breakables.iter().rposition(|b| b.continues.is_some()),
            Some(name) => self.breakables.iter().rposition(|b| {
                b.labels.iter().any(|l| l == name) && b.continues.is_some()
            }),
        };
        let Some(idx) = idx else {
            return self.err(span, "continue outside a loop");
        };
        for b in self.breakables[idx + 1..].iter().rev() {
            if let Some(reg) = b.unwind_ctx {
                emit(&mut self.code, Opcode::PopContext, &[reg]);
            }
        }
        let state = &mut self.breakables[idx];
        emit_jump(
            &mut self.code,
            Opcode::Jump,
            state.continues.as_mut().unwrap(),
        );
        Ok(())
    }

    fn emit_switch(&mut self, s: &SwitchStatement<'_>) -> Result<(), CompileError> {
        let labels = self.take_labels();
        // evaluate the discriminant once into a temp
        self.expr(&s.discriminant)?;
        let d = self.push_value();
        self.breakables.push(Breakable {
            labels,
            breaks: Label::new(),
            continues: None,
            unwind_ctx: None,
        });

        let cases = &s.cases;
        let mut bodies: Vec<Label> = (0..cases.len()).map(|_| Label::new()).collect();
        let mut default_idx = None;

        for (i, case) in cases.iter().enumerate() {
            match &case.test {
                Some(test) => {
                    self.expr(test)?;
                    let t = self.push_value();
                    emit(&mut self.code, Opcode::Load, &[d]);
                    emit(&mut self.code, Opcode::EqualStrict, &[t]);
                    self.pop_value();
                    emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut bodies[i]);
                }
                None => default_idx = Some(i),
            }
        }

        // no case matched: the default body, or past the switch
        let mut end = Label::new();
        match default_idx {
            Some(i) => emit_jump(&mut self.code, Opcode::Jump, &mut bodies[i]),
            None => emit_jump(&mut self.code, Opcode::Jump, &mut end),
        }

        // bodies execute in order; fallthrough is just sequential layout
        for (i, case) in cases.iter().enumerate() {
            bodies[i].bind(&self.code);
            bodies[i].patch_all(&mut self.code);
            for st in &case.consequent {
                self.stmt(st)?;
            }
        }

        let (mut breaks, _) = self.end_breakable();
        breaks.bind(&self.code);
        breaks.patch_all(&mut self.code);
        end.bind(&self.code);
        end.patch_all(&mut self.code);
        self.pop_value(); // discriminant
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
            let try_ctx = g.reserve_temp();
            emit(&mut g.code, Opcode::LoadContext, &[]);
            emit(&mut g.code, Opcode::Store, &[try_ctx]);
            let try_start = g.code.len();
            for st in &s.block.body {
                g.stmt(st)?;
            }
            let try_end = g.code.len();

            let mut end = Label::new();
            emit_jump(&mut g.code, Opcode::Jump, &mut end);

            // handler entry: the exception arrives in the accumulator
            let handler_pc = g.code.len();
            emit(&mut g.code, Opcode::PopContext, &[try_ctx]);
            if let Some(h) = &s.handler {
                if let Some(param) = &h.param {
                    match &param.pattern {
                        BindingPattern::ObjectPattern(_) | BindingPattern::ArrayPattern(_) => {
                            let value = g.push_value();
                            g.emit_binding_pattern(&param.pattern, value)?;
                            g.pop_value();
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

            g.handlers.push(ir::HandlerEntry {
                try_start,
                try_end,
                handler_pc,
            });
            end.bind(&g.code);
            end.patch_all(&mut g.code);
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
            Some(slot) => emit(&mut self.code, Opcode::LoadContextSlot, &[slot, 0]),
            None => emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]),
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
            emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            emit(&mut self.code, Opcode::Return, &[]);
            return Ok(());
        };
        self.expr(v)?;
        let t = self.push_value();
        // acc === undefined → return this
        self.emit_load_undefined();
        let u = self.push_value();
        emit(&mut self.code, Opcode::Load, &[t]);
        emit(&mut self.code, Opcode::EqualStrict, &[u]);
        let mut is_obj = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut is_obj);
        // undefined → return this (context still pushed for the slot read)
        self.emit_this_load_own();
        self.emit_this_initialized_check();
        emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
        emit(&mut self.code, Opcode::Return, &[]);
        is_obj.bind(&self.code);
        is_obj.patch_all(&mut self.code);
        emit(&mut self.code, Opcode::Load, &[t]);
        emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
        emit(&mut self.code, Opcode::Return, &[]);
        self.pop_value(); // u
        self.pop_value(); // t
        Ok(())
    }

    fn emit_function_body(&mut self) -> Result<(), CompileError> {
        // layout: assign this function's slots before any emission
        let layout = self.c.assign_function_slots(self.fid);
        self.c.layouts[self.fid.0 as usize] = layout;

        let layout = &self.c.layouts[self.fid.0 as usize];
        // the script (and each eval compilation) tracks its completion
        // value in a dedicated register between ctx_save and the temps
        self.completion = (self.fid.0 == 0).then_some(layout.register_count + 1);
        self.ctx_save = layout.register_count as i32;
        self.reg_base = layout.register_count + 1 + u32::from(self.completion.is_some());

        // prologue: one context per function (uniform chain), pushed onto
        // the frame context; locals below reg_base are born as the hole.
        // constants[0] is the context's shared ScopeInfo (slot names)
        let names = layout.slot_names.clone();
        self.add_constant(Constant::ContextNames(names));
        emit(&mut self.code, Opcode::CreateFunctionContext, &[0]);
        emit(&mut self.code, Opcode::PushContext, &[self.ctx_save as u32]);

        // parameters: non-simple lists (any default / pattern / rest)
        // stage the incoming arguments, hole-fill the parameter registers,
        // and initialize each binding in order (TDZ until its turn, ES
        // 10.2.11); simple lists only copy context-allocated (captured)
        // parameters into their slots
        let params: Vec<(PatTarget<'_, '_>, Option<&Expression<'_>>, bool)> = self
            .c
            .facts
            .functions[self.fid.0 as usize]
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
            let staged_base = self.reserve_temps(n);
            // stage the incoming arguments (missing ones arrive as
            // undefined through frame padding)
            for i in 0..n {
                emit(&mut self.code, Opcode::Load, &[(-(i as i32 + 2)) as u32]);
                emit(&mut self.code, Opcode::Store, &[staged_base + i]);
            }
            // rest arrays capture the frame's raw argument list — build
            // them before the registers are hole-filled
            for (i, (_, _, rest)) in params.iter().enumerate() {
                if *rest {
                    let i = i as u32;
                    self.emit_runtime_call(
                        bytecode::RuntimeFn::CreateRestParameter,
                        1,
                        |g, b| g.stage_smi(i, b),
                    );
                    emit(&mut self.code, Opcode::Store, &[staged_base + i]);
                }
            }
            // parameters start in their TDZ
            for i in 0..n {
                emit(&mut self.code, Opcode::LoadHole, &[]);
                emit(&mut self.code, Opcode::Store, &[(-(i as i32 + 2)) as u32]);
            }
            // left-to-right initialization
            for (i, (target, default, _)) in params.iter().enumerate() {
                let reg = (-(i as i32 + 2)) as u32;
                let staged = staged_base + i as u32;
                if let Some(default) = default {
                    emit(&mut self.code, Opcode::Load, &[staged]);
                    let mut skip = Label::new();
                    emit_jump(&mut self.code, Opcode::JumpIfNotUndefined, &mut skip);
                    self.expr(default)?;
                    emit(&mut self.code, Opcode::Store, &[staged]);
                    skip.bind(&self.code);
                    skip.patch_all(&mut self.code);
                }
                // InitializeBinding: the register (and, when captured, the
                // context slot) receives the value
                emit(&mut self.code, Opcode::Load, &[staged]);
                emit(&mut self.code, Opcode::Store, &[reg]);
                if let PatTarget::Binding(BindingPattern::BindingIdentifier(b)) = target
                    && let Some(sym) = b.symbol_id.get()
                    && let Some(Slot::Ctx { slot, .. }) = self.c.slots.get(&sym)
                {
                    let slot = *slot;
                    emit(&mut self.code, Opcode::Load, &[reg]);
                    emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
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
            self.next_temp -= n; // staged parameter window
        } else {
            // context-allocated parameters: copy the argument into its
            // slot (captured params and direct-eval scopes force params
            // to contexts)
            for (target, _, _) in &params {
                let PatTarget::Binding(BindingPattern::BindingIdentifier(b)) = target else {
                    unreachable!("simple lists have identifier params")
                };
                let Some(sym) = b.symbol_id.get() else { continue };
                if let Some(Slot::Ctx { slot, .. }) = self.c.slots.get(&sym) {
                    let slot = *slot;
                    let index = self.c.facts.param_symbols[&sym];
                    let reg = (-(index as i32 + 2)) as u32;
                    emit(&mut self.code, Opcode::Load, &[reg]);
                    emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
                }
            }
        }

        // store the receiver into the hidden this-slot when a nested
        // arrow captures it
        if let Some(slot) = self.c.layouts[self.fid.0 as usize].this_slot {
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }
        // expose new.target / the running closure to nested arrows
        // (arrow-delegated super() and arrow new.target reads)
        if let Some(slot) = self.c.layouts[self.fid.0 as usize].new_target_slot {
            emit(&mut self.code, Opcode::LoadNewTarget, &[]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }
        if let Some(slot) = self.c.layouts[self.fid.0 as usize].this_function_slot {
            emit(&mut self.code, Opcode::LoadCurrentClosure, &[]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }

        let kind = self.c.facts.functions[self.fid.0 as usize].kind;

        // the synthesized default derived constructor forwards every
        // argument to super() and returns the bound this (ES 15.7.13)
        if kind == FnKind::DefaultDerivedCtor {
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::ConstructSuperAllArgs as u32, 0, 0],
            );
            emit(&mut self.code, Opcode::Store, &[(-1i32) as u32]);
            if self.fn_has_instance_fields(self.fid) {
                // InitializeInstanceElements on the bound this:
                // native(ctor, instance)
                self.with_temps(|g| {
                    emit(&mut g.code, Opcode::LoadCurrentClosure, &[]);
                    let ctor = g.push_value();
                    g.emit_this_load_own();
                    g.push_value();
                    emit(
                        &mut g.code,
                        Opcode::CallRuntime,
                        &[bytecode::RuntimeFn::InitInstanceFields as u32, ctor, 2],
                    );
                    Ok(())
                })?;
            }
            emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
            emit(&mut self.code, Opcode::Return, &[]);
            return Ok(());
        }

        // base class constructors run their instance field initializers
        // right after the receiver exists (ES 7.3.33, before the body)
        if kind == FnKind::BaseClassCtor && self.fn_has_instance_fields(self.fid) {
            self.with_temps(|g| {
                emit(&mut g.code, Opcode::LoadCurrentClosure, &[]);
                let ctor = g.push_value();
                g.emit_this_load_own();
                g.push_value();
                emit(
                    &mut g.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::InitInstanceFields as u32, ctor, 2],
                );
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
                        let idx = self.add_constant(Constant::String(name.into_bytes()));
                        let feedback = self.feedback_slot();
                        self.emit_load_undefined();
                        emit(&mut self.code, Opcode::StoreGlobal, &[idx, feedback]);
                    }
                    Slot::Param { index, .. } => {
                        self.emit_load_undefined();
                        emit(
                            &mut self.code,
                            Opcode::Store,
                            &[(-(index as i32 + 2)) as u32],
                        );
                    }
                    Slot::Local { reg, .. } => {
                        self.emit_load_undefined();
                        emit(&mut self.code, Opcode::Store, &[reg]);
                    }
                    Slot::Ctx { slot, .. } => {
                        self.emit_load_undefined();
                        emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
                    }
                    Slot::CtxAt { .. } => unreachable!("vars never live in class/for contexts"),
                }
            }
        }
        if let Some(fns) = self.c.facts.hoist_fns.get(&self.fid).cloned() {
            for (sym, fid) in fns {
                let name = self.c.scoping.symbol_name(sym).to_string();
                let idx = self.add_constant(Constant::Callable(IrFunctionId(fid.0)));
                emit(&mut self.code, Opcode::CreateClosure, &[idx]);
                self.store_symbol(sym, &name)?;
            }
        }

        // the script's completion value starts as undefined; only
        // value-producing statements overwrite it (see `stmt`)
        if let Some(completion) = self.completion {
            self.emit_load_undefined();
            emit(&mut self.code, Opcode::Store, &[completion]);
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
                    emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                    emit(&mut self.code, Opcode::Return, &[]);
                    return Ok(());
                }
                FnBody::Empty => {
                    self.emit_load_undefined();
                    emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                    emit(&mut self.code, Opcode::Return, &[]);
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
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                emit(&mut self.code, Opcode::Return, &[]);
                return Ok(());
            }
            FnBody::FieldInit(_) | FnBody::Empty => {}
        }

        // fallthrough: the script yields its completion value; ordinary
        // functions return undefined; derived constructors return `this`
        // (initialization checked — super() must have run)
        match self.completion {
            Some(completion) => {
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                emit(&mut self.code, Opcode::Load, &[completion]);
            }
            None if self.is_derived_ctor() => {
                self.emit_this_load_own();
                self.emit_this_initialized_check();
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            }
            None => {
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                self.emit_load_undefined();
            }
        }
        emit(&mut self.code, Opcode::Return, &[]);
        Ok(())
    }
}
