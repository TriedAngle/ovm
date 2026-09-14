//! The AST walker: one naive pass per function, Ignition-style.
//!
//! Expression results live on an implicit operand stack of frame registers
//! above the resolver's per-function layout (`reg_base`). Registers grow
//! monotonically during expression evaluation and shrink LIFO when parents
//! consume them, so call argument lists are naturally contiguous.
//!
//! Contexts: every function creates one context object holding all of its
//! context-allocated (captured or eval-forced) slots, chained to the
//! captured context of its closure. `LoadContextSlot` depth is therefore
//! the lexical function-nesting distance between the use site and the
//! declaration's owning function.

use bytecode::{Opcode, PropertyFlags, emit};
use parser::{
    Ast, ClassId, FunctionId, Node, NodeId, PropKind, Resolution, Resolved, ScopeId, ScopeKind,
    Symbol, TokenKind, VarKind,
};

use crate::label::Label;
use crate::{CompileError, CompiledFunction, CompiledScript, Constant, HandlerEntry};

/// Emit a forced-wide relative jump to `label`; the offset is patched once
/// the label is bound.
fn emit_jump(code: &mut Vec<u8>, op: Opcode, label: &mut Label) {
    debug_assert!(matches!(
        op,
        Opcode::Jump
            | Opcode::JumpLoop
            | Opcode::JumpIfTruthy
            | Opcode::JumpIfFalsy
            | Opcode::JumpIfNotUndefined
    ));
    let pc = code.len();
    code.push(Opcode::Wide as u8);
    code.push(op as u8);
    code.extend_from_slice(&[0, 0]);
    label.patch_here(pc);
}

/// A breakable statement: loops add a continue target, switches don't.
struct Breakable {
    /// the statement's label set (ES 14.13.1): `break`/`continue` with a
    /// label target the innermost breakable whose set contains it
    labels: Vec<Symbol>,
    breaks: Label,
    continues: Option<Label>,
    /// Loops owning a head context: the register holding the context
    /// current before the loop statement. A break/continue targeting an
    /// outer breakable from inside such a loop bypasses its context pops,
    /// so the jump must unwind past it (`PopContext [unwind_ctx]`).
    unwind_ctx: Option<u32>,
}

/// A store target ready for the final store: the object (and computed key)
/// evaluated into live registers, plus the name constant index.
enum StoreTarget {
    Named {
        obj: u32,
        name_idx: u32,
    },
    Keyed {
        obj: u32,
        key: u32,
    },
    /// `this.#x = v`: the private name symbol loaded into a register
    PrivateKeyed {
        obj: u32,
        key: u32,
    },
    /// `super.x = v` / `super[k] = v`: receiver (this) and home object in
    /// live registers
    SuperNamed {
        recv: u32,
        home: u32,
        name_idx: u32,
    },
    SuperKeyed {
        recv: u32,
        home: u32,
        key: u32,
    },
}

/// Name hint for NamedEvaluation in pattern defaults (ES 8.4.3): the
/// property key (or array index) an anonymous function is named after.
enum NameHint {
    Const(u32),
    Reg(u32),
    None,
}

struct FunctionGen<'a> {
    ast: &'a Ast,
    resolved: &'a Resolved,
    fid: FunctionId,
    code: Vec<u8>,
    constants: Vec<Constant>,
    handlers: Vec<HandlerEntry>,
    /// first temp register: resolver locals + 1 (context-save slot)
    reg_base: u32,
    /// register holding the pushed-over context for the prologue/epilogue
    ctx_save: i32,
    /// script completion-value register (scripts/eval only): updated by every
    /// expression statement, read back by the epilogue (ES 13.2.13,
    /// UpdateEmpty keeps the previous value for non-value statements)
    completion: Option<u32>,
    /// temps currently allocated (monotonic within an expression)
    next_temp: u32,
    max_temps: u32,
    /// lexical scope stack, innermost last; for decl lookups
    scopes: Vec<ScopeId>,
    breakables: Vec<Breakable>,
}

/// The value of a body that is exactly `return <expr>;`, if it is.
fn single_return_value(ast: &Ast, body: NodeId) -> Option<NodeId> {
    let Node::Block { stmts } = *ast.node(body) else {
        return None;
    };
    let [only] = ast.list_items(stmts) else {
        return None;
    };
    match *ast.node(*only) {
        Node::Return { value: Some(v) } => Some(v),
        _ => None,
    }
}

pub fn generate(ast: &Ast, resolved: &Resolved) -> Result<CompiledScript, CompileError> {
    let mut functions = Vec::with_capacity(ast.function_count());
    for fid in 0..ast.function_count() {
        let fid = FunctionId(fid as u32);
        let mut generator = FunctionGen::new(ast, resolved, fid);
        generator.emit_function_body()?;
        debug_assert!(
            generator.next_temp == 0,
            "unbalanced temp allocation in {:?} ({:?})",
            fid,
            ast.function(fid).kind
        );
        functions.push(generator.finish());
    }
    Ok(CompiledScript { functions })
}

impl<'a> FunctionGen<'a> {
    fn new(ast: &'a Ast, resolved: &'a Resolved, fid: FunctionId) -> Self {
        Self {
            ast,
            resolved,
            fid,
            code: Vec::new(),
            constants: Vec::new(),
            handlers: Vec::new(),
            reg_base: 0,
            ctx_save: 0,
            completion: None,
            next_temp: 0,
            max_temps: 0,
            scopes: Vec::new(),
            breakables: Vec::new(),
        }
    }

    fn finish(self) -> CompiledFunction {
        let info = self.ast.function(self.fid);
        CompiledFunction {
            bytecode: self.code,
            constants: self.constants,
            name: info.name.map(|name| self.ast.symbol(name).to_vec()),
            formal_parameter_count: info.params.len() as u32,
            formal_length: info.formal_length,
            kind: info.kind,
            strict: info.strict,
            register_count: self.reg_base + self.max_temps,
            handlers: self.handlers,
        }
    }

    fn err<T>(&self, node: NodeId, feature: &'static str) -> Result<T, CompileError> {
        Err(CompileError::new(self.ast.span(node), feature))
    }

    // -- temps ------------------------------------------------------------

    /// Move the accumulator into a fresh live register.
    fn push_value(&mut self) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += 1;
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(&mut self.code, Opcode::Store, &[r]);
        r
    }

    /// Drop the most recently pushed live register.
    fn pop_value(&mut self) {
        debug_assert!(self.next_temp > 0, "temp underflow");
        self.next_temp -= 1;
    }

    /// Reserve one temp register without storing the accumulator.
    fn reserve_temp(&mut self) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += 1;
        self.max_temps = self.max_temps.max(self.next_temp);
        r
    }

    /// Reserve `n` contiguous temp registers (call-argument windows),
    /// returning the first.
    fn reserve_temps(&mut self, n: u32) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += n;
        self.max_temps = self.max_temps.max(self.next_temp);
        r
    }

    /// Bracket a temp-register region: everything reserved inside `f` is
    /// released on every exit path (`?` early returns included), with the
    /// high-water mark retained. Replaces hand-counted `next_temp -= n`
    /// release arithmetic — a miscounted release silently overlaps
    /// registers and corrupts code, the bracket cannot.
    fn with_temps<F, T>(&mut self, f: F) -> Result<T, CompileError>
    where
        F: FnOnce(&mut Self) -> Result<T, CompileError>,
    {
        let mark = self.next_temp;
        let result = f(self);
        self.next_temp = mark;
        result
    }

    /// Bracket a name-resolution scope region: the scope is on the stack
    /// for `f` and off it on every exit path.
    fn scoped<F, T>(&mut self, scope: ScopeId, f: F) -> Result<T, CompileError>
    where
        F: FnOnce(&mut Self) -> Result<T, CompileError>,
    {
        self.scopes.push(scope);
        let result = f(self);
        self.scopes.pop();
        result
    }

    // -- constants ----------------------------------------------------------

    fn add_constant(&mut self, c: Constant) -> u32 {
        self.constants.push(c);
        (self.constants.len() - 1) as u32
    }

    fn emit_load_constant(&mut self, c: Constant) {
        let idx = self.add_constant(c);
        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
    }

    fn name_constant(&mut self, key: NodeId) -> Result<u32, CompileError> {
        let sym = match *self.ast.node(key) {
            Node::Identifier { sym } => sym,
            Node::StringLiteral(sym) => sym,
            _ => return self.err(key, "non-string property names"),
        };
        Ok(self.add_constant(Constant::String(self.ast.symbol(sym).to_vec())))
    }

    /// ES 8.4.2 IsAnonymousFunctionDefinition: an unnamed function or
    /// class expression in a NamedEvaluation position gets a name.
    fn is_anon_function(&self, node: NodeId) -> bool {
        match *self.ast.node(node) {
            Node::FunctionExpr { function } => self.ast.function(function).name.is_none(),
            Node::ClassExpr { class } => self.ast.class(class).name.is_none(),
            _ => false,
        }
    }

    /// ES 8.4.4 SetFunctionName with a static string (NamedEvaluation).
    fn emit_set_name_const(&mut self, bytes: &[u8]) {
        let idx = self.add_constant(Constant::String(bytes.to_vec()));
        emit(&mut self.code, Opcode::SetFunctionNameConst, &[idx]);
    }

    /// SetFunctionName from a key node usable for naming (StringLiteral or
    /// NumberLiteral keys).
    fn emit_set_name_for_key_node(&mut self, key: NodeId) {
        if let Node::StringLiteral(sym) = *self.ast.node(key) {
            let bytes = self.ast.symbol(sym).to_vec();
            self.emit_set_name_const(&bytes);
        }
    }

    // -- scopes / resolutions -------------------------------------------------

    /// Innermost scope (from the current position) declaring `name`.
    fn find_decl_scope(&self, name: Symbol) -> Option<ScopeId> {
        self.scopes
            .iter()
            .rev()
            .find(|&&s| self.resolved.resolution_for_decl(s, name).is_some())
            .copied()
    }

    fn store_resolution(&mut self, res: Resolution) {
        match res {
            Resolution::Param { index, .. } => emit(
                &mut self.code,
                Opcode::Store,
                &[(-(index as i32 + 2)) as u32],
            ),
            Resolution::Local { reg, .. } => emit(&mut self.code, Opcode::Store, &[reg]),
            Resolution::Context { slot, depth, .. } => {
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth])
            }
            Resolution::GlobalObject => {
                unreachable!("global stores need the name; use store_global")
            }
            Resolution::Dynamic => unreachable!("dynamic resolutions are rejected on load"),
            Resolution::This { .. } => unreachable!("this has no store target"),
            Resolution::NewTarget { .. } => unreachable!("new.target has no store target"),
            Resolution::SuperCall { .. } => unreachable!("super() has no store target"),
            Resolution::Super { .. } => unreachable!("super has no store target"),
        }
    }

    /// Store the accumulator into a declaration's slot. Unlike
    /// [`Self::store_resolution`] this can target the global object, which
    /// needs the name (REPL mode puts script-scope decls there).
    fn store_decl(&mut self, res: Resolution, name: Symbol) {
        match res {
            Resolution::GlobalObject => {
                let idx = self.add_constant(Constant::String(self.ast.symbol(name).to_vec()));
                emit(&mut self.code, Opcode::StoreGlobal, &[idx, 0]);
            }
            other => self.store_resolution(other),
        }
    }

    fn store_name(&mut self, node: NodeId, name: Symbol) -> Result<(), CompileError> {
        let res = self
            .resolved
            .resolution(node)
            .expect("identifier must be resolved");
        match res {
            Resolution::GlobalObject => {
                let idx = self.add_constant(Constant::String(self.ast.symbol(name).to_vec()));
                emit(&mut self.code, Opcode::StoreGlobal, &[idx, 0]);
                Ok(())
            }
            Resolution::Dynamic => {
                let idx = self.add_constant(Constant::String(self.ast.symbol(name).to_vec()));
                emit(&mut self.code, Opcode::StoreDynamicName, &[idx]);
                Ok(())
            }
            other => {
                self.store_resolution(other);
                Ok(())
            }
        }
    }

    // -- loads ----------------------------------------------------------------

    fn emit_identifier(&mut self, node: NodeId, name: Symbol) -> Result<(), CompileError> {
        let res = self
            .resolved
            .resolution(node)
            .expect("identifier must be resolved");
        match res {
            Resolution::Param { index, hole_check } => {
                emit(
                    &mut self.code,
                    Opcode::Load,
                    &[(-(index as i32 + 2)) as u32],
                );
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
                Ok(())
            }
            Resolution::Local { reg, hole_check } => {
                emit(&mut self.code, Opcode::Load, &[reg]);
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
                Ok(())
            }
            Resolution::Context {
                slot,
                depth,
                hole_check,
            } => {
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                if hole_check {
                    emit(&mut self.code, Opcode::ThrowReferenceErrorIfHole, &[]);
                }
                Ok(())
            }
            Resolution::GlobalObject => {
                let idx = self.add_constant(Constant::String(self.ast.symbol(name).to_vec()));
                emit(&mut self.code, Opcode::LoadGlobal, &[idx, 0]);
                Ok(())
            }
            Resolution::Dynamic => {
                let idx = self.add_constant(Constant::String(self.ast.symbol(name).to_vec()));
                emit(&mut self.code, Opcode::LoadDynamicName, &[idx]);
                Ok(())
            }
            Resolution::This { .. } => unreachable!("this is not an identifier load"),
            Resolution::NewTarget { .. } => unreachable!("new.target is not an identifier load"),
            Resolution::SuperCall { .. } => unreachable!("super() is not an identifier load"),
            Resolution::Super { .. } => unreachable!("super is not an identifier load"),
        }
    }

    /// Logical negation of the accumulator (arbitrary value → boolean).
    fn emit_not(&mut self) {
        let mut is_truthy = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut is_truthy);
        self.emit_load_constant(Constant::Boolean(true));
        let mut end = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut end);
        is_truthy.bind(&self.code);
        is_truthy.patch_all(&mut self.code);
        self.emit_load_constant(Constant::Boolean(false));
        end.bind(&self.code);
        end.patch_all(&mut self.code);
    }

    /// Load a private-name symbol from its class-context slot.
    fn emit_private_key_load(&mut self, key: NodeId) -> Result<(), CompileError> {
        match self.resolved.resolution(key) {
            Some(Resolution::Context { slot, depth, .. }) => {
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                Ok(())
            }
            _ => self.err(key, "private name outside its class"),
        }
    }

    /// Load `this` of `owner` (the nearest non-arrow function). Inside a
    /// derived constructor the value may still be the uninitialized hole:
    /// every access must throw "super not called" (ES 10.2.2).
    fn emit_this_for(&mut self, owner: FunctionId, depth: u32) {
        if owner != self.fid {
            let slot = self
                .resolved
                .layout(owner)
                .this_slot
                .expect("owner captures this");
            emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
        } else if let Some(slot) = self.resolved.layout(self.fid).this_slot {
            // captured `this` lives in the context (arrows may rebind it
            // through a delegated super()); the prologue seeded the slot
            emit(&mut self.code, Opcode::LoadContextSlot, &[slot, 0]);
        } else {
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
        }
        if self.ast.function(owner).kind.is_derived_class_constructor() {
            emit(&mut self.code, Opcode::ThrowSuperNotCalledIfHole, &[]);
        }
    }

    // -- expressions ------------------------------------------------------------

    fn expr(&mut self, node: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(node) {
            Node::NumberLiteral(f) => {
                // (2^53 − 1: largest exactly-representable integer)
                const MAX_EXACT_INT: f64 = 9007199254740991.0;
                let is_int = f.fract() == 0.0 && !(f == 0.0 && f.is_sign_negative());
                if is_int && f >= i16::MIN as f64 && f <= i16::MAX as f64 {
                    emit(&mut self.code, Opcode::LoadSmi, &[f as i32 as u32]);
                } else if is_int && f.abs() <= MAX_EXACT_INT {
                    // integer literals too big for the operand go through the
                    // constant pool
                    self.emit_load_constant(Constant::Smi(f as i64));
                } else {
                    self.emit_load_constant(Constant::Float(f));
                }
                Ok(())
            }
            Node::StringLiteral(sym) => {
                self.emit_load_constant(Constant::String(self.ast.symbol(sym).to_vec()));
                Ok(())
            }
            Node::BigIntLiteral(_) => self.err(node, "BigInt literals"),
            Node::BoolLiteral(b) => {
                self.emit_load_constant(Constant::Boolean(b));
                Ok(())
            }
            Node::NullLiteral => {
                self.emit_load_constant(Constant::Null);
                Ok(())
            }
            Node::Identifier { sym } => self.emit_identifier(node, sym),
            Node::This => {
                match self.resolved.resolution(node) {
                    Some(Resolution::This { owner, depth }) => self.emit_this_for(owner, depth),
                    _ => self.emit_this_for(self.fid, 0),
                }
                Ok(())
            }
            Node::Unary { op, expr } => self.emit_unary(node, op, expr),
            Node::Update { op, prefix, target } => self.emit_update(node, op, prefix, target),
            Node::Binary { op, lhs, rhs } => self.emit_binary(node, op, lhs, rhs),
            Node::Assign { op, target, value } => self.emit_assign(node, op, target, value),
            Node::Conditional { cond, then, else_ } => self.emit_conditional(cond, then, else_),
            Node::Call { callee, args } => self.emit_call(node, callee, args),
            Node::New { callee, args } => self.emit_new(node, callee, args),
            Node::Property {
                object,
                key,
                computed,
            } => self.emit_property_load(object, key, computed),
            Node::ArrayLiteral { elements } => self.emit_array_literal(node, elements),
            Node::ObjectLiteral { props } => self.emit_object_literal(node, props),
            Node::Spread { .. } => self.err(node, "spread"),
            Node::FunctionExpr { function } => {
                let idx = self.add_constant(Constant::Callable(function));
                emit(&mut self.code, Opcode::CreateClosure, &[idx]);
                Ok(())
            }
            Node::ClassExpr { class } => self.emit_class(node, class),
            Node::SuperProperty { key, computed, .. } => {
                self.emit_super_property_load(node, key, computed)
            }
            Node::SuperCall { args } => self.emit_super_call(node, args),
            Node::NewTarget => {
                match self.resolved.resolution(node) {
                    Some(Resolution::NewTarget { owner, depth }) if owner != self.fid => {
                        // arrow-delegated: the owner's context slot
                        let slot = self
                            .resolved
                            .layout(owner)
                            .new_target_slot
                            .expect("arrow new.target forces the slot");
                        emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                    }
                    _ => emit(&mut self.code, Opcode::LdaNewTarget, &[]),
                }
                Ok(())
            }
            Node::Hole => self.err(node, "array elision outside array literals"),
            Node::PrivateName { .. } => self.err(
                node,
                "private names are only valid as `obj.#x` or `#x in obj`",
            ),
            Node::ObjectProperty { .. } => unreachable!("handled by object literal codegen"),
            _ => self.err(node, "statement in expression position"),
        }
    }

    fn emit_unary(
        &mut self,
        node: NodeId,
        op: TokenKind,
        expr: NodeId,
    ) -> Result<(), CompileError> {
        match op {
            TokenKind::Bang => {
                self.expr(expr)?;
                self.emit_not();
                Ok(())
            }
            TokenKind::Minus => {
                self.expr(expr)?;
                emit(&mut self.code, Opcode::Negate, &[]);
                Ok(())
            }
            TokenKind::Plus => {
                emit(&mut self.code, Opcode::LoadSmi, &[0]);
                let zero = self.push_value();
                self.expr(expr)?;
                emit(&mut self.code, Opcode::Sub, &[zero]);
                self.pop_value();
                Ok(())
            }
            TokenKind::Typeof => {
                // `typeof` on an unresolved global yields "undefined"
                // instead of throwing
                if let Node::Identifier { sym } = *self.ast.node(expr) {
                    if matches!(
                        self.resolved.resolution(expr),
                        Some(Resolution::GlobalObject)
                    ) {
                        let idx =
                            self.add_constant(Constant::String(self.ast.symbol(sym).to_vec()));
                        emit(&mut self.code, Opcode::LoadGlobalNoThrow, &[idx, 0]);
                        emit(&mut self.code, Opcode::TestTypeof, &[]);
                        return Ok(());
                    }
                }
                self.expr(expr)?;
                emit(&mut self.code, Opcode::TestTypeof, &[]);
                Ok(())
            }
            TokenKind::Void => {
                self.expr(expr)?;
                self.emit_load_constant(Constant::Undefined);
                Ok(())
            }
            TokenKind::Tilde => self.err(node, "bitwise not"),
            TokenKind::Delete => self.emit_delete(node, expr),
            _ => self.err(node, "unary operator"),
        }
    }

    /// `delete` (ES 13.5.1.2). The operand's *reference* is what gets
    /// deleted — never its value: member references evaluate only base
    /// and key (no property load, so getters cannot run), unqualified
    /// sloppy identifiers delete on the global object (declared bindings
    /// are declarative and statically fold to `false`), and non-reference
    /// operands evaluate for side effects and yield `true`.
    fn emit_delete(&mut self, node: NodeId, expr: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(expr) {
            Node::Property {
                object,
                key,
                computed,
            } => {
                self.expr(object)?;
                let base = self.push_value();
                if computed {
                    self.expr(key)?;
                } else {
                    let idx = self.name_constant(key)?;
                    emit(&mut self.code, Opcode::LoadConstant, &[idx]);
                }
                self.push_value();
                let runtime_fn = if self.ast.function(self.fid).strict {
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
            Node::SuperProperty { key, computed, .. } => {
                // ReferenceError in both language modes (ES 13.5.1.2
                // step 4.c). Reference evaluation runs first —
                // GetThisBinding throws for an uninitialized `this`
                // before the key expression evaluates (ES 13.3.7.1) —
                // and the key is never coerced: delete-super fails
                // before any ToPropertyKey
                let Some(store) = self.prepare_super_parts(expr, key, computed)? else {
                    return self.err(node, "super property");
                };
                self.release_store(&store);
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::DeleteSuperProperty as u32, 0, 0],
                );
                Ok(())
            }
            Node::Identifier { sym } => {
                // strict sites are rejected at parse time; sloppy
                // declarative bindings cannot be deleted (false), free
                // names are global-object properties
                match self.resolved.resolution(expr) {
                    Some(Resolution::GlobalObject) | Some(Resolution::Dynamic) => {
                        let idx =
                            self.add_constant(Constant::String(self.ast.symbol(sym).to_vec()));
                        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
                        let name = self.push_value();
                        emit(
                            &mut self.code,
                            Opcode::CallRuntime,
                            &[bytecode::RuntimeFn::DeleteIdentifierSloppy as u32, name, 1],
                        );
                        self.pop_value();
                    }
                    _ => self.emit_load_constant(Constant::Boolean(false)),
                }
                Ok(())
            }
            Node::PrivateName { .. } => self.err(node, "delete of a private name"),
            _ => {
                // not a reference: side effects only, the result is true
                self.expr(expr)?;
                self.emit_load_constant(Constant::Boolean(true));
                Ok(())
            }
        }
    }

    fn emit_binary(
        &mut self,
        node: NodeId,
        op: TokenKind,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Result<(), CompileError> {
        match op {
            TokenKind::Plus => self.binary_arith(lhs, rhs, Opcode::Add),
            TokenKind::Minus => self.binary_arith(lhs, rhs, Opcode::Sub),
            TokenKind::Star => self.binary_arith(lhs, rhs, Opcode::Mul),
            TokenKind::Slash => self.binary_arith(lhs, rhs, Opcode::Div),
            TokenKind::Percent => self.binary_arith(lhs, rhs, Opcode::Mod),
            TokenKind::StarStar => self.binary_arith(lhs, rhs, Opcode::Exp),
            TokenKind::Pipe => self.binary_arith(lhs, rhs, Opcode::BitwiseOr),
            TokenKind::Caret => self.binary_arith(lhs, rhs, Opcode::BitwiseXor),
            TokenKind::Amp => self.binary_arith(lhs, rhs, Opcode::BitwiseAnd),
            TokenKind::Shl => self.binary_arith(lhs, rhs, Opcode::ShiftLeft),
            TokenKind::Shr => self.binary_arith(lhs, rhs, Opcode::ShiftRight),
            TokenKind::Ushr => self.binary_arith(lhs, rhs, Opcode::ShiftRightLogical),
            TokenKind::EqEqEq => self.binary_arith(lhs, rhs, Opcode::EqualStrict),
            TokenKind::EqEq => self.binary_arith(lhs, rhs, Opcode::Equal),
            TokenKind::NotEqEq => {
                self.binary_arith(lhs, rhs, Opcode::EqualStrict)?;
                self.emit_not();
                Ok(())
            }
            TokenKind::NotEq => {
                self.binary_arith(lhs, rhs, Opcode::Equal)?;
                self.emit_not();
                Ok(())
            }
            TokenKind::Lt => self.binary_arith(lhs, rhs, Opcode::LessThan),
            TokenKind::Gt => self.binary_arith(lhs, rhs, Opcode::GreaterThan),
            TokenKind::LtEq => self.binary_arith(lhs, rhs, Opcode::LessThanOrEqual),
            TokenKind::GtEq => self.binary_arith(lhs, rhs, Opcode::GreaterThanOrEqual),
            TokenKind::Instanceof => {
                // acc = lhs instanceof rhs: evaluate rhs first
                self.expr(rhs)?;
                let r = self.push_value();
                self.expr(lhs)?;
                emit(&mut self.code, Opcode::InstanceOf, &[r]);
                self.pop_value();
                Ok(())
            }
            TokenKind::AmpAmp => {
                self.expr(lhs)?;
                let t = self.push_value();
                let mut short = Label::new();
                emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut short);
                self.expr(rhs)?;
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
            TokenKind::OrOr => {
                self.expr(lhs)?;
                let t = self.push_value();
                let mut short = Label::new();
                emit_jump(&mut self.code, Opcode::JumpIfTruthy, &mut short);
                self.expr(rhs)?;
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
            TokenKind::Comma => {
                self.expr(lhs)?;
                self.expr(rhs)
            }
            TokenKind::In => {
                // `key in obj` (ES 14.11.2) or `#x in obj` (private, ES 13.3.9):
                // key in the accumulator, object in a register
                if matches!(self.ast.node(lhs), Node::PrivateName { .. }) {
                    // `#x in obj`: (key, obj) -> bool
                    self.emit_private_key_load(lhs)?;
                    let base = self.push_value();
                    self.expr(rhs)?;
                    self.push_value();
                    emit(
                        &mut self.code,
                        Opcode::CallRuntime,
                        &[bytecode::RuntimeFn::PrivateIn as u32, base, 2],
                    );
                    self.pop_value();
                    self.pop_value();
                } else {
                    // `key in obj`: (key, obj) -> bool
                    self.expr(lhs)?;
                    let base = self.push_value();
                    self.expr(rhs)?;
                    self.push_value();
                    emit(
                        &mut self.code,
                        Opcode::CallRuntime,
                        &[bytecode::RuntimeFn::HasProperty as u32, base, 2],
                    );
                    self.pop_value();
                    self.pop_value();
                }
                Ok(())
            }
            TokenKind::QuestionDot | TokenKind::Nullish => self.err(
                node,
                match op {
                    TokenKind::QuestionDot => "optional chaining",
                    _ => "nullish coalescing",
                },
            ),
            _ => self.err(node, "binary operator"),
        }
    }

    /// `acc = lhs op rhs`: evaluate lhs into a temp, rhs after, then combine.
    fn binary_arith(&mut self, lhs: NodeId, rhs: NodeId, op: Opcode) -> Result<(), CompileError> {
        self.expr(lhs)?;
        let a = self.push_value();
        self.expr(rhs)?;
        let b = self.push_value();
        self.pop_value(); // b
        self.pop_value(); // a
        emit(&mut self.code, Opcode::Load, &[a]);
        emit(&mut self.code, op, &[b]);
        Ok(())
    }

    fn emit_conditional(
        &mut self,
        cond: NodeId,
        then: NodeId,
        else_: NodeId,
    ) -> Result<(), CompileError> {
        self.expr(cond)?;
        let mut else_l = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut else_l);
        self.expr(then)?;
        let mut end = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut end);
        else_l.bind(&self.code);
        else_l.patch_all(&mut self.code);
        self.expr(else_)?;
        end.bind(&self.code);
        end.patch_all(&mut self.code);
        Ok(())
    }

    fn emit_assign(
        &mut self,
        node: NodeId,
        op: TokenKind,
        target: NodeId,
        value: NodeId,
    ) -> Result<(), CompileError> {
        if op == TokenKind::Assign
            && matches!(
                *self.ast.node(target),
                Node::ArrayPattern { .. } | Node::ObjectPattern { .. }
            )
        {
            // destructuring assignment (ES 14.13.3): the RHS value, then
            // the pattern against it; the expression evaluates to the value
            self.expr(value)?;
            let v = self.push_value();
            self.emit_pattern(target, v, false)?;
            emit(&mut self.code, Opcode::Load, &[v]);
            self.pop_value();
            return Ok(());
        }
        match op {
            TokenKind::Assign => self.emit_simple_assign(target, value),
            TokenKind::PlusAssign => self.emit_compound_assign(node, target, value, Opcode::Add),
            TokenKind::MinusAssign => self.emit_compound_assign(node, target, value, Opcode::Sub),
            TokenKind::StarAssign => self.emit_compound_assign(node, target, value, Opcode::Mul),
            TokenKind::SlashAssign => self.emit_compound_assign(node, target, value, Opcode::Div),
            TokenKind::PercentAssign => self.emit_compound_assign(node, target, value, Opcode::Mod),
            TokenKind::StarStarAssign => {
                self.emit_compound_assign(node, target, value, Opcode::Exp)
            }
            TokenKind::ShlAssign => {
                self.emit_compound_assign(node, target, value, Opcode::ShiftLeft)
            }
            TokenKind::ShrAssign => {
                self.emit_compound_assign(node, target, value, Opcode::ShiftRight)
            }
            TokenKind::UshrAssign => {
                self.emit_compound_assign(node, target, value, Opcode::ShiftRightLogical)
            }
            TokenKind::AmpAssign => {
                self.emit_compound_assign(node, target, value, Opcode::BitwiseAnd)
            }
            TokenKind::PipeAssign => {
                self.emit_compound_assign(node, target, value, Opcode::BitwiseOr)
            }
            TokenKind::CaretAssign => {
                self.emit_compound_assign(node, target, value, Opcode::BitwiseXor)
            }
            TokenKind::AmpAmpAssign | TokenKind::OrOrAssign | TokenKind::NullishAssign => {
                self.err(node, "logical assignment")
            }
            _ => self.err(node, "assignment operator"),
        }
    }

    fn emit_simple_assign(&mut self, target: NodeId, value: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(target) {
            Node::Identifier { sym } => {
                self.expr(value)?;
                if self.is_anon_function(value) {
                    self.emit_set_name_const(self.ast.symbol(sym));
                }
                self.store_name(target, sym)
            }
            Node::Property { .. } | Node::SuperProperty { .. } => {
                let store = self.prepare_property_store(target)?;
                self.expr(value)?;
                // NamedEvaluation: `a.b = function () {}` names the closure "b"
                if let StoreTarget::Named { name_idx, .. } = &store
                    && self.is_anon_function(value)
                {
                    let name_idx = *name_idx;
                    emit(&mut self.code, Opcode::SetFunctionNameConst, &[name_idx]);
                }
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(target, "assignment target"),
        }
    }

    // -- destructuring ------------------------------------------------------------

    /// Compile a destructuring pattern against a value held in `value_reg`
    /// (ES 14.13). `binding`: the leaf targets are declarations (var/let/
    /// const/param names, stored by their Identifier resolution); otherwise
    /// they are assignment targets (identifiers and member expressions).
    fn emit_pattern(
        &mut self,
        pattern: NodeId,
        value_reg: u32,
        binding: bool,
    ) -> Result<(), CompileError> {
        match *self.ast.node(pattern) {
            Node::ObjectPattern { props } => {
                // RequireObjectCoercible runs even for the empty pattern
                // (ES 14.13.3)
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[
                        bytecode::RuntimeFn::RequireObjectCoercible as u32,
                        value_reg,
                        1,
                    ],
                );
                emit(&mut self.code, Opcode::Store, &[value_reg]);
                let has_rest = self
                    .ast
                    .list_items(props)
                    .last()
                    .is_some_and(|&p| matches!(self.ast.node(p), Node::PatternRest { .. }));
                // with a rest property, every earlier key stays live in
                // contiguous registers as the CopyDataProperties exclusion set
                let excl_base = self.reg_base + self.next_temp;
                let mut excluded = 0u32;
                for &prop in self.ast.list_items(props) {
                    match *self.ast.node(prop) {
                        Node::PatternRest { target } => {
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
                            self.emit_pattern_leaf(target, rest_obj, binding, NameHint::None)?;
                            self.pop_value(); // source
                            self.pop_value(); // rest_obj
                        }
                        Node::PatternProperty {
                            key,
                            value: elem,
                            computed,
                        } => {
                            let key_reg = if computed {
                                self.expr(key)?;
                                Some(self.push_value())
                            } else if matches!(self.ast.node(key), Node::NumberLiteral(_)) {
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
                                        // materialize the constant key for
                                        // the exclusion set
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
                                    emit(
                                        &mut self.code,
                                        Opcode::LoadKeyedProperty,
                                        &[value_reg, 0],
                                    );
                                }
                                None => {
                                    let NameHint::Const(name_idx) = name_hint else {
                                        unreachable!("constant keys have a constant hint")
                                    };
                                    emit(
                                        &mut self.code,
                                        Opcode::LoadNamedProperty,
                                        &[value_reg, name_idx, 0],
                                    );
                                }
                            }
                            self.emit_pattern_element(elem, binding, name_hint)?;
                            if !has_rest && key_reg.is_some() {
                                self.pop_value();
                            }
                        }
                        _ => return self.err(prop, "object pattern property"),
                    }
                }
                for _ in 0..excluded {
                    self.pop_value();
                }
                Ok(())
            }
            Node::ArrayPattern { elements } => {
                // iterator = GetIterator(value)
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::GetIterator as u32, value_reg, 1],
                );
                let iter = self.push_value();
                // done flag (ES 8.5.9: once done, later elements read
                // undefined without calling next again)
                emit(&mut self.code, Opcode::LoadSmi, &[0]);
                let done = self.push_value();
                let items = self.ast.list_items(elements).to_vec();
                let mut rest: Option<NodeId> = None;
                for &el in &items {
                    match *self.ast.node(el) {
                        Node::Hole => {
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
                        Node::PatternRest { target } => {
                            rest = Some(target);
                        }
                        Node::PatternElement { .. } => {
                            self.emit_iterator_element(el, iter, done, binding)?;
                        }
                        _ => return self.err(el, "array pattern element"),
                    }
                }
                if let Some(target) = rest {
                    // array ← remaining values (loop while !done)
                    emit(&mut self.code, Opcode::CreateEmptyArrayLiteral, &[]);
                    let arr = self.push_value();
                    emit(&mut self.code, Opcode::LoadSmi, &[0]);
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
                    emit(
                        &mut self.code,
                        Opcode::StoreKeyedPropertyNoShadow,
                        &[arr, idx, 0],
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
                    // both loop exits converge here: normal exhaustion and
                    // a done iterator short-circuit — the rest binding runs
                    // either way
                    exit.bind(&self.code);
                    exit.patch_all(&mut self.code);
                    emit(&mut self.code, Opcode::Load, &[arr]);
                    self.emit_pattern_leaf(target, arr, binding, NameHint::None)?;
                    self.pop_value(); // arr
                }
                self.pop_value(); // done
                self.pop_value(); // iter
                Ok(())
            }
            _ => self.err(pattern, "destructuring pattern"),
        }
    }

    /// Load a numeric literal key (small ints inline, floats via the pool).
    fn emit_number_key(&mut self, key: NodeId) {
        if let Node::NumberLiteral(f) = *self.ast.node(key) {
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
        elem: NodeId,
        iter: u32,
        done: u32,
        binding: bool,
    ) -> Result<(), CompileError> {
        let (target, default) = match *self.ast.node(elem) {
            Node::PatternElement { target, default } => (target, default),
            _ => return self.err(elem, "array pattern element"),
        };
        self.emit_load_constant(Constant::Undefined);
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
        let bind_name = match self.ast.node(target) {
            Node::Identifier { sym } if binding => Some(*sym),
            _ => None,
        };
        let hint = if default.is_some_and(|d| self.is_anon_function(d)) && bind_name.is_some() {
            let name = self.ast.symbol(bind_name.unwrap()).to_vec();
            let idx = self.add_constant(Constant::String(name));
            NameHint::Const(idx)
        } else {
            NameHint::None
        };
        self.emit_pattern_element_with(target, default, v, binding, hint)?;
        self.pop_value(); // v
        Ok(())
    }

    /// Value in the accumulator: apply the default and bind. The value is
    /// kept in a fresh register `v` for the (possibly nested) target.
    fn emit_pattern_element(
        &mut self,
        elem: NodeId,
        binding: bool,
        name_hint: NameHint,
    ) -> Result<(), CompileError> {
        let (target, default) = match *self.ast.node(elem) {
            Node::PatternElement { target, default } => (target, default),
            _ => return self.err(elem, "pattern element"),
        };
        let v = self.push_value();
        self.emit_pattern_element_with(target, default, v, binding, name_hint)?;
        self.pop_value();
        Ok(())
    }

    /// Default application + target binding with the current value in `v`.
    fn emit_pattern_element_with(
        &mut self,
        target: NodeId,
        default: Option<NodeId>,
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
            if self.is_anon_function(default)
                && (binding || matches!(self.ast.node(target), Node::Identifier { .. }))
            {
                match name_hint {
                    NameHint::Const(idx) => {
                        emit(&mut self.code, Opcode::SetFunctionNameConst, &[idx])
                    }
                    NameHint::Reg(r) => emit(&mut self.code, Opcode::SetFunctionNameKey, &[r, 0]),
                    NameHint::None => {}
                }
            }
            emit(&mut self.code, Opcode::Store, &[v]);
            skip.bind(&self.code);
            skip.patch_all(&mut self.code);
        }
        emit(&mut self.code, Opcode::Load, &[v]);
        self.emit_pattern_leaf(target, v, binding, NameHint::None)
    }

    /// Store the value in the accumulator into one pattern leaf:
    /// a declaration name (binding), an assignment target, or a nested
    /// pattern destructuring `value_reg`.
    fn emit_pattern_leaf(
        &mut self,
        target: NodeId,
        value_reg: u32,
        binding: bool,
        name_hint: NameHint,
    ) -> Result<(), CompileError> {
        let _ = name_hint;
        match *self.ast.node(target) {
            Node::Identifier { sym } if binding => self.store_name(target, sym),
            Node::Identifier { .. } | Node::Property { .. } => {
                self.emit_store_to_assignment_target(target)
            }
            Node::ArrayPattern { .. } | Node::ObjectPattern { .. } => {
                self.emit_pattern(target, value_reg, binding)
            }
            _ => self.err(target, "destructuring target"),
        }
    }

    /// Store the pattern value (in the accumulator) into an assignment
    /// target. Evaluating a member target's object clobbers the
    /// accumulator, so the value is parked in a register first.
    fn emit_store_to_assignment_target(&mut self, target: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(target) {
            Node::Identifier { .. } => match self.resolved.resolution(target) {
                Some(res) => self.store_decl_like(res, target),
                None => self.err(target, "unresolved assignment target"),
            },
            Node::Property { .. } | Node::SuperProperty { .. } => {
                let value = self.push_value();
                let store = self.prepare_property_store(target)?;
                emit(&mut self.code, Opcode::Load, &[value]);
                self.emit_property_store(&store);
                self.release_store(&store);
                self.pop_value(); // value
                Ok(())
            }
            _ => self.err(target, "assignment target"),
        }
    }

    /// Store acc into an identifier via its node resolution (assignment
    /// patterns; the node was resolved by the walk).
    fn store_decl_like(&mut self, res: Resolution, node: NodeId) -> Result<(), CompileError> {
        match res {
            Resolution::GlobalObject | Resolution::Dynamic => {
                if let Node::Identifier { sym } = *self.ast.node(node) {
                    self.store_name(node, sym)
                } else {
                    unreachable!("identifier target")
                }
            }
            other => {
                self.store_resolution(other);
                Ok(())
            }
        }
    }

    /// A store target ready for the final store: the object (and computed
    /// key) evaluated into live registers, plus the name constant index.

    fn emit_property_store(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { obj, name_idx } => {
                emit(
                    &mut self.code,
                    Opcode::StoreNamedProperty,
                    &[*obj, *name_idx, 0],
                );
            }
            StoreTarget::Keyed { obj, key } => {
                emit(&mut self.code, Opcode::StoreKeyedProperty, &[*obj, *key, 0]);
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
            StoreTarget::SuperNamed {
                recv,
                home,
                name_idx,
            } => {
                emit(
                    &mut self.code,
                    Opcode::StoreNamedPropertyToSuper,
                    &[*home, *recv, *name_idx, 0, 0],
                );
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                emit(
                    &mut self.code,
                    Opcode::StoreKeyedPropertyToSuper,
                    &[*home, *recv, *key, 0, 0],
                );
            }
        }
    }

    /// Load the current value of a prepared store target into the
    /// accumulator (compound assignment / update prefixes).
    fn emit_property_load_of(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { obj, name_idx } => {
                emit(
                    &mut self.code,
                    Opcode::LoadNamedProperty,
                    &[*obj, *name_idx, 0],
                );
            }
            StoreTarget::Keyed { obj, key } => {
                emit(&mut self.code, Opcode::Load, &[*key]);
                emit(&mut self.code, Opcode::LoadKeyedProperty, &[*obj, 0]);
            }
            StoreTarget::PrivateKeyed { obj, key } => {
                emit(&mut self.code, Opcode::Load, &[*key]);
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::PrivateGet as u32, *obj, 2],
                );
            }
            StoreTarget::SuperNamed {
                recv,
                home,
                name_idx,
            } => {
                emit(&mut self.code, Opcode::Load, &[*home]);
                emit(
                    &mut self.code,
                    Opcode::LoadNamedPropertyFromSuper,
                    &[*recv, *name_idx, 0],
                );
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                emit(&mut self.code, Opcode::Load, &[*home]);
                emit(
                    &mut self.code,
                    Opcode::LoadKeyedPropertyFromSuper,
                    &[*recv, *key, 0],
                );
            }
        }
    }

    /// Drop the live registers of a store target in push order (LIFO).
    fn release_store(&mut self, store: &StoreTarget) {
        match store {
            StoreTarget::Named { .. } => self.pop_value(), // obj
            StoreTarget::Keyed { .. } => {
                self.pop_value(); // key
                self.pop_value(); // obj
            }
            StoreTarget::PrivateKeyed { .. } => {
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

    /// Evaluate the object (and computed key) of a property store target,
    /// leaving them as live registers above `reg_base`.
    fn prepare_property_store(&mut self, target: NodeId) -> Result<StoreTarget, CompileError> {
        match *self.ast.node(target) {
            Node::Property {
                object,
                key,
                computed: false,
            } if matches!(self.ast.node(key), Node::PrivateName { .. }) => {
                self.expr(object)?;
                let obj = self.push_value();
                self.emit_private_key_load(key)?;
                let k = self.push_value();
                Ok(StoreTarget::PrivateKeyed { obj, key: k })
            }
            Node::Property {
                object,
                key,
                computed: false,
            } => {
                self.expr(object)?;
                let obj = self.push_value();
                let name_idx = self.name_constant(key)?;
                Ok(StoreTarget::Named { obj, name_idx })
            }
            Node::Property {
                object,
                key,
                computed: true,
            } => {
                self.expr(object)?;
                let obj = self.push_value();
                self.expr(key)?;
                let k = self.push_value();
                Ok(StoreTarget::Keyed { obj, key: k })
            }
            Node::SuperProperty { key, computed, .. } => {
                let Some(store) = self.prepare_super_parts(target, key, computed)? else {
                    return self.err(target, "super assignment target");
                };
                Ok(store)
            }
            _ => self.err(target, "assignment target"),
        }
    }

    /// The receiver and home-object registers of a `super.x` reference
    /// (assignment target or compound-access base).
    fn prepare_super_parts(
        &mut self,
        node: NodeId,
        key: NodeId,
        computed: bool,
    ) -> Result<Option<StoreTarget>, CompileError> {
        let Some(Resolution::Super {
            home_slot,
            depth: home_depth,
            this_owner,
            this_depth,
        }) = self.resolved.resolution(node)
        else {
            return Ok(None);
        };
        self.emit_this_for(this_owner, this_depth);
        let recv = self.push_value();
        if computed {
            self.expr(key)?;
            let k = self.push_value();
            emit(
                &mut self.code,
                Opcode::LoadContextSlot,
                &[home_slot, home_depth],
            );
            let home = self.push_value();
            Ok(Some(StoreTarget::SuperKeyed { recv, home, key: k }))
        } else {
            let name_idx = self.name_constant(key)?;
            emit(
                &mut self.code,
                Opcode::LoadContextSlot,
                &[home_slot, home_depth],
            );
            let home = self.push_value();
            Ok(Some(StoreTarget::SuperNamed {
                recv,
                home,
                name_idx,
            }))
        }
    }

    /// Compound assignment with a single evaluation of property references.
    fn emit_compound_assign(
        &mut self,
        node: NodeId,
        target: NodeId,
        value: NodeId,
        op: Opcode,
    ) -> Result<(), CompileError> {
        if let Node::Identifier { sym } = *self.ast.node(target) {
            self.expr(value)?;
            let v = self.push_value();
            self.emit_identifier(target, sym)?;
            emit(&mut self.code, op, &[v]);
            self.pop_value();
            self.store_name(target, sym)?;
            return Ok(());
        }
        if let Node::Property { .. } | Node::SuperProperty { .. } = *self.ast.node(target) {
            let store = self.prepare_property_store(target)?;
            self.expr(value)?;
            let v = self.push_value();
            self.emit_property_load_of(&store);
            emit(&mut self.code, op, &[v]);
            self.pop_value();
            self.emit_property_store(&store);
            self.release_store(&store);
            return Ok(());
        }
        self.err(node, "assignment target")
    }

    fn emit_update(
        &mut self,
        node: NodeId,
        op: TokenKind,
        prefix: bool,
        target: NodeId,
    ) -> Result<(), CompileError> {
        let delta = match op {
            TokenKind::PlusPlus => 1i32,
            TokenKind::MinusMinus => -1i32,
            _ => return self.err(node, "update operator"),
        };
        let arith = if delta > 0 { Opcode::Add } else { Opcode::Sub };
        let delta = delta.unsigned_abs();

        if let Node::Identifier { sym } = *self.ast.node(target) {
            self.emit_identifier(target, sym)?;
            let orig = self.push_value();
            emit(&mut self.code, Opcode::LoadSmi, &[delta]);
            let d = self.push_value();
            emit(&mut self.code, Opcode::LoadSmi, &[0]);
            let zero = self.push_value();
            emit(&mut self.code, Opcode::Load, &[orig]);
            emit(&mut self.code, Opcode::Sub, &[zero]);
            self.pop_value(); // zero
            emit(&mut self.code, arith, &[d]);
            self.pop_value(); // d
            self.store_name(target, sym)?;
            if !prefix {
                emit(&mut self.code, Opcode::Load, &[orig]);
            }
            self.pop_value(); // orig
            return Ok(());
        }
        if let Node::Property { .. } | Node::SuperProperty { .. } = *self.ast.node(target) {
            let store = self.prepare_property_store(target)?;
            self.emit_property_load_of(&store);
            let orig = self.push_value();
            emit(&mut self.code, Opcode::LoadSmi, &[delta]);
            let d = self.push_value();
            emit(&mut self.code, Opcode::LoadSmi, &[0]);
            let zero = self.push_value();
            emit(&mut self.code, Opcode::Load, &[orig]);
            emit(&mut self.code, Opcode::Sub, &[zero]);
            self.pop_value(); // zero
            emit(&mut self.code, arith, &[d]);
            self.pop_value(); // d
            self.emit_property_store(&store);
            if !prefix {
                emit(&mut self.code, Opcode::Load, &[orig]);
            }
            self.pop_value(); // orig
            self.release_store(&store);
            return Ok(());
        }
        self.err(node, "update target")
    }

    fn emit_property_load(
        &mut self,
        object: NodeId,
        key: NodeId,
        computed: bool,
    ) -> Result<(), CompileError> {
        if !computed && matches!(self.ast.node(key), Node::PrivateName { .. }) {
            // PrivateGet: `obj.#x` — own private field or TypeError
            self.expr(object)?;
            let obj = self.push_value();
            self.emit_private_key_load(key)?;
            self.push_value();
            emit(
                &mut self.code,
                Opcode::CallRuntime,
                &[bytecode::RuntimeFn::PrivateGet as u32, obj, 2],
            );
            self.pop_value();
            self.pop_value();
            return Ok(());
        }
        if computed {
            self.expr(object)?;
            let obj = self.push_value();
            self.expr(key)?;
            emit(&mut self.code, Opcode::LoadKeyedProperty, &[obj, 0]);
            self.pop_value();
        } else {
            self.expr(object)?;
            let obj = self.push_value();
            let name_idx = self.name_constant(key)?;
            emit(
                &mut self.code,
                Opcode::LoadNamedProperty,
                &[obj, name_idx, 0],
            );
            self.pop_value();
        }
        Ok(())
    }

    fn emit_call(
        &mut self,
        _node: NodeId,
        callee: NodeId,
        args: parser::NodeList,
    ) -> Result<(), CompileError> {
        // Method calls: the receiver is the first register of the argument
        // list, so the argument registers must immediately follow it. The
        // callee is stored in a fixed slot above the args (evaluation order:
        // receiver, property get, then arguments).
        if let Node::SuperProperty { key, computed, .. } = *self.ast.node(callee) {
            // super.m(...): the method comes from the home-object chain and
            // runs with the current `this`
            let argc = self.ast.list_items(args).len();
            let Some(store) = self.prepare_super_parts(callee, key, computed)? else {
                return self.err(callee, "super method call");
            };
            match &store {
                StoreTarget::SuperNamed {
                    recv,
                    home,
                    name_idx,
                } => {
                    let (recv, home, name_idx) = (*recv, *home, *name_idx);
                    emit(&mut self.code, Opcode::Load, &[home]);
                    emit(
                        &mut self.code,
                        Opcode::LoadNamedPropertyFromSuper,
                        &[recv, name_idx, 0],
                    );
                }
                StoreTarget::SuperKeyed { recv, home, key } => {
                    let (recv, home, key) = (*recv, *home, *key);
                    emit(&mut self.code, Opcode::Load, &[home]);
                    emit(
                        &mut self.code,
                        Opcode::LoadKeyedPropertyFromSuper,
                        &[recv, key, 0],
                    );
                }
                _ => unreachable!("super store parts"),
            }
            let recv = match &store {
                StoreTarget::SuperNamed { recv, .. } => *recv,
                StoreTarget::SuperKeyed { recv, .. } => *recv,
                _ => unreachable!(),
            };
            let callee_reg = self.reg_base + self.next_temp + argc as u32;
            emit(&mut self.code, Opcode::Store, &[callee_reg]);
            self.next_temp += argc as u32 + 1;
            for (i, &arg) in self.ast.list_items(args).iter().enumerate() {
                self.expr(arg)?;
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
            return Ok(());
        }
        if let Node::Property {
            object,
            key,
            computed: false,
        } = *self.ast.node(callee)
        {
            let argc = self.ast.list_items(args).len();
            self.expr(object)?;
            let recv = self.push_value();
            let name_idx = self.name_constant(key)?;
            emit(
                &mut self.code,
                Opcode::LoadNamedProperty,
                &[recv, name_idx, 0],
            );
            let callee_reg = self.reg_base + self.next_temp + argc as u32;
            emit(&mut self.code, Opcode::Store, &[callee_reg]);
            // reserve args + callee so nested argument temps land above
            self.next_temp += argc as u32 + 1;
            for (i, &arg) in self.ast.list_items(args).iter().enumerate() {
                self.expr(arg)?;
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
        if let Node::Property {
            object,
            key,
            computed: true,
        } = *self.ast.node(callee)
        {
            let argc = self.ast.list_items(args).len();
            self.expr(object)?;
            let recv = self.push_value();
            self.expr(key)?;
            emit(&mut self.code, Opcode::LoadKeyedProperty, &[recv, 0]);
            let callee_reg = self.reg_base + self.next_temp + argc as u32;
            emit(&mut self.code, Opcode::Store, &[callee_reg]);
            self.next_temp += argc as u32 + 1;
            for (i, &arg) in self.ast.list_items(args).iter().enumerate() {
                self.expr(arg)?;
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
        // plain call: receiver = undefined (slot 0), callee evaluated
        // first, then arguments
        let argc = self.ast.list_items(args).len();
        self.expr(callee)?;
        let callee_reg = self.reg_base + self.next_temp + 1 + argc as u32;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        self.emit_load_constant(Constant::Undefined);
        let recv = self.push_value();
        // reserve arguments + callee so nested argument temps land above
        self.next_temp += argc as u32 + 1;
        for (i, &arg) in self.ast.list_items(args).iter().enumerate() {
            self.expr(arg)?;
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

    fn emit_new(
        &mut self,
        _node: NodeId,
        callee: NodeId,
        args: Option<parser::NodeList>,
    ) -> Result<(), CompileError> {
        let items = args
            .map(|a| self.ast.list_items(a).to_vec())
            .unwrap_or_default();
        let argc = items.len();
        self.expr(callee)?;
        let callee_reg = self.reg_base + self.next_temp + argc as u32;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        let arg_base = self.reg_base + self.next_temp;
        // reserve arguments + callee so nested argument temps land above
        self.next_temp += argc as u32 + 1;
        for (i, &arg) in items.iter().enumerate() {
            self.expr(arg)?;
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

    // -- classes ---------------------------------------------------------------

    /// Class definitions (ES 15.7.14 ClassDefinitionEvaluation) as inline
    /// per-member emission: validate the superclass, create the class
    /// context, build the prototype and constructor, install members with
    /// class attributes, wire the prototype chain, and store the bindings.
    /// Leaves the class constructor in the accumulator.
    fn emit_class(&mut self, node: NodeId, class: ClassId) -> Result<(), CompileError> {
        let info = self.ast.class(class);
        let class_scope = self
            .ast
            .node_scope(node)
            .expect("class node owns its scope");
        let needs_ctx = info.name.is_some() || info.uses_super || !info.privates.is_empty();

        // class inner context: the name binding (TDZ until the class value
        // exists) and the super home objects; captured by member closures.
        // Pushed before the superclass evaluation: ClassHeritage sees the
        // inner binding (in TDZ) per ES 15.7.14 step 8.
        let ctx_save = if needs_ctx {
            let slot_count = self.ast.scope(class_scope).decls.len() as u32;
            emit(&mut self.code, Opcode::CreateBlockContext, &[slot_count]);
            let save = self.reserve_temp();
            emit(&mut self.code, Opcode::PushContext, &[save]);
            Some(save)
        } else {
            None
        };

        // private names: one fresh Symbol per private field per class
        // evaluation, stored into the class context before any element
        // evaluates (methods and initializers reference them by slot)
        for hidden in &info.privates {
            let res = self
                .resolved
                .resolution_for_decl(class_scope, *hidden)
                .expect("private slot declared");
            if let Resolution::Context { slot, .. } = res {
                let text = self.ast.symbol(*hidden);
                let desc = text.strip_prefix(&b".priv."[..]).unwrap_or(text).to_vec();
                let idx = self.add_constant(Constant::String(desc));
                emit(&mut self.code, Opcode::LoadConstant, &[idx]);
                let _desc_reg = self.push_value();
                emit(
                    &mut self.code,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::CreatePrivateName as u32, _desc_reg, 1],
                );
                self.pop_value();
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
        }

        // superclass: must be null or a constructor
        let sup = if let Some(sup) = info.superclass {
            self.expr(sup)?;
            emit(&mut self.code, Opcode::ThrowIfNotConstructorOrNull, &[]);
            Some(self.push_value())
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
            // %Function.prototype% (objects are always truthy, so the falsy
            // test identifies null)
            emit(&mut self.code, Opcode::Load, &[sup]);
            let mut null_extends = Label::new();
            emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut null_extends);
            // protoParent = Get(superCtor, "prototype") (full [[Get]])
            let proto_name = self.add_constant(Constant::String(b"prototype".to_vec()));
            emit(
                &mut self.code,
                Opcode::LoadNamedProperty,
                &[sup, proto_name, 0],
            );
            emit(&mut self.code, Opcode::ThrowIfNotObjectOrNull, &[]);
            emit(&mut self.code, Opcode::Store, &[pp]);
            emit(&mut self.code, Opcode::Load, &[sup]);
            emit(&mut self.code, Opcode::Store, &[cp]);
            let mut done = Label::new();
            emit_jump(&mut self.code, Opcode::Jump, &mut done);
            null_extends.bind(&self.code);
            null_extends.patch_all(&mut self.code);
            self.emit_load_constant(Constant::Null);
            emit(&mut self.code, Opcode::Store, &[pp]);
            done.bind(&self.code);
            done.patch_all(&mut self.code);
        }

        // prototype: a fresh ordinary object with protoParent
        emit(&mut self.code, Opcode::CreateEmptyObjectLiteral, &[]);
        let proto = self.push_value();
        emit(&mut self.code, Opcode::Load, &[proto]);
        emit(&mut self.code, Opcode::SetPrototype, &[pp]);

        // constructor closure
        let ctor_idx = self.add_constant(Constant::Callable(info.ctor));
        emit(&mut self.code, Opcode::CreateClosure, &[ctor_idx]);
        let ctor = self.push_value();

        // wiring before member installation (ES 15.7.14 steps 17–18 precede
        // the element loop): computed `['constructor']` members overwrite
        // proto.constructor, computed static `['prototype']` defines fail
        // against the non-configurable ctor.prototype
        // proto.constructor → the class {w+, e−, c+}
        emit(&mut self.code, Opcode::Load, &[ctor]);
        let ctor_name = self.add_constant(Constant::String(b"constructor".to_vec()));
        emit(
            &mut self.code,
            Opcode::DefineNamedOwnProperty,
            &[proto, ctor_name, PropertyFlags::DontEnum.bits(), 0],
        );
        // ctor.prototype → the prototype {w+, e−, c−}
        emit(&mut self.code, Opcode::Load, &[proto]);
        let proto_name = self.add_constant(Constant::String(b"prototype".to_vec()));
        emit(
            &mut self.code,
            Opcode::DefineNamedOwnProperty,
            &[
                ctor,
                proto_name,
                PropertyFlags::DontEnum.bits() | PropertyFlags::DontDelete.bits(),
                0,
            ],
        );
        // the class itself inherits from the superclass constructor
        emit(&mut self.code, Opcode::Load, &[ctor]);
        emit(&mut self.code, Opcode::SetPrototype, &[cp]);

        // instance field list: a JS array [key0, init0, key1, init1, ...]
        // attached to the constructor (its hidden fields slot)
        let has_instance_fields = info
            .members
            .iter()
            .any(|m| m.kind == PropKind::Field && !m.is_static);
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
        for m in &info.members {
            if m.is_constructor {
                continue;
            }
            let function = match *self.ast.node(m.value) {
                Node::FunctionExpr { function } => function,
                _ => return self.err(m.value, "class member function"),
            };
            if m.kind == PropKind::Field {
                // field key: evaluated in element order (ES 15.7.14 step 27)
                let key_reg = if m.is_private {
                    self.emit_private_key_load(m.key)?;
                    Some(self.push_value())
                } else if m.computed {
                    self.expr(m.key)?;
                    Some(self.push_value())
                } else if matches!(*self.ast.node(m.key), Node::NumberLiteral(_)) {
                    self.emit_number_key(m.key);
                    Some(self.push_value())
                } else {
                    None
                };
                let fn_idx = self.add_constant(Constant::Callable(function));
                emit(&mut self.code, Opcode::CreateClosure, &[fn_idx]);
                if m.is_static {
                    let closure = self.push_value();
                    let name_idx = match key_reg {
                        Some(_) => None,
                        None => Some(self.name_constant(m.key)?),
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
                            emit(
                                &mut self.code,
                                Opcode::StoreKeyedPropertyNoShadow,
                                &[arr, i, 0],
                            );
                        }
                        None => {
                            let name_idx = self.name_constant(m.key)?;
                            emit(&mut self.code, Opcode::LoadConstant, &[name_idx]);
                            emit(
                                &mut self.code,
                                Opcode::StoreKeyedPropertyNoShadow,
                                &[arr, i, 0],
                            );
                        }
                    }
                    // initializer at the next slot
                    emit(&mut self.code, Opcode::LoadSmi, &[field_index + 1]);
                    emit(&mut self.code, Opcode::Store, &[i]);
                    emit(&mut self.code, Opcode::Load, &[closure]);
                    emit(
                        &mut self.code,
                        Opcode::StoreKeyedPropertyNoShadow,
                        &[arr, i, 0],
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
            let target = if m.is_static { ctor } else { proto };
            // non-computed keys are strings or numbers; numbers go through
            // the keyed define with the literal loaded into a register
            let numeric_key =
                !m.computed && matches!(*self.ast.node(m.key), Node::NumberLiteral(_));
            let key = if m.computed || numeric_key {
                if m.computed {
                    self.expr(m.key)?;
                } else if let Node::NumberLiteral(f) = *self.ast.node(m.key) {
                    if f.fract() == 0.0 && f.is_sign_positive() && f <= i16::MAX as f64 {
                        emit(&mut self.code, Opcode::LoadSmi, &[f as i32 as u32]);
                    } else {
                        self.emit_load_constant(Constant::Float(f));
                    }
                } else {
                    unreachable!("numeric key checked above");
                }
                Some(self.push_value())
            } else {
                None
            };
            let fn_idx = self.add_constant(Constant::Callable(function));
            emit(&mut self.code, Opcode::CreateClosure, &[fn_idx]);
            // computed keys name the member after their ToPropertyKey value
            if let Some(k) = key {
                let prefix = match m.kind {
                    PropKind::Get => 1,
                    PropKind::Set => 2,
                    _ => 0,
                };
                emit(&mut self.code, Opcode::SetFunctionNameKey, &[k, prefix]);
            }
            match m.kind {
                PropKind::Method => {
                    // {w+, e−, c+}
                    match key {
                        Some(k) => {
                            emit(
                                &mut self.code,
                                Opcode::DefineKeyedOwnProperty,
                                &[target, k, PropertyFlags::DontEnum.bits(), 0],
                            );
                        }
                        None => {
                            let name_idx = self.name_constant(m.key)?;
                            emit(
                                &mut self.code,
                                Opcode::DefineNamedOwnProperty,
                                &[target, name_idx, PropertyFlags::DontEnum.bits(), 0],
                            );
                        }
                    }
                }
                PropKind::Get | PropKind::Set => {
                    // one accessor half; merges with an existing pair
                    let mut flags = PropertyFlags::DontEnum.bits();
                    if m.kind == PropKind::Get {
                        flags |= 1;
                    }
                    match key {
                        Some(k) => {
                            emit(
                                &mut self.code,
                                Opcode::InstallKeyedAccessor,
                                &[target, k, flags],
                            );
                        }
                        None => {
                            let name_idx = self.name_constant(m.key)?;
                            emit(
                                &mut self.code,
                                Opcode::InstallNamedAccessor,
                                &[target, name_idx, flags],
                            );
                        }
                    }
                }
                PropKind::Init | PropKind::Field => {
                    return self.err(m.value, "class field initializers");
                }
            }
            if key.is_some() {
                self.pop_value();
            }
        }

        // home objects and the inner class-name binding
        if info.uses_super {
            let home = info.home.expect("uses_super classes declare home slots");
            let res = self
                .resolved
                .resolution_for_decl(class_scope, home)
                .expect("home slot declared");
            if let Resolution::Context { slot, .. } = res {
                // the class context is pushed here: zero hops
                emit(&mut self.code, Opcode::Load, &[proto]);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
            let static_home = info
                .static_home
                .expect("uses_super classes declare static home slots");
            let res = self
                .resolved
                .resolution_for_decl(class_scope, static_home)
                .expect("static home slot declared");
            if let Resolution::Context { slot, .. } = res {
                emit(&mut self.code, Opcode::Load, &[ctor]);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
        }
        if let Some(name) = info.name {
            if let Resolution::Context { slot, .. } = self
                .resolved
                .resolution_for_decl(class_scope, name)
                .expect("class name declared in the class scope")
            {
                emit(&mut self.code, Opcode::Load, &[ctor]);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
        }

        // static fields: each initializer runs with the constructor as
        // `this` and its result is [[DefineOwnProperty]]'d on it, in
        // declaration order (ES 15.7.14 step 33; keys were already
        // evaluated during the element loop)
        for (closure, key_reg, name_idx) in &static_fields {
            emit(&mut self.code, Opcode::Load, &[*closure]);
            emit(&mut self.code, Opcode::CallNoFeedback, &[*closure, ctor, 1]);
            match (key_reg, name_idx) {
                (Some(k), _) => {
                    emit(
                        &mut self.code,
                        Opcode::DefineKeyedOwnProperty,
                        &[ctor, *k, 0, 0],
                    );
                }
                (None, Some(name_idx)) => {
                    emit(
                        &mut self.code,
                        Opcode::DefineNamedOwnProperty,
                        &[ctor, *name_idx, 0, 0],
                    );
                }
                (None, None) => unreachable!("static field keys are reg or const"),
            }
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
        if let Node::ClassDecl { .. } = *self.ast.node(node)
            && let Some(name) = info.name
        {
            let Some(scope) = self.find_decl_scope(name) else {
                return self.err(node, "class declaration without a binding");
            };
            let res = self
                .resolved
                .resolution_for_decl(scope, name)
                .expect("declared");
            self.store_decl(res, name);
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

    /// `super.x` / `super[key]` load.
    fn emit_super_property_load(
        &mut self,
        node: NodeId,
        key: NodeId,
        computed: bool,
    ) -> Result<(), CompileError> {
        let Some(store) = self.prepare_super_parts(node, key, computed)? else {
            return self.err(node, "super property");
        };
        match store {
            StoreTarget::SuperNamed {
                recv,
                home,
                name_idx,
            } => {
                emit(&mut self.code, Opcode::Load, &[home]);
                emit(
                    &mut self.code,
                    Opcode::LoadNamedPropertyFromSuper,
                    &[recv, name_idx, 0],
                );
            }
            StoreTarget::SuperKeyed { recv, home, key } => {
                emit(&mut self.code, Opcode::Load, &[home]);
                emit(
                    &mut self.code,
                    Opcode::LoadKeyedPropertyFromSuper,
                    &[recv, key, 0],
                );
            }
            _ => unreachable!("super store parts"),
        }
        self.release_store(&store);
        Ok(())
    }

    /// `super(...)`: construct the superclass with the constructor's
    /// new.target and initialize `this` with the result (ES 15.4.3).
    /// Direct calls resolve the super constructor from the frame; calls
    /// delegated through arrows use the constructor's threaded closure and
    /// new.target. Evaluates to the new `this`.
    fn emit_super_call(
        &mut self,
        node: NodeId,
        args: parser::NodeList,
    ) -> Result<(), CompileError> {
        let items = self.ast.list_items(args).to_vec();
        let argc = items.len();
        let (owner, owner_depth) = match self.resolved.resolution(node) {
            Some(Resolution::SuperCall { owner, depth }) => (owner, depth),
            _ => (self.fid, 0),
        };
        let direct = owner == self.fid;
        // where the constructor's `this` lives (the bind target)
        let this_slot = if direct {
            self.resolved.layout(self.fid).this_slot
        } else {
            Some(
                self.resolved
                    .layout(owner)
                    .this_slot
                    .expect("delegated super() forces the this slot"),
            )
        };

        let arg_base = self.reg_base + self.next_temp;
        // reserve arguments + result (+ closure/new.target registers for
        // the delegated variant) so nested temps land above
        let reserved = argc as u32 + 1 + if direct { 0 } else { 2 };
        self.next_temp += reserved;
        for (i, &arg) in items.iter().enumerate() {
            self.expr(arg)?;
            emit(&mut self.code, Opcode::Store, &[arg_base + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        if direct {
            emit(
                &mut self.code,
                Opcode::ConstructSuper,
                &[arg_base, argc as u32],
            );
        } else {
            // .this_function and .new.target of the owning constructor
            let layout = self.resolved.layout(owner);
            let closure_reg = arg_base + argc as u32 + 1;
            let new_target_reg = closure_reg + 1;
            emit(
                &mut self.code,
                Opcode::LoadContextSlot,
                &[
                    layout
                        .this_function_slot
                        .expect("delegated super() forces the closure slot"),
                    owner_depth,
                ],
            );
            emit(&mut self.code, Opcode::Store, &[closure_reg]);
            emit(
                &mut self.code,
                Opcode::LoadContextSlot,
                &[
                    layout
                        .new_target_slot
                        .expect("delegated super() forces the new.target slot"),
                    owner_depth,
                ],
            );
            emit(&mut self.code, Opcode::Store, &[new_target_reg]);
            emit(
                &mut self.code,
                Opcode::ConstructSuperVia,
                &[closure_reg, new_target_reg, arg_base, argc as u32],
            );
        }
        let result = arg_base + argc as u32;
        emit(&mut self.code, Opcode::Store, &[result]);
        // InitializeThisBinding: this must still be uninitialized
        match this_slot {
            Some(slot) => {
                let depth = if direct { 0 } else { owner_depth };
                emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                emit(
                    &mut self.code,
                    Opcode::ThrowSuperAlreadyCalledIfNotHole,
                    &[],
                );
                emit(&mut self.code, Opcode::Load, &[result]);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth]);
            }
            None => {
                emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
                emit(
                    &mut self.code,
                    Opcode::ThrowSuperAlreadyCalledIfNotHole,
                    &[],
                );
                emit(&mut self.code, Opcode::Load, &[result]);
                emit(&mut self.code, Opcode::Store, &[(-1i32) as u32]);
            }
        }
        // InitializeInstanceElements (ES 7.3.33): the derived constructor's
        // own fields are defined on the freshly bound instance (for
        // arrow-delegated super(), the owner is the constructor)
        let field_owner = if direct { self.fid } else { owner };
        if self.fn_has_instance_fields(field_owner) {
            let ctor = self.reserve_temp();
            if direct {
                emit(&mut self.code, Opcode::LdaCurrentClosure, &[]);
            } else {
                let layout = self.resolved.layout(owner);
                emit(
                    &mut self.code,
                    Opcode::LoadContextSlot,
                    &[
                        layout
                            .this_function_slot
                            .expect("delegated super() forces the closure slot"),
                        owner_depth,
                    ],
                );
            }
            emit(&mut self.code, Opcode::Store, &[ctor]);
            emit(&mut self.code, Opcode::Load, &[result]);
            self.push_value();
            let _ = ctor;
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

    fn emit_array_literal(
        &mut self,
        node: NodeId,
        elements: parser::NodeList,
    ) -> Result<(), CompileError> {
        emit(&mut self.code, Opcode::CreateEmptyArrayLiteral, &[]);
        let arr = self.push_value();
        for (i, &el) in self.ast.list_items(elements).iter().enumerate() {
            match *self.ast.node(el) {
                Node::Hole => {}
                Node::Spread { .. } => return self.err(node, "spread in array literals"),
                _ => {
                    emit(&mut self.code, Opcode::LoadSmi, &[i as u32]);
                    let idx = self.push_value();
                    self.expr(el)?;
                    emit(
                        &mut self.code,
                        Opcode::StoreKeyedPropertyNoShadow,
                        &[arr, idx, 0],
                    );
                    self.pop_value();
                }
            }
        }
        emit(&mut self.code, Opcode::Load, &[arr]);
        self.pop_value();
        Ok(())
    }

    fn emit_object_literal(
        &mut self,
        node: NodeId,
        props: parser::NodeList,
    ) -> Result<(), CompileError> {
        // methods using `super` capture a per-literal home-object context
        // (the literal itself), mirroring class scopes
        let scope = self.ast.node_scope(node);
        let slot_count = scope.map_or(0, |s| self.ast.scope(s).decls.len() as u32);
        let ctx_save = if slot_count > 0 {
            emit(&mut self.code, Opcode::CreateBlockContext, &[slot_count]);
            let save = self.reserve_temp();
            emit(&mut self.code, Opcode::PushContext, &[save]);
            Some(save)
        } else {
            None
        };

        emit(&mut self.code, Opcode::CreateEmptyObjectLiteral, &[]);
        let obj = self.push_value();
        for &prop in self.ast.list_items(props) {
            match *self.ast.node(prop) {
                Node::ObjectProperty {
                    key,
                    value,
                    kind,
                    computed,
                } => {
                    let key_reg = if computed {
                        self.expr(key)?;
                        Some(self.push_value())
                    } else if matches!(self.ast.node(key), Node::NumberLiteral(_)) {
                        // numeric literal keys ({ 1: x }) use the keyed path
                        self.emit_number_key(key);
                        Some(self.push_value())
                    } else {
                        None
                    };
                    match kind {
                        PropKind::Init => {
                            self.expr(value)?;
                            match key_reg {
                                None => {
                                    // NamedEvaluation: { m: function () {} }
                                    let name_idx = self.name_constant(key)?;
                                    if self.is_anon_function(value) {
                                        emit(
                                            &mut self.code,
                                            Opcode::SetFunctionNameConst,
                                            &[name_idx],
                                        );
                                    }
                                    emit(
                                        &mut self.code,
                                        Opcode::StoreNamedProperty,
                                        &[obj, name_idx, 0],
                                    );
                                }
                                Some(k) => {
                                    if self.is_anon_function(value) {
                                        emit(&mut self.code, Opcode::SetFunctionNameKey, &[k, 0]);
                                    }
                                    emit(&mut self.code, Opcode::StoreKeyedProperty, &[obj, k, 0]);
                                }
                            }
                        }
                        PropKind::Method | PropKind::Get | PropKind::Set => {
                            let prefix = match kind {
                                PropKind::Get => 1,
                                PropKind::Set => 2,
                                _ => 0,
                            };
                            // the closure is created by evaluating the
                            // method value; name it afterwards
                            self.expr(value)?;
                            if let Some(k) = key_reg {
                                emit(&mut self.code, Opcode::SetFunctionNameKey, &[k, prefix]);
                            }
                            match kind {
                                PropKind::Get | PropKind::Set => {
                                    // enumerable accessor halves, merged
                                    // pairs; bit 0 marks the getter half
                                    let flags = match kind {
                                        PropKind::Get => 1,
                                        _ => 0,
                                    };
                                    if let Some(k) = key_reg {
                                        emit(
                                            &mut self.code,
                                            Opcode::InstallKeyedAccessor,
                                            &[obj, k, flags],
                                        );
                                    } else {
                                        let name_idx = self.name_constant(key)?;
                                        emit(
                                            &mut self.code,
                                            Opcode::InstallNamedAccessor,
                                            &[obj, name_idx, flags],
                                        );
                                    }
                                }
                                PropKind::Method => {
                                    if let Some(k) = key_reg {
                                        emit(
                                            &mut self.code,
                                            Opcode::StoreKeyedProperty,
                                            &[obj, k, 0],
                                        );
                                    } else {
                                        let name_idx = self.name_constant(key)?;
                                        emit(
                                            &mut self.code,
                                            Opcode::StoreNamedProperty,
                                            &[obj, name_idx, 0],
                                        );
                                    }
                                }
                                PropKind::Init | PropKind::Field => unreachable!(),
                            }
                        }
                        PropKind::Field => unreachable!("fields only exist in class bodies"),
                    }
                    if let Some(k) = key_reg {
                        let _ = k;
                        self.pop_value();
                    }
                }
                Node::Spread { .. } => return self.err(prop, "spread in object literals"),
                _ => return self.err(prop, "object literal property"),
            }
        }
        // the home object is the literal itself; methods already captured
        // the context, so the store is visible to them
        if let Some(scope) = scope
            && slot_count > 0
        {
            let home = self
                .ast
                .scope(scope)
                .decls
                .first()
                .expect("uses_super literals declare the home slot");
            if let Resolution::Context { slot, .. } = self
                .resolved
                .resolution_for_decl(scope, home.name)
                .expect("home slot declared")
            {
                emit(&mut self.code, Opcode::Load, &[obj]);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
            }
        }
        if let Some(save) = ctx_save {
            emit(&mut self.code, Opcode::PopContext, &[save]);
            self.next_temp -= 1;
        }
        emit(&mut self.code, Opcode::Load, &[obj]);
        self.pop_value();
        Ok(())
    }

    // -- statements ------------------------------------------------------------

    fn stmt(&mut self, node: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(node) {
            Node::ExprStmt { expr } => {
                self.expr(expr)?;
                // a script-level value-producing statement: record the
                // completion (non-value statements leave it untouched)
                if let Some(completion) = self.completion {
                    emit(&mut self.code, Opcode::Store, &[completion]);
                }
                Ok(())
            }
            Node::VarDecl { kind, decls } => self.emit_var_decl(kind, decls),
            Node::Block { stmts } => {
                let scope = self.ast.node_scope(node);
                if let Some(s) = scope {
                    self.scopes.push(s);
                }
                let result = (|| {
                    for &s in self.ast.list_items(stmts) {
                        self.stmt(s)?;
                    }
                    Ok(())
                })();
                if scope.is_some() {
                    self.scopes.pop();
                }
                result
            }
            Node::If { cond, then, else_ } => {
                self.expr(cond)?;
                let mut else_l = Label::new();
                emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut else_l);
                self.stmt(then)?;
                match else_ {
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
            Node::While {
                ref labels,
                cond,
                body,
            } => self.emit_while(cond, body, labels.clone()),
            Node::For {
                ref labels,
                init,
                cond,
                next,
                body,
            } => self.emit_for(node, init, cond, next, body, labels.clone()),
            Node::ForIn {
                ref labels,
                left,
                object,
                body,
            } => self.emit_for_in(node, left, object, body, labels.clone()),
            Node::Return { value } => {
                // derived constructors: `return v` returns v only when it is
                // an object; `undefined` (and fallthrough) return `this`,
                // which must be initialized (ES 9.2.2.1)
                if self.is_derived_ctor() {
                    return self.emit_derived_return(value);
                }
                match value {
                    Some(v) => self.expr(v)?,
                    None => self.emit_load_constant(Constant::Undefined),
                }
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                emit(&mut self.code, Opcode::Return, &[]);
                Ok(())
            }
            Node::Throw { expr } => {
                self.expr(expr)?;
                emit(&mut self.code, Opcode::Throw, &[]);
                Ok(())
            }
            Node::TryCatch {
                try_block,
                catch_param,
                catch_block,
                finally_block,
            } => self.emit_try_catch(node, try_block, catch_param, catch_block, finally_block),
            Node::Switch {
                ref labels,
                disc,
                cases,
            } => self.emit_switch(node, disc, cases, labels.clone()),
            Node::FunctionDecl { function } => {
                let idx = self.add_constant(Constant::Callable(function));
                emit(&mut self.code, Opcode::CreateClosure, &[idx]);
                let name = self
                    .ast
                    .function(function)
                    .name
                    .expect("named function decl");
                let Some(scope) = self.find_decl_scope(name) else {
                    return self.err(node, "function declaration without a binding");
                };
                let res = self
                    .resolved
                    .resolution_for_decl(scope, name)
                    .expect("declared");
                self.store_decl(res, name);
                Ok(())
            }
            Node::Labeled { label, body } => {
                // a label on a non-loop/switch statement: a break-only
                // breakable around the body (ES 14.13)
                self.breakables.push(Breakable {
                    labels: vec![label],
                    breaks: Label::new(),
                    continues: None,
                    unwind_ctx: None,
                });
                let result = self.stmt(body);
                let (mut breaks, _) = self.end_breakable();
                breaks.bind(&self.code);
                breaks.patch_all(&mut self.code);
                result
            }
            Node::Break { label } => self.emit_break_continue(node, label, true),
            Node::Continue { label } => self.emit_break_continue(node, label, false),
            Node::Empty => Ok(()),
            Node::ClassDecl { class } => self.emit_class(node, class),
            _ => self.err(node, "statement"),
        }
    }

    fn emit_var_decl(
        &mut self,
        kind: VarKind,
        decls: parser::NodeList,
    ) -> Result<(), CompileError> {
        for &d in self.ast.list_items(decls) {
            let Node::VarDeclarator { target, init } = *self.ast.node(d) else {
                return self.err(d, "var declarator");
            };
            match *self.ast.node(target) {
                Node::ArrayPattern { .. } | Node::ObjectPattern { .. } => {
                    let Some(init) = init else {
                        return self.err(d, "destructuring declaration needs an initializer");
                    };
                    self.expr(init)?;
                    let value = self.push_value();
                    self.emit_pattern(target, value, true)?;
                    self.pop_value();
                }
                Node::Identifier { sym } => {
                    let Some(scope) = self.find_decl_scope(sym) else {
                        return self.err(d, "declaration without a binding");
                    };
                    let res = self
                        .resolved
                        .resolution_for_decl(scope, sym)
                        .expect("declared");
                    match init {
                        Some(init) => {
                            self.expr(init)?;
                            if self.is_anon_function(init) {
                                self.emit_set_name_const(self.ast.symbol(sym));
                            }
                            self.store_decl(res, sym);
                        }
                        None if kind == VarKind::Var || res == Resolution::GlobalObject => {
                            // `var` (and REPL globals) initialize to undefined;
                            // local let/const without init stay the hole (TDZ)
                            self.emit_load_constant(Constant::Undefined);
                            self.store_decl(res, sym);
                        }
                        None => {}
                    }
                }
                _ => return self.err(target, "var declarator target"),
            }
        }
        Ok(())
    }

    /// `for (left in object) body` (ES 14.7.5): the subject evaluates in
    /// the head scope (lexical bindings are in TDZ there), null/undefined
    /// enumerate nothing, and the loop pulls one key per iteration from
    /// the hidden enumerator. The assignment target re-evaluates per
    /// iteration (ForIn/OfBodyEvaluation: "it may be evaluated
    /// repeatedly").
    fn emit_for_in(
        &mut self,
        node: NodeId,
        left: NodeId,
        object: NodeId,
        body: NodeId,
        labels: Vec<Symbol>,
    ) -> Result<(), CompileError> {
        let scope = self.ast.node_scope(node);
        let per_iteration = self.resolved.per_iteration_loops.contains(&node);
        // lexical heads (`for (let/const k in …)`) own a block context;
        // captured heads get a fresh copy per iteration so closures in
        // the body capture per-iteration bindings (ES 14.7.5.7)
        // the loop's permanent temps (context save + loop-context register)
        self.with_temps(|g| {
            let mut loop_ctx: Option<u32> = None;
            let ctx_save = scope
                .filter(|s| !g.ast.scope(*s).decls.is_empty())
                .map(|s| {
                    let count = g.ast.scope(s).decls.len() as u32;
                    emit(&mut g.code, Opcode::CreateBlockContext, &[count]);
                    let save = g.reserve_temp();
                    let lc = g.reserve_temp();
                    emit(&mut g.code, Opcode::Store, &[lc]);
                    emit(&mut g.code, Opcode::PushContext, &[save]);
                    loop_ctx = Some(lc);
                    save
                });
            let inner = |g: &mut Self| -> Result<(), CompileError> {
                // head: subject → enumerator (undefined for nullish subjects;
                // ForInNext(undefined) is immediately done). Lexical heads
                // are in TDZ here: `for (let k in k)` throws (ES 14.7.5.6)
                g.expr(object)?;
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
                    let count = g
                        .ast
                        .scope(scope.expect("lexical loop heads own their scope"))
                        .decls
                        .len() as u32;
                    emit(&mut g.code, Opcode::CreateBlockContext, &[count]);
                    emit(&mut g.code, Opcode::PushContext, &[save]);
                }

                // assign the key to the target (per iteration), then the body
                g.emit_for_in_assign(left, key)?;
                g.stmt(body)?;

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
            };
            match scope {
                Some(s) => g.scoped(s, inner),
                None => inner(g),
            }
        })
    }

    /// Store the enumeration key (in register `key`) into the for-in
    /// assignment target. Declaration heads assign their single binding
    /// (patterns destructure); expression targets are stored through the
    /// normal assignment machinery.
    fn emit_for_in_assign(&mut self, left: NodeId, key: u32) -> Result<(), CompileError> {
        match *self.ast.node(left) {
            Node::VarDecl { decls, .. } => {
                let [declarator] = self.ast.list_items(decls) else {
                    return self.err(left, "for-in declarator");
                };
                let Node::VarDeclarator { target, init: None } = *self.ast.node(*declarator) else {
                    return self.err(left, "for-in declarator initializer");
                };
                match *self.ast.node(target) {
                    Node::Identifier { sym } => {
                        emit(&mut self.code, Opcode::Load, &[key]);
                        self.store_name(target, sym)
                    }
                    Node::ObjectPattern { .. } | Node::ArrayPattern { .. } => {
                        self.emit_pattern(target, key, true)
                    }
                    _ => self.err(left, "for-in binding pattern"),
                }
            }
            Node::Identifier { sym } => {
                emit(&mut self.code, Opcode::Load, &[key]);
                self.store_name(left, sym)
            }
            Node::Property { .. } => {
                let store = self.prepare_property_store(left)?;
                emit(&mut self.code, Opcode::Load, &[key]);
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(left, "for-in assignment target"),
        }
    }

    /// Bind a loop's tail: breaks land after the loop, continues patch
    /// to their bind points inside it, and the back-edge returns to
    /// `head`.
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
        cond: NodeId,
        body: NodeId,
        labels: Vec<Symbol>,
    ) -> Result<(), CompileError> {
        // head: cond; JumpIfFalsy breaks; body; continues; back-edge
        let head = self.code.len();
        self.expr(cond)?;
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

    fn emit_for(
        &mut self,
        node: NodeId,
        init: Option<NodeId>,
        cond: Option<NodeId>,
        next: Option<NodeId>,
        body: NodeId,
        labels: Vec<Symbol>,
    ) -> Result<(), CompileError> {
        let scope = self.ast.node_scope(node);
        let per_iteration = self.resolved.per_iteration_loops.contains(&node);
        // `for (let/const i = …; …; …)`: the head bindings own a block
        // context. Captured bindings require a fresh environment per
        // iteration with the values copied forward, so closures in the
        // body observe per-iteration bindings (ES 14.7.5.4,
        // CreatePerIterationEnvironment)
        let for_ctx = scope
            .filter(|s| !self.ast.scope(*s).decls.is_empty())
            .map(|s| (s, self.ast.scope(s).decls.len() as u32));
        // the loop's permanent temps (context save + loop-context
        // register + per-iteration copy registers + iteration context)
        self.with_temps(|g| {
            let mut loop_ctx: Option<u32> = None;
            let ctx_save = for_ctx.map(|(_, count)| {
                emit(&mut g.code, Opcode::CreateBlockContext, &[count]);
                let save = g.reserve_temp();
                let lc = g.reserve_temp();
                emit(&mut g.code, Opcode::Store, &[lc]);
                emit(&mut g.code, Opcode::PushContext, &[save]);
                loop_ctx = Some(lc);
                save
            });
            let inner = |g: &mut Self| -> Result<(), CompileError> {
                if let Some(init) = init {
                    match *g.ast.node(init) {
                        Node::ExprStmt { .. } | Node::VarDecl { .. } | Node::Empty => {
                            g.stmt(init)?
                        }
                        _ => return g.err(init, "for loop initializer"),
                    }
                }
                // per-iteration state: copy registers for the head bindings
                // and the current iteration's context (merge points restore
                // absolutely — a labelled continue may bypass the pops of
                // nested loops still holding contexts)
                let copies: Vec<u32>;
                let mut iter_ctx: Option<u32> = None;
                if per_iteration {
                    let (_, count) = for_ctx.expect("per-iteration loops own a scope");
                    copies = (0..count).map(|_| g.reserve_temp()).collect();
                    let iter_ctx_reg = g.reserve_temp();
                    // the first iteration starts from a copy of the head
                    // context: values copied out, fresh sibling pushed
                    g.emit_iteration_context_copy(Some(iter_ctx_reg), count, &copies, ctx_save);
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
                if let Some(cond) = cond {
                    g.expr(cond)?;
                    emit_jump(
                        &mut g.code,
                        Opcode::JumpIfFalsy,
                        &mut g.breakables.last_mut().unwrap().breaks,
                    );
                }
                g.stmt(body)?;
                g.breakables
                    .last_mut()
                    .unwrap()
                    .continues
                    .as_mut()
                    .unwrap()
                    .bind(&g.code);
                if let Some(iter_ctx) = iter_ctx {
                    let (_, count) = for_ctx.expect("per-iteration loops own a scope");
                    // absolute restore to this iteration's context, then
                    // copy its values into a fresh sibling for the next
                    // iteration (ES 14.7.5.4: the copy precedes the update)
                    emit(&mut g.code, Opcode::PopContext, &[iter_ctx]);
                    g.emit_iteration_context_copy(Some(iter_ctx), count, &copies, ctx_save);
                } else if let Some(lc) = loop_ctx {
                    emit(&mut g.code, Opcode::PopContext, &[lc]);
                }
                if let Some(next) = next {
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
            };
            match scope {
                Some(s) => g.scoped(s, inner),
                None => inner(g),
            }
        })
    }

    /// Copy a loop head's bindings into a fresh sibling context: read
    /// the slots out of the current context, pop to the shared outer,
    /// create a fresh context (holes) and write the values back in.
    /// `ctx_save` holds the outer context (the PushContext invariant);
    /// when `iter_ctx` is given it receives the new context's value (the
    /// absolute restore target at merge points).
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

    /// Pop the innermost breakable; the caller must bind+patch the labels.
    fn end_breakable(&mut self) -> (Label, Option<Label>) {
        let state = self.breakables.pop().expect("breakable state");
        (state.breaks, state.continues)
    }

    fn emit_break_continue(
        &mut self,
        node: NodeId,
        label: Option<Symbol>,
        is_break: bool,
    ) -> Result<(), CompileError> {
        if is_break {
            let idx = match label {
                None => self.breakables.len().checked_sub(1),
                Some(name) => self
                    .breakables
                    .iter()
                    .rposition(|b| b.labels.contains(&name)),
            };
            let Some(idx) = idx else {
                return self.err(node, "break outside a breakable statement");
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
            Some(name) => self
                .breakables
                .iter()
                .rposition(|b| b.labels.contains(&name) && b.continues.is_some()),
        };
        let Some(idx) = idx else {
            return self.err(node, "continue outside a loop");
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

    fn emit_switch(
        &mut self,
        node: NodeId,
        disc: NodeId,
        cases: parser::NodeList,
        labels: Vec<Symbol>,
    ) -> Result<(), CompileError> {
        let scope = self.ast.node_scope(node);
        let inner = |g: &mut Self| -> Result<(), CompileError> {
            // evaluate the discriminant once into a temp
            g.expr(disc)?;
            let d = g.push_value();
            g.breakables.push(Breakable {
                labels,
                breaks: Label::new(),
                continues: None,
                unwind_ctx: None,
            });

            let cases = g.ast.list_items(cases);
            let mut bodies: Vec<Label> = (0..cases.len()).map(|_| Label::new()).collect();
            let mut default_idx = None;

            for (i, &case) in cases.iter().enumerate() {
                let Node::SwitchCase { test, .. } = *g.ast.node(case) else {
                    return g.err(case, "switch case");
                };
                match test {
                    Some(test) => {
                        g.expr(test)?;
                        let t = g.push_value();
                        emit(&mut g.code, Opcode::Load, &[d]);
                        emit(&mut g.code, Opcode::EqualStrict, &[t]);
                        g.pop_value();
                        emit_jump(&mut g.code, Opcode::JumpIfTruthy, &mut bodies[i]);
                    }
                    None => default_idx = Some(i),
                }
            }

            // no case matched: the default body, or past the switch
            let mut end = Label::new();
            match default_idx {
                Some(i) => emit_jump(&mut g.code, Opcode::Jump, &mut bodies[i]),
                None => emit_jump(&mut g.code, Opcode::Jump, &mut end),
            }

            // bodies execute in order; fallthrough is just sequential layout
            for (i, &case) in cases.iter().enumerate() {
                bodies[i].bind(&g.code);
                bodies[i].patch_all(&mut g.code);
                let Node::SwitchCase { stmts, .. } = *g.ast.node(case) else {
                    return g.err(case, "switch case");
                };
                for &s in g.ast.list_items(stmts) {
                    g.stmt(s)?;
                }
            }

            let (mut breaks, _) = g.end_breakable();
            breaks.bind(&g.code);
            breaks.patch_all(&mut g.code);
            end.bind(&g.code);
            end.patch_all(&mut g.code);
            g.pop_value(); // discriminant
            Ok(())
        };
        match scope {
            Some(s) => self.scoped(s, inner),
            None => inner(self),
        }
    }

    fn emit_try_catch(
        &mut self,
        node: NodeId,
        try_block: NodeId,
        catch_param: Option<NodeId>,
        catch_block: Option<NodeId>,
        finally_block: Option<NodeId>,
    ) -> Result<(), CompileError> {
        if finally_block.is_some() {
            return self.err(node, "finally blocks");
        }
        self.with_temps(|g| {
            // snapshot the current context: an exception may unwind out of
            // context-owning constructs (lexical loop heads, class evaluation)
            // whose PushContext the handler entry bypasses; the handler
            // restores absolutely so the catch block's context-slot accesses
            // see the context of the enclosing statement
            let try_ctx = g.reserve_temp();
            emit(&mut g.code, Opcode::LdaContext, &[]);
            emit(&mut g.code, Opcode::Store, &[try_ctx]);
            let try_start = g.code.len();
            g.stmt(try_block)?;
            let try_end = g.code.len();

            let mut end = Label::new();
            emit_jump(&mut g.code, Opcode::Jump, &mut end);

            // handler entry: the exception arrives in the accumulator
            let handler_pc = g.code.len();
            emit(&mut g.code, Opcode::PopContext, &[try_ctx]);
            let inner = |g: &mut Self| -> Result<(), CompileError> {
                if let (Some(param), Some(block)) = (catch_param, catch_block) {
                    match *g.ast.node(param) {
                        Node::ArrayPattern { .. } | Node::ObjectPattern { .. } => {
                            let value = g.push_value();
                            g.emit_pattern(param, value, true)?;
                            g.pop_value();
                        }
                        Node::Identifier { sym } => {
                            let scope = g.ast.node_scope(node);
                            let res = g
                                .resolved
                                .resolution_for_decl(scope.expect("catch scope"), sym)
                                .expect("catch param declared");
                            g.store_resolution(res);
                        }
                        _ => return g.err(param, "catch parameter"),
                    }
                    g.stmt(block)?;
                }
                Ok(())
            };
            let scope = g.ast.node_scope(node);
            match scope {
                Some(s) => g.scoped(s, inner)?,
                None => inner(g)?,
            }

            g.handlers.push(HandlerEntry {
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

    /// The names of this function's context slots (parallel to the slot
    /// indices the resolver allocated), for dynamic name resolution.
    fn context_names(&self) -> Vec<Vec<u8>> {
        let layout = self.resolved.layout(self.fid);
        let mut names: Vec<Option<Vec<u8>>> = vec![None; layout.context_slots as usize];
        if let Some(slot) = layout.this_slot {
            names[slot as usize] = Some(b"this".to_vec());
        }
        if let Some(slot) = layout.new_target_slot {
            names[slot as usize] = Some(b".new.target".to_vec());
        }
        if let Some(slot) = layout.this_function_slot {
            names[slot as usize] = Some(b".this_function".to_vec());
        }
        for s in 0..self.ast.scope_count() {
            let scope = ScopeId(s as u32);
            // class scopes and lexical for-head scopes own their own
            // contexts, not this function's
            if matches!(self.ast.scope(scope).kind, ScopeKind::Class)
                || (self.ast.scope(scope).kind == ScopeKind::For
                    && !self.ast.scope(scope).decls.is_empty())
            {
                continue;
            }
            // owning function of this scope: nearest enclosing function scope
            let mut cur = scope;
            let owner = loop {
                if let Some(f) = self.ast.scope(cur).function {
                    break f;
                }
                cur = self
                    .ast
                    .scope(cur)
                    .parent
                    .expect("scope chain ends at script scope");
            };
            if owner != self.fid {
                continue;
            }
            for decl in &self.ast.scope(scope).decls {
                if let Some(Resolution::Context { slot, .. }) =
                    self.resolved.resolution_for_decl(scope, decl.name)
                {
                    names[slot as usize] = Some(self.ast.symbol(decl.name).to_vec());
                }
            }
        }
        names
            .into_iter()
            .map(|n| n.expect("context slot must have a name"))
            .collect()
    }

    fn is_derived_ctor(&self) -> bool {
        self.ast
            .function(self.fid)
            .kind
            .is_derived_class_constructor()
    }

    /// Whether the class this function constructs has instance fields (the
    /// fields attach to the class's constructor).
    fn ctor_has_instance_fields(&self) -> bool {
        self.fn_has_instance_fields(self.fid)
    }

    fn fn_has_instance_fields(&self, fid: FunctionId) -> bool {
        self.ctor_class(fid)
            .is_some_and(|c| self.class_has_instance_fields(c))
    }

    fn ctor_class(&self, fid: FunctionId) -> Option<parser::ClassId> {
        (0..self.ast.class_count() as u32)
            .map(parser::ClassId)
            .find(|&c| self.ast.class(c).ctor == fid)
    }

    fn class_has_instance_fields(&self, class: parser::ClassId) -> bool {
        self.ast
            .class(class)
            .members
            .iter()
            .any(|m| m.kind == PropKind::Field && !m.is_static)
    }

    /// Load this function's own `this`: the context slot when captured
    /// (the bind target of super(), shared with nested arrows), else the
    /// receiver register.
    fn emit_this_load_own(&mut self) {
        match self.resolved.layout(self.fid).this_slot {
            Some(slot) => emit(&mut self.code, Opcode::LoadContextSlot, &[slot, 0]),
            None => emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]),
        }
    }

    /// `return [expr]` inside a derived constructor: an object result wins,
    /// `undefined` (and missing) return `this` (initialization checked),
    /// other primitives make the [[Construct]] throw (ES 9.2.2.1 — the
    /// primitive escapes and the Construct opcode rejects it).
    fn emit_derived_return(&mut self, value: Option<NodeId>) -> Result<(), CompileError> {
        let Some(v) = value else {
            // `return;` → return this (loaded while the frame context is
            // still pushed: the captured-this slot lives in it)
            self.emit_this_load_own();
            emit(&mut self.code, Opcode::ThrowSuperNotCalledIfHole, &[]);
            emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            emit(&mut self.code, Opcode::Return, &[]);
            return Ok(());
        };
        self.expr(v)?;
        let t = self.push_value();
        // acc === undefined → return this
        self.emit_load_constant(Constant::Undefined);
        let u = self.push_value();
        emit(&mut self.code, Opcode::Load, &[t]);
        emit(&mut self.code, Opcode::EqualStrict, &[u]);
        let mut is_obj = Label::new();
        emit_jump(&mut self.code, Opcode::JumpIfFalsy, &mut is_obj);
        // undefined → return this (context still pushed for the slot read)
        self.emit_this_load_own();
        emit(&mut self.code, Opcode::ThrowSuperNotCalledIfHole, &[]);
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
        let info = self.ast.function(self.fid);
        if info.kind.is_generator() {
            let span = self.ast.span(info.body.expect("parsed"));
            return Err(CompileError::new(span, "generator functions"));
        }
        let body = info.body.expect("function must be parsed");
        let layout = self.resolved.layout(self.fid);
        // the script (and each eval compilation) tracks its completion value
        // in a dedicated register between ctx_save and the temps
        self.completion = (self.fid.0 == 0).then_some(layout.register_count + 1);
        self.ctx_save = layout.register_count as i32;
        self.reg_base = layout.register_count + 1 + u32::from(self.completion.is_some());

        // prologue: one context per function (uniform chain), pushed onto
        // the frame context; locals below reg_base are born as the hole.
        // constants[0] is the context's shared ScopeInfo (slot names)
        let names = self.context_names();
        self.add_constant(Constant::ContextNames(names));
        emit(&mut self.code, Opcode::CreateFunctionContext, &[0]);
        emit(&mut self.code, Opcode::PushContext, &[self.ctx_save as u32]);

        // parameters: non-simple lists (any default / pattern / rest) stage
        // the incoming arguments, hole-fill the parameter registers, and
        // initialize each binding in order (TDZ until its turn, ES 10.2.11
        // FunctionDeclarationInstantiation); simple lists only copy
        // context-allocated (captured) parameters into their slots
        let params: Vec<parser::Param> = self.ast.function(self.fid).params.clone();
        let non_simple = params.iter().any(|p| p.is_non_simple(self.ast));
        if let Some(fscope_id) = self.ast.node_scope(body) {
            let fscope = self.ast.scope(fscope_id);
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
                for (i, p) in params.iter().enumerate() {
                    if p.rest {
                        emit(&mut self.code, Opcode::CreateRestParameter, &[i as u32]);
                        emit(&mut self.code, Opcode::Store, &[staged_base + i as u32]);
                    }
                }
                // parameters start in their TDZ
                for i in 0..n {
                    emit(&mut self.code, Opcode::LdaHole, &[]);
                    emit(&mut self.code, Opcode::Store, &[(-(i as i32 + 2)) as u32]);
                }
                // left-to-right initialization
                for (i, p) in params.iter().enumerate() {
                    let reg = (-(i as i32 + 2)) as u32;
                    let staged = staged_base + i as u32;
                    if let Some(default) = p.default {
                        emit(&mut self.code, Opcode::Load, &[staged]);
                        let mut skip = Label::new();
                        emit_jump(&mut self.code, Opcode::JumpIfNotUndefined, &mut skip);
                        self.expr(default)?;
                        emit(&mut self.code, Opcode::Store, &[staged]);
                        skip.bind(&self.code);
                        skip.patch_all(&mut self.code);
                    }
                    // InitializeBinding: the register (and, when captured,
                    // the context slot) receives the value
                    emit(&mut self.code, Opcode::Load, &[staged]);
                    emit(&mut self.code, Opcode::Store, &[reg]);
                    if let Node::Identifier { sym } = *self.ast.node(p.target) {
                        if let Some(Resolution::Context { slot, .. }) =
                            self.resolved.resolution_for_decl(fscope_id, sym)
                        {
                            emit(&mut self.code, Opcode::Load, &[reg]);
                            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
                        }
                    }
                    // pattern parameters destructure the bound value
                    if matches!(
                        *self.ast.node(p.target),
                        Node::ArrayPattern { .. } | Node::ObjectPattern { .. }
                    ) {
                        self.emit_pattern(p.target, staged, true)?;
                    }
                }
                self.next_temp -= n; // staged parameter window
            } else {
                // context-allocated parameters: copy the argument into its
                // slot (captured params and direct-eval scopes force params
                // to contexts)
                let mut param_index = 0u32;
                for decl in &fscope.decls {
                    if decl.kind != parser::DeclKind::Param {
                        continue;
                    }
                    let reg = -(param_index as i32 + 2);
                    param_index += 1;
                    if let Some(Resolution::Context { slot, .. }) =
                        self.resolved.resolution_for_decl(fscope_id, decl.name)
                    {
                        emit(&mut self.code, Opcode::Load, &[reg as u32]);
                        emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
                    }
                }
            }
        }

        // store the receiver into the hidden this-slot when a nested
        // arrow captures it
        if let Some(slot) = layout.this_slot {
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }
        // expose new.target / the running closure to nested arrows
        // (arrow-delegated super() and arrow new.target reads)
        if let Some(slot) = layout.new_target_slot {
            emit(&mut self.code, Opcode::LdaNewTarget, &[]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }
        if let Some(slot) = layout.this_function_slot {
            emit(&mut self.code, Opcode::LdaCurrentClosure, &[]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }

        // the synthesized default derived constructor forwards every
        // argument to super() and returns the bound this (ES 15.7.13)
        if info.kind == parser::FunctionKind::DefaultDerivedConstructor {
            emit(&mut self.code, Opcode::ConstructSuperAllArgs, &[]);
            emit(&mut self.code, Opcode::Store, &[(-1i32) as u32]);
            if self.ctor_has_instance_fields() {
                // InitializeInstanceElements on the bound this:
                // native(ctor, instance)
                self.with_temps(|g| {
                    emit(&mut g.code, Opcode::LdaCurrentClosure, &[]);
                    let ctor = g.push_value();
                    emit(&mut g.code, Opcode::Load, &[(-1i32) as u32]);
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
        if info.kind == parser::FunctionKind::BaseClassConstructor
            && self.ctor_has_instance_fields()
        {
            self.with_temps(|g| {
                emit(&mut g.code, Opcode::LdaCurrentClosure, &[]);
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

        // the script's completion value starts as undefined; only
        // value-producing statements overwrite it (see `stmt` → ExprStmt)
        if let Some(completion) = self.completion {
            self.emit_load_constant(Constant::Undefined);
            emit(&mut self.code, Opcode::Store, &[completion]);
        }

        // field-initializer functions (`return <init>;`): an anonymous
        // function value is named after the field key (ES 15.7.19)
        if let Some(key) = info.field_key
            && let Some(value) = single_return_value(self.ast, body)
            && self.is_anon_function(value)
        {
            self.expr(value)?;
            self.emit_set_name_for_key_node(key);
            emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            emit(&mut self.code, Opcode::Return, &[]);
            return Ok(());
        }

        self.stmt(body)?;

        // fallthrough: the script yields its completion value; ordinary
        // functions return undefined; derived constructors return `this`
        // (initialization checked — super() must have run; the this load
        // happens while the frame context is still pushed, the captured
        // this slot lives in it)
        match self.completion {
            Some(completion) => {
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                emit(&mut self.code, Opcode::Load, &[completion]);
            }
            None if self.is_derived_ctor() => {
                self.emit_this_load_own();
                emit(&mut self.code, Opcode::ThrowSuperNotCalledIfHole, &[]);
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            }
            None => {
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
                self.emit_load_constant(Constant::Undefined);
            }
        }
        emit(&mut self.code, Opcode::Return, &[]);
        Ok(())
    }
}
