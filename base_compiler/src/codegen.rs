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
        Opcode::Jump | Opcode::JumpLoop | Opcode::JumpIfTruthy | Opcode::JumpIfFalsy
    ));
    let pc = code.len();
    code.push(Opcode::Wide as u8);
    code.push(op as u8);
    code.extend_from_slice(&[0, 0]);
    label.patch_here(pc);
}

/// A breakable statement: loops add a continue target, switches don't.
struct Breakable {
    label: Option<Symbol>,
    breaks: Label,
    continues: Option<Label>,
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

pub fn generate(ast: &Ast, resolved: &Resolved) -> Result<CompiledScript, CompileError> {
    let mut functions = Vec::with_capacity(ast.function_count());
    for fid in 0..ast.function_count() {
        let fid = FunctionId(fid as u32);
        let mut generator = FunctionGen::new(ast, resolved, fid);
        generator.emit_function_body()?;
        debug_assert!(generator.next_temp == 0, "unbalanced temp allocation");
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
            Resolution::Param(i) => {
                emit(&mut self.code, Opcode::Store, &[(-(i as i32 + 2)) as u32])
            }
            Resolution::Local { reg, .. } => emit(&mut self.code, Opcode::Store, &[reg]),
            Resolution::Context { slot, depth, .. } => {
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth])
            }
            Resolution::GlobalObject => {
                unreachable!("global stores need the name; use store_global")
            }
            Resolution::Dynamic => unreachable!("dynamic resolutions are rejected on load"),
            Resolution::This { .. } => unreachable!("this has no store target"),
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
            Resolution::Param(i) => {
                emit(&mut self.code, Opcode::Load, &[(-(i as i32 + 2)) as u32]);
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
            Node::Hole => self.err(node, "array elision outside array literals"),
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
            TokenKind::Delete => self.err(node, "delete"),
            _ => self.err(node, "unary operator"),
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
            TokenKind::In | TokenKind::QuestionDot | TokenKind::Nullish => self.err(
                node,
                match op {
                    TokenKind::In => "in operator",
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
                self.store_name(target, sym)
            }
            Node::Property { .. } | Node::SuperProperty { .. } => {
                let store = self.prepare_property_store(target)?;
                self.expr(value)?;
                self.emit_property_store(&store);
                self.release_store(&store);
                Ok(())
            }
            _ => self.err(target, "assignment target"),
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
        let needs_ctx = info.name.is_some() || info.uses_super;

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

        // members in declaration order; the constructor is already installed
        for m in &info.members {
            if m.is_constructor {
                continue;
            }
            let target = if m.is_static { ctor } else { proto };
            let function = match *self.ast.node(m.value) {
                Node::FunctionExpr { function } => function,
                _ => return self.err(m.value, "class member function"),
            };
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
                PropKind::Init => return self.err(m.value, "class field initializers"),
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

    /// Reserve a temp register without storing the accumulator.
    fn reserve_temp(&mut self) -> u32 {
        let r = self.reg_base + self.next_temp;
        self.next_temp += 1;
        self.max_temps = self.max_temps.max(self.next_temp);
        r
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

    /// `super(...)`: construct the superclass with the frame's new.target and
    /// initialize `this` with the result (ES 15.4.3). Evaluates to the new
    /// `this`.
    fn emit_super_call(
        &mut self,
        _node: NodeId,
        args: parser::NodeList,
    ) -> Result<(), CompileError> {
        let items = self.ast.list_items(args).to_vec();
        let argc = items.len();
        let arg_base = self.reg_base + self.next_temp;
        // reserve arguments + result so nested temps land above
        self.next_temp += argc as u32 + 1;
        for (i, &arg) in items.iter().enumerate() {
            self.expr(arg)?;
            emit(&mut self.code, Opcode::Store, &[arg_base + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(
            &mut self.code,
            Opcode::ConstructSuper,
            &[arg_base, argc as u32],
        );
        let result = arg_base + argc as u32;
        emit(&mut self.code, Opcode::Store, &[result]);
        // InitializeThisBinding: this must still be uninitialized
        emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
        emit(
            &mut self.code,
            Opcode::ThrowSuperAlreadyCalledIfNotHole,
            &[],
        );
        emit(&mut self.code, Opcode::Load, &[result]);
        emit(&mut self.code, Opcode::Store, &[(-1i32) as u32]);
        // keep a context-captured `this` (nested arrows) in sync
        if let Some(slot) = self.resolved.layout(self.fid).this_slot {
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }
        emit(&mut self.code, Opcode::Load, &[result]);
        self.next_temp -= argc as u32 + 1;
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
        _node: NodeId,
        props: parser::NodeList,
    ) -> Result<(), CompileError> {
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
                    if kind != PropKind::Init && kind != PropKind::Method {
                        return self.err(prop, "accessor properties in object literals");
                    }
                    if computed {
                        self.expr(key)?;
                        let k = self.push_value();
                        self.expr(value)?;
                        emit(&mut self.code, Opcode::StoreKeyedProperty, &[obj, k, 0]);
                        self.pop_value();
                    } else {
                        let name_idx = self.name_constant(key)?;
                        self.expr(value)?;
                        emit(
                            &mut self.code,
                            Opcode::StoreNamedProperty,
                            &[obj, name_idx, 0],
                        );
                    }
                }
                Node::Spread { .. } => return self.err(prop, "spread in object literals"),
                _ => return self.err(prop, "object literal property"),
            }
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
            Node::While { cond, body } => self.emit_while(cond, body, None),
            Node::For {
                init,
                cond,
                next,
                body,
            } => self.emit_for(node, init, cond, next, body, None),
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
            Node::Switch { disc, cases } => self.emit_switch(node, disc, cases, None),
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
            Node::Labeled { label, body } => self.emit_labeled(node, label, body),
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
            let Node::VarDeclarator { name, init } = *self.ast.node(d) else {
                return self.err(d, "var declarator");
            };
            let Some(scope) = self.find_decl_scope(name) else {
                return self.err(d, "declaration without a binding");
            };
            let res = self
                .resolved
                .resolution_for_decl(scope, name)
                .expect("declared");
            match init {
                Some(init) => {
                    self.expr(init)?;
                    self.store_decl(res, name);
                }
                None if kind == VarKind::Var || res == Resolution::GlobalObject => {
                    // `var` (and REPL globals) initialize to undefined;
                    // local let/const without init stay the hole (TDZ)
                    self.emit_load_constant(Constant::Undefined);
                    self.store_decl(res, name);
                }
                None => {}
            }
        }
        Ok(())
    }

    fn emit_while(
        &mut self,
        cond: NodeId,
        body: NodeId,
        label: Option<Symbol>,
    ) -> Result<(), CompileError> {
        // head: cond; JumpIfFalsy breaks; body; continues; back-edge
        let head = self.code.len();
        self.expr(cond)?;
        self.breakables.push(Breakable {
            label,
            breaks: Label::new(),
            continues: Some(Label::new()),
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
        let (mut breaks, continues) = self.end_breakable();
        breaks.bind(&self.code);
        breaks.patch_all(&mut self.code);
        continues.unwrap().patch_all(&mut self.code);
        back.bind_at(head);
        back.patch_all(&mut self.code);
        Ok(())
    }

    fn emit_for(
        &mut self,
        node: NodeId,
        init: Option<NodeId>,
        cond: Option<NodeId>,
        next: Option<NodeId>,
        body: NodeId,
        label: Option<Symbol>,
    ) -> Result<(), CompileError> {
        if self.resolved.per_iteration_loops.contains(&node) {
            return self.err(node, "per-iteration loop environments");
        }
        let scope = self.ast.node_scope(node);
        if let Some(s) = scope {
            self.scopes.push(s);
        }
        let result = (|| {
            if let Some(init) = init {
                match *self.ast.node(init) {
                    Node::ExprStmt { .. } | Node::VarDecl { .. } | Node::Empty => {
                        self.stmt(init)?
                    }
                    _ => return self.err(init, "for loop initializer"),
                }
            }
            // head: cond?; JumpIfFalsy breaks; body; continues; update; back-edge
            let head = self.code.len();
            self.breakables.push(Breakable {
                label,
                breaks: Label::new(),
                continues: Some(Label::new()),
            });
            if let Some(cond) = cond {
                self.expr(cond)?;
                emit_jump(
                    &mut self.code,
                    Opcode::JumpIfFalsy,
                    &mut self.breakables.last_mut().unwrap().breaks,
                );
            }
            self.stmt(body)?;
            self.breakables
                .last_mut()
                .unwrap()
                .continues
                .as_mut()
                .unwrap()
                .bind(&self.code);
            if let Some(next) = next {
                self.expr(next)?;
            }
            let mut back = Label::new();
            emit_jump(&mut self.code, Opcode::JumpLoop, &mut back);
            let (mut breaks, continues) = self.end_breakable();
            breaks.bind(&self.code);
            breaks.patch_all(&mut self.code);
            continues.unwrap().patch_all(&mut self.code);
            back.bind_at(head);
            back.patch_all(&mut self.code);
            Ok(())
        })();
        if scope.is_some() {
            self.scopes.pop();
        }
        result
    }

    /// Pop the innermost breakable; the caller must bind+patch the labels.
    fn end_breakable(&mut self) -> (Label, Option<Label>) {
        let state = self.breakables.pop().expect("breakable state");
        (state.breaks, state.continues)
    }

    fn emit_labeled(
        &mut self,
        node: NodeId,
        label: Symbol,
        body: NodeId,
    ) -> Result<(), CompileError> {
        // a label on a loop/switch tags its breakable; elsewhere it wraps
        // a break-only breakable
        match *self.ast.node(body) {
            Node::While { cond, body } => self.emit_while(cond, body, Some(label)),
            Node::For {
                init,
                cond,
                next,
                body,
            } => self.emit_for(node, init, cond, next, body, Some(label)),
            Node::Switch { disc, cases } => self.emit_switch(node, disc, cases, Some(label)),
            _ => {
                self.breakables.push(Breakable {
                    label: Some(label),
                    breaks: Label::new(),
                    continues: None,
                });
                let result = self.stmt(body);
                let (mut breaks, _) = self.end_breakable();
                breaks.bind(&self.code);
                breaks.patch_all(&mut self.code);
                result
            }
        }
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
                Some(name) => self.breakables.iter().rposition(|b| b.label == Some(name)),
            };
            let Some(idx) = idx else {
                return self.err(node, "break outside a breakable statement");
            };
            let state = &mut self.breakables[idx];
            emit_jump(&mut self.code, Opcode::Jump, &mut state.breaks);
            return Ok(());
        }
        let idx = match label {
            None => self.breakables.iter().rposition(|b| b.continues.is_some()),
            Some(name) => self
                .breakables
                .iter()
                .rposition(|b| b.label == Some(name) && b.continues.is_some()),
        };
        let Some(idx) = idx else {
            return self.err(node, "continue outside a loop");
        };
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
        label: Option<Symbol>,
    ) -> Result<(), CompileError> {
        let scope = self.ast.node_scope(node);
        if let Some(s) = scope {
            self.scopes.push(s);
        }
        let result = (|| {
            // evaluate the discriminant once into a temp
            self.expr(disc)?;
            let d = self.push_value();
            self.breakables.push(Breakable {
                label,
                breaks: Label::new(),
                continues: None,
            });

            let cases = self.ast.list_items(cases);
            let mut bodies: Vec<Label> = (0..cases.len()).map(|_| Label::new()).collect();
            let mut default_idx = None;

            for (i, &case) in cases.iter().enumerate() {
                let Node::SwitchCase { test, .. } = *self.ast.node(case) else {
                    return self.err(case, "switch case");
                };
                match test {
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
            for (i, &case) in cases.iter().enumerate() {
                bodies[i].bind(&self.code);
                bodies[i].patch_all(&mut self.code);
                let Node::SwitchCase { stmts, .. } = *self.ast.node(case) else {
                    return self.err(case, "switch case");
                };
                for &s in self.ast.list_items(stmts) {
                    self.stmt(s)?;
                }
            }

            let (mut breaks, _) = self.end_breakable();
            breaks.bind(&self.code);
            breaks.patch_all(&mut self.code);
            end.bind(&self.code);
            end.patch_all(&mut self.code);
            self.pop_value(); // discriminant
            Ok(())
        })();
        if scope.is_some() {
            self.scopes.pop();
        }
        result
    }

    fn emit_try_catch(
        &mut self,
        node: NodeId,
        try_block: NodeId,
        catch_param: Option<Symbol>,
        catch_block: Option<NodeId>,
        finally_block: Option<NodeId>,
    ) -> Result<(), CompileError> {
        if finally_block.is_some() {
            return self.err(node, "finally blocks");
        }
        let try_start = self.code.len();
        self.stmt(try_block)?;
        let try_end = self.code.len();

        let mut end = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut end);

        // handler entry: the exception arrives in the accumulator
        let handler_pc = self.code.len();
        if let (Some(param), Some(block)) = (catch_param, catch_block) {
            let scope = self.ast.node_scope(node);
            if let Some(s) = scope {
                self.scopes.push(s);
            }
            let res = self
                .resolved
                .resolution_for_decl(scope.expect("catch scope"), param)
                .expect("catch param declared");
            self.store_resolution(res);
            self.stmt(block)?;
            if scope.is_some() {
                self.scopes.pop();
            }
        }

        self.handlers.push(HandlerEntry {
            try_start,
            try_end,
            handler_pc,
        });
        end.bind(&self.code);
        end.patch_all(&mut self.code);
        Ok(())
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
        for s in 0..self.ast.scope_count() {
            let scope = ScopeId(s as u32);
            // class scopes own their own contexts, not this function's
            if self.ast.scope(scope).kind == ScopeKind::Class {
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

    /// `return [expr]` inside a derived constructor: an object result wins,
    /// `undefined` (and missing) return `this` (initialization checked),
    /// other primitives make the [[Construct]] throw (ES 9.2.2.1 — the
    /// primitive escapes and the Construct opcode rejects it).
    fn emit_derived_return(&mut self, value: Option<NodeId>) -> Result<(), CompileError> {
        let Some(v) = value else {
            // `return;` → return this
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
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
        emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
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

        // context-allocated parameters: copy the argument into its slot
        // (captured params and direct-eval scopes force params to contexts)
        if let Some(fscope_id) = self.ast.node_scope(body) {
            let fscope = self.ast.scope(fscope_id);
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

        // store the receiver into the hidden this-slot when a nested
        // arrow captures it
        if let Some(slot) = layout.this_slot {
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
            emit(&mut self.code, Opcode::StoreContextSlot, &[slot, 0]);
        }

        // the synthesized default derived constructor forwards every
        // argument to super() and returns the bound this (ES 15.7.13)
        if info.kind == parser::FunctionKind::DefaultDerivedConstructor {
            emit(&mut self.code, Opcode::ConstructSuperAllArgs, &[]);
            emit(&mut self.code, Opcode::Store, &[(-1i32) as u32]);
            emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
            emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
            emit(&mut self.code, Opcode::Return, &[]);
            return Ok(());
        }

        // the script's completion value starts as undefined; only
        // value-producing statements overwrite it (see `stmt` → ExprStmt)
        if let Some(completion) = self.completion {
            self.emit_load_constant(Constant::Undefined);
            emit(&mut self.code, Opcode::Store, &[completion]);
        }

        self.stmt(body)?;

        // fallthrough: the script yields its completion value; ordinary
        // functions return undefined; derived constructors return `this`
        // (initialization checked — super() must have run)
        emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
        match self.completion {
            Some(completion) => emit(&mut self.code, Opcode::Load, &[completion]),
            None if self.is_derived_ctor() => {
                emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
                emit(&mut self.code, Opcode::ThrowSuperNotCalledIfHole, &[]);
            }
            None => self.emit_load_constant(Constant::Undefined),
        }
        emit(&mut self.code, Opcode::Return, &[]);
        Ok(())
    }
}
