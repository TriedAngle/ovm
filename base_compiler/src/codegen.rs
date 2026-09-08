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

use bytecode::{Opcode, emit};
use parser::{
    Ast, FunctionId, Node, NodeId, PropKind, Resolution, Resolved, ScopeId, Symbol, TokenKind,
    VarKind,
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
    Named { obj: u32, name_idx: u32 },
    Keyed { obj: u32, key: u32 },
}

struct FunctionGen<'a> {
    ast: &'a Ast,
    resolved: &'a Resolved,
    /// lexical function nesting depths (parallel to the function table)
    depths: &'a [u32],
    fid: FunctionId,
    depth: u32,
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
    let mut depths = vec![0u32; ast.function_count()];
    compute_depths(ast, FunctionId(0), 0, &mut depths);

    let mut functions = Vec::with_capacity(ast.function_count());
    for fid in 0..ast.function_count() {
        let fid = FunctionId(fid as u32);
        let mut generator = FunctionGen::new(ast, resolved, &depths, fid);
        generator.emit_function_body()?;
        debug_assert!(generator.next_temp == 0, "unbalanced temp allocation");
        functions.push(generator.finish());
    }
    Ok(CompiledScript { functions })
}

/// Lexical nesting depth of every function (script = 0).
fn compute_depths(ast: &Ast, fid: FunctionId, depth: u32, out: &mut [u32]) {
    out[fid.0 as usize] = depth;
    let body = ast.function(fid).body.expect("function must be parsed");
    collect_functions(ast, body, depth, out);
}

fn collect_functions(ast: &Ast, node: NodeId, depth: u32, out: &mut [u32]) {
    match *ast.node(node) {
        Node::FunctionExpr { function } | Node::FunctionDecl { function } => {
            compute_depths(ast, function, depth + 1, out);
        }
        Node::Unary { expr, .. } | Node::Update { target: expr, .. } => {
            collect_functions(ast, expr, depth, out)
        }
        Node::Binary { lhs, rhs, .. } => {
            collect_functions(ast, lhs, depth, out);
            collect_functions(ast, rhs, depth, out);
        }
        Node::Assign { target, value, .. } => {
            collect_functions(ast, target, depth, out);
            collect_functions(ast, value, depth, out);
        }
        Node::Conditional { cond, then, else_ } => {
            collect_functions(ast, cond, depth, out);
            collect_functions(ast, then, depth, out);
            collect_functions(ast, else_, depth, out);
        }
        Node::Call { callee, args } => {
            collect_functions(ast, callee, depth, out);
            for &arg in ast.list_items(args) {
                collect_functions(ast, arg, depth, out);
            }
        }
        Node::New { callee, args } => {
            collect_functions(ast, callee, depth, out);
            if let Some(args) = args {
                for &arg in ast.list_items(args) {
                    collect_functions(ast, arg, depth, out);
                }
            }
        }
        Node::Property {
            object,
            key,
            computed,
        } => {
            collect_functions(ast, object, depth, out);
            if computed {
                collect_functions(ast, key, depth, out);
            }
        }
        Node::ArrayLiteral { elements } => {
            for &el in ast.list_items(elements) {
                collect_functions(ast, el, depth, out);
            }
        }
        Node::Spread { expr } | Node::ExprStmt { expr } | Node::Throw { expr } => {
            collect_functions(ast, expr, depth, out)
        }
        Node::Return { value } => {
            if let Some(v) = value {
                collect_functions(ast, v, depth, out);
            }
        }
        Node::ObjectLiteral { props } => {
            for &p in ast.list_items(props) {
                collect_functions(ast, p, depth, out);
            }
        }
        Node::ObjectProperty {
            key,
            value,
            computed,
            ..
        } => {
            if computed {
                collect_functions(ast, key, depth, out);
            }
            collect_functions(ast, value, depth, out);
        }
        Node::VarDecl { decls, .. } => {
            for &d in ast.list_items(decls) {
                collect_functions(ast, d, depth, out);
            }
        }
        Node::VarDeclarator { init, .. } => {
            if let Some(init) = init {
                collect_functions(ast, init, depth, out);
            }
        }
        Node::Block { stmts } => {
            for &s in ast.list_items(stmts) {
                collect_functions(ast, s, depth, out);
            }
        }
        Node::If { cond, then, else_ } => {
            collect_functions(ast, cond, depth, out);
            collect_functions(ast, then, depth, out);
            if let Some(e) = else_ {
                collect_functions(ast, e, depth, out);
            }
        }
        Node::While { cond, body } => {
            collect_functions(ast, cond, depth, out);
            collect_functions(ast, body, depth, out);
        }
        Node::For {
            init,
            cond,
            next,
            body,
        } => {
            if let Some(n) = init {
                collect_functions(ast, n, depth, out);
            }
            if let Some(c) = cond {
                collect_functions(ast, c, depth, out);
            }
            if let Some(n) = next {
                collect_functions(ast, n, depth, out);
            }
            collect_functions(ast, body, depth, out);
        }
        Node::TryCatch {
            try_block,
            catch_block,
            finally_block,
            ..
        } => {
            collect_functions(ast, try_block, depth, out);
            if let Some(c) = catch_block {
                collect_functions(ast, c, depth, out);
            }
            if let Some(f) = finally_block {
                collect_functions(ast, f, depth, out);
            }
        }
        Node::Switch { disc, cases } => {
            collect_functions(ast, disc, depth, out);
            for &case in ast.list_items(cases) {
                if let Node::SwitchCase { test, stmts } = *ast.node(case) {
                    if let Some(t) = test {
                        collect_functions(ast, t, depth, out);
                    }
                    for &s in ast.list_items(stmts) {
                        collect_functions(ast, s, depth, out);
                    }
                }
            }
        }
        Node::SwitchCase { .. } => unreachable!("switch cases handled by the switch walk"),
        Node::Labeled { body, .. } => collect_functions(ast, body, depth, out),
        Node::ClassExpr { .. } | Node::ClassDecl { .. } => {}
        Node::Identifier { .. }
        | Node::NumberLiteral(_)
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
}

impl<'a> FunctionGen<'a> {
    fn new(ast: &'a Ast, resolved: &'a Resolved, depths: &'a [u32], fid: FunctionId) -> Self {
        Self {
            ast,
            resolved,
            depths,
            fid,
            depth: depths[fid.0 as usize],
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

    /// Chain depth from the current frame context to the context holding
    /// `scope`'s slots: lexical function distance between use site and
    /// the declaration's owning function.
    fn context_depth(&self, scope: ScopeId) -> u32 {
        let mut s = scope;
        loop {
            if let Some(fid) = self.ast.scope(s).function {
                let owner = self.depths[fid.0 as usize];
                debug_assert!(self.depth >= owner, "captures are lexically enclosing");
                return self.depth - owner;
            }
            s = self
                .ast
                .scope(s)
                .parent
                .expect("scope chain ends at script scope");
        }
    }

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
            Resolution::Context { slot, scope, .. } => {
                let depth = self.context_depth(scope);
                emit(&mut self.code, Opcode::StoreContextSlot, &[slot, depth])
            }
            Resolution::GlobalObject => {
                unreachable!("global stores need the name; use store_global")
            }
            Resolution::Dynamic => unreachable!("dynamic resolutions are rejected on load"),
            Resolution::This(_) => unreachable!("this has no store target"),
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
                scope,
                hole_check,
            } => {
                let depth = self.context_depth(scope);
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
            Resolution::This(_) => unreachable!("this is not an identifier load"),
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
                    Some(Resolution::This(owner)) if owner != self.fid => {
                        let slot = self
                            .resolved
                            .layout(owner)
                            .this_slot
                            .expect("owner captures this");
                        let depth = self.depth - self.depths[owner.0 as usize];
                        emit(&mut self.code, Opcode::LoadContextSlot, &[slot, depth]);
                    }
                    _ => emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]),
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
            Node::ClassExpr { .. } => self.err(node, "classes"),
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
            Node::Property { .. } => {
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
            _ => self.err(target, "assignment target"),
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
        if let Node::Property { .. } = *self.ast.node(target) {
            let store = self.prepare_property_store(target)?;
            self.expr(value)?;
            let v = self.push_value();
            match &store {
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
            }
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
        if let Node::Property { .. } = *self.ast.node(target) {
            let store = self.prepare_property_store(target)?;
            match &store {
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
            }
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
            Node::ClassDecl { .. } => self.err(node, "classes"),
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

        // the script's completion value starts as undefined; only
        // value-producing statements overwrite it (see `stmt` → ExprStmt)
        if let Some(completion) = self.completion {
            self.emit_load_constant(Constant::Undefined);
            emit(&mut self.code, Opcode::Store, &[completion]);
        }

        self.stmt(body)?;

        // fallthrough: the script yields its completion value; ordinary
        // functions return undefined
        emit(&mut self.code, Opcode::PopContext, &[self.ctx_save as u32]);
        match self.completion {
            Some(completion) => emit(&mut self.code, Opcode::Load, &[completion]),
            None => self.emit_load_constant(Constant::Undefined),
        }
        emit(&mut self.code, Opcode::Return, &[]);
        Ok(())
    }
}
