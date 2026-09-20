//! The Kette AST walker: one pass per function after the desugar pass.
//!
//! Every `Node::Block` is a function (the same block can be invoked bare
//! or as a send target); the script body is function 0. Each block owns
//! exactly one resolver scope and, mirroring the JS backend, every
//! function pushes exactly one context, so a captured-slot `depth` is
//! exactly the number of `outer` hops from the running frame context.
//!
//! Locals that no nested block captures live in frame registers; captured
//! ones live in the function's context (parameters are copied into it in
//! the prologue). Expression results use the implicit temp stack above
//! the locals, like `js_compiler`.

use bytecode::{Opcode, emit};
use ir::{
    CallableKind, Constant, FunctionBuilder, FunctionId as IrFunctionId, HandlerEntry, Program,
};
use kette_parser::{Ast, Node, NodeId, NodeList, Resolution, Resolved, ScopeId, SlotKind, Symbol};

use crate::CompileError;
use crate::label::Label;

/// Emit a forced-wide relative jump to `label`; the offset is patched once
/// the label is bound.
fn emit_jump(code: &mut Vec<u8>, op: Opcode, label: &mut Label) {
    debug_assert!(matches!(
        op,
        Opcode::Jump | Opcode::JumpIfTruthy | Opcode::JumpIfFalsy
    ));
    let pc = code.len();
    code.push(Opcode::Wide as u8);
    code.push(op as u8);
    code.extend_from_slice(&[0, 0]);
    label.patch_here(pc);
}

/// `node -> function index` for every `Node::Block` (script is 0).
fn collect_blocks(ast: &Ast, node: NodeId, out: &mut Vec<NodeId>) {
    match *ast.node(node) {
        Node::Block { body, .. } => {
            out.push(node);
            collect_blocks(ast, body, out);
        }
        Node::StmtList { stmts } => {
            for &stmt in ast.list_items(stmts) {
                collect_blocks(ast, stmt, out);
            }
        }
        Node::Let { init, .. } => collect_blocks(ast, init, out),
        Node::ExprStmt { expr } => collect_blocks(ast, expr, out),
        Node::If { cond, then, else_ } => {
            collect_blocks(ast, cond, out);
            collect_blocks(ast, then, out);
            if let Some(else_) = else_ {
                collect_blocks(ast, else_, out);
            }
        }
        Node::ForIn { iter, body, .. } => {
            collect_blocks(ast, iter, out);
            collect_blocks(ast, body, out);
        }
        Node::While { cond, body } => {
            collect_blocks(ast, cond, out);
            collect_blocks(ast, body, out);
        }
        Node::Match { scrut, arms } => {
            collect_blocks(ast, scrut, out);
            for &arm in ast.list_items(arms) {
                if let Node::MatchArm { target, handler } = *ast.node(arm) {
                    if let Some(target) = target {
                        collect_blocks(ast, target, out);
                    }
                    collect_blocks(ast, handler, out);
                }
            }
        }
        Node::Object { slots } => {
            for &slot in ast.list_items(slots) {
                if let Node::Slot { key, value, .. } = *ast.node(slot) {
                    collect_blocks(ast, key, out);
                    collect_blocks(ast, value, out);
                }
            }
        }
        Node::Array { elements } => {
            for &element in ast.list_items(elements) {
                collect_blocks(ast, element, out);
            }
        }
        Node::Get { recv, .. } => collect_blocks(ast, recv, out),
        Node::Index { recv, key } => {
            collect_blocks(ast, recv, out);
            collect_blocks(ast, key, out);
        }
        Node::Send { recv, args, .. } => {
            collect_blocks(ast, recv, out);
            for &arg in ast.list_items(args) {
                collect_blocks(ast, arg, out);
            }
        }
        Node::Call { callee, args } => {
            collect_blocks(ast, callee, out);
            for &arg in ast.list_items(args) {
                collect_blocks(ast, arg, out);
            }
        }
        Node::Assign { target, value } => {
            collect_blocks(ast, target, out);
            collect_blocks(ast, value, out);
        }
        Node::Unary { expr, .. } => collect_blocks(ast, expr, out),
        Node::Binary { lhs, rhs, .. } => {
            collect_blocks(ast, lhs, out);
            collect_blocks(ast, rhs, out);
        }
        Node::Return { value } => collect_blocks(ast, value, out),
        Node::Try { body, handler } => {
            collect_blocks(ast, body, out);
            collect_blocks(ast, handler, out);
        }
        Node::Number(_)
        | Node::String(_)
        | Node::Bool(_)
        | Node::Null
        | Node::Self_
        | Node::Ident(_)
        | Node::Slot { .. }
        | Node::MatchArm { .. } => {}
    }
}

pub fn generate(ast: &Ast, resolved: &Resolved) -> Result<Program, CompileError> {
    let root = ast.root().expect("the parser sets a root");
    let mut blocks = Vec::new();
    collect_blocks(ast, root, &mut blocks);

    // Function ids are assigned in `blocks` order (script, then blocks in
    // pre-order), matching the `add_function` calls below.
    let mut block_fids = vec![None; ast.node_count()];
    for (i, &block) in blocks.iter().enumerate() {
        block_fids[block.0 as usize] = Some(i as u32 + 1);
    }

    let mut program = Program::with_capacity(blocks.len() + 1);
    {
        let scope = resolved.scope_of(root).expect("root owns the script scope");
        let mut generator = FunctionGen::new(ast, resolved, scope, root, 0, &block_fids);
        generator.generate()?;
        program.add_function(generator.finish());
    }
    for &block in &blocks {
        let Node::Block { params, body } = *ast.node(block) else {
            unreachable!("collected nodes are blocks");
        };
        let scope = resolved.scope_of(block).expect("block owns a scope");
        let mut generator = FunctionGen::new(
            ast,
            resolved,
            scope,
            body,
            ast.list_items(params).len(),
            &block_fids,
        );
        generator.generate()?;
        program.add_function(generator.finish());
    }
    Ok(program)
}

struct FunctionGen<'a> {
    ast: &'a Ast,
    resolved: &'a Resolved,
    scope: ScopeId,
    body: NodeId,
    /// per scope decl: context-allocated because a nested block captures it
    captured: Vec<bool>,
    /// per scope decl: frame register, when not captured
    reg_of: Vec<Option<u32>>,
    param_count: usize,
    /// `node -> function index` for closure creation
    block_fids: &'a [Option<u32>],
    code: Vec<u8>,
    constants: Vec<Constant>,
    handlers: Vec<HandlerEntry>,
    /// register holding the pushed-over context (prologue/epilogue)
    ctx_save: u32,
    /// block/script result register (last expression statement)
    completion: u32,
    /// first temp register, above the locals
    reg_base: u32,
    next_temp: u32,
    max_temps: u32,
    /// inline-cache slots consumed so far (in slots; each property-access
    /// site reserves a [state, handler] pair)
    feedback_slots: u32,
    /// decl index of the next `let` in this body (params come first)
    next_let: usize,
}

impl<'a> FunctionGen<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ast: &'a Ast,
        resolved: &'a Resolved,
        scope: ScopeId,
        body: NodeId,
        param_count: usize,
        block_fids: &'a [Option<u32>],
    ) -> Self {
        let decls = resolved.scope(scope).decls.len();
        let mut captured = vec![false; decls];
        for i in 0..ast.node_count() {
            if let Some(Resolution::Capture { scope: s, decl, .. }) =
                resolved.resolution(NodeId(i as u32))
                && s == scope
            {
                captured[decl as usize] = true;
            }
        }
        let mut reg_of = vec![None; decls];
        let mut local_count = 0u32;
        for (decl, is_captured) in captured.iter().enumerate() {
            if !is_captured {
                reg_of[decl] = Some(local_count);
                local_count += 1;
            }
        }
        Self {
            ast,
            resolved,
            scope,
            body,
            captured,
            reg_of,
            param_count,
            block_fids,
            code: Vec::new(),
            constants: Vec::new(),
            handlers: Vec::new(),
            ctx_save: local_count,
            completion: local_count + 1,
            reg_base: local_count + 2,
            next_temp: 0,
            max_temps: 0,
            feedback_slots: 0,
            next_let: param_count,
        }
    }

    fn finish(self) -> FunctionBuilder {
        FunctionBuilder {
            bytecode: self.code,
            constants: self.constants,
            handlers: self.handlers,
            name: None,
            register_count: self.reg_base + self.max_temps,
            kind: CallableKind::Method,
            arity: self.param_count as u32,
            length: self.param_count as u32,
            strict: true,
            feedback_count: self.feedback_slots,
        }
    }

    fn err<T>(&self, node: NodeId, feature: &'static str) -> Result<T, CompileError> {
        Err(CompileError::new(self.ast.span(node), feature))
    }

    // -- feedback -----------------------------------------------------------

    /// Reserve a `[state, handler]` feedback-slot pair for a property-access
    /// site and return the base index embedded as the site's feedback
    /// operand.
    fn feedback_slot(&mut self) -> u32 {
        let slot = self.feedback_slots;
        self.feedback_slots += 2;
        slot
    }

    // -- temps -------------------------------------------------------------

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

    fn add_constant(&mut self, c: Constant) -> u32 {
        self.constants.push(c);
        (self.constants.len() - 1) as u32
    }

    fn emit_load_constant(&mut self, c: Constant) {
        let idx = self.add_constant(c);
        emit(&mut self.code, Opcode::LoadConstant, &[idx]);
    }

    fn name_constant(&mut self, name: Symbol) -> u32 {
        self.add_constant(Constant::String(self.ast.symbol(name).to_vec()))
    }

    // -- locals ------------------------------------------------------------

    fn load_decl(&mut self, decl: usize) {
        if self.captured[decl] {
            emit(&mut self.code, Opcode::LoadContextSlot, &[decl as u32, 0]);
        } else {
            emit(
                &mut self.code,
                Opcode::Load,
                &[self.reg_of[decl].expect("uncaptured decl has a register")],
            );
        }
    }

    fn store_decl(&mut self, decl: usize) {
        if self.captured[decl] {
            emit(&mut self.code, Opcode::StoreContextSlot, &[decl as u32, 0]);
        } else {
            emit(
                &mut self.code,
                Opcode::Store,
                &[self.reg_of[decl].expect("uncaptured decl has a register")],
            );
        }
    }

    fn load_ident(&mut self, node: NodeId) {
        match self.resolved.resolution(node) {
            Some(Resolution::Local { decl, .. }) => self.load_decl(decl as usize),
            Some(Resolution::Capture { decl, depth, .. }) => {
                emit(&mut self.code, Opcode::LoadContextSlot, &[decl, depth]);
            }
            Some(Resolution::Global) | None => {
                let Node::Ident(sym) = *self.ast.node(node) else {
                    unreachable!("identifier nodes carry a symbol");
                };
                let idx = self.name_constant(sym);
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::LoadGlobal, &[idx, feedback]);
            }
        }
    }

    fn store_ident(&mut self, node: NodeId) {
        match self.resolved.resolution(node) {
            Some(Resolution::Local { decl, .. }) => self.store_decl(decl as usize),
            Some(Resolution::Capture { decl, depth, .. }) => {
                emit(&mut self.code, Opcode::StoreContextSlot, &[decl, depth]);
            }
            Some(Resolution::Global) | None => {
                let Node::Ident(sym) = *self.ast.node(node) else {
                    unreachable!("identifier nodes carry a symbol");
                };
                let idx = self.name_constant(sym);
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::StoreGlobal, &[idx, feedback]);
            }
        }
    }

    // -- function body -----------------------------------------------------

    fn generate(&mut self) -> Result<(), CompileError> {
        // prologue: one context per function (uniform chain); the shared
        // ScopeInfo (constant 0) names every slot for dynamic resolution
        let names: Vec<Vec<u8>> = self
            .resolved
            .scope(self.scope)
            .decls
            .iter()
            .map(|d| self.ast.symbol(d.name).to_vec())
            .collect();
        let ctx_info = self.add_constant(Constant::ContextNames(names));
        emit(&mut self.code, Opcode::CreateFunctionContext, &[ctx_info]);
        emit(&mut self.code, Opcode::PushContext, &[self.ctx_save]);
        emit(&mut self.code, Opcode::LoadUndefined, &[]);
        emit(&mut self.code, Opcode::Store, &[self.completion]);

        // parameters arrive at -(i + 2); bind them (context copies included)
        for i in 0..self.param_count {
            emit(&mut self.code, Opcode::Load, &[(-(i as i32 + 2)) as u32]);
            self.store_decl(i);
        }

        self.emit_stmts(self.body)?;

        emit(&mut self.code, Opcode::Load, &[self.completion]);
        emit(&mut self.code, Opcode::PopContext, &[self.ctx_save]);
        emit(&mut self.code, Opcode::Return, &[]);
        Ok(())
    }

    fn emit_stmts(&mut self, node: NodeId) -> Result<(), CompileError> {
        let Node::StmtList { stmts } = *self.ast.node(node) else {
            return self.err(node, "function body");
        };
        let stmts = self.ast.list_items(stmts).to_vec();
        for stmt in stmts {
            self.emit_stmt(stmt)?;
        }
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(stmt) {
            Node::Let { init, .. } => {
                self.expr(init)?;
                let decl = self.next_let;
                self.next_let += 1;
                self.store_decl(decl);
                Ok(())
            }
            Node::ExprStmt { expr } => {
                self.expr(expr)?;
                emit(&mut self.code, Opcode::Store, &[self.completion]);
                Ok(())
            }
            _ => self.err(stmt, "statement"),
        }
    }

    // -- expressions -------------------------------------------------------

    fn expr(&mut self, node: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(node) {
            Node::Number(f) => {
                // (2^53 − 1: largest exactly-representable integer)
                const MAX_EXACT_INT: f64 = 9007199254740991.0;
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
            Node::String(sym) => {
                self.emit_load_constant(Constant::String(self.ast.symbol(sym).to_vec()));
                Ok(())
            }
            Node::Bool(true) => {
                emit(&mut self.code, Opcode::LoadTrue, &[]);
                Ok(())
            }
            Node::Bool(false) => {
                emit(&mut self.code, Opcode::LoadFalse, &[]);
                Ok(())
            }
            Node::Null => {
                emit(&mut self.code, Opcode::LoadNull, &[]);
                Ok(())
            }
            Node::Self_ => {
                // `self` is the frame receiver (param 0), bound at send time
                emit(&mut self.code, Opcode::Load, &[(-1i32) as u32]);
                Ok(())
            }
            Node::Ident(_) => {
                self.load_ident(node);
                Ok(())
            }
            Node::Object { slots } => self.emit_object(slots),
            Node::Array { elements } => self.emit_array(elements),
            Node::Block { .. } => {
                let child = self.block_fids[node.0 as usize].expect("block was collected");
                let idx = self.add_constant(Constant::Callable(IrFunctionId(child)));
                emit(&mut self.code, Opcode::CreateClosure, &[idx]);
                Ok(())
            }
            Node::Get { recv, name } => {
                self.expr(recv)?;
                let obj = self.push_value();
                let name = self.name_constant(name);
                let feedback = self.feedback_slot();
                emit(
                    &mut self.code,
                    Opcode::LoadNamedProperty,
                    &[obj, name, feedback],
                );
                self.pop_value();
                Ok(())
            }
            Node::Index { recv, key } => {
                self.expr(recv)?;
                let obj = self.push_value();
                self.expr(key)?;
                let feedback = self.feedback_slot();
                emit(&mut self.code, Opcode::LoadKeyedProperty, &[obj, feedback]);
                self.pop_value();
                Ok(())
            }
            Node::Send { recv, name, args } => self.emit_send(recv, name, args),
            Node::Call { callee, args } => self.emit_call(callee, args),
            Node::Assign { target, value } => self.emit_assign(target, value),
            Node::Return { value } => {
                self.expr(value)?;
                emit(&mut self.code, Opcode::PopContext, &[self.ctx_save]);
                emit(&mut self.code, Opcode::Return, &[]);
                Ok(())
            }
            Node::Try { body, handler } => self.emit_try(body, handler),
            Node::While { .. } => self.err(node, "while loops"),
            Node::ForIn { .. } => self.err(node, "for-in loops"),
            Node::Match { .. } => self.err(node, "match"),
            Node::If { .. } | Node::Binary { .. } | Node::Unary { .. } => {
                unreachable!("if/operators are desugared to sends before codegen")
            }
            Node::Slot { .. } | Node::MatchArm { .. } => {
                unreachable!("slots and match arms are only reachable through their parents")
            }
            Node::StmtList { .. } | Node::Let { .. } | Node::ExprStmt { .. } => {
                self.err(node, "statement in expression position")
            }
        }
    }

    /// `recv.name(args...)`: the receiver is the first register of the
    /// argument window, the callee sits in a fixed slot above the args.
    fn emit_send(
        &mut self,
        recv: NodeId,
        name: Symbol,
        args: NodeList,
    ) -> Result<(), CompileError> {
        let args: Vec<NodeId> = self.ast.list_items(args).to_vec();
        let argc = args.len() as u32;
        self.expr(recv)?;
        let recv_reg = self.push_value();
        let name = self.name_constant(name);
        let feedback = self.feedback_slot();
        emit(
            &mut self.code,
            Opcode::LoadNamedProperty,
            &[recv_reg, name, feedback],
        );
        let callee_reg = self.reg_base + self.next_temp + argc;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        self.next_temp += argc + 1;
        for (i, &arg) in args.iter().enumerate() {
            self.expr(arg)?;
            emit(&mut self.code, Opcode::Store, &[recv_reg + 1 + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(
            &mut self.code,
            Opcode::CallNoFeedback,
            &[callee_reg, recv_reg, argc + 1],
        );
        self.next_temp -= argc + 1;
        self.pop_value();
        Ok(())
    }

    /// `callee(args...)`: no receiver, so the window starts with undefined
    /// (the JS convention); a method invoked this way sees no `self`.
    fn emit_call(&mut self, callee: NodeId, args: NodeList) -> Result<(), CompileError> {
        let args: Vec<NodeId> = self.ast.list_items(args).to_vec();
        let argc = args.len() as u32;
        self.expr(callee)?;
        let callee_reg = self.reg_base + self.next_temp + 1 + argc;
        emit(&mut self.code, Opcode::Store, &[callee_reg]);
        emit(&mut self.code, Opcode::LoadUndefined, &[]);
        let recv = self.push_value();
        self.next_temp += argc + 1;
        for (i, &arg) in args.iter().enumerate() {
            self.expr(arg)?;
            emit(&mut self.code, Opcode::Store, &[recv + 1 + i as u32]);
        }
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(
            &mut self.code,
            Opcode::CallNoFeedback,
            &[callee_reg, recv, argc + 1],
        );
        self.next_temp -= argc + 1;
        self.pop_value();
        Ok(())
    }

    /// `try { body } catch name { handler }`: call the body closure inside
    /// a handler range. On a throw the frame resumes at the handler with
    /// the exception in the accumulator, which then becomes the handler
    /// closure's first argument. Both arms are ordinary blocks, so
    /// `return` stays a block return.
    fn emit_try(&mut self, body: NodeId, handler: NodeId) -> Result<(), CompileError> {
        let body_fid = self.block_fids[body.0 as usize].expect("try body collected");
        let handler_fid = self.block_fids[handler.0 as usize].expect("catch handler collected");
        let body_idx = self.add_constant(Constant::Callable(IrFunctionId(body_fid)));
        let handler_idx = self.add_constant(Constant::Callable(IrFunctionId(handler_fid)));

        // call the body closure with no receiver and no arguments
        let mark = self.next_temp;
        self.next_temp += 2; // [recv, callee]
        let base = self.reg_base + mark;
        let callee = base + 1;
        emit(&mut self.code, Opcode::CreateClosure, &[body_idx]);
        emit(&mut self.code, Opcode::Store, &[callee]);
        emit(&mut self.code, Opcode::LoadUndefined, &[]);
        emit(&mut self.code, Opcode::Store, &[base]);
        let try_start = self.code.len();
        emit(&mut self.code, Opcode::CallNoFeedback, &[callee, base, 1]);
        let try_end = self.code.len();
        let mut end = Label::new();
        emit_jump(&mut self.code, Opcode::Jump, &mut end);

        // handler entry: the exception arrives in the accumulator; pass it
        // as the catch binding (the handler block's first parameter)
        let handler_pc = self.code.len();
        let exception = self.push_value();
        emit(&mut self.code, Opcode::CreateClosure, &[handler_idx]);
        let hcallee = self.reg_base + self.next_temp + 2;
        emit(&mut self.code, Opcode::Store, &[hcallee]);
        emit(&mut self.code, Opcode::LoadUndefined, &[]);
        let hrecv = self.push_value();
        self.next_temp += 2; // [arg0, callee]
        emit(&mut self.code, Opcode::Load, &[exception]);
        emit(&mut self.code, Opcode::Store, &[hrecv + 1]);
        self.max_temps = self.max_temps.max(self.next_temp);
        emit(&mut self.code, Opcode::CallNoFeedback, &[hcallee, hrecv, 2]);
        self.next_temp = mark;

        self.handlers.push(HandlerEntry {
            try_start,
            try_end,
            handler_pc,
        });
        end.bind(&self.code);
        end.patch_all(&mut self.code);
        Ok(())
    }

    /// `=` never creates where the VM can write through: named slots use
    /// the Self-style `NoShadow` store (holder write, own define on miss),
    /// elements the matching keyed form.
    fn emit_assign(&mut self, target: NodeId, value: NodeId) -> Result<(), CompileError> {
        match *self.ast.node(target) {
            Node::Ident(_) => {
                self.expr(value)?;
                self.store_ident(target);
                Ok(())
            }
            Node::Get { recv, name } => {
                self.expr(recv)?;
                let obj = self.push_value();
                let name = self.name_constant(name);
                self.expr(value)?;
                let feedback = self.feedback_slot();
                emit(
                    &mut self.code,
                    Opcode::StoreNamedPropertyNoShadow,
                    &[obj, name, feedback],
                );
                self.pop_value();
                Ok(())
            }
            Node::Index { recv, key } => {
                self.expr(recv)?;
                let obj = self.push_value();
                self.expr(key)?;
                let key = self.push_value();
                self.expr(value)?;
                emit(&mut self.code, Opcode::StoreKeyedSlot, &[obj, key]);
                self.pop_value();
                self.pop_value();
                Ok(())
            }
            _ => self.err(target, "assignment target"),
        }
    }

    /// An object literal: one store per slot, in source order. Parent
    /// slots only extend the object's prototype list (`AddParent`), named
    /// and element slots go through the ordinary stores.
    fn emit_object(&mut self, slots: NodeList) -> Result<(), CompileError> {
        let slots: Vec<NodeId> = self.ast.list_items(slots).to_vec();
        emit(&mut self.code, Opcode::CreateBareObjectLiteral, &[]);
        let obj = self.push_value();

        for &slot in &slots {
            match *self.ast.node(slot) {
                Node::Slot {
                    kind: SlotKind::Parent,
                    key,
                    value,
                } => {
                    // `name*: value` appends to the prototype's parent pair
                    // list; the label is the parent's own slot name
                    let Node::Ident(sym) = *self.ast.node(key) else {
                        return self.err(key, "parent slot name");
                    };
                    let name = self.name_constant(sym);
                    self.expr(value)?;
                    emit(&mut self.code, Opcode::AddParent, &[obj, name]);
                }
                Node::Slot {
                    kind: SlotKind::Named,
                    key,
                    value,
                } => {
                    let Node::Ident(sym) = *self.ast.node(key) else {
                        return self.err(key, "named slot key");
                    };
                    let name = self.name_constant(sym);
                    self.expr(value)?;
                    let feedback = self.feedback_slot();
                    emit(
                        &mut self.code,
                        Opcode::StoreNamedProperty,
                        &[obj, name, feedback],
                    );
                }
                Node::Slot {
                    kind: SlotKind::Element,
                    key,
                    value,
                } => {
                    self.expr(key)?;
                    let key = self.push_value();
                    self.expr(value)?;
                    let feedback = self.feedback_slot();
                    emit(
                        &mut self.code,
                        Opcode::StoreKeyedProperty,
                        &[obj, key, feedback],
                    );
                    self.pop_value();
                }
                _ => unreachable!("object slots are Slot nodes"),
            }
        }

        emit(&mut self.code, Opcode::Load, &[obj]);
        self.pop_value();
        Ok(())
    }

    fn emit_array(&mut self, elements: NodeList) -> Result<(), CompileError> {
        let elements: Vec<NodeId> = self.ast.list_items(elements).to_vec();
        emit(&mut self.code, Opcode::CreateEmptyArrayLiteral, &[]);
        let array = self.push_value();
        for (i, &element) in elements.iter().enumerate() {
            emit(&mut self.code, Opcode::LoadSmi, &[i as u32]);
            let index = self.push_value();
            self.expr(element)?;
            // literal elements are explicit layout: the store may grow
            let feedback = self.feedback_slot();
            emit(
                &mut self.code,
                Opcode::StoreKeyedProperty,
                &[array, index, feedback],
            );
            self.pop_value();
        }
        emit(&mut self.code, Opcode::Load, &[array]);
        self.pop_value();
        Ok(())
    }
}
