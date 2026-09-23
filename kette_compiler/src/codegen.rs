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

use bytecode::{
    CallableKind, Constant, ConstIdx, FnBuilder, FunctionId, FunctionMeta, Program, Reg, RegList,
};
use kette_parser::{Ast, Node, NodeId, NodeList, Resolution, Resolved, ScopeId, SlotKind, Symbol};

use crate::CompileError;

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
    debug_assert!(bytecode::validate(&program).is_ok());
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
    reg_of: Vec<Option<Reg>>,
    param_count: usize,
    /// `node -> function index` for closure creation
    block_fids: &'a [Option<u32>],
    b: FnBuilder,
    /// register holding the pushed-over context (prologue/epilogue)
    ctx_save: Reg,
    /// block/script result register (last expression statement)
    completion: Reg,
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
                reg_of[decl] = Some(Reg::new(local_count as i32));
                local_count += 1;
            }
        }
        // frame layout: locals, then the context-save and completion
        // registers, then the temp stack
        let ctx_save = Reg::new(local_count as i32);
        let completion = Reg::new(local_count as i32 + 1);
        let mut b = FnBuilder::new(param_count as u32);
        b.set_temp_base(local_count + 2);
        Self {
            ast,
            resolved,
            scope,
            body,
            captured,
            reg_of,
            param_count,
            block_fids,
            b,
            ctx_save,
            completion,
            next_let: param_count,
        }
    }

    fn finish(self) -> bytecode::Function {
        // build failures (unbalanced temps, unbound labels) are compiler
        // bugs here: every emit path pairs its temp marks and binds its
        // labels by construction
        self.b
            .finish(FunctionMeta {
                name: None,
                kind: CallableKind::Method,
                length: self.param_count as u32,
                strict: true,
            })
            .expect("kette function builder invariants")
    }

    fn err<T>(&self, node: NodeId, feature: &'static str) -> Result<T, CompileError> {
        Err(CompileError::new(self.ast.span(node), feature))
    }

    // -- constants -----------------------------------------------------------

    fn name_constant(&mut self, name: Symbol) -> ConstIdx {
        self.b.name(self.ast.symbol(name))
    }

    // -- locals ------------------------------------------------------------

    fn load_decl(&mut self, decl: usize) {
        if self.captured[decl] {
            self.b.load_context_slot(decl as u32, 0);
        } else {
            let reg = self.reg_of[decl].expect("uncaptured decl has a register");
            self.b.load(reg);
        }
    }

    fn store_decl(&mut self, decl: usize) {
        if self.captured[decl] {
            self.b.store_context_slot(decl as u32, 0);
        } else {
            let reg = self.reg_of[decl].expect("uncaptured decl has a register");
            self.b.store(reg);
        }
    }

    fn load_ident(&mut self, node: NodeId) {
        match self.resolved.resolution(node) {
            Some(Resolution::Local { decl, .. }) => self.load_decl(decl as usize),
            Some(Resolution::Capture { decl, depth, .. }) => {
                self.b.load_context_slot(decl, depth);
            }
            Some(Resolution::Global) | None => {
                let Node::Ident(sym) = *self.ast.node(node) else {
                    unreachable!("identifier nodes carry a symbol");
                };
                let idx = self.name_constant(sym);
                let feedback = self.b.new_feedback();
                self.b.load_global(idx, feedback);
            }
        }
    }

    fn store_ident(&mut self, node: NodeId) {
        match self.resolved.resolution(node) {
            Some(Resolution::Local { decl, .. }) => self.store_decl(decl as usize),
            Some(Resolution::Capture { decl, depth, .. }) => {
                self.b.store_context_slot(decl, depth);
            }
            Some(Resolution::Global) | None => {
                let Node::Ident(sym) = *self.ast.node(node) else {
                    unreachable!("identifier nodes carry a symbol");
                };
                let idx = self.name_constant(sym);
                let feedback = self.b.new_feedback();
                self.b.store_global(idx, feedback);
            }
        }
    }

    // -- function body -----------------------------------------------------

    fn generate(&mut self) -> Result<(), CompileError> {
        // prologue: one context per function (uniform chain); the shared
        // ScopeInfo (constant 0) names every slot for dynamic resolution
        let names: Vec<Box<[u8]>> = self
            .resolved
            .scope(self.scope)
            .decls
            .iter()
            .map(|d| self.ast.symbol(d.name).into())
            .collect();
        let ctx_info = self.b.constant(Constant::ContextNames(names));
        self.b.create_function_context(ctx_info);
        self.b.push_context(self.ctx_save);
        self.b.load_undefined();
        self.b.store(self.completion);

        // parameters arrive at -(i + 2); bind them (context copies included)
        for i in 0..self.param_count {
            let param = self.b.param(i as u32);
            self.b.load(param);
            self.store_decl(i);
        }

        self.emit_stmts(self.body)?;

        self.b.load(self.completion);
        self.b.pop_context(self.ctx_save);
        self.b.ret();
        Ok(())
    }

    fn emit_stmts(&mut self, node: NodeId) -> Result<(), CompileError> {
        let Node::StmtList { stmts } = *self.ast.node(node) else {
            return self.err(node, "function body");
        };
        let stmts = self.ast.list_items(stmts).to_vec();
        for stmt in stmts {
            // a terminating expression (`return` is kette's expression
            // statement) makes the remaining statements unreachable
            if !self.b.is_live() {
                break;
            }
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
                // `return x` terminates mid-statement: the completion
                // store would read a dead accumulator
                if self.b.is_live() {
                    self.b.store(self.completion);
                }
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
            Node::String(sym) => {
                let c = self.b.constant(Constant::String(self.ast.symbol(sym).into()));
                self.b.load_constant(c);
                Ok(())
            }
            Node::Bool(true) => {
                self.b.load_true();
                Ok(())
            }
            Node::Bool(false) => {
                self.b.load_false();
                Ok(())
            }
            Node::Null => {
                self.b.load_null();
                Ok(())
            }
            Node::Self_ => {
                // `self` is the frame receiver, bound at send time
                self.b.load(self.b.this_reg());
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
                let idx = self.b.constant(Constant::Callable(FunctionId(child)));
                self.b.create_closure(idx);
                Ok(())
            }
            Node::Get { recv, name } => {
                self.expr(recv)?;
                let obj = self.b.stage_acc();
                let name = self.name_constant(name);
                let feedback = self.b.new_feedback();
                self.b.load_named_property(obj, name, feedback);
                self.b.drop_temp();
                Ok(())
            }
            Node::Index { recv, key } => {
                self.expr(recv)?;
                let obj = self.b.stage_acc();
                self.expr(key)?;
                let feedback = self.b.new_feedback();
                self.b.load_keyed_property(obj, feedback);
                self.b.drop_temp();
                Ok(())
            }
            Node::Send { recv, name, args } => self.emit_send(recv, name, args),
            Node::Call { callee, args } => self.emit_call(callee, args),
            Node::Assign { target, value } => self.emit_assign(target, value),
            Node::Return { value } => {
                self.expr(value)?;
                self.b.pop_context(self.ctx_save);
                self.b.ret();
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
        let mark = self.b.temp_depth();
        self.expr(recv)?;
        let recv_reg = self.b.stage_acc();
        let name = self.name_constant(name);
        let feedback = self.b.new_feedback();
        self.b.load_named_property(recv_reg, name, feedback);
        // argument window [recv, args...] plus the callee slot above it;
        // the method rides the accumulator straight into its slot before
        // argument evaluation clobbers it
        let args_base = self.b.reserve_temps(argc + 1);
        let callee_reg = Reg::new(args_base.index() + argc as i32);
        self.b.store(callee_reg);
        for (i, &arg) in args.iter().enumerate() {
            self.expr(arg)?;
            self.b.store(Reg::new(args_base.index() + i as i32));
        }
        self.b
            .call_no_feedback(callee_reg, RegList::new(recv_reg, argc + 1));
        self.b.drop_temps(mark);
        Ok(())
    }

    /// `callee(args...)`: no receiver, so the window starts with undefined
    /// (the JS convention); a method invoked this way sees no `self`.
    fn emit_call(&mut self, callee: NodeId, args: NodeList) -> Result<(), CompileError> {
        let args: Vec<NodeId> = self.ast.list_items(args).to_vec();
        let argc = args.len() as u32;
        let mark = self.b.temp_depth();
        self.expr(callee)?;
        // window [recv, args...] plus the callee slot above it
        let args_base = self.b.reserve_temps(argc + 2);
        let callee_reg = Reg::new(args_base.index() + argc as i32 + 1);
        self.b.store(callee_reg);
        self.b.load_undefined();
        self.b.store(args_base);
        for (i, &arg) in args.iter().enumerate() {
            self.expr(arg)?;
            self.b.store(Reg::new(args_base.index() + 1 + i as i32));
        }
        self.b
            .call_no_feedback(callee_reg, RegList::new(args_base, argc + 1));
        self.b.drop_temps(mark);
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
        let body_idx = self.b.constant(Constant::Callable(FunctionId(body_fid)));
        let handler_idx = self.b.constant(Constant::Callable(FunctionId(handler_fid)));

        // call the body closure with no receiver and no arguments
        let mark = self.b.temp_depth();
        let base = self.b.reserve_temps(2); // [recv, callee]
        let callee = Reg::new(base.index() + 1);
        self.b.create_closure(body_idx);
        self.b.store(callee);
        self.b.load_undefined();
        self.b.store(base);
        let t = self.b.begin_try();
        self.b.call_no_feedback(callee, RegList::new(base, 1));
        self.b.end_try(t);

        let end = self.b.new_label();
        self.b.jump(end);

        // handler entry: the exception arrives in the accumulator; pass it
        // as the catch binding (the handler block's first parameter)
        self.b.handler_entry(t);
        let exception = self.b.stage_acc();
        self.b.create_closure(handler_idx);
        let hwindow = self.b.reserve_temps(3); // [recv, arg0, callee]
        let hcallee = Reg::new(hwindow.index() + 2);
        self.b.store(hcallee);
        self.b.load_undefined();
        self.b.store(hwindow);
        self.b.load(exception);
        self.b.store(Reg::new(hwindow.index() + 1));
        self.b.call_no_feedback(hcallee, RegList::new(hwindow, 2));
        self.b.drop_temps(mark);

        self.b.bind(end);
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
                let obj = self.b.stage_acc();
                let name = self.name_constant(name);
                self.expr(value)?;
                let feedback = self.b.new_feedback();
                self.b.store_named_property_no_shadow(obj, name, feedback);
                self.b.drop_temp();
                Ok(())
            }
            Node::Index { recv, key } => {
                self.expr(recv)?;
                let obj = self.b.stage_acc();
                self.expr(key)?;
                let key = self.b.stage_acc();
                self.expr(value)?;
                self.b.store_keyed_slot(obj, key);
                self.b.drop_temp();
                self.b.drop_temp();
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
        self.b.create_bare_object_literal();
        let obj = self.b.stage_acc();

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
                    self.b.add_parent(obj, name);
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
                    let feedback = self.b.new_feedback();
                    self.b.store_named_property(obj, name, feedback);
                }
                Node::Slot {
                    kind: SlotKind::Element,
                    key,
                    value,
                } => {
                    self.expr(key)?;
                    let key = self.b.stage_acc();
                    self.expr(value)?;
                    let feedback = self.b.new_feedback();
                    self.b.store_keyed_property(obj, key, feedback);
                    self.b.drop_temp();
                }
                _ => unreachable!("object slots are Slot nodes"),
            }
        }

        self.b.load(obj);
        self.b.drop_temp();
        Ok(())
    }

    fn emit_array(&mut self, elements: NodeList) -> Result<(), CompileError> {
        let elements: Vec<NodeId> = self.ast.list_items(elements).to_vec();
        self.b.create_empty_array_literal();
        let array = self.b.stage_acc();
        for (i, &element) in elements.iter().enumerate() {
            self.b.load_smi(i as i32);
            let index = self.b.stage_acc();
            self.expr(element)?;
            // literal elements are explicit layout: the store may grow
            let feedback = self.b.new_feedback();
            self.b.store_keyed_property(array, index, feedback);
            self.b.drop_temp();
        }
        self.b.load(array);
        self.b.drop_temp();
        Ok(())
    }
}
