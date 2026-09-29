//! The Kette AST walker.
//!
//! Every `{ slots | params | body }` literal with a params marker or a
//! non-empty body is a function AND an object: the closure is created
//! first, then the slots are stored onto it. Slot values evaluate in the
//! enclosing scope at creation time. The script body is function 0.
//!
//! Locals that no nested object captures live in frame registers;
//! captured ones live in the function's context (parameters are copied
//! into it in the prologue). `if` branches compile inline in the
//! enclosing function, so their `let`s share its scope. `f(args)` binds
//! the callee itself as `self` — the call is the `value` send.

use bytecode::{
    CallableKind, ConstIdx, Constant, FnBuilder, FunctionId, FunctionMeta, Program, Reg, RegList,
};
use kette_parser::{
    Ast, BinaryOp, Node, NodeId, NodeList, ObjectParts, Resolution, Resolved, ScopeId, SlotKind,
    Symbol, UnaryOp,
};

use crate::CompileError;

/// `node -> function index` for every callable object (script is 0).
fn collect_objects(ast: &Ast, node: NodeId, out: &mut Vec<NodeId>) {
    let walk_stmts = |ast: &Ast, stmts: NodeList, out: &mut Vec<NodeId>| {
        for &stmt in ast.list_items(stmts) {
            collect_objects(ast, stmt, out);
        }
    };
    match ast.node(node) {
        Node::Object(parts) => {
            if ast.is_closure(node) {
                out.push(node);
            }
            for &slot in ast.list_items(parts.slots) {
                if let Node::Slot { kind, key, value } = ast.node(slot) {
                    if *kind == SlotKind::Element {
                        collect_objects(ast, *key, out);
                    }
                    collect_objects(ast, *value, out);
                }
            }
            if let Node::StmtList { stmts } = ast.node(parts.body) {
                walk_stmts(ast, *stmts, out);
            }
        }
        Node::StmtList { stmts } => walk_stmts(ast, *stmts, out),
        Node::Let { init, .. } => collect_objects(ast, *init, out),
        Node::ExprStmt { expr } => collect_objects(ast, *expr, out),
        Node::If { cond, then, else_ } => {
            collect_objects(ast, *cond, out);
            collect_branch(ast, *then, out);
            if let Some(else_) = else_ {
                collect_branch(ast, *else_, out);
            }
        }
        Node::Array { elements } => {
            for &element in ast.list_items(*elements) {
                collect_objects(ast, element, out);
            }
        }
        Node::Get { recv, .. } => collect_objects(ast, *recv, out),
        Node::Index { recv, key } => {
            collect_objects(ast, *recv, out);
            collect_objects(ast, *key, out);
        }
        Node::Send { recv, args, .. } => {
            collect_objects(ast, *recv, out);
            for &arg in ast.list_items(*args) {
                collect_objects(ast, arg, out);
            }
        }
        Node::Call { callee, args } => {
            collect_objects(ast, *callee, out);
            for &arg in ast.list_items(*args) {
                collect_objects(ast, arg, out);
            }
        }
        Node::Assign { target, value } => {
            collect_objects(ast, *target, out);
            collect_objects(ast, *value, out);
        }
        Node::Unary { expr, .. } => collect_objects(ast, *expr, out),
        Node::Binary { lhs, rhs, .. } => {
            collect_objects(ast, *lhs, out);
            collect_objects(ast, *rhs, out);
        }
        Node::Return { value } => collect_objects(ast, *value, out),
        Node::Slot { .. }
        | Node::Number(_)
        | Node::String(_)
        | Node::Bool(_)
        | Node::Null
        | Node::Self_
        | Node::Ident(_) => {}
    }
}

/// An `if` branch: statements inline, or a chained `else if`.
fn collect_branch(ast: &Ast, branch: NodeId, out: &mut Vec<NodeId>) {
    match ast.node(branch) {
        Node::StmtList { stmts } => {
            for &stmt in ast.list_items(*stmts) {
                collect_objects(ast, stmt, out);
            }
        }
        Node::If { .. } => collect_objects(ast, branch, out),
        _ => {}
    }
}

pub fn generate(ast: &Ast, resolved: &Resolved) -> Result<Program, CompileError> {
    let root = ast.root().expect("the parser sets a root");
    let mut objects = Vec::new();
    collect_objects(ast, root, &mut objects);

    // Function ids are assigned in `objects` order (script, then objects
    // in pre-order), matching the `add_function` calls below.
    let mut object_fids = vec![None; ast.node_count()];
    for (i, &object) in objects.iter().enumerate() {
        object_fids[object.0 as usize] = Some(i as u32 + 1);
    }

    let mut program = Program::with_capacity(objects.len() + 1);
    {
        let scope = resolved.scope_of(root).expect("root owns the script scope");
        let mut generator = FunctionGen::new(ast, resolved, scope, root, 0, &object_fids);
        generator.generate()?;
        program.add_function(generator.finish());
    }
    for &object in &objects {
        let Node::Object(parts) = ast.node(object) else {
            unreachable!("collected nodes are objects");
        };
        let scope = resolved
            .scope_of(object)
            .expect("callable object owns a scope");
        let param_count = parts
            .params
            .map(|params| ast.list_items(params).len())
            .unwrap_or(0);
        let mut generator =
            FunctionGen::new(ast, resolved, scope, parts.body, param_count, &object_fids);
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
    /// per scope decl: context-allocated because a nested object captures it
    captured: Vec<bool>,
    /// per scope decl: frame register, when not captured
    reg_of: Vec<Option<Reg>>,
    param_count: usize,
    /// `node -> function index` for closure creation
    object_fids: &'a [Option<u32>],
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
        object_fids: &'a [Option<u32>],
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
            object_fids,
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
                self.b.load_global_fast(idx, feedback);
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
                self.b.store_global_fast(idx);
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

        self.emit_body(self.body, self.completion)?;

        self.b.load(self.completion);
        self.b.pop_context(self.ctx_save);
        self.b.ret();
        Ok(())
    }

    /// Compile a statement list, storing the last expression's value into
    /// `sink` (the completion register, or an `if` result temporary).
    fn emit_body(&mut self, node: NodeId, sink: Reg) -> Result<(), CompileError> {
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
            self.emit_stmt(stmt, sink)?;
        }
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: NodeId, sink: Reg) -> Result<(), CompileError> {
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
                // `return x` terminates mid-statement: the sink store
                // would read a dead accumulator
                if self.b.is_live() {
                    self.b.store(sink);
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
                let c = self
                    .b
                    .constant(Constant::String(self.ast.symbol(sym).into()));
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
                // `self` is the frame receiver, bound at send/call time
                self.b.load(self.b.this_reg());
                Ok(())
            }
            Node::Ident(_) => {
                self.load_ident(node);
                Ok(())
            }
            Node::Object(_) => self.emit_object(node),
            Node::Array { elements } => self.emit_array(elements),
            Node::Get { recv, name } => {
                self.expr(recv)?;
                let obj = self.b.stage_acc();
                let name = self.name_constant(name);
                let feedback = self.b.new_feedback();
                self.b.load_named_property_fast(obj, name, feedback);
                self.b.drop_temp();
                Ok(())
            }
            Node::Index { recv, key } => {
                self.expr(recv)?;
                let obj = self.b.stage_acc();
                self.expr(key)?;
                self.b.load_keyed_property_fast(obj);
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
            Node::If { .. } => {
                let result = self.b.temp();
                self.emit_if(node, result)?;
                self.b.load(result);
                self.b.drop_temp();
                Ok(())
            }
            Node::Binary { op, lhs, rhs } => self.emit_binary(op, lhs, rhs),
            Node::Unary { op, expr } => self.emit_unary(op, expr),
            Node::Slot { .. } => unreachable!("slots are only reachable through their object"),
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
        self.b.load_named_property_fast(recv_reg, name, feedback);
        // argument window [args_base .. args_base+argc]: element 0 (the
        // receiver) rides the top slot, argument i sits argc-1-i in; the
        // callee rides above
        let args_base = self.b.reserve_temps(argc + 2);
        let top = args_base.index() + argc as i32;
        let callee_reg = Reg::new(top + 1);
        self.b.store(callee_reg);
        for (i, &arg) in args.iter().enumerate() {
            self.expr(arg)?;
            self.b.store(Reg::new(top - 1 - i as i32));
        }
        self.b.move_reg(Reg::new(top), recv_reg);
        self.b
            .call_no_feedback(callee_reg, RegList::new(args_base, argc + 1));
        self.b.drop_temps(mark);
        Ok(())
    }

    /// `callee(args...)`: the callee itself is the receiver (`f(x)` is
    /// the `value` send), so `self` inside the body is the callee object.
    fn emit_call(&mut self, callee: NodeId, args: NodeList) -> Result<(), CompileError> {
        let args: Vec<NodeId> = self.ast.list_items(args).to_vec();
        let argc = args.len() as u32;
        let mark = self.b.temp_depth();
        self.expr(callee)?;
        // window [recv, args...]: element 0 rides the top slot, argument
        // i sits argc-1-i in; the callee rides above and is its own recv
        let args_base = self.b.reserve_temps(argc + 2);
        let top = args_base.index() + argc as i32;
        let callee_reg = Reg::new(top + 1);
        self.b.store(callee_reg);
        self.b.move_reg(Reg::new(top), callee_reg);
        for (i, &arg) in args.iter().enumerate() {
            self.expr(arg)?;
            self.b.store(Reg::new(top - 1 - i as i32));
        }
        self.b
            .call_no_feedback(callee_reg, RegList::new(args_base, argc + 1));
        self.b.drop_temps(mark);
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
                self.b
                    .store_named_property_no_shadow_fast(obj, name, feedback);
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

    /// A `{ slots | params | body }` literal: closure or fresh object
    /// first, then one store per slot in source order. Parent slots only
    /// extend the object's prototype list (`AddParent`); named and element
    /// slots go through the ordinary stores. Element slots need an
    /// array-kind receiver (elements only exist there), so a literal with
    /// any element slot is created as an array.
    fn emit_object(&mut self, node: NodeId) -> Result<(), CompileError> {
        let Node::Object(parts) = self.ast.node(node) else {
            unreachable!("emit_object called on an Object node");
        };
        let parts: ObjectParts = parts.clone();
        let has_elements = self.ast.list_items(parts.slots).iter().any(|&slot| {
            matches!(
                self.ast.node(slot),
                Node::Slot {
                    kind: SlotKind::Element,
                    ..
                }
            )
        });
        if self.ast.is_closure(node) {
            let child = self.object_fids[node.0 as usize].expect("callable object was collected");
            let idx = self.b.constant(Constant::Callable(FunctionId(child)));
            self.b.create_closure(idx);
        } else if has_elements {
            self.b.create_empty_array_literal();
        } else {
            self.b.create_bare_object_literal();
        }
        let obj = self.b.stage_acc();
        self.emit_slots(&parts, obj)?;
        self.b.load(obj);
        self.b.drop_temp();
        Ok(())
    }

    fn emit_slots(&mut self, parts: &ObjectParts, obj: Reg) -> Result<(), CompileError> {
        for &slot in self.ast.list_items(parts.slots) {
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
                    // literal slots are explicit layout: the defining
                    // store (with transitions); the receiver is a fresh
                    // bare object
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
                    self.b.store_keyed_property_fast(obj, key);
                    self.b.drop_temp();
                }
                _ => unreachable!("object slots are Slot nodes"),
            }
        }
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
            self.b.store_keyed_property_fast(array, index);
            self.b.drop_temp();
        }
        self.b.load(array);
        self.b.drop_temp();
        Ok(())
    }

    // -- operators and control flow -----------------------------------------

    fn emit_binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> Result<(), CompileError> {
        match op {
            BinaryOp::And | BinaryOp::Or => {
                // short-circuit: the lhs rides a temp; the rhs replaces it
                // only when the lhs did not decide
                self.expr(lhs)?;
                let lhs_reg = self.b.stage_acc();
                let end = self.b.new_label();
                if op == BinaryOp::And {
                    self.b.jump_if_falsy(end);
                } else {
                    self.b.jump_if_truthy(end);
                }
                self.expr(rhs)?;
                self.b.store(lhs_reg);
                self.b.bind(end);
                self.b.load(lhs_reg);
                self.b.drop_temp();
                Ok(())
            }
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Gt
            | BinaryOp::Le
            | BinaryOp::Ge
            | BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Mod => {
                // evaluate in source order into two temps, then orient:
                // comparisons compute `acc OP reg`, arithmetic `reg OP acc`
                self.expr(lhs)?;
                let lhs_reg = self.b.stage_acc();
                self.expr(rhs)?;
                let rhs_reg = self.b.stage_acc();
                match op {
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Gt
                    | BinaryOp::Le
                    | BinaryOp::Ge => {
                        self.b.load(lhs_reg);
                        match op {
                            BinaryOp::Eq => self.b.equal(rhs_reg),
                            BinaryOp::Ne => {
                                self.b.equal(rhs_reg);
                                self.invert_acc();
                            }
                            BinaryOp::Lt => self.b.less_than(rhs_reg),
                            BinaryOp::Gt => self.b.greater_than(rhs_reg),
                            BinaryOp::Le => self.b.less_than_or_equal(rhs_reg),
                            BinaryOp::Ge => self.b.greater_than_or_equal(rhs_reg),
                            _ => unreachable!("not a comparison"),
                        }
                    }
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Mod => {
                        self.b.load(rhs_reg);
                        match op {
                            BinaryOp::Add => self.b.add(lhs_reg),
                            BinaryOp::Sub => self.b.sub(lhs_reg),
                            BinaryOp::Mul => self.b.mul(lhs_reg),
                            BinaryOp::Div => self.b.div(lhs_reg),
                            BinaryOp::Mod => self.b.mod_(lhs_reg),
                            _ => unreachable!("not arithmetic"),
                        }
                    }
                    BinaryOp::And | BinaryOp::Or => unreachable!("handled above"),
                }
                self.b.drop_temp();
                self.b.drop_temp();
                Ok(())
            }
        }
    }

    fn emit_unary(&mut self, op: UnaryOp, expr: NodeId) -> Result<(), CompileError> {
        self.expr(expr)?;
        match op {
            UnaryOp::Neg => self.b.negate(),
            UnaryOp::Not => self.invert_acc(),
        }
        Ok(())
    }

    /// `acc = !acc` via jumps (no dedicated opcode): true/false
    /// singletons on either path.
    fn invert_acc(&mut self) {
        let truthy = self.b.new_label();
        let end = self.b.new_label();
        self.b.jump_if_falsy(truthy);
        self.b.load_false();
        self.b.jump(end);
        self.b.bind(truthy);
        self.b.load_true();
        self.b.bind(end);
    }

    /// `if cond { A } else { B }` inline; the value of the taken branch
    /// (its last expression) lands in `sink`. Without an else branch the
    /// falsy path yields `null`.
    fn emit_if(&mut self, node: NodeId, sink: Reg) -> Result<(), CompileError> {
        let Node::If { cond, then, else_ } = *self.ast.node(node) else {
            unreachable!("emit_if called on an If node");
        };
        match else_ {
            Some(else_branch) => {
                self.expr(cond)?;
                let else_label = self.b.new_label();
                let end = self.b.new_label();
                self.b.jump_if_falsy(else_label);
                self.emit_branch(then, sink)?;
                self.b.jump(end);
                self.b.bind(else_label);
                self.emit_branch(else_branch, sink)?;
                self.b.bind(end);
                Ok(())
            }
            None => {
                let end = self.b.new_label();
                // the falsy path yields null: seed the sink BEFORE the
                // condition, so the accumulator still holds the condition
                // for the jump
                self.b.load_null();
                self.b.store(sink);
                self.expr(cond)?;
                self.b.jump_if_falsy(end);
                self.emit_branch(then, sink)?;
                self.b.bind(end);
                Ok(())
            }
        }
    }

    /// An `if` branch: a statement list storing into `sink`, or a
    /// chained `else if`.
    fn emit_branch(&mut self, branch: NodeId, sink: Reg) -> Result<(), CompileError> {
        match *self.ast.node(branch) {
            Node::StmtList { .. } => self.emit_body(branch, sink),
            Node::If { .. } => self.emit_if(branch, sink),
            _ => self.err(branch, "if branch"),
        }
    }
}
