//! Pre-resolve lowering: Kette control flow and operators are sugar over
//! sends, with the semantics living in the receiver.
//!
//!   if c { A } else { B }  →  c.ifElse({ A }, { B })
//!   if c { A }             →  c.if({ A })
//!   a OP b                 →  a.<op>(b)          (eager operators)
//!   a && b                 →  a.and({ b })       (lazy)
//!   a || b                 →  a.or({ b })        (lazy)
//!   -a / !a                →  a.neg() / a.not()
//!
//! The rewrite runs in place before resolution: the wrapped RHS of
//! `&&`/`||` and an `else if` chain introduce real blocks, and therefore
//! scopes the resolver has to see (a codegen-time wrapper would shift
//! every capture depth inside it by one).

use kette_parser::{Ast, BinaryOp, Node, NodeId, SlotKind, UnaryOp};

pub fn desugar(ast: &mut Ast) {
    if let Some(root) = ast.root() {
        Desugar { ast }.rewrite(root);
    }
}

struct Desugar<'a> {
    ast: &'a mut Ast,
}

fn binary_selector(op: BinaryOp) -> &'static [u8] {
    match op {
        BinaryOp::Or => b"or",
        BinaryOp::And => b"and",
        BinaryOp::Eq => b"eq",
        BinaryOp::Ne => b"neq",
        BinaryOp::Lt => b"lt",
        BinaryOp::Le => b"lte",
        BinaryOp::Gt => b"gt",
        BinaryOp::Ge => b"gte",
        BinaryOp::Add => b"add",
        BinaryOp::Sub => b"sub",
        BinaryOp::Mul => b"mul",
        BinaryOp::Div => b"div",
        BinaryOp::Mod => b"mod",
    }
}

fn unary_selector(op: UnaryOp) -> &'static [u8] {
    match op {
        UnaryOp::Neg => b"neg",
        UnaryOp::Not => b"not",
    }
}

impl Desugar<'_> {
    fn list(&self, list: kette_parser::NodeList) -> Vec<NodeId> {
        self.ast.list_items(list).to_vec()
    }

    /// Wrap `expr` in a fresh zero-arg block (`{ expr }`), owned by a new
    /// scope once resolved.
    fn wrap_block(&mut self, expr: NodeId) -> NodeId {
        let span = self.ast.span(expr);
        let stmt = self.ast.add(Node::ExprStmt { expr }, span);
        let stmts = self.ast.list(&[stmt]);
        let body = self.ast.add(Node::StmtList { stmts }, span);
        let params = self.ast.list(&[]);
        self.ast.add(Node::Block { params, body }, span)
    }

    fn rewrite(&mut self, node: NodeId) {
        match self.ast.node(node).clone() {
            Node::If { cond, then, else_ } => {
                self.rewrite(cond);
                self.rewrite(then);
                let else_ = else_.map(|e| {
                    if matches!(self.ast.node(e), Node::Block { .. }) {
                        self.rewrite(e);
                        e
                    } else {
                        let wrapped = self.wrap_block(e);
                        self.rewrite(wrapped);
                        wrapped
                    }
                });
                let (name, args) = match else_ {
                    Some(e) => (b"ifElse".to_vec(), self.ast.list(&[then, e])),
                    None => (b"if".to_vec(), self.ast.list(&[then])),
                };
                let name = self.ast.intern(&name);
                self.ast.replace(
                    node,
                    Node::Send {
                        recv: cond,
                        name,
                        args,
                    },
                );
            }
            Node::Binary { op, lhs, rhs } => {
                self.rewrite(lhs);
                let rhs = if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    let wrapped = self.wrap_block(rhs);
                    self.rewrite(wrapped);
                    wrapped
                } else {
                    self.rewrite(rhs);
                    rhs
                };
                let name = self.ast.intern(binary_selector(op));
                let args = self.ast.list(&[rhs]);
                self.ast.replace(
                    node,
                    Node::Send {
                        recv: lhs,
                        name,
                        args,
                    },
                );
            }
            Node::Unary { op, expr } => {
                self.rewrite(expr);
                let name = self.ast.intern(unary_selector(op));
                let args = self.ast.list(&[]);
                self.ast.replace(
                    node,
                    Node::Send {
                        recv: expr,
                        name,
                        args,
                    },
                );
            }
            Node::StmtList { stmts } => {
                for stmt in self.list(stmts) {
                    self.rewrite(stmt);
                }
            }
            Node::Let { init, .. } => self.rewrite(init),
            Node::ExprStmt { expr } => self.rewrite(expr),
            Node::Block { body, .. } => self.rewrite(body),
            Node::ForIn { iter, body, .. } => {
                self.rewrite(iter);
                self.rewrite(body);
            }
            Node::While { cond, body } => {
                self.rewrite(cond);
                self.rewrite(body);
            }
            Node::Match { scrut, arms } => {
                self.rewrite(scrut);
                for arm in self.list(arms) {
                    if let Node::MatchArm { target, handler } = self.ast.node(arm).clone() {
                        if let Some(target) = target {
                            self.rewrite(target);
                        }
                        self.rewrite(handler);
                    }
                }
            }
            Node::Object { slots } => {
                for slot in self.list(slots) {
                    if let Node::Slot { kind, key, value } = self.ast.node(slot).clone() {
                        if kind == SlotKind::Element {
                            self.rewrite(key);
                        }
                        self.rewrite(value);
                    }
                }
            }
            Node::Array { elements } => {
                for element in self.list(elements) {
                    self.rewrite(element);
                }
            }
            Node::Get { recv, .. } => self.rewrite(recv),
            Node::Index { recv, key } => {
                self.rewrite(recv);
                self.rewrite(key);
            }
            Node::Send { recv, args, .. } => {
                self.rewrite(recv);
                for arg in self.list(args) {
                    self.rewrite(arg);
                }
            }
            Node::Call { callee, args } => {
                self.rewrite(callee);
                for arg in self.list(args) {
                    self.rewrite(arg);
                }
            }
            Node::Assign { target, value } => {
                self.rewrite(target);
                self.rewrite(value);
            }
            Node::Return { value } => self.rewrite(value),
            Node::Try { body, handler } => {
                self.rewrite(body);
                self.rewrite(handler);
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
}
