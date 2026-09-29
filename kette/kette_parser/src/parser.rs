//! Hand-written recursive descent over the token stream — no lookahead
//! beyond the current token.
//!
//! The one interesting production is the brace literal:
//!
//! ```text
//! object := '{' entry* '}'
//! entry  := slot | statement
//! slot   := name ':' expr          (named)
//!         | name '*' ':' expr      (parent)
//!         | '[' expr ']' ':' expr  (element)
//! ```
//!
//! Each entry is parsed as a statement first; a `:` directly after it
//! reclassifies it as a slot (a `[key]` expression unwraps to the key).
//! The single exception is `name * :` — there the `*` is consumed before
//! the `:` is visible, so a name (and possibly the star) is consumed
//! speculatively and ordinary expression parsing resumes when no `:`
//! follows. Slots must precede `|params|`, which must precede the body.

use crate::ast::{Ast, BinaryOp, Node, NodeId, ObjectParts, SlotKind, UnaryOp};
use crate::plumbing::ParseError;
use crate::token::{ByteSpan, Token, TokenKind, TokenValue};
use crate::{CharStream, Scanner, Symbol, SymbolTable};

fn is_name_kind(kind: TokenKind) -> bool {
    kind == TokenKind::Identifier || kind.is_keyword()
}

/// `*` binds tighter than every other binary operator (its precedence
/// in `TOKEN_INFO`).
const MUL_PREC: u8 = 6;

fn binary_op(kind: TokenKind) -> BinaryOp {
    match kind {
        TokenKind::OrOr => BinaryOp::Or,
        TokenKind::AmpAmp => BinaryOp::And,
        TokenKind::EqEq => BinaryOp::Eq,
        TokenKind::NotEq => BinaryOp::Ne,
        TokenKind::Lt => BinaryOp::Lt,
        TokenKind::Gt => BinaryOp::Gt,
        TokenKind::LtEq => BinaryOp::Le,
        TokenKind::GtEq => BinaryOp::Ge,
        TokenKind::Plus => BinaryOp::Add,
        TokenKind::Minus => BinaryOp::Sub,
        TokenKind::Star => BinaryOp::Mul,
        TokenKind::Slash => BinaryOp::Div,
        TokenKind::Percent => BinaryOp::Mod,
        _ => unreachable!("not a binary operator"),
    }
}

fn unary_op(kind: TokenKind) -> UnaryOp {
    match kind {
        TokenKind::Minus => UnaryOp::Neg,
        TokenKind::Bang => UnaryOp::Not,
        _ => unreachable!("not a unary operator"),
    }
}

enum Entry {
    Slot(NodeId),
    Stmt(NodeId),
}

pub struct Parser<S: CharStream> {
    scanner: Scanner<S>,
    ast: Ast,
}

impl<S: CharStream> Parser<S> {
    pub fn new(stream: S) -> Self {
        Self {
            scanner: Scanner::new(stream),
            ast: Ast::new(),
        }
    }

    pub fn ast(&self) -> &Ast {
        &self.ast
    }

    pub fn symbols(&self) -> &SymbolTable {
        self.scanner.symbols()
    }

    pub fn into_ast(mut self) -> Ast {
        self.ast.set_symbol_table(self.scanner.take_symbols());
        self.ast
    }

    pub fn parse_script(&mut self) -> Result<NodeId, ParseError> {
        let _span = trace::info_span!("kette::parse").entered();
        let start = self.peek()?.span.start;
        let stmts = self.parse_statement_list(TokenKind::Eof)?;
        let end = self.peek()?.span.end;
        let list = self.ast.list(&stmts);
        let root = self
            .ast
            .add(Node::StmtList { stmts: list }, ByteSpan::new(start, end));
        self.ast.set_root(root);
        Ok(root)
    }

    // -- token plumbing ------------------------------------------------------

    fn peek(&mut self) -> Result<Token, ParseError> {
        self.scanner.peek().clone()
    }

    fn next(&mut self) -> Result<Token, ParseError> {
        self.scanner.next_token()
    }

    fn eat(&mut self, kind: TokenKind) -> Result<bool, ParseError> {
        if self.peek()?.kind == kind {
            self.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        let t = self.peek()?;
        if t.kind != kind {
            return Err(ParseError::new(
                t.span,
                format!(
                    "expected `{}`, found `{}`",
                    kind.describe(),
                    t.kind.describe()
                ),
            ));
        }
        self.next()
    }

    fn expect_semicolon(&mut self) -> Result<(), ParseError> {
        let t = self.peek()?;
        if t.kind == TokenKind::Semicolon {
            self.next()?;
            return Ok(());
        }
        if t.kind == TokenKind::RBrace || t.kind == TokenKind::Eof || t.after_newline {
            return Ok(());
        }
        Err(ParseError::new(
            t.span,
            format!("expected `;` or newline, found `{}`", t.kind.describe()),
        ))
    }

    fn token_name(&mut self, t: Token) -> Result<Symbol, ParseError> {
        match t.value {
            TokenValue::Symbol(s) => Ok(Symbol(s)),
            _ if t.kind.is_keyword() => {
                Ok(self.scanner.symbols_mut().intern(t.kind.text().as_bytes()))
            }
            _ => Err(ParseError::new(t.span, "expected a name")),
        }
    }

    fn number_node(&mut self, t: Token) -> Result<NodeId, ParseError> {
        let n = t
            .value
            .number()
            .ok_or_else(|| ParseError::new(t.span, "expected a number"))?;
        Ok(self.ast.add(Node::Number(n), t.span))
    }

    /// The primary node for a consumed name-kind token.
    fn name_primary(&mut self, t: Token) -> Result<NodeId, ParseError> {
        let node = match t.kind {
            TokenKind::Identifier => Node::Ident(Symbol(
                t.value.symbol().expect("identifier carries a symbol"),
            )),
            TokenKind::SelfKw => Node::Self_,
            TokenKind::True => Node::Bool(true),
            TokenKind::False => Node::Bool(false),
            TokenKind::Null => Node::Null,
            _ => return Err(ParseError::new(t.span, "expected an expression")),
        };
        Ok(self.ast.add(node, t.span))
    }

    // -- statements ----------------------------------------------------------

    fn parse_statement_list(&mut self, terminator: TokenKind) -> Result<Vec<NodeId>, ParseError> {
        let mut stmts = Vec::new();
        loop {
            if self.peek()?.kind == terminator {
                break;
            }
            if self.eat(TokenKind::Semicolon)? {
                continue;
            }
            if self.peek()?.kind == TokenKind::Eof {
                break;
            }
            let stmt = self.parse_statement()?;
            stmts.push(stmt);
            self.expect_semicolon()?;
        }
        Ok(stmts)
    }

    fn parse_statement(&mut self) -> Result<NodeId, ParseError> {
        if self.peek()?.kind == TokenKind::Let {
            self.parse_let()
        } else {
            let expr = self.parse_expression()?;
            let span = self.ast.span(expr);
            Ok(self.ast.add(Node::ExprStmt { expr }, span))
        }
    }

    fn parse_let(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::Let)?;
        let name_tok = self.next()?;
        let name = self.token_name(name_tok)?;
        self.expect(TokenKind::Assign)?;
        let init = self.parse_expression()?;
        let span = ByteSpan::new(kw.span.start, self.ast.span(init).end);
        Ok(self.ast.add(Node::Let { name, init }, span))
    }

    // -- expressions ---------------------------------------------------------

    fn parse_expression(&mut self) -> Result<NodeId, ParseError> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<NodeId, ParseError> {
        let lhs = self.parse_binary(1)?;
        self.parse_assignment_from(lhs)
    }

    /// `= value`, when the parsed left side turns out to be the target.
    fn parse_assignment_from(&mut self, lhs: NodeId) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        if t.kind == TokenKind::Assign && !t.after_newline {
            self.next()?;
            self.check_assign_target(lhs)?;
            let value = self.parse_assignment()?;
            let span = ByteSpan::new(self.ast.span(lhs).start, self.ast.span(value).end);
            return Ok(self.ast.add(Node::Assign { target: lhs, value }, span));
        }
        Ok(lhs)
    }

    fn check_assign_target(&self, node: NodeId) -> Result<(), ParseError> {
        match self.ast.node(node) {
            Node::Ident(_) | Node::Get { .. } | Node::Index { .. } => Ok(()),
            _ => Err(ParseError::new(
                self.ast.span(node),
                "invalid assignment target",
            )),
        }
    }

    fn parse_binary(&mut self, min_prec: u8) -> Result<NodeId, ParseError> {
        let lhs = self.parse_unary()?;
        self.parse_binary_from(lhs, min_prec)
    }

    /// Continue the precedence climb over an already-parsed left side.
    /// A binary operator only continues an expression on the same line:
    /// a leading operator starts a new statement (and lets `||` open the
    /// parameter list on its own line).
    fn parse_binary_from(&mut self, mut lhs: NodeId, min_prec: u8) -> Result<NodeId, ParseError> {
        loop {
            let t = self.peek()?;
            if t.after_newline {
                break;
            }
            let prec = t.kind.precedence();
            if prec == 0 || prec < min_prec {
                break;
            }
            let op_tok = self.next()?;
            let rhs = self.parse_binary(prec + 1)?;
            let span = ByteSpan::new(self.ast.span(lhs).start, self.ast.span(rhs).end);
            lhs = self.ast.add(
                Node::Binary {
                    op: binary_op(op_tok.kind),
                    lhs,
                    rhs,
                },
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        match t.kind {
            TokenKind::Minus | TokenKind::Bang => {
                self.next()?;
                let expr = self.parse_unary()?;
                let span = ByteSpan::new(t.span.start, self.ast.span(expr).end);
                Ok(self.ast.add(
                    Node::Unary {
                        op: unary_op(t.kind),
                        expr,
                    },
                    span,
                ))
            }
            TokenKind::Return => {
                self.next()?;
                let value = self.parse_unary()?;
                let span = ByteSpan::new(t.span.start, self.ast.span(value).end);
                Ok(self.ast.add(Node::Return { value }, span))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<NodeId, ParseError> {
        let expr = self.parse_primary()?;
        self.parse_postfix_from(expr)
    }

    /// Continue postfix parsing over an already-parsed primary.
    fn parse_postfix_from(&mut self, mut expr: NodeId) -> Result<NodeId, ParseError> {
        loop {
            let t = self.peek()?;
            match t.kind {
                TokenKind::Period => {
                    self.next()?;
                    let nt = self.peek()?;
                    if nt.kind == TokenKind::Number {
                        self.next()?;
                        let key = self.number_node(nt)?;
                        let span = ByteSpan::new(self.ast.span(expr).start, nt.span.end);
                        expr = self.ast.add(Node::Index { recv: expr, key }, span);
                    } else if is_name_kind(nt.kind) {
                        let name_tok = self.next()?;
                        let name = self.token_name(name_tok)?;
                        expr = self.finish_named(expr, name, name_tok.span.end)?;
                    } else {
                        return Err(ParseError::new(
                            nt.span,
                            format!("expected a name after `.`, found `{}`", nt.kind.describe()),
                        ));
                    }
                }
                TokenKind::LBracket if !t.after_newline => {
                    self.next()?;
                    let key = self.parse_expression()?;
                    let close = self.expect(TokenKind::RBracket)?;
                    let span = ByteSpan::new(self.ast.span(expr).start, close.span.end);
                    expr = self.ast.add(Node::Index { recv: expr, key }, span);
                }
                TokenKind::LParen if !t.after_newline => {
                    // `f(x)` — call shorthand; the callee is the receiver
                    let (args, end) = self.parse_args()?;
                    let span = ByteSpan::new(self.ast.span(expr).start, end);
                    expr = self.ast.add(Node::Call { callee: expr, args }, span);
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn finish_named(
        &mut self,
        recv: NodeId,
        name: Symbol,
        name_end: u32,
    ) -> Result<NodeId, ParseError> {
        let start = self.ast.span(recv).start;
        let next = self.peek()?;
        if next.kind == TokenKind::LParen && !next.after_newline {
            let (args, end) = self.parse_args()?;
            Ok(self
                .ast
                .add(Node::Send { recv, name, args }, ByteSpan::new(start, end)))
        } else {
            Ok(self
                .ast
                .add(Node::Get { recv, name }, ByteSpan::new(start, name_end)))
        }
    }

    fn parse_args(&mut self) -> Result<(crate::ast::NodeList, u32), ParseError> {
        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        if self.peek()?.kind != TokenKind::RParen {
            loop {
                args.push(self.parse_expression()?);
                if !self.eat(TokenKind::Comma)? {
                    break;
                }
                if self.peek()?.kind == TokenKind::RParen {
                    break;
                }
            }
        }
        let close = self.expect(TokenKind::RParen)?;
        Ok((self.ast.list(&args), close.span.end))
    }

    fn parse_primary(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        match t.kind {
            TokenKind::Number => {
                self.next()?;
                self.number_node(t)
            }
            TokenKind::String => {
                self.next()?;
                let sym = Symbol(t.value.symbol().expect("string token carries a symbol"));
                Ok(self.ast.add(Node::String(sym), t.span))
            }
            TokenKind::True | TokenKind::False | TokenKind::Null | TokenKind::SelfKw => {
                self.next()?;
                self.name_primary(t)
            }
            TokenKind::Identifier => {
                self.next()?;
                self.name_primary(t)
            }
            TokenKind::LParen => {
                self.next()?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBrace => self.parse_object(),
            TokenKind::LBracket => self.parse_array(),
            TokenKind::If => self.parse_if(),
            _ => Err(ParseError::new(
                t.span,
                format!("expected an expression, found `{}`", t.kind.describe()),
            )),
        }
    }

    // -- the brace literal ---------------------------------------------------

    /// `{ slots | params | body }`, single pass, entry by entry.
    fn parse_object(&mut self) -> Result<NodeId, ParseError> {
        let open = self.expect(TokenKind::LBrace)?;
        let mut slots = Vec::new();
        let mut stmts = Vec::new();
        let mut params = None;
        let mut saw_stmt = false;
        loop {
            let t = self.peek()?;
            if t.kind == TokenKind::RBrace {
                break;
            }
            // `|params|` — or `||`, the empty marker (an `||` on its own
            // at an entry boundary cannot continue an expression)
            if t.kind == TokenKind::Pipe || t.kind == TokenKind::OrOr {
                if saw_stmt || params.is_some() {
                    return Err(ParseError::new(t.span, "params must precede the body"));
                }
                params = Some(self.parse_params()?);
                continue;
            }
            if self.eat(TokenKind::Comma)? || self.eat(TokenKind::Semicolon)? {
                continue;
            }
            match self.parse_entry()? {
                Entry::Slot(slot) => {
                    if saw_stmt || params.is_some() {
                        return Err(ParseError::new(
                            self.ast.span(slot),
                            "slots must precede the body",
                        ));
                    }
                    slots.push(slot);
                }
                Entry::Stmt(stmt) => {
                    saw_stmt = true;
                    stmts.push(stmt);
                    self.expect_semicolon()?;
                }
            }
        }
        let close = self.expect(TokenKind::RBrace)?;
        let body_span = ByteSpan::new(open.span.end, close.span.start);
        let body_list = self.ast.list(&stmts);
        let body = self.ast.add(Node::StmtList { stmts: body_list }, body_span);
        let slots = self.ast.list(&slots);
        let span = ByteSpan::new(open.span.start, close.span.end);
        Ok(self.ast.add(
            Node::Object(ObjectParts {
                slots,
                params,
                body,
            }),
            span,
        ))
    }

    /// One entry inside braces: a slot (revealed by a trailing `:`) or a
    /// statement.
    fn parse_entry(&mut self) -> Result<Entry, ParseError> {
        let t = self.peek()?;
        // `let` is never an expression, and never a slot key
        if t.kind == TokenKind::Let {
            return Ok(Entry::Stmt(self.parse_let()?));
        }
        // `if` / `return` parse as (odd) expressions; they are keywords,
        // so they can never begin a slot
        if t.kind == TokenKind::If || t.kind == TokenKind::Return {
            let expr = self.parse_expression()?;
            return Ok(Entry::Stmt(self.expr_stmt(expr)?));
        }
        if is_name_kind(t.kind) {
            let name_tok = self.next()?;
            // `name :` — a named slot
            if self.peek()?.kind == TokenKind::Colon {
                return Ok(Entry::Slot(self.name_slot(name_tok, SlotKind::Named)?));
            }
            // `name * :` — a parent slot; any other token after `*` means
            // the star was the multiply operator
            if self.peek()?.kind == TokenKind::Star {
                self.next()?;
                if self.peek()?.kind == TokenKind::Colon {
                    return Ok(Entry::Slot(self.name_slot(name_tok, SlotKind::Parent)?));
                }
                let lhs = self.name_primary(name_tok)?;
                let rhs = self.parse_binary(MUL_PREC + 1)?;
                let span = ByteSpan::new(name_tok.span.start, self.ast.span(rhs).end);
                let lhs = self.ast.add(
                    Node::Binary {
                        op: BinaryOp::Mul,
                        lhs,
                        rhs,
                    },
                    span,
                );
                let expr = self.parse_binary_from(lhs, 1)?;
                let expr = self.parse_assignment_from(expr)?;
                return Ok(Entry::Stmt(self.expr_stmt(expr)?));
            }
            // an ordinary expression beginning with the name
            let prim = self.name_primary(name_tok)?;
            let expr = self.parse_postfix_from(prim)?;
            let expr = self.parse_binary_from(expr, 1)?;
            let expr = self.parse_assignment_from(expr)?;
            return Ok(Entry::Stmt(self.expr_stmt(expr)?));
        }
        // everything else: parse the statement; a trailing `:` turns a
        // `[key]` expression into an element slot
        let expr = self.parse_expression()?;
        if self.eat(TokenKind::Colon)? {
            let key = self.element_key(expr)?;
            let value = self.parse_expression()?;
            let span = ByteSpan::new(self.ast.span(key).start, self.ast.span(value).end);
            let slot = self.ast.add(
                Node::Slot {
                    kind: SlotKind::Element,
                    key,
                    value,
                },
                span,
            );
            return Ok(Entry::Slot(slot));
        }
        Ok(Entry::Stmt(self.expr_stmt(expr)?))
    }

    fn expr_stmt(&mut self, expr: NodeId) -> Result<NodeId, ParseError> {
        let span = self.ast.span(expr);
        Ok(self.ast.add(Node::ExprStmt { expr }, span))
    }

    /// `name` (already consumed) `:` value.
    fn name_slot(&mut self, name_tok: Token, kind: SlotKind) -> Result<NodeId, ParseError> {
        let name = self.token_name(name_tok)?;
        let key = self.ast.add(Node::Ident(name), name_tok.span);
        self.expect(TokenKind::Colon)?;
        let value = self.parse_expression()?;
        let span = ByteSpan::new(name_tok.span.start, self.ast.span(value).end);
        Ok(self.ast.add(Node::Slot { kind, key, value }, span))
    }

    /// The `[key]` of an element slot, unwrapped from the array literal
    /// the expression parser produced.
    fn element_key(&self, expr: NodeId) -> Result<NodeId, ParseError> {
        if let Node::Array { elements } = self.ast.node(expr) {
            let items = self.ast.list_items(*elements);
            if let [key] = items {
                return Ok(*key);
            }
        }
        Err(ParseError::new(
            self.ast.span(expr),
            "an element slot key is a single `[expr]`",
        ))
    }

    /// `|a, b|` between the delimiting pipes; `||` is the empty list.
    fn parse_params(&mut self) -> Result<crate::ast::NodeList, ParseError> {
        if self.eat(TokenKind::OrOr)? {
            return Ok(self.ast.list(&[]));
        }
        self.expect(TokenKind::Pipe)?;
        let mut params = Vec::new();
        if self.peek()?.kind != TokenKind::Pipe {
            loop {
                let t = self.expect(TokenKind::Identifier)?;
                let sym = Symbol(t.value.symbol().expect("identifier carries a symbol"));
                params.push(self.ast.add(Node::Ident(sym), t.span));
                if !self.eat(TokenKind::Comma)? {
                    break;
                }
            }
        }
        self.expect(TokenKind::Pipe)?;
        Ok(self.ast.list(&params))
    }

    fn parse_array(&mut self) -> Result<NodeId, ParseError> {
        let open = self.expect(TokenKind::LBracket)?;
        let mut elements = Vec::new();
        if self.peek()?.kind != TokenKind::RBracket {
            loop {
                elements.push(self.parse_expression()?);
                if !self.eat(TokenKind::Comma)? {
                    break;
                }
                if self.peek()?.kind == TokenKind::RBracket {
                    break;
                }
            }
        }
        let close = self.expect(TokenKind::RBracket)?;
        let elements = self.ast.list(&elements);
        let span = ByteSpan::new(open.span.start, close.span.end);
        Ok(self.ast.add(Node::Array { elements }, span))
    }

    // -- control flow --------------------------------------------------------

    fn parse_if(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::If)?;
        let cond = self.parse_expression()?;
        let then = self.parse_branch()?;
        let mut else_ = None;
        if self.peek()?.kind == TokenKind::Else {
            self.next()?;
            else_ = Some(if self.peek()?.kind == TokenKind::If {
                self.parse_if()?
            } else {
                self.parse_branch()?
            });
        }
        let end = else_
            .map(|e| self.ast.span(e).end)
            .unwrap_or_else(|| self.ast.span(then).end);
        let span = ByteSpan::new(kw.span.start, end);
        Ok(self.ast.add(Node::If { cond, then, else_ }, span))
    }

    /// `{ statements }` compiled inline: a plain statement list, not an
    /// object. The leading `{` has not been eaten.
    fn parse_branch(&mut self) -> Result<NodeId, ParseError> {
        let open = self.expect(TokenKind::LBrace)?;
        let stmts = self.parse_statement_list(TokenKind::RBrace)?;
        let close = self.expect(TokenKind::RBrace)?;
        let list = self.ast.list(&stmts);
        Ok(self.ast.add(
            Node::StmtList { stmts: list },
            ByteSpan::new(open.span.start, close.span.end),
        ))
    }
}
