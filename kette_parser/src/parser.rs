use crate::plumbing::ParseError;

use crate::ast::{Ast, BinaryOp, Node, NodeId, SlotKind, UnaryOp};
use crate::token::{ByteSpan, Token, TokenKind, TokenValue};
use crate::{Bookmark, CharStream, Scanner, Symbol, SymbolTable};

fn is_name_kind(kind: TokenKind) -> bool {
    kind == TokenKind::Identifier || kind.is_keyword()
}

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

    pub fn bookmark(&self) -> Bookmark {
        self.scanner.bookmark()
    }

    pub fn restore(&mut self, bookmark: Bookmark) {
        self.scanner.restore(bookmark);
    }

    pub fn parse_script(&mut self) -> Result<NodeId, ParseError> {
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

    fn peek(&mut self) -> Result<Token, ParseError> {
        self.scanner.peek().clone()
    }

    fn peek_ahead(&mut self) -> Result<Token, ParseError> {
        self.scanner.peek_ahead().clone()
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

    fn parse_expression(&mut self) -> Result<NodeId, ParseError> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<NodeId, ParseError> {
        let lhs = self.parse_binary(1)?;
        if self.peek()?.kind == TokenKind::Assign {
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
        let mut lhs = self.parse_unary()?;
        loop {
            let op = self.peek()?.kind;
            let prec = op.precedence();
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
        let mut expr = self.parse_primary()?;
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
                    // `f(x)` — call shorthand for evaluating the block
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
            TokenKind::True | TokenKind::False => {
                self.next()?;
                Ok(self.ast.add(Node::Bool(t.kind == TokenKind::True), t.span))
            }
            TokenKind::Null => {
                self.next()?;
                Ok(self.ast.add(Node::Null, t.span))
            }
            TokenKind::SelfKw => {
                self.next()?;
                Ok(self.ast.add(Node::Self_, t.span))
            }
            TokenKind::Identifier => {
                let sym = Symbol(t.value.symbol().expect("identifier carries a symbol"));
                self.next()?;
                Ok(self.ast.add(Node::Ident(sym), t.span))
            }
            TokenKind::LParen => {
                self.next()?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBrace => self.parse_brace(),
            TokenKind::LBracket => self.parse_array(),
            TokenKind::If => self.parse_if(),
            TokenKind::While => self.parse_while(),
            TokenKind::For => self.parse_for(),
            TokenKind::Match => self.parse_match(),
            TokenKind::Try => self.parse_try(),
            _ => Err(ParseError::new(
                t.span,
                format!("expected an expression, found `{}`", t.kind.describe()),
            )),
        }
    }

    fn number_node(&mut self, t: Token) -> Result<NodeId, ParseError> {
        let n = t
            .value
            .number()
            .ok_or_else(|| ParseError::new(t.span, "expected a number"))?;
        Ok(self.ast.add(Node::Number(n), t.span))
    }

    fn parse_brace(&mut self) -> Result<NodeId, ParseError> {
        let ahead = self.peek_ahead()?;
        if ahead.kind == TokenKind::Pipe {
            return self.parse_block();
        }
        if ahead.kind == TokenKind::RBrace {
            return self.parse_object();
        }
        let bookmark = self.bookmark();
        self.next()?; // consume `{`
        let is_object = self.at_slot_list();
        self.restore(bookmark);
        if is_object {
            self.parse_object()
        } else {
            self.parse_block()
        }
    }

    fn at_slot_list(&mut self) -> bool {
        let kind = match self.peek() {
            Ok(t) => t.kind,
            Err(_) => return false,
        };
        if kind == TokenKind::LBracket {
            let bookmark = self.bookmark();
            let _ = self.next();
            let mut depth = 1i32;
            loop {
                match self.peek() {
                    Ok(t) => match t.kind {
                        TokenKind::LBracket => {
                            depth += 1;
                            let _ = self.next();
                        }
                        TokenKind::RBracket => {
                            depth -= 1;
                            let _ = self.next();
                            if depth == 0 {
                                break;
                            }
                        }
                        TokenKind::Eof => break,
                        _ => {
                            let _ = self.next();
                        }
                    },
                    Err(_) => break,
                }
            }
            let is = matches!(self.peek(), Ok(t) if t.kind == TokenKind::Colon);
            self.restore(bookmark);
            return is;
        }
        if !is_name_kind(kind) {
            return false;
        }
        let bookmark = self.bookmark();
        let _ = self.next(); // name
        if matches!(self.peek(), Ok(t) if t.kind == TokenKind::Star) {
            let _ = self.next();
        }
        let is = matches!(self.peek(), Ok(t) if t.kind == TokenKind::Colon);
        self.restore(bookmark);
        is
    }

    /// `{ |params| body }` or `{ body }`; the leading `{` has not been eaten.
    fn parse_block(&mut self) -> Result<NodeId, ParseError> {
        let open = self.expect(TokenKind::LBrace)?;
        let mut params = Vec::new();
        if self.eat(TokenKind::Pipe)? {
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
        }
        let params = self.ast.list(&params);
        let stmts = self.parse_statement_list(TokenKind::RBrace)?;
        let close = self.expect(TokenKind::RBrace)?;
        let body_span = ByteSpan::new(open.span.end, close.span.start);
        let body_list = self.ast.list(&stmts);
        let body = self.ast.add(Node::StmtList { stmts: body_list }, body_span);
        let span = ByteSpan::new(open.span.start, close.span.end);
        Ok(self.ast.add(Node::Block { params, body }, span))
    }

    fn parse_object(&mut self) -> Result<NodeId, ParseError> {
        let open = self.expect(TokenKind::LBrace)?;
        let mut slots = Vec::new();
        loop {
            if self.peek()?.kind == TokenKind::RBrace {
                break;
            }
            if self.eat(TokenKind::Comma)? || self.eat(TokenKind::Semicolon)? {
                continue;
            }
            slots.push(self.parse_slot()?);
            let next = self.peek()?;
            if next.kind == TokenKind::RBrace {
                break;
            }
            if self.eat(TokenKind::Comma)? || self.eat(TokenKind::Semicolon)? {
                continue;
            }
            if next.after_newline {
                continue;
            }
            return Err(ParseError::new(
                next.span,
                "expected `,`, `;` or newline between slots",
            ));
        }
        let close = self.expect(TokenKind::RBrace)?;
        let slots = self.ast.list(&slots);
        let span = ByteSpan::new(open.span.start, close.span.end);
        Ok(self.ast.add(Node::Object { slots }, span))
    }

    fn parse_slot(&mut self) -> Result<NodeId, ParseError> {
        if self.peek()?.kind == TokenKind::LBracket {
            let open = self.next()?;
            let key = self.parse_expression()?;
            self.expect(TokenKind::RBracket)?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expression()?;
            let span = ByteSpan::new(open.span.start, self.ast.span(value).end);
            return Ok(self.ast.add(
                Node::Slot {
                    kind: SlotKind::Element,
                    key,
                    value,
                },
                span,
            ));
        }
        let name_tok = self.next()?;
        if !is_name_kind(name_tok.kind) {
            return Err(ParseError::new(
                name_tok.span,
                format!("expected a slot name, found `{}`", name_tok.kind.describe()),
            ));
        }
        let name = self.token_name(name_tok)?;
        let key = self.ast.add(Node::Ident(name), name_tok.span);
        let kind = if self.peek()?.kind == TokenKind::Star {
            self.next()?;
            SlotKind::Parent
        } else {
            SlotKind::Named
        };
        self.expect(TokenKind::Colon)?;
        let value = self.parse_expression()?;
        let span = ByteSpan::new(name_tok.span.start, self.ast.span(value).end);
        Ok(self.ast.add(Node::Slot { kind, key, value }, span))
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

    // -- control flow ---------------------------------------------------------

    fn parse_if(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::If)?;
        let cond = self.parse_expression()?;
        let then = self.parse_block()?;
        let mut else_ = None;
        if self.peek()?.kind == TokenKind::Else {
            self.next()?;
            else_ = Some(if self.peek()?.kind == TokenKind::If {
                self.parse_if()?
            } else {
                self.parse_block()?
            });
        }
        let end = else_
            .map(|e| self.ast.span(e).end)
            .unwrap_or_else(|| self.ast.span(then).end);
        let span = ByteSpan::new(kw.span.start, end);
        Ok(self.ast.add(Node::If { cond, then, else_ }, span))
    }

    fn parse_while(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::While)?;
        let cond = self.parse_expression()?;
        let body = self.parse_block()?;
        let span = ByteSpan::new(kw.span.start, self.ast.span(body).end);
        Ok(self.ast.add(Node::While { cond, body }, span))
    }

    fn parse_for(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::For)?;
        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Symbol(
            name_tok
                .value
                .symbol()
                .expect("identifier carries a symbol"),
        );
        self.expect(TokenKind::In)?;
        let iter = self.parse_expression()?;
        let body = self.parse_block()?;
        let span = ByteSpan::new(kw.span.start, self.ast.span(body).end);
        Ok(self.ast.add(Node::ForIn { name, iter, body }, span))
    }

    fn parse_match(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::Match)?;
        let scrut = self.parse_expression()?;
        self.expect(TokenKind::LBrace)?;
        let mut arms = Vec::new();
        loop {
            if self.peek()?.kind == TokenKind::RBrace {
                break;
            }
            if self.eat(TokenKind::Comma)? || self.eat(TokenKind::Semicolon)? {
                continue;
            }
            let start = self.peek()?.span.start;
            let target = if self.eat(TokenKind::Else)? {
                None
            } else {
                Some(self.parse_expression()?)
            };
            self.expect(TokenKind::Arrow)?;
            let handler = self.parse_block()?;
            let span = ByteSpan::new(start, self.ast.span(handler).end);
            arms.push(self.ast.add(Node::MatchArm { target, handler }, span));
        }
        let close = self.expect(TokenKind::RBrace)?;
        let arms = self.ast.list(&arms);
        let span = ByteSpan::new(kw.span.start, close.span.end);
        Ok(self.ast.add(Node::Match { scrut, arms }, span))
    }

    /// `try { body } catch name { handler }`; the catch binding becomes
    /// the handler block's first parameter.
    fn parse_try(&mut self) -> Result<NodeId, ParseError> {
        let kw = self.expect(TokenKind::Try)?;
        let body = self.parse_block()?;
        self.expect(TokenKind::Catch)?;
        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Symbol(
            name_tok
                .value
                .symbol()
                .expect("identifier carries a symbol"),
        );
        let param = self.ast.add(Node::Ident(name), name_tok.span);
        let params = self.ast.list(&[param]);
        self.expect(TokenKind::LBrace)?;
        let stmts = self.parse_statement_list(TokenKind::RBrace)?;
        let close = self.expect(TokenKind::RBrace)?;
        let stmt_list = self.ast.list(&stmts);
        let handler_body_span = ByteSpan::new(name_tok.span.end, close.span.start);
        let handler_body = self
            .ast
            .add(Node::StmtList { stmts: stmt_list }, handler_body_span);
        let handler = self.ast.add(
            Node::Block {
                params,
                body: handler_body,
            },
            ByteSpan::new(name_tok.span.start, close.span.end),
        );
        let span = ByteSpan::new(kw.span.start, close.span.end);
        Ok(self.ast.add(Node::Try { body, handler }, span))
    }
}
