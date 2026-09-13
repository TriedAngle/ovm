use crate::token::{Span, Token, TokenKind};
use crate::{
    Ast, Bookmark, CharStream, ClassInfo, ClassMember, DeclKind, FunctionId, FunctionInfo,
    FunctionKind, Node, NodeId, NodeList, Param, PropKind, Scanner, ScopeId, ScopeKind, Symbol,
    SymbolTable, VarKind,
};

#[derive(Debug, Clone)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl ParseError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for ParseError {}

fn is_identifier_like(kind: TokenKind) -> bool {
    kind == TokenKind::Identifier || kind.is_contextual()
}

#[derive(Clone, Copy, Default)]
struct FnFlags {
    declaration: bool,
    /// class members (and other always-strict bodies): parameter name
    /// uniqueness applies and the body is strict
    force_strict: bool,
    kind: FunctionKind,
}

fn starts_property_key(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Identifier
            | TokenKind::String
            | TokenKind::Number
            | TokenKind::LBracket
            | TokenKind::PrivateName
    ) || kind.is_keyword()
}

struct Scope {
    id: ScopeId,
    is_function: bool,
    lexically_declared: Vec<Symbol>,
    var_declared: Vec<Symbol>,
}

/// Parse-time class context: attributes `super` uses to the class owning
/// the nearest enclosing non-arrow member function.
struct ClassCtx {
    /// fn_stack depth when the class body started: members live at depths
    /// greater than this
    entry_fn_depth: usize,
    uses_super: bool,
    is_static: bool,
    /// private names declared by this class (hidden `.priv.#x` symbols)
    privates: Vec<Symbol>,
    /// `#name` references inside the body, validated at class end against
    /// this class's and enclosing classes' privates (forward refs are legal)
    private_uses: Vec<(Symbol, Span)>,
}

/// How a binding pattern declares its bound identifiers.
#[derive(Clone, Copy)]
enum PatCtx {
    /// `var`/`let`/`const` declarator
    VarDecl(VarKind),
    /// catch parameter (declared in the catch scope)
    Catch,
    /// function parameter: names collected for `begin_fn_body`
    CollectParams,
}

pub struct Parser<S: CharStream> {
    scanner: Scanner<S>,
    ast: Ast,
    errors: Vec<ParseError>,
    scopes: Vec<Scope>,
    loop_depth: u32,
    /// switch statement nesting (break targets; continue stays loop-only)
    switch_depth: u32,
    /// functions currently being parsed (ids into the Ast table); last = innermost
    fn_stack: Vec<FunctionId>,
    /// classes currently being parsed; last = innermost
    class_stack: Vec<ClassCtx>,
    next_literal_id: u32,
    /// bound names of the parameter list / pattern currently being parsed
    pattern_names: Vec<(Symbol, Span)>,
    /// unconverted CoverInitializedName nodes (`{a = 1}` outside a pattern):
    /// a Syntax Error unless the enclosing literal is rewritten to a pattern
    /// (validated at the end of each statement)
    cover_init: Vec<NodeId>,
}

impl<S: CharStream> Parser<S> {
    pub fn new(stream: S) -> Self {
        Self {
            scanner: Scanner::new(stream),
            ast: Ast::new(),
            errors: Vec::new(),
            scopes: Vec::new(),
            loop_depth: 0,
            switch_depth: 0,
            fn_stack: Vec::new(),
            class_stack: Vec::new(),
            next_literal_id: 0,
            pattern_names: Vec::new(),
            cover_init: Vec::new(),
        }
    }

    pub fn ast(&self) -> &Ast {
        &self.ast
    }

    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    pub fn symbols(&self) -> &SymbolTable {
        self.scanner.symbols()
    }

    pub fn symbols_mut(&mut self) -> &mut SymbolTable {
        self.scanner.symbols_mut()
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

    /// The top level parses as an implicit function, pre-registered as
    /// FunctionId(0); nested functions are added as they complete.
    pub fn parse_script(&mut self) -> Result<FunctionId, ParseError> {
        let start = self.peek()?.span.start;
        let literal_id = self.alloc_literal_id();
        let top_id = self.ast.add_function(FunctionInfo {
            span: Span::new(start, start),
            name: None,
            params: Vec::new(),
            formal_length: 0,
            body: None,
            literal_id,
            is_declaration: false,
            kind: FunctionKind::Normal,
            strict: false,
            field_key: None,
            lazy_data: None,
        });
        self.fn_stack.push(top_id);
        let scope_id = self.push_scope(ScopeKind::Script);
        self.ast.scope_mut(scope_id).function = Some(top_id);
        let stmts = self.parse_statement_list(TokenKind::Eof)?;
        let end = self.next()?.span.end; // consume Eof
        self.scopes.pop();
        let body = self.add_block(stmts, Span::new(start, end));
        self.ast.set_node_scope(body, scope_id);
        self.fn_stack.pop();
        let top = self.ast.function_mut(top_id);
        top.body = Some(body);
        top.span = Span::new(start, end);
        Ok(top_id)
    }

    /// Future lazy entry point: re-parse one function body from its start.
    pub fn parse_function_at(&mut self, start: u32) -> Result<FunctionId, ParseError> {
        self.scanner.seek_to(start);
        todo!("lazy re-parse entry point; needs scope summaries first")
    }

    // -- token helpers --------------------------------------------------------

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
                format!("expected `{}`, found `{}`", kind.text(), kind_text(t)),
            ));
        }
        self.next()
    }

    /// ASI: a statement ends at `;`, or without one before `}`, at Eof, or
    /// across a line terminator.
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
            format!("expected `;`, found `{}`", kind_text(t)),
        ))
    }

    /// Symbol for a token used as a name (identifier or keyword).
    fn ident_symbol(&mut self, t: Token) -> Result<Symbol, ParseError> {
        match t.kind {
            TokenKind::Identifier => Ok(Symbol(t.value.symbol().unwrap())),
            k if k.is_contextual() => Ok(self.symbols_mut().intern(k.text().as_bytes())),
            _ => Err(ParseError::new(t.span, "expected identifier")),
        }
    }

    // -- scopes ---------------------------------------------------------------

    fn alloc_literal_id(&mut self) -> u32 {
        let id = self.next_literal_id;
        self.next_literal_id += 1;
        id
    }

    fn push_scope(&mut self, kind: ScopeKind) -> ScopeId {
        let parent = self.scopes.last().map(|s| s.id);
        let id = self.ast.add_scope(kind, parent);
        if matches!(kind, ScopeKind::Script | ScopeKind::Function) {
            self.ast.scope_mut(id).strict = self
                .fn_stack
                .last()
                .is_some_and(|&f| self.ast.function(f).strict);
        }
        self.scopes.push(Scope {
            id,
            is_function: matches!(kind, ScopeKind::Script | ScopeKind::Function),
            lexically_declared: Vec::new(),
            var_declared: Vec::new(),
        });
        id
    }

    /// Innermost enclosing function/script scope (flag target for eval etc.).
    fn fn_scope_id(&self) -> ScopeId {
        self.scopes
            .iter()
            .rev()
            .find(|s| s.is_function)
            .expect("always inside a function scope")
            .id
    }

    fn declare_var(&mut self, sym: Symbol, span: Span, kind: DeclKind) -> Result<(), ParseError> {
        let idx = self
            .scopes
            .iter()
            .rposition(|s| s.is_function)
            .expect("always inside a function scope");
        let scope = &mut self.scopes[idx];
        if scope.lexically_declared.contains(&sym) {
            return Err(ParseError::new(span, "identifier already declared"));
        }
        let scope_id = scope.id;
        if !scope.var_declared.contains(&sym) {
            scope.var_declared.push(sym);
            self.ast.declare(scope_id, sym, kind, span);
        }
        Ok(())
    }

    fn declare_lexical(
        &mut self,
        sym: Symbol,
        span: Span,
        kind: DeclKind,
    ) -> Result<(), ParseError> {
        let scope_id = self.scopes.last().expect("always inside a scope").id;
        {
            let scope = self.scopes.last_mut().unwrap();
            if scope.lexically_declared.contains(&sym) || scope.var_declared.contains(&sym) {
                return Err(ParseError::new(span, "identifier already declared"));
            }
            scope.lexically_declared.push(sym);
        }
        self.ast.declare(scope_id, sym, kind, span);
        Ok(())
    }

    /// Function declarations: var-like in a function scope, lexical in a block.
    fn declare_function(&mut self, sym: Symbol, span: Span) -> Result<(), ParseError> {
        if self.scopes.last().unwrap().is_function {
            self.declare_var(sym, span, DeclKind::Function)
        } else {
            self.declare_lexical(sym, span, DeclKind::Function)
        }
    }

    // -- binding patterns ------------------------------------------------------

    /// The hidden class-scope declaration symbol for a private name.
    fn private_sym(&mut self, name: Symbol) -> Symbol {
        let mut bytes = b".priv.#".to_vec();
        bytes.extend_from_slice(self.symbols().get(name));
        self.symbols_mut().intern(&bytes)
    }

    /// Declare one pattern-bound name (ES 14.13.1: no duplicates within a
    /// pattern, regardless of `var` redeclaration leniency).
    fn declare_pattern_name(
        &mut self,
        sym: Symbol,
        span: Span,
        ctx: PatCtx,
    ) -> Result<(), ParseError> {
        if self.pattern_names.iter().any(|&(s, _)| s == sym) {
            return Err(ParseError::new(
                span,
                "duplicate name in destructuring pattern",
            ));
        }
        self.pattern_names.push((sym, span));
        match ctx {
            PatCtx::VarDecl(VarKind::Var) => self.declare_var(sym, span, DeclKind::Var),
            PatCtx::VarDecl(kind) => {
                let kind = match kind {
                    VarKind::Var => unreachable!(),
                    VarKind::Let => DeclKind::Let,
                    VarKind::Const => DeclKind::Const,
                };
                self.declare_lexical(sym, span, kind)
            }
            PatCtx::Catch => self.declare_lexical(sym, span, DeclKind::CatchParam),
            PatCtx::CollectParams => Ok(()), // declared by `begin_fn_body`
        }
    }

    /// Parse a binding pattern (or plain binding identifier). Callers reset
    /// `pattern_names` to scope the duplicate-name check (one declarator /
    /// one pattern / one parameter list).
    fn parse_binding_pattern(&mut self, ctx: PatCtx) -> Result<NodeId, ParseError> {
        self.parse_binding_pattern_inner(ctx)
    }

    fn parse_binding_pattern_inner(&mut self, ctx: PatCtx) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        match t.kind {
            TokenKind::LBracket => self.parse_array_binding_pattern(ctx),
            TokenKind::LBrace => self.parse_object_binding_pattern(ctx),
            k if is_identifier_like(k) => {
                let sym = self.ident_symbol(t)?;
                self.next()?;
                self.declare_pattern_name(sym, t.span, ctx)?;
                Ok(self.ast.add(Node::Identifier { sym }, t.span))
            }
            _ => Err(ParseError::new(
                t.span,
                "expected identifier or destructuring pattern",
            )),
        }
    }

    fn parse_array_binding_pattern(&mut self, ctx: PatCtx) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::LBracket)?.span.start;
        let mut elements = Vec::new();
        let end;
        loop {
            let t = self.peek()?;
            match t.kind {
                TokenKind::RBracket => {
                    end = self.next()?.span.end;
                    break;
                }
                TokenKind::Comma => {
                    let t = self.next()?;
                    elements.push(self.ast.add(Node::Hole, t.span));
                }
                TokenKind::Ellipsis => {
                    let t = self.next()?;
                    let target = self.parse_binding_pattern_inner(ctx)?;
                    let span = Span::new(t.span.start, self.ast.span(target).end);
                    elements.push(self.ast.add(Node::PatternRest { target }, span));
                    end = self.expect(TokenKind::RBracket)?.span.end;
                    break;
                }
                _ => {
                    let elem_start = t.span.start;
                    let target = self.parse_binding_pattern_inner(ctx)?;
                    let default = if self.eat(TokenKind::Assign)? {
                        Some(self.parse_assignment()?)
                    } else {
                        None
                    };
                    let elem_end = default
                        .map(|d| self.ast.span(d).end)
                        .unwrap_or(self.ast.span(target).end);
                    elements.push(self.ast.add(
                        Node::PatternElement { target, default },
                        Span::new(elem_start, elem_end),
                    ));
                    if self.eat(TokenKind::Comma)? {
                        continue;
                    }
                    end = self.expect(TokenKind::RBracket)?.span.end;
                    break;
                }
            }
        }
        let elements = self.ast.list(&elements);
        Ok(self
            .ast
            .add(Node::ArrayPattern { elements }, Span::new(start, end)))
    }

    fn parse_object_binding_pattern(&mut self, ctx: PatCtx) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::LBrace)?.span.start;
        let mut props = Vec::new();
        let end = loop {
            let t = self.peek()?;
            if t.kind == TokenKind::RBrace {
                break self.next()?.span.end;
            }
            if t.kind == TokenKind::Ellipsis {
                let t = self.next()?;
                let target = self.parse_binding_pattern_inner(ctx)?;
                let span = Span::new(t.span.start, self.ast.span(target).end);
                props.push(self.ast.add(Node::PatternRest { target }, span));
                if self.eat(TokenKind::Comma)? {
                    continue;
                }
                break self.expect(TokenKind::RBrace)?.span.end;
            }
            let prop_start = t.span.start;
            let (key, shorthand, computed) = self.parse_property_key()?;
            let key_span = self.ast.span(key);
            let elem = if self.eat(TokenKind::Colon)? {
                let target = self.parse_binding_pattern_inner(ctx)?;
                let default = if self.eat(TokenKind::Assign)? {
                    Some(self.parse_assignment()?)
                } else {
                    None
                };
                let end = default
                    .map(|d| self.ast.span(d).end)
                    .unwrap_or(self.ast.span(target).end);
                self.ast.add(
                    Node::PatternElement { target, default },
                    Span::new(key_span.start, end),
                )
            } else {
                // shorthand `{ a }` / `{ a = default }`: identifier key only
                let Some(sym) = shorthand else {
                    return Err(ParseError::new(
                        key_span,
                        "expected `:` after property name",
                    ));
                };
                let target = self.ast.add(Node::Identifier { sym }, key_span);
                self.declare_pattern_name(sym, key_span, ctx)?;
                let default = if self.eat(TokenKind::Assign)? {
                    Some(self.parse_assignment()?)
                } else {
                    None
                };
                let end = default
                    .map(|d| self.ast.span(d).end)
                    .unwrap_or(self.ast.span(target).end);
                self.ast.add(
                    Node::PatternElement { target, default },
                    Span::new(key_span.start, end),
                )
            };
            let end = self.ast.span(elem).end;
            props.push(self.ast.add(
                Node::PatternProperty {
                    key,
                    value: elem,
                    computed,
                },
                Span::new(prop_start, end),
            ));
            if self.eat(TokenKind::Comma)? {
                continue;
            }
            break self.expect(TokenKind::RBrace)?.span.end;
        };
        let props = self.ast.list(&props);
        Ok(self
            .ast
            .add(Node::ObjectPattern { props }, Span::new(start, end)))
    }

    // -- statements -----------------------------------------------------------

    /// Parse statements until `terminator` (not consumed). Detects the
    /// directive prologue ("use strict" etc.) at the start.
    fn parse_statement_list(&mut self, terminator: TokenKind) -> Result<Vec<NodeId>, ParseError> {
        let mut stmts = Vec::new();
        let mut prologue = true;
        loop {
            let t = self.peek()?;
            if t.kind == terminator || t.kind == TokenKind::Case || t.kind == TokenKind::Default {
                break;
            }
            if t.kind == TokenKind::Eof {
                return Err(ParseError::new(t.span, "unexpected end of input"));
            }
            let stmt = self.parse_statement()?;
            // Directive prologue: leading bare string-literal statements.
            // (Deviation: uses decoded text, so 'use\x20strict' counts too.)
            if prologue {
                if let Node::ExprStmt { expr } = *self.ast.node(stmt)
                    && let Node::StringLiteral(sym) = *self.ast.node(expr)
                {
                    if self.symbols().get(sym) == b"use strict" {
                        let fid = *self.fn_stack.last().unwrap();
                        self.ast.function_mut(fid).strict = true;
                        let scope = self.fn_scope_id();
                        self.ast.scope_mut(scope).strict = true;
                    }
                } else {
                    prologue = false;
                }
            }
            stmts.push(stmt);
        }
        Ok(stmts)
    }

    /// Peek for a declaration keyword; `let` only counts when a name or
    /// pattern follows (`let = 5`, `let.x` are identifier uses in sloppy mode).
    fn peek_var_kind(&mut self) -> Result<Option<VarKind>, ParseError> {
        let kind = match self.peek()?.kind {
            TokenKind::Var => VarKind::Var,
            TokenKind::Const => VarKind::Const,
            TokenKind::Let => {
                match self.peek_ahead()?.kind {
                    k if is_identifier_like(k) => {}
                    TokenKind::LBracket | TokenKind::LBrace => {}
                    _ => return Ok(None),
                }
                VarKind::Let
            }
            _ => return Ok(None),
        };
        Ok(Some(kind))
    }

    fn parse_labeled_statement(&mut self) -> Result<NodeId, ParseError> {
        let t = self.next()?;
        let label = self.ident_symbol(t)?;
        self.expect(TokenKind::Colon)?;
        let body = self.parse_statement()?;
        let span = Span::new(t.span.start, self.ast.span(body).end);
        Ok(self.ast.add(Node::Labeled { label, body }, span))
    }

    fn parse_statement(&mut self) -> Result<NodeId, ParseError> {
        // a CoverInitializedName (`{a = 1}`) that survives the statement
        // was never consumed by a pattern rewrite: Syntax Error (ES 14.13.3)
        let stmt = self.parse_statement_inner();
        if let Some(&node) = self.cover_init.last() {
            let span = self.ast.span(node);
            self.cover_init.clear();
            return Err(ParseError::new(
                span,
                "literal-property shorthand with initializer is only valid in destructuring patterns",
            ));
        }
        stmt
    }

    fn parse_statement_inner(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        // labeled statement: `label : Statement` (not before function/class)
        if is_identifier_like(t.kind)
            && !matches!(t.kind, TokenKind::Function | TokenKind::Class)
            && !t.after_newline
            && self.peek_ahead()?.kind == TokenKind::Colon
        {
            return self.parse_labeled_statement();
        }
        if let Some(kind) = self.peek_var_kind()? {
            return self.parse_var_decl(kind, true);
        }
        match t.kind {
            TokenKind::Function => {
                let function = self.parse_function(true)?;
                let span = self.ast.function(function).span;
                Ok(self.ast.add(Node::FunctionDecl { function }, span))
            }
            TokenKind::Class => self.parse_class(true),
            TokenKind::If => self.parse_if(),
            TokenKind::While => self.parse_while(),
            TokenKind::For => self.parse_for(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Throw => self.parse_throw(),
            TokenKind::Try => self.parse_try(),
            TokenKind::Switch => self.parse_switch(),
            TokenKind::Break | TokenKind::Continue => self.parse_break_continue(t.kind),
            TokenKind::LBrace => self.parse_block(),
            TokenKind::Semicolon => {
                let t = self.next()?;
                Ok(self.ast.add(Node::Empty, t.span))
            }
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_statement_block(&mut self) -> Result<NodeId, ParseError> {
        // link the block to the innermost active scope (function body → its
        // function scope, plain block → the block scope from parse_block)
        let scope = self.scopes.last().map(|s| s.id);
        let start = self.expect(TokenKind::LBrace)?.span.start;
        let stmts = self.parse_statement_list(TokenKind::RBrace)?;
        let end = self.expect(TokenKind::RBrace)?.span.end;
        let block = self.add_block(stmts, Span::new(start, end));
        if let Some(scope) = scope {
            self.ast.set_node_scope(block, scope);
        }
        Ok(block)
    }

    fn parse_block(&mut self) -> Result<NodeId, ParseError> {
        self.push_scope(ScopeKind::Block);
        let block = self.parse_statement_block();
        self.scopes.pop();
        block
    }

    fn add_block(&mut self, stmts: Vec<NodeId>, span: Span) -> NodeId {
        let stmts = self.ast.list(&stmts);
        self.ast.add(Node::Block { stmts }, span)
    }

    fn parse_var_decl(&mut self, kind: VarKind, need_semi: bool) -> Result<NodeId, ParseError> {
        let start = self.next()?.span.start; // var / let / const
        let mut decls = Vec::new();
        loop {
            let t = self.peek()?;
            let target = if matches!(t.kind, TokenKind::LBracket | TokenKind::LBrace) {
                self.pattern_names.clear();
                self.parse_binding_pattern(PatCtx::VarDecl(kind))?
            } else {
                let t = self.next()?;
                let name = self.ident_symbol(t)?;
                match kind {
                    VarKind::Var => self.declare_var(name, t.span, DeclKind::Var)?,
                    VarKind::Let => self.declare_lexical(name, t.span, DeclKind::Let)?,
                    VarKind::Const => self.declare_lexical(name, t.span, DeclKind::Const)?,
                }
                self.ast.add(Node::Identifier { sym: name }, t.span)
            };
            let init = if self.eat(TokenKind::Assign)? {
                Some(self.parse_assignment()?)
            } else {
                None
            };
            if kind == VarKind::Const && init.is_none() {
                return Err(ParseError::new(
                    t.span,
                    "missing initializer in const declaration",
                ));
            }
            let end = init
                .map(|i| self.ast.span(i).end)
                .unwrap_or(self.ast.span(target).end);
            decls.push(self.ast.add(
                Node::VarDeclarator { target, init },
                Span::new(t.span.start, end),
            ));
            if !self.eat(TokenKind::Comma)? {
                break;
            }
        }
        if need_semi {
            self.expect_semicolon()?;
        }
        let end = self.ast.span(*decls.last().unwrap()).end;
        let decls = self.ast.list(&decls);
        Ok(self
            .ast
            .add(Node::VarDecl { kind, decls }, Span::new(start, end)))
    }

    fn parse_if(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::If)?.span.start;
        self.expect(TokenKind::LParen)?;
        let cond = self.parse_expression()?;
        self.expect(TokenKind::RParen)?;
        let then = self.parse_statement()?;
        let else_ = if self.eat(TokenKind::Else)? {
            Some(self.parse_statement()?)
        } else {
            None
        };
        let end = else_.unwrap_or(then);
        let end = self.ast.span(end).end;
        Ok(self
            .ast
            .add(Node::If { cond, then, else_ }, Span::new(start, end)))
    }

    fn parse_while(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::While)?.span.start;
        self.expect(TokenKind::LParen)?;
        let cond = self.parse_expression()?;
        self.expect(TokenKind::RParen)?;
        self.loop_depth += 1;
        let body = self.parse_statement();
        self.loop_depth -= 1;
        let body = body?;
        let end = self.ast.span(body).end;
        Ok(self
            .ast
            .add(Node::While { cond, body }, Span::new(start, end)))
    }

    fn parse_for(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::For)?.span.start;
        // the head gets its own scope: `for (let i ...)` binds there and does
        // not leak into the enclosing scope
        let scope_id = self.push_scope(ScopeKind::For);
        self.expect(TokenKind::LParen)?;
        let init = if self.eat(TokenKind::Semicolon)? {
            None
        } else {
            let init = if let Some(kind) = self.peek_var_kind()? {
                self.parse_var_decl(kind, false)?
            } else {
                self.parse_expression()?
            };
            self.expect(TokenKind::Semicolon)?;
            Some(init)
        };
        let cond = if self.eat(TokenKind::Semicolon)? {
            None
        } else {
            let c = self.parse_expression()?;
            self.expect(TokenKind::Semicolon)?;
            Some(c)
        };
        let next = if self.peek()?.kind == TokenKind::RParen {
            None
        } else {
            Some(self.parse_expression()?)
        };
        self.expect(TokenKind::RParen)?;
        self.loop_depth += 1;
        let body = self.parse_statement();
        self.loop_depth -= 1;
        self.scopes.pop();
        let body = body?;
        let end = self.ast.span(body).end;
        let node = self.ast.add(
            Node::For {
                init,
                cond,
                next,
                body,
            },
            Span::new(start, end),
        );
        self.ast.set_node_scope(node, scope_id);
        Ok(node)
    }

    fn parse_return(&mut self) -> Result<NodeId, ParseError> {
        let t = self.expect(TokenKind::Return)?;
        // restricted production: no line terminator before the argument
        let next = self.peek()?;
        let value = if next.after_newline
            || matches!(
                next.kind,
                TokenKind::Semicolon | TokenKind::RBrace | TokenKind::Eof
            ) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        let end = value.map(|v| self.ast.span(v).end).unwrap_or(t.span.end);
        self.expect_semicolon()?;
        Ok(self
            .ast
            .add(Node::Return { value }, Span::new(t.span.start, end)))
    }

    fn parse_break_continue(&mut self, kind: TokenKind) -> Result<NodeId, ParseError> {
        let t = self.next()?;
        let mut end = t.span.end;
        let next = self.peek()?;
        // restricted production: no line terminator before the label
        let label = if !next.after_newline && is_identifier_like(next.kind) {
            let sym = self.ident_symbol(next)?;
            end = next.span.end;
            self.next()?;
            Some(sym)
        } else {
            None
        };
        let breakable = self.loop_depth > 0 || self.switch_depth > 0;
        if (kind == TokenKind::Break && !breakable)
            || (kind == TokenKind::Continue && self.loop_depth == 0)
        {
            return Err(ParseError::new(
                t.span,
                format!("`{}` outside of a loop", kind.text()),
            ));
        }
        self.expect_semicolon()?;
        let node = if kind == TokenKind::Break {
            Node::Break { label }
        } else {
            Node::Continue { label }
        };
        Ok(self.ast.add(node, Span::new(t.span.start, end)))
    }

    fn parse_expr_stmt(&mut self) -> Result<NodeId, ParseError> {
        let expr = self.parse_expression()?;
        let span = self.ast.span(expr);
        self.expect_semicolon()?;
        Ok(self.ast.add(Node::ExprStmt { expr }, span))
    }

    fn parse_throw(&mut self) -> Result<NodeId, ParseError> {
        let t = self.expect(TokenKind::Throw)?;
        // restricted production: no line terminator between throw and its argument
        if self.peek()?.after_newline {
            return Err(ParseError::new(
                t.span,
                "line terminator not allowed after `throw`",
            ));
        }
        let expr = self.parse_expression()?;
        let span = Span::new(t.span.start, self.ast.span(expr).end);
        self.expect_semicolon()?;
        Ok(self.ast.add(Node::Throw { expr }, span))
    }

    fn parse_switch(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::Switch)?.span.start;
        self.expect(TokenKind::LParen)?;
        let disc = self.parse_expression()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::LBrace)?;

        // one lexical scope for the whole switch (ES 16.2.2)
        let scope = self.push_scope(ScopeKind::Block);
        self.switch_depth += 1;

        let mut cases = Vec::new();
        loop {
            let t = self.peek()?;
            let (test, case_start) = match t.kind {
                TokenKind::Case => {
                    self.next()?;
                    let test = self.parse_expression()?;
                    self.expect(TokenKind::Colon)?;
                    (Some(test), t.span.start)
                }
                TokenKind::Default => {
                    self.next()?;
                    self.expect(TokenKind::Colon)?;
                    (None, t.span.start)
                }
                TokenKind::RBrace => break,
                _ => {
                    return Err(ParseError::new(
                        t.span,
                        "expected `case`, `default` or `}` in switch",
                    ));
                }
            };
            let stmts = self.parse_statement_list(TokenKind::RBrace)?;
            let end = stmts
                .last()
                .map(|s| self.ast.span(*s).end)
                .unwrap_or(case_start);
            let stmts = self.ast.list(&stmts);
            cases.push(
                self.ast
                    .add(Node::SwitchCase { test, stmts }, Span::new(case_start, end)),
            );
        }

        self.switch_depth -= 1;
        self.scopes.pop();
        let end = self.expect(TokenKind::RBrace)?.span.end;
        let cases = self.ast.list(&cases);
        let node = self
            .ast
            .add(Node::Switch { disc, cases }, Span::new(start, end));
        self.ast.set_node_scope(node, scope);
        Ok(node)
    }

    fn parse_try(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::Try)?.span.start;
        let try_block = self.parse_block()?;
        let mut catch_param = None;
        let mut catch_block = None;
        let mut finally_block = None;
        let mut catch_scope = None;
        if self.eat(TokenKind::Catch)? {
            // the catch param lives in the catch block's own scope:
            // `catch (e) { let e; }` is an early error, `var e` is not
            catch_scope = Some(self.push_scope(ScopeKind::Catch));
            if self.eat(TokenKind::LParen)? {
                let t = self.peek()?;
                catch_param = Some(
                    if matches!(t.kind, TokenKind::LBracket | TokenKind::LBrace) {
                        self.pattern_names.clear();
                        self.parse_binding_pattern(PatCtx::Catch)?
                    } else {
                        let t = self.next()?;
                        let sym = self.ident_symbol(t)?;
                        self.declare_lexical(sym, t.span, DeclKind::CatchParam)?;
                        self.ast.add(Node::Identifier { sym }, t.span)
                    },
                );
                self.expect(TokenKind::RParen)?;
            }
            catch_block = Some(self.parse_statement_block()?);
            self.scopes.pop();
        }
        if self.eat(TokenKind::Finally)? {
            finally_block = Some(self.parse_block()?);
        }
        if catch_block.is_none() && finally_block.is_none() {
            return Err(ParseError::new(
                Span::new(start, self.ast.span(try_block).end),
                "`try` requires a `catch` or `finally` block",
            ));
        }
        let end = finally_block.or(catch_block).unwrap_or(try_block);
        let end = self.ast.span(end).end;
        let node = self.ast.add(
            Node::TryCatch {
                try_block,
                catch_param,
                catch_block,
                finally_block,
            },
            Span::new(start, end),
        );
        if let Some(scope) = catch_scope {
            self.ast.set_node_scope(node, scope);
        }
        Ok(node)
    }

    // -- functions ------------------------------------------------------------

    fn parse_function(&mut self, is_declaration: bool) -> Result<FunctionId, ParseError> {
        let start = self.expect(TokenKind::Function)?.span.start;
        let generator = self.eat(TokenKind::Star)?;
        let t = self.peek()?;
        let name = if is_identifier_like(t.kind) {
            let sym = self.ident_symbol(t)?;
            self.next()?;
            Some(sym)
        } else if is_declaration {
            return Err(ParseError::new(
                t.span,
                "function declaration requires a name",
            ));
        } else {
            None
        };
        if let Some(name) = name.filter(|_| is_declaration) {
            self.declare_function(name, Span::new(start, start))?;
        }
        self.parse_function_rest(
            start,
            name,
            FnFlags {
                declaration: is_declaration,
                kind: if generator {
                    FunctionKind::Generator
                } else {
                    FunctionKind::Normal
                },
                ..Default::default()
            },
        )
    }

    /// `( params ) { body }` — shared by functions, methods, getters, setters.
    fn parse_function_rest(
        &mut self,
        start: u32,
        name: Option<Symbol>,
        flags: FnFlags,
    ) -> Result<FunctionId, ParseError> {
        // duplicate parameter names are an early error in strict code and
        // in non-simple lists (ES 15.1.2)
        let strict = self
            .fn_stack
            .last()
            .is_some_and(|&f| self.ast.function(f).strict)
            || flags.force_strict;
        let params = self.parse_params(strict)?;
        let fid = self.add_function_info(start, name, params, flags);
        let saved_loop_depth = self.begin_fn_body(fid);
        let body = self.parse_statement_block()?;
        self.end_fn_body(fid, start, body, saved_loop_depth)?;
        Ok(fid)
    }

    /// Formal parameter list (ES 15.1): plain identifiers, binding
    /// patterns, default initializers, and one trailing rest parameter.
    fn parse_params(&mut self, strict_unique: bool) -> Result<Vec<Param>, ParseError> {
        self.expect(TokenKind::LParen)?;
        self.pattern_names.clear();
        let mut params = Vec::new();
        let mut non_simple = false;
        if self.peek()?.kind != TokenKind::RParen {
            loop {
                if self.peek()?.kind == TokenKind::Ellipsis {
                    let t = self.next()?;
                    let target = self.parse_binding_target()?;
                    if self.peek()?.kind != TokenKind::RParen {
                        return Err(ParseError::new(
                            t.span,
                            "rest parameter must be the last formal parameter",
                        ));
                    }
                    params.push(Param {
                        target,
                        default: None,
                        rest: true,
                    });
                    non_simple = true;
                    break;
                }
                let target = self.parse_binding_target()?;
                let default = if self.eat(TokenKind::Assign)? {
                    non_simple = true;
                    Some(self.parse_assignment()?)
                } else {
                    None
                };
                if !matches!(*self.ast.node(target), Node::Identifier { .. }) {
                    non_simple = true;
                }
                params.push(Param {
                    target,
                    default,
                    rest: false,
                });
                if !self.eat(TokenKind::Comma)? {
                    break;
                }
                if self.peek()?.kind == TokenKind::RParen {
                    break; // trailing comma
                }
            }
        }
        self.expect(TokenKind::RParen)?;
        if strict_unique || non_simple {
            // every name in a non-simple list (or any strict parameter
            // list) must be unique (ES 15.1.2); sloppy simple lists keep
            // legacy duplicates (re-checked in `end_fn_body` if the body
            // turns out strict)
            for (i, &(name, span)) in self.pattern_names.iter().enumerate() {
                if self.pattern_names[..i].iter().any(|&(n, _)| n == name) {
                    return Err(ParseError::new(
                        span,
                        "duplicate parameter name not allowed in this context",
                    ));
                }
            }
        }
        Ok(params)
    }

    /// A formal-parameter binding target without an initializer: an
    /// identifier (collected for the function scope) or a nested pattern.
    /// Duplicate plain identifiers are tolerated here (sloppy simple lists
    /// allow them); `parse_params` rejects them where the spec requires.
    fn parse_binding_target(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        match t.kind {
            TokenKind::LBracket | TokenKind::LBrace => {
                self.parse_binding_pattern(PatCtx::CollectParams)
            }
            k if is_identifier_like(k) => {
                let sym = self.ident_symbol(t)?;
                self.next()?;
                self.pattern_names.push((sym, t.span));
                Ok(self.ast.add(Node::Identifier { sym }, t.span))
            }
            _ => Err(ParseError::new(
                t.span,
                "expected identifier or destructuring pattern",
            )),
        }
    }

    fn add_function_info(
        &mut self,
        start: u32,
        name: Option<Symbol>,
        params: Vec<Param>,
        flags: FnFlags,
    ) -> FunctionId {
        let strict = self
            .fn_stack
            .last()
            .is_some_and(|&f| self.ast.function(f).strict);
        // the enclosing function contains a function → gates the for-loop
        // per-iteration-environment desugar
        let enclosing = self.fn_scope_id();
        self.ast.scope_mut(enclosing).contains_function_or_eval = true;
        // f.length: parameters before the first default / rest / pattern
        let formal_length = params
            .iter()
            .position(|p| p.is_non_simple(&self.ast))
            .unwrap_or(params.len()) as u32;
        let literal_id = self.alloc_literal_id();
        self.ast.add_function(FunctionInfo {
            span: Span::new(start, start),
            name,
            params,
            formal_length,
            body: None,
            literal_id,
            is_declaration: flags.declaration,
            kind: flags.kind,
            strict,
            field_key: None,
            lazy_data: None,
        })
    }

    /// Push the function scope and declare its parameters. Plain params
    /// become register-backed `Param` decls (with their positional index);
    /// pattern-bound names become `PatternParam` decls (prologue-initialized).
    /// Returns the outer loop depth for `end_fn_body`.
    fn begin_fn_body(&mut self, fid: FunctionId) -> u32 {
        self.fn_stack.push(fid);
        let scope_id = self.push_scope(ScopeKind::Function);
        self.ast.scope_mut(scope_id).function = Some(fid);
        // params live in the function scope; sloppy simple lists allow
        // duplicates (each declared, sharing a register)
        let scope = self.scopes.last_mut().unwrap();
        let params: Vec<Param> = self.ast.function(fid).params.clone();
        for &p in &params {
            let sym = match *self.ast.node(p.target) {
                Node::Identifier { sym } => Some(sym),
                _ => None,
            };
            if let Some(sym) = sym {
                if !scope.var_declared.contains(&sym) {
                    scope.var_declared.push(sym);
                }
            }
        }
        let span = self.ast.function(fid).span;
        for (i, p) in params.iter().enumerate() {
            match *self.ast.node(p.target) {
                Node::Identifier { sym } => {
                    self.ast
                        .declare_param(scope_id, sym, DeclKind::Param, span, i as u32);
                }
                _ => {
                    let mut names = Vec::new();
                    collect_pattern_names(&self.ast, p.target, &mut names);
                    for (sym, span) in names {
                        // pattern names are var-like in the function scope
                        // (body `var x` redeclares them, `let x` is an error)
                        let scope = self.scopes.last_mut().unwrap();
                        if !scope.var_declared.contains(&sym) {
                            scope.var_declared.push(sym);
                        }
                        self.ast
                            .declare(scope_id, sym, DeclKind::PatternParam, span);
                    }
                }
            }
        }
        std::mem::replace(&mut self.loop_depth, 0)
    }

    fn end_fn_body(
        &mut self,
        fid: FunctionId,
        start: u32,
        body: NodeId,
        saved_loop_depth: u32,
    ) -> Result<(), ParseError> {
        self.loop_depth = saved_loop_depth;
        self.scopes.pop();
        self.fn_stack.pop();
        let end = self.ast.span(body).end;
        let (strict, params) = {
            let info = self.ast.function_mut(fid);
            info.span = Span::new(start, end);
            info.body = Some(body);
            (info.strict, info.params.clone())
        };
        // duplicate parameter names + a strict body directive is an early
        // error (ES 15.1.2, applied retroactively: the directive is only
        // discovered while parsing the body)
        if strict {
            let mut names: Vec<(Symbol, Span)> = Vec::new();
            for p in &params {
                collect_pattern_names(&self.ast, p.target, &mut names);
            }
            for (i, &(name, span)) in names.iter().enumerate() {
                if names[..i].iter().any(|&(n, _)| n == name) {
                    return Err(ParseError::new(
                        span,
                        "duplicate parameter name not allowed in strict mode",
                    ));
                }
            }
        }
        Ok(())
    }

    /// `=> body` after the params: expression bodies get an implicit return.
    fn parse_arrow_rest(&mut self, start: u32, params: Vec<Param>) -> Result<NodeId, ParseError> {
        let arrow = self.expect(TokenKind::Arrow)?;
        // restricted production: no line terminator before `=>`
        if arrow.after_newline {
            return Err(ParseError::new(
                arrow.span,
                "no line terminator allowed before `=>`",
            ));
        }
        // arrow parameters are always unique (non-simple rules apply to
        // every arrow, ES 15.1.2); the params were already checked in
        // `parse_assignment` via `parse_params(true)`
        let fid = self.add_function_info(
            start,
            None,
            params,
            FnFlags {
                kind: FunctionKind::Arrow,
                ..Default::default()
            },
        );
        let saved_loop_depth = self.begin_fn_body(fid);
        let body = if self.peek()?.kind == TokenKind::LBrace {
            self.parse_statement_block()?
        } else {
            let expr = self.parse_assignment()?;
            let span = self.ast.span(expr);
            let ret = self.ast.add(Node::Return { value: Some(expr) }, span);
            let block = self.add_block(vec![ret], span);
            // link the expression body to the arrow's function scope so
            // scope analysis (context depths) sees the function boundary
            let scope = self.scopes.last().expect("arrow fn scope").id;
            self.ast.set_node_scope(block, scope);
            block
        };
        self.end_fn_body(fid, start, body, saved_loop_depth)?;
        let span = self.ast.function(fid).span;
        Ok(self.ast.add(Node::FunctionExpr { function: fid }, span))
    }

    /// On `(`, token-scan ahead to the matching `)` and check for `=>`:
    /// a cheap speculative test for a parenthesized arrow parameter list
    /// (patterns/defaults/rest included). Consumes nothing.
    fn arrow_params_ahead(&mut self) -> Result<bool, ParseError> {
        let bm = self.bookmark();
        let is_arrow = (|| {
            self.next()?; // (
            let mut depth = 1usize;
            while depth > 0 {
                let t = self.next()?;
                match t.kind {
                    TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => depth += 1,
                    TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => depth -= 1,
                    TokenKind::Eof => return Ok(false),
                    _ => {}
                }
            }
            Ok(self.peek()?.kind == TokenKind::Arrow)
        })();
        self.restore(bm);
        is_arrow
    }

    // -- classes --------------------------------------------------------------

    /// Record a `#name` reference for end-of-class validation (forward
    /// references within the class body are legal, so checks are deferred).
    fn note_private_use(&mut self, hidden: Symbol, span: Span) -> Result<(), ParseError> {
        if self.class_stack.is_empty() {
            return Err(ParseError::new(
                span,
                "private names are only valid inside a class",
            ));
        }
        self.class_stack
            .last_mut()
            .unwrap()
            .private_uses
            .push((hidden, span));
        Ok(())
    }

    /// Class declarations and expressions (ES 15.7): methods, accessors,
    /// public/private instance and static fields.
    ///
    /// Returns the ClassDecl/ClassExpr node; the class inner scope (name
    /// binding, super home objects, private names) is attached as the
    /// node's scope.
    fn parse_class(&mut self, is_declaration: bool) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::Class)?.span.start;
        let t = self.peek()?;
        let name = if is_identifier_like(t.kind) {
            let sym = self.ident_symbol(t)?;
            self.next()?;
            Some(sym)
        } else if is_declaration {
            return Err(ParseError::new(t.span, "class declaration requires a name"));
        } else {
            None
        };
        // class declarations are lexical (TDZ like let)
        if let Some(name) = name.filter(|_| is_declaration) {
            self.declare_lexical(name, t.span, DeclKind::Class)?;
        }

        // class inner scope: the immutable class-name binding (in TDZ during
        // heritage/computed-key evaluation) plus the super home-object slots
        let class_scope = self.push_scope(ScopeKind::Class);
        if let Some(name) = name {
            self.declare_lexical(name, t.span, DeclKind::Class)?;
        }

        let superclass = if self.eat(TokenKind::Extends)? {
            Some(self.parse_assignment()?)
        } else {
            None
        };
        self.expect(TokenKind::LBrace)?;
        let mut members = Vec::new();
        let mut has_constructor = false;
        self.class_stack.push(ClassCtx {
            entry_fn_depth: self.fn_stack.len(),
            uses_super: false,
            is_static: false,
            privates: Vec::new(),
            private_uses: Vec::new(),
        });
        let end = loop {
            let t = self.peek()?;
            match t.kind {
                TokenKind::RBrace => break self.next()?.span.end,
                TokenKind::Semicolon => {
                    self.next()?; // stray `;` is allowed in class bodies
                    continue;
                }
                _ => {}
            }
            let member_start = t.span.start;

            // `static` modifier vs a member literally named `static`
            let mut is_static = false;
            if t.kind == TokenKind::Static {
                let ahead = self.peek_ahead()?;
                if ahead.kind == TokenKind::LBrace {
                    return Err(ParseError::new(
                        ahead.span,
                        "class static blocks are not supported yet",
                    ));
                }
                if starts_property_key(ahead.kind) {
                    is_static = true;
                    self.next()?;
                }
            }
            self.class_stack.last_mut().unwrap().is_static = is_static;

            let accessor = self.eat_accessor_prefix()?;
            let (key, key_sym, computed) = self.parse_property_key()?;
            let key_span = self.ast.span(key);
            let is_private = matches!(*self.ast.node(key), Node::PrivateName { .. });

            // early errors (ES 15.7.1)
            let is_constructor_name = !computed
                && !is_static
                && !is_private
                && key_sym.is_some_and(|s| self.symbols().get(s) == b"constructor");
            if is_constructor_name && accessor.is_some() {
                return Err(ParseError::new(
                    key_span,
                    "constructor can't be an accessor",
                ));
            }
            if is_constructor_name {
                if has_constructor {
                    return Err(ParseError::new(key_span, "duplicate constructor"));
                }
                has_constructor = true;
            }
            // a static member named "prototype" is a Syntax Error; a static
            // member named "constructor" is an ordinary method
            if is_static
                && !computed
                && !is_private
                && key_sym.is_some_and(|s| self.symbols().get(s) == b"prototype")
            {
                return Err(ParseError::new(
                    key_span,
                    "class may not have a static method named 'prototype'",
                ));
            }

            if self.peek()?.kind == TokenKind::LParen {
                // method or accessor
                if is_private {
                    return Err(ParseError::new(
                        key_span,
                        "private methods and accessors are not supported yet",
                    ));
                }
                let kind = match accessor {
                    Some(true) => PropKind::Get,
                    Some(false) => PropKind::Set,
                    None => PropKind::Method,
                };
                let function_kind = if is_constructor_name {
                    if superclass.is_some() {
                        FunctionKind::DerivedClassConstructor
                    } else {
                        FunctionKind::BaseClassConstructor
                    }
                } else {
                    function_kind_for_property(kind)
                };
                // class members are always strict; accessors carry the
                // "get x"/"set x" name (computed keys get theirs at runtime)
                let fn_name = match kind {
                    PropKind::Get => self.accessor_name(key_sym, true),
                    PropKind::Set => self.accessor_name(key_sym, false),
                    _ => key_sym,
                };
                let (value, _) = self.parse_method_value(
                    member_start,
                    fn_name,
                    kind,
                    function_kind,
                    true,
                    key_span,
                )?;
                members.push(ClassMember {
                    key,
                    value,
                    kind,
                    is_static,
                    is_constructor: is_constructor_name,
                    computed,
                    is_private,
                });
            } else {
                // field definition (ES 15.7.19): public or private,
                // instance or static
                if accessor.is_some() {
                    return Err(ParseError::new(key_span, "class fields can't be accessors"));
                }
                if is_constructor_name {
                    return Err(ParseError::new(
                        key_span,
                        "class may not have a field named 'constructor'",
                    ));
                }
                if is_static
                    && !computed
                    && !is_private
                    && key_sym.is_some_and(|s| self.symbols().get(s) == b"prototype")
                {
                    return Err(ParseError::new(
                        key_span,
                        "class may not have a static field named 'prototype'",
                    ));
                }
                if is_private && computed {
                    return Err(ParseError::new(key_span, "private names can't be computed"));
                }
                if is_private {
                    // declare the private name in this class's environment:
                    // a hidden class-scope slot holding a fresh Symbol per
                    // class evaluation
                    let Node::PrivateName { sym: hidden } = *self.ast.node(key) else {
                        unreachable!("private key node");
                    };
                    let ctx = self.class_stack.last_mut().unwrap();
                    if ctx.privates.contains(&hidden) {
                        return Err(ParseError::new(key_span, "duplicate private name in class"));
                    }
                    ctx.privates.push(hidden);
                    self.declare_class_slot(class_scope, hidden, key_span);
                }
                let value = self.parse_field_initializer(member_start, key)?;
                members.push(ClassMember {
                    key,
                    value,
                    kind: PropKind::Field,
                    is_static,
                    is_constructor: false,
                    computed,
                    is_private,
                });
                self.expect_semicolon()?;
            }
        };
        let ctx = self.class_stack.pop().unwrap();

        // `#name` references must resolve to a private name of this class
        // or an enclosing one (ES 15.7.2)
        for (hidden, span) in &ctx.private_uses {
            let declared = ctx.privates.contains(hidden)
                || self.class_stack.iter().any(|c| c.privates.contains(hidden));
            if !declared {
                return Err(ParseError::new(
                    *span,
                    "private name must be declared in an enclosing class",
                ));
            }
        }

        // default constructor for classes without an explicit one:
        // base → empty body, derived → forward all arguments to super()
        let ctor = if has_constructor {
            members
                .iter()
                .find(|m| m.is_constructor)
                .and_then(|m| match *self.ast.node(m.value) {
                    Node::FunctionExpr { function } => Some(function),
                    _ => None,
                })
                .expect("constructor member holds a function")
        } else {
            self.synthesize_default_ctor(start, end, name, superclass.is_some())?
        };

        // super home-object slots (hidden const bindings in the class scope)
        let (home, static_home) = if ctx.uses_super {
            let home = self.symbols_mut().intern(b".home_object");
            self.declare_class_slot(class_scope, home, Span::new(start, start));
            let static_home = self.symbols_mut().intern(b".static_home_object");
            self.declare_class_slot(class_scope, static_home, Span::new(start, start));
            (Some(home), Some(static_home))
        } else {
            (None, None)
        };

        self.scopes.pop();
        let class = self.ast.add_class(ClassInfo {
            span: Span::new(start, end),
            name,
            superclass,
            members,
            ctor,
            uses_super: ctx.uses_super,
            home,
            static_home,
            privates: ctx.privates,
        });
        let node = self.ast.add(
            if is_declaration {
                Node::ClassDecl { class }
            } else {
                Node::ClassExpr { class }
            },
            Span::new(start, end),
        );
        self.ast.set_node_scope(node, class_scope);
        Ok(node)
    }

    /// The synthesized field-initializer function (ES 15.7.19): a strict
    /// class-member function `return <init>;` — `= init` is parsed inside
    /// so `this`/`super.x` attribute to this class. Returns the
    /// FunctionExpr node.
    fn parse_field_initializer(&mut self, start: u32, key: NodeId) -> Result<NodeId, ParseError> {
        let name = match *self.ast.node(key) {
            Node::StringLiteral(sym) => Some(sym),
            _ => None,
        };
        let fid = self.add_function_info(
            start,
            name,
            Vec::new(),
            FnFlags {
                force_strict: true,
                kind: FunctionKind::Method,
                ..Default::default()
            },
        );
        // NamedEvaluation: an anonymous function value returned by the
        // initializer is named after the field key (ES 15.7.19)
        let key_for_naming = match *self.ast.node(key) {
            Node::StringLiteral(_) => Some(key),
            // private fields name their values "#name" (ES 6.2.12)
            Node::PrivateName { sym } => {
                let hidden = self.symbols().get(sym).to_vec();
                let raw = hidden
                    .strip_prefix(b".priv.#".as_slice())
                    .unwrap_or(&hidden[6.min(hidden.len())..]);
                let mut bytes = b"#".to_vec();
                bytes.extend_from_slice(raw);
                let text = self.symbols_mut().intern(&bytes);
                let span = self.ast.span(key);
                Some(self.ast.add(Node::StringLiteral(text), span))
            }
            _ => None,
        };
        if let Some(key) = key_for_naming {
            self.ast.function_mut(fid).field_key = Some(key);
        }
        let saved_loop_depth = self.begin_fn_body(fid);
        let body = (|| {
            let init = if self.eat(TokenKind::Assign)? {
                Some(self.parse_assignment()?)
            } else {
                None
            };
            let end = init.map(|i| self.ast.span(i).end).unwrap_or(start);
            let ret = self
                .ast
                .add(Node::Return { value: init }, Span::new(start, end));
            let block = self.add_block(vec![ret], Span::new(start, end));
            let scope = self.scopes.last().expect("initializer fn scope").id;
            self.ast.set_node_scope(block, scope);
            Ok(block)
        })();
        let body = body?;
        self.end_fn_body(fid, start, body, saved_loop_depth)?;
        let span = self.ast.function(fid).span;
        Ok(self.ast.add(Node::FunctionExpr { function: fid }, span))
    }

    fn declare_class_slot(&mut self, scope: ScopeId, sym: Symbol, span: Span) {
        self.ast.declare(scope, sym, DeclKind::Const, span);
        let s = self.scopes.last_mut().unwrap();
        if !s.lexically_declared.contains(&sym) {
            s.lexically_declared.push(sym);
        }
    }

    /// `constructor() {}` (base) / `constructor(...args) { super(...args) }`
    /// (derived) for classes without an explicit constructor (ES 15.7.13).
    fn synthesize_default_ctor(
        &mut self,
        start: u32,
        end: u32,
        name: Option<Symbol>,
        derived: bool,
    ) -> Result<FunctionId, ParseError> {
        let kind = if derived {
            FunctionKind::DefaultDerivedConstructor
        } else {
            FunctionKind::BaseClassConstructor
        };
        let fid = self.add_function_info(
            start,
            name,
            Vec::new(),
            FnFlags {
                declaration: false,
                kind,
                ..Default::default()
            },
        );
        let scope_id = self.push_scope(ScopeKind::Function);
        self.ast.scope_mut(scope_id).function = Some(fid);
        let body = self.add_block(Vec::new(), Span::new(start, end));
        self.ast.set_node_scope(body, scope_id);
        self.scopes.pop();
        let info = self.ast.function_mut(fid);
        info.body = Some(body);
        info.strict = true; // class bodies are strict
        Ok(fid)
    }

    // -- super ------------------------------------------------------------------

    /// Parse `super.x`, `super[key]`, or `super(...)` after validating the
    /// syntactic context (ES 15.4).
    fn parse_super(&mut self, span: Span) -> Result<NodeId, ParseError> {
        // the nearest enclosing non-arrow function must be a class member
        let fn_idx = self
            .fn_stack
            .iter()
            .rposition(|&f| !self.ast.function(f).kind.is_arrow());
        let member_kind = fn_idx.map(|i| self.ast.function(self.fn_stack[i]).kind);
        let Some(kind) = member_kind.filter(|k| k.is_class_member()) else {
            return Err(ParseError::new(span, "'super' outside of a class method"));
        };
        // attribute the use to the class owning that member function
        let owning = match fn_idx {
            Some(i) => self.class_stack.iter().rposition(|c| c.entry_fn_depth <= i),
            None => None,
        };
        let Some(owning) = owning else {
            return Err(ParseError::new(span, "'super' outside of a class method"));
        };
        let is_static = self.class_stack[owning].is_static;
        self.next()?; // consume `super`

        let t = self.peek()?;
        match t.kind {
            TokenKind::Period => {
                self.next()?;
                let name = self.next()?;
                if name.kind == TokenKind::PrivateName {
                    return Err(ParseError::new(
                        name.span,
                        "private fields may not be accessed on 'super'",
                    ));
                }
                let sym = match name.kind {
                    TokenKind::Identifier => Symbol(name.value.symbol().unwrap()),
                    k if k.is_keyword() => self.symbols_mut().intern(k.text().as_bytes()),
                    _ => return Err(ParseError::new(name.span, "expected property name")),
                };
                let key = self.ast.add(Node::StringLiteral(sym), name.span);
                let span = Span::new(span.start, name.span.end);
                self.class_stack[owning].uses_super = true;
                Ok(self.ast.add(
                    Node::SuperProperty {
                        key,
                        computed: false,
                        is_static,
                    },
                    span,
                ))
            }
            TokenKind::LBracket => {
                self.next()?;
                let key = self.parse_expression()?;
                let end = self.expect(TokenKind::RBracket)?.span.end;
                self.class_stack[owning].uses_super = true;
                Ok(self.ast.add(
                    Node::SuperProperty {
                        key,
                        computed: true,
                        is_static,
                    },
                    Span::new(span.start, end),
                ))
            }
            TokenKind::LParen => {
                // super() is valid anywhere lexically inside a derived
                // constructor body: directly, or delegated through arrows
                if kind != FunctionKind::DerivedClassConstructor {
                    return Err(ParseError::new(
                        span,
                        "'super()' call outside a derived class constructor",
                    ));
                }
                let (args, end) = self.parse_args()?;
                Ok(self
                    .ast
                    .add(Node::SuperCall { args }, Span::new(span.start, end)))
            }
            _ => Err(ParseError::new(
                t.span,
                "expected '.', '[' or '(' after 'super'",
            )),
        }
    }

    // -- expressions ------------------------------------------------------------

    /// Sequence expressions: `a, b, c` (lowest precedence, ES 13.16).
    fn parse_expression(&mut self) -> Result<NodeId, ParseError> {
        let mut expr = self.parse_assignment()?;
        while self.eat(TokenKind::Comma)? {
            let rhs = self.parse_assignment()?;
            let span = Span::new(self.ast.span(expr).start, self.ast.span(rhs).end);
            expr = self.ast.add(
                Node::Binary {
                    op: TokenKind::Comma,
                    lhs: expr,
                    rhs,
                },
                span,
            );
        }
        Ok(expr)
    }

    fn parse_assignment(&mut self) -> Result<NodeId, ParseError> {
        // arrow functions: `x => ...` or `(params) => ...`
        let t = self.peek()?;
        if is_identifier_like(t.kind) && self.peek_ahead()?.kind == TokenKind::Arrow {
            let sym = self.ident_symbol(t)?;
            self.next()?;
            let target = self.ast.add(Node::Identifier { sym }, t.span);
            return self.parse_arrow_rest(
                t.span.start,
                vec![Param {
                    target,
                    default: None,
                    rest: false,
                }],
            );
        }
        if t.kind == TokenKind::LParen && self.arrow_params_ahead()? {
            let start = t.span.start;
            let params = self.parse_params(true)?;
            return self.parse_arrow_rest(start, params);
        }
        let lhs = self.parse_conditional()?;
        let t = self.peek()?;
        if !t.kind.is_assignment() {
            return Ok(lhs);
        }
        if t.kind == TokenKind::Assign
            && matches!(
                *self.ast.node(lhs),
                Node::ArrayLiteral { .. } | Node::ObjectLiteral { .. }
            )
        {
            // cover grammar: `[a, b] = v` / `({a} = v)` reinterpret the
            // literal as an assignment pattern (ES 14.13.3)
            let target = self.rewrite_assignment_pattern(lhs)?;
            self.next()?;
            let value = self.parse_assignment()?; // right-associative
            let span = Span::new(self.ast.span(target).start, self.ast.span(value).end);
            return Ok(self.ast.add(
                Node::Assign {
                    op: TokenKind::Assign,
                    target,
                    value,
                },
                span,
            ));
        }
        self.check_assign_target(lhs)?;
        let op = t.kind;
        self.next()?;
        let value = self.parse_assignment()?; // right-associative
        let span = Span::new(self.ast.span(lhs).start, self.ast.span(value).end);
        Ok(self.ast.add(
            Node::Assign {
                op,
                target: lhs,
                value,
            },
            span,
        ))
    }

    /// Reinterpret an array/object literal in place as a destructuring
    /// assignment pattern (node id preserved). `[a = 1]` covers the default
    /// as an Assign node, `{a = 1}` as a CoverInitializedName.
    fn rewrite_assignment_pattern(&mut self, node: NodeId) -> Result<NodeId, ParseError> {
        let span = self.ast.span(node);
        match *self.ast.node(node) {
            Node::ArrayLiteral { elements } => {
                let items = self.ast.list_items(elements).to_vec();
                let mut out = Vec::with_capacity(items.len());
                for &el in &items {
                    let el_span = self.ast.span(el);
                    match *self.ast.node(el) {
                        Node::Hole => out.push(el),
                        Node::Assign {
                            op: TokenKind::Assign,
                            target,
                            value,
                        } => {
                            let target = self.convert_assignment_target(target)?;
                            out.push(self.ast.add(
                                Node::PatternElement {
                                    target,
                                    default: Some(value),
                                },
                                el_span,
                            ));
                        }
                        Node::Spread { expr } => {
                            let target = self.convert_assignment_target(expr)?;
                            out.push(self.ast.add(Node::PatternRest { target }, el_span));
                        }
                        _ => {
                            let target = self.convert_assignment_target(el)?;
                            out.push(self.ast.add(
                                Node::PatternElement {
                                    target,
                                    default: None,
                                },
                                el_span,
                            ));
                        }
                    }
                }
                let elements = self.ast.list(&out);
                self.ast
                    .replace(node, Node::ArrayPattern { elements }, span);
                Ok(node)
            }
            Node::ObjectLiteral { props } => {
                let items = self.ast.list_items(props).to_vec();
                let mut out = Vec::with_capacity(items.len());
                for &prop in &items {
                    let prop_span = self.ast.span(prop);
                    match *self.ast.node(prop) {
                        Node::Spread { expr } => {
                            let target = self.convert_assignment_target(expr)?;
                            out.push(self.ast.add(Node::PatternRest { target }, prop_span));
                        }
                        Node::ObjectProperty {
                            key,
                            value,
                            kind: PropKind::Init,
                            computed,
                        } => {
                            let v_span = self.ast.span(value);
                            let (target, default) = match *self.ast.node(value) {
                                // CoverInitializedName `{a = 1}`
                                Node::Assign {
                                    op: TokenKind::Assign,
                                    target,
                                    value,
                                } => (self.convert_assignment_target(target)?, Some(value)),
                                _ => (self.convert_assignment_target(value)?, None),
                            };
                            let elem = self
                                .ast
                                .add(Node::PatternElement { target, default }, v_span);
                            out.push(self.ast.add(
                                Node::PatternProperty {
                                    key,
                                    value: elem,
                                    computed,
                                },
                                prop_span,
                            ));
                            // a consumed CoverInitializedName is no longer
                            // pending (its value node was the Assign)
                            self.cover_init.retain(|&id| id != value);
                        }
                        _ => {
                            return Err(ParseError::new(
                                prop_span,
                                "invalid destructuring assignment target",
                            ));
                        }
                    }
                }
                let props = self.ast.list(&out);
                self.ast.replace(node, Node::ObjectPattern { props }, span);
                Ok(node)
            }
            _ => Err(ParseError::new(
                span,
                "invalid destructuring assignment target",
            )),
        }
    }

    /// Validate/convert one assignment-pattern target: identifiers, member
    /// expressions, and nested literal-covered patterns.
    fn convert_assignment_target(&mut self, node: NodeId) -> Result<NodeId, ParseError> {
        let span = self.ast.span(node);
        match *self.ast.node(node) {
            Node::Identifier { .. } | Node::Property { .. } => Ok(node),
            Node::ArrayLiteral { .. } | Node::ObjectLiteral { .. } => {
                self.rewrite_assignment_pattern(node)
            }
            _ => Err(ParseError::new(
                span,
                "invalid destructuring assignment target",
            )),
        }
    }

    fn parse_conditional(&mut self) -> Result<NodeId, ParseError> {
        let cond = self.parse_binary(1)?;
        if !self.eat(TokenKind::Question)? {
            return Ok(cond);
        }
        let then = self.parse_assignment()?;
        self.expect(TokenKind::Colon)?;
        let else_ = self.parse_assignment()?;
        let span = Span::new(self.ast.span(cond).start, self.ast.span(else_).end);
        Ok(self.ast.add(Node::Conditional { cond, then, else_ }, span))
    }

    /// Precedence climbing: one loop over the token precedence table.
    fn parse_binary(&mut self, min_prec: u8) -> Result<NodeId, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let t = self.peek()?;
            let prec = t.kind.precedence();
            if prec == 0 || prec < min_prec {
                break;
            }
            let op = t.kind;
            self.next()?;
            let next_min = if op.is_right_associative() {
                prec
            } else {
                prec + 1
            };
            let rhs = self.parse_binary(next_min)?;
            let span = Span::new(self.ast.span(lhs).start, self.ast.span(rhs).end);
            lhs = self.ast.add(Node::Binary { op, lhs, rhs }, span);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        match t.kind {
            TokenKind::Bang
            | TokenKind::Tilde
            | TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Typeof
            | TokenKind::Void
            | TokenKind::Delete => {
                self.next()?;
                let expr = self.parse_unary()?;
                let span = Span::new(t.span.start, self.ast.span(expr).end);
                Ok(self.ast.add(Node::Unary { op: t.kind, expr }, span))
            }
            TokenKind::PlusPlus | TokenKind::MinusMinus => {
                self.next()?;
                let target = self.parse_unary()?;
                self.check_assign_target(target)?;
                let span = Span::new(t.span.start, self.ast.span(target).end);
                Ok(self.ast.add(
                    Node::Update {
                        op: t.kind,
                        prefix: true,
                        target,
                    },
                    span,
                ))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<NodeId, ParseError> {
        let expr = self.parse_member()?;
        let t = self.peek()?;
        // restricted production: no line terminator before postfix ++/--
        if matches!(t.kind, TokenKind::PlusPlus | TokenKind::MinusMinus) && !t.after_newline {
            self.next()?;
            self.check_assign_target(expr)?;
            let span = Span::new(self.ast.span(expr).start, t.span.end);
            return Ok(self.ast.add(
                Node::Update {
                    op: t.kind,
                    prefix: false,
                    target: expr,
                },
                span,
            ));
        }
        Ok(expr)
    }

    /// primary followed by any run of `.name`, `[expr]`, `(args)`.
    fn parse_member(&mut self) -> Result<NodeId, ParseError> {
        let expr = match self.peek()?.kind {
            TokenKind::New => self.parse_new()?,
            _ => self.parse_primary()?,
        };
        self.parse_member_tail(expr, true)
    }

    /// `new f`, `new f()`, `new a.b.c()`, `new new f()()`.
    /// The callee binds member tails but NOT call parens: `new a.b()` is
    /// `new (a.b)()` while `new a().b` is `(new a()).b`.
    fn parse_new(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::New)?.span.start;
        if self.peek()?.kind == TokenKind::Period {
            // `new.target` (ES 13.3.11): a member-style primary, NOT a
            // construction — returned directly so member tails and calls
            // apply to it (`new.target.x`, `new new.target()`)
            let dot = self.next()?;
            let t = self.peek()?;
            let is_target = t.kind == TokenKind::Identifier
                && t.value
                    .symbol()
                    .is_some_and(|sym| self.symbols().get(Symbol(sym)) == b"target");
            if !is_target {
                return Err(ParseError::new(
                    dot.span,
                    "expected property name after `new.`",
                ));
            }
            self.next()?;
            return Ok(self.ast.add(Node::NewTarget, Span::new(start, t.span.end)));
        }
        let callee = match self.peek()?.kind {
            TokenKind::New => self.parse_new()?,
            _ => {
                let base = self.parse_primary()?;
                self.parse_member_tail(base, false)?
            }
        };
        if matches!(*self.ast.node(callee), Node::SuperCall { .. }) {
            return Err(ParseError::new(
                self.ast.span(callee),
                "'super' is not a constructor",
            ));
        }
        let (args, end) = if self.peek()?.kind == TokenKind::LParen {
            let (args, end) = self.parse_args()?;
            (Some(args), end)
        } else {
            (None, self.ast.span(callee).end)
        };
        Ok(self
            .ast
            .add(Node::New { callee, args }, Span::new(start, end)))
    }

    fn parse_member_tail(
        &mut self,
        mut expr: NodeId,
        allow_call: bool,
    ) -> Result<NodeId, ParseError> {
        loop {
            let t = self.peek()?;
            match t.kind {
                TokenKind::Period => {
                    self.next()?;
                    let name = self.next()?;
                    let sym = match name.kind {
                        TokenKind::Identifier => Symbol(name.value.symbol().unwrap()),
                        // any keyword may be a property name: `a.class`
                        k if k.is_keyword() => self.symbols_mut().intern(k.text().as_bytes()),
                        // `obj.#x`: a private name reference, resolved
                        // against the enclosing class's private environment
                        TokenKind::PrivateName => {
                            let name_sym = Symbol(name.value.symbol().unwrap());
                            let hidden = self.private_sym(name_sym);
                            self.note_private_use(hidden, name.span)?;
                            let node = self.ast.add(Node::PrivateName { sym: hidden }, name.span);
                            let span = Span::new(self.ast.span(expr).start, name.span.end);
                            expr = self.ast.add(
                                Node::Property {
                                    object: expr,
                                    key: node,
                                    computed: false,
                                },
                                span,
                            );
                            continue;
                        }
                        _ => return Err(ParseError::new(name.span, "expected property name")),
                    };
                    let key = self.ast.add(Node::StringLiteral(sym), name.span);
                    let span = Span::new(self.ast.span(expr).start, name.span.end);
                    expr = self.ast.add(
                        Node::Property {
                            object: expr,
                            key,
                            computed: false,
                        },
                        span,
                    );
                }
                TokenKind::LBracket => {
                    self.next()?;
                    let key = self.parse_expression()?;
                    let end = self.expect(TokenKind::RBracket)?.span.end;
                    let span = Span::new(self.ast.span(expr).start, end);
                    expr = self.ast.add(
                        Node::Property {
                            object: expr,
                            key,
                            computed: true,
                        },
                        span,
                    );
                }
                TokenKind::LParen if allow_call => {
                    // direct eval: `eval(...)` — every visible function scope
                    // must keep its bindings dynamically resolvable
                    // (context-allocated with names), not just the innermost
                    if let Node::Identifier { sym } = self.ast.node(expr)
                        && self.symbols().get(*sym) == b"eval"
                    {
                        let fns: Vec<ScopeId> = self
                            .scopes
                            .iter()
                            .filter(|s| s.is_function)
                            .map(|s| s.id)
                            .collect();
                        for id in fns {
                            self.ast.scope_mut(id).calls_eval = true;
                        }
                        let scope = self.fn_scope_id();
                        self.ast.scope_mut(scope).contains_function_or_eval = true;
                    }
                    let start = self.ast.span(expr).start;
                    let (args, end) = self.parse_args()?;
                    expr = self
                        .ast
                        .add(Node::Call { callee: expr, args }, Span::new(start, end));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_args(&mut self) -> Result<(NodeList, u32), ParseError> {
        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        if self.peek()?.kind != TokenKind::RParen {
            loop {
                let t = self.peek()?;
                if t.kind == TokenKind::Ellipsis {
                    self.next()?;
                    let expr = self.parse_assignment()?;
                    let span = Span::new(t.span.start, self.ast.span(expr).end);
                    args.push(self.ast.add(Node::Spread { expr }, span));
                } else {
                    args.push(self.parse_assignment()?);
                }
                if !self.eat(TokenKind::Comma)? {
                    break;
                }
                if self.peek()?.kind == TokenKind::RParen {
                    break; // trailing comma
                }
            }
        }
        let end = self.expect(TokenKind::RParen)?.span.end;
        Ok((self.ast.list(&args), end))
    }

    fn parse_primary(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        match t.kind {
            TokenKind::Number => {
                self.next()?;
                let n = t.value.number().unwrap();
                Ok(self.ast.add(Node::NumberLiteral(n), t.span))
            }
            TokenKind::String => {
                self.next()?;
                let sym = Symbol(t.value.symbol().unwrap());
                Ok(self.ast.add(Node::StringLiteral(sym), t.span))
            }
            TokenKind::BigInt => {
                self.next()?;
                let sym = Symbol(t.value.symbol().unwrap());
                Ok(self.ast.add(Node::BigIntLiteral(sym), t.span))
            }
            TokenKind::True | TokenKind::False => {
                self.next()?;
                Ok(self
                    .ast
                    .add(Node::BoolLiteral(t.kind == TokenKind::True), t.span))
            }
            TokenKind::Null => {
                self.next()?;
                Ok(self.ast.add(Node::NullLiteral, t.span))
            }
            TokenKind::This => {
                self.next()?;
                Ok(self.ast.add(Node::This, t.span))
            }
            k if is_identifier_like(k) => {
                let sym = self.ident_symbol(t)?;
                self.next()?;
                Ok(self.ast.add(Node::Identifier { sym }, t.span))
            }
            TokenKind::LParen => {
                self.next()?;
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_array_literal(),
            TokenKind::LBrace => self.parse_object_literal(),
            TokenKind::Function => {
                let function = self.parse_function(false)?;
                let span = self.ast.function(function).span;
                Ok(self.ast.add(Node::FunctionExpr { function }, span))
            }
            TokenKind::Class => self.parse_class(false),
            TokenKind::Super => self.parse_super(t.span),
            // `#x` alone is only meaningful as the left operand of `in`
            // (ES 13.3.9); the codegen rejects it anywhere else
            TokenKind::PrivateName => {
                let name_sym = Symbol(t.value.symbol().unwrap());
                let hidden = self.private_sym(name_sym);
                self.note_private_use(hidden, t.span)?;
                self.next()?;
                Ok(self.ast.add(Node::PrivateName { sym: hidden }, t.span))
            }
            _ => Err(ParseError::new(
                t.span,
                format!("unexpected token `{}`", kind_text(t)),
            )),
        }
    }

    fn parse_array_literal(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::LBracket)?.span.start;
        let mut elements = Vec::new();
        let end;
        loop {
            let t = self.peek()?;
            match t.kind {
                TokenKind::RBracket => {
                    end = self.next()?.span.end;
                    break;
                }
                TokenKind::Comma => {
                    let t = self.next()?;
                    elements.push(self.ast.add(Node::Hole, t.span));
                }
                TokenKind::Ellipsis => {
                    self.next()?;
                    let expr = self.parse_assignment()?;
                    let span = Span::new(t.span.start, self.ast.span(expr).end);
                    elements.push(self.ast.add(Node::Spread { expr }, span));
                    if self.eat(TokenKind::Comma)? {
                        continue;
                    }
                    end = self.expect(TokenKind::RBracket)?.span.end;
                    break;
                }
                _ => {
                    elements.push(self.parse_assignment()?);
                    if self.eat(TokenKind::Comma)? {
                        continue;
                    }
                    end = self.expect(TokenKind::RBracket)?.span.end;
                    break;
                }
            }
        }
        let elements = self.ast.list(&elements);
        Ok(self
            .ast
            .add(Node::ArrayLiteral { elements }, Span::new(start, end)))
    }

    fn parse_object_literal(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::LBrace)?.span.start;
        // a class-style scope: holds the super home-object slot when any
        // method of the literal uses `super` (the home object is the
        // literal itself, ES 15.4.2)
        let obj_scope = self.push_scope(ScopeKind::Class);
        self.class_stack.push(ClassCtx {
            entry_fn_depth: self.fn_stack.len(),
            uses_super: false,
            is_static: false,
            privates: Vec::new(),
            private_uses: Vec::new(),
        });
        let mut props = Vec::new();
        let end = loop {
            let t = self.peek()?;
            match t.kind {
                TokenKind::RBrace => break self.next()?.span.end,
                TokenKind::Ellipsis => {
                    self.next()?;
                    let expr = self.parse_assignment()?;
                    let span = Span::new(t.span.start, self.ast.span(expr).end);
                    props.push(self.ast.add(Node::Spread { expr }, span));
                }
                _ => {
                    props.push(self.parse_object_property()?);
                }
            }
            if self.eat(TokenKind::Comma)? {
                continue;
            }
            break self.expect(TokenKind::RBrace)?.span.end;
        };
        let ctx = self.class_stack.pop().unwrap();
        // `#name` references are never valid inside an object literal
        // (no private environment to declare them)
        if let Some((_, span)) = ctx.private_uses.first() {
            return Err(ParseError::new(
                *span,
                "private names are only valid inside a class",
            ));
        }
        if ctx.uses_super {
            let home = self.symbols_mut().intern(b".home_object");
            self.declare_class_slot(obj_scope, home, Span::new(start, start));
        }
        self.scopes.pop();
        let props = self.ast.list(&props);
        let node = self
            .ast
            .add(Node::ObjectLiteral { props }, Span::new(start, end));
        self.ast.set_node_scope(node, obj_scope);
        Ok(node)
    }

    /// One object literal entry: `k: v`, shorthand, method, accessor,
    /// computed key. Spread is handled by the caller.
    fn parse_object_property(&mut self) -> Result<NodeId, ParseError> {
        let t = self.peek()?;
        if let Some(is_get) = self.eat_accessor_prefix()? {
            let (key, key_sym, computed) = self.parse_property_key()?;
            if matches!(*self.ast.node(key), Node::PrivateName { .. }) {
                return Err(ParseError::new(
                    self.ast.span(key),
                    "private names are only valid in class bodies",
                ));
            }
            let kind = if is_get { PropKind::Get } else { PropKind::Set };
            let fn_name = self.accessor_name(key_sym, is_get);
            let (value, vspan) = self.parse_method_value(
                t.span.start,
                fn_name,
                kind,
                function_kind_for_property(kind),
                false,
                t.span,
            )?;
            return Ok(self.ast.add(
                Node::ObjectProperty {
                    key,
                    value,
                    kind,
                    computed,
                },
                Span::new(t.span.start, vspan.end),
            ));
        }
        let (key, shorthand, computed) = self.parse_property_key()?;
        let key_span = self.ast.span(key);
        if matches!(*self.ast.node(key), Node::PrivateName { .. }) {
            return Err(ParseError::new(
                key_span,
                "private names are only valid in class bodies",
            ));
        }
        if self.peek()?.kind == TokenKind::LParen {
            // method shorthand
            let (value, vspan) = self.parse_method_value(
                key_span.start,
                shorthand,
                PropKind::Method,
                FunctionKind::Method,
                false,
                key_span,
            )?;
            return Ok(self.ast.add(
                Node::ObjectProperty {
                    key,
                    value,
                    kind: PropKind::Method,
                    computed,
                },
                Span::new(key_span.start, vspan.end),
            ));
        }
        let value = if self.eat(TokenKind::Colon)? {
            self.parse_assignment()?
        } else {
            // shorthand: `{ a }` means `{ a: a }`; `{ a = 1 }` is a
            // CoverInitializedName — legal only as (part of) a pattern
            let Some(sym) = shorthand else {
                return Err(ParseError::new(
                    key_span,
                    "expected `:` after property name",
                ));
            };
            let value = self.ast.add(Node::Identifier { sym }, key_span);
            if self.eat(TokenKind::Assign)? {
                let default = self.parse_assignment()?;
                let span = Span::new(key_span.start, self.ast.span(default).end);
                let prop_id = self.ast.add(
                    Node::Assign {
                        op: TokenKind::Assign,
                        target: value,
                        value: default,
                    },
                    span,
                );
                // visible only through the pattern cover grammar; unless the
                // literal is rewritten into a pattern, this is a Syntax
                // Error (checked at statement end)
                self.cover_init.push(prop_id);
                prop_id
            } else {
                value
            }
        };
        let span = Span::new(key_span.start, self.ast.span(value).end);
        Ok(self.ast.add(
            Node::ObjectProperty {
                key,
                value,
                kind: PropKind::Init,
                computed,
            },
            span,
        ))
    }

    /// Detects `get`/`set` before a property key (both are plain identifiers,
    /// so `get: 1`, `{get}`, `{ get() {} }` stay valid). Returns Some(is_get)
    /// and consumes the word, or None.
    fn eat_accessor_prefix(&mut self) -> Result<Option<bool>, ParseError> {
        let t = self.peek()?;
        if t.kind != TokenKind::Identifier {
            return Ok(None);
        }
        let sym = Symbol(t.value.symbol().unwrap());
        let text = self.symbols().get(sym);
        let (is_get, is_set) = (text == b"get", text == b"set");
        if !is_get && !is_set {
            return Ok(None);
        }
        if !starts_property_key(self.peek_ahead()?.kind) {
            return Ok(None);
        }
        self.next()?;
        Ok(Some(is_get))
    }

    /// Parses `(params) { body }`, wraps it in a FunctionExpr node, and checks
    /// accessor arity. `force_strict`: class members are always strict.
    /// Returns the node and its span.

    /// ES 8.4.4 SetFunctionName prefixes: accessors are named "get x"/"set x".
    fn accessor_name(&mut self, key_sym: Option<Symbol>, is_get: bool) -> Option<Symbol> {
        let key = key_sym?;
        let text = self.symbols().get(key);
        let prefix = if is_get { b"get " } else { b"set " };
        let mut full = prefix.to_vec();
        full.extend_from_slice(text);
        Some(self.symbols_mut().intern(&full))
    }

    fn parse_method_value(
        &mut self,
        start: u32,
        name: Option<Symbol>,
        kind: PropKind,
        function_kind: FunctionKind,
        force_strict: bool,
        err_span: Span,
    ) -> Result<(NodeId, Span), ParseError> {
        let fid = self.parse_function_rest(
            start,
            name,
            FnFlags {
                force_strict,
                kind: function_kind,
                ..Default::default()
            },
        )?;
        if force_strict {
            self.ast.function_mut(fid).strict = true;
        }
        let nparams = self.ast.function(fid).params.len();
        if kind == PropKind::Get && nparams != 0 {
            return Err(ParseError::new(err_span, "getter must not have parameters"));
        }
        if kind == PropKind::Set && nparams != 1 {
            return Err(ParseError::new(
                err_span,
                "setter needs exactly one parameter",
            ));
        }
        let span = self.ast.function(fid).span;
        let value = self.ast.add(Node::FunctionExpr { function: fid }, span);
        Ok((value, span))
    }

    /// Returns (key node, name symbol if usable for shorthand/method name,
    /// computed flag).
    fn parse_property_key(&mut self) -> Result<(NodeId, Option<Symbol>, bool), ParseError> {
        let t = self.next()?;
        Ok(match t.kind {
            TokenKind::Identifier => {
                let sym = Symbol(t.value.symbol().unwrap());
                (
                    self.ast.add(Node::StringLiteral(sym), t.span),
                    Some(sym),
                    false,
                )
            }
            TokenKind::PrivateName => {
                let name = Symbol(t.value.symbol().unwrap());
                let hidden = self.private_sym(name);
                (
                    self.ast.add(Node::PrivateName { sym: hidden }, t.span),
                    None,
                    false,
                )
            }
            k if k.is_keyword() => {
                let sym = self.symbols_mut().intern(k.text().as_bytes());
                let shorthand = k.is_contextual().then_some(sym);
                (
                    self.ast.add(Node::StringLiteral(sym), t.span),
                    shorthand,
                    false,
                )
            }
            TokenKind::String => {
                let sym = Symbol(t.value.symbol().unwrap());
                (
                    self.ast.add(Node::StringLiteral(sym), t.span),
                    Some(sym),
                    false,
                )
            }
            TokenKind::Number => {
                let n = t.value.number().unwrap();
                (self.ast.add(Node::NumberLiteral(n), t.span), None, false)
            }
            TokenKind::LBracket => {
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RBracket)?;
                (expr, None, true)
            }
            _ => return Err(ParseError::new(t.span, "expected property name")),
        })
    }

    // -- misc -----------------------------------------------------------------

    fn check_assign_target(&self, node: NodeId) -> Result<(), ParseError> {
        match self.ast.node(node) {
            Node::Identifier { .. } | Node::Property { .. } | Node::SuperProperty { .. } => Ok(()),
            _ => Err(ParseError::new(
                self.ast.span(node),
                "invalid assignment target",
            )),
        }
    }
}

fn function_kind_for_property(kind: PropKind) -> FunctionKind {
    match kind {
        PropKind::Method => FunctionKind::Method,
        PropKind::Get => FunctionKind::Getter,
        PropKind::Set => FunctionKind::Setter,
        PropKind::Init | PropKind::Field => {
            unreachable!("data properties do not contain method functions")
        }
    }
}

/// Collect the (name, span) of every identifier bound by a binding pattern
/// (the target positions only — default expressions are references).
pub fn collect_pattern_names(ast: &Ast, id: NodeId, out: &mut Vec<(Symbol, Span)>) {
    match ast.node(id) {
        Node::Identifier { sym } => out.push((*sym, ast.span(id))),
        Node::ArrayPattern { elements } => {
            for &el in ast.list_items(*elements) {
                collect_pattern_names(ast, el, out);
            }
        }
        Node::ObjectPattern { props } => {
            for &p in ast.list_items(*props) {
                collect_pattern_names(ast, p, out);
            }
        }
        Node::PatternElement { target, .. } => collect_pattern_names(ast, *target, out),
        Node::PatternProperty { value, .. } => collect_pattern_names(ast, *value, out),
        Node::PatternRest { target } => collect_pattern_names(ast, *target, out),
        _ => {}
    }
}

fn kind_text(t: Token) -> String {
    match t.kind {
        TokenKind::Identifier => "identifier".to_string(),
        TokenKind::Number => "number".to_string(),
        TokenKind::String => "string".to_string(),
        TokenKind::Eof => "end of input".to_string(),
        k => format!("`{}`", k.text()),
    }
}
