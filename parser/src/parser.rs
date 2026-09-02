use crate::token::Span;
use crate::{Ast, Bookmark, CharStream, FunctionId, Scanner, SymbolTable};

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

pub struct Parser<S: CharStream> {
    scanner: Scanner<S>,
    ast: Ast,
    errors: Vec<ParseError>,
}

impl<S: CharStream> Parser<S> {
    pub fn new(stream: S) -> Self {
        Self {
            scanner: Scanner::new(stream),
            ast: Ast::new(),
            errors: Vec::new(),
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

    /// The top level parses as an implicit function.
    pub fn parse_script(&mut self) -> Result<FunctionId, ParseError> {
        todo!("grammar lands next")
    }

    /// Future lazy entry point: re-parse one function body from its start.
    pub fn parse_function_at(&mut self, start: u32) -> Result<FunctionId, ParseError> {
        self.scanner.seek_to(start);
        todo!("lazy re-parse entry point; needs scope summaries first")
    }
}
