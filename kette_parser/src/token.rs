pub use parser_utils::ByteSpan;

use TokenInfo as I;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TokenKind {
    #[default]
    Eof,
    Illegal,

    Identifier,
    Number,
    String,

    // punctuation
    LParen,    // (
    RParen,    // )
    LBrace,    // {
    RBrace,    // }
    LBracket,  // [
    RBracket,  // ]
    Comma,     // ,
    Period,    // .
    Colon,     // :
    Semicolon, // ;
    Star,      // * (multiply or parent-slot marker)
    Arrow,     // ->
    Caret,     // ^ (non-local return)
    Assign,    // =
    Pipe,      // | (lambda parameter delimiter)

    // binary operators, low to high precedence
    OrOr,    // ||
    AmpAmp,  // &&
    EqEq,    // ==
    NotEq,   // !=
    Lt,      // <
    Gt,      // >
    LtEq,    // <=
    GtEq,    // >=
    Plus,    // +
    Minus,   // -
    Slash,   // /
    Percent, // %

    Bang, // ! (unary not)

    // keywords
    Let,
    If,
    Else,
    While,
    For,
    In,
    Match,
    SelfKw,
    Null,
    True,
    False,
    Return,
    Try,
    Catch,

    Count,
}

pub struct TokenInfo {
    pub text: &'static str,
    pub precedence: u8,
    pub flags: u8,
}

impl TokenInfo {
    pub const F_KEYWORD: u8 = 1 << 0;

    const fn new(text: &'static str, precedence: u8, flags: u8) -> Self {
        Self {
            text,
            precedence,
            flags,
        }
    }
}

const TOKEN_INFO: &[TokenInfo] = &[
    I::new("", 0, 0),                  // Eof
    I::new("", 0, 0),                  // Illegal
    I::new("", 0, 0),                  // Identifier
    I::new("", 0, 0),                  // Number
    I::new("", 0, 0),                  // String
    I::new("(", 0, 0),                 // LParen
    I::new(")", 0, 0),                 // RParen
    I::new("{", 0, 0),                 // LBrace
    I::new("}", 0, 0),                 // RBrace
    I::new("[", 0, 0),                 // LBracket
    I::new("]", 0, 0),                 // RBracket
    I::new(",", 0, 0),                 // Comma
    I::new(".", 0, 0),                 // Period
    I::new(":", 0, 0),                 // Colon
    I::new(";", 0, 0),                 // Semicolon
    I::new("*", 6, 0),                 // Star
    I::new("->", 0, 0),                // Arrow
    I::new("^", 0, 0),                 // Caret
    I::new("=", 0, 0),                 // Assign
    I::new("|", 0, 0),                 // Pipe
    I::new("||", 1, 0),                // OrOr
    I::new("&&", 2, 0),                // AmpAmp
    I::new("==", 3, 0),                // EqEq
    I::new("!=", 3, 0),                // NotEq
    I::new("<", 4, 0),                 // Lt
    I::new(">", 4, 0),                 // Gt
    I::new("<=", 4, 0),                // LtEq
    I::new(">=", 4, 0),                // GtEq
    I::new("+", 5, 0),                 // Plus
    I::new("-", 5, 0),                 // Minus
    I::new("/", 6, 0),                 // Slash
    I::new("%", 6, 0),                 // Percent
    I::new("!", 0, 0),                 // Bang
    I::new("let", 0, I::F_KEYWORD),    // Let
    I::new("if", 0, I::F_KEYWORD),     // If
    I::new("else", 0, I::F_KEYWORD),   // Else
    I::new("while", 0, I::F_KEYWORD),  // While
    I::new("for", 0, I::F_KEYWORD),    // For
    I::new("in", 0, I::F_KEYWORD),     // In
    I::new("match", 0, I::F_KEYWORD),  // Match
    I::new("self", 0, I::F_KEYWORD),   // SelfKw
    I::new("null", 0, I::F_KEYWORD),   // Null
    I::new("true", 0, I::F_KEYWORD),   // True
    I::new("false", 0, I::F_KEYWORD),  // False
    I::new("return", 0, I::F_KEYWORD), // Return
    I::new("try", 0, I::F_KEYWORD),    // Try
    I::new("catch", 0, I::F_KEYWORD),  // Catch
];

const _: () = assert!(
    TOKEN_INFO.len() == TokenKind::Count as usize,
    "TOKEN_INFO must have exactly one row per TokenKind variant"
);

impl TokenKind {
    #[inline]
    fn info(self) -> &'static TokenInfo {
        debug_assert!(self != TokenKind::Count);
        &TOKEN_INFO[self as usize]
    }

    pub fn text(self) -> &'static str {
        self.info().text
    }

    pub fn precedence(self) -> u8 {
        self.info().precedence
    }

    pub fn is_binary_op(self) -> bool {
        self.precedence() != 0
    }

    pub fn is_keyword(self) -> bool {
        self.info().flags & TokenInfo::F_KEYWORD != 0
    }

    /// Human-readable form for diagnostics.
    pub fn describe(self) -> &'static str {
        let text = self.text();
        if !text.is_empty() {
            return text;
        }
        match self {
            TokenKind::Eof => "end of input",
            TokenKind::Identifier => "identifier",
            TokenKind::Number => "number",
            TokenKind::String => "string",
            _ => "token",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TokenValue {
    None,
    Number(f64),
    Symbol(u32),
}

impl TokenValue {
    pub fn symbol(self) -> Option<u32> {
        match self {
            TokenValue::Symbol(s) => Some(s),
            _ => None,
        }
    }

    pub fn number(self) -> Option<f64> {
        match self {
            TokenValue::Number(n) => Some(n),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub kind: TokenKind,
    /// A line terminator preceded this token (statement/slot separation).
    pub after_newline: bool,
    pub value: TokenValue,
    pub span: ByteSpan,
}

impl Token {
    pub const fn new(kind: TokenKind, span: ByteSpan) -> Self {
        Self {
            kind,
            after_newline: false,
            value: TokenValue::None,
            span,
        }
    }
}
