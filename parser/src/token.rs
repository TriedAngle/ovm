use TokenInfo as I;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TokenKind {
    Eof,
    Illegal,

    Identifier,
    PrivateName, // #name
    Number,
    String,
    BigInt, // 123n

    LParen,      // (
    RParen,      // )
    LBrace,      // {
    RBrace,      // }
    LBracket,    // [
    RBracket,    // ]
    Semicolon,   // ;
    Comma,       // ,
    Period,      // .
    Ellipsis,    // ...
    Question,    // ?
    QuestionDot, // ?.
    Colon,       // :
    Arrow,       // =>

    PlusPlus,   // ++
    MinusMinus, // --

    // binary operators, low to high precedence
    Nullish,  // ??
    OrOr,     // ||
    AmpAmp,   // &&
    Pipe,     // |
    Caret,    // ^
    Amp,      // &
    EqEq,     // ==
    NotEq,    // !=
    EqEqEq,   // ===
    NotEqEq,  // !==
    Lt,       // <
    Gt,       // >
    LtEq,     // <=
    GtEq,     // >=
    Shl,      // <<
    Shr,      // >>
    Ushr,     // >>>
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    Percent,  // %
    StarStar, // **

    Bang,  // !
    Tilde, // ~

    Assign,         // =
    PlusAssign,     // +=
    MinusAssign,    // -=
    StarAssign,     // *=
    SlashAssign,    // /=
    PercentAssign,  // %=
    StarStarAssign, // **=
    ShlAssign,      // <<=
    ShrAssign,      // >>=
    UshrAssign,     // >>>=
    AmpAssign,      // &=
    PipeAssign,     // |=
    CaretAssign,    // ^=
    AmpAmpAssign,   // &&=
    OrOrAssign,     // ||=
    NullishAssign,  // ??=

    Break,
    Case,
    Catch,
    Class,
    Const,
    Continue,
    Debugger,
    Default,
    Delete,
    Do,
    Else,
    Enum,
    Export,
    Extends,
    False,
    Finally,
    For,
    Function,
    If,
    Import,
    In,
    Instanceof,
    New,
    Null,
    Return,
    Super,
    Switch,
    This,
    Throw,
    True,
    Try,
    Typeof,
    Var,
    Void,
    While,
    With,

    // strict-mode only
    Implements,
    Interface,
    Package,
    Private,
    Protected,
    Public,

    // contextual: parser decides if identifier use is allowed
    Async,
    Await,
    Let,
    Static,
    Yield,

    Count,
}

pub struct TokenInfo {
    pub text: &'static str,
    pub precedence: u8,
    pub flags: u8,
}

impl TokenInfo {
    pub const F_KEYWORD: u8 = 1 << 0;
    pub const F_RESERVED: u8 = 1 << 1;
    pub const F_STRICT_RESERVED: u8 = 1 << 2;
    pub const F_CONTEXTUAL: u8 = 1 << 3;
    pub const F_ASSIGNMENT: u8 = 1 << 4;

    const fn new(text: &'static str, precedence: u8, flags: u8) -> Self {
        Self {
            text,
            precedence,
            flags,
        }
    }
}

const TOKEN_INFO: &[TokenInfo] = &[
    I::new("", 0, 0),                                             // Eof
    I::new("", 0, 0),                                             // Illegal
    I::new("", 0, 0),                                             // Identifier
    I::new("", 0, 0),                                             // PrivateName
    I::new("", 0, 0),                                             // Number
    I::new("", 0, 0),                                             // String
    I::new("", 0, 0),                                             // BigInt
    I::new("(", 0, 0),                                            // LParen
    I::new(")", 0, 0),                                            // RParen
    I::new("{", 0, 0),                                            // LBrace
    I::new("}", 0, 0),                                            // RBrace
    I::new("[", 0, 0),                                            // LBracket
    I::new("]", 0, 0),                                            // RBracket
    I::new(";", 0, 0),                                            // Semicolon
    I::new(",", 0, 0),                                            // Comma
    I::new(".", 0, 0),                                            // Period
    I::new("...", 0, 0),                                          // Ellipsis
    I::new("?", 0, 0),                                            // Question
    I::new("?.", 0, 0),                                           // QuestionDot
    I::new(":", 0, 0),                                            // Colon
    I::new("=>", 0, 0),                                           // Arrow
    I::new("++", 0, 0),                                           // PlusPlus
    I::new("--", 0, 0),                                           // MinusMinus
    I::new("??", 1, 0),                                           // Nullish
    I::new("||", 2, 0),                                           // OrOr
    I::new("&&", 3, 0),                                           // AmpAmp
    I::new("|", 4, 0),                                            // Pipe
    I::new("^", 5, 0),                                            // Caret
    I::new("&", 6, 0),                                            // Amp
    I::new("==", 7, 0),                                           // EqEq
    I::new("!=", 7, 0),                                           // NotEq
    I::new("===", 7, 0),                                          // EqEqEq
    I::new("!==", 7, 0),                                          // NotEqEq
    I::new("<", 8, 0),                                            // Lt
    I::new(">", 8, 0),                                            // Gt
    I::new("<=", 8, 0),                                           // LtEq
    I::new(">=", 8, 0),                                           // GtEq
    I::new("<<", 9, 0),                                           // Shl
    I::new(">>", 9, 0),                                           // Shr
    I::new(">>>", 9, 0),                                          // Ushr
    I::new("+", 10, 0),                                           // Plus
    I::new("-", 10, 0),                                           // Minus
    I::new("*", 11, 0),                                           // Star
    I::new("/", 11, 0),                                           // Slash
    I::new("%", 11, 0),                                           // Percent
    I::new("**", 12, 0),                                          // StarStar
    I::new("!", 0, 0),                                            // Bang
    I::new("~", 0, 0),                                            // Tilde
    I::new("=", 0, I::F_ASSIGNMENT),                              // Assign
    I::new("+=", 0, I::F_ASSIGNMENT),                             // PlusAssign
    I::new("-=", 0, I::F_ASSIGNMENT),                             // MinusAssign
    I::new("*=", 0, I::F_ASSIGNMENT),                             // StarAssign
    I::new("/=", 0, I::F_ASSIGNMENT),                             // SlashAssign
    I::new("%=", 0, I::F_ASSIGNMENT),                             // PercentAssign
    I::new("**=", 0, I::F_ASSIGNMENT),                            // StarStarAssign
    I::new("<<=", 0, I::F_ASSIGNMENT),                            // ShlAssign
    I::new(">>=", 0, I::F_ASSIGNMENT),                            // ShrAssign
    I::new(">>>=", 0, I::F_ASSIGNMENT),                           // UshrAssign
    I::new("&=", 0, I::F_ASSIGNMENT),                             // AmpAssign
    I::new("|=", 0, I::F_ASSIGNMENT),                             // PipeAssign
    I::new("^=", 0, I::F_ASSIGNMENT),                             // CaretAssign
    I::new("&&=", 0, I::F_ASSIGNMENT),                            // AmpAmpAssign
    I::new("||=", 0, I::F_ASSIGNMENT),                            // OrOrAssign
    I::new("??=", 0, I::F_ASSIGNMENT),                            // NullishAssign
    I::new("break", 0, I::F_KEYWORD | I::F_RESERVED),             // Break
    I::new("case", 0, I::F_KEYWORD | I::F_RESERVED),              // Case
    I::new("catch", 0, I::F_KEYWORD | I::F_RESERVED),             // Catch
    I::new("class", 0, I::F_KEYWORD | I::F_RESERVED),             // Class
    I::new("const", 0, I::F_KEYWORD | I::F_RESERVED),             // Const
    I::new("continue", 0, I::F_KEYWORD | I::F_RESERVED),          // Continue
    I::new("debugger", 0, I::F_KEYWORD | I::F_RESERVED),          // Debugger
    I::new("default", 0, I::F_KEYWORD | I::F_RESERVED),           // Default
    I::new("delete", 0, I::F_KEYWORD | I::F_RESERVED),            // Delete
    I::new("do", 0, I::F_KEYWORD | I::F_RESERVED),                // Do
    I::new("else", 0, I::F_KEYWORD | I::F_RESERVED),              // Else
    I::new("enum", 0, I::F_KEYWORD | I::F_RESERVED),              // Enum
    I::new("export", 0, I::F_KEYWORD | I::F_RESERVED),            // Export
    I::new("extends", 0, I::F_KEYWORD | I::F_RESERVED),           // Extends
    I::new("false", 0, I::F_KEYWORD | I::F_RESERVED),             // False
    I::new("finally", 0, I::F_KEYWORD | I::F_RESERVED),           // Finally
    I::new("for", 0, I::F_KEYWORD | I::F_RESERVED),               // For
    I::new("function", 0, I::F_KEYWORD | I::F_RESERVED),          // Function
    I::new("if", 0, I::F_KEYWORD | I::F_RESERVED),                // If
    I::new("import", 0, I::F_KEYWORD | I::F_RESERVED),            // Import
    I::new("in", 8, I::F_KEYWORD | I::F_RESERVED),                // In
    I::new("instanceof", 8, I::F_KEYWORD | I::F_RESERVED),        // Instanceof
    I::new("new", 0, I::F_KEYWORD | I::F_RESERVED),               // New
    I::new("null", 0, I::F_KEYWORD | I::F_RESERVED),              // Null
    I::new("return", 0, I::F_KEYWORD | I::F_RESERVED),            // Return
    I::new("super", 0, I::F_KEYWORD | I::F_RESERVED),             // Super
    I::new("switch", 0, I::F_KEYWORD | I::F_RESERVED),            // Switch
    I::new("this", 0, I::F_KEYWORD | I::F_RESERVED),              // This
    I::new("throw", 0, I::F_KEYWORD | I::F_RESERVED),             // Throw
    I::new("true", 0, I::F_KEYWORD | I::F_RESERVED),              // True
    I::new("try", 0, I::F_KEYWORD | I::F_RESERVED),               // Try
    I::new("typeof", 0, I::F_KEYWORD | I::F_RESERVED),            // Typeof
    I::new("var", 0, I::F_KEYWORD | I::F_RESERVED),               // Var
    I::new("void", 0, I::F_KEYWORD | I::F_RESERVED),              // Void
    I::new("while", 0, I::F_KEYWORD | I::F_RESERVED),             // While
    I::new("with", 0, I::F_KEYWORD | I::F_RESERVED),              // With
    I::new("implements", 0, I::F_KEYWORD | I::F_STRICT_RESERVED), // Implements
    I::new("interface", 0, I::F_KEYWORD | I::F_STRICT_RESERVED),  // Interface
    I::new("package", 0, I::F_KEYWORD | I::F_STRICT_RESERVED),    // Package
    I::new("private", 0, I::F_KEYWORD | I::F_STRICT_RESERVED),    // Private
    I::new("protected", 0, I::F_KEYWORD | I::F_STRICT_RESERVED),  // Protected
    I::new("public", 0, I::F_KEYWORD | I::F_STRICT_RESERVED),     // Public
    I::new("async", 0, I::F_KEYWORD | I::F_CONTEXTUAL),           // Async
    I::new("await", 0, I::F_KEYWORD | I::F_CONTEXTUAL),           // Await
    I::new("let", 0, I::F_KEYWORD | I::F_CONTEXTUAL),             // Let
    I::new("static", 0, I::F_KEYWORD | I::F_CONTEXTUAL),          // Static
    I::new("yield", 0, I::F_KEYWORD | I::F_CONTEXTUAL),           // Yield
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

    pub fn is_assignment(self) -> bool {
        self.info().flags & TokenInfo::F_ASSIGNMENT != 0
    }

    pub fn is_keyword(self) -> bool {
        self.info().flags & TokenInfo::F_KEYWORD != 0
    }

    pub fn is_reserved(self) -> bool {
        self.info().flags & TokenInfo::F_RESERVED != 0
    }

    pub fn is_strict_reserved(self) -> bool {
        self.info().flags & TokenInfo::F_STRICT_RESERVED != 0
    }

    pub fn is_contextual(self) -> bool {
        self.info().flags & TokenInfo::F_CONTEXTUAL != 0
    }

    pub fn is_right_associative(self) -> bool {
        self == TokenKind::StarStar || self.is_assignment()
    }

    pub fn is_unary_op(self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Bang | Tilde
                | Plus
                | Minus
                | PlusPlus
                | MinusMinus
                | Typeof
                | Void
                | Delete
                | Await
                | Yield
        )
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
    /// A line terminator preceded this token (ASI, restricted productions).
    pub after_newline: bool,
    pub value: TokenValue,
    pub span: Span,
}

impl Token {
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self {
            kind,
            after_newline: false,
            value: TokenValue::None,
            span,
        }
    }
}
