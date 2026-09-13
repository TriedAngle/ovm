use crate::{CharStream, ParseError, Span, SymbolTable, Token, TokenKind, TokenValue};

pub type ScanResult = Result<Token, ParseError>;

const NL: u32 = 0x0A; // \n
const CR: u32 = 0x0D; // \r
const LS: u32 = 0x2028;
const PS: u32 = 0x2029;
const BACKSLASH: u32 = 0x5C;

fn is_newline(c: u32) -> bool {
    matches!(c, NL | CR | LS | PS)
}

fn is_ident_start(c: u32) -> bool {
    c == b'_' as u32
        || c == b'$' as u32
        || c < 128 && (c as u8 as char).is_ascii_alphabetic()
        || c >= 0x80 && char::from_u32(c).is_some_and(|ch| ch.is_alphabetic())
}

fn is_ident_continue(c: u32) -> bool {
    is_ident_start(c)
        || c < 128 && (c as u8 as char).is_ascii_digit()
        || c >= 0x80 && char::from_u32(c).is_some_and(|ch| ch.is_alphanumeric())
}

fn keyword_kind(text: &[u8]) -> Option<TokenKind> {
    use TokenKind::*;
    Some(match text {
        b"break" => Break,
        b"case" => Case,
        b"catch" => Catch,
        b"class" => Class,
        b"const" => Const,
        b"continue" => Continue,
        b"debugger" => Debugger,
        b"default" => Default,
        b"delete" => Delete,
        b"do" => Do,
        b"else" => Else,
        b"enum" => Enum,
        b"export" => Export,
        b"extends" => Extends,
        b"false" => False,
        b"finally" => Finally,
        b"for" => For,
        b"function" => Function,
        b"if" => If,
        b"import" => Import,
        b"in" => In,
        b"instanceof" => Instanceof,
        b"new" => New,
        b"null" => Null,
        b"return" => Return,
        b"super" => Super,
        b"switch" => Switch,
        b"this" => This,
        b"throw" => Throw,
        b"true" => True,
        b"try" => Try,
        b"typeof" => Typeof,
        b"var" => Var,
        b"void" => Void,
        b"while" => While,
        b"with" => With,
        b"implements" => Implements,
        b"interface" => Interface,
        b"package" => Package,
        b"private" => Private,
        b"protected" => Protected,
        b"public" => Public,
        b"async" => Async,
        b"await" => Await,
        b"let" => Let,
        b"static" => Static,
        b"yield" => Yield,
        _ => return None,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Action {
    Illegal,
    IdentStart,
    Digit,
    Quote,
    /// single-char token from the table
    Punct(TokenKind),
    /// multi-char handling in `scan_special`
    Special,
    /// `#` private name: `#` + identifier continuation
    Hash,
}

const fn first_char_table() -> [Action; 128] {
    use Action::*;
    let mut t = [Illegal; 128];

    let mut c = b'a';
    while c <= b'z' {
        t[c as usize] = IdentStart;
        c += 1;
    }
    let mut c = b'A';
    while c <= b'Z' {
        t[c as usize] = IdentStart;
        c += 1;
    }
    t[b'_' as usize] = IdentStart;
    t[b'$' as usize] = IdentStart;

    let mut c = b'0';
    while c <= b'9' {
        t[c as usize] = Digit;
        c += 1;
    }

    t[b'\'' as usize] = Quote;
    t[b'"' as usize] = Quote;

    t[b'(' as usize] = Punct(TokenKind::LParen);
    t[b')' as usize] = Punct(TokenKind::RParen);
    t[b'{' as usize] = Punct(TokenKind::LBrace);
    t[b'}' as usize] = Punct(TokenKind::RBrace);
    t[b'[' as usize] = Punct(TokenKind::LBracket);
    t[b']' as usize] = Punct(TokenKind::RBracket);
    t[b';' as usize] = Punct(TokenKind::Semicolon);
    t[b',' as usize] = Punct(TokenKind::Comma);
    t[b':' as usize] = Punct(TokenKind::Colon);
    t[b'~' as usize] = Punct(TokenKind::Tilde);

    t[b'=' as usize] = Special;
    t[b'!' as usize] = Special;
    t[b'<' as usize] = Special;
    t[b'>' as usize] = Special;
    t[b'+' as usize] = Special;
    t[b'-' as usize] = Special;
    t[b'*' as usize] = Special;
    t[b'/' as usize] = Special;
    t[b'%' as usize] = Special;
    t[b'&' as usize] = Special;
    t[b'|' as usize] = Special;
    t[b'^' as usize] = Special;
    t[b'?' as usize] = Special;
    t[b'.' as usize] = Special;
    t[b'#' as usize] = Hash;

    t
}

const FIRST_CHAR: [Action; 128] = first_char_table();

/// Longest-match suffixes for multi-char operators: for the consumed first
/// char, the candidate suffixes (longest first) and the fallback kind.
/// `?` and `.` have extra digit rules and stay hand-written in scan_special.
type Suffixes = (&'static [(&'static str, TokenKind)], TokenKind);

fn special_ops(c: u8) -> Option<Suffixes> {
    use TokenKind::*;
    Some(match c {
        b'=' => (&[("==", EqEqEq), ("=", EqEq), (">", Arrow)], Assign),
        b'!' => (&[("==", NotEqEq), ("=", NotEq)], Bang),
        b'<' => (&[("<=", ShlAssign), ("<", Shl), ("=", LtEq)], Lt),
        b'>' => (
            &[
                (">>=", UshrAssign),
                (">>", Ushr),
                (">=", ShrAssign),
                (">", Shr),
                ("=", GtEq),
            ],
            Gt,
        ),
        b'+' => (&[("+", PlusPlus), ("=", PlusAssign)], Plus),
        b'-' => (&[("-", MinusMinus), ("=", MinusAssign)], Minus),
        b'*' => (
            &[("*=", StarStarAssign), ("*", StarStar), ("=", StarAssign)],
            Star,
        ),
        b'/' => (&[("=", SlashAssign)], Slash),
        b'%' => (&[("=", PercentAssign)], Percent),
        b'&' => (
            &[("&=", AmpAmpAssign), ("&", AmpAmp), ("=", AmpAssign)],
            Amp,
        ),
        b'|' => (&[("|=", OrOrAssign), ("|", OrOr), ("=", PipeAssign)], Pipe),
        b'^' => (&[("=", CaretAssign)], Caret),
        _ => return None,
    })
}

/// Saved scanner state for backwards rewinds (speculative parses, lazy
/// re-parse). Only valid within already-consumed input.
#[derive(Clone)]
pub struct Bookmark {
    pos: u32,
    ring: [Option<ScanResult>; 2],
    newline_seen: bool,
}

pub struct Scanner<S: CharStream> {
    stream: S,
    /// parse-local string interner; handed to the Ast when parsing finishes
    symbols: SymbolTable,
    /// [current, next-ahead]
    ring: [Option<ScanResult>; 2],
    /// line terminator crossed since the last produced token (ASI input)
    newline_seen: bool,
    /// sticky first error; scanning never recovers
    failed: Option<ParseError>,
    /// decode buffer for string literals with escapes / span copies
    scratch: Vec<u8>,
}

impl<S: CharStream> Scanner<S> {
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            symbols: SymbolTable::default(),
            ring: [None, None],
            newline_seen: false,
            failed: None,
            scratch: Vec::new(),
        }
    }

    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    pub fn symbols_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbols
    }

    /// Take the table out (e.g. to move it into the finished Ast).
    pub fn take_symbols(&mut self) -> SymbolTable {
        std::mem::take(&mut self.symbols)
    }

    /// Current token, scanning on demand. Idempotent.
    pub fn peek(&mut self) -> &ScanResult {
        self.fill(0);
        self.ring[0].as_ref().unwrap()
    }

    /// The token after the current one. Does not consume anything.
    pub fn peek_ahead(&mut self) -> &ScanResult {
        self.fill(1);
        self.ring[1].as_ref().unwrap()
    }

    /// Consume and return the current token.
    pub fn next_token(&mut self) -> ScanResult {
        self.fill(0);
        let t = self.ring[0].take().unwrap();
        self.ring[0] = self.ring[1].take();
        t
    }

    fn fill(&mut self, n: usize) {
        if self.ring[0].is_none() {
            self.ring[0] = Some(self.scan_token());
        }
        if n == 1 && self.ring[1].is_none() {
            self.ring[1] = Some(self.scan_token());
        }
    }

    pub fn bookmark(&self) -> Bookmark {
        Bookmark {
            pos: self.stream.pos(),
            ring: self.ring.clone(),
            newline_seen: self.newline_seen,
        }
    }

    pub fn restore(&mut self, bookmark: Bookmark) {
        self.stream.seek(bookmark.pos);
        self.ring = bookmark.ring;
        self.newline_seen = bookmark.newline_seen;
    }

    /// Rewind to a raw position, dropping all scanned state.
    pub fn seek_to(&mut self, pos: u32) {
        self.stream.seek(pos);
        self.ring = [None, None];
        self.newline_seen = false;
    }

    // -- token scanning -----------------------------------------------------

    fn scan_token(&mut self) -> ScanResult {
        if let Some(e) = &self.failed {
            return Err(e.clone());
        }
        match self.scan_token_inner() {
            Err(e) => {
                self.failed = Some(e.clone());
                Err(e)
            }
            ok => ok,
        }
    }

    fn scan_token_inner(&mut self) -> ScanResult {
        self.skip_trivia()?;
        let start = self.stream.pos();
        let after_newline = std::mem::take(&mut self.newline_seen);

        let Some(c) = self.stream.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                after_newline,
                value: TokenValue::None,
                span: Span::new(start, start),
            });
        };

        let mut tok = if c < 128 {
            match FIRST_CHAR[c as usize] {
                Action::Illegal => {
                    return Err(ParseError::new(
                        Span::new(start, start + 1),
                        "unexpected character",
                    ));
                }
                Action::IdentStart => self.scan_identifier(start)?,
                Action::Digit => self.scan_number(start)?,
                Action::Quote => self.scan_string(c, start)?,
                Action::Punct(kind) => {
                    self.stream.advance();
                    Token::new(kind, Span::new(start, self.stream.pos()))
                }
                Action::Special => self.scan_special(start)?,
                Action::Hash => self.scan_private_name(start)?,
            }
        } else if is_ident_start(c) {
            self.scan_identifier(start)?
        } else {
            return Err(ParseError::new(
                Span::new(start, start + 1),
                "unexpected non-ascii character",
            ));
        };
        tok.after_newline = after_newline;
        Ok(tok)
    }

    fn skip_trivia(&mut self) -> Result<(), ParseError> {
        loop {
            let Some(c) = self.stream.peek() else {
                return Ok(());
            };
            match c {
                0x09 | 0x0B | 0x0C | 0x20 => self.stream.advance(),
                NL | CR | LS | PS => {
                    self.newline_seen = true;
                    self.stream.advance();
                }
                c if c == b'/' as u32 => {
                    let save = self.stream.pos();
                    self.stream.advance();
                    match self.stream.peek() {
                        Some(c) if c == b'/' as u32 => {
                            while let Some(c) = self.stream.peek() {
                                if is_newline(c) {
                                    break;
                                }
                                self.stream.advance();
                            }
                        }
                        Some(c) if c == b'*' as u32 => {
                            self.stream.advance();
                            loop {
                                match self.stream.peek() {
                                    None => {
                                        return Err(ParseError::new(
                                            Span::new(save, self.stream.pos()),
                                            "unterminated block comment",
                                        ));
                                    }
                                    Some(c) if is_newline(c) => {
                                        self.newline_seen = true;
                                        self.stream.advance();
                                    }
                                    Some(c) if c == b'*' as u32 => {
                                        self.stream.advance();
                                        if self.stream.peek() == Some(b'/' as u32) {
                                            self.stream.advance();
                                            break;
                                        }
                                    }
                                    _ => self.stream.advance(),
                                }
                            }
                        }
                        // division, not a comment: un-consume the '/'
                        _ => {
                            self.stream.seek(save);
                            return Ok(());
                        }
                    }
                }
                c if c >= 0x80 => {
                    let ch = char::from_u32(c).unwrap();
                    if ch == '\u{FEFF}' || ch.is_whitespace() {
                        self.stream.advance();
                    } else {
                        return Ok(());
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn scan_identifier(&mut self, start: u32) -> ScanResult {
        while let Some(c) = self.stream.peek() {
            if !is_ident_continue(c) {
                break;
            }
            self.stream.advance();
        }
        let span = Span::new(start, self.stream.pos());
        Ok(
            match Self::with_span_bytes(&mut self.stream, &mut self.scratch, span, keyword_kind) {
                Some(kind) => Token::new(kind, span),
                None => Token {
                    kind: TokenKind::Identifier,
                    after_newline: false,
                    value: TokenValue::Symbol(self.intern_span(span).0),
                    span,
                },
            },
        )
    }

    /// `#name`: a PrivateName token whose symbol excludes the `#` (ES 12.9).
    fn scan_private_name(&mut self, start: u32) -> ScanResult {
        self.stream.advance(); // #
        if !self.stream.peek().is_some_and(is_ident_start) {
            return Err(ParseError::new(
                Span::new(start, self.stream.pos() + 1),
                "invalid character in private name",
            ));
        }
        while let Some(c) = self.stream.peek() {
            if !is_ident_continue(c) {
                break;
            }
            self.stream.advance();
        }
        let span = Span::new(start + 1, self.stream.pos());
        let sym = self.intern_span(span);
        Ok(Token {
            kind: TokenKind::PrivateName,
            after_newline: false,
            value: TokenValue::Symbol(sym.0),
            span: Span::new(start, self.stream.pos()),
        })
    }

    fn scan_number(&mut self, start: u32) -> ScanResult {
        // radix literals: 0x.. 0b.. 0o.. (optional `n` suffix for BigInt)
        if self.stream.peek() == Some(b'0' as u32) {
            let save = self.stream.pos();
            self.stream.advance();
            let radix = match self.stream.peek() {
                Some(c) if c == b'x' as u32 || c == b'X' as u32 => Some(16),
                Some(c) if c == b'b' as u32 || c == b'B' as u32 => Some(2),
                Some(c) if c == b'o' as u32 || c == b'O' as u32 => Some(8),
                _ => None,
            };
            if let Some(radix) = radix {
                self.stream.advance();
                return self.scan_radix_number(start, radix);
            }
            self.stream.seek(save);
        }
        let started_with_dot = self.stream.peek() == Some(b'.' as u32);
        if started_with_dot {
            self.stream.advance();
            self.consume_digits();
        } else {
            self.consume_digits();
            // BigInt: integer digits followed by `n` (no fraction/exponent)
            if self.stream.peek() == Some(b'n' as u32) {
                self.stream.advance();
                let span = Span::new(start, self.stream.pos());
                // intern the digits without the trailing `n`
                let digits = Span::new(start, span.end - 1);
                let sym = self.intern_span(digits);
                return Ok(Token {
                    kind: TokenKind::BigInt,
                    after_newline: false,
                    value: TokenValue::Symbol(sym.0),
                    span,
                });
            }
            if self.stream.peek() == Some(b'.' as u32) {
                self.stream.advance();
                self.consume_digits();
            }
        }
        // exponent: only consumed if digits actually follow
        if matches!(self.stream.peek(), Some(c) if c == b'e' as u32 || c == b'E' as u32) {
            let save = self.stream.pos();
            self.stream.advance();
            if matches!(self.stream.peek(), Some(c) if c == b'+' as u32 || c == b'-' as u32) {
                self.stream.advance();
            }
            if matches!(self.stream.peek(), Some(c) if c < 128 && (c as u8).is_ascii_digit()) {
                self.consume_digits();
            } else {
                self.stream.seek(save);
            }
        }
        let span = Span::new(start, self.stream.pos());
        let value = Self::with_span_bytes(&mut self.stream, &mut self.scratch, span, |b| {
            std::str::from_utf8(b)
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
        });
        match value {
            Some(n) => Ok(Token {
                kind: TokenKind::Number,
                after_newline: false,
                value: TokenValue::Number(n),
                span,
            }),
            None => Err(ParseError::new(span, "invalid number literal")),
        }
    }

    fn consume_digits(&mut self) {
        while matches!(self.stream.peek(), Some(c) if c < 128 && (c as u8).is_ascii_digit()) {
            self.stream.advance();
        }
    }

    /// After `0x`/`0b`/`0o`: scan radix digits, optional `n` for BigInt.
    fn scan_radix_number(&mut self, start: u32, radix: u32) -> ScanResult {
        let digits_start = self.stream.pos();
        while let Some(c) = self.stream.peek() {
            match hex_digit(c) {
                Some(d) if d < radix => {
                    self.stream.advance();
                }
                // a valid digit of a lower radix is still a syntax error:
                // `0b2`, `0o9`
                Some(_) => {
                    return Err(ParseError::new(
                        Span::new(self.stream.pos(), self.stream.pos() + 1),
                        "invalid digit in radix literal",
                    ));
                }
                None => break,
            }
        }
        let digits_end = self.stream.pos();
        if digits_end == digits_start {
            return Err(ParseError::new(
                Span::new(start, digits_end),
                "expected digits after radix prefix",
            ));
        }
        let span = Span::new(start, digits_end);
        if self.stream.peek() == Some(b'n' as u32) {
            self.stream.advance();
            // intern the full literal text minus the `n` (radix via prefix)
            let sym = self.intern_span(span);
            return Ok(Token {
                kind: TokenKind::BigInt,
                after_newline: false,
                value: TokenValue::Symbol(sym.0),
                span: Span::new(start, digits_end + 1),
            });
        }
        // a letter or digit right after the literal is an error: `0x1g`
        if matches!(self.stream.peek(), Some(c) if is_ident_continue(c)) {
            return Err(ParseError::new(
                Span::new(self.stream.pos(), self.stream.pos() + 1),
                "unexpected character after number literal",
            ));
        }
        let value = Self::with_span_bytes(
            &mut self.stream,
            &mut self.scratch,
            Span::new(digits_start, digits_end),
            |b| {
                b.iter().fold(0f64, |v, &c| {
                    v * radix as f64 + (c as char).to_digit(16).unwrap() as f64
                })
            },
        );
        Ok(Token {
            kind: TokenKind::Number,
            after_newline: false,
            value: TokenValue::Number(value),
            span,
        })
    }

    /// String literal. Contents are WTF-8: raw source chars pass through,
    /// escapes are encoded (lone surrogates as the 3-byte pattern).
    fn scan_string(&mut self, quote: u32, start: u32) -> ScanResult {
        self.stream.advance(); // opening quote
        let content_start = self.stream.pos();
        let mut decoded: Option<Vec<u8>> = None;
        loop {
            let Some(c) = self.stream.peek() else {
                return Err(ParseError::new(
                    Span::new(start, self.stream.pos()),
                    "unterminated string literal",
                ));
            };
            if c == quote {
                let content_end = self.stream.pos();
                self.stream.advance();
                let span = Span::new(start, self.stream.pos());
                let sym = match decoded {
                    Some(buf) => self.symbols.intern(&buf),
                    None => self.intern_span(Span::new(content_start, content_end)),
                };
                return Ok(Token {
                    kind: TokenKind::String,
                    after_newline: false,
                    value: TokenValue::Symbol(sym.0),
                    span,
                });
            }
            if is_newline(c) {
                return Err(ParseError::new(
                    Span::new(start, self.stream.pos()),
                    "unterminated string literal",
                ));
            }
            if c == BACKSLASH {
                self.ensure_decoded(&mut decoded, content_start);
                self.stream.advance();
                self.scan_escape(decoded.as_mut().unwrap())?;
                continue;
            }
            if c > 0x7F || decoded.is_some() {
                self.ensure_decoded(&mut decoded, content_start);
                push_wtf8(decoded.as_mut().unwrap(), c);
            }
            self.stream.advance();
        }
    }

    /// Switch from the zero-copy span path to the decode buffer: copy the
    /// string content scanned so far into `decoded` (once).
    fn ensure_decoded(&mut self, decoded: &mut Option<Vec<u8>>, content_start: u32) {
        if decoded.is_none() {
            let mut v = Vec::new();
            self.copy_span(Span::new(content_start, self.stream.pos()), &mut v);
            *decoded = Some(v);
        }
    }

    fn scan_escape(&mut self, out: &mut Vec<u8>) -> Result<(), ParseError> {
        let pos = self.stream.pos();
        let Some(c) = self.stream.peek() else {
            return Err(ParseError::new(
                Span::new(pos, pos),
                "unterminated string escape",
            ));
        };
        match c {
            c if c == b'n' as u32 => out.push(b'\n'),
            c if c == b't' as u32 => out.push(b'\t'),
            c if c == b'r' as u32 => out.push(b'\r'),
            c if c == b'b' as u32 => out.push(0x08),
            c if c == b'f' as u32 => out.push(0x0C),
            c if c == b'v' as u32 => out.push(0x0B),
            c if c == b'0' as u32 => out.push(0),
            c if c == b'x' as u32 => {
                self.stream.advance();
                let v = self.scan_hex(2)?;
                out.push(v as u8);
                return Ok(());
            }
            c if c == b'u' as u32 => {
                self.stream.advance();
                let v = if self.stream.peek() == Some(b'{' as u32) {
                    self.stream.advance();
                    let mut n = 0u32;
                    while self.stream.peek() != Some(b'}' as u32) {
                        let Some(d) = self.stream.peek().and_then(hex_digit) else {
                            return Err(ParseError::new(
                                Span::new(pos, self.stream.pos()),
                                "invalid \\u{...} escape",
                            ));
                        };
                        n = n * 16 + d;
                        if n > 0x10FFFF {
                            return Err(ParseError::new(
                                Span::new(pos, self.stream.pos()),
                                "code point out of range in \\u{...} escape",
                            ));
                        }
                        self.stream.advance();
                    }
                    n
                } else {
                    self.scan_hex(4)?
                };
                // lone surrogates are legal JS string content (WTF-8)
                push_wtf8(out, v);
                // closing } / last hex char consumed below for the 4-hex case
                if self.stream.peek() == Some(b'}' as u32) {
                    self.stream.advance();
                }
                return Ok(());
            }
            NL => {}
            CR => {
                // \r\n counts as one line continuation
                self.stream.advance();
                if self.stream.peek() == Some(NL) {
                    self.stream.advance();
                }
                return Ok(());
            }
            LS | PS => {}
            c => push_wtf8(out, c), // \', \", \\, and any other char is itself
        }
        self.stream.advance();
        Ok(())
    }

    fn scan_hex(&mut self, n: usize) -> Result<u32, ParseError> {
        let start = self.stream.pos();
        let mut v = 0u32;
        for _ in 0..n {
            let Some(d) = self.stream.peek().and_then(hex_digit) else {
                return Err(ParseError::new(
                    Span::new(start, self.stream.pos()),
                    "invalid hex escape",
                ));
            };
            v = v * 16 + d;
            self.stream.advance();
        }
        Ok(v)
    }

    fn scan_special(&mut self, start: u32) -> ScanResult {
        use TokenKind::*;
        let c = self.stream.peek().unwrap();
        self.stream.advance();
        let kind = match c as u8 {
            b'?' => {
                if self.try_consume("?=") {
                    NullishAssign
                } else if self.try_consume("?") {
                    Nullish
                } else if self.stream.peek() == Some(b'.' as u32) {
                    // `?.` only when not followed by a digit (`a?.3:b` is a ternary)
                    let save = self.stream.pos();
                    self.stream.advance();
                    let digit = matches!(self.stream.peek(), Some(c) if c < 128 && (c as u8).is_ascii_digit());
                    if digit {
                        self.stream.seek(save);
                        Question
                    } else {
                        QuestionDot
                    }
                } else {
                    Question
                }
            }
            b'.' => {
                if matches!(self.stream.peek(), Some(c) if c < 128 && (c as u8).is_ascii_digit()) {
                    // `.5`: number with a leading dot; scanned from `start`
                    self.stream.seek(start);
                    return self.scan_number(start);
                }
                if self.try_consume("..") {
                    Ellipsis
                } else {
                    Period
                }
            }
            c => {
                // comments were handled in skip_trivia; '/' here is division
                let Some((suffixes, default)) = special_ops(c) else {
                    unreachable!("scan_special dispatched on non-special char");
                };
                let mut kind = default;
                for &(suffix, k) in suffixes {
                    if self.try_consume(suffix) {
                        kind = k;
                        break;
                    }
                }
                kind
            }
        };
        Ok(Token::new(kind, Span::new(start, self.stream.pos())))
    }

    // -- helpers --------------------------------------------------------------

    /// Consume `s` if it follows; restore the position otherwise.
    fn try_consume(&mut self, s: &str) -> bool {
        let save = self.stream.pos();
        for &b in s.as_bytes() {
            if self.stream.peek() == Some(b as u32) {
                self.stream.advance();
            } else {
                self.stream.seek(save);
                return false;
            }
        }
        true
    }

    fn intern_span(&mut self, span: Span) -> crate::Symbol {
        Self::with_span_bytes(&mut self.stream, &mut self.scratch, span, |b| {
            self.symbols.intern(b)
        })
    }

    /// Borrow the span's bytes zero-copy when the stream supports it,
    /// otherwise copy them into the scratch buffer via seek/read.
    /// Takes the fields separately so `f` can still touch `self.symbols`.
    fn with_span_bytes<R>(
        stream: &mut S,
        scratch: &mut Vec<u8>,
        span: Span,
        f: impl FnOnce(&[u8]) -> R,
    ) -> R {
        if let Some(b) = stream.slice(span.start, span.end) {
            return f(b);
        }
        let save = stream.pos();
        stream.seek(span.start);
        scratch.clear();
        while stream.pos() < span.end {
            match stream.peek() {
                Some(c) => {
                    let ch = char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER);
                    let mut tmp = [0u8; 4];
                    scratch.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                    stream.advance();
                }
                None => break,
            }
        }
        stream.seek(save);
        f(scratch)
    }

    fn copy_span(&mut self, span: Span, out: &mut Vec<u8>) {
        Self::with_span_bytes(&mut self.stream, &mut self.scratch, span, |b| {
            out.extend_from_slice(b)
        });
    }
}

fn hex_digit(c: u32) -> Option<u32> {
    if c >= 128 {
        return None;
    }
    (c as u8 as char).to_digit(16)
}

/// Encode a code point as WTF-8: identical to UTF-8 for scalar values, plus
/// the 3-byte pattern for lone surrogates (legal JS string content).
fn push_wtf8(out: &mut Vec<u8>, cp: u32) {
    match char::from_u32(cp) {
        Some(ch) => {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
        None => out.extend_from_slice(&[
            0xE0 | (cp >> 12) as u8,
            0x80 | ((cp >> 6) & 0x3F) as u8,
            0x80 | (cp & 0x3F) as u8,
        ]),
    }
}
