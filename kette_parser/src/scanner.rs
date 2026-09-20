use crate::{ByteSpan, CharStream, ParseError, SymbolTable, Token, TokenKind, TokenValue};

pub type ScanResult = Result<Token, ParseError>;

const NL: u32 = 0x0A; // \n
const CR: u32 = 0x0D; // \r
const BACKSLASH: u32 = 0x5C;

fn is_newline(c: u32) -> bool {
    matches!(c, NL | CR)
}

fn is_ident_start(c: u32) -> bool {
    c == b'_' as u32
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
        b"let" => Let,
        b"if" => If,
        b"else" => Else,
        b"while" => While,
        b"for" => For,
        b"in" => In,
        b"match" => Match,
        b"self" => SelfKw,
        b"null" => Null,
        b"true" => True,
        b"false" => False,
        b"return" => Return,
        b"try" => Try,
        b"catch" => Catch,
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

    let mut c = b'0';
    while c <= b'9' {
        t[c as usize] = Digit;
        c += 1;
    }

    t[b'"' as usize] = Quote;
    t[b'\'' as usize] = Quote;

    t[b'(' as usize] = Punct(TokenKind::LParen);
    t[b')' as usize] = Punct(TokenKind::RParen);
    t[b'{' as usize] = Punct(TokenKind::LBrace);
    t[b'}' as usize] = Punct(TokenKind::RBrace);
    t[b'[' as usize] = Punct(TokenKind::LBracket);
    t[b']' as usize] = Punct(TokenKind::RBracket);
    t[b',' as usize] = Punct(TokenKind::Comma);
    t[b'.' as usize] = Punct(TokenKind::Period);
    t[b':' as usize] = Punct(TokenKind::Colon);
    t[b';' as usize] = Punct(TokenKind::Semicolon);
    t[b'*' as usize] = Punct(TokenKind::Star);
    t[b'^' as usize] = Punct(TokenKind::Caret);

    t[b'=' as usize] = Special;
    t[b'!' as usize] = Special;
    t[b'<' as usize] = Special;
    t[b'>' as usize] = Special;
    t[b'+' as usize] = Special;
    t[b'-' as usize] = Special;
    t[b'/' as usize] = Special;
    t[b'%' as usize] = Special;
    t[b'&' as usize] = Special;
    t[b'|' as usize] = Special;

    t
}

const FIRST_CHAR: [Action; 128] = first_char_table();

/// Longest-match suffixes for multi-char operators: for the consumed first
/// char, the candidate suffixes (longest first) and the fallback kind.
/// `&` and `|` have extra rules and stay hand-written in `scan_special`.
type Suffixes = (&'static [(&'static str, TokenKind)], TokenKind);

fn special_ops(c: u8) -> Option<Suffixes> {
    use TokenKind::*;
    Some(match c {
        b'=' => (&[("=", EqEq)], Assign),
        b'!' => (&[("=", NotEq)], Bang),
        b'<' => (&[("=", LtEq)], Lt),
        b'>' => (&[("=", GtEq)], Gt),
        b'-' => (&[(">", Arrow)], Minus),
        b'+' => (&[], Plus),
        b'/' => (&[], Slash),
        b'%' => (&[], Percent),
        _ => return None,
    })
}

/// Saved scanner state for backwards rewinds (speculative slot-list checks).
/// Only valid within already-consumed input.
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
    /// line terminator crossed since the last produced token
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
                span: ByteSpan::new(start, start),
            });
        };

        let mut tok = if c < 128 {
            match FIRST_CHAR[c as usize] {
                Action::Illegal => {
                    return Err(ParseError::new(
                        ByteSpan::new(start, start + 1),
                        "unexpected character",
                    ));
                }
                Action::IdentStart => self.scan_identifier(start)?,
                Action::Digit => self.scan_number(start)?,
                Action::Quote => self.scan_string(c, start)?,
                Action::Punct(kind) => {
                    self.stream.advance();
                    Token::new(kind, ByteSpan::new(start, self.stream.pos()))
                }
                Action::Special => self.scan_special(start)?,
            }
        } else if is_ident_start(c) {
            self.scan_identifier(start)?
        } else {
            return Err(ParseError::new(
                ByteSpan::new(start, start + 1),
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
                NL | CR => {
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
                            let mut depth = 1u32;
                            loop {
                                match self.stream.peek() {
                                    None => {
                                        return Err(ParseError::new(
                                            ByteSpan::new(save, self.stream.pos()),
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
                                            depth -= 1;
                                            if depth == 0 {
                                                break;
                                            }
                                        }
                                    }
                                    Some(c) if c == b'/' as u32 => {
                                        self.stream.advance();
                                        if self.stream.peek() == Some(b'*' as u32) {
                                            self.stream.advance();
                                            depth += 1;
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
        let span = ByteSpan::new(start, self.stream.pos());
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

    fn scan_number(&mut self, start: u32) -> ScanResult {
        self.consume_digits();
        if self.stream.peek() == Some(b'.' as u32) {
            let save = self.stream.pos();
            self.stream.advance();
            if matches!(self.stream.peek(), Some(c) if c < 128 && (c as u8).is_ascii_digit()) {
                self.consume_digits();
            } else {
                self.stream.seek(save);
            }
        }
        let span = ByteSpan::new(start, self.stream.pos());
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

    /// String literal. Contents are decoded into a symbol.
    fn scan_string(&mut self, quote: u32, start: u32) -> ScanResult {
        self.stream.advance(); // opening quote
        let content_start = self.stream.pos();
        let mut decoded: Option<Vec<u8>> = None;
        loop {
            let Some(c) = self.stream.peek() else {
                return Err(ParseError::new(
                    ByteSpan::new(start, self.stream.pos()),
                    "unterminated string literal",
                ));
            };
            if c == quote {
                let content_end = self.stream.pos();
                self.stream.advance();
                let span = ByteSpan::new(start, self.stream.pos());
                let sym = match decoded {
                    Some(buf) => self.symbols.intern(&buf),
                    None => self.intern_span(ByteSpan::new(content_start, content_end)),
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
                    ByteSpan::new(start, self.stream.pos()),
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
                push_utf8(decoded.as_mut().unwrap(), c);
            }
            self.stream.advance();
        }
    }

    fn ensure_decoded(&mut self, decoded: &mut Option<Vec<u8>>, content_start: u32) {
        if decoded.is_none() {
            let mut v = Vec::new();
            self.copy_span(ByteSpan::new(content_start, self.stream.pos()), &mut v);
            *decoded = Some(v);
        }
    }

    fn scan_escape(&mut self, out: &mut Vec<u8>) -> Result<(), ParseError> {
        let pos = self.stream.pos();
        let Some(c) = self.stream.peek() else {
            return Err(ParseError::new(
                ByteSpan::new(pos, pos),
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
                push_utf8(out, v);
                return Ok(());
            }
            c if c == b'u' as u32 => {
                self.stream.advance();
                let v = self.scan_hex(4)?;
                push_utf8(out, v);
                return Ok(());
            }
            NL => {}
            CR => {
                self.stream.advance();
                if self.stream.peek() == Some(NL) {
                    self.stream.advance();
                }
                return Ok(());
            }
            c => push_utf8(out, c), // \', \", \\, and any other char is itself
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
                    ByteSpan::new(start, self.stream.pos()),
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
            b'|' => {
                if self.try_consume("|") {
                    OrOr
                } else {
                    Pipe
                }
            }
            b'&' => {
                if self.try_consume("&") {
                    AmpAmp
                } else {
                    return Err(ParseError::new(
                        ByteSpan::new(start, self.stream.pos()),
                        "unexpected `&` (did you mean `&&`?)",
                    ));
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
        Ok(Token::new(kind, ByteSpan::new(start, self.stream.pos())))
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

    fn intern_span(&mut self, span: ByteSpan) -> crate::Symbol {
        Self::with_span_bytes(&mut self.stream, &mut self.scratch, span, |b| {
            self.symbols.intern(b)
        })
    }

    /// Borrow the span's bytes zero-copy when the stream supports it,
    /// otherwise copy them into the scratch buffer via seek/read.
    fn with_span_bytes<R>(
        stream: &mut S,
        scratch: &mut Vec<u8>,
        span: ByteSpan,
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

    fn copy_span(&mut self, span: ByteSpan, out: &mut Vec<u8>) {
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

/// Encode a code point as UTF-8 (lone surrogates become the replacement char).
fn push_utf8(out: &mut Vec<u8>, cp: u32) {
    let ch = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
    let mut buf = [0u8; 4];
    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
}
