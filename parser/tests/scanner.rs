use parser::TokenKind::*;
use parser::{ParseError, Scanner, Span, Token, TokenKind, Utf8SliceStream};

fn scan_all(src: &str) -> Vec<Token> {
    let mut sc = Scanner::new(Utf8SliceStream::new(src));
    let mut out = vec![];
    loop {
        let t = sc.next_token().expect("scan error");
        out.push(t);
        if t.kind == Eof {
            break;
        }
    }
    out
}

fn kinds(src: &str) -> Vec<TokenKind> {
    scan_all(src).iter().map(|t| t.kind).collect()
}

fn scan_err(src: &str) -> ParseError {
    let mut sc = Scanner::new(Utf8SliceStream::new(src));
    loop {
        match sc.next_token() {
            Err(e) => return e,
            Ok(t) if t.kind == Eof => panic!("expected scan error for {src:?}"),
            _ => {}
        }
    }
}

#[test]
fn single_char_punctuation() {
    let cases = [
        ("(", LParen),
        (")", RParen),
        ("{", LBrace),
        ("}", RBrace),
        ("[", LBracket),
        ("]", RBracket),
        (";", Semicolon),
        (",", Comma),
        (":", Colon),
        ("~", Tilde),
    ];
    for (src, kind) in cases {
        assert_eq!(kinds(src), vec![kind, Eof], "for {src:?}");
    }
}

#[test]
fn operators() {
    let cases = [
        ("=", Assign),
        ("==", EqEq),
        ("===", EqEqEq),
        ("=>", Arrow),
        ("!", Bang),
        ("!=", NotEq),
        ("!==", NotEqEq),
        ("<", Lt),
        ("<<", Shl),
        ("<<=", ShlAssign),
        ("<=", LtEq),
        (">", Gt),
        (">>", Shr),
        (">>>", Ushr),
        (">=", GtEq),
        (">>=", ShrAssign),
        (">>>=", UshrAssign),
        ("+", Plus),
        ("++", PlusPlus),
        ("+=", PlusAssign),
        ("-", Minus),
        ("--", MinusMinus),
        ("-=", MinusAssign),
        ("*", Star),
        ("**", StarStar),
        ("*=", StarAssign),
        ("**=", StarStarAssign),
        ("/", Slash),
        ("/=", SlashAssign),
        ("%", Percent),
        ("%=", PercentAssign),
        ("&", Amp),
        ("&&", AmpAmp),
        ("&=", AmpAssign),
        ("&&=", AmpAmpAssign),
        ("|", Pipe),
        ("||", OrOr),
        ("|=", PipeAssign),
        ("||=", OrOrAssign),
        ("^", Caret),
        ("^=", CaretAssign),
        ("?", Question),
        ("??", Nullish),
        ("??=", NullishAssign),
        ("...", Ellipsis),
        (".", Period),
    ];
    for (src, kind) in cases {
        assert_eq!(kinds(src), vec![kind, Eof], "for {src:?}");
    }
}

#[test]
fn optional_chaining_vs_ternary_digit() {
    // `?.` not followed by a digit
    assert_eq!(
        kinds("a?.b:c"),
        vec![Identifier, QuestionDot, Identifier, Colon, Identifier, Eof]
    );
    // `? .` followed by a digit is a ternary with number `.3`
    assert_eq!(
        kinds("a?.3:b"),
        vec![Identifier, Question, Number, Colon, Identifier, Eof]
    );
}

#[test]
fn identifiers_and_keywords() {
    assert_eq!(kinds("while"), vec![While, Eof]);
    assert_eq!(kinds("while1"), vec![Identifier, Eof]);
    assert_eq!(kinds("_$x9"), vec![Identifier, Eof]);
    // contextual keywords scan as their own kinds; the parser decides
    assert_eq!(
        kinds("let async await yield static"),
        vec![Let, Async, Await, Yield, Static, Eof]
    );
    // strict-mode-only reserved words
    assert_eq!(kinds("implements package"), vec![Implements, Package, Eof]);
    assert_eq!(kinds("letx"), vec![Identifier, Eof]);
    assert_eq!(kinds("é_ident"), vec![Identifier, Eof]);
}

#[test]
fn numbers() {
    let cases = [
        ("0", 0.0),
        ("42", 42.0),
        ("1.25", 1.25),
        (".5", 0.5),
        ("1.", 1.0),
        ("1e3", 1000.0),
        ("1E-2", 0.01),
        ("2.5e+2", 250.0),
    ];
    for (src, want) in cases {
        let toks = scan_all(src);
        assert_eq!(toks[0].kind, Number, "for {src:?}");
        assert_eq!(toks[0].value.number(), Some(want), "for {src:?}");
    }
    // exponent without digits is not consumed
    assert_eq!(
        kinds("1e+x"),
        vec![Number, Identifier, Plus, Identifier, Eof]
    );
    // classic: `1..toString()` is Number(1.) then `.` then identifier
    assert_eq!(
        kinds("1..toString()"),
        vec![Number, Period, Identifier, LParen, RParen, Eof]
    );
}

#[test]
fn strings() {
    let mut sc = Scanner::new(Utf8SliceStream::new(
        r#"'a' "b" 'x\ny' '\x41A' 'A' 'B' 'a\
b' 'é'"#,
    ));
    let mut texts = vec![];
    loop {
        let t = sc.next_token().expect("scan error");
        if t.kind == Eof {
            break;
        }
        assert_eq!(t.kind, String);
        let sym = parser::Symbol(t.value.symbol().unwrap());
        texts.push(sc.symbols().get(sym).to_vec());
    }
    assert_eq!(
        texts,
        vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"x\ny".to_vec(),
            b"AA".to_vec(), // \x41 = A
            b"A".to_vec(),  // A
            b"B".to_vec(),  // B
            b"ab".to_vec(), // line continuation
            vec![0xC3, 0xA9], // é as UTF-8 (WTF-8) bytes
        ]
    );
}

#[test]
fn string_errors() {
    scan_err("'abc"); // unterminated
    scan_err("'a\nb'"); // raw newline
    scan_err("'\\u{110000}'"); // code point out of range
    scan_err("'\\xZZ'");
}

#[test]
fn wtf8_string_contents() {
    let mut sc = Scanner::new(Utf8SliceStream::new("'\\u2028' '\\u{1F600}' '\\uD800'"));
    let mut texts = vec![];
    loop {
        let t = sc.next_token().expect("scan error");
        if t.kind == Eof {
            break;
        }
        texts.push(
            sc.symbols()
                .get(parser::Symbol(t.value.symbol().unwrap()))
                .to_vec(),
        );
    }
    assert_eq!(
        texts,
        vec![
            vec![0xE2, 0x80, 0xA8],       // U+2028, normal UTF-8
            vec![0xF0, 0x9F, 0x98, 0x80], // U+1F600, normal UTF-8
            vec![0xED, 0xA0, 0x80],       // lone surrogate: WTF-8 3-byte pattern
        ]
    );
}

#[test]
fn comments() {
    assert_eq!(kinds("a // hello\nb"), vec![Identifier, Identifier, Eof]);
    assert_eq!(kinds("a /* x */ b"), vec![Identifier, Identifier, Eof]);
    scan_err("a /* x");
    scan_err("/* unterminated");
    // division is not a comment
    assert_eq!(kinds("a / b"), vec![Identifier, Slash, Identifier, Eof]);
}

#[test]
fn asi_newline_flag() {
    let toks = scan_all("a\nb");
    assert!(!toks[0].after_newline);
    assert!(toks[1].after_newline);

    let toks = scan_all("a\r\nb");
    assert!(toks[1].after_newline);

    let toks = scan_all("a b");
    assert!(!toks[1].after_newline);

    let toks = scan_all("a /* \n */ b");
    assert!(toks[1].after_newline);

    let toks = scan_all("a // x\nb");
    assert!(toks[1].after_newline);

    // newline before EOF
    let toks = scan_all("a\n");
    assert!(toks[1].after_newline);
    assert_eq!(toks[1].kind, Eof);
}

#[test]
fn spans_are_byte_offsets() {
    let toks = scan_all("alpha beta");
    assert_eq!(toks[0].span, Span::new(0, 5));
    assert_eq!(toks[1].span, Span::new(6, 10));
    assert_eq!(toks[2].span, Span::new(10, 10)); // eof
}

#[test]
fn eof_repeats_forever() {
    let mut sc = Scanner::new(Utf8SliceStream::new(""));
    for _ in 0..3 {
        assert_eq!(sc.next_token().unwrap().kind, Eof);
    }
}

#[test]
fn peek_and_peek_ahead_do_not_consume() {
    let mut sc = Scanner::new(Utf8SliceStream::new("a + b"));
    assert_eq!(sc.peek().as_ref().unwrap().kind, Identifier);
    assert_eq!(sc.peek().as_ref().unwrap().kind, Identifier); // idempotent
    assert_eq!(sc.peek_ahead().as_ref().unwrap().kind, Plus);
    assert_eq!(sc.next_token().unwrap().kind, Identifier); // still first
    assert_eq!(sc.next_token().unwrap().kind, Plus);
    assert_eq!(sc.peek_ahead().as_ref().unwrap().kind, Eof);
}

#[test]
fn bookmark_restore() {
    let mut sc = Scanner::new(Utf8SliceStream::new("foo + bar"));
    assert_eq!(sc.next_token().unwrap().kind, Identifier);
    let bm = sc.bookmark();
    assert_eq!(sc.next_token().unwrap().kind, Plus);
    assert_eq!(sc.next_token().unwrap().kind, Identifier);
    sc.restore(bm);
    assert_eq!(sc.next_token().unwrap().kind, Plus);
    assert_eq!(sc.next_token().unwrap().kind, Identifier);
    assert_eq!(sc.next_token().unwrap().kind, Eof);
}

#[test]
fn radix_numbers() {
    let cases = [
        ("0x1F", 31.0),
        ("0XFF", 255.0),
        ("0b101", 5.0),
        ("0o17", 15.0),
        ("0O10", 8.0),
        ("0x0", 0.0),
    ];
    for (src, want) in cases {
        let toks = scan_all(src);
        assert_eq!(toks[0].kind, Number, "for {src:?}");
        assert_eq!(toks[0].value.number(), Some(want), "for {src:?}");
    }
    scan_err("0x");
    scan_err("0b");
    scan_err("0b2");
    scan_err("0o9");
    scan_err("0x1g");
}

#[test]
fn bigint_radix() {
    let mut sc = Scanner::new(Utf8SliceStream::new("0xFFn 0b11n 123n"));
    let mut texts = vec![];
    loop {
        let t = sc.next_token().expect("scan error");
        if t.kind == Eof {
            break;
        }
        assert_eq!(t.kind, BigInt, "expected bigint");
        texts.push(
            sc.symbols()
                .get(parser::Symbol(t.value.symbol().unwrap()))
                .to_vec(),
        );
    }
    assert_eq!(
        texts,
        vec![b"0xFF".to_vec(), b"0b11".to_vec(), b"123".to_vec()]
    );
}
