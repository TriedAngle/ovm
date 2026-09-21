//! Language-agnostic parser plumbing shared by the frontend parsers.
//!
//! This is the small set of data types that every hand-written parser in the
//! workspace needs: source offsets, a pull-based character stream, a
//! symbol interner and a uniform parse error. Language-specific scanners and
//! ASTs stay in their own crates.

use std::collections::HashMap;

/// A half-open `[start, end)` range of byte offsets into the source text.
///
/// Named `ByteSpan` to keep it distinct from [`ir::Span`], which addresses a
/// run inside a pool rather than a region of the source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteSpan {
    pub start: u32,
    pub end: u32,
}

impl ByteSpan {
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }
}

/// Interned identifier / string handle into a [`SymbolTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(pub u32);

/// A frontend parse failure, in a form any parser can produce.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub span: ByteSpan,
    pub message: String,
}

impl ParseError {
    pub fn new(span: ByteSpan, message: impl Into<String>) -> Self {
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

/// Parse-local byte-slice interner; heap internalization at materialization.
#[derive(Default)]
pub struct SymbolTable {
    map: HashMap<Vec<u8>, Symbol>,
    strings: Vec<Vec<u8>>,
}

impl SymbolTable {
    pub fn intern(&mut self, s: &[u8]) -> Symbol {
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let sym = Symbol(self.strings.len() as u32);
        let owned = s.to_vec();
        self.strings.push(owned.clone());
        self.map.insert(owned, sym);
        sym
    }

    pub fn get(&self, sym: Symbol) -> &[u8] {
        &self.strings[sym.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Pull-based stream of decoded code points; the only encoding-specific
/// layer. UTF-16 streams may yield lone surrogates (0xD800..=0xDFFF).
pub trait CharStream {
    /// &mut self: buffered/chunked streams may need to fetch data to answer
    fn peek(&mut self) -> Option<u32>;
    fn advance(&mut self);
    /// byte offset for UTF-8, code units for UTF-16
    fn pos(&self) -> u32;
    /// backwards-only, over already-consumed data
    fn seek(&mut self, pos: u32);
    /// zero-copy access to a consumed range; `None` if the bytes are not
    /// contiguous in memory (chunked streams), callers then fall back to
    /// copying via seek/read
    fn slice(&self, _start: u32, _end: u32) -> Option<&[u8]> {
        None
    }
}

pub struct Utf8SliceStream<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Utf8SliceStream<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }
}

impl CharStream for Utf8SliceStream<'_> {
    fn peek(&mut self) -> Option<u32> {
        self.src[self.pos..].chars().next().map(|c| c as u32)
    }

    fn advance(&mut self) {
        if let Some(c) = self.src[self.pos..].chars().next() {
            self.pos += c.len_utf8();
        }
    }

    fn pos(&self) -> u32 {
        self.pos as u32
    }

    fn seek(&mut self, pos: u32) {
        let pos = pos as usize;
        assert!(pos <= self.pos, "streams only seek backwards");
        assert!(self.src.is_char_boundary(pos), "seek to non-boundary {pos}");
        self.pos = pos;
    }

    fn slice(&self, start: u32, end: u32) -> Option<&[u8]> {
        Some(&self.src.as_bytes()[start as usize..end as usize])
    }
}
