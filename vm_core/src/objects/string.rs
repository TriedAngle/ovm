use core::{alloc::Layout, cell::UnsafeCell, cmp::Ordering};

use crate::{
    EdgeVisitable, GcSlot, Handle, HandleScope, Header, Heap, HeapObject, HeapPtr, Map, MapKind,
    ObjectKind, Smi, Value, Visitor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Latin1,
    Utf16,
}

#[derive(Clone, Copy)]
pub enum StringData<'a> {
    Latin1(&'a [u8]),
    Utf16(&'a [u16]),
}

impl StringData<'_> {
    pub fn len(&self) -> usize {
        match *self {
            StringData::Latin1(b) => b.len(),
            StringData::Utf16(u) => u.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The payload encoding.
    pub fn encoding(&self) -> Encoding {
        match *self {
            StringData::Latin1(_) => Encoding::Latin1,
            StringData::Utf16(_) => Encoding::Utf16,
        }
    }

    pub fn code_unit(&self, i: usize) -> u16 {
        match *self {
            StringData::Latin1(b) => b[i] as u16,
            StringData::Utf16(u) => u[i],
        }
    }

    pub fn is_compressed(&self) -> bool {
        match *self {
            StringData::Latin1(_) => true,
            StringData::Utf16(u) => u.iter().any(|&c| c > 0xFF),
        }
    }

    pub fn matches_ascii(&self, bytes: &[u8]) -> bool {
        if self.len() != bytes.len() {
            return false;
        }
        match *self {
            StringData::Latin1(b) => b == bytes,
            StringData::Utf16(u) => u.iter().zip(bytes).all(|(c, b)| *c == *b as u16),
        }
    }

    pub fn write_units(&self, out: &mut Vec<u16>) {
        match *self {
            StringData::Latin1(b) => out.extend(b.iter().map(|&l| l as u16)),
            StringData::Utf16(u) => out.extend_from_slice(u),
        }
    }

    pub fn to_rust_string(&self) -> String {
        match *self {
            StringData::Latin1(b) => b.iter().map(|&l| l as char).collect(),
            StringData::Utf16(u) => String::from_utf16_lossy(u),
        }
    }

    pub fn hash(&self) -> i64 {
        string_content_hash(*self)
    }
}

impl PartialEq for StringData<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        match (*self, *other) {
            (StringData::Latin1(a), StringData::Latin1(b)) => a == b,
            (StringData::Utf16(a), StringData::Utf16(b)) => a == b,
            (StringData::Latin1(a), StringData::Utf16(b)) => {
                b.iter().zip(a.iter()).all(|(u, l)| *u == *l as u16)
            }
            (StringData::Utf16(a), StringData::Latin1(b)) => {
                a.iter().zip(b.iter()).all(|(u, l)| *u == *l as u16)
            }
        }
    }
}

impl Eq for StringData<'_> {}

impl Ord for StringData<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let n = self.len().min(other.len());
        for i in 0..n {
            let (a, b) = (self.code_unit(i), other.code_unit(i));
            if a != b {
                return a.cmp(&b);
            }
        }
        self.len().cmp(&other.len())
    }
}

impl PartialOrd for StringData<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// TODO: decide on a hash algorithm
pub fn string_content_hash(data: StringData<'_>) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |unit: u16| {
        h ^= unit as u64;
        h = h.wrapping_mul(0x100000001b3);
    };
    match data {
        StringData::Latin1(b) => b.iter().for_each(|&l| mix(l as u16)),
        StringData::Utf16(u) => u.iter().for_each(|&c| mix(c)),
    }
    let masked = (h & ((1 << 62) - 1)) as i64;
    if masked == 0 { 1 } else { masked }
}

#[repr(C)]
pub struct DenseString {
    pub header: Header,
    /// Lazy content hash over code units; Smi 0 = not yet computed.
    pub hash: GcSlot<Smi>,
    /// Length in UTF-16 code units.
    pub length: GcSlot<Smi>,
    /// Inline payload: `length` Latin-1 bytes or UTF-16 code units. (written in map)
    pub data: [UnsafeCell<u8>; 0],
}

impl DenseString {
    fn data_ptr(&self) -> *mut u8 {
        UnsafeCell::raw_get(self.data.as_ptr())
    }

    fn encoding_of(header: &Header) -> Encoding {
        // Safety: raw header read (encoding is map-kind metadata).
        let map = header.map.raw();
        let map_ref = unsafe { HeapPtr::<Map>::new(map.raw_addr() as *mut Map).as_ref() };
        if map_ref.kind().contains(MapKind::LATIN1) {
            Encoding::Latin1
        } else {
            Encoding::Utf16
        }
    }

    pub fn encoding(&self) -> Encoding {
        Self::encoding_of(&self.header)
    }

    pub fn data<'s>(&'s self, _heap: &Heap) -> StringData<'s> {
        let ptr = self.data_ptr();
        let len = self.len();
        match self.encoding() {
            Encoding::Latin1 => {
                StringData::Latin1(unsafe { core::slice::from_raw_parts(ptr, len) })
            }
            Encoding::Utf16 => {
                StringData::Utf16(unsafe { core::slice::from_raw_parts(ptr.cast::<u16>(), len) })
            }
        }
    }

    pub fn len(&self) -> usize {
        self.length.to_smi().value() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn hash(&self, heap: &Heap) -> i64 {
        match self.cached_hash() {
            Some(h) => h,
            None => {
                let h = string_content_hash(self.data(heap));
                self.hash.set(heap, self.tagged(heap), Smi::new(h));
                h
            }
        }
    }

    fn cached_hash(&self) -> Option<i64> {
        let h = self.hash.to_smi().value();
        (h != 0).then_some(h)
    }

    pub fn content_eq(&self, heap: &Heap, other: &DenseString) -> bool {
        if let (Some(a), Some(b)) = (self.cached_hash(), other.cached_hash())
            && a != b
        {
            return false;
        }
        self.data(heap) == other.data(heap)
    }

    pub fn code_unit(&self, _heap: &Heap, i: usize) -> u16 {
        debug_assert!(i < self.len());
        match self.encoding() {
            Encoding::Latin1 => unsafe { *self.data_ptr().add(i) as u16 },
            Encoding::Utf16 => unsafe { *self.data_ptr().cast::<u16>().add(i) },
        }
    }

    /// The one-code-unit string at index `i`, freshly allocated (string
    /// comparisons are by content, so identity never shows). `None` when
    /// the receiver is not a string or `i` is out of range (ES 6.1.4:
    /// string indices are code units).
    pub fn char_at<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        receiver: Value,
        i: usize,
    ) -> Option<Handle<'s, DenseString>> {
        let unit = {
            // Safety: caller-supplied word, fresh at entry.
            let s = unsafe { receiver.assume_valid(heap) }.get_as::<DenseString>()?;
            (i < s.len()).then(|| s.code_unit(heap, i))
        }?;
        Some(Self::from_units(heap, scope, &[unit]))
    }

    pub fn to_rust_string(&self, heap: &Heap) -> String {
        self.data(heap).to_rust_string()
    }

    pub fn from_units<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        units: &[u16],
    ) -> Handle<'s, DenseString> {
        if units.iter().all(|&c| c <= 0xFF) {
            let bytes: Vec<u8> = units.iter().map(|&c| c as u8).collect();
            return Self::from_latin1(heap, scope, &bytes);
        }
        Self::from_data(heap, scope, StringData::Utf16(units))
    }

    pub fn from_latin1<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        bytes: &[u8],
    ) -> Handle<'s, DenseString> {
        Self::from_data(heap, scope, StringData::Latin1(bytes))
    }

    pub fn from_utf8<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        s: &str,
    ) -> Handle<'s, DenseString> {
        if s.is_ascii() {
            return Self::from_latin1(heap, scope, s.as_bytes());
        }
        let units: Vec<u16> = s.encode_utf16().collect();
        Self::from_units(heap, scope, &units)
    }

    pub fn from_data<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        data: StringData<'_>,
    ) -> Handle<'s, DenseString> {
        debug_assert!(
            data.is_compressed(),
            "heap strings are always in the compressed encoding"
        );
        heap.allocate_handle::<DenseString>((data, 0), scope)
    }

    pub fn concat<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        a: Handle<'_, Value>,
        b: Handle<'_, Value>,
    ) -> Handle<'s, DenseString> {
        let units = {
            let sa = a
                .as_tagged(heap)
                .get_as::<DenseString>()
                .expect("concat operand must be a string")
                .as_ref();
            let sb = b
                .as_tagged(heap)
                .get_as::<DenseString>()
                .expect("concat operand must be a string")
                .as_ref();
            let mut out = Vec::with_capacity(sa.len() + sb.len());
            sa.data(heap).write_units(&mut out);
            sb.data(heap).write_units(&mut out);
            out
        };
        Self::from_units(heap, scope, &units)
    }
}

impl HeapObject for DenseString {
    const KIND: ObjectKind = ObjectKind::DenseString;
    type Init<'a> = (StringData<'a>, i64);

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        let values = match config.0 {
            StringData::Latin1(b) => Layout::array::<u8>(b.len()).expect("string layout"),
            StringData::Utf16(u) => Layout::array::<u16>(u.len()).expect("string layout"),
        };
        Layout::new::<Self>()
            .extend(values)
            .expect("string layout")
            .0
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        debug_assert!(
            config.0.is_compressed(),
            "heap strings are always in the compressed encoding"
        );
        let host = self.tagged(heap);
        let map = match config.0.encoding() {
            Encoding::Latin1 => heap.known().dense_latin1_string_map,
            Encoding::Utf16 => heap.known().dense_utf16_string_map,
        };
        self.header.map.set(heap, host, map.as_tagged(heap));
        self.hash.set(heap, host, Smi::new(config.1));
        self.length.set(heap, host, Smi::new(config.0.len() as i64));
        let ptr = self.data_ptr();
        match config.0 {
            StringData::Latin1(b) => unsafe {
                core::ptr::copy_nonoverlapping(b.as_ptr(), ptr, b.len())
            },
            StringData::Utf16(u) => unsafe {
                core::ptr::copy_nonoverlapping(
                    u.as_ptr().cast::<u8>(),
                    ptr,
                    core::mem::size_of_val(u),
                )
            },
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        let values = match self.encoding() {
            Encoding::Latin1 => Layout::array::<u8>(self.len()).expect("string layout"),
            Encoding::Utf16 => Layout::array::<u16>(self.len()).expect("string layout"),
        };
        Layout::new::<Self>()
            .extend(values)
            .expect("string layout")
            .0
    }
}

impl EdgeVisitable for DenseString {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}

/// Decode WTF-8 (UTF-8 plus the 3-byte lone-surrogate pattern) into code units.
pub fn decode_wtf8(bytes: &[u8]) -> Option<Vec<u16>> {
    let mut units = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let (width, mut cp): (usize, u32) = match b {
            0x00..=0x7F => (1, b as u32),
            0xC0..=0xDF => (2, (b & 0x1F) as u32),
            0xE0..=0xEF => (3, (b & 0x0F) as u32),
            0xF0..=0xF7 => (4, (b & 0x07) as u32),
            _ => return None,
        };
        if i + width > bytes.len() {
            return None;
        }
        for &cont in &bytes[i + 1..i + width] {
            if cont & 0xC0 != 0x80 {
                return None;
            }
            cp = (cp << 6) | (cont & 0x3F) as u32;
        }
        i += width;
        match width {
            // BMP scalar or a lone surrogate half: one code unit as-is
            1..=3 => units.push(cp as u16),
            // supplementary scalar: split into its surrogate pair
            _ => {
                let v = cp.checked_sub(0x1_0000)?;
                units.push((0xD800 + (v >> 10)) as u16);
                units.push((0xDC00 + (v & 0x3FF)) as u16);
            }
        }
    }
    Some(units)
}
