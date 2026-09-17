use crate::{
    DenseString, Float, HandleScope, Heap, Smi, StringData, Symbol, Tagged, Value, VmError,
};

pub struct Convert;

impl Convert {
    /// ES ToBoolean. Falsey: `false`, `undefined`, `null`, the hole, 0, -0, NaN,
    /// everything else is truthy.
    pub fn is_truthy(heap: &Heap, v: Tagged<'_, Value>) -> bool {
        let known = heap.known();
        if let Some(smi) = Smi::decode(v.erase()) {
            return smi.value() != 0;
        }
        if v.erase() == known.false_object.as_tagged(heap).erase()
            || v.erase() == known.undefined.as_tagged(heap).erase()
            || v.erase() == known.null.as_tagged(heap).erase()
            || v.erase() == known.the_hole.as_tagged(heap).erase()
        {
            return false;
        }
        if v.erase() == known.true_object.as_tagged(heap).erase() {
            return true;
        }
        if let Some(f) = v.get_as::<Float>() {
            let x = f.value.get();
            // -0.0 compares equal to 0.0; NaN compares unequal to everything
            return x != 0.0 && !x.is_nan();
        }
        if let Some(s) = v.get_as::<DenseString>() {
            return !s.is_empty();
        }
        true
    }

    pub fn to_number(heap: &Heap, v: Tagged<'_, Value>) -> Result<f64, VmError> {
        let known = heap.known();
        if let Some(smi) = Smi::decode(v.erase()) {
            return Ok(smi.value() as f64);
        }
        if v.erase() == known.undefined.as_tagged(heap).erase()
            || v.erase() == known.the_hole.as_tagged(heap).erase()
        {
            return Ok(f64::NAN);
        }
        if v.erase() == known.null.as_tagged(heap).erase() {
            return Ok(0.0);
        }
        if v.erase() == known.false_object.as_tagged(heap).erase() {
            return Ok(0.0);
        }
        if v.erase() == known.true_object.as_tagged(heap).erase() {
            return Ok(1.0);
        }
        if let Some(f) = v.get_as::<Float>() {
            return Ok(f.value.get());
        }
        if let Some(s) = v.get_as::<DenseString>() {
            return Ok(Self::string_to_number(s.data(heap)).unwrap_or(f64::NAN));
        }
        Err(VmError::Type)
    }

    /// StringNumericLiteral → f64 (ES 7.1.4.1). `None` means NaN (invalid
    /// numeric content); empty or all-whitespace input is +0.
    fn string_to_number(data: StringData<'_>) -> Option<f64> {
        // ES trims the same whitespace set as before the re-encoding;
        // the numeric grammar itself is pure ASCII
        let is_ws = |c: u16| c == 0x20 || (0x09..=0x0d).contains(&c);
        let unit = |i: usize| data.code_unit(i);
        let mut lo = 0usize;
        let mut hi = data.len();
        while lo < hi && is_ws(unit(lo)) {
            lo += 1;
        }
        while hi > lo && is_ws(unit(hi - 1)) {
            hi -= 1;
        }
        let matches = |lit: &[u8]| {
            hi - lo == lit.len() && (lo..hi).zip(lit).all(|(i, &b)| unit(i) == b as u16)
        };
        if hi == lo {
            return Some(0.0);
        }
        if matches(b"Infinity") || matches(b"+Infinity") {
            return Some(f64::INFINITY);
        }
        if matches(b"-Infinity") {
            return Some(f64::NEG_INFINITY);
        }
        if matches(b"NaN") {
            return Some(f64::NAN);
        }
        // any unit above 0x7F cannot participate in a numeric literal
        let bytes: Vec<u8> = (lo..hi)
            .map(|i| u8::try_from(unit(i)).ok())
            .collect::<Option<Vec<u8>>>()?;
        let text = core::str::from_utf8(&bytes).ok()?;
        text.parse::<f64>().ok()
    }

    /// The true/false singleton for a Rust bool.
    pub fn boolean<'a>(heap: &'a Heap, b: bool) -> Tagged<'a, Value> {
        let known = heap.known();
        if b {
            known.true_object.as_tagged(heap).erase_type()
        } else {
            known.false_object.as_tagged(heap).erase_type()
        }
    }

    /// ES Type check: numbers, strings, symbols, booleans, null, undefined
    /// are primitives; everything else is an object.
    pub fn is_primitive(heap: &Heap, v: Tagged<'_, Value>) -> bool {
        let known = heap.known();
        if v.is_smi() {
            return true;
        }
        v.erase() == known.undefined.as_tagged(heap).erase()
            || v.erase() == known.null.as_tagged(heap).erase()
            || v.erase() == known.true_object.as_tagged(heap).erase()
            || v.erase() == known.false_object.as_tagged(heap).erase()
            || v.get_as::<Float>().is_some()
            || v.get_as::<DenseString>().is_some()
            || v.get_as::<Symbol>().is_some()
    }

    /// ES ToString on a primitive (no ToPrimitive recursion: the input is
    /// already primitive). Numbers allocate a fresh string, symbols are a
    /// TypeError. The oddball identity strings come from the string table.
    /// The input is rooted first; the result is anchored at the `&mut Heap`
    /// borrow.
    pub fn to_string<'a>(
        heap: &'a mut Heap,
        scope: &HandleScope<'_>,
        v: Tagged<'_, Value>,
    ) -> Result<Tagged<'a, Value>, VmError> {
        enum PrimitiveString {
            Smi(i64),
            IsString,
            Float(f64),
            Undefined,
            Null,
            True,
            False,
            Other,
        }
        // root before classifying: the string arms allocate afterwards
        let v = scope.handle(v);
        let kind = heap.no_gc(|heap| {
            let vt = v.as_tagged(heap);
            let known = heap.known();
            let word = vt.erase();
            if let Some(smi) = Smi::decode(word) {
                PrimitiveString::Smi(smi.value())
            } else if word == known.undefined.as_tagged(heap).erase() {
                PrimitiveString::Undefined
            } else if word == known.null.as_tagged(heap).erase() {
                PrimitiveString::Null
            } else if word == known.true_object.as_tagged(heap).erase() {
                PrimitiveString::True
            } else if word == known.false_object.as_tagged(heap).erase() {
                PrimitiveString::False
            } else if vt.get_as::<DenseString>().is_some() {
                PrimitiveString::IsString
            } else if let Some(f) = vt.get_as::<Float>() {
                PrimitiveString::Float(f.value.get())
            } else {
                PrimitiveString::Other
            }
        });
        match kind {
            PrimitiveString::Smi(n) => {
                let s = DenseString::from_utf8(heap, scope, &n.to_string());
                Ok(s.as_tagged(&*heap).erase_type())
            }
            // strings are their own stringification
            PrimitiveString::IsString => Ok(v.as_tagged(&*heap)),
            PrimitiveString::Undefined => Ok(heap
                .known()
                .strings
                .undefined
                .as_tagged(&*heap)
                .erase_type()),
            PrimitiveString::Null => Ok(heap.known().strings.null.as_tagged(&*heap).erase_type()),
            PrimitiveString::True => Ok(heap.known().strings.true_.as_tagged(&*heap).erase_type()),
            PrimitiveString::False => {
                Ok(heap.known().strings.false_.as_tagged(&*heap).erase_type())
            }
            PrimitiveString::Float(x) => {
                let text = if x.is_nan() {
                    "NaN".to_string()
                } else if x == f64::INFINITY {
                    "Infinity".to_string()
                } else if x == f64::NEG_INFINITY {
                    "-Infinity".to_string()
                } else {
                    format!("{x}")
                };
                let s = DenseString::from_utf8(heap, scope, &text);
                Ok(s.as_tagged(&*heap).erase_type())
            }
            // symbols (and anything else reaching this point) are a TypeError
            PrimitiveString::Other => Err(VmError::Type),
        }
    }
}
