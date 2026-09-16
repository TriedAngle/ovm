use crate::{DenseString, Float, HandleScope, Heap, NoGc, Smi, StringData, Symbol, Value, VmError};

pub struct Convert;

impl Convert {
    /// ES ToBoolean. Falsey: `false`, `undefined`, `null`, the hole, 0, -0, NaN,
    /// everything else is truthy.
    pub fn is_truthy<'a>(nogc: &'a NoGc<'a>, v: Value) -> bool {
        let known = nogc.known();
        if let Some(smi) = Smi::decode(v) {
            return smi.value() != 0;
        }
        if v == known.false_object.value()
            || v == known.undefined.value()
            || v == known.null.value()
            || v == known.the_hole.value()
        {
            return false;
        }
        if v == known.true_object.value() {
            return true;
        }
        if let Some(f) = v.get_as::<Float>(nogc) {
            let x = f.value.get();
            // -0.0 compares equal to 0.0; NaN compares unequal to everything
            return x != 0.0 && !x.is_nan();
        }
        if let Some(s) = v.get_as::<DenseString>(nogc) {
            return !s.is_empty();
        }
        true
    }

    pub fn to_number<'a>(nogc: &'a NoGc<'a>, v: Value) -> Result<f64, VmError> {
        let known = nogc.known();
        if let Some(smi) = Smi::decode(v) {
            return Ok(smi.value() as f64);
        }
        if v == known.undefined.value() || v == known.the_hole.value() {
            return Ok(f64::NAN);
        }
        if v == known.null.value() {
            return Ok(0.0);
        }
        if v == known.false_object.value() {
            return Ok(0.0);
        }
        if v == known.true_object.value() {
            return Ok(1.0);
        }
        if let Some(f) = v.get_as::<Float>(nogc) {
            return Ok(f.value.get());
        }
        if let Some(s) = v.get_as::<DenseString>(nogc) {
            return Ok(Self::string_to_number(s.data(nogc)).unwrap_or(f64::NAN));
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

    /// Inverse of `to_number` for computed results: a Smi when the double is
    /// an in-range integer, a freshly allocated Float otherwise.
    /// The true/false singleton for a Rust bool.
    pub fn boolean(heap: &Heap, b: bool) -> Value {
        let known = heap.known();
        if b {
            known.true_object.value()
        } else {
            known.false_object.value()
        }
    }

    /// ES Type check: numbers, strings, symbols, booleans, null, undefined
    /// are primitives; everything else is an object.
    pub fn is_primitive<'a>(nogc: &'a NoGc<'a>, v: Value) -> bool {
        let known = nogc.known();
        if v.is_smi() {
            return true;
        }
        v == known.undefined.value()
            || v == known.null.value()
            || v == known.true_object.value()
            || v == known.false_object.value()
            || v.get_as::<Float>(nogc).is_some()
            || v.get_as::<DenseString>(nogc).is_some()
            || v.get_as::<Symbol>(nogc).is_some()
    }

    /// ES ToString on a primitive (no ToPrimitive recursion: the input is
    /// already primitive). Numbers allocate a fresh string, symbols are a
    /// TypeError. The oddball identity strings come from the string table.
    pub fn to_string(heap: &mut Heap, scope: &HandleScope<'_>, v: Value) -> Result<Value, VmError> {
        if let Some(smi) = Smi::decode(v) {
            return Ok(DenseString::from_utf8(heap, scope, &smi.value().to_string()).value());
        }
        let known = heap.known();
        if v == known.undefined.value() {
            return Ok(known.strings.undefined.value());
        }
        if v == known.null.value() {
            return Ok(known.strings.null.value());
        }
        if v == known.true_object.value() {
            return Ok(known.strings.true_.value());
        }
        if v == known.false_object.value() {
            return Ok(known.strings.false_.value());
        }
        enum PrimitiveString {
            IsString,
            Float(f64),
            Other,
        }
        let kind = heap.no_gc(|nogc| {
            if v.get_as::<DenseString>(nogc).is_some() {
                PrimitiveString::IsString
            } else if let Some(f) = v.get_as::<Float>(nogc) {
                PrimitiveString::Float(f.value.get())
            } else {
                PrimitiveString::Other
            }
        });
        match kind {
            // strings are their own stringification
            PrimitiveString::IsString => Ok(v),
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
                Ok(DenseString::from_utf8(heap, scope, &text).value())
            }
            // symbols (and anything else reaching this point) are a TypeError
            PrimitiveString::Other => Err(VmError::Type),
        }
    }
}
