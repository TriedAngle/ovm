use crate::{
    Float, HandleScope, Heap, NoGc, Smi, StringInterner, Symbol, VMString, Value, VmError,
};

pub struct Convert;

impl Convert {
    /// ES ToBoolean. Falsey: `false`, `undefined`, `null`, the hole, 0, -0, NaN,
    /// everything else is truthy.
    pub fn is_truthy<'a>(nogc: &'a NoGc<'a>, v: Value) -> bool {
        if let Some(smi) = Smi::decode(v) {
            return smi.value() != 0;
        }
        let known = nogc.known();
        if v == known.false_object.value()
            || v == known.undefined.value()
            || v == known.null.value()
            || v == known.void.value()
        {
            return false;
        }
        if v == known.true_object.value() {
            return true;
        }
        if let Some(f) = v.get_as::<Float>(nogc, known.float_map) {
            let x = f.value.get();
            // -0.0 compares equal to 0.0; NaN compares unequal to everything
            return x != 0.0 && !x.is_nan();
        }
        if let Some(s) = v.get_as::<VMString>(nogc, known.string_map) {
            return s.len(nogc) != 0;
        }
        true
    }

    pub fn to_number<'a>(nogc: &'a NoGc<'a>, v: Value) -> Result<f64, VmError> {
        if let Some(smi) = Smi::decode(v) {
            return Ok(smi.value() as f64);
        }
        let known = nogc.known();
        if v == known.undefined.value() || v == known.void.value() {
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
        if let Some(f) = v.get_as::<Float>(nogc, known.float_map) {
            return Ok(f.value.get());
        }
        if let Some(s) = v.get_as::<VMString>(nogc, known.string_map) {
            return Ok(Self::string_to_number(s.as_slice(nogc)).unwrap_or(f64::NAN));
        }
        Err(VmError::Type)
    }

    /// StringNumericLiteral → f64 (ES 7.1.4.1). `None` means NaN (invalid
    /// numeric content); empty or all-whitespace input is +0.
    fn string_to_number(bytes: &[u8]) -> Option<f64> {
        let mut s = bytes;
        while let Some((b, rest)) = s.split_first() {
            if b.is_ascii_whitespace() {
                s = rest;
            } else {
                break;
            }
        }
        while let Some((b, rest)) = s.split_last() {
            if b.is_ascii_whitespace() {
                s = rest;
            } else {
                break;
            }
        }
        if s.is_empty() {
            return Some(0.0);
        }
        if s == b"Infinity" || s == b"+Infinity" {
            return Some(f64::INFINITY);
        }
        if s == b"-Infinity" {
            return Some(f64::NEG_INFINITY);
        }
        if s == b"NaN" {
            return Some(f64::NAN);
        }
        let text = core::str::from_utf8(s).ok()?;
        text.parse::<f64>().ok()
    }

    /// Inverse of `to_number` for computed results: a Smi when the double is
    /// an in-range integer, a freshly allocated Float otherwise.
    pub fn to_value(heap: &mut Heap, scope: &HandleScope<'_>, f: f64) -> Value {
        let r = f as i64; // saturating cast; the round-trip check rejects out-of-range values
        if f.is_finite()
            && f.fract() == 0.0
            && Smi::in_range(r)
            && (r as f64) == f
            && !(f == 0.0 && f.is_sign_negative())
        {
            return Smi::new(r).encode();
        }
        heap.allocate_handle::<Float>(f, scope).value()
    }

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
        if v.is_smi() {
            return true;
        }
        let known = nogc.known();
        v == known.undefined.value()
            || v == known.null.value()
            || v == known.true_object.value()
            || v == known.false_object.value()
            || v.get_as::<Float>(nogc, known.float_map).is_some()
            || v.get_as::<VMString>(nogc, known.string_map).is_some()
            || v.get_as::<Symbol>(nogc, known.symbol_map).is_some()
    }

    /// ES ToString on a primitive (no ToPrimitive recursion: the input is
    /// already primitive). Numbers allocate a fresh string, symbols are a
    /// TypeError. The oddball identity strings come from the string table.
    pub fn to_string(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        interner: &StringInterner,
        v: Value,
    ) -> Result<Value, VmError> {
        if let Some(smi) = Smi::decode(v) {
            return Ok(
                VMString::from_bytes(heap, scope, smi.value().to_string().as_bytes()).value(),
            );
        }
        let known = heap.known();
        if v == known.undefined.value() {
            return Ok(interner.intern(heap, scope, "undefined").value());
        }
        if v == known.null.value() {
            return Ok(interner.intern(heap, scope, "null").value());
        }
        if v == known.true_object.value() {
            return Ok(interner.intern(heap, scope, "true").value());
        }
        if v == known.false_object.value() {
            return Ok(interner.intern(heap, scope, "false").value());
        }
        enum PrimitiveString {
            IsString,
            Float(f64),
            Other,
        }
        let kind = heap.no_gc(|nogc| {
            let known = nogc.known();
            if v.get_as::<VMString>(nogc, known.string_map).is_some() {
                PrimitiveString::IsString
            } else if let Some(f) = v.get_as::<Float>(nogc, known.float_map) {
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
                Ok(VMString::from_bytes(heap, scope, text.as_bytes()).value())
            }
            // symbols (and anything else reaching this point) are a TypeError
            PrimitiveString::Other => Err(VmError::Type),
        }
    }
}
