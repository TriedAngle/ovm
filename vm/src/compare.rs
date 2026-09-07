use crate::{Convert, Float, NoGc, Smi, VMString, Value, VmError};

pub struct Compare;

impl Compare {
    pub fn strict_equal<'a>(nogc: &'a NoGc<'a>, x: Value, y: Value) -> bool {
        let known = nogc.known();
        let x_num = x.is_smi() || x.get_as::<Float>(nogc, known.float_map).is_some();
        let y_num = y.is_smi() || y.get_as::<Float>(nogc, known.float_map).is_some();
        if x_num || y_num {
            // both must be numbers (Float === "1" is false, no parsing);
            // NaN is unequal to everything (even itself), -0 equals +0
            if !x_num || !y_num {
                return false;
            }
            let number_value = |v: Value| match v.get_as::<Float>(nogc, known.float_map) {
                Some(f) => f.value.get(),
                None => Smi::decode(v).unwrap().value() as f64,
            };
            let a = number_value(x);
            let b = number_value(y);
            if a.is_nan() || b.is_nan() {
                return false;
            }
            return a == b;
        }
        // identical bits: same string object, same heap object
        if x == y {
            return true;
        }
        if let (Some(sx), Some(sy)) = (
            x.get_as::<VMString>(nogc, known.string_map),
            y.get_as::<VMString>(nogc, known.string_map),
        ) {
            return sx.as_slice(nogc) == sy.as_slice(nogc);
        }
        false
    }

    /// ES IsLooselyEqual (==) for primitives.
    pub fn equal<'a>(nogc: &'a NoGc<'a>, x: Value, y: Value) -> Result<bool, VmError> {
        if Self::strict_equal(nogc, x, y) {
            return Ok(true);
        }
        let known = nogc.known();
        let nullish = |v: Value| v == known.null.value() || v == known.undefined.value();
        if nullish(x) && nullish(y) {
            return Ok(true);
        }
        let is_bool = |v: Value| v == known.true_object.value() || v == known.false_object.value();
        let is_string = |v: Value| v.get_as::<VMString>(nogc, known.string_map).is_some();
        let is_number = |v: Value| v.is_smi() || v.get_as::<Float>(nogc, known.float_map).is_some();
        // number ↔ string: the string parses as a number
        if is_number(x) && is_string(y) {
            return Ok(Convert::to_number(nogc, x)? == Convert::to_number(nogc, y)?);
        }
        if is_string(x) && is_number(y) {
            return Ok(Convert::to_number(nogc, x)? == Convert::to_number(nogc, y)?);
        }
        // booleans become numbers (exactly representable as smis, no allocation)
        if is_bool(x) {
            let n = Smi::new(if x == known.true_object.value() { 1 } else { 0 }).encode();
            return Self::equal(nogc, n, y);
        }
        if is_bool(y) {
            let n = Smi::new(if y == known.true_object.value() { 1 } else { 0 }).encode();
            return Self::equal(nogc, x, n);
        }
        let is_object =
            |v: Value| !v.is_smi() && !is_bool(v) && !nullish(v) && !is_string(v) && !is_number(v);
        if is_object(x) || is_object(y) {
            return Err(VmError::Type);
        }
        Ok(false)
    }

    /// ES SameValue (7.2.11): NaN equals NaN, +0 and -0 are distinct, and
    /// strings compare by content (identity for everything else).
    /// Unlike `strict_equal` (===); used by [[DefineOwnProperty]] validation.
    pub fn same_value<'a>(nogc: &'a NoGc<'a>, x: Value, y: Value) -> bool {
        // identical bits: same object, same string, same smi, or the very
        // same NaN heap object
        if x == y {
            return true;
        }
        let known = nogc.known();
        let x_num = x.is_smi() || x.get_as::<Float>(nogc, known.float_map).is_some();
        let y_num = y.is_smi() || y.get_as::<Float>(nogc, known.float_map).is_some();
        if x_num && y_num {
            let number_value = |v: Value| match v.get_as::<Float>(nogc, known.float_map) {
                Some(f) => f.value.get(),
                None => Smi::decode(v).unwrap().value() as f64,
            };
            let a = number_value(x);
            let b = number_value(y);
            if a.is_nan() && b.is_nan() {
                return true;
            }
            if a != b {
                return false;
            }
            // equal values: +/-0 are distinct
            return !(a == 0.0 && a.is_sign_negative() != b.is_sign_negative());
        }
        if let (Some(sx), Some(sy)) = (
            x.get_as::<VMString>(nogc, known.string_map),
            y.get_as::<VMString>(nogc, known.string_map),
        ) {
            return sx.as_slice(nogc) == sy.as_slice(nogc);
        }
        false
    }

    pub fn less_than<'a>(nogc: &'a NoGc<'a>, x: Value, y: Value) -> Result<bool, VmError> {
        let known = nogc.known();
        if let (Some(sx), Some(sy)) = (
            x.get_as::<VMString>(nogc, known.string_map),
            y.get_as::<VMString>(nogc, known.string_map),
        ) {
            return Ok(sx.as_slice(nogc) < sy.as_slice(nogc));
        }
        let a = Convert::to_number(nogc, x)?;
        let b = Convert::to_number(nogc, y)?;
        Ok(a < b)
    }

    pub fn less_than_or_equal<'a>(nogc: &'a NoGc<'a>, x: Value, y: Value) -> Result<bool, VmError> {
        let known = nogc.known();
        if let (Some(sx), Some(sy)) = (
            x.get_as::<VMString>(nogc, known.string_map),
            y.get_as::<VMString>(nogc, known.string_map),
        ) {
            return Ok(sx.as_slice(nogc) <= sy.as_slice(nogc));
        }
        let a = Convert::to_number(nogc, x)?;
        let b = Convert::to_number(nogc, y)?;
        Ok(a <= b)
    }

    pub fn greater_than<'a>(nogc: &'a NoGc<'a>, x: Value, y: Value) -> Result<bool, VmError> {
        let known = nogc.known();
        if let (Some(sx), Some(sy)) = (
            x.get_as::<VMString>(nogc, known.string_map),
            y.get_as::<VMString>(nogc, known.string_map),
        ) {
            return Ok(sx.as_slice(nogc) > sy.as_slice(nogc));
        }
        let a = Convert::to_number(nogc, x)?;
        let b = Convert::to_number(nogc, y)?;
        Ok(a > b)
    }

    pub fn greater_than_or_equal<'a>(
        nogc: &'a NoGc<'a>,
        x: Value,
        y: Value,
    ) -> Result<bool, VmError> {
        let known = nogc.known();
        if let (Some(sx), Some(sy)) = (
            x.get_as::<VMString>(nogc, known.string_map),
            y.get_as::<VMString>(nogc, known.string_map),
        ) {
            return Ok(sx.as_slice(nogc) >= sy.as_slice(nogc));
        }
        let a = Convert::to_number(nogc, x)?;
        let b = Convert::to_number(nogc, y)?;
        Ok(a >= b)
    }
}
