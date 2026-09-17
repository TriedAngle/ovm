use crate::{Convert, DenseString, Float, Heap, Smi, Tagged, Value, VmError};

pub struct Compare;

impl Compare {
    pub fn strict_equal<'a>(heap: &'a Heap, x: Tagged<'a, Value>, y: Tagged<'a, Value>) -> bool {
        let x_num = x.is_smi() || x.get_as::<Float>().is_some();
        let y_num = y.is_smi() || y.get_as::<Float>().is_some();
        if x_num || y_num {
            // both must be numbers (Float === "1" is false, no parsing);
            // NaN is unequal to everything (even itself), -0 equals +0
            if !x_num || !y_num {
                return false;
            }
            let number_value = |v: Tagged<'_, Value>| match v.get_as::<Float>() {
                Some(f) => f.value.get(),
                None => v.raw().to_i64().unwrap() as f64,
            };
            let a = number_value(x);
            let b = number_value(y);
            if a.is_nan() || b.is_nan() {
                return false;
            }
            return a == b;
        }
        // identical bits: same interned string, same heap object
        if x.raw() == y.raw() {
            return true;
        }
        if let (Some(sx), Some(sy)) = (x.get_as::<DenseString>(), y.get_as::<DenseString>()) {
            return sx.as_ref().content_eq(heap, sy.as_ref());
        }
        false
    }

    /// ES IsLooselyEqual (==) for primitives.
    pub fn equal<'a>(
        heap: &'a Heap,
        x: Tagged<'a, Value>,
        y: Tagged<'a, Value>,
    ) -> Result<bool, VmError> {
        let known = heap.known();
        if Self::strict_equal(heap, x, y) {
            return Ok(true);
        }
        let nullish = |v: Tagged<'_, Value>| {
            v.raw() == known.null.as_tagged(heap).raw()
                || v.raw() == known.undefined.as_tagged(heap).raw()
        };
        if nullish(x) && nullish(y) {
            return Ok(true);
        }
        let is_bool = |v: Tagged<'_, Value>| {
            v.raw() == known.true_object.as_tagged(heap).raw()
                || v.raw() == known.false_object.as_tagged(heap).raw()
        };
        let is_string = |v: Tagged<'_, Value>| v.get_as::<DenseString>().is_some();
        let is_number = |v: Tagged<'_, Value>| v.is_smi() || v.get_as::<Float>().is_some();
        // number ↔ string: the string parses as a number
        if is_number(x) && is_string(y) {
            return Ok(Convert::to_number(heap, x)? == Convert::to_number(heap, y)?);
        }
        if is_string(x) && is_number(y) {
            return Ok(Convert::to_number(heap, x)? == Convert::to_number(heap, y)?);
        }
        // booleans become numbers (exactly representable as smis, no allocation)
        if is_bool(x) {
            let n = Smi::new(if x.raw() == known.true_object.as_tagged(heap).raw() {
                1
            } else {
                0
            })
            .into_tagged();
            return Self::equal(heap, n, y);
        }
        if is_bool(y) {
            let n = Smi::new(if y.raw() == known.true_object.as_tagged(heap).raw() {
                1
            } else {
                0
            })
            .into_tagged();
            return Self::equal(heap, x, n);
        }
        let is_object = |v: Tagged<'_, Value>| {
            !v.is_smi() && !is_bool(v) && !nullish(v) && !is_string(v) && !is_number(v)
        };
        if is_object(x) || is_object(y) {
            return Err(VmError::Type);
        }
        Ok(false)
    }

    /// ES SameValue (7.2.11): NaN equals NaN, +0 and -0 are distinct, and
    /// strings compare by content (identity for everything else).
    /// Unlike `strict_equal` (===); used by [[DefineOwnProperty]] validation.
    pub fn same_value<'a>(heap: &'a Heap, x: Tagged<'a, Value>, y: Tagged<'a, Value>) -> bool {
        if x.raw() == y.raw() {
            return true;
        }
        let x_num = x.is_smi() || x.get_as::<Float>().is_some();
        let y_num = y.is_smi() || y.get_as::<Float>().is_some();
        if x_num && y_num {
            let number_value = |v: Tagged<'_, Value>| match v.get_as::<Float>() {
                Some(f) => f.value.get(),
                None => v.raw().to_i64().unwrap() as f64,
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
        if let (Some(sx), Some(sy)) = (x.get_as::<DenseString>(), y.get_as::<DenseString>()) {
            return sx.as_ref().content_eq(heap, sy.as_ref());
        }
        false
    }

    /// Relational comparison for two string values: UTF-16 code-unit
    /// order (ES 7.2.13 IsLessThan), encoding-agnostic.
    fn string_cmp<'a>(
        heap: &'a Heap,
        x: Tagged<'a, Value>,
        y: Tagged<'a, Value>,
    ) -> Option<core::cmp::Ordering> {
        let sx = x.get_as::<DenseString>()?.as_ref();
        let sy = y.get_as::<DenseString>()?.as_ref();
        Some(sx.data(heap).cmp(&sy.data(heap)))
    }

    pub fn less_than<'a>(
        heap: &'a Heap,
        x: Tagged<'a, Value>,
        y: Tagged<'a, Value>,
    ) -> Result<bool, VmError> {
        if let Some(ord) = Self::string_cmp(heap, x, y) {
            return Ok(ord == core::cmp::Ordering::Less);
        }
        let a = Convert::to_number(heap, x)?;
        let b = Convert::to_number(heap, y)?;
        Ok(a < b)
    }

    pub fn less_than_or_equal<'a>(
        heap: &'a Heap,
        x: Tagged<'a, Value>,
        y: Tagged<'a, Value>,
    ) -> Result<bool, VmError> {
        if let Some(ord) = Self::string_cmp(heap, x, y) {
            return Ok(ord != core::cmp::Ordering::Greater);
        }
        let a = Convert::to_number(heap, x)?;
        let b = Convert::to_number(heap, y)?;
        Ok(a <= b)
    }

    pub fn greater_than<'a>(
        heap: &'a Heap,
        x: Tagged<'a, Value>,
        y: Tagged<'a, Value>,
    ) -> Result<bool, VmError> {
        if let Some(ord) = Self::string_cmp(heap, x, y) {
            return Ok(ord == core::cmp::Ordering::Greater);
        }
        let a = Convert::to_number(heap, x)?;
        let b = Convert::to_number(heap, y)?;
        Ok(a > b)
    }

    pub fn greater_than_or_equal<'a>(
        heap: &'a Heap,
        x: Tagged<'a, Value>,
        y: Tagged<'a, Value>,
    ) -> Result<bool, VmError> {
        if let Some(ord) = Self::string_cmp(heap, x, y) {
            return Ok(ord != core::cmp::Ordering::Less);
        }
        let a = Convert::to_number(heap, x)?;
        let b = Convert::to_number(heap, y)?;
        Ok(a >= b)
    }
}
