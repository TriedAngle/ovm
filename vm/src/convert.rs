use crate::{Float, Heap, NoGc, Smi, VMString, Value};

/// ES ToBoolean. Falsey: `false`, `undefined`, `null`, the hole, 0, -0, NaN,
/// everything else is truthy.
pub fn is_truthy<'a>(nogc: &'a NoGc<'a>, heap: &Heap, v: Value) -> bool {
    if let Some(smi) = Smi::decode(v) {
        return smi.value() != 0;
    }
    let known = heap.known();
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
