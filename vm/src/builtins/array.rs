//! ES 23.1 + 23.1.5: the Array constructor, Array.isArray,
//! Array.prototype.values/[@@iterator], and the array iterator.

use crate::{Convert, GcSlice, Smi, Tagged, Value, VmError};

/// `Array(...)`: call and construct behave the same (ES 23.1.1.1). No
/// arguments → `[]`; one non-negative Smi → that many holes (negative or
/// non-integer numbers are a RangeError); otherwise the arguments are the
/// elements.
pub fn array_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let argv: Vec<Value> = nctx.heap().no_gc(|heap| {
        (1..args.len())
            .filter_map(|i| args.get(heap, i).map(|v| v.erase()))
            .collect()
    });
    let single_len = match argv.as_slice() {
        [v] => match Smi::decode(*v) {
            Some(s) if s.value() >= 0 => {
                Some(usize::try_from(s.value()).map_err(|_| VmError::OutOfBounds)?)
            }
            Some(_) => return Err(VmError::OutOfBounds), // Array(-1): RangeError
            None => None,
        },
        _ => None,
    };
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        // Safety: fresh root-slot word read for the immediate staging.
        let hole = unsafe { heap.known().the_hole.read_unchecked() };
        let (values, _length) = match single_len {
            Some(n) => (vec![hole; n], n),
            None => {
                let n = argv.len();
                (argv, n)
            }
        };
        let staged = scope.stage_words(&values);
        Ok(heap.new_array(&scope, staged).erase_type().erase())
    })
}

/// `Array.prototype.values` / `Array.prototype[@@iterator]` (ES 23.1.3.41):
/// returns a fresh array-iterator over the receiver (CreateArrayIterator).
pub fn array_values(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let (receiver, is_array) = nctx.heap().no_gc(|heap| {
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        let is_array = receiver
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        Ok::<_, VmError>((receiver.erase(), is_array))
    })?;
    if !is_array {
        // Array.prototype[Symbol.iterator] called on a non-array: per spec
        // the iterator operates on any array-like via length + index gets;
        // only real arrays are supported here
        return Err(VmError::Type);
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().array_iterator_map;
        // Safety: fresh argument word (no allocation since the read).
        let recv = unsafe { Tagged::<Value>::from_value_unchecked(receiver) };
        Ok(heap
            .new_object(&scope, map, scope.stage(&[recv, Smi::new(0).into_tagged()]))
            .erase_type()
            .erase())
    })
}

/// `%ArrayIteratorPrototype%.next` (ES 23.1.5.2.1): one step over the
/// iterated array, producing `{ value, done }`.
pub fn array_iterator_next(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    // Safety: fresh argument word; nothing below allocates before its
    // final re-reads under `no_gc` anchors.
    let receiver = nctx
        .heap()
        .no_gc(|heap| Ok(args.get(heap, 0).ok_or(VmError::Arity)?.erase()))?;
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let (array, index) = heap.no_gc(|heap| {
            let Some(obj) = unsafe { receiver.assume_valid(heap) }.as_heap_object() else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(heap);
            if slots.len() < 2 {
                return Err(VmError::Type);
            }
            Ok((slots.at(heap, 0).erase(), slots.at(heap, 1).erase()))
        })?;
        let Some(index) = Smi::decode(index) else {
            return Err(VmError::Type);
        };
        let done = {
            let len = heap.no_gc(|heap| {
                unsafe { array.assume_valid(heap) }
                    .as_heap_object()
                    .map(|o| o.as_ref().length())
                    .unwrap_or(0)
            });
            index.value() as usize >= len
        };
        let (value, done_value) = if done {
            (
                // Safety: fresh root-slot words read for the immediate staging.
                unsafe { heap.known().undefined.read_unchecked() },
                unsafe { heap.known().true_object.read_unchecked() },
            )
        } else {
            // element reads see holes as undefined
            let v = heap.no_gc(|heap| {
                unsafe { array.assume_valid(heap) }
                    .as_heap_object()
                    .and_then(|o| o.as_ref().element_value(heap, index.value() as usize))
                    .map(|v| v.erase())
                    .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase())
            });
            (
                v,
                // Safety: fresh root-slot word read for the immediate staging.
                unsafe { heap.known().false_object.read_unchecked() },
            )
        };
        // advance the index slot
        heap.no_gc(|heap| {
            let Some(obj) = unsafe { receiver.assume_valid(heap) }.as_heap_object() else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(heap);
            slots.set(heap, 1, Smi::new(index.value() + 1).into_tagged());
            Ok(())
        })?;
        let map = heap.known().iterator_result_map;
        Ok(heap
            .new_object(&scope, map, scope.stage_words(&[value, done_value]))
            .erase_type()
            .erase())
    })
}

/// `%ArrayIteratorPrototype%[@@iterator]`: returns the receiver.
pub fn array_iterator_symbol_iterator(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.heap()
        .no_gc(|heap| Ok(args.get(heap, 0).ok_or(VmError::Arity)?.erase()))
}

/// `Array.isArray(arg)` (ES 24.1.2.1).
pub fn array_is_array(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.heap().no_gc(|heap| {
        let is_array = args
            .get(heap, 1)
            .ok_or(VmError::Arity)?
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        Ok(Convert::boolean(heap, is_array).erase())
    })
}
