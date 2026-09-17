//! ES 23.1 + 23.1.5: the Array constructor, Array.isArray,
//! Array.prototype.values/[@@iterator], and the array iterator.

use crate::{Convert, GcSlice, Smi, Value, VmError};

/// `Array(...)`: call and construct behave the same (ES 23.1.1.1). No
/// arguments → `[]`; one non-negative Smi → that many holes (negative or
/// non-integer numbers are a RangeError); otherwise the arguments are the
/// elements.
pub(crate) fn array_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let argv: Vec<Value> = (1..args.len()).filter_map(|i| args.get(i)).collect();
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
        let hole = heap.known().the_hole.value();
        let (values, _length) = match single_len {
            Some(n) => (vec![hole; n], n),
            None => {
                let n = argv.len();
                (argv, n)
            }
        };
        let staged = scope.stage(&values);
        Ok(heap.new_array(&scope, staged).into_tagged().erase())
    })
}

/// `Array.prototype.values` / `Array.prototype[@@iterator]` (ES 23.1.3.41):
/// returns a fresh array-iterator over the receiver (CreateArrayIterator).
pub(crate) fn array_values(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let is_array = nctx.heap().no_gc(|nogc| {
        receiver
            .as_heap_object(nogc)
            .is_some_and(|o| o.as_ref().is_array(nogc))
    });
    if !is_array {
        // Array.prototype[Symbol.iterator] called on a non-array: per spec
        // the iterator operates on any array-like via length + index gets;
        // only real arrays are supported here
        return Err(VmError::Type);
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().array_iterator_map;
        let zero = Smi::new(0).encode();
        Ok(heap
            .new_object(&scope, map, scope.stage(&[receiver, zero]))
            .into_tagged()
            .erase())
    })
}

/// `%ArrayIteratorPrototype%.next` (ES 23.1.5.2.1): one step over the
/// iterated array, producing `{ value, done }`.
pub(crate) fn array_iterator_next(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let (array, index) = heap.no_gc(|nogc| {
            let Some(obj) = receiver.as_heap_object(nogc) else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(nogc);
            if slots.len() < 2 {
                return Err(VmError::Type);
            }
            Ok((slots.at(0), slots.at(1)))
        })?;
        let Some(index) = Smi::decode(index) else {
            return Err(VmError::Type);
        };
        let done = {
            let len = heap.no_gc(|nogc| {
                array
                    .as_heap_object(nogc)
                    .map(|o| o.as_ref().length())
                    .unwrap_or(0)
            });
            index.value() as usize >= len
        };
        let (value, done_value) = if done {
            (
                heap.known().undefined.value(),
                heap.known().true_object.value(),
            )
        } else {
            // element reads see holes as undefined
            let v = heap.no_gc(|nogc| {
                array
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, index.value() as usize))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            });
            (v, heap.known().false_object.value())
        };
        // advance the index slot
        heap.no_gc(|nogc| {
            let Some(obj) = receiver.as_heap_object(nogc) else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(nogc);
            slots.set(nogc, 1, Smi::new(index.value() + 1).encode());
            Ok(())
        })?;
        let map = heap.known().iterator_result_map;
        Ok(heap
            .new_object(&scope, map, scope.stage(&[value, done_value]))
            .into_tagged()
            .erase())
    })
}

/// `%ArrayIteratorPrototype%[@@iterator]`: returns the receiver.
pub(crate) fn array_iterator_symbol_iterator(
    _nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    args.get(0).ok_or(VmError::Arity)
}

/// `Array.isArray(arg)` (ES 24.1.2.1).
pub(crate) fn array_is_array(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).ok_or(VmError::Arity)?;
    let is_array = nctx.heap().no_gc(|nogc| {
        arg.as_heap_object(nogc)
            .is_some_and(|o| o.as_ref().is_array(nogc))
    });
    Ok(Convert::boolean(nctx.heap(), is_array))
}
