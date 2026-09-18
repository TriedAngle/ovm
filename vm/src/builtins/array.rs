//! ES 23.1 + 23.1.5: the Array constructor, Array.isArray,
//! Array.prototype.values/[@@iterator], and the array iterator.

use crate::natives::NativeContext;
use crate::{Convert, GcSlice, Smi, Tagged, Value, VmError};

/// `Array(...)`: call and construct behave the same (ES 23.1.1.1). No
/// arguments → `[]`; one non-negative Smi → that many holes (negative or
/// non-integer numbers are a RangeError); otherwise the arguments are the
/// elements.
pub fn array_constructor(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let argv: Vec<Tagged<'_, Value>> = args.iter(heap).skip(1).collect();
        let single_len = match argv.as_slice() {
            [v] => match Smi::decode(v.raw()) {
                Some(s) if s.value() >= 0 => {
                    Some(usize::try_from(s.value()).map_err(|_| VmError::OutOfBounds)?)
                }
                Some(_) => return Err(VmError::OutOfBounds), // Array(-1): RangeError
                None => None,
            },
            _ => None,
        };
        let hole = heap.known().the_hole.as_tagged(heap).erase();
        let (values, _length) = match single_len {
            Some(n) => (vec![hole; n], n),
            None => {
                let n = argv.len();
                (argv, n)
            }
        };

        let staged = scope.stage(&values);
        Ok(heap.new_array(&scope, staged).erase().raw())
    })
}

/// `Array.prototype.values` / `Array.prototype[@@iterator]` (ES 23.1.3.41):
/// returns a fresh array-iterator over the receiver (CreateArrayIterator).
pub fn array_values(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let (receiver, is_array) = {
        let heap = &*nctx.heap();
        let receiver = args.get(heap, 0).ok_or(VmError::Arity)?;
        let is_array = receiver
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        Ok::<_, VmError>((receiver.raw(), is_array))
    }?;
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
            .erase()
            .raw())
    })
}

/// `%ArrayIteratorPrototype%.next` (ES 23.1.5.2.1): one step over the
/// iterated array, producing `{ value, done }`.
pub fn array_iterator_next(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    // Safety: fresh argument word; nothing below allocates before its
    // final re-reads under heap-borrow anchors.
    let receiver = {
        let heap = &*nctx.heap();
        args.get(heap, 0).ok_or(VmError::Arity)?.raw()
    };
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let (array, index) = {
            let Some(obj) = unsafe { receiver.assume_valid(heap) }.as_heap_object() else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(heap);
            if slots.len() < 2 {
                return Err(VmError::Type);
            }
            Ok((scope.handle(slots.at(heap, 0)), slots.at(heap, 1).raw()))
        }?;
        let Some(index) = Smi::decode(index) else {
            return Err(VmError::Type);
        };
        let done = {
            let len = array
                .as_tagged(heap)
                .as_heap_object()
                .map(|o| o.as_ref().length())
                .unwrap_or(0);
            index.value() as usize >= len
        };
        let (value, done_value) = if done {
            (
                scope.handle(heap.known().undefined.as_tagged(heap).erase()),
                scope.handle(heap.known().true_object.as_tagged(heap).erase()),
            )
        } else {
            // element reads see holes as undefined
            let v = scope.handle(
                array
                    .as_tagged(heap)
                    .as_heap_object()
                    .and_then(|o| o.as_ref().element_value(heap, index.value() as usize))
                    .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase()),
            );
            (
                v,
                scope.handle(heap.known().false_object.as_tagged(heap).erase()),
            )
        };
        // advance the index slot
        {
            let Some(obj) = unsafe { receiver.assume_valid(heap) }.as_heap_object() else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(heap);
            slots.set(heap, 1, Smi::new(index.value() + 1).into_tagged());
            Ok(())
        }?;
        let map = heap.known().iterator_result_map;
        Ok(heap
            .new_object(
                &scope,
                map,
                scope.stage(&[
                    value.as_tagged(heap).erase(),
                    done_value.as_tagged(heap).erase(),
                ]),
            )
            .erase()
            .raw())
    })
}

/// `%ArrayIteratorPrototype%[@@iterator]`: returns the receiver.
pub fn array_iterator_symbol_iterator(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let heap = &*nctx.heap();
    Ok(args.get(heap, 0).ok_or(VmError::Arity)?.raw())
}

/// `Array.isArray(arg)` (ES 24.1.2.1).
pub fn array_is_array(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let heap = &*nctx.heap();
    let is_array = args
        .get(heap, 1)
        .ok_or(VmError::Arity)?
        .as_heap_object()
        .is_some_and(|o| o.as_ref().is_array(heap));
    Ok(Convert::boolean(heap, is_array).raw())
}
