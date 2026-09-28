//! ES 23.1 + 23.1.5: the Array constructor, Array.isArray,
//! Array.prototype.values/[@@iterator], and the array iterator.

use vm_core::RuntimeContext;
use vm_core::{Convert, HandleSlice, Object, Smi, Tagged, Value, VmError};
use vm_core::{raise_runtime, rt_try};

/// `Array(...)`: call and construct behave the same (ES 23.1.1.1). No
/// arguments → `[]`; one non-negative Smi → that many holes (negative or
/// non-integer numbers are a RangeError); otherwise the arguments are the
/// elements.
pub fn array_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let argv: Vec<Tagged<'_, Value>> = args.iter().map(|h| h.as_tagged(heap)).skip(1).collect();
        let single_len = match argv.as_slice() {
            [v] => match Smi::decode(v.raw()) {
                Some(s) if s.value() >= 0 => {
                    Some(usize::try_from(s.value()).unwrap_or(usize::MAX))
                }
                // Array(-1): RangeError
                Some(_) => return raise_runtime(vm, heap, state, VmError::OutOfBounds),
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
        let array = heap.new_array(&scope, staged).as_handle(&scope);
        if single_len.is_some() {
            // `Array(n)` is a hole-filled array: retire the packed promise
            let obj = array.as_tagged(heap);
            obj.as_ref().mark_holey(heap);
        }
        array.as_tagged(heap).erase()
    })
}

/// `Array.prototype.push` (ES 23.1.3.21): append the arguments in order,
/// growing the elements store; returns the new length.
pub fn array_push<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let Some(receiver) = args.get(0) else {
            return raise_runtime(vm, heap, state, VmError::Arity);
        };
        let Some(obj) = scope.cast::<Object>(receiver.as_tagged(heap)) else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        if !obj.as_tagged(heap).as_ref().is_array(heap) {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        let mut len = obj.as_tagged(heap).as_ref().length();
        for arg in args.iter().skip(1) {
            rt_try!(vm, heap, state, Object::store_array_element(heap, &scope, &obj, len, &arg));
            len += 1;
        }
        Smi::new(len as i64).into_tagged()
    })
}

/// `Array.prototype.pop` (ES 23.1.3.20): remove the last element, shorten
/// `length`, and punch a hole so the store releases the value.
pub fn array_pop<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let Some(receiver) = args.get(0) else {
            return raise_runtime(vm, heap, state, VmError::Arity);
        };
        let Some(obj) = scope.cast::<Object>(receiver.as_tagged(heap)) else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        let is_array = obj.as_tagged(heap).as_ref().is_array(heap);
        if !is_array {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        let this = obj.as_tagged(heap);
        let len = this.as_ref().length();
        if len == 0 {
            return heap.known().undefined.as_tagged(heap).erase();
        }
        let last = len - 1;
        let value = this
            .as_ref()
            .element_value(heap, last)
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let elements = this.as_ref().elements.get(heap);
        if elements.is_strong_ptr() && last < elements.as_ref().len() {
            elements
                .as_ref()
                .set(heap, last, heap.known().the_hole.as_tagged(heap).erase());
        }
        this.as_ref()
            .length
            .set(heap, this.erase(), Smi::new(last as i64));
        value
    })
}

/// `Array.prototype.values` / `Array.prototype[@@iterator]` (ES 23.1.3.41):
/// returns a fresh array-iterator over the receiver (CreateArrayIterator).
pub fn array_values<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    let (receiver, is_array) = {
        let Some(receiver) = args.get(0) else {
            return raise_runtime(vm, heap, state, VmError::Arity);
        };
        let receiver = receiver.as_tagged(heap);
        let is_array = receiver
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        (receiver.raw(), is_array)
    };
    if !is_array {
        // Array.prototype[Symbol.iterator] called on a non-array: per spec
        // the iterator operates on any array-like via length + index gets;
        // only real arrays are supported here
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    state.handle_scope(|scope| {
        let map = heap.known().array_iterator_map;
        // Safety: fresh argument word (no allocation since the read).
        let recv = unsafe { Tagged::<Value>::from_value_unchecked(receiver) };
        heap.new_object(&scope, map, scope.stage(&[recv, Smi::new(0).into_tagged()]))
            .erase()
    })
}

/// `%ArrayIteratorPrototype%.next` (ES 23.1.5.2.1): one step over the
/// iterated array, producing `{ value, done }`.
pub fn array_iterator_next<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    let Some(receiver) = args.get(0) else {
        return raise_runtime(vm, heap, state, VmError::Arity);
    };
    state.handle_scope(|scope| {
        let (array, index) = {
            let Some(obj) = receiver.as_tagged(heap).as_heap_object() else {
                return raise_runtime(vm, heap, state, VmError::Type);
            };
            let slots = obj.as_ref().slots.get(heap);
            if slots.len() < 2 {
                return raise_runtime(vm, heap, state, VmError::Type);
            }
            (scope.handle(slots.at(heap, 0)), slots.at(heap, 1).raw())
        };
        let Some(index) = Smi::decode(index) else {
            return raise_runtime(vm, heap, state, VmError::Type);
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
            let Some(obj) = receiver.as_tagged(heap).as_heap_object() else {
                return raise_runtime(vm, heap, state, VmError::Type);
            };
            let slots = obj.as_ref().slots.get(heap);
            slots.set(heap, 1, Smi::new(index.value() + 1).into_tagged());
        };
        let map = heap.known().iterator_result_map;
        heap.new_object(
                &scope,
                map,
                scope.stage(&[
                    value.as_tagged(heap).erase(),
                    done_value.as_tagged(heap).erase(),
                ]),
            )
            .erase()
    })
}

/// `%ArrayIteratorPrototype%[@@iterator]`: returns the receiver.
pub fn array_iterator_symbol_iterator<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    let Some(arg) = args.get(0) else {
        return raise_runtime(vm, heap, state, VmError::Arity);
    };
    arg.as_tagged(heap)
}

/// `Array.isArray(arg)` (ES 24.1.2.1).
pub fn array_is_array<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    let arg = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
    let is_array = arg
        .as_heap_object()
        .is_some_and(|o| o.as_ref().is_array(heap));
    Convert::boolean(heap, is_array)
}
