//! ES 23.1 + 23.1.5: the Array constructor, Array.isArray,
//! Array.prototype.values/[@@iterator], and the array iterator.

use vm_core::HostCtx;
use vm_core::{
    Args, Convert, DETAILS_CONFIGURABLE, DenseString, Handle, Heap, Object, Prototype, Smi, Tagged,
    ThreadState, VM, Value, VmError,
};
use vm_core::{raise_runtime, rt_try};

/// `Array(...)`: call and construct behave the same (ES 23.1.1.1). No
/// arguments → `[]`; one non-negative Smi → that many holes (negative or
/// non-integer numbers are a RangeError); otherwise the arguments are the
/// elements.
pub fn array_constructor<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let argv: Vec<Tagged<'_, Value>> = args.iter(heap).skip(1).collect();
        let single_len = match argv.as_slice() {
            [v] => match Smi::decode(v.raw()) {
                Some(s) if s.value() >= 0 => Some(usize::try_from(s.value()).unwrap_or(usize::MAX)),
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
            Object::promote_holey(heap, &scope, &array);
        }
        array.as_tagged(heap).erase()
    })
}

/// `Array.prototype.push` (ES 23.1.3.21): append the arguments in order,
/// growing the elements store; returns the new length.
pub fn array_push<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    // The overwhelmingly common shape — `a.push(x)` where `a` has headroom —
    // is handled by a tiny leaf so the general path's frame and spills never
    // touch it.
    if let Some(v) = array_push_fast(heap, args) {
        return v;
    }
    array_push_impl(vm, heap, state, args)
}

/// One argument, receiver already an array with spare capacity: append with
/// no allocation and no handle scope. Returns `None` for anything else.
#[inline(always)]
fn array_push_fast<'a>(heap: &Heap, args: Args) -> Option<Tagged<'a, Value>> {
    if args.len() != 2 {
        return None;
    }
    let receiver = args.get_handle(heap, 0);
    let arg = args.get_handle(heap, 1);
    let r = receiver.as_tagged(heap);
    let obj = r.as_heap_object()?;
    if !obj.as_ref().is_array(heap) {
        return None;
    }
    let len = obj.as_ref().length();
    let elements = obj.as_ref().elements_array(heap)?;
    if len >= elements.len() {
        return None;
    }
    elements.as_ref().set(heap, len, arg.as_tagged(heap));
    obj.as_ref()
        .length
        .set(heap, r.erase(), Smi::new((len + 1) as i64));
    Prototype::element_mutated(heap, obj);
    Some(Smi::new((len + 1) as i64).into_tagged())
}

#[cold]
#[inline(never)]
fn array_push_impl<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ThreadState,
    args: Args,
) -> Tagged<'a, Value> {
    let receiver = args.get_handle(heap, 0);
    let mut len = {
        let r = receiver.as_tagged(heap);
        let Some(obj) = r.as_heap_object() else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        if !obj.as_ref().is_array(heap) {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        obj.as_ref().length()
    };
    for i in 1..args.len() {
        let arg = args.get_handle(heap, i);
        let in_place = {
            let r = receiver.as_tagged(heap);
            let obj = r.as_heap_object().expect("validated above");
            match obj.as_ref().elements_array(heap) {
                Some(elements) if len < elements.len() => {
                    elements.as_ref().set(heap, len, arg.as_tagged(heap));
                    obj.as_ref()
                        .length
                        .set(heap, r.erase(), Smi::new((len + 1) as i64));
                    Prototype::element_mutated(heap, obj);
                    true
                }
                _ => false,
            }
        };
        if !in_place {
            rt_try!(
                vm,
                heap,
                state,
                array_push_grow(heap, state, &receiver, len, &arg)
            );
        }
        len += 1;
    }
    Smi::new(len as i64).into_tagged()
}

/// The rare grow path, kept out of line so `array_push`'s hot in-place loop
/// keeps a small frame and register set. Growth may allocate, so the
/// receiver is rooted here first.
#[cold]
#[inline(never)]
fn array_push_grow<'a>(
    heap: &'a mut Heap,
    state: &'a ThreadState,
    receiver: &Handle<'_, Value>,
    len: usize,
    arg: &Handle<'_, Value>,
) -> Result<(), VmError> {
    state.handle_scope(|scope| {
        let obj = receiver
            .as_tagged(heap)
            .as_heap_object()
            .ok_or(VmError::Type)?;
        Object::store_array_element(heap, &scope, &scope.handle(obj), len, arg)
    })
}

/// `Array.prototype.pop` (ES 23.1.3.20): remove the last element, shorten
/// `length`, and punch a hole so the store releases the value.
pub fn array_pop<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get_handle(heap, 0);
        let Some(obj) = scope.cast::<Object>(heap, receiver.as_tagged(heap)) else {
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
        if let Some(dict) = this.as_ref().element_dictionary(heap) {
            dict.as_ref().delete(heap, last);
        } else {
            let elements = this.as_ref().elements.get(heap);
            if elements.is_strong_ptr() && last < elements.as_ref().len() {
                elements
                    .as_ref()
                    .set(heap, last, heap.known().the_hole.as_tagged(heap).erase());
            }
        }
        this.as_ref()
            .length
            .set(heap, this.erase(), Smi::new(last as i64));
        Prototype::element_mutated(heap, this);
        value
    })
}

/// `Array.prototype.values` / `Array.prototype[@@iterator]` (ES 23.1.3.41):
/// returns a fresh array-iterator over the receiver (CreateArrayIterator).
pub fn array_values<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let (receiver, is_array) = {
        let receiver = args.get_handle(heap, 0);
        let receiver = receiver;
        let is_array = receiver
            .as_tagged(heap)
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
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let receiver = args.get_handle(heap, 0);
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
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm: _,
        heap,
        state: _,
        ..
    } = nctx;
    let arg = args.get(heap, 0);
    arg
}

/// `Array.prototype.join(separator)` (ES 23.1.3.15): ToString each element
/// in index order, separated by `separator` (default `","`). Holes,
/// `undefined`, and `null` render as the empty string.
pub fn array_join<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let receiver = args.get_handle(heap, 0);
    let separator = (args.len() > 1).then(|| args.get_handle(heap, 1));
    join_impl(vm, heap, state, receiver, separator)
}

/// `Array.prototype.toString` (ES 23.1.3.37): `join` with the default
/// separator; any arguments are ignored.
pub fn array_to_string<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let receiver = args.get_handle(heap, 0);
    join_impl(vm, heap, state, receiver, None)
}

fn join_impl<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ThreadState,
    receiver: Handle<'_, Value>,
    separator: Option<Handle<'_, Value>>,
) -> Tagged<'a, Value> {
    state.handle_scope(|scope| {
        let is_array = receiver
            .as_tagged(heap)
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        if !is_array {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        // separator = ToString(separator); absent/undefined defaults to ","
        let sep = match separator {
            Some(h) if h.as_tagged(heap) != heap.known().undefined.as_tagged(heap) => {
                match Object::to_string(vm, heap, state, h) {
                    Ok(Some(s)) => {
                        let word = s.raw();
                        // Safety: fresh string word, no allocation since the read.
                        unsafe { word.assume_valid(heap) }
                            .get_as::<DenseString>(heap)
                            .map(|d| d.to_rust_string(heap))
                            .unwrap_or_default()
                    }
                    Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
                    Err(err) => return raise_runtime(vm, heap, state, err),
                }
            }
            _ => ",".to_string(),
        };
        // length is read once, then elements are Get in index order
        let len = receiver
            .as_tagged(heap)
            .as_heap_object()
            .map(|o| o.as_ref().length())
            .unwrap_or(0);
        let mut out = String::new();
        for i in 0..len {
            if i > 0 {
                out.push_str(&sep);
            }
            let Some(v) = receiver
                .as_tagged(heap)
                .as_heap_object()
                .and_then(|o| o.as_ref().element_value(heap, i))
            else {
                continue;
            };
            if v == heap.known().undefined.as_tagged(heap) || v == heap.known().null.as_tagged(heap)
            {
                continue;
            }
            let h = scope.handle(v);
            match Object::to_string(vm, heap, state, h) {
                Ok(Some(s)) => {
                    let word = s.raw();
                    // Safety: fresh string word, no allocation since the read.
                    if let Some(d) = unsafe { word.assume_valid(heap) }.get_as::<DenseString>(heap)
                    {
                        out.push_str(&d.to_rust_string(heap));
                    }
                }
                Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
                Err(err) => return raise_runtime(vm, heap, state, err),
            }
        }
        DenseString::from_utf8(heap, &scope, &out)
            .as_tagged(heap)
            .erase()
    })
}

pub fn array_is_array<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm: _,
        heap,
        state: _,
        ..
    } = nctx;
    let arg = args.get(heap, 1);
    let is_array = arg
        .as_heap_object()
        .is_some_and(|o| o.as_ref().is_array(heap));
    Convert::boolean(heap, is_array)
}

/// The array `length` accessor getter: reads the internal slot (the
/// descriptor exists so the generic lookup finds `length`; the fast load
/// path never reaches this).
pub fn array_length_get<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm: _,
        heap,
        state: _,
        ..
    } = nctx;
    let receiver = args.get_handle(heap, 0);
    receiver
        .as_tagged(heap)
        .as_heap_object()
        .map(|o| Smi::new(o.as_ref().length() as i64).into_tagged())
        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase())
}

/// The array `length` accessor setter (ES 10.4.2.3 ArraySetLength):
/// ToNumber, then the uint32 truncation must equal it exactly (else a
/// RangeError — approximated by `OutOfBounds`); shrinking punches holes
/// in the dropped elements.
pub fn array_length_set<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get_handle(heap, 0);
        let value = args.get(heap, 1);
        let Some(obj) = receiver.as_tagged(heap).as_heap_object() else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        if !obj.as_ref().is_array(heap) {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        let obj = scope.handle(obj);
        let value = scope.handle(value);
        let n = match Object::to_numeric(vm, heap, state, value) {
            Ok(Some(n)) => n,
            Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        let new_len = Convert::number_to_uint32(n);
        if new_len as f64 != n {
            // spec: RangeError ("invalid array length")
            return raise_runtime(vm, heap, state, VmError::OutOfBounds);
        }
        let new_len = new_len as usize;
        if new_len > 32 * 1024 * 1024
            && obj
                .as_tagged(heap)
                .as_ref()
                .map_ref(heap)
                .as_ref()
                .kind()
                .is_dense_elements()
        {
            Object::normalize_elements(heap, &scope, &obj);
        }
        if let Some(dict) = obj.as_tagged(heap).as_ref().element_dictionary(heap) {
            let old_len = obj.as_tagged(heap).as_ref().length();
            if new_len < old_len {
                // non-configurable entries clamp the shrink
                let mut effective = new_len;
                let mut doomed: Vec<usize> = Vec::new();
                dict.as_ref().for_each_entry(heap, |k, _, details| {
                    if k >= new_len {
                        if details & DETAILS_CONFIGURABLE == 0 {
                            effective = effective.max(k + 1);
                        } else {
                            doomed.push(k);
                        }
                    }
                });
                for k in doomed {
                    dict.as_ref().delete(heap, k);
                }
                Prototype::element_mutated(heap, obj.as_tagged(heap));
                let obj_t = obj.as_tagged(heap);
                obj_t
                    .as_ref()
                    .length
                    .set(heap, obj_t.erase(), Smi::new(effective as i64));
                return heap.known().undefined.as_tagged(heap).erase();
            }
            let obj_t = obj.as_tagged(heap);
            obj_t
                .as_ref()
                .length
                .set(heap, obj_t.erase(), Smi::new(new_len as i64));
            return heap.known().undefined.as_tagged(heap).erase();
        }
        let old_len = obj.as_tagged(heap).as_ref().length();
        if new_len < old_len {
            let hole = heap.known().the_hole.as_tagged(heap).erase();
            if let Some(elements) = obj.as_tagged(heap).as_ref().elements_array(heap) {
                for i in new_len..old_len.min(elements.len()) {
                    elements.as_ref().set(heap, i, hole);
                }
            }
            // a later length grow re-exposes the punched holes
            Object::promote_holey(heap, &scope, &obj);
            Prototype::element_mutated(heap, obj.as_tagged(heap));
        } else if new_len > old_len {
            // the indices [old_len, new_len) are holes: the map must stop
            // promising packed elements (and packed capacity covers length)
            Object::promote_holey(heap, &scope, &obj);
        }
        let obj_t = obj.as_tagged(heap);
        obj_t
            .as_ref()
            .length
            .set(heap, obj_t.erase(), Smi::new(new_len as i64));
        heap.known().undefined.as_tagged(heap).erase()
    })
}

/// A `slice` boundary: absent/undefined → `default`, Smi passes straight
/// through, otherwise ToNumber and truncate (NaN → 0). `Ok(None)` is a
/// pending exception.
fn slice_bound(
    vm: &VM,
    heap: &mut Heap,
    state: &ThreadState,
    arg: Option<Handle<'_, Value>>,
    default: f64,
) -> Result<Option<f64>, VmError> {
    let Some(h) = arg else {
        return Ok(Some(default));
    };
    let v = h.as_tagged(heap);
    if v == heap.known().undefined.as_tagged(heap) {
        return Ok(Some(default));
    }
    if let Some(smi) = Smi::decode(v.raw()) {
        return Ok(Some(smi.value() as f64));
    }
    match Object::to_numeric(vm, heap, state, h)? {
        Some(n) => Ok(Some(if n.is_nan() { 0.0 } else { n.trunc() })),
        None => Ok(None),
    }
}

/// `Array.prototype.slice(start, end)` (ES 23.1.3.34): a new array with
/// the `[start, end)` elements, holes preserved. An absent/undefined bound
/// is the fast path (no coercion; `slice()` clones the whole backing
/// store).
pub fn array_slice<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get_handle(heap, 0);
        let Some(obj) = receiver.as_tagged(heap).as_heap_object() else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        if !obj.as_ref().is_array(heap) {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        let obj = scope.handle(obj);
        let len = obj.as_tagged(heap).as_ref().length() as f64;

        let start = match slice_bound(
            vm,
            heap,
            state,
            (args.len() > 1).then(|| args.get_handle(heap, 1)),
            0.0,
        ) {
            Ok(Some(v)) => v,
            Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        let end = match slice_bound(
            vm,
            heap,
            state,
            (args.len() > 2).then(|| args.get_handle(heap, 2)),
            len,
        ) {
            Ok(Some(v)) => v,
            Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
            Err(err) => return raise_runtime(vm, heap, state, err),
        };
        let start = if start < 0.0 {
            (len + start).max(0.0)
        } else {
            start.min(len)
        };
        let end = if end < 0.0 {
            (len + end).max(0.0)
        } else {
            end.min(len)
        };
        let count = (end - start).max(0.0) as usize;
        let start = start as usize;

        // one gather pass, then a single bulk FixedArray allocation
        let hole = heap.known().the_hole.as_tagged(heap).erase().raw();
        let mut values: Vec<Tagged<'_, Value>> = Vec::with_capacity(count);
        let mut has_hole = false;
        for i in start..start + count {
            match obj.as_tagged(heap).as_ref().element_value(heap, i) {
                Some(v) => values.push(v),
                None => {
                    // Safety: old-gen singleton word.
                    values.push(unsafe { Tagged::<Value>::from_value_unchecked(hole) });
                    has_hole = true;
                }
            }
        }
        let staged = scope.stage(&values);
        let arr = heap.new_array(&scope, staged).as_handle(&scope);
        if has_hole {
            Object::promote_holey(heap, &scope, &arr);
        }
        arr.as_tagged(heap).erase()
    })
}

/// `Array.prototype.sort(comparefn)` (ES 23.1.3.30): in-place, stable.
/// Holes move last, `undefined` before them without invoking `comparefn`;
/// the default comparator is the elements' `ToString` order. Returns the
/// receiver.
pub fn array_sort<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get_handle(heap, 0);
        let Some(obj) = receiver.as_tagged(heap).as_heap_object() else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        if !obj.as_ref().is_array(heap) {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        let obj = scope.handle(obj);
        // an explicit `undefined` comparator means the default one
        let cmpfn = if args.len() > 1 {
            let c = args.get(heap, 1);
            if c != heap.known().undefined.as_tagged(heap) {
                if !Object::is_callable(heap, c) {
                    return raise_runtime(vm, heap, state, VmError::Type);
                }
                Some(scope.handle(c))
            } else {
                None
            }
        } else {
            None
        };
        let undefined = heap.known().undefined.as_tagged(heap).erase().raw();
        let exception = heap.known().exception.as_tagged(heap).erase().raw();
        let len = obj.as_tagged(heap).as_ref().length();

        // Root every element up front. The sort can allocate (`ToString`
        // for the default keys, user code for a comparator); a collection
        // would invalidate raw words, but handles are updated in place by
        // the GC, so they stay valid throughout.
        let mut elems: Vec<Handle<'_, Value>> = Vec::new();
        let mut undefined_count = 0usize;
        for i in 0..len {
            match obj.as_tagged(heap).as_ref().element_value(heap, i) {
                None => {}
                Some(v) if v.raw() == undefined => undefined_count += 1,
                Some(v) => elems.push(scope.handle(v)),
            }
        }
        let count = elems.len();

        match &cmpfn {
            None => {
                // default comparator. When every element is already a
                // string, compare the rooted DenseStrings in place — no key
                // materialization at all.
                let all_strings = elems
                    .iter()
                    .all(|h| h.as_tagged(heap).get_as::<DenseString>(heap).is_some());
                if all_strings {
                    elems.sort_by(|a, b| {
                        cmp_dense_strings(heap, a.as_tagged(heap), b.as_tagged(heap))
                    });
                } else {
                    // materialize each element's ToString once (rooted),
                    // then sort the (value, key) pairs directly by key
                    let mut pairs: Vec<(Handle<'_, Value>, Handle<'_, Value>)> =
                        Vec::with_capacity(count);
                    for h in elems.iter() {
                        let key = match Object::to_string(vm, heap, state, *h) {
                            Ok(Some(s)) => s,
                            Ok(None) => {
                                // Safety: old-gen singleton word.
                                return unsafe { Tagged::<Value>::from_value_unchecked(exception) };
                            }
                            Err(err) => return raise_runtime(vm, heap, state, err),
                        };
                        let kh = scope
                            .handle(unsafe { Tagged::<Value>::from_value_unchecked(key.raw()) });
                        pairs.push((*h, kh));
                    }
                    pairs.sort_by(|a, b| {
                        cmp_dense_strings(heap, a.1.as_tagged(heap), b.1.as_tagged(heap))
                    });
                    elems = pairs.into_iter().map(|p| p.0).collect();
                }
            }
            Some(cmp) => {
                let threw = core::cell::Cell::new(false);
                elems.sort_by(|a, b| {
                    if threw.get() {
                        return core::cmp::Ordering::Equal;
                    }
                    // a fresh scope per comparison reclaims its slots
                    state.handle_scope(|cscope| {
                        let x = a.as_tagged(heap);
                        let y = b.as_tagged(heap);
                        let call_args = cscope.stage(&[
                            // Safety: old-gen singleton word.
                            unsafe { Tagged::<Value>::from_value_unchecked(undefined) },
                            x,
                            y,
                        ]);
                        let v = match HostCtx::enter(vm, heap, state, *cmp, call_args, None) {
                            Ok(v) => v,
                            Err(_) => {
                                threw.set(true);
                                return core::cmp::Ordering::Equal;
                            }
                        };
                        if v.raw() == exception {
                            threw.set(true);
                            return core::cmp::Ordering::Equal;
                        }
                        // Safety: fresh word, no allocation since the read.
                        let arg = unsafe { Tagged::<Value>::from_value_unchecked(v.raw()) };
                        match Object::to_numeric(vm, heap, state, cscope.handle(arg)) {
                            Ok(Some(n)) => {
                                if n < 0.0 {
                                    core::cmp::Ordering::Less
                                } else if n > 0.0 {
                                    core::cmp::Ordering::Greater
                                } else {
                                    core::cmp::Ordering::Equal
                                }
                            }
                            _ => {
                                threw.set(true);
                                core::cmp::Ordering::Equal
                            }
                        }
                    })
                });
                if threw.get() {
                    // Safety: old-gen singleton word.
                    return unsafe { Tagged::<Value>::from_value_unchecked(exception) };
                }
            }
        }

        // write back: a tight direct loop into the elements backing store
        // when it has room; per-element stores only when it must grow
        let total = count + undefined_count;
        let capacity = obj
            .as_tagged(heap)
            .as_ref()
            .elements_array(heap)
            .map(|e| e.len())
            .unwrap_or(0);
        if total <= capacity {
            let elements = obj
                .as_tagged(heap)
                .as_ref()
                .elements_array(heap)
                .expect("capacity > 0");
            for (j, h) in elems.iter().enumerate() {
                // Safety: rooted handle re-read under the shared borrow.
                let v = h.as_tagged(heap);
                elements.as_ref().set(heap, j, v);
            }
            for j in count..total {
                // Safety: old-gen singleton word.
                elements.as_ref().set(heap, j, unsafe {
                    Tagged::<Value>::from_value_unchecked(undefined)
                });
            }
        } else {
            for (j, h) in elems.iter().enumerate() {
                if let Err(e) = Object::store_array_element(heap, &scope, &obj, j, h) {
                    return raise_runtime(vm, heap, state, e);
                }
            }
            for j in count..total {
                let h = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(undefined) });
                if let Err(e) = Object::store_array_element(heap, &scope, &obj, j, &h) {
                    return raise_runtime(vm, heap, state, e);
                }
            }
        }
        // the remainder become holes (or removed dictionary entries)
        if total < len {
            if let Some(dict) = obj.as_tagged(heap).as_ref().element_dictionary(heap) {
                let mut doomed: Vec<usize> = Vec::new();
                dict.as_ref().for_each_entry(heap, |k, _, _| {
                    if k >= total {
                        doomed.push(k);
                    }
                });
                for k in doomed {
                    dict.as_ref().delete(heap, k);
                }
            } else {
                let hole = heap.known().the_hole.as_tagged(heap).erase().raw();
                if let Some(elements) = obj.as_tagged(heap).as_ref().elements_array(heap) {
                    for j in total..len.min(elements.len()) {
                        // Safety: old-gen singleton word.
                        elements.as_ref().set(heap, j, unsafe {
                            Tagged::<Value>::from_value_unchecked(hole)
                        });
                    }
                }
                Object::promote_holey(heap, &scope, &obj);
            }
        }
        Prototype::element_mutated(heap, obj.as_tagged(heap));
        obj.as_tagged(heap).erase()
    })
}

/// Lexicographic `DenseString` comparison over UTF-16 code units, in
/// place (no allocation). Non-strings compare equal (the caller only uses
/// this where both sides are known strings).
#[inline]
fn cmp_dense_strings(
    heap: &Heap,
    a: Tagged<'_, Value>,
    b: Tagged<'_, Value>,
) -> core::cmp::Ordering {
    match (a.get_as::<DenseString>(heap), b.get_as::<DenseString>(heap)) {
        (Some(x), Some(y)) => x.as_ref().data(heap).cmp(&y.as_ref().data(heap)),
        _ => core::cmp::Ordering::Equal,
    }
}
