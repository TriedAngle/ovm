//! ES 21.4 (Date): the minimum the Octane harness needs — `new Date()`
//! and `Date.now()` for wall-clock milliseconds, plus `valueOf` so
//! arithmetic operators coerce instances back to numbers
//! (`new Date() - start`).

use vm_core::Heap;
use vm_core::Object;
use vm_core::RuntimeContext;
use vm_core::{Convert, DenseString, HandleSlice, Tagged, Value, VmError};
use vm_core::{ContextState, VM};
use vm_core::{raise_runtime, rt_try};

/// Milliseconds since the Unix epoch.
fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// `Date(...)` / `new Date(...)`: no arguments → now; one numeric
/// argument → that many epoch milliseconds; anything richer stays
/// unimplemented.
pub fn date_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let is_construct = nctx.is_construct();
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let ms = match args.get(1) {
        None => now_millis(),
        Some(arg) => {
            let ms = rt_try!(vm, heap, state, state.handle_scope(|scope| {
                let arg = scope.handle(arg.as_tagged(heap));
                Object::to_numeric(vm, heap, state, arg)
            }));
            let Some(ms) = ms else {
                return heap.known().exception.as_tagged(heap).erase();
            };
            ms
        }
    };

    if !is_construct {
        // call form answers the current time in decimal (a stand-in for
        // the full ES 21.4.2.2 format)
        return state.handle_scope(|scope| {
            let s = DenseString::from_utf8(heap, &scope, &format!("{}", now_millis() as i64));
            s.as_tagged(heap).erase()
        });
    }

    state.handle_scope(|scope| {
        let map = heap.known().date_instance_map;
        let value = scope.handle(heap.new_number(ms));
        heap
            .new_object(&scope, map, scope.stage(&[value.as_tagged(heap).erase()]))
            .erase()
    })
}

/// `Date.now()` (ES 21.4.2.2): epoch milliseconds as a Number.
pub fn date_now<'a>(
    nctx: RuntimeContext<'a>,
    _args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    nctx.heap.new_number(now_millis())
}

/// `Date.prototype.valueOf` (ES 21.4.4.40): the wrapped epoch
/// milliseconds. Only real Date instances qualify.
pub fn date_value_of<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    let Some(receiver) = args.get(0) else {
        return raise_runtime(vm, heap, state, VmError::Arity);
    };
    date_slot(
        vm,
        heap,
        state,
        // Safety: fresh rooted-slot word, no allocation since the read.
        unsafe { Tagged::<Value>::from_value_unchecked(receiver.raw()) },
    )
}

/// `Date.prototype.toString`: the milliseconds in decimal (a stand-in
/// for the full ES 21.4.4.41 format).
pub fn date_to_string<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext { vm, heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let Some(receiver) = args.get(0) else {
            return raise_runtime(vm, heap, state, VmError::Arity);
        };
        let ms = scope.handle(date_slot(
            vm,
            heap,
            state,
            // Safety: fresh rooted-slot word, no allocation since the read.
            unsafe { Tagged::<Value>::from_value_unchecked(receiver.raw()) },
        ));
        match Convert::to_string(heap, &scope, ms).map(|v| v.raw()) {
        Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
        Err(err) => return raise_runtime(vm, heap, state, err),
    }
    })
}

/// slots[0] of a Date instance (the milliseconds), or a TypeError.
fn date_slot<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    receiver: Tagged<'a, Value>,
) -> Tagged<'a, Value> {
    let Some(obj) = receiver.as_heap_object() else {
        return raise_runtime(vm, heap, state, VmError::Type);
    };
    let map = obj.as_ref().header.map.get(heap);
    if map != heap.known().date_instance_map.as_tagged(heap) {
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    obj.as_ref().slots.get(heap).at(heap, 0)
}
