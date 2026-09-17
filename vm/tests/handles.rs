use core::ptr::NonNull;

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{EdgeVisitable, HandleData, HandleScope, Heap, RawCell, Smi, Tagged, VM, Visitor};

struct Counter(usize);

impl Visitor for Counter {
    fn visit(&mut self, _cell: &RawCell) {
        self.0 += 1;
    }
}

fn root_count(data: &HandleData) -> usize {
    let mut counter = Counter(0);
    data.visit_edges(&mut counter);
    counter.0
}

fn handle_scope(data: &HandleData) -> HandleScope<'_> {
    unsafe { HandleScope::from_raw(NonNull::from(data)) }
}

/// A heap anchor: reading a handle back as a word requires a live
/// `&Heap` borrow (`Handle::as_tagged`).
fn anchor() -> (vm::VM, vm::Thread) {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let thread = vm.attach();
    (vm, thread)
}

fn smi_bits(heap: &Heap, handle: vm::Handle<'_, Smi>) -> i64 {
    Smi::decode(handle.as_tagged(heap).raw()).unwrap().value()
}

fn smi_handle(scope: &HandleScope<'_>, heap: &Heap, v: i64) -> i64 {
    let handle = scope.handle(Tagged::smi(v).unwrap());
    smi_bits(heap, handle)
}

#[test]
fn handles_read_back_their_values() {
    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    let heap = thread.heap();
    let scope = handle_scope(&data);

    let a = scope.handle(Tagged::smi(42).unwrap());
    let b = scope.handle(Tagged::smi(-7).unwrap());

    assert_eq!(smi_bits(heap, a), 42);
    assert_eq!(smi_bits(heap, b), -7);
    // +1: the block fill template is a visited root cell
    assert_eq!(root_count(&data), 3);
}

#[test]
fn scope_tracks_nesting_level() {
    let data = HandleData::new(Smi::new(0).encode());
    assert_eq!(data.level(), 0);
    let outer = handle_scope(&data);
    assert_eq!(data.level(), 1);
    {
        let _inner = handle_scope(&data);
        assert_eq!(data.level(), 2);
    }
    assert_eq!(data.level(), 1);
    drop(outer);
    assert_eq!(data.level(), 0);
}

#[test]
fn closed_scope_unroots_its_handles() {
    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    let outer = handle_scope(&data);
    let keep = outer.handle(Tagged::smi(1).unwrap());

    {
        let inner = handle_scope(&data);
        smi_handle(&inner, thread.heap(), 2);
        smi_handle(&inner, thread.heap(), 3);
        assert_eq!(root_count(&data), 4);
    }

    // inner scope closed: its slots are reclaimed, outer handle survives
    assert_eq!(root_count(&data), 2);
    assert_eq!(smi_bits(thread.heap(), keep), 1);
}

#[test]
fn reclaimed_slots_are_reused() {
    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    {
        let scope = handle_scope(&data);
        for i in 0..10 {
            smi_handle(&scope, thread.heap(), i);
        }
        assert_eq!(root_count(&data), 11);
    }
    assert_eq!(root_count(&data), 1);

    let scope = handle_scope(&data);
    for i in 0..5 {
        smi_handle(&scope, thread.heap(), i * 100);
    }
    assert_eq!(root_count(&data), 6);
    let fourth = scope.handle(Tagged::smi(400).unwrap());
    assert_eq!(smi_bits(thread.heap(), fourth), 400);
}

#[test]
fn blocks_extend_when_full() {
    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    let heap = thread.heap();
    let scope = handle_scope(&data);

    let mut handles = Vec::new();
    for i in 0..1030 {
        handles.push(scope.handle(Tagged::smi(i).unwrap()));
    }

    assert_eq!(root_count(&data), 1031);
    for (i, handle) in handles.iter().enumerate() {
        assert_eq!(smi_bits(heap, *handle), i as i64);
    }
}

#[test]
fn escaped_handle_survives_inner_scope() {
    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    let mut outer = handle_scope(&data);
    let keep = outer.handle(Tagged::smi(1).unwrap());
    assert_eq!(smi_bits(thread.heap(), keep), 1);

    let escaped = {
        let escapable = outer.escapable_scope();
        let h = escapable.handle(Tagged::smi(2).unwrap());
        escapable.handle(Tagged::smi(3).unwrap());
        escapable.escape(h)
    };

    assert_eq!(root_count(&data), 3);
    assert_eq!(smi_bits(thread.heap(), escaped), 2);
}

#[test]
fn escapable_scope_closed_without_escape_reclaims() {
    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    let mut outer = handle_scope(&data);
    let keep = outer.handle(Tagged::smi(1).unwrap());
    assert_eq!(smi_bits(thread.heap(), keep), 1);

    {
        let escapable = outer.escapable_scope();
        escapable.handle(Tagged::smi(2).unwrap());
    }

    assert_eq!(root_count(&data), 3);
}

#[test]
fn strong_handles_are_infallible() {
    use vm::{Object, Tagged};

    let data = HandleData::new(Smi::new(0).encode());
    let (_vm, mut thread) = anchor();
    let scope = handle_scope(&data);

    // smis root without ceremony
    let smi = scope.handle(Tagged::smi(7).unwrap());
    assert_eq!(smi_bits(thread.heap(), smi), 7);

    // strong-tagged values root without ceremony
    let _ = scope.handle(unsafe { Tagged::<Object>::from_value_unchecked(Smi::new(0).encode()) });
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "weak value in a strong Tagged")]
fn weak_bits_rejected_by_the_type_system() {
    use vm::{Object, Tagged, Value, WEAK_PTR};

    let data = HandleData::new(Smi::new(0).encode());
    let scope = handle_scope(&data);
    // Safety: bit-level test; the value is never dereferenced.
    let weak = Value::from_bits(0x1000 | WEAK_PTR);
    // A weak word can no longer be smuggled into the strong-Handle path:
    // the debug invariant on strong `Tagged` fires before rooting.
    let _ = scope.handle(unsafe { Tagged::<Object>::from_value_unchecked(weak) });
}
