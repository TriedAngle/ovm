use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    FixedArray, HandleSlice, Heap, LoadOutcome, Lookup, Object, PropertyDescriptor, SlotName, Smi,
    Tagged, Thread, VM, Value,
};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

/// Safety helper: these tests only pass words that were loaded or
/// allocated under the very heap borrow they are re-anchored at, with no
/// collection in between.
unsafe fn anchored<'a>(heap: &'a Heap, v: Value) -> Tagged<'a, Value> {
    unsafe { v.assume_valid(heap) }
}

/// A name tag from a raw word. Tests only: no collection may run between
/// the word's load and its consumption.
fn raw_name(w: Value) -> Tagged<'static, SlotName> {
    unsafe { Tagged::<Value>::from_value_unchecked(w) }.as_name()
}

fn thread() -> (VM, Thread) {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let thread = vm.attach();
    (vm, thread)
}

/// The well-known `null` word for `thread`'s heap.
fn null_word(thread: &mut Thread) -> Value {
    let heap = thread.heap();
    heap.known().null.as_tagged(heap).raw()
}

/// A fresh ordinary object with `proto` (any of the three shapes) and
/// integer-valued data properties.
fn object_with(thread: &mut Thread, proto: Value, props: &[(&str, i64)]) -> Value {
    thread.handle_scope(|thread, scope| {
        let map = thread.heap().known().object_initial_map;
        let obj = thread
            .heap()
            .new_object(&scope, map, HandleSlice::EMPTY)
            .as_handle(&scope);
        // Safety: caller-supplied proto word, rooted before any allocation.
        let proto = scope.handle(unsafe { proto.assume_valid(&*thread.heap()) });
        Object::set_prototype(thread.heap(), &scope, obj, proto).unwrap();
        for (name, v) in props {
            let name = thread.intern(&scope, name);
            let name = scope.handle(name.as_tagged(&*thread.heap()));
            let value = scope.handle(Smi::new(*v));
            Object::add_own_property(
                thread.heap(),
                &scope,
                obj,
                name,
                PropertyDescriptor::data(value),
            )
            .unwrap();
        }
        obj.as_tagged(&*thread.heap()).raw()
    })
}

fn slot_name(thread: &mut Thread, name: &str) -> Tagged<'static, SlotName> {
    thread.handle_scope(|thread, scope| {
        let interned = thread.intern(&scope, name);
        let heap = &*thread.heap();
        raw_name(interned.as_tagged(heap).raw())
    })
}

fn parents_of(thread: &mut Thread, p1: Value, p2: Value) -> Value {
    thread.handle_scope(|thread, scope| {
        let first = thread.intern(&scope, "parent");
        let second = thread.intern(&scope, "mixin");
        let (first, second) = {
            let heap = &*thread.heap();
            (first.as_tagged(heap).raw(), second.as_tagged(heap).raw())
        };
        let arr = {
            let _heap = &*thread.heap();
            thread.heap().allocate_handle::<FixedArray>(
                scope.stage(&[
                    unsafe { Tagged::<Value>::from_value_unchecked(first) },
                    unsafe { Tagged::from_value_unchecked(p1) },
                    unsafe { Tagged::from_value_unchecked(second) },
                    unsafe { Tagged::from_value_unchecked(p2) },
                ]),
                &scope,
            )
        };
        let heap = &*thread.heap();
        arr.as_tagged(heap).raw()
    })
}

#[test]
fn super_lookup_dispatches_all_three_prototype_shapes() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);

    // single-object prototype: the parent's property is found
    let parent = object_with(&mut thread, null, &[("x", 7)]);
    let home = object_with(&mut thread, parent, &[]);
    let x = slot_name(&mut thread, "x");
    {
        let heap = &*thread.heap();
        assert!(matches!(
            Lookup::super_lookup(heap, unsafe { anchored(heap, home) }, x).unwrap(),
            LoadOutcome::Value(v) if v.raw() == smi(7)
        ));
    };

    // null prototype terminates the chain
    let home = object_with(&mut thread, null, &[("x", 7)]);
    {
        let heap = &*thread.heap();
        assert!(matches!(
            Lookup::super_lookup(heap, unsafe { anchored(heap, home) }, x).unwrap(),
            LoadOutcome::Value(v) if v.raw() == heap.known().undefined.as_tagged(heap).raw()
        ));
    };

    // FixedArray: multiple parents (Self-style), priority order
    let p1 = object_with(&mut thread, null, &[("a", 1)]);
    let p2 = object_with(&mut thread, null, &[("a", 2), ("b", 3)]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let a = slot_name(&mut thread, "a");
    let b = slot_name(&mut thread, "b");
    {
        let heap = &*thread.heap();
        assert!(matches!(
            Lookup::super_lookup(heap, unsafe { anchored(heap, home) }, a).unwrap(),
            LoadOutcome::Value(v) if v.raw() == smi(1)
        ));
        assert!(matches!(
            Lookup::super_lookup(heap, unsafe { anchored(heap, home) }, b).unwrap(),
            LoadOutcome::Value(v) if v.raw() == smi(3)
        ));
    };
}

#[test]
fn lookup_in_parents_respects_priority_order() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let p1 = object_with(&mut thread, null, &[("a", 1)]);
    let p2 = object_with(&mut thread, null, &[("a", 2), ("b", 3)]);
    let protos = parents_of(&mut thread, p1, p2);
    let a = slot_name(&mut thread, "a");
    let b = slot_name(&mut thread, "b");
    let missing = slot_name(&mut thread, "nope");

    {
        let heap = &*thread.heap();
        // "a" exists on both parents: the first in priority order wins
        match Lookup::lookup_in_parents(heap, unsafe { anchored(heap, protos) }, a) {
            vm::Lookup::Data { slot, .. } => {
                assert_eq!(slot.get(heap).raw(), smi(1));
            }
            _ => panic!("a must resolve through the first parent"),
        }
        // "b" only exists on the second parent
        match Lookup::lookup_in_parents(heap, unsafe { anchored(heap, protos) }, b) {
            vm::Lookup::Data { slot, .. } => assert_eq!(slot.get(heap).raw(), smi(3)),
            _ => panic!("b must resolve through the second parent"),
        }
        assert!(matches!(
            Lookup::lookup_in_parents(heap, unsafe { anchored(heap, protos) }, missing),
            vm::Lookup::NotFound
        ));
        // null terminates the chain
        assert!(matches!(
            Lookup::lookup_in_parents(heap, unsafe { anchored(heap, null) }, a),
            vm::Lookup::NotFound
        ));
    };
}

// TODO: write tests for `super_store_lookup` (Shadow / WriteThrough
// semantics, readonly holders, nullish receivers).
