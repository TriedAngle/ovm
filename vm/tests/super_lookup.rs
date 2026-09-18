//! Direct tests of the super-lookup runtime: the three prototype shapes a
//! map's `prototype` slot can hold (null / single object / FixedArray of
//! parents) and both `StoreSemantics` variants of `super_store_lookup`.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    FixedArray, GcSlice, Heap, LoadOutcome, Object, PropertyDescriptor, SlotName, Smi,
    StoreOutcome, StoreSemantics, Tagged, Thread, VM, Value, VmError, home_proto,
    lookup_in_parents, super_lookup, super_store_lookup,
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
    unsafe { Tagged::from_value_unchecked(w) }.as_name()
}

fn thread() -> (VM, Thread) {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
            .new_object(&scope, map, GcSlice::EMPTY)
            .into_handle(&scope);
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

/// Read an integer data property, or panic.
fn get_smi(thread: &mut Thread, obj: Value, name: &str) -> i64 {
    let name = slot_name(thread, name);
    let heap = &*thread.heap();
    match unsafe { anchored(heap, obj) }.lookup(heap, name) {
        vm::Lookup::Data { slot, .. } => Smi::decode(slot.get(heap).raw()).unwrap().value(),
        _ => panic!("property {name:?} must be an own-or-inherited data property"),
    }
}

fn parents_of(thread: &mut Thread, p1: Value, p2: Value) -> Value {
    thread.handle_scope(|thread, scope| {
        let arr = {
            let _heap = &*thread.heap();
            thread.heap().allocate_handle::<FixedArray>(
                scope.stage(&[
                    unsafe { Tagged::<Value>::from_value_unchecked(p1) },
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
            super_lookup(heap, unsafe { anchored(heap, home) }, x).unwrap(),
            LoadOutcome::Value(v) if v.raw() == smi(7)
        ));
    };

    // null prototype terminates the chain
    let home = object_with(&mut thread, null, &[("x", 7)]);
    {
        let heap = &*thread.heap();
        assert!(matches!(
            super_lookup(heap, unsafe { anchored(heap, home) }, x).unwrap(),
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
            super_lookup(heap, unsafe { anchored(heap, home) }, a).unwrap(),
            LoadOutcome::Value(v) if v.raw() == smi(1)
        ));
        assert!(matches!(
            super_lookup(heap, unsafe { anchored(heap, home) }, b).unwrap(),
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
        match lookup_in_parents(heap, unsafe { anchored(heap, protos) }, a) {
            vm::Lookup::Data { slot, .. } => {
                assert_eq!(slot.get(heap).raw(), smi(1));
            }
            _ => panic!("a must resolve through the first parent"),
        }
        // "b" only exists on the second parent
        match lookup_in_parents(heap, unsafe { anchored(heap, protos) }, b) {
            vm::Lookup::Data { slot, .. } => assert_eq!(slot.get(heap).raw(), smi(3)),
            _ => panic!("b must resolve through the second parent"),
        }
        assert!(matches!(
            lookup_in_parents(heap, unsafe { anchored(heap, protos) }, missing),
            vm::Lookup::NotFound
        ));
        // null terminates the chain
        assert!(matches!(
            lookup_in_parents(heap, unsafe { anchored(heap, null) }, a),
            vm::Lookup::NotFound
        ));
    };
}

#[test]
fn super_store_shadow_creates_own_property_on_this() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let parent = object_with(&mut thread, null, &[("x", 1)]);
    let home = object_with(&mut thread, parent, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");

    thread.handle_scope(|thread, scope| {
        {
            let heap = &*thread.heap();
            match super_store_lookup(
                heap,
                &scope,
                home_proto(heap, unsafe { anchored(heap, home) }),
                unsafe { anchored(heap, this_) },
                x,
                unsafe { anchored(heap, smi(42)) },
                StoreSemantics::Shadow,
            )
            .unwrap()
            {
                StoreOutcome::Transition { receiver, name } => {
                    assert!(
                        receiver
                            .as_tagged(heap)
                            .ptr_eq(unsafe { anchored(heap, this_) })
                    );
                    assert!(name.as_tagged(heap).ptr_eq(x.erase()));
                }
                other => panic!("shadow store must define on the receiver, got {other:?}"),
            }
        };
    });
    // the parent keeps its value; the transition handler would add `x` to
    // `this`
    assert_eq!(get_smi(&mut thread, parent, "x"), 1);
}

#[test]
fn super_store_write_through_updates_the_holder() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let parent = object_with(&mut thread, null, &[("x", 1)]);
    let home = object_with(&mut thread, parent, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");

    thread.handle_scope(|thread, scope| {
        {
            let heap = &*thread.heap();
            let outcome = super_store_lookup(
                heap,
                &scope,
                home_proto(heap, unsafe { anchored(heap, home) }),
                unsafe { anchored(heap, this_) },
                x,
                unsafe { anchored(heap, smi(42)) },
                StoreSemantics::WriteThrough,
            )
            .unwrap();
            assert!(matches!(outcome, StoreOutcome::Done));
        };
    });
    assert_eq!(get_smi(&mut thread, parent, "x"), 42);
    // nothing was created on the receiver
    {
        let heap = &*thread.heap();
        assert!(matches!(
            unsafe { anchored(heap, this_) }.lookup(heap, x),
            vm::Lookup::NotFound
        ));
    };
}

#[test]
fn super_store_write_through_hits_second_parent_holder() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let p1 = object_with(&mut thread, null, &[("a", 1)]);
    let p2 = object_with(&mut thread, null, &[("b", 3)]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let b = slot_name(&mut thread, "b");

    thread.handle_scope(|thread, scope| {
        {
            let heap = &*thread.heap();
            let outcome = super_store_lookup(
                heap,
                &scope,
                home_proto(heap, unsafe { anchored(heap, home) }),
                unsafe { anchored(heap, this_) },
                b,
                unsafe { anchored(heap, smi(9)) },
                StoreSemantics::WriteThrough,
            )
            .unwrap();
            assert!(matches!(outcome, StoreOutcome::Done));
        };
    });
    // the holder (second parent) got the write
    assert_eq!(get_smi(&mut thread, p2, "b"), 9);
    assert_eq!(get_smi(&mut thread, p1, "a"), 1);
}

#[test]
fn super_store_readonly_and_nullish_receiver_throw() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let parent = object_with(&mut thread, null, &[]);
    // a non-writable parent property
    thread.handle_scope(|thread, scope| {
        let name = thread.intern(&scope, "x");
        let name = scope.handle(name.as_tagged(&*thread.heap()));
        // Safety: parent word freshly returned, consumed here.
        let parent_obj = scope
            .cast::<Object>(unsafe { parent.assume_valid(&*thread.heap()) })
            .expect("object_with returns an object");
        let value = scope.handle(Smi::new(1));
        Object::add_own_property(
            thread.heap(),
            &scope,
            parent_obj,
            name,
            PropertyDescriptor::Data {
                value,
                writable: false,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
    });
    let home = object_with(&mut thread, parent, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");
    let undefined = {
        let heap = thread.heap();
        heap.known().undefined.as_tagged(heap).raw()
    };

    thread.handle_scope(|thread, scope| {
        {
            let heap = &*thread.heap();
            for semantics in [StoreSemantics::Shadow, StoreSemantics::WriteThrough] {
                assert!(matches!(
                    super_store_lookup(
                        heap,
                        &scope,
                        home_proto(heap, unsafe { anchored(heap, home) }),
                        unsafe { anchored(heap, this_) },
                        x,
                        unsafe { anchored(heap, smi(2)) },
                        semantics
                    ),
                    Err(VmError::Type)
                ));
            }
            // nullish receivers are invalid property store receivers
            for bad in [null, undefined] {
                assert!(matches!(
                    super_store_lookup(
                        heap,
                        &scope,
                        home_proto(heap, unsafe { anchored(heap, home) }),
                        unsafe { anchored(heap, bad) },
                        x,
                        unsafe { anchored(heap, smi(2)) },
                        StoreSemantics::Shadow
                    ),
                    Err(VmError::Type)
                ));
            }
        };
    });
}

#[test]
fn super_store_on_null_proto_chain_defines_on_this() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let home = object_with(&mut thread, null, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");

    thread.handle_scope(|thread, scope| {
        {
            let heap = &*thread.heap();
            // no parent chain at all: both semantics define on the receiver
            for semantics in [StoreSemantics::Shadow, StoreSemantics::WriteThrough] {
                assert!(matches!(
                    super_store_lookup(
                        heap,
                        &scope,
                        home_proto(heap, unsafe { anchored(heap, home) }),
                        unsafe { anchored(heap, this_) },
                        x,
                        unsafe { anchored(heap, smi(2)) },
                        semantics
                    )
                    .unwrap(),
                    StoreOutcome::Transition { receiver, .. }
                        if receiver.as_tagged(heap).ptr_eq(unsafe { anchored(heap, this_) })
                ));
            }
        };
    });
}
