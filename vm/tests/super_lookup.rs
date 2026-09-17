//! Direct tests of the super-lookup runtime: the three prototype shapes a
//! map's `prototype` slot can hold (null / single object / FixedArray of
//! parents) and both `StoreSemantics` variants of `super_store_lookup`.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    FixedArray, GcSlice, LoadOutcome, Object, PropertyDescriptor, SlotName, Smi, StoreOutcome,
    StoreSemantics, Thread, VM, Value, VmError, home_proto, lookup_in_parents, super_lookup,
    super_store_lookup,
};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

fn thread() -> (VM, Thread) {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let thread = vm.attach();
    (vm, thread)
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
        Object::set_prototype(thread.heap(), &scope, obj.value(), proto).unwrap();
        for (name, v) in props {
            let name = thread.intern(&scope, name).value();
            Object::add_own_property_values(
                thread.heap(),
                &scope,
                obj.value(),
                SlotName::from_value(name),
                PropertyDescriptor::data(smi(*v)),
            )
            .unwrap();
        }
        obj.value()
    })
}

fn slot_name(thread: &mut Thread, name: &str) -> SlotName {
    thread.handle_scope(|thread, scope| SlotName::from_value(thread.intern(&scope, name).value()))
}

/// Read an integer data property, or panic.
fn get_smi(thread: &mut Thread, obj: Value, name: &str) -> i64 {
    let name = slot_name(thread, name);
    thread.heap().no_gc(|nogc| match obj.lookup(nogc, name) {
        vm::Lookup::Data { slot, .. } => Smi::decode(slot.inner()).unwrap().value(),
        _ => panic!("property {name:?} must be an own-or-inherited data property"),
    })
}

fn parents_of(thread: &mut Thread, p1: Value, p2: Value) -> Value {
    thread.handle_scope(|thread, scope| {
        thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[p1, p2]), &scope)
            .value()
    })
}

#[test]
fn super_lookup_dispatches_all_three_prototype_shapes() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();

    // single-object prototype: the parent's property is found
    let parent = object_with(&mut thread, null, &[("x", 7)]);
    let home = object_with(&mut thread, parent, &[]);
    let x = slot_name(&mut thread, "x");
    thread.heap().no_gc(|nogc| {
        assert!(matches!(
            super_lookup(nogc, home, x).unwrap(),
            LoadOutcome::Value(v) if v == smi(7)
        ));
    });

    // null prototype terminates the chain
    let home = object_with(&mut thread, null, &[("x", 7)]);
    thread.heap().no_gc(|nogc| {
        assert!(matches!(
            super_lookup(nogc, home, x).unwrap(),
            LoadOutcome::Value(v) if v == nogc.known().undefined.value()
        ));
    });

    // FixedArray: multiple parents (Self-style), priority order
    let p1 = object_with(&mut thread, null, &[("a", 1)]);
    let p2 = object_with(&mut thread, null, &[("a", 2), ("b", 3)]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let a = slot_name(&mut thread, "a");
    let b = slot_name(&mut thread, "b");
    thread.heap().no_gc(|nogc| {
        assert!(matches!(
            super_lookup(nogc, home, a).unwrap(),
            LoadOutcome::Value(v) if v == smi(1)
        ));
        assert!(matches!(
            super_lookup(nogc, home, b).unwrap(),
            LoadOutcome::Value(v) if v == smi(3)
        ));
    });
}

#[test]
fn lookup_in_parents_respects_priority_order() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();
    let p1 = object_with(&mut thread, null, &[("a", 1)]);
    let p2 = object_with(&mut thread, null, &[("a", 2), ("b", 3)]);
    let protos = parents_of(&mut thread, p1, p2);
    let a = slot_name(&mut thread, "a");
    let b = slot_name(&mut thread, "b");
    let missing = slot_name(&mut thread, "nope");

    thread.heap().no_gc(|nogc| {
        // "a" exists on both parents: the first in priority order wins
        match lookup_in_parents(nogc, protos, a) {
            vm::Lookup::Data { slot, .. } => {
                assert_eq!(slot.inner(), smi(1));
            }
            _ => panic!("a must resolve through the first parent"),
        }
        // "b" only exists on the second parent
        match lookup_in_parents(nogc, protos, b) {
            vm::Lookup::Data { slot, .. } => assert_eq!(slot.inner(), smi(3)),
            _ => panic!("b must resolve through the second parent"),
        }
        assert!(matches!(
            lookup_in_parents(nogc, protos, missing),
            vm::Lookup::NotFound
        ));
        // null terminates the chain
        assert!(matches!(
            lookup_in_parents(nogc, null, a),
            vm::Lookup::NotFound
        ));
    });
}

#[test]
fn super_store_shadow_creates_own_property_on_this() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();
    let parent = object_with(&mut thread, null, &[("x", 1)]);
    let home = object_with(&mut thread, parent, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");

    thread.heap().no_gc(|nogc| {
        match super_store_lookup(
            nogc,
            home_proto(nogc, home),
            this_,
            x,
            smi(42),
            StoreSemantics::Shadow,
        )
        .unwrap()
        {
            StoreOutcome::Transition { receiver, name } => {
                assert_eq!(receiver, this_);
                assert_eq!(name, x);
            }
            other => panic!("shadow store must define on the receiver, got {other:?}"),
        }
    });
    // the parent keeps its value; the transition handler would add `x` to
    // `this`
    assert_eq!(get_smi(&mut thread, parent, "x"), 1);
}

#[test]
fn super_store_write_through_updates_the_holder() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();
    let parent = object_with(&mut thread, null, &[("x", 1)]);
    let home = object_with(&mut thread, parent, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");

    thread.heap().no_gc(|nogc| {
        let outcome = super_store_lookup(
            nogc,
            home_proto(nogc, home),
            this_,
            x,
            smi(42),
            StoreSemantics::WriteThrough,
        )
        .unwrap();
        assert!(matches!(outcome, StoreOutcome::Done));
    });
    assert_eq!(get_smi(&mut thread, parent, "x"), 42);
    // nothing was created on the receiver
    thread.heap().no_gc(|nogc| {
        assert!(matches!(this_.lookup(nogc, x), vm::Lookup::NotFound));
    });
}

#[test]
fn super_store_write_through_hits_second_parent_holder() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();
    let p1 = object_with(&mut thread, null, &[("a", 1)]);
    let p2 = object_with(&mut thread, null, &[("b", 3)]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let b = slot_name(&mut thread, "b");

    thread.heap().no_gc(|nogc| {
        let outcome = super_store_lookup(
            nogc,
            home_proto(nogc, home),
            this_,
            b,
            smi(9),
            StoreSemantics::WriteThrough,
        )
        .unwrap();
        assert!(matches!(outcome, StoreOutcome::Done));
    });
    // the holder (second parent) got the write
    assert_eq!(get_smi(&mut thread, p2, "b"), 9);
    assert_eq!(get_smi(&mut thread, p1, "a"), 1);
}

#[test]
fn super_store_readonly_and_nullish_receiver_throw() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();
    let parent = object_with(&mut thread, null, &[]);
    // a non-writable parent property
    thread.handle_scope(|thread, scope| {
        let name = thread.intern(&scope, "x").value();
        Object::add_own_property_values(
            thread.heap(),
            &scope,
            parent,
            SlotName::from_value(name),
            PropertyDescriptor::Data {
                value: smi(1),
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
    let undefined = thread.heap().known().undefined.value();

    thread.heap().no_gc(|nogc| {
        for semantics in [StoreSemantics::Shadow, StoreSemantics::WriteThrough] {
            assert_eq!(
                super_store_lookup(nogc, home_proto(nogc, home), this_, x, smi(2), semantics),
                Err(VmError::Type)
            );
        }
        // nullish receivers are invalid property store receivers
        for bad in [null, undefined] {
            assert_eq!(
                super_store_lookup(
                    nogc,
                    home_proto(nogc, home),
                    bad,
                    x,
                    smi(2),
                    StoreSemantics::Shadow
                ),
                Err(VmError::Type)
            );
        }
    });
}

#[test]
fn super_store_on_null_proto_chain_defines_on_this() {
    let (_vm, mut thread) = thread();
    let null = thread.heap().known().null.value();
    let home = object_with(&mut thread, null, &[]);
    let this_ = object_with(&mut thread, null, &[]);
    let x = slot_name(&mut thread, "x");

    thread.heap().no_gc(|nogc| {
        // no parent chain at all: both semantics define on the receiver
        for semantics in [StoreSemantics::Shadow, StoreSemantics::WriteThrough] {
            assert!(matches!(
                super_store_lookup(nogc, home_proto(nogc, home), this_, x, smi(2), semantics).unwrap(),
                StoreOutcome::Transition { receiver, .. } if receiver == this_
            ));
        }
    });
}
