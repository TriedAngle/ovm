use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{Thread, VM, VmError};
use vm::{Lookup, PropertyDescriptor, SlotName, StoreOutcome, StoreSemantics, Value, ValueRef};

/// Read a data property by interned name value.
fn get_prop(thread: &mut Thread, obj: Value, name: Value) -> Value {
    thread.heap().no_gc(|nogc| {
        let ValueRef::Object(o) = obj.value_ref(nogc) else {
            panic!("expected object");
        };
        match o.as_ref().lookup(nogc, SlotName::from_value(name)) {
            Lookup::Data { slot, .. } => slot.inner(),
            _ => panic!("expected a data property"),
        }
    })
}

fn error_and_props(thread: &mut Thread, err: VmError) -> (Value, Value, Value) {
    let obj = thread.error_object(err).unwrap();
    let (name_key, name_val, message_key) = thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name").value();
        let message_key = thread.intern(&scope, "message").value();
        let name_val = thread.intern(&scope, err.name()).value();
        (name_key, name_val, message_key)
    });
    let name = get_prop(thread, obj, name_key);
    let message = get_prop(thread, obj, message_key);
    assert_eq!(name, name_val, "name property must hold the class name");
    (obj, name, message)
}

#[test]
fn error_names_map_to_spec_classes() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    for (err, expected) in [
        (VmError::Arity, "TypeError"),
        (VmError::Type, "TypeError"),
        (VmError::NotExtensible, "TypeError"),
        (VmError::Overflow, "RangeError"),
        (VmError::OutOfBounds, "RangeError"),
        (VmError::StackOverflow, "RangeError"),
    ] {
        let (_, name, _) = error_and_props(&mut thread, err);
        let expected = thread.handle_scope(|thread, scope| thread.intern(&scope, expected).value());
        assert_eq!(name, expected, "{err:?}");
    }
}

#[test]
fn error_object_carries_name_and_message() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (_, name, message) = error_and_props(&mut thread, VmError::Type);
    let (type_error, msg) = thread.handle_scope(|thread, scope| {
        (
            thread.intern(&scope, "TypeError").value(),
            thread.intern(&scope, "invalid operand type").value(),
        )
    });
    assert_eq!(name, type_error);
    assert_eq!(message, msg);
}

#[test]
fn distinct_vm_errors_have_distinct_messages() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (_, _, type_msg) = error_and_props(&mut thread, VmError::Type);
    let (_, _, bounds_msg) = error_and_props(&mut thread, VmError::OutOfBounds);
    assert_ne!(type_msg, bounds_msg);
}

#[test]
fn error_objects_are_distinct_but_share_shapes() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (a, _, _) = error_and_props(&mut thread, VmError::Type);
    let (b, _, _) = error_and_props(&mut thread, VmError::Type);
    assert_ne!(a, b, "each throw materializes a fresh object");

    // both started from the well-known error map and added the same
    // properties in the same order: the transition cache must yield one
    // shared final shape
    let maps = thread.heap().no_gc(|nogc| {
        let ValueRef::Object(a) = a.value_ref(nogc) else {
            panic!("expected object");
        };
        let ValueRef::Object(b) = b.value_ref(nogc) else {
            panic!("expected object");
        };
        (
            a.as_ref().header.map.get().erase(),
            b.as_ref().header.map.get().erase(),
        )
    });
    assert_eq!(maps.0, maps.1);
}

#[test]
fn error_properties_are_writable() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (obj, _, _) = error_and_props(&mut thread, VmError::Type);
    let (name_key, custom) = thread.handle_scope(|thread, scope| {
        (
            thread.intern(&scope, "name").value(),
            thread.intern(&scope, "MyError").value(),
        )
    });

    let outcome = thread.heap().no_gc(|nogc| {
        obj.store_lookup(
            nogc,
            SlotName::from_value(name_key),
            custom,
            StoreSemantics::WriteThrough,
        )
    });
    assert!(matches!(outcome, Ok(StoreOutcome::Done)));
    assert_eq!(get_prop(&mut thread, obj, name_key), custom);
}

#[test]
fn error_objects_are_extendable() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (obj, _, _) = error_and_props(&mut thread, VmError::Type);
    let (extra_key, extra_val) = thread.handle_scope(|thread, scope| {
        (
            thread.intern(&scope, "extra").value(),
            thread.intern(&scope, "payload").value(),
        )
    });

    // store_lookup on a missing property must propose a transition
    // (proof of extendability), and completing it adds the own property
    let outcome = thread.heap().no_gc(|nogc| {
        obj.store_lookup(
            nogc,
            SlotName::from_value(extra_key),
            extra_val,
            StoreSemantics::WriteThrough,
        )
    });
    match outcome {
        Ok(StoreOutcome::Transition { receiver, name }) => {
            thread.handle_scope(|thread, scope| {
                let receiver = scope
                    .create_handle(unsafe {
                        vm::Tagged::<vm::Object>::from_value_unchecked(receiver)
                    })
                    .expect("receiver is strong");
                let name = scope.create_handle(name.tagged()).expect("name is strong");
                let value = scope
                    .create_handle(vm::Tagged::from_value(extra_val))
                    .expect("value is strong");
                vm::Object::define_own_property(
                    thread.heap(),
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(value.value()),
                )
                .unwrap();
            });
        }
        other => panic!("expected transition, got {other:?}"),
    }
    assert_eq!(get_prop(&mut thread, obj, extra_key), extra_val);
}

/// The object this object's map links to via its `prototype` slot.
fn prototype_of(thread: &mut Thread, obj: Value) -> Option<Value> {
    thread.heap().no_gc(|nogc| {
        let ValueRef::Object(o) = obj.value_ref(nogc) else {
            panic!("expected object");
        };
        let map = o.as_ref().header.map.heap_ref(nogc).as_ref();
        let proto = map.prototype.inner();
        if proto == nogc.known().null.value() {
            None
        } else {
            Some(proto)
        }
    })
}

#[test]
fn startup_prototype_hierarchy() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (error_obj, _, _) = error_and_props(&mut thread, VmError::Type);
    let (error_prototype, object_prototype, undefined, null, true_v, false_v, void) = {
        let k = thread.heap().known();
        (
            k.error_prototype.value(),
            k.object_prototype.value(),
            k.undefined.value(),
            k.null.value(),
            k.true_object.value(),
            k.false_object.value(),
            k.void.value(),
        )
    };

    // error instance -> %Error.prototype% -> %Object.prototype% -> none
    assert_eq!(
        prototype_of(&mut thread, error_obj),
        Some(error_prototype),
        "error instance must chain to %Error.prototype%"
    );
    assert_eq!(
        prototype_of(&mut thread, error_prototype),
        Some(object_prototype),
        "%Error.prototype% must chain to %Object.prototype%"
    );
    assert_eq!(
        prototype_of(&mut thread, object_prototype),
        None,
        "%Object.prototype% is the root"
    );

    // oddballs: undefined/true/false join the ordinary hierarchy ...
    assert_eq!(prototype_of(&mut thread, undefined), Some(object_prototype));
    assert_eq!(prototype_of(&mut thread, true_v), Some(object_prototype));
    assert_eq!(prototype_of(&mut thread, false_v), Some(object_prototype));
    // ... null (no prototype per spec) and the hole (internal) do not
    assert_eq!(prototype_of(&mut thread, null), None);
    assert_eq!(prototype_of(&mut thread, void), None);
}
