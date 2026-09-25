use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{Lookup, PropertyDescriptor, SlotName, StoreOutcome, StoreSemantics, Tagged, Value};
use vm::{Thread, VM, VmError};

/// A name tag re-anchored under a live heap borrow.
fn name<'a>(heap: &'a vm::Heap, w: Value) -> Tagged<'a, SlotName> {
    unsafe { w.assume_valid(heap) }.as_name()
}

/// Read a data property by interned name value.
fn get_prop(thread: &mut Thread, obj: Value, name_word: Value) -> Value {
    {
        let heap = &*thread.heap();
        let Some(o) = unsafe { obj.assume_valid(heap) }.as_heap_object() else {
            panic!("expected object");
        };
        match o.lookup(heap, name(heap, name_word)) {
            Lookup::Data { slot, .. } => slot.get(heap).raw(),
            _ => panic!("expected a data property"),
        }
    }
}

/// A rooted copy of `word`, staged immediately (no allocation between
/// the read and the staging).
fn as_tagged_unchecked(word: Value) -> Tagged<'static, Value> {
    // Safety: only used same-statement under a live anchor in these tests.
    unsafe { Tagged::from_value_unchecked(word) }
}

fn error_and_props(thread: &mut Thread, err: VmError) -> (Value, Value, Value) {
    let obj = thread.error_object(err).unwrap();
    let (name_key, name_val, message_key) = thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name");
        let message_key = thread.intern(&scope, "message");
        let name_val = thread.intern(&scope, err.name());
        let heap = &*thread.heap();
        (
            name_key.as_tagged(heap).raw(),
            name_val.as_tagged(heap).raw(),
            message_key.as_tagged(heap).raw(),
        )
    });
    let name = get_prop(thread, obj, name_key);
    let message = get_prop(thread, obj, message_key);
    assert_eq!(name, name_val, "name property must hold the class name");
    (obj, name, message)
}

#[test]
fn error_names_map_to_spec_classes() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
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
        let expected = thread.handle_scope(|thread, scope| {
            let e = thread.intern(&scope, expected);
            let heap = &*thread.heap();
            e.as_tagged(heap).raw()
        });
        assert_eq!(name, expected, "{err:?}");
    }
}

#[test]
fn error_object_carries_name_and_message() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (_, name, message) = error_and_props(&mut thread, VmError::Type);
    let (type_error, msg) = thread.handle_scope(|thread, scope| {
        let type_error = thread.intern(&scope, "TypeError");
        let msg = thread.intern(&scope, "invalid operand type");
        let heap = &*thread.heap();
        (type_error.as_tagged(heap).raw(), msg.as_tagged(heap).raw())
    });
    assert_eq!(name, type_error);
    assert_eq!(message, msg);
}

#[test]
fn distinct_vm_errors_have_distinct_messages() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (_, _, type_msg) = error_and_props(&mut thread, VmError::Type);
    let (_, _, bounds_msg) = error_and_props(&mut thread, VmError::OutOfBounds);
    assert_ne!(type_msg, bounds_msg);
}

#[test]
fn error_objects_are_distinct_but_share_shapes() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (a, _, _) = error_and_props(&mut thread, VmError::Type);
    let (b, _, _) = error_and_props(&mut thread, VmError::Type);
    assert_ne!(a, b, "each throw materializes a fresh object");

    // both started from the well-known error map and added the same
    // properties in the same order: the transition cache must yield one
    // shared final shape
    let maps = {
        let heap = &*thread.heap();
        let Some(a) = unsafe { a.assume_valid(heap) }.as_heap_object() else {
            panic!("expected object");
        };
        let Some(b) = unsafe { b.assume_valid(heap) }.as_heap_object() else {
            panic!("expected object");
        };
        (
            a.as_ref().header.map.get(heap).raw(),
            b.as_ref().header.map.get(heap).raw(),
        )
    };
    assert_eq!(maps.0, maps.1);
}

#[test]
fn error_properties_are_writable() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (obj, _, _) = error_and_props(&mut thread, VmError::Type);
    let (name_key, custom) = thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name");
        let custom = thread.intern(&scope, "MyError");
        let heap = &*thread.heap();
        (name_key.as_tagged(heap).raw(), custom.as_tagged(heap).raw())
    });

    thread.handle_scope(|thread, scope| {
        let outcome = {
            let heap = &*thread.heap();
            as_tagged_unchecked(obj).store_lookup(
                heap,
                &scope,
                name(heap, name_key),
                as_tagged_unchecked(custom),
                StoreSemantics::WriteThrough,
            )
        };
        assert!(matches!(outcome, Ok(StoreOutcome::Done)));
    });
    assert_eq!(get_prop(&mut thread, obj, name_key), custom);
}

#[test]
fn error_objects_are_extendable() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (obj, _, _) = error_and_props(&mut thread, VmError::Type);
    let (extra_key, extra_val) = thread.handle_scope(|thread, scope| {
        let extra_key = thread.intern(&scope, "extra");
        let extra_val = thread.intern(&scope, "payload");
        let heap = &*thread.heap();
        (
            extra_key.as_tagged(heap).raw(),
            extra_val.as_tagged(heap).raw(),
        )
    });

    // store_lookup on a missing property must propose a transition
    // (proof of extendability), and completing it adds the own property
    thread.handle_scope(|thread, scope| {
        let outcome = {
            let heap = &*thread.heap();
            as_tagged_unchecked(obj).store_lookup(
                heap,
                &scope,
                name(heap, extra_key),
                as_tagged_unchecked(extra_val),
                StoreSemantics::WriteThrough,
            )
        };
        match outcome {
            Ok(StoreOutcome::Transition { receiver, name }) => {
                let value = scope.handle(as_tagged_unchecked(extra_val));
                vm::Object::define_own_property(
                    thread.heap(),
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(value),
                )
                .unwrap();
            }
            other => panic!("expected transition, got {other:?}"),
        }
    });
    assert_eq!(get_prop(&mut thread, obj, extra_key), extra_val);
}

/// The object this object's map links to via its `prototype` slot.
fn prototype_of(thread: &mut Thread, obj: Value) -> Option<Value> {
    {
        let heap = &*thread.heap();
        let Some(o) = unsafe { obj.assume_valid(heap) }.as_heap_object() else {
            panic!("expected object");
        };
        let map = o.as_ref().header.map.get(heap).as_ref();
        let proto = map.prototype.get(heap).raw();
        if proto == heap.known().null.as_tagged(heap).raw() {
            None
        } else {
            Some(proto)
        }
    }
}

#[test]
fn startup_prototype_hierarchy() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (error_obj, _, _) = error_and_props(&mut thread, VmError::Type);
    let (error_prototype, object_prototype, undefined, null, true_v, false_v, the_hole) = {
        let heap = thread.heap();
        let k = heap.known();
        (
            k.error_prototype.as_tagged(heap).raw(),
            k.object_prototype.as_tagged(heap).raw(),
            k.undefined.as_tagged(heap).raw(),
            k.null.as_tagged(heap).raw(),
            k.true_object.as_tagged(heap).raw(),
            k.false_object.as_tagged(heap).raw(),
            k.the_hole.as_tagged(heap).raw(),
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
    assert_eq!(prototype_of(&mut thread, the_hole), None);
}
