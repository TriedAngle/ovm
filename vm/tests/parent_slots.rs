//! Self-style parenting: `parent*:` entries live in the [[Prototype]]
//! link as one inline `[name, parent, ...]` pair array, whether there is
//! one parent or several. Names are own-level parent slots; values are
//! searched in order. No data slot is created.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    FixedArray, Handle, HandleScope, HandleSlice, Lookup, Object, SlotFlags, SlotName, Smi,
    StoreOutcome, StoreSemantics, Thread, VM, VmError,
};

fn thread() -> (VM, Thread) {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let thread = vm.attach();
    (vm, thread)
}

fn fresh_object<'s>(thread: &mut Thread, scope: &'s HandleScope<'_>) -> Handle<'s, Object> {
    let map = thread.heap().known().plain_object_map;
    thread
        .heap()
        .new_object(scope, map, HandleSlice::EMPTY)
        .as_handle(scope)
}

fn intern_name<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    s: &str,
) -> Handle<'s, SlotName> {
    let interned = thread.intern(scope, s);
    let heap = &*thread.heap();
    scope.handle(interned.as_tagged(heap))
}

fn add_prop(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    obj: Handle<'_, Object>,
    name: Handle<'_, SlotName>,
    value: i64,
) {
    let value = scope.handle(Smi::new(value));
    Object::add_own_property(
        thread.heap(),
        scope,
        obj,
        name,
        vm::PropertyDescriptor::data(value),
    )
    .unwrap();
}

fn add_parent(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    obj: Handle<'_, Object>,
    name: Handle<'_, SlotName>,
    parent: Handle<'_, Object>,
) {
    Object::add_parent(thread.heap(), scope, obj, name, parent.erase()).unwrap();
}

fn get_smi(thread: &mut Thread, obj: Handle<'_, Object>, name: Handle<'_, SlotName>) -> i64 {
    let heap = &*thread.heap();
    match obj.as_tagged(heap).lookup(heap, name.as_tagged(heap)) {
        Lookup::Data { slot, .. } => Smi::decode(slot.get(heap).raw()).unwrap().value(),
        _ => panic!("property must resolve to a data slot"),
    }
}

#[test]
fn one_parent_is_a_pair_array() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let parent = fresh_object(thread, &scope);
        let parent_name = intern_name(thread, &scope, "parent");
        let obj = fresh_object(thread, &scope);
        add_parent(&mut *thread, &scope, obj, parent_name, parent);

        let heap = &*thread.heap();
        let proto = obj.as_tagged(heap).map_ref(heap).prototype.get(heap);
        let pairs = proto
            .get_as::<FixedArray>()
            .expect("parents are always an inline pair array");
        assert_eq!(pairs.len(), 2, "[name, parent]");
        assert!(
            pairs
                .at(heap, 0)
                .ptr_eq(parent_name.as_tagged(heap).erase()),
            "the label comes first"
        );
        assert!(
            pairs.at(heap, 1).ptr_eq(parent.as_tagged(heap).erase()),
            "the parent value second"
        );
        let map = obj.as_tagged(heap).map_ref(heap);
        assert_eq!(map.descriptor_count(), 0, "parenting creates no slot");
        assert_eq!(map.value_slot_count(), 0, "parenting stores no value");
    });
}

#[test]
fn several_parents_keep_priority_order() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let p1 = fresh_object(thread, &scope);
        let p2 = fresh_object(thread, &scope);
        let parent = intern_name(thread, &scope, "parent");
        let mixin = intern_name(thread, &scope, "mixin");
        let child = fresh_object(thread, &scope);
        add_parent(&mut *thread, &scope, child, parent, p1);
        add_parent(&mut *thread, &scope, child, mixin, p2);

        let heap = &*thread.heap();
        let proto = child.as_tagged(heap).map_ref(heap).prototype.get(heap);
        let pairs = proto.get_as::<FixedArray>().expect("pair array");
        assert_eq!(pairs.len(), 4);
        assert!(pairs.at(heap, 0).ptr_eq(parent.as_tagged(heap).erase()));
        assert!(pairs.at(heap, 1).ptr_eq(p1.as_tagged(heap).erase()));
        assert!(pairs.at(heap, 2).ptr_eq(mixin.as_tagged(heap).erase()));
        assert!(pairs.at(heap, 3).ptr_eq(p2.as_tagged(heap).erase()));
    });
}

#[test]
fn parent_names_are_own_read_only_slots() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let parent = fresh_object(thread, &scope);
        let parent_name = intern_name(thread, &scope, "parent");
        let obj = fresh_object(thread, &scope);
        add_parent(&mut *thread, &scope, obj, parent_name, parent);

        let heap = &*thread.heap();
        match obj
            .as_tagged(heap)
            .lookup(heap, parent_name.as_tagged(heap))
        {
            Lookup::Data { slot, flags, .. } => {
                assert!(slot.get(heap).ptr_eq(parent.as_tagged(heap).erase()));
                assert!(!flags.is_writable());
            }
            _ => panic!("`obj.parent` must resolve to the parent value"),
        }

        let nine: Handle<'_, Smi> = scope.handle(Smi::new(9));
        let write = obj.store_lookup(
            heap,
            &scope,
            parent_name.as_tagged(heap),
            nine.as_tagged(heap).erase(),
            StoreSemantics::WriteThrough,
        );
        assert!(matches!(write, Err(VmError::Type)));
        assert!(SlotFlags::VALUE.bits() == 0);
    });
}

#[test]
fn parents_are_searched_in_added_order() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let p1 = fresh_object(thread, &scope);
        let p2 = fresh_object(thread, &scope);
        let x = intern_name(thread, &scope, "x");
        let y = intern_name(thread, &scope, "y");
        add_prop(&mut *thread, &scope, p1, x, 1);
        add_prop(&mut *thread, &scope, p2, x, 2);
        add_prop(&mut *thread, &scope, p2, y, 3);

        let parent = intern_name(thread, &scope, "parent");
        let mixin = intern_name(thread, &scope, "mixin");
        let child = fresh_object(thread, &scope);
        add_parent(&mut *thread, &scope, child, parent, p1);
        add_parent(&mut *thread, &scope, child, mixin, p2);

        assert_eq!(get_smi(&mut *thread, child, x), 1, "first parent wins");
        assert_eq!(get_smi(&mut *thread, child, y), 3, "second parent fallback");
    });
}

#[test]
fn write_through_reaches_the_parent_holder() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let parent = fresh_object(thread, &scope);
        let x = intern_name(thread, &scope, "x");
        add_prop(&mut *thread, &scope, parent, x, 1);

        let parent_name = intern_name(thread, &scope, "parent");
        let child = fresh_object(thread, &scope);
        add_parent(&mut *thread, &scope, child, parent_name, parent);

        let nine: Handle<'_, Smi> = scope.handle(Smi::new(9));
        let heap = &*thread.heap();
        let outcome = child
            .store_lookup(
                heap,
                &scope,
                x.as_tagged(heap),
                nine.as_tagged(heap).erase(),
                StoreSemantics::WriteThrough,
            )
            .unwrap();
        assert!(matches!(outcome, StoreOutcome::Done));
        assert_eq!(get_smi(&mut *thread, parent, x), 9);
        let heap = &*thread.heap();
        assert_eq!(
            child.as_tagged(heap).map_ref(heap).descriptor_count(),
            0,
            "the child reaches through the parent, it grows no own slot"
        );
    });
}
