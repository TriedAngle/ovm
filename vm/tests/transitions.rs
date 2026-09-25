use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    Handle, HandleScope, HandleSlice, Heap, Map, Object, PropertyDescriptor, SlotName, Smi, Tagged,
    Thread, VM, Value,
};

fn thread() -> (VM, Thread) {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let thread = vm.attach();
    (vm, thread)
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

fn map_handle<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    obj: Handle<'_, Object>,
) -> Handle<'s, Map> {
    let heap = &*thread.heap();
    let map = obj.as_tagged(heap).as_ref().header.map.get(heap);
    scope.handle(map)
}

fn object_map_word(thread: &mut Thread, obj: Handle<'_, Object>) -> Value {
    let heap = &*thread.heap();
    obj.as_tagged(heap).as_ref().header.map.get(heap).raw()
}

fn pred_word(thread: &mut Thread, map: Value) -> Option<Value> {
    let heap = &*thread.heap();
    let map = unsafe { map.assume_valid(heap) }
        .get_as::<Map>()
        .expect("map");
    map.pred(heap).map(|p| p.raw())
}

fn root_word(thread: &mut Thread, map: Value) -> Value {
    let heap = &*thread.heap();
    let map = unsafe { map.assume_valid(heap) }
        .get_as::<Map>()
        .expect("map");
    map.root_map(heap).raw()
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
        PropertyDescriptor::data(value),
    )
    .unwrap();
}

fn transition_target<'a>(
    heap: &'a Heap,
    map: Tagged<'a, Map>,
    name: Tagged<'a, SlotName>,
) -> Option<Tagged<'a, Value>> {
    let array = map.transitions.load(heap)?;
    for entry in array.as_slice().as_chunks::<2>().0 {
        if entry[0].get(heap).ptr_eq(name.erase()) {
            return entry[1].get_strong(heap);
        }
    }
    None
}

#[test]
fn append_records_pred_and_shares_transitions() {
    let (_vm, mut thread) = thread();

    let (root, map_a, map_ab, map_a_again) = thread.handle_scope(|thread, scope| {
        let name_a = intern_name(thread, &scope, "a");
        let name_b = intern_name(thread, &scope, "b");
        let root_handle = thread.heap().known().object_initial_map;

        let obj = thread
            .heap()
            .new_object(&scope, root_handle, HandleSlice::EMPTY)
            .as_handle(&scope);
        add_prop(thread, &scope, obj, name_a, 1);
        let map_a = map_handle(thread, &scope, obj);
        add_prop(thread, &scope, obj, name_b, 2);
        let map_ab = map_handle(thread, &scope, obj);

        let other = thread
            .heap()
            .new_object(&scope, root_handle, HandleSlice::EMPTY)
            .as_handle(&scope);
        add_prop(thread, &scope, other, name_a, 3);
        let map_a_again = map_handle(thread, &scope, other);

        let heap = &*thread.heap();
        (
            root_handle.as_tagged(heap).raw(),
            map_a.as_tagged(heap).raw(),
            map_ab.as_tagged(heap).raw(),
            map_a_again.as_tagged(heap).raw(),
        )
    });

    assert_eq!(pred_word(&mut thread, root), None);
    assert_eq!(pred_word(&mut thread, map_a), Some(root));
    assert_eq!(pred_word(&mut thread, map_ab), Some(map_a));
    assert_eq!(root_word(&mut thread, map_ab), root);
    assert_eq!(map_a_again, map_a);
    assert_eq!(pred_word(&mut thread, map_a_again), Some(root));
}

#[test]
fn redefine_records_pred() {
    let (_vm, mut thread) = thread();

    let (map_a, map_a_readonly) = thread.handle_scope(|thread, scope| {
        let name_a = intern_name(thread, &scope, "a");
        let root = thread.heap().known().object_initial_map;
        let obj = thread
            .heap()
            .new_object(&scope, root, HandleSlice::EMPTY)
            .as_handle(&scope);

        add_prop(thread, &scope, obj, name_a, 1);
        let map_a = map_handle(thread, &scope, obj);

        let v = scope.handle(Smi::new(1));
        Object::define_own_property(
            thread.heap(),
            &scope,
            obj,
            name_a,
            PropertyDescriptor::Data {
                value: v,
                writable: false,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        let map_a_readonly = map_handle(thread, &scope, obj);

        let heap = &*thread.heap();
        (
            map_a.as_tagged(heap).raw(),
            map_a_readonly.as_tagged(heap).raw(),
        )
    });

    assert_ne!(map_a_readonly, map_a);
    assert_eq!(pred_word(&mut thread, map_a_readonly), Some(map_a));
}

#[test]
fn delete_records_pred() {
    let (_vm, mut thread) = thread();

    let (map_ab, map_b) = thread.handle_scope(|thread, scope| {
        let name_a = intern_name(thread, &scope, "a");
        let name_b = intern_name(thread, &scope, "b");
        let root = thread.heap().known().object_initial_map;
        let obj = thread
            .heap()
            .new_object(&scope, root, HandleSlice::EMPTY)
            .as_handle(&scope);

        add_prop(thread, &scope, obj, name_a, 1);
        add_prop(thread, &scope, obj, name_b, 2);
        let map_ab = map_handle(thread, &scope, obj);

        let key = {
            let heap = &*thread.heap();
            scope.handle(name_a.as_tagged(heap).erase())
        };
        assert!(
            Object::delete_own_property(thread.heap(), &scope, obj, key).unwrap(),
            "the property is configurable and must delete"
        );
        let map_b = map_handle(thread, &scope, obj);

        let heap = &*thread.heap();
        (map_ab.as_tagged(heap).raw(), map_b.as_tagged(heap).raw())
    });

    assert_ne!(map_b, map_ab);
    assert_eq!(pred_word(&mut thread, map_b), Some(map_ab));
}

#[test]
fn live_transition_subtree_survives_gc() {
    let (_vm, mut thread) = thread();

    thread.handle_scope(|thread, scope| {
        let name_a = intern_name(thread, &scope, "a");
        let name_b = intern_name(thread, &scope, "b");
        let root_handle = thread.heap().known().object_initial_map;

        let obj = thread
            .heap()
            .new_object(&scope, root_handle, HandleSlice::EMPTY)
            .as_handle(&scope);
        add_prop(thread, &scope, obj, name_a, 1);
        add_prop(thread, &scope, obj, name_b, 2);

        thread.heap().collect();

        let map_ab = object_map_word(thread, obj);
        let heap = &*thread.heap();
        let root = root_handle.as_tagged(heap);
        assert!(
            transition_target(heap, root, name_a.as_tagged(heap)).is_some(),
            "root -> {{a}} must survive while an object uses {{a, b}}"
        );
        let map_ab = unsafe { map_ab.assume_valid(heap) }
            .get_as::<Map>()
            .expect("map");
        assert_eq!(
            map_ab.root_map(heap).raw(),
            root.raw(),
            "the pred chain must still reach the root"
        );
        let map_a = map_ab.pred(heap).expect("intermediate map alive");
        assert!(
            transition_target(heap, map_a, name_b.as_tagged(heap)).is_some(),
            "{{a}} -> {{a, b}} must survive"
        );
    });
}

#[test]
fn dead_transition_targets_are_cleared_by_gc() {
    let (_vm, mut thread) = thread();

    thread.handle_scope(|thread, scope| {
        let name_a = intern_name(thread, &scope, "a");
        let name_b = intern_name(thread, &scope, "b");
        let root_handle = thread.heap().known().object_initial_map;

        let live = thread
            .heap()
            .new_object(&scope, root_handle, HandleSlice::EMPTY)
            .as_handle(&scope);
        add_prop(thread, &scope, live, name_a, 1);

        thread.handle_scope(|thread, inner| {
            let dead = thread
                .heap()
                .new_object(&inner, root_handle, HandleSlice::EMPTY)
                .as_handle(&inner);
            add_prop(thread, &inner, dead, name_b, 2);
        });

        let heap = &*thread.heap();
        let root = root_handle.as_tagged(heap);
        assert!(transition_target(heap, root, name_a.as_tagged(heap)).is_some());
        assert!(transition_target(heap, root, name_b.as_tagged(heap)).is_some());

        thread.heap().collect();

        let heap = &*thread.heap();
        let root = root_handle.as_tagged(heap);
        assert!(
            transition_target(heap, root, name_a.as_tagged(heap)).is_some(),
            "the live {{a}} branch must survive"
        );
        assert!(
            transition_target(heap, root, name_b.as_tagged(heap)).is_none(),
            "the dead {{b}} branch's weak entry must be cleared"
        );
    });
}

#[test]
fn unused_whole_subtree_is_collected() {
    let (_vm, mut thread) = thread();

    thread.handle_scope(|thread, scope| {
        let name_a = intern_name(thread, &scope, "a");
        let name_b = intern_name(thread, &scope, "b");
        let root_handle = thread.heap().known().object_initial_map;

        thread.handle_scope(|thread, inner| {
            let obj = thread
                .heap()
                .new_object(&inner, root_handle, HandleSlice::EMPTY)
                .as_handle(&inner);
            add_prop(thread, &inner, obj, name_a, 1);
            add_prop(thread, &inner, obj, name_b, 2);
        });

        thread.heap().collect();

        let heap = &*thread.heap();
        let root = root_handle.as_tagged(heap);
        assert!(
            transition_target(heap, root, name_a.as_tagged(heap)).is_none(),
            "an unreferenced subtree must not be retained"
        );
    });
}
