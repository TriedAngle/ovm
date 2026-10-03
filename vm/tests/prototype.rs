use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    Handle, HandleScope, HandleSlice, Map, MapInit, MapKind, Object, PropertyDescriptor, SlotName,
    Smi, Thread, VM, Value,
};

fn thread() -> (VM, Thread) {
    let vm = vm::VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
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

fn map_of<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    obj: Handle<'_, Object>,
) -> Handle<'s, Map> {
    let heap = &*thread.heap();
    scope.handle(obj.as_tagged(heap).as_ref().map_ref(heap))
}

fn private_map<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    prototype: Handle<'_, Value>,
) -> Handle<'s, Map> {
    let heap = thread.heap();
    heap.allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
            value_slot_count: 0,
            descriptors: &[],
            prototype,
        },
        scope,
    )
}

fn new_object<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    map: Handle<'_, Map>,
) -> Handle<'s, Object> {
    let heap = thread.heap();
    heap.new_object(scope, map, HandleSlice::EMPTY)
        .as_handle(scope)
}

fn as_value<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    obj: Handle<'_, Object>,
) -> Handle<'s, Value> {
    let heap = &*thread.heap();
    scope.handle(obj.as_tagged(heap).erase())
}

fn get_or_create<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    map: Handle<'s, Map>,
) -> Option<Handle<'s, vm::Cell>> {
    vm::Prototype::get_or_create_prototype_chain_validity_cell(thread.heap(), scope, map)
}

/// `o -> p -> object_prototype`, with a private map per object.
fn build_chain<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
) -> (
    Handle<'s, Object>,
    Handle<'s, Object>,
    Handle<'s, Map>,
    Handle<'s, Map>,
) {
    let object_prototype = {
        let heap = thread.heap();
        heap.known().object_prototype.erase()
    };
    let proto_map = private_map(thread, scope, object_prototype);
    let proto = new_object(thread, scope, proto_map);
    let proto_value = as_value(thread, scope, proto);
    let receiver_map = private_map(thread, scope, proto_value);
    let receiver = new_object(thread, scope, receiver_map);
    (proto, receiver, proto_map, receiver_map)
}

/// `o -> p -> gp -> object_prototype`, with a private map per object.
fn build_three_level_chain<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
) -> (
    Handle<'s, Object>,
    Handle<'s, Object>,
    Handle<'s, Map>,
    Handle<'s, Map>,
    Handle<'s, Map>,
) {
    let object_prototype = {
        let heap = thread.heap();
        heap.known().object_prototype.erase()
    };
    let gp_map = private_map(thread, scope, object_prototype);
    let gp = new_object(thread, scope, gp_map);
    let gp_value = as_value(thread, scope, gp);
    let p_map = private_map(thread, scope, gp_value);
    let p = new_object(thread, scope, p_map);
    let p_value = as_value(thread, scope, p);
    let o_map = private_map(thread, scope, p_value);
    let _o = new_object(thread, scope, o_map);
    (gp, p, gp_map, p_map, o_map)
}

#[test]
fn fresh_maps_start_invalid_or_sentinel_per_kind() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, _scope| {
        let heap = &*thread.heap();
        let known = heap.known();
        let invalid = known.invalid_prototype_validity_cell.as_tagged(heap);

        // A JSReceiver map starts at the shared invalid cell, so its cell
        // reads invalid but no cell had to be allocated.
        let object_map = known.object_initial_map.as_tagged(heap);
        let published = object_map
            .as_ref()
            .published_validity_cell(heap)
            .expect("JSReceiver maps start with the invalid cell");
        assert!(published.ptr_eq(invalid));
        assert!(!object_map.as_ref().is_prototype_validity_cell_valid(heap));
        assert!(invalid.as_ref().is_cleared());

        // Non-JSReceiver maps carry the Smi(0) sentinel instead.
        let float_map = known.float_map.as_tagged(heap);
        assert!(float_map.as_ref().published_validity_cell(heap).is_none());
        assert!(float_map.as_ref().is_prototype_validity_cell_valid(heap));
    });
}

#[test]
fn get_or_create_publishes_valid_cell_and_shared_info() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let (_proto, _receiver, proto_map, receiver_map) = build_chain(thread, &scope);

        let cell = get_or_create(thread, &scope, receiver_map)
            .expect("ordinary prototype chains are guarded");
        {
            let heap = &*thread.heap();
            assert!(cell.as_tagged(heap).as_ref().is_valid(heap));

            let proto_map = proto_map.as_tagged(heap);
            assert!(proto_map.as_ref().is_prototype());
            assert!(proto_map.as_ref().try_get_prototype_info(heap).is_some());
            let published = proto_map
                .as_ref()
                .published_validity_cell(heap)
                .expect("holder owns the cell");
            assert!(published.ptr_eq(cell.as_tagged(heap)));
        }

        // Repeated creation reuses the same valid cell.
        let again = get_or_create(thread, &scope, receiver_map).unwrap();
        {
            let heap = &*thread.heap();
            assert!(again.as_tagged(heap).ptr_eq(cell.as_tagged(heap)));
        }
    });
}

#[test]
fn prototype_shape_change_invalidates_and_replaces_cell() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let (proto, _receiver, proto_map, receiver_map) = build_chain(thread, &scope);
        let cell = get_or_create(thread, &scope, receiver_map).unwrap();
        let name = intern_name(thread, &scope, "x");

        add_prop(thread, &scope, proto, name, 1);
        let new_map = map_of(thread, &scope, proto);

        {
            let heap = &*thread.heap();
            assert!(cell.as_tagged(heap).as_ref().is_cleared());
            assert!(!proto_map.as_tagged(heap).ptr_eq(new_map.as_tagged(heap)));
        }

        // The next cell creation observes the cleared holder field and
        // publishes a fresh, valid cell.
        let fresh = get_or_create(thread, &scope, receiver_map).unwrap();
        {
            let heap = &*thread.heap();
            assert!(!fresh.as_tagged(heap).ptr_eq(cell.as_tagged(heap)));
            assert!(fresh.as_tagged(heap).as_ref().is_valid(heap));
            assert!(
                new_map
                    .as_tagged(heap)
                    .as_ref()
                    .published_validity_cell(heap)
                    .unwrap()
                    .ptr_eq(fresh.as_tagged(heap))
            );
        }
    });
}

#[test]
fn upstream_shape_change_invalidates_downstream_cell() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let (gp, _p, gp_map, p_map, o_map) = build_three_level_chain(thread, &scope);
        let cell = get_or_create(thread, &scope, o_map).expect("three-level chain is guarded");
        {
            let heap = &*thread.heap();
            assert!(cell.as_tagged(heap).as_ref().is_valid(heap));
            let p_info = p_map
                .as_tagged(heap)
                .as_ref()
                .try_get_prototype_info(heap)
                .expect("intermediate prototype has info");
            assert!(p_info.as_ref().registry_slot() >= 1);
            assert!(gp_map.as_tagged(heap).as_ref().is_prototype());
            let gp_info = gp_map
                .as_tagged(heap)
                .as_ref()
                .try_get_prototype_info(heap)
                .expect("grandparent prototype has info");
            let users = gp_info.as_ref().prototype_users(heap).unwrap();
            let users = users.as_ref();
            let mut live_user_of_p = false;
            for i in 1..users.len() {
                if let Some(user) = users.element_slot(i).get(heap).as_strong()
                    && user.ptr_eq(p_map.as_tagged(heap).erase())
                {
                    live_user_of_p = true;
                }
            }
            assert!(live_user_of_p, "p_map registered with gp_map");
        }

        let name = intern_name(thread, &scope, "y");
        add_prop(thread, &scope, gp, name, 1);

        let heap = &*thread.heap();
        assert!(
            cell.as_tagged(heap).as_ref().is_cleared(),
            "grandparent shape change propagates through the weak user link"
        );
    });
}

#[test]
fn dead_user_maps_clear_from_the_registry() {
    let (_vm, mut thread) = thread();
    thread.handle_scope(|thread, scope| {
        let (_gp, p, _gp_map, p_map, _o_map) = build_three_level_chain(thread, &scope);

        // A throwaway prototype `q` (whose prototype is `p`) plus a throwaway
        // receiver `r` (whose prototype is `q`). Creating r's cell makes
        // q_map a registered *user* of p_map, weakly.
        thread.handle_scope(|thread, inner| {
            let p_value = as_value(thread, &inner, p);
            let q_map = private_map(thread, &inner, p_value);
            let q = new_object(thread, &inner, q_map);
            let q_value = as_value(thread, &inner, q);
            let r_map = private_map(thread, &inner, q_value);
            let _r = new_object(thread, &inner, r_map);
            assert!(get_or_create(thread, &inner, r_map).is_some());
        });

        // Full collection: q/q_map are unreachable; the weak registry slot
        // in p_map clears.
        thread.heap().collect();

        let heap = &*thread.heap();
        let info = p_map
            .as_tagged(heap)
            .as_ref()
            .try_get_prototype_info(heap)
            .expect("p_map forced info");
        let users = info.as_ref().prototype_users(heap).unwrap();
        let users = users.as_ref();
        assert!(users.len() > 1, "the dead user left an array entry");
        for i in 1..users.len() {
            assert!(
                users.element_slot(i).get(heap).as_strong().is_none(),
                "dead user entries are cleared by the collector"
            );
        }
    });
}

#[test]
fn repeated_chains_and_invalidation_under_gc_stress_do_not_crash() {
    let (_vm, mut thread) = thread();
    for _ in 0..8 {
        thread.handle_scope(|thread, scope| {
            let (proto, _receiver, _proto_map, receiver_map) = build_chain(thread, &scope);
            let cell = get_or_create(thread, &scope, receiver_map).unwrap();
            let name = intern_name(thread, &scope, "z");
            add_prop(thread, &scope, proto, name, 1);
            {
                let heap = &*thread.heap();
                assert!(cell.as_tagged(heap).as_ref().is_cleared());
            }
            get_or_create(thread, &scope, receiver_map).unwrap();
        });
        thread.heap().collect();
    }
}
