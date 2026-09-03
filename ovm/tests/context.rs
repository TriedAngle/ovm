use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::VM;
use vm::{
    CallableInfoInit, CallableInfoObject, Context, ContextInit, FixedArray, FixedByteArray, Map,
    MapInit, MapKind, ObjectKind, ObjectSlotsInit, Smi,
};

#[test]
fn empty_context_is_the_well_known_root() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (kind, outer, len) = thread.heap().no_gc(|nogc, heap| {
        let ctx = heap.known().empty_context.heap_ref(nogc).as_ref();
        (
            ctx.header.map.heap_ref(nogc).kind().kind(),
            ctx.outer.inner(),
            ctx.slots.heap_ref(nogc).as_ref().len(),
        )
    });
    assert_eq!(kind, ObjectKind::Context);
    assert_eq!(
        outer,
        thread.heap().known().void.value(),
        "no outer context"
    );
    assert_eq!(len, 0, "no context slots");
}

#[test]
fn contexts_chain_through_outer() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(42).encode()], &scope);
        let inner = thread
            .heap()
            .allocate_handle::<Context>(ContextInit { outer: None, slots }, &scope);
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(7).encode()], &scope);
        let outer = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: Some(inner),
                slots,
            },
            &scope,
        );

        let (own, via_outer) = thread.heap().no_gc(|nogc, heap| {
            let o = outer.heap_ref(nogc).as_ref();
            (
                o.slots.heap_ref(nogc).at(0),
                o.outer
                    .heap_ref(nogc, heap)
                    .expect("outer context")
                    .as_ref()
                    .slots
                    .heap_ref(nogc)
                    .at(0),
            )
        });
        assert_eq!(Smi::decode(own).unwrap().value(), 7);
        assert_eq!(Smi::decode(via_outer).unwrap().value(), 42);
    });
}

#[test]
fn callable_info_carries_typed_context() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(9).encode()], &scope);
        let context = thread
            .heap()
            .allocate_handle::<Context>(ContextInit { outer: None, slots }, &scope);

        let bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                context,
                handlers: None,
            },
            &scope,
        );

        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT.union(MapKind::CALLABLE),
                value_slot_count: 1,
                descriptors: &[],
            },
            &scope,
        );
        let obj = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[info.as_tagged().erase()],
                    elements: void.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);

        let slot0 = thread.heap().no_gc(|nogc, heap| {
            let vm::ValueRef::Object(o) = obj.value().value_ref(nogc) else {
                panic!("callable must be an object");
            };
            let info = o.as_ref().callable_info(nogc, heap).unwrap();
            let context = info
                .context
                .inner()
                .get_as::<Context>(nogc, heap.known().context_map)
                .expect("context must be typed as Context");
            context.slots.heap_ref(nogc).at(0)
        });
        assert_eq!(Smi::decode(slot0).unwrap().value(), 9);
    });
}
