use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{Thread, VM};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, HandlerEntryInit,
    HandlerTable, HandlerTableInit, LocalHeap, Map, MapInit, MapKind, ObjectSlotsInit, ValueRef,
};

fn table<'s>(
    thread: &mut Thread<DummyHeap>,
    scope: &'s vm::HandleScope<'_>,
    entries: &[HandlerEntryInit],
) -> vm::Handle<'s, HandlerTable> {
    thread
        .heap()
        .allocate_handle::<HandlerTable>(HandlerTableInit { entries }, scope)
}

#[test]
fn roundtrip_entries() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(
            thread,
            &scope,
            &[
                HandlerEntryInit::new(0, 10, 40),
                HandlerEntryInit::new(12, 20, 55),
            ],
        );
        let out = thread.heap().no_gc(|nogc, heap| {
            let table = t
                .value()
                .get_as::<HandlerTable>(nogc, heap.known().handler_table_map)
                .expect("handler table value");
            (table.len(), table.entry(0), table.entry(1))
        });
        assert_eq!(out.0, 2);
        assert_eq!(out.1, HandlerEntryInit::new(0, 10, 40));
        assert_eq!(out.2, HandlerEntryInit::new(12, 20, 55));
    });
}

#[test]
fn lookup_finds_handler_inside_range() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[HandlerEntryInit::new(5, 15, 100)]);
        thread.heap().no_gc(|nogc, heap| {
            let table = t
                .value()
                .get_as::<HandlerTable>(nogc, heap.known().handler_table_map)
                .unwrap();
            // try_start is inside (inclusive) ...
            assert_eq!(table.lookup(5), Some(100));
            // ... as is any offset before try_end
            assert_eq!(table.lookup(14), Some(100));
        });
    });
}

#[test]
fn lookup_returns_none_outside_range() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[HandlerEntryInit::new(5, 15, 100)]);
        thread.heap().no_gc(|nogc, heap| {
            let table = t
                .value()
                .get_as::<HandlerTable>(nogc, heap.known().handler_table_map)
                .unwrap();
            // before the region ...
            assert_eq!(table.lookup(4), None);
            // ... try_end is exclusive ...
            assert_eq!(table.lookup(15), None);
            // ... and beyond it
            assert_eq!(table.lookup(16), None);
        });
    });
}

#[test]
fn lookup_returns_innermost_of_nested_ranges() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        // inner entry emitted first: lookup must be independent of
        // emission order, matching properly-nested try regions
        let t = table(
            thread,
            &scope,
            &[
                HandlerEntryInit::new(4, 12, 300),
                HandlerEntryInit::new(0, 20, 100),
            ],
        );
        thread.heap().no_gc(|nogc, heap| {
            let table = t
                .value()
                .get_as::<HandlerTable>(nogc, heap.known().handler_table_map)
                .unwrap();
            // inside both ranges: the innermost (largest try_start) wins
            assert_eq!(table.lookup(5), Some(300));
            // inside the outer range only
            assert_eq!(table.lookup(1), Some(100));
            assert_eq!(table.lookup(15), Some(100));
        });
    });
}

#[test]
fn lookup_on_empty_table_returns_none() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[]);
        thread.heap().no_gc(|nogc, heap| {
            let table = t
                .value()
                .get_as::<HandlerTable>(nogc, heap.known().handler_table_map)
                .unwrap();
            assert_eq!(table.len(), 0);
            assert_eq!(table.lookup(0), None);
        });
    });
}

#[test]
fn callable_info_carries_handler_table() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread: &mut Thread<DummyHeap>, scope| {
        let void = thread.heap().known().void.value();
        let empty_context = thread.heap().known().empty_context;
        let t = table(thread, &scope, &[HandlerEntryInit::new(2, 8, 33)]);

        let bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                context: empty_context,
                handlers: Some(t),
            },
            &scope,
        );

        // wrap in a callable object so `callable_info` can be exercised
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
                    elements: void,
                    length: 0,
                },
            )
            .into_handle(&scope);

        let result = thread.heap().no_gc(|nogc, heap| {
            let ValueRef::Object(o) = obj.value().value_ref(nogc) else {
                panic!("callable must be an object");
            };
            let info = o.as_ref().callable_info(nogc, heap).unwrap();
            let table = info
                .handlers
                .heap_ref(nogc, heap)
                .expect("handler table attached");
            table.lookup(5)
        });
        assert_eq!(result, Some(33));
    });
}

#[test]
fn callable_info_without_handler_table() {
    let vm = VM::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread: &mut Thread<DummyHeap>, scope| {
        let void = thread.heap().known().void.value();
        let empty_context = thread.heap().known().empty_context;
        let bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        thread.heap().no_gc(|nogc, heap| {
            let h = info.heap_ref(nogc).handlers.heap_ref(nogc, heap);
            assert!(h.is_none(), "void handlers slot must mean no table");
        });
    });
}
