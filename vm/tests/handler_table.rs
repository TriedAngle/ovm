use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, HandlerEntryInit,
    HandlerTable, HandlerTableInit, ObjectSlotsInit,
};
use vm::{Thread, VM};

fn table<'s>(
    thread: &mut Thread,
    scope: &'s vm::HandleScope<'_>,
    entries: &[HandlerEntryInit],
) -> vm::Handle<'s, HandlerTable> {
    thread
        .heap()
        .allocate_handle::<HandlerTable>(HandlerTableInit { entries }, scope)
}

#[test]
fn roundtrip_entries() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let out = {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            (table.len(), table.entry(0), table.entry(1))
        };
        assert_eq!(out.0, 2);
        assert_eq!(out.1, HandlerEntryInit::new(0, 10, 40));
        assert_eq!(out.2, HandlerEntryInit::new(12, 20, 55));
    });
}

#[test]
fn lookup_finds_handler_inside_range() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[HandlerEntryInit::new(5, 15, 100)]);
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            // try_start is inside (inclusive) ...
            assert_eq!(table.as_ref().lookup(5), Some(100));
            // ... as is any offset before try_end
            assert_eq!(table.as_ref().lookup(14), Some(100));
        };
    });
}

#[test]
fn lookup_returns_none_outside_range() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[HandlerEntryInit::new(5, 15, 100)]);
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            // before the region ...
            assert_eq!(table.as_ref().lookup(4), None);
            // ... try_end is exclusive ...
            assert_eq!(table.as_ref().lookup(15), None);
            // ... and beyond it
            assert_eq!(table.as_ref().lookup(16), None);
        };
    });
}

#[test]
fn lookup_returns_innermost_of_nested_ranges() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        // inner entry emitted first: lookup must be independent of
        // emission order (properly-nested try regions)
        let t = table(
            thread,
            &scope,
            &[
                HandlerEntryInit::new(4, 12, 300),
                HandlerEntryInit::new(0, 20, 100),
            ],
        );
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            // inside both ranges: the innermost (largest try_start) wins
            assert_eq!(table.as_ref().lookup(5), Some(300));
            // inside the outer range only
            assert_eq!(table.as_ref().lookup(1), Some(100));
            assert_eq!(table.as_ref().lookup(15), Some(100));
        };
    });
}

#[test]
fn lookup_on_empty_table_returns_none() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[]);
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            assert_eq!(table.len(), 0);
            assert_eq!(table.as_ref().lookup(0), None);
        };
    });
}

#[test]
fn callable_info_carries_handler_table() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread: &mut Thread, scope| {
        let the_hole = thread.heap().known().the_hole;
        let empty_context = thread.heap().known().empty_context;
        let t = table(thread, &scope, &[HandlerEntryInit::new(2, 8, 33)]);

        let bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage::<vm::Value>(&[]), &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                handlers: Some(t),
            },
            &scope,
        );

        // wrap in a callable object so `callable_info` can be exercised
        let map = thread.heap().known().function_map;
        let values = {
            let heap = &*thread.heap();
            scope.stage(&[
                info.as_tagged(heap).erase(),
                empty_context.as_tagged(heap).erase(),
            ])
        };
        let obj = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values,
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .as_handle(&scope);

        let result = {
            let heap = &*thread.heap();
            let o = obj.as_tagged(heap);
            let info = o.as_ref().callable_info(heap).unwrap();
            let table = info.handlers.get(heap).expect("handler table attached");
            table.as_ref().lookup(5)
        };
        assert_eq!(result, Some(33));
    });
}

#[test]
fn callable_info_without_handler_table() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread: &mut Thread, scope| {
        let bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage::<vm::Value>(&[]), &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );
        {
            let heap = &*thread.heap();
            let h = info.as_tagged(heap).handlers.get(heap);
            assert!(h.is_none(), "a hole handlers slot must mean no table");
        };
    });
}
