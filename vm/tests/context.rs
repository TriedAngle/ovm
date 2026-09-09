use dummy_heap::{DummyHeap, DummyHeapConfig};
use vm::VM;
use vm::{
    CallableInfoInit, CallableInfoObject, Context, ContextInit, FixedArray, FixedByteArray,
    ObjectKind, ObjectSlotsInit, ScopeInfo, Smi,
};

fn empty_scope_info(thread: &mut vm::Thread) -> vm::Global<ScopeInfo> {
    thread.heap().known().empty_scope_info
}

#[test]
fn empty_context_is_the_well_known_root() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (kind, outer, len) = thread.heap().no_gc(|nogc| {
        let ctx = nogc.known().empty_context.heap_ref(nogc).as_ref();
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
        let scope_info = empty_scope_info(thread);
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(42).encode()], &scope);
        let inner = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: None,
                slots,
                scope_info,
            },
            &scope,
        );
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(7).encode()], &scope);
        let outer = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: Some(inner),
                slots,
                scope_info,
            },
            &scope,
        );

        let (own, via_outer) = thread.heap().no_gc(|nogc| {
            let o = outer.heap_ref(nogc).as_ref();
            (
                o.slots.heap_ref(nogc).at(0),
                o.outer
                    .heap_ref(nogc)
                    .expect("outer context")
                    .as_ref()
                    .slots
                    .heap_ref(nogc)
                    .at(0),
            )
        });
        assert_eq!(own.to_i64().unwrap(), 7);
        assert_eq!(via_outer.to_i64().unwrap(), 42);
    });
}

#[test]
fn closure_object_carries_typed_context() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let scope_info = empty_scope_info(thread);
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(9).encode()], &scope);
        let context = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: None,
                slots,
                scope_info,
            },
            &scope,
        );

        let bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );

        let map = thread.heap().known().function_map;
        let obj = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[info.value(), context.value()],
                    elements: void.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);

        let slot0 = thread.heap().no_gc(|nogc| {
            let Some(o) = obj.value().as_heap_object(nogc) else {
                panic!("callable must be an object");
            };
            let context = o
                .as_ref()
                .closure_context(nogc)
                .expect("context must be typed as Context");
            context.slots.heap_ref(nogc).at(0)
        });
        assert_eq!(slot0.to_i64().unwrap(), 9);
    });
}
