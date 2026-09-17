use mark_sweep::{MarkSweep, MarkSweepConfig};
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (kind, outer, len) = thread.heap().no_gc(|heap| {
        let ctx = heap.known().empty_context.heap_ref(heap).as_ref();
        (
            ctx.header.map.heap_ref(heap).kind().kind(),
            vm::Value::from_bits(ctx.outer.as_raw().load()),
            ctx.slots.heap_ref(heap).as_ref().len(),
        )
    });
    assert_eq!(kind, ObjectKind::Context);
    let heap = thread.heap();
    assert_eq!(
        outer,
        heap.known().the_hole.as_tagged(heap).erase(),
        "no outer context"
    );
    assert_eq!(len, 0, "no context slots");
}

#[test]
fn contexts_chain_through_outer() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let scope_info = empty_scope_info(thread);
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[Smi::new(42).into_tagged()]), &scope);
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
            .allocate_handle::<FixedArray>(scope.stage(&[Smi::new(7).into_tagged()]), &scope);
        let outer = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: Some(inner),
                slots,
                scope_info,
            },
            &scope,
        );

        let (own, via_outer) = thread.heap().no_gc(|heap| {
            let o = outer.heap_ref(heap).as_ref();
            (
                o.slots.heap_ref(heap).at(heap, 0).erase(),
                o.outer
                    .heap_ref(heap)
                    .expect("outer context")
                    .as_ref()
                    .slots
                    .heap_ref(heap)
                    .at(heap, 0)
                    .erase(),
            )
        });
        assert_eq!(own.to_i64().unwrap(), 7);
        assert_eq!(via_outer.to_i64().unwrap(), 42);
    });
}

#[test]
fn closure_object_carries_typed_context() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let the_hole = thread.heap().known().the_hole;
        let scope_info = empty_scope_info(thread);
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[Smi::new(9).into_tagged()]), &scope);
        let context = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: None,
                slots,
                scope_info,
            },
            &scope,
        );

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

        let map = thread.heap().known().function_map;
        let values = {
            let heap = &*thread.heap();
            scope.stage(&[
                info.as_tagged(heap).erase_type(),
                context.as_tagged(heap).erase_type(),
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
            .into_handle(&scope);

        let slot0 = thread.heap().no_gc(|heap| {
            let o = obj.heap_ref(heap);
            let context = o
                .as_ref()
                .closure_context(heap)
                .expect("context must be typed as Context");
            context.slots.heap_ref(heap).at(heap, 0).erase()
        });
        assert_eq!(slot0.to_i64().unwrap(), 9);
    });
}
