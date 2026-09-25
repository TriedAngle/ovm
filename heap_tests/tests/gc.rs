use heap_tests::for_each_backend;

use vm::{DenseString, FixedArray, HeapBackend, Smi};

fn well_known_survive_cycles<B: HeapBackend>()
where
    B::Config: Default,
{
    let vm = heap_tests::vm_with_builtins::<B>(B::Config::default());
    let mut thread = vm.attach();
    for _ in 0..3 {
        thread.heap().collect();
    }

    let heap = &*thread.heap();
    let known = vm.known();
    let values = [
        known.map_map.as_tagged(heap).raw(),
        known.the_hole.as_tagged(heap).raw(),
        known.undefined.as_tagged(heap).raw(),
        known.null.as_tagged(heap).raw(),
        known.smi_map.as_tagged(heap).raw(),
        known.float_map.as_tagged(heap).raw(),
        known.array_map.as_tagged(heap).raw(),
        known.dense_latin1_string_map.as_tagged(heap).raw(),
        known.dense_utf16_string_map.as_tagged(heap).raw(),
        known.object_prototype.as_tagged(heap).raw(),
        known.object_initial_map.as_tagged(heap).raw(),
        known.empty_fixed_array.as_tagged(heap).raw(),
        known.empty_context.as_tagged(heap).raw(),
        known.strings.length.as_tagged(heap).raw(),
        known.strings.prototype.as_tagged(heap).raw(),
    ];
    for value in values {
        assert!(value.is_strong_ptr());
        assert!(vm.heap().contains(value.to_bits() & !vm::TAG_MASK));
    }

    // the interner's strong entries keep well-known strings unique
    thread.handle_scope(|t, scope| {
        let again = t.intern(&scope, "length");
        let heap = &*t.heap();
        assert_eq!(
            // Interned names compare by word identity.
            again.as_tagged(heap).raw().to_bits(),
            known.strings.length.as_tagged(heap).raw().to_bits()
        );
        let _ = scope;
    });
}

fn handles_survive_across_cycles<B: HeapBackend>()
where
    B::Config: Default,
{
    let vm = heap_tests::vm::<B>(B::Config::default());
    let mut thread = vm.attach();

    thread.handle_scope(|t, scope| {
        let string = DenseString::from_latin1(t.heap(), &scope, b"survivor");
        {
            let staged = {
                let heap = &*t.heap();
                scope.stage(&[Smi::new(42).into_tagged(), string.as_tagged(heap).erase()])
            };
            let array = t.heap().allocate_handle::<FixedArray>(staged, &scope);

            t.heap().collect();
            t.heap().collect();

            // a full collection promotes (moves) young objects, so compare
            // survival, not addresses: the rooted handles must still resolve
            // to live strong pointers in the heap
            {
                let heap = &*t.heap();
                let string = string.as_tagged(heap).raw();
                let array = array.as_tagged(heap).raw();
                assert!(string.is_strong_ptr() && vm.heap().contains(string.raw_addr()));
                assert!(array.is_strong_ptr() && vm.heap().contains(array.raw_addr()));
            }
            {
                let heap = &*t.heap();
                let array = array.as_tagged(heap);
                assert_eq!(Smi::decode(array.at(heap, 0).raw()).unwrap().value(), 42);
                assert_eq!(array.at(heap, 1).raw(), string.as_tagged(heap).raw());
                let string = string.as_tagged(heap);
                assert!(string.data(heap).matches_ascii(b"survivor"));
            };
        }
    });
}

fn cycle_does_not_grow_heap<B: HeapBackend>()
where
    B::Config: Default,
{
    let vm = heap_tests::vm::<B>(B::Config::default());
    {
        let mut thread = vm.attach();
        thread.handle_scope(|t, _scope| {
            for _ in 0..8 {
                let _ = t
                    .heap()
                    .allocate::<FixedArray>(_scope.stage(&[Smi::new(0).into_tagged()]));
            }
        });
    }

    // no local attached: the global entry may run cycles from this thread
    let before = vm.heap().stats().used;
    for _ in 0..5 {
        vm.heap().collect();
    }
    let after = vm.heap().stats().used;
    assert!(
        after <= before,
        "used grew during no-op cycles: {before} -> {after}"
    );
}

fn gc_during_interpretation<B: HeapBackend>()
where
    B::Config: Default,
{
    let vm = heap_tests::vm::<B>(B::Config::default());
    let script = "\
        var a = []; var s = 0; \
        for (var i = 0; i < 20000; i++) { a[i % 64] = { x: i, y: i + 1 }; s += i; } \
        s";

    let mut workers = Vec::new();
    for _ in 0..2 {
        let vm2 = vm.clone();
        workers.push(std::thread::spawn(move || {
            let mut thread = vm2.attach();
            let result = thread.eval::<vm::JavascriptCompiler>(script);
            result.map(|v| v.to_i64().expect("smi result"))
        }));
    }

    for _ in 0..25 {
        vm.heap().collect();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    for worker in workers {
        assert_eq!(worker.join().unwrap().unwrap(), 199990000);
    }
}

for_each_backend!(
    well_known_survive_cycles,
    handles_survive_across_cycles,
    cycle_does_not_grow_heap,
    gc_during_interpretation,
);
