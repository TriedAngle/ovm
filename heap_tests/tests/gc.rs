use heap_tests::for_each_backend;

use vm::{FixedArray, HeapBackend, SlotName, Smi, VMString};

fn well_known_survive_cycles<B: HeapBackend>()
where
    B::Config: Default,
{
    let vm = heap_tests::vm_with_builtins::<B>(B::Config::default());
    let mut thread = vm.attach();
    for _ in 0..3 {
        thread.heap().collect();
    }

    let known = vm.known();
    let values = [
        known.map_map.value(),
        known.the_hole.value(),
        known.undefined.value(),
        known.null.value(),
        known.smi_map.value(),
        known.float_map.value(),
        known.array_map.value(),
        known.string_map.value(),
        known.object_prototype.value(),
        known.object_initial_map.value(),
        known.empty_fixed_array.value(),
        known.empty_context.value(),
        known.strings.length.value(),
        known.strings.prototype.value(),
    ];
    for value in values {
        assert!(value.is_strong_ptr());
        assert!(vm.heap().contains(value.to_bits() & !vm::TAG_MASK));
    }

    // the interner's strong entries keep well-known strings unique
    thread.handle_scope(|t, scope| {
        let again = t.intern(&scope, "length");
        assert_eq!(
            SlotName::from(again.as_tagged()).tagged().erase().to_bits(),
            known.strings.length.value().to_bits()
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
        let string = VMString::from_bytes(t.heap(), &scope, b"survivor");
        let array = t
            .heap()
            .allocate_handle::<FixedArray>(&[Smi::new(42).encode(), string.value()], &scope);
        let before = (string.value().to_bits(), array.value().to_bits());

        t.heap().collect();
        t.heap().collect();

        assert_eq!(string.value().to_bits(), before.0);
        assert_eq!(array.value().to_bits(), before.1);
        t.heap().no_gc(|nogc| {
            let array = array.heap_ref(nogc);
            assert_eq!(Smi::decode(array.at(0)).unwrap().value(), 42);
            assert_eq!(array.at(1), string.value());
            let string = string.heap_ref(nogc);
            assert_eq!(string.as_slice(nogc), b"survivor");
        });
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
                let _ = t.heap().allocate::<FixedArray>(&[Smi::new(0).encode()]);
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
            let result = thread.run_script(script);
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
