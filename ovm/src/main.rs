use bytecode::{Opcode, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::VM;
use ovm::natives::NativeIndex;
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, Float, LocalHeap, Map,
    MapInit, MapKind, ObjectSlotsInit, Smi,
};

fn main() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).expect("failed to create heap");
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let interned = thread.intern(&scope, "hello, ovm");
        let again = thread.intern(&scope, "hello, ovm");
        thread.heap().no_gc(|nogc, _| {
            let s = interned.heap_ref(nogc);
            println!(
                "interned: {:?} (hash {}, deduped: {})",
                s.string().as_str(nogc).expect("utf8"),
                s.string().hash(),
                interned.value().to_bits() == again.value().to_bits()
            );
        });
    });
    println!(
        "heap initialized: {} bytes ({} used)",
        vm.heap().capacity(),
        vm.heap().used()
    );

    let receiver = Smi::new(0).encode();
    let result = thread
        .run_native(
            vm.native(NativeIndex::SMI_ADD),
            &[receiver, Smi::new(6).encode(), Smi::new(7).encode()],
        )
        .expect("smi_add failed");
    println!("smi_add(6, 7) = {}", Smi::decode(result).unwrap().value());

    let fa = thread.heap().allocate::<Float>(1.5).erase();
    let fb = thread.heap().allocate::<Float>(2.25).erase();
    let result = thread
        .run_native(vm.native(NativeIndex::FLOAT_ADD), &[receiver, fa, fb])
        .expect("float_add failed");
    let out = thread.heap().no_gc(|nogc, heap| {
        result
            .get_as::<Float>(nogc, heap.known().float_map)
            .expect("float_add returned a float")
            .value
            .get()
    });
    println!("float_add(1.5, 2.25) = {}", out);

    // return 6 + 7
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[6]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[7]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::Add, &[0, 1]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        // TODO: we should have a helper for this callable object creation section
        let void = thread.heap().known().void.value();
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(program.as_slice(), &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 2,
                context: void,
            },
            &scope,
        );
        let callable_map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT.union(MapKind::CALLABLE),
                value_slot_count: 1,
                descriptors: &[],
            },
            &scope,
        );
        let callable_obj = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map: callable_map,
                    values: &[callable.as_tagged().erase()],
                    elements: void,
                    length: 0,
                },
            )
            .into_handle(&scope);
        thread.execute(callable_obj, &[])
    });

    match result {
        Ok(acc) => println!("result: {}", Smi::decode(acc).unwrap().value()),
        Err(err) => {
            eprintln!("runtime error: {err:?}");
            std::process::exit(1);
        }
    }
}
