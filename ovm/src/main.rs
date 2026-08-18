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
        let zero = Smi::new(0).encode();
        let _handle = thread
            .heap()
            .allocate_handle::<FixedArray>(&[zero, zero, zero], &scope);

        let interned = thread.intern(&scope, "hello, ovm");
        let again = thread.intern(&scope, "hello, ovm");
        thread.heap().no_gc(|nogc, _| {
            let s = interned.heap_ref(nogc);
            println!(
                "interned: {:?} (hash {}, deduped: {})",
                s.string().as_str().expect("utf8"),
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
        nogc.get_as::<Float>(result, heap.known().float_map)
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
        let void = thread.heap().known().void.value();
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(program.as_slice(), &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: bytecode.as_tagged(),
                constants: constants.as_tagged(),
                register_count: 2,
                context: void,
            },
            &scope,
        );
        // wrap the callable info in a normal object with a callable map
        let map_map = thread.heap().known().map_map.as_tagged();
        let callable_map = thread.heap().allocate_handle::<Map>(
            MapInit {
                map_map,
                kind: MapKind::OBJECT.union(MapKind::CALLABLE),
                value_slot_count: 1,
                descriptors: &[],
            },
            &scope,
        );
        let callable_obj = thread
            .heap()
            .allocate_object(ObjectSlotsInit {
                map: callable_map.as_tagged(),
                values: &[callable.as_tagged().erase()],
                elements: void,
                length: 0,
            })
            .into_handle(&scope);
        thread.run(callable_obj.as_tagged(), &[])
    });

    match result {
        Ok(acc) => println!("result: {}", Smi::decode(acc).unwrap().value()),
        Err(err) => {
            eprintln!("runtime error: {err:?}");
            std::process::exit(1);
        }
    }
}
