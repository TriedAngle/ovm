use bytecode::{Opcode, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{NativeIndex, VM, VmError};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, HeapPtr, LocalHeap, Map,
    MapInit, MapKind, SlotFlags, SlotName, SlotsObject, SlotsObjectInit, Smi,
};

fn smi(v: i64) -> vm::Value {
    Smi::new(v).encode()
}

fn run_program(
    thread: &mut ovm::Thread<DummyHeap>,
    program: Vec<u8>,
    register_count: usize,
    args: &[vm::Value],
) -> Result<vm::Value, ovm::VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: bytecode.as_tagged(),
                constants: constants.as_tagged(),
                register_count,
                context: void,
            },
            &scope,
        );
        thread.run(callable.as_tagged(), args)
    })
}

#[test]
fn load_smi_signed_immediates() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // byte1 range, byte1-via-sign-extension traps (128..=255 used to
    // decode as negative), and wide-range values
    for v in [0, 1, -1, 127, -128, 200, -201, 32767, -32768] {
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadSmi, &[v as u32]);
        emit(&mut program, Opcode::Return, &[]);

        let result = run_program(&mut thread, program, 0, &[]);
        assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), v as i64);
    }
}

#[test]
fn call_native_passes_receiver_and_args() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = receiver, r1 = 6, r2 = 7; CallNative smi_add, r0, 3
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[6]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[7]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(
        &mut program,
        Opcode::CallNative,
        &[NativeIndex::SMI_ADD.0 as u32, 0, 3],
    );
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 13);
}

#[test]
fn failed_run_does_not_leak_frames_into_next_run() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // Add on uninitialized (void) registers fails with a type error,
    // aborting the run with a frame still on the suspended list.
    let mut bad = Vec::new();
    emit(&mut bad, Opcode::Add, &[0, 1]);
    emit(&mut bad, Opcode::Return, &[]);
    let result = run_program(&mut thread, bad, 2, &[]);
    assert_eq!(result, Err(VmError::Type));

    // The next run on the same thread must start from a clean slate.
    let mut good = Vec::new();
    emit(&mut good, Opcode::LoadSmi, &[41]);
    emit(&mut good, Opcode::Return, &[]);
    let result = run_program(&mut thread, good, 1, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 41);
}

#[test]
fn parameters_are_readable_via_negative_registers() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]); // param 0 (receiver)
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 2, &[smi(42)]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn wide_parameter_operand_uses_two_bytes() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-201i32) as u32]); // param 200: needs wide
    emit(&mut program, Opcode::Return, &[]);

    // -201 does not fit in a signed byte, so the program must start with Wide
    assert_eq!(program[0], Opcode::Wide as u8);

    let args: Vec<_> = (0..201).map(|i| smi(i * 100)).collect();
    let result = run_program(&mut thread, program, 2, &args);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 20000);
}

#[test]
fn call_resolves_target_lookup_and_pushes_frames() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();

        // callee: returns smi 99
        let mut callee_program = Vec::new();
        emit(&mut callee_program, Opcode::LoadSmi, &[99]);
        emit(&mut callee_program, Opcode::Return, &[]);
        let callee_bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&callee_program, &scope);
        let callee_constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callee = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: callee_bytecode.as_tagged(),
                constants: callee_constants.as_tagged(),
                register_count: 2,
                context: void,
            },
            &scope,
        );

        // receiver with a "call" slot pointing at the callee
        let call_name = thread.intern(&scope, "call");
        let map_map = thread.heap().known().map_map.as_tagged();
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                map_map,
                kind: MapKind::OBJECT,
                value_slot_count: 1,
                descriptors: &[(
                    SlotName::from(call_name.as_tagged()),
                    SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                    Smi::new(0).encode(), // slot offset of the value slot
                )],
            },
            &scope,
        );
        let receiver = thread.heap().allocate_handle::<SlotsObject>(
            SlotsObjectInit {
                map: map.as_tagged(),
                values: &[callee.as_tagged().erase()],
            },
            &scope,
        );

        // caller: r0 = receiver; CallNoFeedback r0, r0, 1 -> acc
        let receiver_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[receiver.as_tagged().erase()], &scope);
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::CallNoFeedback, &[0, 0, 1]);
        emit(&mut program, Opcode::Return, &[]);
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let caller = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: bytecode.as_tagged(),
                constants: receiver_consts.as_tagged(),
                register_count: 2,
                context: void,
            },
            &scope,
        );

        thread.run(caller.as_tagged(), &[])
    });

    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 99);
}

#[test]
fn create_array_literal_fills_from_registers() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[2]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::CreateArrayLiteral, &[0, 3]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    let array = result.unwrap();
    thread.heap().no_gc(|nogc, heap| {
        let a = nogc
            .get_as::<FixedArray>(array, heap.known().array_map)
            .expect("array literal result");
        assert_eq!(Smi::decode(a.at(0)).unwrap().value(), 1);
        assert_eq!(Smi::decode(a.at(1)).unwrap().value(), 2);
        assert_eq!(Smi::decode(a.at(2)).unwrap().value(), 3);
    });
}

#[test]
fn create_object_from_map_fills_from_registers() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();

        // map with two writable value slots (offsets 0 and 1)
        let x = thread.intern(&scope, "x");
        let y = thread.intern(&scope, "y");
        let map_map = thread.heap().known().map_map.as_tagged();
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                map_map,
                kind: MapKind::OBJECT,
                value_slot_count: 2,
                descriptors: &[
                    (
                        SlotName::from(x.as_tagged()),
                        SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                        Smi::new(0).encode(),
                    ),
                    (
                        SlotName::from(y.as_tagged()),
                        SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                        Smi::new(1).encode(),
                    ),
                ],
            },
            &scope,
        );

        // constants[0] = the map
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[map.as_tagged().erase()], &scope);

        // r0 = 7, r1 = 9; CreateObjectFromMap 0 (map constant) r0 2
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadSmi, &[7]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[9]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::CreateObjectFromMap, &[0, 0, 2]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: bytecode.as_tagged(),
                constants: consts.as_tagged(),
                register_count: 2,
                context: void,
            },
            &scope,
        );
        thread.run(callable.as_tagged(), &[])
    });

    let obj = result.unwrap();
    thread.heap().no_gc(|_nogc, _| {
        let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
        // Safety: `obj` is a strong, live reference to the object
        // literal returned by `run`, and no collection can happen
        // inside the no-GC scope.
        let o = unsafe { ptr.cast::<SlotsObject>().as_ref() };
        assert_eq!(Smi::decode(o.slot(0).inner()).unwrap().value(), 7);
        assert_eq!(Smi::decode(o.slot(1).inner()).unwrap().value(), 9);
    });
}
