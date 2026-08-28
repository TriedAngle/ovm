use bytecode::{Opcode, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{NativeIndex, Thread, VM, VmError};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, Handle, HandleScope, HeapPtr,
    LocalHeap, Map, MapInit, MapKind, Object, ObjectSlotsInit, SlotFlags, SlotName, Smi, Tagged,
    Value,
};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

/// Wrap a callable info in a normal object with a callable map
/// (kind convention: CALLABLE flag => slots[0] is the callable info).
fn callable_object<'s>(
    thread: &mut Thread<DummyHeap>,
    scope: &'s HandleScope<'_>,
    info: Handle<'_, CallableInfoObject>,
) -> Handle<'s, Object> {
    let void = thread.heap().known().void.value();
    let map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT.union(MapKind::CALLABLE),
            value_slot_count: 1,
            descriptors: &[],
        },
        scope,
    );
    thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[info.as_tagged().erase()],
                elements: void,
                length: 0,
            },
        )
        .into_handle(scope)
}

fn run_program(
    thread: &mut Thread<DummyHeap>,
    program: Vec<u8>,
    register_count: usize,
    args: &[Value],
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count,
                context: void,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.run(callable, args)
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
fn call_resolves_callable_object_and_pushes_frames() {
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
                bytecode: callee_bytecode,
                constants: callee_constants,
                register_count: 2,
                context: void,
            },
            &scope,
        );
        let callee_obj = callable_object(thread, &scope, callee);

        // caller: r0 = callee object; CallNoFeedback r0, r0, 1 -> acc
        let receiver_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[callee_obj.as_tagged().erase()], &scope);
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
                bytecode,
                constants: receiver_consts,
                register_count: 2,
                context: void,
            },
            &scope,
        );
        let caller_obj = callable_object(thread, &scope, caller);

        thread.run(caller_obj, &[])
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
        let a = array
            .get_as::<FixedArray>(nogc, heap.known().array_map)
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

    let (result, map_v) = thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();

        // map with two writable value slots (offsets 0 and 1)
        let x = thread.intern(&scope, "x");
        let y = thread.intern(&scope, "y");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
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
                bytecode,
                constants: consts,
                register_count: 2,
                context: void,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        (thread.run(callable, &[]), map.as_tagged().erase())
    });

    let obj = result.unwrap();
    thread.heap().no_gc(|_nogc, _| {
        let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
        // Safety: `obj` is a strong, live reference to the object
        // literal returned by `run`, and no collection can happen
        // inside the no-GC scope.
        let o = unsafe { ptr.cast::<Object>().as_ref() };
        let slots = unsafe { o.slots.get().as_ptr().unwrap().as_ref() };
        assert_eq!(Smi::decode(slots.at(0)).unwrap().value(), 7);
        assert_eq!(Smi::decode(slots.at(1)).unwrap().value(), 9);
        // the object's map must be the map from the constants table
        assert_eq!(o.header.map.get().erase(), map_v);
    });
}

#[test]
fn keyed_load_reads_array_element() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0..r2 = 10, 20, 30; r3 = [r0, r1, r2]; acc = 1; acc = r3[acc]
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[10]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[20]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[30]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::CreateArrayLiteral, &[0, 3]);
    emit(&mut program, Opcode::Store, &[3]);
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::LoadKeyedProperty, &[3, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 4, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 20);
}

#[test]
fn keyed_store_writes_array_element() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = [1]; r4 = 0 (key); acc = 99 (value); r3[r4] = acc; acc = r3[0]
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::CreateArrayLiteral, &[0, 1]);
    emit(&mut program, Opcode::Store, &[3]);
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[4]);
    emit(&mut program, Opcode::LoadSmi, &[99]);
    emit(&mut program, Opcode::StoreKeyedProperty, &[3, 4, 0]);
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::LoadKeyedProperty, &[3, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 5, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 99);
}

#[test]
fn keyed_load_out_of_bounds_errors() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    for key in [2u32, (-1i32) as u32] {
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadSmi, &[1]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::CreateArrayLiteral, &[0, 1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::LoadSmi, &[key]);
        emit(&mut program, Opcode::LoadKeyedProperty, &[1, 0]);
        emit(&mut program, Opcode::Return, &[]);

        let result = run_program(&mut thread, program, 2, &[]);
        assert_eq!(result, Err(VmError::OutOfBounds));
    }
}

/// Build an object with map (x -> slot 0, y -> slot 1) in the constants table
/// at index 0, and the interned name "x" at index 1.
fn object_program(
    thread: &mut Thread<DummyHeap>,
    build: impl FnOnce(&mut Vec<u8>),
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();
        let x = thread.intern(&scope, "x");
        let y = thread.intern(&scope, "y");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
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
                        Smi::new(1).encode()),
                ],
            },
            &scope,
        );
        let consts = thread.heap().allocate_handle::<FixedArray>(
            &[map.as_tagged().erase(), x.as_tagged().erase()],
            &scope,
        );

        // r0 = 7, r1 = 9; r2 = object
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadSmi, &[7]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[9]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::CreateObjectFromMap, &[0, 0, 2]);
        emit(&mut program, Opcode::Store, &[2]);
        build(&mut program);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 4,
                context: void,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.run(callable, &[])
    })
}

#[test]
fn keyed_load_reads_named_property_via_string_key() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = "x" (constants[1]); acc = r2[acc]
    let result = object_program(&mut thread, |program| {
        emit(program, Opcode::LoadConstant, &[1]);
        emit(program, Opcode::LoadKeyedProperty, &[2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn keyed_store_writes_named_property_via_string_key() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = "x"; acc = 42; r2[r3] = acc; acc = r2.x
    let result = object_program(&mut thread, |program| {
        emit(program, Opcode::LoadConstant, &[1]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreKeyedProperty, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

/// Build an object (x = 7) in r2 with map (x -> slot 0, attributes `x_flags`)
/// in constants at 0, interned "x" at 1, "z" at 2 and "w" at 3.
fn transition_object_program(
    thread: &mut Thread<DummyHeap>,
    kind: MapKind,
    x_flags: SlotFlags,
    build: impl FnOnce(&mut Vec<u8>),
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();
        let x = thread.intern(&scope, "x");
        let z = thread.intern(&scope, "z");
        let w = thread.intern(&scope, "w");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind,
                value_slot_count: 1,
                descriptors: &[(
                    SlotName::from(x.as_tagged()),
                    x_flags,
                    Smi::new(0).encode(),
                )],
            },
            &scope,
        );
        let consts = thread.heap().allocate_handle::<FixedArray>(
            &[
                map.as_tagged().erase(),
                x.as_tagged().erase(),
                z.as_tagged().erase(),
                w.as_tagged().erase(),
            ],
            &scope,
        );

        // r0 = 7; r2 = object
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadSmi, &[7]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::CreateObjectFromMap, &[0, 0, 1]);
        emit(&mut program, Opcode::Store, &[2]);
        build(&mut program);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 5,
                context: void,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.run(callable, &[])
    })
}

const EXTENDABLE: MapKind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
const WRITABLE_VALUE: SlotFlags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);

#[test]
fn named_store_new_property_transitions() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r2.z = 42 (transition); acc = r2.x + r2.z
    let result = transition_object_program(&mut thread, EXTENDABLE, WRITABLE_VALUE, |program| {
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreNamedProperty, &[2, 2, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::Add, &[3, 4]);
    });
    // existing slot preserved (7) and new slot written (42)
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 49);
}

#[test]
fn named_store_chained_transitions() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r2.z = 42 (transition); r2.w = 1 (chained transition);
    // acc = r2.x + r2.z + r2.w
    let result = transition_object_program(&mut thread, EXTENDABLE, WRITABLE_VALUE, |program| {
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreNamedProperty, &[2, 2, 0]);
        emit(program, Opcode::LoadSmi, &[1]);
        emit(program, Opcode::StoreNamedProperty, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::Add, &[3, 4]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadNamedProperty, &[2, 3, 0]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::Add, &[3, 4]);
    });
    // x = 7 (preserved), z = 42, w = 1
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 50);
}

#[test]
fn keyed_store_new_property_via_string_key_transitions() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = "z"; acc = 42; r2[r3] = acc (transition); acc = r2.z
    let result = transition_object_program(&mut thread, EXTENDABLE, WRITABLE_VALUE, |program| {
        emit(program, Opcode::LoadConstant, &[2]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreKeyedProperty, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn named_store_new_property_to_non_extensible_fails() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // plain OBJECT map: not extendable
    let result = transition_object_program(
        &mut thread,
        MapKind::OBJECT,
        WRITABLE_VALUE,
        |program| {
            emit(program, Opcode::LoadSmi, &[42]);
            emit(program, Opcode::StoreNamedProperty, &[2, 2, 0]);
        },
    );
    assert_eq!(result, Err(VmError::NotExtensible));
}

#[test]
fn named_store_to_non_writable_fails() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // x is a non-writable value slot
    let result =
        transition_object_program(&mut thread, EXTENDABLE, SlotFlags::VALUE, |program| {
            emit(program, Opcode::LoadSmi, &[42]);
            emit(program, Opcode::StoreNamedProperty, &[2, 1, 0]);
        });
    assert_eq!(result, Err(VmError::Type));
}

/// Parent object (p = 1) in constants at 2, child object in r2 with a parent
/// descriptor pointing at it; interned "p" at 1.
fn parent_object_program(
    thread: &mut Thread<DummyHeap>,
    store_op: Opcode,
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void.value();
        let p = thread.intern(&scope, "p");
        let parent_map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT,
                value_slot_count: 1,
                descriptors: &[(
                    SlotName::from(p.as_tagged()),
                    WRITABLE_VALUE,
                    Smi::new(0).encode(),
                )],
            },
            &scope,
        );
        let parent = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map: parent_map,
                    values: &[Smi::new(1).encode()],
                    elements: void,
                    length: 0,
                },
            )
            .into_handle(&scope);
        // child: no own slots, parent stored in the map
        let child_map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: EXTENDABLE,
                value_slot_count: 0,
                descriptors: &[(
                    SlotName::from(Tagged::smi(999).unwrap()),
                    SlotFlags::CONST.union(SlotFlags::PARENT),
                    parent.as_tagged().erase(),
                )],
            },
            &scope,
        );
        let consts = thread.heap().allocate_handle::<FixedArray>(
            &[
                child_map.as_tagged().erase(),
                p.as_tagged().erase(),
                parent.as_tagged().erase(),
            ],
            &scope,
        );

        // r2 = child; r2.p = 2 (via `store_op`); acc = r2.p + parent.p
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateObjectFromMap, &[0, 0, 0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::LoadSmi, &[2]);
        emit(&mut program, store_op, &[2, 1, 0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[2, 1, 0]);
        emit(&mut program, Opcode::Store, &[3]);
        emit(&mut program, Opcode::LoadConstant, &[2]);
        emit(&mut program, Opcode::Store, &[4]);
        emit(&mut program, Opcode::LoadNamedProperty, &[4, 1, 0]);
        emit(&mut program, Opcode::Store, &[5]);
        emit(&mut program, Opcode::Add, &[3, 5]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 6,
                context: void,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.run(callable, &[])
    })
}

#[test]
fn self_store_writes_through_to_parent_slot() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedProperty);
    // child.p = 2 (inherited, parent now 2) + parent.p = 2
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 4);
}

#[test]
fn shadow_store_creates_own_slot_and_leaves_parent() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedPropertyShadow);
    // child.p = 2 (new own slot) + parent.p = 1 (untouched)
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 3);
}
