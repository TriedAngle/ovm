use bytecode::{Opcode, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{NativeContext, NativeIndex, Thread, VM, VmError};
use vm::{
    AccessorPair, CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, Float, Handle,
    HandleScope, HeapPtr, Lookup, Map, MapInit, MapKind, Object, ObjectSlotsInit, SlotFlags,
    SlotName, Smi, Tagged, Value, ValueRef,
};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

/// Assert a run escaped uncaught: the sentinel is returned and the pending
/// exception is a materialized error object of the given class name.
fn expect_escaped(thread: &mut Thread, result: Result<Value, VmError>, class: &str) -> Value {
    assert_eq!(
        result,
        Ok(thread.heap().known().exception.value()),
        "run must escape uncaught"
    );
    let ex = thread.take_pending_exception().expect("pending exception");
    assert!(!thread.has_pending_exception(), "pending cleared on take");
    let expected_name = thread.handle_scope(|thread, scope| thread.intern(&scope, class).value());
    thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name").value();
        thread.heap().no_gc(|nogc, heap| {
            let ValueRef::Object(o) = ex.value_ref(nogc) else {
                panic!("pending exception must be an object");
            };
            match o
                .as_ref()
                .lookup(nogc, heap, SlotName::from_value(name_key))
            {
                Lookup::Data { slot, .. } => {
                    assert_eq!(slot.inner(), expected_name, "error class name");
                }
                _ => panic!("error object must have a name property"),
            }
        });
    });
    ex
}

/// Wrap a callable info in a normal object with a callable map
/// (kind convention: CALLABLE flag => slots[0] is the callable info).
fn callable_object<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    info: Handle<'_, CallableInfoObject>,
) -> Handle<'s, Object> {
    let void = thread.heap().known().void;
    let empty_context = thread.heap().known().empty_context;
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
                elements: void.erase(),
                length: 0,
            },
        )
        .into_handle(scope)
}

fn run_program(
    thread: &mut Thread,
    program: Vec<u8>,
    register_count: usize,
    args: &[Value],
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count,
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, args)
    })
}

#[test]
fn load_smi_signed_immediates() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // Add on uninitialized (void) registers throws a TypeError, aborting
    // the run with a frame still on the suspended list.
    let mut bad = Vec::new();
    emit(&mut bad, Opcode::Add, &[0, 1]);
    emit(&mut bad, Opcode::Return, &[]);
    let result = run_program(&mut thread, bad, 2, &[]);
    expect_escaped(&mut thread, result, "TypeError");

    // The next run on the same thread must start from a clean slate.
    let mut good = Vec::new();
    emit(&mut good, Opcode::LoadSmi, &[41]);
    emit(&mut good, Opcode::Return, &[]);
    let result = run_program(&mut thread, good, 1, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 41);
}

#[test]
fn parameters_are_readable_via_negative_registers() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]); // param 0 (receiver)
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 2, &[smi(42)]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn wide_parameter_operand_uses_two_bytes() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;

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
                context: empty_context,
                handlers: None,
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
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let caller_obj = callable_object(thread, &scope, caller);

        thread.execute(caller_obj, &[])
    });

    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 99);
}

#[test]
fn create_array_literal_fills_from_registers() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (result, map_v) = thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;

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
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        (thread.execute(callable, &[]), map.as_tagged().erase())
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
        expect_escaped(&mut thread, result, "RangeError");
    }
}

/// Build an object with map (x -> slot 0, y -> slot 1) in the constants table
/// at index 0, and the interned name "x" at index 1.
fn object_program(thread: &mut Thread, build: impl FnOnce(&mut Vec<u8>)) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
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
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    })
}

#[test]
fn keyed_load_reads_named_property_via_string_key() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    thread: &mut Thread,
    kind: MapKind,
    x_flags: SlotFlags,
    build: impl FnOnce(&mut Vec<u8>),
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
        let x = thread.intern(&scope, "x");
        let z = thread.intern(&scope, "z");
        let w = thread.intern(&scope, "w");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind,
                value_slot_count: 1,
                descriptors: &[(SlotName::from(x.as_tagged()), x_flags, Smi::new(0).encode())],
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
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    })
}

const EXTENDABLE: MapKind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
const WRITABLE_VALUE: SlotFlags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);

#[test]
fn named_store_new_property_transitions() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // plain OBJECT map: not extendable
    let result =
        transition_object_program(&mut thread, MapKind::OBJECT, WRITABLE_VALUE, |program| {
            emit(program, Opcode::LoadSmi, &[42]);
            emit(program, Opcode::StoreNamedProperty, &[2, 2, 0]);
        });
    expect_escaped(&mut thread, result, "TypeError");
}

#[test]
fn named_store_to_non_writable_fails() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // x is a non-writable value slot
    let result = transition_object_program(&mut thread, EXTENDABLE, SlotFlags::VALUE, |program| {
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreNamedProperty, &[2, 1, 0]);
    });
    expect_escaped(&mut thread, result, "TypeError");
}

/// Parent object (p = 1) in constants at 2, child object in r2 with a parent
/// descriptor pointing at it; interned "p" at 1.
fn parent_object_program(thread: &mut Thread, store_op: Opcode) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
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
                    elements: void.erase(),
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
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    })
}

#[test]
fn self_store_writes_through_to_parent_slot() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedProperty);
    // child.p = 2 (inherited, parent now 2) + parent.p = 2
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 4);
}

#[test]
fn shadow_store_creates_own_slot_and_leaves_parent() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedPropertyShadow);
    // child.p = 2 (new own slot) + parent.p = 1 (untouched)
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 3);
}

#[test]
fn fallthrough_return_is_undefined() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 0, &[]).unwrap();
    assert_eq!(result, thread.heap().known().undefined.value());
}

#[test]
fn jump_skips_instructions() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // LoadSmi 1 (0..2); Jump ->6 (2..4); LoadSmi 2 (4..6); Return (6..7)
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::Jump, &[4]); // offset is relative to the jump's own pc
    emit(&mut program, Opcode::LoadSmi, &[2]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 0, &[]).unwrap();
    assert_eq!(Smi::decode(result).unwrap().value(), 1);
}

#[test]
fn jump_loop_counts_down_to_zero() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = 3; r1 = -1;
    // loop: if falsy(r0) goto end; r0 = r0 + r1; goto loop;
    // end: return r0
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[3]); // 0..2
    emit(&mut program, Opcode::Store, &[0]); // 2..4
    emit(&mut program, Opcode::LoadSmi, &[(-1i32) as u32]); // 4..6
    emit(&mut program, Opcode::Store, &[1]); // 6..8
    // loop @ 8
    emit(&mut program, Opcode::Load, &[0]); // 8..10
    emit(&mut program, Opcode::JumpIfFalsy, &[9]); // 10..12 -> 19
    emit(&mut program, Opcode::Add, &[0, 1]); // 12..15
    emit(&mut program, Opcode::Store, &[0]); // 15..17
    emit(&mut program, Opcode::JumpLoop, &[(-9i32) as u32]); // 17..19 -> 8
    // end @ 19
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 2, &[]).unwrap();
    assert_eq!(Smi::decode(result).unwrap().value(), 0);
}

#[test]
fn jump_if_truthy_follows_toboolean() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = param0; JumpIfTruthy L; LoadSmi 0; Return; L: LoadSmi 1; Return
    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]); // 0..2
    emit(&mut program, Opcode::JumpIfTruthy, &[5]); // 2..4 -> 7
    emit(&mut program, Opcode::LoadSmi, &[0]); // 4..6
    emit(&mut program, Opcode::Return, &[]); // 6..7
    emit(&mut program, Opcode::LoadSmi, &[1]); // 7..9
    emit(&mut program, Opcode::Return, &[]); // 9..10

    thread.handle_scope(|thread, scope| {
        let (undefined, null, true_v, false_v, hole, empty) = {
            let k = thread.heap().known();
            (
                k.undefined.value(),
                k.null.value(),
                k.true_object.value(),
                k.false_object.value(),
                k.void.value(),
                k.empty_string.value(),
            )
        };
        let hello = thread.intern(&scope, "hello").value();
        let zero = thread.heap().allocate_handle::<Float>(0.0, &scope).value();
        let neg_zero = thread.heap().allocate_handle::<Float>(-0.0, &scope).value();
        let nan = thread
            .heap()
            .allocate_handle::<Float>(f64::NAN, &scope)
            .value();
        let one_half = thread.heap().allocate_handle::<Float>(1.5, &scope).value();
        let object = {
            let void = thread.heap().known().void;
            let empty_context = thread.heap().known().empty_context;
            let map = thread.heap().allocate_handle::<Map>(
                MapInit {
                    kind: MapKind::OBJECT,
                    value_slot_count: 0,
                    descriptors: &[],
                },
                &scope,
            );
            thread
                .heap()
                .allocate_object(
                    &scope,
                    ObjectSlotsInit {
                        map,
                        values: &[],
                        elements: void.erase(),
                        length: 0,
                    },
                )
                .into_handle(&scope)
                .value()
        };

        let falsey = [
            smi(0),
            undefined,
            null,
            false_v,
            hole,
            empty,
            zero,
            neg_zero,
            nan,
        ];
        let truthy = [smi(1), smi(-1), true_v, hello, one_half, object];

        for v in falsey {
            let result = run_program(thread, program.clone(), 0, &[v]).unwrap();
            assert_eq!(
                Smi::decode(result).unwrap().value(),
                0,
                "{v:?} must be falsey"
            );
        }
        for v in truthy {
            let result = run_program(thread, program.clone(), 0, &[v]).unwrap();
            assert_eq!(
                Smi::decode(result).unwrap().value(),
                1,
                "{v:?} must be truthy"
            );
        }
    });
}

#[test]
fn test_reference_equal_compares_identity() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = param0; TestReferenceEqual param1; Return
    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut program, Opcode::TestReferenceEqual, &[(-2i32) as u32]);
    emit(&mut program, Opcode::Return, &[]);

    let (true_v, false_v, undefined, null) = {
        let k = thread.heap().known();
        (
            k.true_object.value(),
            k.false_object.value(),
            k.undefined.value(),
            k.null.value(),
        )
    };

    for (a, b, expected) in [
        (undefined, undefined, true_v),
        (undefined, null, false_v),
        (smi(1), smi(1), true_v),
        (smi(1), smi(2), false_v),
        (true_v, true_v, true_v),
        (true_v, false_v, false_v),
    ] {
        let result = run_program(&mut thread, program.clone(), 0, &[a, b]).unwrap();
        assert_eq!(result, expected);
    }
}

/// getter: `return this.y` (receiver is param 0, "y" is constants[0])
fn getter_program() -> Vec<u8> {
    let mut p = Vec::new();
    emit(&mut p, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut p, Opcode::Store, &[0]);
    emit(&mut p, Opcode::LoadNamedProperty, &[0, 0, 0]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

/// setter: `this.y = value` (receiver is param 0, value is param 1)
fn setter_program() -> Vec<u8> {
    let mut p = Vec::new();
    emit(&mut p, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut p, Opcode::Store, &[0]);
    emit(&mut p, Opcode::Load, &[(-2i32) as u32]);
    emit(&mut p, Opcode::StoreNamedProperty, &[0, 0, 0]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

/// Build an object (y = 7 in slot 0, accessor `x` backed by the given
/// getter/setter) in r2. Map in constants at 0, interned "x" at 1, "y" at 2,
/// "z" at 3.
fn accessor_object_program(
    thread: &mut Thread,
    getter: Option<&[u8]>,
    setter: Option<&[u8]>,
    build: impl FnOnce(&mut Vec<u8>),
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
        let x = thread.intern(&scope, "x");
        let y = thread.intern(&scope, "y");
        let z = thread.intern(&scope, "z");

        // getter/setter get constants ["y"] so they can reach the backing slot
        let make = |thread: &mut Thread, program: &[u8]| -> Value {
            let bytecode = thread
                .heap()
                .allocate_handle::<FixedByteArray>(program, &scope);
            let constants = thread
                .heap()
                .allocate_handle::<FixedArray>(&[y.as_tagged().erase()], &scope);
            let info = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants,
                    register_count: 1,
                    context: empty_context,
                    handlers: None,
                },
                &scope,
            );
            callable_object(thread, &scope, info).value()
        };
        let get = getter.map_or(void.value(), |p| make(&mut *thread, p));
        let set = setter.map_or(void.value(), |p| make(&mut *thread, p));
        let pair = thread
            .heap()
            .allocate_handle::<AccessorPair>((get, set), &scope);

        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT,
                value_slot_count: 1,
                descriptors: &[
                    (
                        SlotName::from(y.as_tagged()),
                        WRITABLE_VALUE,
                        Smi::new(0).encode(),
                    ),
                    (
                        SlotName::from(x.as_tagged()),
                        SlotFlags::ACCESSOR,
                        pair.value(),
                    ),
                ],
            },
            &scope,
        );
        let consts = thread.heap().allocate_handle::<FixedArray>(
            &[
                map.as_tagged().erase(),
                x.as_tagged().erase(),
                y.as_tagged().erase(),
                z.as_tagged().erase(),
            ],
            &scope,
        );

        // r0 = 7 (initial y); r2 = object
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
                register_count: 4,
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    })
}

#[test]
fn named_load_calls_getter_with_receiver() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = r2.x (calls the getter, which reads this.y)
    let result = accessor_object_program(&mut thread, Some(&getter_program()), None, |program| {
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn named_store_calls_setter_with_receiver_and_value() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r2.x = 21 (calls the setter, which writes this.y); acc = r2.y
    let result = accessor_object_program(&mut thread, None, Some(&setter_program()), |program| {
        emit(program, Opcode::LoadSmi, &[21]);
        emit(program, Opcode::StoreNamedProperty, &[2, 1, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 21);
}

#[test]
fn named_load_without_getter_is_undefined() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let undefined = thread.heap().known().undefined.value();
    let result = accessor_object_program(&mut thread, None, None, |program| {
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
    });
    assert_eq!(result.unwrap(), undefined);
}

#[test]
fn named_store_without_setter_is_ignored() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r2.x = 21 is ignored (no setter); y keeps its initial value 7
    let result = accessor_object_program(&mut thread, None, None, |program| {
        emit(program, Opcode::LoadSmi, &[21]);
        emit(program, Opcode::StoreNamedProperty, &[2, 1, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn keyed_load_calls_getter() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = "x"; acc = r2[acc] (calls the getter)
    let result = accessor_object_program(&mut thread, Some(&getter_program()), None, |program| {
        emit(program, Opcode::LoadConstant, &[1]);
        emit(program, Opcode::LoadKeyedProperty, &[2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn keyed_store_calls_setter() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = "x"; r2[r3] = 21 (calls the setter); acc = r2.y
    let result = accessor_object_program(&mut thread, None, Some(&setter_program()), |program| {
        emit(program, Opcode::LoadConstant, &[1]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[21]);
        emit(program, Opcode::StoreKeyedProperty, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 21);
}

#[test]
fn named_load_missing_property_is_undefined() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let undefined = thread.heap().known().undefined.value();
    // r2.z does not exist on the map
    let result = accessor_object_program(&mut thread, None, None, |program| {
        emit(program, Opcode::LoadNamedProperty, &[2, 3, 0]);
    });
    assert_eq!(result.unwrap(), undefined);
}

#[test]
fn store_new_accessor_property_defines_own_accessor() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
        let x = thread.intern(&scope, "x");
        let y = thread.intern(&scope, "y");

        // object { y: 7 } on an extendable map
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: EXTENDABLE,
                value_slot_count: 1,
                descriptors: &[(
                    SlotName::from(y.as_tagged()),
                    WRITABLE_VALUE,
                    Smi::new(0).encode(),
                )],
            },
            &scope,
        );
        let obj = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[smi(7)],
                    elements: void.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);

        // define `x` as an accessor with a getter (this.y) and no setter
        let getter = {
            let bytecode = thread
                .heap()
                .allocate_handle::<FixedByteArray>(&getter_program(), &scope);
            let constants = thread
                .heap()
                .allocate_handle::<FixedArray>(&[y.as_tagged().erase()], &scope);
            let info = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants,
                    register_count: 1,
                    context: empty_context,
                    handlers: None,
                },
                &scope,
            );
            callable_object(thread, &scope, info)
        };
        let name = scope
            .create_handle(SlotName::from(x.as_tagged()).tagged())
            .expect("name is strong");
        let get = scope
            .create_handle(Tagged::from_value(getter.value()))
            .expect("getter is strong");
        let set = scope
            .create_handle(Tagged::from_value(void.value()))
            .expect("void is strong");
        Object::store_new_accessor_property(thread.heap(), obj, name, get, set).unwrap();

        // program: acc = param0.x
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[x.as_tagged().erase()], &scope);
        let mut program = Vec::new();
        emit(&mut program, Opcode::Load, &[(-1i32) as u32]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 1,
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[obj.value()])
    });

    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

/// A native function object: callable map with NATIVE flag, slots[0] = the
/// native registry index as a Smi.
fn native_function<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    idx: NativeIndex,
) -> Handle<'s, Object> {
    let void = thread.heap().known().void;
    let empty_context = thread.heap().known().empty_context;
    let map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT
                .union(MapKind::CALLABLE)
                .union(MapKind::NATIVE),
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
                values: &[Smi::new(idx.0 as i64).encode()],
                elements: void.erase(),
                length: 0,
            },
        )
        .into_handle(scope)
}

/// Build a bytecode function object from inside a native.
fn bytecode_fn(
    nctx: &mut NativeContext<'_>,
    scope: &HandleScope<'_>,
    program: &[u8],
    constants: &[Value],
    register_count: usize,
) -> Value {
    let void = nctx.heap().known().void;
    let empty_context = nctx.heap().known().empty_context;
    let bytecode = nctx
        .heap()
        .allocate_handle::<FixedByteArray>(program, scope);
    let constants = nctx.heap().allocate_handle::<FixedArray>(constants, scope);
    let info = nctx.heap().allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            bytecode,
            constants,
            register_count,
            context: empty_context,
            handlers: None,
        },
        scope,
    );
    let map = nctx.heap().allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT.union(MapKind::CALLABLE),
            value_slot_count: 1,
            descriptors: &[],
        },
        scope,
    );
    nctx.heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[info.value()],
                elements: void.erase(),
                length: 0,
            },
        )
        .erase()
}

fn forty_two(_: &mut NativeContext<'_>, _: &[Value]) -> Result<Value, VmError> {
    Ok(smi(42))
}

#[test]
fn run_dispatches_native_callable_without_frame() {
    let mut vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(forty_two);
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let f = native_function(thread, &scope, idx);
        thread.execute(f, &[smi(7)])
    });
    assert_eq!(result.unwrap(), smi(42));
}

#[test]
fn call_dispatches_to_native_function_object() {
    let mut vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(forty_two);
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let f = native_function(thread, &scope, idx);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[f.value()], &scope);

        // r0 = native fn; Call r0 with r0 as the (single, receiver) arg
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::CallNoFeedback, &[0, 0, 1]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let void = thread.heap().known().void;
        let empty_context = thread.heap().known().empty_context;
        let caller = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 1,
                context: empty_context,
                handlers: None,
            },
            &scope,
        );
        let caller = callable_object(thread, &scope, caller);
        thread.execute(caller, &[])
    });

    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

/// Native that runs `6 + 7` in a fresh nested interpreter execution, where
/// the inner program itself spills the accumulator for a CallNative.
fn run_inner(nctx: &mut NativeContext<'_>, _args: &[Value]) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
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

        let callable = bytecode_fn(nctx, &scope, &program, &[], 3);
        nctx.call(callable, &[])
    })
}

#[test]
fn native_reenters_interpreter_via_call() {
    let mut vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(run_inner);
    let mut thread = vm.attach();

    // acc = 5 (spilled across the native call); acc = run_inner(); r1 = acc
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[5]);
    emit(&mut program, Opcode::CallNative, &[idx.0 as u32, 0, 1]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::Load, &[1]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 2, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 13);
}

/// Native that runs bytecode which throws one call deep; the suspended inner
/// frames are abandoned and must be unwound when the native recovers.
fn run_failing_inner(nctx: &mut NativeContext<'_>, _args: &[Value]) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        // callee: Add on uninitialized (void) registers -> TypeError throw
        let mut bad = Vec::new();
        emit(&mut bad, Opcode::Add, &[0, 1]);
        emit(&mut bad, Opcode::Return, &[]);
        let callee = bytecode_fn(nctx, &scope, &bad, &[], 2);

        // caller: calls callee, so one frame is suspended above the base
        // depth when the exception escapes the nested run
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::CallNoFeedback, &[0, 0, 1]);
        emit(&mut program, Opcode::Return, &[]);
        let caller = bytecode_fn(nctx, &scope, &program, &[callee], 1);

        match nctx.call(caller, &[]) {
            Ok(exc) if exc == nctx.heap().known().exception.value() => {
                // the exception escapes the nested run as the sentinel with
                // the pending exception set; the native recovers by
                // clearing it
                let ex = nctx.take_pending_exception().expect("pending exception");
                assert!(ex.is_strong_ptr());
                Ok(smi(42))
            }
            other => panic!("expected inner escape, got {other:?}"),
        }
    })
}

#[test]
fn inner_run_error_unwinds_and_native_recovers() {
    let mut vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(run_failing_inner);
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::CallNative, &[idx.0 as u32, 0, 1]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 1, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);

    // the thread is clean afterwards: no leaked frames or stack slots
    let mut good = Vec::new();
    emit(&mut good, Opcode::LoadSmi, &[41]);
    emit(&mut good, Opcode::Return, &[]);
    let result = run_program(&mut thread, good, 1, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 41);
}
