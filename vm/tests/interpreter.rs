use bytecode::{Opcode, PropertyFlags, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use vm::{
    AccessorPair, CallableInfoInit, CallableInfoObject, Context, ContextInit, FixedArray,
    FixedByteArray, Float, FunctionKind, GcSlice, Handle, HandleScope, HeapPtr, Lookup, Map,
    MapInit, MapKind, Object, ObjectSlotsInit, PropertyDescriptor, ScopeInfo, ScopeInfoInit,
    SlotFlags, SlotName, Smi, StoreOutcome, StoreSemantics, VMString, Value, string_content_hash,
};
use vm::{NativeContext, NativeIndex, Thread, VM, VmError};

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
        thread.heap().no_gc(|nogc| {
            let Some(o) = ex.as_heap_object(nogc) else {
                panic!("pending exception must be an object");
            };
            match o.as_ref().lookup(nogc, SlotName::from_value(name_key)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(slot.inner(), expected_name, "error class name");
                }
                _ => panic!("error object must have a name property"),
            }
        });
    });
    ex
}

/// Wrap a callable info in a function object with the well-known function map
/// (slots[0] = callable info, slots[1] = closure context).
fn callable_object<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    info: Handle<'_, CallableInfoObject>,
) -> Handle<'s, Object> {
    let the_hole = thread.heap().known().the_hole;
    let empty_context = thread.heap().known().empty_context;
    let map = thread.heap().known().function_map;
    thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[info.value(), empty_context.value()],
                elements: the_hole.erase(),
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
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let constants = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, args)
    })
}

fn create_closure_of_kind(
    thread: &mut Thread,
    body: &[u8],
    name: &str,
    formal_parameter_count: usize,
    kind: FunctionKind,
    strict: bool,
) -> Value {
    thread.handle_scope(|thread, scope| {
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(body, &scope);
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
        let name = thread.intern(&scope, name).value();
        thread.heap().no_gc(|nogc| {
            info.heap_ref(nogc).set_metadata(
                nogc,
                Some(name),
                formal_parameter_count,
                kind,
                strict,
            );
        });

        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateClosure, &[0]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 0, &[], &[info.value()]).unwrap()
    })
}

/// `run_program` with a `slot_count`-long context-names array at
/// constants[0] (the `CreateFunctionContext` operand; slot count comes
/// from the names array length).
fn run_program_ctx(
    thread: &mut Thread,
    program: Vec<u8>,
    register_count: usize,
    args: &[Value],
    slot_count: usize,
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let dummy: Vec<vm::Value> = (0..slot_count)
            .map(|i| thread.intern(&scope, format!("slot{i}")))
            .map(|h| h.value())
            .collect();
        let names = thread.heap().allocate_handle::<FixedArray>(&dummy, &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(&[scope_info.value()], &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count,
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

    // Add on an object operand needs ToPrimitive (not implemented yet) and
    // throws a TypeError, aborting the run with a frame still suspended.
    let obj = thread.handle_scope(|thread, scope| {
        let known = thread.heap().known();
        thread
            .heap()
            .new_object(&scope, known.object_initial_map, &[])
            .into_handle(&scope)
            .as_tagged()
            .erase()
    });
    let mut bad = Vec::new();
    emit(&mut bad, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut bad, Opcode::Store, &[1]);
    emit(&mut bad, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut bad, Opcode::Add, &[1]);
    emit(&mut bad, Opcode::Return, &[]);
    let result = run_program(&mut thread, bad, 2, &[obj]);
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
                handlers: None,
            },
            &scope,
        );
        let callee_obj = callable_object(thread, &scope, callee);

        // caller: r0 = callee object; CallNoFeedback r0, r0, 1 -> acc
        let receiver_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[callee_obj.value()], &scope);
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
fn array_literal_built_with_manual_stores() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = []; r3[0] = 1; r3[1] = 2; r3[2] = 3; return r3
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Store, &[3]);
    for (key, value) in [(0i32, 1i32), (1, 2), (2, 3)] {
        emit(&mut program, Opcode::LoadSmi, &[key as u32]);
        emit(&mut program, Opcode::Store, &[4]);
        emit(&mut program, Opcode::LoadSmi, &[value as u32]);
        emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[3, 4, 0]);
    }
    emit(&mut program, Opcode::Load, &[3]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 5, &[]);
    let array = result.unwrap();
    thread.heap().no_gc(|nogc| {
        let a = array.get_as::<Object>(nogc).expect("array literal result");
        let a = a.as_ref();
        assert!(a.is_array(nogc));
        assert_eq!(a.length(), 3);
        let elements = a.elements_array(nogc).expect("array elements");
        assert_eq!(Smi::decode(elements.at(0)).unwrap().value(), 1);
        assert_eq!(Smi::decode(elements.at(1)).unwrap().value(), 2);
        assert_eq!(Smi::decode(elements.at(2)).unwrap().value(), 3);
    });
}

#[test]
fn create_empty_array_literal_starts_empty() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 0, &[]);
    let array = result.unwrap();
    thread.heap().no_gc(|nogc| {
        let a = array.get_as::<Object>(nogc).expect("array literal result");
        let a = a.as_ref();
        assert!(a.is_array(nogc));
        assert_eq!(a.length(), 0);
        let elements = a.elements_array(nogc).expect("array elements");
        assert_eq!(elements.len(), 0);
    });
}

#[test]
fn array_literal_with_holes_keeps_length() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // [1, , 2]: never store index 1; storing index 2 grows length to 3
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Store, &[3]);
    for (key, value) in [(0i32, 1i32), (2, 2)] {
        emit(&mut program, Opcode::LoadSmi, &[key as u32]);
        emit(&mut program, Opcode::Store, &[4]);
        emit(&mut program, Opcode::LoadSmi, &[value as u32]);
        emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[3, 4, 0]);
    }
    emit(&mut program, Opcode::Load, &[3]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 5, &[]);
    let array = result.unwrap();
    thread.heap().no_gc(|nogc| {
        let a = array.get_as::<Object>(nogc).expect("array literal result");
        let a = a.as_ref();
        assert_eq!(a.length(), 3);
        let elements = a.elements_array(nogc).expect("array elements");
        assert_eq!(Smi::decode(elements.at(0)).unwrap().value(), 1);
        assert_eq!(
            elements.at(1),
            nogc.known().the_hole.value(),
            "elided index stays a hole"
        );
        assert_eq!(Smi::decode(elements.at(2)).unwrap().value(), 2);
    });
}

#[test]
fn object_literal_built_with_manual_stores() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let build = |thread: &mut Thread| -> Value {
        thread.handle_scope(|thread, scope| {
            let x = thread.intern(&scope, "x");
            let y = thread.intern(&scope, "y");
            let consts = thread
                .heap()
                .allocate_handle::<FixedArray>(&[x.value(), y.value()], &scope);

            // r0 = {}; r0.x = 7; r0.y = 9; return r0
            let mut program = Vec::new();
            emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
            emit(&mut program, Opcode::Store, &[0]);
            emit(&mut program, Opcode::LoadSmi, &[7]);
            emit(&mut program, Opcode::StoreNamedProperty, &[0, 0, 0]);
            emit(&mut program, Opcode::LoadSmi, &[9]);
            emit(&mut program, Opcode::StoreNamedProperty, &[0, 1, 0]);
            emit(&mut program, Opcode::Load, &[0]);
            emit(&mut program, Opcode::Return, &[]);

            let bytecode = thread
                .heap()
                .allocate_handle::<FixedByteArray>(&program, &scope);
            let callable = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants: consts,
                    register_count: 1,
                    handlers: None,
                },
                &scope,
            );
            let callable = callable_object(thread, &scope, callable);
            thread.execute(callable, &[]).unwrap()
        })
    };

    let obj1 = build(&mut thread);
    let obj2 = build(&mut thread);
    let ((x1, y1, map1), (x2, y2, map2), initial) = thread.heap().no_gc(|_nogc| {
        let read = |obj: Value| {
            let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
            // Safety: `obj` is a strong, live reference to the object
            // literal, and no collection can happen inside the no-GC scope.
            let o = unsafe { ptr.cast::<Object>().as_ref() };
            let slots = unsafe { o.slots.get().as_ptr().unwrap().as_ref() };
            (
                Smi::decode(slots.at(0)).unwrap().value(),
                Smi::decode(slots.at(1)).unwrap().value(),
                o.header.map.get().erase(),
            )
        };
        (
            read(obj1),
            read(obj2),
            _nogc.known().object_initial_map.value(),
        )
    });
    assert_eq!((x1, y1), (7, 9));
    assert_eq!((x2, y2), (7, 9));
    // stores transitioned off the initial map...
    assert_ne!(map1, initial);
    // ...and identically-built literals share one transition map
    assert_eq!(map1, map2);
}

// -- define-own-property (class member installation) ------------------------

#[test]
fn define_named_own_property_attributes_and_value() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; define m = 7 {writable, non-enum, configurable};
    // re-define m = 8 with the same attributes; return r0
    let build = |thread: &mut Thread| -> Value {
        thread.handle_scope(|thread, scope| {
            let m = thread.intern(&scope, "m");
            let consts = thread
                .heap()
                .allocate_handle::<FixedArray>(&[m.value()], &scope);
            let mut program = Vec::new();
            emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
            emit(&mut program, Opcode::Store, &[0]);
            emit(&mut program, Opcode::LoadSmi, &[7]);
            emit(
                &mut program,
                Opcode::DefineNamedOwnProperty,
                &[0, 0, PropertyFlags::DontEnum.bits(), 0],
            );
            emit(&mut program, Opcode::LoadSmi, &[8]);
            emit(
                &mut program,
                Opcode::DefineNamedOwnProperty,
                &[0, 0, PropertyFlags::DontEnum.bits(), 0],
            );
            emit(&mut program, Opcode::Load, &[0]);
            emit(&mut program, Opcode::Return, &[]);

            let bytecode = thread
                .heap()
                .allocate_handle::<FixedByteArray>(&program, &scope);
            let callable = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants: consts,
                    register_count: 1,
                    handlers: None,
                },
                &scope,
            );
            let callable = callable_object(thread, &scope, callable);
            thread.execute(callable, &[]).unwrap()
        })
    };

    let obj1 = build(&mut thread);
    let obj2 = build(&mut thread);
    thread.handle_scope(|thread, scope| {
        let m = thread.intern(&scope, "m").value();
        thread.heap().no_gc(|nogc| {
            let read = |obj: Value| {
                let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
                // Safety: `obj` is a strong, live reference and no collection
                // can happen inside the no-GC scope.
                let o = unsafe { ptr.cast::<Object>().as_ref() };
                match o.lookup(nogc, SlotName::from_value(m)) {
                    Lookup::Data { slot, flags, .. } => {
                        assert_eq!(slot.inner(), smi(8), "re-define updates the value");
                        assert_eq!(
                            flags,
                            SlotFlags::VALUE
                                .union(SlotFlags::WRITABLE)
                                .union(SlotFlags::CONFIGURABLE),
                            "method attributes {{w, e-, c}}"
                        );
                        o.header.map.get().erase()
                    }
                    _ => panic!("m must be a data property"),
                }
            };
            let map1 = read(obj1);
            let map2 = read(obj2);
            // identically-built defines share one transition map
            assert_eq!(map1, map2);
        });
    });
}

#[test]
fn define_named_own_property_conflicting_redefine_throws() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; define p = 1 {w-, e-, c-}; re-define p = 2 {w, e-, c}:
    // non-configurable with differing attributes rejects the define
    let p = thread.handle_scope(|thread, scope| thread.intern(&scope, "p").value());
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(
        &mut program,
        Opcode::DefineNamedOwnProperty,
        &[
            0,
            0,
            PropertyFlags::ReadOnly | PropertyFlags::DontEnum | PropertyFlags::DontDelete,
            0,
        ],
    );
    emit(&mut program, Opcode::LoadSmi, &[2]);
    emit(
        &mut program,
        Opcode::DefineNamedOwnProperty,
        &[0, 0, PropertyFlags::DontEnum.bits(), 0],
    );
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program_consts(&mut thread, program, 1, &[], &[p]);
    expect_escaped(&mut thread, result, "TypeError");
}

#[test]
fn define_keyed_own_property_string_and_smi_keys() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r1 = "x"; define r0[r1] = 5 {e-};
    // r1 = 3 (smi key); define r0[r1] = 6 {e-}; return r0
    let x = thread.handle_scope(|thread, scope| thread.intern(&scope, "x").value());
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadConstant, &[0]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[5]);
    emit(
        &mut program,
        Opcode::DefineKeyedOwnProperty,
        &[0, 1, PropertyFlags::DontEnum.bits(), 0],
    );
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[6]);
    emit(
        &mut program,
        Opcode::DefineKeyedOwnProperty,
        &[0, 1, PropertyFlags::DontEnum.bits(), 0],
    );
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let obj = run_program_consts(&mut thread, program, 2, &[], &[x]).unwrap();
    thread.heap().no_gc(|nogc| {
        let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
        // Safety: `obj` is a strong, live reference and no collection can
        // happen inside the no-GC scope.
        let o = unsafe { ptr.cast::<Object>().as_ref() };
        let expected_flags = SlotFlags::VALUE
            .union(SlotFlags::WRITABLE)
            .union(SlotFlags::CONFIGURABLE);
        match o.lookup(nogc, SlotName::from_value(x)) {
            Lookup::Data { slot, flags, .. } => {
                assert_eq!(slot.inner(), smi(5));
                assert_eq!(flags, expected_flags);
            }
            _ => panic!("x must be a data property"),
        }
        // a smi key defines a plain named property, not an element
        match o.lookup(nogc, SlotName::from_value(smi(3))) {
            Lookup::Data { slot, flags, .. } => {
                assert_eq!(slot.inner(), smi(6));
                assert_eq!(flags, expected_flags);
            }
            _ => panic!("the smi key must be a named data property"),
        }
    });
}

#[test]
fn define_own_property_accessor_invokes_getter() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // getter returns 42, setter returns undefined
    let (getter, p) = thread.handle_scope(|thread, scope| {
        let body = {
            let mut b = Vec::new();
            emit(&mut b, Opcode::LoadSmi, &[42]);
            emit(&mut b, Opcode::Return, &[]);
            b
        };
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&body, &scope);
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
        let getter = callable_object(thread, &scope, info).value();
        let p = thread.intern(&scope, "p").value();
        (getter, p)
    });

    // r0 = {}; r1 = getter; r2 = setter (undefined); acc = pair;
    // define r0.p {accessor, e-}; then either load r0.p (invokes the
    // getter) or return r0 to inspect the installed descriptor
    let accessor_program = |load: bool| -> Vec<u8> {
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::CreateAccessorPair, &[1, 2]);
        emit(
            &mut program,
            Opcode::DefineNamedOwnProperty,
            &[0, 2, PropertyFlags::Accessor | PropertyFlags::DontEnum, 0],
        );
        if load {
            emit(&mut program, Opcode::LoadNamedProperty, &[0, 2, 0]);
        } else {
            emit(&mut program, Opcode::Load, &[0]);
        }
        emit(&mut program, Opcode::Return, &[]);
        program
    };

    let undefined = thread.heap().known().undefined.value();
    let constants = [getter, undefined, p];
    let result = run_program_consts(&mut thread, accessor_program(true), 3, &[], &constants);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);

    let obj = run_program_consts(&mut thread, accessor_program(false), 3, &[], &constants).unwrap();
    thread.heap().no_gc(|nogc| {
        let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
        // Safety: `obj` is a strong, live reference and no collection can
        // happen inside the no-GC scope.
        let o = unsafe { ptr.cast::<Object>().as_ref() };
        match o.lookup(nogc, SlotName::from_value(p)) {
            Lookup::Accessor { pair, .. } => {
                assert_eq!(pair.get.inner(), getter);
            }
            _ => panic!("p must be an accessor property"),
        }
        let flags = o
            .map_ref(nogc)
            .descriptors()
            .iter()
            .find(|d| d.name() == SlotName::from_value(p))
            .map(|d| d.flags())
            .expect("p descriptor");
        assert_eq!(
            flags,
            SlotFlags::ACCESSOR.union(SlotFlags::CONFIGURABLE),
            "accessor attributes {{e-, c}}"
        );
    });
}

#[test]
fn keyed_load_reads_array_element() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = []; r3[0..2] = 10, 20, 30; acc = 1; acc = r3[acc]
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Store, &[3]);
    for (key, value) in [(0i32, 10i32), (1, 20), (2, 30)] {
        emit(&mut program, Opcode::LoadSmi, &[key as u32]);
        emit(&mut program, Opcode::Store, &[4]);
        emit(&mut program, Opcode::LoadSmi, &[value as u32]);
        emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[3, 4, 0]);
    }
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::LoadKeyedProperty, &[3, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 5, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 20);
}

#[test]
fn keyed_store_writes_array_element() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = []; r3[0] = 1; r4 = 0 (key); acc = 99; r3[r4] = acc; acc = r3[0]
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Store, &[3]);
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[4]);
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[3, 4, 0]);
    emit(&mut program, Opcode::LoadSmi, &[99]);
    emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[3, 4, 0]);
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::LoadKeyedProperty, &[3, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 5, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 99);
}

#[test]
fn keyed_load_out_of_bounds_yields_undefined() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // out-of-range and negative indices are ordinary property lookups and
    // yield undefined
    for key in [2u32, (-1i32) as u32] {
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::LoadSmi, &[0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::LoadSmi, &[1]);
        emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[1, 2, 0]);
        emit(&mut program, Opcode::LoadSmi, &[key]);
        emit(&mut program, Opcode::LoadKeyedProperty, &[1, 0]);
        emit(&mut program, Opcode::Return, &[]);

        let result = run_program(&mut thread, program, 3, &[]);
        assert_eq!(result.unwrap(), thread.heap().known().undefined.value());
    }
}

#[test]
fn keyed_store_grows_array_and_fills_holes() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r1 = []; r1[0] = 1; r1[3] = 42; acc = r1[3]; then acc = r1[1] (hole -> undefined)
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[1, 2, 0]);
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[1, 2, 0]);
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::LoadKeyedProperty, &[1, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);

    // the grown array must have length 4 with holes at 1..3
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[1, 2, 0]);
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::StoreKeyedPropertyNoShadow, &[1, 2, 0]);
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::LoadKeyedProperty, &[1, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    assert_eq!(result.unwrap(), thread.heap().known().undefined.value());
}

#[test]
fn keyed_store_creates_numeric_property_on_plain_object() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r3 = 0 (key); r2[0] = 42 (numeric property on an object); acc = r2[0]
    let result = transition_object_program(&mut thread, EXTENDABLE, WRITABLE_VALUE, |program| {
        emit(program, Opcode::LoadSmi, &[0]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreKeyedPropertyNoShadow, &[2, 3, 0]);
        emit(program, Opcode::LoadSmi, &[0]);
        emit(program, Opcode::LoadKeyedProperty, &[2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

/// Build `{x: 7, y: 9}` in r2 via manual stores; constants: "x" at 0, "y" at 1.
fn object_program(thread: &mut Thread, build: impl FnOnce(&mut Vec<u8>)) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let x = thread.intern(&scope, "x");
        let y = thread.intern(&scope, "y");
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[x.value(), y.value()], &scope);

        // r2 = {}; r2.x = 7; r2.y = 9
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::LoadSmi, &[7]);
        emit(&mut program, Opcode::StoreNamedProperty, &[2, 0, 0]);
        emit(&mut program, Opcode::LoadSmi, &[9]);
        emit(&mut program, Opcode::StoreNamedProperty, &[2, 1, 0]);
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

    // acc = "x" (constants[0]); acc = r2[acc]
    let result = object_program(&mut thread, |program| {
        emit(program, Opcode::LoadConstant, &[0]);
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
        emit(program, Opcode::LoadConstant, &[0]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreKeyedPropertyNoShadow, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 0, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

/// Build an object (x = 7) host-side in constants at 0, interned "x" at 1,
/// "z" at 2 and "w" at 3; the program loads it into r2.
fn transition_object_program(
    thread: &mut Thread,
    kind: MapKind,
    x_flags: SlotFlags,
    build: impl FnOnce(&mut Vec<u8>),
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let the_hole = thread.heap().known().the_hole;
        let x = thread.intern(&scope, "x");
        let z = thread.intern(&scope, "z");
        let w = thread.intern(&scope, "w");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind,
                value_slot_count: 1,
                descriptors: &[(SlotName::from(x.as_tagged()), x_flags, Smi::new(0).encode())],
                prototype: the_hole.erase(),
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
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[obj.value(), x.value(), z.value(), w.value()], &scope);

        // r2 = object
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
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
        emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 2, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::Add, &[3]);
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
        emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 2, 0]);
        emit(program, Opcode::LoadSmi, &[1]);
        emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::Add, &[3]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadNamedProperty, &[2, 3, 0]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::Add, &[3]);
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
        emit(program, Opcode::StoreKeyedPropertyNoShadow, &[2, 3, 0]);
        emit(program, Opcode::LoadNamedProperty, &[2, 2, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn named_store_new_property_to_non_extensible_is_ignored() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // plain OBJECT map: not extendable. Sloppy-mode [[Set]] on an absent
    // property silently does nothing ([[DefineOwnProperty]] returns false,
    // which the store ignores; strict mode would throw).
    let result =
        transition_object_program(&mut thread, MapKind::OBJECT, WRITABLE_VALUE, |program| {
            emit(program, Opcode::LoadSmi, &[42]);
            emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 2, 0]);
        });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn named_store_to_non_writable_fails() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // x is a non-writable value slot
    let result = transition_object_program(&mut thread, EXTENDABLE, SlotFlags::VALUE, |program| {
        emit(program, Opcode::LoadSmi, &[42]);
        emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 1, 0]);
    });
    expect_escaped(&mut thread, result, "TypeError");
}

/// Parent object (p = 1) in constants at 2, child object in constants at 0
/// whose prototype (a FixedArray) points at it; interned "p" at 1.
fn parent_object_program(thread: &mut Thread, store_op: Opcode) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let the_hole = thread.heap().known().the_hole;
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
                prototype: the_hole.erase(),
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
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        // child: no own slots, prototype = FixedArray([parent]) (multiple
        // parents in priority order; here a single one)
        let parents = thread
            .heap()
            .allocate_handle::<FixedArray>(&[parent.value()], &scope);
        let child_map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: EXTENDABLE,
                value_slot_count: 0,
                descriptors: &[],
                prototype: parents.erase(),
            },
            &scope,
        );
        let child = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map: child_map,
                    values: &[],
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[child.value(), p.value(), parent.value()], &scope);

        // r2 = child; r2.p = 2 (via `store_op`); acc = r2.p + parent.p
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::LoadSmi, &[2]);
        emit(&mut program, store_op, &[2, 1, 0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[2, 1, 0]);
        emit(&mut program, Opcode::Store, &[3]);
        emit(&mut program, Opcode::LoadConstant, &[2]);
        emit(&mut program, Opcode::Store, &[4]);
        emit(&mut program, Opcode::LoadNamedProperty, &[4, 1, 0]);
        emit(&mut program, Opcode::Store, &[5]);
        emit(&mut program, Opcode::Add, &[3]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 6,
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

    let result = parent_object_program(&mut thread, Opcode::StoreNamedPropertyNoShadow);
    // child.p = 2 (inherited, parent now 2) + parent.p = 2
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 4);
}

#[test]
fn shadow_store_creates_own_slot_and_leaves_parent() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedProperty);
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
    assert_eq!(result.to_i64().unwrap(), 1);
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
    emit(&mut program, Opcode::JumpIfFalsy, &[10]); // 10..12 -> 20
    emit(&mut program, Opcode::Load, &[0]); // 12..14
    emit(&mut program, Opcode::Add, &[1]); // 14..16
    emit(&mut program, Opcode::Store, &[0]); // 16..18
    emit(&mut program, Opcode::JumpLoop, &[(-10i32) as u32]); // 18..20 -> 8
    // end @ 20
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 2, &[]).unwrap();
    assert_eq!(result.to_i64().unwrap(), 0);
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
            let empty = thread.intern(&scope, "").value();
            (
                k.undefined.value(),
                k.null.value(),
                k.true_object.value(),
                k.false_object.value(),
                k.the_hole.value(),
                empty,
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
            let the_hole = thread.heap().known().the_hole;
            let map = thread.heap().allocate_handle::<Map>(
                MapInit {
                    kind: MapKind::OBJECT,
                    value_slot_count: 0,
                    descriptors: &[],
                    prototype: the_hole.erase(),
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
                        elements: the_hole.erase(),
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
            assert_eq!(result.to_i64().unwrap(), 0, "{v:?} must be falsey");
        }
        for v in truthy {
            let result = run_program(thread, program.clone(), 0, &[v]).unwrap();
            assert_eq!(result.to_i64().unwrap(), 1, "{v:?} must be truthy");
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
    emit(&mut p, Opcode::StoreNamedPropertyNoShadow, &[0, 0, 0]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

/// Build an object (y = 7 in slot 0, accessor `x` backed by the given
/// getter/setter) host-side in constants at 0; interned "x" at 1, "y" at 2,
/// "z" at 3. The program loads it into r2.
fn accessor_object_program(
    thread: &mut Thread,
    getter: Option<&[u8]>,
    setter: Option<&[u8]>,
    build: impl FnOnce(&mut Vec<u8>),
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let the_hole = thread.heap().known().the_hole;
        let undefined = thread.heap().known().undefined;
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
                .allocate_handle::<FixedArray>(&[y.value()], &scope);
            let info = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants,
                    register_count: 1,
                    handlers: None,
                },
                &scope,
            );
            callable_object(thread, &scope, info).value()
        };
        let get = getter.map_or(undefined.value(), |p| make(&mut *thread, p));
        let set = setter.map_or(undefined.value(), |p| make(&mut *thread, p));
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
                prototype: the_hole.erase(),
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
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[obj.value(), x.value(), y.value(), z.value()], &scope);

        // r2 = object
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
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
        emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 1, 0]);
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
        emit(program, Opcode::StoreNamedPropertyNoShadow, &[2, 1, 0]);
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
        emit(program, Opcode::StoreKeyedPropertyNoShadow, &[2, 3, 0]);
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
        let the_hole = thread.heap().known().the_hole;
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
                prototype: the_hole.erase(),
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
                    elements: the_hole.erase(),
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
                .allocate_handle::<FixedArray>(&[y.value()], &scope);
            let info = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants,
                    register_count: 1,
                    handlers: None,
                },
                &scope,
            );
            callable_object(thread, &scope, info)
        };
        let name = scope.handle(SlotName::from(x.as_tagged()).tagged());
        let get = scope.handle(getter.value());
        let set = scope.handle(thread.heap().known().undefined.value());
        Object::define_own_property(
            thread.heap(),
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: get.value(),
                set: set.value(),
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();

        // program: acc = param0.x
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[x.value()], &scope);
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
    let the_hole = thread.heap().known().the_hole;
    let map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT
                .union(MapKind::CALLABLE)
                .union(MapKind::NATIVE)
                .union(MapKind::CONSTRUCTOR),
            value_slot_count: 1,
            descriptors: &[],
            prototype: the_hole.erase(),
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
                elements: the_hole.erase(),
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
    let the_hole = nctx.heap().known().the_hole;
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
            handlers: None,
        },
        scope,
    );
    let map = nctx.heap().known().function_map;
    nctx.heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[info.value(), empty_context.value()],
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .erase()
}

fn forty_two(_: &mut NativeContext<'_>, _: GcSlice<'_>) -> Result<Value, VmError> {
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
        let caller = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 1,
                handlers: None,
            },
            &scope,
        );
        let caller = callable_object(thread, &scope, caller);
        thread.execute(caller, &[])
    });

    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

/// Native that runs bytecode which throws one call deep; the suspended inner
/// frames are abandoned and must be unwound when the native recovers.
fn run_failing_inner(nctx: &mut NativeContext<'_>, _args: GcSlice<'_>) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        // callee: Add on a non-smi accumulator -> TypeError throw
        let mut bad = Vec::new();
        emit(&mut bad, Opcode::Add, &[1]);
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

        match nctx.call(caller, GcSlice::EMPTY) {
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

/// acc = param0; acc = acc op param1; return acc
fn binary_op_program(op: Opcode) -> Vec<u8> {
    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut program, op, &[(-2i32) as u32]);
    emit(&mut program, Opcode::Return, &[]);
    program
}

#[test]
fn arithmetic_ops_use_accumulator_convention() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let cases: &[(Opcode, i64, i64, i64)] = &[
        (Opcode::Add, 6, 7, 13),
        (Opcode::Sub, 10, 4, 6),
        (Opcode::Mul, 6, 7, 42),
        (Opcode::Div, 42, 7, 6),
        (Opcode::Mod, 42, 10, 2),
        (Opcode::Exp, 2, 10, 1024),
        (Opcode::BitwiseOr, 0b1010, 0b0110, 0b1110),
        (Opcode::BitwiseXor, 12, 10, 6),
        (Opcode::BitwiseAnd, 12, 10, 8),
        (Opcode::ShiftLeft, 1, 4, 16),
        (Opcode::ShiftRight, -8, 1, -4),
        (Opcode::ShiftRightLogical, -1, 1, 0x7fff_ffff),
    ];
    for &(op, a, b, expected) in cases {
        let program = binary_op_program(op);
        let result = run_program(&mut thread, program, 0, &[smi(a), smi(b)]);
        assert_eq!(
            Smi::decode(result.unwrap()).unwrap().value(),
            expected,
            "op {op:?} with {a}, {b}"
        );
    }
}

#[test]
fn shift_counts_are_masked_to_five_bits() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // JS: the shift count is ToUint32(rhs) & 31, so -1 shifts by 31
    let result = run_program(
        &mut thread,
        binary_op_program(Opcode::ShiftLeft),
        0,
        &[smi(1), smi(-1)],
    );
    assert_eq!(
        Smi::decode(result.unwrap()).unwrap().value(),
        i32::MIN as i64
    );
}

#[test]
fn arithmetic_overflow_promotes_to_float() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // Smi::MAX - 1 + 5 no longer fits an smi: the double path rounds it to 2^62
    let result = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[smi(Smi::MAX - 1), smi(5)],
    );
    let result = result.unwrap();
    let value = thread.heap().no_gc(|nogc| {
        result
            .get_as::<Float>(nogc)
            .expect("overflow must promote to float")
            .value
            .get()
    });
    assert_eq!(value, (1u64 << 62) as f64);
}

/// acc = constants[0] op constants[1]; return acc
fn binary_op_consts_program(op: Opcode) -> Vec<u8> {
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadConstant, &[1]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadConstant, &[0]);
    emit(&mut program, op, &[0]);
    emit(&mut program, Opcode::Return, &[]);
    program
}

/// Run a binary op whose operands come from the constants pool (for floats
/// and distinct-but-equal strings, which cannot be params in these tests).
fn run_binary_consts(
    thread: &mut Thread,
    op: Opcode,
    a: Value,
    b: Value,
) -> Result<Value, VmError> {
    let program = binary_op_consts_program(op);
    run_program_consts(thread, program, 1, &[], &[a, b])
}

fn float_value(thread: &mut Thread, v: Value) -> f64 {
    thread.heap().no_gc(|nogc| {
        v.get_as::<Float>(nogc)
            .expect("expected float result")
            .value
            .get()
    })
}

#[test]
fn equal_strict_compares_numbers_strings_and_objects() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let true_v = thread.heap().known().true_object.value();
        let false_v = thread.heap().known().false_object.value();
        let nan1 = thread
            .heap()
            .allocate_handle::<Float>(f64::NAN, &scope)
            .value();
        let nan2 = thread
            .heap()
            .allocate_handle::<Float>(f64::NAN, &scope)
            .value();
        let one_float = thread.heap().allocate_handle::<Float>(1.0, &scope).value();
        let mk_string = |thread: &mut Thread, scope: &HandleScope<'_>, s: &str| {
            let backing = thread
                .heap()
                .allocate_handle::<FixedByteArray>(s.as_bytes(), scope);
            thread
                .heap()
                .allocate_handle::<VMString>((backing, string_content_hash(s.as_bytes())), scope)
                .value()
        };
        let ab1 = mk_string(thread, &scope, "ab");
        let ab2 = mk_string(thread, &scope, "ab");
        let ac = mk_string(thread, &scope, "ac");
        let object_init = thread.heap().known();
        let obj = thread
            .heap()
            .new_object(&scope, object_init.object_initial_map, &[])
            .into_handle(&scope)
            .as_tagged()
            .erase();
        let obj2 = thread
            .heap()
            .new_object(&scope, object_init.object_initial_map, &[])
            .into_handle(&scope)
            .as_tagged()
            .erase();

        // smis
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, smi(1), smi(1)).unwrap(),
            true_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, smi(1), smi(2)).unwrap(),
            false_v
        );
        // NaN is unequal to itself, even as two freshly allocated floats
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, nan1, nan2).unwrap(),
            false_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, nan1, nan1).unwrap(),
            false_v
        );
        // Float(1.0) === 1
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, one_float, smi(1)).unwrap(),
            true_v
        );
        // Float(1.0) === "1" is false: no parsing in strict equality
        let s1 = thread.intern(&scope, "1").value();
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, one_float, s1).unwrap(),
            false_v
        );
        // string content equality, including distinct-but-equal strings
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, ab1, ab2).unwrap(),
            true_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, ab1, ac).unwrap(),
            false_v
        );
        // object identity
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, obj, obj).unwrap(),
            true_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::EqualStrict, obj, obj2).unwrap(),
            false_v
        );
    });
}

#[test]
fn abstract_equality_follows_spec() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let known = thread.heap().known();
    let true_v = known.true_object.value();
    let false_v = known.false_object.value();
    let null = known.null.value();
    let undefined = known.undefined.value();

    // null == undefined
    assert_eq!(
        run_program(
            &mut thread,
            binary_op_program(Opcode::Equal),
            0,
            &[null, undefined]
        )
        .unwrap(),
        true_v
    );
    // false == 0, true == 1
    assert_eq!(
        run_program(
            &mut thread,
            binary_op_program(Opcode::Equal),
            0,
            &[known.false_object.value(), smi(0)]
        )
        .unwrap(),
        true_v
    );
    assert_eq!(
        run_program(
            &mut thread,
            binary_op_program(Opcode::Equal),
            0,
            &[known.true_object.value(), smi(1)]
        )
        .unwrap(),
        true_v
    );
    // 0 == null is false (only null/undefined are loosely equal to null)
    assert_eq!(
        run_program(
            &mut thread,
            binary_op_program(Opcode::Equal),
            0,
            &[smi(0), null]
        )
        .unwrap(),
        false_v
    );
    // "" == 0 parses the empty string as +0
    let empty_string = thread.handle_scope(|thread, scope| thread.intern(&scope, "").value());
    assert_eq!(
        run_program(
            &mut thread,
            binary_op_program(Opcode::Equal),
            0,
            &[smi(0), empty_string]
        )
        .unwrap(),
        true_v
    );

    thread.handle_scope(|thread, scope| {
        let s1 = thread.intern(&scope, "1").value();
        let s15 = thread.intern(&scope, "1.5").value();
        let f15 = thread.heap().allocate_handle::<Float>(1.5, &scope).value();
        let nan = thread
            .heap()
            .allocate_handle::<Float>(f64::NAN, &scope)
            .value();
        // "1" == 1, "1.5" == 1.5
        assert_eq!(
            run_binary_consts(thread, Opcode::Equal, s1, smi(1)).unwrap(),
            true_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::Equal, s15, f15).unwrap(),
            true_v
        );
        // NaN == NaN stays false through the loose path too
        assert_eq!(
            run_binary_consts(thread, Opcode::Equal, nan, nan).unwrap(),
            false_v
        );
    });
}

#[test]
fn relational_operators_follow_spec() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let true_v = thread.heap().known().true_object.value();
    let false_v = thread.heap().known().false_object.value();

    let cases: &[(Opcode, i64, i64, Value)] = &[
        (Opcode::LessThan, 1, 2, true_v),
        (Opcode::LessThan, 2, 1, false_v),
        (Opcode::LessThan, 2, 2, false_v),
        (Opcode::LessThanOrEqual, 1, 1, true_v),
        (Opcode::LessThanOrEqual, 2, 1, false_v),
        (Opcode::GreaterThan, 2, 1, true_v),
        (Opcode::GreaterThan, 1, 2, false_v),
        (Opcode::GreaterThan, 2, 2, false_v),
        (Opcode::GreaterThanOrEqual, 2, 2, true_v),
        (Opcode::GreaterThanOrEqual, 1, 2, false_v),
    ];
    for &(op, a, b, expected) in cases {
        let r = run_program(&mut thread, binary_op_program(op), 0, &[smi(a), smi(b)]);
        assert_eq!(r.unwrap(), expected, "op {op:?} with {a}, {b}");
    }

    thread.handle_scope(|thread, scope| {
        let true_v = thread.heap().known().true_object.value();
        let false_v = thread.heap().known().false_object.value();
        let nan = thread
            .heap()
            .allocate_handle::<Float>(f64::NAN, &scope)
            .value();
        let one = smi(1);

        // NaN compares false against everything, in all four directions
        for op in [
            Opcode::LessThan,
            Opcode::LessThanOrEqual,
            Opcode::GreaterThan,
            Opcode::GreaterThanOrEqual,
        ] {
            assert_eq!(
                run_binary_consts(thread, op, nan, one).unwrap(),
                false_v,
                "{op:?}"
            );
            assert_eq!(
                run_binary_consts(thread, op, one, nan).unwrap(),
                false_v,
                "{op:?}"
            );
        }

        // both-strings comparisons are lexicographic
        let abc = thread.intern(&scope, "abc").value();
        let abd = thread.intern(&scope, "abd").value();
        assert_eq!(
            run_binary_consts(thread, Opcode::LessThan, abc, abd).unwrap(),
            true_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::LessThan, abd, abc).unwrap(),
            false_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::LessThanOrEqual, abc, abc).unwrap(),
            true_v
        );

        // mixed string/number comparisons parse the string
        let s2 = thread.intern(&scope, "2").value();
        let s10 = thread.intern(&scope, "10").value();
        assert_eq!(
            run_binary_consts(thread, Opcode::LessThan, s2, smi(10)).unwrap(),
            true_v
        );
        assert_eq!(
            run_binary_consts(thread, Opcode::LessThan, s10, smi(9)).unwrap(),
            false_v
        );
    });
}

#[test]
fn division_and_modulo_follow_ieee() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // 7 / 2 = 3.5 (float), 42 / 7 = 6 (smi fast path, covered above)
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Div),
        0,
        &[smi(7), smi(2)],
    );
    assert_eq!(float_value(&mut thread, r.unwrap()), 3.5);

    // division by zero: sign-correct infinities and NaN
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Div),
        0,
        &[smi(1), smi(0)],
    );
    assert_eq!(float_value(&mut thread, r.unwrap()), f64::INFINITY);
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Div),
        0,
        &[smi(-1), smi(0)],
    );
    assert_eq!(float_value(&mut thread, r.unwrap()), f64::NEG_INFINITY);
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Div),
        0,
        &[smi(0), smi(0)],
    );
    assert!(float_value(&mut thread, r.unwrap()).is_nan());

    // modulo by zero is NaN; the float path yields the IEEE remainder
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Mod),
        0,
        &[smi(5), smi(0)],
    );
    assert!(float_value(&mut thread, r.unwrap()).is_nan());

    let r = thread.handle_scope(|thread, scope| {
        let f55 = thread.heap().allocate_handle::<Float>(5.5, &scope).value();
        run_binary_consts(thread, Opcode::Mod, f55, smi(2)).unwrap()
    });
    assert_eq!(float_value(&mut thread, r), 1.5);
}

#[test]
fn exp_produces_floats() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // fractional exponent
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Exp),
        0,
        &[smi(2), smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 2);
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Exp),
        0,
        &[smi(4), smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 4);

    // fractional results become floats instead of erroring
    let r = thread.handle_scope(|thread, scope| {
        let half = thread.heap().allocate_handle::<Float>(0.5, &scope).value();
        run_binary_consts(thread, Opcode::Exp, smi(2), half).unwrap()
    });
    assert_eq!(float_value(&mut thread, r), 2f64.sqrt());

    // 2 ** -1 = 0.5, 2 ** 1024 overflows doubles to Infinity
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Exp),
        0,
        &[smi(2), smi(-1)],
    );
    assert_eq!(float_value(&mut thread, r.unwrap()), 0.5);
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Exp),
        0,
        &[smi(2), smi(1024)],
    );
    assert_eq!(float_value(&mut thread, r.unwrap()), f64::INFINITY);
}

#[test]
fn arithmetic_coerces_primitives_to_number() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let known = thread.heap().known();

    // null + 1 = 1
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[known.null.value(), smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 1);

    // undefined + 1 = NaN
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[known.undefined.value(), smi(1)],
    );
    assert!(float_value(&mut thread, r.unwrap()).is_nan());

    // true + 1 = 2, false + 1 = 1
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[known.true_object.value(), smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 2);
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[known.false_object.value(), smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 1);

    // "2" * 3 = 6 (strings parse in numeric contexts)
    let r = thread.handle_scope(|thread, scope| {
        let s2 = thread.intern(&scope, "2").value();
        run_binary_consts(thread, Opcode::Mul, s2, smi(3)).unwrap()
    });
    assert_eq!(r.to_i64().unwrap(), 6);
}

/// Like `run_program` but with a non-empty constants table.
fn run_program_consts(
    thread: &mut Thread,
    program: Vec<u8>,
    register_count: usize,
    args: &[Value],
    constants: &[Value],
) -> Result<Value, VmError> {
    thread.handle_scope(|thread, scope| {
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(constants, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, args)
    })
}

#[test]
fn global_store_then_load_roundtrips() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (_, result) = thread.handle_scope(|thread, scope| {
        let x = thread.intern(&scope, "x");
        let x = x.value();

        // x = 42; return x
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadSmi, &[42]);
        emit(&mut program, Opcode::StoreGlobal, &[0, 0]);
        emit(&mut program, Opcode::LoadGlobal, &[0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let result = run_program_consts(&mut *thread, program, 0, &[], &[x]);
        (x, result)
    });

    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);

    // globals persist on the VM: a fresh program sees the stored value
    let result = thread.handle_scope(|thread, scope| {
        let x = thread.intern(&scope, "x");
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadGlobal, &[0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 0, &[], &[x.value()])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn load_global_missing_name_throws_reference_error() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // unresolvable references throw ReferenceError (GetValue on an
    // unresolvable reference); typeof uses LoadGlobalNoThrow instead
    let result = thread.handle_scope(|thread, scope| {
        let missing = thread.intern(&scope, "not_defined_anywhere");
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadGlobal, &[0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 0, &[], &[missing.value()])
    });
    expect_escaped(&mut thread, result, "ReferenceError");

    // the no-throw variant yields undefined
    let result = thread.handle_scope(|thread, scope| {
        let missing = thread.intern(&scope, "not_defined_anywhere");
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadGlobalNoThrow, &[0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 0, &[], &[missing.value()])
    });
    assert_eq!(result.unwrap(), thread.heap().known().undefined.value());
}

#[test]
fn empty_object_literal_inherits_from_object_prototype() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let p = thread.intern(&scope, "p");
        let name = SlotName::from(p.as_tagged());
        let proto = thread.heap().known().object_prototype.value();

        // host-side: %Object.prototype%.p = 1
        let outcome = thread
            .heap()
            .no_gc(|nogc| proto.store_lookup(nogc, name, smi(1), StoreSemantics::Shadow))
            .unwrap();
        match outcome {
            StoreOutcome::Transition { receiver, name } => {
                Object::define_own_property_values(
                    thread.heap(),
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(smi(1)),
                )
                .expect("the transition receiver is fresh and extensible");
            }
            other => panic!("expected transition, got {other:?}"),
        }

        // {}.p reads through the prototype chain
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let first = run_program_consts(&mut *thread, program, 1, &[], &[p.value()]);
        assert_eq!(Smi::decode(first.unwrap()).unwrap().value(), 1);

        // ({}.p = 2) shadows: own property on the instance...
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[2]);
        emit(&mut program, Opcode::StoreNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let second = run_program_consts(&mut *thread, program, 1, &[], &[p.value()]);
        assert_eq!(Smi::decode(second.unwrap()).unwrap().value(), 2);

        // ...and a fresh {} still sees the prototype value
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 1, &[], &[p.value()])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 1);
}

#[test]
fn create_closure_inherits_current_context_and_is_callable() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let the_hole = thread.heap().known().the_hole;

        // callee info template: return context slot 0
        let mut callee_program = Vec::new();
        emit(&mut callee_program, Opcode::LoadContextSlot, &[0, 0]);
        emit(&mut callee_program, Opcode::Return, &[]);
        let callee_bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&callee_program, &scope);
        let callee_consts = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callee_info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: callee_bytecode,
                constants: callee_consts,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );

        // caller: r1 = CreateClosure(template); call r1; return
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateClosure, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::CallNoFeedback, &[1, 1, 1]);
        emit(&mut program, Opcode::Return, &[]);
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let slot_name = thread.intern(&scope, "slot0");
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(&[slot_name.value()], &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[scope_info.value(), callee_info.value()], &scope);
        let caller_info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 2,
                handlers: None,
            },
            &scope,
        );

        // the caller runs in a context whose slot 0 = 42
        let slots = thread
            .heap()
            .allocate_handle::<FixedArray>(&[smi(42)], &scope);
        let scope_info = thread.heap().known().empty_scope_info;
        let context = thread.heap().allocate_handle::<Context>(
            ContextInit {
                outer: None,
                slots,
                scope_info,
            },
            &scope,
        );

        let map = thread.heap().known().function_map;
        let caller = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[caller_info.value(), context.value()],
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        thread.execute(caller, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn create_closure_shares_callable_info_template() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (result, template) = thread.handle_scope(|thread, scope| {
        let callee_bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let callee_consts = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callee_info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: callee_bytecode,
                constants: callee_consts,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );

        // CreateClosure(template); return the closure
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateClosure, &[0]);
        emit(&mut program, Opcode::Return, &[]);
        let result = run_program_consts(&mut *thread, program, 0, &[], &[callee_info.value()]);
        (result.unwrap(), callee_info.value())
    });

    thread.heap().no_gc(|nogc| {
        let Some(o) = result.as_heap_object(nogc) else {
            panic!("closure must be an object");
        };
        let info = o
            .as_ref()
            .callable_info(nogc)
            .expect("closure carries a callable info");
        // the info is shared, not copied per closure
        assert_eq!(info.into_tagged().erase(), template);
        // the closure's context slot is the caller's (empty) context
        let context = o
            .as_ref()
            .closure_context(nogc)
            .expect("closure carries a context");
        assert_eq!(
            context.into_tagged().erase(),
            nogc.known().empty_context.value()
        );
    });
}

#[test]
fn create_closure_function_kind_controls_call_and_construct() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut method_body = Vec::new();
    emit(&mut method_body, Opcode::LoadSmi, &[7]);
    emit(&mut method_body, Opcode::Return, &[]);
    let method = create_closure_of_kind(
        &mut thread,
        &method_body,
        "method",
        1,
        FunctionKind::Method,
        true,
    );

    let prototype = thread.handle_scope(|thread, scope| thread.intern(&scope, "prototype").value());
    thread.heap().no_gc(|nogc| {
        let Some(method) = method.as_heap_object(nogc) else {
            panic!("method must be an object")
        };
        let method = method.as_ref();
        assert!(method.map_ref(nogc).kind().is_callable());
        assert!(!method.map_ref(nogc).kind().is_constructor());
        assert!(matches!(
            method.lookup(nogc, SlotName::from_value(prototype)),
            Lookup::NotFound
        ));
    });

    let undefined = thread.heap().known().undefined.value();
    let mut call = Vec::new();
    emit(&mut call, Opcode::LoadConstant, &[0]);
    emit(&mut call, Opcode::Store, &[0]);
    emit(&mut call, Opcode::LoadConstant, &[1]);
    emit(&mut call, Opcode::Store, &[1]);
    emit(&mut call, Opcode::CallNoFeedback, &[0, 1, 1]);
    emit(&mut call, Opcode::Return, &[]);
    let result = run_program_consts(&mut thread, call, 2, &[], &[method, undefined]).unwrap();
    assert_eq!(result.to_i64().unwrap(), 7);

    let mut construct = Vec::new();
    emit(&mut construct, Opcode::LoadConstant, &[0]);
    emit(&mut construct, Opcode::Store, &[0]);
    emit(&mut construct, Opcode::Construct, &[0, 0, 0]);
    emit(&mut construct, Opcode::Return, &[]);
    let result = run_program_consts(&mut thread, construct, 1, &[], &[method]);
    expect_escaped(&mut thread, result, "TypeError");

    let mut constructor_body = Vec::new();
    emit(&mut constructor_body, Opcode::LoadSmi, &[1]);
    emit(&mut constructor_body, Opcode::Return, &[]);
    let class_constructor = create_closure_of_kind(
        &mut thread,
        &constructor_body,
        "C",
        0,
        FunctionKind::BaseClassConstructor,
        true,
    );
    thread.heap().no_gc(|nogc| {
        let Some(constructor) = class_constructor.as_heap_object(nogc) else {
            panic!("class constructor must be an object")
        };
        let kind = constructor.as_ref().map_ref(nogc).kind();
        assert!(kind.is_callable());
        assert!(kind.is_constructor());
        assert!(kind.is_class_constructor());
        assert!(matches!(
            constructor
                .as_ref()
                .lookup(nogc, SlotName::from_value(prototype)),
            Lookup::NotFound
        ));
    });

    let mut call = Vec::new();
    emit(&mut call, Opcode::LoadConstant, &[0]);
    emit(&mut call, Opcode::Store, &[0]);
    emit(&mut call, Opcode::LoadConstant, &[1]);
    emit(&mut call, Opcode::Store, &[1]);
    emit(&mut call, Opcode::CallNoFeedback, &[0, 1, 1]);
    emit(&mut call, Opcode::Return, &[]);
    let result = run_program_consts(&mut thread, call, 2, &[], &[class_constructor, undefined]);
    expect_escaped(&mut thread, result, "TypeError");

    let mut construct = Vec::new();
    emit(&mut construct, Opcode::LoadConstant, &[0]);
    emit(&mut construct, Opcode::Store, &[0]);
    emit(&mut construct, Opcode::Construct, &[0, 0, 0]);
    emit(&mut construct, Opcode::Return, &[]);
    let result = run_program_consts(&mut thread, construct, 1, &[], &[class_constructor]).unwrap();
    assert!(
        result.is_strong_ptr(),
        "construction must return a receiver"
    );
}

#[test]
fn function_context_slots_are_readable_and_writable() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // CreateFunctionContext (2 slots); PushContext r0; x = 42; y = 43
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateFunctionContext, &[0]);
    emit(&mut program, Opcode::PushContext, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::StoreContextSlot, &[0, 0]);
    emit(&mut program, Opcode::LoadSmi, &[43]);
    emit(&mut program, Opcode::StoreContextSlot, &[1, 0]);
    emit(&mut program, Opcode::LoadContextSlot, &[0, 0]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadContextSlot, &[1, 0]);
    emit(&mut program, Opcode::Add, &[1]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program_ctx(&mut thread, program, 2, &[], 2);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 85);
}

#[test]
fn push_context_saves_previous_context_to_register() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // PushContext must save the old frame context (empty_context) into r0
    let result = thread.handle_scope(|thread, scope| {
        let empty = thread.heap().known().empty_context.value();
        let slot_name = thread.intern(&scope, "slot0");
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(&[slot_name.value()], &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateFunctionContext, &[0]);
        emit(&mut program, Opcode::PushContext, &[0]);
        emit(&mut program, Opcode::Load, &[0]); // r0 = saved old context
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::LoadConstant, &[1]); // acc = empty_context
        emit(&mut program, Opcode::TestReferenceEqual, &[1]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 2, &[], &[scope_info.value(), empty])
    });
    assert_eq!(result.unwrap(), thread.heap().known().true_object.value());
}

#[test]
fn pop_context_restores_previous_context() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // ctxA[0] = 42; push ctxB; ctxB[0] = 99; pop back to ctxA; read ctxA[0]
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateFunctionContext, &[0]);
    emit(&mut program, Opcode::PushContext, &[0]); // r0 = old; frame = ctxA
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::StoreContextSlot, &[0, 0]);
    emit(&mut program, Opcode::CreateBlockContext, &[1]);
    emit(&mut program, Opcode::PushContext, &[1]); // r1 = ctxA; frame = ctxB
    emit(&mut program, Opcode::LoadSmi, &[99]);
    emit(&mut program, Opcode::StoreContextSlot, &[0, 0]); // ctxB[0] = 99
    emit(&mut program, Opcode::PopContext, &[1]); // frame = ctxA
    emit(&mut program, Opcode::LoadContextSlot, &[0, 0]); // ctxA[0]
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program_ctx(&mut thread, program, 2, &[], 1);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn block_context_reads_outer_scope_via_depth() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // ctxA[0] = 7; inside ctxB (outer = ctxA), read slot 0 at depth 1
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateFunctionContext, &[0]);
    emit(&mut program, Opcode::PushContext, &[0]); // frame = ctxA
    emit(&mut program, Opcode::LoadSmi, &[7]);
    emit(&mut program, Opcode::StoreContextSlot, &[0, 0]);
    emit(&mut program, Opcode::CreateBlockContext, &[1]);
    emit(&mut program, Opcode::PushContext, &[1]); // frame = ctxB, outer = ctxA
    emit(&mut program, Opcode::LoadContextSlot, &[0, 1]); // ctxB.outer[0]
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program_ctx(&mut thread, program, 2, &[], 1);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn catch_context_binds_exception_in_slot_zero() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // CreateCatchContext r2 (exception); PushContext; read slot 0
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[55]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::CreateCatchContext, &[2]);
    emit(&mut program, Opcode::PushContext, &[0]);
    emit(&mut program, Opcode::LoadContextSlot, &[0, 0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 55);
}

#[test]
fn tdz_hole_read_throws_reference_error() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // fresh context slots are the hole; reading one must throw ReferenceError
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateFunctionContext, &[0]);
    emit(&mut program, Opcode::PushContext, &[0]);
    emit(&mut program, Opcode::LoadContextSlot, &[0, 0]);
    emit(&mut program, Opcode::ThrowReferenceErrorIfHole, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program_ctx(&mut thread, program, 1, &[], 1);
    expect_escaped(&mut thread, result, "ReferenceError");
}

#[test]
fn closure_captures_function_context_end_to_end() {
    // function f() { let x = 1; { let y = 2; } return () => x; }
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        // callee (arrow): return x from its (inherited) context
        let mut callee_program = Vec::new();
        emit(&mut callee_program, Opcode::LoadContextSlot, &[0, 0]);
        emit(&mut callee_program, Opcode::Return, &[]);
        let callee_bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&callee_program, &scope);
        let callee_consts = thread.heap().allocate_handle::<FixedArray>(&[], &scope);
        let callee_info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode: callee_bytecode,
                constants: callee_consts,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );

        // f's body
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateFunctionContext, &[0]); // ctxA: [x]
        emit(&mut program, Opcode::PushContext, &[0]); // r0 = old; frame = ctxA
        emit(&mut program, Opcode::LoadSmi, &[1]);
        emit(&mut program, Opcode::StoreContextSlot, &[0, 0]); // x = 1
        emit(&mut program, Opcode::CreateBlockContext, &[1]); // ctxB: [y]
        emit(&mut program, Opcode::PushContext, &[1]); // r1 = ctxA; frame = ctxB
        emit(&mut program, Opcode::LoadSmi, &[2]);
        emit(&mut program, Opcode::StoreContextSlot, &[0, 0]); // y = 2
        emit(&mut program, Opcode::PopContext, &[1]); // frame = ctxA
        emit(&mut program, Opcode::CreateClosure, &[1]); // closure ctx = ctxA
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::CallNoFeedback, &[2, 2, 1]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let slot_name = thread.intern(&scope, "slot0");
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(&[slot_name.value()], &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[scope_info.value(), callee_info.value()], &scope);
        let caller_info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 3,
                handlers: None,
            },
            &scope,
        );
        let caller = callable_object(thread, &scope, caller_info);
        thread.execute(caller, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 1);
}

/// Host-side: an extendable object with a writable `p = 7` slot.
fn proto_object<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    p: vm::Handle<'s, vm::InternedString>,
) -> vm::Handle<'s, Object> {
    let the_hole = thread.heap().known().the_hole;
    let map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: EXTENDABLE,
            value_slot_count: 1,
            descriptors: &[(
                SlotName::from(p.as_tagged()),
                WRITABLE_VALUE,
                Smi::new(0).encode(),
            )],
            prototype: the_hole.erase(),
        },
        scope,
    );
    thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[smi(7)],
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .into_handle(scope)
}

#[test]
fn set_prototype_changes_property_lookup_chain() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r0.[[Prototype]] = objB (p = 7); return r0.p
    let result = thread.handle_scope(|thread, scope| {
        let p = thread.intern(&scope, "p");
        let obj_b = proto_object(&mut *thread, &scope, p);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[p.value(), obj_b.value()], &scope);

        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::SetPrototype, &[1]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 2,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn set_prototype_survives_property_transitions() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r0.[[Prototype]] = objB; r0.x = 1 (transition); return r0.p
    let result = thread.handle_scope(|thread, scope| {
        let p = thread.intern(&scope, "p");
        let x = thread.intern(&scope, "x");
        let obj_b = proto_object(&mut *thread, &scope, p);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[p.value(), obj_b.value(), x.value()], &scope);

        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::SetPrototype, &[1]);
        emit(&mut program, Opcode::LoadSmi, &[1]);
        emit(&mut program, Opcode::StoreNamedProperty, &[0, 2, 0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 2,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn set_prototype_cycle_throws_type_error() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r0.[[Prototype]] = r0 (cycle)
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::SetPrototype, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 1, &[]);
    expect_escaped(&mut thread, result, "TypeError");
}

#[test]
fn set_prototype_on_non_extensible_throws_type_error() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let p = thread.intern(&scope, "p");
        let obj_b = proto_object(&mut *thread, &scope, p);

        // host-side: a plain, non-extendable object
        let the_hole = thread.heap().known().the_hole;
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT,
                value_slot_count: 0,
                descriptors: &[],
                prototype: the_hole.erase(),
            },
            &scope,
        );
        let frozen = thread
            .heap()
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);

        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(&[frozen.value(), obj_b.value()], &scope);
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::SetPrototype, &[1]);
        emit(&mut program, Opcode::Return, &[]);

        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let callable = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants: consts,
                register_count: 2,
                handlers: None,
            },
            &scope,
        );
        let callable = callable_object(thread, &scope, callable);
        thread.execute(callable, &[])
    });
    expect_escaped(&mut thread, result, "TypeError");
}

/// Build a callable function object wrapping `program` with `constants`.
fn make_callable<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    program: &[u8],
    constants: &[Value],
) -> Value {
    let bytecode = thread
        .heap()
        .allocate_handle::<FixedByteArray>(program, scope);
    let constants = thread
        .heap()
        .allocate_handle::<FixedArray>(constants, scope);
    let info = thread.heap().allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            bytecode,
            constants,
            register_count: 1,
            handlers: None,
        },
        scope,
    );
    callable_object(thread, scope, info).value()
}

/// Fresh `{}` with the realm's object initial map.
fn empty_object<'s>(thread: &mut Thread, scope: &'s HandleScope<'_>) -> Handle<'s, Object> {
    let known = thread.heap().known();
    thread
        .heap()
        .new_object(scope, known.object_initial_map, &[])
        .into_handle(scope)
}

/// Own callable data property `name` -> function running `program`.
fn set_property_fn(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    obj: Value,
    name: Value,
    program: &[u8],
    constants: &[Value],
) {
    let f = make_callable(thread, scope, program, constants);
    Object::define_own_property_values(
        thread.heap(),
        scope,
        obj,
        SlotName::from_value(name),
        PropertyDescriptor::data(f),
    )
    .expect("defining a fresh own property must succeed");
}

fn program_return_1() -> Vec<u8> {
    let mut p = Vec::new();
    emit(&mut p, Opcode::LoadSmi, &[1]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

fn program_return_constant() -> Vec<u8> {
    let mut p = Vec::new();
    emit(&mut p, Opcode::LoadConstant, &[0]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

fn program_return_empty_object() -> Vec<u8> {
    let mut p = Vec::new();
    emit(&mut p, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

/// Call a non-callable (smi) → TypeError escapes the function.
fn program_throw_type_error() -> Vec<u8> {
    let mut p = Vec::new();
    emit(&mut p, Opcode::LoadSmi, &[0]);
    emit(&mut p, Opcode::Store, &[0]);
    emit(&mut p, Opcode::LoadSmi, &[0]);
    emit(&mut p, Opcode::CallNoFeedback, &[0, 0, 0]);
    emit(&mut p, Opcode::Return, &[]);
    p
}

#[test]
fn to_primitive_calls_value_of_in_numeric_contexts() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj = empty_object(thread, &scope).value();
        let value_of = thread.intern(&scope, "valueOf").value();
        set_property_fn(thread, &scope, obj, value_of, &program_return_1(), &[]);

        // + and * both coerce the object with hint number → valueOf() = 1
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Add),
            0,
            &[obj, smi(1)],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 2);
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Mul),
            0,
            &[obj, smi(3)],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 3);
    });
}

#[test]
fn to_primitive_falls_back_to_to_string_when_value_of_yields_object() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj = empty_object(thread, &scope).value();
        let value_of = thread.intern(&scope, "valueOf").value();
        let to_string = thread.intern(&scope, "toString").value();
        let x = thread.intern(&scope, "x").value();

        // valueOf returns an object → skipped → toString() = "x"
        set_property_fn(
            thread,
            &scope,
            obj,
            value_of,
            &program_return_empty_object(),
            &[],
        );
        set_property_fn(
            thread,
            &scope,
            obj,
            to_string,
            &program_return_constant(),
            &[x],
        );

        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Add),
            0,
            &[obj, smi(1)],
        )
        .unwrap();
        thread.heap().no_gc(|nogc| {
            let s = r
                .get_as::<VMString>(nogc)
                .expect("concat result must be a string");
            assert_eq!(s.as_str(nogc), Some("x1"));
        });
    });
}

#[test]
fn add_concatenates_strings() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        // string + string, string + number, number + string
        let a = thread.intern(&scope, "a").value();
        let b = thread.intern(&scope, "b").value();
        for (lhs, rhs, expected) in [
            (a, b, "ab"),
            (a, smi(2), "a2"),
            (smi(2), b, "2b"),
            (a, smi(1000), "a1000"),
        ] {
            let r =
                run_program(&mut *thread, binary_op_program(Opcode::Add), 0, &[lhs, rhs]).unwrap();
            thread.heap().no_gc(|nogc| {
                let s = r
                    .get_as::<VMString>(nogc)
                    .expect("concat result must be a string");
                assert_eq!(s.as_str(nogc), Some(expected), "{lhs:?} + {rhs:?}");
            });
        }
    });
}

#[test]
fn to_primitive_uses_to_primitive_symbol_first() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj = empty_object(thread, &scope).value();
        // @@toPrimitive = () => 1: wins over valueOf, called with hint "default"
        let sym = thread.heap().known().to_primitive_symbol.value();
        let value_of = thread.intern(&scope, "valueOf").value();
        set_property_fn(thread, &scope, obj, sym, &program_return_1(), &[]);
        set_property_fn(
            thread,
            &scope,
            obj,
            value_of,
            &{
                // valueOf = () => 2: must NOT be called
                let mut p = Vec::new();
                emit(&mut p, Opcode::LoadSmi, &[2]);
                emit(&mut p, Opcode::Return, &[]);
                p
            },
            &[],
        );

        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Add),
            0,
            &[obj, smi(1)],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 2);
    });
}

#[test]
fn to_primitive_symbol_returning_object_throws() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let obj = thread.handle_scope(|thread, scope| {
        let obj = empty_object(thread, &scope).value();
        let sym = thread.heap().known().to_primitive_symbol.value();
        set_property_fn(
            thread,
            &scope,
            obj,
            sym,
            &program_return_empty_object(),
            &[],
        );
        obj
    });
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[obj, smi(1)],
    );
    expect_escaped(&mut thread, r, "TypeError");
}

#[test]
fn to_primitive_calls_getter_accessors() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj = empty_object(thread, &scope);
        let value_of = thread.intern(&scope, "valueOf");
        // valueOf defined as an accessor whose getter returns a function
        // returning 2 (Get(O, "valueOf") runs the getter, then the result is
        // called with the object as receiver)
        let inner = make_callable(
            thread,
            &scope,
            &{
                let mut p = Vec::new();
                emit(&mut p, Opcode::LoadSmi, &[2]);
                emit(&mut p, Opcode::Return, &[]);
                p
            },
            &[],
        );
        let getter = make_callable(thread, &scope, &program_return_constant(), &[inner]);
        let name = scope.handle(SlotName::from_value(value_of.value()).tagged());
        let get = scope.handle(getter);
        let set = scope.handle(thread.heap().known().undefined.value());
        Object::define_own_property(
            thread.heap(),
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: get.value(),
                set: set.value(),
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();

        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Add),
            0,
            &[obj.value(), smi(1)],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 3);
    });
}

#[test]
fn relational_and_equality_operators_coerce_objects() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let true_v = thread.heap().known().true_object.value();
        let obj = empty_object(thread, &scope).value();
        let value_of = thread.intern(&scope, "valueOf").value();
        // valueOf = () => 2
        set_property_fn(
            thread,
            &scope,
            obj,
            value_of,
            &{
                let mut p = Vec::new();
                emit(&mut p, Opcode::LoadSmi, &[2]);
                emit(&mut p, Opcode::Return, &[]);
                p
            },
            &[],
        );

        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::LessThan),
            0,
            &[obj, smi(3)],
        )
        .unwrap();
        assert_eq!(r, true_v);
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::GreaterThan),
            0,
            &[obj, smi(3)],
        )
        .unwrap();
        assert_eq!(r, thread.heap().known().false_object.value());
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Equal),
            0,
            &[obj, smi(2)],
        )
        .unwrap();
        assert_eq!(r, true_v);
    });
}

#[test]
fn value_of_exception_propagates() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    let obj = thread.handle_scope(|thread, scope| {
        let obj = empty_object(thread, &scope).value();
        let value_of = thread.intern(&scope, "valueOf").value();
        set_property_fn(
            thread,
            &scope,
            obj,
            value_of,
            &program_throw_type_error(),
            &[],
        );
        obj
    });
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[obj, smi(1)],
    );
    expect_escaped(&mut thread, r, "TypeError");
}

/// acc = param0; acc = unary op; return acc
fn unary_program(op: Opcode) -> Vec<u8> {
    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut program, op, &[]);
    emit(&mut program, Opcode::Return, &[]);
    program
}

#[test]
fn typeof_reports_spec_types() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let number = thread.intern(&scope, "number").value();
        let string = thread.intern(&scope, "string").value();
        let undefined = thread.intern(&scope, "undefined").value();
        let object = thread.intern(&scope, "object").value();
        let boolean = thread.intern(&scope, "boolean").value();
        let function = thread.intern(&scope, "function").value();

        let f = make_callable(thread, &scope, &program_return_1(), &[]);
        let obj = empty_object(thread, &scope).value();
        let float = thread.heap().allocate_handle::<Float>(1.5, &scope).value();
        let s = thread.intern(&scope, "x").value();
        let known = thread.heap().known();

        let cases: &[(Value, Value)] = &[
            (smi(3), number),
            (float, number),
            (s, string),
            (known.undefined.value(), undefined),
            (known.null.value(), object),
            (known.true_object.value(), boolean),
            (known.false_object.value(), boolean),
            (f, function),
            (obj, object),
        ];
        for &(input, expected) in cases {
            let r = run_program(&mut *thread, unary_program(Opcode::TestTypeof), 0, &[input]);
            assert_eq!(r.unwrap(), expected, "typeof {input:?}");
        }
    });
}

#[test]
fn negate_arithmetic_rules() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // smi fast paths
    let r = run_program(&mut thread, unary_program(Opcode::Negate), 0, &[smi(5)]).unwrap();
    assert_eq!(r.to_i64().unwrap(), -5);
    let r = run_program(&mut thread, unary_program(Opcode::Negate), 0, &[smi(-7)]).unwrap();
    assert_eq!(r.to_i64().unwrap(), 7);

    // -0 must be the -0.0 HeapNumber (1 / -0 === -Infinity)
    let r = run_program(&mut thread, unary_program(Opcode::Negate), 0, &[smi(0)]).unwrap();
    let r = thread.heap().no_gc(|nogc| {
        r.get_as::<Float>(nogc)
            .expect("-0 must stay a float")
            .value
            .get()
    });
    assert_eq!(r, 0.0);
    assert!(r.is_sign_negative());

    // Smi::MIN overflows to the double path
    let r = run_program(
        &mut thread,
        unary_program(Opcode::Negate),
        0,
        &[smi(Smi::MIN)],
    )
    .unwrap();
    assert_eq!(float_value(&mut thread, r), (1u64 << 62) as f64);

    // floats and string coercion go through ToNumber
    let r = thread.handle_scope(|thread, scope| {
        let half = thread.heap().allocate_handle::<Float>(1.5, &scope).value();
        run_program(&mut *thread, unary_program(Opcode::Negate), 0, &[half]).unwrap()
    });
    assert_eq!(float_value(&mut thread, r), -1.5);
    let r = thread.handle_scope(|thread, scope| {
        let s3 = thread.intern(&scope, "3").value();
        run_program(&mut *thread, unary_program(Opcode::Negate), 0, &[s3]).unwrap()
    });
    assert_eq!(r.to_i64().unwrap(), -3);
}

#[test]
fn instance_of_walks_prototype_chain() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let known = thread.heap().known();
        let true_v = known.true_object.value();
        let false_v = known.false_object.value();
        let prototype = thread.intern(&scope, "prototype").value();

        // F with a .prototype object
        let f = make_callable(thread, &scope, &program_return_1(), &[]);
        let f_proto = empty_object(thread, &scope).value();
        Object::define_own_property_values(
            thread.heap(),
            &scope,
            f,
            SlotName::from_value(prototype),
            PropertyDescriptor::data(f_proto),
        )
        .expect("defining a fresh own property must succeed");

        // obj inherits F.prototype; plain {} does not
        let obj = empty_object(thread, &scope).value();
        Object::set_prototype(thread.heap(), &scope, obj, f_proto).unwrap();
        let plain = empty_object(thread, &scope).value();

        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::InstanceOf),
            0,
            &[obj, f],
        )
        .unwrap();
        assert_eq!(r, true_v);
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::InstanceOf),
            0,
            &[plain, f],
        )
        .unwrap();
        assert_eq!(r, false_v);

        // a non-callable right operand is a TypeError
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::InstanceOf),
            0,
            &[obj, smi(5)],
        );
        expect_escaped(&mut *thread, r, "TypeError");

        // a non-object .prototype is a TypeError
        let f2 = make_callable(thread, &scope, &program_return_1(), &[]);
        Object::define_own_property_values(
            thread.heap(),
            &scope,
            f2,
            SlotName::from_value(prototype),
            PropertyDescriptor::data(smi(42)),
        )
        .expect("defining a fresh own property must succeed");
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::InstanceOf),
            0,
            &[obj, f2],
        );
        expect_escaped(&mut *thread, r, "TypeError");
    });
}

#[test]
fn construct_uses_prototype_receiver_and_prefers_object_result() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let known = thread.heap().known();
        let prototype = thread.intern(&scope, "prototype").value();

        // G returns a primitive (42): the synthesized receiver wins, and its
        // [[Prototype]] is G.prototype
        let g = make_callable(
            thread,
            &scope,
            &{
                let mut p = Vec::new();
                emit(&mut p, Opcode::LoadSmi, &[42]);
                emit(&mut p, Opcode::Return, &[]);
                p
            },
            &[],
        );
        let g_proto = empty_object(thread, &scope).value();
        Object::define_own_property_values(
            thread.heap(),
            &scope,
            g,
            SlotName::from_value(prototype),
            PropertyDescriptor::data(g_proto),
        )
        .expect("defining a fresh own property must succeed");

        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[0]);
        emit(&mut program, Opcode::Construct, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let r = run_program_consts(&mut *thread, program, 1, &[], &[g]).unwrap();
        assert!(r.is_strong_ptr(), "primitive result: receiver must win");

        // the receiver is `instanceof G`
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[0]);
        emit(&mut program, Opcode::Construct, &[0, 0, 0]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[1]);
        emit(&mut program, Opcode::InstanceOf, &[0]);
        emit(&mut program, Opcode::Return, &[]);
        let r = run_program_consts(&mut *thread, program, 2, &[], &[g]).unwrap();
        assert_eq!(r, known.true_object.value());

        // F returns an object: the object result wins over the receiver
        let f = make_callable(thread, &scope, &program_return_empty_object(), &[]);
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[0]);
        emit(&mut program, Opcode::Construct, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let r = run_program_consts(&mut *thread, program, 1, &[], &[f]).unwrap();
        assert!(r.is_strong_ptr(), "object result must win");

        // constructing a non-constructible value is a TypeError
        let plain = empty_object(thread, &scope).value();
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[0]);
        emit(&mut program, Opcode::Construct, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let r = run_program_consts(&mut *thread, program, 1, &[], &[plain]);
        expect_escaped(&mut *thread, r, "TypeError");
    });
}

/// Native constructor probe: reports `nctx.is_construct()` by storing 1/0
/// into the global property "constructProbe".
fn construct_probe(nctx: &mut NativeContext<'_>, _args: GcSlice<'_>) -> Result<Value, VmError> {
    let flag = Smi::new(if nctx.is_construct() { 1 } else { 0 }).encode();
    nctx.handle_scope(|nctx, scope| {
        let name = nctx.intern(&scope, "constructProbe");
        let global = nctx.heap().known().global_object.value();
        let outcome = nctx.heap().no_gc(|nogc| {
            global.store_lookup(
                nogc,
                SlotName::from(name.as_tagged()),
                flag,
                StoreSemantics::WriteThrough,
            )
        })?;
        match outcome {
            StoreOutcome::Done => {}
            StoreOutcome::Transition { .. } => {
                Object::define_own_property_values(
                    nctx.heap(),
                    &scope,
                    global,
                    SlotName::from(name.as_tagged()),
                    PropertyDescriptor::data(flag),
                )?;
            }
            StoreOutcome::CallSetter { setter } => {
                nctx.handle_scope(|nctx, scope| nctx.call(setter, scope.stage(&[global, flag])))?;
            }
        }
        Ok(())
    })?;
    Ok(flag)
}

#[test]
fn construct_sets_native_construct_flag() {
    let mut vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(construct_probe);
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let f = native_function(thread, &scope, idx);
        let f = f.value();
        let name = thread.intern(&scope, "constructProbe").value();

        // Construct: flag is 1, and the probe's primitive result loses to
        // the receiver (an object)
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[0]);
        emit(&mut program, Opcode::Construct, &[0, 0, 0]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::LoadGlobal, &[1, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let r = run_program_consts(&mut *thread, program, 2, &[], &[f, name]).unwrap();
        assert_eq!(r.to_i64().unwrap(), 1);

        // plain Call: flag is 0
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::CallNoFeedback, &[0, 0, 1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::LoadGlobal, &[1, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let r = run_program_consts(&mut *thread, program, 2, &[], &[f, name]).unwrap();
        assert_eq!(r.to_i64().unwrap(), 0);
    });
}

/// parent (own writable p=1) + child whose prototype is FixedArray([parent]).
/// `child_extendable` controls whether shadowing may define an own property.
fn shadow_setup<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    child_extendable: bool,
) -> (Handle<'s, Object>, Handle<'s, Object>, Value) {
    let the_hole = thread.heap().known().the_hole;
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
            prototype: the_hole.erase(),
        },
        &scope,
    );
    let parent = thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map: parent_map,
                values: &[Smi::new(1).encode()],
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .into_handle(scope);
    let parents = thread
        .heap()
        .allocate_handle::<FixedArray>(&[parent.value()], &scope);
    let child_map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: if child_extendable {
                EXTENDABLE
            } else {
                MapKind::OBJECT
            },
            value_slot_count: 0,
            descriptors: &[],
            prototype: parents.erase(),
        },
        &scope,
    );
    let child = thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map: child_map,
                values: &[],
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .into_handle(scope);
    (child, parent, p.value())
}

/// r2 = child (constants[0]); acc = 2; r2.p = acc via `store_op`; return acc
fn shadow_store_program(store_op: Opcode) -> Vec<u8> {
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadConstant, &[0]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::LoadSmi, &[2]);
    emit(&mut program, store_op, &[2, 1, 0]);
    emit(&mut program, Opcode::Return, &[]);
    program
}

#[test]
fn shadow_store_to_non_extensible_receiver_is_ignored() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let (child, parent, p) = shadow_setup(thread, &scope, false);
        // sloppy [[Set]] on an inherited writable property shadows with an
        // own define; the non-extensible receiver rejects it (false) and the
        // store is silently ignored
        let r = run_program_consts(
            &mut *thread,
            shadow_store_program(Opcode::StoreNamedProperty),
            3,
            &[],
            &[child.value(), p, parent.value()],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 2, "acc keeps the value");

        thread.heap().no_gc(|nogc| {
            // no own property appeared on the child, the parent is untouched
            let child_ref = child.heap_ref(nogc);
            assert_eq!(child_ref.header.map.heap_ref(nogc).descriptor_count(), 0);
            let Some(parent_ref) = parent.value().as_heap_object(nogc) else {
                panic!("parent must be an object");
            };
            match parent_ref.as_ref().lookup(nogc, SlotName::from_value(p)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(Smi::decode(slot.inner()).unwrap().value(), 1);
                }
                _ => panic!("parent must keep its writable property"),
            }
        });
    });
}

#[test]
fn shadow_store_defines_default_attributes() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let (child, parent, p) = shadow_setup(thread, &scope, true);
        let r = run_program_consts(
            &mut *thread,
            shadow_store_program(Opcode::StoreNamedProperty),
            3,
            &[],
            &[child.value(), p, parent.value()],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 2);

        thread.heap().no_gc(|nogc| {
            let child_ref = child.heap_ref(nogc);
            let map = child_ref.header.map.heap_ref(nogc);
            assert_eq!(map.descriptor_count(), 1);
            let d = map.descriptor(0);
            assert_eq!(d.name(), SlotName::from_value(p));
            assert_eq!(d.offset(), 0);
            // [[Set]] shadowing defines with the assignment defaults
            assert!(d.flags().is_writable());
            assert!(d.flags().is_enumerable());
            assert!(d.flags().is_configurable());
            // the own slot wins, the parent keeps its value
            match child_ref.as_ref().lookup(nogc, SlotName::from_value(p)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(Smi::decode(slot.inner()).unwrap().value(), 2);
                }
                _ => panic!("expected own data property"),
            }
            let Some(parent_ref) = parent.value().as_heap_object(nogc) else {
                panic!("parent must be an object");
            };
            match parent_ref.as_ref().lookup(nogc, SlotName::from_value(p)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(Smi::decode(slot.inner()).unwrap().value(), 1);
                }
                _ => panic!("parent must keep its writable property"),
            }
        });
    });
}
