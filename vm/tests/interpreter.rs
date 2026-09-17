use bytecode::{Opcode, PropertyFlags, emit};
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    AccessorPair, CallableInfoInit, CallableInfoObject, Context, ContextInit, DenseString,
    FixedArray, FixedByteArray, Float, FunctionKind, GcSlice, Handle, HandleScope, Heap, HeapPtr,
    Lookup, Map, MapInit, MapKind, Object, ObjectSlotsInit, PropertyDescriptor, ScopeInfo,
    ScopeInfoInit, SlotFlags, SlotName, Smi, StoreOutcome, StoreSemantics, Tagged, Value,
};
use vm::{NativeContext, NativeIndex, Thread, VM, VmError};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

/// A handle's current word, read under a heap anchor.
fn word<'s, T>(heap: &Heap, h: Handle<'s, T>) -> Value {
    h.as_tagged(heap).erase()
}

/// A well-known global's word for `thread`'s heap.
fn global_word<T>(
    thread: &mut Thread,
    pick: impl FnOnce(&vm::WellKnown) -> vm::Global<T>,
) -> Value {
    let heap = thread.heap();
    pick(heap.known()).as_tagged(heap).erase()
}

/// An interned string's word.
fn intern_word(thread: &mut Thread, scope: &HandleScope<'_>, s: &str) -> Value {
    let interned = thread.intern(scope, s);
    let heap = &*thread.heap();
    interned.as_tagged(heap).erase()
}

/// Stage raw words: every call site in this file stages words loaded or
/// allocated under a still-live heap borrow, with no GC since the load.
fn stage_values<'s>(scope: &'s HandleScope<'_>, words: &[Value]) -> GcSlice<'s> {
    let tagged: Vec<Tagged<'_, Value>> = words
        .iter()
        .map(|w| unsafe { Tagged::from_value_unchecked(*w) })
        .collect();
    scope.stage(&tagged)
}

/// Re-anchor a raw word under a heap borrow (see `stage_values`).
unsafe fn anchored<'a>(heap: &'a Heap, v: Value) -> Tagged<'a, Value> {
    unsafe { v.assume_valid(heap) }
}

/// Assert a run escaped uncaught: the sentinel is returned and the pending
/// exception is a materialized error object of the given class name.
fn expect_escaped(thread: &mut Thread, result: Result<Value, VmError>, class: &str) -> Value {
    let exception_word = global_word(thread, |k| k.exception);
    assert_eq!(result, Ok(exception_word), "run must escape uncaught");
    let ex = thread.take_pending_exception().expect("pending exception");
    assert!(!thread.has_pending_exception(), "pending cleared on take");
    let expected_name = thread.handle_scope(|thread, scope| intern_word(thread, &scope, class));
    thread.handle_scope(|thread, scope| {
        let name = thread.intern(&scope, "name");
        thread.heap().no_gc(|heap| {
            let Some(o) = unsafe { anchored(heap, ex) }.as_heap_object() else {
                panic!("pending exception must be an object");
            };
            match o
                .as_ref()
                .lookup(heap, SlotName::from(name.as_tagged(heap)))
            {
                Lookup::Data { slot, .. } => {
                    assert_eq!(slot.get(heap).erase(), expected_name, "error class name");
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
    let values = {
        let heap = &*thread.heap();
        stage_values(
            scope,
            &[word(heap, info), word(heap, empty_context.erase())],
        )
    };
    thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values,
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
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
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
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );
        let name = intern_word(&mut *thread, &scope, name);
        thread.heap().no_gc(|heap| {
            info.heap_ref(heap).set_metadata(
                heap,
                Some(unsafe { anchored(heap, name) }),
                formal_parameter_count,
                kind,
                strict,
            );
        });

        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateClosure, &[0]);
        emit(&mut program, Opcode::Return, &[]);
        let w1 = word(&*thread.heap(), info);
        let info_word = w1;
        run_program_consts(&mut *thread, program, 0, &[], &[info_word]).unwrap()
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
            .map(|i| {
                let h = thread.intern(&scope, &format!("slot{i}"));
                let heap = &*thread.heap();
                h.as_tagged(heap).erase()
            })
            .collect();
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &dummy), &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&program, &scope);
        let w2 = word(&*thread.heap(), scope_info);
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w2]), &scope);
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
fn call_runtime_passes_receiver_and_args() {
    fn add(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
        let (a, b) = {
            let heap = &*nctx.heap();
            match (args.get(heap, 1), args.get(heap, 2)) {
                (Some(a), Some(b)) => (a.erase(), b.erase()),
                _ => return Err(VmError::Arity),
            }
        };
        let (a, b) = (
            Smi::decode(a).ok_or(VmError::Type)?,
            Smi::decode(b).ok_or(VmError::Type)?,
        );
        Ok(Smi::new(a.value() + b.value()).encode())
    }

    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let add = vm.register_native(add);
    let mut thread = vm.attach();

    // r0 = receiver, r1 = 6, r2 = 7; CallRuntime add, r0, 3
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[6]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[7]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(&mut program, Opcode::CallRuntime, &[add.0 as u32, 0, 3]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 13);
}

#[test]
fn failed_run_does_not_leak_frames_into_next_run() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // Add on an object operand needs ToPrimitive (not implemented yet) and
    // throws a TypeError, aborting the run with a frame still suspended.
    let obj = thread.handle_scope(|thread, scope| {
        let known = thread.heap().known();
        let obj = thread
            .heap()
            .new_object(&scope, known.object_initial_map, GcSlice::EMPTY)
            .into_handle(&scope);
        word(&*thread.heap(), obj)
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]); // param 0 (receiver)
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 2, &[smi(42)]);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn wide_parameter_operand_uses_two_bytes() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        // callee: returns smi 99
        let mut callee_program = Vec::new();
        emit(&mut callee_program, Opcode::LoadSmi, &[99]);
        emit(&mut callee_program, Opcode::Return, &[]);
        let callee_bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&callee_program, &scope);
        let callee_constants = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
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
        let w3 = word(&*thread.heap(), callee_obj);
        let receiver_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w3]), &scope);
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    thread.heap().no_gc(|heap| {
        let a = unsafe { anchored(heap, array) }
            .get_as::<Object>()
            .expect("array literal result");
        let a = a.as_ref();
        assert!(a.is_array(heap));
        assert_eq!(a.length(), 3);
        let elements = a.elements_array(heap).expect("array elements");
        assert_eq!(
            Smi::decode(elements.at(heap, 0).erase()).unwrap().value(),
            1
        );
        assert_eq!(
            Smi::decode(elements.at(heap, 1).erase()).unwrap().value(),
            2
        );
        assert_eq!(
            Smi::decode(elements.at(heap, 2).erase()).unwrap().value(),
            3
        );
    });
}

#[test]
fn create_empty_array_literal_starts_empty() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyArrayLiteral, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 0, &[]);
    let array = result.unwrap();
    thread.heap().no_gc(|heap| {
        let a = unsafe { anchored(heap, array) }
            .get_as::<Object>()
            .expect("array literal result");
        let a = a.as_ref();
        assert!(a.is_array(heap));
        assert_eq!(a.length(), 0);
        let elements = a.elements_array(heap).expect("array elements");
        assert_eq!(elements.len(), 0);
    });
}

#[test]
fn array_literal_with_holes_keeps_length() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    thread.heap().no_gc(|heap| {
        let a = unsafe { anchored(heap, array) }
            .get_as::<Object>()
            .expect("array literal result");
        let a = a.as_ref();
        assert_eq!(a.length(), 3);
        let elements = a.elements_array(heap).expect("array elements");
        assert_eq!(
            Smi::decode(elements.at(heap, 0).erase()).unwrap().value(),
            1
        );
        assert_eq!(
            elements.at(heap, 1).erase(),
            heap.known().the_hole.as_tagged(heap).erase(),
            "elided index stays a hole"
        );
        assert_eq!(
            Smi::decode(elements.at(heap, 2).erase()).unwrap().value(),
            2
        );
    });
}

#[test]
fn object_literal_built_with_manual_stores() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let build = |thread: &mut Thread| -> Value {
        thread.handle_scope(|thread, scope| {
            let x = intern_word(&mut *thread, &scope, "x");
            let y = intern_word(&mut *thread, &scope, "y");
            let consts = thread
                .heap()
                .allocate_handle::<FixedArray>(stage_values(&scope, &[x, y]), &scope);

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
    let ((x1, y1, map1), (x2, y2, map2), initial) = thread.heap().no_gc(|heap| {
        let read = |heap: &Heap, obj: Value| {
            let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
            // Safety: `obj` is a strong, live reference to the object
            // literal, and no collection can happen inside the no-GC scope.
            let o = unsafe { ptr.cast::<Object>().as_ref() };
            let slots = o.slots.get(heap).as_ptr().unwrap();
            // Safety: anchored slot read under `heap`.
            let slots = unsafe { slots.as_ref() };
            (
                Smi::decode(slots.at(heap, 0).erase()).unwrap().value(),
                Smi::decode(slots.at(heap, 1).erase()).unwrap().value(),
                o.header.map.get(heap).erase(),
            )
        };
        (
            read(heap, obj1),
            read(heap, obj2),
            heap.known().object_initial_map.as_tagged(heap).erase(),
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; define m = 7 {writable, non-enum, configurable};
    // re-define m = 8 with the same attributes; return r0
    // (runtime(obj, key, value, flags) window: r1..r4)
    let build = |thread: &mut Thread| -> Value {
        thread.handle_scope(|thread, scope| {
            let m = intern_word(&mut *thread, &scope, "m");
            let consts = thread
                .heap()
                .allocate_handle::<FixedArray>(stage_values(&scope, &[m]), &scope);
            let define = |program: &mut Vec<u8>, value: i32| {
                emit(program, Opcode::Load, &[0]);
                emit(program, Opcode::Store, &[1]);
                emit(program, Opcode::LoadConstant, &[0]);
                emit(program, Opcode::Store, &[2]);
                emit(program, Opcode::LoadSmi, &[value as u32]);
                emit(program, Opcode::Store, &[3]);
                emit(program, Opcode::LoadSmi, &[PropertyFlags::DontEnum.bits()]);
                emit(program, Opcode::Store, &[4]);
                emit(
                    program,
                    Opcode::CallRuntime,
                    &[bytecode::RuntimeFn::DefineOwnProperty as u32, 1, 4],
                );
            };
            let mut program = Vec::new();
            emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
            emit(&mut program, Opcode::Store, &[0]);
            define(&mut program, 7);
            define(&mut program, 8);
            emit(&mut program, Opcode::Load, &[0]);
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
            thread.execute(callable, &[]).unwrap()
        })
    };

    let obj1 = build(&mut thread);
    let obj2 = build(&mut thread);
    thread.handle_scope(|thread, scope| {
        let m = intern_word(&mut *thread, &scope, "m");
        thread.heap().no_gc(|heap| {
            let read = |heap: &Heap, obj: Value| {
                let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
                // Safety: `obj` is a strong, live reference and no collection
                // can happen inside the no-GC scope.
                let o = unsafe { ptr.cast::<Object>().as_ref() };
                match o.lookup(heap, SlotName::from_value(m)) {
                    Lookup::Data { slot, flags, .. } => {
                        assert_eq!(
                            slot.get(heap).erase(),
                            smi(8),
                            "re-define updates the value"
                        );
                        assert_eq!(
                            flags,
                            SlotFlags::VALUE
                                .union(SlotFlags::WRITABLE)
                                .union(SlotFlags::CONFIGURABLE),
                            "method attributes {{w, e-, c}}"
                        );
                        o.header.map.get(heap).erase()
                    }
                    _ => panic!("m must be a data property"),
                }
            };
            let map1 = read(heap, obj1);
            let map2 = read(heap, obj2);
            // identically-built defines share one transition map
            assert_eq!(map1, map2);
        });
    });
}

#[test]
fn define_named_own_property_conflicting_redefine_throws() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; define p = 1 {w-, e-, c-}; re-define p = 2 {w, e-, c}:
    // non-configurable with differing attributes rejects the define
    // (runtime(obj, key, value, flags) window: r1..r4)
    let p = thread.handle_scope(|thread, scope| intern_word(thread, &scope, "p"));
    let define = |program: &mut Vec<u8>, value: i32, flags: u32| {
        emit(program, Opcode::Load, &[0]);
        emit(program, Opcode::Store, &[1]);
        emit(program, Opcode::LoadConstant, &[0]);
        emit(program, Opcode::Store, &[2]);
        emit(program, Opcode::LoadSmi, &[value as u32]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[flags]);
        emit(program, Opcode::Store, &[4]);
        emit(
            program,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::DefineOwnProperty as u32, 1, 4],
        );
    };
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut program, Opcode::Store, &[0]);
    define(
        &mut program,
        1,
        PropertyFlags::ReadOnly | PropertyFlags::DontEnum | PropertyFlags::DontDelete,
    );
    define(&mut program, 2, PropertyFlags::DontEnum.bits());
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program_consts(&mut thread, program, 5, &[], &[p]);
    expect_escaped(&mut thread, result, "TypeError");
}

#[test]
fn define_keyed_own_property_string_and_smi_keys() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r1 = "x"; define r0[r1] = 5 {e-};
    // r1 = 3 (smi key); define r0[r1] = 6 {e-}; return r0
    // (runtime(obj, key, value, flags) window: r2..r5)
    let x = thread.handle_scope(|thread, scope| intern_word(thread, &scope, "x"));
    let define = |program: &mut Vec<u8>, value: i32| {
        emit(program, Opcode::Load, &[0]);
        emit(program, Opcode::Store, &[2]);
        emit(program, Opcode::Load, &[1]);
        emit(program, Opcode::Store, &[3]);
        emit(program, Opcode::LoadSmi, &[value as u32]);
        emit(program, Opcode::Store, &[4]);
        emit(program, Opcode::LoadSmi, &[PropertyFlags::DontEnum.bits()]);
        emit(program, Opcode::Store, &[5]);
        emit(
            program,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::DefineOwnProperty as u32, 2, 4],
        );
    };
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadConstant, &[0]);
    emit(&mut program, Opcode::Store, &[1]);
    define(&mut program, 5);
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::Store, &[1]);
    define(&mut program, 6);
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let obj = run_program_consts(&mut thread, program, 6, &[], &[x]).unwrap();
    thread.heap().no_gc(|heap| {
        let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
        // Safety: `obj` is a strong, live reference and no collection can
        // happen inside the no-GC scope.
        let o = unsafe { ptr.cast::<Object>().as_ref() };
        let expected_flags = SlotFlags::VALUE
            .union(SlotFlags::WRITABLE)
            .union(SlotFlags::CONFIGURABLE);
        match o.lookup(heap, SlotName::from_value(x)) {
            Lookup::Data { slot, flags, .. } => {
                assert_eq!(slot.get(heap).erase(), smi(5));
                assert_eq!(flags, expected_flags);
            }
            _ => panic!("x must be a data property"),
        }
        // a smi key defines a plain named property, not an element
        match o.lookup(heap, SlotName::from_value(smi(3))) {
            Lookup::Data { slot, flags, .. } => {
                assert_eq!(slot.get(heap).erase(), smi(6));
                assert_eq!(flags, expected_flags);
            }
            _ => panic!("the smi key must be a named data property"),
        }
    });
}

#[test]
fn define_own_property_accessor_invokes_getter() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let constants = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
        let info = thread.heap().allocate_handle::<CallableInfoObject>(
            CallableInfoInit {
                bytecode,
                constants,
                register_count: 0,
                handlers: None,
            },
            &scope,
        );
        let getter_obj = callable_object(thread, &scope, info);
        let p = intern_word(&mut *thread, &scope, "p");
        let w4 = word(&*thread.heap(), getter_obj);
        (w4, p)
    });

    // r0 = {}; r1 = getter; InstallAccessor(r0, "p", getter) installs the
    // getter half {e-, c}; then either load r0.p (invokes the getter) or
    // return r0 to inspect the installed descriptor
    // (runtime(target, key, closure, flags) window: r2..r5)
    let accessor_program = |load: bool| -> Vec<u8> {
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::LoadConstant, &[2]);
        emit(&mut program, Opcode::Store, &[3]);
        emit(&mut program, Opcode::Load, &[1]);
        emit(&mut program, Opcode::Store, &[4]);
        // flags: getter half (bit 0) + non-enumerable
        emit(
            &mut program,
            Opcode::LoadSmi,
            &[(PropertyFlags::DontEnum.bits() | 1)],
        );
        emit(&mut program, Opcode::Store, &[5]);
        emit(
            &mut program,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::InstallAccessor as u32, 2, 4],
        );
        if load {
            emit(&mut program, Opcode::LoadNamedProperty, &[0, 2, 0]);
        } else {
            emit(&mut program, Opcode::Load, &[0]);
        }
        emit(&mut program, Opcode::Return, &[]);
        program
    };

    let undefined = global_word(&mut thread, |k| k.undefined);
    let constants = [getter, undefined, p];
    let result = run_program_consts(&mut thread, accessor_program(true), 6, &[], &constants);
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);

    let obj = run_program_consts(&mut thread, accessor_program(false), 6, &[], &constants).unwrap();
    thread.heap().no_gc(|heap| {
        let ptr = HeapPtr::decode_strong(obj).expect("object literal result");
        // Safety: `obj` is a strong, live reference and no collection can
        // happen inside the no-GC scope.
        let o = unsafe { ptr.cast::<Object>().as_ref() };
        match o.lookup(heap, SlotName::from_value(p)) {
            Lookup::Accessor { pair, .. } => {
                assert_eq!(pair.get.get(heap).erase(), getter);
            }
            _ => panic!("p must be an accessor property"),
        }
        let flags = o
            .map_ref(heap)
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        assert_eq!(result.unwrap(), global_word(&mut thread, |k| k.undefined));
    }
}

#[test]
fn keyed_store_grows_array_and_fills_holes() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    assert_eq!(result.unwrap(), global_word(&mut thread, |k| k.undefined));
}

#[test]
fn keyed_store_creates_numeric_property_on_plain_object() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let x = intern_word(&mut *thread, &scope, "x");
        let y = intern_word(&mut *thread, &scope, "y");
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[x, y]), &scope);

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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let x = intern_word(&mut *thread, &scope, "x");
        let z = intern_word(&mut *thread, &scope, "z");
        let w = intern_word(&mut *thread, &scope, "w");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind,
                value_slot_count: 1,
                descriptors: &[(SlotName::from_value(x), x_flags, scope.handle(Smi::new(0)))],
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
                    values: stage_values(&scope, &[smi(7)]),
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let w5 = word(&*thread.heap(), obj);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w5, x, z, w]), &scope);

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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let p_word = intern_word(&mut *thread, &scope, "p");
        let parent_map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT,
                value_slot_count: 1,
                descriptors: &[(
                    SlotName::from_value(p_word),
                    WRITABLE_VALUE,
                    scope.handle(Smi::new(0)),
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
                    values: stage_values(&scope, &[Smi::new(1).encode()]),
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        // child: no own slots, prototype = FixedArray([parent]) (multiple
        // parents in priority order; here a single one)
        let w6 = word(&*thread.heap(), parent);
        let parents = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w6]), &scope);
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
                    values: GcSlice::EMPTY,
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let w7 = word(&*thread.heap(), child);
        let w8 = word(&*thread.heap(), parent);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w7, p_word, w8]), &scope);

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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedPropertyNoShadow);
    // child.p = 2 (inherited, parent now 2) + parent.p = 2
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 4);
}

#[test]
fn shadow_store_creates_own_slot_and_leaves_parent() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = parent_object_program(&mut thread, Opcode::StoreNamedProperty);
    // child.p = 2 (new own slot) + parent.p = 1 (untouched)
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 3);
}

#[test]
fn fallthrough_return_is_undefined() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 0, &[]).unwrap();
    assert_eq!(result, global_word(&mut thread, |k| k.undefined));
}

#[test]
fn jump_skips_instructions() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let empty = intern_word(&mut *thread, &scope, "");
        let (undefined, null, true_v, false_v, hole) = {
            let heap = thread.heap();
            let k = heap.known();
            (
                k.undefined.as_tagged(heap).erase(),
                k.null.as_tagged(heap).erase(),
                k.true_object.as_tagged(heap).erase(),
                k.false_object.as_tagged(heap).erase(),
                k.the_hole.as_tagged(heap).erase(),
            )
        };
        let hello = intern_word(&mut *thread, &scope, "hello");
        let zero = {
            let h = thread.heap().allocate_handle::<Float>(0.0, &scope);
            word(&*thread.heap(), h)
        };
        let neg_zero = {
            let h = thread.heap().allocate_handle::<Float>(-0.0, &scope);
            word(&*thread.heap(), h)
        };
        let nan = {
            let h = thread.heap().allocate_handle::<Float>(f64::NAN, &scope);
            word(&*thread.heap(), h)
        };
        let one_half = {
            let h = thread.heap().allocate_handle::<Float>(1.5, &scope);
            word(&*thread.heap(), h)
        };
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
            let obj = thread
                .heap()
                .allocate_object(
                    &scope,
                    ObjectSlotsInit {
                        map,
                        values: GcSlice::EMPTY,
                        elements: the_hole.erase(),
                        length: 0,
                    },
                )
                .into_handle(&scope);

            word(&*thread.heap(), obj)
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = param0; TestReferenceEqual param1; Return
    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]);
    emit(&mut program, Opcode::TestReferenceEqual, &[(-2i32) as u32]);
    emit(&mut program, Opcode::Return, &[]);

    let (true_v, false_v, undefined, null) = {
        let heap = thread.heap();
        let k = heap.known();
        (
            k.true_object.as_tagged(heap).erase(),
            k.false_object.as_tagged(heap).erase(),
            k.undefined.as_tagged(heap).erase(),
            k.null.as_tagged(heap).erase(),
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
        let undefined_word = global_word(&mut *thread, |k| k.undefined);
        let x_word = intern_word(&mut *thread, &scope, "x");
        let y_word = intern_word(&mut *thread, &scope, "y");
        let z_word = intern_word(&mut *thread, &scope, "z");

        // getter/setter get constants ["y"] so they can reach the backing slot
        let make = |thread: &mut Thread, program: &[u8]| -> Value {
            let bytecode = thread
                .heap()
                .allocate_handle::<FixedByteArray>(program, &scope);
            let constants = thread
                .heap()
                .allocate_handle::<FixedArray>(stage_values(&scope, &[y_word]), &scope);
            let info = thread.heap().allocate_handle::<CallableInfoObject>(
                CallableInfoInit {
                    bytecode,
                    constants,
                    register_count: 1,
                    handlers: None,
                },
                &scope,
            );
            let f = callable_object(thread, &scope, info);
            word(&*thread.heap(), f)
        };
        let get = scope.handle(unsafe {
            Tagged::from_value_unchecked(getter.map_or(undefined_word, |p| make(&mut *thread, p)))
        });
        let set = scope.handle(unsafe {
            Tagged::from_value_unchecked(setter.map_or(undefined_word, |p| make(&mut *thread, p)))
        });
        let pair = thread
            .heap()
            .allocate_handle::<AccessorPair>((get, set), &scope);

        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT,
                value_slot_count: 1,
                descriptors: &[
                    (
                        SlotName::from_value(y_word),
                        WRITABLE_VALUE,
                        scope.handle(Smi::new(0)),
                    ),
                    (
                        SlotName::from_value(x_word),
                        SlotFlags::ACCESSOR,
                        pair.erase(),
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
                    values: stage_values(&scope, &[smi(7)]),
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let w10 = word(&*thread.heap(), obj);
        let consts = thread.heap().allocate_handle::<FixedArray>(
            stage_values(&scope, &[w10, x_word, y_word, z_word]),
            &scope,
        );

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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // acc = r2.x (calls the getter, which reads this.y)
    let result = accessor_object_program(&mut thread, Some(&getter_program()), None, |program| {
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn named_store_calls_setter_with_receiver_and_value() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let undefined = global_word(&mut thread, |k| k.undefined);
    let result = accessor_object_program(&mut thread, None, None, |program| {
        emit(program, Opcode::LoadNamedProperty, &[2, 1, 0]);
    });
    assert_eq!(result.unwrap(), undefined);
}

#[test]
fn named_store_without_setter_is_ignored() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let undefined = global_word(&mut thread, |k| k.undefined);
    // r2.z does not exist on the map
    let result = accessor_object_program(&mut thread, None, None, |program| {
        emit(program, Opcode::LoadNamedProperty, &[2, 3, 0]);
    });
    assert_eq!(result.unwrap(), undefined);
}

#[test]
fn store_new_accessor_property_defines_own_accessor() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let the_hole = thread.heap().known().the_hole;

        // object { y: 7 } on an extendable map
        let y_word = intern_word(&mut *thread, &scope, "y");
        let x_word = intern_word(&mut *thread, &scope, "x");
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: EXTENDABLE,
                value_slot_count: 1,
                descriptors: &[(
                    SlotName::from_value(y_word),
                    WRITABLE_VALUE,
                    scope.handle(Smi::new(0)),
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
                    values: stage_values(&scope, &[smi(7)]),
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
                .allocate_handle::<FixedArray>(stage_values(&scope, &[y_word]), &scope);
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
        let name = {
            let heap = &*thread.heap();
            scope.handle(unsafe { SlotName::from_value(x_word).tagged(heap) })
        };
        let w11 = word(&*thread.heap(), getter);
        let get_word = w11;
        let set_word = global_word(&mut *thread, |k| k.undefined);
        Object::define_own_property(
            thread.heap(),
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: get_word,
                set: set_word,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();

        // program: acc = param0.x
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[x_word]), &scope);
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
        let w12 = word(&*thread.heap(), obj);
        thread.execute(callable, &[w12])
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
                values: stage_values(scope, &[Smi::new(idx.0 as i64).encode()]),
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
    let constants = nctx
        .heap()
        .allocate_handle::<FixedArray>(stage_values(scope, constants), scope);
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
    let values = {
        let heap = &*nctx.heap();
        stage_values(
            scope,
            &[word(heap, info), word(heap, empty_context.erase())],
        )
    };
    nctx.heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values,
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
    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let idx = vm.register_native(forty_two);
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let f = native_function(thread, &scope, idx);
        let w13 = word(&*thread.heap(), f);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w13]), &scope);

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

        let exception_word = {
            let heap = &*nctx.heap();
            heap.known().exception.as_tagged(heap).erase()
        };
        match nctx.call(
            unsafe { Tagged::from_value_unchecked(caller) },
            GcSlice::EMPTY,
        ) {
            Ok(exc) if exc == exception_word => {
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
    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let idx = vm.register_native(run_failing_inner);
    let mut thread = vm.attach();

    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[0]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::CallRuntime, &[idx.0 as u32, 0, 1]);
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // Smi::MAX - 1 + 5 no longer fits an smi: the double path rounds it to 2^62
    let result = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[smi(Smi::MAX - 1), smi(5)],
    );
    let result = result.unwrap();
    let value = thread.heap().no_gc(|heap| {
        unsafe { anchored(heap, result) }
            .get_as::<Float>()
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
    thread.heap().no_gc(|heap| {
        unsafe { anchored(heap, v) }
            .get_as::<Float>()
            .expect("expected float result")
            .value
            .get()
    })
}

#[test]
fn equal_strict_compares_numbers_strings_and_objects() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let true_v = global_word(&mut *thread, |k| k.true_object);
        let false_v = global_word(&mut *thread, |k| k.false_object);
        let nan1 = {
            let h = thread.heap().allocate_handle::<Float>(f64::NAN, &scope);
            word(&*thread.heap(), h)
        };
        let nan2 = {
            let h = thread.heap().allocate_handle::<Float>(f64::NAN, &scope);
            word(&*thread.heap(), h)
        };
        let one_float = {
            let h = thread.heap().allocate_handle::<Float>(1.0, &scope);
            word(&*thread.heap(), h)
        };
        let mk_string = |thread: &mut Thread, scope: &HandleScope<'_>, s: &str| {
            let h = DenseString::from_utf8(thread.heap(), scope, s);
            let heap = &*thread.heap();
            h.as_tagged(heap).erase()
        };
        let ab1 = mk_string(thread, &scope, "ab");
        let ab2 = mk_string(thread, &scope, "ab");
        let ac = mk_string(thread, &scope, "ac");
        let object_init_map = thread.heap().known().object_initial_map;
        let obj = {
            let h = thread
                .heap()
                .new_object(&scope, object_init_map, GcSlice::EMPTY)
                .into_handle(&scope);
            word(&*thread.heap(), h)
        };
        let obj2 = {
            let h = thread
                .heap()
                .new_object(&scope, object_init_map, GcSlice::EMPTY)
                .into_handle(&scope);
            word(&*thread.heap(), h)
        };

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
        let s1 = intern_word(&mut *thread, &scope, "1");
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let true_v = global_word(&mut thread, |k| k.true_object);
    let false_v = global_word(&mut thread, |k| k.false_object);
    let null = global_word(&mut thread, |k| k.null);
    let undefined = global_word(&mut thread, |k| k.undefined);
    let false_obj = thread.heap().known().false_object;
    let true_obj = thread.heap().known().true_object;

    let false_word = global_word(&mut thread, |_| false_obj);
    let true_word = global_word(&mut thread, |_| true_obj);
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
            &[false_word, smi(0)]
        )
        .unwrap(),
        true_v
    );
    assert_eq!(
        run_program(
            &mut thread,
            binary_op_program(Opcode::Equal),
            0,
            &[true_word, smi(1)]
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
    let empty_string = thread.handle_scope(|thread, scope| intern_word(thread, &scope, ""));
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
        let s1 = intern_word(&mut *thread, &scope, "1");
        let s15 = intern_word(&mut *thread, &scope, "1.5");
        let f15 = {
            let h = thread.heap().allocate_handle::<Float>(1.5, &scope);
            word(&*thread.heap(), h)
        };
        let nan = {
            let h = thread.heap().allocate_handle::<Float>(f64::NAN, &scope);
            word(&*thread.heap(), h)
        };
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let true_v = global_word(&mut thread, |k| k.true_object);
    let false_v = global_word(&mut thread, |k| k.false_object);

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
        let true_v = global_word(&mut *thread, |k| k.true_object);
        let false_v = global_word(&mut *thread, |k| k.false_object);
        let nan = {
            let h = thread.heap().allocate_handle::<Float>(f64::NAN, &scope);
            word(&*thread.heap(), h)
        };
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
        let abc = intern_word(&mut *thread, &scope, "abc");
        let abd = intern_word(&mut *thread, &scope, "abd");
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
        let s2 = intern_word(&mut *thread, &scope, "2");
        let s10 = intern_word(&mut *thread, &scope, "10");
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let f55 = {
            let h = thread.heap().allocate_handle::<Float>(5.5, &scope);
            word(&*thread.heap(), h)
        };
        run_binary_consts(thread, Opcode::Mod, f55, smi(2)).unwrap()
    });
    assert_eq!(float_value(&mut thread, r), 1.5);
}

#[test]
fn exp_produces_floats() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let half = {
            let h = thread.heap().allocate_handle::<Float>(0.5, &scope);
            word(&*thread.heap(), h)
        };
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let known = thread.heap().known();
    let null_word = global_word(&mut thread, |_| known.null);
    let undefined_word = global_word(&mut thread, |_| known.undefined);
    let true_word = global_word(&mut thread, |_| known.true_object);
    let false_word = global_word(&mut thread, |_| known.false_object);

    // null + 1 = 1
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[null_word, smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 1);

    // undefined + 1 = NaN
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[undefined_word, smi(1)],
    );
    assert!(float_value(&mut thread, r.unwrap()).is_nan());

    // true + 1 = 2, false + 1 = 1
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[true_word, smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 2);
    let r = run_program(
        &mut thread,
        binary_op_program(Opcode::Add),
        0,
        &[false_word, smi(1)],
    );
    assert_eq!(Smi::decode(r.unwrap()).unwrap().value(), 1);

    // "2" * 3 = 6 (strings parse in numeric contexts)
    let r = thread.handle_scope(|thread, scope| {
        let s2 = intern_word(&mut *thread, &scope, "2");
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
            .allocate_handle::<FixedArray>(stage_values(&scope, constants), &scope);
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (_, result) = thread.handle_scope(|thread, scope| {
        let x = intern_word(&mut *thread, &scope, "x");

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
        let w14 = word(&*thread.heap(), x.erase());
        let x_word = w14;
        run_program_consts(&mut *thread, program, 0, &[], &[x_word])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn load_global_missing_name_throws_reference_error() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // unresolvable references throw ReferenceError (GetValue on an
    // unresolvable reference); typeof uses LoadGlobalNoThrow instead
    let result = thread.handle_scope(|thread, scope| {
        let missing = thread.intern(&scope, "not_defined_anywhere");
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadGlobal, &[0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let w15 = word(&*thread.heap(), missing);
        run_program_consts(&mut *thread, program, 0, &[], &[w15])
    });
    expect_escaped(&mut thread, result, "ReferenceError");

    // the no-throw variant yields undefined
    let result = thread.handle_scope(|thread, scope| {
        let missing = thread.intern(&scope, "not_defined_anywhere");
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadGlobalNoThrow, &[0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let w16 = word(&*thread.heap(), missing);
        run_program_consts(&mut *thread, program, 0, &[], &[w16])
    });
    assert_eq!(result.unwrap(), global_word(&mut thread, |k| k.undefined));
}

#[test]
fn empty_object_literal_inherits_from_object_prototype() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let p = intern_word(&mut *thread, &scope, "p");
        let name = SlotName::from_value(p);
        let proto = global_word(&mut *thread, |k| k.object_prototype);

        // host-side: %Object.prototype%.p = 1
        let outcome = thread
            .heap()
            .no_gc(|heap| {
                unsafe { anchored(heap, proto) }.store_lookup(
                    heap,
                    name,
                    Smi::new(1).into_tagged(),
                    StoreSemantics::Shadow,
                )
            })
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
        let first = run_program_consts(&mut *thread, program, 1, &[], &[p]);
        assert_eq!(Smi::decode(first.unwrap()).unwrap().value(), 1);

        // ({}.p = 2) shadows: own property on the instance...
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadSmi, &[2]);
        emit(&mut program, Opcode::StoreNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        let second = run_program_consts(&mut *thread, program, 1, &[], &[p]);
        assert_eq!(Smi::decode(second.unwrap()).unwrap().value(), 2);

        // ...and a fresh {} still sees the prototype value
        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
        emit(&mut program, Opcode::Return, &[]);
        run_program_consts(&mut *thread, program, 1, &[], &[p])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 1);
}

#[test]
fn create_closure_inherits_current_context_and_is_callable() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let callee_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
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
        let slot0 = intern_word(&mut *thread, &scope, "slot0");
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[slot0]), &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let w17 = word(&*thread.heap(), scope_info);
        let w18 = word(&*thread.heap(), callee_info);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w17, w18]), &scope);
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
            .allocate_handle::<FixedArray>(stage_values(&scope, &[smi(42)]), &scope);
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
        let values = {
            let heap = &*thread.heap();
            stage_values(
                &scope,
                &[word(heap, caller_info), word(heap, context.erase())],
            )
        };
        let caller = thread
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
        thread.execute(caller, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 42);
}

#[test]
fn create_closure_shares_callable_info_template() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let (result, template) = thread.handle_scope(|thread, scope| {
        let callee_bytecode = thread.heap().allocate_handle::<FixedByteArray>(&[], &scope);
        let callee_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
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
        let w19 = word(&*thread.heap(), callee_info);
        let result = run_program_consts(&mut *thread, program, 0, &[], &[w19]);
        let w20 = word(&*thread.heap(), callee_info);
        (result.unwrap(), w20)
    });

    thread.heap().no_gc(|heap| {
        let Some(o) = unsafe { anchored(heap, result) }.as_heap_object() else {
            panic!("closure must be an object");
        };
        let info = o
            .as_ref()
            .callable_info(heap)
            .expect("closure carries a callable info");
        // the info is shared, not copied per closure
        assert_eq!(info.into_tagged().erase(), template);
        // the closure's context slot is the caller's (empty) context
        let context = o
            .as_ref()
            .closure_context(heap)
            .expect("closure carries a context");
        assert_eq!(
            context.into_tagged().erase(),
            heap.known().empty_context.as_tagged(heap).erase()
        );
    });
}

#[test]
fn create_closure_function_kind_controls_call_and_construct() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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

    let prototype = thread.handle_scope(|thread, scope| intern_word(thread, &scope, "prototype"));
    thread.heap().no_gc(|heap| {
        let Some(method) = unsafe { anchored(heap, method) }.as_heap_object() else {
            panic!("method must be an object")
        };
        let method = method.as_ref();
        assert!(method.map_ref(heap).kind().is_callable());
        assert!(!method.map_ref(heap).kind().is_constructor());
        assert!(matches!(
            method.lookup(heap, SlotName::from_value(prototype)),
            Lookup::NotFound
        ));
    });

    let undefined = global_word(&mut thread, |k| k.undefined);
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
    thread.heap().no_gc(|heap| {
        let Some(constructor) = unsafe { anchored(heap, class_constructor) }.as_heap_object()
        else {
            panic!("class constructor must be an object")
        };
        let kind = constructor.as_ref().map_ref(heap).kind();
        assert!(kind.is_callable());
        assert!(kind.is_constructor());
        assert!(kind.is_class_constructor());
        assert!(matches!(
            constructor
                .as_ref()
                .lookup(heap, SlotName::from_value(prototype)),
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // PushContext must save the old frame context (empty_context) into r0
    let result = thread.handle_scope(|thread, scope| {
        let empty = global_word(&mut *thread, |k| k.empty_context);
        let slot0 = intern_word(&mut *thread, &scope, "slot0");
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[slot0]), &scope);
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
        let w21 = word(&*thread.heap(), scope_info);
        run_program_consts(&mut *thread, program, 2, &[], &[w21, empty])
    });
    assert_eq!(result.unwrap(), global_word(&mut thread, |k| k.true_object));
}

#[test]
fn pop_context_restores_previous_context() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
fn tdz_hole_read_throws_reference_error() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        // callee (arrow): return x from its (inherited) context
        let mut callee_program = Vec::new();
        emit(&mut callee_program, Opcode::LoadContextSlot, &[0, 0]);
        emit(&mut callee_program, Opcode::Return, &[]);
        let callee_bytecode = thread
            .heap()
            .allocate_handle::<FixedByteArray>(&callee_program, &scope);
        let callee_consts = thread
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[]), &scope);
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
        let slot0 = intern_word(&mut *thread, &scope, "slot0");
        let names = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[slot0]), &scope);
        let scope_info = thread
            .heap()
            .allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, &scope);
        let w22 = word(&*thread.heap(), scope_info);
        let w23 = word(&*thread.heap(), callee_info);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w22, w23]), &scope);
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
    p: vm::Handle<'s, vm::DenseString>,
) -> vm::Handle<'s, Object> {
    let the_hole = thread.heap().known().the_hole;
    let p_name = SlotName::from(p.as_tagged(&*thread.heap()));
    let map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: EXTENDABLE,
            value_slot_count: 1,
            descriptors: &[(p_name, WRITABLE_VALUE, scope.handle(Smi::new(0)))],
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
                values: stage_values(scope, &[smi(7)]),
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .into_handle(scope)
}

#[test]
fn set_prototype_changes_property_lookup_chain() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r0.[[Prototype]] = objB (p = 7); return r0.p
    let result = thread.handle_scope(|thread, scope| {
        let p = thread.intern(&scope, "p");
        let obj_b = proto_object(&mut *thread, &scope, p);
        let w24 = word(&*thread.heap(), p.erase());
        let w25 = word(&*thread.heap(), obj_b);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w24, w25]), &scope);

        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::Load, &[1]);
        emit(&mut program, Opcode::Store, &[3]);
        emit(
            &mut program,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::SetPrototype as u32, 2, 2],
        );
        emit(&mut program, Opcode::LoadNamedProperty, &[0, 0, 0]);
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
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
}

#[test]
fn set_prototype_survives_property_transitions() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r0.[[Prototype]] = objB; r0.x = 1 (transition); return r0.p
    let result = thread.handle_scope(|thread, scope| {
        let p = thread.intern(&scope, "p");
        let x = thread.intern(&scope, "x");
        let obj_b = proto_object(&mut *thread, &scope, p);
        let _w28 = word(&*thread.heap(), x.erase());
        let w26 = word(&*thread.heap(), p.erase());
        let w27 = word(&*thread.heap(), obj_b);
        let w28 = word(&*thread.heap(), x.erase());
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w26, w27, w28]), &scope);

        let mut program = Vec::new();
        emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::Load, &[1]);
        emit(&mut program, Opcode::Store, &[3]);
        emit(
            &mut program,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::SetPrototype as u32, 2, 2],
        );
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
                register_count: 4,
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // r0 = {}; r0.[[Prototype]] = r0 (cycle)
    let mut program = Vec::new();
    emit(&mut program, Opcode::CreateEmptyObjectLiteral, &[]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Store, &[2]);
    emit(
        &mut program,
        Opcode::CallRuntime,
        &[bytecode::RuntimeFn::SetPrototype as u32, 1, 2],
    );
    emit(&mut program, Opcode::Return, &[]);

    let result = run_program(&mut thread, program, 3, &[]);
    expect_escaped(&mut thread, result, "TypeError");
}

#[test]
fn set_prototype_on_non_extensible_throws_type_error() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
                    values: GcSlice::EMPTY,
                    elements: the_hole.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let w29 = word(&*thread.heap(), frozen);
        let w30 = word(&*thread.heap(), obj_b);
        let consts = thread
            .heap()
            .allocate_handle::<FixedArray>(stage_values(&scope, &[w29, w30]), &scope);
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::LoadConstant, &[1]);
        emit(&mut program, Opcode::Store, &[1]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::Store, &[2]);
        emit(&mut program, Opcode::Load, &[1]);
        emit(&mut program, Opcode::Store, &[3]);
        emit(
            &mut program,
            Opcode::CallRuntime,
            &[bytecode::RuntimeFn::SetPrototype as u32, 2, 2],
        );
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
    });
    expect_escaped(&mut thread, result, "TypeError");
}

/// Build a callable function object wrapping `program` with `constants`.
fn make_callable(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    program: &[u8],
    constants: &[Value],
) -> Value {
    let bytecode = thread
        .heap()
        .allocate_handle::<FixedByteArray>(program, scope);
    let constants = thread
        .heap()
        .allocate_handle::<FixedArray>(stage_values(scope, constants), scope);
    let info = thread.heap().allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            bytecode,
            constants,
            register_count: 1,
            handlers: None,
        },
        scope,
    );
    let f = callable_object(thread, scope, info);
    word(&*thread.heap(), f)
}

/// Fresh `{}` with the realm's object initial map.
fn empty_object<'s>(thread: &mut Thread, scope: &'s HandleScope<'_>) -> Handle<'s, Object> {
    let known = thread.heap().known();
    thread
        .heap()
        .new_object(scope, known.object_initial_map, GcSlice::EMPTY)
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        let value_of = intern_word(&mut *thread, &scope, "valueOf");
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        let value_of = intern_word(&mut *thread, &scope, "valueOf");
        let to_string = intern_word(&mut *thread, &scope, "toString");
        let x = intern_word(&mut *thread, &scope, "x");

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
        thread.heap().no_gc(|heap| {
            let s = unsafe { anchored(heap, r) }
                .get_as::<DenseString>()
                .expect("concat result must be a string");
            assert_eq!(s.to_rust_string(heap), "x1");
        });
    });
}

#[test]
fn add_concatenates_strings() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        // string + string, string + number, number + string
        let a = intern_word(&mut *thread, &scope, "a");
        let b = intern_word(&mut *thread, &scope, "b");
        for (lhs, rhs, expected) in [
            (a, b, "ab"),
            (a, smi(2), "a2"),
            (smi(2), b, "2b"),
            (a, smi(1000), "a1000"),
        ] {
            let r =
                run_program(&mut *thread, binary_op_program(Opcode::Add), 0, &[lhs, rhs]).unwrap();
            thread.heap().no_gc(|heap| {
                let s = unsafe { anchored(heap, r) }
                    .get_as::<DenseString>()
                    .expect("concat result must be a string");
                assert_eq!(s.to_rust_string(heap), expected, "{lhs:?} + {rhs:?}");
            });
        }
    });
}

#[test]
fn to_primitive_uses_to_primitive_symbol_first() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        // @@toPrimitive = () => 1: wins over valueOf, called with hint "default"
        let sym = global_word(&mut *thread, |k| k.to_primitive_symbol);
        let value_of = intern_word(&mut *thread, &scope, "valueOf");
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let obj = thread.handle_scope(|thread, scope| {
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        let sym = global_word(&mut *thread, |k| k.to_primitive_symbol);
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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
        let (name, get_word, set_word) = {
            let heap = &*thread.heap();
            (
                scope.handle(unsafe { SlotName::from(value_of.as_tagged(heap)).tagged(heap) }),
                getter,
                heap.known().undefined.as_tagged(heap).erase(),
            )
        };
        Object::define_own_property(
            thread.heap(),
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: get_word,
                set: set_word,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();

        let obj_word = word(&*thread.heap(), obj);
        let r = run_program(
            &mut *thread,
            binary_op_program(Opcode::Add),
            0,
            &[obj_word, smi(1)],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 3);
    });
}

#[test]
fn relational_and_equality_operators_coerce_objects() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let true_v = global_word(&mut *thread, |k| k.true_object);
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        let value_of = intern_word(&mut *thread, &scope, "valueOf");
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
        assert_eq!(r, global_word(&mut *thread, |k| k.false_object));
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let obj = thread.handle_scope(|thread, scope| {
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        let value_of = intern_word(&mut *thread, &scope, "valueOf");
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let number = intern_word(&mut *thread, &scope, "number");
        let string = intern_word(&mut *thread, &scope, "string");
        let undefined = intern_word(&mut *thread, &scope, "undefined");
        let object = intern_word(&mut *thread, &scope, "object");
        let boolean = intern_word(&mut *thread, &scope, "boolean");
        let function = intern_word(&mut *thread, &scope, "function");

        let f = make_callable(thread, &scope, &program_return_1(), &[]);
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        let float = {
            let h = thread.heap().allocate_handle::<Float>(1.5, &scope);
            word(&*thread.heap(), h)
        };
        let s = intern_word(&mut *thread, &scope, "x");
        let known = thread.heap().known();

        let cases: &[(Value, Value)] = &[
            (smi(3), number),
            (float, number),
            (s, string),
            (global_word(&mut *thread, |_| known.undefined), undefined),
            (global_word(&mut *thread, |_| known.null), object),
            (global_word(&mut *thread, |_| known.true_object), boolean),
            (global_word(&mut *thread, |_| known.false_object), boolean),
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // smi fast paths
    let r = run_program(&mut thread, unary_program(Opcode::Negate), 0, &[smi(5)]).unwrap();
    assert_eq!(r.to_i64().unwrap(), -5);
    let r = run_program(&mut thread, unary_program(Opcode::Negate), 0, &[smi(-7)]).unwrap();
    assert_eq!(r.to_i64().unwrap(), 7);

    // -0 must be the -0.0 HeapNumber (1 / -0 === -Infinity)
    let r = run_program(&mut thread, unary_program(Opcode::Negate), 0, &[smi(0)]).unwrap();
    let r = thread.heap().no_gc(|heap| {
        unsafe { anchored(heap, r) }
            .get_as::<Float>()
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
        let half = {
            let h = thread.heap().allocate_handle::<Float>(1.5, &scope);
            word(&*thread.heap(), h)
        };
        run_program(&mut *thread, unary_program(Opcode::Negate), 0, &[half]).unwrap()
    });
    assert_eq!(float_value(&mut thread, r), -1.5);
    let r = thread.handle_scope(|thread, scope| {
        let s3 = intern_word(&mut *thread, &scope, "3");
        run_program(&mut *thread, unary_program(Opcode::Negate), 0, &[s3]).unwrap()
    });
    assert_eq!(r.to_i64().unwrap(), -3);
}

#[test]
fn instance_of_walks_prototype_chain() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let known = thread.heap().known();
        let true_v = global_word(&mut *thread, |_| known.true_object);
        let false_v = global_word(&mut *thread, |_| known.false_object);
        let prototype = intern_word(&mut *thread, &scope, "prototype");

        // F with a .prototype object
        let f = make_callable(thread, &scope, &program_return_1(), &[]);
        let f_proto_h = empty_object(thread, &scope);
        let f_proto = word(&*thread.heap(), f_proto_h);
        Object::define_own_property_values(
            thread.heap(),
            &scope,
            f,
            SlotName::from_value(prototype),
            PropertyDescriptor::data(f_proto),
        )
        .expect("defining a fresh own property must succeed");

        // obj inherits F.prototype; plain {} does not
        let obj_h = empty_object(thread, &scope);
        let obj = word(&*thread.heap(), obj_h);
        Object::set_prototype(thread.heap(), &scope, obj, f_proto).unwrap();
        let plain_h = empty_object(thread, &scope);
        let plain = word(&*thread.heap(), plain_h);

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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let known = thread.heap().known();
        let prototype = intern_word(&mut *thread, &scope, "prototype");

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
        let g_proto_h = empty_object(thread, &scope);
        let g_proto = word(&*thread.heap(), g_proto_h);
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
        assert_eq!(r, global_word(&mut *thread, |_| known.true_object));

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
        let plain_h = empty_object(thread, &scope);
        let plain = word(&*thread.heap(), plain_h);
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
    let flag = Smi::new(if nctx.is_construct() { 1 } else { 0 }).into_tagged();
    nctx.handle_scope(|nctx, scope| {
        let name = nctx.intern(&scope, "constructProbe");
        let global = {
            let heap = &*nctx.heap();
            heap.known().global_object.as_tagged(heap).erase()
        };
        let outcome = nctx.heap().no_gc(|heap| {
            unsafe { anchored(heap, global) }.store_lookup(
                heap,
                SlotName::from(name.as_tagged(heap)),
                flag,
                StoreSemantics::WriteThrough,
            )
        })?;
        match outcome {
            StoreOutcome::Done => {}
            StoreOutcome::Transition { .. } => {
                let name_word = {
                    let heap = &*nctx.heap();
                    SlotName::from(name.as_tagged(heap))
                };
                Object::define_own_property_values(
                    nctx.heap(),
                    &scope,
                    global,
                    name_word,
                    PropertyDescriptor::data(flag.erase()),
                )?;
            }
            StoreOutcome::CallSetter { setter } => {
                nctx.handle_scope(|nctx, scope| {
                    nctx.call(
                        unsafe { Tagged::from_value_unchecked(setter) },
                        stage_values(&scope, &[global, flag.erase()]),
                    )
                })?;
            }
        }
        Ok(())
    })?;
    Ok(flag.erase())
}

#[test]
fn construct_sets_native_construct_flag() {
    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let idx = vm.register_native(construct_probe);
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let f_h = native_function(thread, &scope, idx);
        let f = word(&*thread.heap(), f_h);
        let name = intern_word(&mut *thread, &scope, "constructProbe");

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
    let p = thread.intern(scope, "p");
    let p_name = SlotName::from(p.as_tagged(&*thread.heap()));
    let parent_map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT,
            value_slot_count: 1,
            descriptors: &[(p_name, WRITABLE_VALUE, scope.handle(Smi::new(0)))],
            prototype: the_hole.erase(),
        },
        scope,
    );
    let parent = thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map: parent_map,
                values: stage_values(scope, &[Smi::new(1).encode()]),
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .into_handle(scope);
    let w31 = word(&*thread.heap(), parent);
    let parents = thread
        .heap()
        .allocate_handle::<FixedArray>(stage_values(scope, &[w31]), scope);
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
        scope,
    );
    let child = thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map: child_map,
                values: GcSlice::EMPTY,
                elements: the_hole.erase(),
                length: 0,
            },
        )
        .into_handle(scope);
    let p_word = {
        let heap = &*thread.heap();
        p.as_tagged(heap).erase()
    };
    (child, parent, p_word)
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
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let (child, parent, p) = shadow_setup(thread, &scope, false);
        // sloppy [[Set]] on an inherited writable property shadows with an
        // own define; the non-extensible receiver rejects it (false) and the
        let w32 = word(&*thread.heap(), child);
        let w33 = word(&*thread.heap(), parent);
        // store is silently ignored
        let r = run_program_consts(
            &mut *thread,
            shadow_store_program(Opcode::StoreNamedProperty),
            3,
            &[],
            &[w32, p, w33],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 2, "acc keeps the value");

        thread.heap().no_gc(|heap| {
            // no own property appeared on the child, the parent is untouched
            let child_ref = child.heap_ref(heap);
            assert_eq!(child_ref.header.map.heap_ref(heap).descriptor_count(), 0);
            let parent_ref = parent.heap_ref(heap);
            match parent_ref.as_ref().lookup(heap, SlotName::from_value(p)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(Smi::decode(slot.get(heap).erase()).unwrap().value(), 1);
                }
                _ => panic!("parent must keep its writable property"),
            }
        });
    });
}

#[test]
fn shadow_store_defines_default_attributes() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let (child, parent, p) = shadow_setup(thread, &scope, true);
        let w35 = word(&*thread.heap(), parent);
        let w34 = word(&*thread.heap(), child);
        let r = run_program_consts(
            &mut *thread,
            shadow_store_program(Opcode::StoreNamedProperty),
            3,
            &[],
            &[w34, p, w35],
        )
        .unwrap();
        assert_eq!(r.to_i64().unwrap(), 2);

        thread.heap().no_gc(|heap| {
            let child_ref = child.heap_ref(heap);
            let map = child_ref.header.map.heap_ref(heap);
            assert_eq!(map.descriptor_count(), 1);
            let d = map.descriptor(0);
            assert_eq!(d.name(), SlotName::from_value(p));
            assert_eq!(d.offset(), 0);
            // [[Set]] shadowing defines with the assignment defaults
            assert!(d.flags().is_writable());
            assert!(d.flags().is_enumerable());
            assert!(d.flags().is_configurable());
            // the own slot wins, the parent keeps its value
            match child_ref.as_ref().lookup(heap, SlotName::from_value(p)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(Smi::decode(slot.get(heap).erase()).unwrap().value(), 2);
                }
                _ => panic!("expected own data property"),
            }
            let parent_ref = parent.heap_ref(heap);
            match parent_ref.as_ref().lookup(heap, SlotName::from_value(p)) {
                Lookup::Data { slot, .. } => {
                    assert_eq!(Smi::decode(slot.get(heap).erase()).unwrap().value(), 1);
                }
                _ => panic!("parent must keep its writable property"),
            }
        });
    });
}
