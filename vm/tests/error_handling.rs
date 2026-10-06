use bytecode::{Opcode, emit};
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, HandlerEntryInit,
    HandlerTable, HandlerTableInit, ObjectSlotsInit, Smi, Tagged, Value,
};
use vm::{Thread, VM};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

fn callable(
    thread: &mut Thread,
    scope: &vm::HandleScope<'_>,
    program: &[u8],
    constants: &[Tagged<'_, Value>],
    register_count: usize,
    handlers: Option<&[HandlerEntryInit]>,
) -> Value {
    let _the_hole = thread.heap().known().the_hole;
    let empty_context = thread.heap().known().empty_context;
    let bytecode = thread
        .heap()
        .allocate_handle::<FixedByteArray>(program, scope);
    let constants = thread
        .heap()
        .allocate_handle::<FixedArray>(scope.stage(constants), scope);
    let handlers = handlers.map(|entries| {
        thread
            .heap()
            .allocate_handle::<HandlerTable>(HandlerTableInit { entries }, scope)
    });
    let info = thread.heap().allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            expected_slots: 0,
            bytecode,
            constants,
            register_count,
            handlers,
        },
        scope,
    );
    let map = thread.heap().known().function_map;
    let values = {
        let heap = &*thread.heap();
        scope.stage(&[
            info.as_tagged(heap).erase(),
            empty_context.as_tagged(heap).erase(),
        ])
    };
    let empty_fixed_array = thread.heap().known().empty_fixed_array;
    thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values,
                elements: empty_fixed_array,
                length: 0,
            },
        )
        .raw()
}

fn table<'s>(
    thread: &mut Thread,
    scope: &'s vm::HandleScope<'_>,
    entries: &[HandlerEntryInit],
) -> vm::Handle<'s, HandlerTable> {
    thread
        .heap()
        .allocate_handle::<HandlerTable>(HandlerTableInit { entries }, scope)
}

#[test]
fn roundtrip_entries() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(
            thread,
            &scope,
            &[
                HandlerEntryInit::new(0, 10, 40),
                HandlerEntryInit::new(12, 20, 55),
            ],
        );
        let out = {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            (table.len(), table.entry(0), table.entry(1))
        };
        assert_eq!(out.0, 2);
        assert_eq!(out.1, HandlerEntryInit::new(0, 10, 40));
        assert_eq!(out.2, HandlerEntryInit::new(12, 20, 55));
    });
}

#[test]
fn lookup_finds_handler_inside_range() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[HandlerEntryInit::new(5, 15, 100)]);
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            // try_start is inside (inclusive) ...
            assert_eq!(table.as_ref().lookup(5), Some(100));
            // ... as is any offset before try_end
            assert_eq!(table.as_ref().lookup(14), Some(100));
        };
    });
}

#[test]
fn lookup_returns_none_outside_range() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[HandlerEntryInit::new(5, 15, 100)]);
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            // before the region ...
            assert_eq!(table.as_ref().lookup(4), None);
            // ... try_end is exclusive ...
            assert_eq!(table.as_ref().lookup(15), None);
            // ... and beyond it
            assert_eq!(table.as_ref().lookup(16), None);
        };
    });
}

#[test]
fn lookup_returns_innermost_of_nested_ranges() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        // inner entry emitted first: lookup must be independent of
        // emission order (properly-nested try regions)
        let t = table(
            thread,
            &scope,
            &[
                HandlerEntryInit::new(4, 12, 300),
                HandlerEntryInit::new(0, 20, 100),
            ],
        );
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            // inside both ranges: the innermost (largest try_start) wins
            assert_eq!(table.as_ref().lookup(5), Some(300));
            // inside the outer range only
            assert_eq!(table.as_ref().lookup(1), Some(100));
            assert_eq!(table.as_ref().lookup(15), Some(100));
        };
    });
}

#[test]
fn lookup_on_empty_table_returns_none() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let t = table(thread, &scope, &[]);
        {
            let heap = &*thread.heap();
            let table = t.as_tagged(heap);
            assert_eq!(table.len(), 0);
            assert_eq!(table.as_ref().lookup(0), None);
        };
    });
}

#[test]
fn callable_info_carries_handler_table() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let result = thread.handle_scope(|thread, scope| {
        let f = callable(
            thread,
            &scope,
            &[],
            &[],
            0,
            Some(&[HandlerEntryInit::new(2, 8, 33)]),
        );
        let heap = &*thread.heap();
        let o = unsafe { Tagged::<vm::Object>::from_value_unchecked(f) };
        let info = o.as_ref().callable_info(heap).unwrap();
        let table = info.handlers.get(heap).expect("handler table attached");
        table.as_ref().lookup(5)
    });
    assert_eq!(result, Some(33));
}

#[test]
fn callable_info_without_handler_table() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        let f = callable(thread, &scope, &[], &[], 0, None);
        let heap = &*thread.heap();
        let o = unsafe { Tagged::<vm::Object>::from_value_unchecked(f) };
        let info = o.as_ref().callable_info(heap).unwrap();
        assert!(
            info.handlers.get(heap).is_none(),
            "a hole handlers slot must mean no table"
        );
    });
}

#[test]
fn throw_is_caught_in_same_function() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // 0: LoadSmi 99 | 2: Throw | 3: Return | 4: Store r0 | 6: Load r0 | 8: Return
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[99]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);
    emit(&mut program, Opcode::Store, &[0]); // handler: bind exception
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let f = callable(
            thread,
            &scope,
            &program,
            &[],
            1,
            Some(&[HandlerEntryInit::new(0, 3, 4)]),
        );
        let callable = {
            let heap = &*thread.heap();
            scope
                .cast::<vm::Object>(heap, unsafe { Tagged::from_value_unchecked(f) })
                .unwrap()
        };
        thread.execute(callable, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 99);
    // catching consumes the pending exception
    assert!(!thread.has_pending_exception());
}

#[test]
fn throw_any_value_escapes_as_sentinel() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // throw 42: arbitrary values are throwable (ES §14.18), no handlers
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let f = callable(thread, &scope, &program, &[], 0, None);
        let callable = {
            let heap = &*thread.heap();
            scope
                .cast::<vm::Object>(heap, unsafe { Tagged::from_value_unchecked(f) })
                .unwrap()
        };
        thread.execute(callable, &[])
    });
    let exception_word = {
        let heap = thread.heap();
        heap.known().exception.as_tagged(heap).raw()
    };
    assert_eq!(result, Ok(exception_word));
    assert_eq!(
        thread.take_pending_exception(),
        Some(smi(42)),
        "the thrown value itself is the pending exception"
    );
}

#[test]
fn innermost_handler_wins() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // 0: LoadSmi 1 | 2: LoadSmi 2 | 4: Throw | 5: Return
    // 6: LoadSmi 100 | 8: Return      (inner handler)
    // 9: LoadSmi 200 | 11: Return     (outer handler)
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[1]);
    emit(&mut program, Opcode::LoadSmi, &[2]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);
    emit(&mut program, Opcode::LoadSmi, &[100]);
    emit(&mut program, Opcode::Return, &[]);
    emit(&mut program, Opcode::LoadSmi, &[200]);
    emit(&mut program, Opcode::Return, &[]);

    let entries = [
        HandlerEntryInit::new(0, 6, 9), // outer
        HandlerEntryInit::new(4, 5, 6), // inner
    ];
    let result = thread.handle_scope(|thread, scope| {
        let f = callable(thread, &scope, &program, &[], 0, Some(&entries));
        let callable = {
            let heap = &*thread.heap();
            scope
                .cast::<vm::Object>(heap, unsafe { Tagged::from_value_unchecked(f) })
                .unwrap()
        };
        thread.execute(callable, &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 100);
}

#[test]
fn exception_unwinds_to_caller() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    thread.handle_scope(|thread, scope| {
        // callee: throws 7, no handlers
        let mut callee_program = Vec::new();
        emit(&mut callee_program, Opcode::LoadSmi, &[7]);
        emit(&mut callee_program, Opcode::Throw, &[]);
        emit(&mut callee_program, Opcode::Return, &[]);
        let callee = callable(thread, &scope, &callee_program, &[], 0, None);

        // caller: 0: LoadConstant 0 | 2: Store r0 | 4: CallNoFeedback r0 r0 1 |
        //         8: Return | 9: Store r0 | 11: Load r0 | 13: Return
        // the try region covers the call site (pc 4); the handler binds the
        // exception from a suspended frame via the recorded call-site pc
        // (register operands carry the anchor bias: r0 = REGISTER_BASE)
        let r0 = (bytecode::REGISTER_FILE_START - 0) as u32;
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[r0]);
        emit(&mut program, Opcode::CallNoFeedback, &[r0, r0, 1]);
        emit(&mut program, Opcode::Return, &[]);
        emit(&mut program, Opcode::Store, &[r0]);
        emit(&mut program, Opcode::Load, &[r0]);
        emit(&mut program, Opcode::Return, &[]);

        let caller = callable(
            thread,
            &scope,
            &program,
            &[unsafe { Tagged::from_value_unchecked(callee) }],
            1,
            Some(&[HandlerEntryInit::new(0, 9, 9)]),
        );
        let result = {
            let callable = {
                let heap = &*thread.heap();
                scope
                    .cast::<vm::Object>(heap, unsafe { Tagged::from_value_unchecked(caller) })
                    .unwrap()
            };
            thread.execute(callable, &[])
        };
        assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
    });
}

#[test]
fn rethrow_from_finally_escapes_past_its_own_handler() {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    // 0: LoadSmi 3 | 2: Throw | 3: Return | 4: ReThrow (finally handler)
    // the handler pc 4 is outside the try region [0, 4), so the ReThrow
    // finds no handler and escapes the run
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[3]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);
    emit(&mut program, Opcode::ReThrow, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let f = callable(
            thread,
            &scope,
            &program,
            &[],
            0,
            Some(&[HandlerEntryInit::new(0, 4, 4)]),
        );
        let callable = {
            let heap = &*thread.heap();
            scope
                .cast::<vm::Object>(heap, unsafe { Tagged::from_value_unchecked(f) })
                .unwrap()
        };
        thread.execute(callable, &[])
    });
    let exception_word = {
        let heap = thread.heap();
        heap.known().exception.as_tagged(heap).raw()
    };
    assert_eq!(result, Ok(exception_word));
    assert_eq!(thread.take_pending_exception(), Some(smi(3)));
}
