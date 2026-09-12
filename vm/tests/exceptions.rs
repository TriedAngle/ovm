use bytecode::{Opcode, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, HandlerEntryInit,
    HandlerTable, HandlerTableInit, ObjectSlotsInit, Smi, Value,
};
use vm::{Thread, VM};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

fn callable<'s>(
    thread: &mut Thread,
    scope: &'s vm::HandleScope<'_>,
    program: &[u8],
    constants: &[Value],
    register_count: usize,
    handlers: Option<&[HandlerEntryInit]>,
) -> Value {
    let the_hole = thread.heap().known().the_hole;
    let empty_context = thread.heap().known().empty_context;
    let bytecode = thread
        .heap()
        .allocate_handle::<FixedByteArray>(program, scope);
    let constants = thread
        .heap()
        .allocate_handle::<FixedArray>(constants, scope);
    let handlers = handlers.map(|entries| {
        thread
            .heap()
            .allocate_handle::<HandlerTable>(HandlerTableInit { entries }, scope)
    });
    let info = thread.heap().allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            bytecode,
            constants,
            register_count,
            handlers,
        },
        scope,
    );
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
        .erase()
}

#[test]
fn throw_is_caught_in_same_function() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
        thread.execute(scope.cast::<vm::Object>(f).unwrap(), &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 99);
    // catching consumes the pending exception
    assert!(!thread.has_pending_exception());
}

#[test]
fn throw_any_value_escapes_as_sentinel() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // throw 42: arbitrary values are throwable (ES §14.18), no handlers
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let f = callable(thread, &scope, &program, &[], 0, None);
        thread.execute(scope.cast::<vm::Object>(f).unwrap(), &[])
    });
    assert_eq!(result, Ok(thread.heap().known().exception.value()));
    assert_eq!(
        thread.take_pending_exception(),
        Some(smi(42)),
        "the thrown value itself is the pending exception"
    );
}

#[test]
fn innermost_handler_wins() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
        thread.execute(scope.cast::<vm::Object>(f).unwrap(), &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 100);
}

#[test]
fn exception_unwinds_to_caller() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
        let mut program = Vec::new();
        emit(&mut program, Opcode::LoadConstant, &[0]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::CallNoFeedback, &[0, 0, 1]);
        emit(&mut program, Opcode::Return, &[]);
        emit(&mut program, Opcode::Store, &[0]);
        emit(&mut program, Opcode::Load, &[0]);
        emit(&mut program, Opcode::Return, &[]);

        let caller = callable(
            thread,
            &scope,
            &program,
            &[callee],
            1,
            Some(&[HandlerEntryInit::new(0, 9, 9)]),
        );
        let result = thread.execute(scope.cast::<vm::Object>(caller).unwrap(), &[]);
        assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 7);
    });
}

#[test]
fn rethrow_from_finally_escapes_past_its_own_handler() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
        thread.execute(scope.cast::<vm::Object>(f).unwrap(), &[])
    });
    assert_eq!(result, Ok(thread.heap().known().exception.value()));
    assert_eq!(thread.take_pending_exception(), Some(smi(3)));
}

#[test]
fn stack_overflow_during_call_is_throwable() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();

    // self-recursive call with no handler: each Call pushes a frame until
    // the stack is exhausted, then a RangeError must escape the run
    let mut program = Vec::new();
    emit(&mut program, Opcode::Load, &[(-1i32) as u32]); // r0 = param 0 (self)
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::CallNoFeedback, &[0, 0, 1]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let f = callable(thread, &scope, &program, &[], 1, None);
        let handle = scope.cast::<vm::Object>(f).unwrap();
        thread.execute(handle, &[f])
    });
    assert_eq!(result, Ok(thread.heap().known().exception.value()));
    let ex = thread.take_pending_exception().expect("pending exception");
    let expected = thread.handle_scope(|thread, scope| thread.intern(&scope, "RangeError").value());
    thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name").value();
        thread.heap().no_gc(|nogc| {
            let Some(o) = ex.as_heap_object(nogc) else {
                panic!("pending exception must be an object");
            };
            match o.as_ref().lookup(nogc, vm::SlotName::from_value(name_key)) {
                vm::Lookup::Data { slot, .. } => {
                    assert_eq!(slot.inner(), expected, "stack overflow -> RangeError");
                }
                _ => panic!("error object must have a name property"),
            }
        });
    });
    // unwinding must not have consumed additional stack: the thread is
    // immediately usable again
    let mut good = Vec::new();
    emit(&mut good, Opcode::LoadSmi, &[5]);
    emit(&mut good, Opcode::Return, &[]);
    let result = thread.handle_scope(|thread, scope| {
        let f = callable(thread, &scope, &good, &[], 0, None);
        thread.execute(scope.cast::<vm::Object>(f).unwrap(), &[])
    });
    assert_eq!(Smi::decode(result.unwrap()).unwrap().value(), 5);
}
