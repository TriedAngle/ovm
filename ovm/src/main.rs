use bytecode::{Opcode, emit};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::natives::NativeIndex;
use ovm::{Thread, VM};
use vm::{
    CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, Float, Handle, HandleScope,
    HandlerEntryInit, HandlerTable, HandlerTableInit, Object, ObjectSlotsInit, Smi,
};

/// Build a bytecode function object (empty constants table).
fn callable<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    program: &[u8],
    register_count: usize,
    handlers: Option<&[HandlerEntryInit]>,
) -> Handle<'s, Object> {
    let empty_context = thread.heap().known().empty_context;
    let bytecode = thread
        .heap()
        .allocate_handle::<FixedByteArray>(program, scope);
    let constants = thread.heap().allocate_handle::<FixedArray>(&[], scope);
    let handlers = handlers.map(|entries| {
        thread
            .heap()
            .allocate_handle::<HandlerTable>(HandlerTableInit { entries }, scope)
    });
    let callable = thread.heap().allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            bytecode,
            constants,
            register_count,
            handlers,
        },
        scope,
    );
    let map = thread.heap().known().function_map;
    let empty_elements = thread.heap().known().empty_fixed_array.erase();
    thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[
                    callable.as_tagged().erase(),
                    empty_context.as_tagged().erase(),
                ],
                elements: empty_elements,
                length: 0,
            },
        )
        .into_handle(scope)
}

fn main() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).expect("failed to create heap");
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
        vm.heap().stats().capacity,
        vm.heap().stats().used
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
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Add, &[1]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let callable = callable(thread, &scope, &program, 2, None);
        thread.execute(callable, &[])
    });
    println!("6 + 7 = {}", Smi::decode(result.unwrap()).unwrap().value());

    // try { throw 42 } catch (e) { return e }
    // 0: LoadSmi 42 | 2: Throw | 3: Return | 4: Store r0 | 6: Load r0 | 8: Return
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[42]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);
    emit(&mut program, Opcode::Store, &[0]); // handler: bind e in r0
    emit(&mut program, Opcode::Load, &[0]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let callable = callable(
            thread,
            &scope,
            &program,
            1,
            Some(&[HandlerEntryInit::new(0, 3, 4)]),
        );
        thread.execute(callable, &[])
    });
    println!(
        "try {{ throw 42 }} catch (e) => e = {}",
        Smi::decode(result.unwrap()).unwrap().value()
    );

    // throw 7 with no handler: escapes the run as the exception sentinel
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[7]);
    emit(&mut program, Opcode::Throw, &[]);
    emit(&mut program, Opcode::Return, &[]);

    let result = thread.handle_scope(|thread, scope| {
        let callable = callable(thread, &scope, &program, 0, None);
        thread.execute(callable, &[])
    });
    match result {
        Ok(v) if v == thread.heap().known().exception.value() => {
            let ex = thread.take_pending_exception().expect("pending exception");
            println!("throw 7 (uncaught): escaped, pending exception = {ex:?}");
        }
        other => panic!("expected uncaught escape, got {other:?}"),
    }
}
