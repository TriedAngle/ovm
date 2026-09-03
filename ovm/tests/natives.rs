use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::natives::{NativeContext, NativeIndex, native_trampoline};
use ovm::{Thread, VM, VmError};
use vm::{Float, Smi, Value};

fn float(thread: &mut Thread, v: f64) -> Value {
    thread.heap().allocate::<Float>(v).erase()
}

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

#[test]
fn smi_add_adds_and_checks_types() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let add = vm.native(NativeIndex::SMI_ADD);

    let r = thread.run_native(add, &[smi(0), smi(6), smi(7)]).unwrap();
    assert_eq!(Smi::decode(r).unwrap().value(), 13);

    assert_eq!(
        thread.run_native(add, &[smi(0), smi(1)]),
        Err(VmError::Arity)
    );

    let f = float(&mut thread, 1.0);
    assert_eq!(
        thread.run_native(add, &[smi(0), f, smi(1)]),
        Err(VmError::Type)
    );
}

#[test]
fn float_add_adds_and_boxes_result() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let add = vm.native(NativeIndex::FLOAT_ADD);

    let fa = float(&mut thread, 1.5);
    let fb = float(&mut thread, 2.25);
    let r = thread.run_native(add, &[smi(0), fa, fb]).unwrap();
    let out = thread.heap().no_gc(|nogc, heap| {
        r.get_as::<Float>(nogc, heap.known().float_map)
            .unwrap()
            .value
            .get()
    });
    assert_eq!(out, 3.75);

    assert_eq!(
        thread.run_native(add, &[smi(0), smi(1), smi(2)]),
        Err(VmError::Type)
    );
}

#[test]
fn trampoline_maps_errors_to_sentinel_and_pending_exception() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let args = [smi(0), smi(1)]; // arity error for smi_add

    let result = native_trampoline(
        vm.native(NativeIndex::SMI_ADD),
        &mut thread,
        args.as_ptr(),
        args.len() as u32,
    );

    assert_eq!(result, vm.heap().known().exception.value());
    let ex = thread
        .take_pending_exception()
        .expect("pending exception set");
    assert!(!thread.has_pending_exception());

    // the pending value is a materialized TypeError object (Arity -> TypeError)
    let name = thread.handle_scope(|thread, scope| {
        let name = thread.intern(&scope, "name").value();
        let type_error = thread.intern(&scope, "TypeError").value();
        thread.heap().no_gc(|nogc, heap| {
            let vm::ValueRef::Object(o) = ex.value_ref(nogc) else {
                panic!("pending exception must be an object");
            };
            match o
                .as_ref()
                .lookup(nogc, heap, vm::SlotName::from_value(name))
            {
                vm::Lookup::Data { slot, .. } => {
                    assert_eq!(slot.inner(), type_error);
                }
                _ => panic!("error object must have a name property"),
            }
        });
        type_error
    });
    assert_eq!(name, {
        thread.handle_scope(|thread, scope| thread.intern(&scope, "TypeError").value())
    });
}

#[test]
fn register_native_appends_after_well_known() {
    fn double(_nctx: &mut NativeContext<'_>, args: &[Value]) -> Result<Value, VmError> {
        match args {
            [_, v] => {
                let v = Smi::decode(*v).ok_or(VmError::Type)?;
                Ok(Smi::new(v.value() * 2).encode())
            }
            _ => Err(VmError::Arity),
        }
    }

    let mut vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(double as ovm::NativeFn);
    assert!(idx > NativeIndex::FLOAT_ADD);
    assert_eq!(vm.natives().len(), 3);

    let mut thread = vm.attach();
    let r = thread
        .run_native(vm.native(idx), &[smi(0), smi(21)])
        .unwrap();
    assert_eq!(Smi::decode(r).unwrap().value(), 42);
}
