use mark_sweep::{MarkSweep, MarkSweepConfig};

use vm::natives::NativeContext;
use vm::{Float, HandleSlice, Smi, Value};
use vm::{Thread, VM, VmError};

fn float(thread: &mut Thread, v: f64) -> Value {
    thread.heap().allocate::<Float>(v).raw()
}

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

fn smi_add(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
    let (a, b) = {
        let heap = &*nctx.heap();
        match (
            args.get(1).map(|h| h.as_tagged(heap)),
            args.get(2).map(|h| h.as_tagged(heap)),
        ) {
            (Some(a), Some(b)) => (a.raw(), b.raw()),
            _ => return Err(VmError::Arity),
        }
    };
    let (a, b) = (
        Smi::decode(a).ok_or(VmError::Type)?,
        Smi::decode(b).ok_or(VmError::Type)?,
    );
    Ok(Smi::new(a.value() + b.value()).encode())
}

#[test]
fn registered_native_invokes_and_checks_types() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let r = thread
        .run_native(smi_add, &[smi(0), smi(6), smi(7)])
        .unwrap();
    assert_eq!(r.to_i64().unwrap(), 13);

    assert_eq!(
        thread.run_native(smi_add, &[smi(0), smi(1)]),
        Err(VmError::Arity)
    );

    let f = float(&mut thread, 1.0);
    assert_eq!(
        thread.run_native(smi_add, &[smi(0), f, smi(1)]),
        Err(VmError::Type)
    );
}

#[test]
fn native_result_is_boxed_when_not_smi() {
    fn fadd(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
        let sum = {
            let heap = &*nctx.heap();
            let (a, b) = match (
                args.get(1).map(|h| h.as_tagged(heap)),
                args.get(2).map(|h| h.as_tagged(heap)),
            ) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(VmError::Arity),
            };
            let fa = a.get_as::<Float>().ok_or(VmError::Type)?.value.get();
            let fb = b.get_as::<Float>().ok_or(VmError::Type)?.value.get();
            Ok::<f64, VmError>(fa + fb)
        }?;
        nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, sum).raw()))
    }

    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let fa = float(&mut thread, 1.5);
    let fb = float(&mut thread, 2.25);
    let r = thread.run_native(fadd, &[smi(0), fa, fb]).unwrap();
    let out = {
        let heap = &*thread.heap();
        unsafe { r.assume_valid(heap) }
            .get_as::<Float>()
            .unwrap()
            .value
            .get()
    };
    assert_eq!(out, 3.75);

    assert_eq!(
        thread.run_native(fadd, &[smi(0), smi(1), smi(2)]),
        Err(VmError::Type)
    );
}

/*
#[test]
fn trampoline_maps_errors_to_sentinel_and_pending_exception() {
    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let idx = vm.register_native(smi_add);
    let mut thread = vm.attach();
    let args = [smi(0), smi(1)]; // arity error for smi_add

    let result = unsafe { native_trampoline(idx.0, &mut thread, args.as_ptr(), args.len() as u32) };

    let exception_word = {
        let heap = thread.heap();
        heap.known().exception.as_tagged(heap).raw()
    };
    assert_eq!(result, exception_word);
    let ex = thread
        .take_pending_exception()
        .expect("pending exception set");
    assert!(!thread.has_pending_exception());

    // the pending value is a materialized TypeError object (Arity -> TypeError)
    let name = thread.handle_scope(|thread, scope| {
        let name = thread.intern(&scope, "name");
        let type_error = thread.intern(&scope, "TypeError");
        let (name_ok, type_error_word) = {
            let heap = &*thread.heap();
            let Some(o) = unsafe { ex.assume_valid(heap) }.as_heap_object() else {
                panic!("pending exception must be an object");
            };
            match o.as_ref().lookup(heap, name.as_tagged(heap).into()) {
                vm::Lookup::Data { slot, .. } => {
                    let name_ok = slot.get(heap).raw() == type_error.as_tagged(heap).raw();
                    (name_ok, type_error.as_tagged(heap).raw())
                }
                _ => panic!("error object must have a name property"),
            }
        };
        assert!(name_ok, "error object must have a name property");
        type_error_word
    });
    let expected = thread.handle_scope(|thread, scope| {
        let type_error = thread.intern(&scope, "TypeError");
        let heap = &*thread.heap();
        type_error.as_tagged(heap).raw()
    });
    assert_eq!(name, expected);
}
*/

#[test]
fn register_native_appends_after_well_known() {
    fn double(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
        let heap = &*nctx.heap();
        match args.get(1).map(|h| h.as_tagged(heap)) {
            Some(v) => {
                let v = Smi::decode(v.raw()).ok_or(VmError::Type)?;
                Ok(Smi::new(v.value() * 2).encode())
            }
            None => Err(VmError::Arity),
        }
    }

    let mut vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let idx = vm.register_native(double as vm::NativeFn);
    // the registry starts with the fixed RuntimeFn table; dynamic
    // registrations append after it
    let fixed = bytecode::RuntimeFn::COUNT as usize;
    assert_eq!(idx.0, fixed);
    assert_eq!(vm.natives().len(), fixed + 1);

    let mut thread = vm.attach();
    let r = thread
        .run_native(vm.native(idx), &[smi(0), smi(21)])
        .unwrap();
    assert_eq!(r.to_i64().unwrap(), 42);
}
