use mark_sweep::{MarkSweep, MarkSweepConfig};

use vm::natives::{NativeContext, native_trampoline};
use vm::{Float, GcSlice, Smi, Value};
use vm::{Thread, VM, VmError};

fn float(thread: &mut Thread, v: f64) -> Value {
    thread.heap().allocate::<Float>(v).erase()
}

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

fn smi_add(_nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let (a, b) = match (args.get(1), args.get(2)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(VmError::Arity),
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
    fn fadd(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
        let (a, b) = match (args.get(1), args.get(2)) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(VmError::Arity),
        };
        let sum = nctx.heap().no_gc(|nogc| {
            let fa = a.get_as::<Float>(nogc).ok_or(VmError::Type)?.value.get();
            let fb = b.get_as::<Float>(nogc).ok_or(VmError::Type)?.value.get();
            Ok::<_, VmError>(fa + fb)
        })?;
        nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, sum)))
    }

    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();

    let fa = float(&mut thread, 1.5);
    let fb = float(&mut thread, 2.25);
    let r = thread.run_native(fadd, &[smi(0), fa, fb]).unwrap();
    let out = thread
        .heap()
        .no_gc(|nogc| r.get_as::<Float>(nogc).unwrap().value.get());
    assert_eq!(out, 3.75);

    assert_eq!(
        thread.run_native(fadd, &[smi(0), smi(1), smi(2)]),
        Err(VmError::Type)
    );
}

#[test]
fn trampoline_maps_errors_to_sentinel_and_pending_exception() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let args = [smi(0), smi(1)]; // arity error for smi_add

    let result = native_trampoline(smi_add, &mut thread, args.as_ptr(), args.len() as u32);

    assert_eq!(result, vm.known().exception.value());
    let ex = thread
        .take_pending_exception()
        .expect("pending exception set");
    assert!(!thread.has_pending_exception());

    // the pending value is a materialized TypeError object (Arity -> TypeError)
    let name = thread.handle_scope(|thread, scope| {
        let name = thread.intern(&scope, "name").value();
        let type_error = thread.intern(&scope, "TypeError").value();
        thread.heap().no_gc(|nogc| {
            let Some(o) = ex.as_heap_object(nogc) else {
                panic!("pending exception must be an object");
            };
            match o.as_ref().lookup(nogc, vm::SlotName::from_value(name)) {
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
    fn double(_nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
        match (args.get(1), args.get(2)) {
            (Some(v), _) => {
                let v = Smi::decode(v).ok_or(VmError::Type)?;
                Ok(Smi::new(v.value() * 2).encode())
            }
            _ => Err(VmError::Arity),
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
