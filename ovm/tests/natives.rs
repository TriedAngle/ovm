use core::alloc::Layout;

use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::natives::{EXCEPTION_SENTINEL, NativeIndex, native_trampoline};
use ovm::{Context, VM, VmError};
use vm::{Float, HeapObject, HeapPtr, LocalHeap, Map, ObjectKind, Smi, Tagged, Value};

fn make_float_map(ctx: &mut Context<DummyHeap>) -> Tagged<Map> {
    let ptr = ctx.heap().allocate::<Map>(Map::layout_for(0)).into_ptr();
    let map = unsafe { ptr.as_ref() };
    map.kind
        .set(ctx.heap(), map.erase(), ObjectKind::Float.to_smi());
    map.value_slot_count
        .set(ctx.heap(), map.erase(), Smi::new_unchecked(0));
    map.descriptor_count
        .set(ctx.heap(), map.erase(), Smi::new_unchecked(0));
    Tagged::from_ptr(ptr)
}

fn make_float(ctx: &mut Context<DummyHeap>, map: Tagged<Map>, v: f64) -> Value {
    let ptr = ctx
        .heap()
        .allocate::<Float>(Layout::new::<Float>())
        .into_ptr();
    let fl = unsafe { ptr.as_ref() };
    fl.header.map.set(ctx.heap(), fl.erase(), map);
    fl.value.set(v);
    ptr.encode_strong()
}

fn smi(v: i64) -> Value {
    Smi::new_unchecked(v).encode()
}

#[test]
fn smi_add_adds_and_checks_types() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut ctx = vm.attach();
    let add = vm.native(NativeIndex::SMI_ADD);

    let r = add(&mut ctx, &[smi(0), smi(6), smi(7)]).unwrap();
    assert_eq!(Smi::decode(r).unwrap().value(), 13);

    assert_eq!(add(&mut ctx, &[smi(0), smi(1)]), Err(VmError::Arity));

    let map = make_float_map(&mut ctx);
    let f = make_float(&mut ctx, map, 1.0);
    assert_eq!(add(&mut ctx, &[smi(0), f, smi(1)]), Err(VmError::Type));
}

#[test]
fn float_add_adds_and_boxes_result() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut ctx = vm.attach();
    let map = make_float_map(&mut ctx);
    let add = vm.native(NativeIndex::FLOAT_ADD);

    let fa = make_float(&mut ctx, map, 1.5);
    let fb = make_float(&mut ctx, map, 2.25);
    let r = add(&mut ctx, &[smi(0), fa, fb]).unwrap();
    let out = unsafe { HeapPtr::<Float>::decode(r).unwrap().as_ref() };
    assert_eq!(out.value.get(), 3.75);

    assert_eq!(add(&mut ctx, &[smi(0), smi(1), smi(2)]), Err(VmError::Type));
}

#[test]
fn trampoline_maps_errors_to_sentinel_and_pending_exception() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let mut ctx = vm.attach();
    let args = [smi(0), smi(1)]; // arity error for smi_add

    let result = native_trampoline(
        vm.native(NativeIndex::SMI_ADD),
        &mut ctx,
        args.as_ptr(),
        args.len() as u32,
    );

    assert_eq!(result, EXCEPTION_SENTINEL);
    assert_eq!(ctx.take_pending_exception(), Some(VmError::Arity));
    assert!(!ctx.has_pending_exception());
}

#[test]
fn register_native_appends_after_well_known() {
    fn double<H: vm::Heap>(_ctx: &mut Context<H>, args: &[Value]) -> Result<Value, VmError> {
        match args {
            [_, v] => {
                let v = Smi::decode(*v).ok_or(VmError::Type)?;
                Ok(Smi::new_unchecked(v.value() * 2).encode())
            }
            _ => Err(VmError::Arity),
        }
    }

    let mut vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).unwrap();
    let idx = vm.register_native(double as ovm::NativeFn<DummyHeap>);
    assert!(idx > NativeIndex::FLOAT_ADD);
    assert_eq!(vm.natives().len(), 3);

    let mut ctx = vm.attach();
    let r = vm.native(idx)(&mut ctx, &[smi(0), smi(21)]).unwrap();
    assert_eq!(Smi::decode(r).unwrap().value(), 42);
}
