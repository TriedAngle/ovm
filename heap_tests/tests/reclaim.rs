use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{FixedArray, VM};

fn vm() -> VM {
    heap_tests::vm::<MarkSweep>(MarkSweepConfig::default())
}

#[test]
fn unrooted_allocations_are_freed() {
    let vm = vm();
    let mut thread = vm.attach();
    let live_before = thread.handle_scope(|t, _outer| {
        let used0 = vm.heap().stats().used;
        thread_handle_scope(t, |t, scope| {
            for _ in 0..4 {
                let smis = vec![vm::Smi::new(0).into_tagged(); 8192];
                let _ = t
                    .heap()
                    .allocate_handle::<FixedArray>(scope.stage(&smis), &scope);
            }
        });
        let used1 = vm.heap().stats().used;
        assert!(used1 > used0);
        t.heap().collect();
        let used2 = vm.heap().stats().used;
        assert!(
            used2 < used1,
            "garbage not reclaimed: {used0} live, {used1} with garbage, {used2} after cycle"
        );
        used0
    });
    let _ = live_before;
}

fn thread_handle_scope<R>(
    t: &mut vm::Thread,
    f: impl for<'s> FnOnce(&mut vm::Thread, vm::HandleScope<'s>) -> R,
) -> R {
    t.handle_scope(f)
}

#[test]
fn weak_to_dead_is_cleared_and_does_not_retain() {
    let vm = vm();
    let mut thread = vm.attach();
    let index = thread.handle_scope(|t, scope| {
        let smis = vec![vm::Smi::new(0).into_tagged(); 4096];
        let target = t
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&smis), &scope);
        let index = vm.track_weak(target.as_tagged(&*t.heap()).raw());
        assert!(!vm.weak_value(index).is_cleared());
        index
    });
    // scope dropped: the target is garbage
    let used_with_target = vm.heap().stats().used;

    thread.heap().collect();

    assert!(vm.weak_value(index).is_cleared());
    let used_after = vm.heap().stats().used;
    assert!(
        used_after < used_with_target,
        "weak reference retained dead object: {used_with_target} -> {used_after}"
    );
}

#[test]
fn weak_to_live_stays_uncleared() {
    let vm = vm();
    let mut thread = vm.attach();
    thread.handle_scope(|t, scope| {
        let target = t
            .heap()
            .allocate_handle::<FixedArray>(scope.stage(&[vm::Smi::new(1).into_tagged()]), &scope);
        let index = vm.track_weak(target.as_tagged(&*t.heap()).raw());

        t.heap().collect();

        assert!(!vm.weak_value(index).is_cleared());
        assert_eq!(
            vm.weak_value(index).to_bits() & !vm::TAG_MASK,
            target.as_tagged(&*t.heap()).raw().to_bits() & !vm::TAG_MASK
        );
    });
}
