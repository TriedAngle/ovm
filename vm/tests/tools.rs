//! `KetteTools`: the global VM-introspection object with runtime-function
//! methods. Reachable from both frontends through global lookup, present
//! even on a bare `VM::new` (no JS builtins).

use std::sync::mpsc;
use std::time::Duration;

use bytecode::SourceMode;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{Smi, Termination, VM};

fn bare_vm() -> VM {
    VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap()
}

fn builtins_vm() -> VM {
    VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap()
}

fn run_bool(vm: &VM, src: &str) -> bool {
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    let heap = thread.heap();
    if result == heap.known().true_object.as_tagged(heap).raw() {
        true
    } else if result == heap.known().false_object.as_tagged(heap).raw() {
        false
    } else {
        panic!("expected boolean result, got {result:?}");
    }
}

#[test]
fn js_tools_present_and_gc_smoke() {
    assert!(run_bool(
        &builtins_vm(),
        "typeof KetteTools === 'object' \
         && typeof KetteTools.forceMinorGC === 'function' \
         && typeof KetteTools.forceMajorGC === 'function' \
         && typeof KetteTools.shutdown === 'function' \
         && KetteTools.forceMinorGC() === undefined \
         && KetteTools.forceMajorGC() === undefined"
    ));
}

#[test]
fn kette_tools_reachable_without_builtins() {
    let vm = bare_vm();
    let mut thread = vm.attach();
    let value = thread
        .run_source(
            "KetteTools.forceMinorGC()\nKetteTools.forceMajorGC()\n42",
            kette_compiler::compile_kette,
            SourceMode::Script,
        )
        .unwrap();
    assert_eq!(Smi::decode(value).unwrap().value(), 42);
}

#[test]
fn gc_methods_collect_under_allocation() {
    // enough allocation pressure that both collectors really cycle
    let src = "\
        var keep = [];\
        for (var i = 0; i < 2000; i++) { keep[i] = [i, i + 1]; }\
        KetteTools.forceMinorGC();\
        KetteTools.forceMajorGC();\
        var ok = keep.length === 2000 && keep[1999][1] === 2000;\
        ok\
    ";
    assert!(run_bool(&builtins_vm(), src));
}

#[test]
fn shutdown_terminates_script_without_running_rest() {
    let vm = builtins_vm();
    let mut thread = vm.attach();
    let result = thread.run_script("KetteTools.shutdown(); 99").unwrap();
    let heap = thread.heap();
    assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
    assert_eq!(thread.state().termination(), Some(Termination::Shutdown));
    assert!(vm.is_shutdown());
    assert!(!thread.has_pending_exception());
}

#[test]
fn shutdown_is_uncatchable() {
    // no catch handler may run
    let vm = builtins_vm();
    let mut thread = vm.attach();
    let result = thread
        .run_script("try { KetteTools.shutdown() } catch (e) { 1 } 3")
        .unwrap();
    let heap = thread.heap();
    assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
    assert_eq!(thread.state().termination(), Some(Termination::Shutdown));

    // ...also not through nested calls
    let vm = builtins_vm();
    let mut thread = vm.attach();
    let result = thread
        .run_script(
            "function inner() { KetteTools.shutdown(); return 1 }\
             function outer() { try { return inner() } catch (e) { return 2 } }\
             outer()",
        )
        .unwrap();
    let heap = thread.heap();
    assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
    assert_eq!(thread.state().termination(), Some(Termination::Shutdown));
}

#[test]
fn shutdown_cancels_other_threads_at_safepoints() {
    let vm = bare_vm();
    let (done_tx, done_rx) = mpsc::channel();

    // two mutators spinning in interpreted loops: they only ever pause at
    // the loop back-edge safepoint
    let mut handles = Vec::new();
    for _ in 0..2 {
        let done_tx = done_tx.clone();
        handles.push(vm.spawn(move |thread| {
            let result = thread.run_script("while (true) {}").unwrap();
            // graceful exit: the loop halted at its safepoint and unwound
            let heap = thread.heap();
            assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
            assert_eq!(thread.state().termination(), Some(Termination::Shutdown));
            done_tx.send(()).unwrap();
        }));
    }
    drop(done_tx);

    // let the loops spin up, then run the shutdown protocol
    std::thread::sleep(Duration::from_millis(100));
    let mut main = vm.attach();
    main.run_script("KetteTools.shutdown()").unwrap();
    assert!(vm.is_shutdown());

    // both mutators halted and their threads ended (joinable, no leak)
    for _ in 0..2 {
        done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("mutator halted at its safepoint");
    }
    for handle in handles {
        handle.join().expect("mutator thread exited cleanly");
    }
}

#[test]
fn vm_stays_usable_after_shutdown() {
    let vm = bare_vm();

    // a spinning mutator gets cancelled...
    let handle = vm.spawn(move |thread| {
        thread.run_script("while (true) {}").unwrap();
        assert_eq!(thread.state().termination(), Some(Termination::Shutdown));
    });
    std::thread::sleep(Duration::from_millis(50));

    // ...by a shutdown from the main thread...
    {
        let mut main = vm.attach();
        main.run_script("KetteTools.shutdown()").unwrap();
    }
    handle.join().expect("cancelled mutator exited cleanly");

    // ...and the VM keeps running scripts afterwards: the termination
    // belonged to the previous executions only
    {
        let mut again = vm.attach();
        let value = again.run_script("21 * 2").unwrap();
        assert_eq!(Smi::decode(value).unwrap().value(), 42);
        assert_eq!(again.state().termination(), None);
    }

    // a freshly spawned mutator also runs normally after the shutdown
    vm.spawn(move |thread| {
        let value = thread.run_script("6 * 7").unwrap();
        assert_eq!(Smi::decode(value).unwrap().value(), 42);
    })
    .join()
    .expect("fresh mutator runs after shutdown");
}

#[test]
fn second_shutdown_returns_immediately() {
    let vm = builtins_vm();
    {
        let mut a = vm.attach();
        a.run_script("KetteTools.shutdown()").unwrap();
        assert_eq!(a.state().termination(), Some(Termination::Shutdown));
    } // a detaches: an attached-but-idle thread never parks, and the
    // protocol (like the GC) waits for every attached thread

    // a second shutdown on a fresh execution still works
    let mut b = vm.attach();
    b.run_script("KetteTools.shutdown()").unwrap();
    assert_eq!(b.state().termination(), Some(Termination::Shutdown));
}
