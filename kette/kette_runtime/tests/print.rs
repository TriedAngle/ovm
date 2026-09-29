use interpreter_match_loop::MatchLoopInterpreter;
use kette_compiler::KetteCompiler;
use kette_runtime::KetteRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::{Smi, VM};

fn vm() -> VM {
    VM::new::<MarkSweep, MatchLoopInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<KetteRuntime>()
        .unwrap()
}

#[test]
fn console_print_runs() {
    let vm = vm();
    let mut thread = vm.attach();
    let value = thread
        .eval::<KetteCompiler>(
            "let obj = { x: 42\n get: { self.x } }\nConsole.print(\"hello kette\")\nobj.get()",
        )
        .expect("script runs");
    assert_eq!(Smi::decode(value).expect("smi result").value(), 42);
}

#[test]
fn kette_objects_stay_plain() {
    let vm = vm();
    let mut thread = vm.attach();
    let value = thread
        .eval::<KetteCompiler>("let obj = { x: 10\n read: { self.x } }\nobj.x = 5\nobj.x")
        .expect("script runs");
    assert_eq!(Smi::decode(value).expect("smi result").value(), 5);
}

#[test]
fn missing_slot_store_throws() {
    let vm = vm();
    let mut thread = vm.attach();
    let result = thread.eval::<KetteCompiler>("let obj = { x: 1 }\nobj.y = 5");
    assert!(result.is_ok(), "the failure is a runtime throw");
    assert!(
        thread.take_pending_exception().is_some(),
        "`obj.y = 5` on a missing slot must throw"
    );
}
