use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::VM;

fn run(path: &str) {
    let src = std::fs::read_to_string(path).unwrap();
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let _result = thread.eval::<JavascriptCompiler>(&src).expect("nbody runs");
    if let Some(ex) = thread.take_pending_exception() {
        let heap = thread.heap();
        let tagged = unsafe { ex.assume_valid(heap) };
        if let Some(s) = tagged.get_as::<vm_core::DenseString>(heap) {
            panic!("nbody threw: {}", s.to_rust_string(heap));
        }
        panic!("nbody threw: {ex:?}");
    }
}

#[test]
fn nbody() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../benchmarks/nbody/nbody.js");
    run(path);
}
