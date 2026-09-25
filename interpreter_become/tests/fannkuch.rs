use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::VM;

#[test]
fn fannkuch() {
    let src = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/../benchmarks/fannkuch/fannkuch.js"),
        "benchmarks/fannkuch/fannkuch.js",
    ]
    .iter()
    .find_map(|p| std::fs::read_to_string(p).ok())
    .expect("fannkuch source");
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    let mut thread = vm.attach();
    thread.eval::<JavascriptCompiler>(&src).expect("fannkuch runs");
    assert!(
        thread.take_pending_exception().is_none(),
        "fannkuch threw"
    );
}
