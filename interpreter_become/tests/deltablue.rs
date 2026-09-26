use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::VM;

/// Octane's harness objects are not part of the VM; the benchmark only
/// needs the constructors so the suite registration doesn't throw.
const PRELUDE: &str = r#"
function BenchmarkSuite() {}
function Benchmark() {}
function alert(m) { throw new Error("ALERT: " + m); }
"#;

#[test]
fn deltablue() {
    let src = [
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../benchmarks/octane/deltablue/deltablue.js"
        ),
        "benchmarks/octane/deltablue/deltablue.js",
    ]
    .iter()
    .find_map(|p| std::fs::read_to_string(p).ok())
    .expect("deltablue source");
    let src = format!("{PRELUDE}{src}\ndeltaBlue();\n");
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    let mut thread = vm.attach();
    thread
        .eval::<JavascriptCompiler>(&src)
        .expect("deltablue runs");
    assert!(thread.take_pending_exception().is_none(), "deltablue threw");
}
