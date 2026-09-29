use interpreter_become::BecomeInterpreter;
use interpreter_match_loop::MatchLoopInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::{Interpreter, VM};

const NBODY: &str = include_str!("../../benchmarks/nbody/nbody.js");
const FANNKUCH: &str = include_str!("../../benchmarks/fannkuch/fannkuch.js");

fn bench<I: Interpreter>(name: &str, src: &str) {
    let vm = VM::new::<MarkSweep, I>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let start = std::time::Instant::now();
    thread.eval::<JavascriptCompiler>(src).expect("runs");
    assert!(thread.take_pending_exception().is_none());
    println!("{name}: {:?}", start.elapsed());
}

fn once(name: &str, src: &str) {
    bench::<MatchLoopInterpreter>(&format!("{name}/match_loop"), src);
    bench::<BecomeInterpreter>(&format!("{name}/become"), src);
    bench::<MatchLoopInterpreter>(&format!("{name}/match_loop"), src);
    bench::<BecomeInterpreter>(&format!("{name}/become"), src);
}

// benchmark drivers, not correctness tests: opt in with
// `cargo test -- --ignored` (they are unbounded under the
// stress-minor-gc feature — a collection per allocation)
#[test]
#[ignore]
fn bench_all() {
    once("nbody", NBODY);
    once("fannkuch", FANNKUCH);
}
