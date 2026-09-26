use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::VM;

fn main() {
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .expect("failed to create heap")
        .add::<JSRuntime>()
        .expect("failed to install js runtime");
    let mut thread = vm.attach();

    let mut parts = Vec::new();
    for path in std::env::args().skip(1) {
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("ovm: cannot read {path}: {e}");
            std::process::exit(1);
        });
        parts.push(src);
    }
    let joined = parts.join("\n");
    match thread.run_source(
        &joined,
        js_compiler::compile_js,
        js_compiler::SourceMode::Script,
    ) {
        Ok(_) => {
            if let Some(ex) = thread.take_pending_exception() {
                eprintln!("uncaught exception: {ex:?}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
