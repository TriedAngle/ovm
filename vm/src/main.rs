use std::io::Write;

use kette_compiler::KetteCompiler;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{Compiler, JSRuntime, Thread, ThreadedInterpreter, VM};
use vm::{
    DenseString, Float, Heap, JavascriptCompiler, LoadOutcome, Lookup, SlotName, Smi, Tagged, Value,
};

fn main() {
    trace::init();
    let vm = VM::new::<MarkSweep, ThreadedInterpreter>(MarkSweepConfig::default())
        .expect("failed to create heap")
        .add::<JSRuntime>()
        .expect("failed to install js runtime");
    vm.arm_gc_stress();
    let mut thread = vm.attach();

    let mut files = Vec::new();
    let mut then_repl = false;
    for arg in std::env::args().skip(1) {
        if arg == "--repl" {
            then_repl = true;
        } else {
            files.push(arg);
        }
    }
    for path in &files {
        run_file(&mut thread, path);
        if thread.vm().is_shutdown() {
            return;
        }
    }
    if files.is_empty() || then_repl {
        repl(&mut thread);
    }
}

// the frontend is chosen at the call site; the VM stays language-agnostic
fn run_file(thread: &mut Thread, path: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("ovm: cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    match std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("js") => run_with::<JavascriptCompiler>(thread, &src, path),
        Some("ktt") => run_with::<KetteCompiler>(thread, &src, path),
        _ => {
            eprintln!("ovm: cannot tell the language of {path} (expected .js or .ktt)");
            std::process::exit(1);
        }
    }
}

fn run_with<C: Compiler>(thread: &mut Thread, src: &str, path: &str) {
    match thread.eval::<C>(src) {
        Ok(_) => {
            if report_uncaught(thread) {
                eprintln!("{path}: uncaught exception");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn repl(thread: &mut Thread) {
    let stdin = std::io::stdin();
    loop {
        if thread.vm().is_shutdown() {
            break;
        }
        print!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("ovm: read error: {e}");
                break;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == ".exit" {
            break;
        }
        match thread.eval_repl::<JavascriptCompiler>(line) {
            Ok(v) => {
                if report_uncaught(thread) {
                    continue;
                }
                println!("{}", show_value(thread, v));
            }
            Err(e) => eprintln!("{e}"),
        }
    }
}

fn report_uncaught(thread: &mut Thread) -> bool {
    match thread.take_pending_exception() {
        Some(ex) => {
            eprintln!("uncaught exception: {}", show_value(thread, ex));
            true
        }
        None => false,
    }
}

fn show_value(thread: &mut Thread, v: Value) -> String {
    if let Some(smi) = Smi::decode(v) {
        return smi.value().to_string();
    }
    {
        let heap = &*thread.heap();
        let known = heap.known();
        if v == known.undefined.as_tagged(heap).raw() {
            return "undefined".into();
        }
        if v == known.null.as_tagged(heap).raw() {
            return "null".into();
        }
        if v == known.true_object.as_tagged(heap).raw() {
            return "true".into();
        }
        if v == known.false_object.as_tagged(heap).raw() {
            return "false".into();
        }
        if let Some(f) = unsafe { v.assume_valid(heap) }.get_as::<Float>() {
            return f.value.get().to_string();
        }
        if let Some(s) = unsafe { v.assume_valid(heap) }.get_as::<DenseString>() {
            return s.to_rust_string(heap);
        }
        if let Some(text) = error_text(heap, v) {
            return text;
        }
        format!("{v:?}")
    }
}

fn error_text(heap: &Heap, v: Value) -> Option<String> {
    let tagged = unsafe { v.assume_valid(heap) };
    tagged.as_heap_object()?;
    let known = heap.known();
    let name = error_property(heap, tagged, known.strings.name.as_tagged(heap));
    let message = error_property(heap, tagged, known.strings.message.as_tagged(heap));
    match (name, message) {
        (Some(n), Some(m)) if !m.is_empty() => Some(format!("{n}: {m}")),
        (Some(n), _) => Some(n),
        (None, Some(m)) if !m.is_empty() => Some(m),
        _ => None,
    }
}

fn error_property<'a>(
    heap: &'a Heap,
    v: Tagged<'a, Value>,
    name: Tagged<'a, SlotName>,
) -> Option<String> {
    match Lookup::load_outcome_on(heap, v, name) {
        Ok(LoadOutcome::Value(x)) => x.get_as::<DenseString>().map(|s| s.to_rust_string(heap)),
        _ => None,
    }
}
