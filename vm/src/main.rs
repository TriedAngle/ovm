use std::io::Write;
use std::path::Path;

use bytecode::{CompileFn, SourceMode};
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DenseString, Float, Smi, Value};
use vm::{Thread, VM};

fn main() {
    trace::init();
    let vm =
        VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).expect("failed to create heap");
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
    }
    if files.is_empty() || then_repl {
        repl(&mut thread);
    }
}

fn run_file(thread: &mut Thread, path: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("ovm: cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    // the frontend is chosen at the call site; the VM stays language-agnostic
    let compile: CompileFn = match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("js") => js_compiler::compile_js,
        Some(ext) if ext.eq_ignore_ascii_case("ktt") => kette_compiler::compile_kette,
        _ => {
            eprintln!("ovm: cannot tell the language of {path} (expected .js or .ktt)");
            std::process::exit(1);
        }
    };
    match thread.run_source(&src, compile, SourceMode::Script) {
        Ok(_) => {
            if report_uncaught(thread) {
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
        match thread.run_script_repl(line) {
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
        format!("{v:?}")
    }
}
