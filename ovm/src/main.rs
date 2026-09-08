//! ovm command-line front end.
//!
//! `ovm <file.js>...` runs script files in order; `ovm` with no files starts
//! a REPL. Each script runs in its own global scope. `--repl` drops into the
//! REPL after the files have run.

use std::io::Write;

use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{Thread, VM};
use vm::{Float, Smi, VMString, Value};

fn main() {
    let vm =
        VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).expect("failed to create heap");
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
    match thread.run_script(&src) {
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

/// Print and clear an uncaught exception, if one escaped. Returns whether
/// there was one.
fn report_uncaught(thread: &mut Thread) -> bool {
    match thread.take_pending_exception() {
        Some(ex) => {
            eprintln!("uncaught exception: {}", show_value(thread, ex));
            true
        }
        None => false,
    }
}

/// Best-effort display of a result value.
fn show_value(thread: &mut Thread, v: Value) -> String {
    if let Some(smi) = Smi::decode(v) {
        return smi.value().to_string();
    }
    let known = thread.heap().known();
    if v == known.undefined.value() {
        return "undefined".into();
    }
    if v == known.null.value() {
        return "null".into();
    }
    if v == known.true_object.value() {
        return "true".into();
    }
    if v == known.false_object.value() {
        return "false".into();
    }
    thread.heap().no_gc(|nogc| {
        if let Some(f) = v.get_as::<Float>(nogc) {
            return f.value.get().to_string();
        }
        if let Some(s) = v.get_as::<VMString>(nogc) {
            return String::from_utf8_lossy(s.as_slice(nogc)).into_owned();
        }
        format!("{v:?}")
    })
}
