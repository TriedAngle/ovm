//! Minimal test262 runner: runs sta.js + assert.js + each test file in a
//! VM with the builtin library installed.
//!
//! A test passes when it completes without an uncaught exception; tests
//! with `negative: { phase: parse }` frontmatter must fail to parse.
//! Feature-gated tests whose features the VM lacks (BigInt, Symbol, ...)
//! are counted separately as skipped.
//!
//! Usage: cargo run -p ovm --example test262 -- <harness...> <test-file-or-dir>...

use std::path::{Path, PathBuf};

use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{ScriptError, VM};

const UNSUPPORTED_FEATURES: &[&str] = &["BigInt", "Symbol", "Temporal", "regexp-modifiers"];
/// Tests exercising runtime objects the VM does not have yet.
const UNSUPPORTED_PATTERNS: &[&str] = &["new Date"];
/// Tests under paths the VM cannot support yet: missing global namespaces,
/// RegExp (literals and engine), `with` statements, modules/dynamic import.
const UNSUPPORTED_PATHS: &[&str] = &[
    // RegExp: no literal scanning, no engine
    "/built-ins/RegExp/",
    "/literals/regexp/",
    "regexp-literal",
    // `with` statements
    "/statements/with/",
    // modules
    "/dynamic-import/",
    "_FIXTURE",
    // missing global namespaces
    "/built-ins/Temporal/",
    "/built-ins/Reflect/",
    "/built-ins/Promise/",
    "/built-ins/Math/",
    "/built-ins/JSON/",
    "/built-ins/Map/",
    "/built-ins/Set/",
    "/built-ins/WeakMap/",
    "/built-ins/WeakSet/",
    "/built-ins/Proxy/",
    "/built-ins/ArrayBuffer/",
    "/built-ins/SharedArrayBuffer/",
    "/built-ins/DataView/",
    "/built-ins/TypedArray",
    "/built-ins/Atomics/",
    "/built-ins/FinalizationRegistry/",
    "/built-ins/WeakRef/",
    "/built-ins/Iterator/",
    // sparse/dictionary array elements (huge dense allocations)
    "S15.4.5.2_A1_T1",
    "S15.4.2.2_A2.1_T1",
    "S15.4.5.2_A3_T4",
    "property-cast-number",
    "15.4.4.14-9-9",
    "15.4.4.15-8-9",
];

#[derive(Default)]
struct Stats {
    pass: usize,
    fail: Vec<(PathBuf, String)>,
    panicked: Vec<PathBuf>,
    skipped_feature: usize,
    skipped_module: usize,
}

fn frontmatter(src: &str) -> &str {
    let start = src.find("/*---").map(|i| i + 5);
    let end = src.find("---*/");
    match (start, end) {
        (Some(s), Some(e)) if s <= e => &src[s..e],
        _ => "",
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("cannot read dir {}", dir.display());
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "js") {
            out.push(path);
        }
    }
}

fn run_test(harness: &str, path: &Path, stats: &mut Stats) {
    // survive panics (e.g. bytecode operand overflow on huge generated
    // files): count them separately and keep going
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_test_inner(harness, path, stats)
    }));
    if result.is_err() {
        stats.panicked.push(path.to_path_buf());
    }
}

/// The pending exception's `name` property (error class), for categorizing
/// uncaught exceptions in the stats.
fn exception_name(thread: &mut ovm::Thread) -> String {
    thread.handle_scope(|thread, scope| {
        let Some(ex) = thread.take_pending_exception() else {
            return "exception".into();
        };
        let name_key = thread.intern(&scope, "name").value();
        thread.heap().no_gc(|nogc, heap| {
            let vm::ValueRef::Object(o) = ex.value_ref(nogc) else {
                return "exception".into();
            };
            match o
                .as_ref()
                .lookup(nogc, heap, vm::SlotName::from_value(name_key))
            {
                vm::Lookup::Data { slot, .. } => slot
                    .inner()
                    .get_as::<vm::VMString>(nogc, heap.known().string_map)
                    .map(|s| String::from_utf8_lossy(s.as_slice(nogc)).into_owned())
                    .unwrap_or_else(|| "exception".into()),
                _ => "exception".into(),
            }
        })
    })
}

fn run_test_inner(harness: &str, path: &Path, stats: &mut Stats) {
    let Ok(src) = std::fs::read_to_string(path) else {
        stats.fail.push((path.to_path_buf(), "not utf-8".into()));
        return;
    };
    let fm = frontmatter(&src);
    if fm.contains("module") && fm.contains("flags") {
        stats.skipped_module += 1;
        return;
    }
    if UNSUPPORTED_FEATURES.iter().any(|f| fm.contains(f)) {
        stats.skipped_feature += 1;
        return;
    }
    let path_str = path.to_string_lossy();
    if UNSUPPORTED_PATHS.iter().any(|p| path_str.contains(p)) {
        stats.skipped_feature += 1;
        return;
    }
    if UNSUPPORTED_PATTERNS.iter().any(|p| src.contains(p)) {
        stats.skipped_feature += 1;
        return;
    }
    let expect_parse_error =
        fm.contains("negative:") && (fm.contains("phase: parse") || fm.contains("phase: syntax"));
    // `raw` tests run without the harness preludes (INTERPRETING.md)
    let raw = fm.contains("flags") && fm.contains("raw");

    let code = if raw {
        src
    } else {
        format!("{harness}\n{src}\n")
    };
    // realm isolation: every test runs in a fresh VM (INTERPRETING.md)
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig { heap_size: 1 << 30 }).expect("vm");
    let mut thread = vm.attach();
    match thread.run_script(&code) {
        Ok(v) if v == thread.heap().known().exception.value() => {
            let name = exception_name(&mut thread);
            stats
                .fail
                .push((path.to_path_buf(), format!("uncaught {name}")));
        }
        Ok(_) => {
            if expect_parse_error {
                stats.fail.push((
                    path.to_path_buf(),
                    "negative test did not fail to parse".into(),
                ));
            } else {
                stats.pass += 1;
            }
        }
        Err(ScriptError::Parse(_)) if expect_parse_error => stats.pass += 1,
        Err(e) => {
            stats.fail.push((path.to_path_buf(), e.to_string()));
        }
    }
}

fn main() {
    // deeply nested tests overflow the default 8 MiB stack in the recursive
    // parser/codegen; run everything on a thread with a large stack
    let child = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(real_main)
        .expect("spawn runner thread");
    let code = child.join().unwrap_or(101);
    std::process::exit(code);
}

fn real_main() -> i32 {
    // caught panics are counted per test; don't spam stderr for each
    std::panic::set_hook(Box::new(|_| {}));
    let progress = std::env::var_os("OVM_PROGRESS").is_some();
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if args.is_empty() {
        eprintln!("usage: test262 <harness...> <test-file-or-dir>...");
        std::process::exit(2);
    }
    // heuristic: the first two args are harness files (sta.js, assert.js)
    let harness = args
        .iter()
        .take(2)
        .map(|p| std::fs::read_to_string(p).expect("harness file"))
        .collect::<Vec<_>>()
        .join("\n");

    let mut files = Vec::new();
    for root in &args[2..] {
        if root.is_dir() {
            collect(root, &mut files);
        } else {
            files.push(root.clone());
        }
    }
    files.sort();

    let mut stats = Stats::default();
    for file in &files {
        if progress {
            // last line before a crash identifies the culprit test
            eprintln!("running {}", file.display());
        }
        run_test(&harness, file, &mut stats);
    }

    println!("total:           {}", files.len());
    println!("pass:            {}", stats.pass);
    println!("fail:            {}", stats.fail.len());
    println!("panic:           {}", stats.panicked.len());
    println!("skipped feature: {}", stats.skipped_feature);
    println!("skipped module:  {}", stats.skipped_module);
    for path in stats.panicked.iter().take(10) {
        println!("  PANIC {}", path.display());
    }
    if stats.panicked.len() > 10 {
        println!("  ... and {} more panics", stats.panicked.len() - 10);
    }
    // one line per failure, for offline aggregation
    for (path, err) in &stats.fail {
        println!("  FAIL {}: {err}", path.display());
    }
    if !stats.fail.is_empty() || !stats.panicked.is_empty() {
        return 1;
    }
    0
}
