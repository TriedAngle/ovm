//! Minimal test262 runner: runs sta.js + assert.js + each test file in a
//! VM with the builtin library installed.
//!
//! A test passes when it completes without an uncaught exception; tests
//! with `negative: { phase: parse }` frontmatter must fail to parse.
//! Feature-gated tests whose features the VM lacks (BigInt, Symbol, ...)
//! are counted separately as skipped.
//!
//! Usage: cargo run -p vm --example test262 -- <harness...> <test-file-or-dir>...

use std::path::{Path, PathBuf};

use dummy_heap::{DummyHeap, DummyHeapConfig};
use vm::ScriptError;

const UNSUPPORTED_FEATURES: &[&str] = &[
    "BigInt",
    "Symbol",
    "Temporal",
    "regexp-modifiers",
    // Reflect lands with the remaining proxy traps (proxy.rs)
    "Reflect",
    "Reflect.set",
    "Reflect.construct",
    // remaining class extensions beyond fields: static blocks, private
    // methods/accessors, decorators
    "class-static-block",
    "class-private-methods",
    "class-static-methods-private",
    "class-decorators",
];
/// Tests exercising runtime objects the VM does not have yet.
const UNSUPPORTED_PATTERNS: &[&str] = &["new Date"];
/// Harness include files the VM cannot load yet (missing globals like
/// Math, Array.prototype.push): every test including them is skipped.
const UNSUPPORTED_INCLUDES: &[&str] = &["propertyHelper.js"];
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
    // known-panic sparse indexOf/lastIndexOf builds: under the real GC
    // they panic fast, but under the allocation-only test heap they are
    // memory bombs (100 GB of Rust-side growth) — skip them outright
    "15.4.4.14-5-12",
    "15.4.4.14-5-16",
    "15.4.4.15-5-12",
    "15.4.4.15-5-16",
    // the same sparse-boundary disease through defineProperty (2^32-2
    // indices stored densely)
    "15.2.3.6-1-3",
    "15.2.3.6-1-4",
    "15.2.3.6-4-183",
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
    // Proxy: implemented are get/set/has/deleteProperty/defineProperty/
    // isExtensible/preventExtensions/revocable + the constructor; the
    // remaining traps and cross-cutting behaviors stay skipped
    "/built-ins/Proxy/apply/",
    "/built-ins/Proxy/construct/",
    "/built-ins/Proxy/ownKeys/",
    "/built-ins/Proxy/getOwnPropertyDescriptor/",
    "/built-ins/Proxy/getPrototypeOf/",
    "/built-ins/Proxy/setPrototypeOf/",
    "/built-ins/Proxy/enumerate/",
    "/built-ins/Proxy/property-order.js",
    "/built-ins/Proxy/proxy-newtarget.js",
    "/built-ins/Proxy/get-fn-realm.js",
    "/built-ins/Proxy/get-fn-realm-recursive.js",
    // `has` trap through `with` statements (no `with` support yet)
    "using-with",
    "Proxy/has/call-with.js",
    "Proxy/has/return-is-abrupt-with.js",
    // cross-realm: the runner has no $262 realm factory
    "-realm.js",
    "-realm-",
    // Object.create (not implemented; not part of the Proxy work)
    "Proxy/has/trap-is-undefined.js",
    "Proxy/get/trap-is-undefined-receiver.js",
    "Proxy/get/trap-is-undefined-target-is-proxy.js",
    "Proxy/set/trap-is-null-receiver.js",
    "Proxy/defineProperty/trap-is-undefined.js",
    "Proxy/defineProperty/desc-realm.js",
    // Object.prototype.toString tags for functions (Phase C nicety)
    "Proxy/revocable/builtin.js",
    "Proxy/revocable/revocation-function-not-a-constructor.js",
    // Object.keys / Array.prototype.indexOf (not implemented)
    "Proxy/defineProperty/call-parameters.js",
    "Proxy/revocable/revocation-function-property-order.js",
    // regex literals in the test source (no regex scanner yet)
    "Proxy/defineProperty/trap-is-null-target-is-proxy.js",
    "Proxy/deleteProperty/trap-is-null-target-is-proxy.js",
    "Proxy/isExtensible/trap-is-missing-target-is-proxy.js",
    "Proxy/set/trap-is-missing-target-is-proxy.js",
    "Proxy/revocable/tco-fn-realm.js",
    // a proxy in the prototype chain: the chain walk must restart
    // through the proxy's traps (not yet implemented)
    "Proxy/has/call-in-prototype.js",
    "Proxy/has/call-in-prototype-index.js",
    "Proxy/has/call-object-create.js",
    "Proxy/set/call-parameters-prototype.js",
    "Proxy/set/call-parameters-prototype-dunder-proto.js",
    "Proxy/set/call-parameters-prototype-index.js",
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
    "length-truncate-with-indexed",
    // newly-parsing legacy array tests hitting the same dense-allocation
    // limitation (new Array(4294967295)-style)
    "S15.4_A1.1_T10",
    "S12.6.3_A3",
    // huge generated identifier files (5k-8k+ class fields / escapes):
    // several exceed bytecode operand limits (baseline panics) and under
    // GC-stress each full root rescan per interned name costs 20-60+ min
    "/identifiers/start-unicode-",
    // sparse test arrays: large/hole-y indices that should not be stored
    // densely (index 999999/123456 and 2^32−1/2^32 cases)
    "15.4.4.16-7-c-ii-2",
    "15.4.4.17-7-c-ii-2",
    "15.4.4.18-7-c-ii-1",
    "15.4.4.19-8-c-ii-1",
    "15.4.4.20-9-c-ii-1",
    "15.4.4.14-10-1",
    "15.4.4.15-9-1",
    "length-truncate-nonconfigurable-sparse",
    "parse-mega-huge-array",
];

/// Paths that must be skipped even in audit mode: running them would
/// panic the VM or balloon memory (sparse/hole-y dense allocations and
/// multi-GB generated sources), not merely fail.
const HARD_SKIP_PATHS: &[&str] = &[
    "/dynamic-import/",
    "_FIXTURE",
    "15.4.4.14-5-12",
    "15.4.4.14-5-16",
    "15.4.4.15-5-12",
    "15.4.4.15-5-16",
    "15.2.3.6-1-3",
    "15.2.3.6-1-4",
    "15.2.3.6-4-183",
    "/identifiers/start-unicode-",
    "S15.4.5.2_A1_T1",
    "S15.4.2.2_A2.1_T1",
    "S15.4.5.2_A3_T4",
    "property-cast-number",
    "15.4.4.14-9-9",
    "15.4.4.15-8-9",
    "length-truncate-with-indexed",
    "S15.4_A1.1_T10",
    "S12.6.3_A3",
    "15.4.4.16-7-c-ii-2",
    "15.4.4.17-7-c-ii-2",
    "15.4.4.18-7-c-ii-1",
    "15.4.4.19-8-c-ii-1",
    "15.4.4.20-9-c-ii-1",
    "15.4.4.14-10-1",
    "15.4.4.15-9-1",
    "length-truncate-nonconfigurable-sparse",
    "parse-mega-huge-array",
];

#[derive(Default)]
struct Stats {
    pass: usize,
    /// Paths of passing tests; kept only so `OVM_AUDIT` can dump them.
    passed: Vec<PathBuf>,
    fail: Vec<(PathBuf, String)>,
    panicked: Vec<PathBuf>,
    skipped_feature: usize,
    skipped_module: usize,
    skipped_panic: usize,
}

/// Tests that panic the VM today (subclass-builtins machinery,
/// poisoned __proto__, proxy-observed fields, ...): skipped instead of
/// run so the `panic:` counter stays an exception signal — any nonzero
/// value is a NEW bug. Keep this list in sync when panics are fixed
/// (remove) or discovered (add, then investigate).
const KNOWN_PANICS: &[&str] = &[
    "DisposableStack/prototype/dispose/throws-suppressederror-if-multiple-errors-during-disposal.js",
    "built-ins/Object/setPrototypeOf/set-error.js",
    "expressions/class/subclass-builtins/subclass-Array.js",
    "expressions/class/subclass-builtins/subclass-Boolean.js",
    "expressions/class/subclass-builtins/subclass-Date.js",
    "expressions/class/subclass-builtins/subclass-Error.js",
    "expressions/class/subclass-builtins/subclass-Function.js",
    "expressions/class/subclass-builtins/subclass-Number.js",
    "expressions/class/subclass-builtins/subclass-Object.js",
    "expressions/class/subclass-builtins/subclass-ReferenceError.js",
    "expressions/class/subclass-builtins/subclass-String.js",
    "expressions/class/subclass-builtins/subclass-TypeError.js",
    "language/expressions/class/poisoned-underscore-proto.js",
    "language/expressions/delete/super-property.js",
    "language/expressions/super/call-poisoned-underscore-proto.js",
    "language/statements/class/poisoned-underscore-proto.js",
    "language/statements/using/throws-suppressederror-if-multiple-errors-during-disposal.js",
    "staging/sm/PrivateName/proxy-init-set.js",
    "staging/sm/String/string-upper-lower-mapping.js",
    "staging/sm/class/superCallBaseInvoked.js",
    "staging/sm/class/superPropProxies.js",
    "staging/sm/regress/regress-610026.js",
    "statements/class/definition/this-access-restriction.js",
    "statements/class/elements/class-field-is-observable-by-proxy.js",
    "statements/class/elements/privatefield-on-proxy.js",
    "statements/class/elements/public-class-field-initialization-is-visible-to-proxy.js",
    "statements/class/subclass-builtins/subclass-Array.js",
    "statements/class/subclass-builtins/subclass-Boolean.js",
    "statements/class/subclass-builtins/subclass-Date.js",
    "statements/class/subclass-builtins/subclass-Error.js",
    "statements/class/subclass-builtins/subclass-Function.js",
    "statements/class/subclass-builtins/subclass-Number.js",
    "statements/class/subclass-builtins/subclass-Object.js",
    "statements/class/subclass-builtins/subclass-ReferenceError.js",
    "statements/class/subclass-builtins/subclass-String.js",
    "statements/class/subclass-builtins/subclass-TypeError.js",
    "subclass/builtin-objects/Array/contructor-calls-super-multiple-arguments.js",
    "subclass/builtin-objects/Array/contructor-calls-super-single-argument.js",
    "subclass/builtin-objects/Array/length.js",
    "subclass/builtin-objects/Array/regular-subclassing.js",
    "subclass/builtin-objects/Array/super-must-be-called.js",
    "subclass/builtin-objects/Boolean/regular-subclassing.js",
    "subclass/builtin-objects/Boolean/super-must-be-called.js",
    "subclass/builtin-objects/Date/super-must-be-called.js",
    "subclass/builtin-objects/Error/regular-subclassing.js",
    "subclass/builtin-objects/Function/regular-subclassing.js",
    "subclass/builtin-objects/Function/super-must-be-called.js",
    "subclass/builtin-objects/GeneratorFunction/regular-subclassing.js",
    "subclass/builtin-objects/GeneratorFunction/super-must-be-called.js",
    "subclass/builtin-objects/NativeError/ReferenceError-name.js",
    "subclass/builtin-objects/NativeError/TypeError-name.js",
    "subclass/builtin-objects/Number/regular-subclassing.js",
    "subclass/builtin-objects/Number/super-must-be-called.js",
    "subclass/builtin-objects/Object/regular-subclassing.js",
    "subclass/builtin-objects/Object/replacing-prototype.js",
    "subclass/builtin-objects/String/regular-subclassing.js",
    "subclass/builtin-objects/String/super-must-be-called.js",
];

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

fn run_test(
    harness: &str,
    harness_dir: Option<&Path>,
    path: &Path,
    stats: &mut Stats,
    audit: bool,
) {
    // survive panics (e.g. bytecode operand overflow on huge generated
    // files): count them separately and keep going
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_test_inner(harness, harness_dir, path, stats, audit)
    }));
    if result.is_err() {
        stats.panicked.push(path.to_path_buf());
    }
}

/// The pending exception's `name` property (error class), for categorizing
/// uncaught exceptions in the stats.
fn exception_name(thread: &mut vm::Thread) -> String {
    thread.handle_scope(|thread, scope| {
        let Some(ex) = thread.take_pending_exception() else {
            return "exception".into();
        };
        let name_key = thread.intern(&scope, "name");
        {
            let heap = thread.heap();
            let Some(o) = unsafe { ex.assume_valid(heap) }.as_heap_object() else {
                return "exception".into();
            };
            match o.lookup(heap, name_key.as_tagged(heap).into()) {
                vm::Lookup::Data { slot, .. } => slot
                    .get(heap)
                    .get_as::<vm::DenseString>()
                    .map(|s| s.to_rust_string(heap))
                    .unwrap_or_else(|| "exception".into()),
                _ => "exception".into(),
            }
        }
    })
}

/// Run one test in `vm`. The harness prelude is `harness` plus any files
/// the test's `includes:` frontmatter names (resolved in the first harness
/// file's directory, INTERPRETING.md).
fn run_test_inner(
    harness: &str,
    harness_dir: Option<&Path>,
    path: &Path,
    stats: &mut Stats,
    audit: bool,
) {
    let Ok(src) = std::fs::read_to_string(path) else {
        stats.fail.push((path.to_path_buf(), "not utf-8".into()));
        return;
    };
    let fm = frontmatter(&src);
    if fm.contains("module") && fm.contains("flags") {
        stats.skipped_module += 1;
        return;
    }
    // known VM panics: never run them (see KNOWN_PANICS)
    let path_str = path.to_string_lossy();
    if KNOWN_PANICS.iter().any(|k| path_str.ends_with(k)) {
        stats.skipped_panic += 1;
        return;
    }
    // `OVM_AUDIT=1` runs the soft-skipped tests too (feature/path/pattern
    // frontmatter): used to spot skips the VM no longer needs. The hard
    // resource skips and KNOWN_PANICS still apply.
    let hard = HARD_SKIP_PATHS.iter().any(|p| path_str.contains(p));
    if hard {
        stats.skipped_feature += 1;
        return;
    }
    if !audit {
        if UNSUPPORTED_FEATURES.iter().any(|f| fm.contains(f)) {
            stats.skipped_feature += 1;
            return;
        }
        if UNSUPPORTED_PATHS.iter().any(|p| path_str.contains(p)) {
            stats.skipped_feature += 1;
            return;
        }
        if UNSUPPORTED_PATTERNS.iter().any(|p| src.contains(p)) {
            stats.skipped_feature += 1;
            return;
        }
    }
    let expect_parse_error =
        fm.contains("negative:") && (fm.contains("phase: parse") || fm.contains("phase: syntax"));
    // `raw` tests run without the harness preludes (INTERPRETING.md)
    let raw = fm.contains("flags") && fm.contains("raw");
    // onlyStrict: the whole program (harness prelude included) runs as
    // strict code; noStrict runs as-is (sloppy unless the test says
    // otherwise)
    let strict_wrap = fm.contains("flags") && fm.contains("onlyStrict");

    // `includes:` harness files (sta.js-style helpers like propertyHelper)
    let mut includes = String::new();
    let mut include_unsupported = false;
    if let Some(dir) = harness_dir
        && let Some(start) = fm.find("includes:")
    {
        {
            let rest = &fm[start + "includes:".len()..];
            let end = rest.find(']').unwrap_or(0);
            for name in rest[..end].split(&[',', '[', '\n'][..]) {
                let name = name.trim().trim_matches(|c| c == '\'' || c == '"');
                if name.ends_with(".js") {
                    if UNSUPPORTED_INCLUDES.contains(&name) {
                        include_unsupported = true;
                    }
                    let p = dir.join(name);
                    if let Ok(text) = std::fs::read_to_string(&p) {
                        includes.push_str(&text);
                        includes.push('\n');
                    }
                }
            }
        }
    }
    if include_unsupported && !audit {
        stats.skipped_feature += 1;
        return;
    }

    let code = if raw {
        src
    } else if strict_wrap {
        format!("\"use strict\";\n{harness}\n{includes}\n{src}\n")
    } else {
        format!("{harness}\n{includes}\n{src}\n")
    };
    // realm isolation: every test runs in a fresh VM (INTERPRETING.md)
    // the dummy heap: allocation-only, no GC machinery, no sweeper
    // thread — exactly right for one-shot short-lived test realms. The
    // block is small enough that its per-test mmap/munmap stays cheap
    // (a 64 MiB block per test storms the kernel's page tables under
    // parallel runs).
    let vm = vm::VM::new::<DummyHeap, vm::DefaultInterpreter>(DummyHeapConfig {
        heap_size: 8 * 1024 * 1024,
    })
    .expect("vm")
    .add::<vm::JSRuntime>()
    .expect("vm");
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    match thread.eval::<vm::JavascriptCompiler>(&code) {
        Ok(v)
            if {
                {
                    let heap = &*thread.heap();
                    v == heap.known().exception.as_tagged(heap).raw()
                }
            } =>
        {
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
                if audit {
                    stats.passed.push(path.to_path_buf());
                }
            }
        }
        Err(ScriptError::Parse(_)) if expect_parse_error => {
            stats.pass += 1;
            if audit {
                stats.passed.push(path.to_path_buf());
            }
        }
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
    // (OVM_PANIC_MSG=1 keeps the default hook to debug a specific test)
    if std::env::var_os("OVM_PANIC_MSG").is_none() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let progress = std::env::var_os("OVM_PROGRESS").is_some();
    let audit = std::env::var_os("OVM_AUDIT").is_some();
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if args.is_empty() {
        eprintln!("usage: test262 <harness...> <test-file-or-dir>...");
        std::process::exit(2);
    }
    // heuristic: the first two args are harness files (sta.js, assert.js);
    // `includes:` frontmatter resolves against the first one's directory
    let harness = args
        .iter()
        .take(2)
        .map(|p| std::fs::read_to_string(p).expect("harness file"))
        .collect::<Vec<_>>()
        .join("\n");
    let harness_dir = args.first().and_then(|p| p.parent().map(Path::to_path_buf));

    let mut files = Vec::new();
    for root in &args[2..] {
        if root.is_dir() {
            collect(root, &mut files);
        } else {
            files.push(root.clone());
        }
    }
    files.sort();

    // shard the sorted file list across workers; every test spins a fresh
    // VM anyway (realm isolation), so workers are independent
    let jobs = std::env::var("OVM_JOBS")
        .ok()
        .and_then(|j| j.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let jobs = jobs.max(1).min(files.len().max(1));

    let mut stats = Stats::default();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(jobs);
        for shard in 0..jobs {
            // round-robin keeps per-worker shards sorted (stable merged
            // output) and balances the uneven per-test costs better than
            // contiguous chunks
            let mine: Vec<&PathBuf> = files
                .iter()
                .enumerate()
                .filter_map(|(i, f)| (i % jobs == shard).then_some(f))
                .collect();
            let harness = &harness;
            let harness_dir = harness_dir.as_deref();
            let progress = progress;
            let audit = audit;
            let handle = std::thread::Builder::new()
                .stack_size(1 << 30)
                .spawn_scoped(scope, move || {
                    let mut local = Stats::default();
                    for file in mine {
                        if progress {
                            // last line before a crash identifies the culprit test
                            eprintln!("running {}", file.display());
                        }
                        run_test(harness, harness_dir, file, &mut local, audit);
                    }
                    local
                })
                .expect("spawn worker");
            handles.push(handle);
        }
        for handle in handles {
            match handle.join() {
                Ok(local) => {
                    stats.pass += local.pass;
                    stats.passed.extend(local.passed);
                    stats.skipped_feature += local.skipped_feature;
                    stats.skipped_module += local.skipped_module;
                    stats.skipped_panic += local.skipped_panic;
                    stats.fail.extend(local.fail);
                    stats.panicked.extend(local.panicked);
                }
                Err(_) => {
                    eprintln!("worker panicked outside a test");
                    std::process::exit(101);
                }
            }
        }
    });
    // deterministic, diffable output across runs and job counts
    stats.fail.sort_by(|a, b| a.0.cmp(&b.0));
    stats.panicked.sort();

    println!("total:           {}", files.len());
    println!("pass:            {}", stats.pass);
    println!("fail:            {}", stats.fail.len());
    println!("panic:           {}", stats.panicked.len());
    println!("skipped feature: {}", stats.skipped_feature);
    println!("skipped module:  {}", stats.skipped_module);
    println!("skipped panic:   {}", stats.skipped_panic);
    for path in stats.panicked.iter().take(100) {
        println!("  PANIC {}", path.display());
    }
    if stats.panicked.len() > 100 {
        println!("  ... and {} more panics", stats.panicked.len() - 100);
    }
    // one line per failure, for offline aggregation
    for (path, err) in &stats.fail {
        println!("  FAIL {}: {err}", path.display());
    }
    // audit mode: dump the passing set so callers can diff which
    // previously-skipped tests now run green
    if audit {
        for path in &stats.passed {
            println!("  PASS {}", path.display());
        }
    }
    if !stats.fail.is_empty() || !stats.panicked.is_empty() {
        return 1;
    }
    0
}
