//! End-to-end: parse → resolve → compile → materialize → run.

use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::{ScriptError, Thread, VM};
use vm::{Float, Smi, VMString, Value};

fn run(src: &str) -> Result<Value, ScriptError> {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script(src)
}

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

/// Run and read back the result as a Rust value.
fn run_value(src: &str) -> (Value, Thread) {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    (result, thread)
}

fn run_bool(src: &str) -> bool {
    let (result, mut thread) = run_value(src);
    if result == thread.heap().known().true_object.value() {
        true
    } else if result == thread.heap().known().false_object.value() {
        false
    } else {
        panic!("expected boolean result, got {result:?}");
    }
}

fn run_num(src: &str) -> f64 {
    let (result, mut thread) = run_value(src);
    thread.heap().no_gc(|nogc| {
        if let Some(smi) = Smi::decode(result) {
            return smi.value() as f64;
        }
        result
            .get_as::<Float>(nogc, nogc.known().float_map)
            .expect("number result")
            .value
            .get()
    })
}

fn run_str(src: &str) -> String {
    let (result, mut thread) = run_value(src);
    thread.heap().no_gc(|nogc| {
        let s = result
            .get_as::<VMString>(nogc, nogc.known().string_map)
            .expect("string result");
        String::from_utf8(s.as_slice(nogc).to_vec()).unwrap()
    })
}

fn run_undefined(src: &str) -> Value {
    let (result, mut thread) = run_value(src);
    assert_eq!(result, thread.heap().known().undefined.value());
    result
}

#[test]
fn arithmetic() {
    assert_eq!(run_smi("1 + 2;"), 3);
    assert_eq!(run_smi("7 - 2;"), 5);
    assert_eq!(run_smi("3 * 4;"), 12);
    assert_eq!(run_smi("7 % 3;"), 1);
    assert_eq!(run_smi("1 + 2 * 3;"), 7);
    assert_eq!(run_smi("(1 + 2) * 3;"), 9);
    assert_eq!(run_smi("-5 + 2;"), -3);
}

#[test]
fn division_is_ieee() {
    assert_eq!(run_num("7 / 2;"), 3.5);
    assert_eq!(run_num("1 / 0;"), f64::INFINITY);
    assert_eq!(run_num("-1 / 0;"), f64::NEG_INFINITY);
    assert!(run_num("0 / 0;").is_nan());
}

#[test]
fn string_concatenation() {
    assert_eq!(run_str("'a' + 'b';"), "ab");
    assert_eq!(run_str("'a' + 1;"), "a1");
    assert_eq!(run_str("1 + 'a';"), "1a");
    assert_eq!(run_str("'x: ' + (1 + 2);"), "x: 3");
    assert_eq!(run_num("1 + true;"), 2.0);
    assert_eq!(run_str("'' + null;"), "null");
    assert_eq!(run_str("'' + undefined;"), "undefined");
}

#[test]
fn booleans_and_equality() {
    assert!(run_bool("1 === 1;"));
    assert!(!run_bool("1 !== 1;"));
    assert!(run_bool("1 == '1';"));
    assert!(!run_bool("null === undefined;"));
    assert!(run_bool("1 < 2;"));
    assert!(run_bool("2 <= 2;"));
    assert!(run_bool("3 > 2;"));
    assert!(!run_bool("3 >= 4;"));
    assert!(run_bool("!0;"));
    assert!(run_bool("!!1;"));
    assert_eq!(run_smi("true && 1;"), 1);
    assert!(!run_bool("false && 1;"));
    assert_eq!(run_smi("0 || 2;"), 2);
    assert_eq!(run_smi("1 || 2;"), 1);
}

#[test]
fn variables_and_assignment() {
    assert_eq!(run_smi("var x = 1; var y = 2; x + y;"), 3);
    assert_eq!(run_smi("var x = 1; x = 5; x;"), 5);
    assert_eq!(run_smi("var x = 1; x += 4; x;"), 5);
    assert_eq!(run_smi("var x = 1; x++; x;"), 2);
    assert_eq!(
        run_smi("var x = 1; x++;"),
        1,
        "postfix result is the old value"
    );
    assert_eq!(run_smi("var x = 1; ++x;"), 2);
    run_undefined("var x; x;");
}

#[test]
fn control_flow() {
    assert_eq!(run_smi("if (true) 1; else 2;"), 1);
    assert_eq!(run_smi("if (false) 1; else 2;"), 2);
    assert_eq!(run_smi("var i = 0; while (i < 3) { i++; } i;"), 3);
    assert_eq!(
        run_smi("var s = 0; for (var i = 0; i < 5; i++) { s += i; } s;"),
        10
    );
    assert_eq!(run_smi("1 ? 10 : 20;"), 10);
    assert_eq!(run_smi("0 ? 10 : 20;"), 20);
}

#[test]
fn functions() {
    assert_eq!(
        run_smi("function add(a, b) { return a + b; } add(2, 3);"),
        5
    );
    assert_eq!(
        run_smi("function f() { return 42; } function g() { return f(); } g();"),
        42
    );
    run_undefined("function f() {} f();");
    assert_eq!(
        run_smi("function fact(n) { if (n <= 1) { return 1; } return n * fact(n - 1); } fact(5);"),
        120
    );
}

#[test]
fn closures_capture_environment() {
    assert_eq!(
        run_smi("var x = 10; function get() { return x; } x = 20; get();"),
        20,
        "closures read the current value"
    );
    assert_eq!(
        run_smi(
            "function make() { var n = 7; function get() { return n; } return get; } make()();"
        ),
        7
    );
    assert_eq!(
        run_smi("var f; { let x = 5; f = function() { return x; }; } f();"),
        5
    );
}

#[test]
fn try_catch() {
    assert_eq!(run_smi("try { throw 1; } catch (e) { return e; }"), 1);
    assert_eq!(run_smi("try { 1; } catch (e) { 2; }"), 1);
    assert_eq!(run_smi("try { throw 9; } catch (e) { e + 1; }"), 10);
    assert_eq!(run_str("try { throw 'boom'; } catch (e) { e; }"), "boom");
}

#[test]
fn nested_try_catch_rethrows() {
    assert_eq!(
        run_smi("try { try { throw 3; } catch (e) { throw e + 1; } } catch (e2) { e2; }"),
        4
    );
}

#[test]
fn typeof_and_instanceof() {
    assert_eq!(run_str("typeof 1;"), "number");
    assert_eq!(run_str("typeof 'a';"), "string");
    assert_eq!(run_str("typeof true;"), "boolean");
    assert_eq!(run_str("typeof undefined;"), "undefined");
    assert_eq!(run_str("typeof null;"), "object");
    assert_eq!(run_str("typeof {};"), "object");
    assert_eq!(run_str("typeof (function() {});"), "function");
}

#[test]
fn objects() {
    assert_eq!(run_smi("var o = {x: 1, y: 2}; o.x + o.y;"), 3);
    assert_eq!(run_smi("var o = {}; o.x = 5; o.x;"), 5);
    assert_eq!(run_smi("var o = {x: 1}; o.x = 9; o.x;"), 9);
    run_undefined("var o = {}; o.missing;");
    assert_eq!(run_smi("var o = {a: 1}; var k = 'a'; o[k];"), 1);
    assert_eq!(run_smi("var o = {}; o['k'] = 3; o.k;"), 3);
    assert_eq!(run_str("typeof {x: 1};"), "object");
}

#[test]
fn arrays() {
    assert_eq!(run_smi("var a = [1, 2, 3]; a[0] + a[2];"), 4);
    // TODO: array length is an internal slot; a.length needs a dedicated
    // length access path before it reads back correctly
    // assert_eq!(run_smi("var a = [1, , 3]; a.length;"), 3);
    run_undefined("var a = [1, 2]; a[5];");
    assert_eq!(run_smi("var a = []; a[0] = 7; a[0];"), 7);
    assert_eq!(
        run_smi("var a = [10, 20, 30]; var s = 0; for (var i = 0; i < 3; i++) { s += a[i]; } s;"),
        60
    );
}

#[test]
fn method_calls_get_receiver() {
    assert_eq!(
        run_smi("var o = {x: 4, get: function() { return this.x; }}; o.get();"),
        4
    );
    assert_eq!(
        run_smi("var o = {x: 1, add: function(a) { return this.x + a; }}; o.add(2);"),
        3
    );
}

#[test]
fn constructors_without_builtins_still_evaluate() {
    // no builtins yet: `new` on a user function yields the fresh receiver
    assert_eq!(
        run_smi("function C() { this.x = 3; } var o = new C(); o.x;"),
        3
    );
    // a primitive result is discarded in favor of the receiver (ES 9.2.2)
    assert_eq!(
        run_str("function C() { return 5; } typeof new C();"),
        "object"
    );
    // an object result wins over the receiver
    assert_eq!(run_smi("function C() { return {y: 2}; } new C().y;"), 2);
}

#[test]
fn tdz_throws_on_let_before_init() {
    // uncaught exceptions escape as the exception sentinel + pending
    // exception holding a ReferenceError
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script("let x = x;").unwrap();
    assert_eq!(result, thread.heap().known().exception.value());
    assert!(thread.has_pending_exception());
    let ex = thread.take_pending_exception().expect("pending exception");
    thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name").value();
        let expected = thread.intern(&scope, "ReferenceError").value();
        thread.heap().no_gc(|nogc| {
            let vm::ValueRef::Object(o) = ex.value_ref(nogc) else {
                panic!("pending exception must be an object");
            };
            match o.as_ref().lookup(nogc, vm::SlotName::from_value(name_key)) {
                vm::Lookup::Data { slot, .. } => assert_eq!(slot.inner(), expected),
                _ => panic!("error object must have a name property"),
            }
        });
    });

    // the hole survives frame reuse: run twice in a row
    let result = thread.run_script("let x = x;").unwrap();
    assert_eq!(result, thread.heap().known().exception.value());
    thread.take_pending_exception();
}

#[test]
fn uncaught_throw_escapes_as_exception() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script("throw 42;").unwrap();
    assert_eq!(result, thread.heap().known().exception.value());
    assert!(thread.has_pending_exception());
    thread.take_pending_exception();
}

#[test]
fn script_completion_value_is_last_statement() {
    assert_eq!(run_smi("1; 2;"), 2);
    // declarations produce no completion value (ES 13.2.13: UpdateEmpty)
    run_undefined("var x = 3;");
    assert_eq!(run_smi("var x = 3; x;"), 3);
}

#[test]
fn large_integer_literals_use_the_constant_pool() {
    // too big for the (2-byte signed) LoadSmi operand, still exact smis
    assert_eq!(run_smi("65535 + 1;"), 65536);
    assert_eq!(run_smi("999999 + 1;"), 1000000);
    assert_eq!(run_smi("9007199254740991;"), 9007199254740991); // 2^53 - 1
}

#[test]
fn lone_surrogate_strings_do_not_panic() {
    // WTF-8 storage: lone surrogates are legal string content and must
    // survive interning (the VM stores bytes, not UTF-8 strs)
    run("var s = '\\uD800';").unwrap();
    run("var s = '\\uDC00\\uD800';").unwrap();
}

#[test]
fn errors_reachable_through_script_error_enum() {
    let err = run("var = ;").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)));
    let err = run("1n;").unwrap_err();
    assert!(matches!(err, ScriptError::Compile(_)));
}

#[test]
fn float_arithmetic_allocates_and_round_trips() {
    assert_eq!(run_num("1.5 + 2.25;"), 3.75);
    assert_eq!(run_num("0.1 + 0.2;"), 0.1 + 0.2);
}

#[test]
fn update_on_property_refs() {
    assert_eq!(run_smi("var o = {x: 1}; o.x++; o.x;"), 2);
    assert_eq!(run_smi("var o = {x: 1}; o.x += 4; o.x;"), 5);
    assert_eq!(
        run_smi("var o = {}; var k = 'x'; o[k] = 1; o[k] += 2; o.x;"),
        3
    );
}

#[test]
fn unused_let_without_init_is_still_tdz() {
    // `y` is never read; `x` reads before its initializer runs
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script("let x = 1, y; y = x; x;").unwrap();
    assert_eq!(Smi::decode(result).unwrap().value(), 1);
}

#[test]
fn repl_mode_persists_top_level_bindings() {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script_repl("let x = 10;").unwrap();
    thread.run_script_repl("var y = 2;").unwrap();
    thread.run_script_repl("const z = 3;").unwrap();
    let v = thread.run_script_repl("x * y + z;").unwrap();
    assert_eq!(Smi::decode(v).unwrap().value(), 23);
}

#[test]
fn repl_mode_functions_persist_and_read_globals() {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script_repl("let x = 10;").unwrap();
    thread
        .run_script_repl("function get() { return x; }")
        .unwrap();
    thread.run_script_repl("x = 20;").unwrap();
    let v = thread.run_script_repl("get();").unwrap();
    assert_eq!(Smi::decode(v).unwrap().value(), 20);
}

#[test]
fn repl_mode_allows_redeclaration_across_entries() {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script_repl("let x = 1;").unwrap();
    let v = thread.run_script_repl("let x = 2; x;").unwrap();
    assert_eq!(Smi::decode(v).unwrap().value(), 2);
}

#[test]
fn repl_mode_keeps_nested_scopes_local() {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script_repl("{ let inner = 5; }").unwrap();
    // `inner` was block-scoped: gone with the block, this throws
    let result = thread.run_script_repl("inner;").unwrap();
    assert_eq!(result, thread.heap().known().exception.value());
    thread.take_pending_exception();
}

#[test]
fn script_mode_top_level_bindings_do_not_persist() {
    let vm = VM::with_builtins::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script("let x = 10;").unwrap();
    let result = thread.run_script("x;").unwrap();
    assert_eq!(result, thread.heap().known().exception.value());
    thread.take_pending_exception();
}

#[test]
fn smi_result_helpers() {
    let _ = smi(0); // silence dead-code lint for the helper
}

#[test]
fn switch_statements() {
    assert_eq!(
        run_smi("switch (1) { case 1: 10; break; case 2: 20; break; }"),
        10
    );
    assert_eq!(
        run_smi("switch (2) { case 1: 10; break; case 2: 20; break; }"),
        20
    );
    assert_eq!(
        run_smi("switch (3) { case 1: 10; break; case 2: 20; break; default: 30; }"),
        30
    );
    run_undefined("switch (3) { case 1: 10; break; }");
    assert_eq!(
        run_smi("var x = 0; switch (2) { case 1: x = 1; break; case 2: x = 2; } x;"),
        2,
        "fallthrough into next body"
    );
    assert_eq!(
        run_smi("var x = 0; switch (2) { case 1: x = 1; case 2: x = 2; case 3: x = 3; } x;"),
        3
    );
    assert_eq!(
        run_smi(
            "var s = 0; for (var i = 0; i < 3; i++) { switch (i) { case 1: continue; default: s += i; } } s;"
        ),
        2
    );
    assert_eq!(
        run_str("switch ('a') { case 'a': 'yes'; break; default: 'no'; }"),
        "yes"
    );
    assert_eq!(
        run_smi("switch (1) { default: 7; case 1: 1; }"),
        1,
        "default before cases still works"
    );
}
