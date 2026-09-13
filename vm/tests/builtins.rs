//! Builtins + direct eval end-to-end.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::VM;
use vm::Value;

fn vm() -> VM {
    VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap()
}

fn run_smi(vm: &VM, src: &str) -> i64 {
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    result.to_i64().unwrap()
}

fn run_str(vm: &VM, src: &str) -> String {
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    thread.heap().no_gc(|nogc| {
        let s = result.get_as::<vm::VMString>(nogc).expect("string result");
        String::from_utf8(s.as_slice(nogc).to_vec()).unwrap()
    })
}

fn run_bool(vm: &VM, src: &str) -> bool {
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    if result == thread.heap().known().true_object.value() {
        true
    } else if result == thread.heap().known().false_object.value() {
        false
    } else {
        panic!("expected boolean, got {result:?}");
    }
}

fn run_value(vm: &VM, src: &str) -> (Value, vm::Thread) {
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    (result, thread)
}

#[test]
fn eval_of_plain_expression() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "eval('1 + 1');"), 2);
}

#[test]
fn eval_sees_caller_locals() {
    let vm = vm();
    // x is context-allocated (calls_eval); eval must resolve it dynamically
    assert_eq!(run_smi(&vm, "var x = 41; eval('x + 1');"), 42);
    assert_eq!(run_smi(&vm, "var x = 1; x = 5; eval('x');"), 5);
}

#[test]
fn eval_can_write_caller_locals() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "var x = 1; eval('x = 9'); x;"), 9);
}

#[test]
fn eval_inside_functions() {
    let vm = vm();
    assert_eq!(
        run_smi(&vm, "function f(a) { return eval('a + 1'); } f(10);"),
        11
    );
}

#[test]
fn eval_string_conversion_and_whitespace() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "eval(' 1 + 2 ');"), 3);
    assert_eq!(run_smi(&vm, "eval(42);"), 42);
}

#[test]
fn eval_returns_last_statement_value() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "eval('1; 2; 3;');"), 3);
}

#[test]
fn eval_thrown_exceptions_propagate() {
    let vm = vm();
    let (result, thread) = run_value(&vm, "try { eval('throw 7;'); } catch (e) { e; }");
    assert_eq!(result.to_i64().unwrap(), 7);
    assert!(!thread.has_pending_exception());
}

#[test]
fn eval_syntax_error_throws() {
    let vm = vm();
    let (result, thread) = run_value(&vm, "try { eval('var = ;'); } catch (e) { 1; }");
    assert_eq!(result.to_i64().unwrap(), 1);
    assert!(!thread.has_pending_exception());
}

#[test]
fn number_constructor() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "Number('5');"), 5);
    assert_eq!(run_smi(&vm, "Number(true);"), 1);
    assert_eq!(run_smi(&vm, "Number();"), 0);
    assert_eq!(run_str(&vm, "typeof Number(1);"), "number");
    assert_eq!(run_str(&vm, "typeof new Number(1);"), "object");
}

#[test]
fn boxed_numbers_convert_in_addition() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "new Number(1) + 1;"), 2);
    assert_eq!(run_smi(&vm, "1 + new Number(1);"), 2);
    assert_eq!(run_smi(&vm, "new Number(1) + new Number(1);"), 2);
    assert_eq!(run_smi(&vm, "new Number(1).valueOf();"), 1);
    assert_eq!(run_str(&vm, "new Number(1).toString();"), "1");
}

#[test]
fn eval_sees_lexicals_across_function_scopes() {
    let vm = vm();
    // calls_eval must propagate up the whole visible scope chain, so
    // lexicals that eval can only reach dynamically are context-allocated
    assert_eq!(
        run_smi(&vm, "let x = 42; function f() { return eval('x'); } f();"),
        42
    );
    assert_eq!(
        run_smi(
            &vm,
            "function g() { let y = 7; function f() { return eval('y'); } return f(); } g();"
        ),
        7
    );
}

#[test]
fn eval_completion_values() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "eval('1; 2;');"), 2);
    assert_eq!(run_smi(&vm, "eval('if (true) 3;');"), 3);
    assert_eq!(
        run_smi(&vm, "eval('var i = 0; while (i < 3) { i++; 9; }');"),
        9,
        "loop completion is the last body value"
    );
    // declarations (and blocks of declarations) produce no value
    let (result, mut thread) = run_value(&vm, "eval('{ let x = 1; }');");
    assert_eq!(result, thread.heap().known().undefined.value());
    let (result, mut thread) = run_value(&vm, "eval('function fn() {}{}');");
    assert_eq!(result, thread.heap().known().undefined.value());
    let (result, mut thread) = run_value(&vm, "eval('var x = 1;');");
    assert_eq!(result, thread.heap().known().undefined.value());
}

#[test]
fn array_constructor_and_prototype() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "var a = Array(); a.length;"), 0);
    assert_eq!(run_smi(&vm, "var a = new Array(3); a.length;"), 3);
    assert_eq!(run_smi(&vm, "var a = Array(1, 2, 3); a[2] + a.length;"), 6);
    assert!(run_bool(
        &vm,
        "Object.getPrototypeOf([]) === Array.prototype;"
    ));
    assert!(run_bool(&vm, "Array.prototype.constructor === Array;"));
    assert_eq!(run_str(&vm, "typeof Array;"), "function");
    assert!(run_bool(
        &vm,
        "Object.getPrototypeOf({}) === Object.prototype;"
    ));
}

#[test]
fn boolean_constructor() {
    let vm = vm();
    assert!(run_bool(&vm, "Boolean(1);"));
    assert_eq!(run_str(&vm, "typeof Boolean(1);"), "boolean");
    assert_eq!(run_str(&vm, "typeof new Boolean(1);"), "object");
    assert_eq!(run_smi(&vm, "true + 1;"), 2);
    assert_eq!(run_smi(&vm, "new Boolean(true) + 1;"), 2);
    assert_eq!(run_smi(&vm, "1 + new Boolean(true);"), 2);
    assert_eq!(run_smi(&vm, "new Number(1) + new Boolean(true);"), 2);
    assert_eq!(run_str(&vm, "new Boolean(true).toString();"), "true");
}

#[test]
fn error_objects_have_name_message_constructor() {
    let vm = vm();
    assert_eq!(run_str(&vm, "new Error('boom').message;"), "boom");
    assert_eq!(run_str(&vm, "new Error().name;"), "Error");
    assert_eq!(run_str(&vm, "new TypeError().name;"), "TypeError");
    assert_eq!(run_str(&vm, "new Error('x').toString();"), "Error: x");
    assert!(run_bool(&vm, "(new Error()) instanceof Error;"));
    assert!(run_bool(&vm, "(new TypeError()) instanceof TypeError;"));
    assert!(run_bool(&vm, "(new TypeError()) instanceof Error;"));
    assert!(!run_bool(&vm, "(new Error()) instanceof TypeError;"));
}

#[test]
fn function_prototypes_and_constructor_link() {
    let vm = vm();
    // user function gets a .prototype with .constructor pointing back
    assert!(run_bool(
        &vm,
        "function F() {} F.prototype.constructor === F;"
    ));
    assert!(run_bool(
        &vm,
        "function F() {} var o = new F(); o instanceof F;"
    ));
    assert!(run_bool(
        &vm,
        "function F() {} (new F()).constructor === F;"
    ));
}

#[test]
fn vm_error_materialization_uses_type_error_chain() {
    let vm = vm();
    // TypeError thrown by the VM (e.g. ToPrimitive failure) must have a
    // constructor chain: .constructor === TypeError
    assert!(run_bool(
        &vm,
        "try { null.foo(); } catch (e) { e instanceof TypeError; }"
    ));
}

#[test]
fn assert_harness_snippets() {
    let vm = vm();
    // the patterns assert.js relies on
    assert_eq!(
        run_smi(
            &vm,
            "function f(v) { switch (v === null ? 'null' : typeof v) { case 'number': return 1; case 'string': return 2; } } f('x');"
        ),
        2
    );
    assert!(run_bool(
        &vm,
        "function isNegativeZero(value) { return value === 0 && 1 / value === -Infinity; } isNegativeZero(-0.0);"
    ));
    assert_eq!(
        run_smi(
            &vm,
            "function g(a, b) { if (a === b) { if (a !== 0 || 1 / a === 1 / b) { return 1; } } return a !== a && b !== b ? 2 : 0; } g(0 / 0, 0 / 0);"
        ),
        2,
        "NaN equality"
    );
    assert_eq!(run_str(&vm, "'' + Infinity;"), "Infinity");
    assert_eq!(run_str(&vm, "'' + NaN;"), "NaN");
    assert_eq!(run_str(&vm, "'' + undefined;"), "undefined");
}

#[test]
fn harness_style_throw_and_catch() {
    let vm = vm();
    assert_eq!(
        run_str(
            &vm,
            "function Test262Error(message) { this.message = message || ''; }
             try { throw new Test262Error('#1'); } catch (e) { e.message; }"
        ),
        "#1"
    );
}
