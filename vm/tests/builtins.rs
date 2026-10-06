//! Builtins + direct eval end-to-end.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::VM;
use vm::Value;

fn vm() -> VM {
    let vm = vm::VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    vm
}

fn run_smi(vm: &VM, src: &str) -> i64 {
    let mut thread = vm.attach();
    let result = thread.eval::<vm::JavascriptCompiler>(src).unwrap();
    result.to_i64().unwrap()
}

fn run_str(vm: &VM, src: &str) -> String {
    let mut thread = vm.attach();
    let result = thread.eval::<vm::JavascriptCompiler>(src).unwrap();
    {
        let heap = &*thread.heap();
        let s = unsafe { result.assume_valid(heap) }
            .get_as::<vm::DenseString>(heap)
            .expect("string result");
        s.to_rust_string(heap)
    }
}

fn run_bool(vm: &VM, src: &str) -> bool {
    let mut thread = vm.attach();
    let result = thread.eval::<vm::JavascriptCompiler>(src).unwrap();
    let heap = thread.heap();
    if result == heap.known().true_object.as_tagged(heap).raw() {
        true
    } else if result == heap.known().false_object.as_tagged(heap).raw() {
        false
    } else {
        panic!("expected boolean, got {result:?}");
    }
}

fn run_value(vm: &VM, src: &str) -> (Value, vm::Thread) {
    let mut thread = vm.attach();
    let result = thread.eval::<vm::JavascriptCompiler>(src).unwrap();
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
    // (each probe is scoped: an attached-but-idle thread would block
    // every stop-the-world collection of the threads still running)
    {
        let (result, mut thread) = run_value(&vm, "eval('{ let x = 1; }');");
        let heap = thread.heap();
        assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
    }
    {
        let (result, mut thread) = run_value(&vm, "eval('function fn() {}{}');");
        let heap = thread.heap();
        assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
    }
    {
        let (result, mut thread) = run_value(&vm, "eval('var x = 1;');");
        let heap = thread.heap();
        assert_eq!(result, heap.known().undefined.as_tagged(heap).raw());
    }
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

#[test]
fn bitwise_coerces_non_smi_operands() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "var x = 1.5; x | 0;"), 1);
    assert_eq!(run_smi(&vm, "var x = '12'; x | 0;"), 12);
    assert_eq!(run_smi(&vm, "var x = 5.5; var y = 3.2; x ^ y;"), 6);
    assert_eq!(run_smi(&vm, "var x = 2.9; x << 1;"), 4);
    assert_eq!(run_smi(&vm, "var x = 1.5; x >> 0;"), 1);
    assert_eq!(run_smi(&vm, "var x = -1; x >>> 0;"), 4294967295);
    assert_eq!(run_smi(&vm, "var x = 4294967296; x | 0;"), 0);
    assert_eq!(run_smi(&vm, "var x = 3 * 1.1 + 1; x >> 0;"), 4);
    assert_eq!(run_smi(&vm, "var x = NaN; x | 0;"), 0);
    assert_eq!(run_smi(&vm, "var x = null; x | 0;"), 0);
    assert_eq!(run_smi(&vm, "var x = true; x | 0;"), 1);
}

#[test]
fn date_format_and_parse_round_trip() {
    let vm = vm();
    assert_eq!(
        run_str(&vm, "new Date(0).toISOString();"),
        "1970-01-01T00:00:00.000Z"
    );
    assert_eq!(
        run_str(&vm, "new Date(0).toGMTString();"),
        "Thu, 01 Jan 1970 00:00:00 GMT"
    );
    assert_eq!(
        run_str(&vm, "new Date(0).toString();"),
        "Thu Jan 01 1970 00:00:00 GMT+0000 (Coordinated Universal Time)"
    );
    assert_eq!(
        run_str(&vm, "new Date(8.64e15).toISOString();"),
        "+275760-09-13T00:00:00.000Z"
    );
    assert_eq!(
        run_str(&vm, "new Date(-8.64e15).toISOString();"),
        "-271821-04-20T00:00:00.000Z"
    );
    assert_eq!(
        run_str(&vm, "new Date(1609459200123).toISOString();"),
        "2021-01-01T00:00:00.123Z"
    );
    assert!(run_bool(
        &vm,
        "Date.parse(new Date(1609459200123).toISOString()) === 1609459200123;"
    ));
    assert!(run_bool(
        &vm,
        "Date.parse(new Date(0).toGMTString()) === 0;"
    ));
    assert!(run_bool(&vm, "Date.parse(new Date(0).toString()) === 0;"));
    assert!(run_bool(
        &vm,
        "Date.parse(new Date(8.64e15).toISOString()) === 8.64e15;"
    ));
    assert!(run_bool(
        &vm,
        "Date.parse(new Date(-8.64e15).toGMTString()) === -8.64e15;"
    ));
    assert!(run_bool(&vm, "Date.parse('2021-01-01') === 1609459200000;"));
    assert!(run_bool(
        &vm,
        "Date.parse('2021-01-01T00:00:00.500Z') === 1609459200500;"
    ));
    assert!(run_bool(&vm, "isNaN(Date.parse('not a date'));"));
    assert!(run_bool(&vm, "isNaN(Date.parse(''));"));
}

#[test]
fn string_to_number_radix_prefixes() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "Number('0xff');"), 255);
    assert_eq!(run_smi(&vm, "Number('0X10');"), 16);
    assert_eq!(run_smi(&vm, "Number('0o17');"), 15);
    assert_eq!(run_smi(&vm, "Number('0b101');"), 5);
    assert_eq!(run_smi(&vm, "Number('010');"), 10);
    assert_eq!(run_smi(&vm, "'0xff' | 0;"), 255);
    assert_eq!(run_smi(&vm, "+'0b1111';"), 15);
    assert!(run_bool(&vm, "isNaN(Number('0x'));"));
    assert!(run_bool(&vm, "isNaN(Number('0b2'));"));
    assert!(run_bool(&vm, "isNaN(Number('-0x10'));"));
}

#[test]
fn array_length_setter() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "var a=[1,2,3]; a.length=1; a.length;"), 1);
    assert!(run_bool(
        &vm,
        "var a=[1,2,3]; a.length=1; a[1] === undefined;"
    ));
    assert_eq!(run_smi(&vm, "var a=[1,2,3]; a.length=5; a.length;"), 5);
    assert_eq!(
        run_str(
            &vm,
            "var a=[1,2,3]; a.length=1; Object.getOwnPropertyNames(a).join(',');"
        ),
        "0,length"
    );
    assert_eq!(
        run_smi(&vm, "var a=[]; a[0]=1; a[5]=2; a.length=1; a.length;"),
        1
    );
    assert!(run_bool(&vm, "var a=[1]; delete a.length === false;"));
    assert!(run_bool(
        &vm,
        "var a=[1,2,3]; a.length='2'; a.length === 2 && a[2] === undefined;"
    ));
    assert!(run_bool(
        &vm,
        "var a=[1,2,3]; var t=false; try { a.length=-1; } catch(e){ t=true; } t;"
    ));
    assert!(run_bool(
        &vm,
        "var a=[1,2,3]; var t=false; try { a.length=1.5; } catch(e){ t=true; } t;"
    ));
}

#[test]
fn array_slice_and_sort() {
    let vm = vm();
    assert_eq!(run_str(&vm, "[1,2,3,4,5].slice().join(',');"), "1,2,3,4,5");
    assert_eq!(run_str(&vm, "[1,2,3,4,5].slice(1).join(',');"), "2,3,4,5");
    assert_eq!(run_str(&vm, "[1,2,3,4,5].slice(1,3).join(',');"), "2,3");
    assert_eq!(run_str(&vm, "[1,2,3,4,5].slice(-2).join(',');"), "4,5");
    assert_eq!(run_smi(&vm, "[1,2,3].slice(3,1).length;"), 0);
    assert_eq!(run_str(&vm, "[3,1,2].sort().join(',');"), "1,2,3");
    assert_eq!(run_str(&vm, "[10,9,1,100].sort().join(',');"), "1,10,100,9");
    assert_eq!(
        run_str(&vm, "[3,1,2].sort(function(a,b){return a-b;}).join(',');"),
        "1,2,3"
    );
    assert_eq!(
        run_str(&vm, "[3,1,2].sort(function(a,b){return b-a;}).join(',');"),
        "3,2,1"
    );
    assert!(run_bool(
        &vm,
        "var a=[3,1,2]; a.sort() === a && a[0] === 1;"
    ));
    // undefined sorts before holes, both after defined values
    assert_eq!(
        run_str(&vm, "[5,undefined,3,undefined,1].sort().join(',');"),
        "1,3,5,,"
    );
}

#[test]
fn array_literal_elisions_are_holes() {
    let vm = vm();
    assert_eq!(run_smi(&vm, "[1,,3].length;"), 3);
    assert_eq!(run_smi(&vm, "[1,,].length;"), 2);
    assert_eq!(run_smi(&vm, "[,].length;"), 1);
    assert_eq!(run_smi(&vm, "[,,].length;"), 2);
    assert_eq!(run_smi(&vm, "[1,2,].length;"), 2);
    assert_eq!(run_smi(&vm, "[,,3,,].length;"), 4);
    assert!(run_bool(&vm, "!(1 in [1,,3]);"));
    assert!(run_bool(&vm, "!(0 in [,1]);"));
    assert!(run_bool(&vm, "1 in [,1];"));
    assert_eq!(run_str(&vm, "[1,,3].join();"), "1,,3");
}
