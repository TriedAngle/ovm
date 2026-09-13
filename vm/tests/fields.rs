//! End-to-end tests for class fields (ES 15.7): public and private,
//! instance and static, plus the private-name semantics of ES 7.3.26–33.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{Smi, VM, VMString, Value};

fn run(src: &str) -> Result<Value, vm::ScriptError> {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script(src)
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

fn run_str(src: &str) -> String {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let v = thread.run_script(src).unwrap();
    thread.heap().no_gc(|nogc| {
        let s = v.get_as::<VMString>(nogc).expect("string result");
        String::from_utf8(s.as_slice(nogc).to_vec()).unwrap()
    })
}

fn run_bool(src: &str) -> bool {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let v = thread.run_script(src).unwrap();
    v == thread.heap().known().true_object.value()
}

fn throws(src: &str) -> bool {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    match thread.run_script(src) {
        Ok(v) => v == thread.heap().known().exception.value(),
        Err(_) => true,
    }
}

// -- public instance fields ---------------------------------------------------

#[test]
fn field_basic() {
    assert_eq!(run_smi("class A { x = 1; } new A().x;"), 1);
    assert_eq!(
        run_smi("class A { x = 1; y = 2; } var a = new A(); a.x + a.y;"),
        3
    );
    // uninitialized fields are undefined
    assert!(run_bool("class A { x; } var u; new A().x === u;"));
    // fields shadow prototype properties (DefineOwnProperty on the instance)
    assert_eq!(
        run_smi("class A { x = 1; } A.prototype.x = 9; new A().x;"),
        1
    );
}

#[test]
fn field_initializer_sees_this() {
    assert_eq!(run_smi("class A { a = 1; b = this.a + 1; } new A().b;"), 2);
    // initializers run before the constructor body
    assert_eq!(
        run_smi("class A { a = 1; constructor() { this.a += 1; } } new A().a;"),
        2
    );
    // the constructor body sees already-defined fields
    assert_eq!(
        run_smi("class A { a = 5; constructor() { this.b = this.a * 2; } } new A().b;"),
        10
    );
}

#[test]
fn field_evaluation_order() {
    let src = "var o = ''; \
               class A { a = (o += '1'); b = (o += '2'); c = (o += '3'); } \
               new A(); o;";
    assert_eq!(run_str(src), "123");
}

#[test]
fn field_computed_keys() {
    assert_eq!(run_smi("var k = 'ke'; class A { [k] = 5; } new A().ke;"), 5);
    // computed keys evaluate in element order, before the class finishes
    let src = "var o = ''; \
               class A { [(o += '1', 'a')] = 1; b = (o += '2'); } \
               new A(); o;";
    assert_eq!(run_str(src), "12");
    // numeric keys
    assert_eq!(run_smi("class A { 1 = 7; } new A()[1];"), 7);
}

#[test]
fn field_on_derived_classes() {
    assert_eq!(
        run_smi("class B { a = 1; } class D extends B { b = 2; } var d = new D(); d.a + d.b;"),
        3
    );
    // fields initialize when super() returns, before the body
    let src = "class B { constructor() { this.seen = 0; } } \
               class D extends B { d = (this.seen += 1); } \
               new D().seen;";
    assert_eq!(run_smi(src), 1);
    // derived constructor with explicit super() and a default field
    assert_eq!(
        run_smi(
            "class B {} class D extends B { x = 3; constructor() { super(); this.y = this.x; } } \
             var d = new D(); d.x + d.y;"
        ),
        6
    );
    // the default derived constructor initializes fields too
    assert_eq!(
        run_smi("class B {} class D extends B { x = 4; } new D().x;"),
        4
    );
    // base fields initialize before base constructor body; derived after
    let src = "var o = ''; \
               class B { b = (o += '1'); constructor() { o += '2'; } } \
               class D extends B { d = (o += '3'); constructor() { super(); o += '4'; } } \
               new D(); o;";
    assert_eq!(run_str(src), "1234");
}

#[test]
fn field_new_target_prototype_chain() {
    // instances still get new.target.prototype as their [[Prototype]]
    assert_eq!(
        run_smi(
            "class A { x = 1; } var a = new A(); \
             Object.getPrototypeOf(a) === A.prototype ? 1 : 0;"
        ),
        1
    );
}

#[test]
fn field_anonymous_function_naming() {
    // `f = function(){}` fields are named after the key (ES 8.4.3 via
    // ClassFieldEvaluation's MakeClassFieldInitializer naming)
    assert_eq!(
        run_str("class A { fn = function(){}; } new A().fn.name;"),
        "fn"
    );
}

// -- static fields --------------------------------------------------------------

#[test]
fn static_field_basic() {
    assert_eq!(run_smi("class A { static s = 2; } A.s;"), 2);
    assert_eq!(
        run_smi("class A { static a = 1; static b = 2; } A.a + A.b;"),
        3
    );
    assert!(run_bool("class A { static s; } A.s === undefined;"));
    // statics are own properties of the constructor (writable)
    assert_eq!(run_smi("class A { static s = 1; } A.s = 5; A.s;"), 5);
    // statics don't leak onto instances
    assert!(run_bool(
        "class A { static s = 1; } new A().s === undefined;"
    ));
}

#[test]
fn static_field_this_is_class() {
    assert_eq!(
        run_smi("class A { static s = 7; static t = this.s; } A.t;"),
        7
    );
}

#[test]
fn static_field_order() {
    // static initializers run in order after the class is created
    let src = "var o = ''; \
               class A { static a = (o += '1'); static b = (o += '2'); } \
               o;";
    assert_eq!(run_str(src), "12");
    // instance field KEYS evaluate during the element loop (side effects
    // in a computed key would show here); their INITIALIZERS run per
    // instance, not at class definition
    let src = "var o = ''; \
               class A { [(o += '1', 'i')] = 9; static s = (o += '2'); } \
               o;";
    assert_eq!(run_str(src), "12");
}

#[test]
fn static_field_computed_and_derived() {
    assert_eq!(run_smi("var k = 'K'; class A { static [k] = 3; } A.K;"), 3);
    assert_eq!(
        run_smi("class B { static s = 1; } class D extends B { static t = B.s + 1; } D.t;"),
        2
    );
}

// -- private fields ---------------------------------------------------------------

#[test]
fn private_field_basic() {
    assert_eq!(
        run_smi("class A { #x = 1; get() { return this.#x; } } new A().get();"),
        1
    );
    // private names never appear as public properties
    assert!(!run_bool("class A { #x = 1; } var a = new A(); '#x' in a;"));
    // writes through methods
    assert_eq!(
        run_smi(
            "class A { #c = 0; inc() { this.#c += 1; return this.#c; } } \
             var a = new A(); a.inc(); a.inc();"
        ),
        2
    );
    // two instances have independent fields but share the name
    assert_eq!(
        run_smi(
            "class A { #x; set(v) { this.#x = v; } get() { return this.#x; } } \
             var a = new A(); var b = new A(); a.set(1); b.set(2); a.get() + b.get();"
        ),
        3
    );
}

#[test]
fn private_field_brand_checks() {
    // access on a non-instance throws TypeError (ES 7.3.30)
    assert!(throws(
        "class A { #x = 1; static t(o) { return o.#x; } } A.t({});"
    ));
    assert!(throws(
        "class A { #x = 1; m() { return this.#x; } } var a = new A(); a.m.call({});"
    ));
    // access before the field is added (via a base-class hook) throws
    assert!(throws(
        "class B { constructor() { this.read(); } } \
         class D extends B { #x = 1; read() { return this.#x; } } \
         new D();"
    ));
    // `#x in obj` performs the brand check without throwing
    assert_eq!(
        run_str("class A { #x; static t(o) { return #x in o; } } '' + A.t(new A()) + A.t({});"),
        "truefalse"
    );
}

#[test]
fn private_field_static() {
    assert_eq!(
        run_smi("class A { static #s = 5; static get() { return A.#s; } } A.get();"),
        5
    );
    assert!(throws("class A { static #s = 1; } A.#s;"));
}

#[test]
fn private_names_are_per_evaluation() {
    // two classes with the same private name have distinct private names
    assert_eq!(
        run_smi(
            "function make() { return class { #x = 1; read(o) { return o.#x; } }; } \
             var A = make(); var B = make(); \
             var b = new B(); \
             var threw = 0; \
             try { new A().read(b); } catch (e) { threw = 1; } \
             threw;"
        ),
        1
    );
}

#[test]
fn private_field_initializer_this() {
    assert_eq!(
        run_smi("class A { #x = 3; #y = this.#x * 2; get() { return this.#y; } } new A().get();"),
        6
    );
    // nested class privates (ES 15.7.3: inner private environments)
    assert_eq!(
        run_smi(
            "class Outer { #o = 1; static make() { \
                 class Inner { #i = 2; both(x) { return x.#o + this.#i; } } \
                 return Inner; \
             } } \
             var I = Outer.make(); new I().both(new Outer());"
        ),
        3
    );
}

#[test]
fn private_field_with_super() {
    // field initializers are class-member functions: super.x works in them
    // (direct access in the initializer; `super.m()` already invokes m)
    assert_eq!(
        run_smi(
            "class B { m() { return 5; } } \
             class D extends B { f = super.m(); } \
             new D().f;"
        ),
        5
    );
    // ...and through arrows capturing the initializer's home object
    assert_eq!(
        run_smi(
            "class B { m() { return 6; } } \
             class D extends B { f = () => super.m(); } \
             new D().f();"
        ),
        6
    );
}

#[test]
fn static_members_named_name_shadow() {
    // a static method named `name` overwrites the class's own name
    // (SetFunctionName runs before element installation, ES 15.7.14)
    assert_eq!(
        run_smi("var C = class { static name() { return 5; } }; C.name();"),
        5
    );
    // static fields likewise
    assert_eq!(run_smi("var C = class { static name = 9; }; C.name;"), 9);
    // accessors too
    assert_eq!(
        run_smi("var C = class { static get name() { return 7; } }; C.name;"),
        7
    );
    // ...while plain anonymous classes still get their binding name
    assert_eq!(
        run_str(
            "var C = class { static name = 1; }; \
             var named = class { }; \
             '' + (typeof C.name) + '/' + named.name;"
        ),
        "number/named"
    );
    // plain anonymous classes still get named (NamedEvaluation)
    assert_eq!(run_str("var w = class {}; w.name;"), "w");
    assert_eq!(run_str("var {q = class {}} = {}; q.name;"), "q");
    assert_eq!(run_str("var [r = class {}] = []; r.name;"), "r");
}

// -- early errors --------------------------------------------------------------------

#[test]
fn field_early_errors() {
    // class may not have a field named constructor
    assert!(run("class A { constructor = 1; }").is_err());
    // static field named prototype
    assert!(run("class A { static prototype = 1; }").is_err());
    // instance field named prototype is fine
    assert_eq!(run_smi("class A { prototype = 9; } new A().prototype;"), 9);
    // duplicate private names
    assert!(run("class A { #x; #x; }").is_err());
    // undeclared private references
    assert!(run("class A { m() { this.#y; } }").is_err());
    assert!(run("this.#x;").is_err());
    assert!(run("var o = {m() { return this.#p; }};").is_err());
    // private names are only valid in classes
    assert!(run("function f(o) { return #x in o; }").is_err());
    // private accessors/methods: clean errors (not supported yet)
    assert!(run("class A { #m() {} }").is_err());
    assert!(run("class A { get #m() {} }").is_err());
    // super.#x is invalid
    assert!(run("class B {} class A extends B { m() { return super.#x; } }").is_err());
}

#[test]
fn private_in_requires_class_context() {
    // `#x in obj` valid inside the declaring class
    assert_eq!(
        run_smi("class A { #v; static has(o) { return #v in o; } } A.has(new A()) ? 1 : 0;"),
        1
    );
}

// -- interactions ---------------------------------------------------------------------

#[test]
fn fields_and_methods_mixed() {
    assert_eq!(
        run_smi(
            "class A { \
                 x = this.m(); \
                 m() { return 4; } \
                 y = 5; \
             } \
             var a = new A(); a.x + a.y;"
        ),
        9
    );
}

#[test]
fn fields_with_destructured_params() {
    assert_eq!(
        run_smi(
            "class A { x = 1; constructor([a, b] = [2, 3]) { this.v = a + b + this.x; } } \
             new A().v;"
        ),
        6
    );
}
