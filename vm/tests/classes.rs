//! End-to-end class tests: parse → resolve → compile → materialize → run.
//!
//! Covers ES2015 class semantics per ECMA-262 §15.7 (ClassDefinitionEvaluation,
//! [[Construct]] of base/derived constructors, super property access and super
//! calls) plus the observable edge cases.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DenseString, SlotName, Smi, Value};
use vm::{ScriptError, Thread, VM};

fn run(src: &str) -> Result<Value, ScriptError> {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script(src)
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

fn run_value(src: &str) -> (Value, Thread) {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
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

fn run_str(src: &str) -> String {
    let (result, mut thread) = run_value(src);
    thread.heap().no_gc(|nogc| {
        let s = result.get_as::<DenseString>(nogc).expect("string result");
        s.to_rust_string(nogc)
    })
}

/// Run a script that must terminate with an uncaught error; returns the
/// error's `name` (e.g. "TypeError", "ReferenceError").
fn run_error_name(src: &str) -> String {
    let (result, mut thread) = run_value(src);
    assert_eq!(
        result,
        thread.heap().known().exception.value(),
        "script must terminate with an uncaught error"
    );
    assert!(thread.has_pending_exception());
    let ex = thread.take_pending_exception().expect("pending exception");
    thread.handle_scope(|thread, scope| {
        let name_key = thread.intern(&scope, "name").value();
        thread.heap().no_gc(|nogc| {
            let o = ex
                .as_heap_object(nogc)
                .expect("pending exception must be an object");
            match o.as_ref().lookup(nogc, SlotName::from_value(name_key)) {
                vm::Lookup::Data { slot, .. } => {
                    let s = slot
                        .inner()
                        .get_as::<DenseString>(nogc)
                        .expect("error name is a string");
                    s.to_rust_string(nogc)
                }
                _ => panic!("error object must have a name property"),
            }
        })
    })
}

fn assert_error(src: &str, name: &str) {
    assert_eq!(run_error_name(src), name, "unexpected error for: {src}");
}

// ---------------------------------------------------------------------------
// basic classes
// ---------------------------------------------------------------------------

#[test]
fn class_declaration_method_and_this() {
    assert_eq!(
        run_smi(
            "class A { constructor() { this.x = 1; } m() { return this.x + 1; } } var a = new A(); a.m();"
        ),
        2
    );
}

#[test]
fn class_expression_value_and_new() {
    assert_eq!(
        run_smi("var a = new (class { m() { return 7; } })(); a.m();"),
        7
    );
    assert_eq!(run_smi("(new class {}(), 5);"), 5);
}

#[test]
fn instanceof_and_prototype_wiring() {
    assert!(run_bool(
        "class A {} var a = new A();
         a instanceof A && a instanceof Object;"
    ));
    assert!(run_bool(
        "class A {} var a = new A();
         Object.getPrototypeOf(a) === A.prototype;"
    ));
    assert!(run_bool("class A {}; A.prototype.constructor === A;"));
    // %Function.prototype% via the Object constructor's own prototype
    assert!(run_bool(
        "class A {}; Object.getPrototypeOf(A) === Object.getPrototypeOf(Object);"
    ));
}

#[test]
fn class_name_and_length() {
    assert_eq!(run_str("class Foo {} Foo.name;"), "Foo");
    assert_eq!(run_smi("class A {} A.length;"), 0);
    assert_eq!(run_smi("class A { constructor(a, b) {} } A.length;"), 2);
    // anonymous class expressions have an empty name
    assert_eq!(run_str("(class {}).name;"), "");
}

#[test]
fn class_tdz() {
    assert_error("var a = new A(); class A {};", "ReferenceError");
    // the binding is initialized when the class definition finishes
    assert!(run_bool(
        "var f = function () { return A; }; class A {} f() === A;"
    ));
}

#[test]
fn class_inner_name_binding() {
    // methods see the inner class binding
    assert!(run_bool(
        "class A { m() { return A; } } A.prototype.m() === A;"
    ));
    assert!(run_bool(
        "var C = class B { m() { return B; } }; C.prototype.m() === C;"
    ));
    // the inner binding is in TDZ during `extends` and computed keys
    assert_error("class A extends A {};", "ReferenceError");
    assert_error("class A { [A]() {} }", "ReferenceError");
    // the class value in acc
    assert!(run_bool(
        "class A { m() { return typeof A; } } A.prototype.m() === 'function';"
    ));
}

#[test]
fn class_is_strict_and_not_callable() {
    assert_error("class A {}; A();", "TypeError");
}

#[test]
fn methods_are_not_constructible() {
    assert_error("class A { m() {} }; new A.prototype.m();", "TypeError");
}

#[test]
fn class_methods_dispatch_through_prototype() {
    assert_eq!(
        run_smi(
            "class A { m() { return 3; } }
             var a = new A(); var b = new A();
             A.prototype.m = function() { return 4; };
             a.m() + b.m();"
        ),
        8
    );
}

#[test]
fn static_methods() {
    assert_eq!(run_smi("class A { static f() { return 11; } } A.f();"), 11);
    assert_error(
        "class A { static f() {} }; var a = new A(); a.f();",
        "TypeError",
    );
}

#[test]
fn getters_and_setters() {
    assert_eq!(
        run_smi(
            "class A {
                 constructor() { this._v = 0; }
                 get v() { return this._v * 2; }
                 set v(x) { this._v = x; }
             }
             var a = new A(); a.v = 5; a.v;"
        ),
        10
    );
    // get/set pair on the same key merge into one property
    assert_eq!(
        run_smi(
            "class A {
                 get x() { return this.secret; }
                 set x(v) { this.secret = v; }
             }
             var a = new A(); a.x = 21; a.x;"
        ),
        21
    );
    // accessors are inherited
    assert_eq!(
        run_smi(
            "class A { get g() { return 9; } }
             class B extends A {}
             new B().g;"
        ),
        9
    );
    // static accessors
    assert_eq!(
        run_smi("class A { static get s() { return 12; } } A.s;"),
        12
    );
}

#[test]
fn computed_member_names() {
    assert_eq!(
        run_smi("var k = 'm'; class A { [k]() { return 6; } } new A().m();"),
        6
    );
    assert_eq!(
        run_smi("var k = 'v'; class A { get [k]() { return 13; } } new A().v;"),
        13
    );
    assert_eq!(
        run_smi(
            "class A { ['a' + 'b']() { return 1; } static ['st']() { return 2; } }
             new A().ab() + A.st();"
        ),
        3
    );
    // number-like keys
    assert_eq!(run_smi("class A { 1() { return 8; } } new A()[1]();"), 8);
}

#[test]
fn methods_see_computed_keys_of_enclosing_scope() {
    assert_eq!(
        run_smi(
            "var tag = 'x';
             class A { [tag + '1']() { return 5; } }
             new A().x1();"
        ),
        5
    );
}

#[test]
fn constructor_return_object_wins() {
    assert!(run_bool(
        "class A { constructor() { return { x: 1 }; } }
         var a = new A();
         a.x === 1 && !(a instanceof A);"
    ));
    // primitive returns from a base constructor are ignored
    assert!(run_bool(
        "class A { constructor() { return 2; } }
         new A() instanceof A;"
    ));
}

// ---------------------------------------------------------------------------
// extends / super()
// ---------------------------------------------------------------------------

#[test]
fn extends_basic_inheritance() {
    assert_eq!(
        run_smi(
            "class A { constructor(x) { this.x = x; } }
             class B extends A {}
             var b = new B(4);
             b.x;"
        ),
        4
    );
    assert!(run_bool(
        "class A {} class B extends A {}
         var b = new B();
         b instanceof A && b instanceof B && Object.getPrototypeOf(B) === A;"
    ));
    assert!(run_bool(
        "class A {} class B extends A {}
         Object.getPrototypeOf(B.prototype) === A.prototype;"
    ));
}

#[test]
fn default_constructors_forward_arguments() {
    // default derived ctor forwards all arguments
    assert_eq!(
        run_smi(
            "class A { constructor(a, b) { this.s = a + b; } }
             class B extends A {}
             new B(20, 3).s;"
        ),
        23
    );
    // deep chain of default ctors
    assert_eq!(
        run_smi(
            "class A { constructor(v) { this.v = v; } }
             class B extends A {}
             class C extends B {}
             class D extends C {}
             new D(31).v;"
        ),
        31
    );
}

#[test]
fn new_target_threading_sets_prototype() {
    // the instance prototype comes from the original new.target (B), not the
    // executing constructor (A): the whole point of super()'s new_target
    assert!(run_bool(
        "class A {} class B extends A { constructor() { super(); } }
         var b = new B();
         Object.getPrototypeOf(b) === B.prototype;"
    ));
    assert!(run_bool(
        "class A {} class B extends A {}
         var b = new B();
         Object.getPrototypeOf(b) === B.prototype;"
    ));
}

#[test]
fn derived_constructor_super_call() {
    assert_eq!(
        run_smi(
            "class A { constructor(v) { this.v = v; } }
             class B extends A { constructor() { super(9); this.w = 1; } }
             var b = new B();
             b.v + b.w;"
        ),
        10
    );
    // super() evaluates to the new this
    assert!(run_bool(
        "same = false;
         class A {}
         class B extends A { constructor() { same = super() === this; } }
         new B();
         same;"
    ));
}

#[test]
fn derived_this_before_super_throws() {
    assert_error(
        "class A {} class B extends A { constructor() { this.x = 1; super(); } }
         new B();",
        "ReferenceError",
    );
    // `this` in a nested arrow before super() throws too
    assert_error(
        "class A {} class B extends A { constructor() { var f = () => this; f(); super(); } }
         new B();",
        "ReferenceError",
    );
}

#[test]
fn derived_double_super_throws() {
    assert_error(
        "class A {} class B extends A { constructor() { super(); super(); } }
         new B();",
        "ReferenceError",
    );
}

#[test]
fn derived_missing_super_throws() {
    assert_error(
        "class A {} class B extends A { constructor() {} } new B();",
        "ReferenceError",
    );
    assert_error(
        "class A {} class B extends A { constructor() { return undefined; } } new B();",
        "ReferenceError",
    );
}

#[test]
fn derived_return_semantics() {
    // implicit return this
    assert!(run_bool(
        "class A {} class B extends A { constructor() { super(); } }
         new B() instanceof B;"
    ));
    // return undefined → this
    assert!(run_bool(
        "class A {} class B extends A { constructor() { super(); return undefined; } }
         new B() instanceof B;"
    ));
    // return object wins
    assert!(run_bool(
        "class A {} class B extends A { constructor() { super(); return { z: 1 }; } }
         new B().z === 1;"
    ));
    // return primitive (non-undefined) → TypeError
    assert_error(
        "class A {} class B extends A { constructor() { super(); return 2; } }
         new B();",
        "TypeError",
    );
    assert_error(
        "class A {} class B extends A { constructor() { super(); return null; } }
         new B();",
        "TypeError",
    );
}

#[test]
fn extends_value_validation() {
    assert_error("class A extends 5 {};", "TypeError");
    assert_error("class A extends 'x' {};", "TypeError");
    assert_error("class A extends (() => 1) {};", "TypeError");
    // function expressions are constructors
    assert!(run_bool("class A extends (function () {}) {} true;"));
    // superCtor.prototype must be an object or null
    assert_error(
        "function F() {} F.prototype = 5; class A extends F {}",
        "TypeError",
    );
}

#[test]
fn extends_null() {
    // the class definition itself succeeds with a null-proto prototype;
    // construction throws: GetSuperConstructor() is %Function.prototype%,
    // which is not a constructor (node: "Super constructor null of A is
    // not a constructor")
    assert!(run_bool(
        "class A extends null {}
         Object.getPrototypeOf(A) === Object.getPrototypeOf(Object);"
    ));
    assert_error("class A extends null {} new A();", "TypeError");
    assert_error(
        "class A extends null { constructor() { super(); } } new A();",
        "TypeError",
    );
    // an explicit constructor that returns an object works
    assert!(run_bool(
        "class A extends null { constructor() { return { x: 1 }; } }
         new A().x === 1;"
    )); // super.x resolves through the null-proto prototype: finds nothing
    assert_eq!(
        run_smi("class A extends null { m() { return super.m || 42; } } A.prototype.m();"),
        42
    );
}

#[test]
fn extends_expression() {
    assert_eq!(
        run_smi(
            "var Base = class { constructor() { this.v = 3; } };
             class Sub extends Base {}
             new Sub().v;"
        ),
        3
    );
    // extends with a member expression
    assert_eq!(
        run_smi(
            "var ns = { Base: class { m() { return 2; } } };
             class S extends ns.Base {}
             new S().m();"
        ),
        2
    );
}

// ---------------------------------------------------------------------------
// super.x property access
// ---------------------------------------------------------------------------

#[test]
fn super_method_call() {
    assert_eq!(
        run_smi(
            "class A { m() { return 1; } }
             class B extends A { m() { return super.m() + 1; } }
             new B().m();"
        ),
        2
    );
    // super.m() runs with `this` = the instance
    assert_eq!(
        run_smi(
            "class A { m() { return this.v; } }
             class B extends A { constructor() { super(); this.v = 6; } m() { return super.m(); } }
             new B().m();"
        ),
        6
    );
    // deep chain
    assert_eq!(
        run_smi(
            "class A { m() { return 1; } }
             class B extends A { m() { return super.m() + 2; } }
             class C extends B { m() { return super.m() + 4; } }
             new C().m();"
        ),
        7
    );
}

#[test]
fn super_property_load_and_keyed() {
    // super.x looks up the home object's prototype chain — properties on
    // the parent *prototype*, not own instance properties
    assert_eq!(
        run_smi(
            "class A {}
             A.prototype.base = 10;
             class B extends A { m() { return super.base; } }
             new B().m();"
        ),
        10
    );
    // getters on the parent prototype run with this = receiver
    assert_eq!(
        run_smi(
            "class A { get g() { return this.v * 3; } }
             class B extends A { constructor() { super(); this.v = 4; } m() { return super.g; } }
             new B().m();"
        ),
        12
    );
    // keyed access
    assert_eq!(
        run_smi(
            "class A { m() { return 5; } }
             class B extends A { m() { return super['m'](); } }
             new B().m();"
        ),
        5
    );
    // missing properties yield undefined
    assert_eq!(
        run_smi("class A {} class B extends A { m() { return super.nope || 0; } } new B().m();"),
        0
    );
}

#[test]
fn super_property_store() {
    // super.x = v dispatches to the parent's setter with this = instance
    assert_eq!(
        run_smi(
            "class A {
                 set x(v) { this.stored = v; }
             }
             class B extends A { set x(v) { super.x = v * 2; } }
             var b = new B(); b.x = 8; b.stored;"
        ),
        16
    );
    // storing over an inherited data property shadows on `this`, not the parent
    assert_eq!(
        run_smi(
            "class A {}
             A.prototype.p = 1;
             class B extends A { m() { super.p = 2; } }
             var b = new B(); b.m();
             b.p + A.prototype.p;"
        ),
        3
    );
    // storing over a property `this` already owns (from the base
    // constructor) overwrites it in place: OrdinarySet's receiver step
    // (ES 9.1.9.2 step 3.c), never a duplicate add
    assert_eq!(
        run_smi(
            "class A { constructor() { this.x = 1; } }
             class B extends A { setX() { super.x = 5; } }
             var b = new B(); b.setX(); b.x;"
        ),
        5
    );
    // the same define-or-overwrite applies when the parent chain lacks
    // the name entirely (the walk's implicit default descriptor)
    assert_eq!(
        run_smi(
            "class A {}
             class B extends A { m() { super.n = 1; super.n = 2; return this.n; } }
             new B().m();"
        ),
        2
    );
    // keyed form over an own smi-named property of the receiver
    assert_eq!(
        run_smi(
            "class A { constructor() { this[0] = 1; } }
             class B extends A { set() { super[0] = 5; } }
             var b = new B(); b.set(); b[0];"
        ),
        5
    );
}

#[test]
fn super_in_static_methods() {
    assert_eq!(
        run_smi(
            "class A { static f() { return 3; } }
             class B extends A { static f() { return super.f() + 1; } }
             B.f();"
        ),
        4
    );
    // static home object is the constructor: super looks up A itself
    assert_eq!(
        run_smi(
            "class A { static get tag() { return 7; } }
             class B extends A { static m() { return super.tag; } }
             B.m();"
        ),
        7
    );
}

#[test]
fn super_through_prototype_reassignment() {
    // super.x resolves through the [[HomeObject]] captured at class creation
    assert_eq!(
        run_smi(
            "class A { m() { return 1; } }
             class B extends A { m() { return super.m(); } }
             var b = new B();
             B.prototype = {}; // reassigning .prototype does not affect super
             b.m();"
        ),
        1
    );
}

#[test]
fn classes_in_loops_get_fresh_home_objects() {
    assert_eq!(
        run_smi(
            "class Base { constructor(v) { this.v = v; } }
             var out = 0;
             for (var i = 1; i <= 3; i++) {
                 class C extends Base { m() { return this.v; } }
                 out += new C(i).m();
             }
             out;"
        ),
        6
    );
}

#[test]
fn super_in_getters_and_setters() {
    assert_eq!(
        run_smi(
            "class A { get v() { return 2; } }
             class B extends A { get v() { return super.v + 3; } }
             new B().v;"
        ),
        5
    );
    assert_eq!(
        run_smi(
            "class A { set s(v) { this.a = v; } }
             class B extends A { set s(v) { super.s = v + 1; } }
             var b = new B(); b.s = 4; b.a;"
        ),
        5
    );
}

#[test]
fn super_in_nested_arrows() {
    assert_eq!(
        run_smi(
            "class A { m() { return 8; } }
             class B extends A { m() { var f = () => super.m(); return f(); } }
             new B().m();"
        ),
        8
    );
    // `super.x` inside a nested arrow captures through the context chain
    assert_eq!(
        run_smi(
            "class A {}
             A.prototype.v = 3;
             class B extends A { m() { var f = () => super.v; return f(); } }
             new B().m();"
        ),
        3
    );
    // `super()` inside a nested arrow is delegated through the
    // constructor's threaded closure and new.target
    assert_eq!(
        run_smi(
            "class A { constructor(v) { this.v = v; } }
             class B extends A { constructor() { var f = () => super(9); f(); } }
             new B().v;"
        ),
        9
    );
    // the bound this is shared: the ctor sees it after the delegated call
    assert_eq!(
        run_smi(
            "class A { constructor() { this.n = 1; } }
             class B extends A { constructor() { (() => super())(); this.n += 5; } }
             new B().n;"
        ),
        6
    );
    // new.target inside a nested arrow delegates to the constructor
    assert!(run_bool(
        "class A {}
         class B extends A { constructor() { super(); var g = () => new.target; this.t = g(); } }
         new B().t === B;"
    ));
}

#[test]
fn new_target_basic_forms() {
    // undefined outside construction
    assert!(run_bool("new.target === undefined;"));
    assert!(run_bool(
        "function f() { return new.target; } f() === undefined;"
    ));
    // the constructor itself
    assert!(run_bool(
        "function f() { return new.target; } new f() === f;"
    ));
    assert!(run_bool(
        "class A { constructor() { this.t = new.target; } } new A().t === A;"
    ));
    // new.target threads through the default-ctor super chain
    assert!(run_bool(
        "class A { constructor() { this.q = new.target; } }
         class B extends A {}
         new B().q === B;"
    ));
    // arrow delegation in plain functions
    assert!(run_bool(
        "function f() { var g = () => new.target; return g(); } new f() === f;"
    ));
}

// ---------------------------------------------------------------------------
// early errors
// ---------------------------------------------------------------------------

#[test]
fn class_early_errors() {
    // parse errors surface as ScriptError::Parse
    fn assert_parse_error(src: &str) {
        match run(src) {
            Err(ScriptError::Parse(_)) => {}
            other => panic!("expected parse error for {src}, got {other:?}"),
        }
    }
    assert_parse_error("class {};");
    assert_parse_error("class A { constructor() {} constructor() {} }");
    assert_parse_error("class A { get constructor() {} }");
    assert_parse_error("class A { static prototype() {} }");
    assert_parse_error("class A { static get prototype() {} }");
    assert_parse_error("class A { m() { super(); } }"); // super() in a method
    assert_parse_error("function f() { super.x; }"); // super outside a class
    assert_parse_error("super.x;"); // top-level super
    assert_parse_error(
        "class A {} class B extends A { constructor() { function g() { super(); } g(); } }",
    );
    // a static member named "constructor" is an ordinary method
    assert_eq!(
        run_smi("class A { static constructor() { return 5; } } A.constructor();"),
        5
    );
}

// ---------------------------------------------------------------------------
// combinations
// ---------------------------------------------------------------------------

#[test]
fn class_stack_with_closures() {
    assert_eq!(
        run_smi(
            "class Counter {
                 constructor() { this.n = 0; }
                 bump() { this.n = this.n + 1; return this; }
                 get value() { return this.n; }
             }
             var c = new Counter();
             c.bump().bump().bump();
             c.value;"
        ),
        3
    );
}

#[test]
fn class_methods_can_return_classes() {
    assert_eq!(
        run_smi(
            "class A { make() { return class { m() { return 14; } }; } }
             var B = new A().make();
             new B().m();"
        ),
        14
    );
}

#[test]
fn instanceof_walks_the_chain() {
    assert!(run_bool(
        "class A {} class B extends A {} class C extends B {}
         new C() instanceof A && new C() instanceof B && new C() instanceof C;"
    ));
}

#[test]
fn typeof_and_instanceof_class() {
    assert_eq!(run_str("class A {}; typeof A;"), "function");
    assert!(run_bool(
        "class A {}; Object.getPrototypeOf(A) === Object.getPrototypeOf(Object);"
    ));
}

#[test]
fn computed_constructor_member_overwrites_wiring() {
    // wiring (proto.constructor, ctor.prototype) precedes member
    // installation: computed ['constructor'] members win
    assert_eq!(
        run_smi("class A { ['constructor']() { return 5; } } A.prototype.constructor();"),
        5
    );
    // a computed constructor member is NOT the class constructor
    assert_eq!(
        run_smi("class A { m() {} } class B extends A { ['constructor']() {} } 7;"),
        7
    );
    assert!(run_bool(
        "class A { m() {} } class B extends A { ['constructor']() {} }
         typeof new B() === 'object';"
    ));
    // a computed static ['prototype'] collides with the non-configurable
    // ctor.prototype: TypeError at class definition
    assert_error("class A { static ['prototype']() {} }", "TypeError");
}
