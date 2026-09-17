//! Proxy objects (ES 20.2): trap dispatch, forwarding, revocation, and
//! the essential invariants of the implemented internal methods
//! ([[Get]], [[Set]], [[HasProperty]], [[Delete]],
//! [[DefineOwnProperty]], [[Call]], [[Construct]],
//! [[IsExtensible]], [[PreventExtensions]]).

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{ScriptError, Smi, VM};

fn run(src: &str) -> Result<vm::Value, ScriptError> {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script(src)
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

#[allow(dead_code)]
fn run_bool(src: &str) -> bool {
    let (result, mut thread) = run_value(src);
    let heap = thread.heap();
    let known = heap.known();
    if result == known.true_object.as_tagged(heap).erase() {
        true
    } else if result == known.false_object.as_tagged(heap).erase() {
        false
    } else {
        panic!("expected boolean result, got {result:?}");
    }
}

#[allow(dead_code)]
fn run_value(src: &str) -> (vm::Value, vm::Thread) {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    (result, thread)
}

/// The script must complete with an uncaught error whose `name` is
/// "TypeError".
fn assert_type_error(src: &str) {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script(src).expect("script must complete");
    {
        let heap = thread.heap();
        assert_eq!(
            result,
            heap.known().exception.as_tagged(heap).erase(),
            "expected an uncaught exception"
        );
    }
    let ex = thread
        .take_pending_exception()
        .expect("pending exception set");
    thread.handle_scope(|thread, scope| {
        let name_handle = thread.intern(&scope, "name");
        let name = thread.heap().no_gc(|heap| {
            let Some(obj) = unsafe { ex.assume_valid(heap) }.as_heap_object() else {
                panic!("exception is not an object");
            };
            match obj
                .as_ref()
                .lookup(heap, name_handle.as_tagged(heap).into())
            {
                vm::Lookup::Data { slot, .. } => slot
                    .get(heap)
                    .get_as::<vm::DenseString>()
                    .map(|s| s.to_rust_string(heap))
                    .unwrap_or_default(),
                _ => String::new(),
            }
        });
        assert_eq!(name, "TypeError", "wrong exception class");
    });
}

// ---- constructor surface ----------------------------------------------------

#[test]
fn constructor_validates_target_and_handler() {
    assert_type_error("new Proxy(1, {})");
    assert_type_error("new Proxy(true, {})");
    assert_type_error("new Proxy(null, {})");
    assert_type_error("new Proxy(undefined, {})");
    assert_type_error("new Proxy('str', {})");
    assert_type_error("new Proxy({}, 1)");
    assert_type_error("new Proxy({}, null)");
    assert_type_error("new Proxy({}, undefined)");
    assert_type_error("new Proxy()");
    // calling without new throws
    assert_type_error("Proxy({}, {})");
    // callable/non-callable map bits mirror the target
    assert!(run("typeof new Proxy({}, {})").is_ok());
}

#[test]
fn constructor_has_no_prototype_and_metadata() {
    assert_eq!(run_smi("Proxy.prototype === undefined ? 1 : 0"), 1);
    assert_eq!(run_smi("Proxy.length"), 2);
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let name = thread.run_script("Proxy.name").unwrap();
    thread.heap().no_gc(|heap| {
        let s = unsafe { name.assume_valid(heap) }
            .get_as::<vm::DenseString>()
            .expect("Proxy.name is a string");
        assert!(s.data(heap).matches_ascii(b"Proxy"));
    });
}

// ---- get ---------------------------------------------------------------------

#[test]
fn get_trap_result_wins() {
    assert_eq!(
        run_smi("var p = new Proxy({attr: 1}, { get: function() { return 2; } }); p.attr"),
        2
    );
    assert_eq!(
        run_smi("var p = new Proxy({}, { get: function() { return 42; } }); p.missing"),
        42
    );
}

#[test]
fn get_trap_parameters() {
    assert_eq!(
        run_smi(
            r#"
            var target = {};
            var handler = { get: function(t, k, r) { return t === target && k === "x" && r === proxy ? 1 : 0; } };
            var proxy = new Proxy(target, handler);
            proxy.x
            "#
        ),
        1
    );
}

#[test]
fn get_forwards_without_trap() {
    assert_eq!(
        run_smi("var t = {a: 1}; var p = new Proxy(t, {}); p.a + (p.b === undefined ? 10 : 0)"),
        11
    );
    // proxy-of-proxy chains forward recursively
    assert_eq!(
        run_smi("var t = {a: 5}; var p1 = new Proxy(t, {}); var p2 = new Proxy(p1, {}); p2.a"),
        5
    );
    // getters run with this = the proxy receiver
    assert_eq!(
        run_smi(
            r#"
            var t = { get g() { return this; } };
            var p = new Proxy(t, {});
            p.g === p ? 1 : 0
            "#
        ),
        1
    );
}

#[test]
fn get_trap_is_not_callable_throws() {
    assert_type_error("var p = new Proxy({}, { get: 1 }); p.x");
    // null/undefined traps mean forwarding, not an error
    assert_eq!(
        run_smi("var t = {a: 3}; var p = new Proxy(t, { get: null }); p.a"),
        3
    );
    assert_eq!(
        run_smi("var t = {a: 3}; var p = new Proxy(t, { get: undefined }); p.a"),
        3
    );
}

#[test]
fn get_invariant_non_writable_non_configurable() {
    // the trap cannot lie about frozen data properties
    assert_type_error(
        r#"
        var t = {};
        Object.defineProperty(t, "attr", { value: 1, writable: false, configurable: false });
        var p = new Proxy(t, { get: function() { return 2; } });
        p.attr;
        "#,
    );
    // same value passes
    assert_eq!(
        run_smi(
            r#"
            var t = {};
            Object.defineProperty(t, "attr", { value: 1, writable: false, configurable: false });
            var p = new Proxy(t, { get: function() { return 1; } });
            p.attr
            "#,
        ),
        1
    );
    // accessor without a getter must read as undefined
    assert_type_error(
        r#"
        var t = {};
        Object.defineProperty(t, "attr", { get: undefined, configurable: false });
        var p = new Proxy(t, { get: function() { return 2; } });
        p.attr;
        "#,
    );
}

#[test]
fn get_revoked_proxy_throws() {
    assert_type_error(
        r#"
        var r = Proxy.revocable({}, {});
        r.revoke();
        r.proxy.attr;
        "#,
    );
}

// ---- set ---------------------------------------------------------------------

#[test]
fn set_trap_and_parameters() {
    assert_eq!(
        run_smi(
            r#"
            var target = {};
            var seen;
            var proxy = new Proxy(target, {
              set: function(t, k, v, r) { seen = (t === target) + (k === "x") + (v === 7) + (r === proxy); return true; }
            });
            proxy.x = 7;
            seen === 4 ? 1 : 0
            "#
        ),
        1
    );
}

#[test]
fn set_forwards_without_trap() {
    assert_eq!(
        run_smi("var t = {a: 1}; var p = new Proxy(t, {}); p.a = 9; p.a;"),
        9
    );
    // new properties land on the target
    assert_eq!(
        run_smi("var t = {}; var p = new Proxy(t, {}); p.fresh = 3; t.fresh;"),
        3
    );
    // arrays forward element stores
    assert_eq!(
        run_smi("var t = [1]; var p = new Proxy(t, {}); p[0] = 5; p[1] = 6; t[0] + t[1];"),
        11
    );
}

#[test]
fn set_invariant_frozen_property() {
    assert_type_error(
        r#"
        var t = {};
        Object.defineProperty(t, "attr", { value: 1, writable: false, configurable: false });
        var p = new Proxy(t, { set: function() { return true; } });
        p.attr = 2;
        "#,
    );
    assert_type_error(
        r#"
        var t = {};
        Object.defineProperty(t, "attr", { set: undefined, configurable: false });
        var p = new Proxy(t, { set: function() { return true; } });
        p.attr = 2;
        "#,
    );
}

// ---- has ---------------------------------------------------------------------

#[test]
fn has_trap_and_forward() {
    assert_eq!(
        run_smi(
            r#"
            var target = { a: 1 };
            var p = new Proxy(target, { has: function(t, k) { return k === "b"; } });
            (("b" in p) ? 1 : 0) + (("a" in p) ? 0 : 1)
            "#
        ),
        2
    );
    assert_eq!(
        run_smi("var p = new Proxy({a: 1}, {}); (\"a\" in p) ? 1 : 0"),
        1
    );
    // array length through the chain forwards
    assert_eq!(
        run_smi("var p = new Proxy([1, 2], {}); (\"length\" in p) ? 1 : 0"),
        1
    );
    assert_eq!(
        run_smi("var p = new Proxy([1, 2], {}); (0 in p) + (5 in p === false ? 1 : 0)"),
        2
    );
}

#[test]
fn has_invariant_cannot_hide_non_configurable() {
    assert_type_error(
        r#"
        var t = {};
        Object.defineProperty(t, "attr", { value: 1, configurable: false });
        var p = new Proxy(t, { has: function() { return false; } });
        "attr" in p;
        "#,
    );
    // non-extensible targets cannot lose existing properties either
    assert_type_error(
        r#"
        var t = { attr: 1 };
        Object.preventExtensions(t);
        var p = new Proxy(t, { has: function() { return false; } });
        "attr" in p;
        "#,
    );
}

// ---- delete ------------------------------------------------------------------

#[test]
fn delete_trap_and_forward() {
    assert_eq!(
        run_smi(
            r#"
            var t = { a: 1, b: 2 };
            var p = new Proxy(t, { deleteProperty: function(tg, k) { return k === "a"; } });
            (delete p.a) + (delete p.b ? 0 : 1)
            "#
        ),
        2
    );
    assert_eq!(
        run_smi(
            "var t = {a: 1}; var p = new Proxy(t, {}); (delete p.a) + (t.a === undefined ? 1 : 0)"
        ),
        2
    );
}

#[test]
fn delete_invariant_non_configurable() {
    assert_type_error(
        r#"
        var t = {};
        Object.defineProperty(t, "attr", { value: 1, configurable: false });
        var p = new Proxy(t, { deleteProperty: function() { return true; } });
        delete p.attr;
        "#,
    );
    assert_type_error(
        r#"
        var t = { attr: 1 };
        Object.preventExtensions(t);
        var p = new Proxy(t, { deleteProperty: function() { return true; } });
        delete p.attr;
        "#,
    );
}

// ---- defineProperty ------------------------------------------------------------

#[test]
fn define_property_trap() {
    assert_eq!(
        run_smi(
            r#"
            var t = {};
            var got;
            var p = new Proxy(t, {
              defineProperty: function(tg, k, desc) { got = [k, desc.value, desc.writable, desc.enumerable, desc.configurable]; return true; }
            });
            Object.defineProperty(p, "x", { value: 5, writable: true, enumerable: true, configurable: true });
            got[0] === "x" && got[1] === 5 && got[2] && got[3] && got[4] ? 1 : 0
            "#
        ),
        1
    );
}

#[test]
fn define_property_forwards_without_trap() {
    assert_eq!(
        run_smi(
            r#"
            var t = {};
            var p = new Proxy(t, {});
            Object.defineProperty(p, "x", { value: 5, configurable: true, writable: true, enumerable: true });
            t.x
            "#
        ),
        5
    );
}

#[test]
fn define_property_invariants() {
    // non-extensible target: no new properties
    assert_type_error(
        r#"
        var t = {};
        Object.preventExtensions(t);
        var p = new Proxy(t, { defineProperty: function() { return true; } });
        Object.defineProperty(p, "x", { value: 1 });
        "#,
    );
    // cannot claim non-configurable for a configurable target property
    assert_type_error(
        r#"
        var t = { x: 1 };
        var p = new Proxy(t, { defineProperty: function() { return true; } });
        Object.defineProperty(p, "x", { configurable: false });
        "#,
    );
}

// ---- extensibility --------------------------------------------------------------

#[test]
fn is_extensible_and_prevent_extensions() {
    assert_eq!(run_smi("var o = {}; Object.isExtensible(o) ? 1 : 0"), 1);
    assert_eq!(
        run_smi("var o = {}; Object.preventExtensions(o); Object.isExtensible(o) ? 1 : 0"),
        0
    );
    // through proxies: forward
    assert_eq!(
        run_smi(
            "var t = {}; var p = new Proxy(t, {}); Object.preventExtensions(p); Object.isExtensible(t) ? 1 : 0"
        ),
        0
    );
    // isExtensible trap must agree with the target
    assert_type_error(
        r#"
        var p = new Proxy({}, { isExtensible: function() { return false; } });
        Object.isExtensible(p);
        "#,
    );
    // preventExtensions trap returning true requires a non-extensible target
    assert_type_error(
        r#"
        var p = new Proxy({}, { preventExtensions: function() { return true; } });
        Object.preventExtensions(p);
        "#,
    );
    assert_eq!(
        run_smi(
            r#"
            var t = {};
            Object.preventExtensions(t);
            var p = new Proxy(t, { preventExtensions: function() { return true; } });
            Object.preventExtensions(p) === p ? 1 : 0
            "#
        ),
        1
    );
}

// ---- seal / freeze (ordinary) ---------------------------------------------------

#[test]
fn seal_and_freeze_ordinary_objects() {
    assert_eq!(
        run_smi(
            r#"
            var o = { a: 1, b: 2 };
            Object.seal(o);
            var d = Object.getOwnPropertyDescriptor(o, "a");
            (d.configurable === false ? 1 : 0) + (d.writable === true ? 1 : 0)
            "#
        ),
        2
    );
    assert_eq!(
        run_smi(
            r#"
            var o = { a: 1 };
            Object.freeze(o);
            var d = Object.getOwnPropertyDescriptor(o, "a");
            (d.configurable === false ? 1 : 0) + (d.writable === false ? 1 : 0)
            "#
        ),
        2
    );
}

// ---- apply / construct ----------------------------------------------------------

#[test]
fn apply_trap_and_forward() {
    assert_eq!(
        run_smi(
            r#"
            var target = function (a, b) { return a + b; };
            var p = new Proxy(target, { apply: function(t, thisArg, args) { return args[0] * 10; } });
            p(4, 1)
            "#
        ),
        40
    );
    assert_eq!(
        run_smi("var f = function (a) { return a + 1; }; var p = new Proxy(f, {}); p(1)"),
        2
    );
    assert_eq!(
        run_smi("var p = new Proxy(function(){}, {}); typeof p === \"function\" ? 1 : 0"),
        1
    );
    // non-callable proxies are not functions
    assert_eq!(
        run_smi("var p = new Proxy({}, {}); typeof p === \"object\" ? 1 : 0"),
        1
    );
}

#[test]
fn construct_trap_and_forward() {
    assert_eq!(
        run_smi(
            r#"
            function C(x) { this.x = x; }
            var p = new Proxy(C, { construct: function(t, args, nt) { return { tag: args[0] }; } });
            new p(42).tag
            "#
        ),
        42
    );
    assert_eq!(
        run_smi(
            r#"
            function C(x) { this.x = x; }
            var p = new Proxy(C, {});
            new p(7).x
            "#
        ),
        7
    );
    // the trap result must be an object
    assert_type_error(
        r#"
        var p = new Proxy(function () {}, { construct: function() { return 5; } });
        new p();
        "#,
    );
    // a non-constructor target cannot be constructed through a proxy
    assert_type_error(
        r#"
        var p = new Proxy(function () {}, {});
        p;
        new (new Proxy({}, { get: function() { throw 0; } }));
        "#,
    );
}

// ---- revocable --------------------------------------------------------------------

#[test]
fn revocable_and_revoke() {
    assert_eq!(
        run_smi(
            r#"
            var r = Proxy.revocable({ a: 1 }, {});
            r.proxy.a === 1 && typeof r.revoke === "function" ? 1 : 0
            "#
        ),
        1
    );
    // revoke makes every internal method throw
    assert_type_error(
        r#"
        var r = Proxy.revocable({ a: 1 }, {});
        r.revoke();
        r.revoke(); // idempotent
        r.proxy.a;
        "#,
    );
    assert_type_error(
        r#"
        var r = Proxy.revocable({ a: 1 }, {});
        r.revoke();
        "a" in r.proxy;
        "#,
    );
    assert_type_error(
        r#"
        var r = Proxy.revocable({ a: 1 }, {});
        r.revoke();
        delete r.proxy.a;
        "#,
    );
    // a revoked target of another proxy is fine at creation time
    assert_eq!(
        run_smi(
            r#"
            var r = Proxy.revocable({}, {});
            r.revoke();
            var p = new Proxy(r.proxy, {});
            1
            "#
        ),
        1
    );
}

#[test]
fn revoked_proxy_traps_throw() {
    // set through a revoked proxy
    assert_type_error("var r = Proxy.revocable({}, {}); r.revoke(); r.proxy.x = 1;");
    // defineProperty
    assert_type_error(
        "var r = Proxy.revocable({}, {}); r.revoke(); Object.defineProperty(r.proxy, 'x', { value: 1 });",
    );
    // isExtensible
    assert_type_error("var r = Proxy.revocable({}, {}); r.revoke(); Object.isExtensible(r.proxy);");
    // a revoked callable proxy
    assert_type_error("var r = Proxy.revocable(function(){}, {}); r.revoke(); r.proxy();");
}
