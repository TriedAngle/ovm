use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::VM;

fn run(label: &str, src: &str) {
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    thread.eval::<JavascriptCompiler>(src).expect("runs");
    if let Some(ex) = thread.take_pending_exception() {
        panic!("{label}: pending exception: {ex:?}");
    }
}

const CHECK: &str = "function check(c, m) { if (!c) { throw new Error(m); } }\n";

/// `[[Construct]]` semantics through the fast path: receiver synthesis,
/// argument passing, primitive-vs-object return, and prototype methods.
/// (`become` has no strict-equality/instanceof/typeof opcodes, so these use
/// loose `==`.)
#[test]
fn construct_fast_path() {
    run(
        "receiver",
        &format!(
            "{CHECK}
      function C(x) {{ this.x = x; }}
      var c = new C(7);
      check(c.x == 7, 'field');"
        ),
    );
    run(
        "no_args",
        &format!(
            "{CHECK}
      function E() {{ this.tag = 1; }}
      var e = new E();
      check(e.tag == 1, 'no-arg field');"
        ),
    );
    run(
        "primitive_return",
        &format!(
            "{CHECK}
      function C() {{ this.x = 1; return 5; }}
      var c = new C();
      check(c.x == 1, 'primitive return keeps receiver');
      function D() {{ this.y = 2; return undefined; }}
      var d = new D();
      check(d.y == 2, 'undefined return keeps receiver');"
        ),
    );
    run(
        "object_return",
        &format!(
            "{CHECK}
      var replacement = {{ z: 9 }};
      function C() {{ this.x = 1; return replacement; }}
      var c = new C();
      check(c == replacement, 'object return wins');
      check(c.x == undefined, 'receiver discarded');"
        ),
    );
    run(
        "args_padding",
        &format!(
            "{CHECK}
      function C(a, b, c) {{ this.s = (a||0) + (b||0) + (c||0); }}
      var x = new C(1, 2, 3);
      check(x.s == 6, 'three args');
      var y = new C(10);
      check(y.s == 10, 'missing args');"
        ),
    );
    run(
        "prototype_method",
        &format!(
            "{CHECK}
      function C(x) {{ this.x = x; }}
      C.prototype.get = function () {{ return this.x; }};
      var c = new C(42);
      check(c.get() == 42, 'proto method');"
        ),
    );
    run(
        "nested",
        &format!(
            "{CHECK}
      function Inner(v) {{ this.v = v; }}
      function Outer(v) {{ this.inner = new Inner(v); }}
      var o = new Outer(3);
      check(o.inner.v == 3, 'nested new');"
        ),
    );
    run(
        "plain_call_unaffected",
        &format!(
            "{CHECK}
      function f() {{ return 5; }}
      check(f() == 5, 'plain call primitive');
      function g() {{ return {{ a: 1 }}; }}
      check(g().a == 1, 'plain call object');"
        ),
    );
    run(
        "loop",
        &format!(
            "{CHECK}
      function C(i) {{ this.i = i; }}
      var s = 0;
      for (var i = 0; i < 1000; i++) {{ s += new C(i).i; }}
      check(s == 499500, 'loop sum');"
        ),
    );
}
