//! Inline-cache behavior: every test targets a specific staleness hazard.

use bytecode::{Opcode, decode};
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{FeedbackVector, Smi, Tagged, Thread, Value, WeakFixedArray};

fn run(src: &str) -> Result<Value, vm::ScriptError> {
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    thread.eval::<vm::JavascriptCompiler>(src)
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

fn run_bool(src: &str) -> bool {
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let result = thread.eval::<vm::JavascriptCompiler>(src).unwrap();
    let heap = thread.heap();
    result == heap.known().true_object.as_tagged(heap).raw()
}

// -- correctness: the load must always agree with the uncached lookup ----

#[test]
fn mono_hit_reads_through_value_changes() {
    assert_eq!(
        run_smi("function f(o){ return o.x; } const o = {x:1}; f(o); o.x = 2; f(o);"),
        2
    );
}

#[test]
fn late_added_prototype_property_appears() {
    // first call caches NotFound on the whole chain; adding x to the proto
    // changes the proto's map and the next call must see the value
    assert!(run_bool(
        "function P(){} const o = new P();
             function f(r){ return r.x; }
             const a = f(o); P.prototype.x = 1; const b = f(o);
             a === undefined && b === 1;"
    ));
}

#[test]
fn deleted_prototype_property_disappears() {
    assert!(run_bool(
        "function P(){} P.prototype.x = 1; const o = new P();
             function f(r){ return r.x; }
             const a = f(o); delete P.prototype.x; const b = f(o);
             a === 1 && b === undefined;"
    ));
}

#[test]
fn own_property_shadows_cached_prototype_hit() {
    // cache a prototype hit first, then shadow it with an own property
    assert!(run_bool(
        "function P(){} P.prototype.x = 1; const o = new P();
             function f(r){ return r.x; }
             const a = f(o); o.x = 2; const b = f(o);
             a === 1 && b === 2;"
    ));
}

#[test]
fn intermediate_prototype_addition_reroutes_lookup() {
    // cache a hit on the grandparent, then add x to the intermediate
    // prototype: the chain check must miss and find the nearer property
    assert!(run_bool(
        "function GP(){} GP.prototype.x = 1;
             function MID(){} MID.prototype = new GP();
             const o = new MID();
             function f(r){ return r.x; }
             const a = f(o); MID.prototype.x = 2; const b = f(o);
             a === 1 && b === 2;"
    ));
}

#[test]
fn set_prototype_of_after_caching_honors_new_chain() {
    assert!(run_bool(
        "const p1 = {x: 1}; const o = {}; Object.setPrototypeOf(o, p1);
             function f(r){ return r.x; }
             const a = f(o);
             const p2 = {x: 2};
             Object.setPrototypeOf(o, p2);
             const b = f(o);
             a === 1 && b === 2;"
    ));
}

#[test]
fn getter_runs_each_call_with_receiver() {
    assert!(run_bool(
        "let n = 0; let seen = 0;
             const p = { get x() { n++; seen = this.v; return n; } };
             const o = {}; Object.setPrototypeOf(o, p); o.v = 7;
             function f(r){ return r.x; }
             const a = f(o); const b = f(o); const c = f(o);
             a === 1 && b === 2 && c === 3 && seen === 7;"
    ));
}

#[test]
fn deleted_accessor_falls_through() {
    assert!(run_bool(
        "let calls = 0;
             const p = { get x() { calls++; return 1; } };
             const o = {}; Object.setPrototypeOf(o, p);
             function f(r){ return r.x; }
             const a = f(o); const b = f(o); delete p.x;
             const c = f(o);
             a === 1 && b === 1 && calls === 2 && c === undefined;"
    ));
}

#[test]
fn deleted_own_property_after_caching() {
    assert!(run_bool(
        "const o = {x: 1};
             function f(r){ return r.x; }
             const a = f(o); delete o.x; const b = f(o);
             a === 1 && b === undefined;"
    ));
}

#[test]
fn polymorphic_shapes_stay_correct() {
    assert!(run_bool(
        "function f(r){ return r.x; }
             const a = {x: 1}; const b = {y: 0, x: 2}; const c = {z: 0, w: 0, x: 3};
             const r1 = f(a) + f(b) + f(c);
             const r2 = f(b) + f(a) + f(c);
             r1 === 6 && r2 === 6;"
    ));
}

#[test]
fn megamorphic_site_stays_correct() {
    let mut src = String::from("function f(r){ return r.x; } let acc = 0;");
    // six distinct shapes cycle the site through polymorphic into
    // megamorphic; every load must still see the right value
    for i in 0..6 {
        let mut literal = String::new();
        for j in 0..i {
            literal.push_str(&format!("p{j}: 0, "));
        }
        src.push_str(&format!("const o{i} = {{{literal}x: {i}}}; "));
    }
    src.push_str("for (let i = 0; i < 6; i++) { acc += f([o0,o1,o2,o3,o4,o5][i]); }");
    src.push_str("acc;");
    assert_eq!(run_smi(&src), 15);
}

// -- globals ---------------------------------------------------------------

#[test]
fn global_load_caches_and_reads_through() {
    assert_eq!(
        run_smi("var g = 1; function f(){ return g; } f(); g = 2; f();"),
        2
    );
}

#[test]
fn global_added_after_caching_is_found() {
    assert!(run_bool(
        "var a = 1;
             function f(){ return a; }
             const first = f();   // caches the global load
             var b = 5;           // changes the global object's map
             const second = f();
             first === 1 && second === 1 && b === 5;"
    ));
}

#[test]
fn typeof_undeclared_global_before_and_after_assignment() {
    assert!(run_bool(
        "function f(){ return typeof qq; }
         const before = f();
         qq = 3;
         const after = f();
         before === \"undefined\" && after === \"number\";"
    ));
}

// -- white-box: feedback state ----------------------------------------------
// The script exposes `f` as its completion value; its feedback vector is
// inspected directly.

fn run_value(src: &str) -> (Value, Thread) {
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let result = thread.eval::<vm::JavascriptCompiler>(src).unwrap();
    (result, thread)
}

fn feedback_vector_of<'a>(heap: &'a vm::Heap, result: Value) -> Option<Tagged<'a, FeedbackVector>> {
    // Safety: strong result value of a completed script run.
    let f = unsafe { Tagged::<vm::Object>::from_value_unchecked(result) };
    let info = f
        .as_ref()
        .slot(heap, 0)
        .get(heap)
        .get_as::<vm::CallableInfoObject>()
        .expect("closure info");
    info.as_ref().feedback(heap)
}

fn first_named_load_slot(heap: &vm::Heap, result: Value) -> usize {
    let f = unsafe { Tagged::<vm::Object>::from_value_unchecked(result) };
    let info = f
        .as_ref()
        .slot(heap, 0)
        .get(heap)
        .get_as::<vm::CallableInfoObject>()
        .expect("closure info");
    let code = info.as_ref().bytecode.get(heap);
    let bytes = code.as_ref().as_slice();
    let mut pc = 0;
    while pc < bytes.len() {
        let (op, ops, next) = decode(bytes, pc);
        if op == Opcode::LoadNamedProperty {
            return ops.idx(2);
        }
        pc = next;
    }
    panic!("script has no named load site");
}

fn decode_handler(word: Value) -> Option<(i64, i64)> {
    Smi::decode(word).map(|s| {
        let v = s.value();
        (v & 0xff, v >> 8)
    })
}

#[test]
fn own_field_hit_goes_monomorphic() {
    let (result, mut thread) =
        run_value("function f(o){ return o.x; } const o = {x: 1}; f(o); f(o); f;");
    let heap = &*thread.heap();
    let slot = first_named_load_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let state = vector.as_ref().slot(slot);
    let word = state.raw();
    assert!(
        word.is_ptr() && word.is_weak_ptr(),
        "mono state is a weak map"
    );
    let handler = vector.as_ref().slot(slot + 1).raw();
    let (kind, offset) = decode_handler(handler).expect("own-field Smi handler");
    assert_eq!((kind, offset), (0, 0), "OwnField at offset 0");
}

#[test]
fn second_shape_goes_polymorphic() {
    let (result, mut thread) = run_value(
        "function f(o){ return o.x; }
         const a = {x: 1}; const b = {y: 0, x: 2};
         f(a); f(b); f;",
    );
    let heap = &*thread.heap();
    let slot = first_named_load_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let word = vector.as_ref().slot(slot).raw();
    let pairs = unsafe { Tagged::<Value>::from_value_unchecked(word) }
        .get_as::<WeakFixedArray>()
        .expect("poly state is a WeakFixedArray");
    assert_eq!(pairs.as_ref().len(), 4, "two [map, handler] pairs");
}

#[test]
fn five_shapes_go_megamorphic() {
    let (result, mut thread) = run_value(
        "function f(o){ return o.x; }
         f({x:1}); f({y:1, x:1}); f({z:1, x:1}); f({w:1, x:1}); f({v:1, x:1}); f({u:1, x:1});
         f;",
    );
    let heap = &*thread.heap();
    let slot = first_named_load_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let word = vector.as_ref().slot(slot).raw();
    assert_eq!(
        word,
        heap.known().megamorphic_symbol.as_tagged(heap).raw(),
        "state is the megamorphic sentinel"
    );
}

#[test]
fn prototype_hit_installs_chain_handler() {
    let (result, mut thread) = run_value(
        "const p = {x: 1}; const o = {}; Object.setPrototypeOf(o, p);
         function f(r){ return r.x; } f(o); f(o); f;",
    );
    let heap = &*thread.heap();
    let slot = first_named_load_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let state = vector.as_ref().slot(slot).raw();
    assert!(
        state.is_ptr() && state.is_weak_ptr(),
        "mono on the receiver map"
    );
    let handler = vector.as_ref().slot(slot + 1).raw();
    let chain = unsafe { Tagged::<Value>::from_value_unchecked(handler) }
        .get_as::<WeakFixedArray>()
        .expect("chain handler array");
    let heap = thread.heap();
    // [Smi ChainField, payload, Smi hop(-1), Smi owner(-1), weak map]
    assert_eq!(chain.as_ref().len(), 5);
    let head = decode_handler(chain.as_ref().get(heap, 0).raw()).expect("kind Smi");
    assert_eq!(head.0, 0, "ChainField kind");
    // after the chain broke (delete p.x), behavior stays correct — covered
    // by deleted_prototype_property_disappears
}

#[test]
fn try_load_hits_directly() {
    // end-to-end probe of the hit path: same-shape receiver, populated
    // vector, `try_load` must resolve without the lookup
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let r = thread
        .eval::<vm::JavascriptCompiler>(
            "function f(o){ return o.x; } const o = {x: 1}; f(o); f(o); [f, o];",
        )
        .unwrap();
    let heap = &*thread.heap();
    let arr = unsafe { Tagged::<vm::Object>::from_value_unchecked(r) };
    let f_word = arr.as_ref().element_value(heap, 0).expect("f");
    let o_word = arr.as_ref().element_value(heap, 1).expect("o");
    let f = unsafe { Tagged::<vm::Object>::from_value_unchecked(f_word.raw()) };
    let o = unsafe { Tagged::<vm::Object>::from_value_unchecked(o_word.raw()) };
    let info = f
        .as_ref()
        .slot(heap, 0)
        .get(heap)
        .get_as::<vm::CallableInfoObject>()
        .unwrap();
    let vector = info.as_ref().feedback(heap).expect("vector");
    let slot = {
        let code = info.as_ref().bytecode.get(heap);
        let bytes = code.as_ref().as_slice();
        let mut pc = 0;
        let mut found = None;
        while pc < bytes.len() {
            let (op, ops, next) = decode(bytes, pc);
            if op == Opcode::LoadNamedProperty {
                found = Some(ops.idx(2));
                break;
            }
            pc = next;
        }
        found.expect("named load site")
    };
    match vm::ic::InlineCache::try_load(heap, Some(vector), slot, o.erase()) {
        Some(vm::ic::Hit::Value(v)) => assert_eq!(Smi::decode(v.raw()).unwrap().value(), 1),
        Some(vm::ic::Hit::Getter(_)) => panic!("expected a value hit, got a getter"),
        Some(vm::ic::Hit::NotFound) => panic!("expected a value hit, got NotFound"),
        None => panic!("IC miss on a same-shape receiver: the hit path is broken"),
    }
}

// -- store IC --------------------------------------------------------------

#[test]
fn store_transition_repeats_correctly() {
    assert!(run_bool(
        "function add(o){ o.x = 1; }
             const a = {}; const b = {};
             add(a); add(b);
             a.x === 1 && b.x === 1 && a.p === undefined && b.p === undefined;"
    ));
    // second object takes the cached transition: same resulting shape
    assert!(run_bool(
        "function add(o){ o.x = 1; o.y = 2; }
         const a = {}; const b = {};
         add(a); add(b);
         a.x === 1 && a.y === 2 && b.x === 1 && b.y === 2;"
    ));
}

#[test]
fn store_to_existing_property_overwrites() {
    assert!(run_bool(
        "const o = {x: 1};
         function set(r){ r.x = 2; }
         set(o); set(o);
         o.x === 2;"
    ));
}

#[test]
fn store_shadows_inherited_writable_data() {
    assert!(run_bool(
        "function P(){} P.prototype.x = 1;
         const o = new P();
         function set(r){ r.x = 5; }
         set(o); set(o);
         o.x === 5 && P.prototype.x === 1;"
    ));
}

#[test]
fn store_setter_called_with_receiver_and_value() {
    assert!(run_bool(
        "let calls = 0; let got = 0; let seen = 0;
         const p = { set x(v) { calls++; got = v; seen = this.tag; } };
         const o = {}; Object.setPrototypeOf(o, p); o.tag = 7;
         function set(r){ r.x = 3; }
         set(o); set(o);
         calls === 2 && got === 3 && seen === 7;"
    ));
}

#[test]
fn deleted_setter_falls_back_to_data_add() {
    assert!(run_bool(
        "const p = { set x(v) { } };
         const o = {}; Object.setPrototypeOf(o, p);
         function set(r){ r.x = 1; }
         set(o);
         delete p.x;
         set(o);
         o.x === 1;"
    ));
}

#[test]
fn store_after_delete_of_own_readds() {
    assert!(run_bool(
        "const o = {x: 1};
         function set(r){ r.x = 2; }
         set(o);
         delete o.x;
         set(o);
         o.x === 2;"
    ));
}

#[test]
fn store_polymorphic_shapes_stay_correct() {
    assert!(run_bool(
        "function set(r, v){ r.x = v; }
         const a = {}; const b = {y: 0};
         set(a, 1); set(b, 2); set(a, 3); set(b, 4);
         a.x === 3 && b.x === 4 && a.y === undefined;"
    ));
}

#[test]
fn store_megamorphic_site_stays_correct() {
    let mut src = String::from("function set(r, v){ r.x = v; } let ok = true;");
    for i in 0..6 {
        let mut literal = String::new();
        for j in 0..i {
            literal.push_str(&format!("p{j}: 0, "));
        }
        src.push_str(&format!("const o{i} = {{{literal}}}; "));
    }
    src.push_str("for (let i = 0; i < 6; i++) { set([o0,o1,o2,o3,o4,o5][i], i * 10); }");
    src.push_str("o0.x === 0 && o1.x === 10 && o5.x === 50;");
    assert!(run_bool(&src));
}

#[test]
fn object_literals_stay_correct() {
    assert!(run_bool(
        "function make(v){ return {a: v, b: v + 1}; }
         const p = make(1); const q = make(5);
         p.a === 1 && p.b === 2 && q.a === 5 && q.b === 6;"
    ));
}

fn first_named_store_slot(heap: &vm::Heap, result: Value) -> usize {
    let f = unsafe { Tagged::<vm::Object>::from_value_unchecked(result) };
    let info = f
        .as_ref()
        .slot(heap, 0)
        .get(heap)
        .get_as::<vm::CallableInfoObject>()
        .expect("closure info");
    let code = info.as_ref().bytecode.get(heap);
    let bytes = code.as_ref().as_slice();
    let mut pc = 0;
    while pc < bytes.len() {
        let (op, ops, next) = decode(bytes, pc);
        if op == Opcode::StoreNamedProperty {
            return ops.idx(2);
        }
        pc = next;
    }
    panic!("script has no named store site");
}

#[test]
fn store_transition_handler_is_weak_target_map() {
    let (result, mut thread) = run_value("function add(o){ o.x = 1; } add({}); add({}); add;");
    let heap = &*thread.heap();
    let slot = first_named_store_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let state = vector.as_ref().slot(slot).raw();
    assert!(
        state.is_ptr() && state.is_weak_ptr(),
        "mono state is the pre-store map"
    );
    let handler = vector.as_ref().slot(slot + 1).raw();
    assert!(
        handler.is_ptr() && handler.is_weak_ptr(),
        "transition handler is a weak map word"
    );
    // the weak word must upgrade to the transition target map
    let strong = vm::Value::from_bits(handler.raw_addr() | vm::STRONG_PTR);
    let map = unsafe { Tagged::<vm::Object>::from_value_unchecked(strong) };
    assert!(
        map.as_ref().map_ref(heap).kind().kind() == vm::ObjectKind::Map,
        "handler upgrades to a Map"
    );
}

#[test]
fn store_field_handler_is_smi() {
    let (result, mut thread) =
        run_value("function set(o){ o.x = 2; } const o = {x: 1}; set(o); set(o); set;");
    let heap = &*thread.heap();
    let slot = first_named_store_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let state = vector.as_ref().slot(slot).raw();
    assert!(state.is_ptr() && state.is_weak_ptr(), "mono state");
    let handler = vector.as_ref().slot(slot + 1).raw();
    let (kind, offset) = decode_handler(handler).expect("StoreField Smi handler");
    assert_eq!((kind, offset), (0, 0), "StoreField at offset 0");
}

#[test]
fn store_second_shape_goes_polymorphic() {
    let (result, mut thread) = run_value(
        "function set(o){ o.x = 1; }
         set({}); set({y: 0});
         set;",
    );
    let heap = &*thread.heap();
    let slot = first_named_store_slot(heap, result);
    let vector = feedback_vector_of(heap, result).expect("feedback vector");
    let word = vector.as_ref().slot(slot).raw();
    let pairs = unsafe { Tagged::<Value>::from_value_unchecked(word) }
        .get_as::<WeakFixedArray>()
        .expect("poly state");
    assert_eq!(pairs.as_ref().len(), 4, "two [map, handler] pairs");
}

#[test]
fn object_create_builds_proto_chain() {
    assert!(run_bool(
        "const p = {x: 1};
         const o = Object.create(p);
         o.x === 1 && o instanceof Object;"
    ));
    assert!(run_bool(
        "const o = Object.create(null);
         o instanceof Object === false;"
    ));
    // throws on a primitive prototype argument (uncaught → sentinel)
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let r = thread.eval::<vm::JavascriptCompiler>("Object.create(1);");
    let heap = thread.heap();
    assert_eq!(
        r.unwrap(),
        heap.known().exception.as_tagged(heap).raw(),
        "primitive prototype must throw"
    );
}

// -- Kette multi-parent chains (constant parents, cached like JS) ---------

use bytecode::SourceMode;

fn run_kette_smi(src: &str) -> i64 {
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let value = thread
        .run_source(src, kette_compiler::compile_kette, SourceMode::Script)
        .expect("kette script runs");
    Smi::decode(value).expect("kette result is a smi").value()
}

#[test]
fn kette_parent_load_caches_and_reads_through() {
    assert_eq!(
        run_kette_smi(
            "let P = { x: 1 }
             let C = { parent*: P\n read: { self.x } }
             C.read()"
        ),
        1
    );
    // the cached chain must read through parent-slot writes
    assert_eq!(
        run_kette_smi(
            "let P = { x: 1 }
             let C = { parent*: P\n read: { self.x } }
             C.read()
             P.x = 2
             C.read()"
        ),
        2
    );
}

#[test]
fn kette_late_parent_property_appears() {
    // first read caches NotFound over the whole parent chain; adding x to
    // the parent changes its map and the next read must see it
    assert_eq!(
        run_kette_smi(
            "let P = { }
             let C = { parent*: P\n read: { self.x } }
             C.read()
             P.x = 5
             C.read()"
        ),
        5
    );
}

#[test]
fn kette_parent_name_read_is_cached() {
    // `self.parent` is a read-only parent-name slot served from the pair
    // array itself; the second read takes the cached ParentName handler
    assert_eq!(
        run_kette_smi(
            "let P = { x: 1 }
             let C = { parent*: P\n read: { self.parent.x } }
             C.read()
             C.read()"
        ),
        1
    );
}

#[test]
fn kette_multi_parent_priority_and_reroute() {
    // x starts only on the second parent: priority order must find it
    assert_eq!(
        run_kette_smi(
            "let P1 = { }
             let P2 = { x: 2 }
             let C = { a*: P1\n b*: P2\n read: { self.x } }
             C.read()"
        ),
        2
    );
    // adding x to the first parent must reroute the cached lookup
    assert_eq!(
        run_kette_smi(
            "let P1 = { }
             let P2 = { x: 2 }
             let C = { a*: P1\n b*: P2\n read: { self.x } }
             C.read()
             P1.x = 7
             C.read()"
        ),
        7
    );
}

#[test]
fn kette_object_literal_stores_cache() {
    // two literal creations exercise the cached store transition; both
    // objects must hold their own value
    assert_eq!(
        run_kette_smi(
            "let mk = { |v| { x: v } }
             let a = mk(3)
             let b = mk(4)
             b.x"
        ),
        4
    );
    assert_eq!(
        run_kette_smi(
            "let mk = { |v| { x: v } }
             let a = mk(3)
             let b = mk(4)
             a.x"
        ),
        3
    );
}

#[test]
fn kette_chain_handler_records_parent_hop() {
    // white-box: the cached handler's chain entry must carry the parent's
    // element index inside the receiver's pair array
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    let result = thread
        .run_source(
            "let P = { x: 1 }
             let C = { parent*: P\n read: { self.x } }
             C.read()
             C.read()
             C.read",
            kette_compiler::compile_kette,
            SourceMode::Script,
        )
        .expect("kette script runs");
    let heap = &*thread.heap();
    let f = unsafe { Tagged::<vm::Object>::from_value_unchecked(result) };
    let info = f
        .as_ref()
        .slot(heap, 0)
        .get(heap)
        .get_as::<vm::CallableInfoObject>()
        .expect("method info");
    let vector = info.as_ref().feedback(heap).expect("feedback");
    let code = info.as_ref().bytecode.get(heap);
    let bytes = code.as_ref().as_slice();
    let mut pc = 0;
    let mut slot = None;
    while pc < bytes.len() {
        let (op, ops, next) = decode(bytes, pc);
        if op == Opcode::LoadNamedProperty {
            slot = Some(ops.idx(2));
            break;
        }
        pc = next;
    }
    let slot = slot.expect("load site");
    let state = vector.as_ref().slot(slot).raw();
    assert!(
        state.is_ptr() && state.is_weak_ptr(),
        "mono on the receiver map"
    );
    let handler = vector.as_ref().slot(slot + 1).raw();
    let chain = unsafe { Tagged::<Value>::from_value_unchecked(handler) }
        .get_as::<WeakFixedArray>()
        .expect("chain handler");
    // [Smi ChainField, payload, Smi hop, Smi owner, weak parent map]
    assert_eq!(chain.as_ref().len(), 5);
    let hop = Smi::decode(chain.as_ref().get(heap, 2).raw())
        .unwrap()
        .value();
    let owner = Smi::decode(chain.as_ref().get(heap, 3).raw())
        .unwrap()
        .value();
    assert_eq!(
        (hop, owner),
        (1, -1),
        "first parent is pair element 1, owned by the receiver"
    );
}
