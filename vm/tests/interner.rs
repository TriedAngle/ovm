use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DenseString, VM};

#[test]
fn interning_deduplicates_and_preserves_content() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut ctx = vm.attach();

    ctx.handle_scope(|ctx, scope| {
        let a = ctx.intern(&scope, "hello");
        let text = String::from("hello");
        let b = ctx.intern(&scope, &text);
        let c = ctx.intern(&scope, "world");

        // same string, same slot content
        let (ab_eq, ac_ne) = {
            let heap = &*ctx.heap();
            (
                a.as_tagged(heap).raw().to_bits() == b.as_tagged(heap).raw().to_bits(),
                a.as_tagged(heap).raw().to_bits() != c.as_tagged(heap).raw().to_bits(),
            )
        };
        assert!(ab_eq);
        assert!(ac_ne);

        // content round trip (compressed encoding: Latin1)
        let (text, hash_a, hash_b) = {
            let heap = &*ctx.heap();
            (
                a.as_tagged(heap).to_rust_string(heap),
                a.as_tagged(heap).hash(heap),
                b.as_tagged(heap).hash(heap),
            )
        };
        assert_eq!(text, "hello");
        assert_eq!(hash_a, hash_b);

        // re-interning an interned string forwards to the same entry
        let d = ctx.intern(&scope, &text);
        let bits_eq = {
            let heap = &*ctx.heap();
            a.as_tagged(heap).raw().to_bits() == d.as_tagged(heap).raw().to_bits()
        };
        assert!(bits_eq);
    });
}

#[test]
fn interning_compresses_utf16_to_latin1() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut ctx = vm.attach();

    ctx.handle_scope(|ctx, scope| {
        // "héllo" fits Latin1; "€" (0x20AC) does not
        let latin1_content = ctx.intern(&scope, "héllo");
        let utf16_content = ctx.intern(&scope, "€");

        // re-interning the same content through a heap string (the
        // keyed-lookup path) finds the canonical instance
        let again = DenseString::from_utf8(ctx.heap(), &scope, "héllo");
        let canonical = vm.interner().intern_value(ctx.heap(), &scope, &again);
        let bits_eq = {
            let heap = &*ctx.heap();
            latin1_content.as_tagged(heap).raw().to_bits()
                == canonical.as_tagged(heap).raw().to_bits()
        };
        assert!(bits_eq);

        {
            let heap = &*ctx.heap();
            use vm::{Encoding, MapKind};
            let l = latin1_content.as_tagged(heap);
            let u = utf16_content.as_tagged(heap);
            // encodings come from the map's kind bits
            assert_eq!(l.encoding(), Encoding::Latin1);
            assert_eq!(u.encoding(), Encoding::Utf16);
            assert_eq!(l.len(), 5);
            assert_eq!(u.len(), 1);
            // code-unit access is O(1) in both encodings
            assert_eq!(l.code_unit(heap, 1), 0xE9);
            assert_eq!(u.code_unit(heap, 0), 0x20AC);
            let _ = MapKind::LATIN1;
        };
    });
}

#[test]
fn interning_is_thread_safe() {
    let vm = VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut ctx = vm.attach();

    let expected = ctx.handle_scope(|ctx, scope| {
        let shared = ctx.intern(&scope, "shared");
        let heap = &*ctx.heap();
        shared.as_tagged(heap).raw().to_bits()
    });

    let mut threads = Vec::new();
    for _ in 0..4 {
        let vm = vm.clone();
        threads.push(std::thread::spawn(move || {
            let mut ctx = vm.attach();
            ctx.handle_scope(|ctx, scope| {
                let shared = ctx.intern(&scope, "shared");
                let heap = &*ctx.heap();
                shared.as_tagged(heap).raw().to_bits()
            })
        }));
    }
    for t in threads {
        assert_eq!(t.join().unwrap(), expected);
    }
}
