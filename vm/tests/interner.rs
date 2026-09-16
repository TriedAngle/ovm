use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DenseString, VM};

#[test]
fn interning_deduplicates_and_preserves_content() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut ctx = vm.attach();

    ctx.handle_scope(|ctx, scope| {
        let a = ctx.intern(&scope, "hello");
        let text = String::from("hello");
        let b = ctx.intern(&scope, &text);
        let c = ctx.intern(&scope, "world");

        // same string, same slot content
        assert_eq!(a.value().to_bits(), b.value().to_bits());
        assert_ne!(a.value().to_bits(), c.value().to_bits());

        // content round trip (compressed encoding: Latin1)
        let (text, hash_a, hash_b) = ctx.heap().no_gc(|nogc| {
            (
                a.heap_ref(nogc).to_rust_string(nogc),
                a.heap_ref(nogc).hash(nogc),
                b.heap_ref(nogc).hash(nogc),
            )
        });
        assert_eq!(text, "hello");
        assert_eq!(hash_a, hash_b);

        // re-interning an interned string forwards to the same entry
        let d = ctx.intern(&scope, &text);
        assert_eq!(a.value().to_bits(), d.value().to_bits());
    });
}

#[test]
fn interning_compresses_utf16_to_latin1() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut ctx = vm.attach();

    ctx.handle_scope(|ctx, scope| {
        // "héllo" fits Latin1; "€" (0x20AC) does not
        let latin1_content = ctx.intern(&scope, "héllo");
        let utf16_content = ctx.intern(&scope, "€");

        // re-interning the same content through a heap string (the
        // keyed-lookup path) finds the canonical instance
        let again = DenseString::from_utf8(ctx.heap(), &scope, "héllo");
        let canonical = vm.interner().intern_value(ctx.heap(), &scope, &again);
        assert_eq!(
            latin1_content.value().to_bits(),
            canonical.value().to_bits()
        );

        ctx.heap().no_gc(|nogc| {
            use vm::{Encoding, MapKind};
            let l = latin1_content.heap_ref(nogc);
            let u = utf16_content.heap_ref(nogc);
            // encodings come from the map's kind bits
            assert_eq!(l.encoding(), Encoding::Latin1);
            assert_eq!(u.encoding(), Encoding::Utf16);
            assert_eq!(l.len(), 5);
            assert_eq!(u.len(), 1);
            // code-unit access is O(1) in both encodings
            assert_eq!(l.code_unit(nogc, 1), 0xE9);
            assert_eq!(u.code_unit(nogc, 0), 0x20AC);
            let _ = MapKind::LATIN1;
        });
    });
}

#[test]
fn interning_is_thread_safe() {
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut ctx = vm.attach();

    let expected = ctx.handle_scope(|ctx, scope| ctx.intern(&scope, "shared").value().to_bits());

    let mut threads = Vec::new();
    for _ in 0..4 {
        let vm = vm.clone();
        threads.push(std::thread::spawn(move || {
            let mut ctx = vm.attach();
            ctx.handle_scope(|ctx, scope| ctx.intern(&scope, "shared").value().to_bits())
        }));
    }
    for t in threads {
        assert_eq!(t.join().unwrap(), expected);
    }
}
