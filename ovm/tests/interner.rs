use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::VM;

#[test]
fn interning_deduplicates_and_preserves_content() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
    let mut ctx = vm.attach();

    ctx.handle_scope(|ctx, scope| {
        let a = ctx.intern(&scope, "hello");
        let b = ctx.intern(&scope, String::from("hello"));
        let c = ctx.intern(&scope, "world");

        // same string, same slot content
        assert_eq!(a.value().to_bits(), b.value().to_bits());
        assert_ne!(a.value().to_bits(), c.value().to_bits());

        // content round trip
        let (text, hash_a, hash_b) = ctx.heap().no_gc(|nogc, _| {
            (
                a.heap_ref(nogc).string().as_str(nogc).unwrap().to_owned(),
                a.heap_ref(nogc).string().hash(),
                b.heap_ref(nogc).string().hash(),
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
fn interning_is_thread_safe() {
    let vm = VM::new::<DummyHeap>(DummyHeapConfig::default()).unwrap();
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
