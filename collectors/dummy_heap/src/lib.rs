use core::alloc::Layout;
use core::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use vm::{AllocError, EdgeVisitable, Heap, LocalHeap, RawCell, RootVisitor, Value, WellKnown, Word};

#[derive(Debug, Clone, Copy)]
pub struct DummyHeapConfig {
    pub heap_size: usize,
}

impl Default for DummyHeapConfig {
    fn default() -> Self {
        Self {
            heap_size: 64 * 1024 * 1024,
        }
    }
}

pub struct DummyHeapState {
    start: NonNull<u8>,
    layout: Layout,
    offset: AtomicUsize,
    known: OnceLock<WellKnown>,
}

unsafe impl Send for DummyHeapState {}
unsafe impl Sync for DummyHeapState {}

impl DummyHeapState {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let align = layout.align().max(4);
        let mut offset = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = offset.next_multiple_of(align);
            let end = aligned
                .checked_add(layout.size())
                .ok_or(AllocError::OutOfMemory(layout))?;
            if end > self.layout.size() {
                return Err(AllocError::OutOfMemory(layout));
            }
            match self.offset.compare_exchange_weak(
                offset,
                end,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(unsafe { NonNull::new_unchecked(self.start.as_ptr().add(aligned)) });
                }
                Err(current) => offset = current,
            }
        }
    }

    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.layout.size()
    }

    pub fn contains(&self, addr: Word) -> bool {
        let start = self.start.as_ptr() as Word;
        (start..start + self.layout.size() as Word).contains(&addr)
    }
}

impl Drop for DummyHeapState {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.start.as_ptr(), self.layout) };
    }
}

pub struct DummyHeap {
    inner: Arc<DummyHeapState>,
}

impl DummyHeap {
    pub fn used(&self) -> usize {
        self.inner.used()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

impl Heap for DummyHeap {
    type Config = DummyHeapConfig;
    type Local = DummyLocalHeap;

    fn new(config: Self::Config) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(config.heap_size, 16)
            .expect("invalid heap size in DummyHeapConfig");
        let start = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .ok_or(AllocError::OutOfMemory(layout))?;
        Ok(Self {
            inner: Arc::new(DummyHeapState {
                start,
                layout,
                offset: AtomicUsize::new(0),
                known: OnceLock::new(),
            }),
        })
    }

    fn new_local(&self) -> Self::Local {
        DummyLocalHeap {
            shared: Arc::clone(&self.inner),
        }
    }

    fn set_known(&self, known: WellKnown) {
        assert!(
            self.inner.known.set(known).is_ok(),
            "well-known maps already installed"
        );
    }

    fn iterate_roots(&self, roots: &mut impl RootVisitor) {
        // The well-known objects' root table lives in the heap state.
        if let Some(known) = self.inner.known.get() {
            known.roots.visit_edges(roots);
        }
    }

    fn collect(&self) {
        // Never reclaims memory.
    }

    fn should_collect(&self) -> bool {
        false
    }

    fn gc_in_progress(&self) -> bool {
        false
    }

    fn contains(&self, addr: Word) -> bool {
        self.inner.contains(addr)
    }

    fn is_young(&self, _value: Value) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct DummyLocalHeap {
    shared: Arc<DummyHeapState>,
}

impl DummyLocalHeap {
    pub fn shared(&self) -> &DummyHeapState {
        &self.shared
    }
}

impl LocalHeap for DummyLocalHeap {
    fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.shared.allocate(layout)
    }

    fn known(&self) -> &WellKnown {
        self.shared
            .known
            .get()
            .expect("well-known maps not installed")
    }

    fn write_barrier(&self, _host: Value, _slot: &RawCell, _value: Value) {}

    fn collection_requested(&self) -> bool {
        false
    }

    fn park_for_collection(&self) {}

    fn gc_in_progress(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(size: usize) -> DummyLocalHeap {
        DummyHeap::new(DummyHeapConfig { heap_size: size })
            .unwrap()
            .new_local()
    }

    /// A local heap with the well-known maps installed — required for
    /// allocating object kinds whose map comes from `known()`.
    fn local_with_maps(size: usize) -> DummyLocalHeap {
        let heap = DummyHeap::new(DummyHeapConfig { heap_size: size }).unwrap();
        heap.install_well_known_maps();
        heap.new_local()
    }

    #[test]
    fn bump_allocates_forward_and_aligned() {
        let mut heap = local(1024);
        let a = heap.allocate_raw(Layout::new::<u8>()).unwrap();
        let b = heap.allocate_raw(Layout::new::<u64>()).unwrap();
        assert!(b.as_ptr() > a.as_ptr());
        assert!((b.as_ptr() as usize) % 8 == 0);
        assert!(heap.shared().contains(a.as_ptr() as Word));
        assert!(heap.shared().contains(b.as_ptr() as Word));
    }

    #[test]
    fn out_of_memory_when_full() {
        let mut heap = local(16);
        heap.allocate_raw(Layout::new::<[u8; 16]>()).unwrap();
        let err = heap.allocate_raw(Layout::new::<u8>()).unwrap_err();
        assert_eq!(err, AllocError::OutOfMemory(Layout::new::<u8>()));
    }

    #[test]
    fn concurrent_allocation() {
        let mut heap = local(1 << 20);
        let mut threads = Vec::new();
        for _ in 0..4 {
            let mut heap = heap.clone();
            threads.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    heap.allocate_raw(Layout::new::<u64>()).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(heap.shared().used(), 4 * 1000 * size_of::<u64>());
        heap.allocate_raw(Layout::new::<u64>()).unwrap();
    }

    use vm::{
        AccessorPair, Global, HeapPtr, Lookup, Map, MapInit, MapKind, Object, ObjectSlotsInit,
        SlotFlags, SlotName, Tagged, Value,
    };
    use vm::{FixedArray, FixedByteArray, HandleData, HandleScope, Register, Smi};

    fn scope(data: &HandleData) -> HandleScope<'_> {
        unsafe { HandleScope::from_raw(NonNull::from(data)) }
    }

    /// Allocate a map with descriptors given as (name smi, flags, payload).
    /// Requires well-known maps: the map's own map is the map map.
    fn alloc_map(
        heap: &mut DummyLocalHeap,
        value_slots: usize,
        descs: &[(i64, SlotFlags, Value)],
    ) -> Global<Map> {
        let descriptors: Vec<(SlotName, SlotFlags, Value)> = descs
            .iter()
            .map(|(name, flags, value)| (smi_name(*name), *flags, *value))
            .collect();
        let map = heap
            .allocate::<Map>(MapInit {
                kind: MapKind::OBJECT,
                value_slot_count: value_slots,
                descriptors: &descriptors,
            })
            .into_tagged();
        heap.known().roots.create_handle(map)
    }

    fn alloc_object(
        heap: &mut DummyLocalHeap,
        map: Global<Map>,
        values: &[Value],
    ) -> HeapPtr<Object> {
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let elements = heap.known().void.value();
        heap.allocate_object(
            &scope,
            ObjectSlotsInit {
                map,
                values,
                elements,
                length: 0,
            },
        )
        .into_ptr()
    }

    fn smi_name(n: i64) -> SlotName {
        SlotName::from(Tagged::smi(n).unwrap())
    }

    fn expect_data(lookup: Lookup<'_>, expected: i64) {
        match lookup {
            Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => {
                assert_eq!(Smi::decode(slot.inner()).unwrap().value(), expected)
            }
            _ => panic!("expected data lookup result"),
        }
    }

    #[test]
    fn lookup_resolves_own_const_value_and_parent_chain() {
        let mut heap = local_with_maps(1 << 16);

        // parent: smi(3) = const 99
        let parent_map = alloc_map(
            &mut heap,
            0,
            &[(3, SlotFlags::CONST, Smi::new(99).encode())],
        );
        let parent = alloc_object(&mut heap, parent_map, &[]);

        // child: smi(1) = value slot 0, smi(2) = const 42, parent stored in the map
        let child_map = alloc_map(
            &mut heap,
            1,
            &[
                (
                    1,
                    SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                    Smi::new(0).encode(),
                ),
                (2, SlotFlags::CONST, Smi::new(42).encode()),
                (
                    999,
                    SlotFlags::CONST.union(SlotFlags::PARENT),
                    parent.encode_strong(),
                ),
            ],
        );
        let child = alloc_object(&mut heap, child_map, &[Smi::new(7).encode()]);
        let child = unsafe { child.as_ref() };

        heap.no_gc(|nogc, heap| {
            expect_data(child.lookup(nogc, heap, smi_name(1)), 7); // own inline slot
            expect_data(child.lookup(nogc, heap, smi_name(2)), 42); // const in map
            expect_data(child.lookup(nogc, heap, smi_name(3)), 99); // inherited via parent
            assert!(matches!(
                child.lookup(nogc, heap, smi_name(4)),
                Lookup::NotFound
            ));
        });
    }

    #[test]
    fn slot_lookup_dispatches_smi_and_object() {
        let mut heap = local_with_maps(1 << 16);

        let map = alloc_map(
            &mut heap,
            1,
            &[(
                1,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );
        let obj = alloc_object(&mut heap, map, &[Smi::new(7).encode()]);

        heap.no_gc(|nogc, heap| {
            // smi receiver: looks up in the (descriptor-less) smi map
            // Safety: scratch stack registers, only read inside this no-GC scope.
            let smi = unsafe { Register::from_value(Smi::new(42).encode()) };
            assert!(matches!(
                smi.lookup(nogc, heap, smi_name(1)),
                Lookup::NotFound
            ));
            // object receiver: same result as the typed entry point
            // Safety: scratch stack registers, only read inside this no-GC scope.
            let obj_slot = unsafe { Register::from_value(obj.encode_strong()) };
            expect_data(obj_slot.lookup(nogc, heap, smi_name(1)), 7);
        });
    }

    #[test]
    fn lookup_via_specific_parent() {
        let mut heap = local_with_maps(1 << 16);

        // parent A: smi(3) = const 10; parent B: smi(3) = const 20
        let map_a = alloc_map(
            &mut heap,
            0,
            &[(3, SlotFlags::CONST, Smi::new(10).encode())],
        );
        let parent_a = alloc_object(&mut heap, map_a, &[]);
        let map_b = alloc_map(
            &mut heap,
            0,
            &[(3, SlotFlags::CONST, Smi::new(20).encode())],
        );
        let parent_b = alloc_object(&mut heap, map_b, &[]);

        // child: two named parents, both stored in the map
        let child_map = alloc_map(
            &mut heap,
            0,
            &[
                (
                    100,
                    SlotFlags::CONST.union(SlotFlags::PARENT),
                    parent_a.encode_strong(),
                ),
                (
                    101,
                    SlotFlags::CONST.union(SlotFlags::PARENT),
                    parent_b.encode_strong(),
                ),
            ],
        );
        let child = alloc_object(&mut heap, child_map, &[]);
        let child = unsafe { child.as_ref() };

        heap.no_gc(|nogc, heap| {
            expect_data(child.lookup(nogc, heap, smi_name(3)), 10); // default: first parent
            expect_data(
                child.lookup_parent(nogc, heap, smi_name(3), smi_name(101)),
                20,
            ); // directed
        });
    }

    #[test]
    fn lookup_returns_accessor_pair() {
        let mut heap = local_with_maps(1 << 16);

        let pair_ptr = heap
            .allocate::<AccessorPair>((Smi::new(111).encode(), Smi::new(222).encode()))
            .into_ptr();

        let map = alloc_map(
            &mut heap,
            0,
            &[(5, SlotFlags::ACCESSOR, pair_ptr.encode_strong())],
        );
        let obj = alloc_object(&mut heap, map, &[]);
        let obj = unsafe { obj.as_ref() };

        heap.no_gc(|nogc, heap| match obj.lookup(nogc, heap, smi_name(5)) {
            Lookup::Accessor { pair, .. } => {
                assert_eq!(Smi::decode(pair.get.get().erase()).unwrap().value(), 111);
                assert_eq!(Smi::decode(pair.set.get().erase()).unwrap().value(), 222);
            }
            _ => panic!("expected accessor lookup result"),
        });
    }

    #[test]
    fn token_bulk_allocates_and_tracks_remaining() {
        let mut heap = local_with_maps(1 << 16);
        let la = FixedArray::layout_for(2);
        let lb = FixedByteArray::layout_for(8);
        let total = Layout::from_size_align(la.size() + lb.size(), 16).unwrap();
        let before = heap.shared().used();
        {
            let tok = heap.allocate_token(total);
            assert_eq!(tok.remaining(), la.size() + lb.size());
            let a = tok.allocate::<FixedArray>(&[Smi::new(0).encode(); 2]);
            let b = tok.allocate::<FixedByteArray>(&[0u8; 8]);
            assert_ne!(a.as_ptr() as *mut u8, b.as_ptr() as *mut u8);
            assert_eq!(tok.remaining(), 0);
        }
        // the token region starts 16-aligned, so account for the padding
        let start = before.next_multiple_of(16);
        assert_eq!(heap.shared().used(), start + la.size() + lb.size());
    }

    #[test]
    fn token_handles_outlive_the_token() {
        let mut heap = local_with_maps(1 << 16);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let la = FixedArray::layout_for(2);
        let lb = FixedByteArray::layout_for(8);
        let total = Layout::from_size_align(la.size() + lb.size(), 16).unwrap();

        let (ha, hb) = {
            let tok = heap.allocate_token(total);
            let ha = tok
                .allocate::<FixedArray>(&[Smi::new(0).encode(); 2])
                .into_handle(&scope);
            let hb = tok
                .allocate::<FixedByteArray>(&[0u8; 8])
                .into_handle(&scope);
            (ha, hb)
        }; // token dropped here (full use verified)

        // the handles stay rooted and point into the heap
        assert_ne!(ha.value().to_bits(), hb.value().to_bits());
        assert!(heap.shared().contains(ha.value().raw_addr()));
        assert!(heap.shared().contains(hb.value().raw_addr()));
    }

    #[test]
    fn token_fresh_allocations_coexist_and_promote() {
        let mut heap = local_with_maps(1 << 16);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let la = FixedArray::layout_for(1);
        let total = Layout::from_size_align(2 * la.size(), 16).unwrap();

        let tok = heap.allocate_token(total);
        // multiple Fresh alive at once (shared borrows of the token)
        let a = tok.allocate::<FixedArray>(&[Smi::new(0).encode()]);
        let b = tok.allocate::<FixedArray>(&[Smi::new(0).encode()]);
        let ha = a.into_handle(&scope);
        let hb = b.into_handle(&scope);
        assert_ne!(ha.value().to_bits(), hb.value().to_bits());
    }

    #[test]
    fn token_enter_no_gc_allocates_refs() {
        let mut heap = local_with_maps(1 << 16);
        let lb = FixedByteArray::layout_for(8);
        let total = Layout::from_size_align(2 * lb.size(), 16).unwrap();

        let tok = heap.allocate_token(total);
        tok.enter_no_gc(|nogc, _heap| {
            let a = tok.allocate_ref::<FixedByteArray>(&[0u8; 8], nogc);
            let b = tok.allocate_ref::<FixedByteArray>(&[0u8; 8], nogc);
            a.set(0, 1);
            b.set(0, 42);
            b.set(1, 7);
            assert_eq!(a.get(0), 1);
            assert_eq!(b.get(0), 42);
            assert_eq!(b.get(1), 7);
        });
    }

    #[test]
    fn allocate_token_enter_no_gc_combines_both() {
        let mut heap = local_with_maps(1 << 16);
        let lb = FixedByteArray::layout_for(4);
        let total = Layout::from_size_align(lb.size(), 16).unwrap();
        heap.allocate_token_enter_nogc(total, |tok, nogc, _heap| {
            let a = tok.allocate_ref::<FixedByteArray>(&[0u8; 4], nogc);
            a.set(0, 1);
            assert_eq!(a.get(0), 1);
        });
    }

    #[test]
    fn allocate_enter_no_gc_gives_ref_instantly() {
        let mut heap = local_with_maps(1 << 16);
        heap.allocate_enter_nogc::<FixedByteArray, _>(&[1u8, 2, 3, 4], |bytes, _nogc, _heap| {
            assert_eq!(bytes.as_slice(), &[1, 2, 3, 4]);
        });
    }

    #[test]
    #[should_panic(expected = "allocation token dropped")]
    fn token_drop_requires_full_use() {
        let mut heap = local_with_maps(1 << 16);
        let total = Layout::from_size_align(4096, 16).unwrap();
        let tok = heap.allocate_token(total);
        let _ = tok.allocate::<FixedByteArray>(&[0u8; 4]);
        // dropped with most of the reservation unused
    }

    #[test]
    #[should_panic(expected = "allocation token exhausted")]
    fn token_allocate_checks_capacity() {
        let mut heap = local_with_maps(1 << 16);
        let total = Layout::from_size_align(64, 16).unwrap();
        let tok = heap.allocate_token(total);
        let _ = tok.allocate::<FixedByteArray>(&[0u8; 4096]);
    }
}
