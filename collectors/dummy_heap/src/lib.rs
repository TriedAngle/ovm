use core::alloc::Layout;
use core::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use vm::{
    AllocError, GlobalHeap, GlobalVtable, HeapBackend, HeapStats, HeapVtable, RawCell, RootVisitor,
    TransitionLock, Value, WellKnown, Word,
};

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
    /// The currently installed well-known set. Every install leaks its
    /// `WellKnown` (via `Box::into_raw`) so that `known()` references stay
    /// valid at stable addresses for the lifetime of the heap.
    known: AtomicPtr<WellKnown>,
    transition_lock: TransitionLock,
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
    pub fn new(config: DummyHeapConfig) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(config.heap_size, 16)
            .expect("invalid heap size in DummyHeapConfig");
        let start = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .ok_or(AllocError::OutOfMemory(layout))?;
        Ok(Self {
            inner: Arc::new(DummyHeapState {
                start,
                layout,
                offset: AtomicUsize::new(0),
                known: AtomicPtr::new(core::ptr::null_mut()),
                transition_lock: TransitionLock::new(),
            }),
        })
    }

    pub fn used(&self) -> usize {
        self.inner.used()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

/// Concrete per-thread heap used behind [`DUMMY_HEAP_VTABLE`].
pub struct DummyLocalHeap {
    shared: Arc<DummyHeapState>,
}

impl DummyLocalHeap {
    pub fn shared(&self) -> &DummyHeapState {
        &self.shared
    }
}

fn erased_allocate_raw(local: *mut (), layout: Layout) -> Result<NonNull<u8>, AllocError> {
    let local: &DummyLocalHeap = unsafe { &*local.cast::<DummyLocalHeap>() };
    local.shared.allocate(layout)
}

fn erased_known(local: *const ()) -> &'static WellKnown {
    let local: &'static DummyLocalHeap = unsafe { &*local.cast::<DummyLocalHeap>() };
    let ptr = local.shared.known.load(Ordering::Acquire);
    assert!(!ptr.is_null(), "well-known maps not installed");
    unsafe { &*ptr }
}

fn erased_set_known(shared: *const (), known: WellKnown) {
    let state: &DummyHeapState = unsafe { &*shared.cast::<DummyHeapState>() };
    state
        .known
        .store(Box::into_raw(Box::new(known)), Ordering::Release);
}

fn erased_transition_lock(local: *const ()) -> TransitionLock {
    let local: &DummyLocalHeap = unsafe { &*local.cast::<DummyLocalHeap>() };
    local.shared.transition_lock.clone()
}

fn erased_write_barrier(_local: *const (), _host: Value, _slot: &RawCell, _value: Value) {}

fn erased_collection_requested(_local: *const ()) -> bool {
    false
}

fn erased_park_for_collection(_local: *const ()) {}

fn erased_gc_in_progress(_local: *const ()) -> bool {
    false
}

fn erased_drop_local(local: *mut ()) {
    unsafe { drop(Box::from_raw(local.cast::<DummyLocalHeap>())) };
}

fn erased_global_new_local(shared: *const ()) -> *mut () {
    let state: *const DummyHeapState = shared.cast();
    // clone-through-raw: take another reference for the new local
    unsafe { Arc::increment_strong_count(state) };
    Box::into_raw(Box::new(DummyLocalHeap {
        shared: unsafe { Arc::from_raw(state) },
    })) as *mut ()
}

fn erased_global_known(shared: *const ()) -> &'static WellKnown {
    let state: &'static DummyHeapState = unsafe { &*shared.cast::<DummyHeapState>() };
    let ptr = state.known.load(Ordering::Acquire);
    assert!(!ptr.is_null(), "well-known maps not installed");
    unsafe { &*ptr }
}

fn erased_global_iterate_roots(_shared: *const (), _roots: &mut dyn RootVisitor) {}

fn erased_global_collect(_shared: *const ()) {}

fn erased_global_should_collect(_shared: *const ()) -> bool {
    false
}

fn erased_global_gc_in_progress(_shared: *const ()) -> bool {
    false
}

fn erased_global_contains(shared: *const (), addr: Word) -> bool {
    let state: &DummyHeapState = unsafe { &*shared.cast::<DummyHeapState>() };
    state.contains(addr)
}

fn erased_global_is_young(_shared: *const (), _value: Value) -> bool {
    false
}

fn erased_global_stats(shared: *const ()) -> HeapStats {
    let state: &DummyHeapState = unsafe { &*shared.cast::<DummyHeapState>() };
    HeapStats {
        used: state.used(),
        capacity: state.capacity(),
    }
}

fn erased_global_drop_shared(shared: *mut ()) {
    unsafe { Arc::decrement_strong_count(shared.cast::<DummyHeapState>()) };
}

static DUMMY_HEAP_VTABLE: HeapVtable = HeapVtable {
    allocate_raw: erased_allocate_raw,
    known: erased_known,
    set_known: erased_set_known,
    transition_lock: erased_transition_lock,
    write_barrier: erased_write_barrier,
    collection_requested: erased_collection_requested,
    park_for_collection: erased_park_for_collection,
    gc_in_progress: erased_gc_in_progress,
    drop_local: erased_drop_local,
};

static DUMMY_GLOBAL_VTABLE: GlobalVtable = GlobalVtable {
    local_vtable: &DUMMY_HEAP_VTABLE,
    new_local: erased_global_new_local,
    known: erased_global_known,
    iterate_roots: erased_global_iterate_roots,
    collect: erased_global_collect,
    should_collect: erased_global_should_collect,
    gc_in_progress: erased_global_gc_in_progress,
    contains: erased_global_contains,
    is_young: erased_global_is_young,
    stats: erased_global_stats,
    drop_shared: erased_global_drop_shared,
};

impl HeapBackend for DummyHeap {
    type Config = DummyHeapConfig;

    fn new(config: Self::Config) -> Result<Self, AllocError> {
        DummyHeap::new(config)
    }

    fn into_global(self) -> GlobalHeap {
        let state = Arc::into_raw(self.inner) as *mut ();
        GlobalHeap::new(state, &DUMMY_GLOBAL_VTABLE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vm::PropertyDescriptor;

    fn local(size: usize) -> (GlobalHeap, Heap) {
        let global = DummyHeap::new(DummyHeapConfig { heap_size: size })
            .unwrap()
            .into_global();
        let heap = global.new_local();
        (global, heap)
    }

    /// A local heap with the well-known maps installed — required for
    /// allocating object kinds whose map comes from `known()`.
    fn install_well_known(global: &GlobalHeap) -> RootHandles {
        let mut local = global.new_local();
        let roots = unsafe { RootHandles::new(128, Smi::new(0).encode()) };
        // throwaway string table: the interned well-known strings are kept
        // alive by the table's strong entries while the table itself lives
        let interner = vm::StringInterner::new();
        vm::bootstrap_basics(&mut local, &roots);
        vm::intern_well_known_strings(&mut local, &interner);
        vm::bootstrap_well_known(&mut local, &roots);
        roots
    }

    fn local_with_maps(size: usize) -> (GlobalHeap, Heap, RootHandles) {
        let global = DummyHeap::new(DummyHeapConfig { heap_size: size })
            .unwrap()
            .into_global();
        let roots = install_well_known(&global);
        let heap = global.new_local();
        (global, heap, roots)
    }

    #[test]
    fn bump_allocates_forward_and_aligned() {
        let (global, mut heap) = local(1024);
        let a = heap.allocate_raw(Layout::new::<u8>()).unwrap();
        let b = heap.allocate_raw(Layout::new::<u64>()).unwrap();
        assert!(b.as_ptr() > a.as_ptr());
        assert!((b.as_ptr() as usize) % 8 == 0);
        assert!(global.contains(a.as_ptr() as Word));
        assert!(global.contains(b.as_ptr() as Word));
    }

    #[test]
    fn out_of_memory_when_full() {
        let (_global, mut heap) = local(16);
        heap.allocate_raw(Layout::new::<[u8; 16]>()).unwrap();
        let err = heap.allocate_raw(Layout::new::<u8>()).unwrap_err();
        assert_eq!(err, AllocError::OutOfMemory(Layout::new::<u8>()));
    }

    #[test]
    fn concurrent_allocation() {
        let (global, mut heap) = local(1 << 20);
        let mut threads = Vec::new();
        for _ in 0..4 {
            let mut heap = global.new_local();
            threads.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    heap.allocate_raw(Layout::new::<u64>()).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(global.stats().used, 4 * 1000 * size_of::<u64>());
        heap.allocate_raw(Layout::new::<u64>()).unwrap();
    }

    #[test]
    fn well_known_has_no_smi_placeholders_after_install() {
        let global = DummyHeap::new(DummyHeapConfig { heap_size: 1 << 20 })
            .unwrap()
            .into_global();
        let _roots = install_well_known(&global);
        let k = global.known();
        let values = [
            k.map_map.value(),
            k.void.value(),
            k.undefined.value(),
            k.null.value(),
            k.false_object.value(),
            k.true_object.value(),
            k.smi_map.value(),
            k.float_map.value(),
            k.array_map.value(),
            k.byte_array_map.value(),
            k.string_map.value(),
            k.symbol_map.value(),
            k.accessor_pair_map.value(),
            k.callable_map.value(),
            k.handler_table_map.value(),
            k.context_map.value(),
            k.object_prototype.value(),
            k.error_prototype.value(),
            k.error_map.value(),
            k.empty_context.value(),
        ];
        assert!(
            values.iter().all(|v| !v.is_smi()),
            "an SMI sentinel placeholder leaked into the installed set"
        );
    }

    #[test]
    fn install_well_known_maps_can_be_repeated() {
        let global = DummyHeap::new(DummyHeapConfig { heap_size: 1 << 20 })
            .unwrap()
            .into_global();
        let _roots = install_well_known(&global);
        let first_void = global.known().void.value();
        // a second install builds a fresh object graph and replaces the set
        let _roots = install_well_known(&global);
        let second_void = global.known().void.value();
        assert_ne!(first_void, second_void, "rebuilds a fresh graph");
        // both graphs stay rooted (old WellKnowns are intentionally leaked)
        assert!(global.contains(first_void.raw_addr()));
        assert!(global.contains(second_void.raw_addr()));
        // the heap remains fully usable afterwards
        let mut local = global.new_local();
        let bytes = local.allocate::<FixedByteArray>(&[1u8, 2, 3]).into_ptr();
        local.no_gc(|_nogc| {
            assert_eq!(unsafe { bytes.as_ref() }.as_slice(), &[1, 2, 3]);
        });
    }

    #[test]
    fn erased_heap_roundtrip() {
        let global = DummyHeap::new(DummyHeapConfig { heap_size: 1 << 20 })
            .unwrap()
            .into_global();
        let _roots = install_well_known(&global);

        assert_eq!(global.stats().capacity, 1 << 20);
        assert!(global.stats().used > 0);
        let _ = global.known();

        let mut local = global.new_local();
        let before = global.stats().used;
        let layout = FixedByteArray::layout_for(4);
        let raw = local.allocate_raw(layout).unwrap();
        assert!(global.contains(raw.as_ptr() as Word));
        assert!(global.stats().used >= before + layout.size());

        local.no_gc(|_nogc| {
            assert!(_nogc.known().void.value().is_strong_ptr());
            assert!(!_nogc.gc_in_progress());
        });
        drop(local);
        let _ = global.stats();
    }

    use vm::{
        AccessorPair, Float, Global, Heap, HeapPtr, Lookup, Map, MapInit, MapKind, Object,
        ObjectSlotsInit, RootHandles, SlotFlags, SlotName, StoreOutcome, StoreSemantics, Tagged,
        Value, VmError,
    };
    use vm::{
        FixedArray, FixedByteArray, HandleData, HandleScope, InternedString, Smi,
        string_content_hash,
    };

    fn scope(data: &HandleData) -> HandleScope<'_> {
        unsafe { HandleScope::from_raw(NonNull::from(data)) }
    }

    /// Allocate a map with descriptors given as (name smi, flags, payload).
    /// Requires well-known maps: the map's own map is the map map.
    fn alloc_map(
        heap: &mut Heap,
        roots: &RootHandles,
        kind: MapKind,
        value_slots: usize,
        descs: &[(i64, SlotFlags, Value)],
    ) -> Global<Map> {
        let descriptors: Vec<(SlotName, SlotFlags, Value)> = descs
            .iter()
            .map(|(name, flags, value)| (smi_name(*name), *flags, *value))
            .collect();
        let map = heap
            .allocate::<Map>(MapInit {
                kind,
                value_slot_count: value_slots,
                descriptors: &descriptors,
                prototype: roots.create_handle(Tagged::from_value(heap.known().null.value())),
            })
            .into_tagged();
        roots.create_handle(map)
    }

    /// Like `alloc_map` but with an explicit prototype value.
    fn alloc_map_proto(
        heap: &mut Heap,
        roots: &RootHandles,
        kind: MapKind,
        value_slots: usize,
        descs: &[(i64, SlotFlags, Value)],
        proto: Value,
    ) -> Global<Map> {
        let descriptors: Vec<(SlotName, SlotFlags, Value)> = descs
            .iter()
            .map(|(name, flags, value)| (smi_name(*name), *flags, *value))
            .collect();
        let map = heap
            .allocate::<Map>(MapInit {
                kind,
                value_slot_count: value_slots,
                descriptors: &descriptors,
                prototype: roots.create_handle(Tagged::from_value(proto)),
            })
            .into_tagged();
        roots.create_handle(map)
    }

    fn alloc_object(heap: &mut Heap, map: Global<Map>, values: &[Value]) -> HeapPtr<Object> {
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let elements = heap.known().empty_fixed_array.erase();
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
            Lookup::Data { slot, .. } => {
                assert_eq!(Smi::decode(slot.inner()).unwrap().value(), expected)
            }
            _ => panic!("expected data lookup result"),
        }
    }

    #[test]
    fn bootstrap_maps_have_void_transitions() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        heap.no_gc(|nogc| {
            let known = nogc.known();
            let void = known.void.value();
            for map in [
                known.map_map,
                known.smi_map,
                known.float_map,
                known.array_map,
                known.byte_array_map,
                known.string_map,
                known.symbol_map,
                known.accessor_pair_map,
                known.callable_map,
            ] {
                assert_eq!(map.heap_ref(nogc).transitions.inner(), void);
            }
        });
    }

    #[test]
    fn fresh_map_has_no_transitions() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let map = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        heap.no_gc(|nogc| {
            let map = map.heap_ref(nogc);
            assert_eq!(map.transitions.inner(), nogc.known().void.value());
            assert!(
                map.find_transition(nogc, smi_name(1), SlotFlags::VALUE)
                    .is_none()
            );
        });
    }

    #[test]
    fn find_transition_matches_name_and_derived_flags() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let flags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        // parent {}, child {smi(1) -> slot 0}
        let parent = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let child = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(1, flags, Smi::new(0).encode())],
        );
        // hand-built transition pairs [name, target]
        let pairs = heap
            .allocate::<FixedArray>(&[smi_name(1).value(), child.value()])
            .into_tagged();

        heap.no_gc(|nogc| {
            let parent_ref = parent.heap_ref(nogc);
            parent_ref
                .transitions
                .set(nogc.heap(), parent.value(), pairs);

            let found = parent_ref
                .find_transition(nogc, smi_name(1), flags)
                .expect("transition by name and flags");
            assert_eq!(found.into_ptr().as_ptr(), child.get().as_ptr());

            // same name but different attributes is a different transition
            let other = flags.union(SlotFlags::ENUMERABLE);
            assert!(
                parent_ref
                    .find_transition(nogc, smi_name(1), other)
                    .is_none()
            );
            // unknown name
            assert!(
                parent_ref
                    .find_transition(nogc, smi_name(2), flags)
                    .is_none()
            );
        });
    }

    /// Root a freshly allocated object pointer in the scope.
    fn root_object<'s>(scope: &'s HandleScope<'_>, ptr: HeapPtr<Object>) -> vm::Handle<'s, Object> {
        scope
            .create_handle(Tagged::from_ptr(ptr))
            .expect("object pointer is strong")
    }

    /// Root a slot name in the scope.
    fn root_name<'s>(scope: &'s HandleScope<'_>, name: SlotName) -> vm::Handle<'s, SlotName> {
        scope.create_handle(name.tagged()).expect("name is strong")
    }

    /// Root a value in the scope.
    fn root_value<'s>(scope: &'s HandleScope<'_>, value: Value) -> vm::Handle<'s, Value> {
        scope
            .create_handle(Tagged::from_value(value))
            .expect("value is strong")
    }

    #[test]
    fn transition_target_appends_descriptor_and_records_edge() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let flags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let parent = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let name = root_name(&scope, smi_name(1));

        let child = Map::transition_target(&mut heap, &scope, parent, name, flags);
        let child = scope.create_handle(child).expect("child map is strong");

        heap.no_gc(|nogc| {
            let child_ref = child.heap_ref(nogc);
            assert_eq!(child_ref.descriptor_count(), 1);
            assert_eq!(child_ref.value_slot_count(), 1);
            let d = child_ref.descriptor(0);
            assert_eq!(d.name(), smi_name(1));
            assert_eq!(d.flags(), flags);
            assert_eq!(d.offset(), 0);

            // the parent recorded the edge and finds it again
            let found = parent
                .heap_ref(nogc)
                .find_transition(nogc, smi_name(1), flags)
                .expect("recorded transition");
            assert_eq!(found.into_ptr().as_ptr(), child.get().as_ptr());
        });
    }

    #[test]
    fn transition_target_reuses_existing_child() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let flags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let parent = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let name = root_name(&scope, smi_name(1));

        let a = Map::transition_target(&mut heap, &scope, parent, name, flags);
        let b = Map::transition_target(&mut heap, &scope, parent, name, flags);
        assert_eq!(a.erase(), b.erase());

        heap.no_gc(|nogc| {
            let pairs = parent
                .heap_ref(nogc)
                .transitions
                .heap_ref(nogc)
                .expect("transition array");
            assert_eq!(pairs.len(), 2, "exactly one edge recorded");
        });
    }

    #[test]
    fn transition_target_grows_pairs_for_siblings_and_chains() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let flags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let parent = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let name1 = root_name(&scope, smi_name(1));
        let name2 = root_name(&scope, smi_name(2));

        // two different properties off the same parent: sibling edges
        let a = scope
            .create_handle(Map::transition_target(
                &mut heap, &scope, parent, name1, flags,
            ))
            .expect("strong");
        let b = scope
            .create_handle(Map::transition_target(
                &mut heap, &scope, parent, name2, flags,
            ))
            .expect("strong");
        assert_ne!(a.value(), b.value());

        // and a chain: transition from a child map
        let c = scope
            .create_handle(Map::transition_target(&mut heap, &scope, a, name2, flags))
            .expect("strong");

        heap.no_gc(|nogc| {
            let pairs = parent
                .heap_ref(nogc)
                .transitions
                .heap_ref(nogc)
                .expect("transition array");
            assert_eq!(pairs.len(), 4, "two edges recorded on the parent");

            // a: {1 -> slot 0}; c: {1 -> slot 0, 2 -> slot 1}
            let a = a.heap_ref(nogc);
            assert_eq!(a.descriptor_count(), 1);
            let c = c.heap_ref(nogc);
            assert_eq!(c.descriptor_count(), 2);
            assert_eq!(c.value_slot_count(), 2);
            assert_eq!(c.descriptor(0).name(), smi_name(1));
            assert_eq!(c.descriptor(1).name(), smi_name(2));
            assert_eq!(c.descriptor(1).offset(), 1);
        });
    }

    #[test]
    fn store_new_data_property_grows_object_and_writes() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let flags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(
            &mut heap,
            &_roots,
            kind,
            1,
            &[(1, flags, Smi::new(0).encode())],
        );
        let obj = root_object(
            &scope,
            alloc_object(&mut heap, map, &[Smi::new(7).encode()]),
        );

        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            root_name(&scope, smi_name(2)),
            PropertyDescriptor::data(root_value(&scope, Smi::new(9).encode()).value()),
        )
        .unwrap();

        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            let map = obj.header.map.heap_ref(nogc);
            assert_eq!(map.descriptor_count(), 2);
            // old slot intact, new slot written
            expect_data(obj.as_ref().lookup(nogc, smi_name(1)), 7);
            expect_data(obj.as_ref().lookup(nogc, smi_name(2)), 9);
            // appended descriptor carries the CreateDataProperty default attributes
            let d = map.descriptor(1);
            assert_eq!(d.name(), smi_name(2));
            assert_eq!(d.offset(), 1);
            assert!(d.flags().is_writable());
            assert!(d.flags().is_enumerable());
            assert!(d.flags().is_configurable());
        });
    }

    #[test]
    fn store_new_data_property_converges_maps() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, kind, 0, &[]);
        let a = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let b = root_object(&scope, alloc_object(&mut heap, map, &[]));

        Object::define_own_property(
            &mut heap,
            &scope,
            a,
            root_name(&scope, smi_name(1)),
            PropertyDescriptor::data(root_value(&scope, Smi::new(1).encode()).value()),
        )
        .unwrap();
        Object::define_own_property(
            &mut heap,
            &scope,
            b,
            root_name(&scope, smi_name(1)),
            PropertyDescriptor::data(root_value(&scope, Smi::new(2).encode()).value()),
        )
        .unwrap();

        heap.no_gc(|nogc| {
            let map_a = a.heap_ref(nogc).header.map.inner();
            let map_b = b.heap_ref(nogc).header.map.inner();
            assert_eq!(map_a, map_b, "same base shape converges to the same map");
            expect_data(a.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)), 1);
            expect_data(b.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)), 2);
        });
    }

    #[test]
    fn define_own_property_returns_false_on_non_extensible() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        // plain OBJECT: not extendable
        let map = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let obj = root_object(&scope, alloc_object(&mut heap, map, &[]));

        let result = Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            root_name(&scope, smi_name(1)),
            PropertyDescriptor::data(root_value(&scope, Smi::new(1).encode()).value()),
        );
        // spec: [[DefineOwnProperty]] reports false; the caller decides
        // whether that is a TypeError
        assert_eq!(result, Ok(false));

        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            // object untouched: same map, no slots
            assert_eq!(obj.header.map.inner(), map.value());
            assert_eq!(obj.slots.heap_ref(nogc).len(), 0);
        });
    }

    #[test]
    fn empty_objects_share_the_well_known_empty_fixed_array() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let a = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let b = root_object(&scope, alloc_object(&mut heap, map, &[]));

        heap.no_gc(|nogc| {
            let slots_a = a.heap_ref(nogc).slots.inner();
            let slots_b = b.heap_ref(nogc).slots.inner();
            assert_eq!(slots_a, slots_b);
            assert_eq!(slots_a, nogc.known().empty_fixed_array.value());
            assert_eq!(a.heap_ref(nogc).slots.heap_ref(nogc).len(), 0);
            assert_eq!(
                a.heap_ref(nogc).elements.inner(),
                nogc.known().empty_fixed_array.value()
            );
        });

        // a real data slot swaps in a fresh array
        let map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(
                1,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );
        let c = root_object(
            &scope,
            alloc_object(&mut heap, map, &[Smi::new(7).encode()]),
        );
        heap.no_gc(|nogc| {
            assert_ne!(
                c.heap_ref(nogc).slots.inner(),
                nogc.known().empty_fixed_array.value()
            );
            assert_eq!(c.heap_ref(nogc).slots.heap_ref(nogc).len(), 1);
        });
    }

    #[test]
    fn lookup_resolves_own_values_and_parent_chain() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        // parent: smi(3) = value slot 0 (99)
        let parent_map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(
                3,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );
        let parent = alloc_object(&mut heap, parent_map, &[Smi::new(99).encode()]);

        // child: smi(1) = value slot 0, smi(2) = value slot 1,
        // prototype = FixedArray([parent])
        let parents = heap
            .allocate::<FixedArray>(&[parent.encode_strong()])
            .into_ptr();
        let child_map = alloc_map_proto(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            2,
            &[
                (
                    1,
                    SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                    Smi::new(0).encode(),
                ),
                (
                    2,
                    SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                    Smi::new(1).encode(),
                ),
            ],
            parents.encode_strong(),
        );
        let child = alloc_object(
            &mut heap,
            child_map,
            &[Smi::new(7).encode(), Smi::new(42).encode()],
        );
        let child = unsafe { child.as_ref() };

        heap.no_gc(|nogc| {
            expect_data(child.lookup(nogc, smi_name(1)), 7); // own slot 0
            expect_data(child.lookup(nogc, smi_name(2)), 42); // own slot 1
            expect_data(child.lookup(nogc, smi_name(3)), 99); // inherited via parent
            assert!(matches!(child.lookup(nogc, smi_name(4)), Lookup::NotFound));
        });
    }

    #[test]
    fn slot_lookup_dispatches_smi_and_object() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        let map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(
                1,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );
        let obj = alloc_object(&mut heap, map, &[Smi::new(7).encode()]);

        heap.no_gc(|nogc| {
            // smi receiver: looks up in the (descriptor-less) smi map
            let smi = Smi::new(42).encode();
            assert!(matches!(smi.lookup(nogc, smi_name(1)), Lookup::NotFound));
            // object receiver: same result as the typed entry point
            let obj_value = obj.encode_strong();
            expect_data(obj_value.lookup(nogc, smi_name(1)), 7);
        });
    }

    #[test]
    fn lookup_multiple_parents_follow_priority_order() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        // parent A: smi(3) = value slot 0 (10); parent B: smi(3) = slot 0 (20)
        let map_a = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(
                3,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );
        let parent_a = alloc_object(&mut heap, map_a, &[Smi::new(10).encode()]);
        let map_b = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(
                3,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );
        let parent_b = alloc_object(&mut heap, map_b, &[Smi::new(20).encode()]);

        // child with prototype = [a, b]: the first parent wins
        let parents_ab = heap
            .allocate::<FixedArray>(&[parent_a.encode_strong(), parent_b.encode_strong()])
            .into_ptr();
        let child_ab_map = alloc_map_proto(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            0,
            &[],
            parents_ab.encode_strong(),
        );
        let child_ab = alloc_object(&mut heap, child_ab_map, &[]);
        let child_ab = unsafe { child_ab.as_ref() };

        // child with prototype = [b] alone reaches the second parent
        let parents_b = heap
            .allocate::<FixedArray>(&[parent_b.encode_strong()])
            .into_ptr();
        let child_b_map = alloc_map_proto(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            0,
            &[],
            parents_b.encode_strong(),
        );
        let child_b = alloc_object(&mut heap, child_b_map, &[]);
        let child_b = unsafe { child_b.as_ref() };

        heap.no_gc(|nogc| {
            expect_data(child_ab.lookup(nogc, smi_name(3)), 10); // first parent wins
            expect_data(child_b.lookup(nogc, smi_name(3)), 20);
        });
    }

    #[test]
    fn lookup_returns_accessor_pair() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        let pair_ptr = heap
            .allocate::<AccessorPair>((Smi::new(111).encode(), Smi::new(222).encode()))
            .into_ptr();

        let map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            0,
            &[(5, SlotFlags::ACCESSOR, pair_ptr.encode_strong())],
        );
        let obj = alloc_object(&mut heap, map, &[]);
        let obj = unsafe { obj.as_ref() };

        heap.no_gc(|nogc| match obj.lookup(nogc, smi_name(5)) {
            Lookup::Accessor { pair, .. } => {
                assert_eq!(Smi::decode(pair.get.get().erase()).unwrap().value(), 111);
                assert_eq!(Smi::decode(pair.set.get().erase()).unwrap().value(), 222);
            }
            _ => panic!("expected accessor lookup result"),
        });
    }

    #[test]
    fn store_lookup_on_accessor_with_setter_calls_setter() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        let pair_ptr = heap
            .allocate::<AccessorPair>((Smi::new(111).encode(), Smi::new(222).encode()))
            .into_ptr();
        let map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            0,
            &[(5, SlotFlags::ACCESSOR, pair_ptr.encode_strong())],
        );
        let obj = alloc_object(&mut heap, map, &[]);

        heap.no_gc(|nogc| {
            let outcome = obj
                .encode_strong()
                .store_lookup(
                    nogc,
                    smi_name(5),
                    Smi::new(1).encode(),
                    StoreSemantics::WriteThrough,
                )
                .unwrap();
            match outcome {
                StoreOutcome::CallSetter { setter } => {
                    assert_eq!(Smi::decode(setter).unwrap().value(), 222)
                }
                _ => panic!("expected CallSetter outcome"),
            }
        });
    }

    #[test]
    fn store_lookup_on_accessor_without_setter_is_ignored() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        let undefined = heap.known().undefined.value();
        let pair_ptr = heap
            .allocate::<AccessorPair>((undefined, undefined))
            .into_ptr();
        let map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            0,
            &[(5, SlotFlags::ACCESSOR, pair_ptr.encode_strong())],
        );
        let obj = alloc_object(&mut heap, map, &[]);

        heap.no_gc(|nogc| {
            let outcome = obj.encode_strong().store_lookup(
                nogc,
                smi_name(5),
                Smi::new(1).encode(),
                StoreSemantics::WriteThrough,
            );
            assert_eq!(outcome, Ok(StoreOutcome::Done));
        });
    }

    #[test]
    fn store_new_accessor_property_adds_descriptor_without_slot() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        let map = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            1,
            &[(
                1,
                SlotFlags::VALUE.union(SlotFlags::WRITABLE),
                Smi::new(0).encode(),
            )],
        );

        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let obj = heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[Smi::new(7).encode()],
                    elements: heap.known().void.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);

        let undefined = heap.known().undefined.value();
        let name = scope.create_handle(smi_name(5).tagged()).unwrap();
        let get = scope
            .create_handle(Tagged::from_value(Smi::new(111).encode()))
            .unwrap();
        let set = scope.create_handle(Tagged::from_value(undefined)).unwrap();
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: get.value(),
                set: set.value(),
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();

        heap.no_gc(|nogc| {
            let obj_ref = obj.heap_ref(nogc);
            let map = obj_ref.header.map.heap_ref(nogc);
            // the accessor takes no value slot; the existing slot stays put
            assert_eq!(map.value_slot_count(), 1);
            assert_eq!(map.descriptor_count(), 2);
            match obj.value().lookup(nogc, smi_name(5)) {
                Lookup::Accessor { pair, .. } => {
                    assert_eq!(Smi::decode(pair.get.get().erase()).unwrap().value(), 111);
                    assert_eq!(pair.set.get().erase(), undefined);
                }
                _ => panic!("expected accessor lookup result"),
            }
            expect_data(obj.value().lookup(nogc, smi_name(1)), 7);
        });
    }

    #[test]
    fn define_accessor_on_non_extensible_returns_false() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);

        let map = alloc_map(&mut heap, &_roots, MapKind::OBJECT, 0, &[]);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let obj = heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: heap.known().void.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);

        let undefined = heap.known().undefined.value();
        let name = scope.create_handle(smi_name(5).tagged()).unwrap();
        let get = scope.create_handle(Tagged::from_value(undefined)).unwrap();
        let set = scope.create_handle(Tagged::from_value(undefined)).unwrap();
        let result = Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: get.value(),
                set: set.value(),
                enumerable: true,
                configurable: true,
            },
        );
        assert_eq!(result, Ok(false));
    }

    #[test]
    fn define_own_property_redefines_existing_data_in_place() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, kind, 0, &[]);
        let obj = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let name = root_name(&scope, smi_name(1));

        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(1).encode()),
        )
        .unwrap();
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(2).encode()),
        )
        .unwrap();

        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            let map = obj.header.map.heap_ref(nogc);
            // the literal duplicate-key shape: one descriptor, updated value
            assert_eq!(map.descriptor_count(), 1, "redefinition must not duplicate");
            assert_eq!(map.value_slot_count(), 1);
            expect_data(obj.as_ref().lookup(nogc, smi_name(1)), 2);
        });
    }

    #[test]
    fn define_own_property_respects_non_configurable_guards() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, kind, 0, &[]);
        let obj = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let name = root_name(&scope, smi_name(1));

        let define = |heap: &mut Heap,
                      writable: bool,
                      enumerable: bool,
                      configurable: bool,
                      value: i64|
         -> Result<bool, VmError> {
            Object::define_own_property(
                heap,
                &scope,
                obj,
                name,
                PropertyDescriptor::Data {
                    value: Smi::new(value).encode(),
                    writable,
                    enumerable,
                    configurable,
                },
            )
        };

        assert_eq!(define(&mut heap, true, true, false, 1).unwrap(), true);
        // writable stays true: value changes freely, attributes are locked
        assert_eq!(define(&mut heap, true, true, false, 2).unwrap(), true);
        // writable true → false is allowed on a non-configurable property
        assert_eq!(define(&mut heap, false, true, false, 3).unwrap(), true);
        // writable false → true is not
        assert_eq!(define(&mut heap, true, true, false, 4), Ok(false));
        // non-writable value changes must be SameValue
        assert_eq!(define(&mut heap, false, true, false, 5), Ok(false));
        // attribute changes are locked
        assert_eq!(define(&mut heap, false, false, false, 3), Ok(false));
        assert_eq!(define(&mut heap, false, true, true, 3), Ok(false));
        // data → accessor conversion is locked too
        let undefined = heap.known().undefined.value();
        let result = Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: undefined,
                set: undefined,
                enumerable: true,
                configurable: false,
            },
        );
        assert_eq!(result, Ok(false));

        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            expect_data(obj.as_ref().lookup(nogc, smi_name(1)), 3);
        });
    }

    #[test]
    fn define_own_property_converts_between_data_and_accessor() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, kind, 0, &[]);
        let obj = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let name = root_name(&scope, smi_name(1));
        let undefined = heap.known().undefined.value();

        // data → accessor (configurable: allowed)
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(1).encode()),
        )
        .unwrap();
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: Smi::new(11).encode(),
                set: undefined,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            match obj.as_ref().lookup(nogc, smi_name(1)) {
                Lookup::Accessor { pair, .. } => {
                    assert_eq!(Smi::decode(pair.get.inner()).unwrap().value(), 11);
                }
                _ => panic!("expected accessor"),
            }
        });

        // accessor → data appends a fresh slot and restores data semantics
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(2).encode()),
        )
        .unwrap();
        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            let map = obj.header.map.heap_ref(nogc);
            assert_eq!(map.descriptor_count(), 1);
            expect_data(obj.as_ref().lookup(nogc, smi_name(1)), 2);
        });
    }

    #[test]
    fn same_value_matches_spec() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);

        let nan1 = heap.allocate_handle::<Float>(f64::NAN, &scope).value();
        let nan2 = heap.allocate_handle::<Float>(f64::NAN, &scope).value();
        let neg_zero = heap.allocate_handle::<Float>(-0.0, &scope).value();
        let pos_zero = heap.allocate_handle::<Float>(0.0, &scope).value();
        let one_float = heap.allocate_handle::<Float>(1.0, &scope).value();
        let mut mk_string = |s: &str| {
            let backing = heap
                .allocate::<FixedByteArray>(s.as_bytes())
                .into_handle(&scope);
            heap.allocate::<InternedString>((backing, string_content_hash(s.as_bytes())))
                .into_handle(&scope)
                .value()
        };
        let ab1 = mk_string("ab");
        let ab2 = mk_string("ab");
        let ac = mk_string("ac");

        heap.no_gc(|nogc| {
            use vm::Compare;
            // NaN equals NaN
            assert!(Compare::same_value(nogc, nan1, nan2));
            assert!(Compare::same_value(nogc, nan1, nan1));
            // +/-0 are distinct
            assert!(!Compare::same_value(nogc, neg_zero, pos_zero));
            assert!(Compare::same_value(nogc, neg_zero, neg_zero));
            assert!(!Compare::same_value(nogc, Smi::new(0).encode(), neg_zero));
            // number equality across representations
            assert!(Compare::same_value(nogc, Smi::new(1).encode(), one_float));
            assert!(Compare::same_value(
                nogc,
                Smi::new(1).encode(),
                Smi::new(1).encode()
            ));
            // strings by content
            assert!(Compare::same_value(nogc, ab1, ab2));
            assert!(!Compare::same_value(nogc, ab1, ac));
            // objects by identity
            let map = nogc.known().object_initial_map;
            assert!(Compare::same_value(
                nogc,
                nogc.known().undefined.value(),
                nogc.known().undefined.value()
            ));
            let _ = map;
        });
    }

    #[test]
    fn concurrent_transition_target_converges() {
        let global = DummyHeap::new(DummyHeapConfig { heap_size: 1 << 20 })
            .unwrap()
            .into_global();
        let _roots = install_well_known(&global);
        let mut local = global.new_local();
        let flags = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        let parent = alloc_map(&mut local, &_roots, MapKind::OBJECT, 0, &[]);
        let parent_ptr = parent.get();

        // N threads race to add the same property transition to one map;
        // all must converge to the same child map
        let global = &global;
        std::thread::scope(|s| {
            let mut threads = Vec::new();
            for _ in 0..8 {
                threads.push(s.spawn(move || {
                    let mut local = global.new_local();
                    let data = HandleData::new(local.known().void.value());
                    let scope = scope(&data);
                    let parent = scope
                        .create_handle(Tagged::from_ptr(parent_ptr))
                        .expect("parent is strong");
                    let name = root_name(&scope, smi_name(1));
                    Map::transition_target(&mut local, &scope, parent, name, flags).erase()
                }));
            }
            let results: Vec<Value> = threads.into_iter().map(|t| t.join().unwrap()).collect();
            assert!(
                results.windows(2).all(|w| w[0] == w[1]),
                "all threads must converge to the same child map"
            );
        });

        local.no_gc(|nogc| {
            let pairs = parent
                .heap_ref(nogc)
                .transitions
                .heap_ref(nogc)
                .expect("transition array");
            assert_eq!(pairs.len(), 2, "exactly one edge recorded");
        });
    }

    #[test]
    fn token_bulk_allocates_and_tracks_remaining() {
        let (global, mut heap, _roots) = local_with_maps(1 << 16);
        let la = FixedArray::layout_for(2);
        let lb = FixedByteArray::layout_for(8);
        let total = Layout::from_size_align(la.size() + lb.size(), 16).unwrap();
        let before = global.stats().used;
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
        assert_eq!(global.stats().used, start + la.size() + lb.size());
    }

    #[test]
    fn token_handles_outlive_the_token() {
        let (global, mut heap, _roots) = local_with_maps(1 << 16);
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
        assert!(global.contains(ha.value().raw_addr()));
        assert!(global.contains(hb.value().raw_addr()));
    }

    #[test]
    fn token_fresh_allocations_coexist_and_promote() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
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
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let lb = FixedByteArray::layout_for(8);
        let total = Layout::from_size_align(2 * lb.size(), 16).unwrap();

        let tok = heap.allocate_token(total);
        tok.enter_no_gc(|nogc| {
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
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let lb = FixedByteArray::layout_for(4);
        let total = Layout::from_size_align(lb.size(), 16).unwrap();
        heap.allocate_token_enter_nogc(total, |tok, nogc| {
            let a = tok.allocate_ref::<FixedByteArray>(&[0u8; 4], nogc);
            a.set(0, 1);
            assert_eq!(a.get(0), 1);
        });
    }

    #[test]
    fn allocate_enter_no_gc_gives_ref_instantly() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        heap.allocate_enter_nogc::<FixedByteArray, _>(&[1u8, 2, 3, 4], |bytes, _nogc| {
            assert_eq!(bytes.as_slice(), &[1, 2, 3, 4]);
        });
    }

    #[test]
    #[should_panic(expected = "allocation token dropped")]
    fn token_drop_requires_full_use() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let total = Layout::from_size_align(4096, 16).unwrap();
        let tok = heap.allocate_token(total);
        let _ = tok.allocate::<FixedByteArray>(&[0u8; 4]);
        // dropped with most of the reservation unused
    }

    #[test]
    #[should_panic(expected = "allocation token exhausted")]
    fn token_allocate_checks_capacity() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let total = Layout::from_size_align(64, 16).unwrap();
        let tok = heap.allocate_token(total);
        let _ = tok.allocate::<FixedByteArray>(&[0u8; 4096]);
    }

    #[test]
    fn define_same_attributes_reuses_the_map() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, kind, 0, &[]);
        let obj = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let name = root_name(&scope, smi_name(1));

        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(1).encode()),
        )
        .unwrap();
        let after_first = heap.no_gc(|nogc| obj.heap_ref(nogc).header.map.inner());
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(2).encode()),
        )
        .unwrap();
        let after_second = heap.no_gc(|nogc| obj.heap_ref(nogc).header.map.inner());
        // identical attributes: [[DefineOwnProperty]] only writes the slot;
        // the map (and with it the property layout) must not change
        assert_eq!(after_first, after_second);

        heap.no_gc(|nogc| {
            let obj = obj.heap_ref(nogc);
            expect_data(obj.as_ref().lookup(nogc, smi_name(1)), 2);
        });
    }

    #[test]
    fn shadow_store_defines_own_property_above_multiple_parents() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let writable = SlotFlags::VALUE.union(SlotFlags::WRITABLE);

        // parent A (priority): writable smi(1) = 10; parent B: writable = 20
        let map_a = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(1, writable, Smi::new(0).encode())],
        );
        let parent_a = root_object(
            &scope,
            alloc_object(&mut heap, map_a, &[Smi::new(10).encode()]),
        );
        let map_b = alloc_map(
            &mut heap,
            &_roots,
            MapKind::OBJECT,
            1,
            &[(1, writable, Smi::new(0).encode())],
        );
        let parent_b = root_object(
            &scope,
            alloc_object(&mut heap, map_b, &[Smi::new(20).encode()]),
        );

        let parents = heap
            .allocate::<FixedArray>(&[parent_a.as_tagged().erase(), parent_b.as_tagged().erase()])
            .into_handle(&scope);
        let child_map = alloc_map_proto(
            &mut heap,
            &_roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            0,
            &[],
            parents.as_tagged().erase(),
        );
        let child = root_object(&scope, alloc_object(&mut heap, child_map, &[]));
        let name = root_name(&scope, smi_name(1));

        // [[Set]] with Shadow semantics: the lookup hits parent A's writable
        // slot first, so the receiver shadows it with an own property
        let outcome = heap
            .no_gc(|nogc| {
                child.value().store_lookup(
                    nogc,
                    name.into(),
                    Smi::new(30).encode(),
                    StoreSemantics::Shadow,
                )
            })
            .unwrap();
        match outcome {
            StoreOutcome::Transition { receiver, name } => {
                assert_eq!(receiver, child.value());
                Object::add_own_property(
                    &mut heap,
                    &scope,
                    child,
                    root_name(&scope, name),
                    PropertyDescriptor::data(Smi::new(30).encode()),
                )
                .unwrap();
            }
            other => panic!("expected transition, got {other:?}"),
        }

        heap.no_gc(|nogc| {
            // the own slot wins over both parents; both parents untouched
            expect_data(child.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)), 30);
            expect_data(
                parent_a.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)),
                10,
            );
            expect_data(
                parent_b.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)),
                20,
            );
        });
    }
    #[test]
    fn redefine_attributes_share_cached_transition() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        // both objects start from the same map with one writable property
        let writable = SlotFlags::VALUE.union(SlotFlags::WRITABLE);
        let map = alloc_map(
            &mut heap,
            &_roots,
            kind,
            1,
            &[(1, writable, Smi::new(0).encode())],
        );
        let a = root_object(
            &scope,
            alloc_object(&mut heap, map, &[Smi::new(10).encode()]),
        );
        let b = root_object(
            &scope,
            alloc_object(&mut heap, map, &[Smi::new(20).encode()]),
        );
        let name = root_name(&scope, smi_name(1));

        // the same shape change on two objects with the same map converges
        // on ONE target map (cached transition, keyed by name + flags)
        let frozen = PropertyDescriptor::Data {
            value: Smi::new(30).encode(),
            writable: false,
            enumerable: false,
            configurable: false,
        };
        Object::define_own_property(&mut heap, &scope, a, name, frozen).unwrap();
        Object::define_own_property(
            &mut heap,
            &scope,
            b,
            name,
            PropertyDescriptor::Data {
                value: Smi::new(40).encode(),
                writable: false,
                enumerable: false,
                configurable: false,
            },
        )
        .unwrap();

        heap.no_gc(|nogc| {
            let map_a = a.heap_ref(nogc).header.map.inner();
            let map_b = b.heap_ref(nogc).header.map.inner();
            assert_eq!(map_a, map_b, "identical redefines must share the map");
            let map_a = a.heap_ref(nogc).header.map.heap_ref(nogc);
            let d = map_a.descriptor(0);
            // writable/enumerable/configurable all false after the redefine
            assert!(!d.flags().is_writable());
            assert!(!d.flags().is_enumerable());
            assert!(!d.flags().is_configurable());
            // values stay per-object
            expect_data(a.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)), 30);
            expect_data(b.heap_ref(nogc).as_ref().lookup(nogc, smi_name(1)), 40);
        });
    }

    #[test]
    fn accessor_to_data_redefine_appends_slot() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        let kind = MapKind::OBJECT.union(MapKind::EXTENDABLE);
        let data = HandleData::new(heap.known().void.value());
        let scope = scope(&data);
        let map = alloc_map(&mut heap, &_roots, kind, 0, &[]);
        let obj = root_object(&scope, alloc_object(&mut heap, map, &[]));
        let name = root_name(&scope, smi_name(1));

        // accessor first: the descriptor row embeds the per-object pair
        let undefined = heap.known().undefined.value();
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::Accessor {
                get: Smi::new(111).encode(),
                set: undefined,
                enumerable: true,
                configurable: true,
            },
        )
        .unwrap();
        // converting back to data appends a fresh slot for the value
        Object::define_own_property(
            &mut heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(Smi::new(7).encode()),
        )
        .unwrap();

        heap.no_gc(|nogc| {
            let obj_ref = obj.heap_ref(nogc);
            let map = obj_ref.header.map.heap_ref(nogc);
            assert_eq!(
                map.value_slot_count(),
                1,
                "accessor -> data grows the slots"
            );
            expect_data(obj_ref.as_ref().lookup(nogc, smi_name(1)), 7);
        });
    }
    #[test]
    fn bootstrap_wires_function_prototype() {
        let (_global, mut heap, _roots) = local_with_maps(1 << 16);
        heap.no_gc(|nogc| {
            let known = nogc.known();
            let fp = known.function_prototype.heap_ref(nogc);
            let kind = fp.header.map.heap_ref(nogc).kind();
            // ES 19.2.3: callable, not a constructor
            assert!(kind.is_callable());
            assert!(!kind.is_constructor());
            // its [[Prototype]] is %Object.prototype%
            assert_eq!(
                fp.header.map.heap_ref(nogc).prototype.inner(),
                known.object_prototype.value()
            );
            // ordinary function objects' [[Prototype]] points at it
            // (ES 19.2.3.1)
            assert_eq!(
                known.function_map.heap_ref(nogc).prototype.inner(),
                known.function_prototype.value()
            );
            // it is a real callable with callable info
            assert!(fp.as_ref().callable_info(nogc).is_some());
        });
    }
}
