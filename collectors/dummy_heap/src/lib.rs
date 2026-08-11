use core::alloc::Layout;
use core::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use vm::{AllocError, GcSlot, LocalHeap, RootVisitor, Heap, Value, Word};

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
            }),
        })
    }

    fn new_local(&self) -> Self::Local {
        DummyLocalHeap {
            shared: Arc::clone(&self.inner),
        }
    }

    fn iterate_roots(&self, _roots: &mut dyn RootVisitor) {
        // The dummy heap owns no roots.
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

    fn write_barrier(&self, _host: Value, _slot: &GcSlot, _value: Value) {}

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

    use vm::{Array, ByteArray, HandleData, HandleScope, HeapObject, Smi};

    fn scope(data: &HandleData) -> HandleScope<'_> {
        unsafe { HandleScope::from_raw(NonNull::from(data)) }
    }

    #[test]
    fn token_bulk_allocates_and_tracks_remaining() {
        let mut heap = local(1 << 16);
        let la = Array::layout_for(2);
        let lb = ByteArray::layout_for(8);
        let total = Layout::from_size_align(la.size() + lb.size(), 16).unwrap();
        let before = heap.shared().used();
        {
            let tok = heap.allocate_token(total);
            assert_eq!(tok.remaining(), la.size() + lb.size());
            let a = tok.allocate::<Array>(la);
            let b = tok.allocate::<ByteArray>(lb);
            assert_ne!(a.as_ptr() as *mut u8, b.as_ptr() as *mut u8);
            assert_eq!(tok.remaining(), 0);
        }
        assert_eq!(heap.shared().used(), before + la.size() + lb.size());
    }

    #[test]
    fn token_handles_outlive_the_token() {
        let mut heap = local(1 << 16);
        let data = HandleData::new();
        let scope = scope(&data);
        let la = Array::layout_for(2);
        let lb = ByteArray::layout_for(8);
        let total = Layout::from_size_align(la.size() + lb.size(), 16).unwrap();

        let (ha, hb) = {
            let tok = heap.allocate_token(total);
            let ha = tok.allocate::<Array>(la).into_handle(&scope);
            let hb = tok.allocate::<ByteArray>(lb).into_handle(&scope);
            (ha, hb)
        }; // token dropped here (full use verified)

        // the handles stay rooted and point into the heap
        assert_ne!(ha.value().to_bits(), hb.value().to_bits());
        assert!(heap.shared().contains(ha.value().raw_addr()));
        assert!(heap.shared().contains(hb.value().raw_addr()));
    }

    #[test]
    fn token_fresh_allocations_coexist_and_promote() {
        let mut heap = local(1 << 16);
        let data = HandleData::new();
        let scope = scope(&data);
        let la = Array::layout_for(1);
        let total = Layout::from_size_align(2 * la.size(), 16).unwrap();

        let tok = heap.allocate_token(total);
        // multiple Fresh alive at once (shared borrows of the token)
        let a = tok.allocate::<Array>(la);
        let b = tok.allocate::<Array>(la);
        let ha = a.into_handle(&scope);
        let hb = b.into_handle(&scope);
        assert_ne!(ha.value().to_bits(), hb.value().to_bits());
    }

    #[test]
    fn token_enter_no_gc_allocates_refs() {
        let mut heap = local(1 << 16);
        let lb = ByteArray::layout_for(8);
        let total = Layout::from_size_align(2 * lb.size(), 16).unwrap();

        let tok = heap.allocate_token(total);
        tok.enter_no_gc(|nogc, heap| {
            let a = tok.allocate_ref::<ByteArray>(lb, nogc);
            let b = tok.allocate_ref::<ByteArray>(lb, nogc);
            a.size.set(heap, a.erase(), Smi::new_unchecked(8));
            b.size.set(heap, b.erase(), Smi::new_unchecked(8));
            b.set(0, 42);
            b.set(1, 7);
            assert_eq!(a.get(0), 0);
            assert_eq!(b.get(0), 42);
            assert_eq!(b.get(1), 7);
        });
    }

    #[test]
    fn allocate_token_enter_no_gc_combines_both() {
        let mut heap = local(1 << 16);
        let lb = ByteArray::layout_for(4);
        let total = Layout::from_size_align(lb.size(), 16).unwrap();
        heap.allocate_token_enter_nogc(total, |tok, nogc, heap| {
            let a = tok.allocate_ref::<ByteArray>(lb, nogc);
            a.size.set(heap, a.erase(), Smi::new_unchecked(4));
            a.set(0, 1);
            assert_eq!(a.get(0), 1);
        });
    }

    #[test]
    fn allocate_enter_no_gc_gives_ref_instantly() {
        let mut heap = local(1 << 16);
        heap.allocate_enter_nogc::<ByteArray, _>(ByteArray::layout_for(4), |bytes, _nogc, heap| {
            bytes.size.set(heap, bytes.erase(), Smi::new_unchecked(4));
            bytes.set(0, 1);
            bytes.set(1, 2);
            assert_eq!(bytes.as_slice(), &[1, 2, 0, 0]);
        });
    }

    #[test]
    #[should_panic(expected = "allocation token dropped")]
    fn token_drop_requires_full_use() {
        let mut heap = local(1 << 16);
        let total = Layout::from_size_align(4096, 16).unwrap();
        let tok = heap.allocate_token(total);
        tok.allocate::<ByteArray>(ByteArray::layout_for(4));
        // dropped with most of the reservation unused
    }

    #[test]
    #[should_panic(expected = "allocation token exhausted")]
    fn token_allocate_checks_capacity() {
        let mut heap = local(1 << 16);
        let total = Layout::from_size_align(64, 16).unwrap();
        let tok = heap.allocate_token(total);
        tok.allocate::<ByteArray>(ByteArray::layout_for(4096));
    }
}
