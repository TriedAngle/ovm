use core::alloc::Layout;
use core::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use vm::{AllocError, GcSlot, LocalHeap, RootVisitor, SharedHeap, Value, Word};

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

/// The shared heap: owns the arena and the coordination flags.
pub struct DummyHeap {
    start: NonNull<u8>,
    layout: Layout,
    offset: AtomicUsize,
}

unsafe impl Send for DummyHeap {}
unsafe impl Sync for DummyHeap {}

impl DummyHeap {
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
}

impl Drop for DummyHeap {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.start.as_ptr(), self.layout) };
    }
}

impl SharedHeap for DummyHeap {
    type Config = DummyHeapConfig;

    fn new(config: Self::Config) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(config.heap_size, 16)
            .expect("invalid heap size in DummyHeapConfig");
        let start = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .ok_or(AllocError::OutOfMemory(layout))?;
        Ok(Self {
            start,
            layout,
            offset: AtomicUsize::new(0),
        })
    }

    fn collect(&mut self, _roots: &mut dyn RootVisitor) {
    }

    fn should_collect(&self) -> bool {
        false
    }

    fn gc_in_progress(&self) -> bool {
        false
    }

    fn contains(&self, addr: Word) -> bool {
        let start = self.start.as_ptr() as Word;
        (start..start + self.layout.size() as Word).contains(&addr)
    }

    fn is_young(&self, _value: Value) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct DummyLocalHeap {
    shared: Arc<DummyHeap>,
}

impl DummyLocalHeap {
    pub fn new(shared: DummyHeap) -> Self {
        Self {
            shared: Arc::new(shared),
        }
    }

    pub fn shared(&self) -> &DummyHeap {
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
        let heap = DummyHeap::new(DummyHeapConfig { heap_size: size }).unwrap();
        DummyLocalHeap::new(heap)
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
}
