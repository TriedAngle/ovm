use core::alloc::Layout;
use core::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use heap_api::{
    AllocError, GcHost, HeapBackend, HeapStats, LocalHeap, RawCell, SharedHeap, Visitor, Word,
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
}

unsafe impl Send for DummyHeapState {}
unsafe impl Sync for DummyHeapState {}

impl DummyHeapState {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let mut offset = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = offset.next_multiple_of(layout.align());
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

/// Concrete per-thread heap; type-erased behind [`LocalHeap`].
pub struct DummyLocalHeap {
    shared: Arc<DummyHeapState>,
}

impl DummyLocalHeap {
    pub fn shared(&self) -> &DummyHeapState {
        &self.shared
    }
}

impl LocalHeap for DummyLocalHeap {
    fn allocate_raw(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.shared.allocate(layout)
    }

    fn write_barrier(&self, _host: Word, _slot: &RawCell, _value: Word) {}

    fn collection_requested(&self) -> bool {
        false
    }

    fn park_for_collection(&self) -> bool {
        false
    }

    fn take_cancel(&self) -> bool {
        false
    }

    /// The dummy heap has no safepoint barrier: shutdown only terminates
    /// the calling thread's execution itself.
    fn cancel_executions(&self, _protocol: &dyn Fn()) {}

    fn force_collect(&self) {}

    fn collect_minor(&self) {}

    fn gc_in_progress(&self) -> bool {
        false
    }
}

impl SharedHeap for DummyHeap {
    fn new_local(&self) -> Box<dyn LocalHeap> {
        Box::new(DummyLocalHeap {
            shared: Arc::clone(&self.inner),
        })
    }

    /// The dummy heap never collects: it accepts the host registration and
    /// drops it.
    fn set_host(&self, _host: GcHost) {}

    fn iterate_roots(&self, _roots: &mut dyn Visitor) {}

    fn should_collect(&self) -> bool {
        false
    }

    fn gc_in_progress(&self) -> bool {
        false
    }

    fn force_collect(&self) {}

    fn contains(&self, addr: Word) -> bool {
        self.inner.contains(addr)
    }

    fn is_young(&self, _value: Word) -> bool {
        false
    }

    fn stats(&self) -> HeapStats {
        HeapStats {
            used: self.inner.used(),
            capacity: self.inner.capacity(),
        }
    }
}

impl HeapBackend for DummyHeap {
    type Config = DummyHeapConfig;

    fn new(config: Self::Config) -> Result<Self, AllocError> {
        DummyHeap::new(config)
    }

    fn into_shared(self) -> std::sync::Arc<dyn SharedHeap> {
        std::sync::Arc::new(self)
    }
}
