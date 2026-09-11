use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{AllocError, GcHost};
use heap_utils::{Bitmap, MMapBuffer};

use crate::block::{need_for, ALIGN, FreeList};

pub const CHUNK_SIZE: usize = 256 * 1024;

const SWEPT: u8 = 0;
const PENDING: u8 = 1;
const SWEEPING: u8 = 2;

pub struct Chunk {
    index: usize,
    base: NonNull<u8>,
    size: usize,
    pub bitmap: Bitmap,
    free: Mutex<FreeList>,
    free_bytes: AtomicUsize,
    state: AtomicU8,
}

unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

impl Chunk {
    fn new(index: usize, base: NonNull<u8>, size: usize) -> Self {
        Self {
            bitmap: Bitmap::new(base.as_ptr() as usize, size, ALIGN),
            free: Mutex::new(FreeList::covering(base, size)),
            free_bytes: AtomicUsize::new(size),
            state: AtomicU8::new(SWEPT),
            index,
            base,
            size,
        }
    }

    pub fn base(&self) -> usize {
        self.base.as_ptr() as usize
    }

    pub fn base_ptr(&self) -> NonNull<u8> {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn state(&self) -> u8 {
        self.state.load(Ordering::Relaxed)
    }
}

pub struct ChunkedHeap {
    buffer: MMapBuffer,
    chunk_size: usize,
    slots: Vec<Option<Arc<Chunk>>>,
    pooled: Vec<usize>,
    carved: usize,
    pending_chunks: usize,
    sweep_live: usize,
    completed_live: Option<usize>,
    atomic_sweep_block: bool,
}

impl ChunkedHeap {
    pub fn new(reserve: usize) -> Result<Self, AllocError> {
        let buffer = MMapBuffer::new(reserve)?;
        let chunk_size = CHUNK_SIZE.min(buffer.size());
        let slots = buffer.size().div_ceil(chunk_size);
        Ok(Self {
            buffer,
            chunk_size,
            slots: (0..slots).map(|_| None).collect(),
            pooled: Vec::new(),
            carved: 0,
            pending_chunks: 0,
            sweep_live: 0,
            completed_live: None,
            atomic_sweep_block: false,
        })
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn reserve_size(&self) -> usize {
        self.buffer.size()
    }

    pub fn committed_bytes(&self) -> usize {
        self.slots.iter().flatten().map(|c| c.size()).sum()
    }

    pub fn live_bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|c| c.free.lock().unwrap().live_bytes())
            .sum()
    }

    pub fn free_bytes(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .map(|c| c.free_bytes.load(Ordering::Relaxed))
            .sum()
    }

    pub fn active_chunks(&self) -> usize {
        self.carved - self.pooled.len()
    }

    pub fn pending_chunks(&self) -> usize {
        self.pending_chunks
    }

    /// True for addresses inside carved regions, whether active or pooled.
    pub fn in_heap(&self, addr: usize) -> bool {
        let base = self.buffer.start().as_ptr() as usize;
        addr >= base && addr < base + self.carved * self.chunk_size
    }

    pub fn chunk_of(&self, addr: usize) -> &Chunk {
        let base = self.buffer.start().as_ptr() as usize;
        let index = (addr - base) / self.chunk_size;
        debug_assert!(index < self.slots.len(), "address outside reservation");
        self.slots[index]
            .as_ref()
            .expect("pointer into pooled or uncarved chunk")
    }

    /// First-fit
    pub fn allocate(
        &mut self,
        layout: Layout,
        host: Option<&GcHost>,
    ) -> (Option<NonNull<u8>>, Option<Arc<Chunk>>) {
        if let Some(ptr) = self.allocate_swept(layout) {
            return (Some(ptr), None);
        }
        if host.is_some() {
            if let Some(chunk) = self.claim_pending() {
                return (None, Some(chunk));
            }
        }
        match self.activate_chunk() {
            Some(chunk) => {
                let mut free = chunk.free.lock().unwrap();
                let ptr = free.allocate(layout);
                chunk.free_bytes.store(free.free_bytes(), Ordering::Relaxed);
                (ptr, None)
            }
            None => (None, None),
        }
    }

    fn allocate_swept(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let need = need_for(layout);
        for slot in &self.slots {
            let Some(chunk) = slot else { continue };
            if chunk.state() != SWEPT {
                continue;
            }
            if chunk.free_bytes.load(Ordering::Relaxed) < need {
                continue;
            }
            let mut free = chunk.free.lock().unwrap();
            if let Some(ptr) = free.allocate(layout) {
                chunk.free_bytes.store(free.free_bytes(), Ordering::Relaxed);
                return Some(ptr);
            }
        }
        None
    }

    pub fn claim_pending(&mut self) -> Option<Arc<Chunk>> {
        if self.atomic_sweep_block {
            return None;
        }
        self.claim_pending_unblocked()
    }

    fn claim_pending_unblocked(&mut self) -> Option<Arc<Chunk>> {
        for slot in &self.slots {
            let Some(chunk) = slot else { continue };
            if chunk.state() == PENDING {
                chunk.state.store(SWEEPING, Ordering::Relaxed);
                return Some(Arc::clone(chunk));
            }
        }
        None
    }

    pub fn set_sweep_block(&mut self, blocked: bool) {
        self.atomic_sweep_block = blocked;
    }

    pub fn any_sweeping(&self) -> bool {
        self.slots
            .iter()
            .flatten()
            .any(|chunk| chunk.state() == SWEEPING)
    }

    pub fn sweep_claimed(chunk: &Chunk, host: &GcHost) -> usize {
        let mut free = chunk.free.lock().unwrap();
        free.sweep(chunk.base_ptr(), chunk.size(), &chunk.bitmap, host);
        chunk.free_bytes.store(free.free_bytes(), Ordering::Relaxed);
        chunk.bitmap.clear_all();
        free.live_bytes()
    }

    pub fn publish_swept(&mut self, chunk: &Chunk, live: usize) {
        if live == 0 {
            let index = chunk.index;
            let removed = self.slots[index].take().unwrap();
            self.buffer.decommit(removed.base, removed.size);
            drop(removed);
            self.pooled.push(index);
        } else {
            chunk.state.store(SWEPT, Ordering::Relaxed);
        }
        self.sweep_live += live;
        self.pending_chunks -= 1;
        if self.pending_chunks == 0 {
            self.completed_live = Some(self.sweep_live);
        }
    }

    pub fn take_completed_live(&mut self) -> Option<usize> {
        self.completed_live.take()
    }

    fn chunk_region(&self, index: usize) -> (NonNull<u8>, usize) {
        let base =
            unsafe { self.buffer.start().as_ptr().add(index * self.chunk_size) };
        let size = self
            .chunk_size
            .min(self.reserve_size() - index * self.chunk_size);
        (unsafe { NonNull::new_unchecked(base) }, size)
    }

    fn activate_chunk(&mut self) -> Option<Arc<Chunk>> {
        let index = match self.pooled.pop() {
            Some(index) => index,
            None if self.carved < self.slots.len() => {
                let index = self.carved;
                self.carved += 1;
                index
            }
            None => return None,
        };
        let (base, size) = self.chunk_region(index);
        let chunk = Arc::new(Chunk::new(index, base, size));
        self.slots[index] = Some(Arc::clone(&chunk));
        Some(chunk)
    }

    pub fn flag_all_pending(&mut self) {
        let mut count = 0usize;
        for slot in &self.slots {
            if let Some(chunk) = slot {
                debug_assert_ne!(chunk.state(), SWEEPING, "cycle started mid-sweep");
                chunk.state.store(PENDING, Ordering::Relaxed);
                count += 1;
            }
        }
        self.pending_chunks = count;
        self.sweep_live = 0;
        if count == 0 {
            self.completed_live = Some(0);
        }
    }

    pub fn finish_pending(&mut self, host: &GcHost) -> bool {
        while let Some(chunk) = self.claim_pending_unblocked() {
            let live = Self::sweep_claimed(&chunk, host);
            self.publish_swept(&chunk, live);
        }
        !self.any_sweeping()
    }

    pub fn for_each_live(&self, mut f: impl FnMut(usize)) {
        for chunk in self.slots.iter().flatten() {
            for addr in chunk.bitmap.iter_set() {
                f(addr);
            }
        }
    }
}
