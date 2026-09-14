use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{AllocError, GcHost};
use heap_utils::MMapBuffer;

use crate::block::{ALIGN, FreeList, need_for};

pub const CHUNK_SIZE: usize = 256 * 1024;

const SWEPT: u8 = 0;
const PENDING: u8 = 1;
const SWEEPING: u8 = 2;

const FLAG_YOUNG: u64 = 1 << 0;

#[repr(C)]
pub struct ChunkHeader {
    /// bit 0: the chunk is in the young set
    flags: AtomicU64,
    /// Remember bits: 8 byte granule
    remembered: [AtomicU64; REMEMBERED_WORDS],
    /// Mark bits: one bit per 16-byte granule in the usable region.
    mark: [AtomicU64; MARK_WORDS],
    _pad: [u8; HEADER_SIZE - 8 - REMEMBERED_WORDS * 8 - MARK_WORDS * 8],
}

/// 512 words cover 512·64·8 = 256 KiB of 8-byte slots: a full chunk.
const REMEMBERED_WORDS: usize = 512;
/// 256 words cover 256·64·16 = 256 KiB of 16-byte granules: a full chunk.
const MARK_WORDS: usize = 256;
pub const HEADER_SIZE: usize = 8 * 1024;

const _: () = {
    assert!(
        size_of::<ChunkHeader>() == HEADER_SIZE,
        "header must be fixed-size"
    );
    assert!(
        HEADER_SIZE % ALIGN == 0,
        "usable region must start 16-aligned"
    );
    assert!(
        (CHUNK_SIZE - HEADER_SIZE) % ALIGN == 0,
        "usable region must end 16-aligned"
    );
    assert!(REMEMBERED_WORDS * 64 * 8 >= CHUNK_SIZE - HEADER_SIZE);
    assert!(MARK_WORDS * 64 * 16 >= CHUNK_SIZE - HEADER_SIZE);
};

impl ChunkHeader {
    pub fn at(heap_base: usize, addr: usize) -> &'static Self {
        let chunk_base = heap_base + (addr - heap_base) / CHUNK_SIZE * CHUNK_SIZE;
        unsafe { &*(chunk_base as *const Self) }
    }

    fn of_chunk(chunk_base: usize) -> &'static Self {
        unsafe { &*(chunk_base as *const Self) }
    }

    fn init(chunk_base: usize, young: bool) {
        let header = Self::of_chunk(chunk_base);
        for word in &header.remembered {
            word.store(0, Ordering::Release);
        }
        for word in &header.mark {
            word.store(0, Ordering::Release);
        }
        header
            .flags
            .store(if young { FLAG_YOUNG } else { 0 }, Ordering::Release);
    }

    pub fn object_base(&self) -> usize {
        self as *const Self as usize + HEADER_SIZE
    }

    pub fn young(&self) -> bool {
        self.flags.load(Ordering::Acquire) & FLAG_YOUNG != 0
    }

    fn remembered_index(&self, addr: usize) -> (usize, u32) {
        let bit = (addr - self.object_base()) >> 3;
        debug_assert!(
            addr >= self.object_base() && addr & 0b111 == 0,
            "unaligned remembered slot"
        );
        ((bit >> 6) as usize, (bit & 63) as u32)
    }

    pub fn remember(&self, slot_addr: usize) {
        let (word, bit) = self.remembered_index(slot_addr);
        self.remembered[word].fetch_or(1 << bit, Ordering::AcqRel);
    }

    pub fn remembered_iter(&self, usable: usize) -> impl Iterator<Item = usize> {
        let base = self.object_base();
        let bits = usable >> 3;
        HeaderBits {
            words: &self.remembered,
            base,
            shift: 3,
            word: 0,
            bits: 0,
            limit: bits,
        }
    }

    pub fn clear_remembered(&self) {
        for word in &self.remembered {
            word.store(0, Ordering::Release);
        }
    }

    fn mark_index(&self, addr: usize) -> (usize, u32) {
        debug_assert!(
            addr >= self.object_base() && addr & 0b1111 == 0,
            "unaligned mark address"
        );
        let bit = (addr - self.object_base()) >> 4;
        ((bit >> 6) as usize, (bit & 63) as u32)
    }

    pub fn mark_set(&self, addr: usize) -> bool {
        let (word, bit) = self.mark_index(addr);
        self.mark[word].fetch_or(1 << bit, Ordering::AcqRel) & (1 << bit) == 0
    }

    pub fn mark_is_set(&self, addr: usize) -> bool {
        let (word, bit) = self.mark_index(addr);
        self.mark[word].load(Ordering::Acquire) & (1 << bit) != 0
    }

    pub fn mark_iter(&self, usable: usize) -> impl Iterator<Item = usize> {
        let base = self.object_base();
        let bits = usable >> 4;
        HeaderBits {
            words: &self.mark,
            base,
            shift: 4,
            word: 0,
            bits: 0,
            limit: bits,
        }
    }

    pub fn clear_mark(&self) {
        for word in &self.mark {
            word.store(0, Ordering::Release);
        }
    }
}

struct HeaderBits<'a> {
    words: &'a [AtomicU64],
    base: usize,
    shift: u32,
    word: usize,
    bits: u64,
    limit: usize,
}

impl Iterator for HeaderBits<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.bits != 0 {
                let bit = self.bits.trailing_zeros() as usize;
                self.bits &= self.bits - 1;
                let index = ((self.word - 1) << 6) + bit;
                if index < self.limit {
                    return Some(self.base + (index << self.shift));
                }
                continue;
            }
            if self.word >= self.words.len() {
                return None;
            }
            self.bits = self.words[self.word].load(Ordering::Acquire);
            self.word += 1;
        }
    }
}

pub struct Chunk {
    index: usize,
    base: NonNull<u8>,
    size: usize,
    free: Mutex<FreeList>,
    free_bytes: AtomicUsize,
    state: AtomicU8,
}

unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

impl Chunk {
    fn new(index: usize, base: NonNull<u8>, size: usize, young: bool) -> Self {
        let base_addr = base.as_ptr() as usize;
        ChunkHeader::init(base_addr, young);
        Self {
            free: Mutex::new(FreeList::covering(
                NonNull::new(unsafe { base.as_ptr().add(HEADER_SIZE) }).unwrap(),
                size - HEADER_SIZE,
            )),
            free_bytes: AtomicUsize::new(size - HEADER_SIZE),
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

    pub fn header(&self) -> &ChunkHeader {
        ChunkHeader::of_chunk(self.base())
    }

    pub fn object_base(&self) -> usize {
        self.base() + HEADER_SIZE
    }

    pub fn object_size(&self) -> usize {
        self.size - HEADER_SIZE
    }

    pub fn is_young(&self) -> bool {
        self.header().young()
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
    young_chunks: usize,
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
            young_chunks: 0,
        })
    }

    pub fn young_chunk_count(&self) -> usize {
        self.young_chunks
    }

    pub fn reservation(&self) -> (usize, usize) {
        let base = self.buffer.start().as_ptr() as usize;
        (base, base + self.buffer.size())
    }

    pub fn is_young_addr(&self, addr: usize) -> bool {
        let (base, end) = self.reservation();
        addr >= base && addr < end && ChunkHeader::at(base, addr).young()
    }

    pub fn remembered_slots(&self) -> Vec<usize> {
        let mut slots = Vec::new();
        for chunk in self.slots.iter().flatten() {
            slots.extend(chunk.header().remembered_iter(chunk.object_size()));
        }
        slots
    }

    pub fn clear_remembered_sets(&self) {
        for chunk in self.slots.iter().flatten() {
            chunk.header().clear_remembered();
        }
    }

    pub fn young_indices(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().filter(|c| c.is_young()).map(|_| i))
            .collect()
    }

    pub fn remove_chunks(&mut self, indices: &[usize]) {
        for &index in indices {
            let removed = self.slots[index].take().unwrap();
            debug_assert!(removed.is_young(), "only young chunks are freed wholesale");
            self.young_chunks -= 1;
            self.buffer.decommit(removed.base, removed.size);
            drop(removed);
            self.pooled.push(index);
        }
    }

    pub fn finish_promotion(&mut self, chunk: &Chunk, cursor: usize) {
        let remainder = chunk.object_size() - cursor;
        let free = if remainder >= 2 * ALIGN {
            FreeList::from_promotion(chunk.object_base() + cursor, remainder, cursor)
        } else {
            FreeList::full(chunk.object_size())
        };
        chunk.free_bytes.store(free.free_bytes(), Ordering::Relaxed);
        *chunk.free.lock().unwrap() = free;
    }

    pub fn activate_promotion_chunk(&mut self) -> Option<Arc<Chunk>> {
        self.activate_chunk(false)
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn reserve_size(&self) -> usize {
        self.buffer.size()
    }

    pub fn committed_bytes(&self) -> usize {
        self.slots.iter().flatten().map(|c| c.object_size()).sum()
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

    pub fn available_chunks(&self) -> usize {
        self.pooled.len() + (self.slots.len() - self.carved)
    }

    pub fn pending_chunks(&self) -> usize {
        self.pending_chunks
    }

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
        self.allocate_impl(layout, host, false)
    }

    /// Nursery allocation for TLAB refills: only young chunks serve it, so
    /// freshly allocated small objects are always minor-GC candidates.
    pub fn allocate_young(
        &mut self,
        layout: Layout,
        host: Option<&GcHost>,
    ) -> (Option<NonNull<u8>>, Option<Arc<Chunk>>) {
        self.allocate_impl(layout, host, true)
    }

    fn allocate_impl(
        &mut self,
        layout: Layout,
        host: Option<&GcHost>,
        young_only: bool,
    ) -> (Option<NonNull<u8>>, Option<Arc<Chunk>>) {
        if let Some(ptr) = self.allocate_swept(layout, young_only) {
            return (Some(ptr), None);
        }
        if host.is_some() {
            if let Some(chunk) = self.claim_pending(young_only) {
                return (None, Some(chunk));
            }
        }
        match self.activate_chunk(true) {
            Some(chunk) => {
                let mut free = chunk.free.lock().unwrap();
                let ptr = free.allocate(layout);
                chunk.free_bytes.store(free.free_bytes(), Ordering::Relaxed);
                (ptr, None)
            }
            None => (None, None),
        }
    }

    fn allocate_swept(&mut self, layout: Layout, young_only: bool) -> Option<NonNull<u8>> {
        let need = need_for(layout);
        for slot in &self.slots {
            let Some(chunk) = slot else { continue };
            if young_only && !chunk.is_young() {
                continue;
            }
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

    pub fn claim_pending(&mut self, young_only: bool) -> Option<Arc<Chunk>> {
        if self.atomic_sweep_block {
            return None;
        }
        self.claim_pending_unblocked(young_only)
    }

    fn claim_pending_unblocked(&mut self, young_only: bool) -> Option<Arc<Chunk>> {
        for slot in &self.slots {
            let Some(chunk) = slot else { continue };
            if young_only && !chunk.is_young() {
                continue;
            }
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
        free.sweep(chunk.header(), chunk.object_size(), host);
        chunk.free_bytes.store(free.free_bytes(), Ordering::Relaxed);
        chunk.header().clear_mark();
        free.live_bytes()
    }

    pub fn publish_swept(&mut self, chunk: &Chunk, live: usize) {
        if live == 0 {
            let index = chunk.index;
            let removed = self.slots[index].take().unwrap();
            if removed.is_young() {
                self.young_chunks -= 1;
            }
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
        let base = unsafe { self.buffer.start().as_ptr().add(index * self.chunk_size) };
        let size = self
            .chunk_size
            .min(self.reserve_size() - index * self.chunk_size);
        (unsafe { NonNull::new_unchecked(base) }, size)
    }

    fn activate_chunk(&mut self, young: bool) -> Option<Arc<Chunk>> {
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
        let chunk = Arc::new(Chunk::new(index, base, size, young));
        if young {
            self.young_chunks += 1;
        }
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
        while let Some(chunk) = self.claim_pending_unblocked(false) {
            let live = Self::sweep_claimed(&chunk, host);
            self.publish_swept(&chunk, live);
        }
        !self.any_sweeping()
    }
}
