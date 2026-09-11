use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{AllocError, GcHost};
use heap_utils::{Bitmap, MMapBuffer};

use crate::block::{ALIGN, FreeList};

pub const CHUNK_SIZE: usize = 256 * 1024;

pub struct Chunk {
    base: NonNull<u8>,
    size: usize,
    pub bitmap: Bitmap,
    pub free: FreeList,
}

impl Chunk {
    fn new(base: NonNull<u8>, size: usize) -> Self {
        Self {
            bitmap: Bitmap::new(base.as_ptr() as usize, size, ALIGN),
            free: FreeList::covering(base, size),
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
}

pub struct ChunkedHeap {
    buffer: MMapBuffer,
    chunk_size: usize,
    slots: Vec<Option<Chunk>>,
    pooled: Vec<usize>,
    carved: usize,
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
        })
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn reserve_size(&self) -> usize {
        self.buffer.size()
    }

    /// Bytes currently committed: active chunks only, pooled ones excluded.
    pub fn committed_bytes(&self) -> usize {
        self.slots.iter().flatten().map(|c| c.size()).sum()
    }

    pub fn live_bytes(&self) -> usize {
        self.slots.iter().flatten().map(|c| c.free.live_bytes()).sum()
    }

    pub fn free_bytes(&self) -> usize {
        self.slots.iter().flatten().map(|c| c.free.free_bytes()).sum()
    }

    pub fn active_chunks(&self) -> usize {
        self.carved - self.pooled.len()
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

    pub fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        for slot in &mut self.slots {
            if let Some(chunk) = slot {
                if chunk.free.free_bytes() == 0 {
                    continue;
                }
                if let Some(ptr) = chunk.free.allocate(layout) {
                    return Some(ptr);
                }
            }
        }
        self.activate_chunk()
            .and_then(|chunk| chunk.free.allocate(layout))
    }

    fn chunk_region(&self, index: usize) -> (NonNull<u8>, usize) {
        let base =
            unsafe { self.buffer.start().as_ptr().add(index * self.chunk_size) };
        let size = self
            .chunk_size
            .min(self.reserve_size() - index * self.chunk_size);
        (unsafe { NonNull::new_unchecked(base) }, size)
    }

    fn activate_chunk(&mut self) -> Option<&mut Chunk> {
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
        self.slots[index] = Some(Chunk::new(base, size));
        self.slots[index].as_mut()
    }

    pub fn clear_bitmaps(&self) {
        for chunk in self.slots.iter().flatten() {
            chunk.bitmap.clear_all();
        }
    }

    pub fn sweep(&mut self, host: &GcHost) -> usize {
        let mut live = 0usize;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let empty = match slot {
                Some(chunk) => {
                    chunk
                        .free
                        .sweep(chunk.base_ptr(), chunk.size(), &chunk.bitmap, host);
                    chunk.free.live_bytes() == 0
                }
                None => continue,
            };
            if empty {
                let chunk = slot.take().unwrap();
                self.buffer.decommit(chunk.base, chunk.size);
                drop(chunk);
                self.pooled.push(index);
            } else if let Some(chunk) = slot {
                live += chunk.free.live_bytes();
            }        }
        live
    }

    pub fn for_each_live(&self, mut f: impl FnMut(usize)) {
        for chunk in self.slots.iter().flatten() {
            for addr in chunk.bitmap.iter_set() {
                f(addr);
            }
        }
    }
}
