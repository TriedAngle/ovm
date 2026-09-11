use core::alloc::Layout;
use core::ptr::{self, NonNull};

use heap_utils::Bitmap;

pub const ALIGN: usize = 16;
const FREE: usize = 0b1;

#[repr(C)]
pub struct BlockHeader {
    size: usize,
    next_free: *mut BlockHeader,
}

const _: () = assert!(core::mem::size_of::<BlockHeader>() == ALIGN);

impl BlockHeader {
    pub fn size(block: *mut BlockHeader) -> usize {
        unsafe { (*block).size & !FREE }
    }

    pub fn is_free(block: *mut BlockHeader) -> bool {
        unsafe { (*block).size & FREE != 0 }
    }

    pub fn payload(block: *mut BlockHeader) -> NonNull<u8> {
        unsafe { NonNull::new_unchecked(block.cast::<u8>().add(ALIGN)) }
    }

    pub fn block_of(payload: NonNull<u8>) -> *mut BlockHeader {
        unsafe { payload.as_ptr().sub(ALIGN) as *mut BlockHeader }
    }

    pub fn next(block: *mut BlockHeader) -> *mut u8 {
        unsafe { block.cast::<u8>().add(Self::size(block)) }
    }
}

/// First-fit free list over the arena's blocks, addressed through each
/// free block's `next_free` link and kept in address order.
pub struct FreeList {
    head: *mut BlockHeader,
    free_bytes: usize,
    live_bytes: usize,
}

impl FreeList {
    pub fn covering(base: NonNull<u8>, size: usize) -> Self {
        let block = base.as_ptr() as *mut BlockHeader;
        unsafe {
            *block = BlockHeader {
                size: size | FREE,
                next_free: ptr::null_mut(),
            };
        }
        Self {
            head: block,
            free_bytes: size,
            live_bytes: 0,
        }
    }

    pub fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    pub fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        debug_assert!(layout.size() > 0, "zero-sized allocation");
        debug_assert!(
            layout.align() <= ALIGN,
            "alignment above {ALIGN} unsupported"
        );
        let payload = layout.size().next_multiple_of(ALIGN);
        let need = payload + ALIGN;
        let mut link: *mut *mut BlockHeader = ptr::from_mut(&mut self.head);
        let mut block = self.head;
        while !block.is_null() {
            unsafe {
                let avail = BlockHeader::size(block);
                if avail >= need {
                    let remainder = avail - need;
                    if remainder >= 2 * ALIGN {
                        let rest = block.byte_add(need) as *mut BlockHeader;
                        *rest = BlockHeader {
                            size: remainder | FREE,
                            next_free: (*block).next_free,
                        };
                        (*block).size = need;
                        *link = rest;
                        self.free_bytes -= need;
                        self.live_bytes += need;
                    } else {
                        (*block).size = avail;
                        *link = (*block).next_free;
                        self.free_bytes -= avail;
                        self.live_bytes += avail;
                    }
                    return Some(BlockHeader::payload(block));
                }
                link = ptr::from_mut(&mut (*block).next_free);
                block = (*block).next_free;
            }
        }
        None
    }

    /// Frees every unmarked block, coalescing adjacent free runs, and
    /// rebuilds the free list in address order.
    pub fn sweep(&mut self, base: NonNull<u8>, size: usize, bitmap: &Bitmap) {
        let mut addr = base.as_ptr();
        let end = unsafe { base.as_ptr().add(size) };
        self.head = ptr::null_mut();
        self.free_bytes = 0;
        let mut tail: *mut BlockHeader = ptr::null_mut();
        let mut live = 0usize;
        let mut run_start: *mut u8 = ptr::null_mut();
        let mut run_len = 0usize;
        while addr < end {
            let block = addr as *mut BlockHeader;
            let block_size = BlockHeader::size(block);
            let keep = !BlockHeader::is_free(block)
                && bitmap.is_set(BlockHeader::payload(block).as_ptr() as usize);
            if keep {
                push_free(self, &mut tail, run_start, run_len);
                run_start = ptr::null_mut();
                run_len = 0;
                live += block_size;
            } else {
                if run_start.is_null() {
                    run_start = addr;
                }
                run_len += block_size;
            }
            addr = unsafe { addr.add(block_size) };
        }
        push_free(self, &mut tail, run_start, run_len);
        self.live_bytes = live;
    }
}

fn push_free(this: &mut FreeList, tail: &mut *mut BlockHeader, run_start: *mut u8, run_len: usize) {
    if run_start.is_null() {
        return;
    }
    let block = run_start as *mut BlockHeader;
    unsafe {
        *block = BlockHeader {
            size: run_len | FREE,
            next_free: ptr::null_mut(),
        };
        if (*tail).is_null() {
            this.head = block;
        } else {
            (**tail).next_free = block;
        }
    }
    *tail = block;
    this.free_bytes += run_len;
}

pub fn blocks(base: NonNull<u8>, size: usize) -> Blocks {
    Blocks {
        addr: base.as_ptr(),
        end: unsafe { base.as_ptr().add(size) },
    }
}

pub struct Blocks {
    addr: *mut u8,
    end: *mut u8,
}

impl Iterator for Blocks {
    type Item = *mut BlockHeader;

    fn next(&mut self) -> Option<*mut BlockHeader> {
        if self.addr >= self.end {
            return None;
        }
        let block = self.addr as *mut BlockHeader;
        self.addr = BlockHeader::next(block);
        Some(block)
    }
}
