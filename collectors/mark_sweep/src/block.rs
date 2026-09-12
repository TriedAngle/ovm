use core::alloc::Layout;
use core::ptr::{self, NonNull};

use heap_api::GcHost;
use heap_utils::Bitmap;

pub const ALIGN: usize = 16;

pub fn need_for(layout: Layout) -> usize {
    debug_assert!(layout.size() > 0, "zero-sized allocation");
    debug_assert!(
        layout.align() <= ALIGN,
        "alignment above {ALIGN} unsupported"
    );
    layout.size().next_multiple_of(ALIGN)
}

/// Header of a free run
#[repr(C)]
pub struct FreeHeader {
    size: usize,
    next: *mut FreeHeader,
}

pub struct FreeList {
    head: *mut FreeHeader,
    free_bytes: usize,
    live_bytes: usize,
}

impl FreeList {
    pub fn covering(base: NonNull<u8>, size: usize) -> Self {
        let head = base.as_ptr() as *mut FreeHeader;
        unsafe {
            *head = FreeHeader {
                size,
                next: ptr::null_mut(),
            };
        }
        Self {
            head,
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

    /// First-fit
    pub fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let need = need_for(layout);
        let mut link: *mut *mut FreeHeader = ptr::from_mut(&mut self.head);
        let mut run = self.head;
        while !run.is_null() {
            unsafe {
                let available = (*run).size;
                if available >= need {
                    let remainder = available - need;
                    let given = if remainder >= 2 * ALIGN {
                        let remainder_header = run.byte_add(need) as *mut FreeHeader;
                        *remainder_header = FreeHeader {
                            size: remainder,
                            next: (*run).next,
                        };
                        *link = remainder_header;
                        need
                    } else {
                        *link = (*run).next;
                        available
                    };
                    self.free_bytes -= given;
                    self.live_bytes += given;
                    return Some(NonNull::new_unchecked(run.cast::<u8>()));
                }
                link = ptr::from_mut(&mut (*run).next);
                run = (*run).next;
            }
        }
        None
    }

    /// Rebuilds the free list from the mark bitmap
    pub fn sweep(&mut self, base: NonNull<u8>, size: usize, bitmap: &Bitmap, host: &GcHost) {
        let start = base.as_ptr() as usize;
        let end = start + size;
        self.head = ptr::null_mut();
        self.free_bytes = 0;
        let mut live = 0usize;
        let mut tail: *mut FreeHeader = ptr::null_mut();
        let mut gap_start = start;
        for addr in bitmap.iter_set() {
            let object = unsafe { NonNull::new_unchecked(addr as *mut ()) };
            let object_size = (host.layout_of)(object).size().next_multiple_of(ALIGN);
            let gap = addr - gap_start;
            if gap >= 2 * ALIGN {
                push_free(self, &mut tail, gap_start, gap);
            } else {
                live += gap;
            }
            live += object_size;
            gap_start = addr + object_size;
        }
        let gap = end - gap_start;
        if gap >= 2 * ALIGN {
            push_free(self, &mut tail, gap_start, gap);
        } else {
            live += gap;
        }
        self.live_bytes = live;
    }
}

fn push_free(list: &mut FreeList, tail: &mut *mut FreeHeader, addr: usize, size: usize) {
    let header = addr as *mut FreeHeader;
    unsafe {
        *header = FreeHeader {
            size,
            next: ptr::null_mut(),
        };
        if (*tail).is_null() {
            list.head = header;
        } else {
            (**tail).next = header;
        }
    }
    *tail = header;
    list.free_bytes += size;
}
