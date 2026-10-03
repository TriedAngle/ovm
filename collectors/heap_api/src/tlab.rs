use core::ptr::{self, NonNull};

/// A thread-local allocation window: memory the owning thread bumps into
/// objects without touching the collector.
///
/// The VM owns the window; collectors only hand out fresh ones through
/// [`LocalHeap::allocate_slow`](crate::LocalHeap::allocate_slow). Every
/// window starts [`Tlab::ALIGN`]-aligned and is consumed in multiples of
/// that alignment, so `try_alloc` never needs to realign the cursor.
#[derive(Debug)]
pub struct Tlab {
    cursor: *mut u8,
    end: *mut u8,
}

// SAFETY: the cursors are plain addresses into the owning thread's heap; a
// Tlab is only ever touched by the thread that owns it.
unsafe impl Send for Tlab {}

impl Tlab {
    /// The allocation alignment guaranteed by the heap: every object starts
    /// at a multiple of this. Larger object alignments are unsupported.
    pub const ALIGN: usize = 16;

    pub const fn empty() -> Self {
        Self {
            cursor: ptr::null_mut(),
            end: ptr::null_mut(),
        }
    }

    /// A window covering `[start, start + len)`. `start` must be
    /// `ALIGN`-aligned and `len` a multiple of it.
    pub fn new(start: NonNull<u8>, len: usize) -> Self {
        debug_assert_eq!(start.as_ptr() as usize % Self::ALIGN, 0);
        debug_assert_eq!(len % Self::ALIGN, 0);
        Self {
            cursor: start.as_ptr(),
            end: unsafe { start.as_ptr().add(len) },
        }
    }

    /// Reserve `size` bytes (a multiple of `ALIGN`) from the window, or
    /// `None` when it does not fit. Never touches the collector.
    #[inline(always)]
    pub fn try_alloc(&mut self, size: usize) -> Option<NonNull<u8>> {
        let cursor = self.cursor;
        if cursor.is_null() {
            return None;
        }
        let next = unsafe { cursor.add(size) };
        if next > self.end {
            return None;
        }
        self.cursor = next;
        NonNull::new(cursor)
    }

    /// Drop the window: after a park or collection the collector may
    /// reclaim the span, so the owner must treat it as gone.
    pub fn invalidate(&mut self) {
        self.cursor = ptr::null_mut();
        self.end = ptr::null_mut();
    }
}
