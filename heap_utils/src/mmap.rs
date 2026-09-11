use core::ptr::NonNull;

use heap_api::AllocError;

/// Lazily-committed anonymous memory mapping. Pages are provided by the OS
/// on first touch; nothing is committed up front.
pub struct MMapBuffer {
    start: NonNull<u8>,
    size: usize,
}

impl MMapBuffer {
    pub fn new(size: usize) -> Result<Self, AllocError> {
        let page = page_size();
        let size = size.next_multiple_of(page);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | map_noreserve(),
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(AllocError::OutOfMemory(
                core::alloc::Layout::from_size_align(size, page).unwrap(),
            ));
        }
        Ok(Self {
            start: unsafe { NonNull::new_unchecked(ptr.cast()) },
            size,
        })
    }

    pub fn start(&self) -> NonNull<u8> {
        self.start
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn contains(&self, addr: usize) -> bool {
        let base = self.start.as_ptr() as usize;
        (base..base + self.size).contains(&addr)
    }
    
    pub fn decommit(&self, addr: NonNull<u8>, len: usize) {
        debug_assert!(self.contains(addr.as_ptr() as usize));
        unsafe {
            libc::madvise(addr.as_ptr().cast(), len, libc::MADV_DONTNEED);
        }
    }
}

impl Drop for MMapBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.start.as_ptr().cast(), self.size);
        }
    }
}

#[cfg(target_os = "linux")]
fn map_noreserve() -> libc::c_int {
    libc::MAP_NORESERVE
}

#[cfg(not(target_os = "linux"))]
fn map_noreserve() -> libc::c_int {
    0
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
