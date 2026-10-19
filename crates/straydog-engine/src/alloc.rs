//! A global allocator for processes that move large host buffers (prompt rows, logits).
//!
//! Large allocations come from fresh anonymous mappings, so a 160 MB prompt buffer costs
//! some 40,000 page faults on first touch. As NumPy does, the allocator asks Linux to back
//! allocations of 4 MiB and more with transparent huge pages (2 MiB faults instead).

use std::alloc::{GlobalAlloc, Layout, System};

pub struct HugePages;

const HUGE_THRESHOLD: usize = 4 << 20;

unsafe impl GlobalAlloc for HugePages {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let pointer = unsafe { System.alloc(layout) };
        advise(pointer, layout.size());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        advise(pointer, layout.size());
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        advise(pointer, size);
        pointer
    }
}

fn advise(pointer: *mut u8, size: usize) {
    #[cfg(target_os = "linux")]
    if !pointer.is_null() && size >= HUGE_THRESHOLD {
        // The whole 2 MiB-aligned span inside the allocation; advice is only a hint.
        let start = (pointer as usize).next_multiple_of(2 << 20);
        let end = (pointer as usize + size) & !((2 << 20) - 1);
        if end > start {
            // SAFETY: the range lies inside memory this allocation owns; MADV_HUGEPAGE changes no contents.
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (pointer, size);
}
