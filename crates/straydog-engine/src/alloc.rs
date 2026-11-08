//! Transparent huge pages for the long-lived host buffers ([`crate::storage::Buffer`]: the expert host tier's arenas, the loader's slots).
//!
//! Only those are advised. A process whose every large allocation asked for huge pages (a global allocator did, for a prompt's rows and
//! logits) had Linux compact memory at those allocations' first touch whenever it was short of free memory, as a long prompt beside a
//! 102 GiB host tier is; compaction migrated the host tier's small pages, and each migration of a page-locked range stopped every GPU queue
//! of the process (a KFD eviction): a 93K-token prompt on two R9700s stood still some 10 s of its 67. Huge pages are passed by compaction.

#[cfg(target_os = "linux")]
const HUGE_THRESHOLD: usize = 4 << 20;

/// Ask Linux for transparent huge pages behind an allocation of 4 MiB or more (a hint), before its first touch.
pub fn advise(pointer: *mut u8, size: usize) {
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
