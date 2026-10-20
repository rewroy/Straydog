//! Reads of the pinned files at the device's speed: page-aligned host buffers, direct reads
//! that bypass the page cache (Linux), and page-cache residency probes.
//!
//! An NVMe drive reads 4-5 GB/s only with requests in flight: a few threads each reading
//! 8 MiB at a time keep it busy whether the reads go through the page cache or around it.
//! Direct reads are DMA into the caller's buffer: no kernel copy and no cached pages
//! evicting others, at the price of reading from the device again next time. Random row
//! reads of a table larger than memory use the same handles (4 KiB-aligned offsets).

use std::fs::File;
use std::ptr::NonNull;

/// Alignment of direct reads: offsets, lengths and buffers (the logical block size or a
/// multiple of it).
pub const ALIGN: usize = 4096;
/// One direct read request; ~2 ms of the device, so a handful of readers keep its queue full.
pub const PIECE: usize = 8 << 20;
/// One read through the page cache: small enough that hashing it finds it in the core's cache.
pub const CACHED_PIECE: usize = 256 << 10;

/// A zeroed, page-aligned host buffer, backed by transparent huge pages on Linux.
pub struct Buffer {
    pointer: NonNull<u8>,
    len: usize,
}

// A plain owned allocation.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(len: usize) -> Buffer {
        let layout = Self::layout(len);
        // SAFETY: a non-zero size and a valid alignment.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        crate::alloc::advise(pointer.as_ptr(), layout.size());
        Buffer { pointer, len }
    }

    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len.max(1).next_multiple_of(ALIGN), ALIGN).unwrap()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// The allocation's size: `len` rounded up to whole blocks.
    pub fn capacity(&self) -> usize {
        Self::layout(self.len).size()
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: an initialized (zeroed) allocation of `len` bytes owned by self.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), Self::layout(self.len)) }
    }
}

/// A second handle on an open file whose reads bypass the page cache (O_DIRECT), reopened
/// through /proc so that it is the same file, not whatever its path names now; None where
/// unsupported.
pub fn direct(file: &File) -> Option<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        let reopened = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .ok()?;
        let (before, after) = (file.metadata().ok()?, reopened.metadata().ok()?);
        (before.dev() == after.dev() && before.ino() == after.ino()).then_some(reopened)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        None
    }
}

/// The fraction of a mapped range's pages in the page cache (0 where unknown).
pub fn resident(range: &[u8]) -> f64 {
    #[cfg(target_os = "linux")]
    {
        if range.is_empty() {
            return 0.0;
        }
        let start = range.as_ptr() as usize & !(ALIGN - 1);
        let end = range.as_ptr() as usize + range.len();
        let pages = (end - start).div_ceil(ALIGN);
        let mut flags = vec![0u8; pages];
        // SAFETY: the page-aligned span lies within one live mapping; one flag byte per page.
        if unsafe { libc::mincore(start as *mut libc::c_void, end - start, flags.as_mut_ptr()) } != 0 {
            return 0.0;
        }
        flags.iter().filter(|flag| *flag & 1 == 1).count() as f64 / pages as f64
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = range;
        0.0
    }
}

/// Fill `buffer` from `offset` of `file`, through the page cache.
pub fn read_at(file: &File, offset: u64, buffer: &mut [u8]) -> std::io::Result<()> {
    let mut done = 0;
    while done < buffer.len() {
        done += read_some(file, &mut buffer[done..], offset + done as u64)?;
    }
    Ok(())
}

/// Read `len` bytes from `offset` of `file` into `buffer[at..at + len]`. A direct handle needs
/// an aligned offset and position; its reads end on a block boundary, which may run past the
/// file's end (they stop there) and past `len` into the buffer's capacity.
pub fn read_into(file: &File, offset: u64, buffer: &mut Buffer, at: usize, len: usize, direct: bool)
                 -> std::io::Result<()> {
    if !direct {
        return read_at(file, offset, &mut buffer.as_mut_slice()[at..at + len]);
    }
    if (at + len).next_multiple_of(ALIGN) > buffer.capacity() || !offset.is_multiple_of(ALIGN as u64)
        || !at.is_multiple_of(ALIGN)
    {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "misaligned Straydog direct read"));
    }
    let mut done = 0;
    while done < len {
        let want = (len - done).next_multiple_of(ALIGN);
        // SAFETY: [at + done, at + done + want) lies within the buffer's capacity (checked above).
        let target = unsafe { std::slice::from_raw_parts_mut(buffer.pointer.as_ptr().add(at + done), want) };
        done += read_some(file, target, offset + done as u64)?.min(len - done);
    }
    Ok(())
}

/// One read: at least a byte, or an error at the end of the file.
fn read_some(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    let count = std::os::unix::fs::FileExt::read_at(file, buffer, offset)?;
    #[cfg(windows)]
    let count = std::os::windows::fs::FileExt::seek_read(file, buffer, offset)?;
    if count == 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file shorter than its size"));
    }
    Ok(count)
}

/// Start reading the pages under these ranges of a mapped file in the background: just their
/// pages, no read-ahead past them. A later access then finds them in memory instead of waiting
/// on one page fault after another.
pub fn read_ahead<'a>(ranges: impl Iterator<Item = &'a [u8]>) {
    const PAGE: usize = 4096;
    let pages = ranges.filter(|range| !range.is_empty()).map(|range| {
        let start = range.as_ptr() as usize / PAGE * PAGE;
        (start, (range.as_ptr() as usize + range.len()).div_ceil(PAGE) * PAGE - start)
    });
    #[cfg(unix)]
    for (start, length) in pages {
        // SAFETY: advice on pages of a live read-only mapping (the ranges borrow it); it starts
        // reads and changes nothing.
        unsafe { libc::madvise(start as *mut libc::c_void, length, libc::MADV_WILLNEED) };
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let entries: Vec<WIN32_MEMORY_RANGE_ENTRY> = pages
            .map(|(start, length)| WIN32_MEMORY_RANGE_ENTRY { VirtualAddress: start as *mut _, NumberOfBytes: length })
            .collect();
        if !entries.is_empty() {
            // SAFETY: entries describe pages of live mappings; prefetching only reads them in.
            unsafe { PrefetchVirtualMemory(GetCurrentProcess(), entries.len(), entries.as_ptr(), 0) };
        }
    }
}
