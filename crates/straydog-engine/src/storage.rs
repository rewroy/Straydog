//! Reads of the pinned files at the device's speed: page-aligned host buffers, direct reads
//! that bypass the page cache (Linux), and page-cache residency probes.
//!
//! An NVMe drive reads 4-5 GB/s only with requests in flight: a few threads each reading
//! 8 MiB at a time keep it busy whether the reads go through the page cache or around it.
//! Direct reads are DMA into the caller's buffer: no kernel copy and no cached pages
//! evicting others, at the price of reading from the device again next time. Random row
//! reads of a table larger than memory use the same handles (4 KiB-aligned offsets).
//!
//! On Windows the I/O manager runs the reads issued on a synchronous handle (one opened without
//! `FILE_FLAG_OVERLAPPED`, which is what `File::open` makes) one at a time, however many threads
//! issue them, and a duplicate of the handle (`try_clone`) is the same file object. So the handles
//! that threads read at once through are opened for overlapped I/O ([`direct`], [`concurrent`]),
//! and every positional read goes through [`read_partial`], which waits for its own read alone.
//! Std's `seek_read` is not used on them: see [`read_partial`].

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
/// unsupported. On Windows it is opened for overlapped I/O (`FILE_FLAG_NO_BUFFERING |
/// FILE_FLAG_OVERLAPPED`): read it only through this module's functions, from as many threads as
/// there are reads to run at once.
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
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::{FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED};
        overlapped::reopen(file, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = file;
        None
    }
}

/// A second handle on an open file for positional reads ([`read_at`], [`read_into`]) from many
/// threads at once, through the page cache. On Windows, a handle opened for overlapped I/O, made
/// once per file (None if the system refuses one); elsewhere None, as the file's own handle
/// already reads concurrently. The file's own handle stays what maps it and does the rest.
pub fn concurrent(file: &File) -> Option<File> {
    #[cfg(windows)]
    {
        overlapped::reopen(file, windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OVERLAPPED)
    }
    #[cfg(not(windows))]
    {
        let _ = file;
        None
    }
}

/// An owned handle on `file` for the positional reads of many threads: [`concurrent`]'s, or else a
/// clone.
pub fn reader(file: &File) -> std::io::Result<File> {
    concurrent(file).map_or_else(|| file.try_clone(), Ok)
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
    let count = read_partial(file, buffer, offset)?;
    if count == 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file shorter than its size"));
    }
    Ok(count)
}

/// One positional read of `file` at `offset`: the bytes it got (0 at the end of the file), without
/// relying on the handle's position. This is the only way to read a handle of [`direct`] or
/// [`concurrent`], and serves any other handle too (on a synchronous handle on Windows the reads
/// still run one at a time).
///
/// On Windows it is `ReadFile` with an `OVERLAPPED` that holds the offset and an event of the
/// calling thread's own, then a wait for that read alone (`GetOverlappedResult`). Std's `seek_read`
/// cannot do this on an overlapped handle: it calls `NtReadFile` with a status block on its stack
/// and no event, and when the read is pending waits on the file handle itself, which any read
/// completing on the handle signals. With reads in flight on one handle it can return when another
/// thread's read completes and then, finding its own still pending, aborts the process (or have
/// left the kernel writing into a frame that is gone).
pub fn read_partial(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_at(file, buffer, offset)
    }
    #[cfg(windows)]
    {
        overlapped::read_at(file, buffer, offset)
    }
}

#[cfg(windows)]
mod overlapped {
    use std::fs::File;
    use std::io::{Error, Result};
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_HANDLE_EOF, ERROR_IO_PENDING, GENERIC_READ, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{ReOpenFile, ReadFile, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE};
    use windows_sys::Win32::System::Threading::CreateEventW;
    use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};

    /// The most one `ReadFile` takes, a whole number of blocks.
    const MOST: usize = (u32::MAX as usize) & !(super::ALIGN - 1);

    /// A new handle on the same file, read-only, opened with `flags`; None if the system refuses.
    pub fn reopen(file: &File, flags: u32) -> Option<File> {
        // SAFETY: a live handle; the result is a new handle on the same file, or invalid.
        let handle = unsafe {
            ReOpenFile(file.as_raw_handle() as _, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, flags)
        };
        // SAFETY: a handle this call owns.
        (handle != INVALID_HANDLE_VALUE).then(|| unsafe { File::from_raw_handle(handle as _) })
    }

    /// A manual-reset event of one thread, which waits on it for its own reads, one at a time.
    struct Event(HANDLE);

    impl Drop for Event {
        fn drop(&mut self) {
            // SAFETY: an event this thread created and nothing else uses.
            unsafe { CloseHandle(self.0) };
        }
    }

    thread_local! {
        /// The thread's event, or the error code of making it.
        static EVENT: std::result::Result<Event, i32> = {
            // SAFETY: no security attributes and no name: an event only this thread knows.
            let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
            if event.is_null() { Err(unsafe { GetLastError() } as i32) } else { Ok(Event(event)) }
        };
    }

    /// One read of `buffer` at `offset`: the bytes got, 0 at the end of the file. It returns only
    /// when the read has finished (the kernel writes into `buffer` and the `OVERLAPPED` until then).
    pub fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        EVENT.with(|event| {
            let event = event.as_ref().map_err(|&code| Error::from_raw_os_error(code))?;
            // SAFETY: all zero is a valid OVERLAPPED (no event, offset 0).
            let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
            overlapped.Anonymous.Anonymous.Offset = offset as u32;
            overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
            overlapped.hEvent = event.0;
            let handle = file.as_raw_handle() as HANDLE;
            let mut count = 0u32;
            // SAFETY: `buffer` and `overlapped` outlive the read: a pending read is waited for below, and
            // nothing between the two can unwind.
            let started = unsafe {
                ReadFile(handle, buffer.as_mut_ptr().cast(), buffer.len().min(MOST) as u32, &mut count, &mut overlapped)
            };
            if started == 0 {
                // SAFETY: reads the thread's last error.
                match unsafe { GetLastError() } {
                    ERROR_HANDLE_EOF => return Ok(0),
                    ERROR_IO_PENDING => {
                        // SAFETY: the read above is this OVERLAPPED's; waiting on it waits on its event alone.
                        if unsafe { GetOverlappedResult(handle, &overlapped, &mut count, 1) } == 0 {
                            // SAFETY: as above.
                            return match unsafe { GetLastError() } {
                                ERROR_HANDLE_EOF => Ok(0),
                                code => Err(Error::from_raw_os_error(code as i32)),
                            };
                        }
                    }
                    code => return Err(Error::from_raw_os_error(code as i32)),
                }
            }
            Ok(count as usize)
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const THREADS: usize = 16;
    const READS: usize = 300;

    /// The byte at `index` of the test files: unlike a repeating pattern, a read from the wrong place shows.
    fn byte(index: u64) -> u8 {
        (index.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8
    }

    /// A xorshift generator, one per thread.
    struct Random(u64);

    impl Random {
        fn below(&mut self, limit: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % limit
        }
    }

    /// A temporary file of `blocks` blocks and `tail` bytes more, byte `i` being `byte(i)`; removed with it.
    struct Temp {
        path: std::path::PathBuf,
        file: File,
        size: u64,
    }

    impl Temp {
        fn new(blocks: u64, tail: u64) -> Temp {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!("straydog-storage-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            let size = blocks * ALIGN as u64 + tail;
            let data: Vec<u8> = (0..size).map(byte).collect();
            File::create(&path).unwrap().write_all(&data).unwrap();
            Temp { file: File::open(&path).unwrap(), path, size }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn assert_bytes(bytes: &[u8], start: u64) {
        for (index, &found) in bytes.iter().enumerate() {
            assert_eq!(found, byte(start + index as u64), "byte {index} of a read at {start}");
        }
    }

    /// Every thread makes `READS` reads of random ranges of a file of `size` bytes with `read`, which is given the start and the length
    /// and returns the bytes; whole blocks from a block boundary when `aligned`.
    fn threads_read(size: u64, aligned: bool, read: &(dyn Fn(u64, usize) -> Vec<u8> + Sync)) {
        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                scope.spawn(move || {
                    let mut random = Random(thread as u64 * 7919 + 1);
                    for _ in 0..READS {
                        let (start, most) = if aligned {
                            (random.below(size.div_ceil(ALIGN as u64)) * ALIGN as u64, 8 * ALIGN as u64)
                        } else {
                            (random.below(size), 100_000)
                        };
                        let mut len = (random.below(most) + 1).min(size - start) as usize;
                        if aligned {
                            len = len.next_multiple_of(ALIGN).min((size - start) as usize);
                        }
                        assert_bytes(&read(start, len), start);
                    }
                });
            }
        });
    }

    #[cfg(windows)]
    #[test]
    fn many_threads_read_random_blocks_through_one_direct_handle() {
        // Eight MiB and a block cut short by the end of the file, which a read of it stops at.
        let temp = Temp::new(2048, 1000);
        let direct = direct(&temp.file).expect("a direct handle");
        threads_read(temp.size, true, &|start, len| {
            let mut buffer = Buffer::new(8 * ALIGN);
            read_into(&direct, start, &mut buffer, 0, len, true).unwrap();
            buffer.as_slice()[..len].to_vec()
        });
    }

    #[test]
    fn many_threads_read_random_ranges_through_every_kind_of_handle() {
        let temp = Temp::new(300, 123);
        let concurrent = reader(&temp.file).unwrap();
        for file in [&temp.file, &concurrent] {
            threads_read(temp.size, false, &|start, len| {
                let mut bytes = vec![0u8; len];
                read_at(file, start, &mut bytes).unwrap();
                bytes
            });
        }
    }

    #[test]
    fn a_read_at_the_end_of_the_file_gets_what_is_there_and_fails_as_before() {
        let temp = Temp::new(2, 100);
        let (concurrent, direct) = (concurrent(&temp.file), direct(&temp.file));
        let mut handles = vec![(&temp.file, false)];
        handles.extend(concurrent.iter().map(|file| (file, false)));
        handles.extend(direct.iter().map(|file| (file, true)));
        let mut block = Buffer::new(ALIGN);
        let end = 2 * ALIGN as u64;
        for (file, direct) in handles {
            // Across the end: the bytes up to it. At it and past it: none.
            assert_eq!(read_partial(file, block.as_mut_slice(), end).unwrap(), 100);
            if !direct {
                assert_eq!(read_partial(file, block.as_mut_slice(), end + 100).unwrap(), 0);
            }
            assert_eq!(read_partial(file, block.as_mut_slice(), end + ALIGN as u64).unwrap(), 0);
            // `read_into` takes the short last block, and fails past the end as `read_at` does.
            read_into(file, end, &mut block, 0, 100, direct).unwrap();
            assert_bytes(&block.as_slice()[..100], end);
            let error = read_into(file, end + ALIGN as u64, &mut block, 0, 100, direct).unwrap_err();
            assert_eq!((error.kind(), error.to_string().as_str()), (std::io::ErrorKind::UnexpectedEof, "file shorter than its size"));
        }
        let mut bytes = vec![0u8; 200];
        let error = read_at(&temp.file, end, &mut bytes).unwrap_err();
        assert_eq!((error.kind(), error.to_string().as_str()), (std::io::ErrorKind::UnexpectedEof, "file shorter than its size"));
    }

    #[cfg(windows)]
    #[test]
    fn a_direct_read_must_be_aligned() {
        let temp = Temp::new(2, 0);
        let direct = direct(&temp.file).expect("a direct handle");
        let mut block = Buffer::new(2 * ALIGN);
        for (offset, at) in [(1, 0), (0, 1)] {
            let error = read_into(&direct, offset, &mut block, at, ALIGN, true).unwrap_err();
            assert_eq!(error.to_string(), "misaligned Straydog direct read");
        }
    }

    /// Offsets past 4 GiB take their high half through `OVERLAPPED.OffsetHigh`: a sparse file with one block there.
    #[cfg(windows)]
    #[test]
    fn reads_at_offsets_past_four_gib_get_their_bytes() {
        use std::os::windows::fs::FileExt;
        use std::os::windows::io::AsRawHandle;
        let temp = Temp::new(0, 0);
        let file = std::fs::OpenOptions::new().read(true).write(true).open(&temp.path).unwrap();
        let mut returned = 0u32;
        // SAFETY: a live handle; FSCTL_SET_SPARSE takes no buffers (without one it marks the whole file).
        let sparse = unsafe {
            windows_sys::Win32::System::IO::DeviceIoControl(file.as_raw_handle() as _, 0x0009_00C4, std::ptr::null(), 0,
                                                            std::ptr::null_mut(), 0, &mut returned, std::ptr::null_mut())
        };
        if sparse == 0 {
            eprintln!("skipped: the temporary directory's file system has no sparse files");
            return;
        }
        let start = (5u64 << 30) + ALIGN as u64;
        file.set_len(start + 2 * ALIGN as u64).unwrap();
        let data: Vec<u8> = (0..ALIGN as u64).map(|index| byte(start + index)).collect();
        assert_eq!(file.seek_write(&data, start).unwrap(), ALIGN);
        let mut block = Buffer::new(ALIGN);
        for (handle, direct) in [(Some(File::open(&temp.path).unwrap()), false), (concurrent(&file), false), (direct(&file), true)]
            .into_iter().filter_map(|(handle, direct)| handle.map(|handle| (handle, direct)))
        {
            block.as_mut_slice().fill(0);
            read_into(&handle, start, &mut block, 0, ALIGN, direct).unwrap();
            assert_bytes(block.as_slice(), start);
        }
    }
}
