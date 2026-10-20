//! Asynchronous copies beside compute (gpu/transfer.hpp): between page-locked host memory and
//! device buffers, on upload and download streams of the device's own, ordered against the
//! compute stream by marks.

use std::ffi::c_void;

use straydog_gpu::ffi;

use crate::gpu::{Backend, Result};
use crate::ops::Buffer;

/// A device stream: compute (where ops run), uploads, downloads, or later uploads (a second
/// upload queue, so copies that can wait do not hold up those a pass waits for).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Compute = 0,
    Upload = 1,
    Download = 2,
    LaterUpload = 3,
}

/// A point in a stream's queue (ids grow from 1 across a device's streams).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mark(pub i64);

fn check(status: i32) -> Result<()> {
    straydog_gpu::check(status).map(drop).map_err(|error| error.to_string())
}

/// Page-lock a host range that devices may read and write, until [`unpin_host`].
///
/// # Safety
/// The range stays allocated until unpinned.
pub unsafe fn pin_host(pointer: *mut u8, bytes: usize) -> Result<()> {
    // SAFETY: the caller keeps the range alive while it is registered.
    check(unsafe { ffi::sd_host_pin(pointer.cast(), bytes as u64) })
}

/// # Safety
/// `pointer` starts a range given to [`pin_host`], with no copy of it still pending.
pub unsafe fn unpin_host(pointer: *mut u8) -> Result<()> {
    // SAFETY: as documented.
    check(unsafe { ffi::sd_host_unregister(pointer.cast()) })
}

impl Backend {
    /// Queue a copy of `bytes` page-locked bytes at `host` into `target` at `offset` (either
    /// upload stream), or out of it into `host` (downloads).
    ///
    /// # Safety
    /// `host` stays allocated, and unchanged by anything else, until a mark of that stream
    /// recorded after this call is done.
    pub unsafe fn transfer(&mut self, stream: Stream, target: &Buffer, offset: u64, host: *mut u8, bytes: u64) -> Result<()> {
        if stream == Stream::Compute {
            return Err("a transfer runs on the upload or download stream".into());
        }
        // SAFETY: a live handle; the library checks the device range and that the host range is
        // page-locked (writable for downloads); the caller keeps the host bytes as documented.
        check(unsafe { ffi::sd_transfer(self.handle(), stream as i32, target.base(), offset, host.cast::<c_void>(), bytes) })
    }

    /// Mark the work queued so far on `stream`.
    pub fn mark(&mut self, stream: Stream) -> Result<Mark> {
        // SAFETY: a live handle.
        let id = unsafe { ffi::sd_event_record(self.handle(), stream as i32) };
        if id < 1 {
            return Err(straydog_gpu::last_error().to_string());
        }
        Ok(Mark(id))
    }

    /// Make `stream` wait, on the GPU, until `mark` completes.
    pub fn wait_mark(&mut self, stream: Stream, mark: Mark) -> Result<()> {
        // SAFETY: a live handle; the library refuses unknown ids.
        check(unsafe { ffi::sd_event_wait(self.handle(), stream as i32, mark.0) })
    }

    /// Whether `mark` completed; with `block`, after waiting for it.
    pub fn mark_done(&mut self, mark: Mark, block: bool) -> Result<bool> {
        // SAFETY: a live handle; the library refuses unknown ids.
        let state = unsafe { ffi::sd_event_done(self.handle(), mark.0, i32::from(block)) };
        check(state.min(0))?;
        Ok(state == 1)
    }

    /// Uploads and their bytes, then downloads and theirs, since the device was created.
    pub fn transfer_counts(&mut self) -> Result<[u64; 4]> {
        let mut counts = [0u64; 4];
        // SAFETY: a live handle and four counters.
        check(unsafe { ffi::sd_transfer_counts(self.handle(), counts.as_mut_ptr()) })?;
        Ok(counts)
    }
}
