//! Host exchanges (gpu/host_exchange.hpp): page-locked host memory a device's ops read and write
//! within a pass, so the host takes part in a pass queued (or recorded as a graph) ahead of it. An
//! export op hands the host values with a running count; the host computes from them and publishes
//! its result; a later op of the pass waits on the device for that result and adds it. No host call
//! sits between the ops, so the device never waits for a launch, and the host's work overlaps the
//! ops queued between the export and the wait.

use std::ffi::c_void;
use std::sync::atomic::{fence, AtomicU32, Ordering};

use straydog_gpu::ffi;

use crate::gpu::{Backend, Result};
use crate::transfer::Mark;

type P = *const c_void;
type M = *mut c_void;

/// The header's bytes; values start after it.
pub const HEADER: u64 = 256;
const EXPORTED: usize = 0;
const PUBLISHED: usize = 1;
const FAILED: usize = 2;
const CANCEL: usize = 3;

fn check(status: i32) -> Result<()> {
    straydog_gpu::check(status).map(drop).map_err(|error| error.to_string())
}

/// A device's host exchange (freed with the device): the host's side of it.
#[derive(Debug, Clone, Copy)]
pub struct Exchange {
    host: *mut u8,
    bytes: u64,
}

// The exchange is host memory of the device's backend, used on the thread that holds the backend.
unsafe impl Send for Exchange {}

impl Exchange {
    fn word(&self, index: usize) -> &AtomicU32 {
        // SAFETY: the header's words lie in the live allocation, 4-byte aligned; the device writes
        // them only as whole words.
        unsafe { AtomicU32::from_ptr(self.host.cast::<u32>().add(index)) }
    }

    /// The exports the device has made so far.
    pub fn exported(&self) -> u32 {
        self.word(EXPORTED).load(Ordering::Acquire)
    }

    /// Whether a wait for the host's result gave up on the device (that pass's output is void).
    pub fn failed(&self) -> bool {
        self.word(FAILED).load(Ordering::Acquire) != 0
    }

    /// Let the device's waits go on without the host's results (they take zeros), or not.
    pub fn cancel(&self, on: bool) {
        self.word(CANCEL).store(u32::from(on), Ordering::Release);
    }

    /// With the device's stream idle: as many results published as exports made, no failure, no
    /// cancel (the next pass's waits then take exactly its own results).
    pub fn restart(&self) {
        self.word(PUBLISHED).store(self.exported(), Ordering::Release);
        self.word(FAILED).store(0, Ordering::Release);
        self.cancel(false);
    }

    /// Spin until the device's exports reach `count` (the latest export's values are then readable).
    /// Every so often the device's stream is checked: an error fails the wait, and so does the
    /// stream passing `end` (the pass's last op) without the export.
    pub fn wait_export(&self, gpu: &mut Backend, count: u32, end: Mark) -> Result<()> {
        let mut spins = 0u64;
        let started = std::time::Instant::now();
        loop {
            if self.exported().wrapping_sub(count) as i32 >= 0 {
                fence(Ordering::Acquire);
                return Ok(());
            }
            spins += 1;
            if spins.is_multiple_of(4096) {
                if gpu.mark_done(end, false)? && (self.exported().wrapping_sub(count) as i32) < 0 {
                    return Err("a pass ended without the export the host waits for".into());
                }
                if started.elapsed().as_secs() >= 60 {
                    return Err("the device made no export for a minute".into());
                }
            }
            std::hint::spin_loop();
        }
    }

    /// `out.len()` values from `offset` bytes in (an export the host has waited for).
    pub fn read(&self, offset: u64, out: &mut [f32]) -> Result<()> {
        self.range(offset, out.len())?;
        // SAFETY: inside the allocation (checked); the device does not write it until the host
        // publishes (the protocol), and nothing else aliases `out`.
        unsafe { std::ptr::copy_nonoverlapping(self.host.add(offset as usize).cast::<f32>(), out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Write `values` from `offset` bytes in, then publish them: the device's wait for one more
    /// result ends.
    pub fn publish(&self, offset: u64, values: &[f32]) -> Result<()> {
        self.range(offset, values.len())?;
        // SAFETY: as in read; the device reads this range only after the count below.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), self.host.add(offset as usize).cast::<f32>(), values.len()) };
        fence(Ordering::Release);
        self.word(PUBLISHED).fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn range(&self, offset: u64, count: usize) -> Result<()> {
        if offset < HEADER || !offset.is_multiple_of(4) || offset + count as u64 * 4 > self.bytes {
            return Err("a range outside the host exchange".into());
        }
        Ok(())
    }
}

impl Backend {
    /// Whether this device's ops can wait for the host within a pass (see [`Backend::exchange`]).
    pub fn exchange_supported(&self) -> Result<bool> {
        // SAFETY: a live handle.
        let state = unsafe { ffi::sd_exchange_supported(self.handle()) };
        check(state.min(0))?;
        Ok(state == 1)
    }

    /// A host exchange of `bytes` (the header's [`HEADER`] bytes first), zeroed.
    pub fn exchange(&mut self, bytes: u64) -> Result<Exchange> {
        // SAFETY: a live handle.
        let host = unsafe { ffi::sd_exchange(self.handle(), bytes) };
        if host.is_null() {
            return Err(straydog_gpu::last_error().to_string());
        }
        Ok(Exchange { host: host.cast(), bytes })
    }

    /// Copy `parts` (device ranges of so many values, 4-byte words) one after another to `offset`
    /// bytes into `exchange`, then count an export.
    pub fn export(&mut self, parts: &[(P, u64)], exchange: &Exchange, offset: u64) -> Result<()> {
        let sources: Vec<*const f32> = parts.iter().map(|part| part.0.cast()).collect();
        let counts: Vec<u64> = parts.iter().map(|part| part.1).collect();
        // SAFETY: a live handle; the library checks every range against its allocations and the
        // exchange.
        check(unsafe {
            ffi::sd_op_export(self.handle(), sources.as_ptr(), counts.as_ptr(), parts.len() as u64, exchange.host.cast(), offset)
        })
    }

    /// Wait on the device for the host's result to the last export, then out += a (optional) +
    /// the `count` values at `offset` bytes into `exchange`.
    pub fn await_add(&mut self, exchange: &Exchange, offset: u64, a: Option<P>, out: M, count: u64) -> Result<()> {
        // SAFETY: as above.
        check(unsafe {
            ffi::sd_op_await_add(self.handle(), exchange.host.cast(), offset, a.map_or(std::ptr::null(), |a| a.cast()), out.cast(),
                                 count)
        })
    }
}
