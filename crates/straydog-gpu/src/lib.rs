//! The Straydog GPU library: its C ABI ([`ffi`]), generated from gpu/straydog_gpu.h, and
//! the safe pieces every caller shares (errors, devices, allocations).

pub mod ffi;

use std::ffi::{c_int, c_void, CStr};
use std::fmt;
use std::ptr::NonNull;

/// A failed call, with the library's description of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Straydog GPU: {}", self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The calling thread's last failure.
pub fn last_error() -> Error {
    // SAFETY: sd_error returns a thread-local, NUL-terminated string that lives until the next call.
    let message = unsafe { CStr::from_ptr(ffi::sd_error()) };
    Error(message.to_string_lossy().into_owned())
}

/// A status or query result: negative means failure.
pub fn check(status: c_int) -> Result<c_int> {
    if status < 0 {
        Err(last_error())
    } else {
        Ok(status)
    }
}

/// A pointer result: null means failure.
pub fn check_pointer(pointer: *mut c_void) -> Result<NonNull<c_void>> {
    NonNull::new(pointer).ok_or_else(last_error)
}

pub fn device_count() -> Result<usize> {
    // SAFETY: no arguments.
    check(unsafe { ffi::sd_device_count() }).map(|count| count as usize)
}

/// Load the first Qwen prefill GEMM's GPU code before a prompt can need it.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub fn prime_qwen_prefill_gemm(ordinal: usize, tokens: usize) -> Result<()> {
    // SAFETY: the GPU library selects the device and owns the temporary buffers and handle.
    check(unsafe { ffi::sd_prime_qwen_prefill_gemm(ordinal as c_int, tokens as u64) }).map(drop)
}

/// A device's name, free and total memory in bytes.
pub fn device_info(ordinal: usize) -> Result<(String, u64, u64)> {
    let mut name = [0 as std::ffi::c_char; 256];
    let (mut free, mut total) = (0u64, 0u64);
    // SAFETY: the buffer and its capacity match; the library writes a NUL-terminated name.
    check(unsafe { ffi::sd_device_info(ordinal as c_int, name.as_mut_ptr(), name.len(), &mut free, &mut total) })?;
    // SAFETY: NUL-terminated within the buffer.
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
    Ok((name, free, total))
}

/// Whether this CUDA GPU advertises support for registering host memory as read-only.
#[cfg(feature = "cuda")]
pub fn host_register_read_only_supported(ordinal: usize) -> Result<bool> {
    // SAFETY: the query reads a device attribute and returns 0, 1, or a negative error.
    check(unsafe { ffi::sd_host_register_read_only_supported(ordinal as c_int) }).map(|value| value != 0)
}

/// A device's PCI address as `domain:bus:device.0` in lower-case hex, the form Linux's sysfs
/// and NVML use.
pub fn pci_address(ordinal: usize) -> Result<String> {
    let (mut domain, mut bus, mut device): (c_int, c_int, c_int) = (0, 0, 0);
    // SAFETY: three out-parameters.
    check(unsafe { ffi::sd_device_pci(ordinal as c_int, &mut domain, &mut bus, &mut device) })?;
    Ok(format!("{domain:04x}:{bus:02x}:{device:02x}.0"))
}

/// One GPU with a VRAM budget the library enforces.
pub struct Device {
    handle: NonNull<c_void>,
}

// The library serializes each device's work on its own stream; a handle may move between threads.
unsafe impl Send for Device {}

impl Device {
    pub fn new(ordinal: usize, budget_bytes: u64) -> Result<Device> {
        // SAFETY: plain values; failure is a null handle.
        let handle = check_pointer(unsafe { ffi::sd_create(ordinal as c_int, budget_bytes) })?;
        Ok(Device { handle })
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    /// Bytes allocated and the enforced budget.
    pub fn memory(&self) -> Result<(u64, u64)> {
        let (mut used, mut budget) = (0u64, 0u64);
        // SAFETY: a live handle and two out-parameters.
        check(unsafe { ffi::sd_memory(self.handle(), &mut used, &mut budget) })?;
        Ok((used, budget))
    }

    /// Copy host bytes into a new device allocation.
    pub fn upload(&self, data: &[u8]) -> Result<Allocation<'_>> {
        // SAFETY: the slice is valid for its length; the library copies it before returning.
        let pointer = check_pointer(unsafe {
            ffi::sd_upload(self.handle(), data.as_ptr().cast(), data.len() as u64)
        })?;
        Ok(Allocation { device: self, pointer, bytes: data.len() as u64 })
    }

    /// `rows` x `cols` matrix of format `kind` (GGML type id) times `input`, into `output`.
    pub fn matvec(&self, matrix: &Allocation<'_>, kind: i32, rows: usize, cols: usize, input: &[f32],
                  output: &mut [f32]) -> Result<()> {
        assert!(input.len() == cols && output.len() == rows, "matvec shapes");
        // SAFETY: shapes checked; the matrix belongs to this device.
        check(unsafe {
            ffi::sd_matvec(self.handle(), matrix.pointer(), kind, rows as u64, cols as u64, input.as_ptr(),
                           output.as_mut_ptr())
        })?;
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the handle came from sd_create and is destroyed once.
        unsafe { ffi::sd_destroy(self.handle()) }
    }
}

/// Device memory, released when dropped.
pub struct Allocation<'device> {
    device: &'device Device,
    pointer: NonNull<c_void>,
    bytes: u64,
}

impl Allocation<'_> {
    pub fn pointer(&self) -> *mut c_void {
        self.pointer.as_ptr()
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Copy the allocation's bytes back to the host.
    pub fn download(&self, output: &mut [u8]) -> Result<()> {
        assert!(output.len() as u64 <= self.bytes, "download larger than the allocation");
        // SAFETY: the output buffer holds the requested bytes.
        check(unsafe {
            ffi::sd_download(self.device.handle(), self.pointer(), output.as_mut_ptr().cast(), output.len() as u64)
        })?;
        Ok(())
    }
}

impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        // SAFETY: allocated on this device and released once; a failure here leaves nothing to undo.
        unsafe { ffi::sd_release(self.device.handle(), self.pointer()) };
    }
}
