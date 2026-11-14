//! The text of a failed GPU call is never NULL, and says what to do when the NVIDIA driver cannot run CUDA 13.
//! Touches no device and no driver (the library's own function, with the runtime's answer given as NULL or as text).
#![cfg(any(feature = "cuda", feature = "hip"))]

use std::ffi::{c_char, CStr};

fn text(code: i32, runtime_null: bool) -> String {
    let mut buffer = [0 as c_char; 512];
    // SAFETY: the buffer and its capacity match; the function writes a NUL-terminated string and touches no device.
    let length = unsafe { straydog_gpu::ffi::sd_test_gpu_error_text(code, i32::from(runtime_null), buffer.as_mut_ptr(), buffer.len()) };
    assert!(length > 0, "code {code}: no text");
    // SAFETY: NUL-terminated within the buffer.
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy().into_owned();
    assert_eq!(text.len(), length as usize);
    text
}

#[cfg(feature = "cuda")]
#[test]
fn a_driver_that_cannot_run_cuda_13_is_named_when_the_runtime_has_no_text() {
    // cudaErrorStubLibrary 34, cudaErrorInsufficientDriver 35, cudaErrorSystemDriverMismatch 803
    for code in [34, 35, 803] {
        let message = text(code, true);
        assert!(message.contains("NVIDIA driver is missing or too old to run CUDA 13"), "{message}");
        assert!(message.contains("version 580 or later"), "{message}");
        // the sentence stands whether or not the runtime had words of its own
        assert_eq!(text(code, false), message);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn no_nvidia_gpu_is_named() {
    assert_eq!(text(100, true), "No NVIDIA GPU was found");
    assert_eq!(text(100, false), "No NVIDIA GPU was found");
}

#[test]
fn an_unknown_code_without_runtime_text_is_a_plain_line_and_other_codes_keep_the_runtime_text() {
    assert_eq!(text(9999, true), "GPU error 9999");
    assert_eq!(text(2, true), "GPU error 2");
    assert_eq!(text(2, false), "the runtime's own words");
}

/// An AMD build has no sentence of its own for these numbers: they are the runtime's words, or the plain line.
#[cfg(feature = "hip")]
#[test]
fn an_amd_build_keeps_the_runtime_text_for_every_code() {
    for code in [34, 35, 100, 803] {
        assert_eq!(text(code, false), "the runtime's own words");
        assert_eq!(text(code, true), format!("GPU error {code}"));
    }
}

#[test]
fn a_bad_buffer_is_refused() {
    // SAFETY: a null buffer is refused before any write.
    assert_eq!(unsafe { straydog_gpu::ffi::sd_test_gpu_error_text(35, 1, std::ptr::null_mut(), 0) }, -1);
}
