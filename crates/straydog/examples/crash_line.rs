//! Dies of an access violation after setting up Straydog's crash line, to see the line on Windows:
//! `cargo run --release --example crash_line -p straydog --features cuda` prints "Straydog crashed: access violation (0xC0000005)
//! reading 0x0 at crash_line.exe+0x..., called from crash_line.exe+0x... < ..." on stderr and exits with 0xC0000005, as before.
//!
//! It dies as a release did: in the C runtime's `strlen`, given a null pointer, three calls down. The place alone is the same for
//! every caller of `strlen`; the callers the line names are `length_of`, `middle` and `outer` here.
use std::ffi::{c_char, CStr};

#[inline(never)]
fn length_of(text: *const c_char) -> usize {
    // SAFETY: none: reading through the null pointer is the crash this example is for.
    std::hint::black_box(unsafe { CStr::from_ptr(text) }.to_bytes().len())
}

#[inline(never)]
fn middle(text: *const c_char) -> usize {
    std::hint::black_box(length_of(text) + 1)
}

#[inline(never)]
fn outer(text: *const c_char) -> usize {
    std::hint::black_box(middle(text) + 1)
}

fn main() {
    straydog::crash::report_unhandled();
    let length = outer(std::hint::black_box(std::ptr::null()));
    println!("not reached: {length}");
}
