//! Dies of an access violation after setting up Straydog's crash line, to see the line on Windows:
//! `cargo run --release --example crash_line -p straydog --features cuda` prints "Straydog crashed: access violation (0xC0000005)
//! writing 0x10 at crash_line.exe+0x..." on stderr and exits with 0xC0000005, as before.
fn main() {
    straydog::crash::report_unhandled();
    let target = std::hint::black_box(0x10usize) as *mut u32;
    // SAFETY: none: the write is the crash this example is for.
    unsafe { target.write_volatile(1) };
    println!("not reached");
}
