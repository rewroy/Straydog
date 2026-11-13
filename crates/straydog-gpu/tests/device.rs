//! The statically linked library runs on a real GPU (cargo test --features hip, or cuda).
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_gpu::{device_count, device_info, Device};

#[test]
fn an_expired_hybrid_wait_rebases_the_next_handshake() {
    assert!(device_count().unwrap() > 0, "this regression test requires a GPU");
    let device = Device::new(0, 64 << 20).unwrap();
    // SAFETY: the development check requires and receives a fresh, exclusive device.
    straydog_gpu::check(unsafe { straydog_gpu::ffi::sd_hybrid_wait_check(device.handle()) }).unwrap();
}

#[test]
fn f32_matvec_on_every_device_matches_the_cpu() {
    let count = device_count().expect("the GPU runtime loads");
    if count == 0 {
        eprintln!("skipped: no GPU");
        return;
    }
    let (rows, cols) = (37, 300);
    let matrix: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 1024.0).collect();
    let input: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 / 8.0 - 0.75).collect();
    let expected: Vec<f32> = (0..rows)
        .map(|r| (0..cols).map(|c| matrix[r * cols + c] as f64 * input[c] as f64).sum::<f64>() as f32)
        .collect();
    for ordinal in 0..count {
        let (name, _free, total) = device_info(ordinal).unwrap();
        assert!(!name.is_empty() && total > 0);
        let device = Device::new(ordinal, 64 << 20).unwrap();
        let bytes: Vec<u8> = matrix.iter().flat_map(|value| value.to_le_bytes()).collect();
        let allocation = device.upload(&bytes).unwrap();
        let mut output = vec![0f32; rows];
        device.matvec(&allocation, 0, rows, cols, &input, &mut output).unwrap();
        for (actual, wanted) in output.iter().zip(&expected) {
            assert!((actual - wanted).abs() <= 1e-4 * wanted.abs().max(1.0), "{actual} vs {wanted}");
        }
        let mut back = vec![0u8; bytes.len()];
        allocation.download(&mut back).unwrap();
        assert_eq!(back, bytes);
        assert!(device.memory().unwrap().0 > 0);
    }
}

#[test]
fn a_budget_beyond_the_device_is_an_error_with_a_reason() {
    if device_count().unwrap_or(0) == 0 {
        return;
    }
    let error = Device::new(0, 1 << 50).err().expect("no device has a PiB");
    assert!(!error.0.is_empty());
}
