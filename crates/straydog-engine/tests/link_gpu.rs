//! Two GPUs' linked sums (gpu/device_link.hpp), as tensor-parallel Qwen3.5 MoE adds its parts:
//! many sums queued on both devices at once, each with other parts, every result exact and equal
//! on both; the same through recorded graphs replayed with new parts; a sum whose peer never comes
//! fails instead of hanging, and a reset starts the counts over. Skipped without two GPUs.
//!
//! cargo test --release -p straydog-engine --features hip --test link_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;

/// Width of a sum in floats (a decode step's stream; a prefill chunk's is 64 times this).
const WIDTH: usize = 2048;

fn pair() -> Option<(Backend, Backend)> {
    let (Ok(mut first), Ok(mut second)) = (Backend::new(0, 1 << 30), Backend::new(1, 1 << 30)) else {
        eprintln!("skipped: linked sums need two GPUs");
        return None;
    };
    Backend::link_sums(&mut first, &mut second, (64 * WIDTH) as u64).unwrap();
    Some((first, second))
}

/// Parts of `sums` sums of `count` floats: distinct per device, sum and position, exact in F32.
fn parts(device: usize, sums: usize, count: usize, salt: usize) -> Vec<f32> {
    (0..sums * count).map(|i| ((i * 7 + salt * 13 + device * 101) % 100_003) as f32 * if device == 0 { 1.0 } else { -0.25 }).collect()
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut values).unwrap();
    values
}

/// Every result is this device's part plus the other's, the same bits on both devices.
fn check(a: &[f32], b: &[f32], first: &[f32], second: &[f32], base: &[f32]) {
    for i in 0..a.len() {
        let expected = base[i] + (first[i] + second[i]);
        assert_eq!(a[i].to_bits(), expected.to_bits(), "device 0 value {i}: {} against {expected}", a[i]);
        assert_eq!(b[i].to_bits(), expected.to_bits(), "device 1 value {i}: {} against {expected}", b[i]);
    }
}

#[test]
fn linked_sums() {
    let Some((mut a, mut b)) = pair() else { return };
    // Many sums queued back to back without waiting (each reads an inbox slot the one two before
    // it read), of a decode step's width and of a prefill chunk's, with and without accumulating.
    for (count, sums) in [(WIDTH, 600), (64 * WIDTH, 40)] {
        for accumulate in [false, true] {
            let (first, second) = (parts(0, sums, count, count), parts(1, sums, count, count));
            let base: Vec<f32> = (0..sums * count).map(|i| if accumulate { (i % 977) as f32 } else { 0.0 }).collect();
            let bytes = (sums * count * 4) as u64;
            let (part_a, part_b, out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap(), a.buffer(bytes).unwrap(),
                                                  b.buffer(bytes).unwrap());
            a.write_buffer(&part_a, 0, &first).unwrap();
            b.write_buffer(&part_b, 0, &second).unwrap();
            a.write_buffer(&out_a, 0, &base).unwrap();
            b.write_buffer(&out_b, 0, &base).unwrap();
            for sum in 0..sums {
                let at = sum * count;
                a.link_sum(part_a.at(at), count as u64, out_a.at(at), accumulate).unwrap();
                b.link_sum(part_b.at(at), count as u64, out_b.at(at), accumulate).unwrap();
            }
            let (result_a, result_b) = (read(&mut a, &out_a, sums * count), read(&mut b, &out_b, sums * count));
            assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
            check(&result_a, &result_b, &first, &second, &base);
        }
    }
    // Recorded graphs of eight sums each, replayed with new parts every time: the counts carry on
    // in device memory.
    let sums = 8;
    let bytes = (sums * WIDTH * 4) as u64;
    let (part_a, part_b, out_a, out_b) = (a.buffer(bytes).unwrap(), b.buffer(bytes).unwrap(), a.buffer(bytes).unwrap(),
                                          b.buffer(bytes).unwrap());
    for (gpu, part, out) in [(&mut a, &part_a, &out_a), (&mut b, &part_b, &out_b)] {
        gpu.graph_begin().unwrap();
        for sum in 0..sums {
            gpu.link_sum(part.at(sum * WIDTH), WIDTH as u64, out.at(sum * WIDTH), false).unwrap();
        }
        gpu.graph_end(7, true).unwrap();
    }
    let zero = vec![0f32; sums * WIDTH];
    for replay in 0..200 {
        let (first, second) = (parts(0, sums, WIDTH, replay), parts(1, sums, WIDTH, replay));
        a.write_buffer(&part_a, 0, &first).unwrap();
        b.write_buffer(&part_b, 0, &second).unwrap();
        assert!(a.graph_run(7).unwrap() && b.graph_run(7).unwrap());
        let (result_a, result_b) = (read(&mut a, &out_a, sums * WIDTH), read(&mut b, &out_b, sums * WIDTH));
        check(&result_a, &result_b, &first, &second, &zero);
    }
    assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
    // A sum the other device never joins gives up after the await timeout; after a reset both
    // count from zero again and sum as before.
    let started = std::time::Instant::now();
    a.link_sum(part_a.base(), WIDTH as u64, out_a.base(), false).unwrap();
    assert!(a.link_failed().unwrap(), "a lone sum must fail");
    eprintln!("a lone sum failed after {:.1} s", started.elapsed().as_secs_f64());
    a.link_reset().unwrap();
    assert!(!a.link_failed().unwrap() && !b.link_failed().unwrap());
    let (first, second) = (parts(0, 1, WIDTH, 5), parts(1, 1, WIDTH, 5));
    a.write_buffer(&part_a, 0, &first).unwrap();
    b.write_buffer(&part_b, 0, &second).unwrap();
    a.link_sum(part_a.base(), WIDTH as u64, out_a.base(), false).unwrap();
    b.link_sum(part_b.base(), WIDTH as u64, out_b.base(), false).unwrap();
    let (result_a, result_b) = (read(&mut a, &out_a, WIDTH), read(&mut b, &out_b, WIDTH));
    check(&result_a, &result_b, &first, &second, &vec![0f32; WIDTH]);
}
