//! Host exchanges (gpu/host_exchange.hpp), as a pipelined partially resident decode step uses them:
//! many exports and waits queued at once, the host taking each export's values exactly and
//! publishing a result the device adds exactly; the same recorded as a graph and replayed with new
//! values; waits the host cancels go on with zeros. Skipped without a GPU that can wait so.
//!
//! cargo test --release -p straydog-engine --features hip --test exchange_gpu, under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

use straydog_engine::exchange::{Exchange, HEADER};
use straydog_engine::gpu::Backend;
use straydog_engine::ops::Buffer;
use straydog_engine::transfer::Stream;

/// Layers of a step, and the values a layer exports (routes and weights, then its input).
const LAYERS: usize = 40;
const SMALL: usize = 16;
const WIDTH: usize = 2048;

fn device() -> Option<Backend> {
    let gpu = Backend::new(0, 1 << 30).ok()?;
    if !gpu.exchange_supported().unwrap() {
        eprintln!("skipped: this GPU cannot wait for the host within a pass");
        return None;
    }
    Some(gpu)
}

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count).map(|i| ((i * 31 + salt * 977) % 65_521) as f32 * 0.5 - 1000.0).collect()
}

fn read(gpu: &mut Backend, buffer: &Buffer, count: usize) -> Vec<f32> {
    let mut out = vec![0f32; count];
    gpu.read_buffer(buffer, 0, &mut out).unwrap();
    out
}

struct Step {
    small: Vec<Buffer>,
    input: Vec<Buffer>,
    addend: Vec<Buffer>,
    out: Vec<Buffer>,
}

fn export_at(layer: usize) -> u64 {
    HEADER + (layer * (SMALL + WIDTH) * 4) as u64
}

fn result_at(layer: usize) -> u64 {
    HEADER + (LAYERS * (SMALL + WIDTH) * 4 + layer * WIDTH * 4) as u64
}

/// A step's buffers: each layer's exported values, its addend and its output.
fn buffers(gpu: &mut Backend) -> Step {
    let mut make = |count: usize| (0..LAYERS).map(|_| gpu.buffer((count * 4) as u64).unwrap()).collect::<Vec<_>>();
    Step { small: make(SMALL), input: make(WIDTH), addend: make(WIDTH), out: make(WIDTH) }
}

/// Write the step's values for `round` (each layer's distinct).
fn fill(gpu: &mut Backend, step: &Step, round: usize) {
    for layer in 0..LAYERS {
        gpu.write_buffer(&step.small[layer], 0, &values(SMALL, round * 1000 + layer)).unwrap();
        gpu.write_buffer(&step.input[layer], 0, &values(WIDTH, round * 1000 + layer + 500)).unwrap();
        gpu.write_buffer(&step.addend[layer], 0, &values(WIDTH, round * 1000 + layer + 700)).unwrap();
        gpu.write_buffer(&step.out[layer], 0, &values(WIDTH, round * 1000 + layer + 900)).unwrap();
    }
}

/// Queue the step: each layer exports, then waits for the host's result and adds it.
fn queue(gpu: &mut Backend, exchange: &Exchange, step: &Step) {
    for layer in 0..LAYERS {
        gpu.export(&[(step.small[layer].base(), SMALL as u64), (step.input[layer].base(), WIDTH as u64)], exchange, export_at(layer))
            .unwrap();
        gpu.await_add(exchange, result_at(layer), Some(step.addend[layer].base()), step.out[layer].base(), WIDTH as u64).unwrap();
    }
}

/// Serve the queued step (`start` exports before it): each export's values as written, each
/// result published; then every output is its old value plus the addend plus the result, exactly.
fn serve(gpu: &mut Backend, exchange: &Exchange, step: &Step, round: usize, start: u32) {
    let end = gpu.mark(Stream::Compute).unwrap();
    for layer in 0..LAYERS {
        exchange.wait_export(gpu, start + layer as u32 + 1, end).unwrap();
        let mut got = vec![0f32; SMALL + WIDTH];
        exchange.read(export_at(layer), &mut got).unwrap();
        assert_eq!(&got[..SMALL], &values(SMALL, round * 1000 + layer)[..], "layer {layer}'s small export");
        assert_eq!(&got[SMALL..], &values(WIDTH, round * 1000 + layer + 500)[..], "layer {layer}'s input export");
        exchange.publish(result_at(layer), &values(WIDTH, round * 1000 + layer + 300)).unwrap();
    }
    for layer in 0..LAYERS {
        let out = read(gpu, &step.out[layer], WIDTH);
        let (old, addend, result) = (values(WIDTH, round * 1000 + layer + 900), values(WIDTH, round * 1000 + layer + 700),
                                     values(WIDTH, round * 1000 + layer + 300));
        for i in 0..WIDTH {
            let expected = old[i] + (addend[i] + result[i]);
            assert_eq!(out[i].to_bits(), expected.to_bits(), "layer {layer} value {i}");
        }
    }
    assert!(!exchange.failed());
}

#[test]
fn exchange_steps() {
    let Some(mut gpu) = device() else { return };
    let exchange = gpu.exchange(result_at(LAYERS)).unwrap();
    let step = buffers(&mut gpu);
    // Queued op by op, twice.
    for round in 0..2 {
        fill(&mut gpu, &step, round);
        let start = exchange.exported();
        queue(&mut gpu, &exchange, &step);
        serve(&mut gpu, &exchange, &step, round, start);
    }
    // Recorded once, replayed with new values.
    gpu.graph_begin().unwrap();
    queue(&mut gpu, &exchange, &step);
    gpu.graph_end(7, true).unwrap();
    for round in 2..5 {
        fill(&mut gpu, &step, round);
        let start = exchange.exported();
        assert!(gpu.graph_run(7).unwrap());
        serve(&mut gpu, &exchange, &step, round, start);
    }
    // Cancelled: the waits go on, adding zeros; then the counts start over.
    fill(&mut gpu, &step, 5);
    queue(&mut gpu, &exchange, &step);
    exchange.cancel(true);
    for layer in 0..LAYERS {
        let out = read(&mut gpu, &step.out[layer], WIDTH);
        let (old, addend) = (values(WIDTH, 5 * 1000 + layer + 900), values(WIDTH, 5 * 1000 + layer + 700));
        for i in 0..WIDTH {
            assert_eq!(out[i].to_bits(), (old[i] + (addend[i] + 0.0)).to_bits(), "cancelled layer {layer} value {i}");
        }
    }
    exchange.restart();
    fill(&mut gpu, &step, 6);
    let start = exchange.exported();
    assert!(gpu.graph_run(7).unwrap());
    serve(&mut gpu, &exchange, &step, 6, start);
}
