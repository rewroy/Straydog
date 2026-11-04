//! Host-side prefill costs on a real bundle: token rows, PLE rows, tiling (a diagnostic).

use std::path::PathBuf;
use std::time::Instant;

use straydog_engine::weights::{ngram_rows, Weights};

#[global_allocator]
static ALLOCATOR: straydog_engine::alloc::HugePages = straydog_engine::alloc::HugePages;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights = Weights::open(&PathBuf::from(&args[0]), straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, &args[1]).unwrap(), None, 0, &|_, _, _| {}).unwrap();
    weights.authenticate(&|_, _, _| {}).unwrap();
    let tokens: Vec<i64> = (0..4096).map(|i| (i * 7919 + 13) % 248000).collect();
    for _ in 0..3 {
        let started = Instant::now();
        let rows = weights.rows("token_embd.weight", &tokens).unwrap();
        let token_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut tiled = Vec::with_capacity(rows.len() * 4);
        for row in rows.chunks_exact(2560) {
            for _ in 0..4 {
                tiled.extend_from_slice(row);
            }
        }
        let tile_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let indices = ngram_rows(&[], &tokens, &weights.integer_arrays).unwrap();
        let hash_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let ple = weights.rows("per_layer_token_embd.weight", &indices).unwrap();
        let ple_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let copy = tiled.clone();
        let copy_seconds = started.elapsed().as_secs_f64();
        println!("4096 tokens: token rows {token_seconds:.3} s, tile {tile_seconds:.3} s, hashes {hash_seconds:.3} s, \
                  PLE rows {ple_seconds:.3} s, 160 MB copy {copy_seconds:.3} s ({} {} {})", rows.len(), ple.len(), copy.len());
    }
}
