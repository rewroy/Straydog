//! CPU-only costs of the production sampling API, using synthetic logits.
//! cargo run --release -p straydog-engine --example sampling_cost -- [samples=101]
//! No GPU features/model; outputs CSV. Windows CPU model is recorded by the caller.
use std::{hint::black_box, time::Instant};
use straydog_engine::{
    model::Rng,
    sampling::{self, Generated, Sampling},
};

const VOCAB: usize = 129_280;
const SEED: u64 = 0x104f_a773_e4f2_076b;

fn measure<T>(samples: usize, mut f: impl FnMut() -> T) -> (f64, f64, usize) {
    for _ in 0..8 {
        black_box(f());
    }
    let start = Instant::now();
    black_box(f());
    let iterations = (100.0 / start.elapsed().as_secs_f64().max(1e-9) / 1e6)
        .ceil()
        .clamp(1.0, 128.0) as usize;
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(f());
        }
        times.push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
    }
    times.sort_by(f64::total_cmp);
    (
        times[samples / 2],
        times[(samples * 9).div_ceil(10).saturating_sub(1)],
        iterations,
    )
}
fn print<T>(
    shape: &str,
    profile: &str,
    operation: &str,
    samples: usize,
    units: usize,
    f: impl FnMut() -> T,
) {
    let (median, p90, iterations) = measure(samples, f);
    println!("{shape},{profile},{operation},{median:.6},{p90:.6},{samples},{iterations},{units}");
}
fn generated() -> Generated {
    let mut seen = Generated::new(VOCAB);
    // 512 distinct previously generated tokens, set up outside every timed region.
    for i in 0..512 {
        seen.insert((i * 251 % VOCAB) as u32);
    }
    seen
}
fn main() {
    let samples = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>().expect("positive sample count"))
        .unwrap_or(101);
    assert!(samples >= 11);
    println!("shape,profile,operation,median_us,p90_us,samples,iterations,units_per_call");
    let profiles = [
        ("greedy", Sampling::GREEDY),
        (
            "greedy_presence",
            Sampling {
                presence_penalty: 1.5,
                ..Sampling::GREEDY
            },
        ),
        ("recommended_thinking", Sampling::recommended(true)),
        ("recommended_non_thinking", Sampling::recommended(false)),
        (
            "dense_default",
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
        ),
        (
            "dense_temperature",
            Sampling {
                temperature: 0.7,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
        ),
        (
            "top_k64",
            Sampling {
                temperature: 0.7,
                top_k: 64,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
        ),
        (
            "top_k128",
            Sampling {
                temperature: 0.7,
                top_k: 128,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
        ),
        (
            "top_p_only",
            Sampling {
                temperature: 0.7,
                top_k: 0,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
        ),
        (
            "min_p_presence",
            Sampling {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                min_p: 0.05,
                presence_penalty: 1.5,
            },
        ),
        (
            "negative_presence",
            Sampling {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
                presence_penalty: -1.0,
            },
        ),
    ];
    for (shape, scale, ties) in [
        ("broad", 8.0, false),
        ("peaked", 8.0, false),
        ("ties", 0.0, true),
    ] {
        let mut source = Rng::new(SEED);
        let mut logits: Vec<f32> = (0..VOCAB)
            .map(|_| ((source.random() - 0.5) * scale) as f32)
            .collect();
        if shape == "peaked" {
            logits[42] = 20.0;
            logits[67] = 18.0;
            logits[99] = 16.0;
        }
        if ties {
            logits.fill(0.0);
        }
        for (name, config) in &profiles {
            config.check().unwrap();
            let seen = generated();
            let mut rng = Rng::new(SEED);
            print(shape, name, "choose", samples, 1, || {
                sampling::choose(
                    black_box(&logits),
                    black_box(config),
                    black_box(&seen),
                    &mut rng,
                )
            });
            if config.temperature > 0.0 {
                print(shape, name, "distribution", samples, 1, || {
                    sampling::distribution(black_box(&logits), black_box(config), black_box(&seen))
                });
            }
            // Rejection path: a deterministic proposal with negligible target probability.
            let mut reject = Vec::with_capacity(VOCAB * 2);
            reject.extend(&logits);
            reject.extend(&logits);
            reject[0] = -1000.0;
            let mut reject_seen = generated();
            print(shape, name, "accept_reject_first", samples, 1, || {
                sampling::accept(
                    black_box(&[0]),
                    black_box(&reject),
                    VOCAB,
                    black_box(config),
                    &mut reject_seen,
                    &mut rng,
                )
            });
            // Acceptance path: four certain point proposals, then one final draw.
            let drafts = [1u32, 2, 3, 4];
            let mut accept = Vec::with_capacity(VOCAB * 5);
            for row in 0..5 {
                accept.extend(&logits);
                if row < 4 {
                    accept[row * VOCAB + drafts[row] as usize] = 1000.0;
                }
            }
            let mut accept_seen = generated();
            let accepted = sampling::accept(
                &drafts,
                &accept,
                VOCAB,
                config,
                &mut accept_seen,
                &mut Rng::new(SEED),
            );
            assert_eq!(accepted.0, 4, "accept path setup");
            print(shape, name, "accept_four_plus_next", samples, 5, || {
                sampling::accept(
                    black_box(&drafts),
                    black_box(&accept),
                    VOCAB,
                    black_box(config),
                    &mut accept_seen,
                    &mut rng,
                )
            });
        }
    }
    let source: Vec<f32> = (0..VOCAB).map(|i| i as f32).collect();
    print("copy", "host", "vec_zero_allocate_drop", samples, 1, || {
        vec![0.0f32; black_box(VOCAB)]
    });
    print(
        "copy",
        "host",
        "vec_capacity_allocate_drop",
        samples,
        1,
        || Vec::<f32>::with_capacity(black_box(VOCAB)),
    );
    print(
        "copy",
        "host",
        "vec_clone_allocate_copy_drop",
        samples,
        1,
        || black_box(&source).clone(),
    );
    let mut reuse = vec![0.0; VOCAB];
    print("copy", "host", "reuse_copy", samples, 1, || {
        black_box(reuse.as_mut_slice()).copy_from_slice(black_box(&source));
        black_box(&reuse);
    });
    for devices in [1usize, 2] {
        let part = VOCAB / devices;
        print(
            "copy",
            if devices == 1 { "tp1" } else { "tp2" },
            "head_host_reassembly_allocate_copy",
            samples,
            1,
            || {
                let mut logits = vec![0.0f32; VOCAB];
                let mut rows = vec![0.0f32; part + 1];
                for device in 0..devices {
                    // CPU source substitutes for read_buffer: no claim about D2H latency.
                    rows[..part]
                        .copy_from_slice(black_box(&source[device * part..(device + 1) * part]));
                    logits[device * part..(device + 1) * part].copy_from_slice(&rows[..part]);
                }
                logits
            },
        );
    }
}
