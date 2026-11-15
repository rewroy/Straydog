//! DeepSeek-V4.1's prompt passes of 8 to 31 rows take FP16 operands as longer ones do (`Settings::half_gemm`): a text read in passes
//! of 16 rows is no further from F32 than the same text read in passes of 64 rows, which took FP16 before the limit came down from
//! 32 rows. Measured where both have read the same tokens (every 64): the next token's distribution against F32's (KL), and its
//! choice.
//!
//! cargo test -p straydog-engine --features hip --test deepseek41_short_passes -- --nocapture, with STRAYDOG_DS41 (the GSQ-RCO
//! IQ2_XS bundle directory, its verity trees in .straydog there); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::arch::deepseek41::serve::place;
use straydog_engine::arch::served::Settings;
use straydog_engine::group::SplitMode;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&value| (value as f64 - top).exp()).sum();
    logits.iter().map(|&value| value as f64 - top - sum.ln()).collect()
}

/// KL(p || q) of two rows of logits.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (p, q) = (log_softmax(p), log_softmax(q));
    p.iter().zip(&q).map(|(&a, &b)| a.exp() * (a - b)).sum()
}

fn best(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

const LONG: usize = 64;
const SHORT: usize = 16;
const POINTS: usize = 12;

#[test]
fn passes_of_sixteen_rows_in_fp16_are_as_near_f32_as_passes_of_sixty_four() {
    let Some(root) = common::model_env("STRAYDOG_DS41", "the DeepSeek-V4.1 GSQ-RCO IQ2_XS bundle directory") else { return };
    let root = PathBuf::from(root);
    let document: String = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/DESIGN.md")).unwrap()
        .chars().skip(2000).take(9000).collect();
    // One run a process (a model's host tier is page-locked whole, and two do not fit): STRAYDOG_SHORT_PASSES is "f32" or "fp16",
    // and STRAYDOG_SHORT_PASSES_DIR the folder the run's logits are kept in. The comparison is made by the run that finds both.
    let (Ok(mode), Ok(dir)) = (std::env::var("STRAYDOG_SHORT_PASSES"), std::env::var("STRAYDOG_SHORT_PASSES_DIR")) else {
        eprintln!("skipped: STRAYDOG_SHORT_PASSES (f32 or fp16) and STRAYDOG_SHORT_PASSES_DIR are not set");
        return;
    };
    let half = match mode.as_str() {
        "f32" => false,
        "fp16" => true,
        other => panic!("STRAYDOG_SHORT_PASSES is {other}, not f32 or fp16"),
    };
    let dir = PathBuf::from(dir);
    // The logits after every 64 tokens of the text, read in passes of 64 rows and of 16, in F32 or with FP16 operands.
    {
        let settings = Settings {
            devices: vec![0, 1], split_mode: SplitMode::Experts, vram_bytes: vec![28 << 30; 2], context: 16384, batch: 2048,
            host_expert_bytes: None, host_memory_bytes: straydog_engine::host::host_memory_bytes(),
            row_cache_bytes: straydog_engine::rows::DEFAULT_CACHE_BYTES, half_gemm: half,
        };
        let mut model = place(&root, common::deepseek41_bundle(), &root.join(".straydog"), &settings, &|_, _, _| {}).unwrap();
        model.prompt_logits = Some(Vec::new());
        let tokens = Text::from_gguf(model.header()).unwrap().encode(&document).unwrap();
        assert!(tokens.len() > LONG * POINTS, "{} tokens", tokens.len());
        for rows in [LONG, SHORT] {
            model.forget();
            let mut bytes: Vec<u8> = Vec::new();
            for end in (rows..=LONG * POINTS).step_by(rows) {
                // (one token asked for: it is chosen and not fed, so the next prompt goes on from this one)
                model.generate(&tokens[..end], 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |_| {}).unwrap();
                assert!(end == rows || model.last_metrics["resumed_from"] == "live", "{:?} at {end}", model.last_metrics["resumed_from"]);
                assert_eq!(model.last_metrics["prefill_tokens"].as_u64(), Some(rows as u64));
                if end % LONG == 0 {
                    bytes.extend(model.prompt_logits.as_ref().unwrap().iter().flat_map(|value| value.to_le_bytes()));
                }
                if end % (4 * LONG) == 0 {
                    eprintln!("{mode} passes of {rows} rows to {end}: the last in {:.0} ms", 1e3 * model.last_metrics["prefill_seconds"].as_f64().unwrap_or(0.0));
                }
            }
            std::fs::write(dir.join(format!("{mode}-{rows}.bin")), bytes).unwrap();
        }
    }
    let read = |mode: &str, rows: usize| -> Option<Vec<Vec<f32>>> {
        let bytes = std::fs::read(dir.join(format!("{mode}-{rows}.bin"))).ok()?;
        let values: Vec<f32> = bytes.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
        Some(values.chunks_exact(values.len() / POINTS).map(<[f32]>::to_vec).collect())
    };
    let (Some(exact_long), Some(exact_short), Some(half_long), Some(half_short)) = (read("f32", LONG), read("f32", SHORT), read("fp16", LONG), read("fp16", SHORT)) else {
        eprintln!("the other run's logits are not there yet: nothing compared");
        return;
    };
    let (exact, half) = ([exact_long, exact_short], [half_long, half_short]);
    let mut report = Vec::new();
    for (name, index) in [("64 rows", 0), ("16 rows", 1)] {
        let kls: Vec<f64> = exact[index].iter().zip(&half[index]).map(|(exact, half)| kl(exact, half)).collect();
        let same = exact[index].iter().zip(&half[index]).filter(|(exact, half)| best(exact) == best(half)).count();
        // (how far below F32's choice the FP16 run's choice lies under F32, at its worst)
        let gap = exact[index].iter().zip(&half[index]).map(|(exact, half)| exact[best(exact)] - exact[best(half)]).fold(0f32, f32::max);
        let (mean, worst) = (kls.iter().sum::<f64>() / kls.len() as f64, kls.iter().copied().fold(0f64, f64::max));
        eprintln!("passes of {name}: KL from F32 mean {mean:.3e}, worst {worst:.3e}; the same next token at {same} of {}; the choice at most {gap:.3} below F32's",
                  kls.len());
        report.push((mean, worst, same, gap));
    }
    // (what the split into passes alone does, in F32: the reference for both)
    let split: Vec<f64> = exact[0].iter().zip(&exact[1]).map(|(long, short)| kl(long, short)).collect();
    eprintln!("F32, passes of 64 rows against 16: KL mean {:.3e}, worst {:.3e}", split.iter().sum::<f64>() / split.len() as f64,
              split.iter().copied().fold(0f64, f64::max));
    let ((long_mean, long_worst, _, _), (short_mean, short_worst, _, short_gap)) = (report[0], report[1]);
    assert!(short_mean <= 1.5 * long_mean + 1e-4, "passes of 16 rows: mean KL {short_mean:.3e} from F32, passes of 64 rows {long_mean:.3e}");
    assert!(short_worst <= 2.0 * long_worst + 1e-3, "passes of 16 rows: worst KL {short_worst:.3e} from F32, passes of 64 rows {long_worst:.3e}");
    assert!(short_gap <= 0.25, "passes of 16 rows choose a token {short_gap:.3} below F32's");
}
