//! A partially resident prompt batch that fails while its missing experts stream leaves nothing behind: the test hook of the GPU library fails the
//! n-th streamed chunk after its GEMMs are queued and its experts' table entries point at the chunk's buffer (`hybrid_stream_missing`), the request
//! returns the error, and the request after it is the one of a model that never failed (the table entries emptied and both streams drained on the way
//! out, `PrestageUse` and the settle of `hybrid_stream_missing`).
//!
//! cargo test --release --features cuda -p straydog-engine --test stream_fault_gpu, with STRAYDOG_MODEL naming the bundle directory and
//! STRAYDOG_QUANT its quantization (UD-IQ1_S where unset); under the GPU lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The decode passes a run compares.
const PASSES: usize = 12;
/// As in `behind_gpu.rs`: the least relative difference of logits that two runs of one configuration are allowed (which experts the slots hold at a
/// moment depends on timing, and where a route goes moves a pass's logits by that much).
const NOISE_FLOOR: f64 = 0.15;
/// The streamed chunk that fails: the layers' chunks are counted from the first batch's first layer, and a few of them go through before it.
const FAILING_CHUNK: u64 = 7;

/// The model at a 6 GiB card's budget and a prompt of a few hundred tokens, in batches of 64 rows.
fn open(root: &std::ffi::OsStr) -> (Model, Vec<u32>) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let quantization = std::env::var("STRAYDOG_QUANT").unwrap_or_else(|_| "UD-IQ1_S".into());
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, &quantization).unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, None, straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined a screw press with \
                movable metal type, an oil-based ink and a hand mould for casting letters. Within fifty years presses ran in more than \
                two hundred European towns, and the price of a book fell to a fraction of a manuscript's. Scholars could compare \
                identical copies across cities; errors were corrected in later editions instead of multiplying by hand. Printers \
                standardized spelling and punctuation, and vernacular languages gained a written form that crossed regional dialects. \
                Pamphlets carried the arguments of the Reformation faster than any authority could answer them. Merchants printed \
                price lists and tables of exchange; astronomers printed star catalogues whose numbers others could check. In the \
                eighteenth century, newspapers and journals made a public that read the same news on the same day. Steam presses in \
                the nineteenth century, rotary presses and the linotype machine brought cheap daily papers to millions of readers, \
                and lithography and photography brought pictures to the page. Explain, in a few sentences, which of these changes \
                mattered most for science, and why.";
    let prompt = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap().encode(text).unwrap();
    let settings = Settings { context: 8192, state_budget_bytes: 4 << 30, prefill_batch_size: 64, speculative: 0, cpu_threads: 0, row_cache_bytes: None };
    let model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    (model, prompt)
}

struct Run {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

/// A request from the start of the prompt: the model is reset, and the passes' logits kept.
fn request(model: &mut Model, prompt: &[u32]) -> Result<Run, String> {
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    let mut tokens = Vec::new();
    let result = model.generate(prompt, PASSES + 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token));
    let logits = model.logits_capture.take().unwrap_or_default();
    result.map(|_| Run { tokens, logits }).map_err(|error| error.to_string())
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// The passes of two runs up to where their tokens part: how many were compared, and the worst relative difference of their logits.
fn compare(one: &Run, other: &Run) -> (usize, f64) {
    let (mut compared, mut worst) = (0, 0f64);
    for pass in 0..PASSES {
        if one.tokens.get(..=pass) != other.tokens.get(..=pass) {
            break;
        }
        worst = worst.max(relative_difference(&one.logits[pass + 1], &other.logits[pass + 1]));
        compared += 1;
    }
    (compared, worst)
}

/// A streamed chunk fails in the middle of a prompt: the request returns the error, and the next request, of the same prompt, decodes as two runs
/// of a model that never failed do (the same tokens, the logits within what two such runs differ by).
#[test]
fn a_failed_streamed_chunk_leaves_the_next_request_unharmed() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the model bundle directory") else { return };
    let (mut model, prompt) = open(&root);
    assert!(prompt.len() > 128, "the prompt takes {} tokens: too few batches", prompt.len());
    let first = request(&mut model, &prompt).expect("a request that does not fail");
    let second = request(&mut model, &prompt).expect("a request that does not fail");
    let (compared, noise) = compare(&first, &second);
    let tolerance = (4.0 * noise).max(NOISE_FLOOR);
    eprintln!("no failure, twice: {compared} passes compared, the worst relative difference {noise:.2e}; the tolerance {tolerance:.2e}");
    assert!(compared >= 6, "too few passes before the tokens parted ({compared})");

    straydog_engine::gpu::fail_streamed_chunk(FAILING_CHUNK);
    let failed = request(&mut model, &prompt);
    let left = straydog_engine::gpu::fail_streamed_chunk(0);
    let error = failed.err().expect("the request with a failing chunk returns the error");
    assert!(error.contains("a streamed chunk fails"), "the error is the hook's: {error}");
    assert_eq!(left, 0, "the hook fired: {left} chunks were left to go through");

    // the requests after it, twice: the first has the table and streams as the failure left them
    for round in 1..=2 {
        let after = request(&mut model, &prompt).expect("the request after the failure succeeds");
        let (compared, worst) = compare(&first, &after);
        eprintln!("after the failure, request {round}: {compared} passes compared, the worst relative difference {worst:.2e}");
        assert!(compared >= 6, "request {round} after the failure: the tokens parted from the reference after {compared} passes");
        assert!(worst <= tolerance, "request {round} after the failure: the logits differ by {worst:.2e}, past {tolerance:.2e}");
    }
}
