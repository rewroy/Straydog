//! A partially resident prompt's experts with the fewest rows computed by the CPU while the bus streams the rest (`prompt_cpu_share`,
//! `gpu/resident_moe.hpp`): the prompt's logits are those of every expert streamed, but for the CPU's rounding.
//!
//! cargo test --release --features hip -p straydog-engine --test prompt_cpu_gpu, with STRAYDOG_MODEL naming the UD-IQ1_S bundle; under the GPU
//! lock.
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The relative difference of the prompt's logits the share may make. The CPU's routes take 8-bit activations where the GPU's take FP16
/// (decode's own mix of the two moved a pass's logits by up to 6.1e-2 between runs: behind_gpu); a route's result in another route's row, or
/// in none, moves them past 1.
const TOLERANCE: f64 = 0.15;

/// The model at a 6 GiB card's budget and a prompt of a few hundred tokens, in batches of 64 rows.
fn open(root: &std::ffi::OsStr) -> (Model, Vec<u32>) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, true).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
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

/// A request from the start of the prompt, with or without the share: the prompt's logits, the token they give, and the routes the CPU took.
fn prompt(model: &mut Model, prompt: &[u32], share: bool) -> (Vec<f32>, u32, u64) {
    model.prompt_share(share).unwrap();
    model.reset().unwrap();
    model.logits_capture = Some(Vec::new());
    let mut tokens = Vec::new();
    model.generate(prompt, 2, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, false, &mut |token| tokens.push(token)).unwrap();
    let logits = model.logits_capture.take().unwrap().swap_remove(0);
    let routes = model.last_metrics["partial_detail"]["prompt_cpu_routes"].as_f64().unwrap_or(0.0) as u64;
    (logits, tokens[0], routes)
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// Every expert streamed, a prompt's logits are the same from run to run; with the CPU's share they move by its rounding alone, and give
/// the same token.
#[test]
fn a_prompts_cpu_share_moves_its_logits_by_rounding_alone() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let (mut model, tokens) = open(&root);
    assert!(tokens.len() > 128, "the prompt takes {} tokens: too few batches", tokens.len());
    let (streamed, token, routes) = prompt(&mut model, &tokens, false);
    assert_eq!(routes, 0, "the share was off");
    let (again, _, _) = prompt(&mut model, &tokens, false);
    assert!(streamed.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "streamed alone, the logits changed between runs");
    let (shared, shared_token, routes) = prompt(&mut model, &tokens, true);
    assert!(routes > 0, "the CPU took no route of the prompt");
    let difference = relative_difference(&shared, &streamed);
    eprintln!("{} prompt tokens: the CPU took {routes} routes; the logits' relative difference {difference:.2e} (tolerance {TOLERANCE:.2e})",
              tokens.len());
    assert!(difference <= TOLERANCE, "the logits moved by {difference:.2e}");
    assert_eq!(shared_token, token, "the prompt's token");
}
