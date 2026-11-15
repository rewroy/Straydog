//! A conversation's state handed to a fresh model (`Model::export_conversation`, `Model::import_conversation`): the model takes the next request of the
//! conversation from the place the last reply began, and its first token's logits are those of the model that never stopped.
//!
//! cargo test --release --features cuda -p straydog-engine --test conversation_gpu -- --nocapture --test-threads=1, with STRAYDOG_MODEL naming the
//! UD-IQ1_S bundle (and STRAYDOG_DRAFTER its MTP drafter's directory, for the run that carries the MTP layer's state too); under the GPU lock. The model
//! is placed as a 6 GiB card places it with the K/V in host memory: partially resident, as `straydog setup` leaves a small card.
//! STRAYDOG_CONVERSATION_TOKENS sets the conversation's length in tokens (default 6000; 36000 is past the 32768 positions the K/V starts with, so the
//! import grows it), and STRAYDOG_PROMPT_CPU=0 makes a prompt batch stream every expert beyond the slots, with which the two models compared agree to the bit.
//!
//! The comparison: model A reads the conversation, answers, and is asked its next request (the one that goes on from the conversation): it resumes
//! from the checkpoint its last reply began at. Model B is a fresh one handed the state A exported before that request: it must take the same request
//! from the same place, and its logits must be A's. Partial residency's runs differ from one another (which experts the slots hold at a moment depends on
//! timing: docs/DESIGN.md, partial residency), so the tolerance, where they differ, is four times what two fresh models that read the whole request
//! differ by (C and D), and never below 1e-4. (A model that read the whole request is not the comparison: a request that goes on from a checkpoint reads
//! its tokens in other batches than one that begins at the start, which moves the logits of this quantization by a few percent, restart or none.)
#![cfg(any(feature = "hip", feature = "cuda"))]

mod common;

use std::path::PathBuf;

use straydog_engine::group::{Group, SplitMode};
use straydog_engine::model::{Model, Settings};
use straydog_engine::prefix;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;
use straydog_engine::weights::Weights;

/// The tests load a model on the same GPU: one at a time.
static GPUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

const IM_START: u32 = 248045;

/// A model of the bundle, placed in 5.5 GiB with the K/V in host memory, and its tokenizer.
fn open(root: &std::ffi::OsStr, drafter: Option<&std::ffi::OsStr>) -> (Model, Text) {
    open_with(root, drafter, true)
}

/// [`open`] with the prompt's products in FP16 (`half`, as served) or in F32.
fn open_with(root: &std::ffi::OsStr, drafter: Option<&std::ffi::OsStr>, half: bool) -> (Model, Text) {
    let budget = (5.5 * (1u64 << 30) as f64) as u64;
    let mut group = Group::new(&[0], &[budget], SplitMode::Experts).unwrap();
    group.configure_gemm(2, half).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, "UD-IQ1_S").unwrap();
    let drafter = drafter.map(PathBuf::from);
    let weights = Weights::open(&PathBuf::from(root), bundle, drafter.as_deref(), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    // (128K of context: Q8 K/V is over a quarter of this budget, so it stays in host memory and starts at 32K positions)
    let settings = Settings { context: 131072, state_budget_bytes: 8 << 30, prefill_batch_size: 256, speculative: if drafter.is_some() { 3 } else { 0 },
                              cpu_threads: 0, row_cache_bytes: None };
    let mut model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(model.partial, "the model must be partially resident at this budget");
    assert!(model.host_kv, "its K/V must be in host memory");
    // (the prompt's experts all stream: the fresh models compared begin alike)
    model.prompt_share(false).unwrap();
    (model, text)
}

/// A bundle whole in the VRAM of the GPUs named (30 GiB each), with the MTP drafter, and its tokenizer.
fn open_full(root: &std::ffi::OsStr, drafter: &std::ffi::OsStr, half: bool, quantization: &str, devices: &[usize]) -> (Model, Text) {
    let mut group = Group::new(devices, &vec![30 << 30; devices.len()], SplitMode::Experts).unwrap();
    group.configure_gemm(2, half).unwrap();
    let bundle = straydog_engine::bundle::Id::find(Some(straydog_engine::bundle::MODEL), None, quantization).unwrap();
    let weights = Weights::open(&PathBuf::from(root), bundle, Some(&PathBuf::from(drafter)), straydog_engine::host::host_memory_bytes(), &|_, _, _| {}).unwrap();
    let text = Text::new(weights.resources.get("tokenizer.json").unwrap()).unwrap();
    let settings = Settings { context: 131072, state_budget_bytes: 8 << 30, prefill_batch_size: 2048, speculative: 3, cpu_threads: 0, row_cache_bytes: None };
    let model = Model::new(weights, group, &settings, &|_, _, _| {}).unwrap();
    assert!(!model.partial, "the model must be fully resident at these budgets");
    (model, text)
}

/// The conversation, `tokens` long at least, whose last message is the user's: a system block, turns of a user and an assistant, and a last user message
/// with the generation prompt after it. And the request that goes on from it: the same up to the generation prompt, then a reply as the template
/// writes it back (not the tokens the model wrote), a new user message and its generation prompt.
fn conversation(text: &Text, tokens: usize) -> (Vec<u32>, Vec<u32>) {
    let mut chat = String::from("<|im_start|>system\nYou answer questions about rules and sources, briefly.<|im_end|>\n");
    let mut line = 0;
    loop {
        chat.push_str(&format!("<|im_start|>user\nTurn {line}: what do the rules say about topic {} when {} facts are asked, citing source {}?<|im_end|>\n\
                                <|im_start|>assistant\nThe rules for topic {} give {} facts, from source {}.<|im_end|>\n",
                               line * 7 % 13, line % 9 + 3, line * 5 % 17, line * 7 % 13, line % 9 + 3, line * 5 % 17));
        line += 1;
        if line % 50 == 0 && text.encode(&chat).unwrap().len() >= tokens {
            break;
        }
    }
    let first = format!("{chat}<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n");
    let second = format!("{chat}<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>\n\
                          <|im_start|>user\nAnd of Japan?<|im_end|>\n<|im_start|>assistant\n");
    (text.encode(&first).unwrap(), text.encode(&second).unwrap())
}

/// Where the last reply of `prompt` began: its last `<|im_start|>`, rounded down to four.
fn reply_place(prompt: &[u32]) -> usize {
    prompt.iter().rposition(|token| *token == IM_START).unwrap() / 4 * 4
}

/// The logits of the first token after `prompt`, read by `model`.
fn first_logits(model: &mut Model, prompt: &[u32]) -> Vec<f32> {
    model.logits_capture = Some(Vec::new());
    model.generate(prompt, 1, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |_| {}).unwrap();
    model.logits_capture.take().unwrap().remove(0)
}

fn relative_difference(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        diff += f64::from(x - y).powi(2);
        norm += f64::from(*y).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

/// Every way a state can differ from the layout of the model it is offered to, which must refuse it with nothing changed.
fn refusals(model: &mut Model, state: &[u8], positions: usize) {
    let tokens = prefix::parse(state).unwrap().tokens;
    let mut cases: Vec<(&str, Vec<u8>)> = vec![("a byte short", state[..state.len() - 1].to_vec()), ("a byte over", [state, &[0u8]].concat()), ("nothing", Vec::new())];
    let mut other = state.to_vec();
    other[8] ^= 1;
    cases.push(("another version", other));
    let mut other = state.to_vec();
    other[12..16].copy_from_slice(&(tokens as u32 + 4).to_le_bytes());
    cases.push(("more tokens than its sections hold", other));
    let mut other = state.to_vec();
    other[12..16].copy_from_slice(&(tokens as u32 + 1).to_le_bytes());
    cases.push(("a count of tokens that is not a multiple of four", other));
    for (what, bytes) in cases {
        assert!(model.import_conversation(&bytes, positions).is_err(), "{what}: the model took it");
        assert!(model.parked_coverage().is_none(), "{what}: the model changed");
    }
}

fn kept_conversation(drafter: Option<&std::ffi::OsStr>, root_dir: &std::ffi::OsStr) {
    let _gpu = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let length: usize = std::env::var("STRAYDOG_CONVERSATION_TOKENS").ok().and_then(|count| count.parse().ok()).unwrap_or(6000);
    // A: reads the conversation, answers, and has its state to give
    let (mut a, text) = open(root_dir, drafter);
    let (first, second) = conversation(&text, length);
    let place = reply_place(&first);
    assert!(place >= 1024 && second[..place] == first[..place], "the next request goes on from the place the reply began");
    assert!(a.export_conversation().unwrap().is_none() && a.conversation_size().is_none(), "a model that has read nothing keeps nothing");
    eprintln!("the conversation: {} tokens, its last reply began at {place}; the next request is {} tokens", first.len(), second.len());
    let started = std::time::Instant::now();
    a.generate(&first, 8, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |_| {}).unwrap();
    eprintln!("A read the conversation in {:.1} s", started.elapsed().as_secs_f64());
    let (size_tokens, size) = a.conversation_size().expect("a conversation with a place its reply began at");
    assert_eq!(size_tokens, place);
    let started = std::time::Instant::now();
    let kept = a.export_conversation().unwrap().expect("a state to give");
    eprintln!("exported {} tokens, {} MB, in {:.2} s", kept.tokens, kept.bytes.len() >> 20, started.elapsed().as_secs_f64());
    assert_eq!((kept.tokens, kept.digest.as_str()), (place, prefix::tokens_digest(&first[..place]).as_str()));
    assert!(kept.bytes.len() as u64 <= size, "{} bytes where {size} were said", kept.bytes.len());
    // (the state is the checkpoint's, not the sequence's: asking again gives the same bytes, the reply the sequence went on with being no part of it)
    let again = a.export_conversation().unwrap().unwrap();
    assert!(again.bytes == kept.bytes, "the state changed between two exports");
    // A's next request: from the checkpoint its last reply began at
    let continued = first_logits(&mut a, &second);
    assert_eq!(a.last_metrics["cached_tokens"], place, "{:?}", a.last_metrics);
    drop(a);

    // B: a fresh model that is handed the state, and takes the same request
    let (mut b, _) = open(root_dir, drafter);
    refusals(&mut b, &kept.bytes, second.len() + 1);
    assert_eq!(b.held_prefix(&second), 0);
    let started = std::time::Instant::now();
    assert_eq!(b.import_conversation(&kept.bytes, second.len() + 1).unwrap(), place);
    eprintln!("imported in {:.2} s (the K/V holds {} positions)", started.elapsed().as_secs_f64(), b.kv_positions());
    assert_eq!(b.parked_coverage(), Some((Some((0, place)), place)));
    assert_eq!(b.held_prefix(&second), place);
    let started = std::time::Instant::now();
    let handed = first_logits(&mut b, &second);
    eprintln!("B's request took {:.2} s", started.elapsed().as_secs_f64());
    assert_eq!(b.last_metrics["cached_tokens"], place, "{:?}", b.last_metrics);
    assert_eq!(b.last_metrics["resumed_from"], "kept");
    assert_eq!(b.last_metrics["prefill_tokens"], second.len() - place);
    assert!(b.parked_coverage().is_none(), "the state is the held sequence now");
    // (and the place its reply began is a checkpoint again, as after any request that made one: the next start can keep this conversation too)
    let again = b.export_conversation().unwrap().expect("the conversation is kept again after a restore");
    assert_eq!(again.tokens, reply_place(&second), "the new request's own checkpoint is the deepest");
    drop(b);

    // The same bits as the model that never stopped: nothing to calibrate. Otherwise (a partially resident model's slots differ between runs) the tolerance
    // is four times what two fresh models that read the whole request differ by (C and D), and never below 1e-4.
    let same_bits = handed.len() == continued.len() && handed.iter().zip(&continued).all(|(x, y)| x.to_bits() == y.to_bits());
    let worst = relative_difference(&handed, &continued);
    if same_bits {
        eprintln!("the logits of the first token after the restore are the same bits as the model that never stopped");
    } else {
        let (mut c, _) = open(root_dir, drafter);
        let whole = first_logits(&mut c, &second);
        assert_eq!(c.last_metrics["cached_tokens"], 0);
        drop(c);
        let (mut d, _) = open(root_dir, drafter);
        let whole_again = first_logits(&mut d, &second);
        drop(d);
        let noise = relative_difference(&whole, &whole_again);
        let tolerance = (4.0 * noise).max(1e-4);
        eprintln!("two fresh models differ by {noise:.2e}; the one handed the state differs from the one that never stopped by {worst:.2e}; the tolerance {tolerance:.2e}");
        assert!(worst <= tolerance, "{worst} against {tolerance}");
    }
    assert!(!std::env::var("STRAYDOG_PROMPT_CPU").is_ok_and(|value| value == "0") || same_bits,
            "with every expert beyond the slots streamed, the two agree to the bit: they differ by {worst:.2e}");
}

/// A long reply is marked as it grows (the recurrent state, every 512 tokens, at a multiple of four). A client that keeps it only in part
/// (a file it cut at a line's end to have the model go on, a reply its person stopped) sends a prompt that leaves the held sequence before
/// its end: the request goes on from the last mark the part reaches (the mark before the last, where the part ends between the two), where
/// it read the whole reply again. Its first token is the one a model that reads the whole request chooses.
fn kept_part(drafter: Option<&std::ffi::OsStr>, root_dir: &std::ffi::OsStr) {
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (mut model, text) = open(root_dir, drafter);
    let prompt = text.encode("<|im_start|>user\nWrite a long and detailed essay about the sea in winter, of at least two thousand words.<|im_end|>\n<|im_start|>assistant\n").unwrap();
    let tail = text.encode("<|im_end|>\n<|im_start|>user\nGo on from there.<|im_end|>\n<|im_start|>assistant\n").unwrap();
    let top = |logits: &[f32]| logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(token, _)| token);
    // (the part kept ends 40 tokens past the last mark, then 8 tokens before it: after the mark before the last)
    for (past, kind) in [(40isize, "mark"), (-8, "mark_before")] {
        model.reset().unwrap();
        let mut written = Vec::new();
        model.generate(&prompt, 1300, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |token| written.push(token)).unwrap();
        let marks = model.last_metrics.get("marks").and_then(|marks| marks.as_u64()).unwrap_or(0);
        let marked: Vec<usize> = model.last_metrics.get("marked_tokens").and_then(|marked| marked.as_array()).map(|marked| marked.iter().map(|at| at.as_u64().unwrap() as usize).collect())
            .unwrap_or_default();
        eprintln!("a reply of {} tokens: {marks} marks at {marked:?}, {:.1} ms each", written.len(),
                  1e3 * model.last_metrics.get("mark_seconds").and_then(|seconds| seconds.as_f64()).unwrap_or(0.0) / marks.max(1) as f64);
        assert!(marked.len() >= 2 && marked.iter().all(|at| at % 4 == 0), "the reply was long enough to be marked twice: {} tokens", written.len());
        let (last, before) = (marked[marked.len() - 1], marked[marked.len() - 2]);
        let kept = (last as isize + past) as usize - prompt.len();
        assert!(kept + 8 < written.len() && last - before >= 512, "the part kept ends before the reply does");
        let next: Vec<u32> = prompt.iter().chain(&written[..kept]).chain(&tail).copied().collect();
        let stepped = first_logits(&mut model, &next);
        let expected = if kind == "mark" { last } else { before };
        assert_eq!((model.last_metrics["resumed_from"].as_str(), model.last_metrics["cached_tokens"].as_u64()), (Some(kind), Some(expected as u64)));
        // The same request read whole, and (with no marks) from the place the reply began.
        model.reset().unwrap();
        let whole = first_logits(&mut model, &next);
        assert_eq!(model.last_metrics["cached_tokens"].as_u64(), Some(0));
        eprintln!("from the {kind} and read whole: relative difference {:.4}", relative_difference(&stepped, &whole));
        assert_eq!(top(&stepped), top(&whole), "the first token from the {kind}");
    }
    // A model that takes no marks reads the reply again from where it began.
    model.set_reply_marks(false);
    model.reset().unwrap();
    let mut written = Vec::new();
    model.generate(&prompt, 700, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |token| written.push(token)).unwrap();
    assert!(model.last_metrics.get("marks").is_none() && written.len() > 600);
    let next: Vec<u32> = prompt.iter().chain(&written[..600]).chain(&tail).copied().collect();
    first_logits(&mut model, &next);
    assert!(model.last_metrics["cached_tokens"].as_u64().is_some_and(|cached| (cached as usize) <= prompt.len()), "{:?}", model.last_metrics["resumed_from"]);
}

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

/// What the decode path makes of a reply against what the prompt path makes of the same tokens: a reply of 512 greedy tokens is
/// decoded again as known rows (passes of 1, 3 and 2 rows, their logits kept), and at every 32nd token the prompt and the reply
/// so far are read as a prompt by a fresh sequence. The same with F32 products (the reference). Printed: the next token's
/// distribution after the same tokens, decode against the F32 prompt read, beside the FP16 prompt read against it (the form long
/// prompts already have), and how often each chooses the reference's token. The model as a 6 GiB card places it: the decode's
/// routes run on the CPU where the slots hold no expert, the prompt's stream to the GPU.
///
/// STRAYDOG_DECODE_ACCURACY names the run ("fp16" or "f32") and STRAYDOG_DECODE_ACCURACY_DIR the folder its logits are kept in:
/// one run a process, as each locks the model's banks in host memory; the run that finds both compares.
#[test]
fn the_decode_path_is_as_near_the_f32_prompt_read_as_the_fp16_prompt_read() {
    let (Some(root_dir), Some(drafter)) = (common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory"),
                                           common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory")) else { return };
    let (Ok(mode), Ok(dir)) = (std::env::var("STRAYDOG_DECODE_ACCURACY"), std::env::var("STRAYDOG_DECODE_ACCURACY_DIR")) else {
        eprintln!("skipped: STRAYDOG_DECODE_ACCURACY (fp16 or f32) and STRAYDOG_DECODE_ACCURACY_DIR are not set");
        return;
    };
    let dir = PathBuf::from(dir);
    const REPLY: usize = 512;
    const EVERY: usize = 32;
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    {
        let half = match mode.as_str() {
            "fp16" => true,
            "f32" => false,
            other => panic!("STRAYDOG_DECODE_ACCURACY is {other}, not fp16 or f32"),
        };
        // (STRAYDOG_DECODE_ACCURACY_PLACE=full: the model whole in the VRAM of two GPUs, as served there, with STRAYDOG_MODEL its
        // UD-IQ3_XXS bundle; else as a 6 GiB card places UD-IQ1_S)
        // (the F32 reference of the small placement is read by two GPUs that hold the model whole: F32 products fit neither 5.5 GiB nor one card)
        let (mut model, text) = if std::env::var("STRAYDOG_DECODE_ACCURACY_PLACE").is_ok_and(|place| place == "full") { open_full(&root_dir, &drafter, half, "UD-IQ3_XXS", &[0, 1]) }
                                else if half { open_with(&root_dir, Some(&drafter), half) } else { open_full(&root_dir, &drafter, half, "UD-IQ1_S", &[0, 1]) };
        let prompt = text.encode("<|im_start|>user\nWrite a long and detailed essay about the sea in winter, of at least two thousand words.<|im_end|>\n<|im_start|>assistant\n").unwrap();
        // (the reply: the fp16 run's greedy tokens, which the f32 run takes from its file so that both read the same tokens)
        let reply_file = dir.join("reply.bin");
        let reply: Vec<u32> = match std::fs::read(&reply_file) {
            Ok(bytes) => bytes.chunks_exact(4).map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())).collect(),
            Err(_) => {
                let mut reply = Vec::new();
                model.generate(&prompt, REPLY, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |token| reply.push(token)).unwrap();
                assert_eq!(reply.len(), REPLY, "a long reply");
                std::fs::write(&reply_file, reply.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                reply
            }
        };
        let bytes = |rows: &[f32]| -> Vec<u8> { rows.iter().flat_map(|value| value.to_le_bytes()).collect() };
        // The prompt path: after the reply's first k tokens, k = 32, 64, ...
        let mut read = Vec::new();
        for k in (EVERY..REPLY).step_by(EVERY) {
            model.reset().unwrap();
            let whole: Vec<u32> = prompt.iter().chain(&reply[..k]).copied().collect();
            read.extend(first_logits(&mut model, &whole));
        }
        std::fs::write(dir.join(format!("{mode}-read.bin")), bytes(&read)).unwrap();
        // The decode path, where the run's own greedy tokens are the reply (the fp16 run): the same tokens as known rows.
        if half {
            model.reset().unwrap();
            model.forced_rows = Some(straydog_engine::model::ForcedRows { tokens: reply.clone(), widths: vec![1, 3, 2], read_logits: true, keep_logits: true,
                                                                         teacher: true, ..Default::default() });
            let mut again = Vec::new();
            model.generate(&prompt, REPLY, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, true, true, &mut |token| again.push(token)).unwrap();
            let forced = model.forced_rows.take().unwrap();
            assert_eq!(again, reply, "the known rows are the reply");
            assert_eq!(forced.rejected, 0);
            let vocabulary = forced.kept_logits.len() / (REPLY - 1);
            // (row i is the logits after the reply's token i: the rows after tokens 31, 63, ... are after its first 32, 64, ... tokens)
            let decoded: Vec<f32> = (EVERY..REPLY).step_by(EVERY).flat_map(|k| forced.kept_logits[(k - 1) * vocabulary..k * vocabulary].to_vec()).collect();
            std::fs::write(dir.join("fp16-decode.bin"), bytes(&decoded)).unwrap();
        }
    }
    let rows = |name: &str| -> Option<Vec<Vec<f32>>> {
        let values: Vec<f32> = std::fs::read(dir.join(name)).ok()?.chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
        let points = (EVERY..REPLY).step_by(EVERY).count();
        Some(values.chunks_exact(values.len() / points).map(<[f32]>::to_vec).collect())
    };
    let (Some(exact), Some(read), Some(decoded)) = (rows("f32-read.bin"), rows("fp16-read.bin"), rows("fp16-decode.bin")) else {
        eprintln!("the other run's logits are not there yet: nothing compared");
        return;
    };
    for (name, other) in [("the FP16 prompt read", &read), ("the decode path", &decoded)] {
        let kls: Vec<f64> = exact.iter().zip(other).map(|(exact, other)| kl(exact, other)).collect();
        let same = exact.iter().zip(other).filter(|(exact, other)| best(exact) == best(other)).count();
        let gap = exact.iter().zip(other).map(|(exact, other)| exact[best(exact)] - exact[best(other)]).fold(0f32, f32::max);
        eprintln!("{name} against the F32 prompt read: KL mean {:.3e}, worst {:.3e}, by place {:?}; the same next token at {same} of {}; its choice at most {gap:.3} below",
                  kls.iter().sum::<f64>() / kls.len() as f64, kls.iter().copied().fold(0f64, f64::max), kls.iter().map(|kl| format!("{kl:.1e}")).collect::<Vec<_>>(), kls.len());
    }
    let between: Vec<f64> = read.iter().zip(&decoded).map(|(read, decoded)| kl(read, decoded)).collect();
    eprintln!("the decode path against the FP16 prompt read: KL mean {:.3e}, worst {:.3e}", between.iter().sum::<f64>() / between.len() as f64,
              between.iter().copied().fold(0f64, f64::max));
}

/// A client that writes its folder into its system prompt sends a system prompt that ends otherwise at every new conversation:
/// none extends another's root. The second such prompt is checkpointed where it parts from the first (its stem, a root of its
/// own), every later one starts from there, and a new conversation with the same system prompt starts from that prompt's own root.
#[test]
fn a_system_prompt_with_another_end_starts_from_its_stem() {
    let Some(root_dir) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let _gpus = GPUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (mut model, text) = open(&root_dir, None);
    let document: String = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/DESIGN.md")).unwrap().chars().take(14000).collect();
    let prompt = |folder: &str, question: &str| -> Vec<u32> {
        text.encode(&format!("<|im_start|>system\n{document}\n\nWorking folder: /work/{folder}\nToday is a day in November.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")).unwrap()
    };
    let (one, two, three, four) = (prompt("alpha-run", "What is this project?"), prompt("bravo-run", "What is this project?"),
                                   prompt("charlie-run", "What is this project?"), prompt("delta-run", "What is this project?"));
    let shared = one.iter().zip(&two).take_while(|(a, b)| a == b).count();
    assert!(shared > 2200 && shared + 8 < one.len(), "the system prompts part near their end: {shared} of {}", one.len());
    let top = |logits: &[f32]| logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(token, _)| token);
    first_logits(&mut model, &one);
    first_logits(&mut model, &two);
    assert_eq!(model.last_metrics["cached_tokens"].as_u64(), Some(0), "the second ending extends nothing yet");
    for later in [&three, &four] {
        let from_stem = first_logits(&mut model, later);
        let cached = model.last_metrics["cached_tokens"].as_u64().unwrap() as usize;
        eprintln!("resumed from {:?} at {cached} of {} ({shared} shared)", model.last_metrics["resumed_from"], later.len());
        assert!(matches!(model.last_metrics["resumed_from"].as_str(), Some("root") | Some("root_before")), "{:?}", model.last_metrics["resumed_from"]);
        assert!(cached % 4 == 0 && cached < shared && cached + 64 > shared, "from the stem: {cached} of {shared} shared");
        if std::ptr::eq(later, &four) {
            // (the same prompt read whole chooses the same first token)
            model.reset().unwrap();
            let whole = first_logits(&mut model, later);
            eprintln!("from the stem and read whole: relative difference {:.4}", relative_difference(&from_stem, &whole));
            assert_eq!(top(&from_stem), top(&whole), "the first token from the stem");
        }
    }
    // (a new conversation with the last system prompt itself starts from that prompt's own root, past the stem)
    let same = prompt("delta-run", "Name its main parts.");
    first_logits(&mut model, &same);
    let cached = model.last_metrics["cached_tokens"].as_u64().unwrap() as usize;
    assert!(cached > shared && cached + 64 > reply_place(&same) - 32, "from its own root: {cached} ({shared} shared, {:?})", model.last_metrics["resumed_from"]);
}

#[test]
fn a_reply_kept_in_part_goes_on_from_its_mark() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    kept_part(None, &root);
}

#[test]
fn a_reply_kept_in_part_goes_on_from_its_mark_with_the_mtp_layers_too() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let Some(drafter) = common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory") else { return };
    kept_part(Some(&drafter), &root);
}

#[test]
fn a_fresh_model_takes_a_conversations_state() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    kept_conversation(None, &root);
}

#[test]
fn a_fresh_model_takes_the_conversation_with_the_mtp_layers_too() {
    let Some(root) = common::model_env("STRAYDOG_MODEL", "the UD-IQ1_S model bundle directory") else { return };
    let Some(drafter) = common::model_env("STRAYDOG_DRAFTER", "the MTP drafter's directory") else { return };
    kept_conversation(Some(&drafter), &root);
}
