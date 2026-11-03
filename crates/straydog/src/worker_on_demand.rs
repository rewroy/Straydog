//! The worker's side of a bundle read on demand ([`straydog_engine::pinned`]): the architecture the
//! bundle's pins name (DeepSeek-V4.1) places its engines into a served session, the
//! tokenizer and chat template come from the model's pinned resources (else the GGUF's own
//! metadata), and the request loop of [`crate::worker`] runs over it.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value};
use straydog_engine::arch::served::{Engines, Session, Settings, Warmed};
use straydog_engine::arch::deepseek41;
use straydog_engine::bundle;
use straydog_engine::chat::{self, ChatTemplate};
use straydog_engine::constraint::Constraint;
use straydog_engine::reply::THINK_CLOSE;
use straydog_engine::pyjson;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;

use crate::images;
use crate::worker::{engine_failed, gpu_info_or_null, refused, serve, Failure, Output, Plan, Served, WorkerArgs};

/// The model and its template, on the thread that made them: the engines hold device state that
/// stays on one thread.
struct Loaded<E: Engines> {
    model: Session<E>,
    template: ChatTemplate,
    /// Whether the template takes a reasoning effort of its own (DeepSeek-V4.1's levels).
    efforts: bool,
    /// The span rows of the images encoded last (by their bytes' digest), newest last: a
    /// conversation sends its images again with every turn.
    encoded: Vec<([u8; 32], Arc<Vec<f32>>)>,
}

/// Bytes of image rows kept for the turns after ([`Loaded::encoded`]): ~30 screenshots (a
/// 1440x900 one is ~740 positions of DeepSeek-V4.1-Flash's 5120 values, ~15 MB). At 64 MiB an
/// agent's nine screenshots evicted each other in turn (each request sends them all, in order)
/// and every request encoded all nine again, 2.7-2.9 s of each wait.
const ENCODED_BYTES: usize = 512 << 20;

/// How an architecture's engines are placed.
type Place<E> = fn(&Path, &str, &Path, &Settings, &(dyn Fn(&str, u64, u64) + Sync)) -> Result<Session<E>, String>;

/// A request for the model's thread.
enum Job {
    Run { body: pyjson::Value, id: String, output: Output, cancel: Arc<AtomicBool>, started: Instant, text: Arc<Text>,
          reply: mpsc::Sender<Result<Value, Failure>> },
    Reset(mpsc::Sender<Result<(), String>>),
    WarmState(mpsc::Sender<Value>),
    Warm(Value),
}

/// Warm-up experts started per step while the model is idle (a step every millisecond), and
/// the reads (one per device of an expert) at most on their way: enough to keep the drive's
/// sixteen bulk reads busy (~300 MB queued); a request's reads go ahead of them.
const WARM_STEP: usize = 64;
const WARM_IN_FLIGHT: usize = 64;

/// The request loop's side: each request runs on the model's thread.
struct Handle(mpsc::Sender<Job>);

impl Served for Handle {
    fn run(&mut self, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value, cancel: &Arc<AtomicBool>,
           started: Instant) -> Result<Value, Failure> {
        let (reply, result) = mpsc::channel();
        let job = Job::Run { body: body.clone(), id: id.to_string(), output: output.clone(), cancel: cancel.clone(), started,
                             text: text.clone(), reply };
        let gone = || engine_failed("the Straydog model thread has stopped", concat!(file!(), ":", line!(), " Handle::run"));
        self.0.send(job).map_err(|_| gone())?;
        result.recv().map_err(|_| gone())?
    }

    fn reset(&mut self) -> Result<(), String> {
        let (reply, result) = mpsc::channel();
        self.0.send(Job::Reset(reply)).map_err(|_| "the Straydog model thread has stopped".to_string())?;
        result.recv().map_err(|_| "the Straydog model thread has stopped".to_string())?
    }

    fn warm_state(&mut self) -> Result<Value, String> {
        let (reply, result) = mpsc::channel();
        self.0.send(Job::WarmState(reply)).map_err(|_| "the Straydog model thread has stopped".to_string())?;
        result.recv().map_err(|_| "the Straydog model thread has stopped".to_string())
    }

    fn warm(&mut self, state: &Value) -> Result<(), String> {
        self.0.send(Job::Warm(state.clone())).map_err(|_| "the Straydog model thread has stopped".to_string())
    }
}

pub(crate) fn main(args: &WorkerArgs, plan: &Plan, output: Output, report: &(dyn Fn(&str, u64, u64) + Sync)) -> Result<(), String> {
    match bundle::architecture_of(&args.quant)?.as_str() {
        "deepseek41" => run::<deepseek41::serve::Engines>(args, plan, output, report, deepseek41::serve::place),
        other => Err(format!("no engine for the {other} architecture")),
    }
}

fn run<E: Engines + 'static>(args: &WorkerArgs, plan: &Plan, output: Output, report: &(dyn Fn(&str, u64, u64) + Sync),
                             place: Place<E>) -> Result<(), String> {
    let data_dir = args.data_dir.as_deref().ok_or("a bundle read on demand needs its data directory (the verity trees)")?;
    let host_memory_bytes = if args.host_memory_bytes > 0 { args.host_memory_bytes } else { straydog_engine::host::host_memory_bytes() };
    let settings = Settings {
        devices: args.devices.clone(),
        split_mode: plan.split,
        vram_bytes: plan.budgets.clone(),
        context: args.context,
        // (0: the architecture's own choice, from the memory it has: DeepSeek-V4.1's)
        batch: if args.prefill_batch_size == "auto" && bundle::architecture_of(&args.quant)? == "deepseek41" { 0 } else { plan.batch },
        host_expert_bytes: args.host_expert_bytes,
        host_memory_bytes,
        row_cache_bytes: args.row_cache_bytes.unwrap_or(straydog_engine::rows::DEFAULT_CACHE_BYTES),
        half_gemm: args.gemm == "fp16",
    };
    let (jobs, queue) = mpsc::channel::<Job>();
    let (loaded, ready) = mpsc::channel::<Result<(Value, Arc<Text>), String>>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut served = match load(args, data_dir, &settings, report, place) {
                Ok((served, frame, text)) => {
                    let _ = loaded.send(Ok((frame, text)));
                    served
                }
                Err(error) => {
                    let _ = loaded.send(Err(error));
                    return;
                }
            };
            // Between requests, the stores make room for the next prefill a few write-backs at a
            // time (and copy the checkpoints a request took); an earlier run's experts come back a
            // few reads at a time, then the free slots fill, in the time that leaves.
            let mut warming: Option<Instant> = None;
            let mut listed: Option<Instant> = None;
            let mut rooming = false;
            loop {
                let job = if warming.is_some() || rooming {
                    match queue.recv_timeout(std::time::Duration::from_millis(1)) {
                        Ok(job) => job,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if rooming {
                                match served.model.make_room() {
                                    Ok(done) => rooming = !done,
                                    Err(error) => {
                                        eprintln!("Straydog stopped making room for the next prefill: {error}");
                                        rooming = false;
                                    }
                                }
                            } else if let Some(since) = warming {
                                match served.model.warm_step(WARM_STEP, WARM_IN_FLIGHT) {
                                    Ok(Warmed::Going) => {}
                                    Ok(Warmed::Listed) => {
                                        eprintln!("Straydog warmed up in {:.1} s", since.elapsed().as_secs_f64());
                                        listed = Some(Instant::now());
                                    }
                                    Ok(Warmed::Done) => {
                                        let (vram, host, bytes) = served.model.filled();
                                        eprintln!("Straydog filled its free slots with {vram} experts for VRAM, {host} for the host tier \
                                                   ({:.1} GiB from the drive) in {:.1} s", bytes as f64 / (1u64 << 30) as f64,
                                                  listed.map_or(0.0, |listed| listed.elapsed().as_secs_f64()));
                                        warming = None;
                                        rooming = true;
                                    }
                                    Err(error) => {
                                        eprintln!("Straydog stopped warming up: {error}");
                                        warming = None;
                                    }
                                }
                            }
                            continue;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                } else {
                    match queue.recv() {
                        Ok(job) => job,
                        Err(_) => break,
                    }
                };
                match job {
                    Job::Run { body, id, output, cancel, started, text, reply } => {
                        let _ = reply.send(served.run(&text, &output, &id, &body, &cancel, started));
                        rooming = true;
                    }
                    Job::Reset(reply) => {
                        served.model.reset();
                        let _ = reply.send(Ok(()));
                    }
                    Job::WarmState(reply) => {
                        let _ = reply.send(served.model.warm_state());
                    }
                    Job::Warm(state) => match served.model.warm(&state) {
                        Ok((vram, host)) => {
                            eprintln!("Straydog warming up with {vram} experts for VRAM, {host} for the host tier");
                            warming = Some(Instant::now());
                            listed = None;
                        }
                        Err(error) => eprintln!("Straydog cannot warm up: {error}"),
                    },
                }
            }
            if let Err(error) = served.model.settle() {
                eprintln!("Straydog could not settle the expert store: {error}");
            }
        });
        let (frame, text) = ready.recv().map_err(|_| "the Straydog model thread failed to start".to_string())??;
        output.send(&frame);
        serve(Arc::new(Mutex::new(Handle(jobs))), text, output)
    })
}

/// Open, authenticate and place the model; the ready frame.
fn load<E: Engines>(args: &WorkerArgs, data_dir: &Path, settings: &Settings, report: &(dyn Fn(&str, u64, u64) + Sync),
                    place: Place<E>) -> Result<(Loaded<E>, Value, Arc<Text>), String> {
    report("open", 0, 0);
    let mut model = place(&args.model, &args.quant, data_dir, settings, report)?;
    report("prepare", 0, 0);
    let header = model.header();
    let architecture = header.architecture.clone();
    // The official tokenizer and chat template when the bundle pins them, else the GGUF's (but
    // DeepSeek-V4.1's own: not every GGUF carries the reference's).
    let resources = bundle::on_demand(&args.quant)?.resources;
    let resource = |name: &str| resources.iter().find(|asset| asset.name == name)
        .map(|asset| straydog_engine::pinned::resource(&args.model, asset)).transpose();
    let text = match resource("tokenizer.json")? {
        Some(json) => Text::new(&json),
        None => Text::from_gguf(header),
    }.map_err(|exc| exc.to_string())?;
    let source = match resource("chat_template.jinja")? {
        Some(bytes) => String::from_utf8(bytes).map_err(|exc| exc.to_string())?,
        None if architecture == "deepseek41" => straydog_engine::arch::deepseek41::CHAT_TEMPLATE.to_string(),
        None => header.metadata.get("tokenizer.chat_template").and_then(|value| value.text())
            .ok_or("the GGUF has no chat template")?.to_string(),
    };
    let template = ChatTemplate::new(&source).map_err(|exc| exc.0)?;
    let efforts = architecture == "deepseek41";
    report("warm", 0, 0);
    let warming = Instant::now();
    // One short prompt and a token, so no request pays for the first passes' setup.
    model.generate(&[0, 1000, 2000, 3000], 2, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |_| {})?;
    model.reset();
    // (the runtime has what it keeps beside the counted buffers now: the VRAM left over is slots)
    model.lend_spare()?;
    let mut seconds = serde_json::Map::new();
    for (step, value) in &model.load_seconds {
        seconds.insert(step.to_string(), (*value).into());
    }
    seconds.insert("warm".into(), warming.elapsed().as_secs_f64().into());
    let card = model.card.clone();
    let frame = json!({
        "event": "ready", "protocol": 1, "engine": "Straydog",
        "model": straydog_engine::bundle::model_of(&args.quant)?, "architecture": architecture,
        "quantization": args.quant, "context": model.context, "state_bytes": model.state_bytes, "speculative_drafts": 0,
        "speculation": speculation(&model.placement),
        "placement": model.placement,
        "sampling": {"temperature": card.temperature, "top_k": card.top_k, "top_p": card.top_p, "min_p": card.min_p,
                     "presence_penalty": card.presence_penalty},
        "load_seconds": seconds,
        "gpu": model.info()?,
        "vision": model.engines.vision(),
    });
    Ok((Loaded { model, template, efforts, encoded: Vec::new() }, frame, Arc::new(text)))
}

/// The speculation a served model runs, as its ready frame says: DSpark when it is placed (the most
/// tokens it drafts a pass, and the experts it keeps in each stage), and always prompt lookup, which
/// proposes where DSpark does not.
fn speculation(placement: &Value) -> Value {
    let mut kinds = Vec::new();
    if let Some(kept) = placement["dspark"]["resident"].as_array() {
        let drafts = straydog_engine::arch::deepseek41::drafter::MAX_SLOTS - 1;
        kinds.push(json!({"kind": "dspark", "drafts": drafts, "resident": kept}));
    }
    kinds.push(json!({"kind": "lookup"}));
    Value::Array(kinds)
}

impl<E: Engines> Loaded<E> {
    fn run(&mut self, text: &Text, output: &Output, id: &str, body: &pyjson::Value, cancel: &AtomicBool, started: Instant)
           -> Result<Value, Failure> {
        use pyjson::Value as Py;
        let messages = body.get("messages").ok_or_else(|| refused("'messages'"))?;
        // (each image part a marker of its own until the prompt is rendered)
        let (messages, images) = images::take(messages).map_err(refused)?;
        if !images.is_empty() && !self.model.engines.vision() {
            return Err(refused("this model reads no images: its vision encoder file is not beside it (`straydog fetch` builds it)"));
        }
        let messages = &messages;
        let tools = body.get("tools").cloned().unwrap_or(Py::Null);
        let thinking = body.get("thinking").is_some_and(Py::truthy);
        let mut variables = vec![("enable_thinking", Py::Bool(thinking))];
        if self.efforts {
            // As a budget in 1..100 (low 50, medium 62, high 75, max 100), which every DeepSeek-V4.1
            // template takes (the GSQ-RCO bundle's knows no "medium"); it refuses others.
            let effort = match body.get("reasoning_effort") {
                None | Some(Py::Null) => Py::Int(75),
                Some(Py::Str(level)) => match level.as_str() {
                    "low" => Py::Int(50),
                    "medium" => Py::Int(62),
                    "high" => Py::Int(75),
                    "max" => Py::Int(100),
                    _ => Py::Str(level.clone()),
                },
                Some(effort @ Py::Int(_)) => effort.clone(),
                Some(_) => return Err(refused("invalid Straydog reasoning effort")),
            };
            variables.push(("reasoning_effort", effort));
        }
        // (a message's own text never writes the template's tokens: see Text::defuse)
        let messages = &chat::defuse_messages(messages, &|own| text.defuse(own).into_owned());
        let prompt = self.template.render_with(messages, &tools, &variables).map_err(|exc| refused(exc.0))?;
        let pieces = images::pieces(&prompt, images.len()).map_err(refused)?;
        let images: Vec<images::Image> = images.iter().map(|bytes| images::prepare(bytes)).collect::<Result<_, _>>().map_err(refused)?;
        let (ids, spans) = images::ids(text, &pieces, &images).map_err(Failure::from)?;
        let encode_seconds = started.elapsed().as_secs_f64();
        let image_started = Instant::now();
        let rows = self.image_rows(&images, &spans)?;
        self.model.engines.set_images(spans.iter().map(|&(index, start)| (start, rows[index].clone().unwrap_or_default())).collect());
        let image_seconds = image_started.elapsed().as_secs_f64();
        let context = self.model.context;
        let max_new = match body.get("max_tokens") {
            None => 512,
            Some(Py::Null) => context as i128 - ids.len() as i128,
            Some(Py::Int(value)) => *value,
            Some(_) => return Err(refused("Straydog max_tokens must be an integer")),
        };
        // A control the request leaves null follows the model card.
        let card = self.model.card.clone();
        let number = |key: &str, card: f64| match body.get(key) {
            None | Some(Py::Null) => Ok(card),
            Some(Py::Int(value)) => Ok(*value as f64),
            Some(Py::Float(value)) => Ok(*value),
            Some(_) => Err(refused(format!("invalid Straydog {key}"))),
        };
        let top_k = match body.get("top_k") {
            None | Some(Py::Null) => card.top_k,
            Some(Py::Int(value)) if (0..=1_000_000).contains(value) => *value as usize,
            Some(_) => return Err(refused("invalid Straydog top_k")),
        };
        let sampling = Sampling {
            temperature: number("temperature", card.temperature)?,
            top_k,
            top_p: number("top_p", card.top_p)?,
            min_p: number("min_p", card.min_p)?,
            presence_penalty: number("presence_penalty", card.presence_penalty)?,
        };
        let seed = match body.get("seed") {
            None => 0,
            Some(Py::Int(value)) if *value >= 0 => *value as u64,
            Some(_) => return Err(refused("invalid Straydog seed")),
        };
        if ids.is_empty() || max_new < 1 || max_new > context as i128 || ids.len() as i128 + max_new > context as i128 {
            return Err(refused("Straydog prompt plus output exceeds the allocated context"));
        }
        sampling.check().map_err(refused)?;
        let max_new = max_new as usize;
        let mut stream = text.stream();
        let mut generated = 0usize;
        let mut first_text: Option<f64> = None;
        let mut decode_error: Option<Failure> = None;
        let cancelled = || cancel.load(Ordering::SeqCst);
        let constraint = Constraint::from_request(body.get("constraint").unwrap_or(&Py::Null), text.token_bytes(),
                                                  self.model.end_tokens(), thinking).map_err(refused)?;
        let mut constraint = Constraint::suppressing(constraint, body.get("suppress").unwrap_or(&Py::Null), text.token_bytes()).map_err(refused)?;
        // A reply that reasons ends it with the template's `</think>`, its one added token: the deltas
        // go without it and a reasoning_end event marks the place. The same text otherwise written (the
        // reasoning quoting a tag) stays the reasoning's.
        let mut think_close = if thinking { text.added_token(THINK_CLOSE) } else { None };
        let result = self.model.generate_constrained(
            &ids, max_new, &sampling, seed, &cancelled,
            &mut |done, total| output.send(&json!({"id": id, "event": "progress", "done": done, "total": total})),
            &mut |token| {
                generated += 1;
                match stream.push(token) {
                    Ok(delta) => {
                        let ended = think_close == Some(token);
                        let delta = if ended { delta.strip_suffix(THINK_CLOSE).map(str::to_string).unwrap_or(delta) } else { delta };
                        if !delta.is_empty() {
                            first_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
                            output.send(&json!({"id": id, "event": "delta", "content": delta, "tokens": generated}));
                        }
                        if ended {
                            think_close = None;
                            output.send(&json!({"id": id, "event": "reasoning_end", "tokens": generated}));
                        }
                    }
                    Err(exc) => {
                        decode_error.get_or_insert_with(|| exc.into());
                    }
                }
            },
            constraint.as_mut(),
        );
        result.map_err(|message| engine_failed(message, concat!(file!(), ":", line!(), " model.generate")))?;
        if let Some(failure) = decode_error {
            return Err(failure);
        }
        let rest = stream.finish().map_err(Failure::from)?;
        if !rest.is_empty() {
            output.send(&json!({"id": id, "event": "delta", "content": rest, "tokens": generated}));
        }
        let finish_reason = if cancelled() {
            "cancelled"
        } else if generated == max_new {
            "length"
        } else {
            "stop"
        };
        let mut timings = self.model.last_metrics.clone();
        timings.insert("encode_seconds".into(), json!(encode_seconds));
        if !spans.is_empty() {
            timings.insert("images".into(), json!(spans.len()));
            timings.insert("image_seconds".into(), json!(image_seconds));
        }
        timings.insert("first_text_seconds".into(), json!(first_text));
        timings.insert("request_seconds".into(), json!(started.elapsed().as_secs_f64()));
        let cached = timings.get("cached_tokens").and_then(Value::as_u64).unwrap_or(0);
        Ok(json!({
            "id": id, "event": "done", "finish_reason": finish_reason,
            "usage": {"prompt_tokens": ids.len(), "completion_tokens": generated, "total_tokens": ids.len() + generated,
                      "cached_tokens": cached},
            "gpu": gpu_info_or_null(self.model.info(), concat!(file!(), ":", line!(), " model.info")),
            "timings": timings,
        }))
    }
}

impl<E: Engines> Loaded<E> {
    /// The span rows of the placed images (by index; none for an image the prompt does not hold):
    /// those encoded lately kept, the others encoded together.
    fn image_rows(&mut self, images: &[images::Image], spans: &[(usize, usize)]) -> Result<Vec<Option<Arc<Vec<f32>>>>, Failure> {
        let mut rows: Vec<Option<Arc<Vec<f32>>>> = vec![None; images.len()];
        let mut missing = Vec::new();
        for &(index, _) in spans {
            match self.encoded.iter().position(|(digest, _)| *digest == images[index].digest) {
                Some(at) => {
                    let kept = self.encoded.remove(at);
                    rows[index] = Some(kept.1.clone());
                    self.encoded.push(kept);
                }
                None if !missing.contains(&index) => missing.push(index),
                None => {}
            }
        }
        if missing.is_empty() {
            return Ok(rows);
        }
        let patches: Vec<_> = missing.iter().map(|&index| images[index].patches.clone()).collect();
        // (a refusal: a device that failed fails the reset after it)
        let encoded = self.model.engines.encode_images(&patches).map_err(|message| refused(format!("the vision encoder: {message}")))?;
        for (index, values) in missing.into_iter().zip(encoded) {
            let values = Arc::new(values);
            rows[index] = Some(values.clone());
            self.encoded.push((images[index].digest, values));
        }
        while self.encoded.iter().map(|(_, values)| values.len() * 4).sum::<usize>() > ENCODED_BYTES && self.encoded.len() > 1 {
            self.encoded.remove(0);
        }
        Ok(rows)
    }
}
