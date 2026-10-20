//! The worker's DeepSeek-V4.1 side: a bundle read on demand ([`straydog_engine::pinned`]), its
//! tokenizer and chat template from the GGUF's own metadata, and the request loop of
//! [`crate::worker`] over [`straydog_engine::arch::deepseek41::serve::Model`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value};
use straydog_engine::arch::deepseek41::serve::{Model, Settings};
use straydog_engine::chat::ChatTemplate;
use straydog_engine::group::SplitMode;
use straydog_engine::pyjson;
use straydog_engine::sampling::Sampling;
use straydog_engine::text::Text;

use crate::worker::{refused, serve, Failure, Output, Served, WorkerArgs};

/// The model and its template, on the thread that made them: the engines hold device state that
/// stays on one thread.
struct DeepSeek {
    model: Model,
    template: ChatTemplate,
}

/// A request for the model's thread.
enum Job {
    Run { body: pyjson::Value, id: String, output: Output, cancel: Arc<AtomicBool>, started: Instant, text: Arc<Text>,
          reply: mpsc::Sender<Result<Value, Failure>> },
    Reset(mpsc::Sender<Result<(), String>>),
}

/// The request loop's side: each request runs on the model's thread.
struct Handle(mpsc::Sender<Job>);

impl Served for Handle {
    fn run(&mut self, text: &Arc<Text>, output: &Output, id: &str, body: &pyjson::Value, cancel: &Arc<AtomicBool>,
           started: Instant) -> Result<Value, Failure> {
        let (reply, result) = mpsc::channel();
        let job = Job::Run { body: body.clone(), id: id.to_string(), output: output.clone(), cancel: cancel.clone(), started,
                             text: text.clone(), reply };
        let gone = || Failure { message: "the Straydog model thread has stopped".into(), fatal: true };
        self.0.send(job).map_err(|_| gone())?;
        result.recv().map_err(|_| gone())?
    }

    fn reset(&mut self) -> Result<(), String> {
        let (reply, result) = mpsc::channel();
        self.0.send(Job::Reset(reply)).map_err(|_| "the Straydog model thread has stopped".to_string())?;
        result.recv().map_err(|_| "the Straydog model thread has stopped".to_string())?
    }
}

pub(crate) fn main(args: &WorkerArgs, output: Output, report: &(dyn Fn(&str, u64, u64) + Sync)) -> Result<(), String> {
    let data_dir = args.data_dir.as_deref().ok_or("a bundle read on demand needs Straydog's data directory (--data-dir)")?;
    let host_memory_bytes = if args.host_memory_bytes > 0 { args.host_memory_bytes } else { straydog_engine::host::host_memory_bytes() };
    let settings = Settings {
        devices: args.devices.clone(),
        split_mode: if args.split_mode == "layer" { SplitMode::Layer } else { SplitMode::Experts },
        vram_bytes: args.vram_bytes,
        context: args.context,
        batch: args.prefill_batch_size,
        host_expert_bytes: args.host_expert_bytes,
        host_memory_bytes,
        row_cache_bytes: args.row_cache_bytes,
    };
    let (jobs, queue) = mpsc::channel::<Job>();
    let (loaded, ready) = mpsc::channel::<Result<(Value, Arc<Text>), String>>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut deepseek = match load(args, data_dir, &settings, report) {
                Ok((deepseek, frame, text)) => {
                    let _ = loaded.send(Ok((frame, text)));
                    deepseek
                }
                Err(error) => {
                    let _ = loaded.send(Err(error));
                    return;
                }
            };
            for job in queue {
                match job {
                    Job::Run { body, id, output, cancel, started, text, reply } => {
                        let _ = reply.send(deepseek.run(&text, &output, &id, &body, &cancel, started));
                    }
                    Job::Reset(reply) => {
                        deepseek.model.reset();
                        let _ = reply.send(Ok(()));
                    }
                }
            }
            if let Err(error) = deepseek.model.settle() {
                tracing::error!("Straydog could not settle the expert store: {error}");
            }
        });
        let (frame, text) = ready.recv().map_err(|_| "the Straydog model thread failed to start".to_string())??;
        output.send(&frame);
        serve(Arc::new(Mutex::new(Handle(jobs))), text, output)
    })
}

/// Open, authenticate and place the model; the ready frame.
fn load(args: &WorkerArgs, data_dir: &std::path::Path, settings: &Settings, report: &(dyn Fn(&str, u64, u64) + Sync))
        -> Result<(DeepSeek, Value, Arc<Text>), String> {
    report("open", 0, 0);
    let mut model = Model::new(&args.model, &args.quant, data_dir, settings, report)?;
    report("prepare", 0, 0);
    let header = model.header();
    let text = Text::from_gguf(header).map_err(|exc| exc.to_string())?;
    let source = header.metadata.get("tokenizer.chat_template").and_then(|value| value.text())
        .ok_or("the GGUF has no chat template")?;
    let template = ChatTemplate::new(source).map_err(|exc| exc.0)?;
    let architecture = header.architecture.clone();
    report("warm", 0, 0);
    let warming = Instant::now();
    // One short prompt and a token, so no request pays for the first passes' setup.
    model.generate(&[0, 1000, 2000, 3000], 2, &Sampling::GREEDY, 0, &|| false, &mut |_, _| {}, &mut |_| {})?;
    model.reset();
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
        "placement": model.placement,
        "sampling": {"temperature": card.temperature, "top_k": card.top_k, "top_p": card.top_p, "min_p": card.min_p,
                     "presence_penalty": card.presence_penalty},
        "load_seconds": seconds,
        "gpu": model.info()?,
    });
    Ok((DeepSeek { model, template }, frame, Arc::new(text)))
}

impl DeepSeek {
    fn run(&mut self, text: &Text, output: &Output, id: &str, body: &pyjson::Value, cancel: &AtomicBool, started: Instant)
           -> Result<Value, Failure> {
        use pyjson::Value as Py;
        let messages = body.get("messages").ok_or_else(|| refused("'messages'"))?;
        let tools = body.get("tools").cloned().unwrap_or(Py::Null);
        let thinking = body.get("thinking").is_some_and(Py::truthy);
        // The template's own levels (low 50, medium 62, high 75, max 100); it refuses others.
        let effort = match body.get("reasoning_effort") {
            None | Some(Py::Null) => Py::Str("high".into()),
            Some(effort @ Py::Str(_)) => effort.clone(),
            Some(_) => return Err(refused("invalid Straydog reasoning effort")),
        };
        let variables = [("enable_thinking", Py::Bool(thinking)), ("reasoning_effort", effort)];
        let prompt = self.template.render_with(messages, &tools, &variables).map_err(|exc| refused(exc.0))?;
        let ids = text.encode(&prompt).map_err(|exc| Failure { message: exc.to_string(), fatal: true })?;
        let encode_seconds = started.elapsed().as_secs_f64();
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
        let mut decode_error: Option<String> = None;
        let cancelled = || cancel.load(Ordering::SeqCst);
        let result = self.model.generate(
            &ids, max_new, &sampling, seed, &cancelled,
            &mut |done, total| output.send(&json!({"id": id, "event": "progress", "done": done, "total": total})),
            &mut |token| {
                generated += 1;
                match stream.push(token) {
                    Ok(delta) if !delta.is_empty() => {
                        first_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
                        output.send(&json!({"id": id, "event": "delta", "content": delta, "tokens": generated}));
                    }
                    Ok(_) => {}
                    Err(exc) => {
                        decode_error.get_or_insert(exc.to_string());
                    }
                }
            },
        );
        result.map_err(|message| Failure { message, fatal: true })?;
        if let Some(message) = decode_error {
            return Err(Failure { message, fatal: true });
        }
        let rest = stream.finish().map_err(|exc| Failure { message: exc.to_string(), fatal: true })?;
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
        timings.insert("first_text_seconds".into(), json!(first_text));
        timings.insert("request_seconds".into(), json!(started.elapsed().as_secs_f64()));
        let cached = timings.get("cached_tokens").and_then(Value::as_u64).unwrap_or(0);
        Ok(json!({
            "id": id, "event": "done", "finish_reason": finish_reason,
            "usage": {"prompt_tokens": ids.len(), "completion_tokens": generated, "total_tokens": ids.len() + generated,
                      "cached_tokens": cached},
            "gpu": self.model.info().map_err(|message| Failure { message, fatal: true })?,
            "timings": timings,
        }))
    }
}
