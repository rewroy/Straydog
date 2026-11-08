//! The server: settings, logging, the runtime and the HTTP listener.

use std::path::Path;
use std::sync::Arc;

use clap::ArgMatches;
use straydog_api::ReasoningEffort;

use straydog::config::{self, Effort, Vram};
use straydog::dashboard::{self, Dashboard};
use straydog::monitor::{LogBuffer, LogSink};
use straydog::runtime::{self, Runtime};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

/// Log to the file or stderr, and into `buffer` for the dashboard.
fn logging(settings: &config::Settings, buffer: Arc<LogBuffer>) -> Result<(), String> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(if settings.verbose { "debug" } else { "info" }));
    let (file, stderr) = match &settings.log_file {
        Some(path) => {
            let file = std::fs::OpenOptions::new().create(true).append(true).open(path)
                .map_err(|exc| format!("{}: {exc}", path.display()))?;
            (Some(fmt::layer().with_target(false).with_ansi(false).with_writer(std::sync::Mutex::new(file))), None)
        }
        None => (None, Some(fmt::layer().with_target(false).with_writer(std::io::stderr))),
    };
    let dashboard = fmt::layer().with_target(false).with_ansi(false).with_writer(move || LogSink(buffer.clone()));
    tracing_subscriber::registry().with(filter).with(file).with(stderr).with(dashboard).init();
    Ok(())
}

/// What the starting line says of the drafter, after a comma: the MTP drafter given with its
/// drafts, or DeepSeek-V4.1's DSpark, which takes no option and drafts when its file lies beside
/// the model (the ready line then says how it was placed, and with how many experts); nothing
/// otherwise, as only the engine knows whether a drafter can sit beside the layers.
fn drafter_phrase(mtp_drafts: Option<u8>, dspark: bool) -> String {
    match (mtp_drafts, dspark) {
        (Some(drafts), _) => format!(", MTP drafter with {drafts} drafts"),
        (None, true) => ", DSpark drafter".into(),
        (None, false) => String::new(),
    }
}

fn drafter_clause(settings: &config::Settings) -> String {
    let off = |name| std::env::var_os(name).is_some_and(|value| value == "0");
    let mtp = settings.model_draft.is_some().then_some(settings.draft_max).filter(|&drafts| drafts > 0);
    let dspark = !off("DS41_DSPARK") && !off("DS41_DRAFTS")
        && matches!(straydog_engine::bundle::dspark_in(&settings.model, settings.bundle), Ok(Some(_)));
    drafter_phrase(mtp, dspark)
}

pub fn main(matches: &ArgMatches, config_file: Option<&Path>, print_config: bool) -> Result<(), String> {
    let (options, sources) = config::load(matches, config_file)?;
    let settings = config::resolve(options)?;
    // (the engram table named, from an option or the settings file as well as the variable: what the server's checks and
    // its worker, which keeps the variable, read; set before any thread starts)
    match &settings.engram {
        Some(precision) => std::env::set_var("STRAYDOG_ENGRAM", precision),
        None => std::env::remove_var("STRAYDOG_ENGRAM"),
    }
    // (speed first: the worker reads the floor from its environment, as the engine's STRAYDOG_EXPERT_FLOOR)
    match settings.expert_floor {
        Some(floor) => std::env::set_var("STRAYDOG_EXPERT_FLOOR", floor.to_string()),
        None => std::env::remove_var("STRAYDOG_EXPERT_FLOOR"),
    }
    if print_config {
        for line in config::describe(&settings, &sources) {
            println!("{line}");
        }
        return Ok(());
    }
    let logs = Arc::new(LogBuffer::default());
    logging(&settings, logs.clone())?;
    if let Some(reason) = straydog_isolation::unavailable_reason() {
        return Err(reason);
    }
    // Verity trees missing from the data directory are built before the worker starts (shown as
    // loading); without that, the worker's command is checked here, before the server listens.
    let missing_trees = runtime::missing_trees(&settings)?;
    let command = if missing_trees { None } else { Some(runtime::worker_command(&settings)?) };
    let tokio = tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|exc| exc.to_string())?;
    tokio.block_on(async move {
        tracing::info!(
            "Straydog starting: {} from {}, GPU {}, VRAM {}, context {}, prefill batch {}{}",
            settings.bundle,
            settings.model.display(),
            settings.gpu.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
            match &settings.vram { Vram::Auto => "auto".into(), budgets => format!("{budgets} GiB") },
            settings.ctx_size,
            settings.batch_size.map_or_else(|| "auto".into(), |batch| batch.to_string()),
            drafter_clause(&settings),
        );
        // The port before the worker: a server that cannot listen must not load a model beside the one that does.
        let address = format!("{}:{}", settings.host, settings.port);
        let listener = tokio::net::TcpListener::bind(&address).await.map_err(|exc| {
            let message = format!("{address}: {exc}");
            tracing::error!("Straydog cannot listen: {message}");
            message
        })?;
        let engine = Arc::new(match command {
            Some(command) => Runtime::start(&settings, command),
            None => {
                let prepared = settings.clone();
                Runtime::start_prepared(&settings, Box::new(move |progress| {
                    for name in straydog_provision::ensure_trees(&prepared.model, prepared.bundle, &prepared.data_dir, progress)? {
                        tracing::info!("Straydog verified {name} (its verity tree is built)");
                    }
                    runtime::worker_command(&prepared)
                }))
            }
        });
        let api = straydog_api::ApiConfig {
            model_id: settings.alias.clone(),
            context: u64::from(settings.ctx_size),
            quantization: settings.bundle.quant.to_string(),
            api_key: settings.api_key.clone(),
            sampling: straydog_api::SamplingDefaults {
                temperature: settings.temp,
                top_k: settings.top_k,
                top_p: settings.top_p,
                min_p: settings.min_p,
                presence_penalty: settings.presence_penalty,
            },
            seed: settings.seed,
            reasoning_effort: match settings.reasoning_effort {
                Effort::None => ReasoningEffort::None,
                Effort::Low => ReasoningEffort::Low,
                Effort::Medium => ReasoningEffort::Medium,
                Effort::High => ReasoningEffort::High,
            },
            created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |time| time.as_secs()),
            markup: straydog_engine::bundle::architecture_of(settings.bundle)
                .ok()
                .and_then(|architecture| straydog_api::Markup::of_architecture(&architecture))
                .ok_or("no reply markup for this model")?,
        };
        let monitor = Arc::new(Dashboard::new(engine.clone(), logs, &settings));
        let router = straydog_api::router(engine.clone(), api).merge(dashboard::router(monitor.clone()));
        tracing::info!("Straydog listening on http://{address}");
        // A worker that fails (to start or later) leaves nothing to serve: the server ends with
        // its failure, so that whatever runs it can start it again.
        let failure = Arc::new(std::sync::Mutex::new(None));
        let stop = {
            let (failed, failure, stop_on_stdin) = (engine.failed(), failure.clone(), settings.stop_on_stdin);
            async move {
                tokio::select! {
                    () = shutdown(stop_on_stdin) => {}
                    message = failed => *failure.lock().unwrap() = Some(message),
                }
            }
        };
        let served = axum::serve(listener, router).with_graceful_shutdown(stop).await.map_err(|exc| exc.to_string());
        monitor.stop().await;
        drop(monitor);
        // The router and the sampler are gone; the worker exits, and its GPU memory is free,
        // before the server does. A connection that has just closed may still be letting go of its
        // handle on the engine: it is waited for a moment, for an engine left open ends the worker
        // by killing it, and the conversation it holds is not kept (`keep_conversation`).
        let mut engine = engine;
        let began = std::time::Instant::now();
        let owned = loop {
            match Arc::try_unwrap(engine) {
                Ok(owned) => break Some(owned),
                Err(shared) if began.elapsed() < std::time::Duration::from_secs(10) => {
                    engine = shared;
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                Err(_) => break None,
            }
        };
        if let Some(engine) = owned {
            if began.elapsed() > std::time::Duration::from_millis(5) {
                tracing::debug!("Straydog: the engine was let go of after {} ms", began.elapsed().as_millis());
            }
            engine.close().await;
        } else {
            tracing::warn!("Straydog: something still holds the engine after 10 s, so it is not closed in order");
        }
        served?;
        let failure = failure.lock().unwrap().take();
        failure.map_or(Ok(()), Err)
    })
}

/// Resolves when the server is to stop on purpose: Ctrl+C, SIGTERM (Linux), the console being closed or Windows ending the session, or a line `stop`
/// on standard input (with `--stop-on-stdin`, for a program that starts the server and has no console to send it Ctrl+C with: a pipe only that program
/// holds). The conversation the engine holds is kept at such a stop (`keep_conversation`).
async fn shutdown(stop_on_stdin: bool) {
    let interrupt = tokio::signal::ctrl_c();
    let asked = stop_asked(stop_on_stdin);
    #[cfg(unix)]
    {
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
        tokio::select! { _ = interrupt => {}, _ = terminate.recv() => {}, () = asked => {} }
    }
    #[cfg(windows)]
    {
        tokio::select! { _ = interrupt => {}, () = windows_ends() => {}, () = asked => {} }
    }
    #[cfg(not(any(unix, windows)))]
    tokio::select! { _ = interrupt => {}, () = asked => {} }
    tracing::info!("Straydog stopping");
}

/// Ctrl+Break, the console closing, the user logging off and Windows shutting down: each ends the process when its handler returns, which tokio holds off
/// until the server has gone (Windows gives a console process a few seconds more). Never, where none of them can be listened for.
#[cfg(windows)]
async fn windows_ends() {
    use tokio::signal::windows::{ctrl_break, ctrl_close, ctrl_logoff, ctrl_shutdown};
    let (Ok(mut brk), Ok(mut close), Ok(mut logoff), Ok(mut shutdown)) = (ctrl_break(), ctrl_close(), ctrl_logoff(), ctrl_shutdown()) else {
        return std::future::pending().await;
    };
    tokio::select! { _ = brk.recv() => {}, _ = close.recv() => {}, _ = logoff.recv() => {}, _ = shutdown.recv() => {} }
}

/// Resolves when a line `stop` arrives on standard input, if `enabled`; never otherwise, and never when the input ends (a program that is gone takes
/// the engine with it by other means). Nothing else on standard input is read as anything.
async fn stop_asked(enabled: bool) {
    if !enabled {
        return std::future::pending().await;
    }
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let spawned = std::thread::Builder::new().name("straydog-stop".into()).spawn(move || {
        if stop_is_asked(std::io::stdin().lock()) {
            let _ = sender.send(());
        }
    });
    if spawned.is_err() || receiver.await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// Whether a line `stop` comes before `input` ends.
fn stop_is_asked(input: impl std::io::BufRead) -> bool {
    input.lines().map_while(Result::ok).any(|line| line.trim().eq_ignore_ascii_case("stop"))
}

#[cfg(test)]
mod tests {
    use super::{drafter_phrase, stop_is_asked};

    /// A line `stop` on the input stops the server, whatever comes before it and after its line end; an input that ends without one does not.
    #[test]
    fn a_line_stop_on_the_input_asks_for_the_stop() {
        assert!(stop_is_asked(&b"stop\n"[..]));
        assert!(stop_is_asked(&b"hello\r\n  Stop \r\nmore\n"[..]));
        assert!(stop_is_asked(&b"stop"[..]), "a last line with no line end");
        assert!(!stop_is_asked(&b""[..]), "an input that ends: the program is gone");
        assert!(!stop_is_asked(&b"stopping\nhalt\n"[..]));
        assert!(!stop_is_asked(&b"\xff\xfe not text, not stop\n"[..]), "bytes that are no text end the reading");
    }

    /// The starting line names the drafter that will be used, and never says there is none when
    /// DSpark's file is there.
    #[test]
    fn the_starting_line_names_the_drafter_or_nothing() {
        assert_eq!(drafter_phrase(Some(3), false), ", MTP drafter with 3 drafts");
        assert_eq!(drafter_phrase(None, true), ", DSpark drafter");
        assert_eq!(drafter_phrase(None, false), "");
    }
}
