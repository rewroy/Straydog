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
            let (failed, failure) = (engine.failed(), failure.clone());
            async move {
                tokio::select! {
                    () = shutdown() => {}
                    message = failed => *failure.lock().unwrap() = Some(message),
                }
            }
        };
        let served = axum::serve(listener, router).with_graceful_shutdown(stop).await.map_err(|exc| exc.to_string());
        monitor.stop().await;
        drop(monitor);
        // The router and the sampler are gone; the worker exits, and its GPU memory is free,
        // before the server does.
        if let Some(engine) = Arc::into_inner(engine) {
            engine.close().await;
        }
        served?;
        let failure = failure.lock().unwrap().take();
        failure.map_or(Ok(()), Err)
    })
}

async fn shutdown() {
    let interrupt = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
        tokio::select! { _ = interrupt => {}, _ = terminate.recv() => {} }
    }
    #[cfg(not(unix))]
    let _ = interrupt.await;
    tracing::info!("Straydog stopping");
}

#[cfg(test)]
mod tests {
    use super::drafter_phrase;

    /// The starting line names the drafter that will be used, and never says there is none when
    /// DSpark's file is there.
    #[test]
    fn the_starting_line_names_the_drafter_or_nothing() {
        assert_eq!(drafter_phrase(Some(3), false), ", MTP drafter with 3 drafts");
        assert_eq!(drafter_phrase(None, true), ", DSpark drafter");
        assert_eq!(drafter_phrase(None, false), "");
    }
}
