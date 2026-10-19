//! The server: settings, logging, the runtime and the HTTP listener.

use std::path::Path;
use std::sync::Arc;

use clap::ArgMatches;
use straydog_api::ReasoningEffort;

use straydog::config::{self, Effort};
use straydog::runtime::{self, Runtime};

fn logging(settings: &config::Settings) -> Result<(), String> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(if settings.verbose { "debug" } else { "info" }));
    let builder = tracing_subscriber::fmt().with_env_filter(filter).with_target(false);
    match &settings.log_file {
        Some(path) => {
            let file = std::fs::OpenOptions::new().create(true).append(true).open(path)
                .map_err(|exc| format!("{}: {exc}", path.display()))?;
            builder.with_ansi(false).with_writer(std::sync::Mutex::new(file)).init();
        }
        None => builder.with_writer(std::io::stderr).init(),
    }
    Ok(())
}

pub fn main(matches: &ArgMatches, config_file: Option<&Path>, print_config: bool) -> Result<(), String> {
    let (options, sources) = config::load(matches, config_file)?;
    let settings = config::resolve(options)?;
    if print_config {
        for line in config::describe(&settings, &sources) {
            println!("{line}");
        }
        return Ok(());
    }
    logging(&settings)?;
    if let Some(reason) = straydog_isolation::unavailable_reason() {
        return Err(reason);
    }
    let command = straydog_isolation::worker_command(&runtime::worker_args(&settings), &runtime::exposure(&settings)?)?;
    let tokio = tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|exc| exc.to_string())?;
    tokio.block_on(async move {
        let engine = Arc::new(Runtime::start(&settings, command));
        let api = straydog_api::ApiConfig {
            model_id: settings.alias.clone(),
            context: u64::from(settings.ctx_size),
            quantization: settings.quant.clone(),
            api_key: settings.api_key.clone(),
            temperature: settings.temp,
            seed: settings.seed,
            reasoning_effort: match settings.reasoning_effort {
                Effort::None => ReasoningEffort::None,
                Effort::Low => ReasoningEffort::Low,
                Effort::Medium => ReasoningEffort::Medium,
                Effort::High => ReasoningEffort::High,
            },
            created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |time| time.as_secs()),
        };
        let router = straydog_api::router(engine.clone(), api);
        let address = format!("{}:{}", settings.host, settings.port);
        let listener = tokio::net::TcpListener::bind(&address).await.map_err(|exc| format!("{address}: {exc}"))?;
        tracing::info!("Straydog listening on http://{address}");
        let served = axum::serve(listener, router).with_graceful_shutdown(shutdown()).await.map_err(|exc| exc.to_string());
        // The router is gone; the worker exits before the server does.
        if let Some(engine) = Arc::into_inner(engine) {
            engine.close().await;
        }
        served
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
