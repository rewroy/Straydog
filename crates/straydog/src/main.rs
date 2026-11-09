//! The `straydog` command: the server (with no subcommand), its isolated worker, and provisioning.

mod serve;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use straydog::{config, progress, setup, worker};

#[cfg(not(any(feature = "hip", feature = "cuda")))]
compile_error!("the straydog executable links one GPU backend: build it with --features hip or --features cuda");

/// With no subcommand, `straydog` serves the API with the given settings, as llama-server does.
#[derive(Parser)]
#[command(name = "straydog", version, about = "A local inference engine with an OpenAI-compatible server",
          after_help = "Getting started:\n  straydog setup   choose a model and where it goes, download it and write the settings\n  straydog         serve it on http://127.0.0.1:8080 (with no --model, the settings setup wrote)",
          args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(flatten)]
    options: config::Options,
    /// Read settings from a TOML file (the option names, with underscores)
    #[arg(long, env = "STRAYDOG_CONFIG", value_name = "FILE")]
    config: Option<PathBuf>,
    /// Show the settings in effect and where each came from, then exit
    #[arg(long)]
    print_config: bool,
    /// Say whether the model the settings name loads as it is (its pinned files and verity trees), then exit:
    /// 0 when it does, 1 with the reason. Nothing is hashed, written or downloaded
    #[arg(long, conflicts_with = "print_config")]
    check_model: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Choose a model and where it goes, download it and write the settings a plain `straydog` reads
    Setup(Box<setup::SetupArgs>),
    /// Download and authenticate a model bundle
    Fetch {
        /// The bundle directory
        #[arg(short = 'm', long, value_name = "DIR")]
        model: PathBuf,
        /// The bundle's model: Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash-1017 (needed where another model has a bundle of the provider and quantization given)
        #[arg(long = "model-name", value_name = "NAME")]
        model_name: Option<String>,
        /// The bundle's provider: ISTA-DASLab or Unsloth (needed where another provider has a bundle of the model and quantization given)
        #[arg(long, value_name = "NAME")]
        provider: Option<String>,
        /// Qwen3.8-Flash-Next: ISTA-DASLab's Q2_0 (66.4 GB), IQ2_XS (68.0 GB), IQ3_XXS (75.8 GB), IQ3_S (83.6 GB), Unsloth's UD-IQ1_M (68.0 GB); DeepSeek-V4.1-Flash-1017: ISTA-DASLab's IQ2_XXS (248.8 GB), IQ2_XS (265.8 GB), Unsloth's UD-IQ1_M (250.4 GB), UD-IQ2_XXS (300.5 GB). `straydog setup` chooses for you
        #[arg(long, value_name = "NAME")]
        quant: String,
        /// Authenticate what is there; download nothing
        #[arg(long)]
        offline: bool,
        /// Where the shards' verity trees go [default: .straydog in the bundle directory]
        #[arg(long, env = "STRAYDOG_DATA_DIR", value_name = "DIR")]
        data_dir: Option<PathBuf>,
    },
    /// Authenticate a model bundle and build its shards' verity trees
    Verify {
        #[arg(short = 'm', long, value_name = "DIR")]
        model: PathBuf,
        #[arg(long = "model-name", value_name = "NAME")]
        model_name: Option<String>,
        #[arg(long, value_name = "NAME")]
        provider: Option<String>,
        #[arg(long, value_name = "NAME")]
        quant: String,
        /// Where the shards' verity trees go [default: .straydog in the bundle directory]
        #[arg(long, env = "STRAYDOG_DATA_DIR", value_name = "DIR")]
        data_dir: Option<PathBuf>,
    },
    /// Build the pinned MTP drafter from the official tensors
    FetchMtp {
        /// The drafter directory
        #[arg(long = "model-draft", value_name = "DIR")]
        directory: PathBuf,
        /// Keep the fetched BF16 tensors
        #[arg(long)]
        keep_source: bool,
    },
    /// The isolated engine (started by the server)
    #[command(hide = true)]
    Worker(Box<worker::WorkerArgs>),
    /// Windows: start the worker in its AppContainer (started by the server)
    #[command(hide = true)]
    Contain {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<OsString>,
    },
}

fn main() -> ExitCode {
    // (Windows: a process that dies of an exception says where first, on stderr, for whoever runs it)
    straydog::crash::report_unhandled();
    let args = config::expand_short_options(std::env::args_os());
    let matches = <Cli as clap::CommandFactory>::command().get_matches_from(args);
    let cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
    let result = match cli.command {
        None => {
            let file = cli.config.clone().or_else(|| config::implicit_config(&cli.options, config::default_config_path()));
            serve::main(&matches, file.as_deref(), cli.print_config, cli.check_model)
        }
        Some(Command::Setup(args)) => setup::main(&args),
        Some(Command::Fetch { model, model_name, provider, quant, offline, data_dir }) => {
            straydog_engine::bundle::Id::find(model_name.as_deref(), provider.as_deref(), &quant)
                .and_then(|id| straydog_provision::fetch(&model, id, offline, data_dir.as_deref(), &progress::report))
        }
        Some(Command::Verify { model, model_name, provider, quant, data_dir }) => {
            // (a bundle there already: the labels it fits that leave others open name the one whose shards are there)
            config::find_bundle(&model, model_name.as_deref(), provider.as_deref(), Some(&quant))
                .and_then(|id| straydog_provision::verify(&model, id, data_dir.as_deref(), &progress::report))
        }
        Some(Command::FetchMtp { directory, keep_source }) => {
            straydog_provision::fetch_drafter(&directory, keep_source, &progress::report)
                .map(|path| println!("verified {}", path.display()))
        }
        Some(Command::Worker(args)) => worker::main(*args),
        #[cfg(windows)]
        Some(Command::Contain { args }) => return ExitCode::from(straydog_isolation::launcher_main(&args) as u8),
        #[cfg(not(windows))]
        Some(Command::Contain { .. }) => Err("contain is the Windows AppContainer launcher".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            // The server reports the worker's last stderr line as the start failure.
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
