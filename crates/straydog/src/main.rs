//! The `straydog` command: the server (with no subcommand), its isolated worker, and provisioning.

mod serve;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use straydog::{config, progress, worker};

#[cfg(not(any(feature = "hip", feature = "cuda")))]
compile_error!("the straydog executable links one GPU backend: build it with --features hip or --features cuda");

// Prompt rows and logits are large host buffers: back them with huge pages.
#[global_allocator]
static ALLOCATOR: straydog_engine::alloc::HugePages = straydog_engine::alloc::HugePages;

/// With no subcommand, `straydog` serves the API with the given settings, as llama-server does.
#[derive(Parser)]
#[command(name = "straydog", version, about = "A local inference engine with an OpenAI-compatible server",
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
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Download and authenticate a model bundle
    Fetch {
        /// The bundle directory
        #[arg(short = 'm', long, value_name = "DIR")]
        model: PathBuf,
        /// Q2_0 (66.4 GB), IQ2_XS (68.0 GB), IQ3_XXS (75.8 GB), IQ3_S (83.6 GB), UD-IQ2_XXS (DeepSeek-V4.1-Flash, 300.5 GB), UD-Q8_K_XL (Qwen3.6-35B-A3B, 38.5 GB) or Q8_0 (Ornith-1.5-122B, 132.6 GB)
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
    Worker(worker::WorkerArgs),
    /// Windows: start the worker in its AppContainer (started by the server)
    #[command(hide = true)]
    Contain {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<OsString>,
    },
}

fn main() -> ExitCode {
    let args = config::expand_short_options(std::env::args_os());
    let matches = <Cli as clap::CommandFactory>::command().get_matches_from(args);
    let cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
    let result = match cli.command {
        None => serve::main(&matches, cli.config.as_deref(), cli.print_config),
        Some(Command::Fetch { model, quant, offline, data_dir }) => {
            straydog_provision::fetch(&model, &quant, offline, data_dir.as_deref(), &progress::report)
        }
        Some(Command::Verify { model, quant, data_dir }) => {
            straydog_provision::verify(&model, &quant, data_dir.as_deref(), &progress::report)
        }
        Some(Command::FetchMtp { directory, keep_source }) => {
            straydog_provision::fetch_drafter(&directory, keep_source, &progress::report)
                .map(|path| println!("verified {}", path.display()))
        }
        Some(Command::Worker(args)) => worker::main(args),
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
