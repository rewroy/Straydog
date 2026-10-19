//! The `straydog` command: the server, its isolated worker, and provisioning.

mod serve;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use straydog::{config, init, worker};

#[cfg(not(any(feature = "hip", feature = "cuda")))]
compile_error!("the straydog executable links one GPU backend: build it with --features hip or --features cuda");

// Prompt rows and logits are large host buffers: back them with huge pages.
#[global_allocator]
static ALLOCATOR: straydog_engine::alloc::HugePages = straydog_engine::alloc::HugePages;

#[derive(Parser)]
#[command(name = "straydog", version, about = "A local inference engine with an OpenAI-compatible server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the OpenAI-compatible API
    Serve {
        #[command(flatten)]
        options: config::Options,
        /// The same settings as a file [default: ~/.straydog/config.toml when present]
        #[arg(long, env = "STRAYDOG_CONFIG", value_name = "FILE")]
        config: Option<PathBuf>,
        /// Show the settings in effect and where each came from, then exit
        #[arg(long)]
        print_config: bool,
    },
    /// Set up: choose a bundle and GPUs, fetch the model and drafter, write the config
    Init(init::InitArgs),
    /// Download and authenticate a model bundle
    Fetch {
        /// The bundle directory
        #[arg(short = 'm', long, value_name = "DIR")]
        model: PathBuf,
        /// Q2_0 (66.4 GB), IQ2_XS (68.0 GB), IQ3_XXS (75.8 GB) or IQ3_S (83.6 GB)
        #[arg(long, value_name = "NAME")]
        quant: String,
        /// Authenticate what is there; download nothing
        #[arg(long)]
        offline: bool,
    },
    /// Authenticate a model bundle
    Verify {
        #[arg(short = 'm', long, value_name = "DIR")]
        model: PathBuf,
        #[arg(long, value_name = "NAME")]
        quant: String,
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

fn progress(stage: &str, done: u64, total: u64) {
    if total > 0 {
        eprint!("\r{stage}: {:.1}/{:.1} GB", done as f64 / 1e9, total as f64 / 1e9);
        if done >= total {
            eprintln!();
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse_from(config::expand_short_options(std::env::args_os()));
    let result = match cli.command {
        Command::Serve { .. } => {
            let matches = <Cli as clap::CommandFactory>::command()
                .get_matches_from(config::expand_short_options(std::env::args_os()));
            let (_, matches) = matches.subcommand().expect("serve");
            let Command::Serve { config, print_config, .. } = cli.command else { unreachable!() };
            serve::main(matches, config.as_deref(), print_config)
        }
        Command::Init(args) => init::main(args),
        Command::Fetch { model, quant, offline } => straydog_provision::fetch(&model, &quant, offline, &progress),
        Command::Verify { model, quant } => straydog_provision::verify(&model, &quant, &progress),
        Command::FetchMtp { directory, keep_source } => {
            straydog_provision::fetch_drafter(&directory, keep_source, &progress)
                .map(|path| println!("verified {}", path.display()))
        }
        Command::Worker(args) => worker::main(args),
        #[cfg(windows)]
        Command::Contain { args } => return ExitCode::from(straydog_isolation::launcher_main(&args) as u8),
        #[cfg(not(windows))]
        Command::Contain { .. } => Err("contain is the Windows AppContainer launcher".into()),
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
