//! `straydog init`: choose a bundle and GPUs, fetch and authenticate the model and the
//! drafter, write the config (and an API key), and optionally install a systemd user service.
//! Every question has a flag, so it also runs unattended (`--yes`).

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use straydog_engine::bundle;

use crate::config::straydog_home;
use crate::progress;

#[derive(Debug, Clone, Args)]
pub struct InitArgs {
    /// Q2_0, IQ2_XS, IQ3_XXS or IQ3_S
    #[arg(long, value_name = "NAME")]
    pub quant: Option<String>,
    /// The bundle directory [default: ~/.straydog/models/<quant>]
    #[arg(short = 'm', long, value_name = "DIR")]
    pub model: Option<PathBuf>,
    /// GPU ordinals, comma separated [default: every GPU found]
    #[arg(long, value_delimiter = ',', value_name = "LIST")]
    pub device: Option<Vec<usize>>,
    /// VRAM budget per GPU in GiB [default: the smallest card less 2 GiB]
    #[arg(long, value_name = "GIB")]
    pub vram: Option<f64>,
    /// Context in tokens [default: 262144]
    #[arg(short = 'c', long, value_name = "N")]
    pub ctx_size: Option<u32>,
    /// No MTP drafter (no speculative decoding)
    #[arg(long)]
    pub no_drafter: bool,
    /// Listen port [default: 8080]
    #[arg(long, value_name = "PORT")]
    pub port: Option<u16>,
    /// No API key (any local process can then use the server)
    #[arg(long)]
    pub no_api_key: bool,
    /// Install and start a systemd user service
    #[arg(long)]
    pub service: bool,
    /// Authenticate what is present; download nothing
    #[arg(long)]
    pub offline: bool,
    /// Accept every default instead of asking
    #[arg(short = 'y', long)]
    pub yes: bool,
}

struct Prompt {
    ask: bool,
}

impl Prompt {
    fn text(&self, question: &str, default: &str) -> Result<String, String> {
        if !self.ask {
            return Ok(default.to_string());
        }
        print!("{question} [{default}]: ");
        std::io::stdout().flush().map_err(|exc| exc.to_string())?;
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).map_err(|exc| exc.to_string())?;
        let line = line.trim();
        Ok(if line.is_empty() { default.to_string() } else { line.to_string() })
    }

    fn yes(&self, question: &str, default: bool) -> Result<bool, String> {
        let answer = self.text(question, if default { "yes" } else { "no" })?;
        Ok(matches!(answer.to_lowercase().as_str(), "y" | "yes"))
    }
}

/// Each bundle with its size in GB and what sets it apart.
const BUNDLES: [(&str, f64, &str); 4] = [("Q2_0", 66.4, "the fastest"), ("IQ2_XS", 68.0, ""), ("IQ3_XXS", 75.8, ""),
                                         ("IQ3_S", 83.6, "closest to BF16")];


/// A fresh API key: 32 random bytes as hex.
fn new_key() -> Result<String, String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|exc| exc.to_string())?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Write a file only its owner can read.
fn write_private(path: &Path, text: &str) -> Result<(), String> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
    file.write_all(text.as_bytes()).map_err(|exc| exc.to_string())
}

pub fn main(args: InitArgs) -> Result<(), String> {
    let home = straydog_home().ok_or("cannot find the home directory")?;
    std::fs::create_dir_all(&home).map_err(|exc| format!("{}: {exc}", home.display()))?;
    let prompt = Prompt { ask: !args.yes };
    let count = straydog_gpu::device_count().map_err(|error| format!("no GPU runtime: {error}"))?;
    if count == 0 {
        return Err("no GPU found".into());
    }
    let mut cards = Vec::new();
    for ordinal in 0..count {
        let (name, _free, total) = straydog_gpu::device_info(ordinal).map_err(|error| error.to_string())?;
        println!("GPU {ordinal}: {name}, {:.1} GiB", total as f64 / (1u64 << 30) as f64);
        cards.push(total);
    }
    let devices = match args.device {
        Some(devices) => devices,
        None => prompt
            .text("GPUs to use", &(0..count).map(|ordinal| ordinal.to_string()).collect::<Vec<_>>().join(","))?
            .split(',')
            .map(|ordinal| ordinal.trim().parse().map_err(|_| format!("invalid GPU ordinal {ordinal}")))
            .collect::<Result<_, _>>()?,
    };
    if devices.iter().any(|ordinal| *ordinal >= count) {
        return Err("a chosen GPU does not exist".into());
    }
    let smallest = devices.iter().map(|ordinal| cards[*ordinal]).min().unwrap() as f64 / (1u64 << 30) as f64;
    let vram = match args.vram {
        Some(vram) => vram,
        None => prompt.text("VRAM budget per GPU (GiB)", &format!("{}", (smallest - 2.0).floor().max(4.0)))?
            .parse().map_err(|_| "invalid VRAM budget")?,
    };
    // Full residency (and with it MTP) needs a bundle's size plus about a tenth across the GPUs;
    // the default is the best bundle that fits, else the smallest (partial residency).
    let budget_gb = vram * devices.len() as f64 * ((1u64 << 30) as f64 / 1e9);
    let fitting = BUNDLES.iter().rev().find(|(_, size, _)| size * 1.1 <= budget_gb);
    let default_quant = fitting.map_or("Q2_0", |(quant, _, _)| quant);
    let quant = match args.quant {
        Some(quant) => quant,
        None => {
            for (quant, size, note) in BUNDLES {
                println!("  {quant}: {size} GB{}", if note.is_empty() { String::new() } else { format!(", {note}") });
            }
            prompt.text("Quantization", default_quant)?
        }
    };
    bundle::bundle(&quant)?;
    let model = match args.model {
        Some(model) => model,
        None => PathBuf::from(prompt.text("Model directory", &home.join("models").join(&quant).display().to_string())?),
    };
    let ctx_size = match args.ctx_size {
        Some(ctx) => ctx,
        None => prompt.text("Context (tokens)", "262144")?.parse().map_err(|_| "invalid context")?,
    };
    let drafter = !args.no_drafter
        && prompt.yes("Speculative decoding (MTP drafter, ~0.9 GB, built once from ~5.2 GB of official tensors)?", true)?;
    let port = match args.port {
        Some(port) => port,
        None => prompt.text("Port", "8080")?.parse().map_err(|_| "invalid port")?,
    };
    let key = !args.no_api_key && prompt.yes("Require an API key?", true)?;

    std::fs::create_dir_all(&model).map_err(|exc| format!("{}: {exc}", model.display()))?;
    straydog_provision::fetch(&model, &quant, args.offline, &progress::report)?;
    let draft_directory = home.join("models").join("mtp");
    if drafter {
        std::fs::create_dir_all(&draft_directory).map_err(|exc| exc.to_string())?;
        straydog_provision::fetch_drafter(&draft_directory, false, &progress::report)?;
    }
    let mut config = toml::Table::new();
    let text = |path: &Path| toml::Value::String(path.display().to_string());
    config.insert("model".into(), text(&model.canonicalize().map_err(|exc| exc.to_string())?));
    config.insert("quant".into(), quant.clone().into());
    config.insert("device".into(), toml::Value::Array(devices.iter().map(|ordinal| toml::Value::Integer(*ordinal as i64)).collect()));
    config.insert("vram".into(), vram.into());
    config.insert("ctx_size".into(), i64::from(ctx_size).into());
    config.insert("port".into(), i64::from(port).into());
    if drafter {
        config.insert("model_draft".into(), text(&draft_directory));
    }
    if key {
        let key_file = home.join("api-key");
        if !key_file.is_file() {
            write_private(&key_file, &new_key()?)?;
        }
        config.insert("api_key_file".into(), text(&key_file));
        println!("API key: {} (send it as Authorization: Bearer <key>)", key_file.display());
    }
    let config = toml::to_string(&config).map_err(|exc| exc.to_string())?;
    let config_path = home.join("config.toml");
    if config_path.is_file() && !prompt.yes(&format!("Replace {}?", config_path.display()), true)? {
        return Err("kept the existing config".into());
    }
    std::fs::write(&config_path, config).map_err(|exc| format!("{}: {exc}", config_path.display()))?;
    println!("wrote {}", config_path.display());
    if cfg!(target_os = "linux") && (args.service || prompt.yes("Install a systemd user service?", false)?) {
        install_service()?;
    } else {
        println!("start it with: straydog serve");
    }
    Ok(())
}

fn install_service() -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|exc| exc.to_string())?;
    let directory = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .ok_or("cannot find the config directory")?
        .join("systemd/user");
    std::fs::create_dir_all(&directory).map_err(|exc| exc.to_string())?;
    let unit = format!(
        "[Unit]\nDescription=Straydog OpenAI-compatible inference server\nAfter=network.target\n\n[Service]\n\
         ExecStart={} serve\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        executable.display());
    let path = directory.join("straydog.service");
    std::fs::write(&path, unit).map_err(|exc| format!("{}: {exc}", path.display()))?;
    for args in [&["--user", "daemon-reload"][..], &["--user", "enable", "--now", "straydog.service"]] {
        let status = std::process::Command::new("systemctl").args(args).status().map_err(|exc| exc.to_string())?;
        if !status.success() {
            return Err(format!("systemctl {} failed", args.join(" ")));
        }
    }
    println!("installed and started {}", path.display());
    Ok(())
}
