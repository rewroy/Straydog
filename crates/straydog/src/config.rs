//! `straydog serve` settings: one definition for the options, the `STRAYDOG_*` variables and
//! the config file. Precedence: options, then variables, then the file, then defaults;
//! settings left unset are derived from the others and from the machine.

use std::path::{Path, PathBuf};

use clap::parser::ValueSource;
use clap::{ArgMatches, Args, FromArgMatches, ValueEnum};
use serde::Deserialize;
use straydog_engine::bundle::{self, QUANTIZATIONS};
use straydog_engine::model::context_state_bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    Layer,
    Experts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gemm {
    Fp32,
    Fp16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    None,
    Low,
    Medium,
    High,
}

/// Every setting of `straydog serve`. Field names are the long options with underscores;
/// the config file uses the same names.
#[derive(Debug, Clone, Default, Args, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    /// Model bundle directory (the pinned GGUF shards and tokenizer files of one quantization)
    #[arg(short = 'm', long, env = "STRAYDOG_MODEL", value_name = "DIR")]
    pub model: Option<PathBuf>,
    /// The bundle's quantization: Q2_0, IQ2_XS, IQ3_XXS or IQ3_S [default: detected from its files]
    #[arg(long, env = "STRAYDOG_QUANT", value_name = "NAME")]
    pub quant: Option<String>,
    /// MTP drafter directory
    #[arg(long = "model-draft", env = "STRAYDOG_MODEL_DRAFT", value_name = "DIR")]
    pub model_draft: Option<PathBuf>,
    /// MTP drafts per pass, 0 turns speculation off [default: 3 with a drafter]
    #[arg(long, env = "STRAYDOG_DRAFT_MAX", value_name = "N", value_parser = clap::value_parser!(u8).range(0..=3))]
    pub draft_max: Option<u8>,
    /// Context in tokens [default: 262144]
    #[arg(short = 'c', long, env = "STRAYDOG_CTX_SIZE", value_name = "N",
          value_parser = clap::value_parser!(u32).range(1..=262144))]
    pub ctx_size: Option<u32>,
    /// Prefill batch [default: 1024, 4096 with 48 GiB or more of GPU budget]
    #[arg(short = 'b', long, env = "STRAYDOG_BATCH_SIZE", value_name = "N",
          value_parser = clap::value_parser!(u16).range(1..=4096))]
    pub batch_size: Option<u16>,
    /// GPU ordinals, comma separated [default: 0]
    #[arg(long = "device", env = "STRAYDOG_DEVICE", value_name = "LIST", value_delimiter = ',')]
    pub device: Option<Vec<usize>>,
    /// With two GPUs holding the whole model: experts splits every layer's experts, layer splits the layers [default: experts]
    #[arg(long = "split-mode", env = "STRAYDOG_SPLIT_MODE", value_name = "MODE")]
    pub split_mode: Option<SplitMode>,
    /// VRAM budget per GPU in GiB [default: 6]
    #[arg(long, env = "STRAYDOG_VRAM", value_name = "GIB")]
    pub vram: Option<f64>,
    /// Host memory for sequence state in GiB [default: what the context needs]
    #[arg(long, env = "STRAYDOG_STATE_RAM", value_name = "GIB")]
    pub state_ram: Option<f64>,
    /// Input precision of large matrix products [default: fp16]
    #[arg(long, env = "STRAYDOG_GEMM", value_name = "PRECISION")]
    pub gemm: Option<Gemm>,
    /// Smallest batch computed with GEMM [default: 2]
    #[arg(long, env = "STRAYDOG_GEMM_MIN_BATCH", value_name = "N", value_parser = clap::value_parser!(u16).range(2..=257))]
    pub gemm_min_batch: Option<u16>,
    /// Listen address [default: 127.0.0.1]
    #[arg(long, env = "STRAYDOG_HOST", value_name = "ADDRESS")]
    pub host: Option<String>,
    /// Listen port [default: 8080]
    #[arg(long, env = "STRAYDOG_PORT", value_name = "PORT")]
    pub port: Option<u16>,
    /// Bearer key the server then requires [default: none required]
    #[arg(long, env = "STRAYDOG_API_KEY", value_name = "KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// File holding the API key, kept out of the process list
    #[arg(long, env = "STRAYDOG_API_KEY_FILE", value_name = "FILE")]
    pub api_key_file: Option<PathBuf>,
    /// Model id in /v1/models and answers [default: Qwen3.8-Flash-Next-<quant>]
    #[arg(long, env = "STRAYDOG_ALIAS", value_name = "NAME")]
    pub alias: Option<String>,
    /// Temperature when a request gives none [default: 0]
    #[arg(long, env = "STRAYDOG_TEMP", value_name = "T")]
    pub temp: Option<f64>,
    /// Sampling seed when a request gives none [default: 0]
    #[arg(long, env = "STRAYDOG_SEED", value_name = "N")]
    pub seed: Option<u64>,
    /// Reasoning when a request gives none; high is the template's own default [default: high]
    #[arg(long, env = "STRAYDOG_REASONING_EFFORT", value_name = "EFFORT")]
    pub reasoning_effort: Option<Effort>,
    /// Seconds the engine may take to load [default: 180]
    #[arg(long, env = "STRAYDOG_STARTUP_TIMEOUT", value_name = "S")]
    pub startup_timeout: Option<f64>,
    /// Log file [default: stderr]
    #[arg(long, env = "STRAYDOG_LOG_FILE", value_name = "FILE")]
    pub log_file: Option<PathBuf>,
    /// Log the worker's own output
    #[arg(short = 'v', long, env = "STRAYDOG_VERBOSE")]
    #[serde(default)]
    pub verbose: bool,
}

/// The settings in effect, every derived value filled in.
#[derive(Debug, Clone)]
pub struct Settings {
    pub model: PathBuf,
    pub quant: String,
    pub model_draft: Option<PathBuf>,
    pub draft_max: u8,
    pub ctx_size: u32,
    pub batch_size: u16,
    pub device: Vec<usize>,
    pub split_mode: SplitMode,
    pub vram: f64,
    pub state_ram: f64,
    pub gemm: Gemm,
    pub gemm_min_batch: u16,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub alias: String,
    pub temp: f64,
    pub seed: u64,
    pub reasoning_effort: Effort,
    pub startup_timeout: f64,
    pub log_file: Option<PathBuf>,
    pub verbose: bool,
}

pub fn straydog_home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(|home| PathBuf::from(home).join(".straydog"))
}

/// llama.cpp's multi-letter short options, as their long forms (clap's short options are one letter).
pub fn expand_short_options(args: impl IntoIterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    args.into_iter()
        .map(|arg| {
            let text = arg.to_string_lossy();
            for (short, long) in [("-md", "--model-draft"), ("-dev", "--device"), ("-sm", "--split-mode")] {
                if text == short {
                    return long.into();
                }
                if let Some(value) = text.strip_prefix(short).and_then(|rest| rest.strip_prefix('=')) {
                    return format!("{long}={value}").into();
                }
            }
            arg
        })
        .collect()
}

/// Where each setting came from, for `--print-config`.
pub type Sources = Vec<(&'static str, String)>;

/// Merge the command line (with its variables) over the config file.
pub fn load(matches: &ArgMatches, config: Option<&Path>) -> Result<(Options, Sources), String> {
    let given = Options::from_arg_matches(matches).map_err(|exc| exc.to_string())?;
    let path = config.map(Path::to_path_buf).or_else(|| straydog_home().map(|home| home.join("config.toml")).filter(|path| path.is_file()));
    let file: Options = match &path {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
            toml::from_str(&text).map_err(|exc| format!("{}: {exc}", path.display()))?
        }
        None => Options::default(),
    };
    let mut sources = Sources::new();
    macro_rules! merge {
        ($($field:ident),*) => {{
            let mut options = Options::default();
            $(
                let name = stringify!($field);
                let id = name;
                options.$field = if !is_unset(&given.$field) {
                    let source = match matches.value_source(id) {
                        Some(ValueSource::EnvVariable) => format!("env STRAYDOG_{}", name.to_uppercase()),
                        _ => format!("option --{}", name.replace('_', "-")),
                    };
                    sources.push((name, source));
                    given.$field.clone()
                } else if !is_unset(&file.$field) {
                    sources.push((name, format!("config {}", path.as_ref().unwrap().display())));
                    file.$field.clone()
                } else {
                    given.$field.clone()
                };
            )*
            options
        }};
    }
    let options = merge!(model, quant, model_draft, draft_max, ctx_size, batch_size, device, split_mode, vram,
                         state_ram, gemm, gemm_min_batch, host, port, api_key, api_key_file, alias, temp, seed,
                         reasoning_effort, startup_timeout, log_file, verbose);
    Ok((options, sources))
}

trait Unset {
    fn unset(&self) -> bool;
}

impl<T> Unset for Option<T> {
    fn unset(&self) -> bool {
        self.is_none()
    }
}

impl Unset for bool {
    fn unset(&self) -> bool {
        !*self
    }
}

fn is_unset(value: &impl Unset) -> bool {
    value.unset()
}

/// The one quantization whose pinned shards are all in `model`.
pub fn detect_quantization(model: &Path) -> Result<String, String> {
    let found: Vec<&str> = QUANTIZATIONS
        .iter()
        .copied()
        .filter(|quant| {
            bundle::bundle(quant).is_ok_and(|pins| pins.shards.iter().all(|shard| model.join(&shard.name).is_file()))
        })
        .collect();
    match found[..] {
        [quant] => Ok(quant.to_string()),
        [] => Err(format!("{} holds no pinned model bundle; give --quant", model.display())),
        _ => Err(format!("{} holds more than one bundle ({}); give --quant", model.display(), found.join(", "))),
    }
}

/// Fill every derived setting.
pub fn resolve(options: Options) -> Result<Settings, String> {
    let model = options.model.ok_or("no model: give -m DIR or set model in the config file")?;
    let model = model.canonicalize().map_err(|exc| format!("model directory {}: {exc}", model.display()))?;
    if !model.is_dir() {
        return Err(format!("model directory not found: {}", model.display()));
    }
    let quant = match options.quant {
        Some(quant) if QUANTIZATIONS.contains(&quant.as_str()) => quant,
        Some(quant) => return Err(format!("unknown quantization {quant}: Q2_0, IQ2_XS, IQ3_XXS or IQ3_S")),
        None => detect_quantization(&model)?,
    };
    let model_draft = match options.model_draft {
        Some(directory) => Some(directory.canonicalize().map_err(|exc| format!("drafter {}: {exc}", directory.display()))?),
        None => None,
    };
    let draft_max = options.draft_max.unwrap_or(if model_draft.is_some() { 3 } else { 0 });
    let device = options.device.unwrap_or_else(|| vec![0]);
    let mut distinct = device.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if device.is_empty() || device.len() > 4 || distinct.len() != device.len() {
        return Err("GPU ordinals must be one to four distinct numbers".into());
    }
    let vram = options.vram.unwrap_or(6.0);
    if !(0.01..=256.0).contains(&vram) {
        return Err("--vram must be between 0.01 and 256 GiB".into());
    }
    let ctx_size = options.ctx_size.unwrap_or(262144);
    let batch_size = options.batch_size.unwrap_or(if device.len() as f64 * vram >= 48.0 { 4096 } else { 1024 });
    // Enough for the context's state (K/V stays on the host when it exceeds a quarter of the
    // GPU budget), rounded up to half a GiB.
    let needed = (context_state_bytes(u64::from(ctx_size)) as f64 / (1u64 << 30) as f64 * 2.0).ceil() / 2.0;
    let state_ram = options.state_ram.unwrap_or(needed.max(1.0));
    if !(0.01..=256.0).contains(&state_ram) {
        return Err("--state-ram must be between 0.01 and 256 GiB".into());
    }
    let temp = options.temp.unwrap_or(0.0);
    if !(0.0..=2.0).contains(&temp) {
        return Err("--temp must be between 0 and 2".into());
    }
    let startup_timeout = options.startup_timeout.unwrap_or(180.0);
    if !(startup_timeout > 0.0 && startup_timeout <= 1800.0) {
        return Err("--startup-timeout must be between 0 and 1800 seconds".into());
    }
    if options.api_key.is_some() && options.api_key_file.is_some() {
        return Err("give either --api-key or --api-key-file, not both".into());
    }
    let api_key_file = match options.api_key_file {
        Some(file) => Some(file.canonicalize().map_err(|exc| format!("API key file {}: {exc}", file.display()))?),
        None => None,
    };
    let api_key = match (&options.api_key, &api_key_file) {
        (Some(key), _) => Some(key.clone()),
        (None, Some(file)) => {
            Some(std::fs::read_to_string(file).map_err(|exc| format!("{}: {exc}", file.display()))?.trim().to_string())
        }
        (None, None) => None,
    };
    if api_key.as_ref().is_some_and(String::is_empty) {
        return Err("the API key is empty".into());
    }
    let alias = options.alias.unwrap_or_else(|| format!("{}-{quant}", bundle::MODEL));
    Ok(Settings {
        model,
        quant,
        model_draft,
        draft_max,
        ctx_size,
        batch_size,
        device,
        split_mode: options.split_mode.unwrap_or(SplitMode::Experts),
        vram,
        state_ram,
        gemm: options.gemm.unwrap_or(Gemm::Fp16),
        gemm_min_batch: options.gemm_min_batch.unwrap_or(2),
        host: options.host.unwrap_or_else(|| "127.0.0.1".into()),
        port: options.port.unwrap_or(8080),
        api_key,
        api_key_file,
        alias,
        temp,
        seed: options.seed.unwrap_or(0),
        reasoning_effort: options.reasoning_effort.unwrap_or(Effort::High),
        startup_timeout,
        log_file: options.log_file,
        verbose: options.verbose,
    })
}

/// One line per setting: its value and where it came from (`derived` when filled in).
pub fn describe(settings: &Settings, sources: &Sources) -> Vec<String> {
    let path = |path: &Option<PathBuf>| path.as_ref().map(|path| path.display().to_string()).unwrap_or_default();
    let values: Vec<(&str, String)> = vec![
        ("model", settings.model.display().to_string()),
        ("quant", settings.quant.clone()),
        ("model_draft", path(&settings.model_draft)),
        ("draft_max", settings.draft_max.to_string()),
        ("ctx_size", settings.ctx_size.to_string()),
        ("batch_size", settings.batch_size.to_string()),
        ("device", settings.device.iter().map(usize::to_string).collect::<Vec<_>>().join(",")),
        ("split_mode", format!("{:?}", settings.split_mode).to_lowercase()),
        ("vram", settings.vram.to_string()),
        ("state_ram", settings.state_ram.to_string()),
        ("gemm", format!("{:?}", settings.gemm).to_lowercase()),
        ("gemm_min_batch", settings.gemm_min_batch.to_string()),
        ("host", settings.host.clone()),
        ("port", settings.port.to_string()),
        ("api_key", if settings.api_key.is_some() { "(set)".into() } else { String::new() }),
        ("api_key_file", path(&settings.api_key_file)),
        ("alias", settings.alias.clone()),
        ("temp", settings.temp.to_string()),
        ("seed", settings.seed.to_string()),
        ("reasoning_effort", format!("{:?}", settings.reasoning_effort).to_lowercase()),
        ("startup_timeout", settings.startup_timeout.to_string()),
        ("log_file", path(&settings.log_file)),
        ("verbose", settings.verbose.to_string()),
    ];
    values
        .into_iter()
        .map(|(name, value)| {
            let derived = ["quant", "draft_max", "batch_size", "state_ram", "alias"].contains(&name);
            let source = sources
                .iter()
                .find(|(key, _)| *key == name)
                .map_or(if derived { "derived" } else { "default" }, |(_, source)| source.as_str());
            format!("{name:<16} {value}  [{source}]")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Command, FromArgMatches};

    fn command() -> Command {
        Options::augment_args(Command::new("serve").no_binary_name(true))
    }

    /// A model directory holding (empty) files named like one bundle's shards.
    fn bundle_directory(quant: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("straydog-config-{quant}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for shard in bundle::bundle(quant).unwrap().shards {
            std::fs::write(directory.join(shard.name), b"").unwrap();
        }
        directory
    }

    #[test]
    fn options_beat_variables_beat_the_file_and_derived_values_follow() {
        let model = bundle_directory("IQ3_S");
        let file = model.join("config.toml");
        std::fs::write(&file, format!("model = {:?}\nvram = 20\nport = 9000\ndevice = [0, 1]\n", model.display())).unwrap();
        // SAFETY of the test: the only test that sets these variables.
        std::env::set_var("STRAYDOG_PORT", "9100");
        let matches = command().try_get_matches_from(["--vram", "30", "-c", "32768"]).unwrap();
        std::env::remove_var("STRAYDOG_PORT");
        let (options, sources) = load(&matches, Some(&file)).unwrap();
        let settings = resolve(options).unwrap();
        assert_eq!((settings.vram, settings.port, settings.device.clone()), (30.0, 9100, vec![0, 1]));
        assert_eq!(settings.quant, "IQ3_S");
        assert_eq!(settings.batch_size, 4096);   // 2 x 30 GiB of GPU budget
        assert_eq!(settings.ctx_size, 32768);
        assert_eq!(settings.state_ram, 2.0);     // what 32768 tokens of state need, rounded up
        assert_eq!(settings.alias, "Qwen3.8-Flash-Next-IQ3_S");
        assert_eq!(settings.draft_max, 0);
        let lines = describe(&settings, &sources);
        let line = |name: &str| lines.iter().find(|line| line.starts_with(&format!("{name} "))).unwrap().clone();
        assert!(line("vram").ends_with("[option --vram]"));
        assert!(line("port").ends_with("[env STRAYDOG_PORT]"));
        assert!(line("device").ends_with(&format!("[config {}]", file.display())));
        assert!(line("batch_size").ends_with("[derived]"));
        assert!(line("host").ends_with("[default]"));
        std::fs::write(&file, "unknown = 1\n").unwrap();
        assert!(load(&command().try_get_matches_from(Vec::<&str>::new()).unwrap(), Some(&file)).is_err());
        std::fs::remove_dir_all(model).unwrap();
    }

    #[test]
    fn bad_settings_are_refused() {
        let model = bundle_directory("Q2_0");
        let parse = |args: &[&str]| {
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let matches = command().try_get_matches_from(all).map_err(|exc| exc.to_string())?;
            resolve(Options::from_arg_matches(&matches).map_err(|exc| exc.to_string())?)
        };
        assert_eq!(parse(&[]).unwrap().quant, "Q2_0");
        assert!(parse(&["--quant", "Q9"]).is_err());
        assert!(parse(&["--device", "0,0"]).is_err());
        assert!(parse(&["--api-key", "k", "--api-key-file", "/etc/hostname"]).is_err());
        assert!(parse(&["--temp", "3"]).is_err());
        assert!(parse(&["-c", "300000"]).is_err());
        assert!(parse(&["--draft-max", "4"]).is_err());
        std::fs::remove_dir_all(model).unwrap();
    }

    #[test]
    fn llama_short_options_expand() {
        let expanded = expand_short_options(["-md", "/d", "-dev=0,1", "-sm", "layer", "-m", "x"].map(Into::into));
        assert_eq!(expanded, ["--model-draft", "/d", "--device=0,1", "--split-mode", "layer", "-m", "x"]
                   .map(std::ffi::OsString::from));
    }
}
