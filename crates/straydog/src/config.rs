//! The server's settings: one definition for the options, the `STRAYDOG_*` variables and
//! an optional config file. Precedence: options, then variables, then the file, then defaults;
//! settings left unset are derived from the others and from the machine.

use std::path::{Path, PathBuf};

use clap::parser::ValueSource;
use clap::{ArgMatches, Args, FromArgMatches, ValueEnum};
use serde::Deserialize;
use straydog_engine::bundle;
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

/// A VRAM budget for each GPU: its free memory less a margin (`auto`), or GiB, one value for
/// every GPU or one per GPU in `--gpu` order (`30,10` for a 32 GB and a 12 GB card).
#[derive(Debug, Clone, PartialEq)]
pub enum Vram {
    Auto,
    Gib(Vec<f64>),
}

impl std::str::FromStr for Vram {
    type Err = String;
    fn from_str(text: &str) -> Result<Vram, String> {
        if text.trim().eq_ignore_ascii_case("auto") {
            return Ok(Vram::Auto);
        }
        text.split(',')
            .map(|part| part.trim().parse::<f64>().map_err(|_| format!("invalid VRAM budget {text:?}: auto, GiB, or GiB per GPU (30,10)")))
            .collect::<Result<Vec<_>, _>>()
            .map(Vram::Gib)
    }
}

impl<'de> Deserialize<'de> for Vram {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vram, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Number(f64),
            Text(String),
            List(Vec<f64>),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Number(gib) => Ok(Vram::Gib(vec![gib])),
            Raw::Text(text) => text.parse().map_err(serde::de::Error::custom),
            Raw::List(gib) => Ok(Vram::Gib(gib)),
        }
    }
}

impl std::fmt::Display for Vram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vram::Auto => f.write_str("auto"),
            Vram::Gib(gib) => f.write_str(&gib.iter().map(f64::to_string).collect::<Vec<_>>().join(",")),
        }
    }
}

impl Vram {
    /// Each of `gpus` GPUs' budget in bytes, None for `auto` (the worker sizes it).
    pub fn bytes(&self, gpus: usize) -> Option<Vec<u64>> {
        match self {
            Vram::Auto => None,
            Vram::Gib(gib) => Some((0..gpus).map(|gpu| (gib[if gib.len() == 1 { 0 } else { gpu }] * (1u64 << 30) as f64) as u64).collect()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    None,
    Low,
    Medium,
    High,
}

/// Every setting of the server. Field names are the long options with underscores;
/// the config file uses the same names.
#[derive(Debug, Clone, Default, Args, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    /// Model bundle: its directory (the pinned GGUF shards and tokenizer files of one quantization) or one of its shards
    #[arg(short = 'm', long, env = "STRAYDOG_MODEL", value_name = "PATH")]
    pub model: Option<PathBuf>,
    /// The bundle's model: Qwen3.8-Flash-Next or DeepSeek-V4.1-Flash-1017 [default: detected from its files]
    #[arg(long = "model-name", env = "STRAYDOG_MODEL_NAME", value_name = "NAME")]
    pub model_name: Option<String>,
    /// The bundle's provider, who quantized the model: ISTA-DASLab or Unsloth [default: detected from its files]
    #[arg(long, env = "STRAYDOG_PROVIDER", value_name = "NAME")]
    pub provider: Option<String>,
    /// The bundle's quantization: Qwen3.8-Flash-Next's Q2_0, IQ2_XS, IQ3_XXS, IQ3_S (ISTA-DASLab) and UD-IQ1_M (Unsloth), DeepSeek-V4.1-Flash-1017's IQ2_XS, IQ2_XXS (ISTA-DASLab), UD-IQ2_XXS and UD-IQ1_M (Unsloth) [default: detected from its files]
    #[arg(long, env = "STRAYDOG_QUANT", value_name = "NAME")]
    pub quant: Option<String>,
    /// Unsloth's Qwen3.8 bundles: the engram table to read, BF16, Q8_0, Q5_1 or IQ4_NL (the bundle's own), from the bundle's folder or an engram folder beside it [default: the bundle's own, else the most exact there]
    #[arg(long, env = "STRAYDOG_ENGRAM", value_name = "PRECISION")]
    pub engram: Option<String>,
    /// Speed first for a partially resident model: a router weight from 0 to 1 below which a route the CPU would run is left out of its token (Qwen3.8 UD-IQ2_XS: 0.04 keeps the replies close, KL 0.002 against every route; 0.06 does not) [default: 0, every route]
    #[arg(long, env = "STRAYDOG_EXPERT_FLOOR", value_name = "WEIGHT")]
    pub expert_floor: Option<f32>,
    /// Qwen3.8-Flash-Next's MTP drafter: its directory or its GGUF file (also -md). DeepSeek-V4.1's DSpark drafter takes no option: it drafts when its files sit beside the bundle (`straydog setup` puts them there; DS41_DSPARK=0 leaves it out)
    #[arg(long = "model-draft", env = "STRAYDOG_MODEL_DRAFT", value_name = "PATH")]
    pub model_draft: Option<PathBuf>,
    /// Qwen3.8-Flash-Next's vision encoder and projector, mmproj-Qwen3.8-Flash-Next-BF16.gguf: it reads images with it [default: a mmproj-*.gguf in the model's directory or the one above it]. DeepSeek-V4.1 keeps its own beside its shards
    #[arg(long = "mmproj", env = "STRAYDOG_MODEL_VISION", value_name = "FILE")]
    pub model_vision: Option<PathBuf>,
    /// MTP drafts per pass, 0 turns MTP speculation off [default: 3 with a drafter]
    #[arg(long, env = "STRAYDOG_DRAFT_MAX", value_name = "N", value_parser = clap::value_parser!(u8).range(0..=3))]
    pub draft_max: Option<u8>,
    /// Context in tokens, at most the model's own: Qwen3.8-Flash-Next's 262144, DeepSeek-V4.1-Flash-1017's 1048576 [default: 262144]
    #[arg(short = 'c', long, env = "STRAYDOG_CTX_SIZE", value_name = "N")]
    pub ctx_size: Option<u32>,
    /// Prefill batch [default: 1024, 4096 with 48 GiB or more of GPU budget; DeepSeek-V4.1: the largest whose pass buffers take at most an eighth of a GPU's budget, 8192 down to 256, or on a card of 24 GiB or more a third of it, 32768 or 16384]
    #[arg(short = 'b', long, env = "STRAYDOG_BATCH_SIZE", value_name = "N",
          value_parser = clap::value_parser!(u16).range(1..=32768))]
    pub batch_size: Option<u16>,
    /// GPU ordinals, comma separated (also -gpu; -dev and --device are accepted) [default: 0]
    #[arg(long = "gpu", alias = "device", env = "STRAYDOG_GPU", value_name = "LIST", value_delimiter = ',')]
    #[serde(alias = "device")]
    pub gpu: Option<Vec<usize>>,
    /// With two GPUs holding the whole model: experts splits every layer's experts, layer splits the layers (also -sm) [default: experts]
    #[arg(long = "split-mode", env = "STRAYDOG_SPLIT_MODE", value_name = "MODE")]
    pub split_mode: Option<SplitMode>,
    /// VRAM budget per GPU in GiB: one value for every GPU, one per GPU (30,10), or auto, each GPU's free memory less a margin [default: auto]
    #[arg(long, env = "STRAYDOG_VRAM", value_name = "GIB")]
    pub vram: Option<Vram>,
    /// Host memory for sequence state in GiB [default: what the context needs]
    #[arg(long, env = "STRAYDOG_STATE_RAM", value_name = "GIB")]
    pub state_ram: Option<f64>,
    /// Host memory caching rows of the PLE and Engram tables, read from the SSD as needed, in MiB; 0 reads them through the page cache [default: 256; Qwen3.8's PLE table, 28.8 GB, all of it while half of the host memory left beside the sequence state holds it, else that half]
    #[arg(long, env = "STRAYDOG_ROW_CACHE", value_name = "MIB")]
    pub row_cache: Option<u32>,
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
    /// Model id in /v1/models and answers [default: <model>-<provider>-<quant>, e.g. Qwen3.8-Flash-Next-ISTA-DASLab-IQ3_S]
    #[arg(long, env = "STRAYDOG_ALIAS", value_name = "NAME")]
    pub alias: Option<String>,
    /// Temperature when a request gives none; 0 is greedy [default: the model card's, 1.0 thinking and 0.7 not]
    #[arg(long, env = "STRAYDOG_TEMP", value_name = "T")]
    pub temp: Option<f64>,
    /// Keep the K most likely tokens; 0 keeps all [default: the model card's, 20]
    #[arg(long, env = "STRAYDOG_TOP_K", value_name = "K")]
    pub top_k: Option<usize>,
    /// Keep the most likely tokens holding P of the probability [default: the model card's, 0.95 thinking and 0.8 not]
    #[arg(long, env = "STRAYDOG_TOP_P", value_name = "P")]
    pub top_p: Option<f64>,
    /// Keep tokens at least P times as likely as the most likely [default: the model card's, 0]
    #[arg(long, env = "STRAYDOG_MIN_P", value_name = "P")]
    pub min_p: Option<f64>,
    /// Lower the logit of every token the answer already holds by X [default: the model card's, 0 thinking and 1.5 not]
    #[arg(long, env = "STRAYDOG_PRESENCE_PENALTY", value_name = "X", allow_negative_numbers = true)]
    pub presence_penalty: Option<f64>,
    /// Sampling seed when a request gives none [default: a fresh one per request]
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
    /// Where Straydog keeps the shards' verity trees (built at the first start) and the experts' warm-up state [default: .straydog in the model directory]
    #[arg(long, env = "STRAYDOG_DATA_DIR", value_name = "DIR")]
    pub data_dir: Option<PathBuf>,
    /// Host memory for routed experts outside VRAM, in GiB, for a bundle read on demand [default: what Straydog's other host memory leaves of 85% of the host memory]
    #[arg(long, env = "STRAYDOG_EXPERT_RAM", value_name = "GIB")]
    pub expert_ram: Option<f64>,
    /// Windows: keep the state of the first request's system block (the system prompt with its tools), sealed for the user, in the data directory, and take the next start's first request from it [default: true; `prefix_cache = false` in the config file, or --no-prefix-cache]
    #[arg(skip)]
    pub prefix_cache: Option<bool>,
    /// Keep no state of the system block on the drive (see `prefix_cache`)
    #[arg(long = "no-prefix-cache")]
    #[serde(skip)]
    pub no_prefix_cache: bool,
    /// Windows: when the server stops on purpose (Ctrl+C, Windows ending the session, or a stop asked for with --stop-on-stdin), keep the state of the conversation the engine held last, sealed for the user, in the data directory, and take the next start's first request that goes on from it from there, instead of reading it all again [default: off; `keep_conversation = true` in the config file, or --keep-conversation]
    #[arg(long)]
    #[serde(default)]
    pub keep_conversation: bool,
    /// Stop on a line `stop` on standard input, as Ctrl+C stops (for a program that starts the server with no console to send it to; nothing else on standard input is read)
    #[arg(long)]
    #[serde(skip)]
    pub stop_on_stdin: bool,
}

/// The settings in effect, every derived value filled in.
#[derive(Debug, Clone)]
pub struct Settings {
    pub model: PathBuf,
    /// The pinned bundle in `model`: its model, provider and quantization.
    pub bundle: bundle::Id,
    /// The engram table named (its precision), which the server and its worker read ([`bundle::table_in`]).
    pub engram: Option<String>,
    /// Speed first: the router weight below which the CPU's routes are left out, which the worker reads (none: every route).
    pub expert_floor: Option<f32>,
    pub model_draft: Option<PathBuf>,
    /// Qwen3.8's projector, given or found: the worker checks it, and images are refused with the reason if it is not the pinned file.
    pub model_vision: Option<PathBuf>,
    pub draft_max: u8,
    pub ctx_size: u32,
    /// None: chosen by the worker from the GPUs' budgets (`--vram auto`).
    pub batch_size: Option<u16>,
    /// The batch was not given: `batch_size` is what the GPU budgets derive (shown, and sent to the worker as auto), which a placement may narrow;
    /// a batch given is kept as it is where it fits.
    pub batch_derived: bool,
    pub gpu: Vec<usize>,
    pub split_mode: SplitMode,
    pub vram: Vram,
    pub state_ram: f64,
    /// MiB; none: the model's own default.
    pub row_cache: Option<u32>,
    pub gemm: Gemm,
    pub gemm_min_batch: u16,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub alias: String,
    /// Unset sampling controls follow the model card for each request's mode.
    pub temp: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    /// Unset: a fresh seed per request.
    pub seed: Option<u64>,
    pub reasoning_effort: Effort,
    pub startup_timeout: f64,
    pub log_file: Option<PathBuf>,
    pub verbose: bool,
    /// The shards' verity trees and the warm-up state: `--data-dir`, or `.straydog` in the model directory.
    pub data_dir: PathBuf,
    /// Unset: planned from the host memory.
    pub expert_ram: Option<f64>,
    /// Windows: the first request's system block's state is kept sealed in the data directory and handed to the next start (`straydog::prefix`).
    pub prefix_cache: bool,
    /// Windows: the state of the conversation held last is kept sealed in the data directory when the server stops on purpose, and a request of the
    /// next start that goes on from it is taken from there (`straydog::conversation`). Off unless asked for.
    pub keep_conversation: bool,
    /// A line `stop` on standard input stops the server (`--stop-on-stdin`).
    pub stop_on_stdin: bool,
}

/// llama.cpp's multi-letter short options, as their long forms (clap's short options are one letter).
pub fn expand_short_options(args: impl IntoIterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    args.into_iter()
        .map(|arg| {
            let text = arg.to_string_lossy();
            for (short, long) in [("-md", "--model-draft"), ("-gpu", "--gpu"), ("-dev", "--gpu"), ("-sm", "--split-mode")] {
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

/// Which platform's conventions name the folders Straydog keeps its models and settings in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Unix,
    Windows,
}

impl Platform {
    pub const THIS: Platform = if cfg!(windows) { Platform::Windows } else { Platform::Unix };
}

/// The folders for what `straydog setup` keeps: models and the settings file. On Linux the XDG
/// ones (`$XDG_DATA_HOME` or `~/.local/share`, `$XDG_CONFIG_HOME` or `~/.config`; a relative
/// value is not a base directory and is ignored), on Windows `%LOCALAPPDATA%`; `variable` reads
/// the environment. None when the platform's variables name no home.
pub fn user_folders(platform: Platform, variable: &dyn Fn(&str) -> Option<std::ffi::OsString>) -> Option<(PathBuf, PathBuf)> {
    let absolute = |name: &str| variable(name).map(PathBuf::from).filter(|path| path.is_absolute());
    match platform {
        Platform::Unix => {
            let home = absolute("HOME");
            let base = |name: &str, under: &str| absolute(name).or_else(|| home.as_ref().map(|home| home.join(under)));
            Some((base("XDG_DATA_HOME", ".local/share")?.join("straydog"), base("XDG_CONFIG_HOME", ".config")?.join("straydog")))
        }
        Platform::Windows => {
            let local = absolute("LOCALAPPDATA")
                .or_else(|| absolute("USERPROFILE").map(|profile| profile.join("AppData").join("Local")))?;
            Some((local.join("Straydog"), local.join("Straydog")))
        }
    }
}

/// Where `straydog setup` puts the models unless told otherwise: `models` in the data folder.
pub fn default_models_dir() -> Option<PathBuf> {
    user_folders(Platform::THIS, &|name| std::env::var_os(name)).map(|(data, _)| data.join("models"))
}

/// The settings file `straydog setup` writes and a plain `straydog` reads.
pub fn default_config_path() -> Option<PathBuf> {
    user_folders(Platform::THIS, &|name| std::env::var_os(name)).map(|(_, config)| config.join("config.toml"))
}

/// The file a server started without `--config` reads: [`default_config_path`], when it exists
/// and no model is given (an option or `STRAYDOG_MODEL`). Whoever names the model starts the
/// server with everything else in mind; a file's quantization or drafter would not fit another model.
pub fn implicit_config(options: &Options, default: Option<PathBuf>) -> Option<PathBuf> {
    default.filter(|path| options.model.is_none() && path.is_file())
}

/// Merge the command line (with its variables) over the config file, when one is given.
pub fn load(matches: &ArgMatches, config: Option<&Path>) -> Result<(Options, Sources), String> {
    let given = Options::from_arg_matches(matches).map_err(|exc| exc.to_string())?;
    let path = config.map(Path::to_path_buf);
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
    let options = merge!(model, model_name, provider, quant, engram, expert_floor, model_draft, model_vision, draft_max, ctx_size, batch_size, gpu, split_mode, vram,
                         state_ram, row_cache, gemm, gemm_min_batch, host, port, api_key, api_key_file, alias, temp, top_k,
                         top_p, min_p, presence_penalty, seed, reasoning_effort, startup_timeout, log_file, verbose,
                         data_dir, expert_ram, prefix_cache, no_prefix_cache, keep_conversation, stop_on_stdin);
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

/// The pinned bundle these labels name in `model` (a label left out fitting any): the one they fit
/// alone, else the one of those whose shards are all in `model`.
pub fn find_bundle(model: &Path, model_name: Option<&str>, provider: Option<&str>, quant: Option<&str>) -> Result<bundle::Id, String> {
    let fitting = bundle::matching(model_name, provider, quant);
    if let ([id], Some(_)) = (&fitting[..], quant) {
        return Ok(*id);
    }
    if fitting.is_empty() {
        return match quant {
            Some(quant) => bundle::Id::find(model_name, provider, quant),
            None => Err(format!("no pinned bundle is {}", [model_name, provider].into_iter().flatten().collect::<Vec<_>>().join(" / "))),
        };
    }
    let here: Vec<bundle::Id> = fitting.iter().copied()
        .filter(|id| bundle::shards(*id).is_ok_and(|shards| shards.iter().all(|shard| model.join(&shard.name).is_file())))
        .collect();
    let named = |ids: &[bundle::Id]| ids.iter().map(bundle::Id::to_string).collect::<Vec<_>>().join(", ");
    match here[..] {
        [id] => Ok(id),
        [] => Err(format!("{} holds no pinned bundle{}; give --model-name, --provider and --quant",
                          model.display(), if quant.is_some() { format!(" of {}", named(&fitting)) } else { String::new() })),
        _ => Err(format!("{} holds more than one bundle ({}); give --model-name, --provider and --quant", model.display(), named(&here))),
    }
}

/// The directory and bundle of a pinned shard given as the model.
fn bundle_of_shard(shard: &Path) -> Result<(PathBuf, Option<bundle::Id>), String> {
    let name = shard.file_name().and_then(|name| name.to_str()).unwrap_or("");
    let id = bundle::bundles()
        .find(|id| bundle::shards(*id).is_ok_and(|shards| shards.iter().any(|pin| pin.name == name)))
        .ok_or_else(|| format!("{} is not a shard of a pinned model bundle", shard.display()))?;
    Ok((shard.parent().unwrap().to_path_buf(), Some(id)))
}

/// The directory of the pinned drafter given as a file.
fn drafter_directory(file: &Path) -> Result<PathBuf, String> {
    let pin = bundle::bundle(bundle::BUNDLES[0])?.drafter;
    if file.file_name().and_then(|name| name.to_str()) != Some(pin.name.as_str()) {
        return Err(format!("{} is not the pinned drafter ({})", file.display(), pin.name));
    }
    Ok(file.parent().unwrap().to_path_buf())
}

/// Fill every derived setting.
pub fn resolve(options: Options) -> Result<Settings, String> {
    let model = options.model.ok_or("no model: give -m DIR or set model in the config file")?;
    let model = model.canonicalize().map_err(|exc| format!("model {}: {exc}", model.display()))?;
    // A shard of a bundle, as llama.cpp takes a model, names its directory (and bundle).
    let (model, shard_bundle) = if model.is_file() { bundle_of_shard(&model)? } else { (model, None) };
    if !model.is_dir() {
        return Err(format!("model directory not found: {}", model.display()));
    }
    let (model_name, provider, quant) = (options.model_name.as_deref(), options.provider.as_deref(), options.quant.as_deref());
    let id = match shard_bundle {
        Some(id) if !bundle::matching(model_name, provider, quant).contains(&id) => {
            return Err(format!("the shard given is of {id}, not of the bundle the options name"));
        }
        Some(id) => id,
        None => find_bundle(&model, model_name, provider, quant)?,
    };
    // The shards' verity trees and the warm-up state: beside the model unless given elsewhere
    // (made there by the server, never by its worker). A bundle read on demand has no MTP drafter.
    let on_demand = id.is_on_demand();
    // (a table named must be one the bundle reads, there)
    let engram = match options.engram.as_deref().filter(|precision| !precision.is_empty()) {
        Some(precision) => match bundle::table_in_with(&model, id, Some(precision))? {
            Some((table, _)) => Some(table.precision),
            None => return Err(format!("the {id} bundle has no engram table to choose")),
        },
        None => None,
    };
    let expert_floor = match options.expert_floor {
        Some(floor) if !(0.0..1.0).contains(&floor) => return Err(format!("expert floor {floor}: a router weight from 0 to 1")),
        Some(floor) if floor > 0.0 => Some(floor),
        _ => None,
    };
    let data_dir = match options.data_dir {
        Some(dir) => dir.canonicalize().map_err(|exc| format!("data directory {}: {exc}", dir.display()))?,
        None => bundle::default_data_dir(&model),
    };
    if on_demand && options.model_draft.is_some() {
        return Err(format!("the {id} bundle has no MTP drafter"));
    }
    if options.expert_ram.is_some_and(|gib| !(0.0..=4096.0).contains(&gib)) {
        return Err("--expert-ram must be between 0 and 4096 GiB".into());
    }
    let model_draft = match options.model_draft {
        Some(path) => {
            let path = path.canonicalize().map_err(|exc| format!("drafter {}: {exc}", path.display()))?;
            Some(if path.is_file() { drafter_directory(&path)? } else { path })
        }
        None => None,
    };
    let draft_max = options.draft_max.unwrap_or(if model_draft.is_some() { 3 } else { 0 });
    if on_demand && options.model_vision.is_some() {
        return Err(format!("the {id} bundle keeps its own vision encoder beside its shards: it takes no --mmproj"));
    }
    let model_vision = match options.model_vision {
        Some(path) => Some(path.canonicalize().map_err(|exc| format!("vision encoder {}: {exc}", path.display()))?),
        None if on_demand => None,
        None => bundle::qwen_vision_in(&model),
    };
    let gpu = options.gpu.unwrap_or_else(|| vec![0]);
    let mut distinct = gpu.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if gpu.is_empty() || gpu.len() > 4 || distinct.len() != gpu.len() {
        return Err("GPU ordinals must be one to four distinct numbers".into());
    }
    let vram = options.vram.unwrap_or(Vram::Auto);
    if let Vram::Gib(gib) = &vram {
        if gib.is_empty() || gib.iter().any(|gib| !(0.01..=256.0).contains(gib)) {
            return Err("--vram must be auto or between 0.01 and 256 GiB a GPU".into());
        }
        if gib.len() > 1 && gib.len() != gpu.len() {
            return Err(format!("--vram gives {} budgets for {} GPUs: give one, or one per GPU", gib.len(), gpu.len()));
        }
    }
    let deepseek = on_demand && bundle::architecture_of(id)? == "deepseek41";
    if options.batch_size.is_some_and(|batch| batch > 4096) && !deepseek {
        return Err("--batch-size above 4096 is for DeepSeek-V4.1 alone".into());
    }
    // The model's own limit holds whether the value came from an option, a variable or the file; a
    // range on the option could not check the file's, nor know the model.
    let ctx_size = options.ctx_size.unwrap_or(262144);
    let ctx_limit = bundle::context_limit(id)?;
    if !(1..=ctx_limit).contains(&(ctx_size as usize)) {
        return Err(format!("--ctx-size must be between 1 and {ctx_limit} tokens for {}, not {ctx_size}", id.model));
    }
    // The bundles read on demand read each routed expert a pass needs once per pass: the longer
    // the pass, the fewer reads.
    // With the budgets left to the worker, it chooses the same way from those it finds.
    let total = vram.bytes(gpu.len()).map(|bytes| bytes.iter().sum::<u64>());
    let batch_derived = options.batch_size.is_none();
    let batch_size = options.batch_size.or(match total {
        // (DeepSeek-V4.1 chooses its own from what the GPUs' memory holds: up to 32768)
        _ if deepseek => None,
        _ if on_demand => Some(4096),
        // (Qwen3.8 on two R9700s: 4096 rows a batch read a 20K or 60K prompt 7-12% slower than 2048 on 2026-10-29; on 2026-11-12, served,
        // one run each, 4096 against 2048: 8K 1.95 -> 1.75 s, 32K 7.10 -> 6.81 s, 95K 22.83 -> 21.63 s, 242K 66.1 -> 62.6 s, decode the same)
        Some(total) => Some(straydog_engine::model::resident_batch(total) as u16),
        None => None,
    });
    // Enough for the context's state (K/V stays on the host when it exceeds a quarter of the
    // GPU budget), rounded up to half a GiB.
    let needed = (context_state_bytes(u64::from(ctx_size), draft_max > 0) as f64 / (1u64 << 30) as f64 * 2.0).ceil() / 2.0;
    let state_ram = options.state_ram.unwrap_or(needed.max(1.0));
    if !(0.01..=256.0).contains(&state_ram) {
        return Err("--state-ram must be between 0.01 and 256 GiB".into());
    }
    let row_cache = options.row_cache;
    if row_cache.is_some_and(|mib| mib > 65536) {
        return Err("--row-cache must be at most 65536 MiB".into());
    }
    if options.temp.is_some_and(|temp| !(0.0..=2.0).contains(&temp)) {
        return Err("--temp must be between 0 and 2".into());
    }
    if options.top_k.is_some_and(|top_k| top_k > 1_000_000) {
        return Err("--top-k must be at most 1000000".into());
    }
    if options.top_p.is_some_and(|top_p| !(top_p > 0.0 && top_p <= 1.0)) {
        return Err("--top-p must be in (0, 1]".into());
    }
    if options.min_p.is_some_and(|min_p| !(0.0..1.0).contains(&min_p)) {
        return Err("--min-p must be in [0, 1)".into());
    }
    if options.presence_penalty.is_some_and(|penalty| !(-2.0..=2.0).contains(&penalty)) {
        return Err("--presence-penalty must be between -2 and 2".into());
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
    let alias = options.alias.unwrap_or_else(|| format!("{}-{}-{}", id.model, id.provider, id.quant));
    Ok(Settings {
        model,
        bundle: id,
        engram,
        expert_floor,
        model_draft,
        model_vision,
        draft_max,
        ctx_size,
        batch_size,
        batch_derived,
        gpu,
        split_mode: options.split_mode.unwrap_or(SplitMode::Experts),
        vram,
        state_ram,
        row_cache,
        gemm: options.gemm.unwrap_or(Gemm::Fp16),
        gemm_min_batch: options.gemm_min_batch.unwrap_or(2),
        host: options.host.unwrap_or_else(|| "127.0.0.1".into()),
        port: options.port.unwrap_or(8080),
        api_key,
        api_key_file,
        alias,
        temp: options.temp,
        top_k: options.top_k,
        top_p: options.top_p,
        min_p: options.min_p,
        presence_penalty: options.presence_penalty,
        seed: options.seed,
        reasoning_effort: options.reasoning_effort.unwrap_or(Effort::High),
        startup_timeout,
        log_file: options.log_file,
        verbose: options.verbose,
        data_dir,
        expert_ram: options.expert_ram,
        prefix_cache: !options.no_prefix_cache && options.prefix_cache.unwrap_or(true),
        keep_conversation: options.keep_conversation,
        stop_on_stdin: options.stop_on_stdin,
    })
}

/// One line per setting: its value and where it came from (`derived` when filled in).
pub fn describe(settings: &Settings, sources: &Sources) -> Vec<String> {
    let path = |path: &Option<PathBuf>| path.as_ref().map(|path| path.display().to_string()).unwrap_or_default();
    fn card<T: ToString>(value: Option<T>) -> String {
        value.map_or_else(|| "(model card)".into(), |value| value.to_string())
    }
    let values: Vec<(&str, String)> = vec![
        ("model", settings.model.display().to_string()),
        ("model_name", settings.bundle.model.to_string()),
        ("provider", settings.bundle.provider.to_string()),
        ("quant", settings.bundle.quant.to_string()),
        ("engram", settings.engram.clone().unwrap_or_else(|| "(the bundle's own)".into())),
        ("expert_floor", settings.expert_floor.map_or_else(|| "0 (every route)".into(), |floor| floor.to_string())),
        ("model_draft", path(&settings.model_draft)),
        ("model_vision", path(&settings.model_vision)),
        ("draft_max", settings.draft_max.to_string()),
        ("ctx_size", settings.ctx_size.to_string()),
        ("batch_size", settings.batch_size.map_or_else(|| "(from the GPUs' budgets)".into(), |batch| batch.to_string())),
        ("gpu", settings.gpu.iter().map(usize::to_string).collect::<Vec<_>>().join(",")),
        ("split_mode", format!("{:?}", settings.split_mode).to_lowercase()),
        ("vram", settings.vram.to_string()),
        ("state_ram", settings.state_ram.to_string()),
        ("row_cache", settings.row_cache.map_or_else(|| "(the model's default)".into(), |mib| mib.to_string())),
        ("gemm", format!("{:?}", settings.gemm).to_lowercase()),
        ("gemm_min_batch", settings.gemm_min_batch.to_string()),
        ("host", settings.host.clone()),
        ("port", settings.port.to_string()),
        ("api_key", if settings.api_key.is_some() { "(set)".into() } else { String::new() }),
        ("api_key_file", path(&settings.api_key_file)),
        ("alias", settings.alias.clone()),
        ("temp", card(settings.temp)),
        ("top_k", card(settings.top_k)),
        ("top_p", card(settings.top_p)),
        ("min_p", card(settings.min_p)),
        ("presence_penalty", card(settings.presence_penalty)),
        ("seed", settings.seed.map_or_else(|| "(fresh per request)".into(), |seed| seed.to_string())),
        ("reasoning_effort", format!("{:?}", settings.reasoning_effort).to_lowercase()),
        ("startup_timeout", settings.startup_timeout.to_string()),
        ("log_file", path(&settings.log_file)),
        ("verbose", settings.verbose.to_string()),
        ("data_dir", settings.data_dir.display().to_string()),
        ("expert_ram", settings.expert_ram.map_or_else(|| "(planned)".into(), |gib| gib.to_string())),
        ("prefix_cache", settings.prefix_cache.to_string()),
        ("keep_conversation", settings.keep_conversation.to_string()),
        ("stop_on_stdin", settings.stop_on_stdin.to_string()),
    ];
    values
        .into_iter()
        .map(|(name, value)| {
            let derived = ["model_name", "provider", "quant", "draft_max", "batch_size", "state_ram", "alias", "api_key"].contains(&name);
            let named = |wanted: &str| sources.iter().find(|(key, _)| *key == wanted);
            // (`--no-prefix-cache` is the option that sets prefix_cache, and it wins over the file)
            let found = if name == "prefix_cache" { named("no_prefix_cache").or_else(|| named(name)) } else { named(name) };
            let source = found.map_or(if derived { "derived" } else { "default" }, |(_, source)| source.as_str());
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

    /// A pinned bundle the test names: its model's, the only bundle of that name.
    fn id(model: &str, quant: &str) -> bundle::Id {
        bundle::Id::find(Some(model), None, quant).unwrap()
    }

    fn qwen(quant: &str) -> bundle::Id {
        id(bundle::MODEL, quant)
    }

    /// A folder of the test's own in the temporary directory, removed when it goes out of scope, so also when an assert
    /// fails: a failed run once left a bundle-sized file there (47 GB), and the next ones found the disk full.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            // (one per call: the tests run in parallel, each removing its own)
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!("straydog-config-{name}-{}-{}", std::process::id(),
                                                         NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
            std::fs::create_dir_all(&path).unwrap();
            Scratch(path)
        }
    }

    impl std::ops::Deref for Scratch {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A model directory holding (empty) files named like one bundle's shards.
    fn bundle_directory(id: bundle::Id) -> Scratch {
        let directory = Scratch::new(id.quant);
        for shard in bundle::shards(id).unwrap() {
            std::fs::write(directory.join(shard.name), b"").unwrap();
        }
        directory
    }

    /// Makes the file at `path` `bytes` long without writing them. The file is sparse where the file system allows it:
    /// Linux's set_len leaves a hole, and on Windows the file is marked sparse first, or NTFS would allocate every byte.
    /// Where it cannot be sparse, the disk has to hold the bytes, and the reason comes back when it has not the room.
    fn sized(path: &Path, bytes: u64) -> Result<(), String> {
        let file = std::fs::File::options().write(true).open(path).unwrap();
        if !sparse(&file) {
            // (a gibibyte to spare: the disk is shared with the build and the other tests)
            let free = crate::setup::machine::free_space(path).unwrap_or(0);
            if free < bytes.saturating_add(1 << 30) {
                return Err(format!("{} would take {bytes} bytes of a disk with {free} free, and it cannot be sparse there", path.display()));
            }
        }
        file.set_len(bytes).unwrap();
        Ok(())
    }

    #[cfg(windows)]
    fn sparse(file: &std::fs::File) -> bool {
        use std::os::windows::io::AsRawHandle;
        let mut returned = 0u32;
        // SAFETY: a live handle; FSCTL_SET_SPARSE (0x000900C4) takes no buffers (without one it marks the whole file).
        unsafe {
            windows_sys::Win32::System::IO::DeviceIoControl(file.as_raw_handle() as _, 0x0009_00C4, std::ptr::null(), 0, std::ptr::null_mut(), 0,
                                                            &mut returned, std::ptr::null_mut()) != 0
        }
    }

    #[cfg(not(windows))]
    fn sparse(_: &std::fs::File) -> bool {
        true
    }

    #[test]
    fn a_bundle_read_on_demand_keeps_its_trees_beside_it_or_in_the_data_directory() {
        let model = bundle_directory(id(bundle::DEEPSEEK, "UD-IQ2_XXS"));
        let data = Scratch::new("data");
        let parse = |args: &[&str]| {
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let matches = command().try_get_matches_from(all).map_err(|exc| exc.to_string())?;
            resolve(Options::from_arg_matches(&matches).map_err(|exc| exc.to_string())?)
        };
        // By default beside the model (options alone: STRAYDOG_DATA_DIR may be set around the
        // tests); --data-dir moves the trees and the warm-up state.
        let beside = resolve(Options { model: Some(model.to_path_buf()), ..Options::default() }).unwrap();
        assert_eq!(beside.data_dir, model.canonicalize().unwrap().join(".straydog"));
        let settings = parse(&["--data-dir", data.to_str().unwrap(), "--expert-ram", "48"]).unwrap();
        assert_eq!((settings.bundle.quant, settings.alias.as_str()), ("UD-IQ2_XXS", "DeepSeek-V4.1-Flash-1017-Unsloth-UD-IQ2_XXS"));
        assert_eq!((settings.batch_size, settings.expert_ram, settings.data_dir.clone()), (None, Some(48.0), data.canonicalize().unwrap()));
        assert!(parse(&["--model-draft", "/"]).unwrap_err().contains("no MTP drafter"));
        assert_eq!(parse(&["--data-dir", data.to_str().unwrap(), "--batch-size", "8192"]).unwrap().batch_size, Some(8192));
        let worker = crate::runtime::worker_args(&settings);
        let text: Vec<String> = worker.iter().map(|arg| arg.to_string_lossy().into_owned()).collect();
        assert!(text.windows(2).any(|pair| pair[0] == "--host-expert-bytes" && pair[1] == (48u64 << 30).to_string()));
        assert!(text.windows(2).any(|pair| pair[0] == "--data-dir" && pair[1] == data.canonicalize().unwrap().display().to_string()));
        // The trees are what the worker reads beside the shards: they are built before it starts.
        assert!(crate::runtime::missing_trees(&settings).unwrap());
        assert!(crate::runtime::exposure(&settings).unwrap_err().contains("no verity tree"));
    }

    #[test]
    fn verity_trees_in_the_data_directory_are_built_before_the_worker_reads_them() {
        let model = bundle_directory(qwen("IQ3_XXS"));
        let data = Scratch::new("trees");
        let parse = |args: &[&str]| {
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            resolve(Options::from_arg_matches(&command().try_get_matches_from(all).unwrap()).unwrap()).unwrap()
        };
        let pins = bundle::verity_pins(qwen("IQ3_XXS")).unwrap();
        assert_eq!(pins.len(), 2, "both IQ3_XXS shards have pinned verity digests");
        // Beside the model (options alone: STRAYDOG_DATA_DIR may be set around the tests), or given.
        let beside = resolve(Options { model: Some(model.to_path_buf()), ..Options::default() }).unwrap();
        for settings in [beside, parse(&["--data-dir", data.to_str().unwrap()])] {
            assert!(crate::runtime::missing_trees(&settings).unwrap());
            assert!(crate::runtime::exposure(&settings).unwrap_err().contains("no verity tree"));
            for (asset, _) in &pins {
                let tree = bundle::tree_path(&settings.data_dir, asset);
                std::fs::create_dir_all(tree.parent().unwrap()).unwrap();
                std::fs::write(tree, b"").unwrap();
            }
            assert!(!crate::runtime::missing_trees(&settings).unwrap());
            let exposure = crate::runtime::exposure(&settings).unwrap();
            let verity = settings.data_dir.join("verity");
            assert!(exposure.directories.contains(&verity));
            assert!(pins.iter().all(|(asset, _)| exposure.files.contains(&verity.join(format!("{}.verity", asset.name)))));
        }
    }

    #[test]
    fn check_model_says_ready_only_with_every_pinned_size_and_tree_and_writes_nothing() {
        let id = qwen("IQ3_XXS");
        let model = bundle_directory(id);
        // (sparse files of the pinned sizes: nothing is read from them)
        for (asset, _) in bundle::verity_pins(id).unwrap() {
            if let Err(reason) = sized(&model.join(&asset.name), asset.bytes) {
                eprintln!("skipped: {reason}");
                return;
            }
        }
        let settings = resolve(Options { model: Some(model.to_path_buf()), ..Options::default() }).unwrap();
        let error = crate::runtime::check_model(&settings).unwrap_err();
        assert!(error.contains("no verity tree"), "{error}");
        assert!(!settings.data_dir.exists(), "the check builds no trees");
        // A tree that is not the pinned one's.
        for (asset, _) in bundle::verity_pins(id).unwrap() {
            let tree = bundle::tree_path(&settings.data_dir, &asset);
            std::fs::create_dir_all(tree.parent().unwrap()).unwrap();
            std::fs::write(tree, b"").unwrap();
        }
        let error = crate::runtime::check_model(&settings).unwrap_err();
        assert!(error.contains("not the pinned"), "{error}");
        // A file of another size.
        let first = &bundle::verity_pins(id).unwrap()[0].0;
        std::fs::File::options().write(true).open(model.join(&first.name)).unwrap().set_len(first.bytes - 1).unwrap();
        let error = crate::runtime::check_model(&settings).unwrap_err();
        assert!(error.contains("bytes where the pinned file has"), "{error}");
    }

    #[test]
    fn options_beat_variables_beat_the_file_and_derived_values_follow() {
        let model = bundle_directory(qwen("IQ3_S"));
        let file = model.join("config.toml");
        std::fs::write(&file, format!("model = {:?}\nvram = 20\nport = 9000\ndevice = [0, 1]\n", model.display())).unwrap();
        // SAFETY of the test: the only test that sets these variables.
        std::env::set_var("STRAYDOG_PORT", "9100");
        let matches = command().try_get_matches_from(["--vram", "30", "-c", "32768"]).unwrap();
        std::env::remove_var("STRAYDOG_PORT");
        let (options, sources) = load(&matches, Some(&file)).unwrap();
        let settings = resolve(options).unwrap();
        assert_eq!((settings.vram.clone(), settings.port, settings.gpu.clone()), (Vram::Gib(vec![30.0]), 9100, vec![0, 1]));
        assert_eq!(settings.bundle, qwen("IQ3_S"));
        assert_eq!(settings.batch_size, Some(4096));   // 2 x 30 GiB of GPU budget
        assert_eq!(settings.ctx_size, 32768);
        assert_eq!(settings.state_ram, 1.0);     // what 32768 tokens of state need (FP16 K/V), rounded up
        assert_eq!(settings.alias, "Qwen3.8-Flash-Next-ISTA-DASLab-IQ3_S");
        assert_eq!(settings.draft_max, 0);
        let lines = describe(&settings, &sources);
        let line = |name: &str| lines.iter().find(|line| line.starts_with(&format!("{name} "))).unwrap().clone();
        assert!(line("vram").ends_with("[option --vram]"));
        assert!(line("port").ends_with("[env STRAYDOG_PORT]"));
        assert!(line("gpu").ends_with(&format!("[config {}]", file.display())));
        assert!(line("batch_size").ends_with("[derived]"));
        assert!(line("host").ends_with("[default]"));
        std::fs::write(&file, "unknown = 1\n").unwrap();
        assert!(load(&command().try_get_matches_from(Vec::<&str>::new()).unwrap(), Some(&file)).is_err());
    }

    #[test]
    fn vram_is_auto_one_budget_or_one_per_gpu() {
        let model = bundle_directory(qwen("IQ3_S"));
        let resolved = |args: &[&str], file: Option<&str>| {
            let path = model.join("vram.toml");
            if let Some(text) = file {
                std::fs::write(&path, format!("model = {:?}\n{text}\n", model.display())).unwrap();
            }
            let mut argv = vec!["--model".to_string(), model.display().to_string()];
            argv.extend(args.iter().map(|arg| arg.to_string()));
            let matches = command().try_get_matches_from(&argv).unwrap();
            load(&matches, file.map(|_| path.as_path())).and_then(|(options, _)| resolve(options))
        };
        let settings = resolved(&["--gpu", "0,1"], None).unwrap();
        assert_eq!((settings.vram.clone(), settings.batch_size), (Vram::Auto, None));
        let settings = resolved(&["--gpu", "0,1", "--vram", "30,10"], None).unwrap();
        assert_eq!(settings.vram.bytes(2), Some(vec![30 << 30, 10 << 30]));
        assert_eq!(settings.batch_size, Some(1024));   // 40 GiB in all
        assert!(settings.batch_derived);
        // (the batch derived from the budgets and one given are told apart: the placement narrows only the derived, and the worker is told `auto` for it)
        let batch_arg = |settings: &Settings| {
            let args = crate::runtime::worker_args(settings);
            let at = args.iter().position(|arg| arg == "--prefill-batch-size").unwrap();
            args[at + 1].to_string_lossy().into_owned()
        };
        let derived = resolved(&["--gpu", "0,1", "--vram", "30"], None).unwrap();
        assert_eq!((derived.batch_size, derived.batch_derived, batch_arg(&derived).as_str()), (Some(4096), true, "auto"));
        let given = resolved(&["--gpu", "0,1", "--vram", "30", "--batch-size", "4096"], None).unwrap();
        assert_eq!((given.batch_size, given.batch_derived, batch_arg(&given).as_str()), (Some(4096), false, "4096"));
        let given = resolved(&["--gpu", "0,1", "--vram", "30", "--batch-size", "2048"], None).unwrap();
        assert_eq!((given.batch_size, given.batch_derived, batch_arg(&given).as_str()), (Some(2048), false, "2048"));
        let auto = resolved(&["--gpu", "0,1", "--vram", "auto"], None).unwrap();
        assert_eq!((auto.batch_size, auto.batch_derived, batch_arg(&auto).as_str()), (None, true, "auto"));
        assert_eq!(resolved(&["--gpu", "0,1", "--vram", "8"], None).unwrap().vram.bytes(2), Some(vec![8 << 30, 8 << 30]));
        assert_eq!(resolved(&["--gpu", "0,1", "--vram", "auto"], None).unwrap().vram, Vram::Auto);
        assert!(resolved(&["--gpu", "0,1", "--vram", "30,10,5"], None).unwrap_err().contains("3 budgets for 2 GPUs"));
        assert!(resolved(&["--vram", "0"], None).is_err());
        assert!(command().try_get_matches_from(["--vram", "lots"]).is_err());
        for (text, expected) in [("vram = 20", Vram::Gib(vec![20.0])), ("vram = [24, 11]", Vram::Gib(vec![24.0, 11.0])),
                                 ("vram = \"24,11\"", Vram::Gib(vec![24.0, 11.0])), ("vram = \"auto\"", Vram::Auto)] {
            assert_eq!(resolved(&["--gpu", "0,1"], Some(text)).unwrap().vram, expected, "{text}");
        }
    }

    /// The context is held to the model's own limit (DeepSeek-V4.1's 1048576, Qwen3.8's 262144), whether it came from an
    /// option or the settings file; the default is 262144 for both.
    #[test]
    fn the_context_is_held_to_the_models_own_limit() {
        let models = [(bundle_directory(id(bundle::DEEPSEEK, "UD-IQ2_XXS")), 1_048_576u32), (bundle_directory(qwen("IQ3_S")), 262_144)];
        for (model, limit) in &models {
            let resolved = |args: &[&str], file: Option<&str>| {
                let path = model.join("ctx.toml");
                if let Some(text) = file {
                    std::fs::write(&path, format!("model = {:?}\n{text}\n", model.display())).unwrap();
                }
                let mut argv = vec!["--model".to_string(), model.display().to_string()];
                argv.extend(args.iter().map(|arg| arg.to_string()));
                let matches = command().try_get_matches_from(&argv).unwrap();
                load(&matches, file.map(|_| path.as_path())).and_then(|(options, _)| resolve(options))
            };
            let (at, over) = (limit.to_string(), (limit + 1).to_string());
            assert_eq!(resolved(&[], None).unwrap().ctx_size, 262_144);
            assert_eq!(resolved(&["-c", &at], None).unwrap().ctx_size, *limit);
            assert_eq!(resolved(&[], Some(&format!("ctx_size = {limit}"))).unwrap().ctx_size, *limit);
            for error in [resolved(&["--ctx-size", &over], None), resolved(&[], Some(&format!("ctx_size = {over}")))] {
                let error = error.unwrap_err();
                assert!(error.contains(&format!("between 1 and {limit} tokens")) && error.contains(&format!("not {over}")), "{error}");
            }
            // (the other model's maximum is no limit of this one's)
            assert_eq!(resolved(&["-c", "1048576"], None).is_ok(), *limit == 1_048_576);
            assert!(resolved(&["-c", "0"], None).is_err() && resolved(&[], Some("ctx_size = 0")).is_err());
            // (the option replaces the file's value, which is then not checked)
            assert_eq!(resolved(&["-c", "32768"], Some(&format!("ctx_size = {over}"))).unwrap().ctx_size, 32768);
        }
    }

    #[test]
    fn bad_settings_are_refused() {
        let model = bundle_directory(qwen("Q2_0"));
        let parse = |args: &[&str]| {
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let matches = command().try_get_matches_from(all).map_err(|exc| exc.to_string())?;
            resolve(Options::from_arg_matches(&matches).map_err(|exc| exc.to_string())?)
        };
        assert_eq!(parse(&[]).unwrap().bundle, qwen("Q2_0"));
        assert!(parse(&["--quant", "Q9"]).is_err());
        assert!(parse(&["--device", "0,0"]).is_err());
        assert!(parse(&["--api-key", "k", "--api-key-file", "/etc/hostname"]).is_err());
        assert!(parse(&["--temp", "3"]).is_err());
        assert!(parse(&["-c", "300000"]).is_err());
        assert!(parse(&["--draft-max", "4"]).is_err());
    }

    /// The bundle served is the one the labels name, the directory's shards deciding between the bundles they
    /// leave open: UD-IQ1_M alone is the bundle of that name whose shards are there.
    #[test]
    fn the_labels_and_the_directory_name_the_bundle() {
        let unsloth = bundle_directory(qwen("UD-IQ1_M"));
        let parse = |model: &Path, args: &[&str]| {
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let matches = command().try_get_matches_from(all).map_err(|exc| exc.to_string())?;
            resolve(Options::from_arg_matches(&matches).map_err(|exc| exc.to_string()).unwrap())
        };
        let named = bundle::Id { model: bundle::MODEL, provider: bundle::UNSLOTH, quant: "UD-IQ1_M" };
        for args in [&[][..], &["--quant", "UD-IQ1_M"], &["--provider", "unsloth"], &["--model-name", "Qwen3.8-Flash-Next", "--provider", "Unsloth", "--quant", "UD-IQ1_M"]] {
            assert_eq!(parse(&unsloth, args).unwrap().bundle, named, "{args:?}");
        }
        assert_eq!(parse(&unsloth, &[]).unwrap().alias, "Qwen3.8-Flash-Next-Unsloth-UD-IQ1_M");
        // labels that leave more than one bundle open and none of them there, or fit none at all
        let error = parse(&unsloth, &["--provider", "ISTA-DASLab"]).unwrap_err();
        assert!(error.contains("holds no pinned bundle"), "{error}");
        assert!(parse(&unsloth, &["--provider", "ISTA-DASLab", "--quant", "UD-IQ1_M"]).unwrap_err().contains("unknown bundle UD-IQ1_M"));
        // an engram table by name: the bundle's own, or one there; none for a bundle without a table of its own
        assert_eq!(parse(&unsloth, &["--engram", "iq4_nl"]).unwrap().engram.as_deref(), Some("IQ4_NL"));
        assert!(parse(&unsloth, &["--engram", "Q8_0"]).unwrap_err().contains("engram Q8_0: no such table"));
        assert_eq!(parse(&unsloth, &[]).unwrap().engram, None);
        // speed first's floor: a router weight from 0 to 1; 0 or none is every route
        assert_eq!(parse(&unsloth, &["--expert-floor", "0.04"]).unwrap().expert_floor, Some(0.04));
        assert_eq!(parse(&unsloth, &["--expert-floor", "0"]).unwrap().expert_floor, None);
        assert_eq!(parse(&unsloth, &[]).unwrap().expert_floor, None);
        assert!(parse(&unsloth, &["--expert-floor", "1.5"]).unwrap_err().contains("expert floor 1.5"));
        let ista_qwen = bundle_directory(qwen("IQ2_XS"));
        assert!(parse(&ista_qwen, &["--engram", "Q8_0"]).unwrap_err().contains("has no engram table"));
        // 0.1.3's settings: DeepSeek-V4.1's ISTA-DASLab bundle by its GSQ-RCO name
        let ista = bundle_directory(id(bundle::DEEPSEEK, "IQ2_XXS"));
        assert_eq!(parse(&ista, &["--quant", "GSQ-RCO-IQ2_XXS"]).unwrap().bundle, id(bundle::DEEPSEEK, "IQ2_XXS"));
    }

    #[test]
    fn a_shard_names_its_bundle_directory_and_quantization() {
        let model = bundle_directory(qwen("IQ2_XS"));
        let shard = model.join(&bundle::bundle(qwen("IQ2_XS")).unwrap().shards[0].name);
        let matches = command().try_get_matches_from(["-m", shard.to_str().unwrap()]).unwrap();
        let settings = resolve(Options::from_arg_matches(&matches).unwrap()).unwrap();
        assert_eq!((settings.model, settings.bundle), (model.canonicalize().unwrap(), qwen("IQ2_XS")));
        let stray = model.join("other.gguf");
        std::fs::write(&stray, b"").unwrap();
        let matches = command().try_get_matches_from(["-m", stray.to_str().unwrap()]).unwrap();
        assert!(resolve(Options::from_arg_matches(&matches).unwrap()).is_err());
    }

    #[test]
    fn the_folders_for_models_and_settings_follow_the_platform() {
        let environment = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| {
            pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| std::ffi::OsString::from(value))
        };
        // (`/home/u` is not absolute on Windows, as a drive path is not elsewhere)
        if !cfg!(windows) {
            let (data, config) = user_folders(Platform::Unix, &environment(&[("HOME", "/home/u")])).unwrap();
            assert_eq!((data, config), (PathBuf::from("/home/u/.local/share/straydog"), PathBuf::from("/home/u/.config/straydog")));
            let moved = environment(&[("HOME", "/home/u"), ("XDG_DATA_HOME", "/data"), ("XDG_CONFIG_HOME", "/etc/u")]);
            assert_eq!(user_folders(Platform::Unix, &moved).unwrap(), (PathBuf::from("/data/straydog"), PathBuf::from("/etc/u/straydog")));
            // (a relative XDG value is not a base directory)
            let relative = environment(&[("HOME", "/home/u"), ("XDG_DATA_HOME", "data")]);
            assert_eq!(user_folders(Platform::Unix, &relative).unwrap().0, PathBuf::from("/home/u/.local/share/straydog"));
        }
        assert!(user_folders(Platform::Unix, &environment(&[])).is_none());
        let windows = environment(&[("LOCALAPPDATA", "C:\\Users\\u\\AppData\\Local"), ("USERPROFILE", "C:\\Users\\u")]);
        if cfg!(windows) {
            assert_eq!(user_folders(Platform::Windows, &windows).unwrap().0, PathBuf::from("C:\\Users\\u\\AppData\\Local").join("Straydog"));
        } else {
            // (a drive path is not absolute here; the profile's AppData is the fallback only where it is)
            assert!(user_folders(Platform::Windows, &windows).is_none());
            let posix = environment(&[("LOCALAPPDATA", "/c/Local")]);
            assert_eq!(user_folders(Platform::Windows, &posix).unwrap(), (PathBuf::from("/c/Local/Straydog"), PathBuf::from("/c/Local/Straydog")));
            let profile = environment(&[("USERPROFILE", "/c/Users/u")]);
            assert_eq!(user_folders(Platform::Windows, &profile).unwrap().0, PathBuf::from("/c/Users/u/AppData/Local/Straydog"));
        }
    }

    #[test]
    fn a_plain_server_reads_the_default_settings_file_but_a_named_model_does_not() {
        let model = bundle_directory(qwen("IQ3_S"));
        let file = model.join("default.toml");
        std::fs::write(&file, format!("model = {:?}\nport = 9000\n", model.display())).unwrap();
        let none = Options::default();
        let named = Options { model: Some(model.to_path_buf()), ..Options::default() };
        assert_eq!(implicit_config(&none, Some(file.clone())), Some(file.clone()));
        // a model by option or variable: the file's quantization and drafter would not fit another model
        assert_eq!(implicit_config(&named, Some(file.clone())), None);
        assert_eq!(implicit_config(&none, Some(model.join("missing.toml"))), None);
        assert_eq!(implicit_config(&none, None), None);
        // read as the config file, it names itself as the source of what it sets, and options still win
        let matches = command().try_get_matches_from(["--port", "9100"]).unwrap();
        let (options, sources) = load(&matches, implicit_config(&none, Some(file.clone())).as_deref()).unwrap();
        let settings = resolve(options).unwrap();
        assert_eq!((settings.port, settings.bundle), (9100, qwen("IQ3_S")));
        let lines = describe(&settings, &sources);
        assert!(lines.iter().any(|line| line.starts_with("model ") && line.ends_with(&format!("[config {}]", file.display()))));
        assert!(lines.iter().any(|line| line.starts_with("port ") && line.ends_with("[option --port]")));
    }

    /// The state of the first request's system block is kept by default; `prefix_cache = false` in the config file turns it off, and so does
    /// `--no-prefix-cache`, which is named as its source; the file's `true` does not undo the option.
    #[test]
    fn the_prefix_cache_is_on_unless_the_file_or_the_option_turns_it_off() {
        let model = bundle_directory(qwen("IQ3_S"));
        let setting = |file: Option<&str>, args: &[&str]| {
            let config = file.map(|text| {
                let path = model.join("prefix.toml");
                std::fs::write(&path, format!("model = {:?}
{text}
", model.display())).unwrap();
                path
            });
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let matches = command().try_get_matches_from(all).unwrap();
            let (options, sources) = load(&matches, config.as_deref()).unwrap();
            let settings = resolve(options).unwrap();
            let line = describe(&settings, &sources).into_iter().find(|line| line.starts_with("prefix_cache ")).unwrap();
            (settings.prefix_cache, line)
        };
        assert_eq!(setting(None, &[]), (true, "prefix_cache     true  [default]".to_string()));
        let (off, line) = setting(Some("prefix_cache = false"), &[]);
        assert!(!off && line.starts_with("prefix_cache     false  [config "), "{line}");
        assert!(setting(Some("prefix_cache = true"), &[]).0);
        let (off, line) = setting(None, &["--no-prefix-cache"]);
        assert!(!off && line.ends_with("[option --no-prefix-cache]"), "{line}");
        assert!(!setting(Some("prefix_cache = true"), &["--no-prefix-cache"]).0);
        // (it is not a key the file may spell the other way round)
        let path = model.join("wrong.toml");
        std::fs::write(&path, format!("model = {:?}
no_prefix_cache = true
", model.display())).unwrap();
        let matches = command().try_get_matches_from(["-m", model.to_str().unwrap()]).unwrap();
        assert!(load(&matches, Some(&path)).is_err());
    }

    /// The conversation's state is not kept unless it is asked for: off by default, on with `keep_conversation = true` in the file or with
    /// `--keep-conversation`, each named as its source; the stop on standard input is an option and no key of the file's.
    #[test]
    fn the_conversation_is_kept_only_when_asked_for() {
        let model = bundle_directory(qwen("IQ3_S"));
        let setting = |file: Option<&str>, args: &[&str]| {
            let config = file.map(|text| {
                let path = model.join("keep.toml");
                std::fs::write(&path, format!("model = {:?}
{text}
", model.display())).unwrap();
                path
            });
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let matches = command().try_get_matches_from(all).unwrap();
            let (options, sources) = load(&matches, config.as_deref()).unwrap();
            let settings = resolve(options).unwrap();
            let line = describe(&settings, &sources).into_iter().find(|line| line.starts_with("keep_conversation ")).unwrap();
            (settings.keep_conversation, line)
        };
        assert_eq!(setting(None, &[]), (false, "keep_conversation false  [default]".to_string()));
        let (on, line) = setting(Some("keep_conversation = true"), &[]);
        assert!(on && line.starts_with("keep_conversation true  [config "), "{line}");
        assert!(!setting(Some("keep_conversation = false"), &[]).0);
        let (on, line) = setting(None, &["--keep-conversation"]);
        assert!(on && line.ends_with("[option --keep-conversation]"), "{line}");
        // the stop on standard input: an option, off by default, and not a key of the file's
        let stops = |args: &[&str]| {
            let mut all = vec!["-m", model.to_str().unwrap()];
            all.extend_from_slice(args);
            let (options, _) = load(&command().try_get_matches_from(all).unwrap(), None).unwrap();
            resolve(options).unwrap().stop_on_stdin
        };
        assert!(!stops(&[]) && stops(&["--stop-on-stdin"]));
        let path = model.join("wrong.toml");
        std::fs::write(&path, format!("model = {:?}
stop_on_stdin = true
", model.display())).unwrap();
        assert!(load(&command().try_get_matches_from(["-m", model.to_str().unwrap()]).unwrap(), Some(&path)).is_err());
    }

    #[test]
    fn llama_short_options_expand() {
        let expanded = expand_short_options(["-md", "/d", "-gpu=0,1", "-dev", "1", "-sm", "layer", "-m", "x"].map(Into::into));
        assert_eq!(expanded, ["--model-draft", "/d", "--gpu=0,1", "--gpu", "1", "--split-mode", "layer", "-m", "x"]
                   .map(std::ffi::OsString::from));
    }
}
