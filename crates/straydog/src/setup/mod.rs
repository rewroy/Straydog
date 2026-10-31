//! `straydog setup`: choose a model, where it goes and which extras come with it; download and
//! authenticate it; write the settings file a plain `straydog` reads. Questions are asked on a
//! terminal; every answer is also an option, and `--yes` takes the recommended one for the rest.

pub mod catalog;
pub mod machine;
mod settings;

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use straydog_provision::Extras;

use crate::config;
use catalog::{Entry, Extra, Fit, Place, DEFAULT_CONTEXT, GIB};
use machine::Machine;

pub use settings::{write_settings, Settings, Written};

/// The options of `straydog setup`.
#[derive(Args, Debug, Clone, Default)]
pub struct SetupArgs {
    /// The model to set up (DeepSeek-V4.1-Flash or Qwen3.8-Flash-Next): its best bundle for this machine
    #[arg(long, value_name = "NAME")]
    pub model_name: Option<String>,
    /// The bundle to set up (as `fetch --quant`: GSQ-RCO-IQ2_XXS, GSQ-RCO-IQ2_XS, UD-IQ2_XXS, IQ3_S, IQ3_XXS, IQ2_XS or Q2_0)
    #[arg(long, value_name = "NAME")]
    pub quant: Option<String>,
    /// The folder for models: the bundle goes in DIR/<model>/<quant>, or is DIR when DIR holds it already [default: the platform's data folder]
    #[arg(long, value_name = "DIR")]
    pub dir: Option<PathBuf>,
    /// Where the shards' verity trees go [default: .straydog in the bundle directory]
    #[arg(long, env = "STRAYDOG_DATA_DIR", value_name = "DIR")]
    pub data_dir: Option<PathBuf>,
    /// GPU ordinals to use, comma separated [default: every GPU found, at most four]
    #[arg(long, value_name = "LIST", value_delimiter = ',')]
    pub gpu: Option<Vec<usize>>,
    /// Leave out the DSpark drafter (DeepSeek-V4.1)
    #[arg(long)]
    pub no_dspark: bool,
    /// Leave out the vision encoder (DeepSeek-V4.1)
    #[arg(long)]
    pub no_vision: bool,
    /// Leave out the MTP drafter (Qwen3.8)
    #[arg(long)]
    pub no_mtp: bool,
    /// The settings file to write [default: config.toml in the platform's config folder, which a plain `straydog` reads]
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Authenticate what is there; download nothing
    #[arg(long)]
    pub offline: bool,
    /// Ask nothing: take the recommended answer to whatever the options leave open
    #[arg(short = 'y', long)]
    pub yes: bool,
    /// Print this machine and every bundle, with how it would run here and which is recommended, as JSON, and exit
    #[arg(long)]
    pub list: bool,
}

/// What the person has on this machine that setup defaults from.
#[derive(Debug, Clone)]
pub struct Defaults {
    /// Where models go.
    pub models: Option<PathBuf>,
    /// The settings file a plain `straydog` reads.
    pub config: Option<PathBuf>,
    /// The home folder, for a `~` typed in a path.
    pub home: Option<PathBuf>,
    /// The free bytes of the disk a path is on.
    pub free_space: fn(&Path) -> Option<u64>,
}

impl Defaults {
    pub fn from_environment() -> Defaults {
        let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from);
        Defaults { models: config::default_models_dir(), config: config::default_config_path(), home, free_space: machine::free_space }
    }
}

/// Fetching a bundle and the extras chosen with it: the pinned downloads, or in a test a local server's.
pub trait Provision {
    fn fetch(&self, place: &Place, entry: &Entry, extras: &[Extra], offline: bool, progress: straydog_provision::Progress) -> Result<(), String>;
}

/// The pinned bundles from Hugging Face, authenticated.
pub struct Pinned;

impl Provision for Pinned {
    fn fetch(&self, place: &Place, entry: &Entry, extras: &[Extra], offline: bool, progress: straydog_provision::Progress) -> Result<(), String> {
        let wanted = Extras { dspark: extras.contains(&Extra::Dspark), vision: extras.contains(&Extra::Vision) };
        straydog_provision::fetch_extras(&place.bundle, &entry.quant, offline, Some(&place.data), wanted, progress)?;
        // (the drafter is built from tensors fetched for it: offline, only one already built is authenticated)
        if entry.sidecar(Extra::Mtp).is_some_and(|mtp| extras.contains(&Extra::Mtp) && (!offline || place.drafter.join(&mtp.file).is_file())) {
            let path = straydog_provision::fetch_drafter(&place.drafter, false, progress)?;
            println!("verified {}", path.display());
        }
        Ok(())
    }
}

/// Questions and answers on a terminal, or scripted.
pub struct Talk<'a> {
    input: &'a mut dyn BufRead,
    output: &'a mut dyn Write,
    /// Whether questions may be asked.
    pub interactive: bool,
}

impl<'a> Talk<'a> {
    pub fn new(input: &'a mut dyn BufRead, output: &'a mut dyn Write, interactive: bool) -> Talk<'a> {
        Talk { input, output, interactive }
    }

    pub fn say(&mut self, text: &str) {
        let _ = writeln!(self.output, "{text}");
    }

    fn read(&mut self, question: &str) -> Result<String, String> {
        write!(self.output, "{question} ").and_then(|()| self.output.flush()).map_err(|error| error.to_string())?;
        let mut line = String::new();
        match self.input.read_line(&mut line) {
            Ok(0) => Err("the input ended before setup was done: nothing was changed".to_string()),
            Ok(_) => Ok(line.trim().to_string()),
            Err(error) => Err(error.to_string()),
        }
    }

    /// A line, `default` when it is empty.
    fn ask(&mut self, question: &str, default: &str) -> Result<String, String> {
        let answer = self.read(&format!("{question} [{default}]:"))?;
        Ok(if answer.is_empty() { default.to_string() } else { answer })
    }

    fn confirm(&mut self, question: &str, default: bool) -> Result<bool, String> {
        loop {
            let answer = self.read(&format!("{question} [{}]:", if default { "Y/n" } else { "y/N" }))?;
            match answer.to_lowercase().as_str() {
                "" => return Ok(default),
                "y" | "yes" => return Ok(true),
                "n" | "no" => return Ok(false),
                _ => self.say("Please answer y or n."),
            }
        }
    }

    /// The number of one of `count` choices, `default` (0-based) when the line is empty.
    fn choose(&mut self, question: &str, count: usize, default: usize) -> Result<usize, String> {
        loop {
            let answer = self.ask(question, &(default + 1).to_string())?;
            match answer.parse::<usize>() {
                Ok(number) if (1..=count).contains(&number) => return Ok(number - 1),
                _ => self.say(&format!("Please give a number from 1 to {count}.")),
            }
        }
    }
}

/// GB (10^9) for what is downloaded, as the README gives sizes; GiB for memory, as Straydog reports it.
pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn gib(bytes: u64) -> String {
    format!("{:.0} GiB", bytes as f64 / GIB as f64)
}

/// The directory of `entry`'s bundle under the models folder `folder`: the folder itself when
/// it holds the bundle already (or the start of it), else `folder/<model>/<quant>`.
pub fn bundle_directory(folder: &Path, entry: &Entry) -> PathBuf {
    let first = &entry.files[0].name;
    if folder.join(first).exists() || folder.join(format!("{first}.part")).exists() {
        folder.to_path_buf()
    } else {
        folder.join(&entry.model).join(&entry.quant)
    }
}

/// What a finished setup did.
#[derive(Debug)]
pub struct Outcome {
    pub quant: String,
    pub place: Place,
    pub extras: Vec<Extra>,
    /// The context the settings give, when less than the default.
    pub context: Option<u64>,
    pub written: Written,
}

/// A bundle by the options' `--quant` or `--model-name`, if they name one: an error when they
/// name none that exists or runs here.
fn named(args: &SetupArgs, entries: &[Entry], fits: &[(u64, Fit)]) -> Result<Option<usize>, String> {
    if let Some(quant) = &args.quant {
        let index = entries.iter().position(|entry| entry.quant == *quant).ok_or_else(|| {
            format!("unknown bundle {quant}: {}", entries.iter().map(|entry| entry.quant.as_str()).collect::<Vec<_>>().join(", "))
        })?;
        if let Some(model) = &args.model_name {
            if !entries[index].model.eq_ignore_ascii_case(model) {
                return Err(format!("{quant} is a bundle of {}, not {model}", entries[index].model));
            }
        }
        return match &fits[index].1 {
            fit if fit.runs() => Ok(Some(index)),
            fit => Err(format!("{} cannot run on this machine: {}", entries[index].name(), fit.note(&entries[index]))),
        };
    }
    let Some(model) = &args.model_name else { return Ok(None) };
    let of_model: Vec<usize> = (0..entries.len()).filter(|&index| entries[index].model.eq_ignore_ascii_case(model)).collect();
    if of_model.is_empty() {
        return Err(format!("unknown model {model}: {}", catalog::models(entries).join(", ")));
    }
    let runs: Vec<usize> = of_model.iter().copied().filter(|&index| fits[index].1.runs()).collect();
    if runs.is_empty() {
        return Err(format!("{model} cannot run on this machine: {}", fits[of_model[0]].1.note(&entries[of_model[0]])));
    }
    let subset: Vec<Entry> = runs.iter().map(|&index| entries[index].clone()).collect();
    let subset_fits: Vec<Fit> = runs.iter().map(|&index| fits[index].1.clone()).collect();
    Ok(catalog::recommend(&subset, &subset_fits).map(|index| runs[index]))
}

/// Show the models that run here and let the person choose one.
fn menu(talk: &mut Talk, entries: &[Entry], fits: &[(u64, Fit)], gpus: usize, recommended: usize) -> Result<usize, String> {
    let offered: Vec<usize> = (0..entries.len()).filter(|&index| fits[index].1.runs()).collect();
    talk.say("Models that can run here (download size; what each needs):");
    for (number, &index) in offered.iter().enumerate() {
        let entry = &entries[index];
        talk.say(&format!("  {}) {:<22} {:<16} {:>9}{}", number + 1, entry.model, entry.quant, gb(entry.download_bytes(&[])),
                          if index == recommended { "  recommended" } else { "" }));
        let (context, fit) = &fits[index];
        let limited = if *context < DEFAULT_CONTEXT { format!(" with a context of {context} tokens (the most that starts)") } else { String::new() };
        talk.say(&format!("       needs {} of VRAM at the least{limited}; {}", gib(catalog::min_vram(entry, gpus, *context)), fit.note(entry)));
    }
    let left: Vec<String> = (0..entries.len())
        .filter(|&index| !fits[index].1.runs())
        .map(|index| format!("{} {} ({})", entries[index].model, entries[index].quant, fits[index].1.note(&entries[index])))
        .collect();
    if !left.is_empty() {
        talk.say(&format!("Not offered: {}.", left.join("; ")));
    }
    let default = offered.iter().position(|&index| index == recommended).unwrap_or(0);
    Ok(offered[talk.choose("Which one?", offered.len(), default)?])
}

/// Run setup. Questions go to `talk` for what the options leave open (`--yes`, or no terminal:
/// the recommended answer); `provision` fetches; `progress` shows long steps.
pub fn run(args: &SetupArgs, machine: &Machine, entries: &[Entry], defaults: &Defaults, provision: &dyn Provision, talk: &mut Talk,
           progress: straydog_provision::Progress) -> Result<Outcome, String> {
    let ask = talk.interactive && !args.yes;
    if !talk.interactive && !args.yes {
        return Err("setup asks questions: run it in a terminal, or give --yes (with --quant and --dir to choose the model and where it goes)".into());
    }
    talk.say("Straydog setup");
    for line in machine.describe() {
        talk.say(&line);
    }
    let fits: Vec<(u64, Fit)> = entries.iter().map(|entry| catalog::fit_at_best_context(entry, machine)).collect();
    let verdicts: Vec<Fit> = fits.iter().map(|(_, fit)| fit.clone()).collect();
    let index = match named(args, entries, &fits)? {
        Some(index) => index,
        None => {
            let recommended = catalog::recommend(entries, &verdicts).filter(|&index| fits[index].1.runs()).ok_or_else(|| {
                // (the reason of the model that comes closest: the least VRAM short)
                let reason = entries.iter().zip(&fits).min_by_key(|(_, (_, fit))| match fit {
                    Fit::TooLittleVram { needs, has } => needs - has,
                    _ => 0,
                }).map_or_else(String::new, |(entry, (_, fit))| format!(": {} {}", entry.name(), fit.note(entry)));
                format!("none of the models can run here{reason}")
            })?;
            if ask { menu(talk, entries, &fits, machine.gpus.len(), recommended)? } else { recommended }
        }
    };
    let entry = &entries[index];
    let (context, fit) = &fits[index];
    if matches!(fit, Fit::Drive(_)) {
        talk.say(&format!("Note: {}: {}.", entry.name(), fit.note(entry)));
    }
    if *context < DEFAULT_CONTEXT {
        talk.say(&format!("Note: its context is set to {context} tokens, the most that starts here."));
    }

    // the extras: all of them unless the options or the answers leave one out
    let mut extras = Vec::new();
    for sidecar in &entry.sidecars {
        let left_out = match sidecar.kind {
            Extra::Dspark => args.no_dspark,
            Extra::Vision => args.no_vision,
            Extra::Mtp => args.no_mtp,
        };
        let wanted = if left_out {
            false
        } else if ask && !args.offline {
            talk.confirm(&format!("Also get the {} ({}, {})?", sidecar.label, sidecar.purpose, gb(sidecar.download)), true)?
        } else {
            true
        };
        if wanted {
            extras.push(sidecar.kind);
        }
    }

    // where it goes, and whether it fits there
    let mut folder = match (&args.dir, ask) {
        (Some(dir), _) => dir.clone(),
        (None, false) => defaults.models.clone().ok_or("no default folder for models on this machine: give --dir")?,
        (None, true) => {
            let default = defaults.models.as_ref().map_or_else(String::new, |dir| dir.display().to_string());
            machine::expand_home(&talk.ask("Where should the models go?", &default)?, defaults.home.clone())
        }
    };
    let place = loop {
        let bundle = bundle_directory(&folder, entry);
        let data = args.data_dir.clone().unwrap_or_else(|| straydog_engine::bundle::default_data_dir(&bundle));
        let drafter = bundle.parent().map_or_else(|| bundle.join("mtp"), |parent| parent.join("mtp"));
        let place = Place { bundle, data, drafter };
        let missing = if args.offline { 0 } else { entry.disk_bytes(&extras).saturating_sub(entry.present_bytes(&place, &extras)) };
        match (defaults.free_space)(&place.bundle) {
            Some(free) if free < missing => {
                let message = format!("{} needs {} more on the disk of {}, which has {} free.", entry.name(), gb(missing), place.bundle.display(), gb(free));
                if !ask {
                    return Err(message);
                }
                talk.say(&message);
                let default = defaults.models.as_ref().map_or_else(String::new, |dir| dir.display().to_string());
                folder = machine::expand_home(&talk.ask("Another folder?", &default)?, defaults.home.clone());
            }
            _ => break place,
        }
    };

    let download = entry.download_bytes(&extras);
    let present = entry.present_bytes(&place, &extras);
    talk.say(&format!("Setting up {} in {}", entry.name(), place.bundle.display()));
    talk.say(&format!("  with {}", if extras.is_empty() { "no extras".to_string() } else {
        entry.sidecars.iter().filter(|sidecar| extras.contains(&sidecar.kind)).map(|sidecar| sidecar.label).collect::<Vec<_>>().join(", ")
    }));
    if args.offline {
        talk.say("  authenticating what is there; nothing is downloaded");
    } else if present > 0 {
        talk.say(&format!("  {} of {} download, {} already here (resumed or verified)", gb(download.saturating_sub(present)), gb(download), gb(present)));
    } else {
        talk.say(&format!("  {} to download, {} on the disk once done", gb(download), gb(entry.disk_bytes(&extras))));
    }
    if ask && !talk.confirm("Go ahead?", true)? {
        return Err("stopped: nothing was changed".into());
    }

    std::fs::create_dir_all(&place.bundle).map_err(|error| format!("{}: {error}", place.bundle.display()))?;
    provision.fetch(&place, entry, &extras, args.offline, progress)?;

    let model_draft = extras.contains(&Extra::Mtp).then(|| place.drafter.clone()).filter(|directory| entry.sidecar(Extra::Mtp).is_some_and(|mtp| directory.join(&mtp.file).is_file()));
    let gpus: Vec<usize> = machine.gpus.iter().map(|gpu| gpu.ordinal).collect();
    let target = args.config.clone().or_else(|| defaults.config.clone()).ok_or("no default folder for settings on this machine: give --config")?;
    let written = write_settings(&target, &Settings {
        model: place.bundle.clone(),
        quant: entry.quant.clone(),
        model_draft,
        gpu: (gpus != [0]).then_some(gpus),
        ctx_size: (*context < DEFAULT_CONTEXT).then_some(*context as u32),
        data_dir: args.data_dir.clone(),
    })?;
    talk.say(&format!("Settings {} {}", if written.changed { "written to" } else { "already in" }, written.path.display()));
    if let Some(old) = &written.kept {
        talk.say(&format!("  (the earlier file is kept as {})", old.display()));
    }
    let plain = defaults.config.as_ref() == Some(&written.path);
    talk.say(&format!("Start the server with: {}", if plain { "straydog".to_string() } else { format!("straydog --config {}", quote(&written.path)) }));
    talk.say(&format!("It listens on {} (the dashboard is at {}/dashboard).", written.url, written.url));
    Ok(Outcome { quant: entry.quant.clone(), place, extras, context: (*context < DEFAULT_CONTEXT).then_some(*context), written })
}

/// A path as a shell takes it.
fn quote(path: &Path) -> String {
    let text = path.display().to_string();
    if text.chars().all(|c| c.is_ascii_alphanumeric() || "/._-:\\".contains(c)) { text } else { format!("\"{text}\"") }
}

/// The command: look at the machine, then `run` on the terminal.
/// What `--list` prints, for a program that offers the models: the sizes are what setup fetches and
/// keeps with every extra, as it does unless told to leave one out.
fn listing(entries: &[Entry], machine: &Machine) -> serde_json::Value {
    let fits: Vec<(u64, Fit)> = entries.iter().map(|entry| catalog::fit_at_best_context(entry, machine)).collect();
    let verdicts: Vec<Fit> = fits.iter().map(|(_, fit)| fit.clone()).collect();
    let recommended = catalog::recommend(entries, &verdicts);
    let bundles: Vec<serde_json::Value> = entries.iter().zip(&fits).enumerate().map(|(index, (entry, (context, fit)))| {
        let extras = entry.default_extras();
        serde_json::json!({
            "model": entry.model,
            "quant": entry.quant,
            "download_bytes": entry.download_bytes(&extras),
            "disk_bytes": entry.disk_bytes(&extras),
            "extras": entry.sidecars.iter().map(|sidecar| serde_json::json!({
                "name": sidecar.kind.name(),
                "purpose": sidecar.purpose,
                "download_bytes": sidecar.download,
            })).collect::<Vec<_>>(),
            "context": context,
            "fit": fit.class(),
            "runs": fit.runs(),
            "note": fit.note(entry),
            "recommended": recommended == Some(index),
        })
    }).collect();
    serde_json::json!({
        "machine": {
            "backend": machine.backend,
            "gpus": machine.gpus.iter().map(|gpu| serde_json::json!({
                "ordinal": gpu.ordinal, "name": gpu.name, "total_bytes": gpu.total, "free_bytes": gpu.free,
            })).collect::<Vec<_>>(),
            "gpu_error": machine.gpu_error,
            "ram_bytes": machine.ram,
        },
        "bundles": bundles,
    })
}

pub fn main(args: &SetupArgs) -> Result<(), String> {
    let entries = catalog::pinned()?;
    let machine = Machine::probe(args.gpu.as_deref());
    if args.list {
        println!("{}", listing(&entries, &machine));
        return Ok(());
    }
    let stdin = std::io::stdin();
    let interactive = stdin.is_terminal() && std::io::stdout().is_terminal();
    let (mut input, mut output) = (stdin.lock(), std::io::stdout());
    let mut talk = Talk::new(&mut input, &mut output, interactive);
    run(args, &machine, &entries, &Defaults::from_environment(), &Pinned, &mut talk, &crate::progress::report).map(|_| ())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use sha2::{Digest, Sha256};
    use straydog_engine::bundle::{Asset, Footprint};
    use straydog_provision::mock::{self, Scratch, Server};

    use super::catalog::tests::{doggybox, machine as machine_of};
    use super::catalog::File;
    use super::*;
    use crate::config::Options;

    fn quiet(_stage: &str, _done: u64, _total: u64) {}

    fn unlimited(_: &Path) -> Option<u64> {
        Some(u64::MAX)
    }

    fn defaults(root: &Path) -> Defaults {
        Defaults { models: Some(root.join("models")), config: Some(root.join("config").join("config.toml")), home: Some(root.to_path_buf()), free_space: unlimited }
    }

    /// Fetches nothing: records what it was asked.
    #[derive(Default)]
    struct Recorder(Mutex<Vec<(PathBuf, Vec<Extra>, bool)>>);

    impl Provision for Recorder {
        fn fetch(&self, place: &Place, _: &Entry, extras: &[Extra], offline: bool, _: straydog_provision::Progress) -> Result<(), String> {
            self.0.lock().unwrap().push((place.bundle.clone(), extras.to_vec(), offline));
            Ok(())
        }
    }

    /// Fetches the files of a bundle that is not a pinned one from a local server.
    struct Local(Vec<(Asset, String)>);

    impl Provision for Local {
        fn fetch(&self, place: &Place, _: &Entry, _: &[Extra], _: bool, progress: straydog_provision::Progress) -> Result<(), String> {
            straydog_provision::fetch_files_plain(&place.bundle, &self.0, progress)
        }
    }

    const FILES: [(&str, &[u8]); 3] = [("test-00001-of-00002.gguf", b"the first shard's bytes"), ("test-00002-of-00002.gguf", b"the second shard"),
                                       ("tokenizer.json", b"{}")];

    /// A bundle of three small files that no pin knows, as an entry, with the pins and URLs to fetch it from `server`.
    fn small_bundle(server: &Server) -> (Entry, Vec<(Asset, String)>) {
        let assets: Vec<Asset> = FILES.iter()
            .map(|(name, data)| Asset { name: name.to_string(), bytes: data.len() as u64, sha256: format!("{:x}", Sha256::digest(data)) })
            .collect();
        let entry = Entry {
            model: "Test-Model".into(), quant: "T-Q1".into(), architecture: "qwen4exp".into(),
            files: assets.iter().map(|asset| File { name: asset.name.clone(), bytes: asset.bytes, tree: 0 }).collect(),
            footprint: Footprint { dense: 1 << 20, experts: 1 << 20, tables: 0, state_per_token: 1024 }, sidecars: Vec::new(),
        };
        let urls = assets.into_iter().map(|asset| { let url = format!("{}/{}", server.url, asset.name); (asset, url) }).collect();
        (entry, urls)
    }

    fn serve() -> Server {
        Server::start(|request| {
            let data = FILES.iter().find(|(name, _)| request.path.ends_with(name)).map(|(_, data)| *data).unwrap_or_default();
            mock::ranged(data, request.range.as_deref())
        })
    }

    fn run_with(args: &SetupArgs, machine: &Machine, entries: &[Entry], defaults: &Defaults, provision: &dyn Provision, script: &str, interactive: bool)
                -> (Result<Outcome, String>, String) {
        let (mut input, mut output) = (script.as_bytes(), Vec::new());
        let result = run(args, machine, entries, defaults, provision, &mut Talk::new(&mut input, &mut output, interactive), &quiet);
        (result, String::from_utf8(output).unwrap())
    }

    /// `--yes` with the bundle and the folder given asks nothing: the partial download is resumed,
    /// the rest fetched and authenticated, the settings written for a plain `straydog` to read, and
    /// the same run again fetches nothing and leaves the settings alone.
    #[test]
    fn a_setup_with_options_resumes_a_download_authenticates_it_and_writes_the_settings() {
        let scratch = Scratch::new("setup-yes");
        let root = scratch.path().to_path_buf();
        let server = serve();
        let (entry, urls) = small_bundle(&server);
        let bundle = root.join("models").join("Test-Model").join("T-Q1");
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(bundle.join("test-00001-of-00002.gguf.part"), b"the f").unwrap();
        let args = SetupArgs { quant: Some("T-Q1".into()), dir: Some(root.join("models")), yes: true, ..SetupArgs::default() };
        let (result, said) = run_with(&args, &doggybox(), std::slice::from_ref(&entry), &defaults(&root), &Local(urls.clone()), "", false);
        let outcome = result.unwrap();
        for (name, data) in FILES {
            assert_eq!(std::fs::read(bundle.join(name)).unwrap(), data);
        }
        // the first shard from the byte after the five it had; the others whole
        let ranges = server.requests().into_iter().map(|request| (request.path, request.range)).collect::<Vec<_>>();
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges.iter().find(|(path, _)| path.ends_with("00001-of-00002.gguf")).unwrap().1.as_deref(), Some("bytes=5-"));
        assert!(ranges.iter().filter(|(path, _)| !path.ends_with("00001-of-00002.gguf")).all(|(_, range)| range.is_none()));
        // what the server then reads
        let written = std::fs::read_to_string(root.join("config").join("config.toml")).unwrap();
        let read: Options = toml::from_str(&written).unwrap();
        assert_eq!((read.model.as_deref(), read.quant.as_deref(), read.gpu), (Some(bundle.as_path()), Some("T-Q1"), Some(vec![0, 1])));
        assert_eq!(outcome.place.bundle, bundle);
        assert!(said.contains("already here"), "{said}");
        assert!(said.contains("Start the server with: straydog\n") && said.contains("http://127.0.0.1:8080"), "{said}");
        // again: nothing is fetched (what is there is authenticated), nothing changes
        let (again, said) = run_with(&args, &doggybox(), std::slice::from_ref(&entry), &defaults(&root), &Local(urls), "", false);
        assert!(!again.unwrap().written.changed && said.contains("already in"), "{said}");
        assert_eq!(server.requests().len(), 3);
        assert_eq!(std::fs::read_to_string(root.join("config").join("config.toml")).unwrap(), written);
    }

    /// The disk is checked against the bundle less what is there: one byte short fails, before any request.
    #[test]
    fn a_setup_with_too_little_room_stops_before_it_downloads() {
        let scratch = Scratch::new("setup-room");
        let root = scratch.path().to_path_buf();
        let server = serve();
        let (entry, urls) = small_bundle(&server);
        let need: u64 = entry.files.iter().map(|file| file.bytes).sum();
        let args = SetupArgs { quant: Some("T-Q1".into()), dir: Some(root.join("models")), yes: true, ..SetupArgs::default() };
        let mut small = defaults(&root);
        small.free_space = |_| Some(30);
        assert_eq!(need, 41);
        let (result, said) = run_with(&args, &doggybox(), std::slice::from_ref(&entry), &small, &Local(urls.clone()), "", false);
        let error = result.unwrap_err();
        assert!(error.contains(&format!("needs {}", gb(need))) && error.contains("free"), "{error} / {said}");
        assert!(server.requests().is_empty() && !root.join("config").exists() && !root.join("models").exists());
        // some of it there already is not asked for again
        let bundle = root.join("models").join("Test-Model").join("T-Q1");
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(bundle.join(FILES[0].0), FILES[0].1).unwrap();
        let present = FILES[0].1.len() as u64;
        assert_eq!(need - present, 18);
        small.free_space = |_| Some(17);
        assert!(run_with(&args, &doggybox(), std::slice::from_ref(&entry), &small, &Local(urls.clone()), "", false).0.is_err());
        small.free_space = |_| Some(18);
        assert!(run_with(&args, &doggybox(), std::slice::from_ref(&entry), &small, &Local(urls), "", false).0.is_ok());
    }

    /// A transcript of the questions on a terminal, answered mostly by Enter.
    #[test]
    fn the_questions_are_asked_in_order_and_each_answer_is_used() {
        let scratch = Scratch::new("setup-ask");
        let root = scratch.path().to_path_buf();
        let entries = catalog::pinned().unwrap();
        let recorder = Recorder::default();
        // the model (Enter: the recommended), DSpark (n), vision (Enter), the folder (typed: ~ is the home), Go ahead (Enter);
        // a wrong answer is asked again
        let script = "x\n\nn\n\n~/models\n\n";
        let (result, said) = run_with(&SetupArgs::default(), &doggybox(), &entries, &defaults(&root), &recorder, script, true);
        let outcome = result.unwrap();
        println!("{said}");
        assert!(said.contains("2 x GPU (31.9 GiB each)") || said.contains("GPUs (ROCm/HIP)"), "{said}");
        assert!(said.contains("1) DeepSeek-V4.1-Flash") && said.contains("GSQ-RCO-IQ2_XXS") && said.contains("recommended"), "{said}");
        assert!(said.contains("Please give a number from 1 to 7."), "{said}");
        assert_eq!(outcome.quant, "GSQ-RCO-IQ2_XXS");
        assert_eq!(outcome.extras, [Extra::Vision]);
        assert_eq!(outcome.place.bundle, root.join("models").join("DeepSeek-V4.1-Flash").join("GSQ-RCO-IQ2_XXS"));
        let calls = recorder.0.lock().unwrap();
        assert_eq!(*calls, [(outcome.place.bundle.clone(), vec![Extra::Vision], false)]);
        let read: Options = toml::from_str(&std::fs::read_to_string(&outcome.written.path).unwrap()).unwrap();
        assert_eq!((read.quant.as_deref(), read.model_draft, read.gpu), (Some("GSQ-RCO-IQ2_XXS"), None, Some(vec![0, 1])));
    }

    /// The input ending (Ctrl-D, a closed pipe), a "no" at the last question and no terminal without --yes all
    /// stop before anything is made: no folder, no settings.
    #[test]
    fn a_setup_that_stops_changes_nothing() {
        let scratch = Scratch::new("setup-stop");
        let root = scratch.path().to_path_buf();
        let entries = catalog::pinned().unwrap();
        let recorder = Recorder::default();
        for (script, interactive, expected) in [("\n\nn\n\n", true, "input ended"), ("\n\nn\n\nn\n", true, "stopped"), ("", false, "give --yes")] {
            let (result, _) = run_with(&SetupArgs::default(), &doggybox(), &entries, &defaults(&root), &recorder, script, interactive);
            let error = result.unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
        assert!(recorder.0.lock().unwrap().is_empty());
        assert!(std::fs::read_dir(&root).unwrap().next().is_none(), "setup made something");
    }

    #[test]
    fn a_bundle_or_model_named_by_option_is_used_if_it_runs_here() {
        let scratch = Scratch::new("setup-named");
        let root = scratch.path().to_path_buf();
        let entries = catalog::pinned().unwrap();
        let small = machine_of(1, 12 << 30, 32 << 30);
        let recorder = Recorder::default();
        let ask = |args: SetupArgs, machine: &Machine| run_with(&SetupArgs { yes: true, dir: Some(root.join("models")), ..args }, machine, &entries, &defaults(&root), &recorder, "", false).0;
        // by model name, in any case: its best bundle for the machine
        let outcome = ask(SetupArgs { model_name: Some("qwen3.8-flash-next".into()), ..SetupArgs::default() }, &small).unwrap();
        assert_eq!(outcome.quant, "Q2_0");
        let outcome = ask(SetupArgs { model_name: Some("DeepSeek-V4.1-Flash".into()), ..SetupArgs::default() }, &doggybox()).unwrap();
        assert_eq!(outcome.quant, "GSQ-RCO-IQ2_XXS");
        // the extras left out by option
        let outcome = ask(SetupArgs { quant: Some("GSQ-RCO-IQ2_XS".into()), no_dspark: true, no_vision: true, ..SetupArgs::default() }, &doggybox()).unwrap();
        assert!(outcome.extras.is_empty());
        // a bundle this machine cannot run, a bundle nobody pinned, a model and bundle that disagree, no machine at all
        let error = ask(SetupArgs { quant: Some("GSQ-RCO-IQ2_XXS".into()), ..SetupArgs::default() }, &small).unwrap_err();
        assert!(error.contains("cannot run") && error.contains("12 GiB"), "{error}");
        assert!(ask(SetupArgs { quant: Some("Q8_0".into()), ..SetupArgs::default() }, &small).unwrap_err().contains("unknown bundle Q8_0"));
        assert!(ask(SetupArgs { quant: Some("Q2_0".into()), model_name: Some("DeepSeek-V4.1-Flash".into()), ..SetupArgs::default() }, &small).is_err());
        // a model no longer supported
        assert!(ask(SetupArgs { model_name: Some("Qwen3.6-35B-A3B".into()), ..SetupArgs::default() }, &small).unwrap_err().contains("unknown model"));
        assert!(ask(SetupArgs::default(), &machine_of(0, 0, 32 << 30)).unwrap_err().contains("none of the models"));
        assert!(ask(SetupArgs { model_name: Some("Ornith-1.5-122B".into()), ..SetupArgs::default() }, &small).unwrap_err().contains("unknown model"));
    }

    #[test]
    fn a_folder_that_holds_the_bundle_already_is_the_bundle_directory() {
        let scratch = Scratch::new("setup-folder");
        let entry = Entry::of("GSQ-RCO-IQ2_XXS").unwrap();
        let root = scratch.path();
        assert_eq!(bundle_directory(root, &entry), root.join("DeepSeek-V4.1-Flash").join("GSQ-RCO-IQ2_XXS"));
        std::fs::write(root.join(format!("{}.part", entry.files[0].name)), b"x").unwrap();
        assert_eq!(bundle_directory(root, &entry), root);
    }

    #[test]
    fn the_listing_names_the_recommendation_and_what_cannot_run() {
        let entries = catalog::pinned().unwrap();
        let listed = listing(&entries, &doggybox());
        let bundles = listed["bundles"].as_array().unwrap();
        assert_eq!(bundles.len(), entries.len());
        let recommended: Vec<_> = bundles.iter().filter(|bundle| bundle["recommended"] == true).collect();
        assert_eq!(recommended.len(), 1);
        assert_eq!(recommended[0]["quant"], "GSQ-RCO-IQ2_XXS");
        assert_eq!(recommended[0]["fit"], "ram");
        let q38 = bundles.iter().find(|bundle| bundle["quant"] == "IQ3_S").unwrap();
        assert_eq!(q38["extras"][0]["name"], "mtp");
        assert!(q38["disk_bytes"].as_u64().unwrap() > q38["download_bytes"].as_u64().unwrap() / 2);
        let none = listing(&entries, &machine_of(0, 0, 32 << 30));
        for bundle in none["bundles"].as_array().unwrap() {
            assert_eq!((bundle["fit"].as_str(), bundle["runs"].as_bool(), bundle["recommended"].as_bool()), (Some("no_gpu"), Some(false), Some(false)));
        }
    }
}
