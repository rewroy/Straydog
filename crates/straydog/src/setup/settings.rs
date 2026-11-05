//! The settings file `straydog setup` writes: the model it chose, merged into whatever the file
//! already says of the rest.

use std::path::{Path, PathBuf};

use toml::{Table, Value};

/// What setup decides; everything else in the file is the person's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// The bundle's directory.
    pub model: PathBuf,
    /// The bundle's labels: its model, provider and quantization.
    pub model_name: String,
    pub provider: String,
    pub quant: String,
    /// The engram table it was set up with instead of its own (Unsloth's Qwen3.8), named for the server to read.
    pub engram: Option<String>,
    /// The MTP drafter's directory, when the model has one.
    pub model_draft: Option<PathBuf>,
    /// GPU ordinals, when not just the first.
    pub gpu: Option<Vec<usize>>,
    /// The context in tokens, when the model starts with less than the server's default.
    pub ctx_size: Option<u32>,
    /// Where the verity trees go, when not beside the model.
    pub data_dir: Option<PathBuf>,
}

/// What writing did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Written {
    pub path: PathBuf,
    /// False when the file said all of it already.
    pub changed: bool,
    /// Where the earlier file went, when a different one was replaced.
    pub kept: Option<PathBuf>,
    /// Where the server will listen, from the file's `host` and `port` (else the defaults).
    pub url: String,
}

const HEADER: &str = "# Written by `straydog setup`; a plain `straydog` reads it. Options and STRAYDOG_* variables\n\
                      # beat it. Setup changes model, model_name, provider, quant, engram, model_draft, gpu and data_dir, lowers ctx_size to what the model\n\
                      # starts with, and keeps the rest.\n";

fn text(path: &Path) -> Result<Value, String> {
    path.to_str().map(|text| Value::String(text.to_string())).ok_or_else(|| format!("{} is not valid UTF-8, which a settings file needs", path.display()))
}

/// The table the file will hold: `old` (what the file held) with `settings` over it.
fn merged(old: Table, settings: &Settings) -> Result<Table, String> {
    let mut table = old;
    table.insert("model".into(), text(&settings.model)?);
    table.insert("model_name".into(), Value::String(settings.model_name.clone()));
    table.insert("provider".into(), Value::String(settings.provider.clone()));
    table.insert("quant".into(), Value::String(settings.quant.clone()));
    for (key, value) in [
        ("model_draft", settings.model_draft.as_deref().map(text).transpose()?),
        ("gpu", settings.gpu.as_ref().map(|gpu| Value::Array(gpu.iter().map(|&ordinal| Value::Integer(ordinal as i64)).collect()))),
        ("data_dir", settings.data_dir.as_deref().map(text).transpose()?),
        ("engram", settings.engram.clone().map(Value::String)),
    ] {
        match value {
            Some(value) => table.insert(key.into(), value),
            None => table.remove(key),
        };
    }
    // (a context the person set smaller stays; one the model cannot start with is lowered to what it can)
    if let Some(tokens) = settings.ctx_size {
        let kept = table.get("ctx_size").and_then(Value::as_integer).filter(|&earlier| earlier <= i64::from(tokens));
        table.insert("ctx_size".into(), Value::Integer(kept.unwrap_or_else(|| i64::from(tokens))));
    }
    Ok(table)
}

/// Write `settings` into the file at `path` (made, with its folders, if missing), keeping its
/// other settings. A different earlier file is kept beside it as `<name>.old`. Written whole to
/// a part file and renamed, so an interrupted setup leaves the file as it was.
pub fn write_settings(path: &Path, settings: &Settings) -> Result<Written, String> {
    let failed = |error: std::io::Error| format!("{}: {error}", path.display());
    let earlier = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(failed(error)),
    };
    // (a file that is not TOML is replaced, and kept as .old)
    let old: Table = earlier.as_deref().and_then(|text| text.parse().ok()).unwrap_or_default();
    let table = merged(old, settings)?;
    let body = format!("{HEADER}{}", toml::to_string(&table).map_err(|error| error.to_string())?);
    let url = {
        let host = table.get("host").and_then(Value::as_str).filter(|host| *host != "0.0.0.0" && *host != "::").unwrap_or("127.0.0.1");
        format!("http://{host}:{}", table.get("port").and_then(Value::as_integer).unwrap_or(8080))
    };
    if earlier.as_deref() == Some(body.as_str()) {
        return Ok(Written { path: path.to_path_buf(), changed: false, kept: None, url });
    }
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(failed)?;
    }
    let sibling = |suffix: &str| {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(suffix);
        path.with_file_name(name)
    };
    let kept = if earlier.is_some() {
        let old = sibling(".old");
        std::fs::copy(path, &old).map_err(failed)?;
        Some(old)
    } else {
        None
    };
    let part = sibling(".part");
    std::fs::write(&part, body).and_then(|()| std::fs::rename(&part, path)).map_err(failed)?;
    Ok(Written { path: path.to_path_buf(), changed: true, kept, url })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Options;

    fn scratch(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("straydog-settings-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn settings() -> Settings {
        Settings { model: "/models/DeepSeek-V4.1-Flash-1017/ISTA-DASLab-IQ2_XXS".into(), model_name: "DeepSeek-V4.1-Flash-1017".into(), provider: "ISTA-DASLab".into(),
                   quant: "IQ2_XXS".into(), engram: None, model_draft: None, gpu: Some(vec![0, 1]), ctx_size: None, data_dir: None }
    }

    /// What the server reads back is what setup chose; a file's other settings stay, the model's own
    /// (a drafter of the model before) go, and the earlier file is kept.
    #[test]
    fn the_file_setup_writes_reads_back_and_keeps_what_else_it_says() {
        let root = scratch("merge");
        let path = root.join("config").join("config.toml");
        let first = write_settings(&path, &settings()).unwrap();
        assert_eq!((first.changed, first.kept, first.url.as_str()), (true, None, "http://127.0.0.1:8080"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Written by `straydog setup`"));
        let read: Options = toml::from_str(&text).unwrap();
        assert_eq!((read.model.as_deref(), read.quant.as_deref(), read.gpu.clone(), read.model_draft.clone()),
                   (Some(Path::new("/models/DeepSeek-V4.1-Flash-1017/ISTA-DASLab-IQ2_XXS")), Some("IQ2_XXS"), Some(vec![0, 1]), None));
        assert_eq!((read.model_name.as_deref(), read.provider.as_deref()), (Some("DeepSeek-V4.1-Flash-1017"), Some("ISTA-DASLab")));
        // the same again changes nothing
        let again = write_settings(&path, &settings()).unwrap();
        assert_eq!((again.changed, again.kept), (false, None));
        assert!(!root.join("config").join("config.toml.old").exists());
        // the person's own settings, and a drafter of the model before
        std::fs::write(&path, "# mine\nport = 9000\nhost = \"0.0.0.0\"\nctx_size = 32768\nmodel_draft = \"/old/mtp\"\ndata_dir = \"/old/trees\"\nquant = \"Q2_0\"\n").unwrap();
        let second = write_settings(&path, &settings()).unwrap();
        assert_eq!((second.changed, second.url.as_str()), (true, "http://127.0.0.1:9000"));
        let read: Options = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!((read.port, read.host.as_deref(), read.ctx_size, read.model_draft, read.data_dir, read.quant.as_deref()),
                   (Some(9000), Some("0.0.0.0"), Some(32768), None, None, Some("IQ2_XXS")));
        let old = second.kept.unwrap();
        assert!(std::fs::read_to_string(old).unwrap().starts_with("# mine"));
        // a context: lowered when the model starts with less, left as it is when the person's is less
        let small = Settings { ctx_size: Some(65536), ..settings() };
        write_settings(&path, &small).unwrap();
        let ctx = |path: &Path| toml::from_str::<Options>(&std::fs::read_to_string(path).unwrap()).unwrap().ctx_size;
        assert_eq!(ctx(&path), Some(32768));
        std::fs::write(&path, "ctx_size = 262144\nport = 9000\n").unwrap();
        write_settings(&path, &small).unwrap();
        assert_eq!(ctx(&path), Some(65536));
        // a drafter and a data directory, when the model has them
        let with = Settings { model_draft: Some("/models/Qwen3.8-Flash-Next/mtp".into()), data_dir: Some("/trees".into()), gpu: None, ..settings() };
        write_settings(&path, &with).unwrap();
        let read: Options = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!((read.model_draft.as_deref(), read.data_dir.as_deref(), read.gpu), (Some(Path::new("/models/Qwen3.8-Flash-Next/mtp")), Some(Path::new("/trees")), None));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_file_that_is_not_toml_is_replaced_and_kept_and_an_interrupted_write_leaves_the_file() {
        let root = scratch("broken");
        let path = root.join("config.toml");
        std::fs::write(&path, "this is [not toml").unwrap();
        let written = write_settings(&path, &settings()).unwrap();
        assert_eq!(std::fs::read_to_string(written.kept.unwrap()).unwrap(), "this is [not toml");
        assert!(toml::from_str::<Options>(&std::fs::read_to_string(&path).unwrap()).is_ok());
        // the new text is written to a part file and renamed: a part file left by an interrupted run
        // is not the file, and the next run replaces it
        let before = std::fs::read_to_string(&path).unwrap();
        std::fs::write(root.join("config.toml.part"), "half a fi").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
        write_settings(&path, &Settings { quant: "Q2_0".into(), ..settings() }).unwrap();
        assert!(!root.join("config.toml.part").exists());
        assert!(std::fs::read_to_string(&path).unwrap().contains("Q2_0"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
