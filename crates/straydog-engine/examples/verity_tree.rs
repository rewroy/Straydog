//! Build the fs-verity trees of model files (src/verity.rs) into a directory, printing each file's
//! digest: verity_tree OUT_DIR FILE...
//! The tree of FILE is OUT_DIR/<file name>.verity.

use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [out, files @ ..] = &args[..] else {
        return Err("usage: verity_tree OUT_DIR FILE...".into());
    };
    std::fs::create_dir_all(out).map_err(|exc| exc.to_string())?;
    for path in files {
        let path = Path::new(path);
        let file = std::fs::File::open(path).map_err(|exc| format!("{}: {exc}", path.display()))?;
        let name = path.file_name().ok_or("a file without a name")?.to_string_lossy().into_owned();
        let tree = Path::new(out).join(format!("{name}.verity"));
        let started = std::time::Instant::now();
        let size = file.metadata().map_err(|exc| exc.to_string())?.len();
        let digest = straydog_engine::verity::build(&file, &tree, &|_| {}).map_err(|exc| exc.to_string())?;
        let seconds = started.elapsed().as_secs_f64();
        println!("{name} sha256:{} ({:.1} GB in {seconds:.1} s, {:.2} GB/s)",
                 digest.iter().map(|b| format!("{b:02x}")).collect::<String>(), size as f64 / 1e9, size as f64 / seconds / 1e9);
    }
    Ok(())
}
