//! Executables linking the GPU library find its vendor runtime where straydog-gpu found it, and say which commit they were built from.

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_STRAYDOG_GPU_RPATH");
    if let Ok(rpath) = std::env::var("DEP_STRAYDOG_GPU_RPATH") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    }
    // The commit the start's line names (`STRAYDOG_COMMIT` where a build has no git, as a release's source archive): a log then says
    // which build it is of, where a version alone repeats across a release's development builds.
    println!("cargo:rerun-if-env-changed=STRAYDOG_COMMIT");
    let git = |args: &[&str]| std::process::Command::new("git").args(args).output().ok()
        .filter(|output| output.status.success()).map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    let commit = std::env::var("STRAYDOG_COMMIT").ok().filter(|commit| !commit.is_empty())
        .or_else(|| git(&["rev-parse", "--short=7", "HEAD"])).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=STRAYDOG_COMMIT={commit}");
    // (a new commit changes HEAD, or the ref it names; a worktree's git dir is its own)
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        let dir = std::path::Path::new(&dir);
        println!("cargo:rerun-if-changed={}", dir.join("HEAD").display());
        if let Some(reference) = std::fs::read_to_string(dir.join("HEAD")).ok().and_then(|head| head.strip_prefix("ref: ").map(|r| r.trim().to_string())) {
            let common = git(&["rev-parse", "--git-common-dir"]).map(std::path::PathBuf::from).unwrap_or_else(|| dir.to_path_buf());
            println!("cargo:rerun-if-changed={}", common.join(reference).display());
        }
    }
}
