//! Executables linking the GPU library find its vendor runtime where straydog-gpu found it.

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_STRAYDOG_GPU_RPATH");
    if let Ok(rpath) = std::env::var("DEP_STRAYDOG_GPU_RPATH") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    }
}
