//! Builds gpu/ with CMake as a static library for the chosen backend and links it with
//! the vendor libraries it needs.
//!
//! Environment: STRAYDOG_GPU_ARCHITECTURES (default gfx1201 for HIP, "75-real;86-real;89-real;120"
//! for CUDA), ROCM_PATH (default /opt/rocm), CUDA_PATH (where the build finds the toolkit; the executable's
//! run path on Linux names /usr/local/cuda/lib64 and /usr/local/cuda-13/lib64, whatever CUDA_PATH is).

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../gpu");
    println!("cargo:rerun-if-changed={}", source.display());
    for name in ["STRAYDOG_GPU_ARCHITECTURES", "ROCM_PATH", "CUDA_PATH", "CMAKE_GENERATOR"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let hip = env::var_os("CARGO_FEATURE_HIP").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    if hip && cuda {
        panic!("choose one GPU backend: the hip or the cuda feature");
    }
    if !hip && !cuda {
        return;
    }
    let backend = if hip { "HIP" } else { "CUDA" };
    let architectures = env::var("STRAYDOG_GPU_ARCHITECTURES")
        .unwrap_or_else(|_| if hip { "gfx1201" } else { "75-real;86-real;89-real;120" }.to_string());
    let windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";

    let mut config = cmake::Config::new(&source);
    config
        .define("STRAYDOG_BACKEND", backend)
        .define(format!("CMAKE_{backend}_ARCHITECTURES"), &architectures)
        .profile("Release")
        .build_target("straydog_gpu");
    let rocm = PathBuf::from(env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string()));
    if hip {
        config.define("CMAKE_PREFIX_PATH", &rocm);
    }
    if windows {
        // The static C/C++ runtime, as the executable links it (crt-static).
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
        // The Visual Studio generator compiles CUDA only with the toolkit's MSBuild integration.
        if env::var_os("CMAKE_GENERATOR").is_none() {
            config.generator("Ninja");
        }
    }
    let build = config.build().join("build");
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-search=native={}", build.join("Release").display());
    println!("cargo:rustc-link-lib=static=straydog_gpu");

    let runtime = if hip {
        let lib = rocm.join("lib");
        link(&lib, &["amdhip64", "hipblas"]);
        lib
    } else {
        let root = PathBuf::from(env::var("CUDA_PATH").expect("CUDA_PATH must name the CUDA Toolkit"));
        let lib = root.join(if windows { "lib/x64" } else { "lib64" });
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=static=cudart_static");
        link(&lib, &["cublas"]);
        if !windows {
            link(&lib, &["dl", "rt"]);
        }
        lib
    };
    if !windows {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // Where the vendor runtime is found at run time. HIP: the one built against, then the standard link. CUDA: the places NVIDIA's
        // packages and runfile install to (/usr/local/cuda links to the version in use) and not the toolkit this was built with, so that
        // no build path is in the executable; a runtime that ldconfig knows is found by the loader's cache anyway.
        // Dependents' build scripts read this as DEP_STRAYDOG_GPU_RPATH.
        let rpath = if hip {
            format!("{}:/opt/rocm/lib", runtime.display())
        } else {
            "/usr/local/cuda/lib64:/usr/local/cuda-13/lib64".to_string()
        };
        println!("cargo:rpath={rpath}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    }
}

fn link(directory: &Path, names: &[&str]) {
    println!("cargo:rustc-link-search=native={}", directory.display());
    for name in names {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
}
