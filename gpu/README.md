# Straydog GPU library

The kernels and their host code in C++, compiled as HIP or CUDA. `straydog_gpu.h` is the
C ABI, the only interface the engine host uses; `crates/straydog-gpu/src/ffi.rs` is
generated from it, and a test keeps the two in step.

Builds:
- **The executable** links a static build: `cargo build --release --features hip` (or
  `cuda`) runs CMake on this directory from `crates/straydog-gpu/build.rs`. Set
  `STRAYDOG_GPU_ARCHITECTURES` (default `gfx1201`, or `86-real;89-real;120` for CUDA),
  `ROCM_PATH` or `CUDA_PATH`.
- **The kernel tests** load a shared build from Python:
  `uv run straydog build --backend HIP --architecture gfx1201 --prefix /opt/rocm/core-10.0`,
  then `STRAYDOG_GPU_LIBRARY=<library> uv run pytest -m gpu`.

The library implements compressed matrix-vector products and grouped GEMM for the
floating-point formats, Q2_0, Q4_0/Q5_0/Q8_0, Q3_K-Q6_K and IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_NL/IQ4_XS. RDNA4 grouped GEMM uses its own WMMA kernels; general GEMM
uses hipBLAS or cuBLAS, in FP32 or, when configured, FP16 inputs with FP32
accumulation. DeltaNet recurrence and sparse attention have their own kernels. Every
format and state update is compared with a CPU reference in the tests.
