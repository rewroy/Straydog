# Straydog GPU library

The kernels and their host code in C++, compiled as HIP or CUDA. `straydog_gpu.h` is the
C ABI, the only interface the engine host uses; `crates/straydog-gpu/src/ffi.rs` is
generated from it, and a test keeps the two in step.

The executable links a static build: `cargo build --release --features hip` (or `cuda`)
runs CMake on this directory from `crates/straydog-gpu/build.rs`. Set
`STRAYDOG_GPU_ARCHITECTURES` (default `gfx1201`, or `86-real;89-real;120` for CUDA),
`ROCM_PATH` or `CUDA_PATH`.

The library implements compressed matrix-vector products and grouped GEMM for the
floating-point formats, Q2_0, Q4_0/Q5_0/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_NL/IQ4_XS and MXFP4, at any layer shape whose dimensions are whole blocks
(experts.hpp; `sd_expert_rows` runs the routed expert kernels for tests and measurements). RDNA4 grouped GEMM uses its own WMMA kernels; general GEMM
uses hipBLAS or cuBLAS, in FP32 or, when configured, FP16 inputs with FP32
accumulation. DeltaNet recurrence and sparse attention have their own kernels. The
engine's pinned GPU generations (`crates/straydog-engine/tests/model_gpu.rs`) check them
end to end, bit for bit.
