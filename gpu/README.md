# Straydog GPU library

The kernels and their host code in C++, compiled as HIP or CUDA. `straydog_gpu.h` is the
C ABI, the only interface the engine host uses; `crates/straydog-gpu/src/ffi.rs` is
generated from it, and a test keeps the two in step.

The executable links a static build: `cargo build --release --features hip` (or `cuda`)
runs CMake on this directory from `crates/straydog-gpu/build.rs`. Set
`STRAYDOG_GPU_ARCHITECTURES` (default `gfx1201`, or `75-real;86-real;89-real;120` for CUDA),
`ROCM_PATH` or `CUDA_PATH`.

On Linux with an NVIDIA GPU, install the Linux CUDA Toolkit, set `CUDA_PATH` to it and put its
`bin` on `PATH` before `cargo build --release --features cuda`. Under WSL2 the NVIDIA driver is
installed on Windows only and the toolkit inside WSL; `/dev/dxg` and
`/usr/lib/wsl/lib/nvidia-smi` show that the GPU is reachable. The isolated worker is given
`/dev/dxg` when it exists, and reads `/usr` as the host has it, so keep the toolkit where the
executable was linked against it: its runtime is then found inside the isolation too.

The library implements compressed matrix-vector products and grouped GEMM for the
floating-point formats, Q2_0, Q4_0/Q5_0/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_NL/IQ4_XS and MXFP4, at any layer shape whose dimensions are whole blocks
(experts.hpp; `sd_expert_rows` runs the routed expert kernels for tests and measurements). RDNA4 grouped GEMM uses its own WMMA kernels; general GEMM
uses hipBLAS or cuBLAS, in FP32 or, when configured, FP16 inputs with FP32
accumulation. DeltaNet recurrence and sparse attention have their own kernels. The
engine's pinned GPU generations (`crates/straydog-engine/tests/model_gpu.rs`) check them
end to end, bit for bit.
