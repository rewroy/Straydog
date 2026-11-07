# Straydog GPU library

The kernels and their host code in C++, compiled as HIP or CUDA. `straydog_gpu.h` is the
C ABI, the only interface the engine host uses; `crates/straydog-gpu/src/ffi.rs` is
generated from it, and a test keeps the two in step.

The executable links a static build: `cargo build --release --features hip` (or `cuda`)
runs CMake on this directory from `crates/straydog-gpu/build.rs`. Set
`STRAYDOG_GPU_ARCHITECTURES` (default `gfx1201`, or `75-real;86-real;89-real;120` for CUDA),
`ROCM_PATH` or `CUDA_PATH`.

On Linux with NVIDIA, install the Linux CUDA Toolkit and put its `bin` directory on
`PATH`; `CUDA_PATH` must name that same toolkit, for example:

```sh
export CUDA_PATH=/usr/local/cuda-12.8
export PATH="$CUDA_PATH/bin:$PATH"
# Optional: build only for the test machine (RTX 5070 Ti), including forward-compatible PTX.
export STRAYDOG_GPU_ARCHITECTURES=120
cargo build --release --features cuda
cargo test --release -p straydog-engine --features cuda --test gemm_gpu
```

For WSL2, install the NVIDIA driver on **Windows only**, then the Linux toolkit in
WSL following [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).
Check `/dev/dxg`, `/usr/lib/wsl/lib/nvidia-smi`, and `nvcc --version` inside WSL.
The isolated Linux worker exposes `/dev/dxg` only when present; the existing read-only
`/usr` mount includes the WSL driver libraries and toolkits installed under
`/usr/local`. Keep the toolkit at the path the executable was linked against so its
runtime search path also works inside isolation. A successful GPU test outside
bubblewrap does not establish that the isolated server can generate.
Large bundles read through `/mnt/c` can exceed the default 180-second startup
timeout; use the server's `--startup-timeout 600` option when needed.

Model tests require the pinned bundle and, where used, its pinned drafter. Set
`STRAYDOG_REQUIRE_MODEL=1` for model validation so missing model variables fail
instead of silently passing. The current `model_gpu` golden placements need two
GPUs with 30 GiB budgets or one with a 24 GiB budget; `runtime_gpu` uses two GPUs
with 30 GiB budgets and starts its test worker without isolation. Neither replaces
an isolated API generation on the actual hardware. WSL2 measurements are not
native Linux performance results.

The library implements compressed matrix-vector products and grouped GEMM for the
floating-point formats, Q2_0, Q4_0/Q5_0/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_NL/IQ4_XS and MXFP4, at any layer shape whose dimensions are whole blocks
(experts.hpp; `sd_expert_rows` runs the routed expert kernels for tests and measurements). RDNA4 grouped GEMM uses its own WMMA kernels; general GEMM
uses hipBLAS or cuBLAS, in FP32 or, when configured, FP16 inputs with FP32
accumulation. DeltaNet recurrence and sparse attention have their own kernels. The
engine's pinned GPU generations (`crates/straydog-engine/tests/model_gpu.rs`) check them
end to end, bit for bit.
