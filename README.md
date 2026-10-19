# Straydog

A local inference engine for Qwen3.8-Flash-Next with an OpenAI-compatible server.

- **Exact.** Activations are never quantized; weights decode exactly to F32. MTP
  speculative decoding is bitwise identical to token-by-token decode.
- **Isolated.** The engine runs in a worker without network access (bubblewrap on
  Linux, its own AppContainer on Windows). Every model, drafter and native file is
  pinned by SHA-256 and authenticated before use.
- **Fast on one or several GPUs.** Full residency when the model fits, GPU and CPU
  expert slots when it does not; with two GPUs holding the whole model, every
  layer's experts are split between them.

Design: [docs/DESIGN.md](docs/DESIGN.md).

## Status

Phase 1 of the [design](docs/DESIGN.md#7-phases): the engine, the native sources,
the packaged builds, provisioning and the benchmark. `straydog serve` comes next.

## Requirements

- Python 3.11+, [uv](https://docs.astral.sh/uv/).
- Linux x86_64 with bubblewrap and ROCm 10.0 (packaged build for gfx1201, Radeon AI
  PRO R9700 / RX 9070), or Windows x64 with an NVIDIA driver for CUDA 12 (packaged
  build for RTX 30/40/50).
- Other GPUs: build the native components (Rust, CMake 3.24+, and the HIP/hipBLAS or
  CUDA/cuBLAS development files).

## Models

The model is one of four pinned bundles of
[ISTA-DASLab's GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):

| Bundle | Size | |
| --- | ---: | --- |
| Q2_0 | 66.4 GB | fastest |
| IQ2_XS | 68.0 GB | |
| IQ3_XXS | 75.8 GB | |
| IQ3_S | 83.6 GB | closest to BF16 |

```bash
uv sync
uv run straydog fetch --root ~/.straydog/models/IQ3_S --quantization IQ3_S
uv run straydog verify --root ~/.straydog/models/IQ3_S --quantization IQ3_S
# MTP drafter for speculative decoding, built from the official tensors
uv run straydog fetch-mtp --root ~/.straydog/models/mtp --gpu-library src/straydog/native/libstraydog_gpu.so
```

## Native builds

```bash
uv run straydog build --backend HIP --architecture gfx1201 --prefix /opt/rocm/core-10.0
# After changing native/crates/inference: rebuild both packaged builds
uv run straydog bundle --backend HIP --architecture gfx1201 --prefix /opt/rocm/core-10.0   # Linux
uv run straydog bundle --backend CUDA --architecture "86-real;89-real;120"                 # Windows
```

## Tests and benchmarks

```bash
uv run pytest                                   # CPU
STRAYDOG_GPU_LIBRARY=$PWD/src/straydog/native/libstraydog_gpu.so uv run pytest -m gpu
uv run python scripts/benchmark.py -m ~/.straydog/models/IQ3_S -md ~/.straydog/models/mtp \
  -dev 0,1 --vram 30 --case all --repeat 2 --output report.json
```

`scripts/benchmark.py` takes the same options as `straydog serve`.

## License

MIT
