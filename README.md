# Straydog

A local inference engine for Qwen3.8-Flash-Next with an OpenAI-compatible server, in one
executable per GPU backend.

- **Exact.** Activations are never quantized; weights decode exactly to F32. MTP
  speculative decoding is bitwise identical to token-by-token decode.
- **Isolated.** The engine runs in a worker without network access (bubblewrap on
  Linux, its own AppContainer on Windows). Every model and drafter file is pinned by
  SHA-256 and authenticated before use.
- **Fast on one or several GPUs.** Full residency when the model fits, GPU and CPU
  expert slots when it does not; with two GPUs holding the whole model, every layer's
  experts are split between them.

Design: [docs/DESIGN.md](docs/DESIGN.md).

## Build

Linux (ROCm 10, Radeon AI PRO R9700 / RX 9070 by default):

```bash
ROCM_PATH=/opt/rocm/core-10.0 cargo build --release --features hip
# other AMD GPUs: STRAYDOG_GPU_ARCHITECTURES="gfx1100;gfx1201"
```

Windows (CUDA 12, from an "x64 Native Tools Command Prompt" with Ninja and the CUDA
Toolkit; RTX 30/40/50 by default):

```bat
cargo build --release --features cuda
rem other GPUs: set STRAYDOG_GPU_ARCHITECTURES=75-real;86-real
```

The result is `target/release/straydog`. It needs the GPU vendor's runtime: ROCm's
`libamdhip64` and `libhipblas` on Linux, the NVIDIA driver and `cublas64_12.dll`
(next to the executable or from the CUDA Toolkit) on Windows. Linux also needs
bubblewrap.

## Set up and serve

```bash
straydog init      # choose a bundle and GPUs, fetch the model and drafter, write ~/.straydog/config.toml
straydog serve     # http://127.0.0.1:8080
```

`init` asks for what it needs; `straydog init --yes` takes the defaults. Settings work
like llama.cpp's server: `straydog serve -m DIR -md DIR -dev 0,1 --vram 30 -c 262144`.
Options beat `STRAYDOG_*` variables, which beat `~/.straydog/config.toml`;
`straydog serve --print-config` shows what is in effect and where each value came from.

| Bundle | Size | |
| --- | ---: | --- |
| Q2_0 | 66.4 GB | the fastest |
| IQ2_XS | 68.0 GB | |
| IQ3_XXS | 75.8 GB | |
| IQ3_S | 83.6 GB | closest to BF16 |

## Use

Any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $(cat ~/.straydog/api-key)" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Answers carry `reasoning_content`, structured `tool_calls`, `usage` with
`prompt_tokens_details.cached_tokens` and the engine's `timings`. Sampling controls the
engine does not implement are refused (400), never ignored. A client that returns the
reasoning in its assistant messages keeps the conversation cache warm.

## Development

```bash
# Rust tests; STRAYDOG_MODEL enables the golden tests on a real bundle
ROCM_PATH=/opt/rocm/core-10.0 cargo test --workspace --features straydog/hip,straydog-provision/hip
# GPU tests, one at a time: model parity with the reference, the runtime end to end
STRAYDOG_MODEL=DIR STRAYDOG_DRAFTER=DIR cargo test -p straydog-engine --features hip --test model_gpu
STRAYDOG_MODEL=DIR cargo test -p straydog --features hip --test runtime_gpu
# The Python reference (src/straydog) and its kernel tests
uv sync && uv run pytest && STRAYDOG_GPU_LIBRARY=$PWD/src/straydog/native/libstraydog_gpu.so uv run pytest -m gpu
# A/B against the reference
cargo run --release --features hip --example ab -- --model DIR --drafter DIR --output ab.json
```

## License

MIT
