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

## Run

```bash
straydog fetch -m ~/models/IQ3_S --quant IQ3_S      # download and authenticate a bundle (once)
straydog fetch-mtp --model-draft ~/models/mtp       # build the MTP drafter (once, optional)
straydog -m ~/models/IQ3_S -md ~/models/mtp -dev 0,1 --vram 30 --api-key-file ~/.straydog-key
```

Like llama-server, `straydog` with no subcommand is the server (http://127.0.0.1:8080).
`-m` takes the bundle's directory (both GGUF shards with `tokenizer.json`,
`chat_template.jinja` and `config.json`) or either shard; `-md` the drafter's directory
or its GGUF.
Every setting is an option; each also reads a `STRAYDOG_*` variable (`STRAYDOG_CTX_SIZE`),
and `--config FILE` reads them from a TOML file. Options beat variables, which beat the
file. `--print-config` shows the settings in effect and where each came from.

| Bundle | Size | |
| --- | ---: | --- |
| Q2_0 | 66.4 GB | the fastest |
| IQ2_XS | 68.0 GB | |
| IQ3_XXS | 75.8 GB | |
| IQ3_S | 83.6 GB | closest to BF16 |

## Use

Any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $(cat ~/.straydog-key)" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Answers carry `reasoning_content`, structured `tool_calls`, `usage` with
`prompt_tokens_details.cached_tokens` and the engine's `timings`. Sampling follows the
model card unless a request or the server says otherwise: thinking `temperature=1.0,
top_p=0.95, top_k=20`, non-thinking `temperature=0.7, top_p=0.8, top_k=20,
presence_penalty=1.5`; `temperature: 0` is greedy. Sampling controls the engine does
not implement are refused (400), never ignored. A client that returns the reasoning in
its assistant messages keeps the conversation cache warm.

Like llama-server, `GET /props` (`default_generation_settings.n_ctx`) and `GET /slots`
report the context, and `GET /health` the loading progress.

## Dashboard

Open http://127.0.0.1:8080/dashboard in a browser. On one screen: each GPU's load,
temperatures, power and memory, the CPU's and the memory's, as charts of the last two
minutes; prefill and decode speed live and per request; the engine's state. The
environment and the service log are tabs, and a chat opens over the page. Sensors are
read only while a dashboard is open. With an API key set, the page asks for it and
keeps it in that tab.

## Development

```bash
# Rust tests; STRAYDOG_MODEL enables the golden tests on a real bundle
ROCM_PATH=/opt/rocm/core-10.0 cargo test --workspace --features straydog/hip,straydog-provision/hip
# GPU tests, one at a time: the pinned generations bit for bit, the runtime end to end
STRAYDOG_MODEL=DIR STRAYDOG_DRAFTER=DIR cargo test -p straydog-engine --features hip --test model_gpu
STRAYDOG_MODEL=DIR cargo test -p straydog --features hip --test runtime_gpu
# A/B of this build against another
cargo run --release --features hip --example ab -- --baseline BIN --model DIR --drafter DIR --output ab.json
# The dashboard over a simulated engine (no GPU, no model): http://127.0.0.1:18090/dashboard
cargo run --release --features hip --example dashboard
```

## License

MIT
