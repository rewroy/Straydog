# Straydog

A local inference engine for Qwen3.8-Flash-Next with an OpenAI-compatible server, in one
executable per GPU backend.

- **No hidden shortcuts in the numbers.** Weights run in the format their GGUF ships,
  decoded exactly; activations and the K/V cache are never quantized; every expert a
  token routes to is computed. The one rounding on by default, FP16 operands in prompt
  batches' products, is named below and `--gemm fp32` turns it off. Speculative decoding
  never changes an answer. See [Numerics](#numerics).
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
straydog -m ~/models/IQ3_S -md ~/models/mtp -gpu 0,1 --api-key-file ~/.straydog-key
```

Each GPU's VRAM budget is its free memory unless `--vram` gives one (`--vram 28`) or one per
GPU (`--vram 30,10` for a 32 GB and a 12 GB card).

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

## Numerics

Faster engines often trade precision for speed without saying so: weights repacked to 8
or 4 bits, activations rounded to int8 before every product, K/V caches in FP8, experts
skipped. Bandwidth and compute have physical limits, so Straydog finds its speed in not
repeating work (K/V mirrors, conversation checkpoints, row and expert caches, reads and
copies overlapped with compute, fused kernels, both GPUs at once) and states every
rounding it does make.

What Straydog does not do:

- Re-quantize weights. Each GGUF type is decoded exactly as ggml defines it (checked bit
  for bit against reference decoders); products accumulate in F32.
- Quantize activations. No int8 or q8 inputs to any product: decode and speculative
  verification run on F32 activations.
- Quantize the K/V cache. It is F32.
- Skip or approximate work. Every routed expert a token selects is computed, attention is
  never replaced by an approximate replay, and the output head is the model's.
- Let speculation change answers. Drafts are only proposals: a verification pass is
  bitwise identical to token-by-token decode in the same placement.
- Convert silently. An unsupported tensor type is an error.

Where it rounds, and how to turn it off:

| Where | Default | Exact |
| --- | --- | --- |
| Qwen products over prompt batches (2 rows or more) | FP16 operands, F32 sums (`--gemm fp16`) | `--gemm fp32` |
| Long prompts' attention on RDNA4 | FP16 products, F32 softmax and sums | `--gemm fp32` |
| DeepSeek shared/routed FFN products (more than 4 rows) | FP16 operands, F32 sums and output | `--gemm fp32` |
| Decode and speculative verification | F32 | |
| MTP drafter | its experts quantized to Q2_0 when it is built: this changes only how many drafts are accepted, never the output | run without `-md` |
| Experts beyond VRAM (partial residency) | the CPU computes them in F32 and prompt batches use the FP16 expert GEMM; which experts run where depends on timing, so decode is not bit-reproducible from run to run | full residency |

How this is checked: the pinned GPU generations reproduce every greedy token, logits
vector and final hidden state bit for bit in each placement; a deliberate numerical change
is first compared with an F32-GEMM run, by KL divergence and top-1 agreement, then pinned
anew, and F32 products take the same kernels in every process (rocBLAS's own library: on
RDNA4, hipBLASLt's choice of F32 kernels changed from one process to another).
With the FP16 long-prompt attention, KL against `--gemm fp32` averaged 3.4e-3 over 12
prompts of 8K-40K (3.9e-3 before it; top-1 12/12), while reordering one sum alone moves it
by 3.0e-3. DeepSeek-V4.1 rounds only the inputs and weights of shared/routed FFN
products with more than four rows to FP16; their sums and output remain F32. Norms,
router/index projections, attention, HC, residuals and K/V remain F32. `--gemm fp32`
also keeps those FFN products in F32. With the IQ2_XS model, the FP16 FFN path agrees
with F32 on 772/774 next-token choices in six short probes for each of TP1 and TP2;
a native 4083-token prompt plus 16 teacher-forced steps agrees on all 17 choices, with
mean KL 5.3e-4 (TP1) and 3.2e-4 (TP2). Same-mode decode and speculative verification
remain bitwise identical. `examples/ds41_quality --gemm fp32|fp16` and
`examples/ds41_reference` provide the numerical checks; the quality example defaults
to F32.

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
# GPU tests, one at a time: the pinned generations bit for bit, the runtime end to end, the
# F32 products' bits the same in every process
STRAYDOG_MODEL=DIR STRAYDOG_DRAFTER=DIR cargo test -p straydog-engine --features hip --test model_gpu
STRAYDOG_MODEL=DIR cargo test -p straydog --features hip --test runtime_gpu
cargo test -p straydog-engine --features hip --test gemm_gpu
# A/B of this build against another
cargo run --release --features hip --example ab -- --baseline BIN --model DIR --drafter DIR --output ab.json
# The dashboard over a simulated engine (no GPU, no model): http://127.0.0.1:18090/dashboard
cargo run --release --features hip --example dashboard
```

## License

MIT
