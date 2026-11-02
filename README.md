# Straydog

A local inference engine with an OpenAI-compatible server, in one executable per GPU
backend. It serves pinned GGUF bundles of Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash.

- **Every rounding named.** Weights run in the format their GGUF ships, decoded exactly,
  and every expert a token routes to is computed. The roundings that buy speed (FP16
  operands in prompt batches; for DeepSeek-V4.1 also INT8 and FP16 inputs in decode and
  BF16 compressed K/V) are listed under [Numerics](#numerics), each with its switch where
  it has one. Speculative decoding never changes an answer.
- **Isolated.** The engine runs in a worker without network access (bubblewrap on
  Linux, its own AppContainer on Windows). Every model and drafter file is pinned by
  SHA-256 and authenticated when it is downloaded.
- **Fast on one or several GPUs.** Full residency when the model fits. When it does not,
  the experts VRAM lacks wait in host memory: the Qwen models compute them on the CPU, and
  DeepSeek-V4.1 uploads each one a token routes to into VRAM. Two GPUs of similar budgets
  split every layer between them (Qwen3.8 when both together hold the whole model).

Design: [docs/DESIGN.md](docs/DESIGN.md).

## Build

Linux (ROCm 10, Radeon AI PRO R9700 / RX 9070 by default):

```bash
ROCM_PATH=/opt/rocm/core-10.0 cargo build --release --features hip
# other AMD GPUs: STRAYDOG_GPU_ARCHITECTURES="gfx1100;gfx1201"
```

Windows (CUDA 13, from an "x64 Native Tools Command Prompt" with Ninja and the CUDA
Toolkit; RTX 30/40/50 by default):

```bat
cargo build --release --features cuda
rem other GPUs (the default is 75-real;86-real;89-real;120): set STRAYDOG_GPU_ARCHITECTURES=89-real
```

The result is `target/release/straydog`. It needs the GPU vendor's runtime: ROCm's
`libamdhip64` and `libhipblas` on Linux, the NVIDIA driver and `cublas64_13.dll`
(next to the executable or from the CUDA Toolkit) on Windows. Linux also needs
bubblewrap. DeepSeek-V4.1 has been run on Linux with ROCm only.

## Run

```bash
# the first run: look at this machine, choose a model and where it goes, download and
# authenticate it, write the settings
straydog setup
# then serve it (http://127.0.0.1:8080)
straydog
```

`straydog setup` lists the pinned models that can run on the machine it finds, with their
sizes and what each needs, and recommends one; it asks where to put it (by default the
platform's data folder), checks the free space, resumes or verifies what is there already, adds
the extras you choose (DSpark and vision for DeepSeek-V4.1, MTP and vision for Qwen3.8) and writes
`config.toml` in the platform's config folder, which a plain `straydog` reads. Every answer is
also an option: `straydog setup --yes --quant GSQ-RCO-IQ2_XXS --dir ~/models` asks nothing
(`--offline` downloads nothing, `--config FILE` writes the settings elsewhere).

By hand, and for scripts that name their model:

```bash
# DeepSeek-V4.1-Flash: download and authenticate a bundle (once), then serve it
straydog fetch -m ~/models/ds41 --quant GSQ-RCO-IQ2_XXS
straydog -m ~/models/ds41 -gpu 0,1 --api-key-file ~/.straydog-key

# Qwen3.8-Flash-Next with its MTP drafter
straydog fetch -m ~/models/IQ3_S --quant IQ3_S
straydog fetch-mtp --model-draft ~/models/mtp       # build the MTP drafter (once, optional)
straydog -m ~/models/IQ3_S -md ~/models/mtp -gpu 0,1 --api-key-file ~/.straydog-key
```

For DeepSeek-V4.1, `fetch` also downloads the DSpark drafter (into `dspark/` in the bundle's
directory) and builds the vision encoder's file from the official checkpoint's tensors; the
server finds both there and runs without either. Qwen3.8 reads images with
`mmproj-Qwen3.8-Flash-Next-BF16.gguf` from its GGUF repository (`straydog setup` fetches it): in
the bundle's directory or the one above it, or named by `--mmproj`.

Each GPU's VRAM budget is its free memory less a margin (512 MiB or 3%; on Windows, where the
desktop shares the GPU, an eighth of a small card and 256 MiB, and 2 GiB from 16 GB up) unless `--vram` gives
one (`--vram 28`) or one per GPU (`--vram 30,10` for a 32 GB and a 12 GB card).

Like llama-server, `straydog` with no subcommand is the server (http://127.0.0.1:8080).
`-m` takes the bundle's directory or its first shard; `-md` the MTP drafter's directory
or its GGUF.
Every setting is an option; each also reads a `STRAYDOG_*` variable (`STRAYDOG_CTX_SIZE`),
and `--config FILE` reads them from a TOML file (with no `--config` and no model given, the one
`setup` wrote). Options beat variables, which beat the file. `--print-config` shows the
settings in effect and where each came from.

| Model | Bundle (`--quant`) | Size | |
| --- | --- | ---: | --- |
| DeepSeek-V4.1-Flash | GSQ-RCO-IQ2_XXS | 248.8 GB | ISTA's recommended build; the faster where VRAM is short |
| | GSQ-RCO-IQ2_XS | 265.8 GB | |
| | UD-IQ2_XXS | 300.5 GB | Unsloth's |
| Qwen3.8-Flash-Next | Q2_0 | 66.4 GB | the fastest |
| | IQ2_XS | 68.0 GB | |
| | IQ3_XXS | 75.8 GB | |
| | IQ3_S | 83.6 GB | closest to BF16 |

DeepSeek-V4.1's sizes include its 110.6 GB Engram tables, which stay on the SSD and are read
a few rows per token; its DSpark drafter adds 8.0 GB. On two R9700s and 128 GB of RAM,
GSQ-RCO-IQ2_XXS decodes 57-77 tokens/s (prose to code) once a request's experts are in memory.

## Use

Any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $(cat ~/.straydog-key)" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Answers carry `reasoning_content`, structured `tool_calls`, `usage` with
`prompt_tokens_details.cached_tokens` and the engine's `timings`. Sampling follows the
model card unless a request or the server says otherwise. Qwen3.8's card: thinking
`temperature=1.0, top_p=0.95, top_k=20`, non-thinking `temperature=0.7, top_p=0.8,
top_k=20, presence_penalty=1.5`. A GGUF with no `general.sampling.*` (DeepSeek-V4.1's)
samples at `temperature=1.0` without truncation. `temperature: 0` is greedy. Sampling
controls the engine does not implement are refused (400), never ignored. A client that
returns the reasoning in its assistant messages keeps the conversation cache warm.

Like llama-server, `GET /props` (`default_generation_settings.n_ctx`) and `GET /slots`
report the context, and `GET /health` the loading progress.

## Numerics

Faster engines often trade precision for speed without saying so: weights repacked to 8
or 4 bits, activations rounded to int8 before every product, K/V caches in FP8, experts
skipped. Bandwidth and compute have physical limits, so Straydog finds most of its speed in
not repeating work (K/V mirrors, conversation checkpoints, row and expert caches, reads and
copies overlapped with compute, fused kernels, both GPUs at once). Where it rounds, it says
so here.

What Straydog does not do:

- Re-quantize weights. Each GGUF type is decoded exactly as ggml defines it (checked bit
  for bit against reference decoders); products accumulate in F32. The two exceptions are in
  the table below: the MTP drafter's experts and DeepSeek-V4.1's hyper-connection weights.
- Skip or approximate work. Every routed expert a token selects is computed, attention is
  never replaced by an approximate replay, and the output head is the model's.
- Let speculation change answers. Drafts are only proposals: a verification pass is
  bitwise identical to token-by-token decode in the same placement.
- Convert silently. An unsupported tensor type is an error.

Where it rounds, and how to turn it off:

| Where | Default | Exact |
| --- | --- | --- |
| Qwen products over prompt batches (2 rows or more) | FP16 operands, F32 sums (`--gemm fp16`) | `--gemm fp32` |
| Qwen's long prompts' attention on RDNA4 | FP16 products, F32 softmax and sums | `--gemm fp32` |
| Qwen decode and speculative verification, and its K/V cache | F32 | |
| DeepSeek products over prompt passes of at least 32 rows (FFN, attention, index, router and Engram projections; more than 4 rows per expert) | FP16 operands, F32 sums and output | `--gemm fp32` |
| DeepSeek decode and verification: the inputs of the Q8_0 and Q5_K projections and of the IQ2 experts' gate and up | INT8 per 32 values with an F32 scale; the dots exact in INT32, then scaled in F32 | none in the server (`DS41_Q8=0` in the examples) |
| DeepSeek decode and verification: the attention's and the indexer's scores | FP16 operands, F32 sums | `--gemm fp32` |
| DeepSeek's compressed K/V rows and index keys | BF16, rounded to nearest; the window rows stay F32 | none |
| DeepSeek's hyper-connection weights | BF16 on the device (F32 in the GGUF) | none |
| MTP drafter (Qwen3.8) | its experts quantized to Q2_0 when it is built: this changes only how many drafts are accepted, never the output | run without `-md` |
| DSpark drafter (DeepSeek) | each stage routes among its most used experts (98% of its routes): this changes only how many drafts are accepted | `DS41_DSPARK_RESIDENT=all` (`DS41_DSPARK=0`: no drafter) |
| Qwen3.8's experts beyond VRAM (partial residency) | the CPU computes them in F32 and prompt batches use the FP16 expert GEMM; which experts run where depends on timing, so decode is not bit-reproducible from run to run | full residency |

How this is checked: Qwen3.8's pinned GPU generations reproduce every greedy token, logits
vector and final hidden state bit for bit in each placement, and DeepSeek-V4.1's served
tests check that its replies with drafts are the replies without. A deliberate numerical
change is first compared with an F32 run (for DeepSeek-V4.1 also with llama.cpp's outputs),
by KL divergence and top-1 agreement, then pinned anew. F32 products take the same kernels in
every process (rocBLAS's own library: on RDNA4, hipBLASLt's choice of F32 kernels changed from
one process to another).
With the FP16 long-prompt attention, KL against `--gemm fp32` averaged 3.4e-3 over 12
prompts of 8K-40K (3.9e-3 before it; top-1 12/12), while reordering one sum alone moves it
by 3.0e-3. DeepSeek-V4.1 rounds the inputs and weights of a prompt pass's products to
FP16 when the pass has at least 32 rows: the shared/routed FFN (more than four rows per
expert), and the attention, index, router and Engram projections. In F32 the attention's
projections alone held each device ~6 of a 10.7K-token prompt's 36 s. Smaller prompt passes
stay F32, as do sums and outputs, norms, HC and residuals. `--gemm fp32` keeps every
product in F32. With the GSQ-RCO IQ2_XS model, the FP16 FFN path agrees
with F32 on 772/774 next-token choices in six short probes for each of TP1 and TP2;
a native 4083-token prompt plus 16 teacher-forced steps agrees on all 17 choices, with
mean KL 5.3e-4 (TP1) and 3.2e-4 (TP2). `examples/ds41_quality --gemm fp32|fp16` and
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
# DeepSeek-V4.1 served end to end, with and without DSpark (STRAYDOG_DS41_QUANT names the bundle)
STRAYDOG_DS41=DIR STRAYDOG_DS41_QUANT=GSQ-RCO-IQ2_XXS cargo test -p straydog-engine --features hip --test deepseek41_served
STRAYDOG_DS41=DIR STRAYDOG_DS41_QUANT=GSQ-RCO-IQ2_XXS cargo test -p straydog-engine --features hip --test deepseek41_dspark_served
# A/B of this build against another
cargo run --release --features hip --example ab -- --baseline BIN --model DIR --drafter DIR --output ab.json
# The dashboard over a simulated engine (no GPU, no model): http://127.0.0.1:18090/dashboard
cargo run --release --features hip --example dashboard
```

## License

MIT
