# Straydog — design

Status: phases R1-R6 done on Linux but for the first release (the Rust executable serves
the API with its worker contained, provisions bundles and builds the drafter byte for byte;
the Python reference is removed). Windows builds but has not run. Updated: 2026-10-20.

Straydog is a local inference engine for Qwen3.8-Flash-Next and a server with an
OpenAI-compatible API, shipped as one executable per GPU backend.

## 1. Goals

- One long-lived server keeps the model loaded. Clients start, update and restart
  without reloading it (authenticating and loading the IQ3_S model takes 13-23 s, §4.4).
- Any OpenAI-compatible client can use it.
- One executable: the server, the engine, the isolated worker and provisioning. The
  only external pieces are the GPU vendor's runtime and driver.
- Every per-token and per-pass host step is native code. Performance work starts from
  a language without an interpreter in the loop.

## 2. No client logic

Straydog serves the OpenAI-compatible API and nothing specific to one client. Client
integrations, client-shaped benchmarks and a client's own secrets belong to the
client. Straydog's own secret, its API key, is never exposed to the worker.

The conversation cache, the reasoning replay and the tool-call split are not client
logic: they follow from the official chat template and the OpenAI message format.

## 3. Contracts

- **Exactness.** Activations are never quantized. Weights decode exactly to F32.
  FP16 products (GEMM over prompt batches, and RDNA4's prefill attention) are the default
  (`--gemm fp16`); `--gemm f32` computes them in F32.
  A speculative verification pass is bitwise identical to token-by-token decode in the
  same placement. Unsupported formats are an error, never a silent conversion.
- **Isolation.** The engine runs in a separate worker process without network
  access: bubblewrap on Linux, an AppContainer of its own on Windows. The worker is
  the same executable. Every model and drafter file is pinned by SHA-256 and
  authenticated before use; the shards of a model read on demand also by their
  fs-verity digests, every 4 KiB block checked against its Merkle tree before use. The
  other bundles' shards are pinned by fs-verity digest too, and what is read again after
  loading (the PLE table's rows, the expert banks partial residency page-locks in host
  memory) is checked against their trees. Partial residency's banks left to the page cache
  (when host memory cannot lock them all) are read through the mapping unchecked after
  loading verified their chunks. The trees live in the data directory (`.straydog` in the
  model's by default, `--data-dir` elsewhere), written by the server, never by the worker,
  and checked against the compiled-in digests whenever they are opened.
  The SSD is read-only: no swap, weight or K/V files.
- **Measurement.** Performance claims come from alternating A/B runs on the same
  machine, under a GPU lock.

[Strata](https://github.com/Niko1221/Strata) has been used as a benchmark comparison
target.

## 4. Architecture

### 4.1 Processes

```
straydog ──HTTP──▶ clients
   │  runtime: one worker, a bounded queue (4), cancellation
   │  stdio, JSON lines (one frame per event; ~8 µs per token)
   ▼
straydog worker   (same executable, isolated; never opens a socket)
   tokenizer, template, weights, placement, generation loop
   │  C ABI (sd_*, 83 entry points)
   ▼
GPU library (C++ / HIP / CUDA, linked statically)
```

The server never loads the model; a worker failure never takes the server down. The
worker reads only what isolation exposes: the executable, the GPU runtime, the pinned
model and drafter files and the GPU device nodes.

### 4.2 Source layout

```
Cargo.toml                 workspace
crates/
  straydog/                the executable: cli, config, server, runtime, isolation, provisioning
  straydog-engine/         tokenizer, template, gguf, weights, placement, model, speculative, sampling, worker
  straydog-gpu/            bindings to the C ABI; build.rs compiles gpu/ with CMake (HIP or CUDA)
  straydog-gguf/           GGUF structure checks
  straydog-api/            the OpenAI-compatible API over an Engine (the runtime)
  straydog-isolation/      bubblewrap and AppContainer containment of the worker
  straydog-provision/      fetch, verify, and the MTP drafter build
gpu/                       C++/HIP/CUDA kernels; straydog_gpu.h is the C ABI
bench/                     the A/B harness's single-request cases (examples/ab.rs)
docs/
```

The Rust engine is the reference. Its regression oracle is the pinned GPU generations
(`crates/straydog-engine/tests/model_gpu.rs`): for each placement, every request's greedy
tokens, metrics, logits and final hidden state, bit for bit. A deliberate numerical change
is validated against an F32-GEMM run, then re-pinned (`STRAYDOG_GOLDEN_UPDATE=1`). The
golden files of the tokenizer, template and weights (`tests/golden/chat.json` and
`weights.json` in the same crate) came from the Python reference and stay as pinned data.

### 4.3 What moved where

| Python reference | Rust | Notes |
| --- | --- | --- |
| tokenizers (HF) | `tokenizers` crate | the same implementation; pure-Rust regex backend |
| jinja2 sandbox | `minijinja` (+ Python-compatible methods) | renders equal the reference's byte for byte |
| gguf parsing, mmap, SHA-256 | gguf reader (from the inspector crate), `memmap2`, `sha2` | each chunk read once, hashed and placed (§4.4) |
| placement, residency planning | same algorithms | equal plans for the same budgets |
| generation loop, MTP control, checkpoints | same algorithms over the C ABI | host steps are copies, integer math and argmax |
| NumPy layer math (bounded-cache tier) | not ported | see §9, decision 5 |
| sampling with temperature | own RNG | same distribution; a seed gives different samples than the reference |
| HTTP server | `axum` on `tokio` | |
| bubblewrap command, AppContainer launcher | same, in Rust (`windows` crate) | grants: the executable, the CUDA runtime, model files |
| provisioning (download, resume, MTP conversion) | `ureq` + `rustls`, own GGUF writer | the drafter is bit-identical to the reference's |

### 4.4 Loading

Each pinned file is authenticated in 256 MiB chunks, against digests compiled in beside its
SHA-256. Opening reads and verifies each file's first chunk and parses the GGUF header from
it; placement runs on the header alone, allocating every weight it pins without filling it.
Then every core reads chunks of its own into page-locked memory and hashes them there; a
chunk whose digest matched is copied from that memory to the GPU allocations and to the host
copies the engine keeps (token embeddings, vectors, the operands passes upload at first use).
The bytes hashed are the bytes used. The PLE table and partial residency's expert banks are
read through the mapping at run time, after their chunks were verified; every generation
checks the files unchanged. Chunks no GPU waits for are read first, while the host K/V is
page-locked: its 5 s of `hipHostMalloc` hold up the copies.

A cached chunk is read from the page cache. One that is not goes through it when the files fit
in three quarters of the host memory, so that a restart finds them, and around it otherwise
(direct reads). With the IQ3_S bundle and the drafter (84.5 GB) on two R9700s at 262144 tokens,
the server is ready 22-23 s after a cold start, reading the SSD at 4.3-4.6 GB/s throughout (29
s before), and 13-14 s with the files cached, where copying them from the page cache and
SHA-256 on every core (7.5 s) and the K/V page-locking bound the reading (19 s before).
`ready` reports the steps' seconds.

### 4.5 Long-context prefill

Every fourth layer attends sparsely: an indexer scores the prompt's 4-token blocks and each
token attends the keys of its 512 best complete blocks and those of its own incomplete block. A prompt is prefilled
in batches (2048 rows on two R9700s at 262144 tokens), split into 256-row chunks for the
indexer. Its block scores are a tiled F32 product, and a chunk's rows past 6144 blocks are
selected in registers, each row read once; both give the one-token kernels' scores and
selection bit for bit. With FP16 GEMM on RDNA4, a batch's attention runs at once: groups of 16
tokens walk the union of their selected blocks (about 2700 of 32768 at 131072 tokens, against
16 x 513), so each key and value is read once per group, with FP16 WMMA products accumulated in
F32 and an F32 softmax; the union is ordered by the tokens that selected each block, so the
waves of a token skip most blocks it did not select. With host K/V, a layer's prefix is staged
in VRAM for its batch; the copy for a device's next attention layer runs on a stream of its own
while the layers before it compute. Prefill (IQ3_S, two R9700s, 262144 context): 131072 tokens
51.2 -> 34.3-34.8 s, 65536 21.1 -> 15.9-16.1 s, 20480 6.1 -> 5.0 s, 4096 1.48 -> 1.23 s.

### 4.6 Row tables

Embedding tables read a few rows per token, at random, are not kept in memory: the PLE table
(28.8 GB of 90-byte rows, 16 per token) and DeepSeek-V4.1's Engram table (135 GB of 176-byte
rows, 48 per token). Their rows are read from the authenticated file into a row cache of the
engine's own (`crate::rows`, `--row-cache`, 256 MiB). Rows rather than pages: a 4 KiB page
holds dozens of rows, one of them wanted, while 70-98% of row uses in agent sessions and chats
repeat an earlier row, so 64-256 MB of rows hit more often than 2 GB of page cache would. A
pass's missing rows are read together, deduplicated, as whole aligned 4 KiB blocks past the
page cache (O_DIRECT; buffered where unavailable) on 32 threads, and blocks the page cache
holds anyway (the table read at load) through it, as soon as their tokens are known: the next
prompt batch's while the current one runs, a sampled token's while the drafter runs, the
drafts' while the first layer does. Decoding waits only for rows still on their way.
A table can take a verifier of each block read, between read and use (`crate::verity`'s, as
the expert store does; a block that fails fails its read, and a later use reads it again).
DeepSeek-V4.1's Engram table, whose file is not read whole at load, takes its shard's fs-verity
tree as that verifier, and so does the PLE table (through the mapping, with `--row-cache 0`,
each row is copied out of its blocks and checked the same way).
With the IQ3_S bundle on two R9700s, a fresh 4096-token prompt prefills in 1.3 s with the table
in the page cache, as through the mapping, and in 1.5 s with it evicted (through the mapping
3.0-3.3 s); decode is unchanged (75 and 125 tokens/s after 1000 and 4096), peak memory
is 18.3 GB either way (through the mapping 18.7-19.6 GB, the table's pages counted). Engram
gathers for DeepSeek-V4.1 take 0.7-0.9 ms per decoded token (0.8-1.0 ms through the mapping).

## 5. Command line and configuration

With no subcommand, `straydog` is the server, as llama-server is: every setting is an
option, named as llama-server names it where it has the same setting.

| Option | Meaning | Default |
| --- | --- | --- |
| `-m, --model DIR` | model bundle directory (the pinned GGUF shards and tokenizer/template files of one quantization) | from config |
| `--quant NAME` | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S or UD-IQ2_XXS (DeepSeek-V4.1-Flash), when the directory alone does not say | detected from the pinned files |
| `-md, --model-draft DIR` | MTP drafter directory | none |
| `--draft-max N` | most MTP drafts per pass (0 turns speculation off) | 3 with a drafter |
| `-c, --ctx-size N` | context in tokens, at most 262144 | 262144 |
| `-b, --batch-size N` | prefill batch, 1-4096 | 1024 (4096 with 48 GiB or more of GPU budget) |
| `-gpu, --gpu LIST` | GPU ordinals, e.g. `0,1` (`-dev`, `--device` also accepted) | `0` |
| `-sm, --split-mode MODE` | `layer` or `experts` (two fully resident GPUs; for a model read on demand, `experts` splits every layer across the GPUs) | `experts` |
| `--vram GIB` | VRAM budget per GPU | 6 |
| `--state-ram GIB` | host memory for sequence state | what the context needs (13 at 262144) |
| `--row-cache MIB` | host memory caching rows of the PLE table (or a model's Engram table), read from the SSD as needed (§4.6); 0 reads them through the page cache | 256 |
| `--data-dir DIR` | Straydog's data directory: the shards' verity trees, built there at the first start (reads after loading are checked against them, §3), and the expert store's warm-up state | `.straydog` in the model directory |
| `--expert-ram GIB` | host memory for routed experts outside VRAM (a model read on demand) | three quarters of the host memory, less Straydog's other host memory (the row cache, 2 GiB for the rest) |
| `--gemm fp32\|fp16`, `--gemm-min-batch N` | precision of large matrix products, smallest batch that uses GEMM | fp16, 2 |
| `--host`, `--port` | listen address | 127.0.0.1, 8080 |
| `--api-key KEY`, `--api-key-file FILE` | a Bearer key the server then requires | none: no key required |
| `--alias NAME` | model id in `/v1/models` and answers | `Qwen3.8-Flash-Next-<quant>` |
| `--temp T`, `--top-k K`, `--top-p P`, `--min-p P`, `--presence-penalty X` | sampling when a request gives none | the model card's for the request's mode |
| `--seed N` | seed when a request gives none | a fresh one per request |
| `--reasoning-effort E` | reasoning when a request gives none: none, low, medium, high | high (the template's default) |
| `--startup-timeout S`, `--log-file FILE`, `-v` | operations | 180 s, stderr |
| `--config FILE` | the same settings as a TOML file | none |

Precedence: options, then `STRAYDOG_*` environment variables (`STRAYDOG_CTX_SIZE`),
then the config file, then defaults. One `clap` definition gives the options and the
variables; the file uses the same names. `--print-config` shows the settings in effect
and where each came from. Nothing is read from a home directory unless given.

There is no `-ngl`. Residency is decided by measuring what fits:
- full residency first, then host K/V, then a smaller prefill batch;
- partial residency with CPU and GPU expert slots otherwise.

There is no arbitrary-GGUF mode: a model is one of the pinned bundles, authenticated
before use. The pins are compiled into the executable.

Subcommands: `straydog fetch`, `straydog verify`, `straydog fetch-mtp` (provisioning).
There is no setup command: a model is fetched once, then given by path. `fetch` and
`verify` also build each shard's verity tree into the data directory
(`<data-dir>/verity/<shard>.verity`, by default `<model>/.straydog/verity/`, one read of the
files) and keep it only when its digest is the pinned one; the server builds the trees it
lacks the same way at its start, before the worker starts (shown as loading). Only the data
directory is written, and only by these, never by the worker; when it cannot be written the
start fails, naming `--data-dir`. For a model read on demand the trees are its
authentication: no worker starts without them.

## 6. API

- `GET /health` (no key needed): 200 when ready, 503 while loading (with progress),
  500 after a fatal engine error.
- `GET /v1/models`, and llama-server's `GET /props` and `GET /slots` (the context
  as `n_ctx`).
- `POST /v1/chat/completions`, streaming (SSE `chat.completion.chunk`) or not:
  - messages: system (and developer), user, assistant (content, `reasoning_content`,
    `tool_calls`) and tool messages; content as a string or text parts;
  - `tools`, `tool_choice` (`auto` or `none`);
  - `max_tokens` / `max_completion_tokens`, `temperature`, `top_k`, `top_p`, `min_p`,
    `presence_penalty`, `seed`, `stop`;
  - `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) and
    `chat_template_kwargs` (`enable_thinking`, the template's `reasoning_effort`);
  - `stream_options.include_usage`.

  Answers carry `reasoning_content` deltas, structured `tool_calls` and `usage`
  with `prompt_tokens_details.cached_tokens`, plus engine `timings`.
- Sampling: greedy at temperature 0, else `softmax(logits / temperature)` cut to
  `top_k`, then `top_p`, then `min_p` (vLLM's order), after `presence_penalty` is
  subtracted from every token the answer already holds. A control a request leaves
  out takes the server's setting, else the model card's for the request's mode:
  thinking `temperature=1.0, top_p=0.95, top_k=20`, non-thinking `temperature=0.7,
  top_p=0.80, top_k=20, presence_penalty=1.5` (`min_p=0` in both). Greedy decoding
  repeats itself endlessly with this model; it is never the default. MTP verification
  stays exact: a draft is accepted with its probability under the same cut and penalty.
  A pass drafts while the product of the drafts' probabilities under the drafter stays
  at 0.6 or more, up to `--draft-max`.
- Sampling controls the engine does not implement (`n`, `typical_p`, frequency and
  repetition penalties, `logprobs`, `logit_bias`, other `response_format`s, ...) are
  accepted only at their neutral value; anything else is a 400, never ignored.
- Extension (optional `straydog` object in the request): `checkpoint` set to false
  marks a one-off request (a summary, a title) that saves no conversation checkpoint.
  A request that ends with a tool message continues the assistant's reply; one that
  ends with a user message starts a new one. That is all the cache needs from a
  client.
- Reasoning sent back in assistant messages is part of the prompt the official
  template renders, so a client that returns it keeps the conversation cache warm.
- Errors are OpenAI-style objects. Before the first byte of a stream the status says
  it (400 invalid request or context overflow, 401, 503 loading or busy, 500 engine
  failure); after it, an `error` event ends the stream.
- Security: listens on 127.0.0.1 by default. A Bearer API key is optional and
  required once given (`--api-key`, or `--api-key-file` for a key kept out of the
  process list). One engine serializes generation. The waiting queue is bounded, and
  a request whose client disconnects is cancelled.
- Dashboard: `GET /dashboard` is a page compiled into the executable (its script,
  style and icon beside it, a strict CSP, no other origin). It reads
  `GET /dashboard/status?since=T` (settings without the key, engine state and
  placement, the running generation, the latest 120 finished ones with their speeds,
  host, process and GPU sensor figures, and the one-second samples after T for its
  charts) and `GET /dashboard/logs?after=N` (the latest 2000 log lines), and chats
  through `/v1/chat/completions`. The page needs no key; the data and the chat need it
  once one is set. Sensors are read only while a dashboard is open: sampling starts
  with a status request and stops, its samples dropped, 15 s after the last one. GPU
  sensors come from amdgpu's sysfs (Linux) or NVML (CUDA builds, loaded at run time),
  matched to the engine's devices by PCI address.
- One principal: every client that holds the key shares the engine and its
  conversation cache, so whether a prompt prefix was seen before is observable
  (`cached_tokens`, first-token latency). Users who must not see each other's prompts
  need separate servers.

## 7. Build and distribution

- `cargo build --release --features hip` (or `cuda`). `build.rs` runs CMake on `gpu/`
  with the vendor compiler and links the result statically; the architectures come
  from `STRAYDOG_GPU_ARCHITECTURES` (`gfx1201`; `86-real;89-real;120`).
- Release artifacts: `straydog-linux-x86_64-hip` (ROCm 10 runtime required) and
  `straydog-windows-x86_64-cuda.exe` (NVIDIA driver; `cublas64_12.dll` next to it or
  from a CUDA Toolkit). The C/C++ runtimes are linked statically.
- A GPU without code in the executable fails at startup with its architecture named.
- The executable's own SHA-256 is published with each release. The model pins are
  compiled in, so a binary authenticates exactly the bundles it was tested with.

## 8. Phases

- **R0** (done): the Python reference: engine, native sources, packaged builds,
  provisioning and tests, passing in this repository.
- **R1** (done): workspace, `straydog-gpu` (all `sd_*` bindings, static HIP and CUDA
  builds), the gguf reader. The kernel tests ran against the same CMake build.
- **R2** (done): tokenizer and template, with golden files from the reference
  (`tests/golden/chat.json`): token ids and renders over a corpus of conversations (tools,
  reasoning, tool results, Japanese text, 150 seeded random ones), the template's
  errors, decoding, JSON and float formatting. Equal byte for byte.
- **R3** (done): weights, authentication, placement and the generation loop (prefill,
  decode, verification, MTP, conversation checkpoints, partial residency). Gate
  (`tests/model_gpu.rs`): over a scripted conversation that takes every resume path, every
  greedy token, request metric, logits vector and final hidden state equaled the
  reference's bitwise, with two GPUs (layer and expert split, with and without MTP, K/V
  in VRAM and in host memory). Partial residency swaps expert slots asynchronously
  during decode, so whether an expert runs on the GPU or the CPU depends on timing and
  the reference does not reproduce its own decode bit for bit; there, requests whose
  state comes from prefill alone are compared (prompt logits and tokens), and the
  placement's allocations are equal to the byte.
- R4 and R5 are built in parallel in their own crates: `straydog-api` (the HTTP API over
  an `Engine` trait), `straydog-isolation` and `straydog-provision`; the executable
  (`crates/straydog`) wires them to the worker and runtime.
- **R4**: worker, runtime, server, CLI and config. Gate: API tests, including
  streaming, tool calls, stop strings, cancellation on disconnect and errors.
  `straydog-api` is done (its tests). The worker and runtime run end to end on the GPUs
  (`crates/straydog/tests/runtime_gpu.rs`); the executable's large host buffers use
  transparent huge pages, as NumPy's do (a 160 MB prompt buffer otherwise costs some
  40,000 page faults).
- **R5** (done on Linux): isolation (bubblewrap, AppContainer) and provisioning (`fetch`,
  `verify`, `fetch-mtp`). Gate met: the drafter built by Rust equals the pin byte
  for byte. The worker runs under bubblewrap with a cleared environment; the AppContainer
  launcher builds and passes its tests on Linux but has not run on Windows.
- **R6** (done on Linux, but for the first release): alternating A/B against the
  reference on the single-request benchmark and a multi-turn session; no metric may be
  slower. Then the Python engine and its packaged builds are removed and the first
  binaries released. Measured (`examples/ab.rs`, IQ3_S, two R9700s, 262144, MTP 3; the
  two engines produce the same tokens): prefill and first-token latency are lower in
  every case (long prompts 1,592 vs 1,549 tokens/s; a multi-turn session's appended
  prompts 2-4% faster), and the decode medians are equal or higher (128 tokens 122.0 vs
  120.3, 512 tokens 83.3 vs 81.4 tokens/s). Startup 42-45 s for both. The Python
  engine, its kernel tests, golden generators and packaged builds are removed; bitwise
  parity with it is given up, and the GPU generations are pinned from Straydog itself
  (§4.2). `examples/ab.rs` now compares two builds.
- **R7**: clients move onto the server.

## 9. Decisions (2026-10-19)

1. The repository starts a fresh history.
2. The host is Rust; the kernels stay C++/HIP/CUDA behind their C ABI. HTTP:
   `axum` on `tokio` (this replaces Starlette and uvicorn).
3. Port 8080 by default (`--port` changes it). An API key is optional and required
   only once one is given.
4. No home directory: models, the drafter and an optional TOML config file are given
   by path; existing model directories are used in place. (`init`, `serve` and an
   implicit `~/.straydog/config.toml` were dropped: the options are the interface.)
5. The bounded-cache tier, where the dense weights do not fit the GPU budget and
   layers run in NumPy on the host, is not ported: a budget below the dense weights
   (about 3.4 GB) is an error. Partial residency covers every supported GPU.
6. Sampling with a temperature uses Straydog's own RNG; a seed is reproducible
   within Straydog, not across the reference. The defaults are the model card's
   sampling settings for the request's mode, not greedy decoding, which loops.
7. (2026-10-21) What Straydog makes for a model, its shards' verity trees and the expert
   store's warm-up state, is kept with the model (`<model>/.straydog`) unless
   `--data-dir` says elsewhere: no flag is needed. The server writes it (at provisioning
   or before it starts the worker), never the worker; the trees are as trustworthy
   anywhere, being checked against the compiled-in digests.

## 10. Model architectures

Straydog runs one model family today (Qwen3.8 Flash Next) and is to run others: next
DeepSeek-V4.1 Flash (748B total, 16B active, as a 300 GB GGUF), then GLM and more. One
engine serves them all; the GGUF's `general.architecture` selects the architecture.
Nothing an architecture needs is a constant in the engine: dimensions, layer kinds,
router, attention and table parameters come from the file's metadata.

### 10.1 Layers of the engine

1. **Runtime** (no architecture in it): devices, streams, peer links between GPUs,
   page-locked arenas and staging, captured graphs, the storage reader, digests.
2. **Formats.** Every GGUF block type is a decoder plus GEMV and GEMM kernels on the
   GPU and the CPU, each tested against the format's definition. Architectures never
   branch on formats; a type without kernels is an error at load time.
3. **Ops**: building blocks taking their parameters from metadata: norms, rotary
   embeddings (with YaRN), linear layers, SwiGLU (with clamping), MoE routers
   (softmax, sigmoid, square-root softplus; bias, normalization, scale), expert
   execution over a tiered expert store, attention kinds (grouped-query, latent,
   sliding-window, sparse with an indexer, compressed), recurrent kinds (Gated
   DeltaNet with its convolution), hyper-connections (n streams, Sinkhorn), embedding
   tables (token, per-layer n-gram tables gathered by row) and output heads.
4. **Architectures.** An architecture reads its hyperparameters and tensor names,
   gives every tensor a role (dense per layer, routed experts, a gathered table, the
   head, a drafter) and describes each layer as a program of ops. The GPU library runs
   a layer's program as one captured graph; a new op kind is written once and serves
   every architecture that needs it.
5. **Planner.** Roles are placed into tiers from their sizes and access patterns:
   VRAM per device, page-locked RAM, the SSD. Every-token weights go to VRAM; routed
   experts to a VRAM cache backed by RAM and refreshed by use; gathered tables are
   read by row where they lie. Layer split, expert parallelism and tensor parallelism
   are the planner's choices, not the architecture's.
6. **Serving**: the tokenizer from `tokenizer.json` or the GGUF's tokenizer metadata,
   the chat template from the bundle or the GGUF, the reasoning and tool-call parsers
   of the family, and its drafter (an MTP layer, a block drafter) when it has one.

Integrity stays as in §3: every file is pinned by SHA-256 and authenticated before
use. Bytes read from storage after start (table rows, experts outside RAM) are
checked against per-block digests of the pinned file before they are used.

The Qwen path keeps its specialized code until layer programs match its speed on the
A/B harness; then it is expressed in the same terms.

### 10.2 DeepSeek-V4.1 Flash (UD-IQ2_XXS)

Measured from the file and a reference implementation (12 prompts, 1,536 generated
tokens): every token reads 7.5 GB of dense weights (attention, shared expert, router,
hyper-connections, head) and 2.45 GB of routed experts (6 of 384 in each of 40 layers,
157 GB in all), plus 48 rows of 135 GB of n-gram tables. A per-layer cache holding a
third of the experts, refreshed by use, catches 83-90% of the routes (a fixed
placement of the same size: 69%). With dense weights in VRAM and misses served from
page-locked RAM, a token costs 20-25 ms on two R9700s; the goal is 30 tokens/s.

- **D1**: architecture `deepseek41`: metadata, roles, split GGUF files, the GGUF
  tokenizer and template, bundle pins with chunk and block digests.
- **D2**: formats IQ1_S and Q2_K, and every expert type of the file on the GPU (gate,
  up and down) and the CPU.
- **D3**: ops and the layer program, validated layer by layer against the reference's
  tensors and its logits (KL divergence, top-1).
- **D4**: the tiered expert store (VRAM cache, page-locked RAM, CPU execution of
  misses), table rows from the SSD, prefetch.
- **D5**: prefill streaming the experts each batch picks, layer by layer. The model is a
  causal encoder (layers 0-19) and a decoder (20-39) whose global K/V is projected from
  the encoder's output: a prompt runs the encoder over every token and the decoder only
  over the last 20 x 128 tokens (its sliding windows' reach), which is exact.
  Replaying only the last 128 (the model's own bounded replay) is an option, not the
  default.
- **D6**: the block drafter, two-GPU strategies, tuning to the goal.

It is served like the Qwen model: `straydog -m DIR ...` with the same API,
isolation and dashboard. The bundle's pins name the architecture; the worker opens the six
shards against their trees and places them by `--split-mode`: `experts` (the default) splits
every layer across the GPUs, each holding its part of the attention and the head and its rows of
every expert; `layer` gives each GPU a range of layers, cut where no layer reads another part's
compressed rows (balancing the dense bytes). Each GPU's VRAM left after its resident weights and
sequence state goes to expert slots, and the host tier gets what Straydog's other host memory
leaves of three quarters of the host memory (as the server measures it, cgroup limits included).
Tokenizer and chat template come from the GGUF (`tokenizer.ggml.*` gives the official
tokenizer's ids; the template renders the reference encoder's prompts), replies split at the
first `</think>` and at DSML tool calls, and unset sampling follows `general.sampling.*`. A
prompt that extends the held sequence continues it; there are no checkpoints yet.

A start does not begin cold: the runtime asks the worker for the experts its stores hold (least
recently used first, in VRAM and in the host tier) every two minutes after requests and at
shutdown, and keeps them in the data directory (`<data-dir>/warm/<quant>-<key>.json`, the key
hashing the pinned shards and the placement, written by the runtime: the worker has no write
access). After the next start with the same model and placement is ready, the worker reads those
experts back while no request runs, a few reads at a time behind any pass's.

### 10.3 Qwen3.5 MoE (Qwen3.6-35B-A3B UD-Q8_K_XL, Ornith-1.5-122B Q8_0)

The `qwen35moe` architecture: one residual stream through blocks whose mixer is either a Gated
DeltaNet (three of every four) or gated full attention, each followed by a softmax top-8 MoE
with a sigmoid-gated shared expert. Its definitions follow the reference implementation
(llama.cpp `qwen35moe` at ead9f32, over the same GGUF), checked layer by layer on a fixed prompt
(examples/q35_reference on the CPU in F64-accumulated F32, examples/q35_chat on the GPU engine).

- **DeltaNet**: `attn_qkv` gives [q (key heads) | k (key heads) | v (value heads)], convolved per
  channel over the last `ssm.conv_kernel` inputs and passed through SiLU; q and k are
  L2-normalized per head (q also scaled by 1/sqrt(dim)); value head h reads key head
  h mod (key heads); the state decays by exp(`ssm_a` * softplus(alpha + `ssm_dt`)), takes the
  delta rule with sigmoid(beta), and is read by q; the output is RMS-normed per head and gated by
  SiLU of `attn_gate` (the reference's gate: SiLU, where Qwen3.8's is a sigmoid), then `ssm_out`.
- **Attention**: `attn_q` gives each head's query then its gate; q and k are RMS-normed per head,
  the leading `rope.dimension_count` values rotate in pairs (i, i + half) at the text position
  (every IMRoPE section reads the same position for text), causal grouped-query attention over
  the K/V, times sigmoid of the gate, then `attn_output`.
- **MoE**: softmax over the experts, the top `expert_used_count` weighted by their normalized
  probabilities, SwiGLU experts; the shared expert's output scaled by sigmoid(`ffn_gate_inp_shexp`
  · x). The MTP block after the main layers is not run.

Served as a bundle read on demand: the GGUF pinned by size, SHA-256 and fs-verity digest, the
official tokenizer, chat template and generation config pinned as resources (read whole and
hashed; Unsloth's GGUF embeds a modified template, which is not used). Its layers are split across
the GPUs balancing the dense bytes, each GPU's VRAM left after its dense weights, DeltaNet states
and K/V going to expert slots sized per layer's experts (Qwen3.6 mixes Q8_0 and BF16 experts), and
the host tier as for DeepSeek-V4.1. Replies split as Qwen's do (`</think>`, `<tool_call>`) and end
at the generation config's end tokens; unset sampling follows `general.sampling.*`.

A next turn continues the held sequence when its prompt extends it. Qwen3.6's template drops a
past turn's thinking (unless `preserve_thinking`), so its next prompt never extends the sequence
that ran; the DeltaNets' state (conv histories and recurrent states, a few MB a layer) is therefore copied on
the GPU where each prompt's generation prompt begins (its last `<|im_start|>`), keeping the last
two. A prompt that extends such a checkpoint, while the held sequence matches it that far (so the
K/V rows below it are the checkpoint's), restores it and runs only the rest (`resumed_from`
"turn" or "prior").

Where its experts go depends on what the GPUs hold:
- **All of a GPU's experts fit in its VRAM** (Qwen3.6 on two GPUs): they are read in at
  placement into slots sized per size class and kept for good. A pass looks each route's slot
  up on the GPU in a table (`sd_op_experts_table`), so no routes come back to the host. A decode
  step then has no host data past its position and embedding, and each GPU runs it as one
  recorded graph.
- **Otherwise**, the expert store's tiers serve them: VRAM slots, a page-locked host tier, the
  drive. A decode step reads its routes, their weights and its MoE input back each layer.
  Routes to experts in the host tier run on the CPU (`sd_cpu_experts_run`, Q8_0 and BF16) from
  host memory, so only activations cross the bus; their sum is added to the GPU's. Experts in
  VRAM run there. Experts only on the drive come through the host tier into VRAM, as before.
  Prefill still uploads the experts it needs, in waves.
