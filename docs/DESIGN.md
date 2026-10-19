# Straydog — design

Status: phases R1-R5 done on Linux (the Rust executable serves the API with its worker
contained, provisions bundles and builds the drafter byte for byte); R6 measured. Windows
builds but has not run. Updated: 2026-10-19.

Straydog is a local inference engine for Qwen3.8-Flash-Next and a server with an
OpenAI-compatible API, shipped as one executable per GPU backend.

## 1. Goals

- One long-lived server keeps the model loaded. Clients start, update and restart
  without reloading it (authenticating and loading the IQ3_S model takes about 70 s).
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
  FP16 GEMM is used only when configured. A speculative verification pass is
  bitwise identical to token-by-token decode in the same placement. Unsupported
  formats are an error, never a silent conversion.
- **Isolation.** The engine runs in a separate worker process without network
  access: bubblewrap on Linux, an AppContainer of its own on Windows. The worker is
  the same executable. Every model and drafter file is pinned by SHA-256 and
  authenticated before use. The SSD is read-only: no swap, weight or K/V files.
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
   │  C ABI (sd_*, 61 entry points)
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
  straydog-gguf/           GGUF structure checks (and the reference's inspector command)
  straydog-api/            the OpenAI-compatible API over an Engine (the runtime)
  straydog-isolation/      bubblewrap and AppContainer containment of the worker
  straydog-provision/      fetch, verify, and the MTP drafter build
gpu/                       C++/HIP/CUDA kernels; straydog_gpu.h is the C ABI
src/straydog/              the Python reference implementation, until phase R6 removes it
tests/                     kernel tests (Python, NumPy references) against a shared build of gpu/
docs/
```

The kernel tests stay in Python: NumPy is the reference math, and the tests drive the
same C ABI the engine uses. They are development tooling, not part of the product.

### 4.3 What moves where

| Python reference | Rust | Notes |
| --- | --- | --- |
| tokenizers (HF) | `tokenizers` crate | the same implementation; pure-Rust regex backend |
| jinja2 sandbox | `minijinja` (+ Python-compatible methods) | renders must equal the reference byte for byte |
| gguf parsing, mmap, SHA-256 | gguf reader (from the inspector crate), `memmap2`, `sha2` | two files hashed at once, as now |
| placement, residency planning | same algorithms | equal plans for the same budgets |
| generation loop, MTP control, checkpoints | same algorithms over the C ABI | host steps are copies, integer math and argmax |
| NumPy layer math (bounded-cache tier) | not ported | see §9, decision 5 |
| sampling with temperature | own RNG | same distribution; a seed gives different samples than the reference |
| HTTP server | `axum` on `tokio` | |
| bubblewrap command, AppContainer launcher | same, in Rust (`windows` crate) | grants: the executable, the CUDA runtime, model files |
| provisioning (download, resume, MTP conversion) | `ureq` + `rustls`, own GGUF writer | the drafter is bit-identical to the reference's |

## 5. Command line and configuration

With no subcommand, `straydog` is the server, as llama-server is: every setting is an
option, named as llama-server names it where it has the same setting.

| Option | Meaning | Default |
| --- | --- | --- |
| `-m, --model DIR` | model bundle directory (the pinned GGUF shards and tokenizer/template files of one quantization) | from config |
| `--quant NAME` | Q2_0, IQ2_XS, IQ3_XXS or IQ3_S, when the directory alone does not say | detected from the pinned files |
| `-md, --model-draft DIR` | MTP drafter directory | none |
| `--draft-max N` | MTP drafts per pass (0 turns speculation off) | 3 with a drafter |
| `-c, --ctx-size N` | context in tokens, at most 262144 | 262144 |
| `-b, --batch-size N` | prefill batch, 1-4096 | 1024 (4096 with 48 GiB or more of GPU budget) |
| `-dev, --device LIST` | GPU ordinals, e.g. `0,1` | `0` |
| `-sm, --split-mode MODE` | `layer` or `experts` (two fully resident GPUs) | `experts` |
| `--vram GIB` | VRAM budget per GPU | 6 |
| `--state-ram GIB` | host memory for sequence state | what the context needs (13 at 262144) |
| `--gemm fp32\|fp16`, `--gemm-min-batch N` | precision of large matrix products, smallest batch that uses GEMM | fp16, 2 |
| `--host`, `--port` | listen address | 127.0.0.1, 8080 |
| `--api-key KEY`, `--api-key-file FILE` | a Bearer key the server then requires | none: no key required |
| `--alias NAME` | model id in `/v1/models` and answers | `Qwen3.8-Flash-Next-<quant>` |
| `--temp T`, `--seed N` | defaults when a request gives none | 0, 0 |
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
There is no setup command: a model is fetched once, then given by path.

## 6. API

- `GET /health` (no key needed): 200 when ready, 503 while loading (with progress),
  500 after a fatal engine error.
- `GET /v1/models`.
- `POST /v1/chat/completions`, streaming (SSE `chat.completion.chunk`) or not:
  - messages: system (and developer), user, assistant (content, `reasoning_content`,
    `tool_calls`) and tool messages; content as a string or text parts;
  - `tools`, `tool_choice` (`auto` or `none`);
  - `max_tokens` / `max_completion_tokens`, `temperature`, `seed`, `stop`;
  - `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) and
    `chat_template_kwargs` (`enable_thinking`, the template's `reasoning_effort`);
  - `stream_options.include_usage`.

  Answers carry `reasoning_content` deltas, structured `tool_calls` and `usage`
  with `prompt_tokens_details.cached_tokens`, plus engine `timings`.
- Sampling controls the engine does not implement (`top_p`, penalties, `n`,
  `logprobs`, `logit_bias`, `top_k`, `min_p`, other `response_format`s, ...) are
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
  builds), the gguf reader. The kernel tests run against the same CMake build.
- **R2** (done): tokenizer and template, with golden files from the reference
  (`scripts/golden.py`): token ids and renders over a corpus of conversations (tools,
  reasoning, tool results, Japanese text, 150 seeded random ones), the template's
  errors, decoding, JSON and float formatting. Equal byte for byte.
- **R3** (done): weights, authentication, placement and the generation loop (prefill,
  decode, verification, MTP, conversation checkpoints, partial residency). Gate
  (`scripts/golden_model.py`, `tests/model_gpu.rs`): over a scripted conversation that
  takes every resume path, every greedy token, request metric, logits vector and final
  hidden state equals the reference's bitwise, with two GPUs (layer and expert split,
  with and without MTP, K/V in VRAM and in host memory). Partial residency swaps expert
  slots asynchronously during decode, so whether an expert runs on the GPU or the CPU
  depends on timing and the reference does not reproduce its own decode bit for bit;
  there, requests whose state comes from prefill alone are compared (prompt logits and
  tokens), and the placement's allocations are equal to the byte.
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
- **R6**: alternating A/B against the reference on the single-request benchmark and
  a multi-turn session; no metric may be slower. Then the Python engine and its
  packaged builds are removed and the first binaries released. Measured
  (`examples/ab.rs`, IQ3_S, two R9700s, 262144, MTP 3; the two engines produce the same
  tokens): prefill and first-token latency are lower in every case (long prompts
  1,592 vs 1,549 tokens/s; a multi-turn session's appended prompts 2-4% faster), and
  the decode medians are equal or higher (128 tokens 122.0 vs 120.3, 512 tokens
  83.3 vs 81.4 tokens/s). Startup 42-45 s for both.
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
   within Straydog, not across the reference.
