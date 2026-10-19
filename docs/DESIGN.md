# Straydog — design

Status: phase 1 in progress. Updated: 2026-10-19.

Straydog is a local inference engine for Qwen3.8-Flash-Next and a server with an
OpenAI-compatible API.

## 1. Goals

- One long-lived server keeps the model loaded. Clients start, update and restart
  without reloading it (authenticating and loading the IQ3_S model takes about 70 s).
- Any OpenAI-compatible client can use it.
- Its own release cadence, native builds (Linux/HIP, Windows/CUDA), GPU tests and
  benchmarks.

## 2. No client logic

Straydog serves the OpenAI-compatible API and nothing specific to one client. Client
integrations, client-shaped benchmarks and a client's own secrets belong to the
client. Straydog's own secret, its API key, is never granted to the worker.

The conversation cache, the reasoning replay and the tool-call split are not client
logic: they follow from the official chat template and the OpenAI message format.

## 3. Contracts

- **Exactness.** Activations are never quantized. Weights decode exactly to F32.
  FP16 GEMM is used only when configured. A speculative verification pass is
  bitwise identical to token-by-token decode in the same placement. Unsupported
  formats are an error, never a silent conversion.
- **Isolation.** The engine runs in a separate worker without network access:
  bubblewrap on Linux, an AppContainer of its own on Windows. Every model, drafter
  and native file is pinned by SHA-256 and authenticated before use. The SSD is
  read-only: no swap, weight or K/V files.
- **Measurement.** Performance claims come from alternating A/B runs on the same
  machine, under a GPU lock.

[Strata](https://github.com/Niko1221/Strata) has been used as a benchmark comparison
target.

## 4. Layout

```
src/straydog/
  engine: model, weights, gpu, devices, speculative, mtp, hybrid, cpu, host,
          formats, numeric, digest, library, bundles, assets.json, worker
  native/        packaged native builds (Linux HIP at the root, win32-amd64/)
  runtime.py     owns the isolated worker process (one engine, serialized requests)
  isolation.py   bubblewrap command; appcontainer.py the Windows launcher
  chat.py        the official template's output split into reasoning, answer, tool calls
  server.py      the OpenAI-compatible HTTP API
  config.py      settings (~/.straydog/config.yaml)
  provisioning.py, progress.py   model and drafter download and authentication
  cli.py         straydog serve | init | fetch | verify | build | bundle
  platform/      process groups, environment, Windows helpers
native/          Rust inspector crate and the C++/HIP/CUDA GPU library
scripts/         benchmarks (single requests, a multi-turn agent session)
tests/
docs/
```

## 5. Command line and configuration

`straydog serve` takes llama.cpp-style options. Where llama-server has the same
setting, the name is the same.

| Option | Meaning | Default |
| --- | --- | --- |
| `-m, --model DIR` | model bundle directory (the pinned GGUF shards and tokenizer/template files of one quantization) | from config |
| `--quant NAME` | Q2_0, IQ2_XS, IQ3_XXS or IQ3_S, when the directory alone does not say | detected from the pinned files |
| `-md, --model-draft DIR` | MTP drafter bundle | none |
| `--draft-max N` | MTP drafts per pass (0 turns speculation off) | 3 with a drafter |
| `-c, --ctx-size N` | context in tokens, at most 262144 | 262144 |
| `-b, --batch-size N` | prefill batch, 1-4096 | 1024 (4096 with 48 GiB or more of GPU budget) |
| `-dev, --device LIST` | GPU ordinals, e.g. `0,1` | `0` |
| `-sm, --split-mode MODE` | `layer` or `experts` (two fully resident GPUs) | `experts` when possible |
| `--vram GIB` | VRAM budget per GPU | 6 |
| `--state-ram GIB` | host memory for sequence state | 13 |
| `--gemm fp32\|fp16`, `--gemm-min-batch N` | precision of large matrix products, smallest batch that uses GEMM | fp16, 2 |
| `--host`, `--port` | listen address | 127.0.0.1, 8080 |
| `--api-key KEY`, `--api-key-file FILE` | a Bearer key the server then requires | none: no key required |
| `--alias NAME` | model id in `/v1/models` and answers | `Qwen3.8-Flash-Next-<quant>` |
| `--temp T`, `--seed N` | defaults when a request gives none | 0, 0 |
| `--inspector PATH`, `--gpu-library PATH` | native files other than the packaged build for this machine | packaged build |
| `--startup-timeout S`, `--log-file FILE`, `-v` | operations | 180 s, stderr |
| `--config FILE` | the same settings as a file | `~/.straydog/config.yaml` when present |

Precedence: options, then `STRAYDOG_*` environment variables (`STRAYDOG_CTX_SIZE`),
then the config file, then defaults. One schema defines all three, so they cannot
drift. `straydog serve --print-config` shows the settings in effect and where each
came from.

There is no `-ngl`. Residency is decided by measuring what fits:
- full residency first, then host K/V, then a smaller prefill batch;
- partial residency with CPU and GPU expert slots otherwise.

There is no arbitrary-GGUF mode: a model is one of the pinned bundles, authenticated
before use.

Other commands:
- `straydog init`: interactive setup. It chooses a quantization and GPUs, fetches
  and authenticates the model, builds the drafter, writes the config and the key,
  and optionally installs a systemd user service.
- `straydog fetch`, `straydog verify`.
- `straydog build`, `straydog bundle`: native builds, for development.

## 6. API

- `GET /health`, `GET /v1/models`.
- `POST /v1/chat/completions`, streaming (SSE `chat.completion.chunk`) or not:
  - messages: system, user, assistant (content, `reasoning_content`, `tool_calls`)
    and tool messages;
  - `tools`, `tool_choice` (`auto` or `none`);
  - `max_tokens` / `max_completion_tokens`, `temperature`, `seed`;
  - `reasoning_effort` (`none`, `low`, `medium`, `high`);
  - `stream_options.include_usage`.

  Answers carry `reasoning_content` deltas, structured `tool_calls` and `usage`
  with `prompt_tokens_details.cached_tokens`, plus engine `timings`.
- Extension (optional `straydog` object in the request): `checkpoint` set to false
  marks a one-off request (a summary, a title) that saves no conversation checkpoint.
  A request that ends with a tool message continues the assistant's reply; one that
  ends with a user message starts a new one. That is all the cache needs from a
  client.
- Reasoning sent back in assistant messages is part of the prompt the official
  template renders, so a client that returns it keeps the conversation cache warm.
- Security: listens on 127.0.0.1 by default. A Bearer API key is optional and
  required once given (`--api-key`, or `--api-key-file` for a key kept out of the
  process list). One engine serializes generation. The waiting queue is bounded, and
  a request whose client disconnects is cancelled.

## 7. Phases

1. The engine, native sources, packaged builds, scripts and tests, with the tests
   passing in this repository.
2. The server (`straydog serve`) on the runtime and worker.
3. `straydog init / fetch / verify`: provisioning.
4. Clients move onto the server.

## 8. Decisions (2026-10-19)

1. The repository starts a fresh history.
2. HTTP stack: Starlette and uvicorn.
3. Port 8080 by default (`--port` changes it). An API key is optional and required
   only once one is given.
4. Home directory `~/.straydog` (config, models, drafter, native builds). Existing
   model directories are used in place.
