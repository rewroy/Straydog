<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-wordmark-dark.png">
    <img src="assets/logo-wordmark.png" alt="Straydog" width="240">
  </picture>
</p>

# Straydog

A local inference engine with an OpenAI-compatible server, in one executable per GPU
backend. It serves pinned GGUF bundles of Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash.

It runs on Linux with AMD GPUs (ROCm) and on Windows with NVIDIA GPUs (CUDA). Linux with NVIDIA GPUs
(CUDA) has been run under WSL2 and not yet on native Linux. AMD GPUs on Windows are not supported.

- **Every rounding named.** Weights run in the format their GGUF ships, decoded exactly,
  and every expert a token routes to is computed. The roundings that buy speed (FP16
  operands in prompt batches; Qwen3.8's K/V in 8-bit blocks (Q8_0); for DeepSeek-V4.1 also INT8 and FP16
  inputs in decode and BF16 compressed K/V) are listed under [Numerics](#numerics), each with
  its switch where it has one. Speculative decoding never changes an answer.
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

Linux with NVIDIA GPUs (the CUDA 13 Toolkit and its host compiler, CMake, make or Ninja, the NVIDIA
driver and bubblewrap; RTX 20/30/40/50 by default). Name the toolkit in `CUDA_PATH` and put its `bin`
on `PATH`:

```bash
CUDA_PATH=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH cargo build --release --features cuda
# other GPUs (the default is 75-real;86-real;89-real;120): STRAYDOG_GPU_ARCHITECTURES=89-real
```

It has been run under WSL2 (the driver is the Windows one and the toolkit the Linux one inside WSL,
[gpu/README.md](gpu/README.md)), and not yet on native Linux.

Windows, NVIDIA GPUs only (CUDA 13, from an "x64 Native Tools Command Prompt" with Ninja and the
CUDA Toolkit; RTX 20/30/40/50 by default):

```bat
cargo build --release --features cuda
rem other GPUs (the default is 75-real;86-real;89-real;120): set STRAYDOG_GPU_ARCHITECTURES=89-real
```

The result is `target/release/straydog`. It needs the GPU vendor's runtime: ROCm's
`libamdhip64` and `libhipblas` on Linux with AMD GPUs, the NVIDIA driver and the CUDA 13 runtime
(`libcublas.so.13` and `libcublasLt.so.13`, in `/usr/local/cuda/lib64` or `/usr/local/cuda-13/lib64`
or wherever `ldconfig` lists them, as NVIDIA's packages install them) on Linux with NVIDIA GPUs, the
NVIDIA driver and `cublas64_13.dll` (next to the executable or from the CUDA Toolkit) on Windows.
Linux also needs bubblewrap. DeepSeek-V4.1 has been run on Linux with ROCm, and with CUDA under WSL2.

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
also an option: `straydog setup --yes --model-name DeepSeek-V4.1-Flash-1017 --provider ISTA-DASLab --quant IQ2_XXS --dir ~/models` asks nothing
(`--offline` downloads nothing, `--config FILE` writes the settings elsewhere).

By hand, and for scripts that name their model:

```bash
# DeepSeek-V4.1-Flash-1017: download and authenticate a bundle (once), then serve it
straydog fetch -m ~/models/ds41 --model-name DeepSeek-V4.1-Flash-1017 --provider ISTA-DASLab --quant IQ2_XXS
straydog -m ~/models/ds41 -gpu 0,1 --api-key-file ~/.straydog-key

# Qwen3.8-Flash-Next with its MTP drafter
straydog fetch -m ~/models/IQ3_S --model-name Qwen3.8-Flash-Next --provider ISTA-DASLab --quant IQ3_S
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

The budget decides the placement, and with it the speed: Qwen3.8 takes the first of full residency
with its K/V in VRAM (at a prefill batch of 4096, then 2048; a batch given with `--batch-size` is not narrowed to keep the K/V in VRAM), then with the K/V
in host memory at prefill batches of 4096 down to 1024, with its MTP drafter and then without, and below that partial residency. On two R9700s at 262144 tokens,
UD-IQ3_XXS reads a 95K-token prompt in 19.6 s fully resident; at `--vram 26` it is partially resident
and an 8K prompt takes 6.0 s instead of 2.5. The ready line says which placement a start took and,
when the first choice did not fit, what it would have needed: `prefill batch 4096 (K/V in VRAM,
prefill batch 4096, MTP drafter would need 2.6 GiB more VRAM)`, or for a partial placement
`(fully resident: K/V in host memory, prefill batch 1024 would need 2.3 GiB more VRAM)`. A read of 16384 new
tokens or more on the layers in parts is read in wider batches where the VRAM left beside the weights
holds their workspaces (`prefill batch 4096, up to 8192 rows from a read of 16384 new tokens`; what
they lacked otherwise): a 242K-token read takes 49.1 s instead of 50.0 on two R9700s at `--vram 30`.
A batch given with `--batch-size` is read as given, and `STRAYDOG_PREFILL_WIDE=off` turns the wide batches off.
A partially resident model on two NVIDIA GPUs with 40 to 48 GiB of budget in all (two 24 GB cards) reads in batches of 4096 rows from the start
(`prefill batch 4096`), and each layer of a batch asks for the experts its slots do not hold ahead, to come over the bus beside the layer's own
work (`STRAYDOG_PRESTAGE=0` turns that off): on two RTX 4090s a new 32K-token prompt gives its first token in 13.0 s instead of 24.3, an 8K one in 4.0 s
instead of 6.4. The two GPUs of such a model read different batches of a prompt at the same time (the first GPU the next batch while the second
finishes this one; `STRAYDOG_PREFILL_OVERLAP=0` reads them one after the other, the first logits are the same to the bit): the 32K prompt's first
token comes in 7.4 s instead of 12.4.

Like llama-server, `straydog` with no subcommand is the server (http://127.0.0.1:8080).
`-m` takes the bundle's directory or its first shard; `-md` the MTP drafter's directory
or its GGUF.
Every setting is an option; each also reads a `STRAYDOG_*` variable (`STRAYDOG_CTX_SIZE`),
and `--config FILE` reads them from a TOML file (with no `--config` and no model given, the one
`setup` wrote). Options beat variables, which beat the file. `--print-config` shows the
settings in effect and where each came from.

### Keeping a conversation across starts (Windows)

An engine that starts again, as an app's update makes it, holds none of the conversation it held:
the next request of a long conversation reads all of it again (36,169 tokens took 84 s on an
RTX 4060 at 5.5 GiB). With `--keep-conversation` (`keep_conversation = true` in the config file) it
does not.
- **What is kept.** When the server stops on purpose, once, the state of the conversation the engine
  held last, at the place its last reply began: the tokens, and the recurrent state, K/V and indexer
  blocks of its positions. A conversation with an image in it is not kept, nor one under 1024 tokens.
- **When it is written.** At the stop and at no other time: Ctrl+C, the console closing, Windows
  ending the session, or a line `stop` on standard input (`--stop-on-stdin`, for a program that
  starts the server and has no console to send Ctrl+C to). Nothing is written while requests run or
  per request: the drive sees one write a stop. A state larger than the memory or the drive has room
  for is not written, and the log says so in one line.
- **Where, and sealed how.** `<data dir>/conversation`, one file a bundle, a new one replacing the
  old. The state is sealed with AES-256-GCM under a random key that DPAPI wraps for your Windows
  account, as the engine's other saved state is (the system block's, `prefix`): another user cannot
  open it, and a changed byte refuses it. Its tokens are a sealed part of their own.
- **At the next start** the file is opened lazily: its tokens are read, and the state stays on the
  drive until the first request whose prompt begins with them. That request reads it from the file
  and goes on from there; the request line says `reused 38968 from kept, loaded in 1.8 s`. A
  request that does not begin with them leaves the file as it is, for a later one. A file of
  another build, bundle or placement of the engine is deleted, as is one that does not open, and
  with the option off at the start the file is deleted.
- **What it costs and saves.** The file is about 28 KB a token (1.0 GB for 36,000 tokens), written
  in 2 to 2.5 s at the stop (the stop takes 3.5 to 4.7 s). On an RTX 5070 Ti held to 5.5 GiB a
  conversation of 38,975 tokens was read in 48.9 s from nothing, and in 1.8 s of loading and 2.2 s to
  the first token from the kept state.
- **Off by default.** Without the option nothing of a conversation is written to the drive.

A bundle is named by three labels: its model (`--model-name`), its provider, who quantized it
(`--provider`), and its quantization (`--quant`). Where fewer name one bundle, fewer will do;
the server takes all three from the files in `-m`.

| Model | Provider | Quantization | Size | |
| --- | --- | --- | ---: | --- |
| DeepSeek-V4.1-Flash-1017 | Unsloth | UD-IQ1_M | 250.4 GB | recommended |
| | Unsloth | UD-IQ2_XXS | 300.5 GB | |
| | ISTA-DASLab | IQ2_XXS | 248.8 GB | GSQ-RCO; the faster where VRAM is short |
| | ISTA-DASLab | IQ2_XS | 265.8 GB | GSQ-RCO |
| Qwen3.8-Flash-Next | Unsloth | UD-IQ3_XXS | 83.6 GB | Dynamic V3.1 |
| | Unsloth | UD-IQ2_XS | 75.8 GB | |
| | Unsloth | UD-IQ1_M | 68.0 GB | |
| | Unsloth | UD-IQ1_S | 66.4 GB | |
| | ISTA-DASLab | IQ3_S | 83.6 GB | GSQ-RCO; closest to BF16 |
| | ISTA-DASLab | IQ3_XXS | 75.8 GB | |
| | ISTA-DASLab | IQ2_XS | 68.0 GB | |
| | ISTA-DASLab | Q2_0 | 66.4 GB | the fastest |

Unsloth's bundles come first: `straydog setup` recommends from them before ISTA-DASLab's. Unsloth's
Qwen3.8 bundles share their engram (PLE) table, the last shard of each: setup takes it from a bundle
beside it rather than downloading it again.

Unsloth's Qwen3.8 bundles read their engram table, 51.2 billion values read a few rows a token, from
their own last shard (IQ4_NL, 28.8 GB) unless another is named: one of Unsloth's engram files, BF16
(102.4 GB, lossless), Q8_0 (54.4 GB) or Q5_1 (38.4 GB), in the bundle's folder or an `engram` folder
beside it, which the model's bundles share. `straydog setup --engram Q8_0` fetches that file there instead
of the bundle's own table shard and names it in the settings; `--engram` (`STRAYDOG_ENGRAM`) names one
for the server. On two R9700s the precision changes no speed; on a machine short of memory a larger table
costs some (Q5_1 a tenth on 62 GB with half of it held by other programs). Against BF16 over six prompts'
greedy continuations (UD-IQ1_M), the KL of Q8_0 averaged 0.0015 (p99 0.016), Q5_1's 0.0021 (0.019) and
IQ4_NL's 0.0020 (0.038). The server's ready line names the table it reads.

Settings files of 0.1.3 and before name DeepSeek-V4.1's ISTA-DASLab bundles by their files'
names (GSQ-RCO-IQ2_XXS, GSQ-RCO-IQ2_XS) and the model DeepSeek-V4.1-Flash: they are read as these.

DeepSeek-V4.1's sizes include its 110.6 GB Engram tables, which stay on the SSD and are read
a few rows per token; its DSpark drafter adds 8.0 GB. On two R9700s and 128 GB of RAM,
ISTA-DASLab's IQ2_XXS decodes 57-77 tokens/s (prose to code) once a request's experts are in memory.

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

The [timing and cache fields](docs/DESIGN.md#101-timing-and-cache-fields) distinguish prefill
setup and restoration, expert-cache capacity and occupancy, and reused prompt positions.

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
| Qwen products over a batch of more than 4096 rows (a read of 16384 new tokens or more on the layers in parts) | as above, the products tiled for the wider batch, so the bits are those of a batch of that width (UD-IQ3_XXS, first logits against the F32 products: KL 2.0e-6, 2.0e-2, 1.0e-4 at 32K, 95K, 242K tokens against 6.6e-6, 2.6e-2, 2.9e-5 at 4096 rows; the same top token, KL between them 3.7e-6, 1.0e-3, 2.5e-5) | `STRAYDOG_PREFILL_WIDE=off` |
| Qwen's long prompts' attention on RDNA4 | FP16 products, F32 softmax and sums | `--gemm fp32` |
| Qwen's prompt attention on CUDA (Q8_0 K/V, RTX 30 series and later) | FP16 operands on the tensor cores, F32 softmax and sums, the numeric form of the RDNA4 row above. A layer's output against the F32 kernel's: 3.1e-4 at the most, relative to the largest value (FP16's rounding). The first logits after 8192- and 32768-token prompts (IQ2_XS, an RTX 5070 Ti, each kernel in a process of its own): KL 1.3e-3 and 2.2e-3 against the F32 kernel, the largest logit difference 0.58 and 0.83, the same first token; two runs of either kernel are bit for bit the same. Its FP16 operands also widen what reaches them a few F32 bits apart: the chunked DeltaNet scan's first logits against the token scan's (3001 tokens, UD-IQ1_S) are KL 1.7e-3 apart under it and 8.6e-4 under the F32 kernel. One layer of 4096 rows at position 98304: 16.0-17.1 ms -> 3.0-4.4 ms on an RTX 3090, 9.3-10.2 -> 2.5-2.7 ms on an RTX 5070 Ti | `STRAYDOG_QSA_MMA=0` |
| Qwen's DeltaNet layers over a prompt read of 16384 new tokens or more (RDNA4) | the recurrence in chunks of 16 tokens, the products against the state in bf16 pieces (three to a value, six of nine products) with F32 sums: another order of sums, which the model amplifies as it does any other (first logits after a 95K / 242K read against the token-by-token scan's: KL 6.7e-4 / 1.6e-5; the token-by-token scan against itself at another batch width: 4.4e-2 / 1.6e-5). On the activations of a 95K read, against an F64 scan of 4 heads in 12 layers, the outputs' largest error is 4e-7 to 3.6e-6 of the largest output (the token-by-token scan's: 5e-7 to 1.6e-4). Shorter reads keep the token-by-token scan, bit for bit | `STRAYDOG_DELTA_SCAN=token` |
| Qwen's DeltaNet layers over a prompt read of 16384 new tokens or more (CUDA) | the same recurrence in chunks of 16 tokens (the algebra of the RDNA4 row, `gdn_chunk_prep` shared), the products against the state in F32 fused multiply-adds (two lanes a column, each its half of the keys in order, the halves added): another order of sums than the token-by-token scan's, which the model amplifies as it does any other (Qwen3.8-Flash-Next IQ3_S on two RTX 4090s, first logits against the token-by-token scan's: KL 4.1e-6 after a 32K read and 3.0e-7 after a 95K read, the same top token; the token-by-token scan against itself at batch widths of 1024, 2048 and 3072 after a 32K read: 3.9e-6, 2.9e-6 and 7.5e-6; the first logits of a read are the same from run to run). Shorter reads keep the token-by-token scan, bit for bit | `STRAYDOG_DELTA_SCAN=token` |
| Qwen's HyperConnections over a prompt read of 16384 new tokens or more (RDNA4, FP16 projections) | the up projection (10240 x 320) and the finish in one kernel: the projection's sums on the matrix cores in a plain kernel's order of 16 products a step from zero (rocBLAS's order is another), everything after it `hc_finish_hidden`'s arithmetic, expression for expression, so the mixed rows are bit for bit what `hc_finish_hidden` makes of that projection (`gpu/tests/hc_fused_test.cpp`). Against an F64 evaluation of the same FP16 operands, on random rows of a part's 4096 tokens: rms 9.5e-8 and largest 2.9e-6 of a row's scale (hipblasGemmEx and the finish: 9.6e-8 and 3.2e-6; on 4000 rows 9.5e-8 and 3.8e-6 against 9.5e-8 and 3.7e-6). First logits after a 32K / 95K / 242K read against the F32 products': KL 1.8e-6 / 2.1e-3 / 5.9e-5 (the two kernels': 5.2e-6 / 1.7e-2 / 1.6e-5), the same top token. Shorter reads keep the two kernels, bit for bit | `STRAYDOG_HC=blas` |
| Qwen decode and speculative verification: the inputs of the dense Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 projections (RDNA4 and CUDA) | INT8 per 32 values with an F32 scale; the dots exact in INT32, then scaled in F32. A row's sum does not depend on what shares its launch, so a verification is still the tokens fed one at a time; the bits are no longer the F32 kernels' (CUDA, UD-IQ1_S, 356 decode steps: KL mean 1.1e-3, p99 1.2e-2; greedy continuations part after 19, 23 and 30 tokens in three of six prompts and agree for 128 in the others) | `STRAYDOG_Q8=0` |
| Qwen's K/V cache | Q8_0: each row's 256 values as int8 in blocks of 32 with an FP16 scale, rounded to nearest; read back as FP16 into the same F32 sums. Half the memory and half what decode reads of a K/V in host memory over the bus. Against the FP16 K/V (UD-IQ2_XS, one GPU held to 11 GiB, a 128-token reply's logits after a document of 8K / 65K / 250K tokens): on RDNA4 KL mean 6.3e-4 / 1.2e-3 / 1.2e-3, p99 at most 1.5e-2, the same top token on 366 of 367 rows; on CUDA 8.2e-4 / 1.2e-3 at 8K / 65K. The FP16 K/V before it (itself against F32: KL mean 3.6e-4 / 4.6e-4 / 6.6e-4 at 8K / 32K / 95K) | `STRAYDOG_KV=f16` (FP16) |
| Qwen's routed and shared experts' gate and up in decode and speculative verification (CUDA) | The default of a CUDA build (`STRAYDOG_EXPERT_Q8` unset or `2`; a HIP build only when asked): INT8 inputs per 32 values with an F32 scale, integer dots scaled and summed in F32; GGUF weights keep their format. On an RTX 5070 Ti (UD-IQ2_XS, 10 GiB VRAM budget, 16 of 48 layers page-locked), six greedy probes of up to 64 tokens, compared while their continuations agree, gave mean KL 2.2e-3 and 4.5e-3 against the F32-input reference; an F32-input repeat itself gave 1.1e-3, even with `STRAYDOG_PIPELINE=freeze`. These partial-residency runs include routing variation; the difference is not an isolated measure of input rounding. Two RTX 3090 (IQ3_S) decode 146 tok/s with it, 133 with F32 inputs | `STRAYDOG_EXPERT_Q8=0` (F32 inputs for this path) |
| DeepSeek products over prompt passes of at least 32 rows (FFN, attention, index, router and Engram projections; more than 4 rows per expert) | FP16 operands, F32 sums and output | `--gemm fp32` |
| DeepSeek decode and verification: the inputs of the Q8_0 and Q5_K projections and of the IQ2 experts' gate and up | INT8 per 32 values with an F32 scale; the dots exact in INT32, then scaled in F32 | none in the server (`DS41_Q8=0` in the examples) |
| DeepSeek decode and verification: the attention's and the indexer's scores | FP16 operands, F32 sums | `--gemm fp32` |
| DeepSeek's compressed K/V rows and index keys | BF16, rounded to nearest; the window rows stay F32 | none |
| DeepSeek's hyper-connection weights | BF16 on the device (F32 in the GGUF) | none |
| MTP drafter (Qwen3.8) | its experts quantized to Q2_0 when it is built: this changes only how many drafts are accepted, never the output | run without `-md` |
| DSpark drafter (DeepSeek) | each stage routes among its 48, 28 and 16 most used experts, which leaves the target more expert slots: this changes only how many drafts are accepted | `DS41_DSPARK_RESIDENT=all` (`DS41_DSPARK=0`: no drafter) |
| Qwen3.8's experts beyond VRAM (partial residency) | the CPU computes them (with activations rounded to 8-bit integers on a CPU with AVX-512 and VBMI or with AVX2, in F32 elsewhere) and prompt batches use the FP16 expert GEMM; which experts run where depends on timing, so decode is not bit-reproducible from run to run | full residency |

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
keeps it in that tab. The page is in English; a switch in the top bar shows it in Japanese, and
the browser remembers the choice. A link can name the language: `/dashboard?lang=ja`.

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
STRAYDOG_DS41=DIR STRAYDOG_DS41_QUANT=IQ2_XXS cargo test -p straydog-engine --features hip --test deepseek41_served
STRAYDOG_DS41=DIR STRAYDOG_DS41_QUANT=IQ2_XXS cargo test -p straydog-engine --features hip --test deepseek41_dspark_served
# A/B of this build against another
cargo run --release --features hip --example ab -- --baseline BIN --model DIR --drafter DIR --output ab.json
# The dashboard over a simulated engine (no GPU, no model): http://127.0.0.1:18090/dashboard
cargo run --release --features hip --example dashboard
```

## License

MIT. Code and data from other projects, and their licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
No model weights are part of this repository: each bundle is downloaded from its publisher and
is under that publisher's license.
