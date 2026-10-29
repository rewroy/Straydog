# Straydog — design

Straydog is a local inference engine and a server with an OpenAI-compatible API, shipped as one
executable per GPU backend (HIP on Linux, CUDA on Windows). It serves pinned GGUF bundles of three
models:

- **DeepSeek-V4.1-Flash** (748B total, 16B active), the main one;
- **Qwen3.8-Flash-Next**, with its MTP drafter;
- **Qwen3.6-35B-A3B**, for a small GPU and 32 GB of RAM.

This document describes what the code does now. It does not keep the project's history; git
does. The roundings are listed in the README's Numerics, which this document does not repeat.

Updated: 2026-10-29. All three models run on Linux with ROCm. The Qwen models have run on
Windows with CUDA; DeepSeek-V4.1 and `straydog setup` have not.

## 1. Goals

- **One long-lived server keeps the model loaded.** Clients start, update and restart without
  reloading it.
- **Any OpenAI-compatible client works.** Nothing in the server is specific to one client.
- **One executable.** It holds the server, the engine, the isolated worker and provisioning. The
  only outside pieces are the GPU vendor's runtime and driver.
- **Native code on every per-token and per-pass host step.**
- **Speed judged where users feel it:** the wait before the first token, prefill, and decode in
  real conversations (long agent sessions with tools), reported in tokens/s.

## 2. Contracts

- **No client logic.** Straydog serves the OpenAI-compatible API and nothing shaped for one
  client. The conversation cache, the reasoning replay and the tool-call split are not client
  logic: they follow from the official chat template and the OpenAI message format.
  Client integrations, client-shaped benchmarks and a client's secrets belong to the client.
- **Exactness.**
  - Weights are decoded exactly as ggml defines each GGUF type, and products accumulate in F32.
  - Every routed expert a token selects is computed.
  - Every rounding that buys speed is named in the README's Numerics, with its switch where it
    has one.
  - A speculative verification pass is bitwise identical to token-by-token decode in the same
    placement, so drafts never change an answer.
  - An unsupported tensor type is an error, never a silent conversion.
- **Isolation.**
  - The engine runs in a worker process without network access: bubblewrap on Linux, an
    AppContainer of its own on Windows. The worker is the same executable.
  - Every model and drafter file is pinned by size, SHA-256 and fs-verity digest. The pins are
    compiled into the executable, and every file is authenticated before use.
  - Bytes read from the drive after loading (table rows, experts outside RAM, banks left to the
    page cache) are checked block by block against the file's Merkle tree before use.
  - The trees live in the data directory (`.straydog` in the model's by default). Only the
    server writes them, never the worker, and they are checked against the compiled-in digests
    whenever they are opened.
  - The drive is read-only to the engine: no swap, weight or K/V files.
  - Straydog's own secret, its API key, never reaches the worker.
- **Measurement.**
  - A performance change is judged by alternating A/B runs (base, new, new, base) on the same
    machine, under a GPU lock, from the same warm state.
  - A change that alters the generated text is judged on its p10 as well as its median.
  - A numerical change is compared with an F32 run (for DeepSeek-V4.1 also with llama.cpp's
    logits) by KL divergence and top-1 agreement.

## 3. Processes and source layout

```
straydog ──HTTP──▶ clients
   │  runtime: one worker, a bounded queue (4), cancellation
   │  stdio, JSON lines (one frame per event; ~8 µs per token)
   ▼
straydog worker   (same executable, isolated; never opens a socket)
   tokenizer, template, weights, placement, generation loop
   │  C ABI (sd_*)
   ▼
GPU library (C++ / HIP / CUDA, linked statically)
```

The server never loads the model, and a worker failure never takes the server down. The worker
sees only what isolation exposes:
- the executable and the GPU runtime;
- the pinned model and drafter files;
- the GPU device nodes;
- a short list of environment variables (the `DS41_*` controls named below).

```
crates/
  straydog/            the executable: CLI, config, setup, server, runtime, dashboard
  straydog-engine/     tokenizer, template, GGUF, weights, formats, ops, architectures,
                       expert store, row tables, sampling, constraints, worker
  straydog-gpu/        bindings to the C ABI; build.rs compiles gpu/ with CMake (HIP or CUDA)
  straydog-gguf/       GGUF structure checks
  straydog-api/        the OpenAI-compatible API over an Engine trait
  straydog-isolation/  bubblewrap and AppContainer containment of the worker
  straydog-provision/  fetch, verify, bundles and their pins, the MTP drafter build
gpu/                   C++/HIP/CUDA kernels; straydog_gpu.h is the C ABI
bench/                 the A/B harness's cases (examples/ab.rs)
```

## 4. The engine

One engine serves every model. The GGUF's `general.architecture` selects the architecture, and
dimensions, layer kinds, routers, attention and table parameters come from the file's metadata.
The engine is built in layers:

1. **Runtime.** Devices, streams, links between GPUs, page-locked arenas, recorded graphs, the
   storage reader, digests. There is no architecture in it.
2. **Formats.** Every GGUF block type has a decoder plus GEMV and GEMM kernels on the GPU (and
   on the CPU where a model computes experts there), each tested against the format's
   definition. Architectures never branch on formats.
3. **Ops.** Building blocks that take their parameters from metadata:
   - norms, rotary embeddings (with YaRN), linear layers, SwiGLU (with clamping);
   - MoE routers (softmax, sigmoid, square-root softplus; bias, normalization, scale), and
     expert execution over the expert store;
   - attention kinds: grouped-query, latent, sliding-window, sparse with an indexer, compressed;
   - recurrent kinds: Gated DeltaNet with its convolution;
   - hyper-connections (n streams, Sinkhorn);
   - embedding tables: token embeddings, and per-layer n-gram tables gathered by row;
   - output heads.
4. **Architectures** (`arch/deepseek41`, `arch/qwen35moe`, the Qwen3.8 engine). An architecture
   reads its hyperparameters and tensor names, gives every tensor a role (dense per layer,
   routed experts, a gathered table, the head, a drafter) and composes its layers from the ops.
   `arch/served.rs` holds what every served model shares: continuing the held sequence,
   checkpoints, sampling, metrics, the warm-up of expert stores.
5. **Placement.** Roles go to tiers by their sizes and access patterns: VRAM per device,
   page-locked host memory, the drive (§5). The architecture never chooses between splitting
   by layers and splitting every layer across the GPUs; placement does.
6. **Serving.** The tokenizer (from `tokenizer.json` or the GGUF's metadata), the chat template,
   the family's reasoning and tool-call parsers, its drafter, sampling and constrained
   decoding (§6).

### 4.1 Loading

- **Reading and authenticating.**
  - Each pinned file is authenticated in 256 MiB chunks against digests compiled in beside its
    SHA-256.
  - Opening verifies each file's first chunk and parses the GGUF header from it. Placement runs
    on the header alone.
  - Then every core reads chunks into page-locked memory, hashes them there, and copies a chunk
    whose digest matched to the GPU allocations and the host copies. The bytes hashed are the
    bytes used.
  - A cached chunk is read from the page cache. When the files fit in three quarters of host
    memory, an uncached one is read through the page cache, so a restart finds it; otherwise it
    is read around it (direct reads).
- **A model read on demand** (DeepSeek-V4.1, Qwen3.6) opens its shards against their verity
  trees. The routed experts are not read whole at the start: the expert store fills from the
  files as tokens route to them, and the drive blocks they come from are checked against the
  trees.
- **Startup times.** DeepSeek-V4.1's server is ready in 14 s with its dense weights in the page
  cache. Qwen3.8's IQ3_S with its drafter takes 22-23 s from a cold start and 13-14 s cached.
- **Warm-up.** Every two minutes after requests, and at shutdown (also when no request came but
  the worker was handed a state to warm up with), the runtime asks the worker
  which experts its stores hold, least recently used first, and keeps the list in the data
  directory (`<data-dir>/warm/<quant>-<key>.json`). The key hashes the pinned shards and the
  placement. The runtime writes this file; the worker has no write access.
  - **The state never loses what the old state listed.** A run that ends before its warm-up has
    read everything (a request came, or it was stopped) saves the experts still pending, older
    than anything its stores hold, besides what they hold; only what its own requests evicted
    leaves the list. (Saving only what the stores held shrank the file with every short run:
    1914 of 6067 VRAM slots and 373 host slots, against 5291 and 4315 after a long run.)
  - **`counts`.** The state also keeps how many decode and verify passes used each expert
    (`[layer, expert, passes]`). A prefill's streamed layers are not counted: they take nearly every
    expert whatever the prompt. A run adds its own passes to the counts it was given, and ages the given
    ones by what it adds: not at all when it served nothing, hardly for a restart's one reply, by half
    when it added as many passes as they hold (so a long run's weight is half that of the runs before
    it and a restart does not erase the history). A state without `counts` (earlier versions) loads
    as one with none.
- **Start with the same model and placement:** once ready, the worker reads those experts back
  while no request runs, a few reads at a time, behind any pass's own reads.
  - **Most recently used first, in their old places.** A start cut short has the experts likeliest to
    be needed. Each read lands in the LRU place it had in the old state (the stores take a rank for
    each expert before its read, `DeviceExperts::rank_warm`), so reading in the other order leaves
    the old order. They are older than anything a request touches.
  - **The fill.** Once the listed experts have landed, the free slots are offered the rest of the
    model's experts, the ones most used first (`counts`; none counted: by expert, then layer, so
    that each layer gets as many). The likeliest go to the free VRAM slots (beyond each size class's
    reserve and the room a prefill's first waves take), the next to the host tier (beyond a reserve
    of its slots). It never evicts and ends when the stores are full, and the fill's experts are
    older than the old state's in the LRU order. `Warm` (`arch/warm.rs`) drives all of it; between
    requests the worker makes room (and copies checkpoints) before it takes a warm-up step.

### 4.2 Row tables

Some embedding tables are read only a few rows per token, at random:
- Qwen3.8's PLE table: 28.8 GB of 90-byte rows, 16 per token;
- DeepSeek-V4.1's Engram tables: 110.6 GB in the GSQ-RCO bundles.

These tables are never resident. Their rows are read from the authenticated file into a row
cache of the engine's own (`rows.rs`, `--row-cache`). It is 256 MiB, except for Qwen3.8's PLE
table: that one takes every row of the table (36.5 GB with the cache's index) while half of the
host memory left beside Straydog's sequence state holds them, else that half, and never less
than 256 MiB. The other half stays to the page cache, parked sequences, checkpoints and the
rest of the machine. The cache fills as rows are used, so memory is taken only for the rows a
conversation has met.

- **Why rows rather than pages.** A 4 KiB page holds dozens of rows, and one of them is wanted.
  Meanwhile, 70-98% of row uses in agent sessions and chats repeat an earlier row. So 64-256 MB
  of rows hit as often as gigabytes of page cache would.
- **How a pass reads.**
  - A pass's missing rows are deduplicated and read together on 128 threads, as whole aligned
    4 KiB blocks.
  - Reads go around the page cache (O_DIRECT) where the platform allows. Blocks the page cache
    already holds are read through it.
  - Reads start as soon as the tokens are known: the next prompt batch's while the current one
    runs, a sampled token's and its drafts' before the pass that uses them. Decode waits only for
    rows still on their way, at the layer that gathers them.
- **Verification.** Each block is checked against the shard's fs-verity tree between its read
  and its use. A block that fails fails its read; a later use reads it again.
- **Memory pressure.** DeepSeek-V4.1 decode waits 0.1-0.4 ms a token on Engram rows. If the
  page cache loses the tables' blocks to other work on the machine, that wait rises to ~3 ms a
  token: on a 125 GB host with a 102 GiB pinned expert tier, a large build beside the server
  does it.

## 5. Placement and the expert store

There is no `-ngl`. Placement measures what fits, in this order:

1. **Full residency.** Every weight in VRAM, the K/V too.
2. **Host K/V,** then a smaller prefill batch (Qwen3.8).
3. **Partial residency.** Dense weights and sequence state in VRAM; routed experts in an expert
   store of VRAM slots over a page-locked host tier over the files.

Each GPU's budget is its free memory less max(512 MiB, 3%) (`--vram auto`), or what `--vram`
gives. Two GPUs of similar budgets split every layer between them (§7.1, §8, §9); GPUs whose
budgets differ by more than a tenth split by layers, each taking layers in proportion to its
budget.

The expert store (`experts.rs`) has three tiers:

- **VRAM slots per device.** Slots are grouped by size class (a model's layers may use several
  expert types). The VRAM left after everything else is counted in units of the largest
  expert's share, and each class gets its layers' share of the bytes. Each class replaces its
  least recently used slot.
  - VRAM idle between passes holds slots too: the prefill rows' arena, each kv source's
    compressed rows and index keys past what the sequence has reached, and the batch buffers
    (the index scores, the queries, the attended rows, Engram's values) past the rows a pass fills
    and, for the scores, the groups its positions reach: 357 MiB a device at 262144 tokens, nearly
    all of it for a short pass. A pass takes back what it reaches (its experts go to the host tier).
  - Once the server's first passes have run, each device lends its store, for good, the VRAM its
    card still has free beyond 768 MiB (within its budget). The placement's `RESERVE` and the
    budget's margin are for the runtime's own memory and other programs, and what they leave is
    not used otherwise. The 768 MiB is the least automatic margin (512 MiB), the 204 MiB the
    driver counts used that the runtime's free does not show, and 46 MiB for what a session's
    first use of every feature took after the first passes (a 144K-token prompt, an image, a
    JSON schema, a required tool call, thinking, sampling). `DS41_LEND_SPARE=0` lends nothing.
- **A page-locked host tier.** By default it gets what Straydog's other host memory leaves of 85%
  of host memory (as the server measures it, cgroup limits included).
  - The other host memory is the row cache, 2 GiB for the rest, and the checkpoints.
  - For DeepSeek-V4.1 the tiers are exclusive. The host tier takes what VRAM evicts, written back
    over the bus's other direction, and gives up what it hands to VRAM. On traced routes this
    halved the drive's reads against a tier that kept copies.
- **The files.** Read as whole aligned 4 KiB blocks around the page cache, each block verified
  (§2).

Copies run on each device's upload and download streams beside compute
(`straydog-engine/src/transfer.rs`). The compute stream waits only for the copies of the experts
it uses next.

Replacement was measured on traced real sessions, with an exact simulator of the store (DeepSeek
IQ2_XXS with DSpark, 2026-10-28):
- Against LRU, these policies change misses by: LFU with decay −9% (worse after a change of
  topic), SLRU −8%, ARC −2%, a TinyLFU admission filter +18%. Belady's optimum, which needs the
  future, is −45 to −55%.
- Predicting the next layer's experts (`DS41_PREDICT`, off) reaches 30% of the misses only by
  uploading 4-5 times the experts actually missed.
- Computing misses on the CPU did not pay either.
- So the store is plain LRU, and capacity is the lever: 10% more slots cut misses by 16%.

## 6. Generation

### 6.1 The held sequence and checkpoints

The engine holds one sequence.

- A prompt that extends it continues it. The response reports the reused tokens as
  `cached_tokens`.
- Reasoning sent back in assistant messages is part of the prompt the official template
  renders, so a client that returns it keeps the cache warm.
- **DeepSeek-V4.1** keeps up to four conversation checkpoints: the whole sequence state, in host
  memory, within 2 GiB.
  - A checkpoint is saved where a reply's generation prompt begins, plus one at the root.
  - A prompt that leaves the held sequence at an earlier turn, or comes back to a conversation
    switched away from, restarts from the latest checkpoint it extends.
  - Checkpoint copies are made while the engine is idle, not before the first token.
  - Beyond about 160K tokens a checkpoint no longer fits the budget, and none is saved.
- **Qwen3.8** parks a sequence when a prompt departs from it at least 1024 tokens below its end:
  - Its K/V and indexer positions from the departure on go to host memory (up to an eighth of
    it), and its recurrent state to a page-locked slot.
  - A later prompt that resumes further on the parked sequence than on the held one brings it
    back (`parked`: `saved` or `restored`, with `park_seconds`).
  - So a one-off request with another prefix (a title, a summary, a subagent) does not cost the
    conversation it interrupted its whole prompt again.
- **Qwen3.6** keeps the DeltaNets' state where each prompt's generation prompt begins (§9).
- A request may mark itself one-off (`"straydog": {"checkpoint": false}`) so it saves no
  checkpoint.

### 6.2 Speculative decoding

Drafts are proposals; verification decides. A verify pass computes every drafted row with the
target, and each row's logits equal what token-by-token decode would compute, bit for bit.

- **With temperature:** a draft that is a point proposal (a lookup's, an argmax, or any under a
  cut or a penalty) is accepted with its probability under the same cut and penalty, and a
  rejected one is replaced by a draw from the target without it. A draft a drafter drew from its
  own distribution q (DSpark, §7.4) is kept with probability min(1, p/q) and a rejected one is
  replaced by a draw from norm(max(0, p - q)): speculative sampling. Either way the output
  distribution is the target's.
  - A drawn draft costs the host one pass over the target's row, not a distribution of each
    (~0.15 ms a round at 129K logits, where building the target's and the drafter's cost ~1 ms):
    - it is kept or not by the row's sum of exp((logit - peak) / temperature), the draft's logit
      in it, and the draft's probability under the drafter's softmax (which the drafter gives:
      §7.4);
    - a rejected draft's replacement, and the token after the last kept draft, are drawn from the
      rows by their block sums and the weights of the one block the draw falls in, with no
      distribution built;
    - the sums are in single precision (about 30 us over a row) and the probabilities right to a
      part in a million, as finely as the drafter's own draws are known. The distributions the
      other paths build (point proposals, cuts, penalties) are in double precision, by an exp
      that matches libm's to a rounding and vectorizes (256- or 512-bit registers where the CPU
      has them).
- **Greedy decoding:** the text with drafts is the text without.

Three drafters:

- **Qwen3.8's MTP drafter** (`-md`, `--draft-max`, default 3). It drafts while the product of
  the drafts' probabilities under the drafter stays at 0.6 or more.
- **DeepSeek-V4.1's DSpark** (§7.4). It needs no option: it is used when its file sits beside the
  bundle.
- **An n-gram lookup of the sequence's own text.** A match of eight tokens or more (an edit
  repeating its source) is verified as it is.

A constrained reply takes no drafts.

### 6.3 Sampling and constraints

- **Sampling.** Greedy at temperature 0. Otherwise `softmax(logits / temperature)`, after
  `presence_penalty` is subtracted from every token the answer already holds, then cut to
  `top_k`, then `top_p`, then `min_p` (vLLM's order).
  - A control the request leaves out takes the server's setting, else the model card's for the
    request's mode.
  - Qwen3.8's card: thinking `temperature=1.0, top_p=0.95, top_k=20`; non-thinking
    `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`.
  - A GGUF with no `general.sampling.*` (DeepSeek-V4.1's) samples at `temperature=1.0` without
    truncation.
  - Sampling uses Straydog's own RNG, so a seed is reproducible within Straydog only.
  - Greedy decoding loops with Qwen3.8, so it is never the default.
- **Constrained decoding** (`constraint.rs`). After the reasoning (`</think>`):
  - a JSON `response_format` holds the answer to a JSON value the schema allows, racing a tool
    call's opening when tools may be called;
  - `tool_choice` `required`, or a named function, makes the answer start with the markup's
    tool-call opening (and the function's name), free after it.

  Each token is drawn from the model's distribution, and drawn again without a refused one
  (masked after 32), so allowed tokens keep their relative probabilities.

## 7. DeepSeek-V4.1-Flash

The `deepseek41` architecture:
- 40 layers of hyper-connected MoE blocks, with 384 routed experts per layer, 6 used per token,
  and a shared expert;
- latent attention over a sliding window, plus compressed rows selected by an indexer (CSA2);
- n-gram Engram tables.

Its definitions follow the reference implementation, checked layer by layer against it
(examples/ds41_gpu, ds41_chat) and against llama.cpp's logits (examples/ds41_quality).

### 7.1 Bundles and layout

Pinned bundles:
- **GSQ-RCO-IQ2_XXS** (ISTA, the default). Its experts are IQ1_M in 20 layers, IQ2_XXS in 14 and
  IQ2_XS in 6.
- **GSQ-RCO-IQ2_XS** (ISTA).
- **UD-IQ2_XXS** (Unsloth).

Optional files sit beside a bundle: the DSpark drafter (§7.4) and the vision encoder (§7.5).

ISTA's GGUFs come in two layouts:
- **IQ2_XS: four shards,** with the Engram tables in a file beside them.
  - Engram's hash parameters are integer tensors.
  - Its norm weights stand apart (`engram_q`, `engram_k`, `engram_wkv`).
  - The metadata names Engram's source layer and DSpark's target layers.
- **The repository's Oct-27 layout (IQ2_XXS): six shards.**
  - Engram's parameters are in the metadata (`ple.*`, `engram.token_map`); the offsets run on
    across the modules, so each module's are rebased to its first column.
  - `engram_kv`, and the gate folded (`engram_gate`, q times k).
  - Each layer's image routing bias (`exp_probs_b_vl.bias`).
  - No source or target layers: DSpark's come from its own `dflash.target_layers`.

  The IQ2_XXS bundle pins the first layout's Engram file as its sixth shard: the tables are the
  same bytes the Oct-27 file holds under one tensor's header.

The chat template is Straydog's copy of Unsloth's port of the model's `encoding.py`, rendering
the reference encoder's prompts. Other GGUFs embed other templates, and one that writes the DSML
tool-call tags otherwise than the model does never reads a reply that called a tool back as the
model wrote it. With such a template, no turn would continue the held sequence. Replies split at
the first `</think>` and at DSML tool calls.

**Placement on GPUs.** `--split-mode experts`, the default, splits every layer across the GPUs
(tensor parallel). Each device runs:
- its share of the attention heads: its rows of `q_b`, its sinks, its groups of `wo_a` and
  their columns of `wo_b`;
- its rows of the head;
- its rows of every expert, shared and routed: its share of gate and up, its columns of down.

The rest is replicated. Every device sums the devices' parts in device order:
- after the attention;
- of the experts' activations;
- after the experts.

So all devices hold the same values, and every route runs on every device. A missing expert
comes over every device's bus at once, each a share of it.

In a decode pass, the kernels forming a part also store it into the other GPU's inbox and raise
its flag (`gpu/link_tail.hpp`). The step reading the sum waits for the flag and adds the other's
part as it reads it. A pass of several rows sums with a kernel once both parts are done.
`--split-mode layer` gives each GPU a range of layers instead, cut where no layer reads another
part's compressed rows.

**VRAM per device.** The resident share of the dense weights, then the sequence state for the
whole context:
- window rings in F32;
- each kv source's compressed rows and index keys in BF16;
- a batch's new rows.

Then:
- the rows of a prefill pass (lent to the expert store between long passes);
- 512 MiB of scratch;
- 192 MiB for FP16 copies of the dense weights prompt passes use;
- the DSpark drafter's footprint;
- `RESERVE`, 1.625 GiB for the runtime's own memory. On DeepSeek-V4.1 that memory (kernels,
  handles, graphs) is 0.9 GiB; the rest of `RESERVE` covers Straydog's own buffers the terms above
  do not count: the device's workspace (352 MiB: a 128 MiB GEMM expansion, 128 MiB of input and
  output staging, 32 MiB of half inputs, 16 MiB of BLAS workspace, 48 MiB of attention partials)
  and the batch buffers (about 366 MiB, 357 of them lent to the store between passes), less 75 MiB
  the resident share is counted over. DSpark's Markov head (131 MiB) is on the first device alone,
  though the placement counts the first device's footprint on both.

The rest is expert slots. After the first passes each device also lends what its card has free
beyond 768 MiB (§5). Example: two R9700s with GSQ-RCO-IQ2_XXS at 262144 tokens hold 6064 slots
per device with the default prefill batch of 8192 (6067 with 4096: the arena for a larger pass's
rows is lent between passes) and a 102.2 GiB host tier (12,629 experts' worth, of 15,360), using
60.6 of 61.5 GiB.

### 7.2 Passes

- **One pass over all its rows.** A pass runs layer by layer over all its tokens: a layer's
  attention a batch of rows at a time, then its experts over all the rows. Each routed expert is
  read once for them all, in waves, with the next wave's fills behind the one computing.
- **A long prompt** takes back the slots lent to the store and streams its experts through VRAM,
  while the host tier keeps theirs.
- **The prefill batch** is 8192 rows when nothing else is asked for, fewer (down to 256) while
  its pass buffers (~0.3 MB a row on each GPU) would take over an eighth of the smallest GPU
  budget; `--batch-size` takes up to 8192.
  - Every pass reads the ~35 GB a GPU of experts the VRAM does not hold over the bus (~1.5 s at
    ~24 GB/s on both GPUs), whatever its rows. A layer's experts over 4096 rows computed in
    ~16 ms against ~40 ms of copies; over 8192 rows they compute in ~30 ms. A 70K-token prompt
    moved ~600 GB a GPU in passes of 4096 and ~350 GB in 8192.
  - The lent arena (§5) covers the larger buffers: the slot count between passes does not
    change, and a long pass writes back the few hundred experts its buffers take.
  - 16384 rows were slower than 4096 (88 s): the host side of a pass (the Engram rows read ahead
    for three batches, 1 GB of embeddings a pass) outgrew the row cache and the free memory of a
    machine whose host tier pins most of it.
  - A pass embeds its tokens a slab of 1024 at a time, uploading each. The embedding rows of the
    batches to come are read ahead before their Engram rows: those queue the drive 128 deep, and
    a page fault of the embedding table behind them waited seconds.
  - With `DS41_PROFILE`, each pass of 256 rows or more logs its seconds and its phases.
- **The decoder reaches back only through its sliding windows.** The decoder is the layers from
  the last kv source on. Of a pass of n rows, layer k computes only the last
  1 + (layers − 1 − k) × (window − 1) rows, plus the K/V of the window − 1 rows before them. The
  kv source still compresses every row. This is exact; `DS41_FULL_DECODER` computes every row,
  for comparisons.
- **Products in prompt passes of 32 rows or more** use FP16 operands; smaller passes use F32
  (README, Numerics).
- **A decode step.**
  - Each device runs the step as recorded graphs, replayed while their launches stay the same.
  - The router writes its routes into page-locked memory, which the host polls.
  - Then, layer by layer:
    1. The misses are requested on every device before any device waits, filled one expert at a
       time across the devices.
    2. The hits' gate and up run while the misses' uploads land. One event stands behind all
       the misses' gates and ups, and one behind all their downs.
    3. The misses' gate and up run, then every expert's down.
  - Evicted slots are written back to the host tier after the compute stream passes the point
    where the slots were last read. Each size class keeps up to 32 slots free during passes, and
    64 more between requests, for a prefill's first waves.
- **What a miss costs.** On two R9700s (PCIe 5.0 x8 each) a miss costs ~0.22 ms of a pass. The
  copy is ~0.17 ms of that, at ~26 GB/s per GPU with both copying.
- **Engram rows** for a pass's tokens are prefetched before it (§4.2) and gathered at the layers
  that hold a module.
- **Verify passes.** A pass of up to the verify rows can be taken back but for its first few
  (`Engine::rewind`): it keeps the sequence state's counters from before it, and each ratio-2 kv
  source's compressor rows of its tokens. A verify pass computes every row in every layer, with
  every token's logits.

### 7.3 Speed (2026-10-28)

Two R9700s, 125 GiB of RAM, GSQ-RCO-IQ2_XXS with DSpark, the owner's agent sessions:

| Measure | Result |
| --- | --- |
| Decode, 3K context | 78-81 tokens/s |
| Decode, 10-50K | 57-66 tokens/s |
| Decode, 55-73K | ~57 tokens/s |
| Prefill, a turn continuing the conversation | 800-1100 tokens/s |
| First token, a fresh 19.5K prompt | 18-22 s |
| First token, a fresh 70K prompt (2026-10-29, Engram's page cache dropped) | 45-46 s (58-62 s in passes of 4096) |

| Pass, at ~50K context | Wall time | Misses |
| --- | --- | --- |
| 1 row | 18.0 ms | ~19 |
| 2 rows | 25.5 ms | ~33 |
| 3 rows | 32.4 ms | ~48 |
| 4 rows | 40.5 ms | ~70 |

### 7.4 DSpark

DeepSeek's three-stage block drafter (`mtp.0..2`, Unsloth's Q8_0 GGUF, 7.98 GB) is an optional
file beside the bundle.

**The file.**
- It is pinned like the shards, with its repository and revision.
- It lies in the model's `dspark` directory or in one beside the model's, so a model's
  quantizations share it. `straydog fetch` and `setup` download it.
- Its tensor names are the model's own, so it opens apart from the shards (every tensor
  verifies when taken), and its layout is checked against the model's header.
- Absent, the server runs without it. `DS41_DSPARK=0` leaves it out; `DS41_DRAFTS=0` turns off
  drafting altogether.

**How it drafts.**
- It sits beside the engine that runs the target layers, split over the GPUs as the target is.
- It reads the target's features there (the mean of the hyper-connection copies of those layers'
  inputs) into rings of the last 128 positions.
- Its logits come from the target's head. It only proposes: nothing in it changes what the
  target computes.
- Each pass puts its rows' features in the rings, and a rewind keeps only the rows it keeps. A
  reset empties them, and a checkpoint keeps them with the sequence state, so a draft never
  reads another sequence's features.
- **A sampled reply** (a temperature, nothing cut, no presence penalty: DeepSeek's default)
  gets drafts the head draws from the softmax of its own corrected logits at the request's
  temperature, not their argmax:
  - the device draws (the Gumbel-max trick, noise from a seed each round has, the slot and the
    column), and each slot's Markov features follow the token drawn before it;
  - the device also gives each draw's probability under the softmax it was drawn from (a sum of
    exp over the corrected logits: one more small kernel a slot in the head's graph), and the
    accept rule ([`sampling::accept_drawn`], §6.2) keeps a draft with probability min(1, p/q);
    the corrected logits of a draft the target rejects stay on the device until the rule asks for
    that one row;
  - a draft is then kept with the overlap of the two distributions, 1 - TV(p, q), where an
    argmax is kept with p(argmax) alone;
  - with a cut or a penalty the draft stays the argmax, a point proposal.

**Its VRAM.** Each stage keeps its most used routed experts resident: 87, 50 and 29 of 128 (3.1 of
7.2 GB). They cover 98% of its routes and leave its acceptance as it was. The router's bias keeps
the others out of a draft's top three. `DS41_DSPARK_RESIDENT=A,B,C`, or `all`, changes them.

**The width policy** (`drafter::Policy`). After each sampled token the drafter drafts three tokens
(~2.3 ms), and the policy picks how many a verify pass checks. It picks the width with the most
expected tokens a millisecond:
- the expected tokens come from the drafter's confidences, calibrated against teacher-forced
  match rates and then against the run's own drafts;
- a pass's cost by its rows is 1, 1.6, 2.1 and 2.8 one-row passes. One-row passes measure the
  unit, and the 2-4-row passes seen scale the shape.

A draft costs its milliseconds whatever the round makes of it, so the policy also keeps a running
mean of what its rounds saved. When that mean falls below nothing, drafting pauses for 16 rounds.

### 7.5 Images

The official checkpoint's vision tower and its other image tensors make an optional file beside
the bundle, `DeepSeek-V4.1-Flash-1017-Vision-BF16.gguf`:
- the tower: 32 blocks of 1024, 16 heads, 2D rope, 14-pixel patches;
- the aligner: each 3×3 of patches to one row of 5120;
- the image span's START, NEWLINE and END rows.

`straydog fetch` range-fetches those 306 tensors, each SHA-256 pinned at a pinned revision. The
file is itself pinned. Without it, the model reads no images.

How an image is read:
1. The worker decodes the image (PNG, JPEG, WebP, GIF), sizes it as the reference plans it (at
   least 544×544 pixels' worth, at most 1024 positions) and pads it as Pillow's `ImageOps.pad`
   does, bit for bit.
2. The template renders each image part as a marker of its own. The marker's place in the prompt
   takes the image's span, `START (IMAGE × columns, NEWLINE) × rows END`, with position ids from
   2^31 up made from the image's digest. So a held sequence continues only with the same image.
3. The encoder runs on the first GPU, in the arena a long pass takes from the expert store, with
   FP16 products and F32 sums. The last few images' rows are kept for later turns.
4. A pass routes the span's rows with the image routing bias and gathers no Engram rows there.

## 8. Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is served from four pinned bundles (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) with the
tokenizer and template files of the official release, and optionally with its MTP drafter.
`straydog fetch-mtp` builds the drafter from the official tensors, byte for byte the pinned file.
Its experts are quantized to Q2_0, which changes only how many drafts are accepted.

Its regression oracle is the pinned GPU generations (`straydog-engine/tests/model_gpu.rs`). For
each placement it pins every request's greedy tokens, metrics, logits and final hidden state, bit
for bit. A deliberate numerical change is checked against an F32-GEMM run, then pinned anew
(`STRAYDOG_GOLDEN_UPDATE=1`).

**Placement.**
- **Two R9700s:** they hold the weights of every bundle, each layer's experts split between them,
  with the K/V in host memory at 262144 tokens.
- **Smaller budgets:** host K/V and a smaller prefill batch come first, then partial residency.
- **Partial residency:** the CPU computes experts beyond VRAM in F32 from the host tier. Which
  experts run where depends on timing, so decode is not bit-reproducible from run to run.

**Long-context prefill.**
- Every fourth layer attends sparsely: an indexer scores the prompt's 4-token blocks, and each
  token attends the keys of its 512 best complete blocks and of its own incomplete block.
- A prompt is prefilled in batches, split into 256-row chunks for the indexer.
- With FP16 GEMM on RDNA4, a batch's attention runs at once. Groups of 16 tokens walk the union of
  their selected blocks, so each key and value is read once per group.
- With host K/V, a layer's prefix is staged in VRAM for its batch, copied on a stream of its own
  while the layers before it compute.
- Measured at IQ3_S on two R9700s at 262144 context: 131072 tokens prefill in 34.3-34.8 s, and
  4096 in 1.23 s.

**Decode.** A pass runs a token and its MTP drafts as the rows of one verification through every
layer (§6.2).
- **Placement of a pass.** The first GPU owns the layers up to the 25th, the second the rest, the
  head and the MTP layer. A layer's routed experts are split in halves between the GPUs, so the
  one that does not own the layer computes its half of the experts and otherwise waits: a third of
  each GPU's pass is spent in the kernels that wait for the other.
- **A verification's attention** runs its rows side by side, one launch of each kernel for all of
  them (`qsa_decode_attend_rows`): each row has the arithmetic of one token's decode, so the
  rows' state and logits equal token-by-token decode bit for bit, but a row costs the layer no
  launches of its own and its selection runs beside the others'. A 4-row pass takes 3 ms less
  at 20K tokens (33.9 to 30.9 ms) and 5 ms less at 60K (40.2 to 35.2 ms).
- **Measured** at IQ3_S on two R9700s, 262144 context, MTP 3, in tool-shaped conversations of
  about 20K and 60K tokens with the model card's sampling (2026-10-29):

  | Context | Decode | Tokens a pass | A pass of 1 / 2 / 3 / 4 rows |
  | --- | --- | --- | --- |
  | ~20K | 81-98 tok/s | 1.9-3.0 | 17.4-17.6 / 22.5-23.2 / 26.6-26.9 / 30.7-31.1 ms |
  | ~60K | 68-85 tok/s | 2.0-2.9 | 19.2-20.5 / 25.3-27.6 / 30.3-33.7 / 35.1-39.5 ms |

  Without the drafter a step takes 16 ms at 20K (62 tok/s) and 17.5-18.5 ms at 60K.
- **Where a pass goes** (a 20K token, 12 ms): the layers 78%, drafting 12% (the MTP layer 0.85 ms
  and its head 0.55 ms a draft), the head of the verification 4%, the first layer and the PLE
  layer 4%, the commit, accept and prefetches 2%. The bandwidth floor of a pass is 6.8 ms (one
  row) to 9.0 (four) at 640 GB/s: a row reads 5.2 GB (the HyperConnection matrices 1.3, DeltaNet
  and attention 2.0, routed experts 1.0, the head 0.5, shared experts, routers and PLE 0.4). A
  layer takes 300 µs against 130 µs of bandwidth: some 17 kernels, each followed by 3.2 µs of
  graph dispatch, and the projections reach 300-400 GB/s.
- **Host waits of a pass** (2026-10-29). A host-device round trip costs 20 µs (a kernel and a
  stream synchronize, or a 4 KB copy) in every wait mode the runtime has (spin, yield, blocking,
  interrupts off), so what a wait costs is the host's work around it and the device idle meanwhile,
  and the pass drops them where it can:
  - the first layer is queued and the PLE rows are waited for beside it; the PLE layer then runs on
    the first layer's rows where they are, with the PLE state snapshotted and the first layer's rows
    kept on the device, and one wait reads both layers' rows back (0.99 to 0.79 ms a 4-row pass);
  - a partial acceptance replays the DeltaNet states and the PLE layer without a wait: the stream
    orders the replay before everything that reads the state (the commit 0.41 to 0.03 ms);
  - a drafting step is one call: the MTP layer's prologue, the layer and the head on its last row
    run on the device, and a chain of steps keeps the rows there (the token and its probability are
    all that comes back): 0.2 ms a 4-row pass;
  - the heads' read-back of logits and tokens is queued together, with one wait and the copy-out on
    the copier's threads (0.1 ms).
  Together a pass takes 0.15-0.25 ms less at one row and 0.8 ms less at four (30.9 to 30.0 ms at 20K,
  35.2 to 34.4 ms at 60K); the exactness tests are unchanged. What did not pay: launching the
  helper's and the owner's span graphs from two threads (a launch takes the host 0.22 and 0.5 ms, but
  the spans took no less), and splitting the MTP layer's experts across the GPUs (from the trace, its
  experts are 60 µs of a 530 µs layer, and the peer exchange would cost about half of what the split
  saves; it would also change the drafter's sums, hence the pins).
- **What is left in a drafting step** (1 row, 1.25 ms; kernel times from the trace): the prologue 0.1 ms, the MTP layer 0.53 ms
  (its projections 0.22 ms at 460 GB/s, the experts 0.06, the HyperConnections and attention the
  rest) and the head 0.56 ms (0.43 of it the half vocabulary's GEMV at 590 GB/s), all device time;
  the host turnaround between steps (the token to its embedding and back, about 0.15 ms) is what
  chaining cannot take away without the embedding table on the device.
- **INT8 activations.** A decode or verification row's projections (Q4_K, Q5_K, Q6_K, IQ4_XS and
  Q8_0) take the Q8 kernels: each 32 values of a row's input are quantized to INT8 with one F32
  scale, staged once per block, and a unit's dots are exact in INT32 before scaling. A layer's
  qkv, gate, beta and alpha stay one launch: a block owns one matrix, and the BF16 ones read the
  F32 input. `STRAYDOG_Q8=0` keeps every projection on F32.
- **HyperConnections in Q8_0.** The HyperConnections' down and up projections (BF16 in the bundle,
  1.26 GB of the 5.2 GB a row reads) go up quantized to Q8_0 at load, as llama.cpp quantizes:
  each 32 values' scale max |x| / 127. A decode pass reads half their bytes (plain decode
  64.5 to 67.3 tok/s; a 1-row pass 16.8 to 16.2 ms at 20K, a 4-row one 27.2 to 26.5 ms), greedy
  tokens part from the F32 run about as early as before, and 0.6 GB of VRAM comes free.
  `STRAYDOG_HC_Q8=0` keeps them BF16.
- **Prefill batch.** On two R9700s the resident Qwen3.8 reads a prompt in batches of 2048:
  4096 rows took the first token of a 20K and a 60K prompt 7-12% longer (5.7 to 6.5 s, 15.8 to
  17.2 s), so the derived batch for the resident models is 2048 where it was 4096.
- **Fewer, fuller launches in a pass** (from a kernel trace at 20K). Every Q8 unit of Q4_K, Q6_K and IQ4_XS is
  staged with Q5_K's pad: a row's lanes read its units 128 or 64 bytes apart, on the same LDS banks, so four
  vectors took twice one's time (a 2560 x 6144 Q6_K out projection 50 to 29 us at four rows). The shared
  expert's mix goes up as the BF16 row it is, one launch with the router. The rows to the helper and its
  flag are one single-block kernel. A partial acceptance replays every DeltaNet layer of a span in one
  launch (a launch a layer was ~330 us ahead of the drafting). The drafter's dense projections but its
  router (BF16, a fifth of a draft step) go up as Q8_0, as the HyperConnections do; only the drafts
  depend on them. A pass at 20K: 1 row 16.1 to 15.4 ms, 4 rows 26.5 to 24.3.

## 9. Qwen3.6-35B-A3B (`qwen35moe`)

**The architecture.** One residual stream runs through blocks whose mixer is a Gated DeltaNet
(three of every four) or gated full attention. Each mixer is followed by a softmax top-8 MoE with
a sigmoid-gated shared expert. Its definitions follow llama.cpp's `qwen35moe` at ead9f32 over the
same GGUF, checked layer by layer (examples/q35_reference on the CPU, examples/q35_chat on the
GPU).
- **DeltaNet.**
  1. `attn_qkv` gives [q | k | v], convolved per channel over the last `ssm.conv_kernel` inputs
     and passed through SiLU.
  2. q and k are L2-normalized per head (q also scaled by 1/sqrt(dim)). Value head h reads key
     head h mod (key heads).
  3. The state decays by exp(`ssm_a` · softplus(alpha + `ssm_dt`)), takes the delta rule with
     sigmoid(beta), and is read by q.
  4. The output is RMS-normed per head and gated by SiLU of `attn_gate`, then `ssm_out`.
- **Attention.**
  1. `attn_q` gives each head's query, then its gate.
  2. q and k are RMS-normed per head, and the leading `rope.dimension_count` values rotate at the
     text position.
  3. Causal grouped-query attention, times sigmoid of the gate, then `attn_output`.
- **MoE.**
  - The top `expert_used_count` experts, weighted by their normalized probabilities.
  - The shared expert's output, scaled by sigmoid(`ffn_gate_inp_shexp` · x).
  - The MTP block after the main layers is not run.

**The bundle.** It is read on demand: the GGUF pinned by size, SHA-256 and fs-verity digest, plus
the official tokenizer, chat template and generation config pinned as resources. Unsloth's GGUF
embeds a modified template, which is not used. Replies split as Qwen's do (`</think>`,
`<tool_call>`) and end at the generation config's end tokens.

**Continuing a conversation.** Qwen3.6's template drops a past turn's thinking (unless
`preserve_thinking`), so its next prompt never extends the sequence that ran.
- The DeltaNets' state (conv histories and recurrent states, a few MB a layer) is copied on the
  GPU where each prompt's generation prompt begins, keeping the last two.
- A prompt that extends such a checkpoint, while the held sequence matches it that far, restores
  it and runs only the rest (`resumed_from` "turn" or "prior").

**Two GPUs that each hold half of every layer with all its experts** run it tensor parallel
(`arch/qwen35moe/part.rs`).
- **The split.** Each GPU runs every layer with its half of:
  - the attention heads and KV heads;
  - the DeltaNet key heads, with the value heads that read them;
  - the shared expert's rows and every routed expert's rows;
  - the vocabulary.

  Both keep the whole residual stream, the norms and the router, so they route alike.
- **The sums.** Partial sums are added on the devices twice a layer (`gpu/device_link.hpp`).
  - Each GPU copies its part into the other's inbox, raises the other's flag and waits for its
    own.
  - Both add the same two values, so their streams stay equal.
  - The counts live in device memory, so a decode step stays one recorded graph per GPU.
- A model that needs the host tier stays split by layers.

**Where its experts go:**
- **All of a GPU's experts fit in its VRAM.** They are read in at placement and kept for good.
  - A pass looks each route's slot up on the GPU (`sd_op_experts_table`), so no routes come back
    to the host, and a decode step runs as one recorded graph per GPU.
  - A pass of 16 rows or more groups its routes by expert on the device and runs them as RDNA4
    grouped GEMMs (`gpu/moe_prefill.hpp`).
- **Otherwise,** the expert store serves them, and routes to experts outside VRAM run on the CPU
  from the host tier. Only activations cross the bus.
  - A decode step is queued whole as one recorded graph per GPU (`arch/qwen35moe/engine/hybrid.rs`).
  - Each layer exports its routes and MoE input to page-locked memory, runs its shared and VRAM
    experts, then waits on the GPU for the host's sum of the other routes and adds it.
  - The host serves the layers in order. So the CPU's experts overlap the GPU's, with no host
    launch or read-back between layers.
  - Experts the CPU ran often (a count of their routes decaying 5% a step) move into VRAM between
    steps, up to 32 a step into free slots and up to four into slots clearly weaker experts give
    up.

## 10. Command line, configuration and setup

With no subcommand, `straydog` is the server, as llama-server is. Every setting is an option,
named as llama-server names it where it has the same setting.

| Option | Meaning | Default |
| --- | --- | --- |
| `-m, --model DIR` | model bundle directory, or its first shard | the settings `setup` wrote |
| `--quant NAME` | the bundle, when the directory alone does not say | detected from the pinned files |
| `-md, --model-draft DIR` | Qwen3.8's MTP drafter | none |
| `--draft-max N` | most MTP drafts per pass (0 turns MTP off) | 3 with a drafter |
| `-c, --ctx-size N` | context in tokens, at most 262144 | 262144 |
| `-b, --batch-size N` | prefill batch, 1-4096; DeepSeek-V4.1: up to 8192 | 1024; 4096 with 48 GiB or more of GPU budget, and for Qwen3.5 MoE; DeepSeek-V4.1: 8192 (§7.2) |
| `-gpu, --gpu LIST` | GPU ordinals (`-dev`, `--device` also accepted) | `0` |
| `-sm, --split-mode MODE` | `experts`: every layer split across the GPUs; `layer`: ranges of layers | `experts` |
| `--vram GIB[,GIB...]\|auto` | VRAM budget for every GPU, or one per GPU (`30,10`) | `auto` (§5) |
| `--state-ram GIB` | host memory for sequence state | what the context needs |
| `--row-cache MIB` | row cache of the PLE and Engram tables; 0 reads them through the page cache | 256; Qwen3.8's PLE table: as the host memory allows (§4.2) |
| `--expert-ram GIB` | host tier for experts outside VRAM | 85% of host memory less Straydog's other host memory |
| `--data-dir DIR` | verity trees and warm-up state | `.straydog` in the model directory |
| `--gemm fp32\|fp16`, `--gemm-min-batch N` | precision of large products, smallest batch that uses GEMM | fp16, 2 |
| `--host`, `--port` | listen address | 127.0.0.1, 8080 |
| `--api-key KEY`, `--api-key-file FILE` | a Bearer key the server then requires | none |
| `--alias NAME` | model id in `/v1/models` and answers | `<model>-<quant>` |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, `--seed` | sampling when a request gives none | the model card's for the request's mode |
| `--reasoning-effort E` | none, low, medium, high | high |
| `--startup-timeout S`, `--log-file FILE`, `-v` | operations | 180 s, stderr |
| `--config FILE`, `--print-config` | settings from a TOML file; show the settings in effect and where each came from | |

**Precedence.** Options, then `STRAYDOG_*` environment variables, then the config file, then
defaults. One `clap` definition gives the options and the variables, and the file uses the same
names.

The `DS41_*` variables are development controls for DeepSeek-V4.1 (`DS41_DSPARK`,
`DS41_DRAFTS`, `DS41_DSPARK_RESIDENT`, `DS41_PROFILE`). They pass into the worker; nothing else
in the environment does. `DS41_PROFILE` also makes Qwen3.8's request report its decode in phases
(milliseconds a token, and a pass's by its rows: drafting, the first layer, the PLE rows, each
device's layers, the head, the commit).

**Subcommands.**
- `straydog setup`: the first run.
- `straydog fetch -m DIR --quant NAME`: download, resume and authenticate a bundle, with its
  extras.
- `straydog verify`: authenticate a bundle and build its verity trees.
- `straydog fetch-mtp`: build Qwen3.8's MTP drafter.

`fetch` and `verify` build each shard's verity tree into the data directory and keep it only when
its digest is the pinned one. The server builds the trees it lacks the same way before the worker
starts. When the data directory cannot be written, the start fails, naming `--data-dir`.

**`setup`** asks a few questions on a terminal; every answer is also an option, for scripts.
1. It looks at the machine: each GPU's name and memory, the host memory the server measures, the
   free space where the models would go.
2. It lists the pinned bundles that can run there, with their download sizes and what each
   needs, and marks one recommended:
   - The fit is the placement's own accounting over each bundle's footprint
     (`asset_footprint.json`, from the shards' headers). That covers the dense weights, the
     routed experts, the tables read by row, the state a token of context takes, and each
     model's measured working memory (`setup/catalog.rs`).
   - A model gets the largest of 262144, 131072, 65536, 32768 and 16384 tokens that fits. Setup
     writes `ctx_size` when that is less than the default.
   - The first model, in the order DeepSeek-V4.1, Qwen3.8, Qwen3.6, with a bundle that VRAM and
     RAM hold alone is recommended, with its best bundle in that class. Failing that, the bundle
     that reads least from the drive is.
3. It asks where the models go. The default is the platform's data folder:
   `$XDG_DATA_HOME/straydog/models` or `%LOCALAPPDATA%\Straydog\models`. The bundle goes in
   `<dir>/<model>/<quant>`.
4. It checks the free space and asks which extras to add: DSpark and the vision encoder for
   DeepSeek-V4.1, MTP for Qwen3.8.
5. It fetches as `fetch` does.
6. It writes `config.toml` in the platform's config folder. The file is written whole and
   renamed, keeping the earlier one as `config.toml.old`.

A plain `straydog` reads that file when no `--config` and no model are given. A server started
with a model of its own reads no file it was not given.

## 11. API

- **Endpoints.**
  - `GET /health` (no key needed): 200 when ready, 503 while loading (with progress), 500 after a
    fatal engine error.
  - `GET /v1/models`, and llama-server's `GET /props` and `GET /slots` (the context as `n_ctx`).
- **`POST /v1/chat/completions`,** streaming (SSE `chat.completion.chunk`) or not:
  - messages: system (and developer), user, assistant (content, `reasoning_content`,
    `tool_calls`) and tool;
  - content as a string or text parts. For a model that reads images (`/props`
    `modalities.vision`), user and tool content may also hold `image_url` with a `data:` URL, or
    `image` with a base64 source. The server fetches nothing;
  - `tools`, `tool_choice` (`auto`, `none`, `required`, or a function named);
  - `response_format`: `text`, `json_object`, or `json_schema` (the structured-outputs subset;
    other keywords accepted, not enforced);
  - `max_tokens` / `max_completion_tokens`, `temperature`, `top_k`, `top_p`, `min_p`,
    `presence_penalty`, `seed`, `stop`;
  - `reasoning_effort` (`none` to `xhigh`) and `chat_template_kwargs`;
  - `stream_options.include_usage`.

  Answers carry `reasoning_content` deltas, structured `tool_calls`, and `usage` with
  `prompt_tokens_details.cached_tokens`, plus the engine's `timings`. Sampling controls the
  engine does not implement are accepted only at their neutral value; anything else is a 400.
- **Turns.** A request that ends with a tool message continues the assistant's reply; one that
  ends with a user message starts a new one. That is all the cache needs from a client.
- **Errors** are OpenAI-style objects. Before a stream's first byte, the status says it (400,
  401, 503 loading or busy, 500). After it, an `error` event ends the stream.
- **Security.**
  - The server listens on 127.0.0.1 by default.
  - An API key is optional, and required once given.
  - One engine serializes generation behind a bounded queue, and a request whose client
    disconnects is cancelled.
  - Every client that holds the key shares the engine and its cache, so whether a prompt prefix
    was seen before is observable. Users who must not see each other's prompts need separate
    servers.
- **Dashboard.** `GET /dashboard` is a page compiled into the executable, with a strict CSP and no
  other origin.
  - It reads `GET /dashboard/status?since=T`: settings, engine state and placement, the running
    and the latest 120 generations with their speeds and speculative acceptance, and host,
    process and GPU sensors.
  - It reads `GET /dashboard/logs?after=N`, and chats through `/v1/chat/completions`.
  - Sensors are read only while a dashboard is open: amdgpu's sysfs, or NVML in CUDA builds.

## 12. Build and distribution

- `cargo build --release --features hip` (or `cuda`). `build.rs` runs CMake on `gpu/` with the
  vendor's compiler and links the result statically. The architectures come from
  `STRAYDOG_GPU_ARCHITECTURES` (`gfx1201`; `86-real;89-real;120`).
- A GPU without code in the executable fails at startup, with its architecture named.
- Release artifacts: `straydog-linux-x86_64-hip` (needs the ROCm 10 runtime) and
  `straydog-windows-x86_64-cuda.exe` (needs the NVIDIA driver and `cublas64_12.dll`). The
  executable's SHA-256 is published with each release. The model pins are compiled in, so a
  binary authenticates exactly the bundles it was tested with.

## 13. Decisions

- **Rust host, C++ kernels.** The host is Rust; the kernels stay C++/HIP/CUDA behind a C ABI.
  HTTP is `axum` on `tokio`.
- **Pinned bundles only.** There is no arbitrary-GGUF mode: a model is one of the pinned bundles,
  authenticated before use.
- **Dense weights must fit the GPU budget.** Layers are never computed in host memory; a budget
  below the dense weights is an error. Partial residency covers routed experts only.
- **The model card's sampling by default,** not greedy decoding, which loops on some models.
- **What Straydog makes for a model is kept with it** (`<model>/.straydog`) unless `--data-dir`
  says elsewhere: the verity trees and the warm-up state. The trees are as trustworthy anywhere,
  being checked against the compiled-in digests.
- **The settings file `setup` writes is read only when nothing names a model.** Homelab, and any
  script that names its model, is unaffected.
- **DeepSeek-V4.1's misses are uploaded, not computed on the CPU,** and the store is LRU (§5).
  Both were measured against the alternatives.
- **DeepSeek-V4.1's template is Straydog's own** (§7.1): a template that misrenders tool calls
  costs every turn a full prompt.
