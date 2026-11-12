# Straydog — design

Straydog is a local inference engine and a server with an OpenAI-compatible API, shipped as one
executable per GPU backend (HIP on Linux, CUDA on Windows and on Linux). It serves pinned GGUF
bundles of two models:

- **DeepSeek-V4.1-Flash** (748B total, 16B active), the main one;
- **Qwen3.8-Flash-Next**, with its MTP drafter, also the model for a small GPU (5-6 GB of VRAM)
  and 32 GB of RAM.

This document describes what the code does now. It does not keep the project's history; git
does. The roundings are listed in the README's Numerics, which this document does not repeat.

Updated: 2026-11-08. Both models run on Linux with ROCm. Qwen3.8 and `straydog setup` have run on
Windows with CUDA; DeepSeek-V4.1 has not. Both models have run on Linux with CUDA under WSL2, isolated
by bubblewrap, and not yet on native Linux.

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
    compiled into the executable, and every file is authenticated whole when it is downloaded. A later start
    trusts a file whose verity tree is younger than it, and reads what it places without hashing it again.
  - Bytes read from the drive after loading (table rows, experts outside RAM, banks left to the
    page cache) are checked block by block against the file's Merkle tree before use.
  - The trees live in the data directory (`.straydog` in the model's by default). Only the
    server writes them, never the worker, and they are checked against the compiled-in digests
    whenever they are opened.
  - The drive is read-only to the engine: no swap, weight or K/V files.
    - The worker never writes.
    - On Windows the runtime writes two things derived from conversations: the state of the system
      block (its tools included), sealed for the user (DPAPI and AES-256-GCM), at most once per
      run and two per bundle (section 6.4); and, only when the person turns on
      `keep_conversation`, the state of the conversation held last, sealed the same way, once, at a
      stop on purpose, one file a bundle (section 6.5).
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
- a short list of environment variables: the system's and the GPU runtime's, and the engine's switches (the `DS41_*` controls
  named below, `STRAYDOG_ENGRAM`, `STRAYDOG_EXPERT_FLOOR`, `STRAYDOG_CPU_INT8`, `STRAYDOG_CPU_AVX512`, `STRAYDOG_PROMPT_CPU`, `STRAYDOG_HOST_REGISTER_READ_ONLY`, `STRAYDOG_DIRECT`), one list for bubblewrap and
  the AppContainer (`straydog_isolation::ENGINE_VARIABLES`).

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
4. **Architectures** (`arch/deepseek41`, the Qwen3.8 engine). An architecture
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
  - On Windows the I/O manager runs the reads issued on a synchronous handle one at a time, however many
    threads issue them. The handles that threads read through at once (the direct ones, and a
    second one per file for the page cache) are opened for overlapped I/O, and each read waits for
    itself alone.
- **A model read on demand** (DeepSeek-V4.1) opens its shards against their verity
  trees. The routed experts are not read whole at the start: the expert store fills from the
  files as tokens route to them, and the drive blocks they come from are checked against the
  trees.
- **Startup times.** DeepSeek-V4.1's server is ready in 14 s with its dense weights in the page
  cache. Qwen3.8's IQ3_S with its drafter takes 22-23 s from a cold start and 13-14 s cached.
- **Behind "ready"** (Qwen3.8, the worker `behind.rs`). The app starts the engine when it opens and its person then types for a while, so
  two steps of a partially resident start come after ready and give way to a request. The ready frame's `plans` and `warm` are what they cost
  before it; the log says what came after.
  - **The GPU code's warm-up** (two greedy runs of a made-up prompt, 10.6 s on a machine with nothing locked and cold banks) holds the engine for
    its passes. A request, or any command that takes the engine, ends it at the next pass boundary: the model is reset as at its end and
    the request's first pass loads what it needs, as after any reset. The longest it can keep a request waiting is its longest pass: the
    first, which loads the GPU code and streams the banks cold, took 5.8 s here. A warm start asked for meanwhile begins, and the locking below is
    planned, under the lock the warm-up holds as it ends, so that no request has the engine first. One line: done or cut short, its seconds, and
    seconds after ready.
  - **The banks' page locks.** The start plans which whole layers to lock (from the last, within the lock budget) at the point it always did, before
    the expert slots are placed, so what it leaves of the commit for the locks is out of what the slots' VRAM takes; it locks none.
    After the warm-up a thread locks the planned layers one at a time (25.9 GiB of 40 layers took 7.5 s of an 18 s start on an RTX 4060 PC,
    nearly all of it reading the banks from the drive), the layers whose experts the host ran most routes of in the warm-up state first, else
    from the last layer. A layer's turn holds nothing of the engine: it page-locks the three banks, then checks them against the file's tree
    (a locked page cannot change under the check; an unlocked bank's pages may be read from the drive again after the load verified them), and
    a layer that cannot be locked ends the locking where it is, its reason in the log, the layers before it kept. Only then does it take the
    engine's lock, for the moment it takes to register the layers it has locked with the device and map their banks for the GPU's assist (the
    one way the GPU reads a bank in place). A request never waits for a layer's turn, and a turn is not begun while a request has the engine: the
    reads, the hashing and the registration take the cores and the memory bandwidth that a generating request's CPU experts need (on the Linux
    machine, 17.6 s of locking while a request generated took it from 47.6 to 29.8 tok/s, its CPU from 9.7 to 17.9 ms a token). So before each
    turn the thread looks every 5 ms for a request that has the engine, and waits while there is one, for 20 s at the most since its last layer;
    after that it takes the turn as it did before it waited at all. The gap between the requests of an agent (a tool runs) holds a turn, so the wait
    is for the engine to be free, not idle (a locking that waited for the engine to be idle never finished under requests that follow one another).
    A turn that has begun is not stopped. The thread and the threads that check the banks work at a lower priority (Windows: the thread's
    background mode, for the turn only; Linux: nice 19 and the idle I/O class, kept), and the check uses at most a quarter of the cores (one at
    least) where the load's check uses all. The layers locked while a request had the engine are registered together at the first moment the
    engine is free after a turn, or when the last is locked, which waits for a request that has it. A layer
    whose bytes do not match the tree is unlocked, never registered, and every later request fails with the error. The assist and its
    tuner turn on with the first layer, take each one as it comes (the layers' captured graphs are dropped and made again), and time their levels
    anew with each. The working set's limit comes when the locking ends, as it did after the locks. One line: layers, GiB, seconds, seconds
    after ready, whether a request ran meanwhile, how long the turns waited for requests in all and whether the 20 s guard fired, and why it
    stopped if it did.
  - **Copies where the GPU cannot lock the mapping.** A GPU that cannot register host memory read-only (CUDA's
    `cudaDevAttrHostRegisterReadOnlySupported` 0: two RTX 3090s on Windows 10, whose every layer's lock failed, so that none of 48 was ever locked and
    the slots and the prompt read pageable memory) cannot page-lock the model's read-only mapping in place. The start asks each GPU, or takes
    `STRAYDOG_HOST_REGISTER_READ_ONLY=0` as the answer (a test of this on any machine), says in one line which and why, and from then on every plan is
    in the copy mode (`LockMode::Copy`), the mapping never offered to the runtime again. A layer's turn then allocates ONE page-locked copy of each of
    its banks (`hipHostMalloc`, portable and mapped, so every device reads the one copy), copies the bank's bytes from the mapping into it, and
    checks THE COPY against the file's tree, over the whole blocks around the bank at its file offset (`verity::verify_copy_at`): a page-locked copy
    cannot change after the check, so the bytes the GPU and the CPU's experts read are the bytes that were checked, which keeps the rule of section 2.
    A copy that does not match is freed and ends the locking, and the model's service, as a bank that does not match does. Under the engine's lock the
    plans' host pointers are moved to the copies (`sd_hybrid_repoint`; the one place a bank's bytes are read from: the slots' copies, the prompt's
    streaming, the CPU experts' gather and routes, the assist's mapping, which then maps the copy), and the layer counts as locked, in the budget,
    the assist and the give-back. No read-ahead hint is made for a layer held as copies. A give-back moves the plans back to the mapping first, which waits for the
    device's copy thread and streams and for the CPU's workers, and only then lets the copy go. What makes that safe is the engine's lock (every CPU
    read of a bank is a synchronous call made under it, so none runs during a give-back) and the device's waits, `bank_readers` among them, which is an
    error if it does not drain. The model's end does not point the plans back: `Model` has no `Drop`, and its fields drop in their order, the devices
    (which drain their copy thread and destroy their plans) and the CPU engine before the copies, which is why the copies are declared after them. A
    failed give-back keeps the copy; a take-over that failed frees it at once if every device is back on the mapping, and keeps it until the model
    ends if some device could not be put back. The assist maps a copy by the start of its allocation (`host_base`), since a bank begins inside the
    whole blocks that the copy holds and the runtime maps an allocation by its start; a copy it cannot map is a locked layer that assists nothing, and
    the log says so (`WARN Straydog host banks: layer N's page-locked copy could not be mapped`). A turn refused with nothing locked is kept by the
    model (`Refusal`) until the memory available has grown by a layer's banks since, so that no plan of the remaining layers is made after each
    request to fail at its first layer again.
  - **The read-ahead of the banks to lock.** A layer's turn reads its banks itself, so a request that comes before the locking is done finds the
    layers not locked yet out of memory when the page cache is cold, and reads them by faulting, one fault after another (Linux, 125 GB, UD-IQ1_S: the
    first request at 28 to 31 tok/s with 218,547 page faults, against 49.4 after the locking; Windows, 62 GiB with the cache emptied by a 46 GiB read: 33.2
    and 38.6 tok/s, the CPU at 18.5 and 15.4 ms a token). So a thread of its own, started when the plan is made, asks the system for the pages of the planned
    banks in the order the layers are locked (`BankLocks::read_ahead`, `behind::read_ahead_behind`): pieces of at most 256 MiB of one bank, each by
    `host::prefetch` (`PrefetchVirtualMemory`, `MADV_WILLNEED`), so that it can be followed and stopped between them. `PrefetchVirtualMemory` returns when the
    pages are in; `madvise` returns once the reads are queued, so there (Linux) no more than two pieces are in flight: before a piece is asked for, the one
    two back is waited for to be in the page cache (by `mincore`, for 5 s at the most), which keeps a request's own reads from queuing behind gigabytes
    of ours and a stop from coming late. It reads the plan's bytes and no more,
    leaves out the pieces of layers the locking has done and those whose pages are all resident (Linux: the page cache, by `mincore`; Windows: the
    working set, a lower bound, since a page in the standby list is not visible), and works below a request: the thread's background mode (Windows: CPU
    and I/O priority; Linux: nice 19 and the idle I/O class). On Windows it keeps its memory priority at 5 (`Lowered::begin_reading`), because background
    mode lowers that to 1 as well, and a page that is read in waits in the standby list at the priority of the thread that read it: at 1 it is the first
    the system takes back for the next page, the prefetch's own included (12 GiB prefetched in background mode with the standby list full left none
    of it in memory, 13 GiB read again by the first touch in 28 s; at memory priority 5 the touch took 2.9 s and read 412 pages; the prefetch ran at 2.4
    GiB/s in background mode against 5.4 normally, on an idle drive). It ends when the plan is done, when the locking is over, when the engine has
    ended, when less than the memory the plan keeps free (`host::left_free`) is available, or when the system refuses; one line says the bytes asked for,
    the seconds, how far ahead of the locking it stood at the end and at the most, and why it stopped if it did. On the Windows machine, the first request
    after the cache was emptied ran at 46.9 and 43.6 tok/s (CPU 10.6 and 11.9 ms) against 33.2 and 38.6, and the locking was done 60 s after ready
    against 71 to 87 s; with the cache warm, 46.0 against 44.1 tok/s, and the read-ahead of 11 GiB took 1.5 s.
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
- **A partially resident model** (Qwen3.8's expert slots in VRAM, the rest of its routed experts
  computed by the CPU from the mapped banks) keeps route counts instead, in the same directory and
  saved at the same moments (`<model>-<provider>-<quant>-<key>.json`: the key hashes the pinned
  shards, with a domain of its own, so that Unsloth's UD-IQ1_M of the two models never meets). The
  first request after a start is its slowest: the VRAM slots hold what the start-up's own passes
  left, and the banks' pages are not in memory.
  - **`counts`** (`routes.rs`) are `[layer, expert, GPU routes, host routes]` for each of the 48
    layers and the drafter's, by where the route went: to an expert in a slot, or to one the CPU
    computed (or the GPU read over the bus). Decode and verification rows are counted, once in a
    layer's native plan (`hybrid_host_experts`: a counter array, nothing allocated per token); a
    prompt's are not, as it streams nearly every expert. The earlier counts age by the rule of the
    expert stores' (`arch::warm::kept_share`).
  - **A start with a state of its bundle** (kind, format and geometry are checked; any other is
    refused and the start is as it was) sets the slots' scores from the counts at once, then does
    the rest in the background, after ready, with no request waiting for more than a step.
    - The scores a layer's experts are ranked by (the moving average of its routes) are set to what
      it comes to at the counted rate (`sd_hybrid_seed`), so a request's own rebalance already ranks
      with them, over the start-up's leftovers; a prompt of a few dozen tokens then moves little.
    - The OS is asked for the bank pages of the likeliest experts of layers that are not
      page-locked, up to the physical memory available less `host::left_free`, in requests of
      512 MiB with each request's ranges joined and in address order (`host::prefetch`:
      `PrefetchVirtualMemory` on Windows, `madvise(WILLNEED)` in 128 KiB steps on Linux). It begins
      at the likeliest expert, the slots' too: a slot is copied from its bank's pages, and a
      prompt streams every expert the slots do not hold yet. Nothing is locked and no commit is
      taken: the pages wait in memory (Windows' standby list, outside the working set) for a soft
      fault, which the page-fault count of a request still counts.
    - The slots fill, 64 swaps a step under the engine's lock with a pause between, by the same
      rebalance a request's own uses, until they hold what the scores want. A request that begins
      ends it: its routes and its own rebalance take the slots from there.
    - One line in the log: slots filled, MB asked of the OS, seconds.
  - **What it gains** (Qwen3.8 UD-IQ1_M on an RTX 5070 Ti with 61.6 GiB, no bank page-locked as the
    commit left allowed none, the OS's file cache pushed out before each start, a state from an
    earlier run of the same three questions, one run each): the first reply asked 12 s after ready
    took its first token in 2.8 s against 4.0, prefilled 36.9 tok/s against 26.1 and decoded 38.5
    against 34.8; the warm start took 1.8 s (3339 slots, 2762 swapped in; 12.2 GB asked of the OS).
    Asked within a second of ready, the first reply gains little (34.6 tok/s against 32.1-34.8, the
    first token at 4.06 s as without): its prompt streams 8,400 experts, about 11 GB, from the drive
    at the drive's speed, the same pages the hints read. A request's page-fault count does not
    fall with the hints: a page the OS read ahead is a soft fault at its first touch, and the
    count has those too.

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
- **Windows.** Unbuffered reads of a file that has a data section (it is mapped, or has pages in
  the system cache) are each preceded by a flush of their range that NTFS runs one at a time on
  the file: 128 threads read 4 KiB blocks at the rate of one, 11K a second where the drive does
  650K, and a 2K-token prompt waited 1 to 1.8 s for its PLE rows. The table's file holds that
  table alone, so `Weights::cache_rows` lets go of it before the table reads: it drops the
  mapping and the handle, opens the file for unbuffered reads alone (which, with nothing else
  open on it, makes NTFS drop its cached pages), and opens it again. The CPU engine keeps no
  handle on it. The same wait is 30 to 60 ms. A row table must not map its file or read it through
  the cache while it runs.
- **Verification.** Each block is checked against the shard's fs-verity tree between its read
  and its use. A block that fails fails its read; a later use reads it again.
- **Memory pressure.** DeepSeek-V4.1 decode waits 0.1-0.4 ms a token on Engram rows. If the
  page cache loses the tables' blocks to other work on the machine, that wait rises to ~3 ms a
  token: on a 125 GB host with a 102 GiB pinned expert tier, a large build beside the server
  does it.

## 5. Placement and the expert store

There is no `-ngl`. Placement measures what fits, in this order:

1. **Full residency.** Every weight in VRAM, the K/V too (Qwen3.8: at the prefill batch derived, 4096 from 48 GiB of
   budgets, then at 2048; a batch given is not narrowed to keep the K/V in VRAM).
2. **Host K/V,** then a smaller prefill batch (Qwen3.8), with the MTP drafter, then without.
   The first choice that fits is taken; when it is not the first, the placement notes what the first
   would have needed, and with partial residency what full residency would (`resident_placement`),
   and the ready line says so: a few hundred MB decide the batch and the K/V's place.
3. **Partial residency.** Dense weights and sequence state in VRAM; routed experts in an expert
   store of VRAM slots over a page-locked host tier over the files.

Each GPU's budget is its free memory less max(512 MiB, 3%) (`--vram auto`), or what `--vram`
gives. Two GPUs of similar budgets split every layer between them (§7.1, §8); GPUs whose
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
  - For DeepSeek-V4.1 the host tier takes what VRAM evicts, and keeps its copy of an expert it
    hands to VRAM while it has room, so an expert leaving VRAM then costs no write-back (beside a
    pass's miss uploads those write-backs took 2.8 of a 3-row pass's 30 ms). When it needs room it
    gives up first the copies VRAM also holds, so it still holds everything it alone held.
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
  - Checkpoint copies are made while the engine is idle, not before the first token. A new
    conversation finishes the copies still pending first (a 20K conversation's: ~0.1-0.2 s at the
    8-12 GB/s a large read-back takes through both halves of the staging arena).
  - A turn that resumes from a checkpoint the held sequence still extends (a client that sent the
    last reply back without its reasoning) goes back in place (`Engine::rewind_to`): only the
    window's rows and the incomplete groups' tails go back, since a later position never writes a
    complete group's compressed row or index key. Its copy may go on later; the copies of
    checkpoints above it are made first.
  - The checkpoints take at most 2 GiB of host memory, or room for two at the full context when that is
    more: a checkpoint is ~0.85 GB at 262144 tokens and ~3.4 GB at 1048576, so a 1048576-token context
    gets ~6.3 GiB. A checkpoint over the budget is not saved.
- **A void pass** (DeepSeek-V4.1: a part link's wait gave up after 10 s, as when the host stalled)
  voids that pass, not the reply: once, the sequence starts over from the checkpoints with the
  tokens already sent as part of its prompt, and the reply goes on.
- **Qwen3.8** parks a sequence when a prompt departs from it at least 1024 tokens below its end:
  - Its K/V and indexer positions from the departure on go to host memory (up to an eighth of
    it), and its recurrent state to a page-locked slot.
  - A later prompt that resumes further on the parked sequence than on the held one brings it
    back (`parked`: `saved` or `restored`, with `park_seconds`).
  - The copy into host memory is made by the CPU's threads: a K/V in host memory straight from
    its page-locked memory, VRAM's indexer blocks through the page-locked arena (as a large
    download), where the runtime's own copy into pageable memory pins its destination and ran
    at 2 GB/s. A 242K read after a 95K conversation copies its 94027 positions (1.66 GB over
    two R9700s) before its first batch: 0.79 s that way (`park_seconds`, "parking" on the
    request line), 0.20 s now, the bytes the same (every part compared with the runtime's copy).
  - So a one-off request with another prefix (a title, a summary, a subagent) does not cost the
    conversation it interrupted its whole prompt again.
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
  the drafts' probabilities under the drafter stays at 0.6 or more. Partially resident, how many
  drafts a pass takes (0 to `--draft-max`, at most three) is the engine's to choose, by what its
  passes cost this machine (`width.rs`): a pass reads the layers' dense weights once whatever its
  rows, so where the GPU's part of a layer is the longer a second and third row cost little and
  drafts pay, and where the CPU's experts are it each row doubles their routes and one draft gains
  next to nothing. It starts with one draft (two on HIP when the slots hold at least 15% of the
  routed experts at load, where that was measured) and keeps, as windows of the latest passes, the
  seconds of a pass by its rows, the share of passes whose chain of drafts reached each position
  (the drawn-draft cut ends a chain where the drafter is unsure, whatever the width) and the share
  of drafts offered at it that were kept. From those the seconds a token takes at each width
  follow: the cost of the rows a pass of that width has, weighted by how often, over one plus the
  drafts kept; with no cut, the cost of k+1 rows over one plus the shares kept at the first k
  positions. A width replaces the one in force only if 3% faster and by more than two and a half
  times the error of the difference (the assist level asks for 2: passes vary by a quarter, and
  here a decision comes at every probe). The widths not in force are probed in turn, a stretch of
  24 passes at each with the cut in force as in any pass, so that their rows and positions stay
  timed: 3% of the passes, and none before the 256th of a run, when a machine's memory has filled.
  The drawn-draft cut of a sampled request weighs each draft's row by what that row costs. The log
  says each width's milliseconds a token, the shares kept and the passes' costs, at the first
  choice and at each change, and the request line the drafts in force.
- **DeepSeek-V4.1's DSpark** (§7.4). It needs no option: it is used when its file sits beside the
  bundle.
- **An n-gram lookup of the sequence's own text** (`drafts.rs`), for both models. A match of eight
  tokens or more (an edit repeating its source, a tool's output quoted back) is verified as it is,
  without asking a drafter. A resident Qwen3.8 verifies up to eight rows a pass for such copies
  (three for its MTP drafts): a long match is checked seven tokens at a time.

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
- **Qwen's tool calls read back** (`reply.rs`). Well-formed markup is read as the template writes it. Otherwise
  `ToolCallSplitter::repair` keeps what the model meant, since calls left in the answer end an agent's run: prose
  after or between calls goes to the answer, a close left out is supplied, an unreadable block among others is dropped,
  and a block of JSON (`{"name", "arguments"}`) is a call. A value may hold `</parameter>`: it ends at the first one the
  markup goes on from. A closing tag left outside a block, or a block open where another opens, may be a value quoting
  markup: the whole is released as text, visibly, not run cut short. A reply a token limit or stop string ended is not
  repaired at all (it may stop inside a value quoting closes: a write of half a file must not run). The log counts repairs
  and text released.
- **Constrained decoding** (`constraint.rs`). After the reasoning (`</think>`):
  - a JSON `response_format` holds the answer to a JSON value the schema allows, racing a tool
    call's opening when tools may be called;
  - `tool_choice` `required`, or a named function, makes the answer start with the markup's
    tool-call opening (and the function's name), free after it.

  Each token is drawn from the model's distribution, and drawn again without a refused one
  (masked after 32), so allowed tokens keep their relative probabilities.
- **Suppression** (`constraint.rs`). A request's `suppress: "not_cp932_han"` refuses every token that would write a Han
  letter in U+3400-U+9FFF with no CP932 form (a simplified 员, 约, 叠 in a Japanese reply), reasoning and tool arguments
  included. 塡剝頰噓俱繫鷗醬 are real Japanese with no CP932 form, and stay.
  - The tokens come from a table read off the vocabulary by the first request that asks (24603 tokens of Qwen3.8's 248K).
    A token that only finishes a letter from the bytes the last one left open is refused by that state: a character can
    split across tokens.
  - A token is chosen as under a constraint. Alone a suppression keeps the drafts: it masks the rows they are verified on,
    so one it refuses has probability 0 under the target, and is rejected like any other; what comes out is the target's
    distribution without the refused tokens. A point proposal it refuses is not verified at all.
  - Qwen3.8's head otherwise gives a verification's rows as an argmax or a top k picked on the GPU, which cannot be masked:
    a request that suppresses reads the rows' whole logits instead. No `suppress`, nothing changes.

### 6.4 The system block's state across starts (Windows)

An app that starts the engine and then sends its first request sends the same system block every
time: its system prompt with its tools, a few thousand tokens. On a small card that block is most
of the first reply's wait. The root checkpoint (6.1) is the state at its end, so that state is saved
once and handed back at the next start.

Measured on an RTX 5070 Ti held to 5.5 GiB, UD-IQ1_S, a request of 4,048 tokens (4,020 of them the
root): 12.2 s to read it, first token at 12.35 s. At the next start, with the state handed back:
restored 0.9 s after `ready` (308 MB read, opened and imported), first token at 1.63 s, 28 tokens
read. The state was saved in 0.3 s after the reply, and the file is 323 MB.

- **What a state is.** What a parked sequence is (6.1), held at the root:
  - the root's tokens and the last token's hidden state;
  - every device's recurrent state, as the root checkpoint saved it;
  - the K/V and the indexer's pooled blocks of positions 0..root, which the held sequence still has
    as the root left them (the export is refused when it does not begin with the root's tokens);
  - with MTP, the MTP layer's indexer history, as the checkpoint kept it.
- **Export and import** (`Model::export_prefix`, `Model::import_prefix`; every platform).
  - The bytes are a header (`SDPREFIX`, the layout version, the tokens, the sections as a tag for
    what each is and its byte count), then the sections: the tokens, the hidden state, then each
    device's recurrent state, K/V and indexer blocks, and MTP history. A tag names the device and
    the state it is of.
  - An import checks the header against what its own model would give a state of that many tokens
    (every tag, every count, the length) before it reads a section, and refuses a state that
    differs by a byte. Nothing is changed then.
  - An accepted state becomes the parked sequence, covering 0..root. A prompt that begins with its
    tokens takes it back as a parked sequence is taken back (`resumed_from` is `prefix`, and the
    root checkpoint is saved again). A prompt that does not drops it.
- **The channel.** A frame that carries bytes is a JSON header line that gives their count in
  `bytes`, then exactly that many raw bytes (at most 4 GiB; a count over that is refused before a
  byte is read, a short body ends the worker). `prefix_state` (worker to server, asked for) and
  `prefix` (server to worker, after `ready`) are such frames. The worker never opens a file for them.
- **Saving** (server, Windows). After the reply of a request whose prompt made a new root (the done
  event's `root_checkpoint` and `root_digest`), at most once per run, and only if no file has that
  key: the server asks the worker for the state and seals and writes it on a thread of its own. The
  files are in `<data dir>/prefix`, `<bundle>.<build>.<prefix>.<key>.sdprefix`. A bundle keeps two:
  the third written deletes the one used longest ago, and a file of another engine build goes at
  once.
- **The key** is SHA-256 of:
  - the bundle (model, provider, quantization);
  - the engine's version, and its build (a hash of the executable);
  - what the worker placed (the digest of its layout, the context, the prefill batch, where the K/V
    is, how much is resident, the GPUs and their split, the drafts) and the settings that change
    the numbers (the GEMM precision, the expert floor, the engram table);
  - the root's tokens.

  All of these but the tokens make the key's prefix, which the start knows before any request. The
  VRAM budget a start was given is not in it, only what it placed: a card whose free memory moves a
  little keeps its file.
- **The file.** A random 256-bit key, wrapped with DPAPI (`CryptProtectData`, the user's scope, no
  UI) in the header; the state sealed with AES-256-GCM under it and a random 96-bit nonce, the
  header as the associated data.

  ```
  0    8   STRAYPFX         112  8   the state's bytes
  8    4   file version     120  4   the wrapped key's bytes, W
  12   4   state layout     124  W   the wrapped key
  16   32  key prefix       ..   12  nonce
  48   32  tokens' digest   ..       the state sealed, then the 16-byte tag
  80   32  key
  ```

  The ciphertext goes to a `.part` file, which is synced and renamed: nothing in plaintext is ever
  written, not even a part of a file. A file that does not unwrap, open or check (any byte of it
  changed, another user's, another build's) is deleted with one warning, and the start goes on
  without it.
- **Restoring.** After `ready` the server opens, on a thread of its own, the file of this build and
  key prefix that was used last (and marks it used), and hands the state to the worker. A request
  that arrives first is never held up: the server hands nothing over once a request has begun, the
  worker takes the state only while no request has had the engine, and it says in the log what
  became of it (`Straydog prefix: restored 4020 tokens (308 MB) in 0.9 s`).
- **Off.** `prefix_cache = false` in the config file, or `--no-prefix-cache`. Linux saves nothing.

### 6.5 The conversation's state across starts (Windows, an option, off by default)

The system block's state (6.4) spares a start the few thousand tokens that open every conversation.
A long conversation is the larger cost: after an engine restart (an app update, a change of
settings) the next request of a 36,169-token conversation read all of it again, 84 s on an RTX 4060
at 5.5 GiB (`prompt 36169 tokens (reused 0 from none)`). With `keep_conversation = true` or
`--keep-conversation` the conversation's state is sealed on the drive at a stop and taken back by
the request that goes on from it.

- **What is kept.** The state at a checkpoint (6.1): of the places a reply began, the deepest that
  the held sequence still begins with and that is at least 1024 tokens in (`turn`, `step`, `prior`).
  A request of the same conversation begins with those tokens whatever the reply was, which is why
  the live sequence's end is not what is kept: the reply is written back by the chat template, not
  as the model wrote it. The bytes are the system block's (6.4, `Model::export_conversation` is
  `export_prefix` of that checkpoint): the tokens, the last token's hidden state, every device's
  recurrent state, K/V and indexer blocks, and with MTP the MTP layer's indexer history. A
  conversation with an image in it (a token past the vocabulary) is not kept.
- **Import.** `Model::import_conversation` checks the bytes against the model's own layout to the
  byte, grows the K/V to hold the state and the request that goes on from it (the state is larger
  than the 32768 positions a partially resident model's K/V starts with, and a growth drops a parked
  sequence's copies), and installs the state as the parked sequence, marked as a kept conversation.
  The request that begins with its tokens takes it back as a parked sequence is taken back
  (`resumed_from` is `kept`), and the place its reply began is a checkpoint again.
- **Saving** (server, Windows). When the server stops on purpose, after the queue has ended and
  before the worker is told to shut down, the server asks the worker for the state, with the most
  bytes it has room for: the drive's free space of the data directory less 2 GiB, and the commit left
  less 1 GiB. A state over that is not given, and one line in the log says so. The server seals it and
  writes the file on a thread of its own, and the stop waits for it. A stop is: Ctrl+C, SIGTERM
  (Linux, where nothing is kept), the console closing, Windows ending the session (the console
  events `CTRL_CLOSE`, `CTRL_LOGOFF` and `CTRL_SHUTDOWN`, which tokio holds off the process's end
  for), or a line `stop` on standard input with `--stop-on-stdin`. A program that starts the server
  with no console has nothing else to send: that pipe is held by the program alone, which is why it
  needs no address, no key and no console. Nothing is written per request.
- **The file.** `<data dir>/conversation/<bundle>.sdconv`, one a bundle, replaced whole (a `.part`
  file, synced, then renamed over the old one). As the system block's (6.4) with two sealed parts:

  ```
  0    8   STRAYCNV         56   8   the state's bytes, S
  8    4   file version     64   4   the wrapped key's bytes, W
  12   4   state layout     68   W   the wrapped key (DPAPI)
  16   32  key              ..   12  the tokens' nonce
  48   8   tokens, T        ..   12  the state's nonce
                            ..       the tokens sealed (T u32) and their 16-byte tag
                            ..       the state sealed, and its 16-byte tag
  ```

  Both parts are sealed under the one random key with a nonce each. The associated data is the
  header and the part's name, and for the state also the tokens' part as the file holds it: a state
  opens only beside its tokens, and a byte changed anywhere refuses the file.
- **The key** is made as the system block's is, from the same facts (the bundle, the engine's
  version and build, what the worker placed, the numeric settings: `prefix::description`), under a
  domain of its own, and without the tokens: one file a bundle.
- **Restoring.**
  - After `ready` the server opens the file's header and the tokens' part, which are a few hundred
    kilobytes, and sends the worker the tokens (`conversation_tokens`). The state stays on the
    drive. A file whose key is not this start's (another build, bundle or placement), or that does
    not open, is deleted with one warning; with the option off at the start the file is deleted.
  - The worker compares every prompt with those tokens. The first that begins with all of them,
    when the model holds none of them already, checks that the host memory has room (three times
    the state's size: the file's bytes in the server, the same in the worker, and the copies of its
    parts, and 2 GiB more) and asks the server for the state in the middle of the request
    (`conversation_wanted`). The server reads and opens it and sends it (`conversation`, a frame
    that carries bytes, 6.4). A request that does not begin with the tokens leaves the file as it
    is, for a later one. Where the memory is not there, or the file is gone or damaged, the request
    goes on without it, and the log says why.
  - The request line says `reused N from kept, loaded in S s`: the time the server's read, the
    transfer and the import took.
- **Measured** on an RTX 5070 Ti held to 5.5 GiB (`--vram 5.5 -c 128000`), UD-IQ1_S, a conversation
  of 38,975 tokens and then the same plus its reply and a short message (the K/V starts at 32768
  positions, so the import grows it to 65536):
  - read from nothing: first token at 48.9 s (798 tokens/s);
  - stopped with the option on: 38,968 tokens, 1084 MB (28 KB a token, 30 with the MTP layer), kept in
    2.0 to 2.5 s (the stop takes 3.5 to 4.7 s in all, the worker's own end included);
  - the next start opens the header and the tokens in under 10 ms;
  - the request that goes on from it: `reused 38968 from kept, loaded in 1.8 s` (the server's read
    and opening 0.4 s, the pipe 0.3 s, the import 1.1 s), first token 2.2 s.
  - A frame's body is read in reads of 4 MiB (`frames::read_body`): `read_to_end` sizes its reads
    by what the last returned, a pipe gives 4 KiB, and the same bytes took 7.5 s.
  - The first logits after the import are the same bits as a model that never stopped
    (`conversation_gpu`, 8,124 and 38,226 tokens, with and without the MTP layer, with
    `STRAYDOG_PROMPT_CPU=0`). They are not the logits of a model that read the whole request in one
    go: any request that goes on from a checkpoint reads its tokens in other batches, which moves
    this quantization's logits by a few percent (4.7% of the logits' norm at 8K), restart or none.
- **Off.** Without `keep_conversation` nothing of a conversation reaches the drive. Elsewhere than
  Windows the option is accepted and keeps nothing: a state sealed for the user needs DPAPI.

## 7. DeepSeek-V4.1-Flash

The `deepseek41` architecture:
- 40 layers of hyper-connected MoE blocks, with 384 routed experts per layer, 6 used per token,
  and a shared expert;
- latent attention over a sliding window, plus compressed rows selected by an indexer (CSA2);
- n-gram Engram tables.

Its definitions follow the reference implementation, checked layer by layer against it
(examples/ds41_gpu, ds41_chat) and against llama.cpp's logits (examples/ds41_quality).

### 7.1 Bundles and layout

Pinned bundles (a bundle is named by its model, provider and quantization: two providers name their quantizations alike):
- **ISTA-DASLab IQ2_XXS** (GSQ-RCO, the default). Its experts are IQ1_M in 20 layers, IQ2_XXS in 14 and
  IQ2_XS in 6.
- **ISTA-DASLab IQ2_XS** (GSQ-RCO).
- **Unsloth UD-IQ1_M**, the size of ISTA-DASLab's IQ2_XXS: four shards, its Engram tables the same
  bytes as ISTA's file, which it pins as its fifth (a manifest item may name its own repository).
- **Unsloth UD-IQ2_XXS**.

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

The rest is replicated. Every device sums the attention's parts in device order. The experts'
activations and their output are each device's columns, the others zero: the devices gather them,
each reading only the others' columns over the bus (a long pass's activations in FP16, as the
products that read them take them). So all devices hold the same values, and every route runs on
every device. A missing expert
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
- 512 MiB for FP16 copies of one layer's dense weights and its engram_kv;
- the DSpark drafter's footprint;
- `RESERVE`, 1.625 GiB for the runtime's own memory. On DeepSeek-V4.1 that memory (kernels,
  handles, graphs) is 0.9 GiB; the rest of `RESERVE` covers Straydog's own buffers the terms above
  do not count: the device's workspace (352 MiB: a 128 MiB GEMM expansion, 128 MiB of input and
  output staging, 32 MiB of half inputs, 16 MiB of BLAS workspace, 48 MiB of attention partials)
  and the batch buffers (about 366 MiB, 357 of them lent to the store between passes), less 75 MiB
  the resident share is counted over. DSpark's Markov head (131 MiB) is on the first device alone,
  though the placement counts the first device's footprint on both.

The rest is expert slots. After the first passes each device also lends what its card has free
beyond 768 MiB (§5). Example: two R9700s with GSQ-RCO-IQ2_XXS at 262144 tokens hold 6059 slots
per device with their prefill batch of 16384 (the arena for a pass's rows is lent between passes)
and a 102.2 GiB host tier (12,629 experts' worth, of 15,360).

### 7.2 Passes

- **One pass over all its rows.** A pass runs layer by layer over all its tokens: a layer's
  attention a batch of rows at a time, then its experts over all the rows. Each routed expert is
  read once for them all, in waves, with the next wave's fills behind the one computing.
- **A long prompt** takes back the slots lent to the store and streams its experts through VRAM,
  while the host tier keeps theirs.
- **The prefill batch,** when nothing else is asked for, is the largest whose pass buffers (~0.32 MB
  a row on each GPU) take at most an eighth of the smallest GPU budget, from 8192 down to 256; on
  cards of 24 GiB and more, 32768 or 16384 within a third of it. `--batch-size` takes up to 32768.
  - Every pass reads the ~35 GB a GPU of experts the VRAM does not hold over the bus (~1.5 s at
    ~24 GB/s on both GPUs), whatever its rows, so a long prompt takes fewer of those trips in wider
    passes. Two R9700s take 16384: a 45K prompt's first token 30.8 to 27.5 s and a 20K one's 14.6 to
    11.9 s against 8192, the same replies.
  - The lent arena (§5) covers the larger buffers: the slot count between passes barely changes
    (6064 to 6059), and a long pass writes back the few hundred experts its buffers take.
  - Engram decodes a batch at a time, so a wide pass's Engram rows no longer outgrow the row cache.
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
- **What a long pass's kernels share.** The attention's output kernel splits a row into dim / 64
  blocks: a kernel before them forms each row's softmax weights and denominators once, in place. The
  HC step takes 16 rows a block, each thread holding its column's mixing weights, and forms its 26
  wave sums a step of all at a time. A wave's routed outputs go to their rows in one launch, each
  row's routes added in the wave's order (`Grouped::by_row`), only this device's columns. All of it
  bit for bit; on a 45K prompt each GPU's kernels took 17.65 s before the gathers, the one launch a
  wave and the interleaved sums, and 16.88 s after.
- **A decode step.**
  - Each device runs the step as recorded graphs, replayed while their launches stay the same.
  - The router writes its routes into page-locked memory, which the host polls.
  - Then, layer by layer:
    1. The misses are requested on every device before any device waits, filled one expert at a
       time across the devices.
    2. The hits' gate and up run while the misses' uploads land. One event stands behind all
       the misses' gates and ups, and one behind all their downs.
    3. The misses' gate and up run, then every expert's down.
  - An evicted slot whose expert the host tier still holds is dropped; any other is written back
    to the host tier after the compute stream passes the point where the slot was last read. Each
    size class keeps up to 32 slots free during passes, and 64 more between requests, for a
    prefill's first waves.
- **What a miss costs.** On two R9700s (PCIe 5.0 x8 each) a decode pass took 22.3 ms + 0.003 ms a
  distinct expert + 0.125 ms a miss (served stamps, 2026-11-02).
- **Engram rows** for a pass's tokens are prefetched before it (§4.2) and gathered at the layers
  that hold a module.
- **Verify passes.** A pass of up to the verify rows can be taken back but for its first few
  (`Engine::rewind`): it keeps the sequence state's counters from before it, and each ratio-2 kv
  source's compressor rows of its tokens. A verify pass computes every row in every layer, with
  every token's logits.

### 7.3 Speed

Two R9700s, 125 GiB of RAM, with DSpark:

| Measure | Result |
| --- | --- |
| Decode, a 20K conversation's turns, GSQ-RCO-IQ2_XXS (2026-11-03) | 74-80 tokens/s |
| Decode, the same, UD-IQ1_M (2026-11-03) | 63-71 tokens/s |
| Decode, 55-73K, GSQ-RCO-IQ2_XXS (2026-10-28) | ~57 tokens/s |
| First token, a turn continuing a 20K conversation (2026-11-03) | 0.3-0.6 s |
| First token, a fresh 20K prompt (2026-11-02/03; it varies between sessions) | 11.9-20.8 s |
| First token, a fresh 45K prompt (2026-11-02, before the 2026-11-03 prefill kernels) | 27.5 s |
| A 3-row decode pass on fixed tokens (2026-11-02) | 27.4 ms |

### 7.4 DSpark

DeepSeek's three-stage block drafter (`mtp.0..2`, Unsloth's Q8_0 GGUF, 7.98 GB) is an optional
file beside the bundle.

**The file.**
- It is pinned like the shards, with its repository and revision.
- It lies in the model's `dspark` directory or in one beside the model's, so a model's
  quantizations share it. `straydog fetch` and `setup` download it.
- Every bundle of the model uses it, whatever its provider and quantization: it only proposes
  and every draft is verified, so no bundle needs a test of its own for its answers; how many
  drafts are kept is what differs.
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

**Its VRAM.** Each stage keeps its most used routed experts resident: 48, 28 and 16 of 128 (1.7 of
7.2 GB). The router's bias keeps the others out of a draft's top three. The first 87, 50 and 29 (98% of
its routes) left its acceptance as it was, but what the drafter keeps is the target's expert slots: with
48/28/16 a draft's tokens are kept a little less often (a code reply's 81-82% where 85-89%), and a bundle
short of slots runs faster (UD-IQ2_XXS: code 51.7 to 55.7 tokens/s, five agent turns 26.6 to 25.5 s;
UD-IQ1_M the same within its spread). `DS41_DSPARK_RESIDENT=A,B,C`, or `all`, changes them.

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

Qwen3.8-Flash-Next is served from eight pinned bundles: Unsloth's UD-IQ3_XXS, UD-IQ2_XS, UD-IQ1_M and
UD-IQ1_S (Dynamic V3.1; three or four shards, the first holding the metadata alone, the last the engram table,
which UD-IQ1_S, UD-IQ1_M and UD-IQ2_XS share), offered first, and ISTA-DASLab's GSQ-RCO IQ3_S, IQ3_XXS, IQ2_XS
and Q2_0, two shards each. Unsloth's keep their attention in Q5_K/Q6_K and their shared experts and SSM output in
Q8_0: about 1.2 GB more read each token than ISTA-DASLab's bundle of the same size, about 1.1 ms on two R9700s. They
come with the tokenizer and template files of the official release, and optionally with its MTP drafter.
An Unsloth bundle reads its engram (PLE) table from its own last shard (IQ4_NL) unless `--engram`
(`STRAYDOG_ENGRAM`) names one of Unsloth's engram files (BF16, Q8_0, Q5_1: one tensor in a file of its
own, pinned like the shards), in the bundle's directory or an `engram` directory beside it, read in the
last shard's place (`bundle::table_in`); without its shard, the most exact file there. Q5_1 rows decode on the host like the others.
`straydog fetch-mtp` builds the drafter from the official tensors, byte for byte the pinned file.
Its experts are quantized to Q2_0, which changes only how many drafts are accepted.

Its regression oracle is the pinned GPU generations (`straydog-engine/tests/model_gpu.rs`). For
each placement it pins every request's greedy tokens, metrics, logits and final hidden state, bit
for bit. A deliberate numerical change is checked against an F32-GEMM run, then pinned anew
(`STRAYDOG_GOLDEN_UPDATE=1`).

**Placement.**
- **Two R9700s:** they hold the weights of every bundle with every layer in parts (below), with the
  K/V in host memory at 262144 tokens.
- **Smaller budgets:** host K/V and a smaller prefill batch come first, then partial residency.
  What a budget must hold for it is the placement's own arithmetic (`model::partial_slot_bytes`):
  the device's workspaces, the recurrent state, the prefill rows, the K/V stage and the pinned dense
  weights, 4.1 GiB in all for IQ2_XS at 262144 tokens; `straydog setup` counts with the same
  function, and what is over goes to the expert slots.
- **Partial residency:** the CPU computes experts beyond VRAM from the host tier. Which
  experts run where depends on timing, so decode is not bit-reproducible from run to run.
  The PLE and output-head operands join the authenticated load before the expert pool is sized,
  so those fixed dense weights are counted before the pool takes the remaining VRAM.
  - On a CPU with AVX-512 and VBMI, or AVX2, the activations are rounded to 8-bit integers (gate and up: one scale a 256-value block;
    down: one a 64-value block) and the products summed exactly in integers, then scaled in F32 a block (`gpu/cpu_int8.hpp`), as
    the GPU's GEMVs already round theirs. IQ2_XXS's grid rows hold only 8, 25 and 43, so all 256 rows fit in eight registers as
    2-bit codes and a block's rows are looked up there; Q2_0's codes come out of one GFNI affine transform where the CPU has it.
    Where the CPU has VNNI (AVX512-VNNI with the AVX-512 kernels: Intel's 10th and 11th generation, Zen 4 and 5; AVX-VNNI with the
    AVX2 kernels: Intel's 12th generation on, Zen 5), one vpdpwssd takes an IQ2 chunk's scaled pair sums and their add, and one
    vpdpbusd a Q2_0 row's products, their sums and its bias: the same integers (a pair of products stays under 2^15). On the 7500F,
    six threads, IQ2_XXS layers: cold 35.5-38.3 -> 33.7 us a route, hot 29.9 -> 27.3; one thread, the down 34.1 -> 27.5 us. Served at
    5.5 GiB, one build with VNNI off and on, two pairs: the CPU's time a token 8.7 / 8.2 -> 8.2 / 7.8 ms, 49.5 / 50.3 -> 50.3 /
    50.9 tok/s. The 7500F has no AVX-VNNI; the AVX2 kernels' integers with it were checked there with the same instructions' EVEX
    forms (44 -> 42 us a route held to AVX2). A
    row's accumulators stay in registers (an array of them went to memory, four 512-bit stores a block). Ryzen 5 7500F, Qwen3.8
    UD-IQ1_S, a route with the experts cold, F32 -> Q8: on six threads 48.0 -> 35.2 us on its IQ2_XXS layers and 46.0 -> 36.9 us on
    its IQ2_S layers; on one thread 226 -> 168 and 209 -> 165 us. Teacher-forced on six prompts, the KL against an F32 run is 0.0022 (two F32 runs differ by
    0.0009). The AVX2 kernels (Intel's 12th to 14th generation, Zen 3) compute the same lanes in two 256-bit halves, so the same
    bits. They read a block's grid rows, sign masks (a 64-bit mask a sign field, from a table) and scales straight from its bytes, with
    no prepared block written and read back. On the 7500F held to AVX2 (`STRAYDOG_CPU_AVX512=0`), six threads, cold: 73.7 -> 47.4 us
    (IQ2_XXS layers), 84.9 -> 59.0 us (IQ2_S), the down alone 18.7 -> 10.0 (one thread 93 -> 32); served at 5.5 GiB, 42.1 -> 48.4
    tok/s. Formats
    without integer kernels, and CPUs without AVX2, keep F32; `STRAYDOG_CPU_INT8=0` keeps it everywhere.
  - A prompt batch streams each expert the slots do not hold over the bus, whatever its rows; the CPU computes a route at a time.
    So the experts with the fewest rows go to the CPU (`prompt_cpu_share`, `gpu/resident_moe.hpp`): while the bus streams the rest,
    the CPU computes their routes from the batch's rows (copied back as the experts go the other way), and their results reach their
    routes' output rows after the GPU's GEMMs. It takes them while its time for their routes stays under a share of the GPU's for
    what is left, each cost measured in the last batches; the share moves each batch toward the CPU finishing at nine tenths of
    the GPU's time. On a layer whose banks are not page-locked the CPU's workers first gather each streamed expert into pinned
    memory, and its own routes wait behind those gathers on the same workers: there an expert goes to the CPU only when its
    routes cost less than its gather (measured). On the R9700 at 5.5 GiB in a 40 GiB memory scope (27 of 48 layers locked), two
    runs each: 2,100 tokens 2407 / 2228 -> 2048 / 2066 ms, 4,158 tokens 3428 / 3353 -> 3080 / 3059, 10,611 tokens 7566 / 7615 ->
    6538 / 6688. Which experts go where depends on timing, so a partially resident prompt is not bit-reproducible from run to
    run either; `STRAYDOG_PROMPT_CPU=0` (or `Model::prompt_share(false)`, as the golden test does) streams them all. Radeon AI PRO
    R9700 (PCIe 5.0 x16) beside a Ryzen 5 7500F, Qwen3.8 UD-IQ1_S at 5.5 GiB, two runs each: a 2,101-token prompt 2183 / 2087 ->
    1717 / 1711 ms, 2,397 tokens 2081 / 2088 -> 1714 / 1703, 3,394 tokens 2393 / 2395 -> 2119 / 2131 (a third of the streamed
    experts left, 22,600-25,600 routes to the CPU). A narrower bus (a PCIe 4.0 x8 card) makes a streamed expert dearer, and the
    CPU's share larger.
  - A long prompt can borrow expert-slot VRAM to widen its batches, up to 4096 rows. On one GPU
    the threshold is the smaller of 1536 new tokens and two original batches; the cache then
    gives way to the staging pool. With several GPUs the threshold is 65536 new tokens: each
    device keeps the cache that fits beside its wider workspaces, and the smallest supported
    batch limits the group. A batch already at 4096 rows skips borrowing. Completion and
    cancellation run the same workspace/cache restoration path; the restored pool uses what
    fits at that point.
  - An arriving request can interrupt startup before any expert slots are filled. When the
    selected devices have positive cache capacity but no occupied slots, prefill observes at
    least one batch of routing and fills the cache once if more tokens remain and the request
    has not been cancelled. Later batches can reuse those experts. A cache already holding
    any experts keeps its existing policy, as does a staging-only pool with no cache capacity.
  - Its host K/V starts at 32768 positions and doubles as a sequence reaches them (`sd_kv_grow`):
    page-locked at once, the whole 262144 would take 12 GiB from what the expert banks need. The
    indexers hold the whole context from the start and are laid out anew in place. A parked
    sequence does not survive a growth; the live one and its checkpoints do.
  - The page locks give way to the K/V: they are an optimisation of the CPU's experts, and a conversation's K/V is not. A growth that
    the memory refuses (`KV_NO_MEMORY`) ends the locking behind ready if it is under way, takes locked layers back newest first
    (`Model::give_back`: as many as their bytes cover what the memory available lacks, and at least one), and asks again; only with
    no layer left does the request fail. A layer given back is taken out of the assist (its plan reads the bank in place no more, the
    layers' graphs that held the view are captured anew), its banks are unregistered, and the CPU's experts and the slots read it as
    they read any layer that is not locked. One line in the log says what was given. The locking leaves room: before each layer's turn
    it looks at the memory available then and stops, saying why, when the layer would leave less than the plan's margin and the K/V
    that the context's first 64K tokens have yet to grow; past 64K the give-back covers it. The copies of a parked sequence's positions
    (pageable, charged to the commit like the locks) ask for room the same way, and where there is none even with every layer given back
    the sequence is not kept: the request runs without it, where an allocation that fails in a Rust `vec!` would end the process.
  - Page-locked banks let the GPU take some of a layer's absent experts over the bus (`assist`). On
    CUDA a kernel first copies them into VRAM's staging slots with every thread at once
    (`hybrid_stage_assist`), two a layer: read in place they outlast the CPU's. At
    --vram 5 on 62 GiB with 41 layers locked (RTX 5070 Ti): 38.1 tokens/s with none, 35.2 with two
    read in place, 42.8 with two staged.
    How many a layer is given (the level: 0 to 2 on CUDA, 0 to 4 on HIP) is the engine's to choose by
    measuring (`assist.rs`), since it depends on the machine's PCIe link, CPU and VRAM. A round times
    32 decode passes of one row at the level in force and 16 at each other (0, 1, 2 on CUDA; 0, 2, 4
    on HIP), the layers' wall time as `step: layers` has it, every 2048 decode tokens and at a request
    that begins after more; a level takes over only if 2% faster and by more than twice the standard
    error of the difference of the windows' means (passes vary by a quarter, and the log line gives
    each level's mean ± its error). A speculating model leaves its
    drafts out of a round's passes so that each has one row. The level is carried by the layers'
    captured graphs (`hybrid_assist` is passed to their kernels by value): a change drops the hybrid
    layer graphs of decode and of verification (`hybrid_set_assist`) and the next pass captures them
    again. With no layer page-locked nothing is measured. The level is in the request line.
  - With less host memory than twice the banks, the PLE table's row cache is 1 GiB: a token reads
    few rows and hundreds of experts.
  - With no bank page-locked the workers hint a layer's absent experts as they learn its routes
    (`sd_cpu_read_ahead`), since a row's page fault holds up every thread. Touching the pages first
    gains nothing. Only experts unread for 64 tokens are hinted: 15% of the routes, and 98-100% of
    the misses of an LRU of 15 to 27 GiB in a 5592-token trace. A fixed set of the hottest experts
    would miss more than the LRU (47 misses a token against 18, at 23 GiB). IQ3_S on 29 GiB of RAM,
    after a dozen requests: at 6 GiB of VRAM 18 -> 22 tokens/s. Some 20 MB a token still comes from
    the drive. A machine that locks banks sends none.
  - On Windows the worker asks for a working set of what its share of host memory leaves
    (`host::keep_resident`): pages trimmed from it come back one fault at a time under the CPU's
    expert threads.
  - Measured at --vram 5, 262144 tokens, 32 GiB of RAM simulated (26 GiB free) on an RTX 5070 Ti and
    a Ryzen 7 9700X (2026-10-31): 30-33 tokens/s on one topic once warm, about 60 us a CPU route
    (AVX-512, eight threads) and 310 routes a token. A change of topic reads the experts it finds
    missing from the drive, which holds it near 15-25 tokens/s until they are in memory.

**Long-context prefill.**
- Every fourth layer attends sparsely: an indexer scores the prompt's 4-token blocks, and each
  token attends the keys of its 512 best complete blocks and of its own incomplete block.
- A prompt is prefilled in batches, split into 256-row chunks for the indexer.
- With FP16 GEMM on RDNA4, a batch's attention runs at once. In a span that starts below 8192
  tokens, groups of 16 tokens walk the union of their selected blocks, so each key and value is
  read once per group (`qsa_flash_wmma`). From there each token attends its own selection
  (`qsa_prompt_attention`, K/V in the device's memory): a workgroup is a token and a K/V head, its
  four waves take every fourth tile of 16 keys with an online softmax each (merged at the end),
  the K rows go from memory into the WMMA operands and the V rows through the wave's own LDS, so
  no wave waits for another. The union saved less than its walk cost: 16 tokens' 8.2K selected
  blocks are a union of 1.6K (32K tokens), 2.1K (95K) or 2.3K (242K) blocks, 40% of them selected
  by one token at 242K, and a group's tiles of loads, staging and barriers took 71 us a token (both
  GPUs, 13 layers) at 242K, a quarter of it the global loads. The same FP16 keys, values, queries
  and probabilities, F32 sums in another order. Two R9700s, UD-IQ3_XXS, MTP, 262144 context, batch
  2048: the attention 75 us a token (flash 71, union 3, marks 1) to 19.6; a fresh 242K read 64.7 s
  to 58.8 s, 95K 22.3 s to 20.6 s, 32K 8.07 s to 7.82 s. The pinned generations' prompts end
  before 5.3K, so they keep their bits.
- In parts the two devices share the indexer (a span's chunks of 256 rows in turn), and from 8192
  tokens a device attends its own chunks' tokens in the chunk loop, from the selection it made, and
  sends the selected blocks (`qsa_send_blocks`) into the other's inbox. The other's chunks it
  normalizes the queries of into the place of their attended values (`write_attention_kv` reads only
  the keys and values) and attends after the span's one signal and wait, from the block numbers it
  received, in one launch of the span's chunks of the other (`qsa_prompt_attention<..., Received>`):
  the selection list is rebuilt in the order the indexer wrote it (the selected blocks in block order,
  four positions each, then the incomplete block's), so every sum is the one the device's own kernel
  makes and the two paths give the same bits as one device selecting every chunk. The attended
  values replace the queries in place: a workgroup reads its token's queries before it writes its
  values, and no two share a token and K/V head.
- From 6144 complete blocks (24576 tokens), in a span where each token attends its own selection, an
  RDNA4 chunk's indexer makes its selection from FP16 matrix products with a provable bound
  (`qsa_interval.hpp`) and selects the very blocks the exact kernels do, in the same order and with
  the same ties. The exact scores (128 columns of F32 fused multiply-adds for each block and head,
  `qsa_block_scores_tiled`) took 27.7 us a prompt token of a 242K read (both GPUs, 13 layers) and
  the selection of the best 512 (`qsa_select_rows`) 13.3: 41.0 in all, 11.1 now.
  - A block's four heads are WMMA products of the keys and queries rounded to FP16 (F32 sums), their
    positive parts added. With h the rounding and r its residual (exact in F32), the exact score of
    a head differs from the product of the rounded operands by at most |h(k)||r(q)| + |r(k)||h(q)| +
    |r(k)||r(q)| (Cauchy-Schwarz) and by 2^-15 |h(k)||h(q)| for the F32 sums (the matrix unit's
    accumulation measured at 4e-8 to 8e-8 of it), and the positive part is 1-Lipschitz, so a score
    lies within d = |h(k)| (M + 2^-15 N) + |r(k)| (N + M) of the matrix unit's (N and M the heads'
    norms of the rounded queries and their residuals; norms taken up, the whole 2% over). No
    distribution is assumed: elements under FP16's smallest normal count as zero, and an element
    beyond FP16 or not a number leaves its row to the exact kernels. On a 242K prompt's indexer |a -
    e| reaches 0.10 to 0.16 of d, and d is some 2e-4 of the product of the norms.
  - With L = a - d and U = a + d, the 512th largest L is at most the 512th best exact score T and
    the 512th largest U at least it: blocks with L above the second are certainly selected, those
    with U below the first certainly not, and the rest (the band, 45 blocks a row on average and 96
    at most, in 3066 sampled rows of the 95K and 242K reads past 48K tokens) are scored exactly from
    an F32 copy of the keys in the exact kernel's order of operations. T is the (512 - certain)th of
    the band, ties take the earliest blocks and the selection is listed in block order, as
    `qsa_select_rows` lists it.
  - The 512th largest bounds need every block's interval, which would be as large as the exact
    scores to keep. A sample of one block in 32 (scored the same way) gives a threshold at about the
    1000th best score instead; the scan (`qsa_iv_scan`: a workgroup of 8 waves, 128 rows, tiles of
    64 blocks staged as FP16, the keys' loads in flight) keeps as a bit and a score the blocks whose
    upper bound reaches it with the chunk's largest interval taken off (1135 a row on average, 1770
    at most of the sampled rows), and the selection (`qsa_iv_select`: a workgroup a row) checks that
    512 blocks certainly reach the threshold. A row where that fails or that has more than 2048
    candidates (203 of the 3.8 million rows scored in the 32K, 95K and 242K reads, 190 of them with
    2049 to 2332), a band over 512 or an operand beyond FP16 is left to the exact kernels, which run
    for the chunk and leave all but those rows: the time of the exact kernels, not another result.
    Nothing waits for the host.
  - The F32 copy of the keys and their norms (`qsa_iv_keys`) live in the attention scratch, which
    nothing else holds while the tokens attend their own selections, and a layer's chunks in a row
    take up from the blocks the one before left there.
  - Two Radeon AI PRO R9700, UD-IQ3_XXS, MTP, 262144 context, in one session, the old and the new
    kernels alternating: a fresh 242K read at batch 2048 53.53 s to 49.87 s, at batch 4096 50.02 s
    to 46.34 s, 95K 19.73 s to 19.27 s (batch 2048), 32K unchanged; the bits are the same in every
    arm. The scan runs at about 105 TFLOPS of FP16 where the matrix unit gives 170 in a loop of its
    own: its 235 registers leave one workgroup of 8 waves to a workgroup processor, and a micro-
    benchmark finds a SIMD's vector work not overlapping its matrix products.
- With host K/V, a layer's prefix is staged in VRAM for its batch, copied on a stream of its own
  while the layers before it compute.
- Past the stage's positions (a 64th of the budget, grown into what the slots leave: some 45K at 6 GiB),
  a qsa_flash batch copies only the union of its tokens' selected blocks into the stage and renumbers
  the selections to it (`qsa_attention_gathered`): the same keys in the same order, so the same bits,
  where each token read its 2052 rows over the bus before (RTX 5070 Ti, 52K tokens: 119.4 s to 61.0 s).
  As many tokens at a time as the union fits for; the host reads the union's size, so a layer split in
  parts keeps reading host memory. RDNA4's batched attention gathers its span's union the same way, its
  groups reading their blocks at their places in the stage (and as many groups at a time as it fits for).
- The K/V holds FP16 values (`kv_t`): a plane's row of a position is 256 halves, half the bytes of
  F32 in VRAM, in host memory, in the stage and its mirrors (which then hold twice the positions),
  and in a sequence's saved positions. The flash rounded keys and values to FP16 already, so a
  prompt reads the same values; decode and verification read the rounded ones (KL against F32
  replies 3.6e-4 to 6.6e-4 at 8K to 95K, §Numerics in the README).
- A DeltaNet layer's scan (`delta_split_sequence`) gives each column of a head's state eight lanes
  and a block 32 columns, four blocks a head (the columns are independent), and stages 16 tokens'
  q, k, v, decay and beta in LDS a barrier at a time; its sums are explicit FMAs in the order the
  one-block-a-head scan compiled to, the same bits at 1.9x its speed.
- A read of 16384 new tokens or more (`CHUNKED_SCAN_READ`; `STRAYDOG_DELTA_SCAN=token` keeps the scan above)
  runs those layers' recurrence in chunks of 16 tokens (`gpu/gated_delta_chunk.hpp`, RDNA4). With g the chunk's
  decay products, its corrections solve a 16 x 16 unit lower-triangular system, (I + L) C = beta v - diag(beta g) M0^T k,
  its outputs are g M0^T q + B C and its state g M0 + K^T diag(e) C, so only the products against the state wait
  for the chunk before. The rest (`gdn_chunk_prep`: the convolution, the q and k norms, the Gram products and the
  solves) is free of the state and runs for all chunks of a 512-token launch at once, in place of gdn_conv_edges,
  gdn_conv_runs, gdn_normalize_qk and gdn_pack_values. `gdn_chunk_scan` keeps the state in WMMA accumulators and does
  every product against it in bf16 pieces: a value is three bf16 (each the leading 8 bits of what the ones before
  leave), a product is six of the nine pieces' products with F32 sums, which carries about F32's bits (FP16 operands
  would round every product to 11). The F32 FMA form of the same algebra measured 1.3x slower than delta_split_sequence
  on an R9700: its operands stream through LDS a float an FMA, and a chunk's products are too small to tile (24 heads x
  128 columns of state).
  The layer's output norm after it (`gdn_post_rows`) is `gdn_post`'s tree of sums with five of its seven barriers replaced by shuffles:
  the same bits (the logits of a 95K and a 242K read are the same), 35.6 -> 28.6 us a 512-token launch. A wave a row, which sums
  the 128 channels in another association, takes 14.4 us and changes one value in 25 by a last bit, which every read's bits
  would show.
  One R9700, 512 tokens, QwenHeads<2> shapes, random q/k/v: convolution, normalization, packing and scan 340-385 us,
  the two kernels 187-228 us; against an F64 scan the outputs' largest error is 2.3e-7 to 4.8e-7 of the largest output
  (the token scan's: 1.0e-7 to 2.0e-7). On the activations of a 95K read (4 heads of 12 layers against an F64 scan) the
  chunked outputs are 4e-7 to 3.6e-6 of the largest output off, the token scan's 5e-7 to 1.6e-4: within twice the token scan's
  in every layer, and 10x to 100x below it where the state forgets little. A first version was 2e-6 to 7e-4 off, growing
  linearly with the tokens read, from two coherent losses a chunk: a matrix core's accumulate of a product sum into a large
  accumulator (the state, which the update's six chained products were added into) drops the sum's low bits toward zero
  (measured on an R9700: half an F32 spacing a WMMA, with errors of up to three spacings, so six a chunk shrank the state by
  1e-7 of itself), and g_end rounded to F32 lost 3e-8 a chunk. The products now start from a zero accumulator, the decay is
  F32 fused multiply-adds with g_end as two floats, and the real kernel's error matches a staged F64 emulation of the same
  algorithm (every stage rounded as the GPU rounds it, run on the same activations). The first logits after a 95K read differ
  from the token scan's by KL 6.7e-4 (242K: 1.6e-5), where the token scan at another batch width differs by 4.4e-2 (1.6e-5).
- Measured at IQ3_S on two R9700s at 262144 context: 131072 tokens prefill in 34.3-34.8 s, and
  4096 in 1.23 s.

**Layers in parts** (`gpu/qwen_parts.hpp`, `Group::enable_parts`). Two GPUs run every layer of a
pass or a prompt batch at once, each its part, a half-size Qwen3.8 layer (`QwenHeads<2>`):
- **Split:**
  - a DeltaNet layer's key heads 8p..8p+7 with the 24 value heads that read them (16c + 8p.., c = 0,
    1, 2): in the part, value head j reads key head j % 8 as the whole layer's reads h % 16, so the
    kernels are the whole layer's with the head counts halved (templates on the parts);
  - an attention layer's query heads 12p..12p+11 with K/V head p: a part's K/V is two planes, so its
    host state, its mirror and the positions a mirror covers are halves (a 60K decode's attention
    reads its mirrors on both GPUs at `--vram 30`);
  - half the routed experts, as with the layers whole, and half the shared expert's width: rows
    [320p, +320) of gate and up, those columns of down. A pass runs the half as the shared expert; a
    prompt's grouped experts run without it and the half beside them as FP16 GEMMs;
  - their rows of the projections and columns of the output projection (`Model::part_operands`,
    gathered at load: a piece of the mapped matrix a run of rows, or a run of every row's columns).
- **Whole on both:** the residual streams, the HyperConnections and the router, so both select the same
  experts; and the attention's indexer, so both select the same K/V blocks (a pass's rows alike; a
  prompt's chunks of 256 rows go to the devices in turn, and the blocks each selected are sent as 16-bit
  block numbers, 1 KiB a row, into the other's inbox, so both hold every row's selection). The PLE layer
  runs on both as well.
- **Sums.** A layer's attention output and its experts' are each a part of a sum (the SumLink of
  `gpu/device_link.hpp`): a pass's rows in one single-block kernel that sends this device's part,
  raises the other's flag, waits for its own and adds the two parts into the four streams (both add
  the same two values, so the streams stay equal). The pins are the parts' own; speculation equals
  decoding as for whole layers.
- **A prompt batch's rows** are split between the devices for what both compute: each runs the
  HyperConnections and the residual updates of its half of the rows. A HyperConnection's mixed rows go
  to the other device (all-gather) and a layer's output parts are added only for each device's rows
  (reduce-scatter), both in FP16 (each device rounds its own as the other's arrive, so both read the
  same values): the bus carries what an all-reduce of F32 halves would, and the HyperConnections, run
  by both on every row, had been a fifth of a 60K prompt. The exchanges store 16 bytes a thread into
  the other's inbox and read 8 from their own. A scatter's residual update also normalizes the rows
  for the HyperConnection after it, whose FP16 projections then read halves only (its finish
  computes the normalized values again from the rows rather than reading floats back); a product
  is kept an F32 before its FP16 cast (an asm barrier: the compiler would round it once with
  v_fma_mixlo_f16), so the bits are the two kernels'. On a 95K prompt (UD-IQ3_XXS, batch 2048) these
  and the DeltaNet scan took the prefill from 25.5 to 23.3 s. Both hold every row after the first layer
  (the PLE layer's input) and the last. Against the F32 reference a prompt's first logits are as far
  as with whole layers (KL 1.3e-3, 5.8e-5, 2.4e-3 at 600, 2500, 5000 tokens; whole: 3.7e-4, 1.3e-3,
  2.9e-3).
- **A prompt's batches** follow each other with no host wait, the MTP layer's pairs read from rows the
  second GPU keeps (the deferred prefill of layers split, §8 above), and a DeltaNet's causal convolution
  runs over runs of 32 rows at once (bit for bit the rows' one after the other).
- **The host queues** both devices' work at once: a pass's span on two threads, as graphs of a few
  layers (both devices start the span together, and a 47-layer graph had one catch up with its
  submission at the second layer and wait 0.6-0.8 ms); a batch layer by layer, both devices, with no
  host wait until its last span. The shared expert's gate and up run on the device's stream: on the
  side stream every pass of three or four rows took 7.5 ms longer.
- **The MTP layer** is in parts as well (`STRAYDOG_MTP_PARTS=0` keeps it whole on the second GPU). A
  draft step queues the prologue (on both: its rows are both devices') and the layer's part on each
  device from two threads, then the head on the second's rows (`sd_draft_queue`, `sd_draft_head`); a
  prompt's pairs run the prologue on both, from the rows each keeps, in pieces of 512 rows into the
  workspace, and the layer in parts once over the batch. Its float
  projections go up as Q8_0 as when whole (Data::QuantizeGather: a deferred upload quantizes its piece
  of the verified bytes in crate::load, as it does a whole tensor), its shared expert's half as the
  other layers'. A part's K/V is two planes, so at 32768 tokens `two-gpu-experts-mtp`'s K/V now fits
  VRAM. The same tokens and drafts (greedy in-process at 20K); served, a pass of 1 / 4 rows at 60K
  12.83 / 20.98 to 11.93 / 19.37 ms (the layer's attention, which grows with the context, halved), at
  20K 11.88 / 19.43 to 11.61 / 18.91 with the read-back below; 135.2 to 146.5 tok/s on the 60K
  explanation, 135.0 to 151.4 on its code.
- **A head's read-back** waits for each device's streams once: both links' failure counters ride along
  with its tokens (the peer's copy queued after its part), where peer_settle read each with a blocking
  copy after the waits. A draft step's head ended 0.13 ms after its last kernel, now 0.06 (device
  spans of a draft step at 20K: the prologue and the layer 0.38 ms, the head 0.26); a pass at 20K
  16.05 to 15.80 ms greedy in-process, bit for bit.
- **Fewer launches a layer.** The experts' part is formed by its sum's own block (the routes this
  device holds, in route order, then the shared half: moe_part_partial's sum, a launch less); the
  router and the shared expert's half read the same row in one launch (the scores after the decode
  workspace, with the gate and up rows' stride); a HyperConnection's Q8_0 unit is one 8-byte read
  where it was four of two bytes, and at three or four rows stage one takes 16-row blocks (x * norm
  staged once, both of a wave's rows reading it, the wave's twenty weight reads in flight at once)
  and stage two four columns a wave (the low-rank vector rebuilt by 80 blocks, not 320). All bit for
  bit: a pass 18.38 to 17.84 ms at 20K, 19.03 to 18.52 at 60K (greedy in-process, the same passes).
  Then, also bit for bit: a part's routed gate and up and their product in one launch (a wave's lane
  groups take rows j.. of a route's gate and the same rows of its up, the gate's lanes form
  silu(gate) * up; then waves form the shared half's product), and an attention layer's pooling and
  K/V writes in its normalization's launch and its output gate in the merge (three launches a layer
  less). Served, a pass of 1 / 4 rows at 20K 12.24 / 19.87 to 12.00 / 19.55 ms, at 60K 13.14 / 21.50
  to 12.94 / 21.15. A DeltaNet layer's history copy for its tokens rides in its first HyperConnection's
  second launch (four more blocks), and at one or two rows stage one takes 24-row blocks of 768 threads
  (two a WGP: all 56 run at once, where 44 blocks of 1024 threads took two rounds of 32): a pass 1-2%
  shorter, bit for bit.
- **A draft's vocabulary.** A draft's head reads the head's first 100000 rows, split between the
  GPUs, and the rows of the tokens past them that the sequence has held, copied on the second GPU
  (`sd_draft_vocabulary`, at most 16384; 34 MB). The first 100000 tokens are English, code and the
  common CJK characters (every token of the served cases' replies); past them are multi-character
  CJK words and other scripts (a Japanese text's tokens: 82% below 100000, 97-99.6% with the tokens
  of the text before it). A draft step reads 40% of the head's rows: the same drafts and 141.5 to
  146.6 tok/s on English at 20K, 127.9 to 132.0 at 60K; on Japanese 125.9 to 126.8 (fewer drafts
  accepted, 73.5 to 69.5%, outweighed by shorter passes). Verification reads the whole head.
  `STRAYDOG_DRAFT_VOCABULARY=0` drafts over the whole head. With the launches above, served at real
  use a pass of 1 / 2 / 3 / 4 rows at 20K 12.95 / 16.07 / 19.26 / 21.45 to 12.24 / 15.05 / 17.90 /
  19.87 ms, at 60K 13.91 / 17.28 / 20.98 / 23.15 to 13.14 / 16.22 / 19.67 / 21.50.
- **Measured** greedy in-process at IQ3_S, 20K tokens: a pass of 1 / 2 / 3 / 4 rows 14.83 / 17.28 /
  20.22 / 22.60 ms with whole layers, 12.54 / 15.61 / 18.59 / 21.05 in parts (a pass 19.95 to
  17.77 ms); at 60K a 4-row pass 30.28 to 23.37 ms (81 to 120 tok/s). A prompt: 20K tokens 6.07 to
  5.84 s, 60K 16.3 to 16.7 s. Served at real use, a pass of 1 / 2 / 3 / 4 rows at 20K 14.40 / 17.96 /
  20.25 / 22.48 to 12.95 / 16.07 / 19.26 / 21.45 ms, at 60K 14.87 / 18.06 / 21.57 / 23.60 to 13.91 /
  17.28 / 20.98 / 23.15; the first token at 20K 5.67 to 5.43 s, at 60K 15.6 to 16.7 s.
  `STRAYDOG_PARTS=0` keeps the layers whole.
- **What did not pay:** every routed expert's half of its width on each GPU (both run every route, so
  neither waits for the other's share; a pass 0.3 ms shorter, but a prompt's 320-wide grouped tiles ran
  6% longer, and its first logits moved from F32, KL 4.4e-2 at 600 tokens: a fault in the prompt's
  experts not found); 16-byte plain stores for a pass's
  sums (no change); queueing a pass's span from two threads without the graphs of a few layers (the
  wait only moved to the other device); a HyperConnection's two stages in one launch, stage two's
  blocks reading their up rows before they wait for stage one's (RDNA4 places a 1024-thread block on
  a WGP, 32 at once: stage two's could not start beside stage one's); stage one's ten weight reads in
  flight at one or two rows (a 2-row pass 0.3 ms longer); five routed units in flight per expert row
  instead of two (a 4-row pass 0.2 ms longer: fewer waves resident); kernel arguments in device memory
  (`HIP_FORCE_DEV_KERNARG=1`, no change); a draft's head over 65536 rows (fewer drafts accepted); each
  block of the gate and up launch routing its token itself (resident_route's top-10 in every one of
  some 400 blocks: a pass 1.2 ms longer); DRAFT_CUT retuned for the cheaper drafts (a replay of 425
  sampled passes drafting three each: 0.45-0.7 within 1% of the best, 0.6 kept); the part sums folded
  into the kernels around them (with the sums' exchange skipped altogether, a pass is 0.65 ms shorter
  at one row and 1.0 at four, and of that the folding would take only the launches: the rest is the
  experts' share, which differs per token between the GPUs, 17.6 µs apart on average at an experts
  sum against 4.9 at an attention one); the attention sum added by the experts' HyperConnection as it
  reads the streams, its part sent by the projection's last block (every block read the inbox at system
  scope: a pass 0.2 ms longer); a HyperConnection's stages in one launch of resident blocks with a grid
  wait (a wait costs 1.7 µs less than a launch boundary here, and passes of one or two rows were 0.25 ms
  shorter, but served, kernels on other streams left fewer blocks room than the launch's and one-row
  passes took 16.5 ms); a split of the routed experts by co-routing (the larger share a token 6.22 to
  5.98 of 5, held out: worth about 1%); a head's read-back written by a kernel into mapped coherent host
  memory, with no copies queued after it (6 µs a head, no change in a pass); the host's waits spinning
  longer before they sleep (`ROC_ACTIVE_WAIT_TIMEOUT=50000`, no change: they wake as soon as it is); a
  verification's routes by expert, each expert its tokens share read once (four tokens' 40 routes reach
  27.7 experts): with every token's sums 4-row passes took 2.7 ms longer, with only its routed tokens'
  (a mask of active tokens in the units' dots) still 1.2-1.4 ms; the tokens' blocks of a route rank side
  by side for L2 (no change), and every token reading one input row (no change: the experts are not held
  by their inputs); the IQ2_S layers' routed gate and up on INT8 activations (0.06 / 0.15 ms at 1 / 4 rows
  for 20 layers), and the HyperConnections on INT8 activations (slower than their F32 kernels at every
  row count).
- **Where a pass goes now** (launched twice in turn, 20K, ms a pass at 1 / 2 / 3 / 4 rows): each
  HyperConnection 0.52 / 0.74 / 0.95 / 1.16 (two a layer; at one row its bandwidth floor), the
  in-projections 0.61 / 0.77 / 0.81 / 0.91, the out-projections 0.34 / 0.41 / 0.44 / 0.46, the router
  and its route 0.48 / 0.56 / 0.64 / 0.70, the routed gate and up 0.79 / 1.41 / 2.03 / 2.61 and down
  0.57 / 0.89 / 1.24 / 1.52 (at one row some 360 GB/s: five routes a GPU are too few waves).

**Decode.** A pass runs a token and its MTP drafts as the rows of one verification through every
layer (§6.2).
- **Placement of a pass** with the layers whole (`STRAYDOG_PARTS=0`). The first GPU owns the
  layers up to the 25th, the second the rest, the head and the MTP layer. A layer's routed experts
  are split in halves between the GPUs, so the one that does not own the layer computes its half of
  the experts and otherwise waits: a third of each GPU's pass is spent in the kernels that wait for
  the other.
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
  F32 input. `STRAYDOG_Q8=0` keeps every projection on F32. On CUDA the same kernels run with
  `__dp4a` (gpu/gemv.hpp: the quantization, the units and the order of every sum are the RDNA4
  kernels'; only the integer dot is the device's, and the lanes are twice the F32 plan's). They
  replace F32 kernels that were held by their load instructions and L1 wavefronts, not by the
  memory: on an RTX 5070 Ti the dense front of a DeltaNet layer (qkv, gate, beta, alpha) takes 40
  microseconds where it took 56 (the card streams that much in 38), the 2560 x 6144 out
  projections 14.6 and 19.3 where they took 31 and 39. A Q8 row's sum does not depend on what
  shares its launch, so a verification pass still equals token-by-token decode, but it is the
  quantized input's sum, not the F32 kernels': KL mean 1.1e-3, p99 1.2e-2 against them (UD-IQ1_S,
  356 decode steps). DeepSeek-V4.1's dense GEMVs stay on the F32 kernels on CUDA (its Q8 switch,
  `DS41_Q8`, has no CUDA kernels behind it yet: the dense kernels take its formats, but turning
  them on there is a change to `sd_decode_options` and a quality run on a DeepSeek bundle).
- **Routers as BF16.** Unsloth's bundles store Qwen3.8's routers (and shared-expert mixes) in F32,
  but every value is a BF16 value, the model's own weights widened. Such a router goes up as BF16
  (`Weights::narrowed`, `Data::Narrow`): the same values in half the bytes, 251.7 of the MB a decode
  step reads on its 48 layers, and the GPU-side planning counts it so (UD-IQ1_S at 5.5 GiB: 953 ->
  1040 expert slots). The decision is made from the mapping; the upload converts the verified copy
  and refuses a value that is not BF16 (`gpu::narrow_bf16`). GSQ's bundles store them as BF16
  already, and this is their path. The router's sums are BF16 rows' now, so its bits are not the F32
  kernel's.
- **HyperConnections in Q8_0.** The HyperConnections' down and up projections (BF16 in the bundle,
  1.26 GB of the 5.2 GB a row reads) go up quantized to Q8_0 at load, as llama.cpp quantizes:
  each 32 values' scale max |x| / 127. A decode pass reads half their bytes (plain decode
  64.5 to 67.3 tok/s; a 1-row pass 16.8 to 16.2 ms at 20K, a 4-row one 27.2 to 26.5 ms), greedy
  tokens part from the F32 run about as early as before, and 0.6 GB of VRAM comes free.
  `STRAYDOG_HC_Q8=0` keeps them BF16.
- **The decode head in one wait (CUDA).** A plain decode step's head was the HyperConnection mix (a launch, a wait and the mixed row
  copied to the host), then the projection (the row copied back, a launch, a wait, and 248320 logits into a vector the host zeroed,
  scanned for non-finite values and copied again): 1.11 ms a token on an RTX 5070 Ti, of which the Q6_K head's GEMV is 0.63 (521 MB
  at 829 GB/s, as fast as a plain streaming kernel reads the same bytes: the kernel was never what held it). `sd_decode_head`
  launches the mix, the projection and a non-finite check of the logits one after the other and waits once: 0.89 ms, the logits the
  two calls' bits (`decode_head_gpu`). The rest is the launches' and the two copies' latency under WDDM, about 150 us, and the copy of
  the logits out of the staging memory into the vector, 50 us; the vector is the step before's, given back by the draw. HIP keeps
  the two calls. `examples/head_bench.rs` times the GEMV alone.
- **Prefill batch.** On two R9700s the resident Qwen3.8 reads a prompt in batches of 4096, from 48 GiB of
  budgets in all (`resident_batch`). On 2026-10-29 4096 rows took the first token of a 20K and a 60K prompt
  7-12% longer than 2048 (5.7 to 6.5 s, 15.8 to 17.2 s), and the derived batch was 2048 until
  2026-11-12, when served runs (one each, two R9700s, UD-IQ3_XXS, 262144 context) read 8K in 1.75 s
  instead of 1.95, 32K 6.81 instead of 7.10, 95K 21.63 instead of 22.83 and 242K 62.6 instead of 66.1,
  decode the same. The derived batch is the placement's to narrow: K/V in VRAM at 4096, then at 2048, then K/V
  in host memory at 4096, 2048 and 1024 (`placement_candidates`), so a model that fit with its K/V in
  VRAM at 2048 still has it there; a model that is not fully resident, on 48 GiB of budgets or more,
  keeps 2048 for its expert slots' room. A batch given with `--batch-size` is never narrowed that way:
  the server sends the worker `auto` for the derived batch (`Settings::batch_derived`), and the engine
  takes `DERIVED_BATCH` (0) as the batch to derive and narrow, any other as asked, which is tried at
  its rows with the K/V in VRAM, then in host memory at the rows halved to 1024 (as before 4096 was the
  default; the `one-gpu-partial` golden's 4096 on a 24 GiB card stays 4096).
- **A long read's wider batches.** A read of `WIDE_READ_TOKENS` (16384) new tokens or more is read in batches of up to
  `Model::wide_batch_size` rows when the batch was derived, the layers are in parts (§8 above) and the VRAM left holds the
  workspaces of more (`wide_placement`; the ready line says the rows, or what they lack, and the request line the batches it
  ran in; `STRAYDOG_PREFILL_WIDE=off` or N rows). What a batch pays once, whatever its rows, is the FP16 expansion of every
  dense weight (`expand_matrix`, 25.4 us a token at 2048 rows in the 242K kernel table, halving with each doubling), the
  exchanges of the two devices (a wait of 56-89 us each) and the launches. What grows with the rows is the decoder's workspace
  (hidden, mixed and branch rows, 15364 floats), the MTP layer's kept rows and embeddings (12800) and the sums' inbox; the
  experts' workspace (11 routes of 4480 floats a row) does not, because a layer part runs the experts of a wide batch in
  chunks of 4096 rows (`moe_chunk_rows`; a batch of no more is one chunk, as ever, and its bits are as they were) and the
  workspace holds only the other stages' row scratch (13000 floats a row of a part) beyond a chunk. A chunk reads every expert
  of the layer once, so a width between two chunks gains nothing and costs the experts' second read over fewer rows: two
  R9700s, `--vram 30`, a fresh 242K read, 50.0 s at 4096 rows, 50.97 s at 5120 (2 x 2560), 50.05 s at 6144 (2 x 3072),
  49.1 s at 8192 (2 x 4096; 48.9 after a pause, three reads in a row 50.5, 49.12, 49.13), 95K 18.4 and 18.1 s (4096, 8192),
  32K 6.0 s at 8192 once its staging is pinned. Widths are therefore the multiples of 4096: 12288 and 8192.
  What a row of a batch cost in VRAM on each device was 350908 bytes: the experts' workspace 197208 (49302 floats), the decoder's
  61456, the MTP rows 40960 and embeddings 10240, the sums' inbox 20480 (two slots of a row's 2560 floats) and the peer link's
  20564 (inputs, partials and routes), 1.34 GiB at 4096 rows and 2.68 at 8192. Beside the weights, with the drafter and K/V in
  host memory, the placement had 145 MB left at 4096 rows (6144 rows lacked 0.08 GiB, 8192 0.33, 12288 1.16, 16384 1.98): the
  K/V stage, which grows into what is left and decode reads its prefixes from, took it. Three workspaces were larger than
  what they hold. The sums' inbox needs the halves of a batch's rows (2568 FP16 values a row, half of them each: 5136 bytes a
  row, `sums_capacity`) and the shared indexer's selections of a span (a 2048-row span's 512 blocks of 16 bits): 21 MB at 4096
  rows, not 84. The peer link of layers in parts carries only a decode pass's rows (its split output head's half of the
  vocabulary for eight rows is 388 rows of the inbox): 1024 rows, 21 MB, not 84. The experts' workspace of a layer in parts
  needs 32982 floats a row (`moe_compact_row`): its prompt batches run the grouped FP16 path, whose products and inputs are halves
  in 4800 of the gate and up scratch's 14080 floats, with no F32 product rows (7040), so 16320 floats a row were never
  read: 515 MiB at 4096 rows, not 771. The same bits (every digest of a batch up to 4096 rows is as it was, layer split included)
  and 0.54 GiB left at 4096 rows instead of 145 MB: the stage grows into it (decode of 200 tokens after a 95K read, 22.21 ms a
  pass with the stage at 625 MB, 21.81 with 1059; the passes after shorter prompts the same). 8192 rows take 482 MB of it (the
  decoder's 252, the MTP's 210, the sums' 21), which leaves the stage 97 MB beyond its start (decode at 95K 22.09 ms a pass, as
  it was); 12288 rows lack 0.49 GiB and 16384 rows 1.14. A short read, an image, the warm-up pass, a batch given, partial
  residency, one GPU, the layer split and DeepSeek are not widened. The staging buffers of the rows (two page-locked buffers each
  for the MTP layer's, the first layer's and the PLE layer's embeddings, 84 MB each at 8192 rows, a GiB over the two devices) are
  pinned on a thread of each device from the end of the load (`sd_row_staging_prefetch`), at the size of the widest batch, and the
  first batch that stages rows takes them. The first wide read otherwise grew them as its batches asked: 0.06-0.08 s for each of
  the six pairs, and the MTP layer's asked 8191 rows in its first batch and 8192 in the next, so it grew once more: device 0's
  growths of it took 1.34 s each, with a batch queued on that device (device 1's 0.07-0.08 s). A 32K read after an 8K one in the example
  took 6.30 s the first time and 5.62 the second, now 5.64 and 5.62; a served 32K read 6.47 s, now 5.77. Pinned beside the
  weights' reading (the first try) it held the load up by 0.3 s, so it starts once they are read, and the warm-up's first batch
  waits for it. A 242K read ran in 49.1 s. Of the kernels' time of a 242K read (400.8 us a token at 4096 rows, both
  devices, 394.3 at 8192) the weights' expansion went from 14.4 to 9.0 us, the devices' signals from 15.3 to 11.3 and the large
  GEMMs from 71.1 to 68.8; the casts took 1.5 more, the experts' kernels 0.9, the residual updates 1.1.
- **Fewer, fuller launches in a pass** (from a kernel trace at 20K). Every Q8 unit of Q4_K, Q6_K and IQ4_XS is
  staged with Q5_K's pad: a row's lanes read its units 128 or 64 bytes apart, on the same LDS banks, so four
  vectors took twice one's time (a 2560 x 6144 Q6_K out projection 50 to 29 us at four rows). The shared
  expert's mix goes up as the BF16 row it is, one launch with the router. The rows to the helper and its
  flag are one single-block kernel. A partial acceptance replays every DeltaNet layer of a span in one
  launch (a launch a layer was ~330 us ahead of the drafting). The drafter's dense projections but its
  router (BF16, a fifth of a draft step) go up as Q8_0, as the HyperConnections do; only the drafts
  depend on them. A pass at 20K: 1 row 16.1 to 15.4 ms, 4 rows 26.5 to 24.3.
- **One wait a verification.** The first layer, the PLE layer and both spans are queued one after the other,
  the rows handed GPU to GPU (sd_prefill_handoff), and the head reads them where the last span left them and
  hands them back with its tokens: one host wait where there were four. With a cut that keeps at most 64
  (the card's top_k 20, no penalty), the head picks each row's top_k on the GPU (value first, the lower token
  on a tie, as the host's scan orders them) and the host draws among those: the same distributions and
  draws, at a seed the same tokens, without 2 MB of logits a pass. The PLE key and value projections go up
  as Q8_0 too, a row read thread is woken a job (all 128 for a decode row's 16 rows cost the pass ~0.1 ms),
  and two waves share an HC row's units. A pass at 20K: 1 row 15.4 to 15.2 ms, 4 rows 24.3 to 23.8.
- **Kernels the trace showed slow.** The HC row back on one wave (two waves' sums made each block take the
  stream's mean squares twice: 16.7 to 20.6 us at four rows), and for one or two rows 32-row blocks that stage
  the stream's x * norm in LDS once (12.5 to 9.4 us at one row; at three or four the 30-40 KB left a CU one
  block). A decode or verification row's block selection takes the register-held 1024-thread kernel the long
  prompt chunks take (41 to 37 us at 20K). The head's top-k is found by each wave in shuffle rounds and then
  by the block's first wave (a chunk 42 us with a block-wide best each round, two barriers apiece). A pass:
  20K 1 row 15.3 to 14.8 ms, 4 rows 23.8 to 23.4; 60K 4 rows 28.3 to 27.2.
- **Expert kernels of a format.** The decode stages run the kernel of the layer's routed format (gate and up
  alike), with the shared expert's down rows of IQ4_NL or Q8_0 beside: 70-118 VGPRs where the kernel of every
  format took 186 (eight waves a SIMD), so a gate or up row takes 8 lanes (10 units each, not 20 for 4 lanes):
  1 row 26.9 to 19.9 us. The shared expert's gate and up (whose F32 or Q8 rows in that kernel took 204) run on
  the Q8 GEMV on a side stream, forked and joined by events (graph branches). A pass: 20K 1 row 14.8 to 14.3
  ms, 4 rows 23.3 to 22.5; 60K 4 rows 27.2 to 26.4.
- **K/V stage grown after loading.** The stage that mirrors every host K/V state's prefix in decode was a
  sixteenth of the budget, 37K to 43K positions a layer (split over six or seven states), so a 60K decode's
  attention read the positions past them over the bus. After loading (and any expert slots) it takes the VRAM
  left beside the placement's slack, up to every state's whole context: at `--vram 30` 87K positions on the
  first GPU and 51K on the second. A pass at 60K: 1 row 15.7 to 14.9 ms, 4 rows 26.4 to 23.6 (62K: 31.1 to
  26.4); 20K and the first tokens unchanged.

### 8.1 Images

The GGUF repository's `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (encoder and projector, one file for every bundle, pinned by
size and SHA-256) sits in the bundle's directory or the one above it, or is named by `--mmproj`; `straydog setup` fetches it.
The worker hashes it at the start through the page cache, so none of it stays in memory until an image comes. A wrong or
missing file leaves the model serving text: images are refused with what is wrong (`/v1/models`' `meta.vision`, `/props`'
`modalities.vision` say whether it reads them).

How an image is read (`qwen_vision.rs`, after transformers' `qwen3_5` vision model, `Qwen2VLImageProcessorFast` and llama.cpp's
`qwen3vl`):
1. **Size.** `smart_resize` with factor 32: at least 65536 pixels, at most 4096 merged tokens (llama.cpp's limit; the
   processor's own 16777216 pixels would be 65536 patches). Bicubic as Pillow does it, values `(x - 127.5) / 127.5`.
2. **Span.** The template's image part is a marker of its own (§7.5). Its place takes `<|vision_start|>`, one position a merged
   token, `<|vision_end|>`, with ids from 2^31 up made from the image's digest: a held sequence continues only with the
   same image, and the rows of the last images are kept, so a later turn re-reads nothing.
3. **Encoder**, on the CPU, every thread: patches in merge-block order through the two temporal filters (a still image is
   two equal frames), the position table bilinearly with corners aligned, 27 blocks of full attention with axial RoPE (theta
   10000, row on the first 18 pairs of a head's 36, column on the rest), and the merger. Each product reads the BF16 weights
   where they are mapped and sums in F32: the mapping goes when the image is encoded.
4. **mRoPE.** A token has (t, h, w): text one position on all three, an image's tokens the position where the image starts,
   plus their row and column. The next text goes on from there by the grid's longer side, so past an image a token's rope
   position is its K/V index less a delta (`rope_of`). Each rotated pair takes its axis by its index mod 3, which on text is
   the one position every kernel always used. The K/V, the blocks and the history go by the index alone.
   - A decode or verification row is text: beside the GPU position scalar lies its rope position, the same number unless a
     delta is set (`sd_rope_delta`). A replay reads both from memory. Text-only passes run the same kernels and launches.
   - A prefill batch of a sequence with images reads a table of three a row (`sd_rope_rows`), with the row of the first
     position of the block its first row is in: a pooled block turns by its first token's position, which may lie in an
     image or in the last batch. Only such batches launch `qsa_normalize_axes` and `qsa_pool_axes`.
5. **The rest of a pass.** PLE's n-grams hash `<|image_pad|>` at an image's positions, as the reference reads the ids before their
   rows go in. Routing is as for any row. A batch over image rows runs as a batched prefill even when short.
6. **Drafter.** It works over images as over text. Its layer pairs each hidden row with the row of the token after it, at an
   image's position the image's own row (llama.cpp's MTP graph takes `inp->embd` there), at the same rope positions as the
   model's layers: a table in batches, the delta in the reply's rows. So a conversation holding an image keeps its speed.

## 9. Command line, configuration and setup

With no subcommand, `straydog` is the server, as llama-server is. Every setting is an option,
named as llama-server names it where it has the same setting.

| Option | Meaning | Default |
| --- | --- | --- |
| `-m, --model DIR` | model bundle directory, or its first shard | the settings `setup` wrote |
| `--model-name NAME`, `--provider NAME`, `--quant NAME` | the bundle's three labels, where the directory alone does not say | detected from the pinned files |
| `-md, --model-draft DIR` | Qwen3.8's MTP drafter | none |
| `--mmproj FILE` | Qwen3.8's vision encoder and projector (§8.1) | a `mmproj-*.gguf` in the model's directory or the one above it |
| `--draft-max N` | most MTP drafts per pass (0 turns MTP off) | 3 with a drafter |
| `-c, --ctx-size N` | context in tokens, at most the model's own: Qwen3.8 262144, DeepSeek-V4.1 1048576 | 262144 |
| `-b, --batch-size N` | prefill batch, 1-4096; DeepSeek-V4.1: up to 32768 | 1024; 4096 with 48 GiB or more of GPU budget; DeepSeek-V4.1: from its GPUs' budgets (§7.2) |
| `-gpu, --gpu LIST` | GPU ordinals (`-dev`, `--device` also accepted) | `0` |
| `-sm, --split-mode MODE` | `experts`: every layer split across the GPUs; `layer`: ranges of layers | `experts` |
| `--vram GIB[,GIB...]\|auto` | VRAM budget for every GPU, or one per GPU (`30,10`) | `auto` (§5) |
| `--state-ram GIB` | host memory for sequence state | what the context needs |
| `--row-cache MIB` | row cache of the PLE and Engram tables; 0 reads them through the page cache | 256; Qwen3.8's PLE table: as the host memory allows (§4.2) |
| `--expert-ram GIB` | host tier for experts outside VRAM | 85% of host memory less Straydog's other host memory |
| `--data-dir DIR` | verity trees and warm-up state | `.straydog` in the model directory |
| `--no-prefix-cache`, `prefix_cache = false` | Windows: keep no state of the first request's system block on the drive (§6.4) | the state is kept |
| `--keep-conversation`, `keep_conversation = true` | Windows: when the server stops on purpose, keep the state of the conversation held last, sealed for the user, for the next start (§6.5) | off |
| `--stop-on-stdin` | stop on a line `stop` on standard input, as Ctrl+C stops (§6.5) | off |
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
device's layers, the head, the commit). Partially resident, it also gives a token's routes by where
they ran and its swaps, the slots, and what of the layers' time the host waited for routes and spent
computing them.

**Subcommands.**
- `straydog setup`: the first run.
- `straydog fetch -m DIR [--model-name NAME] [--provider NAME] --quant NAME`: download, resume and authenticate a
  bundle, with its extras; the labels that name one bundle will do.
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
   - The first model, in the order DeepSeek-V4.1, Qwen3.8, with a bundle that VRAM and
     RAM hold alone is recommended, with its best bundle in that class. Failing that, the bundle
     that reads least from the drive is.
3. It asks where the models go. The default is the platform's data folder:
   `$XDG_DATA_HOME/straydog/models` or `%LOCALAPPDATA%\Straydog\models`. The bundle goes in
   `<dir>/<model>/<quant>`.
4. It checks the free space and asks which extras to add: DSpark and the vision encoder for
   DeepSeek-V4.1, MTP and the vision encoder for Qwen3.8.
5. It fetches as `fetch` does.
6. It writes `config.toml` in the platform's config folder. The file is written whole and
   renamed, keeping the earlier one as `config.toml.old`.

A plain `straydog` reads that file when no `--config` and no model are given. A server started
with a model of its own reads no file it was not given.

## 10. API

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
  - `suppress`: `not_cp932_han` (§6), listed in `/props` and `/v1/models`' `meta`;
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
    disconnects is cancelled. The worker ends it when its current pass ends. Only a worker that
    has not ended it after 10 minutes counts as hung, which ends the server.
  - Every client that holds the key shares the engine and its cache, so whether a prompt prefix
    was seen before is observable. Users who must not see each other's prompts need separate
    servers.
- **Dashboard.** `GET /dashboard` is a page compiled into the executable, with a strict CSP and no
  other origin. English by default; the top bar's switch shows it in Japanese (kept in the browser's
  local storage), and `?lang=ja` or `?lang=en` on the page's URL chooses it, as an app that knows its
  user's language opens it.
  - It reads `GET /dashboard/status?since=T`: settings, engine state and placement, the running
    and the latest 120 generations with their speeds and speculative acceptance, and host,
    process and GPU sensors.
  - It reads `GET /dashboard/logs?after=N`, and chats through `/v1/chat/completions`.
  - Sensors are read only while a dashboard is open: amdgpu's sysfs, or NVML in CUDA builds.

### 10.1 Timing and cache fields

`timings` is engine-specific; clients must allow fields to be absent. Qwen3.8 reports the
following prefill measurements. Times are seconds, and setup, restoration and early cache
filling are already included in `prefill_seconds`.

| Field | Meaning |
| --- | --- |
| `prefill_seconds` | Time spent prefilling this request's new prompt tokens, including its setup, restoration and queued work settling. |
| `prefill_prepare_seconds` | Long-prefill setup: up-front K/V growth and borrowing of expert-slot VRAM. |
| `prefill_restore_seconds` | Restoring the original prefill workspaces and rebuilding the expert pool after borrowed prefill. |
| `prefill_cache_fill_seconds` | Filling an initially empty expert cache between prompt batches; absent when no such fill ran. |
| `prefill_batch_rows` | The selected batch row limit. Checkpoints and the prompt's end can make a chunk smaller. |
| `rebalance_seconds` | Choosing and filling expert slots after successful prefill, before decode. This is outside `prefill_seconds`. |
| `partial_detail.slot_capacity` | Total expert-cache entries allocated across the selected GPUs, excluding staging slots. |
| `partial_detail.slots_held` | Cache entries assigned to experts, including fills still pending. The legacy `partial_detail.slots` also reports occupancy. |

Capacity and occupancy are separate: a newly allocated cache can have capacity without holding
experts. A restored pool can change capacity when K/V, workspaces or available memory change.
`usage.prompt_tokens_details.cached_tokens` counts reused prompt positions; it is independent
of this expert-weight cache.

The engine's `ttft_seconds` includes work after prefill, such as its final rebalance. A client's
time to first visible text can also include startup warm-up waiting, queueing and transport.
Compare the same backend, quantization, context, GPU budget and batch, and record the initial
expert-cache state and page-locked layer count when comparing startup with a warm request.

## 11. Build and distribution

- `cargo build --release --features hip` (or `cuda`). `build.rs` runs CMake on `gpu/` with the
  vendor's compiler and links the result statically. The architectures come from
  `STRAYDOG_GPU_ARCHITECTURES` (`gfx1201`; `75-real;86-real;89-real;120`).
- A GPU without code in the executable fails at startup, with its architecture named.
- Release artifacts: `straydog-linux-x86_64-hip` (needs the ROCm 10 runtime) and
  `straydog-windows-x86_64-cuda.exe` (needs the NVIDIA driver and `cublas64_13.dll`). The
  executable's SHA-256 is published with each release. The model pins are compiled in, so a
  binary authenticates exactly the bundles it was tested with. Linux with CUDA is built the same
  way and is not a release artifact yet.

## 12. Decisions

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
- **The settings file `setup` writes is read only when nothing names a model.** A process manager, or any
  script that names its model, is unaffected.
- **DeepSeek-V4.1's misses are uploaded, not computed on the CPU,** and the store is LRU (§5).
  Both were measured against the alternatives.
- **DeepSeek-V4.1's template is Straydog's own** (§7.1): a template that misrenders tool calls
  costs every turn a full prompt.
