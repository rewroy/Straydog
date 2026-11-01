/* The Straydog GPU library's C ABI: the only interface between the engine host and the
 * C++/HIP/CUDA code. A negative int result or a null pointer is a failure, which
 * sd_error() (sd_cpu_error() for the CPU expert engine) then describes for the calling
 * thread. Other int results are 0, or a query's answer (sd_device_count, the *_supported
 * checks, sd_peer_direct). Handles are opaque. */
#ifndef STRAYDOG_GPU_H
#define STRAYDOG_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Errors and devices */
const char* sd_error(void);
int sd_device_count(void);
int sd_device_info(int ordinal, char* name, size_t capacity, uint64_t* free_bytes, uint64_t* total_bytes);
/* Where the device sits on the PCI bus: domain, bus and device numbers (function 0). */
int sd_device_pci(int ordinal, int* domain, int* bus, int* device);
void* sd_create(int ordinal, uint64_t budget);
void sd_destroy(void* handle);
int sd_memory(void* handle, uint64_t* used, uint64_t* budget);

/* Allocations and transfers */
void* sd_upload(void* handle, const void* data, uint64_t bytes);
/* An allocation whose bytes arrive later: sd_write stages caller memory into all of it;
 * sd_fill copies page-locked host memory (sd_host_register) into a range of it, on the
 * device's stream, from any thread that owns a filler of the device, while nothing else uses
 * the device. */
void* sd_allocate(void* handle, uint64_t bytes);
/* Keep the FP16 expansions of uploaded weights (sd_upload's: never written after) that a prompt
 * pass's GEMMs take, in a ring of `bytes` (an allocation within the budget; 0 keeps none), for the
 * pass's next batches. Once only. */
int sd_expansions(void* handle, uint64_t bytes);
int sd_write(void* handle, void* pointer, const void* data, uint64_t bytes);
void* sd_filler_create(void* handle);
void sd_filler_destroy(void* filler);
int sd_fill(void* filler, void* pointer, uint64_t offset, const void* host, uint64_t bytes);
/* sd_fill of `height` rows of `width` bytes, `host_pitch` apart in host memory, to rows `pitch` apart from `offset`. */
int sd_fill_rows(void* filler, void* pointer, uint64_t offset, uint64_t pitch, const void* host, uint64_t host_pitch, uint64_t width, uint64_t height);
int sd_release(void* handle, void* pointer);
int sd_zero(void* handle, void* pointer);
int sd_download(void* handle, void* pointer, void* output, uint64_t bytes);
int sd_copy_allocation(void* handle, void* target, const void* source);
int sd_host_register(void* pointer, uint64_t bytes);
int sd_host_unregister(void* pointer);
void* sd_host_state(void* handle, uint64_t bytes);
int sd_host_states_prefetch(void* handle, const uint64_t* sizes, uint64_t count);
// A decoder layer's host K/V grown to `capacity` positions, its `valid` positions kept, and its indexer laid out for them in
// its allocation; the new K/V's pointer in `state`. Its plans and captured graphs follow.
int sd_kv_grow(void* handle, uint64_t layer, uint64_t capacity, uint64_t valid, void** state, void** index_state);
int sd_prefill_buffers(void* handle, uint64_t capacity, uint64_t moe_bytes, uint64_t decoder_bytes, void** moe, void** decoder);
int sd_kv_scratch(void* handle, void** scratch);
int sd_host_state_supported(void* handle);
int sd_kv_stage(void* handle, void* pointer, uint64_t rows);
void* sd_checkpoint_buffer(void* handle, uint64_t bytes);
int sd_checkpoint_copy(void* handle, int save, void* const* states, uint64_t count, void* buffer);
int sd_state_copy(void* handle, void* state, uint64_t offset, uint64_t pitch, uint64_t width, uint64_t height,
                  float* host, uint64_t first, int save);
/* Asynchronous copies: page-locked host memory (sd_host_register, or sd_host_pin where devices
 * also write it) to and from allocations, beside the device's compute stream (stream 0): uploads
 * on stream 1 (and stream 3, a second upload queue for copies that may wait), downloads on
 * stream 2. sd_event_record marks the work queued so far on a stream and returns the mark's id
 * (ids grow from 1); sd_event_wait makes a stream wait for a mark on the GPU; sd_event_done
 * answers 1 once it completed (block: wait for it), after which the copies before it are done
 * with their memory. A copy into memory another stream still reads must wait for a mark after
 * that use. Counts: uploads (both queues), their bytes, downloads, their bytes. */
int sd_host_pin(void* pointer, uint64_t bytes);
int sd_transfer(void* handle, int stream, void* pointer, uint64_t offset, void* host, uint64_t bytes);
int64_t sd_event_record(void* handle, int stream);
int sd_event_wait(void* handle, int stream, int64_t id);
int sd_event_done(void* handle, int64_t id, int block);
int sd_transfer_counts(void* handle, uint64_t* counts);

/* Matrix products */
int sd_gemm_options(void* handle, uint64_t minimum_batch, int half_precision);
// Decode options (RDNA4 only), flags: 1 a decode row's Q5_K/Q8_0 GEMVs and IQ2 experts' gate/up on the Q8
// kernels; 2 a decode or verify pass's index scores and attention scores through WMMA (FP16 operands, F32 sums).
int sd_decode_options(void* handle, int flags);
int sd_gemm_counts(void* handle, uint64_t* calls, uint64_t* vectors);
int sd_matvec(void* handle, void* pointer, int kind, uint64_t rows, uint64_t cols, const float* input, float* output);
int sd_matvec_batch(void* handle, void* const* pointers, const int* kinds, const uint64_t* rows, const uint64_t* cols, uint64_t count, const float* input, uint64_t input_count, float* output, uint64_t output_count);

/* Ops: generic operations over device buffers, queued on the device's stream (shapes are
   arguments; every pointer must lie inside one of this device's allocations). A buffer starts
   zeroed; reading one waits for the stream. Linear: out[t][r] = W[first_row + r] . in[t] (strides in values, 0: packed),
   by GEMV for up to four inputs, else an F32 GEMM over rows expanded into scratch. Rope rotates the last
   rope_dims values of each of a row's vectors, in adjacent pairs, by positions[row] * frequencies. */
void* sd_buffer(void* handle, uint64_t bytes);
int sd_buffer_write(void* handle, void* buffer, uint64_t offset, const void* data, uint64_t bytes);
int sd_buffer_read(void* handle, const void* buffer, uint64_t offset, void* output, uint64_t bytes);
/* An asynchronous read (up to 1 MiB): start queues the copy behind the work queued so far; finish
   waits for that copy alone and returns the bytes. One read is outstanding at a time. */
int sd_buffer_read_start(void* handle, const void* buffer, uint64_t offset, uint64_t bytes);
int sd_buffer_read_finish(void* handle, void* output, uint64_t bytes);
/* A copy from an allocation of one device to one of another (or the same), after the work queued
   on the source; the target's later work waits for it, and so does the source's. */
int sd_buffer_copy_peer(void* to, void* target, void* from, const void* source, uint64_t bytes);
int sd_op_linear(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row, const float* input, uint64_t input_stride, uint64_t steps, float* output, uint64_t output_stride, float* scratch, uint64_t scratch_bytes);
/* The same shape and GEMV for up to four rows; larger products explicitly use F32(0) or
   FP16 inputs/weights with F32 accumulation/output(1), without changing the device policy. */
int sd_op_linear_gemm(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row, const float* input, uint64_t input_stride, uint64_t steps, float* output, uint64_t output_stride, float* scratch, uint64_t scratch_bytes, int half_precision);
/* Optional quality-only observers; start once before any graphs. Read synchronizes the stream.
   Eight counters: nonfinite F32 inputs, nonfinite half inputs, finite-input overflow, input
   underflow-to-zero, nonfinite half weights, input count, weight count, max finite input abs bits.
   Disabled devices launch no observer kernels. */
int sd_gemm_check_start(void* handle);
int sd_gemm_check_read(void* handle, uint64_t* counters, uint64_t count);
/* Graphs of ops: between sd_graph_begin and sd_graph_end (keep: store it under key, replacing
   any; else drop it) the ops queued on the device's stream are recorded rather than run; only
   ops with no host data (no buffer writes, no expert routes) may be recorded. sd_graph_run
   queues a recorded graph: 0, or 1 when none is stored under key. */
int sd_graph_begin(void* handle);
int sd_graph_end(void* handle, uint64_t key, int keep);
int sd_graph_run(void* handle, uint64_t key);
/* Several linear ops in one launch where the GEMV takes them (up to four inputs, and strides shared
   when there are several): out = W[first_row ..] . in per item; up to 24 items. */
typedef struct sd_linear {
    const void* weights;
    int kind;
    uint64_t rows, cols, first_row;
    const float* input;
    float* output;
} sd_linear;
int sd_op_linears(void* handle, const sd_linear* items, uint64_t count, uint64_t steps, uint64_t input_stride, uint64_t output_stride, float* scratch, uint64_t scratch_bytes);
/* DSpark's greedy Markov chain, 1..5 slots. W1/W2 are BF16 [vocabulary,rank], conf is
   F32 [width+rank]; body [slots,vocabulary], hidden [slots,width]. Outputs are corrected
   logits, tokens, confidences and their usable prefix count (0 below minimum).
   With body_tail, the body's columns from tail_first on are [slots, vocabulary - tail_first] there
   and body holds the columns below (rows of vocabulary).
   All inputs/outputs/scratch must be disjoint. Scratch: slots * align16(4*(width+rank)) + 512 bytes.
   sample_at (device, 4 x uint32; null: greedy) makes each slot's token a draw rather than the argmax: with the
   float bits of an inverse temperature (0: greedy again) in word 0 and a 64-bit seed in words 1-2, token i is
   a draw from softmax(corrected logits * inverse temperature) (Gumbel-max, noise from the seed, the slot and the
   column), and the next slot's Markov features follow the drawn token. The parameters are read when the op runs,
   so a recorded graph follows them as it does anchor_at.
   probabilities (device, slots floats; null: none; needs sample_at) gets each drawn token's probability under the softmax it was
   drawn from, the reciprocal of the sum over the corrected logits of exp((logit - the token's) * inverse temperature); 0 if the
   token or the sum is not valid.
   Reference defaults: threshold=0, minimum=0. No host readback; graph-capturable. */
int sd_op_dspark_head(void* handle, const float* body, const float* body_tail, uint64_t tail_first, const float* hidden,
                      const void* w1, const void* w2, const float* conf, uint64_t slots, uint64_t vocabulary, uint64_t rank,
                      uint64_t width, uint32_t anchor, const int32_t* anchor_at, const uint32_t* sample_at, float threshold, uint64_t minimum, float* logits,
                      int32_t* tokens, float* confidences, uint32_t* count, float* probabilities, float* scratch, uint64_t scratch_bytes);
/* Decode fusions (one vector): sd_op_norm_linears stores the RMS norm of x (weight, eps) in
   normed (width floats) and runs the linears over it (every item's input is normed, its columns
   width), in one launch when they share a kind the GEMV takes; sd_op_swiglu_linear sets out = W .
   SwiGLU(gate, up) (cols floats each; hidden takes the SwiGLU where it is formed apart), each row
   times sigmoid(*scale) when scale is given. The results are those of the separate ops. */
int sd_op_norm_linears(void* handle, const float* x, const float* weight, uint64_t width, float eps, float* normed, const sd_linear* items, uint64_t count);
/* One row's norm and rope, as sd_op_norm_rope makes them for rows 1 (input, weight, width, eps,
   rope_dims with a position and its frequencies, out_rows: its row of out, else row 0). */
typedef struct sd_norm_rope {
    const float* input;
    const float* weight;
    uint64_t width;
    float eps;
    uint64_t rope_dims;
    const int32_t* positions;
    const float* frequencies;
    const int32_t* out_rows;
    float* out;
    uint64_t out_count;
} sd_norm_rope;
/* sd_op_norm_linears, and side's norm and rope in the same launch (a block of its own). */
int sd_op_norm_linears_rope(void* handle, const float* x, const float* weight, uint64_t width, float eps, float* normed, const sd_linear* items, uint64_t count, const sd_norm_rope* side);
/* One vector through `items` (sd_op_linears, every item reading the same input), then the RMS norm
   of x (one of their outputs) into normed and side's row as sd_op_norm_rope forms it: the results of
   the three ops, in one launch where it can. */
int sd_op_linears_norm_rope(void* handle, const sd_linear* items, uint64_t count, const float* x, const float* weight, uint64_t width, float eps, float* normed, const sd_norm_rope* side);
int sd_op_swiglu_linear(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* gate, const float* up, const float* scale, float* hidden, float* out);
/* sd_op_swiglu_linear, and meanwhile (one launch where the GEMV takes it) the routes of one row of
   256 router logits as sd_op_route picks them with softmax gating, normalized, no bias, scale 1:
   the same results as the two ops. */
int sd_op_swiglu_linear_route(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* gate, const float* up, const float* scale, float* hidden, float* out, const float* logits, uint64_t k, int32_t* ids, float* route_weights);
int sd_op_rms_norm(void* handle, const float* input, const float* weight, uint64_t rows, uint64_t width, float eps, float* output);
/* rms_norm, then sd_op_rope's rotation of the last rope_dims values (0: none), stored at row
   out_rows[row] of out (out_count rows), or at row `row` without out_rows. */
int sd_op_norm_rope(void* handle, const float* input, const float* weight, uint64_t rows, uint64_t width, float eps, uint64_t rope_dims, const int32_t* positions, const float* frequencies, const int32_t* out_rows, float* out, uint64_t out_count);
typedef struct sd_dspark_inject_stage {
    const void* kv;      /* Q8_0 [512,5120]. */
    const float* norm;   /* F32 [512]. */
    float* ring;         /* F32 [128,512], untouched rows stay unchanged. */
} sd_dspark_inject_stage;
/* DSpark target feature injection: features [rows,15360] -> Q8_0 fc [5120,15360]
   -> encoder RMS -> three independent KV projections / RMS / last-64 RoPE -> K rings.
   Frequencies: 32 F32 values, theta=10000. Positions are original token positions;
   ring index is position modulo 128. If rows collide, the last input row wins.
   encoded [rows,5120] also returns the encoder-normalized features. All ranges disjoint.
   Scratch holds 4096*rows bytes plus linear workspace; when rows>4 the latter needs
   at least align256(30720*rows)+61440 bytes (both F32 and half GEMM modes).
   Three host stage descriptors; their buffers stay on the device. No host readback. */
int sd_op_dspark_inject(void* handle, const float* features, const int32_t* positions, uint64_t rows,
                        const void* fc, const float* encoder_norm, const sd_dspark_inject_stage* stages,
                        const float* frequencies, float* encoded, float* scratch, uint64_t scratch_bytes);
int sd_op_rope(void* handle, float* x, uint64_t rows, uint64_t vectors, uint64_t width, uint64_t rope_dims, const int32_t* positions, const float* frequencies, int inverse);
/* Hyper-connections: coefficients (pre and post per copy, a Sinkhorn-balanced combination) from
   mixes = W_hc . stream; a stream collapsed by pre; a sublayer output expanded back with the
   residual: out[j] = post[j] x + sum_i comb[i][j] residual[i]. */
int sd_op_hc_coefficients(void* handle, const float* mixes, const float* stream, uint64_t rows, uint64_t hc, uint64_t width, const float* base, const float* scale, uint64_t iterations, float eps, float norm_eps, float* pre, float* post, float* comb);
int sd_op_hc_collapse(void* handle, const float* stream, const float* pre, uint64_t rows, uint64_t hc, uint64_t width, float* out);
/* DSpark's target feature of a layer: each row's hc copies summed in order over hc, rows from `skip` on, row r into
   out + (r - skip) * stride + offset. */
int sd_op_hc_mean(void* handle, const float* stream, uint64_t rows, uint64_t hc, uint64_t width, uint64_t skip, float* out, uint64_t stride, uint64_t offset);
int sd_op_hc_expand(void* handle, const float* x, const float* residual, const float* post, const float* comb, uint64_t rows, uint64_t hc, uint64_t width, float* out);
/* A sublayer boundary in one launch: x (optional) expanded with residual, post_in and comb_in into
   stream (else stream is read); the coefficients from fn (optional, with base and scale) into
   pre, post and comb; out = rms_norm(stream collapsed by `collapse`) * norm_weight. The workspace
   (zeroed once, then kept) takes 1 + ceil(width / 64) * ((2 + hc) * hc + 2) values per row, up to
   the most rows a call takes; one workspace serves one shape. */
int sd_op_hc_step(void* handle, const float* x, const float* residual, const float* post_in, const float* comb_in, float* stream, const float* fn, const float* base, const float* scale, uint64_t iterations, float eps, float norm_eps, const float* collapse, const float* norm_weight, uint64_t rows, uint64_t hc, uint64_t width, float* workspace, uint64_t workspace_bytes, float* pre, float* post, float* comb, float* out);
/* Engram: kv = [hc keys | value] per row from the table rows; each copy gains its gated value. */
int sd_op_engram_gate(void* handle, float* stream, const float* kv, const float* weight, uint64_t rows, uint64_t hc, uint64_t width, float eps);
/* Experts: routes (gating 0 softmax, 1 sigmoid, 2 square-root softplus; bias steers the choice,
   not the weights), clamped SwiGLU with optional row weights, row gather and scatter-add. */
int sd_op_route(void* handle, const float* logits, const float* bias, uint64_t rows, uint64_t experts, uint64_t k, int gating, int normalize, float scale, int32_t* ids, float* weights);
/* sd_op_linear (rows = experts, 1-4 steps, a contiguous output) then sd_op_route over its logits,
   in one launch where the router is F32 or F16 (the same results as the two ops), else as the two. */
int sd_op_linear_route(void* handle, const void* weights, int kind, uint64_t experts, uint64_t cols, const float* input, uint64_t steps, float* logits, const float* bias, uint64_t k, int gating, int normalize, float scale, int32_t* ids, float* route_weights, float* scratch, uint64_t scratch_bytes);
/* sd_op_linear_route (the fused router of 1-4 rows alone) whose routes also go straight to the
   device's route mailbox in page-locked host memory, its flag raised once every row's are there:
   sd_route_mailbox_wait takes them without a readback copy; sd_route_mailbox_clear lowers the flag
   before the launch (and makes the mailbox: never while a graph is recorded). Up to 1024 routes.
   The wait spins on the host and fails after 60 s or when the stream fails. */
int sd_op_linear_route_published(void* handle, const void* weights, int kind, uint64_t experts, uint64_t cols, const float* input, uint64_t steps, float* logits, const float* bias, uint64_t k, int gating, int normalize, float scale, int32_t* ids, float* route_weights, float* scratch, uint64_t scratch_bytes);
int sd_route_mailbox_clear(void* handle);
int sd_route_mailbox_wait(void* handle, int32_t* out, uint64_t count);
int sd_op_swiglu(void* handle, const float* gate, const float* up, uint64_t rows, uint64_t width, float clamp, const float* row_weights, float* out);
/* Zero `bytes` from `pointer` (inside an allocation of the device), queued on its stream. */
int sd_op_zero(void* handle, void* pointer, uint64_t bytes);
int sd_op_gather(void* handle, const float* x, uint64_t x_rows, const int32_t* rows, uint64_t count, uint64_t width, float* out);
int sd_op_scatter(void* handle, const float* x, const int32_t* rows, uint64_t count, uint64_t width, float* out, uint64_t out_rows);
int sd_op_scatter_add(void* handle, const float* x, const int32_t* rows, uint64_t count, uint64_t width, float* out, uint64_t out_rows);
// A wave of experts in slots through the RDNA4 grouped WMMA kernels (see kernels.cpp): whether the
// device and these kinds and shapes take them (1), gate/up into FP16 grouped products (clamped
// SwiGLU times the routes' weights), down from FP16 grouped products into F32 rows out_stride apart.
int sd_slot_moe_supported(void* handle, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn, uint64_t hidden_share);
int sd_op_slot_moe_gate_up(void* handle, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts, uint64_t experts,
                           uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn,
                           uint64_t hidden_share, const int32_t* rows, uint64_t steps, const void* input, float clamp,
                           const float* weights, void* product, void* tables, uint64_t table_bytes);
int sd_op_slot_moe_down(void* handle, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts, uint64_t experts,
                        uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn, uint64_t hidden_share,
                        const void* product, float* outputs, uint64_t out_stride, void* tables, uint64_t table_bytes);
// `values` (a multiple of four) rounded to FP16.
int sd_op_half_values(void* handle, const float* input, uint64_t values, void* output);
// Every device's `out` (rows of `full` FP16 values) = each device's `part` (rows of its `share` columns)
// at its columns `firsts[d]`, the peers' read over the peer link; ordered as sd_op_sum_devices.
int sd_op_gather_columns(void* const* handles, void* const* outs, const void* const* parts, const uint64_t* firsts, uint64_t devices,
                         uint64_t rows, uint64_t share, uint64_t full);
/* Experts in slots: route r (rows * top_k of them, token-major) runs the expert at bases[r], a slot
   of slot_bytes holding its gate and up (ffn x hidden) and down (hidden x ffn) matrices at byte
   offsets[0..2] (a zero base: a route another device runs); product[r] = SwiGLU of the route
   (clamped, times weights[r] when given) and out[t] (+)= the sum over the token's routes of
   down . product, added when accumulate is set.
   With split (else whole experts, both stages): a device's rows of each expert (tensor parallel),
   gate and up rows [ffn_first, ffn_first + ffn_rows) writing product[r][ffn_first ..] (rows of ffn
   values), down rows [hidden_first, hidden_first + hidden_rows) writing out[t][hidden_first ..];
   stages 1 runs gate/up, 2 down (from the whole product), 3 both; send (two devices joined by
   sd_link_parts), 1 + the part link channel the last stage's values go to the other device on: the
   down stage's rows of out, or the gate/up stage's (alone) columns of product, at send_at floats
   into the inbox; more (gate/up alone): a later call sends more of the same sum, so this one raises
   no flag (else 0: the sum's part is complete). Without bases, the routes the device last uploaded
   (a call of several routes per row; as many) are used again. */
typedef struct sd_expert_split {
    uint64_t ffn_first, ffn_rows, hidden_first, hidden_rows;
    int stages;
    int send;
    int more;
    uint64_t send_at;
} sd_expert_split;
int sd_op_experts(void* handle, const uint64_t* bases, uint64_t slot_bytes, const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn, uint64_t rows, uint64_t top_k, const float* input, uint64_t input_stride, const float* weights, float clamp, int accumulate, float* product, float* out, uint64_t out_stride, const sd_expert_split* split);
/* Experts by table, for experts that stay in their slots: sd_expert_table writes count slot
   addresses (each slot_bytes, 256-byte aligned, inside this device's allocations, or 0: an expert
   another device runs; checked) into the device array table. sd_op_experts_table runs as sd_op_experts does without split, route r
   taking the slot table[ids[r]] (ids on the device, as sd_op_route leaves them; an id outside the
   table runs nothing), so no host data goes with it and a graph may record it. */
int sd_expert_table(void* handle, uint64_t* table, const uint64_t* bases, uint64_t count, uint64_t slot_bytes);
int sd_op_experts_table(void* handle, const uint64_t* table, uint64_t count, const int32_t* ids, uint64_t slot_bytes, const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn, uint64_t rows, uint64_t top_k, const float* input, const float* weights, float clamp, int accumulate, float* product, float* out);
/* Tensor parallel: every device's outs[i] = parts[0] + parts[1] + ... (count values, in device
   order), each device's stream waiting for the others' parts and reading them over peer access.
   A part must stay unchanged until every device's sum has run; an out is never a part. */
int sd_op_sum_devices(void* const* handles, float* const* outs, const float* const* parts, uint64_t devices, uint64_t count);
/* Every device's later work waits until each other device's last sd_op_sum_devices has read its
   part (that sum reads it in place): before a device writes a summed part's buffer again. */
int sd_devices_settle(void* const* handles, uint64_t devices);
/* Attention rope for one pass: rows of rope_dims/2 pairs, forward then inverse. Allocate once,
   outside a captured graph (a graph keeps the pointer). A larger request replaces it only while
   no graph is stored. rope_dims 0 reserves nothing. */
int sd_attention_turns_reserve(void* handle, uint64_t rows, uint64_t rope_dims);
/* The next sd_op_norm_rope or sd_op_norm_linears_rope of exactly `rows` (0 clears) writes that
   table: sincos(position * frequency), then sincos of that angle times -1. The next
   sd_op_sparse_attention of those rows skips its own rope launch. */
int sd_attention_turns_arm(void* handle, uint64_t rows);
/* Latent attention: each query attends to the window rows of `raw` ending at its position
   (position p in row p % raw_rows, so `raw` may be a ring) and
   the compressed rows `selected` lists (-1 none), with a sink per head. Compression pools ratio
   rows by a per-channel softmax of the gate. The indexer scores visible groups and keeps the top k.
   Attention runs tiled when dim is a multiple of 64 and scratch (optional) holds rows * heads *
   (window + selections) scores and rows * heads * ceil((window + selections) / 32) pairs; then
   rope_dims (else 0) rotates q's last values by positions * frequencies as sd_op_rope does, and
   the output back. */
int sd_op_sparse_attention(void* handle, const float* q, const float* raw, uint64_t raw_rows, const int32_t* positions, uint64_t window, const float* compressed, uint64_t compressed_rows, const int32_t* selected, uint64_t selections, const float* sinks, uint64_t rows, uint64_t heads, uint64_t dim, float scale, uint64_t rope_dims, const float* frequencies, float* scratch, uint64_t scratch_bytes, float* out);
/* DSpark noncausal block attention: q/out [slots,heads,512] (a device's heads, 1..64), K=V ring
   [128,512], block KV [slots,512], sinks [heads]. Q and all keys already have RoPE applied.
   Every query sees history positions [anchor-history,anchor) and ALL block keys.
   slots 1..5, history 0..128, anchor>=history and <=INT32_MAX. Scale 1/sqrt(512);
   the sink contributes to the softmax denominator only. All ranges disjoint; graph-capturable. */
int sd_op_dspark_attention(void* handle, const float* q, const float* ring, const float* block,
                           const float* sinks, uint64_t heads, uint64_t slots, uint32_t anchor, const int32_t* anchor_at,
                           uint64_t history, float* out);
int sd_op_compress(void* handle, const float* kv, const float* gate, uint64_t groups, uint64_t ratio, uint64_t dim, float* out);
int sd_op_index_scores(void* handle, const float* q, const float* keys, const float* weights, const int32_t* visible, uint64_t rows, uint64_t groups, uint64_t heads, uint64_t dim, float scale, float* scores);
int sd_op_top_k(void* handle, const float* scores, uint64_t rows, uint64_t groups, uint64_t k, int32_t* out);
/* DeepSeek-V4.1's candidate blocks (its indexer's first level): each row's `blocks` blocks of `block`
   groups whose highest score is highest (scores rows x groups as sd_op_index_select stores them, -inf
   past the visible; the block of a row's last visible group always kept, blocks of only -inf never), in
   index order, -1 past the last: out rows x blocks. scratch holds rows * ceil(groups / block) floats. */
int sd_op_index_candidates(void* handle, const float* scores, const int32_t* visible, uint64_t rows, uint64_t groups, uint64_t blocks, uint64_t block, int32_t* out, float* scratch, uint64_t scratch_bytes);
/* sd_op_rope of q (rope_dims, positions, frequencies; q itself unchanged), sd_op_index_scores and
   sd_op_top_k of the scores in one launch: the same scores and indices.
   With candidates (rows x blocks, sd_op_index_candidates'), each row's groups scored and chosen are its
   candidates' (blocks * block, scores rows x that in candidate order), the indices the groups'. */
int sd_op_index_select(void* handle, const float* q, const float* keys, const float* weights, const int32_t* visible, uint64_t rows, uint64_t groups, uint64_t heads, uint64_t dim, float scale, uint64_t rope_dims, const int32_t* positions, const float* frequencies, float* scores, uint64_t k, int32_t* out, const int32_t* candidates, uint64_t blocks, uint64_t block);
/* sd_op_index_select of one row whose weights are first formed: weights = proj_weights . x (heads rows of
   proj_cols, as sd_op_linear forms them), in the same launch where it can. */
int sd_op_index_select_proj(void* handle, const void* proj_weights, int proj_kind, uint64_t proj_cols, const float* x, const float* q, const float* keys, float* weights, const int32_t* visible, uint64_t groups, uint64_t heads, uint64_t dim, float scale, uint64_t rope_dims, const int32_t* positions, const float* frequencies, float* scores, uint64_t k, int32_t* out, const int32_t* candidates, uint64_t blocks, uint64_t block);

/* Token mixers (any shape; gpu/token_mixing.hpp). conv_silu: x (steps x channels) becomes SiLU of
   each channel's causal convolution with its taps (channels x kernel, kernel <= 8) over history
   (kernel - 1 rows, oldest first; left holding the last inputs) and x. delta_rule: a block per value
   head h of state (v_heads x dim x dim, [key][value]; dim 64 or 128) over qkv rows [q (k_heads x
   dim) | k | v (v_heads x dim)], head h reading key head h % k_heads: q and k L2-normalized (q also
   by 1/sqrt(dim)), S *= exp(a[h] * softplus(alpha + dt[h])), d = (v - S^T k) * sigmoid(beta),
   S += k d^T, out = S^T q (steps x v_heads x dim). gated_norm: rms_norm(x) * weight * SiLU(gate)
   per row. split: rows of [first | second] (2 x width) into two. rope_halves: the leading rope_dims
   values of each vector, pair i = (i, i + rope_dims / 2), rotated by position * frequencies[i].
   attention: causal grouped-query attention of q (rows x heads x dim) at positions over the
   cache's keys and values (context x kv_heads x dim) up to each position (within the cache),
   times sigmoid(gate) when given; the keys are split among blocks (flash decoding) whose partial
   results take scratch: rows x heads x splits x (dim + 2) values, splits = min(64, max(1, 256 /
   (rows x heads))). add: out += x. scale_rows: each row of x times sigmoid of its scalar. */
int sd_op_conv_silu(void* handle, float* x, float* history, const float* taps, uint64_t channels, uint64_t kernel, uint64_t steps);
int sd_op_delta_rule(void* handle, const float* qkv, const float* beta, const float* alpha, const float* dt, const float* a, float* state, uint64_t k_heads, uint64_t v_heads, uint64_t dim, uint64_t steps, float eps, float* out);
int sd_op_gated_norm(void* handle, const float* x, const float* gate, const float* weight, uint64_t rows, uint64_t width, float eps, float* out);
/* One decode step (one row) of a DeltaNet: sd_op_conv_silu, sd_op_delta_rule and sd_op_gated_norm
   (out = the gated norm of the rule's output with gate and weight) in one launch, the same
   results; qkv is left as it was (its convolution is not stored). 1 when not taken (dim 128 and
   two value heads a key head only): the caller runs the three. */
int sd_op_delta_step(void* handle, const float* qkv, float* history, const float* taps, uint64_t kernel, const float* beta, const float* alpha, const float* dt, const float* a, float* state, uint64_t k_heads, uint64_t v_heads, uint64_t dim, float eps, const float* gate, const float* weight, float* out);
int sd_op_split(void* handle, const float* x, uint64_t rows, uint64_t width, float* first, float* second);
/* An attention step's inputs in one launch: sd_op_split of q_full (rows x heads x [query | gate], each
   dim) into q and gate, sd_op_rms_norm of the queries (q_weight) and of k's heads (k_weight, in place),
   sd_op_rope_halves of both (rope_dims), and sd_op_scatter of k and v (rows x kv_heads x dim) to rows
   positions of keys and values (context rows each); the same results. */
int sd_op_attention_prepare(void* handle, const float* q_full, float* k, const float* v, uint64_t rows, uint64_t heads, uint64_t kv_heads, uint64_t dim, const float* q_weight, const float* k_weight, float eps, uint64_t rope_dims, const int32_t* positions, const float* frequencies, float* q, float* gate, float* keys, float* values, uint64_t context);
int sd_op_rope_halves(void* handle, float* x, uint64_t rows, uint64_t vectors, uint64_t width, uint64_t rope_dims, const int32_t* positions, const float* frequencies);
int sd_op_attention(void* handle, const float* q, const float* keys, const float* values, uint64_t context, const int32_t* positions, uint64_t rows, uint64_t heads, uint64_t kv_heads, uint64_t dim, float scale, const float* gate, float* scratch, uint64_t scratch_bytes, float* out);
int sd_op_add(void* handle, const float* x, uint64_t count, float* out);
/* Development: 64 partial sums (each 32-bit word times a weight of its position, wrapping) of the
   bytes at data (a multiple of 4), queued into partials, for telling a range's contents apart. */
int sd_debug_checksum(void* handle, const void* data, uint64_t bytes, uint64_t* partials);
/* out = x as BF16 (count values; round to nearest, ties to even). */
int sd_op_to_bf16(void* handle, const float* x, uint64_t count, void* out);
/* Development: queue a store of the device's wall clock into stamps[slot] (device memory); the
   clock's rate in kHz (sd_debug_clock_khz) turns stamp differences into time. */
int sd_debug_stamp(void* handle, uint64_t* stamps, uint64_t slot);
int sd_debug_clock_khz(void* handle, uint64_t* khz);
/* Tensor-parallel sums of two devices (gpu/device_link.hpp): sd_link_sums joins them (peer access
   needed) with inboxes for parts of up to `capacity` floats; sd_op_link_sum queues out = this
   device's part + the other's part of the same sum (count floats; out may be part; accumulate:
   out += that sum), waiting on the device, so no host data goes with it and a graph may record
   it; the two devices' results are equal. sd_link_failed (after the stream is idle): 1 when a wait
   gave up (the sums since are void); sd_op_link_status queues a copy of that word (nonzero after a
   wait gave up) into out; sd_link_reset waits for both devices' streams and starts their counts
   over. */
int sd_link_sums(void* first, void* second, uint64_t capacity);
/* A decode step's linked sum folded into the kernel forming this device's part of it (the last
   block adds the parts), the same results as forming the part and then sd_op_link_sum (out += this
   part + the other's): sd_op_linear_link_sum, one vector through rows x cols into part;
   sd_op_experts_table_link_sum, sd_op_experts_table of one row (clamp 0, accumulating into part). */
int sd_op_linear_link_sum(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* input, float* part, float* out);
int sd_op_experts_table_link_sum(void* handle, const uint64_t* table, uint64_t count, const int32_t* ids, uint64_t slot_bytes, const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn, uint64_t top_k, const float* input, const float* weights, float* product, float* part, float* out);
int sd_op_link_sum(void* handle, const float* part, uint64_t count, float* out, int accumulate);
int sd_link_failed(void* handle);
int sd_op_link_status(void* handle, uint32_t* out);
int sd_link_reset(void* handle);
/* Development: in this device's folded linked sums (sd_op_linear_link_sum,
   sd_op_experts_table_link_sum), the last block's waves past the first wait `microseconds` (at most
   a second; 0: none) before they add the parts, as waves of a block can run apart on a loaded GPU:
   tests widen the window of races between them. */
int sd_link_skew(void* handle, uint64_t microseconds);
/* Part links (gpu/link_tail.hpp, gpu/device_link.hpp): sums of two devices' parts whose producer
   sends this device's part to the other as it forms it and whose consumer adds the two as it reads
   them (this part + the other's, the addition sd_op_sum_devices makes), in three channels, each sum
   a producer on both devices and then a consumer on both. sd_link_parts joins two devices (peer
   access needed) with inboxes of up to `capacity` floats per channel. Producers:
   sd_op_linear_send, `steps` (at most 4) vectors through rows x cols into output (as sd_op_linear),
   sent; sd_op_experts with a split's send; sd_op_part_send, columns [first, first + count) of
   `rows` rows `stride` floats apart of a part formed before. Consumers: sd_op_hc_step_linked,
   sd_op_hc_step whose x is the sum of `part` (this device's) and the other's; sd_op_part_receive,
   out = part + the other's (count floats). Values a producer sends land at their offsets from the
   part's start; the inbox stays zero where the other never sends. A wait that gives up (the
   device's await timeout) is recorded and the stream goes on (the sums since are void):
   sd_op_part_status queues 1 | 2 | 4 by channel (0: none gave up) into out, sd_part_failed reads it
   after the stream is idle, and sd_part_reset waits for both devices' streams and starts every
   count over. */
int sd_link_parts(void* first, void* second, uint64_t capacity);
int sd_op_linear_send(void* handle, int channel, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* input, uint64_t steps, float* output);
int sd_op_hc_step_linked(void* handle, int channel, const float* part, const float* residual, const float* post_in, const float* comb_in, float* stream, const float* fn, const float* base, const float* scale, uint64_t iterations, float eps, float norm_eps, const float* collapse, const float* norm_weight, uint64_t rows, uint64_t hc, uint64_t width, float* workspace, uint64_t workspace_bytes, float* pre, float* post, float* comb, float* out);
int sd_op_part_send(void* handle, int channel, const float* part, uint64_t rows, uint64_t stride, uint64_t first, uint64_t count);
int sd_op_part_receive(void* handle, int channel, const float* part, uint64_t count, float* out);
int sd_op_part_status(void* handle, uint32_t* out);
int sd_part_failed(void* handle);
int sd_part_reset(void* handle);
/* Routed experts of many rows at once (a prefill pass; gpu/moe_prefill.hpp): sd_moe_groups writes
   an expert table (experts entries of sd_moe_group_bytes() each) from each expert's slot address
   (0: an expert another device runs) and its gate, up and down byte offsets and kinds in the
   slot; sd_op_experts_grouped adds to out each row's routes (ids and weights, rows x top_k)
   through the table's experts as RDNA4 grouped FP16 GEMMs (F32 sums), routes to absent experts
   left out, its scratch in `scratch`: 1 when not taken here (not RDNA4, or a shape its tiles do
   not fit). */
uint64_t sd_moe_group_bytes(void);
int sd_moe_groups(void* handle, void* groups, const uint64_t* bases, uint64_t experts, const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn);
int sd_op_experts_grouped(void* handle, void* groups, uint64_t experts, const int32_t* ids, const float* weights, uint64_t rows, uint64_t top_k, uint64_t hidden, uint64_t ffn, const int* kinds, const float* input, void* scratch, uint64_t scratch_bytes, float* out);
int sd_op_scale_rows(void* handle, float* x, const float* scalars, uint64_t rows, uint64_t width);

/* Layer operations */
int sd_hyperconnection(void* handle, void* const* pointers, const int* kinds, uint64_t steps, int inject, const float* input, float* output);
int sd_gdn_sequence(void* handle, void* const* pointers, const int* kinds, void* conv, void* recurrent, uint64_t steps, const float* input, float* output);
int sd_qsa_sequence(void* handle, void* const* pointers, const int* kinds, void* kv, void* index, void* scratch, uint64_t context, uint64_t position, uint64_t steps, const float* input, float* output);
int sd_delta_sequence(void* handle, void* state, uint64_t heads, uint64_t dimension, uint64_t steps, const float* input, uint64_t input_count, float* output, uint64_t output_count);
int sd_attention_sequence(void* handle, void* state, uint64_t context, uint64_t base, uint64_t steps, uint64_t selected, const float* input, uint64_t input_count, float* output, uint64_t output_count);
int sd_moe(void* handle, void* const* pointers, const int* kinds, uint64_t experts, void* workspace, uint64_t steps, const int32_t* selected, const float* probabilities, const float* input, float* output);
int sd_ple(void* handle, void* const* pointers, const int* kinds, void* history, uint64_t steps, const float* embeddings, const float* hidden, float* output);
int sd_ple_rows(void* handle, void* const* pointers, const int* kinds, void* history, uint64_t steps, const float* embeddings, const float* hidden, float* output);
int sd_prefill_ple(void* handle, void* const* pointers, const int* kinds, void* history, uint64_t steps, const float* embeddings, uint64_t layer);

/* Resident decoder and spans */
int sd_decoder_layer(void* handle, uint64_t layer, uint64_t context, void* const* pointers, const int* kinds, uint64_t count, void* const* states, void* workspace, void* moe);
int sd_decode_span(void* handle, uint64_t begin, uint64_t end, uint64_t position, const float* input, float* output);
int sd_prefill_span(void* handle, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, const float* input, float* output, uint64_t output_rows, double* timings);
/* Qwen3.8's layers split across two linked GPUs (gpu/qwen_parts.hpp, sd_link_sums): this device's parts (1 or 2), before any
   layer or K/V state is registered on it; then each layer's part (0 or 1 of two), registered as sd_decoder_layer registers a
   whole layer, with the part's operands and states. */
int sd_qwen_parts(void* handle, uint64_t parts);
int sd_decoder_part(void* handle, uint64_t layer, uint64_t context, void* const* pointers, const int* kinds, uint64_t count, void* const* states, void* workspace, void* moe, uint64_t part);
/* A layer part's resident MoE plan (sd_resident_moe_part, without the shared expert): its half of the shared expert's width,
   rows [320p, +320) of gate and up (320 x 2560) and those columns of down (2560 x 320). */
int sd_moe_shared_half(void* handle, void* plan, void* const* pointers, const int* kinds);
/* A prompt batch through layers [begin, end) of parts on both devices (sd_prefill_span's arguments; the output rows from the second). */
int sd_prefill_parts(void* first, void* second, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, const float* input, float* output, uint64_t output_rows);
/* After a pass the host queued on both devices: both streams finished and any wait for a part that gave up reported (an error,
   the counts started over). */
int sd_parts_settle(void* first, void* second);
/* Move the prompt rows a span left in `from`'s decoder workspace (of layer from_layer) into
   `to`'s (of layer to_layer), device to device, for the next span to continue from. */
int sd_prefill_handoff(void* from, void* to, uint64_t from_layer, uint64_t to_layer, uint64_t steps);
int sd_prefill_defer(void* handle, int on);
int sd_prefill_settle(void* handle);
/* Start a prompt in layer's decoder workspace from its embeddings (steps rows of 2560), each
   row repeated into the four hyper-connection streams on the device. */
int sd_prefill_embed(void* handle, uint64_t layer, const float* embeddings, uint64_t steps);
/* Feed the MTP layer (`layer`, on this device) one batch of prompt pairs at `position`: the
   hidden rows are `previous` (one row, or none) then rows first .. of the prompt rows left in
   `stash`'s decoder workspace (of stash_layer), the embeddings (count x 2560) come from the host,
   and nothing returns to the host. */
int sd_mtp_prompt_rows(void* handle, void* const* pointers, const int* kinds, void* stash, uint64_t stash_layer,
                       uint64_t first, uint64_t count, const float* previous, const float* embeddings,
                       uint64_t position, uint64_t layer);
int sd_mtp_keep(void* handle, uint64_t layer, uint64_t steps, void* rows, uint64_t previous, const float* host);
int sd_mtp_embed(void* handle, const float* embeddings, uint64_t count, void* target);
int sd_mtp_prompt_kept(void* handle, void* const* pointers, const int* kinds, void* rows, uint64_t row, void* embeddings,
                       uint64_t offset, uint64_t count, uint64_t position, uint64_t layer);
int sd_verify_span(void* handle, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, const float* input, float* output);
int sd_verify_reserve(void* handle);
int sd_verify_commit(void* handle, uint64_t begin, uint64_t end, uint64_t kept);
// top: each row's `top` (at most 64) greatest logits into best (top values, then top token indices as int32 bits), most likely first.
// input null: the rows of layer `layer`'s decoder workspace, handed back in `rows`.
int sd_output_head(void* handle, void* const* pointers, const int* kinds, uint64_t steps, uint64_t vocabulary, const float* input, int32_t* tokens, float* logits, float* confidence, uint64_t top, float* best, uint64_t layer, float* rows);
int sd_output_head_split(void* handle, void* const* pointers, const int* kinds, uint64_t steps, uint64_t vocabulary, uint64_t split, void* peer_head, const float* input, int32_t* tokens, float* logits, float* confidence, uint64_t top, float* best, uint64_t layer, float* rows);
int sd_mtp_prologue(void* handle, void* const* pointers, const int* kinds, uint64_t steps, const float* embeddings, const float* hidden, float* output);
int sd_verify_queue(void* handle, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, const float* input);
int sd_ple_front(void* handle, void* const* pointers, const int* kinds, void* history, void* snapshot, void* kept, uint64_t layer, uint64_t steps, const float* embeddings, float* first, float* hidden);
int sd_ple_replay(void* handle, void* const* pointers, const int* kinds, void* history, void* snapshot, void* kept, uint64_t steps, const float* embeddings);
int sd_draft_step(void* handle, void* const* mtp, const int* mtp_kinds, uint64_t layer, uint64_t position, uint64_t steps, const float* embeddings, const float* hidden, uint64_t previous, void* const* head, const int* head_kinds, uint64_t vocabulary, uint64_t split, void* peer_head, int32_t* token, float* confidence, uint64_t top, float* best);
// A drafting step's MTP layer queued with nothing waited for (sd_draft_step without its head): on each device of a layer in
// parts, the host queueing both before sd_draft_head.
int sd_draft_queue(void* handle, void* const* mtp, const int* mtp_kinds, uint64_t layer, uint64_t position, uint64_t steps, const float* embeddings, const float* hidden, uint64_t previous);
// sd_draft_step's head on row `row` of the rows sd_draft_queue left in `layer`'s workspace.
int sd_draft_head(void* handle, void* const* head, const int* head_kinds, uint64_t layer, uint64_t row, uint64_t vocabulary, uint64_t split, void* peer_head, int32_t* token, float* confidence, uint64_t top, float* best);
// A draft's vocabulary when the head is split (sd_draft_step): rows [0, rows) of `head` (vocabulary rows of `kind`) and the rows of
// `count` more tokens (each at least `rows`), appended to the ones before unless `reset`. rows 0: drafts read the whole head.
int sd_draft_vocabulary(void* handle, void* head, int kind, uint64_t vocabulary, uint64_t rows, const int32_t* tokens, uint64_t count, int reset);
int sd_execution_counts(void* handle, uint64_t* graphs, uint64_t* replays, uint64_t* resident);

/* Resident experts and two-GPU links */
void* sd_resident_moe_create(void* handle, void* const* pointers, const int* kinds, uint64_t experts, void* workspace, uint64_t capacity);
void* sd_resident_moe_part(void* handle, void* const* pointers, const int* kinds, uint64_t experts, uint64_t first, uint64_t count, int shared, void* workspace, uint64_t capacity);
int sd_resident_moe(void* handle, void* plan, uint64_t steps, const float* input, float* output);
int sd_helper_moe(void* handle, uint64_t layer, void* plan);
int sd_peer_attach(void* first, void* second, uint64_t rows);
/* 1: the linked devices write each other's VRAM; 0: through host memory; -1: not linked; -2: failure. */
int sd_peer_direct(void* handle);

/* Partial residency */
int sd_hybrid_supported(void* handle);
void* sd_hybrid_moe_create(void* handle, void* const* pointers, const int* kinds, const int* routed_kinds, uint64_t experts, const uint8_t* const* host, void* workspace, uint64_t capacity);
int sd_expert_slots(void* handle, uint64_t cache, uint64_t stage, uint64_t slot_bytes);
int sd_hybrid_attach_cpu(void* handle, void* engine);
int sd_hybrid_drain(void* handle);
int sd_hybrid_rebalance(void* handle, uint64_t max_swaps);
int sd_hybrid_assist(void* handle, uint32_t assist, uint64_t* mapped);
int sd_hybrid_decay(double decay);
int sd_hybrid_timing(void* handle, uint64_t* wait_ns, uint64_t* cpu_ns, uint64_t* swapped, uint64_t* assisted);
int sd_hybrid_counts(void* handle, uint64_t* gpu, uint64_t* cpu, uint64_t* streamed, uint64_t* cached);

/* CPU experts and formats */
const char* sd_cpu_error(void);
void* sd_cpu_experts_create(uint32_t threads);
void sd_cpu_experts_destroy(void* handle);
const char* sd_cpu_experts_isa(void* handle);
int sd_cpu_experts_stats(void* handle, uint64_t* values);
int sd_cpu_experts_run(void* handle, const int* kinds, const uint8_t* const* gate, const uint8_t* const* up, const uint8_t* const* down, const float* weights, uint32_t count, uint64_t hidden, uint64_t inter, const float* x, float* out);
int sd_cpu_gather(void* handle, const uint8_t* const* sources, uint8_t* const* targets, const uint64_t* bytes, uint32_t count);
int sd_iq_tables(const uint8_t* iq2_xxs, const uint8_t* iq2_xs, const uint8_t* iq2_s, const uint8_t* iq3_xxs, const uint8_t* iq3_s, const uint8_t* iq1_s);
const uint8_t* sd_iq_grid(int kind, size_t* bytes);
int sd_cpu_decode(int kind, int wide, const uint8_t* block, float* out);
int sd_cpu_rows(int kind, int level, const uint8_t* row, uint64_t stride, uint32_t count, const float* x, uint64_t cols, float* out);
int sd_quantize_q2_0(const uint16_t* input, uint64_t count, uint8_t* output);

/* Host exchanges: page-locked host memory a device's ops read and write within a pass, so a host
   thread takes part in a pass queued (or recorded as a graph) ahead, with no host call between the
   ops. sd_exchange allocates `bytes` (zeroed; at least 256, the header, and a multiple of 4) and
   returns its host address, which the ops take; freed with the device. Header words (uint32): 0 the
   exports so far (the device's), 1 the results the host published, 2 nonzero once an await gave up
   (after about ten seconds), 3 nonzero while the host cancels the awaits (they then take zeros).
   sd_op_export copies `parts` (up to four) device ranges of counts[i] values one after another to
   `offset` bytes into the exchange, then advances word 0 (the values reach the host first).
   sd_op_await_add waits until word 1 reaches word 0's count as of the last export (the host
   publishes a result for each export, in order), then out[i] += a[i] (optional) + result[i] over the
   `count` values at result_offset. sd_exchange_supported answers 1 when the device's ops can wait
   for the host so (RDNA4, CUDA sm_70+), else 0. */
void* sd_exchange(void* handle, uint64_t bytes);
int sd_exchange_supported(void* handle);
int sd_op_export(void* handle, const float* const* sources, const uint64_t* counts, uint64_t parts, void* exchange, uint64_t offset);
int sd_op_await_add(void* handle, void* exchange, uint64_t result_offset, const float* a, float* out, uint64_t count);

/* Formats: the routed expert kernels at any layer shape, for tests and measurements. `banks` are the
   gate and up (`experts` matrices of ffn x hidden each) and down (hidden x ffn) allocations of
   kinds[0..2]; each of `steps` input rows (hidden values) goes to its top_k experts in `selected`.
   path 0 runs the decode GEMV (F32); path 1 the RDNA4 grouped WMMA GEMM (FP16 operands, F32 sums).
   Outputs per route, token-major, each optional: gate_up (the gate then the up row, before the
   activation), product (SiLU(gate) * up; rounded to FP16 on path 1) and output (down of the product,
   or of down_input when given, which then needs no input). timings (optional) receives the
   milliseconds of the gate/up and the down kernels (zero for a stage not run; path 1 times its fused
   gate/up kernel, which runs for product and output). */
int sd_expert_rows(void* handle, void* const* banks, const int* kinds, uint64_t experts, uint64_t hidden, uint64_t ffn,
                   uint64_t steps, uint64_t top_k, const int32_t* selected, int path, const float* input,
                   const float* down_input, float* gate_up, float* product, float* output, double* timings);

/* DeepSeek-V4.1's vision encoder (gpu/vision.hpp), each weight a device address: products FP16
 * (rows of their input's width; gate_up the gate's rows, then the up's), norms and biases F32. */
typedef struct sd_vision_layer {
    const void* norm1;
    const void* qkv;
    const void* qkv_bias;
    const void* out;
    const void* out_bias;
    const void* norm2;
    const void* gate_up;
    const void* down;
} sd_vision_layer;

typedef struct sd_vision {
    uint64_t layers;
    uint64_t width;
    uint64_t heads;
    uint64_t ffn;
    uint64_t patch_values;
    uint64_t out_width;
    uint64_t downsample;
    float eps;
    float rope_theta;
    const void* patch;
    const void* patch_bias;
    const void* norm;
    const void* align1;
    const void* align1_bias;
    const void* align2;
    const void* align2_bias;
    const sd_vision_layer* layer; /* host memory: `layers` of them */
} sd_vision;

/* The scratch bytes an image of grid_h x grid_w patches takes. */
int sd_vision_scratch_bytes(const sd_vision* model, uint64_t grid_h, uint64_t grid_w, uint64_t* bytes);
/* One image on the device's stream: `patches` ([grid_h * grid_w][patch_values], FP16, reading
 * order) to `out`, the aligner's rows ([ceil(grid_h / downsample) * ceil(grid_w / downsample)]
 * [out_width], F32, reading order). */
int sd_vision_encode(void* handle, const sd_vision* model, const void* patches, uint64_t grid_h, uint64_t grid_w, void* scratch,
                     uint64_t scratch_bytes, float* out);
/* `count` BF16 values in place as FP16. */
int sd_vision_half(void* handle, void* values, uint64_t count);

#ifdef __cplusplus
}
#endif

#endif
