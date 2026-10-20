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
int sd_write(void* handle, void* pointer, const void* data, uint64_t bytes);
void* sd_filler_create(void* handle);
void sd_filler_destroy(void* filler);
int sd_fill(void* filler, void* pointer, uint64_t offset, const void* host, uint64_t bytes);
int sd_release(void* handle, void* pointer);
int sd_zero(void* handle, void* pointer);
int sd_download(void* handle, void* pointer, void* output, uint64_t bytes);
int sd_copy_allocation(void* handle, void* target, const void* source);
int sd_host_register(void* pointer, uint64_t bytes);
int sd_host_unregister(void* pointer);
void* sd_host_state(void* handle, uint64_t bytes);
int sd_host_states_prefetch(void* handle, const uint64_t* sizes, uint64_t count);
int sd_host_state_supported(void* handle);
int sd_kv_stage(void* handle, void* pointer, uint64_t rows);
void* sd_checkpoint_buffer(void* handle, uint64_t bytes);
int sd_checkpoint_copy(void* handle, int save, void* const* states, uint64_t count, void* buffer);
/* Asynchronous copies: page-locked host memory (sd_host_register, or sd_host_pin where devices
 * also write it) to and from allocations, beside the device's compute stream (stream 0): uploads
 * on stream 1, downloads on stream 2. sd_event_record marks the work queued so far on a stream and
 * returns the mark's id (ids grow from 1); sd_event_wait makes a stream wait for a mark on the GPU;
 * sd_event_done answers 1 once it completed (block: wait for it), after which the copies before it
 * are done with their memory. A copy into memory another stream still reads must wait for a mark
 * after that use. Counts: uploads, their bytes, downloads, their bytes. */
int sd_host_pin(void* pointer, uint64_t bytes);
int sd_transfer(void* handle, int stream, void* pointer, uint64_t offset, void* host, uint64_t bytes);
int64_t sd_event_record(void* handle, int stream);
int sd_event_wait(void* handle, int stream, int64_t id);
int sd_event_done(void* handle, int64_t id, int block);
int sd_transfer_counts(void* handle, uint64_t* counts);

/* Matrix products */
int sd_gemm_options(void* handle, uint64_t minimum_batch, int half_precision);
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
int sd_op_rms_norm(void* handle, const float* input, const float* weight, uint64_t rows, uint64_t width, float eps, float* output);
/* rms_norm, then sd_op_rope's rotation of the last rope_dims values (0: none), stored at row
   out_rows[row] of out (out_count rows), or at row `row` without out_rows. */
int sd_op_norm_rope(void* handle, const float* input, const float* weight, uint64_t rows, uint64_t width, float eps, uint64_t rope_dims, const int32_t* positions, const float* frequencies, const int32_t* out_rows, float* out, uint64_t out_count);
int sd_op_rope(void* handle, float* x, uint64_t rows, uint64_t vectors, uint64_t width, uint64_t rope_dims, const int32_t* positions, const float* frequencies, int inverse);
/* Hyper-connections: coefficients (pre and post per copy, a Sinkhorn-balanced combination) from
   mixes = W_hc . stream; a stream collapsed by pre; a sublayer output expanded back with the
   residual: out[j] = post[j] x + sum_i comb[i][j] residual[i]. */
int sd_op_hc_coefficients(void* handle, const float* mixes, const float* stream, uint64_t rows, uint64_t hc, uint64_t width, const float* base, const float* scale, uint64_t iterations, float eps, float norm_eps, float* pre, float* post, float* comb);
int sd_op_hc_collapse(void* handle, const float* stream, const float* pre, uint64_t rows, uint64_t hc, uint64_t width, float* out);
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
int sd_op_swiglu(void* handle, const float* gate, const float* up, uint64_t rows, uint64_t width, float clamp, const float* row_weights, float* out);
int sd_op_gather(void* handle, const float* x, uint64_t x_rows, const int32_t* rows, uint64_t count, uint64_t width, float* out);
int sd_op_scatter(void* handle, const float* x, const int32_t* rows, uint64_t count, uint64_t width, float* out, uint64_t out_rows);
int sd_op_scatter_add(void* handle, const float* x, const int32_t* rows, uint64_t count, uint64_t width, float* out, uint64_t out_rows);
/* Experts in slots: route r (rows * top_k of them, token-major) runs the expert at bases[r], a slot
   of slot_bytes holding its gate and up (ffn x hidden) and down (hidden x ffn) matrices at byte
   offsets[0..2]; product[r] = SwiGLU of the route (clamped, times weights[r] when given) and
   out[t] (+)= the sum over the token's routes of down . product, added when accumulate is set. */
int sd_op_experts(void* handle, const uint64_t* bases, uint64_t slot_bytes, const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn, uint64_t rows, uint64_t top_k, const float* input, uint64_t input_stride, const float* weights, float clamp, int accumulate, float* product, float* out, uint64_t out_stride);
/* Latent attention: each query attends to the window rows of `raw` ending at its position
   (position p in row p % raw_rows, so `raw` may be a ring) and
   the compressed rows `selected` lists (-1 none), with a sink per head. Compression pools ratio
   rows by a per-channel softmax of the gate. The indexer scores visible groups and keeps the top k.
   Attention runs tiled when dim is a multiple of 64 and scratch (optional) holds rows * heads *
   (window + selections) scores and rows * heads * ceil((window + selections) / 32) pairs; then
   rope_dims (else 0) rotates q's last values by positions * frequencies as sd_op_rope does, and
   the output back. */
int sd_op_sparse_attention(void* handle, const float* q, const float* raw, uint64_t raw_rows, const int32_t* positions, uint64_t window, const float* compressed, uint64_t compressed_rows, const int32_t* selected, uint64_t selections, const float* sinks, uint64_t rows, uint64_t heads, uint64_t dim, float scale, uint64_t rope_dims, const float* frequencies, float* scratch, uint64_t scratch_bytes, float* out);
int sd_op_compress(void* handle, const float* kv, const float* gate, uint64_t groups, uint64_t ratio, uint64_t dim, float* out);
int sd_op_index_scores(void* handle, const float* q, const float* keys, const float* weights, const int32_t* visible, uint64_t rows, uint64_t groups, uint64_t heads, uint64_t dim, float scale, float* scores);
int sd_op_top_k(void* handle, const float* scores, uint64_t rows, uint64_t groups, uint64_t k, int32_t* out);

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
int sd_verify_commit(void* handle, uint64_t begin, uint64_t end, uint64_t kept);
int sd_output_head(void* handle, void* const* pointers, const int* kinds, uint64_t steps, uint64_t vocabulary, const float* input, int32_t* tokens, float* logits, float* confidence);
int sd_output_head_split(void* handle, void* const* pointers, const int* kinds, uint64_t steps, uint64_t vocabulary, uint64_t split, void* peer_head, const float* input, int32_t* tokens, float* logits, float* confidence);
int sd_mtp_prologue(void* handle, void* const* pointers, const int* kinds, uint64_t steps, const float* embeddings, const float* hidden, float* output);
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

#ifdef __cplusplus
}
#endif

#endif
