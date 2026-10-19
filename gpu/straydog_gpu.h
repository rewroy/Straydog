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
int sd_release(void* handle, void* pointer);
int sd_zero(void* handle, void* pointer);
int sd_download(void* handle, void* pointer, void* output, uint64_t bytes);
int sd_copy_allocation(void* handle, void* target, const void* source);
int sd_host_register(void* pointer, uint64_t bytes);
int sd_host_unregister(void* pointer);
void* sd_host_state(void* handle, uint64_t bytes);
int sd_host_state_supported(void* handle);
int sd_kv_stage(void* handle, void* pointer, uint64_t rows);
void* sd_checkpoint_buffer(void* handle, uint64_t bytes);
int sd_checkpoint_copy(void* handle, int save, void* const* states, uint64_t count, void* buffer);

/* Matrix products */
int sd_gemm_options(void* handle, uint64_t minimum_batch, int half_precision);
int sd_gemm_counts(void* handle, uint64_t* calls, uint64_t* vectors);
int sd_matvec(void* handle, void* pointer, int kind, uint64_t rows, uint64_t cols, const float* input, float* output);
int sd_matvec_batch(void* handle, void* const* pointers, const int* kinds, const uint64_t* rows, const uint64_t* cols, uint64_t count, const float* input, uint64_t input_count, float* output, uint64_t output_count);

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
int sd_verify_span(void* handle, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, const float* input, float* output);
int sd_verify_commit(void* handle, uint64_t begin, uint64_t end, uint64_t kept);
int sd_output_head(void* handle, void* const* pointers, const int* kinds, uint64_t steps, uint64_t vocabulary, const float* input, int32_t* tokens, float* logits);
int sd_output_head_split(void* handle, void* const* pointers, const int* kinds, uint64_t steps, uint64_t vocabulary, uint64_t split, void* peer_head, const float* input, int32_t* tokens, float* logits);
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

#ifdef __cplusplus
}
#endif

#endif
