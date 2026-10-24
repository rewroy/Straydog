// A DSpark key set expressed through the existing sparse-attention calculation:
// identical history positions for every query, then all B block keys as selections.
namespace {
__global__ void dspark_attention_indices(int32_t* positions, int32_t* selected, size_t slots, uint32_t anchor) {
    const unsigned i = threadIdx.x;
    if (i < slots) positions[i] = int32_t(anchor) - 1;
    if (i < slots * slots) selected[i] = int32_t(i % slots);
}
}

int sd_op_dspark_attention(void* handle, const float* q, const float* ring, const float* block,
                           const float* sinks, uint64_t heads, uint64_t slots, uint32_t anchor, uint64_t history,
                           float* out, int32_t* scratch, uint64_t scratch_bytes) {
    SD_OP_BEGIN
    if (!heads || heads > 64 || !slots || slots > 5 || history > 128 || anchor > INT32_MAX || history > anchor)
        throw std::runtime_error("invalid DSpark attention block or history range");
    const size_t needed = size_t(4 * slots * (slots + 1));
    if (scratch_bytes < needed) throw std::runtime_error("DSpark attention scratch is too small");
    op_span(value, q, checked_floats({slots, heads, 512}), "DSpark rotated queries");
    op_span(value, ring, 128 * 512 * 4, "DSpark history K ring");
    op_span(value, block, checked_floats({slots, 512}), "DSpark rotated block keys");
    op_span(value, sinks, checked_floats({heads}), "DSpark attention sinks");
    op_span(value, out, checked_floats({slots, heads, 512}), "DSpark attention output");
    op_span(value, scratch, needed, "DSpark attention scratch");
    int32_t* positions = scratch;
    int32_t* selected = scratch + slots;
    dspark_attention_indices<<<1, 32, 0, value.stream>>>(positions, selected, size_t(slots), anchor);
    // p=anchor-1 and window=history give exactly [anchor-history,anchor).
    // For anchor=history=0, p=-1 gives zero raw keys. All block keys remain visible.
    const size_t shared = size_t(512 + history + slots) * 4;
    op_sparse_attention_kernel<<<unsigned(slots * heads), 256, shared, value.stream>>>(q, ring, 128, positions,
        size_t(history), block, selected, size_t(slots), sinks, size_t(heads), 512, 1.0f / sqrtf(512.0f), out);
    SD_OP_END
}
