// DSpark head orchestration over the existing BF16 GEMV, add and row argmax.
namespace {
__global__ void dspark_features(const float* hidden, const uint16_t* w1, const int32_t* tokens,
                                size_t slot, size_t vocabulary, size_t rank, size_t width,
                                uint32_t anchor, float* features) {
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < width) features[col] = hidden[slot * width + col];
    else if (col < width + rank) {
        const int32_t previous = slot ? tokens[slot - 1] : int32_t(anchor);
        // Non-finite logits can give argmax -1. Never use it as a table index.
        features[col] = previous >= 0 && size_t(previous) < vocabulary
            ? __uint_as_float(uint32_t(w1[size_t(previous) * rank + col - width]) << 16) : 0.0f;
    }
}
__global__ void dspark_finish(const int32_t* tokens, float* confidences, size_t slots, size_t vocabulary,
                              float threshold, size_t minimum, uint32_t* count) {
    if (threadIdx.x != 0) return;
    uint32_t kept = 0;
    bool live = true;
    for (size_t i = 0; i < slots; ++i) {
        const float confidence = 1.0f / (1.0f + expf(-confidences[i]));
        confidences[i] = confidence;
        live = live && tokens[i] >= 0 && size_t(tokens[i]) < vocabulary && isfinite(confidence) && confidence >= threshold;
        if (live) ++kept;
    }
    *count = kept >= minimum ? kept : 0;
}
} // namespace

int sd_op_dspark_head(void* handle, const float* body, const float* hidden, const void* w1, const void* w2,
                      const float* conf, uint64_t slots, uint64_t vocabulary, uint64_t rank, uint64_t width,
                      uint32_t anchor, float threshold, uint64_t minimum, float* logits, int32_t* tokens,
                      float* confidences, uint32_t* count, float* scratch, uint64_t scratch_bytes) {
    SD_OP_BEGIN
    if (!slots || slots > 5 || !vocabulary || vocabulary > INT32_MAX || !rank || !width ||
        anchor >= vocabulary || !std::isfinite(threshold) || threshold < 0 || threshold > 1)
        throw std::runtime_error("invalid DSpark head shape, anchor or prefix limits");
    const size_t features = checked_sum({width, rank});
    const size_t feature_bytes = checked_floats({features});
    const size_t partial_offset = checked_sum({feature_bytes, 15}) & ~size_t(15);
    const size_t needed = checked_sum({partial_offset, 2 * argmax_chunks * 4});
    if (scratch_bytes < needed) throw std::runtime_error("DSpark head scratch is too small");
    const unsigned feature_blocks = checked_blocks(features);
    const unsigned logit_blocks = checked_blocks(size_t(vocabulary));
    op_span(value, body, checked_floats({slots, vocabulary}), "DSpark body logits");
    op_span(value, hidden, checked_floats({slots, width}), "DSpark collapsed hidden");
    op_span(value, w1, checked_product({vocabulary, rank, 2}), "DSpark BF16 W1");
    op_span(value, w2, checked_product({vocabulary, rank, 2}), "DSpark BF16 W2");
    op_span(value, conf, feature_bytes, "DSpark confidence projection");
    op_span(value, logits, checked_floats({slots, vocabulary}), "DSpark corrected logits");
    op_span(value, tokens, checked_floats({slots}), "DSpark draft tokens");
    op_span(value, confidences, checked_floats({slots}), "DSpark confidence output");
    op_span(value, count, 4, "DSpark prefix length");
    op_span(value, scratch, needed, "DSpark scratch");
    auto* partial = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scratch) + partial_offset);
    for (size_t slot = 0; slot < slots; ++slot) {
        dspark_features<<<feature_blocks, 256, 0, value.stream>>>(hidden, static_cast<const uint16_t*>(w1), tokens,
            slot, size_t(vocabulary), size_t(rank), size_t(width), anchor, scratch);
        op_linear(value, static_cast<const uint8_t*>(w2), 30, size_t(vocabulary), size_t(rank), 0,
                  scratch + width, 0, 1, logits + slot * vocabulary, 0, nullptr, 0);
        tm_add_kernel<<<logit_blocks, 256, 0, value.stream>>>(body + slot * vocabulary, size_t(vocabulary), logits + slot * vocabulary);
        row_argmax(value, logits + slot * vocabulary, 1, size_t(vocabulary), 0, partial, tokens + slot);
        op_linear(value, reinterpret_cast<const uint8_t*>(conf), 0, 1, features, 0, scratch, 0, 1, confidences + slot, 0, nullptr, 0);
    }
    dspark_finish<<<1, 32, 0, value.stream>>>(tokens, confidences, size_t(slots), size_t(vocabulary), threshold, size_t(minimum), count);
    SD_OP_END
}
