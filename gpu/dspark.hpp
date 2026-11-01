// DSpark head orchestration over the existing BF16 GEMV, add and row argmax.
namespace {
__global__ void dspark_features(const float* hidden, const uint16_t* w1, const int32_t* tokens,
                                size_t slot, size_t vocabulary, size_t rank, size_t width,
                                uint32_t anchor, const int32_t* anchor_at, float* features) { SD_KERNEL_ACQUIRE;
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < width) features[col] = hidden[slot * width + col];
    else if (col < width + rank) {
        const int32_t previous = slot ? tokens[slot - 1] : anchor_at ? anchor_at[0] : int32_t(anchor);
        // Non-finite logits can give argmax -1. Never use it as a table index.
        features[col] = previous >= 0 && size_t(previous) < vocabulary
            ? __uint_as_float(uint32_t(w1[size_t(previous) * rank + col - width]) << 16) : 0.0f;
    }
}
// out += the body's logits, which lie in two parts: the columns below `split` in `body`, the others in `tail`
// (rows of count - split): the devices that computed the logits' columns apart each have their own rows.
__global__ void dspark_add_kernel(const float* body, const float* tail, size_t split, size_t count, float* out) { SD_KERNEL_ACQUIRE;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] += index < split ? body[index] : tail[index - split];
}
__global__ void dspark_finish(const int32_t* tokens, float* confidences, size_t slots, size_t vocabulary,
                              float threshold, size_t minimum, uint32_t* count) { SD_KERNEL_ACQUIRE;
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

int sd_op_dspark_head(void* handle, const float* body, const float* body_tail, uint64_t tail_first, const float* hidden,
                      const void* w1, const void* w2, const float* conf, uint64_t slots, uint64_t vocabulary, uint64_t rank,
                      uint64_t width, uint32_t anchor, const int32_t* anchor_at, const uint32_t* sample_at, float threshold, uint64_t minimum, float* logits,
                      int32_t* tokens, float* confidences, uint32_t* count, float* probabilities, float* scratch, uint64_t scratch_bytes) {
    SD_OP_BEGIN
    if (!slots || slots > 5 || !vocabulary || vocabulary > INT32_MAX || !rank || !width ||
        anchor >= vocabulary || !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
        (body_tail && (!tail_first || tail_first >= vocabulary)) || (probabilities && !sample_at))
        throw std::runtime_error("invalid DSpark head shape, anchor or prefix limits");
    const size_t features = checked_sum({width, rank});
    const size_t feature_bytes = checked_floats({features});
    // every slot's features, kept apart so that the confidences are formed together after the chain
    const size_t partial_offset = checked_sum({checked_product({feature_bytes, slots}), 15}) & ~size_t(15);
    const size_t needed = checked_sum({partial_offset, 2 * argmax_chunks * 4});
    if (scratch_bytes < needed) throw std::runtime_error("DSpark head scratch is too small");
    const unsigned feature_blocks = checked_blocks(features);
    const unsigned logit_blocks = checked_blocks(size_t(vocabulary));
    op_span(value, body, checked_floats({slots, vocabulary}), "DSpark body logits");
    if (body_tail) op_span(value, body_tail, checked_floats({slots, vocabulary - tail_first}), "DSpark body logits' tail");
    op_span(value, hidden, checked_floats({slots, width}), "DSpark collapsed hidden");
    op_span(value, w1, checked_product({vocabulary, rank, 2}), "DSpark BF16 W1");
    op_span(value, w2, checked_product({vocabulary, rank, 2}), "DSpark BF16 W2");
    op_span(value, conf, feature_bytes, "DSpark confidence projection");
    op_span(value, logits, checked_floats({slots, vocabulary}), "DSpark corrected logits");
    op_span(value, tokens, checked_floats({slots}), "DSpark draft tokens");
    op_span(value, confidences, checked_floats({slots}), "DSpark confidence output");
    op_span(value, count, 4, "DSpark prefix length");
    op_span(value, scratch, needed, "DSpark scratch");
    if (anchor_at) op_span(value, anchor_at, 4, "DSpark anchor token");
    if (sample_at) op_span(value, sample_at, 16, "DSpark sampling parameters");
    if (probabilities) op_span(value, probabilities, checked_floats({slots}), "DSpark draw probabilities");
    auto* partial = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scratch) + partial_offset);
    const size_t tail_split = body_tail ? size_t(tail_first) : size_t(vocabulary);
    for (size_t slot = 0; slot < slots; ++slot) {
        float* feature = scratch + slot * features;
        dspark_features<<<feature_blocks, 256, 0, value.stream>>>(hidden, static_cast<const uint16_t*>(w1), tokens,
            slot, size_t(vocabulary), size_t(rank), size_t(width), anchor, anchor_at, feature);
        op_linear(value, static_cast<const uint8_t*>(w2), 30, size_t(vocabulary), size_t(rank), 0,
                  feature + width, 0, 1, logits + slot * vocabulary, 0, nullptr, 0);
        dspark_add_kernel<<<logit_blocks, 256, 0, value.stream>>>(body + slot * vocabulary,
            body_tail ? body_tail + slot * (vocabulary - tail_first) : nullptr, tail_split, size_t(vocabulary), logits + slot * vocabulary);
        if (sample_at) row_sample(value, logits + slot * vocabulary, 1, size_t(vocabulary), 0, sample_at, uint32_t(slot), partial, tokens + slot);
        else row_argmax(value, logits + slot * vocabulary, 1, size_t(vocabulary), 0, partial, tokens + slot);
        if (probabilities) row_draw_probability(value, logits + slot * vocabulary, size_t(vocabulary), tokens + slot, sample_at, partial, probabilities + slot);
    }
    for (size_t first = 0; first < slots; first += size_t(gemv_max_vectors)) {
        const size_t steps = std::min(slots - first, size_t(gemv_max_vectors));
        op_linear(value, reinterpret_cast<const uint8_t*>(conf), 0, 1, features, 0, scratch + first * features, features, steps,
                  confidences + first, 1, nullptr, 0);
    }
    dspark_finish<<<1, 32, 0, value.stream>>>(tokens, confidences, size_t(slots), size_t(vocabulary), threshold, size_t(minimum), count);
    SD_OP_END
}
