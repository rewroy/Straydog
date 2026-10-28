// DSpark's noncausal block attention: every query sees the history positions [anchor - history, anchor) of
// the K = V ring and all the block's keys, with the head's sink in the denominator only. One block per (row,
// head), the arithmetic of op_sparse_attention_kernel (a key's score one chain of fused multiply-adds over
// the 512 dims in order, the weights' sum the same tree, an output dim one chain over the keys in order) with
// the loads wide and several in flight: that kernel waits on each load in turn, which kept a block ~100 us
// busy over 262 KB of keys.
namespace {
constexpr size_t dspark_dim = 512, dspark_ring_rows = 128;
__global__ void __launch_bounds__(256) dspark_attention_kernel(const float* q, const float* ring, const float* block,
                                                               const float* sinks, size_t heads, size_t slots, uint32_t anchor,
                                                               const int32_t* anchor_at, size_t window, float scale, float* out) {
    const size_t row = blockIdx.x / heads, head = blockIdx.x % heads;
    const int64_t p = int64_t(anchor_at ? anchor_at[0] : int32_t(anchor)) - 1;
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + slots;
    const float* query = q + (row * heads + head) * dspark_dim;
    extern __shared__ __align__(16) float dspark_shared[];
    float* qs = dspark_shared;                     // dspark_dim
    float* scores = dspark_shared + dspark_dim;    // total
    for (size_t i = threadIdx.x; i < dspark_dim; i += blockDim.x) qs[i] = query[i];
    __syncthreads();
    const auto key = [&](size_t index) {
        return index < raw_count ? ring + size_t(first + int64_t(index)) % dspark_ring_rows * dspark_dim
                                 : block + (index - raw_count) * dspark_dim;
    };
    for (size_t index = threadIdx.x; index < total; index += blockDim.x) {
        const float* k = key(index);
        float dot = 0.0f;
        for (size_t i = 0; i < dspark_dim; i += 16) {
            float4 loaded[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) loaded[j] = *reinterpret_cast<const float4*>(k + i + 4 * j);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 a = *reinterpret_cast<const float4*>(qs + i + 4 * j);
                dot = fmaf(a.x, loaded[j].x, dot); dot = fmaf(a.y, loaded[j].y, dot);
                dot = fmaf(a.z, loaded[j].z, dot); dot = fmaf(a.w, loaded[j].w, dot);
            }
        }
        scores[index] = dot * scale;
    }
    __syncthreads();
    __shared__ float reduce[256];
    float top = sinks[head];
    for (size_t index = threadIdx.x; index < total; index += blockDim.x) top = fmaxf(top, scores[index]);
    reduce[threadIdx.x] = top;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        __syncthreads();
    }
    top = reduce[0];
    __syncthreads();
    float sum = 0.0f;
    for (size_t index = threadIdx.x; index < total; index += blockDim.x) {
        const float w = scores[index] == -INFINITY ? 0.0f : expf(scores[index] - top);
        scores[index] = w; sum += w;
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = reduce[0] + expf(sinks[head] - top);
    // two adjacent dims a thread; a key of weight 0 adds nothing (selected, not skipped: the loads of the
    // next keys are then already in flight)
    const size_t dim = 2 * size_t(threadIdx.x);
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll 8
    for (size_t index = 0; index < total; ++index) {
        const float w = scores[index];
        const float2 v = *reinterpret_cast<const float2*>(key(index) + dim);
        a0 = w == 0.0f ? a0 : fmaf(w, v.x, a0);
        a1 = w == 0.0f ? a1 : fmaf(w, v.y, a1);
    }
    float* o = out + (row * heads + head) * dspark_dim;
    o[dim] = a0 / denominator;
    o[dim + 1] = a1 / denominator;
}
}

int sd_op_dspark_attention(void* handle, const float* q, const float* ring, const float* block,
                           const float* sinks, uint64_t heads, uint64_t slots, uint32_t anchor, const int32_t* anchor_at,
                           uint64_t history, float* out) {
    SD_OP_BEGIN
    if (!heads || heads > 64 || !slots || slots > 5 || history > 128 || anchor > INT32_MAX || history > anchor)
        throw std::runtime_error("invalid DSpark attention block or history range");
    op_span(value, q, checked_floats({slots, heads, 512}), "DSpark rotated queries");
    op_span(value, ring, 128 * 512 * 4, "DSpark history K ring");
    op_span(value, block, checked_floats({slots, 512}), "DSpark rotated block keys");
    op_span(value, sinks, checked_floats({heads}), "DSpark attention sinks");
    op_span(value, out, checked_floats({slots, heads, 512}), "DSpark attention output");
    if (anchor_at) op_span(value, anchor_at, 4, "DSpark anchor position");
    // p = anchor - 1 and window = history give exactly [anchor - history, anchor).
    // For anchor = history = 0, p = -1 gives zero raw keys. All block keys remain visible.
    const size_t shared = size_t(512 + history + slots) * 4;
    dspark_attention_kernel<<<unsigned(slots * heads), 256, shared, value.stream>>>(q, ring, block, sinks, size_t(heads), size_t(slots),
        anchor, anchor_at, size_t(history), 1.0f / sqrtf(512.0f), out);
    SD_OP_END
}
