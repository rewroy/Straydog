// Qwen3.8's vision tower on the device, one block at a time, and its merger: the same arithmetic as the CPU encoder
// (straydog-engine's qwen_vision.rs), with FP16 products and F32 sums (vision.hpp's GEMMs and softmax) where the CPU has F32
// throughout. The caller embeds the patches (in merge order) and computes each patch's rope angles; it uploads each block's
// weights, its products converted to FP16 (sd_vision_half), before the block runs.
//
// Per block: x += out(attention(rope(qkv(norm1(x))))), x += down(gelu_tanh(up(norm2(x)))), each product with its bias, the
// norms LayerNorm with weight and bias. Attention is full: 16 heads of 72, each head's first 36 values turned against its last 36
// by the patch's angles (the first 18 by its row, the next 18 by its column). The merger: post norm, each run of four patches
// (a 2x2 block) as one row of 4608, through mm.0, GELU (erf) and mm.2 to the text model's width.

constexpr size_t qwen_vision_width = 1152, qwen_vision_heads = 16, qwen_vision_dim = 72, qwen_vision_ffn = 4304;
constexpr size_t qwen_vision_rope = qwen_vision_dim / 2;
constexpr size_t qwen_vision_merged = qwen_vision_width * 4, qwen_vision_row = 2560;
constexpr float qwen_vision_eps = 1e-6f;

// LayerNorm of each row of x (F32) with weight and bias, into FP16.
__global__ void qwen_vision_layer_norm(const float* x, const float* weight, const float* bias, size_t width, float eps, __half* out) {
    const float* row = x + size_t(blockIdx.x) * width;
    __shared__ float partial[256];
    float sum = 0;
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) sum += row[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = partial[0] / float(width);
    __syncthreads();
    float squares = 0;
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) squares += (row[i] - mean) * (row[i] - mean);
    partial[threadIdx.x] = squares;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / float(width) + eps);
    for (size_t i = threadIdx.x; i < width; i += blockDim.x)
        out[size_t(blockIdx.x) * width + i] = __float2half((row[i] - mean) * scale * weight[i] + bias[i]);
}

// q and k (F32) and v (FP16) of each head ([heads][patches][dim]) from the projection ([patches][3][heads][dim], F32) and its bias: q
// and k turned (value j against value j + rope by the patch's angle j), q scaled by 1/sqrt(dim). (q and k stay F32: the tower's attention
// has logits large enough that FP16 queries and keys moved a row of its output by 9% of its largest value.)
__global__ void qwen_vision_rope_split(const float* qkv, const float* bias, const float* cosines, const float* sines, size_t patches,
                                       float* q, float* k, __half* v) {
    constexpr size_t heads = qwen_vision_heads, dim = qwen_vision_dim, rope = qwen_vision_rope, width = qwen_vision_width;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= patches * heads * rope) return;
    const size_t pair = i % rope, head = (i / rope) % heads, patch = i / (rope * heads);
    const float* row = qkv + patch * 3 * width;
    const size_t at = head * dim + pair;
    const float c = cosines[patch * rope + pair], s = sines[patch * rope + pair];
    const size_t out = (head * patches + patch) * dim + pair;
    const float q_scale = rsqrtf(float(dim));
    for (int part = 0; part < 2; ++part) {
        const float a = row[part * width + at] + bias[part * width + at];
        const float b = row[part * width + at + rope] + bias[part * width + at + rope];
        const float scale = part == 0 ? q_scale : 1.0f;
        float* target = part == 0 ? q : k;
        target[out] = (a * c - b * s) * scale;
        target[out + rope] = (b * c + a * s) * scale;
    }
    v[out] = __float2half(row[2 * width + at] + bias[2 * width + at]);
    v[out + rope] = __float2half(row[2 * width + at + rope] + bias[2 * width + at + rope]);
}

// gelu_pytorch_tanh of each value plus its column's bias, into FP16.
__global__ void qwen_vision_gelu_tanh(const float* x, const float* bias, size_t rows, size_t width, __half* out) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const float value = x[i] + bias[i % width];
    const float inner = 0.7978845608f * (value + 0.044715f * value * value * value);
    out[i] = __float2half(0.5f * value * (1.0f + tanhf(inner)));
}

// Rows of a block's FFN a product takes at a time.
constexpr size_t qwen_vision_ffn_rows = 2048;

// A block's activations in its scratch: the normed input, q, k, v and the attention's output stay; one region holds in turn the
// projection to q, k and v, a run of rows' attention scores and probabilities, and the FFN's rows.
struct QwenVisionScratch {
    size_t query_rows = 0, ffn_rows = 0, bytes = 0;
    size_t normed = 0, q = 0, k = 0, v = 0, attended = 0, phase = 0, probs = 0, middle = 0;
};

static QwenVisionScratch qwen_vision_scratch(size_t n) {
    constexpr size_t width = qwen_vision_width, heads = qwen_vision_heads, ffn = qwen_vision_ffn;
    auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    QwenVisionScratch s;
    s.query_rows = std::max<size_t>(1, std::min(n, vision_score_bytes / (heads * n * 4)));
    s.ffn_rows = std::min(n, qwen_vision_ffn_rows);
    size_t at = 0;
    auto take = [&](size_t bytes) { const size_t start = at; at += align(bytes); return start; };
    s.normed = take(n * width * 2);
    s.q = take(n * width * 4);
    s.k = take(n * width * 4);
    s.v = take(n * width * 2);
    s.attended = take(n * width * 2);
    s.probs = align(heads * s.query_rows * n * 4);
    s.middle = align(s.ffn_rows * ffn * 4);
    const size_t phase = std::max({n * 3 * width * 4, s.probs + heads * s.query_rows * n * 2, s.middle + s.ffn_rows * ffn * 2});
    s.phase = take(phase);
    s.bytes = at;
    return s;
}

// The merger's scratch: the normed rows (FP16, four patches a row), its middle (F32 and FP16).
static size_t qwen_vision_merge_scratch(size_t n) {
    auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    const size_t tokens = n / 4;
    return align(n * qwen_vision_width * 2) + align(tokens * qwen_vision_merged * 4) + align(tokens * qwen_vision_merged * 2);
}

static void qwen_vision_block(Device& d, const sd_qwen_vision_layer& w, float* x, const float* cosines, const float* sines, size_t n,
                              uint8_t* scratch, size_t scratch_bytes) {
    if (!d.blas) throw std::runtime_error("the vision encoder needs the device's GEMM handle");
    if (!n || n % 4) throw std::runtime_error("invalid Straydog Qwen vision patches");
    constexpr size_t width = qwen_vision_width, heads = qwen_vision_heads, dim = qwen_vision_dim, ffn = qwen_vision_ffn;
    const QwenVisionScratch s = qwen_vision_scratch(n);
    if (s.bytes > scratch_bytes) throw std::runtime_error("the vision encoder's scratch is too small");
    op_span(d, scratch, s.bytes, "the vision encoder's scratch");
    op_span(d, x, n * width * 4, "the vision encoder's rows");
    op_span(d, cosines, n * qwen_vision_rope * 4, "the vision encoder's angles");
    op_span(d, sines, n * qwen_vision_rope * 4, "the vision encoder's angles");
    auto f32 = [&](size_t offset) { return reinterpret_cast<float*>(scratch + offset); };
    auto f16 = [&](size_t offset) { return reinterpret_cast<__half*>(scratch + offset); };
    qwen_vision_layer_norm<<<unsigned(n), 256, 0, d.stream>>>(x, static_cast<const float*>(w.norm1), static_cast<const float*>(w.norm1_bias),
        width, qwen_vision_eps, f16(s.normed));
    vision_gemm(d, w.qkv, 3 * width, width, f16(s.normed), n, f32(s.phase), false, 0);
    qwen_vision_rope_split<<<op_blocks(n * heads * qwen_vision_rope), 256, 0, d.stream>>>(f32(s.phase), static_cast<const float*>(w.qkv_bias),
        cosines, sines, n, f32(s.q), f32(s.k), f16(s.v));
    float* scores = f32(s.phase);
    __half* probs = f16(s.phase + s.probs);
    for (size_t first = 0; first < n; first += s.query_rows) {
        const size_t rows = std::min(s.query_rows, n - first);
        const float alpha = 1, zero = 0;
        // scores[h][row][patch] = q[h][first + row] . k[h][patch], in F32 throughout
        blas_check(hipblasGemmStridedBatchedEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(n), int(rows), int(dim), &alpha,
            f32(s.k), HIP_R_32F, int(dim), (long long)(n * dim), f32(s.q) + first * dim, HIP_R_32F, int(dim), (long long)(n * dim),
            &zero, scores, HIP_R_32F, int(n), (long long)(s.query_rows * n), int(heads), HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        vision_softmax_rows(d, scores, rows, heads, n, s.query_rows, probs);
        // attended[first + row][h * dim ..] = probs[h][row] . v[h]
        blas_check(hipblasGemmStridedBatchedEx(d.blas, HIPBLAS_OP_N, HIPBLAS_OP_N, int(dim), int(rows), int(n), &alpha,
            f16(s.v), HIP_R_16F, int(dim), (long long)(n * dim), probs, HIP_R_16F, int(n), (long long)(s.query_rows * n),
            &zero, f16(s.attended) + first * width, HIP_R_16F, int(width), (long long)dim, int(heads), HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT));
    }
    vision_gemm(d, w.out, width, width, f16(s.attended), n, x, false, 1);
    vision_add_bias<<<op_blocks(n * width), 256, 0, d.stream>>>(x, static_cast<const float*>(w.out_bias), n, width);
    qwen_vision_layer_norm<<<unsigned(n), 256, 0, d.stream>>>(x, static_cast<const float*>(w.norm2), static_cast<const float*>(w.norm2_bias),
        width, qwen_vision_eps, f16(s.normed));
    for (size_t first = 0; first < n; first += s.ffn_rows) {
        const size_t rows = std::min(s.ffn_rows, n - first);
        vision_gemm(d, w.up, ffn, width, f16(s.normed) + first * width, rows, f32(s.phase), false, 0);
        qwen_vision_gelu_tanh<<<op_blocks(rows * ffn), 256, 0, d.stream>>>(f32(s.phase), static_cast<const float*>(w.up_bias), rows, ffn,
            f16(s.phase + s.middle));
        vision_gemm(d, w.down, width, ffn, f16(s.phase + s.middle), rows, x + first * width, false, 1);
    }
    vision_add_bias<<<op_blocks(n * width), 256, 0, d.stream>>>(x, static_cast<const float*>(w.down_bias), n, width);
    check(hipGetLastError());
}

static void qwen_vision_merge(Device& d, const float* x, const sd_qwen_vision_merger& w, size_t n, uint8_t* scratch, size_t scratch_bytes,
                              float* out) {
    if (!d.blas) throw std::runtime_error("the vision encoder needs the device's GEMM handle");
    if (!n || n % 4) throw std::runtime_error("invalid Straydog Qwen vision patches");
    const size_t bytes = qwen_vision_merge_scratch(n), tokens = n / 4;
    if (bytes > scratch_bytes) throw std::runtime_error("the vision encoder's scratch is too small");
    op_span(d, scratch, bytes, "the vision encoder's scratch");
    op_span(d, x, n * qwen_vision_width * 4, "the vision encoder's rows");
    op_span(d, out, tokens * qwen_vision_row * 4, "image rows");
    auto align = [](size_t value) { return (value + 255) / 256 * 256; };
    __half* normed = reinterpret_cast<__half*>(scratch);
    float* middle = reinterpret_cast<float*>(scratch + align(n * qwen_vision_width * 2));
    __half* middle_half = reinterpret_cast<__half*>(scratch + align(n * qwen_vision_width * 2) + align(tokens * qwen_vision_merged * 4));
    qwen_vision_layer_norm<<<unsigned(n), 256, 0, d.stream>>>(x, static_cast<const float*>(w.norm), static_cast<const float*>(w.norm_bias),
        qwen_vision_width, qwen_vision_eps, normed);
    // (four patches of a block are four rows in a row: one row of the merger's input)
    vision_gemm(d, w.up, qwen_vision_merged, qwen_vision_merged, normed, tokens, middle, false, 0);
    vision_gelu<<<op_blocks(tokens * qwen_vision_merged), 256, 0, d.stream>>>(middle, static_cast<const float*>(w.up_bias), tokens,
        qwen_vision_merged, middle_half);
    vision_gemm(d, w.down, qwen_vision_row, qwen_vision_merged, middle_half, tokens, out, false, 0);
    vision_add_bias<<<op_blocks(tokens * qwen_vision_row), 256, 0, d.stream>>>(out, static_cast<const float*>(w.down_bias), tokens,
        qwen_vision_row);
    check(hipGetLastError());
}

int sd_qwen_vision_scratch_bytes(uint64_t patches, uint64_t* bytes) {
    try {
        if (!bytes || !patches || patches % 4) throw std::runtime_error("invalid Straydog Qwen vision patches");
        *bytes = std::max(qwen_vision_scratch(size_t(patches)).bytes, qwen_vision_merge_scratch(size_t(patches)));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_qwen_vision_block(void* handle, const sd_qwen_vision_layer* layer, float* x, const float* cosines, const float* sines,
                         uint64_t patches, void* scratch, uint64_t scratch_bytes) {
    try {
        auto& value = device(handle);
        if (!layer) throw std::runtime_error("no vision block");
        qwen_vision_block(value, *layer, x, cosines, sines, size_t(patches), static_cast<uint8_t*>(scratch), size_t(scratch_bytes));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_qwen_vision_merge(void* handle, const float* x, const sd_qwen_vision_merger* merger, uint64_t patches, void* scratch,
                         uint64_t scratch_bytes, float* out) {
    try {
        auto& value = device(handle);
        if (!merger) throw std::runtime_error("no vision merger");
        qwen_vision_merge(value, x, *merger, size_t(patches), static_cast<uint8_t*>(scratch), size_t(scratch_bytes), out);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
