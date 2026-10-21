// Token mixers architectures compose (docs/DESIGN.md §10): a depthwise causal convolution, the
// gated delta rule over per-head states, a gated RMS norm, rotary positions over the leading
// dimensions, causal grouped-query attention over a K/V cache, and small elementwise steps. Shapes
// are arguments; every op is queued on the device's stream. Included in the GPU TU after ops.hpp.

__device__ __forceinline__ float tm_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float tm_silu(float x) { return x * tm_sigmoid(x); }
__device__ __forceinline__ float tm_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

constexpr size_t tm_max_kernel = 8;

// A thread per channel: its window of the last `kernel - 1` inputs (history, oldest first) and each
// step's input, SiLU of their dot with the channel's taps; x is overwritten in place and history
// left holding the last inputs.
__global__ void tm_conv_silu_kernel(float* x, float* history, const float* taps, size_t channels, size_t kernel, size_t steps) {
    const size_t c = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float window[tm_max_kernel];
    for (size_t j = 0; j + 1 < kernel; ++j) window[j] = history[j * channels + c];
    for (size_t t = 0; t < steps; ++t) {
        window[kernel - 1] = x[t * channels + c];
        float sum = 0.0f;
        for (size_t j = 0; j < kernel; ++j) sum += window[j] * taps[c * kernel + j];
        for (size_t j = 0; j + 1 < kernel; ++j) window[j] = window[j + 1];
        x[t * channels + c] = tm_silu(sum);
    }
    for (size_t j = 0; j + 1 < kernel; ++j) history[j * channels + c] = window[j];
}

// The gated delta rule, a block per value head h (reading key head h % k_heads) and a thread per
// value column j holding S[.][j] of the head's state S[key][value]: per step, q and k are
// L2-normalized (x / sqrt(sum x^2 + eps), q also divided by sqrt(DIM)); S *= exp(a[h] *
// softplus(alpha + dt[h])); d = (v - S^T k) * sigmoid(beta); S += k d^T; out = S^T q. Rows of qkv
// are [q (k_heads * DIM) | k (k_heads * DIM) | v (v_heads * DIM)].
template<int DIM>
__global__ void tm_delta_rule_kernel(const float* qkv, const float* beta, const float* alpha, const float* dt, const float* a,
                                     float* state, size_t k_heads, size_t v_heads, size_t steps, float eps, float* out) {
    const size_t h = blockIdx.x, j = threadIdx.x, kh = h % k_heads;
    const size_t channels = (2 * k_heads + v_heads) * DIM;
    __shared__ float q[DIM], k[DIM], sums[2][DIM];
    float column[DIM];
    #pragma unroll
    for (int i = 0; i < DIM; ++i) column[i] = state[(h * DIM + i) * DIM + j];
    for (size_t t = 0; t < steps; ++t) {
        const float* row = qkv + t * channels;
        const float qv = row[kh * DIM + j], kv = row[k_heads * DIM + kh * DIM + j];
        sums[0][j] = qv * qv;
        sums[1][j] = kv * kv;
        __syncthreads();
        for (int stride = DIM / 2; stride; stride /= 2) {
            if (int(j) < stride) {
                sums[0][j] += sums[0][j + stride];
                sums[1][j] += sums[1][j + stride];
            }
            __syncthreads();
        }
        q[j] = qv / sqrtf(sums[0][0] + eps) / sqrtf(float(DIM));
        k[j] = kv / sqrtf(sums[1][0] + eps);
        __syncthreads();
        const float g = expf(a[h] * tm_softplus(alpha[t * v_heads + h] + dt[h]));
        const float b = tm_sigmoid(beta[t * v_heads + h]);
        const float v = row[2 * k_heads * DIM + h * DIM + j];
        float predicted = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
            column[i] *= g;
            predicted += column[i] * k[i];
        }
        const float delta = (v - predicted) * b;
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
            column[i] += k[i] * delta;
            result += column[i] * q[i];
        }
        out[(t * v_heads + h) * DIM + j] = result;
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < DIM; ++i) state[(h * DIM + i) * DIM + j] = column[i];
}

// A block per row of width values: x / sqrt(mean(x^2) + eps) * weight * SiLU(gate).
__global__ void tm_gated_norm_kernel(const float* x, const float* gate, const float* weight, size_t width, float eps, float* out) {
    const float* row = x + size_t(blockIdx.x) * width;
    const float* gates = gate + size_t(blockIdx.x) * width;
    float* target = out + size_t(blockIdx.x) * width;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) sum += row[col] * row[col];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / float(width) + eps);
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) target[col] = row[col] * scale * weight[col] * tm_silu(gates[col]);
}

// Each row of x, [a (width) | b (width)], into a row of `first` and one of `second`.
__global__ void tm_split_kernel(const float* x, size_t count, size_t width, float* first, float* second) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t row = index / width, col = index % width;
    first[index] = x[row * 2 * width + col];
    second[index] = x[row * 2 * width + width + col];
}

// The leading 2 * pairs values of each vector rotated: pair i is (i, i + pairs), by the row's
// position times frequencies[i] (angles in double, as the other rotary ops).
__global__ void tm_rope_halves_kernel(float* x, size_t count, size_t vectors, size_t width, size_t pairs,
                                      const int32_t* positions, const float* frequencies) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t row = index / (vectors * pairs), rest = index % (vectors * pairs);
    const size_t vector = rest / pairs, pair = rest % pairs;
    float* v = x + (row * vectors + vector) * width;
    const double angle = double(positions[row]) * double(frequencies[pair]);
    double s, c;
    sincos(angle, &s, &c);
    const double a = v[pair], b = v[pair + pairs];
    v[pair] = float(a * c - b * s);
    v[pair + pairs] = float(a * s + b * c);
}

constexpr size_t tm_attention_tile = 256;
constexpr size_t tm_attention_max_dim = 1024;

// Causal grouped-query attention, a block per (row, head): q [rows][heads][dim] at positions[row]
// attends the cache's keys and values [0, position] ([context][kv_heads][dim]; head h reads kv
// head h / (heads / kv_heads)) by an online softmax over tiles of 256 keys; out [rows][heads][dim],
// times sigmoid(gate) when a gate is given.
__global__ void tm_attention_kernel(const float* q, const float* keys, const float* values, const int32_t* positions, size_t heads,
                                    size_t kv_heads, size_t dim, float scale, const float* gate, float* out) {
    const size_t row = blockIdx.x / heads, h = blockIdx.x % heads, kh = h / (heads / kv_heads), tid = threadIdx.x;
    extern __shared__ float shared[];
    float* qs = shared;
    float* p = shared + dim;
    float* reduce = p + tm_attention_tile;
    for (size_t d = tid; d < dim; d += blockDim.x) qs[d] = q[(row * heads + h) * dim + d];
    __syncthreads();
    const size_t last = size_t(positions[row]);
    float acc[tm_attention_max_dim / tm_attention_tile] = {};
    float m = -INFINITY, l = 0.0f;
    for (size_t start = 0; start <= last; start += tm_attention_tile) {
        const size_t s = start + tid;
        float score = -INFINITY;
        if (s <= last) {
            const float* key = keys + (s * kv_heads + kh) * dim;
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) sum += qs[d] * key[d];
            score = sum * scale;
        }
        reduce[tid] = score;
        __syncthreads();
        for (unsigned stride = tm_attention_tile / 2; stride; stride /= 2) {
            if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            __syncthreads();
        }
        const float next = fmaxf(m, reduce[0]);
        __syncthreads();
        const float correction = expf(m - next);
        const float weight = s <= last ? expf(score - next) : 0.0f;
        p[tid] = weight;
        reduce[tid] = weight;
        __syncthreads();
        for (unsigned stride = tm_attention_tile / 2; stride; stride /= 2) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        l = l * correction + reduce[0];
        const size_t count = last + 1 - start < tm_attention_tile ? last + 1 - start : tm_attention_tile;
        for (size_t a = 0; a * tm_attention_tile < dim; ++a) {
            const size_t d = tid + a * tm_attention_tile;
            if (d >= dim) break;
            float value = acc[a] * correction;
            for (size_t i = 0; i < count; ++i) value += p[i] * values[((start + i) * kv_heads + kh) * dim + d];
            acc[a] = value;
        }
        m = next;
        __syncthreads();
    }
    for (size_t a = 0; a * tm_attention_tile < dim; ++a) {
        const size_t d = tid + a * tm_attention_tile;
        if (d >= dim) break;
        const size_t at = (row * heads + h) * dim + d;
        float value = acc[a] / l;
        if (gate) value *= tm_sigmoid(gate[at]);
        out[at] = value;
    }
}

__global__ void tm_add_kernel(const float* x, size_t count, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] += x[index];
}

// Each row of x times sigmoid of its scalar.
__global__ void tm_scale_rows_kernel(float* x, const float* scalars, size_t count, size_t width) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) x[index] *= tm_sigmoid(scalars[index / width]);
}

int sd_op_conv_silu(void* handle, float* x, float* history, const float* taps, uint64_t channels, uint64_t kernel, uint64_t steps) {
    SD_OP_BEGIN
    if (!channels || !kernel || kernel > tm_max_kernel || !steps) throw std::runtime_error("invalid Straydog convolution shape");
    op_span(value, x, size_t(steps * channels) * 4, "a convolution input");
    if (kernel > 1) op_span(value, history, size_t((kernel - 1) * channels) * 4, "a convolution history");
    op_span(value, taps, size_t(kernel * channels) * 4, "convolution taps");
    tm_conv_silu_kernel<<<op_blocks(size_t(channels)), 256, 0, value.stream>>>(x, history, taps, size_t(channels), size_t(kernel),
                                                                            size_t(steps));
    SD_OP_END
}

int sd_op_delta_rule(void* handle, const float* qkv, const float* beta, const float* alpha, const float* dt, const float* a, float* state,
                     uint64_t k_heads, uint64_t v_heads, uint64_t dim, uint64_t steps, float eps, float* out) {
    SD_OP_BEGIN
    if (!k_heads || !v_heads || v_heads % k_heads || !steps || (dim != 64 && dim != 128))
        throw std::runtime_error("invalid Straydog delta rule shape (key and value dimensions of 64 or 128)");
    const size_t channels = size_t((2 * k_heads + v_heads) * dim);
    op_span(value, qkv, size_t(steps) * channels * 4, "a delta rule input");
    op_span(value, beta, size_t(steps * v_heads) * 4, "delta rule betas");
    op_span(value, alpha, size_t(steps * v_heads) * 4, "delta rule alphas");
    op_span(value, dt, size_t(v_heads) * 4, "a delta rule time step bias");
    op_span(value, a, size_t(v_heads) * 4, "delta rule decays");
    op_span(value, state, size_t(v_heads * dim * dim) * 4, "a delta rule state");
    op_span(value, out, size_t(steps * v_heads * dim) * 4, "a delta rule output");
    if (dim == 128)
        tm_delta_rule_kernel<128><<<unsigned(v_heads), 128, 0, value.stream>>>(qkv, beta, alpha, dt, a, state, size_t(k_heads),
                                                                           size_t(v_heads), size_t(steps), eps, out);
    else
        tm_delta_rule_kernel<64><<<unsigned(v_heads), 64, 0, value.stream>>>(qkv, beta, alpha, dt, a, state, size_t(k_heads),
                                                                         size_t(v_heads), size_t(steps), eps, out);
    SD_OP_END
}

int sd_op_gated_norm(void* handle, const float* x, const float* gate, const float* weight, uint64_t rows, uint64_t width, float eps,
                     float* out) {
    SD_OP_BEGIN
    if (!rows || !width) throw std::runtime_error("invalid Straydog gated norm shape");
    op_span(value, x, size_t(rows * width) * 4, "a gated norm input");
    op_span(value, gate, size_t(rows * width) * 4, "a gated norm gate");
    op_span(value, weight, size_t(width) * 4, "a gated norm weight");
    op_span(value, out, size_t(rows * width) * 4, "a gated norm output");
    tm_gated_norm_kernel<<<unsigned(rows), 256, 0, value.stream>>>(x, gate, weight, size_t(width), eps, out);
    SD_OP_END
}

int sd_op_split(void* handle, const float* x, uint64_t rows, uint64_t width, float* first, float* second) {
    SD_OP_BEGIN
    if (!rows || !width) throw std::runtime_error("invalid Straydog split shape");
    op_span(value, x, size_t(rows * 2 * width) * 4, "a split input");
    op_span(value, first, size_t(rows * width) * 4, "a split output");
    op_span(value, second, size_t(rows * width) * 4, "a split output");
    tm_split_kernel<<<op_blocks(size_t(rows * width)), 256, 0, value.stream>>>(x, size_t(rows * width), size_t(width), first, second);
    SD_OP_END
}

int sd_op_rope_halves(void* handle, float* x, uint64_t rows, uint64_t vectors, uint64_t width, uint64_t rope_dims,
                      const int32_t* positions, const float* frequencies) {
    SD_OP_BEGIN
    if (!rows || !vectors || !rope_dims || rope_dims % 2 || rope_dims > width) throw std::runtime_error("invalid Straydog rope shape");
    op_span(value, x, size_t(rows * vectors * width) * 4, "a rotated buffer");
    op_span(value, positions, size_t(rows) * 4, "rope positions");
    op_span(value, frequencies, size_t(rope_dims / 2) * 4, "rope frequencies");
    const size_t count = size_t(rows * vectors * (rope_dims / 2));
    tm_rope_halves_kernel<<<op_blocks(count), 256, 0, value.stream>>>(x, count, size_t(vectors), size_t(width), size_t(rope_dims / 2),
                                                                     positions, frequencies);
    SD_OP_END
}

int sd_op_attention(void* handle, const float* q, const float* keys, const float* values, uint64_t context, const int32_t* positions,
                    uint64_t rows, uint64_t heads, uint64_t kv_heads, uint64_t dim, float scale, const float* gate, float* out) {
    SD_OP_BEGIN
    if (!rows || !heads || !kv_heads || heads % kv_heads || !dim || dim > tm_attention_max_dim || !context)
        throw std::runtime_error("invalid Straydog attention shape");
    op_span(value, q, size_t(rows * heads * dim) * 4, "attention queries");
    op_span(value, keys, size_t(context * kv_heads * dim) * 4, "attention keys");
    op_span(value, values, size_t(context * kv_heads * dim) * 4, "attention values");
    op_span(value, positions, size_t(rows) * 4, "attention positions");
    if (gate) op_span(value, gate, size_t(rows * heads * dim) * 4, "an attention gate");
    op_span(value, out, size_t(rows * heads * dim) * 4, "an attention output");
    // Positions past the cache read past it: the caller keeps them inside [0, context).
    const size_t shared = (size_t(dim) + 2 * tm_attention_tile) * 4;
    tm_attention_kernel<<<unsigned(rows * heads), unsigned(tm_attention_tile), shared, value.stream>>>(
        q, keys, values, positions, size_t(heads), size_t(kv_heads), size_t(dim), scale, gate, out);
    SD_OP_END
}

int sd_op_add(void* handle, const float* x, uint64_t count, float* out) {
    SD_OP_BEGIN
    op_span(value, x, size_t(count) * 4, "an addend");
    op_span(value, out, size_t(count) * 4, "a sum");
    tm_add_kernel<<<op_blocks(size_t(count)), 256, 0, value.stream>>>(x, size_t(count), out);
    SD_OP_END
}

int sd_op_scale_rows(void* handle, float* x, const float* scalars, uint64_t rows, uint64_t width) {
    SD_OP_BEGIN
    op_span(value, x, size_t(rows * width) * 4, "scaled rows");
    op_span(value, scalars, size_t(rows) * 4, "row scalars");
    tm_scale_rows_kernel<<<op_blocks(size_t(rows * width)), 256, 0, value.stream>>>(x, scalars, size_t(rows * width), size_t(width));
    SD_OP_END
}
