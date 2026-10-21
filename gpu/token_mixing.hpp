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

// tm_delta_rule_kernel<128> a wave per value head h and 32 of its columns (blockIdx.y), with its
// arithmetic: each wave forms the step's normalized q and k itself, their sums of squares in the
// block kernel's tree order (lane l takes (x[l]^2 + x[l+64]^2) + (x[l+32]^2 + x[l+96]^2), the
// tree's value at l after its stride-32 step, rounded as there, then the wave halves the rest), so
// no wave waits on another's barrier; a column's sums are the block kernel's. A long pass is
// latency-bound step by step: the next step's inputs load while this one computes.
__device__ __forceinline__ float tm_tree128(const float (&x)[4]) {
    float value = __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])),
                            __fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])));
    for (unsigned offset = 16; offset; offset /= 2) value = __fadd_rn(value, __shfl_down(value, offset, 32));
    return __shfl(value, 0, 32);
}
__global__ void __launch_bounds__(32) tm_delta_rule_wave_kernel(const float* qkv, const float* beta, const float* alpha, const float* dt,
                                                               const float* a, float* state, size_t k_heads, size_t v_heads, size_t steps,
                                                               float eps, float* out) {
    constexpr int DIM = 128;
    const size_t h = blockIdx.x, lane = threadIdx.x, j = size_t(blockIdx.y) * 32 + lane, kh = h % k_heads;
    const size_t channels = (2 * k_heads + v_heads) * DIM;
    __shared__ float q[DIM], k[DIM];
    float column[DIM];
    #pragma unroll
    for (int i = 0; i < DIM; ++i) column[i] = state[(h * DIM + i) * DIM + j];
    float qv[4], kv[4], v = 0.0f, al = 0.0f, be = 0.0f;
    auto load = [&](size_t t) {
        const float* row = qkv + t * channels;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            qv[m] = row[kh * DIM + lane + 32 * m];
            kv[m] = row[k_heads * DIM + kh * DIM + lane + 32 * m];
        }
        v = row[2 * k_heads * DIM + h * DIM + j];
        al = alpha[t * v_heads + h];
        be = beta[t * v_heads + h];
    };
    if (steps) load(0);
    for (size_t t = 0; t < steps; ++t) {
        const float q_sum = tm_tree128(qv), k_sum = tm_tree128(kv);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            q[lane + 32 * m] = qv[m] / sqrtf(q_sum + eps) / sqrtf(float(DIM));
            k[lane + 32 * m] = kv[m] / sqrtf(k_sum + eps);
        }
        const float g = expf(a[h] * tm_softplus(al + dt[h]));
        const float b = tm_sigmoid(be);
        const float value = v;
        if (t + 1 < steps) load(t + 1);
        __syncthreads();
        float predicted = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
            column[i] *= g;
            predicted += column[i] * k[i];
        }
        const float delta = (value - predicted) * b;
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

// A block (256 threads) per row of width values: x / sqrt(mean(x^2) + eps) * weight * SiLU(gate).
__global__ void tm_gated_norm_kernel(const float* x, const float* gate, const float* weight, size_t width, float eps, float* out) {
    const float* row = x + size_t(blockIdx.x) * width;
    const float* gates = gate + size_t(blockIdx.x) * width;
    float* target = out + size_t(blockIdx.x) * width;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) sum += row[col] * row[col];
    __shared__ float total;
    sums[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        const float value = tree256_first_wave(sums, threadIdx.x);
        if (!threadIdx.x) total = value;
    }
    __syncthreads();
    const float scale = rsqrtf(total / float(width) + eps);
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

constexpr size_t tm_attention_threads = 256;
constexpr size_t tm_attention_tile = 64;
constexpr size_t tm_attention_max_dim = 1024;
constexpr size_t tm_attention_max_splits = 64;

// Causal grouped-query attention of many rows (a prefill pass): a block takes 16 consecutive rows
// of one head and walks the keys up to the last one's position in tiles of 16, each tile's keys and
// values staged once in shared memory for all 16 rows (the per-row kernel below reads them once a
// row). Thread (row, key) scores a pair; each row's 16 threads keep its online softmax (max and
// sum by shuffles); thread (row, d) accumulates dimensions d, d + 16, ... of the row's output.
// F32 throughout; keys past a row's position are left out. Output times sigmoid(gate) when given.
constexpr int tm_rows_r = 16, tm_rows_k = 16, tm_rows_dim = 256;
__global__ void __launch_bounds__(256) tm_attention_rows_kernel(const float* q, const float* keys, const float* values,
                                                               const int32_t* positions, size_t rows, size_t heads, size_t kv_heads,
                                                               float scale, const float* gate, float* out) {
    constexpr int R = tm_rows_r, K = tm_rows_k, D = tm_rows_dim;
    __shared__ float qs[R][D], ks[K][D + 1], vs[K][D], ps[R][K];
    const size_t h = blockIdx.y, kh = h / (heads / kv_heads), r0 = size_t(blockIdx.x) * R;
    const unsigned t = threadIdx.x, row = t / 16, sub = t % 16;
    const size_t my_row = r0 + row;
    const bool valid = my_row < rows;
    for (unsigned i = t; i < R * D; i += 256) {
        const size_t r = r0 + i / D;
        qs[i / D][i % D] = r < rows ? q[(r * heads + h) * D + i % D] : 0.0f;
    }
    size_t length = 0;
    for (size_t r = r0; r < std::min(rows, r0 + R); ++r) length = std::max(length, size_t(positions[r]) + 1);
    const size_t mine = valid ? size_t(positions[my_row]) + 1 : 1;
    float m = -INFINITY, l = 0.0f, acc[D / 16];
    #pragma unroll
    for (int c = 0; c < D / 16; ++c) acc[c] = 0.0f;
    __syncthreads();
    for (size_t start = 0; start < length; start += K) {
        for (unsigned i = t; i < K * D; i += 256) {
            const size_t key = start + i / D;
            const bool in = key < length;
            ks[i / D][i % D] = in ? keys[(key * kv_heads + kh) * D + i % D] : 0.0f;
            vs[i / D][i % D] = in ? values[(key * kv_heads + kh) * D + i % D] : 0.0f;
        }
        __syncthreads();
        float score = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < D; ++d) score += qs[row][d] * ks[sub][d];
        const bool seen = start + sub < mine;
        score = seen ? score * scale : -INFINITY;
        float top = score;
        for (unsigned offset = 8; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 16));
        const float next = fmaxf(m, top);
        const float p = seen ? expf(score - next) : 0.0f;
        float total = p;
        for (unsigned offset = 8; offset; offset /= 2) total += __shfl_xor(total, offset, 16);
        const float correction = expf(m - next);   // (0 at the first tile: m = -inf, next finite)
        l = l * correction + total;
        m = next;
        ps[row][sub] = p;
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < D / 16; ++c) {
            float value = acc[c] * correction;
            #pragma unroll
            for (int k = 0; k < K; ++k) value += ps[row][k] * vs[k][sub + 16 * c];
            acc[c] = value;
        }
        __syncthreads();
    }
    if (!valid) return;
    #pragma unroll
    for (int c = 0; c < D / 16; ++c) {
        const size_t at = (my_row * heads + h) * D + sub + 16 * c;
        float value = acc[c] / l;
        if (gate) value *= tm_sigmoid(gate[at]);
        out[at] = value;
    }
}

// Causal grouped-query attention, split over the keys (flash decoding): block (row * heads + h,
// split) takes its share of keys [0, positions[row]] (an equal share of each row's length, so a
// fixed grid serves any length) for q's row and head h (kv head h / (heads / kv_heads)); a warp
// scores a key (lanes over the dimensions), then an online softmax over tiles of 64 keys sums the
// values (a thread per dimension). The split's max score, exp-sum and unnormalized sum go to
// partial[(row * heads + h) * splits + split] (dim + 2 values).
__global__ void tm_attention_split_kernel(const float* q, const float* keys, const float* values, size_t context,
                                          const int32_t* positions, size_t heads, size_t kv_heads, size_t dim, float scale,
                                          float* partial) {
    const size_t pair = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
    const size_t row = pair / heads, h = pair % heads, kh = h / (heads / kv_heads);
    const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32, warps = blockDim.x / 32;
    extern __shared__ float shared[];
    float* qs = shared;
    float* scores = shared + dim;
    for (size_t d = tid; d < dim; d += blockDim.x) qs[d] = q[pair * dim + d];
    const size_t length = std::min(size_t(positions[row]) + 1, context);
    const size_t share = (length + splits - 1) / splits;
    const size_t begin = std::min(length, split * share), end = std::min(length, begin + share);
    __syncthreads();
    float acc[tm_attention_max_dim / tm_attention_threads] = {};
    float m = -INFINITY, l = 0.0f;
    for (size_t start = begin; start < end; start += tm_attention_tile) {
        const size_t count = std::min(tm_attention_tile, end - start);
        for (size_t i = warp; i < count; i += warps) {
            const float* key = keys + ((start + i) * kv_heads + kh) * dim;
            float sum = 0.0f;
            for (size_t d = lane; d < dim; d += 32) sum += qs[d] * key[d];
            for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_xor(sum, offset, 32);
            if (!lane) scores[i] = sum * scale;
        }
        __syncthreads();
        float top = -INFINITY;
        for (size_t i = 0; i < count; ++i) top = fmaxf(top, scores[i]);
        const float next = fmaxf(m, top), correction = expf(m - next);
        __syncthreads();
        if (tid < count) scores[tid] = expf(scores[tid] - next);
        __syncthreads();
        float total = 0.0f;
        for (size_t i = 0; i < count; ++i) total += scores[i];
        l = l * correction + total;
        for (size_t a = 0; a * tm_attention_threads < dim; ++a) {
            const size_t d = tid + a * tm_attention_threads;
            if (d >= dim) break;
            float value = acc[a] * correction;
            for (size_t i = 0; i < count; ++i) value += scores[i] * values[((start + i) * kv_heads + kh) * dim + d];
            acc[a] = value;
        }
        m = next;
        __syncthreads();
    }
    float* mine = partial + (pair * splits + split) * (dim + 2);
    for (size_t a = 0; a * tm_attention_threads < dim; ++a) {
        const size_t d = tid + a * tm_attention_threads;
        if (d < dim) mine[d] = acc[a];
    }
    if (!tid) {
        mine[dim] = m;
        mine[dim + 1] = l;
    }
}

// The splits of a (row, head) combined (each scaled by exp(its max - the max)), normalized, and
// times sigmoid(gate) when a gate is given: out [rows][heads][dim].
// Each split's max and sum are read once into shared memory and its scale formed once there (the
// same values and sums, in the same order, as a thread forming them all itself).
__global__ void tm_attention_combine_kernel(const float* partial, size_t splits, size_t dim, const float* gate, float* out) {
    const size_t pair = blockIdx.x;
    const float* first = partial + pair * splits * (dim + 2);
    __shared__ float maxes[tm_attention_max_splits], sums[tm_attention_max_splits], scales[tm_attention_max_splits];
    for (size_t s = threadIdx.x; s < splits; s += blockDim.x) {
        maxes[s] = first[s * (dim + 2) + dim];
        sums[s] = first[s * (dim + 2) + dim + 1];
    }
    __syncthreads();
    float top = -INFINITY;
    for (size_t s = 0; s < splits; ++s) top = fmaxf(top, maxes[s]);
    for (size_t s = threadIdx.x; s < splits; s += blockDim.x) scales[s] = expf(maxes[s] - top);
    __syncthreads();
    float total = 0.0f;
    for (size_t s = 0; s < splits; ++s) total += sums[s] * scales[s];
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll 8
        for (size_t s = 0; s < splits; ++s) sum += first[s * (dim + 2) + d] * scales[s];
        const size_t at = pair * dim + d;
        float value = sum / total;
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
        tm_delta_rule_wave_kernel<<<dim3(unsigned(v_heads), 4), 32, 0, value.stream>>>(qkv, beta, alpha, dt, a, state, size_t(k_heads),
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
                    uint64_t rows, uint64_t heads, uint64_t kv_heads, uint64_t dim, float scale, const float* gate, float* scratch,
                    uint64_t scratch_bytes, float* out) {
    SD_OP_BEGIN
    if (!rows || !heads || !kv_heads || heads % kv_heads || !dim || dim > tm_attention_max_dim || !context || rows * heads > 65535 * 64)
        throw std::runtime_error("invalid Straydog attention shape");
    op_span(value, q, size_t(rows * heads * dim) * 4, "attention queries");
    op_span(value, keys, size_t(context * kv_heads * dim) * 4, "attention keys");
    op_span(value, values, size_t(context * kv_heads * dim) * 4, "attention values");
    op_span(value, positions, size_t(rows) * 4, "attention positions");
    if (gate) op_span(value, gate, size_t(rows * heads * dim) * 4, "an attention gate");
    op_span(value, out, size_t(rows * heads * dim) * 4, "an attention output");
    // Many rows (a prefill pass): blocks of 16 rows sharing each tile of keys.
    if (rows >= 16 && dim == tm_rows_dim && (rows + tm_rows_r - 1) / tm_rows_r <= 65535 && heads <= 65535) {
        tm_attention_rows_kernel<<<dim3(unsigned((rows + tm_rows_r - 1) / tm_rows_r), unsigned(heads)), 256, 0, value.stream>>>(
            q, keys, values, positions, size_t(rows), size_t(heads), size_t(kv_heads), scale, gate, out);
        check(hipGetLastError());
        return 0;
    }
    // Enough blocks to fill the device whatever the rows: few rows split their keys further.
    const size_t pairs = size_t(rows * heads);
    const size_t splits = std::min(tm_attention_max_splits, std::max<size_t>(1, 256 / pairs));
    const size_t partial_bytes = pairs * splits * (size_t(dim) + 2) * 4;
    if (!scratch || scratch_bytes < partial_bytes) throw std::runtime_error("Straydog attention scratch too small");
    op_span(value, scratch, partial_bytes, "attention scratch");
    const size_t shared = (size_t(dim) + tm_attention_tile) * 4;
    tm_attention_split_kernel<<<dim3(unsigned(pairs), unsigned(splits)), unsigned(tm_attention_threads), shared, value.stream>>>(
        q, keys, values, size_t(context), positions, size_t(heads), size_t(kv_heads), size_t(dim), scale, scratch);
    check(hipGetLastError());
    tm_attention_combine_kernel<<<unsigned(pairs), 256, 0, value.stream>>>(scratch, splits, size_t(dim), gate, out);
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
