// Generic operations over device buffers, queued on the device's stream (docs/DESIGN.md §10):
// shapes are arguments, so any architecture composes them. Included in the GPU TU.

// The allocation holding [pointer, pointer + bytes) of this device, or an error: an op never
// touches memory the device did not allocate.
void op_span(Device& d, const void* pointer, size_t bytes, const char* what) {
    const auto* at = static_cast<const uint8_t*>(pointer);
    for (const auto& entry : d.allocations) {
        const auto* base = static_cast<const uint8_t*>(entry.first);
        if (at >= base && bytes <= entry.second && size_t(at - base) <= entry.second - bytes) return;
    }
    throw std::runtime_error(std::string("Straydog op: ") + what + " is outside this device's allocations");
}

// One block per row: x / sqrt(mean(x^2) + eps) * weight.
__global__ void op_rms_norm_kernel(const float* input, const float* weight, size_t width, float eps, float* output) {
    const float* row = input + size_t(blockIdx.x) * width;
    float* out = output + size_t(blockIdx.x) * width;
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
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) out[col] = row[col] * scale * weight[col];
}

// Rotate the last 2 * pairs values of each vector (adjacent pairs) by position * frequency; the
// angle is formed in double so long positions keep their precision.
__global__ void op_rope_kernel(float* x, size_t count, size_t vectors, size_t width, size_t pairs, const int32_t* positions,
                               const float* frequencies, int inverse) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t row = index / (vectors * pairs), rest = index % (vectors * pairs);
    const size_t vector = rest / pairs, pair = rest % pairs;
    float* v = x + (row * vectors + vector) * width + (width - 2 * pairs) + 2 * pair;
    const double angle = double(positions[row]) * double(frequencies[pair]) * (inverse ? -1.0 : 1.0);
    double s, c;
    sincos(angle, &s, &c);
    const double a = v[0], b = v[1];
    v[0] = float(a * c - b * s);
    v[1] = float(a * s + b * c);
}

// out[t][r] = W[first + r] . in[t] for `rows` rows, inputs and outputs `input_stride` and
// `output_stride` values apart: GEMV for up to gemv_max_vectors inputs, else the rows expanded
// to F32 in `scratch` (in pieces that fit) and an F32 GEMM.
void op_linear(Device& d, const uint8_t* weights, int kind, size_t rows, size_t cols, size_t first,
               const float* input, size_t input_stride, size_t steps, float* output, size_t output_stride,
               float* scratch, size_t scratch_bytes) {
    const size_t row_bytes = matrix_row_bytes(kind, cols);
    const Matrix matrix{weights + first * row_bytes, row_bytes, rows, cols, 0, 0, 0, kind};
    if (steps <= size_t(gemv_max_vectors)) {
        if (steps == 1) gemv(d, &matrix, 1, input, output);
        else gemv(d, &matrix, 1, input, output, steps, input_stride, output_stride);   // strides in values
        return;
    }
    if (!d.blas) throw std::runtime_error("Straydog op: GEMM needs a BLAS handle");
    const size_t piece = std::min(rows, scratch_bytes / (cols * 4));
    if (!piece) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
    const float alpha = 1, beta = 0;
    for (size_t done = 0; done < rows; done += piece) {
        const size_t count = std::min(piece, rows - done);
        const Matrix part{matrix.data + done * row_bytes, row_bytes, count, cols, 0, 0, 0, kind};
        expand(part, scratch, d.stream, false);
        blas_check(hipblasSgemm(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(count), int(steps), int(cols), &alpha,
                                scratch, int(cols), input, int(input_stride), &beta, output + done, int(output_stride)));
    }
}

// ---- Hyper-connections: n parallel copies of the residual stream ----

// One block per row: the mixing coefficients of a sublayer from mixes = W_hc . stream (the
// projection is a linear op), scaled by the stream's RMS: pre and post by sigmoids, the
// combination doubly stochastic by Sinkhorn (softmax rows, then alternate normalizations).
__global__ void op_hc_coefficients_kernel(const float* mixes, const float* stream, size_t hc, size_t width,
                                          const float* base, const float* scale, size_t iterations, float eps,
                                          float norm_eps, float* pre, float* post, float* comb) {
    const size_t row = blockIdx.x, flat = hc * width, mix = (2 + hc) * hc;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < flat; i += blockDim.x) { const float v = stream[row * flat + i]; sum += v * v; }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x) return;
    const float rsqrt = rsqrtf(sums[0] / float(flat) + norm_eps);
    const float* m = mixes + row * mix;
    const auto sigmoid = [](float x) { return 1.0f / (1.0f + expf(-x)); };
    for (size_t j = 0; j < hc; ++j) {
        pre[row * hc + j] = sigmoid(m[j] * rsqrt * scale[0] + base[j]) + eps;
        post[row * hc + j] = 2.0f * sigmoid(m[hc + j] * rsqrt * scale[1] + base[hc + j]);
    }
    float c[64];
    for (size_t j = 0; j < hc; ++j) {
        float top = -INFINITY;
        for (size_t k = 0; k < hc; ++k) {
            c[j * hc + k] = m[2 * hc + j * hc + k] * rsqrt * scale[2] + base[2 * hc + j * hc + k];
            top = fmaxf(top, c[j * hc + k]);
        }
        float total = 0.0f;
        for (size_t k = 0; k < hc; ++k) total += expf(c[j * hc + k] - top);
        for (size_t k = 0; k < hc; ++k) c[j * hc + k] = expf(c[j * hc + k] - top) / total + eps;
    }
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        if (iteration)
            for (size_t j = 0; j < hc; ++j) {
                float total = 0.0f;
                for (size_t k = 0; k < hc; ++k) total += c[j * hc + k];
                for (size_t k = 0; k < hc; ++k) c[j * hc + k] /= total + eps;
            }
        for (size_t k = 0; k < hc; ++k) {
            float total = 0.0f;
            for (size_t j = 0; j < hc; ++j) total += c[j * hc + k];
            for (size_t j = 0; j < hc; ++j) c[j * hc + k] /= total + eps;
        }
    }
    for (size_t i = 0; i < hc * hc; ++i) comb[row * hc * hc + i] = c[i];
}

// out[t][d] = sum_j pre[t][j] stream[t][j][d]
__global__ void op_hc_collapse_kernel(const float* stream, const float* pre, size_t rows, size_t hc, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    const size_t row = index / width, col = index % width;
    float sum = 0.0f;
    for (size_t j = 0; j < hc; ++j) sum += pre[row * hc + j] * stream[(row * hc + j) * width + col];
    out[index] = sum;
}

// out[t][j][d] = post[t][j] x[t][d] + sum_i comb[t][i][j] residual[t][i][d]
__global__ void op_hc_expand_kernel(const float* x, const float* residual, const float* post, const float* comb,
                                    size_t rows, size_t hc, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * hc * width) return;
    const size_t row = index / (hc * width), j = (index / width) % hc, col = index % width;
    float sum = post[row * hc + j] * x[row * width + col];
    for (size_t i = 0; i < hc; ++i) sum += comb[(row * hc + i) * hc + j] * residual[(row * hc + i) * width + col];
    out[index] = sum;
}

// ---- Engram: n-gram table rows written into the stream, gated per copy ----

// One block per (row, copy): with key and value from kv = W . rows ([hc keys | value]),
// dot = sum(h * w * key) / (rms(h) rms(key) sqrt(width)), gate = sigmoid(signed sqrt(|dot|)),
// and the copy h += gate * value.
__global__ void op_engram_gate_kernel(float* stream, const float* kv, const float* weight, size_t hc, size_t width,
                                      float eps) {
    const size_t row = blockIdx.x / hc, j = blockIdx.x % hc;
    float* h = stream + (row * hc + j) * width;
    const float* key = kv + row * (hc + 1) * width + j * width;
    const float* value = kv + row * (hc + 1) * width + hc * width;
    const float* w = weight + j * width;
    __shared__ float sums[3][256];
    float hh = 0.0f, kk = 0.0f, dot = 0.0f;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) {
        hh += h[col] * h[col];
        kk += key[col] * key[col];
        dot += h[col] * w[col] * key[col];
    }
    sums[0][threadIdx.x] = hh; sums[1][threadIdx.x] = kk; sums[2][threadIdx.x] = dot;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            for (int s = 0; s < 3; ++s) sums[s][threadIdx.x] += sums[s][threadIdx.x + stride];
        __syncthreads();
    }
    const float rstd = rsqrtf(sums[0][0] / float(width) + eps) * rsqrtf(sums[1][0] / float(width) + eps);
    const float d = sums[2][0] * rstd / sqrtf(float(width));
    const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d)));
    __syncthreads();
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) h[col] += gate * value[col];
}

// ---- Mixture of experts ----

// One block per row: scores from the router logits (0 softmax, 1 sigmoid, 2 square-root
// softplus), the k experts with the highest score + bias (lowest index on ties), their
// unbiased scores normalized (when asked) and scaled.
__global__ void op_route_kernel(const float* logits, const float* bias, size_t experts, size_t k, int gating,
                                int normalize, float scale, int32_t* ids, float* weights) {
    const size_t row = blockIdx.x;
    extern __shared__ float scores[];
    const float* l = logits + row * experts;
    for (size_t e = threadIdx.x; e < experts; e += blockDim.x) {
        const float x = l[e];
        scores[e] = gating == 1 ? 1.0f / (1.0f + expf(-x)) : gating == 2 ? sqrtf(x > 20.0f ? x : log1pf(expf(x))) : x;
    }
    __syncthreads();
    if (threadIdx.x) return;
    if (gating == 0) {
        float top = -INFINITY, total = 0.0f;
        for (size_t e = 0; e < experts; ++e) top = fmaxf(top, scores[e]);
        for (size_t e = 0; e < experts; ++e) total += expf(scores[e] - top);
        for (size_t e = 0; e < experts; ++e) scores[e] = expf(scores[e] - top) / total;
    }
    float chosen[64]; int32_t picked[64]; float sum = 0.0f;
    for (size_t slot = 0; slot < k; ++slot) {
        int32_t best = -1; float value = -INFINITY;
        for (size_t e = 0; e < experts; ++e) {
            bool used = false;
            for (size_t s = 0; s < slot; ++s) used |= picked[s] == int32_t(e);
            const float candidate = scores[e] + (bias ? bias[e] : 0.0f);
            if (!used && (best < 0 || candidate > value)) { best = int32_t(e); value = candidate; }
        }
        picked[slot] = best; chosen[slot] = scores[best]; sum += scores[best];
    }
    for (size_t slot = 0; slot < k; ++slot) {
        ids[row * k + slot] = picked[slot];
        weights[row * k + slot] = (normalize && k > 1 ? chosen[slot] / (sum + 1e-20f) : chosen[slot]) * scale;
    }
}

// out = silu(min(gate, c)) * clamp(up, -c, c) * row_weight (c = 0: no clamp; weights optional).
__global__ void op_swiglu_kernel(const float* gate, const float* up, size_t rows, size_t width, float clamp,
                                 const float* row_weights, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    float g = gate[index], u = up[index];
    if (clamp > 0.0f) { g = fminf(g, clamp); u = fminf(fmaxf(u, -clamp), clamp); }
    out[index] = g / (1.0f + expf(-g)) * u * (row_weights ? row_weights[index / width] : 1.0f);
}

// out[i] = x[rows[i]] (rows of `width`)
__global__ void op_gather_kernel(const float* x, const int32_t* rows, size_t count, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * width) return;
    out[index] = x[size_t(rows[index / width]) * width + index % width];
}

// out[rows[i]] += x[i] (rows distinct within one call)
__global__ void op_scatter_add_kernel(const float* x, const int32_t* rows, size_t count, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * width) return;
    out[size_t(rows[index / width]) * width + index % width] += x[index];
}

// ---- Latent attention over a sliding window and selected compressed rows ----

// One block per (row, head). The keys, which are also the values, are the window rows
// max(0, p - window + 1) .. p of `raw` (positions[row] = p) and the compressed rows `selected`
// lists for the row (-1: none). Softmax with the head's sink in the denominator.
__global__ void op_sparse_attention_kernel(const float* q, const float* raw, const int32_t* positions, size_t window,
                                           const float* compressed, const int32_t* selected, size_t selections,
                                           const float* sinks, size_t heads, size_t dim, float scale, float* out) {
    const size_t row = blockIdx.x / heads, head = blockIdx.x % heads;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const float* query = q + (row * heads + head) * dim;
    extern __shared__ float shared[];
    float* qs = shared;               // dim
    float* scores = shared + dim;     // total
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) qs[i] = query[i];
    __syncthreads();
    const auto key = [&](size_t index) -> const float* {
        if (index < raw_count) return raw + size_t(first + int64_t(index)) * dim;
        const int32_t at = selected[row * selections + (index - raw_count)];
        return at < 0 ? nullptr : compressed + size_t(at) * dim;
    };
    for (size_t index = threadIdx.x; index < total; index += blockDim.x) {
        const float* k = key(index);
        float dot = 0.0f;
        if (k) for (size_t i = 0; i < dim; ++i) dot += qs[i] * k[i];
        scores[index] = k ? dot * scale : -INFINITY;
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
    float* o = out + (row * heads + head) * dim;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (size_t index = 0; index < total; ++index) {
            if (scores[index] == 0.0f) continue;
            acc += scores[index] * key(index)[i];
        }
        o[i] = acc / denominator;
    }
}

// Pool `ratio` consecutive rows into one per group: softmax over the group of the gate, per
// channel (no gate: ratio 1, the row itself).
__global__ void op_compress_kernel(const float* kv, const float* gate, size_t groups, size_t ratio, size_t dim, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= groups * dim) return;
    const size_t group = index / dim, col = index % dim;
    if (!gate) { out[index] = kv[group * dim + col]; return; }
    float top = -INFINITY;
    for (size_t r = 0; r < ratio; ++r) top = fmaxf(top, gate[(group * ratio + r) * dim + col]);
    float sum = 0.0f, acc = 0.0f;
    for (size_t r = 0; r < ratio; ++r) {
        const float w = expf(gate[(group * ratio + r) * dim + col] - top);
        sum += w; acc += w * kv[(group * ratio + r) * dim + col];
    }
    out[index] = acc / sum;
}

// scores[row][g] = sum_h relu(q[row][h] . keys[g]) * weights[row][h] * scale for the rows'
// visible groups g < visible[row]; -inf beyond.
__global__ void op_index_scores_kernel(const float* q, const float* keys, const float* weights, const int32_t* visible,
                                       size_t groups, size_t heads, size_t dim, float scale, float* scores) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y;
    if (index >= groups) return;
    float* out = scores + row * groups + index;
    if (int32_t(index) >= visible[row]) { *out = -INFINITY; return; }
    const float* k = keys + index * dim;
    float total = 0.0f;
    for (size_t h = 0; h < heads; ++h) {
        const float* qh = q + (row * heads + h) * dim;
        float dot = 0.0f;
        for (size_t i = 0; i < dim; ++i) dot += qh[i] * k[i];
        total += fmaxf(dot, 0.0f) * weights[row * heads + h];
    }
    *out = total * scale;
}

// One block per row: the k highest scores' indices in ascending index order, -1 for empty slots
// (fewer finite scores than k). Ties keep the lower index. Rows of up to `groups` <= 8192.
__global__ void op_top_k_kernel(const float* scores, size_t groups, size_t k, int32_t* out) {
    const size_t row = blockIdx.x;
    const float* s = scores + row * groups;
    extern __shared__ int32_t keep[];   // groups flags
    for (size_t g = threadIdx.x; g < groups; g += blockDim.x) {
        // rank: how many beat this score (higher, or equal at a lower index)
        const float value = s[g];
        size_t rank = 0;
        if (value == -INFINITY) { keep[g] = 0; continue; }
        for (size_t other = 0; other < groups && rank < k; ++other)
            rank += s[other] > value || (s[other] == value && other < g);
        keep[g] = rank < k;
    }
    __syncthreads();
    if (threadIdx.x) return;
    size_t slot = 0;
    for (size_t g = 0; g < groups && slot < k; ++g) if (keep[g]) out[row * k + slot++] = int32_t(g);
    for (; slot < k; ++slot) out[row * k + slot] = -1;
}
