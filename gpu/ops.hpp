// Generic operations over device buffers, queued on the device's stream (docs/DESIGN.md §10):
// shapes are arguments, so any architecture composes them. Included in the GPU TU.

// The allocation holding [pointer, pointer + bytes) of this device, or an error: an op never
// touches memory the device did not allocate. The address index answers in O(log n) (a model
// holds thousands of allocations); an entry counts only while the allocation it names is live
// with the same size, and an index that finds none is rebuilt once.
void op_span(Device& d, const void* pointer, size_t bytes, const char* what) {
    const auto* at = static_cast<const uint8_t*>(pointer);
    std::lock_guard<std::mutex> lock(d.spans_mutex);
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto next = d.spans.upper_bound(at);
        if (next != d.spans.begin()) {
            --next;
            const uint8_t* base = next->first;
            const auto live = d.allocations.find(const_cast<uint8_t*>(base));
            if (live != d.allocations.end() && live->second == next->second && bytes <= next->second &&
                size_t(at - base) <= next->second - bytes)
                return;
        }
        if (attempt) break;
        d.spans.clear();
        for (const auto& entry : d.allocations) d.spans.emplace(static_cast<const uint8_t*>(entry.first), entry.second);
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

// One block per row: rms_norm (op_rms_norm_kernel's arithmetic), then the last 2 * pairs values
// rotated as op_rope_kernel rotates them, stored at row rows[row] of out (row itself without rows).
__global__ void op_norm_rope_kernel(const float* input, const float* weight, size_t width, float eps, size_t pairs,
                                    const int32_t* positions, const float* frequencies, const int32_t* rows, float* output) {
    const float* row = input + size_t(blockIdx.x) * width;
    float* out = output + size_t(rows ? rows[blockIdx.x] : blockIdx.x) * width;
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
    const size_t plain = width - 2 * pairs;
    for (size_t col = threadIdx.x; col < plain; col += blockDim.x) out[col] = row[col] * scale * weight[col];
    for (size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const size_t col = plain + 2 * pair;
        const double angle = double(positions[blockIdx.x]) * double(frequencies[pair]);
        double s, c;
        sincos(angle, &s, &c);
        const double a = row[col] * scale * weight[col], b = row[col + 1] * scale * weight[col + 1];
        out[col] = float(a * c - b * s);
        out[col + 1] = float(a * s + b * c);
    }
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

// A sublayer's mixing coefficients from its mixes m (W_hc . stream) and the stream's inverse RMS:
// pre and post by sigmoids, the combination doubly stochastic by Sinkhorn (softmax rows, then
// alternate normalizations). One thread.
__device__ void hc_coefficients_of(const float* m, float rsqrt, size_t hc, const float* base, const float* scale,
                                   size_t iterations, float eps, float* pre, float* post, float* comb) {
    const auto sigmoid = [](float x) { return 1.0f / (1.0f + expf(-x)); };
    for (size_t j = 0; j < hc; ++j) {
        pre[j] = sigmoid(m[j] * rsqrt * scale[0] + base[j]) + eps;
        post[j] = 2.0f * sigmoid(m[hc + j] * rsqrt * scale[1] + base[hc + j]);
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
    for (size_t i = 0; i < hc * hc; ++i) comb[i] = c[i];
}

// One block per row: the coefficients from mixes = W_hc . stream (the projection is a linear op),
// scaled by the stream's RMS.
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
    hc_coefficients_of(mixes + row * mix, rsqrtf(sums[0] / float(flat) + norm_eps), hc, base, scale, iterations, eps,
                       pre + row * hc, post + row * hc, comb + row * hc * hc);
}

// A sublayer boundary in one launch, a block per 64 columns of a row (blockIdx.y) and the row's
// last block finishing it: the previous sublayer's output x, when given, expanded into the stream
// (stream[j] = post[j] x + sum_i comb[i][j] residual[i]; else the stream is read); this
// sublayer's coefficients, when fn is given (the mixes W_hc . stream summed by block, then
// hc_coefficients_of); and its input norm(sum_j collapse[j] stream[j]) * weight, which the
// blocks write unnormalized. `counters` (one per row, zero between launches: the last block resets
// its row's) and `partial` (per row and block the mixes and both sums of squares) are a workspace.
constexpr size_t hc_step_columns = 64;
template<int HC>
__global__ void op_hc_step_kernel(const float* x, const float* residual, const float* post_in, const float* comb_in,
                                  float* stream, const float* fn, const float* base, const float* scale, size_t iterations,
                                  float eps, float norm_eps, const float* collapse, const float* norm_weight, size_t width,
                                  unsigned* counters, float* partial, float* pre, float* post, float* comb, float* out) {
    constexpr int MIX = (2 + HC) * HC, SUMS = MIX + 2;
    const size_t row = blockIdx.y, block = blockIdx.x, blocks = gridDim.x;
    const size_t first = block * hc_step_columns, columns = std::min(hc_step_columns, width - first);
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    __shared__ float values[HC][hc_step_columns];
    __shared__ float reduce[8][SUMS];
    __shared__ float totals[SUMS];
    __shared__ bool last;
    float acc[SUMS] = {};
    float* s_row = stream + row * HC * width;
    for (size_t pair = threadIdx.x; pair < HC * hc_step_columns; pair += blockDim.x) {
        const size_t j = pair / hc_step_columns, c = pair % hc_step_columns, col = first + c;
        if (c >= columns) continue;
        float v;
        if (x) {
            v = post_in[row * HC + j] * x[row * width + col];
            #pragma unroll
            for (int i = 0; i < HC; ++i) v += comb_in[(row * HC + i) * HC + j] * residual[(row * HC + i) * width + col];
            s_row[j * width + col] = v;
        } else v = s_row[j * width + col];
        values[j][c] = v;
        if (fn) {
            #pragma unroll
            for (int m = 0; m < MIX; ++m) acc[m] += fn[(m * HC + j) * width + col] * v;
        }
        acc[MIX] += v * v;
    }
    __syncthreads();
    if (threadIdx.x < columns) {
        float v = 0.0f;
        #pragma unroll
        for (int j = 0; j < HC; ++j) v += collapse[row * HC + j] * values[j][threadIdx.x];
        out[row * width + first + threadIdx.x] = v;
        acc[MIX + 1] = v * v;
    }
    #pragma unroll
    for (int m = 0; m < SUMS; ++m) {
        float v = acc[m];
        for (unsigned offset = 16; offset; offset /= 2) v += __shfl_xor(v, offset, 32);
        if (lane == 0) reduce[wave][m] = v;
    }
    __syncthreads();
    float* row_partial = partial + row * blocks * SUMS;
    if (threadIdx.x < SUMS) {
        float v = 0.0f;
        for (unsigned w = 0; w < blockDim.x / 32; ++w) v += reduce[w][threadIdx.x];
        row_partial[block * SUMS + threadIdx.x] = v;
    }
    // The row's last block to finish sees every block's sums and columns.
    __threadfence();
    unsigned* counter = counters + row;
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == blocks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < SUMS) {
        const volatile float* sums = row_partial + threadIdx.x;
        float v = 0.0f;
        for (size_t b = 0; b < blocks; ++b) v += sums[b * SUMS];
        totals[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *counter = 0;
        if (fn)
            hc_coefficients_of(totals, rsqrtf(totals[MIX] / float(HC * width) + norm_eps), HC, base, scale, iterations, eps,
                               pre + row * HC, post + row * HC, comb + row * HC * HC);
    }
    const float factor = rsqrtf(totals[MIX + 1] / float(width) + norm_eps);
    volatile float* o = out + row * width;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) o[col] = o[col] * factor * norm_weight[col];
}
// Values of op_hc_step_kernel's workspace per row: its counter, then its blocks' sums.
size_t hc_step_row_values(size_t hc, size_t width) {
    return 1 + (width + hc_step_columns - 1) / hc_step_columns * ((2 + hc) * hc + 2);
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
// unbiased scores normalized (when asked) and scaled. Each pick is a block-wide argmax.
__device__ __forceinline__ void route_take(float& value, int32_t& index, float other_value, int32_t other) {
    if (other >= 0 && (index < 0 || other_value > value || (other_value == value && other < index))) { value = other_value; index = other; }
}
__global__ void op_route_kernel(const float* logits, const float* bias, size_t experts, size_t k, int gating,
                                int normalize, float scale, int32_t* ids, float* weights) {
    const size_t row = blockIdx.x;
    extern __shared__ float scores[];
    __shared__ float reduce[256];
    __shared__ int32_t picked[64], best[8];
    __shared__ float best_value[8];
    const float* l = logits + row * experts;
    for (size_t e = threadIdx.x; e < experts; e += blockDim.x) {
        const float x = l[e];
        scores[e] = gating == 1 ? 1.0f / (1.0f + expf(-x)) : gating == 2 ? sqrtf(x > 20.0f ? x : log1pf(expf(x))) : x;
    }
    __syncthreads();
    if (gating == 0) {
        float top = -INFINITY;
        for (size_t e = threadIdx.x; e < experts; e += blockDim.x) top = fmaxf(top, scores[e]);
        reduce[threadIdx.x] = top;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
            __syncthreads();
        }
        top = reduce[0];
        __syncthreads();
        float total = 0.0f;
        for (size_t e = threadIdx.x; e < experts; e += blockDim.x) total += expf(scores[e] - top);
        reduce[threadIdx.x] = total;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        total = reduce[0];
        for (size_t e = threadIdx.x; e < experts; e += blockDim.x) scores[e] = expf(scores[e] - top) / total;
        __syncthreads();
    }
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    for (size_t slot = 0; slot < k; ++slot) {
        float value = -INFINITY;
        int32_t index = -1;
        for (size_t e = threadIdx.x; e < experts; e += blockDim.x) {
            bool used = false;
            for (size_t s = 0; s < slot; ++s) used |= picked[s] == int32_t(e);
            if (!used) route_take(value, index, scores[e] + (bias ? bias[e] : 0.0f), int32_t(e));
        }
        for (unsigned offset = 16; offset; offset /= 2) {
            const float other_value = __shfl_xor(value, offset, 32);
            const int32_t other = __shfl_xor(index, offset, 32);
            route_take(value, index, other_value, other);
        }
        if (lane == 0) { best[wave] = index; best_value[wave] = value; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned w = 1; w < blockDim.x / 32; ++w) route_take(value, index, best_value[w], best[w]);
            picked[slot] = index;
        }
        __syncthreads();
    }
    if (threadIdx.x) return;
    float chosen[64], sum = 0.0f;
    for (size_t slot = 0; slot < k; ++slot) { chosen[slot] = scores[picked[slot]]; sum += chosen[slot]; }
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

// out[rows[i]] = x[i]
__global__ void op_scatter_kernel(const float* x, const int32_t* rows, size_t count, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * width) return;
    out[size_t(rows[index / width]) * width + index % width] = x[index];
}

// out[rows[i]] += x[i] (rows distinct within one call)
__global__ void op_scatter_add_kernel(const float* x, const int32_t* rows, size_t count, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * width) return;
    out[size_t(rows[index / width]) * width + index % width] += x[index];
}

// ---- Latent attention over a sliding window and selected compressed rows ----

// One block per (row, head). The keys, which are also the values, are the window rows
// max(0, p - window + 1) .. p of `raw` (positions[row] = p; position q is row q % raw_rows, a
// ring) and the compressed rows `selected` lists for the row (-1: none). Softmax with the
// head's sink in the denominator.
__global__ void op_sparse_attention_kernel(const float* q, const float* raw, size_t raw_rows, const int32_t* positions, size_t window,
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
        if (index < raw_count) return raw + size_t(first + int64_t(index)) % raw_rows * dim;
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

// The same attention for every shape the tiles cover (heads a multiple of 16, dim of 64), in two
// launches over a scratch of scores: each key is read once per block of 32 keys for all the heads.
// Scores: a block per 32 keys of a row (blockIdx.y) and 64 heads (blockIdx.z) stores
// scale * q . key (-inf for an empty selection) and per head the block's max and sum of
// exp(score - max). Output: a block per 64 dims, 16 heads and row combines those into the head's
// max and denominator (with its sink) and sums the weighted keys.
constexpr int attention_keys = 32;
// Key `index` of a row: window rows first, then the selected compressed rows (null: none).
__device__ __forceinline__ const float* attention_key(size_t index, size_t raw_count, int64_t first, const float* raw,
                                                      size_t raw_rows, const float* compressed, const int32_t* selected,
                                                      size_t total, size_t dim) {
    if (index < raw_count) return raw + size_t(first + int64_t(index)) % raw_rows * dim;
    if (index >= total) return nullptr;
    const int32_t at = selected[index - raw_count];
    return at < 0 ? nullptr : compressed + size_t(at) * dim;
}
__global__ void op_attention_scores_kernel(const float* q, const float* raw, size_t raw_rows, const int32_t* positions,
                                           size_t window, const float* compressed, const int32_t* selected, size_t selections,
                                           size_t heads, size_t dim, float scale, size_t pairs, const float* frequencies,
                                           size_t stride, float* scores, float2* partial) {
    const size_t row = blockIdx.y, head0 = size_t(blockIdx.z) * 64, chunks = gridDim.x;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    __shared__ float qs[64][33], ks[attention_keys][33];
    __shared__ const float* keys[attention_keys];
    __shared__ double turns[64][2];   // the row's rotation per rope pair (op_rope_kernel's angles)
    for (size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        double s, c;
        sincos(double(p) * double(frequencies[pair]), &s, &c);
        turns[pair][0] = c; turns[pair][1] = s;
    }
    const size_t rotated = dim - 2 * pairs;
    const size_t index = size_t(blockIdx.x) * attention_keys + lane;
    if (threadIdx.x < attention_keys)
        keys[threadIdx.x] = attention_key(size_t(blockIdx.x) * attention_keys + threadIdx.x, raw_count, first, raw, raw_rows,
                                          compressed, selected + row * selections, total, dim);
    float acc[8] = {};
    const float* row_q = q + row * heads * dim;
    for (size_t d0 = 0; d0 < dim; d0 += 32) {
        __syncthreads();
        for (unsigned i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
            const size_t h = head0 + i / 32, col = d0 + i % 32;
            float v = 0.0f;
            if (h < heads && col < rotated) v = row_q[h * dim + col];
            else if (h < heads) {
                // q rotated as it is staged, as op_rope_kernel rotates it
                const size_t pair = (col - rotated) / 2;
                const double a = row_q[h * dim + rotated + 2 * pair], b = row_q[h * dim + rotated + 2 * pair + 1];
                const double c = turns[pair][0], s = turns[pair][1];
                v = col % 2 == rotated % 2 ? float(a * c - b * s) : float(a * s + b * c);
            }
            qs[i / 32][i % 32] = v;
        }
        for (unsigned i = threadIdx.x; i < attention_keys * 32; i += blockDim.x) {
            const float* key = keys[i / 32];
            ks[i / 32][i % 32] = key ? key[d0 + i % 32] : 0.0f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int c = 0; c < 32; ++c) {
            const float k = ks[lane][c];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = fmaf(qs[wave * 8 + i][c], k, acc[i]);
        }
    }
    const bool valid = keys[lane] != nullptr;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const size_t head = head0 + wave * 8 + i;
        const float score = valid ? acc[i] * scale : -INFINITY;
        if (head < heads && index < total) scores[(row * heads + head) * stride + index] = score;
        float top = score;
        for (unsigned offset = 16; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 32));
        float sum = valid ? expf(score - top) : 0.0f;
        for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_xor(sum, offset, 32);
        if (lane == 0 && head < heads) partial[(row * heads + head) * chunks + blockIdx.x] = make_float2(top, sum);
    }
}
__global__ void op_attention_output_kernel(const float* scores, const float2* partial, size_t chunks, const float* raw,
                                           size_t raw_rows, const int32_t* positions, size_t window, const float* compressed,
                                           const int32_t* selected, size_t selections, const float* sinks, size_t heads,
                                           size_t dim, size_t pairs, const float* frequencies, size_t stride, float* out) {
    const size_t row = blockIdx.z, head0 = size_t(blockIdx.y) * 16, d0 = size_t(blockIdx.x) * 64;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const unsigned local = threadIdx.x / 16, quad = threadIdx.x % 16;
    __shared__ float tops[16], denominators[16];
    __shared__ float4 vs[attention_keys][16];
    __shared__ float ps[16][attention_keys + 1];
    __shared__ const float* keys[attention_keys];
    if (threadIdx.x < 16) {
        const size_t head = std::min(head0 + threadIdx.x, heads - 1);
        const float2* parts = partial + (row * heads + head) * chunks;
        float top = sinks[head];
        for (size_t c = 0; c < chunks; ++c) top = fmaxf(top, parts[c].x);
        float denominator = expf(sinks[head] - top);
        for (size_t c = 0; c < chunks; ++c) if (parts[c].y > 0.0f) denominator += parts[c].y * expf(parts[c].x - top);
        tops[threadIdx.x] = top;
        denominators[threadIdx.x] = denominator;
    }
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (size_t k0 = 0; k0 < total; k0 += attention_keys) {
        __syncthreads();
        if (threadIdx.x < attention_keys)
            keys[threadIdx.x] = attention_key(k0 + threadIdx.x, raw_count, first, raw, raw_rows, compressed,
                                              selected + row * selections, total, dim);
        for (unsigned i = threadIdx.x; i < 16 * attention_keys; i += blockDim.x) {
            const size_t h = head0 + i / attention_keys, key = k0 + i % attention_keys;
            const float score = h < heads && key < total ? scores[(row * heads + h) * stride + key] : -INFINITY;
            ps[i / attention_keys][i % attention_keys] = score == -INFINITY ? 0.0f : expf(score - tops[i / attention_keys]);
        }
        __syncthreads();
        for (unsigned i = threadIdx.x; i < attention_keys * 16; i += blockDim.x) {
            const float* key = keys[i / 16];
            vs[i / 16][i % 16] = key ? *reinterpret_cast<const float4*>(key + d0 + 4 * (i % 16)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < attention_keys; ++k) {
            const float weight = ps[local][k];
            const float4 v = vs[k][quad];
            acc.x = fmaf(weight, v.x, acc.x); acc.y = fmaf(weight, v.y, acc.y);
            acc.z = fmaf(weight, v.z, acc.z); acc.w = fmaf(weight, v.w, acc.w);
        }
    }
    const size_t head = head0 + local;
    if (head >= heads) return;
    const float denominator = denominators[local];
    float o[4] = {acc.x / denominator, acc.y / denominator, acc.z / denominator, acc.w / denominator};
    const size_t col = d0 + 4 * quad, rotated = dim - 2 * pairs;
    // the inverse rotation, as op_rope_kernel's inverse
    for (int i = 0; i < 4; i += 2)
        if (col + i >= rotated) {
            double s, c;
            sincos(double(p) * double(frequencies[(col + i - rotated) / 2]) * -1.0, &s, &c);
            const double a = o[i], b = o[i + 1];
            o[i] = float(a * c - b * s);
            o[i + 1] = float(a * s + b * c);
        }
    *reinterpret_cast<float4*>(out + (row * heads + head) * dim + col) = make_float4(o[0], o[1], o[2], o[3]);
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
