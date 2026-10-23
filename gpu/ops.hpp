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

// The sum of a 256-thread block's partials p[0..256) in the order of the halving tree
// (p[i] += p[i + stride] for stride 128, 64, ..., 1), formed by the first wave alone: lane i
// takes ((p[i] + p[i+128]) + (p[i+64] + p[i+192])) + ((p[i+32] + p[i+160]) + (p[i+96] + p[i+224]))
// -- the tree's value at i after stride 32 -- and the wave halves the rest. Every lane of the
// first wave calls it; lane 0 has the sum (the tree's bits, with two barriers fewer than eight).
__device__ __forceinline__ float tree256_first_wave(const float* p, unsigned lane) {
    float value = ((p[lane] + p[lane + 128]) + (p[lane + 64] + p[lane + 192])) +
                  ((p[lane + 32] + p[lane + 160]) + (p[lane + 96] + p[lane + 224]));
    for (unsigned offset = 16; offset; offset /= 2) value += __shfl_down(value, offset, 32);
    return value;
}

// One block (256 threads) per row: x / sqrt(mean(x^2) + eps) * weight.
__global__ void op_rms_norm_kernel(const float* input, const float* weight, size_t width, float eps, float* output) {
    const float* row = input + size_t(blockIdx.x) * width;
    float* out = output + size_t(blockIdx.x) * width;
    __shared__ float sums[256];
    __shared__ float total;
    float sum = 0.0f;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) sum += row[col] * row[col];
    sums[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        const float value = tree256_first_wave(sums, threadIdx.x);
        if (!threadIdx.x) total = value;
    }
    __syncthreads();
    const float scale = rsqrtf(total / float(width) + eps);
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) out[col] = row[col] * scale * weight[col];
}

// A block's row `index`: rms_norm (op_rms_norm_kernel's arithmetic), then the last 2 * pairs values
// rotated as op_rope_kernel rotates them, stored at row rows[index] of out (index itself without
// rows); `sums` is 256 floats of the block's shared memory. A block of 256 threads.
__device__ inline void op_norm_rope_row(const float* input, const float* weight, size_t width, float eps, size_t pairs,
                                        const int32_t* positions, const float* frequencies, const int32_t* rows, float* output,
                                        size_t index, float* sums, double2* forward, double2* inverse) {
    const float* row = input + index * width;
    float* out = output + size_t(rows ? rows[index] : index) * width;
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
        const double angle = double(positions[index]) * double(frequencies[pair]);
        double s, c;
        sincos(angle, &s, &c);
        const double a = row[col] * scale * weight[col], b = row[col + 1] * scale * weight[col + 1];
        out[col] = float(a * c - b * s);
        out[col + 1] = float(a * s + b * c);
        // The attention table, when this row's position is the query's. Inverse is a second sincos.
        if (forward) {
            forward[index * pairs + pair] = make_double2(c, s);
            sincos(angle * -1.0, &s, &c);
            inverse[index * pairs + pair] = make_double2(c, s);
        }
    }
}

// One block per row: op_norm_rope_row.
__global__ void op_norm_rope_kernel(const float* input, const float* weight, size_t width, float eps, size_t pairs,
                                    const int32_t* positions, const float* frequencies, const int32_t* rows, float* output,
                                    double2* forward, double2* inverse) {
    __shared__ float sums[256];
    op_norm_rope_row(input, weight, width, eps, pairs, positions, frequencies, rows, output, blockIdx.x, sums, forward, inverse);
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

// steps rows of cols values (stride apart) rounded to FP16, packed.
__global__ void op_half_rows(const float* input, size_t stride, size_t cols, size_t steps, __half* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < steps * cols) out[index] = __float2half(input[(index / cols) * stride + index % cols]);
}

// out[t][r] = W[first + r] . in[t] for `rows` rows, inputs and outputs `input_stride` and
// `output_stride` values apart: GEMV for up to gemv_max_vectors inputs, else the rows expanded
// to F32 in `scratch` (in pieces that fit) and an F32 GEMM -- or, on a device set to FP16 GEMMs
// (sd_gemm_options), the rows expanded to FP16 and the inputs rounded to FP16 (at the scratch's
// end), an FP16 GEMM with F32 sums. The GEMV gives a row at least op_lanes lanes, so that a wave
// reads long runs of each row (short rows otherwise get two).
constexpr unsigned op_lanes = 8;
void op_linear(Device& d, const uint8_t* weights, int kind, size_t rows, size_t cols, size_t first,
               const float* input, size_t input_stride, size_t steps, float* output, size_t output_stride,
               float* scratch, size_t scratch_bytes) {
    const size_t row_bytes = matrix_row_bytes(kind, cols);
    const Matrix matrix{weights + first * row_bytes, row_bytes, rows, cols, 0, 0, 0, kind};
    if (steps <= size_t(gemv_max_vectors)) {
        if (steps == 1) gemv(d, &matrix, 1, input, output, 1, 0, 0, {}, op_lanes);
        else gemv(d, &matrix, 1, input, output, steps, input_stride, output_stride, {}, op_lanes);   // strides in values
        return;
    }
    if (!d.blas) throw std::runtime_error("Straydog op: GEMM needs a BLAS handle");
    if (d.half_gemm) {
        const size_t input_bytes = (steps * cols * 2 + 255) & ~size_t(255);
        if (scratch_bytes <= input_bytes) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
        auto* half_input = reinterpret_cast<__half*>(reinterpret_cast<uint8_t*>(scratch) + scratch_bytes - input_bytes);
        op_half_rows<<<unsigned((steps * cols + 255) / 256), 256, 0, d.stream>>>(input, input_stride, cols, steps, half_input);
        const size_t piece = std::min(rows, (scratch_bytes - input_bytes) / (cols * 2));
        if (!piece) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
        const float alpha = 1, beta = 0;
        for (size_t done = 0; done < rows; done += piece) {
            const size_t count = std::min(piece, rows - done);
            const Matrix part{matrix.data + done * row_bytes, row_bytes, count, cols, 0, 0, 0, kind};
            expand(part, scratch, d.stream, true);
            blas_check(hipblasGemmEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(count), int(steps), int(cols), &alpha, scratch, HIP_R_16F,
                                     int(cols), half_input, HIP_R_16F, int(cols), &beta, output + done, HIP_R_32F, int(output_stride),
                                     HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        }
        return;
    }
    const size_t piece = std::min(rows, scratch_bytes / (cols * 4));
    if (!piece) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
    const float alpha = 1, beta = 0;
    for (size_t done = 0; done < rows; done += piece) {
        const size_t count = std::min(piece, rows - done);
        const Matrix part{matrix.data + done * row_bytes, row_bytes, count, cols, 0, 0, 0, kind};
        expand(part, scratch, d.stream, false);
        blas_check(hipblasSgemm(d.blas_f32, HIPBLAS_OP_T, HIPBLAS_OP_N, int(count), int(steps), int(cols), &alpha,
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

// hc_coefficients_of with a lane per entry of the combination (hc * hc <= 32), called by a whole
// wave: every sum is formed in hc_coefficients_of's order, so the results are the same.
template<int HC>
__device__ void hc_coefficients_wave(const float* m, float rsqrt, const float* base, const float* scale, size_t iterations,
                                     float eps, float* pre, float* post, float* comb) {
    const unsigned lane = threadIdx.x % 32, j = lane / HC, k = lane % HC;
    const auto sigmoid = [](float x) { return 1.0f / (1.0f + expf(-x)); };
    if (lane < HC) {
        pre[lane] = sigmoid(m[lane] * rsqrt * scale[0] + base[lane]) + eps;
        post[lane] = 2.0f * sigmoid(m[HC + lane] * rsqrt * scale[1] + base[HC + lane]);
    }
    const bool live = lane < HC * HC;
    float c = live ? m[2 * HC + lane] * rsqrt * scale[2] + base[2 * HC + lane] : 0.0f;
    float top = -INFINITY;
    #pragma unroll
    for (int i = 0; i < HC; ++i) top = fmaxf(top, __shfl(c, int(j * HC + i), 32));
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < HC; ++i) total += expf(__shfl(c, int(j * HC + i), 32) - top);
    c = expf(c - top) / total + eps;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        if (iteration) {
            float row = 0.0f;
            #pragma unroll
            for (int i = 0; i < HC; ++i) row += __shfl(c, int(j * HC + i), 32);
            c /= row + eps;
        }
        float column = 0.0f;
        #pragma unroll
        for (int i = 0; i < HC; ++i) column += __shfl(c, int(i * HC + k), 32);
        c /= column + eps;
    }
    if (live) comb[lane] = c;
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
constexpr unsigned hc_step_threads = 256;   // a block's threads (its last block's loads are counted by them)
// The last block reads what other blocks wrote from the device-wide cache, without ordering the loads.
__device__ __forceinline__ float hc_step_load(float* at) {
#if defined(__HIPCC__)
    return __hip_atomic_load(at, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return *static_cast<volatile float*>(at);
#endif
}
template<int HC>
__global__ void __launch_bounds__(hc_step_threads) op_hc_step_kernel(const float* x, const float* residual, const float* post_in, const float* comb_in,
                                  float* stream, const float* fn, const float* base, const float* scale, size_t iterations,
                                  float eps, float norm_eps, const float* collapse, const float* norm_weight, size_t width,
                                  unsigned* counters, float* partial, float* pre, float* post, float* comb, float* out,
                                  PartRead link) {
    constexpr int MIX = (2 + HC) * HC, SUMS = MIX + 2;
    const size_t row = blockIdx.y, block = blockIdx.x, blocks = gridDim.x;
    // A prvalue: std::min binds a reference, and CUDA device code cannot ODR-use the namespace constant.
    const size_t first = block * hc_step_columns, columns = std::min(size_t(hc_step_columns), width - first);
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    __shared__ float values[HC][hc_step_columns];
    __shared__ float reduce[8][SUMS];
    constexpr size_t GATHER = (3072 / SUMS) * SUMS;   // whole blocks' sums: one round for DeepSeek's 80 blocks
    __shared__ float totals[SUMS], gathered[GATHER];
    __shared__ bool last;
    float acc[SUMS] = {};
    float* s_row = stream + row * HC * width;
    // Linked (link.state): x is this device's part of the sublayer's output and the other's comes
    // into the inbox (link_tail.hpp); the block's columns of their sum, formed once for every stream.
    __shared__ float summed[hc_step_columns];
    if (link.state) {
        __shared__ uint32_t n;
        if (!threadIdx.x) n = part_wait(link);
        __syncthreads();
        const float* theirs = link.inbox + size_t(n & 1) * link.capacity;
        for (size_t c = threadIdx.x; c < columns; c += blockDim.x) {
            const size_t at = row * width + first + c;
            summed[c] = x[at] + link_load(theirs + at);
        }
        __syncthreads();
        if (!threadIdx.x) part_read_done(link, n);
    }
    for (size_t pair = threadIdx.x; pair < HC * hc_step_columns; pair += blockDim.x) {
        const size_t j = pair / hc_step_columns, c = pair % hc_step_columns, col = first + c;
        if (c >= columns) continue;
        float v;
        if (x) {
            const float xv = link.state ? summed[c] : x[row * width + col];
            v = post_in[row * HC + j] * xv;
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
    // The row's last block to finish sees every block's sums and columns: every wave's writes are
    // out (its fence) before the block counts itself done (else the last block may read a column
    // another wave of this block has not written yet).
    __threadfence();
    __syncthreads();
    unsigned* counter = counters + row;
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == blocks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    // The blocks' sums arrive GATHER at a time, all loads at once (a thread's issued before it
    // stores any); each total adds them in order.
    float total = 0.0f;
    constexpr unsigned LOADS = (GATHER + hc_step_threads - 1) / hc_step_threads;
    for (size_t b0 = 0; b0 < blocks; b0 += GATHER / SUMS) {
        const size_t count = std::min<size_t>(GATHER / SUMS, blocks - b0) * SUMS;
        float* sums = row_partial + b0 * SUMS;
        float loaded[LOADS];
        #pragma unroll
        for (unsigned j = 0; j < LOADS; ++j) {
            const size_t i = threadIdx.x + j * hc_step_threads;
            if (i < count) loaded[j] = hc_step_load(sums + i);
        }
        #pragma unroll
        for (unsigned j = 0; j < LOADS; ++j) {
            const size_t i = threadIdx.x + j * hc_step_threads;
            if (i < count) gathered[i] = loaded[j];
        }
        __syncthreads();
        if (threadIdx.x < SUMS)
            for (size_t b = 0; b < count / SUMS; ++b) total += gathered[b * SUMS + threadIdx.x];
        __syncthreads();
    }
    if (threadIdx.x < SUMS) totals[threadIdx.x] = total;
    __syncthreads();
    if (threadIdx.x == 0) *counter = 0;
    if (fn) {
        const float rsqrt = rsqrtf(totals[MIX] / float(HC * width) + norm_eps);
        if constexpr (HC * HC <= 32) {
            if (wave == 0)
                hc_coefficients_wave<HC>(totals, rsqrt, base, scale, iterations, eps, pre + row * HC, post + row * HC,
                                         comb + row * HC * HC);
        } else if (threadIdx.x == 0)
            hc_coefficients_of(totals, rsqrt, HC, base, scale, iterations, eps, pre + row * HC, post + row * HC, comb + row * HC * HC);
    }
    const float factor = rsqrtf(totals[MIX + 1] / float(width) + norm_eps);
    float* o = out + row * width;
    // the row's columns a few at a time, each group's loads issued before its stores
    constexpr unsigned EACH = 8;
    for (size_t col0 = 0; col0 < width; col0 += size_t(EACH) * blockDim.x) {
        float loaded[EACH], weights[EACH];
        #pragma unroll
        for (unsigned j = 0; j < EACH; ++j) {
            const size_t col = col0 + j * blockDim.x + threadIdx.x;
            if (col < width) { loaded[j] = hc_step_load(o + col); weights[j] = norm_weight[col]; }
        }
        #pragma unroll
        for (unsigned j = 0; j < EACH; ++j) {
            const size_t col = col0 + j * blockDim.x + threadIdx.x;
            if (col < width) o[col] = loaded[j] * factor * weights[j];
        }
    }
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
// op_route_kernel's softmax routing (gating 0, no bias, scale 1) of 256 experts in one wave, with
// its arithmetic: lane i holds experts i + 32m; the max (exact in any order), the sum of the
// exponentials in its tree's order (tree256_first_wave over the thread terms), the same
// probabilities, then each pick the highest (lowest index on ties) of those left, as a wave
// argmax; the normalized weights summed in slot order. The same bits, without barriers.
// A row's routes by one wave (lane = its lane): its ids and weights at ids[slot] and weights[slot].
__device__ __forceinline__ void op_route_softmax256_wave(const float* l, size_t k, int normalize, int32_t* ids, float* weights,
                                                         unsigned lane) {
    float p[8];
    float top = -INFINITY;
    #pragma unroll
    for (int m = 0; m < 8; ++m) { p[m] = l[lane + 32 * m]; top = fmaxf(top, p[m]); }
    for (unsigned offset = 16; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 32));
    #pragma unroll
    for (int m = 0; m < 8; ++m) p[m] = expf(p[m] - top);
    float total = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
    for (unsigned offset = 16; offset; offset /= 2) total += __shfl_down(total, offset, 32);
    total = __shfl(total, 0, 32);
    #pragma unroll
    for (int m = 0; m < 8; ++m) p[m] = p[m] / total;
    __shared__ float chosen[64];
    __shared__ int32_t chosen_ids[64];
    for (size_t slot = 0; slot < k; ++slot) {
        float value = -INFINITY;
        int32_t index = -1;
        #pragma unroll
        for (int m = 0; m < 8; ++m) route_take(value, index, p[m], int32_t(lane + 32 * m));
        for (unsigned offset = 16; offset; offset /= 2) {
            const float other_value = __shfl_xor(value, offset, 32);
            const int32_t other = __shfl_xor(index, offset, 32);
            route_take(value, index, other_value, other);
        }
        #pragma unroll
        for (int m = 0; m < 8; ++m) if (index == int32_t(lane + 32 * m)) p[m] = -INFINITY;
        if (!lane) { chosen[slot] = value; chosen_ids[slot] = index; }
    }
    if (lane) return;
    float sum = 0.0f;
    for (size_t slot = 0; slot < k; ++slot) sum += chosen[slot];
    for (size_t slot = 0; slot < k; ++slot) {
        ids[slot] = chosen_ids[slot];
        weights[slot] = (normalize && k > 1 ? chosen[slot] / (sum + 1e-20f) : chosen[slot]) * 1.0f;
    }
}
__global__ void op_route_softmax256_kernel(const float* logits, size_t k, int normalize, int32_t* ids, float* weights) {
    const size_t row = blockIdx.x;
    op_route_softmax256_wave(logits + row * 256, k, normalize, ids + row * k, weights + row * k, threadIdx.x);
}

// One row's routes by the block (256 threads, `experts` x 2 floats of dynamic LDS at `scores`): its
// logits read through the device-wide cache (another block of the launch may have written them).
// With `published`, the ids also go there (a route mailbox in host memory: sd_op_linear_route_published).
__device__ void route_row(const float* l, const float* bias, size_t experts, size_t k, int gating, int normalize, float scale,
                          int32_t* ids, float* weights, float* scores, int32_t* published = nullptr) {
    float* biased = scores + experts;
    __shared__ float reduce[256];
    __shared__ int32_t picked[64], best[8];
    __shared__ float best_value[8];
    for (size_t e = threadIdx.x; e < experts; e += blockDim.x) {
        const float x = hc_step_load(const_cast<float*>(l + e));
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
    for (size_t e = threadIdx.x; e < experts; e += blockDim.x) biased[e] = scores[e] + (bias ? bias[e] : 0.0f);
    __syncthreads();
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    for (size_t slot = 0; slot < k; ++slot) {
        float value = -INFINITY;
        int32_t index = -1;
        for (size_t e = threadIdx.x; e < experts; e += blockDim.x) {
            bool used = false;
            for (size_t s = 0; s < slot; ++s) used |= picked[s] == int32_t(e);
            if (!used) route_take(value, index, biased[e], int32_t(e));
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
    float sum = 0.0f;
    for (size_t slot = 0; slot < k; ++slot) sum += scores[picked[slot]];
    for (size_t slot = 0; slot < k; ++slot) {
        const float chosen = scores[picked[slot]];
        ids[slot] = picked[slot];
        weights[slot] = (normalize && k > 1 ? chosen / (sum + 1e-20f) : chosen) * scale;
        if (published) published[slot] = picked[slot];
    }
}
__global__ void op_route_kernel(const float* logits, const float* bias, size_t experts, size_t k, int gating,
                                int normalize, float scale, int32_t* ids, float* weights) {
    extern __shared__ float scores[];   // the scores, then the scores + bias
    const size_t row = blockIdx.x;
    route_row(logits + row * experts, bias, experts, k, gating, normalize, scale, ids + row * k, weights + row * k, scores);
}
// A router's logits (gemv_fixed_kind's rows, lanes and dots: the same bits) and, in the launch's
// last block to finish, every row's routes as op_route_kernel forms them: one launch for two.
struct RouteTail {
    const float* bias;
    size_t experts, k;
    int gating, normalize;
    float scale;
    int32_t* ids;
    float* weights;
    unsigned* counter;   // zero between launches: the last block resets it
    // A route mailbox (host memory): every row's ids there, then its flag raised (system scope).
    int32_t* published;
    unsigned* flag;
};
template<int kind, int V>
__global__ void gemv_routed_kind(FixedGemv list, const float* input, float* output, RouteTail tail) {
    extern __shared__ float scores[];
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const GemvItem& item = list.items[0];
    gemv_rows<kind, V>(item.matrix, wave * (32 / item.lanes), item.lanes, input, list.input_stride, output, list.output_stride,
                       list.residual);
    // Every block's logits are out before it counts itself done.
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(tail.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) *tail.counter = 0;
    const size_t stride = V > 1 ? list.output_stride : 0;
    for (int row = 0; row < V; ++row) {
        route_row(output + row * stride, tail.bias, tail.experts, tail.k, tail.gating, tail.normalize, tail.scale, tail.ids + row * tail.k,
                  tail.weights + row * tail.k, scores, tail.published ? tail.published + row * tail.k : nullptr);
        __syncthreads();
    }
    // (thread 0 wrote every row's published ids)
    if (tail.flag && threadIdx.x == 0) {
        __threadfence_system();
#if defined(__HIPCC__)
        __hip_atomic_store(tail.flag, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *static_cast<volatile unsigned*>(tail.flag) = 1u;
#endif
    }
}
// gemv() of one matrix (an F32, F16 or BF16 router) for 1-4 vectors with the rows' routes in its
// last block, on the kernel and lanes gemv() would take; false (nothing launched) otherwise.
bool gemv_route_launch(Device& d, const Matrix& matrix, const float* input, float* output, size_t vectors, size_t input_stride,
                       size_t output_stride, RouteTail tail, unsigned min_lanes) {
    if (matrix.kind != 0 && matrix.kind != 1 || vectors < 1 || vectors > 4) return false;
    unsigned lanes = gemv_lanes(matrix, input);
    size_t alignment = 0;
    if (lanes == 2 && min_lanes > 2 && matrix.cols / gemv_unit(matrix.kind, alignment) >= 32) lanes = std::min(32u, min_lanes);
    if (!lanes) return false;
    FixedGemv list{};
    list.items[0] = {matrix, 0, lanes};
    list.count = 1;
    list.input_stride = uint32_t(input_stride);
    list.output_stride = uint32_t(output_stride);
    const unsigned blocks = unsigned((gemv_waves(matrix, lanes) + 7) / 8);
    const size_t shared = tail.experts * 8;
    #define SD_ROUTED(format, v) gemv_routed_kind<format, v><<<blocks, 256, shared, d.stream>>>(list, input, output, tail)
    if (matrix.kind == 0) {
        switch (vectors) { case 1: SD_ROUTED(0, 1); break; case 2: SD_ROUTED(0, 2); break; case 3: SD_ROUTED(0, 3); break; default: SD_ROUTED(0, 4); }
    } else {
        switch (vectors) { case 1: SD_ROUTED(1, 1); break; case 2: SD_ROUTED(1, 2); break; case 3: SD_ROUTED(1, 3); break; default: SD_ROUTED(1, 4); }
    }
    #undef SD_ROUTED
    return true;
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

// The same attention over a scratch of scores, for dims a multiple of 64. Scores: a block per 32
// keys of a row (blockIdx.y) and 16 heads (blockIdx.z) stores scale * q . key (-inf for an empty
// selection) and per head the block's max and sum of exp(score - max); q and the keys come through
// LDS 64 dims at a time, the next step's loaded while this one's is summed. Output: a block per 64
// dims, 8 heads, row and share of the keys combines those into the head's max and denominator
// (with its sink) and sums its weighted keys, 32 at a time, double-buffered too; with one share it
// stores the output, with several each stores its sum and a last launch adds them in share order
// (op_attention_combine_kernel). (Larger steps took more registers than they saved in waits.)
constexpr int attention_keys = 32, attention_heads = 16, attention_out_heads = 8;
// Key `index` of a row: window rows first, then the selected compressed rows (null: none).
__device__ __forceinline__ const float* attention_key(size_t index, size_t raw_count, int64_t first, const float* raw,
                                                      size_t raw_rows, const float* compressed, const int32_t* selected,
                                                      size_t total, size_t dim) {
    if (index < raw_count) return raw + size_t(first + int64_t(index)) % raw_rows * dim;
    if (index >= total) return nullptr;
    const int32_t at = selected[index - raw_count];
    return at < 0 ? nullptr : compressed + size_t(at) * dim;
}
// One row's rope, once: forward, then the inverse from the same angle times -1 (not a negated sine).
__global__ void op_attention_turns_kernel(const int32_t* positions, const float* frequencies, size_t pairs, double2* forward,
                                          double2* inverse) {
    const int64_t p = positions[blockIdx.x];
    for (size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        double s, c;
        sincos(double(p) * double(frequencies[pair]), &s, &c);
        forward[size_t(blockIdx.x) * pairs + pair] = make_double2(c, s);
        sincos(double(p) * double(frequencies[pair]) * -1.0, &s, &c);
        inverse[size_t(blockIdx.x) * pairs + pair] = make_double2(c, s);
    }
}
__global__ void __launch_bounds__(256) op_attention_scores_kernel(
        const float* q, const float* raw, size_t raw_rows, const int32_t* positions, size_t window, const float* compressed,
        const int32_t* selected, size_t selections, size_t heads, size_t dim, float scale, size_t pairs, const double2* forward,
        size_t stride, float* scores, float2* partial) {
    const size_t row = blockIdx.y, head0 = size_t(blockIdx.z) * attention_heads, chunks = gridDim.x;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32;
    constexpr int S = 64;
    __shared__ float qs[2][attention_heads][S + 1], ks[2][attention_keys][S + 1];
    __shared__ const float* keys[attention_keys];
    __shared__ double turns[64][2];   // the row's rotation per rope pair, filled once for the token
    for (size_t pair = t; pair < pairs; pair += blockDim.x) {
        const double2 g = forward[row * pairs + pair];
        turns[pair][0] = g.x; turns[pair][1] = g.y;
    }
    if (t < attention_keys)
        keys[t] = attention_key(size_t(blockIdx.x) * attention_keys + t, raw_count, first, raw, raw_rows, compressed,
                                selected + row * selections, total, dim);
    __syncthreads();
    const size_t rotated = dim - 2 * pairs;
    const float* row_q = q + row * heads * dim;
    constexpr int QV = attention_heads * S / 256, KV = attention_keys * S / 256;
    float qv[QV], kv[KV];
    // One thread takes four consecutive columns. A 16-byte load, then four scalar stores into the
    // S+1 LDS. Rope stays the scalar double path, and so does a row that is not 16-byte aligned.
    auto load = [&](size_t d0) {
        const unsigned q_head = t / 16, q_col = (t % 16) * 4;
        const size_t h = head0 + q_head, col = d0 + q_col;
        const float* q_row = h < heads ? row_q + h * dim + col : nullptr;
        if (q_row && col + 3 < rotated && (reinterpret_cast<uintptr_t>(q_row) & 15) == 0) {
            const float4 g = *reinterpret_cast<const float4*>(q_row);
            qv[0] = g.x; qv[1] = g.y; qv[2] = g.z; qv[3] = g.w;
        } else {
            #pragma unroll
            for (int j = 0; j < QV; ++j) {
                const size_t c = col + j;
                float v = 0.0f;
                if (h < heads && c < rotated) v = row_q[h * dim + c];
                else if (h < heads) {
                    const size_t pair = (c - rotated) / 2;
                    const double a = row_q[h * dim + rotated + 2 * pair], b = row_q[h * dim + rotated + 2 * pair + 1];
                    const double cs = turns[pair][0], sn = turns[pair][1];
                    v = (c - rotated) % 2 == 0 ? float(a * cs - b * sn) : float(a * sn + b * cs);
                }
                qv[j] = v;
            }
        }
        #pragma unroll
        for (int i = 0; i < KV / 4; ++i) {
            const unsigned group = t + unsigned(i) * 256;
            const float* key = keys[group / 16];
            const float* addr = key ? key + d0 + (group % 16) * 4 : nullptr;
            if (addr && (reinterpret_cast<uintptr_t>(addr) & 15) == 0) {
                const float4 g = *reinterpret_cast<const float4*>(addr);
                kv[i * 4] = g.x; kv[i * 4 + 1] = g.y; kv[i * 4 + 2] = g.z; kv[i * 4 + 3] = g.w;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) kv[i * 4 + j] = addr ? addr[j] : 0.0f;
            }
        }
    };
    auto store = [&](int buffer) {
        const unsigned q_head = t / 16, q_col = (t % 16) * 4;
        #pragma unroll
        for (int j = 0; j < QV; ++j) qs[buffer][q_head][q_col + j] = qv[j];
        #pragma unroll
        for (int i = 0; i < KV / 4; ++i) {
            const unsigned group = t + unsigned(i) * 256;
            const unsigned key_index = group / 16, key_col = (group % 16) * 4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) ks[buffer][key_index][key_col + j] = kv[i * 4 + j];
        }
    };
    load(0);
    store(0);
    __syncthreads();
    float acc[2] = {};
    const size_t steps = dim / S;
    for (size_t step = 0; step < steps; ++step) {
        const int buffer = int(step & 1);
        if (step + 1 < steps) load((step + 1) * S);
        #pragma unroll 8
        for (int c = 0; c < S; ++c) {
            const float k = ks[buffer][lane][c];
            acc[0] = fmaf(qs[buffer][2 * wave][c], k, acc[0]);
            acc[1] = fmaf(qs[buffer][2 * wave + 1][c], k, acc[1]);
        }
        if (step + 1 < steps) store(buffer ^ 1);
        __syncthreads();
    }
    const size_t index = size_t(blockIdx.x) * attention_keys + lane;
    const bool valid = keys[lane] != nullptr;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const size_t head = head0 + 2 * wave + i;
        const float score = valid ? acc[i] * scale : -INFINITY;
        if (head < heads && index < total) scores[(row * heads + head) * stride + index] = score;
        float top = score;
        for (unsigned offset = 16; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 32));
        float sum = valid ? expf(score - top) : 0.0f;
        for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_xor(sum, offset, 32);
        if (lane == 0 && head < heads) partial[(row * heads + head) * chunks + blockIdx.x] = make_float2(top, sum);
    }
}
// The inverse rotation of an output's values col .. col + 3, from the token's inverse table.
__device__ __forceinline__ void attention_unrotate(float (&o)[4], size_t row, size_t col, size_t rotated, size_t pairs,
                                                   const double2* inverse) {
    for (int i = 0; i < 4; i += 2)
        if (col + i >= rotated) {
            const double2 g = inverse[row * pairs + (col + i - rotated) / 2];
            const double c = g.x, s = g.y;
            const double a = o[i], b = o[i + 1];
            o[i] = float(a * c - b * s);
            o[i + 1] = float(a * s + b * c);
        }
}
// Dynamic LDS: the share's key addresses. blockIdx.z = row * shares + share; a share takes
// `share_keys` keys. With shares, `sums` takes each share's sum and `head_denominators` each
// head's denominator.
__global__ void __launch_bounds__(128) op_attention_output_kernel(
        const float* scores, const float2* partial, size_t chunks, const float* raw, size_t raw_rows, const int32_t* positions,
        size_t window, const float* compressed, const int32_t* selected, size_t selections, const float* sinks, size_t heads,
        size_t dim, size_t pairs, const double2* inverse, size_t stride, size_t shares, size_t share_keys, float* sums,
        float* head_denominators, float* out, unsigned* tiles) {
    const size_t row = blockIdx.z / shares, share = blockIdx.z % shares;
    const size_t head0 = size_t(blockIdx.y) * attention_out_heads, d0 = size_t(blockIdx.x) * 64;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = std::min(raw_count + selections, (share + 1) * share_keys);
    const size_t begin = share * share_keys;
    const unsigned t = threadIdx.x, local = t / 16, quad = t % 16;
    __shared__ float tops[attention_out_heads], denominators[attention_out_heads];
    __shared__ float4 vs[2][attention_keys][16];
    __shared__ float ps[2][attention_out_heads][attention_keys + 1];
    extern __shared__ const float* keys[];
    for (size_t index = begin + t; index < total; index += blockDim.x)
        keys[index - begin] = attention_key(index, raw_count, first, raw, raw_rows, compressed, selected + row * selections, total,
                                            dim);
    if (t < attention_out_heads) {
        const size_t head = std::min(head0 + t, heads - 1);
        const float2* parts = partial + (row * heads + head) * chunks;
        float top = sinks[head];
        for (size_t c = 0; c < chunks; ++c) top = fmaxf(top, parts[c].x);
        float denominator = expf(sinks[head] - top);
        for (size_t c = 0; c < chunks; ++c) if (parts[c].y > 0.0f) denominator += parts[c].y * expf(parts[c].x - top);
        tops[t] = top;
        denominators[t] = denominator;
        if (shares > 1 && share == 0 && blockIdx.x == 0 && head0 + t < heads) head_denominators[row * heads + head0 + t] = denominator;
    }
    __syncthreads();
    float4 v[4];
    float w[2];
    auto load = [&](size_t k0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned at = t + i * 128;
            const size_t index = k0 + at / 16;
            const float* key = index < total ? keys[index - begin] : nullptr;
            v[i] = key ? *reinterpret_cast<const float4*>(key + d0 + 4 * (at % 16)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned at = t + i * 128, h = at / attention_keys;
            const size_t head = head0 + h, index = k0 + at % attention_keys;
            const float score = head < heads && index < total ? scores[(row * heads + head) * stride + index] : -INFINITY;
            w[i] = score == -INFINITY ? 0.0f : expf(score - tops[h]);
        }
    };
    auto store = [&](int buffer) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) vs[buffer][(t + i * 128) / 16][(t + i * 128) % 16] = v[i];
        #pragma unroll
        for (int i = 0; i < 2; ++i) ps[buffer][(t + i * 128) / attention_keys][(t + i * 128) % attention_keys] = w[i];
    };
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    load(begin);
    store(0);
    __syncthreads();
    for (size_t k0 = begin, step = 0; k0 < total; k0 += attention_keys, ++step) {
        const int buffer = int(step & 1);
        const bool more = k0 + attention_keys < total;
        if (more) load(k0 + attention_keys);
        #pragma unroll 4
        for (int k = 0; k < attention_keys; ++k) {
            const float weight = ps[buffer][local][k];
            const float4 value = vs[buffer][k][quad];
            acc.x = fmaf(weight, value.x, acc.x); acc.y = fmaf(weight, value.y, acc.y);
            acc.z = fmaf(weight, value.z, acc.z); acc.w = fmaf(weight, value.w, acc.w);
        }
        if (more) store(buffer ^ 1);
        __syncthreads();
    }
    const size_t head = head0 + local, col = d0 + 4 * quad;
    const bool valid = head < heads;
    if (shares > 1) {
        if (valid) *reinterpret_cast<float4*>(sums + ((row * shares + share) * heads + head) * dim + col) = acc;
        if (!tiles) return;
        // With tile counters: the tile's last share to finish adds the shares' sums in order and
        // finishes them, as op_attention_combine_kernel does (the same denominators: every block
        // forms its heads' alike). Every share's writes are out before it counts itself done.
        __shared__ bool last;
        __threadfence();
        __syncthreads();
        unsigned* tile = tiles + (row * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        if (t == 0) last = atomicAdd(tile, 1u) == shares - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        if (t == 0) *tile = 0;
        if (!valid) return;
        float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t other = 0; other < shares; ++other) {
            float* at = sums + ((row * shares + other) * heads + head) * dim + col;
            #pragma unroll
            for (int i = 0; i < 4; ++i) total[i] += hc_step_load(at + i);
        }
        const float denominator = denominators[local];
        float o[4] = {total[0] / denominator, total[1] / denominator, total[2] / denominator, total[3] / denominator};
        attention_unrotate(o, row, col, dim - 2 * pairs, pairs, inverse);
        *reinterpret_cast<float4*>(out + (row * heads + head) * dim + col) = make_float4(o[0], o[1], o[2], o[3]);
        return;
    }
    if (!valid) return;
    const float denominator = denominators[local];
    float o[4] = {acc.x / denominator, acc.y / denominator, acc.z / denominator, acc.w / denominator};
    attention_unrotate(o, row, col, dim - 2 * pairs, pairs, inverse);
    *reinterpret_cast<float4*>(out + (row * heads + head) * dim + col) = make_float4(o[0], o[1], o[2], o[3]);
}
// The shares' sums added in order, over the head's denominator, rotated back: a thread per four values.
__global__ void op_attention_combine_kernel(const float* sums, const float* head_denominators, size_t rows, size_t shares,
                                            size_t heads, size_t dim, size_t pairs, const double2* inverse, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x, quads = rows * heads * dim / 4;
    if (index >= quads) return;
    const size_t row = index / (heads * dim / 4), head = index / (dim / 4) % heads, col = index % (dim / 4) * 4;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (size_t share = 0; share < shares; ++share) {
        const float4 v = *reinterpret_cast<const float4*>(sums + ((row * shares + share) * heads + head) * dim + col);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    const float denominator = head_denominators[row * heads + head];
    float o[4] = {acc.x / denominator, acc.y / denominator, acc.z / denominator, acc.w / denominator};
    attention_unrotate(o, row, col, dim - 2 * pairs, pairs, inverse);
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
// visible groups g < visible[row]; -inf beyond. A block per 32 groups of a row (blockIdx.y), up to
// 64 heads: q and the keys come through LDS 32 dims at a time, each dot summed in dim order and
// the heads' terms in head order.
__global__ void __launch_bounds__(256) op_index_scores_kernel(const float* q, const float* keys, const float* weights,
                                                              const int32_t* visible, size_t groups, size_t heads, size_t dim,
                                                              float scale, float* scores) {
    const size_t row = blockIdx.y, g0 = size_t(blockIdx.x) * 32;
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32;
    const size_t limit = std::min<size_t>(groups, size_t(max(visible[row], 0)));
    __shared__ float qs[64][33], ks[32][33], terms[64][33];
    float acc[8] = {};
    const float* row_q = q + row * heads * dim;
    for (size_t d0 = 0; d0 < dim; d0 += 32) {
        __syncthreads();
        for (unsigned i = t; i < 64 * 32; i += blockDim.x) {
            const size_t h = i / 32, col = d0 + i % 32;
            qs[i / 32][i % 32] = h < heads && col < dim ? row_q[h * dim + col] : 0.0f;
        }
        for (unsigned i = t; i < 32 * 32; i += blockDim.x) {
            const size_t g = g0 + i / 32, col = d0 + i % 32;
            ks[i / 32][i % 32] = g < limit && col < dim ? keys[g * dim + col] : 0.0f;
        }
        __syncthreads();
        const int width = int(std::min<size_t>(32, dim - d0));
        for (int c = 0; c < width; ++c) {
            const float k = ks[lane][c];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = fmaf(qs[wave + 8 * i][c], k, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const size_t h = wave + 8 * i;
        if (h < heads) terms[h][lane] = fmaxf(acc[i], 0.0f) * weights[row * heads + h];
    }
    __syncthreads();
    if (wave) return;
    const size_t g = g0 + lane;
    if (g >= groups) return;
    float total = 0.0f;
    for (size_t h = 0; h < heads; ++h) total += terms[h][lane];
    scores[row * groups + g] = g < limit ? total * scale : -INFINITY;
}

// One block per row: the k highest scores' indices in ascending index order, -1 for empty slots
// (fewer finite scores than k). Ties keep the lower index. The k-th highest score is found bit by
// bit over order-preserving keys (32 block-wide counts), then each thread takes its stretch of
// the row in order.
__device__ __forceinline__ uint32_t top_k_key(float value) {
    const uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);   // -0 ties +0, as the floats compare
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
// The block's sum of every thread's `value`, on every thread.
__device__ uint32_t top_k_sum(uint32_t value, uint32_t* shared) {
    for (unsigned offset = 16; offset; offset /= 2) value += __shfl_xor(value, offset, 32);
    __syncthreads();
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = value;
    __syncthreads();
    uint32_t total = 0;
    for (unsigned w = 0; w < blockDim.x / 32; ++w) total += shared[w];
    return total;
}
// Exclusive prefix of every thread's `value` in thread order.
__device__ uint32_t top_k_offset(uint32_t value, uint32_t* shared) {
    __syncthreads();
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        const uint32_t add = threadIdx.x >= stride ? shared[threadIdx.x - stride] : 0;
        __syncthreads();
        shared[threadIdx.x] += add;
        __syncthreads();
    }
    return shared[threadIdx.x] - value;
}
// A row's top k over its `groups` scores (score(g) loads score g) by the block, into out (k slots).
template<class Score>
__device__ void top_k_row(Score score, size_t groups, size_t k, int32_t* out, uint32_t* shared) {
    const size_t per = (groups + blockDim.x - 1) / blockDim.x, begin = std::min(groups, threadIdx.x * per);
    const size_t end = std::min(groups, begin + per);
    const uint32_t floor = top_k_key(-INFINITY);
    auto count = [&](uint32_t at_least) {
        uint32_t n = 0;
        for (size_t g = begin; g < end; ++g) {
            const uint32_t key = top_k_key(score(g));
            n += key > floor && key >= at_least;
        }
        return top_k_sum(n, shared);
    };
    // the k-th highest key, or the floor when fewer than k are finite (then every finite one is kept)
    uint32_t threshold = floor + 1;
    if (count(threshold) > k) {
        uint32_t prefix = 0;
        for (int bit = 31; bit >= 0; --bit)
            if (count(prefix | (1u << bit)) >= k) prefix |= 1u << bit;
        threshold = prefix;
    }
    // keys above the threshold are kept; of those equal to it, the lowest indices fill the rest
    uint32_t above = 0, ties = 0;
    for (size_t g = begin; g < end; ++g) {
        const uint32_t key = top_k_key(score(g));
        above += key > floor && key > threshold;
        ties += key > floor && key == threshold;
    }
    const uint32_t need = uint32_t(k) - std::min<uint32_t>(uint32_t(k), top_k_sum(above, shared));
    uint32_t tie = top_k_offset(ties, shared), kept = 0;
    for (size_t g = begin; g < end; ++g) {
        const uint32_t key = top_k_key(score(g));
        const bool equal = key > floor && key == threshold;
        kept += (key > floor && key > threshold) || (equal && tie < need);
        tie += equal;
    }
    uint32_t slot = top_k_offset(kept, shared);
    tie = top_k_offset(ties, shared);
    for (size_t g = begin; g < end; ++g) {
        const uint32_t key = top_k_key(score(g));
        const bool equal = key > floor && key == threshold;
        if ((key > floor && key > threshold) || (equal && tie < need)) out[slot++] = int32_t(g);
        tie += equal;
    }
    const uint32_t filled = top_k_sum(kept, shared);
    for (size_t i = filled + threadIdx.x; i < k; i += blockDim.x) out[i] = -1;
}
__global__ void __launch_bounds__(256) op_top_k_kernel(const float* scores, size_t groups, size_t k, int32_t* out) {
    const float* s = scores + size_t(blockIdx.x) * groups;
    __shared__ uint32_t shared[256];
    top_k_row([s](size_t g) { return s[g]; }, groups, k, out + size_t(blockIdx.x) * k, shared);
}

// A count other blocks of the launch move (device-wide), and the fences around it.
__device__ inline unsigned index_count(const unsigned* at) {
#if defined(__HIPCC__)
    return __hip_atomic_load(at, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return *static_cast<const volatile unsigned*>(at);
#endif
}
__device__ inline void index_release() {
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
    __threadfence();
#endif
}
__device__ inline void index_acquire() {
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#else
    __threadfence();
#endif
}
// A block of op_index_scores_kernel (groups [32 chunk, 32 chunk + 32) of row `row`, of `chunks` blocks a
// row) whose q has each head's last 2 * pairs values rotated as they load (op_rope_kernel's angles and
// arithmetic: the rotation it would have stored), then, with counters, the row's last block to finish
// takes the row's top k (op_top_k_kernel's), reading the scores the other blocks stored from the
// device-wide cache. counters: a row's blocks done (its last block zeroes it). With weights_done, the
// weights are another block's of the launch: read once that many blocks counted themselves done there.
__device__ __forceinline__ void index_select_block(const float* q, const float* keys, const float* weights, const int32_t* visible,
                                                   size_t groups, size_t heads, size_t dim, float scale, size_t pairs,
                                                   const int32_t* positions, const float* frequencies, float* scores, size_t k,
                                                   int32_t* out, unsigned* counters, size_t chunk, size_t row, size_t chunks,
                                                   const unsigned* weights_done, unsigned weight_blocks) {
    const size_t g0 = chunk * 32;
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32;
    const size_t limit = std::min<size_t>(groups, size_t(max(visible[row], 0)));
    __shared__ float qs[64][33], ks[32][33], terms[64][33];
    __shared__ double turns[64][2];
    for (size_t pair = t; pair < pairs; pair += blockDim.x) {
        double sine, cosine;
        sincos(double(positions[row]) * double(frequencies[pair]), &sine, &cosine);
        turns[pair][0] = cosine; turns[pair][1] = sine;
    }
    const size_t rotated = dim - 2 * pairs;
    float acc[8] = {};
    const float* row_q = q + row * heads * dim;
    for (size_t d0 = 0; d0 < dim; d0 += 32) {
        __syncthreads();
        for (unsigned i = t; i < 64 * 32; i += blockDim.x) {
            const size_t h = i / 32, col = d0 + i % 32;
            float v = 0.0f;
            if (h < heads && col < rotated) v = row_q[h * dim + col];
            else if (h < heads && col < dim) {
                const size_t pair = (col - rotated) / 2;
                const double a = row_q[h * dim + rotated + 2 * pair], b = row_q[h * dim + rotated + 2 * pair + 1];
                const double c = turns[pair][0], s = turns[pair][1];
                v = (col - rotated) % 2 == 0 ? float(a * c - b * s) : float(a * s + b * c);
            }
            qs[i / 32][i % 32] = v;
        }
        for (unsigned i = t; i < 32 * 32; i += blockDim.x) {
            const size_t g = g0 + i / 32, col = d0 + i % 32;
            ks[i / 32][i % 32] = g < limit && col < dim ? keys[g * dim + col] : 0.0f;
        }
        __syncthreads();
        const int width = int(std::min<size_t>(32, dim - d0));
        for (int c = 0; c < width; ++c) {
            const float key = ks[lane][c];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = fmaf(qs[wave + 8 * i][c], key, acc[i]);
        }
    }
    if (weights_done) {
        if (!t)
            while (index_count(weights_done) < weight_blocks) {
#if defined(__HIPCC__)
                __builtin_amdgcn_s_sleep(1);
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
                __nanosleep(32);
#endif
            }
        __syncthreads();
        index_acquire();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const size_t h = wave + 8 * i;
        if (h < heads) {
            const float* at = weights + row * heads + h;
            terms[h][lane] = fmaxf(acc[i], 0.0f) * (weights_done ? hc_step_load(const_cast<float*>(at)) : *at);
        }
    }
    __syncthreads();
    const size_t g = g0 + lane;
    if (!wave && g < groups) {
        float total = 0.0f;
        for (size_t h = 0; h < heads; ++h) total += terms[h][lane];
        scores[row * groups + g] = g < limit ? total * scale : -INFINITY;
    }
    if (!counters) return;
    // the block's scores out before it counts itself done; the row's last block sees them all
    __shared__ bool last;
    index_release();
    __syncthreads();
    if (!t) last = atomicAdd(counters + row, 1u) == chunks - 1;
    __syncthreads();
    if (!last) return;
    index_acquire();
    if (!t) counters[row] = 0;
    __shared__ uint32_t shared[256];
    float* row_scores = scores + row * groups;
    top_k_row([row_scores](size_t index) { return hc_step_load(row_scores + index); }, groups, k, out + row * k, shared);
}
__global__ void __launch_bounds__(256) op_index_select_kernel(const float* q, const float* keys, const float* weights,
                                                              const int32_t* visible, size_t groups, size_t heads, size_t dim,
                                                              float scale, size_t pairs, const int32_t* positions,
                                                              const float* frequencies, float* scores, size_t k, int32_t* out,
                                                              unsigned* counters) {
    index_select_block(q, keys, weights, visible, groups, heads, dim, scale, pairs, positions, frequencies, scores, k, out, counters,
                       blockIdx.x, blockIdx.y, gridDim.x, nullptr, 0);
}
// op_index_select_kernel of one row with the indexer's projection of x into its weights in the same launch
// (proj: gemv_fixed_kind<kind, 1>'s plan and waves). A block takes a ticket as it starts: the first
// proj_blocks tickets run the projection's blocks, the rest the score blocks, which read the weights once every
// projection block has counted itself done. Those took the earlier tickets and already run, so that wait
// always ends. counters: [0] tickets, [1] projection blocks done, [2] blocks done (the last block zeroes the
// three), [3] the row's score blocks done (its last block zeroes it).
template<int kind>
__global__ void __launch_bounds__(256) op_index_select_proj_kernel(
        FixedGemv proj, const float* x, unsigned proj_blocks, const float* q, const float* keys, float* weights,
        const int32_t* visible, size_t groups, size_t heads, size_t dim, float scale, size_t pairs, const int32_t* positions,
        const float* frequencies, float* scores, size_t k, int32_t* out, unsigned* counters) {
    __shared__ unsigned ticket;
    if (!threadIdx.x) ticket = atomicAdd(counters, 1u);
    __syncthreads();
    const unsigned mine = ticket, chunks = unsigned((groups + 31) / 32), blocks = proj_blocks + chunks;
    if (mine < proj_blocks) {
        const size_t wave = size_t(mine) * (blockDim.x / 32) + threadIdx.x / 32;
        const GemvItem& item = proj.items[0];
        gemv_rows<kind, 1>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, x, proj.input_stride, weights,
                           proj.output_stride, proj.residual);
        index_release();
        __syncthreads();
        if (!threadIdx.x) atomicAdd(counters + 1, 1u);
    } else {
        index_select_block(q, keys, weights, visible, groups, heads, dim, scale, pairs, positions, frequencies, scores, k, out,
                           counters + 3, mine - proj_blocks, 0, chunks, counters + 1, proj_blocks);
    }
    __syncthreads();
    if (!threadIdx.x && atomicAdd(counters + 2, 1u) == blocks - 1) {
        counters[0] = 0;
        counters[1] = 0;
        counters[2] = 0;
    }
}
