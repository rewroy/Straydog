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
// One row's rms_norm by a block of 256 threads; `sums` is 256 floats of the block's shared memory.
__device__ inline void op_rms_norm_row(const float* row, const float* weight, size_t width, float eps, float* out, float* sums) {
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
__global__ void op_rms_norm_kernel(const float* input, const float* weight, size_t width, float eps, float* output) { SD_KERNEL_ACQUIRE;
    __shared__ float sums[256];
    op_rms_norm_row(input + size_t(blockIdx.x) * width, weight, width, eps, output + size_t(blockIdx.x) * width, sums);
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
                               const float* frequencies, int inverse) { SD_KERNEL_ACQUIRE;
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

// Each of `rows` rows of `width` values, `copies` times over, row after row.
__global__ void op_repeat_rows_kernel(const float* input, size_t rows, size_t copies, size_t width, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    const size_t row = index / width, column = index % width;
    const float value = input[index];
    for (size_t copy = 0; copy < copies; ++copy) out[(row * copies + copy) * width + column] = value;
}

// steps rows of cols values (stride apart) rounded to FP16, packed.
__global__ void op_half_rows(const float* input, size_t stride, size_t cols, size_t steps, __half* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < steps * cols) out[index] = __float2half(input[(index / cols) * stride + index % cols]);
}

// Quality-only observers of the actual cast operands; never launched unless explicitly enabled.
__global__ void op_half_input_checks(const float* input, size_t stride, size_t cols, size_t steps,
                                     const __half* half, uint64_t* counters) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = steps * cols;
    __shared__ unsigned maximum[256];
    unsigned bits = 0;
    if (index < count) {
        const float before = input[(index / cols) * stride + index % cols];
        const float after = __half2float(half[index]);
        if (!isfinite(before)) atomicAdd(reinterpret_cast<unsigned long long*>(counters), 1ull);
        if (!isfinite(after)) atomicAdd(reinterpret_cast<unsigned long long*>(counters + 1), 1ull);
        if (isfinite(before) && isinf(after)) atomicAdd(reinterpret_cast<unsigned long long*>(counters + 2), 1ull);
        if (isfinite(before) && before != 0 && after == 0) atomicAdd(reinterpret_cast<unsigned long long*>(counters + 3), 1ull);
        if (isfinite(before)) bits = __float_as_uint(fabsf(before));
    }
    maximum[threadIdx.x] = bits;
    __syncthreads();
    for (unsigned offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) maximum[threadIdx.x] = max(maximum[threadIdx.x], maximum[threadIdx.x + offset]);
        __syncthreads();
    }
    if (!threadIdx.x) atomicMax(reinterpret_cast<unsigned long long*>(counters + 7),
                               static_cast<unsigned long long>(maximum[0]));
    if (!index) atomicAdd(reinterpret_cast<unsigned long long*>(counters + 5), static_cast<unsigned long long>(count));
}

__global__ void op_half_weight_checks(const __half* weights, size_t count, uint64_t* counters) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(__half2float(weights[index])))
        atomicAdd(reinterpret_cast<unsigned long long*>(counters + 4), 1ull);
    if (!index) atomicAdd(reinterpret_cast<unsigned long long*>(counters + 6), static_cast<unsigned long long>(count));
}

// out[t][r] = W[first + r] . in[t] for `rows` rows, inputs and outputs `input_stride` and
// `output_stride` values apart: GEMV for up to gemv_max_vectors inputs, else the rows expanded
// to F32 in `scratch` (in pieces that fit) and an F32 GEMM -- or, on a device set to FP16 GEMMs
// (sd_gemm_options), the rows expanded to FP16 and the inputs rounded to FP16 (at the scratch's
// end), an FP16 GEMM with F32 sums. The GEMV gives a row at least op_lanes lanes, so that a wave
// reads long runs of each row (short rows otherwise get two).
constexpr unsigned op_lanes = 8;
// Whether [pointer, pointer + bytes) lies in an allocation sd_upload made (weights).
bool in_uploaded(const Device& d, const uint8_t* pointer, size_t bytes) {
    auto next = d.uploaded.upper_bound(pointer);
    if (next == d.uploaded.begin()) return false;
    --next;
    return size_t(pointer - next->first) <= next->second && bytes <= next->second - size_t(pointer - next->first);
}
// A write into an allocation (or its release): an upload's it touches holds weights no more, and
// the expansions kept of weights go.
void written(Device& d, const void* pointer) {
    if (d.uploaded.empty()) return;
    const auto* at = static_cast<const uint8_t*>(pointer);
    auto next = d.uploaded.upper_bound(at);
    if (next == d.uploaded.begin()) return;
    --next;
    if (size_t(at - next->first) < next->second) {
        d.uploaded.erase(next);
        d.expanded_weights.clear();
    }
}

// The FP16 expansion of a weights matrix (an upload's: sd_upload) that the device keeps for a prompt
// pass's next GEMMs, made the first time (a pass's attention runs its rows ATTENTION_ROWS at a time,
// every batch taking the same weights): in a ring (sd_expansions), the expansions it overwrites
// forgotten (queued on one stream, a GEMM still reading one runs before the expansion that
// overwrites it). Null when the device keeps none of this matrix.
const __half* kept_expansion(Device& d, const Matrix& matrix) {
    if (!d.expansions || d.gemm_checks) return nullptr;
    const size_t bytes = (matrix.rows * matrix.cols * 2 + 255) & ~size_t(255);
    if (bytes > d.expansions_bytes || !in_uploaded(d, matrix.data, matrix.rows * matrix.row_bytes)) return nullptr;
    for (const auto& kept : d.expanded_weights)
        if (kept.data == matrix.data && kept.rows == matrix.rows && kept.cols == matrix.cols && kept.kind == matrix.kind)
            return reinterpret_cast<const __half*>(d.expansions + kept.at);
    const size_t at = d.expansions_at + bytes > d.expansions_bytes ? 0 : d.expansions_at;
    auto& list = d.expanded_weights;
    list.erase(std::remove_if(list.begin(), list.end(), [&](const auto& kept) { return kept.at < at + bytes && at < kept.at + kept.bytes; }),
               list.end());
    expand(matrix, reinterpret_cast<float*>(d.expansions + at), d.stream, true);
    d.expanded_weights.push_back({matrix.data, matrix.rows, matrix.cols, matrix.kind, at, bytes});
    d.expansions_at = at + bytes;
    return reinterpret_cast<const __half*>(d.expansions + at);
}
void op_linear(Device& d, const uint8_t* weights, int kind, size_t rows, size_t cols, size_t first,
               const float* input, size_t input_stride, size_t steps, float* output, size_t output_stride,
               float* scratch, size_t scratch_bytes, int precision = -1) {
    const size_t row_bytes = matrix_row_bytes(kind, cols);
    const Matrix matrix{weights + first * row_bytes, row_bytes, rows, cols, 0, 0, 0, kind};
    if (steps <= size_t(gemv_max_vectors)) {
        if (steps == 1) gemv(d, &matrix, 1, input, output, 1, 0, 0, {}, op_lanes);
        else gemv(d, &matrix, 1, input, output, steps, input_stride, output_stride, {}, op_lanes);   // strides in values
        return;
    }
    if (!d.blas) throw std::runtime_error("Straydog op: GEMM needs a BLAS handle");
    // -1 retains the device policy; explicit 0/1 selects one op without changing the device
    // or its graph cache. Small-row GEMV above is identical for every policy.
    if (precision < 0 ? d.half_gemm : precision != 0) {
        const size_t input_bytes = (steps * cols * 2 + 255) & ~size_t(255);
        if (scratch_bytes <= input_bytes) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
        auto* half_input = reinterpret_cast<__half*>(reinterpret_cast<uint8_t*>(scratch) + scratch_bytes - input_bytes);
        op_half_rows<<<unsigned((steps * cols + 255) / 256), 256, 0, d.stream>>>(input, input_stride, cols, steps, half_input);
        if (d.gemm_checks)
            op_half_input_checks<<<unsigned((steps * cols + 255) / 256), 256, 0, d.stream>>>(
                input, input_stride, cols, steps, half_input, d.gemm_checks);
        if (const __half* kept = kept_expansion(d, matrix)) {
            const float alpha = 1, beta = 0;
            blas_check(hipblasGemmEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(rows), int(steps), int(cols), &alpha, kept, HIP_R_16F, int(cols),
                                     half_input, HIP_R_16F, int(cols), &beta, output, HIP_R_32F, int(output_stride), HIPBLAS_COMPUTE_32F,
                                     HIPBLAS_GEMM_DEFAULT));
            return;
        }
        const size_t piece = std::min(rows, (scratch_bytes - input_bytes) / (cols * 2));
        if (!piece) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
        const float alpha = 1, beta = 0;
        for (size_t done = 0; done < rows; done += piece) {
            const size_t count = std::min(piece, rows - done);
            const Matrix part{matrix.data + done * row_bytes, row_bytes, count, cols, 0, 0, 0, kind};
            expand(part, scratch, d.stream, true);
            if (d.gemm_checks)
                op_half_weight_checks<<<unsigned((count * cols + 255) / 256), 256, 0, d.stream>>>(
                    reinterpret_cast<const __half*>(scratch), count * cols, d.gemm_checks);
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

#if defined(__gfx1200__) || defined(__gfx1201__)
// The value of the lane DPP control CTRL names (every lane active).
template<int CTRL>
__device__ __forceinline__ float dpp_mov(float v) {
    const int bits = __float_as_int(v);
    return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, CTRL, 0xf, 0xf, false));
}
#endif
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
#if defined(__gfx1200__) || defined(__gfx1201__)
    if constexpr (HC == 4) {
        // The same exchanges without LDS permutes: a row's entries are a quad (quad_perm broadcasts),
        // a column's the lanes ^ 4, ^ 8 and ^ 12 of a 16-lane row (row_xmask), taken in row order.
        const unsigned jj = j % 4;
        const auto pick = [jj](unsigned i, float g0, float g1, float g2, float g3) {
            const unsigned at = i ^ jj;
            return at == 0 ? g0 : at == 1 ? g1 : at == 2 ? g2 : g3;
        };
        float r0 = dpp_mov<0x00>(c), r1 = dpp_mov<0x55>(c), r2 = dpp_mov<0xAA>(c), r3 = dpp_mov<0xFF>(c);
        const float top = fmaxf(fmaxf(fmaxf(fmaxf(-INFINITY, r0), r1), r2), r3);
        float total = 0.0f;
        total += expf(r0 - top); total += expf(r1 - top); total += expf(r2 - top); total += expf(r3 - top);
        c = expf(c - top) / total + eps;
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            if (iteration) {
                r0 = dpp_mov<0x00>(c); r1 = dpp_mov<0x55>(c); r2 = dpp_mov<0xAA>(c); r3 = dpp_mov<0xFF>(c);
                float row = 0.0f;
                row += r0; row += r1; row += r2; row += r3;
                c /= row + eps;
            }
            const float g0 = c, g1 = dpp_mov<0x164>(c), g2 = dpp_mov<0x168>(c), g3 = dpp_mov<0x16C>(c);
            float column = 0.0f;
            column += pick(0, g0, g1, g2, g3); column += pick(1, g0, g1, g2, g3);
            column += pick(2, g0, g1, g2, g3); column += pick(3, g0, g1, g2, g3);
            c /= column + eps;
        }
        if (live) comb[lane] = c;
        return;
    }
#endif
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
// v plus the value of lane ^ OFFSET of a 32-lane wave (OFFSET 1, 2, 4, 8 or 16): __shfl_xor's exchange, on
// RDNA4 through DPP (row_xmask) and permlanex16 instead of an LDS permute (ds_bpermute's ~50-cycle
// round trip), the same values in the same sums.
template<unsigned OFFSET>
__device__ __forceinline__ float wave_xor_add(float v) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    const int bits = __float_as_int(v);
    if constexpr (OFFSET == 16) return v + __int_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210u, 0xfedcba98u, false, false));
    else return v + __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, 0x160 | OFFSET, 0xf, 0xf, false));
#else
    return v + __shfl_xor(v, OFFSET, 32);
#endif
}
template<int HC>
__global__ void __launch_bounds__(hc_step_threads) op_hc_step_kernel(const float* x, const float* residual, const float* post_in, const float* comb_in,
                                  float* stream, const float* fn, const float* base, const float* scale, size_t iterations,
                                  float eps, float norm_eps, const float* collapse, const float* norm_weight, size_t width,
                                  unsigned* counters, float* partial, float* pre, float* post, float* comb, float* out,
                                  PartRead link) { SD_KERNEL_ACQUIRE;
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
        // the weights' loads first: issued with x's and the residual's, not after the stream's store
        // (which the compiler cannot tell from fn's memory): one trip to DRAM, not two
        float f[MIX];
        if (fn && (reinterpret_cast<uintptr_t>(fn) & 1u)) {
            // BF16 weights (the address's bit 0 set, as state rows: state_bf16): half the bytes
            const uint16_t* halves = reinterpret_cast<const uint16_t*>(reinterpret_cast<uintptr_t>(fn) & ~uintptr_t(1));
            #pragma unroll
            for (int m = 0; m < MIX; ++m) f[m] = __uint_as_float(uint32_t(halves[(m * HC + j) * width + col]) << 16);
        } else if (fn) {
            #pragma unroll
            for (int m = 0; m < MIX; ++m) f[m] = fn[(m * HC + j) * width + col];
        }
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
            for (int m = 0; m < MIX; ++m) acc[m] += f[m] * v;
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
        v = wave_xor_add<16>(v); v = wave_xor_add<8>(v); v = wave_xor_add<4>(v); v = wave_xor_add<2>(v); v = wave_xor_add<1>(v);
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
    // The row's columns, as the blocks left them (their stores are out: each block counted itself in after its fence), loaded now with
    // their norm weights, so that those loads' latency is spent under the gather and the coefficients and not after them.
    constexpr unsigned EACH = 24;
    const bool early = width <= EACH * hc_step_threads;
    float* o = out + row * width;
    float early_x[EACH], early_w[EACH];
    if (early) {
        #pragma unroll
        for (unsigned j = 0; j < EACH; ++j) {
            const size_t col = threadIdx.x + j * hc_step_threads;
            if (col < width) { early_x[j] = hc_step_load(o + col); early_w[j] = norm_weight[col]; }
        }
    }
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
    const float factor = rsqrtf(totals[MIX + 1] / float(width) + norm_eps);
    // (the norm needs only its factor: the other waves store their columns while the first forms the coefficients)
    if (early && (!fn || wave)) {
        #pragma unroll
        for (unsigned j = 0; j < EACH; ++j) {
            const size_t col = threadIdx.x + j * hc_step_threads;
            if (col < width) o[col] = early_x[j] * factor * early_w[j];
        }
    }
    if (fn) {
        const float rsqrt = rsqrtf(totals[MIX] / float(HC * width) + norm_eps);
        if constexpr (HC * HC <= 32) {
            if (wave == 0)
                hc_coefficients_wave<HC>(totals, rsqrt, base, scale, iterations, eps, pre + row * HC, post + row * HC,
                                         comb + row * HC * HC);
        } else if (threadIdx.x == 0)
            hc_coefficients_of(totals, rsqrt, HC, base, scale, iterations, eps, pre + row * HC, post + row * HC, comb + row * HC * HC);
    }
    if (early) {
        if (fn && wave == 0) {
            #pragma unroll
            for (unsigned j = 0; j < EACH; ++j) {
                const size_t col = threadIdx.x + j * hc_step_threads;
                if (col < width) o[col] = early_x[j] * factor * early_w[j];
            }
        }
        return;
    }
    // the row's columns a few at a time, each group's loads issued before its stores
    for (size_t col0 = 0; col0 < width; col0 += size_t(8) * blockDim.x) {
        float loaded[8], weights[8];
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            const size_t col = col0 + j * blockDim.x + threadIdx.x;
            if (col < width) { loaded[j] = hc_step_load(o + col); weights[j] = norm_weight[col]; }
        }
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
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
// A DSpark target feature: each row's hc copies summed in order, ((x0 + x1) + x2) + x3, over hc, from
// row `skip` on, row r (of `rows`) into out + (r - skip) * stride + offset.
__global__ void op_hc_mean_kernel(const float* stream, size_t rows, size_t hc, size_t width, size_t skip, float* out, size_t stride,
                                  size_t offset) { SD_KERNEL_ACQUIRE;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    const size_t row = index / width, col = index % width;
    if (row < skip) return;
    const float* at = stream + row * hc * width + col;
    float sum = at[0];
    for (size_t copy = 1; copy < hc; ++copy) sum += at[copy * width];
    out[(row - skip) * stride + offset + col] = sum / float(hc);
}
__global__ void op_hc_collapse_kernel(const float* stream, const float* pre, size_t rows, size_t hc, size_t width, float* out) { SD_KERNEL_ACQUIRE;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    const size_t row = index / width, col = index % width;
    float sum = 0.0f;
    for (size_t j = 0; j < hc; ++j) sum += pre[row * hc + j] * stream[(row * hc + j) * width + col];
    out[index] = sum;
}

// out[t][j][d] = post[t][j] x[t][d] + sum_i comb[t][i][j] residual[t][i][d]
__global__ void op_hc_expand_kernel(const float* x, const float* residual, const float* post, const float* comb,
                                    size_t rows, size_t hc, size_t width, float* out) { SD_KERNEL_ACQUIRE;
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
    // Up to 512 experts, the picks by the first wave alone, with no barrier (lane i holds experts
    // i + 32m): each the highest biased score (the lowest index on a tie) of those left, as the
    // block's argmax picks it. (Six block-wide picks were most of a DeepSeek router's tail.)
    constexpr int WAVE_PICKS = 16;
    if (experts <= 32 * WAVE_PICKS && k <= 64) {
        if (wave) return;
        float b[WAVE_PICKS];
        unsigned left = 0;   // bit m: expert lane + 32m exists and is not picked
        #pragma unroll
        for (int m = 0; m < WAVE_PICKS; ++m) {
            const size_t e = lane + 32 * size_t(m);
            b[m] = e < experts ? biased[e] : 0.0f;
            if (e < experts) left |= 1u << m;
        }
        for (size_t slot = 0; slot < k; ++slot) {
            float value = -INFINITY;
            int32_t index = -1;
            #pragma unroll
            for (int m = 0; m < WAVE_PICKS; ++m) if (left >> m & 1u) route_take(value, index, b[m], int32_t(lane + 32 * m));
            for (unsigned offset = 16; offset; offset /= 2) {
                const float other_value = __shfl_xor(value, offset, 32);
                const int32_t other = __shfl_xor(index, offset, 32);
                route_take(value, index, other_value, other);
            }
            #pragma unroll
            for (int m = 0; m < WAVE_PICKS; ++m) if (index == int32_t(lane + 32 * m)) left &= ~(1u << m);
            if (!lane) picked[slot] = index;
        }
    } else {
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
// Whether the rows of a launch of `vectors` rows are routed by a wave a row (route_row_wave) rather than one after another by
// the block (route_row): their picks are a wave's (up to 512 experts) and no softmax sums them first (gating 0 does).
__host__ __device__ inline bool route_by_waves(size_t vectors, size_t experts, size_t k, int gating) {
    return vectors > 1 && vectors <= 8 && gating != 0 && experts <= 512 && k <= 64;
}
// The dynamic LDS bytes a launch of `vectors` rows' routes takes: route_row's scores and biased scores, and for
// route_row_wave's rows each their picks (64 words) too.
__host__ __device__ inline size_t route_scratch_bytes(size_t vectors, size_t experts, size_t k, int gating) {
    return route_by_waves(vectors, experts, k, gating) ? vectors * (experts * 8 + 256) : experts * 8;
}
// route_row's routes of one row by one wave alone (lane = threadIdx.x % 32), for the rows of a launch that waves route side by
// side with no barrier: the same scores, biased scores, picks and weights, with the same arithmetic. Not gating 0, and up to 512
// experts. `scores` is this row's experts x 2 floats, `picked` its 64 words.
__device__ void route_row_wave(const float* l, const float* bias, size_t experts, size_t k, int gating, int normalize, float scale,
                               int32_t* ids, float* weights, float* scores, int32_t* picked, int32_t* published = nullptr) {
    float* biased = scores + experts;
    const unsigned lane = threadIdx.x % 32;
    for (size_t e = lane; e < experts; e += 32) {
        const float x = hc_step_load(const_cast<float*>(l + e));
        scores[e] = gating == 1 ? 1.0f / (1.0f + expf(-x)) : gating == 2 ? sqrtf(x > 20.0f ? x : log1pf(expf(x))) : x;
    }
    for (size_t e = lane; e < experts; e += 32) biased[e] = scores[e] + (bias ? bias[e] : 0.0f);
    constexpr int WAVE_PICKS = 16;
    float b[WAVE_PICKS];
    unsigned left = 0;   // bit m: expert lane + 32m exists and is not picked
    #pragma unroll
    for (int m = 0; m < WAVE_PICKS; ++m) {
        const size_t e = lane + 32 * size_t(m);
        b[m] = e < experts ? biased[e] : 0.0f;
        if (e < experts) left |= 1u << m;
    }
    for (size_t slot = 0; slot < k; ++slot) {
        float value = -INFINITY;
        int32_t index = -1;
        #pragma unroll
        for (int m = 0; m < WAVE_PICKS; ++m) if (left >> m & 1u) route_take(value, index, b[m], int32_t(lane + 32 * m));
        for (unsigned offset = 16; offset; offset /= 2) {
            const float other_value = __shfl_xor(value, offset, 32);
            const int32_t other = __shfl_xor(index, offset, 32);
            route_take(value, index, other_value, other);
        }
        #pragma unroll
        for (int m = 0; m < WAVE_PICKS; ++m) if (index == int32_t(lane + 32 * m)) left &= ~(1u << m);
        if (!lane) picked[slot] = index;
    }
    // (the first lane read the scores and picks the others wrote, in one wave's order)
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
#else
    __syncwarp();
#endif
    if (lane) return;
    float sum = 0.0f;
    for (size_t slot = 0; slot < k; ++slot) sum += scores[picked[slot]];
    for (size_t slot = 0; slot < k; ++slot) {
        const float chosen = scores[picked[slot]];
        ids[slot] = picked[slot];
        weights[slot] = (normalize && k > 1 ? chosen / (sum + 1e-20f) : chosen) * scale;
        if (published) published[slot] = picked[slot];
    }
    if (published) __threadfence_system();
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
// Issue both units' loads before their dots; keep each dot and the single sum in order.
__device__ void gemv_routed_f32_rows(const Matrix& m, size_t first_row, unsigned lanes, const float* input,
                                     float* output, uint32_t output_stride, const GemvResidual& residual) {
    const unsigned lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = first_row + lane / lanes, units = m.cols / 4;
    float sum = 0.0f;
    if (row < m.rows) {
        const float4* weights = reinterpret_cast<const float4*>(m.data + row * m.row_bytes);
        const float* x = input + m.input_offset;
        for (size_t unit = sub; unit < units; unit += 2 * lanes) {
            const float4 w0 = weights[unit], a0 = gemv_load4(x + unit * 4);
            const bool next = unit + lanes < units;
            float4 w1, a1;
            if (next) { w1 = weights[unit + lanes]; a1 = gemv_load4(x + (unit + lanes) * 4); }
            sum += fmaf(w0.x, a0.x, fmaf(w0.y, a0.y, fmaf(w0.z, a0.z, w0.w * a0.w)));
            if (next) sum += fmaf(w1.x, a1.x, fmaf(w1.y, a1.y, fmaf(w1.z, a1.z, w1.w * a1.w)));
        }
    }
    float total = sum;
    for (unsigned offset = lanes / 2; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
    if (sub == 0 && row < m.rows) gemv_store(m, row, 0, total, output, output_stride, residual);
}
template<int kind, int V>
__global__ void gemv_routed_kind(FixedGemv list, const float* input, float* output, RouteTail tail) { SD_KERNEL_ACQUIRE;
    extern __shared__ float scores[];
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const GemvItem& item = list.items[0];
    if constexpr (kind == 0 && V == 1)
        gemv_routed_f32_rows(item.matrix, wave * (32 / item.lanes), item.lanes, input, output, list.output_stride, list.residual);
    else
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
    if (route_by_waves(V, tail.experts, tail.k, tail.gating)) {
        // a wave a row, side by side (the rows' routes one after another by the block took most of a four-row router's time)
        const unsigned row = threadIdx.x / 32;
        if (row < V) {
            float* mine = scores + row * (tail.experts * 2 + 64);
            route_row_wave(output + row * stride, tail.bias, tail.experts, tail.k, tail.gating, tail.normalize, tail.scale,
                           tail.ids + row * tail.k, tail.weights + row * tail.k, mine, reinterpret_cast<int32_t*>(mine + tail.experts * 2),
                           tail.published ? tail.published + row * tail.k : nullptr);
        }
        __syncthreads();
    } else {
        for (int row = 0; row < V; ++row) {
            route_row(output + row * stride, tail.bias, tail.experts, tail.k, tail.gating, tail.normalize, tail.scale, tail.ids + row * tail.k,
                      tail.weights + row * tail.k, scores, tail.published ? tail.published + row * tail.k : nullptr);
            __syncthreads();
        }
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
    if ((matrix.kind != 0 && matrix.kind != 1 && matrix.kind != 30) || vectors < 1 || vectors > 4) return false;
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
    const size_t shared = route_scratch_bytes(vectors, tail.experts, tail.k, tail.gating);
    #define SD_ROUTED(format, v) gemv_routed_kind<format, v><<<blocks, 256, shared, d.stream>>>(list, input, output, tail)
    if (matrix.kind == 0) {
        switch (vectors) { case 1: SD_ROUTED(0, 1); break; case 2: SD_ROUTED(0, 2); break; case 3: SD_ROUTED(0, 3); break; default: SD_ROUTED(0, 4); }
    } else if (matrix.kind == 30) {
        switch (vectors) { case 1: SD_ROUTED(30, 1); break; case 2: SD_ROUTED(30, 2); break; case 3: SD_ROUTED(30, 3); break; default: SD_ROUTED(30, 4); }
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

// DeepSeek-V4.1's compressed rows and index keys are held as BF16: their pointer then carries
// bit 0 (state_tag), and a reader turns each value into F32 as it loads it (a window row, F32,
// shares the attention's key table with them). An untagged pointer is F32.
__host__ __device__ inline const float* state_tag(const void* bf16) {
    return reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(bf16) | 1u);
}
__host__ __device__ inline bool state_bf16(const void* p) { return reinterpret_cast<uintptr_t>(p) & 1u; }
__host__ __device__ inline const uint16_t* state_halves(const void* p) {
    return reinterpret_cast<const uint16_t*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(1));
}
__device__ __forceinline__ float state_bf16_value(uint32_t half) { return __uint_as_float(half << 16); }
// Row `row` of `dim` values from `base`, tagged as base is.
__device__ __forceinline__ const float* state_row(const float* base, size_t row, size_t dim) {
    return state_bf16(base) ? state_tag(state_halves(base) + row * dim) : base + row * dim;
}
__device__ __forceinline__ float state_value(const float* row, size_t col) {
    return state_bf16(row) ? state_bf16_value(state_halves(row)[col]) : row[col];
}
// Four values from column `col` (a multiple of 4; a BF16 row 8-byte aligned), zeros for no row: an
// F32 row's in one 16-byte load when aligned.
__device__ __forceinline__ float4 state_float4(const float* row, size_t col) {
    if (!row) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (state_bf16(row)) {
        const uint2 h = *reinterpret_cast<const uint2*>(state_halves(row) + col);
        return make_float4(state_bf16_value(h.x & 0xffffu), state_bf16_value(h.x >> 16), state_bf16_value(h.y & 0xffffu),
                           state_bf16_value(h.y >> 16));
    }
    const float* at = row + col;
    if ((reinterpret_cast<uintptr_t>(at) & 15) == 0) return *reinterpret_cast<const float4*>(at);
    return make_float4(at[0], at[1], at[2], at[3]);
}
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
        return at < 0 ? nullptr : state_row(compressed, size_t(at), dim);
    };
    for (size_t index = threadIdx.x; index < total; index += blockDim.x) {
        const float* k = key(index);
        float dot = 0.0f;
        if (k) for (size_t i = 0; i < dim; ++i) dot += qs[i] * state_value(k, i);
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
            acc += scores[index] * state_value(key(index), i);
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
    return at < 0 ? nullptr : state_row(compressed, size_t(at), dim);
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
    // (rows of S + 4: 16-byte aligned for the dot's four-column LDS reads)
    __shared__ __align__(16) float qs[2][attention_heads][S + 4];
    __shared__ __align__(16) float ks[2][attention_keys][S + 4];
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
    // One thread takes four consecutive columns: a 16-byte load, and a 16-byte store into LDS. Rope
    // stays the scalar double path, and so does a row that is not 16-byte aligned.
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
            const float4 g = state_float4(keys[group / 16], d0 + (group % 16) * 4);
            kv[i * 4] = g.x; kv[i * 4 + 1] = g.y; kv[i * 4 + 2] = g.z; kv[i * 4 + 3] = g.w;
        }
    };
    auto store = [&](int buffer) {
        const unsigned q_head = t / 16, q_col = (t % 16) * 4;
        static_assert(QV == 4, "a thread's q columns are one float4");
        *reinterpret_cast<float4*>(&qs[buffer][q_head][q_col]) = make_float4(qv[0], qv[1], qv[2], qv[3]);
        #pragma unroll
        for (int i = 0; i < KV / 4; ++i) {
            const unsigned group = t + unsigned(i) * 256;
            const unsigned key_index = group / 16, key_col = (group % 16) * 4;
            *reinterpret_cast<float4*>(&ks[buffer][key_index][key_col]) = make_float4(kv[i * 4], kv[i * 4 + 1], kv[i * 4 + 2], kv[i * 4 + 3]);
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
        // Four columns a read (the key's own, both heads' broadcast): a third of the LDS reads, which
        // the dot was bound by (~0.8 us a step). The same products, summed in column order.
        #pragma unroll 4
        for (int c = 0; c < S; c += 4) {
            const float4 k = *reinterpret_cast<const float4*>(&ks[buffer][lane][c]);
            const float4 a = *reinterpret_cast<const float4*>(&qs[buffer][2 * wave][c]);
            const float4 b = *reinterpret_cast<const float4*>(&qs[buffer][2 * wave + 1][c]);
            acc[0] = fmaf(a.x, k.x, acc[0]); acc[1] = fmaf(b.x, k.x, acc[1]);
            acc[0] = fmaf(a.y, k.y, acc[0]); acc[1] = fmaf(b.y, k.y, acc[1]);
            acc[0] = fmaf(a.z, k.z, acc[0]); acc[1] = fmaf(b.z, k.z, acc[1]);
            acc[0] = fmaf(a.w, k.w, acc[0]); acc[1] = fmaf(b.w, k.w, acc[1]);
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
#if defined(__HIPCC__)
// op_attention_scores_kernel with FP16 operands and F32 sums through WMMA (a decode or verify pass with half_decode,
// RDNA4, 512 dims): the same grid, scores and partials. Wave w takes dims 64 w .. 64 w + 63 of the block's 16 heads (q
// as the A operand, its rotated values as the step kernel forms them) against its 32 keys (two B tiles, a missing key
// zeros), straight from memory; the waves' products then add in wave order through LDS, and the epilogue is the step
// kernel's.
__global__ void __launch_bounds__(256) op_attention_scores_half_kernel(
        const float* q, const float* raw, size_t raw_rows, const int32_t* positions, size_t window, const float* compressed,
        const int32_t* selected, size_t selections, size_t heads, size_t dim, float scale, size_t pairs, const double2* forward,
        size_t stride, float* scores, float2* partial) { SD_KERNEL_ACQUIRE;
#if defined(__gfx1200__) || defined(__gfx1201__)
    const size_t row = blockIdx.y, head0 = size_t(blockIdx.z) * attention_heads, chunks = gridDim.x;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32, side = lane / 16;
    __shared__ float sums[8][attention_heads][attention_keys + 1];
    __shared__ const float* keys[attention_keys];
    __shared__ double turns[64][2];
    for (size_t pair = t; pair < pairs; pair += blockDim.x) {
        const double2 g = forward[row * pairs + pair];
        turns[pair][0] = g.x; turns[pair][1] = g.y;
    }
    if (t < attention_keys)
        keys[t] = attention_key(size_t(blockIdx.x) * attention_keys + t, raw_count, first, raw, raw_rows, compressed,
                                selected + row * selections, total, dim);
    __syncthreads();
    const size_t rotated = dim - 2 * pairs, d0 = size_t(wave) * 64, h = head0 + lane % 16;
    const float* row_q = q + row * heads * dim;
    // A: the lane's head, columns d0 + 16 s + 8 side .. + 8
    moe_half8 a[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        const size_t c0 = d0 + 16 * s + 8 * side;
        float x[8] = {};
        if (h < heads && c0 + 8 <= rotated) {
            const float4 lo = *reinterpret_cast<const float4*>(row_q + h * dim + c0), hi = *reinterpret_cast<const float4*>(row_q + h * dim + c0 + 4);
            x[0] = lo.x; x[1] = lo.y; x[2] = lo.z; x[3] = lo.w; x[4] = hi.x; x[5] = hi.y; x[6] = hi.z; x[7] = hi.w;
        } else if (h < heads) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const size_t c = c0 + j;
                if (c < rotated) { x[j] = row_q[h * dim + c]; continue; }
                const size_t pair = (c - rotated) / 2;
                const double u = row_q[h * dim + rotated + 2 * pair], v = row_q[h * dim + rotated + 2 * pair + 1];
                const double cs = turns[pair][0], sn = turns[pair][1];
                x[j] = (c - rotated) % 2 == 0 ? float(u * cs - v * sn) : float(u * sn + v * cs);
            }
        }
        a[s] = moe_half8{_Float16(x[0]), _Float16(x[1]), _Float16(x[2]), _Float16(x[3]), _Float16(x[4]), _Float16(x[5]),
                         _Float16(x[6]), _Float16(x[7])};
    }
    // B: key 16 n + lane % 16, the same columns; acc[n][i] is head 8 side + i against that key
    moe_float8 acc[2] = {};
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        const float* key = keys[n * 16 + lane % 16];
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const size_t c0 = d0 + 16 * s + 8 * side;
            const float4 lo = key ? state_float4(key, c0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 hi = key ? state_float4(key, c0 + 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const moe_half8 b{_Float16(lo.x), _Float16(lo.y), _Float16(lo.z), _Float16(lo.w), _Float16(hi.x), _Float16(hi.y),
                              _Float16(hi.z), _Float16(hi.w)};
            acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[s], b, acc[n]);
        }
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n)
        #pragma unroll
        for (int i = 0; i < 8; ++i) sums[wave][side * 8 + i][n * 16 + lane % 16] = acc[n][i];
    __syncthreads();
    const size_t index = size_t(blockIdx.x) * attention_keys + lane;
    const bool valid = keys[lane] != nullptr;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned local = 2 * wave + i;
        float dot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) dot += sums[w][local][lane];
        const size_t head = head0 + local;
        const float score = valid ? dot * scale : -INFINITY;
        if (head < heads && index < total) scores[(row * heads + head) * stride + index] = score;
        float top = score;
        for (unsigned offset = 16; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 32));
        float sum = valid ? expf(score - top) : 0.0f;
        for (unsigned offset = 16; offset; offset /= 2) sum += __shfl_xor(sum, offset, 32);
        if (lane == 0 && head < heads) partial[(row * heads + head) * chunks + blockIdx.x] = make_float2(top, sum);
    }
#endif
}
#endif
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
        float* head_denominators, float* out, unsigned* tiles) { SD_KERNEL_ACQUIRE;
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
        float top = sinks[head], denominator;
        if (chunks <= 32) {
            // (every chunk's load issued before the first is used: they were waited for one by one)
            float2 got[32];
            #pragma unroll
            for (int c = 0; c < 32; ++c) if (size_t(c) < chunks) got[c] = parts[c];
            #pragma unroll
            for (int c = 0; c < 32; ++c) if (size_t(c) < chunks) top = fmaxf(top, got[c].x);
            denominator = expf(sinks[head] - top);
            #pragma unroll
            for (int c = 0; c < 32; ++c) if (size_t(c) < chunks && got[c].y > 0.0f) denominator += got[c].y * expf(got[c].x - top);
        } else {
            for (size_t c = 0; c < chunks; ++c) top = fmaxf(top, parts[c].x);
            denominator = expf(sinks[head] - top);
            for (size_t c = 0; c < chunks; ++c) if (parts[c].y > 0.0f) denominator += parts[c].y * expf(parts[c].x - top);
        }
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
            v[i] = state_float4(index < total ? keys[index - begin] : nullptr, d0 + 4 * (at % 16));
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
            ks[i / 32][i % 32] = g < limit && col < dim ? state_value(state_row(keys, g, dim), col) : 0.0f;
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
// (fewer finite scores than k). Ties keep the lower index. The k-th highest score is found over
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
// order-preserving keys (top_k_row), then each thread takes its stretch of the row in order.
__device__ __forceinline__ uint32_t top_k_key(float value) {
    const uint32_t bits = __float_as_uint(value == 0.0f ? 0.0f : value);   // -0 ties +0, as the floats compare
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
// The exclusive prefix of every thread's `value` in thread order, and their sum in `total`
// (wave shuffles, then the waves' sums through `scratch`, a slot a wave).
__device__ inline uint32_t top_k_scan(uint32_t value, uint32_t* scratch, uint32_t& total) {
    const unsigned lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = (blockDim.x + warpSize - 1) / warpSize;
    uint32_t inclusive = value;
    for (unsigned offset = 1; offset < warpSize; offset *= 2) {
#if defined(__HIPCC__)
        const uint32_t other = __shfl_up(inclusive, offset, warpSize);
#else
        const uint32_t other = __shfl_up_sync(0xffffffffu, inclusive, offset, warpSize);
#endif
        if (lane >= offset) inclusive += other;
    }
    __syncthreads();
    if (lane == warpSize - 1 || threadIdx.x == blockDim.x - 1) scratch[wave] = inclusive;
    __syncthreads();
    uint32_t before = 0, sum = 0;
    for (unsigned w = 0; w < waves; ++w) {
        const uint32_t c = scratch[w];
        before += w < wave ? c : 0u;
        sum += c;
    }
    total = sum;
    return before + inclusive - value;
}
// Values of a row each thread takes at once (consecutive), and so a tile of the block's.
constexpr unsigned top_k_run = 16;
// A row's top k over its `groups` scores (score(g) loads score g) by the block (256 threads),
// into out (k slots). The k-th highest key is found a byte at a time, most significant first: a
// 256-bin histogram (in shared memory) of the keys that match the bytes found so far picks the next
// byte, and the rank left within it, the threads scanning the bins together (four passes over the
// row). Then the row is taken a tile at a time, in order: keys above the threshold, and of those
// equal to it the lowest indices, as many as k leaves (the last pass's bin counts them).
template<class Score, class Index>
__device__ void top_k_row(Score score, size_t groups, size_t k, int32_t* out, uint32_t* shared, Index index) {
    const unsigned t = threadIdx.x, n = blockDim.x;
    const size_t tile = size_t(n) * top_k_run;
    const uint32_t floor = top_k_key(-INFINITY);
    __shared__ uint32_t bins[256];
    __shared__ uint32_t chosen, left, equal_count;
    // the k-th highest key, or the floor when fewer than k are finite (then every finite one is kept)
    uint32_t threshold = floor + 1, prefix = 0, rank = uint32_t(k);
    bool all = false;
    for (int byte = 3; byte >= 0; --byte) {
        __syncthreads();
        for (unsigned bin = t; bin < 256; bin += n) bins[bin] = 0;
        __syncthreads();
        const int shift = 8 * byte;
        // (a thread's loads of a run go out together, then it counts them)
        for (size_t first = size_t(t) * top_k_run; first < groups; first += tile) {
            uint32_t keys[top_k_run];
            for (unsigned i = 0; i < top_k_run; ++i) keys[i] = first + i < groups ? top_k_key(score(first + i)) : floor;
            for (unsigned i = 0; i < top_k_run; ++i) {
                const uint32_t key = keys[i];
                if (key > floor && (byte == 3 || (key >> (shift + 8)) == (prefix >> (shift + 8)))) atomicAdd(&bins[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        // thread i takes bin 255 - i: the bin where the count from the top reaches rank (else bin 0)
        const uint32_t count = t < 256 ? bins[255 - t] : 0u;
        uint32_t every;
        const uint32_t above = top_k_scan(count, shared, every);
        if (t < 256 && ((above < rank && above + count >= rank) || (t == 255 && above + count < rank))) {
            chosen = 255 - t;
            left = rank - above;
            equal_count = count;
        }
        __syncthreads();
        if (byte == 3 && every <= k) { all = true; break; }
        prefix |= chosen << shift;
        rank = left;
        if (!byte) threshold = prefix;
    }
    // (with the threshold the k-th key: rank keys equal to it are needed, of equal_count)
    const uint32_t need = all ? 0u : rank;
    const bool every_tie = all || equal_count <= need;
    uint32_t slot = 0, tie = 0;
    for (size_t base = 0; base < groups; base += tile) {
        const size_t first = base + size_t(t) * top_k_run;
        uint32_t keys[top_k_run];
        uint32_t ties = 0;
        for (unsigned i = 0; i < top_k_run; ++i) {
            keys[i] = first + i < groups ? top_k_key(score(first + i)) : floor;
            ties += !all && keys[i] > floor && keys[i] == threshold;
        }
        // (the ties before this thread's: every thread scans, the scan waits for the whole block)
        uint32_t tie_before = 0, tile_ties = 0;
        if (!every_tie) tie_before = tie + top_k_scan(ties, shared, tile_ties);
        bool keep[top_k_run];
        uint32_t kept = 0;
        for (unsigned i = 0; i < top_k_run; ++i) {
            const bool equal = !all && keys[i] > floor && keys[i] == threshold;
            keep[i] = keys[i] > floor && (all || keys[i] > threshold || (equal && (every_tie || tie_before < need)));
            tie_before += equal;
            kept += keep[i];
        }
        uint32_t tile_kept;
        uint32_t at = slot + top_k_scan(kept, shared, tile_kept);
        for (unsigned i = 0; i < top_k_run; ++i)
            if (keep[i]) out[at++] = index(first + i);
        slot += tile_kept;
        tie += tile_ties;
    }
    for (size_t i = slot + t; i < k; i += n) out[i] = -1;
}
template<class Score>
__device__ void top_k_row(Score score, size_t groups, size_t k, int32_t* out, uint32_t* shared) {
    top_k_row(score, groups, k, out, shared, [](size_t g) { return int32_t(g); });
}
// The bins of a row's keys by their top 12 bits (sd_op_index_select's, rows of top_k_bins), and the
// candidates a block holds in shared memory (a score and its index each, in top_k_bins words).
constexpr unsigned top_k_bins = 4096, top_k_room = top_k_bins / 2;
// A row's top k (top_k_row's, the same indices in the same order) from the bins its score blocks
// counted: the bin of the k-th highest key found from the bins (the block's threads 16 bins each,
// the highest first), the keys of that bin and above gathered in index order into `pool` (shared,
// top_k_bins words), and top_k_row over them alone. Zeroes the bins for the next launch. False
// (nothing written) when fewer than k keys are finite or too many share the bin: top_k_row over the
// row then.
__device__ bool top_k_from_bins(const float* row_scores, unsigned* bins, uint32_t* pool, size_t groups, size_t k, int32_t* out,
                                uint32_t* shared) {
    const unsigned t = threadIdx.x, n = blockDim.x;
    constexpr unsigned per = top_k_bins / 256;
    uint32_t mine[per], sum = 0;
    #pragma unroll
    for (unsigned i = 0; i < per; ++i) {
        mine[i] = t < 256 ? index_count(bins + (top_k_bins - 1 - (per * t + i))) : 0u;
        sum += mine[i];
    }
    uint32_t total;
    const uint32_t above = top_k_scan(sum, shared, total);
    __shared__ uint32_t bound, gathered;
    if (t < 256 && above < k && above + sum >= k) {
        uint32_t higher = above;
        unsigned i = 0;
        for (; i + 1 < per && higher + mine[i] < k; ++i) higher += mine[i];
        bound = top_k_bins - 1 - (per * t + i);
        gathered = higher + mine[i];
    }
    for (unsigned i = 0; t < 256 && i < per; ++i) bins[top_k_bins - 1 - (per * t + i)] = 0u;
    __syncthreads();
    if (total <= k || gathered > top_k_room) return false;
    const uint32_t floor = top_k_key(-INFINITY), lowest = bound;
    float* scores = reinterpret_cast<float*>(pool);
    uint32_t* indices = pool + top_k_room;
    const size_t tile = size_t(n) * top_k_run;
    uint32_t slot = 0;
    for (size_t base = 0; base < groups; base += tile) {
        const size_t first = base + size_t(t) * top_k_run;
        float values[top_k_run];
        uint32_t count = 0;
        for (unsigned i = 0; i < top_k_run; ++i) {
            values[i] = first + i < groups ? hc_step_load(const_cast<float*>(row_scores + first + i)) : -INFINITY;
            const uint32_t key = top_k_key(values[i]);
            count += key > floor && (key >> 20) >= lowest;
        }
        uint32_t tile_count;
        uint32_t at = slot + top_k_scan(count, shared, tile_count);
        for (unsigned i = 0; i < top_k_run; ++i) {
            const uint32_t key = top_k_key(values[i]);
            if (key > floor && (key >> 20) >= lowest) { scores[at] = values[i]; indices[at] = uint32_t(first + i); ++at; }
        }
        slot += tile_count;
    }
    __syncthreads();
    top_k_row([scores](size_t i) { return scores[i]; }, slot, k, out, shared, [indices](size_t i) { return int32_t(indices[i]); });
    return true;
}
__global__ void __launch_bounds__(256) op_top_k_kernel(const float* scores, size_t groups, size_t k, int32_t* out) {
    const float* s = scores + size_t(blockIdx.x) * groups;
    __shared__ uint32_t shared[256];
    top_k_row([s](size_t g) { return s[g]; }, groups, k, out + size_t(blockIdx.x) * k, shared);
}
// DeepSeek-V4.1's two-level indexer (the reference's select_candidate_blocks): the candidate source's
// scores pick a row's candidate blocks (`block` groups each), and the later indexers score and choose
// only among their groups. A row's candidates (`blocks` slots, in index order, -1 past the last) number
// its groups blocks * block in candidate order: candidate group v is group candidates[v / block] * block
// + v % block, past the row's groups where its slot is -1. Index order both ways, so a top k over the
// candidates is the top k over every group with the others at -inf, ties and order alike.
__device__ __forceinline__ size_t index_candidate(const int32_t* candidates, size_t block, size_t v, size_t groups) {
    const int32_t b = candidates[v / block];
    return b < 0 ? groups : size_t(b) * block + v % block;
}
// A row's blocks scored for its candidate pick (then op_top_k_kernel's top blocks: in index order, -1
// past the last): each block's highest of the candidate source's scores (as its select stored them: -inf
// past the row's visible groups), the block of the row's last visible group +inf (always kept: the newest,
// partly filled, could lose to an older full one); blocks of only -inf are never kept. A thread a block.
__global__ void __launch_bounds__(256) op_index_block_tops_kernel(const float* scores, const int32_t* visible, size_t groups,
                                                                  size_t block, size_t count, float* tops) {
    const size_t row = blockIdx.y, b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= count) return;
    const float* s = scores + row * groups;
    const size_t seen = std::min<size_t>(groups, size_t(max(visible[row], 0)));
    float top = seen && b == (seen - 1) / block ? INFINITY : -INFINITY;
    for (size_t g = b * block, end = std::min(groups, g + block); g < end; ++g) top = fmaxf(top, s[g]);
    tops[row * count + b] = top;
}
// op_top_k_kernel over a row's candidate groups (scores of every group): the groups' indices.
__global__ void __launch_bounds__(256) op_top_k_candidates_kernel(const float* scores, size_t groups, const int32_t* candidates,
                                                                  size_t blocks, size_t block, size_t k, int32_t* out) {
    const float* s = scores + size_t(blockIdx.x) * groups;
    const int32_t* c = candidates + size_t(blockIdx.x) * blocks;
    __shared__ uint32_t shared[256];
    top_k_row([=](size_t v) { const size_t g = index_candidate(c, block, v, groups); return g < groups ? s[g] : -INFINITY; },
              blocks * block, k, out + size_t(blockIdx.x) * k, shared,
              [=](size_t v) { return int32_t(index_candidate(c, block, v, groups)); });
}

// Groups of a row a block of the indexer scores (a thread each).
constexpr size_t index_block_groups = 256;
// A block of op_index_select_kernel (groups [256 chunk, 256 chunk + 256) of row `row`, of `chunks` blocks a
// row, a thread a group): q's rows (H of them, its heads and zero rows past them) a 32-column slice at a
// time in shared memory, each head's last 2 * pairs values rotated as they load (op_rope_kernel's angles
// and arithmetic: the rotation it would have stored), against the thread's group's key; a group's score
// is the sum over heads in order of relu(dot) * weight, each dot a chain of fmaf over the columns in
// order. Then, with counters, the row's last block to finish takes the row's top k (op_top_k_kernel's),
// reading the scores the other blocks stored from the device-wide cache. counters: a row's blocks done
// (its last block zeroes it). With weights_done, the weights are another block's of the launch: read
// once that many blocks counted themselves done there. With candidates (blocks slots a row of `block`
// groups, index_candidate's), the row's groups scored and chosen are its candidates', in candidate order:
// scores [row][candidate], the top k mapped back to the groups' indices. With half (RDNA4, 32 heads of
// 128), the dots take FP16 operands and sum in F32 through WMMA, as the prompt pass's
// ds_index_scores_wmma_rows (the reference's indexer rounds q and the keys to FP4): q's heads two tiles of
// 16 against each wave's 32 groups as two, a lane's eight heads of each weighted, then the halves added.
template<int H>
__device__ __forceinline__ void index_select_block(const float* q, const float* keys, const float* weights, const int32_t* visible,
                                                   size_t groups, size_t heads, size_t dim, float scale, size_t pairs,
                                                   const int32_t* positions, const float* frequencies, float* scores, size_t k,
                                                   int32_t* out, unsigned* counters, unsigned* bins, size_t chunk, size_t row,
                                                   size_t chunks, const unsigned* weights_done, unsigned weight_blocks,
                                                   float* rotated_q = nullptr, const unsigned* rotated_done = nullptr,
                                                   const int32_t* candidates = nullptr, size_t blocks = 0, size_t block = 1,
                                                   bool half = false) {
    const unsigned t = threadIdx.x;
    // the thread's place in the row's scored groups, and the group there
    const size_t scored = candidates ? blocks * block : groups, place = chunk * index_block_groups + t;
    const int32_t* row_candidates = candidates ? candidates + row * blocks : nullptr;
    const size_t g = !row_candidates ? place : place < scored ? index_candidate(row_candidates, block, place, groups) : groups;
    // (with bins: the block's keys counted by their top 12 bits here, then added to the row's)
    __shared__ uint32_t pool[top_k_bins];
    for (unsigned b = t; bins && b < top_k_bins; b += blockDim.x) pool[b] = 0u;
    const size_t limit = std::min<size_t>(groups, size_t(max(visible[row], 0)));
    __shared__ __align__(16) float qs[H][32];
    __shared__ double turns[64][2];
    if (rotated_q) {
        // q as another block of the launch rotated it (index_rotate_q), once it says so
        if (!t)
            while (index_count(rotated_done) < 1u) {
#if defined(__HIPCC__)
                __builtin_amdgcn_s_sleep(1);
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
                __nanosleep(32);
#endif
            }
        __syncthreads();
        index_acquire();
    }
    for (size_t pair = t; !rotated_q && pair < pairs; pair += blockDim.x) {
        double sine, cosine;
        sincos(double(positions[row]) * double(frequencies[pair]), &sine, &cosine);
        turns[pair][0] = cosine; turns[pair][1] = sine;
    }
    const size_t rotated = dim - 2 * pairs;
    float acc[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) acc[h] = 0.0f;
    const float* row_q = q + row * heads * dim;
    // (a thread past the row's groups reads its last group's key, and stores nothing)
    const float* key = state_row(keys, std::min(g, groups - 1), dim);
    const bool wide = dim % 4 == 0 && (state_bf16(keys) || reinterpret_cast<uintptr_t>(keys) % 16 == 0);
#if defined(__gfx1200__) || defined(__gfx1201__)
    const bool wmma = H == 32 && half && dim == 128 && wide;
    const unsigned lane = t % 32, side = lane / 16;
    moe_float8 tiles[2][2] = {};
    if (wmma) {
        __shared__ __attribute__((aligned(16))) _Float16 qh[32][128 + 8];
        __syncthreads();   // (the turns)
        for (unsigned at = t; at < 32 * 16; at += blockDim.x) {
            const unsigned h = at / 16, c0 = (at % 16) * 8;
            moe_half8 v = {};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const size_t col = c0 + j;
                float x = 0.0f;
                if (h >= heads) x = 0.0f;
                else if (rotated_q) x = hc_step_load(rotated_q + h * dim + col);
                else if (col < rotated) x = row_q[h * dim + col];
                else {
                    const size_t pair = (col - rotated) / 2;
                    const double a = row_q[h * dim + rotated + 2 * pair], b = row_q[h * dim + rotated + 2 * pair + 1];
                    const double c = turns[pair][0], s = turns[pair][1];
                    x = (col - rotated) % 2 == 0 ? float(a * c - b * s) : float(a * s + b * c);
                }
                v[j] = _Float16(x);
            }
            *reinterpret_cast<moe_half8*>(&qh[h][c0]) = v;
        }
        __syncthreads();
        // the wave's groups: 16 n + lane % 16 of its 32 (the last group's key past the row's), columns 16 s + 8 side .. + 8
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            const size_t at = chunk * index_block_groups + (t / 32) * 32 + n * 16 + lane % 16;
            const size_t gn = !row_candidates ? at : at < scored ? index_candidate(row_candidates, block, at, groups) : groups;
            const float* row_key = state_row(keys, std::min(gn, groups - 1), dim);
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                const float4 a = state_float4(row_key, 16 * s + 8 * side), b = state_float4(row_key, 16 * s + 8 * side + 4);
                const moe_half8 kv = {_Float16(a.x), _Float16(a.y), _Float16(a.z), _Float16(a.w), _Float16(b.x), _Float16(b.y),
                                      _Float16(b.z), _Float16(b.w)};
                #pragma unroll
                for (int m = 0; m < 2; ++m)
                    tiles[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        *reinterpret_cast<const moe_half8*>(&qh[m * 16 + lane % 16][16 * s + 8 * side]), kv, tiles[m][n]);
            }
        }
    }
#else
    constexpr bool wmma = false;
#endif
    for (size_t d0 = 0; !wmma && d0 < dim; d0 += 32) {
        __syncthreads();
        for (unsigned i = t; i < H * 32; i += blockDim.x) {
            const size_t h = i / 32, col = d0 + i % 32;
            float v = 0.0f;
            if (rotated_q) v = h < heads && col < dim ? hc_step_load(rotated_q + h * dim + col) : 0.0f;
            else if (h < heads && col < rotated) v = row_q[h * dim + col];
            else if (h < heads && col < dim) {
                const size_t pair = (col - rotated) / 2;
                const double a = row_q[h * dim + rotated + 2 * pair], b = row_q[h * dim + rotated + 2 * pair + 1];
                const double c = turns[pair][0], s = turns[pair][1];
                v = (col - rotated) % 2 == 0 ? float(a * c - b * s) : float(a * s + b * c);
            }
            qs[h][i % 32] = v;
        }
        __syncthreads();
        const int width = int(std::min<size_t>(32, dim - d0));
        float k_[32];
        if (wide && width == 32) {
            #pragma unroll
            for (int c = 0; c < 32; c += 4) {
                const float4 four = state_float4(key, d0 + c);
                k_[c] = four.x; k_[c + 1] = four.y; k_[c + 2] = four.z; k_[c + 3] = four.w;
            }
        } else {
            #pragma unroll
            for (int c = 0; c < 32; ++c) k_[c] = c < width ? state_value(key, d0 + c) : 0.0f;
        }
        // (the columns past a short slice are zero on both sides: each fmaf adds +0 or keeps -0's sign, as
        // an fmaf of zeros does; the chains stop at the slice's width as before)
        #pragma unroll
        for (int c = 0; c < 32; c += 4) {
            if (c >= width) break;
            #pragma unroll
            for (int h = 0; h < H; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(&qs[h][c]);
                acc[h] = fmaf(q4.x, k_[c], acc[h]);
                if (c + 1 < width) acc[h] = fmaf(q4.y, k_[c + 1], acc[h]);
                if (c + 2 < width) acc[h] = fmaf(q4.z, k_[c + 2], acc[h]);
                if (c + 3 < width) acc[h] = fmaf(q4.w, k_[c + 3], acc[h]);
            }
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
    float mine = 0.0f;
#if defined(__gfx1200__) || defined(__gfx1201__)
    if (wmma) {
        // the lane's heads 16 m + 8 side + i of both tiles, weighted, then the other side's added: the score
        // of group 16 n + lane % 16, of which lane L keeps the wave's group L
        float w[2][8];
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const size_t h = m * 16 + side * 8 + i;
                const float* at = weights + row * heads + h;
                w[m][i] = h < heads ? (weights_done ? hc_step_load(const_cast<float*>(at)) : *at) : 0.0f;
            }
        float sums[2];
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            float total = 0.0f;
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int i = 0; i < 8; ++i) total += fmaxf(tiles[m][n][i], 0.0f) * w[m][i];
            sums[n] = total + __shfl_xor(total, 16, 32);
        }
        mine = side ? sums[1] : sums[0];
    }
#endif
    if (place < scored) {
        // each head's term rounded on its own, then added in order (op_index_scores_kernel's, never fused)
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        float total = mine;
        #pragma unroll
        for (int h = 0; h < H; ++h) {
            if (wmma || size_t(h) >= heads) break;
            const float* at = weights + row * heads + h;
            const float term = fmaxf(acc[h], 0.0f) * (weights_done ? hc_step_load(const_cast<float*>(at)) : *at);
            total = total + term;
        }
        const float score = g < limit ? total * scale : -INFINITY;
        scores[row * scored + place] = score;
        const uint32_t key = top_k_key(score);
        if (bins && key > top_k_key(-INFINITY)) atomicAdd(&pool[key >> 20], 1u);
    }
    if (bins) {
        __syncthreads();
        for (unsigned b = t; b < top_k_bins; b += blockDim.x)
            if (const uint32_t count = pool[b]) atomicAdd(bins + row * top_k_bins + b, count);
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
    float* row_scores = scores + row * scored;
    if (!bins || !top_k_from_bins(row_scores, bins + row * top_k_bins, pool, scored, k, out + row * k, shared))
        top_k_row([row_scores](size_t index) { return hc_step_load(row_scores + index); }, scored, k, out + row * k, shared);
    if (!row_candidates) return;
    // (the block's own stores of the indices, seen after the barrier)
    __syncthreads();
    for (size_t i = t; i < k; i += blockDim.x)
        if (const int32_t at = out[row * k + i]; at >= 0) out[row * k + i] = int32_t(index_candidate(row_candidates, block, size_t(at), groups));
}
// (H: index_select_block's, 32 for up to 32 heads, else 64; one each, the launch picking: both in one kernel
// took both one's shared memory, 61 KB a block, and spilled)
template<int H>
__global__ void __launch_bounds__(256) op_index_select_kernel(const float* q, const float* keys, const float* weights,
                                                              const int32_t* visible, size_t groups, size_t heads, size_t dim,
                                                              float scale, size_t pairs, const int32_t* positions,
                                                              const float* frequencies, float* scores, size_t k, int32_t* out,
                                                              unsigned* counters, unsigned* bins, const int32_t* candidates,
                                                              size_t blocks, size_t block, bool half) {
    index_select_block<H>(q, keys, weights, visible, groups, heads, dim, scale, pairs, positions, frequencies, scores, k, out, counters,
                          bins, blockIdx.x, blockIdx.y, gridDim.x, nullptr, 0, nullptr, nullptr, candidates, blocks, block, half);
}
// op_index_select_kernel of one row with the indexer's projection of x into its weights in the same launch
// (proj: gemv_fixed_kind<kind, 1>'s plan and waves). A block takes a ticket as it starts: the first
// proj_blocks tickets run the projection's blocks, the rest the score blocks, which read the weights once every
// projection block has counted itself done. Those took the earlier tickets and already run, so that wait
// always ends. counters: [0] tickets, [1] projection blocks done, [2] blocks done (the last block zeroes the
// three and [4]), [3] the row's score blocks done (its last block zeroes it), [4] q rotated.
// The first ticket's block also rotates q once (index_select_block's arithmetic) into index_rotated_q, which
// the score blocks read: each of them rotated it on its own before.
constexpr size_t index_rotated_floats = 64 * 256;
__device__ float index_rotated_q[index_rotated_floats];
__device__ __forceinline__ void index_rotate_q(const float* q, size_t heads, size_t dim, size_t pairs, const int32_t* positions,
                                               const float* frequencies, unsigned* done) {
    __shared__ double turns[64][2];
    const unsigned t = threadIdx.x;
    for (size_t pair = t; pair < pairs; pair += blockDim.x) {
        double sine, cosine;
        sincos(double(positions[0]) * double(frequencies[pair]), &sine, &cosine);
        turns[pair][0] = cosine; turns[pair][1] = sine;
    }
    __syncthreads();
    const size_t rotated = dim - 2 * pairs;
    for (size_t i = t; i < heads * dim; i += blockDim.x) {
        const size_t h = i / dim, col = i % dim;
        float v = q[i];
        if (col >= rotated) {
            const size_t pair = (col - rotated) / 2;
            const double a = q[h * dim + rotated + 2 * pair], b = q[h * dim + rotated + 2 * pair + 1];
            const double c = turns[pair][0], s = turns[pair][1];
            v = (col - rotated) % 2 == 0 ? float(a * c - b * s) : float(a * s + b * c);
        }
        index_rotated_q[i] = v;
    }
    index_release();
    __syncthreads();
    if (!t) atomicAdd(done, 1u);
}
template<int kind, int H>
__global__ void __launch_bounds__(256) op_index_select_proj_kernel(
        FixedGemv proj, const float* x, unsigned proj_blocks, const float* q, const float* keys, float* weights,
        const int32_t* visible, size_t groups, size_t heads, size_t dim, float scale, size_t pairs, const int32_t* positions,
        const float* frequencies, float* scores, size_t k, int32_t* out, unsigned* counters, unsigned* bins,
        const int32_t* candidates, size_t candidate_blocks, size_t block, bool half) {
    __shared__ unsigned ticket;
    if (!threadIdx.x) ticket = atomicAdd(counters, 1u);
    __syncthreads();
    const size_t scored = candidates ? candidate_blocks * block : groups;
    const unsigned mine = ticket, chunks = unsigned((scored + index_block_groups - 1) / index_block_groups), blocks = proj_blocks + chunks;
    const bool once = heads * dim <= index_rotated_floats;
    if (mine < proj_blocks) {
        if (!mine && once) index_rotate_q(q, heads, dim, pairs, positions, frequencies, counters + 4);
        const size_t wave = size_t(mine) * (blockDim.x / 32) + threadIdx.x / 32;
        const GemvItem& item = proj.items[0];
        gemv_rows<kind, 1>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, x, proj.input_stride, weights,
                           proj.output_stride, proj.residual);
        index_release();
        __syncthreads();
        if (!threadIdx.x) atomicAdd(counters + 1, 1u);
    } else {
        index_select_block<H>(q, keys, weights, visible, groups, heads, dim, scale, pairs, positions, frequencies, scores, k, out,
                              counters + 3, bins, mine - proj_blocks, 0, chunks, counters + 1, proj_blocks,
                              once ? index_rotated_q : nullptr, counters + 4, candidates, candidate_blocks, block, half);
    }
    __syncthreads();
    if (!threadIdx.x && atomicAdd(counters + 2, 1u) == blocks - 1) {
        counters[0] = 0;
        counters[1] = 0;
        counters[2] = 0;
        counters[4] = 0;
    }
}
