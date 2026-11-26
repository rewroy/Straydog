// DeepSeek-V4.1's vision tower and aligner over one image (the reference's inference/vision.py):
// FP16 products with F32 sums (the device's hipBLAS handle) and a few row kernels. The residual
// stream stays F32; what a product reads is rounded to FP16 (the reference rounds to BF16).
//
// Per block: x += out(attention(rope(qkv(norm1(x))))), x += down(silu(gate) * up)(norm2(x)), where
// attention is full (every patch sees every patch), 16 heads of 64, the rope two-dimensional: the
// first half of a head's rotated pairs turns with the patch's row, the second with its column.
// After the last block and its norm, the aligner unfolds each 3x3 of patches (zeros past the
// grid's edge) into one row and maps it through w1, GELU and w2 to the model's width.

// Rows of attention scores a launch holds at most, in bytes (each head's scores of a block of rows).
constexpr size_t vision_score_bytes = size_t(128) << 20;

__global__ void vision_rms_norm(const float* x, const float* weight, size_t width, float eps, __half* half_out, float* out) {
    const float* row = x + size_t(blockIdx.x) * width;
    __shared__ float partial[256];
    float sum = 0;
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) sum += row[i] * row[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / float(width) + eps);
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = weight[i] * (row[i] * scale);
        if (half_out) half_out[size_t(blockIdx.x) * width + i] = __float2half(value);
        if (out) out[size_t(blockIdx.x) * width + i] = value;
    }
}

__global__ void vision_add_bias(float* x, const float* bias, size_t rows, size_t width) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < rows * width) x[i] += bias[i % width];
}

// q, k and v of each head ([heads][patches][dim], FP16) from the projection ([patches][3][heads][dim])
// and its bias: q and k turned by the 2D rope, q scaled by 1/sqrt(dim).
__global__ void vision_rope_split(const float* qkv, const float* bias, size_t patches, size_t grid_w, size_t heads, size_t dim,
                                  float theta, float q_scale, __half* q, __half* k, __half* v) {
    const size_t half = dim / 2, quarter = dim / 4;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= patches * heads * half) return;
    const size_t pair = i % half, head = (i / half) % heads, patch = i / (half * heads);
    const size_t width = heads * dim;
    const float* row = qkv + patch * 3 * width;
    const size_t at = head * dim + pair;
    // inv_freq over rope_dim = dim / 2 values: theta^(-2j / rope_dim), j < dim / 4
    const size_t j = pair % quarter;
    const float inv = 1.0f / powf(theta, float(2 * j) / float(half));
    const float position = float(pair < quarter ? patch / grid_w : patch % grid_w);
    const float angle = position * inv;
    const float c = cosf(angle), s = sinf(angle);
    const size_t out = (head * patches + patch) * dim + pair;
    for (int part = 0; part < 2; ++part) {
        const float x1 = row[part * width + at] + bias[part * width + at];
        const float x2 = row[part * width + at + half] + bias[part * width + at + half];
        const float scale = part == 0 ? q_scale : 1.0f;
        __half* target = part == 0 ? q : k;
        target[out] = __float2half((x1 * c - x2 * s) * scale);
        target[out + half] = __float2half((x2 * c + x1 * s) * scale);
    }
    v[out] = __float2half(row[2 * width + at] + bias[2 * width + at]);
    v[out + half] = __float2half(row[2 * width + at + half] + bias[2 * width + at + half]);
}

// A softmax row's threads, in both kernels and their launch.
constexpr unsigned vision_softmax_threads = 256;

// Softmax of each row of `cols` scores (row blockIdx.x of head blockIdx.y, heads `stride` rows
// apart), into FP16.
__global__ void vision_softmax(const float* scores, size_t cols, size_t stride, __half* probs) {
    const size_t at = (size_t(blockIdx.y) * stride + blockIdx.x) * cols;
    const float* row = scores + at;
    __half* out = probs + at;
    __shared__ float partial[vision_softmax_threads];
    float top = -INFINITY;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) top = fmaxf(top, row[i]);
    partial[threadIdx.x] = top;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    top = partial[0];
    __syncthreads();
    float sum = 0;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) sum += expf(row[i] - top);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / partial[0];
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) out[i] = __float2half(expf(row[i] - top) * scale);
}

// vision_softmax for a row of at most 256 * vision_softmax_kept scores, which stays in its threads'
// registers: the scores are read once and each exponential is taken once, the same values in the
// same order, so the probabilities are the same to the bit. (A 1280x800 picture's 5,336 patches:
// 230 of the encoder's 444 ms read three times, 194 so. Its own kernel: the registers it holds
// slowed a longer row's, 15,960 patches 4.22 -> 4.48 s on CUDA, when one kernel took both.)
constexpr unsigned vision_softmax_kept = 32;
__global__ void __launch_bounds__(vision_softmax_threads) vision_softmax_kept_row(const float* scores, size_t cols, size_t stride, __half* probs) {
    const size_t at = (size_t(blockIdx.y) * stride + blockIdx.x) * cols;
    const float* row = scores + at;
    __half* out = probs + at;
    __shared__ float partial[vision_softmax_threads];
    float value[vision_softmax_kept];
    float top = -INFINITY;
#pragma unroll
    for (unsigned k = 0; k < vision_softmax_kept; ++k) {
        const size_t i = threadIdx.x + size_t(k) * blockDim.x;
        if (i < cols) {
            value[k] = row[i];
            top = fmaxf(top, value[k]);
        }
    }
    partial[threadIdx.x] = top;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    top = partial[0];
    __syncthreads();
    float sum = 0;
#pragma unroll
    for (unsigned k = 0; k < vision_softmax_kept; ++k) {
        const size_t i = threadIdx.x + size_t(k) * blockDim.x;
        if (i < cols) {
            value[k] = expf(value[k] - top);
            sum += value[k];
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = 1.0f / partial[0];
#pragma unroll
    for (unsigned k = 0; k < vision_softmax_kept; ++k) {
        const size_t i = threadIdx.x + size_t(k) * blockDim.x;
        if (i < cols) out[i] = __float2half(value[k] * scale);
    }
}

// The softmax of `rows` rows of `cols` scores for each of `heads` heads (the kept row's kernel where it fits).
static void vision_softmax_rows(Device& d, const float* scores, size_t rows, size_t heads, size_t cols, size_t stride, __half* probs) {
    if (cols <= size_t(vision_softmax_threads) * vision_softmax_kept)
        vision_softmax_kept_row<<<dim3(unsigned(rows), unsigned(heads)), vision_softmax_threads, 0, d.stream>>>(scores, cols, stride, probs);
    else
        vision_softmax<<<dim3(unsigned(rows), unsigned(heads)), vision_softmax_threads, 0, d.stream>>>(scores, cols, stride, probs);
}

#if defined(__HIPCC__)
typedef _Float16 vflash_half8 __attribute__((ext_vector_type(8)));
typedef _Float16 vflash_half4 __attribute__((ext_vector_type(4)));
typedef float vflash_float8 __attribute__((ext_vector_type(8)));
constexpr unsigned vflash_keys = 64;   // keys a block stages a step: four of the waves' 16
constexpr unsigned vflash_ld = 72;     // halves an LDS row of 64 takes (fewer bank conflicts)
constexpr unsigned vflash_vt = vflash_keys + 8;

// The encoder's full attention for heads of 64 dimensions on RDNA4, without the scores in memory: grid (tiles of 64 queries, heads),
// 128 threads, each wave 16 queries. With the keys as rows, S^T = K Q^T leaves each lane one query's scores (column lane % 16) for
// eight keys, so the online softmax's maximum and sum need one exchange with the lane 16 away; and those scores, in FP16, are as they
// stand the B operand of O^T += V^T P^T (the keys taken in the order the first product's rows give them, the same for both operands).
// FP16 operands, F32 sums and softmax, as the GEMMs and vision_softmax: other bits than theirs (the probabilities are rounded before
// they are normalized, and summed in another order). q holds the softmax's scale already (vision_rope_split); q, k and v are each
// head's n rows of 64, `out` the rows of all heads (`width` values a row).
__global__ void __launch_bounds__(128) vision_flash(const __half* q, const __half* k, const __half* v, size_t n, size_t width, __half* out) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32, g = lane / 16, c = lane % 16;
    const size_t head = blockIdx.y, query = size_t(blockIdx.x) * 64 + wave * 16 + c;
    const _Float16* qh = reinterpret_cast<const _Float16*>(q) + head * n * 64;
    const _Float16* kh = reinterpret_cast<const _Float16*>(k) + head * n * 64;
    const _Float16* vh = reinterpret_cast<const _Float16*>(v) + head * n * 64;
    __shared__ __attribute__((aligned(16))) _Float16 ks[vflash_keys][vflash_ld];
    __shared__ __attribute__((aligned(16))) _Float16 vt[64][vflash_vt];
    // (the k index of a 16x16x16 operand's element i in lane group g)
    auto kof = [g](unsigned i) { return g * 4 + (i / 4) * 8 + i % 4; };
    vflash_half8 qb[4];
    #pragma unroll
    for (unsigned s = 0; s < 4; ++s) {
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) qb[s][i] = query < n ? qh[query * 64 + s * 16 + kof(i)] : _Float16(0);
    }
    vflash_float8 o[4] = {};   // O^T: tile s holds dimensions s*16 + g*8 + i of the lane's query
    float m = -INFINITY, l = 0;
    for (size_t k0 = 0; k0 < n; k0 += vflash_keys) {
        __syncthreads();
        for (unsigned idx = t; idx < vflash_keys * 8; idx += 128) {
            const unsigned row = idx / 8, part = idx % 8;
            const size_t key = k0 + row;
            uint4 kv = {0, 0, 0, 0}, vv = {0, 0, 0, 0};
            if (key < n) {
                kv = *reinterpret_cast<const uint4*>(kh + key * 64 + part * 8);
                vv = *reinterpret_cast<const uint4*>(vh + key * 64 + part * 8);
            }
            *reinterpret_cast<uint4*>(&ks[row][part * 8]) = kv;
            _Float16 values[8];
            memcpy(values, &vv, 16);
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) vt[part * 8 + j][row] = values[j];
        }
        __syncthreads();
        #pragma unroll
        for (unsigned sub = 0; sub < vflash_keys / 16; ++sub) {
            const size_t first = k0 + sub * 16;
            if (first >= n) break;
            vflash_float8 st = {};
            #pragma unroll
            for (unsigned s = 0; s < 4; ++s) {
                const _Float16* row = &ks[sub * 16 + c][s * 16 + g * 4];
                const vflash_half4 lo = *reinterpret_cast<const vflash_half4*>(row), hi = *reinterpret_cast<const vflash_half4*>(row + 8);
                const vflash_half8 a = {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
                st = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qb[s], st);
            }
            float top = m;
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                if (first + g * 8 + i >= n) st[i] = -INFINITY;
                top = fmaxf(top, st[i]);
            }
            top = fmaxf(top, __shfl_xor(top, 16));
            const float rescale = expf(m - top);
            float sum = 0;
            vflash_half8 p;
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const float e = expf(st[i] - top);
                sum += e;
                p[i] = _Float16(e);
            }
            sum += __shfl_xor(sum, 16);
            l = l * rescale + sum;
            m = top;
            #pragma unroll
            for (unsigned s = 0; s < 4; ++s) {
                o[s] *= rescale;
                const vflash_half8 a = *reinterpret_cast<const vflash_half8*>(&vt[s * 16 + c][sub * 16 + g * 8]);
                o[s] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, p, o[s]);
            }
        }
    }
    if (query < n) {
        const float scale = 1.0f / l;
        _Float16* row = reinterpret_cast<_Float16*>(out) + query * width + head * 64;
        #pragma unroll
        for (unsigned s = 0; s < 4; ++s) {
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) row[s * 16 + g * 8 + i] = _Float16(o[s][i] * scale);
        }
    }
#endif
}
#endif

// silu(gate) * up of each row of [gate | up] (`ffn` each), into FP16.
__global__ void vision_swiglu(const float* gate_up, size_t rows, size_t ffn, __half* out) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * ffn) return;
    const size_t row = i / ffn, col = i % ffn;
    const float gate = gate_up[row * 2 * ffn + col], up = gate_up[row * 2 * ffn + ffn + col];
    out[i] = __float2half(gate / (1.0f + expf(-gate)) * up);
}

// F.unfold of the patch grid ([grid_h][grid_w][width], zero past its edge) by r x r with stride r:
// each output row's values in (channel, row in the window, column in it) order.
__global__ void vision_unfold(const float* x, size_t grid_h, size_t grid_w, size_t width, size_t r, __half* out) {
    const size_t out_w = (grid_w + r - 1) / r, out_h = (grid_h + r - 1) / r;
    const size_t values = width * r * r;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= out_h * out_w * values) return;
    const size_t row = i / values, value = i % values;
    const size_t channel = value / (r * r), ki = (value / r) % r, kj = value % r;
    const size_t y = (row / out_w) * r + ki, x_at = (row % out_w) * r + kj;
    out[i] = __float2half(y < grid_h && x_at < grid_w ? x[(y * grid_w + x_at) * width + channel] : 0.0f);
}

// GELU (erf) of each value plus its column's bias, into FP16.
__global__ void vision_gelu(const float* x, const float* bias, size_t rows, size_t width, __half* out) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const float value = x[i] + bias[i % width];
    out[i] = __float2half(0.5f * value * (1.0f + erff(value * 0.70710678118654752f)));
}

__global__ void vision_bf16_to_half(uint16_t* values, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = __uint_as_float(uint32_t(values[i]) << 16);
    const __half half = __float2half(value);
    values[i] = *reinterpret_cast<const uint16_t*>(&half);
}

// output[steps][rows] (F32 or FP16) = input[steps][cols] . weights[rows][cols]^T (+ beta * output)
#if defined(__HIPCC__)
// Whether the attention takes vision_flash: RDNA4 and heads of 64, unless STRAYDOG_VISION_FLASH=0 (the scores in memory, as before).
static bool vision_flash_on(const Device& d, size_t dim) {
    static const bool off = [] { const char* value = std::getenv("STRAYDOG_VISION_FLASH"); return value && value[0] == '0' && !value[1]; }();
    return d.rdna4 && dim == 64 && !off;
}
#endif

static void vision_gemm(Device& d, const void* weights, size_t rows, size_t cols, const __half* input, size_t steps, void* output,
                        bool half_output, float beta) {
    const float alpha = 1;
    blas_check(hipblasGemmEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(rows), int(steps), int(cols), &alpha,
        weights, HIP_R_16F, int(cols), input, HIP_R_16F, int(cols), &beta,
        output, half_output ? HIP_R_16F : HIP_R_32F, int(rows), HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

// Rows of a block's FFN a product takes at a time.
constexpr size_t vision_ffn_rows = 2048;

// The activations of an image, laid out in its scratch: the stream, a block's normed input, q, k,
// v and the attention's output stay; one region holds in turn the projection to q, k and v, a block
// of rows' attention scores and probabilities, the FFN's rows, and the aligner's.
struct VisionScratch {
    size_t query_rows = 0, ffn_rows = 0, bytes = 0;
    size_t x = 0, normed = 0, q = 0, k = 0, v = 0, attended = 0, phase = 0;
    // offsets within `phase`
    size_t probs = 0, product = 0, unfolded = 0, hidden = 0, hidden_half = 0;
};

static VisionScratch vision_scratch(const sd_vision& model, size_t grid_h, size_t grid_w) {
    const size_t n = grid_h * grid_w, width = model.width, heads = model.heads, r = model.downsample;
    const size_t tokens = ((grid_h + r - 1) / r) * ((grid_w + r - 1) / r);
    auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    VisionScratch s;
    s.query_rows = std::max<size_t>(1, std::min(n, vision_score_bytes / (heads * n * 4)));
    s.ffn_rows = std::min(n, vision_ffn_rows);
    size_t at = 0;
    auto take = [&](size_t bytes) { const size_t start = at; at += align(bytes); return start; };
    s.x = take(n * width * 4);
    s.normed = take(n * width * 2);
    s.q = take(n * width * 2);
    s.k = take(n * width * 2);
    s.v = take(n * width * 2);
    s.attended = take(n * width * 2);
    s.probs = align(heads * s.query_rows * n * 4);
    s.product = align(s.ffn_rows * 2 * model.ffn * 4);
    s.unfolded = align(n * width * 4);
    s.hidden = s.unfolded + align(tokens * width * r * r * 2);
    s.hidden_half = s.hidden + align(tokens * model.out_width * 4);
    const size_t phase = std::max({n * 3 * width * 4, s.probs + heads * s.query_rows * n * 2, s.product + s.ffn_rows * model.ffn * 2,
                                   s.hidden_half + tokens * model.out_width * 2});
    s.phase = take(phase);
    s.bytes = at;
    return s;
}

static void vision_encode(Device& d, const sd_vision& model, const __half* patches, size_t grid_h, size_t grid_w, uint8_t* scratch,
                          size_t scratch_bytes, float* out) {
    if (!d.blas) throw std::runtime_error("the vision encoder needs the device's GEMM handle");
    if (!model.layers || !model.width || !model.heads || model.width % model.heads || (model.width / model.heads) % 4 || !grid_h || !grid_w)
        throw std::runtime_error("invalid Straydog vision encoder shape");
    const VisionScratch s = vision_scratch(model, grid_h, grid_w);
    if (s.bytes > scratch_bytes) throw std::runtime_error("the vision encoder's scratch is too small");
    op_span(d, scratch, s.bytes, "the vision encoder's scratch");
    const size_t n = grid_h * grid_w, width = model.width, heads = model.heads, dim = width / heads, ffn = model.ffn, r = model.downsample;
    const size_t tokens = ((grid_h + r - 1) / r) * ((grid_w + r - 1) / r);
    op_span(d, patches, n * model.patch_values * 2, "image patches");
    op_span(d, out, tokens * model.out_width * 4, "image rows");
    auto f32 = [&](size_t offset) { return reinterpret_cast<float*>(scratch + offset); };
    auto f16 = [&](size_t offset) { return reinterpret_cast<__half*>(scratch + offset); };
    float* x = f32(s.x);
    // patches -> x
    vision_gemm(d, model.patch, width, model.patch_values, patches, n, x, false, 0);
    vision_add_bias<<<op_blocks(n * width), 256, 0, d.stream>>>(x, static_cast<const float*>(model.patch_bias), n, width);
    for (size_t layer = 0; layer < model.layers; ++layer) {
        const sd_vision_layer& w = model.layer[layer];
        vision_rms_norm<<<unsigned(n), 256, 0, d.stream>>>(x, static_cast<const float*>(w.norm1), width, model.eps, f16(s.normed), nullptr);
        vision_gemm(d, w.qkv, 3 * width, width, f16(s.normed), n, f32(s.phase), false, 0);
        vision_rope_split<<<op_blocks(n * heads * dim / 2), 256, 0, d.stream>>>(f32(s.phase), static_cast<const float*>(w.qkv_bias), n, grid_w,
            heads, dim, model.rope_theta, 1.0f / sqrtf(float(dim)), f16(s.q), f16(s.k), f16(s.v));
#if defined(__HIPCC__)
        if (vision_flash_on(d, dim)) {
            vision_flash<<<dim3(unsigned((n + 63) / 64), unsigned(heads)), 128, 0, d.stream>>>(f16(s.q), f16(s.k), f16(s.v), n, width, f16(s.attended));
        } else
#endif
        {
            float* scores = f32(s.phase);
            __half* probs = f16(s.phase + s.probs);
            for (size_t first = 0; first < n; first += s.query_rows) {
                const size_t rows = std::min(s.query_rows, n - first);
                const float alpha = 1, zero = 0;
                // scores[h][row][patch] = q[h][first + row] . k[h][patch]
                blas_check(hipblasGemmStridedBatchedEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(n), int(rows), int(dim), &alpha,
                    f16(s.k), HIP_R_16F, int(dim), (long long)(n * dim), f16(s.q) + first * dim, HIP_R_16F, int(dim), (long long)(n * dim),
                    &zero, scores, HIP_R_32F, int(n), (long long)(s.query_rows * n), int(heads), HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
                vision_softmax_rows(d, scores, rows, heads, n, s.query_rows, probs);
                // attended[first + row][h * dim ..] = probs[h][row] . v[h]
                blas_check(hipblasGemmStridedBatchedEx(d.blas, HIPBLAS_OP_N, HIPBLAS_OP_N, int(dim), int(rows), int(n), &alpha,
                    f16(s.v), HIP_R_16F, int(dim), (long long)(n * dim), probs, HIP_R_16F, int(n), (long long)(s.query_rows * n),
                    &zero, f16(s.attended) + first * width, HIP_R_16F, int(width), (long long)dim, int(heads), HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT));
            }
        }
        vision_gemm(d, w.out, width, width, f16(s.attended), n, x, false, 1);
        vision_add_bias<<<op_blocks(n * width), 256, 0, d.stream>>>(x, static_cast<const float*>(w.out_bias), n, width);
        vision_rms_norm<<<unsigned(n), 256, 0, d.stream>>>(x, static_cast<const float*>(w.norm2), width, model.eps, f16(s.normed), nullptr);
        for (size_t first = 0; first < n; first += s.ffn_rows) {
            const size_t rows = std::min(s.ffn_rows, n - first);
            vision_gemm(d, w.gate_up, 2 * ffn, width, f16(s.normed) + first * width, rows, f32(s.phase), false, 0);
            vision_swiglu<<<op_blocks(rows * ffn), 256, 0, d.stream>>>(f32(s.phase), rows, ffn, f16(s.phase + s.product));
            vision_gemm(d, w.down, width, ffn, f16(s.phase + s.product), rows, x + first * width, false, 1);
        }
    }
    float* normed = f32(s.phase);
    vision_rms_norm<<<unsigned(n), 256, 0, d.stream>>>(x, static_cast<const float*>(model.norm), width, model.eps, nullptr, normed);
    __half* unfolded = f16(s.phase + s.unfolded);
    vision_unfold<<<op_blocks(tokens * width * r * r), 256, 0, d.stream>>>(normed, grid_h, grid_w, width, r, unfolded);
    vision_gemm(d, model.align1, model.out_width, width * r * r, unfolded, tokens, f32(s.phase + s.hidden), false, 0);
    vision_gelu<<<op_blocks(tokens * model.out_width), 256, 0, d.stream>>>(f32(s.phase + s.hidden), static_cast<const float*>(model.align1_bias),
        tokens, model.out_width, f16(s.phase + s.hidden_half));
    vision_gemm(d, model.align2, model.out_width, model.out_width, f16(s.phase + s.hidden_half), tokens, out, false, 0);
    vision_add_bias<<<op_blocks(tokens * model.out_width), 256, 0, d.stream>>>(out, static_cast<const float*>(model.align2_bias), tokens,
        model.out_width);
    check(hipGetLastError());
}

int sd_vision_scratch_bytes(const sd_vision* model, uint64_t grid_h, uint64_t grid_w, uint64_t* bytes) {
    try {
        if (!model || !bytes || !model->heads || !grid_h || !grid_w) throw std::runtime_error("invalid Straydog vision encoder shape");
        *bytes = vision_scratch(*model, size_t(grid_h), size_t(grid_w)).bytes;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_vision_encode(void* handle, const sd_vision* model, const void* patches, uint64_t grid_h, uint64_t grid_w, void* scratch,
                     uint64_t scratch_bytes, float* out) {
    try {
        auto& value = device(handle);
        if (!model) throw std::runtime_error("no vision encoder");
        vision_encode(value, *model, static_cast<const __half*>(patches), size_t(grid_h), size_t(grid_w), static_cast<uint8_t*>(scratch),
                      size_t(scratch_bytes), out);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// BF16 values in place as FP16 (the encoder's weights, uploaded as the checkpoint has them).
int sd_vision_half(void* handle, void* values, uint64_t count) {
    try {
        auto& value = device(handle);
        op_span(value, values, size_t(count) * 2, "vision weights");
        vision_bf16_to_half<<<unsigned((count + 255) / 256), 256, 0, value.stream>>>(static_cast<uint16_t*>(values), size_t(count));
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
