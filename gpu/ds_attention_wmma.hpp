// DeepSeek-V4.1's sparse attention over a prompt pass's rows in FP16 mode (RDNA4): its two products
// through WMMA, every head of a row in one block, so a row's keys are read once a block rather than
// once per group of heads (the F32 kernels read them two to four times, and their F32 FMA loops
// ran at a tenth of the device's rate). The scores Q K^T and the weighted sums P V take FP16
// operands and sum in F32, as the pass's other products do; the softmax stays F32. The scores and
// each chunk's maximum and sum are laid out as op_attention_scores_kernel's (chunks of
// ds_attention_chunk keys), the output as op_attention_output_kernel's.
#pragma once

#if defined(__HIPCC__)
constexpr int ds_attention_chunk = 64;   // keys a scores block takes: four waves of 16
constexpr int ds_attention_lds = 72;     // halves an LDS row of 64 values takes (16-byte rows, fewer bank conflicts)

// Eight F32 values as FP16 (round to nearest).
__device__ __forceinline__ moe_half8 ds_half8(float4 a, float4 b) {
    return moe_half8{_Float16(a.x), _Float16(a.y), _Float16(a.z), _Float16(a.w), _Float16(b.x), _Float16(b.y), _Float16(b.z),
                     _Float16(b.w)};
}

// Scores: grid (chunks of ds_attention_chunk keys, rows), 128 threads; HM tiles of 16 heads (all a
// row's). Wave w takes keys 16w.. of the chunk for every head. The next 64 columns of q and of the
// keys load into registers while the current ones' products run (LDS in two buffers).
template<int HM>
__global__ void __launch_bounds__(128) ds_attention_scores_wmma(
        const float* q, const float* raw, size_t raw_rows, const int32_t* positions, size_t window, const float* compressed,
        const int32_t* selected, size_t selections, size_t heads, size_t dim, float scale, size_t pairs, const double2* forward,
        size_t stride, float* scores, float2* partial) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int S = 64, L = ds_attention_lds, H = HM * 16, QG = H * 8 / 128, KG = ds_attention_chunk * 8 / 128;
    const size_t row = blockIdx.y, chunk = blockIdx.x, chunks = gridDim.x;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32, half = lane / 16;
    __shared__ __attribute__((aligned(16))) _Float16 qs[2][H][L];
    __shared__ __attribute__((aligned(16))) _Float16 ks[2][ds_attention_chunk][L];
    __shared__ const float* keys[ds_attention_chunk];
    __shared__ double turns[64][2];   // the row's rotation per rope pair
    __shared__ float2 reduce[4][H];
    for (size_t pair = t; pair < pairs; pair += blockDim.x) {
        const double2 g = forward[row * pairs + pair];
        turns[pair][0] = g.x;
        turns[pair][1] = g.y;
    }
    if (t < ds_attention_chunk)
        keys[t] = attention_key(chunk * ds_attention_chunk + t, raw_count, first, raw, raw_rows, compressed, selected + row * selections,
                                total, dim);
    __syncthreads();
    const size_t rotated = dim - 2 * pairs;
    const float* row_q = q + row * heads * dim;
    // A thread's eight q values of a head (rotated past `rotated`) and eight values of a key.
    float4 qa[QG][2], ka[KG][2];
    auto load = [&](size_t d0) {
        #pragma unroll
        for (int g = 0; g < QG; ++g) {
            const unsigned at = t + g * 128, h = at / 8, c0 = (at % 8) * 8;
            qa[g][0] = qa[g][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (h >= heads) continue;
            const float* source = row_q + h * dim + d0 + c0;
            if (d0 + c0 + 8 <= rotated) {
                qa[g][0] = reinterpret_cast<const float4*>(source)[0];
                qa[g][1] = reinterpret_cast<const float4*>(source)[1];
            } else {
                float x[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const size_t c = d0 + c0 + j;
                    if (c < rotated) { x[j] = row_q[h * dim + c]; continue; }
                    const size_t pair = (c - rotated) / 2;
                    const double a = row_q[h * dim + rotated + 2 * pair], b = row_q[h * dim + rotated + 2 * pair + 1];
                    const double cs = turns[pair][0], sn = turns[pair][1];
                    x[j] = (c - rotated) % 2 == 0 ? float(a * cs - b * sn) : float(a * sn + b * cs);
                }
                qa[g][0] = make_float4(x[0], x[1], x[2], x[3]);
                qa[g][1] = make_float4(x[4], x[5], x[6], x[7]);
            }
        }
        #pragma unroll
        for (int g = 0; g < KG; ++g) {
            const unsigned at = t + g * 128, k = at / 8, c0 = (at % 8) * 8;
            const float* key = keys[k];
            ka[g][0] = key ? reinterpret_cast<const float4*>(key + d0 + c0)[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            ka[g][1] = key ? reinterpret_cast<const float4*>(key + d0 + c0)[1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store = [&](int buffer) {
        #pragma unroll
        for (int g = 0; g < QG; ++g) {
            const unsigned at = t + g * 128;
            *reinterpret_cast<moe_half8*>(&qs[buffer][at / 8][(at % 8) * 8]) = ds_half8(qa[g][0], qa[g][1]);
        }
        #pragma unroll
        for (int g = 0; g < KG; ++g) {
            const unsigned at = t + g * 128;
            *reinterpret_cast<moe_half8*>(&ks[buffer][at / 8][(at % 8) * 8]) = ds_half8(ka[g][0], ka[g][1]);
        }
    };
    moe_float8 acc[HM] = {};
    load(0);
    store(0);
    __syncthreads();
    const size_t steps = dim / S;
    for (size_t step = 0; step < steps; ++step) {
        const int buffer = int(step & 1);
        if (step + 1 < steps) load((step + 1) * S);
        #pragma unroll
        for (int kk = 0; kk < S; kk += 16) {
            const moe_half8 b = *reinterpret_cast<const moe_half8*>(&ks[buffer][wave * 16 + lane % 16][kk + 8 * half]);
            #pragma unroll
            for (int m = 0; m < HM; ++m) {
                const moe_half8 a = *reinterpret_cast<const moe_half8*>(&qs[buffer][m * 16 + lane % 16][kk + 8 * half]);
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[m]);
            }
        }
        if (step + 1 < steps) store(buffer ^ 1);
        __syncthreads();
    }
    // acc[m][i]: head m * 16 + half * 8 + i, this lane's key. Each half of the wave reduces its 16
    // keys per head; the waves' maxima and sums combine below.
    const unsigned key = wave * 16 + lane % 16;
    const size_t index = chunk * ds_attention_chunk + key;
    const bool valid = keys[key] != nullptr;
    #pragma unroll
    for (int m = 0; m < HM; ++m)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned head = m * 16 + half * 8 + i;
            const float score = valid ? acc[m][i] * scale : -INFINITY;
            if (head < heads && index < total) scores[(row * heads + head) * stride + index] = score;
            float top = score;
            for (unsigned offset = 8; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 32));
            float sum = valid ? expf(score - top) : 0.0f;
            for (unsigned offset = 8; offset; offset /= 2) sum += __shfl_xor(sum, offset, 32);
            if (lane % 16 == 0) reduce[wave][head] = make_float2(top, sum);
        }
    __syncthreads();
    if (t < H && t < heads) {
        float top = -INFINITY;
        for (int w = 0; w < 4; ++w) top = fmaxf(top, reduce[w][t].x);
        float sum = 0.0f;
        if (top != -INFINITY)
            for (int w = 0; w < 4; ++w)
                if (reduce[w][t].y > 0.0f) sum += reduce[w][t].y * expf(reduce[w][t].x - top);
        partial[(row * heads + t) * chunks + chunk] = make_float2(top, sum);
    }
#endif
}

// Weighted sums: grid (dim / 64, rows), 128 threads, every head of the row (HM tiles of 16); wave w
// takes columns d0 + 16w ... Keys go 64 a stage: the next stage's weights and values load into
// registers while the current one's products run (LDS in two buffers). Dynamic LDS: the row's key
// addresses.
template<int HM>
__global__ void __launch_bounds__(128) ds_attention_output_wmma(
        const float* scores, const float2* partial, size_t chunks, const float* raw, size_t raw_rows, const int32_t* positions,
        size_t window, const float* compressed, const int32_t* selected, size_t selections, const float* sinks, size_t heads,
        size_t dim, size_t pairs, const double2* inverse, size_t stride, float* out) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int H = HM * 16, KS = 64, L = ds_attention_lds, PW = H * KS / 128;
    const size_t row = blockIdx.y, d0 = size_t(blockIdx.x) * 64;
    const int64_t p = positions[row];
    const int64_t first = p - int64_t(window) + 1 < 0 ? 0 : p - int64_t(window) + 1;
    const size_t raw_count = size_t(p - first + 1), total = raw_count + selections;
    const unsigned t = threadIdx.x, lane = t % 32, wave = t / 32, half = lane / 16;
    __shared__ float tops[H], denominators[H];
    __shared__ __attribute__((aligned(16))) _Float16 ps[2][H][L];
    __shared__ __attribute__((aligned(16))) _Float16 vt[2][64][L];
    extern __shared__ const float* keys[];
    for (size_t index = t; index < total; index += blockDim.x)
        keys[index] = attention_key(index, raw_count, first, raw, raw_rows, compressed, selected + row * selections, total, dim);
    if (t < H) {
        const size_t head = std::min<size_t>(t, heads - 1);
        const float2* parts = partial + (row * heads + head) * chunks;
        float top = sinks[head];
        for (size_t c = 0; c < chunks; ++c) top = fmaxf(top, parts[c].x);
        float denominator = expf(sinks[head] - top);
        for (size_t c = 0; c < chunks; ++c)
            if (parts[c].y > 0.0f) denominator += parts[c].y * expf(parts[c].x - top);
        tops[t] = top;
        denominators[t] = denominator;
    }
    __syncthreads();
    // A thread's weights of the stage (heads x 64 keys, PW of them) and one key's 32 columns.
    float pw[PW];
    float4 va[8];
    const unsigned vkey = t / 2, vc0 = (t % 2) * 32;
    auto load = [&](size_t k0) {
        #pragma unroll
        for (int j = 0; j < PW; ++j) {
            const unsigned at = t + j * 128, h = at / KS, k = at % KS;
            const size_t index = k0 + k;
            float w = 0.0f;
            if (h < heads && index < total) {
                const float score = scores[(row * heads + h) * stride + index];
                if (score != -INFINITY) w = expf(score - tops[h]);
            }
            pw[j] = w;
        }
        const size_t index = k0 + vkey;
        const float* key = index < total ? keys[index] : nullptr;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            va[j] = key ? reinterpret_cast<const float4*>(key + d0 + vc0)[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    auto store = [&](int buffer) {
        #pragma unroll
        for (int j = 0; j < PW; ++j) {
            const unsigned at = t + j * 128;
            ps[buffer][at / KS][at % KS] = _Float16(pw[j]);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            vt[buffer][vc0 + 4 * j][vkey] = _Float16(va[j].x);
            vt[buffer][vc0 + 4 * j + 1][vkey] = _Float16(va[j].y);
            vt[buffer][vc0 + 4 * j + 2][vkey] = _Float16(va[j].z);
            vt[buffer][vc0 + 4 * j + 3][vkey] = _Float16(va[j].w);
        }
    };
    moe_float8 acc[HM] = {};
    load(0);
    store(0);
    __syncthreads();
    for (size_t k0 = 0, step = 0; k0 < total; k0 += KS, ++step) {
        const int buffer = int(step & 1);
        const bool more = k0 + KS < total;
        if (more) load(k0 + KS);
        #pragma unroll
        for (int kk = 0; kk < KS; kk += 16) {
            const moe_half8 b = *reinterpret_cast<const moe_half8*>(&vt[buffer][wave * 16 + lane % 16][kk + 8 * half]);
            #pragma unroll
            for (int m = 0; m < HM; ++m) {
                const moe_half8 a = *reinterpret_cast<const moe_half8*>(&ps[buffer][m * 16 + lane % 16][kk + 8 * half]);
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[m]);
            }
        }
        if (more) store(buffer ^ 1);
        __syncthreads();
    }
    // acc[m][i]: head m * 16 + half * 8 + i, column d0 + 16 * wave + lane % 16; a rope pair's two
    // columns are neighbouring lanes.
    const size_t col = d0 + wave * 16 + lane % 16, rotated = dim - 2 * pairs;
    #pragma unroll
    for (int m = 0; m < HM; ++m)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned head = m * 16 + half * 8 + i;
            float o = acc[m][i] / denominators[head];
            const float other = __shfl_xor(o, 1, 32);
            if (col >= rotated) {
                const double2 g = inverse[row * pairs + (col - rotated) / 2];
                const bool even = (col - rotated) % 2 == 0;
                const double a = even ? o : other, b = even ? other : o;
                o = even ? float(a * g.x - b * g.y) : float(a * g.y + b * g.x);
            }
            if (head < heads) out[(row * heads + head) * dim + col] = o;
        }
#endif
}
#endif
