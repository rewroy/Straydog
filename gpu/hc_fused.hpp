// A prompt batch's HyperConnection up projection and its finish in one kernel (FP16 projections: RDNA4 on a long read, Device::hc_fused;
// CUDA from sm_80, hc_up_finish_mma below, unless STRAYDOG_HC_FUSED=0).
// hc_prefill's pair was hipblasGemmEx for the 10240 x 320 projection, which wrote its F32 rows (168 MB for a part's 4096 rows), and
// hc_finish_hidden, which read them back with the rows' hidden values (another 168 MB) and wrote the mixed rows (42 MB): a kernel bound
// by memory for the most part. Here the four streams' sums of a column meet in registers: only the hidden rows are read and the mixed rows
// written. Both kernels are 128 rows of the projection (32 columns of each of the four streams) by 128 tokens a block.
//
// The operands are the pair's: the FP16 weights as expanded (Device::expanded) and the FP16 low-rank rows (hc_prefill's half buffer); the
// products are summed on the matrix cores in the order of an unfused WMMA kernel's (K in steps of 16 from a zero accumulator, ascending),
// which is not rocBLAS's, so the F32 projection differs from the pair's in its last bits (the mixed rows against an F64 evaluation: rms
// 9.5e-8 and max 2.9e-6 of a row's scale for this kernel, 9.6e-8 and 3.2e-6 for the pair, README, Numerics). Everything after the product is
// hc_finish_hidden's expressions in its order, so the mixed rows are, bit for bit, hc_finish_hidden's of this order's projection
// (tests/hc_fused_test.cpp checks that). Needs hc_sigmoid (hc_decode.hpp).
#pragma once

#if defined(__HIPCC__)
typedef _Float16 hcf_half8 __attribute__((ext_vector_type(8)));
typedef float hcf_float8 __attribute__((ext_vector_type(8)));
typedef float hcf_float4 __attribute__((ext_vector_type(4)));
typedef unsigned hcf_word4 __attribute__((ext_vector_type(4)));

constexpr int hcf_k = 320, hcf_columns = 2560, hcf_block_columns = 32, hcf_block_tokens = 128, hcf_step = 64;

// A block: 256 threads, eight waves of two column halves (wm: 16 of the block's 32 columns, in every stream's fragment) by four token
// quarters (wn: 32 tokens). In a fragment of the accumulator the row 8h + 4q + j (lane half h, run q, j < 4) is the column 8q + 4h + j
// of the wave's sixteen, so a lane holds, for a token, its eight columns as two runs of four (one 16-byte access each) in each of the four
// streams (the fragments). W and X go through LDS, K = 320 in five steps of 64 with one buffer.
__global__ void __launch_bounds__(256) hc_up_finish(const __half* __restrict__ weights, const __half* __restrict__ rows, const float* __restrict__ input,
                                                    const float* __restrict__ norm, const float* __restrict__ denominators,
                                                    const float* __restrict__ injection, int steps, float* __restrict__ output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int T = 256, BM = 128, BN = hcf_block_tokens, FM = 4, FN = 2, LDK = hcf_step + 8, PPR = hcf_step / 8;
    constexpr int PA = BM * PPR / T, PB = BN * PPR / T, STEPS = hcf_k / hcf_step;
    __shared__ __attribute__((aligned(16))) _Float16 lds[(BM + BN) * LDK];
    const int t = threadIdx.x, lane = t & 31, wave = __builtin_amdgcn_readfirstlane(t >> 5);
    const int wm = wave / 4, wn = wave % 4, h = lane >> 4, ln = lane & 15;
    const int column_block = blockIdx.x % (hcf_columns / hcf_block_columns), token_block = blockIdx.x / (hcf_columns / hcf_block_columns);
    const int c0 = column_block * hcf_block_columns, n0 = token_block * BN;
    const auto* w = reinterpret_cast<const _Float16*>(weights);
    const auto* x = reinterpret_cast<const _Float16*>(rows);
    const hcf_word4* ga[PA]; const hcf_word4* gb[PB];
    #pragma unroll
    for (int j = 0; j < PA; ++j) {
        const int p = t + j * T, row = p / PPR, cp = p % PPR;
        // (LDS row = wave half, stream, accumulator row of the fragment; the weights' row = stream, column)
        const int r = row & 15, stream = (row >> 4) & 3, half_of = row >> 6;
        const int column = c0 + half_of * 16 + ((r >> 2) & 1) * 8 + (r >> 3) * 4 + (r & 3);
        ga[j] = reinterpret_cast<const hcf_word4*>(w + size_t(stream * hcf_columns + column) * hcf_k + cp * 8);
    }
    #pragma unroll
    for (int j = 0; j < PB; ++j) {
        const int p = t + j * T, row = p / PPR, cp = p % PPR;
        gb[j] = reinterpret_cast<const hcf_word4*>(x + size_t(min(n0 + row, steps - 1)) * hcf_k + cp * 8);
    }
    hcf_word4 ra[PA], rb[PB];
    auto load = [&](int step) {
        #pragma unroll
        for (int j = 0; j < PA; ++j) ra[j] = ga[j][step * PPR];
        #pragma unroll
        for (int j = 0; j < PB; ++j) rb[j] = gb[j][step * PPR];
    };
    auto store = [&]() {
        #pragma unroll
        for (int j = 0; j < PA; ++j) {
            const int p = t + j * T, row = p / PPR, cp = p % PPR;
            *reinterpret_cast<hcf_word4*>(lds + row * LDK + cp * 8) = ra[j];
        }
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            const int p = t + j * T, row = p / PPR, cp = p % PPR;
            *reinterpret_cast<hcf_word4*>(lds + (BM + row) * LDK + cp * 8) = rb[j];
        }
    };
    hcf_float8 acc[FM][FN];
    #pragma unroll
    for (int i = 0; i < FM; ++i)
        #pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = hcf_float8{0, 0, 0, 0, 0, 0, 0, 0};
    auto compute = [&]() {
        #pragma unroll
        for (int kk = 0; kk < hcf_step / 16; ++kk) {
            hcf_half8 a[FM], b[FN];
            #pragma unroll
            for (int i = 0; i < FM; ++i) a[i] = *reinterpret_cast<const hcf_half8*>(lds + (wm * 64 + i * 16 + ln) * LDK + kk * 16 + 8 * h);
            #pragma unroll
            for (int j = 0; j < FN; ++j) b[j] = *reinterpret_cast<const hcf_half8*>(lds + (BM + wn * 32 + j * 16 + ln) * LDK + kk * 16 + 8 * h);
            #pragma unroll
            for (int i = 0; i < FM; ++i)
                #pragma unroll
                for (int j = 0; j < FN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    };
    load(0);
    for (int s = 0; s + 1 < STEPS; ++s) {
        store();
        __syncthreads();
        load(s + 1);
        compute();
        __syncthreads();
    }
    store();
    __syncthreads();
    compute();
    // hc_finish_hidden's arithmetic, expression for expression: per column the four streams' sigmoid(up) times the stream's normalized value
    const int column0 = c0 + wm * 16 + 4 * h;
    hcf_float4 weight[4][2];
    #pragma unroll
    for (int s = 0; s < 4; ++s)
        #pragma unroll
        for (int q = 0; q < 2; ++q) weight[s][q] = *reinterpret_cast<const hcf_float4*>(norm + s * hcf_columns + column0 + 8 * q);
    #pragma unroll
    for (int fn = 0; fn < FN; ++fn) {
        const int token = n0 + wn * 32 + fn * 16 + ln;
        if (token >= steps) continue;
        hcf_float4 value[4][2];
        #pragma unroll
        for (int s = 0; s < 4; ++s)
            #pragma unroll
            for (int q = 0; q < 2; ++q)
                value[s][q] = *reinterpret_cast<const hcf_float4*>(input + size_t(token) * (4 * hcf_columns) + s * hcf_columns + column0 + 8 * q);
        const hcf_float4 d4 = *reinterpret_cast<const hcf_float4*>(denominators + size_t(token) * 4);
        const float denominator[4] = {d4[0], d4[1], d4[2], d4[3]};
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            float out[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sum = 0;
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float normalized = value[s][q][j] / denominator[s] * weight[s][q][j];
                    sum += hc_sigmoid(acc[s][fn][4 * q + j]) * normalized;
                }
                out[j] = sum / 4.0f;
            }
            *reinterpret_cast<hcf_float4*>(output + size_t(token) * hcf_columns + column0 + 8 * q) = hcf_float4{out[0], out[1], out[2], out[3]};
        }
    }
    // the rows' four injections, after the mixed rows (the first column block writes the block's tokens')
    if (column_block == 0) {
        #pragma unroll
        for (int it = 0; it < BN * 4 / T; ++it) {
            const int index = t + it * T, token = n0 + index / 4, col = index % 4;
            if (token < steps) output[size_t(steps) * hcf_columns + size_t(token) * 4 + col] = 2.0f * hc_sigmoid(injection[size_t(token) * 4 + col] / 4.0f);
        }
    }
#endif
}

// The up projection and finish of `steps` rows: `input` the rows' hidden values (10240 floats a row), `denominators` four a row (hc_normalize_half's),
// `injection` four a row; `output` the mixed rows of 2560 and then each row's four injections.
void hc_up_finish_launch(Device& d, const __half* weights, const __half* rows, const float* input, const float* norm, const float* denominators,
                         const float* injection, size_t steps, float* output) {
    const unsigned blocks = unsigned((hcf_columns / hcf_block_columns) * ((steps + hcf_block_tokens - 1) / hcf_block_tokens));
    hc_up_finish<<<blocks, 256, 0, d.stream>>>(weights, rows, input, norm, denominators, injection, int(steps), output);
    check(hipGetLastError());
}
#else
// CUDA: the same fusion on mma.sync m16n8k16 (FP16 operands, F32 accumulation), hc_up_finish's block and expressions: a block is 128 projection
// rows (32 columns of each of the four streams) by 128 tokens, 8 warps (two column halves by four token quarters), K = 320 in five steps of 64
// through shared memory (ldmatrix from rows of 72 halves). A warp's four m-fragments are the four streams of its sixteen columns, so a lane's
// accumulators hold, for a column and a token, the four streams' sums, and the finish uses them where they are. The products are summed K in
// steps of 16 from zero, ascending, which is the order cuBLAS's FP16 GEMM sums this projection in on an RTX 5070 Ti: the mixed rows are then
// the pair's bit for bit (one layer, 4096 rows: the same output digest); where cuBLAS sums in another order they differ in the projection's
// last bits only, as RDNA4's do. Staged: after the products the block's sigmoid(sums) go through shared memory (the tile's, in two halves of
// 64 tokens, [token][stream][32 columns + 4]) so that a thread finishes four consecutive columns of a token with 16-byte reads and writes; a
// finish straight from the accumulators wrote four bytes at a time and took 1.36 ms. One layer, 4096 rows, RTX 5070 Ti: cublasGemmEx 0.338 ms +
// hc_finish_hidden 0.491 ms, the pair 0.845 ms, this kernel 0.483 ms with the pair's output digest; in a 32K read hc_up 0.31 s and hc_finish 0.45 s
// of 23.2 s.
constexpr int hcm_k = 320, hcm_columns = 2560, hcm_block_columns = 32, hcm_block_tokens = 128, hcm_step = 64, hcm_ldk = hcm_step + 8;
__device__ __forceinline__ void hcm_ldmatrix4(uint32_t (&r)[4], const void* address) {
    const unsigned a = unsigned(__cvta_generic_to_shared(address));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void hcm_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800   // (sm_75 has no m16n8k16: hc_fused_enabled does not take this kernel there)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
// Shared row `row` of the A tile (0..127) is the projection row stream * 2560 + c0 + half * 16 + r for row = half * 64 + stream * 16 + r,
// so a warp's four m-fragments are the four streams of its sixteen columns, and a lane's accumulators hold, for a column and a token, the
// four streams' sums: the finish happens in registers.
__global__ void __launch_bounds__(256, 2) hc_up_finish_mma(const __half* __restrict__ weights, const __half* __restrict__ rows, const float* __restrict__ input,
                                                        const float* __restrict__ norm, const float* __restrict__ denominators,
                                                        const float* __restrict__ injection, int steps, float* __restrict__ output) {
    constexpr int T = 256, BM = 128, BN = hcm_block_tokens, PPR = hcm_step / 8, PA = BM * PPR / T, PB = BN * PPR / T, STEPS = hcm_k / hcm_step;
    // (after the products the tile's memory holds the block's sigmoid(sums) as [token][stream][32 columns + 4], in two halves of 64 tokens)
    extern __shared__ __align__(16) unsigned char hcm_shared[];
    __half* tile = reinterpret_cast<__half*>(hcm_shared);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, wm = warp / 4, wn = warp % 4, gid = lane >> 2, tig = lane & 3;
    const int column_block = blockIdx.x % (hcm_columns / hcm_block_columns), token_block = blockIdx.x / (hcm_columns / hcm_block_columns);
    const int c0 = column_block * hcm_block_columns, n0 = token_block * BN;
    const uint4* ga[PA]; const uint4* gb[PB];
    #pragma unroll
    for (int j = 0; j < PA; ++j) {
        const int p = t + j * T, row = p / PPR, cp = p % PPR;
        const int r = row & 15, stream = (row >> 4) & 3, half_of = row >> 6;
        ga[j] = reinterpret_cast<const uint4*>(weights + size_t(stream * hcm_columns + c0 + half_of * 16 + r) * hcm_k + cp * 8);
    }
    #pragma unroll
    for (int j = 0; j < PB; ++j) {
        const int p = t + j * T, row = p / PPR, cp = p % PPR;
        gb[j] = reinterpret_cast<const uint4*>(rows + size_t(min(n0 + row, steps - 1)) * hcm_k + cp * 8);
    }
    uint4 ra[PA], rb[PB];
    auto load = [&](int step) {
        #pragma unroll
        for (int j = 0; j < PA; ++j) ra[j] = ga[j][step * PPR];
        #pragma unroll
        for (int j = 0; j < PB; ++j) rb[j] = gb[j][step * PPR];
    };
    auto store = [&]() {
        #pragma unroll
        for (int j = 0; j < PA; ++j) { const int p = t + j * T; *reinterpret_cast<uint4*>(tile + (p / PPR) * hcm_ldk + (p % PPR) * 8) = ra[j]; }
        #pragma unroll
        for (int j = 0; j < PB; ++j) { const int p = t + j * T; *reinterpret_cast<uint4*>(tile + (BM + p / PPR) * hcm_ldk + (p % PPR) * 8) = rb[j]; }
    };
    float acc[4][4][4];   // [stream][token fragment of 8][c0..c3]
    #pragma unroll
    for (int s = 0; s < 4; ++s)
        #pragma unroll
        for (int f = 0; f < 4; ++f) acc[s][f][0] = acc[s][f][1] = acc[s][f][2] = acc[s][f][3] = 0.0f;
    auto compute = [&]() {
        #pragma unroll
        for (int kk = 0; kk < hcm_step / 16; ++kk) {
            uint32_t a[4][4], b[2][4];
            #pragma unroll
            for (int s = 0; s < 4; ++s)   // rows wm * 64 + s * 16 + (lane % 16), columns kk * 16 + 8 * (lane / 16)
                hcm_ldmatrix4(a[s], tile + (wm * 64 + s * 16 + (lane & 15)) * hcm_ldk + kk * 16 + 8 * (lane >> 4));
            #pragma unroll
            for (int f2 = 0; f2 < 2; ++f2)   // tokens wn * 32 + f2 * 16 + (lane & 7) + 8 * (lane >> 4), k halves by (lane >> 3) & 1
                hcm_ldmatrix4(b[f2], tile + (BM + wn * 32 + f2 * 16 + (lane & 7) + 8 * (lane >> 4)) * hcm_ldk + kk * 16 + 8 * ((lane >> 3) & 1));
            #pragma unroll
            for (int s = 0; s < 4; ++s)
                #pragma unroll
                for (int f = 0; f < 4; ++f) hcm_mma(acc[s][f], a[s], b[f / 2][(f & 1) * 2], b[f / 2][(f & 1) * 2 + 1]);
        }
    };
    load(0);
    for (int s = 0; s + 1 < STEPS; ++s) {
        store();
        __syncthreads();
        load(s + 1);
        compute();
        __syncthreads();
    }
    store();
    __syncthreads();
    compute();
    float* sums = reinterpret_cast<float*>(hcm_shared);
    // in two halves of 64 tokens (the four warps of a half's tokens write theirs, then all read), so 33.8 KB hold them
    #pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        __syncthreads();   // (the tile's last reads, or the half before's, are done)
        if (wn / 2 == half) {
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    #pragma unroll
                    for (int f = 0; f < 4; ++f)
                        #pragma unroll
                        for (int j = 0; j < 2; ++j)
                            sums[((wn % 2) * 32 + f * 8 + 2 * tig + j) * 132 + s * 32 + wm * 16 + gid + 8 * rr] = hc_sigmoid(acc[s][f][2 * rr + j]);
        }
        __syncthreads();
        // hc_finish_hidden's arithmetic, four columns of a token a thread (16-byte reads and writes)
        #pragma unroll
        for (int it = 0; it < 64 * 8 / T; ++it) {
            const int index = t + it * T, local = index / 8, cq = index % 8, token = n0 + half * 64 + local, column = c0 + 4 * cq;
            if (token >= steps) continue;
            const float4 d4 = *reinterpret_cast<const float4*>(denominators + size_t(token) * 4);
            const float denominator[4] = {d4.x, d4.y, d4.z, d4.w};
            float sum[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float4 in = *reinterpret_cast<const float4*>(input + size_t(token) * 10240 + s * hcm_columns + column);
                const float4 nw = *reinterpret_cast<const float4*>(norm + s * hcm_columns + column);
                const float4 up = *reinterpret_cast<const float4*>(sums + local * 132 + s * 32 + 4 * cq);
                sum[0] += up.x * (in.x / denominator[s] * nw.x); sum[1] += up.y * (in.y / denominator[s] * nw.y);
                sum[2] += up.z * (in.z / denominator[s] * nw.z); sum[3] += up.w * (in.w / denominator[s] * nw.w);
            }
            *reinterpret_cast<float4*>(output + size_t(token) * hcm_columns + column) = make_float4(sum[0] / 4.0f, sum[1] / 4.0f, sum[2] / 4.0f, sum[3] / 4.0f);
        }
    }
    if (column_block == 0)
        for (int index = t; index < BN * 4; index += T) {
            const int token = n0 + index / 4, col = index % 4;
            if (token < steps) output[size_t(steps) * hcm_columns + size_t(token) * 4 + col] = 2.0f * hc_sigmoid(injection[size_t(token) * 4 + col] / 4.0f);
        }
}
// The up projection and finish of `steps` rows, as the RDNA4 kernel's launch: `weights` the FP16 expanded matrix, `rows` the FP16 low-rank rows,
// `input` the rows' hidden values (10240 floats a row), `denominators` four a row, `injection` four a row; `output` the mixed rows of 2560 and
// then each row's four injections. The dynamic shared memory is 36864 bytes: under 48 KB, so no attribute is needed.
constexpr size_t hcm_tile_bytes = (128 + 128) * hcm_ldk * 2, hcm_staged_bytes = 64 * 132 * 4, hcm_shared_bytes = hcm_tile_bytes > hcm_staged_bytes ? hcm_tile_bytes : hcm_staged_bytes;
static_assert(hcm_shared_bytes == 36864 && hcm_shared_bytes <= 48 * 1024, "the tile and the staged sums share the dynamic memory, under the default limit");
void hc_up_finish_launch(Device& d, const __half* weights, const __half* rows, const float* input, const float* norm, const float* denominators,
                         const float* injection, size_t steps, float* output) {
    const unsigned blocks = unsigned((hcm_columns / hcm_block_columns) * ((steps + hcm_block_tokens - 1) / hcm_block_tokens));
    hc_up_finish_mma<<<blocks, 256, hcm_shared_bytes, d.stream>>>(weights, rows, input, norm, denominators, injection, int(steps), output);
    check(hipGetLastError());
}
#endif
// Whether hc_prefill takes the fused kernel: RDNA4 on a long read; CUDA (sm_80+) always unless STRAYDOG_HC_FUSED=0 (development; read at the first
// batch of a Device, so a process that loads a model again after changing it gets the new value).
inline bool hc_fused_enabled(Device& d) {
#if defined(__HIPCC__)
    return d.hc_fused && d.rdna4;
#else
    if (d.hc_fused_mode < 0) { const char* mode = std::getenv("STRAYDOG_HC_FUSED"); d.hc_fused_mode = mode && std::string(mode) == "0" ? 0 : 1; }
    return d.hc_fused_mode && d.mma_f16;
#endif
}
