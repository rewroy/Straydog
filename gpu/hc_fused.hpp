// A prompt batch's HyperConnection up projection and its finish in one kernel (RDNA4, FP16 projections, a long read: Device::hc_fused).
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
#endif
