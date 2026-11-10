// Independently authored CUDA grouped expert GEMM for prefill (tensor cores, sm_70+): the
// counterpart of the RDNA4 kernels in moe_gemm.hpp with the same contract. Routed rows are
// grouped by expert, the host launches exactly the needed 64-row tiles, packed weights are
// decoded exactly to F32 and rounded to FP16 (exact for Q2_0's scale*(code-1)), activations
// are rounded to FP16 and products accumulate in F32, as in the explicit FP16 GEMM mode.
// CUDA subdivides Q2 tiles into 32-row blocks; the generic/IQ geometry remains
// 64 rows with K=32. Gate and up share a tile and form SiLU(gate)*up in F32.
// K panels remain ascending 16-value WMMA steps in both specializations.
constexpr int moe_block_m = 64;
constexpr int moe_cuda_m = 32;
constexpr int moe_cuda_k = 64;

// The input is reused by all ten routes and column tiles. Convert it once in
// the existing MoE workspace, rather than once per expert/column/K tile.
__global__ void moe_input_half(size_t values, const float* input, __half* output) {
    const size_t index = (size_t(blockIdx.x) * 256 + threadIdx.x) * 4;
    if (index >= values) return;   // values = steps * 2560, always divisible by four
    const float4 v = reinterpret_cast<const float4*>(input)[index / 4];
    reinterpret_cast<__half2*>(output)[index / 2] = __floats2half2_rn(v.x, v.y);
    reinterpret_cast<__half2*>(output)[index / 2 + 1] = __floats2half2_rn(v.z, v.w);
}

// Each lane transfers eight aligned half values. On Ampere+ this can proceed
// while that lane decodes its weights, without staging activations in registers.
__device__ __forceinline__ void moe_copy8(__half* dst, const __half* src, const __half* fallback) {
#if __CUDA_ARCH__ >= 800
    const unsigned address = unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(address), "l"(src ? src : fallback), "r"(src ? 16 : 0) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    uint4 v = {};
    if (src) v = *reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst) = v;
#endif
}
__device__ __forceinline__ void moe_copy_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

// Sixteen consecutive values from column k0 (a multiple of 16) of row `row`, as weight<kind> decodes them.
__device__ __forceinline__ void moe_values16(const Matrix& m, size_t row, size_t k0, float (&values)[16]) {
    const uint8_t* data = m.data + row * m.row_bytes;
    if (iq_kind(m.kind)) {
        const uint8_t* block = data + (k0 / 256) * iq_block_bytes(m.kind);
        const unsigned ib = unsigned(k0 % 256) / 32, l = unsigned(k0 % 32) / 8;
        float first[8], second[8];
        #define SD_IQ(format) case format: \
            iq_decode8<format>(block, ib, l, first); iq_decode8<format>(block, ib, l + 1, second); break;
        switch (m.kind) { SD_IQ(16) SD_IQ(17) SD_IQ(18) SD_IQ(19) SD_IQ(21) SD_IQ(22) SD_IQ(29) }
        #undef SD_IQ
        for (int i = 0; i < 8; ++i) { values[i] = first[i]; values[8 + i] = second[i]; }
        return;
    }
    #define SD_DECODE(format) case format: \
        for (int i = 0; i < 16; ++i) values[i] = weight<format>(data, k0 + i); break;
    switch (m.kind) {
        SD_FORMATS(SD_DECODE)
        default: for (int i = 0; i < 16; ++i) values[i] = 0;
    }
    #undef SD_DECODE
}

// moe_values16 for a format known when compiling (one of moe_cuda_mode's): the IQ formats decode
// their two eight-value groups as iq_decode8 does, IQ4_NL reads its block's scale once and takes the
// sixteen nibbles of one half of the block (k0 is a multiple of 16), and the rest go through
// weight<Format>. The values are the ones moe_values16 forms, without its switch on the kind for
// every sixteen.
template<int Format>
__device__ __forceinline__ void moe_values16_of(const uint8_t* data, size_t k0, float (&values)[16]) {
    if constexpr (Format == 16 || Format == 17 || Format == 18 || Format == 19 || Format == 21 || Format == 22 || Format == 29) {
        const uint8_t* block = data + (k0 / 256) * iq_block_bytes(Format);
        const unsigned ib = unsigned(k0 % 256) / 32, l = unsigned(k0 % 32) / 8;
        float first[8], second[8];
        iq_decode8<Format>(block, ib, l, first);
        iq_decode8<Format>(block, ib, l + 1, second);
        #pragma unroll
        for (int i = 0; i < 8; ++i) { values[i] = first[i]; values[8 + i] = second[i]; }
    } else if constexpr (Format == 20) {
        const uint8_t* b = data + (k0 / 32) * 18;
        const float d = half_at(b);
        const unsigned shift = unsigned(k0 % 32) / 16 * 4;
        #pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = d * nonlinear((b[2 + i] >> shift) & 15);
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = weight<Format>(data, k0 + i);
    }
}

// The kernels' decode for a weight format: 42 (Q2_0) its own path, a format with a compiled decode
// (moe_values16_of) itself, any other 0 (moe_values16's switch).
__host__ __device__ constexpr int moe_cuda_mode(int kind) {
    return kind == 42 || kind == 16 || kind == 17 || kind == 18 || kind == 19 || kind == 20 || kind == 21 || kind == 22
        || kind == 29 ? kind : 0;
}

// A Q2_0 quarter has one scale and four consecutive code bytes. Read it once,
// rather than reloading the scale and a code byte for each scalar weight. The
// two u16 reads are aligned even when an 18-byte block starts at address 2 mod 4.
// Keep the F32 multiply then FP16 rounding contract, including subnormal scales.
template<int Mode>
__device__ __forceinline__ void moe_store16(const Matrix& m, size_t row, size_t k0, __half* out) {
    uint32_t words[8];
    if constexpr (Mode == 42) {
        const uint8_t* block = m.data + row * m.row_bytes + (k0 / 64) * 18;
        const auto* codes = reinterpret_cast<const uint16_t*>(block + 2 + (k0 % 64) / 4);
        const uint32_t packed = uint32_t(codes[0]) | (uint32_t(codes[1]) << 16);
        const float scale = half_at(block);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float a = scale * float(int((packed >> (4 * i)) & 3) - 1);
            const float b = scale * float(int((packed >> (4 * i + 2)) & 3) - 1);
            union { __half2 h; uint32_t u; } pair = {__floats2half2_rn(a, b)};
            words[i] = pair.u;
        }
    } else {
        float values[16];
        if constexpr (Mode == 0) moe_values16(m, row, k0, values);
        else moe_values16_of<Mode>(m.data + row * m.row_bytes, k0, values);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            union { __half2 h; uint32_t u; } pair = {__floats2half2_rn(values[2 * i], values[2 * i + 1])};
            words[i] = pair.u;
        }
    }
    // Four adjacent half pairs per store. Scalar stores from the four lanes
    // decoding a row otherwise contend for the same banks across rows.
    reinterpret_cast<uint4*>(out)[0] = make_uint4(words[0], words[1], words[2], words[3]);
    reinterpret_cast<uint4*>(out)[1] = make_uint4(words[4], words[5], words[6], words[7]);
}

// One block: 32 (Q2) or 64 (generic) routed rows x 64 gate/up product columns. Mode: moe_cuda_mode of
// gate's and up's format (0 when they differ).
template<int Mode>
__global__ void __launch_bounds__(256) moe_gate_up_mma(const MoeGroup* groups, const MoeTile* tiles,
                                                       const int32_t* positions, size_t steps,
                                                       const __half* input, __half* product) {
    using namespace nvcuda;
    constexpr bool Q2 = Mode == 42;
    constexpr int M = Q2 ? moe_cuda_m : moe_block_m, LOAD_K = Q2 ? 64 : 32;
    constexpr int K = Q2 ? moe_cuda_k : 32, L = K + 8, WARPS_N = Q2 ? 4 : 2, FRAGMENTS_N = 4 / WARPS_N;
    const MoeTile base = tiles[blockIdx.y];
    const MoeTile tile = {base.group, base.first + blockIdx.z * M};
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    if (tile.first >= count) return;
    const int t = threadIdx.x, warp = t / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
    const size_t j0 = size_t(blockIdx.x) * 64;
    __shared__ SD_ALIGN(32) union {
        struct { __half x[M * L]; __half w[128 * L]; } in;
        float out[M * 68];
    } lds;
    const int arow = t / (LOAD_K / 8), acol = (t % (LOAD_K / 8)) * 8;
    const int brow = t / (LOAD_K / 16), bunit = t % (LOAD_K / 16);
    const __half* source = nullptr;
    if (tile.first + arow < count) {
        const size_t route = size_t(positions[first + tile.first + arow]);
        const size_t token = route < steps * 10 ? route / 10 : route - steps * 10;
        source = input + token * 2560 + acol;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> gate[FRAGMENTS_N], up[FRAGMENTS_N];
    for (int n = 0; n < FRAGMENTS_N; ++n) { wmma::fill_fragment(gate[n], 0.0f); wmma::fill_fragment(up[n], 0.0f); }
    for (size_t k = 0; k < 2560; k += K) {
        #pragma unroll
        for (int chunk = 0; chunk < K; chunk += LOAD_K) {
            moe_copy8(&lds.in.x[arow * L + chunk + acol], source ? source + k + chunk : nullptr, input);
            #pragma unroll
            for (int row = brow; row < 128; row += 256 / (LOAD_K / 16))
                moe_store16<Mode>(group.weights[row / 64], j0 + row % 64, k + chunk + bunit * 16,
                                &lds.in.w[row * L + chunk + bunit * 16]);
        }
        moe_copy_wait();
        __syncthreads();
        #pragma unroll
        for (int panel = 0; panel < K; panel += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x;
            wmma::load_matrix_sync(x, &lds.in.x[wm * 16 * L + panel], L);
            #pragma unroll
            for (int n = 0; n < FRAGMENTS_N; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bg, bu;
                wmma::load_matrix_sync(bg, &lds.in.w[((wn * FRAGMENTS_N + n) * 16) * L + panel], L);
                wmma::load_matrix_sync(bu, &lds.in.w[(64 + (wn * FRAGMENTS_N + n) * 16) * L + panel], L);
                wmma::mma_sync(gate[n], x, bg, gate[n]);
                wmma::mma_sync(up[n], x, bu, up[n]);
            }
        }
        __syncthreads();
    }
    // Same-type accumulator fragments map elements identically: form SiLU(gate)*up in place.
    #pragma unroll
    for (int n = 0; n < FRAGMENTS_N; ++n) {
        for (int i = 0; i < gate[n].num_elements; ++i) {
            const float g = gate[n].x[i];
            gate[n].x[i] = g * hc_sigmoid(g) * up[n].x[i];
        }
        wmma::store_matrix_sync(&lds.out[wm * 16 * 68 + (wn * FRAGMENTS_N + n) * 16], gate[n], 68, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = t; index < M * 64; index += 256) {
        const size_t row = tile.first + index / 64;
        if (row < count) product[(first + row) * 640 + j0 + index % 64] = __float2half(lds.out[(index / 64) * 68 + index % 64]);
    }
}

// One block: 32 (Q2) or 64 (generic) product rows x 128 columns; scatter to routes. Mode: moe_cuda_mode
// of down's format.
template<int Mode>
__global__ void __launch_bounds__(256) moe_down_mma(const MoeGroup* groups, const MoeTile* tiles,
                                                    const int32_t* positions, const __half* product, float* outputs) {
    using namespace nvcuda;
    constexpr bool Q2 = Mode == 42;
    constexpr int M = Q2 ? moe_cuda_m : moe_block_m, LOAD_K = Q2 ? 64 : 32;
    constexpr int K = Q2 ? moe_cuda_k : 32, L = K + 8, WARPS_N = Q2 ? 4 : 2, FRAGMENTS_N = 8 / WARPS_N;
    const MoeTile base = tiles[blockIdx.y];
    const MoeTile tile = {base.group, base.first + blockIdx.z * M};
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    if (tile.first >= count) return;
    const Matrix down = group.weights[2];
    const int t = threadIdx.x, warp = t / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
    const size_t n0 = size_t(blockIdx.x) * 128;
    __shared__ SD_ALIGN(32) union {
        struct { __half x[M * L]; __half w[128 * L]; } in;
        float out[M * 132];
    } lds;
    const int arow = t / (LOAD_K / 8), acol = (t % (LOAD_K / 8)) * 8;
    const int brow = t / (LOAD_K / 16), bunit = t % (LOAD_K / 16);
    const __half* source = tile.first + arow < count ? product + (first + tile.first + arow) * 640 + acol : nullptr;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum[FRAGMENTS_N];
    for (int n = 0; n < FRAGMENTS_N; ++n) wmma::fill_fragment(sum[n], 0.0f);
    for (size_t k = 0; k < 640; k += K) {
        #pragma unroll
        for (int chunk = 0; chunk < K; chunk += LOAD_K) {
            moe_copy8(&lds.in.x[arow * L + chunk + acol], source ? source + k + chunk : nullptr, product);
            #pragma unroll
            for (int row = brow; row < 128; row += 256 / (LOAD_K / 16))
                moe_store16<Mode>(down, n0 + row, k + chunk + bunit * 16, &lds.in.w[row * L + chunk + bunit * 16]);
        }
        moe_copy_wait();
        __syncthreads();
        #pragma unroll
        for (int panel = 0; panel < K; panel += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x;
            wmma::load_matrix_sync(x, &lds.in.x[wm * 16 * L + panel], L);
            #pragma unroll
            for (int n = 0; n < FRAGMENTS_N; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w;
                wmma::load_matrix_sync(w, &lds.in.w[((wn * FRAGMENTS_N + n) * 16) * L + panel], L);
                wmma::mma_sync(sum[n], x, w, sum[n]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int n = 0; n < FRAGMENTS_N; ++n)
        wmma::store_matrix_sync(&lds.out[wm * 16 * 132 + (wn * FRAGMENTS_N + n) * 16], sum[n], 132, wmma::mem_row_major);
    __syncthreads();
    for (int index = t; index < M * 128; index += 256) {
        const size_t row = tile.first + index / 128;
        if (row < count) outputs[size_t(positions[first + row]) * 2560 + n0 + index % 128] = lds.out[(index / 128) * 132 + index % 128];
    }
}
