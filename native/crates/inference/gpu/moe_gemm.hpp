// Independently authored RDNA4 grouped expert GEMM for prefill. Routed rows are
// grouped by expert on the device, the host reads the per-expert counts once and
// launches exactly the needed 64-row tiles. Packed weights are decoded into LDS
// as FP16 (Q2_0 values scale*(code-1) are exact in FP16); activations are rounded
// to FP16 and products accumulate in F32, as in the explicit FP16 GEMM mode.
// Gate and up share each tile so SiLU(gate)*up is formed before leaving the block.
#if defined(__HIPCC__)
typedef _Float16 moe_half2 __attribute__((ext_vector_type(2)));
typedef _Float16 moe_half8 __attribute__((ext_vector_type(8)));
typedef float moe_float8 __attribute__((ext_vector_type(8)));
constexpr int moe_block_k = 32, moe_lds_k = moe_block_k + 8, moe_block_m = 64;

// Within each 16-value k panel the four quads are stored as [0,2,1,3], so a
// lane's WMMA fragment (quads q and q+2) is one contiguous 16-byte LDS read.
__device__ __forceinline__ int moe_panel(int k) {
    const int quad = (k >> 2) & 3;
    return (k & ~15) | ((quad == 1 ? 2 : quad == 2 ? 1 : quad) << 2) | (k & 3);
}
__device__ __forceinline__ uint32_t moe_pack(float a, float b) {
    const moe_half2 v = {_Float16(a), _Float16(b)};
    return __builtin_bit_cast(uint32_t, v);
}
// Groups l and l + 1 (l is 0 or 2) of 32-value block ib of an IQ2/IQ3 super-block, read with
// 16-bit header loads and one load per grid row as the GEMV units do. Every value is formed
// exactly as iq_decode8 forms it (scale, grid byte and sign in the same order).
template<int kind>
__device__ __forceinline__ void moe_iq16(const uint8_t* block, unsigned ib, unsigned l, float (&values)[16]) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(block);
    const float d = gemv_half(h[0]);
    #pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        const unsigned g = l + half;
        float db;
        unsigned signs;
        uint32_t low, high;   // the group's eight grid bytes
        if constexpr (kind == 21) {   // IQ3_S
            const unsigned top = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
            const unsigned scale = ((h[53 + ib / 4] >> (8 * ((ib / 2) & 1))) >> (4 * (ib & 1))) & 15;
            db = d * float(1 + 2 * scale);
            signs = (iq_pair(h, 37 + 2 * ib) >> (8 * g)) & 255;
            const unsigned pair = h[1 + 4 * ib + g];
            const uint32_t* grid = reinterpret_cast<const uint32_t*>(iq_grid3s);
            low = grid[(pair & 255) | ((top << (8 - 2 * g)) & 256)];
            high = grid[((pair >> 8) & 255) | ((top << (7 - 2 * g)) & 256)];
        } else if constexpr (kind == 18) {   // IQ3_XXS
            const uint32_t aux = iq_pair(h, 33 + 2 * ib);
            db = d * (0.5f + float(aux >> 28)) * 0.5f;
            signs = iq_signs((aux >> (7 * g)) & 127);
            const unsigned pair = h[1 + 4 * ib + g];
            const uint32_t* grid = reinterpret_cast<const uint32_t*>(iq_grid3xxs);
            low = grid[pair & 255]; high = grid[pair >> 8];
        } else if constexpr (kind == 16) {   // IQ2_XXS
            const uint32_t aux = iq_pair(h, 3 + 4 * ib);
            db = d * (0.5f + float(aux >> 28)) * 0.25f;
            signs = iq_signs((aux >> (7 * g)) & 127);
            const uint2 row = reinterpret_cast<const uint2*>(iq_grid2xxs)[(iq_pair(h, 1 + 4 * ib) >> (8 * g)) & 255];
            low = row.x; high = row.y;
        } else if constexpr (kind == 17) {   // IQ2_XS
            const unsigned q = h[1 + 4 * ib + g];
            db = d * (0.5f + float((((h[33 + ib / 2] >> (8 * (ib & 1))) & 255) >> (4 * (g / 2))) & 15)) * 0.25f;
            signs = iq_signs(q >> 9);
            const uint2 row = reinterpret_cast<const uint2*>(iq_grid2xs)[q & 511];
            low = row.x; high = row.y;
        } else {   // IQ2_S
            const unsigned top = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
            const unsigned index = ((iq_pair(h, 1 + 2 * ib) >> (8 * g)) & 255) | ((top << (8 - 2 * g)) & 0x300);
            db = d * (0.5f + float((((h[37 + ib / 2] >> (8 * (ib & 1))) & 255) >> (4 * (g / 2))) & 15)) * 0.25f;
            signs = (iq_pair(h, 17 + 2 * ib) >> (8 * g)) & 255;
            const uint2 row = reinterpret_cast<const uint2*>(iq_grid2s)[index];
            low = row.x; high = row.y;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t byte = ((j < 4 ? low : high) >> (8 * (j % 4))) & 255u;
            values[8 * half + j] = db * float(byte) * ((signs >> j) & 1 ? -1.0f : 1.0f);
        }
    }
}

// Sixteen consecutive values starting at column k0 (a multiple of 16), as eight
// dwords of FP16 pairs in natural order.
__device__ __forceinline__ void moe_decode16(const Matrix& m, size_t row, size_t k0, uint32_t (&out)[8]) {
    const uint8_t* data = m.data + row * m.row_bytes;
    if (m.kind == 42) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(data + (k0 / 64) * 18);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const bool odd = (address & 2) != 0;
        const size_t quarter = (k0 % 64) / 16;
        const uint32_t first = base[0], low = base[quarter], high = base[quarter + 1];
        const uint32_t bits = odd ? first >> 16 : first & 0xffff;
        const moe_half2 scale = __builtin_bit_cast(moe_half2, bits | (bits << 16));
        const uint32_t codes = odd ? high : __builtin_amdgcn_alignbyte(high, low, 2);
        // FP16 high bytes of (code-1): -1 0xBC, 0 0x00, 1 0x3C, 2 0x40; the low bytes are zero.
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (codes >> (8 * b)) & 255u;
            const uint32_t spread = (byte & 3u) | ((byte & 12u) << 6) | ((byte & 48u) << 12) | ((byte & 192u) << 18);
            const uint32_t high_bytes = __builtin_amdgcn_perm(0u, 0x403C00BCu, spread);
            const uint32_t pair0 = __builtin_amdgcn_perm(high_bytes, 0u, 0x050C040Cu);
            const uint32_t pair1 = __builtin_amdgcn_perm(high_bytes, 0u, 0x070C060Cu);
            out[2 * b] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(moe_half2, pair0) * scale);
            out[2 * b + 1] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(moe_half2, pair1) * scale);
        }
        return;
    }
    float values[16];
    if (iq_kind(m.kind)) {
        // Two eight-value groups of one 32-value block (IQ super-blocks of 256 values).
        const uint8_t* block = data + (k0 / 256) * iq_block_bytes(m.kind);
        const unsigned ib = unsigned(k0 % 256) / 32, l = unsigned(k0 % 32) / 8;
        if (m.kind == 29) {
            float first[8], second[8];
            iq_decode8<29>(block, ib, l, first); iq_decode8<29>(block, ib, l + 1, second);
            for (int i = 0; i < 8; ++i) { values[i] = first[i]; values[8 + i] = second[i]; }
        } else {
            #define SD_IQ(format) case format: moe_iq16<format>(block, ib, l, values); break;
            switch (m.kind) { SD_IQ(16) SD_IQ(17) SD_IQ(18) SD_IQ(21) SD_IQ(22) }
            #undef SD_IQ
        }
    } else if (m.kind == 20 || m.kind == 8) {
        // Half of a 32-value IQ4_NL / Q8_0 block, its 16 code bytes read as 16-bit pairs. IQ4_NL
        // codes go through the GEMV byte table (value + 128): float(byte) - 128 is the table's
        // integer exactly, so each value equals d * float(nonlinear(code)).
        const uint8_t* block = data + (k0 / 32) * (m.kind == 20 ? 18 : 34);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(block);
        const float d = gemv_half(h[0]);
        const unsigned half = unsigned(k0 % 32) / 16;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned first = m.kind == 20 ? 1 + 2 * w : 1 + 8 * half + 2 * w;
            const uint32_t word = uint32_t(h[first]) | uint32_t(h[first + 1]) << 16;
            const uint32_t bytes = m.kind == 20 ? gemv_iq4((word >> (4 * half)) & 0x0F0F0F0Fu) : word;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t byte = (bytes >> (8 * j)) & 255u;
                values[4 * w + j] = m.kind == 20 ? d * (float(byte) - 128.0f) : d * float(int8_t(uint8_t(byte)));
            }
        }
    } else {
        #define SD_DECODE(format) case format: \
            for (int i = 0; i < 16; ++i) values[i] = weight<format>(data, k0 + i); break;
        switch (m.kind) {
            SD_FORMATS(SD_DECODE)
            default: for (int i = 0; i < 16; ++i) values[i] = 0;
        }
        #undef SD_DECODE
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = moe_pack(values[2 * i], values[2 * i + 1]);
}

// One block: 64 routed rows of one expert x 64 product columns (gate and up rows).
__global__ void __launch_bounds__(256) moe_gate_up_wmma(const MoeGroup* groups, const MoeTile* tiles,
                                                        const int32_t* positions, size_t steps,
                                                        const float* input, _Float16* product) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int L = moe_lds_k;
    const MoeTile tile = tiles[blockIdx.y];
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    const Matrix gate = group.weights[0], up = group.weights[1];
    const int t = threadIdx.x, wave = t / 32, lane = t % 32, wm = wave / 4, wn = wave % 4;
    const size_t j0 = size_t(blockIdx.x) * 64;
    __shared__ __attribute__((aligned(16))) _Float16 as[2][moe_block_m * L];
    __shared__ __attribute__((aligned(16))) _Float16 bs[2][128 * L];
    const int arow = t / 4, acol = (t % 4) * 8;
    const float* source = nullptr;
    if (tile.first + arow < count) {
        const size_t route = size_t(positions[first + tile.first + arow]);
        const size_t token = route < steps * 10 ? route / 10 : route - steps * 10;
        source = input + token * 2560 + acol;
    }
    const int brow = t / 2, bunit = t % 2;
    const Matrix& weights = brow < 64 ? gate : up;
    const size_t wrow = j0 + size_t(brow & 63);
    float a[8]; uint32_t b[8];
    auto load = [&](size_t k) {
        if (source) {
            const float4 x = *reinterpret_cast<const float4*>(source + k), y = *reinterpret_cast<const float4*>(source + k + 4);
            a[0] = x.x; a[1] = x.y; a[2] = x.z; a[3] = x.w; a[4] = y.x; a[5] = y.y; a[6] = y.z; a[7] = y.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = 0;
        }
        moe_decode16(weights, wrow, k + bunit * 16, b);
    };
    auto store = [&](int buffer) {
        *reinterpret_cast<uint2*>(&as[buffer][arow * L + moe_panel(acol)]) = uint2{moe_pack(a[0], a[1]), moe_pack(a[2], a[3])};
        *reinterpret_cast<uint2*>(&as[buffer][arow * L + moe_panel(acol + 4)]) = uint2{moe_pack(a[4], a[5]), moe_pack(a[6], a[7])};
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            *reinterpret_cast<uint2*>(&bs[buffer][brow * L + moe_panel(bunit * 16 + q * 4)]) = uint2{b[2 * q], b[2 * q + 1]};
    };
    moe_float8 gate_sum[2] = {}, up_sum[2] = {};
    load(0); store(0); __syncthreads();
    for (size_t k = 0; k < 2560; k += moe_block_k) {
        const int buffer = int(k / moe_block_k) & 1;
        if (k + moe_block_k < 2560) load(k + moe_block_k);
        #pragma unroll
        for (int panel = 0; panel < moe_block_k; panel += 16) {
            const int offset = panel + (lane / 16) * 8;
            const moe_half8 bg = *reinterpret_cast<const moe_half8*>(&bs[buffer][(wn * 16 + lane % 16) * L + offset]);
            const moe_half8 bu = *reinterpret_cast<const moe_half8*>(&bs[buffer][(64 + wn * 16 + lane % 16) * L + offset]);
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const moe_half8 x = *reinterpret_cast<const moe_half8*>(&as[buffer][(wm * 32 + m * 16 + lane % 16) * L + offset]);
                gate_sum[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(x, bg, gate_sum[m]);
                up_sum[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(x, bu, up_sum[m]);
            }
        }
        if (k + moe_block_k < 2560) store(buffer ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t row = tile.first + wm * 32 + m * 16 + (lane / 16) * 8 + i;
            if (row < count) {
                const float g = gate_sum[m][i];
                product[(first + row) * 640 + j0 + wn * 16 + lane % 16] = _Float16(g * hc_sigmoid(g) * up_sum[m][i]);
            }
        }
#endif
}

// One block: 64 grouped product rows x 128 output columns; results scatter to routes.
__global__ void __launch_bounds__(256) moe_down_wmma(const MoeGroup* groups, const MoeTile* tiles,
                                                     const int32_t* positions, const _Float16* product, float* outputs) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int L = moe_lds_k;
    const MoeTile tile = tiles[blockIdx.y];
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    const Matrix down = group.weights[2];
    const int t = threadIdx.x, wave = t / 32, lane = t % 32, wm = wave / 4, wn = wave % 4;
    const size_t n0 = size_t(blockIdx.x) * 128;
    __shared__ __attribute__((aligned(16))) _Float16 as[2][moe_block_m * L];
    __shared__ __attribute__((aligned(16))) _Float16 bs[2][128 * L];
    const int arow = t / 4, acol = (t % 4) * 8;
    const _Float16* source = tile.first + arow < count ? product + (first + tile.first + arow) * 640 + acol : nullptr;
    const int brow = t / 2, bunit = t % 2;
    uint4 a; uint32_t b[8];
    auto load = [&](size_t k) {
        a = source ? *reinterpret_cast<const uint4*>(source + k) : uint4{0, 0, 0, 0};
        moe_decode16(down, n0 + brow, k + bunit * 16, b);
    };
    auto store = [&](int buffer) {
        *reinterpret_cast<uint2*>(&as[buffer][arow * L + moe_panel(acol)]) = uint2{a.x, a.y};
        *reinterpret_cast<uint2*>(&as[buffer][arow * L + moe_panel(acol + 4)]) = uint2{a.z, a.w};
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            *reinterpret_cast<uint2*>(&bs[buffer][brow * L + moe_panel(bunit * 16 + q * 4)]) = uint2{b[2 * q], b[2 * q + 1]};
    };
    moe_float8 sum[2][2] = {};
    load(0); store(0); __syncthreads();
    for (size_t k = 0; k < 640; k += moe_block_k) {
        const int buffer = int(k / moe_block_k) & 1;
        if (k + moe_block_k < 640) load(k + moe_block_k);
        #pragma unroll
        for (int panel = 0; panel < moe_block_k; panel += 16) {
            const int offset = panel + (lane / 16) * 8;
            moe_half8 w[2];
            #pragma unroll
            for (int n = 0; n < 2; ++n) w[n] = *reinterpret_cast<const moe_half8*>(&bs[buffer][(wn * 32 + n * 16 + lane % 16) * L + offset]);
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const moe_half8 x = *reinterpret_cast<const moe_half8*>(&as[buffer][(wm * 32 + m * 16 + lane % 16) * L + offset]);
                #pragma unroll
                for (int n = 0; n < 2; ++n) sum[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(x, w[n], sum[m][n]);
            }
        }
        if (k + moe_block_k < 640) store(buffer ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t row = tile.first + wm * 32 + m * 16 + (lane / 16) * 8 + i;
            if (row >= count) continue;
            float* target = outputs + size_t(positions[first + row]) * 2560 + n0 + wn * 32 + lane % 16;
            target[0] = sum[m][0][i]; target[16] = sum[m][1][i];
        }
#endif
}
#endif

// Grouping helpers shared with the CUDA kernels (moe_gemm_cuda.hpp).
__global__ void moe_counts(const MoeGroup* groups, size_t count, unsigned* counts) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index < count) counts[index] = unsigned(groups[index].count);
}
__global__ void moe_set_groups(MoeGroup* groups, size_t count, const uint32_t* firsts) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index < count) groups[index].first = firsts[index];
}
