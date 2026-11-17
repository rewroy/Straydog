// Independently authored CUDA grouped expert GEMM for prompt batches on INT8 operands: the routed experts' gate/up and down of
// moe_gemm_cuda.hpp's contract (the same groups, tiles and positions; the same outputs) where both sides of every product are INT8.
// A GeForce card's tensor cores take INT8 at four times the rate of FP16 with F32 sums (two RTX 3090 or one RTX 5070 Ti: 284 / 71 and
// 352 / 88 dense TOPS / TFLOPS), and the routed experts' products are a prompt read's largest share.
//
// The arithmetic. An activation unit of 32 values is x = d_x * a with a = round(x / d_x) in -127..127 and d_x = max |x| / 127 (F32). A
// weight unit of 32 values is w = f * c * g with g the format's signed grid value (int8, expert_q8_words), c the unit's integer sub-scale
// (2 s + 1; one to each 16 values for IQ2_S and IQ2_XS) and f the block's F32 factor (expert_q8_factor). So
//     sum w x = f * d_x * sum_halves c_h * sum g a,     |sum g a| <= 32 * 62 * 127 < 2^18,
// an exact INT32 dot from mma.sync (m16n8k32; two m16n8k16 for a scale to every 16 values), turned into F32 once and scaled by
// f * c_h * d_x into the F32 sum. The F32 sums go over the units in order, so a route's sums do not depend on what shares its tile.
// The gate/up product SiLU(gate) * up leaves as INT8 per 32 values with its F32 scale (a route's row of 1280 bytes: 640 values, then 20
// scales, where the FP16 row took the same bytes), which down reads as its activations.
#pragma once
#if !defined(__HIPCC__)

constexpr size_t moe_q8_row_bytes = 1280;   // a route's product row: 640 INT8 values, then its 20 F32 scales
// The kinds the INT8 kernels decode: the IQ formats of resident_expert_q8.hpp, and IQ4_NL and IQ4_XS for down.
__host__ __device__ constexpr bool moe_q8_gate_kind(int kind) { return kind == 16 || kind == 17 || kind == 18 || kind == 21 || kind == 22; }
__host__ __device__ constexpr bool moe_q8_down_kind(int kind) { return moe_q8_gate_kind(kind) || kind == 20 || kind == 23; }
__host__ __device__ constexpr bool moe_q8_halves(int kind) { return kind == 17 || kind == 22; }   // a sub-scale to every 16 values

// The token rows as INT8 units: row r's 2560 values at q + r * 2560, its 80 scales at s + r * 80. A warp a unit, a lane a value.
__global__ void __launch_bounds__(256) moe_input_q8(size_t rows, const float* input, int8_t* q, float* s) {
    const size_t unit = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
    if (unit >= rows * 80) return;
    const unsigned lane = threadIdx.x % 32;
    const float v = input[unit * 32 + lane];
    float m = fabsf(v);
    #pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float d = m / 127.0f;
    const int code = d > 0.0f ? max(-127, min(127, __float2int_rn(v / d))) : 0;
    q[unit * 32 + lane] = int8_t(code);
    if (!lane) s[unit] = d;
}

__device__ __forceinline__ void moe_q8_mma32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; __trap();
#endif
}
__device__ __forceinline__ void moe_q8_mma16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
#else
    (void)c; (void)a0; (void)a1; (void)b0; __trap();
#endif
}

// The grid of an IQ kind and its sign masks in a block's shared memory (resident_expert_q8.hpp's tables, built rather than copied: a mask is
// eight bytes of 0xff where a sign bit is set; IQ2_XXS, IQ2_XS and IQ3_XXS store seven bits and the eighth is their parity).
template<int Kind>
__device__ __forceinline__ void moe_q8_tables(uint8_t* grid, uint2* masks) {
    if constexpr (moe_q8_gate_kind(Kind)) {
        const uint4* from = reinterpret_cast<const uint4*>(expert_q8_grid(Kind));
        for (uint32_t i = threadIdx.x; i < expert_q8_grid_bytes(Kind) / 16; i += blockDim.x) reinterpret_cast<uint4*>(grid)[i] = from[i];
        for (uint32_t i = threadIdx.x; i < uint32_t(expert_q8_masks(Kind)); i += blockDim.x) {
            const unsigned bits = expert_q8_masks(Kind) == 256 ? i : iq_signs(i);
            masks[i] = uint2{expert_q8_spread4(bits & 15u), expert_q8_spread4(bits >> 4)};
        }
    } else {
        (void)grid; (void)masks;
    }
}

// One unit (32 values from column 32 * unit) of a weight row, in two parts so that a K step's loads are in flight while the step before
// computes: moe_q8_fetch reads the unit's fields from global memory as they lie (16-bit halves, bytes), moe_q8_decode puts them together a
// step later and forms its eight words of signed bytes (with the grid and masks in shared memory) and the F32 factors of its two halves
// (equal unless the kind has a sub-scale to every 16 values): the values are factor * byte. Putting the halves together at the fetch had
// the warp wait for the loads right after the barrier, before its step's products (profiled on sm_120, IQ2_S gate/up: with the
// descriptor reads below, a quarter of the warp stall samples).
// Where a thread's row is read from, taken from the Matrix once before the K loop: reading group.weights[] at every step put a second
// dependent global read (the layout, the row's bytes) in front of the unit's.
struct MoeQ8Source {
    const uint8_t* at;   // the row's first byte (native layout), or the matrix's (SoA)
    uint32_t rows;       // (SoA: the matrix's rows)
    bool soa;
};
__device__ __forceinline__ MoeQ8Source moe_q8_source(const Matrix& m, uint32_t row) {
    MoeQ8Source s;
    s.soa = m.layout != 0;
    s.rows = uint32_t(m.rows);
    s.at = s.soa ? m.data : m.data + size_t(row) * m.row_bytes;
    return s;
}
// A unit's fields as read. IQ kinds: h[0] the block's scale, then each 32-bit field as two 16-bit halves (low, high; in the SoA layout the
// whole word and a zero), then the 16-bit words holding its bytes (high bits, sub-scales; in the SoA layout the byte itself). IQ4_NL: the
// scale, then the 16 code bytes as eight halves. IQ4_XS: the scale, the high scale bits, the unit's low scale byte, its eight halves of codes.
__host__ __device__ constexpr int moe_q8_raw_words(int kind) {
    return kind == 16 ? 5 : kind == 17 ? 6 : kind == 20 || kind == 21 ? 9 : kind == 23 ? 11 : 7;
}
template<int Kind>
struct MoeQ8Raw {
    uint32_t h[moe_q8_raw_words(Kind)];
};
template<int Kind>
__device__ __forceinline__ MoeQ8Raw<Kind> moe_q8_fetch(const MoeQ8Source& s, uint32_t row, uint32_t unit) {
    MoeQ8Raw<Kind> r{};
    if constexpr (Kind == 20) {
        const uint16_t* b = reinterpret_cast<const uint16_t*>(s.at + size_t(unit) * 18);
        #pragma unroll
        for (int i = 0; i < 9; ++i) r.h[i] = b[i];
    } else if constexpr (Kind == 23) {
        const uint8_t* b = s.at + size_t(unit / 8) * 136;
        const unsigned group = unit % 8;
        r.h[0] = reinterpret_cast<const uint16_t*>(b)[0];
        r.h[1] = reinterpret_cast<const uint16_t*>(b)[1];
        r.h[2] = b[4 + group / 2];
        const uint16_t* codes = reinterpret_cast<const uint16_t*>(b + 8 + group * 16);
        #pragma unroll
        for (int i = 0; i < 8; ++i) r.h[3 + i] = codes[i];
    } else {
        if constexpr (Kind == 16 || Kind == 18 || Kind == 21 || Kind == 22) {
            if (s.soa) {
                // (expert_soa_fields' places; an SoA matrix has 2560 columns, 80 units a row)
                const size_t count = size_t(s.rows) * 80, index = size_t(unit / 4) * s.rows * 4 + size_t(row) * 4 + unit % 4;
                const uint32_t* fields = reinterpret_cast<const uint32_t*>(s.at);
                r.h[1] = fields[index];
                if constexpr (Kind == 16) {
                    r.h[3] = fields[count + index];
                    r.h[0] = reinterpret_cast<const uint16_t*>(s.at + 8 * count)[size_t(unit / 8) * s.rows + row];
                } else if constexpr (Kind == 22) {
                    r.h[3] = fields[count + index];
                    r.h[5] = s.at[8 * count + index];
                    r.h[6] = s.at[9 * count + index];
                    r.h[0] = reinterpret_cast<const uint16_t*>(s.at + 10 * count)[size_t(unit / 8) * s.rows + row];
                } else {
                    r.h[3] = fields[count + index];
                    r.h[5] = fields[2 * count + index];
                    if constexpr (Kind == 18) {
                        r.h[0] = reinterpret_cast<const uint16_t*>(s.at + 12 * count)[size_t(unit / 8) * s.rows + row];
                    } else {
                        r.h[7] = s.at[12 * count + index];
                        r.h[8] = s.at[13 * count + size_t(unit / 8) * s.rows * 4 + size_t(row) * 4 + (unit % 8) / 2];
                        r.h[0] = reinterpret_cast<const uint16_t*>(s.at + 13 * count + count / 2)[size_t(unit / 8) * s.rows + row];
                    }
                }
                return r;
            }
        }
        constexpr uint32_t block_bytes = Kind == 16 ? 66 : Kind == 17 ? 74 : Kind == 18 ? 98 : Kind == 21 ? 110 : 82;
        const uint16_t* h = reinterpret_cast<const uint16_t*>(s.at + size_t(unit / 8) * block_bytes);
        const unsigned ib = unit % 8;
        r.h[0] = h[0];
        if constexpr (Kind == 22) {
            r.h[1] = h[1 + 2 * ib]; r.h[2] = h[2 + 2 * ib];
            r.h[3] = h[17 + 2 * ib]; r.h[4] = h[18 + 2 * ib];
            r.h[5] = h[33 + ib / 2]; r.h[6] = h[37 + ib / 2];
        } else {
            r.h[1] = h[1 + 4 * ib]; r.h[2] = h[2 + 4 * ib];
            r.h[3] = h[3 + 4 * ib]; r.h[4] = h[4 + 4 * ib];
            if constexpr (Kind == 17) r.h[5] = h[33 + ib / 2];
            if constexpr (Kind == 18) { r.h[5] = h[33 + 2 * ib]; r.h[6] = h[34 + 2 * ib]; }
            if constexpr (Kind == 21) { r.h[5] = h[37 + 2 * ib]; r.h[6] = h[38 + 2 * ib]; r.h[7] = h[33 + ib / 2]; r.h[8] = h[53 + ib / 4]; }
        }
    }
    return r;
}
// The fields of moe_q8_fetch put together as expert_q8_unit has them. `unit` is the unit fetched; in the SoA layout the bytes need no shift.
template<int Kind>
__device__ __forceinline__ ExpertQ8Unit moe_q8_join(const MoeQ8Raw<Kind>& r, bool soa, uint32_t unit) {
    ExpertQ8Unit u{};
    const unsigned ib = unit % 8, byte = soa ? 0u : 8 * (ib & 1);
    u.d = r.h[0];
    u.q0 = r.h[1] | (r.h[2] << 16);
    if constexpr (Kind == 22) {
        u.signs = r.h[3] | (r.h[4] << 16);
        u.high = (r.h[5] >> byte) & 255;
        u.scale = (r.h[6] >> byte) & 255;
    } else if constexpr (Kind == 16) {
        u.signs = r.h[3] | (r.h[4] << 16);
        u.scale = u.signs >> 28;
    } else {
        u.q1 = r.h[3] | (r.h[4] << 16);
        if constexpr (Kind == 17) u.scale = (r.h[5] >> (8 * (ib & 1))) & 255;
        else if constexpr (Kind == 18) { u.signs = r.h[5] | (r.h[6] << 16); u.scale = u.signs >> 28; }
        else {
            u.signs = r.h[5] | (r.h[6] << 16);
            u.high = (r.h[7] >> byte) & 255;
            u.scale = (r.h[8] >> ((soa ? 0u : 8 * ((ib / 2) & 1)) + 4 * (ib & 1))) & 15;
        }
    }
    return u;
}
template<int Kind>
__device__ __forceinline__ void moe_q8_decode(const MoeQ8Raw<Kind>& r, bool soa, uint32_t unit, const uint8_t* grid, const uint2* masks,
                                              uint32_t (&words)[8], float& low, float& high) {
    if constexpr (Kind == 20 || Kind == 23) {
        uint32_t codes[4];
        float factor;
        if constexpr (Kind == 20) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) codes[i] = r.h[1 + 2 * i] | (r.h[2 + 2 * i] << 16);
            factor = gemv_half(r.h[0]);
        } else {
            const unsigned group = unit % 8, head = r.h[1];
            const int scale = int(((r.h[2] >> ((group % 2) * 4)) & 15) | (((head >> (2 * group)) & 3) << 4)) - 32;
            #pragma unroll
            for (int i = 0; i < 4; ++i) codes[i] = r.h[3 + 2 * i] | (r.h[4 + 2 * i] << 16);
            factor = gemv_half(r.h[0]) * float(scale);
        }
        // (the nonlinear table's sixteen levels by byte permutes, as moe_values16_of reads them: bytes 0-15 hold values 0-15 in their low
        // nibbles and values 16-31 in their high ones)
        constexpr uint32_t low0 = 0xBFAD9881u, low1 = 0xF6EADDCFu, high0 = 0x26190D01u, high1 = 0x71594535u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t lo = 0, hi = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned a = (codes[i] >> (8 * j)) & 15, b = (codes[i] >> (8 * j + 4)) & 15;
                const uint32_t va = a < 8 ? __byte_perm(low0, low1, a) : __byte_perm(high0, high1, a - 8);
                const uint32_t vb = b < 8 ? __byte_perm(low0, low1, b) : __byte_perm(high0, high1, b - 8);
                lo |= (va & 0xffu) << (8 * j);
                hi |= (vb & 0xffu) << (8 * j);
            }
            words[i] = lo;
            words[4 + i] = hi;
        }
        low = high = factor;
    } else {
        const ExpertQ8Unit u = moe_q8_join<Kind>(r, soa, unit);
        expert_q8_words<Kind>(u, grid, masks, words);
        const float factor = expert_q8_factor<Kind>(u);
        if constexpr (moe_q8_halves(Kind)) {
            low = factor * float(2 * (u.scale & 15) + 1);
            high = factor * float(2 * (u.scale >> 4) + 1);
        } else {
            low = high = factor * float(2 * u.scale + 1);
        }
    }
}

// Shared memory of a gate/up or down block: the INT8 activations of 64 rows and two units (64 values, padded to 80 bytes a row so that a
// warp's fragment loads fall in 32 banks), the 128 weight rows' two units likewise, their scales, and the kind's grid and masks.
// The activations in two stages: a step's are copied by cp.async during the step before, straight to shared memory, so that they take no
// registers while in flight (the IQ2_S gate/up kernel is at its 128 and spilled with them there).
constexpr int moe_q8_m = 64, moe_q8_stride = 80;
template<int Kind>
struct MoeQ8Lds {
    uint8_t x[2][moe_q8_m * moe_q8_stride];
    uint8_t w[128 * moe_q8_stride];
    float xs[2][moe_q8_m][2];
    float ws[128][2][2];   // (row, unit, half)
    uint8_t grid[moe_q8_gate_kind(Kind) ? expert_q8_grid_bytes(Kind) : 16];
    uint2 masks[moe_q8_gate_kind(Kind) ? expert_q8_masks(Kind) : 1];
};

// A warp's share of a K step of two units: its 16 rows against its eight tiles of eight weight rows (tiles 0-3 from row `col_a` of the
// block's 128, tiles 4-7 from `col_b`), each unit's INT32 dots scaled into the F32 sums.
template<int Kind>
__device__ __forceinline__ void moe_q8_step(const MoeQ8Lds<Kind>& lds, int stage, int row0, int col_a, int col_b, float (&sums)[8][4]) {
    const int lane = threadIdx.x % 32, g = lane / 4, t4 = lane % 4;
    #pragma unroll 1   // (the two units in turn: unrolled, the IQ2_S kernel spilled at its 128 registers)
    for (int unit = 0; unit < 2; ++unit) {
        const uint8_t* xa = lds.x[stage] + (row0 + g) * moe_q8_stride + unit * 32 + t4 * 4;
        const uint8_t* xb = xa + 8 * moe_q8_stride;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(xa), a1 = *reinterpret_cast<const uint32_t*>(xb);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(xa + 16), a3 = *reinterpret_cast<const uint32_t*>(xb + 16);
        const float xs0 = lds.xs[stage][row0 + g][unit], xs1 = lds.xs[stage][row0 + g + 8][unit];
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int col = n < 4 ? col_a + 8 * n : col_b + 8 * (n - 4);
            const uint8_t* wb = lds.w + (col + g) * moe_q8_stride + unit * 32 + t4 * 4;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(wb), b1 = *reinterpret_cast<const uint32_t*>(wb + 16);
            const int c0 = col + 2 * t4, c1 = c0 + 1;
            if constexpr (moe_q8_halves(Kind)) {
                // (the two halves one after the other, each into the sums: fewer registers live than both dots at once)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int c[4] = {0, 0, 0, 0};
                    if (h == 0) moe_q8_mma16(c, a0, a1, b0); else moe_q8_mma16(c, a2, a3, b1);
                    const float w0 = lds.ws[c0][unit][h], w1 = lds.ws[c1][unit][h];
                    sums[n][0] += (xs0 * w0) * float(c[0]);
                    sums[n][1] += (xs0 * w1) * float(c[1]);
                    sums[n][2] += (xs1 * w0) * float(c[2]);
                    sums[n][3] += (xs1 * w1) * float(c[3]);
                }
            } else {
                int c[4] = {0, 0, 0, 0};
                moe_q8_mma32(c, a0, a1, a2, a3, b0, b1);
                const float w0 = lds.ws[c0][unit][0], w1 = lds.ws[c1][unit][0];
                sums[n][0] += (xs0 * w0) * float(c[0]);
                sums[n][1] += (xs0 * w1) * float(c[1]);
                sums[n][2] += (xs1 * w0) * float(c[2]);
                sums[n][3] += (xs1 * w1) * float(c[3]);
            }
        }
    }
}

// Four bytes to shared memory by cp.async, or zero where `src` is null (`fallback` is any valid address); committed with moe_copy_commit.
__device__ __forceinline__ void moe_q8_copy4(void* dst, const void* src, const void* fallback) {
#if __CUDA_ARCH__ >= 800
    const unsigned address = unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(address), "l"(src ? src : fallback), "r"(src ? 4 : 0) : "memory");
#else
    *reinterpret_cast<uint32_t*>(dst) = src ? *reinterpret_cast<const uint32_t*>(src) : 0u;
#endif
}

// One block: 64 routed rows x 64 gate/up product columns (128 weight rows: 64 of gate, then the same 64 of up). Warp w takes rows
// 16 (w / 2) and product columns 32 (w % 2): eight tiles, four of gate and four of up. `xq`, `xs`: moe_input_q8's rows; `product`: the
// routes' INT8 product rows (moe_q8_row_bytes each).
template<int Kind>
__global__ void __launch_bounds__(256, 2) moe_gate_up_q8(const MoeGroup* groups, const MoeTile* tiles, const int32_t* positions, size_t steps,
                                                      const int8_t* xq, const float* xs, uint8_t* product, const unsigned* tile_count = nullptr) {
    if (tile_count && blockIdx.y >= *tile_count) return;
    const MoeTile tile = tiles[blockIdx.y];
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    if (tile.first >= count) return;
    __shared__ __align__(16) MoeQ8Lds<Kind> lds;
    moe_q8_tables<Kind>(lds.grid, lds.masks);
    const int t = threadIdx.x, warp = t / 32, wm = warp / 2, wn = warp % 2;
    const uint32_t j0 = blockIdx.x * 64;
    // (a thread's activation chunk: row t / 4, 16 bytes at 16 (t % 4) of the step's 64 values; its weight unit: row t / 2, unit t % 2)
    const int arow = t / 4, achunk = t % 4, wrow = t / 2, wunit = t % 2;
    size_t token = SIZE_MAX;
    if (tile.first + arow < count) {
        const size_t route = size_t(positions[first + tile.first + arow]);
        token = route < steps * 10 ? route / 10 : route - steps * 10;
    }
    const uint32_t wmrow = j0 + uint32_t(wrow % 64);
    const MoeQ8Source wsrc = moe_q8_source(group.weights[wrow / 64], wmrow);
    float sums[8][4] = {};
    const bool rows = tile.first + size_t(wm) * 16 < count;
    __syncthreads();
    // (a step's activation chunk and its scale are copied in the step before, to the other stage; its weight unit is read then too)
    const int8_t* xrow = token != SIZE_MAX ? xq + token * 2560 + achunk * 16 : nullptr;
    const float* srow = token != SIZE_MAX && achunk < 2 ? xs + token * 80 + achunk : nullptr;
    auto copy_x = [&](uint32_t k, int stage) {
        moe_copy16(lds.x[stage] + arow * moe_q8_stride + achunk * 16, xrow ? xrow + k * 32 : xq, xrow ? 16u : 0u);
        if (achunk < 2) moe_q8_copy4(&lds.xs[stage][arow][achunk], srow ? srow + k : nullptr, xs);
        moe_copy_commit();
    };
    copy_x(0, 0);
    MoeQ8Raw<Kind> raw = moe_q8_fetch<Kind>(wsrc, wmrow, wunit);
    for (uint32_t k = 0; k < 2560 / 32; k += 2) {
        const int stage = (k / 2) % 2;
        uint32_t words[8];
        float low, high;
        moe_q8_decode<Kind>(raw, wsrc.soa, k + wunit, lds.grid, lds.masks, words, low, high);
        uint4* to = reinterpret_cast<uint4*>(lds.w + wrow * moe_q8_stride + wunit * 32);
        to[0] = make_uint4(words[0], words[1], words[2], words[3]);
        to[1] = make_uint4(words[4], words[5], words[6], words[7]);
        lds.ws[wrow][wunit][0] = low; lds.ws[wrow][wunit][1] = high;
        moe_copy_wait();
        __syncthreads();
        if (k + 2 < 2560 / 32) { copy_x(k + 2, stage ^ 1); raw = moe_q8_fetch<Kind>(wsrc, wmrow, k + 2 + wunit); }
        // (tiles 0-3: gate's columns 32 wn to 32 wn + 31; tiles 4-7: up's, 64 weight rows on)
        if (rows) moe_q8_step<Kind>(lds, stage, wm * 16, wn * 32, 64 + wn * 32, sums);
        __syncthreads();
    }
    if (tile.first + size_t(wm) * 16 >= count) return;
    // SiLU(gate) * up, then a unit of 32 product columns (this warp's) as INT8 with its scale: a row's 32 values are in the four lanes of a quad
    // (eight each), so its largest magnitude is a quad's reduction.
    const int lane = t % 32, g = lane / 4, t4 = lane % 4;
    float v[2][8];   // (row g, row g + 8) x (tile n, column 2 t4 + i)
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float gv = sums[n][i], uv = sums[4 + n][i];
            v[i / 2][2 * n + i % 2] = gv * hc_sigmoid(gv) * uv;
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        float m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(v[r][i]));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        const size_t row = tile.first + size_t(wm) * 16 + g + 8 * r;
        if (row >= count) continue;
        const float d = m / 127.0f;
        uint8_t* out = product + (first + row) * moe_q8_row_bytes;
        const uint32_t col0 = j0 + uint32_t(wn) * 32;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int q0 = d > 0.0f ? max(-127, min(127, __float2int_rn(v[r][2 * n] / d))) : 0;
            const int q1 = d > 0.0f ? max(-127, min(127, __float2int_rn(v[r][2 * n + 1] / d))) : 0;
            *reinterpret_cast<uint16_t*>(out + col0 + 8 * n + 2 * t4) = uint16_t(uint8_t(q0) | (uint32_t(uint8_t(q1)) << 8));
        }
        if (!t4) reinterpret_cast<float*>(out + 640)[col0 / 32] = d;
    }
}

// One block: 64 routed rows x 128 output columns (from 128 blockIdx.x) over the 640 product columns. Warp w takes rows 16 (w / 2) and
// columns 64 (w % 2): eight tiles. The sums go to the routes' output rows as moe_down_mma's do.
template<int Kind>
__global__ void __launch_bounds__(256, 2) moe_down_q8(const MoeGroup* groups, const MoeTile* tiles, const int32_t* positions, const uint8_t* product,
                                                   float* outputs, const unsigned* tile_count = nullptr) {
    if (tile_count && blockIdx.y >= *tile_count) return;
    const MoeTile tile = tiles[blockIdx.y];
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    if (tile.first >= count) return;
    __shared__ __align__(16) MoeQ8Lds<Kind> lds;
    moe_q8_tables<Kind>(lds.grid, lds.masks);
    const int t = threadIdx.x, warp = t / 32, wm = warp / 2, wn = warp % 2;
    const uint32_t n0 = blockIdx.x * 128;
    const int arow = t / 4, achunk = t % 4, wrow = t / 2, wunit = t % 2;
    const uint32_t wmrow = n0 + uint32_t(wrow);
    const MoeQ8Source wsrc = moe_q8_source(group.weights[2], wmrow);
    const uint8_t* source = tile.first + arow < count ? product + (first + tile.first + arow) * moe_q8_row_bytes : nullptr;
    float sums[8][4] = {};
    const bool rows = tile.first + size_t(wm) * 16 < count;
    __syncthreads();
    auto copy_x = [&](uint32_t k, int stage) {
        moe_copy16(lds.x[stage] + arow * moe_q8_stride + achunk * 16, source ? source + k * 32 + achunk * 16 : product, source ? 16u : 0u);
        if (achunk < 2)
            moe_q8_copy4(&lds.xs[stage][arow][achunk], source ? reinterpret_cast<const float*>(source + 640) + k + achunk : nullptr, product);
        moe_copy_commit();
    };
    copy_x(0, 0);
    MoeQ8Raw<Kind> raw = moe_q8_fetch<Kind>(wsrc, wmrow, wunit);
    for (uint32_t k = 0; k < 640 / 32; k += 2) {
        const int stage = (k / 2) % 2;
        uint32_t words[8];
        float low, high;
        moe_q8_decode<Kind>(raw, wsrc.soa, k + wunit, lds.grid, lds.masks, words, low, high);
        uint4* to = reinterpret_cast<uint4*>(lds.w + wrow * moe_q8_stride + wunit * 32);
        to[0] = make_uint4(words[0], words[1], words[2], words[3]);
        to[1] = make_uint4(words[4], words[5], words[6], words[7]);
        lds.ws[wrow][wunit][0] = low; lds.ws[wrow][wunit][1] = high;
        moe_copy_wait();
        __syncthreads();
        if (k + 2 < 640 / 32) { copy_x(k + 2, stage ^ 1); raw = moe_q8_fetch<Kind>(wsrc, wmrow, k + 2 + wunit); }
        if (rows) moe_q8_step<Kind>(lds, stage, wm * 16, wn * 64, wn * 64 + 32, sums);
        __syncthreads();
    }
    if (!rows) return;
    const int lane = t % 32, g = lane / 4, t4 = lane % 4;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const size_t row = tile.first + size_t(wm) * 16 + g + 8 * r;
        if (row >= count) continue;
        float* out = outputs + size_t(positions[first + row]) * 2560 + n0 + uint32_t(wn) * 64 + 2 * t4;
        #pragma unroll
        for (int n = 0; n < 8; ++n) *reinterpret_cast<float2*>(out + 8 * n) = make_float2(sums[n][2 * r], sums[n][2 * r + 1]);
    }
}
#endif
