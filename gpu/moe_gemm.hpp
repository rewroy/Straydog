// Independently authored RDNA4 grouped expert GEMM for prefill. Routed rows are
// grouped by expert on the device and the grids cover every 64-row tile there can be.
// Shapes are parameters (MoeShape): any hidden and expert width that are multiples of
// the tiles and of each format's unit.
// Each lane decodes its packed weights straight into FP16 WMMA fragments, every value
// as the F32 decoders round it (Q2_0 values scale*(code-1) are exact in FP16);
// activations are rounded to FP16 and products accumulate in F32, as in the explicit
// FP16 GEMM mode. Gate and up share each tile so SiLU(gate)*up is formed before
// leaving the block.
#if defined(__HIPCC__)
typedef _Float16 moe_half2 __attribute__((ext_vector_type(2)));
typedef _Float16 moe_half8 __attribute__((ext_vector_type(8)));
typedef float moe_float8 __attribute__((ext_vector_type(8)));
typedef uint32_t moe_word4 __attribute__((ext_vector_type(4)));
// Blocks of 110, 98, 82, 18 or 34 bytes start at any even address; RDNA4 loads them unaligned.
typedef uint32_t moe_word_u __attribute__((aligned(2)));
typedef uint2 moe_word2_u __attribute__((aligned(2)));
typedef moe_word4 moe_word4_u __attribute__((aligned(2)));
typedef moe_word4 moe_word4_b __attribute__((aligned(1)));   // MXFP4: 17-byte blocks at any byte
constexpr int moe_block_k = 64, moe_lds_k = moe_block_k + 8, moe_block_m = 64;

__device__ __forceinline__ uint32_t moe_pack(float a, float b) {
    const moe_half2 v = {_Float16(a), _Float16(b)};
    return __builtin_bit_cast(uint32_t, v);
}
__device__ __forceinline__ uint32_t moe_u16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ uint32_t moe_u32(const uint8_t* p) { return *reinterpret_cast<const moe_word_u*>(p); }
__device__ __forceinline__ uint2 moe_u64(const uint8_t* p) { return *reinterpret_cast<const moe_word2_u*>(p); }
__device__ __forceinline__ moe_word4 moe_u128(const uint8_t* p) { return *reinterpret_cast<const moe_word4_u*>(p); }
__device__ __forceinline__ void moe_store8(const float (&v)[8], uint32_t (&out)[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = moe_pack(v[2 * i], v[2 * i + 1]);
}
// Eight IQ4_XS values (nibbles at bit 4 * shift of two words) times d * scale in F32, as weight<23> forms
// them (a zero scale giving +0, as it compiles).
__device__ __forceinline__ void moe_nonlinear8(uint32_t first, uint32_t second, unsigned shift, float scale, uint32_t (&out)[4]) {
    const uint32_t bytes[2] = {gemv_iq4((first >> shift) & 0x0F0F0F0Fu), gemv_iq4((second >> shift) & 0x0F0F0F0Fu)};
    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = scale * (float((bytes[j / 4] >> (8 * (j % 4))) & 255u) - 128.0f) + 0.0f;
    moe_store8(v, out);
}

// Values k .. k + 7 of a weight row of any other format (the shared expert's mixes of gate and up
// formats without a unit), as weight<kind> forms them.
__device__ __forceinline__ void moe_decode8(const uint8_t* row, int kind, size_t k, uint32_t (&out)[4]) {
    kind = __builtin_amdgcn_readfirstlane(kind);   // one format per wave: scalar branches
    float v[8];
    #define SD_DECODE(format) case format: for (int j = 0; j < 8; ++j) v[j] = weight<format>(row, k + j); break;
    switch (kind) {
        SD_FORMATS(SD_DECODE)
        default: for (int j = 0; j < 8; ++j) v[j] = 0;
    }
    #undef SD_DECODE
    moe_store8(v, out);
}

// One lane's share of a weight row for the WMMA B operand. A row is consumed in units of `span`
// values; the lane holds half (lane / 16) of each unit, loaded a unit ahead into registers, and panel
// p (16 values of K: eight per half) takes values half * span / 2 + 8p .. + 7 of the unit. Activations
// are staged in the same order: the products are those of the plain K order, summed in another.
// Kind 0 decodes any format from global memory, one K step per unit.
template<int kind> struct MoeUnit {
    static constexpr int span = moe_block_k, table_words = 0;
    const uint8_t* row; size_t k; int format;
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* weights, int kind_, size_t unit, unsigned half) {
        row = weights; format = kind_; k = unit * span + span / 2 * half;
    }
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const { moe_decode8(row, format, k + 8 * p, out); }
};
// Packed FP16 decoding. An IQ2/IQ3 value is d * c * g * sign, with c the block's odd scale factor and
// g a grid value (both scaled by a power of two so c * g is an integer below 2048 times that power):
// c * g is exact in FP16, so d * (c * g) is the one rounding of the exact F32 product, the value the
// F32 decoders round to FP16. Each block stages the grid in LDS as FP16 g.
__device__ __forceinline__ uint32_t moe_hmul(uint32_t a, uint32_t b) {
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(moe_half2, a) * __builtin_bit_cast(moe_half2, b));
}
__device__ __forceinline__ uint32_t moe_hadd(uint32_t a, uint32_t b) {
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(moe_half2, a) + __builtin_bit_cast(moe_half2, b));
}
__device__ __forceinline__ uint32_t moe_hpair(unsigned integer) {   // FP16 integer (below 2048) in both halves
    const uint32_t bits = __builtin_bit_cast(uint16_t, _Float16(uint16_t(integer)));
    return bits | (bits << 16);
}
// Bytes 0-1 and 2-3 of `bytes` as FP16 pairs 1024 + byte; adding -1024 - bias gives byte - bias exactly.
__device__ __forceinline__ uint32_t moe_byte_pair(uint32_t bytes, int high) {
    return __builtin_amdgcn_perm(0x64646464u, bytes, high ? 0x04030402u : 0x04010400u);
}
// Byte `byte` of `word` in both 16-bit halves: an IQ sign byte for moe_scaled8.
__device__ __forceinline__ uint32_t moe_sign_byte(uint32_t word, int byte) {
    return __builtin_amdgcn_perm(word, word, 0x0c000c00u | byte | (byte << 16));
}
// d * (c * g) for four grid pairs, value j taking the sign of bit j of the sign byte. c * g is
// positive, so each pair's two sign bits are set (bits 2q and 2q + 1 shifted to bit 15 of each half)
// before the rounding multiply, which is symmetric.
__device__ __forceinline__ void moe_scaled8(uint32_t g0, uint32_t g1, uint32_t g2, uint32_t g3, uint32_t c, uint32_t d,
                                            uint32_t signs, uint32_t (&out)[4]) {
    typedef uint16_t moe_u16x2 __attribute__((ext_vector_type(2)));
    const uint32_t g[4] = {g0, g1, g2, g3};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const moe_u16x2 shift = {uint16_t(15 - 2 * q), uint16_t(14 - 2 * q)};
        const uint32_t sign = __builtin_bit_cast(uint32_t, __builtin_bit_cast(moe_u16x2, signs) << shift) & 0x80008000u;
        out[q] = moe_hmul(moe_hmul(g[q], c) | sign, d);
    }
}
// The grid as FP16 values times `scale` (a power of two: exact), staged to LDS by every block four bytes at a time.
__device__ __forceinline__ void moe_stage_grid(uint32_t* table, const uint8_t* grid, int bytes, float scale) {
    const uint32_t bias = 0xE400E400u, factor = moe_pack(scale, scale);   // -1024
    for (int i = threadIdx.x; i < bytes / 4; i += 256) {
        const uint32_t word = reinterpret_cast<const uint32_t*>(grid)[i];
        *reinterpret_cast<uint2*>(table + 2 * i) = uint2{moe_hmul(moe_hadd(moe_byte_pair(word, 0), bias), factor),
                                                         moe_hmul(moe_hadd(moe_byte_pair(word, 1), bias), factor)};
    }
}
template<> struct MoeUnit<21> {   // IQ3_S: a half is 32 qs, 4 qh and 16 sign bytes and two scale bytes
    static constexpr int span = 256, table_words = 1024;
    uint32_t qs[8], signs[4], qh, header;   // header: d, then the half's scale bytes
    __device__ static void stage(uint32_t* table) { moe_stage_grid(table, iq_grid3s, 2048, 1.0f); }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 110;
        const moe_word4 q0 = moe_u128(b + 2 + 32 * half), q1 = moe_u128(b + 18 + 32 * half), s = moe_u128(b + 74 + 16 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; signs[i] = s[i]; }
        qh = moe_u32(b + 66 + 4 * half);
        header = moe_u16(b) | moe_u16(b + 106 + 2 * half) << 16;
    }
    __device__ void load_soa(const Matrix& m, size_t row, size_t unit, unsigned half) {
        qh = 0; header = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const ExpertSoaFields f = expert_soa_fields<21>(m, row, unit * 8 + half * 4 + i);
            qs[2 * i] = f.q0; qs[2 * i + 1] = f.q1; signs[i] = f.signs;
            qh |= f.high << (8 * i);
            header |= f.scale << (16 + 4 * i);
            if (i == 0) header |= f.d;
        }
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, g = p % 4;
        const unsigned pair = qs[2 * i + g / 2] >> (16 * (g % 2)), top = qh >> (8 * i);
        const uint2 low = reinterpret_cast<const uint2*>(table)[(pair & 255) | ((top << (8 - 2 * g)) & 256)];
        const uint2 high = reinterpret_cast<const uint2*>(table)[((pair >> 8) & 255) | ((top << (7 - 2 * g)) & 256)];
        moe_scaled8(low.x, low.y, high.x, high.y, moe_hpair(1 + 2 * ((header >> (16 + 4 * i)) & 15)), header & 0xffff | header << 16,
                    moe_sign_byte(signs[i], g), out);
    }
};
template<> struct MoeUnit<18> {   // IQ3_XXS: a half is 32 qs bytes and four sign/scale words; grid values / 4
    static constexpr int span = 256, table_words = 512;
    uint32_t qs[8], aux[4], d;
    __device__ static void stage(uint32_t* table) { moe_stage_grid(table, iq_grid3xxs, 1024, 0.25f); }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 98;
        const moe_word4 q0 = moe_u128(b + 2 + 32 * half), q1 = moe_u128(b + 18 + 32 * half), a = moe_u128(b + 66 + 16 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; aux[i] = a[i]; }
        d = moe_u16(b);
    }
    __device__ void load_soa(const Matrix& m, size_t row, size_t unit, unsigned half) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const ExpertSoaFields f = expert_soa_fields<18>(m, row, unit * 8 + half * 4 + i);
            qs[2 * i] = f.q0; qs[2 * i + 1] = f.q1; aux[i] = f.signs;
            if (i == 0) d = f.d;
        }
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, g = p % 4;
        const unsigned pair = qs[2 * i + g / 2] >> (16 * (g % 2));
        const uint2 low = reinterpret_cast<const uint2*>(table)[pair & 255];
        const uint2 high = reinterpret_cast<const uint2*>(table)[(pair >> 8) & 255];
        moe_scaled8(low.x, low.y, high.x, high.y, moe_hpair(1 + 2 * (aux[i] >> 28)), d | d << 16,
                    moe_sign_byte(iq_signs((aux[i] >> (7 * g)) & 127), 0), out);
    }
};
template<> struct MoeUnit<22> {   // IQ2_S: a half is 16 qs, 16 sign, 4 qh and 4 scale bytes; grid values / 8
    static constexpr int span = 256, table_words = 4096;
    uint32_t qs[4], signs[4], qh, scales, d;
    __device__ static void stage(uint32_t* table) { moe_stage_grid(table, iq_grid2s, 8192, 0.125f); }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 82;
        const moe_word4 q = moe_u128(b + 2 + 16 * half), s = moe_u128(b + 34 + 16 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q[i]; signs[i] = s[i]; }
        qh = moe_u32(b + 66 + 4 * half); scales = moe_u32(b + 74 + 4 * half); d = moe_u16(b);
    }
    __device__ void load_soa(const Matrix& m, size_t row, size_t unit, unsigned half) {
        qh = 0; scales = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const ExpertSoaFields f = expert_soa_fields<22>(m, row, unit * 8 + half * 4 + i);
            qs[i] = f.q0; signs[i] = f.signs;
            qh |= f.high << (8 * i); scales |= f.scale << (8 * i);
            if (i == 0) d = f.d;
        }
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, g = p % 4;
        const unsigned index = ((qs[i] >> (8 * g)) & 255) | (((qh >> (8 * i)) << (8 - 2 * g)) & 0x300);
        const moe_word4 row = reinterpret_cast<const moe_word4*>(table)[index];
        moe_scaled8(row[0], row[1], row[2], row[3], moe_hpair(1 + 2 * ((scales >> (8 * i + 4 * (g / 2))) & 15)), d | d << 16,
                    moe_sign_byte(signs[i], g), out);
    }
};
template<> struct MoeUnit<16> {   // IQ2_XXS: a half is four (grid bytes, sign/scale word) pairs; grid values / 8
    static constexpr int span = 256, table_words = 1024;
    uint32_t qs[8], d;   // per 32-value block: its four grid bytes, then its signs and scale
    __device__ static void stage(uint32_t* table) { moe_stage_grid(table, iq_grid2xxs, 2048, 0.125f); }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 66;
        const moe_word4 q0 = moe_u128(b + 2 + 32 * half), q1 = moe_u128(b + 18 + 32 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; }
        d = moe_u16(b);
    }
    __device__ void load_soa(const Matrix& m, size_t row, size_t unit, unsigned half) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const ExpertSoaFields f = expert_soa_fields<16>(m, row, unit * 8 + half * 4 + i);
            qs[2 * i] = f.q0; qs[2 * i + 1] = f.signs;
            if (i == 0) d = f.d;
        }
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, g = p % 4;
        const uint32_t aux = qs[2 * i + 1];
        const moe_word4 row = reinterpret_cast<const moe_word4*>(table)[(qs[2 * i] >> (8 * g)) & 255];
        moe_scaled8(row[0], row[1], row[2], row[3], moe_hpair(1 + 2 * (aux >> 28)), d | d << 16,
                    moe_sign_byte(iq_signs((aux >> (7 * g)) & 127), 0), out);
    }
};
template<> struct MoeUnit<17> {   // IQ2_XS: a half is 16 grid/sign words and 4 scale bytes; grid values / 8
    static constexpr int span = 256, table_words = 2048;
    uint32_t qs[8], scales, d;
    __device__ static void stage(uint32_t* table) { moe_stage_grid(table, iq_grid2xs, 4096, 0.125f); }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 74;
        const moe_word4 q0 = moe_u128(b + 2 + 32 * half), q1 = moe_u128(b + 18 + 32 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; }
        scales = moe_u32(b + 66 + 4 * half); d = moe_u16(b);
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, g = p % 4;
        const unsigned q = (qs[2 * i + g / 2] >> (16 * (g % 2))) & 0xffff;
        const moe_word4 row = reinterpret_cast<const moe_word4*>(table)[q & 511];
        moe_scaled8(row[0], row[1], row[2], row[3], moe_hpair(1 + 2 * ((scales >> (8 * i + 4 * (g / 2))) & 15)), d | d << 16,
                    moe_sign_byte(iq_signs(q >> 9), 0), out);
    }
};
// IQ1_M: a value is d * c * (g - 1 + delta), g in {0, 1, 2} and delta +-1/8, so c * (g - 1 + delta) is an
// integer below 136 over 8 (exact in FP16) and d times it the one rounding. Its 2048-row grid stays bytes
// in LDS; a half is 16 grid bytes, 8 high-bit/delta bytes and the scale words.
template<> struct MoeUnit<29> {
    static constexpr int span = 256, table_words = 4096;
    uint32_t qs[4], qh[2], scales, d;   // scales: the half's two sub-scale words
    __device__ static void stage(uint32_t* table) {
        for (int i = threadIdx.x; i < 4096; i += 256) table[i] = reinterpret_cast<const uint32_t*>(iq_grid1s)[i];
    }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 56;
        const moe_word4 q = moe_u128(b + 16 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) qs[i] = q[i];
        const uint2 h = moe_u64(b + 32 + 8 * half), w = moe_u64(b + 48);
        qh[0] = h.x; qh[1] = h.y;
        scales = half ? w.y : w.x;
        const uint32_t bits = ((w.x >> 12) & 0xf) | ((w.x >> 24) & 0xf0) | ((w.y >> 4) & 0xf00) | ((w.y >> 16) & 0xf000);
        d = bits | bits << 16;
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, l = p % 4;
        const unsigned high = (qh[i / 2] >> (16 * (i % 2) + 4 * l)) & 15;
        const uint2 grid = reinterpret_cast<const uint2*>(table)[((qs[i] >> (8 * l)) & 255) | ((high & 7) << 8)];
        const uint32_t c = moe_hpair(1 + 2 * ((scales >> (16 * (i / 2) + 6 * (i % 2) + 3 * (l / 2))) & 7));
        const uint32_t delta = high & 8 ? 0xB000B000u : 0x30003000u;   // -1/8, 1/8
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t g = moe_byte_pair(q < 2 ? grid.x : grid.y, q % 2);
            out[q] = moe_hmul(moe_hmul(moe_hadd(moe_hadd(g, 0xE401E401u), delta), c), d);   // 1024 + g - 1025 = g - 1
        }
    }
};
template<> struct MoeUnit<23> {   // IQ4_XS: a half is four 32-value blocks' 64 code bytes and their scales
    static constexpr int span = 256, table_words = 0;
    uint32_t qs[16], d, scales;   // scales: the half's four low nibbles, then its eight high bits
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 136;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const moe_word4 q = moe_u128(b + 8 + 64 * half + 16 * i);
            for (int j = 0; j < 4; ++j) qs[4 * i + j] = q[j];
        }
        const uint32_t head = moe_u32(b);
        d = head & 0xffff;
        scales = ((moe_u32(b + 4) >> (16 * half)) & 0xffff) | (((head >> (16 + 8 * half)) & 255) << 16);
    }
    // (d * scale) * value in F32 as weight<23>: the product can round in F32, so it stays there.
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const int i = p / 4, c = p % 4;
        const int scale = int(((scales >> (4 * i)) & 15) | (((scales >> (16 + 2 * i)) & 3) << 4)) - 32;
        moe_nonlinear8(qs[4 * i + 2 * (c % 2)], qs[4 * i + 2 * (c % 2) + 1], 4 * (c / 2), gemv_half(d) * float(scale), out);
    }
};
template<> struct MoeUnit<20> {   // IQ4_NL: a half is one 18-byte block; d * value rounds once in FP16
    static constexpr int span = 64, table_words = 0;
    uint32_t qs[4], d;
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + (2 * unit + half) * 18;
        const moe_word4 q = moe_u128(b + 2);
        #pragma unroll
        for (int i = 0; i < 4; ++i) qs[i] = q[i];
        d = moe_u16(b);
    }
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const uint32_t scale = d | d << 16, bias = 0xE480E480u;   // -1152 = -1024 - 128
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            const uint32_t bytes = gemv_iq4((qs[2 * (p % 2) + w] >> (4 * (p / 2))) & 0x0F0F0F0Fu);
            out[2 * w] = moe_hmul(moe_hadd(moe_byte_pair(bytes, 0), bias), scale);
            out[2 * w + 1] = moe_hmul(moe_hadd(moe_byte_pair(bytes, 1), bias), scale);
        }
    }
};
template<> struct MoeUnit<8> {   // Q8_0: a half is one 34-byte block; d * value rounds once in FP16
    static constexpr int span = 64, table_words = 0;
    uint32_t qs[8], d;
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + (2 * unit + half) * 34;
        const moe_word4 q0 = moe_u128(b + 2), q1 = moe_u128(b + 18);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; }
        d = moe_u16(b);
    }
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const uint32_t scale = d | d << 16, bias = 0xE480E480u;   // codes + 128 are the bytes with the top bit flipped
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            const uint32_t bytes = qs[2 * p + w] ^ 0x80808080u;
            out[2 * w] = moe_hmul(moe_hadd(moe_byte_pair(bytes, 0), bias), scale);
            out[2 * w + 1] = moe_hmul(moe_hadd(moe_byte_pair(bytes, 1), bias), scale);
        }
    }
};
template<> struct MoeUnit<42> {   // Q2_0: a half is one 18-byte block of 64 values
    static constexpr int span = 128, table_words = 0;
    uint32_t codes[4], d;
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + (2 * unit + half) * 18;
        const moe_word4 q = moe_u128(b + 2);
        #pragma unroll
        for (int i = 0; i < 4; ++i) codes[i] = q[i];
        d = moe_u16(b);
    }
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const uint32_t scale = d | d << 16, pair = codes[p / 2] >> (16 * (p % 2));
        // FP16 high bytes of (code-1): -1 0xBC, 0 0x00, 1 0x3C, 2 0x40; the low bytes are zero.
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            const uint32_t byte = (pair >> (8 * b)) & 255u;
            const uint32_t spread = (byte & 3u) | ((byte & 12u) << 6) | ((byte & 48u) << 12) | ((byte & 192u) << 18);
            const uint32_t high_bytes = __builtin_amdgcn_perm(0u, 0x403C00BCu, spread);
            out[2 * b] = moe_hmul(__builtin_amdgcn_perm(high_bytes, 0u, 0x050C040Cu), scale);
            out[2 * b + 1] = moe_hmul(__builtin_amdgcn_perm(high_bytes, 0u, 0x070C060Cu), scale);
        }
    }
};

// IQ1_S: as IQ1_M, c * (g - 1 + delta) is an integer below 136 over 8 and d times it the one rounding;
// a half is 16 grid bytes, four 16-bit words (grid-row high bits, scale and delta) and d.
template<> struct MoeUnit<19> {
    static constexpr int span = 256, table_words = 4096;
    uint32_t qs[4], qh[2], d;
    __device__ static void stage(uint32_t* table) { MoeUnit<29>::stage(table); }
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 50;
        const moe_word4 q = moe_u128(b + 2 + 16 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) qs[i] = q[i];
        const uint2 h = moe_u64(b + 34 + 8 * half);
        qh[0] = h.x; qh[1] = h.y; d = moe_u16(b) | moe_u16(b) << 16;
    }
    __device__ void decode(int p, const uint32_t* table, uint32_t (&out)[4]) const {
        const int i = p / 4, l = p % 4;
        const unsigned high = (qh[i / 2] >> (16 * (i % 2))) & 0xffff;
        const uint2 grid = reinterpret_cast<const uint2*>(table)[((qs[i] >> (8 * l)) & 255) | (((high >> (3 * l)) & 7) << 8)];
        const uint32_t c = moe_hpair(1 + 2 * ((high >> 12) & 7));
        const uint32_t delta = high & 0x8000 ? 0xB000B000u : 0x30003000u;   // -1/8, 1/8
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t g = moe_byte_pair(q < 2 ? grid.x : grid.y, q % 2);
            out[q] = moe_hmul(moe_hmul(moe_hadd(moe_hadd(g, 0xE401E401u), delta), c), d);   // 1024 + g - 1025 = g - 1
        }
    }
};
// Q2_K: a half is 32 code bytes, its eight scale bytes, d and dmin. A value is
// d * scale * code - dmin * minimum in F32, as weight<10> forms it, then rounded to FP16.
template<> struct MoeUnit<10> {
    static constexpr int span = 256, table_words = 0;
    uint32_t qs[8], scales[2], dm;
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + unit * 84;
        const moe_word4 q0 = moe_u128(b + 16 + 32 * half), q1 = moe_u128(b + 32 + 32 * half);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { qs[i] = q0[i]; qs[4 + i] = q1[i]; }
        const uint2 s = moe_u64(b + 8 * half);
        scales[0] = s.x; scales[1] = s.y; dm = moe_u32(b + 80);
    }
    // Values half * 128 + 8p .. + 7: two bits at shift 2 * (p / 4) of code bytes 8 * (p % 4) ..
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const int quarter = p / 4, c = p % 4, k = 2 * quarter + c / 2;
        const unsigned scale = (scales[k / 4] >> (8 * (k % 4))) & 255;
        const float ds = gemv_half(dm) * float(scale & 15), dmin = gemv_half(dm >> 16) * float(scale >> 4);
        float v[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = ds * float((qs[2 * c + j / 4] >> (8 * (j % 4) + 2 * quarter)) & 3) - dmin;
        moe_store8(v, out);
    }
};
// MXFP4: a half is one 17-byte block; k * 2^(e - 128) in F32, as weight<39> forms it, then rounded
// to FP16 (exact unless the scale leaves FP16's range).
template<> struct MoeUnit<39> {
    static constexpr int span = 64, table_words = 0;
    uint32_t qs[4], e;
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int, size_t unit, unsigned half) {
        const uint8_t* b = row + (2 * unit + half) * 17;
        const moe_word4 q = *reinterpret_cast<const moe_word4_b*>(b + 1);
        #pragma unroll
        for (int i = 0; i < 4; ++i) qs[i] = q[i];
        e = b[0];
    }
    // Values 8p .. 8p + 7: low nibbles of code bytes 8 * (p % 2) .. for p < 2, high nibbles after.
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const float scale = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
        float v[8];
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            const uint32_t biased = gemv_fp4((qs[2 * (p % 2) + w] >> (4 * (p / 2))) & 0x0F0F0F0Fu);
            #pragma unroll
            for (int j = 0; j < 4; ++j) v[4 * w + j] = (float((biased >> (8 * j)) & 255u) - 12.0f) * scale;
        }
        moe_store8(v, out);
    }
};

// The shared expert's 256-value K-quant and IQ4_XS banks, gate and up in any mix of Q4_K, Q5_K, Q6_K
// and IQ4_XS: a unit loads the lane's half of a super-block's scales; each panel reads its values.
template<> struct MoeUnit<-1> {
    static constexpr int span = 256, table_words = 0;
    const uint8_t* block; int format; unsigned half; uint32_t d, scales, minimums;   // Q6_K: eight scales
    __device__ static void stage(uint32_t*) {}
    __device__ void load(const uint8_t* row, int kind, size_t unit, unsigned h) {
        format = kind; half = h;
        if (kind == 12 || kind == 13) {   // d, dmin, then twelve bytes of six-bit scales and minimums
            block = row + unit * (kind == 12 ? 144 : 176);
            const moe_word4 head = moe_u128(block);
            d = head[0];
            scales = h ? (head[3] & 0x0f0f0f0fu) | ((head[1] >> 2) & 0x30303030u) : head[1] & 0x3f3f3f3fu;
            minimums = h ? ((head[3] >> 4) & 0x0f0f0f0fu) | ((head[2] >> 2) & 0x30303030u) : head[2] & 0x3f3f3f3fu;
        } else if (kind == 14) {
            block = row + unit * 210;
            const uint2 s = moe_u64(block + 192 + 8 * h);
            scales = s.x; minimums = s.y; d = moe_u16(block + 208);
        } else {   // IQ4_XS, as MoeUnit<23>
            block = row + unit * 136;
            const uint32_t head = moe_u32(block);
            d = head & 0xffff;
            scales = ((moe_u32(block + 4) >> (16 * h)) & 0xffff) | (((head >> (16 + 8 * h)) & 255) << 16);
        }
    }
    // Values half * 128 + 8p .. + 7 of the super-block, each as weight<kind> forms it.
    __device__ void decode(int p, const uint32_t*, uint32_t (&out)[4]) const {
        const int kind = __builtin_amdgcn_readfirstlane(format);
        float v[8];
        if (kind == 12 || kind == 13) {
            const unsigned group = 4 * half + p / 4;
            const uint2 codes = moe_u64(block + (kind == 13 ? 48 : 16) + (2 * half + p / 8) * 32 + 8 * (p % 4));
            const uint2 high = kind == 13 ? moe_u64(block + 16 + 8 * (p % 4)) : uint2{0, 0};
            const float ds = gemv_half(d) * float((scales >> (8 * (p / 4))) & 255);
            const float dm = gemv_half(d >> 16) * float((minimums >> (8 * (p / 4))) & 255);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned code = ((j < 4 ? codes.x : codes.y) >> (8 * (j % 4) + 4 * ((p / 4) % 2))) & 15;
                code |= (((j < 4 ? high.x : high.y) >> (8 * (j % 4) + group)) & 1) << 4;
                v[j] = ds * float(code) - dm;
            }
        } else if (kind == 14) {
            const uint2 low = moe_u64(block + 64 * half + 8 * (p % 8)), high = moe_u64(block + 128 + 32 * half + 8 * (p % 4));
            const int scale = int8_t(((p < 8 ? scales : minimums) >> (8 * ((p / 2) % 4))) & 255);
            const float dd = gemv_half(d);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned l = ((j < 4 ? low.x : low.y) >> (8 * (j % 4) + 4 * (p / 8))) & 15;
                const unsigned h = ((j < 4 ? high.x : high.y) >> (8 * (j % 4) + 2 * (p / 4 % 4))) & 3;
                v[j] = dd * float(scale * (int(l | (h << 4)) - 32)) + 0.0f;   // exact; zero is +0 as weight<14> compiles
            }
        } else {
            const int i = p / 4, c = p % 4;
            const int scale = int(((scales >> (4 * i)) & 15) | (((scales >> (16 + 2 * i)) & 3) << 4)) - 32;
            const uint2 codes = moe_u64(block + 8 + (4 * half + i) * 16 + 8 * (c % 2));
            moe_nonlinear8(codes.x, codes.y, 4 * (c / 2), gemv_half(d) * float(scale), out);
            return;
        }
        moe_store8(v, out);
    }
};

// The block's tile: from the device list (`count`), the routed tiles skip the last `shared` (the
// shared expert's) and the shared launch takes them; without `count` the grid is the host's list.
__device__ __forceinline__ bool moe_tile(const MoeTile* tiles, const unsigned* count, unsigned shared, bool tail, MoeTile& tile) {
    unsigned index = blockIdx.y;
    if (count) {
        if (tail) index += *count - shared;
        else if (index + shared >= *count) return false;
    }
    tile = tiles[index];
    return true;
}

// Activations for one K step: chunk c (eight values) of a row holds half c & 1 of panel c >> 1, at LDS
// offset 8c, the order of the B fragments; thread t stages chunks t % 4 + 4j of row t / 4.
constexpr int moe_chunks = moe_block_k / 32;
template<int span>
__device__ __forceinline__ size_t moe_step_column(size_t step, int c) {
    constexpr size_t per_unit = span / moe_block_k;
    return (step / per_unit) * span + (c & 1) * (span / 2) + ((step % per_unit) * (moe_block_k / 16) + (c >> 1)) * 8;
}

// Each wave holds 16 * moe_wave_n weight rows: each lane decodes moe_wave_n fragments per panel and
// every activation fragment read from LDS serves moe_wave_n WMMAs.
constexpr int moe_wave_n = 2;

// One block: 64 routed rows of one expert x 64 * moe_wave_n product columns; wave w decodes gate
// (w < 4) or up rows (w % 4) * 16 * moe_wave_n .. straight into WMMA fragments for all 64 rows.
// Unfused (tests): the F32 gate and up sums go to `raw`, per route its gate then its up row.
// Nonzero H, F, K fix the shape's hidden, ffn and top_k at compile time (the K loop's trip count
// and the route-to-token division constant); zero takes them from `shape`.
// With `weighted` (DeepSeek-V4.1's launch), a positive `clamp` caps gate above and up on both sides
// first, and `route_weights` (one per grouped row) scale the product, as its SwiGLU does
// (op_swiglu_kernel). Qwen3.8's instantiations are without: compiled in, the clamp and the weight
// moved the FP16 products of its prefill (and its pinned generations with them).
// SoA only changes a specialized unit's fields: its span and the activation K order stay native.
template<int kind, bool allow_soa>
__device__ __forceinline__ void moe_load_unit(MoeUnit<kind>& unit, const Matrix& weights, size_t row, size_t index, unsigned half) {
    if constexpr (allow_soa && (kind == 16 || kind == 18 || kind == 21 || kind == 22)) {
        if (weights.layout == 1) {
            unit.load_soa(weights, row, index, half);
            return;
        }
    }
    unit.load(weights.data + row * weights.row_bytes, weights.kind, index, half);
}

template<int kind, bool fused = true, int H = 0, int F = 0, int K = 0, bool weighted = false, bool allow_soa = false>
__global__ void __launch_bounds__(256) moe_gate_up_wmma(const MoeGroup* groups, const MoeTile* tiles,
                                                        const int32_t* positions, size_t steps, MoeShape shape,
                                                        const _Float16* input, _Float16* product,
                                                        const unsigned* tile_count, unsigned shared, bool tail,
                                                        float* raw = nullptr, float clamp = 0.0f,
                                                        const float* route_weights = nullptr) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using Unit = MoeUnit<kind>;
    constexpr int L = moe_lds_k, N = moe_wave_n, P = moe_block_k / 16, per_unit = Unit::span / moe_block_k;
    constexpr int up_stride = 68;
    const size_t hidden = H ? H : shape.hidden, ffn = F ? F : shape.ffn, top_k = K ? K : shape.top_k;
    const int total = int(hidden) / moe_block_k;
    MoeTile tile;
    if (!moe_tile(tiles, tile_count, shared, tail, tile)) return;
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    // 16-row subtiles past the tile's rows take no WMMA.
    const int subtiles = int(std::min<size_t>(4, (count - tile.first + 15) / 16));
    const int t = threadIdx.x, lane = t % 32, wave = __builtin_amdgcn_readfirstlane(t / 32);
    const unsigned half = lane / 16;
    const Matrix& weights = group.weights[wave / 4];
    const size_t j0 = size_t(blockIdx.x) * 64 * N + (wave % 4) * 16 * N + lane % 16;
    __shared__ __attribute__((aligned(16))) union { _Float16 a[2][moe_block_m * L]; float up[moe_block_m * up_stride]; } lds;
    __shared__ __attribute__((aligned(16))) uint32_t table[Unit::table_words ? Unit::table_words : 1];
    Unit::stage(table);
    const int arow = t / 4, chunk = t % 4, aoffset = arow * L + 8 * chunk;
    const _Float16* source = nullptr;
    if (tile.first + arow < count) {
        const size_t route = size_t(positions[first + tile.first + arow]);
        const size_t token = route < steps * top_k ? route / top_k : route - steps * top_k;
        source = input + token * hidden;
    }
    uint4 a[moe_chunks];
    auto load_a = [&](size_t step) {
        #pragma unroll
        for (int j = 0; j < moe_chunks; ++j)
            a[j] = source ? *reinterpret_cast<const uint4*>(source + moe_step_column<Unit::span>(step, chunk + 4 * j)) : uint4{0, 0, 0, 0};
    };
    auto store_a = [&](int buffer) {
        #pragma unroll
        for (int j = 0; j < moe_chunks; ++j) *reinterpret_cast<uint4*>(&lds.a[buffer][aoffset + 32 * j]) = a[j];
    };
    Unit next[N], unit[N];
    #pragma unroll
    for (int n = 0; n < N; ++n) moe_load_unit<kind, allow_soa>(next[n], weights, j0 + 16 * n, 0, half);
    load_a(0); store_a(0);
    __syncthreads();
    moe_float8 sum[4][N] = {};
    for (int u = 0; u < total / per_unit; ++u) {
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            unit[n] = next[n];
            if (u + 1 < total / per_unit) moe_load_unit<kind, allow_soa>(next[n], weights, j0 + 16 * n, u + 1, half);
        }
        #pragma unroll
        for (int s = 0; s < per_unit; ++s) {
            const int step = u * per_unit + s, buffer = step & 1;
            if (step + 1 < total) load_a(step + 1);
            #pragma unroll
            for (int q = 0; q < P; ++q) {
                moe_half8 b[N];
                #pragma unroll
                for (int n = 0; n < N; ++n) {
                    uint32_t w[4];
                    unit[n].decode(P * s + q, table, w);
                    b[n] = __builtin_bit_cast(moe_half8, moe_word4{w[0], w[1], w[2], w[3]});
                }
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m >= subtiles) break;
                    const moe_half8 x = *reinterpret_cast<const moe_half8*>(&lds.a[buffer][(m * 16 + lane % 16) * L + 16 * q + 8 * half]);
                    #pragma unroll
                    for (int n = 0; n < N; ++n) sum[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(x, b[n], sum[m][n]);
                }
            }
            if (step + 1 < total) store_a(buffer ^ 1);
            __syncthreads();
        }
    }
    if constexpr (!fused) {
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const size_t row = tile.first + m * 16 + half * 8 + i;
                if (row >= count) continue;
                float* target = raw + size_t(positions[first + row]) * 2 * ffn + (wave / 4) * ffn + j0;
                #pragma unroll
                for (int n = 0; n < N; ++n) target[16 * n] = sum[m][n][i];
            }
        return;
    }
    // The gate and up waves of the same columns swap half their rows through LDS (gate rows 32-63,
    // up rows 0-31), a subtile column at a time: each finishes SiLU(gate)*up for its own half.
    const int own = wave < 4 ? 0 : 2;
    float* const mine = lds.up + (wave < 4 ? 32 : 0) * up_stride;
    const float* const theirs = lds.up + (wave < 4 ? 0 : 32) * up_stride;
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                mine[(m * 16 + half * 8 + i) * up_stride + (wave % 4) * 16 + lane % 16] = wave < 4 ? sum[2 + m][n][i] : sum[m][n][i];
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const size_t row = tile.first + (own + m) * 16 + half * 8 + i;
                if (row < count) {
                    const float partner = theirs[(m * 16 + half * 8 + i) * up_stride + (wave % 4) * 16 + lane % 16];
                    const float value = wave < 4 ? sum[m][n][i] : sum[2 + m][n][i];
                    if constexpr (weighted) {
                        float g = wave < 4 ? value : partner, up = wave < 4 ? partner : value;
                        if (clamp > 0.0f) { g = fminf(g, clamp); up = fminf(fmaxf(up, -clamp), clamp); }
                        const float scale = route_weights ? route_weights[first + row] : 1.0f;
                        product[(first + row) * ffn + j0 + 16 * n] = _Float16(g * hc_sigmoid(g) * up * scale);
                    } else {
                        const float g = wave < 4 ? value : partner, up = wave < 4 ? partner : value;
                        product[(first + row) * ffn + j0 + 16 * n] = _Float16(g * hc_sigmoid(g) * up);
                    }
                }
            }
        if (n + 1 < N) __syncthreads();
    }
#endif
}

// One block: 64 grouped product rows x 128 * moe_wave_n output columns; results scatter to routes
// (without `positions`, each to its grouped row), rows `out_stride` values apart (zero: hidden).
template<int kind, int H = 0, int F = 0>
__global__ void __launch_bounds__(256) moe_down_wmma(const MoeGroup* groups, const MoeTile* tiles,
                                                     const int32_t* positions, MoeShape shape, const _Float16* product,
                                                     float* outputs, const unsigned* tile_count, unsigned shared, bool tail,
                                                     size_t out_stride = 0) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using Unit = MoeUnit<kind>;
    constexpr int L = moe_lds_k, N = moe_wave_n, P = moe_block_k / 16, per_unit = Unit::span / moe_block_k;
    const size_t hidden = H ? H : shape.hidden, ffn = F ? F : shape.ffn;
    const int total = int(ffn) / moe_block_k;
    MoeTile tile;
    if (!moe_tile(tiles, tile_count, shared, tail, tile)) return;
    const MoeGroup& group = groups[tile.group];
    const size_t first = group.first, count = group.count;
    const int subtiles = int(std::min<size_t>(4, (count - tile.first + 15) / 16));
    const Matrix& down = group.weights[2];
    const int t = threadIdx.x, lane = t % 32, wave = __builtin_amdgcn_readfirstlane(t / 32);
    const unsigned half = lane / 16;
    const size_t n0 = size_t(blockIdx.x) * 128 * N + wave * 16 * N + lane % 16;
    __shared__ __attribute__((aligned(16))) _Float16 as[2][moe_block_m * L];
    __shared__ __attribute__((aligned(16))) uint32_t table[Unit::table_words ? Unit::table_words : 1];
    Unit::stage(table);
    const int arow = t / 4, chunk = t % 4, aoffset = arow * L + 8 * chunk;
    const _Float16* source = tile.first + arow < count ? product + (first + tile.first + arow) * ffn : nullptr;
    uint4 a[moe_chunks];
    auto load_a = [&](size_t step) {
        #pragma unroll
        for (int j = 0; j < moe_chunks; ++j)
            a[j] = source ? *reinterpret_cast<const uint4*>(source + moe_step_column<Unit::span>(step, chunk + 4 * j)) : uint4{0, 0, 0, 0};
    };
    auto store_a = [&](int buffer) {
        #pragma unroll
        for (int j = 0; j < moe_chunks; ++j) *reinterpret_cast<uint4*>(&as[buffer][aoffset + 32 * j]) = a[j];
    };
    Unit next[N], unit[N];
    #pragma unroll
    for (int n = 0; n < N; ++n) next[n].load(down.data + (n0 + 16 * n) * down.row_bytes, down.kind, 0, half);
    load_a(0); store_a(0);
    __syncthreads();
    moe_float8 sum[4][N] = {};
    for (int u = 0; u < total / per_unit; ++u) {
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            unit[n] = next[n];
            if (u + 1 < total / per_unit) next[n].load(down.data + (n0 + 16 * n) * down.row_bytes, down.kind, u + 1, half);
        }
        #pragma unroll
        for (int s = 0; s < per_unit; ++s) {
            const int step = u * per_unit + s, buffer = step & 1;
            if (step + 1 < total) load_a(step + 1);
            #pragma unroll
            for (int q = 0; q < P; ++q) {
                moe_half8 b[N];
                #pragma unroll
                for (int n = 0; n < N; ++n) {
                    uint32_t w[4];
                    unit[n].decode(P * s + q, table, w);
                    b[n] = __builtin_bit_cast(moe_half8, moe_word4{w[0], w[1], w[2], w[3]});
                }
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (m >= subtiles) break;
                    const moe_half8 x = *reinterpret_cast<const moe_half8*>(&as[buffer][(m * 16 + lane % 16) * L + 16 * q + 8 * half]);
                    #pragma unroll
                    for (int n = 0; n < N; ++n) sum[m][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(x, b[n], sum[m][n]);
                }
            }
            if (step + 1 < total) store_a(buffer ^ 1);
            __syncthreads();
        }
    }
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t r = tile.first + m * 16 + half * 8 + i;
            if (r >= count) continue;
            const size_t out_row = positions ? size_t(positions[first + r]) : first + r;
            float* target = outputs + out_row * (out_stride ? out_stride : hidden) + n0;
            #pragma unroll
            for (int n = 0; n < N; ++n) target[16 * n] = sum[m][n][i];
        }
#endif
}

// The input is reused by all routes and column tiles: round it to FP16 once (as moe_gemm_cuda.hpp).
__global__ void moe_input_half(size_t values, const float* input, _Float16* output) {
    const size_t index = (size_t(blockIdx.x) * 256 + threadIdx.x) * 4;
    if (index >= values) return;   // values = steps * hidden, always divisible by four
    const float4 v = reinterpret_cast<const float4*>(input)[index / 4];
    reinterpret_cast<uint2*>(output)[index / 4] = uint2{moe_pack(v.x, v.y), moe_pack(v.z, v.w)};
}

// Formats with a unit of their own, used where the product's K is a multiple of the unit's span; mixes
// of the K-quant formats take unit -1, any other format the generic one (0).
__host__ __device__ constexpr int moe_unit_span(int kind) { return kind == 20 || kind == 8 || kind == 39 ? 64 : kind == 42 ? 128 : 256; }
__host__ __device__ constexpr int moe_unit_kind(int kind, size_t k) {
    return (kind == 21 || kind == 18 || kind == 22 || kind == 16 || kind == 17 || kind == 29 || kind == 23 || kind == 20 ||
            kind == 8 || kind == 42 || kind == 19 || kind == 10 || kind == 39) && k % moe_unit_span(kind) == 0 ? kind : 0;
}
struct MoeKinds { int gate, up, down; };
// Gate/up then down over a tile list of `listed` tiles (`count` on the device, which the grids'
// `listed` blocks cover; null for a host list): routed experts with their banks' formats and, when
// its formats differ, the shared expert (the last `shared` tiles) with its own.
// Whether moe_expert_launch takes this shape: whole tiles of product and output columns, whole K steps.
inline bool moe_shape_fits(MoeShape shape) {
    return shape.hidden && shape.ffn && shape.top_k && shape.ffn % (64 * moe_wave_n) == 0 &&
           shape.hidden % (128 * moe_wave_n) == 0 && shape.hidden % moe_block_k == 0 && shape.ffn % moe_block_k == 0;
}
// The units of a gate/up pair and of a down matrix at a shape.
inline int moe_gate_up_unit(int gate, int up, MoeShape shape) {
    auto k_quant = [](int kind) { return kind == 12 || kind == 13 || kind == 14 || kind == 23; };
    return gate == up && moe_unit_kind(gate, shape.hidden) ? gate
         : k_quant(gate) && k_quant(up) && shape.hidden % 256 == 0 ? -1 : 0;
}
inline int moe_down_unit(int down, MoeShape shape) {
    const bool k_quant = down == 12 || down == 13 || down == 14 || down == 23;
    return moe_unit_kind(down, shape.ffn) ? down : k_quant && shape.ffn % 256 == 0 ? -1 : 0;
}
#define SD_MOE_UNITS(X) X(21) X(18) X(22) X(16) X(17) X(29) X(19) X(10) X(23) X(20) X(8) X(39) X(42) X(-1) X(0)
// H, F, K: as moe_gate_up_wmma's (a caller with one fixed shape passes it).
template<int H = 0, int F = 0, int K = 0, bool allow_soa = false>
inline void moe_expert_launch(hipStream_t stream, MoeKinds routed, MoeKinds common, const MoeGroup* groups, const MoeTile* tiles,
                              const int32_t* positions, size_t steps, MoeShape shape, const _Float16* input, _Float16* product,
                              float* outputs, const unsigned* count, unsigned listed, unsigned shared) {
    const MoeTile* last = count ? tiles : tiles + (listed - shared);
    const unsigned gate_columns = shape.ffn / (64 * moe_wave_n), down_columns = shape.hidden / (128 * moe_wave_n);
    auto gate_up = [&](int kind, const MoeTile* list, unsigned blocks, unsigned skip, bool tail) {
        #define SD_GATE_UP(format) case format: moe_gate_up_wmma<format, true, H, F, K, false, allow_soa><<<dim3(gate_columns, blocks), 256, 0, stream>>>( \
            groups, list, positions, steps, shape, input, product, count, skip, tail); break;
        if (blocks) switch (kind) { SD_MOE_UNITS(SD_GATE_UP) }
        #undef SD_GATE_UP
    };
    auto down = [&](int kind, const MoeTile* list, unsigned blocks, unsigned skip, bool tail) {
        #define SD_DOWN(format) case format: moe_down_wmma<format, H, F><<<dim3(down_columns, blocks), 256, 0, stream>>>( \
            groups, list, positions, shape, product, outputs, count, skip, tail); break;
        if (blocks) switch (kind) { SD_MOE_UNITS(SD_DOWN) }
        #undef SD_DOWN
    };
    const int routed_gate = moe_gate_up_unit(routed.gate, routed.up, shape), shared_gate = moe_gate_up_unit(common.gate, common.up, shape);
    const int routed_down = moe_down_unit(routed.down, shape), shared_down = moe_down_unit(common.down, shape);
    if (shared && shared_gate != routed_gate) {
        gate_up(routed_gate, tiles, listed - shared, shared, false);
        gate_up(shared_gate, last, shared, shared, true);
    } else gate_up(routed_gate, tiles, listed, 0, false);
    if (shared && shared_down != routed_down) {
        down(routed_down, tiles, listed - shared, shared, false);
        down(shared_down, last, shared, shared, true);
    } else down(routed_down, tiles, listed, 0, false);
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
