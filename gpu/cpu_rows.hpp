// Fused multi-row kernels for the block-decoded CPU expert formats (included by
// cpu_experts.cpp on x86). Every row keeps the decoded path's arithmetic exactly:
// each value is the block decoder's F32 value, one 16-lane accumulator per row takes
// acc[k] = fma(value, x, acc[k]) in order, and reduce16 sums the lanes. Lattice rows
// come from F32 copies of the gguf grids (no gathers or integer conversions), and up
// to four rows run together so their accumulation chains overlap and share the
// activations.

// (0.5 + nibble) * 0.25: a half's 11 bits times these five are exact, so a chunk's scale is one product, without a conversion.
constexpr float iq2_level[16] = {0.125f, 0.375f, 0.625f, 0.875f, 1.125f, 1.375f, 1.625f, 1.875f, 2.125f, 2.375f, 2.625f, 2.875f, 3.125f, 3.375f, 3.625f, 3.875f};
// Sixteen values of a lattice format: two 8-value grid rows (IQ1, IQ2) or four
// 4-value rows (IQ3), one scale, sixteen sign bits (IQ1: two deltas instead).
struct GridChunk {
    const float* row[4];
    float scale;
    uint32_t signs;
    float delta[2];
};

template<int kind> struct Block;
template<> struct Block<16> {   // IQ2_XXS
    static constexpr size_t values = 256, bytes = 66;
    static constexpr int grid = 16, width = 8;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, l = 2 * (c % 2);
        const uint32_t aux = load32(b + 2 + 8 * ib + 4);
        GridChunk k;
        k.row[0] = table + 8 * b[2 + 8 * ib + l]; k.row[1] = table + 8 * b[3 + 8 * ib + l];
        k.scale = d * iq2_level[aux >> 28];
        k.signs = iq_signs_table((aux >> (7 * l)) & 127) | iq_signs_table((aux >> (7 * l + 7)) & 127) << 8;
        return k;
    }
};
template<> struct Block<17> {   // IQ2_XS
    static constexpr size_t values = 256, bytes = 74;
    static constexpr int grid = 17, width = 8;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, h = c % 2, g = 4 * ib + 2 * h;
        const unsigned q0 = unsigned(b[2 + 2 * g]) | unsigned(b[3 + 2 * g]) << 8;
        const unsigned q1 = unsigned(b[4 + 2 * g]) | unsigned(b[5 + 2 * g]) << 8;
        GridChunk k;
        k.row[0] = table + 8 * (q0 & 511); k.row[1] = table + 8 * (q1 & 511);
        k.scale = d * iq2_level[(b[66 + ib] >> (4 * h)) & 15];
        k.signs = iq_signs_table(q0 >> 9) | iq_signs_table(q1 >> 9) << 8;
        return k;
    }
};
template<> struct Block<22> {   // IQ2_S
    static constexpr size_t values = 256, bytes = 82;
    static constexpr int grid = 22, width = 8;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, h = c % 2, l = 2 * h;
        GridChunk k;
        for (unsigned i = 0; i < 2; ++i)
            k.row[i] = table + 8 * (b[2 + 4 * ib + l + i] | ((unsigned(b[66 + ib]) << (8 - 2 * (l + i))) & 0x300));
        k.scale = d * iq2_level[(b[74 + ib] >> (4 * h)) & 15];
        k.signs = unsigned(b[34 + 4 * ib + l]) | unsigned(b[35 + 4 * ib + l]) << 8;
        return k;
    }
};
template<> struct Block<18> {   // IQ3_XXS
    static constexpr size_t values = 256, bytes = 98;
    static constexpr int grid = 18, width = 4;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, l = 2 * (c % 2);
        const uint32_t aux = load32(b + 66 + 4 * ib);
        GridChunk k;
        for (unsigned t = 0; t < 4; ++t) k.row[t] = table + 4 * b[2 + 8 * ib + 2 * l + t];
        k.scale = d * (0.5f + float(aux >> 28)) * 0.5f;
        k.signs = iq_signs_table((aux >> (7 * l)) & 127) | iq_signs_table((aux >> (7 * l + 7)) & 127) << 8;
        return k;
    }
};
template<> struct Block<21> {   // IQ3_S
    static constexpr size_t values = 256, bytes = 110;
    static constexpr int grid = 21, width = 4;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, l = 2 * (c % 2);
        GridChunk k;
        for (unsigned t = 0; t < 4; ++t) {
            const unsigned r = 2 * l + t;
            k.row[t] = table + 4 * (b[2 + 8 * ib + r] | ((unsigned(b[66 + ib]) << (8 - r)) & 256));
        }
        k.scale = d * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        k.signs = unsigned(b[74 + 4 * ib + l]) | unsigned(b[75 + 4 * ib + l]) << 8;
        return k;
    }
};
template<> struct Block<29> {   // IQ1_M: value = scale * ((grid value) + delta), no signs
    static constexpr size_t values = 256, bytes = 56;
    static constexpr int grid = 29, width = 8;
    const uint8_t* b; float d; uint16_t words[4];
    void load(const uint8_t* block) { b = block; d = iq1m_scale(block, words); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, h = c % 2;
        const unsigned high = b[32 + 2 * ib + h];
        GridChunk k;
        for (unsigned i = 0; i < 2; ++i) {
            const unsigned nibble = high >> (4 * i);
            k.row[i] = table + 8 * (b[4 * ib + 2 * h + i] | ((nibble & 7) << 8));
            k.delta[i] = iq1_delta(nibble);
        }
        k.scale = iq1m_group_scale(d, words, int(ib), int(h));
        return k;
    }
};
template<> struct Block<19> {   // IQ1_S: value = scale * (grid value + delta), one scale and delta per 32 values
    static constexpr size_t values = 256, bytes = 50;
    static constexpr int grid = 19, width = 8;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
    GridChunk chunk(unsigned c, const float* table) const {
        const unsigned ib = c / 2, h = c % 2;
        const unsigned qh = unsigned(b[34 + 2 * ib]) | unsigned(b[35 + 2 * ib]) << 8;
        GridChunk k;
        for (unsigned i = 0; i < 2; ++i) {
            const unsigned l = 2 * h + i;
            k.row[i] = table + 8 * (b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8));
            k.delta[i] = qh & 0x8000 ? -0.125f : 0.125f;
        }
        k.scale = d * float(2 * ((qh >> 12) & 7) + 1);
        return k;
    }
};
// Code formats: sixteen values are the scale times a table entry or an int8, Q2_K's scale times
// a code minus its minimum.
template<> struct Block<20> {   // IQ4_NL
    static constexpr size_t values = 32, bytes = 18;
    static constexpr int grid = 0;
    const uint8_t* b; float d;
    SD_TARGET("f16c") void load(const uint8_t* block) { b = block; uint16_t bits; std::memcpy(&bits, block, 2); d = SD_HALF_TO_FLOAT(bits); }
    const uint8_t* codes(unsigned) const { return b + 2; }
    int shift(unsigned c) const { return 4 * int(c); }
    float scale(unsigned) const { return d; }
};
template<> struct Block<23> {   // IQ4_XS
    static constexpr size_t values = 256, bytes = 136;
    static constexpr int grid = 0;
    const uint8_t* b; float d; unsigned high;
    void load(const uint8_t* block) { b = block; d = half_at(block); high = unsigned(block[2]) | unsigned(block[3]) << 8; }
    const uint8_t* codes(unsigned c) const { return b + 8 + 16 * (c / 2); }
    int shift(unsigned c) const { return 4 * int(c % 2); }
    float scale(unsigned c) const {
        const unsigned g = c / 2;
        return d * float(int(((b[4 + g / 2] >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4)) - 32);
    }
};
template<> struct Block<8> {    // Q8_0
    static constexpr size_t values = 32, bytes = 34;
    static constexpr int grid = -1;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = half_at(block); }
};
template<> struct Block<10> {   // Q2_K: group c's scale byte is c; its codes as decode_q2_k
    static constexpr size_t values = 256, bytes = 84;
    static constexpr int grid = -2;
    const uint8_t* b; float d, minimum;
    void load(const uint8_t* block) { b = block; d = half_at(block + 80); minimum = half_at(block + 82); }
    const uint8_t* codes(unsigned c) const { return b + 16 + 32 * (c / 8) + 16 * (c % 2); }
    int shift(unsigned c) const { return 2 * int((c / 2) % 4); }
    float scale(unsigned c) const { return d * float(b[c] & 15); }
    float offset(unsigned c) const { return minimum * float(b[c] >> 4); }
};
template<> struct Block<39> {   // MXFP4: doubled E2M1 values times 2^(e - 128)
    static constexpr size_t values = 32, bytes = 17;
    static constexpr int grid = -3;
    const uint8_t* b; float d;
    void load(const uint8_t* block) { b = block; d = e8m0_half(block[0]); }
    const uint8_t* codes(unsigned) const { return b + 1; }
    int shift(unsigned c) const { return 4 * int(c); }
};

SD_TARGET("avx512f,avx2,f16c")
inline __m512 grid16(const GridChunk& k, int width) {
    if (width == 8) {
        const __m256d low = _mm256_castps_pd(_mm256_loadu_ps(k.row[0])), high = _mm256_castps_pd(_mm256_loadu_ps(k.row[1]));
        return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(low), high, 1));
    }
    __m512 v = _mm512_castps128_ps512(_mm_loadu_ps(k.row[0]));
    v = _mm512_insertf32x4(v, _mm_loadu_ps(k.row[1]), 1);
    v = _mm512_insertf32x4(v, _mm_loadu_ps(k.row[2]), 2);
    return _mm512_insertf32x4(v, _mm_loadu_ps(k.row[3]), 3);
}
template<int kind>
SD_TARGET("avx512f,avx2,f16c")
inline __m512 values16(const Block<kind>& block, unsigned c, const float* table) {
    using B = Block<kind>;
    if constexpr (B::grid > 0) {
        const GridChunk k = block.chunk(c, table);
        __m512 v = grid16(k, B::width);
        if constexpr (kind == 29 || kind == 19) {
            const __m512 delta = _mm512_mask_blend_ps(__mmask16(0xff00), _mm512_set1_ps(k.delta[0]), _mm512_set1_ps(k.delta[1]));
            return _mm512_mul_ps(_mm512_set1_ps(k.scale), _mm512_add_ps(v, delta));
        } else {
            const __m512i bits = _mm512_castps_si512(_mm512_mul_ps(v, _mm512_set1_ps(k.scale)));
            return _mm512_castsi512_ps(_mm512_mask_xor_epi32(bits, __mmask16(k.signs), bits, _mm512_set1_epi32(int(0x80000000u))));
        }
    } else if constexpr (B::grid == 0) {
        const __m512 table16 = _mm512_setr_ps(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes(c)));
        // (the permute reads each lane's low four bits alone)
        const __m512i codes = _mm512_srli_epi32(_mm512_cvtepu8_epi32(bytes), unsigned(block.shift(c)));
        return _mm512_mul_ps(_mm512_set1_ps(block.scale(c)), _mm512_permutexvar_ps(codes, table16));
    } else if constexpr (B::grid == -2) {
        const __m512i bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes(c))));
        const __m512i codes = _mm512_and_si512(_mm512_srli_epi32(bytes, unsigned(block.shift(c))), _mm512_set1_epi32(3));
        return _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(block.scale(c)), _mm512_cvtepi32_ps(codes)), _mm512_set1_ps(block.offset(c)));
    } else if constexpr (B::grid == -3) {
        const __m512 fp4 = _mm512_setr_ps(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        const __m512i bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes(c))));
        const __m512i codes = _mm512_and_si512(_mm512_srli_epi32(bytes, unsigned(block.shift(c))), _mm512_set1_epi32(15));
        return _mm512_mul_ps(_mm512_permutexvar_ps(codes, fp4), _mm512_set1_ps(block.d));
    } else {
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.b + 2 + 16 * c));
        return _mm512_mul_ps(_mm512_set1_ps(block.d), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)));
    }
}
// Rows of an expert are contiguous: ask for the next N rows' block b while this one computes.
template<size_t bytes, int N>
inline void prefetch_next(const uint8_t* row, size_t stride, size_t b) {
    for (int r = 0; r < N; ++r)
        for (size_t line = 0; line < bytes; line += 64)
            _mm_prefetch(reinterpret_cast<const char*>(row + size_t(N + r) * stride + b * bytes + line), _MM_HINT_T0);
}
// AVX-512 lattice blocks: one pass over a 256-value block computes, sixteen lanes at a
// time, every grid row's byte offset in the F32 table, every sixteen values' scale (the
// same expression and rounding as Block::chunk), the sign bytes and IQ1 deltas, so the
// per-chunk work is loads, one multiply, a sign flip and the FMA.
struct Prepared {
    alignas(64) uint16_t offset[64];   // grid rows: 32 (IQ1, IQ2) or 64 (IQ3); the largest is 2047 * 32
    alignas(64) float scale[16];
    alignas(64) float delta[32];       // IQ1_M
    alignas(64) uint8_t sign[32];
    const uint8_t* signs;              // the block's own sign bytes (IQ2_S, IQ3_S), or its qh bytes (IQ1_M), else sign
};
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512i lanes16(int first = 0) { return _mm512_add_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(first)); }
// Bytes at p (16 of them) widened to sixteen 32-bit lanes.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512i bytes16(const uint8_t* p) { return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }
// Lane i takes (source[select_i] >> shift_i) & mask.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512i field16(__m512i source, __m512i select, __m512i shift, int mask) {
    return _mm512_and_si512(_mm512_srlv_epi32(_mm512_permutexvar_epi32(select, source), shift), _mm512_set1_epi32(mask));
}
// Seven sign bits plus the parity bit, sixteen groups at a time (as iq_signs).
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512i signs16(__m512i bits) {
    __m512i parity = _mm512_xor_si512(bits, _mm512_srli_epi32(bits, 4));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi32(parity, 2));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi32(parity, 1));
    return _mm512_or_si512(bits, _mm512_slli_epi32(_mm512_and_si512(parity, _mm512_set1_epi32(1)), 7));
}
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline void store_signs(uint8_t* out, __m512i low, __m512i high) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm512_cvtepi32_epi8(low));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16), _mm512_cvtepi32_epi8(high));
}
// A block's scale by the instruction: half_value's branches cost more than the rest of a scale.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline float wide_half(const uint8_t* p) {
    uint16_t bits;
    std::memcpy(&bits, p, 2);
    return SD_HALF_TO_FLOAT(bits);
}
// (0.5 + nibble) * 0.25: a half's 11 bits times these five are exact, so the order of the products does not round.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512 iq2_levels() {
    return _mm512_setr_ps(0.125f, 0.375f, 0.625f, 0.875f, 1.125f, 1.375f, 1.625f, 1.875f, 2.125f, 2.375f, 2.625f, 2.875f, 3.125f, 3.375f, 3.625f, 3.875f);
}
// Sixteen rows' grid indices as byte offsets in the table, an entry being 1 << shift bytes.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline void store_offsets(uint16_t* out, __m512i index, int shift) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(out), _mm512_cvtepi32_epi16(_mm512_slli_epi32(index, unsigned(shift))));
}
// IQ2 scale per sixteen values c: d * (0.5 + nibble) * 0.25.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline void iq2_scales(float d, __m512i nibbles, float* out) {
    const __m512 sum = _mm512_add_ps(_mm512_set1_ps(0.5f), _mm512_cvtepi32_ps(nibbles));
    _mm512_store_ps(out, _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(d), sum), _mm512_set1_ps(0.25f)));
}
// Thirty-two 7-bit sign fields as bytes, the top bit the parity of the seven (as iq_signs), nibble by nibble.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m256i signs32(__m256i fields) {
    const __m256i odd = _mm256_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0,
                                         0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
    const __m256i low = _mm256_and_si256(fields, _mm256_set1_epi8(15)), high = _mm256_and_si256(_mm256_srli_epi32(fields, 4), _mm256_set1_epi8(15));
    return _mm256_or_si256(fields, _mm256_xor_si256(_mm256_shuffle_epi8(odd, low), _mm256_shuffle_epi8(odd, high)));
}
// Each aux word's four 7-bit fields as its four bytes.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m256i aux_fields(__m256i aux) {
    const __m256i pairs = _mm256_or_si256(_mm256_and_si256(aux, _mm256_set1_epi32(0x3FFF)), _mm256_and_si256(_mm256_slli_epi32(aux, 2), _mm256_set1_epi32(0x3FFF0000)));
    return _mm256_or_si256(_mm256_and_si256(pairs, _mm256_set1_epi32(0x007F007F)), _mm256_and_si256(_mm256_slli_epi32(pairs, 1), _mm256_set1_epi32(0x7F007F00)));
}
// The sixteen nibbles of eight bytes, low nibble first, as bytes.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m128i nibbles16(const uint8_t* p) {
    const __m128i n = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)), low = _mm_set1_epi8(15);
    return _mm_unpacklo_epi8(_mm_and_si128(n, low), _mm_and_si128(_mm_srli_epi16(n, 4), low));
}
template<int kind>
SD_TARGET("avx512f,avx512bw,avx2,f16c")
void prepare(const uint8_t* b, Prepared& p) {
    if constexpr (kind == 22) {   // IQ2_S: qs[2..34) | 2 bits of qh[66 + ib]; scales nibbles at 74; signs at 34
        // Word g takes qh[g / 4] in both bytes, then the shift brings its pair of bits to bits 8 and 9.
        const __m512i qh = _mm512_broadcastq_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 66)));
        const __m512i spread = _mm512_setr_epi32(0x00000000, 0x00000000, 0x01010101, 0x01010101, 0x02020202, 0x02020202, 0x03030303, 0x03030303,
                                                 0x04040404, 0x04040404, 0x05050505, 0x05050505, 0x06060606, 0x06060606, 0x07070707, 0x07070707);
        const __m512i shifts = _mm512_set1_epi64(0x0006000400020000);
        const __m512i high = _mm512_srlv_epi16(_mm512_shuffle_epi8(qh, spread), shifts);
        const __m512i index = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2))),
                                              _mm512_and_si512(high, _mm512_set1_epi16(0x300)));
        _mm512_store_si512(p.offset, _mm512_slli_epi16(index, 5));
        const __m128i n = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 74)), low = _mm_set1_epi8(15);
        const __m128i nibbles = _mm_unpacklo_epi8(_mm_and_si128(n, low), _mm_and_si128(_mm_srli_epi16(n, 4), low));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(wide_half(b)), _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(nibbles), iq2_levels())));
        p.signs = b + 34;
    } else if constexpr (kind == 16) {   // IQ2_XXS: per ib, four index bytes then aux (four 7-bit signs, the scale's nibble)
        const __m512i words = _mm512_loadu_si512(reinterpret_cast<const void*>(b + 2));
        const __m512i split = _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15), words);
        _mm512_store_si512(p.offset, _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(split)), 5));
        // Each aux's four fields to four bytes; the sign byte's top bit is the parity of its seven, nibble by nibble.
        const __m256i aux = _mm512_extracti64x4_epi64(split, 1);
        const __m256i pairs = _mm256_or_si256(_mm256_and_si256(aux, _mm256_set1_epi32(0x3FFF)), _mm256_and_si256(_mm256_slli_epi32(aux, 2), _mm256_set1_epi32(0x3FFF0000)));
        const __m256i fields = _mm256_or_si256(_mm256_and_si256(pairs, _mm256_set1_epi32(0x007F007F)), _mm256_and_si256(_mm256_slli_epi32(pairs, 1), _mm256_set1_epi32(0x7F007F00)));
        const __m256i odd = _mm256_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0,
                                             0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
        const __m256i low = _mm256_and_si256(fields, _mm256_set1_epi8(15)), high = _mm256_and_si256(_mm256_srli_epi32(fields, 4), _mm256_set1_epi8(15));
        const __m256i parity = _mm256_xor_si256(_mm256_shuffle_epi8(odd, low), _mm256_shuffle_epi8(odd, high));
        _mm256_store_si256(reinterpret_cast<__m256i*>(p.sign), _mm256_or_si256(fields, parity));
        const __m512i nibbles = _mm512_shuffle_epi32(_mm512_srli_epi32(words, 28), _MM_PERM_DDBB);
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(wide_half(b)), _mm512_permutexvar_ps(nibbles, iq2_levels())));
        p.signs = p.sign;
    } else if constexpr (kind == 17) {   // IQ2_XS: 16-bit q per group (9-bit index, 7 sign bits); scales at 66
        const __m512i q = _mm512_loadu_si512(reinterpret_cast<const void*>(b + 2));
        _mm512_store_si512(p.offset, _mm512_slli_epi16(_mm512_and_si512(q, _mm512_set1_epi16(511)), 5));
        _mm256_store_si256(reinterpret_cast<__m256i*>(p.sign), signs32(_mm512_cvtepi16_epi8(_mm512_srli_epi16(q, 9))));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(wide_half(b)), _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(nibbles16(b + 66)), iq2_levels())));
        p.signs = p.sign;
    } else if constexpr (kind == 29) {   // IQ1_M: qs[0..32) | 3 bits of a qh nibble; nibble bit 3 picks the delta, read per chunk
        const __m128i qh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 32)), mask = _mm_set1_epi8(7);
        const __m128i low = _mm_and_si128(qh, mask), high = _mm_and_si128(_mm_srli_epi16(qh, 4), mask);
        const __m256i nibbles = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(low, high)), _mm_unpackhi_epi8(low, high), 1);
        const __m512i index = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b))),
                                              _mm512_slli_epi16(_mm512_cvtepu8_epi16(nibbles), 8));
        _mm512_store_si512(p.offset, _mm512_slli_epi16(index, 5));
        uint16_t words[4];
        std::memcpy(words, b + 48, 8);
        const float d = SD_HALF_TO_FLOAT(uint16_t((words[0] >> 12) | ((words[1] >> 8) & 0xf0) | ((words[2] >> 4) & 0xf00) | (words[3] & 0xf000)));
        const __m512i word = _mm512_cvtepu16_epi32(_mm256_castsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(words))));
        const __m512i lane = lanes16();
        const __m512i ib = _mm512_srli_epi32(lane, 1);
        const __m512i shift = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_and_si512(ib, _mm512_set1_epi32(1)), _mm512_set1_epi32(6)),
                                               _mm512_mullo_epi32(_mm512_and_si512(lane, _mm512_set1_epi32(1)), _mm512_set1_epi32(3)));
        const __m512i odd = _mm512_or_si512(_mm512_slli_epi32(field16(word, _mm512_srli_epi32(lane, 2), shift, 7), 1), _mm512_set1_epi32(1));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(odd)));
        p.signs = b + 32;
    } else if constexpr (kind == 19) {   // IQ1_S: qs[2..34) | 3 bits of the ib's 16-bit word at 34; its scale and delta sign
        const __m512i lane = lanes16();
        const __m512i qh = _mm512_cvtepu16_epi32(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 34))));
        for (int half = 0; half < 2; ++half) {
            const __m512i g = lanes16(16 * half), ib = _mm512_srli_epi32(g, 2);
            const __m512i high = field16(qh, ib, _mm512_mullo_epi32(_mm512_and_si512(g, _mm512_set1_epi32(3)), _mm512_set1_epi32(3)), 7);
            const __m512i index = _mm512_or_si512(bytes16(b + 2 + 16 * half), _mm512_slli_epi32(high, 8));
            store_offsets(p.offset + 16 * half, index, 5);
            const __m512i delta = _mm512_or_si512(_mm512_set1_epi32(0x3E000000), _mm512_slli_epi32(field16(qh, ib, _mm512_set1_epi32(15), 1), 31));
            _mm512_store_ps(p.delta + 16 * half, _mm512_castsi512_ps(delta));
        }
        const __m512i odd = _mm512_or_si512(_mm512_slli_epi32(field16(qh, _mm512_srli_epi32(lane, 1), _mm512_set1_epi32(12), 7), 1), _mm512_set1_epi32(1));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(wide_half(b)), _mm512_cvtepi32_ps(odd)));
    } else if constexpr (kind == 18) {   // IQ3_XXS: 64 index bytes; per ib aux at 66 (four 7-bit signs, the scale's nibble)
        for (int half = 0; half < 2; ++half)
            _mm512_store_si512(p.offset + 32 * half, _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2 + 32 * half))), 4));
        const __m256i aux = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 66));
        _mm256_store_si256(reinterpret_cast<__m256i*>(p.sign), signs32(aux_fields(aux)));
        const __m512i wide = _mm512_castsi256_si512(aux);
        const __m512i nibbles = _mm512_srli_epi32(_mm512_permutexvar_epi32(_mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7), wide), 28);
        const __m512 levels = iq2_levels();   // d * (0.5 + nibble) * 0.5
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(wide_half(b)), _mm512_permutexvar_ps(nibbles, _mm512_add_ps(levels, levels))));
        p.signs = p.sign;
    } else {   // IQ3_S: qs[2..66) | bit r of qh[66 + ib]; signs at 74; scales nibbles at 106
        static_assert(kind == 21);
        uint64_t high;
        std::memcpy(&high, b + 66, 8);
        for (int half = 0; half < 2; ++half) {
            const __m512i bytes = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2 + 32 * half)));
            const __m512i index = _mm512_mask_add_epi16(bytes, __mmask32(high >> (32 * half)), bytes, _mm512_set1_epi16(256));
            _mm512_store_si512(p.offset + 32 * half, _mm512_slli_epi16(index, 4));
        }
        // An odd scale per ib, 2 * nibble + 1, of the block's half: the eight nibbles at 106, each for two chunks of sixteen values.
        const __m128i n = _mm_cvtsi32_si128(int(load32(b + 106))), low = _mm_set1_epi8(15);
        const __m128i eight = _mm_unpacklo_epi8(_mm_and_si128(n, low), _mm_and_si128(_mm_srli_epi16(n, 4), low));
        const __m512i nibbles = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(eight, eight));
        const __m512 odd = _mm512_setr_ps(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(wide_half(b)), _mm512_permutexvar_ps(nibbles, odd)));
        p.signs = b + 74;
    }
}
struct Iq1Deltas {
    alignas(64) float value[4][16];
    constexpr Iq1Deltas() : value() {
        for (int k = 0; k < 4; ++k)
            for (int lane = 0; lane < 16; ++lane) value[k][lane] = (k >> (lane / 8)) & 1 ? -0.125f : 0.125f;
    }
};
constexpr Iq1Deltas iq1_deltas;
template<int kind>
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512 prepared16(const Prepared& p, unsigned c, const char* table) {
    __m512 v;
    if constexpr (Block<kind>::width == 8) {
        uint32_t both;
        std::memcpy(&both, p.offset + 2 * c, 4);
        const __m256d low = _mm256_castps_pd(_mm256_loadu_ps(reinterpret_cast<const float*>(table + (both & 0xffff))));
        const __m256d high = _mm256_castps_pd(_mm256_loadu_ps(reinterpret_cast<const float*>(table + (both >> 16))));
        v = _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(low), high, 1));
    } else {
        // Two 256-bit inserts and one 512-bit: a double-pumped core runs the narrow ones in half the time of three wide ones.
        uint64_t all;
        std::memcpy(&all, p.offset + 4 * c, 8);
        const __m256 low = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<const float*>(table + (all & 0xffff)))),
                                                _mm_loadu_ps(reinterpret_cast<const float*>(table + ((all >> 16) & 0xffff))), 1);
        const __m256 high = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(reinterpret_cast<const float*>(table + ((all >> 32) & 0xffff)))),
                                                 _mm_loadu_ps(reinterpret_cast<const float*>(table + (all >> 48))), 1);
        v = _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(low)), _mm256_castps_pd(high), 1));
    }
    if constexpr (kind == 29) {
        // The qh byte holds both rows' nibbles: bits 3 and 7 name one of four delta vectors.
        const unsigned byte = p.signs[c];
        const __m512 delta = _mm512_load_ps(iq1_deltas.value[((byte >> 3) & 1) | ((byte >> 6) & 2)]);
        return _mm512_mul_ps(_mm512_set1_ps(p.scale[c]), _mm512_add_ps(v, delta));
    } else if constexpr (kind == 19) {
        const __m512 delta = _mm512_mask_blend_ps(__mmask16(0xff00), _mm512_set1_ps(p.delta[2 * c]), _mm512_set1_ps(p.delta[2 * c + 1]));
        return _mm512_mul_ps(_mm512_set1_ps(p.scale[c]), _mm512_add_ps(v, delta));
    } else {
        uint16_t signs;
        std::memcpy(&signs, p.signs + 2 * c, 2);
        const __m512i bits = _mm512_castps_si512(_mm512_mul_ps(v, _mm512_set1_ps(p.scale[c])));
        return _mm512_castsi512_ps(_mm512_mask_xor_epi32(bits, __mmask16(signs), bits, _mm512_set1_epi32(int(0x80000000u))));
    }
}
// A row's total from its sixteen lanes.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline float total512(__m512 acc) {
    // reduce16's additions, in its order, on registers: a store and fifteen scalar reads cost a row more than its last chunks.
    const __m256 eight = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc), 1)));
    const __m128 four = _mm_add_ps(_mm256_castps256_ps128(eight), _mm256_extractf128_ps(eight, 1));
    const __m128 two = _mm_add_ps(four, _mm_movehl_ps(four, four));
    return _mm_cvtss_f32(_mm_add_ss(two, _mm_shuffle_ps(two, two, 1)));
}
// The accumulators are separate locals, not an array: cl.exe keeps an array of vectors in memory, a store and a reload through
// every row's FMA each chunk.
template<int kind, int N>
SD_TARGET("avx512f,avx512bw,avx2,f16c")
void rows512(const uint8_t* row, size_t stride, const float* x, size_t cols, const float* table, float* out) {
    using B = Block<kind>;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    if constexpr (B::grid > 0) {
        // Each block is prepared while the one before it computes: reading a wide store still in the store queue costs more than the wait.
        Prepared buffers[2][N];
        const Prepared* prep = buffers[0];
        const char* bytes = reinterpret_cast<const char*>(table);
        const size_t blocks = cols / B::values;
        for (int r = 0; r < N; ++r) prepare<kind>(row + size_t(r) * stride, buffers[0][r]);
        for (size_t b = 0; b < blocks; ++b) {
            prefetch_next<B::bytes, N>(row, stride, b);
            Prepared* next = buffers[(b + 1) & 1];
            if (b + 1 < blocks) for (int r = 0; r < N; ++r) prepare<kind>(row + size_t(r) * stride + (b + 1) * B::bytes, next[r]);
            for (unsigned c = 0; c < B::values / 16; ++c) {
                const __m512 xv = _mm512_loadu_ps(x + b * B::values + 16 * c);
                acc0 = _mm512_fmadd_ps(prepared16<kind>(prep[0], c, bytes), xv, acc0);
                if constexpr (N > 1) acc1 = _mm512_fmadd_ps(prepared16<kind>(prep[1], c, bytes), xv, acc1);
                if constexpr (N > 2) acc2 = _mm512_fmadd_ps(prepared16<kind>(prep[2], c, bytes), xv, acc2);
                if constexpr (N > 3) acc3 = _mm512_fmadd_ps(prepared16<kind>(prep[3], c, bytes), xv, acc3);
            }
            prep = next;
        }
    } else {
        B block[N];
        for (size_t b = 0; b < cols / B::values; ++b) {
            prefetch_next<B::bytes, N>(row, stride, b);
            for (int r = 0; r < N; ++r) block[r].load(row + size_t(r) * stride + b * B::bytes);
            for (unsigned c = 0; c < B::values / 16; ++c) {
                const __m512 xv = _mm512_loadu_ps(x + b * B::values + 16 * c);
                acc0 = _mm512_fmadd_ps(values16(block[0], c, table), xv, acc0);
                if constexpr (N > 1) acc1 = _mm512_fmadd_ps(values16(block[1], c, table), xv, acc1);
                if constexpr (N > 2) acc2 = _mm512_fmadd_ps(values16(block[2], c, table), xv, acc2);
                if constexpr (N > 3) acc3 = _mm512_fmadd_ps(values16(block[3], c, table), xv, acc3);
            }
        }
    }
    out[0] = total512(acc0);
    if constexpr (N > 1) out[1] = total512(acc1);
    if constexpr (N > 2) out[2] = total512(acc2);
    if constexpr (N > 3) out[3] = total512(acc3);
}

// Q2_0 rows together (row_avx512's values, order and rounding): alone, one row's chain of FMAs waits out the FMA latency at
// every step whatever the width; beside other rows the same steps overlap, and the activations are read once.
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512i q2_codes(const uint8_t* block) { return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2))); }
SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline __m512 q2_scale(const uint8_t* block) { return _mm512_set1_ps(wide_half(block)); }
template<int N>
SD_TARGET("avx512f,avx512bw,avx2,f16c")
void q2_rows512(const uint8_t* row, size_t stride, const float* x, size_t blocks, float* out) {
    const __m512 table = _mm512_setr_ps(-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2);
    __m512 total0 = _mm512_setzero_ps(), total1 = total0, total2 = total0, total3 = total0;
    const uint8_t* row0 = row; const uint8_t* row1 = row + stride; const uint8_t* row2 = row + 2 * stride; const uint8_t* row3 = row + 3 * stride;
    for (size_t b = 0; b < blocks; ++b) {
        prefetch_next<18, N>(row, stride, b);
        const float* values = x + 64 * b;
        const __m512 x0 = _mm512_loadu_ps(values), x1 = _mm512_loadu_ps(values + 16), x2 = _mm512_loadu_ps(values + 32), x3 = _mm512_loadu_ps(values + 48);
        __m512i codes0 = q2_codes(row0 + 18 * b), codes1, codes2, codes3;
        __m512 sum0 = _mm512_mul_ps(_mm512_permutevar_ps(table, codes0), x0), sum1, sum2, sum3;
        if constexpr (N > 1) { codes1 = q2_codes(row1 + 18 * b); sum1 = _mm512_mul_ps(_mm512_permutevar_ps(table, codes1), x0); }
        if constexpr (N > 2) { codes2 = q2_codes(row2 + 18 * b); sum2 = _mm512_mul_ps(_mm512_permutevar_ps(table, codes2), x0); }
        if constexpr (N > 3) { codes3 = q2_codes(row3 + 18 * b); sum3 = _mm512_mul_ps(_mm512_permutevar_ps(table, codes3), x0); }
#define SD_Q2_STEP(shift, xv) \
        sum0 = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes0, shift)), xv, sum0); \
        if constexpr (N > 1) sum1 = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes1, shift)), xv, sum1); \
        if constexpr (N > 2) sum2 = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes2, shift)), xv, sum2); \
        if constexpr (N > 3) sum3 = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes3, shift)), xv, sum3);
        SD_Q2_STEP(2, x1)
        SD_Q2_STEP(4, x2)
        SD_Q2_STEP(6, x3)
#undef SD_Q2_STEP
        total0 = _mm512_fmadd_ps(q2_scale(row0 + 18 * b), sum0, total0);
        if constexpr (N > 1) total1 = _mm512_fmadd_ps(q2_scale(row1 + 18 * b), sum1, total1);
        if constexpr (N > 2) total2 = _mm512_fmadd_ps(q2_scale(row2 + 18 * b), sum2, total2);
        if constexpr (N > 3) total3 = _mm512_fmadd_ps(q2_scale(row3 + 18 * b), sum3, total3);
    }
    out[0] = _mm512_reduce_add_ps(total0);
    if constexpr (N > 1) out[1] = _mm512_reduce_add_ps(total1);
    if constexpr (N > 2) out[2] = _mm512_reduce_add_ps(total2);
    if constexpr (N > 3) out[3] = _mm512_reduce_add_ps(total3);
}
inline void q2_rows(const uint8_t* row, size_t stride, size_t count, const float* x, size_t blocks, float* out) {
    size_t r = 0;
    for (; r + 4 <= count; r += 4) q2_rows512<4>(row + r * stride, stride, x, blocks, out + r);
    switch (count - r) {
        case 3: q2_rows512<3>(row + r * stride, stride, x, blocks, out + r); break;
        case 2: q2_rows512<2>(row + r * stride, stride, x, blocks, out + r); break;
        case 1: q2_rows512<1>(row + r * stride, stride, x, blocks, out + r); break;
        default: break;
    }
}

SD_TARGET("avx512f,avx512bw,avx2,f16c")
inline void add_weighted512(float* out, const float* values, float weight, size_t count) {
    const __m512 w = _mm512_set1_ps(weight);
    size_t i = 0;
    for (; i + 16 <= count; i += 16)
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(out + i), _mm512_mul_ps(w, _mm512_loadu_ps(values + i))));
    for (; i < count; ++i) out[i] += weight * values[i];
}

// AVX2: the same sixteen values as two halves of eight.
struct Halves { __m256 low, high; };
// Eight sign bits as a vector to xor: a table's load against a broadcast, a shift and a mask.
struct Flips {
    alignas(32) uint32_t value[256][8];
    constexpr Flips() : value() {
        for (unsigned bits = 0; bits < 256; ++bits)
            for (unsigned lane = 0; lane < 8; ++lane) value[bits][lane] = (bits >> lane) & 1 ? 0x80000000u : 0u;
    }
};
constexpr Flips flips;
SD_TARGET("avx2,fma,f16c")
inline __m256 flip8(__m256 v, unsigned signs) {
    return _mm256_xor_ps(v, _mm256_load_ps(reinterpret_cast<const float*>(flips.value[signs])));
}
SD_TARGET("avx2,fma,f16c")
inline __m256 rows8(const float* first, const float* second, int width) {
    if (width == 8) return _mm256_loadu_ps(first);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(first)), _mm_loadu_ps(second), 1);
}
SD_TARGET("avx2,fma,f16c")
inline __m256 nonlinear8(__m128i bytes, int shift, float scale, bool fp4 = false) {
    const __m256 low = fp4 ? _mm256_setr_ps(0, 1, 2, 3, 4, 6, 8, 12) : _mm256_setr_ps(-127, -104, -83, -65, -49, -35, -22, -10);
    const __m256 high = fp4 ? _mm256_setr_ps(0, -1, -2, -3, -4, -6, -8, -12) : _mm256_setr_ps(1, 13, 25, 38, 53, 69, 89, 113);
    const __m256i codes = _mm256_and_si256(_mm256_srli_epi32(_mm256_cvtepu8_epi32(bytes), shift), _mm256_set1_epi32(15));
    const __m256 value = _mm256_blendv_ps(_mm256_permutevar8x32_ps(low, codes), _mm256_permutevar8x32_ps(high, codes),
                                          _mm256_castsi256_ps(_mm256_slli_epi32(codes, 28)));
    return _mm256_mul_ps(_mm256_set1_ps(scale), value);
}
template<int kind>
SD_TARGET("avx2,fma,f16c")
inline Halves values8x2(const Block<kind>& block, unsigned c, const float* table) {
    using B = Block<kind>;
    if constexpr (B::grid > 0) {
        const GridChunk k = block.chunk(c, table);
        const bool wide = B::width == 8;
        __m256 low = rows8(k.row[0], k.row[1], B::width), high = wide ? rows8(k.row[1], nullptr, 8) : rows8(k.row[2], k.row[3], 4);
        if constexpr (kind == 29 || kind == 19) {
            const __m256 scale = _mm256_set1_ps(k.scale);
            return {_mm256_mul_ps(scale, _mm256_add_ps(low, _mm256_set1_ps(k.delta[0]))),
                    _mm256_mul_ps(scale, _mm256_add_ps(high, _mm256_set1_ps(k.delta[1])))};
        } else {
            const __m256 scale = _mm256_set1_ps(k.scale);
            return {flip8(_mm256_mul_ps(low, scale), k.signs & 255), flip8(_mm256_mul_ps(high, scale), k.signs >> 8)};
        }
    } else if constexpr (B::grid == 0) {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes(c)));
        const float scale = block.scale(c);
        return {nonlinear8(bytes, block.shift(c), scale), nonlinear8(_mm_srli_si128(bytes, 8), block.shift(c), scale)};
    } else if constexpr (B::grid == -2) {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes(c)));
        const __m256 scale = _mm256_set1_ps(block.scale(c)), offset = _mm256_set1_ps(block.offset(c));
        const __m256i shift = _mm256_set1_epi32(block.shift(c)), three = _mm256_set1_epi32(3);
        const __m256i low = _mm256_and_si256(_mm256_srlv_epi32(_mm256_cvtepu8_epi32(bytes), shift), three);
        const __m256i high = _mm256_and_si256(_mm256_srlv_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(bytes, 8)), shift), three);
        return {_mm256_sub_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(low)), offset),
                _mm256_sub_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(high)), offset)};
    } else if constexpr (B::grid == -3) {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes(c)));
        return {nonlinear8(bytes, block.shift(c), block.d, true), nonlinear8(_mm_srli_si128(bytes, 8), block.shift(c), block.d, true)};
    } else {
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.b + 2 + 16 * c));
        const __m256 d = _mm256_set1_ps(block.d);
        return {_mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q))),
                _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))))};
    }
}
SD_TARGET("avx2,fma,f16c")
inline float total256(__m256 low, __m256 high) {
    // (total512's additions)
    const __m256 eight = _mm256_add_ps(low, high);
    const __m128 four = _mm_add_ps(_mm256_castps256_ps128(eight), _mm256_extractf128_ps(eight, 1));
    const __m128 two = _mm_add_ps(four, _mm_movehl_ps(four, four));
    return _mm_cvtss_f32(_mm_add_ss(two, _mm_shuffle_ps(two, two, 1)));
}
template<int kind>
SD_TARGET("avx2,fma,f16c")
inline void accumulate8x2(const Block<kind>& block, unsigned c, const float* table, __m256 xl, __m256 xh, __m256& low, __m256& high) {
    const Halves v = values8x2(block, c, table);
    low = _mm256_fmadd_ps(v.low, xl, low);
    high = _mm256_fmadd_ps(v.high, xh, high);
}
template<int kind, int N>
SD_TARGET("avx2,fma,f16c")
void rows256(const uint8_t* row, size_t stride, const float* x, size_t cols, const float* table, float* out) {
    using B = Block<kind>;
    __m256 low0 = _mm256_setzero_ps(), low1 = low0, low2 = low0, low3 = low0, high0 = low0, high1 = low0, high2 = low0, high3 = low0;
    B block[N];
    for (size_t b = 0; b < cols / B::values; ++b) {
        prefetch_next<B::bytes, N>(row, stride, b);
        for (int r = 0; r < N; ++r) block[r].load(row + size_t(r) * stride + b * B::bytes);
        for (unsigned c = 0; c < B::values / 16; ++c) {
            const float* values = x + b * B::values + 16 * c;
            const __m256 xl = _mm256_loadu_ps(values), xh = _mm256_loadu_ps(values + 8);
            accumulate8x2(block[0], c, table, xl, xh, low0, high0);
            if constexpr (N > 1) accumulate8x2(block[1], c, table, xl, xh, low1, high1);
            if constexpr (N > 2) accumulate8x2(block[2], c, table, xl, xh, low2, high2);
            if constexpr (N > 3) accumulate8x2(block[3], c, table, xl, xh, low3, high3);
        }
    }
    out[0] = total256(low0, high0);
    if constexpr (N > 1) out[1] = total256(low1, high1);
    if constexpr (N > 2) out[2] = total256(low2, high2);
    if constexpr (N > 3) out[3] = total256(low3, high3);
}

SD_TARGET("avx2,fma,f16c")
inline __m256 q2_values8(const uint8_t* at, __m256 table, __m256i shifts, __m256i three) {
    uint16_t codes;
    std::memcpy(&codes, at, 2);
    return _mm256_permutevar8x32_ps(table, _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(codes), shifts), three));
}
SD_TARGET("avx2,fma,f16c")
inline __m256 q2_scale8(const uint8_t* block) {
    uint16_t bits;
    std::memcpy(&bits, block, 2);
    return _mm256_set1_ps(SD_HALF_TO_FLOAT(bits));
}
SD_TARGET("avx2,fma,f16c")
inline float q2_total8(__m256 total) {
    __m128 low = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    return _mm_cvtss_f32(_mm_add_ss(low, _mm_shuffle_ps(low, low, 1)));
}
// row_avx2 rows together, as q2_rows512. The activations stay in their natural order.
template<int N>
SD_TARGET("avx2,fma,f16c")
void q2_rows256(const uint8_t* row, size_t stride, const float* x, size_t blocks, float* out) {
    const __m256 table = _mm256_setr_ps(-1, 0, 1, 2, 0, 0, 0, 0);
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), three = _mm256_set1_epi32(3);
    __m256 total0 = _mm256_setzero_ps(), total1 = total0, total2 = total0, total3 = total0;
    const uint8_t* row0 = row; const uint8_t* row1 = row + stride; const uint8_t* row2 = row + 2 * stride; const uint8_t* row3 = row + 3 * stride;
    for (size_t b = 0; b < blocks; ++b) {
        prefetch_next<18, N>(row, stride, b);
        const float* values = x + 64 * b;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        for (int q = 0; q < 8; ++q) {
            const __m256 xv = _mm256_loadu_ps(values + 8 * q);
            const size_t at = 18 * b + 2 + 2 * q;
            sum0 = _mm256_fmadd_ps(q2_values8(row0 + at, table, shifts, three), xv, sum0);
            if constexpr (N > 1) sum1 = _mm256_fmadd_ps(q2_values8(row1 + at, table, shifts, three), xv, sum1);
            if constexpr (N > 2) sum2 = _mm256_fmadd_ps(q2_values8(row2 + at, table, shifts, three), xv, sum2);
            if constexpr (N > 3) sum3 = _mm256_fmadd_ps(q2_values8(row3 + at, table, shifts, three), xv, sum3);
        }
        total0 = _mm256_fmadd_ps(q2_scale8(row0 + 18 * b), sum0, total0);
        if constexpr (N > 1) total1 = _mm256_fmadd_ps(q2_scale8(row1 + 18 * b), sum1, total1);
        if constexpr (N > 2) total2 = _mm256_fmadd_ps(q2_scale8(row2 + 18 * b), sum2, total2);
        if constexpr (N > 3) total3 = _mm256_fmadd_ps(q2_scale8(row3 + 18 * b), sum3, total3);
    }
    out[0] = q2_total8(total0);
    if constexpr (N > 1) out[1] = q2_total8(total1);
    if constexpr (N > 2) out[2] = q2_total8(total2);
    if constexpr (N > 3) out[3] = q2_total8(total3);
}
inline void q2_rows_avx2(const uint8_t* row, size_t stride, size_t count, const float* x, size_t blocks, float* out) {
    size_t r = 0;
    for (; r + 4 <= count; r += 4) q2_rows256<4>(row + r * stride, stride, x, blocks, out + r);
    switch (count - r) {
        case 3: q2_rows256<3>(row + r * stride, stride, x, blocks, out + r); break;
        case 2: q2_rows256<2>(row + r * stride, stride, x, blocks, out + r); break;
        case 1: q2_rows256<1>(row + r * stride, stride, x, blocks, out + r); break;
        default: break;
    }
}

// count rows of one bank (stride bytes apart) against x; level 2 = AVX-512, 1 = AVX2.
template<int kind>
void block_rows(int level, const uint8_t* row, size_t stride, size_t count, const float* x, size_t cols, float* out) {
    const float* table = Block<kind>::grid > 0 ? iq_float(Block<kind>::grid) : nullptr;
    const auto run = [&](size_t first, auto rows) {
        constexpr int n = decltype(rows)::value;
        if (level == 2) rows512<kind, n>(row + first * stride, stride, x, cols, table, out + first);
        else rows256<kind, n>(row + first * stride, stride, x, cols, table, out + first);
    };
    size_t r = 0;
    for (; r + 4 <= count; r += 4) run(r, std::integral_constant<int, 4>{});
    switch (count - r) {
        case 3: run(r, std::integral_constant<int, 3>{}); break;
        case 2: run(r, std::integral_constant<int, 2>{}); break;
        case 1: run(r, std::integral_constant<int, 1>{}); break;
        default: break;
    }
}
