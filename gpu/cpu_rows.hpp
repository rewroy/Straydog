// Fused multi-row kernels for the block-decoded CPU expert formats (included by
// cpu_experts.cpp on x86). Every row keeps the decoded path's arithmetic exactly:
// each value is the block decoder's F32 value, one 16-lane accumulator per row takes
// acc[k] = fma(value, x, acc[k]) in order, and reduce16 sums the lanes. Lattice rows
// come from F32 copies of the gguf grids (no gathers or integer conversions), and up
// to four rows run together so their accumulation chains overlap and share the
// activations.

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
        GridChunk k{};
        k.row[0] = table + 8 * b[2 + 8 * ib + l]; k.row[1] = table + 8 * b[3 + 8 * ib + l];
        k.scale = d * (0.5f + float(aux >> 28)) * 0.25f;
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
        GridChunk k{};
        k.row[0] = table + 8 * (q0 & 511); k.row[1] = table + 8 * (q1 & 511);
        k.scale = d * (0.5f + float((b[66 + ib] >> (4 * h)) & 15)) * 0.25f;
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
        GridChunk k{};
        for (unsigned i = 0; i < 2; ++i)
            k.row[i] = table + 8 * (b[2 + 4 * ib + l + i] | ((unsigned(b[66 + ib]) << (8 - 2 * (l + i))) & 0x300));
        k.scale = d * (0.5f + float((b[74 + ib] >> (4 * h)) & 15)) * 0.25f;
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
        GridChunk k{};
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
        GridChunk k{};
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
        GridChunk k{};
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
        GridChunk k{};
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
    void load(const uint8_t* block) { b = block; d = half_at(block); }
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
        const __m512i codes = _mm512_and_si512(_mm512_srli_epi32(_mm512_cvtepu8_epi32(bytes), unsigned(block.shift(c))),
                                               _mm512_set1_epi32(15));
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
    const uint8_t* signs;              // the block's own sign bytes (IQ2_S, IQ3_S), else sign
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
        // Scale = d * (0.5 + nibble) * 0.25: the half's 11 bits times the five of 0.5 + nibble are exact, so the order does not round.
        const __m128i n = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 74)), low = _mm_set1_epi8(15);
        const __m128i nibbles = _mm_unpacklo_epi8(_mm_and_si128(n, low), _mm_and_si128(_mm_srli_epi16(n, 4), low));
        const __m512 levels = _mm512_setr_ps(0.125f, 0.375f, 0.625f, 0.875f, 1.125f, 1.375f, 1.625f, 1.875f,
                                             2.125f, 2.375f, 2.625f, 2.875f, 3.125f, 3.375f, 3.625f, 3.875f);
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(half_at(b)), _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(nibbles), levels)));
        p.signs = b + 34;
    } else if constexpr (kind == 16) {   // IQ2_XXS: per ib, four index bytes then aux (signs, scale)
        const __m512i lane = lanes16();
        const __m512i words = _mm512_loadu_si512(reinterpret_cast<const void*>(b + 2));   // 2ib: indices, 2ib + 1: aux
        __m512i sign_bits[2];
        for (int half = 0; half < 2; ++half) {
            const __m512i g = lanes16(16 * half), l = _mm512_and_si512(g, _mm512_set1_epi32(3));
            const __m512i word = _mm512_slli_epi32(_mm512_srli_epi32(g, 2), 1);
            const __m512i index = field16(words, word, _mm512_slli_epi32(l, 3), 255);
            store_offsets(p.offset + 16 * half, index, 5);
            const __m512i aux = _mm512_permutexvar_epi32(_mm512_add_epi32(word, _mm512_set1_epi32(1)), words);
            sign_bits[half] = signs16(_mm512_and_si512(_mm512_srlv_epi32(aux, _mm512_mullo_epi32(l, _mm512_set1_epi32(7))), _mm512_set1_epi32(127)));
        }
        store_signs(p.sign, sign_bits[0], sign_bits[1]);
        const __m512i aux = _mm512_permutexvar_epi32(_mm512_add_epi32(_mm512_slli_epi32(_mm512_srli_epi32(lane, 1), 1), _mm512_set1_epi32(1)), words);
        iq2_scales(half_at(b), _mm512_srli_epi32(aux, 28), p.scale);
        p.signs = p.sign;
    } else if constexpr (kind == 17) {   // IQ2_XS: 16-bit q per group (9-bit index, 7 sign bits); scales at 66
        const __m512i lane = lanes16();
        __m512i sign_bits[2];
        for (int half = 0; half < 2; ++half) {
            const __m512i q = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2 + 32 * half)));
            store_offsets(p.offset + 16 * half, _mm512_and_si512(q, _mm512_set1_epi32(511)), 5);
            sign_bits[half] = signs16(_mm512_srli_epi32(q, 9));
        }
        store_signs(p.sign, sign_bits[0], sign_bits[1]);
        const __m512i scales = _mm512_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 66)));
        iq2_scales(half_at(b), field16(scales, _mm512_srli_epi32(lane, 1), _mm512_slli_epi32(_mm512_and_si512(lane, _mm512_set1_epi32(1)), 2), 15), p.scale);
        p.signs = p.sign;
    } else if constexpr (kind == 29) {   // IQ1_M: qs[0..32) | 3 bits of a qh nibble; nibble bit 3 picks the delta
        const __m512i lane = lanes16();
        const __m512i qh = bytes16(b + 32);
        for (int half = 0; half < 2; ++half) {
            const __m512i g = lanes16(16 * half);
            const __m512i nibble = field16(qh, _mm512_srli_epi32(g, 1), _mm512_slli_epi32(_mm512_and_si512(g, _mm512_set1_epi32(1)), 2), 15);
            const __m512i index = _mm512_or_si512(bytes16(b + 16 * half), _mm512_slli_epi32(_mm512_and_si512(nibble, _mm512_set1_epi32(7)), 8));
            store_offsets(p.offset + 16 * half, index, 5);
            const __m512i delta = _mm512_or_si512(_mm512_set1_epi32(0x3E000000), _mm512_slli_epi32(_mm512_and_si512(nibble, _mm512_set1_epi32(8)), 28));
            _mm512_store_ps(p.delta + 16 * half, _mm512_castsi512_ps(delta));
        }
        uint16_t words[4];
        const float d = iq1m_scale(b, words);
        const __m512i word = _mm512_cvtepu16_epi32(_mm256_castsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(words))));
        const __m512i ib = _mm512_srli_epi32(lane, 1);
        const __m512i shift = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_and_si512(ib, _mm512_set1_epi32(1)), _mm512_set1_epi32(6)),
                                               _mm512_mullo_epi32(_mm512_and_si512(lane, _mm512_set1_epi32(1)), _mm512_set1_epi32(3)));
        const __m512i odd = _mm512_or_si512(_mm512_slli_epi32(field16(word, _mm512_srli_epi32(lane, 2), shift, 7), 1), _mm512_set1_epi32(1));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(odd)));
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
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(half_at(b)), _mm512_cvtepi32_ps(odd)));
    } else if constexpr (kind == 18) {   // IQ3_XXS: 64 index bytes; per ib aux at 66 (signs, scale)
        const __m512i lane = lanes16();
        for (int quarter = 0; quarter < 4; ++quarter)
            store_offsets(p.offset + 16 * quarter, bytes16(b + 2 + 16 * quarter), 4);
        const __m512i aux = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 66)));
        __m512i sign_bits[2];
        for (int half = 0; half < 2; ++half) {
            const __m512i g = lanes16(16 * half), l = _mm512_and_si512(g, _mm512_set1_epi32(3));
            sign_bits[half] = signs16(field16(aux, _mm512_srli_epi32(g, 2), _mm512_mullo_epi32(l, _mm512_set1_epi32(7)), 127));
        }
        store_signs(p.sign, sign_bits[0], sign_bits[1]);
        const __m512i scale = _mm512_srli_epi32(_mm512_permutexvar_epi32(_mm512_srli_epi32(lane, 1), aux), 28);
        const __m512 sum = _mm512_add_ps(_mm512_set1_ps(0.5f), _mm512_cvtepi32_ps(scale));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(half_at(b)), sum), _mm512_set1_ps(0.5f)));
        p.signs = p.sign;
    } else {   // IQ3_S: qs[2..66) | bit r of qh[66 + ib]; signs at 74; scales nibbles at 106
        static_assert(kind == 21);
        const __m512i lane = lanes16();
        const __m512i qh = _mm512_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 66)));
        for (int quarter = 0; quarter < 4; ++quarter) {
            const __m512i t = lanes16(16 * quarter);
            const __m512i high = field16(qh, _mm512_srli_epi32(t, 3), _mm512_and_si512(t, _mm512_set1_epi32(7)), 1);
            const __m512i index = _mm512_or_si512(bytes16(b + 2 + 16 * quarter), _mm512_slli_epi32(high, 8));
            store_offsets(p.offset + 16 * quarter, index, 4);
        }
        const __m512i scales = _mm512_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 106)));
        const __m512i ib = _mm512_srli_epi32(lane, 1);
        const __m512i nibble = field16(scales, _mm512_srli_epi32(ib, 1), _mm512_slli_epi32(_mm512_and_si512(ib, _mm512_set1_epi32(1)), 2), 15);
        const __m512i odd = _mm512_or_si512(_mm512_slli_epi32(nibble, 1), _mm512_set1_epi32(1));
        _mm512_store_ps(p.scale, _mm512_mul_ps(_mm512_set1_ps(half_at(b)), _mm512_cvtepi32_ps(odd)));
        p.signs = b + 74;
    }
}
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
        v = _mm512_castps128_ps512(_mm_loadu_ps(reinterpret_cast<const float*>(table + p.offset[4 * c])));
        v = _mm512_insertf32x4(v, _mm_loadu_ps(reinterpret_cast<const float*>(table + p.offset[4 * c + 1])), 1);
        v = _mm512_insertf32x4(v, _mm_loadu_ps(reinterpret_cast<const float*>(table + p.offset[4 * c + 2])), 2);
        v = _mm512_insertf32x4(v, _mm_loadu_ps(reinterpret_cast<const float*>(table + p.offset[4 * c + 3])), 3);
    }
    if constexpr (kind == 29 || kind == 19) {
        const __m512 delta = _mm512_mask_blend_ps(__mmask16(0xff00), _mm512_set1_ps(p.delta[2 * c]), _mm512_set1_ps(p.delta[2 * c + 1]));
        return _mm512_mul_ps(_mm512_set1_ps(p.scale[c]), _mm512_add_ps(v, delta));
    } else {
        uint16_t signs;
        std::memcpy(&signs, p.signs + 2 * c, 2);
        const __m512i bits = _mm512_castps_si512(_mm512_mul_ps(v, _mm512_set1_ps(p.scale[c])));
        return _mm512_castsi512_ps(_mm512_mask_xor_epi32(bits, __mmask16(signs), bits, _mm512_set1_epi32(int(0x80000000u))));
    }
}
template<int kind, int N>
SD_TARGET("avx512f,avx512bw,avx2,f16c")
void rows512(const uint8_t* row, size_t stride, const float* x, size_t cols, const float* table, float* out) {
    using B = Block<kind>;
    __m512 acc[N];
    for (int r = 0; r < N; ++r) acc[r] = _mm512_setzero_ps();
    if constexpr (B::grid > 0) {
        // Each block is prepared while the one before it computes: reading a wide store still in the store queue costs more than the wait.
        Prepared buffers[2][N];
        Prepared* prep = buffers[0];
        const char* bytes = reinterpret_cast<const char*>(table);
        const size_t blocks = cols / B::values;
        for (int r = 0; r < N; ++r) prepare<kind>(row + size_t(r) * stride, prep[r]);
        for (size_t b = 0; b < blocks; ++b) {
            prefetch_next<B::bytes, N>(row, stride, b);
            Prepared* next = buffers[(b + 1) & 1];
            if (b + 1 < blocks) for (int r = 0; r < N; ++r) prepare<kind>(row + size_t(r) * stride + (b + 1) * B::bytes, next[r]);
            for (unsigned c = 0; c < B::values / 16; ++c) {
                const __m512 xv = _mm512_loadu_ps(x + b * B::values + 16 * c);
                for (int r = 0; r < N; ++r) acc[r] = _mm512_fmadd_ps(prepared16<kind>(prep[r], c, bytes), xv, acc[r]);
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
                for (int r = 0; r < N; ++r) acc[r] = _mm512_fmadd_ps(values16(block[r], c, table), xv, acc[r]);
            }
        }
    }
    for (int r = 0; r < N; ++r) {
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, acc[r]);
        out[r] = reduce16(lanes);
    }
}

// AVX2: the same sixteen values as two halves of eight.
struct Halves { __m256 low, high; };
SD_TARGET("avx2,fma,f16c")
inline __m256 flip8(__m256 v, unsigned signs) {
    const __m256i moved = _mm256_sllv_epi32(_mm256_set1_epi32(int(signs)), _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    return _mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_and_si256(moved, _mm256_set1_epi32(int(0x80000000u)))));
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
template<int kind, int N>
SD_TARGET("avx2,fma,f16c")
void rows256(const uint8_t* row, size_t stride, const float* x, size_t cols, const float* table, float* out) {
    using B = Block<kind>;
    __m256 low[N], high[N];
    B block[N];
    for (int r = 0; r < N; ++r) low[r] = high[r] = _mm256_setzero_ps();
    for (size_t b = 0; b < cols / B::values; ++b) {
        prefetch_next<B::bytes, N>(row, stride, b);
        for (int r = 0; r < N; ++r) block[r].load(row + size_t(r) * stride + b * B::bytes);
        for (unsigned c = 0; c < B::values / 16; ++c) {
            const float* values = x + b * B::values + 16 * c;
            const __m256 xl = _mm256_loadu_ps(values), xh = _mm256_loadu_ps(values + 8);
            for (int r = 0; r < N; ++r) {
                const Halves v = values8x2(block[r], c, table);
                low[r] = _mm256_fmadd_ps(v.low, xl, low[r]);
                high[r] = _mm256_fmadd_ps(v.high, xh, high[r]);
            }
        }
    }
    for (int r = 0; r < N; ++r) {
        alignas(32) float lanes[16];
        _mm256_store_ps(lanes, low[r]); _mm256_store_ps(lanes + 8, high[r]);
        out[r] = reduce16(lanes);
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
