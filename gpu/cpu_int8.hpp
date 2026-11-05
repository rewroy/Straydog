// Integer CPU expert path (included by cpu_experts.cpp): activations rounded to 8-bit integers once a call, weights taken as the small
// integers the block formats store, products summed exactly in integers, and one F32 scale applied a block: the arithmetic of
// llama.cpp's CPU dot products, independently written.
//
// gate and up (rows of `hidden` values, 256-value blocks) take x in Q8K blocks (one scale a 256 values); down (rows of `inter` values,
// Q2_0's 64-value blocks) takes each expert's products in Q8S blocks (one scale a 64). Sixteen integer lanes: lane j of a 64-value chunk
// sums its values 4j..4j+3 (times their integer group scales) across a block's chunks; each block then adds float(lane) times the
// block's F32 scale to the lane's F32 accumulator with one fused multiply-add, and a row ends with reduce16. The scalar kernels compute
// the same lanes, so every width gives the same bits (the integer sums are exact).

struct Q8K {
    float d;
    alignas(64) int8_t q[256];
};
struct Q8S {
    float d;
    alignas(64) int8_t q[64];       // value 4k + m at 16m + k (the order of Q2_0's codes, two bits at a time)
    alignas(64) int32_t lanes[16];  // each lane's sum of q (Q2_0's codes are value + 1)
};

// The scale and integers of one block: d = amax / 127, q = round(x / d) (ties to even), zeros for a block of zeros.
inline float q8_block(const float* x, size_t n, int8_t* q, size_t (*place)(size_t)) {
    float amax = 0;
    for (size_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    if (amax == 0 || !std::isfinite(amax)) {
        for (size_t i = 0; i < n; ++i) q[place(i)] = 0;
        return 0;
    }
    const float inverse = 127.0f / amax;
    for (size_t i = 0; i < n; ++i) q[place(i)] = int8_t(std::clamp(int(std::nearbyint(x[i] * inverse)), -127, 127));
    return amax / 127.0f;
}
inline size_t natural_place(size_t i) { return i; }
inline size_t q2_place(size_t i) { return 16 * (i % 4) + i / 4; }

void quantize_q8k(const float* x, size_t blocks, Q8K* out) {
    for (size_t b = 0; b < blocks; ++b) out[b].d = q8_block(x + 256 * b, 256, out[b].q, natural_place);
}
void quantize_q8s(const float* x, size_t blocks, Q8S* out) {
    for (size_t b = 0; b < blocks; ++b) {
        out[b].d = q8_block(x + 64 * b, 64, out[b].q, q2_place);
        for (int j = 0; j < 16; ++j) out[b].lanes[j] = out[b].q[4 * j] + out[b].q[4 * j + 1] + out[b].q[4 * j + 2] + out[b].q[4 * j + 3];
    }
}

#ifdef SD_X86
// The same rounding with AVX-512: the same maximum, the same F32 products, and conversion by the current rounding mode (to nearest even,
// as std::nearbyint): the same integers.
SD_TARGET("avx512f,avx512bw,avx512vbmi")
inline __m512i q8_rounded(const float* x, float& d) {
    // (64 values: four vectors)
    __m512 v[4], most = _mm512_setzero_ps();
    for (int k = 0; k < 4; ++k) { v[k] = _mm512_loadu_ps(x + 16 * k); most = _mm512_max_ps(most, _mm512_abs_ps(v[k])); }
    const float amax = _mm512_reduce_max_ps(most);
    if (amax == 0 || !std::isfinite(amax)) { d = 0; return _mm512_setzero_si512(); }
    const __m512 inverse = _mm512_set1_ps(127.0f / amax);
    d = amax / 127.0f;
    __m128i bytes[4];
    for (int k = 0; k < 4; ++k) {
        const __m512i value = _mm512_min_epi32(_mm512_max_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v[k], inverse)), _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
        bytes[k] = _mm512_cvtepi32_epi8(value);
    }
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set_m128i(bytes[1], bytes[0])), _mm256_set_m128i(bytes[3], bytes[2]), 1);
}
// 256-value blocks: the amax of all 256 first.
SD_TARGET("avx512f,avx512bw,avx512vbmi")
void quantize_q8k_avx512(const float* x, size_t blocks, Q8K* out) {
    for (size_t b = 0; b < blocks; ++b) {
        const float* block = x + 256 * b;
        __m512 most = _mm512_setzero_ps();
        for (int k = 0; k < 16; ++k) most = _mm512_max_ps(most, _mm512_abs_ps(_mm512_loadu_ps(block + 16 * k)));
        const float amax = _mm512_reduce_max_ps(most);
        if (amax == 0 || !std::isfinite(amax)) { out[b].d = 0; std::memset(out[b].q, 0, 256); continue; }
        const __m512 inverse = _mm512_set1_ps(127.0f / amax);
        out[b].d = amax / 127.0f;
        for (int k = 0; k < 16; ++k) {
            const __m512i value = _mm512_min_epi32(_mm512_max_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(block + 16 * k), inverse)),
                                                                    _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
            _mm_store_si128(reinterpret_cast<__m128i*>(out[b].q + 16 * k), _mm512_cvtepi32_epi8(value));
        }
    }
}
// 64-value blocks in q2_place order, with their lane sums.
alignas(64) constexpr uint8_t q2_order[64] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
                                              2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};
SD_TARGET("avx512f,avx512bw,avx512vbmi")
void quantize_q8s_avx512(const float* x, size_t blocks, Q8S* out) {
    for (size_t b = 0; b < blocks; ++b) {
        float d;
        const __m512i q = _mm512_permutexvar_epi8(_mm512_load_si512(q2_order), q8_rounded(x + 64 * b, d));
        out[b].d = d;
        _mm512_store_si512(out[b].q, q);
        _mm512_store_si512(out[b].lanes, _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_set1_epi8(1), q), _mm512_set1_epi16(1)));
    }
}
// The same rounding with AVX2 (the same maximum, products and conversion): 32 values rounded, clamped and packed in their order.
SD_TARGET("avx2,fma,f16c")
inline __m256i q8_pack32(const float* x, __m256 inverse) {
    const __m256i low = _mm256_set1_epi32(-127), high = _mm256_set1_epi32(127);
    __m256i v[4];
    for (int k = 0; k < 4; ++k) v[k] = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + 8 * k), inverse)), low), high);
    // (the packs work within each 128 bits: dwords a0-3 b0-3 c0-3 d0-3 | a4-7 b4-7 c4-7 d4-7, put back in order)
    const __m256i bytes = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
    return _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
SD_TARGET("avx2,fma,f16c")
inline float amax256(const float* x, size_t n) {
    const __m256 magnitude = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 most = _mm256_setzero_ps();
    for (size_t i = 0; i < n; i += 8) most = _mm256_max_ps(most, _mm256_and_ps(_mm256_loadu_ps(x + i), magnitude));
    alignas(32) float lane[8];
    _mm256_store_ps(lane, most);
    float amax = 0;
    for (int i = 0; i < 8; ++i) amax = std::max(amax, lane[i]);
    return amax;
}
SD_TARGET("avx2,fma,f16c")
void quantize_q8k_avx2(const float* x, size_t blocks, Q8K* out) {
    for (size_t b = 0; b < blocks; ++b) {
        const float* block = x + 256 * b;
        const float amax = amax256(block, 256);
        if (amax == 0 || !std::isfinite(amax)) { out[b].d = 0; std::memset(out[b].q, 0, 256); continue; }
        const __m256 inverse = _mm256_set1_ps(127.0f / amax);
        out[b].d = amax / 127.0f;
        for (int k = 0; k < 8; ++k) _mm256_store_si256(reinterpret_cast<__m256i*>(out[b].q + 32 * k), q8_pack32(block + 32 * k, inverse));
    }
}
SD_TARGET("avx2,fma,f16c")
void quantize_q8s_avx2(const float* x, size_t blocks, Q8S* out) {
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    for (size_t b = 0; b < blocks; ++b) {
        const float* block = x + 64 * b;
        const float amax = amax256(block, 64);
        alignas(32) int8_t natural[64] = {};
        if (amax == 0 || !std::isfinite(amax)) out[b].d = 0;
        else {
            const __m256 inverse = _mm256_set1_ps(127.0f / amax);
            out[b].d = amax / 127.0f;
            _mm256_store_si256(reinterpret_cast<__m256i*>(natural), q8_pack32(block, inverse));
            _mm256_store_si256(reinterpret_cast<__m256i*>(natural + 32), q8_pack32(block + 32, inverse));
        }
        for (int p = 0; p < 64; ++p) out[b].q[p] = natural[q2_order[p]];
        for (int h = 0; h < 2; ++h) {
            const __m256i q = _mm256_load_si256(reinterpret_cast<const __m256i*>(out[b].q + 32 * h));
            _mm256_store_si256(reinterpret_cast<__m256i*>(out[b].lanes + 8 * h), _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q), ones16));
        }
    }
}
#endif
using QuantizeK = void (*)(const float*, size_t, Q8K*);
using QuantizeS = void (*)(const float*, size_t, Q8S*);
inline QuantizeK quantize_k(int level) {
#ifdef SD_X86
    if (level >= 2) return quantize_q8k_avx512;
    if (level == 1) return quantize_q8k_avx2;
#endif
    (void)level;
    return quantize_q8k;
}
inline QuantizeS quantize_s(int level) {
#ifdef SD_X86
    if (level >= 2) return quantize_q8s_avx512;
    if (level == 1) return quantize_q8s_avx2;
#endif
    (void)level;
    return quantize_q8s;
}

// --- scalar kernels: the reference the wide ones must equal ---

// IQ2_XXS: a group of 32 values (ib) has a scale nibble ls; a value is d * (2 ls + 1) / 8 * grid * sign.
void iq2_xxs_int_scalar(const uint8_t* row, size_t stride, size_t count, const Q8K* x, size_t blocks, float* out) {
    const uint8_t* grid = iq_table(16);
    for (size_t r = 0; r < count; ++r) {
        float acc[16] = {};
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t* block = row + r * stride + 66 * b;
            int32_t lane[16] = {};
            for (int v = 0; v < 256; ++v) {
                const int ib = v / 32, l = (v % 32) / 8, j = v % 8;
                const uint32_t aux = load32(block + 2 + 8 * ib + 4);
                const unsigned signs = iq_signs((aux >> (7 * l)) & 127);
                const int value = x[b].q[v];
                lane[(v % 64) / 4] += int32_t(2 * (aux >> 28) + 1) * int(grid[8 * block[2 + 8 * ib + l] + j]) * ((signs >> j) & 1 ? -value : value);
            }
            const float scale = half_at(block) * x[b].d * 0.125f;
            for (int k = 0; k < 16; ++k) acc[k] = std::fma(float(lane[k]), scale, acc[k]);
        }
        out[r] = reduce16(acc);
    }
}
// IQ2_S: 16 values share a scale nibble; ten-bit grid indices, a sign byte a grid row.
void iq2_s_int_scalar(const uint8_t* row, size_t stride, size_t count, const Q8K* x, size_t blocks, float* out) {
    const uint8_t* grid = iq_table(22);
    for (size_t r = 0; r < count; ++r) {
        float acc[16] = {};
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t* block = row + r * stride + 82 * b;
            int32_t lane[16] = {};
            for (int v = 0; v < 256; ++v) {
                const int ib = v / 32, l = (v % 32) / 8, j = v % 8, half = (v % 32) / 16;
                const unsigned index = block[2 + 4 * ib + l] | ((unsigned(block[66 + ib]) << (8 - 2 * l)) & 0x300);
                const unsigned signs = block[34 + 4 * ib + l];
                const int value = x[b].q[v];
                lane[(v % 64) / 4] += int32_t(2 * ((block[74 + ib] >> (4 * half)) & 15) + 1) * int(grid[8 * index + j]) * ((signs >> j) & 1 ? -value : value);
            }
            const float scale = half_at(block) * x[b].d * 0.125f;
            for (int k = 0; k < 16; ++k) acc[k] = std::fma(float(lane[k]), scale, acc[k]);
        }
        out[r] = reduce16(acc);
    }
}
// Q2_0: 64 values of an FP16 scale and two-bit codes, the value scale * (code - 1); the activations in q2_place order.
void q2_0_int_scalar(const uint8_t* row, size_t stride, size_t count, const Q8S* x, size_t blocks, float* out) {
    for (size_t r = 0; r < count; ++r) {
        float acc[16] = {};
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t* block = row + r * stride + 18 * b;
            int32_t lane[16] = {};
            for (int position = 0; position < 64; ++position) {
                const int m = position / 16, k = position % 16;
                lane[position / 4] += int32_t((block[2 + k] >> (2 * m)) & 3) * x[b].q[position];
            }
            const float scale = half_at(block) * x[b].d;
            for (int k = 0; k < 16; ++k) acc[k] = std::fma(float(lane[k] - x[b].lanes[k]), scale, acc[k]);
        }
        out[r] = reduce16(acc);
    }
}

#ifdef SD_X86
// --- AVX-512 (F, BW, VBMI) kernels, up to four rows at a time sharing the activations ---

// A block's F16 scale by the F16C instruction (half_at's portable conversion branches: twice the time of Q2_0's whole down on Zen 5).
SD_TARGET("f16c")
inline float half_f16c(const uint8_t* p) { uint16_t bits; std::memcpy(&bits, p, 2); return SD_HALF_TO_FLOAT(bits); }
inline uint64_t grid_row(const uint8_t* grid, unsigned offset) { uint64_t row; std::memcpy(&row, grid + offset, 8); return row; }
// The eight grid rows at these byte offsets, as 64 bytes.
SD_TARGET("avx512f,avx512bw,avx2")
inline __m512i grid64(const uint8_t* grid, const uint16_t* offset) {
    const __m128i a = _mm_set_epi64x(int64_t(grid_row(grid, offset[1])), int64_t(grid_row(grid, offset[0])));
    const __m128i b = _mm_set_epi64x(int64_t(grid_row(grid, offset[3])), int64_t(grid_row(grid, offset[2])));
    const __m128i c = _mm_set_epi64x(int64_t(grid_row(grid, offset[5])), int64_t(grid_row(grid, offset[4])));
    const __m128i d = _mm_set_epi64x(int64_t(grid_row(grid, offset[7])), int64_t(grid_row(grid, offset[6])));
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set_m128i(b, a)), _mm256_set_m128i(d, c), 1);
}
// Word orders as tables (there is no _mm512_setr_epi16): a group's scale twice, and the chunk's four groups a 128 bits each.
alignas(64) constexpr int16_t scale_pairs[32] = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29,
                                                 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29};
alignas(64) constexpr int16_t chunk_groups[4][32] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
    {8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11},
    {12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15}};
// The IQ2_XXS block's index bytes as zero-extended words: one VBMI permute with the odd bytes masked to zero.
alignas(64) constexpr uint8_t index_words[64] = {0, 0, 1, 0, 2, 0, 3, 0, 8, 0, 9, 0, 10, 0, 11, 0, 16, 0, 17, 0, 18, 0, 19, 0, 24, 0, 25, 0, 26, 0, 27, 0,
                                                 32, 0, 33, 0, 34, 0, 35, 0, 40, 0, 41, 0, 42, 0, 43, 0, 48, 0, 49, 0, 50, 0, 51, 0, 56, 0, 57, 0, 58, 0, 59, 0};
// A block's integer work for one row, prepared at once with vector code: its 32 grid rows' byte offsets, sign bytes (one a grid row) and
// 16 group scales (2 ls + 1, a 16-value group each).
struct IntPrepared {
    alignas(64) uint16_t offset[32];
    alignas(64) uint16_t code[32];     // IQ2_XXS: each grid row as eight 2-bit codes (iq2_xxs_codes)
    alignas(64) uint8_t sign[32];
    alignas(64) int16_t scale[32];
    const uint8_t* signs;
};
// IQ2_XXS's grid rows hold only 8, 25 and 43: a row is eight 2-bit codes (0, 1, 2), and the 256 rows' codes (512 bytes) fit in eight
// registers, so a block's 32 rows are looked up there, not in memory. Null if the grid is not of those values.
const uint16_t* iq2_xxs_codes() {
    static const std::vector<uint16_t> codes = [] {
        const uint8_t* grid = iq_table(16);
        std::vector<uint16_t> out(256);
        for (int row = 0; row < 256; ++row)
            for (int j = 0; j < 8; ++j) {
                const uint8_t value = grid[8 * row + j];
                if (value != 8 && value != 25 && value != 43) return std::vector<uint16_t>();
                out[row] |= uint16_t((value == 8 ? 0 : value == 25 ? 1 : 2) << (2 * j));
            }
        return out;
    }();
    return codes.empty() ? nullptr : codes.data();
}
struct CodeTable { __m512i part[8]; };
SD_TARGET("avx512f,avx512bw,avx512vbmi,avx2")
inline __m512i codes_of(const CodeTable& table, __m512i index) {
    const __m512i r0 = _mm512_permutex2var_epi16(table.part[0], index, table.part[1]), r1 = _mm512_permutex2var_epi16(table.part[2], index, table.part[3]);
    const __m512i r2 = _mm512_permutex2var_epi16(table.part[4], index, table.part[5]), r3 = _mm512_permutex2var_epi16(table.part[6], index, table.part[7]);
    const __mmask32 bit6 = _mm512_test_epi16_mask(index, _mm512_set1_epi16(64)), bit7 = _mm512_test_epi16_mask(index, _mm512_set1_epi16(128));
    return _mm512_mask_mov_epi16(_mm512_mask_mov_epi16(r0, bit6, r1), bit7, _mm512_mask_mov_epi16(r2, bit6, r3));
}
SD_TARGET("avx512f,avx512bw,avx512vbmi,avx2")
inline void prepare_iq2_xxs(const uint8_t* block, IntPrepared& p, __m512i low, __m512i high, const CodeTable* table) {
    // Bytes 2..66: per 32-value group four index bytes, then its aux word (four 7-bit sign fields and the scale nibble).
    const __m512i words = _mm512_loadu_si512(block + 2);
    const __m512i index = _mm512_maskz_permutexvar_epi8(0x5555555555555555ull, _mm512_load_si512(index_words), words);
    const __m256i aux = _mm512_castsi512_si256(_mm512_permutexvar_epi32(_mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0), words));
    if (table) _mm512_store_si512(p.code, codes_of(*table, index));
    else _mm512_store_si512(p.offset, _mm512_slli_epi16(index, 3));
    // (the 128-entry sign table, seven bits and their parity, is in two registers: one VBMI lookup takes all 32 fields)
    _mm256_store_si256(reinterpret_cast<__m256i*>(p.sign),
                       _mm512_castsi512_si256(_mm512_permutex2var_epi8(low, _mm512_castsi256_si512(aux_fields(aux)), high)));
    // (each group's scale for both of its 16-value halves)
    const __m512i scale = _mm512_cvtepu32_epi64(_mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux, 28), 1), _mm256_set1_epi32(1)));
    const __m512i doubled = _mm512_or_si512(scale, _mm512_slli_epi64(scale, 16));
    _mm512_store_si512(p.scale, _mm512_permutexvar_epi16(_mm512_load_si512(scale_pairs), doubled));
    p.signs = p.sign;
}
SD_TARGET("avx512f,avx512bw,avx512vbmi,avx2")
inline void prepare_iq2_s(const uint8_t* b, IntPrepared& p) {
    // Word g takes qh[g / 4] in both bytes; shifting by 2 (g % 4) brings its two bits to bits 8 and 9 (as the F32 kernels do).
    const __m512i qh = _mm512_broadcastq_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 66)));
    const __m512i spread = _mm512_setr_epi32(0x00000000, 0x00000000, 0x01010101, 0x01010101, 0x02020202, 0x02020202, 0x03030303, 0x03030303,
                                             0x04040404, 0x04040404, 0x05050505, 0x05050505, 0x06060606, 0x06060606, 0x07070707, 0x07070707);
    const __m512i high = _mm512_srlv_epi16(_mm512_shuffle_epi8(qh, spread), _mm512_set1_epi64(0x0006000400020000));
    const __m512i index = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2))),
                                          _mm512_and_si512(high, _mm512_set1_epi16(0x300)));
    _mm512_store_si512(p.offset, _mm512_slli_epi16(index, 3));
    const __m256i nibbles = _mm256_cvtepu8_epi16(nibbles16(b + 74));
    _mm256_store_si256(reinterpret_cast<__m256i*>(p.scale), _mm256_add_epi16(_mm256_slli_epi16(nibbles, 1), _mm256_set1_epi16(1)));
    p.signs = b + 34;
}
// One 64-value chunk c of a prepared row against x's chunk q: its sixteen lanes added to `lanes`.
SD_TARGET("avx512f,avx512bw,avx512vbmi,avx2")
inline __m512i chunk_lanes(const uint8_t* grid, bool coded, const IntPrepared& p, unsigned c, __m512i q, __m512i lanes) {
    uint64_t signs; std::memcpy(&signs, p.signs + 8 * c, 8);
    __m512i weights;
    if (coded) {
        // (a row's code in each 64 bits; multishift takes its bits 2j.. into byte j, and the byte's two low bits name 8, 25 or 43)
        const __m512i rows = _mm512_cvtepu16_epi64(_mm_load_si128(reinterpret_cast<const __m128i*>(p.code + 8 * c)));
        const __m512i fields = _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_set1_epi64(0x0E0C0A0806040200), rows), _mm512_set1_epi8(3));
        weights = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_setr_epi8(8, 25, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), fields);
    } else weights = grid64(grid, p.offset + 8 * c);
    const __m512i signed_q = _mm512_mask_sub_epi8(q, __mmask64(signs), _mm512_setzero_si512(), q);
    // (each 128 bits: sixteen values of one group: the chunk's groups 4c .. 4c + 3)
    const __m512i scales = _mm512_permutexvar_epi16(_mm512_load_si512(chunk_groups[c]), _mm512_load_si512(p.scale));
    return _mm512_add_epi32(lanes, _mm512_madd_epi16(_mm512_maddubs_epi16(weights, signed_q), scales));
}
// A block's integer lanes times its F32 scale, added to the row's F32 lanes.
SD_TARGET("avx512f,avx512bw,avx2")
inline __m512 scaled_lanes(__m512i lanes, float scale, __m512 acc) { return _mm512_fmadd_ps(_mm512_cvtepi32_ps(lanes), _mm512_set1_ps(scale), acc); }
template<int kind, int N>
SD_TARGET("avx512f,avx512bw,avx512vbmi,avx2,f16c")
void iq2_int_rows512(const uint8_t* row, size_t stride, const Q8K* x, size_t blocks, float* out) {
    constexpr size_t bytes = kind == 16 ? 66 : 82;
    const uint8_t* grid = iq_table(kind);
    const __m512i low = _mm512_loadu_si512(sign_table.value), high = _mm512_loadu_si512(sign_table.value + 64);
    const uint16_t* codes = kind == 16 ? iq2_xxs_codes() : nullptr;
    CodeTable table;
    if (codes) for (int k = 0; k < 8; ++k) table.part[k] = _mm512_loadu_si512(codes + 32 * k);
    // (the accumulators are separate locals, not arrays: GCC kept an array of them in memory, four 512-bit stores every block, and
    // cl.exe does the same: rows512)
    __m512 acc0 = _mm512_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    IntPrepared prepared[N];
    for (size_t b = 0; b < blocks; ++b) {
        prefetch_next<bytes, N>(row, stride, b);
        for (int r = 0; r < N; ++r) {
            const uint8_t* block = row + size_t(r) * stride + b * bytes;
            if constexpr (kind == 16) prepare_iq2_xxs(block, prepared[r], low, high, codes ? &table : nullptr);
            else prepare_iq2_s(block, prepared[r]);
        }
        __m512i lanes0 = _mm512_setzero_si512(), lanes1 = lanes0, lanes2 = lanes0, lanes3 = lanes0;
        for (unsigned c = 0; c < 4; ++c) {
            const __m512i q = _mm512_load_si512(x[b].q + 64 * c);
            lanes0 = chunk_lanes(grid, codes != nullptr, prepared[0], c, q, lanes0);
            if constexpr (N > 1) lanes1 = chunk_lanes(grid, codes != nullptr, prepared[1], c, q, lanes1);
            if constexpr (N > 2) lanes2 = chunk_lanes(grid, codes != nullptr, prepared[2], c, q, lanes2);
            if constexpr (N > 3) lanes3 = chunk_lanes(grid, codes != nullptr, prepared[3], c, q, lanes3);
        }
        const float d = x[b].d * 0.125f;
        const uint8_t* block = row + b * bytes;
        acc0 = scaled_lanes(lanes0, half_f16c(block) * d, acc0);
        if constexpr (N > 1) acc1 = scaled_lanes(lanes1, half_f16c(block + stride) * d, acc1);
        if constexpr (N > 2) acc2 = scaled_lanes(lanes2, half_f16c(block + 2 * stride) * d, acc2);
        if constexpr (N > 3) acc3 = scaled_lanes(lanes3, half_f16c(block + 3 * stride) * d, acc3);
    }
    out[0] = total512(acc0);
    if constexpr (N > 1) out[1] = total512(acc1);
    if constexpr (N > 2) out[2] = total512(acc2);
    if constexpr (N > 3) out[3] = total512(acc3);
}
// Q2_0's 2-bit codes, each 128 bits m = 0..3 taking bits 2m and 2m + 1 of the block's 16 code bytes: a 16-bit variable shift and a
// mask, or (GFNI) one affine transform whose matrix in each 64 bits moves those two bits down (3 uops on Intel against 1).
template<bool gfni>
SD_TARGET("avx512f,avx512bw,gfni")
inline __m512i q2_int_codes(const uint8_t* block) {
    const __m512i bytes = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2)));
    if constexpr (gfni) {
        // (row 7 - i of a byte's matrix picks its output bit i: bit 0 <- bit 2m, bit 1 <- bit 2m + 1)
        const __m512i matrix = _mm512_setr_epi64(0x0102000000000000, 0x0102000000000000, 0x0408000000000000, 0x0408000000000000,
                                                 0x1020000000000000, 0x1020000000000000, 0x4080000000000000, 0x4080000000000000);
        return _mm512_gf2p8affine_epi64_epi8(bytes, matrix, 0);
    } else {
        const __m512i shifts = _mm512_setr_epi64(0, 0, 0x0002000200020002, 0x0002000200020002, 0x0004000400040004, 0x0004000400040004,
                                                 0x0006000600060006, 0x0006000600060006);
        return _mm512_and_si512(_mm512_srlv_epi16(bytes, shifts), _mm512_set1_epi8(3));
    }
}
template<int N, bool gfni>
SD_TARGET("avx512f,avx512bw,avx2,f16c,gfni")
void q2_0_int_rows512(const uint8_t* row, size_t stride, const Q8S* x, size_t blocks, float* out) {
    const __m512i ones = _mm512_set1_epi16(1);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (size_t b = 0; b < blocks; ++b) {
        prefetch_next<18, N>(row, stride, b);
        const __m512i q = _mm512_load_si512(x[b].q);
        const __m512i bias = _mm512_load_si512(x[b].lanes);
        const uint8_t* block = row + 18 * b;
#define SD_Q2_INT(r, acc) \
        acc = scaled_lanes(_mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(q2_int_codes<gfni>(block + (r) * stride), q), ones), bias), \
                           half_f16c(block + (r) * stride) * x[b].d, acc);
        SD_Q2_INT(0, acc0)
        if constexpr (N > 1) { SD_Q2_INT(1, acc1) }
        if constexpr (N > 2) { SD_Q2_INT(2, acc2) }
        if constexpr (N > 3) { SD_Q2_INT(3, acc3) }
#undef SD_Q2_INT
    }
    out[0] = total512(acc0);
    if constexpr (N > 1) out[1] = total512(acc1);
    if constexpr (N > 2) out[2] = total512(acc2);
    if constexpr (N > 3) out[3] = total512(acc3);
}
// --- AVX2 kernels: the AVX-512 kernels' sixteen lanes as two 256-bit halves (lanes 0-7 and 8-15), the same integers and the same F32
// operations in the same order, so the same bits. For CPUs without AVX-512 (Intel's 12th to 14th generation, Zen 3 and before) ---

// Four grid rows' eight bytes each, as 32 bytes.
SD_TARGET("avx2,fma,f16c")
inline __m256i grid32(const uint8_t* grid, const uint16_t* offset) {
    return _mm256_set_epi64x(int64_t(grid_row(grid, offset[3])), int64_t(grid_row(grid, offset[2])), int64_t(grid_row(grid, offset[1])),
                             int64_t(grid_row(grid, offset[0])));
}
// IQ2_XXS's 7-bit sign fields as eight bytes each, 0xFF where the value is negative (sign_table's bit for bit, its parity bit too).
struct SignMasks {
    uint64_t value[128];
    constexpr SignMasks() : value() {
        for (unsigned bits = 0; bits < 128; ++bits) {
            unsigned parity = bits ^ (bits >> 4); parity ^= parity >> 2; parity ^= parity >> 1;
            const unsigned signs = bits | ((parity & 1u) << 7);
            for (unsigned j = 0; j < 8; ++j) if (signs >> j & 1) value[bits] |= uint64_t(0xFF) << (8 * j);
        }
    }
};
constexpr SignMasks sign_masks;
// IQ2_XXS's group g (32 values) read straight from its block against x's 32 values: the four grid rows its index bytes name, the
// signs of its four 7-bit fields, its one scale; eight of the AVX-512 kernels' lanes (no prepared block in between).
SD_TARGET("avx2,fma,f16c")
inline __m256i iq2_xxs_group256(const uint8_t* grid, const uint8_t* block, unsigned g, __m256i q, __m256i lanes) {
    const uint8_t* group = block + 2 + 8 * g;
    const uint32_t aux = load32(group + 4);
    const __m256i weights = _mm256_set_epi64x(int64_t(grid_row(grid, 8u * group[3])), int64_t(grid_row(grid, 8u * group[2])),
                                              int64_t(grid_row(grid, 8u * group[1])), int64_t(grid_row(grid, 8u * group[0])));
    const __m256i negative = _mm256_set_epi64x(int64_t(sign_masks.value[(aux >> 21) & 127]), int64_t(sign_masks.value[(aux >> 14) & 127]),
                                               int64_t(sign_masks.value[(aux >> 7) & 127]), int64_t(sign_masks.value[aux & 127]));
    const __m256i signed_q = _mm256_sub_epi8(_mm256_xor_si256(q, negative), negative);
    const __m256i scale = _mm256_set1_epi16(int16_t(2 * (aux >> 28) + 1));
    return _mm256_add_epi32(lanes, _mm256_madd_epi16(_mm256_maddubs_epi16(weights, signed_q), scale));
}
// A sign byte as eight bytes, 0xFF where its bit is set (IQ2_S's sign bytes).
struct ByteMasks {
    uint64_t value[256];
    constexpr ByteMasks() : value() {
        for (unsigned bits = 0; bits < 256; ++bits)
            for (unsigned j = 0; j < 8; ++j) if (bits >> j & 1) value[bits] |= uint64_t(0xFF) << (8 * j);
    }
};
constexpr ByteMasks byte_masks;
// IQ2_S's group g (32 values) read straight from its block: four ten-bit grid indices (a byte each and two bits of qh[g]), four
// sign bytes, the scale nibbles of its two 16-value halves; eight of the AVX-512 kernels' lanes.
SD_TARGET("avx2,fma,f16c")
inline __m256i iq2_s_group256(const uint8_t* grid, const uint8_t* block, unsigned g, __m256i q, __m256i lanes) {
    const uint8_t* qs = block + 2 + 4 * g;
    const unsigned high = block[66 + g];
    auto row = [&](unsigned l) { return int64_t(grid_row(grid, 8u * (qs[l] | ((high << (8 - 2 * l)) & 0x300)))); };
    const __m256i weights = _mm256_set_epi64x(row(3), row(2), row(1), row(0));
    const uint8_t* signs = block + 34 + 4 * g;
    const __m256i negative = _mm256_set_epi64x(int64_t(byte_masks.value[signs[3]]), int64_t(byte_masks.value[signs[2]]),
                                               int64_t(byte_masks.value[signs[1]]), int64_t(byte_masks.value[signs[0]]));
    const __m256i signed_q = _mm256_sub_epi8(_mm256_xor_si256(q, negative), negative);
    const unsigned nibbles = block[74 + g];
    const __m256i scales = _mm256_set_m128i(_mm_set1_epi16(int16_t(2 * (nibbles >> 4) + 1)), _mm_set1_epi16(int16_t(2 * (nibbles & 15) + 1)));
    return _mm256_add_epi32(lanes, _mm256_madd_epi16(_mm256_maddubs_epi16(weights, signed_q), scales));
}
template<int kind, int N>
SD_TARGET("avx2,fma,f16c")
void iq2_int_rows256(const uint8_t* row, size_t stride, const Q8K* x, size_t blocks, float* out) {
    constexpr size_t bytes = kind == 16 ? 66 : 82;
    const uint8_t* grid = iq_table(kind);
    __m256 low0 = _mm256_setzero_ps(), low1 = low0, low2 = low0, low3 = low0, high0 = low0, high1 = low0, high2 = low0, high3 = low0;
    for (size_t b = 0; b < blocks; ++b) {
        prefetch_next<bytes, N>(row, stride, b);
        __m256i l0 = _mm256_setzero_si256(), l1 = l0, l2 = l0, l3 = l0, h0 = l0, h1 = l0, h2 = l0, h3 = l0;
        for (unsigned c = 0; c < 4; ++c) {
            const __m256i ql = _mm256_load_si256(reinterpret_cast<const __m256i*>(x[b].q + 64 * c));
            const __m256i qh = _mm256_load_si256(reinterpret_cast<const __m256i*>(x[b].q + 64 * c + 32));
#define SD_IQ2_HALVES(r, l, h) \
            if constexpr (kind == 16) { \
                l = iq2_xxs_group256(grid, row + (r) * stride + b * bytes, 2 * c, ql, l); \
                h = iq2_xxs_group256(grid, row + (r) * stride + b * bytes, 2 * c + 1, qh, h); \
            } else { \
                l = iq2_s_group256(grid, row + (r) * stride + b * bytes, 2 * c, ql, l); \
                h = iq2_s_group256(grid, row + (r) * stride + b * bytes, 2 * c + 1, qh, h); \
            }
            SD_IQ2_HALVES(0, l0, h0)
            if constexpr (N > 1) { SD_IQ2_HALVES(1, l1, h1) }
            if constexpr (N > 2) { SD_IQ2_HALVES(2, l2, h2) }
            if constexpr (N > 3) { SD_IQ2_HALVES(3, l3, h3) }
#undef SD_IQ2_HALVES
        }
        const float d = x[b].d * 0.125f;
        const uint8_t* block = row + b * bytes;
#define SD_IQ2_SCALE(r, l, h, low, high) { \
            const __m256 scale = _mm256_set1_ps(half_f16c(block + (r) * stride) * d); \
            low = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l), scale, low); high = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h), scale, high); }
        SD_IQ2_SCALE(0, l0, h0, low0, high0)
        if constexpr (N > 1) SD_IQ2_SCALE(1, l1, h1, low1, high1)
        if constexpr (N > 2) SD_IQ2_SCALE(2, l2, h2, low2, high2)
        if constexpr (N > 3) SD_IQ2_SCALE(3, l3, h3, low3, high3)
#undef SD_IQ2_SCALE
    }
    out[0] = total256(low0, high0);
    if constexpr (N > 1) out[1] = total256(low1, high1);
    if constexpr (N > 2) out[2] = total256(low2, high2);
    if constexpr (N > 3) out[3] = total256(low3, high3);
}
// Q2_0's down: the 16 code bytes in both 128-bit halves, each 128 bits m taking bits 2m and 2m + 1 (a 32-bit shift: what crosses a
// byte lands above the two bits kept).
template<int N>
SD_TARGET("avx2,fma,f16c")
void q2_0_int_rows256(const uint8_t* row, size_t stride, const Q8S* x, size_t blocks, float* out) {
    const __m256i shift_low = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2), shift_high = _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6);
    const __m256i three = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    __m256 low0 = _mm256_setzero_ps(), low1 = low0, low2 = low0, low3 = low0, high0 = low0, high1 = low0, high2 = low0, high3 = low0;
    for (size_t b = 0; b < blocks; ++b) {
        prefetch_next<18, N>(row, stride, b);
        const __m256i ql = _mm256_load_si256(reinterpret_cast<const __m256i*>(x[b].q));
        const __m256i qh = _mm256_load_si256(reinterpret_cast<const __m256i*>(x[b].q + 32));
        const __m256i bias_low = _mm256_load_si256(reinterpret_cast<const __m256i*>(x[b].lanes));
        const __m256i bias_high = _mm256_load_si256(reinterpret_cast<const __m256i*>(x[b].lanes + 8));
        const uint8_t* block = row + 18 * b;
#define SD_Q2_INT256(r, low, high) { \
            const uint8_t* at = block + (r) * stride; \
            const __m256i codes = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(at + 2))); \
            const __m256i cl = _mm256_and_si256(_mm256_srlv_epi32(codes, shift_low), three), ch = _mm256_and_si256(_mm256_srlv_epi32(codes, shift_high), three); \
            const __m256 scale = _mm256_set1_ps(half_f16c(at) * x[b].d); \
            low = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(cl, ql), ones), bias_low)), scale, low); \
            high = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ch, qh), ones), bias_high)), scale, high); }
        SD_Q2_INT256(0, low0, high0)
        if constexpr (N > 1) SD_Q2_INT256(1, low1, high1)
        if constexpr (N > 2) SD_Q2_INT256(2, low2, high2)
        if constexpr (N > 3) SD_Q2_INT256(3, low3, high3)
#undef SD_Q2_INT256
    }
    out[0] = total256(low0, high0);
    if constexpr (N > 1) out[1] = total256(low1, high1);
    if constexpr (N > 2) out[2] = total256(low2, high2);
    if constexpr (N > 3) out[3] = total256(low3, high3);
}
// `count` rows four at a time, then the rest.
template<typename Rows>
inline void by_fours(const uint8_t* row, size_t stride, size_t count, float* out, Rows rows) {
    size_t r = 0;
    for (; r + 4 <= count; r += 4) rows(std::integral_constant<int, 4>{}, row + r * stride, out + r);
    switch (count - r) {
        case 3: rows(std::integral_constant<int, 3>{}, row + r * stride, out + r); break;
        case 2: rows(std::integral_constant<int, 2>{}, row + r * stride, out + r); break;
        case 1: rows(std::integral_constant<int, 1>{}, row + r * stride, out + r); break;
        default: break;
    }
}
void iq2_xxs_int_avx512(const uint8_t* row, size_t stride, size_t count, const Q8K* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { iq2_int_rows512<16, decltype(n)::value>(rows, stride, x, blocks, o); });
}
void iq2_s_int_avx512(const uint8_t* row, size_t stride, size_t count, const Q8K* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { iq2_int_rows512<22, decltype(n)::value>(rows, stride, x, blocks, o); });
}
void q2_0_int_avx512(const uint8_t* row, size_t stride, size_t count, const Q8S* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { q2_0_int_rows512<decltype(n)::value, false>(rows, stride, x, blocks, o); });
}
void q2_0_int_gfni(const uint8_t* row, size_t stride, size_t count, const Q8S* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { q2_0_int_rows512<decltype(n)::value, true>(rows, stride, x, blocks, o); });
}
void iq2_xxs_int_avx2(const uint8_t* row, size_t stride, size_t count, const Q8K* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { iq2_int_rows256<16, decltype(n)::value>(rows, stride, x, blocks, o); });
}
void iq2_s_int_avx2(const uint8_t* row, size_t stride, size_t count, const Q8K* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { iq2_int_rows256<22, decltype(n)::value>(rows, stride, x, blocks, o); });
}
void q2_0_int_avx2(const uint8_t* row, size_t stride, size_t count, const Q8S* x, size_t blocks, float* out) {
    by_fours(row, stride, count, out, [&](auto n, const uint8_t* rows, float* o) { q2_0_int_rows256<decltype(n)::value>(rows, stride, x, blocks, o); });
}
#endif

// The integer row kernels of `kind` at `level` (2: AVX-512 with VBMI, and GFNI where the CPU has it; 3: the same without GFNI, for the
// tests; 1: AVX2; else the scalar reference), or null where the format has none.
using GateKernel = void (*)(const uint8_t*, size_t, size_t, const Q8K*, size_t, float*);
using DownKernel = void (*)(const uint8_t*, size_t, size_t, const Q8S*, size_t, float*);
GateKernel gate_int_kernel(int kind, int level) {
#ifdef SD_X86
    if (level >= 2) {
        if (kind == 16) return iq2_xxs_int_avx512;
        if (kind == 22) return iq2_s_int_avx512;
    }
    if (level == 1) {
        if (kind == 16) return iq2_xxs_int_avx2;
        if (kind == 22) return iq2_s_int_avx2;
    }
#endif
    (void)level;
    if (kind == 16) return iq2_xxs_int_scalar;
    if (kind == 22) return iq2_s_int_scalar;
    return nullptr;
}
DownKernel down_int_kernel(int kind, int level) {
#ifdef SD_X86
    if (level >= 2 && kind == 42) {
        static const bool gfni = cpu_has("gfni");
        return level == 2 && gfni ? q2_0_int_gfni : q2_0_int_avx512;
    }
    if (level == 1 && kind == 42) return q2_0_int_avx2;
#endif
    (void)level;
    return kind == 42 ? q2_0_int_scalar : nullptr;
}
