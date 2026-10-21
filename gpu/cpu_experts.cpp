// Independently authored CPU expert engine for partially resident models. Routed
// Q2_0 experts that are not in GPU memory are evaluated here, directly from the
// authenticated read-only model mapping, in F32: every value is scale*(code-1)
// times an F32 activation with F32 accumulation (no activation quantization).
// Kernels are selected at run time: AVX-512F, AVX2+FMA, or portable scalar.
#include "straydog_gpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#if defined(__linux__)
#include <fstream>
#include <map>
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SD_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define SD_TARGET(x)
// MSVC has no scalar F16C conversion; the packed one converts the same half exactly.
#define SD_HALF_TO_FLOAT(bits) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int(bits))))
#else
#define SD_TARGET(x) __attribute__((target(x)))
#define SD_HALF_TO_FLOAT(bits) _cvtsh_ss(bits)
#endif
#endif

#include "cpu_workers.hpp"

namespace {
thread_local std::string cpu_error;

float half_value(uint16_t bits) {
    const uint32_t sign = uint32_t(bits >> 15) << 31, exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    uint32_t value;
    if (exponent == 0) {
        if (!mantissa) value = sign;
        else {
            int shift = 0; uint32_t m = mantissa;
            while (!(m & 1024)) { m <<= 1; ++shift; }
            value = sign | uint32_t(113 - shift) << 23 | (m & 1023) << 13;
        }
    } else if (exponent == 31) value = sign | 0x7F800000u | mantissa << 13;
    else value = sign | (exponent + 112) << 23 | mantissa << 13;
    float result; std::memcpy(&result, &value, 4);
    return result;
}

// One Q2_0 row: blocks of 64 values (FP16 scale + 16 bytes of two-bit codes).
float row_scalar(const uint8_t* row, const float* x, size_t blocks) {
    float total = 0;
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* block = row + 18 * b;
        uint16_t bits; std::memcpy(&bits, block, 2);
        float sum = 0;
        for (size_t i = 0; i < 64; ++i) sum += float(int((block[2 + i / 4] >> ((i % 4) * 2)) & 3) - 1) * x[64 * b + i];
        total += half_value(bits) * sum;
    }
    return total;
}

#ifdef SD_X86
SD_TARGET("avx2,fma,f16c")
float row_avx2(const uint8_t* row, const float* x, size_t blocks) {
    const __m256 table = _mm256_setr_ps(-1, 0, 1, 2, 0, 0, 0, 0);
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), three = _mm256_set1_epi32(3);
    __m256 total = _mm256_setzero_ps();
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* block = row + 18 * b;
        uint16_t bits; std::memcpy(&bits, block, 2);
        uint16_t codes[8]; std::memcpy(codes, block + 2, 16);
        const float* values = x + 64 * b;
        __m256 sum = _mm256_setzero_ps();
        for (int q = 0; q < 8; ++q) {
            const __m256i index = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(codes[q]), shifts), three);
            sum = _mm256_fmadd_ps(_mm256_permutevar8x32_ps(table, index), _mm256_loadu_ps(values + 8 * q), sum);
        }
        total = _mm256_fmadd_ps(_mm256_set1_ps(SD_HALF_TO_FLOAT(bits)), sum, total);
    }
    __m128 low = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    low = _mm_add_ss(low, _mm_shuffle_ps(low, low, 1));
    return _mm_cvtss_f32(low);
}

// AVX-512: each code byte zero-extends to a 32-bit lane; immediate shifts expose its
// four codes in turn and vpermilps maps the low two bits to codes-1 within each
// 128-bit lane (no cross-lane shuffles). Vector m of a block holds values 4i+m, so
// activations are passed in that order (permute_lanes). Products are exact and
// accumulate in F32.
SD_TARGET("avx512f,f16c")
float row_avx512(const uint8_t* row, const float* x, size_t blocks) {
    const __m512 table = _mm512_setr_ps(-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2);
    __m512 total = _mm512_setzero_ps();
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* block = row + 18 * b;
        uint16_t bits; std::memcpy(&bits, block, 2);
        const __m512i codes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2)));
        const float* values = x + 64 * b;
        __m512 sum = _mm512_mul_ps(_mm512_permutevar_ps(table, codes), _mm512_loadu_ps(values));
        sum = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes, 2)), _mm512_loadu_ps(values + 16), sum);
        sum = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes, 4)), _mm512_loadu_ps(values + 32), sum);
        sum = _mm512_fmadd_ps(_mm512_permutevar_ps(table, _mm512_srli_epi32(codes, 6)), _mm512_loadu_ps(values + 48), sum);
        // Convert one scale, then broadcast its exact F32 value; converting sixteen
        // replicated halves spends wide-conversion throughput on identical values.
        total = _mm512_fmadd_ps(_mm512_set1_ps(SD_HALF_TO_FLOAT(bits)), sum, total);
    }
    return _mm512_reduce_add_ps(total);
}

bool cpu_has(const char* feature) {
#if defined(_MSC_VER) && !defined(__clang__)
    int info[4];
    __cpuidex(info, 7, 0);
    const bool f = info[1] & (1 << 16), avx2 = info[1] & (1 << 5);
    __cpuid(info, 1);
    const bool fma = info[2] & (1 << 12), f16c = info[2] & (1 << 29), osxsave = info[2] & (1 << 27);
    const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    const bool ymm = (xcr0 & 6) == 6, zmm = (xcr0 & 0xE6) == 0xE6;
    if (!std::strcmp(feature, "avx512")) return f && f16c && zmm;
    return avx2 && fma && f16c && ymm;
#else
    __builtin_cpu_init();
    if (!std::strcmp(feature, "avx512")) return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("f16c");
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
}
#endif

using RowKernel = float (*)(const uint8_t*, const float*, size_t);

// Other expert formats decode one block to F32 (every value exact: a scale times
// small integers) and accumulate value*activation with fused multiply-adds in 16
// lanes, reduced 8/4/2/1 apart; the scalar, AVX2 and AVX-512 paths agree bitwise.
// IQ1/IQ2/IQ3 grids are gguf's reference tables (one byte per value, sd_iq_tables);
// IQ1 rows store value + 1 (-1, 0, 1 as 0, 1, 2).
std::mutex iq_mutex;
std::vector<uint8_t> iq_tables[6];   // IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S/IQ1_M
constexpr size_t iq_table_bytes[6] = {256 * 8, 512 * 8, 1024 * 8, 256 * 4, 512 * 4, 2048 * 8};
int iq_index(int kind) {
    return kind == 16 ? 0 : kind == 17 ? 1 : kind == 22 ? 2 : kind == 18 ? 3 : kind == 21 ? 4 : kind == 29 || kind == 19 ? 5 : -1;
}
const uint8_t* iq_table(int kind) {
    const int index = iq_index(kind);
    if (index < 0 || iq_tables[index].empty()) throw std::runtime_error("IQ grid tables were not provided");
    return iq_tables[index].data();
}
// The same grids as F32 values for the fused row kernels (IQ1: value - 1, as decoded).
std::vector<float> iq_floats[6];
const float* iq_float(int kind) {
    const int index = iq_index(kind);
    if (index < 0 || iq_floats[index].empty()) throw std::runtime_error("IQ grid tables were not provided");
    return iq_floats[index].data();
}
constexpr int iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// MXFP4: doubled E2M1 values; the E8M0 block scale is halved to match, 2^(e - 128).
constexpr int fp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? 0x00200000u << e : uint32_t(e - 1) << 23;
    float value; std::memcpy(&value, &bits, 4);
    return value;
}
unsigned iq_signs(unsigned bits) {
    unsigned parity = bits;
    parity ^= parity >> 4; parity ^= parity >> 2; parity ^= parity >> 1;
    return bits | ((parity & 1u) << 7);
}
// The same as a table (the fused kernels; parity per group would cost more than the lookup).
struct SignTable {
    uint8_t value[128];
    constexpr SignTable() : value() {
        for (unsigned bits = 0; bits < 128; ++bits) {
            unsigned parity = bits ^ (bits >> 4); parity ^= parity >> 2; parity ^= parity >> 1;
            value[bits] = uint8_t(bits | ((parity & 1u) << 7));
        }
    }
};
constexpr SignTable sign_table;
unsigned iq_signs_table(unsigned bits) { return sign_table.value[bits & 127]; }
// IQ1_M delta, +-1/8 by bit 3, without a branch (real data mixes both signs).
float iq1_delta(unsigned nibble) {
    const uint32_t bits = 0x3E000000u | (uint32_t(nibble & 8) << 28);
    float value; std::memcpy(&value, &bits, 4);
    return value;
}
uint32_t load32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
float half_at(const uint8_t* p) { uint16_t bits; std::memcpy(&bits, p, 2); return half_value(bits); }
float iq_sign(unsigned signs, unsigned j) { return (signs >> j) & 1 ? -1.0f : 1.0f; }

void decode_q8_0(const uint8_t* b, float* out) {
    const float d = half_at(b);
    for (int i = 0; i < 32; ++i) out[i] = d * float(int8_t(b[2 + i]));
}
// BF16, taken 32 values (64 bytes) at a time: each the high half of an F32.
void decode_bf16(const uint8_t* b, float* out) {
    for (int i = 0; i < 32; ++i) {
        const uint32_t bits = uint32_t(b[2 * i] | b[2 * i + 1] << 8) << 16;
        std::memcpy(out + i, &bits, 4);
    }
}
void decode_iq4_nl(const uint8_t* b, float* out) {
    const float d = half_at(b);
    for (int i = 0; i < 16; ++i) { out[i] = d * float(iq4_values[b[2 + i] & 15]); out[i + 16] = d * float(iq4_values[b[2 + i] >> 4]); }
}
void decode_iq4_xs(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const unsigned high = unsigned(b[2]) | unsigned(b[3]) << 8;
    for (int g = 0; g < 8; ++g) {
        const int scale = int(((b[4 + g / 2] >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4)) - 32;
        const float dl = d * float(scale);
        for (int j = 0; j < 16; ++j) {
            out[32 * g + j] = dl * float(iq4_values[b[8 + 16 * g + j] & 15]);
            out[32 * g + j + 16] = dl * float(iq4_values[b[8 + 16 * g + j] >> 4]);
        }
    }
}
void decode_iq2_xxs(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(16);
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t aux = load32(b + 2 + 8 * ib + 4);
        const float db = d * (0.5f + float(aux >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* g = grid + 8 * b[2 + 8 * ib + l];
            const unsigned signs = iq_signs((aux >> (7 * l)) & 127);
            for (int j = 0; j < 8; ++j) out[32 * ib + 8 * l + j] = db * float(g[j]) * iq_sign(signs, j);
        }
    }
}
void decode_iq2_xs(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(17);
    for (int ib = 0; ib < 8; ++ib)
        for (int l = 0; l < 4; ++l) {
            const unsigned q = unsigned(b[2 + 2 * (4 * ib + l)]) | unsigned(b[3 + 2 * (4 * ib + l)]) << 8;
            const float db = d * (0.5f + float((b[66 + ib] >> (4 * (l / 2))) & 15)) * 0.25f;
            const uint8_t* g = grid + 8 * (q & 511);
            for (int j = 0; j < 8; ++j) out[32 * ib + 8 * l + j] = db * float(g[j]) * iq_sign(iq_signs(q >> 9), j);
        }
}
void decode_iq2_s(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(22);
    for (int ib = 0; ib < 8; ++ib)
        for (int l = 0; l < 4; ++l) {
            const unsigned index = b[2 + 4 * ib + l] | ((unsigned(b[66 + ib]) << (8 - 2 * l)) & 0x300);
            const float db = d * (0.5f + float((b[74 + ib] >> (4 * (l / 2))) & 15)) * 0.25f;
            const uint8_t* g = grid + 8 * index;
            for (int j = 0; j < 8; ++j) out[32 * ib + 8 * l + j] = db * float(g[j]) * iq_sign(b[34 + 4 * ib + l], j);
        }
}
void decode_iq3_xxs(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(18);
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t aux = load32(b + 66 + 4 * ib);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const unsigned signs = iq_signs((aux >> (7 * l)) & 127);
            for (int j = 0; j < 8; ++j)
                out[32 * ib + 8 * l + j] = db * float(grid[4 * b[2 + 8 * ib + 2 * l + j / 4] + j % 4]) * iq_sign(signs, j);
        }
    }
}
// IQ1_M (56 bytes): qs[32], qh[16], then four sub-scale words whose top nibbles hold the FP16 scale.
float iq1m_scale(const uint8_t* b, uint16_t (&words)[4]) {
    std::memcpy(words, b + 48, 8);
    return half_value(uint16_t((words[0] >> 12) | ((words[1] >> 8) & 0xf0) | ((words[2] >> 4) & 0xf00) | (words[3] & 0xf000)));
}
float iq1m_group_scale(float d, const uint16_t (&words)[4], int ib, int half) {
    return d * float(2 * ((words[ib / 2] >> (6 * (ib % 2) + 3 * half)) & 7) + 1);
}
void decode_iq1_m(const uint8_t* b, float* out) {
    uint16_t words[4];
    const float d = iq1m_scale(b, words);
    const uint8_t* grid = iq_table(29);
    for (int ib = 0; ib < 8; ++ib)
        for (int l = 0; l < 4; ++l) {
            const unsigned high = b[32 + 2 * ib + l / 2] >> (4 * (l % 2));
            const float dl = iq1m_group_scale(d, words, ib, l / 2), delta = iq1_delta(high);
            const uint8_t* g = grid + 8 * (b[4 * ib + l] | ((high & 7) << 8));
            for (int j = 0; j < 8; ++j) out[32 * ib + 8 * l + j] = dl * (float(g[j]) - 1.0f + delta);
        }
}
// IQ1_S (50 bytes): d, qs[32], then per 32 values a 16-bit word of four 3-bit grid-row high parts,
// a 3-bit scale and the delta's sign.
void decode_iq1_s(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(19);
    for (int ib = 0; ib < 8; ++ib) {
        const unsigned qh = unsigned(b[34 + 2 * ib]) | unsigned(b[35 + 2 * ib]) << 8;
        const float dl = d * float(2 * ((qh >> 12) & 7) + 1), delta = qh & 0x8000 ? -0.125f : 0.125f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* g = grid + 8 * (b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8));
            for (int j = 0; j < 8; ++j) out[32 * ib + 8 * l + j] = dl * (float(g[j]) - 1.0f + delta);
        }
    }
}
// Q2_K (84 bytes): scales[16] qs[64] d dmin; group g of 16 values has scale byte g (scale and minimum
// nibbles) and two bits at shift 2 * ((g / 2) % 4) of code bytes 32 * (g / 8) + 16 * (g % 2) ..
void decode_q2_k(const uint8_t* b, float* out) {
    const float d = half_at(b + 80), minimum = half_at(b + 82);
    for (int g = 0; g < 16; ++g) {
        const float dl = d * float(b[g] & 15), ml = minimum * float(b[g] >> 4);
        const uint8_t* codes = b + 16 + 32 * (g / 8) + 16 * (g % 2);
        for (int l = 0; l < 16; ++l) out[16 * g + l] = dl * float((codes[l] >> (2 * ((g / 2) % 4))) & 3) - ml;
    }
}
// MXFP4 (17 bytes): an E8M0 exponent, then 16 bytes of E2M1 codes (low nibbles values 0-15).
void decode_mxfp4(const uint8_t* b, float* out) {
    const float scale = e8m0_half(b[0]);
    for (int j = 0; j < 16; ++j) {
        out[j] = float(fp4_values[b[1 + j] & 15]) * scale;
        out[j + 16] = float(fp4_values[b[1 + j] >> 4]) * scale;
    }
}
void decode_iq3_s(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(21);
    for (int ib = 0; ib < 8; ++ib) {
        const float db = d * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        for (int l = 0; l < 4; ++l)
            for (int j = 0; j < 8; ++j) {
                const unsigned high = unsigned(j / 4);
                const unsigned index = b[2 + 8 * ib + 2 * l + high] | ((unsigned(b[66 + ib]) << (8 - 2 * l - high)) & 256);
                out[32 * ib + 8 * l + j] = db * float(grid[4 * index + j % 4]) * iq_sign(b[74 + 4 * ib + l], j);
            }
    }
}
#ifdef SD_X86
// AVX-512 block decoders producing the same values bit for bit: grid rows are
// gathered and widened, scaled by the same F32 factor, and signs flip the sign bit
// (as the scalar multiplication by -1 does, zeros included).
SD_TARGET("avx512f,avx2")
inline void store_scaled(float* out, __m128i bytes, float scale, uint32_t signs) {
    __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes)), _mm512_set1_ps(scale));
    const __m512i bits = _mm512_castps_si512(v);
    v = _mm512_castsi512_ps(_mm512_mask_xor_epi32(bits, __mmask16(signs), bits, _mm512_set1_epi32(int(0x80000000u))));
    _mm512_storeu_ps(out, v);
}
SD_TARGET("avx512f,avx2")
void decode_iq3_s_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const int* grid = reinterpret_cast<const int*>(iq_table(21));
    for (int ib = 0; ib < 8; ++ib) {
        const float db = d * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        alignas(32) int32_t index[8];
        for (int t = 0; t < 8; ++t) index[t] = int32_t(b[2 + 8 * ib + t] | ((unsigned(b[66 + ib]) << (8 - t)) & 256));
        const __m256i g = _mm256_i32gather_epi32(grid, _mm256_load_si256(reinterpret_cast<const __m256i*>(index)), 4);
        const uint32_t signs = load32(b + 74 + 4 * ib);
        store_scaled(out + 32 * ib, _mm256_castsi256_si128(g), db, signs & 0xffff);
        store_scaled(out + 32 * ib + 16, _mm256_extracti128_si256(g, 1), db, signs >> 16);
    }
}
SD_TARGET("avx512f,avx2")
void decode_iq3_xxs_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const int* grid = reinterpret_cast<const int*>(iq_table(18));
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t aux = load32(b + 66 + 4 * ib);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        alignas(32) int32_t index[8];
        for (int t = 0; t < 8; ++t) index[t] = b[2 + 8 * ib + t];
        const __m256i g = _mm256_i32gather_epi32(grid, _mm256_load_si256(reinterpret_cast<const __m256i*>(index)), 4);
        uint32_t signs = 0;
        for (int l = 0; l < 4; ++l) signs |= iq_signs((aux >> (7 * l)) & 127) << (8 * l);
        store_scaled(out + 32 * ib, _mm256_castsi256_si128(g), db, signs & 0xffff);
        store_scaled(out + 32 * ib + 16, _mm256_extracti128_si256(g, 1), db, signs >> 16);
    }
}
// IQ2: four eight-byte grid rows per 32 values; scales may differ between halves.
SD_TARGET("avx512f,avx2")
void store_iq2(float* out, const uint8_t* grid, const int32_t (&index)[4], float first, float second, uint32_t signs) {
    const __m256i g = _mm256_i32gather_epi64(reinterpret_cast<const long long*>(grid),
                                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(index)), 8);
    store_scaled(out, _mm256_castsi256_si128(g), first, signs & 0xffff);
    store_scaled(out + 16, _mm256_extracti128_si256(g, 1), second, signs >> 16);
}
SD_TARGET("avx512f,avx2")
void decode_iq2_xxs_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(16);
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t aux = load32(b + 2 + 8 * ib + 4);
        const float db = d * (0.5f + float(aux >> 28)) * 0.25f;
        int32_t index[4];
        uint32_t signs = 0;
        for (int l = 0; l < 4; ++l) { index[l] = b[2 + 8 * ib + l]; signs |= iq_signs((aux >> (7 * l)) & 127) << (8 * l); }
        store_iq2(out + 32 * ib, grid, index, db, db, signs);
    }
}
SD_TARGET("avx512f,avx2")
void decode_iq2_xs_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(17);
    for (int ib = 0; ib < 8; ++ib) {
        int32_t index[4];
        uint32_t signs = 0;
        for (int l = 0; l < 4; ++l) {
            const unsigned q = unsigned(b[2 + 2 * (4 * ib + l)]) | unsigned(b[3 + 2 * (4 * ib + l)]) << 8;
            index[l] = int32_t(q & 511); signs |= iq_signs(q >> 9) << (8 * l);
        }
        const float first = d * (0.5f + float(b[66 + ib] & 15)) * 0.25f, second = d * (0.5f + float(b[66 + ib] >> 4)) * 0.25f;
        store_iq2(out + 32 * ib, grid, index, first, second, signs);
    }
}
SD_TARGET("avx512f,avx2")
void decode_iq2_s_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const uint8_t* grid = iq_table(22);
    for (int ib = 0; ib < 8; ++ib) {
        int32_t index[4];
        for (int l = 0; l < 4; ++l) index[l] = int32_t(b[2 + 4 * ib + l] | ((unsigned(b[66 + ib]) << (8 - 2 * l)) & 0x300));
        const float first = d * (0.5f + float(b[74 + ib] & 15)) * 0.25f, second = d * (0.5f + float(b[74 + ib] >> 4)) * 0.25f;
        store_iq2(out + 32 * ib, grid, index, first, second, load32(b + 34 + 4 * ib));
    }
}
// IQ1_M: sixteen values (two grid rows) of one group scale, each row with its own delta.
SD_TARGET("avx512f,avx2")
inline void store_iq1(float* out, __m128i bytes, float scale, float first, float second) {
    const __m512 g = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes)), _mm512_set1_ps(1.0f));
    const __m512 delta = _mm512_mask_blend_ps(__mmask16(0xff00), _mm512_set1_ps(first), _mm512_set1_ps(second));
    _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_set1_ps(scale), _mm512_add_ps(g, delta)));
}
SD_TARGET("avx512f,avx2")
void decode_iq1_m_avx512(const uint8_t* b, float* out) {
    uint16_t words[4];
    const float d = iq1m_scale(b, words);
    const uint8_t* grid = iq_table(29);
    for (int ib = 0; ib < 8; ++ib) {
        int32_t index[4];
        float delta[4];
        for (int l = 0; l < 4; ++l) {
            const unsigned high = b[32 + 2 * ib + l / 2] >> (4 * (l % 2));
            index[l] = int32_t(b[4 * ib + l] | ((high & 7) << 8));
            delta[l] = iq1_delta(high);
        }
        const __m256i g = _mm256_i32gather_epi64(reinterpret_cast<const long long*>(grid),
                                                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(index)), 8);
        store_iq1(out + 32 * ib, _mm256_castsi256_si128(g), iq1m_group_scale(d, words, ib, 0), delta[0], delta[1]);
        store_iq1(out + 32 * ib + 16, _mm256_extracti128_si256(g, 1), iq1m_group_scale(d, words, ib, 1), delta[2], delta[3]);
    }
}
// IQ4: sixteen nonlinear values looked up in one register.
SD_TARGET("avx512f,avx2")
inline void store_iq4(float* out, __m128i bytes, int shift, float scale) {
    const __m512 table = _mm512_setr_ps(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
    const __m512i codes = _mm512_and_si512(_mm512_srli_epi32(_mm512_cvtepu8_epi32(bytes), unsigned(shift)), _mm512_set1_epi32(15));
    _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_set1_ps(scale), _mm512_permutexvar_ps(codes, table)));
}
SD_TARGET("avx512f,avx2")
void decode_iq4_nl_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 2));
    store_iq4(out, bytes, 0, d); store_iq4(out + 16, bytes, 4, d);
}
SD_TARGET("avx512f,avx2")
void decode_iq4_xs_avx512(const uint8_t* b, float* out) {
    const float d = half_at(b);
    const unsigned high = unsigned(b[2]) | unsigned(b[3]) << 8;
    for (int g = 0; g < 8; ++g) {
        const int scale = int(((b[4 + g / 2] >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4)) - 32;
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 8 + 16 * g));
        const float dl = d * float(scale);
        store_iq4(out + 32 * g, bytes, 0, dl); store_iq4(out + 32 * g + 16, bytes, 4, dl);
    }
}
SD_TARGET("avx512f,avx2")
void decode_q8_0_avx512(const uint8_t* b, float* out) {
    const __m512 d = _mm512_set1_ps(half_at(b));
    for (int i = 0; i < 2; ++i) {
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 2 + 16 * i));
        _mm512_storeu_ps(out + 16 * i, _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q))));
    }
}
#endif
float reduce16(float (&acc)[16]) {
    for (int width = 8; width; width /= 2)
        for (int k = 0; k < width; ++k) acc[k] += acc[k + width];
    return acc[0];
}
// count rows of one bank (stride bytes apart) against x; level 2 = AVX-512, 1 = AVX2.
using RowsKernel = void (*)(int level, const uint8_t* row, size_t stride, size_t count, const float* x, size_t cols,
                            float* out);
#ifdef SD_X86
#include "cpu_rows.hpp"
#endif
struct CpuFormat {
    size_t values, bytes;
    void (*decode)(const uint8_t*, float*);
    void (*wide)(const uint8_t*, float*);
    RowsKernel rows;   // fused multi-row kernel (x86), else null
};
CpuFormat cpu_format(int kind) {
#ifdef SD_X86
    #define SD_WIDE(name) name##_avx512
    #define SD_ROWS(kind) block_rows<kind>
#else
    #define SD_WIDE(name) name
    #define SD_ROWS(kind) nullptr
#endif
    switch (kind) {
        case 8: return {32, 34, decode_q8_0, SD_WIDE(decode_q8_0), SD_ROWS(8)};
        case 30: return {32, 64, decode_bf16, decode_bf16, nullptr};
        case 20: return {32, 18, decode_iq4_nl, SD_WIDE(decode_iq4_nl), SD_ROWS(20)};
        case 23: return {256, 136, decode_iq4_xs, SD_WIDE(decode_iq4_xs), SD_ROWS(23)};
        case 16: return {256, 66, decode_iq2_xxs, SD_WIDE(decode_iq2_xxs), SD_ROWS(16)};
        case 17: return {256, 74, decode_iq2_xs, SD_WIDE(decode_iq2_xs), SD_ROWS(17)};
        case 22: return {256, 82, decode_iq2_s, SD_WIDE(decode_iq2_s), SD_ROWS(22)};
        case 18: return {256, 98, decode_iq3_xxs, SD_WIDE(decode_iq3_xxs), SD_ROWS(18)};
        case 21: return {256, 110, decode_iq3_s, SD_WIDE(decode_iq3_s), SD_ROWS(21)};
        case 29: return {256, 56, decode_iq1_m, SD_WIDE(decode_iq1_m), SD_ROWS(29)};
        // No wide decoders: their fused row kernels take the AVX paths.
        case 19: return {256, 50, decode_iq1_s, decode_iq1_s, SD_ROWS(19)};
        case 10: return {256, 84, decode_q2_k, decode_q2_k, SD_ROWS(10)};
        case 39: return {32, 17, decode_mxfp4, decode_mxfp4, SD_ROWS(39)};
    }
    #undef SD_WIDE
    #undef SD_ROWS
    throw std::runtime_error("unsupported Straydog CPU expert format");
}
float decoded_scalar(const CpuFormat& f, const uint8_t* row, const float* x, size_t cols) {
    float w[256], acc[16] = {};
    for (size_t b = 0; b < cols / f.values; ++b) {
        f.decode(row + b * f.bytes, w);
        for (size_t i = 0; i < f.values; i += 16)
            for (int k = 0; k < 16; ++k) acc[k] = std::fma(w[i + k], x[b * f.values + i + k], acc[k]);
    }
    return reduce16(acc);
}
#ifdef SD_X86
SD_TARGET("avx2,fma")
float decoded_avx2(const CpuFormat& f, const uint8_t* row, const float* x, size_t cols) {
    alignas(32) float w[256], acc[16];
    __m256 low = _mm256_setzero_ps(), high = _mm256_setzero_ps();
    for (size_t b = 0; b < cols / f.values; ++b) {
        f.decode(row + b * f.bytes, w);
        for (size_t i = 0; i < f.values; i += 16) {
            low = _mm256_fmadd_ps(_mm256_load_ps(w + i), _mm256_loadu_ps(x + b * f.values + i), low);
            high = _mm256_fmadd_ps(_mm256_load_ps(w + i + 8), _mm256_loadu_ps(x + b * f.values + i + 8), high);
        }
    }
    _mm256_store_ps(acc, low); _mm256_store_ps(acc + 8, high);
    return reduce16(acc);
}
SD_TARGET("avx512f")
float decoded_avx512(const CpuFormat& f, const uint8_t* row, const float* x, size_t cols) {
    alignas(64) float w[256], acc[16];
    __m512 sum = _mm512_setzero_ps();
    for (size_t b = 0; b < cols / f.values; ++b) {
        f.wide(row + b * f.bytes, w);
        for (size_t i = 0; i < f.values; i += 16)
            sum = _mm512_fmadd_ps(_mm512_load_ps(w + i), _mm512_loadu_ps(x + b * f.values + i), sum);
    }
    _mm512_store_ps(acc, sum);
    return reduce16(acc);
}
#endif
using DecodedKernel = float (*)(const CpuFormat&, const uint8_t*, const float*, size_t);

// Activation order for row_avx512: within each 64-value block, position 16m+i holds value 4i+m.
size_t lane_position(size_t index) {
    const size_t within = index % 64;
    return index - within + 16 * (within % 4) + within / 4;
}

// Hardware threads grouped by physical core (Linux sysfs, within the process
// affinity; Windows processor-core records, per processor group), ordered by their
// lowest CPU. Each worker is bound to one core's siblings: two workers never share
// a core, yet the scheduler may still pick either sibling. Elsewhere, or without
// topology, threads float.
#if defined(__linux__) || defined(_WIN32)
using sd_cpu::physical_cores;
#endif
using sd_cpu::Workers;

struct Engine {
    std::unique_ptr<Workers> workers;
    RowKernel kernel = row_scalar;
    DecodedKernel decoded = decoded_scalar;
    int level = 0;        // fused row kernels: 2 = AVX-512, 1 = AVX2, 0 = none (decoded rows)
    bool lanes = false;   // the Q2_0 kernel takes activations in lane_position order
    const char* isa = "scalar";
    std::vector<float> gate_up, products, x_lanes, down_rows;
    uint64_t calls = 0, experts = 0, projection_ns = 0, product_ns = 0, down_ns = 0;
};
uint64_t elapsed_ns(std::chrono::steady_clock::time_point since) {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

float silu(float x) {
    const float e = std::exp(-std::fabs(x));
    return x * (x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e));
}
// Round-to-nearest-even FP32 -> FP16 bits (finite inputs).
uint16_t half_bits(float value) {
    uint32_t x; std::memcpy(&x, &value, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exponent = int((x >> 23) & 255) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFFu;
    if (exponent >= 31) return uint16_t(sign | 0x7C00u);
    if (exponent <= 0) {
        if (exponent < -10) return uint16_t(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1), midpoint = 1u << (shift - 1);
        if (rest > midpoint || (rest == midpoint && (half & 1))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = (uint32_t(exponent) << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1))) ++half;
    return uint16_t(sign | half);
}

// Deterministic Q2_0 encoding of BF16 values (used for Straydog's own MTP drafter
// experts): each 64-value block keeps the FP16 scale with the least squared error
// among 66 candidates +-amax*f/2, f in [0.2, 1], codes clamp(rint(w/s), -1, 2) + 1.
void quantize_q2_0_blocks(const uint16_t* input, size_t first, size_t last, uint8_t* output) {
    for (size_t block = first; block < last; ++block) {
        float w[64], amax = 0;
        for (int i = 0; i < 64; ++i) {
            const uint32_t bits = uint32_t(input[block * 64 + i]) << 16;
            std::memcpy(&w[i], &bits, 4);
            amax = std::max(amax, std::fabs(w[i]));
        }
        float best = INFINITY; uint16_t best_scale = 0; int8_t best_codes[64] = {};
        for (int step = 0; step <= 32 && amax > 0; ++step) {
            const float fraction = 0.2f + 0.8f * float(step) / 32.0f;
            for (int sign = 1; sign >= -1; sign -= 2) {
                const uint16_t bits = half_bits(float(sign) * amax * fraction / 2.0f);
                const float scale = half_value(bits);
                if (scale == 0) continue;
                float error = 0; int8_t codes[64];
                for (int i = 0; i < 64; ++i) {
                    const float code = std::min(2.0f, std::max(-1.0f, std::nearbyint(w[i] / scale)));
                    codes[i] = int8_t(code);
                    const float delta = code * scale - w[i];
                    error += delta * delta;
                }
                if (error < best) { best = error; best_scale = bits; std::memcpy(best_codes, codes, 64); }
            }
        }
        uint8_t* out = output + block * 18;
        std::memcpy(out, &best_scale, 2);
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = 0;
            for (int j = 0; j < 4; ++j) byte |= uint8_t((best_codes[4 * i + j] + 1) & 3) << (2 * j);
            out[2 + i] = byte;
        }
    }
}

}  // namespace

extern "C" {
const char* sd_cpu_error() { return cpu_error.c_str(); }

void* sd_cpu_experts_create(uint32_t threads) {
    try {
        if (!threads) {
#if defined(__linux__) || defined(_WIN32)
            threads = uint32_t(physical_cores().size());
#endif
            if (!threads) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        }
        if (threads > 256) throw std::runtime_error("invalid Straydog CPU expert thread count");
        auto engine = std::make_unique<Engine>();
#ifdef SD_X86
        if (cpu_has("avx512")) {
            engine->kernel = row_avx512; engine->decoded = decoded_avx512; engine->lanes = true; engine->isa = "avx512-f32";
            engine->level = 2;
        } else if (cpu_has("avx2")) {
            engine->kernel = row_avx2; engine->decoded = decoded_avx2; engine->isa = "avx2-fma"; engine->level = 1;
        }
#endif
        engine->workers = std::make_unique<Workers>(threads);
        return engine.release();
    } catch (const std::exception& e) { cpu_error = e.what(); return nullptr; }
}

void sd_cpu_experts_destroy(void* handle) { delete static_cast<Engine*>(handle); }

const char* sd_cpu_experts_isa(void* handle) { return handle ? static_cast<Engine*>(handle)->isa : ""; }

// The gguf reference grids of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S, one byte per
// value. Set before GPU devices are created (each device uploads them); a repeated
// call must pass identical tables.
int sd_iq_tables(const uint8_t* iq2_xxs, const uint8_t* iq2_xs, const uint8_t* iq2_s, const uint8_t* iq3_xxs,
                 const uint8_t* iq3_s, const uint8_t* iq1_s) {
    try {
        const uint8_t* tables[6] = {iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s};
        std::lock_guard<std::mutex> lock(iq_mutex);
        for (int i = 0; i < 6; ++i) {
            if (!tables[i]) throw std::runtime_error("invalid IQ grid table");
            if (!iq_tables[i].empty() && !std::equal(iq_tables[i].begin(), iq_tables[i].end(), tables[i]))
                throw std::runtime_error("IQ grid tables cannot change");
        }
        for (int i = 0; i < 6; ++i) {
            iq_tables[i].assign(tables[i], tables[i] + iq_table_bytes[i]);
            iq_floats[i].resize(iq_table_bytes[i]);
            for (size_t j = 0; j < iq_table_bytes[i]; ++j) iq_floats[i][j] = float(tables[i][j]) - (i == 5 ? 1.0f : 0.0f);
        }
        return 0;
    } catch (const std::exception& e) { cpu_error = e.what(); return -1; }
}
// Decode one block of `kind` with the portable (wide=0) or the selected wide decoder;
// both must produce identical values.
int sd_cpu_decode(int kind, int wide, const uint8_t* block, float* out) {
    try {
        if (!block || !out) throw std::runtime_error("invalid Straydog CPU decode request");
        const CpuFormat format = cpu_format(kind);
#ifdef SD_X86
        if (wide && !cpu_has("avx512")) throw std::runtime_error("AVX-512 is not available");
#endif
        (wide ? format.wide : format.decode)(block, out);
        return 0;
    } catch (const std::exception& e) { cpu_error = e.what(); return -1; }
}
// Test hook: count rows (stride bytes apart) of one format against x, through the fused
// kernel at level (2 = AVX-512, 1 = AVX2) or the per-row decoded scalar path (0).
int sd_cpu_rows(int kind, int level, const uint8_t* row, uint64_t stride, uint32_t count, const float* x, uint64_t cols,
                float* out) {
    try {
        if (!row || !x || !out || !count || !cols) throw std::runtime_error("invalid Straydog CPU rows request");
        const CpuFormat format = cpu_format(kind);
        if (cols % format.values || stride < cols / format.values * format.bytes)
            throw std::runtime_error("invalid Straydog CPU rows shape");
        if (level == 0) {
            for (uint32_t i = 0; i < count; ++i) out[i] = decoded_scalar(format, row + i * stride, x, size_t(cols));
            return 0;
        }
#ifdef SD_X86
        if ((level == 2 && cpu_has("avx512")) || (level == 1 && cpu_has("avx2"))) {
            format.rows(level, row, size_t(stride), count, x, size_t(cols), out);
            return 0;
        }
#endif
        throw std::runtime_error("this CPU lacks that instruction set");
    } catch (const std::exception& e) { cpu_error = e.what(); return -1; }
}
const uint8_t* sd_iq_grid(int kind, size_t* bytes) {
    std::lock_guard<std::mutex> lock(iq_mutex);
    const int index = iq_index(kind);
    if (index < 0 || iq_tables[index].empty()) { if (bytes) *bytes = 0; return nullptr; }
    if (bytes) *bytes = iq_table_bytes[index];
    return iq_tables[index].data();
}

int sd_quantize_q2_0(const uint16_t* input, uint64_t count, uint8_t* output) {
    try {
        if (!input || !output || !count || count % 64) throw std::runtime_error("invalid Q2_0 quantization request");
        const size_t blocks = size_t(count / 64);
        const size_t threads = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), (blocks + 4095) / 4096));
        std::vector<std::thread> pool;
        for (size_t t = 0; t < threads; ++t)
            pool.emplace_back(quantize_q2_0_blocks, input, blocks * t / threads, blocks * (t + 1) / threads, output);
        for (auto& thread : pool) thread.join();
        return 0;
    } catch (const std::exception& e) { cpu_error = e.what(); return -1; }
}

// calls, experts, projection_ns, product_ns, down_ns
int sd_cpu_experts_stats(void* handle, uint64_t* values) {
    if (!handle || !values) { cpu_error = "invalid Straydog CPU expert statistics"; return -1; }
    const auto* engine = static_cast<Engine*>(handle);
    values[0] = engine->calls; values[1] = engine->experts; values[2] = engine->projection_ns;
    values[3] = engine->product_ns; values[4] = engine->down_ns;
    return 0;
}

// out[hidden] = sum_i weights[i] * down_i(silu(gate_i x) * up_i x) for expert slices of
// kinds[0..2] (gate, up, down): Q2_0 or a format with a block decoder (cpu_format), at any
// hidden and inter that are whole blocks.
int sd_cpu_experts_run(void* handle, const int* kinds, const uint8_t* const* gate, const uint8_t* const* up,
                       const uint8_t* const* down, const float* weights, uint32_t count, uint64_t hidden, uint64_t inter,
                       const float* x, float* out) {
    try {
        auto* engine = static_cast<Engine*>(handle);
        if (!engine || !kinds || !gate || !up || !down || !weights || !x || !out || !count || count > 64 ||
            !hidden || !inter || hidden > 65536 || inter > 65536)
            throw std::runtime_error("invalid Straydog CPU expert request");
        for (uint32_t i = 0; i < count; ++i)
            if (!gate[i] || !up[i] || !down[i] || !std::isfinite(weights[i]))
                throw std::runtime_error("invalid Straydog CPU expert operand");
        // Per role: the Q2_0 kernel (possibly lane-ordered activations) or a block decoder.
        CpuFormat formats[3]{};
        size_t bytes[3];
        for (int role = 0; role < 3; ++role) {
            const size_t cols = role == 2 ? inter : hidden;
            if (kinds[role] == 42) {
                if (cols % 64) throw std::runtime_error("invalid Straydog CPU expert shape");
                bytes[role] = cols / 64 * 18;
            } else {
                formats[role] = cpu_format(kinds[role]);
                if (cols % formats[role].values) throw std::runtime_error("invalid Straydog CPU expert shape");
                bytes[role] = cols / formats[role].values * formats[role].bytes;
            }
        }
        engine->gate_up.resize(size_t(count) * 2 * inter);
        engine->products.resize(size_t(count) * inter);
        float* gate_up = engine->gate_up.data();
        float* products = engine->products.data();
        const RowKernel kernel = engine->kernel;
        const DecodedKernel decoded = engine->decoded;
        const bool lanes = engine->lanes;
        const size_t projection_rows = size_t(count) * 2 * inter;
        const float* lane_input = x;
        if (lanes && (kinds[0] == 42 || kinds[1] == 42)) {
            engine->x_lanes.resize(hidden);
            for (size_t i = 0; i < hidden; ++i) engine->x_lanes[lane_position(i)] = x[i];
            lane_input = engine->x_lanes.data();
        }
        auto project = [&](int role, const uint8_t* row, const float* natural, const float* ordered, size_t cols) {
            return kinds[role] == 42 ? kernel(row, ordered, cols / 64) : decoded(formats[role], row, natural, cols);
        };
        // Consecutive rows of one bank go to the fused kernel together (same values and order).
        const int level = engine->level;
        auto project_rows = [&](int role, const uint8_t* row, size_t count, const float* natural, const float* ordered,
                                size_t cols, float* out) {
            if (kinds[role] != 42 && level && formats[role].rows) {
                formats[role].rows(level, row, bytes[role], count, natural, cols, out);
                return;
            }
            for (size_t i = 0; i < count; ++i) out[i] = project(role, row + i * bytes[role], natural, ordered, cols);
        };
        auto phase = std::chrono::steady_clock::now();
        engine->workers->run_blocks(projection_rows, 32, [&](size_t first, size_t last) {
            for (size_t row = first; row < last;) {
                const size_t expert = row / (2 * inter), local = row % (2 * inter);
                const int role = local < inter ? 0 : 1;
                const size_t end = std::min(last, expert * 2 * inter + size_t(role + 1) * inter);
                const uint8_t* bank = role == 0 ? gate[expert] : up[expert];
                project_rows(role, bank + (local % inter) * bytes[role], end - row, x, lane_input, hidden, gate_up + row);
                row = end;
            }
        });
        engine->projection_ns += elapsed_ns(phase);
        phase = std::chrono::steady_clock::now();
        const size_t product_count = size_t(count) * inter;
        const bool ordered = lanes && kinds[2] == 42;   // inter is then a multiple of 64
        engine->workers->run_blocks(product_count, 128, [&](size_t first, size_t last) {
            for (size_t i = first; i < last; ++i) {
                const size_t expert = i / inter, j = i % inter;
                const float value = silu(gate_up[expert * 2 * inter + j]) * gate_up[expert * 2 * inter + inter + j];
                products[ordered ? lane_position(i) : i] = value;
            }
        });
        engine->product_ns += elapsed_ns(phase);
        phase = std::chrono::steady_clock::now();
        engine->down_rows.resize(size_t(count) * hidden);
        float* down_rows = engine->down_rows.data();
        engine->workers->run_blocks(hidden, 32, [&](size_t first, size_t last) {
            for (size_t expert = 0; expert < count; ++expert) {
                const float* p = products + expert * inter;
                project_rows(2, down[expert] + first * bytes[2], last - first, p, p, inter, down_rows + expert * hidden + first);
            }
            for (size_t row = first; row < last; ++row) {
                float sum = 0;
                for (size_t expert = 0; expert < count; ++expert) sum += weights[expert] * down_rows[expert * hidden + row];
                out[row] = sum;
            }
        });
        engine->down_ns += elapsed_ns(phase);
        ++engine->calls; engine->experts += count;
        return 0;
    } catch (const std::exception& e) { cpu_error = e.what(); return -1; }
}

// Copy `count` byte ranges (sources[i] -> targets[i], bytes[i]) on every worker: prefill gathers
// absent experts from the read-only model mapping into page-locked staging this way, so cold
// pages fault in concurrently and resident ones copy at memory bandwidth. Bytes only.
int sd_cpu_gather(void* handle, const uint8_t* const* sources, uint8_t* const* targets, const uint64_t* bytes,
                  uint32_t count) {
    try {
        auto* engine = static_cast<Engine*>(handle);
        if (!engine || !sources || !targets || !bytes || count > 4096)
            throw std::runtime_error("invalid Straydog CPU gather request");
        uint64_t total = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (!sources[i] || !targets[i]) throw std::runtime_error("invalid Straydog CPU gather range");
            total += bytes[i];
        }
        if (!total) return 0;
        // 64 KiB units of the concatenated ranges, claimed by the workers as they free up.
        constexpr uint64_t unit = 64 * 1024;
        engine->workers->run_blocks(size_t((total + unit - 1) / unit), 4, [&](size_t first_unit, size_t last_unit) {
            const uint64_t first = first_unit * unit, last = std::min(total, uint64_t(last_unit) * unit);
            uint64_t offset = 0;
            for (uint32_t i = 0; i < count && offset < last; offset += bytes[i], ++i) {
                const uint64_t begin = std::max(first, offset), end = std::min(last, offset + bytes[i]);
                if (begin < end) std::memcpy(targets[i] + (begin - offset), sources[i] + (begin - offset), size_t(end - begin));
            }
        });
        return 0;
    } catch (const std::exception& e) { cpu_error = e.what(); return -1; }
}
}
