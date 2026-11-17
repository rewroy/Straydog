#pragma once
#include "link_tail_common.hpp"
// Independently authored bandwidth-oriented GEMV. Each format decodes a whole
// aligned unit (a 16-byte vector, a Q2_0 block or a K-quant group pair) per lane
// and multiplies it with F32 activations; there is no activation quantization.
// Several lanes share a row; their partial sums are reduced in F32. Formats or
// layouts outside these fast paths use the reference element decoder.
__device__ __forceinline__ float gemv_half(uint32_t bits) { return __half2float(__ushort_as_half(uint16_t(bits))); }
__device__ __forceinline__ float gemv_byte(uint32_t word, int index) { return float((word >> (8 * index)) & 255u); }
// (high:low) >> 8*shift, and a byte gather from (high:low) with byte selectors 0..7.
#if defined(__HIPCC__)
__device__ __forceinline__ uint32_t gemv_realign(uint32_t low, uint32_t high, uint32_t shift) {
    return __builtin_amdgcn_alignbyte(high, low, shift);
}
__device__ __forceinline__ uint32_t gemv_perm(uint32_t high, uint32_t low, uint32_t select) {
    return __builtin_amdgcn_perm(high, low, select);
}
#else
__device__ __forceinline__ uint32_t gemv_realign(uint32_t low, uint32_t high, uint32_t shift) {
    return __funnelshift_r(low, high, 8 * (shift & 3));
}
__device__ __forceinline__ uint32_t gemv_perm(uint32_t high, uint32_t low, uint32_t select) {
    const uint32_t nibbles = (select & 0x7u) | ((select >> 4) & 0x70u) | ((select >> 8) & 0x700u) | ((select >> 12) & 0x7000u);
    return __byte_perm(low, high, nibbles);
}
#endif
__device__ __forceinline__ float4 gemv_load4(const float* pointer) { return *reinterpret_cast<const float4*>(pointer); }
__device__ __forceinline__ float gemv_sum4(float4 x) { return (x.x + x.y) + (x.z + x.w); }
__device__ __forceinline__ float gemv_dot4(uint32_t word, float4 x, float accumulator) {
    accumulator = fmaf(gemv_byte(word, 0), x.x, accumulator);
    accumulator = fmaf(gemv_byte(word, 1), x.y, accumulator);
    accumulator = fmaf(gemv_byte(word, 2), x.z, accumulator);
    return fmaf(gemv_byte(word, 3), x.w, accumulator);
}

template<int kind> struct GemvFormat;
// dots<V> decodes one unit once and forms V dot products with activations
// x + v*stride; each vector's arithmetic is exactly the one-vector form.

template<> struct GemvFormat<0> {
    static constexpr size_t elements = 4, alignment = 16;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const float4 w = reinterpret_cast<const float4*>(row)[unit];
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float4 a = gemv_load4(x + v * stride + unit * 4);
            r[v] = fmaf(w.x, a.x, fmaf(w.y, a.y, fmaf(w.z, a.z, w.w * a.w)));
        }
    }
};
template<int kind>
__device__ __forceinline__ void gemv_dots8(const float (&w)[8], const float* x, uint32_t stride, size_t unit, float* r, int V) {
    for (int v = 0; v < V; ++v) {
        const float4 a = gemv_load4(x + v * stride + unit * 8), b = gemv_load4(x + v * stride + unit * 8 + 4);
        float sum = w[0] * a.x;
        sum = fmaf(w[1], a.y, sum); sum = fmaf(w[2], a.z, sum);
        sum = fmaf(w[3], a.w, sum); sum = fmaf(w[4], b.x, sum);
        sum = fmaf(w[5], b.y, sum); sum = fmaf(w[6], b.z, sum);
        r[v] = fmaf(w[7], b.w, sum);
    }
}
template<> struct GemvFormat<1> {
    static constexpr size_t elements = 8, alignment = 16;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint4 q = reinterpret_cast<const uint4*>(row)[unit];
        const float w[8] = {gemv_half(q.x), gemv_half(q.x >> 16), gemv_half(q.y), gemv_half(q.y >> 16),
                            gemv_half(q.z), gemv_half(q.z >> 16), gemv_half(q.w), gemv_half(q.w >> 16)};
        #pragma unroll
        for (int v = 0; v < V; ++v) gemv_dots8<1>(w, x + v * stride, 0, unit, &r[v], 1);
    }
};
template<> struct GemvFormat<30> {
    static constexpr size_t elements = 8, alignment = 16;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint4 q = reinterpret_cast<const uint4*>(row)[unit];
        const float w[8] = {__uint_as_float(q.x << 16), __uint_as_float(q.x & 0xffff0000u),
                            __uint_as_float(q.y << 16), __uint_as_float(q.y & 0xffff0000u),
                            __uint_as_float(q.z << 16), __uint_as_float(q.z & 0xffff0000u),
                            __uint_as_float(q.w << 16), __uint_as_float(q.w & 0xffff0000u)};
        #pragma unroll
        for (int v = 0; v < V; ++v) gemv_dots8<30>(w, x + v * stride, 0, unit, &r[v], 1);
    }
};
// Q2_0: one 18-byte block of 64 values; blocks are only two-byte aligned.
template<> struct GemvFormat<42> {
    static constexpr size_t elements = 64, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(row + unit * 18);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const bool odd = (address & 2) != 0;
        uint32_t words[5];
        #pragma unroll
        for (int i = 0; i < 5; ++i) words[i] = base[i];
        const float scale = gemv_half(odd ? words[0] >> 16 : words[0]);
        float sum[V], total[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { sum[v] = 0; total[v] = 0; }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t codes = odd ? words[i + 1] : gemv_realign(words[i], words[i + 1], 2);
            const uint32_t t0 = codes & 0x03030303u, t1 = (codes >> 2) & 0x03030303u;
            const uint32_t t2 = (codes >> 4) & 0x03030303u, t3 = (codes >> 6) & 0x03030303u;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 64;
                const float4 a = gemv_load4(values + 16 * i), b = gemv_load4(values + 16 * i + 4);
                const float4 c = gemv_load4(values + 16 * i + 8), e = gemv_load4(values + 16 * i + 12);
                total[v] += (gemv_sum4(a) + gemv_sum4(b)) + (gemv_sum4(c) + gemv_sum4(e));
                float s = sum[v];
                s = fmaf(gemv_byte(t0, 0), a.x, s); s = fmaf(gemv_byte(t0, 1), b.x, s);
                s = fmaf(gemv_byte(t0, 2), c.x, s); s = fmaf(gemv_byte(t0, 3), e.x, s);
                s = fmaf(gemv_byte(t1, 0), a.y, s); s = fmaf(gemv_byte(t1, 1), b.y, s);
                s = fmaf(gemv_byte(t1, 2), c.y, s); s = fmaf(gemv_byte(t1, 3), e.y, s);
                s = fmaf(gemv_byte(t2, 0), a.z, s); s = fmaf(gemv_byte(t2, 1), b.z, s);
                s = fmaf(gemv_byte(t2, 2), c.z, s); s = fmaf(gemv_byte(t2, 3), e.z, s);
                s = fmaf(gemv_byte(t3, 0), a.w, s); s = fmaf(gemv_byte(t3, 1), b.w, s);
                s = fmaf(gemv_byte(t3, 2), c.w, s); s = fmaf(gemv_byte(t3, 3), e.w, s);
                sum[v] = s;
            }
        }
        // Codes store value + 1; subtract the block sum once instead of per value.
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = scale * (sum[v] - total[v]);
    }
};
// The scales and mins of a Q4_K/Q5_K pair's two groups (the reference's get_scale_min_k4 of groups 2 * pair and 2 * pair + 1; the first
// in the low byte of each word, the second in the next), from the block's 12 packed bytes as the words y, z, w with a group's 8 bits
// apart in a field: pairs 0 and 1 hold them in 6 bits of y (scales) and z (mins), pairs 2 and 3 in the nibbles of w with their top two
// bits in y and z. (An array of the bytes indexed by the pair compiles to chains of compares and selects, some 300 instructions a unit:
// the Q5_K kernels were bound by them, not by the weights.)
__device__ __forceinline__ void gemv_pair_scale_min(int pair, uint32_t y, uint32_t z, uint32_t w, uint32_t& scales, uint32_t& mins) {
    const bool low = pair < 2;
    const unsigned shift = 16 * (pair & 1);
    const uint32_t mask = low ? 0x3F3Fu : 0x0F0Fu;
    scales = (((low ? y : w) >> shift) & mask) | ((((low ? 0u : y) >> shift) >> 2) & 0x3030u);
    mins = (((low ? z : w >> 4) >> shift) & mask) | ((((low ? 0u : z) >> shift) >> 2) & 0x3030u);
}
// Q4_K/Q5_K: one lane owns two 32-value groups that share 32 packed bytes.
template<int kind> struct GemvK45 {
    static constexpr size_t elements = 64, alignment = 16;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        constexpr size_t block_bytes = kind == 13 ? 176 : 144;
        const uint8_t* block = row + (unit / 4) * block_bytes;
        const int pair = int(unit % 4);
        const uint4 header = *reinterpret_cast<const uint4*>(block);
        const uint4* packed = reinterpret_cast<const uint4*>(block + (kind == 13 ? 48 : 16) + 32 * pair);
        const uint4 q0 = packed[0], q1 = packed[1];
        uint4 h0{}, h1{};
        if (kind == 13) { h0 = reinterpret_cast<const uint4*>(block + 16)[0]; h1 = reinterpret_cast<const uint4*>(block + 16)[1]; }
        uint32_t scales, mins;
        gemv_pair_scale_min(pair, header.y, header.z, header.w, scales, mins);
        const unsigned scale0 = scales & 255, scale1 = scales >> 8, min0 = mins & 255, min1 = mins >> 8;
        const uint32_t codes[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t high[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        float a0[V], a1[V], s0[V], s1[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { a0[v] = 0; a1[v] = 0; s0[v] = 0; s1[v] = 0; }
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            uint32_t c0 = codes[m] & 0x0F0F0F0Fu, c1 = (codes[m] >> 4) & 0x0F0F0F0Fu;
            if (kind == 13) {
                c0 |= ((high[m] >> (2 * pair)) & 0x01010101u) << 4;
                c1 |= ((high[m] >> (2 * pair + 1)) & 0x01010101u) << 4;
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 64;
                const float4 xa = gemv_load4(values + 4 * m), xb = gemv_load4(values + 32 + 4 * m);
                a0[v] = gemv_dot4(c0, xa, a0[v]); a1[v] = gemv_dot4(c1, xb, a1[v]);
                s0[v] += gemv_sum4(xa); s1[v] += gemv_sum4(xb);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v)
            r[v] = gemv_half(header.x) * (float(scale0) * a0[v] + float(scale1) * a1[v])
                 - gemv_half(header.x >> 16) * (float(min0) * s0[v] + float(min1) * s1[v]);
    }
};
template<> struct GemvFormat<12> : GemvK45<12> {};
template<> struct GemvFormat<13> : GemvK45<13> {};
// IQ4_XS: nonlinear four-bit values are offset by 128 so byte conversion is unsigned.
__device__ __forceinline__ uint32_t gemv_iq4(uint32_t nibbles) {
    const uint32_t select = nibbles & 0x07070707u;
    const uint32_t low = gemv_perm(0x766A5D4Fu, 0x3F2D1801u, select);
    const uint32_t high = gemv_perm(0xF1D9C5B5u, 0xA6998D81u, select);
    const uint32_t mask = ((nibbles >> 3) & 0x01010101u) * 0xFFu;
    return (low & ~mask) | (high & mask);
}
template<> struct GemvFormat<23> {
    static constexpr size_t elements = 64, alignment = 8;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint8_t* block = row + (unit / 4) * 136;
        const int pair = int(unit % 4);
        const uint2 header = *reinterpret_cast<const uint2*>(block);
        const uint2* packed = reinterpret_cast<const uint2*>(block + 8 + 32 * pair);
        const uint2 qa = packed[0], qb = packed[1], qc = packed[2], qd = packed[3];
        const uint32_t high = header.x >> 16, low = header.y;
        const int g0 = 2 * pair, g1 = 2 * pair + 1;
        const int scale0 = int(((low >> (4 * g0)) & 15) | (((high >> (2 * g0)) & 3) << 4)) - 32;
        const int scale1 = int(((low >> (4 * g1)) & 15) | (((high >> (2 * g1)) & 3) << 4)) - 32;
        const uint32_t first[4] = {qa.x, qa.y, qb.x, qb.y}, second[4] = {qc.x, qc.y, qd.x, qd.y};
        float a0[V], a1[V], s0[V], s1[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { a0[v] = 0; a1[v] = 0; s0[v] = 0; s1[v] = 0; }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t w0 = gemv_iq4(first[m] & 0x0F0F0F0Fu), w1 = gemv_iq4((first[m] >> 4) & 0x0F0F0F0Fu);
            const uint32_t w2 = gemv_iq4(second[m] & 0x0F0F0F0Fu), w3 = gemv_iq4((second[m] >> 4) & 0x0F0F0F0Fu);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 64;
                const float4 l0 = gemv_load4(values + 4 * m), h0 = gemv_load4(values + 16 + 4 * m);
                const float4 l1 = gemv_load4(values + 32 + 4 * m), h1 = gemv_load4(values + 48 + 4 * m);
                a0[v] = gemv_dot4(w0, l0, a0[v]);
                a0[v] = gemv_dot4(w1, h0, a0[v]);
                a1[v] = gemv_dot4(w2, l1, a1[v]);
                a1[v] = gemv_dot4(w3, h1, a1[v]);
                s0[v] += gemv_sum4(l0) + gemv_sum4(h0); s1[v] += gemv_sum4(l1) + gemv_sum4(h1);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v)
            r[v] = gemv_half(header.x) * (float(scale0) * (a0[v] - 128.0f * s0[v]) + float(scale1) * (a1[v] - 128.0f * s1[v]));
    }
};
// Q6_K: one lane owns half of a 210-byte super-block (128 values).
template<> struct GemvFormat<14> {
    static constexpr size_t elements = 128, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(row + (unit / 2) * 210);
        const int half = int(unit % 2);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const uint32_t shift = uint32_t(address & 3);
        uint32_t low[16], high[8], scales[2];
        {
            uint32_t t[17];
            #pragma unroll
            for (int i = 0; i < 17; ++i) t[i] = base[16 * half + i];
            #pragma unroll
            for (int i = 0; i < 16; ++i) low[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        {
            uint32_t t[9];
            #pragma unroll
            for (int i = 0; i < 9; ++i) t[i] = base[32 + 8 * half + i];
            #pragma unroll
            for (int i = 0; i < 8; ++i) high[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        const uint32_t s0 = base[48 + 2 * half], s1 = base[49 + 2 * half], s2 = base[50 + 2 * half];
        scales[0] = gemv_realign(s0, s1, shift); scales[1] = gemv_realign(s1, s2, shift);
        const float d = gemv_half(gemv_realign(base[52], base[53], shift));
        float result[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) result[v] = 0;
        #pragma unroll
        for (int group = 0; group < 8; ++group) {
            const int scale = int(int8_t(uint8_t(scales[group / 4] >> (8 * (group % 4)))));
            float sum[V], total[V];
            #pragma unroll
            for (int v = 0; v < V; ++v) { sum[v] = 0; total[v] = 0; }
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int i = group * 16 + 4 * m;
                const uint32_t l = (low[(i % 64) / 4] >> (4 * (i / 64))) & 0x0F0F0F0Fu;
                const uint32_t h = (high[(i % 32) / 4] >> (2 * (i / 32))) & 0x03030303u;
                #pragma unroll
                for (int v = 0; v < V; ++v) {
                    const float4 a = gemv_load4(x + v * stride + unit * 128 + i);
                    sum[v] = gemv_dot4(l | (h << 4), a, sum[v]);
                    total[v] += gemv_sum4(a);
                }
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) result[v] = fmaf(float(scale), sum[v] - 32.0f * total[v], result[v]);
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d * result[v];
    }
};
// Q3_K: one lane owns half of a 110-byte super-block (128 values).
template<> struct GemvFormat<11> {
    static constexpr size_t elements = 128, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(row + (unit / 2) * 110);
        const int half = int(unit % 2);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const uint32_t shift = uint32_t(address & 3);
        uint32_t mask[8], codes[8], packed[3];
        {
            uint32_t t[9];
            #pragma unroll
            for (int i = 0; i < 9; ++i) t[i] = base[i];
            #pragma unroll
            for (int i = 0; i < 8; ++i) mask[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        {
            uint32_t t[9];
            #pragma unroll
            for (int i = 0; i < 9; ++i) t[i] = base[8 + 8 * half + i];
            #pragma unroll
            for (int i = 0; i < 8; ++i) codes[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        {
            uint32_t t[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) t[i] = base[24 + i];
            #pragma unroll
            for (int i = 0; i < 3; ++i) packed[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        const float d = gemv_half(gemv_realign(base[27], base[28], shift));
        auto scale_byte = [&](int k) { return (packed[k / 4] >> (8 * (k % 4))) & 255u; };
        float result[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) result[v] = 0;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int sg = 8 * half + q;
            const int scale = int(((scale_byte(sg % 8) >> (4 * (sg / 8))) & 15)
                                | (((scale_byte(8 + sg % 4) >> (2 * (sg / 4))) & 3) << 4)) - 32;
            const int group = 4 * half + q / 2;
            float sum[V], total[V];
            #pragma unroll
            for (int v = 0; v < V; ++v) { sum[v] = 0; total[v] = 0; }
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int i = q * 16 + 4 * m, t = i % 32;
                const uint32_t l = (codes[t / 4] >> (2 * (i / 32))) & 0x03030303u;
                const uint32_t h = (mask[t / 4] >> group) & 0x01010101u;
                #pragma unroll
                for (int v = 0; v < V; ++v) {
                    const float4 a = gemv_load4(x + v * stride + unit * 128 + i);
                    sum[v] = gemv_dot4(l | (h << 2), a, sum[v]);
                    total[v] += gemv_sum4(a);
                }
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) result[v] = fmaf(float(scale), sum[v] - 4.0f * total[v], result[v]);
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d * result[v];
    }
};

// Q8_0: a 34-byte block of 32 signed bytes, biased by 128 for the unsigned byte
// conversion; 128 * sum(x) is subtracted once.
template<> struct GemvFormat<8> {
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint8_t* block = row + unit * 34;
        const float d = gemv_half(*reinterpret_cast<const uint16_t*>(block));
        const uintptr_t address = reinterpret_cast<uintptr_t>(block + 2);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const bool odd = (address & 2) != 0;
        uint32_t words[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) words[i] = base[i];
        float a[V], s[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { a[v] = 0; s[v] = 0; }
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t q = (odd ? gemv_realign(words[m], words[m + 1], 2) : words[m]) ^ 0x80808080u;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float4 xv = gemv_load4(x + v * stride + unit * 32 + 4 * m);
                a[v] = gemv_dot4(q, xv, a[v]); s[v] += gemv_sum4(xv);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d * (a[v] - 128.0f * s[v]);
    }
};
// IQ4_NL: an 18-byte block of 32 nonlinear four-bit values (low nibbles first).
template<> struct GemvFormat<20> {
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(row + unit * 18);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const bool odd = (address & 2) != 0;
        uint32_t words[5];
        #pragma unroll
        for (int i = 0; i < 5; ++i) words[i] = base[i];
        const float d = gemv_half(odd ? words[0] >> 16 : words[0]);
        float a[V], s[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { a[v] = 0; s[v] = 0; }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t codes = odd ? words[m + 1] : gemv_realign(words[m], words[m + 1], 2);
            const uint32_t low = gemv_iq4(codes & 0x0F0F0F0Fu), high = gemv_iq4((codes >> 4) & 0x0F0F0F0Fu);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 32;
                const float4 xl = gemv_load4(values + 4 * m), xh = gemv_load4(values + 16 + 4 * m);
                a[v] = gemv_dot4(low, xl, a[v]); a[v] = gemv_dot4(high, xh, a[v]);
                s[v] += gemv_sum4(xl) + gemv_sum4(xh);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d * (a[v] - 128.0f * s[v]);
    }
};
// IQ2/IQ3 lattice formats: a unit is one 32-value group of a 256-value super-block.
// Grid rows are unsigned bytes; the signs flip the activations instead.
__device__ __forceinline__ float4 iq_flip(float4 a, unsigned signs) {
    a.x = __int_as_float(__float_as_int(a.x) ^ int((signs & 1u) << 31));
    a.y = __int_as_float(__float_as_int(a.y) ^ int((signs & 2u) << 30));
    a.z = __int_as_float(__float_as_int(a.z) ^ int((signs & 4u) << 29));
    a.w = __int_as_float(__float_as_int(a.w) ^ int((signs & 8u) << 28));
    return a;
}
// Eight values: grid words g0/g1 against activations [8l, 8l+8) with eight sign bits. SIGNED puts
// each sign on the grid value rather than the activation: a product's sign is the two signs'
// exclusive or either way and its magnitude the same, so fmaf(-g, x, a) is fmaf(g, -x, a). A grid
// value is never negative, so its sign bit is just set, from the signs shifted to put bit k at bit 31
// (the bits below it masked off): two instructions a value instead of the flip's three. IQ2_XXS
// keeps the flip: its experts' gate and up, unrolled by two, came out 5% slower signed on gfx1201
// (their loads overlapped worse), where IQ2_S's took 9% less time and the IQ downs 3 to 8%.
template<int V, bool SIGNED = true>
__device__ __forceinline__ void iq_dot8(uint32_t g0, uint32_t g1, unsigned signs, const float* x, uint32_t stride,
                                        size_t offset, float (&sum)[V]) {
    if constexpr (SIGNED) {
        const uint32_t high = uint32_t(signs) << 24;
        float g[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t value = __float_as_uint(gemv_byte(k < 4 ? g0 : g1, k % 4)), sign = high << (7 - k);
            g[k] = __uint_as_float((sign & 0x80000000u) | (value & 0x7fffffffu));
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* values = x + v * stride + offset;
            const float4 a = gemv_load4(values), b = gemv_load4(values + 4);
            sum[v] = fmaf(g[0], a.x, sum[v]); sum[v] = fmaf(g[1], a.y, sum[v]); sum[v] = fmaf(g[2], a.z, sum[v]);
            sum[v] = fmaf(g[3], a.w, sum[v]); sum[v] = fmaf(g[4], b.x, sum[v]); sum[v] = fmaf(g[5], b.y, sum[v]);
            sum[v] = fmaf(g[6], b.z, sum[v]); sum[v] = fmaf(g[7], b.w, sum[v]);
        }
    } else {
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* values = x + v * stride + offset;
            sum[v] = gemv_dot4(g0, iq_flip(gemv_load4(values), signs), sum[v]);
            sum[v] = gemv_dot4(g1, iq_flip(gemv_load4(values + 4), signs >> 4), sum[v]);
        }
    }
}
__device__ __forceinline__ uint32_t iq_pair(const uint16_t* h, size_t index) { return uint32_t(h[index]) | uint32_t(h[index + 1]) << 16; }
template<> struct GemvFormat<16> {   // IQ2_XXS
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 66);
        const unsigned ib = unsigned(unit % 8);
        const uint32_t grid = iq_pair(h, 1 + 4 * ib), aux = iq_pair(h, 3 + 4 * ib);
        float sum[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) sum[v] = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid2xxs)[(grid >> (8 * l)) & 255];
            iq_dot8<V, false>(g.x, g.y, iq_signs((aux >> (7 * l)) & 127), x, stride, unit * 32 + 8 * l, sum);
        }
        const float db = gemv_half(h[0]) * (0.5f + float(aux >> 28)) * 0.25f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = db * sum[v];
    }
};
template<> struct GemvFormat<17> {   // IQ2_XS
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 74);
        const unsigned ib = unsigned(unit % 8), scales = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
        float first[V], second[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { first[v] = 0; second[v] = 0; }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned q = h[1 + 4 * ib + l];
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid2xs)[q & 511];
            iq_dot8<V>(g.x, g.y, iq_signs(q >> 9), x, stride, unit * 32 + 8 * l, l < 2 ? first : second);
        }
        const float d = gemv_half(h[0]);
        const float d0 = d * (0.5f + float(scales & 15)) * 0.25f, d1 = d * (0.5f + float(scales >> 4)) * 0.25f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d0 * first[v] + d1 * second[v];
    }
};
template<> struct GemvFormat<22> {   // IQ2_S
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 82);
        const unsigned ib = unsigned(unit % 8);
        const uint32_t low = iq_pair(h, 1 + 2 * ib), signs = iq_pair(h, 17 + 2 * ib);
        const unsigned high = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255, scales = (h[37 + ib / 2] >> (8 * (ib & 1))) & 255;
        float first[V], second[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { first[v] = 0; second[v] = 0; }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned index = ((low >> (8 * l)) & 255) | ((high << (8 - 2 * l)) & 0x300);
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid2s)[index];
            iq_dot8<V>(g.x, g.y, (signs >> (8 * l)) & 255, x, stride, unit * 32 + 8 * l, l < 2 ? first : second);
        }
        const float d = gemv_half(h[0]);
        const float d0 = d * (0.5f + float(scales & 15)) * 0.25f, d1 = d * (0.5f + float(scales >> 4)) * 0.25f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d0 * first[v] + d1 * second[v];
    }
};
template<> struct GemvFormat<18> {   // IQ3_XXS
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 98);
        const unsigned ib = unsigned(unit % 8);
        const uint32_t q[2] = {iq_pair(h, 1 + 4 * ib), iq_pair(h, 3 + 4 * ib)}, aux = iq_pair(h, 33 + 2 * ib);
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(iq_grid3xxs);
        float sum[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) sum[v] = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t pair = q[l / 2] >> (16 * (l & 1));
            iq_dot8<V>(grid[pair & 255], grid[(pair >> 8) & 255], iq_signs((aux >> (7 * l)) & 127), x, stride,
                       unit * 32 + 8 * l, sum);
        }
        const float db = gemv_half(h[0]) * (0.5f + float(aux >> 28)) * 0.5f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = db * sum[v];
    }
};
template<> struct GemvFormat<21> {   // IQ3_S
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 110);
        const unsigned ib = unsigned(unit % 8);
        const uint32_t q[2] = {iq_pair(h, 1 + 4 * ib), iq_pair(h, 3 + 4 * ib)}, signs = iq_pair(h, 37 + 2 * ib);
        const unsigned high = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
        const unsigned scale = ((h[53 + ib / 4] >> (8 * ((ib / 2) & 1))) >> (4 * (ib & 1))) & 15;
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(iq_grid3s);
        float sum[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) sum[v] = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t pair = q[l / 2] >> (16 * (l & 1));
            const unsigned first = (pair & 255) | ((high << (8 - 2 * l)) & 256);
            const unsigned second = ((pair >> 8) & 255) | ((high << (7 - 2 * l)) & 256);
            iq_dot8<V>(grid[first], grid[second], (signs >> (8 * l)) & 255, x, stride, unit * 32 + 8 * l, sum);
        }
        const float db = gemv_half(h[0]) * float(1 + 2 * scale);
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = db * sum[v];
    }
};
template<> struct GemvFormat<29> {   // IQ1_M: grid rows store value + 1, so a group adds (delta - 1) * sum(x)
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 56);
        const unsigned ib = unsigned(unit % 8);
        const uint32_t q = iq_pair(h, 2 * ib);
        const unsigned high = h[16 + ib], scales = h[24 + ib / 2] >> (6 * (ib & 1));
        const unsigned bits = (h[24] >> 12) | ((h[25] >> 8) & 0xf0) | ((h[26] >> 4) & 0xf00) | (h[27] & 0xf000);
        float first[V], second[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { first[v] = 0; second[v] = 0; }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned nibble = (high >> (4 * l)) & 15;
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid1s)[((q >> (8 * l)) & 255) | ((nibble & 7) << 8)];
            const float shift = nibble & 8 ? -1.125f : -0.875f;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 32 + 8 * l;
                const float4 lo = gemv_load4(values), hi = gemv_load4(values + 4);
                float& sum = l < 2 ? first[v] : second[v];
                sum = gemv_dot4(g.y, hi, gemv_dot4(g.x, lo, sum)) + shift * (gemv_sum4(lo) + gemv_sum4(hi));
            }
        }
        const float d = gemv_half(bits);
        const float d0 = d * float(2 * (scales & 7) + 1), d1 = d * float(2 * ((scales >> 3) & 7) + 1);
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d0 * first[v] + d1 * second[v];
    }
};
// IQ1_M dots of two rows' units against the same activations (one vector), loaded and summed once: each row's arithmetic is GemvFormat<29>::dots'.
__device__ __forceinline__ void gemv_iq1m_dots2(const uint8_t* row_a, const uint8_t* row_b, size_t unit, const float* x, const uint2* grid, float& result_a,
                                                float& result_b) {
    const uint16_t* h[2] = {reinterpret_cast<const uint16_t*>(row_a + (unit / 8) * 56), reinterpret_cast<const uint16_t*>(row_b + (unit / 8) * 56)};
    const unsigned ib = unsigned(unit % 8);
    uint32_t q[2];
    unsigned high[2], scales[2], bits[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        q[r] = iq_pair(h[r], 2 * ib);
        high[r] = h[r][16 + ib];
        scales[r] = h[r][24 + ib / 2] >> (6 * (ib & 1));
        bits[r] = (h[r][24] >> 12) | ((h[r][25] >> 8) & 0xf0) | ((h[r][26] >> 4) & 0xf00) | (h[r][27] & 0xf000);
    }
    float first[2] = {0, 0}, second[2] = {0, 0};
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const float* values = x + unit * 32 + 8 * l;
        const float4 lo = gemv_load4(values), hi = gemv_load4(values + 4);
        const float total = gemv_sum4(lo) + gemv_sum4(hi);
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const unsigned nibble = (high[r] >> (4 * l)) & 15;
            const uint2 g = grid[((q[r] >> (8 * l)) & 255) | ((nibble & 7) << 8)];
            const float shift = nibble & 8 ? -1.125f : -0.875f;
            float& sum = l < 2 ? first[r] : second[r];
            sum = gemv_dot4(g.y, hi, gemv_dot4(g.x, lo, sum)) + shift * total;
        }
    }
    float result[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float d = gemv_half(bits[r]);
        const float d0 = d * float(2 * (scales[r] & 7) + 1), d1 = d * float(2 * ((scales[r] >> 3) & 7) + 1);
        result[r] = d0 * first[r] + d1 * second[r];
    }
    result_a = result[0];
    result_b = result[1];
}
template<> struct GemvFormat<19> {   // IQ1_S: grid rows store value + 1, so a group adds (delta - 1) * sum(x)
    static constexpr size_t elements = 32, alignment = 2;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 50);
        const unsigned ib = unsigned(unit % 8);
        const uint32_t q = iq_pair(h, 1 + 2 * ib);
        const unsigned high = h[17 + ib];
        float sum[V], total[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { sum[v] = 0; total[v] = 0; }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid1s)[((q >> (8 * l)) & 255) | (((high >> (3 * l)) & 7) << 8)];
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 32 + 8 * l;
                const float4 lo = gemv_load4(values), hi = gemv_load4(values + 4);
                sum[v] = gemv_dot4(g.y, hi, gemv_dot4(g.x, lo, sum[v]));
                total[v] += gemv_sum4(lo) + gemv_sum4(hi);
            }
        }
        const float dl = gemv_half(h[0]) * float(2 * ((high >> 12) & 7) + 1), shift = high & 0x8000 ? -1.125f : -0.875f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = dl * fmaf(shift, total[v], sum[v]);
    }
};
// Q2_K: one lane owns half of an 84-byte super-block (128 values): eight groups of 16, each
// with a four-bit scale and minimum; d * sum(scale * code * x) - dmin * sum(minimum * x).
template<> struct GemvFormat<10> {
    static constexpr size_t elements = 128, alignment = 4;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint32_t* words = reinterpret_cast<const uint32_t*>(row + (unit / 2) * 84);
        const int half = int(unit % 2);
        uint32_t codes[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) codes[i] = words[4 + 8 * half + i];
        const uint32_t scales[2] = {words[2 * half], words[2 * half + 1]}, dm = words[20];
        float scaled[V], minimums[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { scaled[v] = 0; minimums[v] = 0; }
        #pragma unroll
        for (int group = 0; group < 8; ++group) {
            const int quarter = group / 2, side = group % 2;
            const unsigned scale = (scales[group / 4] >> (8 * (group % 4))) & 255;
            float sum[V], total[V];
            #pragma unroll
            for (int v = 0; v < V; ++v) { sum[v] = 0; total[v] = 0; }
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint32_t c = (codes[4 * side + m] >> (2 * quarter)) & 0x03030303u;
                #pragma unroll
                for (int v = 0; v < V; ++v) {
                    const float4 a = gemv_load4(x + v * stride + unit * 128 + 32 * quarter + 16 * side + 4 * m);
                    sum[v] = gemv_dot4(c, a, sum[v]);
                    total[v] += gemv_sum4(a);
                }
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                scaled[v] = fmaf(float(scale & 15), sum[v], scaled[v]);
                minimums[v] = fmaf(float(scale >> 4), total[v], minimums[v]);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = gemv_half(dm) * scaled[v] - gemv_half(dm >> 16) * minimums[v];
    }
};
// MXFP4: doubled E2M1 values, biased by 12 for the unsigned byte conversion (12 * sum(x) is
// subtracted once), against low nibbles 0-15 and high nibbles 16-31.
__device__ __forceinline__ uint32_t gemv_fp4(uint32_t nibbles) {
    const uint32_t select = nibbles & 0x07070707u;
    const uint32_t low = gemv_perm(0x18141210u, 0x0F0E0D0Cu, select);
    const uint32_t high = gemv_perm(0x00040608u, 0x090A0B0Cu, select);
    const uint32_t mask = ((nibbles >> 3) & 0x01010101u) * 0xFFu;
    return (low & ~mask) | (high & mask);
}
// A 17-byte block starts at any byte: its codes are read as four realigned words.
template<> struct GemvFormat<39> {
    static constexpr size_t elements = 32, alignment = 1;
    template<int V>
    __device__ static void dots(const uint8_t* row, size_t unit, const float* x, uint32_t stride, float (&r)[V]) {
        const uint8_t* block = row + unit * 17;
        const uintptr_t address = reinterpret_cast<uintptr_t>(block + 1);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const uint32_t shift = uint32_t(address & 3), e = block[0];
        uint32_t words[5];
        #pragma unroll
        for (int i = 0; i < 5; ++i) words[i] = base[i];
        float a[V], s[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { a[v] = 0; s[v] = 0; }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t codes = gemv_realign(words[m], words[m + 1], shift);
            const uint32_t low = gemv_fp4(codes & 0x0F0F0F0Fu), high = gemv_fp4((codes >> 4) & 0x0F0F0F0Fu);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float* values = x + v * stride + unit * 32;
                const float4 xl = gemv_load4(values + 4 * m), xh = gemv_load4(values + 16 + 4 * m);
                a[v] = gemv_dot4(low, xl, a[v]); a[v] = gemv_dot4(high, xh, a[v]);
                s[v] += gemv_sum4(xl) + gemv_sum4(xh);
            }
        }
        const float scale = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = scale * (a[v] - 12.0f * s[v]);
    }
};
// Formats with a fast unit decoder (others use the reference element decoder).
#define SD_FAST_FORMATS(X) X(0) X(1) X(30) X(42) X(11) X(12) X(13) X(14) X(23) X(8) X(20) X(16) X(17) X(18) X(21) X(22) \
    X(29) X(19) X(10) X(39)

constexpr size_t gemv_fixed_items = 24;
// Up to four activation vectors share one pass over the weights; vector v reads
// input + v*input_stride and writes output + v*output_stride. Each vector's sum is
// formed exactly as in the one-vector kernel, so results are bitwise identical.
constexpr int gemv_max_vectors = 4;
// A verification's rows past four (gemv()): the rows' sums do not depend on what shares their launch.
constexpr int gemv_wide_vectors = 8;
// Optional epilogue of a 2560-row projection: instead of storing row r's result y,
// add mix[2560 + s] * y to hidden[s * 2560 + r] for the four streams s -- the
// decoder residual update, with vector v at hidden + v*hidden_stride, mix + v*mix_stride.
// Or (gate, without hidden) store vector v's y times sigmoid(gate[v]), as tm_scale_rows_kernel
// scales a row.
// peer: each stored value also into the other device's inbox, at its output index (link_tail.hpp).
struct GemvResidual { float* hidden; const float* mix; uint32_t hidden_stride, mix_stride; const float* gate = nullptr; float* peer = nullptr; };
struct FixedGemv { GemvItem items[gemv_fixed_items]; uint32_t count, input_stride, output_stride; GemvResidual residual; };
__device__ __forceinline__ void gemv_store(const Matrix& m, size_t row, int v, float total, float* output, uint32_t output_stride,
                                           const GemvResidual& residual) {
    if (!residual.hidden) {
        const float value = residual.gate ? total * (1.0f / (1.0f + expf(-residual.gate[v]))) : total;
        output[m.output_offset + row + v * output_stride] = value;
        if (residual.peer) link_send(residual.peer + m.output_offset + row + v * output_stride, value);
        return;
    }
    #pragma unroll
    for (int s = 0; s < 4; ++s)
        residual.hidden[v * residual.hidden_stride + s * 2560 + row] += residual.mix[v * residual.mix_stride + 2560 + s] * total;
}

// Each group of `lanes` lanes owns one row; a vector's units, their order and the
// reduction tree do not depend on how many vectors share the pass. input_padding
// accounts for optional LDS padding between units without changing the dot order.
template<int kind, int V, size_t input_padding = 0>
__device__ void gemv_rows(const Matrix& m, size_t first_row, unsigned lanes, const float* input, uint32_t input_stride,
                          float* output, uint32_t output_stride, const GemvResidual& residual = {}) {
    const unsigned lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = first_row + lane / lanes;
    float sum[V] = {};
    const float* x = input + m.input_offset;
    const size_t units = m.cols / GemvFormat<kind>::elements;
    // Two units in flight hide load latency; with three or more vectors they spill.
    constexpr int depth = V <= 2 ? 2 : 1;
    #pragma unroll depth
    for (size_t unit = sub; unit < units; unit += lanes) {
        if (row < m.rows) {
            float r[V];
            GemvFormat<kind>::template dots<V>(m.data + row * m.row_bytes, unit, x + unit * input_padding, input_stride, r);
            #pragma unroll
            for (int v = 0; v < V; ++v) sum[v] += r[v];
        }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = sum[v];
        for (unsigned offset = lanes / 2; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (sub == 0 && row < m.rows) gemv_store(m, row, v, total, output, output_stride, residual);
    }
}
// Reference element decoder for formats or layouts without a fast unit.
template<int V>
__device__ void gemv_reference_rows(const Matrix& m, size_t row, const float* input, uint32_t input_stride,
                                    float* output, uint32_t output_stride, const GemvResidual& residual = {}) {
    const unsigned lane = threadIdx.x % 32;
    float sum[V] = {};
    if (row < m.rows) {
        const uint8_t* data = m.data + row * m.row_bytes;
        #define SD_REFERENCE(format) case format: \
            for (size_t col = lane; col < m.cols; col += 32) { \
                const float w = weight<format>(data, col); \
                for (int v = 0; v < V; ++v) sum[v] += w * input[m.input_offset + col + v * input_stride]; \
            } break;
        switch (m.kind) {
            SD_FORMATS(SD_REFERENCE)
        }
        #undef SD_REFERENCE
    }
    for (int v = 0; v < V; ++v) {
        float total = sum[v];
        for (unsigned offset = 16; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (lane == 0 && row < m.rows) gemv_store(m, row, v, total, output, output_stride, residual);
    }
}
// lanes==0 marks the reference path (one row per wave).
template<int V>
__device__ void gemv_wave(const Matrix& m, size_t local_wave, unsigned lanes, const float* input, uint32_t input_stride,
                          float* output, uint32_t output_stride, const GemvResidual& residual = {}) {
    if (!lanes) { gemv_reference_rows<V>(m, local_wave, input, input_stride, output, output_stride, residual); return; }
    const size_t first_row = local_wave * (32 / lanes);
    #define SD_ROWS(format) case format: gemv_rows<format, V>(m, first_row, lanes, input, input_stride, output, output_stride, residual); break;
    switch (m.kind) { SD_FAST_FORMATS(SD_ROWS) }
    #undef SD_ROWS
}

// ---- Q8 decode GEMVs (RDNA4: v_dot4; CUDA: dp4a) ----
// One token's input quantized per 32 values to INT8 with an F32 scale (and, given `sums`, the group's
// F32 sum, for a format's mins), as llama.cpp's q8_1 vectors, staged in LDS by every block; a unit's
// dot is exact in INT32 (v_dot4 / dp4a), then scaled. The F32 kernels spend some four VALU instructions a
// value decoding and converting. The integer dots and the quantization are the same on both backends, so
// a row's bits do not depend on which one runs it.
// The dot of four weight bytes with four INT8 activations, added to `acc`: `_u8s8`'s weight bytes are
// unsigned (a Q4_K or Q5_K code, 0 to 31), `_s8s8`'s signed. (dp4a takes both as signed: a code below 128 is the same value.)
#if defined(__HIPCC__)
__device__ __forceinline__ int gemv_dot_u8s8(uint32_t weights, uint32_t x, int acc) {
    return __builtin_amdgcn_sudot4(false, int(weights), true, int(x), acc, false);
}
__device__ __forceinline__ int gemv_dot_s8s8(uint32_t weights, uint32_t x, int acc) {
    return __builtin_amdgcn_sudot4(true, int(weights), true, int(x), acc, false);
}
#else
__device__ __forceinline__ int gemv_dot_u8s8(uint32_t weights, uint32_t x, int acc) { return __dp4a(int(weights), int(x), acc); }
__device__ __forceinline__ int gemv_dot_s8s8(uint32_t weights, uint32_t x, int acc) { return __dp4a(int(weights), int(x), acc); }
#endif
// The most dynamic shared memory a Q8 kernel stages (a launch needs no attribute below it).
#if defined(__HIPCC__)
constexpr size_t gemv_q8_lds_limit = 65536;
#else
constexpr size_t gemv_q8_lds_limit = 49152;
#endif
// One group of 32 values: its INT8 values (8 words) at `values`, its scale at `scale` and, given `sum`, its F32 sum.
__device__ __forceinline__ void gemv_q8_group(const float* from, uint32_t* values, float* scale, float* sum) {
    float4 v[8];
    float top = 0.0f, total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = gemv_load4(from + 4 * i);
        top = fmaxf(top, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
        total += gemv_sum4(v[i]);
    }
    const float inverse = top > 0.0f ? 127.0f / top : 0.0f;
    uint32_t words[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto byte = [&](float x) { return uint32_t(int(rintf(x * inverse)) & 255); };
        words[i] = byte(v[i].x) | byte(v[i].y) << 8 | byte(v[i].z) << 16 | byte(v[i].w) << 24;
    }
    uint4* to = reinterpret_cast<uint4*>(values);
    to[0] = uint4{words[0], words[1], words[2], words[3]};
    to[1] = uint4{words[4], words[5], words[6], words[7]};
    *scale = top / 127.0f;
    if (sum) *sum = total;
}
__device__ __forceinline__ void gemv_q8_stage(uint32_t* values, float* scales, float* sums, const float* from, size_t groups) {
    for (size_t group = threadIdx.x; group < groups; group += blockDim.x)
        gemv_q8_group(from + group * 32, values + group * 8, scales + group, sums ? sums + group : nullptr);
}
// The staged input of V vectors: vector v's INT8 values at xq + v * stride, its scales at
// xd + v * stride and its sums at xs + v * stride (stride in 4-byte words).
struct GemvQ8Input {
    const uint32_t* xq;
    const float *xd, *xs;
    size_t stride;
};
// A unit of a row against every vector: decoded once, each vector's dot exact in INT32 and scaled
// as one vector's alone (a vector's result does not depend on how many share the pass).
template<int kind> struct GemvQ8;
// (PAD: words after each unit's INT8 words in the staged input, see gemv_q8_stage_list; the slot kernels' staging has none)
template<> struct GemvQ8<13> {   // Q5_K: a unit is two 32-value groups of a 176-byte block, as GemvK45<13> reads them
    static constexpr size_t elements = 64;
    // A unit's bytes: its block's header, its pair's codes and the block's high bits
    struct Unit { uint4 header, q0, q1, h0, h1; };
    __device__ static Unit load(const uint8_t* row, size_t unit) {
        const uint8_t* block = row + (unit / 4) * 176;
        const uint4* packed = reinterpret_cast<const uint4*>(block + 48 + 32 * (unit % 4));
        return {*reinterpret_cast<const uint4*>(block), packed[0], packed[1], reinterpret_cast<const uint4*>(block + 16)[0],
                reinterpret_cast<const uint4*>(block + 16)[1]};
    }
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        dots<V>(load(row, unit), unit, Staged<PAD>{in, unit}, r);
    }
    // A unit's activations as dots reads them: word(v, i), i below 16, the INT8 words of the two groups (8 each), and group g's (0 or 1) scale and sum.
    // Staged: in the block's LDS, PAD words after each unit's; Held: in the lane's registers.
    template<uint32_t PAD>
    struct Staged {
        const GemvQ8Input& in;
        size_t unit;
        __device__ uint32_t word(int v, int i) const { return in.xq[v * in.stride + unit * (16 + PAD) + i]; }
        __device__ float scale(int v, int g) const { return in.xd[v * in.stride + 2 * unit + g]; }
        __device__ float sum(int v, int g) const { return in.xs[v * in.stride + 2 * unit + g]; }
    };
    template<int V>
    struct Held {
        uint32_t words[V][16];
        float scales[V][2], sums[V][2];
        template<uint32_t PAD>
        __device__ void take(const GemvQ8Input& in, size_t unit) {
            const Staged<PAD> staged{in, unit};
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                // (a unit's 16 words are 16-byte aligned in the staged input: gemv_q8_vector_words, PAD)
                const uint4* at = reinterpret_cast<const uint4*>(in.xq + v * in.stride + unit * (16 + PAD));
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint4 four = at[k];
                    words[v][4 * k] = four.x; words[v][4 * k + 1] = four.y; words[v][4 * k + 2] = four.z; words[v][4 * k + 3] = four.w;
                }
                #pragma unroll
                for (int g = 0; g < 2; ++g) { scales[v][g] = staged.scale(v, g); sums[v][g] = staged.sum(v, g); }
            }
        }
        __device__ uint32_t word(int v, int i) const { return words[v][i]; }
        __device__ float scale(int v, int g) const { return scales[v][g]; }
        __device__ float sum(int v, int g) const { return sums[v][g]; }
    };
    template<int V, class X>
    __device__ static void dots(const Unit& bytes, size_t unit, const X& x, float (&r)[V]) {
        const int pair = int(unit % 4);
        const uint4 header = bytes.header, q0 = bytes.q0, q1 = bytes.q1, h0 = bytes.h0, h1 = bytes.h1;
        uint32_t scales, mins;
        gemv_pair_scale_min(pair, header.y, header.z, header.w, scales, mins);
        const unsigned scale0 = scales & 255, scale1 = scales >> 8, min0 = mins & 255, min1 = mins >> 8;
        const uint32_t codes[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t high[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        int s0[V] = {}, s1[V] = {};
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t c0 = (codes[m] & 0x0F0F0F0Fu) | (((high[m] >> (2 * pair)) & 0x01010101u) << 4);
            const uint32_t c1 = ((codes[m] >> 4) & 0x0F0F0F0Fu) | (((high[m] >> (2 * pair + 1)) & 0x01010101u) << 4);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                s0[v] = gemv_dot_u8s8(c0, x.word(v, m), s0[v]);
                s1[v] = gemv_dot_u8s8(c1, x.word(v, 8 + m), s1[v]);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            r[v] = gemv_half(header.x) * (float(scale0) * x.scale(v, 0) * float(s0[v]) + float(scale1) * x.scale(v, 1) * float(s1[v]))
                 - gemv_half(header.x >> 16) * (float(min0) * x.sum(v, 0) + float(min1) * x.sum(v, 1));
        }
    }
};
#if defined(__HIPCC__)
// (blocks of 34 bytes start at any even address; RDNA4 loads them unaligned)
typedef uint32_t gemv_word4 __attribute__((ext_vector_type(4)));
typedef gemv_word4 gemv_word4_u __attribute__((aligned(2)));
template<> struct GemvQ8<8> {   // Q8_0: a unit is one 34-byte block of 32 values
    static constexpr size_t elements = 32;
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        static_assert(PAD == 0, "Q8_0 units are staged end to end");
        const uint8_t* block = row + unit * 34;
        const gemv_word4 q0 = *reinterpret_cast<const gemv_word4_u*>(block + 2), q1 = *reinterpret_cast<const gemv_word4_u*>(block + 18);
        const float d = gemv_half(*reinterpret_cast<const uint16_t*>(block));
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const uint32_t* x = in.xq + v * in.stride + unit * 8;
            int s = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s = __builtin_amdgcn_sudot4(true, int(q0[i]), true, int(x[i]), s, false);
                s = __builtin_amdgcn_sudot4(true, int(q1[i]), true, int(x[4 + i]), s, false);
            }
            r[v] = d * in.xd[v * in.stride + unit] * float(s);
        }
    }
};
#else
// (CUDA's loads are naturally aligned: a block's 32 bytes as nine words, realigned when the block starts two bytes in, as GemvFormat<8> reads them)
template<> struct GemvQ8<8> {   // Q8_0: a unit is one 34-byte block of 32 values
    static constexpr size_t elements = 32;
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        static_assert(PAD == 0, "Q8_0 units are staged end to end");
        const uint8_t* block = row + unit * 34;
        const float d = gemv_half(*reinterpret_cast<const uint16_t*>(block));
        const uintptr_t address = reinterpret_cast<uintptr_t>(block + 2);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const bool odd = (address & 2) != 0;
        uint32_t words[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) words[i] = base[i];
        uint32_t q[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = odd ? gemv_realign(words[i], words[i + 1], 2) : words[i];
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const uint32_t* x = in.xq + v * in.stride + unit * 8;
            int s = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) s = gemv_dot_s8s8(q[i], x[i], s);
            r[v] = d * in.xd[v * in.stride + unit] * float(s);
        }
    }
};
#endif
// The wide kinds (Device::q8_wide: Qwen3.8's projections) read the staged input end to end, a group's 8 words at 8 * group, its
// scale and sum at group (PAD 0).
template<> struct GemvQ8<12> {   // Q4_K: GemvQ8<13>'s unit without the high bits, in a 144-byte block
    static constexpr size_t elements = 64;
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        const uint8_t* block = row + (unit / 4) * 144;
        const int pair = int(unit % 4);
        const uint4 header = *reinterpret_cast<const uint4*>(block);
        const uint4* packed = reinterpret_cast<const uint4*>(block + 16 + 32 * pair);
        const uint4 q0 = packed[0], q1 = packed[1];
        uint32_t scales, mins;
        gemv_pair_scale_min(pair, header.y, header.z, header.w, scales, mins);
        const unsigned scale0 = scales & 255, scale1 = scales >> 8, min0 = mins & 255, min1 = mins >> 8;
        const uint32_t codes[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        int s0[V] = {}, s1[V] = {};
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t c0 = codes[m] & 0x0F0F0F0Fu, c1 = (codes[m] >> 4) & 0x0F0F0F0Fu;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const uint32_t* x = in.xq + v * in.stride + unit * (16 + PAD);
                s0[v] = gemv_dot_u8s8(c0, x[m], s0[v]);
                s1[v] = gemv_dot_u8s8(c1, x[8 + m], s1[v]);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* xd = in.xd + v * in.stride + 2 * unit;
            const float* xs = in.xs + v * in.stride + 2 * unit;
            r[v] = gemv_half(header.x) * (float(scale0) * xd[0] * float(s0[v]) + float(scale1) * xd[1] * float(s1[v]))
                 - gemv_half(header.x >> 16) * (float(min0) * xs[0] + float(min1) * xs[1]);
        }
    }
};
// The signed bytes code - 32 of four 6-bit codes (no borrow crosses a byte: each is code + 96 before the top bit flips back)
__device__ __forceinline__ uint32_t gemv_q6_signed(uint32_t codes) { return ((codes | 0x80808080u) - 0x20202020u) ^ 0x80808080u; }
template<> struct GemvQ8<14> {   // Q6_K: GemvFormat<14>'s unit, half of a 210-byte block (four groups, eight 16-value scales)
    static constexpr size_t elements = 128;
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(row + (unit / 2) * 210);
        const int half = int(unit % 2);
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const uint32_t shift = uint32_t(address & 3);
        uint32_t low[16], high[8];
        {
            uint32_t t[17];
            #pragma unroll
            for (int i = 0; i < 17; ++i) t[i] = base[16 * half + i];
            #pragma unroll
            for (int i = 0; i < 16; ++i) low[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        {
            uint32_t t[9];
            #pragma unroll
            for (int i = 0; i < 9; ++i) t[i] = base[32 + 8 * half + i];
            #pragma unroll
            for (int i = 0; i < 8; ++i) high[i] = gemv_realign(t[i], t[i + 1], shift);
        }
        const uint32_t w0 = base[48 + 2 * half], w1 = base[49 + 2 * half], w2 = base[50 + 2 * half];
        const uint32_t scales[2] = {gemv_realign(w0, w1, shift), gemv_realign(w1, w2, shift)};
        const float d = gemv_half(gemv_realign(base[52], base[53], shift));
        float result[V] = {};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            // value i = 32k + 4m of the half: its low nibble in low[(i % 64) / 4] (the high nibble from 64 on), its two high bits in
            // high[(i % 32) / 4] at 2 * (i / 32); a 16-value scale for m below 4 and one for the rest
            int lo[V] = {}, hi[V] = {};
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                const uint32_t l = (low[8 * (k % 2) + m] >> (4 * (k / 2))) & 0x0F0F0F0Fu;
                const uint32_t h = (high[m] >> (2 * k)) & 0x03030303u;
                const uint32_t q = gemv_q6_signed(l | (h << 4));
                #pragma unroll
                for (int v = 0; v < V; ++v) {
                    const uint32_t x = in.xq[v * in.stride + unit * (32 + PAD) + 8 * k + m];
                    if (m < 4) lo[v] = gemv_dot_s8s8(q, x, lo[v]);
                    else hi[v] = gemv_dot_s8s8(q, x, hi[v]);
                }
            }
            const int scale_lo = int(int8_t(uint8_t(scales[k / 2] >> (16 * (k % 2))))),
                      scale_hi = int(int8_t(uint8_t(scales[k / 2] >> (16 * (k % 2) + 8))));
            #pragma unroll
            for (int v = 0; v < V; ++v)
                result[v] = fmaf(in.xd[v * in.stride + unit * 4 + k], float(scale_lo * lo[v] + scale_hi * hi[v]), result[v]);
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d * result[v];
    }
};
template<> struct GemvQ8<23> {   // IQ4_XS: GemvFormat<23>'s unit, two groups of a 136-byte block, the values as signed bytes
    static constexpr size_t elements = 64;
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        const uint8_t* block = row + (unit / 4) * 136;
        const int pair = int(unit % 4);
        const uint2 header = *reinterpret_cast<const uint2*>(block);
        const uint2* packed = reinterpret_cast<const uint2*>(block + 8 + 32 * pair);
        const uint2 qa = packed[0], qb = packed[1], qc = packed[2], qd = packed[3];
        const uint32_t high = header.x >> 16, low = header.y;
        const int g0 = 2 * pair, g1 = 2 * pair + 1;
        const int scale0 = int(((low >> (4 * g0)) & 15) | (((high >> (2 * g0)) & 3) << 4)) - 32;
        const int scale1 = int(((low >> (4 * g1)) & 15) | (((high >> (2 * g1)) & 3) << 4)) - 32;
        const uint32_t first[4] = {qa.x, qa.y, qb.x, qb.y}, second[4] = {qc.x, qc.y, qd.x, qd.y};
        int s0[V] = {}, s1[V] = {};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            // (gemv_iq4's bytes are the values + 128: the top bit flipped, the signed values)
            const uint32_t w0 = gemv_iq4(first[m] & 0x0F0F0F0Fu) ^ 0x80808080u, w1 = gemv_iq4((first[m] >> 4) & 0x0F0F0F0Fu) ^ 0x80808080u;
            const uint32_t w2 = gemv_iq4(second[m] & 0x0F0F0F0Fu) ^ 0x80808080u, w3 = gemv_iq4((second[m] >> 4) & 0x0F0F0F0Fu) ^ 0x80808080u;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const uint32_t* x = in.xq + v * in.stride + unit * (16 + PAD);
                s0[v] = gemv_dot_s8s8(w0, x[m], s0[v]);
                s0[v] = gemv_dot_s8s8(w1, x[4 + m], s0[v]);
                s1[v] = gemv_dot_s8s8(w2, x[8 + m], s1[v]);
                s1[v] = gemv_dot_s8s8(w3, x[12 + m], s1[v]);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* xd = in.xd + v * in.stride + 2 * unit;
            r[v] = gemv_half(header.x) * (float(scale0) * xd[0] * float(s0[v]) + float(scale1) * xd[1] * float(s1[v]));
        }
    }
};
__host__ __device__ constexpr bool gemv_q8_kind_taken(int kind) { return kind == 13 || kind == 8 || kind == 12 || kind == 14 || kind == 23; }
__host__ __device__ constexpr bool gemv_q8_wide_kind(int kind) { return kind == 12 || kind == 13 || kind == 14 || kind == 23 || kind == 8; }
// gemv_rows on the staged input: the same rows, lanes, units and reduction for every vector.
template<int kind, int V, uint32_t PAD = 0>
__device__ __forceinline__ void gemv_q8_rows(const Matrix& m, size_t first_row, unsigned lanes, const GemvQ8Input& in, float* output,
                                             uint32_t output_stride, const GemvResidual& residual) {
    const unsigned lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = first_row + lane / lanes, units = m.cols / GemvQ8<kind>::elements;
    float sum[V] = {};
    if (row < m.rows) {
        const uint8_t* data = m.data + row * m.row_bytes;
        // (two units in flight with one or two vectors; more spill)
        constexpr int depth = V <= 2 ? 2 : 1;
        #pragma unroll depth
        for (size_t unit = sub; unit < units; unit += lanes) {
            float r[V];
            GemvQ8<kind>::template dots<V, PAD>(data, unit, in, r);
            #pragma unroll
            for (int v = 0; v < V; ++v) sum[v] += r[v];
        }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = sum[v];
        for (unsigned offset = lanes / 2; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (sub == 0 && row < m.rows) gemv_store(m, row, v, total, output, output_stride, residual);
    }
}
// Every item reads the same input (gemv_q8_input): its V vectors are staged once, a vector's words
// gemv_q8_vector_words(cols, kind) apart. A Q5_K unit's INT8 words are 16 (two groups) and the lanes of a row read different
// units at once: laid end to end they fall on a few LDS banks (a q_b row's 20 lanes, 64 bytes apart, on two: its four vectors
// took four times one's time), so GEMV_Q8_PAD words follow each unit, which puts the lanes of eight on distinct 16-byte bank
// groups. Only where the words are: every sum is formed as before. (Q8_0 rows, one group of 8 words a unit, lost more to the
// larger LDS footprint, fewer blocks to a CU, than they gained.)
// The wide kinds' units (Q4_K and IQ4_XS: two groups, Q6_K: four) took the same pad: a Q6_K out projection's lanes, 128 bytes apart, read on the
// same banks, and four vectors took twice one's time (2560 x 6144: 27 us one, 50 four; 23 on Q5_K's padded units).
constexpr uint32_t GEMV_Q8_PAD = 4;
__host__ __device__ constexpr uint32_t gemv_q8_pad(int kind) { return kind == 12 || kind == 13 || kind == 14 || kind == 23 ? GEMV_Q8_PAD : 0; }
__host__ __device__ constexpr size_t gemv_q8_unit_groups(int kind) { return kind == 14 ? 4 : kind == 8 ? 1 : 2; }
__host__ __device__ constexpr size_t gemv_q8_group_word_of(int kind, size_t group) {
    return (group / gemv_q8_unit_groups(kind)) * (8 * gemv_q8_unit_groups(kind) + gemv_q8_pad(kind)) + (group % gemv_q8_unit_groups(kind)) * 8;
}
template<int kind>
__host__ __device__ constexpr size_t gemv_q8_group_word(size_t group) { return gemv_q8_group_word_of(kind, group); }
__host__ __device__ constexpr size_t gemv_q8_value_words(size_t cols, int kind) {
    return cols / 4 + cols / (32 * gemv_q8_unit_groups(kind)) * gemv_q8_pad(kind);
}
__host__ __device__ constexpr size_t gemv_q8_vector_words(size_t cols, int kind) { return gemv_q8_value_words(cols, kind) + cols / 32 * 2; }
template<int kind, int V>
__device__ __forceinline__ GemvQ8Input gemv_q8_stage_list(const FixedGemv& list, const float* input) {
    extern __shared__ __align__(16) uint32_t gemv_lds[];
    const Matrix& first = list.items[0].matrix;
    const size_t cols = first.cols, groups = cols / 32, stride = gemv_q8_vector_words(cols, kind);
    float* xd = reinterpret_cast<float*>(gemv_lds + gemv_q8_value_words(cols, kind));
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        uint32_t* values = gemv_lds + v * stride;
        float* scales = xd + v * stride;
        const float* from = input + first.input_offset + v * list.input_stride;
        for (size_t group = threadIdx.x; group < groups; group += blockDim.x)
            gemv_q8_group(from + group * 32, values + gemv_q8_group_word<kind>(group), scales + group, scales + groups + group);
    }
    __syncthreads();
    return {gemv_lds, xd, xd + groups, stride};
}
__device__ __forceinline__ const GemvItem& gemv_q8_item(const FixedGemv& list, size_t wave) {
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= wave) selected = i;
    return list.items[selected];
}

// ---- The launches of gemv_kernels.hpp's kernels ----
// A kernel is defined, and launched, in one unit (gemv_a.cu or gemv_b.cu, the formats divided between them): the dispatch of gemv.hpp calls these,
// each of which is its kernel's launch with the grid, the block's size, the dynamic shared memory and the stream it is given. They are defined in
// gemv_kernels.hpp and instantiated in gemv_a.cu and gemv_b.cu.
template<int V> void launch_gemv_fixed(unsigned blocks, unsigned threads, hipStream_t stream, const FixedGemv& list, const float* input, float* output);
template<int kind, int V> void launch_gemv_fixed_kind(unsigned blocks, unsigned threads, hipStream_t stream, const FixedGemv& list, const float* input, float* output);
template<int kind> void launch_gemv_linked_kind(unsigned blocks, hipStream_t stream, const FixedGemv& list, const float* input, float* output, const LinkTail& tail);
void launch_gemv_list(unsigned blocks, hipStream_t stream, const GemvItem* items, size_t count, const float* input, float* output);
template<int kind, int V, bool LINKED> void launch_gemv_q8_kind(unsigned blocks, size_t lds, hipStream_t stream, const FixedGemv& list, const float* input, float* output, const LinkTail& tail);
template<int V> void launch_gemv_q8_mixed(unsigned blocks, size_t lds, hipStream_t stream, const FixedGemv& list, const float* input, float* output);
#if !defined(__HIPCC__)
template<bool compact, int V> void launch_gemv_mixed_shared(unsigned blocks, hipStream_t stream, const FixedGemv& list, const float* input, float* output);
template<int kind> void launch_gemv_decode_shared(unsigned blocks, hipStream_t stream, const Matrix& matrix, unsigned lanes, const float* input, float* output, const GemvResidual& residual);
#else
template<int V, int ROWS> void launch_gemv_q8_rows_wave(unsigned blocks, size_t lds, hipStream_t stream, const FixedGemv& list, const float* input, float* output);
template<int kind> void launch_gemv_decode_shared_rdna4(unsigned blocks, hipStream_t stream, const Matrix& matrix, unsigned lanes, const float* input, float* output, const GemvResidual& residual);
#endif


// ---- The dispatch (gemv.hpp, in kernels.cpp) that the other units call ----
unsigned gemv_lanes(const Matrix& m, const float* input, size_t concurrent = 1);
size_t gemv_waves(const Matrix& m, unsigned lanes);
void gemv(Device& d, const Matrix* matrices, size_t count, const float* input, float* output,
          size_t vectors=1, size_t input_stride=0, size_t output_stride=0, const GemvResidual& residual={},
          unsigned min_lanes=0);
