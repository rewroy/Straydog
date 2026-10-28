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
template<int V>
__global__ void gemv_fixed(FixedGemv list, const float* input, float* output) {
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= wave) selected = i;
    const GemvItem& item = list.items[selected];
    gemv_wave<V>(item.matrix, wave - item.first_wave, item.lanes, input, list.input_stride, output, list.output_stride,
                 list.residual);
}
// Same kernel for one shared format: only that format's registers are allocated.
template<int kind, int V>
__global__ void gemv_fixed_kind(FixedGemv list, const float* input, float* output) {
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= wave) selected = i;
    const GemvItem& item = list.items[selected];
    gemv_rows<kind, V>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, input, list.input_stride,
                       output, list.output_stride, list.residual);
}
// gemv_fixed_kind of one matrix and one vector whose output is this device's part of a linked sum:
// the rows also go to the other device's inbox, and the last block adds the parts (link_tail; or,
// sending only, raises the other's flag).
template<int kind>
__global__ void gemv_linked_kind(FixedGemv list, const float* input, float* output, LinkTail tail) {
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const GemvItem& item = list.items[0];
    GemvResidual residual = list.residual;
    residual.peer = link_tail_slot(tail);
    gemv_rows<kind, 1>(item.matrix, wave * (32 / item.lanes), item.lanes, input, list.input_stride, output, list.output_stride,
                       residual);
    link_tail(tail);
}
__global__ void gemv_list(const GemvItem* items, size_t count, const float* input, float* output) {
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    size_t low = 0, high = count;
    while (low + 1 < high) {
        const size_t middle = (low + high) / 2;
        if (items[middle].first_wave <= wave) low = middle; else high = middle;
    }
    const GemvItem item = items[low];
    gemv_wave<1>(item.matrix, wave - item.first_wave, item.lanes, input, 0, output, 0);
}

// Host-side unit geometry: values per lane unit and required weight alignment.
size_t gemv_unit(int kind, size_t& alignment) {
    switch (kind) {
        case 0: alignment = 16; return 4;
        case 1: case 30: alignment = 16; return 8;
        case 42: alignment = 2; return 64;
        case 39: alignment = 1; return 32;
        case 10: alignment = 4; return 128;
        case 12: case 13: alignment = 16; return 64;
        case 23: alignment = 8; return 64;
        case 11: case 14: alignment = 2; return 128;
        case 8: case 20: case 16: case 17: case 18: case 19: case 21: case 22: case 29: alignment = 2; return 32;
    }
    alignment = 0; return 0;
}
// Lanes per row: a lane takes at most 768 values (a few long-running lanes keep
// more loads in flight than many short ones), at least two lanes share a row so
// loads stay contiguous, and lanes widen until `concurrent` such matrices fill
// 512 waves. A call site passes a fixed concurrency, so a matrix's reduction tree
// -- and with it the result -- does not depend on the batch or on what else shares
// the launch. Measured on RDNA4 decode shapes; units need not divide evenly.
unsigned gemv_lanes(const Matrix& m, const float* input, size_t concurrent = 1) {
    size_t alignment = 0;
    const size_t unit = gemv_unit(m.kind, alignment);
    const uintptr_t data = reinterpret_cast<uintptr_t>(m.data);
    if (!unit || m.cols % unit || data % alignment || m.row_bytes % alignment ||
        reinterpret_cast<uintptr_t>(input + m.input_offset) % 16)
        return 0;
    const size_t units = m.cols / unit;
    unsigned lanes = 2;
    while (lanes < 32 && units > lanes * std::max<size_t>(768 / unit, 1)) lanes *= 2;
    while (lanes < 32 && m.rows * concurrent * lanes < 512 * 32) lanes *= 2;
    return lanes;
}
size_t gemv_waves(const Matrix& m, unsigned lanes) {
    const size_t rows = lanes ? 32 / lanes : 1;
    return (m.rows + rows - 1) / rows;
}
// The lanes of a matrix on the Q8 decode kernels (gemv_q8_input): gemv_lanes', but a Q5_K row of 16 to 31 units (1024 to 1984 values;
// DeepSeek-V4.1's q_b, 20 units) takes a lane a unit, a row a wave. Over two lanes a wave's loads fan out to sixteen rows and each
// lane walks ten units in turn: 16384 x 1280 takes 29.2 us there, 27.2 on 32 lanes (the rows of 64 units, wo_a and wo_b, stream
// faster on eight: 32 cost them a microsecond). Decided by the matrix and the device alone, as gemv_lanes is, so a row's sum is
// the same whatever shares its launch.
unsigned gemv_q8_lanes(const Device& d, const Matrix& m, unsigned lanes) {
    return lanes && d.q8_decode && m.kind == 13 && m.cols >= 1024 && m.cols < 2048 && m.cols % 256 == 0 ? 32 : lanes;
}
#if !defined(__HIPCC__)
// A block owns only one matrix, so its waves can share the same activation
// vector even when adjacent matrices have different kinds or input offsets.
// Padding is in waves only; each row keeps its original lanes and reduction.
template<bool compact>
__global__ void gemv_mixed_shared(FixedGemv list, const float* input, float* output) {
    __shared__ __align__(16) float cached[2560];
    const size_t first_wave = size_t(blockIdx.x) * (blockDim.x / 32);
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= first_wave) selected = i;
    const GemvItem& item = list.items[selected];
    Matrix matrix = item.matrix;
    for (unsigned i = threadIdx.x; i < 2560; i += blockDim.x) cached[i] = input[matrix.input_offset + i];
    __syncthreads();
    matrix.input_offset = 0;
    // Eligibility excludes the reference decoder; keeping its code out of this
    // kernel also keeps its register and instruction footprint out of the mix.
    const size_t row = (first_wave - item.first_wave + threadIdx.x / 32) * (32 / item.lanes);
    #define SD_MIXED(format) case format: \
        if constexpr (!compact || format == 12 || format == 23 || format == 30) \
            gemv_rows<format, 1>(matrix, row, item.lanes, cached, 0, output, 0, list.residual); \
        break;
    switch (matrix.kind) { SD_FAST_FORMATS(SD_MIXED) }
    #undef SD_MIXED
}

// Wide single-token output projections reuse the same F32 input for every row.
// Cache it once per block without changing lane assignment or sum order.
template<int kind>
__global__ void gemv_decode_shared(Matrix matrix, unsigned lanes, const float* input, float* output,
                                    GemvResidual residual) {
    __shared__ __align__(16) float cached[6144];
    for (unsigned i = threadIdx.x; i < 6144; i += blockDim.x) cached[i] = input[matrix.input_offset + i];
    __syncthreads();
    matrix.input_offset = 0;
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    gemv_rows<kind, 1>(matrix, wave * (32 / lanes), lanes, cached, 0, output, 0, residual);
}
#endif


#if defined(__HIPCC__)
// Q3_K/Q6_K use 128-value lane units: padding each unit by four floats keeps
// the eight lanes on distinct LDS banks. Dots and reductions retain their order.
template<int kind>
__global__ void gemv_decode_shared_rdna4(Matrix matrix, unsigned lanes, const float* input, float* output,
                                        GemvResidual residual) {
    constexpr size_t elements=GemvFormat<kind>::elements, units=6144/elements;
    __shared__ __align__(16) float cached[units*(elements+4)];
    for (unsigned i=threadIdx.x;i<6144;i+=blockDim.x)
        cached[(i/elements)*(elements+4)+i%elements]=input[matrix.input_offset+i];
    __syncthreads();
    matrix.input_offset=0;
    const size_t wave=size_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    gemv_rows<kind,1,4>(matrix,wave*(32/lanes),lanes,cached,0,output,0,residual);
}
#endif

#if defined(__HIPCC__)
// ---- Q8 decode GEMVs (RDNA4) ----
// One token's input quantized per 32 values to INT8 with an F32 scale (and, given `sums`, the group's
// F32 sum, for a format's mins), as llama.cpp's q8_1 vectors, staged in LDS by every block; a unit's
// dot is exact in INT32 (v_dot4), then scaled. The F32 kernels spend some four VALU instructions a
// value decoding and converting.
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
    template<int V, uint32_t PAD = 0>
    __device__ static void dots(const uint8_t* row, size_t unit, const GemvQ8Input& in, float (&r)[V]) {
        const uint8_t* block = row + (unit / 4) * 176;
        const int pair = int(unit % 4);
        const uint4 header = *reinterpret_cast<const uint4*>(block);
        const uint4* packed = reinterpret_cast<const uint4*>(block + 48 + 32 * pair);
        const uint4 q0 = packed[0], q1 = packed[1];
        const uint4 h0 = reinterpret_cast<const uint4*>(block + 16)[0], h1 = reinterpret_cast<const uint4*>(block + 16)[1];
        uint32_t scales, mins;
        gemv_pair_scale_min(pair, header.y, header.z, header.w, scales, mins);
        const unsigned scale0 = scales & 255, scale1 = scales >> 8, min0 = mins & 255, min1 = mins >> 8;
        const uint32_t codes[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t high[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const size_t g0 = 2 * unit, g1 = g0 + 1, at = unit * (16 + PAD);
        int s0[V] = {}, s1[V] = {};
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t c0 = (codes[m] & 0x0F0F0F0Fu) | (((high[m] >> (2 * pair)) & 0x01010101u) << 4);
            const uint32_t c1 = ((codes[m] >> 4) & 0x0F0F0F0Fu) | (((high[m] >> (2 * pair + 1)) & 0x01010101u) << 4);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const uint32_t* x = in.xq + v * in.stride;
                s0[v] = __builtin_amdgcn_sudot4(false, int(c0), true, int(x[at + m]), s0[v], false);
                s1[v] = __builtin_amdgcn_sudot4(false, int(c1), true, int(x[at + 8 + m]), s1[v], false);
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float *xd = in.xd + v * in.stride, *xs = in.xs + v * in.stride;
            r[v] = gemv_half(header.x) * (float(scale0) * xd[g0] * float(s0[v]) + float(scale1) * xd[g1] * float(s1[v]))
                 - gemv_half(header.x >> 16) * (float(min0) * xs[g0] + float(min1) * xs[g1]);
        }
    }
};
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
__host__ __device__ constexpr bool gemv_q8_kind_taken(int kind) { return kind == 13 || kind == 8; }
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
constexpr uint32_t GEMV_Q8_PAD = 4;
__host__ __device__ constexpr uint32_t gemv_q8_pad(int kind) { return kind == 13 ? GEMV_Q8_PAD : 0; }
template<int kind>
__host__ __device__ constexpr size_t gemv_q8_group_word(size_t group) {
    return kind == 13 ? (group / 2) * (16 + GEMV_Q8_PAD) + (group % 2) * 8 : group * 8;
}
__host__ __device__ constexpr size_t gemv_q8_value_words(size_t cols, int kind) { return cols / 4 + (kind == 13 ? cols / 64 * GEMV_Q8_PAD : 0); }
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
// gemv_fixed_kind (LINKED, one vector: gemv_linked_kind) on the staged input.
template<int kind, int V, bool LINKED>
__global__ void __launch_bounds__(256) gemv_q8_kind(FixedGemv list, const float* input, float* output, LinkTail tail) {
    const GemvQ8Input in = gemv_q8_stage_list<kind, V>(list, input);
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const GemvItem& item = gemv_q8_item(list, wave);
    GemvResidual residual = list.residual;
    if (LINKED) residual.peer = link_tail_slot(tail);
    gemv_q8_rows<kind, V, gemv_q8_pad(kind)>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, in, output, list.output_stride,
                                             residual);
    if (LINKED) link_tail(tail);
}
// Whether a list of `vectors` takes the Q8 kernels: one kind they take, every matrix the same input
// (offset and width: staged once), 16-byte aligned vectors; its LDS bytes, else 0.
size_t gemv_q8_input(const Device& d, const FixedGemv& list, const float* input, size_t vectors) {
    const Matrix& first = list.items[0].matrix;
    const size_t bytes = vectors * gemv_q8_vector_words(first.cols, first.kind) * 4;
    if (!d.q8_decode || !list.count || !gemv_q8_kind_taken(first.kind) || first.cols % 256 || bytes > 65536 ||
        reinterpret_cast<uintptr_t>(input + first.input_offset) % 16 || (vectors > 1 && list.input_stride % 4))
        return 0;
    for (unsigned i = 0; i < list.count; ++i) {
        const Matrix& m = list.items[i].matrix;
        if (!list.items[i].lanes || m.kind != first.kind || m.cols != first.cols || m.input_offset != first.input_offset) return 0;
    }
    return bytes;
}
// The Q8 kernel for V vectors through a list's matrices of a kind it takes (with a tail: one vector
// through one matrix); false (nothing launched) otherwise. (A list's waves: its last item's first
// wave and waves.)
template<int V>
bool gemv_q8_launch(Device& d, const FixedGemv& list, const float* input, float* output, const LinkTail* tail) {
    const size_t lds = gemv_q8_input(d, list, input, V);
    if (!lds || (tail && (V != 1 || list.count != 1))) return false;
    const GemvItem& last = list.items[list.count - 1];
    const unsigned blocks = unsigned((last.first_wave + gemv_waves(last.matrix, last.lanes) + 7) / 8);
    #define SD_Q8_LAUNCH(format) \
        if (tail) gemv_q8_kind<format, V, true><<<blocks, 256, lds, d.stream>>>(list, input, output, *tail); \
        else gemv_q8_kind<format, V, false><<<blocks, 256, lds, d.stream>>>(list, input, output, LinkTail{});
    if (list.items[0].matrix.kind == 13) { SD_Q8_LAUNCH(13) } else { SD_Q8_LAUNCH(8) }
    #undef SD_Q8_LAUNCH
    check(hipGetLastError());
    return true;
}
#endif
bool gemv_shared_input_launch(Device& d, const FixedGemv& list, const float* input, float* output);   // gemv_fused.hpp
template<int V>
void gemv_launch(Device& d, const FixedGemv& list, unsigned blocks, bool uniform, const float* input, float* output) {
#if !defined(__HIPCC__)
    if constexpr (V == 1) {
        if (!uniform && list.count >= 3) {
            bool supported = true, compact = true, q3 = false;
            unsigned tall = 0;
            for (unsigned i = 0; i < list.count; ++i) {
                const Matrix& matrix = list.items[i].matrix;
                bool fast = false;
                #define SD_ALIGNED(format) case format: fast = true; break;
                switch (matrix.kind) { SD_FAST_FORMATS(SD_ALIGNED) }
                #undef SD_ALIGNED
                supported &= fast && list.items[i].lanes != 0 && matrix.cols == 2560;
                compact &= matrix.kind == 12 || matrix.kind == 23 || matrix.kind == 30;
                tall += matrix.rows >= 6144;
                q3 |= matrix.kind == 11 && matrix.rows >= 6144;
            }
            // Multiple tall projections amortize staging. Keep QSA, router and
            // Q2-only mixes on their original path: measured staging regressions.
            if (supported && tall >= 2 && (compact || q3)) {
                FixedGemv padded = list;
                size_t waves = 0;
                for (unsigned i = 0; i < list.count; ++i) {
                    padded.items[i].first_wave = uint32_t(waves);
                    waves += (gemv_waves(list.items[i].matrix, list.items[i].lanes) + 3) / 4 * 4;
                }
                if (compact) gemv_mixed_shared<true><<<unsigned(waves / 4), 128, 0, d.stream>>>(padded, input, output);
                else gemv_mixed_shared<false><<<unsigned(waves / 4), 128, 0, d.stream>>>(padded, input, output);
                return;
            }
        }
        const GemvItem& item = list.items[0];
        if (uniform && list.count == 1 && item.matrix.rows == 2560 && item.matrix.cols == 6144) {
            const unsigned shared_blocks = unsigned((gemv_waves(item.matrix, item.lanes) + 3) / 4);
            #define SD_SHARED(format) case format: gemv_decode_shared<format><<<shared_blocks, 128, 0, d.stream>>>(item.matrix, item.lanes, input, output, list.residual); return;
            switch (item.matrix.kind) {
                SD_FAST_FORMATS(SD_SHARED)
                default: break;   // other kinds take the generic launch below
            }
            #undef SD_SHARED
        }
    }
#endif
#if defined(__HIPCC__)
    if constexpr (V == 1) {
        const GemvItem& item = list.items[0];
        if (d.rdna4 && uniform && item.matrix.kind == 30 && gemv_shared_input_launch(d, list, input, output)) return;
        if (d.rdna4 && uniform && list.count == 1 && item.matrix.rows == 2560 && item.matrix.cols == 6144) {
            const unsigned shared_blocks = unsigned((gemv_waves(item.matrix, item.lanes) + 7) / 8);
            // Other packed formats lost in alternating RDNA4 measurements.
            #define SD_RDNA4_SHARED(format) case format: gemv_decode_shared_rdna4<format><<<shared_blocks, 256, 0, d.stream>>>(item.matrix, item.lanes, input, output, list.residual); return;
            switch (item.matrix.kind) {
                SD_RDNA4_SHARED(11)
                SD_RDNA4_SHARED(14)
                default: break;
            }
            #undef SD_RDNA4_SHARED
        }
    }
#endif
#if defined(__HIPCC__)
    if (uniform && gemv_q8_launch<V>(d, list, input, output, nullptr)) return;
#endif
    const unsigned threads = blocks < 2u * d.compute_units ? 64 : 256;
    blocks *= 256 / threads; // same logical waves, rows, lanes and reduction tree
    #define SD_UNIFORM(format) case format: gemv_fixed_kind<format, V><<<blocks, threads, 0, d.stream>>>(list, input, output); break;
    if (uniform) {
        switch (list.items[0].matrix.kind) {
            SD_FAST_FORMATS(SD_UNIFORM)
            default: gemv_fixed<V><<<blocks, threads, 0, d.stream>>>(list, input, output);
        }
    } else gemv_fixed<V><<<blocks, threads, 0, d.stream>>>(list, input, output);
    #undef SD_UNIFORM
}
// One vector through one matrix into `output`, this device's part of `tail`'s linked sum, in one
// launch: gemv()'s rows, lanes and dots (min_lanes as given it) on the kernel it would take,
// gemv_fixed_kind; false (nothing launched) where it would take another.
bool gemv_linked_launch(Device& d, const Matrix& matrix, const float* input, float* output, const LinkTail& tail, unsigned min_lanes) {
    unsigned lanes = gemv_lanes(matrix, input);
    size_t alignment = 0;
    if (lanes == 2 && min_lanes > 2 && matrix.cols / gemv_unit(matrix.kind, alignment) >= 32) lanes = std::min(32u, min_lanes);
    lanes = gemv_q8_lanes(d, matrix, lanes);
    // (BF16 rows may read their input from shared memory, and 2560 x 6144 has kernels of its own)
    if (!lanes || matrix.kind == 30 || (matrix.rows == 2560 && matrix.cols == 6144)) return false;
    FixedGemv list{};
    list.items[0] = {matrix, 0, lanes};
    list.count = 1;
    const unsigned blocks = unsigned((gemv_waves(matrix, lanes) + 7) / 8);
#if defined(__HIPCC__)
    if (gemv_q8_launch<1>(d, list, input, output, &tail)) return true;
#endif
    #define SD_LINKED(format) case format: gemv_linked_kind<format><<<blocks, 256, 0, d.stream>>>(list, input, output, tail); return true;
    switch (matrix.kind) {
        SD_FAST_FORMATS(SD_LINKED)
    }
    #undef SD_LINKED
    return false;
}
// Launch up to 24 independent row ranges in one dispatch, for up to four vectors. min_lanes (a
// power of two) widens a row that gemv_lanes splits over only two lanes, when it has at least 32
// units: two lanes read short runs of many rows at once.
void gemv(Device& d, const Matrix* matrices, size_t count, const float* input, float* output,
          size_t vectors=1, size_t input_stride=0, size_t output_stride=0, const GemvResidual& residual={},
          unsigned min_lanes=0) {
    if (!count || count > gemv_fixed_items || !vectors || vectors > gemv_max_vectors ||
        (vectors > 1 && (input_stride % 4 || !input_stride || (!output_stride && !residual.hidden) ||
                         input_stride > UINT32_MAX || output_stride > UINT32_MAX)) ||
        (residual.hidden && (count != 1 || matrices[0].rows != 2560 || !residual.mix)))
        throw std::runtime_error("invalid Straydog GEMV launch");
    FixedGemv list{};
    size_t waves = 0;
    for (size_t i = 0; i < count; ++i) {
        unsigned lanes = gemv_lanes(matrices[i], input);
        size_t alignment = 0;
        if (lanes == 2 && min_lanes > 2 && matrices[i].cols / gemv_unit(matrices[i].kind, alignment) >= 32)
            lanes = std::min(32u, min_lanes);
        lanes = gemv_q8_lanes(d, matrices[i], lanes);
        list.items[i] = {matrices[i], uint32_t(waves), lanes};
        waves += gemv_waves(matrices[i], lanes);
    }
    list.count = uint32_t(count); list.input_stride = uint32_t(input_stride); list.output_stride = uint32_t(output_stride);
    list.residual = residual;
    bool uniform = true;
    for (size_t i = 0; i < count; ++i)
        uniform &= list.items[i].lanes != 0 && list.items[i].matrix.kind == list.items[0].matrix.kind;
    if (!uniform && vectors > 1) {
        // The generic multi-vector kernel spills; one launch per kind runs the same dots.
        bool aligned = true;
        for (size_t i = 0; i < count; ++i) aligned &= list.items[i].lanes != 0;
        if (aligned) {
            bool done[gemv_fixed_items]{};
            for (size_t i = 0; i < count; ++i) {
                if (done[i]) continue;
                Matrix group[gemv_fixed_items];
                size_t size = 0;
                for (size_t j = i; j < count; ++j)
                    if (!done[j] && matrices[j].kind == matrices[i].kind) { group[size++] = matrices[j]; done[j] = true; }
                gemv(d, group, size, input, output, vectors, input_stride, output_stride, {}, min_lanes);
            }
            return;
        }
    }
    const unsigned blocks = unsigned((waves + 7) / 8);
    switch (vectors) {
        case 1: gemv_launch<1>(d, list, blocks, uniform, input, output); break;
        case 2: gemv_launch<2>(d, list, blocks, uniform, input, output); break;
        case 3: gemv_launch<3>(d, list, blocks, uniform, input, output); break;
        default: gemv_launch<4>(d, list, blocks, uniform, input, output); break;
    }
    check(hipGetLastError());
}
