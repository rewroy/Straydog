// The Gated DeltaNet recurrence of a prompt batch over chunks of 16 tokens (RDNA4 prompt reads), where delta_split_sequence takes the
// tokens one after the other. With g_t the product of the decays f_1..f_t of a chunk and R[t][s] the product f_{s+1}..f_t,
// beta_t the write strengths, G[t][s] = k_s . k_t and the state M0 at the chunk's start,
//   (I + L) C = beta v - diag(beta g) P      with L[t][s] = beta_t R[t][s] G[t][s] (s < t) and P = M0^T k,
//   out_t     = g_t M0^T q_t + sum_{s<=t} R[t][s] (k_s . q_t) C_s
//   M_end     = g_end M0 + sum_s R[end][s] k_s (x) C_s.
// Only the state's products M0^T [k q] and the rank-16 update depend on the state, so the rest is done for every chunk of a launch
// at once (gdn_chunk_prep: the factor (I+L)^-1 applied to beta v and to diag(beta g), as U and W, and the R-weighted Gram as B), and
// the scan over the chunks (gdn_chunk_scan) is only products: C = U - W P, out = g S + B C. No division: the decay products are formed
// as products (in F64, rounded once; g_end as two floats), so a long-forgotten token underflows to zero where g_t/g_s would not. The
// arithmetic is F32 but for the products against the state, the keys, the queries and W, B and C: those run on the matrix cores in
// bf16 pieces (gdn_split), three to a value and six of the nine products of a pair summed in F32, which carries about F32's bits. The
// sums are in another order than the token-by-token scan's, but the scan's error against an F64 scan is within twice the token scan's
// on real activations, and far below it where the state forgets little (README, Numerics), because no matrix-core accumulate adds a
// small sum to a large value: its accumulate of a product sum into an accumulator drops the sum's low bits toward zero (measured:
// half an F32 spacing a wmma, with errors of up to three spacings), and six of them a chunk shrank a state that forgets little by
// 1e-7 of itself every chunk, a coherent loss that put the outputs of such heads 10x to 100x further from an F64 scan than the token
// scan's. So the products start from a zero accumulator (their sums are small against the state) and the state's decay joins them in
// F32 fused multiply-adds, the decay's own rounding (a coherent 3e-8 a chunk) kept in a second float. Tokens past a launch's
// end are inert (zero keys, queries and values, decay 1, strength 0), so each chunk is whole.
#pragma once

constexpr unsigned gdn_chunk_tokens = 16;

// A launch's scratch, in floats from `scratch`: U (token, head, 128), per chunk and head the lower-triangular W and B (16 x 16
// each) and g, e (16 each: e[s] = R[15][s], what the state update weights token s by; e[15] is 1, so its slot holds g_end's low
// part), per chunk and key head the keys and queries as bf16 pieces (gdn_piece_floats floats: three planes of 16 rows of 128, 136
// apart, for the keys, the same for the queries, and three of 128 rows of 16 tokens, 24 apart, for the keys again, transposed), and
// last the convolution's history after the launch (its last four inputs).
#if defined(__HIPCC__)
constexpr size_t gdn_piece_floats = (2 * 3 * 16 * 136 + 3 * 128 * 24) / 2;
#else
// (CUDA: the normalized keys, then the queries, as floats, a key's 16 tokens a row: what gdn_chunk_scan_f32 reads in place of the pieces)
constexpr size_t gdn_piece_floats = 2 * 128 * 16;
#endif
template<int P>
struct GdnChunkLayout {
    using H = QwenHeads<P>;
    size_t u, w, b, v, pieces, history, total;
    __host__ __device__ explicit GdnChunkLayout(size_t count) {
        const size_t chunks = (count + gdn_chunk_tokens - 1) / gdn_chunk_tokens;
        u = 0;
        w = u + chunks * gdn_chunk_tokens * H::value * 128;
        b = w + chunks * H::value * 256;
        v = b + chunks * H::value * 256;
        pieces = (v + chunks * H::value * 32 + 3) / 4 * 4;
        history = pieces + chunks * H::key * gdn_piece_floats;
        total = history + 4 * H::channels;
    }
};

// x as three bf16 pieces, each the leading eight bits of what the ones before it leave (the high halves of the three words): their sum
// is x to 2^-23, and a product of two such sums is formed from six of the nine pieces' products to about as many bits as F32 holds.
__device__ __forceinline__ void gdn_split(float x, unsigned& p1, unsigned& p2, unsigned& p3) {
    p1 = __float_as_uint(x) & 0xFFFF0000u;
    const float r1 = x - __uint_as_float(p1);
    p2 = __float_as_uint(r1) & 0xFFFF0000u;
    p3 = __float_as_uint(r1 - __uint_as_float(p2)) & 0xFFFF0000u;
}

// Grid (chunks, key heads), 640 threads: the 128 queries' channels, the 128 keys' and the 3 * 128 values' of a key head's three value
// heads (heads key + j * H::key), a thread a channel. `rows`: the qkv rows as the projection left them (not convolved); `history`:
// the convolution's last four inputs (the scan moves it on, from the last chunk's copy); beta, alpha, bias and decay as
// gdn_pack_values takes them. The convolution is gdn_conv_runs' arithmetic, the normalization gdn_normalize_qk's, and the decays and
// strengths gdn_pack_values'.
template<int P>
__global__ void __launch_bounds__(640) gdn_chunk_prep(const float* rows, const float* history, const float* coefficients, const float* beta,
                                                      const float* alpha, const float* bias, const float* decay, size_t count, float* scratch) {
    using H = QwenHeads<P>;
    constexpr unsigned CT = gdn_chunk_tokens, KS = 129, J = H::value / H::key;
    static_assert(J == 3 && CT == 16, "one block takes a key head's three value heads");
    const GdnChunkLayout<P> layout(count);
    const size_t chunk = blockIdx.x, kh = blockIdx.y, first = chunk * CT, chunks = (count + CT - 1) / CT;
    const unsigned valid = unsigned(count - first < CT ? count - first : CT), tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    // k and q rows, then (after the Grams) the L factors in their place; the Grams; each head's decay, strength and running g
    __shared__ SD_ALIGN(16) float arena[2 * CT * KS + 2 * CT * (CT + 1) + 3 * CT * 3];
    float* sk = arena;
    float* sq = sk + CT * KS;
    float* sg = sq + CT * KS;            // G[t][s], s < t
    float* sgq = sg + CT * (CT + 1);     // k_s . q_t, s <= t
    float* sf = sgq + CT * (CT + 1);     // [3][CT] decay
    float* sb = sf + 3 * CT;             // [3][CT] strength
    float* sgt = sb + 3 * CT;            // [3][CT] g
    float* sl = arena;                   // [3][CT][CT], over sk
    // The convolution of this thread's channel over the chunk's tokens: queries and keys to LDS, values kept in registers.
    const unsigned value_head = tid >= 256 ? (tid - 256) / 128 : 0, c = tid >= 256 ? (tid - 256) % 128 : tid % 128;
    const size_t channel = tid < 128 ? kh * 128 + tid : tid < 256 ? H::key * 128 + kh * 128 + (tid - 128)
                                                                   : H::key * 256 + (kh + value_head * H::key) * 128 + c;
    float y[CT];
    {
        float x[CT + 3];
        #pragma unroll
        for (unsigned i = 0; i < CT + 3; ++i) {
            const long at = long(first) + long(i) - 3;
            x[i] = at < 0 ? history[(4 + at) * H::channels + channel] : at < long(count) ? rows[at * H::channels + channel] : 0.0f;
        }
        const float c0 = coefficients[channel * 4], c1 = coefficients[channel * 4 + 1], c2 = coefficients[channel * 4 + 2], c3 = coefficients[channel * 4 + 3];
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            float value = x[t] * c0 + x[t + 1] * c1;
            value += x[t + 2] * c2; value += x[t + 3] * c3;
            y[t] = t < valid ? value * hc_sigmoid(value) : 0.0f;
        }
        // The launch's last four inputs are the history for the next: the last chunk holds them all (the scan stores them).
        if (chunk + 1 == chunks) {
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                const long at = long(count) - 4 + r;
                float value = 0.0f;
                #pragma unroll
                for (unsigned i = 0; i < CT + 3; ++i) if (long(i) == at - long(first) + 3) value = x[i];
                scratch[layout.history + r * H::channels + channel] = value;
            }
        }
    }
    if (tid < 256) {
        float* target = (tid < 128 ? sq : sk) + (tid % 128);
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) target[t * KS] = y[t];
    }
    // Each head's decays and strengths as gdn_pack_values forms them (inert past the end).
    if (tid < 3 * CT) {
        const unsigned j = tid / CT, t = tid % CT, head = unsigned(kh) + j * H::key;
        float f = 1.0f, b = 0.0f;
        if (t < valid) {
            const size_t scalar = (first + t) * H::value + head;
            const float value = alpha[scalar] + bias[head];
            const float softplus = fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
            f = expf(decay[head] * softplus);
            b = hc_sigmoid(beta[scalar]);
        }
        sf[tid] = f; sb[tid] = b;
    }
    __syncthreads();
    // Normalized as gdn_normalize_qk does (the same sums in the same order, the same divisions): a wave a row.
    for (unsigned v = wave; v < 2 * CT; v += 20) {
        const bool query = v >= CT;
        const unsigned t = query ? v - CT : v;
        float* row = (query ? sq : sk) + t * KS;
        float x[4];
        for (unsigned i = 0; i < 4; ++i) x[i] = row[lane + 32 * i];
        float sum = __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])), __fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])));
        for (unsigned stride = 16; stride; stride /= 2) sum = __fadd_rn(sum, __shfl_down(sum, stride, 32));
        sum = __shfl(sum, 0, 32);
        for (unsigned i = 0; i < 4; ++i)
            row[lane + 32 * i] = query ? x[i] / sqrtf(sum + 1e-6f) / sqrtf(128.0f) : x[i] / sqrtf(sum + 1e-6f);
    }
    __syncthreads();
#if defined(__HIPCC__)
    // The keys and queries as pieces for the scan: a wave a row of four keys a lane, then a thread a key's column of tokens.
    {
        uint16_t* pieces = reinterpret_cast<uint16_t*>(scratch + layout.pieces + (chunk * H::key + kh) * gdn_piece_floats);
        for (unsigned v = wave; v < 2 * CT; v += 20) {
            const bool query = v >= CT;
            const unsigned t = query ? v - CT : v;
            const float* row = (query ? sq : sk) + t * KS + 4 * lane;
            unsigned p[3][4];
            #pragma unroll
            for (unsigned e = 0; e < 4; ++e) gdn_split(row[e], p[0][e], p[1][e], p[2][e]);
            #pragma unroll
            for (unsigned k = 0; k < 3; ++k)
                *reinterpret_cast<uint2*>(pieces + (query ? 3 * 16 * 136 : 0) + (k * 16 + t) * 136 + 4 * lane) =
                    make_uint2((p[k][0] >> 16) | p[k][1], (p[k][2] >> 16) | p[k][3]);
        }
        if (tid < 128) {
            unsigned p[3][16];
            #pragma unroll
            for (unsigned t = 0; t < 16; ++t) gdn_split(sk[t * KS + tid], p[0][t], p[1][t], p[2][t]);
            #pragma unroll
            for (unsigned k = 0; k < 3; ++k)
                #pragma unroll
                for (unsigned h = 0; h < 2; ++h)
                    *reinterpret_cast<uint4*>(pieces + 2 * 3 * 16 * 136 + (k * 128 + tid) * 24 + 8 * h) =
                        make_uint4((p[k][8 * h] >> 16) | p[k][8 * h + 1], (p[k][8 * h + 2] >> 16) | p[k][8 * h + 3],
                                   (p[k][8 * h + 4] >> 16) | p[k][8 * h + 5], (p[k][8 * h + 6] >> 16) | p[k][8 * h + 7]);
        }
    }
#else
    // The normalized keys and queries as floats for the scan, a key's 16 tokens a row (keys, then queries).
    {
        float* kq = scratch + layout.pieces + (chunk * H::key + kh) * gdn_piece_floats;
        for (unsigned index = tid; index < 128 * CT; index += 640) {
            const unsigned key = index / CT, t = index % CT;
            kq[index] = sk[t * KS + key];
            kq[128 * CT + index] = sq[t * KS + key];
        }
    }
#endif
    // The decay products in F64, rounded once (g, R and e: the state forgets by g a chunk, and a head that forgets little keeps what F32 products
    // get wrong by a coherent 1e-7 of each chunk's decay: the F32 products put the outputs of such heads 10x further from an F64 scan than the
    // token-by-token scan's, F64 products back at its error).
    if (tid < 3) {
        double g = 1.0;
        float* vec = scratch + layout.v;
        for (unsigned t = 0; t < CT; ++t) {
            g *= double(sf[tid * CT + t]);
            sgt[tid * CT + t] = float(g);
            vec[((chunk * H::value + kh + tid * H::key) * 2) * CT + t] = float(g);
        }
    }
    // Gram products: a pair (t, s <= t) to four lanes, each over a quarter of the keys (four apart), their sums added in a fixed order.
    if (tid < 4 * (CT * (CT + 1) / 2)) {
        const unsigned pair = tid / 4, quarter = tid % 4;
        unsigned t = unsigned((sqrtf(8.0f * float(pair) + 1.0f) - 1.0f) * 0.5f);
        while (t * (t + 1) / 2 > pair) --t;
        while ((t + 1) * (t + 2) / 2 <= pair) ++t;
        const unsigned s = pair - t * (t + 1) / 2;
        float gk = 0, gq = 0;
        #pragma unroll 8
        for (unsigned i = quarter; i < 128; i += 4) {
            const float ks = sk[s * KS + i];
            gk = fmaf(ks, sk[t * KS + i], gk);
            gq = fmaf(ks, sq[t * KS + i], gq);
        }
        gk += __shfl_xor(gk, 1, 4); gk += __shfl_xor(gk, 2, 4);
        gq += __shfl_xor(gq, 1, 4); gq += __shfl_xor(gq, 2, 4);
        if (quarter == 0) { sg[t * (CT + 1) + s] = gk; sgq[t * (CT + 1) + s] = gq; }
    }
    __syncthreads();
    // L and B by column s: the running product R[t][s].
    float* bout = scratch + layout.b;
    float* vec = scratch + layout.v;
    if (tid < 3 * CT) {
        const unsigned j = tid / CT, s = tid % CT, head = unsigned(kh) + j * H::key;
        double r = 1.0;
        for (unsigned t = 0; t < CT; ++t) {
            if (t > s) r *= double(sf[j * CT + t]);
            bout[((chunk * H::value + head) * CT + t) * CT + s] = t >= s ? float(r * double(sgq[t * (CT + 1) + s])) : 0.0f;
            sl[(j * CT + t) * CT + s] = t > s ? float(double(sb[j * CT + t]) * r * double(sg[t * (CT + 1) + s])) : 0.0f;
        }
        // e_s = R[end][s] (e_end is 1 whatever the decays: its slot carries the low part of g_end, which the scan adds as a second float)
        if (s == CT - 1) {
            double g = 1.0;
            for (unsigned t = 0; t < CT; ++t) g *= double(sf[j * CT + t]);
            vec[((chunk * H::value + head) * 2 + 1) * CT + s] = float(g - double(float(g)));
        } else {
            vec[((chunk * H::value + head) * 2 + 1) * CT + s] = float(r);
        }
    }
    __syncthreads();
    // (I + L) X = rhs by forward substitution, a column to a thread: U's columns (beta v) from the value threads' convolved
    // values, W's (diag(beta g)) from the first 3 * CT threads.
    if (tid >= 256) {
        const unsigned j = value_head, head = unsigned(kh) + j * H::key;
        float x[CT];
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            float acc = sb[j * CT + t] * y[t];
            #pragma unroll
            for (unsigned s = 0; s < t; ++s) acc = fmaf(-sl[(j * CT + t) * CT + s], x[s], acc);
            x[t] = acc;
        }
        float* u = scratch + layout.u;
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) u[((first + t) * H::value + head) * 128 + c] = x[t];   // (zeros past the end)
    } else if (tid < 3 * CT) {
        const unsigned j = tid / CT, column = tid % CT, head = unsigned(kh) + j * H::key;
        float x[CT];
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            float acc = t == column ? sb[j * CT + t] * sgt[j * CT + t] : 0.0f;
            #pragma unroll
            for (unsigned s = 0; s < t; ++s) acc = fmaf(-sl[(j * CT + t) * CT + s], x[s], acc);
            x[t] = acc;
        }
        float* w = scratch + layout.w + ((chunk * H::value + head) * CT) * CT + column;
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) w[t * CT] = x[t];
    }
}

#if defined(__HIPCC__)
typedef __bf16 gdn_bf8 __attribute__((ext_vector_type(8)));
typedef float gdn_f8 __attribute__((ext_vector_type(8)));
typedef unsigned gdn_u4 __attribute__((ext_vector_type(4)));

// A fragment's eight F32 values as three bf16 fragments (gdn_split).
__device__ __forceinline__ void gdn_pieces(const gdn_f8& x, gdn_bf8 (&out)[3]) {
    unsigned p[3][8];
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) gdn_split(x[i], p[0][i], p[1][i], p[2][i]);
    #pragma unroll
    for (unsigned k = 0; k < 3; ++k)
        out[k] = __builtin_bit_cast(gdn_bf8, gdn_u4{(p[k][0] >> 16) | p[k][1], (p[k][2] >> 16) | p[k][3], (p[k][4] >> 16) | p[k][5], (p[k][6] >> 16) | p[k][7]});
}
__device__ __forceinline__ gdn_f8 gdn_wmma(const gdn_bf8& a, const gdn_bf8& b, const gdn_f8& c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    return c;
#endif
}
// a * b of two pieced fragments (16x16x16): the six larger products of nine, the three small ones summed apart from the three large.
__device__ __forceinline__ gdn_f8 gdn_product(const gdn_bf8 (&a)[3], const gdn_bf8 (&b)[3]) {
    gdn_f8 large = {}, small = {};
    large = gdn_wmma(a[0], b[0], large); large = gdn_wmma(a[0], b[1], large); large = gdn_wmma(a[1], b[0], large);
    small = gdn_wmma(a[0], b[2], small); small = gdn_wmma(a[1], b[1], small); small = gdn_wmma(a[2], b[0], small);
    return large + small;
}

// A chunk's inputs of the scan on their way from the scratch to LDS: this thread's share of the pieces (14 words of 16 bytes and a
// 15th for the first 96 threads), of W and B, of g and e, and the U values this lane's accumulators take. Macros over named variables:
// the same in a struct or an array the compiler kept in scratch memory (a store and a load a word a chunk).
#define GDN_FETCH(chunk_) { \
        const uint4* source = reinterpret_cast<const uint4*>(scratch + layout.pieces + ((chunk_) * H::key + kh) * gdn_piece_floats); \
        w0 = source[tid + 0 * THREADS]; w1 = source[tid + 1 * THREADS]; w2 = source[tid + 2 * THREADS]; w3 = source[tid + 3 * THREADS]; \
        w4 = source[tid + 4 * THREADS]; w5 = source[tid + 5 * THREADS]; w6 = source[tid + 6 * THREADS]; w7 = source[tid + 7 * THREADS]; \
        w8 = source[tid + 8 * THREADS]; w9 = source[tid + 9 * THREADS]; w10 = source[tid + 10 * THREADS]; w11 = source[tid + 11 * THREADS]; \
        w12 = source[tid + 12 * THREADS]; w13 = source[tid + 13 * THREADS]; \
        if (tid < REST) wl = source[tid + FULL * THREADS]; \
        const unsigned jj = tid / 64, e = tid % 64; \
        const size_t at = ((chunk_) * H::value + kh + jj * H::key) * 256 + 4 * e; \
        ww = *reinterpret_cast<const uint4*>(scratch + layout.w + at); \
        wb = *reinterpret_cast<const uint4*>(scratch + layout.b + at); \
        wv = scratch[layout.v + ((chunk_) * H::value + kh + (tid % 96 / 32) * H::key) * 2 * CT + tid % 32]; \
        for (unsigned i = 0; i < 8; ++i) un[i] = scratch[layout.u + (((chunk_) * CT + 8 * lh + i) * H::value + head) * 128 + c0 + lc]; }
#define GDN_STAGE() { \
        uint4* out_ = reinterpret_cast<uint4*>(sp); \
        out_[tid + 0 * THREADS] = w0; out_[tid + 1 * THREADS] = w1; out_[tid + 2 * THREADS] = w2; out_[tid + 3 * THREADS] = w3; \
        out_[tid + 4 * THREADS] = w4; out_[tid + 5 * THREADS] = w5; out_[tid + 6 * THREADS] = w6; out_[tid + 7 * THREADS] = w7; \
        out_[tid + 8 * THREADS] = w8; out_[tid + 9 * THREADS] = w9; out_[tid + 10 * THREADS] = w10; out_[tid + 11 * THREADS] = w11; \
        out_[tid + 12 * THREADS] = w12; out_[tid + 13 * THREADS] = w13; \
        if (tid < REST) out_[tid + FULL * THREADS] = wl; \
        reinterpret_cast<uint4*>(&sw[0][0][0])[tid] = ww; reinterpret_cast<uint4*>(&sb[0][0][0])[tid] = wb; \
        if (tid < 96) (&sv[0][0][0])[tid] = wv; \
        for (unsigned i = 0; i < 8; ++i) u[i] = un[i]; }

// The scan over a launch's chunks of 16 tokens on the matrix cores. A block takes a key head's three value heads and 16 of their 128
// columns (grid: key heads * 8, 6 waves: a wave a head and half of its keys). The state lives in the waves' accumulators (16 keys x 16
// columns a tile, the layout the products read it in); the chunk's pieces are staged in LDS once for the three heads, the next chunk's
// loaded into registers while this one's products run. Every 16 x 16 product here (M0^T k, M0^T q, W P, B C, the update) is a sum
// of bf16 pieces' products whose operand fragments are the layout the previous result was left in.
template<int P>
__global__ void __launch_bounds__(192) gdn_chunk_scan(float* state, size_t count, const float* scratch, float* output, float* history) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using H = QwenHeads<P>;
    constexpr unsigned CT = gdn_chunk_tokens, THREADS = 192, PIECE_WORDS = gdn_piece_floats / 4;
    const GdnChunkLayout<P> layout(count);
    const unsigned tid = threadIdx.x, wave = tid / 32, lane = tid % 32, lh = lane / 16, lc = lane % 16, j = wave / 2, hk = wave % 2;
    const size_t kh = blockIdx.x / 8, c0 = (blockIdx.x % 8) * 16, head = kh + j * H::key, chunks = (count + CT - 1) / CT;
    __shared__ SD_ALIGN(16) uint16_t sp[PIECE_WORDS * 8];
    __shared__ SD_ALIGN(16) float sw[3][CT][CT], sb[3][CT][CT], sv[3][2][CT], sex[6][2][8][32];
    const uint16_t* skp = sp;
    const uint16_t* sqp = sp + 3 * 16 * 136;
    const uint16_t* skt = sp + 2 * 3 * 16 * 136;
    // The convolution's history for the next launch (every chunk's prep has read the old).
    for (size_t i = size_t(blockIdx.x) * THREADS + tid; i < 4 * H::channels; i += size_t(gridDim.x) * THREADS) history[i] = scratch[layout.history + i];
    gdn_f8 m[4];
    #pragma unroll
    for (unsigned kt = 0; kt < 4; ++kt)
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) m[kt][i] = state[(head * 128 + 64 * hk + 16 * kt + 8 * lh + i) * 128 + c0 + lc];
    constexpr unsigned FULL = PIECE_WORDS / THREADS, REST = PIECE_WORDS - FULL * THREADS;
    static_assert(FULL == 14, "fourteen named words a thread");
    uint4 w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, wl, ww, wb;
    float wv = 0, u[8], un[8];
    GDN_FETCH(0)
    GDN_STAGE()
    __syncthreads();
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t first = chunk * CT;
        if (chunk + 1 < chunks) GDN_FETCH(chunk + 1)
        // M0^T k and M0^T q over this wave's 64 keys
        gdn_f8 ph = {}, pl = {}, qh = {}, ql = {};
        #pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            __builtin_amdgcn_sched_barrier(0);   // (else every step's fragment loads are issued ahead, and the registers the next chunk's inputs need go)
            gdn_bf8 b[3], ak[3], aq[3];
            gdn_pieces(m[ks], b);
            const unsigned kk = 64 * hk + 16 * ks + 8 * lh;
            #pragma unroll
            for (unsigned k = 0; k < 3; ++k) {
                ak[k] = *reinterpret_cast<const gdn_bf8*>(skp + (k * 16 + lc) * 136 + kk);
                aq[k] = *reinterpret_cast<const gdn_bf8*>(sqp + (k * 16 + lc) * 136 + kk);
            }
            ph = gdn_wmma(ak[0], b[0], ph); ph = gdn_wmma(ak[0], b[1], ph); ph = gdn_wmma(ak[1], b[0], ph);
            qh = gdn_wmma(aq[0], b[0], qh); qh = gdn_wmma(aq[0], b[1], qh); qh = gdn_wmma(aq[1], b[0], qh);
            pl = gdn_wmma(ak[0], b[2], pl); pl = gdn_wmma(ak[1], b[1], pl); pl = gdn_wmma(ak[2], b[0], pl);
            ql = gdn_wmma(aq[0], b[2], ql); ql = gdn_wmma(aq[1], b[1], ql); ql = gdn_wmma(aq[2], b[0], ql);
        }
        gdn_f8 pt = pl + ph, st = ql + qh;
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) { sex[wave][0][i][lane] = pt[i]; sex[wave][1][i][lane] = st[i]; }
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) { pt[i] += sex[wave ^ 1][0][i][lane]; st[i] += sex[wave ^ 1][1][i][lane]; }
        // C = U - W P and the output g S + B C, both as products: W[t][.] is this lane's row operand, P (then C) the column operand
        // this lane's accumulator already is
        gdn_f8 cr, wf;
        {
            gdn_bf8 pieces[3], wa[3];
            gdn_pieces(pt, pieces);
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) wf[i] = sw[j][lc][8 * lh + i];
            gdn_pieces(wf, wa);
            const gdn_f8 d = gdn_product(wa, pieces);
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) cr[i] = u[i] - d[i];
        }
        gdn_bf8 cp[3];
        gdn_pieces(cr, cp);
        if (hk == 0) {
            gdn_bf8 ba[3];
            gdn_f8 bf;
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) bf[i] = sb[j][lc][8 * lh + i];
            gdn_pieces(bf, ba);
            const gdn_f8 d = gdn_product(ba, cp);
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned t = 8 * lh + i;
                if (first + t < count) output[((first + t) * H::value + head) * 128 + c0 + lc] = sv[j][0][t] * st[i] + d[i];
            }
        }
        // M = g M + sum_t k_t (x) (e_t C_t)
        const float gend = sv[j][0][CT - 1], gend_low = sv[j][1][CT - 1];
        gdn_f8 ce;
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) ce[i] = (lh == 1 && i == 7 ? 1.0f : sv[j][1][8 * lh + i]) * cr[i];
        gdn_bf8 cb[3];
        gdn_pieces(ce, cb);
        #pragma unroll
        for (unsigned kt = 0; kt < 4; ++kt) {
            gdn_bf8 a[3];
            #pragma unroll
            for (unsigned k = 0; k < 3; ++k) a[k] = *reinterpret_cast<const gdn_bf8*>(skt + (k * 128 + 64 * hk + 16 * kt + lc) * 24 + 8 * lh);
            gdn_f8 acc = {};
            acc = gdn_wmma(a[2], cb[0], acc); acc = gdn_wmma(a[1], cb[1], acc); acc = gdn_wmma(a[0], cb[2], acc);
            acc = gdn_wmma(a[1], cb[0], acc); acc = gdn_wmma(a[0], cb[1], acc); acc = gdn_wmma(a[0], cb[0], acc);
            // the state's own decay is F32 fused multiply-adds (g_end as two floats: its rounding is a coherent loss each chunk), the matrix core's
            // sum is the small change alone: added into the state itself its accumulation drops the sum's low bits towards zero, a shrinking of the
            // state each chunk (the outputs of a long-memory head ended 10x to 100x further from an F64 scan than the token scan's)
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) m[kt][i] = fmaf(m[kt][i], gend, fmaf(m[kt][i], gend_low, acc[i]));
        }
        __syncthreads();
        if (chunk + 1 < chunks) GDN_STAGE()
        __syncthreads();
    }
    #pragma unroll
    for (unsigned kt = 0; kt < 4; ++kt)
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) state[(head * 128 + 64 * hk + 16 * kt + 8 * lh + i) * 128 + c0 + lc] = m[kt][i];
#endif
}
#undef GDN_FETCH
#undef GDN_STAGE

// gdn_post (the output's RMS norm and gate) with its bits and fewer barriers: its sum of squares is the same tree over the 128 channels (the
// strides 64 and 32 through LDS, the rest of the tree within the first wave by shuffles instead of five more barriers).
__global__ void __launch_bounds__(128) gdn_post_rows(const float* recurrent, const float* gate, const float* norm, float* output) {
    const size_t head = blockIdx.x, col = threadIdx.x, index = head * 128 + col;
    const float x = recurrent[index];
    __shared__ float sum[128];
    sum[col] = x * x;
    __syncthreads();
    if (col < 64) sum[col] += sum[col + 64];
    __syncthreads();
    if (col < 32) {
        float v = sum[col] + sum[col + 32];
        #pragma unroll
        for (unsigned stride = 16; stride; stride /= 2) v += __shfl_down(v, stride, 32);
        if (col == 0) sum[0] = v;
    }
    __syncthreads();
    output[index] = x / sqrtf(sum[0] / 128.0f + 1e-6f) * norm[col] * hc_sigmoid(gate[index]);
}
#endif

#if !defined(__HIPCC__)
// 16 bytes from global memory to shared memory without a register in between (Ampere and later): copies in flight while the matrix is multiplied.
__device__ __forceinline__ void gdn_copy16(void* shared, const void* global) {
    const unsigned address = unsigned(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(global) : "memory");
}
__device__ __forceinline__ void gdn_copy_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void gdn_copy_wait() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// The scan over a launch's chunks of 16 tokens on CUDA, in F32 FMAs (gdn_chunk_prep's U, W, B, g and e and the normalized keys and queries as floats
// are its inputs): a block of two warps takes 32 columns of one value head, a warp 16 of them, two lanes a column, each its column's state for 64 of
// the 128 keys in registers. Per chunk every product against the state is formed at once, 16 tokens' keys and queries over the keys with 32
// independent sums where the token scan's sums are one chain a token (and a barrier between tokens): P = M0^T k, S = M0^T q (a lane's half of the
// keys in ascending order from zero, the two halves added), C = U - W P, out = g S + B C and M = g_end M0 + sum_t k_t (x) e_t C_t, the state's decay as
// fused multiply-adds with g_end as two floats (as the RDNA4 kernel's). The chunk's inputs (the keys and queries, W, B, g, e: 18 KB) come to shared
// memory for both warps by cp.async, the next chunk's while this one's products are formed. Another order of sums than the token scan's, the same
// recurrence (docs/DESIGN.md, README: Numerics).
constexpr unsigned gdn_scan_warps = 2;
template<int P>
__global__ void __launch_bounds__(32 * gdn_scan_warps) gdn_chunk_scan_f32(float* state, size_t count, const float* scratch, float* output, float* history) {
    using H = QwenHeads<P>;
    constexpr unsigned CT = gdn_chunk_tokens, KEYS = 64, COLS = 16, KQ = 2 * 128 * CT, TILE = KQ + 2 * CT * CT + 2 * CT, THREADS = 32 * gdn_scan_warps;
    const GdnChunkLayout<P> layout(count);
    const unsigned thread = threadIdx.x, warp = thread / 32, lane = thread % 32, half = lane & 1, own = half * KEYS, own_col = lane >> 1;
    const size_t head = blockIdx.x, kh = head % H::key, chunks = (count + CT - 1) / CT;
    const size_t first_col = (size_t(blockIdx.y) * gdn_scan_warps + warp) * COLS, col = first_col + own_col;
    __shared__ SD_ALIGN(16) float tile[2][TILE];
    __shared__ float su[gdn_scan_warps][CT * COLS];
    // The convolution's history for the next launch (every chunk's prep has read the old).
    const size_t block = size_t(blockIdx.y) * gridDim.x + blockIdx.x, blocks = size_t(gridDim.x) * gridDim.y;
    for (size_t i = block * THREADS + thread; i < 4 * H::channels; i += blocks * THREADS) history[i] = scratch[layout.history + i];
    // A chunk's inputs into a tile: the keys and queries (1024 pieces of 16 bytes), W and B (64 each), g and e (8).
    auto issue = [&](size_t chunk, unsigned buffer) {
        const float* keys = scratch + layout.pieces + (chunk * H::key + kh) * gdn_piece_floats;
        const size_t at = (chunk * H::value + head) * 256;
        float* target = tile[buffer];
        #pragma unroll
        for (unsigned i = thread; i < KQ / 4; i += THREADS) gdn_copy16(target + 4 * i, keys + 4 * i);
        if (thread < CT * CT / 4) {
            gdn_copy16(target + KQ + 4 * thread, scratch + layout.w + at + 4 * thread);
            gdn_copy16(target + KQ + CT * CT + 4 * thread, scratch + layout.b + at + 4 * thread);
        }
        if (thread < CT / 2) gdn_copy16(target + KQ + 2 * CT * CT + 4 * thread, scratch + layout.v + (chunk * H::value + head) * 2 * CT + 4 * thread);
        gdn_copy_commit();
    };
    // This warp's U for a chunk, a token's row of its 16 columns a run of 64 bytes: held in registers a chunk ahead.
    float next_u[CT * COLS / 32];
    auto fetch = [&](size_t chunk) {
        #pragma unroll
        for (unsigned j = 0; j < CT * COLS / 32; ++j) {
            const unsigned i = lane + 32 * j;
            next_u[j] = scratch[layout.u + ((chunk * CT + i / COLS) * H::value + head) * 128 + first_col + i % COLS];
        }
    };
    float m[KEYS];
    #pragma unroll
    for (unsigned key = 0; key < KEYS; ++key) m[key] = state[(head * 128 + own + key) * 128 + col];
    issue(0, 0);
    fetch(0);
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t first = chunk * CT;
        const float* skq = tile[chunk & 1];
        const float* sw = skq + KQ, *sb = sw + CT * CT, *sg = sb + CT * CT, *se = sg + CT;
        #pragma unroll
        for (unsigned j = 0; j < CT * COLS / 32; ++j) su[warp][lane + 32 * j] = next_u[j];
        gdn_copy_wait();
        __syncthreads();   // (this chunk's tile is in, and the tile the chunk before read is free)
        if (chunk + 1 < chunks) { issue(chunk + 1, (chunk + 1) & 1); fetch(chunk + 1); }
        // This lane's half of M0^T k and M0^T q: 32 independent sums over its keys in order, then the two halves added.
        float p[CT], q[CT];
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) { p[t] = 0.0f; q[t] = 0.0f; }
        #pragma unroll
        for (unsigned key = 0; key < KEYS; ++key) {
            const float4* kk = reinterpret_cast<const float4*>(skq + (own + key) * CT);
            const float4* qq = reinterpret_cast<const float4*>(skq + 128 * CT + (own + key) * CT);
            const float value = m[key];
            #pragma unroll
            for (unsigned v = 0; v < CT / 4; ++v) {
                const float4 a = kk[v], b = qq[v];
                p[4 * v] = fmaf(value, a.x, p[4 * v]); p[4 * v + 1] = fmaf(value, a.y, p[4 * v + 1]);
                p[4 * v + 2] = fmaf(value, a.z, p[4 * v + 2]); p[4 * v + 3] = fmaf(value, a.w, p[4 * v + 3]);
                q[4 * v] = fmaf(value, b.x, q[4 * v]); q[4 * v + 1] = fmaf(value, b.y, q[4 * v + 1]);
                q[4 * v + 2] = fmaf(value, b.z, q[4 * v + 2]); q[4 * v + 3] = fmaf(value, b.w, q[4 * v + 3]);
            }
        }
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            p[t] += __shfl_xor_sync(0xffffffffu, p[t], 1);   // (both lanes of a column hold the same sum: a + b is b + a)
            q[t] += __shfl_xor_sync(0xffffffffu, q[t], 1);
        }
        __syncwarp();
        // C = U - W P (W lower triangular, its diagonal included), and the outputs g S + B C (a column's first lane writes them).
        float c[CT];
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            float d = 0.0f;
            #pragma unroll
            for (unsigned s = 0; s <= t; ++s) d = fmaf(sw[t * CT + s], p[s], d);
            c[t] = su[warp][t * COLS + own_col] - d;
        }
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) {
            float d = 0.0f;
            #pragma unroll
            for (unsigned s = 0; s <= t; ++s) d = fmaf(sb[t * CT + s], c[s], d);
            if (half == 0 && first + t < count) output[((first + t) * H::value + head) * 128 + col] = fmaf(sg[t], q[t], d);
        }
        // M = g_end M0 + sum_t k_t (x) (e_t C_t): the state's decay in fused multiply-adds with g_end as two floats (e_end is 1: its slot carries g_end's low part).
        #pragma unroll
        for (unsigned t = 0; t < CT; ++t) c[t] = (t == CT - 1 ? 1.0f : se[t]) * c[t];
        const float gend = sg[CT - 1], gend_low = se[CT - 1];
        #pragma unroll
        for (unsigned key = 0; key < KEYS; ++key) {
            const float4* kk = reinterpret_cast<const float4*>(skq + (own + key) * CT);
            float acc = 0.0f;
            #pragma unroll
            for (unsigned v = 0; v < CT / 4; ++v) {
                const float4 a = kk[v];
                acc = fmaf(a.x, c[4 * v], acc); acc = fmaf(a.y, c[4 * v + 1], acc); acc = fmaf(a.z, c[4 * v + 2], acc); acc = fmaf(a.w, c[4 * v + 3], acc);
            }
            m[key] = fmaf(m[key], gend, fmaf(m[key], gend_low, acc));
        }
    }
    #pragma unroll
    for (unsigned key = 0; key < KEYS; ++key) state[(head * 128 + own + key) * 128 + col] = m[key];
}
#endif
