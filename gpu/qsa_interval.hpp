// Exact block selection of a prefill chunk from FP16 matrix products with a provable error bound.
//
// The indexer's block scores (qsa_block_scores_tiled: four heads, 128 columns of F32 fused multiply-adds in order, the positive parts
// added) take 28 us a prompt token of a 242K read (both GPUs, 13 layers), and the selection of the 512 best of each row's blocks
// (qsa_select_rows) 13. Here the scores come from FP16 WMMA, and only the blocks that can be among the best are scored exactly, so that
// the selection is the one qsa_select_rows makes, block for block, with its order and its ties.
//
// The bound. For a pooled key k and a query head q (F32), let h(.) round each element to FP16 (elements below the smallest normal FP16
// are taken as zero, so that no hardware's handling of denormals matters) and r(.) = . - h(.) the residual (exact in F32). The matrix
// unit forms the products of h(k) and h(q) with F32 accumulation; with k = h(k) + r(k) and q = h(q) + r(q) the products differ by
//     k.q - h(k).h(q) = h(k).r(q) + r(k).h(q) + r(k).r(q),
// so by Cauchy-Schwarz by at most |h(k)||r(q)| + |r(k)||h(q)| + |r(k)||r(q)|. The F32 sums (the exact kernel's 128 fused multiply-adds, the
// matrix unit's accumulation, the four positive parts' additions) differ from the real sums by at most 2^-15 |h(k)||h(q)|: the exact
// kernel's own is gamma_128 = 2^-17 of the sum of magnitudes, the additions 2^-22, and the matrix unit is taken to accumulate within
// 2^-16 of it (qsa_interval_test measures 4e-8 to 8e-8). The positive part is 1-Lipschitz, so the score of the four heads is within
//     d(k) = kappa (M + 2^-15 N) + rho (N + M),   kappa = |h(k)|, rho = |r(k)|, N = sum |h(q_h)|, M = sum |r(q_h)|
// of the matrix unit's (the norms taken up and the whole inflated by 2% for the rounding of the interval's own arithmetic). A block's
// score a from the matrix unit has its exact score e in [a - d, a + d]: L = a - d, U = a + d. Nothing but the FP16 rounding of the
// operands (what the residuals measure) and F32 rounding enters the bound; no distribution is assumed. On the Qwen3.8 indexer's pooled
// keys and normalized queries |a - e| is at most 0.10 to 0.16 d (real rows of a 242K prompt, and synthetic ones), d some 2e-4 times
// the product of the norms.
//
// The selection. Let T be the 512th largest exact score. The 512th largest L is at most T and the 512th largest U at least it, so a
// block with L above the second is certainly selected (its exact score exceeds T), one with U below the first certainly is not, and the
// rest (the band, some 40 blocks a row at 242K) are scored exactly, from an F32 copy of the keys in block order, in the exact kernel's
// order of operations; T is the (512 - certain)th largest of them. Ties at T take the earliest blocks, and the selected blocks are
// listed in block order, as qsa_select_rows does.
//
// Finding the two order statistics needs every block's interval, which the scan does not keep (its output would be as large as the
// exact scores'): a sample (one block in 32, scored the same way) gives a threshold at about the 1000th best score; the scan keeps,
// as a bit and the matrix unit's score, every block whose upper bound reaches it with the chunk's largest interval taken off; and the
// selection checks that at least 512 blocks certainly reach the threshold. A row that fails that check, or has more candidates than
// qsa_iv_cap or a band over qsa_iv_band, or whose queries or keys are beyond FP16 or not numbers, is left to the exact kernels (launched
// for the chunk, and early out except for the rows flagged): the time of the exact kernels, not a different result. No kernel waits
// for the host.
#pragma once
#if defined(__HIPCC__)

// Blocks the scan stages at a time, waves of a scan workgroup (16 rows each), candidates a row may hold, blocks of the band it may rescore,
// the rank (of every 32nd block, so about this many blocks) the sample's threshold aims at, tiles a scan workgroup takes in turn, and the
// sampled blocks a row keeps (one block in 32 of a 65536-block pool). The shapes were taken by measurement at 242K: 16 waves, 4 or 16
// tiles a workgroup and a stage of 32 blocks were no faster, and a cap of 1536 left 1 row in 64 of gaussian keys to the exact kernels.
constexpr unsigned qsa_iv_unit = 64;
constexpr unsigned qsa_iv_waves = 8;
constexpr unsigned qsa_iv_rows = 16 * qsa_iv_waves;
constexpr unsigned qsa_iv_cap = 2048;
constexpr unsigned qsa_iv_band = 512;
constexpr unsigned qsa_iv_aim = 1024;
constexpr unsigned qsa_iv_tiles_per_group = 8;
constexpr unsigned qsa_iv_sample_room = 2112;

// What the interval scheme keeps in the device's attention scratch (nothing else holds it while a prompt's chunks are attended each by its
// own selection): the F32 copy of the keys, the norms, the sample, a row's values (threshold, lowest score kept, A, B, flag), the rows'
// flags, and the candidate words. The scores themselves go to the indexer's scratch, as the exact kernels' do.
constexpr size_t qsa_iv_words(size_t pool) { return (((pool + 15) / 16) + 1) & ~size_t(1); }
constexpr size_t qsa_iv_round(size_t floats) { return (floats + 63) & ~size_t(63); }
constexpr size_t qsa_iv_copy_at(size_t) { return 0; }
constexpr size_t qsa_iv_stats_at(size_t pool) { return qsa_iv_copy_at(pool) + qsa_iv_round(pool * 128); }
constexpr size_t qsa_iv_partial_at(size_t pool) { return qsa_iv_stats_at(pool) + qsa_iv_round(pool * 2); }
constexpr size_t qsa_iv_sample_at(size_t pool) { return qsa_iv_partial_at(pool) + qsa_iv_round(2 * ((pool + 63) / 64 + 8)); }
constexpr size_t qsa_iv_info_at(size_t pool) { return qsa_iv_sample_at(pool) + 256 * qsa_iv_sample_room; }
constexpr size_t qsa_iv_flags_at(size_t pool) { return qsa_iv_info_at(pool) + 256 * 8; }
constexpr size_t qsa_iv_masks_at(size_t pool) { return qsa_iv_flags_at(pool) + 320; }
constexpr size_t qsa_iv_floats(size_t pool) { return qsa_iv_masks_at(pool) + 128 * qsa_iv_words(pool); }

__device__ __forceinline__ unsigned qsa_iv_key(float value) {
    const unsigned bits = __float_as_uint(value);
    return bits >> 31 ? ~bits : bits | 0x80000000u;
}
// (a bound that is not a number says nothing: the lowest lower bound, the highest upper one)
__device__ __forceinline__ unsigned qsa_iv_lower(float value) { return value == value ? qsa_iv_key(value) : 0u; }
__device__ __forceinline__ unsigned qsa_iv_upper(float value) { return value == value ? qsa_iv_key(value) : 0xffffffffu; }
__device__ __forceinline__ float qsa_iv_value(unsigned key) { return __uint_as_float(key >> 31 ? key ^ 0x80000000u : ~key); }
// An element as the matrix unit takes it: FP16, elements under the smallest normal FP16 as zero (so that no hardware's denormal handling matters).
__device__ __forceinline__ float qsa_iv_half(float value) { return fabsf(value) < 6.103515625e-05f ? 0.0f : float(_Float16(value)); }

// Inclusive prefix sum over a block of 256 threads (eight waves of 32 lanes) in one barrier: each call writes its wave totals to its own half
// of `waves` (16 words, the half by `parity`, which a caller alternates from one call to the next, so that a half is not rewritten before
// every thread has read it: another barrier lies between two uses of one).
__device__ __forceinline__ unsigned qsa_iv_prefix(unsigned value, unsigned* waves, unsigned parity) {
    const unsigned lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (unsigned offset = 1; offset < 32; offset *= 2) {
        const unsigned other = __shfl_up(value, offset, 32);
        if (lane >= offset) value += other;
    }
    if (lane == 31) waves[(parity & 1) * 8 + wave] = value;
    __syncthreads();
    unsigned before = 0;
    for (unsigned w = 0; w < wave; ++w) before += waves[(parity & 1) * 8 + w];
    return value + before;
}

// The k1-th largest of n keys of `a` and the k2-th largest of n keys of `b` (both in LDS, n at most 65535, 1 <= k <= n), by the radix passes of
// qsa_select, the two in step: per pass an accumulation, a prefix over the buckets (one count in each half of a word) and the winning
// bucket's publication, three barriers. 256 threads; `hist` 2 (parity) * 2 * 256 words, `waves` 16, `state` 4: the keys at 0 and 2.
// `parity` is the caller's, advanced by each prefix.
__device__ void qsa_iv_kth2(const unsigned* a, const unsigned* b, unsigned n, unsigned k1, unsigned k2, unsigned (*hist)[2][256], unsigned* waves,
                            unsigned* state, unsigned& parity) {
    const unsigned t = threadIdx.x;
    if (t == 0) { state[0] = 0; state[1] = k1; state[2] = 0; state[3] = k2; }
    hist[0][0][t] = 0; hist[0][1][t] = 0; hist[1][0][t] = 0; hist[1][1][t] = 0;
    unsigned mask = 0;
    __syncthreads();
    for (unsigned pass = 0; pass < 4; ++pass) {
        const unsigned shift = 24 - 8 * pass, q = pass & 1;
        const unsigned prefix1 = state[0], wanted1 = state[1], prefix2 = state[2], wanted2 = state[3];
        for (unsigned i = t; i < n; i += 256) {
            const unsigned x = a[i], y = b[i];
            if ((x & mask) == prefix1) atomicAdd(&hist[q][0][(x >> shift) & 255], 1u);
            if ((y & mask) == prefix2) atomicAdd(&hist[q][1][(y >> shift) & 255], 1u);
        }
        __syncthreads();
        const unsigned bucket = 255 - t, total1 = hist[q][0][bucket], total2 = hist[q][1][bucket];
        hist[q][0][bucket] = 0; hist[q][1][bucket] = 0;   // (this thread's alone: ready for the pass after next)
        const unsigned at = qsa_iv_prefix(total1 | (total2 << 16), waves, parity++);   // (two counts of at most 65535 each, scanned in one)
        const unsigned at1 = at & 0xffffu, at2 = at >> 16;
        if (at1 >= wanted1 && at1 - total1 < wanted1) { state[0] = prefix1 | (bucket << shift); state[1] = wanted1 - (at1 - total1); }
        if (at2 >= wanted2 && at2 - total2 < wanted2) { state[2] = prefix2 | (bucket << shift); state[3] = wanted2 - (at2 - total2); }
        mask |= 255u << shift;
        __syncthreads();
    }
}

// qsa_iv_kth2 for one set of keys.
__device__ unsigned qsa_iv_kth(const unsigned* a, unsigned n, unsigned k, unsigned (*hist)[2][256], unsigned* waves, unsigned* state, unsigned& parity) {
    const unsigned t = threadIdx.x;
    if (t == 0) { state[0] = 0; state[1] = k; }
    hist[0][0][t] = 0; hist[1][0][t] = 0;
    unsigned mask = 0;
    __syncthreads();
    for (unsigned pass = 0; pass < 4; ++pass) {
        const unsigned shift = 24 - 8 * pass, q = pass & 1;
        const unsigned prefix = state[0], wanted = state[1];
        for (unsigned i = t; i < n; i += 256) {
            const unsigned x = a[i];
            if ((x & mask) == prefix) atomicAdd(&hist[q][0][(x >> shift) & 255], 1u);
        }
        __syncthreads();
        const unsigned bucket = 255 - t, total = hist[q][0][bucket];
        hist[q][0][bucket] = 0;
        const unsigned at = qsa_iv_prefix(total, waves, parity++);
        if (at >= wanted && at - total < wanted) { state[0] = prefix | (bucket << shift); state[1] = wanted - (at - total); }
        mask |= 255u << shift;
        __syncthreads();
    }
    return state[0];
}

// The pooled keys of blocks [from & ~63, complete) in block order as F32 (the exact rescoring reads a block in one piece), each block's
// kappa = |h(k)| and rho = |r(k)| (norms rounded up; both the largest F32 where an element is beyond FP16 or not a number) and the
// largest of each of the 64 blocks one workgroup takes. Thread 0 of the first workgroup clears the flag of rows left to the exact kernels.
__global__ void __launch_bounds__(256) qsa_iv_keys(const float* blocks, size_t pool, unsigned complete, unsigned from, float* copy, float2* stats,
                                                   float2* partial, unsigned* flags) {
    __shared__ float tile[64][129];
    __shared__ float sk[4][64], sr[4][64];
    __shared__ unsigned sb[4][64];
    __shared__ float2 wave_max[2];
    const unsigned t = threadIdx.x, tb = t & 63, quarter = t >> 6;
    const unsigned group = blockIdx.x + from / 64;
    const size_t first = size_t(group) * 64, block = first + tb;
    if (blockIdx.x == 0 && t == 0) flags[256] = 0;
    float kk = 0, rr = 0;
    unsigned bad = 0;
    float loaded[32];
    #pragma unroll
    for (unsigned j = 0; j < 32; ++j) loaded[j] = block < complete ? blocks[size_t(quarter * 32 + j) * pool + block] : 0.0f;
    #pragma unroll
    for (unsigned j = 0; j < 32; ++j) {
        const unsigned column = quarter * 32 + j;
        const float value = loaded[j];
        tile[tb][column] = value;
        const float hat = qsa_iv_half(value), residual = value - hat;
        kk = fmaf(hat, hat, kk); rr = fmaf(residual, residual, rr);
        bad |= !(fabsf(value) <= 32768.0f);
    }
    sk[quarter][tb] = kk; sr[quarter][tb] = rr; sb[quarter][tb] = bad;
    __syncthreads();
    for (unsigned i = t; i < 64 * 128; i += 256) {
        const unsigned row = i >> 7;
        if (first + row < complete) copy[(first + row) * 128 + (i & 127)] = tile[row][i & 127];
    }
    float kappa = 0, rho = 0;
    if (t < 64) {
        const float k2 = ((sk[0][t] + sk[1][t]) + sk[2][t]) + sk[3][t], r2 = ((sr[0][t] + sr[1][t]) + sr[2][t]) + sr[3][t];
        if (first + t < complete) {
            kappa = sqrtf(k2) * 1.001f; rho = sqrtf(r2) * 1.001f;
            if (sb[0][t] | sb[1][t] | sb[2][t] | sb[3][t]) { kappa = 3.0e38f; rho = 3.0e38f; }
            stats[first + t] = float2{kappa, rho};
        }
        for (unsigned offset = 16; offset; offset >>= 1) {
            kappa = fmaxf(kappa, __shfl_xor(kappa, offset)); rho = fmaxf(rho, __shfl_xor(rho, offset));
        }
        if ((t & 31) == 0) wave_max[t >> 5] = float2{kappa, rho};
    }
    __syncthreads();
    if (t == 0) partial[group] = float2{fmaxf(wave_max[0].x, wave_max[1].x), fmaxf(wave_max[0].y, wave_max[1].y)};
}

// The sum of the positive parts over a row's four heads is what the matrix unit's accumulators hold: lanes n and n ^ 4, n ^ 8 of a
// 16-lane row are the other heads of one query row (a query column is head * 4 + row).
__device__ __forceinline__ float qsa_iv_xor4(float value) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x164, 0xf, 0xf, false)); }
__device__ __forceinline__ float qsa_iv_xor8(float value) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(value), 0x168, 0xf, 0xf, false)); }

// The sampled block j (of the `sampled` a row's threshold is made from): one in 32 blocks, at an offset of its own in its 32 so that no
// period of the text lines up with the sample.
__host__ __device__ __forceinline__ unsigned qsa_iv_sample_block(unsigned j) { return j * 32 + ((j * 0x9E3779B1u) >> 27); }

typedef _Float16 qsa_iv_half4 __attribute__((ext_vector_type(4)));

// The scan: a workgroup takes 8 waves of 16 rows each, and a run of tiles of U blocks (staged as FP16 in LDS from the F32 copy, the next
// tile's loads in flight, a wave to a block at a time so that a load is a block's 512 contiguous bytes); a wave forms for each 16 blocks of
// a tile the products of its rows' four heads with FP16 WMMA (keys as rows, a head to each of four accumulators, so that the positive
// parts of a block's four heads add within a lane) and with them the row's score of each block of the tile. SAMPLE: tile s holds the
// sampled blocks 16 s ... 16 s + 15, and their scores go to `sample`; otherwise a block of a row (below the row's own complete blocks)
// with a score that reaches `thr` of the row (minus the interval) is a candidate: its bit in the row's 16-block word and its score in
// `scores`.
template <unsigned U, bool SAMPLE>
__global__ void __launch_bounds__(32 * qsa_iv_waves) qsa_iv_scan(const float* queries, const float* copy, size_t pool, size_t base, size_t steps,
                                                   const float* rowinfo, float* scores, uint16_t* masks, unsigned words, float* sample,
                                                   unsigned complete, unsigned tiles, unsigned tiles_per_group) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    __shared__ SD_ALIGN(16) _Float16 keys[2][U][136];
    const unsigned t = threadIdx.x, lane = t & 31, wave = __builtin_amdgcn_readfirstlane(t >> 5);
    const size_t row = size_t(blockIdx.y) * (16 * qsa_iv_waves) + wave * 16 + (lane & 15);
    const bool live = row < steps;
    const float thr = live && !SAMPLE ? rowinfo[row * 8 + 1] : INFINITY;
    const int limit = live ? int((base + row + 1) / 4) : 0;
    int wave_limit = live ? limit : 0x7fffffff;   // (the lowest of the wave's rows)
    for (unsigned offset = 16; offset; offset >>= 1) wave_limit = min(wave_limit, __shfl_xor(wave_limit, offset));
    moe_half8 q[4][8];
    #pragma unroll
    for (unsigned h = 0; h < 4; ++h)
        #pragma unroll
        for (unsigned s = 0; s < 8; ++s) {
            moe_half8 value{};
            if (live) {
                const float4 low = *reinterpret_cast<const float4*>(queries + row * 512 + h * 128 + 16 * s + 8 * (lane >> 4));
                const float4 high = *reinterpret_cast<const float4*>(queries + row * 512 + h * 128 + 16 * s + 8 * (lane >> 4) + 4);
                value[0] = _Float16(qsa_iv_half(low.x)); value[1] = _Float16(qsa_iv_half(low.y));
                value[2] = _Float16(qsa_iv_half(low.z)); value[3] = _Float16(qsa_iv_half(low.w));
                value[4] = _Float16(qsa_iv_half(high.x)); value[5] = _Float16(qsa_iv_half(high.y));
                value[6] = _Float16(qsa_iv_half(high.z)); value[7] = _Float16(qsa_iv_half(high.w));
            }
            q[h][s] = value;
        }
    const unsigned first_tile = blockIdx.x * tiles_per_group, end_tile = min(first_tile + tiles_per_group, tiles);
    float4 vals[U / qsa_iv_waves];
    auto block_of = [&](unsigned tile, unsigned j) -> size_t { return SAMPLE ? size_t(qsa_iv_sample_block(tile * U + wave + qsa_iv_waves * j)) : size_t(tile) * U + wave + qsa_iv_waves * j; };
    auto load = [&](unsigned tile) {
        #pragma unroll
        for (unsigned j = 0; j < U / qsa_iv_waves; ++j) {
            const size_t block = block_of(tile, j);
            vals[j] = block < complete ? *reinterpret_cast<const float4*>(copy + block * 128 + 4 * lane) : float4{0, 0, 0, 0};
        }
    };
    auto store = [&](unsigned buffer) {
        #pragma unroll
        for (unsigned j = 0; j < U / qsa_iv_waves; ++j) {
            qsa_iv_half4 packed = {_Float16(qsa_iv_half(vals[j].x)), _Float16(qsa_iv_half(vals[j].y)), _Float16(qsa_iv_half(vals[j].z)), _Float16(qsa_iv_half(vals[j].w))};
            *reinterpret_cast<qsa_iv_half4*>(&keys[buffer][wave + qsa_iv_waves * j][4 * lane]) = packed;
        }
    };
    if (first_tile < end_tile) { load(first_tile); store(0); }
    __syncthreads();
    unsigned buffer = 0;
    for (unsigned tile = first_tile; tile < end_tile; ++tile) {
        const bool more = tile + 1 < end_tile;
        if (more) load(tile + 1);
        #pragma unroll
        for (unsigned mt = 0; mt < U / 16; ++mt) {
            moe_float8 acc[4] = {};
            #pragma unroll
            for (unsigned s = 0; s < 8; ++s) {
                const moe_half8 a = *reinterpret_cast<const moe_half8*>(&keys[buffer][mt * 16 + (lane & 15)][16 * s + 8 * (lane >> 4)]);
                #pragma unroll
                for (unsigned h = 0; h < 4; ++h) acc[h] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, q[h][s], acc[h]);
            }
            const size_t tile_block = size_t(tile) * U + mt * 16;   // (the first block of the tile's 16, or the sample's first)
            const size_t lane_block = tile_block + 8 * (lane >> 4);
            float x[8];
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) x[i] = fmaxf(acc[0][i], 0.0f) + fmaxf(acc[1][i], 0.0f) + fmaxf(acc[2][i], 0.0f) + fmaxf(acc[3][i], 0.0f);
            if constexpr (SAMPLE) {
                if (live) {
                    float4* out = reinterpret_cast<float4*>(sample + row * qsa_iv_sample_room + lane_block);
                    out[0] = float4{x[0], x[1], x[2], x[3]}; out[1] = float4{x[4], x[5], x[6], x[7]};
                }
            } else {
                // (a tile that lies below every row's own complete blocks, all but the last of a row's, needs no test of the block's place)
                const bool inside = tile_block + 16 <= size_t(wave_limit);
                unsigned byte = 0;
                if (inside) {
                    #pragma unroll
                    for (unsigned i = 0; i < 8; ++i) {
                        const bool pass = !(x[i] < thr);
                        byte |= unsigned(pass) << i;
                        if (pass) scores[row * pool + lane_block + i] = x[i];
                    }
                } else {
                    const int room = limit - int(lane_block);
                    #pragma unroll
                    for (unsigned i = 0; i < 8; ++i) {
                        const bool pass = !(x[i] < thr) && int(i) < room;
                        byte |= unsigned(pass) << i;
                        if (pass) scores[row * pool + lane_block + i] = x[i];
                    }
                }
                const unsigned word = byte | (unsigned(__shfl_xor(int(byte), 16)) << 8);
                if (lane < 16 && live) masks[row * words + tile_block / 16] = uint16_t(word);
            }
        }
        if (more) store(buffer ^ 1);
        __syncthreads();
        buffer ^= 1;
    }
#endif
}

// A row's thresholds (one workgroup each; rows of at most 512 complete blocks have none): the rounding norms of its queries, the interval
// coefficients A and B, the largest interval of the chunk's blocks, and the sample's threshold: the lower bound of the sampled block
// at rank `aim * sampled / complete`, which falls short of the 512th best lower bound only when the sample is unlucky (the scan's
// count then sends the row to the exact kernels, which costs time, not exactness).
__global__ void __launch_bounds__(256) qsa_iv_threshold(const float* queries, const float* sample, unsigned sampled, const float2* stats,
                                                        const float2* partial, unsigned parts, size_t base, size_t steps, float* rowinfo) {
    const size_t token = blockIdx.x, complete = (base + token + 1) / 4;
    if (complete <= 512) return;
    const unsigned t = threadIdx.x, lane = t & 31, wave = t >> 5;
    __shared__ unsigned keys[qsa_iv_sample_room], hist[2][2][256], waves[16], state[4];
    __shared__ float reduce[3][8][4], scalars[8];
    // (every load this row needs first, so that their waits overlap: the queries, the sample's scores and keys, the chunk's key norms)
    float value[2] = {queries[token * 512 + t], queries[token * 512 + 256 + t]};
    float scored[8];
    float2 sampled_keys[8], norms[4];
    #pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const unsigned i = t + 256 * j;
        scored[j] = 0; sampled_keys[j] = float2{0, 0};
        if (i < sampled) { scored[j] = sample[token * qsa_iv_sample_room + i]; sampled_keys[j] = stats[qsa_iv_sample_block(i)]; }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) { const unsigned i = t + 256 * j; norms[j] = i < parts ? partial[i] : float2{0, 0}; }
    // norms of the rounded queries and of their residuals, per head: thread t holds elements t (head t / 128) and t + 256 (head 2 + t / 128)
    float hat[2] = {0, 0}, residual[2] = {0, 0};
    unsigned bad = 0;
    #pragma unroll
    for (unsigned e = 0; e < 2; ++e) {
        const float h = qsa_iv_half(value[e]), r = value[e] - h;
        hat[e] = h * h; residual[e] = r * r; bad |= !(fabsf(value[e]) <= 32768.0f);
    }
    for (unsigned offset = 16; offset; offset >>= 1) {
        #pragma unroll
        for (unsigned e = 0; e < 2; ++e) { hat[e] += __shfl_xor(hat[e], offset); residual[e] += __shfl_xor(residual[e], offset); }
        bad |= __shfl_xor(bad, offset);
    }
    // the chunk's largest key norms
    float kappa = 0, rho = 0;
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) { kappa = fmaxf(kappa, norms[j].x); rho = fmaxf(rho, norms[j].y); }
    for (unsigned offset = 16; offset; offset >>= 1) { kappa = fmaxf(kappa, __shfl_xor(kappa, offset)); rho = fmaxf(rho, __shfl_xor(rho, offset)); }
    if (lane == 0) {
        #pragma unroll
        for (unsigned e = 0; e < 2; ++e) { reduce[0][wave][2 * e + 0] = hat[e]; reduce[1][wave][2 * e + 0] = residual[e]; }
        reduce[2][wave][0] = float(bad);
        reduce[0][wave][1] = kappa; reduce[1][wave][1] = rho;
    }
    __syncthreads();
    if (t == 0) {
        // (waves 0-3 hold heads 0 and 2, waves 4-7 heads 1 and 3)
        float n_hat = 0, m_res = 0;
        for (unsigned h = 0; h < 4; ++h) {
            float n2 = 0, m2 = 0;
            for (unsigned w = 0; w < 4; ++w) {
                n2 += reduce[0][(h & 1) * 4 + w][h >> 1 ? 2 : 0]; m2 += reduce[1][(h & 1) * 4 + w][h >> 1 ? 2 : 0];
            }
            n_hat += sqrtf(n2) * 1.001f; m_res += sqrtf(m2) * 1.001f;
        }
        float unsafe = 0, kmax = 0, rmax = 0;
        for (unsigned w = 0; w < 8; ++w) { unsafe += reduce[2][w][0]; kmax = fmaxf(kmax, reduce[0][w][1]); rmax = fmaxf(rmax, reduce[1][w][1]); }
        const bool broken = unsafe != 0 || !(n_hat <= 3.0e38f) || !(m_res <= 3.0e38f);
        scalars[0] = broken ? 3.0e38f : 1.02f * (m_res + n_hat * (1.0f / 32768.0f));
        scalars[1] = broken ? 3.0e38f : 1.02f * (n_hat + m_res);
        scalars[2] = broken ? 1.0f : 0.0f;
        scalars[3] = kmax; scalars[4] = rmax;
    }
    __syncthreads();
    const float A = scalars[0], B = scalars[1];
    // the sample's lower bounds
    #pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const unsigned i = t + 256 * j;
        if (i < sampled) keys[i] = qsa_iv_lower(scored[j] - fmaf(sampled_keys[j].x, A, sampled_keys[j].y * B));
    }
    const unsigned rank = unsigned((size_t(qsa_iv_aim) * sampled + complete - 1) / complete);
    const unsigned wanted = rank < 8 ? 8 : rank;
    __syncthreads();
    float threshold = -INFINITY;
    unsigned parity = 0;
    if (sampled >= wanted) threshold = qsa_iv_value(qsa_iv_kth(keys, sampled, wanted, hist, waves, state, parity));
    if (t == 0) {
        const float width = fmaf(scalars[3], A, scalars[4] * B);
        float lowest = threshold - width;
        lowest -= (fabsf(threshold) + width) * 2.4e-7f;   // (more than the rounding of any sum of a score, its interval and the threshold)
        if (!(lowest == lowest) || !(threshold > -INFINITY)) lowest = -INFINITY;
        float* info = rowinfo + token * 8;
        info[0] = threshold; info[1] = lowest; info[2] = A; info[3] = B; info[4] = scalars[2];
    }
}

// (QSA_IV_STATS: a line for a row now and then, and one for every row left to the exact kernels, with the reason)
#ifdef QSA_IV_STATS
#define QSA_IV_LEFT(reason) do { if (t == 0) { flags[token] = 1; flags[256] = 1; printf("QIVFB pos %lu complete %lu reason %d found %u\n", (unsigned long)position, (unsigned long)complete, reason, found); } return; } while (0)
#else
#define QSA_IV_LEFT(reason) do { if (t == 0) { flags[token] = 1; flags[256] = 1; } return; } while (0)
#endif

// The selection of a row from its candidates (see the top of this file). `flags` holds a row's reason to be left to the exact
// kernels at its index, and at 256 whether any is.
template <int P = 1>
__global__ void __launch_bounds__(256) qsa_iv_select(const float* queries, const float* scores, const float2* stats, const float* copy, size_t pool,
                                                     size_t base, size_t steps, size_t selected, float* packed, const float* rowinfo,
                                                     const uint16_t* masks, unsigned words, unsigned* flags) {
    const size_t token = blockIdx.x, position = base + token, complete = (position + 1) / 4;
    const unsigned t = threadIdx.x, lane = t & 31;
    const size_t index_start = steps * QwenHeads<P>::packed;
    float* indices = packed + index_start + token * selected;
    if (complete <= 512) {
        for (size_t i = t; i <= position; i += 256) indices[i] = float(i);
        if (t == 0) { packed[index_start + steps * selected + token] = float(position + 1); flags[token] = 0; }
        return;
    }
    // (27 KB with the block numbers and band's places in 16 bits and the query row over the histograms, which are done with by then:
    // four of these to a workgroup processor)
    __shared__ unsigned short cb[qsa_iv_cap], bl[qsa_iv_band];
    __shared__ unsigned lk[qsa_iv_cap], uk[qsa_iv_cap], ev[qsa_iv_band];
    __shared__ SD_ALIGN(16) unsigned shared_words[1056];
    __shared__ unsigned waves[16], state[4], counts[4];
    unsigned (*hist)[2][256] = reinterpret_cast<unsigned (*)[2][256]>(shared_words);
    float* qrow = reinterpret_cast<float*>(shared_words);
    unsigned parity = 0;
    const float* info = rowinfo + token * 8;
    const float threshold = info[0], A = info[2], B = info[3];
    const unsigned broken = info[4] != 0.0f;
    // the candidates in block order: this thread's run of words
    const unsigned total_words = unsigned((complete + 15) / 16), per = (total_words + 255) / 256;
    const unsigned w0 = min(total_words, t * per);
    const uint16_t* row_masks = masks + token * words;
    unsigned word[16], mine = 0;
    #pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        word[i] = i < per && w0 + i < total_words ? unsigned(row_masks[w0 + i]) : 0u;
        mine += __popc(word[i]);
    }
    const unsigned inclusive = qsa_iv_prefix(mine, waves, parity++), before = inclusive - mine;
    if (t == 255) counts[0] = inclusive;
    if (t == 0) { counts[1] = 0; counts[2] = 0; }
    __syncthreads();
    const unsigned found = counts[0];
    if (broken) QSA_IV_LEFT(1);
    if (found > qsa_iv_cap) QSA_IV_LEFT(2);
    if (found < 512) QSA_IV_LEFT(3);
    {
        unsigned at = before;
        #pragma unroll
        for (unsigned i = 0; i < 16; ++i) {
            unsigned bits = word[i];
            while (bits) { cb[at++] = (w0 + i) * 16 + (__ffs(bits) - 1); bits &= bits - 1; }
        }
    }
    __syncthreads();
    // each candidate's score, its interval and keys (loads of all of a thread's candidates in flight together)
    {
        float av[qsa_iv_cap / 256];
        float2 kv[qsa_iv_cap / 256];
        #pragma unroll
        for (unsigned j = 0; j < qsa_iv_cap / 256; ++j) {
            const unsigned i = t + 256 * j;
            av[j] = 0; kv[j] = float2{0, 0};
            if (i < found) { const unsigned block = cb[i]; av[j] = scores[token * pool + block]; kv[j] = stats[block]; }
        }
        #pragma unroll
        for (unsigned j = 0; j < qsa_iv_cap / 256; ++j) {
            const unsigned i = t + 256 * j;
            if (i < found) {
                const float width = fmaf(kv[j].x, A, kv[j].y * B);
                lk[i] = qsa_iv_lower(av[j] - width); uk[i] = qsa_iv_upper(av[j] + width);
            }
        }
    }
    __syncthreads();
    // blocks that certainly reach the sample's threshold: at least 512, or its rank was too high
    const unsigned threshold_key = qsa_iv_lower(threshold);
    unsigned reached = 0;
    for (unsigned i = t; i < found; i += 256) reached += lk[i] >= threshold_key;
    atomicAdd(&counts[1], reached);
    __syncthreads();
    const unsigned reached_total = counts[1];
    if (reached_total < 512) QSA_IV_LEFT(4);
    qsa_iv_kth2(lk, uk, found, 512, 512, hist, waves, state, parity);
    const unsigned lowest = state[0], highest = state[2];
    __syncthreads();
    // classification in runs of candidates (block order): certain, band
    const unsigned run = (found + 255) / 256, c0 = min(found, t * run), c1 = min(found, c0 + run);
    unsigned certain = 0, in_band = 0;
    for (unsigned c = c0; c < c1; ++c) {
        const bool above = lk[c] > highest;
        certain += above; in_band += !above && uk[c] >= lowest;
    }
    const unsigned both_before = qsa_iv_prefix(in_band | (certain << 16), waves, parity++) - (in_band | (certain << 16));
    const unsigned band_before = both_before & 0xffffu;
    if (t == 255) counts[2] = (both_before & 0xffffu) + in_band, counts[3] = (both_before >> 16) + certain;
    __syncthreads();
    const unsigned band = counts[2], sure = counts[3];
    // (a bound that failed: more certain blocks than there are places, no band to choose from, or a band too wide to rescore)
    if (sure > 511) QSA_IV_LEFT(5);
    if (band + sure < 512) QSA_IV_LEFT(6);
    if (band > qsa_iv_band) QSA_IV_LEFT(7);
    {
        unsigned at = band_before;
        for (unsigned c = c0; c < c1; ++c)
            if (!(lk[c] > highest) && uk[c] >= lowest) bl[at++] = c;
    }
    for (unsigned i = t; i < 512; i += 256) qrow[(i >> 7) * 132 + (i & 127)] = queries[token * 512 + i];
    __syncthreads();
    // the band's exact scores: four lanes a block (a head each), the exact kernel's fused multiply-adds in column order, the positive parts
    // added in head order
    {
        const unsigned group = t >> 2, head = t & 3, first = lane & ~3u;
        for (unsigned e = group; e < band; e += 64) {
            const float* key = copy + size_t(cb[bl[e]]) * 128;
            float acc = 0;
            #pragma unroll 8
            for (unsigned c = 0; c < 32; ++c) {
                const float4 k = *reinterpret_cast<const float4*>(key + 4 * c);
                const float4 v = *reinterpret_cast<const float4*>(&qrow[head * 132 + 4 * c]);
                acc = fmaf(k.x, v.x, acc); acc = fmaf(k.y, v.y, acc); acc = fmaf(k.z, v.z, acc); acc = fmaf(k.w, v.w, acc);
            }
            const float positive = fmaxf(acc, 0.0f);
            const float h0 = __shfl(positive, first), h1 = __shfl(positive, first + 1), h2 = __shfl(positive, first + 2), h3 = __shfl(positive, first + 3);
            if (head == 0) ev[e] = qsa_iv_key(h0 + h1 + h2 + h3);
        }
    }
    __syncthreads();
    // the (512 - certain)th best exact score of the band, and how many of those equal to it the row takes (the earliest)
    const unsigned need = 512 - sure;
    // (the band is small: each entry counts the entries above it and equal to it, and the one that stands at rank `need` names the score)
    for (unsigned e = t; e < band; e += 256) {
        const unsigned mine_key = ev[e];
        unsigned above = 0, equal = 0;
        for (unsigned j = 0; j < band; ++j) { above += ev[j] > mine_key; equal += ev[j] == mine_key; }
        if (above < need && need <= above + equal) { state[0] = mine_key; state[1] = above; }
    }
    __syncthreads();
    const unsigned limit = state[0], take = need - state[1];
    // block order: this thread's run again, the band's entries in step
    unsigned equal_here = 0, taken = 0;
    {
        unsigned at = band_before;
        for (unsigned c = c0; c < c1; ++c)
            if (!(lk[c] > highest) && uk[c] >= lowest) equal_here += ev[at++] == limit;
    }
    const unsigned equal_before = qsa_iv_prefix(equal_here, waves, parity++) - equal_here;
    {
        unsigned at = band_before, seen = equal_before;
        for (unsigned c = c0; c < c1; ++c) {
            if (lk[c] > highest) { ++taken; continue; }
            if (uk[c] < lowest) continue;
            const unsigned score = ev[at++];
            if (score > limit) ++taken;
            else if (score == limit && seen++ < take) ++taken;
        }
    }
    const unsigned taken_before = qsa_iv_prefix(taken, waves, parity++) - taken;
    {
        unsigned at = band_before, seen = equal_before, out = taken_before;
        for (unsigned c = c0; c < c1; ++c) {
            bool take_it = false;
            if (lk[c] > highest) take_it = true;
            else if (uk[c] >= lowest) {
                const unsigned score = ev[at++];
                take_it = score > limit || (score == limit && seen++ < take);
            }
            if (take_it) {
                #pragma unroll
                for (unsigned tap = 0; tap < 4; ++tap) indices[size_t(out) * 4 + tap] = float(size_t(cb[c]) * 4 + tap);
                ++out;
            }
        }
    }
    const size_t head = 512 * 4;
    for (size_t i = t; complete * 4 + i <= position; i += 256) indices[head + i] = float(complete * 4 + i);
    if (t == 0) { packed[index_start + steps * selected + token] = float(head + position + 1 - complete * 4); flags[token] = 0; }
#ifdef QSA_IV_STATS
    if (t == 0 && (position & 1023) == 0) printf("QIV pos %lu complete %lu found %u reached %u sure %u band %u\n", (unsigned long)position, (unsigned long)complete, found, reached_total, sure, band);
#endif
}

// qsa_index_chunk's selection by intervals: the scores of the candidates go to `scores`, the rest to the attention scratch. The scratch
// still holds the F32 copy and norms of the first `valid` blocks if the call before it (this layer's, of the same keys) left them there; the
// call leaves `valid` at its own complete blocks.
template <int P = 1>
void qsa_iv_chunk(Device& d, const float* queries, const float* blocks, size_t pool, size_t base, size_t steps, size_t selected,
                  float* scores, float* packed, size_t& valid) {
    float* scratch = d.attention_scores;
    const unsigned complete = unsigned((base + steps) / 4), first = unsigned((base + 1) / 4);
    float* copy = scratch + qsa_iv_copy_at(pool);
    float2* stats = reinterpret_cast<float2*>(scratch + qsa_iv_stats_at(pool));
    float2* partial = reinterpret_cast<float2*>(scratch + qsa_iv_partial_at(pool));
    float* sample = scratch + qsa_iv_sample_at(pool);
    float* info = scratch + qsa_iv_info_at(pool);
    unsigned* flags = reinterpret_cast<unsigned*>(scratch + qsa_iv_flags_at(pool));
    uint16_t* masks = reinterpret_cast<uint16_t*>(scratch + qsa_iv_masks_at(pool));
    const unsigned words = unsigned(qsa_iv_words(pool)), parts = (complete + 63) / 64;
    const unsigned sampled = (first / 32) & ~15u, sample_tiles = sampled / 16;
    const unsigned tiles = (complete + qsa_iv_unit - 1) / qsa_iv_unit;
    const unsigned groups = unsigned((steps + qsa_iv_rows - 1) / qsa_iv_rows);
    const unsigned from = std::min(unsigned(valid) & ~63u, (parts - 1) * 64);
    qsa_iv_keys<<<parts - from / 64, 256, 0, d.stream>>>(blocks, pool, complete, from, copy, stats, partial, flags);
    valid = complete;
    qsa_iv_scan<16, true><<<dim3((sample_tiles + 3) / 4, groups), 32 * qsa_iv_waves, 0, d.stream>>>(
        queries, copy, pool, base, steps, info, scores, masks, words, sample, complete, sample_tiles, 4);
    qsa_iv_threshold<<<unsigned(steps), 256, 0, d.stream>>>(queries, sample, sampled, stats, partial, parts, base, steps, info);
    qsa_iv_scan<qsa_iv_unit, false><<<dim3((tiles + qsa_iv_tiles_per_group - 1) / qsa_iv_tiles_per_group, groups), 32 * qsa_iv_waves, 0, d.stream>>>(
        queries, copy, pool, base, steps, info, scores, masks, words, sample, complete, tiles, qsa_iv_tiles_per_group);
    qsa_iv_select<P><<<unsigned(steps), 256, 0, d.stream>>>(queries, scores, stats, copy, pool, base, steps, selected, packed, info, masks, words, flags);
}
#endif
