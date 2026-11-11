// Batched prompt attention (RDNA4) where each token attends its own selection, its K/V rows read straight from memory. qsa_flash_wmma
// walks the union of 16 tokens' blocks, 3.6 to 5 times fewer blocks than the tokens select between them, but its tiles of loads, LDS
// staging and block-wide barriers, one workgroup a CU, take 56 to 98 us a token (both GPUs, 13 layers) against 19 here.
// A workgroup is a token and a K/V head; each of its W waves takes every W-th tile of 16 selected keys, all 256 dimensions of it, with
// its own online softmax, so no wave waits for another until the merge. Per tile S^T = K Q^T and O^T += V^T P^T are FP16 WMMA with F32
// accumulation, as in qsa_flash_wmma (keys as rows, so a lane's scores are already the B operand of the second product): the same
// FP16 keys, values, queries and probabilities and the same F32 softmax, summed in another order. The K rows go from memory into the
// A operands, the V rows are transposed through the wave's own LDS (no barrier: a wave's LDS accesses happen in order).
// A part's prefill shares the indexer (qsa_mark_send): the chunks the other device selected reach it as their 512 selected block numbers
// a row (16 bits, in the sum link's inbox), and `Received` reads those where the local kernel reads the selection list: the same
// entries in the same order (block order, four positions a block, then the incomplete block's positions), so the same sums.
#if defined(__HIPCC__)
typedef _Float16 qsa_half2 __attribute__((ext_vector_type(2)));

// Sixteen Q8_0 values (a uint4, in one block, so one scale) as FP16: kv_half4's halves, each the int8 times the block's FP16 scale
// rounded once. A byte b + 128 behind the exponent bits of 1024 is the half 1024 + b + 128; minus 1152 it is the int8, exactly.
__device__ __forceinline__ void qsa_q8_half16(uint4 raw, uint32_t scale_pair, moe_half8& low, moe_half8& high) {
    qsa_half2 scale;
    memcpy(&scale, &scale_pair, 4);
    const qsa_half2 bias = {_Float16(1152.0f), _Float16(1152.0f)};
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t out[8];
    #pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
        const uint32_t x = words[w] ^ 0x80808080u;
        const uint32_t a = __builtin_amdgcn_perm(0x64646464u, x, 0x05010400u), b = __builtin_amdgcn_perm(0x64646464u, x, 0x05030402u);
        qsa_half2 ha, hb;
        memcpy(&ha, &a, 4); memcpy(&hb, &b, 4);
        ha = (ha - bias) * scale; hb = (hb - bias) * scale;
        memcpy(&out[2 * w], &ha, 4); memcpy(&out[2 * w + 1], &hb, 4);
    }
    memcpy(&low, out, 16); memcpy(&high, out + 4, 16);
}

constexpr unsigned qsa_prompt_stride = 24;   // a transposed V row in halves: 16 keys, padded to 48 bytes
// Where a wave's transposed V tile (32 dimensions of 16 keys) holds `row` (0..31) and `key`; the second 16 rows shifted by 8 banks.
__device__ __forceinline__ unsigned qsa_prompt_at(unsigned row, unsigned key) { return row * qsa_prompt_stride + (row / 16) * 16 + key; }

// Grid (the chunk's tokens, H::kv), W waves of 32. `packed`: the chunk's queries and selections as qsa_flash reads them (the queries
// are rounded to FP16 here, as cast_half does for qsa_flash_wmma); `base`: the first token's position; `output`: the attended values.
// `Received`: grid (256, H::kv, the span's turns of the other device): token `blockIdx.x` of chunk `first_chunk + 2 * blockIdx.z` of a
// span of `steps` rows from position `base`, its selection the block numbers in `inbox` (row `token` of the slot `link[0] & 1`, 512
// each, as qsa_mark_send puts them) and its queries the F32 rows at `packed` (only the queries, `steps` rows of them), the attended
// values going over them: a workgroup reads its token's twelve heads before it writes them, and none touches another's.
// (a span from qsa_prompt_from: every row selects 512 blocks, past 2048 positions and the incomplete block's)
constexpr size_t qsa_share_head = qsa_share_blocks * 4;   // positions of the selected blocks, where a row's incomplete block's begin
template<int P = 1, unsigned W = 4, bool Q8 = true, bool Received = false>
__global__ void __launch_bounds__(W * 32) qsa_prompt_attention(unsigned kv_row, const kv_t* state, size_t capacity, size_t base, size_t steps,
                                                               size_t selected, const float* packed, float* output,
                                                               const uint16_t* inbox = nullptr, const uint32_t* link = nullptr,
                                                               size_t slot_halves = 0, size_t first_chunk = 0) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using H = QwenHeads<P>;
    constexpr unsigned threads = W * 32, query_stride = 264;
    size_t token = blockIdx.x;
    if constexpr (Received) {
        token += (first_chunk + 2 * size_t(blockIdx.z)) * 256;
        if (token >= steps) return;   // (the whole workgroup: before any barrier)
    }
    const unsigned kv = blockIdx.y, t = threadIdx.x, lane = t % 32, half = lane / 16, n = lane % 16;
    const unsigned wave = __builtin_amdgcn_readfirstlane(t / 32);
    const size_t index_start = Received ? 0 : steps * H::packed;
    [[maybe_unused]] const size_t complete = (base + token + 1) / 4;   // (Received: the complete blocks of the token's position)
    const size_t count = Received ? qsa_share_head + base + token + 1 - complete * 4
                                  : std::min(selected, size_t(packed[index_start + steps * selected + token]));
    const float* list = Received ? nullptr : packed + index_start + token * selected;
    const uint16_t* blocks = Received ? inbox + size_t(link[0] & 1) * slot_halves + token * qsa_share_blocks : nullptr;
    const float limit = float(base + token), bound = float(capacity);
    // The token's twelve query heads (and four rows of zeros: the B operand's 16 columns), then the merge's piece (the queries are done with).
    __shared__ SD_ALIGN(16) char shared[threads * 48 > 16 * query_stride * 2 ? threads * 48 : 16 * query_stride * 2];
    __shared__ SD_ALIGN(16) _Float16 staging[W][32 * qsa_prompt_stride + 32];
    __shared__ float statistics[W][2][16], denominators[16];
    _Float16* queries = reinterpret_cast<_Float16*>(shared);
    {
        typedef _Float16 half4 __attribute__((ext_vector_type(4)));
        const float4* source = reinterpret_cast<const float4*>(packed + (size_t(token) * H::query + kv * 12) * 256);
        for (unsigned i = t; i < 16 * 64; i += threads) {
            const float4 x = i < 12 * 64 ? source[i] : float4{0, 0, 0, 0};
            *reinterpret_cast<half4*>(queries + (i / 64) * query_stride + i % 64 * 4) = half4{_Float16(x.x), _Float16(x.y), _Float16(x.z), _Float16(x.w)};
        }
    }
    __syncthreads();
    const _Float16* query_row = queries + n * query_stride + 16 * half;
    const kv_t* keys = state + size_t(kv) * capacity * kv_row;
    const kv_t* values = state + size_t(H::kv + kv) * capacity * kv_row;
    _Float16* tile = staging[wave];
    moe_float8 sums[16] = {};
    float maximum = -INFINITY, total = 0;
    constexpr float scale = 0.0625f * 1.4426950408889634f;   // log2(e) / 16, as qsa_flash_wmma
    for (size_t first = size_t(wave) * 16; first < count; first += size_t(W) * 16) {
        // This lane's entry of the tile's 16 (key slot n), the ones that are keys, and the K and V rows' 16 dimensions of each 32.
        const size_t slot = first + n;
        float raw = -1.0f;
        if constexpr (Received) {
            // (a tile of 16 keys is four blocks' 16-bit numbers, one 8-byte load every lane shares; the tile past them is the tail)
            if (first < qsa_share_head) {
                const uint64_t four = link_load64(blocks + first / 4);
                raw = float(size_t((four >> (16 * (n / 4))) & 0xffffu) * 4 + n % 4);
            } else if (slot < count) raw = float(complete * 4 + slot - qsa_share_head);
        } else if (slot < count) raw = list[slot];
        const bool live = raw >= 0 && raw <= limit && raw < bound;
        const size_t position = live ? size_t(raw) : 0;
        const uint32_t mask = __builtin_amdgcn_ballot_w32(live);
        const kv_t* key = keys + position * kv_row;
        const kv_t* value = values + position * kv_row;
        uint4 k[8] = {}, v[8] = {}, key_scales = {}, value_scales = {};
        if constexpr (Q8) {
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) k[j] = *reinterpret_cast<const uint4*>(key + 32 * j + 16 * half);
            key_scales = *reinterpret_cast<const uint4*>(key + 256);
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) v[j] = *reinterpret_cast<const uint4*>(value + 32 * j + 16 * half);
            value_scales = *reinterpret_cast<const uint4*>(value + 256);
        }
        // The lane's 16 dimensions j * 32 + 16 * half of a row as two operands of 8 halves (the j-th block's scale).
        auto halves = [&](const uint4* dims, const uint4& scales, const kv_t* row, unsigned j, moe_half8& low, moe_half8& high) {
            if constexpr (Q8) {
                const uint32_t pair = ((j / 2 == 0 ? scales.x : j / 2 == 1 ? scales.y : j / 2 == 2 ? scales.z : scales.w) >> (16 * (j % 2))) & 0xffffu;
                qsa_q8_half16(dims[j], pair * 0x10001u, low, high);
            } else {
                const uint4* source = reinterpret_cast<const uint4*>(row + (32 * j + 16 * half) * 2);
                const uint4 a = source[0], b = source[1];
                memcpy(&low, &a, 16); memcpy(&high, &b, 16);
            }
        };
        moe_float8 score = {};
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            moe_half8 low, high;
            halves(k, key_scales, key, j, low, high);
            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(low, *reinterpret_cast<const moe_half8*>(query_row + 32 * j), score);
            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(high, *reinterpret_cast<const moe_half8*>(query_row + 32 * j + 8), score);
        }
        // Lane: query head n, keys 8 * half .. + 7 of the 16.
        float p[8], highest = -INFINITY;
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            p[i] = (mask >> (8 * half + i) & 1u) ? score[i] * scale : -INFINITY;
            highest = fmaxf(highest, p[i]);
        }
        highest = fmaxf(highest, __shfl_xor(highest, 16, 32));
        const float updated = fmaxf(maximum, highest), offset = updated == -INFINITY ? 0.0f : updated;
        float sum = 0;
        moe_half8 probabilities;
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            p[i] = exp2f(p[i] - offset);
            sum += p[i];
            probabilities[i] = _Float16(p[i]);
        }
        sum += __shfl_xor(sum, 16, 32);
        const float rescale = exp2f(maximum - offset);
        total = total * rescale + sum; maximum = updated;
        if (rescale != 1.0f) {
            #pragma unroll
            for (unsigned i = 0; i < 16; ++i) sums[i] *= rescale;
        }
        // The values, 32 dimensions at a time: transposed through the wave's LDS, then two WMMA tiles of 16 dimensions.
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            moe_half8 low, high;
            halves(v, value_scales, value, j, low, high);
            __builtin_amdgcn_wave_barrier();
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                tile[qsa_prompt_at(16 * half + i, n)] = low[i];
                tile[qsa_prompt_at(16 * half + 8 + i, n)] = high[i];
            }
            __builtin_amdgcn_wave_barrier();
            #pragma unroll
            for (unsigned part = 0; part < 2; ++part)
                sums[2 * j + part] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    *reinterpret_cast<const moe_half8*>(&tile[qsa_prompt_at(16 * part + n, 8 * half)]), probabilities, sums[2 * j + part]);
        }
    }
    // The waves' partial results, merged in a fixed order: each one's statistics, then 32 dimensions at a time.
    if (half == 0) { statistics[wave][0][n] = maximum; statistics[wave][1][n] = total; }
    __syncthreads();
    float peak = -INFINITY;
    #pragma unroll
    for (unsigned w = 0; w < W; ++w) peak = fmaxf(peak, statistics[w][0][n]);
    float weights[W], denominator = 0;
    #pragma unroll
    for (unsigned w = 0; w < W; ++w) {
        weights[w] = statistics[w][0][n] == -INFINITY ? 0.0f : exp2f(statistics[w][0][n] - peak);
        denominator += weights[w] * statistics[w][1][n];
    }
    const float mine = weights[wave];
    if (wave == 0 && half == 0) denominators[n] = denominator;
    float* merge = reinterpret_cast<float*>(shared);
    float* target = output + (size_t(token) * H::query + kv * 12) * 256;
    #pragma unroll
    for (unsigned part = 0; part < 8; ++part) {
        if (n < 12) {
            #pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                float* at = merge + (wave * 12 + n) * 32 + i * 16 + 8 * half;
                const moe_float8 value = sums[2 * part + i] * mine;
                *reinterpret_cast<float4*>(at) = float4{value[0], value[1], value[2], value[3]};
                *reinterpret_cast<float4*>(at + 4) = float4{value[4], value[5], value[6], value[7]};
            }
        }
        __syncthreads();
        for (unsigned i = t; i < 12 * 32; i += threads) {
            float acc = 0;
            #pragma unroll
            for (unsigned w = 0; w < W; ++w) acc += merge[(w * 12 + i / 32) * 32 + i % 32];
            target[(i / 32) * 256 + part * 32 + i % 32] = acc / denominators[i / 32];
        }
        __syncthreads();
    }
#endif
}
#endif

// The first position of a span from which a prompt batch attends token by token (qsa_prompt_attention) instead of by union
// (qsa_flash_wmma). The pinned generations' prompts end before 5.3K, so a span from here on keeps their bits.
constexpr size_t qsa_prompt_from = 8192;
// A chunk's tokens attending their own selections: `base` is the first token's position, `selected` the selections' stride in `packed`.
template<int P = 1>
void qsa_prompt_chunk(Device& d, const kv_t* state, size_t capacity, size_t base, size_t steps, size_t selected, const float* packed,
                      float* output) {
#if defined(__HIPCC__)
    constexpr unsigned heads = QwenHeads<P>::kv;
    const dim3 grid(unsigned(steps), heads);
    if (d.kv_row == kv_row_q8)
        qsa_prompt_attention<P, 4, true><<<grid, 128, 0, d.stream>>>(d.kv_row, state, capacity, base, steps, selected, packed, output);
    else
        qsa_prompt_attention<P, 4, false><<<grid, 128, 0, d.stream>>>(d.kv_row, state, capacity, base, steps, selected, packed, output);
    check(hipGetLastError());
#else
    throw std::runtime_error("QSA prompt attention needs RDNA4");
#endif
}
// The rows of a span's chunks `first_chunk`, `first_chunk` + 2, ... that the other device selected, attending the block numbers it sent
// (qsa_mark_send's inbox, after its flag): `queries` the span's first row of normalized F32 queries, which the attended values replace.
// `base` is the span's first position, `rows` its rows, `turns` the chunks.
template<int P = 1>
void qsa_prompt_received(Device& d, const kv_t* state, size_t capacity, size_t base, size_t rows, size_t first_chunk, size_t turns,
                         float* queries, const uint16_t* inbox, const uint32_t* link, size_t slot_halves) {
#if defined(__HIPCC__)
    constexpr unsigned heads = QwenHeads<P>::kv;
    const dim3 grid(256, heads, unsigned(turns));
    if (d.kv_row == kv_row_q8)
        qsa_prompt_attention<P, 4, true, true><<<grid, 128, 0, d.stream>>>(d.kv_row, state, capacity, base, rows, 0, queries, queries, inbox,
                                                                           link, slot_halves, first_chunk);
    else
        qsa_prompt_attention<P, 4, false, true><<<grid, 128, 0, d.stream>>>(d.kv_row, state, capacity, base, rows, 0, queries, queries, inbox,
                                                                            link, slot_halves, first_chunk);
    check(hipGetLastError());
#else
    throw std::runtime_error("QSA prompt attention needs RDNA4");
#endif
}
