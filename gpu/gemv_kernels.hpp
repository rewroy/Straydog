#pragma once
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
__global__ void gemv_fixed_kind(FixedGemv list, const float* input, float* output) { SD_KERNEL_ACQUIRE;
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

#if !defined(__HIPCC__)
// A block owns only one matrix, so its waves can share the same activation
// vector even when adjacent matrices have different kinds or input offsets.
// Padding is in waves only; each row keeps its original lanes and reduction.
// Two vectors: the second's activations after the first's.
template<bool compact, int V>
__global__ void gemv_mixed_shared(FixedGemv list, const float* input, float* output) {
    __shared__ __align__(16) float cached[V * 2560];
    const size_t first_wave = size_t(blockIdx.x) * (blockDim.x / 32);
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= first_wave) selected = i;
    const GemvItem& item = list.items[selected];
    Matrix matrix = item.matrix;
    for (unsigned i = threadIdx.x; i < 2560; i += blockDim.x) {
        #pragma unroll
        for (int v = 0; v < V; ++v) cached[v * 2560 + i] = input[matrix.input_offset + v * list.input_stride + i];
    }
    __syncthreads();
    matrix.input_offset = 0;
    // Eligibility excludes the reference decoder; keeping its code out of this
    // kernel also keeps its register and instruction footprint out of the mix.
    const size_t row = (first_wave - item.first_wave + threadIdx.x / 32) * (32 / item.lanes);
    #define SD_MIXED(format) case format: \
        if constexpr (!compact || format == 12 || format == 23 || format == 30) \
            gemv_rows<format, V>(matrix, row, item.lanes, cached, 2560, output, list.output_stride, list.residual); \
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

// gemv_fixed_kind (LINKED, one vector: gemv_linked_kind) on the staged input.
#if !defined(__HIPCC__)
// (CUDA) The lanes of a row ask the L2 for the first 128-byte line each of its weights before the block stages the input: the staging leaves
// the memory idle otherwise, some microseconds of a launch of ten to thirty. (More lines a lane, all of a row, were slower: the requests
// fill the queues ahead of the loads the first units wait for.)
__device__ __forceinline__ void gemv_q8_prefetch_row(const Matrix& m, size_t first_row, unsigned lanes) {
    const unsigned lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = first_row + lane / lanes;
    if (row < m.rows && size_t(sub) * 128 < m.row_bytes)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(m.data + row * m.row_bytes + size_t(sub) * 128));
}
#endif
template<int kind, int V, bool LINKED>
__global__ void __launch_bounds__(256) gemv_q8_kind(FixedGemv list, const float* input, float* output, LinkTail tail) { SD_KERNEL_ACQUIRE;
#if !defined(__HIPCC__)
    {
        const size_t first = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
        const GemvItem& at = gemv_q8_item(list, first);
        gemv_q8_prefetch_row(at.matrix, (first - at.first_wave) * (32 / at.lanes), at.lanes);
    }
#endif
    const GemvQ8Input in = gemv_q8_stage_list<kind, V>(list, input);
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const GemvItem& item = gemv_q8_item(list, wave);
    GemvResidual residual = list.residual;
    if (LINKED) residual.peer = link_tail_slot(tail);
    gemv_q8_rows<kind, V, gemv_q8_pad(kind)>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, in, output, list.output_stride,
                                             residual);
    if (LINKED) link_tail(tail);
}
#if defined(__HIPCC__)
// (DeepSeek-V4.1's q_b rows: HIP only)
// The 32-lane butterfly sums of gemv_q8_rows (total += __shfl_xor(total, offset, 32), offsets 16, 8, 4, 2, 1) of four rows' partial sums at
// once, 6 exchanges where four rows one after another take 20. After the offset-16 step a row's sums are the same in lanes l and l ^ 16, and after
// the offset-8 step in l and l ^ 8: so the step of two rows is one exchange (lanes with the bit set send the row the others keep, the others the
// row they do not), their sums in the lanes the bit picks, and the last three steps run on the four rows at once. Every lane adds the same two
// values it would in the row's own butterfly (an addition commutes), so the sums are the same. The rows' totals: row 0 in lane 0, row 1 in lane 16,
// row 2 in lane 8, row 3 in lane 24.
__device__ __forceinline__ float gemv_reduce4(float p0, float p1, float p2, float p3) {
    const unsigned lane = threadIdx.x % 32;
    const bool upper = lane & 16, middle = lane & 8;
    const float ab = (upper ? p1 : p0) + __shfl_xor(upper ? p0 : p1, 16, 32);
    const float cd = (upper ? p3 : p2) + __shfl_xor(upper ? p2 : p3, 16, 32);
    float q = (middle ? cd : ab) + __shfl_xor(middle ? ab : cd, 8, 32);
    q += __shfl_xor(q, 4, 32);
    q += __shfl_xor(q, 2, 32);
    q += __shfl_xor(q, 1, 32);
    return q;
}
// gemv_q8_kind of a Q5_K list whose rows take a wave each and a unit a lane at most (gemv_q8_lanes: 32 lanes over 16 to 31 units; DeepSeek-V4.1's
// q_b, 20), ROWS consecutive waves' rows to a wave, ROWS a multiple of four. Past its weights (14 MB: 28 us) a q_b row cost the LDS pipe, per vector
// 4 reads of the lane's unit of the staged activations and 5 exchanges of the butterfly: 4 vectors took 59 us. Here the lane keeps its unit's
// activations in registers for every row it meets, the next row's unit is loaded while this one is summed (the first, while the block stages),
// and four rows' butterflies are one (gemv_reduce4): 33 us. Every row's sum is gemv_q8_rows': its unit's dot added to zero in the lane, the
// lanes' sums butterflied.
template<int V, int ROWS>
__global__ void __launch_bounds__(256) gemv_q8_rows_wave(FixedGemv list, const float* input, float* output) { SD_KERNEL_ACQUIRE;
    using Q5 = GemvQ8<13>;
    const GemvItem& last = list.items[list.count - 1];
    const size_t waves = last.first_wave + last.matrix.rows;
    const size_t first = (size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32) * ROWS;
    const unsigned sub = threadIdx.x % 32;
    const size_t units = list.items[0].matrix.cols / Q5::elements;
    const auto fetch = [&](size_t wave) {
        Q5::Unit unit{};
        if (wave < waves && sub < units) {
            const GemvItem& item = gemv_q8_item(list, wave);
            const size_t row = wave - item.first_wave;
            if (row < item.matrix.rows) unit = Q5::load(item.matrix.data + row * item.matrix.row_bytes, sub);
        }
        return unit;
    };
    Q5::Unit current = fetch(first);
    const GemvQ8Input in = gemv_q8_stage_list<13, V>(list, input);
    Q5::Held<V> held{};
    if (sub < units) held.template take<gemv_q8_pad(13)>(in, sub);
    #pragma unroll 1
    for (int k = 0; k < ROWS && first + k < waves; k += 4) {
        float part[V][4] = {};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t wave = first + k + j;
            const Q5::Unit next = fetch(wave + 1);
            if (wave < waves && sub < units) {
                const GemvItem& item = gemv_q8_item(list, wave);
                if (wave - item.first_wave < item.matrix.rows) {
                    float r[V];
                    Q5::dots<V>(current, sub, held, r);
                    #pragma unroll
                    for (int v = 0; v < V; ++v) part[v][j] += r[v];
                }
            }
            current = next;
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float total = gemv_reduce4(part[v][0], part[v][1], part[v][2], part[v][3]);
            if (sub % 8 == 0) {
                const size_t wave = first + k + (sub & 16 ? 1 : 0) + (sub & 8 ? 2 : 0);
                if (wave < waves) {
                    const GemvItem& item = gemv_q8_item(list, wave);
                    const size_t row = wave - item.first_wave;
                    if (row < item.matrix.rows) gemv_store(item.matrix, row, v, total, output, list.output_stride, list.residual);
                }
            }
        }
    }
}
#endif

// A list of the wide kinds and BF16 (Device::q8_wide: Qwen3.8's projections, a layer's qkv, gate, beta and alpha in one launch; a
// matrix once a token, hc_project's): every block owns one matrix (its waves padded to the block) and stages that matrix's input of
// the V vectors, end to end; a wide kind's rows run on it as gemv_q8_kind's, a BF16 row on the F32 input as gemv_fixed_kind's. A
// row's sum is the same whatever shares its launch, so one token's decode and a verification's rows agree.
template<int V>
__global__ void __launch_bounds__(256) gemv_q8_mixed(FixedGemv list, const float* input, float* output) { SD_KERNEL_ACQUIRE;
    extern __shared__ __align__(16) uint32_t gemv_lds[];
    const size_t first_wave = size_t(blockIdx.x) * (blockDim.x / 32);
    const GemvItem& item = gemv_q8_item(list, first_wave);
    const int kind = item.matrix.kind;
#if !defined(__HIPCC__)
    if (kind != 30) gemv_q8_prefetch_row(item.matrix, (first_wave - item.first_wave + threadIdx.x / 32) * (32 / item.lanes), item.lanes);
#endif
    const size_t cols = item.matrix.cols, groups = cols / 32, stride = gemv_q8_vector_words(cols, kind);
    float* xd = reinterpret_cast<float*>(gemv_lds + gemv_q8_value_words(cols, kind));
    if (kind != 30) {
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* from = input + item.matrix.input_offset + v * list.input_stride;
            for (size_t group = threadIdx.x; group < groups; group += blockDim.x)
                gemv_q8_group(from + group * 32, gemv_lds + v * stride + gemv_q8_group_word_of(kind, group), xd + v * stride + group,
                              xd + v * stride + groups + group);
        }
    }
    __syncthreads();
    Matrix matrix = item.matrix;
    const GemvQ8Input in{gemv_lds, xd, xd + groups, stride};
    const size_t row = (first_wave - item.first_wave + threadIdx.x / 32) * (32 / item.lanes);
    #define SD_Q8_MIXED(format) case format: \
        matrix.input_offset = 0; \
        gemv_q8_rows<format, V, gemv_q8_pad(format)>(matrix, row, item.lanes, in, output, list.output_stride, list.residual); break;
    switch (matrix.kind) {
        SD_Q8_MIXED(8) SD_Q8_MIXED(12) SD_Q8_MIXED(13) SD_Q8_MIXED(14) SD_Q8_MIXED(23)
        case 30: gemv_rows<30, V>(matrix, row, item.lanes, input, list.input_stride, output, list.output_stride, list.residual); break;
    }
    #undef SD_Q8_MIXED
}

// ---- The launches (declared in gemv_common.hpp) ----
template<int V> void launch_gemv_fixed(unsigned blocks, unsigned threads, hipStream_t stream, const FixedGemv& list, const float* input, float* output) {
    gemv_fixed<V><<<blocks, threads, 0, stream>>>(list, input, output);
}
template<int kind, int V> void launch_gemv_fixed_kind(unsigned blocks, unsigned threads, hipStream_t stream, const FixedGemv& list, const float* input, float* output) {
    gemv_fixed_kind<kind, V><<<blocks, threads, 0, stream>>>(list, input, output);
}
template<int kind> void launch_gemv_linked_kind(unsigned blocks, hipStream_t stream, const FixedGemv& list, const float* input, float* output, const LinkTail& tail) {
    gemv_linked_kind<kind><<<blocks, 256, 0, stream>>>(list, input, output, tail);
}
template<int kind, int V, bool LINKED> void launch_gemv_q8_kind(unsigned blocks, size_t lds, hipStream_t stream, const FixedGemv& list, const float* input, float* output, const LinkTail& tail) {
    gemv_q8_kind<kind, V, LINKED><<<blocks, 256, lds, stream>>>(list, input, output, tail);
}
template<int V> void launch_gemv_q8_mixed(unsigned blocks, size_t lds, hipStream_t stream, const FixedGemv& list, const float* input, float* output) {
    gemv_q8_mixed<V><<<blocks, 256, lds, stream>>>(list, input, output);
}
#if !defined(__HIPCC__)
template<bool compact, int V> void launch_gemv_mixed_shared(unsigned blocks, hipStream_t stream, const FixedGemv& list, const float* input, float* output) {
    gemv_mixed_shared<compact, V><<<blocks, 128, 0, stream>>>(list, input, output);
}
template<int kind> void launch_gemv_decode_shared(unsigned blocks, hipStream_t stream, const Matrix& matrix, unsigned lanes, const float* input, float* output, const GemvResidual& residual) {
    gemv_decode_shared<kind><<<blocks, 128, 0, stream>>>(matrix, lanes, input, output, residual);
}
#else
template<int V, int ROWS> void launch_gemv_q8_rows_wave(unsigned blocks, size_t lds, hipStream_t stream, const FixedGemv& list, const float* input, float* output) {
    gemv_q8_rows_wave<V, ROWS><<<blocks, 256, lds, stream>>>(list, input, output);
}
template<int kind> void launch_gemv_decode_shared_rdna4(unsigned blocks, hipStream_t stream, const Matrix& matrix, unsigned lanes, const float* input, float* output, const GemvResidual& residual) {
    gemv_decode_shared_rdna4<kind><<<blocks, 256, 0, stream>>>(matrix, lanes, input, output, residual);
}
#endif
