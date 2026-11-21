#include "gemv_common.hpp"

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
unsigned gemv_lanes(const Matrix& m, const float* input, size_t concurrent) {
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
#if !defined(__HIPCC__)
    // CUDA: twice the lanes of the F32 kernels' plan (the same function of the matrix, so a row's sum is the same whatever shares its launch). A
    // row of the Q8 kernels has little to compute and each lane's units are a chain of dependent loads: half as many a lane, and twice the waves
    // to fill the SMs (Qwen3.8's 2560 x 6144 projections: 80 blocks of eight waves, an SM short of one each, become 160), took the dense front from
    // 45 to 40 microseconds on an RTX 5070 Ti, a floor of 38.
    if (lanes && d.q8_wide && (m.kind == 8 || m.kind == 12 || m.kind == 13 || m.kind == 14 || m.kind == 23) && m.cols % 256 == 0) lanes = std::min(32u, 2 * lanes);
#endif
    return lanes && d.q8_decode && m.kind == 13 && m.cols >= 1024 && m.cols < 2048 && m.cols % 256 == 0 ? 32 : lanes;
}
#if !defined(__HIPCC__)
// Which of gemv_mixed_shared's lists this is: 0 none, 1 compact, 2 Q3_K. Multiple tall projections amortize staging. Keep QSA,
// router and Q2-only mixes on their original path: measured staging regressions.
int gemv_mixed_shared_list(const FixedGemv& list, bool uniform) {
    if (uniform || list.count < 3) return 0;
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
    return !supported || tall < 2 ? 0 : compact ? 1 : q3 ? 2 : 0;
}
#endif
// Whether a list of `vectors` takes the Q8 kernels: one kind they take, every matrix the same input
// (offset and width: staged once), 16-byte aligned vectors; its LDS bytes, else 0.
size_t gemv_q8_input(const Device& d, const FixedGemv& list, const float* input, size_t vectors) {
    const Matrix& first = list.items[0].matrix;
    const size_t bytes = vectors * gemv_q8_vector_words(first.cols, first.kind) * 4;
    const bool taken = (d.q8_decode && gemv_q8_kind_taken(first.kind)) || (d.q8_wide && gemv_q8_wide_kind(first.kind));
    if (!taken || !list.count || first.cols % 256 || bytes > gemv_q8_lds_limit ||
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
    const size_t waves = last.first_wave + gemv_waves(last.matrix, last.lanes);
    const unsigned blocks = unsigned((waves + 7) / 8);
#if defined(__HIPCC__)
    // A Q5_K row of one unit a lane (q_b: 16384 rows of 20 units) takes four rows to a wave (eight for three or four vectors) when that leaves every CU
    // four blocks or more
    constexpr size_t rows_per_wave = V >= 3 ? 8 : 4;
    bool lone = V <= 4 && !tail && list.items[0].matrix.kind == 13 && list.items[0].matrix.cols / 64 <= 32 && waves >= 8 * rows_per_wave * 4 * d.compute_units;
    for (unsigned i = 0; i < list.count; ++i) lone &= list.items[i].lanes == 32;
    if constexpr (V <= 4) {
        if (lone) {
            launch_gemv_q8_rows_wave<V, rows_per_wave>(unsigned((waves + 8 * rows_per_wave - 1) / (8 * rows_per_wave)), lds, d.stream, list, input, output);
            check(hipGetLastError());
            return true;
        }
    }
#endif
    // (past four vectors, no tail)
    #define SD_Q8_LAUNCH(format) \
        if constexpr (V <= 4) { if (tail) launch_gemv_q8_kind<format, V, true>(blocks, lds, d.stream, list, input, output, *tail); \
            else launch_gemv_q8_kind<format, V, false>(blocks, lds, d.stream, list, input, output, LinkTail{}); } \
        else launch_gemv_q8_kind<format, V, false>(blocks, lds, d.stream, list, input, output, LinkTail{});
    switch (list.items[0].matrix.kind) {
        case 13: SD_Q8_LAUNCH(13) break;
        case 12: SD_Q8_LAUNCH(12) break;
        case 14: SD_Q8_LAUNCH(14) break;
        case 23: SD_Q8_LAUNCH(23) break;
        default: SD_Q8_LAUNCH(8) break;
    }
    #undef SD_Q8_LAUNCH
    check(hipGetLastError());
    return true;
}
// gemv_q8_mixed for a list it takes (its waves padded to whole blocks); false (nothing launched) otherwise.
// Its LDS bytes for `vectors`, else 0. With `decode`, a list of Device::q8_decode's kinds alone (no BF16 rows), on such a device.
size_t gemv_q8_mixed_input(const Device& d, const FixedGemv& list, const float* input, size_t vectors, bool decode = false) {
    if (!(decode ? d.q8_decode : d.q8_wide) || !list.count || (vectors > 1 && list.input_stride % 4)) return 0;
    bool wide = false;
    size_t words = 0;
    for (unsigned i = 0; i < list.count; ++i) {
        const Matrix& m = list.items[i].matrix;
        // (decode: not Q8_0, whose F32 rows, a value's decode a byte, took 35.7 / 36.5 us for wo_a's one and two rows against 37.4 / 38.7 here;
        //  Q6_K's took 38.2 / 55.6 against 32.6 / 34.9: 4 x 1024 x 4096 on an R9700, 2026-11-20)
        if (!list.items[i].lanes || (decode ? !gemv_q8_kind_taken(m.kind) || m.kind == 8 : (!gemv_q8_wide_kind(m.kind) && m.kind != 30))) return 0;
        if (m.kind != 30) {
            if (m.cols % 256 || reinterpret_cast<uintptr_t>(input + m.input_offset) % 16) return 0;
            words = std::max(words, gemv_q8_vector_words(m.cols, m.kind));
            wide = true;
        }
    }
    const size_t lds = vectors * words * 4;
    return wide && lds <= gemv_q8_lds_limit ? lds : 0;
}
template<int V>
bool gemv_q8_mixed_launch(Device& d, const FixedGemv& list, const float* input, float* output, bool decode = false) {
    const size_t lds = gemv_q8_mixed_input(d, list, input, V, decode);
    if (!lds) return false;
    FixedGemv padded = list;
    size_t waves = 0;
    for (unsigned i = 0; i < list.count; ++i) {
        padded.items[i].first_wave = uint32_t(waves);
        waves += (gemv_waves(list.items[i].matrix, list.items[i].lanes) + 7) / 8 * 8;
    }
    launch_gemv_q8_mixed<V>(unsigned(waves / 8), lds, d.stream, padded, input, output);
    check(hipGetLastError());
    return true;
}
bool gemv_shared_input_launch(Device& d, const FixedGemv& list, const float* input, float* output);   // gemv_fused.hpp
template<int V>
void gemv_launch(Device& d, const FixedGemv& list, unsigned blocks, bool uniform, const float* input, float* output) {
    // (the wide kinds take the Q8 kernels ahead of the F32 kernels' special shapes)
    if (d.q8_wide && ((uniform && gemv_q8_launch<V>(d, list, input, output, nullptr)) || gemv_q8_mixed_launch<V>(d, list, input, output))) return;
#if !defined(__HIPCC__)
    if constexpr (V <= 2) {
        const int mixed = gemv_mixed_shared_list(list, uniform);
        // (two vectors: the compact lists, RTX 5070 Ti, Qwen3.8's 16480-row DeltaNet projection: 94 us to 82)
        if (mixed == 1 || (V == 1 && mixed == 2)) {
            FixedGemv padded = list;
            size_t waves = 0;
            for (unsigned i = 0; i < list.count; ++i) {
                padded.items[i].first_wave = uint32_t(waves);
                waves += (gemv_waves(list.items[i].matrix, list.items[i].lanes) + 3) / 4 * 4;
            }
            if (mixed == 1) launch_gemv_mixed_shared<true, V>(unsigned(waves / 4), d.stream, padded, input, output);
            else if constexpr (V == 1) launch_gemv_mixed_shared<false, 1>(unsigned(waves / 4), d.stream, padded, input, output);
            return;
        }
    }
    if constexpr (V == 1) {
        const GemvItem& item = list.items[0];
        if (uniform && list.count == 1 && item.matrix.rows == 2560 && item.matrix.cols == 6144) {
            const unsigned shared_blocks = unsigned((gemv_waves(item.matrix, item.lanes) + 3) / 4);
            #define SD_SHARED(format) case format: launch_gemv_decode_shared<format>(shared_blocks, d.stream, item.matrix, item.lanes, input, output, list.residual); return;
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
            #define SD_RDNA4_SHARED(format) case format: launch_gemv_decode_shared_rdna4<format>(shared_blocks, d.stream, item.matrix, item.lanes, input, output, list.residual); return;
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
    // A list of one Q8 decode kind but Q8_0 whose matrices read different inputs (DeepSeek-V4.1's wo_a: a device's four groups, each its
    // own 4096 of the attention's output) took the F32 kernel, which gemv_q8_kind's one staged input cannot replace: gemv_q8_mixed stages
    // each block's own matrix's input. A row's sum is gemv_q8_kind's (the same staging, lanes and dots), whatever shares the launch.
    if (uniform && gemv_q8_mixed_launch<V>(d, list, input, output, true)) return;
#endif
    const unsigned threads = blocks < 2u * d.compute_units ? 64 : 256;
    blocks *= 256 / threads; // same logical waves, rows, lanes and reduction tree
    #define SD_UNIFORM(format) case format: launch_gemv_fixed_kind<format, V>(blocks, threads, d.stream, list, input, output); break;
    if (uniform) {
        switch (list.items[0].matrix.kind) {
            SD_FAST_FORMATS(SD_UNIFORM)
            default: launch_gemv_fixed<V>(blocks, threads, d.stream, list, input, output);
        }
    } else launch_gemv_fixed<V>(blocks, threads, d.stream, list, input, output);
    #undef SD_UNIFORM
}
// Five to eight vectors: the Q8 kernels, and the BF16 kernel a router's list takes; false (nothing launched) otherwise.
template<int V>
bool gemv_launch_wide(Device& d, const FixedGemv& list, bool uniform, const float* input, float* output) {
    if (d.q8_wide && ((uniform && gemv_q8_launch<V>(d, list, input, output, nullptr)) || gemv_q8_mixed_launch<V>(d, list, input, output))) return true;
#if defined(__HIPCC__)
    if (d.rdna4 && uniform && list.items[0].matrix.kind == 30) {
        const GemvItem& last = list.items[list.count - 1];
        const unsigned blocks = unsigned((last.first_wave + gemv_waves(last.matrix, last.lanes) + 7) / 8);
        launch_gemv_fixed_kind<30, V>(blocks, 256, d.stream, list, input, output);
        check(hipGetLastError());
        return true;
    }
#endif
    (void)d; (void)list; (void)uniform; (void)input; (void)output;
    return false;
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
#else
    // (CUDA's Q8 kernels have no linked form: gemv() runs the matrix on them, as it would here, and the sum is its own launch)
    if (d.q8_wide && gemv_q8_input(d, list, input, 1)) return false;
#endif
    #define SD_LINKED(format) case format: launch_gemv_linked_kind<format>(blocks, d.stream, list, input, output, tail); return true;
    switch (matrix.kind) {
        SD_FAST_FORMATS(SD_LINKED)
    }
    #undef SD_LINKED
    return false;
}
// Launch up to 24 independent row ranges in one dispatch, for up to eight vectors (past four, gemv_launch_wide). min_lanes (a
// power of two) widens a row that gemv_lanes splits over only two lanes, when it has at least 32
// units: two lanes read short runs of many rows at once.
void gemv(Device& d, const Matrix* matrices, size_t count, const float* input, float* output,
          size_t vectors, size_t input_stride, size_t output_stride, const GemvResidual& residual,
          unsigned min_lanes) {
    if (!count || count > gemv_fixed_items || !vectors || vectors > size_t(gemv_wide_vectors) ||
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
#if defined(__HIPCC__)
    const bool mixed_q8 = !uniform && gemv_q8_mixed_input(d, list, input, vectors);
#else
    // (or it goes whole to gemv_mixed_shared, or to the mixed Q8 kernel)
    const bool mixed_q8 = (!uniform && gemv_q8_mixed_input(d, list, input, vectors)) ||
                          (vectors == 2 && gemv_mixed_shared_list(list, uniform) == 1);
#endif
    // (with the wide Q8 kernels, a list they cannot take whole is split for one vector too: a row then
    // takes the kernel of its kind whatever shares its launch, as a verification's rows do)
    if (!uniform && (vectors > 1 || d.q8_wide) && !mixed_q8) {
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
        case 4: gemv_launch<4>(d, list, blocks, uniform, input, output); break;
        default: {
            // Past four vectors, the kernels that share their weights' reads for them; other lists in two launches.
            const bool joined = (vectors == 5 ? gemv_launch_wide<5>(d, list, uniform, input, output) :
                                                       vectors == 6 ? gemv_launch_wide<6>(d, list, uniform, input, output) :
                                                       vectors == 7 ? gemv_launch_wide<7>(d, list, uniform, input, output) :
                                                                      gemv_launch_wide<8>(d, list, uniform, input, output));
            if (joined) break;
            const size_t first = size_t(gemv_max_vectors);
            gemv(d, matrices, count, input, output, first, input_stride, output_stride, residual, min_lanes);
            GemvResidual rest = residual;
            if (rest.hidden) { rest.hidden += first * rest.hidden_stride; rest.mix += first * rest.mix_stride; }
            if (rest.gate) rest.gate += first;
            gemv(d, matrices, count, input + first * input_stride, output ? output + first * output_stride : nullptr, vectors - first, input_stride,
                 output_stride, rest, min_lanes);
            return;
        }
    }
    check(hipGetLastError());
}
