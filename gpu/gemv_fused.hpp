// Decode GEMVs (one vector) whose input a prologue puts in shared memory: as it is, SwiGLU of a
// gate and an up projection (op_swiglu_kernel's arithmetic, no clamp or weights) or the RMS norm
// of a raw vector (op_rms_norm_kernel's sums and products; the first block also stores the normed
// vector for the ops after it), so the results are those of the separate ops. Rows take the lanes
// the plain GEMV gives them and read the vector from shared memory as gemv_decode_shared does. A
// block forming the norm itself costs it a few microseconds; with the weights streaming from VRAM
// that hides behind the other blocks', and the norm's launch is saved. Included in the GPU TU
// after gemv.hpp and ops.hpp.

namespace {

// One row's norm and rope (op_norm_rope_row) for a block of its own beside a GEMV's.
struct NormRopeJob {
    const float* input;
    const float* weight;
    uint32_t width;
    float eps;
    uint32_t pairs;
    const int32_t* positions;
    const float* frequencies;
    const int32_t* rows;
    float* output;         // null: no job
    double2* forward;      // with inverse: this row's attention rope, else null
    double2* inverse;
    bool publish;
};

struct GemvPrologue {
    int mode;              // 0: x as it is; 1: silu(x) * y; 2: the RMS norm of x times y (the weight)
    const float* x;
    const float* y;
    uint32_t width;
    float eps;
    float* normed;         // mode 2: where block 0 stores the normed vector (or null)
    // With route_logits (one GEMV kind, one more block): that block's first wave picks a row of
    // routes meanwhile, as op_route_softmax256_kernel does (256 experts, normalized).
    const float* route_logits;
    uint32_t route_k;
    int32_t* route_ids;
    float* route_weights;
    // With side.output (one more block, the last; not with route_logits): that block forms the
    // side job's row meanwhile.
    NormRopeJob side;
};

// The prologue GEMV's last block, when it runs the side job: true once it has (the block is done).
__device__ inline bool gemv_side_job(const GemvPrologue& pro, float* sums) {
    if (!pro.side.output || blockIdx.x != gridDim.x - 1) return false;
    const NormRopeJob& job = pro.side;
    op_norm_rope_row(job.input, job.weight, job.width, job.eps, job.pairs, job.positions, job.frequencies, job.rows, job.output, 0,
                     sums, job.forward, job.inverse);
    return true;
}

// After one vector's GEMV (gemv_fixed_kind's rows), its launch's last block forms the RMS norm of
// x (an output of the GEMV) into normed, then the side job's row: op_rms_norm_kernel's and
// op_norm_rope_kernel's arithmetic.
struct NormTail {
    const float* x;
    const float* weight;
    uint32_t width;
    float eps;
    float* normed;
    NormRopeJob side;
};
// The blocks of a gemv_norm_tail_kind launch done so far, per device: zero between launches (the last
// block resets it). A module variable, not an allocation: the first launch may be in a graph's capture.
__device__ unsigned gemv_norm_tail_done = 0;
template<int kind>
__global__ void gemv_norm_tail_kind(FixedGemv list, const float* input, float* output, NormTail tail) {
    __shared__ float sums[256];
    __shared__ bool last;
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= wave) selected = i;
    const GemvItem& item = list.items[selected];
    gemv_rows<kind, 1>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, input, list.input_stride, output,
                       list.output_stride, list.residual);
    // Every block's rows are out before it counts itself done.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&gemv_norm_tail_done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) gemv_norm_tail_done = 0;
    op_rms_norm_row(tail.x, tail.weight, tail.width, tail.eps, tail.normed, sums);
    __syncthreads();
    const NormRopeJob& job = tail.side;
    op_norm_rope_row(job.input, job.weight, job.width, job.eps, job.pairs, job.positions, job.frequencies, job.rows, job.output, 0,
                     sums, job.forward, job.inverse);
}

#if defined(__HIPCC__)
// gemv_norm_tail_kind on the Q8 kernels' rows (gemv_q8_input: one input staged once), the same tail.
template<int kind>
__global__ void __launch_bounds__(256) gemv_norm_tail_q8(FixedGemv list, const float* input, float* output, NormTail tail) {
    __shared__ float sums[256];
    __shared__ bool last;
    const GemvQ8Input in = gemv_q8_stage_list<1>(list, input);
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const GemvItem& item = gemv_q8_item(list, wave);
    gemv_q8_rows<kind, 1>(item.matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, in, output, list.output_stride,
                          list.residual);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&gemv_norm_tail_done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) gemv_norm_tail_done = 0;
    op_rms_norm_row(tail.x, tail.weight, tail.width, tail.eps, tail.normed, sums);
    __syncthreads();
    const NormRopeJob& job = tail.side;
    op_norm_rope_row(job.input, job.weight, job.width, job.eps, job.pairs, job.positions, job.frequencies, job.rows, job.output, 0,
                     sums, job.forward, job.inverse);
}
#endif

template<int kind>
__global__ void gemv_prologue_kind(FixedGemv list, GemvPrologue pro, float* output) {
    extern __shared__ __align__(16) float cached[];
    __shared__ float sums[256];
    __shared__ float total;
    const unsigned tid = threadIdx.x;
    if (pro.route_logits && blockIdx.x == gridDim.x - 1) {
        if (tid < 32) op_route_softmax256_wave(pro.route_logits, pro.route_k, 1, pro.route_ids, pro.route_weights, tid);
        return;
    }
    if (gemv_side_job(pro, sums)) return;
    if (pro.mode == 2) {
        float sum = 0.0f;
        for (size_t col = tid; col < pro.width; col += blockDim.x) sum += pro.x[col] * pro.x[col];
        sums[tid] = sum;
        __syncthreads();
        if (tid < 32) {
            const float value = tree256_first_wave(sums, tid);
            if (!tid) total = value;
        }
        __syncthreads();
        const float scale = rsqrtf(total / float(pro.width) + pro.eps);
        for (size_t col = tid; col < pro.width; col += blockDim.x) {
            const float value = pro.x[col] * scale * pro.y[col];
            cached[col] = value;
            if (!blockIdx.x && pro.normed) pro.normed[col] = value;
        }
    } else if (pro.mode == 1) {
        for (size_t col = tid; col < pro.width; col += blockDim.x) {
            const float g = pro.x[col], u = pro.y[col];
            cached[col] = g / (1.0f + expf(-g)) * u * 1.0f;
        }
    } else {
        for (size_t col = tid; col < pro.width; col += blockDim.x) cached[col] = pro.x[col];
    }
    __syncthreads();
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + tid / 32;
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= wave) selected = i;
    const GemvItem& item = list.items[selected];
    Matrix matrix = item.matrix;
    matrix.input_offset = 0;
    gemv_rows<kind, 1>(matrix, (wave - item.first_wave) * (32 / item.lanes), item.lanes, cached, 0, output, 0, list.residual);
}

// The same over items of two kinds (each wave runs its item's kind): the F32 router or a DeltaNet's
// F32 beta and alpha in the launch of the Q8_0 or BF16 projections beside them.
template<int K1, int K2>
__global__ void gemv_prologue_kind2(FixedGemv list, GemvPrologue pro, float* output) {
    extern __shared__ __align__(16) float cached[];
    __shared__ float sums[256];
    __shared__ float total;
    const unsigned tid = threadIdx.x;
    if (gemv_side_job(pro, sums)) return;
    if (pro.mode == 2) {
        float sum = 0.0f;
        for (size_t col = tid; col < pro.width; col += blockDim.x) sum += pro.x[col] * pro.x[col];
        sums[tid] = sum;
        __syncthreads();
        if (tid < 32) {
            const float value = tree256_first_wave(sums, tid);
            if (!tid) total = value;
        }
        __syncthreads();
        const float scale = rsqrtf(total / float(pro.width) + pro.eps);
        for (size_t col = tid; col < pro.width; col += blockDim.x) {
            const float value = pro.x[col] * scale * pro.y[col];
            cached[col] = value;
            if (!blockIdx.x && pro.normed) pro.normed[col] = value;
        }
    } else {
        for (size_t col = tid; col < pro.width; col += blockDim.x) cached[col] = pro.x[col];
    }
    __syncthreads();
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + tid / 32;
    unsigned selected = 0;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].first_wave <= wave) selected = i;
    const GemvItem& item = list.items[selected];
    Matrix matrix = item.matrix;
    matrix.input_offset = 0;
    const size_t first = (wave - item.first_wave) * (32 / item.lanes);
    if (matrix.kind == K1) gemv_rows<K1, 1>(matrix, first, item.lanes, cached, 0, output, 0, list.residual);
    else gemv_rows<K2, 1>(matrix, first, item.lanes, cached, 0, output, 0, list.residual);
}

// The rows' plan as gemv() makes it for one vector (lanes per row, waves), so each row's dot is
// the plain GEMV's; false when a matrix takes the reference decoder or there are more than two
// kinds.
bool gemv_prologue_plan(const Device& d, const Matrix* matrices, size_t count, const float* input, unsigned min_lanes, FixedGemv& list,
                        size_t& waves) {
    if (!count || count > gemv_fixed_items) return false;
    list = FixedGemv{};
    waves = 0;
    for (size_t i = 0; i < count; ++i) {
        unsigned lanes = gemv_lanes(matrices[i], input);
        size_t alignment = 0;
        if (lanes == 2 && min_lanes > 2 && matrices[i].cols / gemv_unit(matrices[i].kind, alignment) >= 32)
            lanes = std::min(32u, min_lanes);
        lanes = gemv_q8_lanes(d, matrices[i], lanes);
        if (!lanes) return false;
        bool seen = false;
        for (size_t j = 0; j < i; ++j) seen |= matrices[j].kind == matrices[i].kind;
        if (!seen) {
            bool second = false;
            for (size_t j = 0; j < i; ++j) second |= matrices[j].kind != matrices[0].kind;
            if (second && matrices[i].kind != matrices[0].kind) return false;   // a third kind
        }
        list.items[i] = {matrices[i], uint32_t(waves), lanes};
        waves += gemv_waves(matrices[i], lanes);
    }
    list.count = uint32_t(count);
    return true;
}

// Launch the prologue GEMV over `list` (its rows' kind must have a fast unit); false when the
// kind has none here.
bool gemv_prologue_launch(Device& d, const FixedGemv& list, size_t waves, const GemvPrologue& pro, float* output) {
    if (pro.route_logits && pro.side.output) return false;
    const unsigned blocks = unsigned((waves + 7) / 8) + (pro.route_logits || pro.side.output ? 1 : 0);
    const size_t shared = size_t(pro.width) * 4;
    int other = list.items[0].matrix.kind;
    for (unsigned i = 1; i < list.count; ++i) if (list.items[i].matrix.kind != other) { other = list.items[i].matrix.kind; break; }
    if (other != list.items[0].matrix.kind) {
        if (pro.mode == 1 || pro.route_logits) return false;
        const int a = std::min(other, list.items[0].matrix.kind), b = std::max(other, list.items[0].matrix.kind);
        if (a == 0 && b == 8) gemv_prologue_kind2<8, 0><<<blocks, 256, shared, d.stream>>>(list, pro, output);
        else if (a == 0 && b == 30) gemv_prologue_kind2<30, 0><<<blocks, 256, shared, d.stream>>>(list, pro, output);
        else return false;
        return true;
    }
    #define SD_PROLOGUE(format) case format: gemv_prologue_kind<format><<<blocks, 256, shared, d.stream>>>(list, pro, output); return true;
    switch (list.items[0].matrix.kind) {
        SD_FAST_FORMATS(SD_PROLOGUE)
    }
    #undef SD_PROLOGUE
    return false;
}

// One vector over BF16 rows that all read the same input (offset 0, the same columns): the input
// in shared memory (measured on RDNA4 at 4096 x 2048 from VRAM: 307 GB/s with the input read
// from global memory, 416 with it in shared memory; Q8_0 rows are faster reading it from global
// memory, 491 against 447). Same rows' lanes and dots, so the same results.
bool gemv_shared_input_launch(Device& d, const FixedGemv& list, const float* input, float* output) {
    if (list.items[0].matrix.kind != 30) return false;
    const size_t cols = list.items[0].matrix.cols;
    size_t waves = 0;
    for (unsigned i = 0; i < list.count; ++i) {
        const GemvItem& item = list.items[i];
        if (item.matrix.input_offset || item.matrix.cols != cols || !item.lanes || item.matrix.kind != 30) return false;
        waves = item.first_wave + gemv_waves(item.matrix, item.lanes);
    }
    if (cols > 8192 || waves * (32 / list.items[0].lanes) < 2048) return false;   // (few rows: the copy is not worth it)
    return gemv_prologue_launch(d, list, waves, GemvPrologue{0, input, nullptr, uint32_t(cols), 0.0f, nullptr}, output);
}

// The linears' matrices (one vector, every item reading `input`) as sd_op_linears plans them;
// `output` is the lowest item output, each item's offset from it.
void fused_matrices(Device& value, const sd_linear* items, size_t count, const float* input, Matrix* matrices, float*& output) {
    output = nullptr;
    for (size_t i = 0; i < count; ++i) {
        const sd_linear& item = items[i];
        if (!item.rows || item.rows > (uint64_t(1) << 24) || item.first_row > (uint64_t(1) << 32) || !item.cols || item.cols > 8192 ||
            item.input != input || uintptr_t(item.output) % 4)
            throw std::runtime_error("invalid Straydog fused linear");
        const size_t row_bytes = matrix_row_bytes(item.kind, item.cols);
        op_span(value, item.weights, checked_product({item.first_row + item.rows, row_bytes}), "a weight matrix");
        op_span(value, item.output, checked_floats({item.rows}), "a linear output");
        matrices[i] = {static_cast<const uint8_t*>(item.weights) + item.first_row * row_bytes, row_bytes, size_t(item.rows),
                       size_t(item.cols), 0, 0, 0, item.kind};
        output = !output || item.output < output ? item.output : output;
    }
    for (size_t i = 0; i < count; ++i) matrices[i].output_offset = size_t(items[i].output - output);
}

} // namespace

namespace {
// The side job of sd_op_norm_linears_rope, checked as sd_op_norm_rope checks one row.
NormRopeJob norm_rope_job(Device& value, const sd_norm_rope& side) {
    if (!side.width || side.width > (uint64_t(1) << 24) || side.rope_dims % 2 || side.rope_dims > side.width ||
        (!side.out_rows && side.out_count < 1))
        throw std::runtime_error("invalid Straydog norm shape");
    op_span(value, side.input, checked_floats({side.width}), "a norm input");
    op_span(value, side.weight, checked_floats({side.width}), "a norm weight");
    if (side.rope_dims) {
        op_span(value, side.positions, 4, "rope positions");
        op_span(value, side.frequencies, checked_floats({side.rope_dims / 2}), "rope frequencies");
    }
    op_span(value, side.out, checked_floats({side.out_count, side.width}), "a norm output");
    if (side.out_rows) {
        op_span(value, side.out_rows, 4, "output rows");
        if (side.out == side.input) throw std::runtime_error("a scattered norm cannot run in place");
    }
    AttentionTurnsView turns;
    if (side.rope_dims) turns = attention_turns_for_rope(value, 1, size_t(side.rope_dims / 2));
    return NormRopeJob{side.input, side.weight, uint32_t(side.width), side.eps, uint32_t(side.rope_dims / 2), side.positions,
                       side.frequencies, side.out_rows, side.out, turns.forward, turns.inverse, turns.publish};
}

// sd_op_norm_linears, with a side job (else output null) in the same launch where it can (else
// after the linears).
void norm_linears(Device& value, const float* x, const float* weight, uint64_t width, float eps, float* normed,
                  const sd_linear* items, uint64_t count, const NormRopeJob& side) {
    if (!width || width > 8192 || !items || !count || count > gemv_fixed_items) throw std::runtime_error("invalid Straydog norm linears");
    op_span(value, x, size_t(width) * 4, "a norm input");
    op_span(value, weight, size_t(width) * 4, "a norm weight");
    op_span(value, normed, size_t(width) * 4, "a norm output");
    Matrix matrices[gemv_fixed_items];
    float* output = nullptr;
    fused_matrices(value, items, size_t(count), normed, matrices, output);
    for (size_t i = 0; i < count; ++i)
        if (matrices[i].cols != width) throw std::runtime_error("a norm linear's columns differ from the norm's width");
    FixedGemv list;
    size_t waves = 0;
    GemvPrologue pro{2, x, weight, uint32_t(width), eps, normed};
    pro.side = side;
    if (gemv_prologue_plan(value, matrices, size_t(count), normed, op_lanes, list, waves) && gemv_prologue_launch(value, list, waves, pro, output)) {
        check(hipGetLastError());
        if (side.publish) attention_turns_mark_ready(value, 1, side.pairs);
        return;
    }
    op_rms_norm_kernel<<<1, 256, 0, value.stream>>>(x, weight, size_t(width), eps, normed);
    // a launch per kind, as sd_op_linears groups them
    bool done[gemv_fixed_items]{};
    for (size_t i = 0; i < count; ++i) {
        if (done[i]) continue;
        Matrix group[gemv_fixed_items];
        size_t size = 0;
        float* group_output = nullptr;
        for (size_t j = i; j < count; ++j)
            if (!done[j] && matrices[j].kind == matrices[i].kind) {
                group[size] = matrices[j];
                float* at = output + matrices[j].output_offset;
                group_output = !group_output || at < group_output ? at : group_output;
                ++size;
                done[j] = true;
            }
        for (size_t j = 0; j < size; ++j) group[j].output_offset = size_t(output + group[j].output_offset - group_output);
        gemv(value, group, size, normed, group_output, 1, 0, 0, {}, op_lanes);
    }
    if (side.output)
        op_norm_rope_kernel<<<1, 256, 0, value.stream>>>(side.input, side.weight, side.width, side.eps, side.pairs, side.positions,
                                                          side.frequencies, side.rows, side.output, side.forward, side.inverse);
    if (side.publish) attention_turns_mark_ready(value, 1, side.pairs);
    check(hipGetLastError());
}
} // namespace

namespace {
// sd_op_linears_norm_rope: the linears in gemv()'s launch, its last block forming the norm and the
// side job, where gemv() takes gemv_fixed_kind; else the three ops apart, in order.
void linears_norm(Device& value, const sd_linear* items, uint64_t count, const float* x, const float* weight, uint64_t width,
                  float eps, float* normed, const NormRopeJob& side) {
    if (!items || !count || count > gemv_fixed_items || !width || width > 8192) throw std::runtime_error("invalid Straydog linears norm");
    op_span(value, x, size_t(width) * 4, "a norm input");
    op_span(value, weight, size_t(width) * 4, "a norm weight");
    op_span(value, normed, size_t(width) * 4, "a norm output");
    const float* input = items[0].input;
    Matrix matrices[gemv_fixed_items];
    float* output = nullptr;
    fused_matrices(value, items, size_t(count), input, matrices, output);
    // (every item reads `input`, so the widest one bounds it)
    size_t cols = 0;
    for (size_t i = 0; i < count; ++i) cols = matrices[i].cols > cols ? matrices[i].cols : cols;
    op_span(value, input, checked_floats({cols}), "a linear input");
    FixedGemv list;
    size_t waves = 0;
    bool uniform = gemv_prologue_plan(value, matrices, size_t(count), input, op_lanes, list, waves);
    for (size_t i = 0; uniform && i < count; ++i) uniform = matrices[i].kind == matrices[0].kind;
    // (gemv() takes other kernels for BF16 rows and a lone 2560 x 6144 matrix)
    const int kind = matrices[0].kind;
    if (uniform && kind != 30 && !(count == 1 && matrices[0].rows == 2560 && matrices[0].cols == 6144)) {
        const NormTail tail{x, weight, uint32_t(width), eps, normed, side};
        const unsigned blocks = unsigned((waves + 7) / 8);
        // (256 threads, whatever gemv() would take: the same waves, rows, lanes and dots)
        bool launched = false;
#if defined(__HIPCC__)
        if (const size_t lds = gemv_q8_input(value, list, input, 1)) {
            if (kind == 13) gemv_norm_tail_q8<13><<<blocks, 256, lds, value.stream>>>(list, input, output, tail);
            else gemv_norm_tail_q8<8><<<blocks, 256, lds, value.stream>>>(list, input, output, tail);
            launched = true;
        }
#endif
        #define SD_NORM_TAIL(format) case format: \
            gemv_norm_tail_kind<format><<<blocks, 256, 0, value.stream>>>(list, input, output, tail); launched = true; break;
        if (!launched) switch (kind) { SD_FAST_FORMATS(SD_NORM_TAIL) }
        #undef SD_NORM_TAIL
        if (launched) {
            check(hipGetLastError());
            if (side.publish) attention_turns_mark_ready(value, 1, side.pairs);
            return;
        }
    }
    gemv(value, matrices, size_t(count), input, output, 1, 0, 0, {}, op_lanes);
    op_rms_norm_kernel<<<1, 256, 0, value.stream>>>(x, weight, size_t(width), eps, normed);
    op_norm_rope_kernel<<<1, 256, 0, value.stream>>>(side.input, side.weight, side.width, side.eps, side.pairs, side.positions,
                                                      side.frequencies, side.rows, side.output, side.forward, side.inverse);
    if (side.publish) attention_turns_mark_ready(value, 1, side.pairs);
    check(hipGetLastError());
}
} // namespace

int sd_op_linears_norm_rope(void* handle, const sd_linear* items, uint64_t count, const float* x, const float* weight, uint64_t width,
                            float eps, float* normed, const sd_norm_rope* side) {
    SD_OP_BEGIN
    if (!side) throw std::runtime_error("invalid Straydog norm shape");
    linears_norm(value, items, count, x, weight, width, eps, normed, norm_rope_job(value, *side));
    SD_OP_END
}

int sd_op_norm_linears(void* handle, const float* x, const float* weight, uint64_t width, float eps, float* normed,
                       const sd_linear* items, uint64_t count) {
    SD_OP_BEGIN
    norm_linears(value, x, weight, width, eps, normed, items, count, NormRopeJob{});
    SD_OP_END
}

int sd_op_norm_linears_rope(void* handle, const float* x, const float* weight, uint64_t width, float eps, float* normed,
                            const sd_linear* items, uint64_t count, const sd_norm_rope* side) {
    SD_OP_BEGIN
    if (!side) throw std::runtime_error("invalid Straydog norm shape");
    norm_linears(value, x, weight, width, eps, normed, items, count, norm_rope_job(value, *side));
    SD_OP_END
}

namespace {
// sd_op_swiglu_linear, and with route_logits sd_op_route's softmax routes of that row (256 experts,
// normalized) in the same launch (else after it).
void swiglu_linear(Device& value, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* gate, const float* up,
                   const float* scale, float* hidden, float* out, const float* route_logits, uint64_t route_k, int32_t* route_ids,
                   float* route_weights) {
    if (!rows || rows > (uint64_t(1) << 24) || !cols || cols > 8192) throw std::runtime_error("invalid Straydog SwiGLU linear");
    if (route_logits) {
        if (!route_k || route_k > 64) throw std::runtime_error("invalid Straydog route shape");
        op_span(value, route_logits, 256 * 4, "router logits");
        op_span(value, route_ids, size_t(route_k) * 4, "routed experts");
        op_span(value, route_weights, size_t(route_k) * 4, "routing weights");
    }
    op_span(value, gate, checked_floats({cols}), "a gate projection");
    op_span(value, up, checked_floats({cols}), "an up projection");
    if (scale) op_span(value, scale, 4, "a row scale");
    op_span(value, hidden, checked_floats({cols}), "a SwiGLU output");
    const size_t row_bytes = matrix_row_bytes(kind, cols);
    op_span(value, weights, checked_product({rows, row_bytes}), "a weight matrix");
    op_span(value, out, checked_floats({rows}), "a linear output");
    const Matrix matrix{static_cast<const uint8_t*>(weights), row_bytes, size_t(rows), size_t(cols), 0, 0, 0, kind};
    FixedGemv list;
    size_t waves = 0;
    if (gemv_prologue_plan(value, &matrix, 1, hidden, op_lanes, list, waves)) {
        list.residual.gate = scale;
        const GemvPrologue pro{1, gate, up, uint32_t(cols), 0.0f, nullptr, route_logits, uint32_t(route_k), route_ids, route_weights};
        if (gemv_prologue_launch(value, list, waves, pro, out)) {
            check(hipGetLastError());
            return;
        }
    }
    op_swiglu_kernel<<<op_blocks(size_t(cols)), 256, 0, value.stream>>>(gate, up, 1, size_t(cols), 0.0f, nullptr, hidden);
    GemvResidual gated{};
    gated.gate = scale;
    gemv(value, &matrix, 1, hidden, out, 1, 0, 0, gated, op_lanes);
    if (route_logits) op_route_softmax256_kernel<<<1, 32, 0, value.stream>>>(route_logits, size_t(route_k), 1, route_ids, route_weights);
    check(hipGetLastError());
}
} // namespace

int sd_op_swiglu_linear(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* gate,
                        const float* up, const float* scale, float* hidden, float* out) {
    SD_OP_BEGIN
    swiglu_linear(value, weights, kind, rows, cols, gate, up, scale, hidden, out, nullptr, 0, nullptr, nullptr);
    SD_OP_END
}

int sd_op_swiglu_linear_route(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* gate,
                              const float* up, const float* scale, float* hidden, float* out, const float* logits, uint64_t k,
                              int32_t* ids, float* route_weights) {
    SD_OP_BEGIN
    if (!logits) throw std::runtime_error("invalid Straydog route shape");
    swiglu_linear(value, weights, kind, rows, cols, gate, up, scale, hidden, out, logits, k, ids, route_weights);
    SD_OP_END
}
