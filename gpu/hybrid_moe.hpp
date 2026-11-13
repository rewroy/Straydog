// Single-token hybrid MoE: GPU part and the host CPU-expert step (see hybrid_slots.hpp).
// blockIdx.y a row, in every kernel of this file that takes one.
__device__ inline uint64_t await_clock() {
#if defined(__HIPCC__)
    return wall_clock64();
#else
    uint64_t now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); return now;
#endif
}
// Where the host's wait for a layer's export goes, on the GPU's own clock, in device memory (d.flags + 8, read by
// sd_hybrid_gpu_timing when a request ends): [0] the layer's start, once the previous layer's CPU result is in
// (hybrid_finish), [1] its export (hybrid_tick), [2] start to export summed, [3] export to the resident experts'
// combine summed. One thread's store in kernels that run anyway: nothing crosses the bus for it.
inline uint64_t* hybrid_clock(Device& d) { return reinterpret_cast<uint64_t*>(d.flags + 8); }
__global__ void hybrid_combine(const float* outputs, const float* probabilities, const int32_t* selected,
                               const MoeGroup* groups, ExpertGemv plan, float* result, uint64_t* clock) { SD_KERNEL_ACQUIRE;
    const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (column == 0 && blockIdx.y == 0) clock[3] += await_clock() - clock[1];
    if (column >= 2560) return;
    outputs += size_t(blockIdx.y) * 11 * 2560; probabilities += size_t(blockIdx.y) * 11; selected += size_t(blockIdx.y) * 10;
    result += size_t(blockIdx.y) * 2560;
    float sum = 0;
    for (size_t route = 0; route < 10; ++route)
        if (expert_source(groups, selected, route, 0, plan)) sum += probabilities[route] * outputs[route * 2560 + column];
    result[column] = sum + probabilities[10] * outputs[10 * 2560 + column];
}
// The GPU exports which routes it evaluates, so host decisions always follow the
// table entries in effect for this token, even while slots are being swapped. Written straight to mapped
// pinned host memory (no memcpy graph node), a row at each hybrid_row_floats.
__global__ void hybrid_export(const int32_t* selected, const float* probabilities, const float* input, size_t input_stride,
                              const MoeGroup* groups, ExpertGemv plan, volatile float* target) { SD_KERNEL_ACQUIRE;
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    selected += size_t(blockIdx.y) * 10; probabilities += size_t(blockIdx.y) * 11; input += size_t(blockIdx.y) * input_stride;
    target += size_t(blockIdx.y) * hybrid_row_floats;
    if (i < 10) {
        target[i] = float(selected[i]);
        target[2592 + i] = groups[selected[i]].weights[0].data ? 1.0f : expert_source(groups, selected, i, 0, plan) ? 2.0f : 0.0f;
    }
    if (i < 11) target[16 + i] = probabilities[i];
    if (i < 2560) target[32 + i] = input[i];
}
// Graph-safe completion signal: each replay advances a device counter and copies
// it after the exported data, so the host waits for exactly this layer's export.
// `timed`: the layer has a start in clock[0] (the span's first layer has none).
__global__ void hybrid_tick(uint32_t* counter, volatile uint32_t* published, uint64_t* clock, bool timed) { SD_KERNEL_ACQUIRE;
    const uint64_t now = await_clock();
    if (timed) clock[2] += now - clock[0];
    clock[1] = now;
    __threadfence_system();
    *published = ++*counter;
    __threadfence_system();
}
__global__ void hybrid_finish(float* hidden, const float* mixed, size_t mixed_stride, const float* gpu, const float* cpu, size_t cpu_stride,
                              uint64_t* clock) { SD_KERNEL_ACQUIRE;
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i == 0 && blockIdx.y == 0) clock[0] = await_clock();
    hidden += size_t(blockIdx.y) * 10240; mixed += size_t(blockIdx.y) * mixed_stride; gpu += size_t(blockIdx.y) * 2560;
    cpu += size_t(blockIdx.y) * cpu_stride;
    if (i < 10240) hidden[i] += mixed[2560 + i / 2560] * (gpu[i % 2560] + cpu[i % 2560]);
}
// Graph-safe wait for the host's CPU result of the previous layer (one block): each
// replay advances the awaited count in pinned memory and spins until the host has
// published as many results, optionally copying the result to device memory. After
// `timeout` ticks (~1 s under Windows' 2 s kernel watchdog, 10 s elsewhere) it gives up
// and marks the failure for the host instead of hanging the device; the layer's output is
// then void.
__global__ void hybrid_await(volatile uint32_t* counters, const volatile float* result, float* target, uint64_t timeout,
                             volatile uint32_t* device_failed) { SD_KERNEL_ACQUIRE;
    __shared__ uint32_t ready;
    if (threadIdx.x == 0) {
        const uint32_t expected = counters[hybrid_awaited_slot] + 1;
        counters[hybrid_awaited_slot] = expected;
        const uint64_t start = await_clock();
        // The healthy entry reads only VRAM for failure state, not mapped host memory.
        ready = *device_failed == 0;
        uint32_t spin = 0;
        while (ready && counters[hybrid_published_slot] != expected) {
            // Host fault injection can cancel this wait after entry, even when
            // unwinding has published beyond expected. Avoid reading it every spin.
            if (((++spin & 255) == 0 && counters[hybrid_await_failed_slot]) || await_clock() - start > timeout) {
                ready = 0; break;
            }
#if defined(__HIPCC__)
            __builtin_amdgcn_s_sleep(2);
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
        }
        if (!ready) {
            *device_failed = 1;
            // A later span may have cleared the host word in its constructor.
            // Re-notify even when the result was already published at entry.
            counters[hybrid_await_failed_slot] = 1;
        }
        __threadfence_system();
    }
    __syncthreads();
#if defined(__HIPCC__)
    // Fine-grained HIP exchange memory can be read directly by hybrid_finish;
    // a null target requests the wait alone, avoiding a separate staging copy.
    if (target)
        for (size_t i = threadIdx.x; i < 2560; i += blockDim.x) target[i] = ready ? result[i] : 0.0f;
#else
    // Uncached 16-byte reads, all in flight at once (volatile scalar reads serialize).
    const float4* source = reinterpret_cast<const float4*>(const_cast<const float*>(result));
    for (size_t i = threadIdx.x; i < 640; i += blockDim.x)
        reinterpret_cast<float4*>(target)[i] = ready ? __ldcv(source + i) : make_float4(0, 0, 0, 0);
#endif
}
#if !defined(__HIPCC__)
// The later rows' CPU results to device memory, once hybrid_await has seen them all published.
__global__ void hybrid_stage(const volatile float* result, float* target) {
    result += size_t(blockIdx.y) * hybrid_row_floats; target += size_t(blockIdx.y) * 2560;
    const float4* source = reinterpret_cast<const float4*>(const_cast<const float*>(result));
    for (size_t i = threadIdx.x; i < 640; i += blockDim.x) reinterpret_cast<float4*>(target)[i] = __ldcv(source + i);
}
#endif
// Host side of hybrid_await for one span. Counts restart from the GPU's awaited count
// (the stream is idle between spans); a span that unwinds still publishes every
// result its launched graphs await, then drains the stream.
struct HybridWaitTimeout : std::runtime_error {
    HybridWaitTimeout() : std::runtime_error("hybrid CPU result was not published in time") {}
};
// Development fault injection, consumed only at the host's checked wait boundary.
std::atomic<uint64_t> test_hybrid_timeouts{0};

struct HybridAwait {
    Device& d;
    volatile uint32_t* counters;
    uint32_t launched = 0, published = 0;
    bool timed_out = false;
    const int unwinding = std::uncaught_exceptions();
    explicit HybridAwait(Device& device)
        : d(device), counters(reinterpret_cast<volatile uint32_t*>(device.exchange + hybrid_sequence_slot)) {
        counters[hybrid_published_slot] = counters[hybrid_awaited_slot];
        counters[hybrid_await_failed_slot] = 0;
    }
    void publish() {
        std::atomic_thread_fence(std::memory_order_release);
        counters[hybrid_published_slot] = counters[hybrid_published_slot] + 1;
        ++published;
    }
    void check_failed() {
        uint64_t left = test_hybrid_timeouts.load();
        while (left && !test_hybrid_timeouts.compare_exchange_weak(left, left - 1)) {}
        if (left) {
            counters[hybrid_await_failed_slot] = 1;
            std::atomic_thread_fence(std::memory_order_release);
        }
        if (counters[hybrid_await_failed_slot]) {
            timed_out = true;
            throw HybridWaitTimeout();
        }
    }
    ~HybridAwait() {
        if (std::uncaught_exceptions() == unwinding) return;
        while (published < launched) publish();
        // A timeout is drained with checked errors by its C ABI catch. The first
        // CUDA/HIP failure must reach that catch instead of being discarded here.
        if (!timed_out) cleanup(hipStreamSynchronize(d.stream));
    }
};

// Single-token GPU part, split so the host can start its experts as soon as the
// routing and activation are exported while the GPU evaluates resident experts.
struct HybridScratch { float *gate_up, *product, *outputs, *probabilities; int32_t* selected; };
// (token_moe's layout)
HybridScratch hybrid_scratch(ResidentMoe& plan, size_t rows) {
    const size_t routes = rows * 11;
    HybridScratch s{};
    s.gate_up = plan.workspace; s.product = s.gate_up + routes * 1280;
    s.outputs = s.product + routes * 640; s.probabilities = s.outputs + routes * 2560;
    s.selected = reinterpret_cast<int32_t*>(s.probabilities + routes) + routes;
    return s;
}
ExpertGemv hybrid_assist(const Device& d, const ResidentMoe& plan) {
    ExpertGemv assist{};
    for (size_t role = 0; role < 3; ++role) { assist.mapped[role] = plan.mapped[role]; assist.bytes[role] = plan.host_bytes[role]; }
    assist.assist = plan.mapped[0] ? d.assist : 0;
    return assist;
}
#if !defined(__HIPCC__)
// A layer's router in one launch (CUDA): the router (experts rows of 2560, BF16 or F32) and the shared expert's gate (one row), each on the
// lanes and in the order gemv() gives it, then the routing and the export in the last block to finish: every block stores its rows, fences
// and counts itself in, and the last block (which resets the count: a graph replays the launch) forms the top-10 and the softmax and
// exports them. The scores land in d.output as the separate kernels leave them, and every value is the one those compute
// (hybrid_route_export_separate: gemv(), resident_route and hybrid_export), as tests/router_gpu.rs checks bit for bit.
struct HybridRouter { Matrix scores, gate; unsigned scores_lanes, gate_lanes, scores_waves; };
// gemv_rows<kind, 1> with a lane's units loaded all at once (ten at a time) and then summed in the order gemv_rows sums them: the loads of a
// row are independent, so the row costs a memory latency or two in place of one for every two units; every dot and every addition is the same.
template<int kind>
__device__ __forceinline__ void hybrid_router_rows(const Matrix& matrix, size_t first_row, unsigned lanes, const float* x, float* y) {
    const unsigned lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = first_row + lane / lanes;
    const size_t units = matrix.cols / GemvFormat<kind>::elements;
    constexpr int batch = 10;
    float sum = 0;
    for (size_t first = sub; first < units; first += size_t(batch) * lanes) {
        float dots[batch];
        #pragma unroll
        for (int j = 0; j < batch; ++j) {
            const size_t unit = first + size_t(j) * lanes;
            dots[j] = 0;
            if (row < matrix.rows && unit < units) {
                float one[1];
                GemvFormat<kind>::template dots<1>(matrix.data + row * matrix.row_bytes, unit, x, 0, one);
                dots[j] = one[0];
            }
        }
        #pragma unroll
        for (int j = 0; j < batch; ++j)
            if (row < matrix.rows && first + size_t(j) * lanes < units) sum += dots[j];
    }
    for (unsigned offset = lanes / 2; offset; offset /= 2) sum += __shfl_xor(sum, offset, 32);
    if (sub == 0 && row < matrix.rows) gemv_store(matrix, row, 0, sum, y, 0, GemvResidual{});
}
__device__ __forceinline__ void hybrid_router_wave(const Matrix& matrix, size_t first_row, unsigned lanes, const float* x, float* y) {
    if (matrix.kind == 30) hybrid_router_rows<30>(matrix, first_row, lanes, x, y);
    else hybrid_router_rows<0>(matrix, first_row, lanes, x, y);
}
// blockIdx.y a row; 8 waves a block, the router's waves and then the gate's. The last block to finish takes the row's top-10 as resident_route does,
// by rank in place of rounds: a wave's 64 scores each count the scores of the wave that beat them (higher, or equal and a lower expert) and the
// ten with fewer than ten are its candidates, in order; the 80 candidates count each other in the same way. Experts are ordered totally (no two
// alike), so the ten and their order are the ones resident_route's rounds find; then the same softmax (every exp, the sum in order, every division).
__global__ void __launch_bounds__(256) hybrid_router(HybridRouter router, const float* input, size_t input_stride, float* scores,
                                                     int32_t* selected, float* probabilities, const MoeGroup* groups, ExpertGemv plan,
                                                     volatile float* target, uint32_t* counters) { SD_KERNEL_ACQUIRE;
    __shared__ uint32_t last;
    __shared__ float candidate_value[80], top_value[10], weight[10];
    __shared__ int candidate_index[80], below[3][80], top_index[10];
    const size_t row = blockIdx.y, experts = router.scores.rows, stride = experts + 1;
    const size_t wave = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
    const float* x = input + row * input_stride;
    float* y = scores + row * stride;
    // (candidates no score takes, as resident_route's masked ones)
    if (threadIdx.x < 80) { candidate_value[threadIdx.x] = -INFINITY; candidate_index[threadIdx.x] = 0x7fffffff; }
    if (threadIdx.x < 10) { top_value[threadIdx.x] = -INFINITY; top_index[threadIdx.x] = 0x7fffffff; }
    if (wave < router.scores_waves) hybrid_router_wave(router.scores, wave * (32 / router.scores_lanes), router.scores_lanes, x, y);
    else hybrid_router_wave(router.gate, (wave - router.scores_waves) * (32 / router.gate_lanes), router.gate_lanes, x, y);
    __threadfence();
    __syncthreads();
    if (!threadIdx.x) {
        const bool done = atomicAdd(counters + row, 1u) == gridDim.x - 1;
        if (done) { counters[row] = 0; __threadfence(); }
        last = done;
    }
    __syncthreads();
    // (the activation to the host: the grid's first threads, a value each, as hybrid_export did)
    target += row * hybrid_row_floats;
    const auto activation = [&]() {
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < 2560; i += size_t(gridDim.x) * 256) target[32 + i] = x[i];
    };
    if (!last) { activation(); return; }
    const unsigned lane = threadIdx.x % 32, own = threadIdx.x / 32;
    // a wave's 64 scores, two a lane (the scores another block stored: read at L2)
    const int index0 = int(own * 64 + lane), index1 = index0 + 32;
    const float value0 = size_t(index0) < experts ? __ldcg(scores + row * stride + index0) : -INFINITY;
    const float value1 = size_t(index1) < experts ? __ldcg(scores + row * stride + index1) : -INFINITY;
    int rank0 = 0, rank1 = 0;
    #pragma unroll 8
    for (unsigned s = 0; s < 32; ++s) {
        const float other0 = __shfl_sync(0xffffffffu, value0, s), other1 = __shfl_sync(0xffffffffu, value1, s);
        const int at0 = int(own * 64 + s), at1 = at0 + 32;
        rank0 += int(route_better(other0, at0, value0, index0)) + int(route_better(other1, at1, value0, index0));
        rank1 += int(route_better(other0, at0, value1, index1)) + int(route_better(other1, at1, value1, index1));
    }
    if (rank0 < 10) { candidate_value[own * 10 + rank0] = value0; candidate_index[own * 10 + rank0] = index0; }
    if (rank1 < 10) { candidate_value[own * 10 + rank1] = value1; candidate_index[own * 10 + rank1] = index1; }
    __syncthreads();
    // the 80 candidates, each counted by three threads over a third of them
    if (threadIdx.x < 240) {
        const unsigned who = threadIdx.x % 80, part = threadIdx.x / 80;
        const float mine = candidate_value[who]; const int at = candidate_index[who];
        int beat = 0;
        for (unsigned j = part; j < 80; j += 3) beat += int(route_better(candidate_value[j], candidate_index[j], mine, at));
        below[part][who] = beat;
    }
    __syncthreads();
    if (threadIdx.x < 80) {
        const int rank = below[0][threadIdx.x] + below[1][threadIdx.x] + below[2][threadIdx.x];
        if (rank < 10) { top_value[rank] = candidate_value[threadIdx.x]; top_index[rank] = candidate_index[threadIdx.x]; }
    }
    __syncthreads();
    // the softmax over the ten
    if (threadIdx.x < 10) weight[threadIdx.x] = expf(top_value[threadIdx.x] - top_value[0]);
    __syncthreads();
    selected += row * 10; probabilities += row * 11;
    const size_t i = threadIdx.x;
    if (i < 10) {
        float sum = 0;
        for (size_t k = 0; k < 10; ++k) sum += weight[k];
        const int expert = top_index[i] < int(experts) ? top_index[i] : int(i);
        const float probability = weight[i] / sum;
        selected[i] = expert; probabilities[i] = probability;
        target[i] = float(expert); target[16 + i] = probability;
    } else if (i == 10) {
        const float gate = hc_sigmoid(__ldcg(scores + row * stride + experts));
        probabilities[10] = gate; target[26] = gate;
    }
    __syncthreads();
    if (i < 10) target[2592 + i] = groups[selected[i]].weights[0].data ? 1.0f : expert_source(groups, selected, i, 0, plan) ? 2.0f : 0.0f;
    activation();
}
// The launch of hybrid_router for the router and the gate of `plan`; false (nothing launched) where gemv() would not run them as it does
// (other formats or shapes, misaligned input), and the separate kernels do.
bool hybrid_router_launch(Device& d, ResidentMoe& plan, const HybridScratch& s, const float* input, size_t rows, size_t input_stride) {
    const size_t experts = plan.experts;
    HybridRouter router{plan.weights[(experts + 1) * 3], plan.weights[(experts + 1) * 3 + 1], 0, 0, 0};
    router.gate.output_offset = experts;
    const Matrix& scores = router.scores;
    const Matrix& gate = router.gate;
    if ((scores.kind != 30 && scores.kind != 0) || (gate.kind != 30 && gate.kind != 0) || scores.rows != experts || experts > 512 || gate.rows != 1 ||
        scores.cols != 2560 || gate.cols != 2560 || scores.input_offset || gate.input_offset || scores.output_offset ||
        !rows || rows > hybrid_rows || input_stride % 4)
        return false;
    router.scores_lanes = gemv_q8_lanes(d, scores, gemv_lanes(scores, input));
    router.gate_lanes = gemv_q8_lanes(d, gate, gemv_lanes(gate, input));
    if (!router.scores_lanes || !router.gate_lanes) return false;
    router.scores_waves = unsigned(gemv_waves(scores, router.scores_lanes));
    const size_t waves = router.scores_waves + gemv_waves(gate, router.gate_lanes);
    hybrid_router<<<dim3(unsigned((waves + 7) / 8), unsigned(rows)), 256, 0, d.stream>>>(
        router, input, input_stride, d.output, s.selected, s.probabilities, static_cast<const MoeGroup*>(plan.table), hybrid_assist(d, plan),
        d.exchange_device, d.flags + hybrid_router_counters);
    check(hipGetLastError());
    return true;
}
#endif
// The rows' routes and activations to the host: every row in one launch of each kernel, as moe_tokens_experts.
void hybrid_route_export_separate(Device& d, ResidentMoe& plan, const float* input, size_t rows = 1, size_t input_stride = 2564) {
    const size_t experts = plan.experts;
    const HybridScratch s = hybrid_scratch(plan, rows);
    Matrix projections[2] = {plan.weights[(experts + 1) * 3], plan.weights[(experts + 1) * 3 + 1]};
    projections[1].output_offset = experts;
    gemv(d, projections, 2, input, d.output, rows, input_stride, experts + 1);
    resident_route<<<unsigned(rows), 256, 0, d.stream>>>(d.output, d.output + experts, experts, s.selected, s.probabilities, experts + 1, experts + 1);
    hybrid_export<<<dim3(10, unsigned(rows)), 256, 0, d.stream>>>(s.selected, s.probabilities, input, input_stride,
                                                                 static_cast<const MoeGroup*>(plan.table), hybrid_assist(d, plan),
                                                                 d.exchange_device);
    check(hipGetLastError());
}
// (one kernel where the router's shapes allow it: CUDA's hybrid_router, else the three above)
void hybrid_route_export(Device& d, ResidentMoe& plan, const float* input, size_t rows = 1, size_t input_stride = 2564) {
#if !defined(__HIPCC__)
    if (hybrid_router_launch(d, plan, hybrid_scratch(plan, rows), input, rows, input_stride)) return;
#endif
    hybrid_route_export_separate(d, plan, input, rows, input_stride);
}
#if !defined(__HIPCC__)
// The experts a layer's GPU takes over the bus, copied into VRAM's staging slots by every thread at once. A kernel reads host
// memory as fast as it has loads in flight: 43 GB/s with all of these, 7 with few (a GEMV's lanes are taken to be the latter: unmeasured).
__global__ void hybrid_stage_assist(const MoeGroup* groups, const int32_t* selected, ExpertGemv plan) {
    const size_t step = size_t(gridDim.x) * blockDim.x, thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t absent = 0, slot = 0;
    selected += size_t(blockIdx.y) * 10; plan.stage += size_t(blockIdx.y) * plan.assist * plan.slot_bytes;
    for (size_t r = 0; r < 10; ++r) absent += groups[selected[r]].weights[0].data == nullptr;
    const uint32_t share = min(plan.assist, (5 * absent + 4) / 12);   // (expert_source's)
    for (size_t r = 0; r < 10 && slot < share; ++r) {
        if (groups[selected[r]].weights[0].data) continue;
        const uint8_t* to = plan.stage + slot * plan.slot_bytes;
        for (size_t role = 0; role < 3; ++role) {
            const uint4* from = reinterpret_cast<const uint4*>(plan.mapped[role] + size_t(selected[r]) * plan.bytes[role]);
            uint4* into = reinterpret_cast<uint4*>(const_cast<uint8_t*>(to));
            for (size_t i = thread; i < plan.bytes[role] / 16; i += step) into[i] = from[i];
            to += plan.bytes[role];
        }
        ++slot;
    }
}
#endif
// The experts the GPU evaluates of the rows hybrid_route_export routed, the rows side by side.
void hybrid_resident_experts(Device& d, ResidentMoe& plan, const float* input, float* partial, size_t rows = 1, size_t input_stride = 2564) {
    const HybridScratch s = hybrid_scratch(plan, rows);
    const size_t routes = rows * 11;
    auto* groups = static_cast<MoeGroup*>(plan.table);
    const uint8_t* stage = nullptr;
#if !defined(__HIPCC__)
    if (plan.mapped[0] && d.assist) {
        stage = d.slots.base + d.slots.arena;
        ExpertGemv copy = hybrid_assist(d, plan);
        copy.stage = stage; copy.slot_bytes = d.slots.bytes;
        hybrid_stage_assist<<<dim3(64, unsigned(rows)), 256, 0, d.stream>>>(groups, s.selected, copy);
    }
#endif
    resident_expert_launch(d, plan, groups, s.selected, false, input, s.gate_up, rows, input_stride, 11 * 1280, false, stage, d.slots.bytes);
    moe_product<<<unsigned((routes * 640 + 255) / 256), 256, 0, d.stream>>>(s.gate_up, routes, s.product);
    resident_expert_launch(d, plan, groups, s.selected, true, s.product, s.outputs, rows, 11 * 640, 11 * 2560, false, stage, d.slots.bytes);
    hybrid_combine<<<dim3(10, unsigned(rows)), 256, 0, d.stream>>>(s.outputs, s.probabilities, s.selected, groups, hybrid_assist(d, plan), partial,
                                                                    hybrid_clock(d));
    check(hipGetLastError());
}

// Host half of a hybrid layer: CPU experts for absent routes, then the residual update.
extern "C" int sd_cpu_experts_run_rows(void*, const int*, uint32_t, const uint32_t*, const uint8_t* const*, const uint8_t* const*,
                                       const uint8_t* const*, const float*, uint64_t, uint64_t, const float* const*, float* const*);
extern "C" const char* sd_cpu_error();
extern "C" int sd_cpu_read_ahead(void*, const uint8_t* const*, const uint64_t*, uint32_t);
// `rows` rows' exports (a decode step's one, a verification's tokens) at exported[r], their CPU experts' sums to result[r]: the
// rows' CPU routes in one call of the CPU engine, so an expert that rows share is read once and the engine's phases hand over once.
void hybrid_host_experts_rows(Device& d, ResidentMoe& plan, size_t rows, const float* const* exported, float* const* result) {
    if (rows > 16) throw std::runtime_error("invalid hybrid rows");
    BankReading reading(d);   // (the CPU's workers read the banks' bytes in here: sd_hybrid_repoint waits)
    const uint8_t* gate[160]; const uint8_t* up[160]; const uint8_t* down[160]; float weights[160];
    uint32_t counts[16];
    size_t count = 0;
    const uint8_t* ranges[480]; uint64_t sizes[480];
    uint32_t hinted = 0;
    for (size_t r = 0; r < rows; ++r) {
        if (d.hint_experts) ++plan.host_token;   // (a token a row, as when each row's routes were a call of their own)
        const float* e = exported[r];
        size_t selected[10];
        const size_t before = count;
        for (size_t route = 0; route < 10; ++route) {
            const float raw = e[route];
            if (!(raw >= 0 && raw < float(plan.experts))) throw std::runtime_error("invalid hybrid expert route");
            const size_t expert = size_t(raw);
            selected[route] = expert;
            // (the counts a later start warms up from: where the route went, an assisted one to the host's side)
            ++(e[2592 + route] == 1.0f ? plan.routed_gpu : plan.routed_host)[expert];
            if (e[2592 + route] != 0) { ++(e[2592 + route] > 1.5f ? d.assisted_experts : d.gpu_experts); continue; }
            // (speed first: a CPU route below the floor's weight is left out, sd_hybrid_route_floor)
            if (e[16 + route] < d.route_floor) { ++d.dropped_experts; continue; }
            gate[count] = plan.host[0] + expert * plan.host_bytes[0];
            up[count] = plan.host[1] + expert * plan.host_bytes[1];
            down[count] = plan.host[2] + expert * plan.host_bytes[2];
            weights[count++] = e[16 + route];
            // (a layer held as page-locked copies has nothing to read ahead: the copies are in memory)
            if (d.hint_experts && !plan.host_copy) {
                // Only experts unread for 64 tokens: 15% of the routes, and 98-100% of the misses of an LRU of 15 to 27 GiB in a
                // 5592-token trace. Hinting every route costs 3% here and 5.4 ms a token on Windows with the pages in memory.
                const bool old = plan.host_token - plan.host_read[expert] > 64;
                plan.host_read[expert] = plan.host_token;
                if (old) {
                    for (size_t role = 0; role < 3; ++role) {
                        ranges[hinted + role] = role == 0 ? gate[count - 1] : role == 1 ? up[count - 1] : down[count - 1];
                        sizes[hinted + role] = plan.host_bytes[role];
                    }
                    hinted += 3;
                }
            }
        }
        counts[r] = uint32_t(count - before);
        hybrid_observe_token(plan, selected);
    }
    if (count) {
        if (!d.cpu) throw std::runtime_error("hybrid experts need an attached CPU expert engine");
        if (hinted && sd_cpu_read_ahead(d.cpu, ranges, sizes, hinted))
            throw std::runtime_error(std::string("Straydog CPU experts: ") + sd_cpu_error());
        const int kinds[3] = {plan.weights[0].kind, plan.weights[1].kind, plan.weights[2].kind};
        const float* xs[16];
        for (size_t r = 0; r < rows; ++r) xs[r] = exported[r] + 32;
        if (sd_cpu_experts_run_rows(d.cpu, kinds, uint32_t(rows), counts, gate, up, down, weights, 2560, 640, xs, result))
            throw std::runtime_error(std::string("Straydog CPU experts: ") + sd_cpu_error());
        d.cpu_experts += count;
    } else for (size_t r = 0; r < rows; ++r) std::fill(result[r], result[r] + 2560, 0.0f);
}
void hybrid_host_experts(Device& d, ResidentMoe& plan, const float* exported, float* result) {
    hybrid_host_experts_rows(d, plan, 1, &exported, &result);
}

