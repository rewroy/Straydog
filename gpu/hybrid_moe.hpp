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
    const ExpertRow row{plan.stage, plan.assign ? plan.assign + size_t(blockIdx.y) * 10 : nullptr};
    float sum = 0;
    for (size_t route = 0; route < 10; ++route)
        if (expert_source(groups, selected, route, 0, plan, row)) sum += probabilities[route] * outputs[route * 2560 + column];
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
#if defined(__HIPCC__)
// Graph-safe completion signal: each replay advances a device counter and copies
// it after the exported data, so the host waits for exactly this layer's export.
// `timed`: the layer has a start in clock[0] (the span's first layer has none).
__global__ void hybrid_tick(uint32_t* counter, volatile uint32_t* published, uint64_t* clock, bool timed, volatile int32_t* need,
                            const int32_t* selected, const MoeGroup* groups, uint32_t rows, volatile int32_t* histogram) { SD_KERNEL_ACQUIRE;
    // (HIP's finish reads the result in place, so the host always has a route to wait for)
    (void)selected; (void)groups; (void)rows; (void)histogram;
    *need = 1;
    const uint64_t now = await_clock();
    if (timed) clock[2] += now - clock[0];
    clock[1] = now;
    __threadfence_system();
    *published = ++*counter;
    __threadfence_system();
}
#else
// The layer's staging plan, formed by a block, a thread a route (forty at most): the misses, the first route to name each missed expert and the slots in the
// order of those firsts come from shared memory in three barriers, where one thread walking the routes took a dependent load a route (about 0.4 us each,
// 16 us a layer of four rows). The values are those a walk in route order forms: with `staged`, if the pass misses no more than `cap` distinct experts each
// gets a staging slot, in the order the rows' routes meet it (a later route to it, of any row, the same slot), and the host has no route in the layer; with
// more, none is staged and the CPU takes them all (staging costs the GPU about 95 us an expert, in series, where the CPU computes a layer's misses in
// parallel: it wins with a few, loses with many). The plan goes to `assign` (a slot, or -1, for each route) and to `stage_list` (the experts by slot), and
// the export's mark of each route that has a slot becomes 2 (the router exported 0 for every missed one): the host computes the ones left at 0. Without
// `staged`, whether the host has a route (a missed one) is all that is formed. The layer passes are counted by how many distinct experts they miss (none,
// one, ... five or more: DS41_PROFILE=hc reports them). With `check` (hc) the plan is held against a walk of the routes in order, in the same pass; the
// differences are counted at `bad`.
__device__ void hybrid_plan_block(const int32_t* selected, const MoeGroup* groups, uint32_t rows, uint32_t cap, bool staged, int32_t* assign, int32_t* stage_list,
                                  int32_t* stage_count, volatile float* target, volatile int32_t* need, volatile int32_t* histogram, bool check,
                                  volatile int32_t* bad, const float* activation, size_t activation_stride) {
    __shared__ int32_t shared_expert[40], shared_first[40], shared_rank[40];
    const uint32_t routes = rows * 10, route = threadIdx.x;
    int32_t expert = -1;
    if (route < routes) {
        const int32_t selected_expert = selected[route];
        if (!groups[selected_expert].weights[0].data) expert = selected_expert;
        shared_expert[route] = expert;
    }
    __syncthreads();
    // (a missed route is a first when no route before it names its expert)
    int32_t first_route = -1, first = 0;
    if (route < routes && expert >= 0) {
        first_route = int32_t(route);
        for (uint32_t j = 0; j < route; ++j)
            if (shared_expert[j] == expert) { first_route = int32_t(j); break; }
        first = first_route == int32_t(route);
    }
    if (route < routes) shared_first[route] = first;
    __syncthreads();
    uint32_t used = 0, rank = 0;
    for (uint32_t j = 0; j < routes; ++j) {
        if (j == route) rank = used;
        used += shared_first[j];
    }
    if (route < routes && first) shared_rank[route] = int32_t(rank);
    __syncthreads();
    const bool stage = staged && used <= cap;
    if (route < routes && staged) {
        const int32_t slot = stage && expert >= 0 ? shared_rank[first_route] : -1;
        assign[route] = slot;
        if (slot >= 0) target[size_t(route / 10) * hybrid_row_floats + 2592 + route % 10] = 2.0f;
        if (stage && first) stage_list[rank] = expert;
    }
    // (the activation, for the host's own routes: where it has none it reads nothing of it)
    const bool host_routes = used && !stage;
    if (activation && host_routes)
        for (uint32_t i = threadIdx.x; i < rows * 640; i += blockDim.x) {
            const uint32_t row = i / 640, k = i % 640;
            reinterpret_cast<float4*>(const_cast<float*>(target) + size_t(row) * hybrid_row_floats + 32)[k] =
                reinterpret_cast<const float4*>(activation + size_t(row) * activation_stride)[k];
        }
    __threadfence();
    __syncthreads();
    if (threadIdx.x) return;
    histogram[used < 5 ? used : 5] += 1;
    if (staged) *stage_count = stage ? int32_t(used) : 0;
    *need = (used && !stage) ? 1 : 0;   // (the CPU has a route in this layer: the next layer's await waits for its result)
    if (check) {
        int32_t list[40];
        uint32_t listed = 0;
        for (uint32_t i = 0; i < routes; ++i) {
            const int32_t e = selected[i];
            if (groups[e].weights[0].data) continue;
            bool seen = false;
            for (uint32_t j = 0; j < listed && !seen; ++j) seen = list[j] == e;
            if (!seen) list[listed++] = e;
        }
        const bool whole = staged && listed <= cap;
        uint32_t differing = listed != used;
        if (staged) {
            for (uint32_t i = 0; i < routes; ++i) {
                const int32_t e = selected[i];
                int32_t slot = -1;
                if (whole && !groups[e].weights[0].data)
                    for (uint32_t j = 0; j < listed; ++j)
                        if (list[j] == e) { slot = int32_t(j); break; }
                differing += *reinterpret_cast<volatile int32_t*>(assign + i) != slot;
                differing += (target[size_t(i / 10) * hybrid_row_floats + 2592 + i % 10] == 2.0f) != (slot >= 0);
            }
            for (uint32_t j = 0; whole && j < listed; ++j) differing += *reinterpret_cast<volatile int32_t*>(stage_list + j) != list[j];
            differing += *reinterpret_cast<volatile int32_t*>(stage_count) != int32_t(whole ? listed : 0);
        }
        differing += *need != int32_t((listed && !whole) ? 1 : 0);
        if (activation && host_routes)
            for (uint32_t i = 0; i < rows * 2560; ++i)
                differing += target[size_t(i / 2560) * hybrid_row_floats + 32 + i % 2560] != activation[size_t(i / 2560) * activation_stride + i % 2560];
        if (differing) atomicAdd(const_cast<int32_t*>(bad), int32_t(differing));
    }
}
// Graph-safe completion signal: each replay advances a device counter and copies it after the exported data, so the host waits for exactly this layer's
// export. `timed`: the layer has a start in clock[0] (the span's first layer has none). The count is published after one system fence that orders the
// export before it; the count itself needs none after it (the kernel's end sends it).
__device__ __forceinline__ void hybrid_publish_count(uint32_t* counter, volatile uint32_t* published, uint64_t* clock, bool timed) {
    const uint64_t now = await_clock();
    if (timed) clock[2] += now - clock[0];
    clock[1] = now;
    __threadfence_system();
    *published = ++*counter;
}
__global__ void hybrid_tick(uint32_t* counter, volatile uint32_t* published, uint64_t* clock, bool timed, volatile int32_t* need,
                            const int32_t* selected, const MoeGroup* groups, uint32_t rows, volatile int32_t* histogram, bool check, volatile int32_t* bad,
                            volatile float* target, const float* activation, size_t activation_stride) {
    SD_KERNEL_ACQUIRE;
    hybrid_plan_block(selected, groups, rows, 0, false, nullptr, nullptr, nullptr, target, need, histogram, check, bad, activation, activation_stride);
    if (threadIdx.x) return;
    hybrid_publish_count(counter, published, clock, timed);
}
__global__ void hybrid_tick_plan(uint32_t* counter, volatile uint32_t* published, uint64_t* clock, bool timed, const int32_t* selected,
                                 const MoeGroup* groups, uint32_t rows, uint32_t cap, int32_t* assign, int32_t* stage_list,
                                 int32_t* stage_count, volatile float* target, volatile int32_t* need, volatile int32_t* histogram, bool check,
                                 volatile int32_t* bad, const float* activation, size_t activation_stride) { SD_KERNEL_ACQUIRE;
    hybrid_plan_block(selected, groups, rows, cap, true, assign, stage_list, stage_count, target, need, histogram, check, bad, activation, activation_stride);
    if (threadIdx.x) return;
    hybrid_publish_count(counter, published, clock, timed);
}
#endif
__global__ void hybrid_finish(float* hidden, const float* mixed, size_t mixed_stride, const float* gpu, const float* cpu, size_t cpu_stride,
                              uint64_t* clock) { SD_KERNEL_ACQUIRE;
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i == 0 && blockIdx.y == 0) clock[0] = await_clock();
    hidden += size_t(blockIdx.y) * 10240; mixed += size_t(blockIdx.y) * mixed_stride; gpu += size_t(blockIdx.y) * 2560;
    cpu += size_t(blockIdx.y) * cpu_stride;
    if (i < 10240) hidden[i] += mixed[2560 + i / 2560] * (gpu[i % 2560] + cpu[i % 2560]);
}
// Graph-safe wait for the host's CPU result of the previous layer (one block): each
// replay advances the awaited count and spins until the host has
// published as many results, optionally copying the result to device memory. After
// `timeout` ticks (~1 s under Windows' 2 s kernel watchdog, 10 s elsewhere) it gives up
// and marks the failure for the host instead of hanging the device; the layer's output is
// then void.
__global__ void hybrid_await(volatile uint32_t* counters, const volatile float* result, float* target, uint64_t timeout,
                             const volatile int32_t* need, volatile uint32_t* awaited, volatile uint32_t* device_failed) { SD_KERNEL_ACQUIRE;
    __shared__ uint32_t ready, none;
    if (threadIdx.x == 0) {
        ready = *device_failed == 0;
#if defined(__HIPCC__)
        (void)awaited;
        const uint32_t expected = counters[hybrid_awaited_slot] + 1;
        counters[hybrid_awaited_slot] = expected;
        const uint64_t start = await_clock();
        // A layer whose routes the GPU holds or stages has no CPU result: its next layer does not wait for the host's work on it, only
        // for the host to have read the export of the layer before it, whose place the next export takes (hybrid_slot_floats).
        none = *need == 0;
        const uint32_t wanted = none ? expected - 1 : expected;
        uint32_t spin = 0;
        while (ready && int32_t(counters[hybrid_published_slot] - wanted) < 0) {
            // The host can cancel a wait after entry; keep its clock deadline checked every spin.
            if (((++spin & 255) == 0 && counters[hybrid_await_failed_slot]) || await_clock() - start > timeout) {
                ready = 0; break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        __threadfence_system();
#else
        // The awaits of the span are counted on the device (the host counts its publishes from the span's start). A layer with no CPU result is waited for
        // by nothing: its export has a place of its own (hybrid_slot_of), and a healthy wait touches no host memory.
        const uint32_t expected = *awaited + 1;
        *awaited = expected;
        none = *need == 0;
        if (!none) {
            const uint64_t start = await_clock();
            uint32_t spin = 0;
            while (ready && int32_t(counters[hybrid_published_slot] - expected) < 0) {
                if (((++spin & 255) == 0 && counters[hybrid_await_failed_slot]) || await_clock() - start > timeout) {
                    ready = 0; break;
                }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
                __nanosleep(256);
#endif
            }
            __threadfence_system();
        }
#endif
        if (!ready) {
            // The host may have cleared its flag at a span boundary; an earlier device failure must be reported again, even with no CPU route.
            *device_failed = 1;
            counters[hybrid_await_failed_slot] = 1;
            __threadfence_system();
        }
    }
    __syncthreads();
#if defined(__HIPCC__)
    // Fine-grained HIP exchange memory can be read directly by hybrid_finish;
    // a null target requests the wait alone, avoiding a separate staging copy.
    if (target)
        for (size_t i = threadIdx.x; i < 2560; i += blockDim.x) target[i] = ready && !none ? result[i] : 0.0f;
#else
    // Uncached 16-byte reads, all in flight at once (volatile scalar reads serialize).
    const float4* source = reinterpret_cast<const float4*>(const_cast<const float*>(result));
    for (size_t i = threadIdx.x; i < 640; i += blockDim.x)
        reinterpret_cast<float4*>(target)[i] = ready && !none ? __ldcv(source + i) : make_float4(0, 0, 0, 0);
#endif
}
#if !defined(__HIPCC__)
// The later rows' CPU results to device memory, once hybrid_await has seen them all published.
__global__ void hybrid_stage(const volatile float* result, float* target, const volatile int32_t* need) {
    result += size_t(blockIdx.y) * hybrid_row_floats; target += size_t(blockIdx.y) * 2560;
    const float4* source = reinterpret_cast<const float4*>(const_cast<const float*>(result));
    const bool none = *need == 0;
    for (size_t i = threadIdx.x; i < 640; i += blockDim.x) reinterpret_cast<float4*>(target)[i] = none ? make_float4(0, 0, 0, 0) : __ldcv(source + i);
}
#endif
inline int32_t* hybrid_plan_room(const Device& d) { return reinterpret_cast<int32_t*>(d.cpu_partial + hybrid_rows * 2560); }
constexpr size_t hybrid_plan_awaited = 90;   // CUDA: the span's awaits so far, uint32 (hybrid_await)
inline uint32_t* hybrid_awaited(const Device& d) { return reinterpret_cast<uint32_t*>(hybrid_plan_room(d) + hybrid_plan_awaited); }
#if defined(__HIPCC__)
#define HYBRID_AWAITED(d) (static_cast<volatile uint32_t*>(nullptr))
#else
#define HYBRID_AWAITED(d) (reinterpret_cast<volatile uint32_t*>(hybrid_awaited(d)))
#endif
// Host side of hybrid_await for one span. HIP resumes from its mapped awaited count;
// CUDA resets its device count at the idle span boundary. A span that unwinds still publishes every
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
#if defined(__HIPCC__)
        counters[hybrid_published_slot] = counters[hybrid_awaited_slot];
#else
        // (the span's awaits and publishes are both counted from 0; the stream is idle, the reset is stream-ordered before the first await)
        counters[hybrid_published_slot] = 0;
        check(hipMemsetAsync(hybrid_awaited(device), 0, 4, device.stream));
#endif
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
        if (!timed_out) {
            const hipError_t drained=hipStreamSynchronize(d.stream);
            if (drained!=hipSuccess) d.span_drain_error=drained;
            cleanup(drained);
        }
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
// Whether this layer's missed routes go to the staging plan (hybrid_tick_plan), CUDA's: its banks are mapped, the level is above 0 and the staging slots
// and the plan's room exist. The plan is in the tail of d.cpu_partial after the rows' results: assign[40], stage_list[32], stage_count.
inline bool hybrid_staged(const Device& d, const ResidentMoe& plan) {
#if !defined(__HIPCC__)
    return plan.mapped[0] && hybrid_level(d, plan) && d.slots.base && d.slots.stage >= 2 && d.cpu_partial;
#else
    (void)d; (void)plan;
    return false;
#endif
}
constexpr size_t hybrid_plan_list = 40, hybrid_plan_count = 72, hybrid_plan_need = 73;
inline int32_t* hybrid_need(const Device& d) { return hybrid_plan_room(d) + hybrid_plan_need; }
// DS41_PROFILE=hc counts: the layer passes by how many distinct experts they miss (none ... five or more), the MTP layer's apart from the model's (the room's
// ints 74 ... 79 and 84 ... 89; 80 is the count of staged bytes differing from the bank's).
constexpr size_t hybrid_plan_histogram = 74, hybrid_plan_draft_histogram = 84, hybrid_plan_bad = 80, hybrid_plan_tick_bad = 81, hybrid_plan_overlap_bad = 82;
inline int32_t* hybrid_histogram(const Device& d, const ResidentMoe& plan) {
    return hybrid_plan_room(d) + (plan.drafter ? hybrid_plan_draft_histogram : hybrid_plan_histogram);
}
// `exporting`: for the router's mark of each route (a missed one is 0 then: hybrid_tick_plan makes the staged ones 2).
ExpertGemv hybrid_assist(const Device& d, const ResidentMoe& plan, bool exporting = false) {
    ExpertGemv assist{};
    for (size_t role = 0; role < 3; ++role) { assist.mapped[role] = plan.mapped[role]; assist.bytes[role] = plan.host_bytes[role]; }
    assist.assist = plan.mapped[0] ? hybrid_level(d, plan) : 0;
    if (hybrid_staged(d, plan)) {
        if (exporting) assist.assist = 0;
        else {
            assist.assign = hybrid_plan_room(d);
            assist.stage = d.slots.base + d.slots.arena; assist.slot_bytes = d.slots.bytes;
        }
    }
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
                                                     volatile float* target, uint32_t* counters, bool defer_activation) { SD_KERNEL_ACQUIRE;
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
    if (!last) { if (!defer_activation) activation(); return; }
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
    if (!defer_activation) activation();
}
// The launch of hybrid_router for the router and the gate of `plan`; false (nothing launched) where gemv() would not run them as it does
// (other formats or shapes, misaligned input), and the separate kernels do.
bool hybrid_router_launch(Device& d, ResidentMoe& plan, const HybridScratch& s, const float* input, size_t rows, size_t input_stride, size_t slot = 0,
                          bool defer_activation = false) {
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
        router, input, input_stride, d.output, s.selected, s.probabilities, static_cast<const MoeGroup*>(plan.table), hybrid_assist(d, plan, true),
        d.exchange_device + slot * hybrid_slot_floats, d.flags + hybrid_router_counters, defer_activation);
    check(hipGetLastError());
    return true;
}
#endif
// The rows' routes and activations to the host: every row in one launch of each kernel, as moe_tokens_experts.
void hybrid_route_export_separate(Device& d, ResidentMoe& plan, const float* input, size_t rows = 1, size_t input_stride = 2564, size_t slot = 0) {
    const size_t experts = plan.experts;
    const HybridScratch s = hybrid_scratch(plan, rows);
    Matrix projections[2] = {plan.weights[(experts + 1) * 3], plan.weights[(experts + 1) * 3 + 1]};
    projections[1].output_offset = experts;
    gemv(d, projections, 2, input, d.output, rows, input_stride, experts + 1);
    resident_route<<<unsigned(rows), 256, 0, d.stream>>>(d.output, d.output + experts, experts, s.selected, s.probabilities, experts + 1, experts + 1);
    hybrid_export<<<dim3(10, unsigned(rows)), 256, 0, d.stream>>>(s.selected, s.probabilities, input, input_stride,
                                                                 static_cast<const MoeGroup*>(plan.table), hybrid_assist(d, plan, true),
                                                                 d.exchange_device + slot * hybrid_slot_floats);
    check(hipGetLastError());
}
// (one kernel where the router's shapes allow it: CUDA's hybrid_router, else the three above)
// CUDA: with the router's one kernel the activation is not exported here: it goes to the host with the plan, and only where the host has a route (true is
// returned: hybrid_publish is to be given `input`). The host reads the activation for its own routes and for nothing else, and every byte exported is a
// byte the tick's fence has to see across the bus before the count is published.
bool hybrid_route_export(Device& d, ResidentMoe& plan, const float* input, size_t rows = 1, size_t input_stride = 2564, size_t slot = 0) {
#if !defined(__HIPCC__)
    if (hybrid_router_launch(d, plan, hybrid_scratch(plan, rows), input, rows, input_stride, slot, true)) return true;
#endif
    hybrid_route_export_separate(d, plan, input, rows, input_stride, slot);
    return false;
}
#if !defined(__HIPCC__)
// The experts a layer's GPU takes over the bus, copied into VRAM's staging slots by every thread at once: the experts of the plan (hybrid_tick_plan),
// each once. A kernel reads host memory as fast as it has loads in flight: 43 GB/s with all of these, 7 with few (a GEMV's lanes are taken to be the
// latter: unmeasured).
__global__ void hybrid_stage_assist(ExpertGemv plan, const int32_t* stage_list, const int32_t* stage_count) {
    const size_t step = size_t(gridDim.x) * blockDim.x, thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t count = *stage_count;
    for (int32_t slot = 0; slot < count; ++slot) {
        uint8_t* to = const_cast<uint8_t*>(plan.stage) + size_t(slot) * plan.slot_bytes;
        for (size_t role = 0; role < 3; ++role) {
            const uint4* from = reinterpret_cast<const uint4*>(plan.mapped[role] + size_t(stage_list[slot]) * plan.bytes[role]);
            uint4* into = reinterpret_cast<uint4*>(to);
            for (size_t i = thread; i < plan.bytes[role] / 16; i += step) into[i] = from[i];
            to += plan.bytes[role];
        }
    }
}
#endif
// DS41_PROFILE containing "hc": each staged expert's bytes are checked against the bank's (hybrid_stage_verify), and the counts are said when a span
// ends (hybrid_check_report): the differences found, and the layer passes by how many distinct experts they missed.
inline bool hybrid_check() {
    static const bool on = [] { const char* p = std::getenv("DS41_PROFILE"); return p && std::strstr(p, "hc"); }();
    return on;
}
#if !defined(__HIPCC__)
__global__ void hybrid_stage_verify(ExpertGemv plan, const int32_t* stage_list, const int32_t* stage_count, volatile int32_t* bad) {
    const size_t step = size_t(gridDim.x) * blockDim.x, thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t count = *stage_count;
    for (int32_t slot = 0; slot < count; ++slot) {
        const uint8_t* to = plan.stage + size_t(slot) * plan.slot_bytes;
        for (size_t role = 0; role < 3; ++role) {
            const uint4* from = reinterpret_cast<const uint4*>(plan.mapped[role] + size_t(stage_list[slot]) * plan.bytes[role]);
            const uint4* into = reinterpret_cast<const uint4*>(to);
            for (size_t i = thread; i < plan.bytes[role] / 16; i += step) {
                const uint4 a = from[i], b = into[i];
                if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) atomicAdd(const_cast<int32_t*>(bad), 1);
            }
            to += plan.bytes[role];
        }
    }
}
#endif
void hybrid_check_report(Device& d) {
    if (!hybrid_check() || !d.cpu_partial) return;
    // (per device, in the order the devices first report: the layers of one device's plan are the passes it counts)
    // (two devices' spans end on two threads where a read overlaps them: the list is taken under a lock)
    static std::mutex devices_lock;
    static std::vector<std::pair<const void*, uint64_t>> devices;
    size_t which = 0;
    uint64_t spans = 0;
    {
        std::lock_guard<std::mutex> hold(devices_lock);
        while (which < devices.size() && devices[which].first != d.cpu_partial) ++which;
        if (which == devices.size()) devices.emplace_back(d.cpu_partial, 0);
        spans = ++devices[which].second;
    }
    int32_t counts[7] = {}, draft[6] = {}, tick_bad = 0, overlap_bad = 0;
    download(d, counts, hybrid_plan_room(d) + hybrid_plan_histogram, 6 * 4);
    download(d, &counts[6], hybrid_plan_room(d) + hybrid_plan_bad, 4);
    download(d, &tick_bad, hybrid_plan_room(d) + hybrid_plan_tick_bad, 4);
    download(d, &overlap_bad, hybrid_plan_room(d) + hybrid_plan_overlap_bad, 4);
    download(d, draft, hybrid_plan_room(d) + hybrid_plan_draft_histogram, 6 * 4);
    if (spans == 1 || spans % 100 == 0 || counts[6] || tick_bad || overlap_bad)
        std::fprintf(stderr, "Straydog hybrid check: device %zu, %llu spans, staged bytes differing from the bank's: %d, tick plans differing from a walk of the routes in order: %d, overlapped layers' outputs differing from one launch's: %d; layer passes by distinct missed experts 0/1/2/3/4/5+: %d %d %d %d %d %d; the MTP layer's: %d %d %d %d %d %d\n",
                     which, (unsigned long long)spans, counts[6], tick_bad, overlap_bad, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5],
                     draft[0], draft[1], draft[2], draft[3], draft[4], draft[5]);
}
#if !defined(__HIPCC__)
// How a staging layer overlaps its copy with its resident experts (hybrid_resident_experts): 1, the default, beside their gate and up; 2 (DS41_PROFILE
// containing "so2"), beside their down as well, two launches more; 0 ("so0"), none, the copy first in the device stream as before.
inline int hybrid_overlap() {
    static const int level = [] {
        const char* p = std::getenv("DS41_PROFILE");
        return p && std::strstr(p, "so0") ? 0 : p && std::strstr(p, "so2") ? 2 : 1;
    }();
    return level;
}
// DS41_PROFILE=hc: an overlapped layer's outputs against those of one launch for all its routes (the scratch after d.cpu_partial's room), bit for bit, for
// the routes the GPU evaluates and the shared expert.
__global__ void hybrid_check_outputs(const float* outputs, const float* reference, const int32_t* selected, const MoeGroup* groups, ExpertGemv plan,
                                     volatile int32_t* bad) {
    const size_t row = blockIdx.y, column = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (column >= 2560) return;
    selected += row * 10; outputs += row * 11 * 2560; reference += row * 11 * 2560;
    const ExpertRow staging{plan.stage, plan.assign ? plan.assign + row * 10 : nullptr};
    int32_t differing = 0;
    for (size_t route = 0; route < 11; ++route)
        if (route == 10 || expert_source(groups, selected, route, 0, plan, staging))
            differing += __float_as_uint(outputs[route * 2560 + column]) != __float_as_uint(reference[route * 2560 + column]);
    if (differing) atomicAdd(const_cast<int32_t*>(bad), differing);
}
#endif
// The experts the GPU evaluates of the rows hybrid_route_export routed, the rows side by side.
// CUDA, a layer that stages (hybrid_staged): the copy of its staged experts (hybrid_stage_assist) goes to a stream of its own, forked from the device's after the
// tick, and the resident routes' experts (and the shared expert) run beside it; the staged routes' follow once it is joined, so that the layer takes the
// longer of the two and the staged experts' compute, not their sum. A launch's routes are chosen by ExpertGemv::only (1 resident, 2 staged): each route's
// tiles are the same tiles as in one launch for all, and moe_product forms the same products wherever it runs, so the sums are the same.
void hybrid_resident_experts(Device& d, ResidentMoe& plan, const float* input, float* partial, size_t rows = 1, size_t input_stride = 2564) {
    const HybridScratch s = hybrid_scratch(plan, rows);
    const size_t routes = rows * 11;
    auto* groups = static_cast<MoeGroup*>(plan.table);
    const uint8_t* stage = nullptr;
    const int32_t* assign = nullptr;
    int overlap = 0;
#if !defined(__HIPCC__)
    if (hybrid_staged(d, plan)) {
        const ExpertGemv copy = hybrid_assist(d, plan);
        stage = copy.stage; assign = copy.assign;
        overlap = d.stage_stream && d.stage_fork && d.stage_join ? hybrid_overlap() : 0;
        hipStream_t copier = d.stream;
        if (overlap) {
            copier = d.stage_stream;
            check(hipEventRecord(d.stage_fork, d.stream));
            check(hipStreamWaitEvent(d.stage_stream, d.stage_fork, 0));
        }
        hybrid_stage_assist<<<64, 256, 0, copier>>>(copy, assign + hybrid_plan_list, assign + hybrid_plan_count);
        if (hybrid_check())
            hybrid_stage_verify<<<64, 256, 0, copier>>>(copy, assign + hybrid_plan_list, assign + hybrid_plan_count, hybrid_plan_room(d) + hybrid_plan_bad);
        if (overlap) check(hipEventRecord(d.stage_join, d.stage_stream));
    }
#endif
    if (overlap) {
#if !defined(__HIPCC__)
        // beside the copy: the resident routes' gate and up, and with level 2 their products and down
        resident_expert_launch(d, plan, groups, s.selected, false, input, s.gate_up, rows, input_stride, 11 * 1280, false, stage, d.slots.bytes, assign, 1);
        if (overlap >= 2) {
            moe_product<<<unsigned((routes * 640 + 255) / 256), 256, 0, d.stream>>>(s.gate_up, routes, s.product);
            resident_expert_launch(d, plan, groups, s.selected, true, s.product, s.outputs, rows, 11 * 640, 11 * 2560, false, stage, d.slots.bytes, assign, 1);
        }
        check(hipStreamWaitEvent(d.stream, d.stage_join, 0));
        // after it: the staged routes', and the products of all routes (the staged routes' gate and up are in now; the resident routes' products come out as before)
        resident_expert_launch(d, plan, groups, s.selected, false, input, s.gate_up, rows, input_stride, 11 * 1280, false, stage, d.slots.bytes, assign, 2);
        moe_product<<<unsigned((routes * 640 + 255) / 256), 256, 0, d.stream>>>(s.gate_up, routes, s.product);
        resident_expert_launch(d, plan, groups, s.selected, true, s.product, s.outputs, rows, 11 * 640, 11 * 2560, false, stage, d.slots.bytes, assign,
                               overlap >= 2 ? 2 : 0);
        if (hybrid_check() && hybrid_check_scratch_floats()) {
            // (the same layer in one launch for all its routes, on the scratch, and the outputs held against it)
            float* reference = d.cpu_partial + hybrid_partial_floats;
            float* reference_product = reference + routes * 1280;
            float* reference_outputs = reference_product + routes * 640;
            resident_expert_launch(d, plan, groups, s.selected, false, input, reference, rows, input_stride, 11 * 1280, false, stage, d.slots.bytes, assign);
            moe_product<<<unsigned((routes * 640 + 255) / 256), 256, 0, d.stream>>>(reference, routes, reference_product);
            resident_expert_launch(d, plan, groups, s.selected, true, reference_product, reference_outputs, rows, 11 * 640, 11 * 2560, false, stage,
                                   d.slots.bytes, assign);
            hybrid_check_outputs<<<dim3(10, unsigned(rows)), 256, 0, d.stream>>>(s.outputs, reference_outputs, s.selected, groups, hybrid_assist(d, plan),
                                                                                  hybrid_plan_room(d) + hybrid_plan_overlap_bad);
        }
#endif
    } else {
        resident_expert_launch(d, plan, groups, s.selected, false, input, s.gate_up, rows, input_stride, 11 * 1280, false, stage, d.slots.bytes, assign);
        moe_product<<<unsigned((routes * 640 + 255) / 256), 256, 0, d.stream>>>(s.gate_up, routes, s.product);
        resident_expert_launch(d, plan, groups, s.selected, true, s.product, s.outputs, rows, 11 * 640, 11 * 2560, false, stage, d.slots.bytes, assign);
    }
    hybrid_combine<<<dim3(10, unsigned(rows)), 256, 0, d.stream>>>(s.outputs, s.probabilities, s.selected, groups, hybrid_assist(d, plan), partial,
                                                                    hybrid_clock(d));
    check(hipGetLastError());
}

// The layer's export is complete: publish it, with the staging plan where the GPU takes the missed routes.
// `activation` (CUDA: the router's input, rows of `activation_stride`; null: the export has it already) goes to the host here where the host has a route.
void hybrid_publish(Device& d, ResidentMoe& plan, size_t rows, bool finish, size_t slot, const float* activation = nullptr, size_t activation_stride = 0) {
    auto* published = reinterpret_cast<uint32_t*>(d.exchange_device + hybrid_sequence_slot);
#if !defined(__HIPCC__)
    if (hybrid_staged(d, plan)) {
        const HybridScratch s = hybrid_scratch(plan, rows);
        int32_t* room = hybrid_plan_room(d);
        const uint32_t cap = uint32_t(std::min<size_t>(std::min<size_t>(hybrid_level(d, plan), d.slots.stage), 32));
        hybrid_tick_plan<<<1, 256, 0, d.stream>>>(d.flags + 1, published, hybrid_clock(d), finish, s.selected, static_cast<const MoeGroup*>(plan.table),
                                                uint32_t(rows), cap, room, room + hybrid_plan_list, room + hybrid_plan_count, d.exchange_device + slot * hybrid_slot_floats,
                                                hybrid_need(d), hybrid_histogram(d, plan), hybrid_check(), hybrid_plan_room(d) + hybrid_plan_tick_bad, activation, activation_stride);
        return;
    }
#endif
#if defined(__HIPCC__)
    hybrid_tick<<<1, 1, 0, d.stream>>>(d.flags + 1, published, hybrid_clock(d), finish, hybrid_need(d), hybrid_scratch(plan, rows).selected,
                                       static_cast<const MoeGroup*>(plan.table), uint32_t(rows), hybrid_histogram(d, plan));
#else
    hybrid_tick<<<1, 256, 0, d.stream>>>(d.flags + 1, published, hybrid_clock(d), finish, hybrid_need(d), hybrid_scratch(plan, rows).selected,
                                         static_cast<const MoeGroup*>(plan.table), uint32_t(rows), hybrid_histogram(d, plan), hybrid_check(),
                                         hybrid_plan_room(d) + hybrid_plan_tick_bad, d.exchange_device + slot * hybrid_slot_floats, activation, activation_stride);
#endif
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

