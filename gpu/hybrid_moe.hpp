// Single-token hybrid MoE: GPU part and the host CPU-expert step (see hybrid_slots.hpp).
__global__ void hybrid_combine(const float* outputs, const float* probabilities, const int32_t* selected,
                               const MoeGroup* groups, ExpertGemv plan, float* result) {
    const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (column >= 2560) return;
    float sum = 0;
    for (size_t route = 0; route < 10; ++route)
        if (expert_source(groups, selected, route, 0, plan)) sum += probabilities[route] * outputs[route * 2560 + column];
    result[column] = sum + probabilities[10] * outputs[10 * 2560 + column];
}
// The GPU exports which routes it evaluates, so host decisions always follow the
// table entries in effect for this token, even while slots are being swapped.
__global__ void hybrid_export(const int32_t* selected, const float* probabilities, const float* input,
                              const MoeGroup* groups, ExpertGemv plan, float* target) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < 10) {
        target[i] = float(selected[i]);
        target[2592 + i] = groups[selected[i]].weights[0].data ? 1.0f : expert_source(groups, selected, i, 0, plan) ? 2.0f : 0.0f;
    }
    if (i < 11) target[16 + i] = probabilities[i];
    if (i < 2560) target[32 + i] = input[i];
}
// Graph-safe completion signal: each replay advances a device counter and copies
// it after the exported data, so the host waits for exactly this layer's export.
__global__ void hybrid_tick(uint32_t* counter, volatile uint32_t* published) {
    __threadfence_system();
    *published = ++*counter;
    __threadfence_system();
}
// Copy the device export into mapped pinned host memory (no memcpy graph node).
__global__ void hybrid_publish(const float* exported, volatile float* host) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < 2602) host[i] = exported[i];
}
__global__ void hybrid_finish(float* hidden, const float* mixed, const float* gpu, const float* cpu) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < 10240) hidden[i] += mixed[2560 + i / 2560] * (gpu[i % 2560] + cpu[i % 2560]);
}
__device__ inline uint64_t await_clock() {
#if defined(__HIPCC__)
    return wall_clock64();
#else
    uint64_t now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); return now;
#endif
}
// Graph-safe wait for the host's CPU result of the previous layer (one block): each
// replay advances the awaited count in pinned memory and spins until the host has
// published as many results, optionally copying the result to device memory. After
// `timeout` ticks (~1 s under Windows' 2 s kernel watchdog, 10 s elsewhere) it gives up
// and marks the failure for the host instead of hanging the device; the layer's output is
// then void.
__global__ void hybrid_await(volatile uint32_t* counters, const volatile float* result, float* target, uint64_t timeout) {
    __shared__ uint32_t ready;
    if (threadIdx.x == 0) {
        const uint32_t expected = counters[hybrid_awaited_slot] + 1;
        counters[hybrid_awaited_slot] = expected;
        const uint64_t start = await_clock();
        ready = 1;
        while (counters[hybrid_published_slot] != expected) {
            if (await_clock() - start > timeout) { counters[hybrid_await_failed_slot] = 1; ready = 0; break; }
#if defined(__HIPCC__)
            __builtin_amdgcn_s_sleep(2);
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
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
// Host side of hybrid_await for one span. Counts restart from the GPU's awaited count
// (the stream is idle between spans); a span that unwinds still publishes every
// result its launched graphs await, then drains the stream.
struct HybridAwait {
    Device& d;
    volatile uint32_t* counters;
    uint32_t launched = 0, published = 0;
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
    void check_failed() const {
        if (counters[hybrid_await_failed_slot]) throw std::runtime_error("hybrid CPU result was not published in time");
    }
    ~HybridAwait() {
        if (std::uncaught_exceptions() == unwinding) return;
        while (published < launched) publish();
        cleanup(hipStreamSynchronize(d.stream));
    }
};

// Single-token GPU part, split so the host can start its experts as soon as the
// routing and activation are exported while the GPU evaluates resident experts.
struct HybridScratch { float *gate_up, *product, *outputs, *probabilities; int32_t* selected; };
HybridScratch hybrid_scratch(ResidentMoe& plan) {
    constexpr size_t routes = 11;
    HybridScratch s{};
    s.gate_up = plan.workspace; s.product = s.gate_up + routes * 1280;
    s.outputs = s.product + routes * 640; s.probabilities = s.outputs + routes * 2560;
    s.selected = reinterpret_cast<int32_t*>(s.probabilities + 11) + routes;
    return s;
}
ExpertGemv hybrid_assist(const Device& d, const ResidentMoe& plan) {
    ExpertGemv assist{};
    for (size_t role = 0; role < 3; ++role) { assist.mapped[role] = plan.mapped[role]; assist.bytes[role] = plan.host_bytes[role]; }
    assist.assist = plan.mapped[0] ? d.assist : 0;
    return assist;
}
void hybrid_route_export(Device& d, ResidentMoe& plan, const float* input, float* exported) {
    const size_t experts = plan.experts;
    const HybridScratch s = hybrid_scratch(plan);
    Matrix projections[2] = {plan.weights[(experts + 1) * 3], plan.weights[(experts + 1) * 3 + 1]};
    projections[1].output_offset = experts;
    gemv(d, projections, 2, input, d.output);
    resident_route<<<1, 256, 0, d.stream>>>(d.output, d.output + experts, experts, s.selected, s.probabilities);
    hybrid_export<<<10, 256, 0, d.stream>>>(s.selected, s.probabilities, input, static_cast<const MoeGroup*>(plan.table),
                                             hybrid_assist(d, plan), exported);
    check(hipGetLastError());
}
#if !defined(__HIPCC__)
// The experts a layer's GPU takes over the bus, copied into VRAM's staging slots by every thread at once. A kernel reads host
// memory as fast as it has loads in flight: 43 GB/s with all of these, 7 with few (a GEMV's lanes are taken to be the latter: unmeasured).
__global__ void hybrid_stage_assist(const MoeGroup* groups, const int32_t* selected, ExpertGemv plan) {
    const size_t step = size_t(gridDim.x) * blockDim.x, thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t absent = 0, slot = 0;
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
void hybrid_resident_experts(Device& d, ResidentMoe& plan, const float* input, float* partial) {
    const HybridScratch s = hybrid_scratch(plan);
    auto* groups = static_cast<MoeGroup*>(plan.table);
    const uint8_t* stage = nullptr;
#if !defined(__HIPCC__)
    if (plan.mapped[0] && d.assist) {
        stage = d.slots.base + d.slots.cache * d.slots.bytes;
        ExpertGemv copy = hybrid_assist(d, plan);
        copy.stage = stage; copy.slot_bytes = d.slots.bytes;
        hybrid_stage_assist<<<64, 256, 0, d.stream>>>(groups, s.selected, copy);
    }
#endif
    resident_expert_launch(d, plan, groups, s.selected, false, input, s.gate_up, 1, 0, 0, false, stage, d.slots.bytes);
    moe_product<<<unsigned((11 * 640 + 255) / 256), 256, 0, d.stream>>>(s.gate_up, 11, s.product);
    resident_expert_launch(d, plan, groups, s.selected, true, s.product, s.outputs, 1, 0, 0, false, stage, d.slots.bytes);
    hybrid_combine<<<10, 256, 0, d.stream>>>(s.outputs, s.probabilities, s.selected, groups, hybrid_assist(d, plan), partial);
    check(hipGetLastError());
}

// Host half of a hybrid layer: CPU experts for absent routes, then the residual update.
extern "C" int sd_cpu_experts_run(void*, const int*, const uint8_t* const*, const uint8_t* const*, const uint8_t* const*,
                                  const float*, uint32_t, uint64_t, uint64_t, const float*, float*);
extern "C" const char* sd_cpu_error();
extern "C" int sd_cpu_read_ahead(void*, const uint8_t* const*, const uint64_t*, uint32_t);
void hybrid_host_experts(Device& d, ResidentMoe& plan, const float* exported) {
    const uint8_t* gate[10]; const uint8_t* up[10]; const uint8_t* down[10]; float weights[10];
    uint32_t count = 0;
    size_t selected[10], taken[10];
    for (size_t route = 0; route < 10; ++route) {
        const float raw = exported[route];
        if (!(raw >= 0 && raw < float(plan.experts))) throw std::runtime_error("invalid hybrid expert route");
        const size_t expert = size_t(raw);
        selected[route] = expert;
        if (exported[2592 + route] != 0) { ++(exported[2592 + route] > 1.5f ? d.assisted_experts : d.gpu_experts); continue; }
        gate[count] = plan.host[0] + expert * plan.host_bytes[0];
        up[count] = plan.host[1] + expert * plan.host_bytes[1];
        down[count] = plan.host[2] + expert * plan.host_bytes[2];
        taken[count] = expert;
        weights[count++] = exported[16 + route];
    }
    hybrid_observe_token(plan, selected);
    float* result = d.exchange + 4096;
    if (count) {
        if (!d.cpu) throw std::runtime_error("hybrid experts need an attached CPU expert engine");
        const int kinds[3] = {plan.weights[0].kind, plan.weights[1].kind, plan.weights[2].kind};
        if (d.hint_experts) {
            // Only experts unread for 64 tokens: 15% of the routes, and 98-100% of the misses of an LRU of 15 to 27 GiB in a
            // 5592-token trace. Hinting every route costs 3% here and 5.4 ms a token on Windows with the pages in memory.
            const uint8_t* ranges[30]; uint64_t sizes[30];
            uint32_t hinted = 0;
            ++plan.host_token;
            for (uint32_t i = 0; i < count; ++i) {
                const bool old = plan.host_token - plan.host_read[taken[i]] > 64;
                plan.host_read[taken[i]] = plan.host_token;
                if (!old) continue;
                ranges[hinted] = gate[i]; ranges[hinted + 1] = up[i]; ranges[hinted + 2] = down[i];
                for (size_t role = 0; role < 3; ++role) sizes[hinted + role] = plan.host_bytes[role];
                hinted += 3;
            }
            if (hinted && sd_cpu_read_ahead(d.cpu, ranges, sizes, hinted))
                throw std::runtime_error(std::string("Straydog CPU experts: ") + sd_cpu_error());
        }
        if (sd_cpu_experts_run(d.cpu, kinds, gate, up, down, weights, count, 2560, 640, exported + 32, result))
            throw std::runtime_error(std::string("Straydog CPU experts: ") + sd_cpu_error());
        d.cpu_experts += count;
    } else std::fill(result, result + 2560, 0.0f);
}

