// Partially resident experts. Dense layer operands, the router and the shared
// expert stay pinned on the GPU; routed experts occupy a bounded pool of device
// slots chosen from routing statistics. Decode evaluates resident experts on the
// GPU and the rest on the host CPU engine in F32 from the authenticated mapping;
// prefill streams every missing expert through staging slots, so prompt rows are
// still computed by the GPU expert GEMM. Every routed expert always contributes.
constexpr size_t hybrid_exchange_floats = 8192;   // selected | probabilities | activation | CPU result
constexpr size_t hybrid_sequence_slot = 4000;     // export completion counter (uint32)
// From hybrid_sequence_slot, as uint32: +1 CPU results published by the host, +2 results
// awaited by the GPU (hybrid_await), +3 nonzero once an await gave up.
constexpr size_t hybrid_published_slot = 1, hybrid_awaited_slot = 2, hybrid_await_failed_slot = 3;

size_t expert_slot_bytes(const ResidentMoe& plan) {
    return plan.host_bytes[0] + plan.host_bytes[1] + plan.host_bytes[2];
}

// Ask the OS to read these experts' host bank slices now, as concurrent large reads. Cold mapped
// weights otherwise arrive one page fault at a time in whichever copy or CPU thread touches them
// first (~1 GB/s from an NVMe that reads 5 GB/s in parallel). The hint itself costs a system call
// even for resident pages, so an expert hinted within the last 30 s is skipped. Linux hints on the
// device's read-ahead thread (see ReadAhead). A hint only: it never changes results.
void prefetch_experts(Device& d, ResidentMoe& plan, const size_t* experts, size_t count) {
#if defined(_WIN32) || defined(__linux__)
    const auto now = std::chrono::steady_clock::now();
    if (plan.prefetched.size() != plan.experts) plan.prefetched.assign(plan.experts, {});
    std::vector<std::pair<const uint8_t*, size_t>> ranges;
    for (size_t i = 0; i < count; ++i) {
        auto& last = plan.prefetched[experts[i]];
        if (last != std::chrono::steady_clock::time_point{} && now - last < std::chrono::seconds(30)) continue;
        last = now;
        for (size_t role = 0; role < 3; ++role)
            ranges.emplace_back(plan.host[role] + experts[i] * plan.host_bytes[role], plan.host_bytes[role]);
    }
    if (ranges.empty()) return;
#endif
#if defined(_WIN32)
    std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
    entries.reserve(ranges.size());
    for (const auto& range : ranges) entries.push_back({const_cast<uint8_t*>(range.first), range.second});
    PrefetchVirtualMemory(GetCurrentProcess(), entries.size(), entries.data(), 0);
    (void)d;
#elif defined(__linux__)
    if (!d.read_ahead) d.read_ahead = std::make_unique<ReadAhead>();
    d.read_ahead->add(ranges);
#else
    (void)d; (void)plan; (void)experts; (void)count;
#endif
}

// routed: the GGML kinds of the host banks' gate, up and down slices.
void* create_hybrid_moe(Device& d, void* const* pointers, const int* kinds, const int* routed_kinds, size_t experts,
                        const uint8_t* const* host, void* workspace, size_t capacity) {
    if (!pointers || !kinds || !routed_kinds || !host || experts < 10 || experts > 512 || !capacity || capacity > 4096)
        throw std::runtime_error("invalid hybrid MoE plan");
    // Prefill computes absent experts only with the grouped FP16 expert GEMM: refuse the plan at load,
    // not at first use.
    if (!(d.expert_prefill && d.half_gemm))
        throw std::runtime_error("partially resident experts need matmul_precision fp16 on an RDNA4 or CUDA sm_70+ GPU");
    for (size_t role = 0; role < 3; ++role)
        if (!host[role]) throw std::runtime_error("hybrid MoE requires the host expert banks");
    auto storage = d.allocations.find(workspace);
    if (storage == d.allocations.end() || storage->second < (capacity * 49302 + 65536) * 4)
        throw std::runtime_error("invalid hybrid MoE workspace");
    auto plan = std::make_unique<ResidentMoe>();
    plan->experts = experts; plan->capacity = capacity; plan->hybrid = true;
    plan->host_read.assign(experts, 0); plan->host_token = 65;   // every expert starts out old
    plan->workspace = static_cast<float*>(workspace); plan->dependencies.assign(pointers, pointers + 5);
    // Routed experts are slices of 640x2560, 640x2560 and 2560x640.
    const size_t rows[8] = {640, 640, 2560, 640, 640, 2560, experts, 1};
    const size_t cols[8] = {2560, 2560, 640, 2560, 2560, 640, 2560, 2560};
    Matrix routed[3]{};
    for (size_t role = 0; role < 3; ++role) {
        const size_t row_bytes = matrix_row_bytes(routed_kinds[role], cols[role]);
        routed[role] = {nullptr, row_bytes, rows[role], cols[role], 0, 0, 0, routed_kinds[role]};
        plan->host[role] = host[role]; plan->host_bytes[role] = rows[role] * row_bytes;
    }
    Matrix dense[5]{};
    for (size_t i = 0; i < 5; ++i) {
        const size_t row_bytes = matrix_row_bytes(kinds[i], cols[i + 3]);
        auto found = d.allocations.find(pointers[i]);
        if (found == d.allocations.end() || found->second != rows[i + 3] * row_bytes)
            throw std::runtime_error("invalid hybrid MoE dense operand size");
        dense[i] = {static_cast<const uint8_t*>(pointers[i]), row_bytes, rows[i + 3], cols[i + 3], 0, 0, 0, kinds[i]};
    }
    for (size_t expert = 0; expert < experts; ++expert)
        for (size_t role = 0; role < 3; ++role) plan->weights.push_back(routed[role]);
    for (size_t i = 0; i < 5; ++i) plan->weights.push_back(dense[i]);
    plan->slot_of.assign(experts, -1);
    plan->usage.assign(experts, 0);
    std::vector<MoeGroup> groups(experts + 1);
    for (size_t group = 0; group <= experts; ++group)
        groups[group] = {{plan->weights[group * 3], plan->weights[group * 3 + 1], plan->weights[group * 3 + 2]}, 0, 0, 0};
    const size_t bytes = groups.size() * sizeof(MoeGroup);
    if (bytes > d.budget - d.used) throw std::runtime_error("hybrid MoE metadata exceeds VRAM budget");
    void* table = nullptr; check(hipMalloc(&table, bytes));
    try {
        upload(d, table, groups.data(), bytes);
        check(hipStreamSynchronize(d.stream));
    } catch (...) { cleanup(hipFree(table)); throw; }
    d.allocations.emplace(table, bytes); d.used += bytes; plan->table = table;
    void* handle = plan.get(); d.resident.emplace(handle, std::move(plan));
    return handle;
}

void create_expert_slots(Device& d, size_t cache, size_t stage, size_t slot_bytes) {
    if (d.slots.base || !slot_bytes || slot_bytes % 256 || stage < 2 || cache + stage > 65536)
        throw std::runtime_error("invalid Straydog expert slot pool");
    for (const auto& entry : d.resident)
        if (entry.second->hybrid && expert_slot_bytes(*entry.second) > slot_bytes)
            throw std::runtime_error("an expert does not fit the Straydog expert slot size");
    const size_t bytes = (cache + stage) * slot_bytes;
    const size_t extra = hybrid_exchange_floats * 4;
    if (bytes > d.budget - d.used || extra > d.budget - d.used - bytes)
        throw std::runtime_error("expert slot pool exceeds the Straydog VRAM budget");
    check(hipMalloc(reinterpret_cast<void**>(&d.slots.base), bytes));
    d.used += bytes;
    d.slots.cache = cache; d.slots.stage = stage; d.slots.bytes = slot_bytes;
    d.slots.owner.assign(cache, {nullptr, -1});
    check(hipMalloc(reinterpret_cast<void**>(&d.cpu_partial), extra));
    d.used += extra;
    // Host (not VRAM) memory: the staging slots' page-locked mirror for gathered prefill copies.
    check(hipHostMalloc(reinterpret_cast<void**>(&d.slots.host), stage * slot_bytes, hipHostMallocDefault));
    check(hipHostMalloc(reinterpret_cast<void**>(&d.exchange), hybrid_exchange_floats * 4, hipHostMallocDefault));
    std::memset(d.exchange, 0, hybrid_exchange_floats * 4);
    check(hipHostGetDevicePointer(reinterpret_cast<void**>(&d.exchange_device), d.exchange, 0));
    check(hipMemsetAsync(d.flags, 0, 16, d.stream));
    check(hipStreamSynchronize(d.stream));
    d.hybrid_sequence = 0;
    check(hipStreamCreate(&d.copy_stream));
    check(hipEventCreateWithFlags(&d.swapped, hipEventDisableTiming));
    d.copier = std::make_unique<Background>(d.ordinal);
}

// Spin until the GPU publishes this export (pinned memory written by stream-ordered
// copies), failing instead of hanging if the stream reports an error.
void hybrid_wait_export(Device& d, uint32_t expected) {
    volatile uint32_t* published = reinterpret_cast<volatile uint32_t*>(d.exchange + hybrid_sequence_slot);
    for (uint64_t spin = 0; *published != expected; ++spin) {
        if ((spin & 4095) == 4095) {
            const hipError_t state = hipStreamQuery(d.stream);
            if (state != hipSuccess && state != hipErrorNotReady) check(state);
            if (state == hipSuccess && *published != expected) throw std::runtime_error("hybrid export was not published");
        }
#ifdef __x86_64__
        __builtin_ia32_pause();
#elif defined(_M_X64)
        _mm_pause();
#endif
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

// Point the plan's host mirror and device table entry for `expert` at `base` (nullptr evicts).
void set_expert(Device& d, ResidentMoe& plan, size_t expert, uint8_t* base, hipStream_t stream) {
    size_t offset = 0;
    for (size_t role = 0; role < 3; ++role) {
        plan.weights[expert * 3 + role].data = base ? base + offset : nullptr;
        offset += plan.host_bytes[role];
    }
    auto* entry = static_cast<MoeGroup*>(plan.table) + expert;
    check(hipMemcpyAsync(entry, &plan.weights[expert * 3], 3 * sizeof(Matrix), hipMemcpyHostToDevice, stream));
}
void copy_expert(Device& d, const ResidentMoe& plan, size_t expert, uint8_t* base, hipStream_t stream) {
    size_t offset = 0;
    for (size_t role = 0; role < 3; ++role) {
        check(hipMemcpyAsync(base + offset, plan.host[role] + expert * plan.host_bytes[role], plan.host_bytes[role],
                             hipMemcpyHostToDevice, stream));
        offset += plan.host_bytes[role];
    }
}

// Prefill: missing experts are copied into staging slots in chunks on the copy
// stream while earlier chunks compute. Device table entries are switched by a
// kernel in stream order (no host round trip) and restored afterwards.
constexpr size_t hybrid_point_limit = 64;
struct ExpertPointers { int32_t expert[hybrid_point_limit]; const uint8_t* base[hybrid_point_limit]; uint32_t count; uint64_t gate, up; };
__global__ void hybrid_point(MoeGroup* groups, ExpertPointers list) {
    const uint32_t i = threadIdx.x;
    if (i >= list.count) return;
    MoeGroup& group = groups[list.expert[i]];
    const uint8_t* base = list.base[i];
    group.weights[0].data = base;
    group.weights[1].data = base ? base + list.gate : nullptr;
    group.weights[2].data = base ? base + list.gate + list.up : nullptr;
}
struct StreamEvents {
    hipEvent_t loaded[2]{}, used[2]{};
    StreamEvents() {
        for (int i = 0; i < 2; ++i) {
            check(hipEventCreateWithFlags(&loaded[i], hipEventDisableTiming));
            check(hipEventCreateWithFlags(&used[i], hipEventDisableTiming));
        }
    }
    ~StreamEvents() { for (int i = 0; i < 2; ++i) { cleanup(hipEventDestroy(loaded[i])); cleanup(hipEventDestroy(used[i])); } }
};
extern "C" int sd_cpu_gather(void*, const uint8_t* const*, uint8_t* const*, const uint64_t*, uint32_t);
extern "C" const char* sd_cpu_error();
// Gather experts' gate, up and down slices from the mapped banks into the slot images at `slots`.
void hybrid_gather(Device& d, const ResidentMoe& plan, const size_t* experts, size_t count, uint8_t* const* slots) {
    std::vector<const uint8_t*> sources;
    std::vector<uint8_t*> targets;
    std::vector<uint64_t> bytes;
    for (size_t i = 0; i < count; ++i) {
        size_t offset = 0;
        for (size_t role = 0; role < 3; ++role) {
            sources.push_back(plan.host[role] + experts[i] * plan.host_bytes[role]);
            targets.push_back(slots[i] + offset);
            bytes.push_back(plan.host_bytes[role]);
            offset += plan.host_bytes[role];
        }
    }
    if (sd_cpu_gather(d.cpu, sources.data(), targets.data(), bytes.data(), uint32_t(sources.size())))
        throw std::runtime_error(std::string("Straydog CPU gather: ") + sd_cpu_error());
}
void hybrid_point_experts(Device& d, ResidentMoe& plan, const size_t* experts, size_t count, const uint8_t* base) {
    for (size_t first = 0; first < count; first += hybrid_point_limit) {
        ExpertPointers list{};
        list.count = uint32_t(std::min(hybrid_point_limit, count - first));
        list.gate = plan.host_bytes[0]; list.up = plan.host_bytes[1];
        for (uint32_t i = 0; i < list.count; ++i) {
            list.expert[i] = int32_t(experts[first + i]);
            list.base[i] = base ? base + (first + i) * d.slots.bytes : nullptr;
        }
        hybrid_point<<<1, hybrid_point_limit, 0, d.stream>>>(static_cast<MoeGroup*>(plan.table), list);
    }
    check(hipGetLastError());
}
// Routing scores are exponential moving averages over tokens (factor per token),
// so recent prompt and decode routing decide residency and every layer's scores
// are comparable: the cache holds the globally highest (layer, expert) scores.
// A prompt batch counts its head as a uniform average and its last tokens one by
// one, so the end of the prompt (the request itself) weighs most.
double hybrid_token_decay = 0.95;
constexpr size_t hybrid_prompt_tail = 64;
void hybrid_observe_batch(ResidentMoe& plan, const unsigned* counts, size_t tokens, const int32_t* tail, size_t tail_tokens) {
    const double decay = hybrid_token_decay;
    std::vector<double> recent(plan.experts, 0.0), tail_counts(plan.experts, 0.0);
    double weight = 1;
    for (size_t t = tail_tokens; t-- > 0; weight *= decay)
        for (size_t route = 0; route < 10; ++route) {
            const size_t expert = size_t(tail[t * 10 + route]);
            if (expert >= plan.experts) throw std::runtime_error("invalid hybrid prompt route");
            recent[expert] += weight; tail_counts[expert] += 1;
        }
    const size_t head = tokens - tail_tokens;
    const double keep = std::pow(decay, double(tokens));
    const double head_weight = head ? weight * (1 - std::pow(decay, double(head))) / ((1 - decay) * double(head)) : 0;
    for (size_t expert = 0; expert < plan.experts; ++expert)
        plan.usage[expert] = plan.usage[expert] * keep + (counts[expert] - tail_counts[expert]) * head_weight + recent[expert];
}
void hybrid_observe_token(ResidentMoe& plan, const size_t* selected) {
    for (double& value : plan.usage) value *= hybrid_token_decay;
    for (size_t route = 0; route < 10; ++route) plan.usage[selected[route]] += 1;
}

// Swap in higher-scoring experts. Table switches run in stream order through
// kernels and the copy stream, so asynchronous swaps never expose a slot that is
// still being written or read; the host follows the residency the GPU exports.
// Table updates for (expert, base) pairs of one plan; a null base marks the expert absent.
void hybrid_point_list(Device& d, ResidentMoe& plan, const std::vector<std::pair<size_t, const uint8_t*>>& items) {
    for (size_t first = 0; first < items.size(); first += hybrid_point_limit) {
        ExpertPointers list{};
        list.count = uint32_t(std::min(hybrid_point_limit, items.size() - first));
        list.gate = plan.host_bytes[0]; list.up = plan.host_bytes[1];
        for (uint32_t i = 0; i < list.count; ++i) {
            list.expert[i] = int32_t(items[first + i].first);
            list.base[i] = items[first + i].second;
        }
        hybrid_point<<<1, hybrid_point_limit, 0, d.stream>>>(static_cast<MoeGroup*>(plan.table), list);
    }
    check(hipGetLastError());
}
// Swapped-in experts become visible only after their copies have landed; until
// then they stay absent (CPU in decode, streamed in prefill). Returns false while
// the copies are still in flight and `wait` is not set.
bool hybrid_settle(Device& d, bool wait) {
    if (d.slots.pending.empty()) return true;
    if (d.copier) {
        if (!wait && d.copier->busy()) return false;
        d.copier->wait();   // the copies are issued and d.swapped is recorded (or the failure rethrown)
    }
    if (wait) check(hipEventSynchronize(d.swapped));
    else {
        const hipError_t state = hipEventQuery(d.swapped);
        if (state == hipErrorNotReady) return false;
        check(state);
    }
    std::unordered_map<ResidentMoe*, std::vector<std::pair<size_t, const uint8_t*>>> loaded;
    for (const size_t slot : d.slots.pending) {
        ResidentMoe* plan = d.slots.owner[slot].first;
        const size_t expert = size_t(d.slots.owner[slot].second);
        uint8_t* base = d.slots.base + slot * d.slots.bytes;
        loaded[plan].emplace_back(expert, base);
        plan->slot_of[expert] = int32_t(slot);
        plan->weights[expert * 3].data = base;
    }
    for (const auto& item : loaded) hybrid_point_list(d, *item.first, item.second);
    d.slots.pending.clear();
    return true;
}

// Bounded swaps (max_swaps > 0) never block the compute stream: victims are
// unmapped in stream order, the copy stream loads the new experts after that,
// and hybrid_settle maps them once loaded. A refresh in flight skips the round.
void hybrid_rebalance(Device& d, size_t max_swaps, bool wait) {
    if (!d.slots.base || !hybrid_settle(d, wait)) return;
    struct Entry { ResidentMoe* plan; size_t expert; double score; };
    std::vector<Entry> resident, candidates;
    std::vector<size_t> free_slots;
    for (auto& item : d.resident) {
        if (!item.second->hybrid) continue;
        ResidentMoe& plan = *item.second;
        for (size_t expert = 0; expert < plan.experts; ++expert) {
            if (plan.slot_of[expert] >= 0) resident.push_back({&plan, expert, plan.usage[expert]});
            else if (plan.slot_of[expert] == -1 && plan.usage[expert] > 0) candidates.push_back({&plan, expert, plan.usage[expert]});
        }
    }
    for (size_t slot = 0; slot < d.slots.cache; ++slot) if (!d.slots.owner[slot].first) free_slots.push_back(slot);
    auto order = [](const Entry& a, const Entry& b) {
        return a.score != b.score ? a.score > b.score : a.plan != b.plan ? a.plan < b.plan : a.expert < b.expert;
    };
    // Bounded refreshes only need the strongest candidates and the weakest residents.
    auto weakest = [&](const Entry& a, const Entry& b) { return order(b, a); };
    if (max_swaps && candidates.size() > max_swaps) {
        std::partial_sort(candidates.begin(), candidates.begin() + max_swaps, candidates.end(), order);
        candidates.resize(max_swaps);
    } else std::sort(candidates.begin(), candidates.end(), order);
    if (max_swaps && resident.size() > max_swaps) {
        std::partial_sort(resident.begin(), resident.begin() + max_swaps, resident.end(), weakest);
        resident.resize(max_swaps);
    } else std::sort(resident.begin(), resident.end(), weakest);
    std::unordered_map<ResidentMoe*, std::vector<std::pair<size_t, const uint8_t*>>> released;
    size_t victim = 0, next_free = 0;
    for (const Entry& candidate : candidates) {
        if (max_swaps && d.slots.pending.size() == max_swaps) break;
        size_t slot;
        if (next_free < free_slots.size()) slot = free_slots[next_free++];
        else {
            // Hysteresis: replace only clearly weaker residents.
            if (victim == resident.size() || candidate.score <= resident[victim].score * 1.25 + 0.05) break;
            const Entry& old = resident[victim++];
            slot = size_t(old.plan->slot_of[old.expert]);
            released[old.plan].emplace_back(old.expert, nullptr);
            old.plan->slot_of[old.expert] = -1;
            old.plan->weights[old.expert * 3].data = nullptr;
        }
        candidate.plan->slot_of[candidate.expert] = -2;   // loading
        d.slots.owner[slot] = {candidate.plan, int32_t(candidate.expert)};
        d.slots.pending.push_back(slot);
    }
    if (d.slots.pending.empty()) return;
    for (const auto& item : released) hybrid_point_list(d, *item.first, item.second);
    hipEvent_t unmapped = nullptr;
    check(hipEventCreateWithFlags(&unmapped, hipEventDisableTiming));
    check(hipEventRecord(unmapped, d.stream));
    check(hipStreamWaitEvent(d.copy_stream, unmapped, 0));
    cleanup(hipEventDestroy(unmapped));
    // Copies from pageable mappings block the issuing thread until staged, so bounded
    // swaps are issued from the device's background thread; hybrid_settle waits for it.
    struct Copy { ResidentMoe* plan; size_t expert; uint8_t* base; };
    std::vector<Copy> copies;
    for (const size_t slot : d.slots.pending)
        copies.push_back({d.slots.owner[slot].first, size_t(d.slots.owner[slot].second), d.slots.base + slot * d.slots.bytes});
    auto issue = [&d, copies]() {
        for (const Copy& copy : copies) prefetch_experts(d, *copy.plan, &copy.expert, 1);
        for (const Copy& copy : copies) copy_expert(d, *copy.plan, copy.expert, copy.base, d.copy_stream);
        check(hipEventRecord(d.swapped, d.copy_stream));
    };
    if (wait || !d.copier) issue();
    else d.copier->run(std::move(issue));
    d.swapped_experts += d.slots.pending.size();
    if (wait) { hybrid_settle(d, true); check(hipStreamSynchronize(d.stream)); }
}

// compute(first, count) launches the expert GEMMs for missing[first, first+count).
template<class Compute>
void hybrid_stream_missing(Device& d, ResidentMoe& plan, const std::vector<size_t>& missing, Compute&& compute) {
    if (missing.empty()) return;
    if (d.copier) d.copier->wait();   // the copy stream and read-ahead state are this thread's again
    prefetch_experts(d, plan, missing.data(), missing.size());
    const size_t half = std::min(d.slots.stage / 2, hybrid_point_limit);
    // Pageable banks: the CPU workers gather each chunk into the page-locked mirror while the GPU
    // computes the previous one, so cold pages fault in concurrently instead of one copy at a time
    // (cached ones pay an extra pass through host memory); the experts then copy by DMA.
    // Page-locked (registered) banks copy directly.
    const bool gather = d.slots.host && d.cpu && !plan.mapped[0];
    StreamEvents events;
    std::vector<uint8_t*> targets;
    for (size_t first = 0, chunk = 0; first < missing.size(); first += half, ++chunk) {
        const size_t count = std::min(half, missing.size() - first), buffer = chunk % 2;
        uint8_t* base = d.slots.base + (d.slots.cache + buffer * half) * d.slots.bytes;
        uint8_t* host = gather ? d.slots.host + buffer * half * d.slots.bytes : nullptr;
        if (gather) {
            // The DMA two chunks back read this mirror (its `loaded` event follows it on the copy stream).
            if (chunk >= 2) check(hipEventSynchronize(events.loaded[buffer]));
            targets.clear();
            for (size_t i = 0; i < count; ++i) targets.push_back(host + i * d.slots.bytes);
            hybrid_gather(d, plan, missing.data() + first, count, targets.data());
        }
        if (chunk >= 2) check(hipStreamWaitEvent(d.copy_stream, events.used[buffer], 0));
        for (size_t i = 0; i < count; ++i) {
            uint8_t* target = base + i * d.slots.bytes;
            if (gather)
                check(hipMemcpyAsync(target, host + i * d.slots.bytes, expert_slot_bytes(plan), hipMemcpyHostToDevice,
                                     d.copy_stream));
            else copy_expert(d, plan, missing[first + i], target, d.copy_stream);
        }
        check(hipEventRecord(events.loaded[buffer], d.copy_stream));
        check(hipStreamWaitEvent(d.stream, events.loaded[buffer], 0));
        hybrid_point_experts(d, plan, missing.data() + first, count, base);
        compute(first, count);
        check(hipEventRecord(events.used[buffer], d.stream));
        d.streamed_experts += count;
    }
    hybrid_point_experts(d, plan, missing.data(), missing.size(), nullptr);
    check(hipStreamSynchronize(d.copy_stream));
    check(hipStreamSynchronize(d.stream));
}
