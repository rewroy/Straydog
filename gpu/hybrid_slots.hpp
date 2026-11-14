#include "layer_trace.hpp"
// Partially resident experts. Dense layer operands, the router and the shared
// expert stay pinned on the GPU; routed experts occupy a bounded pool of device
// slots chosen from routing statistics. Decode evaluates resident experts on the
// GPU and the rest on the host CPU engine in F32 from the authenticated mapping;
// prefill streams every missing expert through staging slots, so prompt rows are
// still computed by the GPU expert GEMM. Every routed expert always contributes.
// Each row: selected | probabilities | activation, then its CPU result (a kernel steps through the rows by hybrid_row_floats).
constexpr size_t hybrid_row_export = 2608, hybrid_row_floats = hybrid_row_export + 2560, hybrid_rows = 4;
// Two layers' exchanges, by the layer's parity: a layer's router writes its export beside the one the host may still be reading, so the GPU needs
// the host's publish of the layer before the last (hybrid_await) and not of the one before.
constexpr size_t hybrid_slot_floats = hybrid_rows * hybrid_row_floats;
constexpr size_t hybrid_export_at(size_t row, size_t slot = 0) { return slot * hybrid_slot_floats + row * hybrid_row_floats; }
constexpr size_t hybrid_result_at(size_t row, size_t slot = 0) { return hybrid_export_at(row, slot) + hybrid_row_export; }
// CUDA: an export slot to each layer of the model (the MTP layer 48 last), so that a layer's export is written where no other layer's was and nothing
// waits for the host to have read one before the next is written; HIP: the two by parity, as above.
#if defined(__HIPCC__)
constexpr size_t hybrid_layer_slots = 2;
constexpr size_t hybrid_slot_of(size_t layer) { return layer & 1; }
#else
constexpr size_t hybrid_layer_slots = 49;
constexpr size_t hybrid_slot_of(size_t layer) { return layer; }
#endif
constexpr size_t hybrid_sequence_slot = hybrid_layer_slots * hybrid_slot_floats;     // export completion counter (uint32)
constexpr size_t hybrid_exchange_floats = hybrid_sequence_slot + 16;
// d.cpu_partial (device): the rows' staged CPU results and the plan's room after them, two exports' worth as before.
constexpr size_t hybrid_partial_floats = 2 * hybrid_slot_floats + 16;
// DS41_PROFILE=hc (CUDA): after those, a layer's gate and up, products and down once more, in one launch for all its routes (hybrid_resident_experts holds
// the overlapped layer's outputs against them).
#if !defined(__HIPCC__)
inline size_t hybrid_check_scratch_floats() {
    const char* profile = std::getenv("DS41_PROFILE");
    return profile && std::strstr(profile, "hc") ? hybrid_rows * 11 * (1280 + 640 + 2560) : 0;
}
#else
inline size_t hybrid_check_scratch_floats() { return 0; }
#endif
// From hybrid_sequence_slot, as uint32: +1 CPU results published by the host, +2 results
// awaited by the GPU (hybrid_await), +3 nonzero once an await gave up.
constexpr size_t hybrid_published_slot = 1, hybrid_awaited_slot = 2, hybrid_await_failed_slot = 3;
// Device::flags[4]: a failed await stays latched until checked recovery; the existing device initialization clears it.
constexpr size_t hybrid_await_device_failed = 4;
// Device::flags[16 + row]: the blocks of a row that hybrid_router has counted in (zero between launches, from create_hybrid_moe on).
constexpr size_t hybrid_router_counters = 16;

size_t expert_slot_bytes(const ResidentMoe& plan) {
    return plan.host_bytes[0] + plan.host_bytes[1] + plan.host_bytes[2];
}

// Ask the OS to read these experts' host bank slices now, as concurrent large reads. Cold mapped
// weights otherwise arrive one page fault at a time in whichever copy or CPU thread touches them
// first (~1 GB/s from an NVMe that reads 5 GB/s in parallel). The hint itself costs a system call
// even for resident pages, so an expert hinted within the last 30 s is skipped. Linux hints on the
// device's read-ahead thread (see ReadAhead). A hint only: it never changes results.
void prefetch_experts(Device& d, ResidentMoe& plan, const size_t* experts, size_t count) {
    // A layer held as page-locked copies (sd_hybrid_repoint) no longer reads the mapping: there is nothing to ask the OS for.
    if (plan.host_copy) return;
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
    plan->routed_gpu.assign(experts, 0); plan->routed_host.assign(experts, 0);
    std::vector<MoeGroup> groups(experts + 1);
    for (size_t group = 0; group <= experts; ++group)
        groups[group] = {{plan->weights[group * 3], plan->weights[group * 3 + 1], plan->weights[group * 3 + 2]}, 0, 0, 0};
    const size_t bytes = groups.size() * sizeof(MoeGroup);
    if (bytes > d.budget - d.used) throw std::runtime_error("hybrid MoE metadata exceeds VRAM budget");
    void* table = nullptr; check(hipMalloc(&table, bytes));
    try {
        upload(d, table, groups.data(), bytes);
        check(hipMemsetAsync(d.flags + hybrid_router_counters, 0, hybrid_rows * sizeof(unsigned), d.stream));
        check(hipStreamSynchronize(d.stream));
    } catch (...) { cleanup(hipFree(table)); throw; }
    d.allocations.emplace(table, bytes); d.used += bytes; plan->table = table;
    void* handle = plan.get(); d.resident.emplace(handle, std::move(plan));
    return handle;
}

bool slot_soa_plan(const ResidentMoe& plan) {
    const Matrix& gate=plan.weights[0]; const Matrix& up=plan.weights[1];
    return gate.kind==up.kind && expert_soa_eligible(gate.kind,gate.rows,gate.cols) &&
           expert_soa_eligible(up.kind,up.rows,up.cols);
}
uint32_t slot_soa_layout(const Device& d,const ResidentMoe& plan,const uint8_t* base,size_t role) {
    if (!base || role>=2 || !d.slots.soa_scratch || !slot_soa_plan(plan)) return 0;
    const uintptr_t address=reinterpret_cast<uintptr_t>(base),begin=reinterpret_cast<uintptr_t>(d.slots.base);
    return address>=begin && address-begin<d.slots.arena-d.slots.soa_scratch ? 1 : 0;
}

// The pool in slots of the sizes its layers take. A mixed bundle's layers differ by as much as 1.8 times, and a slot of the largest size for
// all of them holds a quarter fewer experts. Sizes within a tenth share a group, a group's slots are in proportion to its layers, and an
// expert only replaces one of its own group, so the scores compare within a group: a layer's routes want about what another's do.
void expert_slots_carve(Device& d, size_t arena) {
    std::vector<std::pair<size_t, ResidentMoe*>> layers;
    for (auto& item : d.resident) if (item.second->hybrid) layers.emplace_back(expert_slot_bytes(*item.second), item.second.get());
    std::sort(layers.begin(), layers.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first < b.first : a.second < b.second; });
    struct Group { size_t bytes, layers; };
    std::vector<Group> groups;
    size_t floor_bytes = 0;
    for (const auto& layer : layers) {
        if (groups.empty() || layer.first > floor_bytes + floor_bytes / 10) { groups.push_back({layer.first, 0}); floor_bytes = layer.first; }
        groups.back().bytes = (layer.first + 255) / 256 * 256;
        ++groups.back().layers;
        layer.second->slot_group = uint8_t(groups.size() - 1);
    }
    size_t weight = 0;
    for (const Group& group : groups) weight += group.bytes * group.layers;
    d.slots.at.clear(); d.slots.group.clear();
    size_t at = 0;
    for (size_t g = 0; g < groups.size() && weight; ++g) {
        const size_t count = size_t(double(arena) * double(groups[g].layers) / double(weight));
        for (size_t i = 0; i < count; ++i, at += groups[g].bytes) {
            d.slots.at.push_back(at);
            d.slots.group.push_back(uint8_t(g));
        }
    }
    size_t scratch=0;
    for (const auto& layer:layers) if (slot_soa_plan(*layer.second))
        for (size_t role=0;role<2;++role) {
            const Matrix& matrix=layer.second->weights[role];
            scratch=std::max(scratch,(matrix.rows*matrix.row_bytes+255)/256*256);
        }
    d.slots.soa_scratch=0;
    // Reuse the unused tail first; at most one cache slot gives way to the
    // private conversion buffer. Allocation bytes and the stage boundary stay.
    if (scratch && !d.slots.at.empty()) {
        if (arena-at<scratch && d.slots.at.size()>1) {
            at=d.slots.at.back(); d.slots.at.pop_back(); d.slots.group.pop_back();
        }
        if (arena-at>=scratch) d.slots.soa_scratch=scratch;
    }
    d.slots.arena = arena;
    d.slots.cache = d.slots.at.size();
    d.slots.owner.assign(d.slots.cache, {nullptr, -1});
}
// A failed attempt leaves the pool as it was: empty. The caller retries a smaller one. This does not shrink the cache itself.
void rollback_expert_slots(Device& d, size_t added) {
    d.copier.reset();
    if (d.swapped) { cleanup(hipEventDestroy(d.swapped)); d.swapped = nullptr; }
    if (d.copy_stream) { cleanup(hipStreamDestroy(d.copy_stream)); d.copy_stream = nullptr; }
    if (d.stage_stream) { cleanup(hipStreamDestroy(d.stage_stream)); d.stage_stream = nullptr; }
    if (d.stage_fork) { cleanup(hipEventDestroy(d.stage_fork)); d.stage_fork = nullptr; }
    if (d.stage_join) { cleanup(hipEventDestroy(d.stage_join)); d.stage_join = nullptr; }
    if (d.exchange) { cleanup(hipHostFree(d.exchange)); d.exchange = nullptr; }
    d.exchange_device = nullptr;
    if (d.slots.host) { cleanup(hipHostFree(d.slots.host)); d.slots.host = nullptr; }
    if (d.cpu_partial) { cleanup(hipFree(d.cpu_partial)); d.cpu_partial = nullptr; }
    if (d.slots.base) { cleanup(hipFree(d.slots.base)); d.slots.base = nullptr; }
    d.used -= added;
    d.slots.stage = 0; d.slots.chunk = 0; d.slots.bytes = 0; d.slots.cache = 0; d.slots.arena = 0; d.slots.soa_scratch=0;
    d.slots.at.clear(); d.slots.group.clear(); d.slots.owner.clear(); d.slots.pending.clear();
    d.hybrid_sequence = 0;
}
void create_expert_slots(Device& d, size_t cache, size_t stage, size_t slot_bytes, bool mirror) {
    if (d.slots.base || !slot_bytes || slot_bytes % 256 || stage < 2 || cache + stage > 65536)
        throw std::runtime_error("invalid Straydog expert slot pool");
    for (const auto& entry : d.resident)
        if (entry.second->hybrid && expert_slot_bytes(*entry.second) > slot_bytes)
            throw std::runtime_error("an expert does not fit the Straydog expert slot size");
    const size_t bytes = (cache + stage) * slot_bytes;   // (`cache` slots of the largest size: expert_slots_carve makes more of it)
    const size_t extra = (hybrid_partial_floats + hybrid_check_scratch_floats()) * 4;
    if (bytes > d.budget - d.used || extra > d.budget - d.used - bytes)
        throw std::runtime_error("expert slot pool exceeds the Straydog VRAM budget");
    size_t added = 0;
    try {
        check(hipMalloc(reinterpret_cast<void**>(&d.slots.base), bytes));
        d.used += bytes;
        added += bytes;
        d.slots.stage = stage; d.slots.bytes = slot_bytes;
        // (the first 32 staging slots are the chunks' two buffers; those past them, where there are any, hold prestaged experts)
        d.slots.chunk = std::min<size_t>(stage, 32);
        expert_slots_carve(d, cache * slot_bytes);
        check(hipMalloc(reinterpret_cast<void**>(&d.cpu_partial), extra));
#if !defined(__HIPCC__)
        check(hipMemset(d.cpu_partial, 0, extra));   // (the counts of a layer's missed experts live here and are added to on every pass: they start at 0)
#else
        { const char* profile = std::getenv("DS41_PROFILE"); if (profile && std::strstr(profile, "hc")) check(hipMemset(d.cpu_partial, 0, extra)); }   // (the check's counters start at 0)
#endif
        d.used += extra;
        added += extra;
        // Host (not VRAM) memory: the staging slots' page-locked mirror for gathered prefill copies.
        // A refusal here is named, so the caller can start without the mirror instead of shrinking the cache.
        if (mirror) {
            const hipError_t pinned = hipHostMalloc(reinterpret_cast<void**>(&d.slots.host), d.slots.chunk * slot_bytes, hipHostMallocDefault);
            if (pinned != hipSuccess)
                throw std::runtime_error(std::string(gpu_error_text(pinned)) + " (expert slot staging mirror, code " + std::to_string(int(pinned)) + ")");
        }
        check(hipHostMalloc(reinterpret_cast<void**>(&d.exchange), hybrid_exchange_floats * 4, hipHostMallocDefault));
        std::memset(d.exchange, 0, hybrid_exchange_floats * 4);
        check(hipHostGetDevicePointer(reinterpret_cast<void**>(&d.exchange_device), d.exchange, 0));
        check(hipMemsetAsync(d.flags, 0, 16, d.stream));
        check(hipStreamSynchronize(d.stream));
        d.hybrid_sequence = 0;
        check(hipStreamCreate(&d.copy_stream));
        check(hipEventCreateWithFlags(&d.swapped, hipEventDisableTiming));
#if !defined(__HIPCC__)
        // (the staging copy's stream and its fork and join, made here, not in a graph's capture; of the highest priority, as the send stream of qwen_parts.hpp is:
        // the copy starts at once beside the resident experts' launches, which fill the device)
        int least = 0, greatest = 0;
        check(hipDeviceGetStreamPriorityRange(&least, &greatest));
        check(hipStreamCreateWithPriority(&d.stage_stream, hipStreamDefault, greatest));
        check(hipEventCreateWithFlags(&d.stage_fork, hipEventDisableTiming));
        check(hipEventCreateWithFlags(&d.stage_join, hipEventDisableTiming));
#endif
        d.copier = std::make_unique<Background>(d.ordinal);
    } catch (...) {
        rollback_expert_slots(d, added);
        throw;
    }
}

// Spin until the GPU publishes this export (pinned memory written by stream-ordered
// copies), failing instead of hanging if the stream reports an error.
void hybrid_wait_export(Device& d, uint32_t expected) {
    volatile uint32_t* published = reinterpret_cast<volatile uint32_t*>(d.exchange + hybrid_sequence_slot);
    // (the count may be ahead of this layer's: the GPU goes on, up to a layer, while the host has not yet read this export)
    for (uint64_t spin = 0; int32_t(*published - expected) < 0; ++spin) {
        if ((spin & 4095) == 4095) {
            const hipError_t state = hipStreamQuery(d.stream);
            if (state != hipSuccess && state != hipErrorNotReady) check(state);
            if (state == hipSuccess && int32_t(*published - expected) < 0) throw std::runtime_error("hybrid export was not published");
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
        plan.weights[expert * 3 + role].layout=slot_soa_layout(d,plan,base,role);
        offset += plan.host_bytes[role];
    }
    auto* entry = static_cast<MoeGroup*>(plan.table) + expert;
    check(hipMemcpyAsync(entry, &plan.weights[expert * 3], 3 * sizeof(Matrix), hipMemcpyHostToDevice, stream));
}
void copy_expert(Device& d, const ResidentMoe& plan, size_t expert, uint8_t* base, hipStream_t stream) {
    size_t offset = 0;
    for (size_t role = 0; role < 3; ++role) {
        const uint8_t* source=plan.host[role]+expert*plan.host_bytes[role];
        if (slot_soa_layout(d,plan,base,role)) {
            uint8_t* scratch=d.slots.base+d.slots.arena-d.slots.soa_scratch;
            check(hipMemcpyAsync(scratch,source,plan.host_bytes[role],hipMemcpyHostToDevice,stream));
            const Matrix& matrix=plan.weights[role];
            expert_soa_pack_launch(scratch,base+offset,matrix.kind,matrix.rows,matrix.cols,stream);
        } else check(hipMemcpyAsync(base+offset,source,plan.host_bytes[role],hipMemcpyHostToDevice,stream));
        offset += plan.host_bytes[role];
    }
}

// Prefill: missing experts are copied into staging slots in chunks on the copy
// stream while earlier chunks compute. Device table entries are switched by a
// kernel in stream order (no host round trip) and restored afterwards.
constexpr size_t hybrid_point_limit = 64;
// (with `region` set, a chunk's matrices lie role by role: the gate of its i-th expert at region[0] + (index + i) * stride[0], and so on)
struct ExpertPointers {
    int32_t expert[hybrid_point_limit]; const uint8_t* base[hybrid_point_limit]; uint32_t count; uint64_t gate, up;
    const uint8_t* region[3]; uint64_t stride[3]; uint32_t index;
    uint32_t layout[hybrid_point_limit][3];
};
__global__ void hybrid_point(MoeGroup* groups, ExpertPointers list) {
    const uint32_t i = threadIdx.x;
    if (i >= list.count) return;
    MoeGroup& group = groups[list.expert[i]];
    if (list.region[0]) {
        for (int role = 0; role < 3; ++role) {
            group.weights[role].data = list.region[role] + size_t(list.index + i) * list.stride[role];
            group.weights[role].layout=0;
        }
        return;
    }
    const uint8_t* base = list.base[i];
    group.weights[0].data = base;
    group.weights[1].data = base ? base + list.gate : nullptr;
    group.weights[2].data = base ? base + list.gate + list.up : nullptr;
    for (int role=0;role<3;++role) group.weights[role].layout=list.layout[i][role];
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
void hybrid_gather(Device& d, const ResidentMoe& plan, const size_t* experts, size_t count, uint8_t* const* slots,
                   const uint8_t* const* bank_sources = nullptr) {
    BankReading reading(d);   // (sd_hybrid_repoint waits for the CPU's workers)
    std::vector<const uint8_t*> sources;
    std::vector<uint8_t*> targets;
    std::vector<uint64_t> bytes;
    for (size_t i = 0; i < count; ++i) {
        size_t offset = 0;
        for (size_t role = 0; role < 3; ++role) {
            sources.push_back((bank_sources ? bank_sources[role] : plan.host[role]) + experts[i] * plan.host_bytes[role]);
            targets.push_back(slots[i] + offset);
            bytes.push_back(plan.host_bytes[role]);
            offset += plan.host_bytes[role];
        }
    }
    if (sd_cpu_gather(d.cpu, sources.data(), targets.data(), bytes.data(), uint32_t(sources.size())))
        throw std::runtime_error(std::string("Straydog CPU gather: ") + sd_cpu_error());
}
void hybrid_point_experts(Device& d, ResidentMoe& plan, const size_t* experts, size_t count, const uint8_t* base,
                          uint8_t* const* regions = nullptr) {
    for (size_t first = 0; first < count; first += hybrid_point_limit) {
        ExpertPointers list{};
        list.count = uint32_t(std::min(hybrid_point_limit, count - first));
        list.gate = plan.host_bytes[0]; list.up = plan.host_bytes[1];
        if (regions)
            for (int role = 0; role < 3; ++role) { list.region[role] = regions[role]; list.stride[role] = plan.host_bytes[role]; }
        list.index = uint32_t(first);
        for (uint32_t i = 0; i < list.count; ++i) {
            list.expert[i] = int32_t(experts[first + i]);
            list.base[i] = base ? base + (first + i) * d.slots.bytes : nullptr;
        }
        hybrid_point<<<1, hybrid_point_limit, 0, d.stream>>>(static_cast<MoeGroup*>(plan.table), list);
    }
    check(hipGetLastError());
}
#if !defined(__HIPCC__)
// Prestage (CUDA): the staging slots past the chunks' two buffers hold, for a wide prompt batch's layer, the experts its slots do not hold
// that the last batch through the layer used most. Their copies go over the bus from the layer's page-locked banks beside the layer's own
// dense work (they are asked for before its router has run), and the experts the batch then needs among them are computed with the resident
// ones, with no wait for the bus; the others stream in chunks as ever. The bytes the GEMMs read are the banks' bytes, whichever way they came.
struct EachPointers { int32_t expert[hybrid_point_limit]; const uint8_t* data[hybrid_point_limit][3]; uint32_t count; };
__global__ void hybrid_point_each(MoeGroup* groups, EachPointers list) {
    const uint32_t i = threadIdx.x;
    if (i >= list.count) return;
    MoeGroup& group = groups[list.expert[i]];
    for (int role = 0; role < 3; ++role) { group.weights[role].data=list.data[i][role]; group.weights[role].layout=0; }
}
inline size_t prestage_capacity(const Device& d) { return d.slots.stage > d.slots.chunk ? d.slots.stage - d.slots.chunk : 0; }
inline void prestage_regions(const Device& d, const ResidentMoe& plan, uint8_t* (&regions)[3]) {
    uint8_t* base = d.slots.base + d.slots.arena + d.slots.chunk * d.slots.bytes;
    const size_t capacity = prestage_capacity(d);
    regions[0] = base;
    regions[1] = regions[0] + capacity * plan.host_bytes[0];
    regions[2] = regions[1] + capacity * plan.host_bytes[1];
}
// Whether this layer's prompt batch of `steps` rows may prestage: wide, the banks page-locked, which the copy engine reads asynchronously (the
// engine's copies of them, or the model's mapping registered in place, which sd_hybrid_assist has then mapped), and no mirror to gather through.
// (The mapping registered in place was left out: two RTX 3090 on Linux, whose driver registers host memory read-only, never prestaged, and
// streamed every expert beyond their slots in chunks whose copies wait for the GEMMs two chunks back.)
inline bool prestage_wanted(const Device& d, const ResidentMoe& plan, size_t steps) {
    const bool locked = plan.host_copy || plan.mapped[0];
    return prestage_capacity(d) > 0 && steps >= 2048 && plan.hybrid && !plan.layer_prefill && locked && d.slots.base &&
           !(d.slots.host && d.cpu && !plan.mapped[0]) && d.copy_stream;
}
void prestage_issue(Device& d, ResidentMoe& plan) {
    auto& ps = d.prestage;
    ps.plan = nullptr; ps.experts.clear();
    if (d.copier) d.copier->wait();   // the copy stream is this thread's again
    if (!ps.event) {
        check(hipEventCreateWithFlags(&ps.event, hipEventDisableTiming));
    } else {
        check(hipEventSynchronize(ps.event));   // (copies of a layer that did not use them are done before these overwrite the region)
    }
    const size_t capacity = prestage_capacity(d);
    std::vector<size_t> candidates;
    const bool history = plan.last_counts.size() == plan.experts;
    for (size_t expert = 0; expert < plan.experts; ++expert)
        if (plan.slot_of[expert] < 0 && (!history || plan.last_counts[expert] > 0)) candidates.push_back(expert);
    if (candidates.size() > capacity) {
        std::stable_sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
            return history ? plan.last_counts[a] > plan.last_counts[b] : plan.usage[a] > plan.usage[b];
        });
        candidates.resize(capacity);
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.empty()) return;
    uint8_t* regions[3];
    prestage_regions(d, plan, regions);
    for (size_t i = 0; i < candidates.size();) {
        size_t j = i + 1;
        while (j < candidates.size() && candidates[j] == candidates[j - 1] + 1) ++j;
        for (size_t role = 0; role < 3; ++role)
            check(hipMemcpyAsync(regions[role] + i * plan.host_bytes[role], plan.host[role] + candidates[i] * plan.host_bytes[role],
                                 (j - i) * plan.host_bytes[role], hipMemcpyHostToDevice, d.copy_stream));
        i = j;
    }
    check(hipEventRecord(ps.event, d.copy_stream));
    ps.position.assign(plan.experts, -1);
    for (size_t i = 0; i < candidates.size(); ++i) ps.position[candidates[i]] = int32_t(i);
    if (d.phase_on) { d.phase_prestage_issued += candidates.size(); d.phase_bus_bytes += candidates.size() * expert_slot_bytes(plan); }
    ps.experts = std::move(candidates);
    ps.plan = &plan;
}
// Point (or, `clear`, empty) the table entries of `experts` at (from) their prestaged copies.
void hybrid_point_staged(Device& d, ResidentMoe& plan, const size_t* experts, size_t count, bool clear) {
    uint8_t* regions[3];
    prestage_regions(d, plan, regions);
    for (size_t first = 0; first < count; first += hybrid_point_limit) {
        EachPointers list{};
        list.count = uint32_t(std::min(hybrid_point_limit, count - first));
        for (uint32_t i = 0; i < list.count; ++i) {
            const size_t expert = experts[first + i];
            list.expert[i] = int32_t(expert);
            for (size_t role = 0; role < 3; ++role)
                list.data[i][role] = clear ? nullptr : regions[role] + size_t(d.prestage.position[expert]) * plan.host_bytes[role];
        }
        hybrid_point_each<<<1, hybrid_point_limit, 0, d.stream>>>(static_cast<MoeGroup*>(plan.table), list);
    }
    check(hipGetLastError());
}
// A layer's prestage over its prompt batch, to every way out of it, an error included: the table entries pointed at prestaged copies are emptied
// (the next layer's prestage overwrites the region they point into), the copies and GEMMs that read it drained, and the layer is no longer the
// prestaged one. `end` is the way out without an error, the destructor the way with one (each step by itself, so that one that fails does not skip
// the others; the failure is the caller's, already thrown).
struct PrestageUse {
    Device& d; ResidentMoe& plan; const bool active; std::vector<size_t> pointed; bool ended = false;
    PrestageUse(Device& device, ResidentMoe& layer, bool wanted) : d(device), plan(layer), active(wanted) {}
    void point(const std::vector<size_t>& experts) {
        pointed = experts;
        hybrid_point_staged(d, plan, pointed.data(), pointed.size(), false);
    }
    void end() {
        if (!active) return;
        if (!pointed.empty()) hybrid_point_staged(d, plan, pointed.data(), pointed.size(), true);
        d.prestage.plan = nullptr;
        ended = true;
    }
    ~PrestageUse() {
        if (!active || ended) return;
        try { if (!pointed.empty()) hybrid_point_staged(d, plan, pointed.data(), pointed.size(), true); } catch (...) {}
        cleanup(hipStreamSynchronize(d.stream));
        cleanup(hipStreamSynchronize(d.copy_stream));
        d.prestage.plan = nullptr;
    }
};
#endif
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
            for (size_t role=0;role<3;++role) list.layout[i][role]=slot_soa_layout(d,plan,list.base[i],role);
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
        uint8_t* base = d.slots.base + d.slots.at[slot];
        loaded[plan].emplace_back(expert, base);
        plan->slot_of[expert] = int32_t(slot);
        plan->weights[expert * 3].data = base;
    }
    for (const auto& item : loaded) hybrid_point_list(d, *item.first, item.second);
    d.slots.pending.clear();
    return true;
}

// Every expert leaves the slots and their VRAM goes; the table says so before the next kernel. Keeps the stage's size and
// its host mirror; restore_expert_slots gives the pool its VRAM back at the size that then fits.
void release_expert_slots(Device& d) {
    if (!d.slots.base) return;
    hybrid_settle(d, true);
    for (auto& item : d.resident) {
        if (!item.second->hybrid) continue;
        ResidentMoe& plan = *item.second;
        std::vector<std::pair<size_t, const uint8_t*>> absent;
        for (size_t expert = 0; expert < plan.experts; ++expert)
            if (plan.slot_of[expert] >= 0) { absent.emplace_back(expert, nullptr); plan.slot_of[expert] = -1; }
        hybrid_point_list(d, plan, absent);
    }
    check(hipStreamSynchronize(d.stream));
    if (d.copy_stream) check(hipStreamSynchronize(d.copy_stream));
    check(hipFree(d.slots.base));
    d.prestage.plan = nullptr; d.prestage.experts.clear();   // (the region went with the pool)
    d.used -= d.slots.arena + d.slots.stage * d.slots.bytes;
    d.slots.base = nullptr;
    d.slots.cache = 0;
    d.slots.arena = 0;
    d.slots.soa_scratch=0;
    d.slots.at.clear(); d.slots.group.clear();
    d.slots.owner.clear();
}
// The end of the message of a growth or a pool that found no memory (crates/straydog/src/worker.rs: not fatal).
constexpr const char* kv_no_memory = "not enough host memory for the K/V of this sequence";
// The same for a K/V held in VRAM (sd_kv_grow): the device has no room for the longer one beside what it holds.
constexpr const char* kv_no_vram = "not enough VRAM for the K/V of this sequence";
void restore_expert_slots(Device& d, size_t cache) {
    if (d.slots.base || !d.slots.bytes) throw std::runtime_error("invalid Straydog expert slot pool");
    size_t bytes = (cache + d.slots.stage) * d.slots.bytes;
    if (bytes > d.budget - d.used) throw std::runtime_error("expert slot pool exceeds the Straydog VRAM budget");
    // The budget was measured at the start: other programs may hold what it counted by now, and a sequence must not die of that
    // when the experts that find no slot can run on the CPU. A smaller pool, then, down to the stage alone.
    while (true) {
        const hipError_t error = hipMalloc(reinterpret_cast<void**>(&d.slots.base), bytes);
        if (error == hipSuccess) break;
        if (error != hipErrorOutOfMemory) check(error);
        (void)hipGetLastError();
        d.slots.base = nullptr;
        std::fprintf(stderr, "Straydog: no VRAM for %zu MiB of expert slots\n", bytes >> 20);
        // (not even the stage: the request fails, and the next one asks for the pool again)
        if (!cache) throw std::runtime_error(kv_no_memory);
        cache = cache < 16 ? 0 : cache / 8 * 7;
        bytes = (cache + d.slots.stage) * d.slots.bytes;
    }
    d.used += bytes;
    expert_slots_carve(d, cache * d.slots.bytes);
    // The pool is at another address now, and its arena another size: the staging slots (base + arena, CUDA's assist) that the hybrid
    // layers' captured graphs hold by value are gone, and a replay would read and write freed memory. A K/V growth has dropped them
    // already (sd_kv_grow); a raised budget and a long prompt's end have not. Restores are rare: capturing again costs little.
    kv_clear_graphs(d);
}

// Bounded swaps (max_swaps > 0) never block the compute stream: victims are
// unmapped in stream order, the copy stream loads the new experts after that,
// and hybrid_settle maps them once loaded. A refresh in flight skips the round.
void hybrid_rebalance(Device& d, size_t max_swaps, bool wait) {
    if (!d.slots.base || !hybrid_settle(d, wait)) return;
    struct Entry { ResidentMoe* plan; size_t expert; double score; };
    const auto order = [](const Entry& a, const Entry& b) {
        return a.score != b.score ? a.score > b.score : a.plan != b.plan ? a.plan < b.plan : a.expert < b.expert;
    };
    const auto weakest = [&](const Entry& a, const Entry& b) { return order(b, a); };
    size_t group_count = 0;
    for (const uint8_t group : d.slots.group) group_count = std::max(group_count, size_t(group) + 1);
    std::vector<std::vector<Entry>> residents(group_count), waiting(group_count);
    std::vector<std::vector<size_t>> free_slots(group_count);
    // Bounded refreshes only need each group's strongest candidates and weakest residents: the `max_swaps` first of each order are kept as
    // the experts are scanned, in that order (the orders are total, so they are the ones a sort of every expert would put first). Every
    // expert listed and then partially sorted took the host 0.15 ms a refresh, and 1.85 ms where its cores are shared and slow (27 plans,
    // 9,720 slots: a verification pass of a pair of RTX 3090 there spent that between a span's end and the next device's start).
    // (a bound above the few hundred a decoding refresh asks for is a prompt's whole re-pick: every expert is listed and sorted, as with no
    // bound. Kept in order as they were scanned, a device's 9,700 residents were 560 MB of moves, 35 ms of a slow host after every prompt.)
    const bool listed = !max_swaps || max_swaps > 512;
    const auto keep = [max_swaps, listed](std::vector<Entry>& kept, const Entry& entry, const auto& before, bool strongest) {
        if (listed) { kept.push_back(entry); return; }
        if (kept.size() == max_swaps) {
            // (most experts fall here, at one comparison of scores: the order's other keys only part equal scores)
            const double last = kept.back().score;
            if (strongest ? entry.score < last : entry.score > last) return;
            if (!before(entry, kept.back())) return;
            kept.pop_back();
        }
        kept.insert(std::upper_bound(kept.begin(), kept.end(), entry, before), entry);
    };
    if (!listed) for (size_t g = 0; g < group_count; ++g) { residents[g].reserve(max_swaps + 1); waiting[g].reserve(max_swaps + 1); }
    for (auto& item : d.resident) {
        if (!item.second->hybrid) continue;
        ResidentMoe& plan = *item.second;
        for (size_t expert = 0; expert < plan.experts; ++expert) {
            if (plan.slot_of[expert] >= 0) keep(residents[plan.slot_group], {&plan, expert, plan.usage[expert]}, weakest, false);
            else if (plan.slot_of[expert] == -1 && plan.usage[expert] > 0) keep(waiting[plan.slot_group], {&plan, expert, plan.usage[expert]}, order, true);
        }
    }
    for (size_t slot = 0; slot < d.slots.cache; ++slot) if (!d.slots.owner[slot].first) free_slots[d.slots.group[slot]].push_back(slot);
    std::vector<Entry> candidates;
    for (size_t g = 0; g < group_count; ++g) {
        candidates.insert(candidates.end(), waiting[g].begin(), waiting[g].end());
        // (a refresh that listed every resident ranks them)
        if (listed) std::sort(residents[g].begin(), residents[g].end(), weakest);
    }
    std::sort(candidates.begin(), candidates.end(), order);
    std::unordered_map<ResidentMoe*, std::vector<std::pair<size_t, const uint8_t*>>> released;
    std::vector<size_t> victim(group_count, 0), next_free(group_count, 0);
    for (const Entry& candidate : candidates) {
        if (max_swaps && d.slots.pending.size() == max_swaps) break;
        const size_t g = candidate.plan->slot_group;
        size_t slot;
        if (next_free[g] < free_slots[g].size()) slot = free_slots[g][next_free[g]++];
        else {
            // Hysteresis: replace only clearly weaker residents.
            if (victim[g] == residents[g].size() || candidate.score <= residents[g][victim[g]].score * 1.25 + 0.05) continue;
            const Entry& old = residents[g][victim[g]++];
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
        copies.push_back({d.slots.owner[slot].first, size_t(d.slots.owner[slot].second), d.slots.base + d.slots.at[slot]});
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

// Whether the slots are refreshed at one of the points where decoding looks (every four tokens: hybrid_rebalance, 32 swaps at the most):
// once 64 routes have missed the slots since the last refresh, or 64 tokens have passed. A refresh moves a waiting expert in only where
// routes missed it, so with few misses it finds little to do, and it is not free: it scans every plan's experts and its swaps copy and
// pack on the device. Where the slots miss many routes (a GPU holding a part of the experts: 46 routes a token on one 26 GiB card, 143 on a
// 16 GiB one) the 64 are there at every look and the refresh is every four tokens, as it always was. Where they hold nearly all (two RTX
// 3090 with every layer's bank page-locked: 99.4% of the routes resident, 3.2 missed a token) a refresh every four tokens took 1.6 ms a
// token of a slow host for 0.3 ms of staging: 100 tok/s with it, 115 with the slots held as they were.
constexpr uint64_t hybrid_refresh_misses = 64, hybrid_refresh_tokens = 64;
inline bool hybrid_refresh_due(Device& d) {
    const uint64_t misses = d.cpu_experts + d.assisted_experts;
    // (the counts may have gone back: a speculative pass's rows put back, a request's statistics cleared)
    if (misses < d.refresh_misses) d.refresh_misses = misses;
    if (d.hybrid_tokens < d.refresh_tokens) d.refresh_tokens = d.hybrid_tokens;
    if (misses - d.refresh_misses < hybrid_refresh_misses && d.hybrid_tokens - d.refresh_tokens < hybrid_refresh_tokens) return false;
    d.refresh_misses = misses; d.refresh_tokens = d.hybrid_tokens;
    return true;
}

// compute(first, count) launches the expert GEMMs for missing[first, first+count); meanwhile() runs on this thread while they
// stream, before they are waited for. The GPU's time from the first copy to the last expert's GEMMs, in ns (0 with none missing).
template<class Compute, class Meanwhile>
uint64_t hybrid_stream_missing(Device& d, ResidentMoe& plan, const std::vector<size_t>& missing, Compute&& compute, Meanwhile&& meanwhile) {
    if (missing.empty()) { meanwhile(); return 0; }
    if (d.copier) d.copier->wait();   // the copy stream and read-ahead state are this thread's again
    struct Span {
        hipEvent_t began = nullptr, ended = nullptr;
        ~Span() { if (began) cleanup(hipEventDestroy(began)); if (ended) cleanup(hipEventDestroy(ended)); }
    } span;
    // What a layer's streamed experts leave behind on every way out of here, an error included: their table entries pointing into the chunks'
    // buffers (the next layer's chunks go through the same buffers, and its GEMMs would read another layer's bytes through a stale entry), and
    // both streams holding copies and GEMMs that read those buffers. `settle` is the way out without an error, the destructor the way with one
    // (each step by itself, so that one that fails does not skip the others; the failure is the caller's, already thrown).
    struct Settle {
        Device& d; ResidentMoe& plan; const std::vector<size_t>& missing; bool settled = false;
        void settle() {
            hybrid_point_experts(d, plan, missing.data(), missing.size(), nullptr);
            check(hipStreamSynchronize(d.copy_stream));
            check(hipStreamSynchronize(d.stream));
            settled = true;
        }
        ~Settle() {
            if (settled) return;
            try { hybrid_point_experts(d, plan, missing.data(), missing.size(), nullptr); } catch (...) {}
            cleanup(hipStreamSynchronize(d.copy_stream));
            cleanup(hipStreamSynchronize(d.stream));
        }
    } streamed{d, plan, missing};
    check(hipEventCreate(&span.began)); check(hipEventCreate(&span.ended));
    check(hipEventRecord(span.began, d.copy_stream));
    prefetch_experts(d, plan, missing.data(), missing.size());
    const size_t half = std::min(d.slots.chunk / 2, hybrid_point_limit);
    // Pageable banks: the CPU workers gather each chunk into the page-locked mirror while the GPU
    // computes the previous one, so cold pages fault in concurrently instead of one copy at a time
    // (cached ones pay an extra pass through host memory); the experts then copy by DMA.
    // Page-locked (registered) banks copy directly.
    const bool gather = d.slots.host && d.cpu && !plan.mapped[0];
    uint64_t gathered_ns = 0;
    StreamEvents events;
    std::vector<uint8_t*> targets;
    for (size_t first = 0, chunk = 0; first < missing.size(); first += half, ++chunk) {
        const size_t count = std::min(half, missing.size() - first), buffer = chunk % 2;
        uint8_t* base = d.slots.base + d.slots.arena + buffer * half * d.slots.bytes;
        uint8_t* host = gather ? d.slots.host + buffer * half * d.slots.bytes : nullptr;
        if (gather) {
            // The DMA two chunks back read this mirror (its `loaded` event follows it on the copy stream).
            if (chunk >= 2) check(hipEventSynchronize(events.loaded[buffer]));
            targets.clear();
            for (size_t i = 0; i < count; ++i) targets.push_back(host + i * d.slots.bytes);
            const auto began = std::chrono::steady_clock::now();
            hybrid_gather(d, plan, missing.data() + first, count, targets.data());
            gathered_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
        }
        if (chunk >= 2) check(hipStreamWaitEvent(d.copy_stream, events.used[buffer], 0));
        // Straight from the banks, a chunk's matrices land role by role, so a run of consecutive experts is one copy a role: three copies
        // an expert, each a third of a megabyte, kept the bus at 19 GB/s (22 so) on a Radeon AI PRO R9700.
        uint8_t* regions[3] = {base, base + half * plan.host_bytes[0], base + half * (plan.host_bytes[0] + plan.host_bytes[1])};
        if (gather) {
            for (size_t i = 0; i < count; ++i)
                check(hipMemcpyAsync(base + i * d.slots.bytes, host + i * d.slots.bytes, expert_slot_bytes(plan), hipMemcpyHostToDevice,
                                     d.copy_stream));
        } else {
            for (size_t i = 0; i < count;) {
                size_t j = i + 1;
                while (j < count && missing[first + j] == missing[first + j - 1] + 1) ++j;
                for (size_t role = 0; role < 3; ++role)
                    check(hipMemcpyAsync(regions[role] + i * plan.host_bytes[role], plan.host[role] + missing[first + i] * plan.host_bytes[role],
                                         (j - i) * plan.host_bytes[role], hipMemcpyHostToDevice, d.copy_stream));
                i = j;
            }
        }
        check(hipEventRecord(events.loaded[buffer], d.copy_stream));
        check(hipStreamWaitEvent(d.stream, events.loaded[buffer], 0));
        phase(d, "moe_stream_wait");
        hybrid_point_experts(d, plan, missing.data() + first, count, base, gather ? nullptr : regions);
        compute(first, count);
        phase(d, "moe_stream_gemm");
        if (fail_streamed_chunk.load() && fail_streamed_chunk.fetch_sub(1) == 1) throw std::runtime_error("Straydog test: a streamed chunk fails");
        check(hipEventRecord(events.used[buffer], d.stream));
        d.streamed_experts += count;
    }
    check(hipEventRecord(span.ended, d.stream));
    meanwhile();
    streamed.settle();
    float ms = 0;
    check(hipEventElapsedTime(&ms, span.began, span.ended));
    // (what a gathered expert cost the CPU's workers: prompt_cpu_share weighs a route against it)
    if (gather && missing.size() >= 16) {
        const double sample = double(gathered_ns) / double(missing.size());
        d.prompt_gather_ns = d.prompt_gather_ns > 0 ? 0.7 * d.prompt_gather_ns + 0.3 * sample : sample;
    }
    return uint64_t(double(ms) * 1e6);
}

// A whole prompt segment uses one layer at a time. Its table may point into a borrowed bank pool only until it is cleared.
void clear_prefill_bank(Device& d) {
    if (d.copy_stream) check(hipStreamSynchronize(d.copy_stream));
    check(hipStreamSynchronize(d.stream));
    if (!d.layer_prefill_plan) return;
    auto& plan = *d.layer_prefill_plan;
    std::vector<std::pair<size_t, const uint8_t*>> absent;
    for (size_t expert = 0; expert < plan.experts; ++expert) absent.emplace_back(expert, nullptr);
    hybrid_point_list(d, plan, absent);
    check(hipStreamSynchronize(d.stream));
    plan.layer_prefill = false;
    d.layer_prefill_plan = nullptr;
    d.layer_prefill_storage = nullptr;
}
void load_prefill_bank(Device& d, size_t layer, void* storage, size_t bytes,
                       const uint8_t* const* sources = nullptr, const uint64_t* source_bytes = nullptr) {
    clear_prefill_bank(d);
    const auto decoder = d.decoder.find(layer);
    const auto allocation = d.allocations.find(storage);
    if (decoder == d.decoder.end() || allocation == d.allocations.end() || bytes > allocation->second || d.qwen_parts != 1)
        throw std::runtime_error("invalid layer prefill bank storage");
    auto& plan = *d.resident.at(decoder->second.moe);
    const size_t entry = expert_slot_bytes(plan), required = plan.experts * entry;
    if (!plan.hybrid || required > bytes || d.slots.cache || !d.slots.host || d.slots.stage < 2 || !d.cpu)
        throw std::runtime_error("layer prefill bank needs empty expert cache and staging");
    if (sources) {
        for (size_t role = 0; role < 3; ++role)
            if (!source_bytes || !sources[role] || source_bytes[role] != plan.host_bytes[role] * plan.experts)
                throw std::runtime_error("invalid copied layer bank source");
    }
    if (d.copier) d.copier->wait();
    std::vector<size_t> experts(plan.experts);
    for (size_t i = 0; i < experts.size(); ++i) experts[i] = i;
#if defined(STRAYDOG_PREFILL_OBSERVER)
    LayerBankTrace trace(d, layer, required);
    const uint64_t hint_started = layer_trace_tick();
#endif
    if (!sources) prefetch_experts(d, plan, experts.data(), experts.size());
#if defined(STRAYDOG_PREFILL_OBSERVER)
    trace.hint_ns += layer_trace_tick() - hint_started;
#endif
    auto* bank = static_cast<uint8_t*>(storage);
    if (!sources && plan.mapped[0]) {
        size_t offset = 0;
#if defined(STRAYDOG_PREFILL_OBSERVER)
        trace.before_copy(0);
#endif
        for (size_t role = 0; role < 3; ++role) {
            check(hipMemcpy2DAsync(bank + offset, entry, plan.host[role], plan.host_bytes[role], plan.host_bytes[role], plan.experts,
                                   hipMemcpyHostToDevice, d.copy_stream));
            offset += plan.host_bytes[role];
        }
#if defined(STRAYDOG_PREFILL_OBSERVER)
        trace.after_copy(0);
#endif
    } else {
        // The same authenticated mapping/file gather as ordinary streaming, overlapped with the previous half's DMA.
        const size_t half = d.slots.chunk / 2;
        StreamEvents events;
        for (size_t first = 0, chunk = 0; first < plan.experts; first += half, ++chunk) {
            const size_t count = std::min(half, plan.experts - first), buffer = chunk % 2;
#if defined(STRAYDOG_PREFILL_OBSERVER)
            const uint64_t wait_started = layer_trace_tick();
#endif
            if (chunk >= 2) check(hipEventSynchronize(events.loaded[buffer]));
#if defined(STRAYDOG_PREFILL_OBSERVER)
            trace.wait_ns += layer_trace_tick() - wait_started;
#endif
            uint8_t* host = d.slots.host + buffer * half * d.slots.bytes;
            std::vector<uint8_t*> targets;
            for (size_t i = 0; i < count; ++i) targets.push_back(host + i * d.slots.bytes);
#if defined(STRAYDOG_PREFILL_OBSERVER)
            const uint64_t gather_started = layer_trace_tick();
#endif
            hybrid_gather(d, plan, experts.data() + first, count, targets.data(), sources);
#if defined(STRAYDOG_PREFILL_OBSERVER)
            trace.gather_ns += layer_trace_tick() - gather_started;
            trace.before_copy(buffer);
            const uint64_t submit_started = layer_trace_tick();
#endif
            for (size_t i = 0; i < count; ++i)
                check(hipMemcpyAsync(bank + (first + i) * entry, targets[i], entry, hipMemcpyHostToDevice, d.copy_stream));
#if defined(STRAYDOG_PREFILL_OBSERVER)
            trace.submit_ns += layer_trace_tick() - submit_started;
            trace.after_copy(buffer);
#endif
            check(hipEventRecord(events.loaded[buffer], d.copy_stream));
        }
#if defined(STRAYDOG_PREFILL_OBSERVER)
        const uint64_t drain_started = layer_trace_tick();
#endif
        check(hipStreamSynchronize(d.copy_stream));
#if defined(STRAYDOG_PREFILL_OBSERVER)
        trace.wait_ns += layer_trace_tick() - drain_started;
#endif
    }
    check(hipStreamSynchronize(d.copy_stream));
    std::vector<std::pair<size_t, const uint8_t*>> loaded;
    for (size_t expert = 0; expert < plan.experts; ++expert) loaded.emplace_back(expert, bank + expert * entry);
    d.layer_prefill_plan = &plan;
    d.layer_prefill_storage = storage;
    hybrid_point_list(d, plan, loaded);
    check(hipStreamSynchronize(d.stream));
    plan.layer_prefill = true;
    d.streamed_experts += plan.experts;
#if defined(STRAYDOG_PREFILL_OBSERVER)
    trace.emit();
#endif
}
