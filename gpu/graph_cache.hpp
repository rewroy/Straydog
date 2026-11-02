// A graph's kernels written straight to the stream's own queue (RDNA4): HIP's packets carry a system-scope acquire, which the
// command processor turns into an invalidation of every shader cache before each kernel (2.5-3.6 us a kernel boundary against
// 1.35-1.45). A kernel that invalidates the vector L0 itself (SD_KERNEL_ACQUIRE first in its body: global_inv scope:SCOPE_DEV) needs
// no acquire: its code and its kernel arguments, which do not change in a graph, stay cached, and the build has every uniform
// read of global memory go through the vector path (-amdgpu-scalarize-global-loads=false), so nothing else it reads waits in the
// scalar cache. A kernel without it keeps HIP's acquire: each packet's code is read to see which. HIP knows nothing of these
// packets; a HIP kernel before them keeps them after what HIP has queued, and one after them holds the stream's place, so what HIP
// then queues or waits for comes after them.
#if defined(__HIPCC__)
__global__ void direct_queue_of(uint64_t* out) {
    *out = reinterpret_cast<uint64_t>(__builtin_amdgcn_queue_ptr());
}
__global__ void direct_tail() {}
struct DirectApi {
    uint64_t (*load_write)(const void*) = nullptr; uint64_t (*load_read)(const void*) = nullptr;
    uint64_t (*add_write)(const void*, uint64_t) = nullptr; void (*doorbell)(uint64_t, int64_t) = nullptr;
    hsa_ven_amd_loader_1_00_pfn_t loader{};
    bool ready = false;
    DirectApi() {
        void* lib = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) return;
        load_write = reinterpret_cast<decltype(load_write)>(dlsym(lib, "hsa_queue_load_write_index_relaxed"));
        load_read = reinterpret_cast<decltype(load_read)>(dlsym(lib, "hsa_queue_load_read_index_relaxed"));
        add_write = reinterpret_cast<decltype(add_write)>(dlsym(lib, "hsa_queue_add_write_index_screlease"));
        doorbell = reinterpret_cast<decltype(doorbell)>(dlsym(lib, "hsa_signal_store_screlease"));
        auto table = reinterpret_cast<hsa_status_t (*)(uint16_t, uint16_t, size_t, void*)>(dlsym(lib, "hsa_system_get_major_extension_table"));
        ready = load_write && load_read && add_write && doorbell && table &&
                table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader) == HSA_STATUS_SUCCESS && loader.hsa_ven_amd_loader_query_host_address;
    }
};
struct DirectQueue { uint32_t type, features; DirectPacket* base; uint64_t doorbell; uint32_t size; };
inline DirectApi& direct_api() { static DirectApi api; return api; }
// (the kernel descriptor's code offset, then the first instructions)
inline bool direct_kernel_acquires(Device& d, uint64_t object) {
    const auto known = d.direct_fenced.find(object);
    if (known != d.direct_fenced.end()) return known->second;
    DirectApi& api = direct_api();
    bool fenced = false;
    const void* host = nullptr;
    if (api.loader.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void*>(object), &host) == HSA_STATUS_SUCCESS && host) {
        const int64_t offset = *reinterpret_cast<const int64_t*>(static_cast<const char*>(host) + 16);
        const void* code = nullptr;
        if (api.loader.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void*>(object + offset), &code) == HSA_STATUS_SUCCESS && code) {
            const auto* words = static_cast<const uint32_t*>(code);
            for (size_t i = 0; i + 2 < 32 && !fenced; ++i) fenced = words[i] == 0xEE0AC07Cu && words[i + 1] == 0x00080000u;
        }
    }
    d.direct_fenced.emplace(object, fenced);
    return fenced;
}
inline bool direct_ready(Device& d) {
    if (d.direct_queue) return true;
    DirectApi& api = direct_api();
    if (!d.rdna4 || !api.ready) return false;
    uint64_t* out = nullptr;
    check(hipMalloc(&out, 64));
    direct_queue_of<<<1, 1, 0, d.stream>>>(out);
    check(hipGetLastError());
    uint64_t queue = 0;
    check(hipMemcpyAsync(&queue, out, 8, hipMemcpyDeviceToHost, d.stream));
    check(hipStreamSynchronize(d.stream));
    check(hipFree(out));
    d.direct_queue = reinterpret_cast<void*>(queue);
    return d.direct_queue != nullptr;
}
// A launch of a graph through HIP, its kernels read back from the queue's ring. (The ring's bodies stay until it wraps.) Empty when
// the launch waits for a signal: a graph with a copy off this queue stays HIP's.
inline std::vector<DirectPacket> direct_window(Device& d, hipGraphExec_t exec) {
    auto* q = static_cast<DirectQueue*>(d.direct_queue);
    DirectApi& api = direct_api();
    const uint64_t before = api.load_write(q);
    check(hipGraphLaunch(exec, d.stream));
    const uint64_t after = api.load_write(q);
    std::vector<DirectPacket> packets;
    for (uint64_t i = before; i < after; ++i) {
        DirectPacket p = q->base[i & (q->size - 1)];
        if ((p.header & 255) == 3 || !p.object || !p.gx) {
            // (a barrier that waits for no signal orders nothing the replay's own barrier bits do not)
            const auto* words = reinterpret_cast<const uint64_t*>(&p);
            for (size_t word = 1; word < 6; ++word)
                if (words[word]) return {};
            continue;
        }
        p.signal = 0;
        p.setup = p.setup ? p.setup : 3;
        const int acquire = (d.direct_mode != 2 && direct_kernel_acquires(d, p.object)) ? 0 : 2;
        p.header = uint16_t(2 | (1 << 8) | (acquire << 9) | (1 << 11));
        packets.push_back(p);
    }
    return packets;
}
inline bool direct_replay(Device& d, const std::vector<DirectPacket>& packets) {
    auto* q = static_cast<DirectQueue*>(d.direct_queue);
    DirectApi& api = direct_api();
    // (HIP's kernel first as well: it waits for a copy HIP has queued, which may be off this queue, and invalidates what a copy wrote
    // past this GPU's L2, which the packets' own invalidation does not reach. HIP may return with that kernel not yet in the queue, held
    // behind such a copy: then only HIP can keep the order, and the graph is left to it.)
    const uint64_t queued = api.load_write(q);
    direct_tail<<<1, 1, 0, d.stream>>>();
    if (api.load_write(q) == queued) return false;
    const uint64_t n = packets.size(), first = api.add_write(q, n);
    while (first + n - api.load_read(q) >= q->size) {}
    for (uint64_t k = 0; k < n; ++k)
        std::memcpy(reinterpret_cast<char*>(&q->base[(first + k) & (q->size - 1)]) + 4, reinterpret_cast<const char*>(&packets[k]) + 4, 60);
    for (uint64_t k = 0; k < n; ++k)
        __atomic_store_n(reinterpret_cast<uint32_t*>(&q->base[(first + k) & (q->size - 1)]),
                         uint32_t(packets[k].header) | (uint32_t(packets[k].setup) << 16), __ATOMIC_RELEASE);
    api.doorbell(q->doorbell, int64_t(first + n - 1));
    direct_tail<<<1, 1, 0, d.stream>>>();
    check(hipGetLastError());
    return true;
}
#endif
// Launch a cached graph: through HIP twice, the second to see that its packets are the first's (their arguments the graph's own, not
// a launch's), then straight to the queue.
inline void graph_launch(Device& d, CachedGraph& g) {
#if defined(__HIPCC__)
    if (d.direct_mode < 0) { const char* mode = std::getenv("STRAYDOG_DIRECT"); d.direct_mode = mode ? std::atoi(mode) : 1; }
    if (d.direct_mode && g.direct_state < 2) {
        if (direct_ready(d)) {
            auto seen = direct_window(d, g.executable);
            if (g.direct_state == 0) { g.direct = std::move(seen); g.direct_state = 1; return; }
            bool same = seen.size() == g.direct.size() && !seen.empty();
            for (size_t i = 0; same && i < seen.size(); ++i) same = seen[i].args == g.direct[i].args && seen[i].object == g.direct[i].object;
            g.direct_state = same ? 2 : 3;
            return;
        }
        g.direct_state = 3;
    }
    if (g.direct_state == 2 && direct_replay(d, g.direct)) return;
#endif
    check(hipGraphLaunch(g.executable, d.stream));
}
// Graphs own no model state: all referenced allocations are immutable weights or
// existing bounded workspaces. Eviction invalidates a graph before freeing data.
template<class Function>
void run_graph(Device& d, const char* tag, void* const* dependencies, size_t count,
                size_t variant, Function&& submit) {
    if (!d.graph_capture) { submit(); return; }
    std::string key(tag);
    key.append(reinterpret_cast<const char*>(&variant),sizeof(variant));
    key.append(reinterpret_cast<const char*>(&d.half_gemm),sizeof(d.half_gemm));
    key.append(reinterpret_cast<const char*>(&d.gemm_min_batch),sizeof(d.gemm_min_batch));
    key.append(reinterpret_cast<const char*>(dependencies),count*sizeof(void*));
    auto found=d.graphs.find(key);
    if (found!=d.graphs.end()) {
        graph_launch(d,found->second);
        ++d.graph_replays;
        return;
    }
    if (d.graphs.size()>=256) { submit(); return; }
    CachedGraph cached{};
    cached.dependencies.assign(dependencies,dependencies+count);
    bool capturing=false;
    try {
        check(hipStreamBeginCapture(d.stream,hipStreamCaptureModeThreadLocal)); capturing=true;
        submit();
        const auto ended=hipStreamEndCapture(d.stream,&cached.graph); capturing=false;
        check(ended);
        check(hipGraphInstantiate(&cached.executable,cached.graph,nullptr,nullptr,0));
        check(hipGraphLaunch(cached.executable,d.stream));
        d.graphs.emplace(std::move(key),std::move(cached));
    } catch (...) {
        if (capturing) {
            hipGraph_t unfinished=nullptr;
            const auto ended=hipStreamEndCapture(d.stream,&unfinished);
            if (unfinished) cleanup(hipGraphDestroy(unfinished));
            if (ended!=hipSuccess) std::fprintf(stderr,"Straydog graph capture cancelled\n");
        }
        if (cached.executable) cleanup(hipGraphExecDestroy(cached.executable));
        if (cached.graph) cleanup(hipGraphDestroy(cached.graph));
        throw;
    }
}
