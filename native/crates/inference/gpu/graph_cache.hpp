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
        check(hipGraphLaunch(found->second.executable,d.stream));
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
