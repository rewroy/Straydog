// Asynchronous copies beside compute (docs/DESIGN.md §10): between page-locked host memory and
// device allocations, on two streams of the device's own (uploads, and downloads on the other
// copy direction), ordered against its compute stream by numbered events. A caller fills weights
// it will need soon (experts of a later layer) while the current layer computes, and makes the
// compute stream wait only where it uses them. Included in the GPU TU.

// Ranges page-locked by sd_host_register (the device only reads them) or sd_host_pin (it may also
// write them): transfers use only page-locked memory, which the DMA engines copy directly
// (pageable memory would make a copy synchronous, or fault the engine).
std::mutex registered_mutex;
std::map<uintptr_t, std::pair<size_t, bool>> registered;   // start -> bytes, writable

void check_registered(const void* host, size_t bytes, bool write) {
    const auto at = reinterpret_cast<uintptr_t>(host);
    std::lock_guard<std::mutex> lock(registered_mutex);
    auto found = registered.upper_bound(at);
    if (found == registered.begin()) throw std::runtime_error("Straydog transfer: the host range is not page-locked");
    --found;
    const size_t length = found->second.first;
    if (bytes > length || at - found->first > length - bytes)
        throw std::runtime_error("Straydog transfer: the host range is not page-locked");
    if (write && !found->second.second)
        throw std::runtime_error("Straydog transfer: the host range is page-locked read-only");
}

// Events in record order per stream (both streams run in order, so an event completes after
// every earlier one of its stream). Ids grow from 1 across both streams; an id no longer
// pending has completed.
struct Transfers {
    hipStream_t streams[2] = {nullptr, nullptr};   // uploads, downloads
    std::deque<std::pair<int64_t, hipEvent_t>> pending[3];   // per stream: compute, upload, download
    std::vector<hipEvent_t> spare;
    int64_t next = 1;
    uint64_t copies[2] = {0, 0}, bytes[2] = {0, 0};

    ~Transfers() {
        for (auto& queue : pending)
            for (auto& entry : queue) { cleanup(hipEventSynchronize(entry.second)); cleanup(hipEventDestroy(entry.second)); }
        for (auto event : spare) cleanup(hipEventDestroy(event));
        for (auto stream : streams)
            if (stream) { cleanup(hipStreamSynchronize(stream)); cleanup(hipStreamDestroy(stream)); }
    }

    // Retire the completed events at the front of each stream's queue.
    void reap() {
        for (auto& queue : pending)
            while (!queue.empty()) {
                const hipError_t state = hipEventQuery(queue.front().second);
                if (state == hipErrorNotReady) break;
                check(state);
                spare.push_back(queue.front().second);
                queue.pop_front();
            }
    }

    hipEvent_t* find(int64_t id) {
        for (auto& queue : pending)
            for (auto& entry : queue)
                if (entry.first == id) return &entry.second;
        return nullptr;
    }
};

Transfers& transfers(Device& d) {
    if (!d.transfers) {
        auto value = std::make_shared<Transfers>();
        for (auto& stream : value->streams) check(hipStreamCreate(&stream));
        d.transfers = value;
    }
    return *d.transfers;
}

// Stream 0 computes, 1 uploads, 2 downloads.
hipStream_t transfer_stream_of(Device& d, int which) {
    if (which < 0 || which > 2) throw std::runtime_error("Straydog transfer: no such stream");
    return which == 0 ? d.stream : transfers(d).streams[which - 1];
}
