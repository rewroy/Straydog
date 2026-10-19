// Transfers of caller memory. Rust vectors, std::vector storage and the stack are
// pageable: an asynchronous copy from or to them makes the runtime pin those pages
// for the call, and ROCm can reuse such a pin after the pages were unmapped and
// reallocated, so the copy engine reads unmapped memory (an SDMA page fault,
// reported as an illegal memory access). These copies are staged through the
// device's page-locked arena instead. An upload's slot is not reused before the
// stream has passed it: the arena wraps only after a synchronization, and a
// download synchronizes the stream.
// Large staged copies on several threads: one core reads the page cache at ~11 GB/s, four
// at ~18 (a model upload is ~57 GB). The caller copies a slice and waits for the rest.
class StagingCopier {
public:
    static StagingCopier& instance() { static StagingCopier copier; return copier; }
    void copy(void* to, const void* from, size_t bytes) {
        if (bytes < (size_t(4) << 20)) { std::memcpy(to, from, bytes); return; }
        std::unique_lock<std::mutex> lock(mutex_);
        to_ = static_cast<uint8_t*>(to); from_ = static_cast<const uint8_t*>(from); bytes_ = bytes;
        pending_ = workers_; ++generation_;
        lock.unlock();
        wake_.notify_all();
        slice(0);
        lock.lock();
        done_.wait(lock, [&]() { return pending_ == 0; });
    }
    ~StagingCopier() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; ++generation_; }
        wake_.notify_all();
        for (auto& thread : threads_) thread.join();
    }
private:
    static constexpr size_t workers_ = 3;
    StagingCopier() { for (size_t i = 1; i <= workers_; ++i) threads_.emplace_back([this, i]() { run(i); }); }
    void slice(size_t index) {
        const size_t each = (bytes_ / (workers_ + 1) + 63) & ~size_t(63), first = std::min(bytes_, index * each);
        std::memcpy(to_ + first, from_ + first, std::min(bytes_, first + each) - first);
    }
    void run(size_t index) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&]() { return generation_ != seen; });
            seen = generation_;
            if (stopping_) return;
            lock.unlock();
            slice(index);
            lock.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::vector<std::thread> threads_;
    uint8_t* to_ = nullptr; const uint8_t* from_ = nullptr; size_t bytes_ = 0, pending_ = 0;
    uint64_t generation_ = 0; bool stopping_ = false;
};

uint8_t* io_slot(Device& d, size_t bytes) {
    const size_t need = (bytes + 255) & ~size_t(255);
    if (need > io_arena_bytes - d.io_used) { check(hipStreamSynchronize(d.stream)); d.io_used = 0; }
    uint8_t* slot = d.io + d.io_used;
    d.io_used += need;
    return slot;
}
void upload(Device& d, void* device, const void* host, size_t bytes) {
    constexpr size_t half = io_arena_bytes / 2;
    if (bytes <= half) {
        uint8_t* slot = io_slot(d, bytes);
        std::memcpy(slot, host, bytes);
        check(hipMemcpyAsync(device, slot, bytes, hipMemcpyHostToDevice, d.stream));
        return;
    }
    // A large transfer (weights, prefill rows) alternates the arena's halves, so the host
    // fills one while the other is copied.
    check(hipStreamSynchronize(d.stream));
    for (size_t done = 0, piece = 0; done < bytes; ++piece) {
        const size_t part = std::min(bytes - done, half), which = piece % 2;
        if (piece >= 2) check(hipEventSynchronize(d.io_halves[which]));
        uint8_t* slot = d.io + which * half;
        StagingCopier::instance().copy(slot, static_cast<const uint8_t*>(host) + done, part);
        check(hipMemcpyAsync(static_cast<uint8_t*>(device) + done, slot, part, hipMemcpyHostToDevice, d.stream));
        check(hipEventRecord(d.io_halves[which], d.stream));
        done += part;
    }
    d.io_used = io_arena_bytes;   // both halves may still be in flight: the next slot waits for the stream
}
// Waits for the stream: the bytes are in `host` on return.
void download(Device& d, void* host, const void* device, size_t bytes) {
    for (size_t done = 0; done < bytes;) {
        const size_t part = std::min(bytes - done, io_arena_bytes);
        uint8_t* slot = io_slot(d, part);
        check(hipMemcpyAsync(slot, static_cast<const uint8_t*>(device) + done, part, hipMemcpyDeviceToHost, d.stream));
        check(hipStreamSynchronize(d.stream));
        std::memcpy(static_cast<uint8_t*>(host) + done, slot, part);
        d.io_used = 0;   // the stream is idle, so every staged upload has landed
        done += part;
    }
}
