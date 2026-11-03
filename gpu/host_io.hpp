// Transfers of caller memory. Rust vectors, std::vector storage and the stack are
// pageable: an asynchronous copy from or to them makes the runtime pin those pages
// for the call, and ROCm can reuse such a pin after the pages were unmapped and
// reallocated, so the copy engine reads unmapped memory (an SDMA page fault,
// reported as an illegal memory access). These copies are staged through the
// device's page-locked arena instead. An upload's slot is not reused before the
// stream has passed it: the arena wraps only after a synchronization, and a
// download synchronizes the stream.
// Large staged copies on several threads: one core reads the page cache at ~11 GB/s, four
// at ~18 (a model upload is ~57 GB). The caller copies a slice and waits for the rest; callers
// on several threads (uploads to several devices at once) take turns.
class StagingCopier {
public:
    static StagingCopier& instance() { static StagingCopier copier; return copier; }
    void copy(void* to, const void* from, size_t bytes) {
        if (bytes < (size_t(4) << 20)) { std::memcpy(to, from, bytes); return; }
        std::lock_guard<std::mutex> turn(caller_);
        std::unique_lock<std::mutex> lock(mutex_);
        to_ = static_cast<uint8_t*>(to); from_ = static_cast<const uint8_t*>(from); bytes_ = bytes; items_ = nullptr;
        pending_ = workers_; ++generation_;
        lock.unlock();
        wake_.notify_all();
        slice(0);
        lock.lock();
        done_.wait(lock, [&]() { return pending_ == 0; });
    }
    // Several copies (to, from, bytes), each taken whole by one of the threads in turn: the pieces of a read-back.
    struct Item { void* to; const void* from; size_t bytes; };
    void copy_items(const Item* items, size_t count) {
        size_t total = 0;
        for (size_t i = 0; i < count; ++i) total += items[i].bytes;
        if (count < 2 || total < (size_t(1) << 20)) { for (size_t i = 0; i < count; ++i) std::memcpy(items[i].to, items[i].from, items[i].bytes); return; }
        std::lock_guard<std::mutex> turn(caller_);
        std::unique_lock<std::mutex> lock(mutex_);
        items_ = items; count_ = count;
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
        if (items_) {
            for (size_t i = index; i < count_; i += workers_ + 1) std::memcpy(items_[i].to, items_[i].from, items_[i].bytes);
            return;
        }
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
    std::mutex caller_, mutex_;
    std::condition_variable wake_, done_;
    std::vector<std::thread> threads_;
    uint8_t* to_ = nullptr; const uint8_t* from_ = nullptr; size_t bytes_ = 0, pending_ = 0;
    const Item* items_ = nullptr; size_t count_ = 0;
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
// Waits for the stream: the bytes are in `host` on return. A large read lands in the arena's halves in turn, each half
// copied out by the staging threads while the next lands (one thread copying a piece out after another piece's wait
// read back a checkpoint's rows at ~6 GB/s).
void download(Device& d, void* host, const void* device, size_t bytes) {
    constexpr size_t half = io_arena_bytes / 2;
    if (bytes <= half) {
        uint8_t* slot = io_slot(d, bytes);
        check(hipMemcpyAsync(slot, device, bytes, hipMemcpyDeviceToHost, d.stream));
        check(hipStreamSynchronize(d.stream));
        StagingCopier::instance().copy(host, slot, bytes);
        d.io_used = 0;   // the stream is idle, so every staged upload has landed
        return;
    }
    check(hipStreamSynchronize(d.stream));   // (the staged uploads have landed: both halves are free)
    const size_t pieces = (bytes + half - 1) / half;
    const auto piece_bytes = [&](size_t piece) { return std::min(half, bytes - piece * half); };
    const auto queue = [&](size_t piece) {
        check(hipMemcpyAsync(d.io + (piece % 2) * half, static_cast<const uint8_t*>(device) + piece * half, piece_bytes(piece),
                             hipMemcpyDeviceToHost, d.stream));
        check(hipEventRecord(d.io_halves[piece % 2], d.stream));
    };
    queue(0);
    if (pieces > 1) queue(1);
    for (size_t piece = 0; piece < pieces; ++piece) {
        check(hipEventSynchronize(d.io_halves[piece % 2]));
        StagingCopier::instance().copy(static_cast<uint8_t*>(host) + piece * half, d.io + (piece % 2) * half, piece_bytes(piece));
        if (piece + 2 < pieces) queue(piece + 2);
    }
    check(hipStreamSynchronize(d.stream));
    d.io_used = 0;
}
// Several device ranges read back with one wait (`download` waits for each): the copies are queued into the staging arena
// together, and the pieces are copied out to the caller's memory on the copier's threads when they add up to enough.
struct Piece { void* host; const void* device; size_t bytes; };
void download_pieces(Device& d, const Piece* pieces, size_t count) {
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) total += (pieces[i].bytes + 255) & ~size_t(255);
    if (total > io_arena_bytes) { for (size_t i = 0; i < count; ++i) download(d, pieces[i].host, pieces[i].device, pieces[i].bytes); return; }
    if (total > io_arena_bytes - d.io_used) { check(hipStreamSynchronize(d.stream)); d.io_used = 0; }
    std::vector<StagingCopier::Item> items(count);
    for (size_t i = 0; i < count; ++i) {
        uint8_t* slot = io_slot(d, pieces[i].bytes);
        check(hipMemcpyAsync(slot, pieces[i].device, pieces[i].bytes, hipMemcpyDeviceToHost, d.stream));
        items[i] = {pieces[i].host, slot, pieces[i].bytes};
    }
    check(hipStreamSynchronize(d.stream));
    StagingCopier::instance().copy_items(items.data(), count);
    d.io_used = 0;   // the stream is idle, so every staged upload has landed
}
// A tracked allocation within the budget; only its tail padding is zeroed.
void* allocate(Device& value, uint64_t bytes) {
    if (!bytes || bytes > value.budget - value.used || allocation_tail > value.budget - value.used - bytes)
        throw std::runtime_error("Straydog VRAM budget exceeded");
    void* pointer = nullptr;
    check(hipMalloc(&pointer, size_t(bytes) + allocation_tail));
    // Track ownership before copying so the destructor also cleans up copy failures.
    try { value.allocations.emplace(pointer, size_t(bytes)); }
    catch (...) { cleanup(hipFree(pointer)); throw; }
    value.used += size_t(bytes) + allocation_tail;
    check(hipMemsetAsync(static_cast<uint8_t*>(pointer) + bytes, 0, allocation_tail, value.stream));
    return pointer;
}
// One host thread's copies into one device's allocations (sd_fill), on the device's stream:
// streams of their own would leave hardware queues behind (ROCm keeps them) that slow the
// device's later work by ~1%. The thread sleeps on an event of its own instead of spinning
// beside the threads that read and hash.
struct Filler {
    Device* device = nullptr;
    hipEvent_t done = nullptr;
    ~Filler() {
        if (device) cleanup(hipSetDevice(device->ordinal));
        if (done) cleanup(hipEventDestroy(done));
    }
};
