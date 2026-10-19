// Transfers of caller memory. Python arrays, std::vector storage and the stack are
// pageable: an asynchronous copy from or to them makes the runtime pin those pages
// for the call, and ROCm can reuse such a pin after the pages were unmapped and
// reallocated, so the copy engine reads unmapped memory (an SDMA page fault,
// reported as an illegal memory access). These copies are staged through the
// device's page-locked arena instead. An upload's slot is not reused before the
// stream has passed it: the arena wraps only after a synchronization, and a
// download synchronizes the stream.
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
        std::memcpy(slot, static_cast<const uint8_t*>(host) + done, part);
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
