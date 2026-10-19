// Two GPUs on the same layers. A device writes rows into its peer's
// inbox, then raises the peer's flag; the peer waits for the flag before reading.
// Each side counts the messages it sent and received in its own memory, so graph
// replays need no host reset: message n raises the flag to n, and its reader waits
// for n. The flag and inbox live in the receiving device's VRAM when the devices
// have peer access, else in page-locked host memory both devices map.
__device__ inline uint64_t peer_clock() {
#if defined(__HIPCC__)
    return wall_clock64();
#else
    uint64_t now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); return now;
#endif
}
__device__ inline uint32_t peer_load(uint32_t* flag) {
#if defined(__HIPCC__)
    return __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    const uint32_t value = *static_cast<volatile uint32_t*>(flag);
    __threadfence_system();
    return value;
#endif
}
__device__ inline void peer_store(uint32_t* flag, uint32_t value) {
#if defined(__HIPCC__)
    __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    __threadfence_system();
    *static_cast<volatile uint32_t*>(flag) = value;
#endif
}
// rows x width floats into the peer's inbox (from rows `stride` floats apart).
__global__ void peer_copy(float* to, const float* from, size_t rows, size_t width, size_t stride) {
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < rows * width; i += size_t(gridDim.x) * 256)
        to[i] = from[(i / width) * stride + i % width];
}
// Raises the peer's flag once everything this device wrote before it is visible to the peer:
// the system-scope release here publishes the writes of the earlier kernels in the stream,
// so producers need no fence of their own (a fence per thread writes the cache back per wave).
__global__ void peer_signal(uint32_t* counters, uint32_t* peer_flag) {
    if (threadIdx.x) return;
    __threadfence_system();
    const uint32_t sent = counters[0] + 1;
    counters[0] = sent;
    peer_store(peer_flag, sent);
}
// Waits for the next message. After `timeout` clock ticks it records the failure for
// the host and lets the stream go on (its results are void) instead of hanging the GPU.
__global__ void peer_wait(uint32_t* counters, uint32_t* flag, uint64_t timeout) {
    if (threadIdx.x) return;
    const uint32_t expected = counters[1] + 1;
    counters[1] = expected;
    const uint64_t start = peer_clock();
    while (int32_t(peer_load(flag) - expected) < 0) {
        if (peer_clock() - start > timeout) { counters[2] = expected; break; }
#if defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    __threadfence_system();
}

inline float* peer_inputs(PeerLink& link) { return link.inbox; }
inline float* peer_partials(PeerLink& link) { return link.inbox + link.rows * 2560; }
inline float* peer_remote_partials(PeerLink& link) { return link.peer_inbox + link.rows * 2560; }
// `count` rows of `width` floats (`stride` apart) into the peer's input area at `offset` floats.
void peer_send(Device& d, const float* rows, size_t count, size_t width, size_t stride, size_t offset) {
    if (!d.link.peer || offset + count * width > d.link.rows * 2560) throw std::runtime_error("invalid Straydog peer transfer");
    peer_copy<<<unsigned(std::min<size_t>((count * width + 255) / 256, 1024)), 256, 0, d.stream>>>(
        d.link.peer_inbox + offset, rows, count, width, stride);
    peer_signal<<<1, 32, 0, d.stream>>>(d.link.counters, d.link.peer_flag);
}
void peer_send_rows(Device& d, const float* rows, size_t count, size_t stride) { peer_send(d, rows, count, 2560, stride, 0); }
void peer_signal_peer(Device& d) { peer_signal<<<1, 32, 0, d.stream>>>(d.link.counters, d.link.peer_flag); }
void peer_receive(Device& d) { peer_wait<<<1, 32, 0, d.stream>>>(d.link.counters, d.link.flag, d.await_timeout); }

// Runs work with `ordinal` as the current device, then restores the caller's.
template <class Function>
void on_device(int ordinal, Function&& work) {
    int current = 0;
    check(hipGetDevice(&current));
    if (current != ordinal) check(hipSetDevice(ordinal));
    try { work(); }
    catch (...) { if (current != ordinal) cleanup(hipSetDevice(current)); throw; }
    if (current != ordinal) check(hipSetDevice(current));
}

// After both streams are idle: a wait that gave up voids the results and leaves the
// counters out of step, so both links restart from zero and the call fails.
void peer_settle(Device& d) {
    Device* peer = d.link.peer;
    if (!peer) return;
    uint32_t failed[2] = {0, 0};
    check(hipMemcpy(&failed[0], d.link.counters + 2, 4, hipMemcpyDeviceToHost));
    on_device(peer->ordinal, [&]() { check(hipMemcpy(&failed[1], peer->link.counters + 2, 4, hipMemcpyDeviceToHost)); });
    if (!failed[0] && !failed[1]) return;
    auto reset = [](Device& side) {
        on_device(side.ordinal, [&]() {
            check(hipMemset(side.link.counters, 0, 16));
            if (side.link.direct) check(hipMemset(side.link.flag, 0, 4));
            else *static_cast<volatile uint32_t*>(side.link.flag) = 0;
        });
    };
    reset(d); reset(*peer);
    throw std::runtime_error("a Straydog GPU waited too long for its peer");
}

// Link two devices for expert parallelism with inboxes of `rows` rows each way.
void attach_peer(Device& a, Device& b, size_t rows) {
    if (&a == &b || a.link.peer || b.link.peer || !rows || rows > 4096)
        throw std::runtime_error("invalid Straydog peer link");
    int ab = 0, ba = 0;
    check(hipDeviceCanAccessPeer(&ab, a.ordinal, b.ordinal));
    check(hipDeviceCanAccessPeer(&ba, b.ordinal, a.ordinal));
    const bool direct = ab && ba;
    const size_t inbox_bytes = rows * 2560 * 2 * 4;
    auto enable = [](int from, int to) {
        on_device(from, [&]() {
            const hipError_t state = hipDeviceEnablePeerAccess(to, 0);
            if (state == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
            else check(state);
        });
    };
    if (direct) { enable(a.ordinal, b.ordinal); enable(b.ordinal, a.ordinal); }
    for (Device* side : {&a, &b}) {
        on_device(side->ordinal, [&]() {
            if (direct) {
                if (inbox_bytes + 256 > side->budget - side->used) throw std::runtime_error("Straydog peer inbox exceeds the VRAM budget");
                check(hipMalloc(reinterpret_cast<void**>(&side->link.inbox), inbox_bytes));
                check(hipMalloc(reinterpret_cast<void**>(&side->link.flag), 256));
                check(hipMemset(side->link.flag, 0, 256));
                side->used += inbox_bytes + 256;
            } else {
                check(hipHostMalloc(&side->link.host, inbox_bytes + 256,
                                    hipHostMallocMapped | hipHostMallocPortable));
                std::memset(side->link.host, 0, inbox_bytes + 256);
                side->link.flag = static_cast<uint32_t*>(side->link.host);
                side->link.inbox = reinterpret_cast<float*>(static_cast<uint8_t*>(side->link.host) + 256);
            }
            check(hipMalloc(reinterpret_cast<void**>(&side->link.counters), 256));
            check(hipMemset(side->link.counters, 0, 256));
            side->link.rows = rows;
            side->link.direct = direct;
        });
    }
    a.link.peer = &b; b.link.peer = &a;
    a.link.index = 0; b.link.index = 1;
    a.link.peer_inbox = b.link.inbox; b.link.peer_inbox = a.link.inbox;
    a.link.peer_flag = b.link.flag; b.link.peer_flag = a.link.flag;
}
