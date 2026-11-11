// Two GPUs on the same layers. A device writes rows into its peer's
// inbox, then raises the peer's flag; the peer waits for the flag before reading.
// Two channels, each with its own flag and counters: rows go owner -> helper on the
// first, partial sums helper -> owner on the second, so a device that is owner of one
// batch and helper of another at the same time keeps the two message streams apart.
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
// A flag as the other device wrote it, without the acquire (peer_load's): the caller orders its later reads with a fence.
__device__ inline uint32_t peer_flag_load(uint32_t* flag) {
#if defined(__HIPCC__)
    return __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *static_cast<volatile uint32_t*>(flag);
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

// Work queued on `stream` instead of the device's own while in scope.
struct StreamScope {
    Device& d; hipStream_t saved;
    StreamScope(Device& device, hipStream_t stream) : d(device), saved(device.stream) { d.stream = stream; }
    ~StreamScope() { d.stream = saved; }
    StreamScope(const StreamScope&) = delete;
    StreamScope& operator=(const StreamScope&) = delete;
};

// WDDM can hold a CUDA stream's queued kernels until the driver is asked about that stream.
// A peer may already be spinning on a flag those kernels will publish, so submit the producer
// promptly. HIP launches do not need this query.
inline void flush_launches(Device& d) {
#if !defined(__HIPCC__)
    const hipError_t state = hipStreamQuery(d.stream);
    if (state != hipSuccess && state != hipErrorNotReady) check(state);
#else
    (void)d;
#endif
}

// Graph capture records a peer wait for later replay; querying that stream during capture is invalid.
inline void flush_if_not_capturing(Device& d) {
#if !defined(__HIPCC__)
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(d.stream, &capture));
    if (capture == cudaStreamCaptureStatusNone) flush_launches(d);
#else
    (void)d;
#endif
}

inline float* peer_inputs(PeerLink& link) { return link.inbox; }
inline float* peer_partials(PeerLink& link) { return link.inbox + link.rows * 2560; }
inline float* peer_remote_partials(PeerLink& link) { return link.peer_inbox + link.rows * 2560; }
// peer_copy and peer_signal in one block, for a decode pass's few rows: each value stored at system scope, the stores
// waited for (an agent-scope release, as link_tail.hpp's sums wait for theirs) before one thread raises the flag.
__global__ void __launch_bounds__(1024) peer_copy_signal(float* to, const float* from, size_t rows, size_t width, size_t stride,
                                                         uint32_t* counters, uint32_t* peer_flag) {
    for (size_t i = threadIdx.x; i < rows * width; i += blockDim.x) {
        const float value = from[(i / width) * stride + i % width];
#if defined(__HIPCC__)
        __hip_atomic_store(to + i, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *static_cast<volatile float*>(to + i) = value;
#endif
    }
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
    __threadfence();
#endif
    __syncthreads();
    if (threadIdx.x) return;
    __threadfence_system();
    const uint32_t sent = counters[0] + 1;
    counters[0] = sent;
    peer_store(peer_flag, sent);
}
// `count` rows of `width` floats (`stride` apart) into the peer's input area at `offset` floats.
void peer_send(Device& d, const float* rows, size_t count, size_t width, size_t stride, size_t offset) {
    if (!d.link.peer || offset + count * width > d.link.rows * 2560) throw std::runtime_error("invalid Straydog peer transfer");
    if (count * width <= 4 * 2560) {
        peer_copy_signal<<<1, 1024, 0, d.stream>>>(d.link.peer_inbox + offset, rows, count, width, stride, d.link.counters,
                                                   d.link.peer_flag);
        return;
    }
    peer_copy<<<unsigned(std::min<size_t>((count * width + 255) / 256, 1024)), 256, 0, d.stream>>>(
        d.link.peer_inbox + offset, rows, count, width, stride);
    peer_signal<<<1, 32, 0, d.stream>>>(d.link.counters, d.link.peer_flag);
}
// Channel offsets: counters (sent, received, gave up) and flag words of the second channel.
constexpr size_t peer_partial_counters = 4, peer_partial_flag = 16;
void peer_send_rows(Device& d, const float* rows, size_t count, size_t stride) { peer_send(d, rows, count, 2560, stride, 0); }
// The helper's partial sums are in the owner's inbox (second channel).
void peer_signal_peer(Device& d) {
    peer_signal<<<1, 32, 0, d.stream>>>(d.link.counters + peer_partial_counters, d.link.peer_flag + peer_partial_flag);
}
// The helper waits for the owner's rows; the owner for the helper's partial sums.
void peer_receive_rows(Device& d) { peer_wait<<<1, 32, 0, d.stream>>>(d.link.counters, d.link.flag, d.await_timeout); }
void peer_receive(Device& d) {
    peer_wait<<<1, 32, 0, d.stream>>>(d.link.counters + peer_partial_counters, d.link.flag + peer_partial_flag,
                                      d.await_timeout);
}

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
// peer_settle's check of both links' counters as read back (this device's, then its peer's).
void peer_counters_check(Device& d, const uint32_t (&failed)[2][8]) {
    Device* peer = d.link.peer;
    if (!peer || (!failed[0][2] && !failed[0][6] && !failed[1][2] && !failed[1][6])) return;
    auto reset = [](Device& side) {
        on_device(side.ordinal, [&]() {
            check(hipMemset(side.link.counters, 0, 32));
            if (side.link.direct) check(hipMemset(side.link.flag, 0, (peer_partial_flag + 1) * 4));
            else for (size_t i : {size_t(0), peer_partial_flag}) static_cast<volatile uint32_t*>(side.link.flag)[i] = 0;
        });
    };
    reset(d); reset(*peer);
    throw std::runtime_error("a Straydog GPU waited too long for its peer");
}
void peer_settle(Device& d) {
    Device* peer = d.link.peer;
    if (!peer) return;
    uint32_t failed[2][8] = {};
    check(hipMemcpy(failed[0], d.link.counters, 32, hipMemcpyDeviceToHost));
    on_device(peer->ordinal, [&]() { check(hipMemcpy(failed[1], peer->link.counters, 32, hipMemcpyDeviceToHost)); });
    peer_counters_check(d, failed);
}

// Link two devices for expert parallelism with inboxes of `rows` rows each way.
void attach_peer(Device& a, Device& b, size_t rows) {
    if (&a == &b || a.link.peer || b.link.peer || !rows || rows > 4096)
        throw std::runtime_error("invalid Straydog peer link");
    int ab = 0, ba = 0;
    check(hipDeviceCanAccessPeer(&ab, a.ordinal, b.ordinal));
    check(hipDeviceCanAccessPeer(&ba, b.ordinal, a.ordinal));
    // STRAYDOG_PEER_DIRECT=0 links them through page-locked host memory even where they reach each other directly: the link two
    // GPUs get on Windows (WDDM), measured on a pair that has the direct one.
    const char* peer_direct = std::getenv("STRAYDOG_PEER_DIRECT");
    const bool direct = ab && ba && !(peer_direct && std::string(peer_direct) == "0");
    // Inputs, partial sums, then the owner's routes (rows x 21 words).
    const size_t inbox_bytes = rows * (2560 * 2 + 21) * 4;
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
    for (Device* side : {&a, &b}) side->link.routes = reinterpret_cast<int32_t*>(side->link.inbox + rows * 2560 * 2);
    a.link.peer_routes = b.link.routes; b.link.peer_routes = a.link.routes;
}
