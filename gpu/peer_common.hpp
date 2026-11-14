#pragma once
// The flag accessors of peer.hpp that the device code of the other units reads (the linked sums of link_tail_common.hpp, the GEMV
// kernels): device functions only.
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

// The peer link's transfers (peer.hpp): each queues a copy or a wait on the device's stream.
void peer_send_rows(Device& d, const float* rows, size_t count, size_t stride);
void peer_signal_peer(Device& d);
void peer_receive_rows(Device& d);
void peer_receive(Device& d);
void peer_settle(Device& d);
