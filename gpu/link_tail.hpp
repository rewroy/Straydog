// A tensor-parallel linked sum (device_link.hpp) folded into the end of the kernel that forms this
// device's part of it, so that a decode step saves the sum's launch: each block, as it stores its
// share of the part, also stores it into the other device's inbox at system scope (link_send),
// waits for those stores (an agent-scope release) and counts itself in; the last block in raises
// the other's flag with a system-scope release, waits for its own flag and adds the two parts:
// link_sum_kernel's steps and arithmetic, its count, flag and inbox slots. Included in the GPU TU
// after peer.hpp (peer_load, peer_store, peer_clock).

// A value of this device's inbox as the other device wrote it (never a stale cached line).
__device__ inline float link_load(const float* value) {
#if defined(__HIPCC__)
    return __hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *static_cast<const volatile float*>(value);
#endif
}

// A value of a part into the other device's inbox.
__device__ __forceinline__ void link_send(float* at, float value) {
#if defined(__HIPCC__)
    __hip_atomic_store(at, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    *static_cast<volatile float*>(at) = value;
#endif
}

struct LinkTail {
    const float* part;      // this device's part of the sum (count floats), as the kernel stores it
    uint32_t count;
    uint32_t* state;        // this device's sums: [0] sent, [1] its flag (the other raises it), [2] a
                            // failed wait, [3] the blocks in of the kernel that runs it
    uint32_t* peer_flag;
    float* peer_inbox;
    const float* inbox;
    uint32_t capacity;
    int accumulate;
    uint64_t timeout;
    float* out;
};

// The other's inbox slot this sum's part goes to (the count is the last sum's until the last block
// of this kernel moves it on).
__device__ __forceinline__ float* link_tail_slot(const LinkTail& tail) {
    return tail.peer_inbox + size_t((tail.state[0] + 1) & 1) * tail.capacity;
}

// Every thread of every block, after its stores of the part and their link_send.
__device__ inline void link_tail(const LinkTail& tail) {
    __shared__ uint32_t last;
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
    __threadfence();
#endif
    __syncthreads();
    if (!threadIdx.x) last = atomicAdd(tail.state + 3, 1u) == gridDim.x * gridDim.y - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const uint32_t sent = tail.state[0] + 1;
    const size_t slot = size_t(sent & 1) * tail.capacity;
    if (!threadIdx.x) {
        tail.state[3] = 0;
        tail.state[0] = sent;
        __threadfence_system();
        peer_store(tail.peer_flag, sent);
        const uint64_t start = peer_clock();
        while (int32_t(peer_load(tail.state + 1) - sent) < 0) {
            if (peer_clock() - start > tail.timeout) { tail.state[2] = sent; break; }
#if defined(__HIPCC__)
            __builtin_amdgcn_s_sleep(1);
#endif
        }
    }
    __syncthreads();
    // A few values a thread, each one's loads issued before any add (one block adds them all).
    constexpr unsigned each = 8;
    for (size_t first = 0; first < tail.count; first += size_t(each) * blockDim.x) {
        float theirs[each], mine[each], old[each];
        #pragma unroll
        for (unsigned j = 0; j < each; ++j) {
            const size_t i = first + j * blockDim.x + threadIdx.x;
            if (i >= tail.count) continue;
            theirs[j] = link_load(tail.inbox + slot + i);
#if defined(__HIPCC__)
            mine[j] = __hip_atomic_load(tail.part + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
            mine[j] = *static_cast<const volatile float*>(tail.part + i);
#endif
            old[j] = tail.accumulate ? tail.out[i] : 0.0f;
        }
        #pragma unroll
        for (unsigned j = 0; j < each; ++j) {
            const size_t i = first + j * blockDim.x + threadIdx.x;
            if (i >= tail.count) continue;
            const float sum = mine[j] + theirs[j];
            tail.out[i] = tail.accumulate ? old[j] + sum : sum;
        }
    }
}
