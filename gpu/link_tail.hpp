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

// Eight bytes of this device's inbox as the other device wrote them (never a stale cached line).
__device__ inline uint64_t link_load64(const void* at) {
#if defined(__HIPCC__)
    return __hip_atomic_load(static_cast<const uint64_t*>(at), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *static_cast<const volatile uint64_t*>(at);
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

// Without out, the tail only sends: the last block raises the other's flag (a part link's sum,
// below, whose consumer adds the parts).
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
    uint64_t skew = 0;      // development (sd_link_skew): the last block's waves past the first wait
                            // this many clock ticks before they add
};

// The other's inbox slot this sum's part goes to (the count is the last sum's until the last block
// of this kernel moves it on).
__device__ __forceinline__ float* link_tail_slot(const LinkTail& tail) {
    return tail.peer_inbox + size_t((tail.state[0] + 1) & 1) * tail.capacity;
}

// Every thread of every block, after its stores of the part and their link_send.
__device__ inline void link_tail(const LinkTail& tail) {
    __shared__ uint32_t last, count;
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
    __threadfence();
#endif
    __syncthreads();
    // The sum's count, read by one thread for the whole block before the barrier: thread 0 of the
    // last block stores the next count right after it, and a wave that read the count after that
    // store would add the other's part of the sum before (waves of a block can run far apart on a
    // GPU another process loads).
    if (!threadIdx.x) {
        last = atomicAdd(tail.state + 3, 1u) == gridDim.x * gridDim.y - 1;
        count = tail.state[0];
    }
    __syncthreads();
    if (!last) return;
    if (tail.skew && threadIdx.x >= 32) {
        const uint64_t start = peer_clock();
        while (peer_clock() - start < tail.skew) {
#if defined(__HIPCC__)
            __builtin_amdgcn_s_sleep(8);
#endif
        }
    }
    __threadfence();
    const uint32_t sent = count + 1;
    if (!tail.out) {
        // a part sent as it formed (part_wait's sums): the other adds it as it reads it
        if (!threadIdx.x) {
            tail.state[3] = 0;
            tail.state[0] = sent;
            __threadfence_system();
            peer_store(tail.peer_flag, sent);
        }
        return;
    }
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

// ---- Parts sent as they form (a part link: device_link.hpp, sd_link_parts) ----
// A decode sum of two devices whose producer sends this device's part into the other's inbox as it
// stores it and raises the other's flag (a LinkTail without out), and whose consumer, a later
// kernel, waits for its own flag and adds the two parts as it reads them: own + theirs, the one
// addition sd_op_sum_devices makes (two terms commute exactly). Before reading sum n, a consumer
// also waits until the other has read this device's sum n - 1: the producer of n + 1, after it in
// the stream, writes that sum's slot of the other's inbox. Each consumer counts its read of sum n
// into the other's state once all its blocks have read.
struct PartRead {
    const float* inbox;     // this device's inbox: two slots of capacity floats, the other writes
    uint32_t* state;        // this device's: [0] sums sent, [1] its flag, [2] a failed wait, [3] the
                            // producer's blocks in, [4] the other's reads of its sums, [5] the
                            // consumer's blocks done reading
    uint32_t* peer_reads;   // the other's state[4]
    uint32_t capacity;
    uint64_t timeout;
};

// One thread of a consumer's block: the count of the sum to read (its slot is count & 1), once
// the other's part is in and the other has read this device's sum before. A wait that gives up,
// or any wait after one did, records the failure and goes on (the pass's results are void).
__device__ inline uint32_t part_wait(const PartRead& link) {
    const uint32_t n = *static_cast<volatile uint32_t*>(link.state);
    const uint64_t start = peer_clock();
    // Both flags are read together and relaxed, and the acquire is taken once, after the wait: a load of each with its own acquire (peer_load)
    // made two round trips and two cache invalidations in series at the start of every consumer block.
    for (;;) {
        const uint32_t sent = peer_flag_load(link.state + 1), read = peer_flag_load(link.state + 4);
        if (int32_t(sent - n) >= 0 && int32_t(read - (n - 1)) >= 0) break;
        if (*static_cast<volatile uint32_t*>(link.state + 2) || peer_clock() - start > link.timeout) {
            link.state[2] = n;
            break;
        }
#if defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#endif
    }
#if defined(__HIPCC__)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#else
    __threadfence_system();
#endif
    return n;
}

// One thread of a consumer's block, after every thread's reads of the inbox returned: the last
// block of the grid tells the other this sum is read.
__device__ inline void part_read_done(const PartRead& link, uint32_t n) {
    if (atomicAdd(link.state + 5, 1u) != gridDim.x * gridDim.y - 1) return;
    link.state[5] = 0;
    peer_store(link.peer_reads, n);
}

// A part link's channel of a device (two linked devices), as its kernels take it.
inline PartLink& part_link(Device& d, int channel) {
    if (!d.part_peer || channel < 0 || channel >= int(sizeof(d.part_links) / sizeof(d.part_links[0])))
        throw std::runtime_error("no Straydog part link");
    return d.part_links[channel];
}
inline LinkTail part_send_tail(Device& d, int channel) {
    PartLink& link = part_link(d, channel);
    return LinkTail{nullptr, 0, link.state, link.peer_state + 1, link.peer_inbox, link.inbox, uint32_t(link.capacity), 0,
                    d.await_timeout, nullptr};
}
inline PartRead part_read(Device& d, int channel) {
    PartLink& link = part_link(d, channel);
    return PartRead{link.inbox, link.state, link.peer_state + 4, uint32_t(link.capacity), d.await_timeout};
}

// A part's columns [first, first + count) of `rows` rows (`stride` floats apart) into the other's
// inbox at their places, then its flag raised (after the kernels forming the part, when they could
// not send it themselves): link_copy_kernel and link_signal_kernel's steps without the wait.
__global__ void part_copy_kernel(const float* part, size_t rows, size_t stride, size_t first, size_t count, float* peer_inbox,
                                 const uint32_t* state, size_t capacity) {
    float* slot = peer_inbox + size_t((state[0] + 1) & 1) * capacity;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * count; i += size_t(gridDim.x) * blockDim.x) {
        const size_t at = (i / count) * stride + first + i % count;
        link_send(slot + at, part[at]);
    }
}
__global__ void part_signal_kernel(uint32_t* state, uint32_t* peer_flag) {
    const uint32_t sent = state[0] + 1;
    state[0] = sent;
    __threadfence_system();
    peer_store(peer_flag, sent);
}
inline void part_send(Device& d, int channel, const float* part, size_t rows, size_t stride, size_t first, size_t count) {
    const LinkTail tail = part_send_tail(d, channel);
    // (the part's end in floats, (rows - 1) * stride + first + count, without wrapping)
    uint64_t end = 0;
    if (!rows || !count || mul_overflows(uint64_t(rows - 1), uint64_t(stride), &end) ||
        add_overflows(end, uint64_t(first), &end) || add_overflows(end, uint64_t(count), &end) || end > tail.capacity)
        throw std::runtime_error("a part exceeds its inbox");
    const unsigned blocks = unsigned(std::min<size_t>((rows * count + 255) / 256, 1024));
    part_copy_kernel<<<blocks, 256, 0, d.stream>>>(part, rows, stride, first, count, tail.peer_inbox, tail.state, tail.capacity);
    check(hipGetLastError());
    part_signal_kernel<<<1, 1, 0, d.stream>>>(tail.state, tail.peer_flag);
    check(hipGetLastError());
}
