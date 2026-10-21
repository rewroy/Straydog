// Tensor-parallel partial sums of two devices inside recorded graphs (docs/DESIGN.md §10.3):
// each device holds, in its own VRAM, an inbox of two slots (alternating by sum) and a flag the
// other raises. A sum copies this device's part into the other's inbox slot, then, once every
// share of the copy is stored (a kernel later, or in the one-block kernel of a few rows after each
// thread waited for its stores and a barrier), raises the other's flag to its count of sums with a
// system-scope release, waits for its own flag to reach the same count, and adds the two parts,
// reading the inbox at system scope (never a stale line). Both devices add the same two values
// (addition commutes exactly), so their sums are equal. The counts live in device memory: graph
// replays need no host reset, and a flag left from an earlier sum is below the count a wait
// wants. A device can be at most one sum ahead of the other (its next sum waits for the other's
// part of this one, which the other sends only after its reads of the sum before), so two slots
// keep a part until it is read. A wait that outlasts the device's await timeout records the
// failure and lets the stream go on (its result void; sd_link_failed), so a peer that stops
// errors the pass instead of hanging it; sd_link_reset starts both counts over.
// Included in the GPU TU after peer.hpp (peer_load, peer_store, peer_clock).

// A value of this device's inbox as the other device wrote it (never a stale cached line).
__device__ inline float link_load(const float* value) {
#if defined(__HIPCC__)
    return __hip_atomic_load(value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *static_cast<const volatile float*>(value);
#endif
}

__global__ void link_copy_kernel(const float* part, size_t count, float* peer_inbox, const uint32_t* state, size_t capacity) {
    const size_t slot = size_t((state[0] + 1) & 1) * capacity;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        peer_inbox[slot + i] = part[i];
}

// A larger sum (a prefill pass's rows) in three kernels, so that the add runs on every unit:
// the copy; one thread raising the other's flag and waiting for its own; then the add over the
// whole grid (the slot from the count the wait set).
__global__ void link_signal_kernel(uint32_t* state, uint32_t* peer_flag, uint64_t timeout) {
    const uint32_t sent = state[0] + 1;
    state[0] = sent;
    __threadfence_system();
    peer_store(peer_flag, sent);
    const uint64_t start = peer_clock();
    while (int32_t(peer_load(state + 1) - sent) < 0) {
        if (peer_clock() - start > timeout) { state[2] = sent; break; }
#if defined(__HIPCC__)
        __builtin_amdgcn_s_sleep(1);
#endif
    }
    __threadfence_system();
}
__global__ void link_add_grid_kernel(const float* part, size_t count, const uint32_t* state, const float* inbox, size_t capacity,
                                     int accumulate, float* out) {
    const size_t slot = size_t(state[0] & 1) * capacity;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const float theirs = link_load(inbox + slot + i);
        const float sum = part[i] + theirs;
        out[i] = accumulate ? out[i] + sum : sum;
    }
}

// A sum of at most link_single_floats floats in one kernel of one block: its threads store their
// shares of the part into the other's inbox at system scope and each waits for its own stores
// (an agent-scope release); after the block's barrier, one thread raises the other's flag with a
// system-scope release, so the flag follows every share; then the wait and the add.
constexpr size_t link_single_floats = 8192;
#if defined(__HIPCC__)
__global__ void link_sum_kernel(const float* part, size_t count, uint32_t* state, uint32_t* peer_flag, float* peer_inbox,
                                const float* inbox, size_t capacity, uint64_t timeout, int accumulate, float* out) {
    __shared__ uint32_t sent;
    const uint32_t next = state[0] + 1;
    const size_t slot = size_t(next & 1) * capacity;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
        __hip_atomic_store(peer_inbox + slot + i, part[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (!threadIdx.x) {
        sent = next;
        state[0] = sent;
        __threadfence_system();
        peer_store(peer_flag, sent);
        const uint64_t start = peer_clock();
        while (int32_t(peer_load(state + 1) - sent) < 0) {
            if (peer_clock() - start > timeout) { state[2] = sent; break; }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float theirs = link_load(inbox + slot + i);
        const float sum = part[i] + theirs;
        out[i] = accumulate ? out[i] + sum : sum;
    }
}
#endif

int sd_link_sums(void* first, void* second, uint64_t capacity) {
    try {
        Device& a = device(first);
        Device& b = device(second);
        if (&a == &b || a.sums.peer || b.sums.peer || !capacity || capacity > (uint64_t(1) << 26))
            throw std::runtime_error("invalid Straydog sum link");
        int ab = 0, ba = 0;
        check(hipDeviceCanAccessPeer(&ab, a.ordinal, b.ordinal));
        check(hipDeviceCanAccessPeer(&ba, b.ordinal, a.ordinal));
        if (!ab || !ba) throw std::runtime_error("Straydog tensor-parallel sums need peer access between the GPUs");
        auto enable = [](int from, int to) {
            on_device(from, [&]() {
                const hipError_t state = hipDeviceEnablePeerAccess(to, 0);
                if (state == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
                else check(state);
            });
        };
        enable(a.ordinal, b.ordinal);
        enable(b.ordinal, a.ordinal);
        for (Device* side : {&a, &b}) {
            on_device(side->ordinal, [&]() {
                side->sums.inbox = static_cast<float*>(allocate(*side, capacity * 2 * 4));
                side->sums.state = static_cast<uint32_t*>(allocate(*side, 256));
                check(hipMemset(side->sums.state, 0, 256));
                check(hipDeviceSynchronize());
                side->sums.capacity = size_t(capacity);
            });
        }
        a.sums.peer = &b; b.sums.peer = &a;
        a.sums.peer_inbox = b.sums.inbox; b.sums.peer_inbox = a.sums.inbox;
        a.sums.peer_flag = b.sums.state + 1; b.sums.peer_flag = a.sums.state + 1;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_link_sum(void* handle, const float* part, uint64_t count, float* out, int accumulate) {
    SD_OP_BEGIN
    SumLink& link = value.sums;
    if (!link.peer || !count || count > link.capacity) throw std::runtime_error("invalid Straydog linked sum");
    op_span(value, part, size_t(count) * 4, "a linked sum's part");
    op_span(value, out, size_t(count) * 4, "a linked sum");
#if defined(__HIPCC__)
    if (count <= link_single_floats) {
        link_sum_kernel<<<1, 1024, 0, value.stream>>>(part, size_t(count), link.state, link.peer_flag, link.peer_inbox, link.inbox,
                                                      link.capacity, value.await_timeout, accumulate ? 1 : 0, out);
        check(hipGetLastError());
        return 0;
    }
#endif
    const unsigned blocks = unsigned(std::min<size_t>((size_t(count) + 255) / 256, 1024));
    link_copy_kernel<<<blocks, 256, 0, value.stream>>>(part, size_t(count), link.peer_inbox, link.state, link.capacity);
    check(hipGetLastError());
    link_signal_kernel<<<1, 1, 0, value.stream>>>(link.state, link.peer_flag, value.await_timeout);
    check(hipGetLastError());
    link_add_grid_kernel<<<blocks, 256, 0, value.stream>>>(part, size_t(count), link.state, link.inbox, link.capacity,
                                                           accumulate ? 1 : 0, out);
    SD_OP_END
}

__global__ void link_status_kernel(const uint32_t* state, uint32_t* out) { *out = state[2]; }

int sd_op_link_status(void* handle, uint32_t* out) {
    SD_OP_BEGIN
    if (!value.sums.peer) throw std::runtime_error("no Straydog sum link");
    op_span(value, out, 4, "a link status");
    link_status_kernel<<<1, 1, 0, value.stream>>>(value.sums.state, out);
    SD_OP_END
}

int sd_link_failed(void* handle) {
    try {
        Device& value = device(handle);
        if (!value.sums.peer) return 0;
        uint32_t state[3] = {};
        check(hipStreamSynchronize(value.stream));
        check(hipMemcpy(state, value.sums.state, sizeof(state), hipMemcpyDeviceToHost));
        return state[2] ? 1 : 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_link_reset(void* handle) {
    try {
        Device& value = device(handle);
        Device* peer = value.sums.peer;
        if (!peer) return 0;
        // Both streams idle (a wait gives up within the await timeout), then both counts from zero.
        for (Device* side : {&value, peer}) on_device(side->ordinal, [&]() { check(hipStreamSynchronize(side->stream)); });
        for (Device* side : {&value, peer})
            on_device(side->ordinal, [&]() { check(hipMemset(side->sums.state, 0, 256)); check(hipDeviceSynchronize()); });
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
