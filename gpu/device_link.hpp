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

// Peer access both ways between two devices (once; an error without it).
static void link_peers(Device& a, Device& b) {
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
}

int sd_link_sums(void* first, void* second, uint64_t capacity) {
    try {
        Device& a = device(first);
        Device& b = device(second);
        if (&a == &b || a.sums.peer || b.sums.peer || !capacity || capacity > (uint64_t(1) << 26))
            throw std::runtime_error("invalid Straydog sum link");
        link_peers(a, b);
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

int sd_link_skew(void* handle, uint64_t microseconds) {
    try {
        Device& value = device(handle);
        if (!value.sums.peer || microseconds > 1000000) throw std::runtime_error("invalid Straydog link skew");
        // (the await timeout is ten seconds of the clock, one with a watchdog)
        value.sums.skew = microseconds * (value.await_timeout / (value.await_watchdog ? 1000000 : 10000000));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
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

// A decode step's linked sums folded into the kernels forming their parts (link_tail.hpp): a
// fallback runs the part's kernels and then sd_op_link_sum's.
int sd_op_linear_link_sum(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* input, float* part,
                          float* out) {
    SD_OP_BEGIN
    SumLink& link = value.sums;
    if (!link.peer || !rows || !cols || cols > (uint64_t(1) << 24) || rows > link.capacity)
        throw std::runtime_error("invalid Straydog linked linear");
    const size_t row_bytes = matrix_row_bytes(kind, cols);
    op_span(value, weights, checked_product({rows, row_bytes}), "a weight matrix");
    op_span(value, input, checked_floats({cols}), "a linear input");
    op_span(value, part, checked_floats({rows}), "a linked sum's part");
    op_span(value, out, checked_floats({rows}), "a linked sum");
    const Matrix matrix{static_cast<const uint8_t*>(weights), row_bytes, size_t(rows), size_t(cols), 0, 0, 0, kind};
    const LinkTail tail{part, uint32_t(rows), link.state, link.peer_flag, link.peer_inbox, link.inbox, uint32_t(link.capacity), 1,
                        value.await_timeout, out, link.skew};
    if (rows > link_single_floats || !gemv_linked_launch(value, matrix, input, part, tail, op_lanes)) {
        gemv(value, &matrix, 1, input, part, 1, 0, 0, {}, op_lanes);
        if (sd_op_link_sum(handle, part, rows, out, 1)) throw std::runtime_error(last_error);
    }
    SD_OP_END
}

int sd_op_experts_table_link_sum(void* handle, const uint64_t* table, uint64_t count, const int32_t* ids, uint64_t slot_bytes,
                                 const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn, uint64_t top_k,
                                 const float* input, const float* weights, float* product, float* part, float* out) {
    SD_OP_BEGIN
    SumLink& link = value.sums;
    if (!offsets || !kinds || !hidden || !ffn || ffn > (uint64_t(1) << 20) || !top_k || top_k > 64 || !count ||
        count > (uint64_t(1) << 24) || !link.peer || hidden > link.capacity)
        throw std::runtime_error("invalid Straydog linked experts");
    const sd_expert_split whole{0, ffn, 0, hidden, 3};
    SlotExperts plan = slot_plan(slot_bytes, offsets, kinds, hidden, ffn, whole);
    op_span(value, table, checked_product({count, 8}), "an expert table");
    op_span(value, ids, checked_floats({top_k}), "expert routes");
    op_span(value, input, checked_floats({hidden}), "an expert input");
    if (weights) op_span(value, weights, checked_floats({top_k}), "route weights");
    op_span(value, product, checked_floats({top_k, ffn}), "an expert product");
    op_span(value, part, checked_floats({hidden}), "a linked sum's part");
    op_span(value, out, checked_floats({hidden}), "a linked sum");
    plan.top_k = uint32_t(top_k);
    plan.product_stride = uint32_t(ffn);
    plan.clamp = 0.0f;
    plan.accumulate = 1;
    plan.table = table;
    plan.ids = ids;
    plan.table_count = uint32_t(count);
    const LinkTail tail{part, uint32_t(hidden), link.state, link.peer_flag, link.peer_inbox, link.inbox, uint32_t(link.capacity), 1,
                        value.await_timeout, out, link.skew};
    if (hidden > link_single_floats ||
        !slot_experts(value, plan, 3, nullptr, 1, input, size_t(hidden), weights, product, part, size_t(hidden), &tail)) {
        slot_experts(value, plan, 3, nullptr, 1, input, size_t(hidden), weights, product, part, size_t(hidden));
        if (sd_op_link_sum(handle, part, hidden, out, 1)) throw std::runtime_error(last_error);
    }
    SD_OP_END
}

// ---- Part links (link_tail.hpp): a decode sum's parts sent as they form, added as they are read ----

int sd_link_parts(void* first, void* second, uint64_t capacity) {
    try {
        Device& a = device(first);
        Device& b = device(second);
        if (&a == &b || a.part_peer || b.part_peer || !capacity || capacity > (uint64_t(1) << 26))
            throw std::runtime_error("invalid Straydog part link");
        link_peers(a, b);
        for (Device* side : {&a, &b}) {
            on_device(side->ordinal, [&]() {
                for (PartLink& link : side->part_links) {
                    link.inbox = static_cast<float*>(allocate(*side, capacity * 2 * 4));
                    link.state = static_cast<uint32_t*>(allocate(*side, 256));
                    // (on the device's stream: no legacy-stream ordering against other work on the device)
                    check(hipMemsetAsync(link.inbox, 0, capacity * 2 * 4, side->stream));
                    check(hipMemsetAsync(link.state, 0, 256, side->stream));
                    link.capacity = size_t(capacity);
                }
                check(hipStreamSynchronize(side->stream));
            });
        }
        a.part_peer = &b; b.part_peer = &a;
        for (size_t channel = 0; channel < sizeof(a.part_links) / sizeof(a.part_links[0]); ++channel) {
            a.part_links[channel].peer_inbox = b.part_links[channel].inbox;
            b.part_links[channel].peer_inbox = a.part_links[channel].inbox;
            a.part_links[channel].peer_state = b.part_links[channel].state;
            b.part_links[channel].peer_state = a.part_links[channel].state;
        }
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_linear_send(void* handle, int channel, const void* weights, int kind, uint64_t rows, uint64_t cols, const float* input,
                      uint64_t steps, float* output) {
    SD_OP_BEGIN
    const LinkTail tail = part_send_tail(value, channel);
    if (!rows || !cols || cols > (uint64_t(1) << 24) || !steps || steps > uint64_t(gemv_max_vectors) || steps * rows > tail.capacity)
        throw std::runtime_error("invalid Straydog sent linear");
    const size_t row_bytes = matrix_row_bytes(kind, cols);
    op_span(value, weights, checked_product({rows, row_bytes}), "a weight matrix");
    op_span(value, input, checked_floats({steps, cols}), "a linear input");
    op_span(value, output, checked_floats({steps, rows}), "a sum's part");
    const Matrix matrix{static_cast<const uint8_t*>(weights), row_bytes, size_t(rows), size_t(cols), 0, 0, 0, kind};
    // (several vectors: their GEMV, then the part sent in one copy; each value stored on its own across the bus cost
    // more than the sum it saves)
    if (steps != 1 || !gemv_linked_launch(value, matrix, input, output, tail, op_lanes)) {
        op_linear(value, static_cast<const uint8_t*>(weights), kind, size_t(rows), size_t(cols), 0, input, size_t(cols), size_t(steps),
                  output, size_t(rows), nullptr, 0);
        part_send(value, channel, output, size_t(steps), size_t(rows), 0, size_t(rows));
    }
    SD_OP_END
}

int sd_op_part_send(void* handle, int channel, const float* part, uint64_t rows, uint64_t stride, uint64_t first, uint64_t count) {
    SD_OP_BEGIN
    if (!rows || !count || rows > (uint64_t(1) << 24) || stride > (uint64_t(1) << 26) || first + count > stride)
        throw std::runtime_error("invalid Straydog part send");
    op_span(value, part, checked_floats({rows - 1, stride}) + size_t(first + count) * 4, "a sum's part");
    part_send(value, channel, part, size_t(rows), size_t(stride), size_t(first), size_t(count));
    SD_OP_END
}

// out = this device's part + the other's, as part_wait's consumers add them (for one that is no
// hyper-connection step).
__global__ void part_receive_kernel(const float* part, size_t count, PartRead link, float* out) {
    __shared__ uint32_t n;
    if (!threadIdx.x) n = part_wait(link);
    __syncthreads();
    const float* theirs = link.inbox + size_t(n & 1) * link.capacity;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        out[i] = part[i] + link_load(theirs + i);
    __syncthreads();
    if (!threadIdx.x) part_read_done(link, n);
}

int sd_op_part_receive(void* handle, int channel, const float* part, uint64_t count, float* out) {
    SD_OP_BEGIN
    const PartRead link = part_read(value, channel);
    if (!count || count > link.capacity) throw std::runtime_error("invalid Straydog part receive");
    op_span(value, part, size_t(count) * 4, "a sum's part");
    op_span(value, out, size_t(count) * 4, "a sum");
    part_receive_kernel<<<unsigned(std::min<size_t>((size_t(count) + 255) / 256, 80)), 256, 0, value.stream>>>(part, size_t(count),
                                                                                                                 link, out);
    SD_OP_END
}

__global__ void part_status_kernel(const uint32_t* first, const uint32_t* second, uint32_t* out) {
    *out = (first[2] ? 1u : 0u) | (second[2] ? 2u : 0u);
}

int sd_op_part_status(void* handle, uint32_t* out) {
    SD_OP_BEGIN
    if (!value.part_peer) throw std::runtime_error("no Straydog part link");
    op_span(value, out, 4, "a link status");
    part_status_kernel<<<1, 1, 0, value.stream>>>(value.part_links[0].state, value.part_links[1].state, out);
    SD_OP_END
}

int sd_part_failed(void* handle) {
    try {
        Device& value = device(handle);
        if (!value.part_peer) return 0;
        check(hipStreamSynchronize(value.stream));
        int failed = 0;
        for (PartLink& link : value.part_links) {
            uint32_t state[3] = {};
            check(hipMemcpy(state, link.state, sizeof(state), hipMemcpyDeviceToHost));
            failed |= state[2] != 0;
        }
        return failed;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_part_reset(void* handle) {
    try {
        Device& value = device(handle);
        Device* peer = value.part_peer;
        if (!peer) return 0;
        // Both streams idle (a wait gives up within the await timeout), then every count from zero.
        for (Device* side : {&value, peer}) on_device(side->ordinal, [&]() { check(hipStreamSynchronize(side->stream)); });
        for (Device* side : {&value, peer})
            on_device(side->ordinal, [&]() {
                for (PartLink& link : side->part_links) check(hipMemsetAsync(link.state, 0, 256, side->stream));
                check(hipStreamSynchronize(side->stream));
            });
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
