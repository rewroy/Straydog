// Qwen3.8's layers split across two GPUs (docs/DESIGN.md §8). Both devices run every layer of a pass at once, each its part
// (QwenHeads<2>): half of a DeltaNet's heads (8 key heads and the 24 value heads that read them) or of an attention layer's
// (12 query heads and their K/V head), with their rows of the projections and their columns of the output projection; half of the
// routed experts; and half of the shared expert's width (rows of gate and up, columns of down). Both keep the whole residual
// streams and compute the HyperConnections and the router alike, so they select the same experts; the attention's indexer alike
// for a pass's few rows, and for a prompt's chunks shared out in turn (qsa_prefill's), the selected blocks exchanged: both hold the
// same K/V blocks (as marks, or as each token's own selection list from 8192 tokens). A
// layer's two outputs, the attention's projection and the experts', are each this device's part of a sum the devices add (the
// SumLink of device_link.hpp) before it goes into the streams; both add the same two values, so their streams stay equal. A
// part's states are its own: its DeltaNet heads' recurrence and convolution, its K/V head's planes.
// Included in the GPU TU after device_link.hpp; its steps join the TU's anonymous namespace (verify_span.hpp declares
// verify_layer_part there).

namespace {

// A part's sum for a pass's rows (at most ROWS, 4 or 8): this device's part of `steps` rows of 2560 goes into the other's inbox slot,
// the other's flag is raised, and once this device's is, each row's two parts are added and go into the four residual streams
// as a whole layer's residual update does (hidden[s] += mix[s] * sum, mix after the row's 2560 inputs, `mixed_stride` apart):
// link_sum_kernel's steps.
#if defined(__HIPCC__)
// A layer's expert outputs as a part (moe_part_partial's sum, formed by the sum's own block: one launch less a layer).
struct QwenExpertPart {
    const float* outputs; const float* probabilities; const int32_t* selected; const MoeGroup* groups; bool shared;
};
template<bool experts, int ROWS = 4>
__global__ void __launch_bounds__(1024) qwen_part_sum(const float* part, QwenExpertPart moe, uint32_t steps, float* hidden,
                                                       const float* mixed, size_t mixed_stride, uint32_t* state, uint32_t* peer_flag,
                                                       float* peer_inbox, const float* inbox, size_t capacity, uint64_t timeout) { SD_KERNEL_ACQUIRE;
    const uint32_t count = steps * 2560, next = state[0] + 1;
    const size_t slot = size_t(next & 1) * capacity;
    constexpr uint32_t values = ROWS * 2560 / 1024;
    __shared__ bool held[ROWS][10];
    if constexpr (experts) {
        if (threadIdx.x < steps * 10) held[threadIdx.x / 10][threadIdx.x % 10] = moe.groups[moe.selected[threadIdx.x]].weights[2].data != nullptr;
        __syncthreads();
    }
    // (ten values a thread a pass of four rows)
    float own[values];
    #pragma unroll
    for (uint32_t k = 0; k < values; ++k) {
        const uint32_t i = threadIdx.x + k * 1024;
        if (i >= count) break;
        if constexpr (experts) {
            // moe_own_partial's sum in route order, then the shared expert's (moe_part_partial's)
            const uint32_t token = i / 2560, column = i % 2560;
            float sum = 0;
            for (size_t route = 0; route < 10; ++route)
                if (held[token][route])
                    sum += moe.probabilities[token * 11 + route] * moe_route_output(moe.outputs, steps, false, token, route)[column];
            if (moe.shared) sum += moe.probabilities[token * 11 + 10] * moe.outputs[(token * 11 + 10) * 2560 + column];
            own[k] = sum;
        } else {
            own[k] = part[i];
        }
        link_send(peer_inbox + slot + i, own[k]);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (!threadIdx.x) {
        state[0] = next;
        __threadfence_system();
        peer_store(peer_flag, next);
        const uint64_t start = peer_clock();
        while (int32_t(peer_load(state + 1) - next) < 0) {
            if (peer_clock() - start > timeout) { state[2] = next; break; }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t k = 0; k < values; ++k) {
        const uint32_t i = threadIdx.x + k * 1024;
        if (i >= count) break;
        const uint32_t token = i / 2560, column = i % 2560;
        const float total = own[k] + link_load(inbox + slot + i);
        const float* mix = mixed + token * mixed_stride + 2560;
        #pragma unroll
        for (uint32_t stream = 0; stream < 4; ++stream) hidden[token * 10240 + stream * 2560 + column] += mix[stream] * total;
    }
}
#endif
// A prompt batch's rows are split between the devices for what they compute alike (BatchRows: part 0 the first (steps + 1) / 2):
// each runs the HyperConnections and the residual updates of its rows only, and their outputs are exchanged where every row is
// read: a HyperConnection's mixed rows go to the other (all-gather, the rows rounded to FP16 on both: the projections take them
// in FP16), and a layer's two parts of an output are added only for each device's rows (reduce-scatter: the other's rows of this
// device's part rounded to FP16 into its inbox, this device's alike when it adds them). Both cost the bus what an all-reduce
// did, and the HyperConnections, run by both on every row, were a fifth of a 60K prompt.
struct BatchRows { size_t first, count; };
inline BatchRows batch_rows(size_t steps, size_t part) {
    const size_t half = (steps + 1) / 2;
    return part ? BatchRows{half, steps - half} : BatchRows{0, half};
}
// The exchanges move 8 values a thread into the other's inbox (16-byte stores) and 4 out of this device's (8-byte system-scope
// loads, never a stale cached line): a 95K prompt's exchanges were 1.2% faster than with a value or two a thread.
__device__ inline uint32_t qwen_half2_bits(__half2 value) { uint32_t bits; memcpy(&bits, &value, 4); return bits; }
__device__ inline uint64_t qwen_inbox_bits(const void* at) { return link_load64(at); }
// Four halves of this device's inbox, as floats.
__device__ inline float4 qwen_inbox_half4(const __half* at) {
    const uint64_t bits = qwen_inbox_bits(at);
    const uint32_t low = uint32_t(bits), high = uint32_t(bits >> 32);
    __half2 a, b;
    memcpy(&a, &low, 4); memcpy(&b, &high, 4);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}
// Four floats of this device's inbox.
__device__ inline float4 qwen_inbox_float4(const float* at) {
    const uint64_t a = qwen_inbox_bits(at), b = qwen_inbox_bits(at + 2);
    float4 value;
    memcpy(&value, &a, 8); memcpy(reinterpret_cast<char*>(&value) + 8, &b, 8);
    return value;
}
// 8 floats rounded to FP16: their halves as 16 bytes, and the halves back as floats into `back` unless it is null.
__device__ inline uint4 qwen_round8(const float* from, float* back) {
    const float4 a = reinterpret_cast<const float4*>(from)[0], b = reinterpret_cast<const float4*>(from)[1];
    const __half2 h0 = __floats2half2_rn(a.x, a.y), h1 = __floats2half2_rn(a.z, a.w), h2 = __floats2half2_rn(b.x, b.y),
                  h3 = __floats2half2_rn(b.z, b.w);
    if (back) {
        reinterpret_cast<float4*>(back)[0] = make_float4(__low2float(h0), __high2float(h0), __low2float(h1), __high2float(h1));
        reinterpret_cast<float4*>(back)[1] = make_float4(__low2float(h2), __high2float(h2), __low2float(h3), __high2float(h3));
    }
    return make_uint4(qwen_half2_bits(h0), qwen_half2_bits(h1), qwen_half2_bits(h2), qwen_half2_bits(h3));
}
// The other's rows [first, first + count) of this device's part, in FP16, into the other's inbox slot.
__global__ void qwen_scatter_send(const float* part, size_t first, size_t count, __half* peer_inbox, const uint32_t* state,
                                  size_t capacity) {
    uint4* slot = reinterpret_cast<uint4*>(peer_inbox + size_t((state[0] + 1) & 1) * capacity);
    const float* rows = part + first * 2560;
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < count * 320; g += size_t(gridDim.x) * blockDim.x)
        slot[g] = qwen_round8(rows + 8 * g, nullptr);
}
// This device's rows of the sum into their streams: its part rounded as the other's was, plus the other's (the mixes after all
// of the batch's 2560-value rows, decoder_batch_residual's update). A thread adds four columns into the four streams.
__global__ void qwen_scatter_residual(const float* part, size_t steps, size_t first, size_t count, const uint32_t* state,
                                      const __half* inbox, size_t capacity, const float* mixed, float* hidden) {
    const size_t g = size_t(blockIdx.x) * 256 + threadIdx.x, local = g / 640, row = first + local;
    if (local >= count) return;
    const size_t column = (g % 640) * 4;
    const float4 mine = *reinterpret_cast<const float4*>(part + row * 2560 + column);
    const float4 theirs = qwen_inbox_half4(inbox + size_t(state[0] & 1) * capacity + local * 2560 + column);
    const float4 total = make_float4(__half2float(__float2half_rn(mine.x)) + theirs.x, __half2float(__float2half_rn(mine.y)) + theirs.y,
                                     __half2float(__float2half_rn(mine.z)) + theirs.z, __half2float(__float2half_rn(mine.w)) + theirs.w);
    const float* mixes = mixed + steps * 2560 + row * 4;
    for (size_t stream = 0; stream < 4; ++stream) {
        float4* at = reinterpret_cast<float4*>(hidden + row * 10240 + stream * 2560 + column);
        float4 value = *at;
        const float mix = mixes[stream];
        value.x += mix * total.x; value.y += mix * total.y; value.z += mix * total.z; value.w += mix * total.w;
        *at = value;
    }
}
// The next HyperConnection's normalize folded into a scatter's residual update (hc_prefill's FP16 path, `normalized`): where
// hc_normalize_half's halves and denominators of this device's rows go, and the norm it reads.
struct HcNext { const float* norm; __half* half; float* denominators; };
inline HcNext hc_next(const float* norm, float* scratch, size_t steps) { return {norm, hc_scratch_half(scratch, steps), scratch}; }
// A half of this device's inbox.
__device__ inline float qwen_inbox_half(const __half* at) {
#if defined(__HIPCC__)
    return __half2float(__ushort_as_half(__hip_atomic_load(reinterpret_cast<const uint16_t*>(at), __ATOMIC_RELAXED,
                                                           __HIP_MEMORY_SCOPE_SYSTEM)));
#else
    return __half2float(__ushort_as_half(*reinterpret_cast<const volatile uint16_t*>(at)));
#endif
}
// qwen_scatter_residual with hc_normalize_half after it, on the updated rows still in registers: a block a row, its threads the
// columns t + 256k of the four streams, each stream's sum of squares in hc_normalize_half's order (the same bits as the two).
__global__ void __launch_bounds__(256) qwen_scatter_residual_normalize(const float* part, size_t steps, size_t first, size_t count,
                                                                       const uint32_t* state, const __half* inbox, size_t capacity,
                                                                       const float* mixed, float* hidden, HcNext next) {
    const size_t local = blockIdx.x, row = first + local, t = threadIdx.x;
    const __half* theirs = inbox + size_t(state[0] & 1) * capacity + local * 2560;
    const float* mixes = mixed + steps * 2560 + row * 4;
    const float mix[4] = {mixes[0], mixes[1], mixes[2], mixes[3]};
    float* streams = hidden + row * 10240;
    float values[4][10], totals[4] = {0, 0, 0, 0};
    for (size_t k = 0; k < 10; ++k) {
        const size_t column = t + 256 * k;
        const float total = __half2float(__float2half_rn(part[row * 2560 + column])) + qwen_inbox_half(theirs + column);
        for (size_t stream = 0; stream < 4; ++stream) {
            // (explicit FMAs: the residual's as qwen_scatter_residual's compile to, the squares' as hc_normalize_half's)
            const float value = fmaf(mix[stream], total, streams[stream * 2560 + column]);
            streams[stream * 2560 + column] = value;
            values[stream][k] = value;
            totals[stream] = fmaf(value, value, totals[stream]);
        }
    }
    __shared__ float sum[4][256];
    for (size_t stream = 0; stream < 4; ++stream) sum[stream][t] = totals[stream];
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (t < stride)
            for (size_t stream = 0; stream < 4; ++stream) sum[stream][t] += sum[stream][t + stride];
        __syncthreads();
    }
    for (size_t stream = 0; stream < 4; ++stream) {
        const float denominator = sqrtf(sum[stream][0] / 2560.0f + 1e-6f);
        if (t == 0) next.denominators[local * 4 + stream] = denominator;
        for (size_t k = 0; k < 10; ++k) {
            const size_t column = t + 256 * k;
            next.half[local * 10240 + stream * 2560 + column] =
                __float2half(hc_f32(values[stream][k] / denominator * next.norm[stream * 2560 + column]));
        }
    }
}
// A HyperConnection's mixed rows of this device's rows (`local`: their 2560-value rows, then their four mixes each) into the
// batch's layout, rounded to FP16 as they go to the other's inbox slot (the rows' halves, then the mixes as floats).
__global__ void qwen_gather_send(const float* local, size_t steps, size_t first, size_t count, float* mixed, __half* peer_inbox,
                                 const uint32_t* state, size_t capacity) {
    __half* slot = peer_inbox + size_t((state[0] + 1) & 1) * capacity;
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < count * 321; g += size_t(gridDim.x) * blockDim.x) {
        if (g < count * 320) {
            reinterpret_cast<uint4*>(slot)[g] = qwen_round8(local + 8 * g, mixed + first * 2560 + 8 * g);
        } else {
            const size_t j = g - count * 320;
            const float4 mixes = reinterpret_cast<const float4*>(local + count * 2560)[j];
            reinterpret_cast<float4*>(mixed + steps * 2560 + first * 4)[j] = mixes;
            reinterpret_cast<float4*>(slot + count * 2560)[j] = mixes;
        }
    }
}
// The other's rows [first, first + count) of mixed from this device's inbox slot (`fixed`: from `inbox` as given, one of the slots past
// the two the sums alternate between).
template<bool fixed = false>
__global__ void qwen_gather_receive(size_t steps, size_t first, size_t count, const uint32_t* state, const __half* inbox, size_t capacity,
                                    float* mixed) {
    const __half* slot = fixed ? inbox : inbox + size_t(state[0] & 1) * capacity;
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < count * 641; g += size_t(gridDim.x) * blockDim.x) {
        if (g < count * 640) reinterpret_cast<float4*>(mixed + first * 2560)[g] = qwen_inbox_half4(slot + 4 * g);
        else {
            const size_t j = g - count * 640;
            reinterpret_cast<float4*>(mixed + steps * 2560 + first * 4)[j] =
                qwen_inbox_float4(reinterpret_cast<const float*>(slot + count * 2560) + 4 * j);
        }
    }
}
// `part` null: the part is the experts' of `moe` (verify_layer_part's).
void qwen_sum_rows(Device& d, const float* part, size_t steps, float* hidden, const float* mixed, size_t mixed_stride,
                   const TokenMoe* moe = nullptr) {
    SumLink& link = d.sums;
    if (!link.peer || !steps || steps > size_t(gemv_wide_vectors) || steps * 2560 > link.capacity || (!part && !moe))
        throw std::runtime_error("invalid Straydog part sum");
#if defined(__HIPCC__)
    const QwenExpertPart experts = part ? QwenExpertPart{} : QwenExpertPart{moe->outputs, moe->probabilities, moe->selected, moe->groups, moe->plan->shared};
    #define SD_PART_SUM(HOLDS, ROWS) qwen_part_sum<HOLDS, ROWS><<<1, 1024, 0, d.stream>>>(part, experts, uint32_t(steps), hidden, mixed, mixed_stride, link.state, \
        link.peer_flag, link.peer_inbox, link.inbox, link.capacity, d.await_timeout)
    if (steps <= 4) { if (part) SD_PART_SUM(false, 4); else SD_PART_SUM(true, 4); }
    else { if (part) SD_PART_SUM(false, 8); else SD_PART_SUM(true, 8); }
    #undef SD_PART_SUM
    check(hipGetLastError());
#else
    (void)hidden; (void)mixed; (void)mixed_stride;
    throw std::runtime_error("Straydog splits Qwen3.8's layers on RDNA4 only");
#endif
}
// This device's rows of a batch's sum into their streams (reduce-scatter).
// `next`: the residual update also normalizes the rows for the next HyperConnection (qwen_scatter_residual_normalize).
void qwen_scatter_prefill(Device& d, const float* part, size_t steps, BatchRows own, BatchRows other, float* hidden, const float* mixed,
                          const HcNext* next = nullptr) {
    SumLink& link = d.sums;
    // (an inbox slot of `capacity` floats holds twice as many halves; whole 32 bytes, for the 16-byte stores and loads)
    if (!link.peer || !steps || std::max(own.count, other.count) * 2560 > link.capacity * 2 || link.capacity % 8)
        throw std::runtime_error("invalid Straydog part sum");
    qwen_scatter_send<<<unsigned(std::min<size_t>((other.count * 320 + 255) / 256, 1024)), 256, 0, d.stream>>>(
        part, other.first, other.count, reinterpret_cast<__half*>(link.peer_inbox), link.state, link.capacity * 2);
    link_signal_kernel<<<1, 1, 0, d.stream>>>(link.state, link.peer_flag, d.await_timeout);
    if (next)
        qwen_scatter_residual_normalize<<<unsigned(own.count), 256, 0, d.stream>>>(
            part, steps, own.first, own.count, link.state, reinterpret_cast<const __half*>(link.inbox), link.capacity * 2, mixed, hidden,
            *next);
    else
        qwen_scatter_residual<<<unsigned((own.count * 640 + 255) / 256), 256, 0, d.stream>>>(
            part, steps, own.first, own.count, link.state, reinterpret_cast<const __half*>(link.inbox), link.capacity * 2, mixed, hidden);
    check(hipGetLastError());
}
// A HyperConnection's mixed rows of every row of the batch, from this device's rows' (`local`, hc_prefill's layout) and the other's.
void qwen_gather_mixed(Device& d, const float* local, size_t steps, BatchRows own, BatchRows other, float* mixed) {
    SumLink& link = d.sums;
    if (!link.peer || std::max(own.count, other.count) * 2568 > link.capacity * 2 || link.capacity % 8)
        throw std::runtime_error("invalid Straydog row gather");
    qwen_gather_send<<<unsigned(std::min<size_t>((own.count * 321 + 255) / 256, 1024)), 256, 0, d.stream>>>(
        local, steps, own.first, own.count, mixed, reinterpret_cast<__half*>(link.peer_inbox), link.state, link.capacity * 2);
    link_signal_kernel<<<1, 1, 0, d.stream>>>(link.state, link.peer_flag, d.await_timeout);
    qwen_gather_receive<false><<<unsigned(std::min<size_t>((other.count * 641 + 255) / 256, 1024)), 256, 0, d.stream>>>(
        steps, other.first, other.count, link.state, reinterpret_cast<const __half*>(link.inbox), link.capacity * 2, mixed);
    check(hipGetLastError());
}
// qwen_gather_mixed in the experts' order (prefill_layer_part): the send of this device's rows to the other goes on the send stream while
// the device stream goes on with them, and once it has done what it can with its own rows alone it signals and takes the other's.
// The send is a copy (the copy engine, which takes no compute units: a send as a kernel did not overlap a GEMM, 1.4 ms for two
// that take 0.7 ms each) of this device's rows as an inbox slot holds them, staged by qwen_gather_stage in the other's rows of
// `mixed`, which nothing reads until qwen_gather_receive; into a slot of its own in the other's inbox (past the two the sums alternate
// between: the copy's address is the host's, the alternating slot's the device's count). The slot is free to write once this device's
// last signal has returned: the other raised its flag for it after it had read the slot's last use (the gather before), as with every slot.
__global__ void qwen_gather_stage(const float* local, size_t steps, size_t first, size_t count, float* mixed, __half* stage) {
    for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < count * 321; g += size_t(gridDim.x) * blockDim.x) {
        if (g < count * 320) {
            reinterpret_cast<uint4*>(stage)[g] = qwen_round8(local + 8 * g, mixed + first * 2560 + 8 * g);
        } else {
            const size_t j = g - count * 320;
            const float4 mixes = reinterpret_cast<const float4*>(local + count * 2560)[j];
            reinterpret_cast<float4*>(mixed + steps * 2560 + first * 4)[j] = mixes;
            reinterpret_cast<float4*>(stage + count * 2560)[j] = mixes;
        }
    }
}
constexpr size_t qwen_gather_slot = 2;   // (in slots of `capacity` floats from the inbox's start)
void qwen_send_streams(Device& d) {
    if (d.send_stream) return;
    check(hipStreamCreate(&d.send_stream));
    check(hipEventCreateWithFlags(&d.send_fork, hipEventDisableTiming));
    check(hipEventCreateWithFlags(&d.send_done, hipEventDisableTiming));
}
void qwen_gather_fork(Device& d, const float* local, size_t steps, BatchRows own, BatchRows other, float* mixed) {
    SumLink& link = d.sums;
    if (!link.peer || d.send_pending || own.count * 2568 > link.capacity * 2 || link.capacity % 8 || own.count * 2568 * 2 > other.count * 2560 * 4)
        throw std::runtime_error("invalid Straydog row gather");
    qwen_send_streams(d);
    auto* stage = reinterpret_cast<__half*>(mixed + other.first * 2560);
    qwen_gather_stage<<<unsigned(std::min<size_t>((own.count * 321 + 255) / 256, 1024)), 256, 0, d.stream>>>(local, steps, own.first, own.count,
                                                                                                             mixed, stage);
    check(hipGetLastError());
    // (after the stream's last signal and the stage: the slot is free, and the copy reads what the stage wrote)
    check(hipEventRecord(d.send_fork, d.stream));
    check(hipStreamWaitEvent(d.send_stream, d.send_fork, 0));
    check(hipMemcpyAsync(link.peer_inbox + qwen_gather_slot * link.capacity, stage, own.count * 2568 * 2, hipMemcpyDefault, d.send_stream));
    check(hipEventRecord(d.send_done, d.send_stream));
    d.send_pending = true;
}
// The signal waits for the copy (an event after it on the send stream); then the other's rows.
void qwen_gather_complete(Device& d, size_t steps, BatchRows other, float* mixed) {
    SumLink& link = d.sums;
    if (!link.peer || !d.send_pending || other.count * 2568 > link.capacity * 2) throw std::runtime_error("invalid Straydog row gather");
    check(hipStreamWaitEvent(d.stream, d.send_done, 0));
    d.send_pending = false;
    link_signal_kernel<<<1, 1, 0, d.stream>>>(link.state, link.peer_flag, d.await_timeout);
    qwen_gather_receive<true><<<unsigned(std::min<size_t>((other.count * 641 + 255) / 256, 1024)), 256, 0, d.stream>>>(
        steps, other.first, other.count, link.state, reinterpret_cast<const __half*>(link.inbox + qwen_gather_slot * link.capacity),
        link.capacity * 2, mixed);
    check(hipGetLastError());
}
void qsa_share_signal(Device& d) {
    link_signal_kernel<<<1, 1, 0, d.stream>>>(d.sums.state, d.sums.peer_flag, d.await_timeout);
    check(hipGetLastError());
}
// This device's rows of the residual streams into the other's (`peer_hidden`, its workspace), then the other's flag raised and this
// device's waited for: every row on both, as F32.
void qwen_gather_hidden(Device& d, const float* hidden, float* peer_hidden, BatchRows own) {
    check(hipMemcpyAsync(peer_hidden + own.first * 10240, hidden + own.first * 10240, own.count * 10240 * 4, hipMemcpyDefault, d.stream));
    link_signal_kernel<<<1, 1, 0, d.stream>>>(d.sums.state, d.sums.peer_flag, d.await_timeout);
    check(hipGetLastError());
}

// After both devices' streams are idle: a sum whose wait gave up voids the pass and leaves the counts out of step, so both
// start over from zero and the call fails (peer_settle's for the sums).
void qwen_sums_settle(Device& d) {
    Device* peer = d.sums.peer;
    if (!peer) return;
    uint32_t failed[2] = {};
    for (size_t side = 0; side < 2; ++side) {
        Device& x = side ? *peer : d;
        on_device(x.ordinal, [&]() { check(hipMemcpy(&failed[side], x.sums.state + 2, 4, hipMemcpyDeviceToHost)); });
    }
    if (!failed[0] && !failed[1]) return;
    for (Device* side : {&d, peer})
        on_device(side->ordinal, [&]() { check(hipMemset(side->sums.state, 0, 256)); check(hipDeviceSynchronize()); });
    throw std::runtime_error("a Straydog GPU waited too long for its peer's part");
}

// A device's part of a layer's experts: its routed experts' weighted outputs, in route order, then the shared expert's when it
// holds it (decode: eleven output slots a row; prefill: the routed rows, then the shared rows).
__global__ void moe_part_partial(const float* outputs, const float* probabilities, const int32_t* selected, const MoeGroup* groups,
                                 size_t steps, bool prefill, bool shared, float* partial) {
    const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x, token = blockIdx.y;
    if (column >= 2560) return;
    float sum = moe_own_partial(outputs, probabilities, selected, groups, steps, prefill, token, column);
    if (shared) {
        const float* row = prefill ? outputs + (steps * 10 + token) * 2560 : outputs + (token * 11 + 10) * 2560;
        sum += probabilities[token * 11 + 10] * row[column];
    }
    partial[token * 2560 + column] = sum;
}

// The shared expert's half (ResidentMoe::half): silu(gate) * up of its 320 columns, rows of gate and up `stride` apart into rows
// `product_stride` apart.
__global__ void moe_half_product(const float* gate, const float* up, size_t stride, float* product, size_t product_stride) {
    const size_t row = blockIdx.x;
    for (size_t column = threadIdx.x; column < 320; column += blockDim.x) {
        const float g = gate[row * stride + column];
        product[row * product_stride + column] = g * hc_sigmoid(g) * up[row * stride + column];
    }
}
// A prompt batch's rows through the part's half of the shared expert (beside the grouped experts, which run without it): FP16
// GEMMs as the batch's other projections, its rows into the shared rows of the outputs. (Its scratch: the gate/up area, whose
// routed rows the kernels queued before it have read.)
void moe_half_batch(Device& d, const StepMoe& moe, size_t steps, const float* input) {
    const ResidentMoe& plan = *moe.plan;
    float* gate = moe.gate_up; float* up = gate + steps * 320; float* product = up + steps * 320;
    auto* half = reinterpret_cast<__half*>(product + steps * 320);
    const bool cast = batch_half(d, steps);
    if (cast) batch_cast(d, input, half, steps * 2560);
    batch_project(d, plan.half[0], steps, input, half, gate);
    batch_project(d, plan.half[1], steps, input, half, up);
    moe_half_product<<<unsigned(steps), 320, 0, d.stream>>>(gate, up, 320, product, 320);
    if (cast) batch_cast(d, product, half, steps * 320);
    batch_project(d, plan.half[2], steps, product, half, moe.outputs + steps * 10 * 2560);
    check(hipGetLastError());
}

// The part's matrices of a layer, from its registered operands (register_decoder_layer's rows and columns for two parts).
void qwen_part_matrices(const DecoderLayer& plan, bool linear, Matrix (&m)[6]) {
    using H = QwenHeads<2>;
    const size_t gd_rows[5] = {H::channels, H::gate, H::value, H::value, 2560}, gd_cols[5] = {2560, 2560, 2560, 2560, H::gate};
    const size_t qs_rows[6] = {H::query * 512, H::kv * 256, H::kv * 256, 512, 128, 2560};
    const size_t qs_cols[6] = {2560, 2560, 2560, 2560, 2560, H::query * 256};
    const size_t count = linear ? 5 : 6;
    for (size_t i = 0; i < count; ++i) {
        const size_t rows = linear ? gd_rows[i] : qs_rows[i], cols = linear ? gd_cols[i] : qs_cols[i];
        const int kind = plan.kinds[4 + i];
        m[i] = {static_cast<const uint8_t*>(plan.pointers[4 + i]), matrix_row_bytes(kind, cols), rows, cols, 0, 0, 0, kind};
    }
}

// verify_layer's steps for a part: the attention or DeltaNet heads of the part into its part of the output projection, summed
// into the streams; then the experts it holds, summed alike.
void verify_layer_part(Device& d, DecoderLayer& plan, size_t steps, const VerifyLayout& v) {
    using H = QwenHeads<2>;
    void* const* pointers = plan.pointers.data(); const int* kinds = plan.kinds.data();
    const bool linear = decoder_linear(plan.index);
    const size_t attention = linear ? 9 : 10;
    Matrix hc[4];
    hc_operands(pointers, kinds, hc);
    // (a DeltaNet layer's history copy for its tokens rides in the HyperConnection's second launch)
    const HcCopy copy = linear ? gdn_history_copy(d, static_cast<float*>(plan.states[0]), 2) : HcCopy{};
    hc_decode(d, hc, true, v.hidden, d.output, v.mixed, steps, 2564, copy);
    Matrix m[6];
    qwen_part_matrices(plan, linear, m);
    if (linear) {
        size_t offset = 0;
        for (size_t i = 0; i < 4; ++i) { m[i].output_offset = offset; offset += m[i].rows; }
        gemv(d, m, 4, v.mixed, plan.projections, steps, 2564, H::projection);
        void* const* p = pointers + 4;
        auto* conv = static_cast<float*>(plan.states[0]); auto* recurrent = static_cast<float*>(plan.states[1]);
        gdn_tokens<2><<<unsigned(H::value), gdn_threads, 0, d.stream>>>(recurrent, gdn_history_snapshot(d, conv, 2, false), recurrent, conv,
            static_cast<const float*>(p[5]), static_cast<const float*>(p[6]), static_cast<const float*>(p[7]),
            static_cast<const float*>(p[8]), plan.projections, H::projection, steps, plan.snapshot, v.post);
        gemv(d, &m[4], 1, v.post, v.branch, steps, H::gate, 2560);
    } else {
        size_t offset = 0;
        for (size_t i = 0; i < 5; ++i) { m[i].output_offset = offset; offset += m[i].rows; }
        gemv(d, m, 5, v.mixed, v.qsa, steps, 2564, verify_qsa);
        qsa_decode_attend_rows<2>(d, pointers + 4, plan.states[0], plan.states[1], plan.states[2], plan.context, v.positions,
                                  v.positions + verify_ropes, v.qsa, verify_qsa, v.attended, steps);
        gemv(d, &m[5], 1, v.attended, v.branch, steps, H::query * 256, 2560);
    }
    qwen_sum_rows(d, v.branch, steps, v.hidden, v.mixed, 2564);
    hc_operands(pointers + 4 + attention, kinds + 4 + attention, hc);
    hc_decode(d, hc, true, v.hidden, d.output, v.mixed, steps, 2564);
    const TokenMoe moe = token_moe(d, plan.moe, steps);
    moe_tokens_experts(d, moe, steps, v.mixed, 2564);
    qwen_sum_rows(d, nullptr, steps, v.hidden, v.mixed, 2564, &moe);
}

// A prefill batch through a layer's part: prefill_span's steps, the HyperConnections and residual updates of this device's rows
// (BatchRows), each output a part of a sum.
// `normalized`: the layer before's last scatter normalized the rows for this layer's first HyperConnection; `next`: this layer's
// last scatter does so for the next layer's.
void prefill_layer_part(Device& d, size_t layer, size_t position, size_t steps, float* hidden, float* mixed, float* branch,
                        bool normalized = false, bool next = false) {
    const auto& plan = d.decoder.find(layer)->second;
    auto* pointers = plan.pointers.data(); const int* kinds = plan.kinds.data();
    const auto& experts = *d.resident.find(plan.moe)->second;
    float* scratch = experts.workspace;
    // (the workspace holds the rows' scratch of every stage but the experts', which run in chunks of the plan's capacity, below: a part's, with the
    // HyperConnections' over this device's half of the rows)
    const auto workspace = d.allocations.find(scratch);
    if (workspace == d.allocations.end() || steps > prefill_rows_max ||
        std::max({(steps + 1) / 2 * hc_batch_floats, steps * gdn_batch_floats_of<2>(), steps * qsa_batch_floats_of<2>()}) > workspace->second / 4)
        throw std::runtime_error("prefill batch exceeds the resident expert workspace");
    const bool linear = decoder_linear(layer);
    const BatchRows own = batch_rows(steps, plan.part), other = batch_rows(steps, 1 - plan.part);
    // (with FP16 projections, each scatter's residual update normalizes the rows for the HyperConnection after it)
    const bool fold = batch_half(d, own.count);
    // (the HyperConnections' rows in `branch` before they are gathered: the last sum has read the branch there)
    hc_prefill(d, pointers, kinds, own.count, hidden + own.first * 10240, branch, scratch, fold && normalized);
    qwen_gather_mixed(d, branch, steps, own, other, mixed);
    if (linear) {
        const size_t chunk = d.input_capacity >= 16 * 1024 * 1024 ? 512 : 128;
        gdn_prefill<2>(d, pointers + 4, kinds + 4, plan.states[0], plan.states[1], steps, chunk, mixed, branch, scratch);
    } else {
        qsa_prefill<2>(d, pointers + 4, kinds + 4, plan.states[0], plan.states[1], plan.states[2], plan.context, position, steps, mixed,
                       branch, scratch, int(plan.part));
        if (steps > 1) kv_prefetch_from(d, layer + 1, position);
    }
    void* const* second = pointers + 4 + (linear ? 9 : 10);
    const HcNext second_next = hc_next(static_cast<const float*>(second[0]), scratch, own.count);
    qwen_scatter_prefill(d, branch, steps, own, other, hidden, mixed, fold ? &second_next : nullptr);
    hc_prefill(d, second, kinds + 4 + (linear ? 9 : 10), own.count, hidden + own.first * 10240, branch, scratch, fold);
    // The experts of at most `experts.capacity` rows at a time (a batch of no more is one chunk, as ever): each row's routes and outputs are its own,
    // so the chunks give a row the sum a batch of its own would; a wider batch's chunks read the experts again, as the batches of that width do. The
    // chunks are as even as can be, none of a lone row (a row alone takes the decode layout of the experts' outputs).
    const size_t chunks = (steps + experts.capacity - 1) / experts.capacity;
    struct Chunk { size_t first, count; };
    std::vector<Chunk> order(chunks);
    for (size_t chunk_index = 0, first = 0; chunk_index < chunks; ++chunk_index) {
        order[chunk_index] = {first, steps / chunks + (chunk_index < steps % chunks ? 1 : 0)};
        first += order[chunk_index].count;
    }
    // Where the chunks fall within the devices' rows (BatchRows), this device's rows' chunks run first, beside its send of them to the other
    // (qwen_gather_fork), and the other's rows' chunks once those have arrived: only the order of the chunks changes.
    size_t own_chunks = 0;
    if (d.parts_overlap && chunks >= 2)
        for (size_t at = 1; at < chunks; ++at)
            if (order[at].first == batch_rows(steps, 0).count) own_chunks = plan.part ? chunks - at : at;
    if (own_chunks) {
        if (plan.part) std::rotate(order.begin(), order.begin() + (chunks - own_chunks), order.end());
        qwen_gather_fork(d, branch, steps, own, other, mixed);
    } else {
        qwen_gather_mixed(d, branch, steps, own, other, mixed);
    }
    for (size_t at = 0; at < chunks; ++at) {
        if (own_chunks && at == own_chunks) qwen_gather_complete(d, steps, other, mixed);
        const size_t first = order[at].first, count = order[at].count;
        const float* rows = mixed + first * 2560;
        const StepMoe moe = step_moe(d, plan.moe, count);
        moe_steps_route(d, moe, count, rows);
        moe_steps_compute(d, moe, count, rows);
        const bool half = moe.plan->half[0].data != nullptr;
        if (half) moe_half_batch(d, moe, count, rows);
        moe_part_partial<<<dim3(10, unsigned(count)), 256, 0, d.stream>>>(moe.outputs, moe.probabilities, moe.selected, moe.groups, count,
                                                                         true, moe.plan->shared || half, branch + first * 2560);
    }
    if (fold && next) {
        const auto& following = d.decoder.find(layer + 1)->second;
        const HcNext first_next = hc_next(static_cast<const float*>(following.pointers[0]),
                                          d.resident.find(following.moe)->second->workspace, own.count);
        qwen_scatter_prefill(d, branch, steps, own, other, hidden, mixed, &first_next);
    } else {
        qwen_scatter_prefill(d, branch, steps, own, other, hidden, mixed);
    }
}

// Layers [begin, end) of a prompt's batch on both devices, layer by layer (each device queues a layer before the other device's
// sums of it wait for it: nothing on the host waits until both have queued the span). The input rows (none: the rows the
// span before left in the workspaces) go to both; the last `output_rows` rows come back from the second. Between a batch's spans
// each device holds its rows (BatchRows); after the first layer (the PLE layer reads every row on both) and the last, both hold
// every row. A span without output rows is only queued: its check for non-finite rows accumulates (flags[3]) until a span with
// them waits for both devices (a batch's spans then run with no host round trip between them).
void prefill_span_parts(Device& a, Device& b, size_t begin, size_t end, size_t position, size_t steps, const float* input,
                        float* output, size_t output_rows) {
    if (begin >= end || end > 49 || !steps || steps > prefill_rows_max || end - begin > 4 || output_rows > steps || (output_rows && !output) ||
        a.sums.peer != &b || b.sums.peer != &a)
        throw std::runtime_error("invalid prefill span of parts");
    Device* sides[2] = {&a, &b};
    float* workspaces[2] = {};
    for (size_t side = 0; side < 2; ++side) {
        Device& d = *sides[side];
        for (size_t layer = begin; layer < end; ++layer) {
            auto found = d.decoder.find(layer);
            if (found == d.decoder.end() || found->second.parts != 2 || found->second.part != side || steps > found->second.capacity ||
                position > found->second.context || steps > found->second.context - position ||
                (workspaces[side] && workspaces[side] != found->second.workspace))
                throw std::runtime_error("prefill span has no compatible layer part");
            workspaces[side] = found->second.workspace;
        }
    }
    for (size_t side = 0; side < 2; ++side) {
        Device& d = *sides[side];
        on_device(d.ordinal, [&]() {
            // (a send left queued by a span that failed half way: finished, and forgotten)
            if (d.send_pending) { check(hipStreamSynchronize(d.send_stream)); d.send_pending = false; }
            if (input) upload(d, workspaces[side], input, steps * 10240 * 4);
            if (steps > 1) kv_prefetch_from(d, begin, position);
        });
    }
    for (size_t layer = begin; layer < end; ++layer)
        for (size_t side = 0; side < 2; ++side) {
            Device& d = *sides[side];
            float* hidden = workspaces[side]; float* mixed = hidden + steps * 10240; float* branch = mixed + steps * 2564;
            on_device(d.ordinal, [&]() { prefill_layer_part(d, layer, position, steps, hidden, mixed, branch, layer > begin, layer + 1 < end); });
        }
    const bool whole = end == 1 || end == 48 || output_rows;
    for (size_t side = 0; side < 2; ++side) {
        Device& d = *sides[side];
        const BatchRows own = batch_rows(steps, side);
        on_device(d.ordinal, [&]() {
            if (whole) qwen_gather_hidden(d, workspaces[side], workspaces[1 - side], own);
            decoder_nonfinite<<<256, 256, 0, d.stream>>>(workspaces[side] + own.first * 10240, own.count * 10240, d.flags + 3);
            check(hipGetLastError());
        });
    }
    if (!output_rows) return;
    unsigned flags[2] = {};
    for (size_t side = 0; side < 2; ++side) {
        Device& d = *sides[side];
        on_device(d.ordinal, [&]() {
            download(d, &flags[side], d.flags + 3, 4);
            check(hipMemsetAsync(d.flags + 3, 0, 4, d.stream));
            if (side == 1) download(d, output, workspaces[side] + (steps - output_rows) * 10240, output_rows * 10240 * 4);
        });
    }
    qwen_sums_settle(a);
    if (flags[0] || flags[1]) throw std::runtime_error("Straydog GPU produced non-finite decoder state");
}

} // namespace

// Split Qwen3.8's layers across this many GPUs (1 or 2): before any of its layers or K/V states are registered.
int sd_qwen_parts(void* handle, uint64_t parts) {
    try {
        Device& d = device(handle);
        if ((parts != 1 && parts != 2) || !d.decoder.empty() || !d.host_states.empty() || d.kv_stage)
            throw std::runtime_error("invalid Straydog layer parts");
#if !defined(__HIPCC__)
        if (parts == 2) throw std::runtime_error("Straydog splits Qwen3.8's layers on RDNA4 only");
#endif
        d.qwen_parts = size_t(parts);
        // STRAYDOG_PARTS_OVERLAP=off: a prompt batch's sends stay in the device stream's order.
        const char* overlap = std::getenv("STRAYDOG_PARTS_OVERLAP");
        d.parts_overlap = !(overlap && std::string(overlap) == "off");
        check(hipMemset(d.flags + 3, 0, 4));   // (prefill_span_parts' accumulated check)
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int sd_decoder_part(void* handle, uint64_t layer, uint64_t context, void* const* pointers, const int* kinds, uint64_t count,
                    void* const* states, void* workspace, void* moe, uint64_t part) {
    try {
        register_decoder_layer(device(handle), size_t(layer), size_t(context), pointers, kinds, size_t(count), states, workspace, moe, 2,
                               size_t(part));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
// Both devices' streams finished, then qwen_sums_settle (after a pass whose parts the host queued on both).
int sd_parts_settle(void* first, void* second) {
    try {
        Device& a = device(first);
        Device& b = device(second);
        if (a.sums.peer != &b) throw std::runtime_error("invalid Straydog parts");
        for (Device* side : {&a, &b}) on_device(side->ordinal, [&]() { check(hipStreamSynchronize(side->stream)); });
        qwen_sums_settle(a);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
// A layer part's resident MoE plan (sd_resident_moe_part, without the shared expert): its half of the shared expert's width,
// rows [320p, +320) of gate and up (320 x 2560) and those columns of down (2560 x 320).
int sd_moe_shared_half(void* handle, void* plan, void* const* pointers, const int* kinds) {
    try {
        Device& d = device(handle);
        auto found = d.resident.find(plan);
        if (found == d.resident.end() || !pointers || !kinds || found->second->shared || found->second->half[0].data ||
            found->second->hybrid)
            throw std::runtime_error("invalid Straydog shared expert half");
        const size_t rows[3] = {320, 320, 2560}, cols[3] = {2560, 2560, 320};
        ResidentMoe& moe = *found->second;
        for (size_t i = 0; i < 3; ++i) {
            const size_t row_bytes = matrix_row_bytes(kinds[i], cols[i]);
            auto allocation = d.allocations.find(pointers[i]);
            if (allocation == d.allocations.end() || allocation->second != rows[i] * row_bytes)
                throw std::runtime_error("invalid Straydog shared expert half");
            moe.half[i] = {static_cast<const uint8_t*>(pointers[i]), row_bytes, rows[i], cols[i], 0, 0, 0, kinds[i]};
            moe.dependencies.push_back(pointers[i]);
        }
        // The table as it is (no shared expert) for prompts; the half as the shared expert for decode.
        const size_t bytes = (moe.experts + 1) * sizeof(MoeGroup);
        if (bytes > d.budget - d.used) throw std::runtime_error("resident MoE metadata exceeds VRAM budget");
        void* table = nullptr;
        check(hipMalloc(&table, bytes));
        d.allocations.emplace(table, bytes); d.used += bytes;
        check(hipMemcpy(table, moe.table, bytes, hipMemcpyDeviceToDevice));
        moe.batch_table = table;
        moe.dependencies.push_back(table);
        for (size_t i = 0; i < 3; ++i) moe.weights[moe.experts * 3 + i] = moe.half[i];
        const MoeGroup shared{{moe.half[0], moe.half[1], moe.half[2]}, 0, 0, 0};
        check(hipMemcpy(static_cast<MoeGroup*>(moe.table) + moe.experts, &shared, sizeof(MoeGroup), hipMemcpyHostToDevice));
        moe.shared = true;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int sd_prefill_parts(void* first, void* second, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, const float* input,
                     float* output, uint64_t output_rows) {
    try {
        prefill_span_parts(device(first), device(second), size_t(begin), size_t(end), size_t(position), size_t(steps), input, output,
                           size_t(output_rows));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
