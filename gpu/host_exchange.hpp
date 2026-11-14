// Host exchanges (straydog_gpu.h: sd_exchange): page-locked host memory a device's kernels read and
// write within a pass, so a host thread takes part in a pass queued (or recorded as a graph) ahead:
// the pass exports values with a running count, the host computes from them and publishes its
// result, and a later op of the pass waits for that result on the device. No host call sits
// between the ops, so the device never waits for a launch.
//
// Header words (uint32, from the exchange's start): exchange_exported (exports so far, the
// device's copy for the host), exchange_published (results the host published), exchange_failed
// (nonzero once an await gave up), exchange_cancel (nonzero: the host cancels the awaits, which
// then take zeros). The device counts its exports in device memory as well, so an await reads its
// own target without a bus round trip.
constexpr size_t exchange_header = 256;
constexpr size_t exchange_exported = 0, exchange_published = 1, exchange_failed = 2, exchange_cancel = 3;
constexpr size_t exchange_parts = 4;

struct ExchangeExport { const float* sources[exchange_parts]; uint32_t counts[exchange_parts]; uint32_t parts; };

// The exchange starting at host address `host` with [offset, offset + bytes) inside it.
ExchangeArea& exchange_area(Device& d, const void* host, size_t offset, size_t bytes) {
    const auto found = d.exchanges.find(const_cast<void*>(host));
    if (found == d.exchanges.end()) throw std::runtime_error("Straydog op: no such host exchange on this device");
    ExchangeArea& area = found->second;
    if (offset < exchange_header || bytes > area.bytes || offset > area.bytes - bytes || offset % 4)
        throw std::runtime_error("Straydog op: a range outside its host exchange");
    return area;
}

__device__ inline uint64_t exchange_clock() {
#if defined(__HIPCC__)
    return wall_clock64();
#else
    uint64_t now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); return now;
#endif
}

// One block: the parts one after another into the host's memory, then the count. Every thread's
// stores reach system scope before the barrier; the count follows the barrier.
__global__ void exchange_export_kernel(ExchangeExport parts, volatile float* target, uint32_t* count, volatile uint32_t* header) {
    size_t at = 0;
    // (the parts at constant indices, the loop unrolled to its most: an index the thread computes into the argument keeps the whole argument in the
    // thread's stack, written at the kernel's entry)
    #pragma unroll
    for (uint32_t part = 0; part < exchange_parts; ++part) {
        if (part >= parts.parts) break;
        for (size_t i = threadIdx.x; i < parts.counts[part]; i += blockDim.x) target[at + i] = parts.sources[part][i];
        at += parts.counts[part];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t next = *count + 1;
        *count = next;
        header[exchange_exported] = next;
        __threadfence_system();
    }
}

// Each block waits until the host has published as many results as there were exports (or the
// host cancels, or `timeout` ticks pass: then the failure word is raised), then adds its share:
// out[i] += a[i] + result[i] (the result taken as zero when the wait gave up).
__global__ void exchange_await_kernel(volatile uint32_t* header, const uint32_t* count, const volatile float* result, const float* a,
                                      float* out, size_t values, uint64_t timeout) {
    __shared__ uint32_t ready;
    if (threadIdx.x == 0) {
        const uint32_t expected = *count;
        const uint64_t start = exchange_clock();
        ready = 1;
        while (int32_t(header[exchange_published] - expected) < 0) {
            if (header[exchange_cancel]) { ready = 0; break; }
            if (exchange_clock() - start > timeout) { header[exchange_failed] = 1; ready = 0; break; }
#if defined(__HIPCC__)
            __builtin_amdgcn_s_sleep(2);
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < values) out[i] += (a ? a[i] : 0.0f) + (ready ? result[i] : 0.0f);
}

extern "C" {
// Host memory a device's ops exchange values with during a pass; freed with the device.
void* sd_exchange(void* handle, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (bytes < exchange_header || bytes > (uint64_t(1) << 30) || bytes % 4) throw std::runtime_error("invalid Straydog host exchange");
        if (!value.hybrid_prelaunch || !value.await_timeout)
            throw std::runtime_error("this GPU cannot wait for the host within a pass");
        ExchangeArea area;
        area.bytes = size_t(bytes);
        check(hipHostMalloc(reinterpret_cast<void**>(&area.host), area.bytes, hipHostMallocDefault));
        try {
            std::memset(area.host, 0, area.bytes);
            check(hipHostGetDevicePointer(reinterpret_cast<void**>(&area.view), area.host, 0));
            area.count = static_cast<uint32_t*>(allocate(value, 256));
            // (zeroed in stream order before any op uses it: no wait, so a stream that waits for
            // another device's work may take an exchange)
            check(hipMemsetAsync(area.count, 0, 256, value.stream));
            value.exchanges.emplace(area.host, area);
        } catch (...) {
            cleanup(hipHostFree(area.host));
            throw;
        }
        return area.host;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

// Whether this device's ops can wait for the host within a pass (sd_exchange).
int sd_exchange_supported(void* handle) {
    try {
        auto& value = device(handle);
        return value.hybrid_prelaunch && value.await_timeout ? 1 : 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_export(void* handle, const float* const* sources, const uint64_t* counts, uint64_t parts, void* exchange, uint64_t offset) {
    SD_OP_BEGIN
    if (!sources || !counts || !parts || parts > exchange_parts) throw std::runtime_error("invalid Straydog export");
    ExchangeExport list{};
    size_t total = 0;
    for (size_t part = 0; part < parts; ++part) {
        if (!counts[part] || counts[part] > (uint64_t(1) << 20)) throw std::runtime_error("invalid Straydog export");
        op_span(value, sources[part], size_t(counts[part]) * 4, "an exported buffer");
        list.sources[part] = sources[part];
        list.counts[part] = uint32_t(counts[part]);
        total += size_t(counts[part]);
    }
    list.parts = uint32_t(parts);
    ExchangeArea& area = exchange_area(value, exchange, size_t(offset), total * 4);
    exchange_export_kernel<<<1, 256, 0, value.stream>>>(list, reinterpret_cast<float*>(area.view + offset), area.count,
                                                         reinterpret_cast<uint32_t*>(area.view));
    SD_OP_END
}

int sd_op_await_add(void* handle, void* exchange, uint64_t result_offset, const float* a, float* out, uint64_t count) {
    SD_OP_BEGIN
    if (!out || !count || count > (uint64_t(1) << 20)) throw std::runtime_error("invalid Straydog await");
    ExchangeArea& area = exchange_area(value, exchange, size_t(result_offset), size_t(count) * 4);
    if (a) op_span(value, a, size_t(count) * 4, "an awaited addend");
    op_span(value, out, size_t(count) * 4, "an awaited output");
    exchange_await_kernel<<<op_blocks(size_t(count)), 256, 0, value.stream>>>(reinterpret_cast<uint32_t*>(area.view), area.count,
        reinterpret_cast<const float*>(area.view + result_offset), a, out, size_t(count), value.await_timeout);
    SD_OP_END
}
}  // extern C
