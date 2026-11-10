// Explicit diagnostic builds only. Existing transfer ordering is unchanged.
#pragma once
#if defined(STRAYDOG_PREFILL_OBSERVER)
#include <cstdio>
uint64_t layer_trace_tick() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
uint64_t layer_trace_read_bytes() {
#if defined(__linux__)
    FILE* file = std::fopen("/proc/self/io", "r");
    if (!file) return ~uint64_t(0);
    char line[160]; unsigned long long value = 0; bool found = false;
    while (std::fgets(line, sizeof line, file)) {
        if (std::sscanf(line, "read_bytes: %llu", &value) == 1) { found = true; break; }
    }
    std::fclose(file);
    if (found) return uint64_t(value);
#endif
    return ~uint64_t(0);
}
struct LayerBankTrace {
    Device& device; size_t layer, bytes;
    uint64_t started = layer_trace_tick(), io = layer_trace_read_bytes();
    uint64_t gather_ns = 0, submit_ns = 0, wait_ns = 0, hint_ns = 0;
    double copy_seconds = 0;
    hipEvent_t first[2]{}, last[2]{}; bool used[2]{};
    LayerBankTrace(Device& d, size_t l, size_t b) : device(d), layer(l), bytes(b) {
        try {
            for (size_t i=0;i<2;++i) {
                check(hipEventCreateWithFlags(&first[i], 0));
                check(hipEventCreateWithFlags(&last[i], 0));
            }
        } catch (...) { release(); throw; }
    }
    void release() {
        for (size_t i=0;i<2;++i) {
            if (first[i]) cleanup(hipEventDestroy(first[i]));
            if (last[i]) cleanup(hipEventDestroy(last[i]));
            first[i] = last[i] = nullptr;
        }
    }
    ~LayerBankTrace() { release(); }
    void collect(size_t i) {
        if (!used[i]) return;
        float ms = 0; check(hipEventElapsedTime(&ms, first[i], last[i]));
        copy_seconds += double(ms) / 1000; used[i] = false;
    }
    // The caller's existing completion event has already settled this buffer before it is reused.
    void before_copy(size_t i) { collect(i); check(hipEventRecord(first[i], device.copy_stream)); }
    void after_copy(size_t i) { check(hipEventRecord(last[i], device.copy_stream)); used[i] = true; }
    void emit() {
        collect(0); collect(1);
        const uint64_t ended = layer_trace_tick(), after = layer_trace_read_bytes();
        const long long reads = io == ~uint64_t(0) || after == ~uint64_t(0) ? -1 : static_cast<long long>(after >= io ? after-io : 0);
        std::fprintf(stderr, "Straydog layer trace: {\"source\":\"native_bank\",\"layer\":%zu,\"logical_bytes\":%zu,\"start_ns\":%llu,\"end_ns\":%llu,\"wall_ns\":%llu,\"gather_ns\":%llu,\"submit_ns\":%llu,\"wait_ns\":%llu,\"hint_ns\":%llu,\"h2d_seconds\":%.9f,\"read_bytes\":%lld}\n",
            layer, bytes, (unsigned long long)started, (unsigned long long)ended, (unsigned long long)(ended-started),
            (unsigned long long)gather_ns, (unsigned long long)submit_ns, (unsigned long long)wait_ns,
            (unsigned long long)hint_ns, copy_seconds, reads);
    }
};
#endif
