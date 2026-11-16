#pragma once
// What the experts' unit (moe.cu) takes from hybrid_slots.hpp: the partially resident layers' slots, the copies of the experts a prompt batch
// streams in chunks and the prestage of a wide batch (CUDA), as declarations of the functions kernels.cpp defines and the template that
// drives a layer's streamed experts.
#include "phase_profile_common.hpp"
#include "grouped_gemm_common.hpp"
constexpr size_t hybrid_point_limit = 64;
inline uint8_t* expert_stage_pointer(const Device& d) {
    return d.slots.segmented ? d.slots.base : d.slots.base+d.slots.arena;
}
struct StreamEvents {
    hipEvent_t loaded[2]{}, used[2]{};
    StreamEvents() {
        for (int i = 0; i < 2; ++i) {
            check(hipEventCreateWithFlags(&loaded[i], hipEventDisableTiming));
            check(hipEventCreateWithFlags(&used[i], hipEventDisableTiming));
        }
    }
    ~StreamEvents() { for (int i = 0; i < 2; ++i) { cleanup(hipEventDestroy(loaded[i])); cleanup(hipEventDestroy(used[i])); } }
};
extern "C" int sd_cpu_gather(void*, const uint8_t* const*, uint8_t* const*, const uint64_t*, uint32_t);
extern "C" const char* sd_cpu_error();
constexpr size_t hybrid_prompt_tail = 64;
size_t expert_slot_bytes(const ResidentMoe& plan);
void prefetch_experts(Device& d, ResidentMoe& plan, const size_t* experts, size_t count);
// (the experts' gate, up and down slices from the mapped banks, or from `bank_sources`, into the slot images at `slots`)
void hybrid_gather(Device& d, const ResidentMoe& plan, const size_t* experts, size_t count, uint8_t* const* slots,
                   const uint8_t* const* bank_sources = nullptr);
void hybrid_point_experts(Device& d, ResidentMoe& plan, const size_t* experts, size_t count, const uint8_t* base,
                          uint8_t* const* regions = nullptr);
void hybrid_observe_batch(ResidentMoe& plan, const unsigned* counts, size_t tokens, const int32_t* tail, size_t tail_tokens);
// The deferred observations of a layer-order pass's chunks (Device::deferred): those whose copies have landed, oldest first, or with `all`
// every one, each waited for.
void observe_deferred(Device& d, bool all);
// Copies a chunk's expert counts and last routes to the host behind its routing, on the device's stream, to be observed by observe_deferred;
// false (nothing done) where an entry cannot hold them.
bool defer_observation(Device& d, ResidentMoe& plan, const unsigned* counts, size_t groups_count, const int32_t* selected, size_t steps);
// Whether an entry holds a chunk's counts and last routes (defer_observation does not refuse them).
constexpr size_t deferred_entry_bytes = 16384, deferred_tail_offset = 8192, deferred_slots = 256;
inline bool observation_deferrable(size_t groups_count, size_t steps) {
    return groups_count * 4 <= deferred_tail_offset && deferred_tail_offset + std::min<size_t>(steps, hybrid_prompt_tail) * 40 <= deferred_entry_bytes;
}
#if !defined(__HIPCC__)
void hybrid_point_staged(Device& d, ResidentMoe& plan, const size_t* experts, size_t count, bool clear);
void prestage_issue(Device& d, ResidentMoe& plan);
inline size_t prestage_capacity(const Device& d) { return d.slots.stage > d.slots.chunk ? d.slots.stage - d.slots.chunk : 0; }
inline void prestage_regions(const Device& d, const ResidentMoe& plan, uint8_t* (&regions)[3]) {
    uint8_t* base = expert_stage_pointer(d) + d.slots.chunk * d.slots.bytes;
    const size_t capacity = prestage_capacity(d);
    regions[0] = base;
    regions[1] = regions[0] + capacity * plan.host_bytes[0];
    regions[2] = regions[1] + capacity * plan.host_bytes[1];
}
// Whether this layer's prompt batch of `steps` rows may prestage: wide, the banks page-locked, which the copy engine reads asynchronously (the
// engine's copies of them, or the model's mapping registered in place, which sd_hybrid_assist has then mapped), and no mirror to gather through.
// (The mapping registered in place was left out: two RTX 3090 on Linux, whose driver registers host memory read-only, never prestaged, and
// streamed every expert beyond their slots in chunks whose copies wait for the GEMMs two chunks back.)
inline bool prestage_wanted(const Device& d, const ResidentMoe& plan, size_t steps) {
    const bool locked = plan.host_copy || plan.mapped[0];
    return prestage_capacity(d) > 0 && steps >= 2048 && plan.hybrid && !plan.layer_prefill && locked && d.slots.base &&
           !(d.slots.host && d.cpu && !plan.mapped[0]) && d.copy_stream;
}
// A layer's prestage over its prompt batch, to every way out of it, an error included: the table entries pointed at prestaged copies are emptied
// (the next layer's prestage overwrites the region they point into), the copies and GEMMs that read it drained, and the layer is no longer the
// prestaged one. `end` is the way out without an error, the destructor the way with one (each step by itself, so that one that fails does not skip
// the others; the failure is the caller's, already thrown).
struct PrestageUse {
    Device& d; ResidentMoe& plan; const bool active; std::vector<size_t> pointed; bool ended = false;
    PrestageUse(Device& device, ResidentMoe& layer, bool wanted) : d(device), plan(layer), active(wanted) {}
    void point(const std::vector<size_t>& experts) {
        pointed = experts;
        hybrid_point_staged(d, plan, pointed.data(), pointed.size(), false);
    }
    void end() {
        if (!active) return;
        if (!pointed.empty()) hybrid_point_staged(d, plan, pointed.data(), pointed.size(), true);
        d.prestage.plan = nullptr;
        ended = true;
    }
    ~PrestageUse() {
        if (!active || ended) return;
        try { if (!pointed.empty()) hybrid_point_staged(d, plan, pointed.data(), pointed.size(), true); } catch (...) {}
        cleanup(hipStreamSynchronize(d.stream));
        cleanup(hipStreamSynchronize(d.copy_stream));
        d.prestage.plan = nullptr;
    }
};
#endif
// compute(first, count) launches the expert GEMMs for missing[first, first+count); meanwhile() runs on this thread while they
// stream, before they are waited for. The GPU's time from the first copy to the last expert's GEMMs, in ns (0 with none missing).
template<class Compute, class Meanwhile>
uint64_t hybrid_stream_missing(Device& d, ResidentMoe& plan, const std::vector<size_t>& missing, Compute&& compute, Meanwhile&& meanwhile) {
    if (missing.empty()) { meanwhile(); return 0; }
    if (d.copier) d.copier->wait();   // the copy stream and read-ahead state are this thread's again
    struct Span {
        hipEvent_t began = nullptr, ended = nullptr;
        ~Span() { if (began) cleanup(hipEventDestroy(began)); if (ended) cleanup(hipEventDestroy(ended)); }
    } span;
    // What a layer's streamed experts leave behind on every way out of here, an error included: their table entries pointing into the chunks'
    // buffers (the next layer's chunks go through the same buffers, and its GEMMs would read another layer's bytes through a stale entry), and
    // both streams holding copies and GEMMs that read those buffers. `settle` is the way out without an error, the destructor the way with one
    // (each step by itself, so that one that fails does not skip the others; the failure is the caller's, already thrown).
    struct Settle {
        Device& d; ResidentMoe& plan; const std::vector<size_t>& missing; bool settled = false;
        void settle() {
            hybrid_point_experts(d, plan, missing.data(), missing.size(), nullptr);
            check(hipStreamSynchronize(d.copy_stream));
            check(hipStreamSynchronize(d.stream));
            settled = true;
        }
        ~Settle() {
            if (settled) return;
            try { hybrid_point_experts(d, plan, missing.data(), missing.size(), nullptr); } catch (...) {}
            cleanup(hipStreamSynchronize(d.copy_stream));
            cleanup(hipStreamSynchronize(d.stream));
        }
    } streamed{d, plan, missing};
    check(hipEventCreate(&span.began)); check(hipEventCreate(&span.ended));
    check(hipEventRecord(span.began, d.copy_stream));
    prefetch_experts(d, plan, missing.data(), missing.size());
    const size_t half = std::min(d.slots.chunk / 2, hybrid_point_limit);
    // Pageable banks: the CPU workers gather each chunk into the page-locked mirror while the GPU
    // computes the previous one, so cold pages fault in concurrently instead of one copy at a time
    // (cached ones pay an extra pass through host memory); the experts then copy by DMA.
    // Page-locked (registered) banks copy directly.
    const bool gather = d.slots.host && d.cpu && !plan.mapped[0];
    uint64_t gathered_ns = 0;
    StreamEvents events;
    std::vector<uint8_t*> targets;
    for (size_t first = 0, chunk = 0; first < missing.size(); first += half, ++chunk) {
        const size_t count = std::min(half, missing.size() - first), buffer = chunk % 2;
        uint8_t* base = expert_stage_pointer(d) + buffer * half * d.slots.bytes;
        uint8_t* host = gather ? d.slots.host + buffer * half * d.slots.bytes : nullptr;
        if (gather) {
            // The DMA two chunks back read this mirror (its `loaded` event follows it on the copy stream).
            if (chunk >= 2) check(hipEventSynchronize(events.loaded[buffer]));
            targets.clear();
            for (size_t i = 0; i < count; ++i) targets.push_back(host + i * d.slots.bytes);
            const auto began = std::chrono::steady_clock::now();
            hybrid_gather(d, plan, missing.data() + first, count, targets.data());
            gathered_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
        }
        if (chunk >= 2) check(hipStreamWaitEvent(d.copy_stream, events.used[buffer], 0));
        // Straight from the banks, a chunk's matrices land role by role, so a run of consecutive experts is one copy a role: three copies
        // an expert, each a third of a megabyte, kept the bus at 19 GB/s (22 so) on a Radeon AI PRO R9700.
        uint8_t* regions[3] = {base, base + half * plan.host_bytes[0], base + half * (plan.host_bytes[0] + plan.host_bytes[1])};
        if (gather) {
            for (size_t i = 0; i < count; ++i)
                check(hipMemcpyAsync(base + i * d.slots.bytes, host + i * d.slots.bytes, expert_slot_bytes(plan), hipMemcpyHostToDevice,
                                     d.copy_stream));
        } else {
            for (size_t i = 0; i < count;) {
                size_t j = i + 1;
                while (j < count && missing[first + j] == missing[first + j - 1] + 1) ++j;
                for (size_t role = 0; role < 3; ++role)
                    check(hipMemcpyAsync(regions[role] + i * plan.host_bytes[role], plan.host[role] + missing[first + i] * plan.host_bytes[role],
                                         (j - i) * plan.host_bytes[role], hipMemcpyHostToDevice, d.copy_stream));
                i = j;
            }
        }
        check(hipEventRecord(events.loaded[buffer], d.copy_stream));
        check(hipStreamWaitEvent(d.stream, events.loaded[buffer], 0));
        phase(d, "moe_stream_wait");
        hybrid_point_experts(d, plan, missing.data() + first, count, base, gather ? nullptr : regions);
        compute(first, count);
        phase(d, "moe_stream_gemm");
        if (fail_streamed_chunk.load() && fail_streamed_chunk.fetch_sub(1) == 1) throw std::runtime_error("Straydog test: a streamed chunk fails");
        check(hipEventRecord(events.used[buffer], d.stream));
        d.streamed_experts += count;
    }
    check(hipEventRecord(span.ended, d.stream));
    meanwhile();
    streamed.settle();
    float ms = 0;
    check(hipEventElapsedTime(&ms, span.began, span.ended));
    // (what a gathered expert cost the CPU's workers: prompt_cpu_share weighs a route against it)
    if (gather && missing.size() >= 16) {
        const double sample = double(gathered_ns) / double(missing.size());
        d.prompt_gather_ns = d.prompt_gather_ns > 0 ? 0.7 * d.prompt_gather_ns + 0.3 * sample : sample;
    }
    return uint64_t(double(ms) * 1e6);
}
