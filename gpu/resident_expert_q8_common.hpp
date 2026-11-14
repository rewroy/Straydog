#pragma once
// The scratch and the switch of resident_expert_q8.hpp (the routed gate and up on INT8 activations) that the devices' creation and the benchmarks
// reach from kernels.cpp.
// Words a row's quantized activations take: `cols` INT8 values and a scale to every 32.
__host__ __device__ constexpr size_t expert_q8_row_words(size_t cols) { return cols / 4 + cols / 32; }
constexpr unsigned expert_q8_rows_most = 8, expert_q8_share = 4;
// The device's scratch (bytes): the masks of eight sign bits, the masks of seven stored ones, a (row, route)'s entry of twelve words (how
// many rows it evaluates, 0 for none; the rows, each row * 10 + route, itself the first; bit 0 and 1 set where its gate and up are in the
// SoA layout; its gate's and its up's weights), then each row's activations (INT8 words, then a scale to every 32).
constexpr size_t expert_q8_masks8_at = 0, expert_q8_masks7_at = 2048, expert_q8_entries_at = 3072, expert_q8_entry_words = 12,
                 expert_q8_rows_at = expert_q8_entries_at + expert_q8_rows_most * 10 * expert_q8_entry_words * 4;
__host__ __device__ constexpr size_t expert_q8_scratch_bytes(size_t cols) { return expert_q8_rows_at + expert_q8_rows_most * expert_q8_row_words(cols) * 4; }
// Whether a pass's routed gate and up take the kernels above: STRAYDOG_EXPERT_Q8=1; 2, with every (row, route) reading its expert itself
// (the sums a row has are the same either way: the measurement of what the sharing saves, and the check that it changes nothing).
inline int expert_q8_forced = -1;   // (the benchmark's: a level whatever the environment says)
inline int expert_q8_asked = 0;     // (the environment's level, as expert_q8_read_environment found it)
inline int expert_q8_level() {
    if (expert_q8_forced >= 0) return expert_q8_forced;
    return expert_q8_asked;
}
// The switch as the environment has it: read when a device's scratch is made (a model's load), not at every layer's launches, and again at
// the next load (a process that loads a model twice may have changed it between: examples/qwen_floor does).
// Unset, a CUDA build takes level 2 (the routed and the shared gate and up on INT8 activations) and a HIP build none; STRAYDOG_EXPERT_Q8=0
// keeps the F32 kernels, 1 or 2 names the level. On two RTX 3090 (IQ3_S, pipelined) the level took decoding from 132.7 to 146.0 tok/s; its
// outputs against F32's (RTX 5070 Ti, IQ2_XS, 64 tokens of six prompts): KL mean 0.002-0.004 where two F32 runs of that placement differ by
// 0.001 (README, Numerics).
inline void expert_q8_read_environment() {
    const char* p = std::getenv("STRAYDOG_EXPERT_Q8");
#if defined(__HIPCC__)
    const int unset = 0;
#else
    const int unset = 2;
#endif
    expert_q8_asked = !p ? unset : (p[0] == '1' || p[0] == '2') && !p[1] ? p[0] - '0' : 0;
}
// The device's scratch: made with the device (a launch may be recorded into a graph, where nothing can be allocated), for rows of
// expert_q8_scratch_cols values at the most.
constexpr size_t expert_q8_scratch_cols = 2560;
inline void expert_q8_scratch_make(Device& d, bool always = false) {
    expert_q8_read_environment();
#if defined(__HIPCC__)
    // (a HIP build takes these kernels only when asked, or for the benchmark: otherwise nothing is allocated and the device is as it was
    // without them)
    if (!always && !expert_q8_asked) return;
#else
    (void)always;
#endif
    if (!d.expert_q8_scratch) {
        void* pointer = nullptr;
        check(hipMalloc(&pointer, expert_q8_scratch_bytes(expert_q8_scratch_cols)));
        d.expert_q8_scratch = static_cast<uint8_t*>(pointer);
    }
#if !defined(__HIPCC__)
    // (the stream the shared expert's gate and up run on beside the routed ones, resident_expert_launch: made here on CUDA, where a launch
    // may be recorded into a graph. Not on HIP: a stream made this early takes a hardware queue the streams made after it then share
    // differently, and the kernels on INT8 activations are not a HIP build's)
    if (!d.side_stream) {
        check(hipStreamCreate(&d.side_stream));
        check(hipEventCreateWithFlags(&d.side_fork, hipEventDisableTiming));
        check(hipEventCreateWithFlags(&d.side_join, hipEventDisableTiming));
    }
#endif
}
