#pragma once
// experts.hpp, in moe.cu: the decode over a cache of whole experts and the prefill's test driver, called from kernels.cpp's API.
#include "link_tail_common.hpp"
// ---- Experts in slots: decode over a cache of whole experts ----
// A route's expert is a slot at bases[route] holding its gate, up and down matrices at byte offsets
// role[r].data within it (a slot cache's layout, or a dense expert's); a zero base is a route this
// device does not run (another holds its expert). One launch per stage takes every route of
// `tokens` rows: gate and up rows share a lane group, so SwiGLU (clamped, times the route's
// weight) is formed before the store; down sums a row over the token's routes into `out`.
// With experts split across devices (tensor parallel), a device holds rows of each: product rows are
// product_stride apart with this device's at product_first, and its down rows land at out_first.
// A uniform base (every route the same slot, as the shared expert's) is passed by value.
// Routes whose slots an experts call passes by value (in its launch's arguments): a decode row's,
// or a verify pass's (4 rows of 6); more go through an upload, a copy kernel before the launch.
constexpr size_t slot_inline_routes = 32;
struct SlotExperts {
    Matrix role[3];
    uint32_t top_k, lanes[2], waves[2], accumulate, product_stride, product_first, out_first;
    float clamp;
    uint64_t uniform;
    // With a table: route r takes the slot table[ids[r]] (none for an id outside the table).
    const uint64_t* table;
    const int32_t* ids;
    uint32_t table_count;
    // A few routes' slots by value (a decode row's, a verify pass's), else from `bases` in device
    // memory.
    uint32_t inline_count;
    uint64_t inline_bases[slot_inline_routes];
    // A verify pass's gate/up: its distinct experts, each its routes as four bytes (255 past the last).
    uint32_t groups;
    uint32_t group_routes[slot_inline_routes];
};
// (passed by value: well inside a launch's 4 KiB of arguments)
static_assert(sizeof(SlotExperts) <= 1024, "SlotExperts outgrew its launch arguments");
bool slot_experts(Device& d, SlotExperts plan, int stages, const uint64_t* bases, size_t tokens, const float* input,
                  size_t input_stride, const float* weights, float* product, float* out, size_t out_stride,
                  const LinkTail* tail = nullptr, size_t send_at = 0, int raise = 1);
void expert_rows(Device& d, void* const* pointers, const int* kinds, size_t experts, size_t hidden, size_t ffn,
                 size_t steps, size_t top_k, const int32_t* selected, int path, const float* input,
                 const float* down_input, float* gate_up, float* product, float* output, double* timings);
