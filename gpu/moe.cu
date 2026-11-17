// The experts: the kernels of a layer's routed and shared experts and the host code that runs them, in one translation unit (moe_gemm_cuda.hpp,
// experts.hpp, resident_moe.hpp, resident_expert_q8.hpp, with the grouped GEMM's and the mixture's kernels they launch). kernels.cpp calls it through the
// declarations of the *_common.hpp headers, which also hold the types and device functions the two units share. A kernel is defined here and nowhere
// else; the launches the other units make of these kernels are the functions at the end.
#include "core.hpp"
namespace sdgpu {
#include "phase_profile_common.hpp"
#include "host_io_common.hpp"
#include "peer_common.hpp"
#include "link_tail_common.hpp"
#include "gemv_common.hpp"
#include "expert_soa_common.hpp"
#include "resident_soa_common.hpp"
#include "graph_cache_common.hpp"
#include "hyperconnection_common.hpp"
#include "grouped_gemm_common.hpp"
#include "hybrid_slots_common.hpp"
// (ops.hpp's check of a range against the device's allocations)
void op_span(Device& d, const void* pointer, size_t bytes, const char* what);
#include "grouped_gemm.hpp"
#include "mixture_of_experts_common.hpp"
#include "mixture_of_experts.hpp"
#include "moe_gemm.hpp"
#if !defined(__HIPCC__)
#include "moe_gemm_cuda.hpp"
#endif
#include "resident_moe.hpp"
#include "experts.hpp"

SD_IQ_GRIDS_UPLOAD(iq_grids_upload_moe)
void launch_moe_product(unsigned blocks, hipStream_t stream, const float* gate_up, size_t routes, float* product) {
    moe_product<<<blocks, 256, 0, stream>>>(gate_up, routes, product);
}
void launch_resident_route(unsigned blocks, hipStream_t stream, const float* scores, const float* mix, size_t experts, int32_t* selected,
                           float* probabilities, size_t row_stride, size_t mix_stride) {
    resident_route<<<blocks, 256, 0, stream>>>(scores, mix, experts, selected, probabilities, row_stride, mix_stride);
}
const void* resident_expert_gemv_function() { return reinterpret_cast<const void*>(&resident_expert_gemv); }
const void* resident_expert_gemv_kind_22_function() { return reinterpret_cast<const void*>(&resident_expert_gemv_kind<22>); }
} // namespace sdgpu
using namespace sdgpu;
#include "moe_prefill.hpp"

// A function that throws keeps C++ linkage: outside this block, or in extern "C++" within it.
// MSVC (/EHsc) takes a C function for one that never throws, and drops the catch around its call.
extern "C" {

// ---- A wave of experts in slots through the RDNA4 grouped WMMA kernels (DeepSeek's prefill) ----
// Expert i's slot at bases[i] holds gate and up (ffn_share rows of `width` each, gate_bytes apart)
// then down (hidden_share rows of `ffn`); its routes are the grouped rows firsts[i]..+counts[i].
// The weights decode straight into FP16 WMMA fragments: none is expanded into VRAM first.
#if defined(__HIPCC__)
// The wave's group and tile lists, uploaded into `tables` (stream-ordered); the tiles' count, and
// the grouped rows the wave covers (`rows_out`). Every slot is checked against its allocation;
// the experts' grouped rows follow one another from 0, without gaps or overlaps.
static unsigned slot_moe_tables(Device& d, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts,
                                uint64_t experts, uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share,
                                uint64_t ffn, uint64_t hidden_share, void* tables, uint64_t table_bytes, MoeGroup*& groups,
                                MoeTile*& tiles, size_t& rows_out) {
    if (!bases || !firsts || !counts || !kinds || !experts || experts > 4096 || !tables || !width || !ffn_share || !ffn ||
        !hidden_share || ffn_share > ffn)
        throw std::runtime_error("invalid Straydog slot experts");
    const size_t gate_rows = matrix_row_bytes(kinds[0], size_t(width)), up_rows = matrix_row_bytes(kinds[1], size_t(width));
    const size_t down_bytes = size_t(hidden_share) * matrix_row_bytes(kinds[2], size_t(ffn));
    if (gate_bytes < size_t(ffn_share) * std::max(gate_rows, up_rows)) throw std::runtime_error("Straydog slot expert roles overlap");
    std::vector<MoeGroup> list(experts);
    std::vector<MoeTile> tile_list;
    size_t next = 0;
    for (size_t i = 0; i < experts; ++i) {
        const auto* base = reinterpret_cast<const uint8_t*>(bases[i]);
        if (!base || firsts[i] != next || !counts[i]) throw std::runtime_error("invalid Straydog slot expert routes");
        next += counts[i];
        op_span(d, base, size_t(2 * gate_bytes) + down_bytes, "a slot expert");
        list[i].weights[0] = {base, matrix_row_bytes(kinds[0], size_t(width)), size_t(ffn_share), size_t(width), 0, 0, 0, kinds[0]};
        list[i].weights[1] = {base + gate_bytes, matrix_row_bytes(kinds[1], size_t(width)), size_t(ffn_share), size_t(width), 0, 0, 0,
                              kinds[1]};
        list[i].weights[2] = {base + 2 * gate_bytes, matrix_row_bytes(kinds[2], size_t(ffn)), size_t(hidden_share), size_t(ffn), 0, 0,
                              0, kinds[2]};
        list[i].first = firsts[i];
        list[i].count = counts[i];
        list[i].half = 1;
        for (uint32_t row = 0; row < counts[i]; row += moe_block_m) tile_list.push_back({uint32_t(i), row});
    }
    rows_out = next;
    const size_t group_bytes = (experts * sizeof(MoeGroup) + 255) & ~size_t(255);
    if (group_bytes + tile_list.size() * sizeof(MoeTile) > table_bytes) throw std::runtime_error("Straydog slot expert tables overflow");
    op_span(d, tables, size_t(table_bytes), "slot expert tables");
    groups = static_cast<MoeGroup*>(tables);
    tiles = reinterpret_cast<MoeTile*>(static_cast<uint8_t*>(tables) + group_bytes);
    upload(d, groups, list.data(), experts * sizeof(MoeGroup));
    if (!tile_list.empty()) upload(d, tiles, tile_list.data(), tile_list.size() * sizeof(MoeTile));
    return unsigned(tile_list.size());
}

#endif

int sd_slot_moe_supported(void* handle, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn, uint64_t hidden_share) {
    try {
#if defined(__HIPCC__)
        const auto& value = device(handle);
        const MoeShape gate{uint32_t(width), uint32_t(ffn_share), 1}, down{uint32_t(hidden_share), uint32_t(ffn), 1};
        return value.rdna4 && kinds && moe_shape_fits(gate) && moe_shape_fits(down) && moe_gate_up_unit(kinds[0], kinds[1], gate) &&
               moe_down_unit(kinds[2], down) ? 1 : 0;
#else
        (void)handle; (void)kinds; (void)width; (void)ffn_share; (void)ffn; (void)hidden_share;
        return 0;
#endif
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_slot_moe_gate_up(void* handle, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts, uint64_t experts,
                           uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn,
                           uint64_t hidden_share, const int32_t* rows, uint64_t steps, const void* input, float clamp,
                           const float* weights, void* product, void* tables, uint64_t table_bytes) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    const MoeShape shape{uint32_t(width), uint32_t(ffn_share), 1};
    const int unit = moe_gate_up_unit(kinds ? kinds[0] : 0, kinds ? kinds[1] : 0, shape);
    if (!value.rdna4 || !moe_shape_fits(shape) || !unit || !rows || !input || !product || !steps)
        throw std::runtime_error("invalid Straydog slot expert gate/up");
    op_span(value, input, size_t(steps * width) * 2, "slot expert inputs");
    MoeGroup* groups = nullptr;
    MoeTile* tiles = nullptr;
    size_t grouped = 0;
    const unsigned count = slot_moe_tables(value, bases, firsts, counts, experts, gate_bytes, kinds, width, ffn_share, ffn, hidden_share,
                                           tables, table_bytes, groups, tiles, grouped);
    // (each route's row of the input is the kernel's to bound: device data, as op_gather's rows)
    op_span(value, rows, grouped * 4, "slot expert routes");
    if (weights) op_span(value, weights, grouped * 4, "slot expert route weights");
    op_span(value, product, grouped * size_t(ffn_share) * 2, "slot expert products");
    if (count) {
        #define SD_GATE_UP(format) case format: moe_gate_up_wmma<format, true, 0, 0, 0, true><<<dim3(shape.ffn / (64 * moe_wave_n), count), 256, 0, \
            value.stream>>>(groups, tiles, rows, size_t(steps), shape, static_cast<const _Float16*>(input), static_cast<_Float16*>(product), \
            nullptr, 0, false, nullptr, clamp, weights); break;
        switch (unit) { SD_MOE_UNITS(SD_GATE_UP) }
        #undef SD_GATE_UP
    }
#else
    (void)value; (void)bases; (void)firsts; (void)counts; (void)experts; (void)gate_bytes; (void)kinds; (void)width; (void)ffn_share;
    (void)ffn; (void)hidden_share; (void)rows; (void)steps; (void)input; (void)clamp; (void)weights; (void)product; (void)tables;
    (void)table_bytes;
    throw std::runtime_error("the grouped expert GEMM is RDNA4's");
#endif
    SD_OP_END
}

int sd_op_slot_moe_down(void* handle, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts, uint64_t experts,
                        uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn, uint64_t hidden_share,
                        const void* product, float* outputs, uint64_t out_stride, void* tables, uint64_t table_bytes) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    const MoeShape shape{uint32_t(hidden_share), uint32_t(ffn), 1};
    const int unit = moe_down_unit(kinds ? kinds[2] : 0, shape);
    if (!value.rdna4 || !moe_shape_fits(shape) || !unit || !product || !outputs || out_stride < hidden_share)
        throw std::runtime_error("invalid Straydog slot expert down");
    MoeGroup* groups = nullptr;
    MoeTile* tiles = nullptr;
    size_t grouped = 0;
    const unsigned count = slot_moe_tables(value, bases, firsts, counts, experts, gate_bytes, kinds, width, ffn_share, ffn, hidden_share,
                                           tables, table_bytes, groups, tiles, grouped);
    op_span(value, product, grouped * size_t(ffn) * 2, "slot expert products");
    op_span(value, outputs, ((grouped - 1) * size_t(out_stride) + size_t(hidden_share)) * 4, "slot expert outputs");
    if (count) {
        #define SD_DOWN(format) case format: moe_down_wmma<format><<<dim3(shape.hidden / (128 * moe_wave_n), count), 256, 0, \
            value.stream>>>(groups, tiles, nullptr, shape, static_cast<const _Float16*>(product), outputs, nullptr, 0, false, \
            size_t(out_stride)); break;
        switch (unit) { SD_MOE_UNITS(SD_DOWN) }
        #undef SD_DOWN
    }
#else
    (void)value; (void)bases; (void)firsts; (void)counts; (void)experts; (void)gate_bytes; (void)kinds; (void)width; (void)ffn_share;
    (void)ffn; (void)hidden_share; (void)product; (void)outputs; (void)out_stride; (void)tables; (void)table_bytes;
    throw std::runtime_error("the grouped expert GEMM is RDNA4's");
#endif
    SD_OP_END
}


int sd_op_half_values(void* handle, const float* input, uint64_t values, void* output) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    if (!input || !output || !values || values % 4) throw std::runtime_error("invalid Straydog half conversion");
    op_span(value, input, size_t(values) * 4, "values to round");
    op_span(value, output, size_t(values) * 2, "rounded values");
    moe_input_half<<<unsigned((values / 4 + 255) / 256), 256, 0, value.stream>>>(size_t(values), input, static_cast<_Float16*>(output));
#else
    (void)value; (void)input; (void)values; (void)output;
    throw std::runtime_error("FP16 rounding is RDNA4's here");
#endif
    SD_OP_END
}
} // extern C
