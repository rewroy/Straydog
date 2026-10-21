// Routed experts of many rows at once (a prefill pass) on RDNA4, from experts that stay in their
// slots: the routes grouped by expert on the device and run as moe_gemm.hpp's grouped WMMA GEMM
// (FP16 operands, F32 sums: the explicit FP16 GEMM mode), each expert's weights read once a tile of
// 64 of its rows instead of once a route. An expert table (sd_moe_groups) holds a layer's experts'
// matrices; an entry without weights is an expert another device runs (tensor parallel), whose
// routes this device leaves out. Nothing waits for the host: the tile list is built on the device
// and the grids cover every tile there can be. The grouping's order within an expert does not
// matter: a row's sums are its own. Included in the GPU TU after moe_gemm.hpp.

#if defined(__HIPCC__)
namespace {

__global__ void moe_prefill_clear(MoeGroup* groups, unsigned* cursors, size_t count) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index < count) { groups[index].count = 0; cursors[index] = 0; }
}
__global__ void moe_prefill_histogram(MoeGroup* groups, const int32_t* ids, size_t routes) {
    const size_t route = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (route < routes) atomicAdd(reinterpret_cast<unsigned long long*>(&groups[size_t(ids[route])].count), 1ull);
}
__global__ void moe_prefill_scatter(const MoeGroup* groups, unsigned* cursors, const int32_t* ids, size_t routes,
                                    int32_t* positions) {
    const size_t route = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (route >= routes) return;
    const size_t expert = size_t(ids[route]);
    positions[groups[expert].first + atomicAdd(cursors + expert, 1u)] = int32_t(route);
}
// out[t] += the token's routes' outputs (the experts this device holds), each times its weight,
// summed in route order.
__global__ void moe_prefill_combine(const MoeGroup* groups, const int32_t* ids, const float* weights, const float* outputs,
                                    size_t top_k, size_t hidden, float* out) {
    const size_t col = size_t(blockIdx.x) * 256 + threadIdx.x, token = blockIdx.y;
    if (col >= hidden) return;
    float sum = 0.0f;
    for (size_t k = 0; k < top_k; ++k) {
        const size_t route = token * top_k + k;
        if (groups[size_t(ids[route])].weights[0].data) sum += weights[route] * outputs[route * hidden + col];
    }
    out[token * hidden + col] += sum;
}

size_t moe_prefill_align(size_t bytes) { return (bytes + 255) & ~size_t(255); }

// The scratch of a pass of `routes` routes over `rows` rows: cursors, counts, firsts, the tile
// count, the tiles, the grouped positions, the FP16 input and products, the routes' F32 outputs.
struct MoePrefillScratch {
    unsigned *cursors, *counts, *tile_count;
    uint32_t* firsts;
    MoeTile* tiles;
    int32_t* positions;
    _Float16 *input, *product;
    float* outputs;
    size_t max_tiles, bytes;
};
MoePrefillScratch moe_prefill_scratch(uint8_t* base, size_t experts, size_t rows, size_t routes, size_t hidden, size_t ffn) {
    MoePrefillScratch s{};
    size_t at = 0;
    auto take = [&](size_t bytes) { uint8_t* p = base + at; at += moe_prefill_align(bytes); return p; };
    s.max_tiles = (routes + moe_block_m - 1) / moe_block_m + experts;
    s.cursors = reinterpret_cast<unsigned*>(take(experts * 4));
    s.counts = reinterpret_cast<unsigned*>(take(experts * 4));
    s.firsts = reinterpret_cast<uint32_t*>(take(experts * 4));
    s.tile_count = reinterpret_cast<unsigned*>(take(16));
    s.tiles = reinterpret_cast<MoeTile*>(take(s.max_tiles * sizeof(MoeTile)));
    s.positions = reinterpret_cast<int32_t*>(take(routes * 4));
    s.input = reinterpret_cast<_Float16*>(take(rows * hidden * 2));
    s.product = reinterpret_cast<_Float16*>(take(routes * ffn * 2));
    s.outputs = reinterpret_cast<float*>(take(routes * hidden * 4));
    s.bytes = at;
    return s;
}

} // namespace
#endif

uint64_t sd_moe_group_bytes() { return sizeof(MoeGroup); }

int sd_moe_groups(void* handle, void* groups, const uint64_t* bases, uint64_t experts, const uint64_t* offsets, const int* kinds,
                  uint64_t hidden, uint64_t ffn) {
    SD_OP_BEGIN
    if (!bases || !offsets || !kinds || !experts || experts > 65536 || !hidden || !ffn) throw std::runtime_error("invalid Straydog expert groups");
    op_span(value, groups, size_t(experts) * sizeof(MoeGroup), "expert groups");
    std::vector<MoeGroup> table(experts);
    for (size_t expert = 0; expert < experts; ++expert) {
        MoeGroup& group = table[expert];
        for (size_t role = 0; role < 3; ++role) {
            const size_t rows = role == 2 ? hidden : ffn, cols = role == 2 ? ffn : hidden;
            const size_t row_bytes = matrix_row_bytes(kinds[role], cols);
            const uint8_t* data = bases[expert] ? reinterpret_cast<const uint8_t*>(bases[expert] + offsets[role]) : nullptr;
            if (data) op_span(value, data, rows * row_bytes, "an expert matrix");
            group.weights[role] = {data, row_bytes, rows, cols, 0, 0, 0, kinds[role]};
        }
    }
    upload(value, groups, table.data(), table.size() * sizeof(MoeGroup));
    SD_OP_END
}

int sd_op_experts_grouped(void* handle, void* groups, uint64_t experts, const int32_t* ids, const float* weights, uint64_t rows,
                          uint64_t top_k, uint64_t hidden, uint64_t ffn, const int* kinds, const float* input, void* scratch,
                          uint64_t scratch_bytes, float* out) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    const MoeShape shape{uint32_t(hidden), uint32_t(ffn), uint32_t(top_k)};
    if (!value.rdna4 || !moe_shape_fits(shape)) return 1;   // not taken: the caller runs the experts otherwise
    if (!rows || !top_k || top_k > 64 || !experts || experts > 65536 || rows * top_k > (size_t(1) << 24))
        throw std::runtime_error("invalid Straydog grouped experts");
    const size_t routes = size_t(rows * top_k);
    op_span(value, groups, size_t(experts) * sizeof(MoeGroup), "expert groups");
    op_span(value, ids, routes * 4, "expert routes");
    op_span(value, weights, routes * 4, "route weights");
    op_span(value, input, size_t(rows * hidden) * 4, "an expert input");
    op_span(value, out, size_t(rows * hidden) * 4, "an expert output");
    MoePrefillScratch s = moe_prefill_scratch(static_cast<uint8_t*>(scratch), size_t(experts), size_t(rows), routes, size_t(hidden),
                                              size_t(ffn));
    if (s.bytes > scratch_bytes) throw std::runtime_error("Straydog grouped experts need more scratch");
    op_span(value, scratch, s.bytes, "grouped experts' scratch");
    auto* table = static_cast<MoeGroup*>(groups);
    const unsigned route_blocks = unsigned((routes + 255) / 256), group_blocks = unsigned((experts + 255) / 256);
    moe_prefill_clear<<<group_blocks, 256, 0, value.stream>>>(table, s.cursors, size_t(experts));
    moe_prefill_histogram<<<route_blocks, 256, 0, value.stream>>>(table, ids, routes);
    moe_counts<<<group_blocks, 256, 0, value.stream>>>(table, size_t(experts), s.counts);
    moe_device_tiles<<<1, moe_tile_threads, 0, value.stream>>>(table, size_t(experts), s.counts, s.firsts, s.tiles, s.tile_count);
    moe_set_groups<<<group_blocks, 256, 0, value.stream>>>(table, size_t(experts), s.firsts);
    moe_prefill_scatter<<<route_blocks, 256, 0, value.stream>>>(table, s.cursors, ids, routes, s.positions);
    moe_input_half<<<unsigned((rows * hidden / 4 + 255) / 256), 256, 0, value.stream>>>(size_t(rows * hidden), input, s.input);
    const MoeKinds routed{kinds[0], kinds[1], kinds[2]};
    moe_expert_launch(value.stream, routed, routed, table, s.tiles, s.positions, size_t(rows), shape, s.input, s.product, s.outputs,
                      s.tile_count, unsigned(s.max_tiles), 0);
    check(hipGetLastError());
    moe_prefill_combine<<<dim3(unsigned((hidden + 255) / 256), unsigned(rows)), 256, 0, value.stream>>>(
        table, ids, weights, s.outputs, size_t(top_k), size_t(hidden), out);
#else
    (void)value; (void)groups; (void)experts; (void)ids; (void)weights; (void)rows; (void)top_k; (void)hidden; (void)ffn; (void)kinds;
    (void)input; (void)scratch; (void)scratch_bytes; (void)out;
    return 1;
#endif
    SD_OP_END
}
