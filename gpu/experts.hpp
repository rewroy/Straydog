// The formats' routed expert kernels at any layer shape: banks of `experts` gate and up
// (ffn x hidden) and down (hidden x ffn) matrices, each token's row routed to top_k of
// them. Decode runs gemv.hpp's units per route (F32); prefill on RDNA4 runs moe_gemm.hpp's
// grouped WMMA GEMM (FP16 operands, F32 sums). sd_expert_rows drives both, for tests and
// measurements.

// Expert e's matrix of a role is the role's first matrix moved by e strides.
struct ExpertBanks { Matrix role[3]; size_t stride[3]; };
struct RoutedGemv { ExpertBanks banks; uint32_t top_k, waves[2], lanes[2]; bool down; };

// A wave's own values, kept in scalar registers on RDNA.
#if defined(__HIPCC__)
#define SD_WAVE_UNIFORM(value) __builtin_amdgcn_readfirstlane(value)
#else
#define SD_WAVE_UNIFORM(value) (value)
#endif
// Formats whose light units gain from two rows per lane group (measured on RDNA4 at DeepSeek-V4.1's
// expert shapes); heavier units (K-quants, Q2_K) lose to the registers it takes.
__host__ __device__ constexpr bool expert_pairs(int kind) {
    return kind == 16 || kind == 18 || kind == 19 || kind == 21 || kind == 29;
}
// gemv_rows with two rows per lane group: both rows' units read the same activations, which are
// loaded (and for the biased formats summed) once. Each row's units, their order and its reduction
// are gemv_rows'. A second row past the end repeats the first and is not stored.
template<int kind>
__device__ void expert_row_pair(const Matrix& m, size_t first_row, unsigned lanes, const float* input, float* output) {
    const unsigned lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = first_row + 2 * (lane / lanes), last = m.rows - 1;
    const uint8_t* rows[2] = {m.data + std::min(row, last) * m.row_bytes, m.data + std::min(row + 1, last) * m.row_bytes};
    const float* x = input + m.input_offset;
    const size_t units = m.cols / GemvFormat<kind>::elements;
    float sum[2] = {};
    for (size_t unit = sub; unit < units; unit += lanes) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float dot[1];
            GemvFormat<kind>::template dots<1>(rows[r], unit, x, 0, dot);
            sum[r] += dot[0];
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        float total = sum[r];
        for (unsigned offset = lanes / 2; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (sub == 0 && row + r < m.rows) output[m.output_offset + row + r] = total;
    }
}
// blockIdx.y selects a token. The waves cover every route's gate rows, then every route's up
// rows (or every route's down rows). kind >= 0: that one format's unit for every matrix of the
// stage (two rows per lane group where expert_pairs); -1: the generic dispatch.
template<int kind>
__global__ void expert_gemv(RoutedGemv plan, const int32_t* selected, const float* input, size_t input_stride,
                            float* output, size_t output_stride) {
    size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    selected += size_t(blockIdx.y) * plan.top_k;
    input += size_t(blockIdx.y) * input_stride;
    output += size_t(blockIdx.y) * output_stride;
    const size_t roles = plan.down ? 1 : 2, ffn = plan.banks.role[2].cols, hidden = plan.banks.role[2].rows;
    for (size_t r = 0; r < roles; ++r) {
        const size_t span = size_t(plan.top_k) * plan.waves[r];
        if (wave >= span) { wave -= span; continue; }
        const size_t route = wave / plan.waves[r], local = wave % plan.waves[r], role = plan.down ? 2 : r;
        Matrix m = plan.banks.role[role];
        m.data += size_t(SD_WAVE_UNIFORM(selected[route])) * plan.banks.stride[role];
        m.input_offset = plan.down ? route * ffn : 0;
        m.output_offset = plan.down ? route * hidden : (2 * route + r) * ffn;
        if constexpr (kind >= 0 && expert_pairs(kind)) expert_row_pair<kind>(m, local * (64 / plan.lanes[r]), plan.lanes[r], input, output);
        else if constexpr (kind >= 0) gemv_rows<kind, 1>(m, local * (32 / plan.lanes[r]), plan.lanes[r], input, 0, output, 0);
        else gemv_wave<1>(m, local, plan.lanes[r], input, 0, output, 0);
        return;
    }
}
// Each route's gate rows then up rows -> SiLU(gate) * up, F32.
__global__ void expert_product(const float* gate_up, size_t routes, size_t ffn, float* product) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index >= routes * ffn) return;
    const size_t base = (index / ffn) * 2 * ffn + index % ffn;
    const float gate = gate_up[base];
    product[index] = gate * hc_sigmoid(gate) * gate_up[base + ffn];
}
// One decode stage for `tokens` rows (input_stride apart): gate and up, or down, of every route.
// Gate/up writes per route its gate then up rows; down reads per route its product row.
void expert_gemv_launch(Device& d, const ExpertBanks& banks, const int32_t* selected, size_t tokens, uint32_t top_k,
                        bool down, const float* input, size_t input_stride, float* output, size_t output_stride) {
    RoutedGemv plan{banks, top_k, {}, {}, down};
    const size_t roles = down ? 1 : 2;
    int kind = -2;
    for (size_t r = 0; r < roles; ++r) {
        const Matrix& m = banks.role[down ? 2 : r];
        plan.lanes[r] = gemv_lanes(m, input, top_k * roles);
        kind = !plan.lanes[r] || (kind != -2 && kind != m.kind) ? -1 : m.kind;
    }
    // Waves per route: lane groups take two rows each where expert_pairs.
    size_t waves = 0;
    for (size_t r = 0; r < roles; ++r) {
        const Matrix& m = banks.role[down ? 2 : r];
        const size_t rows = plan.lanes[r] ? 32 / plan.lanes[r] * (kind >= 0 && expert_pairs(kind) ? 2 : 1) : 1;
        plan.waves[r] = uint32_t((m.rows + rows - 1) / rows);
        waves += top_k * size_t(plan.waves[r]);
    }
    const dim3 grid(unsigned((waves + 7) / 8), unsigned(tokens));
    #define SD_EXPERT(format) case format: \
        expert_gemv<format><<<grid, 256, 0, d.stream>>>(plan, selected, input, input_stride, output, output_stride); break;
    switch (kind) {
        SD_FAST_FORMATS(SD_EXPERT)
        default: expert_gemv<-1><<<grid, 256, 0, d.stream>>>(plan, selected, input, input_stride, output, output_stride);
    }
    #undef SD_EXPERT
    check(hipGetLastError());
}

#if defined(__HIPCC__)
// Grouped rows (FP16) to and from route order (F32).
__global__ void expert_routes_half(const float* routed, const int32_t* positions, size_t rows, size_t width,
                                   _Float16* grouped) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index < rows * width) grouped[index] = _Float16(routed[size_t(positions[index / width]) * width + index % width]);
}
__global__ void expert_half_routes(const _Float16* grouped, const int32_t* positions, size_t rows, size_t width,
                                   float* routed) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index < rows * width) routed[size_t(positions[index / width]) * width + index % width] = float(grouped[index]);
}
// Gate/up of every tile, fused (FP16 SiLU(gate) * up in grouped order) or not (F32 sums in route order).
template<bool fused>
void expert_gate_up_wmma(Device& d, int unit, MoeShape shape, const MoeGroup* groups, const MoeTile* tiles, unsigned count,
                         const int32_t* positions, size_t steps, const _Float16* input, _Float16* product, float* raw) {
    const dim3 grid(shape.ffn / (64 * moe_wave_n), count);
    #define SD_GATE_UP(format) case format: moe_gate_up_wmma<format, fused><<<grid, 256, 0, d.stream>>>( \
        groups, tiles, positions, steps, shape, input, product, nullptr, 0, false, raw); break;
    switch (unit) { SD_MOE_UNITS(SD_GATE_UP) }
    #undef SD_GATE_UP
    check(hipGetLastError());
}
#endif

// Temporary device buffers of one call, within the budget, released on return.
struct ExpertScratch {
    Device& d;
    std::vector<std::pair<void*, size_t>> buffers;
    template<class T> T* get(size_t count) {
        const size_t bytes = std::max<size_t>(count * sizeof(T), 16) + allocation_tail;
        if (bytes > d.budget - d.used) throw std::runtime_error("Straydog VRAM budget exceeded");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, bytes));
        buffers.emplace_back(pointer, bytes);
        d.used += bytes;
        return static_cast<T*>(pointer);
    }
    ~ExpertScratch() {
        cleanup(hipStreamSynchronize(d.stream));
        for (const auto& buffer : buffers) { cleanup(hipFree(buffer.first)); d.used -= buffer.second; }
    }
};
struct ExpertEvents {
    hipEvent_t marks[4]{};
    ExpertEvents() { for (auto& mark : marks) check(hipEventCreate(&mark)); }
    ~ExpertEvents() { for (auto& mark : marks) if (mark) cleanup(hipEventDestroy(mark)); }
    double span(int first) const {
        float ms = 0;
        check(hipEventElapsedTime(&ms, marks[first], marks[first + 1]));
        return ms;
    }
};

void expert_rows(Device& d, void* const* pointers, const int* kinds, size_t experts, size_t hidden, size_t ffn,
                 size_t steps, size_t top_k, const int32_t* selected, int path, const float* input,
                 const float* down_input, float* gate_up, float* product, float* output, double* timings) {
    const bool first_stage = gate_up || product || (output && !down_input);
    if (!pointers || !kinds || !selected || !experts || experts > 65536 || !hidden || !ffn || hidden > 65536 ||
        ffn > 65536 || !steps || steps > 65536 || !top_k || top_k > experts || (path != 0 && path != 1) ||
        (first_stage && !input) || (down_input && !output) || (!first_stage && !output) || hidden % 4 || ffn % 4)
        throw std::runtime_error("invalid Straydog expert rows request");
    const size_t routes = steps * top_k;
    for (size_t route = 0; route < routes; ++route)
        if (selected[route] < 0 || size_t(selected[route]) >= experts) throw std::runtime_error("invalid Straydog expert route");
    ExpertBanks banks{};
    for (size_t role = 0; role < 3; ++role) {
        const size_t rows = role == 2 ? hidden : ffn, cols = role == 2 ? ffn : hidden;
        const size_t row_bytes = matrix_row_bytes(kinds[role], cols);
        auto found = d.allocations.find(pointers[role]);
        if (found == d.allocations.end() || found->second != experts * rows * row_bytes)
            throw std::runtime_error("invalid Straydog expert bank size");
        banks.role[role] = {static_cast<const uint8_t*>(pointers[role]), row_bytes, rows, cols, 0, 0, 0, kinds[role]};
        banks.stride[role] = rows * row_bytes;
    }
    ExpertScratch scratch{d, {}};
    ExpertEvents events;
    float* x = input ? scratch.get<float>(steps * hidden) : nullptr;
    int32_t* routing = scratch.get<int32_t>(routes);
    float* outputs = output ? scratch.get<float>(routes * hidden) : nullptr;
    if (input) upload(d, x, input, steps * hidden * 4);
    upload(d, routing, selected, routes * 4);
    bool timed[2] = {};
    if (path == 0) {
        float* sums = first_stage ? scratch.get<float>(routes * 2 * ffn) : nullptr;
        float* products = scratch.get<float>(routes * ffn);
        if (first_stage) {
            check(hipEventRecord(events.marks[0], d.stream));
            expert_gemv_launch(d, banks, routing, steps, uint32_t(top_k), false, x, hidden, sums, top_k * 2 * ffn);
            check(hipEventRecord(events.marks[1], d.stream));
            timed[0] = true;
            expert_product<<<unsigned((routes * ffn + 255) / 256), 256, 0, d.stream>>>(sums, routes, ffn, products);
            check(hipGetLastError());
        }
        if (down_input) upload(d, products, down_input, routes * ffn * 4);
        if (output) {
            check(hipEventRecord(events.marks[2], d.stream));
            expert_gemv_launch(d, banks, routing, steps, uint32_t(top_k), true, products, top_k * ffn, outputs, top_k * hidden);
            check(hipEventRecord(events.marks[3], d.stream));
            timed[1] = true;
        }
        if (gate_up) download(d, gate_up, sums, routes * 2 * ffn * 4);
        if (product) download(d, product, products, routes * ffn * 4);
    } else {
#if defined(__HIPCC__)
        const MoeShape shape{uint32_t(hidden), uint32_t(ffn), uint32_t(top_k)};
        if (!d.rdna4 || !moe_shape_fits(shape)) throw std::runtime_error("the grouped expert GEMM needs RDNA4 and whole tiles");
        // Routes grouped by expert, in route order within each; tiles of moe_block_m rows.
        std::vector<size_t> counts(experts);
        for (size_t route = 0; route < routes; ++route) ++counts[size_t(selected[route])];
        std::vector<MoeGroup> groups(experts);
        std::vector<int32_t> positions(routes);
        std::vector<MoeTile> tiles;
        for (size_t expert = 0, first = 0; expert < experts; first += counts[expert], ++expert) {
            MoeGroup& group = groups[expert];
            for (size_t role = 0; role < 3; ++role) {
                group.weights[role] = banks.role[role];
                group.weights[role].data += expert * banks.stride[role];
            }
            group.first = first; group.count = 0; group.half = 1;
            for (size_t row = 0; row < counts[expert]; row += moe_block_m) tiles.push_back({uint32_t(expert), uint32_t(row)});
        }
        for (size_t route = 0; route < routes; ++route) {
            MoeGroup& group = groups[size_t(selected[route])];
            positions[group.first + group.count++] = int32_t(route);
        }
        auto* table = scratch.get<MoeGroup>(experts);
        auto* tile_list = scratch.get<MoeTile>(tiles.size());
        auto* grouped = scratch.get<int32_t>(routes);
        upload(d, table, groups.data(), experts * sizeof(MoeGroup));
        upload(d, tile_list, tiles.data(), tiles.size() * sizeof(MoeTile));
        upload(d, grouped, positions.data(), routes * 4);
        auto* half_input = input ? scratch.get<_Float16>(steps * hidden) : nullptr;
        auto* products = scratch.get<_Float16>(routes * ffn);
        const unsigned count = unsigned(tiles.size());
        const int gate_unit = moe_gate_up_unit(kinds[0], kinds[1], shape), down_unit = moe_down_unit(kinds[2], shape);
        if (input) {
            moe_input_half<<<unsigned((steps * hidden / 4 + 255) / 256), 256, 0, d.stream>>>(steps * hidden, x, half_input);
            check(hipGetLastError());
        }
        if (gate_up) {
            float* sums = scratch.get<float>(routes * 2 * ffn);
            expert_gate_up_wmma<false>(d, gate_unit, shape, table, tile_list, count, grouped, steps, half_input, nullptr, sums);
            download(d, gate_up, sums, routes * 2 * ffn * 4);
        }
        if (product || (output && !down_input)) {
            check(hipEventRecord(events.marks[0], d.stream));
            expert_gate_up_wmma<true>(d, gate_unit, shape, table, tile_list, count, grouped, steps, half_input, products, nullptr);
            check(hipEventRecord(events.marks[1], d.stream));
            timed[0] = true;
        }
        if (product) {
            float* routed = scratch.get<float>(routes * ffn);
            expert_half_routes<<<unsigned((routes * ffn + 255) / 256), 256, 0, d.stream>>>(products, grouped, routes, ffn, routed);
            check(hipGetLastError());
            download(d, product, routed, routes * ffn * 4);
        }
        if (down_input) {
            float* routed = scratch.get<float>(routes * ffn);
            upload(d, routed, down_input, routes * ffn * 4);
            expert_routes_half<<<unsigned((routes * ffn + 255) / 256), 256, 0, d.stream>>>(routed, grouped, routes, ffn, products);
            check(hipGetLastError());
        }
        if (output) {
            check(hipEventRecord(events.marks[2], d.stream));
            #define SD_DOWN(format) case format: \
                moe_down_wmma<format><<<dim3(shape.hidden / (128 * moe_wave_n), count), 256, 0, d.stream>>>( \
                    table, tile_list, grouped, shape, products, outputs, nullptr, 0, false); break;
            switch (down_unit) { SD_MOE_UNITS(SD_DOWN) }
            #undef SD_DOWN
            check(hipGetLastError());
            check(hipEventRecord(events.marks[3], d.stream));
            timed[1] = true;
        }
#else
        throw std::runtime_error("the grouped expert GEMM is RDNA4's");
#endif
    }
    if (output) download(d, output, outputs, routes * hidden * 4);
    check(hipStreamSynchronize(d.stream));
    if (timings)
        for (int stage = 0; stage < 2; ++stage) timings[stage] = timed[stage] ? events.span(2 * stage) : 0;
}
