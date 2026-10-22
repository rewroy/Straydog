// The formats' routed expert kernels at any layer shape: banks of `experts` gate and up
// (ffn x hidden) and down (hidden x ffn) matrices, each token's row routed to top_k of
// them. Decode runs gemv.hpp's units per route (F32); prefill on RDNA4 runs moe_gemm.hpp's
// grouped WMMA GEMM (FP16 operands, F32 sums). sd_expert_rows drives both, for tests and
// measurements; slot_experts runs decode over a cache of whole experts.

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
};
// (passed by value: well inside a launch's 4 KiB of arguments)
static_assert(sizeof(SlotExperts) <= 1024, "SlotExperts outgrew its launch arguments");
__device__ __forceinline__ uint64_t slot_base(const SlotExperts& plan, const uint64_t* bases, size_t index) {
    if (plan.uniform) return plan.uniform;
    if (plan.inline_count) return index < plan.inline_count ? plan.inline_bases[index] : 0;
    if (plan.table) {
        const int32_t id = plan.ids[index];
        return id >= 0 && uint32_t(id) < plan.table_count ? plan.table[id] : 0;
    }
    return bases[index];
}

__host__ __device__ constexpr bool slot_heavy(int kind) {
    return kind == 10 || kind == 11 || kind == 12 || kind == 13 || kind == 14 || kind == 23;
}
// This lane's shares of two rows' dots with x: `kind`'s units over `lanes` lanes, or (kind < 0) the
// reference element decoder of `format` over the wave. (PAIRED: gate and up, two rows a lane)
template<int kind, bool PAIRED = false>
__device__ __forceinline__ void slot_dots(const uint8_t* first, const uint8_t* second, int format, int second_format, size_t cols,
                                          unsigned lanes, unsigned sub, const float* x, float& a, float& b) {
    if constexpr (kind >= 0 && slot_heavy(kind)) {
        // the K-quant units take too many registers for two rows at once
        const size_t units = cols / GemvFormat<kind>::elements;
        for (int r = 0; r < (second ? 2 : 1); ++r)
            for (size_t unit = sub; unit < units; unit += lanes) {
                float dot[1];
                GemvFormat<kind>::template dots<1>(r ? second : first, unit, x, 0, dot);
                (r ? b : a) += dot[0];
            }
    } else if constexpr (kind >= 0) {
        const size_t units = cols / GemvFormat<kind>::elements;
        const auto step = [&](size_t unit) {
            float da[1], db[1];
            GemvFormat<kind>::template dots<1>(first, unit, x, 0, da);
            if (second) GemvFormat<kind>::template dots<1>(second, unit, x, 0, db);
            a += da[0];
            if (second) b += db[0];
        };
        if constexpr (PAIRED && kind != 16) {
            // Gate and up a unit at a time: unrolled, their registers kept fewer waves resident (DeepSeek-V4.1's
            // shapes: IQ2_S 64 -> 58 us, IQ1_S 49 -> 46, Q8_0 58 -> 49; IQ2_XXS 57 -> 59, so it stays unrolled).
            #pragma unroll 1
            for (size_t unit = sub; unit < units; unit += lanes) step(unit);
        } else {
            #pragma unroll 2
            for (size_t unit = sub; unit < units; unit += lanes) step(unit);
        }
    } else {
        #define SD_REFERENCE(format_) case format_: \
            for (size_t col = sub; col < cols; col += lanes) a += weight<format_>(first, col) * x[col]; \
            break;
        switch (format) { SD_FORMATS(SD_REFERENCE) }
        if (second) {
            float* sum = &b;
            first = second;
            switch (second_format) {
                #define SD_SECOND(format_) case format_: \
                    for (size_t col = sub; col < cols; col += lanes) *sum += weight<format_>(first, col) * x[col]; \
                    break;
                SD_FORMATS(SD_SECOND)
                #undef SD_SECOND
            }
        }
        #undef SD_REFERENCE
    }
}
__device__ __forceinline__ float slot_reduce(float value, unsigned lanes) {
    for (unsigned offset = lanes / 2; offset; offset /= 2) value += __shfl_xor(value, offset, 32);
    return value;
}
// blockIdx.y selects a token; the waves cover every route's rows. product[token][route][row] =
// silu(min(gate, c)) * clamp(up, -c, c) * weight, as op_swiglu_kernel forms it.
template<int kind>
__global__ void slot_gate_up(SlotExperts plan, const uint64_t* bases, const float* input, size_t input_stride,
                             const float* weights, float* product) {
    const size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const size_t route = wave / plan.waves[0];
    if (route >= plan.top_k) return;
    const size_t index = size_t(blockIdx.y) * plan.top_k + route;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(slot_base(plan, bases, index));
    if (!base) return;
    const Matrix& gate = plan.role[0];
    const unsigned lanes = plan.lanes[0], lane = threadIdx.x % 32, sub = lane % lanes;
    const Matrix& up = plan.role[1];
    const size_t row = (wave % plan.waves[0]) * (32 / lanes) + lane / lanes, last = std::min(row, gate.rows - 1);
    float g = 0.0f, u = 0.0f;
    slot_dots<kind, true>(base + size_t(gate.data) + last * gate.row_bytes, base + size_t(up.data) + last * up.row_bytes, gate.kind,
                    up.kind, gate.cols, lanes, sub, input + size_t(blockIdx.y) * input_stride, g, u);
    g = slot_reduce(g, lanes);
    u = slot_reduce(u, lanes);
    if (sub || row >= gate.rows) return;
    if (plan.clamp > 0.0f) { g = fminf(g, plan.clamp); u = fminf(fmaxf(u, -plan.clamp), plan.clamp); }
    product[index * plan.product_stride + plan.product_first + row] = g / (1.0f + expf(-g)) * u * (weights ? weights[index] : 1.0f);
}
// blockIdx.y selects a token; the waves cover the output rows, each summed over the token's routes
// in route order, then added to `out` (or stored, without `accumulate`).
// (each value also into `peer` at its index from out, when given: link_tail.hpp)
template<int kind>
__device__ __forceinline__ void slot_down_rows(const SlotExperts& plan, const uint64_t* bases, const float* product, float* out,
                                               size_t out_stride, float* peer) {
    const size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const Matrix& down = plan.role[2];
    const unsigned lanes = plan.lanes[1], lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = wave * (32 / lanes) + lane / lanes;
    if (wave >= plan.waves[1]) return;
    const size_t at = size_t(down.data) + std::min(row, down.rows - 1) * down.row_bytes;
    float sum = 0.0f, unused = 0.0f;
    for (size_t route = 0; route < plan.top_k; ++route) {
        const size_t index = size_t(blockIdx.y) * plan.top_k + route;
        const uint64_t base = slot_base(plan, bases, index);
        if (!base) continue;
        slot_dots<kind>(reinterpret_cast<const uint8_t*>(base) + at, nullptr, down.kind, down.kind, down.cols, lanes, sub,
                        product + index * plan.product_stride, sum, unused);
    }
    sum = slot_reduce(sum, lanes);
    if (sub || row >= down.rows) return;
    float* target = out + size_t(blockIdx.y) * out_stride + plan.out_first + row;
    const float value = plan.accumulate ? *target + sum : sum;
    *target = value;
    if (peer) link_send(peer + (target - out), value);
}
template<int kind>
__global__ void slot_down(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride) {
    slot_down_rows<kind>(plan, bases, product, out, out_stride, nullptr);
}
// slot_down whose output is this device's part of a linked sum (one row): the last block adds the
// parts (link_tail).
template<int kind>
__global__ void slot_down_linked(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride,
                                 LinkTail tail) {
    slot_down_rows<kind>(plan, bases, product, out, out_stride, link_tail_slot(tail));
    link_tail(tail);
}
// The stages (1 gate/up, 2 down) for `tokens` rows: product = SwiGLU of every route (tokens x top_k
// rows), then out. Lanes per row as the GEMV chooses them; a format without units, or gate and up
// of different formats, takes the reference decoder a row per wave.
// With a tail (one row, both stages), out is this device's part of its linked sum, the down stage's
// kernel adding the parts; false (nothing launched) when the down rows take the reference decoder.
bool slot_experts(Device& d, SlotExperts plan, int stages, const uint64_t* bases, size_t tokens, const float* input,
                  size_t input_stride, const float* weights, float* product, float* out, size_t out_stride,
                  const LinkTail* tail = nullptr) {
    const Matrix &gate = plan.role[0], &up = plan.role[1], &down = plan.role[2];
    auto fast = [](int kind) {
        switch (kind) {
            #define SD_FAST(format) case format: return true;
            SD_FAST_FORMATS(SD_FAST)
            #undef SD_FAST
        }
        return false;
    };
    // Offsets stand in for the slots' addresses, which are 256-byte aligned.
    const bool paired = gate.kind == up.kind && gate.row_bytes == up.row_bytes;
    const unsigned gate_lanes = std::min(gemv_lanes(gate, input, plan.top_k * 2), gemv_lanes(up, input, plan.top_k * 2));
    const int gate_kind = paired && gate_lanes && fast(gate.kind) ? gate.kind : -1;
    plan.lanes[0] = gate_kind < 0 ? 32 : gate_lanes;
    const unsigned down_lanes = gemv_lanes(down, product, 1);
    const int down_kind = down_lanes && fast(down.kind) ? down.kind : -1;
    plan.lanes[1] = down_kind < 0 ? 32 : down_lanes;
    plan.waves[0] = uint32_t((gate.rows + 32 / plan.lanes[0] - 1) / (32 / plan.lanes[0]));
    plan.waves[1] = uint32_t((down.rows + 32 / plan.lanes[1] - 1) / (32 / plan.lanes[1]));
    if (tail && (down_kind < 0 || stages != 3 || tokens != 1)) return false;
    const dim3 first(unsigned((size_t(plan.top_k) * plan.waves[0] + 7) / 8), unsigned(tokens));
    const dim3 second(unsigned((plan.waves[1] + 7) / 8), unsigned(tokens));
    #define SD_SLOT_GATE(format) case format: \
        slot_gate_up<format><<<first, 256, 0, d.stream>>>(plan, bases, input, input_stride, weights, product); break;
    if (stages & 1) {
        switch (gate_kind) {
            SD_FAST_FORMATS(SD_SLOT_GATE)
            default: slot_gate_up<-1><<<first, 256, 0, d.stream>>>(plan, bases, input, input_stride, weights, product);
        }
        check(hipGetLastError());
    }
    #undef SD_SLOT_GATE
    if (!(stages & 2)) return true;
    if (tail) {
        #define SD_SLOT_LINKED(format) case format: \
            slot_down_linked<format><<<second, 256, 0, d.stream>>>(plan, bases, product, out, out_stride, *tail); break;
        switch (down_kind) {
            SD_FAST_FORMATS(SD_SLOT_LINKED)
        }
        #undef SD_SLOT_LINKED
        check(hipGetLastError());
        return true;
    }
    #define SD_SLOT_DOWN(format) case format: \
        slot_down<format><<<second, 256, 0, d.stream>>>(plan, bases, product, out, out_stride); break;
    switch (down_kind) {
        SD_FAST_FORMATS(SD_SLOT_DOWN)
        default: slot_down<-1><<<second, 256, 0, d.stream>>>(plan, bases, product, out, out_stride);
    }
    #undef SD_SLOT_DOWN
    check(hipGetLastError());
    return true;
}
