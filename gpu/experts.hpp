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
    // A verify pass's gate/up: its distinct experts, each its routes as four bytes (255 past the last).
    uint32_t groups;
    uint32_t group_routes[slot_inline_routes];
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
// One IQ2_XXS unit's scale and codebook index. The bytes are the same ones GemvFormat<16> reads.
struct Iq2xxsTerms {
    uint32_t grid, aux;
    float scale;
};
__device__ __forceinline__ Iq2xxsTerms iq2xxs_terms(const uint8_t* row, size_t unit) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(row + (unit / 8) * 66);
    const unsigned ib = unsigned(unit % 8);
    const uint32_t grid = iq_pair(h, 1 + 4 * ib), aux = iq_pair(h, 3 + 4 * ib);
    const float scale = gemv_half(h[0]) * (0.5f + float(aux >> 28)) * 0.25f;
    return {grid, aux, scale};
}
// Paired gate/up only: the unit's eight activation vectors and both rows' grid rows are loaded
// before either row's FMAs. The flips, the four-group sum and the scale stay in GemvFormat<16>'s
// order. Down still calls GemvFormat::dots.
__device__ __forceinline__ void slot_iq2xxs_grids(const uint8_t* first, const uint8_t* second, size_t unit, const float* x,
                                                  float& a, float& b) {
    const Iq2xxsTerms t0 = iq2xxs_terms(first, unit);
    Iq2xxsTerms t1{};
    if (second) t1 = iq2xxs_terms(second, unit);
    const float* base = x + unit * 32;
    float4 act[8];
    uint2 g0[4], g1[4];
    #pragma unroll
    for (int i = 0; i < 8; ++i) act[i] = gemv_load4(base + 4 * i);
    #pragma unroll
    for (int l = 0; l < 4; ++l) g0[l] = reinterpret_cast<const uint2*>(iq_grid2xxs)[(t0.grid >> (8 * l)) & 255];
    if (second) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) g1[l] = reinterpret_cast<const uint2*>(iq_grid2xxs)[(t1.grid >> (8 * l)) & 255];
    }
#if defined(__HIPCC__)
    // Keep the loads above the FMAs. This is not a wait.
    asm volatile("" ::: "memory");
#endif
    auto consume = [](const Iq2xxsTerms& terms, const uint2 (&grid)[4], const float4 (&act)[8], float& acc) {
        float sum = 0.0f;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned signs = iq_signs((terms.aux >> (7 * l)) & 127);
            sum = gemv_dot4(grid[l].x, iq_flip(act[2 * l], signs), sum);
            sum = gemv_dot4(grid[l].y, iq_flip(act[2 * l + 1], signs >> 4), sum);
        }
        const float dot = terms.scale * sum;
        acc += dot;
    };
    consume(t0, g0, act, a);
    if (second) consume(t1, g1, act, b);
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
            if constexpr (PAIRED && kind == 16) {
                slot_iq2xxs_grids(first, second, unit, x, a, b);
            } else {
                float da[1], db[1];
                GemvFormat<kind>::template dots<1>(first, unit, x, 0, da);
                if (second) GemvFormat<kind>::template dots<1>(second, unit, x, 0, db);
                a += da[0];
                if (second) b += db[0];
            }
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
// (each value also into `peer` at its index from product, when given: link_tail.hpp)
template<int kind>
__device__ __forceinline__ void slot_gate_up_rows(const SlotExperts& plan, const uint64_t* bases, const float* input, size_t input_stride,
                                                  const float* weights, float* product, float* peer) {
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
    const size_t at = index * plan.product_stride + plan.product_first + row;
    const float value = g / (1.0f + expf(-g)) * u * (weights ? weights[index] : 1.0f);
    product[at] = value;
    if (peer) link_send(peer + at, value);
}
template<int kind>
__global__ void slot_gate_up(SlotExperts plan, const uint64_t* bases, const float* input, size_t input_stride,
                             const float* weights, float* product) { SD_KERNEL_ACQUIRE;
    slot_gate_up_rows<kind>(plan, bases, input, input_stride, weights, product, nullptr);
}
// slot_gate_up whose product is this device's part (its columns) of a part link's sum: each value
// also goes to the other device's inbox, `send_at` floats in plus its index from product, and with
// `raise`, the last block raises the other's flag (link_tail; without, a later call sends the sum's
// other values and raises it).
template<int kind>
__global__ void slot_gate_up_sent(SlotExperts plan, const uint64_t* bases, const float* input, size_t input_stride,
                                  const float* weights, float* product, LinkTail tail, size_t send_at, int raise) {
    slot_gate_up_rows<kind>(plan, bases, input, input_stride, weights, product, link_tail_slot(tail) + send_at);
    if (raise) link_tail(tail);
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
__global__ void slot_down(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride) { SD_KERNEL_ACQUIRE;
    slot_down_rows<kind>(plan, bases, product, out, out_stride, nullptr);
}
// slot_down of IQ1_M for several tokens, two rows to a lane group (`waves` of them): a lane's unit of both rows is dotted with the activations it
// loads once. The F32 kernel's activation loads bind it (a unit's 32 values, 128 bytes from the next unit's, are 8 cache lines to a load) and
// every lane group of every wave loads them again; here a row's share of them is half (2 to 4 tokens: 52 / 74 / 92 us to 49 / 68 / 87). Each row
// is summed as slot_down_rows sums it.
template<int kind>
__global__ void slot_down_pairs(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride, uint32_t waves) {
    static_assert(kind == 29, "slot_down_pairs: IQ1_M");
    const uint2* grid = reinterpret_cast<const uint2*>(iq_grid1s);
    const size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    if (wave >= waves) return;
    const Matrix& down = plan.role[2];
    const unsigned lanes = plan.lanes[1], lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = (wave * (32 / lanes) + lane / lanes) * 2, last = down.rows - 1;
    const size_t at = size_t(down.data) + std::min(row, last) * down.row_bytes, at_b = size_t(down.data) + std::min(row + 1, last) * down.row_bytes;
    const size_t units = down.cols / GemvFormat<kind>::elements;
    float sum = 0.0f, sum_b = 0.0f;
    for (size_t route = 0; route < plan.top_k; ++route) {
        const size_t index = size_t(blockIdx.y) * plan.top_k + route;
        const uint64_t base = slot_base(plan, bases, index);
        if (!base) continue;
        const float* x = product + index * plan.product_stride;
        #pragma unroll 2
        for (size_t unit = sub; unit < units; unit += lanes) {
            float a, b;
            gemv_iq1m_dots2(reinterpret_cast<const uint8_t*>(base) + at, reinterpret_cast<const uint8_t*>(base) + at_b, unit, x, grid, a, b);
            sum += a;
            sum_b += b;
        }
    }
    sum = slot_reduce(sum, lanes);
    sum_b = slot_reduce(sum_b, lanes);
    if (sub) return;
    float* target = out + size_t(blockIdx.y) * out_stride + plan.out_first + row;
    if (row < down.rows) *target = plan.accumulate ? *target + sum : sum;
    if (row + 1 < down.rows) target[1] = plan.accumulate ? target[1] + sum_b : sum_b;
}
// slot_down whose output is this device's part of a linked sum (one row): the last block adds the
// parts (link_tail).
template<int kind>
__global__ void slot_down_linked(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride,
                                 LinkTail tail) {
    slot_down_rows<kind>(plan, bases, product, out, out_stride, link_tail_slot(tail));
    link_tail(tail);
}
#if defined(__HIPCC__)
// ---- Q8 slot gate/up: a decode row's IQ2 experts on RDNA4 ----
// The F32 kernel spends five 16-bit loads a unit and three VALU instructions a value. This one loads
// two units' packed bytes a lane at once, takes each eight values' sign masks from a table in LDS,
// and forms each four values' products with v_dot4 against the row's activations quantized per 32
// values to INT8 (gemv_q8_stage): a unit's sum is exact in INT32, then scaled in F32. The grid's
// lookups stay in (cached) global memory: in LDS beside the masks, the lookups held the kernel.
// Eight grid bytes `g` against eight INT8 activations, value j negative where mask byte j is set:
// sum s g x = sum g x - 2 sum (g & m) x (a grid byte is below 128: g and g & m are signed bytes too).
__device__ __forceinline__ void slot_q8_dot8(uint2 g, uint2 m, uint32_t x0, uint32_t x1, int& pos, int& neg) {
    pos = __builtin_amdgcn_sudot4(true, int(g.x), true, int(x0), pos, false);
    neg = __builtin_amdgcn_sudot4(true, int(g.x & m.x), true, int(x0), neg, false);
    pos = __builtin_amdgcn_sudot4(true, int(g.y), true, int(x1), pos, false);
    neg = __builtin_amdgcn_sudot4(true, int(g.y & m.y), true, int(x1), neg, false);
}
// Two units (64 values) of a row: load() their packed bytes, dot() unit h with its activations
// (eight words of INT8 values and their scale), as GemvFormat<kind> reads and scales them.
template<int kind> struct SlotQ8;
template<> struct SlotQ8<16> {   // IQ2_XXS: 66-byte blocks; a unit's four grid bytes and its sign/scale word
    static constexpr int sign_entries = 128;
    moe_word4 q;
    uint32_t d;
    __device__ void load(const uint8_t* row, size_t pair) {
        const uint8_t* b = row + (pair / 4) * 66;
        q = moe_u128(b + 2 + 16 * (pair % 4));
        d = moe_u16(b);
    }
    __device__ float dot(int h, const uint2* masks, const uint32_t* x, float scale) const {
        const uint2* grid = reinterpret_cast<const uint2*>(iq_grid2xxs);
        const uint32_t g = q[2 * h], aux = q[2 * h + 1];
        int pos = 0, neg = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) slot_q8_dot8(grid[(g >> (8 * l)) & 255], masks[(aux >> (7 * l)) & 127], x[2 * l], x[2 * l + 1], pos, neg);
        return gemv_half(d) * (0.5f + float(aux >> 28)) * 0.25f * scale * float(pos - 2 * neg);
    }
};
template<> struct SlotQ8<17> {   // IQ2_XS: 74-byte blocks; a unit's four 9-bit grid and 7-bit sign codes, two scales
    static constexpr int sign_entries = 128;
    moe_word4 q;
    uint32_t d, scales;
    __device__ void load(const uint8_t* row, size_t pair) {
        const uint8_t* b = row + (pair / 4) * 74;
        q = moe_u128(b + 2 + 16 * (pair % 4));
        scales = moe_u16(b + 66 + 2 * (pair % 4));
        d = moe_u16(b);
    }
    __device__ float dot(int h, const uint2* masks, const uint32_t* x, float scale) const {
        const uint2* grid = reinterpret_cast<const uint2*>(iq_grid2xs);
        int pos[2] = {0, 0}, neg[2] = {0, 0};
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned code = (q[2 * h + l / 2] >> (16 * (l & 1))) & 0xffffu;
            slot_q8_dot8(grid[code & 511], masks[code >> 9], x[2 * l], x[2 * l + 1], pos[l / 2], neg[l / 2]);
        }
        const unsigned s = (scales >> (8 * h)) & 255;
        const float d4 = gemv_half(d) * 0.25f * scale;
        return d4 * (0.5f + float(s & 15)) * float(pos[0] - 2 * neg[0]) + d4 * (0.5f + float(s >> 4)) * float(pos[1] - 2 * neg[1]);
    }
};
template<> struct SlotQ8<22> {   // IQ2_S: 82-byte blocks; a unit's grid bytes, sign bytes, high bits and two scales
    static constexpr int sign_entries = 256;
    uint2 low, signs;
    uint32_t high, scales, d;
    __device__ void load(const uint8_t* row, size_t pair) {
        const uint8_t* b = row + (pair / 4) * 82;
        const unsigned j = unsigned(pair % 4);
        low = moe_u64(b + 2 + 8 * j);
        signs = moe_u64(b + 34 + 8 * j);
        high = moe_u16(b + 66 + 2 * j);
        scales = moe_u16(b + 74 + 2 * j);
        d = moe_u16(b);
    }
    __device__ float dot(int h, const uint2* masks, const uint32_t* x, float scale) const {
        const uint2* grid = reinterpret_cast<const uint2*>(iq_grid2s);
        const uint32_t g = h ? low.y : low.x, sg = h ? signs.y : signs.x, hi = (high >> (8 * h)) & 255;
        int pos[2] = {0, 0}, neg[2] = {0, 0};
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned index = ((g >> (8 * l)) & 255) | ((hi << (8 - 2 * l)) & 0x300);
            slot_q8_dot8(grid[index], masks[(sg >> (8 * l)) & 255], x[2 * l], x[2 * l + 1], pos[l / 2], neg[l / 2]);
        }
        const unsigned s = (scales >> (8 * h)) & 255;
        const float d4 = gemv_half(d) * 0.25f * scale;
        return d4 * (0.5f + float(s & 15)) * float(pos[0] - 2 * neg[0]) + d4 * (0.5f + float(s >> 4)) * float(pos[1] - 2 * neg[1]);
    }
};
template<> struct SlotQ8<29> {   // IQ1_M: 56-byte blocks; a unit's four grid bytes and four high nibbles (3 grid bits, the delta's), two 3-bit scales
    static constexpr int sign_entries = 0;
    uint2 low, words;
    uint32_t high, scales;
    __device__ void load(const uint8_t* row, size_t pair) {
        const uint8_t* b = row + (pair / 4) * 56;
        const unsigned j = unsigned(pair % 4);
        low = moe_u64(b + 8 * j);
        high = moe_u32(b + 32 + 4 * j);
        words = moe_u64(b + 48);
        scales = (j & 1 ? words.x >> 16 : words.x) & 0xffffu;
        if (j >= 2) scales = (j & 1 ? words.y >> 16 : words.y) & 0xffffu;
    }
    // A grid row stores value + 1 (0 or 2), so a group adds (delta - 1) * sum(x): -7/8 or -9/8. Eight times over, all in INT32.
    __device__ float dot(int h, const uint2*, const uint32_t* x, float scale) const {
        const uint2* grid = reinterpret_cast<const uint2*>(iq_grid1s);
        const uint32_t g = h ? low.y : low.x, hi = (high >> (16 * h)) & 0xffffu;
        int part[2] = {0, 0};
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned nibble = (hi >> (4 * l)) & 15;
            const uint2 row = grid[((g >> (8 * l)) & 255) | ((nibble & 7) << 8)];
            int dot = __builtin_amdgcn_sudot4(true, int(row.x), true, int(x[2 * l]), 0, false);
            dot = __builtin_amdgcn_sudot4(true, int(row.y), true, int(x[2 * l + 1]), dot, false);
            int sum = __builtin_amdgcn_sudot4(true, 0x01010101, true, int(x[2 * l]), 0, false);
            sum = __builtin_amdgcn_sudot4(true, 0x01010101, true, int(x[2 * l + 1]), sum, false);
            part[l / 2] += 8 * dot - (nibble & 8 ? 9 : 7) * sum;
        }
        const unsigned bits = (words.x & 0xffffu) >> 12 | ((words.x >> 24) & 0xf0) | (((words.y & 0xffffu) >> 4) & 0xf00) | ((words.y >> 16) & 0xf000);
        const unsigned s = scales >> (6 * h);
        const float d8 = gemv_half(bits) * 0.125f * scale;
        return d8 * float(2 * (s & 7) + 1) * float(part[0]) + d8 * float(2 * ((s >> 3) & 7) + 1) * float(part[1]);
    }
};
__host__ __device__ constexpr bool slot_q8_kind(int kind) { return kind == 16 || kind == 17 || kind == 22 || kind == 29; }
// Q5_K (DeepSeek-V4.1's shared expert) runs gemv.hpp's GemvQ8<13> units: no sign masks, and the activations staged with the group sums its
// mins take.
__host__ __device__ constexpr bool slot_q5k_kind(int kind) { return kind == 13; }
// LDS bytes of a kind's launch: its sign masks, then the activations of `vectors` rows of `cols` (INT8, a scale every 32; Q5_K's a sum too).
__host__ __device__ constexpr size_t slot_q8_lds(int kind, size_t cols, size_t vectors = 1) {
    return (kind == 13 || kind == 29 ? 0 : kind == 22 ? 256 : 128) * 8 + vectors * (cols + cols / 32 * (kind == 13 ? 8 : 4));
}
// Byte j of mask x (y) is 0xff when sign bit j (4 + j) of a sign index's eight is set.
__device__ __forceinline__ uint32_t slot_spread4(uint32_t bits) { return ((bits * 0x00204081u) & 0x01010101u) * 0xffu; }
template<int kind>
__device__ __forceinline__ void slot_q8_stage_masks(uint2* masks) {
    for (int i = threadIdx.x; i < SlotQ8<kind>::sign_entries; i += blockDim.x) {
        const unsigned bits = kind == 22 ? unsigned(i) : iq_signs(unsigned(i));
        masks[i] = uint2{slot_spread4(bits & 15), slot_spread4(bits >> 4)};
    }
}
// slot_gate_up_rows with the lane's sum of the row pair's dots (sum(gate_row, up_row, g, u) adds this lane's units' to g and u) on the staged
// activations: the same rows, lanes and epilogue.
template<class Sum>
__device__ __forceinline__ void slot_gate_up_q8_rows(const SlotExperts& plan, const uint64_t* bases, const float* weights, float* product, float* peer,
                                                     Sum sum) {
    const size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const size_t route = wave / plan.waves[0];
    if (route >= plan.top_k) return;
    const size_t index = size_t(blockIdx.y) * plan.top_k + route;
    const uint8_t* base = reinterpret_cast<const uint8_t*>(slot_base(plan, bases, index));
    if (!base) return;
    const Matrix &gate = plan.role[0], &up = plan.role[1];
    const unsigned lanes = plan.lanes[0], lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = (wave % plan.waves[0]) * (32 / lanes) + lane / lanes, last = std::min(row, gate.rows - 1);
    float g = 0.0f, u = 0.0f;
    sum(base + size_t(gate.data) + last * gate.row_bytes, base + size_t(up.data) + last * up.row_bytes, sub, lanes, gate.cols, g, u);
    g = slot_reduce(g, lanes);
    u = slot_reduce(u, lanes);
    if (sub || row >= gate.rows) return;
    if (plan.clamp > 0.0f) { g = fminf(g, plan.clamp); u = fminf(fmaxf(u, -plan.clamp), plan.clamp); }
    const size_t at = index * plan.product_stride + plan.product_first + row;
    const float value = g / (1.0f + expf(-g)) * u * (weights ? weights[index] : 1.0f);
    product[at] = value;
    if (peer) link_send(peer + at, value);
}
// slot_gate_up (SENT: slot_gate_up_sent) with the masks and the token's activations staged first.
template<int kind, bool SENT>
__global__ void __launch_bounds__(512) slot_gate_up_q8(SlotExperts plan, const uint64_t* bases, const float* input, size_t input_stride,
                                                       const float* weights, float* product, LinkTail tail, size_t send_at, int raise) {
    extern __shared__ __align__(16) uint32_t slot_lds[];
    uint2* masks = reinterpret_cast<uint2*>(slot_lds);
    uint32_t* xq = reinterpret_cast<uint32_t*>(masks + SlotQ8<kind>::sign_entries);
    const size_t cols = plan.role[0].cols;
    float* xd = reinterpret_cast<float*>(xq + cols / 4);
    slot_q8_stage_masks<kind>(masks);
    gemv_q8_stage(xq, xd, nullptr, input + size_t(blockIdx.y) * input_stride, cols / 32);
    __syncthreads();
    slot_gate_up_q8_rows(plan, bases, weights, product, SENT ? link_tail_slot(tail) + send_at : nullptr,
                         [&](const uint8_t* gate_row, const uint8_t* up_row, unsigned sub, unsigned lanes, size_t width, float& g, float& u) {
        const size_t pairs = width / 64;
        #pragma unroll 2
        for (size_t pair = sub; pair < pairs; pair += lanes) {
            SlotQ8<kind> a, b;
            a.load(gate_row, pair);
            b.load(up_row, pair);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const size_t unit = 2 * pair + h;
                g += a.dot(h, masks, xq + unit * 8, xd[unit]);
                u += b.dot(h, masks, xq + unit * 8, xd[unit]);
            }
        }
    });
    if (SENT && raise) link_tail(tail);
}
// slot_gate_up_q8 of a pass of a few rows, an expert's rows read once: rows of a pass share a quarter of their routes, and these kernels are
// bound by those reads (DeepSeek-V4.1 3-row pass 26.89 -> 26.30 ms with IQ1_M on it; IQ1_M alone on the Q8 kernel without this, 0.3%). Every row's activations are staged; a wave takes a distinct expert's matrix rows, and each route of it its own dots and
// epilogue, in the order a route's own wave would: the same values.
template<int kind>
__global__ void __launch_bounds__(512) slot_gate_up_q8_groups(SlotExperts plan, const float* input, size_t input_stride, const float* weights,
                                                              float* product, uint32_t tokens) {
    extern __shared__ __align__(16) uint32_t slot_lds[];
    uint2* masks = reinterpret_cast<uint2*>(slot_lds);
    uint32_t* staged = reinterpret_cast<uint32_t*>(masks + SlotQ8<kind>::sign_entries);
    const size_t cols = plan.role[0].cols, words = cols / 4, each = words + cols / 32;
    slot_q8_stage_masks<kind>(masks);
    for (uint32_t token = 0; token < tokens; ++token)
        gemv_q8_stage(staged + token * each, reinterpret_cast<float*>(staged + token * each + words), nullptr, input + size_t(token) * input_stride, cols / 32);
    __syncthreads();
    const size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const size_t group = wave / plan.waves[0];
    if (group >= plan.groups) return;
    const uint32_t routes = plan.group_routes[group];
    const uint8_t* base = reinterpret_cast<const uint8_t*>(plan.inline_bases[routes & 255]);
    const Matrix &gate = plan.role[0], &up = plan.role[1];
    const unsigned lanes = plan.lanes[0], lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = (wave % plan.waves[0]) * (32 / lanes) + lane / lanes, last = std::min(row, gate.rows - 1);
    const uint8_t* gate_row = base + size_t(gate.data) + last * gate.row_bytes;
    const uint8_t* up_row = base + size_t(up.data) + last * up.row_bytes;
    float g[4] = {0.0f, 0.0f, 0.0f, 0.0f}, u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const size_t pairs = cols / 64;
    #pragma unroll 2
    for (size_t pair = sub; pair < pairs; pair += lanes) {
        SlotQ8<kind> a, b;
        a.load(gate_row, pair);
        b.load(up_row, pair);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const size_t unit = 2 * pair + h;
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const unsigned route = (routes >> (8 * v)) & 255;
                if (route == 255) break;
                const uint32_t* xq = staged + (route / plan.top_k) * each;
                const float scale = reinterpret_cast<const float*>(xq + words)[unit];
                g[v] += a.dot(h, masks, xq + unit * 8, scale);
                u[v] += b.dot(h, masks, xq + unit * 8, scale);
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        const unsigned route = (routes >> (8 * v)) & 255;
        if (route == 255) break;
        float gv = slot_reduce(g[v], lanes), uv = slot_reduce(u[v], lanes);
        if (sub || row >= gate.rows) continue;
        if (plan.clamp > 0.0f) { gv = fminf(gv, plan.clamp); uv = fminf(fmaxf(uv, -plan.clamp), plan.clamp); }
        product[route * plan.product_stride + plan.product_first + row] = gv / (1.0f + expf(-gv)) * uv * (weights ? weights[route] : 1.0f);
    }
}
// slot_gate_up_q8 of a Q5_K pair of matrices.
template<bool SENT>
__global__ void __launch_bounds__(512) slot_gate_up_q5k(SlotExperts plan, const uint64_t* bases, const float* input, size_t input_stride,
                                                        const float* weights, float* product, LinkTail tail, size_t send_at, int raise) {
    extern __shared__ __align__(16) uint32_t slot_lds[];
    const size_t cols = plan.role[0].cols;
    float* xd = reinterpret_cast<float*>(slot_lds + cols / 4);
    float* xs = xd + cols / 32;
    gemv_q8_stage(slot_lds, xd, xs, input + size_t(blockIdx.y) * input_stride, cols / 32);
    __syncthreads();
    const GemvQ8Input in{slot_lds, xd, xs, 0};
    slot_gate_up_q8_rows(plan, bases, weights, product, SENT ? link_tail_slot(tail) + send_at : nullptr,
                         [&](const uint8_t* gate_row, const uint8_t* up_row, unsigned sub, unsigned lanes, size_t width, float& g, float& u) {
        for (size_t unit = sub; unit < width / 64; unit += lanes) {
            float r[1];
            GemvQ8<13>::dots<1>(gate_row, unit, in, r);
            g += r[0];
            GemvQ8<13>::dots<1>(up_row, unit, in, r);
            u += r[0];
        }
    });
    if (SENT && raise) link_tail(tail);
}

// ---- Q8 slot down ----
// slot_down_rows on INT8 activations, as the gate/up above: the block quantizes each live route's product row (per 32 values) into LDS
// first, and a lane sums its units of the row route by route.
// The block's routes with a slot (a bit each), and their product rows staged: route r's values at xq + r * cols / 4, scales at xd +
// r * cols / 32 (and sums at xs, given).
__device__ __forceinline__ void slot_down_stage(const SlotExperts& plan, const uint64_t* bases, const float* product, uint32_t* xq, float* xd,
                                                float* xs) {
    const size_t cols = plan.role[2].cols, groups = cols / 32, first = size_t(blockIdx.y) * plan.top_k;
    uint32_t live = 0;
    for (uint32_t route = 0; route < plan.top_k; ++route) live |= (slot_base(plan, bases, first + route) ? 1u : 0u) << route;
    for (size_t g = threadIdx.x; g < size_t(plan.top_k) * groups; g += blockDim.x) {
        const size_t route = g / groups, local = g - route * groups;
        if (live >> route & 1)
            gemv_q8_group(product + (first + route) * plan.product_stride + local * 32, xq + route * (cols / 4) + local * 8,
                          xd + route * groups + local, xs ? xs + route * groups + local : nullptr);
    }
}
// slot_down_rows with the lane's sum of a route's units (sum(row, route, sub, lanes, cols, total) adds them to total).
template<class Sum>
__device__ __forceinline__ void slot_down_q8_rows(const SlotExperts& plan, const uint64_t* bases, float* out, size_t out_stride, float* peer, Sum sum) {
    const size_t wave = SD_WAVE_UNIFORM(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    if (wave >= plan.waves[1]) return;
    const Matrix& down = plan.role[2];
    const unsigned lanes = plan.lanes[1], lane = threadIdx.x % 32, sub = lane % lanes;
    const size_t row = wave * (32 / lanes) + lane / lanes;
    const size_t at = size_t(down.data) + std::min(row, down.rows - 1) * down.row_bytes;
    float total = 0.0f;
    for (size_t route = 0; route < plan.top_k; ++route) {
        const uint64_t base = slot_base(plan, bases, size_t(blockIdx.y) * plan.top_k + route);
        if (base) sum(reinterpret_cast<const uint8_t*>(base) + at, route, sub, lanes, down.cols, total);
    }
    total = slot_reduce(total, lanes);
    if (sub || row >= down.rows) return;
    float* target = out + size_t(blockIdx.y) * out_stride + plan.out_first + row;
    const float value = plan.accumulate ? *target + total : total;
    *target = value;
    if (peer) link_send(peer + (target - out), value);
}
// LINKED: slot_down_linked's (the part link's sum), else slot_down's.
template<int kind, bool LINKED>
__global__ void __launch_bounds__(512) slot_down_q8(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride,
                                                    LinkTail tail) {
    extern __shared__ __align__(16) uint32_t slot_lds[];
    uint2* masks = reinterpret_cast<uint2*>(slot_lds);
    uint32_t* xq = reinterpret_cast<uint32_t*>(masks + SlotQ8<kind>::sign_entries);
    const size_t cols = plan.role[2].cols;
    float* xd = reinterpret_cast<float*>(xq + plan.top_k * cols / 4);
    slot_q8_stage_masks<kind>(masks);
    slot_down_stage(plan, bases, product, xq, xd, nullptr);
    __syncthreads();
    slot_down_q8_rows(plan, bases, out, out_stride, LINKED ? link_tail_slot(tail) : nullptr,
                      [&](const uint8_t* row, size_t route, unsigned sub, unsigned lanes, size_t width, float& total) {
        const uint32_t* x = xq + route * (width / 4);
        const float* scales = xd + route * (width / 32);
        const size_t pairs = width / 64;
        #pragma unroll 2
        for (size_t pair = sub; pair < pairs; pair += lanes) {
            SlotQ8<kind> slot;
            slot.load(row, pair);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const size_t unit = 2 * pair + h;
                total += slot.dot(h, masks, x + unit * 8, scales[unit]);
            }
        }
    });
    if (LINKED) link_tail(tail);
}
// slot_down_q8 of a Q5_K matrix.
template<bool LINKED>
__global__ void __launch_bounds__(512) slot_down_q5k(SlotExperts plan, const uint64_t* bases, const float* product, float* out, size_t out_stride,
                                                     LinkTail tail) {
    extern __shared__ __align__(16) uint32_t slot_lds[];
    const size_t cols = plan.role[2].cols;
    float* xd = reinterpret_cast<float*>(slot_lds + plan.top_k * cols / 4);
    float* xs = xd + plan.top_k * cols / 32;
    slot_down_stage(plan, bases, product, slot_lds, xd, xs);
    __syncthreads();
    slot_down_q8_rows(plan, bases, out, out_stride, LINKED ? link_tail_slot(tail) : nullptr,
                      [&](const uint8_t* row, size_t route, unsigned sub, unsigned lanes, size_t width, float& total) {
        const GemvQ8Input in{slot_lds + route * (width / 4), xd + route * (width / 32), xs + route * (width / 32), 0};
        for (size_t unit = sub; unit < width / 64; unit += lanes) {
            float r[1];
            GemvQ8<13>::dots<1>(row, unit, in, r);
            total += r[0];
        }
    });
    if (LINKED) link_tail(tail);
}
#endif
// The stages (1 gate/up, 2 down) for `tokens` rows: product = SwiGLU of every route (tokens x top_k
// rows), then out. Lanes per row as the GEMV chooses them; a format without units, or gate and up
// of different formats, takes the reference decoder a row per wave.
// With a tail (one row, both stages), out is this device's part of its linked sum, the down stage's
// kernel adding the parts (a tail that only sends: any rows, with the down stage; of the gate/up
// stage alone, its product goes, at `send_at` floats into the inbox, the flag raised with `raise`);
// false (nothing launched) when the down rows take the reference decoder.
bool slot_experts(Device& d, SlotExperts plan, int stages, const uint64_t* bases, size_t tokens, const float* input,
                  size_t input_stride, const float* weights, float* product, float* out, size_t out_stride,
                  const LinkTail* tail = nullptr, size_t send_at = 0, int raise = 1) {
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
    unsigned gate_lanes = std::min(gemv_lanes(gate, input, plan.top_k * 2), gemv_lanes(up, input, plan.top_k * 2));
    const int gate_kind = paired && gate_lanes && fast(gate.kind) ? gate.kind : -1;
    unsigned down_lanes = gemv_lanes(down, product, 1);
    const int down_kind = down_lanes && fast(down.kind) ? down.kind : -1;
    // The Q8 kernels (DeepSeek-V4.1's decode: d.q8_decode): gate/up of the IQ2 kinds and Q5_K, down of IQ2_XXS, IQ2_XS and Q5_K
    // (IQ2_S's down lost nothing: 33.0 us on the F32 kernel, 33.7 on this one).
    bool q8_gate = false, q8_down = false;
    size_t gate_lds = 0, down_lds = 0;
#if defined(__HIPCC__)
    gate_lds = slot_q8_lds(gate_kind, gate.cols);
    q8_gate = d.q8_decode && gate_kind >= 0 && (slot_q8_kind(gate_kind) || slot_q5k_kind(gate_kind)) && gate.cols % 256 == 0 &&
              reinterpret_cast<uintptr_t>(input) % 16 == 0 && input_stride % 4 == 0 && gate_lds <= 65536;
    down_lds = slot_q8_lds(down_kind, down.cols, plan.top_k);
    q8_down = d.q8_decode && down_kind >= 0 &&
              (down_kind == 16 || down_kind == 17 || down_kind == 29 || slot_q5k_kind(down_kind)) &&
              down.cols % 256 == 0 && plan.top_k <= 16 && reinterpret_cast<uintptr_t>(product) % 16 == 0 && plan.product_stride % 4 == 0 &&
              down_lds <= 65536;
    // An IQ2 row of 4096 to 8191 values (DeepSeek-V4.1's gate and up, 5120) takes at most four lanes on the Q8 kernels: over eight a wave's loads
    // fan out to four rows each and a lane walks ten pairs (IQ2_XXS 44.5 us a layer's gate/up, 40.5 on four lanes; IQ2_XS 53.1 -> 46.5; IQ2_S
    // 53.1 -> 47.8); two lanes or sixteen lose. Decided by the matrix and the device alone (a row's sum is the same whatever shares its launch).
    // (IQ1_M's blocks are shorter: 2 lanes 26.42, 4 25.99, 8 25.80, 16 25.87 ms a DeepSeek-V4.1 3-row pass)
    if (q8_gate && slot_q8_kind(gate_kind) && gate.cols >= 4096 && gate.cols < 8192) gate_lanes = std::min(gate_lanes, gate_kind == 29 ? 8u : 4u);
#endif
    plan.lanes[0] = gate_kind < 0 ? 32 : gate_lanes;
    plan.lanes[1] = down_kind < 0 ? 32 : down_lanes;
    plan.waves[0] = uint32_t((gate.rows + 32 / plan.lanes[0] - 1) / (32 / plan.lanes[0]));
    plan.waves[1] = uint32_t((down.rows + 32 / plan.lanes[1] - 1) / (32 / plan.lanes[1]));
    // A launch of fewer than two 256-thread blocks a CU (a decode row's shared expert, the downs) takes
    // 64-thread blocks: the same waves over every CU. A wave's rows, lanes and sums are the same either way.
    const auto threads = [&](size_t waves) { return tokens * ((waves + 7) / 8) < 2 * size_t(d.compute_units) ? 64u : 256u; };
    const auto blocks_of = [](size_t waves, unsigned threads) { return unsigned((waves + threads / 32 - 1) / (threads / 32)); };
    const unsigned first_threads = threads(size_t(plan.top_k) * plan.waves[0]), second_threads = threads(plan.waves[1]);
#if defined(__HIPCC__)
    // Each Q8 block stages a token's activations once: as few blocks as fill every CU twice, each covering as many waves.
    const auto gate_q8 = [&](bool sent) {
        if (!sent && plan.groups && slot_q8_kind(gate_kind)) {
            const size_t waves = size_t(plan.groups) * plan.waves[0], slots = 2 * size_t(d.compute_units);
            const size_t per_block = std::clamp<size_t>((waves + slots - 1) / slots, 2, 16);
            const unsigned blocks = unsigned((waves + per_block - 1) / per_block), threads = unsigned(32 * per_block);
            const size_t lds = slot_q8_lds(gate_kind, gate.cols, tokens);
            #define SD_SLOT_GROUPS(format) case format: \
                slot_gate_up_q8_groups<format><<<blocks, threads, lds, d.stream>>>(plan, input, input_stride, weights, product, uint32_t(tokens)); break;
            switch (gate_kind) { SD_SLOT_GROUPS(16) SD_SLOT_GROUPS(17) SD_SLOT_GROUPS(22) SD_SLOT_GROUPS(29) }
            #undef SD_SLOT_GROUPS
            check(hipGetLastError());
            return;
        }
        const size_t waves = size_t(plan.top_k) * plan.waves[0], slots = 2 * size_t(d.compute_units);
        const size_t per_block = std::clamp<size_t>((waves + slots - 1) / slots, 2, 16);
        const dim3 blocks(unsigned((waves + per_block - 1) / per_block), unsigned(tokens));
        const unsigned threads = unsigned(32 * per_block);
        #define SD_SLOT_GATE_Q8(format) case format: \
            if (sent) slot_gate_up_q8<format, true><<<blocks, threads, gate_lds, d.stream>>>(plan, bases, input, input_stride, weights, \
                                                                                         product, *tail, send_at, raise); \
            else slot_gate_up_q8<format, false><<<blocks, threads, gate_lds, d.stream>>>(plan, bases, input, input_stride, weights, \
                                                                                       product, LinkTail{}, 0, 0); \
            break;
        switch (gate_kind) {
            SD_SLOT_GATE_Q8(16) SD_SLOT_GATE_Q8(17) SD_SLOT_GATE_Q8(22) SD_SLOT_GATE_Q8(29)
            case 13:
                if (sent) slot_gate_up_q5k<true><<<blocks, threads, gate_lds, d.stream>>>(plan, bases, input, input_stride, weights, product, *tail,
                                                                                         send_at, raise);
                else slot_gate_up_q5k<false><<<blocks, threads, gate_lds, d.stream>>>(plan, bases, input, input_stride, weights, product,
                                                                                    LinkTail{}, 0, 0);
                break;
        }
        #undef SD_SLOT_GATE_Q8
        check(hipGetLastError());
    };
    const auto down_q8 = [&](bool linked) {
        const size_t waves = plan.waves[1], slots = 2 * size_t(d.compute_units);
        // (a pass of rows already fills every CU with blocks, which each stage the same routes' product rows: wider blocks, fewer of them; IQ2_XXS,
        // 2 to 4 tokens: 58 / 74 / 94 us to 47 / 70 / 88)
        const size_t per_block = std::clamp<size_t>(std::max<size_t>((waves + slots - 1) / slots, tokens >= 2 ? 8 : 0), 2, 16);
        const dim3 blocks(unsigned((waves + per_block - 1) / per_block), unsigned(tokens));
        const unsigned threads = unsigned(32 * per_block);
        #define SD_SLOT_DOWN_Q8(format) case format: \
            if (linked) slot_down_q8<format, true><<<blocks, threads, down_lds, d.stream>>>(plan, bases, product, out, out_stride, *tail); \
            else slot_down_q8<format, false><<<blocks, threads, down_lds, d.stream>>>(plan, bases, product, out, out_stride, LinkTail{}); \
            break;
        switch (down_kind) {
            SD_SLOT_DOWN_Q8(16) SD_SLOT_DOWN_Q8(17) SD_SLOT_DOWN_Q8(29)
            case 13:
                if (linked) slot_down_q5k<true><<<blocks, threads, down_lds, d.stream>>>(plan, bases, product, out, out_stride, *tail);
                else slot_down_q5k<false><<<blocks, threads, down_lds, d.stream>>>(plan, bases, product, out, out_stride, LinkTail{});
                break;
        }
        #undef SD_SLOT_DOWN_Q8
        check(hipGetLastError());
    };
#else
    const auto gate_q8 = [](bool) {};
    const auto down_q8 = [](bool) {};
#endif
    if (tail && !tail->out && stages == 1) {
        if (q8_gate) {
            gate_q8(true);
            return true;
        }
        const dim3 blocks(blocks_of(size_t(plan.top_k) * plan.waves[0], first_threads), unsigned(tokens));
        #define SD_SLOT_GATE_SENT(format) case format: \
            slot_gate_up_sent<format><<<blocks, first_threads, 0, d.stream>>>(plan, bases, input, input_stride, weights, product, *tail, \
                                                                              send_at, raise); break;
        switch (gate_kind) {
            SD_FAST_FORMATS(SD_SLOT_GATE_SENT)
            default: slot_gate_up_sent<-1><<<blocks, first_threads, 0, d.stream>>>(plan, bases, input, input_stride, weights, product,
                                                                                   *tail, send_at, raise);
        }
        #undef SD_SLOT_GATE_SENT
        check(hipGetLastError());
        return true;
    }
    if (tail && (down_kind < 0 || !(stages & 2) || (tail->out && (stages != 3 || tokens != 1)))) return false;
    const dim3 first(blocks_of(size_t(plan.top_k) * plan.waves[0], first_threads), unsigned(tokens));
    const dim3 second(blocks_of(plan.waves[1], second_threads), unsigned(tokens));
    #define SD_SLOT_GATE(format) case format: \
        slot_gate_up<format><<<first, first_threads, 0, d.stream>>>(plan, bases, input, input_stride, weights, product); break;
    if ((stages & 1) && q8_gate) gate_q8(false);
    else if (stages & 1) {
        switch (gate_kind) {
            SD_FAST_FORMATS(SD_SLOT_GATE)
            default: slot_gate_up<-1><<<first, first_threads, 0, d.stream>>>(plan, bases, input, input_stride, weights, product);
        }
        check(hipGetLastError());
    }
    #undef SD_SLOT_GATE
    if (!(stages & 2)) return true;
    if (tail && q8_down) {
        down_q8(true);
        return true;
    }
    if (tail) {
        #define SD_SLOT_LINKED(format) case format: \
            slot_down_linked<format><<<second, second_threads, 0, d.stream>>>(plan, bases, product, out, out_stride, *tail); break;
        switch (down_kind) {
            SD_FAST_FORMATS(SD_SLOT_LINKED)
        }
        #undef SD_SLOT_LINKED
        check(hipGetLastError());
        return true;
    }
    if (q8_down) {
        down_q8(false);
        return true;
    }
    if (down_kind == 29 && tokens >= 2) {
        const uint32_t pair_waves = uint32_t((down.rows + 2 * (32 / plan.lanes[1]) - 1) / (2 * (32 / plan.lanes[1])));
        const unsigned pair_threads = threads(pair_waves);
        slot_down_pairs<29><<<dim3(blocks_of(pair_waves, pair_threads), unsigned(tokens)), pair_threads, 0, d.stream>>>(plan, bases, product, out, out_stride, pair_waves);
        check(hipGetLastError());
        return true;
    }
    #define SD_SLOT_DOWN(format) case format: \
        slot_down<format><<<second, second_threads, 0, d.stream>>>(plan, bases, product, out, out_stride); break;
    switch (down_kind) {
        SD_FAST_FORMATS(SD_SLOT_DOWN)
        default: slot_down<-1><<<second, second_threads, 0, d.stream>>>(plan, bases, product, out, out_stride);
    }
    #undef SD_SLOT_DOWN
    check(hipGetLastError());
    return true;
}
