// Qwen's routed gate and up of the lattice formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) on INT8 activations, the rows of a pass that route to
// one expert reading its weights once. Included after resident_moe.hpp's expert_source and before resident_expert_launch.
//
// The F32 kernel (resident_expert_gemv) loads 128 bytes of activations and turns 32 grid bytes into floats for every 10 to 14 bytes of
// weights it reads, once for every (row, route) of the pass. Here a unit's eight grid words take their signs once, whatever the rows, and
// a row's unit is eight integer dots against its activations quantized to INT8 per 32 values with an F32 scale (gemv_q8_group, as the
// dense projections' inputs are): exact in INT32, then one F32 multiply-add.
//
// Two launches a pass. The first, one small block a row, writes what every block of the second reads, to a scratch of the device: the
// sign masks, the rows' quantized activations, and for each (row, route) of the pass the rows it evaluates -- rows of a pass share about
// a quarter of their routes, and the first row that routes to an expert from given weights takes up to three more that do, the others'
// blocks returning at their first load. The second is the weights' read: a block of it computes nothing before its rows but copies what
// it looks up -- the grid, the masks, its rows' activations -- from the scratch to its shared memory, and keeps no stack (a block that
// quantized its own activations and formed its own masks cost more than its rows' work once a row had more than two lanes, and one that
// kept the launch's plan in its threads' stacks took 3 us to return with nothing to do).
//
// A row's sums are those of the row alone: its own activations' scales, its own integers, the same lanes in the same order, so that what
// shares a pass does not change a row.
//
// A row in the SoA layout (expert_soa.hpp: a field of four units side by side) is walked four units at a load, as words of 16 and 4
// bytes -- a fifth of the load instructions -- every lanes-th such quad a lane; a native row (IQ2_XXS, a staged expert) a unit at a time
// (STRAYDOG_EXPERT_Q8_LANES: the lanes of a matrix row, a power of two up to 16; two, where it is not set).
#pragma once

__host__ __device__ constexpr bool expert_q8_kind(int kind) { return kind == 16 || kind == 17 || kind == 18 || kind == 21 || kind == 22; }
// Sign masks a kind keeps: a mask to every eight sign bits (IQ2_S, IQ3_S), or to every seven stored ones (IQ2_XXS, IQ2_XS, IQ3_XXS: the
// eighth is their parity).
__host__ __device__ constexpr int expert_q8_masks(int kind) { return kind == 21 || kind == 22 ? 256 : 128; }
// Words a row's quantized activations take: `cols` INT8 values and a scale to every 32.
__host__ __device__ constexpr size_t expert_q8_row_words(size_t cols) { return cols / 4 + cols / 32; }
constexpr unsigned expert_q8_rows_most = 8, expert_q8_share = 4;
// The device's scratch (bytes): the masks of eight sign bits, the masks of seven stored ones, a (row, route)'s entry of twelve words (how
// many rows it evaluates, 0 for none; the rows, each row * 10 + route, itself the first; bit 0 and 1 set where its gate and up are in the
// SoA layout; its gate's and its up's weights), then each row's activations (INT8 words, then a scale to every 32).
constexpr size_t expert_q8_masks8_at = 0, expert_q8_masks7_at = 2048, expert_q8_entries_at = 3072, expert_q8_entry_words = 12,
                 expert_q8_rows_at = expert_q8_entries_at + expert_q8_rows_most * 10 * expert_q8_entry_words * 4;
__host__ __device__ constexpr size_t expert_q8_gate_bytes(size_t cols) { return expert_q8_rows_at + expert_q8_rows_most * expert_q8_row_words(cols) * 4; }
// After the gate and up's part, the down's: a (row, route)'s entry of four words (1 where the GPU evaluates it, a word unused, its down's
// weights), then each (row, route)'s product (expert_q8_down_words words: INT8 words, a scale to every 32 values, and the sum of every 32
// INT8 values, which Q2_0's codes -- a value plus one -- take off once a group).
constexpr size_t expert_q8_down_cols = 640, expert_q8_down_words = expert_q8_down_cols / 4 + 2 * (expert_q8_down_cols / 32), expert_q8_down_entry_words = 4;
__host__ __device__ constexpr size_t expert_q8_down_entries_at(size_t cols) { return expert_q8_gate_bytes(cols); }
__host__ __device__ constexpr size_t expert_q8_down_rows_at(size_t cols) {
    return expert_q8_down_entries_at(cols) + expert_q8_rows_most * 10 * expert_q8_down_entry_words * 4;
}
__host__ __device__ constexpr size_t expert_q8_scratch_bytes(size_t cols) {
    return expert_q8_down_rows_at(cols) + expert_q8_rows_most * 10 * expert_q8_down_words * 4;
}
__device__ __forceinline__ const uint8_t* expert_q8_grid(int kind) {
    return kind == 16 ? iq_grid2xxs : kind == 17 ? iq_grid2xs : kind == 22 ? iq_grid2s : kind == 18 ? iq_grid3xxs : iq_grid3s;
}
// Byte j of the word is 0xff where bit j of the four is set.
__device__ __forceinline__ uint32_t expert_q8_spread4(uint32_t bits) { return ((bits * 0x00204081u) & 0x01010101u) * 0xffu; }
// expert_source of route `route` of the pass's row `row`, for the gate (role 0) or the up (1). No copy of the plan, and none of its arrays
// indexed by a number the thread computes: either makes the compiler keep the whole plan in every thread's stack, written at the
// kernel's entry -- 120 bytes of local memory a thread before a block with nothing to do could return (3200 such blocks took 116 us).
__device__ __forceinline__ const uint8_t* expert_q8_source(const MoeGroup* groups, const int32_t* selected, const ExpertGemv& plan, uint32_t row,
                                                           uint32_t route, uint32_t role) {
    const int32_t* routes = selected + row * 10;
    const uint32_t expert = uint32_t(routes[route]);
    const uint8_t* data = role ? groups[expert].weights[1].data : groups[expert].weights[0].data;
    if (data) return plan.only == 2 ? nullptr : data;
    const uint8_t* mapped = role ? plan.mapped[1] : plan.mapped[0];
    if (plan.only == 1 || !plan.assist || !mapped) return nullptr;
    const uint64_t within = role ? plan.bytes[0] : 0, bytes = role ? plan.bytes[1] : plan.bytes[0];
    if (plan.assign) {
        const int32_t slot = plan.assign[row * 10 + route];
        if (slot < 0) return nullptr;
        if (plan.stage) return plan.stage + uint64_t(slot) * plan.slot_bytes + within;
        return mapped + uint64_t(expert) * bytes;
    }
    uint32_t earlier = 0, absent = 0;
    for (uint32_t r = 0; r < 10; ++r) {
        const bool missing = groups[routes[r]].weights[0].data == nullptr;
        absent += missing; earlier += missing && r < route;
    }
    if (earlier >= min(plan.assist, (5 * absent + 4) / 12)) return nullptr;
    if (plan.stage) return plan.stage + (uint64_t(row) * plan.assist + earlier) * plan.slot_bytes + within;
    return mapped + uint64_t(expert) * bytes;
}
// A unit's packed fields, of either layout: the grid codes, the signs, the high bits, the sub-scales and the block's scale.
struct ExpertQ8Unit { uint32_t q0, q1, signs, high, scale, d; };
template<int kind>
__device__ __forceinline__ ExpertQ8Unit expert_q8_unit(const uint8_t* data, uint32_t rows, uint32_t cols, uint32_t row_bytes, uint32_t layout, uint32_t row,
                                                       uint32_t unit) {
    ExpertQ8Unit u{};
    if constexpr (kind == 16) {
        // (IQ2_XXS in the SoA layout: the codes at 0, the sign and scale words after them, the blocks' scales by (block, row) after those)
        if (layout) {
            const uint32_t count = rows * (cols / 32), index = (unit / 4) * rows * 4 + row * 4 + unit % 4;
            u.q0 = reinterpret_cast<const uint32_t*>(data)[index];
            u.signs = reinterpret_cast<const uint32_t*>(data)[count + index];
            u.scale = u.signs >> 28;
            u.d = reinterpret_cast<const uint16_t*>(data + 8 * count)[(unit / 8) * rows + row];
            return u;
        }
    }
    if constexpr (kind == 18 || kind == 21 || kind == 22) {
        if (layout) {
            Matrix m{};
            m.data = data; m.rows = rows; m.cols = cols;
            const ExpertSoaFields f = expert_soa_fields<kind>(m, row, unit);
            u.q0 = f.q0; u.q1 = f.q1; u.signs = f.signs; u.high = f.high; u.scale = f.scale; u.d = f.d;
            if constexpr (kind == 18) u.scale = f.signs >> 28;
            return u;
        }
    }
    constexpr uint32_t block_bytes = kind == 16 ? 66 : kind == 17 ? 74 : kind == 18 ? 98 : kind == 21 ? 110 : 82;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(data + row * row_bytes + (unit / 8) * block_bytes);
    const unsigned ib = unit % 8;
    u.d = h[0];
    if constexpr (kind == 22) {
        u.q0 = iq_pair(h, 1 + 2 * ib);
        u.signs = iq_pair(h, 17 + 2 * ib);
        u.high = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
        u.scale = (h[37 + ib / 2] >> (8 * (ib & 1))) & 255;
    } else if constexpr (kind == 16) {
        u.q0 = iq_pair(h, 1 + 4 * ib);
        u.signs = iq_pair(h, 3 + 4 * ib);
        u.scale = u.signs >> 28;
    } else {
        u.q0 = iq_pair(h, 1 + 4 * ib);
        u.q1 = iq_pair(h, 3 + 4 * ib);
        if constexpr (kind == 17) u.scale = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
        else if constexpr (kind == 18) { u.signs = iq_pair(h, 33 + 2 * ib); u.scale = u.signs >> 28; }
        else {
            u.signs = iq_pair(h, 37 + 2 * ib);
            u.high = (h[33 + ib / 2] >> (8 * (ib & 1))) & 255;
            u.scale = ((h[53 + ib / 4] >> (8 * ((ib / 2) & 1))) >> (4 * (ib & 1))) & 15;
        }
    }
    return u;
}
// The unit's 32 grid values as eight words of signed bytes: a grid byte is 1 to 62, so where its mask byte is set (g ^ 0xff) + 1 is its
// negative in the byte, without a carry.
template<int kind>
__device__ __forceinline__ void expert_q8_words(const ExpertQ8Unit& u, const uint8_t* table, const uint2* masks, uint32_t (&words)[8]) {
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        uint2 g, m;
        if constexpr (kind == 22) {
            g = reinterpret_cast<const uint2*>(table)[((u.q0 >> (8 * l)) & 255) | ((u.high << (8 - 2 * l)) & 0x300)];
            m = masks[(u.signs >> (8 * l)) & 255];
        } else if constexpr (kind == 16) {
            g = reinterpret_cast<const uint2*>(table)[(u.q0 >> (8 * l)) & 255];
            m = masks[(u.signs >> (7 * l)) & 127];
        } else if constexpr (kind == 17) {
            const unsigned code = ((l < 2 ? u.q0 : u.q1) >> (16 * (l & 1))) & 0xffffu;
            g = reinterpret_cast<const uint2*>(table)[code & 511];
            m = masks[code >> 9];
        } else {
            const uint32_t pair = (l < 2 ? u.q0 : u.q1) >> (16 * (l & 1));
            if constexpr (kind == 18) {
                const uint32_t* grid = reinterpret_cast<const uint32_t*>(table);
                g = uint2{grid[pair & 255], grid[(pair >> 8) & 255]};
                m = masks[(u.signs >> (7 * l)) & 127];
            } else {
                const uint32_t* grid = reinterpret_cast<const uint32_t*>(table);
                g = uint2{grid[(pair & 255) | ((u.high << (8 - 2 * l)) & 256)], grid[((pair >> 8) & 255) | ((u.high << (7 - 2 * l)) & 256)]};
                m = masks[(u.signs >> (8 * l)) & 255];
            }
        }
        words[2 * l] = (g.x ^ m.x) + (m.x & 0x01010101u);
        words[2 * l + 1] = (g.y ^ m.y) + (m.y & 0x01010101u);
    }
}
// The unit against one row's staged activations: its integer sum with the sub-scales in, exact in INT32 (16 values of at most 62 * 127
// each, times at most 31).
template<int kind>
__device__ __forceinline__ int expert_q8_sum(const ExpertQ8Unit& u, const uint32_t (&words)[8], const uint32_t* x) {
    const uint4 a = reinterpret_cast<const uint4*>(x)[0], b = reinterpret_cast<const uint4*>(x)[1];
    int first = gemv_dot_s8s8(words[0], a.x, 0), second = gemv_dot_s8s8(words[4], b.x, 0);
    first = gemv_dot_s8s8(words[1], a.y, first); second = gemv_dot_s8s8(words[5], b.y, second);
    first = gemv_dot_s8s8(words[2], a.z, first); second = gemv_dot_s8s8(words[6], b.z, second);
    first = gemv_dot_s8s8(words[3], a.w, first); second = gemv_dot_s8s8(words[7], b.w, second);
    if constexpr (kind == 22 || kind == 17) return int(2 * (u.scale & 15) + 1) * first + int(2 * (u.scale >> 4) + 1) * second;
    else return int(2 * u.scale + 1) * (first + second);
}
// What a unit's integer sum is scaled by besides the activations' scale: IQ2's d * (0.5 + s) * 0.25 (IQ2_XXS's s the unit's, the others'
// one to each half), IQ3_XXS's d * (0.5 + s) * 0.5 and IQ3_S's d * (1 + 2 s), the (2 s + 1) in the integer.
template<int kind>
__device__ __forceinline__ float expert_q8_factor(const ExpertQ8Unit& u) {
    return gemv_half(u.d) * (kind == 21 ? 1.0f : kind == 18 ? 0.25f : 0.125f);
}
// Four units of a row in the SoA layout (units 4 * quad to 4 * quad + 3), their fields loaded as words: the planes hold a field of the
// four side by side (expert_soa_fields' places). `row16` is 16 * row; `step` is the bytes between a quad's fields and the next quad's in a
// plane of 4-byte fields (16 * rows), `count` the fields of a plane. The matrix's data is at a multiple of 16 bytes (the caller's check).
// All in 32-bit arithmetic: a slice is far below 4 GiB.
template<int kind>
__device__ __forceinline__ void expert_q8_quad(const uint8_t* data, uint32_t rows, uint32_t count, uint32_t row16, uint32_t quad, ExpertQ8Unit (&u)[4]) {
    static_assert(kind == 16 || kind == 18 || kind == 21 || kind == 22);
    const uint32_t at = quad * rows * 16 + row16, narrow = at / 4, block = ((quad / 2) * rows) * 2 + row16 / 8;
    const uint4 a = *reinterpret_cast<const uint4*>(data + at), b = *reinterpret_cast<const uint4*>(data + 4 * count + at);
    const uint32_t first[4] = {a.x, a.y, a.z, a.w}, second[4] = {b.x, b.y, b.z, b.w};
    if constexpr (kind == 16) {
        const uint32_t d = *reinterpret_cast<const uint16_t*>(data + 8 * count + block);
        #pragma unroll
        for (int j = 0; j < 4; ++j) u[j] = ExpertQ8Unit{first[j], 0, second[j], 0, second[j] >> 28, d};
        (void)narrow;
    } else if constexpr (kind == 22) {
        const uint32_t high = *reinterpret_cast<const uint32_t*>(data + 8 * count + narrow);
        const uint32_t scale = *reinterpret_cast<const uint32_t*>(data + 9 * count + narrow);
        const uint32_t d = *reinterpret_cast<const uint16_t*>(data + 10 * count + block);
        #pragma unroll
        for (int j = 0; j < 4; ++j) u[j] = ExpertQ8Unit{first[j], 0, second[j], (high >> (8 * j)) & 255, (scale >> (8 * j)) & 255, d};
    } else {
        const uint4 c = *reinterpret_cast<const uint4*>(data + 8 * count + at);
        const uint32_t signs[4] = {c.x, c.y, c.z, c.w};
        if constexpr (kind == 18) {
            const uint32_t d = *reinterpret_cast<const uint16_t*>(data + 12 * count + block);
            #pragma unroll
            for (int j = 0; j < 4; ++j) u[j] = ExpertQ8Unit{first[j], second[j], signs[j], 0, signs[j] >> 28, d};
        } else {
            const uint32_t high = *reinterpret_cast<const uint32_t*>(data + 12 * count + narrow);
            const uint32_t pairs = *reinterpret_cast<const uint16_t*>(data + 13 * count + (quad / 2) * rows * 4 + row16 / 4 + (quad % 2) * 2);
            const uint32_t d = *reinterpret_cast<const uint16_t*>(data + 13 * count + count / 2 + block);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                u[j] = ExpertQ8Unit{first[j], second[j], signs[j], (high >> (8 * j)) & 255, (pairs >> (8 * (j / 2) + 4 * (j & 1))) & 15, d};
        }
    }
}
// One unit against V staged rows, added to their sums.
template<int kind, int V>
__device__ __forceinline__ void expert_q8_add(const ExpertQ8Unit& u, uint32_t unit, const uint8_t* table, const uint2* masks,
                                              const uint32_t* const (&rows_x)[4], uint32_t words, float (&sums)[4]) {
    uint32_t grid[8];
    expert_q8_words<kind>(u, table, masks, grid);
    const float factor = expert_q8_factor<kind>(u);
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const uint32_t* x = rows_x[v];
        const float scale = reinterpret_cast<const float*>(x + words)[unit];
        sums[v] = fmaf(float(expert_q8_sum<kind>(u, grid, x + unit * 8)), factor * scale, sums[v]);
    }
}
// A lane's units of matrix row `row` against V staged rows. A row of the SoA layout is walked four units (a quad) at a load, every
// lanes-th quad from `sub`; a native row a unit at a time, every lanes-th unit from `sub`. A lane's sum is in the order of its units.
template<int kind, int V>
__device__ __forceinline__ void expert_q8_lane(const uint8_t* data, uint32_t rows, uint32_t cols, uint32_t row_bytes, uint32_t layout, uint32_t row, uint32_t sub,
                                               uint32_t lanes, const uint8_t* table, const uint2* masks, const uint32_t* const (&rows_x)[4], float (&sums)[4]) {
    const uint32_t units = cols / 32, words = cols / 4;
    if constexpr (kind == 16 || kind == 18 || kind == 21 || kind == 22) {
        // (a slot whose planes are not at a multiple of 16 bytes, or whose units are not whole quads, is read a unit at a time below)
        if (layout && reinterpret_cast<uintptr_t>(data) % 16 == 0 && units % 8 == 0) {
            const uint32_t count = rows * units, row16 = row * 16;
            for (uint32_t quad = sub; quad < units / 4; quad += lanes) {
                ExpertQ8Unit u[4];
                expert_q8_quad<kind>(data, rows, count, row16, quad, u);
                #pragma unroll
                for (int j = 0; j < 4; ++j) expert_q8_add<kind, V>(u[j], quad * 4 + j, table, masks, rows_x, words, sums);
            }
            return;
        }
    }
    for (uint32_t unit = sub; unit < units; unit += lanes)
        expert_q8_add<kind, V>(expert_q8_unit<kind>(data, rows, cols, row_bytes, layout, row, unit), unit, table, masks, rows_x, words, sums);
}
// The pass's scratch: a block a row (grid.y), 256 threads. Its first lanes quantize the row's activations, a lane a group of 32; its first
// ten write the row's routes' entries; the first row's block writes the masks as well.
__global__ void __launch_bounds__(256) resident_gate_up_q8_plan(const MoeGroup* groups, const int32_t* selected, ExpertGemv plan, const float* input,
                                                                size_t input_stride, uint8_t* scratch, uint32_t rows, uint32_t share, uint32_t cols) { SD_KERNEL_ACQUIRE;
    const uint32_t row = blockIdx.y, lane = threadIdx.x, each = uint32_t(expert_q8_row_words(cols));
    if (row == 0) {
        uint2* masks8 = reinterpret_cast<uint2*>(scratch + expert_q8_masks8_at);
        uint2* masks7 = reinterpret_cast<uint2*>(scratch + expert_q8_masks7_at);
        if (lane < 256) masks8[lane] = uint2{expert_q8_spread4(lane & 15), expert_q8_spread4(lane >> 4)};
        if (lane < 128) { const unsigned bits = iq_signs(lane); masks7[lane] = uint2{expert_q8_spread4(bits & 15), expert_q8_spread4(bits >> 4)}; }
    }
    if (lane < cols / 32) {
        uint32_t* to = reinterpret_cast<uint32_t*>(scratch + expert_q8_rows_at) + row * each;
        gemv_q8_group(input + size_t(row) * input_stride + lane * 32, to + lane * 8, reinterpret_cast<float*>(to + cols / 4) + lane, nullptr);
    }
    if (lane >= 10) return;
    const uint32_t route = lane, pair = row * 10 + route;
    uint32_t* entry = reinterpret_cast<uint32_t*>(scratch + expert_q8_entries_at) + pair * expert_q8_entry_words;
    const int32_t expert = selected[pair];
    const uint8_t* gate = expert_q8_source(groups, selected, plan, row, route, 0);
    const Matrix &first = groups[expert].weights[0], &second = groups[expert].weights[1];
    // (the experts in SoA slots alone: a staged expert's image, the host's native layout, is the F32 kernel's -- a lane a row reads such
    // rows a cache line a lane, and took three times as long as the SoA rows)
    if (!gate || !first.data || !first.layout || !second.data || !second.layout) { entry[0] = 0; return; }
    // The rows that route to this expert from these weights, in order: every fourth leads, and takes the three after it.
    uint32_t earlier = 0, count = 1;
    entry[1] = pair;
    for (uint32_t other = 0; other < rows; ++other) {
        if (other == row || !share) continue;
        for (uint32_t r = 0; r < 10; ++r) {
            if (selected[other * 10 + r] != expert || expert_q8_source(groups, selected, plan, other, r, 0) != gate) continue;
            if (other < row) ++earlier;
            else if (count < expert_q8_share) entry[1 + count++] = other * 10 + r;
        }
    }
    const uint8_t* up = expert_q8_source(groups, selected, plan, row, route, 1);
    entry[5] = 3;
    *reinterpret_cast<const uint8_t**>(entry + 6) = gate;
    *reinterpret_cast<const uint8_t**>(entry + 8) = up;
    entry[0] = earlier % expert_q8_share || !up ? 0 : count;
}
// Bytes of a kind's grid: 8 values a row (IQ2: 256, 512 and 1024 rows) or 4 (IQ3: 256 and 512 rows).
__host__ __device__ constexpr uint32_t expert_q8_grid_bytes(int kind) {
    return kind == 16 ? 256 * 8 : kind == 17 ? 512 * 8 : kind == 22 ? 1024 * 8 : kind == 18 ? 256 * 4 : 512 * 4;
}
// Shared memory of a block of the read (bytes): the kind's grid, its masks and the activations of the rows it evaluates, `rows` at the most.
__host__ __device__ constexpr size_t expert_q8_shared_bytes(int kind, size_t cols, size_t rows) {
    return expert_q8_grid_bytes(kind) + size_t(expert_q8_masks(kind)) * 8 + (rows < expert_q8_share ? rows : expert_q8_share) * expert_q8_row_words(cols) * 4;
}
// The weights' read. grid.x: the waves of a route's gate and up (`waves` each, `lanes` lanes a matrix row) in blocks of eight; grid.y: the
// pass's (row, route), row * 10 + route. `output` takes a route's gate and up as resident_expert_gemv leaves them.
// A block that has rows to evaluate first copies the grid, the masks and those rows' activations to its shared memory, 16 bytes a lane at
// a time (some ten instructions a lane): read from global memory in the loop they were two thirds of the launch (RTX 3090, 40 routes:
// 86 us, 59 without the grid and the masks' loads, 63 without the activations'; without the weights' own loads, 79).
template<int kind>
__global__ void __launch_bounds__(256) resident_gate_up_q8(const uint8_t* scratch, float* output, size_t output_stride, uint32_t matrix_rows, uint32_t cols,
                                                           uint32_t row_bytes, uint32_t waves, uint32_t lanes) { SD_KERNEL_ACQUIRE;
    extern __shared__ __align__(16) uint32_t expert_q8_lds[];
    const uint32_t* entry = reinterpret_cast<const uint32_t*>(scratch + expert_q8_entries_at) + blockIdx.y * expert_q8_entry_words;
    const uint32_t count = entry[0];
    if (!count) return;
    const uint32_t each = uint32_t(expert_q8_row_words(cols));
    constexpr uint32_t grid_chunks = expert_q8_grid_bytes(kind) / 16, mask_chunks = uint32_t(expert_q8_masks(kind)) / 2;
    {
        const uint4* grid = reinterpret_cast<const uint4*>(expert_q8_grid(kind));
        const uint4* signs = reinterpret_cast<const uint4*>(scratch + (kind == 21 || kind == 22 ? expert_q8_masks8_at : expert_q8_masks7_at));
        const uint4* rows = reinterpret_cast<const uint4*>(scratch + expert_q8_rows_at);
        uint4* to = reinterpret_cast<uint4*>(expert_q8_lds);
        const uint32_t row_chunks = each / 4, chunks = grid_chunks + mask_chunks + count * row_chunks;
        // (a lane's loads first, all of them, and then its stores: a load that misses the first cache takes about a microsecond here, and a
        // lane that stored each chunk before it loaded the next waited for its six loads one after another)
        constexpr uint32_t most = (expert_q8_grid_bytes(kind) / 16 + uint32_t(expert_q8_masks(kind)) / 2 + expert_q8_share * 256 + 255) / 256;
        const uint32_t user_rows[4] = {entry[1] / 10, entry[2] / 10, entry[3] / 10, entry[4] / 10};
        uint4 values[most];
        #pragma unroll
        for (uint32_t k = 0; k < most; ++k) {
            const uint32_t i = threadIdx.x + k * blockDim.x;
            if (i < grid_chunks) values[k] = grid[i];
            else if (i < grid_chunks + mask_chunks) values[k] = signs[i - grid_chunks];
            else if (i < chunks) {
                const uint32_t at = i - grid_chunks - mask_chunks, v = at / row_chunks;
                const uint32_t user = v == 0 ? user_rows[0] : v == 1 ? user_rows[1] : v == 2 ? user_rows[2] : user_rows[3];
                values[k] = rows[user * row_chunks + at % row_chunks];
            }
        }
        #pragma unroll
        for (uint32_t k = 0; k < most; ++k) {
            const uint32_t i = threadIdx.x + k * blockDim.x;
            if (i < chunks) to[i] = values[k];
        }
    }
    __syncthreads();
    const uint32_t wave = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, role = wave / waves;
    if (role >= 2) return;
    const uint32_t lane = threadIdx.x % 32, sub = lane % lanes, matrix_row = (wave % waves) * (32 / lanes) + lane / lanes;
    if (matrix_row >= matrix_rows) return;
    const uint8_t* data = role ? *reinterpret_cast<const uint8_t* const*>(entry + 8) : *reinterpret_cast<const uint8_t* const*>(entry + 6);
    const uint32_t layout = (entry[5] >> role) & 1;
    const uint8_t* table = reinterpret_cast<const uint8_t*>(expert_q8_lds);
    const uint2* masks = reinterpret_cast<const uint2*>(expert_q8_lds + grid_chunks * 4);
    const uint32_t* staged = expert_q8_lds + (grid_chunks + mask_chunks) * 4;
    const uint32_t* const rows_x[4] = {staged, staged + each, staged + 2 * each, staged + 3 * each};
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    switch (count) {
        case 1: expert_q8_lane<kind, 1>(data, matrix_rows, cols, row_bytes, layout, matrix_row, sub, lanes, table, masks, rows_x, sums); break;
        case 2: expert_q8_lane<kind, 2>(data, matrix_rows, cols, row_bytes, layout, matrix_row, sub, lanes, table, masks, rows_x, sums); break;
        case 3: expert_q8_lane<kind, 3>(data, matrix_rows, cols, row_bytes, layout, matrix_row, sub, lanes, table, masks, rows_x, sums); break;
        default: expert_q8_lane<kind, 4>(data, matrix_rows, cols, row_bytes, layout, matrix_row, sub, lanes, table, masks, rows_x, sums);
    }
    const uint32_t users[4] = {entry[1], entry[2], entry[3], entry[4]};
    #pragma unroll
    for (uint32_t v = 0; v < expert_q8_share; ++v) {
        if (v >= count) break;
        float total = sums[v];
        for (uint32_t offset = lanes / 2; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (sub == 0) output[size_t(users[v] / 10) * output_stride + (users[v] % 10) * 1280 + role * 640 + matrix_row] = total;
    }
}
// ---- the routed down (2560 x 640, IQ4_NL or Q2_0, native rows of 18-byte units) on INT8 products ----
// The F32 kernel gives a matrix row two lanes, sixteen rows a wave: a wave's load is then 32 pieces of a few bytes each, 90 bytes apart and
// rows apart, and the launch took 110 us of a four-row pass on an RTX 3090 where the gate and up's had come down to 70. Here a lane takes
// one unit and a wave 32 units that follow one another in memory (576 bytes), whatever rows they are of: a block of 640 lanes holds whole
// rows (32 of 20 units, 64 of 10), leaves each lane's part in shared memory, and the row's first lane adds its row's parts in their order.
// Its two launches are as the gate and up's: a block a row writes each (row, route)'s product quantized and its down's weights to the
// scratch, then the read. A staged expert's down is read as a resident one's: both are native rows.
__host__ __device__ constexpr bool expert_q8_down_kind(int kind) { return kind == 20 || kind == 42; }
// expert_source of the down of route `route` of the pass's row `row` (as expert_q8_source: no array of the plan indexed by a computed number).
__device__ __forceinline__ const uint8_t* expert_q8_down_source(const MoeGroup* groups, const int32_t* selected, const ExpertGemv& plan, uint32_t row,
                                                                uint32_t route) {
    const int32_t* routes = selected + row * 10;
    const uint32_t expert = uint32_t(routes[route]);
    const uint8_t* data = groups[expert].weights[2].data;
    if (data) return plan.only == 2 ? nullptr : data;
    if (plan.only == 1 || !plan.assist || !plan.mapped[2]) return nullptr;
    const uint64_t within = plan.bytes[0] + plan.bytes[1];
    if (plan.assign) {
        const int32_t slot = plan.assign[row * 10 + route];
        if (slot < 0) return nullptr;
        if (plan.stage) return plan.stage + uint64_t(slot) * plan.slot_bytes + within;
        return plan.mapped[2] + uint64_t(expert) * plan.bytes[2];
    }
    uint32_t earlier = 0, absent = 0;
    for (uint32_t r = 0; r < 10; ++r) {
        const bool missing = groups[routes[r]].weights[0].data == nullptr;
        absent += missing; earlier += missing && r < route;
    }
    if (earlier >= min(plan.assist, (5 * absent + 4) / 12)) return nullptr;
    if (plan.stage) return plan.stage + (uint64_t(row) * plan.assist + earlier) * plan.slot_bytes + within;
    return plan.mapped[2] + uint64_t(expert) * plan.bytes[2];
}
// A block a row (grid.y), 256 threads: lane 20 * route + group quantizes group `group` of the route's product, and a route's first lane
// writes its entry.
__global__ void __launch_bounds__(256) resident_down_q8_plan(const MoeGroup* groups, const int32_t* selected, ExpertGemv plan, const float* product,
                                                             size_t product_stride, uint8_t* scratch, uint32_t cols) { SD_KERNEL_ACQUIRE;
    const uint32_t row = blockIdx.y, route = threadIdx.x / 20, group = threadIdx.x % 20, pair = row * 10 + route;
    if (route >= 10) return;
    const uint8_t* down = expert_q8_down_source(groups, selected, plan, row, route);
    if (group == 0) {
        uint32_t* entry = reinterpret_cast<uint32_t*>(scratch + expert_q8_down_entries_at(cols)) + pair * expert_q8_down_entry_words;
        *reinterpret_cast<const uint8_t**>(entry + 2) = down;
        entry[0] = down ? 1u : 0u;
    }
    if (!down) return;
    const float* from = product + size_t(row) * product_stride + route * expert_q8_down_cols + group * 32;
    uint32_t* to = reinterpret_cast<uint32_t*>(scratch + expert_q8_down_rows_at(cols)) + pair * expert_q8_down_words;
    float4 v[8];
    float top = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = gemv_load4(from + 4 * i);
        top = fmaxf(top, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    }
    const float inverse = top > 0.0f ? 127.0f / top : 0.0f;
    int total = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int a = int(rintf(v[i].x * inverse)), b = int(rintf(v[i].y * inverse)), c = int(rintf(v[i].z * inverse)), e = int(rintf(v[i].w * inverse));
        total += a + b + c + e;
        to[group * 8 + i] = uint32_t(a & 255) | uint32_t(b & 255) << 8 | uint32_t(c & 255) << 16 | uint32_t(e & 255) << 24;
    }
    reinterpret_cast<float*>(to + expert_q8_down_cols / 4)[group] = top / 127.0f;
    reinterpret_cast<int32_t*>(to + expert_q8_down_cols / 4 + expert_q8_down_cols / 32)[group] = total;
}
// The read. grid.x: blocks of 640 lanes, each whole rows of the matrix (640 / units of them); grid.y: the pass's (row, route).
template<int kind>
__global__ void __launch_bounds__(640) resident_down_q8(const uint8_t* scratch, float* output, size_t output_stride, uint32_t matrix_rows, uint32_t cols) {
    SD_KERNEL_ACQUIRE;
    static_assert(kind == 20 || kind == 42);
    extern __shared__ __align__(16) uint32_t expert_q8_down_lds[];
    const uint32_t* entry = reinterpret_cast<const uint32_t*>(scratch + expert_q8_down_entries_at(cols)) + blockIdx.y * expert_q8_down_entry_words;
    if (!entry[0]) return;
    constexpr uint32_t units = kind == 20 ? expert_q8_down_cols / 32 : expert_q8_down_cols / 64, chunks = expert_q8_down_words / 4;
    if (threadIdx.x < chunks)
        reinterpret_cast<uint4*>(expert_q8_down_lds)[threadIdx.x] =
            reinterpret_cast<const uint4*>(scratch + expert_q8_down_rows_at(cols))[blockIdx.y * chunks + threadIdx.x];
    __syncthreads();
    const uint32_t* x = expert_q8_down_lds;
    const float* scales = reinterpret_cast<const float*>(x + expert_q8_down_cols / 4);
    const int32_t* sums = reinterpret_cast<const int32_t*>(x + expert_q8_down_cols / 4 + expert_q8_down_cols / 32);
    float* parts = reinterpret_cast<float*>(expert_q8_down_lds + expert_q8_down_words);
    const uint32_t row = blockIdx.x * (blockDim.x / units) + threadIdx.x / units, unit = threadIdx.x % units;
    float part = 0.0f;
    if (row < matrix_rows) {
        const uint8_t* data = *reinterpret_cast<const uint8_t* const*>(entry + 2);
        // (a unit's 18 bytes start at an even address: five aligned words hold them, a half word on when the address is not a multiple of 4)
        const uintptr_t address = reinterpret_cast<uintptr_t>(data) + (row * units + unit) * 18;
        const uint32_t* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t(3));
        const bool odd = (address & 2) != 0;
        const uint32_t w0 = base[0], w1 = base[1], w2 = base[2], w3 = base[3], w4 = base[4];
        const uint32_t words[5] = {w0, w1, w2, w3, w4};
        const float d = gemv_half(odd ? w0 >> 16 : w0);
        if constexpr (kind == 20) {
            // IQ4_NL: byte j's low nibble is value j, its high nibble value 16 + j; gemv_iq4's bytes are the values + 128.
            int sum = 0;
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint32_t codes = odd ? words[m + 1] : gemv_realign(words[m], words[m + 1], 2);
                sum = gemv_dot_s8s8(gemv_iq4(codes & 0x0F0F0F0Fu) ^ 0x80808080u, x[unit * 8 + m], sum);
                sum = gemv_dot_s8s8(gemv_iq4((codes >> 4) & 0x0F0F0F0Fu) ^ 0x80808080u, x[unit * 8 + 4 + m], sum);
            }
            part = float(sum) * (d * scales[unit]);
        } else {
            // Q2_0: byte j of word i holds values 16 i + 4 j to 16 i + 4 j + 3, two bits each, a value plus one; a unit is two groups of 32.
            int first = 0, second = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t codes = odd ? words[i + 1] : gemv_realign(words[i], words[i + 1], 2);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // (the four fields a byte each, by shifts and ORs: a multiplication would carry where the shifted copies overlap)
                    const uint32_t byte = (codes >> (8 * j)) & 255u, spread = (byte | byte << 6 | byte << 12 | byte << 18) & 0x03030303u;
                    if (i < 2) first = gemv_dot_s8s8(spread, x[unit * 16 + 4 * i + j], first);
                    else second = gemv_dot_s8s8(spread, x[unit * 16 + 4 * i + j], second);
                }
            }
            // (one fused multiply-add, written: the two groups' sum is then the same whatever the compiler contracts)
            part = d * fmaf(float(first - sums[2 * unit]), scales[2 * unit], float(second - sums[2 * unit + 1]) * scales[2 * unit + 1]);
        }
    }
    parts[threadIdx.x] = part;
    __syncthreads();
    if (unit || row >= matrix_rows) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < units; ++i) total += parts[threadIdx.x + i];
    const uint32_t pass_row = blockIdx.y / 10, route = blockIdx.y % 10;
    output[size_t(pass_row) * output_stride + route * 2560 + row] = total;
}
// Whether a pass's routed gate and up take the kernels above: STRAYDOG_EXPERT_Q8=1; 2, with every (row, route) reading its expert itself
// (the sums a row has are the same either way: the measurement of what the sharing saves, and the check that it changes nothing).
inline int expert_q8_forced = -1;   // (the benchmark's: a level whatever the environment says)
inline int expert_q8_asked = 0;     // (the environment's level and whether the down is taken, as expert_q8_read_environment found them)
inline bool expert_q8_down_asked = true;
inline int expert_q8_level() {
    if (expert_q8_forced >= 0) return expert_q8_forced;
    return expert_q8_asked;
}
// (development: STRAYDOG_EXPERT_Q8_DOWN=0 leaves the routed down to the F32 kernels, to measure the two stages apart)
inline bool expert_q8_down_on() { return expert_q8_down_asked; }
// The two switches as the environment has them: read when a device's scratch is made (a model's load), not at every layer's launches, and
// again at the next load (a process that loads a model twice may have changed them between: examples/qwen_floor does).
inline void expert_q8_read_environment() {
    const char* p = std::getenv("STRAYDOG_EXPERT_Q8");
    expert_q8_asked = p && (p[0] == '1' || p[0] == '2') && !p[1] ? p[0] - '0' : 0;
    const char* down = std::getenv("STRAYDOG_EXPERT_Q8_DOWN");
    expert_q8_down_asked = !(down && down[0] == '0' && !down[1]);
}
// The device's scratch: made with the device (a launch may be recorded into a graph, where nothing can be allocated), for rows of
// expert_q8_scratch_cols values at the most.
constexpr size_t expert_q8_scratch_cols = 2560;
inline void expert_q8_scratch_make(Device& d) {
    expert_q8_read_environment();
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
// Whether the pass's stage is one the kernels above take (the caller's kernel where it is not).
inline bool resident_gate_up_q8_takes(const Device& d, const ResidentMoe& plan, const float* input, size_t tokens, size_t input_stride) {
    const Matrix &gate = plan.weights[0], &up = plan.weights[1];
    const int kind = gate.kind;
    // (the kinds that have an SoA layout, expert_soa_eligible: the kernels read native rows as well, but a lane a row reads them a cache line
    // a lane, and they stay with the F32 kernel)
    return expert_q8_level() && expert_q8_kind(kind) && expert_soa_eligible(kind, gate.rows, gate.cols) && up.kind == kind &&
           gate.rows == up.rows && gate.cols == up.cols && gate.rows == 640 && gate.cols % 256 == 0 && gate.cols / 32 <= 256 &&
           gate.cols <= expert_q8_scratch_cols && d.expert_q8_scratch && expert_q8_row_words(gate.cols) <= 4 * 256 &&
           expert_q8_shared_bytes(kind, gate.cols, tokens) <= gemv_q8_lds_limit && tokens && tokens <= expert_q8_rows_most &&
           reinterpret_cast<uintptr_t>(input) % 16 == 0 && (input_stride * 4) % 16 == 0;
}
// Launches them for the pass's rows (a stage resident_gate_up_q8_takes).
inline void resident_gate_up_q8_launch(Device& d, const ResidentMoe& plan, const MoeGroup* groups, const int32_t* selected, ExpertGemv geometry,
                                       const float* input, float* output, size_t tokens, size_t input_stride, size_t output_stride) {
    const Matrix &gate = plan.weights[0];
    const int kind = gate.kind;
    const int level = expert_q8_level();
    const auto choice = [](const char* name, int otherwise) { const char* p = std::getenv(name); return p && *p ? std::atoi(p) : otherwise; };
    // Two lanes a matrix row: a wave's load of a field is then two runs of 16 rows side by side, whole cache lines still; with more lanes a
    // load spans as many separate pieces (RTX 3090, 40 routes: 71 us with 1 lane, 68 with 2, 89 with 4, 128 with 8, 143 with 16; RTX 5070
    // Ti, 4 rows at 71% resident: 53 with 1, 46 with 2, 56 with 4).
    int lanes = choice("STRAYDOG_EXPERT_Q8_LANES", 2);
    if (lanes != 1 && lanes != 2 && lanes != 4 && lanes != 8 && lanes != 16) lanes = 2;
    static bool told[64] = {};
    if (!told[kind & 63]) {
        told[kind & 63] = true;
        std::fprintf(stderr, "Straydog expert Q8: routed gate and up of kind %d on INT8 activations, level %d, %d lanes a row, %zu rows a pass at the first\n",
                     kind, level, lanes, tokens);
    }
    uint8_t* scratch = d.expert_q8_scratch;
    const uint32_t waves = uint32_t((gate.rows * lanes + 31) / 32);
    const int part = choice("STRAYDOG_EXPERT_Q8_PART", 0);   // (development: 1 the first launch alone, 2 the second alone, on the scratch as it is)
    if (part != 2)
        resident_gate_up_q8_plan<<<dim3(1, unsigned(tokens)), 256, 0, d.stream>>>(groups, selected, geometry, input, input_stride, scratch, uint32_t(tokens),
                                                                                 level == 1 ? 1u : 0u, uint32_t(gate.cols));
    if (part == 1) { check(hipGetLastError()); return; }
    // (Blocks of 256 threads: one copies some 14 to 22 KB of tables before its rows, and an SM loads about 10 GB/s whatever the source, so
    // larger blocks copy less, but each takes longer to end and they came out slower: RTX 3090, 40 routes, 71 us at 256 threads, 87 at 512.)
    const int block = 256;
    const dim3 grid(unsigned((2 * size_t(waves) + 7) / 8), unsigned(tokens * 10));
    const size_t shared = expert_q8_shared_bytes(kind, gate.cols, tokens);
    #define SD_EXPERT_Q8(format) \
        case format: resident_gate_up_q8<format><<<grid, block, shared, d.stream>>>(scratch, output, output_stride, uint32_t(gate.rows), uint32_t(gate.cols), \
                                                                             uint32_t(gate.row_bytes), waves, uint32_t(lanes)); break;
    switch (kind) { SD_EXPERT_Q8(16) SD_EXPERT_Q8(17) SD_EXPERT_Q8(18) SD_EXPERT_Q8(21) SD_EXPERT_Q8(22) }
    #undef SD_EXPERT_Q8
    check(hipGetLastError());
}
// Whether the pass's down stage is one resident_down_q8 takes: the kinds it reads, the shape, the scratch, and a product it can load aligned.
inline bool resident_down_q8_takes(const Device& d, const ResidentMoe& plan, const float* product, size_t tokens, size_t product_stride) {
    const Matrix& down = plan.weights[2];
    return expert_q8_level() && expert_q8_down_on() && expert_q8_down_kind(down.kind) && down.rows == 2560 && down.cols == expert_q8_down_cols && d.expert_q8_scratch &&
           tokens && tokens <= expert_q8_rows_most && reinterpret_cast<uintptr_t>(product) % 16 == 0 && (product_stride * 4) % 16 == 0;
}
// Launches it for the pass's rows: every routed down the GPU evaluates, a resident one or a staged one.
inline void resident_down_q8_launch(Device& d, const ResidentMoe& plan, const MoeGroup* groups, const int32_t* selected, ExpertGemv geometry,
                                    const float* product, float* output, size_t tokens, size_t product_stride, size_t output_stride) {
    const Matrix& down = plan.weights[2];
    uint8_t* scratch = d.expert_q8_scratch;
    resident_down_q8_plan<<<dim3(1, unsigned(tokens)), 256, 0, d.stream>>>(groups, selected, geometry, product, product_stride, scratch,
                                                                         uint32_t(expert_q8_scratch_cols));
    const unsigned units = down.kind == 20 ? unsigned(down.cols / 32) : unsigned(down.cols / 64), rows_a_block = 640 / units;
    const dim3 grid(unsigned((down.rows + rows_a_block - 1) / rows_a_block), unsigned(tokens * 10));
    const size_t shared = (expert_q8_down_words + 640) * 4;
    if (down.kind == 20)
        resident_down_q8<20><<<grid, 640, shared, d.stream>>>(scratch, output, output_stride, uint32_t(down.rows), uint32_t(expert_q8_scratch_cols));
    else
        resident_down_q8<42><<<grid, 640, shared, d.stream>>>(scratch, output, output_stride, uint32_t(down.rows), uint32_t(expert_q8_scratch_cols));
    check(hipGetLastError());
}
