#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// One expert retains its native kind and byte count. Unit fields are stored as
// [unit / 4][row][unit % 4]; block scales are [block][row]. Include after Matrix,
// weight, GemvFormat and the IQ helpers. Layout 0 is native and layout 1 is SoA.
// IQ2_XXS: N = rows * cols / 32, q32 at 0, aux32 at 4N, d16 at 8N; still 66 bytes per 256 values.
inline bool expert_soa_eligible(int kind, size_t rows, size_t cols) {
    return (kind == 16 || kind == 18 || kind == 21 || kind == 22) && (rows == 320 || rows == 640) && cols == 2560;
}

template<int kind>
__global__ void expert_soa_pack_kernel(const uint8_t* native, uint8_t* packed, size_t rows, size_t cols) {
    static_assert(kind == 16 || kind == 18 || kind == 21 || kind == 22);
    constexpr size_t block_bytes = kind == 16 ? 66 : kind == 18 ? 98 : kind == 21 ? 110 : 82;
    const size_t block = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t blocks_per_row = cols / 256;
    if (block >= rows * blocks_per_row) return;
    const size_t row = block / blocks_per_row, column_block = block % blocks_per_row;
    const size_t count = rows * (cols / 32);
    const uint8_t* source = native + block * block_bytes;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(source);
    for (unsigned part = 0; part < 8; ++part) {
        const size_t unit = column_block * 8 + part;
        const size_t index = (unit / 4) * rows * 4 + row * 4 + unit % 4;
        if constexpr (kind == 16) {
            reinterpret_cast<uint32_t*>(packed)[index] = iq_pair(h, 1 + 4 * part);
            reinterpret_cast<uint32_t*>(packed)[count + index] = iq_pair(h, 3 + 4 * part);
        } else if constexpr (kind == 22) {
            reinterpret_cast<uint32_t*>(packed)[index] = iq_pair(h, 1 + 2 * part);
            reinterpret_cast<uint32_t*>(packed)[count + index] = iq_pair(h, 17 + 2 * part);
            packed[8 * count + index] = source[66 + part];
            packed[9 * count + index] = source[74 + part];
        } else {
            reinterpret_cast<uint32_t*>(packed)[index] = iq_pair(h, 1 + 4 * part);
            reinterpret_cast<uint32_t*>(packed)[count + index] = iq_pair(h, 3 + 4 * part);
            if constexpr (kind == 18)
                reinterpret_cast<uint32_t*>(packed)[2 * count + index] = iq_pair(h, 33 + 2 * part);
            else {
                reinterpret_cast<uint32_t*>(packed)[2 * count + index] = iq_pair(h, 37 + 2 * part);
                packed[12 * count + index] = source[66 + part];
            }
        }
    }
    if constexpr (kind == 21) {
        for (unsigned pair = 0; pair < 4; ++pair)
            packed[13 * count + column_block * rows * 4 + row * 4 + pair] = source[106 + pair];
        reinterpret_cast<uint16_t*>(packed + 13 * count + count / 2)[column_block * rows + row] = h[0];
    } else {
        constexpr size_t planes = kind == 16 ? 8 : kind == 18 ? 12 : 10;
        reinterpret_cast<uint16_t*>(packed + planes * count)[column_block * rows + row] = h[0];
    }
}

// Buffers are disjoint, equally sized, and owned by the caller through stream completion.
inline void expert_soa_pack_launch(const uint8_t* native, uint8_t* packed, int kind,
                                   size_t rows, size_t cols, hipStream_t stream) {
    if (!expert_soa_eligible(kind, rows, cols) || !native || !packed)
        throw std::runtime_error("invalid expert SoA packing shape or buffers");
    const size_t bytes = rows * (cols / 256) * (kind == 16 ? 66 : kind == 18 ? 98 : kind == 21 ? 110 : 82);
    const uintptr_t source = reinterpret_cast<uintptr_t>(native), target = reinterpret_cast<uintptr_t>(packed);
    if (source % 2 || target % 4 || (source < target ? target - source : source - target) < bytes)
        throw std::runtime_error("expert SoA packing needs aligned, disjoint buffers");
    const unsigned blocks = unsigned((rows * (cols / 256) + 255) / 256);
    switch (kind) {
        case 16: expert_soa_pack_kernel<16><<<blocks,256,0,stream>>>(native, packed, rows, cols); break;
        case 18: expert_soa_pack_kernel<18><<<blocks,256,0,stream>>>(native, packed, rows, cols); break;
        case 21: expert_soa_pack_kernel<21><<<blocks,256,0,stream>>>(native, packed, rows, cols); break;
        case 22: expert_soa_pack_kernel<22><<<blocks,256,0,stream>>>(native, packed, rows, cols); break;
    }
    check(hipGetLastError());
}

struct ExpertSoaFields {
    uint32_t q0, q1, signs;
    unsigned high, scale;
    uint16_t d;
};

template<int kind>
__device__ __forceinline__ ExpertSoaFields expert_soa_fields(const Matrix& m, size_t row, size_t unit) {
    static_assert(kind == 16 || kind == 18 || kind == 21 || kind == 22);
    const size_t count = m.rows * (m.cols / 32);
    const size_t index = (unit / 4) * m.rows * 4 + row * 4 + unit % 4;
    const uint32_t* fields = reinterpret_cast<const uint32_t*>(m.data);
    ExpertSoaFields f{};
    f.q0 = fields[index];
    if constexpr (kind == 16) {
        f.signs = fields[count + index];
        f.d = reinterpret_cast<const uint16_t*>(m.data + 8 * count)[(unit / 8) * m.rows + row];
    } else if constexpr (kind == 22) {
        f.signs = fields[count + index];
        f.high = m.data[8 * count + index];
        f.scale = m.data[9 * count + index];
        f.d = reinterpret_cast<const uint16_t*>(m.data + 10 * count)[(unit / 8) * m.rows + row];
    } else {
        f.q1 = fields[count + index];
        f.signs = fields[2 * count + index];
        if constexpr (kind == 18)
            f.d = reinterpret_cast<const uint16_t*>(m.data + 12 * count)[(unit / 8) * m.rows + row];
        else {
            f.high = m.data[12 * count + index];
            const unsigned pair = m.data[13 * count + (unit / 8) * m.rows * 4 + row * 4 + (unit % 8) / 2];
            f.scale = (pair >> (4 * (unit & 1))) & 15;
            f.d = reinterpret_cast<const uint16_t*>(m.data + 13 * count + count / 2)[(unit / 8) * m.rows + row];
        }
    }
    return f;
}

template<int kind, int V>
__device__ __forceinline__ void expert_soa_dot(const Matrix& m, size_t row, size_t unit, const float* input,
                                             uint32_t stride, float (&r)[V]) {
    static_assert(kind == 16 || kind == 18 || kind == 21 || kind == 22);
    if (m.layout == 0) {
        GemvFormat<kind>::template dots<V>(m.data + row * m.row_bytes, unit, input, stride, r);
        return;
    }
    const ExpertSoaFields f = expert_soa_fields<kind>(m, row, unit);
    if constexpr (kind == 16) {
        float sum[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) sum[v] = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid2xxs)[(f.q0 >> (8 * l)) & 255];
            iq_dot8<V, false>(g.x, g.y, iq_signs((f.signs >> (7 * l)) & 127), input, stride, unit * 32 + 8 * l, sum);
        }
        const float db = gemv_half(f.d) * (0.5f + float(f.signs >> 28)) * 0.25f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = db * sum[v];
    } else if constexpr (kind == 22) {
        float first[V], second[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) { first[v] = 0; second[v] = 0; }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned index = ((f.q0 >> (8 * l)) & 255) | ((f.high << (8 - 2 * l)) & 0x300);
            const uint2 g = reinterpret_cast<const uint2*>(iq_grid2s)[index];
            iq_dot8<V>(g.x, g.y, (f.signs >> (8 * l)) & 255, input, stride, unit * 32 + 8 * l, l < 2 ? first : second);
        }
        const float d = gemv_half(f.d);
        const float d0 = d * (0.5f + float(f.scale & 15)) * 0.25f, d1 = d * (0.5f + float(f.scale >> 4)) * 0.25f;
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = d0 * first[v] + d1 * second[v];
    } else {
        const uint32_t q[2] = {f.q0, f.q1};
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(kind == 18 ? iq_grid3xxs : iq_grid3s);
        float sum[V];
        #pragma unroll
        for (int v = 0; v < V; ++v) sum[v] = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t pair = q[l / 2] >> (16 * (l & 1));
            if constexpr (kind == 18) {
                iq_dot8<V>(grid[pair & 255], grid[(pair >> 8) & 255], iq_signs((f.signs >> (7 * l)) & 127),
                           input, stride, unit * 32 + 8 * l, sum);
            } else {
                const unsigned first = (pair & 255) | ((f.high << (8 - 2 * l)) & 256);
                const unsigned second = ((pair >> 8) & 255) | ((f.high << (7 - 2 * l)) & 256);
                iq_dot8<V>(grid[first], grid[second], (f.signs >> (8 * l)) & 255, input, stride, unit * 32 + 8 * l, sum);
            }
        }
        const float db = kind == 18 ? gemv_half(f.d) * (0.5f + float(f.signs >> 28)) * 0.5f
                                    : gemv_half(f.d) * float(1 + 2 * f.scale);
        #pragma unroll
        for (int v = 0; v < V; ++v) r[v] = db * sum[v];
    }
}

template<int kind>
__device__ __forceinline__ float expert_soa_weight(const Matrix& m, size_t row, size_t col) {
    static_assert(kind == 16 || kind == 18 || kind == 21 || kind == 22);
    if (m.layout == 0) return weight<kind>(m.data + row * m.row_bytes, col);
    const size_t unit = col / 32;
    const unsigned l = unsigned((col % 32) / 8), j = unsigned(col % 8);
    const ExpertSoaFields f = expert_soa_fields<kind>(m, row, unit);
    const float d = gemv_half(f.d);
    float db, g;
    unsigned sign;
    if constexpr (kind == 16) {
        db = d * (0.5f + float(f.signs >> 28)) * 0.25f;
        g = float(iq_grid2xxs[((f.q0 >> (8 * l)) & 255) * 8 + j]);
        sign = (iq_signs((f.signs >> (7 * l)) & 127) >> j) & 1;
    } else if constexpr (kind == 22) {
        const unsigned index = ((f.q0 >> (8 * l)) & 255) | ((f.high << (8 - 2 * l)) & 0x300);
        db = d * (0.5f + float((f.scale >> (4 * (l / 2))) & 15)) * 0.25f;
        g = float(iq_grid2s[index * 8 + j]);
        sign = (f.signs >> (8 * l + j)) & 1;
    } else {
        const uint32_t pair = (l < 2 ? f.q0 : f.q1) >> (16 * (l & 1));
        unsigned index = (pair >> (8 * (j / 4))) & 255;
        if constexpr (kind == 18) {
            db = d * (0.5f + float(f.signs >> 28)) * 0.5f;
            g = float(iq_grid3xxs[index * 4 + j % 4]);
            sign = (iq_signs((f.signs >> (7 * l)) & 127) >> j) & 1;
        } else {
            index |= (f.high << (8 - 2 * l - j / 4)) & 256;
            db = d * float(1 + 2 * f.scale);
            g = float(iq_grid3s[index * 4 + j % 4]);
            sign = (f.signs >> (8 * l + j)) & 1;
        }
    }
    return db * g * (sign ? -1.0f : 1.0f);
}

__device__ inline float expert_soa_weight(const Matrix& m, size_t row, size_t col) {
    if (m.layout == 1) {
        switch (m.kind) {
            case 16: return expert_soa_weight<16>(m, row, col);
            case 18: return expert_soa_weight<18>(m, row, col);
            case 21: return expert_soa_weight<21>(m, row, col);
            case 22: return expert_soa_weight<22>(m, row, col);
            default: return 0;
        }
    }
    #define SD_SOA_NATIVE(format) case format: return weight<format>(m.data + row * m.row_bytes, col);
    switch (m.kind) { SD_FORMATS(SD_SOA_NATIVE) default: return 0; }
    #undef SD_SOA_NATIVE
}
