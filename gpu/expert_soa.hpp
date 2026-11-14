#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "expert_soa_common.hpp"

// One expert retains its native kind and byte count. Unit fields are stored as
// [unit / 4][row][unit % 4]; block scales are [block][row]. Include after Matrix,
// weight, GemvFormat and the IQ helpers. Layout 0 is native and layout 1 is SoA.
// IQ2_XXS: N = rows * cols / 32, q32 at 0, aux32 at 4N, d16 at 8N; still 66 bytes per 256 values.

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

