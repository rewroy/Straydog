// Single-token HyperConnection in two dispatches. Stage one splits the down and
// injection projections by residual stream, so each block owns its stream's RMS;
// stage two rebuilds the activated low-rank vector and finishes all four
// streams per output column. Arithmetic stays F32 with the same equations.
__device__ float hc_sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// A Q8_0 unit's eight bytes, two-byte aligned in its block.
#if defined(__HIPCC__)
typedef uint32_t hc_word2 __attribute__((ext_vector_type(2)));
typedef hc_word2 hc_word2_u __attribute__((aligned(2)));
__device__ __forceinline__ hc_word2 hc_load_word2(const uint8_t* at) { return *reinterpret_cast<const hc_word2_u*>(at); }
#else
// (nvcc has no vector types, and PTX loads are naturally aligned: four loads of a pair each)
struct hc_word2 {
    uint32_t words[2];
    __device__ __forceinline__ uint32_t operator[](int i) const { return words[i]; }
};
__device__ __forceinline__ hc_word2 hc_load_word2(const uint8_t* at) {
    const uint16_t* q = reinterpret_cast<const uint16_t*>(at);
    return {{uint32_t(q[0]) | uint32_t(q[1]) << 16, uint32_t(q[2]) | uint32_t(q[3]) << 16}};
}
#endif
// A Q8_0 unit's eight values from its block's scale and its eight bytes.
__device__ __forceinline__ void hc_q8_unit(uint16_t scale, hc_word2 q, float (&w)[8]) {
    const float d = gemv_half(scale);
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = d * float(int8_t(uint8_t(q[i / 4] >> (8 * (i % 4)))));
}
// A Q8_0 unit's bytes as read, decoded by hc_q8_unit when used (hc_load8<8>'s).
struct HcRaw8 { uint16_t scale; hc_word2 quants; };
__device__ __forceinline__ HcRaw8 hc_raw8(const uint8_t* row, size_t unit) {
    const uint8_t* block = row + (unit / 4) * 34;
    return {*reinterpret_cast<const uint16_t*>(block), hc_load_word2(block + 2 + (unit % 4) * 8)};
}
template<int kind>
__device__ __forceinline__ void hc_load8(const uint8_t* row, size_t unit, float (&w)[8]) {
    if constexpr (kind == 8) {
        // Q8_0 (the weights quantized at load): a unit is a quarter of a 34-byte block, its scale the block's; its eight
        // bytes in one two-byte-aligned load (four loads of a pair each were a HyperConnection's issue limit)
        const uint8_t* block = row + (unit / 4) * 34;
        hc_q8_unit(*reinterpret_cast<const uint16_t*>(block), hc_load_word2(block + 2 + (unit % 4) * 8), w);
    } else if (kind == 0) {
        const float4 a = reinterpret_cast<const float4*>(row)[2 * unit], b = reinterpret_cast<const float4*>(row)[2 * unit + 1];
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else {
        const uint4 v = reinterpret_cast<const uint4*>(row)[unit];
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (kind == 30) { w[2 * i] = __uint_as_float(words[i] << 16); w[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u); }
            else { w[2 * i] = gemv_half(words[i]); w[2 * i + 1] = gemv_half(words[i] >> 16); }
        }
    }
}

// V tokens share each weight read: token v reads x + v*10240 and keeps its
// partial sums at scratch + v*hc_decode_scratch, with the one-token arithmetic.
constexpr size_t hc_decode_scratch = 4 * 324 + 4;
// One or two tokens: a block of 24 rows (768 threads, a row a wave) and one stream, the stream's x * norm of the V tokens
// staged in LDS once: the same products, sums and scale as hc_decode_partial_pairs'. Two blocks share a WGP, so all 56 of a
// launch run at once (32-row blocks of 1024 threads, one a WGP, took two rounds for 44).
constexpr unsigned hc_partial_rows = 24;
template<int down_kind, int inject_kind, int V>
__global__ void __launch_bounds__(hc_partial_rows * 32) hc_decode_partial(Matrix down, Matrix inject, bool has_inject, const float* x,
                                                          const float* norm, float* partial, float* inverse) { SD_KERNEL_ACQUIRE;
    const size_t stream = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float reduce[V][8];
    __shared__ __align__(16) float scaled[V][2560];
    const float* ns = norm + stream * 2560;
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float* xs = x + v * 10240 + stream * 2560;
        for (size_t c = threadIdx.x * 4; c < 2560; c += 4 * hc_partial_rows * 32) {
            const float4 a = gemv_load4(xs + c), p = gemv_load4(ns + c);
            *reinterpret_cast<float4*>(&scaled[v][c]) = make_float4(a.x * p.x, a.y * p.y, a.z * p.z, a.w * p.w);
        }
        // (the first eight waves take the mean squares as 256 threads did)
        if (threadIdx.x < 256) {
            float squares = 0;
            for (size_t c = threadIdx.x * 4; c < 2560; c += 1024) {
                const float4 value = gemv_load4(xs + c);
                squares += (value.x * value.x + value.y * value.y) + (value.z * value.z + value.w * value.w);
            }
            for (unsigned offset = 16; offset; offset /= 2) squares += __shfl_xor(squares, offset, 32);
            if (lane == 0) reduce[v][wave] = squares;
        }
    }
    __syncthreads();
    float scale[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) total += reduce[v][i];
        scale[v] = 1.0f / sqrtf(total / 2560.0f + 1e-6f);
        if (blockIdx.x == 0 && threadIdx.x == 0) inverse[v * hc_decode_scratch + stream] = scale[v];
    }
    const size_t row = size_t(blockIdx.x) * hc_partial_rows + wave, rows = 320 + (has_inject ? 4 : 0);
    if (row >= rows) return;
    const bool injection = row >= 320;
    const Matrix& m = injection ? inject : down;
    const uint8_t* data = m.data + (injection ? row - 320 : row) * m.row_bytes;
    float sum[V] = {};
    auto accumulate = [&](size_t unit, const float (&w)[8]) {
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float4 a = *reinterpret_cast<const float4*>(&scaled[v][unit * 8]);
            const float4 b = *reinterpret_cast<const float4*>(&scaled[v][unit * 8 + 4]);
            float total = sum[v];
            total = fmaf(w[0], a.x, total); total = fmaf(w[1], a.y, total);
            total = fmaf(w[2], a.z, total); total = fmaf(w[3], a.w, total);
            total = fmaf(w[4], b.x, total); total = fmaf(w[5], b.y, total);
            total = fmaf(w[6], b.z, total); total = fmaf(w[7], b.w, total);
            sum[v] = total;
        }
    };
    for (size_t unit = lane; unit < 320; unit += 32) {
        float w[8];
        if (injection) hc_load8<inject_kind>(data, stream * 320 + unit, w);
        else hc_load8<down_kind>(data, stream * 320 + unit, w);
        accumulate(unit, w);
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = sum[v];
        for (unsigned offset = 16; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (lane == 0) partial[v * hc_decode_scratch + stream * 324 + row] = total * scale[v];
    }
}

// Three or four tokens: sixteen rows of one stream a block of 256 threads, two a wave, the stream's x * norm of the V tokens
// staged in LDS once and its mean squares taken once (as the first 256 threads of hc_decode_partial take them); each unit's
// staged values are read once for both of a wave's rows. 8-row blocks had read x and the norm from global memory for every row
// (a 4-row pass 0.35 ms longer at 20K and 60K). Each row's products and sums are hc_decode_partial's.
template<int down_kind, int inject_kind, int V>
__global__ void __launch_bounds__(256) hc_decode_partial_pairs(Matrix down, Matrix inject, bool has_inject, const float* x,
                                                               const float* norm, float* partial, float* inverse) { SD_KERNEL_ACQUIRE;
    const size_t stream = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float reduce[V][8];
    __shared__ __align__(16) float scaled[V][2560];
    const float* ns = norm + stream * 2560;
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float* xs = x + v * 10240 + stream * 2560;
        float squares = 0;
        for (size_t c = threadIdx.x * 4; c < 2560; c += 1024) {
            const float4 a = gemv_load4(xs + c), p = gemv_load4(ns + c);
            *reinterpret_cast<float4*>(&scaled[v][c]) = make_float4(a.x * p.x, a.y * p.y, a.z * p.z, a.w * p.w);
            squares += (a.x * a.x + a.y * a.y) + (a.z * a.z + a.w * a.w);
        }
        for (unsigned offset = 16; offset; offset /= 2) squares += __shfl_xor(squares, offset, 32);
        if (lane == 0) reduce[v][wave] = squares;
    }
    __syncthreads();
    float scale[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) total += reduce[v][i];
        scale[v] = 1.0f / sqrtf(total / 2560.0f + 1e-6f);
        if (blockIdx.x == 0 && threadIdx.x == 0) inverse[v * hc_decode_scratch + stream] = scale[v];
    }
    const size_t first = size_t(blockIdx.x) * 16 + wave * 2, rows = 320 + (has_inject ? 4 : 0);
    if (first >= rows) return;
    const bool two = first + 1 < rows;
    const uint8_t* data[2];
    bool injection[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const size_t row = std::min(first + r, rows - 1);
        injection[r] = row >= 320;
        const Matrix& m = injection[r] ? inject : down;
        data[r] = m.data + (injection[r] ? row - 320 : row) * m.row_bytes;
    }
    float sum[2][V] = {};
    auto accumulate = [&](size_t unit, const float (&w)[2][8]) {
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float4 a = *reinterpret_cast<const float4*>(&scaled[v][unit * 8]);
            const float4 b = *reinterpret_cast<const float4*>(&scaled[v][unit * 8 + 4]);
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                float total = sum[r][v];
                total = fmaf(w[r][0], a.x, total); total = fmaf(w[r][1], a.y, total);
                total = fmaf(w[r][2], a.z, total); total = fmaf(w[r][3], a.w, total);
                total = fmaf(w[r][4], b.x, total); total = fmaf(w[r][5], b.y, total);
                total = fmaf(w[r][6], b.z, total); total = fmaf(w[r][7], b.w, total);
                sum[r][v] = total;
            }
        }
    };
    // (a wave's rows are both the injection's or both down's)
    if (down_kind == 8 && !injection[0]) {
        // (the wave's twenty weight reads in flight at once, their bytes held until used: a unit at a time, a 4-row pass
        // was 0.15 ms longer; hc_decode_partial's ten so lost 0.3 ms at two rows)
        HcRaw8 raw[2][10];
        #pragma unroll
        for (int k = 0; k < 10; ++k)
            #pragma unroll
            for (int r = 0; r < 2; ++r) raw[r][k] = hc_raw8(data[r], stream * 320 + lane + 32 * k);
        #pragma unroll
        for (int k = 0; k < 10; ++k) {
            float w[2][8];
            #pragma unroll
            for (int r = 0; r < 2; ++r) hc_q8_unit(raw[r][k].scale, raw[r][k].quants, w[r]);
            accumulate(lane + 32 * k, w);
        }
    } else {
        for (size_t unit = lane; unit < 320; unit += 32) {
            float w[2][8];
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (injection[r]) hc_load8<inject_kind>(data[r], stream * 320 + unit, w[r]);
                else hc_load8<down_kind>(data[r], stream * 320 + unit, w[r]);
            }
            accumulate(unit, w);
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r)
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            float total = sum[r][v];
            for (unsigned offset = 16; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
            if (lane == 0 && (r == 0 || two)) partial[v * hc_decode_scratch + stream * 324 + first + r] = total * scale[v];
        }
}

// A copy that rides in a HyperConnection's second launch (blocks from `first_block` on): a DeltaNet layer's convolution history
// into the scratch its tokens read (gdn_history_snapshot's copy, a launch a layer less). count 0: none.
struct HcCopy { const float4* from = nullptr; float4* to = nullptr; uint32_t count = 0, first_block = 0; };
__device__ __forceinline__ bool hc_copy_block(const HcCopy& copy) {
    if (blockIdx.x < copy.first_block) return false;
    for (size_t i = size_t(blockIdx.x - copy.first_block) * blockDim.x + threadIdx.x; i < copy.count; i += size_t(gridDim.x - copy.first_block) * blockDim.x)
        copy.to[i] = copy.from[i];
    return true;
}
// Three or four tokens: hc_decode_finish with four columns a wave (32 a block, 80 blocks), so the activated low-rank vector is
// rebuilt by a quarter as many blocks (at four tokens its partial sums, read by every block, are more bytes than the up rows).
// Each column's products and sums are hc_decode_finish's.
template<int up_kind, int V>
__global__ void __launch_bounds__(256) hc_decode_finish_columns(Matrix up, bool has_inject, const float* partial, const float* inverse,
                                                                const float* x, const float* norm, float* output, size_t output_stride,
                                                                HcCopy copy) { SD_KERNEL_ACQUIRE;
    if (copy.count && hc_copy_block(copy)) return;
    __shared__ float low[V][320];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float* own = partial + v * hc_decode_scratch;
        for (size_t i = threadIdx.x; i < 320; i += 256) {
            const float value = ((own[i] + own[324 + i]) + (own[648 + i] + own[972 + i])) / 4.0f;
            low[v][i] = value * hc_sigmoid(value);
        }
    }
    __syncthreads();
    const size_t wave = threadIdx.x / 32, lane = threadIdx.x % 32, stream = lane / 8, sub = lane % 8;
    if (has_inject && blockIdx.x == 0 && threadIdx.x < 4) {
        const size_t k = 320 + threadIdx.x;
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* own = partial + v * hc_decode_scratch;
            const float value = (own[k] + own[324 + k]) + (own[648 + k] + own[972 + k]);
            output[v * output_stride + 2560 + threadIdx.x] = 2.0f * hc_sigmoid(value / 4.0f);
        }
    }
    float inverses[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) inverses[v] = inverse[v * hc_decode_scratch + stream];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const size_t column = size_t(blockIdx.x) * 32 + wave * 4 + j, row = stream * 2560 + column;
        const uint8_t* data = up.data + row * up.row_bytes;
        float dot[V] = {};
        #pragma unroll
        for (size_t i = 0; i < 5; ++i) {
            const size_t unit = sub + 8 * i;
            float w[8];
            hc_load8<up_kind>(data, unit, w);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const float4 a = *reinterpret_cast<const float4*>(low[v] + unit * 8), b = *reinterpret_cast<const float4*>(low[v] + unit * 8 + 4);
                float total = dot[v];
                total = fmaf(w[0], a.x, total); total = fmaf(w[1], a.y, total); total = fmaf(w[2], a.z, total); total = fmaf(w[3], a.w, total);
                total = fmaf(w[4], b.x, total); total = fmaf(w[5], b.y, total); total = fmaf(w[6], b.z, total); total = fmaf(w[7], b.w, total);
                dot[v] = total;
            }
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            float total = dot[v];
            for (unsigned offset = 4; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
            float value = hc_sigmoid(total) * (x[v * 10240 + row] * inverses[v] * norm[row]);
            value += __shfl_xor(value, 8, 32);
            value += __shfl_xor(value, 16, 32);
            if (lane == 0) output[v * output_stride + column] = value / 4.0f;
        }
    }
}


template<int up_kind, int V>
__global__ void hc_decode_finish(Matrix up, bool has_inject, const float* partial, const float* inverse,
                                 const float* x, const float* norm, float* output, size_t output_stride, HcCopy copy = {}) { SD_KERNEL_ACQUIRE;
    if (copy.count && hc_copy_block(copy)) return;
    __shared__ float low[V][320];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float* own = partial + v * hc_decode_scratch;
        for (size_t i = threadIdx.x; i < 320; i += 256) {
            const float value = ((own[i] + own[324 + i]) + (own[648 + i] + own[972 + i])) / 4.0f;
            low[v][i] = value * hc_sigmoid(value);
        }
    }
    __syncthreads();
    const size_t wave = threadIdx.x / 32, lane = threadIdx.x % 32, stream = lane / 8, sub = lane % 8;
    const size_t column = size_t(blockIdx.x) * 8 + wave;
    if (has_inject && blockIdx.x == 0 && threadIdx.x < 4) {
        const size_t k = 320 + threadIdx.x;
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* own = partial + v * hc_decode_scratch;
            const float value = (own[k] + own[324 + k]) + (own[648 + k] + own[972 + k]);
            output[v * output_stride + 2560 + threadIdx.x] = 2.0f * hc_sigmoid(value / 4.0f);
        }
    }
    if (column >= 2560) return;
    const size_t row = stream * 2560 + column;
    const uint8_t* data = up.data + row * up.row_bytes;
    float dot[V] = {};
    #pragma unroll
    for (size_t i = 0; i < 5; ++i) {
        const size_t unit = sub + 8 * i;
        float w[8];
        hc_load8<up_kind>(data, unit, w);
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float4 a = *reinterpret_cast<const float4*>(low[v] + unit * 8), b = *reinterpret_cast<const float4*>(low[v] + unit * 8 + 4);
            float total = dot[v];
            total = fmaf(w[0], a.x, total); total = fmaf(w[1], a.y, total); total = fmaf(w[2], a.z, total); total = fmaf(w[3], a.w, total);
            total = fmaf(w[4], b.x, total); total = fmaf(w[5], b.y, total); total = fmaf(w[6], b.z, total); total = fmaf(w[7], b.w, total);
            dot[v] = total;
        }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = dot[v];
        for (unsigned offset = 4; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        float value = hc_sigmoid(total) * (x[v * 10240 + row] * inverse[v * hc_decode_scratch + stream] * norm[row]);
        value += __shfl_xor(value, 8, 32);
        value += __shfl_xor(value, 16, 32);
        if (lane == 0) output[v * output_stride + column] = value / 4.0f;
    }
}

// Supported weight kinds are F32, F16, BF16 and (down and up) Q8_0 (validated by the caller).
template<int down_kind, int V>
void hc_decode_stage_one(Device& d, const Matrix& down, const Matrix& inject, bool has_inject,
                         const float* x, const float* norm, float* partial, float* inverse) {
    if (V >= 3) {
        const dim3 grid(has_inject ? 21 : 20, 4);
        switch (has_inject ? inject.kind : down_kind) {
            case 0: hc_decode_partial_pairs<down_kind, 0, V><<<grid, 256, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
            case 1: hc_decode_partial_pairs<down_kind, 1, V><<<grid, 256, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
            default: hc_decode_partial_pairs<down_kind, 30, V><<<grid, 256, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
        }
        return;
    }
    const dim3 grid((320 + (has_inject ? 4 : 0) + hc_partial_rows - 1) / hc_partial_rows, 4);
    constexpr unsigned threads = hc_partial_rows * 32;
    switch (has_inject ? inject.kind : down_kind) {
        case 0: hc_decode_partial<down_kind, 0, V><<<grid, threads, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
        case 1: hc_decode_partial<down_kind, 1, V><<<grid, threads, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
        default: hc_decode_partial<down_kind, 30, V><<<grid, threads, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
    }
}
template<int V>
void hc_decode_vectors(Device& d, const Matrix* matrices, bool has_inject, const float* x, float* scratch, float* output,
                       size_t output_stride, const HcCopy& copy) {
    const unsigned copy_blocks = copy.count ? 4 : 0;
    HcCopy own = copy, own_wide = copy;
    own.first_block = 80; own_wide.first_block = 320;
    const Matrix& down = matrices[1];
    const Matrix& up = matrices[2];
    const Matrix inject = has_inject ? matrices[3] : matrices[1];
    const float* norm = reinterpret_cast<const float*>(matrices[0].data);
    float* partial = scratch;
    float* inverse = scratch + 4 * 324;
    switch (down.kind) {
        case 0: hc_decode_stage_one<0, V>(d, down, inject, has_inject, x, norm, partial, inverse); break;
        case 1: hc_decode_stage_one<1, V>(d, down, inject, has_inject, x, norm, partial, inverse); break;
        case 8: hc_decode_stage_one<8, V>(d, down, inject, has_inject, x, norm, partial, inverse); break;
        default: hc_decode_stage_one<30, V>(d, down, inject, has_inject, x, norm, partial, inverse); break;
    }
    if (V >= 3) {
        switch (up.kind) {
            case 0: hc_decode_finish_columns<0, V><<<80 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own); break;
            case 1: hc_decode_finish_columns<1, V><<<80 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own); break;
            case 8: hc_decode_finish_columns<8, V><<<80 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own); break;
            default: hc_decode_finish_columns<30, V><<<80 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own); break;
        }
        check(hipGetLastError());
        return;
    }
    switch (up.kind) {
        case 0: hc_decode_finish<0, V><<<320 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own_wide); break;
        case 1: hc_decode_finish<1, V><<<320 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own_wide); break;
        case 8: hc_decode_finish<8, V><<<320 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own_wide); break;
        default: hc_decode_finish<30, V><<<320 + copy_blocks, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride, own_wide); break;
    }
    check(hipGetLastError());
}
// x holds `vectors` rows of 10240; output rows are output_stride apart (2560 mixed + 4 injection).
void hc_decode(Device& d, const Matrix* matrices, bool has_inject, const float* x, float* scratch, float* output,
               size_t vectors=1, size_t output_stride=2564, const HcCopy& copy={}) {
    switch (vectors) {
        case 1: hc_decode_vectors<1>(d, matrices, has_inject, x, scratch, output, output_stride, copy); break;
        case 2: hc_decode_vectors<2>(d, matrices, has_inject, x, scratch, output, output_stride, copy); break;
        case 3: hc_decode_vectors<3>(d, matrices, has_inject, x, scratch, output, output_stride, copy); break;
        case 4: hc_decode_vectors<4>(d, matrices, has_inject, x, scratch, output, output_stride, copy); break;
        case 5: case 6: case 7: case 8:
            // Two launches, each row's sums as ever. One launch of stage two took 6 ms more a pass at eight rows (two launches of
            // stage one, whose stream's staged rows of seven are past a block's 64 KiB of LDS, the same: 27.3 ms to 33.4), and stage
            // one of five or six in one launch saved nothing.
            hc_decode_vectors<4>(d, matrices, has_inject, x, scratch, output, output_stride, copy);
            hc_decode(d, matrices, has_inject, x + 4 * 10240, scratch + 4 * hc_decode_scratch, output + 4 * output_stride, vectors - 4, output_stride);
            break;
        default: throw std::runtime_error("invalid Straydog HyperConnection vector count");
    }
}

