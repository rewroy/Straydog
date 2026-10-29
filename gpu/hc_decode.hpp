// Single-token HyperConnection in two dispatches. Stage one splits the down and
// injection projections by residual stream, so each block owns its stream's RMS;
// stage two rebuilds the activated low-rank vector and finishes all four
// streams per output column. Arithmetic stays F32 with the same equations.
__device__ float hc_sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

template<int kind>
__device__ __forceinline__ void hc_load8(const uint8_t* row, size_t unit, float (&w)[8]) {
    if constexpr (kind == 8) {
        // Q8_0 (the weights quantized at load): a unit is a quarter of a 34-byte block, its scale the block's
        const uint8_t* block = row + (unit / 4) * 34;
        const float d = gemv_half(*reinterpret_cast<const uint16_t*>(block));
        const uint16_t* q = reinterpret_cast<const uint16_t*>(block + 2 + (unit % 4) * 8);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint16_t pair = q[i];
            w[2 * i] = d * float(int8_t(uint8_t(pair))); w[2 * i + 1] = d * float(int8_t(uint8_t(pair >> 8)));
        }
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
// A block of 32 rows (1024 threads, a row a wave) and one stream: the stream's x * norm of the V tokens staged in LDS once,
// where 8-row blocks read them from global memory per row (and five times the blocks took the stream's mean squares):
// the same products, sums and scale.
template<int down_kind, int inject_kind, int V>
__global__ void __launch_bounds__(1024) hc_decode_partial(Matrix down, Matrix inject, bool has_inject, const float* x,
                                                          const float* norm, float* partial, float* inverse) {
    const size_t stream = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float reduce[V][8];
    __shared__ __align__(16) float scaled[V][2560];
    const float* ns = norm + stream * 2560;
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float* xs = x + v * 10240 + stream * 2560;
        for (size_t c = threadIdx.x * 4; c < 2560; c += 4096) {
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
    const size_t row = size_t(blockIdx.x) * 32 + wave, rows = 320 + (has_inject ? 4 : 0);
    if (row >= rows) return;
    const bool injection = row >= 320;
    const Matrix& m = injection ? inject : down;
    const uint8_t* data = m.data + (injection ? row - 320 : row) * m.row_bytes;
    float sum[V] = {};
    for (size_t unit = lane; unit < 320; unit += 32) {
        float w[8];
        if (injection) hc_load8<inject_kind>(data, stream * 320 + unit, w);
        else hc_load8<down_kind>(data, stream * 320 + unit, w);
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
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = sum[v];
        for (unsigned offset = 16; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (lane == 0) partial[v * hc_decode_scratch + stream * 324 + row] = total * scale[v];
    }
}

// Three or four tokens: 8-row blocks reading x and the norm per row (their staged x * norm, 30 or 40 KB a block, left a
// CU one block: 16.7 us became 19.0 at four rows). The same arithmetic as hc_decode_partial's.
template<int down_kind, int inject_kind, int V>
__global__ void hc_decode_partial_rows8(Matrix down, Matrix inject, bool has_inject, const float* x, const float* norm,
                                  float* partial, float* inverse) {
    const size_t stream = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float reduce[V][8];
    float scale[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        const float* xs = x + v * 10240 + stream * 2560;
        float squares = 0;
        for (size_t c = threadIdx.x * 4; c < 2560; c += 1024) {
            const float4 value = gemv_load4(xs + c);
            squares += (value.x * value.x + value.y * value.y) + (value.z * value.z + value.w * value.w);
        }
        for (unsigned offset = 16; offset; offset /= 2) squares += __shfl_xor(squares, offset, 32);
        if (lane == 0) reduce[v][wave] = squares;
    }
    __syncthreads();
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) total += reduce[v][i];
        scale[v] = 1.0f / sqrtf(total / 2560.0f + 1e-6f);
        if (blockIdx.x == 0 && threadIdx.x == 0) inverse[v * hc_decode_scratch + stream] = scale[v];
    }
    const size_t row = size_t(blockIdx.x) * 8 + wave, rows = 320 + (has_inject ? 4 : 0);
    if (row >= rows) return;
    const bool injection = row >= 320;
    const Matrix& m = injection ? inject : down;
    const uint8_t* data = m.data + (injection ? row - 320 : row) * m.row_bytes;
    const float* ns = norm + stream * 2560;
    float sum[V] = {};
    for (size_t unit = lane; unit < 320; unit += 32) {
        float w[8];
        if (injection) hc_load8<inject_kind>(data, stream * 320 + unit, w);
        else hc_load8<down_kind>(data, stream * 320 + unit, w);
        const float4 p = gemv_load4(ns + unit * 8), q = gemv_load4(ns + unit * 8 + 4);
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            const float* xs = x + v * 10240 + stream * 2560;
            const float4 a = gemv_load4(xs + unit * 8), b = gemv_load4(xs + unit * 8 + 4);
            float total = sum[v];
            total = fmaf(w[0], a.x * p.x, total); total = fmaf(w[1], a.y * p.y, total);
            total = fmaf(w[2], a.z * p.z, total); total = fmaf(w[3], a.w * p.w, total);
            total = fmaf(w[4], b.x * q.x, total); total = fmaf(w[5], b.y * q.y, total);
            total = fmaf(w[6], b.z * q.z, total); total = fmaf(w[7], b.w * q.w, total);
            sum[v] = total;
        }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        float total = sum[v];
        for (unsigned offset = 16; offset; offset /= 2) total += __shfl_xor(total, offset, 32);
        if (lane == 0) partial[v * hc_decode_scratch + stream * 324 + row] = total * scale[v];
    }
}

template<int up_kind, int V>
__global__ void hc_decode_finish(Matrix up, bool has_inject, const float* partial, const float* inverse,
                                 const float* x, const float* norm, float* output, size_t output_stride) {
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
        const dim3 grid(has_inject ? 41 : 40, 4);
        switch (has_inject ? inject.kind : down_kind) {
            case 0: hc_decode_partial_rows8<down_kind, 0, V><<<grid, 256, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
            case 1: hc_decode_partial_rows8<down_kind, 1, V><<<grid, 256, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
            default: hc_decode_partial_rows8<down_kind, 30, V><<<grid, 256, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
        }
        return;
    }
    const dim3 grid(has_inject ? 11 : 10, 4);
    switch (has_inject ? inject.kind : down_kind) {
        case 0: hc_decode_partial<down_kind, 0, V><<<grid, 1024, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
        case 1: hc_decode_partial<down_kind, 1, V><<<grid, 1024, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
        default: hc_decode_partial<down_kind, 30, V><<<grid, 1024, 0, d.stream>>>(down, inject, has_inject, x, norm, partial, inverse); break;
    }
}
template<int V>
void hc_decode_vectors(Device& d, const Matrix* matrices, bool has_inject, const float* x, float* scratch, float* output,
                       size_t output_stride) {
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
    switch (up.kind) {
        case 0: hc_decode_finish<0, V><<<320, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride); break;
        case 1: hc_decode_finish<1, V><<<320, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride); break;
        case 8: hc_decode_finish<8, V><<<320, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride); break;
        default: hc_decode_finish<30, V><<<320, 256, 0, d.stream>>>(up, has_inject, partial, inverse, x, norm, output, output_stride); break;
    }
    check(hipGetLastError());
}
// x holds `vectors` rows of 10240; output rows are output_stride apart (2560 mixed + 4 injection).
void hc_decode(Device& d, const Matrix* matrices, bool has_inject, const float* x, float* scratch, float* output,
               size_t vectors=1, size_t output_stride=2564) {
    switch (vectors) {
        case 1: hc_decode_vectors<1>(d, matrices, has_inject, x, scratch, output, output_stride); break;
        case 2: hc_decode_vectors<2>(d, matrices, has_inject, x, scratch, output, output_stride); break;
        case 3: hc_decode_vectors<3>(d, matrices, has_inject, x, scratch, output, output_stride); break;
        case 4: hc_decode_vectors<4>(d, matrices, has_inject, x, scratch, output, output_stride); break;
        default: throw std::runtime_error("invalid Straydog HyperConnection vector count");
    }
}

