// Independently authored fused HyperConnection execution. Included in the GPU TU.

__global__ void hc_normalize(const float* input, const float* norm, float* output) {
    const size_t base = size_t(blockIdx.x) * 2560;
    __shared__ float sum[256];
    float total = 0;
    for (size_t col = threadIdx.x; col < 2560; col += 256) {
        const float x = input[base + col]; total += x * x;
    }
    sum[threadIdx.x] = total;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) sum[threadIdx.x] += sum[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = sqrtf(sum[0] / 2560.0f + 1e-6f);
    for (size_t col = threadIdx.x; col < 2560; col += 256)
        output[base + col] = input[base + col] / denominator * norm[(base + col) % 10240];
}

__global__ void hc_activate_low(float* low, size_t count) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index < count) {
        const float value = low[index] / 4.0f;
        low[index] = value * hc_sigmoid(value);
    }
}

__global__ void hc_finish(const float* normalized, const float* up, const float* injection,
                          size_t steps, bool inject, float* output) {
    const size_t index = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (index >= steps * 2560) return;
    const size_t token = index / 2560, col = index % 2560;
    float sum = 0;
    for (size_t stream = 0; stream < 4; ++stream) {
        const size_t source = token * 10240 + stream * 2560 + col;
        sum += hc_sigmoid(up[source]) * normalized[source];
    }
    output[index] = sum / 4.0f;
    if (inject && col < 4)
        output[steps * 2560 + token * 4 + col] = 2.0f * hc_sigmoid(injection[token * 4 + col] / 4.0f);
}

void hc_project(Device& d, const Matrix& matrix, size_t steps, const float* input, float* output,
                Matrix* descriptors) {
    if (d.blas && steps >= d.gemm_min_batch) {
        const float alpha = 1, beta = 0;
        if (d.half_gemm) {
            const size_t count = steps * matrix.cols;
            cast_half<<<unsigned((count+255)/256),256,0,d.stream>>>(input, d.half_input, count);
            expand(matrix, d.expanded, d.stream, true);
            blas_check(hipblasGemmEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                int(matrix.rows), int(steps), int(matrix.cols), &alpha,
                d.expanded, HIP_R_16F, int(matrix.cols), d.half_input, HIP_R_16F, int(matrix.cols), &beta,
                output, HIP_R_32F, int(matrix.rows), HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        } else {
            const float* weights = reinterpret_cast<const float*>(matrix.data);
            if (matrix.kind != 0) { expand(matrix, d.expanded, d.stream, false); weights = d.expanded; }
            blas_check(hipblasSgemm(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                int(matrix.rows), int(steps), int(matrix.cols), &alpha, weights, int(matrix.cols),
                input, int(matrix.cols), &beta, output, int(matrix.rows)));
        }
        ++d.gemm_calls; d.gemm_vectors += steps;
    } else {
        Matrix tokens[gemv_fixed_items];
        for (size_t first = 0; first < steps; first += gemv_fixed_items) {
            const size_t count = std::min(gemv_fixed_items, steps - first);
            for (size_t i = 0; i < count; ++i) {
                tokens[i] = matrix;
                tokens[i].input_offset = (first + i) * matrix.cols;
                tokens[i].output_offset = (first + i) * matrix.rows;
            }
            gemv(d, tokens, count, input, output);
        }
        return;
    }
    check(hipGetLastError());
}

void hyperconnection(Device& d, void* const* pointers, const int* kinds, size_t steps, bool inject,
                      const float* input, float* output, bool device_io=false) {
    if (!pointers || !kinds || !input || !output || !steps || steps > 1024 || !d.half_input ||
        steps * 10564 > d.input_capacity || steps * 10240 > d.output_capacity ||
        steps * 2564 > d.input_capacity / 2)
        throw std::runtime_error("invalid Straydog HyperConnection workspace");
    const size_t rows[4] = {1,320,10240,4}, cols[4] = {10240,10240,320,10240};
    Matrix matrices[4]{};
    for (size_t i = 0; i < (inject ? 4 : 3); ++i) {
        if ((i == 0 && kinds[i] != 0) || (kinds[i] != 0 && kinds[i] != 1 && kinds[i] != 30))
            throw std::runtime_error("invalid Straydog HyperConnection weight type");
        const size_t bytes = matrix_row_bytes(kinds[i], cols[i]);
        auto found = d.allocations.find(pointers[i]);
        if (found == d.allocations.end() || found->second != rows[i] * bytes)
            throw std::runtime_error("invalid Straydog HyperConnection weight size");
        matrices[i] = {static_cast<const uint8_t*>(pointers[i]),bytes,rows[i],cols[i],0,0,0,kinds[i]};
    }
    if (steps==1) {
        const float* x=input;
        if (!device_io) {
            upload(d,d.input,input,10240*4);
            x=d.input;
        }
        float* target=device_io ? output : d.output+2048;
        auto submit=[&]() { hc_decode(d,matrices,inject,x,d.output,target); };
        if (device_io) submit(); else run_graph(d,"hc",pointers,inject ? 4 : 3,size_t(inject),submit);
        check(hipGetLastError());
        if (!device_io) download(d,output,target,(inject ? 2564 : 2560)*4);
        return;
    }
    // Reuse already-budgeted workspaces. Normalized rows remain in output until
    // finish; low/injection/up occupy disjoint slices of input. Descriptor host
    // buffers remain alive until the final stream synchronization.
    Matrix descriptors[3][max_batch]{};
    if (device_io) check(hipMemcpyAsync(d.input, input, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    else upload(d, d.input, input, steps * 10240 * 4);
    hc_normalize<<<unsigned(steps*4),256,0,d.stream>>>(d.input, static_cast<const float*>(pointers[0]), d.output);
    float* low = d.input;
    float* injection = low + steps * 320;
    float* up = injection + steps * 4;
    hc_project(d, matrices[1], steps, d.output, low, descriptors[0]);
    if (inject) hc_project(d, matrices[3], steps, d.output, injection, descriptors[1]);
    hc_activate_low<<<unsigned((steps*320+255)/256),256,0,d.stream>>>(low, steps*320);
    hc_project(d, matrices[2], steps, low, up, descriptors[2]);
    float* compact = reinterpret_cast<float*>(d.half_input);
    hc_finish<<<unsigned((steps*2560+255)/256),256,0,d.stream>>>(d.output, up, injection, steps, inject, compact);
    check(hipGetLastError());
    if (device_io) check(hipMemcpyAsync(output, compact, steps * (inject ? 2564 : 2560) * 4, hipMemcpyDeviceToDevice, d.stream));
    else download(d, output, compact, steps * (inject ? 2564 : 2560) * 4);
}
