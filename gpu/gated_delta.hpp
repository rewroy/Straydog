// Independently authored Qwen Gated DeltaNet: fixed geometry, persistent state.
template<int P = 1>
__global__ void gdn_conv(float* qkv, float* history, const float* coefficients, size_t steps) {
    constexpr size_t C = QwenHeads<P>::channels;
    const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (column >= C) return;
    float a = history[column], b = history[C+column];
    float c = history[2*C+column], d = history[3*C+column];
    for (size_t token = 0; token < steps; ++token) {
        a=b; b=c; c=d; d=qkv[token*C+column];
        float value = a*coefficients[column*4] + b*coefficients[column*4+1];
        value += c*coefficients[column*4+2]; value += d*coefficients[column*4+3];
        qkv[token*C+column] = value * hc_sigmoid(value);
    }
    history[column]=a; history[C+column]=b; history[2*C+column]=c; history[3*C+column]=d;
}

// gdn_conv over runs of a chunk's rows at once (its arithmetic, the same windows): first the inputs before each run of
// gdn_conv_run rows (the history's for the first) set aside in `edges` and the history moved on to the chunk's last inputs,
// then a thread a column and run. (gdn_conv walked a chunk's rows one after the other: 20 or 40 blocks, 0.65 s of a 60K prompt.)
constexpr size_t gdn_conv_run = 32;
template<int P = 1>
__global__ void gdn_conv_edges(const float* qkv, float* history, size_t steps, float* edges) {
    constexpr size_t C = QwenHeads<P>::channels;
    const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (column >= C) return;
    const float old[4] = {history[column], history[C + column], history[2 * C + column], history[3 * C + column]};
    // input i of the chunk with the history before it: i < 0 the history's (old[4 + i])
    auto at = [&](long i) { return i < 0 ? old[4 + i] : qkv[size_t(i) * C + column]; };
    for (size_t run = 0; run * gdn_conv_run < steps; ++run)
        for (int tap = 0; tap < 3; ++tap) edges[(run * 3 + tap) * C + column] = at(long(run * gdn_conv_run) - 3 + tap);
    for (int row = 0; row < 4; ++row) history[row * C + column] = at(long(steps) - 4 + row);
}
template<int P = 1>
__global__ void gdn_conv_runs(float* qkv, const float* edges, const float* coefficients, size_t steps) {
    constexpr size_t C = QwenHeads<P>::channels;
    const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x, run = blockIdx.y;
    if (column >= C) return;
    float b = edges[(run * 3) * C + column], c = edges[(run * 3 + 1) * C + column], d = edges[(run * 3 + 2) * C + column];
    const size_t last = std::min(steps, (run + 1) * gdn_conv_run);
    for (size_t token = run * gdn_conv_run; token < last; ++token) {
        const float a = b;
        b = c; c = d; d = qkv[token * C + column];
        float value = a*coefficients[column*4] + b*coefficients[column*4+1];
        value += c*coefficients[column*4+2]; value += d*coefficients[column*4+3];
        qkv[token * C + column] = value * hc_sigmoid(value);
    }
}

template<int P = 1>
__global__ void gdn_normalize_qk(const float* qkv, size_t steps, float* packed) {
    using H = QwenHeads<P>;
    const size_t token = blockIdx.x / H::key, head = blockIdx.x % H::key, col = threadIdx.x;
    const size_t source = token*H::channels+head*128+col, elements = steps*H::value*128;
    const float q = qkv[source], k = qkv[source+H::key*128];
    __shared__ float sums[2][128];
    sums[0][col]=q*q; sums[1][col]=k*k;
    __syncthreads();
    for (unsigned stride = 64; stride; stride /= 2) {
        if (col < stride) { sums[0][col]+=sums[0][col+stride]; sums[1][col]+=sums[1][col+stride]; }
        __syncthreads();
    }
    const float query = q / sqrtf(sums[0][0]+1e-6f) / sqrtf(128.0f);
    const float key = k / sqrtf(sums[1][0]+1e-6f);
    for (size_t tile=0; tile<3; ++tile) {
        const size_t target = (token*H::value+head+tile*H::key)*128+col;
        packed[target]=query; packed[elements+target]=key;
    }
}

template<int P = 1>
__global__ void gdn_pack_values(const float* qkv, const float* beta, const float* alpha,
                               const float* bias, const float* decay, size_t steps, float* packed) {
    using H = QwenHeads<P>;
    const size_t index = size_t(blockIdx.x)*256+threadIdx.x, elements=steps*H::value*128;
    if (index >= elements) return;
    const size_t token=index/H::gate, head=(index%H::gate)/128, col=index%128;
    packed[2*elements+index]=qkv[token*H::channels+H::key*256+head*128+col];
    if (col == 0) {
        const size_t scalar=token*H::value+head;
        const float value=alpha[scalar]+bias[head];
        const float softplus=fmaxf(value,0.0f)+log1pf(expf(-fabsf(value)));
        packed[3*elements+scalar]=expf(decay[head]*softplus);
        packed[3*elements+steps*H::value+scalar]=hc_sigmoid(beta[scalar]);
    }
}

__global__ void gdn_post(const float* recurrent, const float* gate, const float* norm, float* output) {
    const size_t head=blockIdx.x, col=threadIdx.x, index=head*128+col;
    const float x=recurrent[index];
    __shared__ float sum[128]; sum[col]=x*x;
    __syncthreads();
    for (unsigned stride=64; stride; stride/=2) {
        if (col<stride) sum[col]+=sum[col+stride];
        __syncthreads();
    }
    output[index]=x/sqrtf(sum[0]/128.0f+1e-6f)*norm[col]*hc_sigmoid(gate[index]);
}


// Single-token DeltaNet: causal convolution, q/k normalization and value/decay
// packing in one dispatch (blocks 0-15 queries, 16-31 keys, 32-79 values), then
// the recurrence fused with its gated RMS output norm. Same equations as the
// batched kernels above.
__global__ void gdn_prepare_one(float* qkv, float* history, const float* coefficients, const float* beta,
                                const float* alpha, const float* bias, const float* decay, float* packed) {
    const size_t block = blockIdx.x, t = threadIdx.x;
    const size_t column = block < 16 ? block * 128 + t : block < 32 ? 2048 + (block - 16) * 128 + t : 4096 + (block - 32) * 128 + t;
    const float a = history[10240 + column], b = history[20480 + column], c = history[30720 + column], d = qkv[column];
    history[column] = a; history[10240 + column] = b; history[20480 + column] = c; history[30720 + column] = d;
    float value = a * coefficients[column * 4] + b * coefficients[column * 4 + 1];
    value += c * coefficients[column * 4 + 2]; value += d * coefficients[column * 4 + 3];
    value *= hc_sigmoid(value);
    constexpr size_t elements = 48 * 128;
    if (block >= 32) {
        const size_t head = block - 32;
        packed[2 * elements + head * 128 + t] = value;
        if (t == 0) {
            const float raw = alpha[head] + bias[head];
            const float softplus = fmaxf(raw, 0.0f) + log1pf(expf(-fabsf(raw)));
            packed[3 * elements + head] = expf(decay[head] * softplus);
            packed[3 * elements + 48 + head] = hc_sigmoid(beta[head]);
        }
        return;
    }
    __shared__ float sums[128];
    sums[t] = value * value;
    __syncthreads();
    for (unsigned stride = 64; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    const bool query = block < 16;
    const size_t head = query ? block : block - 16;
    const float normalized = query ? value / sqrtf(sums[0] + 1e-6f) / sqrtf(128.0f) : value / sqrtf(sums[0] + 1e-6f);
    for (size_t tile = 0; tile < 3; ++tile) packed[(query ? 0 : elements) + (head + tile * 16) * 128 + t] = normalized;
}
__global__ void delta_post_one(float* state, const float* packed, const float* gate, const float* norm, float* output) {
    const size_t head = blockIdx.x, col = threadIdx.x;
    constexpr size_t elements = 48 * 128;
    __shared__ float query[128], keys[128], sums[128];
    query[col] = packed[head * 128 + col]; keys[col] = packed[elements + head * 128 + col];
    __syncthreads();
    const float factor = packed[3 * elements + head], beta = packed[3 * elements + 48 + head];
    float memory[128];
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) memory[key] = state[(head * 128 + key) * 128 + col];
    float prediction = 0;
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) prediction += (memory[key] * factor) * keys[key];
    const float correction = (packed[2 * elements + head * 128 + col] - prediction) * beta;
    float result = 0;
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) {
        memory[key] = memory[key] * factor + keys[key] * correction;
        result += memory[key] * query[key];
        state[(head * 128 + key) * 128 + col] = memory[key];
    }
    sums[col] = result * result;
    __syncthreads();
    for (unsigned stride = 64; stride; stride /= 2) {
        if (col < stride) sums[col] += sums[col + stride];
        __syncthreads();
    }
    const size_t index = head * 128 + col;
    output[index] = result / sqrtf(sums[0] / 128.0f + 1e-6f) * norm[col] * hc_sigmoid(gate[index]);
}

// Up to four consecutive tokens with the single-token equations of
// gdn_prepare_one/delta_post_one. Block h owns value head h, recomputes its query/key
// head (h % 16) from a private convolution window and keeps its recurrent state in
// registers across the tokens: two threads per state column, each holding 64 keys
// (their partial sums are added low half first). State and history are read from
// *_in and written to state/history at the end; snapshot, when set, receives them
// first. projections: per token qkv (10240) | gate (6144) | beta (48) | alpha (48).
constexpr unsigned gdn_threads = 256;
// Q/K convolution columns are shared by heads h, h+16 and h+32. history_in
// must not alias history: an owner block can finish before another block reads.
// (output null: the tokens' outputs are not kept, a replay's)
template<int P = 1>
__device__ __forceinline__ void
gdn_tokens_head(size_t head, const float* state_in, const float* history_in, float* state, float* history,
                const float* coefficients, const float* bias, const float* decay, const float* norm,
                const float* projections, size_t stride, size_t tokens, float* snapshot, float* output) {
    using H = QwenHeads<P>;
    constexpr size_t C = H::channels;
    const size_t t = threadIdx.x % 128, qk = head % H::key;
    const bool low = threadIdx.x < 128;
    const unsigned first = low ? 0 : 64;
    const size_t columns[3] = {qk * 128 + t, H::key * 128 + qk * 128 + t, H::key * 256 + head * 128 + t};
    float window[3][4] = {};
    if (low)
        #pragma unroll
        for (int c = 0; c < 3; ++c)
            #pragma unroll
            for (int i = 0; i < 4; ++i) window[c][i] = history_in[i * C + columns[c]];
    float memory[64];
    #pragma unroll
    for (unsigned key = 0; key < 64; ++key) memory[key] = state_in[(head * 128 + first + key) * 128 + t];
    if (snapshot) {
        if (low)
            #pragma unroll
            for (int c = 0; c < 3; ++c)
                if (c == 2 || head < H::key)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) snapshot[i * C + columns[c]] = window[c][i];
        #pragma unroll
        for (unsigned key = 0; key < 64; ++key) snapshot[4 * C + (head * 128 + first + key) * 128 + t] = memory[key];
    }
    __shared__ float query[128], keys[128], sums[128], high[128], scalars[2];
    for (size_t token = 0; token < tokens; ++token) {
        const float* p = projections + token * stride;
        float values[3] = {};
        if (low)
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                const size_t column = columns[c];
                const float a = window[c][1], b = window[c][2], e = window[c][3], f = p[column];
                window[c][0] = a; window[c][1] = b; window[c][2] = e; window[c][3] = f;
                float value = a * coefficients[column * 4] + b * coefficients[column * 4 + 1];
                value += e * coefficients[column * 4 + 2]; value += f * coefficients[column * 4 + 3];
                value *= hc_sigmoid(value);
                values[c] = value;
            }
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (low) sums[t] = values[c] * values[c];
            __syncthreads();
            for (unsigned width = 64; width; width /= 2) {
                if (low && t < width) sums[t] += sums[t + width];
                __syncthreads();
            }
            if (low && c == 0) query[t] = values[0] / sqrtf(sums[0] + 1e-6f) / sqrtf(128.0f);
            else if (low) keys[t] = values[1] / sqrtf(sums[0] + 1e-6f);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float raw = p[H::alpha + head] + bias[head];
            const float softplus = fmaxf(raw, 0.0f) + log1pf(expf(-fabsf(raw)));
            scalars[0] = expf(decay[head] * softplus);
            scalars[1] = hc_sigmoid(p[H::beta + head]);
        }
        __syncthreads();
        const float factor = scalars[0], beta = scalars[1];
        float prediction = 0;
        #pragma unroll
        for (unsigned key = 0; key < 64; ++key) prediction += (memory[key] * factor) * keys[first + key];
        if (!low) high[t] = prediction;
        __syncthreads();
        if (low) high[t] = (values[2] - (prediction + high[t])) * beta;
        __syncthreads();
        const float correction = high[t];
        float result = 0;
        #pragma unroll
        for (unsigned key = 0; key < 64; ++key) {
            memory[key] = memory[key] * factor + keys[first + key] * correction;
            result += memory[key] * query[first + key];
        }
        __syncthreads();
        if (!low) high[t] = result;
        __syncthreads();
        if (low) { result += high[t]; sums[t] = result * result; }
        __syncthreads();
        for (unsigned width = 64; width; width /= 2) {
            if (low && t < width) sums[t] += sums[t + width];
            __syncthreads();
        }
        const size_t index = head * 128 + t;
        if (low && output) output[token * H::gate + index] = result / sqrtf(sums[0] / 128.0f + 1e-6f) * norm[t] * hc_sigmoid(p[C + index]);
        __syncthreads();
    }
    #pragma unroll
    for (unsigned key = 0; key < 64; ++key) state[(head * 128 + first + key) * 128 + t] = memory[key];
    if (low)
        #pragma unroll
        for (int c = 0; c < 3; ++c)
            if (c == 2 || head < H::key)
                #pragma unroll
                for (int i = 0; i < 4; ++i) history[i * C + columns[c]] = window[c][i];
}
template<int P = 1>
__global__ void __launch_bounds__(gdn_threads)
gdn_tokens(const float* state_in, const float* history_in, float* state, float* history,
           const float* coefficients, const float* bias, const float* decay, const float* norm,
           const float* projections, size_t stride, size_t tokens, float* snapshot, float* output) {
    gdn_tokens_head<P>(blockIdx.x, state_in, history_in, state, history, coefficients, bias, decay, norm, projections, stride, tokens,
                    snapshot, output);
}

// Keep every head's old convolution window immutable until all blocks have read
// it. Reuse the input arena after the 6144-value projected output, so graphs keep
// fixed pointers and no allocation or floating-point operation is added.
const float* gdn_history_snapshot(Device& d, const float* history, size_t parts = 1) {
    constexpr size_t offset = 4 * 48 * 128 + 6144;
    const size_t count = 4 * 10240 / parts;
    if (offset + count > d.input_capacity) throw std::runtime_error("DeltaNet history scratch exceeds capacity");
    float* snapshot = d.input + offset;
    check(hipMemcpyAsync(snapshot, history, count * sizeof(float), hipMemcpyDeviceToDevice, d.stream));
    return snapshot;
}

// residual (one device-resident token): the output projection adds into the
// decoder hidden state instead of writing `output`.
void gated_delta(Device& d, void* const* pointers, const int* kinds, void* conv, void* recurrent,
                 size_t steps, const float* input, float* output, bool device_io=false,
                 const GemvResidual& residual={}) {
    if (!pointers || !kinds || !input || (!output && !residual.hidden) || !steps || steps>512 || !d.half_input ||
        (residual.hidden && (steps!=1 || !device_io)) ||
        steps*48*(3*128+2)>d.input_capacity || steps*16480>d.output_capacity)
        throw std::runtime_error("invalid Straydog fused DeltaNet workspace");
    const size_t rows[9]={10240,6144,48,48,2560,10240,1,1,1};
    const size_t cols[9]={2560,2560,2560,2560,6144,4,48,48,128};
    Matrix matrices[9]{};
    for (size_t i=0;i<9;++i) {
        if (i>=5 && kinds[i]!=0) throw std::runtime_error("invalid DeltaNet vector type");
        const size_t bytes=matrix_row_bytes(kinds[i],cols[i]);
        auto found=d.allocations.find(pointers[i]);
        if (found==d.allocations.end() || found->second!=rows[i]*bytes)
            throw std::runtime_error("invalid DeltaNet weight size");
        matrices[i]={static_cast<const uint8_t*>(pointers[i]),bytes,rows[i],cols[i],0,0,0,kinds[i]};
    }
    auto cs=d.allocations.find(conv), rs=d.allocations.find(recurrent);
    if (cs==d.allocations.end() || cs->second!=4*10240*4 ||
        rs==d.allocations.end() || rs->second!=48*128*128*4)
        throw std::runtime_error("invalid DeltaNet state size");
    Matrix descriptors[5][max_batch]{};
    // Device-resident inputs are read in place; host inputs are staged once.
    const float* source=input;
    if (!device_io) { upload(d,d.input,input,steps*2560*4); source=d.input; }
    float* result=device_io ? output : d.output;
    float* qkv=d.output;
    float* gate=qkv+steps*10240;
    float* beta=gate+steps*6144;
    float* alpha=beta+steps*48;
    float* targets[4]={qkv,gate,beta,alpha};
    auto submit=[&]() {
    if (steps==1) {
        Matrix projections[4];
        size_t offset=0;
        for (size_t i=0;i<4;++i) {
            projections[i]=matrices[i]; projections[i].output_offset=offset; offset+=matrices[i].rows;
        }
        gemv(d,projections,4,source,d.output);
    } else {
        for (size_t i=0;i<4;++i) hc_project(d,matrices[i],steps,source,targets[i],descriptors[i]);
    }
    if (steps==1) {
        // The same kernel verifies speculative tokens, so both paths agree bitwise.
        const float* history_in = gdn_history_snapshot(d, static_cast<float*>(conv));
        gdn_tokens<<<48,gdn_threads,0,d.stream>>>(static_cast<float*>(recurrent),history_in,
            static_cast<float*>(recurrent),static_cast<float*>(conv),static_cast<const float*>(pointers[5]),
            static_cast<const float*>(pointers[6]),static_cast<const float*>(pointers[7]),static_cast<const float*>(pointers[8]),
            qkv,16480,1,nullptr,d.input+4*48*128);
    } else {
    gdn_conv<<<40,256,0,d.stream>>>(qkv,static_cast<float*>(conv),static_cast<const float*>(pointers[5]),steps);
    gdn_normalize_qk<<<unsigned(steps*16),128,0,d.stream>>>(qkv,steps,d.input);
    gdn_pack_values<<<unsigned((steps*6144+255)/256),256,0,d.stream>>>(qkv,beta,alpha,
        static_cast<const float*>(pointers[6]),static_cast<const float*>(pointers[7]),steps,d.input);
    if (d.rdna4 && steps>1)
        delta_split_sequence<<<48,1024,0,d.stream>>>(static_cast<float*>(recurrent),steps,d.input,qkv);
    else delta_register_sequence<<<48,128,0,d.stream>>>(static_cast<float*>(recurrent),48,steps,d.input,qkv);
    gdn_post<<<unsigned(steps*48),128,0,d.stream>>>(qkv,gate,static_cast<const float*>(pointers[8]),d.input);
    }
    const float* projected=steps==1 ? d.input+4*48*128 : d.input;
    if (residual.hidden) gemv(d,&matrices[4],1,projected,nullptr,1,0,0,residual);
    else hc_project(d,matrices[4],steps,projected,result,descriptors[4]);
    };
    if (steps==1 && !device_io) {
        void* dependencies[11]; std::copy(pointers,pointers+9,dependencies);
        dependencies[9]=conv; dependencies[10]=recurrent;
        run_graph(d,"gdn",dependencies,11,1,submit);
    } else submit();
    check(hipGetLastError());
    if (!device_io) download(d,output,d.output,steps*2560*4);
}
