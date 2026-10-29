// Speculative verification: up to four consecutive tokens pass each decoder layer
// with the one-token arithmetic, so their logits match token-by-token decode
// bitwise. Weight reads are shared by the multi-vector GEMV and HyperConnection
// kernels; recurrence, attention and experts advance token by token in causal
// order. Each DeltaNet layer keeps a snapshot of its states and the tokens'
// projections, so a partial acceptance restores the snapshot and replays only the
// accepted tokens. QSA K/V needs no rollback: rejected positions are rewritten
// before any later token can select them.
constexpr size_t verify_max = 4, verify_projection = 16480, verify_qsa = 13952;
constexpr size_t verify_snapshot = 4 * 10240 + 48 * 128 * 128;

struct VerifyLayout {
    float* hidden; float* mixed; float* branch; uint32_t* positions;
    float* post; float* qsa; float* attended;
};
constexpr size_t verify_extent = 65536 + verify_max * (6144 + verify_qsa + 6144);
VerifyLayout verify_layout(float* workspace) {
    VerifyLayout v{};
    v.hidden = workspace; v.mixed = v.hidden + verify_max * 10240; v.branch = v.mixed + verify_max * 2564;
    v.positions = reinterpret_cast<uint32_t*>(v.branch + verify_max * 2560);
    v.post = workspace + 65536; v.qsa = v.post + verify_max * 6144; v.attended = v.qsa + verify_max * verify_qsa;
    return v;
}
void hc_operands(void* const* pointers, const int* kinds, Matrix (&matrices)[4]) {
    const size_t rows[4] = {1,320,10240,4}, cols[4] = {10240,10240,320,10240};
    for (size_t i = 0; i < 4; ++i)
        matrices[i] = {static_cast<const uint8_t*>(pointers[i]),matrix_row_bytes(kinds[i],cols[i]),rows[i],cols[i],0,0,0,kinds[i]};
}

void verify_buffers(Device& d, DecoderLayer& plan) {
    if (plan.snapshot) return;
    const size_t bytes = (verify_snapshot + verify_max * verify_projection) * 4;
    if (bytes > d.budget - d.used || allocation_tail > d.budget - d.used - bytes)
        throw std::runtime_error("speculative verification state exceeds the Straydog VRAM budget");
    float* buffer = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&buffer), bytes + allocation_tail));
    d.allocations.emplace(buffer, bytes); d.used += bytes + allocation_tail;
    plan.snapshot = buffer; plan.projections = buffer + verify_snapshot;
}

// steps tokens of a DeltaNet layer; a replay reads the states from the snapshot.
void gdn_verify(Device& d, const DecoderLayer& plan, size_t steps, bool replay, float* post) {
    void* const* p = plan.pointers.data() + 4;
    auto* conv = static_cast<float*>(plan.states[0]); auto* recurrent = static_cast<float*>(plan.states[1]);
    const float* history_in = replay ? plan.snapshot : gdn_history_snapshot(d, conv);
    gdn_tokens<<<48,gdn_threads,0,d.stream>>>(replay ? plan.snapshot + 4 * 10240 : recurrent, history_in,
        recurrent, conv, static_cast<const float*>(p[5]), static_cast<const float*>(p[6]), static_cast<const float*>(p[7]),
        static_cast<const float*>(p[8]), plan.projections, verify_projection, steps, replay ? nullptr : plan.snapshot, post);
}

void verify_layer(Device& d, DecoderLayer& plan, size_t steps, const VerifyLayout& v) {
    void* const* pointers = plan.pointers.data(); const int* kinds = plan.kinds.data();
    const bool linear = decoder_linear(plan.index);
    const size_t attention = linear ? 9 : 10;
    Matrix hc[4];
    hc_operands(pointers, kinds, hc);
    hc_decode(d, hc, true, v.hidden, d.output, v.mixed, steps, 2564);
    const GemvResidual residual{v.hidden, v.mixed, 10240, 2564};
    if (linear) {
        const size_t rows[5] = {10240,6144,48,48,2560}, cols[5] = {2560,2560,2560,2560,6144};
        Matrix m[5];
        size_t offset = 0;
        for (size_t i = 0; i < 5; ++i) {
            m[i] = {static_cast<const uint8_t*>(pointers[4+i]),matrix_row_bytes(kinds[4+i],cols[i]),rows[i],cols[i],0,0,0,kinds[4+i]};
            if (i < 4) { m[i].output_offset = offset; offset += rows[i]; }
        }
        gemv(d, m, 4, v.mixed, plan.projections, steps, 2564, verify_projection);
        gdn_verify(d, plan, steps, false, v.post);
        gemv(d, &m[4], 1, v.post, nullptr, steps, 6144, 0, residual);
    } else {
        Matrix m[6];
        qsa_operands(pointers + 4, kinds + 4, m);
        size_t offset = 0;
        for (size_t i = 0; i < 5; ++i) { m[i].output_offset = offset; offset += m[i].rows; }
        gemv(d, m, 5, v.mixed, v.qsa, steps, 2564, verify_qsa);
        if (steps == 1)
            qsa_decode_attend(d, pointers + 4, plan.states[0], plan.states[1], plan.states[2], plan.context,
                              v.positions, v.qsa, v.attended);
        else   // (the rows side by side: each row the one token's arithmetic)
            qsa_decode_attend_rows(d, pointers + 4, plan.states[0], plan.states[1], plan.states[2], plan.context,
                                   v.positions, v.qsa, verify_qsa, v.attended, steps);
        gemv(d, &m[5], 1, v.attended, nullptr, steps, 6144, 0, residual);
    }
    hc_operands(pointers + 4 + attention, kinds + 4 + attention, hc);
    hc_decode(d, hc, true, v.hidden, d.output, v.mixed, steps, 2564);
    moe_tokens(d, plan.moe, steps, v.mixed, 2564, v.hidden);
}

// The span's layers queued on the device (and on its peer when the experts are in parts), with nothing waited for or
// downloaded: the rows come from the host (`input`), from device memory (`source`), or are already in the span's workspace
// (neither). `parts` tells the caller to finish_with_helper before it reads anything back. Returns the workspace's rows.
float* verify_span_queue(Device& d, size_t begin, size_t end, size_t position, size_t steps, const float* input,
                         const float* source, bool& parts) {
    if (!steps || steps > verify_max || begin >= end || end > 49) throw std::runtime_error("invalid verification span");
    float* workspace = nullptr;
    std::vector<void*> dependencies;
    for (size_t layer = begin; layer < end; ++layer) {
        auto found = d.decoder.find(layer);
        if (found == d.decoder.end() || position + steps > found->second.context ||
            (workspace && workspace != found->second.workspace) || d.resident.find(found->second.moe)->second->hybrid)
            throw std::runtime_error("verification span has no compatible resident layer");
        workspace = found->second.workspace;
        if (decoder_linear(layer)) verify_buffers(d, found->second);
        dependencies.insert(dependencies.end(), found->second.dependencies.begin(), found->second.dependencies.end());
    }
    if (d.allocations.find(workspace)->second / 4 < verify_extent)
        throw std::runtime_error("speculative verification needs a larger decoder workspace");
    kv_prepare_span(d,begin,end,position,steps);
    const VerifyLayout v = verify_layout(workspace);
    if (input) upload(d, v.hidden, input, steps * 10240 * 4);
    else if (source && source != v.hidden) check(hipMemcpyAsync(v.hidden, source, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    uint32_t positions[verify_max];
    for (size_t b = 0; b < steps; ++b) positions[b] = uint32_t(position + b);
    upload(d, v.positions, positions, steps * 4);
    auto submit = [&]() {
        for (size_t layer = begin; layer < end; ++layer) verify_layer(d, d.decoder.find(layer)->second, steps, v);
        check(hipGetLastError());
    };
    parts = span_has_parts(d, begin, end);
    if (parts) {
        run_with_helper(d, begin, end, steps, [&]() {
            run_graph(d, "verify-span", dependencies.data(), dependencies.size(), (begin * 64 + end) * 8 + steps, submit);
        });
    } else run_graph(d, "verify-span", dependencies.data(), dependencies.size(), (begin * 64 + end) * 8 + steps, submit);
    for (size_t layer = begin; layer < end; ++layer) d.decoder.find(layer)->second.verified = steps;
    return v.hidden;
}

// input==nullptr continues from rows already in the span's workspace; output may be nullptr.
void verify_span(Device& d, size_t begin, size_t end, size_t position, size_t steps, const float* input, float* output) {
    bool parts = false;
    float* hidden = verify_span_queue(d, begin, end, position, steps, input, nullptr, parts);
    if (parts) finish_with_helper(d);
    if (output) download(d, output, hidden, steps * 10240 * 4);
    else check(hipStreamSynchronize(d.stream));
}

// A verification's replay of the kept tokens for every DeltaNet layer of a span in one launch (blockIdx.y a layer): each
// from its snapshot over its saved projections into its states, gdn_tokens' arithmetic. (A launch a layer cost the
// span ~18 us a layer, ahead of the drafting on the device that runs the MTP layer.)
struct GdnReplay {
    const float* snapshot; float* state; float* history;
    const float* coefficients; const float* bias; const float* decay; const float* norm; const float* projections;
};
__global__ void __launch_bounds__(gdn_threads) gdn_replay_layers(const GdnReplay* layers, size_t tokens) {
    const GdnReplay& l = layers[blockIdx.y];
    gdn_tokens_head(blockIdx.x, l.snapshot + 4 * 10240, l.snapshot, l.state, l.history, l.coefficients, l.bias, l.decay, l.norm,
                    l.projections, verify_projection, tokens, nullptr, nullptr);
}
void gdn_replay_span(Device& d, size_t begin, size_t end, size_t kept) {
    std::vector<GdnReplay> layers;
    for (size_t layer = begin; layer < end; ++layer) {
        if (!decoder_linear(layer)) continue;
        const DecoderLayer& plan = d.decoder.find(layer)->second;
        void* const* p = plan.pointers.data() + 4;
        layers.push_back({plan.snapshot, static_cast<float*>(plan.states[1]), static_cast<float*>(plan.states[0]),
                          static_cast<const float*>(p[5]), static_cast<const float*>(p[6]), static_cast<const float*>(p[7]),
                          static_cast<const float*>(p[8]), plan.projections});
    }
    if (layers.empty()) return;
    // (the list is the same pointers at every commit of the span: uploaded once, again only if a layer was registered anew)
    const auto* bytes = reinterpret_cast<const uint8_t*>(layers.data());
    const std::vector<uint8_t> host(bytes, bytes + layers.size() * sizeof(GdnReplay));
    auto& cached = d.gdn_replays[{begin, end}];
    if (cached.first != host) {
        if (cached.second && cached.first.size() < host.size()) cached.second = nullptr;
        if (!cached.second) cached.second = allocate(d, host.size());
        upload(d, cached.second, host.data(), host.size());
        cached.first = host;
    }
    gdn_replay_layers<<<dim3(48, unsigned(layers.size())), gdn_threads, 0, d.stream>>>(static_cast<const GdnReplay*>(cached.second), kept);
}

// Keep the first `kept` verified tokens: DeltaNet layers restore their snapshot and
// replay those tokens; everything else already holds causal state.
void verify_commit(Device& d, size_t begin, size_t end, size_t kept) {
    if (begin >= end || !kept) throw std::runtime_error("invalid speculative commit");
    size_t steps = 0;
    for (size_t layer = begin; layer < end; ++layer) {
        auto found = d.decoder.find(layer);
        if (found == d.decoder.end() || !found->second.verified || (steps && found->second.verified != steps))
            throw std::runtime_error("invalid speculative commit");
        steps = found->second.verified;
    }
    if (kept > steps) throw std::runtime_error("invalid speculative commit");
    if (kept < steps) {
        gdn_replay_span(d, begin, end, kept);
        // (queued, not waited for: the stream orders the replay before every later use of the state, and nothing reads it here)
        check(hipGetLastError());
    }
    for (size_t layer = begin; layer < end; ++layer) d.decoder.find(layer)->second.verified = 0;
}

// Row argmax in two passes (64 chunks per row, then one block per row); ties keep
// the lowest index, like numpy.argmax.
constexpr unsigned argmax_chunks = 64;
__device__ void argmax_reduce(float& best, uint32_t& index, float* values, uint32_t* indices) {
    values[threadIdx.x] = best; indices[threadIdx.x] = index;
    __syncthreads();
    for (unsigned offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) {
            const float other = values[threadIdx.x + offset]; const uint32_t at = indices[threadIdx.x + offset];
            if (other > values[threadIdx.x] || (other == values[threadIdx.x] && at < indices[threadIdx.x])) {
                values[threadIdx.x] = other; indices[threadIdx.x] = at;
            }
        }
        __syncthreads();
    }
    best = values[0]; index = indices[0];
}
__global__ void row_argmax_chunks(const float* values, size_t count, size_t stride, float* best_values, uint32_t* best_indices) {
    const float* row = values + size_t(blockIdx.y) * stride;
    const size_t per = (count + argmax_chunks - 1) / argmax_chunks, first = blockIdx.x * per, last = std::min(count, first + per);
    float best = -INFINITY; uint32_t index = UINT32_MAX;
    for (size_t i = first + threadIdx.x; i < last; i += blockDim.x)
        if (row[i] > best || (row[i] == best && uint32_t(i) < index)) { best = row[i]; index = uint32_t(i); }
    __shared__ float shared_values[256]; __shared__ uint32_t shared_indices[256];
    argmax_reduce(best, index, shared_values, shared_indices);
    if (threadIdx.x == 0) {
        best_values[blockIdx.y * argmax_chunks + blockIdx.x] = best; best_indices[blockIdx.y * argmax_chunks + blockIdx.x] = index;
    }
}
__global__ void row_argmax_finish(const float* best_values, const uint32_t* best_indices, int32_t* result) {
    float best = -INFINITY; uint32_t index = UINT32_MAX;
    if (threadIdx.x < argmax_chunks) {
        best = best_values[blockIdx.x * argmax_chunks + threadIdx.x]; index = best_indices[blockIdx.x * argmax_chunks + threadIdx.x];
    }
    __shared__ float shared_values[256]; __shared__ uint32_t shared_indices[256];
    argmax_reduce(best, index, shared_values, shared_indices);
    if (threadIdx.x == 0) result[blockIdx.x] = int32_t(index);
}
void row_argmax(Device& d, const float* values, size_t rows, size_t count, size_t stride, float* scratch, int32_t* result) {
    auto* indices = reinterpret_cast<uint32_t*>(scratch + rows * argmax_chunks);
    row_argmax_chunks<<<dim3(argmax_chunks, unsigned(rows)),256,0,d.stream>>>(values, count, stride, scratch, indices);
    row_argmax_finish<<<unsigned(rows),256,0,d.stream>>>(scratch, indices, result);
}

// A Gumbel(0, 1) variate for (seed, salt, index): a counter-based draw, so that a row's noise needs no state
// and a seed gives the same tokens however the launch is scheduled. In double: the float tail would cut it at 17.
__device__ __forceinline__ float gumbel_noise(uint64_t seed, uint64_t salt, uint64_t index) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ull * (index + 1) + 0xd1b54a32d192ed03ull * (salt + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    const double uniform = (double(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return float(-log(-log(uniform)));
}
// row_argmax_chunks over values * inverse_temperature + Gumbel noise: the argmax of that is a draw from softmax(values * inverse_temperature).
// `sample_at` holds, as uint32, the inverse temperature's float bits (0: no sampling, the plain argmax) and a 64-bit seed (low word first).
__global__ void row_sample_chunks(const float* values, size_t count, size_t stride, const uint32_t* sample_at, uint32_t salt, float* best_values, uint32_t* best_indices) {
    const float inverse_temperature = __uint_as_float(sample_at[0]);
    const uint64_t seed = uint64_t(sample_at[1]) | uint64_t(sample_at[2]) << 32;
    const float* row = values + size_t(blockIdx.y) * stride;
    const size_t per = (count + argmax_chunks - 1) / argmax_chunks, first = blockIdx.x * per, last = std::min(count, first + per);
    float best = -INFINITY; uint32_t index = UINT32_MAX;
    for (size_t i = first + threadIdx.x; i < last; i += blockDim.x) {
        const float value = inverse_temperature != 0.0f ? row[i] * inverse_temperature + gumbel_noise(seed, salt, i) : row[i];
        if (value > best || (value == best && uint32_t(i) < index)) { best = value; index = uint32_t(i); }
    }
    __shared__ float shared_values[256]; __shared__ uint32_t shared_indices[256];
    argmax_reduce(best, index, shared_values, shared_indices);
    if (threadIdx.x == 0) {
        best_values[blockIdx.y * argmax_chunks + blockIdx.x] = best; best_indices[blockIdx.y * argmax_chunks + blockIdx.x] = index;
    }
}
// row_argmax, or a draw from the row's softmax when sample_at asks for one (see row_sample_chunks); `salt` tells the rows of one seed apart.
void row_sample(Device& d, const float* values, size_t rows, size_t count, size_t stride, const uint32_t* sample_at, uint32_t salt, float* scratch, int32_t* result) {
    auto* indices = reinterpret_cast<uint32_t*>(scratch + rows * argmax_chunks);
    row_sample_chunks<<<dim3(argmax_chunks, unsigned(rows)),256,0,d.stream>>>(values, count, stride, sample_at, salt, scratch, indices);
    row_argmax_finish<<<unsigned(rows),256,0,d.stream>>>(scratch, indices, result);
}

// The probability of a drawn token under the softmax it was drawn from: the reciprocal of the sum over the row of
// exp((value - the token's value) * inverse temperature). `sample_at` holds the inverse temperature's float bits in word 0
// (the draw's: see row_sample_chunks). A sum that overflows (a token drawn at a probability below e^-88: none is) or a token
// outside the row gives 0.
__global__ void row_draw_sum_chunks(const float* values, size_t count, const int32_t* chosen, const uint32_t* sample_at, float* sums) {
    const float inverse_temperature = __uint_as_float(sample_at[0]);
    const int32_t token = chosen[0];
    const float best = token >= 0 && size_t(token) < count ? values[token] : 0.0f;
    const size_t per = (count + argmax_chunks - 1) / argmax_chunks, first = blockIdx.x * per, last = std::min(count, first + per);
    float sum = 0.0f;
    for (size_t i = first + threadIdx.x; i < last; i += blockDim.x) sum += expf((values[i] - best) * inverse_temperature);
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = shared[0];
}
__global__ void row_draw_probability_finish(const float* sums, size_t count, const int32_t* chosen, float* result) {
    if (threadIdx.x) return;
    float sum = 0.0f;
    for (unsigned i = 0; i < argmax_chunks; ++i) sum += sums[i];
    const int32_t token = chosen[0];
    result[0] = token >= 0 && size_t(token) < count && isfinite(sum) && sum > 0.0f ? 1.0f / sum : 0.0f;
}
// result[0] = the probability of `chosen[0]`, a token drawn from a row by row_sample, under that row's softmax (see above);
// `scratch` holds argmax_chunks floats.
void row_draw_probability(Device& d, const float* values, size_t count, const int32_t* chosen, const uint32_t* sample_at, float* scratch, float* result) {
    row_draw_sum_chunks<<<argmax_chunks,256,0,d.stream>>>(values, count, chosen, sample_at, scratch);
    row_draw_probability_finish<<<1,32,0,d.stream>>>(scratch, count, chosen, result);
}

// Per row, the chosen logit and the sum of exp(logit - chosen) over the row, as two floats
// at result[2 * row]: a softmax's top probability, also across the halves of a split head.
__global__ void row_exp_sum_chunks(const float* values, size_t count, size_t stride, const int32_t* chosen, float* sums) {
    const float* row = values + size_t(blockIdx.y) * stride;
    const float best = row[chosen[blockIdx.y]];
    const size_t per = (count + argmax_chunks - 1) / argmax_chunks, first = blockIdx.x * per, last = std::min(count, first + per);
    float sum = 0.0f;
    for (size_t i = first + threadIdx.x; i < last; i += blockDim.x) sum += expf(row[i] - best);
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.y * argmax_chunks + blockIdx.x] = shared[0];
}
__global__ void row_exp_sum_finish(const float* values, size_t stride, const int32_t* chosen, const float* sums, float* result) {
    if (threadIdx.x) return;
    float sum = 0.0f;
    for (unsigned i = 0; i < argmax_chunks; ++i) sum += sums[blockIdx.x * argmax_chunks + i];
    result[2 * blockIdx.x] = values[size_t(blockIdx.x) * stride + chosen[blockIdx.x]];
    result[2 * blockIdx.x + 1] = sum;
}
void row_exp_sum(Device& d, const float* values, size_t rows, size_t count, size_t stride, const int32_t* chosen,
                 float* scratch, float* result) {
    row_exp_sum_chunks<<<dim3(argmax_chunks, unsigned(rows)),256,0,d.stream>>>(values, count, stride, chosen, scratch);
    row_exp_sum_finish<<<unsigned(rows),32,0,d.stream>>>(values, stride, chosen, scratch, result);
}
// The `top` greatest of each row of `count` values, the lower index first on a tie (the order of sampling's candidates: a draw
// among them is the draw among the whole row's when its cut keeps at most `top`), as (value, index) pairs most likely
// first: every chunk's `top` (argmax_chunks blocks a row), then one block a row the chunks'. Each round takes the block's
// best not yet taken: a thread's best of the values it holds, then the block's of the threads'.
constexpr unsigned head_top_max = 64, head_top_held = 16;
__device__ __forceinline__ bool top_better(float a, uint32_t ia, float b, uint32_t ib) { return a > b || (a == b && ia < ib); }
// The block's `top` of the values each thread holds (`held` of them, some not set: -inf at index ~0u); thread 0 writes them.
__device__ void top_rounds(float (&values)[head_top_held], uint32_t (&indices)[head_top_held], unsigned top, float* out_values,
                           uint32_t* out_indices) {
    __shared__ float wave_values[32];
    __shared__ uint32_t wave_indices[32];
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32, waves = blockDim.x / 32;
    for (unsigned round = 0; round < top; ++round) {
        float value = -INFINITY; uint32_t index = ~0u;
        #pragma unroll
        for (unsigned i = 0; i < head_top_held; ++i)
            if (top_better(values[i], indices[i], value, index)) { value = values[i]; index = indices[i]; }
        for (unsigned offset = 16; offset; offset /= 2) {
            const float other = __shfl_xor(value, offset, 32); const uint32_t at = __shfl_xor(index, offset, 32);
            if (top_better(other, at, value, index)) { value = other; index = at; }
        }
        if (!lane) { wave_values[wave] = value; wave_indices[wave] = index; }
        __syncthreads();
        value = wave_values[0]; index = wave_indices[0];
        for (unsigned w = 1; w < waves; ++w)
            if (top_better(wave_values[w], wave_indices[w], value, index)) { value = wave_values[w]; index = wave_indices[w]; }
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < head_top_held; ++i)
            if (indices[i] == index) { values[i] = -INFINITY; indices[i] = ~0u; }
        if (!threadIdx.x) { out_values[round] = value; out_indices[round] = index; }
    }
}
__global__ void __launch_bounds__(256) row_top_chunks(const float* values, size_t count, size_t stride, unsigned top,
                                                      float* chunk_values, uint32_t* chunk_indices) {
    const size_t row = blockIdx.y, per = (count + argmax_chunks - 1) / argmax_chunks;
    const size_t first = size_t(blockIdx.x) * per, last = std::min(count, first + per);
    float held[head_top_held]; uint32_t at[head_top_held];
    #pragma unroll
    for (unsigned i = 0; i < head_top_held; ++i) {
        const size_t index = first + threadIdx.x + size_t(i) * blockDim.x;
        const bool in = index < last;
        held[i] = in ? values[row * stride + index] : -INFINITY; at[i] = in ? uint32_t(index) : ~0u;
    }
    const size_t slot = (row * argmax_chunks + blockIdx.x) * top;
    top_rounds(held, at, top, chunk_values + slot, chunk_indices + slot);
}
__global__ void __launch_bounds__(256) row_top_finish(const float* chunk_values, const uint32_t* chunk_indices, unsigned top,
                                                      float* result) {
    const size_t row = blockIdx.x, candidates = size_t(argmax_chunks) * top;
    float held[head_top_held]; uint32_t at[head_top_held];
    #pragma unroll
    for (unsigned i = 0; i < head_top_held; ++i) {
        const size_t index = threadIdx.x + size_t(i) * blockDim.x;
        const bool in = index < candidates;
        held[i] = in ? chunk_values[row * candidates + index] : -INFINITY; at[i] = in ? chunk_indices[row * candidates + index] : ~0u;
    }
    float* out = result + row * 2 * top;
    top_rounds(held, at, top, out, reinterpret_cast<uint32_t*>(out + top));
}
// The two halves' `top` pairs of each row (the peer's rows [0, split), this device's from split on) as one row's.
__global__ void __launch_bounds__(256) head_top_merge(const float* own, const float* peer, uint32_t split, unsigned top, float* result) {
    const size_t row = blockIdx.x;
    const float* low = peer + row * 2 * top; const float* high = own + row * 2 * top;
    float held[head_top_held]; uint32_t at[head_top_held];
    #pragma unroll
    for (unsigned i = 0; i < head_top_held; ++i) {
        const size_t index = threadIdx.x + size_t(i) * blockDim.x;
        held[i] = -INFINITY; at[i] = ~0u;
        if (index < top) { held[i] = low[index]; at[i] = __float_as_uint(low[top + index]); }
        else if (index < 2 * size_t(top)) { held[i] = high[index - top]; at[i] = __float_as_uint(high[index]) + split; }
    }
    float* out = result + row * 2 * top;
    top_rounds(held, at, top, out, reinterpret_cast<uint32_t*>(out + top));
}
// Each row's `top` pairs into result: `top` values, then `top` indices (bits); scratch: rows x argmax_chunks x top x 2 words.
void row_top(Device& d, const float* values, size_t rows, size_t count, size_t stride, unsigned top, float* scratch, float* result) {
    if (!top || top > head_top_max || (count + argmax_chunks - 1) / argmax_chunks > 256 * head_top_held)
        throw std::runtime_error("invalid Straydog row top");
    auto* indices = reinterpret_cast<uint32_t*>(scratch + rows * argmax_chunks * top);
    row_top_chunks<<<dim3(argmax_chunks, unsigned(rows)), 256, 0, d.stream>>>(values, count, stride, top, scratch, indices);
    row_top_finish<<<unsigned(rows), 256, 0, d.stream>>>(scratch, indices, top, result);
}
// The top token's probability from the (logit, sum) pairs of the head, or of both halves.
__global__ void head_confidence(const float* own, const float* other, size_t steps, float* result) {
    const size_t row = threadIdx.x;
    if (row >= steps) return;
    float best = own[2 * row], sum = own[2 * row + 1];
    if (other) {
        const float peer = other[2 * row], top = fmaxf(best, peer);
        sum = sum * expf(best - top) + other[2 * row + 1] * expf(peer - top);
    }
    result[row] = 1.0f / sum;
}

// HyperConnection mix (norm, down, up) and the vocabulary head for up to four rows,
// each with the one-row arithmetic; returns the first maximum of every row and, when
// asked, its probability (drafts stop at an unlikely one).
// The rows come from the host (`input`) or, when that is null, from device memory (`source`).
// With `top` (at most head_top_max), each row's `top` greatest logits and their tokens (row_top) come back in `best`.
constexpr size_t head_top_floats = verify_max * 2 * head_top_max + verify_max * argmax_chunks * head_top_max * 2;
// `source` rows on the device (rather than `input`'s on the host) come back in `rows` when asked, with the tokens.
void output_head(Device& d, void* const* pointers, const int* kinds, size_t steps, size_t vocabulary,
                 const float* input, int32_t* tokens, float* logits, float* confidence, const float* source = nullptr,
                 unsigned top_count = 0, float* best = nullptr, float* rows_out = nullptr) {
    if (!steps || steps > verify_max || (!input && !source) || !tokens || !vocabulary || steps * vocabulary > d.output_capacity ||
        steps * (10240 + 2564) + 2 * verify_max * argmax_chunks + 4 * verify_max + head_top_floats > d.input_capacity ||
        top_count > head_top_max || (top_count && !best))
        throw std::runtime_error("invalid Straydog output head");
    Matrix hc[4];
    hc_operands(pointers, kinds, hc);
    const Matrix head{static_cast<const uint8_t*>(pointers[3]),matrix_row_bytes(kinds[3],2560),vocabulary,2560,0,0,0,kinds[3]};
    for (size_t i = 0; i < 3; ++i) {
        auto found = d.allocations.find(pointers[i]);
        if (found == d.allocations.end() || found->second != hc[i].rows * hc[i].row_bytes) throw std::runtime_error("invalid head operand");
    }
    auto found = d.allocations.find(pointers[3]);
    if (found == d.allocations.end() || found->second != vocabulary * head.row_bytes) throw std::runtime_error("invalid head operand");
    float* rows = d.input; float* mixed = rows + steps * 10240;
    float* partial = mixed + steps * 2564;
    auto* chosen = reinterpret_cast<int32_t*>(partial + 2 * verify_max * argmax_chunks);
    if (input) upload(d, rows, input, steps * 10240 * 4);
    else check(hipMemcpyAsync(rows, source, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    hc_decode(d, hc, false, rows, d.output, mixed, steps, 2564);
    gemv(d, &head, 1, mixed, d.output, steps, 2564, vocabulary);
    row_argmax(d, d.output, steps, vocabulary, vocabulary, partial, chosen);
    float* top = reinterpret_cast<float*>(chosen + verify_max);
    if (confidence) {
        row_exp_sum(d, d.output, steps, vocabulary, vocabulary, chosen, partial, top);
        head_confidence<<<1, 32, 0, d.stream>>>(top, nullptr, steps, top + 2 * verify_max);
    }
    float* ranked = top + 4 * verify_max;
    if (top_count) row_top(d, d.output, steps, vocabulary, vocabulary, top_count, ranked + verify_max * 2 * head_top_max, ranked);
    check(hipGetLastError());
    Piece pieces[5]; size_t count = 0;
    if (rows_out) pieces[count++] = {rows_out, rows, steps * 10240 * 4};
    pieces[count++] = {tokens, chosen, steps * 4};
    if (confidence) pieces[count++] = {confidence, top + 2 * verify_max, steps * 4};
    if (logits) pieces[count++] = {logits, d.output, steps * vocabulary * 4};
    if (top_count) pieces[count++] = {best, ranked, steps * 2 * top_count * 4};
    download_pieces(d, pieces, count);
}

// The vocabulary head split by rows across two linked GPUs: the peer evaluates
// rows [0, split) from the mixed rows this device sends it, this device rows [split, vocabulary).
// Every logit is formed as on one GPU, and the first maximum of the whole row is the lower
// half's on a tie, so tokens (and logits) equal the one-GPU head's.
__global__ void head_part(const float* logits, size_t width, const int32_t* chosen, size_t steps, float* result) {
    const size_t row = threadIdx.x;
    if (row >= steps) return;
    result[2 * row] = logits[row * width + chosen[row]];
    result[2 * row + 1] = __int_as_float(chosen[row]);
}
__global__ void head_choose(const float* logits, size_t width, size_t split, const int32_t* chosen, const float* peer,
                            size_t steps, int32_t* tokens) {
    const size_t row = threadIdx.x;
    if (row >= steps) return;
    const float own = logits[row * width + chosen[row]];
    tokens[row] = peer[2 * row] >= own ? __float_as_int(peer[2 * row + 1]) : int32_t(split) + chosen[row];
}
void output_head_split(Device& d, void* const* pointers, const int* kinds, size_t steps, size_t vocabulary, size_t split,
                       void* peer_head, const float* input, int32_t* tokens, float* logits, float* confidence,
                       const float* source = nullptr, unsigned top_count = 0, float* best_out = nullptr, float* rows_out = nullptr) {
    Device* peer = d.link.peer;
    if (!peer || !steps || steps > verify_max || (!input && !source) || !tokens || !split || split >= vocabulary ||
        steps * vocabulary > d.output_capacity || steps * split > peer->output_capacity ||
        64 + steps * split > d.link.rows * 2560 || 2560 * steps > peer->link.rows * 2560 ||
        steps * (10240 + 2564) + 2 * verify_max * argmax_chunks + 5 * verify_max + 2 * head_top_floats > d.input_capacity ||
        2 * verify_max * argmax_chunks + verify_max + head_top_floats > peer->input_capacity ||
        top_count > head_top_max || (top_count && (!best_out || logits)))
        throw std::runtime_error("invalid Straydog split output head");
    Matrix hc[4];
    hc_operands(pointers, kinds, hc);
    const size_t row_bytes = matrix_row_bytes(kinds[3], 2560);
    for (size_t i = 0; i < 3; ++i) {
        auto found = d.allocations.find(pointers[i]);
        if (found == d.allocations.end() || found->second != hc[i].rows * hc[i].row_bytes) throw std::runtime_error("invalid head operand");
    }
    auto found = d.allocations.find(pointers[3]);
    auto lower = peer->allocations.find(peer_head);
    if (found == d.allocations.end() || found->second != vocabulary * row_bytes ||
        lower == peer->allocations.end() || lower->second != split * row_bytes)
        throw std::runtime_error("invalid head operand");
    float* rows = d.input; float* mixed = rows + steps * 10240;
    float* partial = mixed + steps * 2564;
    auto* chosen = reinterpret_cast<int32_t*>(partial + 2 * verify_max * argmax_chunks);
    int32_t* final_tokens = chosen + verify_max;
    if (input) upload(d, rows, input, steps * 10240 * 4);
    else check(hipMemcpyAsync(rows, source, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    hc_decode(d, hc, false, rows, d.output, mixed, steps, 2564);
    peer_send(d, mixed, steps, 2560, 2564, 0);
    float* received = peer_partials(d.link);
    on_device(peer->ordinal, [&]() {
        Device& h = *peer;
        const Matrix low{static_cast<const uint8_t*>(peer_head), row_bytes, split, 2560, 0, 0, 0, kinds[3]};
        float* scratch = h.input;
        auto* best = reinterpret_cast<int32_t*>(scratch + 2 * verify_max * argmax_chunks);
        peer_receive_rows(h);
        gemv(h, &low, 1, peer_inputs(h.link), h.output, steps, 2560, split);
        row_argmax(h, h.output, steps, split, split, scratch, best);
        head_part<<<1, 32, 0, h.stream>>>(h.output, split, best, steps, peer_remote_partials(h.link));
        // The lower half's (logit, sum) pairs follow its choices, before the logits at 64.
        if (confidence) row_exp_sum(h, h.output, steps, split, split, best, scratch, peer_remote_partials(h.link) + 16);
        if (logits) peer_copy<<<1024, 256, 0, h.stream>>>(peer_remote_partials(h.link) + 64, h.output, steps, split, split);
        // (its rows' tops at 64 instead of its logits: a few pairs a row over the link)
        if (top_count) {
            float* ranked = scratch + 2 * verify_max * argmax_chunks + verify_max;
            row_top(h, h.output, steps, split, split, top_count, ranked + verify_max * 2 * head_top_max, ranked);
            peer_copy<<<1, 256, 0, h.stream>>>(peer_remote_partials(h.link) + 64, ranked, 1, steps * 2 * top_count, 0);
        }
        peer_signal_peer(h);
        check(hipGetLastError());
    });
    Matrix high{static_cast<const uint8_t*>(pointers[3]) + split * row_bytes, row_bytes, vocabulary - split, 2560, 0, 0, 0, kinds[3]};
    gemv(d, &high, 1, mixed, d.output, steps, 2564, vocabulary - split);
    row_argmax(d, d.output, steps, vocabulary - split, vocabulary - split, partial, chosen);
    peer_receive(d);
    head_choose<<<1, 32, 0, d.stream>>>(d.output, vocabulary - split, split, chosen, received, steps, final_tokens);
    float* top = reinterpret_cast<float*>(final_tokens + verify_max);
    if (confidence) {
        row_exp_sum(d, d.output, steps, vocabulary - split, vocabulary - split, chosen, partial, top);
        head_confidence<<<1, 32, 0, d.stream>>>(top, received + 16, steps, top + 2 * verify_max);
    }
    float* ranked = top + 4 * verify_max;
    float* merged = ranked + head_top_floats;
    if (top_count) {
        row_top(d, d.output, steps, vocabulary - split, vocabulary - split, top_count, ranked + verify_max * 2 * head_top_max, ranked);
        head_top_merge<<<unsigned(steps), 256, 0, d.stream>>>(ranked, received + 64, uint32_t(split), top_count, merged);
    }
    check(hipGetLastError());
    finish_with_helper(d);
    Piece pieces[4 + 2 * verify_max]; size_t count = 0;
    if (rows_out) pieces[count++] = {rows_out, rows, steps * 10240 * 4};
    pieces[count++] = {tokens, final_tokens, steps * 4};
    if (confidence) pieces[count++] = {confidence, top + 2 * verify_max, steps * 4};
    if (top_count) pieces[count++] = {best_out, merged, steps * 2 * top_count * 4};
    if (logits)
        for (size_t row = 0; row < steps; ++row) {
            pieces[count++] = {logits + row * vocabulary, received + 64 + row * split, split * 4};
            pieces[count++] = {logits + row * vocabulary + split, d.output + row * (vocabulary - split), (vocabulary - split) * 4};
        }
    download_pieces(d, pieces, count);
}

__global__ void mtp_norm(const float* input, size_t width, const float* weight, float* output) {
    const float* row = input + size_t(blockIdx.x) * width;
    float squares = 0;
    for (size_t i = threadIdx.x; i < width; i += 256) squares += row[i] * row[i];
    __shared__ float sums[256]; sums[threadIdx.x] = squares;
    __syncthreads();
    for (unsigned offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(sums[0] / float(width) + 1e-6f);
    for (size_t i = threadIdx.x; i < width; i += 256) output[size_t(blockIdx.x) * width + i] = row[i] * scale * weight[i];
}
__global__ void mtp_combine(const float* streams, const float* embedding, float* output) {
    const size_t row = blockIdx.y, i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < 10240) output[row * 10240 + i] = streams[row * 10240 + i] + embedding[row * 2560 + i % 2560];
}
// MTP input: RMSNorm(embedding)·fc_embedding added to every stream of
// fc_hidden(RMSNorm over the four-stream hidden), operands (norm_e, fc_e, norm_h, fc_h).
// The prologue's inputs and output live in the device's input area: embeddings (steps x 2560),
// then hidden rows (steps x 10240), which the prologue replaces with its output rows.
float* mtp_prologue_embeddings(Device& d) { return d.input; }
float* mtp_prologue_hidden(Device& d, size_t steps) { return d.input + steps * 2560; }
void mtp_prologue_check(Device& d, void* const* pointers, const int* kinds, size_t steps) {
    if (!steps || steps > 4096 || steps * (2 * 10240 + 2 * 2560) > d.input_capacity || steps * (10240 + 2560) > d.output_capacity)
        throw std::runtime_error("invalid MTP prologue");
    const size_t widths[4] = {2560, 2560, 10240, 2560};
    for (size_t i = 0; i < 4; ++i) {
        auto found = d.allocations.find(pointers[i]);
        const size_t bytes = (i % 2 == 0) ? widths[i] * 4 : 2560 * matrix_row_bytes(kinds[i], 2560);
        if (found == d.allocations.end() || found->second != bytes || (i % 2 == 0 && kinds[i] != 0))
            throw std::runtime_error("invalid MTP operand");
    }
}
// The prologue over the rows already in the input area (see mtp_prologue_hidden).
void mtp_prologue_device(Device& d, void* const* pointers, const int* kinds, size_t steps) {
    mtp_prologue_check(d, pointers, kinds, steps);
    float* e = mtp_prologue_embeddings(d); float* h = mtp_prologue_hidden(d, steps);
    float* ne = h + steps * 10240; float* nh = ne + steps * 2560;
    mtp_norm<<<unsigned(steps),256,0,d.stream>>>(e, 2560, static_cast<const float*>(pointers[0]), ne);
    mtp_norm<<<unsigned(steps),256,0,d.stream>>>(h, 10240, static_cast<const float*>(pointers[2]), nh);
    const Matrix fe{static_cast<const uint8_t*>(pointers[1]),matrix_row_bytes(kinds[1],2560),2560,2560,0,0,0,kinds[1]};
    const Matrix fh{static_cast<const uint8_t*>(pointers[3]),matrix_row_bytes(kinds[3],2560),2560,2560,0,0,0,kinds[3]};
    float* projected = d.output;   // steps x 10240 streams, then steps x 2560 embeddings
    if (steps <= verify_max) {
        gemv(d, &fe, 1, ne, projected + steps * 10240, steps, 2560, 2560);
        for (size_t b = 0; b < steps; ++b) gemv(d, &fh, 1, nh + b * 10240, projected + b * 10240, 4, 2560, 2560);
    } else {   // prompt rows: drafter numerics never change verified tokens
        Matrix descriptors[max_batch]{};
        hc_project(d, fe, steps, ne, projected + steps * 10240, descriptors);
        hc_project(d, fh, steps * 4, nh, projected, descriptors);
    }
    mtp_combine<<<dim3(40, unsigned(steps)),256,0,d.stream>>>(projected, projected + steps * 10240, h);
    check(hipGetLastError());
}
void mtp_prologue(Device& d, void* const* pointers, const int* kinds, size_t steps, const float* embeddings,
                  const float* hidden, float* output) {
    if (!embeddings || !hidden || !output) throw std::runtime_error("invalid MTP prologue");
    mtp_prologue_check(d, pointers, kinds, steps);
    upload(d, mtp_prologue_embeddings(d), embeddings, steps * 2560 * 4);
    upload(d, mtp_prologue_hidden(d, steps), hidden, steps * 10240 * 4);
    mtp_prologue_device(d, pointers, kinds, steps);
    download(d, output, mtp_prologue_hidden(d, steps), steps * 10240 * 4);
}

// One drafting step with the host in none of its stages: the MTP layer's prologue over `embeddings` and the hidden rows,
// the layer itself (`layer`), and when `head` is given the vocabulary head on the layer's last row, whose token and
// probability come back. The hidden rows come from the host, or (`hidden` null, one row) are the last of the `previous`
// rows the layer's last step left in its workspace: a chain of steps reads nothing back until its token.
void draft_step(Device& d, void* const* mtp, const int* mtp_kinds, size_t layer, size_t position, size_t steps,
                const float* embeddings, const float* hidden, size_t previous,
                void* const* head, const int* head_kinds, size_t vocabulary, size_t split, void* peer_head,
                int32_t* token, float* confidence) {
    auto found = d.decoder.find(layer);
    if (!embeddings || found == d.decoder.end() || (!hidden && (steps != 1 || !previous || previous > verify_max)) ||
        (head && (!head_kinds || !token || !confidence)))
        throw std::runtime_error("invalid drafting step");
    mtp_prologue_check(d, mtp, mtp_kinds, steps);
    upload(d, mtp_prologue_embeddings(d), embeddings, steps * 2560 * 4);
    float* rows = mtp_prologue_hidden(d, steps);
    if (hidden) upload(d, rows, hidden, steps * 10240 * 4);
    else check(hipMemcpyAsync(rows, verify_layout(found->second.workspace).hidden + (previous - 1) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, d.stream));
    mtp_prologue_device(d, mtp, mtp_kinds, steps);
    bool parts = false;
    float* output = verify_span_queue(d, layer, layer + 1, position, steps, nullptr, rows, parts);
    if (!head) {
        if (parts) finish_with_helper(d);
        else check(hipStreamSynchronize(d.stream));
        return;
    }
    const float* last = output + (steps - 1) * 10240;
    int32_t token_row[verify_max];
    float confidence_row[verify_max];
    if (split) output_head_split(d, head, head_kinds, 1, vocabulary, split, peer_head, nullptr, token_row, nullptr, confidence_row, last);
    else {
        output_head(d, head, head_kinds, 1, vocabulary, nullptr, token_row, nullptr, confidence_row, last);
        if (parts) finish_with_helper(d);
    }
    *token = token_row[0]; *confidence = confidence_row[0];
}

// The PLE layer after a verification's first layer, with no wait between them: `layer`'s verified rows (still in its
// workspace; see verify_span_queue) are copied to `kept` for a replay, the PLE state to `snapshot` for one, and the layer
// runs on them in place. Both the PLE layer's rows and the first layer's are read back with one wait.
float* verified_rows(Device& d, size_t layer, size_t steps) {
    auto found = d.decoder.find(layer);
    if (found == d.decoder.end() || !steps || steps > verify_max || found->second.verified != steps)
        throw std::runtime_error("PLE layer has no verified rows");
    return verify_layout(found->second.workspace).hidden;
}
void ple_state_check(Device& d, void* history, void* snapshot, void* kept) {
    auto state = d.allocations.find(history), copy = d.allocations.find(snapshot), rows = d.allocations.find(kept);
    if (state == d.allocations.end() || copy == d.allocations.end() || rows == d.allocations.end() ||
        state->second != 10 * 10240 * 4 || copy->second != state->second || rows->second < verify_max * 10240 * 4)
        throw std::runtime_error("invalid PLE state");
}
void ple_front(Device& d, void* const* pointers, const int* kinds, void* history, void* snapshot, void* kept, size_t layer,
               size_t steps, const float* embeddings, float* first, float* hidden) {
    if (!first != !hidden) throw std::runtime_error("invalid PLE rows");
    float* rows = verified_rows(d, layer, steps);
    ple_state_check(d, history, snapshot, kept);
    check(hipMemcpyAsync(snapshot, history, 10 * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    check(hipMemcpyAsync(kept, rows, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    ple(d, pointers, kinds, history, steps, embeddings, nullptr, nullptr, rows, true, false);
    // (no rows asked for: the layers after it continue from the workspace, nothing waited for)
    if (!first) return;
    const Piece pieces[2] = {{hidden, rows, steps * 10240 * 4}, {first, kept, steps * 10240 * 4}};
    download_pieces(d, pieces, 2);
}
// A partial acceptance: the PLE state back to the snapshot, then the layer over the first `steps` rows kept by ple_front
// (their output is not wanted, only the state). Queued, not waited for: the stream orders it before what follows.
void ple_replay(Device& d, void* const* pointers, const int* kinds, void* history, void* snapshot, void* kept, size_t steps,
                const float* embeddings) {
    ple_state_check(d, history, snapshot, kept);
    check(hipMemcpyAsync(history, snapshot, 10 * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    ple(d, pointers, kinds, history, steps, embeddings, nullptr, nullptr, static_cast<float*>(kept), true, false);
}
