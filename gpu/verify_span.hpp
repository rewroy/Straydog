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
    gdn_tokens<<<48,gdn_threads,0,d.stream>>>(replay ? plan.snapshot + 4 * 10240 : recurrent, replay ? plan.snapshot : conv,
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
        for (size_t b = 0; b < steps; ++b)
            qsa_decode_attend(d, pointers + 4, plan.states[0], plan.states[1], plan.states[2], plan.context,
                              v.positions + b, v.qsa + b * verify_qsa, v.attended + b * 6144);
        gemv(d, &m[5], 1, v.attended, nullptr, steps, 6144, 0, residual);
    }
    hc_operands(pointers + 4 + attention, kinds + 4 + attention, hc);
    hc_decode(d, hc, true, v.hidden, d.output, v.mixed, steps, 2564);
    moe_tokens(d, plan.moe, steps, v.mixed, 2564, v.hidden);
}

// input==nullptr continues from rows already in the span's workspace; output may be nullptr.
void verify_span(Device& d, size_t begin, size_t end, size_t position, size_t steps, const float* input, float* output) {
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
    uint32_t positions[verify_max];
    for (size_t b = 0; b < steps; ++b) positions[b] = uint32_t(position + b);
    upload(d, v.positions, positions, steps * 4);
    auto submit = [&]() {
        for (size_t layer = begin; layer < end; ++layer) verify_layer(d, d.decoder.find(layer)->second, steps, v);
        check(hipGetLastError());
    };
    if (span_has_parts(d, begin, end)) {
        run_with_helper(d, begin, end, steps, [&]() {
            run_graph(d, "verify-span", dependencies.data(), dependencies.size(), (begin * 64 + end) * 8 + steps, submit);
        });
        finish_with_helper(d);
    } else run_graph(d, "verify-span", dependencies.data(), dependencies.size(), (begin * 64 + end) * 8 + steps, submit);
    if (output) download(d, output, v.hidden, steps * 10240 * 4);
    else check(hipStreamSynchronize(d.stream));
    for (size_t layer = begin; layer < end; ++layer) d.decoder.find(layer)->second.verified = steps;
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
        for (size_t layer = begin; layer < end; ++layer) {
            if (!decoder_linear(layer)) continue;
            gdn_verify(d, d.decoder.find(layer)->second, kept, true, d.output);
        }
        check(hipGetLastError());
        check(hipStreamSynchronize(d.stream));
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
void output_head(Device& d, void* const* pointers, const int* kinds, size_t steps, size_t vocabulary,
                 const float* input, int32_t* tokens, float* logits, float* confidence) {
    if (!steps || steps > verify_max || !input || !tokens || !vocabulary || steps * vocabulary > d.output_capacity ||
        steps * (10240 + 2564) + 2 * verify_max * argmax_chunks + 4 * verify_max > d.input_capacity)
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
    upload(d, rows, input, steps * 10240 * 4);
    hc_decode(d, hc, false, rows, d.output, mixed, steps, 2564);
    gemv(d, &head, 1, mixed, d.output, steps, 2564, vocabulary);
    row_argmax(d, d.output, steps, vocabulary, vocabulary, partial, chosen);
    float* top = reinterpret_cast<float*>(chosen + verify_max);
    if (confidence) {
        row_exp_sum(d, d.output, steps, vocabulary, vocabulary, chosen, partial, top);
        head_confidence<<<1, 32, 0, d.stream>>>(top, nullptr, steps, top + 2 * verify_max);
    }
    check(hipGetLastError());
    download(d, tokens, chosen, steps * 4);
    if (confidence) download(d, confidence, top + 2 * verify_max, steps * 4);
    if (logits) download(d, logits, d.output, steps * vocabulary * 4);
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
                       void* peer_head, const float* input, int32_t* tokens, float* logits, float* confidence) {
    Device* peer = d.link.peer;
    if (!peer || !steps || steps > verify_max || !input || !tokens || !split || split >= vocabulary ||
        steps * vocabulary > d.output_capacity || steps * split > peer->output_capacity ||
        64 + steps * split > d.link.rows * 2560 || 2560 * steps > peer->link.rows * 2560 ||
        steps * (10240 + 2564) + 2 * verify_max * argmax_chunks + 5 * verify_max > d.input_capacity ||
        2 * verify_max * argmax_chunks + verify_max > peer->input_capacity)
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
    upload(d, rows, input, steps * 10240 * 4);
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
    check(hipGetLastError());
    finish_with_helper(d);
    download(d, tokens, final_tokens, steps * 4);
    if (confidence) download(d, confidence, top + 2 * verify_max, steps * 4);
    if (logits)
        for (size_t row = 0; row < steps; ++row) {
            download(d, logits + row * vocabulary, received + 64 + row * split, split * 4);
            download(d, logits + row * vocabulary + split, d.output + row * (vocabulary - split), (vocabulary - split) * 4);
        }
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
