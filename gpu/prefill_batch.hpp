// Logical-batch prefill for fully resident layers. Dense projections run once over
// every row of the span's batch, so each packed weight is expanded once per layer;
// causal convolution, recurrence, selection and attention still advance in bounded
// physical chunks. The idle MoE workspace of the device provides the row scratch.

// Contiguous-row projection. In FP16 mode `half` must already hold the inputs.
void batch_project(Device& d, const Matrix& m, size_t steps, const float* input, const __half* half, float* output) {
    if (d.blas && steps >= d.gemm_min_batch && m.rows * m.cols <= gemm_elements) {
        const float alpha = 1, beta = 0;
        if (d.half_gemm) {
            expand(m, d.expanded, d.stream, true);
            blas_check(hipblasGemmEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, int(m.rows), int(steps), int(m.cols), &alpha,
                d.expanded, HIP_R_16F, int(m.cols), half, HIP_R_16F, int(m.cols), &beta,
                output, HIP_R_32F, int(m.rows), HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        } else {
            const float* weights = reinterpret_cast<const float*>(m.data);
            if (m.kind != 0) { expand(m, d.expanded, d.stream, false); weights = d.expanded; }
            blas_check(hipblasSgemm(d.blas_f32, HIPBLAS_OP_T, HIPBLAS_OP_N, int(m.rows), int(steps), int(m.cols), &alpha,
                weights, int(m.cols), input, int(m.cols), &beta, output, int(m.rows)));
        }
        ++d.gemm_calls; d.gemm_vectors += steps;
        return;
    }
    Matrix tokens[gemv_fixed_items];
    for (size_t first = 0; first < steps; first += gemv_fixed_items) {
        const size_t count = std::min(gemv_fixed_items, steps - first);
        for (size_t i = 0; i < count; ++i) {
            tokens[i] = m;
            tokens[i].input_offset = (first + i) * m.cols;
            tokens[i].output_offset = (first + i) * m.rows;
        }
        gemv(d, tokens, count, input, output);
    }
}
bool batch_half(const Device& d, size_t steps) { return d.blas && d.half_gemm && steps >= d.gemm_min_batch; }
void batch_cast(Device& d, const float* input, __half* half, size_t count) {
    cast_half<<<unsigned((count + 255) / 256), 256, 0, d.stream>>>(input, half, count);
    check(hipGetLastError());
}
Matrix batch_matrix(void* const* pointers, const int* kinds, size_t index, size_t rows, size_t cols) {
    return {static_cast<const uint8_t*>(pointers[index]), matrix_row_bytes(kinds[index], cols), rows, cols, 0, 0, 0, kinds[index]};
}

// Scratch: normalized 10240 + half 5120 + low 320 + injection 4 + up 10240 floats per row.
constexpr size_t hc_batch_floats = 10240 + 5120 + 324 + 10240;
// Where hc_prefill's FP16 path keeps a batch's halves in its scratch; the row streams' denominators are at its start, where the
// F32 path's normalized floats are.
inline __half* hc_scratch_half(float* scratch, size_t steps) { return reinterpret_cast<__half*>(scratch + steps * 10240); }
// `prenormalized`: the halves and denominators are in the scratch already (a scatter's residual update normalized the rows,
// qwen_scatter_residual_normalize); with FP16 projections only.
void hc_prefill(Device& d, void* const* pointers, const int* kinds, size_t steps, const float* hidden, float* mixed, float* scratch,
                bool prenormalized = false) {
    const Matrix down = batch_matrix(pointers, kinds, 1, 320, 10240), up = batch_matrix(pointers, kinds, 2, 10240, 320);
    const Matrix inject = batch_matrix(pointers, kinds, 3, 4, 10240);
    float* normalized = scratch;
    __half* half = hc_scratch_half(scratch, steps);
    float* low = normalized + steps * (10240 + 5120);
    float* injection = low + steps * 320;
    float* projected = injection + steps * 4;
    const bool cast = batch_half(d, steps);
    if (cast) {
        // (the values as halves only, for the projections; the denominators where the floats would have been)
        const float* norm = static_cast<const float*>(pointers[0]);
        float* denominators = normalized;
        if (!prenormalized) hc_normalize_half<<<unsigned(steps * 4), 256, 0, d.stream>>>(hidden, norm, half, denominators);
        batch_project(d, down, steps, nullptr, half, low);
        batch_project(d, inject, steps, nullptr, half, injection);
        hc_activate_low<<<unsigned((steps * 320 + 255) / 256), 256, 0, d.stream>>>(low, steps * 320);
        batch_cast(d, low, half, steps * 320);
        batch_project(d, up, steps, low, half, projected);
        hc_finish_hidden<<<unsigned((steps * 2560 + 255) / 256), 256, 0, d.stream>>>(hidden, norm, denominators, projected, injection,
                                                                                    steps, mixed);
        check(hipGetLastError());
        return;
    }
    hc_normalize<<<unsigned(steps * 4), 256, 0, d.stream>>>(hidden, static_cast<const float*>(pointers[0]), normalized);
    if (cast) batch_cast(d, normalized, half, steps * 10240);
    batch_project(d, down, steps, normalized, half, low);
    batch_project(d, inject, steps, normalized, half, injection);
    hc_activate_low<<<unsigned((steps * 320 + 255) / 256), 256, 0, d.stream>>>(low, steps * 320);
    if (cast) batch_cast(d, low, half, steps * 320);
    batch_project(d, up, steps, low, half, projected);
    // The finish writes the decoder's mixed layout: rows of 2560, then four injections per row.
    hc_finish<<<unsigned((steps * 2560 + 255) / 256), 256, 0, d.stream>>>(normalized, projected, injection, steps, true, mixed);
    check(hipGetLastError());
}

// Scratch: qkv 10240 + gate 6144 + beta 48 + alpha 48 + post 6144 + half 3072 floats per row (a part's, QwenHeads<P>, fewer).
constexpr size_t gdn_batch_floats = 10240 + 6144 + 96 + 6144 + 3072;
template<int P = 1>
void gdn_prefill(Device& d, void* const* pointers, const int* kinds, void* conv, void* recurrent, size_t steps,
                 size_t chunk, const float* mixed, float* branch, float* scratch) {
    using H = QwenHeads<P>;
    const Matrix qkv_weights = batch_matrix(pointers, kinds, 0, H::channels, 2560), gate_weights = batch_matrix(pointers, kinds, 1, H::gate, 2560);
    const Matrix beta_weights = batch_matrix(pointers, kinds, 2, H::value, 2560), alpha_weights = batch_matrix(pointers, kinds, 3, H::value, 2560);
    const Matrix output_weights = batch_matrix(pointers, kinds, 4, 2560, H::gate);
    float* qkv = scratch;
    float* gate = qkv + steps * H::channels;
    float* beta = gate + steps * H::gate;
    float* alpha = beta + steps * H::value;
    float* post = alpha + steps * H::value;
    auto* half = reinterpret_cast<__half*>(post + steps * H::gate);
    const bool cast = batch_half(d, steps);
    if (cast) batch_cast(d, mixed, half, steps * 2560);
    batch_project(d, qkv_weights, steps, mixed, half, qkv);
    batch_project(d, gate_weights, steps, mixed, half, gate);
    batch_project(d, beta_weights, steps, mixed, half, beta);
    batch_project(d, alpha_weights, steps, mixed, half, alpha);
    for (size_t first = 0; first < steps; first += chunk) {
        const size_t count = std::min(chunk, steps - first);
        float* rows = qkv + first * H::channels;
#if defined(__HIPCC__)
        // A long read's chunk goes through the chunked scan (gated_delta_chunk.hpp): its first kernel convolves, normalizes and packs, its
        // second scans and moves the convolution's history on.
        if (d.gdn_chunked && count > 1 && GdnChunkLayout<P>(count).total <= d.input_capacity) {
            gdn_chunk_prep<P><<<dim3(unsigned((count + gdn_chunk_tokens - 1) / gdn_chunk_tokens), unsigned(H::key)), 640, 0, d.stream>>>(rows,
                static_cast<const float*>(conv), static_cast<const float*>(pointers[5]), beta + first * H::value, alpha + first * H::value,
                static_cast<const float*>(pointers[6]), static_cast<const float*>(pointers[7]), count, d.input);
            gdn_chunk_scan<P><<<unsigned(H::key * 8), 192, 0, d.stream>>>(static_cast<float*>(recurrent), count, d.input, rows,
                static_cast<float*>(conv));
            gdn_post<<<unsigned(count * H::value), 128, 0, d.stream>>>(rows, gate + first * H::gate, static_cast<const float*>(pointers[8]),
                post + first * H::gate);
            continue;
        }
#endif
        // (the runs' edges in the packed values' scratch, d.input, before the chunk's values are packed there)
        const unsigned columns = unsigned((H::channels + 255) / 256), runs = unsigned((count + gdn_conv_run - 1) / gdn_conv_run);
        gdn_conv_edges<P><<<columns, 256, 0, d.stream>>>(rows, static_cast<float*>(conv), count, d.input);
        gdn_conv_runs<P><<<dim3(columns, runs), 256, 0, d.stream>>>(rows, d.input, static_cast<const float*>(pointers[5]), count);
        gdn_normalize_qk<P><<<unsigned(count * H::key), 128, 0, d.stream>>>(rows, count, d.input);
        gdn_pack_values<P><<<unsigned((count * H::gate + 255) / 256), 256, 0, d.stream>>>(rows, beta + first * H::value,
            alpha + first * H::value, static_cast<const float*>(pointers[6]), static_cast<const float*>(pointers[7]), count, d.input);
        // Every qkv row of this chunk is packed before the recurrence overwrites the chunk's rows.
        if (d.rdna4 && count > 1)
            delta_split_sequence<P><<<delta_split_blocks<P>(), delta_columns * 8, 0, d.stream>>>(static_cast<float*>(recurrent), count, d.input, rows);
        else delta_register_sequence<<<unsigned(H::value), 128, 0, d.stream>>>(static_cast<float*>(recurrent), H::value, count, d.input, rows);
        gdn_post<<<unsigned(count * H::value), 128, 0, d.stream>>>(rows, gate + first * H::gate, static_cast<const float*>(pointers[8]),
            post + first * H::gate);
    }
    if (cast) batch_cast(d, post, half, steps * H::gate);
    batch_project(d, output_weights, steps, post, half, branch);
    check(hipGetLastError());
}

// qsa_attention for a batch whose host K/V is past the stage: tokens [0, steps) of the selections in d.input
// attend over their blocks' union copied into the stage (qsa_gather_union), as many tokens at a time as
// the stage holds the union of. `places` and `blocks`: scratch of (base + steps + 3) / 4 + 1 words each.
template<int P = 1>
void qsa_attention_gathered(Device& d, void* state, size_t context, size_t base, size_t steps, size_t selected, float* output,
                            uint32_t* places, uint32_t* blocks) {
    using H = QwenHeads<P>;
    const size_t end = (base + steps - 1) / 4 + 1, planes = kv_planes_of(d, state), capacity = kv_stage_capacity(d, state);
    kv_stage_overwritten(d);
    for (size_t first = 0; first < steps;) {
        size_t count = steps - first, total = 0;
        for (;;) {
            check(hipMemsetAsync(places, 0, end * 4, d.stream));
            qsa_mark_selection<P><<<unsigned(count), 256, 0, d.stream>>>(d.input, steps, selected, first, places);
            qsa_place_union<<<1, 1024, 0, d.stream>>>(places, end, blocks);
            total = kv_union_count(d, places + end);
            if (total * 4 <= capacity || count == 1) break;
            count = (count + 1) / 2;
        }
        if (total * 4 <= capacity) {
            qsa_gather_union<<<unsigned(total), 256, 0, d.stream>>>(d.kv_row, static_cast<const kv_t*>(state), context, planes, blocks,
                                                                     kv_stage_rows(d), capacity);
            qsa_renumber_selection<P><<<unsigned(count), 256, 0, d.stream>>>(d.input, steps, selected, first, places);
            qsa_flash<P><<<dim3(unsigned(count), unsigned(H::kv)), 256, 0, d.stream>>>(d.kv_row, kv_stage_rows(d), capacity, base, steps, selected,
                                                                                     d.input, output, first);
        } else   // (a stage too small for one token's selection)
            qsa_flash<P><<<dim3(1, unsigned(H::kv)), 256, 0, d.stream>>>(d.kv_row, static_cast<const kv_t*>(state), context, base, steps, selected,
                                                                       d.input, output, first);
        first += count;
    }
}

// This device's flag in the other's raised and the other's waited for (qwen_parts.hpp: device_link.hpp comes after this file).
void qsa_share_signal(Device& d);

// Scratch: q/gate 12288 + k/v/index-q 1536 + index-k 128 + attended 6144 + half 3072 floats per row (a part's fewer).
constexpr size_t qsa_batch_floats = 12288 + 1536 + 128 + 6144 + 3072;
// `part` (a part's prefill, QwenHeads<2>): which of the two devices this is, when the pair share the indexer (qsa_mark_send): each
// span of more than one chunk has its chunks scored and selected by one device in turn, and the selected blocks go to the other
// through the sum link's inbox, so each makes the whole span's marks. The other's selections are waited for by the device's stream,
// after its own chunks (a part's host must not wait inside a layer). From qsa_prompt_from (per_token) there are no marks: a device attends
// its own chunks in the loop, from the selection it made (and sends its blocks), keeps the other chunks' normalized queries in the place of
// their attended values, and after the wait attends those from the blocks it received (qsa_prompt_received).
template<int P = 1>
void qsa_prefill(Device& d, void* const* pointers, const int* kinds, void* kv_state, void* index_state, void* scores,
                 size_t context, size_t base, size_t steps, const float* mixed, float* branch, float* scratch, int part = -1) {
    using H = QwenHeads<P>;
    constexpr size_t Q = H::query * 256, K = H::kv * 256;   // a row's attended values (and query rows), its keys
    const size_t rows[6] = {2 * Q, K, K, 512, 128, 2560}, cols[6] = {2560, 2560, 2560, 2560, 2560, Q};
    Matrix matrices[6];
    for (size_t i = 0; i < 6; ++i) matrices[i] = batch_matrix(pointers, kinds, i, rows[i], cols[i]);
    float* qgate = scratch;
    float* key = qgate + steps * 2 * Q;
    float* value = key + steps * K;
    float* iq = value + steps * K;
    float* ik = iq + steps * 512;
    float* attended = ik + steps * 128;
    auto* half = reinterpret_cast<__half*>(attended + steps * Q);
    float* targets[5] = {qgate, key, value, iq, ik};
    const bool cast = batch_half(d, steps);
    if (cast) batch_cast(d, mixed, half, steps * 2560);
    for (size_t i = 0; i < 5; ++i) batch_project(d, matrices[i], steps, mixed, half, targets[i]);
    const size_t pool = context / 4 + 1;
    float* blocks = static_cast<float*>(index_state);
    float* history = blocks + pool * 128;
    // In FP16 mode RDNA4 attends each span of up to 2048 rows at once (qsa_flash_wmma), with the
    // span's queries in FP16 where the projections' FP16 inputs were and the block marks and
    // unions in the decode attention scratch.
    const bool batched = d.rdna4 && d.half_gemm && d.attention_scores && steps > 1;
    // (both devices must take the same exchanges: the decision is the pair's, from what both hold)
    const Device* peer = d.sums.peer;
    const bool sharing = P == 2 && part >= 0 && batched && peer && peer->rdna4 && peer->half_gemm && peer->attention_scores &&
                         context <= 262144 && d.sums.capacity % 8 == 0 && d.sums.capacity == peer->sums.capacity;
    for (size_t span = 0; span < steps; span += 2048) {
        const size_t span_rows = std::min<size_t>(2048, steps - span), groups = (span_rows + qsa_group - 1) / qsa_group;
        // (a row's blocks as 16 bits in a slot of the inbox: a thousand bytes against the 2560 of a row of the sums)
        const size_t chunks = (span_rows + 255) / 256;
        const bool shared = sharing && chunks > 1 && span_rows * qsa_share_blocks * 2 <= d.sums.capacity * 4;
        const size_t stride = (base + span + span_rows) / 4 + 1;
        auto* marks = reinterpret_cast<uint32_t*>(d.attention_scores);
        uint32_t* union_blocks = marks + groups * stride;
        uint32_t* union_masks = union_blocks + groups * qsa_group_capacity;
        uint32_t* union_counts = union_masks + groups * qsa_group_capacity;
        const KvView kv = kv_for_prefill(d, kv_state, context, base + span, span_rows);
        // (the union of a gathered batch: its blocks' places and the blocks in place order, after the marks. The host waits for the
        // union's size, which a part must not: its exchanges wait for the other part's layer, queued after this one returns.)
        const bool gathered = P == 1 && kv_gathers(d, kv, kv_state, span_rows);
        // (past qsa_prompt_from each token attends its own selection (qsa_prompt_attention): no marks, no unions)
        const bool per_token = batched && !gathered && (kv.staged || !d.host_states.count(kv_state)) && base + span >= qsa_prompt_from;
        uint32_t* places = batched ? union_counts + groups : marks;
        uint32_t* placed = places + stride + 1;
        if ((batched ? groups * (stride + 2 * qsa_group_capacity) + groups : 0) + (gathered ? 2 * (stride + 1) : 0) > attention_score_bytes / 4)
            throw std::runtime_error("QSA prefill marks exceed the attention scratch");
        if (batched && !per_token) check(hipMemsetAsync(marks, 0, groups * stride * 4, d.stream));
        for (size_t first = span; first < span + span_rows; first += 256) {
            const size_t count = std::min<size_t>(256, span + span_rows - first), position = base + first;
            const size_t selected = std::min(position + count, size_t(2052));
            // (a chunk the other device selects: its normalized index queries and the selection are not this device's to make)
            const bool own = !shared || (first - span) / 256 % 2 == size_t(part);
            float* chunk_q = qgate + first * 2 * Q;
            float* chunk_iq = iq + first * 512;
            float* chunk_ik = ik + first * 128;
            // (a batch of a sequence with images has its rope positions from the table: sd_rope_rows)
            const uint32_t* axes = d.rope_count ? d.rope_rows + (position - d.rope_first) * 3 : nullptr;
            // (a chunk the other device selects, past qsa_prompt_from, keeps its normalized queries in its attended values' place until the
            // other's blocks have come: write_attention_kv reads the keys and values only)
            float* queries = per_token && !own ? attended + first * Q : d.input;
            if (axes) qsa_normalize_axes<<<unsigned(count * H::query), 256, 0, d.stream>>>(chunk_q, static_cast<const float*>(pointers[6]), H::query, 256,
                512, axes, queries);
            else qsa_normalize<<<unsigned(count * H::query), 256, 0, d.stream>>>(chunk_q, static_cast<const float*>(pointers[6]), H::query, 256,
                512, position, queries);
            if (axes) qsa_normalize_axes<<<unsigned(count * H::kv), 256, 0, d.stream>>>(key + first * K, static_cast<const float*>(pointers[7]), H::kv,
                256, 256, axes, d.input + count * Q);
            else qsa_normalize<<<unsigned(count * H::kv), 256, 0, d.stream>>>(key + first * K, static_cast<const float*>(pointers[7]), H::kv,
                256, 256, position, d.input + count * Q);
            check(hipMemcpyAsync(d.input + count * (Q + K), value + first * K, count * K * 4, hipMemcpyDeviceToDevice, d.stream));
            if (own && axes) qsa_normalize_axes<<<unsigned(count * 4), 128, 0, d.stream>>>(chunk_iq, static_cast<const float*>(pointers[8]), 4, 128, 128, axes, chunk_iq);
            else if (own) qsa_normalize<<<unsigned(count * 4), 128, 0, d.stream>>>(chunk_iq, static_cast<const float*>(pointers[8]), 4, 128, 128, position, chunk_iq);
            const size_t completed = (position + count) / 4 - position / 4;
            if (completed && axes)
                qsa_pool_axes<<<unsigned(completed), 128, 0, d.stream>>>(chunk_ik, history, static_cast<const float*>(pointers[9]), pool, position,
                    d.rope_rows, d.rope_first, blocks);
            else if (completed)
                qsa_pool<<<unsigned(completed), 128, 0, d.stream>>>(chunk_ik, history, static_cast<const float*>(pointers[9]), pool, position, blocks);
            qsa_save_history<<<unsigned((std::min(count, size_t(4)) * 128 + 255) / 256), 256, 0, d.stream>>>(chunk_ik, position, count, history);
            if (own) qsa_index_chunk<P>(d, chunk_iq, blocks, pool, position, count, selected, static_cast<float*>(scores), d.input);
            write_attention_kv<P><<<unsigned((count * K + 255) / 256), 256, 0, d.stream>>>(d.kv_row, kv.kv, kv.capacity, position, count, d.input,
                nullptr, nullptr, kv.mirror, kv.mirror_rows);
            if (per_token) {
                if (own) {
                    if (shared)
                        qsa_send_blocks<P><<<unsigned(count), 128, 0, d.stream>>>(d.input, count, selected, first - span,
                            reinterpret_cast<uint16_t*>(d.sums.peer_inbox), d.sums.state, d.sums.capacity * 2);
                    qsa_prompt_chunk<P>(d, kv.kv, kv.capacity, position, count, selected, d.input, attended + first * Q);
                }
            } else if (batched) {
                cast_half<<<unsigned((count * Q + 255) / 256), 256, 0, d.stream>>>(d.input, half + first * Q, count * Q);
                if (!shared)
                    qsa_mark_blocks<P><<<unsigned(count), 256, 0, d.stream>>>(d.input, count, selected, first - span, stride, marks);
                else if (own)
                    qsa_mark_send<P><<<unsigned(count), 128, 0, d.stream>>>(d.input, count, selected, position, first - span, stride, marks,
                        reinterpret_cast<uint16_t*>(d.sums.peer_inbox), d.sums.state, d.sums.capacity * 2);
            } else if (gathered) qsa_attention_gathered<P>(d, kv_state, context, position, count, selected, attended + first * Q, places, placed);
            else qsa_attention<P>(d, kv.kv, kv.capacity, position, count, selected, d.input, attended + first * Q, nullptr, kv.mirror,
                                  kv.mirror_rows);
        }
        if (shared) {
            // The other's chunks' selections: its stores are done before its flag, which this device's stream waits for.
            qsa_share_signal(d);
            const size_t theirs = size_t(1 - part), turns = chunks > theirs ? (chunks - theirs + 1) / 2 : 0;
            if (per_token)
                qsa_prompt_received<P>(d, kv.kv, kv.capacity, base + span, span_rows, theirs, turns, attended + span * Q,
                    reinterpret_cast<const uint16_t*>(d.sums.inbox), d.sums.state, d.sums.capacity * 2);
            else
                qsa_mark_received<<<dim3(256, unsigned(turns)), 128, 0, d.stream>>>(base + span, theirs, span_rows, stride, marks,
                    reinterpret_cast<const uint16_t*>(d.sums.inbox), d.sums.state, d.sums.capacity * 2);
        }
#if defined(__HIPCC__)
        if (batched && !per_token) {
            qsa_union_blocks<<<unsigned(groups), 256, 0, d.stream>>>(marks, stride, base + span, span_rows, union_blocks, union_masks, union_counts);
            if (gathered) {
                // As many groups at a time as the stage holds the union of (qsa_attention_gathered).
                const size_t end = (base + span + span_rows - 1) / 4 + 1, capacity = kv_stage_capacity(d, kv_state);
                kv_stage_overwritten(d);
                for (size_t first = 0; first < groups;) {
                    size_t count = groups - first, total = 0;
                    for (;;) {
                        qsa_union_of_groups<<<unsigned((end + 255) / 256), 256, 0, d.stream>>>(marks, stride, first, first + count, end, places);
                        qsa_place_union<<<1, 1024, 0, d.stream>>>(places, end, placed);
                        total = kv_union_count(d, places + end);
                        if (total * 4 <= capacity || count == 1) break;
                        count = (count + 1) / 2;
                    }
                    const bool fits = total * 4 <= capacity;
                    if (fits)
                        qsa_gather_union<<<unsigned(total), 256, 0, d.stream>>>(d.kv_row, static_cast<const kv_t*>(kv_state), context,
                                                                                 kv_planes_of(d, kv_state), placed, kv_stage_rows(d), capacity);
                    qsa_flash_wmma<P><<<dim3(unsigned(count), unsigned(H::kv)), unsigned(qsa_group * 48), 0, d.stream>>>(
                        d.kv_row, fits ? kv_stage_rows(d) : kv.kv, fits ? capacity : kv.capacity, base + span, span_rows, half + span * Q, union_blocks,
                        union_masks, union_counts, attended + span * Q, fits ? places : nullptr, unsigned(first));
                    first += count;
                }
            } else
                qsa_flash_wmma<P><<<dim3(unsigned(groups), unsigned(H::kv)), unsigned(qsa_group * 48), 0, d.stream>>>(d.kv_row, kv.kv, kv.capacity,
                    base + span, span_rows, half + span * Q, union_blocks, union_masks, union_counts, attended + span * Q);
        }
#endif
        kv_after_prefill(d, kv, kv_state, context, base + span, span_rows);
    }
    qsa_gate<<<unsigned((steps * Q + 255) / 256), 256, 0, d.stream>>>(attended, qgate, steps * Q);
    if (cast) batch_cast(d, attended, half, steps * Q);
    batch_project(d, matrices[5], steps, attended, half, branch);
    check(hipGetLastError());
}
