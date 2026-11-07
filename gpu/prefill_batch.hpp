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

// Scratch: q/gate 12288 + k/v/index-q 1536 + index-k 128 + attended 6144 + half 3072 floats per row (a part's fewer).
constexpr size_t qsa_batch_floats = 12288 + 1536 + 128 + 6144 + 3072;
template<int P = 1>
void qsa_prefill(Device& d, void* const* pointers, const int* kinds, void* kv_state, void* index_state, void* scores,
                 size_t context, size_t base, size_t steps, const float* mixed, float* branch, float* scratch) {
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
    for (size_t span = 0; span < steps; span += 2048) {
        const size_t span_rows = std::min<size_t>(2048, steps - span), groups = (span_rows + qsa_group - 1) / qsa_group;
        const size_t stride = (base + span + span_rows) / 4 + 1;
        auto* marks = reinterpret_cast<uint32_t*>(d.attention_scores);
        uint32_t* union_blocks = marks + groups * stride;
        uint32_t* union_masks = union_blocks + groups * qsa_group_capacity;
        uint32_t* union_counts = union_masks + groups * qsa_group_capacity;
        if (batched) {
            if ((groups * (stride + 2 * qsa_group_capacity) + groups) * 4 > attention_score_bytes)
                throw std::runtime_error("QSA prefill marks exceed the attention scratch");
            check(hipMemsetAsync(marks, 0, groups * stride * 4, d.stream));
        }
        const KvView kv = kv_for_prefill(d, kv_state, context, base + span, span_rows);
        for (size_t first = span; first < span + span_rows; first += 256) {
            const size_t count = std::min<size_t>(256, span + span_rows - first), position = base + first;
            const size_t selected = std::min(position + count, size_t(2052));
            float* chunk_q = qgate + first * 2 * Q;
            float* chunk_iq = iq + first * 512;
            float* chunk_ik = ik + first * 128;
            // (a batch of a sequence with images has its rope positions from the table: sd_rope_rows)
            const uint32_t* axes = d.rope_count ? d.rope_rows + (position - d.rope_first) * 3 : nullptr;
            if (axes) qsa_normalize_axes<<<unsigned(count * H::query), 256, 0, d.stream>>>(chunk_q, static_cast<const float*>(pointers[6]), H::query, 256,
                512, axes, d.input);
            else qsa_normalize<<<unsigned(count * H::query), 256, 0, d.stream>>>(chunk_q, static_cast<const float*>(pointers[6]), H::query, 256,
                512, position, d.input);
            if (axes) qsa_normalize_axes<<<unsigned(count * H::kv), 256, 0, d.stream>>>(key + first * K, static_cast<const float*>(pointers[7]), H::kv,
                256, 256, axes, d.input + count * Q);
            else qsa_normalize<<<unsigned(count * H::kv), 256, 0, d.stream>>>(key + first * K, static_cast<const float*>(pointers[7]), H::kv,
                256, 256, position, d.input + count * Q);
            check(hipMemcpyAsync(d.input + count * (Q + K), value + first * K, count * K * 4, hipMemcpyDeviceToDevice, d.stream));
            if (axes) qsa_normalize_axes<<<unsigned(count * 4), 128, 0, d.stream>>>(chunk_iq, static_cast<const float*>(pointers[8]), 4, 128, 128, axes, chunk_iq);
            else qsa_normalize<<<unsigned(count * 4), 128, 0, d.stream>>>(chunk_iq, static_cast<const float*>(pointers[8]), 4, 128, 128, position, chunk_iq);
            const size_t completed = (position + count) / 4 - position / 4;
            if (completed && axes)
                qsa_pool_axes<<<unsigned(completed), 128, 0, d.stream>>>(chunk_ik, history, static_cast<const float*>(pointers[9]), pool, position,
                    d.rope_rows, d.rope_first, blocks);
            else if (completed)
                qsa_pool<<<unsigned(completed), 128, 0, d.stream>>>(chunk_ik, history, static_cast<const float*>(pointers[9]), pool, position, blocks);
            qsa_save_history<<<unsigned((std::min(count, size_t(4)) * 128 + 255) / 256), 256, 0, d.stream>>>(chunk_ik, position, count, history);
            qsa_index_chunk<P>(d, chunk_iq, blocks, pool, position, count, selected, static_cast<float*>(scores), d.input);
            write_attention_kv<P><<<unsigned((count * K + 255) / 256), 256, 0, d.stream>>>(kv.kv, kv.capacity, position, count, d.input,
                nullptr, nullptr, kv.mirror, kv.mirror_rows);
            if (batched) {
                cast_half<<<unsigned((count * Q + 255) / 256), 256, 0, d.stream>>>(d.input, half + first * Q, count * Q);
                qsa_mark_blocks<P><<<unsigned(count), 256, 0, d.stream>>>(d.input, count, selected, first - span, stride, marks);
            } else qsa_attention<P>(d, kv.kv, kv.capacity, position, count, selected, d.input, attended + first * Q, nullptr, kv.mirror,
                                    kv.mirror_rows);
        }
#if defined(__HIPCC__)
        if (batched) {
            qsa_union_blocks<<<unsigned(groups), 256, 0, d.stream>>>(marks, stride, base + span, span_rows, union_blocks, union_masks, union_counts);
            qsa_flash_wmma<P><<<dim3(unsigned(groups), unsigned(H::kv)), unsigned(qsa_group * 48), 0, d.stream>>>(kv.kv, kv.capacity,
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
