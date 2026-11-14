// Speculative verification: up to eight consecutive tokens pass each decoder layer
// with the one-token arithmetic, so their logits match token-by-token decode
// bitwise. Weight reads are shared by the multi-vector GEMV and HyperConnection
// kernels; recurrence, attention and experts advance token by token in causal
// order. Each DeltaNet layer keeps a snapshot of its states and the tokens'
// projections, so a partial acceptance restores the snapshot and replays only the
// accepted tokens. QSA K/V needs no rollback: rejected positions are rewritten
// before any later token can select them.
constexpr size_t verify_max = 8, verify_projection = 16480, verify_qsa = 13952;
static_assert(verify_max <= kv_mirror_margin && verify_max + 2 <= qsa_history, "a rejected row must leave what the kept rows read");
constexpr size_t verify_snapshot = 4 * 10240 + 48 * 128 * 128;
// The rows' positions, then at this distance their rope positions (the same numbers but past an image: Device::rope_delta).
constexpr size_t verify_ropes = verify_max;

struct VerifyLayout {
    float* hidden; float* mixed; float* branch; uint32_t* positions;
    float* post; float* qsa; float* attended;
};
constexpr size_t verify_front = 131072;
static_assert(verify_max * (10240 + 2564 + 2560) + verify_max * 2 <= verify_front, "a verification's rows overlap what follows them");
constexpr size_t verify_extent = verify_front + verify_max * (6144 + verify_qsa + 6144);
VerifyLayout verify_layout(float* workspace) {
    VerifyLayout v{};
    v.hidden = workspace; v.mixed = v.hidden + verify_max * 10240; v.branch = v.mixed + verify_max * 2564;
    v.positions = reinterpret_cast<uint32_t*>(v.branch + verify_max * 2560);
    v.post = workspace + verify_front; v.qsa = v.post + verify_max * 6144; v.attended = v.qsa + verify_max * verify_qsa;
    return v;
}
void hc_operands(void* const* pointers, const int* kinds, Matrix (&matrices)[4]) {
    const size_t rows[4] = {1,320,10240,4}, cols[4] = {10240,10240,320,10240};
    for (size_t i = 0; i < 4; ++i)
        matrices[i] = {static_cast<const uint8_t*>(pointers[i]),matrix_row_bytes(kinds[i],cols[i]),rows[i],cols[i],0,0,0,kinds[i]};
}

// A DeltaNet layer's snapshot and saved projections; a part's are its heads' (QwenHeads<2>).
size_t verify_snapshot_floats(const DecoderLayer& plan) { return verify_snapshot / plan.parts; }
size_t verify_projection_floats(const DecoderLayer& plan) {
    return plan.parts == 2 ? QwenHeads<2>::projection : QwenHeads<1>::projection;
}
void verify_buffers(Device& d, DecoderLayer& plan) {
    if (plan.snapshot && (!d.pipelined || plan.snapshot_alt)) return;
    // (a partially resident layer's rows are at most hybrid_rows: a low VRAM budget's expert slots keep the rest)
    const size_t rows = d.resident.find(plan.moe)->second->hybrid ? hybrid_rows : verify_max;
    const size_t bytes = (verify_snapshot_floats(plan) + rows * verify_projection_floats(plan)) * 4;
    // (a pipelined device holds a second window beside the first: the pass behind one in flight saves its own)
    for (int copy = plan.snapshot ? 1 : 0; copy < (d.pipelined ? 2 : 1); ++copy) {
        if (bytes > d.budget - d.used || allocation_tail > d.budget - d.used - bytes)
            throw std::runtime_error("speculative verification state exceeds the Straydog VRAM budget");
        float* buffer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&buffer), bytes + allocation_tail));
        d.allocations.emplace(buffer, bytes); d.used += bytes + allocation_tail;
        (copy ? plan.snapshot_alt : plan.snapshot) = buffer;
        (copy ? plan.projections_alt : plan.projections) = buffer + verify_snapshot_floats(plan);
    }
}

// Pipelined verification (the first device runs the next pass's layers while the second finishes this one's): each pass of a
// device is in a window, 0 or 1 by turns, that holds its DeltaNet snapshots, projections, verified marks and replay lists. The
// window in force is the one `plan.snapshot` and the rest name (the pairs swap), so nothing else reads a window.
void set_window(Device& d, size_t window) {
    if (window > 1) throw std::runtime_error("invalid verification window");
    if (window == d.window) return;
    if (window && !d.pipelined) throw std::runtime_error("a second verification window needs the pipelined setup");
    for (auto& entry : d.decoder) {
        DecoderLayer& plan = entry.second;
        std::swap(plan.snapshot, plan.snapshot_alt); std::swap(plan.projections, plan.projections_alt);
        std::swap(plan.verified, plan.verified_alt);
        std::swap(plan.touched_epoch, plan.touched_epoch_alt);
    }
    d.gdn_replays.swap(d.gdn_replays_alt);
    d.window = window;
}
// The graphs of a layer capture its window's buffers: another window, another graph.
#if defined(__HIPCC__)
constexpr size_t window_variant(const Device&) { return 0; }
#else
size_t window_variant(const Device& d) { return d.window << 24; }
#endif
// A span the other thread has told to stop (sd_abort_span) unwinds as an exception does: HybridAwait publishes what the graphs
// launched ahead still wait for and drains the stream.
struct SpanAborted : std::runtime_error { SpanAborted() : std::runtime_error("Straydog span aborted") {} };

// steps tokens of a DeltaNet layer; a replay reads the states from the snapshot.
void gdn_verify(Device& d, const DecoderLayer& plan, size_t steps, bool replay, float* post) {
    void* const* p = plan.pointers.data() + 4;
    auto* conv = static_cast<float*>(plan.states[0]); auto* recurrent = static_cast<float*>(plan.states[1]);
    const float* history_in = replay ? plan.snapshot : gdn_history_snapshot(d, conv);
    gdn_tokens<<<48,gdn_threads,0,d.stream>>>(replay ? plan.snapshot + 4 * 10240 : recurrent, history_in,
        recurrent, conv, static_cast<const float*>(p[5]), static_cast<const float*>(p[6]), static_cast<const float*>(p[7]),
        static_cast<const float*>(p[8]), plan.projections, verify_projection, steps, replay ? nullptr : plan.snapshot, post);
}

// A layer's part on a device of two (qwen_parts.hpp).
void verify_layer_part(Device& d, DecoderLayer& plan, size_t steps, const VerifyLayout& v);

// A layer up to its experts: the rows in v.mixed, the attention's residual in v.hidden.
void verify_layer_front(Device& d, DecoderLayer& plan, size_t steps, const VerifyLayout& v) {
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
                              v.positions, v.positions + verify_ropes, v.qsa, v.attended);
        else   // (the rows side by side: each row the one token's arithmetic)
            qsa_decode_attend_rows(d, pointers + 4, plan.states[0], plan.states[1], plan.states[2], plan.context,
                                   v.positions, v.positions + verify_ropes, v.qsa, verify_qsa, v.attended, steps);
        gemv(d, &m[5], 1, v.attended, nullptr, steps, 6144, 0, residual);
    }
    hc_operands(pointers + 4 + attention, kinds + 4 + attention, hc);
    hc_decode(d, hc, true, v.hidden, d.output, v.mixed, steps, 2564);
}

void verify_layer(Device& d, DecoderLayer& plan, size_t steps, const VerifyLayout& v) {
    verify_layer_front(d, plan, steps, v);
    moe_tokens(d, plan.moe, steps, v.mixed, 2564, v.hidden);
}

// Rows of a verification through partially resident layers: verify_layer's arithmetic up to the experts, then the rows' routes
// side by side (hybrid_moe.hpp), each row's CPU experts on the host. Under prelaunch a layer's graph awaits the last one's.
// The rows come from the host (`input`) or from device memory (`source`). The MTP layer (begin 48), running ahead of the
// model's layers in the same pass, counts no token and starts no swaps of its own.
void hybrid_verify_span(Device& d, size_t begin, size_t end, size_t position, size_t steps, const float* input, float* output,
                        const float* source = nullptr) {
    if ((!input && !source) || !steps || steps > verify_max || steps > hybrid_rows) throw std::runtime_error("invalid hybrid verification");
#if !defined(__HIPCC__)
    if (end > hybrid_layer_slots) throw std::runtime_error("invalid hybrid verification");   // (a layer a slot of the exchange)
#endif
    float* workspace = nullptr;
    for (size_t layer = begin; layer < end; ++layer) {
        auto found = d.decoder.find(layer);
        if (found == d.decoder.end() || position + steps > found->second.context || (workspace && workspace != found->second.workspace) ||
            !d.resident.find(found->second.moe)->second->hybrid)
            throw std::runtime_error("verification span has no compatible hybrid layer");
        workspace = found->second.workspace;
        if (decoder_linear(layer)) verify_buffers(d, found->second);
    }
    if (d.allocations.find(workspace)->second / 4 < verify_extent)
        throw std::runtime_error("speculative verification needs a larger decoder workspace");
#if !defined(__HIPCC__)
    if (d.aborted()) throw SpanAborted();
#endif
    const uint64_t epoch=d.front_active ? d.front_epoch[d.window] : 0;
    if (epoch && (d.front_position[d.window]!=position || d.front_rows[d.window]!=steps))
        throw std::runtime_error("verification front differs from its generation");
    kv_prepare_span(d, begin, end, position, steps);
    const VerifyLayout v = verify_layout(workspace);
    if (input) upload(d, v.hidden, input, steps * 10240 * 4);
    else if (source != v.hidden) check(hipMemcpyAsync(v.hidden, source, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    uint32_t positions[verify_max + verify_ropes]{};
    for (size_t b = 0; b < steps; ++b) {
        positions[b] = uint32_t(position + b);
        positions[verify_ropes + b] = positions[b] - d.rope_delta;
    }
    upload(d, v.positions, positions, sizeof(positions));
    if (begin < 48 && position != d.hybrid_position) { d.hybrid_position = position; d.hybrid_tokens += steps; }
    hybrid_settle(d, false);
    const bool ahead = hybrid_ahead(d, begin, end);
    HybridAwait await(d);
    auto* counters = reinterpret_cast<volatile uint32_t*>(d.exchange_device + hybrid_sequence_slot);
#if defined(__HIPCC__)
    const bool stage = false;
#else
    const bool stage = ahead;
#endif
    // The previous layer's CPU result: its exchange is the one of its parity.
    auto finish_previous = [&](size_t previous) {
        const size_t slot = hybrid_slot_of(previous);
        if (ahead) hybrid_await<<<1,256,0,d.stream>>>(counters, d.exchange_device + hybrid_result_at(0, slot), stage ? d.cpu_partial : nullptr, d.await_timeout, hybrid_need(d),
                                                      HYBRID_AWAITED(d), d.flags + hybrid_await_device_failed);
#if !defined(__HIPCC__)
        if (stage && steps > 1)
            hybrid_stage<<<dim3(1,unsigned(steps - 1)),256,0,d.stream>>>(d.exchange_device + hybrid_result_at(1, slot), d.cpu_partial + 2560, hybrid_need(d));
#endif
        hybrid_finish<<<dim3(40,unsigned(steps)),256,0,d.stream>>>(v.hidden, v.mixed, 2564, v.branch, stage ? d.cpu_partial : d.exchange_device + hybrid_result_at(0, slot),
                                                                   stage ? 2560 : hybrid_row_floats, hybrid_clock(d));
    };
    auto launch = [&](size_t layer) {
        DecoderLayer& plan = d.decoder.find(layer)->second;
        ResidentMoe& experts = *d.resident.find(plan.moe)->second;
        const bool finish = layer > begin;
        if (epoch && plan.touched_epoch==epoch) throw std::runtime_error("a verification front overwrote its snapshot");
        auto submit = [&]() {
            if (finish) finish_previous(layer - 1);
            verify_layer_front(d, plan, steps, v);
            const bool deferred = hybrid_route_export(d, experts, v.mixed, steps, 2564, hybrid_slot_of(layer));
            hybrid_publish(d, experts, steps, finish, hybrid_slot_of(layer), deferred ? v.mixed : nullptr, 2564);
            hybrid_resident_experts(d, experts, v.mixed, v.branch, steps);
            check(hipGetLastError());
        };
        if (ahead && finish) ++await.launched;
        run_graph(d, ahead ? "hybrid-verify-await" : "hybrid-verify", plan.dependencies.data(), plan.dependencies.size(),
                  (layer * 2 + finish) * 8 + steps + window_variant(d), submit);
        flush_launches(d);
        // CUDA gdn_tokens saves every state before its token loop and has no abort path.
        // A successful issue plus the checked drain proves this generation's snapshot was written.
        if (epoch) plan.touched_epoch=epoch;
    };
    auto launch_after = [&](size_t layer) {
        if (layer + 1 < end) launch(layer + 1);
        else {
            ++await.launched;
            finish_previous(end - 1);
            check(hipGetLastError());
            flush_launches(d);
        }
    };
    if (ahead) launch(begin);
    for (size_t layer = begin; layer < end; ++layer) {
        const bool before = ahead && hybrid_early(d, layer);
        if (!ahead) launch(layer);
        else if (before) launch_after(layer);
        ResidentMoe& experts = *d.resident.find(d.decoder.find(layer)->second.moe)->second;
        const auto waited = std::chrono::steady_clock::now();
        hybrid_wait_export(d, ++d.hybrid_sequence);
        if (ahead) await.check_failed();
#if !defined(__HIPCC__)
        if (d.aborted()) throw SpanAborted();
#endif
        const auto computed = std::chrono::steady_clock::now();
        {
            const float* rows_exported[16]; float* rows_result[16];
            for (size_t row = 0; row < steps; ++row) {
                rows_exported[row] = d.exchange + hybrid_export_at(row, hybrid_slot_of(layer));
                rows_result[row] = d.exchange + hybrid_result_at(row, hybrid_slot_of(layer));
            }
            hybrid_host_experts_rows(d, experts, steps, rows_exported, rows_result);
        }
        if (ahead) await.publish();
        const auto finished = std::chrono::steady_clock::now();
        d.hybrid_wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(computed - waited).count());
        d.hybrid_cpu_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(finished - computed).count());
        if (ahead && !before) launch_after(layer);
    }
    if (!ahead) finish_previous(end - 1);
    d.hybrid_layers += end - begin; d.hybrid_timed_layers += end - begin - 1;
    check(hipGetLastError());
    for (size_t layer = begin; layer < end; ++layer) d.decoder.find(layer)->second.verified = steps;
    if (output) download(d, output, v.hidden, steps * 10240 * 4);
    else check(hipStreamSynchronize(d.stream));
    if (ahead) await.check_failed();
    hybrid_check_report(d);
    if (begin < 48 && (end == 48 || d.decoder.find(end) == d.decoder.end()) && d.hybrid_verify_balanced / 4 != d.hybrid_tokens / 4 &&
        !d.frozen_slots) {
        d.hybrid_verify_balanced = d.hybrid_tokens;
        if (hybrid_refresh_due(d)) hybrid_rebalance(d, 32, false);
    }
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
    uint32_t positions[verify_max + verify_ropes]{};
    for (size_t b = 0; b < steps; ++b) {
        positions[b] = uint32_t(position + b);
        positions[verify_ropes + b] = positions[b] - d.rope_delta;
    }
    upload(d, v.positions, positions, sizeof(positions));
    auto submit = [&]() {
        for (size_t layer = begin; layer < end; ++layer) {
            DecoderLayer& plan = d.decoder.find(layer)->second;
            if (plan.parts == 2) verify_layer_part(d, plan, steps, v);
            else verify_layer(d, plan, steps, v);
        }
        check(hipGetLastError());
    };
    parts = span_has_parts(d, begin, end);
    if (parts) {
        run_with_helper(d, begin, end, steps, [&]() {
            run_graph(d, "verify-span", dependencies.data(), dependencies.size(), (begin * 64 + end) * 8 + steps, submit);
        });
    } else if (d.decoder.find(begin)->second.parts == 2) {
        // Layers in parts run as graphs of a few layers: the host submits a graph's ~20 nodes a layer at ~1 us each, and both
        // devices start the span at once (qwen_parts.hpp), so a 47-layer graph had the device catch up with its submission at
        // the second layer and wait there 0.6-0.8 ms. The first graph is two layers, the device's start; the host submits the
        // rest while it runs.
        for (size_t first = begin; first < end;) {
            const size_t last = std::min(end, first + (first == begin ? 2 : 6));
            std::vector<void*> chunk;
            for (size_t layer = first; layer < last; ++layer) {
                const auto& plan = d.decoder.find(layer)->second;
                chunk.insert(chunk.end(), plan.dependencies.begin(), plan.dependencies.end());
            }
            run_graph(d, "verify-span", chunk.data(), chunk.size(), (first * 64 + last) * 8 + steps, [&]() {
                for (size_t layer = first; layer < last; ++layer) verify_layer_part(d, d.decoder.find(layer)->second, steps, v);
                check(hipGetLastError());
            });
            first = last;
        }
    } else run_graph(d, "verify-span", dependencies.data(), dependencies.size(), (begin * 64 + end) * 8 + steps, submit);
    for (size_t layer = begin; layer < end; ++layer) d.decoder.find(layer)->second.verified = steps;
    return v.hidden;
}

// input==nullptr continues from rows already in the span's workspace; output may be nullptr.
void verify_span(Device& d, size_t begin, size_t end, size_t position, size_t steps, const float* input, float* output) {
    auto first = d.decoder.find(begin);
    if (begin < end && first != d.decoder.end() && d.resident.find(first->second.moe)->second->hybrid) {
        // (no rows given: they are in the span's workspace, where the layers before it left them)
        hybrid_verify_span(d, begin, end, position, steps, input, output, input ? nullptr : verify_layout(first->second.workspace).hidden);
        return;
    }
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
template<int P = 1>
__global__ void __launch_bounds__(gdn_threads) gdn_replay_layers(const GdnReplay* layers, size_t tokens) {
    const GdnReplay& l = layers[blockIdx.y];
    gdn_tokens_head<P>(blockIdx.x, l.snapshot + 4 * QwenHeads<P>::channels, l.snapshot, l.state, l.history, l.coefficients, l.bias,
                       l.decay, l.norm, l.projections, QwenHeads<P>::projection, tokens, nullptr, nullptr);
}
void gdn_replay_span(Device& d, size_t begin, size_t end, size_t kept) {
    std::vector<GdnReplay> layers;
    size_t parts = 0;
    for (size_t layer = begin; layer < end; ++layer) {
        if (!decoder_linear(layer)) continue;
        const DecoderLayer& plan = d.decoder.find(layer)->second;
        if (parts && plan.parts != parts) throw std::runtime_error("a replay span mixes whole and partial layers");
        parts = plan.parts;
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
    if (parts == 2)
        gdn_replay_layers<2><<<dim3(unsigned(QwenHeads<2>::value), unsigned(layers.size())), gdn_threads, 0, d.stream>>>(
            static_cast<const GdnReplay*>(cached.second), kept);
    else gdn_replay_layers<<<dim3(48, unsigned(layers.size())), gdn_threads, 0, d.stream>>>(static_cast<const GdnReplay*>(cached.second), kept);
}

// A partially resident model's verification buffers before its expert slots take what VRAM is left.
void verify_reserve(Device& d) {
    for (auto& entry : d.decoder) if (decoder_linear(entry.first)) verify_buffers(d, entry.second);
}

// Keep the first `kept` verified tokens: DeltaNet layers restore their snapshot and
// replay those tokens; everything else already holds causal state.
void verify_commit(Device& d, size_t begin, size_t end, size_t kept, bool force = false) {
    if (begin >= end || !kept) throw std::runtime_error("invalid speculative commit");
    size_t steps = 0;
    for (size_t layer = begin; layer < end; ++layer) {
        auto found = d.decoder.find(layer);
        if (found == d.decoder.end() || !found->second.verified || (steps && found->second.verified != steps))
            throw std::runtime_error("invalid speculative commit");
        steps = found->second.verified;
    }
    if (kept > steps) throw std::runtime_error("invalid speculative commit");
    // (`force`: the states went on past the kept rows, with a pass behind this one that is given up, and are put back to them
    // even where every row was kept)
    if (kept < steps || force) {
        gdn_replay_span(d, begin, end, kept);
        // (queued, not waited for: the stream orders the replay before every later use of the state, and nothing reads it here)
        check(hipGetLastError());
    }
    for (size_t layer = begin; layer < end; ++layer) d.decoder.find(layer)->second.verified = 0;
}

// Restore only graphs issued in this front. Validate the whole set before copying any state.
// Aborted CUDA fronts drain all issued graphs, whose DeltaNet snapshots are unconditional
// (gdn_tokens_head, before the token loop). A new generation never reuses its own snapshot.
int verify_window_restore(Device& d,size_t begin,size_t end,uint64_t epoch,size_t position,size_t rows,uint64_t* touched) {
    if (!touched || begin>=end || end>48) throw std::runtime_error("invalid snapshot recovery span");
    *touched=0;
    check(d.span_drain_error);
    check(hipStreamSynchronize(d.stream));
    if (!d.pipelined || !d.front_active || !epoch || d.front_epoch[d.window]!=epoch || d.front_position[d.window]!=position ||
        d.front_rows[d.window]!=rows || d.usage_epoch!=epoch) return 1;
    uint64_t mask=0;
    for (size_t layer=begin;layer<end;++layer) {
        auto found=d.decoder.find(layer);
        if (found==d.decoder.end()) return 1;
        const DecoderLayer& plan=found->second;
        const auto resident=d.resident.find(plan.moe);
        // Only hybrid_verify_span stamps successful issues into this window.
        if (resident==d.resident.end() || !resident->second->hybrid || plan.parts!=1) return 1;
        if (plan.touched_epoch!=epoch) continue;
        if (decoder_linear(layer) && (!plan.snapshot || plan.parts!=1)) return 1;
        mask|=uint64_t(1)<<layer;
    }
    kv_stage_settle(d);
    for (size_t layer=begin;layer<end;++layer) {
        DecoderLayer& plan=d.decoder.find(layer)->second;
        // Span preparation advances these views before individual graphs are issued.
        // The real K/V and QSA pools remain causal: future slots are rewritten before selection.
        if (!decoder_linear(layer)) kv_invalidate_write(d,plan.states[0],position);
        if (!(mask&(uint64_t(1)<<layer))) continue;
        if (decoder_linear(layer)) {
            constexpr size_t history=4*10240;
            check(hipMemcpyAsync(plan.states[0],plan.snapshot,history*sizeof(float),hipMemcpyDeviceToDevice,d.stream));
            check(hipMemcpyAsync(plan.states[1],plan.snapshot+history,(verify_snapshot-history)*sizeof(float),hipMemcpyDeviceToDevice,d.stream));
        }
        plan.verified=0;
    }
    check(hipStreamSynchronize(d.stream));
    d.front_epoch[d.window]=0;
    *touched=mask;
    return 0;
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
__global__ void row_argmax_finish(const float* best_values, const uint32_t* best_indices, int32_t* result) { SD_KERNEL_ACQUIRE;
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
__global__ void row_sample_chunks(const float* values, size_t count, size_t stride, const uint32_t* sample_at, uint32_t salt, float* best_values, uint32_t* best_indices) { SD_KERNEL_ACQUIRE;
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
__global__ void row_draw_sum_chunks(const float* values, size_t count, const int32_t* chosen, const uint32_t* sample_at, float* sums) { SD_KERNEL_ACQUIRE;
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
__global__ void row_draw_probability_finish(const float* sums, size_t count, const int32_t* chosen, float* result) { SD_KERNEL_ACQUIRE;
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
// first: every chunk's `top` (argmax_chunks blocks a row), then one block a row the chunks'. A wave takes its share's `top`
// in rounds of shuffles alone (each lane's best kept, found again only when taken), the block's first wave the waves'
// (rounds of block-wide bests, two barriers each, took a chunk 42 us).
constexpr unsigned head_top_max = 64, head_top_held = 16;
__device__ __forceinline__ bool top_better(float a, uint32_t ia, float b, uint32_t ib) { return a > b || (a == b && ia < ib); }
// The wave's `top` best of the pairs its lanes hold (unset: -inf at index ~0u), most likely first, written by lane 0.
__device__ void top_wave(float (&values)[head_top_held], uint32_t (&indices)[head_top_held], unsigned top, float* out_values,
                         uint32_t* out_indices) {
    float best = -INFINITY; uint32_t at = ~0u;
    const auto rescan = [&]() {
        best = -INFINITY; at = ~0u;
        #pragma unroll
        for (unsigned i = 0; i < head_top_held; ++i)
            if (top_better(values[i], indices[i], best, at)) { best = values[i]; at = indices[i]; }
    };
    rescan();
    for (unsigned round = 0; round < top; ++round) {
        float value = best; uint32_t index = at;
        for (unsigned offset = 16; offset; offset /= 2) {
            const float other = __shfl_xor(value, offset, 32); const uint32_t other_at = __shfl_xor(index, offset, 32);
            if (top_better(other, other_at, value, index)) { value = other; index = other_at; }
        }
        if (threadIdx.x % 32 == 0) { out_values[round] = value; out_indices[round] = index; }
        if (index != ~0u && at == index) {
            #pragma unroll
            for (unsigned i = 0; i < head_top_held; ++i)
                if (indices[i] == index) { values[i] = -INFINITY; indices[i] = ~0u; }
            rescan();
        }
    }
}
// The block's `top` of `count` pairs (from_values[i], from_indices ? from_indices[i] : first + i), into out: each wave its run's,
// then the first wave the waves'. At most 8 x 32 x head_top_held pairs.
__device__ void top_block(const float* from_values, const uint32_t* from_indices, size_t first, size_t count, unsigned top,
                          float* out_values, uint32_t* out_indices) {
    __shared__ float wave_values[8 * head_top_max];
    __shared__ uint32_t wave_indices[8 * head_top_max];
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32, waves = blockDim.x / 32;
    const size_t share = (count + waves - 1) / waves, begin = std::min(count, size_t(wave) * share), end = std::min(count, begin + share);
    float held[head_top_held]; uint32_t at[head_top_held];
    #pragma unroll
    for (unsigned i = 0; i < head_top_held; ++i) {
        const size_t j = begin + lane + size_t(i) * 32;
        const bool in = j < end;
        held[i] = in ? from_values[j] : -INFINITY;
        at[i] = in ? (from_indices ? from_indices[j] : uint32_t(first + j)) : ~0u;
    }
    top_wave(held, at, top, wave_values + wave * top, wave_indices + wave * top);
    __syncthreads();
    if (wave) return;
    #pragma unroll
    for (unsigned i = 0; i < head_top_held; ++i) {
        const size_t j = lane + size_t(i) * 32;
        const bool in = j < size_t(waves) * top;
        held[i] = in ? wave_values[j] : -INFINITY; at[i] = in ? wave_indices[j] : ~0u;
    }
    top_wave(held, at, top, out_values, out_indices);
}
__global__ void __launch_bounds__(256) row_top_chunks(const float* values, size_t count, size_t stride, unsigned top,
                                                      float* chunk_values, uint32_t* chunk_indices) {
    const size_t row = blockIdx.y, per = (count + argmax_chunks - 1) / argmax_chunks;
    const size_t first = std::min(count, size_t(blockIdx.x) * per), last = std::min(count, first + per);
    const size_t slot = (row * argmax_chunks + blockIdx.x) * top;
    top_block(values + row * stride + first, nullptr, first, last - first, top, chunk_values + slot, chunk_indices + slot);
}
__global__ void __launch_bounds__(256) row_top_finish(const float* chunk_values, const uint32_t* chunk_indices, unsigned top,
                                                      float* result) {
    const size_t row = blockIdx.x, candidates = size_t(argmax_chunks) * top;
    float* out = result + row * 2 * top;
    top_block(chunk_values + row * candidates, chunk_indices + row * candidates, 0, candidates, top, out,
              reinterpret_cast<uint32_t*>(out + top));
}
// The two halves' `top` pairs of each row (the peer's rows [0, split), this device's from split on) as one row's: one wave.
// (this device's rows are the head's from `first`, then the tokens of `extra_tokens`, a draft's vocabulary)
__global__ void __launch_bounds__(32) head_top_merge(const float* own, const float* peer, uint32_t first, uint32_t own_rows, const int32_t* extra_tokens,
                                                     unsigned top, float* result) {
    const size_t row = blockIdx.x;
    const float* low = peer + row * 2 * top; const float* high = own + row * 2 * top;
    float held[head_top_held]; uint32_t at[head_top_held];
    #pragma unroll
    for (unsigned i = 0; i < head_top_held; ++i) {
        const size_t index = threadIdx.x + size_t(i) * 32;
        held[i] = -INFINITY; at[i] = ~0u;
        if (index < top) { held[i] = low[index]; at[i] = __float_as_uint(low[top + index]); }
        else if (index < 2 * size_t(top)) {
            const uint32_t row_at = __float_as_uint(high[index]);
            held[i] = high[index - top]; at[i] = row_at < own_rows ? row_at + first : uint32_t(extra_tokens[row_at - own_rows]);
        }
    }
    float* out = result + row * 2 * top;
    top_wave(held, at, top, out, reinterpret_cast<uint32_t*>(out + top));
}
// Each row's `top` pairs into result: `top` values, then `top` indices (bits); scratch: rows x argmax_chunks x top x 2 words.
void row_top(Device& d, const float* values, size_t rows, size_t count, size_t stride, unsigned top, float* scratch, float* result) {
    if (!top || top > head_top_max || (count + argmax_chunks - 1) / argmax_chunks > 256 * head_top_held)
        throw std::runtime_error("invalid Straydog row top");
    // (a block's waves hand on at most 8 x top pairs, which the first wave holds: top at most 64)
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

// HyperConnection mix (norm, down, up) and the vocabulary head for up to eight rows,
// each with the one-row arithmetic; returns the first maximum of every row and, when
// asked, its probability (drafts stop at an unlikely one).
// The rows come from the host (`input`) or, when that is null, from device memory (`source`).
// With `top` (at most head_top_max), each row's `top` greatest logits and their tokens (row_top) come back in `best`.
constexpr size_t head_top_floats = verify_max * 2 * head_top_max + verify_max * argmax_chunks * head_top_max * 2;
// Diagnostics and their cleanup restore the caller's device with checked errors.
template <class Function>
void head_on_device(int ordinal, Function&& work) {
    int current = 0;
    check(hipGetDevice(&current));
    if (current != ordinal) check(hipSetDevice(ordinal));
    std::exception_ptr failure;
    try { work(); } catch (...) { failure = std::current_exception(); }
    if (current != ordinal) check(hipSetDevice(current));
    if (failure) std::rethrow_exception(failure);
}

// Events exist only for explicit diagnostics, outside capture, on their owning GPU.
class HeadEventTimer {
public:
    HeadEventTimer(Device& d, bool enabled) : d_(d), stream_(d.stream) {
        if (!enabled) return;
        try {
            head_on_device(d_.ordinal, [&]() {
                check(d_.head_timing_error);
                hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
                check(hipStreamIsCapturing(stream_, &capture));
                if (d_.op_recording || capture != hipStreamCaptureStatusNone)
                    throw std::runtime_error("head timing requires an uncaptured stream");
                check(hipEventCreate(&start_));
                check(hipEventCreate(&stop_));
            });
        } catch (...) { release(); throw; }
    }
    ~HeadEventTimer() { release(); }
    HeadEventTimer(const HeadEventTimer&) = delete;
    HeadEventTimer& operator=(const HeadEventTimer&) = delete;
    void begin() {
        if (start_) head_on_device(d_.ordinal, [&]() { check(hipEventRecord(start_, stream_)); });
    }
    void end() {
        if (stop_) head_on_device(d_.ordinal, [&]() { check(hipEventRecord(stop_, stream_)); });
    }
    double milliseconds() {
        if (!stop_) return 0;
        float elapsed = 0;
        head_on_device(d_.ordinal, [&]() {
            check(hipEventSynchronize(stop_));
            check(hipEventElapsedTime(&elapsed, start_, stop_));
        });
        return double(elapsed);
    }
    void close() { check(release()); }
private:
    Device& d_;
    hipStream_t stream_;
    hipEvent_t start_ = nullptr, stop_ = nullptr;
    hipError_t release() noexcept {
        if (!start_ && !stop_) return hipSuccess;
        hipError_t first = hipSuccess;
        auto note = [&](hipError_t error) {
            if (error != hipSuccess) {
                if (first == hipSuccess) first = error;
                if (d_.head_timing_error == hipSuccess) d_.head_timing_error = error;
            }
        };
        int previous = -1;
        const hipError_t current = hipGetDevice(&previous);
        note(current);
        const hipError_t selected = hipSetDevice(d_.ordinal);
        note(selected);
        if (selected == hipSuccess) {
            if (start_) note(hipEventDestroy(start_));
            if (stop_) note(hipEventDestroy(stop_));
        }
        if (current == hipSuccess && previous != d_.ordinal) note(hipSetDevice(previous));
        start_ = stop_ = nullptr;
        return first;
    }
};

// `source` rows on the device (rather than `input`'s on the host) come back in `rows` when asked, with the tokens.
// `allocated` (a draft's vocabulary on one device, draft_head): the head's rows the allocation holds, where `vocabulary` rows of it are read.
void output_head(Device& d, void* const* pointers, const int* kinds, size_t steps, size_t vocabulary,
                 const float* input, int32_t* tokens, float* logits, float* confidence, const float* source = nullptr,
                 unsigned top_count = 0, float* best = nullptr, float* rows_out = nullptr, size_t allocated = 0,
                 bool draft_timing = false) {
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
    if (found == d.allocations.end() || found->second != (allocated ? allocated : vocabulary) * head.row_bytes) throw std::runtime_error("invalid head operand");
    const bool timed = draft_timing && d.head_timing_enabled;
    HeadEventTimer total_time(d, timed), projection_time(d, timed);
    float* rows = d.input; float* mixed = rows + steps * 10240;
    float* partial = mixed + steps * 2564;
    auto* chosen = reinterpret_cast<int32_t*>(partial + 2 * verify_max * argmax_chunks);
    if (input) upload(d, rows, input, steps * 10240 * 4);
    else check(hipMemcpyAsync(rows, source, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    hc_decode(d, hc, false, rows, d.output, mixed, steps, 2564);
    total_time.begin(); projection_time.begin();
    gemv(d, &head, 1, mixed, d.output, steps, 2564, vocabulary);
    projection_time.end();
    row_argmax(d, d.output, steps, vocabulary, vocabulary, partial, chosen);
    float* top = reinterpret_cast<float*>(chosen + verify_max);
    if (confidence) {
        row_exp_sum(d, d.output, steps, vocabulary, vocabulary, chosen, partial, top);
        head_confidence<<<1, 32, 0, d.stream>>>(top, nullptr, steps, top + 2 * verify_max);
    }
    float* ranked = top + 4 * verify_max;
    if (top_count) row_top(d, d.output, steps, vocabulary, vocabulary, top_count, ranked + verify_max * 2 * head_top_max, ranked);
    check(hipGetLastError());
    total_time.end();
    Piece pieces[5]; size_t count = 0;
    if (rows_out) pieces[count++] = {rows_out, rows, steps * 10240 * 4};
    pieces[count++] = {tokens, chosen, steps * 4};
    if (confidence) pieces[count++] = {confidence, top + 2 * verify_max, steps * 4};
    if (logits) pieces[count++] = {logits, d.output, steps * vocabulary * 4};
    if (top_count) pieces[count++] = {best, ranked, steps * 2 * top_count * 4};
    download_pieces(d, pieces, count);
    if (timed) {
        const double projection = projection_time.milliseconds(), total = total_time.milliseconds();
        projection_time.close(); total_time.close();
        ++d.head_timing_calls[0];
        d.head_timing_ms[0] += projection; d.head_timing_ms[1] += total;
    }
}

// The non-finite values among `count` floats: each block's verdict into flags[block].
__global__ void __launch_bounds__(256) head_nonfinite(const float* values, size_t count, uint32_t* flags) {
    unsigned bad = 0;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += size_t(gridDim.x) * 256)
        bad |= (__float_as_uint(values[i]) & 0x7f800000u) == 0x7f800000u;
    __shared__ unsigned any;
    if (threadIdx.x == 0) any = 0;
    __syncthreads();
    if (bad) atomicOr(&any, 1u);
    __syncthreads();
    if (threadIdx.x == 0) flags[blockIdx.x] = any;
}
constexpr size_t decode_head_flags = 64;

// A decode step's head: the HyperConnection mix of one row of 10240 and the vocabulary projection of the mixed row, both launched
// before the host waits (the mix was a launch, a wait and a copy to the host, and the projection a copy back, a launch and a wait),
// the logits checked for non-finite values on the device (the host's pass over them took 100 us), then read back in the one wait,
// followed by the checks' flags. The arithmetic is output_head's one row's, which is sd_hyperconnection's mix and
// sd_matvec_batch's product. `logits` has room for vocabulary + decode_head_flags floats.
void decode_head(Device& d, void* const* pointers, const int* kinds, size_t vocabulary, const float* input, float* logits) {
    if (!input || !logits || !vocabulary || vocabulary + decode_head_flags > d.output_capacity || 10240 + 2564 > d.input_capacity)
        throw std::runtime_error("invalid Straydog decode head");
    Matrix hc[4];
    hc_operands(pointers, kinds, hc);
    const Matrix head{static_cast<const uint8_t*>(pointers[3]),matrix_row_bytes(kinds[3],2560),vocabulary,2560,0,0,0,kinds[3]};
    for (size_t i = 0; i < 3; ++i) {
        auto found = d.allocations.find(pointers[i]);
        if (found == d.allocations.end() || found->second != hc[i].rows * hc[i].row_bytes) throw std::runtime_error("invalid head operand");
    }
    auto found = d.allocations.find(pointers[3]);
    if (found == d.allocations.end() || found->second != vocabulary * head.row_bytes) throw std::runtime_error("invalid head operand");
    float* rows = d.input; float* mixed = rows + 10240;
    upload(d, rows, input, 10240 * 4);
    hc_decode(d, hc, false, rows, d.output, mixed, 1, 2564);
    gemv(d, &head, 1, mixed, d.output, 1, 2564, vocabulary);
    head_nonfinite<<<unsigned(decode_head_flags), 256, 0, d.stream>>>(d.output, vocabulary, reinterpret_cast<uint32_t*>(d.output + vocabulary));
    check(hipGetLastError());
    download(d, logits, d.output, (vocabulary + decode_head_flags) * 4);
    const auto* flags = reinterpret_cast<const uint32_t*>(logits + vocabulary);
    for (size_t i = 0; i < decode_head_flags; ++i)
        if (flags[i]) throw std::runtime_error("Straydog GPU produced non-finite values");
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
// (rows past `own_rows`: the tokens of `extra_tokens`, a draft's vocabulary)
__global__ void head_choose(const float* logits, size_t width, size_t split, const int32_t* chosen, const float* peer,
                            size_t steps, int32_t* tokens, size_t own_rows = SIZE_MAX, const int32_t* extra_tokens = nullptr) {
    const size_t row = threadIdx.x;
    if (row >= steps) return;
    const float own = logits[row * width + chosen[row]];
    const int32_t token = size_t(chosen[row]) < own_rows ? int32_t(split) + chosen[row] : extra_tokens[size_t(chosen[row]) - own_rows];
    tokens[row] = peer[2 * row] >= own ? __float_as_int(peer[2 * row + 1]) : token;
}
// `rows_of` (a draft's vocabulary): the peer's rows [0, low) and this device's [first, end) and `extra_count` rows of the tokens of
// `extra_tokens` at `extras` (rows_of.end 0: the whole head).
struct HeadRows { size_t low = 0, first = 0, end = 0, extra_count = 0; const uint8_t* extras = nullptr; const int32_t* extra_tokens = nullptr; };
void output_head_split(Device& d, void* const* pointers, const int* kinds, size_t steps, size_t vocabulary, size_t split,
                       void* peer_head, const float* input, int32_t* tokens, float* logits, float* confidence,
                       const float* source = nullptr, unsigned top_count = 0, float* best_out = nullptr, float* rows_out = nullptr,
                       HeadRows rows_of = {}, bool draft_timing = false) {
    Device* peer = d.link.peer;
    if (!rows_of.end) rows_of = {split, split, vocabulary};
    if (rows_of.low > split || !rows_of.low || rows_of.first >= rows_of.end || rows_of.end > vocabulary ||
        (logits && (rows_of.low != split || rows_of.first != split || rows_of.end != vocabulary || rows_of.extra_count)) ||
        (rows_of.extra_count && (!rows_of.extras || !rows_of.extra_tokens)))
        throw std::runtime_error("invalid Straydog split output head rows");
    const size_t low_rows = rows_of.low, high_rows = rows_of.end - rows_of.first, own_rows = high_rows + rows_of.extra_count;
    if (steps * own_rows > d.output_capacity) throw std::runtime_error("invalid Straydog split output head rows");
    if (!peer || !steps || steps > verify_max || (!input && !source) || !tokens || !split || split >= vocabulary ||
        steps * vocabulary > d.output_capacity || steps * split > peer->output_capacity ||
        64 + steps * split > d.link.rows * 2560 || 2560 * steps > peer->link.rows * 2560 ||
        steps * (10240 + 2564) + 2 * verify_max * argmax_chunks + 5 * verify_max + 2 * head_top_floats > d.input_capacity ||
        2 * verify_max * argmax_chunks + verify_max + head_top_floats + (peer->link.direct ? 0 : steps * 2560) > peer->input_capacity ||
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
    const bool timed = draft_timing && d.head_timing_enabled;
    HeadEventTimer total_time(d, timed), local_time(d, timed), peer_time(*peer, timed);
    float* rows = d.input; float* mixed = rows + steps * 10240;
    float* partial = mixed + steps * 2564;
    auto* chosen = reinterpret_cast<int32_t*>(partial + 2 * verify_max * argmax_chunks);
    int32_t* final_tokens = chosen + verify_max;
    if (input) upload(d, rows, input, steps * 10240 * 4);
    else check(hipMemcpyAsync(rows, source, steps * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
    hc_decode(d, hc, false, rows, d.output, mixed, steps, 2564);
    total_time.begin();
    peer_send(d, mixed, steps, 2560, 2564, 0);
    // (WDDM holds what is queued until the stream is asked about: the peer's half waits for these rows, and this device's
    // kernels below wait for the peer's half, so each stream is asked as soon as its part is queued)
    flush_launches(d);
    float* received = peer_partials(d.link);
    // The logits of the lower half reach the host through this device's inbox only over a direct link (the inbox in its VRAM).
    const bool host_logits = logits && !d.link.direct;
    uint8_t* helper_slots[verify_max] = {};
    head_on_device(peer->ordinal, [&]() {
        Device& h = *peer;
        const Matrix low{static_cast<const uint8_t*>(peer_head), row_bytes, low_rows, 2560, 0, 0, 0, kinds[3]};
        float* scratch = h.input;
        auto* best = reinterpret_cast<int32_t*>(scratch + 2 * verify_max * argmax_chunks);
        peer_receive_rows(h);
        // (linked through page-locked host memory, as two GPUs without peer access are: every block of the product stages the rows into
        // its shared memory, and 3,900 blocks reading them over the bus took 2.5 ms of the half's 0.3. They are copied into this device's
        // memory first, behind the head's other scratch, and the product reads them there: the same values, the same sums.)
        const float* mixed_rows = peer_inputs(h.link);
        if (!h.link.direct) {
            float* local = scratch + 2 * verify_max * argmax_chunks + verify_max + head_top_floats;
            peer_copy<<<unsigned((steps * 2560 + 255) / 256), 256, 0, h.stream>>>(local, mixed_rows, steps, 2560, 2560);
            mixed_rows = local;
        }
        peer_time.begin();
        gemv(h, &low, 1, mixed_rows, h.output, steps, 2560, low_rows);
        peer_time.end();
        row_argmax(h, h.output, steps, low_rows, low_rows, scratch, best);
        head_part<<<1, 32, 0, h.stream>>>(h.output, low_rows, best, steps, peer_remote_partials(h.link));
        // The lower half's (logit, sum) pairs follow its choices, before the logits at 64.
        if (confidence) row_exp_sum(h, h.output, steps, low_rows, low_rows, best, scratch, peer_remote_partials(h.link) + 16);
        if (logits && !host_logits) peer_copy<<<1024, 256, 0, h.stream>>>(peer_remote_partials(h.link) + 64, h.output, steps, split, split);
        // (its rows' tops at 64 instead of its logits: a few pairs a row over the link)
        if (top_count) {
            float* ranked = scratch + 2 * verify_max * argmax_chunks + verify_max;
            row_top(h, h.output, steps, low_rows, low_rows, top_count, ranked + verify_max * 2 * head_top_max, ranked);
            peer_copy<<<1, 256, 0, h.stream>>>(peer_remote_partials(h.link) + 64, ranked, 1, steps * 2 * top_count, 0);
        }
        peer_signal_peer(h);
        // (the lower half's logits, over a link through host memory: this device reads them back itself, into its own page-locked staging, beside
        // the owner's read-back of its half. Through the owner's inbox they took a kernel of 409 us writing over the bus and then, on the owner's
        // stream, a host-to-host copy of 392 us a row, where a read-back of a row's half is 30 us)
        if (host_logits)
            for (size_t row = 0; row < steps; ++row) {
                helper_slots[row] = io_slot(h, split * 4);
                check(hipMemcpyAsync(helper_slots[row], h.output + row * split, split * 4, hipMemcpyDeviceToHost, h.stream));
            }
        check(hipGetLastError());
        flush_launches(h);
    });
    const Matrix own[2] = {{static_cast<const uint8_t*>(pointers[3]) + rows_of.first * row_bytes, row_bytes, high_rows, 2560, 0, 0, 0, kinds[3]},
                           {rows_of.extras, row_bytes, rows_of.extra_count, 2560, 0, high_rows, 0, kinds[3]}};
    local_time.begin();
    gemv(d, own, rows_of.extra_count ? 2 : 1, mixed, d.output, steps, 2564, own_rows);
    local_time.end();
    row_argmax(d, d.output, steps, own_rows, own_rows, partial, chosen);
    peer_receive(d);
    head_choose<<<1, 32, 0, d.stream>>>(d.output, own_rows, rows_of.first, chosen, received, steps, final_tokens, high_rows,
                                        rows_of.extra_tokens);
    float* top = reinterpret_cast<float*>(final_tokens + verify_max);
    if (confidence) {
        row_exp_sum(d, d.output, steps, own_rows, own_rows, chosen, partial, top);
        head_confidence<<<1, 32, 0, d.stream>>>(top, received + 16, steps, top + 2 * verify_max);
    }
    float* ranked = top + 4 * verify_max;
    float* merged = ranked + head_top_floats;
    if (top_count) {
        row_top(d, d.output, steps, own_rows, own_rows, top_count, ranked + verify_max * 2 * head_top_max, ranked);
        head_top_merge<<<unsigned(steps), 32, 0, d.stream>>>(ranked, received + 64, uint32_t(rows_of.first), uint32_t(high_rows), rows_of.extra_tokens,
                                                             top_count, merged);
    }
    check(hipGetLastError());
    total_time.end();
    // (finish_with_helper's waits with the read-back's: both links' counters ride along, the peer's queued after its part, where
    // peer_settle read each with a blocking copy after the streams' waits: a draft step's head ended 0.13 ms after its last
    // kernel, now 0.06)
    uint32_t counters[2][8] = {};
    uint8_t* peer_counters = nullptr;
    head_on_device(peer->ordinal, [&]() {
        peer_counters = io_slot(*peer, 32);
        check(hipMemcpyAsync(peer_counters, peer->link.counters, 32, hipMemcpyDeviceToHost, peer->stream));
    });
    Piece pieces[5 + 2 * verify_max]; size_t count = 0;
    pieces[count++] = {counters[0], d.link.counters, 32};
    if (rows_out) pieces[count++] = {rows_out, rows, steps * 10240 * 4};
    pieces[count++] = {tokens, final_tokens, steps * 4};
    if (confidence) pieces[count++] = {confidence, top + 2 * verify_max, steps * 4};
    if (top_count) pieces[count++] = {best_out, merged, steps * 2 * top_count * 4};
    if (logits)
        for (size_t row = 0; row < steps; ++row) {
            if (!host_logits) pieces[count++] = {logits + row * vocabulary, received + 64 + row * split, split * 4};
            pieces[count++] = {logits + row * vocabulary + split, d.output + row * (vocabulary - split), (vocabulary - split) * 4};
        }
    download_pieces(d, pieces, count);
    check(hipStreamSynchronize(d.helper_stream));
    head_on_device(peer->ordinal, [&]() {
        check(hipStreamSynchronize(peer->stream)); check(hipStreamSynchronize(peer->helper_stream));
        std::memcpy(counters[1], peer_counters, 32);
    });
    if (host_logits) {
        StagingCopier::Item items[verify_max];
        for (size_t row = 0; row < steps; ++row) items[row] = {logits + row * vocabulary, helper_slots[row], split * 4};
        StagingCopier::instance().copy_items(items, steps);
    }
    peer->io_used = 0;   // (its stream is idle: download's, and its read-backs are copied out)
    peer_counters_check(d, counters);
    if (timed) {
        const double local = local_time.milliseconds(), remote = peer_time.milliseconds(), total = total_time.milliseconds();
        local_time.close(); peer_time.close(); total_time.close();
        ++d.head_timing_calls[1];
        d.head_timing_ms[2] += local; d.head_timing_ms[3] += remote; d.head_timing_ms[4] += total;
        d.head_timing_ms[5] += total - std::max(local, remote);
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
// The prologue over the rows already in the input area (see mtp_prologue_hidden). `exact`: every row on the one-row GEMV's
// arithmetic (a drafting step); otherwise a prompt's rows past four take the batch projection.
void mtp_prologue_device(Device& d, void* const* pointers, const int* kinds, size_t steps, bool exact = false) {
    mtp_prologue_check(d, pointers, kinds, steps);
    float* e = mtp_prologue_embeddings(d); float* h = mtp_prologue_hidden(d, steps);
    float* ne = h + steps * 10240; float* nh = ne + steps * 2560;
    mtp_norm<<<unsigned(steps),256,0,d.stream>>>(e, 2560, static_cast<const float*>(pointers[0]), ne);
    mtp_norm<<<unsigned(steps),256,0,d.stream>>>(h, 10240, static_cast<const float*>(pointers[2]), nh);
    phase(d, "pro_norm");
    const Matrix fe{static_cast<const uint8_t*>(pointers[1]),matrix_row_bytes(kinds[1],2560),2560,2560,0,0,0,kinds[1]};
    const Matrix fh{static_cast<const uint8_t*>(pointers[3]),matrix_row_bytes(kinds[3],2560),2560,2560,0,0,0,kinds[3]};
    float* projected = d.output;   // steps x 10240 streams, then steps x 2560 embeddings
    if (exact || steps <= size_t(gemv_max_vectors)) {
        gemv(d, &fe, 1, ne, projected + steps * 10240, steps, 2560, 2560);
        for (size_t b = 0; b < steps; ++b) gemv(d, &fh, 1, nh + b * 10240, projected + b * 10240, 4, 2560, 2560);
    } else {   // prompt rows: drafter numerics never change verified tokens
        Matrix descriptors[max_batch]{};
        hc_project(d, fe, steps, ne, projected + steps * 10240, descriptors);
        hc_project(d, fh, steps * 4, nh, projected, descriptors);
    }
    phase(d, "pro_gemm");
    mtp_combine<<<dim3(40, unsigned(steps)),256,0,d.stream>>>(projected, projected + steps * 10240, h);
    phase(d, "pro_combine");
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

// A drafting step's MTP layer with nothing waited for: the prologue over `embeddings` and the hidden rows, then the layer
// (`layer`) on them. The hidden rows come from the host, or (`hidden` null, one row) are the last of the `previous` rows the
// layer's last step left in its workspace. A layer in parts is queued on this device alone: the host queues the other's too.
// Returns the layer's rows; `parts` as verify_span_queue's.
float* draft_queue(Device& d, void* const* mtp, const int* mtp_kinds, size_t layer, size_t position, size_t steps,
                   const float* embeddings, const float* hidden, size_t previous, bool& parts) {
    auto found = d.decoder.find(layer);
    if (!embeddings || found == d.decoder.end() || (!hidden && (steps != 1 || !previous || previous > verify_max)))
        throw std::runtime_error("invalid drafting step");
    mtp_prologue_check(d, mtp, mtp_kinds, steps);
    upload(d, mtp_prologue_embeddings(d), embeddings, steps * 2560 * 4);
    float* rows = mtp_prologue_hidden(d, steps);
    if (hidden) upload(d, rows, hidden, steps * 10240 * 4);
    else check(hipMemcpyAsync(rows, verify_layout(found->second.workspace).hidden + (previous - 1) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, d.stream));
    if (steps > verify_max) throw std::runtime_error("invalid drafting step");
    mtp_prologue_device(d, mtp, mtp_kinds, steps, true);
    if (d.resident.find(found->second.moe)->second->hybrid) {
        hybrid_verify_span(d, layer, layer + 1, position, steps, nullptr, nullptr, rows);
        return verify_layout(found->second.workspace).hidden;
    }
    return verify_span_queue(d, layer, layer + 1, position, steps, nullptr, rows, parts);
}
size_t draft_read_rows(const Device& d, size_t vocabulary, unsigned top) {
    return d.pipelined_chain && !top && d.draft_rows && d.draft_rows < vocabulary && !d.draft_extra_count
        ? d.draft_rows : vocabulary;
}
// The vocabulary head on a draft's row `last` (on this device), its token and probability back; with `top`, instead of the
// probability, the row's `top` greatest logits and their tokens for the host to draw the draft from (sampling::accept_top_drawn).
void draft_head(Device& d, void* const* head, const int* head_kinds, size_t vocabulary, size_t split, void* peer_head,
                const float* last, bool parts, int32_t* token, float* confidence, unsigned top = 0, float* best = nullptr) {
    if (!head || !head_kinds || !token || !confidence || (top && !best)) throw std::runtime_error("invalid drafting step");
    int32_t token_row[verify_max];
    float confidence_row[verify_max];
    // (a draft's vocabulary, sd_draft_vocabulary: the extra tokens' rows here, the head's first rows split so that both devices
    // read as many)
    HeadRows rows_of{};
    if (split && d.draft_rows && d.draft_rows < vocabulary) {
        const size_t low = std::min({split, (d.draft_rows + d.draft_extra_count) / 2, d.draft_rows - 1});
        rows_of = {low, low, d.draft_rows, d.draft_extra_count, d.draft_extras, d.draft_extra_tokens};
    }
    float* probability = top ? nullptr : confidence_row;
    confidence_row[0] = 1;
    if (split) output_head_split(d, head, head_kinds, 1, vocabulary, split, peer_head, nullptr, token_row, nullptr, probability, last,
                                 top, best, nullptr, rows_of, true);
    else {
        // (pipelined verification: the other device is busy, so the draft's vocabulary is read on this one alone: its first rows, where no
        // token past them was noted for it; the head of every row otherwise. The probability is under those rows' softmax, as in the split head.)
        const size_t read = draft_read_rows(d, vocabulary, top);
        output_head(d, head, head_kinds, 1, read, nullptr, token_row, nullptr, probability, last, top, best, nullptr,
                    read < vocabulary ? vocabulary : 0, true);
        if (parts) finish_with_helper(d);
    }
    *token = token_row[0]; *confidence = confidence_row[0];
}
// One drafting step with the host in none of its stages: draft_queue, and when `head` is given the vocabulary head on the
// layer's last row, whose token and probability come back. A chain of steps reads nothing back until its token.
void draft_step(Device& d, void* const* mtp, const int* mtp_kinds, size_t layer, size_t position, size_t steps,
                const float* embeddings, const float* hidden, size_t previous,
                void* const* head, const int* head_kinds, size_t vocabulary, size_t split, void* peer_head,
                int32_t* token, float* confidence, unsigned top = 0, float* best = nullptr) {
    if (head && (!head_kinds || !token || !confidence)) throw std::runtime_error("invalid drafting step");
    bool parts = false;
    float* output = draft_queue(d, mtp, mtp_kinds, layer, position, steps, embeddings, hidden, previous, parts);
    if (!head) {
        if (parts) finish_with_helper(d);
        else check(hipStreamSynchronize(d.stream));
        return;
    }
    draft_head(d, head, head_kinds, vocabulary, split, peer_head, output + (steps - 1) * 10240, parts, token, confidence, top, best);
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
