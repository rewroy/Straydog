// One allocation, two layouts: prefill stages one layer; decode mirrors every
// registered host state's prefix. All copies preserve the state's F32 planes (each K/V head's
// keys, then each one's values: four for a whole layer, two for a GPU's part, Device::kv_planes).
struct KvView {
    float* kv; size_t capacity; bool staged;
    float* mirror = nullptr; size_t mirror_rows = 0;
};
size_t kv_planes_of(const Device& d, const void* state) {
    auto found=d.kv_planes.find(state);
    return found==d.kv_planes.end() ? 4 : found->second;
}
// Positions of `state`'s planes the stage holds in its one-layer (prefill) layout.
size_t kv_stage_capacity(const Device& d, const void* state) { return d.kv_stage_floats/(256*kv_planes_of(d,state)); }
void kv_copy_rows(Device& d, float* to, size_t to_capacity, const float* from, size_t from_capacity,
                  size_t first, size_t count, size_t planes, hipStream_t stream=nullptr) {
    for (size_t segment=0;count && segment<planes;++segment)
        check(hipMemcpyAsync(to+(segment*to_capacity+first)*256,from+(segment*from_capacity+first)*256,count*256*4,
                             hipMemcpyDefault,stream ? stream : d.stream));
}
// The decode layout: every host state's mirror, the same positions of each one's planes. (Before any graph reads it.)
void kv_layout(Device& d) {
    d.kv_mirrors.clear(); d.kv_mirror_rows = 0;
    d.kv_stage_owner = nullptr; d.kv_stage_valid = 0;
    if (!d.kv_stage || d.host_states.empty()) return;
    size_t planes = 0;
    for (const auto& state : d.host_states) planes += kv_planes_of(d,state.first);
    d.kv_mirror_rows = d.kv_stage_floats/(256*planes);
    if (!d.kv_mirror_rows) return;
    size_t at = 0;
    for (const auto& state : d.host_states) {
        d.kv_mirrors.emplace(state.first,KvMirror{d.kv_stage+at,0});
        at += d.kv_mirror_rows*256*kv_planes_of(d,state.first);
    }
}
// Work on the device stream that reads or writes the stage first waits for a prefetch's copy.
void kv_stage_settle(Device& d) {
    if (!d.kv_prefetching) return;
    check(hipStreamWaitEvent(d.stream,d.kv_prefetched,0));
    d.kv_prefetching=false;
}
// A prefill batch past a layer's decode mirror stages the layer's whole prefix. Between two of
// a device's attention layers the stage is idle, so the next one's prefix is copied then, on a
// stream of its own, while the layers in between compute.
void kv_prefetch(Device& d, void* state, size_t context, size_t base) {
    if (!d.kv_stage || base>kv_stage_capacity(d,state) || !d.host_states.count(state) ||
        (d.kv_mirrors.count(state) && base<d.kv_mirror_rows))
        return;
    for (auto& entry : d.kv_mirrors) entry.second.valid=0;
    if (d.kv_stage_owner!=state) { d.kv_stage_owner=state; d.kv_stage_valid=0; }
    if (d.kv_stage_valid>=base) return;
    if (!d.kv_stream) {
        check(hipStreamCreate(&d.kv_stream));
        check(hipEventCreateWithFlags(&d.kv_released,hipEventDisableTiming));
        check(hipEventCreateWithFlags(&d.kv_prefetched,hipEventDisableTiming));
    }
    // After the stage's last reader and the host rows' last writer on the device stream.
    check(hipEventRecord(d.kv_released,d.stream));
    check(hipStreamWaitEvent(d.kv_stream,d.kv_released,0));
    kv_copy_rows(d,d.kv_stage,kv_stage_capacity(d,state),static_cast<const float*>(state),context,d.kv_stage_valid,
                 base-d.kv_stage_valid,kv_planes_of(d,state),d.kv_stream);
    check(hipEventRecord(d.kv_prefetched,d.kv_stream));
    d.kv_prefetching=true;
    d.kv_stage_valid=base;
}
// The first attention layer from `layer` on this device whose K/V is in host memory, prefetched for `base`.
void kv_prefetch_from(Device& d, size_t layer, size_t base) {
    for (;layer<=48;++layer) {
        if (layer%4!=3 && layer!=48) continue;
        auto plan=d.decoder.find(layer);
        if (plan==d.decoder.end()) continue;
        if (base<=plan->second.context) kv_prefetch(d,plan->second.states[0],plan->second.context,base);
        return;
    }
}
void kv_invalidate_write(Device& d, void* state, size_t base) {
    if (state==d.kv_stage) {
        d.kv_stage_valid=0;
        for (auto& entry : d.kv_mirrors) entry.second.valid=0;
    }
    if (d.kv_stage_owner==state) d.kv_stage_valid=std::min(d.kv_stage_valid,base);
    auto mirror=d.kv_mirrors.find(state);
    if (mirror!=d.kv_mirrors.end()) mirror->second.valid=std::min(mirror->second.valid,base);
}
// Stage pointers/layout are captured in decode graphs. Changing registration or
// releasing the stage must discard those graphs before their storage can change.
void kv_clear_graphs(Device& d) {
    for (auto& entry : d.graphs) {
        check(hipGraphExecDestroy(entry.second.executable));
        check(hipGraphDestroy(entry.second.graph));
    }
    d.graphs.clear();
}
KvView kv_decode_view(Device& d, void* state, size_t context) {
    auto found=d.kv_mirrors.find(state);
    if (found==d.kv_mirrors.end()) return {static_cast<float*>(state),context,false};
    return {static_cast<float*>(state),context,false,found->second.data,d.kv_mirror_rows};
}
// Called OUTSIDE capture, once per span (also on graph replay). The same stream
// orders fills after old writes and before attention. Rewrites truncate validity;
// the current token(s) are supplied by write_attention_kv's duplicate stores.
KvView kv_prepare_decode(Device& d, void* state, size_t context, size_t base, size_t count=1) {
    kv_stage_settle(d);
    const KvView view=kv_decode_view(d,state,context);
    kv_invalidate_write(d,state,base);
    if (!view.mirror) return view;
    d.kv_stage_owner=nullptr; d.kv_stage_valid=0;
    auto& mirror=d.kv_mirrors.find(state)->second;
    const size_t end=std::min(base,view.mirror_rows);
    const size_t valid=std::min(mirror.valid,end);
    kv_copy_rows(d,view.mirror,view.mirror_rows,view.kv,context,valid,end-valid,kv_planes_of(d,state));
    mirror.valid=std::min(base+count,view.mirror_rows);
    return view;
}
// A batch that fits its layer's decode mirror runs there: the mirror stays valid for the
// decode and the follow-up batches after it, instead of re-copying the whole prefix per layer.
KvView kv_for_prefill(Device& d, void* state, size_t context, size_t base, size_t count) {
    kv_stage_settle(d);
    if (count==1) return kv_prepare_decode(d,state,context,base);
    if (d.kv_mirrors.count(state) && base+count<=d.kv_mirror_rows) {
        const KvView view=kv_prepare_decode(d,state,context,base,count);
        return {view.mirror,view.mirror_rows,true};
    }
    kv_invalidate_write(d,state,base);
    if (!d.kv_stage || base+count>kv_stage_capacity(d,state) || !d.host_states.count(state))
        return {static_cast<float*>(state),context,false};
    // The prefill layout overwrites every layer's decode mirror.
    for (auto& entry : d.kv_mirrors) entry.second.valid=0;
    if (d.kv_stage_owner!=state) { d.kv_stage_owner=state; d.kv_stage_valid=0; }
    const size_t valid=std::min(d.kv_stage_valid,base);
    const size_t capacity=kv_stage_capacity(d,state);
    kv_copy_rows(d,d.kv_stage,capacity,static_cast<const float*>(state),context,valid,base-valid,kv_planes_of(d,state));
    return {d.kv_stage,capacity,true};
}
void kv_after_prefill(Device& d, const KvView& view, void* state, size_t context, size_t base, size_t count) {
    if (!view.staged) return;
    kv_copy_rows(d,static_cast<float*>(state),context,view.kv,view.capacity,base,count,kv_planes_of(d,state));
    if (view.kv==d.kv_stage && d.kv_stage_owner==state) d.kv_stage_valid=base+count;
}
// The layer map is fully registered before decode. Preparing here, rather than
// in qsa_decode_attend, also covers graphs reused after prefill or state restore.
void kv_prepare_span(Device& d, size_t begin, size_t end, size_t base, size_t count=1) {
    for (size_t layer=begin;layer<end;++layer) {
        auto plan=d.decoder.find(layer);
        if (plan!=d.decoder.end() && (layer%4==3 || layer==48)) {
            if (base>plan->second.context || count>plan->second.context-base)
                throw std::runtime_error("invalid Straydog K/V mirror position");
            kv_prepare_decode(d,plan->second.states[0],plan->second.context,base,count);
        }
    }
}
