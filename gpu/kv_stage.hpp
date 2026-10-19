// One allocation, two layouts: prefill stages one layer; decode mirrors every
// registered host state's prefix. All copies preserve the four F32 segments.
struct KvView {
    float* kv; size_t capacity; bool staged;
    float* mirror = nullptr; size_t mirror_rows = 0;
};
void kv_copy_rows(Device& d, float* to, size_t to_capacity, const float* from, size_t from_capacity,
                  size_t first, size_t count) {
    for (size_t segment=0;count && segment<4;++segment)
        check(hipMemcpyAsync(to+(segment*to_capacity+first)*256,from+(segment*from_capacity+first)*256,count*256*4,
                             hipMemcpyDefault,d.stream));
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
    const KvView view=kv_decode_view(d,state,context);
    kv_invalidate_write(d,state,base);
    if (!view.mirror) return view;
    d.kv_stage_owner=nullptr; d.kv_stage_valid=0;
    auto& mirror=d.kv_mirrors.find(state)->second;
    const size_t end=std::min(base,view.mirror_rows);
    const size_t valid=std::min(mirror.valid,end);
    kv_copy_rows(d,view.mirror,view.mirror_rows,view.kv,context,valid,end-valid);
    mirror.valid=std::min(base+count,view.mirror_rows);
    return view;
}
KvView kv_for_prefill(Device& d, void* state, size_t context, size_t base, size_t count) {
    if (count==1) return kv_prepare_decode(d,state,context,base);
    kv_invalidate_write(d,state,base);
    if (!d.kv_stage || base+count>d.kv_stage_rows || !d.host_states.count(state))
        return {static_cast<float*>(state),context,false};
    // The prefill layout overwrites every layer's decode mirror.
    for (auto& entry : d.kv_mirrors) entry.second.valid=0;
    if (d.kv_stage_owner!=state) { d.kv_stage_owner=state; d.kv_stage_valid=0; }
    const size_t valid=std::min(d.kv_stage_valid,base);
    kv_copy_rows(d,d.kv_stage,d.kv_stage_rows,static_cast<const float*>(state),context,valid,base-valid);
    return {d.kv_stage,d.kv_stage_rows,true};
}
void kv_after_prefill(Device& d, const KvView& view, void* state, size_t context, size_t base, size_t count) {
    if (!view.staged) return;
    kv_copy_rows(d,static_cast<float*>(state),context,view.kv,view.capacity,base,count);
    d.kv_stage_valid=base+count;
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
