// Decode spans keep the residual streams on device across contiguous layers.
// PLE remains at its existing CPU/GPU boundary between layers zero and one.
__global__ void decoder_residual(float* hidden,const float* mixed,const float* branch) { SD_KERNEL_ACQUIRE;
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index<10240) hidden[index]+=mixed[2560+index/2560]*branch[index%2560];
}

#if defined(STRAYDOG_PREFILL_OBSERVER)
// A validation observer is absent from normal execution. Downloads deliberately settle the stream: these are not performance runs.
void observe_prefill(Device& d, size_t layer, size_t phase, size_t position, size_t steps, size_t width, const float* values) {
    if (!d.prefill_observer) return;
    std::vector<float> rows(steps * width);
    download(d, rows.data(), values, rows.size() * sizeof(float));
    if (d.prefill_observer(d.prefill_observer_context, layer, phase, position, steps, width, rows.data()))
        throw std::runtime_error("prefill validation observer failed");
}

#define SD_PREFILL_OBSERVE(...) observe_prefill(__VA_ARGS__)
#else
#define SD_PREFILL_OBSERVE(...) ((void)0)
#endif

// Layer 48 is the MTP draft layer: full (QSA) attention like every fourth layer.
bool decoder_linear(size_t layer) { return layer % 4 != 3 && layer < 48; }

// parts: 2 registers this device's part `part` of a layer split across two GPUs (QwenHeads<2>: its rows of the projections,
// its columns of the output projection, its heads' states), which needs Device::qwen_parts set (sd_qwen_parts).
void register_decoder_layer(Device& d,size_t layer,size_t context,void* const* pointers,const int* kinds,
                             size_t count,void* const* states,void* workspace,void* moe,size_t parts=1,size_t part=0) {
    const bool linear=decoder_linear(layer);
    if (layer>48 || !context || context>262144 || !pointers || !kinds || !states ||
        count!=(linear ? 17 : 18) || d.decoder.find(layer)!=d.decoder.end() || d.resident.find(moe)==d.resident.end() ||
        (parts!=1 && parts!=2) || part>=parts || (parts==2 && (layer>48 || d.qwen_parts!=2 || !d.sums.peer)))
        throw std::runtime_error("invalid decoder layer registration");
    auto allocation=[&](void* pointer,size_t bytes) {
        auto found=d.allocations.find(pointer);
        if (found==d.allocations.end() || found->second!=bytes)
            throw std::runtime_error("invalid decoder layer allocation");
    };
    auto scratch=d.allocations.find(workspace);
    constexpr size_t extra=1024*2564+8;
    if (scratch==d.allocations.end() || scratch->second%4 || scratch->second/4<extra+15364 ||
        (scratch->second/4-extra)%15364)
        throw std::runtime_error("invalid decoder workspace allocation");
    const size_t capacity=(scratch->second/4-extra)/15364;
    if (capacity>4096 || capacity>d.resident.find(moe)->second->capacity)
        throw std::runtime_error("invalid decoder workspace capacity");
    const size_t hc_rows[4]={1,320,10240,4},hc_cols[4]={10240,10240,320,10240};
    const size_t gd_rows[9]={10240/parts,6144/parts,48/parts,48/parts,2560,10240/parts,1,1,1};
    const size_t gd_cols[9]={2560,2560,2560,2560,6144/parts,4,48/parts,48/parts,128};
    const size_t qs_rows[10]={12288/parts,512/parts,512/parts,512,128,2560,1,1,1,1};
    const size_t qs_cols[10]={2560,2560,2560,2560,2560,6144/parts,256,256,128,128};
    const size_t attention=linear ? 9 : 10;
    for (size_t i=0;i<count;++i) {
        size_t rows,cols;
        if (i<4 || i>=4+attention) {
            const size_t slot=i<4 ? i : i-4-attention;
            rows=hc_rows[slot]; cols=hc_cols[slot];
            if ((slot==0 && kinds[i]!=0) || (kinds[i]!=0 && kinds[i]!=1 && kinds[i]!=30 && !(kinds[i]==8 && (slot==1 || slot==2))))
                throw std::runtime_error("invalid decoder HC type");
        } else {
            const size_t slot=i-4;
            rows=linear ? gd_rows[slot] : qs_rows[slot]; cols=linear ? gd_cols[slot] : qs_cols[slot];
            if (slot>=(linear ? 5 : 6) && kinds[i]!=0)
                throw std::runtime_error("invalid decoder norm type");
        }
        allocation(pointers[i],rows*matrix_row_bytes(kinds[i],cols));
    }
    if (linear) { allocation(states[0],4*10240/parts*4); allocation(states[1],48/parts*128*128*4); }
    else {
        // (a K/V that grows: its indexer and scratch hold the whole context from the start)
        auto room=[&](void* pointer,size_t bytes) {
            auto found=d.allocations.find(pointer);
            if (found==d.allocations.end() || found->second<bytes)
                throw std::runtime_error("invalid decoder layer allocation");
        };
        allocation(states[0],context*4/parts*d.kv_row); room(states[1],((context/4+1)*128+qsa_history*128)*4);
        room(states[2],256*(context/4+1)*4);
    }
    DecoderLayer plan{}; plan.index=layer; plan.context=context; plan.capacity=capacity; plan.moe=moe;
    plan.parts=parts; plan.part=part;
    plan.workspace=static_cast<float*>(workspace);
    plan.pointers.assign(pointers,pointers+count); plan.kinds.assign(kinds,kinds+count);
    std::copy(states,states+3,plan.states);
    plan.dependencies=plan.pointers;
    plan.dependencies.push_back(workspace);
    for (size_t i=0;i<(linear ? 2 : 3);++i) plan.dependencies.push_back(states[i]);
    if (!linear && parts==2) {
        // (a part's K/V has its head's two planes: the decode mirrors are laid out anew)
        d.kv_planes[states[0]]=2;
        if (d.kv_stage) { kv_clear_graphs(d); kv_layout(d); }
    }
    const auto& expert=*d.resident.find(moe)->second;
    plan.dependencies.insert(plan.dependencies.end(),expert.dependencies.begin(),expert.dependencies.end());
    plan.dependencies.push_back(expert.table); plan.dependencies.push_back(expert.workspace);
    d.decoder.emplace(layer,std::move(plan));
}

// With prelaunch, layer L+1's graph is launched before the host computes layer L's
// CPU experts and starts by awaiting that result on the GPU, so no launch latency
// sits between the host's result and the GPU's next layer.
// Under a kernel run-time limit (Windows display GPUs) an await may last ~1 s at
// most, so layer L+1 goes ahead of layer L's CPU experts only where those read
// page-locked banks (they never wait for storage); otherwise it is launched once
// L's result is published.
bool hybrid_early(const Device& d,size_t layer) {
    return !d.await_watchdog || d.resident.find(d.decoder.find(layer)->second.moe)->second->mapped[0]!=nullptr;
}
bool hybrid_ahead(const Device& d,size_t begin,size_t end) {
    bool any_early=false;
    for (size_t layer=begin;layer<end;++layer) any_early|=hybrid_early(d,layer);
#if defined(__HIPCC__)
    // With high residency, the CPU fits inside the resident GPU work and the
    // extra wait dispatch loses. Use accumulated timings, not a VRAM cutoff:
    // prelaunch when CPU work is at least half the time waiting for exports.
    return d.hybrid_prelaunch && any_early && d.hybrid_cpu_ns>=d.hybrid_wait_ns/2;
#else
    return d.hybrid_prelaunch && any_early;
#endif
}

// Partially resident layers alternate a captured GPU part with the host CPU experts.
void hybrid_decode_span(Device& d,size_t begin,size_t end,size_t position,const float* input,float* output) {
    float* workspace=nullptr;
    for (size_t layer=begin;layer<end;++layer) {
        auto found=d.decoder.find(layer);
        if (found==d.decoder.end() || position>=found->second.context ||
            (workspace && workspace!=found->second.workspace) || !d.resident.find(found->second.moe)->second->hybrid)
            throw std::runtime_error("decoder span has no compatible hybrid layer");
        workspace=found->second.workspace;
    }
    kv_prepare_span(d,begin,end,position);
    float* hidden=workspace; float* mixed=hidden+10240; float* branch=mixed+2564;
    auto* gpu_position=reinterpret_cast<uint32_t*>(branch+2560);
    const uint32_t current[2]={uint32_t(position),uint32_t(position)-d.rope_delta};
    upload(d,hidden,input,10240*4);
    upload(d,gpu_position,current,sizeof(current));
    if (position!=d.hybrid_position) { d.hybrid_position=position; ++d.hybrid_tokens; }
    hybrid_settle(d,false);
    auto early=[&](size_t layer) { return hybrid_early(d,layer); };
    const bool ahead=hybrid_ahead(d,begin,end);
    HybridAwait await(d);
    auto* counters=reinterpret_cast<volatile uint32_t*>(d.exchange_device+hybrid_sequence_slot);
#if defined(__HIPCC__)
    // RDNA4's fine-grained exchange is coherent: finish can consume it after the
    // wait without a staging kernel's serialized PCIe reads.
    const bool stage_cpu_result=false;
#else
    const bool stage_cpu_result=ahead;
#endif
    const float* cpu_result=stage_cpu_result ? d.cpu_partial : d.exchange_device+hybrid_result_at(0);
    auto finish_previous=[&]() {
        if (ahead) hybrid_await<<<1,256,0,d.stream>>>(counters,d.exchange_device+hybrid_result_at(0),stage_cpu_result ? d.cpu_partial : nullptr,d.await_timeout);
        hybrid_finish<<<40,256,0,d.stream>>>(hidden,mixed,0,branch,cpu_result,0,hybrid_clock(d));
    };
    // One captured graph per layer: the previous layer's residual (adding the CPU
    // result), this layer up to routing, the export to pinned memory and its
    // completion count, then the resident experts.
    auto launch=[&](size_t layer) {
        const auto& plan=d.decoder.find(layer)->second;
        auto& experts=*d.resident.find(plan.moe)->second;
        auto* pointers=plan.pointers.data(); const int* kinds=plan.kinds.data();
        const size_t attention=decoder_linear(layer) ? 9 : 10;
        const bool finish=layer>begin;
        auto submit=[&]() {
            if (finish) finish_previous();
            hyperconnection(d,pointers,kinds,1,true,hidden,mixed,true);
            if (decoder_linear(layer))
                gated_delta(d,pointers+4,kinds+4,plan.states[0],plan.states[1],1,mixed,branch,true);
            else
                qsa_decode_device(d,pointers+4,kinds+4,plan.states[0],plan.states[1],plan.states[2],
                    plan.context,gpu_position,mixed,branch);
            decoder_residual<<<40,256,0,d.stream>>>(hidden,mixed,branch);
            hyperconnection(d,pointers+4+attention,kinds+4+attention,1,true,hidden,mixed,true);
            hybrid_route_export(d,experts,mixed);
            hybrid_tick<<<1,1,0,d.stream>>>(d.flags+1,reinterpret_cast<uint32_t*>(d.exchange_device+hybrid_sequence_slot),hybrid_clock(d),finish);
            hybrid_resident_experts(d,experts,mixed,branch);
            check(hipGetLastError());
        };
        // Counted before launching: publishing a result nobody awaits is harmless.
        if (ahead && finish) ++await.launched;
        run_graph(d,ahead ? "hybrid-await" : "hybrid-layer",plan.dependencies.data(),plan.dependencies.size(),
                  layer*2+finish,submit);
        flush_launches(d);
    };
    auto launch_after=[&](size_t layer) {
        if (layer+1<end) launch(layer+1);
        else { ++await.launched; finish_previous(); check(hipGetLastError()); flush_launches(d); }
    };
    if (ahead) launch(begin);
    for (size_t layer=begin;layer<end;++layer) {
        const bool before=ahead && early(layer);
        if (!ahead) launch(layer);
        else if (before) launch_after(layer);
        auto& experts=*d.resident.find(d.decoder.find(layer)->second.moe)->second;
        const auto waited=std::chrono::steady_clock::now();
        hybrid_wait_export(d,++d.hybrid_sequence);
        if (ahead) await.check_failed();
        const auto computed=std::chrono::steady_clock::now();
        hybrid_host_experts(d,experts,d.exchange,d.exchange+hybrid_result_at(0));
        if (ahead) await.publish();
        const auto finished=std::chrono::steady_clock::now();
        d.hybrid_wait_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(computed-waited).count());
        d.hybrid_cpu_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(finished-computed).count());
        if (ahead && !before) launch_after(layer);
    }
    if (!ahead) finish_previous();
    d.hybrid_layers+=end-begin; d.hybrid_timed_layers+=end-begin-1;
    check(hipGetLastError());
    download(d,output,hidden,10240*4);
    if (ahead) await.check_failed();
    // Follow decode routing: up to 32 swaps every four tokens, issued after this
    // device's last layer so the copies overlap the head and the next token's start.
    if ((end==48 || d.decoder.find(end)==d.decoder.end()) && d.hybrid_tokens%4==0) hybrid_rebalance(d,32,false);
}

void decode_span(Device& d,size_t begin,size_t end,size_t position,const float* input,float* output) {
    if (!input || !output || begin>=end || end>49) throw std::runtime_error("invalid decoder span");
    auto first=d.decoder.find(begin);
    if (first!=d.decoder.end() && d.resident.find(first->second.moe)->second->hybrid) {
        hybrid_decode_span(d,begin,end,position,input,output);
        return;
    }
    float* workspace=nullptr;
    std::vector<void*> dependencies;
    for (size_t layer=begin;layer<end;++layer) {
        auto found=d.decoder.find(layer);
        if (found==d.decoder.end() || position>=found->second.context ||
            (workspace && workspace!=found->second.workspace))
            throw std::runtime_error("decoder span has no compatible resident layer");
        workspace=found->second.workspace;
        dependencies.insert(dependencies.end(),found->second.dependencies.begin(),found->second.dependencies.end());
    }
    kv_prepare_span(d,begin,end,position);
    float* hidden=workspace; float* mixed=hidden+10240; float* branch=mixed+2564;
    auto* gpu_position=reinterpret_cast<uint32_t*>(branch+2560);
    const uint32_t current[2]={uint32_t(position),uint32_t(position)-d.rope_delta};
    upload(d,hidden,input,10240*4);
    upload(d,gpu_position,current,sizeof(current));
    auto submit=[&]() {
        // Both residual updates are fused: into the attention output projection and
        // into the expert combination, exactly as speculative verification does.
        const GemvResidual residual{hidden,mixed,0,0};
        for (size_t layer=begin;layer<end;++layer) {
            const auto& plan=d.decoder.find(layer)->second;
            auto* pointers=plan.pointers.data(); const int* kinds=plan.kinds.data();
            hyperconnection(d,pointers,kinds,1,true,hidden,mixed,true);
            const size_t attention=decoder_linear(layer) ? 9 : 10;
            if (decoder_linear(layer))
                gated_delta(d,pointers+4,kinds+4,plan.states[0],plan.states[1],1,mixed,nullptr,true,residual);
            else
                qsa_decode_device(d,pointers+4,kinds+4,plan.states[0],plan.states[1],plan.states[2],
                    plan.context,gpu_position,mixed,nullptr,residual);
            hyperconnection(d,pointers+4+attention,kinds+4+attention,1,true,hidden,mixed,true);
            moe_tokens(d,plan.moe,1,mixed,2564,hidden);
        }
        check(hipGetLastError());
    };
    if (span_has_parts(d,begin,end)) {
        run_with_helper(d,begin,end,1,[&]() {
            run_graph(d,"decode-span",dependencies.data(),dependencies.size(),begin*64+end,submit);
        });
        finish_with_helper(d);
    } else run_graph(d,"decode-span",dependencies.data(),dependencies.size(),begin*64+end,submit);
    download(d,output,hidden,10240*4);
}

// Logical prefill batches retain routed expert reuse while projections process
// their bounded physical chunks. Each CPU boundary is also a cancellation point.
__global__ void decoder_batch_residual(float* hidden,const float* mixed,const float* branch,size_t steps) {
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x,token=i/10240;
    if (token<steps) hidden[i]+=mixed[steps*2560+token*4+(i%10240)/2560]*branch[token*2560+i%2560];
}
struct PrefillTiming {
    hipEvent_t first=nullptr,last=nullptr;
    PrefillTiming() {
        check(hipEventCreate(&first));
        try { check(hipEventCreate(&last)); }
        catch (...) { cleanup(hipEventDestroy(first)); throw; }
    }
    ~PrefillTiming() { cleanup(hipEventDestroy(first)); cleanup(hipEventDestroy(last)); }
};
__global__ void decoder_nonfinite(const float* values,size_t count,unsigned* flag) {
    unsigned bad=0;
    for (size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<count;i+=size_t(gridDim.x)*256) bad|=!isfinite(values[i]);
    if (bad) atomicOr(flag,1u);
}
// Prefill keeps the hidden rows in this device's decoder workspace between spans.
// input==nullptr continues from the previous span; output receives only the
// trailing output_rows rows (zero leaves everything on the device).
void prefill_span(Device& d,size_t begin,size_t end,size_t position,size_t steps,const float* input,float* output,
                  size_t output_rows,double* timings=nullptr) {
    if (begin>=end || end>49 || !steps || steps>4096 || end-begin>4 || output_rows>steps || (output_rows && !output))
        throw std::runtime_error("invalid prefill span");
    float* workspace=nullptr;
    for (size_t layer=begin;layer<end;++layer) {
        auto found=d.decoder.find(layer);
        if (found==d.decoder.end() || steps>found->second.capacity || position>found->second.context ||
            steps>found->second.context-position || (workspace && workspace!=found->second.workspace))
            throw std::runtime_error("prefill span has no compatible resident layer");
        workspace=found->second.workspace;
    }
    float* hidden=workspace; float* mixed=hidden+steps*10240; float* branch=mixed+steps*2564;
    std::vector<std::unique_ptr<PrefillTiming>> events;
    std::vector<size_t> categories;
    if (timings) {
        std::fill(timings,timings+4,0);
        for (size_t i=0;i<(end-begin)*4;++i) events.push_back(std::make_unique<PrefillTiming>());
    }
    auto record=[&](size_t category,auto&& work) {
        const size_t index=categories.size();
        if (timings) check(hipEventRecord(events[index]->first,d.stream));
        work();
        if (timings) { check(hipEventRecord(events[index]->last,d.stream)); categories.push_back(category); }
    };
    if (input) upload(d,hidden,input,steps*10240*4);
    // The span's first host-K/V attention layer stages its prefix while the layers before it run.
    if (steps>1 && !d.layer_prefill_plan) kv_prefetch_from(d,begin,position);
    for (size_t layer=begin;layer<end;++layer) {
        const auto& plan=d.decoder.find(layer)->second;
        auto* pointers=plan.pointers.data(); const int* kinds=plan.kinds.data();
        // Between attention and MoE the expert workspace is idle; it holds batch rows.
        const auto& experts=*d.resident.find(plan.moe)->second;
        float* scratch=experts.workspace;
        if (steps*std::max({hc_batch_floats,gdn_batch_floats,qsa_batch_floats})>experts.capacity*49302+65536)
            throw std::runtime_error("prefill batch exceeds the resident expert workspace");
        const bool linear=decoder_linear(layer);
        SD_PREFILL_OBSERVE(d,layer,0,position,steps,10240,hidden);
        record(0,[&]() { hc_prefill(d,pointers,kinds,steps,hidden,mixed,scratch); });
        SD_PREFILL_OBSERVE(d,layer,1,position,steps,2560,mixed);
        record(linear ? 1 : 2,[&]() {
            if (linear) {
                const size_t chunk=d.input_capacity>=16*1024*1024 ? 512 : 128;
                gdn_prefill(d,pointers+4,kinds+4,plan.states[0],plan.states[1],steps,chunk,mixed,branch,scratch);
            } else {
                qsa_prefill(d,pointers+4,kinds+4,plan.states[0],plan.states[1],plan.states[2],
                            plan.context,position,steps,mixed,branch,scratch);
                if (steps>1 && !d.layer_prefill_plan) kv_prefetch_from(d,layer+1,position);
            }
        });
        SD_PREFILL_OBSERVE(d,layer,2,position,steps,2560,branch);
        decoder_batch_residual<<<unsigned((steps*10240+255)/256),256,0,d.stream>>>(hidden,mixed,branch,steps);
        SD_PREFILL_OBSERVE(d,layer,3,position,steps,10240,hidden);
        record(0,[&]() { hc_prefill(d,pointers+4+(linear ? 9 : 10),kinds+4+(linear ? 9 : 10),steps,hidden,mixed,scratch); });
        SD_PREFILL_OBSERVE(d,layer,4,position,steps,2560,mixed);
        record(3,[&]() {
            if (d.resident.find(plan.moe)->second->part) resident_moe_parts(d,plan.moe,layer,steps,mixed,branch);
            else resident_moe(d,plan.moe,steps,mixed,branch,true);
        });
        SD_PREFILL_OBSERVE(d,layer,5,position,steps,2560,branch);
        decoder_batch_residual<<<unsigned((steps*10240+255)/256),256,0,d.stream>>>(hidden,mixed,branch,steps);
        SD_PREFILL_OBSERVE(d,layer,6,position,steps,10240,hidden);
    }
    const bool parts=span_has_parts(d,begin,end);
    if (d.prefill_deferred && !output_rows && !timings) {
        decoder_nonfinite<<<256,256,0,d.stream>>>(hidden,steps*10240,d.flags+1);
        check(hipGetLastError());
        return;
    }
    check(hipMemsetAsync(d.flags,0,4,d.stream));
    decoder_nonfinite<<<256,256,0,d.stream>>>(hidden,steps*10240,d.flags);
    check(hipGetLastError());
    if (parts) finish_with_helper(d);
    unsigned flag=0;
    download(d,&flag,d.flags,4);
    if (output_rows) {
        // (the flag's read above waited for the span: what follows is the rows' copy alone)
        const auto started=std::chrono::steady_clock::now();
        download(d,output,hidden+(steps-output_rows)*10240,output_rows*10240*4);
        d.rows_down_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count());
        d.rows_down_bytes+=uint64_t(output_rows)*10240*4;
    }
    if (flag) throw std::runtime_error("Straydog GPU produced non-finite decoder state");
    for (size_t i=0;i<categories.size();++i) {
        float elapsed=0; check(hipEventElapsedTime(&elapsed,events[i]->first,events[i]->last));
        timings[categories[i]]+=double(elapsed)/1000.0;
    }
}

#undef SD_PREFILL_OBSERVE
