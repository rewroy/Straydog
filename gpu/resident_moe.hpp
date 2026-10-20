// Fully resident execution is an optional budget-checked path. Expert banks stay
// packed; routing, grouping, every top-10 expert and shared expert remain on GPU.
__global__ void resident_mix(Matrix matrix, const float* input, float* output) {
    const size_t token=blockIdx.x,lane=threadIdx.x;
    float sum=0;
    for (size_t col=lane;col<2560;col+=256) sum+=grouped_weight(matrix,0,col)*input[token*2560+col];
    __shared__ float partial[256]; partial[lane]=sum; __syncthreads();
    for (unsigned offset=128;offset;offset/=2) {
        if (lane<offset) partial[lane]+=partial[lane+offset];
        __syncthreads();
    }
    if (lane==0) output[token]=partial[0];
}

// Exact top-10 of up to 512 router scores per token, ordered by value then lower
// index: each wave extracts its 64-value top-10 with shuffles, then one wave merges
// the 80 candidates. Softmax over the selected scores matches the prior sort.
struct RouteCandidate { float value; int index; };
__device__ __forceinline__ bool route_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__device__ __forceinline__ void route_best(float& value, int& index) {
    for (unsigned offset = 16; offset; offset /= 2) {
        const float other = __shfl_xor(value, offset, 32);
        const int other_index = __shfl_xor(index, offset, 32);
        if (route_better(other, other_index, value, index)) { value = other; index = other_index; }
    }
}
// Token t reads scores + t*row_stride and mix[t*mix_stride] (defaults: packed rows).
__global__ void resident_route(const float* scores,const float* mix,size_t experts,
                               int32_t* selected,float* probabilities,size_t row_stride=0,size_t mix_stride=1) {
    const size_t token=blockIdx.x,lane=threadIdx.x%32,wave=threadIdx.x/32;
    if (!row_stride) row_stride=experts;
    __shared__ RouteCandidate candidates[8*10];
    float v[2]; int ix[2];
    for (int j=0;j<2;++j) {
        const size_t i=wave*64+lane+32*j;
        ix[j]=int(i); v[j]=i<experts ? scores[token*row_stride+i] : -INFINITY;
    }
    for (int round=0;round<10;++round) {
        const bool first=route_better(v[0],ix[0],v[1],ix[1]);
        float value=first ? v[0] : v[1]; int index=first ? ix[0] : ix[1];
        route_best(value,index);
        if (lane==0) candidates[wave*10+round]={value,index};
        for (int j=0;j<2;++j) if (ix[j]==index) { v[j]=-INFINITY; ix[j]=0x7fffffff; }
    }
    __syncthreads();
    if (wave) return;
    float c[3]; int ci[3];
    for (int j=0;j<3;++j) {
        const size_t i=lane+32*j;
        c[j]=i<80 ? candidates[i].value : -INFINITY; ci[j]=i<80 ? candidates[i].index : 0x7fffffff;
    }
    float top[10]; int top_index[10];
    for (int round=0;round<10;++round) {
        int best=0;
        for (int j=1;j<3;++j) if (route_better(c[j],ci[j],c[best],ci[best])) best=j;
        float value=c[best]; int index=ci[best];
        route_best(value,index);
        top[round]=value; top_index[round]=index;
        for (int j=0;j<3;++j) if (ci[j]==index) { c[j]=-INFINITY; ci[j]=0x7fffffff; }
    }
    if (lane==0) {
        float sum=0,weights[10];
        for (size_t i=0;i<10;++i) { weights[i]=expf(top[i]-top[0]); sum+=weights[i]; }
        for (size_t i=0;i<10;++i) {
            selected[token*10+i]=top_index[i]<int(experts) ? top_index[i] : int(i);
            probabilities[token*11+i]=weights[i]/sum;
        }
        probabilities[token*11+10]=hc_sigmoid(mix[token*mix_stride]);
    }
}

// Decode expert GEMV: ten routed experts plus the shared expert, one dispatch per stage.
// Routed slices of a bank share kind and shape; every role keeps its own lanes.
struct ExpertGemv {
    uint32_t routed_waves[2], routed_lanes[2], shared_waves[2], shared_lanes[2];
    // Zero-copy assist: the first `assist` absent routes read page-locked host banks.
    const uint8_t* mapped[3]; uint64_t bytes[3]; uint32_t assist;
};
// Whether the GPU evaluates `route`: resident, or among the first assisted absent routes.
__device__ __forceinline__ const uint8_t* expert_source(const MoeGroup* groups,const int32_t* selected,size_t route,
                                                        size_t role,const ExpertGemv& plan) {
    const size_t expert=size_t(selected[route]);
    const uint8_t* data=groups[expert].weights[role].data;
    if (data || !plan.assist || !plan.mapped[role]) return data;
    // Balance the PCIe-bound GPU share against the CPU: about 5 of every 12 absent routes.
    uint32_t earlier=0,absent=0;
    for (size_t r=0;r<10;++r) {
        const bool missing=groups[selected[r]].weights[0].data==nullptr;
        absent+=missing; earlier+=missing && r<route;
    }
    const uint32_t share=min(plan.assist,(5*absent+4)/12);
    return earlier<share ? plan.mapped[role]+expert*plan.bytes[role] : nullptr;
}
// blockIdx.y selects a token: its routes, its input row and its output block.
__global__ void resident_expert_gemv(const MoeGroup* groups,const int32_t* selected,size_t experts,bool down,
                                     ExpertGemv plan,const float* input,float* output,
                                     size_t input_stride=0,size_t output_stride=0) {
    size_t wave=size_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    selected+=size_t(blockIdx.y)*10; input+=size_t(blockIdx.y)*input_stride; output+=size_t(blockIdx.y)*output_stride;
    const size_t roles=down ? 1 : 2;
    size_t route=0,role=0,local=0; unsigned lanes=0; bool found=false;
    for (size_t r=0;r<roles && !found;++r) {
        const size_t span=10*size_t(plan.routed_waves[r]);
        if (wave<span) { route=wave/plan.routed_waves[r]; local=wave%plan.routed_waves[r]; role=r; lanes=plan.routed_lanes[r]; found=true; }
        else wave-=span;
    }
    for (size_t r=0;r<roles && !found;++r) {
        if (wave<plan.shared_waves[r]) { route=10; local=wave; role=r; lanes=plan.shared_lanes[r]; found=true; }
        else wave-=plan.shared_waves[r];
    }
    if (!found) return;
    if (down) role=2;
    const size_t expert=route<10 ? size_t(selected[route]) : experts;
    Matrix matrix=groups[expert].weights[role];
    if (route<10) matrix.data=expert_source(groups,selected,route,role,plan);
    if (!matrix.data) return;   // a partially resident expert evaluated on the host
    matrix.input_offset=down ? route*640 : 0;
    matrix.output_offset=down ? route*2560 : route*1280+role*640;
    gemv_wave<1>(matrix,local,lanes,input,0,output,0);
}
// Lanes for every routed expert of a bank: all slices share kind, shape and stride.
// A stage runs ten routed experts per role (and the shared expert) side by side.
unsigned resident_lanes(const std::vector<Matrix>& weights,size_t role,const float* input,size_t concurrent) {
    const unsigned first=gemv_lanes(weights[role],input,concurrent), second=gemv_lanes(weights[3+role],input,concurrent);
    return first==second ? first : 0;
}
void resident_expert_launch(Device& d,const ResidentMoe& plan,const MoeGroup* groups,const int32_t* selected,
                            bool down,const float* input,float* output,
                            size_t tokens=1,size_t input_stride=0,size_t output_stride=0) {
    const auto& weights=plan.weights;
    const size_t shared=plan.experts*3, roles=down ? 1 : 2;
    ExpertGemv geometry{};
    for (size_t role=0;role<3;++role) { geometry.mapped[role]=plan.mapped[role]; geometry.bytes[role]=plan.host_bytes[role]; }
    geometry.assist=plan.mapped[0] ? d.assist : 0;
    size_t waves=0;
    for (size_t r=0;r<roles;++r) {
        const size_t role=down ? 2 : r;
        geometry.routed_lanes[r]=resident_lanes(weights,role,input,10*roles);
        geometry.routed_waves[r]=uint32_t(gemv_waves(weights[role],geometry.routed_lanes[r]));
        Matrix matrix=weights[shared+role];
        if (down) matrix.input_offset=10*640;
        geometry.shared_lanes[r]=gemv_lanes(matrix,input,10*roles);
        geometry.shared_waves[r]=uint32_t(gemv_waves(matrix,geometry.shared_lanes[r]));
        waves+=10*size_t(geometry.routed_waves[r])+geometry.shared_waves[r];
    }
    resident_expert_gemv<<<dim3(unsigned((waves+7)/8),unsigned(tokens)),256,0,d.stream>>>(
        groups,selected,plan.experts,down,geometry,input,output,input_stride,output_stride);
    check(hipGetLastError());
}

__global__ void resident_clear(MoeGroup* groups,unsigned* cursors,size_t count) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index<count) { groups[index].count=0; cursors[index]=0; }
}
__global__ void resident_histogram(MoeGroup* groups,const int32_t* selected,size_t steps,size_t experts) {
    const size_t route=size_t(blockIdx.x)*256+threadIdx.x;
    if (route>=steps*11) return;
    const size_t expert=route<steps*10 ? size_t(selected[route]) : experts;
    atomicAdd(reinterpret_cast<unsigned long long*>(&groups[expert].count),1ull);
}
__global__ void resident_prefix(MoeGroup* groups,size_t count,bool half,size_t minimum,
                                MoeTile* tiles,size_t limit,bool rdna4,unsigned* total) {
    size_t first=0,half_tile=0,scalar_tile=0;
    for (size_t group=0;group<count;++group) {
        groups[group].first=first; groups[group].half=int(half && groups[group].count>=minimum);
        for (size_t row=0;row<groups[group].count;row+=rdna4 && groups[group].half && groups[group].count>=32 ? 32 : 16) {
            if (rdna4 && groups[group].half) tiles[half_tile++]={uint32_t(group),uint32_t(row)};
            else tiles[limit+scalar_tile++]={uint32_t(group),uint32_t(row)};
        }
        first+=groups[group].count;
    }
    total[0]=unsigned(half_tile); total[1]=unsigned(scalar_tile);
}
__global__ void resident_scatter(const MoeGroup* groups,unsigned* cursors,const int32_t* selected,
                                 size_t steps,size_t experts,int32_t* positions) {
    const size_t route=size_t(blockIdx.x)*256+threadIdx.x;
    if (route>=steps*11) return;
    const size_t expert=route<steps*10 ? size_t(selected[route]) : experts;
    const unsigned offset=atomicAdd(cursors+expert,1u);
    positions[groups[expert].first+offset]=int32_t(route);
}

// Banks hold routed experts [first, first + count); a part plan (expert parallelism)
// leaves the others with null data, and without `shared` the shared
// expert too (null pointers 3-5), so this device evaluates only its own routes.
void* create_resident_moe(Device& d,void* const* pointers,const int* kinds,size_t experts,
                          void* workspace,size_t capacity,size_t first=0,size_t count=SIZE_MAX,bool shared=true) {
    if (count==SIZE_MAX) count=experts;
    if (!pointers || !kinds || experts<10 || experts>512 || !capacity || capacity>4096 || !count ||
        first>experts || count>experts-first)
        throw std::runtime_error("invalid resident MoE plan");
    auto storage=d.allocations.find(workspace);
    if (storage==d.allocations.end() || storage->second<(capacity*49302+65536)*4)
        throw std::runtime_error("invalid resident MoE workspace");
    auto plan=std::make_unique<ResidentMoe>(); plan->experts=experts; plan->capacity=capacity;
    plan->part=count<experts || !shared; plan->shared=shared;
    plan->workspace=static_cast<float*>(workspace);
    const size_t rows[8]={640,640,2560,640,640,2560,experts,1};
    const size_t cols[8]={2560,2560,640,2560,2560,640,2560,2560};
    Matrix matrices[8]{};
    for (size_t i=0;i<8;++i) {
        const size_t row_bytes=matrix_row_bytes(kinds[i],cols[i]);
        if (!shared && i>=3 && i<6) {
            if (pointers[i]) throw std::runtime_error("invalid resident MoE part");
        } else {
            auto found=d.allocations.find(pointers[i]);
            if (found==d.allocations.end() || found->second!=(i<3 ? count : 1)*rows[i]*row_bytes)
                throw std::runtime_error("invalid resident MoE bank size");
            plan->dependencies.push_back(pointers[i]);
        }
        matrices[i]={static_cast<const uint8_t*>(pointers[i]),row_bytes,rows[i],cols[i],0,0,0,kinds[i]};
    }
    for (size_t expert=0;expert<experts;++expert)
        for (size_t role=0;role<3;++role) {
            Matrix matrix=matrices[role];
            const bool held=expert>=first && expert-first<count;
            matrix.data=held ? matrix.data+(expert-first)*matrix.rows*matrix.row_bytes : nullptr;
            plan->weights.push_back(matrix);
        }
    for (size_t i=3;i<8;++i) plan->weights.push_back(matrices[i]);
    std::vector<MoeGroup> groups(experts+1);
    for (size_t group=0;group<=experts;++group)
        groups[group]={{plan->weights[group*3],plan->weights[group*3+1],plan->weights[group*3+2]},0,0,0};
    const size_t bytes=groups.size()*sizeof(MoeGroup);
    if (bytes>d.budget-d.used) throw std::runtime_error("resident MoE metadata exceeds VRAM budget");
    void* table=nullptr; check(hipMalloc(&table,bytes));
    try {
        upload(d,table,groups.data(),bytes);
        check(hipStreamSynchronize(d.stream));
    } catch (...) { cleanup(hipFree(table)); throw; }
    d.allocations.emplace(table,bytes); d.used+=bytes; plan->table=table;
    void* handle=plan.get(); d.resident.emplace(handle,std::move(plan));
    return handle;
}

// Prefill experts on RDNA4: device grouping, one host read of the expert counts,
// then exactly sized gate/up and down tile launches. Gate/up stores FP16
// SiLU(gate)*up per grouped row in the gate/up scratch; down scatters per route.
#if defined(__HIPCC__)
// The resident plans' layer: hidden 2560, experts 640 wide, ten routes per token (fixed at compile time
// in their expert launches).
constexpr MoeShape resident_shape{2560,640,10};
// Formats of a group's gate, up and down banks (routed experts share theirs).
inline MoeKinds moe_kinds(const ResidentMoe& plan,size_t group) {
    return {plan.weights[group*3].kind,plan.weights[group*3+1].kind,plan.weights[group*3+2].kind};
}
// The tile list below, built on the device so that the host need not wait for the routing: the
// groups' first rows, then each present group's rows in blocks of moe_block_m, in group order (a
// part's absent groups, the peer's experts, have no weights). `count` receives the number.
// One block of moe_tile_threads threads, each scanning a run of groups.
constexpr unsigned moe_tile_threads=256;
__global__ void __launch_bounds__(moe_tile_threads) moe_device_tiles(const MoeGroup* groups,size_t groups_count,
                                                                   const unsigned* counts,uint32_t* firsts,MoeTile* tiles,
                                                                   unsigned* count) {
    __shared__ unsigned rows[moe_tile_threads],blocks[moe_tile_threads];
    const unsigned t=threadIdx.x,per=unsigned((groups_count+moe_tile_threads-1)/moe_tile_threads);
    const size_t begin=std::min(groups_count,size_t(t)*per),end=std::min(groups_count,begin+per);
    unsigned own_rows=0,own_blocks=0;
    for (size_t group=begin;group<end;++group) {
        own_rows+=counts[group];
        if (groups[group].weights[0].data) own_blocks+=(counts[group]+moe_block_m-1)/moe_block_m;
    }
    rows[t]=own_rows; blocks[t]=own_blocks;
    __syncthreads();
    // Inclusive scans of both, then each thread's exclusive start.
    for (unsigned offset=1;offset<moe_tile_threads;offset*=2) {
        const unsigned r=t>=offset ? rows[t-offset] : 0,b=t>=offset ? blocks[t-offset] : 0;
        __syncthreads();
        rows[t]+=r; blocks[t]+=b;
        __syncthreads();
    }
    unsigned first=rows[t]-own_rows,tile=blocks[t]-own_blocks;
    for (size_t group=begin;group<end;++group) {
        firsts[group]=first; first+=counts[group];
        if (!groups[group].weights[0].data) continue;
        for (unsigned row=0;row<counts[group];row+=moe_block_m) tiles[tile++]=MoeTile{uint32_t(group),row};
    }
    if (t==moe_tile_threads-1) *count=blocks[t];
}
#endif
void grouped_expert_prefill(Device& d,ResidentMoe& plan,MoeGroup* groups,size_t experts,size_t steps,const int32_t* selected,
                            int32_t* positions,unsigned* cursors,const float* input,float* scratch,float* outputs) {
    const size_t routes=steps*11, groups_count=experts+1;
    unsigned* counts=cursors+((experts+2)&~size_t(1));
    uint32_t* firsts=counts+groups_count;
    auto* tiles=reinterpret_cast<MoeTile*>(firsts+((groups_count+1)&~size_t(1)));
    const size_t max_tiles=(routes+moe_block_m-1)/moe_block_m+groups_count;
    if (steps*10*4+(((experts+2)&~size_t(1))+2*groups_count+2)*4+(max_tiles+1)*sizeof(MoeTile)>65536*4)
        throw std::runtime_error("resident MoE grouping scratch overflow");
    resident_clear<<<unsigned((experts+256)/256),256,0,d.stream>>>(groups,cursors,groups_count);
    resident_histogram<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,selected,steps,experts);
    moe_counts<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,counts);
#if defined(__HIPCC__)
    if (!plan.hybrid) {
        // All on the device: the host queues the experts (and then the peer's half, the next
        // layer) without waiting for this layer's routing.
        auto* tile_count=reinterpret_cast<unsigned*>(tiles+max_tiles);
        moe_device_tiles<<<1,moe_tile_threads,0,d.stream>>>(groups,groups_count,counts,firsts,tiles,tile_count);
        moe_set_groups<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,firsts);
        resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,cursors,selected,steps,experts,positions);
        const unsigned shared=plan.weights[experts*3].data ? unsigned((steps+moe_block_m-1)/moe_block_m) : 0;
        // FP16 products (routes*1280 bytes), then the FP16 token inputs (steps*5120), within the gate/up scratch.
        auto* product=reinterpret_cast<_Float16*>(scratch);
        auto* half_input=product+routes*640;
        moe_input_half<<<unsigned((steps*2560+1023)/1024),256,0,d.stream>>>(steps*2560,input,half_input);
        moe_expert_launch<2560,640,10>(d.stream,moe_kinds(plan,0),moe_kinds(plan,experts),groups,tiles,positions,steps,resident_shape,half_input,
                          product,outputs,tile_count,unsigned(max_tiles),shared);
        check(hipGetLastError());
        return;
    }
#endif
    std::vector<unsigned> host_counts(groups_count);
    download(d,host_counts.data(),counts,groups_count*4);
    const size_t tail=plan.hybrid ? std::min(steps,hybrid_prompt_tail) : 0;
    std::vector<int32_t> host_tail(tail*10);
    if (tail) download(d,host_tail.data(),selected+(steps-tail)*10,tail*10*4);
    std::vector<uint32_t> host_firsts(groups_count);
    std::vector<size_t> missing;
    std::vector<bool> absent(groups_count,false);
    if (plan.part)
        for (size_t group=0;group<groups_count;++group) absent[group]=plan.weights[group*3].data==nullptr;
    if (plan.hybrid) {
        hybrid_observe_batch(plan,host_counts.data(),steps,host_tail.data(),tail);
        for (size_t expert=0;expert<experts;++expert)
            if (host_counts[expert] && plan.slot_of[expert]<0) { missing.push_back(expert); absent[expert]=true; }
    }
    size_t first=0;
    for (size_t group=0;group<groups_count;++group) { host_firsts[group]=uint32_t(first); first+=host_counts[group]; }
    if (first!=routes) throw std::runtime_error("inconsistent Straydog expert grouping");
    // Resident groups (and the shared expert) first; then each streamed expert's tiles in order.
    std::vector<MoeTile> host_tiles;
    host_tiles.reserve(max_tiles);
    auto add=[&](size_t group) {
        for (size_t row=0;row<host_counts[group];row+=moe_block_m) host_tiles.push_back({uint32_t(group),uint32_t(row)});
    };
    for (size_t group=0;group<groups_count;++group) if (!absent[group]) add(group);
    const size_t resident_tiles=host_tiles.size();
    std::vector<size_t> offsets{resident_tiles};
    for (size_t expert : missing) { add(expert); offsets.push_back(host_tiles.size()); }
    if (host_tiles.size()>max_tiles) throw std::runtime_error("inconsistent Straydog expert grouping");
    upload(d,firsts,host_firsts.data(),groups_count*4);
    if (!host_tiles.empty()) upload(d,tiles,host_tiles.data(),host_tiles.size()*sizeof(MoeTile));
    moe_set_groups<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,firsts);
    resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,cursors,selected,steps,experts,positions);
#if defined(__HIPCC__)
    auto* product=reinterpret_cast<_Float16*>(scratch);
#else
    auto* product=reinterpret_cast<__half*>(scratch);
#endif
    // FP16 product uses routes*1280 bytes, then the converted token inputs use
    // steps*5120. Both fit before outputs (routes*7680 bytes from scratch).
    auto* half_input=product+routes*640;
    moe_input_half<<<unsigned((steps*2560+1023)/1024),256,0,d.stream>>>(steps*2560,input,half_input);
    auto launch=[&](size_t begin,size_t end) {
        if (end<=begin) return;
#if defined(__HIPCC__)
        // The resident list ends with the shared expert's tiles.
        const unsigned shared=begin==0 && end==resident_tiles && !absent[experts] ?
            unsigned((host_counts[experts]+moe_block_m-1)/moe_block_m) : 0;
        moe_expert_launch<2560,640,10>(d.stream,moe_kinds(plan,0),moe_kinds(plan,experts),groups,tiles+begin,positions,steps,resident_shape,half_input,
                          product,outputs,nullptr,unsigned(end-begin),shared);
#else
        // Banks have one format per role. Separate the final shared-expert
        // tiles so its format cannot inflate the routed Q2 kernel's registers.
        auto cuda_launch=[&](size_t start,size_t finish,size_t group) {
            if (finish<=start) return;
            const dim3 gate_grid(10,unsigned(finish-start),moe_block_m/moe_cuda_m);
            const dim3 down_grid(20,unsigned(finish-start),moe_block_m/moe_cuda_m);
            if (plan.weights[group*3].kind==42 && plan.weights[group*3+1].kind==42)
                moe_gate_up_mma<true><<<gate_grid,256,0,d.stream>>>(groups,tiles+start,positions,steps,half_input,product);
            else
                moe_gate_up_mma<false><<<dim3(10,unsigned(finish-start)),256,0,d.stream>>>(groups,tiles+start,positions,steps,half_input,product);
            if (plan.weights[group*3+2].kind==42)
                moe_down_mma<true><<<down_grid,256,0,d.stream>>>(groups,tiles+start,positions,product,outputs);
            else
                moe_down_mma<false><<<dim3(20,unsigned(finish-start)),256,0,d.stream>>>(groups,tiles+start,positions,product,outputs);
        };
        if (begin==0 && end==resident_tiles) {
            const size_t shared_begin=end-(host_counts[experts]+moe_block_m-1)/moe_block_m;
            cuda_launch(begin,shared_begin,0);
            cuda_launch(shared_begin,end,experts);
        } else cuda_launch(begin,end,0);
#endif
        check(hipGetLastError());
    };
    launch(0,resident_tiles);
    hybrid_stream_missing(d,plan,missing,[&](size_t begin,size_t count) { launch(offsets[begin],offsets[begin+count]); });
    // Host vectors reach the device through the staging arena, so resident experts need no wait
    // here and the caller can queue the next work (the peer's half, the next layer) at once.
    if (plan.hybrid) check(hipStreamSynchronize(d.stream));
}

// moe_combine for per-token blocks of 11 expert outputs (10 routed, then shared).
// moe_combine_blocks followed by the decoder residual update: token t adds
// mixed[t*stride + 2560 + s] times its combined row to stream s of hidden row t.
__global__ void moe_combine_residual(const float* outputs,const float* probabilities,const float* mixed,size_t stride,
                                     float* hidden) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x,token=blockIdx.y;
    if (index>=10240) return;
    const size_t col=index%2560;
    float sum=0;
    for (size_t expert=0;expert<10;++expert)
        sum+=probabilities[token*11+expert]*outputs[(token*11+expert)*2560+col];
    const float combined=sum+probabilities[token*11+10]*outputs[(token*11+10)*2560+col];
    hidden[token*10240+index]+=mixed[token*stride+2560+index/2560]*combined;
}
// Up to four decode tokens (HyperConnection rows input_stride apart: 2560 inputs
// then the four stream mixes) with the one-token arithmetic: every kernel takes
// all tokens, the expert GEMVs per token, and the result is added to hidden rows.
// Decode-token workspace: routes then per-route outputs, probabilities and the top ten.
struct TokenMoe {
    ResidentMoe* plan; MoeGroup* groups; float* outputs; float* probabilities; int32_t* selected;
};
TokenMoe token_moe(Device& d,void* handle,size_t tokens) {
    auto found=d.resident.find(handle);
    if (found==d.resident.end() || !tokens || tokens>4 || tokens>found->second->capacity || found->second->hybrid)
        throw std::runtime_error("invalid resident MoE token batch");
    ResidentMoe& plan=*found->second;
    const size_t routes=tokens*11;
    float* outputs=plan.workspace+routes*1280+routes*640; float* probabilities=outputs+routes*2560;
    return {&plan,static_cast<MoeGroup*>(plan.table),outputs,probabilities,
            reinterpret_cast<int32_t*>(probabilities+tokens*11)+routes};
}
// Route the tokens and evaluate every route whose expert this device holds.
void moe_tokens_experts(Device& d,const TokenMoe& moe,size_t tokens,const float* input,size_t input_stride) {
    const ResidentMoe& plan=*moe.plan;
    const size_t experts=plan.experts,routes=tokens*11;
    Matrix projections[2]={plan.weights[(experts+1)*3],plan.weights[(experts+1)*3+1]};
    projections[1].output_offset=experts;   // each token: router scores then mix
    float* gate_up=plan.workspace; float* product=gate_up+routes*1280;
    gemv(d,projections,2,input,d.output,tokens,input_stride,experts+1);
    resident_route<<<unsigned(tokens),256,0,d.stream>>>(d.output,d.output+experts,experts,moe.selected,moe.probabilities,
                                                        experts+1,experts+1);
    resident_expert_launch(d,plan,moe.groups,moe.selected,false,input,gate_up,tokens,input_stride,11*1280);
    moe_product<<<unsigned((routes*640+255)/256),256,0,d.stream>>>(gate_up,routes,product);
    resident_expert_launch(d,plan,moe.groups,moe.selected,true,product,moe.outputs,tokens,11*640,11*2560);
}
void resident_moe_tokens(Device& d,void* handle,size_t tokens,const float* input,size_t input_stride,float* hidden) {
    const TokenMoe moe=token_moe(d,handle,tokens);
    if (moe.plan->part) throw std::runtime_error("a resident MoE part needs its peer");
    moe_tokens_experts(d,moe,tokens,input,input_stride);
    moe_combine_residual<<<dim3(40,unsigned(tokens)),256,0,d.stream>>>(moe.outputs,moe.probabilities,input,input_stride,hidden);
    check(hipGetLastError());
}

// Expert parallelism. Route outputs are laid out per token as eleven
// slots (decode: ten routed, then shared) or as all routed rows then all shared rows
// (prefill; the same for one token). A device's partial sum adds, in slot order, the
// routed outputs of the experts it holds. The owner then adds the two partials in
// device order and the shared expert after them.
__device__ __forceinline__ const float* moe_route_output(const float* outputs,size_t steps,bool prefill,
                                                         size_t token,size_t route) {
    return outputs+((prefill ? token*10 : token*11)+route)*2560;
}
__device__ __forceinline__ float moe_own_partial(const float* outputs,const float* probabilities,const int32_t* selected,
                                                 const MoeGroup* groups,size_t steps,bool prefill,size_t token,size_t col) {
    float sum=0;
    for (size_t route=0;route<10;++route)
        if (groups[selected[token*10+route]].weights[2].data)
            sum+=probabilities[token*11+route]*moe_route_output(outputs,steps,prefill,token,route)[col];
    return sum;
}
__global__ void moe_partial(const float* outputs,const float* probabilities,const int32_t* selected,
                            const MoeGroup* groups,size_t steps,bool prefill,float* partial) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index>=steps*2560) return;
    partial[index]=moe_own_partial(outputs,probabilities,selected,groups,steps,prefill,index/2560,index%2560);
}
// combined = (partial of device 0 + partial of device 1) + shared; decode adds it to the
// hidden streams (hidden != null), prefill writes it to `result`.
__global__ void moe_combine_parts(const float* outputs,const float* probabilities,const int32_t* selected,
                                  const MoeGroup* groups,size_t steps,bool prefill,const float* peer,unsigned index,
                                  const float* mixed,size_t stride,float* hidden,float* result) {
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x,token=blockIdx.y;
    if (i>=(hidden ? 10240 : 2560) || token>=steps) return;
    const size_t col=i%2560;
    const float own=moe_own_partial(outputs,probabilities,selected,groups,steps,prefill,token,col);
    const float other=peer[token*2560+col];
    const float* shared=prefill ? outputs+(steps*10+token)*2560 : outputs+(token*11+10)*2560;
    const float combined=(index==0 ? own+other : other+own)+probabilities[token*11+10]*shared[col];
    if (hidden) hidden[token*10240+i]+=mixed[token*stride+2560+i/2560]*combined;
    else result[token*2560+col]=combined;
}
// The owner's MoE for up to four decode tokens: its peer evaluates the other experts
// (helper_moe_tokens, in the peer's graph) at the same time.
void resident_moe_tokens_parts(Device& d,void* handle,size_t tokens,const float* input,size_t input_stride,float* hidden) {
    const TokenMoe moe=token_moe(d,handle,tokens);
    if (!moe.plan->part || !moe.plan->shared || !d.link.peer) throw std::runtime_error("invalid resident MoE part");
    peer_send_rows(d,input,tokens,input_stride);
    moe_tokens_experts(d,moe,tokens,input,input_stride);
    peer_receive(d);
    moe_combine_parts<<<dim3(40,unsigned(tokens)),256,0,d.stream>>>(moe.outputs,moe.probabilities,moe.selected,moe.groups,
        tokens,false,peer_partials(d.link),d.link.index,input,input_stride,hidden,nullptr);
    check(hipGetLastError());
}
// A decode layer's MoE: the whole plan here, or this device's part with its peer's.
void moe_tokens(Device& d,void* handle,size_t tokens,const float* input,size_t input_stride,float* hidden) {
    auto found=d.resident.find(handle);
    if (found!=d.resident.end() && found->second->part) resident_moe_tokens_parts(d,handle,tokens,input,input_stride,hidden);
    else resident_moe_tokens(d,handle,tokens,input,input_stride,hidden);
}
// The helper's side for one layer: the owner's rows arrive in this device's inbox and its
// partial sums go straight into the owner's.
void helper_moe_tokens(Device& h,size_t layer,size_t tokens) {
    auto found=h.helper_moe.find(layer);
    if (found==h.helper_moe.end() || !h.link.peer) throw std::runtime_error("no resident MoE part helps this layer");
    const TokenMoe moe=token_moe(h,found->second,tokens);
    peer_receive_rows(h);
    moe_tokens_experts(h,moe,tokens,peer_inputs(h.link),2560);
    moe_partial<<<unsigned((tokens*2560+255)/256),256,0,h.stream>>>(moe.outputs,moe.probabilities,moe.selected,
        moe.groups,tokens,false,peer_remote_partials(h.link));
    peer_signal_peer(h);
    check(hipGetLastError());
}
std::vector<void*> helper_dependencies(Device& h,size_t begin,size_t end) {
    std::vector<void*> dependencies{h.link.inbox,h.link.peer_inbox};
    for (size_t layer=begin;layer<end;++layer) {
        auto found=h.helper_moe.find(layer);
        if (found==h.helper_moe.end()) continue;
        const auto& plan=*h.resident.find(found->second)->second;
        dependencies.insert(dependencies.end(),plan.dependencies.begin(),plan.dependencies.end());
        dependencies.push_back(plan.table); dependencies.push_back(plan.workspace);
    }
    return dependencies;
}
// The helper's graph for a span the owner runs: one MoE part per layer that has one.
void helper_span(Device& h,size_t begin,size_t end,size_t tokens) {
    const auto dependencies=helper_dependencies(h,begin,end);
    run_graph(h,"helper-span",dependencies.data(),dependencies.size(),(begin*64+end)*8+tokens,[&]() {
        for (size_t layer=begin;layer<end;++layer)
            if (h.helper_moe.count(layer)) helper_moe_tokens(h,layer,tokens);
    });
}
// Whether the owner runs [begin, end) with its peer helping: every layer's MoE is a part.
bool span_has_parts(Device& d,size_t begin,size_t end) {
    bool parts=false;
    for (size_t layer=begin;layer<end;++layer) {
        auto found=d.decoder.find(layer);
        if (found==d.decoder.end()) return false;
        const bool part=d.resident.find(found->second.moe)->second->part;
        if (layer>begin && part!=parts) throw std::runtime_error("a decoder span mixes whole and partial experts");
        parts=part;
    }
    if (parts && (!d.link.peer || !d.link.peer->helper_moe.count(begin)))
        throw std::runtime_error("a resident MoE part has no linked peer");
    return parts;
}
// The owner runs its graph while the peer runs the helper's; both streams finish here.
template <class Owner>
void run_with_helper(Device& d,size_t begin,size_t end,size_t tokens,Owner&& owner) {
    Device& h=*d.link.peer;
    on_device(h.ordinal,[&]() { helper_span(h,begin,end,tokens); });
    owner();
}
void finish_with_helper(Device& d) {
    Device& h=*d.link.peer;
    on_device(h.ordinal,[&]() { check(hipStreamSynchronize(h.stream)); check(hipStreamSynchronize(h.helper_stream)); });
    check(hipStreamSynchronize(d.stream)); check(hipStreamSynchronize(d.helper_stream));
    peer_settle(d);
}
// Prefill workspace: routes in the prefill layout (moe_route_output), the probabilities,
// the top ten and the grouping scratch.
struct StepMoe {
    ResidentMoe* plan; MoeGroup* groups; float* gate_up; float* outputs; float* probabilities;
    int32_t* positions; int32_t* selected; unsigned* cursors; unsigned* total; MoeTile* tiles; size_t max_tiles;
};
StepMoe step_moe(Device& d,void* handle,size_t steps) {
    auto found=d.resident.find(handle);
    if (found==d.resident.end() || !steps || steps>found->second->capacity)
        throw std::runtime_error("invalid resident MoE execution");
    ResidentMoe& plan=*found->second;
    if (plan.hybrid && (steps==1 || !(d.expert_prefill && d.half_gemm)))
        throw std::runtime_error("partially resident experts need the hybrid decoder or the grouped FP16 prefill path");
    if (plan.part && steps>1 && !(d.expert_prefill && d.half_gemm))
        throw std::runtime_error("a resident MoE part needs the grouped FP16 prefill path");
    const size_t experts=plan.experts,routes=steps*11;
    StepMoe moe{};
    moe.plan=&plan; moe.groups=static_cast<MoeGroup*>(plan.table);
    moe.gate_up=plan.workspace; float* product=moe.gate_up+routes*1280;
    moe.outputs=product+routes*640; moe.probabilities=moe.outputs+routes*2560;
    moe.positions=reinterpret_cast<int32_t*>(moe.probabilities+steps*11);
    moe.selected=moe.positions+routes;
    moe.cursors=reinterpret_cast<unsigned*>(moe.selected+steps*10);
    moe.total=moe.cursors+((experts+2)&~size_t(1));
    moe.tiles=reinterpret_cast<MoeTile*>(moe.total+2);
    moe.max_tiles=(routes+15)/16+experts+1;
    if (steps*10*4+((experts+2)&~size_t(1))*4+8+2*moe.max_tiles*sizeof(MoeTile)>65536*4 ||
        steps*2560>d.input_capacity || steps*2560>d.output_capacity)
        throw std::runtime_error("resident MoE scratch overflow");
    return moe;
}
// Route `steps` rows and evaluate every route whose expert this device holds.
// The routing of `steps` rows: router, mix and the top ten into moe.selected and moe.probabilities.
void moe_steps_route(Device& d,const StepMoe& moe,size_t steps,const float* source) {
    const ResidentMoe& plan=*moe.plan;
    const size_t experts=plan.experts;
    const Matrix router=plan.weights[(experts+1)*3],mix=plan.weights[(experts+1)*3+1];
    Matrix descriptors[max_batch]{};
    if (steps==1) {
        Matrix projections[2]={router,mix};
        projections[1].output_offset=experts;
        gemv(d,projections,2,source,d.output);
    } else {
        hc_project(d,router,steps,source,d.output,descriptors);
        resident_mix<<<unsigned(steps),256,0,d.stream>>>(mix,source,d.output+steps*experts);
    }
    resident_route<<<unsigned(steps),256,0,d.stream>>>(d.output,d.output+steps*experts,experts,moe.selected,moe.probabilities);
}
// Every routed row whose expert this device holds, from moe.selected. Only the plan's own
// workspace and table: a helper stream can run it beside the device's other work.
void moe_steps_compute(Device& d,const StepMoe& moe,size_t steps,const float* source) {
    const ResidentMoe& plan=*moe.plan;
    const size_t experts=plan.experts,routes=steps*11,max_tiles=moe.max_tiles;
    float* gate_up=moe.gate_up; float* product=gate_up+routes*1280; float* outputs=moe.outputs;
    MoeGroup* groups=moe.groups; MoeTile* tiles=moe.tiles; unsigned* total=moe.total;
    if (steps==1) {
        resident_expert_launch(d,plan,groups,moe.selected,false,source,gate_up);
        moe_product<<<unsigned((routes*640+255)/256),256,0,d.stream>>>(gate_up,routes,product);
        resident_expert_launch(d,plan,groups,moe.selected,true,product,outputs);
    } else if (d.expert_prefill && d.half_gemm) {
        grouped_expert_prefill(d,*moe.plan,groups,experts,steps,moe.selected,moe.positions,moe.cursors,source,gate_up,outputs);
    } else {
        resident_clear<<<unsigned((experts+256)/256),256,0,d.stream>>>(groups,moe.cursors,experts+1);
        resident_histogram<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,moe.selected,steps,experts);
        resident_prefix<<<1,1,0,d.stream>>>(groups,experts+1,d.half_gemm,d.gemm_min_batch,tiles,max_tiles,d.rdna4,total);
        resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,moe.cursors,moe.selected,steps,experts,moe.positions);
        if (d.rdna4 && d.half_gemm) {
            grouped_moe_wmma<16><<<dim3(10,unsigned(max_tiles),2),128,0,d.stream>>>(groups,tiles,moe.positions,steps,false,source,gate_up,total,2);
            grouped_moe_wmma<32><<<dim3(10,unsigned(max_tiles),2),256,0,d.stream>>>(groups,tiles,moe.positions,steps,false,source,gate_up,total);
        }
        grouped_moe<<<dim3(10,unsigned(max_tiles),2),256,0,d.stream>>>(groups,tiles+max_tiles,moe.positions,steps,false,source,gate_up,total+1);
        moe_product<<<unsigned((routes*640+255)/256),256,0,d.stream>>>(gate_up,routes,product);
        if (d.rdna4 && d.half_gemm) {
            grouped_moe_wmma<16><<<dim3(40,unsigned(max_tiles),1),128,0,d.stream>>>(groups,tiles,moe.positions,steps,true,product,outputs,total,2);
            grouped_moe_wmma<32><<<dim3(40,unsigned(max_tiles),1),256,0,d.stream>>>(groups,tiles,moe.positions,steps,true,product,outputs,total);
        }
        grouped_moe<<<dim3(40,unsigned(max_tiles),1),256,0,d.stream>>>(groups,tiles+max_tiles,moe.positions,steps,true,product,outputs,total+1);
    }
}
void moe_steps_experts(Device& d,const StepMoe& moe,size_t steps,const float* source) {
    moe_steps_route(d,moe,steps,source);
    moe_steps_compute(d,moe,steps,source);
}
void resident_moe(Device& d,void* handle,size_t steps,const float* input,float* output,bool device_io=false) {
    if (!input || !output) throw std::runtime_error("invalid resident MoE execution");
    const StepMoe moe=step_moe(d,handle,steps);
    if (moe.plan->part) throw std::runtime_error("a resident MoE part needs its peer");
    // Device-resident inputs are read in place and results written straight to the caller.
    const float* source=input;
    if (!device_io) { upload(d,d.input,input,steps*2560*4); source=d.input; }
    float* result=device_io ? output : d.output;
    auto submit=[&]() {
        moe_steps_experts(d,moe,steps,source);
        moe_combine<<<unsigned((steps*2560+255)/256),256,0,d.stream>>>(moe.outputs,moe.probabilities,steps,result);
        check(hipGetLastError());
    };
    std::vector<void*> dependencies=moe.plan->dependencies;
    dependencies.push_back(moe.plan->table); dependencies.push_back(moe.plan->workspace);
    // Multi-token routing reads expert counts on the host, so it is never captured.
    if (!device_io && steps==1)
        run_graph(d,"resident-moe",dependencies.data(),dependencies.size(),steps,submit);
    else submit();
    if (!device_io) download(d,output,d.output,steps*2560*4);
}
// Prefill MoE with the peer evaluating the other experts (device rows in and out). Each
// device's grouping reads its expert counts on the host and then queues its experts without
// waiting for them, so the owner's experts (with the shared expert, the larger half) are
// queued first and run while the helper routes and evaluates its own.
void resident_moe_parts(Device& d,void* handle,size_t layer,size_t steps,const float* input,float* output) {
    const StepMoe moe=step_moe(d,handle,steps);
    if (!moe.plan->part || !moe.plan->shared || !d.link.peer || steps>d.link.rows)
        throw std::runtime_error("invalid resident MoE part");
    Device& h=*d.link.peer;
    auto helper=h.helper_moe.find(layer);
    if (helper==h.helper_moe.end()) throw std::runtime_error("no resident MoE part helps this layer");
    // The owner routes; its top ten go with the rows, so the helper needs no router (and none of
    // the device scratch a router uses) and runs on its helper stream beside its own batch.
    moe_steps_route(d,moe,steps,input);
    check(hipMemcpyAsync(d.link.peer_routes,moe.selected,steps*10*4,hipMemcpyDefault,d.stream));
    check(hipMemcpyAsync(d.link.peer_routes+steps*10,moe.probabilities,steps*11*4,hipMemcpyDefault,d.stream));
    peer_send_rows(d,input,steps,2560);
    moe_steps_compute(d,moe,steps,input);
    on_device(h.ordinal,[&]() {
        StreamScope helping(h,h.helper_stream);
        peer_receive_rows(h);
        // A help workspace smaller than the batch takes it in pieces: every row's experts are
        // its own, so the partial sums are those of one pass.
        const ResidentMoe& plan=*h.resident.find(helper->second)->second;
        const int32_t* selected=h.link.routes;
        const float* probabilities=reinterpret_cast<const float*>(h.link.routes+steps*10);
        for (size_t first=0;first<steps;first+=plan.capacity) {
            const size_t count=std::min(plan.capacity,steps-first);
            StepMoe part=step_moe(h,helper->second,count);
            // The piece's routes in the piece's layout: its top ten, then its probabilities.
            check(hipMemcpyAsync(part.selected,selected+first*10,count*10*4,hipMemcpyDeviceToDevice,h.stream));
            check(hipMemcpyAsync(part.probabilities,probabilities+first*11,count*11*4,hipMemcpyDeviceToDevice,h.stream));
            moe_steps_compute(h,part,count,peer_inputs(h.link)+first*2560);
            moe_partial<<<unsigned((count*2560+255)/256),256,0,h.stream>>>(part.outputs,part.probabilities,part.selected,
                part.groups,count,true,peer_remote_partials(h.link)+first*2560);
        }
        peer_signal_peer(h);
        check(hipGetLastError());
    });
    peer_receive(d);
    moe_combine_parts<<<dim3(10,unsigned(steps)),256,0,d.stream>>>(moe.outputs,moe.probabilities,moe.selected,moe.groups,
        steps,true,peer_partials(d.link),d.link.index,nullptr,0,nullptr,output);
    check(hipGetLastError());
}
