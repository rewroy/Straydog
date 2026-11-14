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
                               int32_t* selected,float* probabilities,size_t row_stride=0,size_t mix_stride=1) { SD_KERNEL_ACQUIRE;
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
    const uint8_t* stage; uint64_t slot_bytes;   // CUDA: assisted experts are read from VRAM's staging slots (hybrid_stage_assist)
    // CUDA: the pass's staging plan (hybrid_tick_plan): of each row's ten routes the staging slot its expert was given, or -1 (resident, or left to
    // the CPU). One slot to a distinct missed expert of the pass, whatever rows route to it. A kernel moves it to its row (expert_row_stage).
    const int32_t* assign;
    // CUDA: which of the GPU's routes a launch evaluates: 0 all, 1 the resident ones alone, 2 the staged ones alone (3: all but the experts whose
    // gate and up are in SoA slots, which the INT8 kernels took). A layer whose staging copy runs beside its
    // resident experts (hybrid_resident_experts) launches the resident ones first and the staged ones after the copy; each route's tiles are the same tiles
    // in either launch, so the sums are those of one launch for all.
    uint32_t only;
};
// Whether the GPU evaluates `route`: resident, or among the first assisted absent routes.
// (No array of the plan is read at an index the thread computes, here or in expert_wave: with one such read nvcc keeps the whole plan, a
// kernel's argument, in every thread's stack and writes it there at the kernel's entry -- 17 stores of local memory before the first branch
// of resident_expert_gemv_kind, 136 bytes a thread, some 30 MB a launch of a four-row down, as much as the weights it reads.)
__device__ __forceinline__ const uint8_t* expert_source(const MoeGroup* groups,const int32_t* selected,size_t route,
                                                        size_t role,const ExpertGemv& plan) {
    const size_t expert=size_t(selected[route]);
    const uint8_t* data=groups[expert].weights[role].data;
    const uint8_t* mapped=role==0 ? plan.mapped[0] : role==1 ? plan.mapped[1] : plan.mapped[2];
    const uint64_t bytes=role==0 ? plan.bytes[0] : role==1 ? plan.bytes[1] : plan.bytes[2];
    if (data) {
        if (plan.only==2) return nullptr;
        // (3: the INT8 kernels took the experts whose gate and up are in SoA slots, resident_expert_q8.hpp)
        if (plan.only==3 && groups[expert].weights[0].layout && groups[expert].weights[1].layout) return nullptr;
        return data;
    }
    if (plan.only==1 || !plan.assist || !mapped) return nullptr;
    if (plan.assign) {
        const int32_t slot=plan.assign[route];
        if (slot<0) return nullptr;
        if (plan.stage) return plan.stage+size_t(slot)*plan.slot_bytes+(role>0 ? plan.bytes[0] : 0)+(role>1 ? plan.bytes[1] : 0);
        return mapped+expert*bytes;
    }
    // Balance the PCIe-bound GPU share against the CPU: about 5 of every 12 absent routes.
    uint32_t earlier=0,absent=0;
    for (size_t r=0;r<10;++r) {
        const bool missing=groups[selected[r]].weights[0].data==nullptr;
        absent+=missing; earlier+=missing && r<route;
    }
    const uint32_t share=min(plan.assist,(5*absent+4)/12);
    if (earlier>=share) return nullptr;
    if (plan.stage) return plan.stage+earlier*plan.slot_bytes+(role>0 ? plan.bytes[0] : 0)+(role>1 ? plan.bytes[1] : 0);
    return mapped+expert*bytes;
}
// A row's assisted experts stage after the earlier rows' (hybrid_stage_assist).
__device__ __forceinline__ void expert_row_stage(ExpertGemv& plan) {
    if (plan.assign) plan.assign+=size_t(blockIdx.y)*10;
    else if (plan.stage) plan.stage+=size_t(blockIdx.y)*plan.assist*plan.slot_bytes;
}
// A stage's wave: its route (10: the shared expert), role, wave within the matrix and lanes; false past the stage's waves.
// (The roles one after another, written out: a loop over them would read the plan's arrays at its counter, see expert_source.)
__device__ __forceinline__ bool expert_wave(size_t wave,bool down,const ExpertGemv& plan,size_t& route,size_t& role,size_t& local,
                                            unsigned& lanes) {
    const size_t first=plan.routed_waves[0], second=down ? 0 : plan.routed_waves[1];
    if (wave<10*first) { route=wave/first; local=wave%first; role=down ? 2 : 0; lanes=plan.routed_lanes[0]; return true; }
    wave-=10*first;
    if (!down) {
        if (wave<10*second) { route=wave/second; local=wave%second; role=1; lanes=plan.routed_lanes[1]; return true; }
        wave-=10*second;
    }
    if (wave<plan.shared_waves[0]) { route=10; local=wave; role=down ? 2 : 0; lanes=plan.shared_lanes[0]; return true; }
    wave-=plan.shared_waves[0];
    if (!down) {
        if (wave<plan.shared_waves[1]) { route=10; local=wave; role=1; lanes=plan.shared_lanes[1]; return true; }
    }
    return false;
}
// The wave's matrix (null data: a partially resident expert the host evaluates) at its input and output places.
__device__ __forceinline__ Matrix expert_matrix(const MoeGroup* groups,const int32_t* selected,size_t experts,bool down,
                                                const ExpertGemv& plan,size_t route,size_t role) {
    const size_t expert=route<10 ? size_t(selected[route]) : experts;
    Matrix matrix=groups[expert].weights[role];
    if (route<10) {
        matrix.data=expert_source(groups,selected,route,role,plan);
        // An absent route's mapped or staging image is still the host's native layout.
        if (!groups[expert].weights[role].data) matrix.layout=0;
    }
    matrix.input_offset=down ? route*640 : 0;
    matrix.output_offset=down ? route*2560 : route*1280+role*640;
    return matrix;
}
// blockIdx.y selects a token: its routes, its input row and its output block.
__global__ void resident_expert_gemv(const MoeGroup* groups,const int32_t* selected,size_t experts,bool down,
                                     ExpertGemv plan,const float* input,float* output,
                                     size_t input_stride=0,size_t output_stride=0) { SD_KERNEL_ACQUIRE;
    const size_t wave=size_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    selected+=size_t(blockIdx.y)*10; input+=size_t(blockIdx.y)*input_stride; output+=size_t(blockIdx.y)*output_stride;
    expert_row_stage(plan);
    size_t route=0,role=0,local=0; unsigned lanes=0;
    if (!expert_wave(wave,down,plan,route,role,local,lanes)) return;
    const Matrix matrix=expert_matrix(groups,selected,experts,down,plan,route,role);
    if (!matrix.data) return;
    resident_wave<1>(matrix,local,lanes,input,0,output,0);
}
// resident_expert_gemv of a stage whose routed slices take one format, `routed` (gate and up alike): only that format's F32
// rows are compiled, and the shared expert's down rows of a light format, so the kernel keeps few registers (the kernel of
// every format took 186 VGPRs, eight waves a SIMD) and a gate or up row takes more lanes (its 80 units walked by 8 lanes at
// 10 each, not 4 at 20). The shared expert's gate and up go beside it (resident_expert_launch); their Q8 units here took 204.
#define SD_EXPERT_ROUTED_FORMATS(X) X(12) X(13) X(14) X(18) X(20) X(21) X(22) X(23) X(42)
#define SD_EXPERT_SHARED_DOWN(X) X(8) X(20)
#define SD_EXPERT_LISTED(format) format,
template<int routed>
__global__ void __launch_bounds__(256) resident_expert_gemv_kind(const MoeGroup* groups,const int32_t* selected,size_t experts,bool down,
                                                                 ExpertGemv plan,const float* input,float* output,
                                                                 size_t input_stride,size_t output_stride) { SD_KERNEL_ACQUIRE;
    const size_t wave=size_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    selected+=size_t(blockIdx.y)*10; input+=size_t(blockIdx.y)*input_stride; output+=size_t(blockIdx.y)*output_stride;
    expert_row_stage(plan);
    size_t route=0,role=0,local=0; unsigned lanes=0;
    if (!expert_wave(wave,down,plan,route,role,local,lanes)) return;
    const Matrix matrix=expert_matrix(groups,selected,experts,down,plan,route,role);
    if (!matrix.data) return;
    const size_t first=local*(32/lanes);
    if (route<10) { resident_rows<routed,1>(matrix,first,lanes,input,0,output,0,{}); return; }
    #define SD_EXPERT_DOWN(format) case format: gemv_rows<format,1>(matrix,first,lanes,input,0,output,0,{}); break;
    switch (matrix.kind) { SD_EXPERT_SHARED_DOWN(SD_EXPERT_DOWN) }
    #undef SD_EXPERT_DOWN
}
// A layer part's gate and up stage (qwen_parts.hpp) with moe_product's product formed in the same wave: a wave's lane groups take
// rows j.. of a route's gate and the same rows of its up, each row summed as gemv_rows sums it, and the gate's lanes form
// silu(gate) * up, a launch a layer less; then waves form the shared expert's product from the gate and up its router's launch
// formed (moe_tokens_experts). The same sums and products as resident_expert_gemv_kind's and moe_product's.
template<int routed>
__global__ void __launch_bounds__(256) resident_expert_pairs(const MoeGroup* groups,const int32_t* selected,unsigned waves_per_route,
                                                             unsigned lanes,const float* input,size_t input_stride,const float* gate_up,
                                                             float* product,size_t shared_cols) { SD_KERNEL_ACQUIRE;
    const size_t wave=size_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32, token=blockIdx.y;
    selected+=token*10; input+=token*input_stride;
    const float* own=gate_up+token*11*1280; float* products=product+token*11*640;
    const unsigned lane=threadIdx.x%32, group=lane/lanes, sub=lane%lanes, half=32/lanes/2;
    if (wave>=10*size_t(waves_per_route)) {
        const size_t i=(wave-10*size_t(waves_per_route))*32+lane;
        if (i<shared_cols) {
            const float gate=own[10*1280+i];
            products[10*640+i]=gate*hc_sigmoid(gate)*own[10*1280+640+i];
        }
        return;
    }
    const size_t route=wave/waves_per_route, row=(wave%waves_per_route)*half+group%half;
    const Matrix& matrix=groups[selected[route]].weights[group<half ? 0 : 1];
    if (!matrix.data) return;
    float sum;
    if constexpr (routed==16 || routed==18 || routed==21 || routed==22) {
        if (matrix.layout) sum=resident_pair_sum<routed,true>(matrix,row,sub,lanes,input);
        else sum=resident_pair_sum<routed,false>(matrix,row,sub,lanes,input);
    } else sum=resident_pair_sum<routed,false>(matrix,row,sub,lanes,input);
    for (unsigned offset=lanes/2;offset;offset/=2) sum+=__shfl_xor(sum,offset,32);
    const float up=__shfl(sum,int((lane+half*lanes)%32),32);
    if (group<half && sub==0 && row<matrix.rows) products[route*640+row]=sum*hc_sigmoid(sum)*up;
}
// Lanes for every routed expert of a bank: all slices share kind, shape and stride.
// A stage runs ten routed experts per role (and the shared expert) side by side.
unsigned resident_lanes(const std::vector<Matrix>& weights,size_t role,const float* input,size_t concurrent) {
    const unsigned first=gemv_lanes(weights[role],input,concurrent), second=gemv_lanes(weights[3+role],input,concurrent);
    return first==second ? first : 0;
}
// The assist level a layer's routes run at (the experts a pass may take over the bus, ExpertGemv::assist). On CUDA the MTP draft layer's missed experts go
// to the CPU whatever the level: they cost the step nothing there (the layer has few CPU routes and the hand-off overlaps the GPU's own experts), where a
// staged one costs the GPU about 95 us in series inside the step, and a draft step is a few of those on every hit. DS41_PROFILE containing "dg" gives the
// draft layer the level as any other (the measurement of the difference).
inline size_t hybrid_level(const Device& d,const ResidentMoe& plan) {
#if !defined(__HIPCC__)
    static const bool drafter_takes=[] { const char* p=std::getenv("DS41_PROFILE"); return p && std::strstr(p,"dg"); }();
    if (plan.drafter && !drafter_takes) return 0;
#else
    (void)plan;
#endif
    return d.assist;
}
#include "resident_expert_q8.hpp"
// shared_done: the shared expert's gate and up are in `output` already (moe_tokens_experts' router launch).
void resident_expert_launch(Device& d,const ResidentMoe& plan,const MoeGroup* groups,const int32_t* selected,
                            bool down,const float* input,float* output,
                            size_t tokens=1,size_t input_stride=0,size_t output_stride=0,bool shared_done=false,
                            const uint8_t* assist_stage=nullptr,uint64_t assist_slot_bytes=0,const int32_t* assist_assign=nullptr,unsigned only=0) {
    const auto& weights=plan.weights;
    const size_t shared=plan.experts*3, roles=down ? 1 : 2;
    const bool with_shared=plan.shared && !(shared_done && !down);
    ExpertGemv geometry{};
    for (size_t role=0;role<3;++role) { geometry.mapped[role]=plan.mapped[role]; geometry.bytes[role]=plan.host_bytes[role]; }
    geometry.assist=plan.mapped[0] ? hybrid_level(d,plan) : 0;
    geometry.stage=assist_stage; geometry.slot_bytes=assist_slot_bytes; geometry.assign=assist_assign; geometry.only=only;
    // The routed gate and up of the experts in SoA slots on INT8 activations (resident_expert_q8.hpp). The shared expert's gate and up go to the
    // dense GEMV on the side stream, forked before the routed launches and joined after them, so that the two run together: behind them on the
    // device's stream they added 11 to 33 us a launch (RTX 3090, 1 to 4 rows). What else the GPU evaluates of the launch -- a staged expert,
    // whose image is the host's native layout -- stays with the kernels below, which then take those routes alone (ExpertGemv::only 3) and no
    // shared expert.
    bool q8=false;
    if (!down && only!=2 && resident_gate_up_q8_takes(d,plan,input,tokens,input_stride)) {
        if (with_shared) {
            // (a HIP build's development switch: the stream made at its first use, as the layers in parts make it)
            if (!d.side_stream) {
                check(hipStreamCreate(&d.side_stream));
                check(hipEventCreateWithFlags(&d.side_fork,hipEventDisableTiming));
                check(hipEventCreateWithFlags(&d.side_join,hipEventDisableTiming));
            }
            check(hipEventRecord(d.side_fork,d.stream));
            check(hipStreamWaitEvent(d.side_stream,d.side_fork,0));
            {
                StreamScope side(d,d.side_stream);
                Matrix pair[2]={weights[shared],weights[shared+1]};
                pair[1].output_offset=640;
                gemv(d,pair,2,input,output+10*1280,tokens,input_stride ? input_stride : 2560,output_stride ? output_stride : 11*1280);
            }
            check(hipEventRecord(d.side_join,d.side_stream));
        }
        resident_gate_up_q8_launch(d,plan,groups,selected,geometry,input,output,tokens,input_stride,output_stride);
        if (with_shared) check(hipStreamWaitEvent(d.stream,d.side_join,0));
        // (a partially resident plan's resident experts are all in slots: a launch of them alone has nothing left, nor has any launch while
        // the GPU takes no expert over the bus -- the F32 kernel launched for nothing took 28 to 41 us of an RTX 3090's four-row pass)
        if (plan.hybrid && (only==1 || !geometry.assist)) return;
        q8=true;
        geometry.only=3;
    }
    // The routed down on INT8 products (resident_expert_q8.hpp): every route the GPU evaluates, a resident expert's or a staged one's, so that
    // the kernels below are left the shared expert's down alone (no routed wave in their grid).
    bool q8_down=false;
    if (down && resident_down_q8_takes(d,plan,input,tokens,input_stride)) {
        resident_down_q8_launch(d,plan,groups,selected,geometry,input,output,tokens,input_stride,output_stride);
        if (!with_shared || only==2) return;
        q8_down=true;
    }
    // The kernel of the stage's routed format (resident_expert_gemv_kind) when it has one: its shared down rows of a light
    // format; the shared gate and up (on the Q8 GEMV: Device::q8_wide) on a stream of their own beside it, joined after.
    const int routed=weights[down ? 2 : 0].kind;
    const auto in=[](int kind,std::initializer_list<int> kinds) { return std::find(kinds.begin(),kinds.end(),kind)!=kinds.end(); };
    bool kind=in(routed,{12,13,14,18,20,21,22,23,42}) && (down || weights[1].kind==routed) && (!down || !with_shared ||
              in(weights[shared+2].kind,{8,20})) && (down || (d.q8_wide && d.rdna4));
    size_t waves=0;
    for (size_t r=0;r<roles;++r) {
        const size_t role=down ? 2 : r;
        geometry.routed_lanes[r]=resident_lanes(weights,role,input,kind && !down ? 4 : 10*roles);
        geometry.routed_waves[r]=q8_down ? 0 : uint32_t(gemv_waves(weights[role],geometry.routed_lanes[r]));
        Matrix matrix=weights[shared+role];
        if (down) matrix.input_offset=10*640;
        geometry.shared_lanes[r]=gemv_lanes(matrix,input,10*roles);
        // (a part without the shared expert has none of its waves; nor a gate and up stage that runs it beside)
        geometry.shared_waves[r]=with_shared && !(kind && !down) ? uint32_t(gemv_waves(matrix,geometry.shared_lanes[r])) : 0;
        waves+=10*size_t(geometry.routed_waves[r])+geometry.shared_waves[r];
        kind&=geometry.routed_lanes[r]!=0;
    }
    // (the staged routes alone: no shared expert in the launch, and the routed tiles as they are in the launch for all)
    if (only==2 || q8) {
        waves=0;
        for (size_t r=0;r<roles;++r) { geometry.shared_waves[r]=0; waves+=10*size_t(geometry.routed_waves[r]); }
    }
    const bool beside=kind && !down && with_shared && only!=2 && !q8;
    // (Layers in parts, qwen_parts.hpp, run it first on the device's stream: on the side stream, every pass of three or
    // four rows took 7.5 ms more at 20K.)
    // (Past four rows, not on it either: such a pass left every pass after it ~4.3 ms longer, whatever its rows.)
    const bool aside=beside && d.qwen_parts==1 && tokens<=4;
    if (beside && !aside) {
        Matrix pair[2]={weights[shared],weights[shared+1]};
        pair[1].output_offset=640;
        gemv(d,pair,2,input,output+10*1280,tokens,input_stride ? input_stride : 2560,output_stride ? output_stride : 11*1280);
    } else if (aside) {
        if (!d.side_stream) {
            check(hipStreamCreate(&d.side_stream));
            check(hipEventCreateWithFlags(&d.side_fork,hipEventDisableTiming));
            check(hipEventCreateWithFlags(&d.side_join,hipEventDisableTiming));
        }
        check(hipEventRecord(d.side_fork,d.stream));
        check(hipStreamWaitEvent(d.side_stream,d.side_fork,0));
        {
            StreamScope side(d,d.side_stream);
            Matrix pair[2]={weights[shared],weights[shared+1]};
            pair[1].output_offset=640;
            gemv(d,pair,2,input,output+10*1280,tokens,input_stride ? input_stride : 2560,output_stride ? output_stride : 11*1280);
        }
        check(hipEventRecord(d.side_join,d.side_stream));
    }
    const dim3 grid(unsigned((waves+7)/8),unsigned(tokens));
    #define SD_EXPERT_ROUTED(format) \
        case format: resident_expert_gemv_kind<format><<<grid,256,0,d.stream>>>(groups,selected,plan.experts,down,geometry,input,output, \
                                                                                  input_stride,output_stride); break;
    switch (kind ? routed : -1) {
        SD_EXPERT_ROUTED_FORMATS(SD_EXPERT_ROUTED)
        default:
            resident_expert_gemv<<<grid,256,0,d.stream>>>(groups,selected,plan.experts,down,geometry,input,output,input_stride,
                                                          output_stride);
    }
    #undef SD_EXPERT_ROUTED
    check(hipGetLastError());
    if (aside) check(hipStreamWaitEvent(d.stream,d.side_join,0));
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
    if (storage==d.allocations.end() || storage->second<moe_workspace_floats(capacity,d.qwen_parts==2 && (count<experts || !shared))*4)
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
    // The allocation may still be empty: the verified load fills it after placement.
    // Existing direct callers retain native banks unless they explicitly finalize.
    for (size_t role=0;role<2;++role) {
        const auto layout=d.expert_layouts.find(matrices[role].data);
        if (layout!=d.expert_layouts.end() && (layout->second.layout || layout->second.packing))
            throw std::runtime_error("cannot create a native expert plan from an already converted bank");
        plan->routed_banks[role]=matrices[role];
    }
    plan->routed_count=count;
    plan->pack_routed=kinds[0]==kinds[1];
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

// All deferred writes/fills must have completed before this explicit load boundary.
void finalize_resident_experts(Device& d) {
    check(hipStreamSynchronize(d.stream));
    struct Bank { Matrix matrix; size_t count; bool blocked; };
    std::unordered_map<const uint8_t*,Bank> banks;
    for (const auto& item:d.resident) {
        const ResidentMoe& plan=*item.second;
        if (plan.hybrid || !plan.routed_count) continue;
        for (const Matrix& matrix:plan.routed_banks) {
            const auto scratch=d.allocations.find(plan.workspace);
            const bool allowed=plan.pack_routed && expert_soa_eligible(matrix.kind,matrix.rows,matrix.cols) &&
                scratch!=d.allocations.end() && scratch->second>=matrix.rows*matrix.row_bytes;
            const auto inserted=banks.emplace(matrix.data,Bank{matrix,plan.routed_count,!allowed});
            Bank& bank=inserted.first->second;
            bank.blocked|=!allowed || bank.matrix.kind!=matrix.kind || bank.matrix.rows!=matrix.rows ||
                bank.matrix.cols!=matrix.cols || bank.matrix.row_bytes!=matrix.row_bytes || bank.count!=plan.routed_count;
        }
    }
    // Down and shared weights keep their native readers, even if a direct caller
    // reused an allocation as both a routed bank and one of these matrices.
    for (const auto& item:d.resident) {
        const ResidentMoe& plan=*item.second;
        for (size_t index=0;index<plan.weights.size();++index) {
            if (index<plan.experts*3 && index%3!=2) continue;
            const auto bank=banks.find(plan.weights[index].data);
            if (bank!=banks.end()) bank->second.blocked=true;
        }
    }
    // An incompatible or native-only alias prevents conversion, independent of plan order.
    for (const auto& item:banks) if (item.second.blocked) {
        const Bank& bank=item.second;
        const auto old=d.expert_layouts.find(item.first);
        if (old!=d.expert_layouts.end() && (old->second.layout || old->second.packing))
            throw std::runtime_error("an already converted expert bank has a native-only alias");
        d.expert_layouts.emplace(item.first,ExpertBankLayout{bank.matrix.kind,bank.matrix.rows,bank.matrix.cols,bank.count,0});
    }
    for (auto& item:d.resident) {
        ResidentMoe& plan=*item.second;
        if (plan.hybrid || !plan.routed_count) continue;
        for (size_t role=0;role<2;++role) {
            Matrix bank=plan.routed_banks[role];
            resident_pack_bank(d,bank,plan.routed_count,plan.workspace,plan.pack_routed);
            for (size_t expert=0;expert<plan.experts;++expert) {
                Matrix& matrix=plan.weights[expert*3+role];
                matrix.layout=matrix.data ? bank.layout : 0;
            }
        }
        // The shared group can hold a half expert in table and remain absent in
        // batch_table. Update routed groups alone, preserving both shared meanings.
        std::vector<MoeGroup> groups(plan.experts);
        for (size_t expert=0;expert<plan.experts;++expert)
            groups[expert]={{plan.weights[expert*3],plan.weights[expert*3+1],plan.weights[expert*3+2]},0,0,0};
        const size_t bytes=groups.size()*sizeof(MoeGroup);
        upload(d,plan.table,groups.data(),bytes);
        if (plan.batch_table) upload(d,plan.batch_table,groups.data(),bytes);
    }
    check(hipStreamSynchronize(d.stream));
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
constexpr size_t prompt_cpu_routes_max=4096;
// Whether a prompt batch's streamed experts of `plan` are gathered by the CPU's workers (banks not page-locked; hybrid_stream_missing).
inline bool prompt_gathered(const Device& d,const ResidentMoe& plan) { return d.slots.host && d.cpu && !plan.mapped[0]; }
// The experts of `missing` (a batch's rows each in `counts`) the CPU takes instead of the bus: the GPU streams an expert whatever
// its rows, the CPU computes a route at a time, so those with the fewest rows go to the CPU while its time for their routes stays
// under its share (Device::prompt_scale) of the GPU's for the experts left to stream (each cost from the last batches,
// Device::prompt_stream_ns and prompt_cpu_ns; before any, a 16 GB/s bus and 60 us a route). They leave `missing`.
inline std::vector<size_t> prompt_cpu_share(const Device& d,const ResidentMoe& plan,const unsigned* counts,std::vector<size_t>& missing) {
    std::vector<size_t> taken;
    if (!d.prompt_cpu || !d.cpu || !plan.host[0] || missing.empty()) return taken;
    const double stream=d.prompt_stream_ns>0 ? d.prompt_stream_ns : double(expert_slot_bytes(plan))/16.0;
    const double route=d.prompt_cpu_ns>0 ? d.prompt_cpu_ns : 60000.0;
    std::vector<size_t> order(missing);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b) { return counts[a]<counts[b]; });
    // (at most prompt_cpu_routes_max routes a batch: the CPU's buffers and the batch's pinned results stay within some tens of MB)
    size_t routes_taken=0;
    if (prompt_gathered(d,plan)) {
        // Banks not page-locked: the CPU's workers gather each streamed expert into pinned memory before the bus takes it, and the
        // CPU's own routes wait behind those gathers on the same workers, so the CPU's time is the layer's. An expert goes to the CPU
        // only when its routes cost less than its gather (before any measurement, a gather at 8 GB/s).
        const double gather=d.prompt_gather_ns>0 ? d.prompt_gather_ns : double(expert_slot_bytes(plan))/8.0;
        for (size_t expert : order) {
            if (route*counts[expert]>0.9*gather || routes_taken+counts[expert]>prompt_cpu_routes_max) break;
            routes_taken+=counts[expert];
            taken.push_back(expert);
        }
    } else {
        double cpu=0;
        size_t left=missing.size();
        for (size_t expert : order) {
            const double next=cpu+route*counts[expert];
            if (next>d.prompt_scale*stream*double(left-1) || routes_taken+counts[expert]>prompt_cpu_routes_max) break;
            routes_taken+=counts[expert];
            cpu=next; --left; taken.push_back(expert);
        }
    }
    if (taken.empty()) return taken;
    std::vector<bool> gone(plan.experts,false);
    for (size_t expert : taken) gone[expert]=true;
    missing.erase(std::remove_if(missing.begin(),missing.end(),[&](size_t expert) { return bool(gone[expert]); }),missing.end());
    return taken;
}
// One CPU route's output row (blockIdx.x of `rows`) to its route's row of the outputs.
__global__ void prompt_cpu_scatter(const float* rows,const int32_t* routes,float* outputs) {
    const float* from=rows+size_t(blockIdx.x)*2560;
    float* to=outputs+size_t(routes[blockIdx.x])*2560;
    for (unsigned i=threadIdx.x;i<2560;i+=256) to[i]=from[i];
}
inline void prompt_cost(double& average,double sample) { average=average>0 ? 0.7*average+0.3*sample : sample; }
// Room for `floats` in the pinned prompt buffer (between batches: no copy reads it).
// Pinned memory, or where the system refuses it (a machine short of memory) pageable memory, whose copies the driver stages; null
// when even that is refused: the batch then streams the CPU's experts as well (grouped_expert_prefill).
inline float* prompt_host(Device& d,size_t floats) {
    if (!d.prompt_rows) check(hipEventCreateWithFlags(&d.prompt_rows,hipEventDisableTiming));
    if (d.prompt_host_floats>=floats) return d.prompt_host;
    if (d.prompt_host && d.prompt_host_pinned) check(hipHostFree(d.prompt_host));
    d.prompt_host=nullptr; d.prompt_host_floats=0; d.prompt_host_pinned=false;
    std::vector<float>().swap(d.prompt_pageable);
    void* pinned=nullptr;
    if (hipHostMalloc(&pinned,floats*4,hipHostMallocDefault)==hipSuccess) {
        d.prompt_host=static_cast<float*>(pinned); d.prompt_host_pinned=true;
    } else {
        (void)hipGetLastError();   // (the refusal is not the stream's error)
        try { d.prompt_pageable.resize(floats); }
        catch (const std::bad_alloc&) { return nullptr; }
        d.prompt_host=d.prompt_pageable.data();
    }
    d.prompt_host_floats=floats;
    return d.prompt_host;
}
#if !defined(__HIPCC__)
// moe_device_tiles' lists for the CUDA kernels, the shared expert's group `skip` left out of the tiles (its own list: its format is not the routed
// experts'), the groups' first rows made all the same. The tiles of the groups that have weights now, which are the layer's resident experts: those
// beyond the slots have none until a copy is pointed at.
constexpr unsigned moe_tile_threads_cuda=256;
__global__ void __launch_bounds__(moe_tile_threads_cuda) moe_device_tiles_cuda(const MoeGroup* groups,size_t groups_count,const unsigned* counts,
                                                                              uint32_t* firsts,MoeTile* tiles,unsigned* count,size_t skip) {
    __shared__ unsigned rows[moe_tile_threads_cuda],blocks[moe_tile_threads_cuda];
    const unsigned t=threadIdx.x,per=unsigned((groups_count+moe_tile_threads_cuda-1)/moe_tile_threads_cuda);
    const size_t begin=std::min(groups_count,size_t(t)*per),end=std::min(groups_count,begin+per);
    unsigned own_rows=0,own_blocks=0;
    for (size_t group=begin;group<end;++group) {
        own_rows+=counts[group];
        if (group!=skip && groups[group].weights[0].data) own_blocks+=(counts[group]+moe_block_m-1)/moe_block_m;
    }
    rows[t]=own_rows; blocks[t]=own_blocks;
    __syncthreads();
    for (unsigned offset=1;offset<moe_tile_threads_cuda;offset*=2) {
        const unsigned r=t>=offset ? rows[t-offset] : 0,b=t>=offset ? blocks[t-offset] : 0;
        __syncthreads();
        rows[t]+=r; blocks[t]+=b;
        __syncthreads();
    }
    unsigned first=rows[t]-own_rows,tile=blocks[t]-own_blocks;
    for (size_t group=begin;group<end;++group) {
        firsts[group]=first; first+=counts[group];
        if (group==skip || !groups[group].weights[0].data) continue;
        for (unsigned row=0;row<counts[group];row+=moe_block_m) tiles[tile++]=MoeTile{uint32_t(group),row};
    }
    if (t==moe_tile_threads_cuda-1) *count=blocks[t];
}
// The shared expert's tiles: its rows are all of the batch's, in blocks of moe_block_m.
__global__ void moe_shared_tiles(MoeTile* tiles,size_t group,size_t steps) {
    const size_t tile=size_t(blockIdx.x)*256+threadIdx.x;
    if (tile*moe_block_m<steps) tiles[tile]=MoeTile{uint32_t(group),uint32_t(tile*moe_block_m)};
}
// `download` on a stream of its own: the host waits for the copy, not for what the device's stream has queued since (the stream must have been made
// to wait for the event the bytes follow).
inline void download_side(Device& d,hipStream_t stream,void* host,const void* device,size_t bytes) {
    uint8_t* slot=io_slot(d,bytes);
    check(hipMemcpyAsync(slot,device,bytes,hipMemcpyDeviceToHost,stream));
    check(hipStreamSynchronize(stream));
    std::memcpy(host,slot,bytes);
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
#if !defined(__HIPCC__)
    // A hybrid layer's resident experts (and its shared expert) are queued before the host knows the counts: the device builds their tile list, and the
    // host reads the counts, plans the experts beyond the slots and builds their tiles while the GEMMs run, where it waited for the counts with the
    // stream idle after them, then planned and uploaded (a 1.2 ms gap a layer on the first device of two RTX 4090s).
    const bool eager=d.moe_eager && plan.hybrid && !plan.part && !plan.batch_table && plan.weights[experts*3].data && steps>1;
    if (eager) {
        if (!d.counts_event) check(hipEventCreateWithFlags(&d.counts_event,hipEventDisableTiming));
        check(hipEventRecord(d.counts_event,d.stream));
    }
    // (CUDA, a wide batch: the layer's experts beyond the slots that its last batch used most start over the bus now, beside the layer's
    // own work that is still queued ahead of this)
    const bool prestaging=prestage_wanted(d,plan,steps);
    PrestageUse prestage(d,plan,prestaging);   // (undone on every way out: an error leaves no table entry at the region, nor the layer as the prestaged one)
    if (prestaging) prestage_issue(d,plan);
#endif
    phase(d,"moe_group");
#if !defined(__HIPCC__)
    // FP16 product uses routes*1280 bytes, then the converted token inputs use steps*5120. Both fit before outputs (routes*7680 bytes from scratch).
    auto* product=reinterpret_cast<__half*>(scratch);
    auto* half_input=product+routes*640;
    // (the tile lists: the device's from 0, the shared expert's first and the routed experts' after it, the host's from host_base)
    const size_t routed_most=(steps*10+moe_block_m-1)/moe_block_m+experts;
    const size_t shared_tiles=eager ? (steps+moe_block_m-1)/moe_block_m : 0;
    const size_t host_base=eager ? (shared_tiles+routed_most+63)/64*64 : 0;
    auto* built_count=reinterpret_cast<unsigned*>(tiles+host_base+routed_most);   // (the host's list, as long as the routed one at most, then the count)
    if (eager && (host_base+routed_most+2)*sizeof(MoeTile)+steps*10*4+(((experts+2)&~size_t(1))+2*groups_count+2)*4>65536*4)
        throw std::runtime_error("resident MoE grouping scratch overflow");
    const char* phase_gate="res_gate_up",*phase_down="res_down";
    // Banks have one format per role. Separate the final shared-expert
    // tiles so its format cannot inflate the routed Q2 kernel's registers.
    auto cuda_launch=[&](const MoeTile* list,size_t start,size_t finish,size_t group,const unsigned* built=nullptr) {
        if (finish<=start) return;
        const dim3 gate_grid(10,unsigned(finish-start),moe_block_m/moe_cuda_m);
        const dim3 down_grid(20,unsigned(finish-start),moe_block_m/moe_cuda_m);
        // Each format with a compiled decode has its own kernel (moe_cuda_mode): a switch on the kind for every
        // sixteen weights made the generic kernel's prefill about ten times the RDNA4 kernels' a routed row.
        const int gate_kind=plan.weights[group*3].kind, up_kind=plan.weights[group*3+1].kind;
        const int gate_mode=gate_kind==up_kind ? moe_cuda_gate_mode(gate_kind) : 0, down_mode=moe_cuda_mode(plan.weights[group*3+2].kind);
        const int gate_only=moe_cuda_gate_mode(gate_kind), up_only=moe_cuda_gate_mode(up_kind);
        const dim3 gate_generic(10,unsigned(finish-start)), down_generic(20,unsigned(finish-start));
        #define SD_CUDA_GATE(mode) case mode: moe_gate_up_mma<mode,mode,true><<<mode==42 ? gate_grid : gate_generic,256,0,d.stream>>>( \
            groups,list+start,positions,steps,half_input,product,built); break;
        #define SD_CUDA_DOWN(mode) case mode: moe_down_mma<mode><<<mode==42 ? down_grid : down_generic,256,0,d.stream>>>( \
            groups,list+start,positions,product,outputs,built); break;
        // (a shared expert's gate and up of two formats, each one with a compiled decode: the pair's kernel; any other pair the generic)
        bool paired=false;
        #define SD_CUDA_PAIR(g,u) if (!paired && gate_only==g && up_only==u) { paired=true; \
            moe_gate_up_mma<g,u,true><<<gate_generic,256,0,d.stream>>>(groups,list+start,positions,steps,half_input,product,built); }
        #define SD_CUDA_PAIRS(g) SD_CUDA_PAIR(g,12) SD_CUDA_PAIR(g,13) SD_CUDA_PAIR(g,14) SD_CUDA_PAIR(g,23)
        if (gate_kind!=up_kind && gate_only && up_only) { SD_CUDA_PAIRS(12) SD_CUDA_PAIRS(13) SD_CUDA_PAIRS(14) SD_CUDA_PAIRS(23) }
        #undef SD_CUDA_PAIRS
        #undef SD_CUDA_PAIR
        #define SD_CUDA_MODES(X) X(0) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(29) X(42)
        #define SD_CUDA_GATE_MODES(X) SD_CUDA_MODES(X) X(12) X(13) X(14)
        if (!paired) switch (gate_mode) { SD_CUDA_GATE_MODES(SD_CUDA_GATE) }
        phase(d,group==0 ? phase_gate : "shared_gate_up");
        switch (down_mode) { SD_CUDA_MODES(SD_CUDA_DOWN) }
        phase(d,group==0 ? phase_down : "shared_down");
        #undef SD_CUDA_GATE_MODES
        #undef SD_CUDA_MODES
        #undef SD_CUDA_DOWN
        #undef SD_CUDA_GATE
        check(hipGetLastError());
    };
    if (eager) {
        // The layer's resident experts and its shared expert: their lists, the groups' first rows and places, the FP16 rows, and the GEMMs, queued.
        moe_device_tiles_cuda<<<1,moe_tile_threads_cuda,0,d.stream>>>(groups,groups_count,counts,firsts,tiles+shared_tiles,built_count,experts);
        moe_shared_tiles<<<unsigned((shared_tiles+255)/256),256,0,d.stream>>>(tiles,experts,steps);
        moe_set_groups<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,firsts);
        resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,cursors,selected,steps,experts,positions);
        moe_input_half<<<unsigned((steps*2560+1023)/1024),256,0,d.stream>>>(steps*2560,input,half_input);
        cuda_launch(tiles+shared_tiles,0,routed_most,0,built_count);
        cuda_launch(tiles,0,shared_tiles,experts);
    }
#endif
#if defined(__HIPCC__)
    if (!plan.hybrid) {
        // All on the device: the host queues the experts (and then the peer's half, the next
        // layer) without waiting for this layer's routing.
        auto* tile_count=reinterpret_cast<unsigned*>(tiles+max_tiles);
        moe_device_tiles<<<1,moe_tile_threads,0,d.stream>>>(groups,groups_count,counts,firsts,tiles,tile_count);
        moe_set_groups<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,firsts);
        resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,cursors,selected,steps,experts,positions);
        // (a layer part's half of the shared expert runs beside the grouped experts: qwen_parts.hpp)
        const unsigned shared=plan.weights[experts*3].data && !plan.batch_table ? unsigned((steps+moe_block_m-1)/moe_block_m) : 0;
        // FP16 products (routes*1280 bytes), then the FP16 token inputs (steps*5120), within the gate/up scratch.
        auto* product=reinterpret_cast<_Float16*>(scratch);
        auto* half_input=product+routes*640;
        moe_input_half<<<unsigned((steps*2560+1023)/1024),256,0,d.stream>>>(steps*2560,input,half_input);
        moe_expert_launch<2560,640,10,true>(d.stream,moe_kinds(plan,0),moe_kinds(plan,experts),groups,tiles,positions,steps,resident_shape,half_input,
                          product,outputs,tile_count,unsigned(max_tiles),shared);
        check(hipGetLastError());
        return;
    }
#endif
    // (the host's reads of the device: eagerly queued layers read on a stream of their own that waits for the counts alone)
#if defined(__HIPCC__)
    auto read_back=[&](void* host,const void* device,size_t bytes) { download(d,host,device,bytes); };
#else
    if (eager) {
        if (!d.rows_stream) check(hipStreamCreate(&d.rows_stream));
        check(hipStreamWaitEvent(d.rows_stream,d.counts_event,0));
    }
    auto read_back=[&](void* host,const void* device,size_t bytes) {
        if (eager) download_side(d,d.rows_stream,host,device,bytes); else download(d,host,device,bytes);
    };
#endif
    std::vector<unsigned> host_counts(groups_count);
    read_back(host_counts.data(),counts,groups_count*4);
    const size_t tail=plan.hybrid ? std::min(steps,hybrid_prompt_tail) : 0;
    std::vector<int32_t> host_tail(tail*10);
    if (tail) read_back(host_tail.data(),selected+(steps-tail)*10,tail*10*4);
    std::vector<uint32_t> host_firsts(groups_count);
    std::vector<size_t> missing, prestaged;
    std::vector<bool> absent(groups_count,false);
    if (plan.part)
        for (size_t group=0;group<groups_count;++group) absent[group]=plan.weights[group*3].data==nullptr;
    if (plan.batch_table) absent[experts]=true;
    std::vector<size_t> on_cpu;
    if (plan.hybrid) {
        hybrid_observe_batch(plan,host_counts.data(),steps,host_tail.data(),tail);
        for (size_t expert=0;expert<experts;++expert)
            if (!plan.layer_prefill && host_counts[expert] && plan.slot_of[expert]<0) { missing.push_back(expert); absent[expert]=true; }
#if !defined(__HIPCC__)
        // Those prestaged (and still beyond the slots) are computed beside the resident ones: they leave the experts to stream.
        if (prestaging && d.prestage.plan==&plan) {
            std::vector<size_t> rest;
            for (size_t expert : missing) (d.prestage.position[expert]>=0 ? prestaged : rest).push_back(expert);
            missing.swap(rest);
        }
        if (prestaging) plan.last_counts.assign(host_counts.begin(),host_counts.begin()+plan.experts);
#endif
        on_cpu=prompt_cpu_share(d,plan,host_counts.data(),missing);
    }
    // The CPU's routes, expert by expert (before the GPU's work is queued: a download waits for it), and the rows of their tokens
    // (queued before the GPU's work: they come over while the bus takes the experts the other way).
    std::vector<int32_t> cpu_routes;
    std::vector<uint32_t> cpu_experts;   // each CPU route's expert
    float* cpu_rows=nullptr;
    float* cpu_results=nullptr;
    if (!on_cpu.empty()) {
        std::vector<int32_t> chosen(steps*10);
        read_back(chosen.data(),selected,steps*10*4);
        std::vector<std::vector<int32_t>> by_expert(experts);
        std::vector<bool> taken(experts,false);
        for (size_t expert : on_cpu) taken[expert]=true;
        for (size_t route=0;route<steps*10;++route)
            if (taken[size_t(chosen[route])]) by_expert[size_t(chosen[route])].push_back(int32_t(route));
        for (size_t expert : on_cpu) {
            cpu_routes.insert(cpu_routes.end(),by_expert[expert].begin(),by_expert[expert].end());
            cpu_experts.insert(cpu_experts.end(),by_expert[expert].size(),uint32_t(expert));
        }
        const size_t n=cpu_routes.size();
        cpu_rows=prompt_host(d,steps*2560+n*2561);
        if (cpu_rows) {
            cpu_results=cpu_rows+steps*2560;
            std::memcpy(cpu_results+n*2560,cpu_routes.data(),n*4);
#if defined(__HIPCC__)
            check(hipMemcpyAsync(cpu_rows,input,steps*2560*4,hipMemcpyDeviceToHost,d.stream));
            check(hipEventRecord(d.prompt_rows,d.stream));
#else
            // (the rows are final: the counts' download above waited for the work that made them. Their copy on a stream of its own leaves the
            // GEMMs, which are queued on the device's stream next, to start at once: a wide batch's rows are 42 MB, 2 ms of the bus)
            if (!d.rows_stream) check(hipStreamCreate(&d.rows_stream));
            check(hipMemcpyAsync(cpu_rows,input,steps*2560*4,hipMemcpyDeviceToHost,d.rows_stream));
            check(hipEventRecord(d.prompt_rows,d.rows_stream));
#endif
        } else {
            // (no host memory for the rows: the CPU's experts are streamed with the rest)
            missing.insert(missing.end(),on_cpu.begin(),on_cpu.end());
            std::sort(missing.begin(),missing.end());
            on_cpu.clear(); cpu_routes.clear(); cpu_experts.clear();
        }
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
#if !defined(__HIPCC__)
    if (!eager)   // (an eagerly queued layer's resident experts have their tiles from the device)
#endif
    for (size_t group=0;group<groups_count;++group) if (!absent[group]) add(group);
    const size_t resident_tiles=host_tiles.size();
    for (size_t expert : prestaged) add(expert);
    const size_t prestaged_end=host_tiles.size();
    std::vector<size_t> offsets{prestaged_end};
    for (size_t expert : missing) { add(expert); offsets.push_back(host_tiles.size()); }
    if (host_tiles.size()>max_tiles) throw std::runtime_error("inconsistent Straydog expert grouping");
#if defined(__HIPCC__)
    upload(d,firsts,host_firsts.data(),groups_count*4);
    if (!host_tiles.empty()) upload(d,tiles,host_tiles.data(),host_tiles.size()*sizeof(MoeTile));
    moe_set_groups<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,firsts);
    resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,cursors,selected,steps,experts,positions);
    auto* product=reinterpret_cast<_Float16*>(scratch);
    // FP16 product uses routes*1280 bytes, then the converted token inputs use
    // steps*5120. Both fit before outputs (routes*7680 bytes from scratch).
    auto* half_input=product+routes*640;
    moe_input_half<<<unsigned((steps*2560+1023)/1024),256,0,d.stream>>>(steps*2560,input,half_input);
#else
    if (host_tiles.size()>(eager ? routed_most : max_tiles)) throw std::runtime_error("inconsistent Straydog expert grouping");
    if (!host_tiles.empty()) upload(d,tiles+host_base,host_tiles.data(),host_tiles.size()*sizeof(MoeTile));
    if (!eager) {
        upload(d,firsts,host_firsts.data(),groups_count*4);
        moe_set_groups<<<unsigned((groups_count+255)/256),256,0,d.stream>>>(groups,groups_count,firsts);
        resident_scatter<<<unsigned((routes+255)/256),256,0,d.stream>>>(groups,cursors,selected,steps,experts,positions);
        moe_input_half<<<unsigned((steps*2560+1023)/1024),256,0,d.stream>>>(steps*2560,input,half_input);
    }
#endif
    phase(d,"moe_hostgap");
    if (d.phase_on) { d.phase_prestage_used+=prestaged.size(); d.phase_streamed+=missing.size(); d.phase_missing_layers+=missing.empty() ? 0 : 1;
                      d.phase_cpu_routes+=cpu_routes.size(); d.phase_bus_bytes+=(missing.size()+prestaged.size()*0)*expert_slot_bytes(plan);
                      for (size_t group=0;group<experts;++group) if (!absent[group]) d.phase_routes[0]+=host_counts[group];
                      for (size_t expert : prestaged) d.phase_routes[1]+=host_counts[expert];
                      for (size_t expert : missing) d.phase_routes[2]+=host_counts[expert];
                      d.phase_routes[3]+=cpu_routes.size(); }
#if defined(__HIPCC__)
    const char* phase_gate="res_gate_up",*phase_down="res_down";
#endif
    auto launch=[&](size_t begin,size_t end) {
        if (end<=begin) return;
#if defined(__HIPCC__)
        // The resident list ends with the shared expert's tiles.
        const unsigned shared=begin==0 && end==resident_tiles && !absent[experts] ?
            unsigned((host_counts[experts]+moe_block_m-1)/moe_block_m) : 0;
        moe_expert_launch<2560,640,10,true>(d.stream,moe_kinds(plan,0),moe_kinds(plan,experts),groups,tiles+begin,positions,steps,resident_shape,half_input,
                          product,outputs,nullptr,unsigned(end-begin),shared);
#else
        if (begin==0 && end==resident_tiles) {
            const size_t shared_begin=end-(host_counts[experts]+moe_block_m-1)/moe_block_m;
            cuda_launch(tiles+host_base,begin,shared_begin,0);
            cuda_launch(tiles+host_base,shared_begin,end,experts);
        } else cuda_launch(tiles+host_base,begin,end,0);
#endif
        check(hipGetLastError());
    };
    launch(0,resident_tiles);
    phase(d,"moe_resident");
#if !defined(__HIPCC__)
    if (!prestaged.empty()) {
        // (the copies are asked for ahead on the copy stream: the GEMMs of the experts they hold wait for them, and for nothing else)
        check(hipStreamWaitEvent(d.stream,d.prestage.event,0));
        phase(d,"pre_wait");
        prestage.point(prestaged);
        phase_gate="pre_gate_up"; phase_down="pre_down";
        launch(resident_tiles,prestaged_end);
        phase(d,"moe_prestaged");
    }
#endif
    // Meanwhile the CPU computes its routes (one row a route, weight 1: the combine weighs them) and they go to their output rows
    // through the gate/up scratch, after the GPU's GEMMs (stream order).
    uint64_t cpu_ns=0,compute_ns=0;
    auto meanwhile=[&]() {
        const size_t n=cpu_routes.size();
        if (!n) return;
        BankReading reading(d);   // (the CPU's workers read the banks' bytes below: sd_hybrid_repoint waits)
        const auto began=std::chrono::steady_clock::now();
        check(hipEventSynchronize(d.prompt_rows));
        const auto computing=std::chrono::steady_clock::now();
        std::vector<const uint8_t*> gate(n),up(n),down(n);
        std::vector<const float*> xs(n);
        std::vector<float*> outs(n);
        std::vector<float> weights(n,1.0f);
        std::vector<uint32_t> counts(n,1);
        for (size_t i=0;i<n;++i) {
            const size_t expert=cpu_experts[i];
            gate[i]=plan.host[0]+expert*plan.host_bytes[0];
            up[i]=plan.host[1]+expert*plan.host_bytes[1];
            down[i]=plan.host[2]+expert*plan.host_bytes[2];
            xs[i]=cpu_rows+size_t(cpu_routes[i])/10*2560;
            outs[i]=cpu_results+i*2560;
        }
        const int kinds[3]={plan.weights[0].kind,plan.weights[1].kind,plan.weights[2].kind};
        if (sd_cpu_experts_run_prompt(d.cpu,kinds,uint32_t(n),counts.data(),gate.data(),up.data(),down.data(),weights.data(),2560,640,
                                    xs.data(),outs.data()))
            throw std::runtime_error(std::string("Straydog CPU experts: ")+sd_cpu_error());
        const auto done=std::chrono::steady_clock::now();
        cpu_ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(done-began).count());
        compute_ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(done-computing).count());
        d.prompt_cpu_routes+=n;
        // (the gate/up scratch and the product after it: routes*1920 floats, a staged row and its route each)
        const size_t room=routes*1920/(2560+1);
        const auto* routes_host=reinterpret_cast<const int32_t*>(cpu_results+n*2560);
        for (size_t at=0;at<n;at+=room) {
            const size_t part=std::min(room,n-at);
            auto* staged_routes=reinterpret_cast<int32_t*>(scratch+part*2560);
            check(hipMemcpyAsync(scratch,cpu_results+at*2560,part*2560*4,hipMemcpyHostToDevice,d.stream));
            check(hipMemcpyAsync(staged_routes,routes_host+at,part*4,hipMemcpyHostToDevice,d.stream));
            prompt_cpu_scatter<<<unsigned(part),256,0,d.stream>>>(scratch,staged_routes,outputs);
            check(hipGetLastError());
        }
        phase(d,"moe_cpu_scatter");
    };
    phase_gate="str_gate_up"; phase_down="str_down";
    const uint64_t stream_ns=hybrid_stream_missing(d,plan,missing,[&](size_t begin,size_t count) { launch(offsets[begin],offsets[begin+count]); },
                                                   meanwhile);
#if !defined(__HIPCC__)
    prestage.end();
#endif
    if (missing.size()>=16) prompt_cost(d.prompt_stream_ns,double(stream_ns)/double(missing.size()));
    if (cpu_routes.size()>=16) prompt_cost(d.prompt_cpu_ns,double(compute_ns)/double(cpu_routes.size()));
    // The share moves toward the CPU finishing at nine tenths of the GPU's time (a CPU that took everything finished last); a gathered
    // layer's CPU time is its gathers' too, and its share is set otherwise (prompt_cpu_share).
    if (!cpu_routes.empty() && !prompt_gathered(d,plan)) {
        const double ratio=stream_ns ? double(cpu_ns)/double(stream_ns) : 2.0;
        d.prompt_scale=std::clamp(d.prompt_scale*std::clamp(0.9/ratio,0.6,1.2),0.05,2.0);
    }

    // Host vectors reach the device through the staging arena, so resident experts need no wait
    // here and the caller can queue the next work (the peer's half, the next layer) at once.
    if (plan.hybrid) check(hipStreamSynchronize(d.stream));
#if !defined(__HIPCC__)
    if (eager) {
        // (the stream is idle: the device's tile list is read back and must be the one the plan implies, the resident groups' tiles)
        unsigned built=0;
        download(d,&built,built_count,4);
        size_t expected=0;
        for (size_t group=0;group<experts;++group) if (!absent[group]) expected+=(host_counts[group]+moe_block_m-1)/moe_block_m;
        if (built!=expected) throw std::runtime_error("inconsistent Straydog expert grouping: the device's tiles are not the resident experts'");
    }
#endif
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
// Up to eight decode tokens (HyperConnection rows input_stride apart: 2560 inputs
// then the four stream mixes) with the one-token arithmetic: every kernel takes
// all tokens, the expert GEMVs per token, and the result is added to hidden rows.
// Decode-token workspace: routes then per-route outputs, probabilities and the top ten.
struct TokenMoe {
    ResidentMoe* plan; MoeGroup* groups; float* outputs; float* probabilities; int32_t* selected;
};
TokenMoe token_moe(Device& d,void* handle,size_t tokens) {
    auto found=d.resident.find(handle);
    if (found==d.resident.end() || !tokens || tokens>size_t(gemv_wide_vectors) || tokens>found->second->capacity || found->second->hybrid)
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
    // Layers in parts (qwen_parts.hpp): the shared expert's half, gate and up, runs in the router's launch (the same row read;
    // one launch a layer less), the scores after the decode workspace with the gate and up rows' stride.
    const size_t room=tokens<=4 ? 4 : size_t(gemv_wide_vectors), scores=room*11*4480+512;
    const auto scratch=d.allocations.find(plan.workspace);
    const bool together=d.qwen_parts==2 && plan.shared && plan.half[0].data && scratch!=d.allocations.end() && scratch->second/4>=scores+room*11*1280;
    if (together) {
        Matrix list[4]={projections[0],projections[1],plan.weights[experts*3],plan.weights[experts*3+1]};
        list[0].output_offset=scores; list[1].output_offset=scores+experts;
        list[2].output_offset=10*1280; list[3].output_offset=10*1280+640;
        gemv(d,list,4,input,gate_up,tokens,input_stride,11*1280);
        resident_route<<<unsigned(tokens),256,0,d.stream>>>(gate_up+scores,gate_up+scores+experts,experts,moe.selected,
                                                            moe.probabilities,11*1280,11*1280);
    } else {
        gemv(d,projections,2,input,d.output,tokens,input_stride,experts+1);
        resident_route<<<unsigned(tokens),256,0,d.stream>>>(d.output,d.output+experts,experts,moe.selected,moe.probabilities,
                                                            experts+1,experts+1);
    }
    // (with the shared expert's gate and up formed, the routed gate and up and the product in one launch; Q6_K's pairs spill)
    const int routed=plan.weights[0].kind;
    const unsigned lanes=together && plan.weights[1].kind==routed && routed!=14 && !plan.mapped[0] && d.q8_wide && d.rdna4 ?
        resident_lanes(plan.weights,0,input,4) : 0;
    // (only the formats it is compiled for: IQ2_XS's experts have others among them)
    bool listed=false;
    for (const int kind : {SD_EXPERT_ROUTED_FORMATS(SD_EXPERT_LISTED)}) listed|=kind==routed;
    const bool paired=listed && lanes>=2 && lanes<=16 && lanes==resident_lanes(plan.weights,1,input,4);
    if (paired) {
        const unsigned per_route=unsigned((640+16/lanes-1)/(16/lanes)), shared_cols=unsigned(plan.weights[experts*3].rows);
        const dim3 grid(unsigned((10*size_t(per_route)+(shared_cols+31)/32+7)/8),unsigned(tokens));
        #define SD_EXPERT_PAIRS(format) case format: if (format==14) break; resident_expert_pairs<format><<<grid,256,0,d.stream>>>( \
            moe.groups,moe.selected,per_route,lanes,input,input_stride,gate_up,product,shared_cols); break;
        switch (routed) {
            SD_EXPERT_ROUTED_FORMATS(SD_EXPERT_PAIRS)
            default: throw std::runtime_error("invalid Straydog expert pairs");
        }
        #undef SD_EXPERT_PAIRS
        check(hipGetLastError());
    } else {
        resident_expert_launch(d,plan,moe.groups,moe.selected,false,input,gate_up,tokens,input_stride,11*1280,together);
        moe_product<<<unsigned((routes*640+255)/256),256,0,d.stream>>>(gate_up,routes,product);
    }
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
// The owner's MoE for up to eight decode tokens: its peer evaluates the other experts
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
// Whether the owner runs [begin, end) with its peer helping: every layer's MoE is a part. (A layer split
// across both devices, DecoderLayer::parts, has no helper: each device runs its part and they add them.)
bool span_has_parts(Device& d,size_t begin,size_t end) {
    bool parts=false;
    for (size_t layer=begin;layer<end;++layer) {
        auto found=d.decoder.find(layer);
        if (found==d.decoder.end()) return false;
        const bool part=d.resident.find(found->second.moe)->second->part && found->second.parts==1;
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
    // Both graphs can contain a peer wait. Submit only after both were queued: the first
    // graph's capture/instantiation may otherwise use much of a display GPU's wait timeout.
#if !defined(__HIPCC__)
    flush_launches(d);
    on_device(h.ordinal,[&]() { flush_launches(h); });
#endif
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
    moe.plan=&plan; moe.groups=static_cast<MoeGroup*>(plan.batch_table ? plan.batch_table : plan.table);
    // (a layer in parts' prompt batch: the grouped FP16 path's halves in a gate and up scratch of 4800 floats a row, and no F32 product rows)
    const bool compact=plan.part && d.qwen_parts==2 && steps>1;
    moe.gate_up=plan.workspace; float* product=moe.gate_up+(compact ? steps*moe_compact_gate_up_row : routes*1280);
    moe.outputs=compact ? product : product+routes*640; moe.probabilities=moe.outputs+routes*2560;
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
    phase(d,"moe_router");
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
            grouped_moe_wmma<16,true><<<dim3(10,unsigned(max_tiles),2),128,0,d.stream>>>(groups,tiles,moe.positions,steps,false,source,gate_up,total,2);
            grouped_moe_wmma<32,true><<<dim3(10,unsigned(max_tiles),2),256,0,d.stream>>>(groups,tiles,moe.positions,steps,false,source,gate_up,total);
        }
        grouped_moe<true><<<dim3(10,unsigned(max_tiles),2),256,0,d.stream>>>(groups,tiles+max_tiles,moe.positions,steps,false,source,gate_up,total+1);
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
    // The owner's rows and the helper's partials each have a peer wait behind them. Both
    // streams must be queued before either is flushed, including the helper's separate stream.
#if !defined(__HIPCC__)
    flush_launches(d);
    on_device(h.ordinal,[&]() {
        StreamScope helping(h,h.helper_stream);
        flush_launches(h);
    });
#endif
}
