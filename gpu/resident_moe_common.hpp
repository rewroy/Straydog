#pragma once
// What kernels.cpp takes from resident_moe.hpp: the types of a layer's experts and of a pass's workspace, the device functions the hybrid and
// the Qwen parts' kernels share with the experts', and the calls into moe.cu, which defines the layers' experts.
#include "grouped_gemm_common.hpp"
#include "peer_common.hpp"
// Exact top-10 of the router's scores (resident_route): the order of two candidates, and a wave's best.
struct RouteCandidate { float value; int index; };
__device__ __forceinline__ bool route_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__device__ __forceinline__ void route_best(float& value, int& index) {
    for (unsigned offset = 16; offset; offset /= 2) {
        const float other = __shfl_xor(value, offset, 32);
        const int other_index = __shfl_xor(index, offset, 32);
        if (route_better(other, other_index, value, index)) { value = other; index = other_index; }
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
// A row's staging: where its assisted experts are staged (`stage`, after the earlier rows') or its slice of the pass's staging plan
// (`assign`), each null where the plan has none (expert_row_stage). A value the helpers are passed: the plan itself, a kernel's by-value
// argument, is never written.
struct ExpertRow { const uint8_t* stage; const int32_t* assign; };
// Whether the GPU evaluates `route`: resident, or among the first assisted absent routes.
// (No array of the plan is read at an index the thread computes, here or in expert_wave, and the plan is not written: with either nvcc keeps
// the whole plan, a kernel's argument, in every thread's stack and writes it there at the kernel's entry -- 17 stores of local memory before
// the first branch of resident_expert_gemv_kind, 136 bytes a thread, some 30 MB a launch of a four-row down, as much as the weights it reads.)
__device__ __forceinline__ const uint8_t* expert_source(const MoeGroup* groups,const int32_t* selected,size_t route,
                                                        size_t role,const ExpertGemv& plan,const ExpertRow& row) {
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
    if (row.assign) {
        const int32_t slot=row.assign[route];
        if (slot<0) return nullptr;
        if (row.stage) return row.stage+size_t(slot)*plan.slot_bytes+(role>0 ? plan.bytes[0] : 0)+(role>1 ? plan.bytes[1] : 0);
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
    if (row.stage) return row.stage+earlier*plan.slot_bytes+(role>0 ? plan.bytes[0] : 0)+(role>1 ? plan.bytes[1] : 0);
    return mapped+expert*bytes;
}
// The plan's own staging, for a caller whose rows are not offset.
__device__ __forceinline__ const uint8_t* expert_source(const MoeGroup* groups,const int32_t* selected,size_t route,
                                                        size_t role,const ExpertGemv& plan) {
    return expert_source(groups,selected,route,role,plan,ExpertRow{plan.stage,plan.assign});
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
// Decode-token workspace: routes then per-route outputs, probabilities and the top ten.
struct TokenMoe {
    ResidentMoe* plan; MoeGroup* groups; float* outputs; float* probabilities; int32_t* selected;
};
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
// Prefill workspace: routes in the prefill layout (moe_route_output), the probabilities,
// the top ten and the grouping scratch.
struct StepMoe {
    ResidentMoe* plan; MoeGroup* groups; float* gate_up; float* outputs; float* probabilities;
    int32_t* positions; int32_t* selected; unsigned* cursors; unsigned* total; MoeTile* tiles; size_t max_tiles;
};

// ---- moe.cu's, called from kernels.cpp ----
void* create_resident_moe(Device& d,void* const* pointers,const int* kinds,size_t experts,
                          void* workspace,size_t capacity,size_t first=0,size_t count=SIZE_MAX,bool shared=true);
void finalize_resident_experts(Device& d);
unsigned resident_lanes(const std::vector<Matrix>& weights,size_t role,const float* input,size_t concurrent);
void resident_expert_launch(Device& d,const ResidentMoe& plan,const MoeGroup* groups,const int32_t* selected,
                            bool down,const float* input,float* output,
                            size_t tokens=1,size_t input_stride=0,size_t output_stride=0,bool shared_done=false,
                            const uint8_t* assist_stage=nullptr,uint64_t assist_slot_bytes=0,const int32_t* assist_assign=nullptr,unsigned only=0);
TokenMoe token_moe(Device& d,void* handle,size_t tokens);
void moe_tokens_experts(Device& d,const TokenMoe& moe,size_t tokens,const float* input,size_t input_stride);
void moe_tokens(Device& d,void* handle,size_t tokens,const float* input,size_t input_stride,float* hidden);
void helper_span(Device& h,size_t begin,size_t end,size_t tokens);
bool span_has_parts(Device& d,size_t begin,size_t end);
void finish_with_helper(Device& d);
StepMoe step_moe(Device& d,void* handle,size_t steps);
void moe_steps_route(Device& d,const StepMoe& moe,size_t steps,const float* source);
void moe_steps_compute(Device& d,const StepMoe& moe,size_t steps,const float* source);
void resident_moe(Device& d,void* handle,size_t steps,const float* input,float* output,bool device_io=false);
void resident_moe_parts(Device& d,void* handle,size_t layer,size_t steps,const float* input,float* output);
// (a layer's kernels are in moe.cu: their launches, and the addresses a benchmark asks the occupancy of)
void launch_resident_route(unsigned blocks, hipStream_t stream, const float* scores, const float* mix, size_t experts, int32_t* selected,
                           float* probabilities, size_t row_stride, size_t mix_stride);
const void* resident_expert_gemv_function();
const void* resident_expert_gemv_kind_22_function();

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
