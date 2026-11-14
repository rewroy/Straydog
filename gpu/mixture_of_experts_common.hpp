#pragma once
// mixture_of_experts.hpp, in moe.cu: the independent top-10 mixture and the product kernel the resident experts' launches use.
void mixture_of_experts(Device& d, void* const* pointers, const int* kinds, size_t experts,
                        void* workspace, size_t steps, const int32_t* selected, const float* probabilities,
                        const float* input, float* output);
void launch_moe_product(unsigned blocks, hipStream_t stream, const float* gate_up, size_t routes, float* product);
