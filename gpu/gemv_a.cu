// The GEMV kernels of gemv.hpp that read the F32 input, the first of two units (gemv_a.cu, gemv_b.cu): the lists of any formats (gemv_fixed), the
// one-format lists of every format (gemv_fixed_kind and its linked and shared-input forms) and the lists that stage an input in shared memory. The
// Q8 decode kernels, which stage a quantized input, are gemv_b.cu's. A kernel is defined in one of the two and launched from the dispatch in
// kernels.cpp through the launches of gemv_common.hpp, which these units instantiate: one missing is a link error, one in both a kernel defined
// twice. The units are divided by family because a kernel's code depends on the kernels it is compiled with (the Q8 kernels of one format,
// split from the other formats', came out with their address arithmetic ordered differently).
#include "core.hpp"
namespace sdgpu {
#include "gemv_common.hpp"
#include "gemv_kernels.hpp"
SD_IQ_GRIDS_UPLOAD(iq_grids_upload_gemv_a)

__global__ void gemv_list(const GemvItem* items, size_t count, const float* input, float* output) {
    const size_t wave = size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    size_t low = 0, high = count;
    while (low + 1 < high) {
        const size_t middle = (low + high) / 2;
        if (items[middle].first_wave <= wave) low = middle; else high = middle;
    }
    const GemvItem item = items[low];
    gemv_wave<1>(item.matrix, wave - item.first_wave, item.lanes, input, 0, output, 0);
}
void launch_gemv_list(unsigned blocks, hipStream_t stream, const GemvItem* items, size_t count, const float* input, float* output) {
    gemv_list<<<blocks, 256, 0, stream>>>(items, count, input, output);
}

// The lists of any formats, and the lists that stage an input in shared memory.
template void launch_gemv_fixed<1>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_fixed<2>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_fixed<3>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_fixed<4>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
#if !defined(__HIPCC__)
template void launch_gemv_mixed_shared<true, 1>(unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_mixed_shared<true, 2>(unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_mixed_shared<false, 1>(unsigned, hipStream_t, const FixedGemv&, const float*, float*);
#endif

// Every format's one-format kernels.
#define SD_INST_FIXED_KIND(format) \
    template void launch_gemv_fixed_kind<format, 1>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*); \
    template void launch_gemv_fixed_kind<format, 2>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*); \
    template void launch_gemv_fixed_kind<format, 3>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*); \
    template void launch_gemv_fixed_kind<format, 4>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
#define SD_INST_LINKED_KIND(format) \
    template void launch_gemv_linked_kind<format>(unsigned, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&);
SD_FAST_FORMATS(SD_INST_FIXED_KIND)
SD_FAST_FORMATS(SD_INST_LINKED_KIND)
#if !defined(__HIPCC__)
#define SD_INST_DECODE_SHARED(format) \
    template void launch_gemv_decode_shared<format>(unsigned, hipStream_t, const Matrix&, unsigned, const float*, float*, const GemvResidual&);
SD_FAST_FORMATS(SD_INST_DECODE_SHARED)
#else
// (RDNA4's one-token 2560 x 6144 projections: Q3_K and Q6_K; BF16 rows of five to eight vectors)
template void launch_gemv_decode_shared_rdna4<11>(unsigned, hipStream_t, const Matrix&, unsigned, const float*, float*, const GemvResidual&);
template void launch_gemv_decode_shared_rdna4<14>(unsigned, hipStream_t, const Matrix&, unsigned, const float*, float*, const GemvResidual&);
template void launch_gemv_fixed_kind<30, 5>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_fixed_kind<30, 6>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_fixed_kind<30, 7>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_fixed_kind<30, 8>(unsigned, unsigned, hipStream_t, const FixedGemv&, const float*, float*);
#endif
} // namespace sdgpu
