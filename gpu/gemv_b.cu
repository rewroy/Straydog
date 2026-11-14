// The GEMV kernels of gemv.hpp that read a quantized input (the Q8 decode kernels), the second of two units (gemv_a.cu, gemv_b.cu): the one-format
// kernels of the kinds they take and the kernels of the lists that mix the wide formats and BF16. See gemv_a.cu.
#include "core.hpp"
namespace sdgpu {
#include "gemv_common.hpp"
#include "gemv_kernels.hpp"
SD_IQ_GRIDS_UPLOAD(iq_grids_upload_gemv_b)

// The kernels of the lists that mix the wide formats and BF16 (to eight vectors).
template void launch_gemv_q8_mixed<1>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<2>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<3>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<4>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<5>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<6>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<7>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_mixed<8>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);

// A Q8 kind: one to eight vectors and, to four, the linked form.
#define SD_INST_Q8_KIND(format) \
    template void launch_gemv_q8_kind<format, 1, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 2, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 3, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 4, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 5, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 6, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 7, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 8, false>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 1, true>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 2, true>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 3, true>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&); \
    template void launch_gemv_q8_kind<format, 4, true>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*, const LinkTail&);
SD_INST_Q8_KIND(8)
SD_INST_Q8_KIND(12)
SD_INST_Q8_KIND(13)
SD_INST_Q8_KIND(14)
SD_INST_Q8_KIND(23)
#if defined(__HIPCC__)
// (RDNA4: Q5_K rows of a unit a lane, four to a wave)
template void launch_gemv_q8_rows_wave<1, 4>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_rows_wave<2, 4>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_rows_wave<3, 8>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
template void launch_gemv_q8_rows_wave<4, 8>(unsigned, size_t, hipStream_t, const FixedGemv&, const float*, float*);
#endif
} // namespace sdgpu
