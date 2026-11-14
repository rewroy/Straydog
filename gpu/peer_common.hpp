#pragma once
// The flag accessors of peer.hpp that the device code of the other units reads (the linked sums of link_tail_common.hpp, the GEMV
// kernels): device functions only.
__device__ inline uint64_t peer_clock() {
#if defined(__HIPCC__)
    return wall_clock64();
#else
    uint64_t now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); return now;
#endif
}
__device__ inline uint32_t peer_load(uint32_t* flag) {
#if defined(__HIPCC__)
    return __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    const uint32_t value = *static_cast<volatile uint32_t*>(flag);
    __threadfence_system();
    return value;
#endif
}
// A flag as the other device wrote it, without the acquire (peer_load's): the caller orders its later reads with a fence.
__device__ inline uint32_t peer_flag_load(uint32_t* flag) {
#if defined(__HIPCC__)
    return __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *static_cast<volatile uint32_t*>(flag);
#endif
}
__device__ inline void peer_store(uint32_t* flag, uint32_t value) {
#if defined(__HIPCC__)
    __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    __threadfence_system();
    *static_cast<volatile uint32_t*>(flag) = value;
#endif
}
