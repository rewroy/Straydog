// What every translation unit of the GPU backend shares: the platform prelude (HIP and CUDA spellings, includes), then in namespace sdgpu the
// device, its allocations and the types the stages exchange, the error checks, the weight decoders and the host helpers. The backend is
// compiled as several units (kernels.cpp, gemv_a.cu, gemv_b.cu, moe.cu); a kernel is defined in exactly one of them, a function that more than
// one includes is inline, and the state they share is inline variables.
#pragma once
#include "straydog_gpu.h"
#if defined(__HIPCC__)
#define SD_KERNEL_ACQUIRE __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")
#else
#define SD_KERNEL_ACQUIRE (void)0
#endif
#if defined(__HIPCC__)
#include <dlfcn.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#define SD_ALIGN(n) __attribute__((aligned(n)))
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#define hipblasHandle_t cublasHandle_t
#define hipblasStatus_t cublasStatus_t
#define HIPBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define HIPBLAS_OP_N CUBLAS_OP_N
#define HIPBLAS_OP_T CUBLAS_OP_T
#define HIPBLAS_DEFAULT_MATH CUBLAS_DEFAULT_MATH
#define hipblasCreate cublasCreate
#define hipblasDestroy cublasDestroy
#define hipblasSetStream cublasSetStream
#define hipblasSetWorkspace cublasSetWorkspace
#define hipblasSetMathMode cublasSetMathMode
#define hipblasSgemm cublasSgemm
#define hipblasGemmEx cublasGemmEx
#define hipblasGemmStridedBatchedEx cublasGemmStridedBatchedEx
#define HIP_R_16F CUDA_R_16F
#define HIP_R_32F CUDA_R_32F
#define HIPBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define HIPBLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipErrorOutOfMemory cudaErrorMemoryAllocation
#define hipGetErrorString cudaGetErrorString
#define hipSetDevice cudaSetDevice
#define hipGetDeviceCount cudaGetDeviceCount
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipFuncAttributes cudaFuncAttributes
#define hipFuncGetAttributes cudaFuncGetAttributes
#define hipMemGetInfo cudaMemGetInfo
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventElapsedTime cudaEventElapsedTime
#define hipStream_t cudaStream_t
#define hipStreamCreate cudaStreamCreate
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamCreateWithPriority cudaStreamCreateWithPriority
#define hipDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define hipStreamDefault cudaStreamDefault
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpy2DAsync cudaMemcpy2DAsync
#define hipMemcpyPeerAsync cudaMemcpyPeerAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipMemcpyDefault cudaMemcpyDefault
#define hipGetLastError cudaGetLastError
#define hipMemsetAsync cudaMemsetAsync
#define hipMemcpyToSymbol cudaMemcpyToSymbol
#define HIP_SYMBOL(symbol) symbol
#define hipGraph_t cudaGraph_t
#define hipGraphExec_t cudaGraphExec_t
#define hipGraphDestroy cudaGraphDestroy
#define hipGraphExecDestroy cudaGraphExecDestroy
#define hipGraphInstantiate cudaGraphInstantiate
#define hipGraphLaunch cudaGraphLaunch
#define hipStreamBeginCapture cudaStreamBeginCapture
#define hipStreamEndCapture cudaStreamEndCapture
#define hipStreamIsCapturing cudaStreamIsCapturing
#define hipStreamCaptureStatus cudaStreamCaptureStatus
#define hipStreamCaptureStatusNone cudaStreamCaptureStatusNone
#define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal
#define hipHostMalloc(pointer,bytes,flags) cudaHostAlloc(pointer,bytes,flags)
#define hipHostMallocDefault cudaHostAllocDefault
#define hipHostMallocMapped cudaHostAllocMapped
#define hipHostFree cudaFreeHost
#define hipHostGetDevicePointer cudaHostGetDevicePointer
#define hipHostRegister cudaHostRegister
#define hipHostUnregister cudaHostUnregister
#define hipHostRegisterPortable cudaHostRegisterPortable
#define hipHostRegisterReadOnly cudaHostRegisterReadOnly
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipStreamWaitEvent cudaStreamWaitEvent
#define hipEventSynchronize cudaEventSynchronize
#define hipEventQuery cudaEventQuery
#define __shfl_xor(...) __shfl_xor_sync(0xffffffffu, __VA_ARGS__)
#define __shfl_up(...) __shfl_up_sync(0xffffffffu, __VA_ARGS__)
#define __shfl_down(...) __shfl_down_sync(0xffffffffu, __VA_ARGS__)
#define __shfl(...) __shfl_sync(0xffffffffu, __VA_ARGS__)
#define hipStreamQuery cudaStreamQuery
#define hipErrorNotReady cudaErrorNotReady
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define hipMemcpy cudaMemcpy
#define hipMemset cudaMemset
#define hipGetDevice cudaGetDevice
#define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define hipErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define hipErrorNotSupported cudaErrorNotSupported
#define hipHostMallocPortable cudaHostAllocPortable
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventBlockingSync cudaEventBlockingSync
#define hipDeviceSynchronize cudaDeviceSynchronize
// (CUDA's page-locked memory is coherent with the devices that map it)
#define hipHostMallocCoherent cudaHostAllocDefault
// MSVC host compilation has no GNU attributes; CUDA's __align__ maps to the host's form.
#define SD_ALIGN(n) __align__(n)
#include <mma.h>
#if defined(_M_X64)
#include <intrin.h>   // _mm_pause in the hybrid export spin
#endif
#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>   // PrefetchVirtualMemory: read-ahead hints for mapped expert banks
#include <dxgi1_4.h>
#endif
#endif
#if defined(__linux__)
#include <sys/mman.h>   // madvise: read-ahead hints for mapped expert banks
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <condition_variable>
#include <cstddef>   // (before the namespace below: a header in it that includes this first would declare a `std` inside it)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sdgpu {
inline thread_local std::string last_error;
#if !defined(__HIPCC__)
inline std::mutex registered_host_allocations_mutex;
inline std::set<void*> registered_host_allocations;
#endif
constexpr size_t max_cols = 16384, max_rows = 262144;
constexpr size_t max_batch = 256;
// The rows of a prompt batch a decoder workspace holds at most. The experts of a batch run in chunks of at most moe_chunk_rows rows (a layer part of
// qwen_parts.hpp), so a wider batch costs the experts' workspace no more than a batch of that many does: the other stages' row scratch (a row of whole
// layers takes batch_scratch_row floats, the largest of hc_batch_floats, gdn_batch_floats and qsa_batch_floats; a row of a layer in parts part_scratch_row,
// checked where they are declared) is what grows with it.
constexpr size_t prefill_rows_max = 16384, moe_chunk_rows = 4096, batch_scratch_row = 25924, part_scratch_row = 13000;
// The floats of a resident MoE plan's workspace a row of its experts' chunk takes: the gate and up rows' scratch, the route outputs and the rest of
// step_moe's layout, 49302; a layer in parts runs its prompt batches on the grouped FP16 path, whose products and inputs are halves within the gate and up
// scratch (11 routes x 640 and 2560 halves a row, 4800 floats) and which has no F32 product rows: 32982.
constexpr size_t moe_compact_gate_up_row = 4800, moe_compact_row = moe_compact_gate_up_row + 11 * 2560 + 22;
constexpr size_t moe_row_floats(bool parts) { return parts ? moe_compact_row : 49302; }
// The floats of a resident MoE plan's workspace, which also holds a batch's row scratch, for batches of up to `rows` rows: its experts' chunk and the
// 65536 of their tables (step_moe), the other stages' row scratch, and what a decode pass of up to eight rows reads (token_moe's layout, 49302 a row, and
// moe_tokens_experts' scores beyond them; up to sixteen rows keep the size every batch had, whichever the layers).
constexpr size_t moe_workspace_floats(size_t rows, bool parts) {
    return std::max({std::min(rows, moe_chunk_rows) * moe_row_floats(parts) + 65536, rows * (parts ? part_scratch_row : batch_scratch_row),
                     std::min<size_t>(rows, 16) * 49302 + 65536});
}
struct Matrix {
    const uint8_t* data;
    size_t row_bytes, rows, cols, input_offset, output_offset, launch_offset;
    int kind;
    uint32_t layout=0;   // Qwen routed gate/up: 0 native, 1 capacity-preserving SoA.
};
static_assert(sizeof(Matrix)==64);
struct ExpertBankLayout { int kind; size_t rows,cols,count; uint32_t layout; bool packing=false; };
// One GEMV row range: lanes cooperating per row (0 selects the reference decoder).
struct GemvItem { Matrix matrix; uint32_t first_wave, lanes; };
// Every GGML type the GPU kernels decode (weight<kind>); dispatch sites expand it.
#define SD_FORMATS(X) X(0) X(1) X(2) X(6) X(8) X(10) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) \
    X(29) X(30) X(39) X(42)
// Allocations carry a small tail so aligned word loads of two-byte-aligned
// quant blocks never cross the end of an owned buffer.
constexpr size_t allocation_tail = 64;
struct DirectPacket { uint16_t header, setup, wx, wy, wz, pad0; uint32_t gx, gy, gz, priv, group; uint64_t object; uint64_t args; uint64_t pad1; uint64_t signal; };
static_assert(sizeof(DirectPacket)==64);
struct CachedGraph {
    hipGraph_t graph=nullptr;
    hipGraphExec_t executable=nullptr;
    std::vector<void*> dependencies;
    std::vector<DirectPacket> direct;
    int direct_state=0;
};
struct ResidentMoe {
    std::vector<Matrix> weights;
    std::vector<void*> dependencies;
    void* table=nullptr;
    float* workspace=nullptr;
    size_t experts=0, capacity=0;
    // Original routed bank geometry, before the verified load fills its allocation.
    Matrix routed_banks[2]{};
    size_t routed_count=0;
    bool pack_routed=false;
    // Partial residency: routed experts live in device slots (null data when absent)
    // and are otherwise read from the authenticated host mapping.
    bool hybrid=false;
    bool layer_prefill=false;   // this layer's routed banks are held for every prompt chunk
    bool drafter=false;         // the MTP draft layer (48): on CUDA its missed routes go to the CPU, whatever the assist level (hybrid_level)
    const uint8_t* host[3]{};
    // host points at the engine's page-locked copies of the banks, not at the model's mapping (sd_hybrid_repoint): nothing is read ahead of them.
    bool host_copy=false;
    // With host_copy: the start of the allocation each bank of the copy lies in (a bank begins inside it; sd_hybrid_assist maps an allocation by its start).
    const uint8_t* host_base[3]{};
    size_t host_bytes[3]{};
    std::vector<int32_t> slot_of;
    uint8_t slot_group=0;   // the slots its experts take (expert_slots_carve)
    std::vector<double> usage;
    // The decode and verification rows routed to each expert, to one in a slot and not (hybrid_host_experts): the warm-up state's counts.
    std::vector<uint32_t> routed_gpu, routed_host;
    const uint8_t* mapped[3]{};   // device view of page-locked host banks (zero-copy assist)
    // The pinned tier (sd_hybrid_tier): per role, each expert's host bytes in page-locked memory of the engine's, null where the expert has
    // none (its bank's bytes are read); empty where the layer has no tier. Read through expert_host.
    std::vector<const uint8_t*> tier[3];
    std::vector<uint32_t> host_read;   // the token each expert was last read on the host (hybrid_host_experts)
    uint32_t host_token=0;
    // Expert parallelism: this device holds only some routed experts (the
    // others have null data and are skipped) and computes a partial combination.
    bool part=false, shared=true;
    std::vector<std::chrono::steady_clock::time_point> prefetched;   // last read-ahead hint per expert
    // A layer's part (qwen_parts.hpp): its half of the shared expert's width, rows [320p, +320) of gate and up and those
    // columns of down (no data: none). Decode runs it as the shared expert (weights and table); a prompt's grouped experts
    // run without it (batch_table: the table without the shared expert) and it beside them.
    Matrix half[3]{};
    void* batch_table=nullptr;
    std::vector<unsigned> last_counts;   // the rows each expert had in the last prompt batch through this layer (prestage's guess of the next)
};
// An expert's bytes of `role` on the host, wherever its layer keeps them (the bank's mapping or its copy): every reader of a routed expert's
// host bytes goes through this, so that a layer's experts need not sit at one stride from one base.
inline const uint8_t* expert_host(const ResidentMoe& plan, size_t role, size_t expert) {
    if (!plan.tier[role].empty() && plan.tier[role][expert]) return plan.tier[role][expert];
    return plan.host[role] + expert * plan.host_bytes[role];
}
// Whether the expert's host bytes are in the pinned tier (copies by DMA whatever its layer's banks are).
inline bool expert_tiered(const ResidentMoe& plan, size_t expert) { return !plan.tier[0].empty() && plan.tier[0][expert]; }
// Whether `next` follows `expert` in host memory in every role (one copy can take both).
inline bool expert_host_follows(const ResidentMoe& plan, size_t expert, size_t next) {
    for (size_t role = 0; role < 3; ++role)
        if (expert_host(plan, role, next) != expert_host(plan, role, expert) + plan.host_bytes[role]) return false;
    return true;
}
#if defined(__linux__)
// Read-ahead hints off the compute thread. One MADV_WILLNEED reads at most the device's largest
// request (e.g. 256 KiB) of its range, and submitting it runs at device speed, so this thread
// hints queued ranges in 128 KiB steps while prefill computes. Hints only; results never change.
class ReadAhead {
public:
    ReadAhead() : thread_([this]() { run(); }) {}
    ~ReadAhead() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        wake_.notify_one();
        thread_.join();
    }
    void add(const std::vector<std::pair<const uint8_t*, size_t>>& ranges) {
        { std::lock_guard<std::mutex> lock(mutex_); queue_.insert(queue_.end(), ranges.begin(), ranges.end()); }
        wake_.notify_one();
    }
private:
    void run() {
        const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE)), step = 128 * 1024;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            wake_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
            if (stop_) return;
            const auto range = queue_.front();
            queue_.pop_front();
            lock.unlock();
            const uintptr_t end = uintptr_t(range.first) + range.second;
            for (uintptr_t at = uintptr_t(range.first) & ~(page - 1); at < end; at += step)
                madvise(reinterpret_cast<void*>(at), size_t(std::min(step, end - at)), MADV_WILLNEED);
            lock.lock();
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::pair<const uint8_t*, size_t>> queue_;
    bool stop_ = false;
    std::thread thread_;   // last: starts after the members it uses
};
#elif defined(_WIN32)
// Read-ahead hints off the compute thread, as on Linux: PrefetchVirtualMemory walks the pages of every range it is given, those in
// memory too, and the thread that asks waits for it (a 765-token continuation of Qwen3.8 UD-IQ1_S on an RTX 5070 Ti at --vram 5.5:
// 24-29 ms a layer, 1.2-1.4 s a prompt, asked by the thread that then gathers the layer). Here no layer waits for it: a cold expert is
// still asked for in large concurrent reads, ahead of the gather's page faults. What is queued while a call runs goes in the next one.
class ReadAhead {
public:
    ReadAhead() : thread_([this]() { run(); }) {}
    ~ReadAhead() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        wake_.notify_one();
        thread_.join();
    }
    void add(const std::vector<std::pair<const uint8_t*, size_t>>& ranges) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& range : ranges) queued_.push_back({const_cast<uint8_t*>(range.first), range.second});
        }
        wake_.notify_one();
    }
private:
    void run() {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            wake_.wait(lock, [this]() { return stop_ || !queued_.empty(); });
            if (stop_) return;
            entries.clear();
            entries.swap(queued_);
            lock.unlock();
            PrefetchVirtualMemory(GetCurrentProcess(), ULONG_PTR(entries.size()), entries.data(), 0);
            lock.lock();
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<WIN32_MEMORY_RANGE_ENTRY> queued_;
    bool stop_ = false;
    std::thread thread_;   // last: starts after the members it uses
};
#endif
// Fixed-size device slots holding one routed expert's gate, up and down slices.
struct ExpertCacheChunk {
    uint8_t* base=nullptr;
    size_t bytes=0, stride=0, count=0;
    uint8_t group=0;
    size_t reserve=0;
};
struct RetainedExpert {
    ResidentMoe* plan=nullptr;
    size_t expert=0;
    uint8_t* original=nullptr;
    bool temporary=false, moved=false;
};
struct ExpertCacheGrowth {
    size_t arena=0, expected_fixed=0, expected_reserve=0;
    std::vector<size_t> stride, counts;
    std::vector<RetainedExpert> target;
};
struct ExpertSlots {
    uint8_t* base=nullptr;
    uint8_t* host=nullptr;   // page-locked mirror of the staging slots: prefill gathers absent experts here
    size_t cache=0, stage=0, bytes=0;   // slots, staging slots, a staging slot's bytes
    size_t chunk=0;                     // of the staging slots, those a prompt batch streams its experts through in chunks; the rest (prestage) hold a layer's experts ahead
    size_t arena=0;                     // pool bytes apart from nominal staging; contiguous before stage only in legacy mode
    size_t soa_scratch=0;               // conversion payload size; legacy arena tail or a separate fixed allocation
    std::vector<size_t> at;             // legacy offsets; segmented slots use pointers below
    std::vector<uint8_t> group;         // and only a slot of its size group
    std::vector<std::pair<ResidentMoe*,int32_t>> owner;
    std::vector<size_t> pending;   // decode swaps still loading on the copy stream
    // Opt-in cache allocations are independent; base then owns only the fixed stage.
    // (gpu.rs's kv_cache_retention_requested: unless STRAYDOG_KV_CACHE_RETAIN=0)
    bool retained_requested=[]() { const char* v=std::getenv("STRAYDOG_KV_CACHE_RETAIN"); return !(v && std::strcmp(v,"0")==0); }();
    bool segmented=false, retained_disabled=false, retained_oom=false;
    bool restore_cap_live=false;
    size_t restore_cap=0;
    uint8_t* scratch=nullptr;
    uint8_t* padding=nullptr;
    uint8_t* untracked_kv=nullptr;       // owns malloc before the allocation map can accept it
    size_t untracked_kv_bytes=0;
    size_t untracked_kv_reserve=0;
    size_t scratch_bytes=0, padding_bytes=0, stage_bytes=0;
    size_t base_reserve=0, scratch_reserve=0, padding_reserve=0;
    std::vector<ExpertCacheChunk> chunks;
    std::vector<uint8_t*> pointers;
    std::unique_ptr<ExpertCacheGrowth> growth;
    uint64_t retained_stats[16]{};
};
struct DecoderLayer {
    std::vector<void*> pointers,dependencies;
    std::vector<int> kinds;
    void* states[3]{};
    void* moe=nullptr;
    float* workspace=nullptr;
    size_t context=0,index=0,capacity=0;
    float* snapshot=nullptr;      // speculative verification: DeltaNet states before the tokens
    float* projections=nullptr;   // and the tokens' DeltaNet projections for replay
    size_t verified=0;            // tokens of the last uncommitted verification
    size_t parts=1,part=0;        // the layer split across GPUs: this device's part of `parts` (QwenHeads)
    // The other window's (Device::window): a pipelined pass keeps its own, so the pass behind it does not overwrite what a partial
    // acceptance of the first replays from. sd_set_window swaps each pair, so `snapshot` etc. are always the window in force.
    float* snapshot_alt=nullptr; float* projections_alt=nullptr; size_t verified_alt=0;
    uint64_t touched_epoch=0,touched_epoch_alt=0;   // successful graph issues in the window whose snapshot is active
};

// A layer's share of Qwen3.8's heads: P = 1 the whole layer, 2 one GPU's half of it (docs/DESIGN.md §8). A half is the
// same layer with half the heads: 8 DeltaNet key heads (value head h reads key head h % 8, as the whole layer's reads h % 16)
// and 24 value heads; 12 attention query heads with one K/V head.
template<int P> struct QwenHeads {
    static_assert(P == 1 || P == 2, "Qwen3.8 layers split in two at most");
    static constexpr size_t key = 16 / P, value = 48 / P;                  // DeltaNet heads
    static constexpr size_t channels = 10240 / P, gate = 6144 / P;        // convolved columns (q | k | v); gate values
    static constexpr size_t beta = channels + gate, alpha = beta + value;  // a token's projections: qkv | gate | beta | alpha
    static constexpr size_t projection = alpha + value;
    static constexpr size_t query = 24 / P, kv = 2 / P;                    // attention: query heads, K/V heads
    static constexpr size_t qsa_projection = query * 512 + kv * 512 + 640; // qgate | key | value | index query | index key
    static constexpr size_t packed = query * 256 + kv * 512;               // a row's normalized queries, keys, values
};
constexpr size_t gemm_elements = 32 * 1024 * 1024, blas_workspace_bytes = 16 * 1024 * 1024;
// A pass reads from a mirror the rows at least this many newer than the ring's oldest: a verification's last rows taken back (up to
// verify_max) leave what it reads in place, with nothing to copy again.
constexpr size_t kv_mirror_margin = 8;
// Qwen3.8's attention K/V: its planes of `context` positions (each K/V head's keys, then each one's values), a plane's row of a
// position one head's 256 values in the device's format (Device::kv_row, sd_kv_format): FP16, a row of 512 bytes; or Q8_0, the 256
// values as int8 and then the FP16 scales of their eight 32-value blocks, 272 bytes (17 pieces of 16: every row stays 16-byte aligned
// and is read in one run). kv_t is a byte of it; copies, stage and mirrors count in rows of Device::kv_row bytes.
typedef uint8_t kv_t;
constexpr unsigned kv_row_f16 = 512, kv_row_q8 = 272;
constexpr size_t attention_score_bytes = max_batch * 24 * 2052 * 4;   // decode attention partials (QSA prefill marks)
constexpr size_t io_arena_bytes = size_t(32) << 20;   // page-locked transfer staging per device

// What a failed GPU call says, never NULL. The CUDA runtime returns NULL from cudaGetErrorString when the NVIDIA driver cannot be loaded
// (and std::string(NULL) reads address 0), and its own words for a driver that is too old do not say what to do. `runtime` is what the
// runtime answered (it may be NULL). The CUDA codes mapped to the driver sentence: cudaErrorStubLibrary 34 (a stub nvcuda loaded),
// cudaErrorInsufficientDriver 35 (the driver is missing or older than the runtime: what cudaGetDeviceCount returns without nvcuda.dll)
// and cudaErrorSystemDriverMismatch 803; cudaErrorNoDevice 100 is a driver with no NVIDIA GPU.
inline const char* gpu_error_text_of(hipError_t error, const char* runtime) {
#if !defined(__HIPCC__)
    if (error == cudaErrorInsufficientDriver || error == cudaErrorStubLibrary || error == cudaErrorSystemDriverMismatch)
        return "The NVIDIA driver is missing or too old to run CUDA 13: install or update the NVIDIA driver to version 580 or later (the latest is recommended)";
    if (error == cudaErrorNoDevice) return "No NVIDIA GPU was found";
#endif
    if (runtime != nullptr && *runtime != '\0') return runtime;
    thread_local std::string unknown;
    unknown = "GPU error " + std::to_string(int(error));
    return unknown.c_str();
}
inline const char* gpu_error_text(hipError_t error) { return gpu_error_text_of(error, hipGetErrorString(error)); }

// The site and the code go with the message: the same words can mean memory that ran out or a device that was reset.
inline void check_at(hipError_t error, const char* file, int line) {
    if (error == hipSuccess) return;
    const std::string path(file);
    throw std::runtime_error(std::string(gpu_error_text(error)) + " (" + path.substr(path.find_last_of("/\\") + 1) + ":" + std::to_string(line) +
                             ", code " + std::to_string(int(error)) + ")");
}
#define check(error) check_at((error), __FILE__, __LINE__)
// An op of the API: the device the handle names, and the failure as the error text.
#define SD_OP_BEGIN try { auto& value = device(handle);
#define SD_OP_END check(hipGetLastError()); return 0; } catch (const std::exception& e) { last_error = e.what(); return -1; }

inline void blas_check(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Straydog BLAS failure: " + std::to_string(int(status)));
}

inline void cleanup(hipError_t error) noexcept {
    if (error != hipSuccess) std::fprintf(stderr, "Straydog GPU cleanup: %s\n", gpu_error_text(error));
}

// Page-locked host memory for what a request asks (a K/V growth, the MTP rows' staging, a read's buffer). Tests make the next
// `refuse_host_allocations` of these answer as a machine short of memory does (sd_test_refuse_host).
inline std::atomic<uint64_t> refuse_host_allocations{0};
// Tests: the streamed chunk of a prompt batch's layer that fails, counted from the next one (0: none; sd_test_fail_streamed, hybrid_stream_missing).
inline std::atomic<uint64_t> fail_streamed_chunk{0};
inline hipError_t host_allocate(void** pointer, size_t bytes, unsigned int flags) {
    uint64_t left = refuse_host_allocations.load();
    while (left && !refuse_host_allocations.compare_exchange_weak(left, left - 1)) {}
    if (left) { *pointer = nullptr; return hipErrorOutOfMemory; }
    return hipHostMalloc(pointer, bytes, flags);
}

// A BLAS handle for F32 products whose kernel choice is the same in every process. On ROCm,
// rocBLAS hands GEMMs to hipBLASLt, whose ranking of F32 kernels changes with the process's
// address layout on RDNA4 (one process in five to thirty ranks them otherwise, for every shape
// at once; with address randomization off, never; its FP16 ranking is stable), so the same F32
// product summed in another order and gave other bits from one run to the next. rocBLAS's own
// kernel library chooses from a fixed table. rocBLAS reads ROCBLAS_USE_HIPBLASLT when a handle
// is created, so this handle is created with it set to 0, and the variable is put back.
// Changing the environment races with getenv on other threads: it is done at device creation,
// before this device's readers and loaders start, under one lock for the process; what can still
// read the environment meanwhile is the HIP runtime's own threads.
inline hipblasHandle_t create_f32_blas() {
    hipblasHandle_t handle = nullptr;
#if defined(__HIPCC__) && !defined(_WIN32)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    constexpr const char* name = "ROCBLAS_USE_HIPBLASLT";
    const char* set = std::getenv(name);
    const std::string before = set ? set : "";
    setenv(name, "0", 1);
    const hipblasStatus_t status = hipblasCreate(&handle);
    if (set) setenv(name, before.c_str(), 1);
    else unsetenv(name);
    blas_check(status);
#else
    blas_check(hipblasCreate(&handle));
#endif
    return handle;
}

// One job at a time on a thread of its own with the device current. Bounded slot
// swaps issue their copies from pageable mappings here (hybrid_rebalance), so the
// decode thread never blocks in them. Destruction finishes a queued job first.
class Background {
public:
    explicit Background(int ordinal) : thread_([this, ordinal]() { loop(ordinal); }) {}
    ~Background() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        wake_.notify_all();
        thread_.join();
    }
    bool busy() const { return busy_.load(std::memory_order_acquire); }
    void run(std::function<void()> job) {
        wait();
        { std::lock_guard<std::mutex> lock(mutex_); job_ = std::move(job); busy_.store(true, std::memory_order_release); }
        wake_.notify_all();
    }
    // Waits for the job and rethrows its failure.
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&]() { return !busy_.load(std::memory_order_acquire); });
        if (error_) { std::exception_ptr error = error_; error_ = nullptr; std::rethrow_exception(error); }
    }
private:
    void loop(int ordinal) {
        cleanup(hipSetDevice(ordinal));
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            wake_.wait(lock, [&]() { return stop_ || bool(job_); });
            if (!job_) return;
            std::function<void()> job = std::move(job_);
            job_ = nullptr;
            lock.unlock();
            std::exception_ptr error;
            try { job(); } catch (...) { error = std::current_exception(); }
            lock.lock();
            error_ = error;
            busy_.store(false, std::memory_order_release);
            done_.notify_all();
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::function<void()> job_;
    std::exception_ptr error_;
    std::atomic<bool> busy_{false};
    bool stop_ = false;
    std::thread thread_;   // last: starts once the members above exist
};

// A K/V's newest rows in VRAM, a ring of Device::kv_mirror_rows rows (position % rows): rows [oldest, valid) are there. A token selects
// recent rows: IQ3_XXS on an RTX 5070 Ti at --vram 8.5, a 15K-token conversation, a mirror of 2901 rows of the same VRAM held 3-10% of the
// rows the 12 K/V layers selected when it was their first rows and holds 32-44% as their newest; a K/V layer 480 -> 350 us, 39 -> 44 tok/s.
struct KvMirror { kv_t* data; size_t valid; size_t oldest = 0; };
// A host exchange (host_exchange.hpp): its page-locked host memory, the device's view of it, and the
// device's count of exports (an allocation of the device).
struct ExchangeArea {
    uint8_t* host = nullptr;
    uint8_t* view = nullptr;
    uint32_t* count = nullptr;
    size_t bytes = 0;
};
struct Device;
// Two devices' tensor-parallel sums (device_link.hpp): this device's inbox (two slots of
// `capacity` floats, alternating by sum) and state ([0] sums sent, [1] the flag the peer raises,
// [2] nonzero after a wait gave up), both in its own VRAM, and the peer's as this device writes them.
struct SumLink {
    Device* peer = nullptr;
    float* inbox = nullptr;
    float* peer_inbox = nullptr;
    uint32_t* state = nullptr;
    uint32_t* peer_flag = nullptr;
    size_t capacity = 0;
    uint64_t skew = 0;   // sd_link_skew's, in clock ticks
};
// A channel of two devices' part link (device_link.hpp: sd_link_parts, link_tail.hpp): this
// device's inbox (two slots of `capacity` floats, which the other writes; zero where it never
// does) and state, both in its own VRAM, and the other's as this device writes them.
struct PartLink {
    float* inbox = nullptr;
    float* peer_inbox = nullptr;
    uint32_t* state = nullptr;
    uint32_t* peer_state = nullptr;
    size_t capacity = 0;
};
// This device's side of a two-GPU link: an inbox the peer writes rows
// into, a flag the peer raises per message, and this device's message counters.
struct PeerLink {
    Device* peer=nullptr;
    unsigned index=0;            // 0 or 1: the order partial sums are added in
    float* inbox=nullptr;        // two areas of `rows` rows of 2560 floats: inputs, then partial sums
    float* peer_inbox=nullptr;   // the peer's inbox, as this device writes it
    uint32_t* flag=nullptr;      // raised by the peer, once per message
    uint32_t* peer_flag=nullptr;
    uint32_t* counters=nullptr;  // [0] messages sent, [1] received, [2] nonzero after a wait gave up
    int32_t* routes=nullptr;     // after the inbox: the owner's top ten (rows x 10) then probabilities (rows x 11)
    int32_t* peer_routes=nullptr;
    size_t rows=0;
    bool direct=false;           // inbox and flag in this device's VRAM (peer access), else host memory
    void* host=nullptr;          // the page-locked allocation when !direct
};
// Two page-locked buffers for rows a stream copies in turn (stage_rows).
struct RowStaging {
    float* host[2] = {nullptr, nullptr};
    hipEvent_t staged[2] = {nullptr, nullptr};
    size_t floats = 0;
    size_t refused = 0;   // the size (floats) the system refused to page-lock last: not asked for again at or over it
    unsigned turn = 0;
};
// Layer order's hidden rows kept on the host (page-locked for the pass) and moved beside the compute (sd_rows_overlap_*, kernels.cpp): a ring of
// two chunk slots in VRAM stands for the rows a pass would hold there, fed by the up stream and drained by the down stream, the device stream
// working between them. Everything is ordered by events: a slot is written only after its rows have left it, and a chunk's rows are read
// from the host only after the layer before has written them there.
struct RowsOverlap {
    // A copy's start and end on its stream (or the device stream's wait for rows coming up), summed into the device's counters when done.
    struct Timed { hipEvent_t first = nullptr, last = nullptr; int kind = 0; };   // kind: 0 up, 1 down, 2 the device stream's wait
    enum SlotState : uint8_t { slot_free, slot_uploaded, slot_loaded };   // rows queued up into it; copied on into the layer's workspace
    hipStream_t up = nullptr, down = nullptr;
    // By slot: its rows are on the device (queued on up); the span's rows are in it (device stream); they have reached the host (down).
    hipEvent_t uploaded[2] = {nullptr, nullptr}, computed[2] = {nullptr, nullptr}, left[2] = {nullptr, nullptr};
    // By chunk: its rows have reached the host (the last layer's), and whether a layer of this pass has written them (else the host's are the CPU's).
    std::vector<hipEvent_t> written;
    std::vector<char> valid;
    SlotState state[2] = {slot_free, slot_free};
    size_t slot_steps[2] = {0, 0};        // the rows queued up into it
    bool slot_used[2] = {false, false};   // `left` is of a copy out of the slot
    bool active = false;
    void* ring = nullptr;          // the ring's allocation (sd_prefill_bank_allocate): `slots` slots of `slot_rows` rows
    size_t slot_rows = 0, slots = 0;
    float* host = nullptr;         // the pass's rows, page-locked from sd_rows_overlap_begin to its end
    size_t host_rows = 0;
    std::vector<Timed> timed, spare;
    ~RowsOverlap() {
        if (up) cleanup(hipStreamSynchronize(up));
        if (down) cleanup(hipStreamSynchronize(down));
        if (active && host) cleanup(hipHostUnregister(host));
        for (hipEvent_t event : {uploaded[0], uploaded[1], computed[0], computed[1], left[0], left[1]}) if (event) cleanup(hipEventDestroy(event));
        for (hipEvent_t event : written) if (event) cleanup(hipEventDestroy(event));
        for (auto* list : {&timed, &spare})
            for (auto& entry : *list) {
                if (entry.first) cleanup(hipEventDestroy(entry.first));
                if (entry.last) cleanup(hipEventDestroy(entry.last));
            }
        if (up) cleanup(hipStreamDestroy(up));
        if (down) cleanup(hipStreamDestroy(down));
    }
};
struct Device;
void phase_report(Device& d);
// The flag that tells the span running on a device to stop (sd_abort_flag): its own counted allocation, one reference the device's and one
// each holder's, so that whoever raises it from another thread can never write to freed memory, whatever became of the device.
struct AbortFlag { std::atomic<bool> raised{false}; std::atomic<uint32_t> references{1}; };
// A reference more (as the standard library's shared pointers do: a count past half its range means something is leaking references, and
// continuing would let it wrap to zero and free a flag that is held).
inline void abort_flag_retain(AbortFlag* flag) {
    if (flag->references.fetch_add(1, std::memory_order_relaxed) >= UINT32_MAX / 2) std::abort();
}
inline void abort_flag_release(AbortFlag* flag) {
    if (flag && flag->references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete flag;
}
// Device::flags[5]: deferred prefill spans' non-finite check (sd_prefill_defer, sd_prefill_settle).
constexpr size_t prefill_deferred_flag = 5;
struct Device {
    int ordinal;
    size_t budget = 0, used = 0;
    // Requested bytes remain in used. Independently owned retained allocations
    // also keep a conservative allocator charge until their checked free succeeds.
    size_t retained_reserve=0, retained_reserve_peak=0, cpu_partial_reserve=0;
    std::vector<void*> retained_kv_allocations;
    size_t batch_capacity = 32, input_capacity = 32 * max_cols, output_capacity = max_rows;
    size_t gemm_min_batch = 8;
    bool half_gemm = false;
    bool gdn_chunked = false;   // a long prompt read's DeltaNet layers scan in chunks (gated_delta_chunk.hpp; RDNA4), set around the read (sd_prompt_options)
    bool hc_fused = false;   // a long prompt read's HyperConnection up projection and finish run as one kernel (hc_fused.hpp; RDNA4), set around the read (sd_prompt_options)
    bool half_decode = false;   // a decode or verify pass's index and attention scores through WMMA, FP16 operands (RDNA4: the FP16 prompt pass's)
    bool q8_wide = false;   // a decode row's Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 GEMVs on the Q8 kernels, mixed lists with BF16 rows in one launch (RDNA4 and CUDA: Qwen3.8)
    bool q8_decode = false;   // a decode row's Q5_K GEMVs, IQ2 experts' gate/up and down (IQ2_S's down excepted) and Q5_K shared expert on the Q8 kernels (RDNA4)
    uint64_t* gemm_checks = nullptr;   // optional quality-only cast counters, enabled before graphs
    bool rdna4 = false;
    unsigned compute_units = 0;   // the device's CUs (SMs on CUDA)
    // A grouped FP16 expert GEMM for prefill exists: RDNA4 WMMA (HIP) or tensor cores (CUDA sm_70+).
    bool expert_prefill = false;
    // Decode work is captured and replayed as graphs: RDNA4 (HIP) and every CUDA device.
    bool graph_capture = false;
    void* direct_queue = nullptr;
    uint64_t direct_tail_object = 0;
    int direct_mode = -1;
    int qsa_mma = -1;   // the prompt's sparse attention on tensor cores over Q8_0 K/V, unless STRAYDOG_QSA_MMA=0 (read at the first prompt batch)
    int hc_fused_mode = -1;   // CUDA: STRAYDOG_HC_FUSED=0 keeps a prompt batch's HyperConnection up projection and finish as two kernels (read at the first batch)
    bool mma_f16 = false;   // sm_80+: mma.sync m16n8k16 with FP16 operands (qsa_flash_mma, hc_up_finish_mma)
    bool prompt_q8 = false;   // sm_80+ and STRAYDOG_PROMPT_Q8 not 0: a prompt batch's routed experts on INT8 operands (moe_gemm_cuda_q8.hpp)
    std::unordered_map<uint64_t,bool> direct_fenced;
    // Partially resident decode launches each layer's graph before the previous layer's
    // CPU result exists; the graph waits for it on the GPU (hybrid_await).
    bool hybrid_prelaunch = false;
    bool hybrid_interleave = false;   // sd_hybrid_interleave: a verification's rows as two interleaved groups (STRAYDOG_INTERLEAVE overrides)
    // Kernels have a run-time limit (Windows display GPUs, 2 s): awaits give up after
    // ~1 s and only follow layers whose CPU experts read page-locked banks. Otherwise
    // an await may wait 10 s for a host stalled on storage.
    bool await_watchdog = false;
    uint64_t await_timeout = 0;   // hybrid_await's limit in GPU timer ticks
    hipStream_t stream = nullptr;
    // Expert-parallel help for the peer's batch, beside this device's own (resident_moe_parts).
    hipStream_t helper_stream = nullptr;
    PeerLink link;
    SumLink sums;
    Device* part_peer = nullptr;
    PartLink part_links[3];
    std::unordered_map<size_t, void*> helper_moe;   // layer -> this device's expert part, when it helps the owner
    uint8_t* io = nullptr;   // page-locked staging for caller-memory transfers (host_io.hpp)
    size_t io_used = 0;
    hipEvent_t io_halves[2] = {nullptr, nullptr};   // a large upload's last copy out of each half
    float* input = nullptr;
    float* output = nullptr;
    GemvItem* matrices = nullptr;
    hipblasHandle_t blas = nullptr;
    hipblasHandle_t blas_f32 = nullptr;   // F32 products: see create_f32_blas
    float* expanded = nullptr;
    size_t expanded_bytes = 0;
    __half* half_input = nullptr;
    float* attention_scores = nullptr;
    void* blas_workspace = nullptr;
    unsigned* flags = nullptr;
    // Deferred prefill (sd_prefill_defer): spans end without a host wait; their non-finite
    // check accumulates in flags[prefill_deferred_flag] until sd_prefill_settle (its own slot:
    // flags[1] is a partially resident model's export count, hybrid_tick, which a layer-order
    // pass's deferral zeroed and the next decode then waited for in vain). `handoff` orders a row
    // handoff between this device's stream and its peer's.
    bool prefill_deferred = false;
    ResidentMoe* layer_prefill_plan = nullptr;
    void* layer_prefill_storage = nullptr;
    // A layer-order pass's bank pool in two halves (load_prefill_bank): the half the layer loaded now uses, and for each half the event after the
    // compute of the layer that last used it, which the next copy into it waits for (and not the compute of the layer between).
    int layer_bank_half = -1;
    hipEvent_t layer_bank_free[2] = {nullptr, nullptr};
    bool layer_bank_used[2] = {false, false};
    // The last bank load's copies (recorded on the copy stream, which the device's stream waits for): the next load waits for them on the host,
    // before it reuses the staging and its caller lets that layer's registration go, a layer's compute later, not the load itself.
    hipEvent_t layer_bank_loaded = nullptr;
    bool layer_bank_loading = false;
    // A bank copied ahead, into the half the computing layer does not use, while that layer computes: the layer, its plan, half and bank,
    // until the layer's turn points the table at it (commit_prefill_bank). One at a time.
    // A layer-order pass's bank loads, counted where they run (the loader's thread too): loads, and in ns their time, their waits for the
    // staging's copies before a gather, and their gathers (sd_prefill_bank_times).
    std::atomic<uint64_t> bank_loads{0}, bank_load_ns{0}, bank_staging_wait_ns{0}, bank_gather_ns{0};
    ResidentMoe* layer_bank_ahead_plan = nullptr;
    size_t layer_bank_ahead_layer = 0;
    int layer_bank_ahead_half = -1;
    uint8_t* layer_bank_ahead_bank = nullptr;
    void* layer_bank_ahead_storage = nullptr;
    // A layer-order pass's chunks' expert counts and last routes, copied to the host behind each chunk's routing on the device's stream and
    // observed for the slots in order once they have landed (observe_deferred), so that the host does not wait for each chunk's routing:
    // a ring of page-locked entries, each with its event, and the entries not yet observed, oldest first.
    struct DeferredObservation { ResidentMoe* plan; size_t steps, tail, slot; };
    uint8_t* deferred_host = nullptr;
    std::vector<hipEvent_t> deferred_events;
    std::deque<DeferredObservation> deferred;
    size_t deferred_next = 0;
#if defined(STRAYDOG_PREFILL_OBSERVER)
    sd_prefill_observer prefill_observer = nullptr;
    void* prefill_observer_context = nullptr;
#endif
    // A sequence with images (sd_rope_delta, sd_rope_rows). The table is allocated at its first image, beside the budget:
    // a few KiB the runtime's reserve holds, needed when the slots have taken all that is counted.
    uint32_t rope_delta = 0;
    uint32_t* rope_rows = nullptr;
    size_t rope_first = 0, rope_count = 0;
    hipEvent_t handoff = nullptr;
    // Host states allocated beside reading the weights (sd_host_states_prefetch): zeroing and
    // page-locking ~13 GB of K/V. The thread alone writes `host_ready` until joined.
    std::thread host_prefetch;
    std::vector<std::pair<size_t, void*>> host_ready;
    std::string host_prefetch_error;
    // Page-locked staging for a prompt batch's rows, two batches deep (stage_rows): a batch's copy waits only for the copy two
    // batches before, not for the stream (upload's arena waits for everything queued when it wraps, which emptied both GPUs
    // between a long prompt's batches). The MTP layer's embeddings (sd_mtp_embed), the first layer's (sd_prefill_embed) and the
    // PLE layer's (sd_prefill_ple).
    RowStaging mtp_rows, prefill_rows, ple_rows;
    // Those three's page-locked buffers (two each) for a wide batch's rows, pinned on a thread of their own beside the weights' reading
    // (sd_row_staging_prefetch) and taken by the first stage_rows, so that the first wide read does not pin them.
    std::thread staging_prefetch;
    float* staging_ready[3][2] = {};
    size_t staging_ready_floats = 0;
    uint64_t gemm_calls = 0, gemm_vectors = 0;
    std::unordered_map<void*, size_t> allocations;
    // `allocations` by address, for the ops' range checks (ops.hpp: op_span); rebuilt when stale.
    std::map<const uint8_t*, size_t> spans;
    std::mutex spans_mutex;
    // sd_op_experts' route addresses (an allocation of slot_route_capacity entries), and the
    // page-locked landing of an asynchronous read (sd_buffer_read_start) with its copy's event.
    uint64_t* slot_routes = nullptr;
    unsigned* route_counter = nullptr;   // sd_op_linear_route's (zero between launches)
    // sd_op_linear_route_published's mailbox, coherent page-locked memory: its flag, then from byte
    // 64 the routes.
    uint8_t* route_mailbox = nullptr;
    // sd_op_sparse_attention's tile counters (zero between launches: each tile's last share resets its own)
    unsigned* attention_tiles = nullptr;
    // sd_op_index_select's row counters (zero between launches: each row's last block resets its own)
    unsigned* index_rows = nullptr;
    void* index_queries = nullptr;   // ds_index_query_rows's FP16 rows (ds_index_queries; HIP)
    // Forward then inverse rope for one pass, reserved once outside capture (sd_attention_turns_reserve).
    // A captured graph keeps the pointer, so it is not grown while a graph is stored.
    double2* attention_turns = nullptr;
    size_t attention_turns_bytes = 0;
    size_t attention_turns_rows = 0;
    size_t attention_turns_pairs = 0;
    size_t attention_turns_armed = 0;   // the next norm_rope of this many rows publishes the table
    bool attention_turns_ready = false;
    size_t attention_turns_ready_rows = 0;
    size_t attention_turns_ready_pairs = 0;
    size_t slot_routes_count = 0;   // the routes uploaded there last (sd_op_experts may reuse them)
    // Or, when the last call passed its few routes by value, those (reused alike).
    uint64_t inline_routes[32] = {};
    size_t inline_routes_count = 0;
    hipEvent_t parts_ready = nullptr;   // sd_op_sum_devices: this device's part is written
    hipEvent_t parts_read = nullptr;    // sd_op_sum_devices: this device has read the others' parts
    hipStream_t sum_stream = nullptr;   // sd_op_sum_devices_start: the copies of the other devices' parts into this one's inbox
    hipEvent_t sum_queued = nullptr;    // sd_op_sum_devices_start: the work queued on the device's stream before the copies
    hipEvent_t sum_copied = nullptr;    // sd_op_sum_devices_start: the copies into this device's inbox
    bool op_recording = false;   // between sd_graph_begin and sd_graph_end
    uint8_t* readback = nullptr;
    bool readback_pinned = true;   // false: pageable (the system refused to page-lock it), held in readback_pageable
    std::vector<uint8_t> readback_pageable;
    hipEvent_t readback_done = nullptr;
    size_t readback_bytes = 0;
    // Allocations that live in mapped, page-locked host memory (sd_host_state): device pointer -> host pointer.
    std::unordered_map<void*, void*> host_states;
    std::vector<void*> retained_host_new; // new host K/V owned until commit or checked cleanup
    // sd_upload's allocations by address (weights: none of them is written after), and the FP16
    // expansions of their matrices that a prompt pass's GEMMs keep for its next batches (op_linear):
    // a ring in `expansions` (sd_expansions).
    std::map<const uint8_t*, size_t> uploaded;
    struct Expansion { const uint8_t* data; size_t rows, cols; int kind; size_t at, bytes; };
    std::vector<Expansion> expanded_weights;
    uint8_t* expansions = nullptr;
    size_t expansions_bytes = 0, expansions_at = 0;
    // Page-locked host copies of recurrent sequence state (conversation checkpoints): pointer -> bytes.
    // Neither VRAM nor mapped state, so they take no budget and no K/V mirror rows.
    std::unordered_map<void*, size_t> checkpoints;
    // A checkpoint taken inside the next prompt batch (sd_capture_at): after the batch's row `row`, each recurrent state the checkpoint
    // holds is copied to its place in `buffer` (one of `checkpoints`, laid out as sd_checkpoint_copy lays it out) by the row loop that
    // carries it, a loop that ends a piece at that row: the states are those of a batch that ended there. No buffer: none.
    struct Capture { size_t row = 0; uint8_t* buffer = nullptr; std::unordered_map<const void*, size_t> offsets; size_t done = 0; } capture;
    // VRAM rows that multi-token prefill uses for host-resident K/V (sd_kv_stage): the copy holds
    // rows [0, kv_stage_valid) of kv_stage_owner's K/V in a kv_stage_rows-row layout.
    float* kv_stage = nullptr;
    size_t kv_stage_floats = 0, kv_stage_valid = 0;
    // Planes of a host K/V state (each K/V head's keys, then each one's values): 4 for a whole layer, 2 for a part
    // (QwenHeads<2>::kv); set as its layer registers.
    std::unordered_map<const void*,size_t> kv_planes;
    unsigned kv_row = kv_row_f16;   // bytes of a K/V row: its format (sd_kv_format), set before any layer registers
    // Layer order's hidden rows to the device (sd_prefill_rows) and back (prefill_span's output), wall nanoseconds and bytes so far
    // (sd_prefill_row_transfers): each timed between waits the transfers make anyway, none added.
    uint64_t rows_up_ns = 0, rows_up_bytes = 0, rows_down_ns = 0, rows_down_bytes = 0;
    // Where rows moved beside the compute (rows_overlap): the copies' own times are the ones above (each between its events on its stream);
    // these are the time the device stream waited for rows to come up, and the time the host waited for the last rows to go down.
    uint64_t rows_overlap_stream_wait_ns = 0, rows_overlap_host_wait_ns = 0;
    std::unique_ptr<RowsOverlap> rows_overlap;
    size_t qwen_parts = 1;    // Qwen3.8's layers split across this many GPUs (DecoderLayer::parts)
    // A draft's vocabulary on the device holding the head (sd_draft_vocabulary): the head's first `draft_rows` rows, and
    // `draft_extra_count` rows of tokens past them, copied into `draft_extras` (their tokens in `draft_extra_tokens`).
    // draft_rows 0: drafts read the whole head.
    size_t draft_rows = 0, draft_extra_count = 0, draft_extra_capacity = 0, draft_extra_row_bytes = 0;
    uint8_t* draft_extras = nullptr;
    int32_t* draft_extra_tokens = nullptr;
    bool head_timing_enabled = false;
    uint64_t head_timing_calls[2]{};
    double head_timing_ms[6]{};
    hipError_t head_timing_error = hipSuccess;
    void* kv_stage_owner = nullptr;
    // A prefetch (kv_prefetch) fills the stage on kv_stream; device-stream work waits for
    // kv_prefetched before it touches the stage while kv_prefetching.
    hipStream_t kv_stream = nullptr;
    // A stage's work beside the device stream's (the shared expert's gate and up, resident_expert_launch), forked and joined by events.
    hipStream_t side_stream = nullptr;
    hipEvent_t side_fork = nullptr, side_join = nullptr;
    // CUDA: a layer's staging copy (hybrid_stage_assist) runs on a stream of its own beside the resident experts' launches, forked after the tick and
    // joined before the staged experts' (hybrid_resident_experts).
    hipStream_t stage_stream = nullptr;
    hipEvent_t stage_fork = nullptr, stage_join = nullptr;
    // A prompt batch's sends to the other device of a layer in parts (qwen_parts.hpp) run on a stream of their own beside the device's
    // compute (STRAYDOG_PARTS_OVERLAP=off keeps them in the device stream's order), forked and joined by events.
    hipStream_t send_stream = nullptr;
    hipEvent_t send_fork = nullptr, send_done = nullptr;
    bool send_pending = false;   // a send is queued on send_stream that the device stream has not yet waited for
    bool parts_overlap = true;
    hipEvent_t kv_released = nullptr, kv_prefetched = nullptr;
    bool kv_prefetching = false;
    // Decode partitions the same stage into fixed per-host-state F32 prefixes.
    // Valid ends are prepared outside graph replay; pointers stay fixed until re-registration.
    size_t kv_mirror_rows = 0;
    std::unordered_map<void*,KvMirror> kv_mirrors;
    std::unordered_map<std::string,CachedGraph> graphs;
    std::unordered_map<void*,std::unique_ptr<ResidentMoe>> resident;
    std::unordered_map<size_t,DecoderLayer> decoder;
    // A span's DeltaNet replay list (verify_span.hpp's GdnReplay entries) by (begin, end): its host bytes and its device copy.
    std::map<std::pair<size_t,size_t>,std::pair<std::vector<uint8_t>,void*>> gdn_replays;
    std::map<std::pair<size_t,size_t>,std::pair<std::vector<uint8_t>,void*>> gdn_replays_alt;   // the other window's (sd_set_window)
    // Pipelined verification (sd_pipeline_setup; CUDA's partially resident layers on two devices): the window of the pass in
    // flight (0 or 1: snapshots, projections, verified marks and replay lists, which sd_set_window swaps), whether a second window
    // is allocated, whether the expert slots stay as they are (no swaps at the end of a span), and the flag a span being abandoned
    // is told by (set from another thread: sd_abort_span).
    size_t window=0;
    uint64_t front_generation=0,front_epoch[2]{}; bool front_active=false;
    size_t front_position[2]{SIZE_MAX,SIZE_MAX},front_rows[2]{};
    uint64_t usage_epoch=0;
    bool pipelined=false, frozen_slots=false;
    bool pipelined_chain=false;   // a draft's head on this device alone reads the draft's vocabulary (sd_pipeline_setup flag 4)
    AbortFlag* abort_flag = new AbortFlag();   // this device's reference of its stop flag (released with the device)
    // The flag is honoured by one entry alone, sd_verify_span (which drains and resyncs after a stopped span, AbortScope): a span run by any
    // other entry (the queue, the drafter's steps, the decode and prompt paths) ignores it, so that no other caller can be stopped midway.
    bool abortable=false;
    bool aborted() const { return abortable && abort_flag->raised.load(std::memory_order_acquire); }
    hipError_t span_drain_error=hipSuccess;
    // What a speculative pass's rows observe (the layers' route scores and counts, hybrid_tokens), as it was before the pass: put back
    // when the pass is given up (sd_hybrid_usage), so that a pass that never was leaves no trace in what the slots are chosen by.
    struct UsageMark { std::vector<double> usage; std::vector<uint32_t> routed_gpu, routed_host, host_read; uint32_t host_token=0; };
    std::unordered_map<ResidentMoe*,UsageMark> usage_marks;
    uint64_t mark_tokens=0, mark_balanced=0; size_t mark_position=SIZE_MAX;
    std::unordered_map<void*,ExchangeArea> exchanges;   // host exchanges by host address (sd_exchange)
    uint64_t graph_replays=0;
    ExpertSlots slots;
    std::unordered_map<const uint8_t*,ExpertBankLayout> expert_layouts;
    uint64_t expert_repack_ns=0, expert_repack_bytes=0;
    // Prestage (hybrid_slots.hpp): a wide prompt batch's layer asks for the experts its slots do not hold that its last batch used most, before
    // its router has run, so that they come over the bus beside the layer's own work.
    struct Prestage {
        ResidentMoe* plan=nullptr;
        std::vector<size_t> experts;          // ascending; the i-th is at position i of the prestage region's three role arrays
        std::vector<int32_t> position;        // by expert: its position there, or -1
        hipEvent_t event=nullptr;
    } prestage;
    hipStream_t copy_stream=nullptr;
    void* cpu=nullptr;
    float* exchange=nullptr;      // pinned host: selected, probabilities, activation, CPU result
    float* cpu_partial=nullptr;   // device copy of the CPU expert result
    uint8_t* expert_q8_scratch=nullptr;   // what a pass's INT8 routed gate and up read (resident_expert_q8.hpp), made with the device
    float* exchange_device=nullptr;   // the pinned exchange as seen by kernels
    uint64_t cpu_experts=0, gpu_experts=0, assisted_experts=0, streamed_experts=0, swapped_experts=0, hybrid_tokens=0;
    // A prompt batch's experts beyond the slots that the CPU took (prompt_cpu_share), and what a streamed expert and a CPU route
    // cost in the last batches (ns, a moving average; 0 before the first measurement)
    bool prompt_cpu=true;   // sd_hybrid_prompt_share
    uint64_t prompt_cpu_routes=0;
    double prompt_stream_ns=0, prompt_cpu_ns=0;
    double prompt_scale=0.7;   // the CPU's share of the GPU's time it is given: moved by how the last batches' CPU and GPU finished
    double prompt_gather_ns=0;   // what gathering an expert from pageable banks into pinned memory cost in the last batches (ns)
    float* prompt_host=nullptr;   // pinned (or pageable where pinning was refused): a batch's rows to the CPU, its routes' results back
    size_t prompt_host_floats=0;
    bool prompt_host_pinned=false;
    std::vector<float> prompt_pageable;
    hipEvent_t prompt_rows=nullptr;   // the rows have reached prompt_host
    hipStream_t rows_stream=nullptr;  // CUDA: the rows' copy to prompt_host runs beside the batch's GEMMs, not ahead of them (grouped_expert_prefill)
    uint32_t assist=0;
    bool hint_experts=true;   // no bank is page-locked (sd_hybrid_assist): hint a layer's absent experts (hybrid_host_experts)
    std::atomic<uint32_t> bank_readers{0};   // the CPU's workers reading banks' bytes now (BankReading): sd_hybrid_repoint waits for none
    float route_floor=0;      // speed first (sd_hybrid_route_floor): CPU routes below this weight are left out
    uint64_t dropped_experts=0;   // the routes it left out
    uint64_t hybrid_wait_ns=0, hybrid_cpu_ns=0;
    // The host's longest steps in the hybrid span under way (from its HybridAwait on), said when a GPU wait times out (recover_hybrid_wait):
    // a launch or a stream query (and which), a unit's CPU experts, a wait for an export; and the layer the host last took an export of.
    uint64_t span_call_ns=0, span_cpu_ns=0, span_export_ns=0;
    const char* span_call="none";
    size_t span_layer=SIZE_MAX;
    // The GPU's waits that passed their deadline and found the host's result there (hybrid_await_past_deadline): read from the exchange,
    // told in the log at most every ten seconds (HybridAwait's start), with when they were told last.
    uint32_t late_seen=0;
    uint64_t late_waits=0;
    std::chrono::steady_clock::time_point late_told{};
    uint64_t hybrid_layers=0, hybrid_timed_layers=0;   // layers decoded hybrid, and of them those with a start on the GPU clock (hybrid_clock)
    uint32_t hybrid_sequence=0;
    size_t hybrid_position=SIZE_MAX;
    uint64_t hybrid_verify_balanced=0;   // hybrid_tokens at a verification's last swaps
    // The routes that missed the slots (the CPU's and the staged ones) and the tokens at the last refresh of the slots (hybrid_refresh_due)
    uint64_t refresh_misses=0, refresh_tokens=0;
    size_t head_graph_extras=SIZE_MAX;   // the extra rows of the draft's head whose graph is cached (output_head_forget)
    hipEvent_t swapped=nullptr;   // the pending swaps' copies have landed
    std::unique_ptr<Background> copier;   // issues bounded swaps' copies (hybrid_rebalance)
    std::unique_ptr<Background> bank_loader;   // gathers and copies a layer-order pass's bank loaded ahead (load_prefill_bank)
    std::shared_ptr<struct Transfers> transfers;   // asynchronous uploads (transfer.hpp), from the first use
#if defined(__linux__) || defined(_WIN32)
    std::unique_ptr<ReadAhead> read_ahead;   // started by the first expert read-ahead hint
#endif
    // STRAYDOG_PREFILL_PHASES (development): events between a prompt batch's phases, summed by name (phase_profile.hpp), printed when the device goes
    bool phase_on = std::getenv("STRAYDOG_PREFILL_PHASES") != nullptr;
    bool moe_eager = !std::getenv("STRAYDOG_MOE_EAGER") || std::getenv("STRAYDOG_MOE_EAGER")[0] != '0';
    bool phase_mtp = false;               // the layer in hand is the MTP layer's (its phases are listed apart, mtp_ before their names)
    std::set<std::string> phase_names;
    hipEvent_t counts_event = nullptr;   // after a prompt batch's expert counts: the host's reads of them wait for it, not for the work queued after (eager_moe)
    bool phase_trace = phase_on && std::getenv("STRAYDOG_PREFILL_PHASES")[0] == '2';
    struct PhaseMark { const char* name; hipEvent_t event; bool start; };   // (start: the first after a break, its time from the mark before it not counted)
    bool phase_broken = false;
    std::vector<PhaseMark> phase_marks;
    std::vector<hipEvent_t> phase_pool;
    struct PhaseSum { double sum = 0, largest = 0; uint64_t count = 0; };
    std::map<std::string, PhaseSum> phase_totals;
    uint64_t phase_batches = 0, phase_layers = 0, phase_prestage_issued = 0, phase_prestage_used = 0, phase_streamed = 0, phase_bus_bytes = 0,
             phase_cpu_routes = 0, phase_missing_layers = 0;
    // The host's time in a prompt batch's MoE layers (STRAYDOG_PREFILL_PHASES): waiting for the expert counts, asking for the read-ahead
    // hints, gathering streamed experts into the staging mirror, and waiting at the end for the streams the layer queued (settle).
    uint64_t phase_host_counts_ns = 0, phase_host_hints_ns = 0, phase_host_gather_ns = 0, phase_host_settle_ns = 0, phase_host_layers = 0;
    size_t phase_first = size_t(-1);
    uint64_t phase_routes[4] = {0, 0, 0, 0};   // routes of the resident, prestaged, streamed experts and the CPU's
    explicit Device(int value) : ordinal(value) {}
    ~Device() {
        abort_flag_release(abort_flag);
        if (phase_on) phase_report(*this);
        cleanup(hipSetDevice(ordinal));
        if (host_prefetch.joinable()) host_prefetch.join();
        for (auto& ready : host_ready) cleanup(hipHostFree(ready.second));
        if (staging_prefetch.joinable()) staging_prefetch.join();
        for (auto& pair : staging_ready) for (float* ready : pair) if (ready) cleanup(hipHostFree(ready));
        copier.reset();
        bank_loader.reset();
        transfers.reset();   // waits for its copies before the allocations go
        rows_overlap.reset();
        if (stream) cleanup(hipStreamSynchronize(stream));
        if (helper_stream) cleanup(hipStreamSynchronize(helper_stream));
        if (kv_stream) cleanup(hipStreamSynchronize(kv_stream));
        if (side_stream) cleanup(hipStreamSynchronize(side_stream));
        if (stage_stream) cleanup(hipStreamSynchronize(stage_stream));
        if (send_stream) cleanup(hipStreamSynchronize(send_stream));
        if (sum_stream) cleanup(hipStreamSynchronize(sum_stream));
        for (auto& entry : graphs) {
            cleanup(hipGraphExecDestroy(entry.second.executable));
            cleanup(hipGraphDestroy(entry.second.graph));
        }
        decoder.clear(); resident.clear();
        for (const auto& chunk : slots.chunks) if (chunk.base) cleanup(hipFree(chunk.base));
        if (slots.scratch) cleanup(hipFree(slots.scratch));
        if (slots.padding) cleanup(hipFree(slots.padding));
        if (slots.untracked_kv) cleanup(hipFree(slots.untracked_kv));
        if (slots.base) cleanup(hipFree(slots.base));
        if (slots.host) cleanup(hipHostFree(slots.host));
        if (prestage.event) cleanup(hipEventDestroy(prestage.event));
        if (counts_event) cleanup(hipEventDestroy(counts_event));
        for (hipEvent_t& event : layer_bank_free) if (event) cleanup(hipEventDestroy(event));
        if (layer_bank_loaded) cleanup(hipEventDestroy(layer_bank_loaded));
        for (hipEvent_t& event : deferred_events) if (event) cleanup(hipEventDestroy(event));
        if (deferred_host) cleanup(hipHostFree(deferred_host));
        if (exchange) cleanup(hipHostFree(exchange));
        if (prompt_host && prompt_host_pinned) cleanup(hipHostFree(prompt_host));
        if (prompt_rows) cleanup(hipEventDestroy(prompt_rows));
        if (rows_stream) cleanup(hipStreamDestroy(rows_stream));
        if (readback && readback_pinned) cleanup(hipHostFree(readback));
        if (route_mailbox) cleanup(hipHostFree(route_mailbox));
        if (readback_done) cleanup(hipEventDestroy(readback_done));
        if (parts_ready) cleanup(hipEventDestroy(parts_ready));
        if (parts_read) cleanup(hipEventDestroy(parts_read));
        if (sum_queued) cleanup(hipEventDestroy(sum_queued));
        if (sum_copied) cleanup(hipEventDestroy(sum_copied));
        if (cpu_partial) cleanup(hipFree(cpu_partial));
        if (expert_q8_scratch) cleanup(hipFree(expert_q8_scratch));
        if (draft_extras) cleanup(hipFree(draft_extras));
        if (draft_extra_tokens) cleanup(hipFree(draft_extra_tokens));
        if (copy_stream) cleanup(hipStreamDestroy(copy_stream));
        if (swapped) cleanup(hipEventDestroy(swapped));
        for (hipblasHandle_t handle : {blas, blas_f32})
            if (handle && hipblasDestroy(handle) != HIPBLAS_STATUS_SUCCESS)
                std::fprintf(stderr, "Straydog BLAS cleanup failed\n");
        if (expanded) cleanup(hipFree(expanded));
        if (half_input) cleanup(hipFree(half_input));
        if (attention_scores) cleanup(hipFree(attention_scores));
        if (blas_workspace) cleanup(hipFree(blas_workspace));
        if (flags) cleanup(hipFree(flags));
        if (rope_rows) cleanup(hipFree(rope_rows));
        if (handoff) cleanup(hipEventDestroy(handoff));
        for (RowStaging* staging : {&mtp_rows, &prefill_rows, &ple_rows})
            for (int i = 0; i < 2; ++i) {
                if (staging->staged[i]) { cleanup(hipEventSynchronize(staging->staged[i])); cleanup(hipEventDestroy(staging->staged[i])); }
                if (staging->host[i]) cleanup(hipHostFree(staging->host[i]));
            }
        for (auto entry : allocations) {
            const auto host = host_states.find(entry.first);
            cleanup(host == host_states.end() ? hipFree(entry.first) : hipHostFree(host->second));
        }
        for (auto entry : checkpoints) cleanup(hipHostFree(entry.first));
        for (void* pointer : retained_host_new) cleanup(hipHostFree(pointer));
        for (auto& entry : exchanges) cleanup(hipHostFree(entry.second.host));
        if (input) cleanup(hipFree(input));
        if (output) cleanup(hipFree(output));
        if (matrices) cleanup(hipFree(matrices));
        if (io) cleanup(hipHostFree(io));
        if (link.direct) {
            if (link.inbox) cleanup(hipFree(link.inbox));
            if (link.flag) cleanup(hipFree(link.flag));
        } else if (link.host) cleanup(hipHostFree(link.host));
        if (link.counters) cleanup(hipFree(link.counters));
        if (link.peer) link.peer->link.peer=nullptr;
        for (auto event : io_halves) if (event) cleanup(hipEventDestroy(event));
        if (stream) cleanup(hipStreamDestroy(stream));
        if (helper_stream) cleanup(hipStreamDestroy(helper_stream));
        if (kv_stream) cleanup(hipStreamDestroy(kv_stream));
        if (side_stream) cleanup(hipStreamDestroy(side_stream));
        if (stage_stream) cleanup(hipStreamDestroy(stage_stream));
        if (send_stream) cleanup(hipStreamDestroy(send_stream));
        if (sum_stream) cleanup(hipStreamDestroy(sum_stream));
        for (auto event : {kv_released, kv_prefetched, side_fork, side_join, stage_fork, stage_join, send_fork, send_done}) if (event) cleanup(hipEventDestroy(event));
    }
};
// Held while the CPU's workers read the bytes of a layer's banks (the CPU experts' routes, a gather into the staging mirror): the engine moves a layer's
// banks to other memory and frees the old only after none is held (sd_hybrid_repoint).
struct BankReading {
    std::atomic<uint32_t>& count;
    explicit BankReading(Device& d) : count(d.bank_readers) { count.fetch_add(1); }
    ~BankReading() { count.fetch_sub(1); }
    BankReading(const BankReading&) = delete;
    BankReading& operator=(const BankReading&) = delete;
};

__device__ inline float half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// Entry `code` (below 16) of sixteen signed bytes held in four words (byte i of word w is entry 4w + i), picked in registers: a table
// indexed by a run-time code is built in local memory, and a kernel with such a decoder in it has a stack frame for it.
__device__ __forceinline__ int table16_s8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, unsigned code) {
    const uint32_t word = (code & 8) ? ((code & 4) ? w3 : w2) : ((code & 4) ? w1 : w0);
    return int(int8_t(uint8_t(word >> ((code & 3) * 8))));
}
__device__ inline int nonlinear(unsigned code) {
    // {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}
    return table16_s8(0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u, code);
}
// Both signs' logistic without overflow (the HyperConnection's gates; the experts' products).
inline __device__ float hc_sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// IQ1/IQ2/IQ3 lattice grids, one byte per value: the gguf reference tables, uploaded to
// every device at creation (sd_iq_tables). A grid row is 8 (IQ1, IQ2) or 4 (IQ3) values;
// IQ1 rows store value + 1 (-1, 0, 1 as 0, 1, 2).
#if defined(__HIPCC__)
#define SD_IQ_GRID_STORAGE __constant__
#else
// Warp lanes select unrelated codebook entries. CUDA constant loads serialize
// those addresses; device storage lets the read-only global/L1 path serve them.
// The symbols are still populated once by the existing cudaMemcpyToSymbol path.
#define SD_IQ_GRID_STORAGE __device__
#endif
SD_IQ_GRID_STORAGE SD_ALIGN(16) uint8_t iq_grid2xxs[256 * 8], iq_grid2xs[512 * 8], iq_grid2s[1024 * 8];
SD_IQ_GRID_STORAGE SD_ALIGN(16) uint8_t iq_grid3xxs[256 * 4], iq_grid3s[512 * 4], iq_grid1s[2048 * 8];
#undef SD_IQ_GRID_STORAGE
// The tables are device variables, and the units are not linked as one module: each unit that includes this header has a set of its own, which its
// kernels read, and fills it with an upload of its own (sd_create calls them all). SD_IQ_GRIDS_UPLOAD(name) defines a unit's: the six tables, in
// the order sd_create reads them (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_M) with their sizes.
#define SD_IQ_GRIDS_UPLOAD(name) \
    void name(const uint8_t* const* grids, const size_t* bytes) { \
        check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid2xxs), grids[0], bytes[0])); \
        check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid2xs), grids[1], bytes[1])); \
        check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid2s), grids[2], bytes[2])); \
        check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid3xxs), grids[3], bytes[3])); \
        check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid3s), grids[4], bytes[4])); \
        check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid1s), grids[5], bytes[5])); \
    }
void iq_grids_upload_gemv_a(const uint8_t* const* grids, const size_t* bytes);
void iq_grids_upload_gemv_b(const uint8_t* const* grids, const size_t* bytes);
void iq_grids_upload_moe(const uint8_t* const* grids, const size_t* bytes);
extern "C" const uint8_t* sd_iq_grid(int kind, size_t* bytes);
inline std::atomic<bool> iq_uploaded{false};
__host__ __device__ inline bool iq_kind(int kind) {
    return kind == 16 || kind == 17 || kind == 18 || kind == 19 || kind == 21 || kind == 22 || kind == 29;
}
// Bytes of one 256-value IQ super-block.
__host__ __device__ constexpr size_t iq_block_bytes(int kind) {
    return kind == 16 ? 66 : kind == 17 ? 74 : kind == 22 ? 82 : kind == 18 ? 98 : kind == 29 ? 56 : kind == 19 ? 50 : 110;
}
// Seven stored sign bits plus the eighth, which keeps the number of negatives even.
__device__ __forceinline__ unsigned iq_signs(unsigned bits) { return bits | ((__popc(bits) & 1u) << 7); }
__device__ __forceinline__ uint32_t iq_u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
// IQ1_M (56 bytes): qs[32], qh[16], then four sub-scale words whose top nibbles hold the
// FP16 scale. Group l of block ib: its scale, its delta and its grid row.
__device__ __forceinline__ void iq1m_group(const uint8_t* b, unsigned ib, unsigned l, float& dl, float& delta,
                                           unsigned& index) {
    const auto word = [b](unsigned k) { return unsigned(b[48 + 2 * k]) | unsigned(b[49 + 2 * k]) << 8; };
    const unsigned bits = (word(0) >> 12) | ((word(1) >> 8) & 0xf0) | ((word(2) >> 4) & 0xf00) | (word(3) & 0xf000);
    const uint8_t scale[2] = {uint8_t(bits), uint8_t(bits >> 8)};
    dl = half_at(scale) * float(2 * ((word(ib / 2) >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1);
    const unsigned high = b[32 + 2 * ib + l / 2] >> (4 * (l % 2));
    delta = high & 8 ? -0.125f : 0.125f;
    index = b[4 * ib + l] | ((high & 7) << 8);
}
// IQ1_S (50 bytes): d, qs[32], then per 32 values a 16-bit word of four 3-bit grid-row high
// parts, a 3-bit scale and the delta's sign. Group l of block ib: its scale, delta and grid row.
__device__ __forceinline__ void iq1s_group(const uint8_t* b, unsigned ib, unsigned l, float& dl, float& delta,
                                           unsigned& index) {
    const unsigned high = unsigned(b[34 + 2 * ib]) | unsigned(b[35 + 2 * ib]) << 8;
    dl = half_at(b) * float(2 * ((high >> 12) & 7) + 1);
    delta = high & 0x8000 ? -0.125f : 0.125f;
    index = b[2 + 4 * ib + l] | (((high >> (3 * l)) & 7) << 8);
}
// The eight values of group l (0-3) of 32-value block ib in an IQ super-block b,
// exactly as iq_weight decodes each of them.
template<int kind>
__device__ void iq_decode8(const uint8_t* b, unsigned ib, unsigned l, float (&v)[8]) {
    if (kind == 29 || kind == 19) {
        float dl, delta;
        unsigned index;
        if (kind == 29) iq1m_group(b, ib, l, dl, delta, index);
        else iq1s_group(b, ib, l, dl, delta, index);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = dl * (float(iq_grid1s[index * 8 + j]) - 1.0f + delta);
        return;
    }
    const float d = half_at(b);
    float db;
    unsigned signs;
    uint8_t g[8];
    if (kind == 16 || kind == 18) {
        const uint32_t aux = iq_u32(b + (kind == 16 ? 2 + 8 * ib + 4 : 66 + 4 * ib));
        db = d * (0.5f + float(aux >> 28)) * (kind == 16 ? 0.25f : 0.5f);
        signs = iq_signs((aux >> (7 * l)) & 127);
        if (kind == 16) {
            for (int j = 0; j < 8; ++j) g[j] = iq_grid2xxs[b[2 + 8 * ib + l] * 8 + j];
        } else {
            for (int j = 0; j < 8; ++j) g[j] = iq_grid3xxs[b[2 + 8 * ib + 2 * l + j / 4] * 4 + j % 4];
        }
    } else if (kind == 17) {
        const unsigned q = unsigned(b[2 + 2 * (4 * ib + l)]) | unsigned(b[3 + 2 * (4 * ib + l)]) << 8;
        db = d * (0.5f + float((b[66 + ib] >> (4 * (l / 2))) & 15)) * 0.25f;
        signs = iq_signs(q >> 9);
        for (int j = 0; j < 8; ++j) g[j] = iq_grid2xs[(q & 511) * 8 + j];
    } else if (kind == 22) {
        const unsigned index = b[2 + 4 * ib + l] | ((unsigned(b[66 + ib]) << (8 - 2 * l)) & 0x300);
        db = d * (0.5f + float((b[74 + ib] >> (4 * (l / 2))) & 15)) * 0.25f;
        signs = b[34 + 4 * ib + l];
        for (int j = 0; j < 8; ++j) g[j] = iq_grid2s[index * 8 + j];
    } else {
        db = d * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
        signs = b[74 + 4 * ib + l];
        for (int half = 0; half < 2; ++half) {
            const unsigned index = b[2 + 8 * ib + 2 * l + half] | ((unsigned(b[66 + ib]) << (8 - 2 * l - half)) & 256);
            for (int j = 0; j < 4; ++j) g[4 * half + j] = iq_grid3s[index * 4 + j];
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = db * float(g[j]) * ((signs >> j) & 1 ? -1.0f : 1.0f);
}
// One value of an IQ2_XXS (16), IQ2_XS (17), IQ2_S (22), IQ3_XXS (18), IQ3_S (21), IQ1_S (19)
// or IQ1_M (29) super-block of 256: scale * grid value * sign (IQ1: scale * (grid value
// + delta)), every product exact in F32.
template<int kind>
__device__ float iq_weight(const uint8_t* b, unsigned i) {
    const unsigned ib = i / 32, l = (i % 32) / 8, j = i % 8;
    if (kind == 29 || kind == 19) {
        float dl, delta;
        unsigned index;
        if (kind == 29) iq1m_group(b, ib, l, dl, delta, index);
        else iq1s_group(b, ib, l, dl, delta, index);
        return dl * (float(iq_grid1s[index * 8 + j]) - 1.0f + delta);
    }
    const float d = half_at(b);
    if (kind == 16) {
        const uint32_t aux = iq_u32(b + 2 + 8 * ib + 4);
        const float db = d * (0.5f + float(aux >> 28)) * 0.25f;
        const float g = float(iq_grid2xxs[b[2 + 8 * ib + l] * 8 + j]);
        return db * g * ((iq_signs((aux >> (7 * l)) & 127) >> j) & 1 ? -1.0f : 1.0f);
    }
    if (kind == 17) {
        const unsigned q = unsigned(b[2 + 2 * (4 * ib + l)]) | unsigned(b[3 + 2 * (4 * ib + l)]) << 8;
        const float db = d * (0.5f + float((b[66 + ib] >> (4 * (l / 2))) & 15)) * 0.25f;
        const float g = float(iq_grid2xs[(q & 511) * 8 + j]);
        return db * g * ((iq_signs(q >> 9) >> j) & 1 ? -1.0f : 1.0f);
    }
    if (kind == 22) {
        const unsigned index = b[2 + 4 * ib + l] | ((unsigned(b[66 + ib]) << (8 - 2 * l)) & 0x300);
        const float db = d * (0.5f + float((b[74 + ib] >> (4 * (l / 2))) & 15)) * 0.25f;
        const float g = float(iq_grid2s[index * 8 + j]);
        return db * g * ((b[34 + 4 * ib + l] >> j) & 1 ? -1.0f : 1.0f);
    }
    if (kind == 18) {
        const uint32_t aux = iq_u32(b + 66 + 4 * ib);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        const float g = float(iq_grid3xxs[b[2 + 8 * ib + 2 * l + j / 4] * 4 + j % 4]);
        return db * g * ((iq_signs((aux >> (7 * l)) & 127) >> j) & 1 ? -1.0f : 1.0f);
    }
    // IQ3_S
    const unsigned high = j / 4;
    const unsigned index = b[2 + 8 * ib + 2 * l + high] | ((unsigned(b[66 + ib]) << (8 - 2 * l - high)) & 256);
    const float db = d * float(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15));
    const float g = float(iq_grid3s[index * 4 + j % 4]);
    return db * g * ((b[74 + 4 * ib + l] >> j) & 1 ? -1.0f : 1.0f);
}

template<int kind>
__device__ float weight(const uint8_t* data, size_t col) {
    if (kind == 0) return reinterpret_cast<const float*>(data)[col];
    if (kind == 1) return half_at(data + col * 2);
    if (kind == 30) {
        const uint32_t bits = uint32_t(reinterpret_cast<const uint16_t*>(data)[col]) << 16;
        return __uint_as_float(bits);
    }
    if (kind == 42) {
        const uint8_t* b = data + (col / 64) * 18;
        const unsigned code = (b[2 + (col % 64) / 4] >> ((col % 4) * 2)) & 3;
        return half_at(b) * (int(code) - 1);
    }
    if (kind == 39) {   // MXFP4: doubled E2M1 values times 2^(e - 128) (subnormal below e = 2)
        const uint8_t* b = data + (col / 32) * 17;
        const unsigned i = unsigned(col % 32), e = b[0];
        // (values {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12})
        const int value = table16_s8(0x03020100u, 0x0c080604u, 0xfdfeff00u, 0xf4f8fafcu, (b[1 + i % 16] >> (4 * (i / 16))) & 15);
        return float(value) * __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
    }
    if (kind == 2 || kind == 6 || kind == 8 || kind == 20) {
        const size_t stride = kind == 6 ? 22 : kind == 8 ? 34 : 18;
        const uint8_t* b = data + (col / 32) * stride;
        const unsigned i = unsigned(col % 32);
        if (kind == 8) return half_at(b) * int(reinterpret_cast<const int8_t*>(b + 2)[i]);
        unsigned code = (b[(kind == 6 ? 6 : 2) + i % 16] >> (i / 16 * 4)) & 15;
        if (kind == 20) return half_at(b) * nonlinear(code);
        if (kind == 6) code |= ((b[2 + i / 8] >> (i % 8)) & 1) << 4;
        return half_at(b) * (int(code) - (kind == 6 ? 16 : 8));
    }
    if (iq_kind(kind)) return iq_weight<kind>(data + (col / 256) * iq_block_bytes(kind), unsigned(col % 256));
    const size_t stride = kind == 10 ? 84 : kind == 11 ? 110 : kind == 12 ? 144 : kind == 13 ? 176 : kind == 14 ? 210 : 136;
    const uint8_t* b = data + (col / 256) * stride;
    const unsigned i = unsigned(col % 256), group = i / 32;
    if (kind == 10) {   // Q2_K: scales[16] qs[64] d dmin; in each half, two bits of qs at shift 2 * quarter
        const unsigned half = i / 128, quarter = (i % 128) / 32, lane = i % 32;
        const unsigned scale = b[8 * half + 2 * quarter + lane / 16];
        const unsigned code = (b[16 + 32 * half + lane] >> (2 * quarter)) & 3;
        return half_at(b + 80) * float(scale & 15) * float(code) - half_at(b + 82) * float(scale >> 4);
    }
    if (kind == 11) {
        const unsigned scale_group = i / 16;
        const uint8_t* scales = b + 96;
        int scale = int(((scales[scale_group % 8] >> (scale_group / 8 * 4)) & 15)
                    | (((scales[8 + scale_group % 4] >> (scale_group / 4 * 2)) & 3) << 4)) - 32;
        int code = int((b[32 + (i / 128) * 32 + i % 32] >> ((i / 32 % 4) * 2)) & 3);
        code -= ((b[i % 32] >> group) & 1) ? 0 : 4;
        return half_at(b + 108) * scale * code;
    }
    if (kind == 12 || kind == 13) {
        const uint8_t* scales = b + 4;
        unsigned scale = group < 4 ? scales[group] & 63
            : (scales[8 + group - 4] & 15) | ((scales[group - 4] >> 2) & 48);
        unsigned minimum = group < 4 ? scales[4 + group] & 63
            : (scales[8 + group - 4] >> 4) | ((scales[group] >> 2) & 48);
        const uint8_t* qs = b + (kind == 13 ? 48 : 16);
        unsigned code = (qs[(group / 2) * 32 + i % 32] >> ((group % 2) * 4)) & 15;
        if (kind == 13) code |= ((b[16 + i % 32] >> group) & 1) << 4;
        return half_at(b) * scale * code - half_at(b + 2) * minimum;
    }
    if (kind == 14) {
        unsigned low = (b[(i / 128) * 64 + i % 64] >> ((i % 128 / 64) * 4)) & 15;
        unsigned high = (b[128 + (i / 128) * 32 + i % 32] >> ((i / 32 % 4) * 2)) & 3;
        int scale = reinterpret_cast<const int8_t*>(b + 192)[i / 16];
        return half_at(b + 208) * scale * (int(low | (high << 4)) - 32);
    }
    // IQ4_XS: eight signed six-bit scales and nonlinear four-bit values.
    unsigned high = unsigned(b[2]) | (unsigned(b[3]) << 8);
    int scale = int(((b[4 + group / 2] >> ((group % 2) * 4)) & 15)
                  | (((high >> (2 * group)) & 3) << 4)) - 32;
    unsigned code = (b[8 + group * 16 + i % 16] >> ((i % 32 / 16) * 4)) & 15;
    return half_at(b) * scale * nonlinear(code);
}


inline size_t matrix_row_bytes(int kind, uint64_t cols) {
    if (kind == 0) return size_t(cols) * 4;
    if (kind == 1 || kind == 30) return size_t(cols) * 2;
    if (kind == 42 && cols % 64 == 0) return size_t(cols / 64) * 18;
    if ((kind == 2 || kind == 6 || kind == 8 || kind == 20 || kind == 39) && cols % 32 == 0)
        return size_t(cols / 32) * (kind == 6 ? 22 : kind == 8 ? 34 : kind == 39 ? 17 : 18);
    if ((kind == 10 || kind == 11 || kind == 12 || kind == 13 || kind == 14 || kind == 23) && cols % 256 == 0)
        return size_t(cols / 256) * (kind == 10 ? 84 : kind == 11 ? 110 : kind == 12 ? 144 : kind == 13 ? 176 : kind == 14 ? 210 : 136);
    if (iq_kind(kind) && cols % 256 == 0) {
        if (!iq_uploaded) throw std::runtime_error("IQ grid tables were not provided before the device was created");
        return size_t(cols / 256) * iq_block_bytes(kind);
    }
    throw std::runtime_error("unsupported Straydog GPU matrix format");
}

inline Device& device(void* handle) {
    if (!handle) throw std::runtime_error("null Straydog device");
    auto& value = *static_cast<Device*>(handle);
    check(hipSetDevice(value.ordinal));
    return value;
}
// The host's side of words a device also reads or writes (a mapped flag): release stores, acquire
// loads and a spin's pause, by GCC/Clang builtins or MSVC's volatile accesses with fences; and
// size arithmetic that reports a wrap, by builtins or by hand.
inline void host_store_release(unsigned* word, unsigned value) {
#if defined(_MSC_VER)
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile unsigned*>(word) = value;
#else
    __atomic_store_n(word, value, __ATOMIC_RELEASE);
#endif
}
inline unsigned host_load_acquire(const unsigned* word) {
#if defined(_MSC_VER)
    const unsigned value = *static_cast<const volatile unsigned*>(word);
    std::atomic_thread_fence(std::memory_order_acquire);
    return value;
#else
    return __atomic_load_n(word, __ATOMIC_ACQUIRE);
#endif
}
inline void host_pause() {
#ifdef __x86_64__
    __builtin_ia32_pause();
#elif defined(_M_X64)
    _mm_pause();
#endif
}
inline bool mul_overflows(uint64_t a, uint64_t b, uint64_t* product) {
#if defined(_MSC_VER)
    if (b && a > UINT64_MAX / b) return true;
    *product = a * b;
    return false;
#else
    return __builtin_mul_overflow(a, b, product);
#endif
}
inline bool add_overflows(uint64_t a, uint64_t b, uint64_t* sum) {
#if defined(_MSC_VER)
    if (a > UINT64_MAX - b) return true;
    *sum = a + b;
    return false;
#else
    return __builtin_add_overflow(a, b, sum);
#endif
}

// An op's sizes from the caller's counts, checked: the product of counts (an error when it
// overflows, so that an op_span check never passes on a wrapped size), the bytes of that many
// floats, blocks of 256 threads over a count, and a launch dimension (at most `limit`).
inline size_t checked_product(std::initializer_list<uint64_t> factors) {
    uint64_t product = 1;
    for (const uint64_t factor : factors)
        if (mul_overflows(product, factor, &product)) throw std::runtime_error("a Straydog op's shape overflows");
    return size_t(product);
}
inline size_t checked_floats(std::initializer_list<uint64_t> factors) {
    return checked_product({uint64_t(checked_product(factors)), 4});
}
inline size_t checked_sum(std::initializer_list<uint64_t> terms) {
    uint64_t sum = 0;
    for (const uint64_t term : terms)
        if (add_overflows(sum, term, &sum)) throw std::runtime_error("a Straydog op's shape overflows");
    return size_t(sum);
}
constexpr size_t expert_cache_allocation_reserve=2*1024*1024;
inline size_t budget_available(const Device& d) {
    if (d.used>d.budget || d.retained_reserve>d.budget-d.used) return 0;
    return d.budget-d.used-d.retained_reserve;
}
inline void retained_reserve_add(Device& d,size_t bytes) {
    d.retained_reserve=checked_sum({uint64_t(d.retained_reserve),uint64_t(bytes)});
    d.retained_reserve_peak=std::max(d.retained_reserve_peak,d.retained_reserve);
}
inline bool retained_kv_charged(const Device& d,const void* pointer) {
    return std::find(d.retained_kv_allocations.begin(),d.retained_kv_allocations.end(),pointer)!=d.retained_kv_allocations.end();
}
// The allocation map is erased by the caller after this succeeds. On a failed
// GPU free, both requested bytes and the allocation's reserve remain owned.
inline void retained_tracked_free(Device& d,void* pointer,size_t total) {
    const auto charged=std::find(d.retained_kv_allocations.begin(),d.retained_kv_allocations.end(),pointer);
    const size_t reserve=charged==d.retained_kv_allocations.end() ? 0 : expert_cache_allocation_reserve;
    if (total>d.used || reserve>d.retained_reserve) throw std::runtime_error("invalid retained allocation accounting");
    check(hipFree(pointer));
    d.used-=total; d.retained_reserve-=reserve;
    if (charged!=d.retained_kv_allocations.end()) d.retained_kv_allocations.erase(charged);
}
inline unsigned checked_blocks(size_t count) {
    const size_t blocks = count / 256 + (count % 256 != 0);
    if (blocks > size_t(INT32_MAX)) throw std::runtime_error("a Straydog op's launch is too large");
    return unsigned(blocks);
}
inline unsigned checked_dim(uint64_t count, uint64_t limit = 65535) {
    if (!count || count > limit) throw std::runtime_error("a Straydog op's launch is out of range");
    return unsigned(count);
}
#if defined(__HIPCC__)
// RDNA4 vector types of the expert GEMM (moe_gemm.hpp, compiled in moe.cu) that the attention kernels (sparse_attention_compute.hpp,
// qsa_prompt_attention.hpp, compiled in kernels.cpp) use as well.
typedef _Float16 moe_half2 __attribute__((ext_vector_type(2)));
typedef _Float16 moe_half8 __attribute__((ext_vector_type(8)));
typedef float moe_float8 __attribute__((ext_vector_type(8)));
typedef uint32_t moe_word4 __attribute__((ext_vector_type(4)));
// Blocks of 110, 98, 82, 18 or 34 bytes start at any even address; RDNA4 loads them unaligned.
typedef uint32_t moe_word_u __attribute__((aligned(2)));
typedef uint2 moe_word2_u __attribute__((aligned(2)));
typedef moe_word4 moe_word4_u __attribute__((aligned(2)));
typedef moe_word4 moe_word4_b __attribute__((aligned(1)));   // MXFP4: 17-byte blocks at any byte
#endif
} // namespace sdgpu
