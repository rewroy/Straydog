// Straydog's independent, correctness-first GPU matrix/vector backend.
// HIP and CUDA are selected computational dependencies, not inference engines.
// Q2_0 format: FP16 scale + 16 packed bytes, 64 LSB-first two-bit values,
// decoded as scale * (code - 1).
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

namespace {
thread_local std::string last_error;
#if !defined(__HIPCC__)
std::mutex registered_host_allocations_mutex;
std::set<void*> registered_host_allocations;
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
#endif
// Fixed-size device slots holding one routed expert's gate, up and down slices.
struct ExpertSlots {
    uint8_t* base=nullptr;
    uint8_t* host=nullptr;   // page-locked mirror of the staging slots: prefill gathers absent experts here
    size_t cache=0, stage=0, bytes=0;   // slots, staging slots, a staging slot's bytes
    size_t chunk=0;                     // of the staging slots, those a prompt batch streams its experts through in chunks; the rest (prestage) hold a layer's experts ahead
    size_t arena=0;                     // the slots' bytes, all before the staging slots
    size_t soa_scratch=0;               // private tail of this same arena, never a cache or staging slot
    std::vector<size_t> at;             // a slot's offset from base: an expert takes its own layer's size
    std::vector<uint8_t> group;         // and only a slot of its size group
    std::vector<std::pair<ResidentMoe*,int32_t>> owner;
    std::vector<size_t> pending;   // decode swaps still loading on the copy stream
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
const char* gpu_error_text_of(hipError_t error, const char* runtime) {
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
const char* gpu_error_text(hipError_t error) { return gpu_error_text_of(error, hipGetErrorString(error)); }

// The site and the code go with the message: the same words can mean memory that ran out or a device that was reset.
void check_at(hipError_t error, const char* file, int line) {
    if (error == hipSuccess) return;
    const std::string path(file);
    throw std::runtime_error(std::string(gpu_error_text(error)) + " (" + path.substr(path.find_last_of("/\\") + 1) + ":" + std::to_string(line) +
                             ", code " + std::to_string(int(error)) + ")");
}
#define check(error) check_at((error), __FILE__, __LINE__)

void blas_check(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Straydog BLAS failure: " + std::to_string(int(status)));
}

void cleanup(hipError_t error) noexcept {
    if (error != hipSuccess) std::fprintf(stderr, "Straydog GPU cleanup: %s\n", gpu_error_text(error));
}

// Page-locked host memory for what a request asks (a K/V growth, the MTP rows' staging, a read's buffer). Tests make the next
// `refuse_host_allocations` of these answer as a machine short of memory does (sd_test_refuse_host).
std::atomic<uint64_t> refuse_host_allocations{0};
// Tests: the streamed chunk of a prompt batch's layer that fails, counted from the next one (0: none; sd_test_fail_streamed, hybrid_stream_missing).
std::atomic<uint64_t> fail_streamed_chunk{0};
hipError_t host_allocate(void** pointer, size_t bytes, unsigned int flags) {
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
hipblasHandle_t create_f32_blas() {
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
struct Device {
    int ordinal;
    size_t budget = 0, used = 0;
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
    std::unordered_map<uint64_t,bool> direct_fenced;
    // Partially resident decode launches each layer's graph before the previous layer's
    // CPU result exists; the graph waits for it on the GPU (hybrid_await).
    bool hybrid_prelaunch = false;
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
    // check accumulates in flags[1] until sd_prefill_settle. `handoff` orders a row handoff
    // between this device's stream and its peer's.
    bool prefill_deferred = false;
    ResidentMoe* layer_prefill_plan = nullptr;
    void* layer_prefill_storage = nullptr;
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
    bool op_recording = false;   // between sd_graph_begin and sd_graph_end
    uint8_t* readback = nullptr;
    bool readback_pinned = true;   // false: pageable (the system refused to page-lock it), held in readback_pageable
    std::vector<uint8_t> readback_pageable;
    hipEvent_t readback_done = nullptr;
    size_t readback_bytes = 0;
    // Allocations that live in mapped, page-locked host memory (sd_host_state): device pointer -> host pointer.
    std::unordered_map<void*, void*> host_states;
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
    size_t qwen_parts = 1;    // Qwen3.8's layers split across this many GPUs (DecoderLayer::parts)
    // A draft's vocabulary on the device holding the head (sd_draft_vocabulary): the head's first `draft_rows` rows, and
    // `draft_extra_count` rows of tokens past them, copied into `draft_extras` (their tokens in `draft_extra_tokens`).
    // draft_rows 0: drafts read the whole head.
    size_t draft_rows = 0, draft_extra_count = 0, draft_extra_capacity = 0, draft_extra_row_bytes = 0;
    uint8_t* draft_extras = nullptr;
    int32_t* draft_extra_tokens = nullptr;
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
    uint64_t hybrid_layers=0, hybrid_timed_layers=0;   // layers decoded hybrid, and of them those with a start on the GPU clock (hybrid_clock)
    uint32_t hybrid_sequence=0;
    size_t hybrid_position=SIZE_MAX;
    uint64_t hybrid_verify_balanced=0;   // hybrid_tokens at a verification's last swaps
    hipEvent_t swapped=nullptr;   // the pending swaps' copies have landed
    std::unique_ptr<Background> copier;   // issues bounded swaps' copies (hybrid_rebalance)
    std::shared_ptr<struct Transfers> transfers;   // asynchronous uploads (transfer.hpp), from the first use
#if defined(__linux__)
    std::unique_ptr<ReadAhead> read_ahead;   // started by the first expert read-ahead hint
#endif
    // STRAYDOG_PREFILL_PHASES (development): events between a prompt batch's phases, summed by name (phase_profile.hpp), printed when the device goes
    bool phase_on = std::getenv("STRAYDOG_PREFILL_PHASES") != nullptr;
    bool moe_eager = !std::getenv("STRAYDOG_MOE_EAGER") || std::getenv("STRAYDOG_MOE_EAGER")[0] != '0';
    bool phase_mtp = false;               // the layer in hand is the MTP layer's (its phases are listed apart, mtp_ before their names)
    std::set<std::string> phase_names;
    hipEvent_t counts_event = nullptr;   // after a prompt batch's expert counts: the host's reads of them wait for it, not for the work queued after (eager_moe)
    bool phase_trace = phase_on && std::getenv("STRAYDOG_PREFILL_PHASES")[0] == '2';
    struct PhaseMark { const char* name; hipEvent_t event; };
    std::vector<PhaseMark> phase_marks;
    std::vector<hipEvent_t> phase_pool;
    struct PhaseSum { double sum = 0, largest = 0; uint64_t count = 0; };
    std::map<std::string, PhaseSum> phase_totals;
    uint64_t phase_batches = 0, phase_layers = 0, phase_prestage_issued = 0, phase_prestage_used = 0, phase_streamed = 0, phase_bus_bytes = 0,
             phase_cpu_routes = 0, phase_missing_layers = 0;
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
        transfers.reset();   // waits for its copies before the allocations go
        if (stream) cleanup(hipStreamSynchronize(stream));
        if (helper_stream) cleanup(hipStreamSynchronize(helper_stream));
        if (kv_stream) cleanup(hipStreamSynchronize(kv_stream));
        if (side_stream) cleanup(hipStreamSynchronize(side_stream));
        if (stage_stream) cleanup(hipStreamSynchronize(stage_stream));
        if (send_stream) cleanup(hipStreamSynchronize(send_stream));
        for (auto& entry : graphs) {
            cleanup(hipGraphExecDestroy(entry.second.executable));
            cleanup(hipGraphDestroy(entry.second.graph));
        }
        decoder.clear(); resident.clear();
        if (slots.base) cleanup(hipFree(slots.base));
        if (slots.host) cleanup(hipHostFree(slots.host));
        if (prestage.event) cleanup(hipEventDestroy(prestage.event));
        if (counts_event) cleanup(hipEventDestroy(counts_event));
        if (exchange) cleanup(hipHostFree(exchange));
        if (prompt_host && prompt_host_pinned) cleanup(hipHostFree(prompt_host));
        if (prompt_rows) cleanup(hipEventDestroy(prompt_rows));
        if (rows_stream) cleanup(hipStreamDestroy(rows_stream));
        if (readback && readback_pinned) cleanup(hipHostFree(readback));
        if (route_mailbox) cleanup(hipHostFree(route_mailbox));
        if (readback_done) cleanup(hipEventDestroy(readback_done));
        if (parts_ready) cleanup(hipEventDestroy(parts_ready));
        if (parts_read) cleanup(hipEventDestroy(parts_read));
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

__device__ float half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ int nonlinear(unsigned code) {
    const int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    return values[code];
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
extern "C" const uint8_t* sd_iq_grid(int kind, size_t* bytes);
std::atomic<bool> iq_uploaded{false};
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
        const int values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        return float(values[(b[1 + i % 16] >> (4 * (i / 16))) & 15]) * __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
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

template<int kind, typename Output>
__global__ void expand_matrix(const uint8_t* data, size_t row_bytes, size_t rows, size_t cols, Output* output) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * cols) output[index] = weight<kind>(data + (index / cols) * row_bytes, index % cols);
}

void expand(const Matrix& item, float* output, hipStream_t stream, bool half) {
    const unsigned blocks = unsigned((item.rows * item.cols + 255) / 256);
    #define SD_EXPAND(format) case format: \
        if (half) expand_matrix<format,__half><<<blocks,256,0,stream>>>(item.data,item.row_bytes,item.rows,item.cols,reinterpret_cast<__half*>(output)); \
        else expand_matrix<format,float><<<blocks,256,0,stream>>>(item.data,item.row_bytes,item.rows,item.cols,output); break;
    switch (item.kind) {
        SD_FORMATS(SD_EXPAND)
        default: throw std::runtime_error("unsupported Straydog GEMM format");
    }
    #undef SD_EXPAND
    check(hipGetLastError());
}

__global__ void cast_half(const float* input, __half* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half(input[index]);
}

// A K/V row's values as they are read: FP16, or Q8_0's int8 times its block's scale. `kv_row`: the row's bytes, its format.
__device__ __forceinline__ float kv_scale(const kv_t* row, size_t column) {
    return __half2float(reinterpret_cast<const __half*>(row + 256)[column / 32]);
}
__device__ __forceinline__ float kv_load(const kv_t* row, size_t column, unsigned kv_row) {
    if (kv_row == kv_row_q8) return float(reinterpret_cast<const int8_t*>(row)[column]) * kv_scale(row, column);
    return __half2float(reinterpret_cast<const __half*>(row)[column]);
}
// Four values from `column` (a multiple of four, so in one block) as FP16, eight bytes as the WMMA prefill stages them.
__device__ __forceinline__ uint2 kv_half4(const kv_t* row, size_t column, unsigned kv_row) {
    if (kv_row != kv_row_q8) return *reinterpret_cast<const uint2*>(row + column * 2);
    const uint32_t word = *reinterpret_cast<const uint32_t*>(row + column);
    const float scale = kv_scale(row, column);
    uint32_t halves[4];
    #pragma unroll
    for (unsigned b = 0; b < 4; ++b) halves[b] = __half_as_ushort(__float2half_rn(float(int8_t(word >> (8 * b))) * scale));
    return uint2{halves[0] | halves[1] << 16, halves[2] | halves[3] << 16};
}
// A value to store. Q8_0 takes its block's scale from the largest magnitude of the block's 32 values, which the 32 lanes of a warp
// hold (lane = column % 32): every lane of the warp encodes together. The scale is rounded to FP16 first and the value divided by
// that, so what is read back is what the int8 meant (its error at most half a scale).
struct KvValue { float value; __half scale; int q; };
__device__ __forceinline__ KvValue kv_encode(float value, unsigned kv_row) {
    KvValue out{value, __float2half_rn(0.0f), 0};
    if (kv_row != kv_row_q8) return out;
    float top = fabsf(value);
    for (unsigned offset = 16; offset; offset /= 2) top = fmaxf(top, __shfl_xor(top, offset, 32));
    out.scale = __float2half_rn(top / 127.0f);
    const float scale = __half2float(out.scale);
    const int q = scale > 0.0f ? __float2int_rn(value / scale) : 0;
    out.q = q > 127 ? 127 : q < -127 ? -127 : q;
    return out;
}
__device__ __forceinline__ void kv_put(kv_t* row, size_t column, const KvValue& value, unsigned kv_row) {
    if (kv_row != kv_row_q8) { reinterpret_cast<__half*>(row)[column] = __float2half_rn(value.value); return; }
    reinterpret_cast<int8_t*>(row)[column] = int8_t(value.q);
    if (column % 32 == 0) reinterpret_cast<__half*>(row + 256)[column / 32] = value.scale;
}

// values (optional) holds the rows' values instead of the packed input. A layer's K/V is
// 2 * QwenHeads<P>::kv planes of `context` rows: each K/V head's keys, then each one's values.
// (a warp's 32 threads are one block of a row: whole warps return, `steps * width` being a multiple of 256)
template<int P=1>
__global__ void write_attention_kv(unsigned kv_row, kv_t* state, size_t context, size_t base, size_t steps, const float* input,
                                   const uint32_t* position=nullptr, const float* values=nullptr,
                                   kv_t* mirror=nullptr, size_t mirror_rows=0) { SD_KERNEL_ACQUIRE;
    constexpr size_t K = QwenHeads<P>::kv, width = K * 256;
    if (position) base=*position;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= steps * width) return;
    const size_t step = index / width, head = (index % width) / 256, column = index % 256;
    const size_t keys_start = steps * QwenHeads<P>::query * 256;
    const KvValue key = kv_encode(input[keys_start + index], kv_row);
    const KvValue value = kv_encode(values ? values[index] : input[keys_start + steps * width + index], kv_row);
    kv_put(state + (head * context + base + step) * kv_row, column, key, kv_row);
    kv_put(state + ((K + head) * context + base + step) * kv_row, column, value, kv_row);
    if (mirror_rows) {
        const size_t slot=unsigned(base+step)%unsigned(mirror_rows);
        kv_put(mirror+(head*mirror_rows+slot)*kv_row,column,key,kv_row);
        kv_put(mirror+((K+head)*mirror_rows+slot)*kv_row,column,value,kv_row);
    }
}

__global__ void attention_scores(unsigned kv_row, const kv_t* state, size_t context, size_t base, size_t steps,
                                 size_t selected, const float* input, float* scores, const uint32_t* position=nullptr) {
    if (position) base=*position;
    const size_t item = blockIdx.x, step = item / 24, head = item % 24;
    const size_t index_start = steps * (24 * 256 + 1024);
    const size_t count = size_t(input[index_start + steps * selected + step]);
    for (size_t slot = threadIdx.x; slot < selected; slot += blockDim.x) {
        float result = -INFINITY;
        if (slot < count) {
            const float raw = input[index_start + step * selected + slot];
            if (raw >= 0 && raw <= float(base + step) && raw < float(context)) {
                const size_t token = size_t(raw);
                result = 0;
                for (size_t column = 0; column < 256; ++column)
                    result += kv_load(state + ((head / 12) * context + token) * kv_row, column, kv_row)
                            * input[item * 256 + column];
                result *= 0.0625f;
            }
        }
        scores[item * selected + slot] = result;
    }
}

__global__ void attention_reduce(unsigned kv_row, const kv_t* state, size_t context, size_t base, size_t steps,
                                 size_t selected, const float* input, const float* scores, float* output, const uint32_t* position=nullptr) {
    if (position) base=*position;
    const size_t item = blockIdx.x, step = item / 24, head = item % 24, column = threadIdx.x;
    const size_t index_start = steps * (24 * 256 + 1024);
    const size_t count = size_t(input[index_start + steps * selected + step]);
    __shared__ float partial[256];
    __shared__ float probabilities[2052];
    float maximum = -INFINITY;
    for (size_t slot = column; slot < selected; slot += 256) maximum = fmaxf(maximum, scores[item * selected + slot]);
    partial[column] = maximum; __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (column < stride) partial[column] = fmaxf(partial[column], partial[column + stride]);
        __syncthreads();
    }
    maximum = partial[0];
    __syncthreads();
    float denominator = 0;
    for (size_t slot = column; slot < selected; slot += 256) {
        const float probability = expf(scores[item * selected + slot] - maximum);
        probabilities[slot] = probability;
        denominator += probability;
    }
    partial[column] = denominator; __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (column < stride) partial[column] += partial[column + stride];
        __syncthreads();
    }
    denominator = partial[0];
    __syncthreads();
    for (size_t slot = column; slot < selected; slot += 256) probabilities[slot] /= denominator;
    __syncthreads();
    float result = 0;
    for (size_t slot = 0; slot < count && slot < selected; ++slot) {
        const float raw = input[index_start + step * selected + slot];
        if (raw >= 0 && raw <= float(base + step) && raw < float(context)) {
            const size_t token = size_t(raw);
            result += probabilities[slot]
                    * kv_load(state + (2 * context + (head / 12) * context + token) * kv_row, column, kv_row);
        }
    }
    output[item * 256 + column] = result;
}

__global__ void delta_sequence(float* state, size_t heads, size_t dimension, size_t steps,
                               const float* packed, float* output) {
    const size_t head = blockIdx.x, column = threadIdx.x;
    if (head >= heads || column >= dimension) return;
    const size_t width = heads * dimension, elements = steps * width;
    for (size_t step = 0; step < steps; ++step) {
        const size_t offset = step * width + head * dimension;
        const float factor = packed[3 * elements + step * heads + head];
        const float beta = packed[3 * elements + steps * heads + step * heads + head];
        float prediction = 0;
        for (size_t key = 0; key < dimension; ++key)
            prediction += (state[(head * dimension + key) * dimension + column] * factor)
                        * packed[elements + offset + key];
        const float correction = (packed[2 * elements + offset + column] - prediction) * beta;
        float result = 0;
        for (size_t key = 0; key < dimension; ++key) {
            const size_t index = (head * dimension + key) * dimension + column;
            const float updated = state[index] * factor + packed[elements + offset + key] * correction;
            state[index] = updated;
            result += updated * packed[offset + key];
        }
        output[offset + column] = result;
    }
}

__global__ void delta_register_sequence(float* state, size_t heads, size_t steps,
                                         const float* packed, float* output) {
    const size_t head = blockIdx.x, col = threadIdx.x, elements = steps * heads * 128;
    float memory[128];
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) memory[key] = state[(head * 128 + key) * 128 + col];
    __shared__ float query[128], keys[128];
    for (size_t step = 0; step < steps; ++step) {
        const size_t offset = (step * heads + head) * 128;
        query[col] = packed[offset + col]; keys[col] = packed[elements + offset + col];
        __syncthreads();
        const float factor = packed[3 * elements + step * heads + head];
        const float beta = packed[3 * elements + steps * heads + step * heads + head];
        float prediction = 0;
        #pragma unroll
        for (unsigned key = 0; key < 128; ++key) prediction += (memory[key] * factor) * keys[key];
        const float correction = (packed[2 * elements + offset + col] - prediction) * beta;
        float result = 0;
        #pragma unroll
        for (unsigned key = 0; key < 128; ++key) {
            memory[key] = memory[key] * factor + keys[key] * correction;
            result += memory[key] * query[key];
        }
        output[offset + col] = result;
        __syncthreads();
    }
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) state[(head * 128 + key) * 128 + col] = memory[key];
}

#if !defined(__HIPCC__)
// delta_register_sequence with the next token's inputs fetched a token ahead (three buffers of the shared inputs, one barrier a token, where the
// scan above has two and waits on its global loads each token) and the next token's prediction formed in the loop that updates the state,
// beside the result's sum, the two sums being independent chains. The same expressions on the same values in the same orders: its outputs and
// state are the scan's bit for bit (checked on random inputs of 4096 tokens: 25 M outputs, none differing), 1.5x faster on an RTX 4090
// (4096 tokens, 48 heads: 6.17 -> 4.09 ms).
__global__ void delta_register_sequence_ahead(float* state, size_t heads, size_t steps, const float* packed, float* output) {
    const size_t head = blockIdx.x, col = threadIdx.x, elements = steps * heads * 128;
    float memory[128];
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) memory[key] = state[(head * 128 + key) * 128 + col];
    __shared__ float query[3][128], keys[3][128];
    float value, factor, beta;
    {
        const size_t offset = head * 128;
        query[0][col] = packed[offset + col]; keys[0][col] = packed[elements + offset + col];
        value = packed[2 * elements + offset + col];
        factor = packed[3 * elements + head];
        beta = packed[3 * elements + steps * heads + head];
    }
    __syncthreads();
    float prediction = 0;
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) prediction += (memory[key] * factor) * keys[0][key];
    for (size_t step = 0; step < steps; ++step) {
        const unsigned now = step % 3, next = (step + 1) % 3;
        const bool more = step + 1 < steps;
        float next_value = 0, next_factor = 0, next_beta = 0;
        if (more) {
            const size_t offset = ((step + 1) * heads + head) * 128;
            query[next][col] = packed[offset + col]; keys[next][col] = packed[elements + offset + col];
            next_value = packed[2 * elements + offset + col];
            next_factor = packed[3 * elements + (step + 1) * heads + head];
            next_beta = packed[3 * elements + steps * heads + (step + 1) * heads + head];
        }
        const float correction = (value - prediction) * beta;
        __syncthreads();
        float result = 0, following = 0;
        #pragma unroll
        for (unsigned key = 0; key < 128; ++key) {
            memory[key] = memory[key] * factor + keys[now][key] * correction;
            result += memory[key] * query[now][key];
            following += (memory[key] * next_factor) * keys[next][key];
        }
        output[(step * heads + head) * 128 + col] = result;
        prediction = following; value = next_value; factor = next_factor; beta = next_beta;
    }
    #pragma unroll
    for (unsigned key = 0; key < 128; ++key) state[(head * 128 + key) * 128 + col] = memory[key];
}
#endif

// The delta rule's scan over a chunk's tokens, each column of a head's state split across eight lanes (16 of its 128 keys each).
// The columns are independent, so a block takes 32 of them and a head spreads over four blocks (24 or 48 heads: 96 or 192 blocks,
// where one block a head kept 24 or 48 CUs busy); each block stages 16 tokens' q, k, v, decay and beta in LDS while it loads the
// next 16 (one barrier per 16 tokens, not two per token). The sums are explicit FMAs in the order the one-block-a-head scan
// compiled to: its outputs and state bit for bit, 1.9x faster on an R9700 (24 heads, 512 tokens: 0.446 -> 0.236 ms).
constexpr unsigned delta_columns = 32, delta_staged = 16;
template<int P, unsigned COLS, unsigned T, unsigned QK, unsigned V>
__device__ __forceinline__ void delta_stage_load(const float* packed, size_t steps, size_t elements, size_t heads, size_t head,
                                                 size_t first_col, size_t base, float (&q)[QK], float (&k)[QK], float (&v)[V],
                                                 float& decay, float& beta) {
    constexpr unsigned threads = COLS * 8;
    #pragma unroll
    for (unsigned j = 0; j < QK; ++j) {
        const unsigned e = threadIdx.x + j * threads, t = e / 128;
        if (e < T * 128 && base + t < steps) {
            const size_t at = ((base + t) * heads + head) * 128 + e % 128;
            q[j] = packed[at]; k[j] = packed[elements + at];
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < V; ++j) {
        const unsigned e = threadIdx.x + j * threads, t = e / COLS;
        if (e < T * COLS && base + t < steps) v[j] = packed[2 * elements + ((base + t) * heads + head) * 128 + first_col + e % COLS];
    }
    if (threadIdx.x < T && base + threadIdx.x < steps) {
        decay = packed[3 * elements + (base + threadIdx.x) * heads + head];
        beta = packed[3 * elements + steps * heads + (base + threadIdx.x) * heads + head];
    }
}
template<int P = 1>
__global__ void __launch_bounds__(delta_columns * 8) delta_split_sequence(float* state, size_t steps, const float* packed, float* output) {
    constexpr size_t heads = 48 / P, groups = 128 / delta_columns;
    constexpr unsigned COLS = delta_columns, T = delta_staged, threads = COLS * 8;
    constexpr unsigned QK = (T * 128 + threads - 1) / threads, V = (T * COLS + threads - 1) / threads;
    const size_t head = blockIdx.x / groups, first_col = (blockIdx.x % groups) * COLS, col = first_col + threadIdx.x / 8;
    const size_t split = threadIdx.x % 8, elements = steps * heads * 128;
    float memory[16];
    #pragma unroll
    for (size_t i = 0; i < 16; ++i) memory[i] = state[(head * 128 + split + i * 8) * 128 + col];
    __shared__ float sq[2][T][128], sk[2][T][128], sv[2][T][COLS], sd[2][T], sb[2][T];
    float q[QK], k[QK], v[V], decay = 0, beta = 0;
    auto stage = [&](unsigned b) {
        #pragma unroll
        for (unsigned j = 0; j < QK; ++j) {
            const unsigned e = threadIdx.x + j * threads;
            if (e < T * 128) { sq[b][e / 128][e % 128] = q[j]; sk[b][e / 128][e % 128] = k[j]; }
        }
        #pragma unroll
        for (unsigned j = 0; j < V; ++j) {
            const unsigned e = threadIdx.x + j * threads;
            if (e < T * COLS) sv[b][e / COLS][e % COLS] = v[j];
        }
        if (threadIdx.x < T) { sd[b][threadIdx.x] = decay; sb[b][threadIdx.x] = beta; }
    };
    delta_stage_load<P, COLS, T>(packed, steps, elements, heads, head, first_col, 0, q, k, v, decay, beta);
    stage(0);
    __syncthreads();
    for (size_t base = 0, round = 0; base < steps; base += T, ++round) {
        const unsigned b = round & 1;
        if (base + T < steps) delta_stage_load<P, COLS, T>(packed, steps, elements, heads, head, first_col, base + T, q, k, v, decay, beta);
        const size_t count = steps - base < T ? steps - base : T;
        for (size_t t = 0; t < count; ++t) {
            const float factor = sd[b][t], scale = sb[b][t];
            float decayed[16], prediction = 0;
            #pragma unroll
            for (size_t i = 0; i < 16; ++i) { decayed[i] = memory[i] * factor; prediction = fmaf(decayed[i], sk[b][t][split + i * 8], prediction); }
            for (unsigned lanes = 1; lanes < 8; lanes *= 2) prediction += __shfl_xor(prediction, lanes, 8);
            const float correction = (sv[b][t][threadIdx.x / 8] - prediction) * scale;
            float result = 0;
            #pragma unroll
            for (size_t i = 0; i < 16; ++i) {
                memory[i] = fmaf(sk[b][t][split + i * 8], correction, decayed[i]);
                result = fmaf(memory[i], sq[b][t][split + i * 8], result);
            }
            for (unsigned lanes = 1; lanes < 8; lanes *= 2) result += __shfl_xor(result, lanes, 8);
            if (split == 0) output[((base + t) * heads + head) * 128 + col] = result;
        }
        if (base + T < steps) stage(b ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (size_t i = 0; i < 16; ++i) state[(head * 128 + split + i * 8) * 128 + col] = memory[i];
}
// delta_split_sequence's grid: four blocks a head.
template<int P = 1>
inline unsigned delta_split_blocks() { return unsigned((48 / P) * (128 / delta_columns)); }

size_t matrix_row_bytes(int kind, uint64_t cols) {
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

Device& device(void* handle) {
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
inline unsigned checked_blocks(size_t count) {
    const size_t blocks = count / 256 + (count % 256 != 0);
    if (blocks > size_t(INT32_MAX)) throw std::runtime_error("a Straydog op's launch is too large");
    return unsigned(blocks);
}
inline unsigned checked_dim(uint64_t count, uint64_t limit = 65535) {
    if (!count || count > limit) throw std::runtime_error("a Straydog op's launch is out of range");
    return unsigned(count);
}

#include "phase_profile.hpp"
#include "host_io.hpp"
#include "peer.hpp"
#include "link_tail.hpp"
#include "gemv.hpp"
#include "expert_soa.hpp"
#include "resident_soa.hpp"
#include "graph_cache.hpp"
#include "hc_decode.hpp"
#include "hyperconnection.hpp"
#include "hc_fused.hpp"
#include "gated_delta.hpp"
#include "gated_delta_chunk.hpp"
#include "ple.hpp"
#include "grouped_gemm.hpp"
#include "mixture_of_experts.hpp"
#include "moe_gemm.hpp"
#if !defined(__HIPCC__)
#include "moe_gemm_cuda.hpp"
#endif
#include "sparse_attention_compute.hpp"
#include "qsa_prompt_attention.hpp"
#include "kv_stage.hpp"
#include "sparse_attention.hpp"
#include "hybrid_slots.hpp"
#include "resident_moe.hpp"
#include "experts.hpp"
#include "hybrid_moe.hpp"
#include "qsa_decode.hpp"
#include "prefill_batch.hpp"
#include "decoder_span.hpp"
#include "verify_span.hpp"
#include "ops.hpp"
#include "ds_attention_wmma.hpp"
#include "transfer.hpp"

// HybridAwait has already published every launched graph's missing CPU result.
// A successful drain makes the interrupted pass disposable and aligns the next export.
int recover_hybrid_wait(void* handle) {
    try {
        auto& d = device(handle);
        check(hipStreamSynchronize(d.stream));
        hybrid_settle(d, true);
        check(hipStreamSynchronize(d.stream));
        for (hipStream_t stream : {d.copy_stream, d.kv_stream, d.helper_stream, d.side_stream, d.send_stream, d.stage_stream, d.rows_stream})
            if (stream) check(hipStreamSynchronize(stream));
        if (d.transfers)
            for (hipStream_t stream : d.transfers->streams)
                if (stream) check(hipStreamSynchronize(stream));
        auto* counters = reinterpret_cast<volatile uint32_t*>(d.exchange + hybrid_sequence_slot);
        std::atomic_thread_fence(std::memory_order_acquire);
#if !defined(__HIPCC__)
        // CUDA's awaits live in device memory and restart at zero for each span.
        // Reset them only after every use of the interrupted pass has drained.
        check(hipMemsetAsync(hybrid_awaited(d), 0, sizeof(uint32_t), d.stream));
#endif
        check(hipMemsetAsync(d.flags + hybrid_await_device_failed, 0, sizeof(uint32_t), d.stream));
        check(hipStreamSynchronize(d.stream));
#if defined(__HIPCC__)
        counters[hybrid_published_slot] = counters[hybrid_awaited_slot];
#else
        counters[hybrid_published_slot] = 0;
#endif
        d.hybrid_sequence = counters[0];
        counters[hybrid_await_failed_slot] = 0;
        std::atomic_thread_fence(std::memory_order_release);
        last_error = "the GPU wait timed out; the interrupted pass was discarded";
    } catch (const std::exception& error) {
        last_error = error.what();
    }
    return -1;
}

// The attention rope table: forward[row, pair], then the inverse at the same stride, one allocation
// for the pass. Publishing consumes an arm from sd_attention_turns_arm.
struct AttentionTurnsView {
    double2* forward = nullptr;
    double2* inverse = nullptr;
    bool publish = false;
};

void attention_turns_free(Device& value) {
    if (!value.attention_turns) return;
    auto found = value.allocations.find(value.attention_turns);
    if (found == value.allocations.end()) throw std::runtime_error("the attention rope table is not an allocation");
    if (value.stream) check(hipStreamSynchronize(value.stream));
    check(hipFree(value.attention_turns));
    value.used -= found->second + allocation_tail;
    value.allocations.erase(found);
    value.attention_turns = nullptr;
    value.attention_turns_bytes = 0;
    value.attention_turns_rows = 0;
    value.attention_turns_pairs = 0;
    value.attention_turns_armed = 0;
    value.attention_turns_ready = false;
}

void attention_turns_reserve(Device& value, size_t rows, size_t pairs) {
    if (!rows || !pairs) return;
    if (pairs > 64) throw std::runtime_error("invalid Straydog attention rope");
    if (value.attention_turns && value.attention_turns_pairs == pairs && value.attention_turns_rows >= rows) return;
    if (value.op_recording) throw std::runtime_error("an attention rope table cannot be allocated while a graph is recorded");
    if (value.attention_turns && !value.graphs.empty())
        throw std::runtime_error("an attention rope table cannot grow after a graph was recorded");
    if (value.attention_turns) attention_turns_free(value);
    const size_t bytes = rows * pairs * sizeof(double2) * 2;
    value.attention_turns = static_cast<double2*>(allocate(value, bytes));
    value.attention_turns_bytes = bytes;
    value.attention_turns_rows = rows;
    value.attention_turns_pairs = pairs;
    value.attention_turns_armed = 0;
    value.attention_turns_ready = false;
}

AttentionTurnsView attention_turns_for_rope(Device& value, size_t rows, size_t pairs) {
    if (!value.attention_turns_armed) return {};
    const size_t armed = value.attention_turns_armed;
    value.attention_turns_armed = 0;
    if (armed != rows || !pairs || pairs != value.attention_turns_pairs || !value.attention_turns || value.attention_turns_rows < rows)
        return {};
    AttentionTurnsView view;
    view.forward = value.attention_turns;
    view.inverse = view.forward + value.attention_turns_rows * value.attention_turns_pairs;
    view.publish = true;
    return view;
}

void attention_turns_mark_ready(Device& value, size_t rows, size_t pairs) {
    value.attention_turns_ready = true;
    value.attention_turns_ready_rows = rows;
    value.attention_turns_ready_pairs = pairs;
}
} // namespace

// A function that throws keeps C++ linkage: outside this block, or in extern "C++" within it.
// MSVC (/EHsc) takes a C function for one that never throws, and drops the catch around its call.
extern "C" {
int sd_ple(void* handle,void* const* pointers,const int* kinds,void* history,uint64_t steps,
            const float* embeddings,const float* hidden,float* output) {
    try { ple(device(handle),pointers,kinds,history,size_t(steps),embeddings,hidden,output); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_decoder_layer(void* handle,uint64_t layer,uint64_t context,void* const* pointers,const int* kinds,
                      uint64_t count,void* const* states,void* workspace,void* moe) {
    try {
        register_decoder_layer(device(handle),size_t(layer),size_t(context),pointers,kinds,size_t(count),states,workspace,moe);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_decode_span(void* handle,uint64_t begin,uint64_t end,uint64_t position,const float* input,float* output) {
    try { decode_span(device(handle),size_t(begin),size_t(end),size_t(position),input,output); return 0; }
    catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Makes the stop flag count while this entry's span runs (and no longer).
struct AbortScope {
    Device& d;
    explicit AbortScope(Device& device) : d(device) { d.abortable=true; }
    ~AbortScope() { d.abortable=false; }
};
int sd_verify_span(void* handle,uint64_t begin,uint64_t end,uint64_t position,uint64_t steps,
                   const float* input,float* output) {
    try {
        Device& d=device(handle); d.span_drain_error=hipSuccess;
        AbortScope scope(d);
        verify_span(d,size_t(begin),size_t(end),size_t(position),size_t(steps),input,output); return 0;
    }
    catch (const SpanAborted& e) {
        // The graphs launched ahead have ticked their exports by now (the unwinding drained the stream): the host's count follows them.
        Device& d=*static_cast<Device*>(handle);
        if (d.span_drain_error!=hipSuccess) { last_error=hipGetErrorString(d.span_drain_error); return -1; }
        const hipError_t drained=hipStreamSynchronize(d.stream);
        if (drained!=hipSuccess) { last_error=hipGetErrorString(drained); return -1; }
        if (d.exchange) d.hybrid_sequence=*reinterpret_cast<volatile uint32_t*>(d.exchange+hybrid_sequence_slot);
        last_error=e.what(); return -1;
    }
    catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_verify_reserve(void* handle) {
    try { verify_reserve(device(handle)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_verify_commit(void* handle,uint64_t begin,uint64_t end,uint64_t kept) {
    try { verify_commit(device(handle),size_t(begin),size_t(end),size_t(kept)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// sd_verify_commit that replays the kept rows even where every row was kept (a pipelined pass given up behind the one committed).
int sd_verify_commit_forced(void* handle,uint64_t begin,uint64_t end,uint64_t kept) {
    try { verify_commit(device(handle),size_t(begin),size_t(end),size_t(kept),true); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Pipelined verification, CUDA's partially resident layers (flags: bit 0 a second window of snapshots, made when the verification
// buffers are (sd_verify_reserve); bit 1 the expert slots left as they are, no swaps at the end of a span).
int sd_pipeline_setup(void* handle,uint64_t flags) {
    try {
        Device& d=device(handle);
#if defined(__HIPCC__)
        if (flags&1) throw std::runtime_error("pipelined verification is not available on this build");
#else
        d.pipelined=(flags&1)!=0;
#endif
        d.frozen_slots=(flags&2)!=0;
        d.pipelined_chain=(flags&4)!=0;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The expert slots held as they are (1: no swaps at the end of a span) or let swap again (0).
int sd_freeze_slots(void* handle,uint64_t frozen) {
    try { device(handle).frozen_slots=frozen!=0; return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The window of the pass about to run or be committed on this device (0 or 1).
int sd_set_window(void* handle,uint64_t window) {
    try { set_window(device(handle),size_t(window)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Begin a front with a fresh generation for both of its verification spans.
int sd_verify_window_begin(void* handle,uint64_t window,uint64_t position,uint64_t rows,uint64_t* epoch) {
    try {
        Device& d=device(handle);
        if (!epoch) throw std::runtime_error("invalid verification front generation");
        if (!rows) {
            d.front_active=false; d.front_epoch[0]=d.front_epoch[1]=0; d.usage_epoch=0; *epoch=0;
            return 0;
        }
        if (!d.pipelined || rows>hybrid_rows || d.front_generation==UINT64_MAX)
            throw std::runtime_error("invalid verification front generation");
        set_window(d,size_t(window));
        d.front_active=true;
        d.front_epoch[d.window]=++d.front_generation;
        d.front_position[d.window]=size_t(position); d.front_rows[d.window]=size_t(rows);
        for (auto& entry:d.decoder) entry.second.verified=0;
        *epoch=d.front_epoch[d.window];
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// 0: recovered and the touched mask returned; 1: proof absent and the states untouched.
int sd_verify_window_restore(void* handle,uint64_t begin,uint64_t end,uint64_t epoch,uint64_t position,uint64_t rows,uint64_t* touched) {
    try { return verify_window_restore(device(handle),size_t(begin),size_t(end),epoch,size_t(position),size_t(rows),touched); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// restore 0: note what the layers' routes have observed so far (before a speculative pass); 1: put it back (the pass was given up).
int sd_hybrid_usage(void* handle,uint64_t restore) {
    try {
        Device& d=device(handle);
        if (!restore) {
            d.usage_epoch=d.front_epoch[d.window];
            d.usage_marks.clear();
            for (auto& item : d.resident) {
                ResidentMoe& plan=*item.second;
                if (!plan.hybrid) continue;
                d.usage_marks[&plan]={plan.usage,plan.routed_gpu,plan.routed_host,plan.host_read,plan.host_token};
            }
            d.mark_tokens=d.hybrid_tokens; d.mark_balanced=d.hybrid_verify_balanced; d.mark_position=d.hybrid_position;
            return 0;
        }
        for (auto& item : d.usage_marks) {
            ResidentMoe& plan=*item.first;
            plan.usage=item.second.usage; plan.routed_gpu=item.second.routed_gpu; plan.routed_host=item.second.routed_host;
            plan.host_read=item.second.host_read; plan.host_token=item.second.host_token;
        }
        d.hybrid_tokens=d.mark_tokens; d.hybrid_verify_balanced=d.mark_balanced; d.hybrid_position=d.mark_position;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The stop flag of this device's running span, with a reference of its own for the caller (null: none): raise it (sd_abort_flag_set, from
// another thread) and the span stops at its next layer, lower it and the next span runs. No device call; release the reference when done.
void* sd_abort_flag(void* handle) {
    if (!handle) return nullptr;
    AbortFlag* flag=static_cast<Device*>(handle)->abort_flag;
    abort_flag_retain(flag);
    return flag;
}
void sd_abort_flag_retain(void* flag) { if (flag) abort_flag_retain(static_cast<AbortFlag*>(flag)); }
void sd_abort_flag_release(void* flag) { abort_flag_release(static_cast<AbortFlag*>(flag)); }
void sd_abort_flag_set(void* flag,uint64_t raised) { if (flag) static_cast<AbortFlag*>(flag)->raised.store(raised!=0,std::memory_order_release); }
// input null: the rows of layer `layer`'s workspace (the last span's), handed back in `rows`.
const float* head_source(Device& d,const float* input,uint64_t layer) {
    if (input) return nullptr;
    auto found=d.decoder.find(size_t(layer));
    if (found==d.decoder.end()) throw std::runtime_error("invalid Straydog output head rows");
    return verify_layout(found->second.workspace).hidden;
}
int sd_output_head(void* handle,void* const* pointers,const int* kinds,uint64_t steps,uint64_t vocabulary,
                   const float* input,int32_t* tokens,float* logits,float* confidence,uint64_t top,float* best,uint64_t layer,
                   float* rows) {
    try {
        if (!pointers || !kinds || top > head_top_max || (!input && !rows)) throw std::runtime_error("invalid Straydog output head");
        Device& d=device(handle);
        output_head(d,pointers,kinds,size_t(steps),size_t(vocabulary),input,tokens,logits,confidence,head_source(d,input,layer),
                    unsigned(top),best,input ? nullptr : rows);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The head with rows [0, split) evaluated by the linked peer from its copy `peer_head`.
int sd_output_head_split(void* handle,void* const* pointers,const int* kinds,uint64_t steps,uint64_t vocabulary,
                         uint64_t split,void* peer_head,const float* input,int32_t* tokens,float* logits,
                         float* confidence,uint64_t top,float* best,uint64_t layer,float* rows) {
    try {
        if (!pointers || !kinds || top > head_top_max || (!input && !rows)) throw std::runtime_error("invalid Straydog output head");
        Device& d=device(handle);
        output_head_split(d,pointers,kinds,size_t(steps),size_t(vocabulary),size_t(split),peer_head,input,
                          tokens,logits,confidence,head_source(d,input,layer),unsigned(top),best,input ? nullptr : rows);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// One decode row's head (verify_span.hpp: decode_head): the four head operands, the row of 10240, the vocabulary's logits.
int sd_decode_head(void* handle,void* const* pointers,const int* kinds,uint64_t vocabulary,const float* input,float* logits) {
    try {
        if (!pointers || !kinds) throw std::runtime_error("invalid Straydog decode head");
        decode_head(device(handle),pointers,kinds,size_t(vocabulary),input,logits);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_mtp_prologue(void* handle,void* const* pointers,const int* kinds,uint64_t steps,
                    const float* embeddings,const float* hidden,float* output) {
    try {
        if (!pointers || !kinds) throw std::runtime_error("invalid MTP prologue");
        mtp_prologue(device(handle),pointers,kinds,size_t(steps),embeddings,hidden,output); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Layers [begin, end) queued with nothing waited for: their rows stay in the workspace for sd_ple_front.
// input null: the rows are in the span's workspace already (the layers before it, or sd_prefill_handoff, left them there).
int sd_verify_queue(void* handle,uint64_t begin,uint64_t end,uint64_t position,uint64_t steps,const float* input) {
    try {
        {
            // A partially resident first layer: its hybrid handshake runs here and ends in a wait; the rows stay in the workspace.
            Device& d=device(handle);
            auto first=d.decoder.find(size_t(begin));
            if (begin<end && first!=d.decoder.end() && d.resident.find(first->second.moe)->second->hybrid) {
                hybrid_verify_span(d,size_t(begin),size_t(end),size_t(position),size_t(steps),input,nullptr,
                                   input ? nullptr : verify_layout(first->second.workspace).hidden);
                return 0;
            }
        }
        bool parts=false;
        verify_span_queue(device(handle),size_t(begin),size_t(end),size_t(position),size_t(steps),input,nullptr,parts);
        return 0;
    } catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_ple_front(void* handle,void* const* pointers,const int* kinds,void* history,void* snapshot,void* kept,uint64_t layer,
                 uint64_t steps,const float* embeddings,float* first,float* hidden) {
    try {
        if (!pointers || !kinds || !embeddings) throw std::runtime_error("invalid PLE rows");
        ple_front(device(handle),pointers,kinds,history,snapshot,kept,size_t(layer),size_t(steps),embeddings,first,hidden);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_ple_replay(void* handle,void* const* pointers,const int* kinds,void* history,void* snapshot,void* kept,uint64_t steps,
                  const float* embeddings) {
    try {
        if (!pointers || !kinds || !embeddings || !steps || steps > verify_max) throw std::runtime_error("invalid PLE replay");
        ple_replay(device(handle),pointers,kinds,history,snapshot,kept,size_t(steps),embeddings);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
__global__ void draft_gather_rows(const uint8_t* head, size_t row_bytes, const int32_t* tokens, uint8_t* rows) {
    const uint8_t* from = head + size_t(tokens[blockIdx.x]) * row_bytes;
    uint8_t* to = rows + size_t(blockIdx.x) * row_bytes;
    for (size_t i = threadIdx.x; i < row_bytes; i += blockDim.x) to[i] = from[i];
}
// A draft's vocabulary (draft_step with the head split, docs/DESIGN.md §8): the head's rows [0, rows) and those of `count` more
// tokens (each at least `rows`), after the ones before unless `reset`. Its space (draft_vocabulary_capacity rows) is taken by the
// first call. rows 0: drafts read the whole head.
constexpr size_t draft_vocabulary_capacity = 16384;
int sd_draft_vocabulary(void* handle, void* head, int kind, uint64_t vocabulary, uint64_t rows, const int32_t* tokens,
                        uint64_t count, int reset) {
    try {
        Device& d = device(handle);
        const size_t row_bytes = matrix_row_bytes(kind, 2560);
        auto found = d.allocations.find(head);
        if (found == d.allocations.end() || found->second != vocabulary * row_bytes || rows > vocabulary || (count && !tokens) ||
            (d.draft_extras && d.draft_extra_row_bytes != row_bytes))
            throw std::runtime_error("invalid Straydog draft vocabulary");
        for (uint64_t i = 0; i < count; ++i)
            if (uint64_t(tokens[i]) < rows || uint64_t(tokens[i]) >= vocabulary) throw std::runtime_error("invalid Straydog draft token");
        if (reset) d.draft_extra_count = 0;
        d.draft_rows = size_t(rows);
        if (rows && !d.draft_extras) {
            const size_t bytes = draft_vocabulary_capacity * (row_bytes + 4);
            if (bytes > d.budget - d.used) throw std::runtime_error("the Straydog draft vocabulary exceeds the VRAM budget");
            check(hipMalloc(reinterpret_cast<void**>(&d.draft_extras), draft_vocabulary_capacity * row_bytes));
            check(hipMalloc(reinterpret_cast<void**>(&d.draft_extra_tokens), draft_vocabulary_capacity * 4));
            d.used += bytes;
            d.draft_extra_capacity = draft_vocabulary_capacity; d.draft_extra_row_bytes = row_bytes;
        }
        if (!count) return 0;
        if (d.draft_extra_count + count > d.draft_extra_capacity) throw std::runtime_error("the Straydog draft vocabulary is full");
        upload(d, d.draft_extra_tokens + d.draft_extra_count, tokens, size_t(count) * 4);
        draft_gather_rows<<<unsigned(count), 256, 0, d.stream>>>(static_cast<const uint8_t*>(head), row_bytes,
                                                                  d.draft_extra_tokens + d.draft_extra_count,
                                                                  d.draft_extras + d.draft_extra_count * row_bytes);
        check(hipGetLastError());
        d.draft_extra_count += size_t(count);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
// One drafting step (see draft_step): `head` null stops after the MTP layer, `split` 0 evaluates the whole head here.
int sd_draft_step(void* handle,void* const* mtp,const int* mtp_kinds,uint64_t layer,uint64_t position,uint64_t steps,
                  const float* embeddings,const float* hidden,uint64_t previous,void* const* head,const int* head_kinds,
                  uint64_t vocabulary,uint64_t split,void* peer_head,int32_t* token,float* confidence,uint64_t top,float* best) {
    try {
        if (!mtp || !mtp_kinds) throw std::runtime_error("invalid drafting step");
        draft_step(device(handle),mtp,mtp_kinds,size_t(layer),size_t(position),size_t(steps),embeddings,hidden,
                   size_t(previous),head,head_kinds,size_t(vocabulary),size_t(split),peer_head,token,confidence,unsigned(top),best);
        return 0;
    } catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Rows [first, first + count) of the rows `layer`'s last drafting step left in its workspace, to the host (the drafter's own hidden states, 10240 floats a row);
// the range must lie within the rows that step wrote (the layer's `verified` count).
int sd_draft_rows(void* handle,uint64_t layer,uint64_t first,uint64_t count,float* out) {
    try {
        Device& d=device(handle);
        auto found=d.decoder.find(size_t(layer));
        // (only rows the layer's last step wrote: `verified` is that step's row count)
        if (!out || found==d.decoder.end() || !count || first>verify_max || count>verify_max-first || first+count>found->second.verified)
            throw std::runtime_error("invalid drafting rows");
        download(d,out,verify_layout(found->second.workspace).hidden+size_t(first)*10240,size_t(count)*10240*4);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_draft_queue(void* handle,void* const* mtp,const int* mtp_kinds,uint64_t layer,uint64_t position,uint64_t steps,
                   const float* embeddings,const float* hidden,uint64_t previous) {
    try {
        if (!mtp || !mtp_kinds) throw std::runtime_error("invalid drafting step");
        bool parts=false;
        draft_queue(device(handle),mtp,mtp_kinds,size_t(layer),size_t(position),size_t(steps),embeddings,hidden,size_t(previous),parts);
        if (parts) throw std::runtime_error("a queued drafting step needs a layer in parts");
        return 0;
    } catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_draft_head(void* handle,void* const* head,const int* head_kinds,uint64_t layer,uint64_t row,uint64_t vocabulary,
                  uint64_t split,void* peer_head,int32_t* token,float* confidence,uint64_t top,float* best) {
    try {
        Device& d=device(handle);
        auto found=d.decoder.find(size_t(layer));
        if (found==d.decoder.end() || row>=verify_max || found->second.verified<=row)
            throw std::runtime_error("invalid drafting head row");
        draft_head(d,head,head_kinds,size_t(vocabulary),size_t(split),peer_head,
                   verify_layout(found->second.workspace).hidden+size_t(row)*10240,false,token,confidence,unsigned(top),best);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// A deferred prompt's MTP pairs stay on the MTP layer's device (sd_mtp_keep, sd_mtp_embed,
// sd_mtp_prompt_kept): nothing waits on the host between the prompt's batches.
float* mtp_allocation(Device& d,void* base,size_t floats) {
    auto found=d.allocations.find(base);
    if (!base || found==d.allocations.end() || found->second<floats*4) throw std::runtime_error("invalid MTP prompt buffer");
    return static_cast<float*>(base);
}
// Row 0 of `rows` becomes the pair row before the batch: row `previous` of the last batch kept
// there (0: none), or `host` when given; rows 1..steps receive the batch's rows from the
// workspace of `layer`.
int sd_mtp_keep(void* handle,uint64_t layer,uint64_t steps,void* rows,uint64_t previous,const float* host) {
    try {
        Device& d=device(handle);
        auto found=d.decoder.find(size_t(layer));
        if (found==d.decoder.end() || !steps || steps>found->second.capacity) throw std::runtime_error("invalid MTP rows");
        float* kept=mtp_allocation(d,rows,(std::max(size_t(steps),size_t(previous))+1)*10240);
        if (host) upload(d,kept,host,10240*4);
        else if (previous)
            check(hipMemcpyAsync(kept,kept+size_t(previous)*10240,10240*4,hipMemcpyDeviceToDevice,d.stream));
        check(hipMemcpyAsync(kept+10240,found->second.workspace,size_t(steps)*10240*4,hipMemcpyDeviceToDevice,d.stream));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The buffers sd_row_staging_prefetch pinned become the three stagings' (each given its two, at the size they were pinned at), where nothing was
// staged yet; freed where a staging has buffers already.
void adopt_staging_prefetch(Device& d) {
    if (!d.staging_prefetch.joinable()) return;
    d.staging_prefetch.join();
    RowStaging* stagings[3] = {&d.mtp_rows, &d.prefill_rows, &d.ple_rows};
    for (int k = 0; k < 3; ++k) {
        RowStaging& staging = *stagings[k];
        float** ready = d.staging_ready[k];
        const bool free_to_take = d.staging_ready_floats && ready[0] && ready[1] && !staging.floats && !staging.host[0] && !staging.host[1];
        for (int i = 0; i < 2; ++i) {
            if (free_to_take) {
                if (!staging.staged[i]) check(hipEventCreateWithFlags(&staging.staged[i], hipEventDisableTiming));
                staging.host[i] = ready[i];
            } else if (ready[i]) cleanup(hipHostFree(ready[i]));
            ready[i] = nullptr;
        }
        if (free_to_take) staging.floats = d.staging_ready_floats;
    }
}
// `count` embedding rows into `target`, staged without waiting for the device's stream.
// `floats` rows from the host into `device` on the stream, through one of `staging`'s two page-locked buffers: the host waits only
// for that buffer's copy before (two batches back), never for the stream. Pinned at the size first asked, grown when asked more;
// a machine short of memory copies straight from the caller's memory, which the driver stages.
void stage_rows(Device& d, RowStaging& staging, float* device, const float* host, size_t floats) {
    adopt_staging_prefetch(d);
    if (floats > staging.floats && (!staging.refused || floats < staging.refused)) {
        for (int i = 0; i < 2; ++i) {
            if (!staging.staged[i]) check(hipEventCreateWithFlags(&staging.staged[i], hipEventDisableTiming));
            check(hipEventSynchronize(staging.staged[i]));
            if (staging.host[i]) { check(hipHostFree(staging.host[i])); staging.host[i] = nullptr; }
        }
        staging.floats = 0;
        bool pinned = true;
        for (int i = 0; i < 2 && pinned; ++i)
            pinned = host_allocate(reinterpret_cast<void**>(&staging.host[i]), floats * 4, hipHostMallocDefault) == hipSuccess;
        if (pinned) staging.floats = floats;
        else {
            (void)hipGetLastError();
            for (int i = 0; i < 2; ++i) if (staging.host[i]) { cleanup(hipHostFree(staging.host[i])); staging.host[i] = nullptr; }
            staging.refused = floats;
        }
    }
    if (floats > staging.floats) {
        check(hipMemcpyAsync(device, host, floats * 4, hipMemcpyHostToDevice, d.stream));
        return;
    }
    const unsigned slot = staging.turn++ % 2;
    check(hipEventSynchronize(staging.staged[slot]));
    StagingCopier::instance().copy(reinterpret_cast<uint8_t*>(staging.host[slot]), reinterpret_cast<const uint8_t*>(host), floats * 4);
    check(hipMemcpyAsync(device, staging.host[slot], floats * 4, hipMemcpyHostToDevice, d.stream));
    check(hipEventRecord(staging.staged[slot], d.stream));
}
int sd_mtp_embed(void* handle,const float* embeddings,uint64_t count,void* target) {
    try {
        Device& d=device(handle);
        if (!embeddings || !count) throw std::runtime_error("invalid MTP embeddings");
        float* device_rows=mtp_allocation(d,target,size_t(count)*2560);
        stage_rows(d,d.mtp_rows,device_rows,embeddings,size_t(count)*2560);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// `count` pairs: kept rows from `row`, embeddings from `offset` (see sd_mtp_keep, sd_mtp_embed).
int sd_mtp_prompt_kept(void* handle,void* const* pointers,const int* kinds,void* rows,uint64_t row,void* embeddings,
                       uint64_t offset,uint64_t count,uint64_t position,uint64_t layer) {
    try {
        if (!pointers || !kinds || !count) throw std::runtime_error("invalid MTP prompt rows");
        Device& d=device(handle);
        const float* kept=mtp_allocation(d,rows,size_t(row+count)*10240)+size_t(row)*10240;
        const float* embedded=mtp_allocation(d,embeddings,size_t(offset+count)*2560)+size_t(offset)*2560;
        {
            // A pass of more rows than the prologue's scratch holds (512 of its 25,600 floats a row in the input area): the prologue runs over pieces,
            // each piece's rows into the layer's workspace at their place, and the layer takes the pass's rows at once, one pass's fixed costs (the
            // CPU's routes, the streamed experts, the host's reads) for all of them.
            constexpr size_t piece=512;
            auto wide=d.decoder.find(size_t(layer));
            if (count>piece && !(wide!=d.decoder.end() && wide->second.parts!=2 && count<=wide->second.capacity)) {
                // (a layer that cannot take the pass whole, in parts or a workspace of fewer rows: the pass in passes of 512)
                for (uint64_t at=0;at<count;at+=piece)
                    if (sd_mtp_prompt_kept(handle,pointers,kinds,rows,row+at,embeddings,offset+at,std::min<uint64_t>(piece,count-at),position+at,layer)) return -1;
                return 0;
            }
            if (count>piece) {
                for (size_t at=0;at<size_t(count);at+=piece) {
                    const size_t n=std::min(piece,size_t(count)-at);
                    mtp_prologue_check(d,pointers,kinds,n);
                    float* part=mtp_prologue_hidden(d,n);
                    check(hipMemcpyAsync(part,kept+at*10240,n*10240*4,hipMemcpyDeviceToDevice,d.stream));
                    check(hipMemcpyAsync(mtp_prologue_embeddings(d),embedded+at*2560,n*2560*4,hipMemcpyDeviceToDevice,d.stream));
                    mtp_prologue_device(d,pointers,kinds,n);
                    check(hipMemcpyAsync(wide->second.workspace+at*10240,part,n*10240*4,hipMemcpyDeviceToDevice,d.stream));
                }
                prefill_span(d,size_t(layer),size_t(layer)+1,size_t(position),size_t(count),nullptr,nullptr,0);
                return 0;
            }
        }
        mtp_prologue_check(d,pointers,kinds,size_t(count));
        float* hidden=mtp_prologue_hidden(d,size_t(count));
        phase(d,"pro_begin");
        check(hipMemcpyAsync(hidden,kept,size_t(count)*10240*4,hipMemcpyDeviceToDevice,d.stream));
        check(hipMemcpyAsync(mtp_prologue_embeddings(d),embedded,size_t(count)*2560*4,hipMemcpyDeviceToDevice,d.stream));
        phase(d,"pro_copy");
        mtp_prologue_device(d,pointers,kinds,size_t(count));
        auto target=d.decoder.find(size_t(layer));
        // (a layer in parts: its rows into the workspace at the pairs' offset, the host running the layer on both devices once
        // over the batch's pairs)
        if (target!=d.decoder.end() && target->second.parts==2) {
            if (offset+count>target->second.capacity) throw std::runtime_error("MTP prompt rows have no compatible MTP layer");
            check(hipMemcpyAsync(target->second.workspace+size_t(offset)*10240,hidden,size_t(count)*10240*4,hipMemcpyDeviceToDevice,
                                 d.stream));
            return 0;
        }
        if (count==1) {
            std::vector<float> one(10240),output(10240);
            download(d,one.data(),hidden,10240*4);
            verify_span(d,size_t(layer),size_t(layer)+1,size_t(position),1,one.data(),output.data());
            return 0;
        }
        if (target==d.decoder.end() || count>target->second.capacity)
            throw std::runtime_error("MTP prompt rows have no compatible MTP layer");
        check(hipMemcpyAsync(target->second.workspace,hidden,size_t(count)*10240*4,hipMemcpyDeviceToDevice,d.stream));
        prefill_span(d,size_t(layer),size_t(layer)+1,size_t(position),size_t(count),nullptr,nullptr,0);
        return 0;
    } catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_mtp_prompt_rows(void* handle,void* const* pointers,const int* kinds,void* stash,uint64_t stash_layer,
                       uint64_t first,uint64_t count,const float* previous,const float* embeddings,
                       uint64_t position,uint64_t layer) {
    try {
        if (!pointers || !kinds || !embeddings || !count || (previous && count<1))
            throw std::runtime_error("invalid MTP prompt rows");
        const size_t stashed=size_t(count)-(previous ? 1 : 0);
        Device& s=device(stash);
        auto source=s.decoder.find(size_t(stash_layer));
        if (source==s.decoder.end() || first+stashed>source->second.capacity)
            throw std::runtime_error("MTP prompt rows have no compatible stash");
        const float* rows=source->second.workspace;
        // The rows reached the stash on its own stream (sd_prefill_handoff): wait for them there.
        check(hipStreamSynchronize(s.stream));
        Device& d=device(handle);
        mtp_prologue_check(d,pointers,kinds,size_t(count));
        float* hidden=mtp_prologue_hidden(d,size_t(count));
        if (previous) upload(d,hidden,previous,10240*4);
        if (stashed) check(hipMemcpyPeerAsync(hidden+(size_t(count)-stashed)*10240,d.ordinal,rows+size_t(first)*10240,
                                              s.ordinal,stashed*10240*4,d.stream));
        upload(d,mtp_prologue_embeddings(d),embeddings,size_t(count)*2560*4);
        mtp_prologue_device(d,pointers,kinds,size_t(count));
        if (count==1) {
            // One pair goes through the MTP layer as a verification row, as a host row would.
            std::vector<float> row(10240),output(10240);
            download(d,row.data(),hidden,10240*4);
            verify_span(d,size_t(layer),size_t(layer)+1,size_t(position),1,row.data(),output.data());
            return 0;
        }
        auto target=d.decoder.find(size_t(layer));
        if (target==d.decoder.end() || count>target->second.capacity)
            throw std::runtime_error("MTP prompt rows have no compatible MTP layer");
        check(hipMemcpyAsync(target->second.workspace,hidden,size_t(count)*10240*4,hipMemcpyDeviceToDevice,d.stream));
        prefill_span(d,size_t(layer),size_t(layer)+1,size_t(position),size_t(count),nullptr,nullptr,0);
        return 0;
    } catch (const HybridWaitTimeout&) { return recover_hybrid_wait(handle); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_ple_rows(void* handle,void* const* pointers,const int* kinds,void* history,uint64_t steps,
                const float* embeddings,const float* hidden,float* output) {
    try { ple(device(handle),pointers,kinds,history,size_t(steps),embeddings,hidden,output,nullptr,true); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Copy one owned allocation onto another of the same size (state snapshots).
int sd_copy_allocation(void* handle,void* target,const void* source) {
    try {
        auto& d=device(handle);
        auto t=d.allocations.find(target), s=d.allocations.find(const_cast<void*>(source));
        if (t==d.allocations.end() || s==d.allocations.end() || t->second!=s->second || target==source)
            throw std::runtime_error("invalid Straydog state copy");
        kv_invalidate_write(d,target,0);
        check(hipMemcpyAsync(target,source,t->second,hipMemcpyDeviceToDevice,d.stream));
        check(hipStreamSynchronize(d.stream));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_bank_allocate(void* handle, uint64_t bytes, void** storage) {
    try {
        if (!storage || !bytes) throw std::runtime_error("invalid layer bank allocation");
        *storage = nullptr;
        auto& d = device(handle);
        if (bytes > d.budget - d.used || allocation_tail > d.budget - d.used - bytes) return 1;
        const hipError_t status = hipMalloc(storage, size_t(bytes) + allocation_tail);
        if (status == hipErrorOutOfMemory) { (void)hipGetLastError(); *storage = nullptr; return 1; }
        check(status);
        try { d.allocations.emplace(*storage, size_t(bytes)); }
        catch (...) { cleanup(hipFree(*storage)); *storage = nullptr; throw; }
        d.used += size_t(bytes) + allocation_tail;
        check(hipMemsetAsync(static_cast<uint8_t*>(*storage) + bytes, 0, allocation_tail, d.stream));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_bank(void* handle, uint64_t layer, void* storage, uint64_t bytes) {
    try { load_prefill_bank(device(handle), size_t(layer), storage, size_t(bytes)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_bank_from(void* handle, uint64_t layer, void* storage, uint64_t bytes,
                         const uint8_t* const* sources, const uint64_t* source_bytes) {
    try {
        if (!sources || !source_bytes) throw std::runtime_error("missing copied layer bank sources");
        load_prefill_bank(device(handle), size_t(layer), storage, size_t(bytes), sources, source_bytes);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_bank_clear(void* handle) {
    try { clear_prefill_bank(device(handle)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_observe(void* handle, sd_prefill_observer observer, void* context) {
#if defined(STRAYDOG_PREFILL_OBSERVER)
    try {
        auto& d = device(handle);
        check(hipStreamSynchronize(d.stream));
        d.prefill_observer = observer;
        d.prefill_observer_context = observer ? context : nullptr;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
#else
    (void)handle; (void)context;
    if (!observer) return 0;
    last_error = "prefill observer is not included in this build";
    return -1;
#endif
}
int sd_prefill_rows(void* handle, uint64_t layer, const float* rows, uint64_t steps) {
    try {
        auto& d = device(handle);
        const auto found = d.decoder.find(size_t(layer));
        if (!rows || found == d.decoder.end() || !steps || steps > found->second.capacity)
            throw std::runtime_error("invalid prefill hidden rows");
        // (the stream is idle: the span before this one waited for its output. The last piece's copy lands under the next span)
        const auto started = std::chrono::steady_clock::now();
        upload(d, found->second.workspace, rows, size_t(steps) * 10240 * 4);
        d.rows_up_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        d.rows_up_bytes += uint64_t(steps) * 10240 * 4;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Layer order's hidden rows held in VRAM for a pass (an allocation of sd_prefill_bank_allocate): rows [first, first + steps) of `storage`,
// 10240 floats a row.
static float* held_layer_rows(Device& d, void* storage, uint64_t first, uint64_t steps) {
    auto found = d.allocations.find(storage);
    if (!storage || found == d.allocations.end() || !steps || first + steps > found->second / (10240 * 4))
        throw std::runtime_error("invalid layer rows held in VRAM");
    return static_cast<float*>(storage) + size_t(first) * 10240;
}
// A pass's rows from the host into the rows held in VRAM (the embeddings, once a pass).
int sd_layer_rows_put(void* handle, void* storage, uint64_t first, const float* rows, uint64_t steps) {
    try {
        auto& d = device(handle);
        float* held = held_layer_rows(d, storage, first, steps);
        if (!rows) throw std::runtime_error("no layer rows to put");
        const auto started = std::chrono::steady_clock::now();
        upload(d, held, rows, size_t(steps) * 10240 * 4);
        d.rows_up_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        d.rows_up_bytes += uint64_t(steps) * 10240 * 4;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The rows held in VRAM back to the host (the last layer's output, once a pass), after the work queued before them.
int sd_layer_rows_get(void* handle, void* storage, uint64_t first, float* rows, uint64_t steps) {
    try {
        auto& d = device(handle);
        const float* held = held_layer_rows(d, storage, first, steps);
        if (!rows) throw std::runtime_error("no place for the layer rows");
        // (the wait the download makes anyway, first: what is timed is the copy alone)
        check(hipStreamSynchronize(d.stream));
        const auto started = std::chrono::steady_clock::now();
        download(d, rows, held, size_t(steps) * 10240 * 4);
        d.rows_down_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        d.rows_down_bytes += uint64_t(steps) * 10240 * 4;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// sd_prefill_rows from rows held in VRAM: a device copy into the layer's workspace.
int sd_prefill_rows_held(void* handle, uint64_t layer, void* storage, uint64_t first, uint64_t steps) {
    try {
        auto& d = device(handle);
        const auto found = d.decoder.find(size_t(layer));
        if (found == d.decoder.end() || !steps || steps > found->second.capacity)
            throw std::runtime_error("invalid prefill hidden rows");
        const float* held = held_layer_rows(d, storage, first, steps);
        check(hipMemcpyAsync(found->second.workspace, held, size_t(steps) * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// sd_prefill_span of the rows a workspace holds, its output copied back into the rows held in VRAM: the same span, with its check of
// the values each span (as with an output to the host), and a device copy where the host's was.
int sd_prefill_span_held(void* handle, uint64_t begin, uint64_t end, uint64_t position, uint64_t steps, void* storage, uint64_t first) {
    auto& d = device(handle);
    const bool deferred = d.prefill_deferred;
    try {
        float* held = held_layer_rows(d, storage, first, steps);
        const auto found = d.decoder.find(size_t(begin));
        if (found == d.decoder.end()) throw std::runtime_error("prefill span has no compatible resident layer");
        d.prefill_deferred = false;
        prefill_span(d, size_t(begin), size_t(end), size_t(position), size_t(steps), nullptr, nullptr, 0, nullptr);
        d.prefill_deferred = deferred;
        check(hipMemcpyAsync(held, found->second.workspace, size_t(steps) * 10240 * 4, hipMemcpyDeviceToDevice, d.stream));
        return 0;
    } catch (const std::exception& e) { d.prefill_deferred = deferred; last_error=e.what(); return -1; }
}
// Layer order's hidden rows moved so far: up nanoseconds, up bytes, down nanoseconds, down bytes.
int sd_prefill_row_transfers(void* handle, uint64_t* out) {
    try {
        auto& d = device(handle);
        if (!out) throw std::runtime_error("no place for the row transfers");
        out[0] = d.rows_up_ns; out[1] = d.rows_up_bytes; out[2] = d.rows_down_ns; out[3] = d.rows_down_bytes;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_span(void* handle,uint64_t begin,uint64_t end,uint64_t position,uint64_t steps,
                     const float* input,float* output,uint64_t output_rows,double* timings) {
    try {
        prefill_span(device(handle),size_t(begin),size_t(end),size_t(position),size_t(steps),input,output,
                     size_t(output_rows),timings);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Deferred prefill: while on, spans that return no rows end without waiting for the device.
int sd_rope_delta(void* handle,uint32_t delta) {
    try {
        device(handle).rope_delta=delta;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}

int sd_rope_rows(void* handle,uint64_t first,uint64_t count,const uint32_t* axes) {
    try {
        Device& d=device(handle);
        constexpr size_t most=4096+4;
        if (count>most || (count && !axes)) throw std::runtime_error("invalid Straydog rope table");
        if (count && !d.rope_rows) check(hipMalloc(reinterpret_cast<void**>(&d.rope_rows),most*3*4));
        if (count) upload(d,d.rope_rows,axes,size_t(count)*3*4);
        d.rope_first=size_t(first); d.rope_count=size_t(count);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}

int sd_prefill_defer(void* handle,int on) {
    try {
        Device& d=device(handle);
        if (on && !d.prefill_deferred) check(hipMemsetAsync(d.flags+1,0,4,d.stream));
        d.prefill_deferred=on!=0;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Waits for the device, ends deferral and fails if a deferred span produced non-finite rows.
int sd_prefill_settle(void* handle) {
    try {
        Device& d=device(handle);
        unsigned flag=0;
        download(d,&flag,d.flags+1,4);
        check(hipStreamSynchronize(d.helper_stream));
        d.prefill_deferred=false;
        if (d.link.peer) peer_settle(d);
        if (flag) throw std::runtime_error("Straydog GPU produced non-finite decoder state");
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_handoff(void* from,void* to,uint64_t from_layer,uint64_t to_layer,uint64_t steps) {
    try {
        auto workspace=[](Device& d,uint64_t layer,uint64_t steps) {
            auto found=d.decoder.find(size_t(layer));
            if (found==d.decoder.end() || !steps || steps>found->second.capacity)
                throw std::runtime_error("prefill handoff has no compatible resident layer");
            return found->second.workspace;
        };
        Device& a=device(from);
        const float* source=workspace(a,from_layer,steps);
        Device& b=device(to);
        float* target=workspace(b,to_layer,steps);
        // The copy starts once the source stream has written the rows, and the source's later
        // work (the next batch, with deferred spans) waits until the rows have left.
        on_device(a.ordinal,[&]() { check(hipEventRecord(a.handoff,a.stream)); });
        on_device(b.ordinal,[&]() {
            check(hipStreamWaitEvent(b.stream,a.handoff,0));
            check(hipMemcpyPeerAsync(target,b.ordinal,source,a.ordinal,size_t(steps)*10240*4,b.stream));
            check(hipEventRecord(b.handoff,b.stream));
        });
        on_device(a.ordinal,[&]() { check(hipStreamWaitEvent(a.stream,b.handoff,0)); });
        // Both sides of the handoff now have their work queued.
#if !defined(__HIPCC__)
        on_device(a.ordinal,[&]() { flush_launches(a); });
        on_device(b.ordinal,[&]() { flush_launches(b); });
#endif
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
__global__ void prefill_tile(const float* embeddings,size_t steps,float* hidden) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index>=steps*10240) return;
    const size_t row=index/10240,column=index%10240%2560;
    hidden[index]=embeddings[row*2560+column];
}
int sd_prefill_embed(void* handle,uint64_t layer,const float* embeddings,uint64_t steps) {
    try {
        Device& d=device(handle);
        auto found=d.decoder.find(size_t(layer));
        if (!embeddings || found==d.decoder.end() || !steps || steps>found->second.capacity)
            throw std::runtime_error("prefill embeddings have no compatible resident layer");
        // The rows land after the hidden rows (where a span's mixed rows go) and are tiled from there.
        float* hidden=found->second.workspace; float* rows=hidden+size_t(steps)*10240;
        stage_rows(d,d.prefill_rows,rows,embeddings,size_t(steps)*2560);
        prefill_tile<<<unsigned((size_t(steps)*10240+255)/256),256,0,d.stream>>>(rows,size_t(steps),hidden);
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_prefill_ple(void* handle,void* const* pointers,const int* kinds,void* history,uint64_t steps,
                   const float* embeddings,uint64_t layer) {
    try {
        auto& d=device(handle);
        auto found=d.decoder.find(size_t(layer));
        if (found==d.decoder.end() || !steps || steps>found->second.capacity)
            throw std::runtime_error("PLE has no compatible resident decoder workspace");
        // (the batch's rows staged at once where the first layer's were, the rows after the hidden ones, which the spans after it
        // overwrite only once it has read them: 256-row uploads wrapped upload's arena, which waits for the stream. Nothing reads
        // the layer's rows on the host, and what follows is queued on the stream: no wait at its end either.)
        float* rows=found->second.workspace+size_t(steps)*10240;
        stage_rows(d,d.ple_rows,rows,embeddings,size_t(steps)*2560);
        for (size_t first=0;first<steps;first+=256) {
            const size_t count=std::min<size_t>(256,size_t(steps)-first);
            ple(d,pointers,kinds,history,count,rows+first*2560,nullptr,nullptr,found->second.workspace+first*10240,false,false,true);
        }
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_experts_finalize(void* handle) {
    try { finalize_resident_experts(device(handle)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
void* sd_resident_moe_create(void* handle,void* const* pointers,const int* kinds,uint64_t experts,
                            void* workspace,uint64_t capacity) {
    try { return create_resident_moe(device(handle),pointers,kinds,size_t(experts),workspace,size_t(capacity)); }
    catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
// Expert parallelism: a plan holding routed experts [first, first + count),
// and the shared expert when `shared` (pointers 3-5 null otherwise).
void* sd_resident_moe_part(void* handle,void* const* pointers,const int* kinds,uint64_t experts,uint64_t first,
                           uint64_t count,int shared,void* workspace,uint64_t capacity) {
    try {
        return create_resident_moe(device(handle),pointers,kinds,size_t(experts),workspace,size_t(capacity),
                                   size_t(first),size_t(count),shared!=0);
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
// The plan this device evaluates for `layer` while its linked peer owns the layer.
int sd_helper_moe(void* handle,uint64_t layer,void* plan) {
    try {
        auto& value=device(handle);
        auto found=value.resident.find(plan);
        if (layer>48 || found==value.resident.end() || !found->second->part || found->second->shared ||
            value.helper_moe.count(size_t(layer)))
            throw std::runtime_error("invalid resident MoE helper");
        value.helper_moe.emplace(size_t(layer),plan);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Link two devices for expert parallelism; inboxes hold `rows` rows each way.
int sd_peer_attach(void* first,void* second,uint64_t rows) {
    try { attach_peer(device(first),device(second),size_t(rows)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// 1 when the linked devices write each other's VRAM, 0 through host memory, -1 unlinked.
int sd_peer_direct(void* handle) {
    try { const auto& value=device(handle); return value.link.peer ? int(value.link.direct) : -1; }
    catch (const std::exception& e) { last_error=e.what(); return -2; }
}
int sd_resident_moe(void* handle,void* plan,uint64_t steps,const float* input,float* output) {
    try { resident_moe(device(handle),plan,size_t(steps),input,output); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
void* sd_hybrid_moe_create(void* handle,void* const* pointers,const int* kinds,const int* routed_kinds,uint64_t experts,
                           const uint8_t* const* host,void* workspace,uint64_t capacity) {
    try { return create_hybrid_moe(device(handle),pointers,kinds,routed_kinds,size_t(experts),host,workspace,size_t(capacity)); }
    catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
int sd_expert_slots(void* handle,uint64_t cache,uint64_t stage,uint64_t slot_bytes,int mirror) {
    try { create_expert_slots(device(handle),size_t(cache),size_t(stage),size_t(slot_bytes), mirror != 0); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// A K/V growth takes the VRAM of the expert slots for its longer indexers and gives back what is left.
int sd_expert_slots_release(void* handle) {
    try { release_expert_slots(device(handle)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_expert_slots_restore(void* handle,uint64_t cache) {
    try { restore_expert_slots(device(handle),size_t(cache)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Speed first: a route whose expert the CPU would run, of a router weight below `floor`, is left out of its token (its weight
// taken as 0); 0 leaves none out. The GPU's routes are kept whatever their weight: they cost no read from host memory.
int sd_hybrid_route_floor(void* handle, float floor) {
    try {
        if (!(floor >= 0.0f && floor < 1.0f)) throw std::runtime_error("invalid Straydog route floor");
        device(handle).route_floor = floor;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int sd_hybrid_prompt_share(void* handle, int on) {
    try { device(handle).prompt_cpu = on != 0; return 0; } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int sd_hybrid_supported(void* handle) {
    try { const Device& d=device(handle); return int(d.expert_prefill && d.half_gemm); }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_attach_cpu(void* handle,void* engine) {
    try { device(handle).cpu=engine; return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// max_swaps==0 refreshes the whole cache and waits for it.
// Finish background swap copies, which read the mapped banks: call before unmapping them.
int sd_hybrid_drain(void* handle) {
    try {
        auto& d=device(handle);
        if (d.copier) d.copier->wait();
        if (d.copy_stream) check(hipStreamSynchronize(d.copy_stream));
        return 0;
    }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_rebalance(void* handle,uint64_t max_swaps) {
    try {
        auto& d=device(handle);
        hybrid_rebalance(d,size_t(max_swaps),max_swaps==0);
        if (!max_swaps) d.hybrid_tokens=0;   // decode swaps count tokens from here
        return 0;
    }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_timing(void* handle,uint64_t* wait_ns,uint64_t* cpu_ns,uint64_t* swapped,uint64_t* assisted) {
    try {
        if (!wait_ns || !cpu_ns || !swapped || !assisted) throw std::runtime_error("invalid hybrid timers");
        auto& d=device(handle); *wait_ns=d.hybrid_wait_ns; *cpu_ns=d.hybrid_cpu_ns; *swapped=d.swapped_experts;
        *assisted=d.assisted_experts; return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The GPU's side of the hybrid layers (hybrid_clock), in ns: its layer work from the CPU result's arrival to the export,
// over every layer (the span's first has no start: the timed layers' mean stands for it), and its resident experts.
int sd_hybrid_gpu_timing(void* handle,uint64_t* layer_ns,uint64_t* experts_ns) {
    try {
        if (!layer_ns || !experts_ns) throw std::runtime_error("invalid hybrid timers");
        auto& d=device(handle);
        uint64_t clock[4];
        download(d,clock,hybrid_clock(d),sizeof(clock));
#if defined(__HIPCC__)
        int rate=0;
        check(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,d.ordinal));
        const double ns=rate>0 ? 1e6/rate : 0;
#else
        const double ns=1;   // %globaltimer counts nanoseconds
#endif
        *layer_ns=d.hybrid_timed_layers ? uint64_t(double(clock[2])*ns*double(d.hybrid_layers)/double(d.hybrid_timed_layers)) : 0;
        *experts_ns=uint64_t(double(clock[3])*ns);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_prompt_cpu(void* handle,uint64_t* routes) {
    try {
        if (!routes) throw std::runtime_error("invalid hybrid counters");
        *routes=device(handle).prompt_cpu_routes;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_counts(void* handle,uint64_t* gpu,uint64_t* cpu,uint64_t* streamed,uint64_t* cached) {
    try {
        if (!gpu || !cpu || !streamed || !cached) throw std::runtime_error("invalid hybrid counters");
        auto& d=device(handle);
        size_t occupied=0;
        for (const auto& owner : d.slots.owner) occupied+=owner.first!=nullptr;
        *gpu=d.gpu_experts; *cpu=d.cpu_experts; *streamed=d.streamed_experts; *cached=occupied;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// A partially resident layer's plan: the decode and verification rows routed to each of its experts since the last clear, to the device (the expert
// held a slot) and to the host. `out` (room for `room` values) takes the device's counts, then the host's. The experts counted: 0 for a plan of
// another kind; `clear` zeroes the counts after they are read.
int sd_hybrid_routes(void* handle,void* plan,uint32_t* out,uint64_t room,int clear) {
    try {
        auto& d=device(handle);
        const auto found=d.resident.find(plan);
        if (found==d.resident.end()) throw std::runtime_error("invalid hybrid plan");
        ResidentMoe& moe=*found->second;
        if (!moe.hybrid) return 0;
        if (!out || room<2*moe.experts) throw std::runtime_error("invalid hybrid route counts");
        std::copy(moe.routed_gpu.begin(),moe.routed_gpu.end(),out);
        std::copy(moe.routed_host.begin(),moe.routed_host.end(),out+moe.experts);
        if (clear) {
            std::fill(moe.routed_gpu.begin(),moe.routed_gpu.end(),0u);
            std::fill(moe.routed_host.begin(),moe.routed_host.end(),0u);
        }
        return int(moe.experts);
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The scores a partially resident layer's experts are ranked by (hybrid_rebalance), set from `rates`, the routes a token to each expert: what the moving
// average of the routes (hybrid_observe_token) comes to when they keep that rate. A warm start sets them before the first request, so that the slots fill
// with the experts an earlier run used most rather than with what the start-up's own passes left.
int sd_hybrid_seed(void* handle,void* plan,const double* rates,uint64_t experts) {
    try {
        auto& d=device(handle);
        const auto found=d.resident.find(plan);
        if (found==d.resident.end() || !found->second->hybrid || !rates || experts!=found->second->experts)
            throw std::runtime_error("invalid hybrid scores");
        for (size_t expert=0;expert<size_t(experts);++expert)
            if (!(rates[expert]>=0)) throw std::runtime_error("invalid hybrid scores");
        for (size_t expert=0;expert<size_t(experts);++expert)
            found->second->usage[expert]=rates[expert]/(1-hybrid_token_decay);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// One bounded refresh of the expert slots by those scores that never waits for its copies (sd_hybrid_rebalance with max_swaps > 0): `*pending` is the
// swaps still on their way after it (0: the slots hold what the scores want), and the pool's size and occupation are `*slots` and `*held`.
int sd_hybrid_refresh(void* handle,uint64_t max_swaps,uint64_t* pending,uint64_t* slots,uint64_t* held) {
    try {
        if (!max_swaps || !pending || !slots || !held) throw std::runtime_error("invalid hybrid refresh");
        auto& d=device(handle);
        hybrid_rebalance(d,size_t(max_swaps),false);
        size_t occupied=0;
        for (const auto& owner : d.slots.owner) occupied+=owner.first!=nullptr;
        *pending=d.slots.pending.size(); *slots=d.slots.cache; *held=occupied;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_cache_bytes(void* handle,uint64_t* bytes) {
    try {
        if (!bytes) throw std::runtime_error("invalid hybrid cache bytes");
        auto& d=device(handle);
        *bytes=d.slots.base ? d.slots.arena : 0;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The pool's size and occupation, as sd_hybrid_refresh reports them.
int sd_hybrid_slots(void* handle,uint64_t* slots,uint64_t* held) {
    try {
        if (!slots || !held) throw std::runtime_error("invalid hybrid slots");
        auto& d=device(handle);
        size_t occupied=0;
        for (const auto& owner : d.slots.owner) occupied+=owner.first!=nullptr;
        *slots=d.slots.cache; *held=occupied;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
extern "C++" {
// The most absent routes a layer's GPU takes that the engine tries (sd_hybrid_assist_level). CUDA copies them to VRAM's staging slots
// first (hybrid_stage_assist): two a layer, where four saved the CPU 1.3 ms a token and cost 1.7 of waiting on an RTX 5070 Ti. HIP
// reads them in place: four, on an R9700.
#if defined(__HIPCC__)
constexpr uint32_t hybrid_assist_cap=4;
#else
constexpr uint32_t hybrid_assist_cap=8;   // CUDA: the distinct missed experts of a layer's pass the GPU stages (hybrid_tick_plan); the level is that many
#endif
// A layer's assist level and the banks it reads in place are baked into its captured graphs (hybrid_assist is passed to the layer's
// kernels by value, in decode's graphs and in verification's alike): a change of level, or of the plans whose banks are mapped, drops
// every hybrid layer graph, whose replay would keep the old, and the next pass captures them again. Graphs of other tags hold no level.
void hybrid_drop_graphs(Device& d) {
    check(hipStreamSynchronize(d.stream));
    for (auto graph=d.graphs.begin();graph!=d.graphs.end();) {
        if (graph->first.compare(0,7,"hybrid-")==0) {
            check(hipGraphExecDestroy(graph->second.executable));
            check(hipGraphDestroy(graph->second.graph));
            graph=d.graphs.erase(graph);
        } else ++graph;
    }
}
void hybrid_set_assist(Device& d,uint32_t level) {
    if (level==d.assist) return;
    hybrid_drop_graphs(d);
    d.assist=level;
}
} // extern "C++"
uint32_t sd_hybrid_assist_cap(void) { return hybrid_assist_cap; }
// Let the GPU read the first `assist` absent routes of a layer straight from
// page-locked host banks while the CPU evaluates the rest. Returns plans mapped.
// It may be called again as more banks are page-locked (the engine locks them behind "ready", a layer at a time): a plan already mapped
// stays so (its banks stay locked until the device ends), those now locked are mapped, and the layers' graphs are captured anew if
// any were, since a graph captured before holds the plan's banks as they were.
int sd_hybrid_assist(void* handle,uint32_t assist,uint64_t* mapped) {
    try {
        if (!mapped || assist>10) throw std::runtime_error("invalid hybrid assist");
        auto& d=device(handle);
        *mapped=0;
        bool changed=false;
        for (auto& item : d.resident) {
            ResidentMoe& plan=*item.second;
            if (!plan.hybrid) continue;
            if (plan.mapped[0]) { ++*mapped; continue; }
            void* views[3]{};
            bool ok=true;
            for (size_t role=0;role<3 && ok;++role) {
                if (plan.host_copy && plan.host_base[role]) {
                    // A copy's bank lies inside its allocation: the runtime maps the allocation, not an address in it, and the bank's view is at the same offset.
                    ok=hipHostGetDevicePointer(&views[role],const_cast<uint8_t*>(plan.host_base[role]),0)==hipSuccess;
                    if (ok) views[role]=static_cast<uint8_t*>(views[role])+(plan.host[role]-plan.host_base[role]);
                } else ok=hipHostGetDevicePointer(&views[role],const_cast<uint8_t*>(plan.host[role]),0)==hipSuccess;
            }
            (void)hipGetLastError();   // clear the failed lookup
            for (size_t role=0;role<3;++role) plan.mapped[role]=ok ? static_cast<const uint8_t*>(views[role]) : nullptr;
            *mapped+=ok;
            changed|=ok;
        }
        if (changed) hybrid_drop_graphs(d);
        d.hint_experts=*mapped==0;
#if defined(__HIPCC__)
        hybrid_set_assist(d,assist);
#else
        // CUDA: an assisted expert read over PCIe by the zero-copy GEMV outlasts the CPU experts on the
        // measured RTX 5070 Ti, so each layer waits for it: a kernel copies it to a staging slot first
        // (hybrid_stage_assist).
        hybrid_set_assist(d,std::min(assist,hybrid_assist_cap));
#endif
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The GPU stops reading a layer's page-locked banks in place: the plan whose gate bank is at `gate` has its banks as it had them before
// sd_hybrid_assist mapped them, so the slots' copies and the CPU's experts read them as they read any layer's that is not locked. The copies
// of the banks' experts that were in flight are waited for, and so are the layers' graphs that hold the view (they are dropped, and the next
// pass captures them again). `found` is 1 where this device holds the plan. The caller unregisters the banks after: the engine gives
// locked layers back when a K/V growth finds no memory (Model::give_back).
int sd_hybrid_unmap(void* handle,const void* gate,uint64_t* found) {
    try {
        if (!gate || !found) throw std::runtime_error("invalid hybrid unmap");
        auto& d=device(handle);
        *found=0;
        for (auto& item : d.resident) {
            ResidentMoe& plan=*item.second;
            if (!plan.hybrid || plan.host[0]!=gate) continue;
            *found=1;
            if (!plan.mapped[0]) break;
            if (d.copier) d.copier->wait();
            if (d.copy_stream) check(hipStreamSynchronize(d.copy_stream));
            hybrid_drop_graphs(d);   // (it waits for the stream: no kernel is still reading the banks)
            for (size_t role=0;role<3;++role) plan.mapped[role]=nullptr;
            break;
        }
        bool any=false;
        for (const auto& item : d.resident) any|=item.second->hybrid && item.second->mapped[0];
        d.hint_experts=!any;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The plan whose host banks begin at `from` (its gate bank's address) reads the same banks from `host` (gate, up and down) from now on: the engine
// holds page-locked copies of the banks where the GPU cannot page-lock the model's read-only mapping (`copy` 1, with `base`: the start of the allocation
// each bank lies in, which the assist maps), and points the plan back at the mapping before it frees them (`copy` 0, `base` not read). Nothing may read the old memory when this returns: the swap copies in flight (the device's thread and the copy
// stream) and the kernels that read the banks in place or hold the view in a captured graph (the device) are waited for, and so are the CPU's workers
// (BankReading; they run inside the engine's calls on this thread, so a count above zero is a fault, and the plan is left as it was). The plan's view
// for the GPU's assist is dropped: sd_hybrid_assist maps the new memory. `found` is 1 where this device holds the plan.
int sd_hybrid_repoint(void* handle,const void* from,const uint8_t* const* host,const uint8_t* const* base,int copy,uint64_t* found) {
    try {
        if (!from || !host || !host[0] || !host[1] || !host[2] || !found) throw std::runtime_error("invalid hybrid repoint");
        if (copy && (!base || !base[0] || !base[1] || !base[2])) throw std::runtime_error("invalid hybrid repoint: a copy's allocations");
        auto& d=device(handle);
        *found=0;
        ResidentMoe* target=nullptr;
        for (auto& item : d.resident)
            if (item.second->hybrid && item.second->host[0]==from) { target=item.second.get(); break; }
        if (!target) return 0;
        *found=1;
        if (d.copier) d.copier->wait();
        if (d.copy_stream) check(hipStreamSynchronize(d.copy_stream));
        hybrid_drop_graphs(d);   // (it waits for the stream: no kernel is still reading the banks)
        check(hipDeviceSynchronize());
        for (unsigned waited=0;d.bank_readers.load()!=0;++waited) {
            if (waited>5000) throw std::runtime_error("the CPU's experts are still reading the layer's banks");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (size_t role=0;role<3;++role) {
            target->mapped[role]=nullptr; target->host[role]=host[role]; target->host_base[role]=copy ? base[role] : nullptr;
        }
        target->host_copy=copy!=0;
        target->prefetched.clear();
        bool any=false;
        for (const auto& item : d.resident) any|=item.second->hybrid && item.second->mapped[0];
        d.hint_experts=!any;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The assist level alone, once sd_hybrid_assist has mapped the banks (mapping them again is the dear part of that call). A level past
// the cap is refused.
int sd_hybrid_assist_level(void* handle,uint32_t assist) {
    try {
        if (assist>hybrid_assist_cap) throw std::runtime_error("invalid hybrid assist level");
        hybrid_set_assist(device(handle),assist);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_hybrid_decay(double decay) {
    if (!(decay>0 && decay<1)) { last_error="invalid hybrid routing decay"; return -1; }
    hybrid_token_decay=decay; return 0;
}
// Page-lock an already mapped read-only range so expert copies use direct DMA.
int sd_host_register(void* pointer,uint64_t bytes) {
    try {
        if (!pointer || !bytes) throw std::runtime_error("invalid host registration");
        hipError_t state=hipHostRegister(pointer,size_t(bytes),hipHostRegisterPortable|hipHostRegisterReadOnly|hipHostRegisterMapped);
        // A device that cannot register memory read-only (CUDA's cudaDevAttrHostRegisterReadOnlySupported 0: two RTX 3090s on Windows
        // 10 refused a loader's slot so, and the start failed) takes it without the flag: a writable range such as that slot registers
        // so, and a read-only mapping then fails as it would have, which its caller goes on without (the layer is not page-locked).
        if (state==hipErrorNotSupported) {
            (void)hipGetLastError();
            state=hipHostRegister(pointer,size_t(bytes),hipHostRegisterPortable|hipHostRegisterMapped);
        }
        check(state);
        std::lock_guard<std::mutex> lock(registered_mutex);
        registered[reinterpret_cast<uintptr_t>(pointer)]={size_t(bytes),false};
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// sd_host_register from a thread that is not the device's own (the engine's locking behind "ready"): the thread's current device is set to
// `ordinal` first, so that no context opens on another device. Page locks are portable: every device sees them.
int sd_host_register_on(int ordinal,void* pointer,uint64_t bytes) {
    try {
        check(hipSetDevice(ordinal));
        return sd_host_register(pointer,bytes);
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// `bytes` of page-locked host memory that every device sees (portable) and can read in place (mapped): the engine's copies of the expert banks where
// the GPU cannot register the model's read-only mapping (sd_hybrid_repoint). Allocated under the context of device `ordinal`, from any thread.
int sd_host_alloc_on(int ordinal,uint64_t bytes,void** pointer) {
    try {
        if (!pointer || !bytes || bytes>std::numeric_limits<size_t>::max()) throw std::runtime_error("invalid host allocation");
        *pointer=nullptr;
        check(hipSetDevice(ordinal));
        void* value=nullptr;
#if !defined(__HIPCC__)
        value=::operator new(size_t(bytes),std::align_val_t(4096),std::nothrow);
        if (value) {
            // Record ownership while the pointer is private, before registering its writable pages.
            try {
                std::lock_guard<std::mutex> lock(registered_host_allocations_mutex);
                registered_host_allocations.insert(value);
            } catch (const std::bad_alloc&) {
                ::operator delete(value,std::align_val_t(4096));
                value=nullptr;
            } catch (...) {
                ::operator delete(value,std::align_val_t(4096));
                throw;
            }
        }
        if (value) {
            const cudaError_t state=cudaHostRegister(value,size_t(bytes),cudaHostRegisterPortable|cudaHostRegisterMapped);
            if (state==cudaSuccess) {
                *pointer=value;
                return 0;
            }
            (void)cudaGetLastError();
            {
                std::lock_guard<std::mutex> lock(registered_host_allocations_mutex);
                registered_host_allocations.erase(value);
            }
            ::operator delete(value,std::align_val_t(4096));
            value=nullptr;
        }
#endif
        check(hipHostMalloc(&value,size_t(bytes),hipHostMallocPortable|hipHostMallocMapped));
        *pointer=value;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Give back memory sd_host_alloc_on returned, once nothing reads it.
int sd_host_free(int ordinal,void* pointer) {
    try {
        if (!pointer) throw std::runtime_error("invalid host free");
        check(hipSetDevice(ordinal));
#if !defined(__HIPCC__)
        bool registered=false;
        {
            std::lock_guard<std::mutex> lock(registered_host_allocations_mutex);
            registered=registered_host_allocations.find(pointer)!=registered_host_allocations.end();
        }
        if (registered) {
            check(cudaHostUnregister(pointer));
            {
                std::lock_guard<std::mutex> lock(registered_host_allocations_mutex);
                registered_host_allocations.erase(pointer);
            }
            ::operator delete(pointer,std::align_val_t(4096));
            return 0;
        }
#endif
        check(hipHostFree(pointer));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Page-lock host memory that devices may also write (downloads into it).
int sd_host_pin(void* pointer,uint64_t bytes) {
    try {
        if (!pointer || !bytes) throw std::runtime_error("invalid host registration");
        check(hipHostRegister(pointer,size_t(bytes),hipHostRegisterPortable));
        std::lock_guard<std::mutex> lock(registered_mutex);
        registered[reinterpret_cast<uintptr_t>(pointer)]={size_t(bytes),true};
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_host_unregister(void* pointer) {
    try {
        check(hipHostUnregister(pointer));
        std::lock_guard<std::mutex> lock(registered_mutex);
        registered.erase(reinterpret_cast<uintptr_t>(pointer));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Development: the next `count` page-locked host allocations a request makes are refused (0: none; host_allocate).
int sd_test_refuse_host(uint64_t count) {
    refuse_host_allocations.store(count);
    return 0;
}
// Development (tests/stream_fault_gpu.rs): the `count`-th streamed chunk of a prompt batch's layers from now fails, once its GEMMs are queued and its
// experts' table entries point at its buffer (0: none). The answer is the count that was left of the one before (0: it fired, or none was asked).
int sd_test_fail_streamed(uint64_t count) {
    return int(std::min<uint64_t>(fail_streamed_chunk.exchange(count), 1 << 30));
}
// Development: the text a failed call with `code` carries when the runtime answers NULL (runtime_null != 0) or its own text; copied into
// `out` (NUL-terminated, cut to `capacity`); the length, or -1 for a bad buffer. Touches no device and no driver.
int sd_test_gpu_error_text(int code, int runtime_null, char* out, size_t capacity) {
    if (out == nullptr || capacity == 0) return -1;
    const hipError_t error = static_cast<hipError_t>(code);
    const std::string text(gpu_error_text_of(error, runtime_null != 0 ? nullptr : "the runtime's own words"));
    const size_t length = std::min(text.size(), capacity - 1);
    std::memcpy(out, text.data(), length);
    out[length] = '\0';
    return int(length);
}
int sd_test_hybrid_timeout(uint64_t count) {
    test_hybrid_timeouts.store(count);
    return 0;
}
int sd_hybrid_wait_check(void* handle) {
    try {
        auto& d = device(handle);
        if (d.exchange || !d.resident.empty() || !d.graphs.empty())
            throw std::runtime_error("the hybrid wait check needs a fresh device");
        create_expert_slots(d, 0, 2, 256, false);
        auto* counters = reinterpret_cast<volatile uint32_t*>(d.exchange + hybrid_sequence_slot);
        auto* gpu_counters = reinterpret_cast<volatile uint32_t*>(d.exchange_device + hybrid_sequence_slot);
        const int32_t needs_result = 1, no_result = 0;
        auto launch_wait = [&](uint64_t timeout, bool cpu_result = true) {
            check(hipMemcpyAsync(hybrid_need(d), cpu_result ? &needs_result : &no_result, sizeof(needs_result), hipMemcpyHostToDevice, d.stream));
            hybrid_await<<<1, 256, 0, d.stream>>>(gpu_counters, d.exchange_device, d.cpu_partial, timeout, hybrid_need(d), HYBRID_AWAITED(d),
                                               d.flags + hybrid_await_device_failed);
#if defined(__HIPCC__)
            hybrid_tick<<<1, 1, 0, d.stream>>>(d.flags + 1, gpu_counters, hybrid_clock(d), false, hybrid_need(d), nullptr, nullptr, 0,
                                             hybrid_plan_room(d) + hybrid_plan_histogram);
#else
            // No routes are needed for this handshake; the existing tick still advances the export counter.
            hybrid_tick<<<1, 256, 0, d.stream>>>(d.flags + 1, gpu_counters, hybrid_clock(d), false, hybrid_need(d), nullptr, nullptr, 0,
                                               hybrid_plan_room(d) + hybrid_plan_histogram, false, nullptr, nullptr, nullptr, 0);
#endif
            check(hipGetLastError());
        };
        auto awaited_count = [&]() -> uint32_t {
#if defined(__HIPCC__)
            return counters[hybrid_awaited_slot];
#else
            uint32_t count = 0;
            check(hipMemcpy(&count, hybrid_awaited(d), sizeof(count), hipMemcpyDeviceToHost));
            return count;
#endif
        };
        auto device_failed = [&]() -> uint32_t {
            uint32_t failed = 0;
            check(hipMemcpy(&failed, d.flags + hybrid_await_device_failed, sizeof(failed), hipMemcpyDeviceToHost));
            return failed;
        };
        if (device_failed()) throw std::runtime_error("the fresh hybrid device failure flag was not clear");
#if defined(__HIPCC__)
        constexpr uint32_t recovered_awaits = 3, completed_awaits = 4;
#else
        constexpr uint32_t recovered_awaits = 0, completed_awaits = 1;
#endif
        {
            HybridAwait await(d);
            await.launched = 2;
            d.hybrid_sequence = 1; // The host export count has not caught up to every issued wait.
            launch_wait(0);
            launch_wait(0);
            check(hipStreamSynchronize(d.stream));
            if (!device_failed() || !counters[hybrid_await_failed_slot])
                throw std::runtime_error("the expired hybrid wait did not latch both failure flags");
        }
        bool failed = false;
        try {
            // A new span clears the host flag, but must preserve and re-report the device failure.
            HybridAwait await(d);
            if (counters[hybrid_await_failed_slot] || !device_failed())
                throw std::runtime_error("the hybrid span did not preserve its device failure latch");
            await.launched = 1;
            await.publish();
            // Publication already satisfies the wait; even a no-CPU-result layer must re-report the device failure.
            launch_wait(d.await_timeout, false);
            check(hipStreamSynchronize(d.stream));
            if (!device_failed() || !counters[hybrid_await_failed_slot])
                throw std::runtime_error("the latched hybrid wait did not re-report its failure");
            await.check_failed();
        } catch (const HybridWaitTimeout&) {
            recover_hybrid_wait(handle);
            failed = last_error == "the GPU wait timed out; the interrupted pass was discarded";
        }
        if (!failed || d.hybrid_sequence != 3 || counters[0] != 3 ||
            counters[hybrid_published_slot] != recovered_awaits || awaited_count() != recovered_awaits || counters[hybrid_await_failed_slot] || device_failed())
            throw std::runtime_error("the expired hybrid wait was not recovered");
        {
            HybridAwait await(d);
            for (size_t i = 0; i < 2560; ++i) d.exchange[i] = 42.0f;
            await.launched = 1;
            await.publish();
            launch_wait(d.await_timeout);
            hybrid_wait_export(d, ++d.hybrid_sequence);
            check(hipStreamSynchronize(d.stream));
            await.check_failed();
        }
        std::vector<float> result(2560);
        check(hipMemcpy(result.data(), d.cpu_partial, result.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (d.hybrid_sequence != 4 || counters[0] != 4 ||
            counters[hybrid_published_slot] != completed_awaits || awaited_count() != completed_awaits || counters[hybrid_await_failed_slot] || device_failed() ||
            std::any_of(result.begin(), result.end(), [](float value) { return value != 42.0f; }))
            throw std::runtime_error("the hybrid wait after recovery failed");
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
// Development (tests/router_gpu.rs): a partially resident layer's router on the given weights, once as the separate kernels run it and then as
// the one kernel (hybrid_router) runs it, `repeats` launches of a captured graph with the buffers cleared between them. `router` is `experts` rows of
// 2560 values and `gate` one row, of kind 30 (BF16) or 0 (F32); `input` is `rows` rows of 2560 floats. Every result holds two copies, the separate
// kernels' and then the one kernel's: scores (rows x (experts + 1)), selected (rows x 10), probabilities (rows x 11) and exported (rows x
// hybrid_row_floats, as the host reads it). With `assist` more than 0 the layer's GPU assists that many absent routes (as sd_hybrid_assist_level does), where
// the export marks them. *fused is 0, and the second copy left alone, where the one kernel does not take these (HIP, other shapes).
int sd_hybrid_route_check(void* handle, const void* router, int router_kind, const void* gate, int gate_kind, uint64_t experts, uint64_t rows,
                          const float* input, uint32_t repeats, uint32_t assist, int* fused, float* scores, int32_t* selected,
                          float* probabilities, float* exported) {
    try {
        auto& d = device(handle);
        if (!router || !gate || !input || !fused || !scores || !selected || !probabilities || !exported || experts < 10 || experts > 512 ||
            !rows || rows > hybrid_rows || !repeats || (router_kind != 0 && router_kind != 30) || (gate_kind != 0 && gate_kind != 30))
            throw std::runtime_error("invalid hybrid router check");
        const size_t E = size_t(experts), R = size_t(rows), stride = 2564;
        std::vector<void*> buffers;
        struct Release {
            std::vector<void*>& all; Device& device; float* exchange; uint32_t assist;
            ~Release() { device.exchange_device = exchange; device.assist = assist; for (void* pointer : all) cleanup(hipFree(pointer)); }
        } release{buffers, d, d.exchange_device, d.assist};
        const auto allocate = [&](size_t bytes) { void* pointer = nullptr; check(hipMalloc(&pointer, bytes)); buffers.push_back(pointer); return pointer; };
        const size_t router_row = 2560 * (router_kind == 30 ? 2 : 4), gate_row = 2560 * (gate_kind == 30 ? 2 : 4);
        void* device_router = allocate(E * router_row); upload(d, device_router, router, E * router_row);
        void* device_gate = allocate(gate_row); upload(d, device_gate, gate, gate_row);
        std::vector<float> rows_in(R * stride, 0.0f);
        for (size_t r = 0; r < R; ++r) std::copy(input + r * 2560, input + (r + 1) * 2560, rows_in.begin() + r * stride);
        auto* device_input = static_cast<float*>(allocate(rows_in.size() * 4)); upload(d, device_input, rows_in.data(), rows_in.size() * 4);
        ResidentMoe plan;
        plan.experts = E; plan.hybrid = true;
        if (assist) {   // (the banks the GPU reads in place: any address, a byte each)
            for (size_t role = 0; role < 3; ++role) { plan.mapped[role] = static_cast<const uint8_t*>(device_router); plan.host_bytes[role] = 1; }
            d.assist = assist;
        }
        plan.weights.assign((E + 1) * 3 + 2, Matrix{});
        plan.weights[(E + 1) * 3] = {static_cast<const uint8_t*>(device_router), router_row, E, 2560, 0, 0, 0, router_kind};
        plan.weights[(E + 1) * 3 + 1] = {static_cast<const uint8_t*>(device_gate), gate_row, 1, 2560, 0, 0, 0, gate_kind};
        // (every third expert is resident: any address stands for its data)
        std::vector<MoeGroup> groups(E + 1, MoeGroup{});
        for (size_t e = 0; e < E; e += 3) groups[e].weights[0].data = static_cast<const uint8_t*>(device_router);
        plan.table = allocate(groups.size() * sizeof(MoeGroup)); upload(d, plan.table, groups.data(), groups.size() * sizeof(MoeGroup));
        const size_t workspace_bytes = R * 11 * 4500 * 4, exported_bytes = R * hybrid_row_floats * 4, output_bytes = R * (E + 1) * 4;
        plan.workspace = static_cast<float*>(allocate(workspace_bytes));
        auto* exchange = static_cast<float*>(allocate(exported_bytes));
        d.exchange_device = exchange;
        const HybridScratch s = hybrid_scratch(plan, R);
        const auto clear = [&]() {
            check(hipMemsetAsync(plan.workspace, 0, workspace_bytes, d.stream));
            check(hipMemsetAsync(exchange, 0, exported_bytes, d.stream));
            check(hipMemsetAsync(d.output, 0, output_bytes, d.stream));
        };
        const auto collect = [&](size_t copy) {
            download(d, scores + copy * R * (E + 1), d.output, output_bytes);
            download(d, selected + copy * R * 10, s.selected, R * 10 * 4);
            download(d, probabilities + copy * R * 11, s.probabilities, R * 11 * 4);
            download(d, exported + copy * R * hybrid_row_floats, exchange, exported_bytes);
        };
        clear();
        hybrid_route_export_separate(d, plan, device_input, R, stride);
        collect(0);
        *fused = 0;
#if !defined(__HIPCC__)
        clear();
        if (hybrid_router_launch(d, plan, s, device_input, R, stride)) {   // (the shapes it takes: once as ever, then as the graph's replays)
            static std::atomic<size_t> variants{0};
            const size_t variant = ++variants;
            for (uint32_t launch = 0; launch < repeats; ++launch) {
                clear();
                run_graph(d, "hybrid-router-check", nullptr, 0, variant, [&]() { hybrid_router_launch(d, plan, s, device_input, R, stride); });
            }
            collect(1);
            *fused = 1;
        }
#endif
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
extern "C++" {
#if !defined(__HIPCC__)
// What sd_bench_resident releases when it ends, however it ends: its device buffers and events.
struct ResidentBenchScope {
    Device& device;
    std::vector<void*> buffers; std::vector<hipEvent_t> events;
    explicit ResidentBenchScope(Device& d) : device(d) {}
    ResidentBenchScope(const ResidentBenchScope&) = delete;
    ResidentBenchScope& operator=(const ResidentBenchScope&) = delete;
    ~ResidentBenchScope() {
        cleanup(hipStreamSynchronize(device.stream));
        for (hipEvent_t event : events) cleanup(hipEventDestroy(event));
        for (void* pointer : buffers) cleanup(hipFree(pointer));
    }
};
// The value at `fraction` of the sorted samples (0.5 the median), 0 for none.
inline double resident_bench_percentile(std::vector<double> samples, double fraction) {
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    size_t at = size_t(std::ceil(fraction * double(samples.size())));
    at = at ? at - 1 : 0;
    return samples[std::min(at, samples.size() - 1)];
}
// Pseudo-random bytes (a hash of each word's index): the grid indices a decoder looks up are spread as a model's are, not one entry for every lane.
__global__ void resident_bench_fill(uint64_t* words, size_t count, uint64_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        uint64_t x = (uint64_t(i) + seed) * 0x9e3779b97f4a7c15ull;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27; x *= 0x94d049bb133111ebull;
        x ^= x >> 31;
        words[i] = x;
    }
}
// The shared expert's gate and up of a group's entry, pointed at another copy of their bytes.
__global__ void resident_bench_point(MoeGroup* entry, const uint8_t* gate, const uint8_t* up) {
    entry->weights[0].data = gate; entry->weights[1].data = up;
}
// A plain streaming read of `count` 16-byte words (the figure a GB/s of the expert kernel is read against); `sink` keeps the loads alive.
// Holds the stream for some tens of milliseconds, so that the launches recorded behind it wait in the queue and then run back to back: their
// events then time the GPU alone, not the pace at which the host issues them (19 us a launch on a slow rented host, which every figure had as a
// floor).
__global__ void resident_bench_hold(unsigned* sink, uint64_t turns) {
    volatile uint64_t at = 0;
    for (uint64_t i = 0; i < turns; ++i) at = at + 1;
    sink[0] = unsigned(at);
}
__global__ void resident_bench_read(const uint4* words, size_t count, unsigned* sink) {
    unsigned sum = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const uint4 word = words[i];
        sum ^= word.x ^ word.y ^ word.z ^ word.w;
    }
    if (sum == 0x9e3779b9u) atomicAdd(sink, 1u);
}
#endif
} // extern "C++"
// Development, one-off (crates/straydog-gpu/tests/resident_cost.rs): what the decode gate and up launch of a partially resident Qwen3.8 layer costs
// on CUDA, resident_expert_launch as hybrid_resident_experts calls it (generic resident_expert_gemv, the shared expert's waves in the launch), on
// a plan formed by hand as create_hybrid_moe forms it: `experts` routed groups of 640 x 2560 gate and up and a 2560 x 640 down, in the engine's slots
// (a slot holds an expert's gate, up and down one after another, rounded to 256 bytes; `slot_of[e]` is expert e's slot, or -1: absent, null data)
// and the shared expert's 640 x 2560 gate and up, on synthetic pseudo-random bytes. Routed gate and up of kind 18, 21 or 22 in a slot are in the
// capacity-preserving SoA layout (layout 1), as the engine's cache slots have it; any other kind stays native. Nothing here changes the engine's
// behaviour and nothing else calls it. `kinds` is gate, up and down of the routed experts, then the shared expert's gate, up and down (GGML kinds).
// `selected` holds `warmup + reps` reps of `rows` rows of 10 routes: the expert of each. The shared expert's bytes rotate through enough copies that
// no rep finds the last one's in the L2 (a table entry pointed at the rep's copy before its events).
//   mode 0: the launch as the engine makes it: the routed routes' waves and the shared expert's gate and up waves;
//   mode 1: routed only (the plan has no shared expert: its waves are not in the grid);
//   mode 2: the shared expert only (the plan's routed matrices have no rows: only its waves are in the grid; `selected` is read by no wave);
//   mode 3: empty: no shared expert, and every route absent (checked): the waves return at once, the launch's fixed cost;
//   mode 4: the device and the kernels (no plan): out[0..5] SMs, L2 bytes, threads, registers and shared bytes of an SM, compute capability; then
//           out[8..13] for resident_expert_gemv and out[16..21] for resident_expert_gemv_kind<22>: registers, static shared bytes, local bytes,
//           largest block, blocks an SM holds at 256 threads, constant bytes;
//   mode 5: a plain 512 MiB read kernel, for a reference: out[0] median and out[1] least microseconds, out[17] its bytes.
// `only` is ExpertGemv::only (0 all routes, 1 the resident ones alone, as the overlapped layer's launch beside the staging copy has it). Of modes 0
// to 3 `out` (32 doubles) takes: [0] median GPU microseconds of a launch (events around it), [1] least, [2] most, [3] 10th and [4] 90th percentile,
// [5] median microseconds the host spent issuing it, [6] blocks along x and [7] along y (rows), [8] waves, [9] and [10] the routed gate and up lanes,
// [11] and [12] the shared gate and up lanes, [13] routed waves of a route's gate, [14] shared waves, [15] resident and [16] absent routes of a launch
// (mean), [17] bytes of resident routed gate and up a launch reads, each row's route counted, [18] the same, a distinct expert once, [19] the shared
// expert's gate and up bytes (0 where the launch has none), [20] to [25] the six matrices' bytes, [26] slot bytes, [27] resident experts, [28] shared
// copies, [29] the slots' bytes, [30] 1 where the resident routed gate and up are SoA. The first `warmup` launches are left out of every figure.
int sd_bench_resident(void* handle, const int* kinds, uint32_t experts, const int32_t* slot_of, uint32_t rows, const int32_t* selected,
                      uint32_t mode, uint32_t only, uint32_t warmup, uint32_t reps, double* out) {
    try {
#if defined(__HIPCC__)
        (void)handle; (void)kinds; (void)experts; (void)slot_of; (void)rows; (void)selected; (void)mode; (void)only; (void)warmup; (void)reps; (void)out;
        throw std::runtime_error("the resident expert benchmark is CUDA's");
#else
        auto& d = device(handle);
        if (!out || mode > 5 || !reps) throw std::runtime_error("invalid resident expert benchmark");
        for (size_t i = 0; i < 32; ++i) out[i] = 0;
        using bench_clock = std::chrono::steady_clock;
        const auto since = [](bench_clock::time_point from) { return std::chrono::duration<double, std::micro>(bench_clock::now() - from).count(); };
        hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, d.ordinal));
        expert_q8_scratch_make(d);
        ResidentBenchScope scope(d);
        const auto allocate = [&](size_t bytes) {
            void* pointer = nullptr;
            check(hipMalloc(&pointer, bytes));
            scope.buffers.push_back(pointer);
            check(hipMemset(pointer, 0, bytes));
            return pointer;
        };
        // (`bytes` a multiple of 8)
        const auto fill = [&](void* pointer, size_t bytes, uint64_t seed) {
            const size_t words = bytes / 8;
            resident_bench_fill<<<unsigned(std::min<size_t>((words + 255) / 256, size_t(d.compute_units) * 16)), 256, 0, d.stream>>>(
                static_cast<uint64_t*>(pointer), words, seed);
            check(hipGetLastError());
        };
        const auto make_events = [&](size_t count) {
            std::vector<hipEvent_t> made(count);
            for (auto& event : made) { check(hipEventCreate(&event)); scope.events.push_back(event); }
            return made;
        };
        const auto elapsed = [](hipEvent_t from, hipEvent_t to) { float ms = 0; check(hipEventElapsedTime(&ms, from, to)); return double(ms) * 1000.0; };
        const size_t total = size_t(warmup) + reps;

        if (mode == 4) {
            out[0] = prop.multiProcessorCount; out[1] = double(prop.l2CacheSize); out[2] = prop.maxThreadsPerMultiProcessor;
            out[3] = prop.regsPerMultiprocessor; out[4] = double(prop.sharedMemPerMultiprocessor); out[5] = prop.major * 10 + prop.minor;
            const auto describe = [&](const void* kernel, size_t at) {
                hipFuncAttributes attributes{}; check(hipFuncGetAttributes(&attributes, kernel));
                int blocks = 0; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, 0));
                out[at] = attributes.numRegs; out[at + 1] = double(attributes.sharedSizeBytes); out[at + 2] = double(attributes.localSizeBytes);
                out[at + 3] = attributes.maxThreadsPerBlock; out[at + 4] = blocks; out[at + 5] = double(attributes.constSizeBytes);
            };
            describe(reinterpret_cast<const void*>(&resident_expert_gemv), 8);
            describe(reinterpret_cast<const void*>(&resident_expert_gemv_kind<22>), 16);
            return 0;
        }
        if (mode == 5) {
            const size_t bytes = size_t(512) << 20;
            void* buffer = allocate(bytes + 256);
            fill(buffer, bytes, 7);
            auto* sink = static_cast<unsigned*>(allocate(256));
            int per_sm = 0;
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, reinterpret_cast<const void*>(&resident_bench_read), 256, 0));
            const unsigned blocks = unsigned(std::max(1, per_sm) * prop.multiProcessorCount);
            const auto start = make_events(total), stop = make_events(total);
            check(hipStreamSynchronize(d.stream));
            for (size_t rep = 0; rep < total; ++rep) {
                check(hipEventRecord(start[rep], d.stream));
                resident_bench_read<<<blocks, 256, 0, d.stream>>>(static_cast<const uint4*>(buffer), bytes / 16, sink);
                check(hipGetLastError());
                check(hipEventRecord(stop[rep], d.stream));
            }
            check(hipStreamSynchronize(d.stream));
            std::vector<double> times;
            for (size_t rep = warmup; rep < total; ++rep) times.push_back(elapsed(start[rep], stop[rep]));
            out[0] = resident_bench_percentile(times, 0.5); out[1] = *std::min_element(times.begin(), times.end()); out[17] = double(bytes);
            return 0;
        }

        // Modes 0 to 3.
        if (!kinds || !slot_of || !selected || experts < 10 || experts > 512 || rows < 1 || rows > 4 || only > 1)
            throw std::runtime_error("invalid resident expert benchmark");
        size_t resident = 0;
        for (size_t e = 0; e < experts; ++e) resident += slot_of[e] >= 0;
        {
            std::vector<char> taken(resident, 0);
            for (size_t e = 0; e < experts; ++e) {
                if (slot_of[e] < 0) continue;
                if (size_t(slot_of[e]) >= resident || taken[size_t(slot_of[e])]) throw std::runtime_error("the resident expert benchmark's slots are not one to an expert");
                taken[size_t(slot_of[e])] = 1;
            }
        }
        // (a route that names no expert would read beyond the table; mode 3 is the launch of routes that are all absent)
        for (size_t i = 0; i < total * rows * 10; ++i) {
            if (selected[i] < 0 || size_t(selected[i]) >= experts) throw std::runtime_error("the resident expert benchmark's route names no expert");
            if (mode == 3 && slot_of[selected[i]] >= 0) throw std::runtime_error("the resident expert benchmark's empty launch has a resident route");
        }
        const size_t shape_rows[6] = {640, 640, 2560, 640, 640, 2560}, shape_cols[6] = {2560, 2560, 640, 2560, 2560, 640};
        size_t row_bytes[6], bytes[6];
        for (size_t m = 0; m < 6; ++m) {
            row_bytes[m] = matrix_row_bytes(kinds[m], shape_cols[m]);
            bytes[m] = shape_rows[m] * row_bytes[m];
        }
        const size_t slot_bytes = (bytes[0] + bytes[1] + bytes[2] + 255) / 256 * 256;
        // (the engine's rule, slot_soa_plan: gate and up of one SoA-eligible kind)
        const bool soa = kinds[0] == kinds[1] && expert_soa_eligible(kinds[0], 640, 2560) && expert_soa_eligible(kinds[1], 640, 2560);
        // (the slots as the engine's arena: `resident` of them one after another, and a tail for the aligned word loads of a last block)
        const size_t arena_bytes = std::max<size_t>(resident, 1) * slot_bytes + 256;
        auto* arena = static_cast<uint8_t*>(allocate(arena_bytes));
        fill(arena, arena_bytes, 1);
        // (the shared expert's gate and up, copy after copy: enough to be twice the L2 and one over)
        const size_t shared_gate_at = 0, shared_up_at = (bytes[3] + 255) / 256 * 256, shared_pair = shared_up_at + (bytes[4] + 255) / 256 * 256;
        const size_t copies = std::min<size_t>(std::max<size_t>(2 * size_t(prop.l2CacheSize) / shared_pair + 1, 2), 64);
        auto* shared = static_cast<uint8_t*>(allocate(copies * shared_pair + 256));
        fill(shared, copies * shared_pair + 256, 2);

        Matrix routed_matrix[3]{}, shared_matrix[3]{};
        for (size_t role = 0; role < 3; ++role) {
            routed_matrix[role] = {nullptr, row_bytes[role], shape_rows[role], shape_cols[role], 0, 0, 0, kinds[role]};
            shared_matrix[role] = {nullptr, row_bytes[3 + role], shape_rows[3 + role], shape_cols[3 + role], 0, 0, 0, kinds[3 + role]};
        }
        shared_matrix[0].data = shared + shared_gate_at; shared_matrix[1].data = shared + shared_up_at;
        // (the shared expert's down, index 2, has no data: a gate and up launch does not read it)
        std::vector<MoeGroup> groups(size_t(experts) + 1);
        for (size_t e = 0; e < experts; ++e) {
            size_t offset = 0;
            for (size_t role = 0; role < 3; ++role) {
                Matrix matrix = routed_matrix[role];
                if (slot_of[e] >= 0) {
                    matrix.data = arena + size_t(slot_of[e]) * slot_bytes + offset;
                    matrix.layout = soa && role < 2 ? 1 : 0;
                }
                groups[e].weights[role] = matrix;
                offset += bytes[role];
            }
        }
        for (size_t role = 0; role < 3; ++role) groups[experts].weights[role] = shared_matrix[role];
        // The plan's host side (create_hybrid_moe): a routed matrix of each expert and role, then the shared expert's three and two more.
        ResidentMoe plan;
        plan.experts = experts; plan.hybrid = true; plan.shared = mode == 0 || mode == 2;
        plan.weights.assign(size_t(experts) * 3 + 5, Matrix{});
        for (size_t e = 0; e < experts; ++e)
            for (size_t role = 0; role < 3; ++role) {
                plan.weights[e * 3 + role] = groups[e].weights[role];
                if (mode == 2) plan.weights[e * 3 + role].rows = 0;   // (no routed waves in the grid)
            }
        for (size_t role = 0; role < 3; ++role) plan.weights[size_t(experts) * 3 + role] = shared_matrix[role];
        auto* table = static_cast<MoeGroup*>(allocate(groups.size() * sizeof(MoeGroup)));
        check(hipMemcpy(table, groups.data(), groups.size() * sizeof(MoeGroup), hipMemcpyHostToDevice));
        plan.table = table;
        auto* input = static_cast<float*>(allocate(size_t(rows) * 2564 * 4));
        {
            std::vector<float> values(size_t(rows) * 2564);
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i * 7 % 13) - 6) / 16.0f;
            check(hipMemcpy(input, values.data(), values.size() * 4, hipMemcpyHostToDevice));
        }
        auto* gate_up = static_cast<float*>(allocate(size_t(rows) * 11 * 1280 * 4));
        auto* routes = static_cast<int32_t*>(allocate(total * rows * 10 * 4));
        check(hipMemcpy(routes, selected, total * rows * 10 * 4, hipMemcpyHostToDevice));
        check(hipStreamSynchronize(d.stream));

        // The launch's geometry, by the arithmetic resident_expert_launch does it with on CUDA (the generic kernel: 20 concurrent matrices).
        size_t waves = 0;
        unsigned lanes[2], shared_lanes[2];
        size_t routed_waves[2], shared_waves = 0;
        for (size_t r = 0; r < 2; ++r) {
            lanes[r] = resident_lanes(plan.weights, r, input, 20);
            routed_waves[r] = gemv_waves(plan.weights[r], lanes[r]);
            const Matrix& pair = plan.weights[size_t(experts) * 3 + r];
            shared_lanes[r] = gemv_lanes(pair, input, 20);
            const size_t own = plan.shared ? gemv_waves(pair, shared_lanes[r]) : 0;
            shared_waves += own;
            waves += 10 * routed_waves[r] + own;
        }
        const double grid_x = double((waves + 7) / 8);

        const auto start = make_events(total), stop = make_events(total);
        std::vector<double> issue;
        auto* held = static_cast<unsigned*>(allocate(256));
        check(hipStreamSynchronize(d.stream));
        resident_bench_hold<<<1, 1, 0, d.stream>>>(held, 20000000ull);
        check(hipGetLastError());
        for (size_t rep = 0; rep < total; ++rep) {
            const size_t copy = rep % copies;
            resident_bench_point<<<1, 1, 0, d.stream>>>(table + experts, shared + copy * shared_pair + shared_gate_at, shared + copy * shared_pair + shared_up_at);
            check(hipGetLastError());
            const auto begin = bench_clock::now();
            check(hipEventRecord(start[rep], d.stream));
            resident_expert_launch(d, plan, table, routes + rep * rows * 10, false, input, gate_up, rows, 2564, 11 * 1280, false, nullptr, 0, nullptr, only);
            check(hipEventRecord(stop[rep], d.stream));
            if (rep >= warmup) issue.push_back(since(begin));
        }
        check(hipStreamSynchronize(d.stream));
        std::vector<double> times;
        for (size_t rep = warmup; rep < total; ++rep) times.push_back(elapsed(start[rep], stop[rep]));
        // The bytes a launch reads of resident routed gate and up, each row's routes and then each distinct expert once.
        double resident_routes = 0, absent_routes = 0, route_bytes = 0, unique_bytes = 0;
        for (size_t rep = warmup; rep < total; ++rep) {
            std::set<int32_t> distinct;
            for (size_t row = 0; row < rows; ++row)
                for (size_t route = 0; route < 10; ++route) {
                    const int32_t expert = selected[(rep * rows + row) * 10 + route];
                    if (slot_of[expert] < 0) { absent_routes += 1; continue; }
                    resident_routes += 1; route_bytes += double(bytes[0] + bytes[1]); distinct.insert(expert);
                }
            unique_bytes += double(distinct.size()) * double(bytes[0] + bytes[1]);
        }
        // With the INT8 routed kernel on (STRAYDOG_EXPERT_Q8): the last rep's routes once more with every row reading alone, with the rows
        // sharing, and on the F32 kernel; out[31] takes the floats of resident routed routes that differ between the first two, bit for bit
        // (it must be 0), and a line says how far the INT8 sums are from the F32 ones.
        if (expert_q8_level() && mode <= 1) {
            const size_t floats = size_t(rows) * 11 * 1280;
            std::vector<float> alone(floats), together(floats), exact(floats);
            const int32_t* last = routes + (total - 1) * rows * 10;
            const auto run = [&](int level, std::vector<float>& to) {
                expert_q8_forced = level;
                check(hipMemset(gate_up, 0, floats * 4));
                resident_expert_launch(d, plan, table, last, false, input, gate_up, rows, 2564, 11 * 1280, false, nullptr, 0, nullptr, only);
                check(hipStreamSynchronize(d.stream));
                check(hipMemcpy(to.data(), gate_up, floats * 4, hipMemcpyDeviceToHost));
            };
            run(2, alone); run(1, together); run(0, exact);
            expert_q8_forced = -1;
            // (The synthetic bytes make a block's scale any half, one in 32 of them an infinity or a NaN: a sum that is not finite in either
            // kernel is left out of the error and counted, and the error is the median of |INT8 - F32| / |F32| over the rest, which the few
            // sums of huge scales do not decide as they would an rms.)
            double differing = 0, compared = 0, unfinite = 0;
            std::vector<double> errors;
            for (size_t row = 0; row < rows; ++row)
                for (size_t route = 0; route < 10; ++route) {
                    if (slot_of[selected[((total - 1) * rows + row) * 10 + route]] < 0) continue;
                    for (size_t i = 0; i < 1280; ++i) {
                        const size_t at = (row * 11 + route) * 1280 + i;
                        differing += std::memcmp(&alone[at], &together[at], 4) != 0;
                        compared += 1;
                        if (!std::isfinite(together[at]) || !std::isfinite(exact[at]) || exact[at] == 0.0f) { unfinite += 1; continue; }
                        errors.push_back(std::fabs(double(together[at]) - exact[at]) / std::fabs(double(exact[at])));
                    }
                }
            out[31] = differing;
            std::sort(errors.begin(), errors.end());
            const auto at_share = [&](double share) { return errors.empty() ? 0.0 : errors[std::min(errors.size() - 1, size_t(share * double(errors.size())))]; };
            std::fprintf(stderr, "resident bench check: %u rows, %.0f floats of resident routed routes; sharing against alone: %.0f differ; INT8 against F32: "
                         "|difference| / |F32 sum| median %.5f, 90th percentile %.5f, 99th %.5f over %zu finite sums (%.0f not finite or zero)\n", rows,
                         compared, differing, at_share(0.5), at_share(0.9), at_share(0.99), errors.size(), unfinite);
        }
        const double launches = double(reps);
        out[0] = resident_bench_percentile(times, 0.5); out[1] = *std::min_element(times.begin(), times.end()); out[2] = *std::max_element(times.begin(), times.end());
        out[3] = resident_bench_percentile(times, 0.1); out[4] = resident_bench_percentile(times, 0.9); out[5] = resident_bench_percentile(issue, 0.5);
        out[6] = grid_x; out[7] = double(rows); out[8] = double(waves);
        out[9] = lanes[0]; out[10] = lanes[1]; out[11] = shared_lanes[0]; out[12] = shared_lanes[1];
        out[13] = double(routed_waves[0]); out[14] = double(shared_waves);
        out[15] = resident_routes / launches; out[16] = absent_routes / launches; out[17] = route_bytes / launches; out[18] = unique_bytes / launches;
        out[19] = plan.shared ? double(bytes[3] + bytes[4]) : 0;
        for (size_t m = 0; m < 6; ++m) out[20 + m] = double(bytes[m]);
        out[26] = double(slot_bytes); out[27] = double(resident); out[28] = double(copies); out[29] = double(arena_bytes); out[30] = soa ? 1 : 0;
        return 0;
#endif
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int sd_transfer(void* handle,int stream,void* pointer,uint64_t offset,void* host,uint64_t bytes) {
    try {
        auto& d=device(handle);
        if (!host || !bytes || offset>UINT64_MAX-bytes || stream<1 || stream>3)
            throw std::runtime_error("invalid Straydog transfer");
        auto* at=static_cast<uint8_t*>(pointer)+offset;
        op_span(d,at,size_t(bytes),"a transferred range");
        if (stream!=2) written(d,pointer);
        check_registered(host,size_t(bytes),stream==2);
        auto& t=transfers(d);
        if (stream==2) check(hipMemcpyAsync(host,at,size_t(bytes),hipMemcpyDeviceToHost,t.streams[1]));
        else check(hipMemcpyAsync(at,host,size_t(bytes),hipMemcpyHostToDevice,t.streams[stream==1?0:2]));
        const int kind=stream==2?1:0;   // counts: uploads (both upload streams), downloads
        ++t.copies[kind]; t.bytes[kind]+=bytes;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int64_t sd_event_record(void* handle,int stream) {
    try {
        auto& d=device(handle);
        auto& t=transfers(d);
        const hipStream_t on=transfer_stream_of(d,stream);
        t.reap();
        hipEvent_t event=nullptr;
        if (t.spare.empty()) check(hipEventCreateWithFlags(&event,hipEventDisableTiming));
        else { event=t.spare.back(); t.spare.pop_back(); }
        try { check(hipEventRecord(event,on)); }
        catch (...) { t.spare.push_back(event); throw; }
        t.pending[stream].emplace_back(t.next,event);
        return t.next++;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_event_wait(void* handle,int stream,int64_t id) {
    try {
        auto& d=device(handle);
        auto& t=transfers(d);
        const hipStream_t on=transfer_stream_of(d,stream);
        if (id<1 || id>=t.next) throw std::runtime_error("Straydog transfer: no such event");
        if (hipEvent_t* event=t.find(id)) check(hipStreamWaitEvent(on,*event,0));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_event_done(void* handle,int64_t id,int block) {
    try {
        auto& d=device(handle);
        auto& t=transfers(d);
        if (id<1 || id>=t.next) throw std::runtime_error("Straydog transfer: no such event");
        if (block)
            if (hipEvent_t* event=t.find(id)) check(hipEventSynchronize(*event));
        t.reap();
        return t.find(id) ? 0 : 1;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_transfer_counts(void* handle,uint64_t* counts) {
    try {
        if (!counts) throw std::runtime_error("invalid transfer counters");
        auto& d=device(handle);
        auto& t=transfers(d);
        counts[0]=t.copies[0]; counts[1]=t.bytes[0]; counts[2]=t.copies[1]; counts[3]=t.bytes[1];
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_execution_counts(void* handle,uint64_t* graphs,uint64_t* replays,uint64_t* resident) {
    try {
        if (!graphs || !replays || !resident) throw std::runtime_error("invalid execution counters");
        auto& d=device(handle); *graphs=d.graphs.size(); *replays=d.graph_replays; *resident=d.resident.size();
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_qsa_sequence(void* handle, void* const* pointers, const int* kinds, void* kv, void* index,
                    void* scratch, uint64_t context, uint64_t position, uint64_t steps,
                    const float* input, float* output) {
    try {
        sparse_attention(device(handle),pointers,kinds,kv,index,scratch,size_t(context),size_t(position),size_t(steps),input,output);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_moe(void* handle, void* const* pointers, const int* kinds, uint64_t experts, void* workspace,
            uint64_t steps, const int32_t* selected, const float* probabilities,
            const float* input, float* output) {
    try {
        mixture_of_experts(device(handle),pointers,kinds,size_t(experts),workspace,size_t(steps),
                            selected,probabilities,input,output);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}

int sd_gdn_sequence(void* handle, void* const* pointers, const int* kinds, void* conv, void* recurrent,
                    uint64_t steps, const float* input, float* output) {
    try {
        gated_delta(device(handle), pointers, kinds, conv, recurrent, size_t(steps), input, output);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_hyperconnection(void* handle, void* const* pointers, const int* kinds,
                        uint64_t steps, int inject, const float* input, float* output) {
    try {
        if (inject != 0 && inject != 1) throw std::runtime_error("invalid HyperConnection injection flag");
        hyperconnection(device(handle), pointers, kinds, size_t(steps), inject != 0, input, output);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

const char* sd_error() { return last_error.c_str(); }

int sd_device_count() {
    try { int count = 0; check(hipGetDeviceCount(&count)); return count; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_prime_qwen_prefill_gemm(int ordinal, uint64_t prompt_tokens) {
    try {
        if (prompt_tokens < 2 || prompt_tokens > 4096) throw std::runtime_error("invalid prefill GEMM batch");
        check(hipSetDevice(ordinal));
        struct Resources {
            hipStream_t stream = nullptr;
            hipblasHandle_t blas = nullptr;
            void* weights = nullptr;
            void* inputs = nullptr;
            void* outputs = nullptr;
            void* workspace = nullptr;
            ~Resources() {
                if (stream) (void)hipStreamSynchronize(stream);
                if (blas) (void)hipblasDestroy(blas);
                if (stream) (void)hipStreamDestroy(stream);
                for (void* allocation : {weights, inputs, outputs, workspace}) {
                    if (allocation) (void)hipFree(allocation);
                }
            }
        } prime;
        constexpr int rows = 320, width = 10240;
        const int tokens = int(prompt_tokens);
        check(hipStreamCreate(&prime.stream));
        check(hipMalloc(&prime.weights, size_t(rows) * width * 2));
        check(hipMalloc(&prime.inputs, size_t(tokens) * width * 2));
        check(hipMalloc(&prime.outputs, size_t(rows) * tokens * 4));
        check(hipMalloc(&prime.workspace, blas_workspace_bytes));
        check(hipMemset(prime.weights, 0, size_t(rows) * width * 2));
        check(hipMemset(prime.inputs, 0, size_t(tokens) * width * 2));
        blas_check(hipblasCreate(&prime.blas));
        blas_check(hipblasSetStream(prime.blas, prime.stream));
        blas_check(hipblasSetWorkspace(prime.blas, prime.workspace, blas_workspace_bytes));
        blas_check(hipblasSetMathMode(prime.blas, HIPBLAS_DEFAULT_MATH));
        const float alpha = 1, beta = 0;
        blas_check(hipblasGemmEx(prime.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, rows, tokens, width, &alpha,
            prime.weights, HIP_R_16F, width, prime.inputs, HIP_R_16F, width, &beta,
            prime.outputs, HIP_R_32F, rows, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        check(hipStreamSynchronize(prime.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

#if defined(_WIN32) && !defined(__HIPCC__)
// The memory free on the whole GPU, as NVML has it; false where NVML cannot say. CUDA's own figure on Windows leaves out most of what
// the desktop and other programs hold: RTX 5070 Ti, 14.68 GiB against NVML's 13.07, so `auto`'s margin came to 0.6 GiB, and at that
// WDDM moves memory in and out under a kernel: the GPU at 100% and 65 W, the desktop stalled, a start 170 s in its warm-up.
static bool nvml_free_bytes(const hipDeviceProp_t& prop, uint64_t& free_bytes) {
    struct Memory { unsigned long long total, free, used; };
    using Init = int (*)();
    using Handle = int (*)(const char*, void**);
    using Info = int (*)(void*, Memory*);
    static Handle handle = nullptr;
    static Info info = nullptr;
    static const bool ready = []() {
        const HMODULE library = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library) return false;
        const auto init = reinterpret_cast<Init>(GetProcAddress(library, "nvmlInit_v2"));
        handle = reinterpret_cast<Handle>(GetProcAddress(library, "nvmlDeviceGetHandleByPciBusId_v2"));
        info = reinterpret_cast<Info>(GetProcAddress(library, "nvmlDeviceGetMemoryInfo"));
        return init && handle && info && init() == 0;
    }();
    if (!ready) return false;
    char address[32];
    std::snprintf(address, sizeof(address), "%08x:%02x:%02x.0", unsigned(prop.pciDomainID), unsigned(prop.pciBusID), unsigned(prop.pciDeviceID));
    void* device = nullptr;
    Memory memory{};
    if (handle(address, &device) != 0 || info(device, &memory) != 0) return false;
    free_bytes = memory.free;
    return true;
}

struct DxgiMemoryInfo {
    HRESULT status = DXGI_ERROR_UNSUPPORTED;
    const char* reason = "unavailable";
    UINT node = 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
};

// Match CUDA's adapter and physical node before asking for this process's LOCAL budget.
static DxgiMemoryInfo dxgi_memory_info(const hipDeviceProp_t& prop) {
    DxgiMemoryInfo result;
    if (prop.tccDriver) { result.reason = "tcc"; return result; }
    const unsigned mask = prop.luidDeviceNodeMask;
    if (!mask || (mask & (mask - 1))) { result.reason = "node_mask"; return result; }
    if (std::all_of(prop.luid, prop.luid + sizeof(prop.luid), [](char byte) { return byte == 0; })) {
        result.reason = "luid_unavailable";
        return result;
    }
    while (!(mask & (1u << result.node))) ++result.node;
    using CreateFactory = HRESULT (WINAPI*)(REFIID, void**);
    static const CreateFactory create = []() {
        const HMODULE library = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        return library ? reinterpret_cast<CreateFactory>(GetProcAddress(library, "CreateDXGIFactory1")) : nullptr;
    }();
    if (!create) { result.reason = "factory_unavailable"; return result; }
    IDXGIFactory1* factory = nullptr;
    result.status = create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
    if (FAILED(result.status)) { result.reason = "create_factory"; return result; }
    result.reason = "adapter_not_found";
    for (UINT index = 0; ; ++index) {
        IDXGIAdapter1* adapter = nullptr;
        result.status = factory->EnumAdapters1(index, &adapter);
        if (FAILED(result.status)) break;
        DXGI_ADAPTER_DESC1 description{};
        const HRESULT described = adapter->GetDesc1(&description);
        if (SUCCEEDED(described) && std::memcmp(&description.AdapterLuid, prop.luid, sizeof(LUID)) == 0) {
            IDXGIAdapter3* adapter3 = nullptr;
            result.status = adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3));
            result.reason = "adapter3";
            if (SUCCEEDED(result.status)) {
                result.status = adapter3->QueryVideoMemoryInfo(result.node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &result.memory);
                result.reason = SUCCEEDED(result.status) ? "ok" : "query_memory";
                adapter3->Release();
            }
            adapter->Release();
            break;
        }
        adapter->Release();
    }
    factory->Release();
    return result;
}
#endif
// A build holds code only for the GPU architectures it was compiled for. The device is current.
static void require_code_for(int ordinal) {
    hipFuncAttributes kernel{};
    const hipError_t probe = hipFuncGetAttributes(&kernel, reinterpret_cast<const void*>(&cast_half));
    #if !defined(__HIPCC__)
    // Only missing code names the architecture: a context faulted by an earlier illegal
    // access fails every call and is reported as such.
    if (probe != cudaErrorInvalidDeviceFunction && probe != cudaErrorNoKernelImageForDevice) check(probe);
    #endif
    if (probe != hipSuccess) {
        (void)hipGetLastError();
        hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, ordinal));
        #if defined(__HIPCC__)
        const std::string arch = prop.gcnArchName;
        #else
        const std::string arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
        #endif
        throw std::runtime_error("this Straydog build has no GPU code for " + arch + "; build it for this GPU");
    }
}

int sd_device_info(int ordinal, char* name, size_t capacity, uint64_t* free_bytes, uint64_t* total_bytes) {
    try {
        if (!name || capacity < 2 || !free_bytes || !total_bytes) throw std::runtime_error("invalid info buffers");
        check(hipSetDevice(ordinal));
        require_code_for(ordinal);
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, ordinal));
        std::strncpy(name, prop.name, capacity - 1); name[capacity - 1] = 0;
        size_t available, total; check(hipMemGetInfo(&available, &total));
        *free_bytes = available; *total_bytes = total;
#if defined(_WIN32) && !defined(__HIPCC__)
        uint64_t measured = 0;
        if (nvml_free_bytes(prop, measured)) *free_bytes = std::min<uint64_t>(available, measured);
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_device_memory_diagnostic(int ordinal, int ready) {
    try {
#if defined(_WIN32) && !defined(__HIPCC__)
        check(hipSetDevice(ordinal));
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, ordinal));
        size_t raw_free = 0, raw_total = 0;
        const hipError_t runtime_status = hipMemGetInfo(&raw_free, &raw_total);
        char name[256]{};
        uint64_t reported_free = 0, reported_total = 0;
        const int reported_status = sd_device_info(ordinal, name, sizeof(name), &reported_free, &reported_total);
        const DxgiMemoryInfo dxgi = dxgi_memory_info(prop);
        char runtime_values[96], reported_values[96], dxgi_values[256];
        if (runtime_status == hipSuccess)
            std::snprintf(runtime_values, sizeof(runtime_values), "%llu,\"cuda_total_bytes\":%llu",
                          static_cast<unsigned long long>(raw_free), static_cast<unsigned long long>(raw_total));
        else std::strcpy(runtime_values, "null,\"cuda_total_bytes\":null");
        if (reported_status == 0)
            std::snprintf(reported_values, sizeof(reported_values), "%llu,\"device_info_total_bytes\":%llu",
                          static_cast<unsigned long long>(reported_free), static_cast<unsigned long long>(reported_total));
        else std::strcpy(reported_values, "null,\"device_info_total_bytes\":null");
        if (SUCCEEDED(dxgi.status))
            std::snprintf(dxgi_values, sizeof(dxgi_values), "%llu,\"dxgi_current_usage_bytes\":%llu,"
                          "\"dxgi_available_for_reservation_bytes\":%llu,\"dxgi_current_reservation_bytes\":%llu",
                          static_cast<unsigned long long>(dxgi.memory.Budget), static_cast<unsigned long long>(dxgi.memory.CurrentUsage),
                          static_cast<unsigned long long>(dxgi.memory.AvailableForReservation),
                          static_cast<unsigned long long>(dxgi.memory.CurrentReservation));
        else std::strcpy(dxgi_values, "null,\"dxgi_current_usage_bytes\":null,"
                         "\"dxgi_available_for_reservation_bytes\":null,\"dxgi_current_reservation_bytes\":null");
        LUID luid{};
        std::memcpy(&luid, prop.luid, sizeof(luid));
        std::fprintf(stderr, "Straydog VRAM diagnostic: {\"stage\":\"%s\",\"pid\":%lu,\"gpu\":%d,"
                     "\"cuda_status\":%d,\"cuda_free_bytes\":%s,\"device_info_status\":%d,"
                     "\"device_info_free_bytes\":%s,\"dxgi_status\":\"0x%08lx\",\"dxgi_reason\":\"%s\","
                     "\"dxgi_budget_bytes\":%s,\"luid\":\"%08lx%08lx\",\"node_mask\":%u,\"node\":%u}\n",
                     ready ? "ready" : "startup", GetCurrentProcessId(), ordinal, int(runtime_status), runtime_values,
                     reported_status, reported_values, static_cast<unsigned long>(dxgi.status), dxgi.reason, dxgi_values,
                     static_cast<unsigned long>(luid.HighPart), static_cast<unsigned long>(luid.LowPart),
                     prop.luidDeviceNodeMask, dxgi.node);
#else
        (void)ordinal; (void)ready;
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_device_headroom(int ordinal, int64_t* bytes, int* process_budget) {
    try {
        if (!bytes || !process_budget) throw std::runtime_error("invalid memory headroom buffers");
        char name[256]{};
        uint64_t free = 0, total = 0;
        if (sd_device_info(ordinal, name, sizeof(name), &free, &total) < 0) return -1;
        const uint64_t most = uint64_t(std::numeric_limits<int64_t>::max());
        *bytes = int64_t(std::min(free, most));
        *process_budget = 0;
#if defined(_WIN32) && !defined(__HIPCC__)
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, ordinal));
        const DxgiMemoryInfo dxgi = dxgi_memory_info(prop);
        if (SUCCEEDED(dxgi.status)) {
            const uint64_t budget = dxgi.memory.Budget, used = dxgi.memory.CurrentUsage;
            const int64_t headroom = budget >= used ? int64_t(std::min(budget - used, most))
                                                    : -int64_t(std::min(used - budget, most));
            *bytes = std::min(*bytes, headroom);
            *process_budget = 1;
        }
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_host_register_read_only_supported(int ordinal) {
    try {
#if defined(__HIPCC__)
        (void)ordinal;
        throw std::runtime_error("read-only host registration support query requires CUDA");
#else
        int supported=0;
        check(cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterReadOnlySupported,ordinal));
        return supported!=0;
#endif
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}

int sd_device_pci(int ordinal, int* domain, int* bus, int* device) {
    try {
        if (!domain || !bus || !device) throw std::runtime_error("invalid PCI buffers");
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, ordinal));
        *domain = prop.pciDomainID; *bus = prop.pciBusID; *device = prop.pciDeviceID;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

void* sd_create(int ordinal, uint64_t budget) {
    try {
        auto value = std::make_unique<Device>(ordinal);
        check(hipSetDevice(ordinal));
        require_code_for(ordinal);
        size_t grid_bytes[6]{};
        const uint8_t* grids[6];
        const int grid_kinds[6] = {16, 17, 22, 18, 21, 29};
        for (size_t i = 0; i < 6; ++i) grids[i] = sd_iq_grid(grid_kinds[i], &grid_bytes[i]);
        if (grids[0]) {
            check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid2xxs), grids[0], grid_bytes[0]));
            check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid2xs), grids[1], grid_bytes[1]));
            check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid2s), grids[2], grid_bytes[2]));
            check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid3xxs), grids[3], grid_bytes[3]));
            check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid3s), grids[4], grid_bytes[4]));
            check(hipMemcpyToSymbol(HIP_SYMBOL(iq_grid1s), grids[5], grid_bytes[5]));
            iq_uploaded = true;
        }
        {
            hipDeviceProp_t units{}; check(hipGetDeviceProperties(&units, ordinal));
            value->compute_units = unsigned(units.multiProcessorCount);
        }
        #if defined(__HIPCC__)
        hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop,ordinal));
        value->rdna4=std::strncmp(prop.gcnArchName,"gfx120",6)==0;
        // (in WGP mode, HIP's default on RDNA, multiProcessorCount counts workgroup processors: two CUs each)
        if (value->rdna4) value->compute_units *= 2;
        value->expert_prefill=value->rdna4;
        value->graph_capture=value->rdna4;
        if (value->rdna4) {
            int wall_clock_khz=0;
            check(hipDeviceGetAttribute(&wall_clock_khz,hipDeviceAttributeWallClockRate,ordinal));
            if (wall_clock_khz<=0) throw std::runtime_error("invalid HIP wall clock rate");
            // KFD compute queues have no kernel run-time limit.
            value->await_timeout=uint64_t(wall_clock_khz)*1000*10;   // wall_clock64 ticks in ten seconds
            value->hybrid_prelaunch=true;
        }
        #else
        hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop,ordinal));
        value->expert_prefill=prop.major>=7;
        value->graph_capture=true;
        value->hybrid_prelaunch=prop.major>=7;   // __nanosleep
        // (CUDA 13 no longer says whether kernels have a run-time limit: the shorter wait is the safe one)
        #if CUDART_VERSION >= 13000
        value->await_watchdog=true;
        #else
        value->await_watchdog=prop.kernelExecTimeoutEnabled!=0;
        #endif
        value->await_timeout=value->await_watchdog ? 1000000000ull : 10000000000ull;   // %globaltimer: ns
        #endif
        size_t available, total; check(hipMemGetInfo(&available, &total));
        if (budget >= 256 * 1024 * 1024) {
            value->batch_capacity = max_batch;
            value->input_capacity = max_batch * max_cols;
            value->output_capacity = max_rows * 32;
            if (budget>=2ull*1024*1024*1024) {
                value->input_capacity*=4;
                value->output_capacity*=2;
            }
        }
        const size_t workspace_bytes = (value->input_capacity + value->output_capacity) * 4
                                     + value->batch_capacity * sizeof(GemvItem);
        if (budget < workspace_bytes + 256 || budget > available || budget > SIZE_MAX)
            throw std::runtime_error("Straydog VRAM budget is unavailable");
        value->budget = size_t(budget);
        check(hipStreamCreate(&value->stream));
        // The peer's owner waits on this device's help: it runs ahead of the device's own work.
        int least = 0, greatest = 0;
        check(hipDeviceGetStreamPriorityRange(&least, &greatest));
        check(hipStreamCreateWithPriority(&value->helper_stream, hipStreamDefault, greatest));
        check(hipHostMalloc(reinterpret_cast<void**>(&value->io), io_arena_bytes, hipHostMallocDefault));
        for (auto& event : value->io_halves) check(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        check(hipEventCreateWithFlags(&value->handoff, hipEventDisableTiming));
        check(hipMalloc(reinterpret_cast<void**>(&value->input), value->input_capacity * 4));
        check(hipMalloc(reinterpret_cast<void**>(&value->output), value->output_capacity * 4));
        check(hipMalloc(reinterpret_cast<void**>(&value->matrices), value->batch_capacity * sizeof(GemvItem)));
        check(hipMalloc(reinterpret_cast<void**>(&value->flags), 256));
        check(hipMemset(value->flags, 0, 256));   // (hybrid_clock sums and the hybrid wait failure latch start at zero)
        value->used = workspace_bytes + 256;
        if (budget >= 256 * 1024 * 1024) {
            check(hipMalloc(reinterpret_cast<void**>(&value->expanded), gemm_elements * 4));
            value->expanded_bytes = gemm_elements * 4;
            check(hipMalloc(reinterpret_cast<void**>(&value->half_input), value->input_capacity * 2));
            check(hipMalloc(&value->blas_workspace, blas_workspace_bytes));
            check(hipMalloc(reinterpret_cast<void**>(&value->attention_scores), attention_score_bytes));
            value->used += gemm_elements * 4 + blas_workspace_bytes + value->input_capacity * 2 + attention_score_bytes;
            blas_check(hipblasCreate(&value->blas));
            value->blas_f32 = create_f32_blas();
            // Both handles run on the device's stream, so they share its workspace.
            for (hipblasHandle_t handle : {value->blas, value->blas_f32}) {
                blas_check(hipblasSetStream(handle, value->stream));
                blas_check(hipblasSetWorkspace(handle, value->blas_workspace, blas_workspace_bytes));
                blas_check(hipblasSetMathMode(handle, HIPBLAS_DEFAULT_MATH));
            }
        }
        return value.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

void sd_destroy(void* handle) { delete static_cast<Device*>(handle); }

// A budget raised while serving: GPU memory that other programs freed since the start, under `auto`. The expert slots take it when
// they are placed again (sd_expert_slots_release, sd_expert_slots_restore). Never below what is allocated.
int sd_set_budget(void* handle, uint64_t budget) {
    try {
        auto& value = device(handle);
        if (budget < value.used || budget > SIZE_MAX) throw std::runtime_error("invalid Straydog VRAM budget");
        value.budget = size_t(budget);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
int sd_memory(void* handle, uint64_t* used, uint64_t* budget) {
    try {
        auto& value = device(handle);
        if (!used || !budget) throw std::runtime_error("invalid memory counters");
        *used = value.used; *budget = value.budget;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

void* sd_upload(void* handle, const void* data, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!data) throw std::runtime_error("invalid Straydog upload");
        void* pointer = allocate(value, bytes);
        upload(value, pointer, data, size_t(bytes));
        check(hipStreamSynchronize(value.stream));
        value.uploaded.emplace(static_cast<const uint8_t*>(pointer), size_t(bytes));
        return pointer;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

int sd_expansions(void* handle, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (value.expansions) throw std::runtime_error("Straydog expansions are kept already");
        if (bytes) value.expansions = static_cast<uint8_t*>(allocate(value, bytes));
        value.expansions_bytes = size_t(bytes);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

void* sd_allocate(void* handle, uint64_t bytes) {
    try { return allocate(device(handle), bytes); }
    catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

int sd_write(void* handle, void* pointer, const void* data, uint64_t bytes) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (!data || found == value.allocations.end() || found->second != bytes)
            throw std::runtime_error("invalid Straydog write");
        written(value, pointer);
        upload(value, pointer, data, size_t(bytes));
        check(hipStreamSynchronize(value.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

void* sd_filler_create(void* handle) {
    try {
        auto filler = std::make_unique<Filler>();
        filler->device = &device(handle);
        check(hipSetDevice(filler->device->ordinal));
        check(hipEventCreateWithFlags(&filler->done, hipEventDisableTiming | hipEventBlockingSync));
        return filler.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

void sd_filler_destroy(void* filler) { delete static_cast<Filler*>(filler); }

int sd_fill(void* filler, void* pointer, uint64_t offset, const void* host, uint64_t bytes) {
    try {
        if (!filler) throw std::runtime_error("invalid Straydog fill");
        auto& copier = *static_cast<Filler*>(filler);
        const auto& value = *copier.device;
        // Lookups only: fills run while nothing else changes the device's allocations.
        auto found = value.allocations.find(pointer);
        if (!host || !bytes || found == value.allocations.end() || offset > found->second ||
            bytes > found->second - offset)
            throw std::runtime_error("invalid Straydog fill");
        check(hipSetDevice(value.ordinal));
        check(hipMemcpyAsync(static_cast<uint8_t*>(pointer) + offset, host, size_t(bytes), hipMemcpyHostToDevice,
                             value.stream));
        check(hipEventRecord(copier.done, value.stream));
        check(hipEventSynchronize(copier.done));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_fill_rows(void* filler, void* pointer, uint64_t offset, uint64_t pitch, const void* host, uint64_t host_pitch, uint64_t width,
                 uint64_t height) {
    try {
        if (!filler) throw std::runtime_error("invalid Straydog fill");
        auto& copier = *static_cast<Filler*>(filler);
        const auto& value = *copier.device;
        auto found = value.allocations.find(pointer);
        if (!host || !width || !height || width > pitch || width > host_pitch || found == value.allocations.end() ||
            offset > found->second || (height - 1) * pitch + width > found->second - offset)
            throw std::runtime_error("invalid Straydog fill");
        check(hipSetDevice(value.ordinal));
        check(hipMemcpy2DAsync(static_cast<uint8_t*>(pointer) + offset, size_t(pitch), host, size_t(host_pitch), size_t(width),
                               size_t(height), hipMemcpyHostToDevice, value.stream));
        check(hipEventRecord(copier.done, value.stream));
        check(hipEventSynchronize(copier.done));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// The device bytes sd_release would give back for `pointer` (its size and the allocation tail), without releasing it.
int sd_allocation_bytes(void* handle, const void* pointer, uint64_t* bytes) {
    try {
        auto& value = device(handle);
        if (!bytes) throw std::runtime_error("invalid allocation bytes");
        auto found = value.allocations.find(const_cast<void*>(pointer));
        if (found == value.allocations.end()) throw std::runtime_error("unknown Straydog allocation");
        *bytes = uint64_t(found->second + allocation_tail);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_release(void* handle, void* pointer) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (found == value.allocations.end()) throw std::runtime_error("unknown Straydog allocation");
        if (pointer == value.layer_prefill_storage) throw std::runtime_error("cannot release an active layer prefill bank");
        written(value, pointer);
        if (value.kv_prefetching && (pointer == value.kv_stage || pointer == value.kv_stage_owner)) {
            check(hipStreamSynchronize(value.kv_stream));
            value.kv_prefetching = false;
        }
        for (const auto& entry : value.resident) {
            const auto& plan=*entry.second;
            if (plan.table==pointer || plan.workspace==pointer ||
                std::find(plan.dependencies.begin(),plan.dependencies.end(),pointer)!=plan.dependencies.end())
                throw std::runtime_error("cannot release a resident Straydog dependency");
        }
        for (const auto& entry : value.decoder) {
            const auto& dependencies=entry.second.dependencies;
            if (std::find(dependencies.begin(),dependencies.end(),pointer)!=dependencies.end())
                throw std::runtime_error("cannot release a Straydog decoder dependency");
        }
        for (auto graph=value.graphs.begin();graph!=value.graphs.end();) {
            const auto& dependencies=graph->second.dependencies;
            if (std::find(dependencies.begin(),dependencies.end(),pointer)!=dependencies.end()) {
                check(hipGraphExecDestroy(graph->second.executable));
                check(hipGraphDestroy(graph->second.graph));
                graph=value.graphs.erase(graph);
            } else ++graph;
        }
        if (value.kv_stage_owner == pointer) { value.kv_stage_owner = nullptr; value.kv_stage_valid = 0; }
        value.kv_mirrors.erase(pointer);
        if (value.kv_stage == pointer) {
            kv_clear_graphs(value);
            value.kv_stage = nullptr; value.kv_stage_floats = 0;
            value.kv_stage_owner = nullptr; value.kv_stage_valid = 0;
            value.kv_mirrors.clear(); value.kv_mirror_rows = 0;
        }
        const auto host = value.host_states.find(pointer);
        if (host != value.host_states.end()) {
            check(hipHostFree(host->second)); value.host_states.erase(host);
        } else {
            check(hipFree(pointer)); value.used -= found->second + allocation_tail;
        }
        value.expert_layouts.erase(static_cast<const uint8_t*>(pointer));
        value.allocations.erase(found);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Zeroed sequence state in mapped, page-locked host memory that the kernels read and write
// over the bus (the long-context K/V that does not fit the VRAM budget). It is owned like a
// device allocation but not counted in the VRAM budget.
// Page-lock and zero host states of these sizes on a thread of their own; sd_host_state then
// takes them (in any order) once the thread is done.
int sd_host_states_prefetch(void* handle, const uint64_t* sizes, uint64_t count) {
    try {
        auto& value = device(handle);
        if (!sizes || value.host_prefetch.joinable() || !value.host_ready.empty())
            throw std::runtime_error("invalid Straydog host state prefetch");
        std::vector<size_t> wanted(sizes, sizes + count);
        const int ordinal = value.ordinal;
        value.host_prefetch_error.clear();
        value.host_prefetch = std::thread([&value, wanted, ordinal]() {
            try {
                check(hipSetDevice(ordinal));
                for (size_t bytes : wanted) {
                    void* host = nullptr;
                    check(hipHostMalloc(&host, bytes + allocation_tail, hipHostMallocMapped));
                    value.host_ready.emplace_back(bytes, host);
                    std::memset(host, 0, bytes + allocation_tail);
                }
            } catch (const std::exception& e) { value.host_prefetch_error = e.what(); }
        });
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Page-lock the row stagings of batches of `floats` floats (two buffers for each of the MTP layer's, the first layer's and the PLE layer's rows) on a
// thread of their own, for the first stage_rows to take: a read of wide batches would otherwise pin them in its first batch (0.2-0.5 s, 8192 rows). Where a
// buffer is refused, none is kept, and the stagings grow when a batch asks, as they do without this.
int sd_row_staging_prefetch(void* handle, uint64_t floats) {
    try {
        auto& value = device(handle);
        if (!floats || value.staging_prefetch.joinable() || value.staging_ready_floats)
            throw std::runtime_error("invalid Straydog row staging prefetch");
        const int ordinal = value.ordinal;
        value.staging_prefetch = std::thread([&value, floats, ordinal]() {
            float* ready[3][2] = {};
            bool pinned = hipSetDevice(ordinal) == hipSuccess;
            for (auto& pair : ready)
                for (float*& host : pair)
                    pinned = pinned && host_allocate(reinterpret_cast<void**>(&host), size_t(floats) * 4, hipHostMallocDefault) == hipSuccess;
            if (!pinned) {
                (void)hipGetLastError();
                for (auto& pair : ready) for (float* host : pair) if (host) cleanup(hipHostFree(host));
                return;
            }
            std::memcpy(value.staging_ready, ready, sizeof(ready));
            value.staging_ready_floats = size_t(floats);
        });
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

void* sd_host_state(void* handle, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!bytes) throw std::runtime_error("invalid Straydog host state");
        if (value.host_prefetch.joinable()) value.host_prefetch.join();
        if (!value.host_prefetch_error.empty()) {
            const std::string error = value.host_prefetch_error;
            value.host_prefetch_error.clear();
            throw std::runtime_error(error);
        }
        void* host = nullptr;
        auto prefetched = std::find_if(value.host_ready.begin(), value.host_ready.end(),
                                       [&](const auto& ready) { return ready.first == size_t(bytes); });
        const bool zeroed = prefetched != value.host_ready.end();   // the prefetch zeroed it
        if (zeroed) { host = prefetched->second; value.host_ready.erase(prefetched); }
        else check(hipHostMalloc(&host, size_t(bytes) + allocation_tail, hipHostMallocMapped));
        void* pointer = nullptr;
        try {
            if (!zeroed) std::memset(host, 0, size_t(bytes) + allocation_tail);
            check(hipHostGetDevicePointer(&pointer, host, 0));
            value.host_states.emplace(pointer, host);
            try { value.allocations.emplace(pointer, size_t(bytes)); }
            catch (...) { value.host_states.erase(pointer); throw; }
        } catch (...) { cleanup(hipHostFree(host)); throw; }
        return pointer;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

// A page-locked host buffer for one conversation checkpoint, freed with the device.
void* sd_checkpoint_buffer(void* handle, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!bytes) throw std::runtime_error("invalid Straydog checkpoint buffer");
        void* host = nullptr;
        check(hipHostMalloc(&host, size_t(bytes), hipHostMallocDefault));
        try { value.checkpoints.emplace(host, size_t(bytes)); }
        catch (...) { cleanup(hipHostFree(host)); throw; }
        return host;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

// Copy `count` whole device state allocations into (save) or out of a checkpoint buffer, packed
// in order, with one stream synchronization. Restored states invalidate their K/V mirrors.
int sd_checkpoint_copy(void* handle, int save, void* const* states, uint64_t count, void* buffer) {
    try {
        auto& value = device(handle);
        const auto found = value.checkpoints.find(buffer);
        if (found == value.checkpoints.end() || !states || !count || count > 4096)
            throw std::runtime_error("invalid Straydog checkpoint copy");
        size_t total = 0;
        for (uint64_t i = 0; i < count; ++i) {
            const auto state = value.allocations.find(states[i]);
            if (state == value.allocations.end() || value.host_states.count(states[i]) ||
                state->second > found->second - total)
                throw std::runtime_error("invalid Straydog checkpoint state");
            total += state->second;
        }
        auto* host = static_cast<uint8_t*>(buffer);
        for (uint64_t i = 0; i < count; ++i) {
            const size_t bytes = value.allocations.find(states[i])->second;
            if (save) {
                check(hipMemcpyAsync(host, states[i], bytes, hipMemcpyDeviceToHost, value.stream));
            } else {
                kv_invalidate_write(value, states[i], 0);
                check(hipMemcpyAsync(states[i], host, bytes, hipMemcpyHostToDevice, value.stream));
            }
            host += bytes;
        }
        check(hipStreamSynchronize(value.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Part of a sequence state to (save) or from host memory: `height` runs of `width` floats,
// `pitch` floats apart, from float `offset` (a K/V state's positions are one run, an indexer's
// pooled blocks one run per column). A restore truncates the K/V mirrors and stage at the first
// position it rewrites (`first`).
int sd_state_copy(void* handle, void* state, uint64_t offset, uint64_t pitch, uint64_t width, uint64_t height,
                  float* host, uint64_t first, int save) {
    try {
        auto& value = device(handle);
        const auto found = value.allocations.find(state);
        if (found == value.allocations.end() || !host || !width || !height || pitch < width)
            throw std::runtime_error("invalid Straydog state copy");
        const uint64_t floats = found->second / 4;
        if (offset > floats || height - 1 > (floats - offset) / pitch || width > floats - offset - (height - 1) * pitch)
            throw std::runtime_error("Straydog state copy outside its state");
        float* base = static_cast<float*>(state) + offset;
        kv_stage_settle(value);
        // A K/V state in host memory saved to host memory is copied by the CPU, on the copier's threads: the runtime pins the pageable
        // destination for a copy it makes into it, and a conversation of 94000 positions (1.66 GB over two devices) took 0.79 s that way, 2.1 GB/s.
        // Nothing is in flight once the stream is idle, and the bytes are the same ones.
        // (RDNA4 only, where both ways below were measured: any other device keeps the runtime's copy, as it was.)
        const auto host_state = value.host_states.find(state);
        if (save && value.rdna4 && host_state != value.host_states.end() && !value.allocations.count(host)) {
            check(hipStreamSynchronize(value.stream));
            const auto* source = static_cast<const uint8_t*>(host_state->second) + offset * 4;
            auto* target = reinterpret_cast<uint8_t*>(host);
            for (uint64_t run = 0; run < height; ++run)
                StagingCopier::instance().copy(target + run * width * 4, source + run * pitch * 4, size_t(width) * 4);
            return 0;
        }
        // Device memory saved to host memory goes through the page-locked arena, as large downloads do (8-12 GB/s), a run of rows at a time, where a row
        // fits it (the indexer's blocks of 94000 positions, 0.3 GB over two devices: 0.03 s of the 0.23 s that were left with the K/V's copy alone).
        if (save && value.rdna4 && !value.allocations.count(host) && width * 4 <= io_arena_bytes) {
            const size_t row_bytes = size_t(width) * 4, rows_per_run = io_arena_bytes / row_bytes;
            check(hipStreamSynchronize(value.stream));   // (the staged uploads have landed: the whole arena is free)
            for (uint64_t run = 0; run < height; run += rows_per_run) {
                const size_t rows = size_t(std::min<uint64_t>(rows_per_run, height - run));
                check(hipMemcpy2DAsync(value.io, row_bytes, base + run * pitch, pitch * 4, row_bytes, rows, hipMemcpyDeviceToHost, value.stream));
                check(hipStreamSynchronize(value.stream));
                StagingCopier::instance().copy(reinterpret_cast<uint8_t*>(host) + run * row_bytes, value.io, rows * row_bytes);
            }
            value.io_used = 0;
            return 0;
        }
        if (!save) kv_invalidate_write(value, state, size_t(first));
        if (save)
            check(hipMemcpy2DAsync(host, width * 4, base, pitch * 4, width * 4, height, hipMemcpyDefault, value.stream));
        else
            check(hipMemcpy2DAsync(base, pitch * 4, host, width * 4, width * 4, height, hipMemcpyDefault, value.stream));
        check(hipStreamSynchronize(value.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// A decoder layer's K/V grown to `capacity` positions, wherever it is held: in mapped host memory (the planes copied by the CPU, `valid`
// positions of them) or in VRAM (a new allocation of the device, the planes' first `from` positions copied on the device and the old one
// released, so the VRAM a growth takes is the new K/V and, for the length of one layer's copy, the old). Its indexer follows at the new pitch.
int sd_kv_grow(void* handle, uint64_t layer, uint64_t capacity, uint64_t valid, void** state, void** index_state) {
    try {
        auto& d = device(handle);
        auto found = d.decoder.find(size_t(layer));
        if (found == d.decoder.end() || decoder_linear(size_t(layer)) || !state || !index_state) throw std::runtime_error("invalid Straydog K/V growth");
        DecoderLayer& plan = found->second;
        const size_t from = plan.context, to = size_t(capacity);
        auto host = d.host_states.find(plan.states[0]);
        const bool on_host = host != d.host_states.end();
        auto kv = d.allocations.find(plan.states[0]);
        auto index = d.allocations.find(plan.states[1]);
        const size_t old_pool = from / 4 + 1, pool = to / 4 + 1, history = qsa_history * 128, kv_bytes = to * 4 * d.kv_row;
        if (kv == d.allocations.end() || index == d.allocations.end() || plan.parts != 1 || to <= from || to > 262144 || valid > from
            || (!on_host && kv->second != from * 4 * d.kv_row))
            throw std::runtime_error("invalid Straydog K/V growth");
        check(hipStreamSynchronize(d.stream));
        if (d.kv_prefetching) { check(hipStreamSynchronize(d.kv_stream)); d.kv_prefetching = false; }
        void* grown = nullptr;
        void* pointer = nullptr;
        void* grown_index = nullptr;
        void* grown_scratch = nullptr;
        // What a growth that fails gives back: nothing is handed to a plan yet, every state stays as it was.
        const auto discard_new = [&]() {
            if (grown && on_host) cleanup(hipHostFree(grown));
            for (void* made : {on_host ? nullptr : grown, grown_index, grown_scratch}) {
                const auto allocation = made ? d.allocations.find(made) : d.allocations.end();
                if (allocation == d.allocations.end()) continue;
                d.used -= allocation->second + allocation_tail;
                cleanup(hipFree(made));
                d.allocations.erase(allocation);
            }
        };
        // VRAM the budget or the device cannot give: the request fails and the engine stays (see kv_no_vram).
        const auto vram = [&](size_t bytes) {
            try { return allocate(d, bytes); }
            catch (const std::exception& e) {
                (void)hipGetLastError();
                std::fprintf(stderr, "Straydog: no VRAM for %zu MiB of K/V growth (%s)\n", bytes >> 20, e.what());
                throw std::runtime_error(kv_no_vram);
            }
        };
        // No memory leaves every state as it was: the request fails, the engine stays (see kv_no_memory); the engine may give page locks
        // back and ask again (Model::ensure_kv): a layer grown by an earlier call is left as it is.
        if (on_host && host_allocate(&grown, kv_bytes + allocation_tail, hipHostMallocMapped) != hipSuccess) {
            (void)hipGetLastError();
            throw std::runtime_error(kv_no_memory);
        }
        try {
            if (on_host) {
                std::memset(grown, 0, kv_bytes + allocation_tail);
                for (size_t plane = 0; plane < 4; ++plane)
                    std::memcpy(static_cast<kv_t*>(grown) + plane * to * d.kv_row, static_cast<const kv_t*>(host->second) + plane * from * d.kv_row,
                                valid * d.kv_row);
                check(hipHostGetDevicePointer(&pointer, grown, 0));
            } else {
                // Every position the old K/V has, not only the `valid` ones: what a later read finds is then what it would have found in a K/V
                // that had the whole context from the start (the rows the sequence has not written are never read, whatever they hold).
                pointer = grown = vram(kv_bytes);
                check(hipMemsetAsync(grown, 0, kv_bytes, d.stream));
                for (size_t plane = 0; plane < 4; ++plane)
                    check(hipMemcpyAsync(static_cast<kv_t*>(grown) + plane * to * d.kv_row, static_cast<const kv_t*>(plan.states[0]) + plane * from * d.kv_row,
                                         from * d.kv_row, hipMemcpyDeviceToDevice, d.stream));
                check(hipStreamSynchronize(d.stream));
            }
            // The indexer's blocks at their new pitch, then its rolling history: all of it through host memory, in its allocation.
            std::vector<float> before(old_pool * 128 + history), after(pool * 128 + history, 0.0f);
            check(hipMemcpy(before.data(), plan.states[1], before.size() * 4, hipMemcpyDeviceToHost));
            for (size_t column = 0; column < 128; ++column)
                std::copy_n(before.data() + column * old_pool, std::min(old_pool, valid / 4 + 1), after.data() + column * pool);
            std::copy_n(before.data() + 128 * old_pool, history, after.data() + 128 * pool);
            if (index->second < after.size() * 4) grown_index = vram(after.size() * 4);
            check(hipMemcpy(grown_index ? grown_index : plan.states[1], after.data(), after.size() * 4, hipMemcpyHostToDevice));
            // The layers' one scratch, once: it holds nothing between uses.
            if (plan.states[2] && d.allocations.at(plan.states[2]) < 256 * pool * 4) grown_scratch = vram(256 * pool * 4);
        } catch (const std::bad_alloc&) {
            // (the indexer's blocks go through host memory: a few tens of MB a layer; the growth is refused as when the K/V found none)
            discard_new();
            throw std::runtime_error(kv_no_memory);
        } catch (...) { discard_new(); throw; }
        // Every captured graph holds the old K/V or the old pitches.
        kv_clear_graphs(d);
        void* old_kv = plan.states[0];
        if (on_host) {
            check(hipHostFree(host->second));
            d.host_states.erase(host); d.allocations.erase(old_kv); d.kv_mirrors.erase(old_kv);
            d.host_states.emplace(pointer, grown); d.allocations.emplace(pointer, kv_bytes);
        }
        std::replace(plan.dependencies.begin(), plan.dependencies.end(), old_kv, pointer);
        plan.states[0] = pointer; plan.context = to;
        if (!on_host && sd_release(handle, old_kv)) throw std::runtime_error(last_error);
        if (grown_index) {
            void* old_index = plan.states[1];
            std::replace(plan.dependencies.begin(), plan.dependencies.end(), old_index, grown_index);
            plan.states[1] = grown_index;
            if (sd_release(handle, old_index)) throw std::runtime_error(last_error);
        }
        if (grown_scratch) {
            void* old_scratch = plan.states[2];
            for (auto& entry : d.decoder) {
                if (entry.second.states[2] != old_scratch) continue;
                std::replace(entry.second.dependencies.begin(), entry.second.dependencies.end(), old_scratch, grown_scratch);
                entry.second.states[2] = grown_scratch;
            }
            if (sd_release(handle, old_scratch)) throw std::runtime_error(last_error);
        }
        // (a DeltaNet layer holds no positions: its context is only what spans are checked against)
        for (auto& entry : d.decoder) if (decoder_linear(entry.first)) entry.second.context = std::max(entry.second.context, to);
        kv_layout(d);
        *state = pointer;
        *index_state = plan.states[1];
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// The scratch the K/V layers share, as sd_kv_grow left it.
int sd_kv_scratch(void* handle, void** scratch) {
    try {
        auto& d = device(handle);
        if (!scratch) throw std::runtime_error("invalid Straydog K/V scratch request");
        *scratch = nullptr;
        for (const auto& entry : d.decoder) if (entry.second.states[2]) *scratch = entry.second.states[2];
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// A long prompt's batches: the experts' and the decoder's workspaces of `capacity` rows in place of what they were, every plan following.
// Between requests' work: the caller has given the slots' VRAM up (release_expert_slots, restore at 0) and takes it back after.
// 1: the device refused these, and the plans have workspaces of the size they had.
int sd_prefill_buffers(void* handle, uint64_t capacity, uint64_t moe_bytes, uint64_t decoder_bytes, void** moe, void** decoder) {
    try {
        auto& d = device(handle);
        if (!capacity || capacity > 4096 || !moe_bytes || !decoder_bytes || !moe || !decoder) throw std::runtime_error("invalid Straydog prefill buffers");
        check(hipStreamSynchronize(d.stream));
        if (d.kv_prefetching) { check(hipStreamSynchronize(d.kv_stream)); d.kv_prefetching = false; }
        kv_clear_graphs(d);
        void* old_moe = nullptr; void* old_decoder = nullptr;
        for (const auto& entry : d.resident) {
            if (!entry.second->hybrid) throw std::runtime_error("prefill buffers are for partially resident models");
            old_moe = entry.second->workspace;
        }
        for (const auto& entry : d.decoder) old_decoder = entry.second.workspace;
        if (!old_moe || !old_decoder) throw std::runtime_error("invalid Straydog prefill buffers");
        const auto forget = [](std::vector<void*>& list, void* pointer) { list.erase(std::remove(list.begin(), list.end(), pointer), list.end()); };
        for (auto& entry : d.resident) { forget(entry.second->dependencies, old_moe); entry.second->workspace = nullptr; }
        for (auto& entry : d.decoder) { forget(entry.second.dependencies, old_moe); forget(entry.second.dependencies, old_decoder); entry.second.workspace = nullptr; }
        const size_t old_capacity = d.resident.begin()->second->capacity;
        const size_t old_moe_bytes = d.allocations.at(old_moe), old_decoder_bytes = d.allocations.at(old_decoder);
        if (sd_release(handle, old_moe) || sd_release(handle, old_decoder)) throw std::runtime_error(last_error);
        void* new_moe = nullptr;
        void* new_decoder = nullptr;
        bool refused = false;
        try {
            new_moe = allocate(d, moe_bytes);
            new_decoder = allocate(d, decoder_bytes);
        } catch (const std::exception&) {
            refused = true;
            (void)hipGetLastError();
            if (new_moe && sd_release(handle, new_moe)) throw std::runtime_error(last_error);
            new_moe = allocate(d, old_moe_bytes);
            new_decoder = allocate(d, old_decoder_bytes);
            capacity = old_capacity;
        }
        for (auto& entry : d.resident) {
            entry.second->workspace = static_cast<float*>(new_moe); entry.second->capacity = size_t(capacity);
            entry.second->dependencies.push_back(new_moe);
        }
        for (auto& entry : d.decoder) {
            entry.second.workspace = static_cast<float*>(new_decoder); entry.second.capacity = size_t(capacity);
            entry.second.dependencies.push_back(new_decoder);
            entry.second.dependencies.push_back(new_moe);
        }
        *moe = new_moe; *decoder = new_decoder;
        return refused ? 1 : 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// The format of the K/V: the bytes of its rows, 512 (FP16) or 272 (Q8_0). Before any decoder layer registers.
int sd_kv_format(void* handle, uint32_t row) {
    try {
        auto& value = device(handle);
        if ((row != kv_row_f16 && row != kv_row_q8) || !value.decoder.empty() || !value.host_states.empty())
            throw std::runtime_error("invalid Straydog K/V format");
        value.kv_row = row;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Register the existing VRAM stage: one layer for prefill, all host-state prefixes for decode.
int sd_kv_stage(void* handle, void* pointer, uint64_t rows) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (!rows || rows > (uint64_t(1) << 22) || found == value.allocations.end() || found->second != size_t(rows) * 1024 * 4 ||
            value.host_states.count(pointer))
            throw std::runtime_error("invalid Straydog K/V stage");
        if (value.kv_prefetching) { check(hipStreamSynchronize(value.kv_stream)); value.kv_prefetching = false; }
        kv_clear_graphs(value);
        value.kv_stage = static_cast<float*>(pointer); value.kv_stage_floats = size_t(rows) * 1024;
        kv_layout(value);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Mapped host K/V runs the same kernels bitwise on CUDA and on RDNA4 (HIP).
int sd_host_state_supported(void* handle) {
    try {
        (void)device(handle);
        return 1;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_gemm_options(void* handle, uint64_t minimum_batch, int half_precision) {
    try {
        auto& value = device(handle);
        if (minimum_batch < 2 || minimum_batch > max_batch + 1 || (half_precision != 0 && half_precision != 1))
            throw std::runtime_error("invalid Straydog GEMM options");
        value.gemm_min_batch = size_t(minimum_batch); value.half_gemm = half_precision != 0;
        // A matrix expanded to FP16 takes half of what it does in FP32, and the rest is a token's worth of slots on a small card. Only there:
        // DeepSeek-V4.1 turns FP16 on and off between passes, and a single op may ask for FP32 (op_linear's precision), on a larger card.
        const bool small_card = value.budget < size_t(8) << 30;
        const size_t expanded_bytes = gemm_elements * (value.half_gemm && small_card ? 2 : 4);
        if (value.expanded && value.expanded_bytes != expanded_bytes) {
            check(hipStreamSynchronize(value.stream));
            check(hipFree(value.expanded));
            value.expanded = nullptr;
            value.used -= value.expanded_bytes;
            value.expanded_bytes = 0;
            check(hipMalloc(reinterpret_cast<void**>(&value.expanded), expanded_bytes));
            value.expanded_bytes = expanded_bytes;
            value.used += expanded_bytes;
        }
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_decode_options(void* handle, int flags) {
    try {
        auto& value = device(handle);
        if (flags & ~7) throw std::runtime_error("invalid Straydog decode options");
        value.q8_decode = (flags & 1) && value.rdna4;
#if defined(__HIPCC__)
        value.q8_wide = (flags & 4) && value.rdna4;
#else
        // (dp4a, sm_61 and later: every architecture this is built for)
        value.q8_wide = (flags & 4) != 0;
#endif
        value.half_decode = (flags & 2) && value.rdna4;
        expert_q8_scratch_make(value);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_prompt_options(void* handle, int flags) {
    try {
        auto& value = device(handle);
        if (flags & ~3) throw std::runtime_error("invalid Straydog prompt options");
#if defined(__HIPCC__)
        value.gdn_chunked = (flags & 1) && value.rdna4;
#else
        value.gdn_chunked = (flags & 1) != 0;   // (CUDA: gdn_chunk_scan_f32)
#endif
        value.hc_fused = (flags & 2) && value.rdna4;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_gemm_counts(void* handle, uint64_t* calls, uint64_t* vectors) {
    try {
        auto& value = device(handle);
        if (!calls || !vectors) throw std::runtime_error("invalid GEMM counters");
        *calls = value.gemm_calls; *vectors = value.gemm_vectors;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_zero(void* handle, void* pointer) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (found == value.allocations.end()) throw std::runtime_error("unknown Straydog state allocation");
        kv_invalidate_write(value,pointer,0);
        check(hipMemsetAsync(pointer, 0, found->second, value.stream));
        check(hipStreamSynchronize(value.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_download(void* handle, void* pointer, void* output, uint64_t bytes) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (!output || found == value.allocations.end() || found->second != bytes)
            throw std::runtime_error("Straydog state readback size mismatch");
        download(value, output, pointer, size_t(bytes));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_attention_sequence(void* handle, void* state, uint64_t context, uint64_t base, uint64_t steps,
                          uint64_t selected, const float* input, uint64_t input_count, float* output, uint64_t output_count) {
    try {
        auto& value = device(handle);
        if (!input || !output || !value.attention_scores || !context || context > 262144 ||
            !steps || steps > max_batch || base > context || steps > context - base || !selected || selected > 2052)
            throw std::runtime_error("invalid Straydog attention dimensions");
        auto found = value.allocations.find(state);
        if (found == value.allocations.end() || found->second != context * 4 * value.kv_row)
            throw std::runtime_error("Straydog attention state size mismatch");
        if (input_count != steps * (24 * 256 + 1024 + selected + 1) || output_count != steps * 24 * 256 ||
            input_count > value.input_capacity || output_count > value.output_capacity)
            throw std::runtime_error("Straydog attention workspace mismatch");
        // Counts are host-readable metadata; validate before any kernel reads them.
        const size_t count_offset = size_t(steps * (24 * 256 + 1024 + selected));
        for (size_t step = 0; step < steps; ++step) {
            const float count = input[count_offset + step];
            if (!(count >= 1 && count <= float(selected)) || count != float(size_t(count)))
                throw std::runtime_error("invalid Straydog attention selection count");
        }
        upload(value, value.input, input, size_t(input_count) * 4);
        kv_invalidate_write(value,state,size_t(base));
        write_attention_kv<<<unsigned((steps*512+255)/256),256,0,value.stream>>>(value.kv_row, static_cast<kv_t*>(state),
            size_t(context), size_t(base), size_t(steps), value.input);
        attention_scores<<<unsigned(steps*24),256,0,value.stream>>>(value.kv_row, static_cast<const kv_t*>(state),
            size_t(context), size_t(base), size_t(steps), size_t(selected), value.input, value.attention_scores);
        attention_reduce<<<unsigned(steps*24),256,0,value.stream>>>(value.kv_row, static_cast<const kv_t*>(state),
            size_t(context), size_t(base), size_t(steps), size_t(selected), value.input, value.attention_scores, value.output);
        check(hipGetLastError());
        download(value, output, value.output, size_t(output_count) * 4);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_delta_sequence(void* handle, void* state, uint64_t heads, uint64_t dimension, uint64_t steps,
                      const float* input, uint64_t input_count, float* output, uint64_t output_count) {
    try {
        auto& value = device(handle);
        if (!input || !output || !heads || heads > 64 || !dimension || dimension > 128 || !steps || steps > 128)
            throw std::runtime_error("invalid Straydog DeltaNet dimensions");
        auto found = value.allocations.find(state);
        if (found == value.allocations.end() || found->second != heads * dimension * dimension * 4)
            throw std::runtime_error("Straydog DeltaNet state size mismatch");
        if (input_count != steps * heads * (3 * dimension + 2) || output_count != steps * heads * dimension ||
            input_count > value.input_capacity || output_count > value.output_capacity)
            throw std::runtime_error("Straydog DeltaNet workspace mismatch");
        upload(value, value.input, input, size_t(input_count) * 4);
        if (dimension == 128)
            delta_register_sequence<<<unsigned(heads),128,0,value.stream>>>(static_cast<float*>(state),
                size_t(heads),size_t(steps),value.input,value.output);
        else
            delta_sequence<<<unsigned(heads), unsigned(dimension), 0, value.stream>>>(static_cast<float*>(state),
                size_t(heads), size_t(dimension), size_t(steps), value.input, value.output);
        check(hipGetLastError());
        download(value, output, value.output, size_t(output_count) * 4);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_matvec_batch(void* handle, void* const* pointers, const int* kinds,
                    const uint64_t* rows, const uint64_t* cols, uint64_t count,
                    const float* input, uint64_t input_count, float* output, uint64_t output_count) {
    try {
        auto& value = device(handle);
        if (!pointers || !kinds || !rows || !cols || !input || !output || !count || count > value.batch_capacity)
            throw std::runtime_error("invalid Straydog batch");
        Matrix descriptors[max_batch]{};
        size_t total_in = 0, total_out = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!rows[i] || !cols[i] || rows[i] > max_rows || cols[i] > max_cols)
                throw std::runtime_error("invalid Straydog matrix dimensions");
            const size_t row_bytes = matrix_row_bytes(kinds[i], cols[i]);
            auto found = value.allocations.find(pointers[i]);
            if (found == value.allocations.end() || found->second != size_t(rows[i]) * row_bytes)
                throw std::runtime_error("matrix allocation size mismatch");
            descriptors[i] = {static_cast<const uint8_t*>(pointers[i]), row_bytes,
                              size_t(rows[i]), size_t(cols[i]), total_in, total_out, total_out, kinds[i]};
            total_in += size_t(cols[i]); total_out += size_t(rows[i]);
        }
        if (total_in > value.input_capacity || total_out > value.output_capacity ||
            total_in != input_count || total_out != output_count)
            throw std::runtime_error("Straydog batch workspace exceeded");
        upload(value, value.input, input, total_in * 4);
        GemvItem fallback[max_batch]{};
        size_t fallback_count = 0, fallback_waves = 0;
        bool half_inputs_ready = false;
        for (size_t first = 0; first < count;) {
            size_t last = first + 1;
            const Matrix& item = descriptors[first];
            while (last < count && descriptors[last].data == item.data && descriptors[last].kind == item.kind
                   && descriptors[last].rows == item.rows && descriptors[last].cols == item.cols) ++last;
            const size_t members = last - first;
            if (value.blas && members >= value.gemm_min_batch && item.rows * item.cols <= gemm_elements) {
                const float alpha = 1, beta = 0;
                if (value.half_gemm) {
                    if (!half_inputs_ready) {
                        cast_half<<<unsigned((total_in+255)/256),256,0,value.stream>>>(value.input, value.half_input, total_in);
                        check(hipGetLastError()); half_inputs_ready = true;
                    }
                    expand(item, value.expanded, value.stream, true);
                    blas_check(hipblasGemmEx(value.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                        int(item.rows), int(members), int(item.cols), &alpha, value.expanded, HIP_R_16F, int(item.cols),
                        value.half_input + item.input_offset, HIP_R_16F, int(item.cols), &beta,
                        value.output + item.output_offset, HIP_R_32F, int(item.rows), HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
                } else {
                    const float* matrix = reinterpret_cast<const float*>(item.data);
                    if (item.kind != 0) { expand(item, value.expanded, value.stream, false); matrix = value.expanded; }
                    blas_check(hipblasSgemm(value.blas_f32, HIPBLAS_OP_T, HIPBLAS_OP_N,
                        int(item.rows), int(members), int(item.cols), &alpha, matrix, int(item.cols),
                        value.input + item.input_offset, int(item.cols), &beta,
                        value.output + item.output_offset, int(item.rows)));
                }
                ++value.gemm_calls; value.gemm_vectors += members;
            } else {
                for (size_t i = first; i < last; ++i) {
                    const unsigned lanes = gemv_lanes(descriptors[i], value.input);
                    fallback[fallback_count++] = {descriptors[i], uint32_t(fallback_waves), lanes};
                    fallback_waves += gemv_waves(descriptors[i], lanes);
                }
            }
            first = last;
        }
        if (fallback_count && value.q8_wide) {
            // (gemv()'s kernels, so a decode step's head takes the Q8 kernels as a verification's does)
            for (size_t first = 0; first < fallback_count; first += gemv_fixed_items) {
                Matrix group[gemv_fixed_items];
                const size_t size = std::min(gemv_fixed_items, fallback_count - first);
                for (size_t i = 0; i < size; ++i) group[i] = fallback[first + i].matrix;
                gemv(value, group, size, value.input, value.output);
            }
        } else if (fallback_count) {
            upload(value, value.matrices, fallback, fallback_count * sizeof(GemvItem));
            gemv_list<<<unsigned((fallback_waves+7)/8), 256, 0, value.stream>>>(value.matrices, fallback_count, value.input, value.output);
            check(hipGetLastError());
        }
        download(value, output, value.output, total_out * 4);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_matvec(void* handle, void* pointer, int kind, uint64_t rows, uint64_t cols,
              const float* input, float* output) {
    return sd_matvec_batch(handle, &pointer, &kind, &rows, &cols, 1, input, cols, output, rows);
}

// ---- Ops over device buffers (ops.hpp) ----

int sd_gemm_check_start(void* handle) {
    try {
        auto& value = device(handle);
        if (value.gemm_checks || value.op_recording || !value.graphs.empty())
            throw std::runtime_error("GEMM checks must start once before graphs");
        value.gemm_checks = static_cast<uint64_t*>(allocate(value, 8 * sizeof(uint64_t)));
        check(hipMemsetAsync(value.gemm_checks, 0, 8 * sizeof(uint64_t), value.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_gemm_check_read(void* handle, uint64_t* counters, uint64_t count) {
    try {
        auto& value = device(handle);
        if (!value.gemm_checks || !counters || count != 8)
            throw std::runtime_error("GEMM checks need an enabled device and eight output counters");
        download(value, counters, value.gemm_checks, 8 * sizeof(uint64_t));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

void* sd_buffer(void* handle, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!bytes || bytes > value.budget - value.used || allocation_tail > value.budget - value.used - bytes)
            throw std::runtime_error(budget_exceeded(value, bytes));
        void* pointer = nullptr;
        check(hipMalloc(&pointer, size_t(bytes) + allocation_tail));
        try { value.allocations.emplace(pointer, size_t(bytes)); }
        catch (...) { cleanup(hipFree(pointer)); throw; }
        value.used += size_t(bytes) + allocation_tail;
        check(hipMemsetAsync(pointer, 0, size_t(bytes) + allocation_tail, value.stream));
        // Uploads use another stream. Return only after the zeros land, or a late memset wipes them.
        check(hipStreamSynchronize(value.stream));
        return pointer;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

int sd_buffer_write(void* handle, void* buffer, uint64_t offset, const void* data, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!data || offset > UINT64_MAX - bytes) throw std::runtime_error("invalid Straydog buffer write");
        op_span(value, static_cast<uint8_t*>(buffer) + offset, size_t(bytes), "a written buffer");
        written(value, buffer);
        upload(value, static_cast<uint8_t*>(buffer) + offset, data, size_t(bytes));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_buffer_read(void* handle, const void* buffer, uint64_t offset, void* output, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!output || offset > UINT64_MAX - bytes) throw std::runtime_error("invalid Straydog buffer read");
        op_span(value, static_cast<const uint8_t*>(buffer) + offset, size_t(bytes), "a read buffer");
        download(value, output, static_cast<const uint8_t*>(buffer) + offset, size_t(bytes));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// An asynchronous read of up to readback_capacity bytes: the copy is queued behind the work queued so
// far and lands in page-locked memory; finish waits for that copy alone (work queued after the start
// keeps the device busy meanwhile) and returns the bytes. One read is outstanding at a time.
constexpr size_t readback_capacity = size_t(1) << 20;
int sd_buffer_read_start(void* handle, const void* buffer, uint64_t offset, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!bytes || bytes > readback_capacity || offset > UINT64_MAX - bytes) throw std::runtime_error("invalid Straydog read");
        op_span(value, static_cast<const uint8_t*>(buffer) + offset, size_t(bytes), "a read buffer");
        if (!value.readback) {
            if (host_allocate(reinterpret_cast<void**>(&value.readback), readback_capacity, hipHostMallocDefault) != hipSuccess) {
                // (a machine short of memory: a pageable buffer, which a copy to it fills before it returns)
                (void)hipGetLastError();
                value.readback_pageable.resize(readback_capacity);
                value.readback = value.readback_pageable.data();
                value.readback_pinned = false;
            }
            check(hipEventCreateWithFlags(&value.readback_done, hipEventDisableTiming));
        }
        check(hipMemcpyAsync(value.readback, static_cast<const uint8_t*>(buffer) + offset, size_t(bytes), hipMemcpyDeviceToHost,
                             value.stream));
        check(hipEventRecord(value.readback_done, value.stream));
        value.readback_bytes = size_t(bytes);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_buffer_read_finish(void* handle, void* output, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!output || !value.readback_bytes || bytes != value.readback_bytes) throw std::runtime_error("no such Straydog read");
        check(hipEventSynchronize(value.readback_done));
        std::memcpy(output, value.readback, size_t(bytes));
        value.readback_bytes = 0;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// A copy between two devices' allocations, after the work queued on the source: the target's later
// work waits for the bytes and the source's for the copy (which may overwrite them). Peer access is
// enabled at the first copy between two devices that have it.
int sd_buffer_copy_peer(void* to, void* target, void* from, const void* source, uint64_t bytes) {
    try {
        Device& a = device(from);
        if (!bytes) throw std::runtime_error("invalid Straydog copy");
        op_span(a, source, size_t(bytes), "a copied buffer");
        Device& b = device(to);
        op_span(b, target, size_t(bytes), "a copy target");
        if (&a == &b) {
            check(hipMemcpyAsync(target, source, size_t(bytes), hipMemcpyDeviceToDevice, b.stream));
            return 0;
        }
        static std::mutex mutex;
        static std::vector<std::pair<int, int>> linked;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (std::find(linked.begin(), linked.end(), std::make_pair(a.ordinal, b.ordinal)) == linked.end()) {
                for (auto [x, y] : {std::make_pair(a.ordinal, b.ordinal), std::make_pair(b.ordinal, a.ordinal)}) {
                    int can = 0;
                    check(hipDeviceCanAccessPeer(&can, x, y));
                    if (can)
                        on_device(x, [&]() {
                            const hipError_t state = hipDeviceEnablePeerAccess(y, 0);
                            if (state == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
                            else check(state);
                        });
                }
                linked.emplace_back(a.ordinal, b.ordinal);
                linked.emplace_back(b.ordinal, a.ordinal);
            }
        }
        on_device(a.ordinal, [&]() { check(hipEventRecord(a.handoff, a.stream)); });
        on_device(b.ordinal, [&]() {
            check(hipStreamWaitEvent(b.stream, a.handoff, 0));
            check(hipMemcpyPeerAsync(target, b.ordinal, source, a.ordinal, size_t(bytes), b.stream));
            check(hipEventRecord(b.handoff, b.stream));
        });
        on_device(a.ordinal, [&]() { check(hipStreamWaitEvent(a.stream, b.handoff, 0)); });
#if !defined(__HIPCC__)
        on_device(a.ordinal, [&]() { flush_launches(a); });
        on_device(b.ordinal, [&]() { flush_launches(b); });
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

static int op_linear_checked(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row,
                             const float* input, uint64_t input_stride, uint64_t steps, float* output, uint64_t output_stride,
                             float* scratch, uint64_t scratch_bytes, int precision) {
    try {
        auto& value = device(handle);
        input_stride = input_stride ? input_stride : cols;
        output_stride = output_stride ? output_stride : rows;
        if (!rows || !cols || !steps || input_stride < cols || output_stride < rows)
            throw std::runtime_error("invalid Straydog linear shape");
        const size_t row_bytes = matrix_row_bytes(kind, cols);
        op_span(value, weights, size_t(first_row + rows) * row_bytes, "a weight matrix");
        op_span(value, input, size_t((steps - 1) * input_stride + cols) * 4, "a linear input");
        op_span(value, output, size_t((steps - 1) * output_stride + rows) * 4, "a linear output");
        if (steps > uint64_t(gemv_max_vectors)) op_span(value, scratch, size_t(scratch_bytes), "a linear scratch");
        op_linear(value, static_cast<const uint8_t*>(weights), kind, size_t(rows), size_t(cols), size_t(first_row), input,
                  size_t(input_stride), size_t(steps), output, size_t(output_stride), scratch, size_t(scratch_bytes), precision);
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_linear_gemm_half(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row,
                           const void* input, uint64_t steps, float* output, uint64_t output_stride, float* scratch, uint64_t scratch_bytes) {
    try {
        auto& value = device(handle);
        output_stride = output_stride ? output_stride : rows;
        if (!rows || !cols || steps <= uint64_t(gemv_max_vectors) || output_stride < rows || !value.blas)
            throw std::runtime_error("invalid Straydog FP16 linear shape");
        const size_t row_bytes = matrix_row_bytes(kind, cols);
        op_span(value, weights, size_t(first_row + rows) * row_bytes, "a weight matrix");
        op_span(value, input, size_t(steps * cols) * 2, "an FP16 linear input");
        op_span(value, output, size_t((steps - 1) * output_stride + rows) * 4, "a linear output");
        op_span(value, scratch, size_t(scratch_bytes), "a linear scratch");
        const Matrix matrix{static_cast<const uint8_t*>(weights) + first_row * row_bytes, row_bytes, size_t(rows), size_t(cols), 0, 0, 0, kind};
        // (the pieces sd_op_linear_gemm's would be, its rounded input at the scratch's end: the same GEMMs)
        const size_t input_bytes = (size_t(steps * cols) * 2 + 255) & ~size_t(255);
        if (scratch_bytes <= input_bytes) throw std::runtime_error("Straydog op: GEMM scratch holds no row");
        op_linear_half(value, matrix, static_cast<const __half*>(input), size_t(steps), output, size_t(output_stride), scratch,
                       size_t(scratch_bytes) - input_bytes);
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_half_rows(void* handle, const float* input, uint64_t stride, uint64_t cols, uint64_t steps, void* output) {
    try {
        auto& value = device(handle);
        if (!input || !output || !cols || !steps || stride < cols) throw std::runtime_error("invalid Straydog half rows");
        op_span(value, input, size_t((steps - 1) * stride + cols) * 4, "rows to round");
        op_span(value, output, size_t(steps * cols) * 2, "rounded rows");
        op_half_rows<<<unsigned((steps * cols + 255) / 256), 256, 0, value.stream>>>(input, size_t(stride), size_t(cols), size_t(steps),
                                                                                     static_cast<__half*>(output));
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_linear(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row,
                 const float* input, uint64_t input_stride, uint64_t steps, float* output, uint64_t output_stride,
                 float* scratch, uint64_t scratch_bytes) {
    return op_linear_checked(handle, weights, kind, rows, cols, first_row, input, input_stride, steps, output, output_stride,
                             scratch, scratch_bytes, -1);
}

int sd_op_linear_gemm(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row,
                      const float* input, uint64_t input_stride, uint64_t steps, float* output, uint64_t output_stride,
                      float* scratch, uint64_t scratch_bytes, int half_precision) {
    if (half_precision != 0 && half_precision != 1) {
        last_error = "Straydog linear GEMM precision must be 0 (F32) or 1 (FP16)";
        return -1;
    }
    return op_linear_checked(handle, weights, kind, rows, cols, first_row, input, input_stride, steps, output, output_stride,
                             scratch, scratch_bytes, half_precision);
}

// ---- Graphs of ops (kept in Device::graphs under "op" and the caller's key) ----

extern "C++" {
static std::string op_graph_key(uint64_t key) {
    std::string name("op");
    name.append(reinterpret_cast<const char*>(&key), sizeof(key));
    return name;
}
} // extern "C++"

int sd_graph_begin(void* handle) {
    try {
        auto& value = device(handle);
        if (value.op_recording) throw std::runtime_error("a Straydog graph is already being recorded");
        // (made here, not by the route's launch: a capture takes no allocation)
        if (!value.route_counter) {
            value.route_counter = static_cast<unsigned*>(allocate(value, 4));
            check(hipMemset(value.route_counter, 0, 4));
        }
        check(hipStreamBeginCapture(value.stream, hipStreamCaptureModeThreadLocal));
        value.op_recording = true;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_graph_end(void* handle, uint64_t key, int keep) {
    try {
        auto& value = device(handle);
        if (!value.op_recording) throw std::runtime_error("no Straydog graph is being recorded");
        value.op_recording = false;
        CachedGraph made{};
        check(hipStreamEndCapture(value.stream, &made.graph));
        if (!keep) {
            cleanup(hipGraphDestroy(made.graph));
            return 0;
        }
        const hipError_t instantiated = hipGraphInstantiate(&made.executable, made.graph, nullptr, nullptr, 0);
        if (instantiated != hipSuccess) {
            cleanup(hipGraphDestroy(made.graph));
            check(instantiated);
        }
        auto& slot = value.graphs[op_graph_key(key)];
        if (slot.executable) cleanup(hipGraphExecDestroy(slot.executable));
        if (slot.graph) cleanup(hipGraphDestroy(slot.graph));
        slot = std::move(made);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_graph_run(void* handle, uint64_t key) {
    try {
        auto& value = device(handle);
        const auto found = value.graphs.find(op_graph_key(key));
        if (found == value.graphs.end()) return 1;
        graph_launch(value, found->second);
        ++value.graph_replays;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_devices_flush(void* const* handles, uint64_t devices) {
    try {
#if !defined(__HIPCC__)
        if (!handles || !devices || devices > 8) throw std::runtime_error("invalid Straydog devices");
        for (size_t i = 0; i < devices; ++i) {
            Device& d = device(handles[i]);
            on_device(d.ordinal, [&]() { flush_launches(d); });
        }
#else
        (void)handles;
        (void)devices;
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_linears(void* handle, const sd_linear* items, uint64_t count, uint64_t steps, uint64_t input_stride,
                  uint64_t output_stride, float* scratch, uint64_t scratch_bytes) {
    try {
        auto& value = device(handle);
        if (!items || !count || count > gemv_fixed_items || !steps) throw std::runtime_error("invalid Straydog linears");
        Matrix matrices[gemv_fixed_items];
        const float* input = nullptr;
        float* output = nullptr;
        bool shared = true;
        for (size_t i = 0; i < count; ++i) {
            const sd_linear& item = items[i];
            const size_t in_stride = input_stride ? input_stride : item.cols, out_stride = output_stride ? output_stride : item.rows;
            if (!item.rows || !item.cols || in_stride < item.cols || out_stride < item.rows || uintptr_t(item.input) % 4 ||
                uintptr_t(item.output) % 4)
                throw std::runtime_error("invalid Straydog linear shape");
            const size_t row_bytes = matrix_row_bytes(item.kind, item.cols);
            op_span(value, item.weights, size_t(item.first_row + item.rows) * row_bytes, "a weight matrix");
            op_span(value, item.input, ((steps - 1) * in_stride + item.cols) * 4, "a linear input");
            op_span(value, item.output, ((steps - 1) * out_stride + item.rows) * 4, "a linear output");
            matrices[i] = {static_cast<const uint8_t*>(item.weights) + item.first_row * row_bytes, row_bytes, size_t(item.rows),
                           size_t(item.cols), 0, 0, 0, item.kind};
            input = !input || item.input < input ? item.input : input;
            output = !output || item.output < output ? item.output : output;
            shared &= steps == 1 || (input_stride && output_stride);
        }
        if (steps <= uint64_t(gemv_max_vectors) && shared) {
            // One launch: every item addressed from the lowest input and output.
            for (size_t i = 0; i < count; ++i) {
                matrices[i].input_offset = size_t(items[i].input - input);
                matrices[i].output_offset = size_t(items[i].output - output);
            }
            gemv(value, matrices, size_t(count), input, output, size_t(steps), size_t(input_stride), size_t(output_stride), {},
                 op_lanes);
        } else if (steps > uint64_t(gemv_max_vectors)) {
            op_span(value, scratch, size_t(scratch_bytes), "a linear scratch");
            // Several half GEMMs of one row range used to round those rows once per matrix. Round once
            // when every item reads the same rows (same pointer, width and stride); each product is then
            // the half GEMM op_linear would run on that cast.
            const size_t stride0 = input_stride ? size_t(input_stride) : size_t(items[0].cols);
            bool one_input = value.half_gemm;
            for (size_t i = 0; one_input && i < count; ++i) {
                const size_t stride = input_stride ? size_t(input_stride) : size_t(items[i].cols);
                one_input = items[i].input == items[0].input && items[i].cols == items[0].cols && stride == stride0;
            }
            if (one_input) {
                if (!value.blas) throw std::runtime_error("Straydog op: GEMM needs a BLAS handle");
                const StagedHalfInput half = stage_half_gemm_input(value, items[0].input, stride0, size_t(items[0].cols),
                                                                   size_t(steps), scratch, size_t(scratch_bytes));
                for (size_t i = 0; i < count; ++i)
                    op_linear_half(value, matrices[i], half.rows, size_t(steps), items[i].output,
                                   output_stride ? size_t(output_stride) : size_t(items[i].rows), scratch, half.scratch_bytes);
            } else {
                for (size_t i = 0; i < count; ++i)
                    op_linear(value, static_cast<const uint8_t*>(items[i].weights), items[i].kind, size_t(items[i].rows),
                              size_t(items[i].cols), size_t(items[i].first_row), items[i].input,
                              size_t(input_stride ? input_stride : items[i].cols), size_t(steps), items[i].output,
                              size_t(output_stride ? output_stride : items[i].rows), scratch, size_t(scratch_bytes));
            }
        } else {
            for (size_t i = 0; i < count; ++i)
                op_linear(value, static_cast<const uint8_t*>(items[i].weights), items[i].kind, size_t(items[i].rows),
                          size_t(items[i].cols), size_t(items[i].first_row), items[i].input,
                          size_t(input_stride ? input_stride : items[i].cols), size_t(steps), items[i].output,
                          size_t(output_stride ? output_stride : items[i].rows), scratch, size_t(scratch_bytes));
        }
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_rms_norm(void* handle, const float* input, const float* weight, uint64_t rows, uint64_t width, float eps,
                   float* output) {
    try {
        auto& value = device(handle);
        if (!rows || !width) throw std::runtime_error("invalid Straydog norm shape");
        op_span(value, input, size_t(rows * width) * 4, "a norm input");
        op_span(value, weight, size_t(width) * 4, "a norm weight");
        op_span(value, output, size_t(rows * width) * 4, "a norm output");
        op_rms_norm_kernel<<<unsigned(rows), 256, 0, value.stream>>>(input, weight, size_t(width), eps, output);
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_norm_rope(void* handle, const float* input, const float* weight, uint64_t rows, uint64_t width, float eps,
                   uint64_t rope_dims, const int32_t* positions, const float* frequencies, const int32_t* out_rows, float* out,
                   uint64_t out_count) {
    try {
        auto& value = device(handle);
        if (!rows || !width || rope_dims % 2 || rope_dims > width || (!out_rows && out_count < rows))
            throw std::runtime_error("invalid Straydog norm shape");
        op_span(value, input, size_t(rows * width) * 4, "a norm input");
        op_span(value, weight, size_t(width) * 4, "a norm weight");
        if (rope_dims) {
            op_span(value, positions, size_t(rows) * 4, "rope positions");
            op_span(value, frequencies, size_t(rope_dims / 2) * 4, "rope frequencies");
        }
        op_span(value, out, size_t(out_count * width) * 4, "a norm output");
        if (out_rows) {
            op_span(value, out_rows, size_t(rows) * 4, "output rows");
            if (out == input) throw std::runtime_error("a scattered norm cannot run in place");
        }
        const AttentionTurnsView turns = attention_turns_for_rope(value, size_t(rows), size_t(rope_dims / 2));
        op_norm_rope_kernel<<<unsigned(rows), 256, 0, value.stream>>>(input, weight, size_t(width), eps, size_t(rope_dims / 2),
                                                                     positions, frequencies, out_rows, out, turns.forward, turns.inverse);
        check(hipGetLastError());
        if (turns.publish) attention_turns_mark_ready(value, size_t(rows), size_t(rope_dims / 2));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_rope(void* handle, float* x, uint64_t rows, uint64_t vectors, uint64_t width, uint64_t rope_dims,
               const int32_t* positions, const float* frequencies, int inverse) {
    try {
        auto& value = device(handle);
        if (!rows || !vectors || !rope_dims || rope_dims % 2 || rope_dims > width) throw std::runtime_error("invalid Straydog rope shape");
        op_span(value, x, size_t(rows * vectors * width) * 4, "a rotated buffer");
        op_span(value, positions, size_t(rows) * 4, "rope positions");
        op_span(value, frequencies, size_t(rope_dims / 2) * 4, "rope frequencies");
        const size_t count = size_t(rows * vectors * (rope_dims / 2));
        op_rope_kernel<<<unsigned((count + 255) / 256), 256, 0, value.stream>>>(x, count, size_t(vectors), size_t(width),
                                                                               size_t(rope_dims / 2), positions, frequencies, inverse);
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

#define SD_OP_BEGIN try { auto& value = device(handle);
#define SD_OP_END check(hipGetLastError()); return 0; } catch (const std::exception& e) { last_error = e.what(); return -1; }
static unsigned op_blocks(size_t count) { return unsigned((count + 255) / 256); }

int sd_op_hc_coefficients(void* handle, const float* mixes, const float* stream, uint64_t rows, uint64_t hc, uint64_t width,
                          const float* base, const float* scale, uint64_t iterations, float eps, float norm_eps,
                          float* pre, float* post, float* comb) {
    SD_OP_BEGIN
    if (!rows || !hc || hc > 8 || !width) throw std::runtime_error("invalid Straydog hyper-connection shape");
    const size_t mix = size_t((2 + hc) * hc);
    op_span(value, mixes, size_t(rows) * mix * 4, "hyper-connection mixes");
    op_span(value, stream, size_t(rows * hc * width) * 4, "a stream");
    op_span(value, base, mix * 4, "hyper-connection base");
    op_span(value, scale, 12, "hyper-connection scale");
    op_span(value, pre, size_t(rows * hc) * 4, "pre coefficients");
    op_span(value, post, size_t(rows * hc) * 4, "post coefficients");
    op_span(value, comb, size_t(rows * hc * hc) * 4, "combination coefficients");
    op_hc_coefficients_kernel<<<unsigned(rows), 256, 0, value.stream>>>(mixes, stream, size_t(hc), size_t(width), base, scale,
                                                                     size_t(iterations), eps, norm_eps, pre, post, comb);
    SD_OP_END
}

int sd_op_hc_collapse(void* handle, const float* stream, const float* pre, uint64_t rows, uint64_t hc, uint64_t width, float* out) {
    SD_OP_BEGIN
    op_span(value, stream, size_t(rows * hc * width) * 4, "a stream");
    op_span(value, pre, size_t(rows * hc) * 4, "pre coefficients");
    op_span(value, out, size_t(rows * width) * 4, "a collapsed output");
    op_hc_collapse_kernel<<<op_blocks(size_t(rows * width)), 256, 0, value.stream>>>(stream, pre, size_t(rows), size_t(hc), size_t(width), out);
    SD_OP_END
}

int sd_op_hc_mean(void* handle, const float* stream, uint64_t rows, uint64_t hc, uint64_t width, uint64_t skip, float* out,
                  uint64_t stride, uint64_t offset) {
    SD_OP_BEGIN
    if (!rows || !hc || !width || skip >= rows || offset + width > stride) throw std::runtime_error("invalid Straydog hc mean");
    op_span(value, stream, size_t(rows * hc * width) * 4, "a stream");
    op_span(value, out, size_t(((rows - skip - 1) * stride + offset + width)) * 4, "hc means");
    op_hc_mean_kernel<<<op_blocks(size_t(rows * width)), 256, 0, value.stream>>>(stream, size_t(rows), size_t(hc), size_t(width),
                                                                                  size_t(skip), out, size_t(stride), size_t(offset));
    SD_OP_END
}

int sd_op_hc_expand(void* handle, const float* x, const float* residual, const float* post, const float* comb, uint64_t rows,
                    uint64_t hc, uint64_t width, float* out) {
    SD_OP_BEGIN
    op_span(value, x, size_t(rows * width) * 4, "a sublayer output");
    op_span(value, residual, size_t(rows * hc * width) * 4, "a residual stream");
    op_span(value, post, size_t(rows * hc) * 4, "post coefficients");
    op_span(value, comb, size_t(rows * hc * hc) * 4, "combination coefficients");
    op_span(value, out, size_t(rows * hc * width) * 4, "an expanded stream");
    if (out == residual) throw std::runtime_error("Straydog hyper-connection expansion cannot run in place");
    op_hc_expand_kernel<<<op_blocks(size_t(rows * hc * width)), 256, 0, value.stream>>>(x, residual, post, comb, size_t(rows),
                                                                                    size_t(hc), size_t(width), out);
    SD_OP_END
}

// sd_op_hc_step; with `channel` (0 or 1, else -1), x is this device's part of a part link's sum.
static int hc_step_op(void* handle, int channel, const float* x, const float* residual, const float* post_in, const float* comb_in,
                      float* stream, const float* fn, const float* base, const float* scale, uint64_t iterations, float eps,
                      float norm_eps, const float* collapse, const float* norm_weight, uint64_t rows, uint64_t hc, uint64_t width,
                      float* workspace, uint64_t workspace_bytes, float* pre, float* post, float* comb, float* out) {
    SD_OP_BEGIN
    // (rows, hc and width bounded so that no size below wraps)
    if (!rows || rows > 65535 || !hc || hc > 8 || !width || width > (uint64_t(1) << 24) || (x && stream == residual) ||
        (channel >= 0 && !x))
        throw std::runtime_error("invalid Straydog hyper-connection step");
    const PartRead link = channel >= 0 ? part_read(value, channel) : PartRead{};
    if (channel >= 0 && width > link.capacity / rows) throw std::runtime_error("a linked hyper-connection step exceeds its inbox");
    const size_t mix = size_t((2 + hc) * hc), flat = size_t(rows * hc * width);
    op_span(value, stream, flat * 4, "a stream");
    if (x) {
        op_span(value, x, size_t(rows * width) * 4, "a sublayer output");
        op_span(value, residual, flat * 4, "a residual stream");
        op_span(value, post_in, size_t(rows * hc) * 4, "post coefficients");
        op_span(value, comb_in, size_t(rows * hc * hc) * 4, "combination coefficients");
    }
    if (fn) {
        // (BF16 weights: tagged, as state rows are)
        if (state_bf16(fn)) op_span(value, state_halves(fn), mix * size_t(hc * width) * 2, "hyper-connection weights");
        else op_span(value, fn, mix * size_t(hc * width) * 4, "hyper-connection weights");
        op_span(value, base, mix * 4, "hyper-connection base");
        op_span(value, scale, 12, "hyper-connection scale");
        op_span(value, pre, size_t(rows * hc) * 4, "pre coefficients");
        op_span(value, post, size_t(rows * hc) * 4, "post coefficients");
        op_span(value, comb, size_t(rows * hc * hc) * 4, "combination coefficients");
    }
    op_span(value, collapse, size_t(rows * hc) * 4, "collapse coefficients");
    op_span(value, norm_weight, size_t(width) * 4, "a norm weight");
    op_span(value, out, size_t(rows * width) * 4, "a collapsed output");
    // The counters lead the workspace, so the same rows keep the same counters whatever the batch.
    const size_t capacity = size_t(workspace_bytes / 4) / hc_step_row_values(size_t(hc), size_t(width));
    if (capacity < rows) throw std::runtime_error("Straydog hyper-connection step workspace too small");
    op_span(value, workspace, size_t(workspace_bytes), "a hyper-connection workspace");
    auto* counters = reinterpret_cast<unsigned*>(workspace);
    const dim3 grid(unsigned((width + hc_step_columns - 1) / hc_step_columns), unsigned(rows));
    // (a long pass's rows take the weights once a block of rows: op_hc_step_rows_kernel's values are op_hc_step_kernel's)
    if (x && fn && channel < 0 && rows >= 32) {
        const size_t blocks = grid.x;
        float* const partial = workspace + capacity;
        const dim3 rows_grid(grid.x, unsigned((size_t(rows) + op_hc_step_rows - 1) / op_hc_step_rows)), finish_grid(1, unsigned(rows));
        #define SD_HC_STEP_ROWS(n) case n: \
            op_hc_step_rows_kernel<n><<<rows_grid, hc_step_threads, 0, value.stream>>>(x, residual, post_in, comb_in, stream, fn, collapse, size_t(rows), \
                size_t(width), partial, out); \
            check(hipGetLastError()); \
            op_hc_step_finish_kernel<n><<<finish_grid, hc_step_threads, 0, value.stream>>>(blocks, size_t(width), fn, base, scale, size_t(iterations), eps, norm_eps, \
                norm_weight, partial, pre, post, comb, out); break;
        switch (hc) { SD_HC_STEP_ROWS(1) SD_HC_STEP_ROWS(2) SD_HC_STEP_ROWS(3) SD_HC_STEP_ROWS(4) SD_HC_STEP_ROWS(5) SD_HC_STEP_ROWS(6) SD_HC_STEP_ROWS(7) SD_HC_STEP_ROWS(8) }
        #undef SD_HC_STEP_ROWS
        check(hipGetLastError());
        return 0;
    }
    #define SD_HC_STEP(n) case n: op_hc_step_kernel<n><<<grid, hc_step_threads, 0, value.stream>>>(x, residual, post_in, comb_in, stream, fn, \
        base, scale, size_t(iterations), eps, norm_eps, collapse, norm_weight, size_t(width), counters, workspace + capacity, pre, \
        post, comb, out, link); break;
    switch (hc) { SD_HC_STEP(1) SD_HC_STEP(2) SD_HC_STEP(3) SD_HC_STEP(4) SD_HC_STEP(5) SD_HC_STEP(6) SD_HC_STEP(7) SD_HC_STEP(8) }
    #undef SD_HC_STEP
    SD_OP_END
}

int sd_op_hc_step(void* handle, const float* x, const float* residual, const float* post_in, const float* comb_in, float* stream,
                  const float* fn, const float* base, const float* scale, uint64_t iterations, float eps, float norm_eps,
                  const float* collapse, const float* norm_weight, uint64_t rows, uint64_t hc, uint64_t width, float* workspace,
                  uint64_t workspace_bytes, float* pre, float* post, float* comb, float* out) {
    return hc_step_op(handle, -1, x, residual, post_in, comb_in, stream, fn, base, scale, iterations, eps, norm_eps, collapse,
                      norm_weight, rows, hc, width, workspace, workspace_bytes, pre, post, comb, out);
}

int sd_op_hc_step_linked(void* handle, int channel, const float* part, const float* residual, const float* post_in,
                         const float* comb_in, float* stream, const float* fn, const float* base, const float* scale,
                         uint64_t iterations, float eps, float norm_eps, const float* collapse, const float* norm_weight, uint64_t rows,
                         uint64_t hc, uint64_t width, float* workspace, uint64_t workspace_bytes, float* pre, float* post, float* comb,
                         float* out) {
    if (channel < 0) {
        last_error = "invalid Straydog part link channel";
        return -1;
    }
    return hc_step_op(handle, channel, part, residual, post_in, comb_in, stream, fn, base, scale, iterations, eps, norm_eps, collapse,
                      norm_weight, rows, hc, width, workspace, workspace_bytes, pre, post, comb, out);
}

int sd_op_engram_gate(void* handle, float* stream, const float* kv, const float* weight, uint64_t rows, uint64_t hc,
                      uint64_t width, float eps) {
    SD_OP_BEGIN
    op_span(value, stream, size_t(rows * hc * width) * 4, "a stream");
    op_span(value, kv, size_t(rows * (hc + 1) * width) * 4, "Engram keys and values");
    op_span(value, weight, size_t(hc * width) * 4, "an Engram gate weight");
    op_engram_gate_kernel<<<unsigned(rows * hc), 256, 0, value.stream>>>(stream, kv, weight, size_t(hc), size_t(width), eps);
    SD_OP_END
}

int sd_op_route(void* handle, const float* logits, const float* bias, uint64_t rows, uint64_t experts, uint64_t k, int gating,
                int normalize, float scale, int32_t* ids, float* weights) {
    SD_OP_BEGIN
    if (!rows || !experts || !k || k > 64 || k > experts || gating < 0 || gating > 2 || experts * 8 > 48 * 1024)
        throw std::runtime_error("invalid Straydog route shape");
    op_span(value, logits, size_t(rows * experts) * 4, "router logits");
    if (bias) op_span(value, bias, size_t(experts) * 4, "a router bias");
    op_span(value, ids, size_t(rows * k) * 4, "routed experts");
    op_span(value, weights, size_t(rows * k) * 4, "routing weights");
    if (experts == 256 && gating == 0 && !bias && scale == 1.0f && k <= 64)
        op_route_softmax256_kernel<<<unsigned(rows), 32, 0, value.stream>>>(logits, size_t(k), normalize, ids, weights);
    else
        op_route_kernel<<<unsigned(rows), 256, size_t(experts) * 8, value.stream>>>(logits, bias, size_t(experts), size_t(k), gating,
                                                                                  normalize, scale, ids, weights);
    SD_OP_END
}

// The routes a route mailbox holds at most.
constexpr size_t route_mailbox_ids = 1024;

static int linear_route(void* handle, const void* weights, int kind, uint64_t experts, uint64_t cols, const float* input, uint64_t steps,
                        float* logits, const float* bias, uint64_t k, int gating, int normalize, float scale, int32_t* ids,
                        float* route_weights, float* scratch, uint64_t scratch_bytes, bool publish) {
    try {
        auto& value = device(handle);
        // (the softmax router of 256 experts has a one-wave route kernel of its own)
        const bool fused = (kind == 0 || kind == 1 || kind == 30) && steps >= 1 && steps <= 4 && !(experts == 256 && gating == 0 && !bias && scale == 1.0f);
        if (publish && (!fused || !value.route_mailbox || steps * k > route_mailbox_ids))
            throw std::runtime_error("Straydog routes are published by the fused router alone, into a cleared mailbox");
        if (!fused) {
            if (sd_op_linear(handle, weights, kind, experts, cols, 0, input, 0, steps, logits, 0, scratch, scratch_bytes)) return -1;
            return sd_op_route(handle, logits, bias, steps, experts, k, gating, normalize, scale, ids, route_weights);
        }
        if (!experts || !cols || !k || k > 64 || k > experts || gating < 0 || gating > 2 || experts * 8 > 48 * 1024)
            throw std::runtime_error("invalid Straydog router shape");
        const size_t row_bytes = matrix_row_bytes(kind, cols);
        op_span(value, weights, size_t(experts) * row_bytes, "a router matrix");
        op_span(value, input, size_t(steps * cols) * 4, "a router input");
        op_span(value, logits, size_t(steps * experts) * 4, "router logits");
        if (bias) op_span(value, bias, size_t(experts) * 4, "a router bias");
        op_span(value, ids, size_t(steps * k) * 4, "routed experts");
        op_span(value, route_weights, size_t(steps * k) * 4, "routing weights");
        if (!value.route_counter) {
            value.route_counter = static_cast<unsigned*>(allocate(value, 4));
            check(hipMemsetAsync(value.route_counter, 0, 4, value.stream));
        }
        const Matrix matrix{static_cast<const uint8_t*>(weights), row_bytes, size_t(experts), size_t(cols), 0, 0, 0, kind};
        RouteTail tail{bias, size_t(experts), size_t(k), gating, normalize, scale, ids, route_weights, value.route_counter, nullptr, nullptr};
        if (publish) {
            check(hipHostGetDevicePointer(reinterpret_cast<void**>(&tail.flag), value.route_mailbox, 0));
            tail.published = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(tail.flag) + 64);
        }
        // (strides in values, as op_linear passes them: a contiguous input and output)
        if (!gemv_route_launch(value, matrix, input, logits, size_t(steps), steps > 1 ? size_t(cols) : 0, steps > 1 ? size_t(experts) : 0,
                               tail, op_lanes)) {
            if (publish) throw std::runtime_error("Straydog routes are published by the fused router alone");
            if (sd_op_linear(handle, weights, kind, experts, cols, 0, input, 0, steps, logits, 0, scratch, scratch_bytes)) return -1;
            return sd_op_route(handle, logits, bias, steps, experts, k, gating, normalize, scale, ids, route_weights);
        }
        check(hipGetLastError());
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_linear_route(void* handle, const void* weights, int kind, uint64_t experts, uint64_t cols, const float* input, uint64_t steps,
                       float* logits, const float* bias, uint64_t k, int gating, int normalize, float scale, int32_t* ids,
                       float* route_weights, float* scratch, uint64_t scratch_bytes) {
    return linear_route(handle, weights, kind, experts, cols, input, steps, logits, bias, k, gating, normalize, scale, ids, route_weights,
                        scratch, scratch_bytes, false);
}

int sd_op_linear_route_published(void* handle, const void* weights, int kind, uint64_t experts, uint64_t cols, const float* input,
                                 uint64_t steps, float* logits, const float* bias, uint64_t k, int gating, int normalize, float scale,
                                 int32_t* ids, float* route_weights, float* scratch, uint64_t scratch_bytes) {
    return linear_route(handle, weights, kind, experts, cols, input, steps, logits, bias, k, gating, normalize, scale, ids, route_weights,
                        scratch, scratch_bytes, true);
}

int sd_route_mailbox_clear(void* handle) {
    try {
        auto& value = device(handle);
        if (!value.route_mailbox)   // (made here, outside any graph being recorded)
            check(hipHostMalloc(reinterpret_cast<void**>(&value.route_mailbox), 64 + route_mailbox_ids * 4,
                                hipHostMallocCoherent | hipHostMallocMapped));
        host_store_release(reinterpret_cast<unsigned*>(value.route_mailbox), 0u);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_route_mailbox_wait(void* handle, int32_t* out, uint64_t count) {
    try {
        auto& value = device(handle);
        if (!value.route_mailbox || !out || !count || count > route_mailbox_ids) throw std::runtime_error("no Straydog route mailbox");
        const auto* flag = reinterpret_cast<const unsigned*>(value.route_mailbox);
        const auto started = std::chrono::steady_clock::now();
        for (unsigned spins = 1; host_load_acquire(flag) != 1u; ++spins) {
            host_pause();
            if (spins % 4096 == 0) {
                // A stream that failed never raises the flag: its error instead of a wait without end.
                const hipError_t state = hipStreamQuery(value.stream);
                if (state != hipSuccess && state != hipErrorNotReady) check(state);
                if (std::chrono::steady_clock::now() - started > std::chrono::seconds(60))
                    throw std::runtime_error("Straydog routes never arrived (60 s)");
            }
        }
        std::memcpy(out, value.route_mailbox + 64, size_t(count) * 4);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_zero(void* handle, void* pointer, uint64_t bytes) {
    SD_OP_BEGIN
    op_span(value, pointer, size_t(bytes), "a zeroed range");
    check(hipMemsetAsync(pointer, 0, size_t(bytes), value.stream));
    SD_OP_END
}

int sd_op_swiglu(void* handle, const float* gate, const float* up, uint64_t rows, uint64_t width, float clamp,
                 const float* row_weights, float* out) {
    SD_OP_BEGIN
    op_span(value, gate, size_t(rows * width) * 4, "a gate projection");
    op_span(value, up, size_t(rows * width) * 4, "an up projection");
    if (row_weights) op_span(value, row_weights, size_t(rows) * 4, "row weights");
    op_span(value, out, size_t(rows * width) * 4, "a SwiGLU output");
    op_swiglu_kernel<<<op_blocks(size_t(rows * width)), 256, 0, value.stream>>>(gate, up, size_t(rows), size_t(width), clamp,
                                                                             row_weights, out);
    SD_OP_END
}

int sd_op_gather(void* handle, const float* x, uint64_t x_rows, const int32_t* rows, uint64_t count, uint64_t width, float* out) {
    SD_OP_BEGIN
    op_span(value, x, size_t(x_rows * width) * 4, "gathered rows");
    op_span(value, rows, size_t(count) * 4, "row indices");
    op_span(value, out, size_t(count * width) * 4, "a gather output");
    op_gather_kernel<<<op_blocks(size_t(count * width)), 256, 0, value.stream>>>(x, rows, size_t(count), size_t(width), out);
    SD_OP_END
}

int sd_op_scatter(void* handle, const float* x, const int32_t* rows, uint64_t count, uint64_t width, float* out,
                  uint64_t out_rows) {
    SD_OP_BEGIN
    op_span(value, x, size_t(count * width) * 4, "written rows");
    op_span(value, rows, size_t(count) * 4, "row indices");
    op_span(value, out, size_t(out_rows * width) * 4, "a scatter output");
    op_scatter_kernel<<<op_blocks(size_t(count * width)), 256, 0, value.stream>>>(x, rows, size_t(count), size_t(width), out);
    SD_OP_END
}

int sd_op_scatter_add(void* handle, const float* x, const int32_t* rows, uint64_t count, uint64_t width, float* out,
                      uint64_t out_rows) {
    SD_OP_BEGIN
    op_span(value, x, size_t(count * width) * 4, "added rows");
    op_span(value, rows, size_t(count) * 4, "row indices");
    op_span(value, out, size_t(out_rows * width) * 4, "a scatter output");
    op_scatter_add_kernel<<<op_blocks(size_t(count * width)), 256, 0, value.stream>>>(x, rows, size_t(count), size_t(width), out);
    SD_OP_END
}

int sd_op_scatter_rows(void* handle, const float* x, uint64_t x_rows, const int32_t* offsets, const int32_t* table, uint64_t table_values,
                       uint64_t rows, uint64_t cols, uint64_t stride, float* out) {
    SD_OP_BEGIN
    if (!x || !x_rows || !offsets || !table || !table_values || !rows || !cols || cols > stride) throw std::runtime_error("invalid Straydog row scatter");
    op_span(value, x, size_t((x_rows - 1) * stride + cols) * 4, "added rows");
    op_span(value, offsets, size_t(rows + 1) * 4, "row offsets");
    op_span(value, table, size_t(table_values) * 4, "routes by row");
    op_span(value, out, size_t((rows - 1) * stride + cols) * 4, "a scatter output");
    if (offsets < table || offsets + rows + 1 > table + table_values) throw std::runtime_error("row offsets outside their table");
    op_scatter_rows_kernel<<<op_blocks(size_t(rows * cols)), 256, 0, value.stream>>>(x, size_t(x_rows), offsets, table, size_t(table_values),
                                                                                     size_t(rows), size_t(cols), size_t(stride), out);
    SD_OP_END
}

// ---- A wave of experts in slots through the RDNA4 grouped WMMA kernels (DeepSeek's prefill) ----
// Expert i's slot at bases[i] holds gate and up (ffn_share rows of `width` each, gate_bytes apart)
// then down (hidden_share rows of `ffn`); its routes are the grouped rows firsts[i]..+counts[i].
// The weights decode straight into FP16 WMMA fragments: none is expanded into VRAM first.
#if defined(__HIPCC__)
// The wave's group and tile lists, uploaded into `tables` (stream-ordered); the tiles' count, and
// the grouped rows the wave covers (`rows_out`). Every slot is checked against its allocation;
// the experts' grouped rows follow one another from 0, without gaps or overlaps.
static unsigned slot_moe_tables(Device& d, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts,
                                uint64_t experts, uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share,
                                uint64_t ffn, uint64_t hidden_share, void* tables, uint64_t table_bytes, MoeGroup*& groups,
                                MoeTile*& tiles, size_t& rows_out) {
    if (!bases || !firsts || !counts || !kinds || !experts || experts > 4096 || !tables || !width || !ffn_share || !ffn ||
        !hidden_share || ffn_share > ffn)
        throw std::runtime_error("invalid Straydog slot experts");
    const size_t gate_rows = matrix_row_bytes(kinds[0], size_t(width)), up_rows = matrix_row_bytes(kinds[1], size_t(width));
    const size_t down_bytes = size_t(hidden_share) * matrix_row_bytes(kinds[2], size_t(ffn));
    if (gate_bytes < size_t(ffn_share) * std::max(gate_rows, up_rows)) throw std::runtime_error("Straydog slot expert roles overlap");
    std::vector<MoeGroup> list(experts);
    std::vector<MoeTile> tile_list;
    size_t next = 0;
    for (size_t i = 0; i < experts; ++i) {
        const auto* base = reinterpret_cast<const uint8_t*>(bases[i]);
        if (!base || firsts[i] != next || !counts[i]) throw std::runtime_error("invalid Straydog slot expert routes");
        next += counts[i];
        op_span(d, base, size_t(2 * gate_bytes) + down_bytes, "a slot expert");
        list[i].weights[0] = {base, matrix_row_bytes(kinds[0], size_t(width)), size_t(ffn_share), size_t(width), 0, 0, 0, kinds[0]};
        list[i].weights[1] = {base + gate_bytes, matrix_row_bytes(kinds[1], size_t(width)), size_t(ffn_share), size_t(width), 0, 0, 0,
                              kinds[1]};
        list[i].weights[2] = {base + 2 * gate_bytes, matrix_row_bytes(kinds[2], size_t(ffn)), size_t(hidden_share), size_t(ffn), 0, 0,
                              0, kinds[2]};
        list[i].first = firsts[i];
        list[i].count = counts[i];
        list[i].half = 1;
        for (uint32_t row = 0; row < counts[i]; row += moe_block_m) tile_list.push_back({uint32_t(i), row});
    }
    rows_out = next;
    const size_t group_bytes = (experts * sizeof(MoeGroup) + 255) & ~size_t(255);
    if (group_bytes + tile_list.size() * sizeof(MoeTile) > table_bytes) throw std::runtime_error("Straydog slot expert tables overflow");
    op_span(d, tables, size_t(table_bytes), "slot expert tables");
    groups = static_cast<MoeGroup*>(tables);
    tiles = reinterpret_cast<MoeTile*>(static_cast<uint8_t*>(tables) + group_bytes);
    upload(d, groups, list.data(), experts * sizeof(MoeGroup));
    if (!tile_list.empty()) upload(d, tiles, tile_list.data(), tile_list.size() * sizeof(MoeTile));
    return unsigned(tile_list.size());
}

#endif

int sd_slot_moe_supported(void* handle, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn, uint64_t hidden_share) {
    try {
#if defined(__HIPCC__)
        const auto& value = device(handle);
        const MoeShape gate{uint32_t(width), uint32_t(ffn_share), 1}, down{uint32_t(hidden_share), uint32_t(ffn), 1};
        return value.rdna4 && kinds && moe_shape_fits(gate) && moe_shape_fits(down) && moe_gate_up_unit(kinds[0], kinds[1], gate) &&
               moe_down_unit(kinds[2], down) ? 1 : 0;
#else
        (void)handle; (void)kinds; (void)width; (void)ffn_share; (void)ffn; (void)hidden_share;
        return 0;
#endif
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_op_slot_moe_gate_up(void* handle, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts, uint64_t experts,
                           uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn,
                           uint64_t hidden_share, const int32_t* rows, uint64_t steps, const void* input, float clamp,
                           const float* weights, void* product, void* tables, uint64_t table_bytes) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    const MoeShape shape{uint32_t(width), uint32_t(ffn_share), 1};
    const int unit = moe_gate_up_unit(kinds ? kinds[0] : 0, kinds ? kinds[1] : 0, shape);
    if (!value.rdna4 || !moe_shape_fits(shape) || !unit || !rows || !input || !product || !steps)
        throw std::runtime_error("invalid Straydog slot expert gate/up");
    op_span(value, input, size_t(steps * width) * 2, "slot expert inputs");
    MoeGroup* groups = nullptr;
    MoeTile* tiles = nullptr;
    size_t grouped = 0;
    const unsigned count = slot_moe_tables(value, bases, firsts, counts, experts, gate_bytes, kinds, width, ffn_share, ffn, hidden_share,
                                           tables, table_bytes, groups, tiles, grouped);
    // (each route's row of the input is the kernel's to bound: device data, as op_gather's rows)
    op_span(value, rows, grouped * 4, "slot expert routes");
    if (weights) op_span(value, weights, grouped * 4, "slot expert route weights");
    op_span(value, product, grouped * size_t(ffn_share) * 2, "slot expert products");
    if (count) {
        #define SD_GATE_UP(format) case format: moe_gate_up_wmma<format, true, 0, 0, 0, true><<<dim3(shape.ffn / (64 * moe_wave_n), count), 256, 0, \
            value.stream>>>(groups, tiles, rows, size_t(steps), shape, static_cast<const _Float16*>(input), static_cast<_Float16*>(product), \
            nullptr, 0, false, nullptr, clamp, weights); break;
        switch (unit) { SD_MOE_UNITS(SD_GATE_UP) }
        #undef SD_GATE_UP
    }
#else
    (void)value; (void)bases; (void)firsts; (void)counts; (void)experts; (void)gate_bytes; (void)kinds; (void)width; (void)ffn_share;
    (void)ffn; (void)hidden_share; (void)rows; (void)steps; (void)input; (void)clamp; (void)weights; (void)product; (void)tables;
    (void)table_bytes;
    throw std::runtime_error("the grouped expert GEMM is RDNA4's");
#endif
    SD_OP_END
}

int sd_op_slot_moe_down(void* handle, const uint64_t* bases, const uint32_t* firsts, const uint32_t* counts, uint64_t experts,
                        uint64_t gate_bytes, const int* kinds, uint64_t width, uint64_t ffn_share, uint64_t ffn, uint64_t hidden_share,
                        const void* product, float* outputs, uint64_t out_stride, void* tables, uint64_t table_bytes) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    const MoeShape shape{uint32_t(hidden_share), uint32_t(ffn), 1};
    const int unit = moe_down_unit(kinds ? kinds[2] : 0, shape);
    if (!value.rdna4 || !moe_shape_fits(shape) || !unit || !product || !outputs || out_stride < hidden_share)
        throw std::runtime_error("invalid Straydog slot expert down");
    MoeGroup* groups = nullptr;
    MoeTile* tiles = nullptr;
    size_t grouped = 0;
    const unsigned count = slot_moe_tables(value, bases, firsts, counts, experts, gate_bytes, kinds, width, ffn_share, ffn, hidden_share,
                                           tables, table_bytes, groups, tiles, grouped);
    op_span(value, product, grouped * size_t(ffn) * 2, "slot expert products");
    op_span(value, outputs, ((grouped - 1) * size_t(out_stride) + size_t(hidden_share)) * 4, "slot expert outputs");
    if (count) {
        #define SD_DOWN(format) case format: moe_down_wmma<format><<<dim3(shape.hidden / (128 * moe_wave_n), count), 256, 0, \
            value.stream>>>(groups, tiles, nullptr, shape, static_cast<const _Float16*>(product), outputs, nullptr, 0, false, \
            size_t(out_stride)); break;
        switch (unit) { SD_MOE_UNITS(SD_DOWN) }
        #undef SD_DOWN
    }
#else
    (void)value; (void)bases; (void)firsts; (void)counts; (void)experts; (void)gate_bytes; (void)kinds; (void)width; (void)ffn_share;
    (void)ffn; (void)hidden_share; (void)product; (void)outputs; (void)out_stride; (void)tables; (void)table_bytes;
    throw std::runtime_error("the grouped expert GEMM is RDNA4's");
#endif
    SD_OP_END
}

int sd_op_repeat_rows(void* handle, const float* input, uint64_t rows, uint64_t copies, uint64_t width, float* output) {
    SD_OP_BEGIN
    if (!input || !output || !rows || !copies || !width) throw std::runtime_error("invalid Straydog row repeat");
    op_span(value, input, size_t(rows * width) * 4, "rows to repeat");
    op_span(value, output, size_t(rows * copies * width) * 4, "repeated rows");
    op_repeat_rows_kernel<<<op_blocks(size_t(rows * width)), 256, 0, value.stream>>>(input, size_t(rows), size_t(copies), size_t(width), output);
    SD_OP_END
}

int sd_op_half_values(void* handle, const float* input, uint64_t values, void* output) {
    SD_OP_BEGIN
#if defined(__HIPCC__)
    if (!input || !output || !values || values % 4) throw std::runtime_error("invalid Straydog half conversion");
    op_span(value, input, size_t(values) * 4, "values to round");
    op_span(value, output, size_t(values) * 2, "rounded values");
    moe_input_half<<<unsigned((values / 4 + 255) / 256), 256, 0, value.stream>>>(size_t(values), input, static_cast<_Float16*>(output));
#else
    (void)value; (void)input; (void)values; (void)output;
    throw std::runtime_error("FP16 rounding is RDNA4's here");
#endif
    SD_OP_END
}

int sd_attention_turns_reserve(void* handle, uint64_t rows, uint64_t rope_dims) {
    try {
        if (rope_dims > 128 || rope_dims % 2) throw std::runtime_error("invalid Straydog attention rope");
        attention_turns_reserve(device(handle), size_t(rows), size_t(rope_dims / 2));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_attention_turns_arm(void* handle, uint64_t rows) {
    try {
        auto& value = device(handle);
        value.attention_turns_armed = size_t(rows);
        value.attention_turns_ready = false;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// `values` values of DeepSeek-V4.1's state from `p`: BF16 when tagged (state_tag), else F32.
static void op_state_span(Device& value, const float* p, size_t values, const char* what) {
    if (state_bf16(p)) op_span(value, state_halves(p), values * 2, what);
    else op_span(value, p, values * 4, what);
}

int sd_op_sparse_attention(void* handle, const float* q, const float* raw, uint64_t raw_rows, const int32_t* positions,
                           uint64_t window, const float* compressed, uint64_t compressed_rows, const int32_t* selected,
                           uint64_t selections, const float* sinks, uint64_t rows, uint64_t heads, uint64_t dim,
                           float scale, uint64_t rope_dims, const float* frequencies, float* scratch, uint64_t scratch_bytes,
                           float* out) {
    SD_OP_BEGIN
    const size_t shared = size_t(dim + window + selections) * 4;
    if (!rows || rows > 65535 || !heads || !dim || !window || rope_dims % 2 || rope_dims > dim || rope_dims > 128)
        throw std::runtime_error("invalid Straydog attention shape");
    if (rope_dims) op_span(value, frequencies, size_t(rope_dims / 2) * 4, "rope frequencies");
    op_span(value, q, size_t(rows * heads * dim) * 4, "attention queries");
    op_span(value, raw, size_t(raw_rows * dim) * 4, "window rows");
    op_span(value, positions, size_t(rows) * 4, "query positions");
    if (selections) {
        op_state_span(value, compressed, size_t(compressed_rows * dim), "compressed rows");
        op_span(value, selected, size_t(rows * selections) * 4, "selected rows");
    }
    op_span(value, sinks, size_t(heads) * 4, "attention sinks");
    op_span(value, out, size_t(rows * heads * dim) * 4, "an attention output");
    // Tiled: scores (rows x heads x the most keys a row has), then per head and block of keys its max and sum.
    const size_t stride = size_t(window + selections), chunks = (stride + attention_keys - 1) / attention_keys;
    const size_t score_bytes = size_t(rows * heads) * stride * 4, need = score_bytes + size_t(rows * heads) * chunks * 8;
    // The weighted sums split into shares of at least 128 keys, when the scratch holds their sums.
    size_t shares = std::min<size_t>(4, (stride + 127) / 128);
    const size_t share_bytes = size_t(rows * heads) * (shares * dim + 1) * 4;
    if (shares > 1 && scratch_bytes < need + share_bytes) shares = 1;
    const size_t share_keys = (stride + shares * attention_keys - 1) / (shares * attention_keys) * attention_keys;
#if defined(__HIPCC__)
    // An FP16 prompt pass on RDNA4: both products through WMMA, every head of a row in one block
    // (ds_attention_wmma.hpp). Decode and short passes keep the F32 kernels below.
    const size_t wmma_chunks = (stride + ds_attention_chunk - 1) / ds_attention_chunk;
    const size_t wmma_need = score_bytes + size_t(rows * heads) * wmma_chunks * 8;
    if (value.rdna4 && value.half_gemm && rows >= 32 && dim % 64 == 0 && (heads == 32 || heads == 64) && scratch &&
        scratch_bytes >= wmma_need && stride * sizeof(float*) <= 32 * 1024) {
        op_span(value, scratch, wmma_need, "attention scratch");
        float* scores = scratch;
        auto* partial = reinterpret_cast<float2*>(reinterpret_cast<uint8_t*>(scratch) + score_bytes);
        const size_t pairs = size_t(rope_dims / 2);
        double2* forward = nullptr;
        double2* inverse = nullptr;
        if (pairs) {
            attention_turns_reserve(value, size_t(rows), pairs);
            forward = value.attention_turns;
            inverse = forward + value.attention_turns_rows * pairs;
            const bool filled = value.attention_turns_ready && value.attention_turns_ready_rows == size_t(rows) &&
                                value.attention_turns_ready_pairs == pairs;
            value.attention_turns_ready = false;
            if (!filled)
                op_attention_turns_kernel<<<unsigned(rows), 64, 0, value.stream>>>(positions, frequencies, pairs, forward, inverse);
        } else {
            value.attention_turns_ready = false;
        }
        const dim3 score_grid{unsigned(wmma_chunks), unsigned(rows)}, weight_grid{1, unsigned(rows)}, out_grid{unsigned(dim / 64), unsigned(rows)};
        const size_t keys_bytes = size_t(stride) * sizeof(float*);
        #define SD_DS_ATTENTION(HM) \
            ds_attention_scores_wmma<HM><<<score_grid, 128, 0, value.stream>>>(q, raw, size_t(raw_rows), positions, size_t(window), \
                compressed, selected, size_t(selections), size_t(heads), size_t(dim), scale, pairs, forward, stride, scores, partial); \
            check(hipGetLastError()); \
            ds_attention_weights_wmma<HM><<<weight_grid, 256, 0, value.stream>>>(scores, partial, wmma_chunks, positions, size_t(window), \
                size_t(selections), sinks, size_t(heads), stride); \
            check(hipGetLastError()); \
            ds_attention_output_wmma<HM><<<out_grid, 128, keys_bytes, value.stream>>>(scores, partial, wmma_chunks, raw, size_t(raw_rows), \
                positions, size_t(window), compressed, selected, size_t(selections), size_t(heads), size_t(dim), pairs, inverse, \
                stride, out);
        if (heads == 32) { SD_DS_ATTENTION(2) } else { SD_DS_ATTENTION(4) }
        #undef SD_DS_ATTENTION
        check(hipGetLastError());
        return 0;
    }
#endif
    if (dim % 64 == 0 && scratch && scratch_bytes >= need && share_keys * sizeof(float*) <= 32 * 1024 && rows * shares <= 65535) {
        op_span(value, scratch, need, "attention scratch");
        float* scores = scratch;
        auto* partial = reinterpret_cast<float2*>(reinterpret_cast<uint8_t*>(scratch) + score_bytes);
        const size_t pairs = size_t(rope_dims / 2);
        double2* forward = nullptr;
        double2* inverse = nullptr;
        if (pairs) {
            attention_turns_reserve(value, size_t(rows), pairs);
            forward = value.attention_turns;
            inverse = forward + value.attention_turns_rows * pairs;
            const bool filled = value.attention_turns_ready && value.attention_turns_ready_rows == size_t(rows) &&
                                value.attention_turns_ready_pairs == pairs;
            value.attention_turns_ready = false;
            if (!filled)
                op_attention_turns_kernel<<<unsigned(rows), 64, 0, value.stream>>>(positions, frequencies, pairs, forward, inverse);
        } else {
            value.attention_turns_ready = false;
        }
        // (FP16 decode: the scores through WMMA, at 512 dims: a wave each 64 of them)
        const dim3 score_grid{unsigned(chunks), unsigned(rows), unsigned((heads + attention_heads - 1) / attention_heads)};
#if defined(__HIPCC__)
        if (value.rdna4 && value.half_decode && dim == 512)
            op_attention_scores_half_kernel<<<score_grid, 256, 0, value.stream>>>(
                q, raw, size_t(raw_rows), positions, size_t(window), compressed, selected, size_t(selections), size_t(heads),
                size_t(dim), scale, pairs, forward, stride, scores, partial);
        else
#endif
        op_attention_scores_kernel<<<score_grid, 256, 0, value.stream>>>(
            q, raw, size_t(raw_rows), positions, size_t(window), compressed, selected, size_t(selections), size_t(heads),
            size_t(dim), scale, pairs, forward, stride, scores, partial);
        check(hipGetLastError());
        float* sums = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scratch) + need);
        float* denominators = sums + size_t(rows * heads) * shares * dim;
        if (shares > 1) op_span(value, sums, share_bytes, "attention scratch");
        const dim3 grid(unsigned(dim / 64), unsigned((heads + attention_out_heads - 1) / attention_out_heads), unsigned(rows * shares));
        // The shares' sums finish in the output kernel's last share of each tile, with counters for
        // up to this many tiles; beyond, in a kernel of their own.
        constexpr size_t tile_capacity = 16384;
        const size_t tiles = size_t(grid.x) * grid.y * size_t(rows);
        if (shares > 1 && tiles <= tile_capacity && !value.attention_tiles) {
            value.attention_tiles = static_cast<unsigned*>(allocate(value, tile_capacity * 4));
            check(hipMemsetAsync(value.attention_tiles, 0, tile_capacity * 4, value.stream));
        }
        unsigned* counters = shares > 1 && tiles <= tile_capacity ? value.attention_tiles : nullptr;
        op_attention_output_kernel<<<grid, 128, share_keys * sizeof(float*), value.stream>>>(
            scores, partial, chunks, raw, size_t(raw_rows), positions, size_t(window), compressed, selected, size_t(selections),
            sinks, size_t(heads), size_t(dim), pairs, inverse, stride, shares, share_keys, sums, denominators,
            out, counters);
        if (shares > 1 && !counters) {
            check(hipGetLastError());
            op_attention_combine_kernel<<<op_blocks(size_t(rows * heads * dim / 4)), 256, 0, value.stream>>>(
                sums, denominators, size_t(rows), shares, size_t(heads), size_t(dim), pairs, inverse, out);
        }
    } else {
        if (shared > 60 * 1024 || rope_dims) throw std::runtime_error("invalid Straydog attention shape");
        op_sparse_attention_kernel<<<unsigned(rows * heads), 256, shared, value.stream>>>(q, raw, size_t(raw_rows), positions,
            size_t(window), compressed, selected, size_t(selections), sinks, size_t(heads), size_t(dim), scale, out);
    }
    SD_OP_END
}

int sd_op_compress(void* handle, const float* kv, const float* gate, uint64_t groups, uint64_t ratio, uint64_t dim, float* out) {
    SD_OP_BEGIN
    op_span(value, kv, size_t(groups * ratio * dim) * 4, "rows to compress");
    if (gate) op_span(value, gate, size_t(groups * ratio * dim) * 4, "a compression gate");
    op_span(value, out, size_t(groups * dim) * 4, "compressed rows");
    op_compress_kernel<<<op_blocks(size_t(groups * dim)), 256, 0, value.stream>>>(kv, gate, size_t(groups), size_t(ratio), size_t(dim), out);
    SD_OP_END
}

int sd_op_index_scores(void* handle, const float* q, const float* keys, const float* weights, const int32_t* visible,
                       uint64_t rows, uint64_t groups, uint64_t heads, uint64_t dim, float scale, float* scores) {
    SD_OP_BEGIN
    op_span(value, q, size_t(rows * heads * dim) * 4, "index queries");
    op_state_span(value, keys, size_t(groups * dim), "index keys");
    op_span(value, weights, size_t(rows * heads) * 4, "index weights");
    op_span(value, visible, size_t(rows) * 4, "visible counts");
    op_span(value, scores, size_t(rows * groups) * 4, "index scores");
    if (heads > 64) throw std::runtime_error("invalid Straydog index shape");
    op_index_scores_kernel<<<dim3(unsigned((groups + 31) / 32), unsigned(rows)), 256, 0, value.stream>>>(q, keys, weights, visible,
        size_t(groups), size_t(heads), size_t(dim), scale, scores);
    SD_OP_END
}

extern "C++" {
// sd_op_index_select's counters: a row's score blocks done (the first index_row_capacity, each zeroed by its
// row's last block), then sd_op_index_select_proj's four (op_index_select_proj_kernel), then from
// index_bins_at the bins of index_bin_rows rows' keys (op_index_select_kernel's, each zeroed by its row's
// last block).
constexpr size_t index_row_capacity = 65536, index_bins_at = index_row_capacity + 8, index_bin_rows = 4;
static unsigned* index_counters(Device& value) {
    constexpr size_t words = index_bins_at + index_bin_rows * top_k_bins;
    if (!value.index_rows) {
        value.index_rows = static_cast<unsigned*>(allocate(value, words * 4));
        check(hipMemsetAsync(value.index_rows, 0, words * 4, value.stream));
    }
    return value.index_rows;
}
#if defined(__HIPCC__)
// Room for ds_index_query_rows's FP16 queries of ds_index_query_capacity rows at most heads (64) of 128 columns: 4 MiB, once.
static _Float16* ds_index_queries(Device& value) {
    if (!value.index_queries) value.index_queries = allocate(value, ds_index_query_capacity * 64 * 128 * 2);
    return static_cast<_Float16*>(value.index_queries);
}
#endif
// The decode and verify passes' index scores through WMMA (index_select_block's half): with FP16 decode
// (half_decode, which the engine sets with the prompt pass's FP16 products: its GEMM policy turns them off for
// a pass of a few rows) or those products, on RDNA4, at its shape.
static bool index_half(const Device& value, uint64_t heads, uint64_t dim) {
    return value.rdna4 && (value.half_decode || value.half_gemm) && heads <= 32 && dim == 128;
}
// A few rows' bins (decode and verify passes), else none.
static unsigned* index_bins(Device& value, uint64_t rows) {
    return rows <= index_bin_rows ? index_counters(value) + index_bins_at : nullptr;
}
// A row's candidate groups (index_candidate's): blocks slots of block groups, or every group without them.
static uint64_t index_scored(const int32_t* candidates, uint64_t rows, uint64_t groups, uint64_t blocks, uint64_t block, Device& value) {
    if (!candidates) return groups;
    if (!blocks || !block || blocks > (uint64_t(1) << 20) || block > 4096) throw std::runtime_error("invalid Straydog index candidates");
    op_span(value, candidates, checked_floats({rows, blocks}), "index candidates");
    return blocks * block;
}
} // extern "C++"

int sd_op_index_candidates(void* handle, const float* scores, const int32_t* visible, uint64_t rows, uint64_t groups, uint64_t blocks,
                           uint64_t block, int32_t* out, float* scratch, uint64_t scratch_bytes) {
    SD_OP_BEGIN
    if (!rows || rows > 65535 || !groups || groups > (uint64_t(1) << 31) || !blocks || blocks > (uint64_t(1) << 20) || !block ||
        block > 4096)
        throw std::runtime_error("invalid Straydog index candidates");
    const uint64_t count = (groups + block - 1) / block;
    op_span(value, scores, checked_floats({rows, groups}), "index scores");
    op_span(value, visible, checked_floats({rows}), "visible counts");
    op_span(value, out, checked_floats({rows, blocks}), "index candidates");
    if (checked_floats({rows, count}) > scratch_bytes) throw std::runtime_error("Straydog index candidates need more scratch");
    op_span(value, scratch, checked_floats({rows, count}), "index candidates' scratch");
    // (each block's top by a thread of many blocks, then each row's top blocks: one block a row reading one value each)
    op_index_block_tops_kernel<<<dim3(unsigned((count + 255) / 256), unsigned(rows)), 256, 0, value.stream>>>(scores, visible, size_t(groups),
        size_t(block), size_t(count), scratch);
    check(hipGetLastError());
    op_top_k_kernel<<<unsigned(rows), 256, 0, value.stream>>>(scratch, size_t(count), size_t(blocks), out);
    SD_OP_END
}

int sd_op_index_select(void* handle, const float* q, const float* keys, const float* weights, const int32_t* visible,
                       uint64_t rows, uint64_t groups, uint64_t heads, uint64_t dim, float scale, uint64_t rope_dims,
                       const int32_t* positions, const float* frequencies, float* scores, uint64_t k, int32_t* out,
                       const int32_t* candidates, uint64_t blocks, uint64_t block) {
    SD_OP_BEGIN
    if (!rows || rows > 65535 || !groups || groups > (uint64_t(1) << 31) || !heads || heads > 64 || !dim || !k ||
        k > (uint64_t(1) << 31) || rope_dims % 2 || rope_dims > dim || rope_dims > 128)
        throw std::runtime_error("invalid Straydog index shape");
    const uint64_t scored = index_scored(candidates, rows, groups, blocks, block, value);
    op_span(value, q, checked_floats({rows, heads, dim}), "index queries");
    op_state_span(value, keys, checked_product({groups, dim}), "index keys");
    op_span(value, weights, checked_floats({rows, heads}), "index weights");
    op_span(value, visible, checked_floats({rows}), "visible counts");
    op_span(value, positions, checked_floats({rows}), "query positions");
    if (rope_dims) op_span(value, frequencies, checked_floats({rope_dims / 2}), "rope frequencies");
    op_span(value, scores, checked_floats({rows, std::max(groups, scored)}), "index scores");
    op_span(value, out, checked_floats({rows, k}), "top-k indices");
#if defined(__HIPCC__)
    // An FP16 prompt pass on RDNA4: the scores through WMMA (ds_index_scores_wmma_rows), every index head
    // of 32 rows and 256 groups a block, then each row's top groups. Decode keeps the kernel below.
    if (value.rdna4 && value.half_gemm && rows >= 32 && dim == 128 && (heads == 32 || heads == 64)) {
        // The queries FP16 and rotated once a row (ds_index_query_rows), then the scores, ds_index_query_capacity rows at a time.
        _Float16* queries = ds_index_queries(value);
        for (uint64_t first = 0; first < rows; first += ds_index_query_capacity) {
            const uint64_t count = std::min<uint64_t>(ds_index_query_capacity, rows - first);
            const dim3 grid{unsigned((groups + ds_index_wide_groups - 1) / ds_index_wide_groups), unsigned((count + ds_index_rows - 1) / ds_index_rows)};
            const float* chunk_q = q + first * heads * dim;
            if (heads == 32) {
                ds_index_query_rows<32, 128><<<unsigned(count), 256, 0, value.stream>>>(chunk_q, size_t(heads), size_t(rope_dims / 2),
                                                                                       positions + first, frequencies, queries);
                ds_index_scores_wmma_rows<2, 128><<<grid, 256, 0, value.stream>>>(queries, keys, weights + first * heads, visible + first,
                                                                               size_t(count), size_t(groups), size_t(heads), scale,
                                                                               scores + first * groups);
            } else {
                ds_index_query_rows<64, 128><<<unsigned(count), 256, 0, value.stream>>>(chunk_q, size_t(heads), size_t(rope_dims / 2),
                                                                                       positions + first, frequencies, queries);
                ds_index_scores_wmma_rows<4, 128><<<grid, 256, 0, value.stream>>>(queries, keys, weights + first * heads, visible + first,
                                                                               size_t(count), size_t(groups), size_t(heads), scale,
                                                                               scores + first * groups);
            }
            check(hipGetLastError());
        }
        // (every group scored, the top k over the candidates)
        if (candidates)
            op_top_k_candidates_kernel<<<unsigned(rows), 256, 0, value.stream>>>(scores, size_t(groups), candidates, size_t(blocks),
                                                                                 size_t(block), size_t(k), out);
        else
            op_top_k_kernel<<<unsigned(rows), 256, 0, value.stream>>>(scores, size_t(groups), size_t(k), out);
        check(hipGetLastError());
        return 0;
    }
#endif
    (heads <= 32 ? op_index_select_kernel<32> : op_index_select_kernel<64>)<<<dim3(unsigned((scored + index_block_groups - 1) / index_block_groups),
        unsigned(rows)), 256, 0, value.stream>>>(q, keys, weights, visible, size_t(groups), size_t(heads), size_t(dim), scale,
        size_t(rope_dims / 2), positions, frequencies, scores, size_t(k), out, index_counters(value), index_bins(value, rows), candidates,
        size_t(blocks), size_t(block), index_half(value, heads, dim));
    SD_OP_END
}

int sd_op_index_select_proj(void* handle, const void* proj_weights, int proj_kind, uint64_t proj_cols, const float* x,
                            const float* q, const float* keys, float* weights, const int32_t* visible, uint64_t groups,
                            uint64_t heads, uint64_t dim, float scale, uint64_t rope_dims, const int32_t* positions,
                            const float* frequencies, float* scores, uint64_t k, int32_t* out, const int32_t* candidates,
                            uint64_t blocks, uint64_t block) {
    SD_OP_BEGIN
    if (!proj_weights || !proj_cols || proj_cols > 8192 || !groups || groups > (uint64_t(1) << 31) || !heads || heads > 64 || !dim ||
        !k || k > (uint64_t(1) << 31) || rope_dims % 2 || rope_dims > dim || rope_dims > 128)
        throw std::runtime_error("invalid Straydog index shape");
    const size_t row_bytes = matrix_row_bytes(proj_kind, size_t(proj_cols));
    op_span(value, proj_weights, checked_product({heads, row_bytes}), "a weight matrix");
    op_span(value, x, checked_floats({proj_cols}), "a linear input");
    op_span(value, q, checked_floats({heads, dim}), "index queries");
    op_state_span(value, keys, checked_product({groups, dim}), "index keys");
    op_span(value, weights, checked_floats({heads}), "index weights");
    op_span(value, visible, 4, "visible counts");
    op_span(value, positions, 4, "query positions");
    if (rope_dims) op_span(value, frequencies, checked_floats({rope_dims / 2}), "rope frequencies");
    const uint64_t scored = index_scored(candidates, 1, groups, blocks, block, value);
    op_span(value, scores, checked_floats({scored}), "index scores");
    op_span(value, out, checked_floats({k}), "top-k indices");
    // The projection as sd_op_linear runs it for one vector (gemv: its lanes, widened to op_lanes), in the
    // scores' launch where that is gemv_fixed_kind<30, 1> (BF16 rows too few for the shared-input launch).
    const Matrix matrix{static_cast<const uint8_t*>(proj_weights), row_bytes, size_t(heads), size_t(proj_cols), 0, 0, 0, proj_kind};
    unsigned lanes = gemv_lanes(matrix, x);
    size_t alignment = 0;
    if (lanes == 2 && op_lanes > 2 && matrix.cols / gemv_unit(matrix.kind, alignment) >= 32) lanes = std::min(32u, op_lanes);
    const size_t waves = lanes ? gemv_waves(matrix, lanes) : 0;
    if (proj_kind == 30 && lanes && waves * (32 / lanes) < 2048) {
        FixedGemv list{};
        list.items[0] = {matrix, 0, lanes};
        list.count = 1;
        const unsigned proj_blocks = unsigned((waves + 7) / 8), chunks = unsigned((scored + index_block_groups - 1) / index_block_groups);
        (heads <= 32 ? op_index_select_proj_kernel<30, 32> : op_index_select_proj_kernel<30, 64>)<<<proj_blocks + chunks, 256, 0,
            value.stream>>>(list, x, proj_blocks, q, keys, weights, visible,
            size_t(groups), size_t(heads), size_t(dim), scale, size_t(rope_dims / 2), positions, frequencies, scores, size_t(k), out,
            index_counters(value) + index_row_capacity, index_bins(value, 1), candidates, size_t(blocks), size_t(block),
            index_half(value, heads, dim));
        check(hipGetLastError());
        return 0;
    }
    gemv(value, &matrix, 1, x, weights, 1, 0, 0, {}, op_lanes);
    (heads <= 32 ? op_index_select_kernel<32> : op_index_select_kernel<64>)<<<dim3(unsigned((scored + index_block_groups - 1) / index_block_groups),
        1), 256, 0, value.stream>>>(q, keys, weights, visible, size_t(groups), size_t(heads), size_t(dim), scale, size_t(rope_dims / 2),
        positions, frequencies, scores, size_t(k), out, index_counters(value), index_bins(value, 1), candidates, size_t(blocks),
        size_t(block), index_half(value, heads, dim));
    SD_OP_END
}

int sd_op_top_k(void* handle, const float* scores, uint64_t rows, uint64_t groups, uint64_t k, int32_t* out) {
    SD_OP_BEGIN
    if (!rows || !groups || !k || groups > (size_t(1) << 31) || k > (size_t(1) << 31)) throw std::runtime_error("invalid Straydog top-k shape");
    op_span(value, scores, size_t(rows * groups) * 4, "scores");
    op_span(value, out, size_t(rows * k) * 4, "top-k indices");
    op_top_k_kernel<<<unsigned(rows), 256, 0, value.stream>>>(scores, size_t(groups), size_t(k), out);
    SD_OP_END
}

// out = the sum of every device's part, in device order (so every device adds the same values the
// same way), the other devices' parts read over the peer link at system scope (never a stale line).
constexpr size_t sum_devices_max = 8;
struct DeviceParts { const float* parts[sum_devices_max]; };
__global__ void sum_devices_kernel(DeviceParts list, unsigned devices, size_t count, float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float sum = link_load(list.parts[0] + index);
    for (unsigned d = 1; d < devices; ++d) sum += link_load(list.parts[d] + index);
    out[index] = sum;
}

extern "C++" {
// Peer access, once per pair (a device without it cannot take part).
static void link_peers(const std::vector<Device*>& list) {
    static std::mutex mutex;
    static std::vector<std::pair<int, int>> linked;
    std::lock_guard<std::mutex> lock(mutex);
    for (Device* a : list)
        for (Device* b : list) {
            if (a == b || std::find(linked.begin(), linked.end(), std::make_pair(a->ordinal, b->ordinal)) != linked.end()) continue;
            int can = 0;
            check(hipDeviceCanAccessPeer(&can, a->ordinal, b->ordinal));
            if (!can) throw std::runtime_error("Straydog devices without peer access cannot sum their parts");
            on_device(a->ordinal, [&]() {
                const hipError_t state = hipDeviceEnablePeerAccess(b->ordinal, 0);
                if (state == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
                else check(state);
            });
            linked.emplace_back(a->ordinal, b->ordinal);
        }
}
} // extern "C++"

int sd_op_sum_devices(void* const* handles, float* const* outs, const float* const* parts, uint64_t devices, uint64_t count) {
    try {
        if (!handles || !outs || !parts || devices < 1 || devices > sum_devices_max || !count)
            throw std::runtime_error("invalid Straydog device sum");
        std::vector<Device*> list;
        DeviceParts all{};
        for (size_t i = 0; i < devices; ++i) {
            Device& d = device(handles[i]);
            for (Device* other : list) if (other == &d) throw std::runtime_error("a device twice in a Straydog device sum");
            op_span(d, parts[i], size_t(count) * 4, "a device's part");
            op_span(d, outs[i], size_t(count) * 4, "a device's sum");
            for (size_t j = 0; j < devices; ++j)
                if (static_cast<const void*>(outs[i]) == static_cast<const void*>(parts[j]))
                    throw std::runtime_error("a device sum cannot overwrite a part");
            list.push_back(&d);
            all.parts[i] = parts[i];
        }
        // Two devices with a sum link and a sum that fits it: over the link, on the devices (device_link.hpp). No host event, so a graph
        // can record it, and no read over the bus of the other's part.
        if (devices == 2 && list[0]->sums.peer == list[1] && size_t(count) <= list[0]->sums.capacity) {
            for (size_t i = 0; i < 2; ++i)
                on_device(list[i]->ordinal, [&]() {
                    if (sd_op_link_sum(handles[i], parts[i], count, outs[i], 0)) throw std::runtime_error(last_error);
                });
            // Either link sum may wait for the other device's flag.
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
            return 0;
        }
        link_peers(list);
        // Every part's end is marked before any device waits, so no device waits for another's sum.
        for (Device* d : list)
            on_device(d->ordinal, [&]() {
                if (!d->parts_ready) check(hipEventCreateWithFlags(&d->parts_ready, hipEventDisableTiming));
                check(hipEventRecord(d->parts_ready, d->stream));
            });
        // A consumer's stream waits for each peer's parts_ready event below.
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        for (size_t i = 0; i < devices; ++i)
            on_device(list[i]->ordinal, [&]() {
                for (size_t j = 0; j < devices; ++j)
                    if (j != i) check(hipStreamWaitEvent(list[i]->stream, list[j]->parts_ready, 0));
                sum_devices_kernel<<<unsigned((count + 255) / 256), 256, 0, list[i]->stream>>>(all, unsigned(devices), size_t(count),
                                                                                             outs[i]);
                check(hipGetLastError());
                if (!list[i]->parts_read) check(hipEventCreateWithFlags(&list[i]->parts_read, hipEventDisableTiming));
                check(hipEventRecord(list[i]->parts_read, list[i]->stream));
            });
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Every device's later work waits until each other device's last sd_op_sum_devices has read its
// part: before a device writes a summed part's buffer again (another device's sum reads it there).
int sd_devices_settle(void* const* handles, uint64_t devices) {
    try {
        if (!handles || devices < 1 || devices > sum_devices_max) throw std::runtime_error("invalid Straydog devices");
        std::vector<Device*> list;
        for (size_t i = 0; i < devices; ++i) list.push_back(&device(handles[i]));
        for (Device* d : list)
            on_device(d->ordinal, [&]() {
                for (Device* other : list)
                    if (other != d && other->parts_read) check(hipStreamWaitEvent(d->stream, other->parts_read, 0));
            });
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Every device's `out` (rows of `full` FP16 values) = each device's `part` (rows of its `share` columns,
// FP16) at that device's columns `firsts[d]`.., the other devices' read over the peer link at system
// scope as sd_op_sum_devices reads them; their bits copied, nothing summed (the columns are disjoint).
struct DeviceHalves { const uint32_t* parts[sum_devices_max]; uint64_t firsts[sum_devices_max]; };
__global__ void gather_columns_kernel(DeviceHalves list, unsigned devices, size_t rows, size_t share, size_t full, uint32_t* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x, pairs = full / 2;
    if (index >= rows * pairs) return;
    const size_t row = index / pairs, column = (index % pairs) * 2;
    for (unsigned d = 0; d < devices; ++d) {
        if (column < list.firsts[d] || column >= list.firsts[d] + share) continue;
        const uint32_t* at = list.parts[d] + (row * share + column - list.firsts[d]) / 2;
#if defined(__HIPCC__)
        out[index] = __hip_atomic_load(at, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        out[index] = *static_cast<const volatile uint32_t*>(at);
#endif
        return;
    }
    out[index] = 0;
}

int sd_op_gather_columns(void* const* handles, void* const* outs, const void* const* parts, const uint64_t* firsts, uint64_t devices,
                         uint64_t rows, uint64_t share, uint64_t full) {
    try {
        if (!handles || !outs || !parts || !firsts || devices < 1 || devices > sum_devices_max || !rows || !share || share % 2 ||
            full % 2 || share > full)
            throw std::runtime_error("invalid Straydog column gather");
        std::vector<Device*> list;
        DeviceHalves all{};
        for (size_t i = 0; i < devices; ++i) {
            Device& d = device(handles[i]);
            for (Device* other : list) if (other == &d) throw std::runtime_error("a device twice in a Straydog column gather");
            if (firsts[i] % 2 || firsts[i] + share > full) throw std::runtime_error("invalid Straydog column gather");
            for (size_t j = 0; j < i; ++j)
                if (firsts[i] < firsts[j] + share && firsts[j] < firsts[i] + share)
                    throw std::runtime_error("a Straydog column gather's shares overlap");
            op_span(d, parts[i], size_t(rows * share) * 2, "a device's columns");
            op_span(d, outs[i], size_t(rows * full) * 2, "a device's gathered rows");
            for (size_t j = 0; j < devices; ++j)
                if (outs[i] == parts[j]) throw std::runtime_error("a column gather cannot overwrite a part");
            list.push_back(&d);
            all.parts[i] = static_cast<const uint32_t*>(parts[i]);
            all.firsts[i] = firsts[i];
        }
        // (the devices have peer access already: a sum of theirs set it up, as every pass of several
        // devices sums its attention's parts before its experts')
        for (Device* d : list)
            on_device(d->ordinal, [&]() {
                if (!d->parts_ready) check(hipEventCreateWithFlags(&d->parts_ready, hipEventDisableTiming));
                check(hipEventRecord(d->parts_ready, d->stream));
            });
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        for (size_t i = 0; i < devices; ++i)
            on_device(list[i]->ordinal, [&]() {
                for (size_t j = 0; j < devices; ++j)
                    if (j != i) check(hipStreamWaitEvent(list[i]->stream, list[j]->parts_ready, 0));
                gather_columns_kernel<<<unsigned((rows * full / 2 + 255) / 256), 256, 0, list[i]->stream>>>(
                    all, unsigned(devices), size_t(rows), size_t(share), size_t(full), static_cast<uint32_t*>(outs[i]));
                check(hipGetLastError());
                if (!list[i]->parts_read) check(hipEventCreateWithFlags(&list[i]->parts_read, hipEventDisableTiming));
                check(hipEventRecord(list[i]->parts_read, list[i]->stream));
            });
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Every device's `out` (rows of `full` F32 values) = at each device's columns firsts[d].. (`share` of
// them) that device's `part` (rows of `full` values, its other columns unread), the others' read over
// the peer link at system scope as sd_op_sum_devices reads them: the devices' disjoint columns of one
// result. The sum of parts whose other columns are zero, bit for bit but for a zero's sign, with only
// the others' columns over the bus (the sum read every part whole, half of it zeros with two devices).
struct DeviceColumns { const float* parts[sum_devices_max]; uint64_t firsts[sum_devices_max]; };
__global__ void gather_owned_kernel(DeviceColumns list, unsigned devices, unsigned self, size_t count, size_t share, size_t full,
                                    float* out) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t column = index % full;
    for (unsigned d = 0; d < devices; ++d) {
        if (column - list.firsts[d] >= share) continue;   // (unsigned: a column before firsts[d] too)
        out[index] = d == self ? list.parts[d][index] : link_load(list.parts[d] + index);
        return;
    }
    out[index] = 0;
}

int sd_op_gather_owned(void* const* handles, float* const* outs, const float* const* parts, const uint64_t* firsts, uint64_t devices,
                       uint64_t rows, uint64_t share, uint64_t full) {
    try {
        if (!handles || !outs || !parts || !firsts || devices < 1 || devices > sum_devices_max || !rows || !share || share > full ||
            rows > (uint64_t(1) << 40) / full)
            throw std::runtime_error("invalid Straydog column gather");
        const size_t count = size_t(rows * full);
        std::vector<Device*> list;
        DeviceColumns all{};
        for (size_t i = 0; i < devices; ++i) {
            Device& d = device(handles[i]);
            for (Device* other : list) if (other == &d) throw std::runtime_error("a device twice in a Straydog column gather");
            if (firsts[i] + share > full) throw std::runtime_error("invalid Straydog column gather");
            for (size_t j = 0; j < i; ++j)
                if (firsts[i] < firsts[j] + share && firsts[j] < firsts[i] + share)
                    throw std::runtime_error("a Straydog column gather's shares overlap");
            op_span(d, parts[i], count * 4, "a device's part");
            op_span(d, outs[i], count * 4, "a device's gathered rows");
            for (size_t j = 0; j < devices; ++j)
                if (static_cast<const void*>(outs[i]) == static_cast<const void*>(parts[j]))
                    throw std::runtime_error("a column gather cannot overwrite a part");
            list.push_back(&d);
            all.parts[i] = parts[i];
            all.firsts[i] = firsts[i];
        }
        // What the sum link takes goes over it as a sum (no host event, so a graph can record it).
        if (devices == 2 && list[0]->sums.peer == list[1] && count <= list[0]->sums.capacity)
            return sd_op_sum_devices(handles, outs, parts, devices, rows * full);
        link_peers(list);
        for (Device* d : list)
            on_device(d->ordinal, [&]() {
                if (!d->parts_ready) check(hipEventCreateWithFlags(&d->parts_ready, hipEventDisableTiming));
                check(hipEventRecord(d->parts_ready, d->stream));
            });
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        for (size_t i = 0; i < devices; ++i)
            on_device(list[i]->ordinal, [&]() {
                for (size_t j = 0; j < devices; ++j)
                    if (j != i) check(hipStreamWaitEvent(list[i]->stream, list[j]->parts_ready, 0));
                gather_owned_kernel<<<unsigned((count + 255) / 256), 256, 0, list[i]->stream>>>(all, unsigned(devices), unsigned(i), count,
                                                                                                size_t(share), size_t(full), outs[i]);
                check(hipGetLastError());
                if (!list[i]->parts_read) check(hipEventCreateWithFlags(&list[i]->parts_read, hipEventDisableTiming));
                check(hipEventRecord(list[i]->parts_read, list[i]->stream));
            });
#if !defined(__HIPCC__)
        for (Device* d : list) on_device(d->ordinal, [&]() { flush_launches(*d); });
#endif
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

constexpr size_t slot_route_capacity = 65536;
// An expert's three matrices in its slot (rows of each as `part` takes them), checked to lie inside it.
extern "C++" {
static SlotExperts slot_plan(uint64_t slot_bytes, const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn,
                             const sd_expert_split& part) {
    SlotExperts plan{};
    // (hidden and ffn bounded, the rows a split's: a matrix's bytes cannot wrap)
    if (hidden > (uint64_t(1) << 24) || ffn > (uint64_t(1) << 24) || part.hidden_rows > hidden || part.ffn_rows > ffn)
        throw std::runtime_error("invalid Straydog experts shape");
    for (int role = 0; role < 3; ++role) {
        const size_t matrix_rows = size_t(role == 2 ? part.hidden_rows : part.ffn_rows), cols = size_t(role == 2 ? ffn : hidden);
        const size_t row_bytes = matrix_row_bytes(kinds[role], cols);
        if (offsets[role] > slot_bytes || checked_product({matrix_rows, row_bytes}) > slot_bytes - offsets[role])
            throw std::runtime_error("an expert's matrices lie outside its slot");
        plan.role[role] = {reinterpret_cast<const uint8_t*>(offsets[role]), row_bytes, matrix_rows, cols, 0, 0, 0, kinds[role]};
    }
    return plan;
}
} // extern "C++"

int sd_op_experts(void* handle, const uint64_t* bases, uint64_t slot_bytes, const uint64_t* offsets, const int* kinds,
                  uint64_t hidden, uint64_t ffn, uint64_t rows, uint64_t top_k, const float* input, uint64_t input_stride,
                  const float* weights, float clamp, int accumulate, float* product, float* out, uint64_t out_stride,
                  const sd_expert_split* split) {
    SD_OP_BEGIN
    input_stride = input_stride ? input_stride : hidden;
    out_stride = out_stride ? out_stride : hidden;
    const sd_expert_split whole{0, ffn, 0, hidden, 3};
    const sd_expert_split& part = split ? *split : whole;
    if (!offsets || !kinds || !hidden || !ffn || !rows || !top_k || top_k > 64 || rows > 65535 ||
        rows * top_k > slot_route_capacity || input_stride < hidden || out_stride < hidden || !(clamp >= 0.0f) ||
        part.stages < 1 || part.stages > 3 || !part.ffn_rows || part.ffn_first > ffn - part.ffn_rows || !part.hidden_rows ||
        part.hidden_first > hidden - part.hidden_rows || (part.more && (!part.send || part.stages != 1)) ||
        (part.send_at && part.stages != 1))
        throw std::runtime_error("invalid Straydog experts shape");
    // The input's and out's floats ((rows - 1) * stride + hidden) and the product's, checked.
    const uint64_t input_floats = checked_sum({checked_product({rows - 1, input_stride}), hidden});
    const uint64_t out_floats = checked_sum({checked_product({rows - 1, out_stride}), hidden});
    const uint64_t product_floats = checked_product({rows, top_k, ffn});
    // (sending: the last stage's values also go to the other device, its flag raised after them)
    const LinkTail tail = part.send ? part_send_tail(value, part.send - 1) : LinkTail{};
    if (part.send && (part.stages == 1 ? part.send_at > tail.capacity || product_floats > tail.capacity - part.send_at
                                       : out_floats > tail.capacity))
        throw std::runtime_error("an experts part exceeds its inbox");
    SlotExperts plan = slot_plan(slot_bytes, offsets, kinds, hidden, ffn, part);
    const size_t routes = size_t(rows * top_k);
    if (!bases && (value.inline_routes_count ? value.inline_routes_count : value.slot_routes_count) != routes)
        throw std::runtime_error("no Straydog expert routes to reuse");
    bool uniform = bases != nullptr && top_k == 1;
    for (size_t route = 0; bases && route < routes; ++route) {
        uniform &= bases[route] == bases[0];
        if (!bases[route]) continue;   // a route another device runs
        if (bases[route] % 256) throw std::runtime_error("an expert slot is not 256-byte aligned");
        op_span(value, reinterpret_cast<const void*>(bases[route]), size_t(slot_bytes), "an expert slot");
    }
    if (part.stages & 1) {
        op_span(value, input, checked_floats({input_floats}), "an expert input");
        if (weights) op_span(value, weights, size_t(rows * top_k) * 4, "route weights");
    }
    op_span(value, product, checked_floats({product_floats}), "an expert product");
    if (part.stages & 2) op_span(value, out, checked_floats({out_floats}), "an expert output");
    plan.top_k = uint32_t(top_k);
    plan.product_stride = uint32_t(ffn);
    plan.product_first = uint32_t(part.ffn_first);
    plan.out_first = uint32_t(part.hidden_first);
    plan.clamp = clamp;
    plan.accumulate = accumulate ? 1 : 0;
    if (!value.slot_routes) value.slot_routes = static_cast<uint64_t*>(allocate(value, slot_route_capacity * 8));
    // One slot for a token's only route (a shared expert) goes by value, as do a few routes (a
    // decode row's) in the launch's arguments; more routes are uploaded. Without bases, the last
    // routes passed are used again (the down stage after its gate/up).
    static_assert(sizeof(value.inline_routes) == slot_inline_routes * 8, "inline routes");
    if (uniform && bases[0]) {
        plan.uniform = bases[0];
    } else if (bases && routes <= slot_inline_routes) {
        std::copy(bases, bases + routes, value.inline_routes);
        value.inline_routes_count = routes;
    } else if (bases) {
        upload(value, value.slot_routes, bases, routes * 8);
        value.slot_routes_count = routes;
        value.inline_routes_count = 0;
    }
    if (!plan.uniform && value.inline_routes_count) {
        std::copy(value.inline_routes, value.inline_routes + value.inline_routes_count, plan.inline_bases);
        plan.inline_count = uint32_t(value.inline_routes_count);
        // (a pass of a few rows: an expert's routes together)
        if (bases && rows >= 2 && rows <= 4 && (part.stages & 1) && !part.send) {
            uint32_t taken = 0;
            for (size_t route = 0; route < routes; ++route) {
                if (!bases[route] || (taken >> route) & 1) continue;
                uint32_t list = 0xffffff00u | uint32_t(route);
                for (size_t other = route + 1, n = 1; other < routes && n < 4; ++other)
                    if (bases[other] == bases[route]) {
                        list = (list & ~(0xffu << (8 * n))) | uint32_t(other) << (8 * n);
                        taken |= 1u << other;
                        ++n;
                    }
                plan.group_routes[plan.groups++] = list;
            }
        }
    }
    if (!part.send || !slot_experts(value, plan, part.stages, value.slot_routes, size_t(rows), input, size_t(input_stride), weights,
                                    product, out, size_t(out_stride), &tail, size_t(part.send_at), part.more ? 0 : 1)) {
        slot_experts(value, plan, part.stages, value.slot_routes, size_t(rows), input, size_t(input_stride), weights, product, out,
                     size_t(out_stride));
        if (part.send)
            part_send(value, part.send - 1, out, size_t(rows), size_t(out_stride), size_t(part.hidden_first), size_t(part.hidden_rows));
    }
    SD_OP_END
}

int sd_expert_table(void* handle, uint64_t* table, const uint64_t* bases, uint64_t count, uint64_t slot_bytes) {
    SD_OP_BEGIN
    if (!bases || !count || count > (uint64_t(1) << 24) || !slot_bytes) throw std::runtime_error("invalid Straydog expert table");
    op_span(value, table, size_t(count) * 8, "an expert table");
    for (size_t index = 0; index < count; ++index) {
        if (!bases[index]) continue;   // no expert here: another device runs it
        if (bases[index] % 256) throw std::runtime_error("an expert slot is not 256-byte aligned");
        op_span(value, reinterpret_cast<const void*>(bases[index]), size_t(slot_bytes), "an expert slot");
    }
    upload(value, table, bases, size_t(count) * 8);
    SD_OP_END
}

int sd_op_experts_table(void* handle, const uint64_t* table, uint64_t count, const int32_t* ids, uint64_t slot_bytes,
                        const uint64_t* offsets, const int* kinds, uint64_t hidden, uint64_t ffn, uint64_t rows, uint64_t top_k,
                        const float* input, const float* weights, float clamp, int accumulate, float* product, float* out) {
    SD_OP_BEGIN
    if (!offsets || !kinds || !hidden || !ffn || !rows || !top_k || top_k > 64 || rows > 65535 || !count ||
        count > (uint64_t(1) << 24) || !(clamp >= 0.0f))
        throw std::runtime_error("invalid Straydog experts shape");
    const sd_expert_split whole{0, ffn, 0, hidden, 3};
    SlotExperts plan = slot_plan(slot_bytes, offsets, kinds, hidden, ffn, whole);
    op_span(value, table, size_t(count) * 8, "an expert table");
    op_span(value, ids, checked_floats({rows, top_k}), "expert routes");
    op_span(value, input, checked_floats({rows, hidden}), "an expert input");
    if (weights) op_span(value, weights, checked_floats({rows, top_k}), "route weights");
    op_span(value, product, checked_floats({rows, top_k, ffn}), "an expert product");
    op_span(value, out, checked_floats({rows, hidden}), "an expert output");
    plan.top_k = uint32_t(top_k);
    plan.product_stride = uint32_t(ffn);
    plan.clamp = clamp;
    plan.accumulate = accumulate ? 1 : 0;
    plan.table = table;
    plan.ids = ids;
    plan.table_count = uint32_t(count);
    slot_experts(value, plan, 3, nullptr, size_t(rows), input, size_t(hidden), weights, product, out, size_t(hidden));
    SD_OP_END
}

int sd_expert_rows(void* handle, void* const* banks, const int* kinds, uint64_t experts, uint64_t hidden, uint64_t ffn,
                   uint64_t steps, uint64_t top_k, const int32_t* selected, int path, const float* input,
                   const float* down_input, float* gate_up, float* product, float* output, double* timings) {
    try {
        expert_rows(device(handle), banks, kinds, size_t(experts), size_t(hidden), size_t(ffn), size_t(steps), size_t(top_k),
                    selected, path, input, down_input, gate_up, product, output, timings);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
} // extern C
#include "token_mixing.hpp"
#include "device_link.hpp"
#include "qwen_parts.hpp"
#include "gemv_fused.hpp"
#include "dspark.hpp"
#include "dspark_inject.hpp"
#include "moe_prefill.hpp"
#include "host_exchange.hpp"
#include "vision.hpp"
#include "dspark_attention.hpp"
