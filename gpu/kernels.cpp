// Straydog's independent, correctness-first GPU matrix/vector backend.
// HIP and CUDA are selected computational dependencies, not inference engines.
// Q2_0 format: FP16 scale + 16 packed bytes, 64 LSB-first two-bit values,
// decoded as scale * (code - 1).
#include "straydog_gpu.h"
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
#define HIP_R_16F CUDA_R_16F
#define HIP_R_32F CUDA_R_32F
#define HIPBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define HIPBLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
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
#define hipHostMallocPortable cudaHostAllocPortable
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventBlockingSync cudaEventBlockingSync
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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
thread_local std::string last_error;
constexpr size_t max_cols = 16384, max_rows = 262144;
constexpr size_t max_batch = 256;
struct Matrix {
    const uint8_t* data;
    size_t row_bytes, rows, cols, input_offset, output_offset, launch_offset;
    int kind;
};
// One GEMV row range: lanes cooperating per row (0 selects the reference decoder).
struct GemvItem { Matrix matrix; uint32_t first_wave, lanes; };
// Every GGML type the GPU kernels decode (weight<kind>); dispatch sites expand it.
#define SD_FORMATS(X) X(0) X(1) X(2) X(6) X(8) X(10) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) \
    X(29) X(30) X(39) X(42)
// Allocations carry a small tail so aligned word loads of two-byte-aligned
// quant blocks never cross the end of an owned buffer.
constexpr size_t allocation_tail = 64;
struct CachedGraph {
    hipGraph_t graph=nullptr;
    hipGraphExec_t executable=nullptr;
    std::vector<void*> dependencies;
};
struct ResidentMoe {
    std::vector<Matrix> weights;
    std::vector<void*> dependencies;
    void* table=nullptr;
    float* workspace=nullptr;
    size_t experts=0, capacity=0;
    // Partial residency: routed experts live in device slots (null data when absent)
    // and are otherwise read from the authenticated host mapping.
    bool hybrid=false;
    const uint8_t* host[3]{};
    size_t host_bytes[3]{};
    std::vector<int32_t> slot_of;
    std::vector<double> usage;
    const uint8_t* mapped[3]{};   // device view of page-locked host banks (zero-copy assist)
    // Expert parallelism: this device holds only some routed experts (the
    // others have null data and are skipped) and computes a partial combination.
    bool part=false, shared=true;
    std::vector<std::chrono::steady_clock::time_point> prefetched;   // last read-ahead hint per expert
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
    size_t cache=0, stage=0, bytes=0;
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
};
constexpr size_t gemm_elements = 32 * 1024 * 1024, blas_workspace_bytes = 16 * 1024 * 1024;
constexpr size_t io_arena_bytes = size_t(32) << 20;   // page-locked transfer staging per device

void check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}

void blas_check(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Straydog BLAS failure: " + std::to_string(int(status)));
}

void cleanup(hipError_t error) noexcept {
    if (error != hipSuccess) std::fprintf(stderr, "Straydog GPU cleanup: %s\n", hipGetErrorString(error));
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

struct KvMirror { float* data; size_t valid; };
struct Device;
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
struct Device {
    int ordinal;
    size_t budget = 0, used = 0;
    size_t batch_capacity = 32, input_capacity = 32 * max_cols, output_capacity = max_rows;
    size_t gemm_min_batch = 8;
    bool half_gemm = false;
    bool rdna4 = false;
    // A grouped FP16 expert GEMM for prefill exists: RDNA4 WMMA (HIP) or tensor cores (CUDA sm_70+).
    bool expert_prefill = false;
    // Decode work is captured and replayed as graphs: RDNA4 (HIP) and every CUDA device.
    bool graph_capture = false;
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
    std::unordered_map<size_t, void*> helper_moe;   // layer -> this device's expert part, when it helps the owner
    uint8_t* io = nullptr;   // page-locked staging for caller-memory transfers (host_io.hpp)
    size_t io_used = 0;
    hipEvent_t io_halves[2] = {nullptr, nullptr};   // a large upload's last copy out of each half
    float* input = nullptr;
    float* output = nullptr;
    GemvItem* matrices = nullptr;
    hipblasHandle_t blas = nullptr;
    float* expanded = nullptr;
    __half* half_input = nullptr;
    float* attention_scores = nullptr;
    void* blas_workspace = nullptr;
    unsigned* flags = nullptr;
    // Deferred prefill (sd_prefill_defer): spans end without a host wait; their non-finite
    // check accumulates in flags[1] until sd_prefill_settle. `handoff` orders a row handoff
    // between this device's stream and its peer's.
    bool prefill_deferred = false;
    hipEvent_t handoff = nullptr;
    // Host states allocated beside reading the weights (sd_host_states_prefetch): zeroing and
    // page-locking ~13 GB of K/V. The thread alone writes `host_ready` until joined.
    std::thread host_prefetch;
    std::vector<std::pair<size_t, void*>> host_ready;
    std::string host_prefetch_error;
    // Page-locked staging for a deferred prompt's MTP embeddings, two batches deep
    // (sd_mtp_embed): a batch's copy waits only for the copy two batches before.
    float* mtp_staging[2] = {nullptr, nullptr};
    hipEvent_t mtp_staged[2] = {nullptr, nullptr};
    size_t mtp_staging_floats = 0;
    unsigned mtp_turn = 0;
    uint64_t gemm_calls = 0, gemm_vectors = 0;
    std::unordered_map<void*, size_t> allocations;
    // Allocations that live in mapped, page-locked host memory (sd_host_state): device pointer -> host pointer.
    std::unordered_map<void*, void*> host_states;
    // Page-locked host copies of recurrent sequence state (conversation checkpoints): pointer -> bytes.
    // Neither VRAM nor mapped state, so they take no budget and no K/V mirror rows.
    std::unordered_map<void*, size_t> checkpoints;
    // VRAM rows that multi-token prefill uses for host-resident K/V (sd_kv_stage): the copy holds
    // rows [0, kv_stage_valid) of kv_stage_owner's K/V in a kv_stage_rows-row layout.
    float* kv_stage = nullptr;
    size_t kv_stage_rows = 0, kv_stage_valid = 0;
    void* kv_stage_owner = nullptr;
    // Decode partitions the same stage into fixed per-host-state F32 prefixes.
    // Valid ends are prepared outside graph replay; pointers stay fixed until re-registration.
    size_t kv_mirror_rows = 0;
    std::unordered_map<void*,KvMirror> kv_mirrors;
    std::unordered_map<std::string,CachedGraph> graphs;
    std::unordered_map<void*,std::unique_ptr<ResidentMoe>> resident;
    std::unordered_map<size_t,DecoderLayer> decoder;
    uint64_t graph_replays=0;
    ExpertSlots slots;
    hipStream_t copy_stream=nullptr;
    void* cpu=nullptr;
    float* exchange=nullptr;      // pinned host: selected, probabilities, activation, CPU result
    float* cpu_partial=nullptr;   // device copy of the CPU expert result
    float* exchange_device=nullptr;   // the pinned exchange as seen by kernels
    uint64_t cpu_experts=0, gpu_experts=0, assisted_experts=0, streamed_experts=0, swapped_experts=0, hybrid_tokens=0;
    uint32_t assist=0;
    uint64_t hybrid_wait_ns=0, hybrid_cpu_ns=0;
    uint32_t hybrid_sequence=0;
    size_t hybrid_position=SIZE_MAX;
    hipEvent_t swapped=nullptr;   // the pending swaps' copies have landed
    std::unique_ptr<Background> copier;   // issues bounded swaps' copies (hybrid_rebalance)
#if defined(__linux__)
    std::unique_ptr<ReadAhead> read_ahead;   // started by the first expert read-ahead hint
#endif
    explicit Device(int value) : ordinal(value) {}
    ~Device() {
        cleanup(hipSetDevice(ordinal));
        if (host_prefetch.joinable()) host_prefetch.join();
        for (auto& ready : host_ready) cleanup(hipHostFree(ready.second));
        copier.reset();
        if (stream) cleanup(hipStreamSynchronize(stream));
        if (helper_stream) cleanup(hipStreamSynchronize(helper_stream));
        for (auto& entry : graphs) {
            cleanup(hipGraphExecDestroy(entry.second.executable));
            cleanup(hipGraphDestroy(entry.second.graph));
        }
        decoder.clear(); resident.clear();
        if (slots.base) cleanup(hipFree(slots.base));
        if (slots.host) cleanup(hipHostFree(slots.host));
        if (exchange) cleanup(hipHostFree(exchange));
        if (cpu_partial) cleanup(hipFree(cpu_partial));
        if (copy_stream) cleanup(hipStreamDestroy(copy_stream));
        if (swapped) cleanup(hipEventDestroy(swapped));
        if (blas && hipblasDestroy(blas) != HIPBLAS_STATUS_SUCCESS)
            std::fprintf(stderr, "Straydog BLAS cleanup failed\n");
        if (expanded) cleanup(hipFree(expanded));
        if (half_input) cleanup(hipFree(half_input));
        if (attention_scores) cleanup(hipFree(attention_scores));
        if (blas_workspace) cleanup(hipFree(blas_workspace));
        if (flags) cleanup(hipFree(flags));
        if (handoff) cleanup(hipEventDestroy(handoff));
        for (int i = 0; i < 2; ++i) {
            if (mtp_staged[i]) { cleanup(hipEventSynchronize(mtp_staged[i])); cleanup(hipEventDestroy(mtp_staged[i])); }
            if (mtp_staging[i]) cleanup(hipHostFree(mtp_staging[i]));
        }
        for (auto entry : allocations) {
            const auto host = host_states.find(entry.first);
            cleanup(host == host_states.end() ? hipFree(entry.first) : hipHostFree(host->second));
        }
        for (auto entry : checkpoints) cleanup(hipHostFree(entry.first));
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
    }
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
__constant__ SD_ALIGN(16) uint8_t iq_grid2xxs[256 * 8], iq_grid2xs[512 * 8], iq_grid2s[1024 * 8];
__constant__ SD_ALIGN(16) uint8_t iq_grid3xxs[256 * 4], iq_grid3s[512 * 4], iq_grid1s[2048 * 8];
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

// values (optional) holds the rows' values instead of the packed input.
__global__ void write_attention_kv(float* state, size_t context, size_t base, size_t steps, const float* input,
                                   const uint32_t* position=nullptr, const float* values=nullptr,
                                   float* mirror=nullptr, size_t mirror_rows=0) {
    if (position) base=*position;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= steps * 512) return;
    const size_t step = index / 512, head = (index % 512) / 256, column = index % 256;
    const size_t keys_start = steps * 24 * 256;
    state[(head * context + base + step) * 256 + column] = input[keys_start + index];
    state[context * 512 + (head * context + base + step) * 256 + column] =
        values ? values[index] : input[keys_start + steps * 512 + index];
    if (base+step<mirror_rows) {
        mirror[(head*mirror_rows+base+step)*256+column]=input[keys_start+index];
        mirror[((2+head)*mirror_rows+base+step)*256+column]=
            values ? values[index] : input[keys_start+steps*512+index];
    }
}

__global__ void attention_scores(const float* state, size_t context, size_t base, size_t steps,
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
                    result += state[((head / 12) * context + token) * 256 + column]
                            * input[item * 256 + column];
                result *= 0.0625f;
            }
        }
        scores[item * selected + slot] = result;
    }
}

__global__ void attention_reduce(const float* state, size_t context, size_t base, size_t steps,
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
                    * state[context * 512 + ((head / 12) * context + token) * 256 + column];
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

// Split each recurrent column across eight lanes. More independent threads
// keep the fixed 48-head scan occupied without changing the delta-rule equation.
__global__ void delta_split_sequence(float* state,size_t steps,const float* packed,float* output) {
    const size_t head=blockIdx.x,col=threadIdx.x/8,split=threadIdx.x%8,elements=steps*48*128;
    float memory[16];
    #pragma unroll
    for (size_t i=0;i<16;++i) memory[i]=state[(head*128+split+i*8)*128+col];
    __shared__ float query[128],keys[128];
    for (size_t step=0;step<steps;++step) {
        const size_t offset=(step*48+head)*128;
        if (threadIdx.x<128) {
            query[threadIdx.x]=packed[offset+threadIdx.x];
            keys[threadIdx.x]=packed[elements+offset+threadIdx.x];
        }
        __syncthreads();
        const float factor=packed[3*elements+step*48+head],beta=packed[3*elements+steps*48+step*48+head];
        float prediction=0;
        #pragma unroll
        for (size_t i=0;i<16;++i) prediction+=(memory[i]*factor)*keys[split+i*8];
        for (unsigned offset=1;offset<8;offset*=2) prediction+=__shfl_xor(prediction,offset,8);
        const float correction=(packed[2*elements+offset+col]-prediction)*beta;
        float result=0;
        #pragma unroll
        for (size_t i=0;i<16;++i) {
            memory[i]=memory[i]*factor+keys[split+i*8]*correction;
            result+=memory[i]*query[split+i*8];
        }
        for (unsigned offset=1;offset<8;offset*=2) result+=__shfl_xor(result,offset,8);
        if (split==0) output[offset+col]=result;
        __syncthreads();
    }
    #pragma unroll
    for (size_t i=0;i<16;++i) state[(head*128+split+i*8)*128+col]=memory[i];
}

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
#include "host_io.hpp"
#include "peer.hpp"
#include "gemv.hpp"
#include "graph_cache.hpp"
#include "hc_decode.hpp"
#include "hyperconnection.hpp"
#include "gated_delta.hpp"
#include "ple.hpp"
#include "grouped_gemm.hpp"
#include "mixture_of_experts.hpp"
#include "moe_gemm.hpp"
#if !defined(__HIPCC__)
#include "moe_gemm_cuda.hpp"
#endif
#include "sparse_attention_compute.hpp"
#include "kv_stage.hpp"
#include "sparse_attention.hpp"
#include "hybrid_slots.hpp"
#include "resident_moe.hpp"
#include "hybrid_moe.hpp"
#include "qsa_decode.hpp"
#include "prefill_batch.hpp"
#include "decoder_span.hpp"
#include "verify_span.hpp"
#include "ops.hpp"
} // namespace

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
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_verify_span(void* handle,uint64_t begin,uint64_t end,uint64_t position,uint64_t steps,
                   const float* input,float* output) {
    try { verify_span(device(handle),size_t(begin),size_t(end),size_t(position),size_t(steps),input,output); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_verify_commit(void* handle,uint64_t begin,uint64_t end,uint64_t kept) {
    try { verify_commit(device(handle),size_t(begin),size_t(end),size_t(kept)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_output_head(void* handle,void* const* pointers,const int* kinds,uint64_t steps,uint64_t vocabulary,
                   const float* input,int32_t* tokens,float* logits,float* confidence) {
    try {
        if (!pointers || !kinds) throw std::runtime_error("invalid Straydog output head");
        output_head(device(handle),pointers,kinds,size_t(steps),size_t(vocabulary),input,tokens,logits,confidence); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// The head with rows [0, split) evaluated by the linked peer from its copy `peer_head`.
int sd_output_head_split(void* handle,void* const* pointers,const int* kinds,uint64_t steps,uint64_t vocabulary,
                         uint64_t split,void* peer_head,const float* input,int32_t* tokens,float* logits,
                         float* confidence) {
    try {
        if (!pointers || !kinds) throw std::runtime_error("invalid Straydog output head");
        output_head_split(device(handle),pointers,kinds,size_t(steps),size_t(vocabulary),size_t(split),peer_head,input,
                          tokens,logits,confidence);
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
// `count` embedding rows into `target`, staged without waiting for the device's stream.
int sd_mtp_embed(void* handle,const float* embeddings,uint64_t count,void* target) {
    try {
        Device& d=device(handle);
        if (!embeddings || !count) throw std::runtime_error("invalid MTP embeddings");
        float* device_rows=mtp_allocation(d,target,size_t(count)*2560);
        const size_t floats=size_t(count)*2560;
        if (floats>d.mtp_staging_floats) {
            for (int i=0;i<2;++i) {
                if (!d.mtp_staged[i]) check(hipEventCreateWithFlags(&d.mtp_staged[i],hipEventDisableTiming));
                check(hipEventSynchronize(d.mtp_staged[i]));
                if (d.mtp_staging[i]) { check(hipHostFree(d.mtp_staging[i])); d.mtp_staging[i]=nullptr; }
            }
            d.mtp_staging_floats=0;
            for (int i=0;i<2;++i) check(hipHostMalloc(reinterpret_cast<void**>(&d.mtp_staging[i]),floats*4,hipHostMallocDefault));
            d.mtp_staging_floats=floats;
        }
        const unsigned slot=d.mtp_turn++%2;
        check(hipEventSynchronize(d.mtp_staged[slot]));
        std::memcpy(d.mtp_staging[slot],embeddings,floats*4);
        check(hipMemcpyAsync(device_rows,d.mtp_staging[slot],floats*4,hipMemcpyHostToDevice,d.stream));
        check(hipEventRecord(d.mtp_staged[slot],d.stream));
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
        mtp_prologue_check(d,pointers,kinds,size_t(count));
        float* hidden=mtp_prologue_hidden(d,size_t(count));
        check(hipMemcpyAsync(hidden,kept,size_t(count)*10240*4,hipMemcpyDeviceToDevice,d.stream));
        check(hipMemcpyAsync(mtp_prologue_embeddings(d),embedded,size_t(count)*2560*4,hipMemcpyDeviceToDevice,d.stream));
        mtp_prologue_device(d,pointers,kinds,size_t(count));
        if (count==1) {
            std::vector<float> one(10240),output(10240);
            download(d,one.data(),hidden,10240*4);
            verify_span(d,size_t(layer),size_t(layer)+1,size_t(position),1,one.data(),output.data());
            return 0;
        }
        auto target=d.decoder.find(size_t(layer));
        if (target==d.decoder.end() || count>target->second.capacity)
            throw std::runtime_error("MTP prompt rows have no compatible MTP layer");
        check(hipMemcpyAsync(target->second.workspace,hidden,size_t(count)*10240*4,hipMemcpyDeviceToDevice,d.stream));
        prefill_span(d,size_t(layer),size_t(layer)+1,size_t(position),size_t(count),nullptr,nullptr,0);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
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
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
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
int sd_prefill_span(void* handle,uint64_t begin,uint64_t end,uint64_t position,uint64_t steps,
                     const float* input,float* output,uint64_t output_rows,double* timings) {
    try {
        prefill_span(device(handle),size_t(begin),size_t(end),size_t(position),size_t(steps),input,output,
                     size_t(output_rows),timings);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
// Deferred prefill: while on, spans that return no rows end without waiting for the device.
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
        upload(d,rows,embeddings,size_t(steps)*2560*4);
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
        for (size_t first=0;first<steps;first+=256) {
            const size_t count=std::min<size_t>(256,size_t(steps)-first);
            ple(d,pointers,kinds,history,count,embeddings+first*2560,nullptr,nullptr,found->second.workspace+first*10240);
        }
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
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
int sd_expert_slots(void* handle,uint64_t cache,uint64_t stage,uint64_t slot_bytes) {
    try { create_expert_slots(device(handle),size_t(cache),size_t(stage),size_t(slot_bytes)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
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
// Let the GPU read the first `assist` absent routes of a layer straight from
// page-locked host banks while the CPU evaluates the rest. Returns plans mapped.
int sd_hybrid_assist(void* handle,uint32_t assist,uint64_t* mapped) {
    try {
        if (!mapped || assist>10) throw std::runtime_error("invalid hybrid assist");
        auto& d=device(handle);
        *mapped=0;
        for (auto& item : d.resident) {
            ResidentMoe& plan=*item.second;
            if (!plan.hybrid) continue;
            void* views[3]{};
            bool ok=true;
            for (size_t role=0;role<3 && ok;++role)
                ok=hipHostGetDevicePointer(&views[role],const_cast<uint8_t*>(plan.host[role]),0)==hipSuccess;
            (void)hipGetLastError();   // clear the failed lookup
            for (size_t role=0;role<3;++role) plan.mapped[role]=ok ? static_cast<const uint8_t*>(views[role]) : nullptr;
            *mapped+=ok;
        }
#if defined(__HIPCC__)
        d.assist=assist;
#else
        // CUDA: an assisted expert read over PCIe by the zero-copy GEMV outlasts the CPU experts
        // on the measured RTX 5070 Ti, so each layer waits for it. Page-locked banks still serve
        // the DMA copies; no route is assisted.
        d.assist=0;
#endif
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
        check(hipHostRegister(pointer,size_t(bytes),hipHostRegisterPortable|hipHostRegisterReadOnly|hipHostRegisterMapped));
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_host_unregister(void* pointer) {
    try { check(hipHostUnregister(pointer)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
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

int sd_device_info(int ordinal, char* name, size_t capacity, uint64_t* free_bytes, uint64_t* total_bytes) {
    try {
        if (!name || capacity < 2 || !free_bytes || !total_bytes) throw std::runtime_error("invalid info buffers");
        check(hipSetDevice(ordinal));
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, ordinal));
        std::strncpy(name, prop.name, capacity - 1); name[capacity - 1] = 0;
        size_t available, total; check(hipMemGetInfo(&available, &total));
        *free_bytes = available; *total_bytes = total;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
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
        // A build holds code only for the GPU architectures it was compiled for.
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
        #if defined(__HIPCC__)
        hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop,ordinal));
        value->rdna4=std::strncmp(prop.gcnArchName,"gfx120",6)==0;
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
        value->await_watchdog=prop.kernelExecTimeoutEnabled!=0;
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
        value->used = workspace_bytes + 256;
        if (budget >= 256 * 1024 * 1024) {
            check(hipMalloc(reinterpret_cast<void**>(&value->expanded), gemm_elements * 4));
            check(hipMalloc(reinterpret_cast<void**>(&value->half_input), value->input_capacity * 2));
            check(hipMalloc(&value->blas_workspace, blas_workspace_bytes));
            constexpr size_t score_bytes = max_batch * 24 * 2052 * 4;
            check(hipMalloc(reinterpret_cast<void**>(&value->attention_scores), score_bytes));
            value->used += gemm_elements * 4 + blas_workspace_bytes + value->input_capacity * 2 + score_bytes;
            blas_check(hipblasCreate(&value->blas));
            blas_check(hipblasSetStream(value->blas, value->stream));
            blas_check(hipblasSetWorkspace(value->blas, value->blas_workspace, blas_workspace_bytes));
            blas_check(hipblasSetMathMode(value->blas, HIPBLAS_DEFAULT_MATH));
        }
        return value.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

void sd_destroy(void* handle) { delete static_cast<Device*>(handle); }

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
        return pointer;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
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

int sd_release(void* handle, void* pointer) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (found == value.allocations.end()) throw std::runtime_error("unknown Straydog allocation");
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
            value.kv_stage = nullptr; value.kv_stage_rows = 0;
            value.kv_stage_owner = nullptr; value.kv_stage_valid = 0;
            value.kv_mirrors.clear(); value.kv_mirror_rows = 0;
        }
        const auto host = value.host_states.find(pointer);
        if (host != value.host_states.end()) {
            check(hipHostFree(host->second)); value.host_states.erase(host);
        } else {
            check(hipFree(pointer)); value.used -= found->second + allocation_tail;
        }
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

// Register the existing VRAM stage: one layer for prefill, all host-state prefixes for decode.
int sd_kv_stage(void* handle, void* pointer, uint64_t rows) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (!rows || rows > 262144 || found == value.allocations.end() || found->second != size_t(rows) * 1024 * 4 ||
            value.host_states.count(pointer))
            throw std::runtime_error("invalid Straydog K/V stage");
        kv_clear_graphs(value);
        value.kv_mirrors.clear(); value.kv_mirror_rows = 0;
        value.kv_stage = static_cast<float*>(pointer); value.kv_stage_rows = size_t(rows);
        value.kv_stage_owner = nullptr; value.kv_stage_valid = 0;
        if (!value.host_states.empty()) {
            value.kv_mirror_rows = size_t(rows)/value.host_states.size();
            size_t slot=0;
            if (value.kv_mirror_rows) for (const auto& state : value.host_states) {
                value.kv_mirrors.emplace(state.first,KvMirror{value.kv_stage+slot*value.kv_mirror_rows*1024,0});
                ++slot;
            }
        }
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
        if (found == value.allocations.end() || found->second != context * 1024 * 4)
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
        write_attention_kv<<<unsigned((steps*512+255)/256),256,0,value.stream>>>(static_cast<float*>(state),
            size_t(context), size_t(base), size_t(steps), value.input);
        attention_scores<<<unsigned(steps*24),256,0,value.stream>>>(static_cast<float*>(state),
            size_t(context), size_t(base), size_t(steps), size_t(selected), value.input, value.attention_scores);
        attention_reduce<<<unsigned(steps*24),256,0,value.stream>>>(static_cast<float*>(state),
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
                    blas_check(hipblasSgemm(value.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
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
        if (fallback_count) {
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

void* sd_buffer(void* handle, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!bytes || bytes > value.budget - value.used || allocation_tail > value.budget - value.used - bytes)
            throw std::runtime_error("Straydog VRAM budget exceeded");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, size_t(bytes) + allocation_tail));
        try { value.allocations.emplace(pointer, size_t(bytes)); }
        catch (...) { cleanup(hipFree(pointer)); throw; }
        value.used += size_t(bytes) + allocation_tail;
        check(hipMemsetAsync(pointer, 0, size_t(bytes) + allocation_tail, value.stream));
        return pointer;
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

int sd_buffer_write(void* handle, void* buffer, uint64_t offset, const void* data, uint64_t bytes) {
    try {
        auto& value = device(handle);
        if (!data || offset > UINT64_MAX - bytes) throw std::runtime_error("invalid Straydog buffer write");
        op_span(value, static_cast<uint8_t*>(buffer) + offset, size_t(bytes), "a written buffer");
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

int sd_op_linear(void* handle, const void* weights, int kind, uint64_t rows, uint64_t cols, uint64_t first_row,
                 const float* input, uint64_t input_stride, uint64_t steps, float* output, uint64_t output_stride,
                 float* scratch, uint64_t scratch_bytes) {
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
                  size_t(input_stride), size_t(steps), output, size_t(output_stride), scratch, size_t(scratch_bytes));
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
    if (!rows || !experts || !k || k > 64 || k > experts || gating < 0 || gating > 2 || experts * 4 > 48 * 1024)
        throw std::runtime_error("invalid Straydog route shape");
    op_span(value, logits, size_t(rows * experts) * 4, "router logits");
    if (bias) op_span(value, bias, size_t(experts) * 4, "a router bias");
    op_span(value, ids, size_t(rows * k) * 4, "routed experts");
    op_span(value, weights, size_t(rows * k) * 4, "routing weights");
    op_route_kernel<<<unsigned(rows), 256, size_t(experts) * 4, value.stream>>>(logits, bias, size_t(experts), size_t(k), gating,
                                                                              normalize, scale, ids, weights);
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

int sd_op_sparse_attention(void* handle, const float* q, const float* raw, uint64_t raw_rows, const int32_t* positions,
                           uint64_t window, const float* compressed, uint64_t compressed_rows, const int32_t* selected,
                           uint64_t selections, const float* sinks, uint64_t rows, uint64_t heads, uint64_t dim,
                           float scale, float* out) {
    SD_OP_BEGIN
    const size_t shared = size_t(dim + window + selections) * 4;
    if (!rows || !heads || !dim || !window || shared > 60 * 1024) throw std::runtime_error("invalid Straydog attention shape");
    op_span(value, q, size_t(rows * heads * dim) * 4, "attention queries");
    op_span(value, raw, size_t(raw_rows * dim) * 4, "window rows");
    op_span(value, positions, size_t(rows) * 4, "query positions");
    if (selections) {
        op_span(value, compressed, size_t(compressed_rows * dim) * 4, "compressed rows");
        op_span(value, selected, size_t(rows * selections) * 4, "selected rows");
    }
    op_span(value, sinks, size_t(heads) * 4, "attention sinks");
    op_span(value, out, size_t(rows * heads * dim) * 4, "an attention output");
    op_sparse_attention_kernel<<<unsigned(rows * heads), 256, shared, value.stream>>>(q, raw, size_t(raw_rows), positions, size_t(window), compressed,
        selected, size_t(selections), sinks, size_t(heads), size_t(dim), scale, out);
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
    op_span(value, keys, size_t(groups * dim) * 4, "index keys");
    op_span(value, weights, size_t(rows * heads) * 4, "index weights");
    op_span(value, visible, size_t(rows) * 4, "visible counts");
    op_span(value, scores, size_t(rows * groups) * 4, "index scores");
    op_index_scores_kernel<<<dim3(op_blocks(size_t(groups)), unsigned(rows)), 256, 0, value.stream>>>(q, keys, weights, visible,
        size_t(groups), size_t(heads), size_t(dim), scale, scores);
    SD_OP_END
}

int sd_op_top_k(void* handle, const float* scores, uint64_t rows, uint64_t groups, uint64_t k, int32_t* out) {
    SD_OP_BEGIN
    if (!rows || !groups || !k || groups > 8192) throw std::runtime_error("invalid Straydog top-k shape");
    op_span(value, scores, size_t(rows * groups) * 4, "scores");
    op_span(value, out, size_t(rows * k) * 4, "top-k indices");
    op_top_k_kernel<<<unsigned(rows), 256, size_t(groups) * 4, value.stream>>>(scores, size_t(groups), size_t(k), out);
    SD_OP_END
}
} // extern C
