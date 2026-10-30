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
#define hipblasGemmStridedBatchedEx cublasGemmStridedBatchedEx
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
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <map>
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
    // A layer's part (qwen_parts.hpp): its half of the shared expert's width, rows [320p, +320) of gate and up and those
    // columns of down (no data: none). Decode runs it as the shared expert (weights and table); a prompt's grouped experts
    // run without it (batch_table: the table without the shared expert) and it beside them.
    Matrix half[3]{};
    void* batch_table=nullptr;
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
    size_t parts=1,part=0;        // the layer split across GPUs: this device's part of `parts` (QwenHeads)
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
constexpr size_t attention_score_bytes = max_batch * 24 * 2052 * 4;   // decode attention partials (QSA prefill marks)
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

struct KvMirror { float* data; size_t valid; };
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
struct Device {
    int ordinal;
    size_t budget = 0, used = 0;
    size_t batch_capacity = 32, input_capacity = 32 * max_cols, output_capacity = max_rows;
    size_t gemm_min_batch = 8;
    bool half_gemm = false;
    bool half_decode = false;   // a decode or verify pass's index and attention scores through WMMA, FP16 operands (RDNA4: the FP16 prompt pass's)
    bool q8_wide = false;   // a decode row's Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 GEMVs on the Q8 kernels, mixed lists with BF16 rows in one launch (RDNA4: Qwen3.8)
    bool q8_decode = false;   // a decode row's Q5_K GEMVs, IQ2 experts' gate/up and down (IQ2_S's down excepted) and Q5_K shared expert on the Q8 kernels (RDNA4)
    uint64_t* gemm_checks = nullptr;   // optional quality-only cast counters, enabled before graphs
    bool rdna4 = false;
    unsigned compute_units = 0;   // the device's CUs (SMs on CUDA)
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
    std::unordered_map<void*,ExchangeArea> exchanges;   // host exchanges by host address (sd_exchange)
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
    std::shared_ptr<struct Transfers> transfers;   // asynchronous uploads (transfer.hpp), from the first use
#if defined(__linux__)
    std::unique_ptr<ReadAhead> read_ahead;   // started by the first expert read-ahead hint
#endif
    explicit Device(int value) : ordinal(value) {}
    ~Device() {
        cleanup(hipSetDevice(ordinal));
        if (host_prefetch.joinable()) host_prefetch.join();
        for (auto& ready : host_ready) cleanup(hipHostFree(ready.second));
        copier.reset();
        transfers.reset();   // waits for its copies before the allocations go
        if (stream) cleanup(hipStreamSynchronize(stream));
        if (helper_stream) cleanup(hipStreamSynchronize(helper_stream));
        if (kv_stream) cleanup(hipStreamSynchronize(kv_stream));
        if (side_stream) cleanup(hipStreamSynchronize(side_stream));
        for (auto& entry : graphs) {
            cleanup(hipGraphExecDestroy(entry.second.executable));
            cleanup(hipGraphDestroy(entry.second.graph));
        }
        decoder.clear(); resident.clear();
        if (slots.base) cleanup(hipFree(slots.base));
        if (slots.host) cleanup(hipHostFree(slots.host));
        if (exchange) cleanup(hipHostFree(exchange));
        if (readback) cleanup(hipHostFree(readback));
        if (route_mailbox) cleanup(hipHostFree(route_mailbox));
        if (readback_done) cleanup(hipEventDestroy(readback_done));
        if (parts_ready) cleanup(hipEventDestroy(parts_ready));
        if (parts_read) cleanup(hipEventDestroy(parts_read));
        if (cpu_partial) cleanup(hipFree(cpu_partial));
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
        for (auto event : {kv_released, kv_prefetched, side_fork, side_join}) if (event) cleanup(hipEventDestroy(event));
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

// values (optional) holds the rows' values instead of the packed input. A layer's K/V is
// 2 * QwenHeads<P>::kv planes of `context` rows: each K/V head's keys, then each one's values.
template<int P=1>
__global__ void write_attention_kv(float* state, size_t context, size_t base, size_t steps, const float* input,
                                   const uint32_t* position=nullptr, const float* values=nullptr,
                                   float* mirror=nullptr, size_t mirror_rows=0) {
    constexpr size_t K = QwenHeads<P>::kv, width = K * 256;
    if (position) base=*position;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= steps * width) return;
    const size_t step = index / width, head = (index % width) / 256, column = index % 256;
    const size_t keys_start = steps * QwenHeads<P>::query * 256;
    state[(head * context + base + step) * 256 + column] = input[keys_start + index];
    state[context * width + (head * context + base + step) * 256 + column] =
        values ? values[index] : input[keys_start + steps * width + index];
    if (base+step<mirror_rows) {
        mirror[(head*mirror_rows+base+step)*256+column]=input[keys_start+index];
        mirror[((K+head)*mirror_rows+base+step)*256+column]=
            values ? values[index] : input[keys_start+steps*width+index];
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
template<int P=1>
__global__ void delta_split_sequence(float* state,size_t steps,const float* packed,float* output) {
    constexpr size_t heads=48/P;
    const size_t head=blockIdx.x,col=threadIdx.x/8,split=threadIdx.x%8,elements=steps*heads*128;
    float memory[16];
    #pragma unroll
    for (size_t i=0;i<16;++i) memory[i]=state[(head*128+split+i*8)*128+col];
    __shared__ float query[128],keys[128];
    for (size_t step=0;step<steps;++step) {
        const size_t offset=(step*heads+head)*128;
        if (threadIdx.x<128) {
            query[threadIdx.x]=packed[offset+threadIdx.x];
            keys[threadIdx.x]=packed[elements+offset+threadIdx.x];
        }
        __syncthreads();
        const float factor=packed[3*elements+step*heads+head],beta=packed[3*elements+steps*heads+step*heads+head];
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

#include "host_io.hpp"
#include "peer.hpp"
#include "link_tail.hpp"
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
#include "experts.hpp"
#include "hybrid_moe.hpp"
#include "qsa_decode.hpp"
#include "prefill_batch.hpp"
#include "decoder_span.hpp"
#include "verify_span.hpp"
#include "ops.hpp"
#include "ds_attention_wmma.hpp"
#include "transfer.hpp"

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
        bool parts=false;
        verify_span_queue(device(handle),size_t(begin),size_t(end),size_t(position),size_t(steps),input,nullptr,parts);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
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
                  uint64_t vocabulary,uint64_t split,void* peer_head,int32_t* token,float* confidence) {
    try {
        if (!mtp || !mtp_kinds) throw std::runtime_error("invalid drafting step");
        draft_step(device(handle),mtp,mtp_kinds,size_t(layer),size_t(position),size_t(steps),embeddings,hidden,
                   size_t(previous),head,head_kinds,size_t(vocabulary),size_t(split),peer_head,token,confidence);
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
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int sd_draft_head(void* handle,void* const* head,const int* head_kinds,uint64_t layer,uint64_t row,uint64_t vocabulary,
                  uint64_t split,void* peer_head,int32_t* token,float* confidence) {
    try {
        Device& d=device(handle);
        auto found=d.decoder.find(size_t(layer));
        if (found==d.decoder.end() || row>=verify_max || found->second.verified<=row)
            throw std::runtime_error("invalid drafting head row");
        draft_head(d,head,head_kinds,size_t(vocabulary),size_t(split),peer_head,
                   verify_layout(found->second.workspace).hidden+size_t(row)*10240,false,token,confidence);
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
        auto target=d.decoder.find(size_t(layer));
        // (a layer in parts: its rows into the workspace, the host running the layer on both devices)
        if (target!=d.decoder.end() && target->second.parts==2) {
            if (count>target->second.capacity) throw std::runtime_error("MTP prompt rows have no compatible MTP layer");
            check(hipMemcpyAsync(target->second.workspace,hidden,size_t(count)*10240*4,hipMemcpyDeviceToDevice,d.stream));
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
        std::lock_guard<std::mutex> lock(registered_mutex);
        registered[reinterpret_cast<uintptr_t>(pointer)]={size_t(bytes),false};
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

int sd_release(void* handle, void* pointer) {
    try {
        auto& value = device(handle);
        auto found = value.allocations.find(pointer);
        if (found == value.allocations.end()) throw std::runtime_error("unknown Straydog allocation");
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
        if (!save) kv_invalidate_write(value, state, size_t(first));
        if (save)
            check(hipMemcpy2DAsync(host, width * 4, base, pitch * 4, width * 4, height, hipMemcpyDefault, value.stream));
        else
            check(hipMemcpy2DAsync(base, pitch * 4, host, width * 4, width * 4, height, hipMemcpyDefault, value.stream));
        check(hipStreamSynchronize(value.stream));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// The end of the message of a growth that found no memory (crates/straydog/src/worker.rs: not fatal).
constexpr const char* kv_no_memory = "not enough host memory for the K/V of this sequence";
int sd_kv_grow(void* handle, uint64_t layer, uint64_t capacity, uint64_t valid, void** state) {
    try {
        auto& d = device(handle);
        auto found = d.decoder.find(size_t(layer));
        if (found == d.decoder.end() || decoder_linear(size_t(layer)) || !state) throw std::runtime_error("invalid Straydog K/V growth");
        DecoderLayer& plan = found->second;
        const size_t from = plan.context, to = size_t(capacity);
        auto host = d.host_states.find(plan.states[0]);
        auto index = d.allocations.find(plan.states[1]);
        const size_t old_pool = from / 4 + 1, pool = to / 4 + 1, history = qsa_history * 128;
        if (host == d.host_states.end() || index == d.allocations.end() || plan.parts != 1 || to <= from || to > 262144 || valid > from ||
            index->second < (pool * 128 + history) * 4)
            throw std::runtime_error("invalid Straydog K/V growth");
        check(hipStreamSynchronize(d.stream));
        if (d.kv_prefetching) { check(hipStreamSynchronize(d.kv_stream)); d.kv_prefetching = false; }
        void* grown = nullptr;
        // No memory leaves every state as it was: the request fails, the engine stays (see kv_no_memory).
        if (hipHostMalloc(&grown, to * 1024 * 4 + allocation_tail, hipHostMallocMapped) != hipSuccess) {
            (void)hipGetLastError();
            throw std::runtime_error(kv_no_memory);
        }
        void* pointer = nullptr;
        try {
            std::memset(grown, 0, to * 1024 * 4 + allocation_tail);
            for (size_t plane = 0; plane < 4; ++plane)
                std::memcpy(static_cast<float*>(grown) + plane * to * 256, static_cast<const float*>(host->second) + plane * from * 256, valid * 1024);
            check(hipHostGetDevicePointer(&pointer, grown, 0));
            // The indexer's blocks at their new pitch, then its rolling history: all of it through host memory, in its allocation.
            std::vector<float> before(old_pool * 128 + history), after(pool * 128 + history, 0.0f);
            check(hipMemcpy(before.data(), plan.states[1], before.size() * 4, hipMemcpyDeviceToHost));
            for (size_t column = 0; column < 128; ++column)
                std::copy_n(before.data() + column * old_pool, std::min(old_pool, valid / 4 + 1), after.data() + column * pool);
            std::copy_n(before.data() + 128 * old_pool, history, after.data() + 128 * pool);
            check(hipMemcpy(plan.states[1], after.data(), after.size() * 4, hipMemcpyHostToDevice));
        } catch (...) { cleanup(hipHostFree(grown)); throw; }
        // Every captured graph holds the old K/V or the old pitches.
        kv_clear_graphs(d);
        check(hipHostFree(host->second));
        d.host_states.erase(host); d.allocations.erase(plan.states[0]); d.kv_mirrors.erase(plan.states[0]);
        d.host_states.emplace(pointer, grown); d.allocations.emplace(pointer, to * 1024 * 4);
        std::replace(plan.dependencies.begin(), plan.dependencies.end(), plan.states[0], pointer);
        plan.states[0] = pointer; plan.context = to;
        // (a DeltaNet layer holds no positions: its context is only what spans are checked against)
        for (auto& entry : d.decoder) if (decoder_linear(entry.first)) entry.second.context = std::max(entry.second.context, to);
        kv_layout(d);
        *state = pointer;
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
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int sd_decode_options(void* handle, int flags) {
    try {
        auto& value = device(handle);
        if (flags & ~7) throw std::runtime_error("invalid Straydog decode options");
        value.q8_decode = (flags & 1) && value.rdna4;
        value.q8_wide = (flags & 4) && value.rdna4;
        value.half_decode = (flags & 2) && value.rdna4;
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
            throw std::runtime_error("Straydog VRAM budget exceeded");
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
            check(hipHostMalloc(reinterpret_cast<void**>(&value.readback), readback_capacity, hipHostMallocDefault));
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
        check(hipGraphLaunch(found->second.executable, value.stream));
        ++value.graph_replays;
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
        } else {
            if (steps > uint64_t(gemv_max_vectors)) op_span(value, scratch, size_t(scratch_bytes), "a linear scratch");
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
        const dim3 score_grid{unsigned(wmma_chunks), unsigned(rows)}, out_grid{unsigned(dim / 64), unsigned(rows)};
        const size_t keys_bytes = size_t(stride) * sizeof(float*);
        #define SD_DS_ATTENTION(HM) \
            ds_attention_scores_wmma<HM><<<score_grid, 128, 0, value.stream>>>(q, raw, size_t(raw_rows), positions, size_t(window), \
                compressed, selected, size_t(selections), size_t(heads), size_t(dim), scale, pairs, forward, stride, scores, partial); \
            check(hipGetLastError()); \
            ds_attention_output_wmma<HM><<<out_grid, 128, keys_bytes, value.stream>>>(scores, partial, wmma_chunks, raw, size_t(raw_rows), \
                positions, size_t(window), compressed, selected, size_t(selections), sinks, size_t(heads), size_t(dim), pairs, inverse, \
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
        const dim3 grid{unsigned((groups + ds_index_wide_groups - 1) / ds_index_wide_groups), unsigned((rows + ds_index_rows - 1) / ds_index_rows)};
        if (heads == 32)
            ds_index_scores_wmma_rows<2, 128><<<grid, 256, 0, value.stream>>>(q, keys, weights, visible, size_t(rows), size_t(groups),
                                                                           size_t(heads), scale, size_t(rope_dims / 2), positions,
                                                                           frequencies, scores);
        else
            ds_index_scores_wmma_rows<4, 128><<<grid, 256, 0, value.stream>>>(q, keys, weights, visible, size_t(rows), size_t(groups),
                                                                           size_t(heads), scale, size_t(rope_dims / 2), positions,
                                                                           frequencies, scores);
        check(hipGetLastError());
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
        // Peer access, once per pair (a device without it cannot take part).
        static std::mutex mutex;
        static std::vector<std::pair<int, int>> linked;
        {
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
        // Every part's end is marked before any device waits, so no device waits for another's sum.
        for (Device* d : list)
            on_device(d->ordinal, [&]() {
                if (!d->parts_ready) check(hipEventCreateWithFlags(&d->parts_ready, hipEventDisableTiming));
                check(hipEventRecord(d->parts_ready, d->stream));
            });
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
