// Private CPU scheduling; rows keep their existing kernel and arithmetic order.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <utility>
#if defined(__linux__)
#include <fstream>
#include <map>
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sd_cpu {
namespace {
#ifdef SD_CPU_WORKER_TEST
// Test-only scheduling control; absent from the production library.
struct RowDispatch;
std::function<void(size_t)> worker_start_test_hook;
std::function<void(size_t, const RowDispatch*)> worker_claim_test_hook;
#endif
#if defined(_WIN32)
std::vector<GROUP_AFFINITY> physical_cores() {
    DWORD bytes = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return {};
    std::vector<uint8_t> buffer(bytes);
    auto* records = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, records, &bytes)) return {};
    // The process's allowed CPUs in its primary group; cores elsewhere keep their full mask.
    USHORT group = 0, count = 1;
    DWORD_PTR process_mask = 0, system_mask = 0;
    const bool restricted = GetProcessGroupAffinity(GetCurrentProcess(), &count, &group) &&
                            GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
    std::vector<GROUP_AFFINITY> cores;
    for (DWORD offset = 0; offset < bytes;) {
        const auto* record = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (record->Relationship == RelationProcessorCore && record->Processor.GroupCount == 1) {
            GROUP_AFFINITY core = record->Processor.GroupMask[0];
            if (restricted && core.Group == group) core.Mask &= process_mask;
            if (core.Mask) cores.push_back(core);
        }
        offset += record->Size;
    }
    std::sort(cores.begin(), cores.end(), [](const GROUP_AFFINITY& a, const GROUP_AFFINITY& b) {
        return a.Group != b.Group ? a.Group < b.Group : (a.Mask & (~a.Mask + 1)) < (b.Mask & (~b.Mask + 1));
    });
    return cores;
}
#elif defined(__linux__)
std::vector<cpu_set_t> read_physical_cores() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) return {};
    std::map<std::string, size_t> index;
    std::vector<cpu_set_t> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        std::string siblings;
        if (!std::getline(file, siblings) || siblings.empty()) return {};
        auto found = index.find(siblings);
        if (found == index.end()) {
            found = index.emplace(siblings, cores.size()).first;
            cores.emplace_back();
            CPU_ZERO(&cores.back());
        }
        CPU_SET(cpu, &cores[found->second]);
    }
    return cores;
}
// The CPUs the process started with, by core. A thread's own affinity is narrower once a pool
// has bound it, and a pool made from such a thread would put every worker on that one core
// (six threads ran at a fifth of one thread's speed): the set is read once, at load.
const std::vector<cpu_set_t> process_cores = read_physical_cores();
std::vector<cpu_set_t> physical_cores() { return process_cores; }
#endif

// A dispatch owns its cursor and callback. A worker delayed before claiming
// work can skip a finished phase without touching its expired captures.
// Completion counts rows, not worker arrivals (AGENTS.md §1 / §6).
struct RowDispatch {
    size_t rows = 0, block = 0, parallelism = 1;
    std::function<void(size_t, size_t)> job;
    alignas(64) std::atomic<size_t> next{0};
    alignas(64) std::atomic<size_t> done{0};
    std::mutex error_mutex;
    std::exception_ptr error;
    void reset(size_t row_count, size_t chunk, size_t workers, std::function<void(size_t, size_t)> callback) {
        rows = row_count; block = chunk; parallelism = workers;
        job = std::move(callback);
        next.store(0, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
        error = nullptr;
    }
    void execute() {
        size_t completed = 0;
        while (true) {
            // Stream longer runs initially, then leave fine blocks for stealing.
            // A stale estimate only changes the chunk length: fetch_add still
            // gives disjoint, contiguous ranges, clipped to the final row.
            const size_t cursor = next.load(std::memory_order_relaxed);
            const size_t chunk = cursor < rows && rows - cursor > parallelism * 4 * block ? 4 * block : block;
            const size_t first = next.fetch_add(chunk, std::memory_order_relaxed);
            if (first >= rows) break;
            const size_t last = std::min(rows, first + chunk);
            try { job(first, last); }
            catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
            completed += last - first;
        }
        // Publish this worker's completed rows together: avoid a second
        // contended atomic for every small row block. An idle/late worker has
        // no contribution, so completion still never waits for its arrival.
        // The acquire/release chain publishes EVERY block's stores before reuse.
        if (completed) done.fetch_add(completed, std::memory_order_acq_rel);
    }
};

// Persistent workers spin briefly between per-layer jobs, then sleep. Worker 0 is
// the calling thread.
class Workers {
public:
    explicit Workers(size_t count) : count_(count),
        slots_(new RowDispatch[count + 1]), hazards_(new Hazard[count]) {
#if defined(__linux__) || defined(_WIN32)
        cores_ = physical_cores();
#endif
        for (size_t i = 1; i < count_; ++i) {
            threads_.emplace_back([this, i]() {
                bind(i);
#ifdef SD_CPU_WORKER_TEST
                if (worker_start_test_hook) worker_start_test_hook(i);
#endif
                loop(i);
            });
        }
    }
    ~Workers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_.store(true, std::memory_order_release); generation_.fetch_add(1);
        }
        wake_.notify_all();
        for (auto& thread : threads_) thread.join();
    }
    void run_blocks(size_t rows, size_t block, std::function<void(size_t, size_t)> job) {
        // The calling thread is not bound (it also drives the GPUs): the workers take cores 1 on,
        // core 0 is left to it.
        if (count_ == 1) {
            for (size_t first = 0; first < rows; first += block) job(first, std::min(rows, first + block));
            return;
        }
        // Never reuse the published slot, or one a worker has protected.
        // One slot per worker plus the current slot bounds storage even when
        // a worker is descheduled before it has claimed any work.
        RowDispatch* dispatch = nullptr;
        while (!dispatch) {
            const auto* published = current_.load();
            for (size_t n = 0; n <= count_; ++n) {
                const size_t index = (next_slot_ + n) % (count_ + 1);
                auto* candidate = &slots_[index];
                if (candidate == published) continue;
                bool protected_slot = false;
                for (size_t i = 1; i < count_; ++i)
                    if (hazards_[i].value.load() == candidate) { protected_slot = true; break; }
                if (!protected_slot) {
                    dispatch = candidate; next_slot_ = (index + 1) % (count_ + 1); break;
                }
            }
            if (!dispatch) std::this_thread::yield();
        }
        dispatch->reset(rows, block, count_, std::move(job));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_.store(dispatch);
            generation_.fetch_add(1, std::memory_order_release);
        }
        wake_.notify_all();
        dispatch->execute();
        while (dispatch->done.load(std::memory_order_acquire) != rows) {
            std::this_thread::yield();
        }
        if (dispatch->error) std::rethrow_exception(dispatch->error);
    }
private:
    void bind(size_t index) {
        // Best effort: placement only affects speed, never results. Linux threads float: bound
        // to cores they ran no faster with one per core and slower with more threads than cores,
        // and a busy core held its worker (Qwen3.6 Q8_0 experts, Ryzen 7500F: 50 GB/s on six
        // unbound threads).
#if defined(_WIN32)
        if (!cores_.empty()) SetThreadGroupAffinity(GetCurrentThread(), &cores_[index % cores_.size()], nullptr);
#else
        (void)index;
#endif
    }
    void loop(size_t index) {
        uint64_t seen = 0;
        while (true) {
            uint64_t now = generation_.load(std::memory_order_acquire);
            for (int spin = 0; now == seen && spin < 200000; ++spin) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
                now = generation_.load(std::memory_order_acquire);
            }
            if (now == seen) {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&]() { return generation_.load() != seen; });
                now = generation_.load(std::memory_order_acquire);
            }
            if (stop_.load(std::memory_order_acquire)) return;
            seen = now;
            RowDispatch* dispatch;
            // Sequential consistency pairs publication, protection and validation
            // with the caller's scan. A late worker must validate BEFORE reading
            // the callback; the caller never resets a protected/current slot.
            do {
                dispatch = current_.load();
                hazards_[index].value.store(dispatch);
            } while (dispatch != current_.load());
#ifdef SD_CPU_WORKER_TEST
            if (worker_claim_test_hook) worker_claim_test_hook(index, dispatch);
#endif
            dispatch->execute();
            hazards_[index].value.store(nullptr);
        }
    }
    struct alignas(64) Hazard { std::atomic<RowDispatch*> value{nullptr}; };
    size_t count_;
    std::unique_ptr<RowDispatch[]> slots_;
    std::unique_ptr<Hazard[]> hazards_;
    size_t next_slot_ = 0;
#if defined(__linux__)
    std::vector<cpu_set_t> cores_;
#elif defined(_WIN32)
    std::vector<GROUP_AFFINITY> cores_;
#endif
    std::vector<std::thread> threads_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<RowDispatch*> current_{nullptr};
    std::mutex mutex_;
    std::condition_variable wake_;
    std::atomic<bool> stop_{false};
};

} // namespace
} // namespace sd_cpu
