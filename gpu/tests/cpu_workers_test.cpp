// Standalone regression: cl /O2 /EHsc /std:c++17 cpu_workers_test.cpp
// or g++ -O2 -std=c++17 -pthread cpu_workers_test.cpp -o cpu_workers_test
#define SD_CPU_WORKER_TEST
#include "../cpu_experts.cpp"
#include <future>
#include <iostream>
#include <stdexcept>

static void require(bool ok) { if (!ok) throw std::runtime_error("CPU worker regression"); }

int main() {
    // A worker paused before its first claim must not delay call completion.
    {
        std::promise<void> entered, release;
        auto started = entered.get_future();
        auto resume = release.get_future();
        std::atomic<bool> completed{false};
        bool skipped_paused_worker = false;
        std::vector<int> output(257, 0);
        sd_cpu::worker_start_test_hook = [&](size_t i) {
            if (i == 1) { entered.set_value(); resume.wait(); }
        };
        {
            Workers pool(3);
            std::thread controller([&]() {
                started.wait();
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!completed.load() && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                skipped_paused_worker = completed.load();
                release.set_value();
            });
            pool.run_blocks(output.size(), 32, [&](size_t first, size_t last) {
                for (size_t i = first; i < last; ++i) ++output[i];
            });
            completed.store(true);
            controller.join();
        }
        sd_cpu::worker_start_test_hook = {};
        require(skipped_paused_worker);
        for (int n : output) require(n == 1);
    }
    // Keep an old slot protected BEFORE its claim while the caller finishes
    // that phase, destroys its captures, and cycles through later job slots.
    {
        std::promise<void> entered, release;
        auto resume = release.get_future();
        std::atomic<const sd_cpu::RowDispatch*> protected_state{nullptr};
        std::atomic<bool> once{false}, parked{false};
        bool preserved = true;
        sd_cpu::worker_claim_test_hook = [&](size_t i, const sd_cpu::RowDispatch* state) {
            if (i == 1 && !once.exchange(true)) {
                protected_state.store(state);
                parked.store(true);
                entered.set_value();
                resume.wait_for(std::chrono::seconds(10));
                parked.store(false);
            }
        };
        {
            Workers pool(3);
            {
                auto started = entered.get_future();
                std::vector<int> old_output(257, 0);
                pool.run_blocks(old_output.size(), 32, [&](size_t first, size_t last) {
                    if (first == 0) started.wait();
                    for (size_t i = first; i < last; ++i) ++old_output[i];
                });
                for (int n : old_output) preserved = preserved && n == 1;
            } // the protected slot's callback now contains expired captures
            const auto* old = protected_state.load();
            for (size_t phase = 0; phase < 100; ++phase) {
                std::vector<int> output(1000 + phase, 0);
                pool.run_blocks(output.size(), 32, [&](size_t first, size_t last) {
                    for (size_t i = first; i < last; ++i) ++output[i];
                });
                for (int n : output) preserved = preserved && n == 1;
                preserved = preserved && parked.load() && old && old->rows == 257
                    && old->next.load() >= 257 && old->done.load() == 257;
            }
            release.set_value();
        }
        sd_cpu::worker_claim_test_hook = {};
        require(preserved);
    }
    // A worker held inside one block must not strand a static slice: the
    // remaining workers can claim every other row before it is released.
    {
        Workers pool(3);
        std::promise<void> entered, release;
        auto blocked = release.get_future();
        std::atomic<size_t> remaining{0};
        std::atomic<size_t> held{0};
        bool progressed = false;
        std::thread controller([&]() {
            entered.get_future().wait();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (remaining.load() != 257 - held.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            progressed = remaining.load() == 257 - held.load();
            release.set_value();
        });
        pool.run_blocks(257, 1, [&](size_t first, size_t last) {
            if (first == 0) { held.store(last - first); entered.set_value(); blocked.wait(); }
            else remaining.fetch_add(last - first);
        });
        const bool released = blocked.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        controller.join();
        require(progressed && released);
    }
    for (size_t threads : {size_t(1), size_t(3), size_t(6), size_t(8)}) {
        Workers pool(threads);
        for (size_t phase = 0; phase < 2000; ++phase) {
            const size_t rows = phase % 2049;
            std::vector<int> output(rows, 0);
            pool.run_blocks(rows, 1 + phase % 64, [&](size_t first, size_t last) {
                for (size_t i = first; i < last; ++i) ++output[i];
            });
            for (int n : output) require(n == 1);
        }
        bool threw = false;
        try {
            pool.run_blocks(257, 32, [](size_t first, size_t) {
                if (first == 32) throw std::runtime_error("expected");
            });
        } catch (const std::runtime_error&) { threw = true; }
        require(threw);
        std::vector<int> output(257, 0);
        pool.run_blocks(output.size(), 32, [&](size_t first, size_t last) {
            for (size_t i = first; i < last; ++i) output[i] = 7;
        });
        for (int n : output) require(n == 7);
    }
#if defined(__linux__)
    // The cores come from the process, not from the calling thread: a thread bound to one CPU
    // (as an earlier pool once bound its caller) still sees every core, and the pool leaves it
    // bound as it was.
    {
        const size_t cores = sd_cpu::physical_cores().size();
        cpu_set_t before, one;
        require(pthread_getaffinity_np(pthread_self(), sizeof(before), &before) == 0);
        CPU_ZERO(&one);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &before)) { CPU_SET(cpu, &one); break; }
        require(pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0);
        require(sd_cpu::physical_cores().size() == cores);
        {
            Workers pool(2);
            std::vector<int> output(64, 0);
            pool.run_blocks(output.size(), 8, [&](size_t first, size_t last) { for (size_t i = first; i < last; ++i) output[i] = 1; });
            for (int n : output) require(n == 1);
        }
        cpu_set_t after;
        require(pthread_getaffinity_np(pthread_self(), sizeof(after), &after) == 0 && CPU_EQUAL(&after, &one));
        require(pthread_setaffinity_np(pthread_self(), sizeof(before), &before) == 0);
    }
#endif
    std::cout << "CPU worker unique row ownership / exceptions / 8000 phases / process cores passed\n";
}
