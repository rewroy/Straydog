// Standalone: g++ -O2 -std=c++17 cpu_read_ahead_test.cpp -o cpu_read_ahead_test
#include "../cpu_read_ahead.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool ok) { if (!ok) throw std::runtime_error("CPU read-ahead range coverage regression"); }

static void covered(const std::vector<uint64_t>& sizes) {
    sd_cpu::ReadAheadRanges plan(sizes.data(), sizes.size());
    std::vector<uint64_t> reached(sizes.size(), 0);
    for (size_t i = 0; i < plan.size(); ++i) {
        const auto chunk = plan.at(i);
        require(chunk.range < sizes.size());
        require(chunk.offset == reached[chunk.range]);
        require(chunk.bytes > 0 && chunk.bytes <= sd_cpu::ReadAheadRanges::unit);
        require(chunk.offset + chunk.bytes <= sizes[chunk.range]);
        reached[chunk.range] += chunk.bytes;
    }
    require(reached == sizes);
}

int main() {
    // The original concatenated 128 KiB partition skipped the second range's first 31072 bytes.
    const uint64_t sizes[] = {100000, 100000};
    sd_cpu::ReadAheadRanges plan(sizes, 2);
    require(plan.size() == 2);
    require(plan.at(0).range == 0 && plan.at(0).offset == 0 && plan.at(0).bytes == 100000);
    require(plan.at(1).range == 1 && plan.at(1).offset == 0 && plan.at(1).bytes == 100000);
    const uint64_t unit = sd_cpu::ReadAheadRanges::unit;
    covered({});
    covered({0, 0});
    covered({100000, 100000});
    covered({0, unit + 17, 0, 3 * unit, 31});
    covered({unit, 2 * unit, unit - 1, 1});
    covered({473600, 473600, 460800});
    std::cout << "CPU read-ahead prefixes, tails, unequal and empty ranges passed\n";
}
