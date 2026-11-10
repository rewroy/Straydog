// Split each mapped byte range independently so no range prefix is skipped at a worker boundary.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd_cpu {
struct ReadAheadChunk {
    size_t range;
    uint64_t offset, bytes;
};

class ReadAheadRanges {
public:
    static constexpr uint64_t unit = 128 * 1024;
    ReadAheadRanges(const uint64_t* bytes, size_t count) : bytes_(bytes) {
        ends_.reserve(count);
        uint64_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            total += bytes[i] / unit + uint64_t(bytes[i] % unit != 0);
            ends_.push_back(total);
        }
    }
    size_t size() const { return ends_.empty() ? 0 : size_t(ends_.back()); }
    ReadAheadChunk at(size_t index) const {
        const size_t range = size_t(std::upper_bound(ends_.begin(), ends_.end(), index) - ends_.begin());
        const uint64_t offset = (index - (range ? ends_[range - 1] : 0)) * unit;
        return {range, offset, std::min(unit, bytes_[range] - offset)};
    }
private:
    const uint64_t* bytes_;
    std::vector<uint64_t> ends_;
};
} // namespace sd_cpu
