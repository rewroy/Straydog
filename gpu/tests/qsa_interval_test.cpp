// Regression of the interval indexer (qsa_interval.hpp) on a GPU with FP16 WMMA (gfx12), against the exact kernels it replaces, on synthetic
// pooled keys and queries: every row of every chunk must select the blocks, in the order and with the ties, the exact kernels do, and the
// interval around every score the matrix unit forms must hold the exact score. Standalone, as the CPU regressions are:
//     hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mllvm -amdgpu-scalarize-global-loads=false qsa_interval_test.cpp -lhipblas -o qsa_interval_test
// (or the straydog_qsa_interval_test target of `cmake -DSTRAYDOG_GPU_TESTS=ON`). Exit code 0: all cases agree.
#include "../kernels.cpp"
#include <cstdio>
#include <string>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 12345) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return uint32_t(s >> 16); }
    double uniform() { return (next() + 0.5) / 4294967296.0; }
    double normal() { return std::sqrt(-2 * std::log((next() % 1000003 + 0.5) / 1000003.0)) * std::cos(6.283185307179586 * ((next() % 1000003 + 0.5) / 1000003.0)); }
};

constexpr int P = 2;
using H = QwenHeads<P>;

// |a - e| <= the interval's half width, for every pair of the chunk
__global__ void bound_check(const float* exact, const float* approx, const float* info, size_t pool, size_t base, size_t steps,
                            unsigned long long* violations, float* worst, const float2* stats) {
    const size_t row = blockIdx.y, block = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (row >= steps || block >= (base + row + 1) / 4) return;
    const float width = fmaf(stats[block].x, info[row * 8 + 2], stats[block].y * info[row * 8 + 3]);
    const float error = fabsf(approx[row * pool + block] - exact[row * pool + block]);
    if (!(error <= width)) atomicAdd(violations, 1ull);
    if (width > 0) atomicMax(reinterpret_cast<unsigned*>(worst), __float_as_uint(error / width));
}

// The matrix unit's accumulation alone: its score of a pair against the sum of the FP16-rounded operands' products in double, relative to
// |h(k)| * sum of the heads' |h(q)| (the unit of the bound's 2^-15 term).
__global__ void accumulation_check(const float* approx, const float* queries, const float* blocks, size_t pool, size_t base, size_t steps, float* worst) {
    const size_t row = size_t(blockIdx.y) * 4, block = (size_t(blockIdx.x) * 256 + threadIdx.x) * 29;
    if (row >= steps || block >= (base + row + 1) / 4) return;
    double total = 0, norm = 0, k2 = 0;
    for (int h = 0; h < 4; ++h) {
        double dot = 0, q2 = 0;
        for (int c = 0; c < 128; ++c) {
            const double k = qsa_iv_half(blocks[size_t(c) * pool + block]), q = qsa_iv_half(queries[row * 512 + h * 128 + c]);
            dot += k * q; q2 += q * q;
            if (h == 0) k2 += k * k;
        }
        total += dot > 0 ? dot : 0; norm += std::sqrt(q2);
    }
    const float ratio = float(std::fabs(double(approx[row * pool + block]) - total) / (std::sqrt(k2) * norm + 1e-30));
    atomicMax(reinterpret_cast<unsigned*>(worst), __float_as_uint(ratio));
}

struct Case {
    char mode;   // g gaussian, t topics, T strong topics, d duplicates (exact ties at the 512th best score), n near duplicates, z zero queries,
                 // s small magnitudes (under FP16's normal range), o a key beyond FP16
    size_t base, steps;
    int chunks;
    bool bound;
};

bool run(const Case& c, size_t& left_rows) {
    const size_t pool = 65537, complete_all = (c.base + c.chunks * c.steps) / 4;
    Rng rng(7 + c.mode);
    const int topics = 64;
    std::vector<float> weight_k(128), weight_q(128), topic(topics * 128);
    for (auto& w : weight_k) w = float(0.4 + 1.2 * rng.uniform());
    for (auto& w : weight_q) w = float(0.4 + 1.2 * rng.uniform());
    for (auto& v : topic) v = float(rng.normal());
    const float strength = c.mode == 't' ? 1.0f : c.mode == 'T' || c.mode == 'd' || c.mode == 'n' ? 2.5f : 0.0f;
    const float scale = c.mode == 's' ? 1.0e-4f : 1.0f;
    std::vector<float> blocks(128 * pool, 0.0f), queries(c.chunks * c.steps * 512), x(128);
    for (size_t b = 0; b < complete_all; ++b) {
        const int z = int(((b / 48) * 2654435761u) % topics);
        double ss = 0;
        for (int i = 0; i < 128; ++i) { x[i] = float(rng.normal() + strength * topic[z * 128 + i] * 0.6); ss += double(x[i]) * x[i]; }
        const float inverse = float(1.0 / std::sqrt(ss / 128 + 1e-6));
        for (int i = 0; i < 128; ++i) blocks[size_t(i) * pool + b] = x[i] * weight_k[i] * inverse * scale;
    }
    for (size_t r = 0; r < c.chunks * c.steps; ++r)
        for (int h = 0; h < 4; ++h) {
            const int z = int((((c.base + r) / 700 + h * 5) * 40503u) % topics);
            double ss = 0;
            for (int i = 0; i < 128; ++i) { x[i] = float(rng.normal() + strength * topic[z * 128 + i] * 0.6); ss += double(x[i]) * x[i]; }
            const float inverse = float(1.0 / std::sqrt(ss / 128 + 1e-6));
            for (int i = 0; i < 128; ++i) queries[r * 512 + h * 128 + i] = x[i] * weight_q[i] * inverse * scale;
        }
    if (c.mode == 'z') for (size_t r = 0; r < c.chunks * c.steps; r += 7) std::fill(queries.begin() + r * 512, queries.begin() + (r + 1) * 512, 0.0f);
    if (c.mode == 'o') blocks[5 * pool + complete_all / 2] = 1.0e6f;

    float *d_blocks, *d_queries, *d_scores, *d_exact_scores, *d_packed_old, *d_packed_new;
    const size_t selected = std::min(c.base + c.steps, size_t(2052)), packed_floats = c.steps * H::packed + c.steps * 2052 + c.steps;
    Device d(0);
    d.rdna4 = true;
    check(hipMalloc(&d_blocks, blocks.size() * 4)); check(hipMalloc(&d_queries, c.steps * 512 * 4));
    check(hipMalloc(&d_scores, 256 * pool * 4)); check(hipMalloc(&d_exact_scores, 256 * pool * 4));
    check(hipMalloc(reinterpret_cast<void**>(&d.attention_scores), attention_score_bytes));
    check(hipMalloc(&d_packed_old, packed_floats * 4)); check(hipMalloc(&d_packed_new, packed_floats * 4));
    check(hipMemcpy(d_blocks, blocks.data(), blocks.size() * 4, hipMemcpyHostToDevice));
    check(hipMemset(d.attention_scores, 0, attention_score_bytes));
    auto exact_scores = [&](size_t base, size_t steps, float* scores, float* packed) {
        qsa_index_exact<P>(d, d_queries, d_blocks, pool, base, steps, std::min(base + steps, size_t(2052)), scores, packed);
    };
    auto fetch = [&](float* device, size_t steps) { std::vector<float> v(packed_floats); check(hipMemcpy(v.data(), device, packed_floats * 4, hipMemcpyDeviceToHost)); (void)steps; return v; };
    bool ok = true;
    size_t valid = 0;
    for (int chunk = 0; chunk < c.chunks; ++chunk) {
        const size_t base = c.base + chunk * c.steps, steps = c.steps, sel = std::min(base + steps, size_t(2052));
        check(hipMemcpy(d_queries, queries.data() + chunk * steps * 512, steps * 512 * 4, hipMemcpyHostToDevice));
        check(hipMemset(d_packed_old, 0, packed_floats * 4)); check(hipMemset(d_packed_new, 0, packed_floats * 4));
        exact_scores(base, steps, d_exact_scores, d_packed_old);
        if (c.mode == 'd' || c.mode == 'n') {
            // blocks around the 512th best of some rows given the keys of the 512th best: their exact scores tie (or nearly), for every row
            std::vector<float> scores(steps * pool);
            check(hipDeviceSynchronize());
            check(hipMemcpy(scores.data(), d_exact_scores, steps * pool * 4, hipMemcpyDeviceToHost));
            for (size_t r = 0; r < steps; r += std::max<size_t>(1, steps / 12)) {
                const size_t ct = (base + r + 1) / 4;
                std::vector<size_t> order(ct);
                for (size_t i = 0; i < ct; ++i) order[i] = i;
                std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[r * pool + a] > scores[r * pool + b] || (scores[r * pool + a] == scores[r * pool + b] && a < b); });
                for (int k = 0; k < 6; ++k) {
                    const size_t from = order[505 + 3 * k], to = order[512 + 2 * k + (k & 1)];
                    if (from == to) continue;
                    for (int i = 0; i < 128; ++i) {
                        float v = blocks[size_t(i) * pool + from];
                        if (c.mode == 'n') v *= 1.0f + float((rng.uniform() - 0.5) * 6e-5);
                        blocks[size_t(i) * pool + to] = v;
                    }
                }
            }
            check(hipMemcpy(d_blocks, blocks.data(), blocks.size() * 4, hipMemcpyHostToDevice));
            check(hipMemset(d_packed_old, 0, packed_floats * 4));
            exact_scores(base, steps, d_exact_scores, d_packed_old);
        }
        // the exact kernels' own use of the scores array, then the interval path as qsa_prefill calls it
        qsa_index_chunk<P>(d, d_queries, d_blocks, pool, base, steps, sel, d_scores, d_packed_new, true, &valid);
        check(hipDeviceSynchronize());
        const auto o = fetch(d_packed_old, steps), n = fetch(d_packed_new, steps);
        size_t different = 0;
        const size_t index_start = steps * H::packed;
        for (size_t t = 0; t < steps; ++t) {
            const float count_o = o[index_start + steps * sel + t], count_n = n[index_start + steps * sel + t];
            bool same = std::memcmp(&count_o, &count_n, 4) == 0;
            for (size_t i = 0; same && i < size_t(count_o); ++i) same = std::memcmp(&o[index_start + t * sel + i], &n[index_start + t * sel + i], 4) == 0;
            different += !same;
        }
        std::vector<unsigned> flags(257);
        check(hipMemcpy(flags.data(), reinterpret_cast<const unsigned*>(d.attention_scores + qsa_iv_flags_at(pool)), 257 * 4, hipMemcpyDeviceToHost));
        size_t left = 0;
        for (size_t t = 0; t < steps; ++t) left += flags[t];
        left_rows += left;
        std::printf("  mode %c base %zu steps %zu chunk %d: %zu of %zu rows differ, %zu rows left to the exact kernels%s\n", c.mode, base, steps, chunk, different, steps, left,
                    different ? "  <<< MISMATCH" : "");
        ok = ok && different == 0;
        // (the cases with plain keys must not lean on the exact kernels: a few rows may fail the sample's count)
        if (std::strchr("gtTdn", c.mode) && left * 10 > steps) { std::printf("  too many rows left to the exact kernels\n"); ok = false; }
    }
    if (c.bound) {
        // every pair: the scan with an open threshold keeps every score; the exact scores; the interval around each
        const size_t base = c.base + (c.chunks - 1) * c.steps, steps = c.steps, complete = (base + steps) / 4;
        float* info = d.attention_scores + qsa_iv_info_at(pool);
        std::vector<float> host(256 * 8);
        check(hipMemcpy(host.data(), info, 256 * 8 * 4, hipMemcpyDeviceToHost));
        for (size_t r = 0; r < steps; ++r) host[r * 8 + 1] = -INFINITY;
        check(hipMemcpy(info, host.data(), 256 * 8 * 4, hipMemcpyHostToDevice));
        const unsigned tiles = unsigned((complete + qsa_iv_unit - 1) / qsa_iv_unit), groups = unsigned((steps + qsa_iv_rows - 1) / qsa_iv_rows);
        qsa_iv_scan<qsa_iv_unit, false><<<dim3((tiles + qsa_iv_tiles_per_group - 1) / qsa_iv_tiles_per_group, groups), 32 * qsa_iv_waves, 0, d.stream>>>(
            d_queries, d.attention_scores + qsa_iv_copy_at(pool), pool, base, steps, info, d_scores,
            reinterpret_cast<uint16_t*>(d.attention_scores + qsa_iv_masks_at(pool)), unsigned(qsa_iv_words(pool)), d.attention_scores + qsa_iv_sample_at(pool),
            unsigned(complete), tiles, qsa_iv_tiles_per_group);
        exact_scores(base, steps, d_exact_scores, d_packed_old);
        check(hipDeviceSynchronize());
        unsigned long long* violations; float *worst, *accumulation;
        check(hipMalloc(&violations, 8)); check(hipMalloc(&worst, 4)); check(hipMalloc(&accumulation, 4));
        check(hipMemset(violations, 0, 8)); check(hipMemset(worst, 0, 4)); check(hipMemset(accumulation, 0, 4));
        bound_check<<<dim3(unsigned((complete + 255) / 256), unsigned(steps)), 256, 0, d.stream>>>(
            d_exact_scores, d_scores, info, pool, base, steps, violations, worst, reinterpret_cast<const float2*>(d.attention_scores + qsa_iv_stats_at(pool)));
        accumulation_check<<<dim3(unsigned((complete / 29 + 255) / 256), unsigned((steps + 3) / 4)), 256, 0, d.stream>>>(d_scores, d_queries, d_blocks, pool, base, steps, accumulation);
        check(hipDeviceSynchronize());
        unsigned long long v; float w, a;
        check(hipMemcpy(&v, violations, 8, hipMemcpyDeviceToHost)); check(hipMemcpy(&w, worst, 4, hipMemcpyDeviceToHost)); check(hipMemcpy(&a, accumulation, 4, hipMemcpyDeviceToHost));
        std::printf("  mode %c bound: %llu pairs outside the interval, largest |a - e| / half width %.3f; accumulation error %.3g of |h(k)| sum |h(q)| (allowed %.3g)\n", c.mode, v, w, a, 1.0 / 32768);
        ok = ok && v == 0 && w < 1.0f && a < 1.0f / 32768;
        check(hipFree(violations)); check(hipFree(worst)); check(hipFree(accumulation));
    }
    check(hipFree(d_blocks)); check(hipFree(d_queries)); check(hipFree(d_scores)); check(hipFree(d_exact_scores));
    check(hipFree(d_packed_old)); check(hipFree(d_packed_new));   // (the device frees its attention scratch)
    return ok;
}

}   // namespace

int main() {
    const Case cases[] = {
        {'g', 242000, 256, 1, true}, {'t', 242000, 256, 1, true}, {'T', 242000, 256, 1, true}, {'T', 95000, 256, 1, false},
        {'d', 242000, 256, 1, false}, {'n', 242000, 256, 1, false}, {'z', 242000, 256, 1, false}, {'s', 242000, 256, 1, false}, {'o', 242000, 256, 1, false},
        {'T', 241990, 37, 1, false}, {'t', 30000, 200, 1, false}, {'t', 240000, 256, 3, false}, {'g', 24000, 256, 3, false},
    };
    bool ok = true;
    size_t left = 0;
    for (const Case& c : cases) ok = run(c, left) && ok;
    std::printf("%s\n", ok ? "qsa_interval_test: all cases agree" : "qsa_interval_test: FAILED");
    return ok ? 0 : 1;
}
