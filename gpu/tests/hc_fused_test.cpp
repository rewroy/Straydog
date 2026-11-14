// Regression of the fused HyperConnection up projection and finish (hc_fused.hpp) on a GPU with FP16 WMMA (gfx12): its mixed rows must be,
// bit for bit, hc_finish_hidden's of an F32 projection summed in the same order by a plain WMMA kernel (so everything after the product is
// hc_finish_hidden's arithmetic, expression for expression, fused multiply-adds and all), the injections must be the same bits, and its
// rows' error against an F64 evaluation of the same FP16 operands must be as small as the matrix cores' sums make it (checked against a
// fixed bound, a few F32 spacings of a row's scale). Standalone, as the other regressions:
//     hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mllvm -amdgpu-scalarize-global-loads=false hc_fused_test.cpp ../cpu_experts.cpp -lhipblas -o hc_fused_test
// (or the straydog_hc_fused_test target of `cmake -DSTRAYDOG_GPU_TESTS=ON`). Exit code 0: all cases agree.
#include "../kernels.cpp"
#include <cstdio>
#include <random>
#include <vector>

namespace {

// C[token][row] = sum_k W[row][k] X[token][k], one wave a 16 x 16 tile, K in steps of 16 from a zero accumulator, ascending (the order of
// hc_up_finish's sums: the same operand placement in the fragments).
__global__ void reference_projection(const _Float16* w, const _Float16* x, int steps, float* up) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    const int lane = threadIdx.x, h = lane >> 4, ln = lane & 15;
    const int row0 = blockIdx.x * 16, token0 = blockIdx.y * 16;
    hcf_float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const int token = min(token0 + ln, steps - 1);
    for (int k = 0; k < hcf_k; k += 16) {
        const hcf_half8 a = *reinterpret_cast<const hcf_half8*>(w + size_t(row0 + ln) * hcf_k + k + 8 * h);
        const hcf_half8 b = *reinterpret_cast<const hcf_half8*>(x + size_t(token) * hcf_k + k + 8 * h);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    if (token0 + ln < steps)
        for (int i = 0; i < 8; ++i) up[size_t(token0 + ln) * 10240 + row0 + 8 * h + i] = acc[i];
#endif
}

double sigmoid64(double x) { const double e = std::exp(-std::fabs(x)); return x >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e); }

bool run(int steps, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0, 1);
    const size_t M = 10240;
    std::vector<float> hidden(size_t(steps) * 10240), norm(10240), denominators(size_t(steps) * 4), injection(size_t(steps) * 4);
    std::vector<__half> w(M * hcf_k), x(size_t(steps) * hcf_k);
    for (size_t i = 0; i < hidden.size(); ++i) { float v = normal(rng); if ((i % 10240) % 509 == 0) v *= 9; hidden[i] = v; }
    for (auto& v : norm) v = 0.5f + float(rng() % 1000) / 1000.0f;
    for (int n = 0; n < steps; ++n)
        for (int s = 0; s < 4; ++s) {
            float sum = 0;
            for (int c = 0; c < 2560; ++c) { const float v = hidden[size_t(n) * 10240 + s * 2560 + c]; sum = std::fmaf(v, v, sum); }
            denominators[size_t(n) * 4 + s] = std::sqrt(sum / 2560.0f + 1e-6f);
        }
    for (auto& v : injection) v = normal(rng) * 2;
    for (auto& v : w) v = __float2half(normal(rng) * 0.05f);
    for (auto& v : x) v = __float2half(normal(rng) * 0.7f);
    float *d_hidden, *d_norm, *d_den, *d_inj, *d_up, *d_fused, *d_ref;
    __half *d_w, *d_x;
    const size_t out_floats = size_t(steps) * 2564;
    check(hipMalloc(&d_hidden, hidden.size() * 4)); check(hipMalloc(&d_norm, norm.size() * 4)); check(hipMalloc(&d_den, denominators.size() * 4));
    check(hipMalloc(&d_inj, injection.size() * 4)); check(hipMalloc(&d_up, size_t(steps) * 10240 * 4)); check(hipMalloc(&d_fused, out_floats * 4));
    check(hipMalloc(&d_ref, out_floats * 4)); check(hipMalloc(&d_w, w.size() * 2)); check(hipMalloc(&d_x, x.size() * 2));
    check(hipMemcpy(d_hidden, hidden.data(), hidden.size() * 4, hipMemcpyHostToDevice)); check(hipMemcpy(d_norm, norm.data(), norm.size() * 4, hipMemcpyHostToDevice));
    check(hipMemcpy(d_den, denominators.data(), denominators.size() * 4, hipMemcpyHostToDevice)); check(hipMemcpy(d_inj, injection.data(), injection.size() * 4, hipMemcpyHostToDevice));
    check(hipMemcpy(d_w, w.data(), w.size() * 2, hipMemcpyHostToDevice)); check(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    check(hipMemset(d_fused, 0xFF, out_floats * 4)); check(hipMemset(d_ref, 0xFF, out_floats * 4));
    reference_projection<<<dim3(unsigned(M / 16), unsigned((steps + 15) / 16)), 32>>>(reinterpret_cast<const _Float16*>(d_w), reinterpret_cast<const _Float16*>(d_x), steps, d_up);
    hc_finish_hidden<<<unsigned((size_t(steps) * 2560 + 255) / 256), 256>>>(d_hidden, d_norm, d_den, d_up, d_inj, size_t(steps), d_ref);
    const unsigned blocks = unsigned((hcf_columns / hcf_block_columns) * ((steps + hcf_block_tokens - 1) / hcf_block_tokens));
    hc_up_finish<<<blocks, 256>>>(d_w, d_x, d_hidden, d_norm, d_den, d_inj, steps, d_fused);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    std::vector<float> fused(out_floats), ref(out_floats);
    check(hipMemcpy(fused.data(), d_fused, out_floats * 4, hipMemcpyDeviceToHost)); check(hipMemcpy(ref.data(), d_ref, out_floats * 4, hipMemcpyDeviceToHost));
    size_t differing = 0, injections = 0;
    for (size_t i = 0; i < out_floats; ++i)
        if (std::memcmp(&fused[i], &ref[i], 4) != 0) { ++differing; if (i >= size_t(steps) * 2560) ++injections; }
    // the rows against an F64 evaluation of the same operands
    double worst_rms = 0, worst_max = 0;
    for (int n = 0; n < steps; n += std::max(1, steps / 24)) {
        std::vector<double> up(10240), out(2560);
        for (size_t m = 0; m < M; ++m) {
            double a = 0;
            for (int k = 0; k < hcf_k; ++k) a += double(__half2float(w[m * hcf_k + k])) * double(__half2float(x[size_t(n) * hcf_k + k]));
            up[m] = a;
        }
        double ss = 0;
        for (int c = 0; c < 2560; ++c) {
            double sum = 0;
            for (int s = 0; s < 4; ++s)
                sum += sigmoid64(up[s * 2560 + c]) * (double(hidden[size_t(n) * 10240 + s * 2560 + c]) / double(denominators[size_t(n) * 4 + s]) * double(norm[s * 2560 + c]));
            out[c] = sum / 4.0; ss += out[c] * out[c];
        }
        const double scale = std::sqrt(ss / 2560);
        double e2 = 0, emax = 0;
        for (int c = 0; c < 2560; ++c) { const double e = (double(fused[size_t(n) * 2560 + c]) - out[c]) / scale; e2 += e * e; emax = std::max(emax, std::fabs(e)); }
        worst_rms = std::max(worst_rms, std::sqrt(e2 / 2560)); worst_max = std::max(worst_max, emax);
    }
    const bool ok = differing == 0 && worst_rms < 3e-7 && worst_max < 1e-5;
    std::printf("steps %5d: %zu of %zu bits differ from the unfused order's finish (%zu of them injections); error against F64: rms %.2e max %.2e of a row's scale: %s\n",
                steps, differing, out_floats, injections, worst_rms, worst_max, ok ? "ok" : "FAILED");
    check(hipFree(d_hidden)); check(hipFree(d_norm)); check(hipFree(d_den)); check(hipFree(d_inj)); check(hipFree(d_up)); check(hipFree(d_fused));
    check(hipFree(d_ref)); check(hipFree(d_w)); check(hipFree(d_x));
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    for (int steps : {1, 8, 127, 128, 129, 300, 1024, 4096}) ok = run(steps, 11 + steps) && ok;
    std::printf("%s\n", ok ? "hc_fused_test: all cases agree" : "hc_fused_test: FAILED");
    return ok ? 0 : 1;
}
