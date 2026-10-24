// DSpark feature injection, using the existing Q8 GEMV/GEMM, RMS and RoPE ops.
namespace {
// One block owns each ring row, so repeated positions (including N > 128) have a
// deterministic last writer. Other ring rows retain their previous contents.
__global__ void dspark_ring_last(const float* rotated, const int32_t* positions, size_t rows, float* ring) {
    __shared__ size_t source;
    if (!threadIdx.x) {
        source = rows;
        for (size_t i = rows; i; --i)
            if ((uint32_t(positions[i - 1]) & 127u) == blockIdx.x) { source = i - 1; break; }
    }
    __syncthreads();
    if (source == rows) return;
    for (size_t col = threadIdx.x; col < 512; col += blockDim.x)
        ring[size_t(blockIdx.x) * 512 + col] = rotated[source * 512 + col];
}
}

int sd_op_dspark_inject(void* handle, const float* features, const int32_t* positions, uint64_t rows,
                        const void* fc, const float* encoder_norm, const sd_dspark_inject_stage* stages,
                        const float* frequencies, float* encoded, float* scratch, uint64_t scratch_bytes) {
    SD_OP_BEGIN
    if (!rows || rows > INT32_MAX || !stages) throw std::runtime_error("invalid DSpark injection rows or stages");
    const size_t kv_bytes = checked_floats({rows, 512});
    const size_t working_bytes = checked_product({kv_bytes, 2});
    size_t linear_min = 0;
    if (rows > uint64_t(gemv_max_vectors)) {
        const size_t half_input = checked_sum({checked_product({rows, 15360, 2}), 255}) & ~size_t(255);
        linear_min = checked_sum({half_input, 15360 * 4});
    }
    if (scratch_bytes < checked_sum({working_bytes, linear_min}))
        throw std::runtime_error("DSpark injection scratch is too small");
    op_span(value, features, checked_floats({rows, 15360}), "DSpark target features");
    op_span(value, positions, checked_floats({rows}), "DSpark target positions");
    op_span(value, fc, 5120 * size_t(15360 / 32 * 34), "DSpark Q8 encoder");
    op_span(value, encoder_norm, 5120 * 4, "DSpark encoder norm");
    op_span(value, frequencies, 32 * 4, "DSpark rope frequencies");
    op_span(value, encoded, checked_floats({rows, 5120}), "DSpark encoded features");
    op_span(value, scratch, size_t(scratch_bytes), "DSpark injection scratch");
    for (size_t i = 0; i < 3; ++i) {
        op_span(value, stages[i].kv, 512 * size_t(5120 / 32 * 34), "DSpark Q8 KV projection");
        op_span(value, stages[i].norm, 512 * 4, "DSpark KV norm");
        op_span(value, stages[i].ring, 128 * 512 * 4, "DSpark K ring");
    }
    float* kv = scratch;
    float* rotated = scratch + kv_bytes / 4;
    float* linear_scratch = scratch + working_bytes / 4;
    const size_t linear_bytes = size_t(scratch_bytes) - working_bytes;
    const float eps = 1e-20f;
    op_linear(value, static_cast<const uint8_t*>(fc), 8, 5120, 15360, 0, features, 15360, size_t(rows),
              encoded, 5120, linear_scratch, linear_bytes);
    op_rms_norm_kernel<<<unsigned(rows), 256, 0, value.stream>>>(encoded, encoder_norm, 5120, eps, encoded);
    for (size_t i = 0; i < 3; ++i) {
        op_linear(value, static_cast<const uint8_t*>(stages[i].kv), 8, 512, 5120, 0, encoded, 5120, size_t(rows),
                  kv, 512, linear_scratch, linear_bytes);
        op_norm_rope_kernel<<<unsigned(rows), 256, 0, value.stream>>>(kv, stages[i].norm, 512, eps, 32,
            positions, frequencies, nullptr, rotated, nullptr, nullptr);
        dspark_ring_last<<<128, 256, 0, value.stream>>>(rotated, positions, size_t(rows), stages[i].ring);
    }
    SD_OP_END
}
