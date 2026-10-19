# Straydog native components

The GPU library (`gpu/`, HIP or CUDA, chosen at compile time) and the Rust GGUF
structure checker (`straydog-inspect`).

```bash
cargo test --locked --manifest-path native/Cargo.toml -p straydog
cargo run --manifest-path native/Cargo.toml -p straydog --bin straydog-inspect -- inspect /path/model.gguf
```

The checker does not run inference. It validates a GGUF v3 header with size limits
and checked arithmetic. `inspect-header` reads, on stdin, a header snapshot taken
from the same file handle the worker authenticated. Structure checks alone prove
neither authenticity nor correctness.

The GPU library implements compressed matrix-vector products and grouped GEMM for
the floating-point formats, Q2_0, Q4_0/Q5_0/Q8_0, Q3_K-Q6_K and IQ2_XS, IQ3_XXS,
IQ3_S, IQ4_NL/IQ4_XS. RDNA4 grouped GEMM uses its own WMMA kernels; general GEMM
uses hipBLAS or cuBLAS, in FP32 or, when configured, FP16 inputs with FP32
accumulation. DeltaNet recurrence and sparse attention have their own kernels.
Every format and state update is compared with a CPU reference in the tests.

`python -m straydog.provisioning build|bundle` builds these sources; a bundle places the result,
with a manifest pinning its digest and the native source digest, in
`src/straydog/native/`.
