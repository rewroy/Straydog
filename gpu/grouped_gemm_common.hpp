#pragma once
// The tables of the grouped expert GEMM (grouped_gemm.hpp), which the units that hold and read an expert layer share.
struct MoeGroup { Matrix weights[3]; size_t first,count; int half; };
struct MoeTile { uint32_t group,first; };
// Independently authored small-group GEMM. Packed weights are decoded into a
// shared K tile once, then reused by sixteen routed input rows in FP32 accumulators.
// An expert layer's shape: hidden (gate/up K, down N), the expert width ffn (gate/up N, down K)
// and the routed rows per token (shared rows follow all routed ones).
struct MoeShape { uint32_t hidden, ffn, top_k; };
