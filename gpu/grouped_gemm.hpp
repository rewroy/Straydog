// Independently authored small-group GEMM. Packed weights are decoded into a
// shared K tile once, then reused by sixteen routed input rows in FP32 accumulators.
struct MoeGroup { Matrix weights[3]; size_t first,count; int half; };
struct MoeTile { uint32_t group,first; };
// An expert layer's shape: hidden (gate/up K, down N), the expert width ffn (gate/up N, down K)
// and the routed rows per token (shared rows follow all routed ones).
struct MoeShape { uint32_t hidden, ffn, top_k; };

template<bool allow_soa=false>
__device__ float grouped_weight(const Matrix& matrix, size_t row, size_t column) {
    if constexpr (allow_soa) if (matrix.layout) return expert_soa_weight(matrix,row,column);
    const uint8_t* data=matrix.data+row*matrix.row_bytes;
    #define SD_GROUP(format) case format: return weight<format>(data,column);
    switch (matrix.kind) {
        SD_FORMATS(SD_GROUP)
    }
    #undef SD_GROUP
    return 0;
}

template<bool allow_soa=false>
__global__ void grouped_moe(const MoeGroup* groups, const MoeTile* tiles, const int32_t* positions,
                            size_t steps, bool down, const float* input, float* output,
                            const unsigned* tile_count=nullptr, int filter=0) {
    if (tile_count && blockIdx.y>=*tile_count) return;
    const MoeTile tile=tiles[blockIdx.y];
    if (filter && bool(groups[tile.group].half)!=(filter>0)) return;
    const size_t role=down ? 2 : blockIdx.z;
    const Matrix matrix=groups[tile.group].weights[role];
    const size_t first=groups[tile.group].first, count=groups[tile.group].count;
    const bool half=groups[tile.group].half!=0;
    const size_t lane=threadIdx.x, out_col=lane%64, out_row=lane/64;
    const size_t column=size_t(blockIdx.x)*64;
    __shared__ float x[16*32], w[64*32];
    float result[4]={0,0,0,0};
    for (size_t begin=0;begin<matrix.cols;begin+=32) {
        for (size_t i=lane;i<16*32;i+=256) {
            const size_t row=tile.first+i/32;
            float value=0;
            if (row<count) {
                const size_t route=size_t(positions[first+row]);
                const size_t token=down ? route : route<steps*10 ? route/10 : route-steps*10;
                value=input[token*matrix.cols+begin+i%32];
                if (half) value=__half2float(__float2half(value));
            }
            x[i]=value;
        }
        for (size_t i=lane;i<64*32;i+=256) {
            float value=grouped_weight<allow_soa>(matrix,column+i/32,begin+i%32);
            if (half) value=__half2float(__float2half(value));
            w[i]=value;
        }
        __syncthreads();
        for (size_t k=0;k<32;++k) {
            const float value=w[out_col*32+k];
            #pragma unroll
            for (size_t row=0;row<4;++row) result[row]+=x[(out_row+row*4)*32+k]*value;
        }
        __syncthreads();
    }
    #pragma unroll
    for (size_t row=0;row<4;++row) {
        const size_t member=tile.first+out_row+row*4;
        if (member<count) {
            const size_t route=size_t(positions[first+member]);
            output[route*(down ? 2560 : 1280)+(down ? 0 : role*640)+column+out_col]=result[row];
        }
    }
}

// RDNA4 fragments follow AMD's documented lane-group-split WMMA ABI.
// Other targets retain the portable tiled implementation above.
template<size_t tile_rows,bool allow_soa=false>
__global__ void grouped_moe_wmma(const MoeGroup* groups, const MoeTile* tiles, const int32_t* positions,
                                 size_t steps, bool down, const float* input, float* output,
                                 const unsigned* tile_count=nullptr, int filter=0) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    if (tile_count && blockIdx.y>=*tile_count) return;
    using Half8=_Float16 __attribute__((ext_vector_type(8)));
    using Float8=float __attribute__((ext_vector_type(8)));
    const MoeTile tile=tiles[blockIdx.y];
    if (filter && bool(groups[tile.group].half)!=(filter>0)) return;
    const size_t role=down ? 2 : blockIdx.z;
    const Matrix matrix=groups[tile.group].weights[role];
    const size_t first=groups[tile.group].first, count=groups[tile.group].count;
    if (tile_rows==32 && count<32) return;
    if (tile_rows==16 && filter==2 && count>=32) return;
    const size_t thread=threadIdx.x, wave=thread/32, lane=thread%32;
    const size_t column=size_t(blockIdx.x)*64;
    __shared__ _Float16 x[tile_rows*34], w[64*34];
    Float8 result{};
    for (size_t begin=0;begin<matrix.cols;begin+=32) {
        for (size_t i=thread;i<tile_rows*32;i+=tile_rows*8) {
            const size_t row=tile.first+i/32;
            float value=0;
            if (row<count) {
                const size_t route=size_t(positions[first+row]);
                const size_t token=down ? route : route<steps*10 ? route/10 : route-steps*10;
                value=input[token*matrix.cols+begin+i%32];
            }
            x[(i/32)*34+i%32]=_Float16(value);
        }
        if (matrix.kind==42) {
            for (size_t i=thread*4;i<64*32;i+=tile_rows*32) {
                const size_t col=begin+i%32;
                const uint8_t* packed=matrix.data+(column+i/32)*matrix.row_bytes+(col/64)*18;
                const unsigned code=packed[2+(col%64)/4];
                const float scale=half_at(packed);
                #pragma unroll
                for (size_t j=0;j<4;++j)
                    w[(i/32)*34+i%32+j]=_Float16(scale*float(int((code>>(j*2))&3)-1));
            }
        } else {
            for (size_t i=thread;i<64*32;i+=tile_rows*8)
                w[(i/32)*34+i%32]=_Float16(grouped_weight<allow_soa>(matrix,column+i/32,begin+i%32));
        }
        __syncthreads();
        #pragma unroll
        for (size_t panel=0;panel<32;panel+=16) {
            Half8 a,b;
            #pragma unroll
            for (size_t i=0;i<8;++i) {
                const size_t k=panel+(lane/16)*4+(i/4)*8+i%4;
                a[i]=x[((wave/4)*16+lane%16)*34+k];
                b[i]=w[((wave%4)*16+lane%16)*34+k];
            }
            result=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,result);
        }
        __syncthreads();
    }
    #pragma unroll
    for (size_t i=0;i<8;++i) {
        const size_t member=tile.first+(wave/4)*16+(lane/16)*8+i;
        if (member<count) {
            const size_t route=size_t(positions[first+member]);
            output[route*(down ? 2560 : 1280)+(down ? 0 : role*640)+column+(wave%4)*16+lane%16]=result[i];
        }
    }
#endif
}

