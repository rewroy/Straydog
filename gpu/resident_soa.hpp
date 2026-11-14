// A bank's storage can be shared by plans. Finalize fixes its layout once, after
// the verified load; a small workspace leaves that bank native for every alias.
void resident_pack_bank(Device& d,Matrix& matrix,size_t count,void* workspace,bool enabled) {
    if (!expert_soa_eligible(matrix.kind,matrix.rows,matrix.cols)) return;
    const auto found=d.expert_layouts.find(matrix.data);
    if (found!=d.expert_layouts.end()) {
        const auto& layout=found->second;
        if (layout.packing) throw std::runtime_error("an expert bank conversion did not complete; discard this device");
        // Native aliases may intentionally reinterpret the same allocation.
        if (!layout.layout) { matrix.layout=0; return; }
        if (layout.kind!=matrix.kind || layout.rows!=matrix.rows || layout.cols!=matrix.cols || layout.count!=count)
            throw std::runtime_error("incompatible expert bank layouts");
        matrix.layout=layout.layout;
        return;
    }
    auto& layout=d.expert_layouts.emplace(matrix.data,ExpertBankLayout{matrix.kind,matrix.rows,matrix.cols,count,0}).first->second;
    const size_t bytes=matrix.rows*matrix.row_bytes;
    const auto scratch=d.allocations.find(workspace);
    if (!enabled || scratch==d.allocations.end() || scratch->second<bytes) return;
    layout.packing=true;
    const auto began=std::chrono::steady_clock::now();
    for (size_t expert=0;expert<count;++expert) {
        auto* source=const_cast<uint8_t*>(matrix.data)+expert*bytes;
        expert_soa_pack_launch(source,static_cast<uint8_t*>(workspace),matrix.kind,matrix.rows,matrix.cols,d.stream);
        check(hipMemcpyAsync(source,workspace,bytes,hipMemcpyDeviceToDevice,d.stream));
    }
    check(hipStreamSynchronize(d.stream));
    layout.packing=false;
    layout.layout=matrix.layout=1;
    const uint64_t elapsed=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-began).count());
    d.expert_repack_ns+=elapsed; d.expert_repack_bytes+=count*bytes;
    std::fprintf(stderr,"Straydog expert layout: %zu kind %d matrices, %zu bytes, repacked in %.6f s\n",count,matrix.kind,count*bytes,double(elapsed)*1e-9);
}

// Keep layout selection outside a paired row's loop. A per-unit branch rounds
// the scaled dot before the outer add, preventing the native loop's FMA.
template<int kind,bool packed>
__device__ __forceinline__ float resident_pair_sum(const Matrix& matrix,size_t row,unsigned sub,unsigned lanes,const float* input) {
    Matrix weights=matrix;
    if constexpr (packed) weights.layout=1;
    float sum=0;
    const size_t units=matrix.cols/GemvFormat<kind>::elements;
    #pragma unroll 2
    for (size_t unit=sub;unit<units;unit+=lanes) {
        if (row<matrix.rows) {
            float r[1];
            if constexpr (packed) expert_soa_dot<kind,1>(weights,row,unit,input,0,r);
            else GemvFormat<kind>::template dots<1>(matrix.data+row*matrix.row_bytes,unit,input,0,r);
            sum+=r[0];
        }
    }
    return sum;
}

// Qwen's packed expert rows retain the native GEMV's unit and reduction order.
template<int kind, int V>
__device__ __forceinline__ void resident_dots(const Matrix& m, size_t row, size_t unit,
                                             const float* input, uint32_t stride, float (&result)[V]) {
    if constexpr (kind==18 || kind==21 || kind==22)
        expert_soa_dot<kind,V>(m,row,unit,input,stride,result);
    else GemvFormat<kind>::template dots<V>(m.data+row*m.row_bytes,unit,input,stride,result);
}
template<int kind, int V>
__device__ void resident_rows(const Matrix& m,size_t first_row,unsigned lanes,const float* input,uint32_t input_stride,
                              float* output,uint32_t output_stride,const GemvResidual& residual={}) {
    if (!m.layout) { gemv_rows<kind,V>(m,first_row,lanes,input,input_stride,output,output_stride,residual); return; }
    const unsigned lane=threadIdx.x%32,sub=lane%lanes;
    const size_t row=first_row+lane/lanes,units=m.cols/GemvFormat<kind>::elements;
    const float* x=input+m.input_offset;
    float sum[V]={};
    constexpr int depth=V<=2 ? 2 : 1;
    #pragma unroll depth
    for (size_t unit=sub;unit<units;unit+=lanes) {
        if (row<m.rows) {
            float r[V]; resident_dots<kind,V>(m,row,unit,x,input_stride,r);
            #pragma unroll
            for (int v=0;v<V;++v) sum[v]+=r[v];
        }
    }
    #pragma unroll
    for (int v=0;v<V;++v) {
        float total=sum[v];
        for (unsigned offset=lanes/2;offset;offset/=2) total+=__shfl_xor(total,offset,32);
        if (sub==0 && row<m.rows) gemv_store(m,row,v,total,output,output_stride,residual);
    }
}
template<int V>
__device__ void resident_wave(const Matrix& m,size_t local_wave,unsigned lanes,const float* input,uint32_t input_stride,
                              float* output,uint32_t output_stride) {
    if (!m.layout) { gemv_wave<V>(m,local_wave,lanes,input,input_stride,output,output_stride); return; }
    if (!lanes) {
        const unsigned lane=threadIdx.x%32;
        float sum[V]={};
        if (local_wave<m.rows)
            for (size_t col=lane;col<m.cols;col+=32) {
                const float w=expert_soa_weight(m,local_wave,col);
                for (int v=0;v<V;++v) sum[v]+=w*input[m.input_offset+col+v*input_stride];
            }
        for (int v=0;v<V;++v) {
            float total=sum[v];
            for (unsigned offset=16;offset;offset/=2) total+=__shfl_xor(total,offset,32);
            if (lane==0 && local_wave<m.rows) gemv_store(m,local_wave,v,total,output,output_stride,{});
        }
        return;
    }
    #define SD_SOA_ROWS(kind) case kind: resident_rows<kind,V>(m,local_wave*(32/lanes),lanes,input,input_stride,output,output_stride); break;
    switch (m.kind) { SD_SOA_ROWS(18) SD_SOA_ROWS(21) SD_SOA_ROWS(22) }
    #undef SD_SOA_ROWS
}
