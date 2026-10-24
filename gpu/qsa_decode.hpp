// Single-token QSA uses a GPU position scalar, so a replayed decoder graph
// advances causal state without rebuilding host kernel arguments each token.
__global__ void qsa_pool_one(const float* incoming,const float* norm,size_t pool,
                             const uint32_t* position,float* blocks,float* history) {
    const size_t column=threadIdx.x,pos=*position;
    history[(pos%qsa_history)*128+column]=incoming[column];
    if ((pos+1)%4) return;
    float x=0;
    for (size_t row=0;row<4;++row) x+=history[((pos-3+row)%qsa_history)*128+column];
    x/=4.0f;
    __shared__ float values[128],sums[128]; values[column]=x*norm[column]; sums[column]=x*x;
    __syncthreads();
    for (unsigned offset=64;offset;offset/=2) {
        if (column<offset) sums[column]+=sums[column+offset];
        __syncthreads();
    }
    const float denominator=sqrtf(sums[0]/128.0f+1e-6f);
    const float own=values[column]/denominator;
    const float partner=column<64 ? values[(column+32)%64]/denominator : 0;
    blocks[column*pool+pos/4]=qsa_rope(own,partner,column,pos-3);
}

// One token's attention from its projections (qgate | key | value | index query |
// index key, 13952 values): normalization, pooling, block selection, attention and
// the output gate, leaving the 6144 gated values in `attended`.
void qsa_decode_attend(Device& d,void* const* pointers,void* kv,void* index,void* scratch,size_t context,
                       const uint32_t* position,float* projections,float* attended) {
    float* qgate=projections; float* key=qgate+12288; float* value=key+512; float* iq=value+512; float* ik=iq+512;
    qsa_normalize_decode<<<30,256,0,d.stream>>>(qgate,key,iq,static_cast<const float*>(pointers[6]),
        static_cast<const float*>(pointers[7]),static_cast<const float*>(pointers[8]),d.input,position);
    const size_t pool=context/4+1;
    auto* blocks=static_cast<float*>(index);
    qsa_pool_one<<<1,128,0,d.stream>>>(ik,static_cast<const float*>(pointers[9]),pool,position,blocks,blocks+pool*128);
    qsa_block_scores<<<dim3(unsigned((pool+255)/256),1),256,0,d.stream>>>(iq,blocks,pool,0,1,static_cast<float*>(scratch),position);
    constexpr size_t selected=2052;
    qsa_select<<<1,256,0,d.stream>>>(static_cast<float*>(scratch),pool,0,1,selected,d.input,position);
    const KvView view=kv_decode_view(d,kv,context);
    write_attention_kv<<<2,256,0,d.stream>>>(static_cast<float*>(kv),context,0,1,d.input,position,value,
                                           view.mirror,view.mirror_rows);
    qsa_attention(d,kv,context,0,1,selected,d.input,attended,position,view.mirror,view.mirror_rows);
    qsa_gate<<<24,256,0,d.stream>>>(attended,qgate,6144);
}
void qsa_operands(void* const* pointers,const int* kinds,Matrix (&matrices)[6]) {
    const size_t rows[6]={12288,512,512,512,128,2560},cols[6]={2560,2560,2560,2560,2560,6144};
    for (size_t i=0;i<6;++i)
        matrices[i]={static_cast<const uint8_t*>(pointers[i]),matrix_row_bytes(kinds[i],cols[i]),rows[i],cols[i],0,0,0,kinds[i]};
}
void qsa_decode_device(Device& d,void* const* pointers,const int* kinds,void* kv,void* index,
                       void* scratch,size_t context,const uint32_t* position,const float* input,float* output,
                       const GemvResidual& residual={}) {
    Matrix matrices[6]{};
    qsa_operands(pointers,kinds,matrices);
    Matrix projections[5];
    size_t offset=0;
    for (size_t i=0;i<5;++i) {
        projections[i]=matrices[i]; projections[i].output_offset=offset; offset+=matrices[i].rows;
    }
    gemv(d,projections,5,input,d.output);
    float* attended=d.output+13952;
    qsa_decode_attend(d,pointers,kv,index,scratch,context,position,d.output,attended);
    if (residual.hidden) { gemv(d,&matrices[5],1,attended,nullptr,1,0,0,residual); return; }
    Matrix descriptors[max_batch]{};
    hc_project(d,matrices[5],1,attended,output,descriptors);
}
