// Single-token QSA uses a GPU position scalar, so a replayed decoder graph
// advances causal state without rebuilding host kernel arguments each token. Beside it lies the
// rope position, the same number but past an image (Device::rope_delta): the K/V, the blocks and
// the history go by `position`, the rotations by `rope`.
__global__ void qsa_pool_one(const float* incoming,const float* norm,size_t pool,
                             const uint32_t* position,const uint32_t* rope,float* blocks,float* history) { SD_KERNEL_ACQUIRE;
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
    blocks[column*pool+pos/4]=qsa_rope(own,partner,column,size_t(*rope)-3);
}

// One token's attention from its projections (qgate | key | value | index query |
// index key, 13952 values): normalization, pooling, block selection, attention and
// the output gate, leaving the 6144 gated values in `attended`.
void qsa_decode_attend(Device& d,void* const* pointers,void* kv,void* index,void* scratch,size_t context,
                       const uint32_t* position,const uint32_t* rope,float* projections,float* attended) {
    float* qgate=projections; float* key=qgate+12288; float* value=key+512; float* iq=value+512; float* ik=iq+512;
    qsa_normalize_decode<<<30,256,0,d.stream>>>(qgate,key,iq,static_cast<const float*>(pointers[6]),
        static_cast<const float*>(pointers[7]),static_cast<const float*>(pointers[8]),d.input,rope);
    const size_t pool=context/4+1;
    auto* blocks=static_cast<float*>(index);
    qsa_pool_one<<<1,128,0,d.stream>>>(ik,static_cast<const float*>(pointers[9]),pool,position,rope,blocks,blocks+pool*128);
    qsa_block_scores<<<dim3(unsigned((pool+255)/256),1),256,0,d.stream>>>(iq,blocks,pool,0,1,static_cast<float*>(scratch),position);
    constexpr size_t selected=2052;
    qsa_select_rows<1024><<<1,1024,0,d.stream>>>(static_cast<float*>(scratch),pool,0,1,selected,d.input,position);
    const KvView view=kv_decode_view(d,kv,context);
    write_attention_kv<<<2,256,0,d.stream>>>(static_cast<float*>(kv),context,0,1,d.input,position,value,
                                           view.mirror,view.mirror_rows);
    qsa_attention(d,kv,context,0,1,selected,d.input,attended,position,view.mirror,view.mirror_rows);
    qsa_gate<<<24,256,0,d.stream>>>(attended,qgate,6144);
}

// A verification's rows (up to four consecutive tokens, positions[0..steps), rope positions ropes[0..steps)): qsa_decode_attend's
// kernels with a row to each block of a second grid dimension, every row's arithmetic the one
// token's. Row r's projections are at projections + r * stride (qgate | key | value | index query |
// index key), its attended values go to attended + r * 6144; the normalized queries, keys and the
// selections are in d.input in the layout of `steps` rows.
// qsa_pool_one for each row in turn (a row completes a block from the history the rows before it wrote), by a block of 256 threads
// (columns 0..127 work; every thread passes the barriers).
__device__ void qsa_pool_rows_block(const float* incoming,size_t stride,const float* norm,size_t pool,const uint32_t* positions,
                                    const uint32_t* ropes,size_t steps,float* blocks,float* history) {
    const size_t column=threadIdx.x;
    const bool own=column<128;
    __shared__ float values[128],sums[128];
    for (size_t row=0;row<steps;++row) {
        const size_t pos=positions[row];
        if (own) history[(pos%qsa_history)*128+column]=incoming[row*stride+column];
        if ((pos+1)%4==0) {
            float x=0;
            if (own) {
                for (size_t tap=0;tap<4;++tap) x+=history[((pos-3+tap)%qsa_history)*128+column];
                x/=4.0f;
                values[column]=x*norm[column]; sums[column]=x*x;
            }
            __syncthreads();
            for (unsigned offset=64;offset;offset/=2) {
                if (column<offset) sums[column]+=sums[column+offset];
                __syncthreads();
            }
            if (own) {
                const float denominator=sqrtf(sums[0]/128.0f+1e-6f);
                const float value=values[column]/denominator;
                const float partner=column<64 ? values[(column+32)%64]/denominator : 0;
                blocks[column*pool+pos/4]=qsa_rope(value,partner,column,size_t(ropes[row])-3);
            }
        }
        __syncthreads();
    }
}
// A verification's rows' normalization (queries, keys and index queries, a block a head and row), their keys and values written to
// the K/V (and its mirror) by the keys' blocks as write_attention_kv did, and the index keys pooled (qsa_pool_one for each row in
// turn) by one more block: three launches in one, the same values.
template<int P=1>
__global__ void qsa_normalize_rows(float* projections,size_t stride,const float* qnorm,const float* knorm,const float* iqnorm,
                                   float* target,size_t steps,const uint32_t* positions,const uint32_t* ropes,float* state,size_t context,
                                   float* mirror,size_t mirror_rows,const float* pool_norm,size_t pool,float* blocks,float* history) { SD_KERNEL_ACQUIRE;
    using H=QwenHeads<P>;
    constexpr size_t keys=H::query*512, values=keys+H::kv*256, index=keys+H::kv*512;
    const size_t block=blockIdx.x,row=blockIdx.y,pos=positions[row],rope=ropes[row];
    float* own=projections+row*stride;
    if (block<H::query) qsa_normalize_block(own,qnorm,block,256,512,rope,target+row*H::query*256,256);
    else if (block<H::query+H::kv) {
        const size_t head=block-H::query,column=threadIdx.x;
        float* normalized=target+steps*H::query*256+row*H::kv*256;
        qsa_normalize_block(own+keys,knorm,head,256,256,rope,normalized,256);
        constexpr size_t width=H::kv*256;
        const float key=normalized[head*256+column], value=own[values+head*256+column];
        state[(head*context+pos)*256+column]=key;
        state[context*width+(head*context+pos)*256+column]=value;
        if (pos<mirror_rows) {
            mirror[(head*mirror_rows+pos)*256+column]=key;
            mirror[((H::kv+head)*mirror_rows+pos)*256+column]=value;
        }
    }
    else if (block<H::query+H::kv+4) qsa_normalize_block(own+index,iqnorm,block-H::query-H::kv,128,128,rope,own+index,128);
    else if (row==0) qsa_pool_rows_block(projections+index+512,stride,pool_norm,pool,positions,ropes,steps,blocks,history);
}
// P: the layer's part's heads (QwenHeads), its projections a row (qgate | key | value | index query | index key) and K/V.
template<int P=1>
void qsa_decode_attend_rows(Device& d,void* const* pointers,void* kv,void* index,void* scratch,size_t context,
                            const uint32_t* positions,const uint32_t* ropes,float* projections,size_t stride,float* attended,size_t steps) {
    using H=QwenHeads<P>;
    constexpr size_t selected=2052,queries=H::query*512+H::kv*512;
    const size_t pool=context/4+1;
    auto* blocks=static_cast<float*>(index);
    const KvView view=kv_decode_view(d,kv,context);
    qsa_normalize_rows<P><<<dim3(unsigned(H::query+H::kv+5),unsigned(steps)),256,0,d.stream>>>(projections,stride,
        static_cast<const float*>(pointers[6]),static_cast<const float*>(pointers[7]),static_cast<const float*>(pointers[8]),d.input,
        steps,positions,ropes,static_cast<float*>(kv),context,view.mirror,view.mirror_rows,static_cast<const float*>(pointers[9]),pool,
        blocks,blocks+pool*128);
    qsa_block_scores<<<dim3(unsigned((pool+255)/256),unsigned(steps)),256,0,d.stream>>>(projections+queries,blocks,pool,0,steps,
        static_cast<float*>(scratch),positions,stride);
    qsa_select_rows<1024,P><<<unsigned(steps),1024,0,d.stream>>>(static_cast<float*>(scratch),pool,0,steps,selected,d.input,positions);
    qsa_attention_rows<P>(d,kv,context,steps,selected,d.input,attended,positions,view.mirror,view.mirror_rows,projections,stride);
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
    qsa_decode_attend(d,pointers,kv,index,scratch,context,position,position+1,d.output,attended);
    if (residual.hidden) { gemv(d,&matrices[5],1,attended,nullptr,1,0,0,residual); return; }
    Matrix descriptors[max_batch]{};
    hc_project(d,matrices[5],1,attended,output,descriptors);
}
