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

// A verification's rows (up to four consecutive tokens, positions[0..steps)): qsa_decode_attend's
// kernels with a row to each block of a second grid dimension, every row's arithmetic the one
// token's. Row r's projections are at projections + r * stride (qgate | key | value | index query |
// index key), its attended values go to attended + r * 6144; the normalized queries, keys and the
// selections are in d.input in the layout of `steps` rows.
__global__ void qsa_normalize_rows(float* projections,size_t stride,const float* qnorm,const float* knorm,const float* iqnorm,
                                   float* target,size_t steps,const uint32_t* positions) {
    const size_t block=blockIdx.x,row=blockIdx.y,pos=positions[row];
    float* own=projections+row*stride;
    if (block<24) qsa_normalize_block(own,qnorm,block,256,512,pos,target+row*6144,256);
    else if (block<26) qsa_normalize_block(own+12288,knorm,block-24,256,256,pos,target+steps*6144+row*512,256);
    else qsa_normalize_block(own+13312,iqnorm,block-26,128,128,pos,own+13312,128);
}
// qsa_pool_one for each row in turn (a row completes a block from the history the rows before it wrote).
__global__ void qsa_pool_rows(const float* incoming,size_t stride,const float* norm,size_t pool,const uint32_t* positions,
                              size_t steps,float* blocks,float* history) {
    const size_t column=threadIdx.x;
    __shared__ float values[128],sums[128];
    for (size_t row=0;row<steps;++row) {
        const size_t pos=positions[row];
        history[(pos%qsa_history)*128+column]=incoming[row*stride+column];
        if ((pos+1)%4==0) {
            float x=0;
            for (size_t tap=0;tap<4;++tap) x+=history[((pos-3+tap)%qsa_history)*128+column];
            x/=4.0f;
            values[column]=x*norm[column]; sums[column]=x*x;
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
        __syncthreads();
    }
}
// write_attention_kv for each row: its key from the normalized keys, its value from the projections.
__global__ void write_attention_kv_rows(float* state,size_t context,size_t steps,const float* keys,const float* values,
                                        size_t value_stride,const uint32_t* positions,float* mirror,size_t mirror_rows) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=steps*512) return;
    const size_t step=index/512,head=(index%512)/256,column=index%256,pos=positions[step];
    const float value=values[step*value_stride+index%512];
    state[(head*context+pos)*256+column]=keys[index];
    state[context*512+(head*context+pos)*256+column]=value;
    if (pos<mirror_rows) {
        mirror[(head*mirror_rows+pos)*256+column]=keys[index];
        mirror[((2+head)*mirror_rows+pos)*256+column]=value;
    }
}
__global__ void qsa_gate_rows(float* attended,const float* query_gates,size_t stride) {
    const size_t row=blockIdx.y,i=size_t(blockIdx.x)*256+threadIdx.x;
    if (i<6144) attended[row*6144+i]*=hc_sigmoid(query_gates[row*stride+(i/256)*512+256+i%256]);
}
void qsa_decode_attend_rows(Device& d,void* const* pointers,void* kv,void* index,void* scratch,size_t context,
                            const uint32_t* positions,float* projections,size_t stride,float* attended,size_t steps) {
    constexpr size_t selected=2052;
    qsa_normalize_rows<<<dim3(30,unsigned(steps)),256,0,d.stream>>>(projections,stride,static_cast<const float*>(pointers[6]),
        static_cast<const float*>(pointers[7]),static_cast<const float*>(pointers[8]),d.input,steps,positions);
    const size_t pool=context/4+1;
    auto* blocks=static_cast<float*>(index);
    qsa_pool_rows<<<1,128,0,d.stream>>>(projections+13824,stride,static_cast<const float*>(pointers[9]),pool,positions,steps,
                                        blocks,blocks+pool*128);
    qsa_block_scores<<<dim3(unsigned((pool+255)/256),unsigned(steps)),256,0,d.stream>>>(projections+13312,blocks,pool,0,steps,
        static_cast<float*>(scratch),positions,stride);
    qsa_select<<<unsigned(steps),256,0,d.stream>>>(static_cast<float*>(scratch),pool,0,steps,selected,d.input,positions);
    const KvView view=kv_decode_view(d,kv,context);
    write_attention_kv_rows<<<unsigned((steps*512+255)/256),256,0,d.stream>>>(static_cast<float*>(kv),context,steps,
        d.input+steps*6144,projections+12800,stride,positions,view.mirror,view.mirror_rows);
    qsa_attention_rows(d,kv,context,steps,selected,d.input,attended,positions,view.mirror,view.mirror_rows);
    qsa_gate_rows<<<dim3(24,unsigned(steps)),256,0,d.stream>>>(attended,projections,stride);
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
