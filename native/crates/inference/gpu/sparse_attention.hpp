// Independent causal QSA with exact positive-float radix top-512 block selection.
__device__ float qsa_rope(float own, float partner, size_t column, size_t position) {
    if (column>=64) return own;
    const float angle=float(position)/powf(1e7f,float(column%32)/32.0f);
    return own*cosf(angle)+(column<32 ? -partner : partner)*sinf(angle);
}

__global__ void qsa_normalize(const float* source, const float* norm, size_t heads, size_t width,
                               size_t stride, size_t position, float* target, const uint32_t* dynamic_position=nullptr) {
    if (dynamic_position) position=*dynamic_position;
    const size_t head=blockIdx.x, token=head/heads, column=threadIdx.x;
    __shared__ float values[256], sums[256];
    const float x=source[head*stride+column]; values[column]=x*norm[column]; sums[column]=x*x;
    __syncthreads();
    for (unsigned offset=unsigned(width/2);offset;offset/=2) {
        if (column<offset) sums[column]+=sums[column+offset];
        __syncthreads();
    }
    const float denominator=sqrtf(sums[0]/float(width)+1e-6f);
    const float own=values[column]/denominator;
    const float partner=column<64 ? values[(column+32)%64]/denominator : 0;
    target[head*width+column]=qsa_rope(own,partner,column,position+token);
}

// The three single-token normalizations of QSA decode in one launch: blocks 0-23
// query heads (width 256, stride 512), 24-25 key heads, 26-29 index-query heads
// (width 128, in place); each block is exactly qsa_normalize's.
__device__ void qsa_normalize_block(const float* source, const float* norm, size_t head, size_t width, size_t stride,
                                    size_t position, float* target, size_t threads) {
    const size_t column=threadIdx.x;
    __shared__ float values[256], sums[256];
    float x=0;
    if (column<threads) { x=source[head*stride+column]; values[column]=x*norm[column]; sums[column]=x*x; }
    __syncthreads();
    for (unsigned offset=unsigned(width/2);offset;offset/=2) {
        if (column<offset) sums[column]+=sums[column+offset];
        __syncthreads();
    }
    if (column>=threads) return;
    const float denominator=sqrtf(sums[0]/float(width)+1e-6f);
    const float own=values[column]/denominator;
    const float partner=column<64 ? values[(column+32)%64]/denominator : 0;
    target[head*width+column]=qsa_rope(own,partner,column,position);
}
__global__ void qsa_normalize_decode(const float* qgate, const float* key, float* iq, const float* qnorm, const float* knorm,
                                     const float* iqnorm, float* target, const uint32_t* position) {
    const size_t block=blockIdx.x, pos=*position;
    if (block<24) qsa_normalize_block(qgate,qnorm,block,256,512,pos,target,256);
    else if (block<26) qsa_normalize_block(key,knorm,block-24,256,256,pos,target+6144,256);
    else qsa_normalize_block(iq,iqnorm,block-26,128,128,pos,iq,128);
}

__global__ void qsa_pool(const float* incoming, const float* history, const float* norm,
                         size_t pool, size_t base, float* blocks) {
    const size_t block=base/4+blockIdx.x, column=threadIdx.x;
    const size_t begin=block*4;
    float x=0;
    for (size_t tap=0;tap<4;++tap) {
        const size_t pos=begin+tap;
        x+=pos<base ? history[(pos%4)*128+column] : incoming[(pos-base)*128+column];
    }
    x/=4.0f;
    __shared__ float values[128], sums[128]; values[column]=x*norm[column]; sums[column]=x*x;
    __syncthreads();
    for (unsigned offset=64;offset;offset/=2) {
        if (column<offset) sums[column]+=sums[column+offset];
        __syncthreads();
    }
    const float denominator=sqrtf(sums[0]/128.0f+1e-6f);
    const float own=values[column]/denominator;
    const float partner=column<64 ? values[(column+32)%64]/denominator : 0;
    blocks[column*pool+block]=qsa_rope(own,partner,column,begin);
}

__global__ void qsa_save_history(const float* incoming, size_t base, size_t steps, float* history) {
    const size_t i=threadIdx.x+size_t(blockIdx.x)*256;
    const size_t kept=steps<4 ? steps : 4;
    if (i>=kept*128) return;
    const size_t token=steps-kept+i/128;
    history[((base+token)%4)*128+i%128]=incoming[token*128+i%128];
}

__global__ void qsa_block_scores(const float* queries, const float* blocks, size_t pool,
                                 size_t base, size_t steps, float* scores, const uint32_t* dynamic_position=nullptr) {
    if (dynamic_position) base=*dynamic_position;
    const size_t token=blockIdx.y, block=size_t(blockIdx.x)*256+threadIdx.x;
    const size_t complete=(base+token+1)/4;
    if (complete<=512 || block>=complete) return;
    float a=0,b=0,c=0,d=0;
    for (size_t col=0;col<128;++col) {
        const float key=blocks[col*pool+block];
        const float* query=queries+token*512+col;
        a+=key*query[0]; b+=key*query[128]; c+=key*query[256]; d+=key*query[384];
    }
    scores[token*pool+block]=fmaxf(a,0.0f)+fmaxf(b,0.0f)+fmaxf(c,0.0f)+fmaxf(d,0.0f);
}

// Inclusive prefix sum over a 256-thread block (eight waves of 32 lanes).
__device__ unsigned qsa_block_scan(unsigned value, unsigned* waves) {
    const unsigned lane=threadIdx.x%32, wave=threadIdx.x/32;
    for (unsigned offset=1;offset<32;offset*=2) {
        const unsigned other=__shfl_up(value,offset,32);
        if (lane>=offset) value+=other;
    }
    if (lane==31) waves[wave]=value;
    __syncthreads();
    unsigned before=0;
    for (unsigned w=0;w<wave;++w) before+=waves[w];
    __syncthreads();
    return value+before;
}

__global__ void qsa_select(const float* scores, size_t pool, size_t base, size_t steps,
                           size_t selected, float* packed, const uint32_t* dynamic_position=nullptr) {
    if (dynamic_position) base=*dynamic_position;
    const size_t token=blockIdx.x, position=base+token, complete=(position+1)/4, lane=threadIdx.x;
    const size_t index_start=steps*(24*256+1024);
    float* indices=packed+index_start+token*selected;
    const float* row=scores+token*pool;
    if (complete<=512) {
        // Every causal position is selected, in order.
        for (size_t i=lane;i<=position;i+=256) indices[i]=float(i);
        if (lane==0) packed[index_start+steps*selected+token]=float(position+1);
        return;
    }
    // Radix select of the 512th largest score, eight bits per pass. Each wave keeps
    // its own histogram; a suffix scan finds the bucket holding the 512th value.
    __shared__ unsigned counts[8*256], waves[8], threshold, remaining;
    if (lane==0) { threshold=0; remaining=512; }
    unsigned mask=0;
    for (int shift=24;shift>=0;shift-=8) {
        #pragma unroll
        for (int w=0;w<8;++w) counts[w*256+lane]=0;
        __syncthreads();
        const unsigned prefix=threshold, wanted=remaining;
        unsigned* own=counts+(lane/32)*256;
        for (size_t block=lane;block<complete;block+=256) {
            const unsigned value=__float_as_uint(row[block]);
            if ((value&mask)==prefix) atomicAdd(own+((value>>shift)&255),1u);
        }
        __syncthreads();
        const unsigned bucket=255-unsigned(lane);
        unsigned total=0;
        #pragma unroll
        for (int w=0;w<8;++w) total+=counts[w*256+bucket];
        const unsigned at_least=qsa_block_scan(total,waves);   // values in buckets >= bucket
        if (at_least>=wanted && at_least-total<wanted) {
            threshold=prefix|(bucket<<shift); remaining=wanted-(at_least-total);
        }
        mask|=255u<<shift;
        __syncthreads();
    }
    // Each lane owns a contiguous run of blocks; prefix sums keep block order and
    // take the earliest blocks equal to the threshold, exactly as a serial scan.
    const unsigned prefix=threshold, wanted=remaining;
    const size_t per=(complete+255)/256, first=std::min(complete,lane*per), last=std::min(complete,first+per);
    unsigned greater=0, equal=0;
    for (size_t block=first;block<last;++block) {
        const unsigned value=__float_as_uint(row[block]);
        greater+=value>prefix; equal+=value==prefix;
    }
    const unsigned equal_before=qsa_block_scan(equal,waves)-equal;
    const unsigned take_equal=equal_before>=wanted ? 0u : std::min(equal,wanted-equal_before);
    const unsigned taken=greater+take_equal;
    const unsigned taken_before=qsa_block_scan(taken,waves)-taken;
    size_t out=size_t(taken_before)*4; unsigned equal_seen=0;
    for (size_t block=first;block<last;++block) {
        const unsigned value=__float_as_uint(row[block]);
        bool take=value>prefix;
        if (value==prefix && equal_seen<take_equal) { take=true; ++equal_seen; }
        if (take) {
            #pragma unroll
            for (size_t tap=0;tap<4;++tap) indices[out+tap]=float(block*4+tap);
            out+=4;
        }
    }
    const size_t head=512*4;
    for (size_t i=lane;complete*4+i<=position;i+=256) indices[head+i]=float(complete*4+i);
    if (lane==0) packed[index_start+steps*selected+token]=float(head+position+1-complete*4);
}

__global__ void qsa_gate(float* output, const float* query_gate, size_t count) {
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if (i<count) output[i]*=hc_sigmoid(query_gate[(i/256)*512+256+i%256]);
}

void sparse_attention(Device& d, void* const* pointers, const int* kinds, void* kv_state, void* index_state,
                       void* scratch, size_t context, size_t base, size_t steps, const float* input, float* output, bool device_io=false) {
    if (!pointers || !kinds || !input || !output || !d.attention_scores || !steps || steps>256 ||
        !context || context>262144 || base>context || steps>context-base || steps*20096>d.output_capacity)
        throw std::runtime_error("invalid Straydog fused QSA geometry");
    const size_t pool=context/4+1;
    auto check_state=[&](void* pointer,size_t bytes) {
        auto found=d.allocations.find(pointer);
        if (found==d.allocations.end() || found->second!=bytes)
            throw std::runtime_error("invalid Straydog QSA state allocation");
    };
    check_state(kv_state,context*1024*4);
    check_state(index_state,(pool*128+512)*4);
    check_state(scratch,256*pool*4);
    const size_t rows[10]={12288,512,512,512,128,2560,1,1,1,1};
    const size_t cols[10]={2560,2560,2560,2560,2560,6144,256,256,128,128};
    Matrix matrices[10]{};
    for (size_t i=0;i<10;++i) {
        if (i>=6 && kinds[i]!=0) throw std::runtime_error("invalid QSA norm type");
        const size_t bytes=matrix_row_bytes(kinds[i],cols[i]);
        check_state(pointers[i],rows[i]*bytes);
        matrices[i]={static_cast<const uint8_t*>(pointers[i]),bytes,rows[i],cols[i],0,0,0,kinds[i]};
    }
    const size_t selected=std::min(base+steps,size_t(2052));
    if (device_io) check(hipMemcpyAsync(d.input,input,steps*2560*4,hipMemcpyDeviceToDevice,d.stream));
    else upload(d,d.input,input,steps*2560*4);
    float* qgate=d.output;
    float* key=qgate+steps*12288;
    float* value=key+steps*512;
    float* iq=value+steps*512;
    float* ik=iq+steps*512;
    float* targets[5]={qgate,key,value,iq,ik};
    Matrix descriptors[6][max_batch]{};
    if (steps==1) {
        Matrix projections[5];
        size_t offset=0;
        for (size_t i=0;i<5;++i) {
            projections[i]=matrices[i]; projections[i].output_offset=offset; offset+=matrices[i].rows;
        }
        gemv(d,projections,5,d.input,d.output);
    } else {
        for (size_t i=0;i<5;++i) hc_project(d,matrices[i],steps,d.input,targets[i],descriptors[i]);
    }
    qsa_normalize<<<unsigned(steps*24),256,0,d.stream>>>(qgate,static_cast<const float*>(pointers[6]),24,256,512,base,d.input);
    qsa_normalize<<<unsigned(steps*2),256,0,d.stream>>>(key,static_cast<const float*>(pointers[7]),2,256,256,base,d.input+steps*6144);
    check(hipMemcpyAsync(d.input+steps*(6144+512),value,steps*512*4,hipMemcpyDeviceToDevice,d.stream));
    qsa_normalize<<<unsigned(steps*4),128,0,d.stream>>>(iq,static_cast<const float*>(pointers[8]),4,128,128,base,iq);
    float* blocks=static_cast<float*>(index_state);
    float* history=blocks+pool*128;
    const size_t completed=(base+steps)/4-base/4;
    if (completed) qsa_pool<<<unsigned(completed),128,0,d.stream>>>(ik,history,static_cast<const float*>(pointers[9]),pool,base,blocks);
    qsa_save_history<<<unsigned((std::min(steps,size_t(4))*128+255)/256),256,0,d.stream>>>(ik,base,steps,history);
    if ((base+steps)/4>512)
        qsa_block_scores<<<dim3(unsigned((pool+255)/256),unsigned(steps)),256,0,d.stream>>>(iq,blocks,pool,base,steps,static_cast<float*>(scratch));
    qsa_select<<<unsigned(steps),256,0,d.stream>>>(static_cast<float*>(scratch),pool,base,steps,selected,d.input);
    const KvView kv=kv_for_prefill(d,kv_state,context,base,steps);
    write_attention_kv<<<unsigned((steps*512+255)/256),256,0,d.stream>>>(kv.kv,kv.capacity,base,steps,d.input,nullptr,nullptr,kv.mirror,kv.mirror_rows);
    float* attended=d.output+steps*13952;
    qsa_attention(d,kv.kv,kv.capacity,base,steps,selected,d.input,attended,nullptr,kv.mirror,kv.mirror_rows);
    kv_after_prefill(d,kv,kv_state,context,base,steps);
    qsa_gate<<<unsigned((steps*6144+255)/256),256,0,d.stream>>>(attended,qgate,steps*6144);
    hc_project(d,matrices[5],steps,attended,d.input,descriptors[5]);
    check(hipGetLastError());
    if (device_io) check(hipMemcpyAsync(output,d.input,steps*2560*4,hipMemcpyDeviceToDevice,d.stream));
    else download(d,output,d.input,steps*2560*4);
}
