// QSA attention over each token's explicit selection. The KV state stores keys
// then values, both token-major per KV head: [kv][token][256].
// Single-token flash decoding over the explicit causal selection. Each block
// owns 32 selected keys of one KV head and all twelve query heads sharing it;
// partial softmax statistics are merged exactly in F32 afterwards.
constexpr size_t qsa_decode_chunk=32;
template<bool mirrored=false>
__global__ void qsa_decode_parts(const float* state,size_t context,size_t base,size_t selected,size_t chunks,
                                 const float* input,float* partial,const uint32_t* position=nullptr,
                                 const float* mirror=nullptr,size_t mirror_rows=0) {
    const size_t chunk=blockIdx.x,kv=blockIdx.y,t=threadIdx.x,lane=t%32,wave=t/32;
    const size_t index_start=24*256+1024;
    const size_t count=size_t(input[index_start+selected]);
    const size_t first=chunk*qsa_decode_chunk;
    if (first>=count) return;
    const size_t limit=position ? size_t(*position) : base;
    __shared__ float query[12*256],sums[8*12*32],probabilities[12*32];
    __shared__ int tokens[qsa_decode_chunk];
    #pragma unroll
    for (size_t i=t;i<12*256;i+=256) query[i]=input[kv*12*256+i];
    if (t<qsa_decode_chunk) {
        const size_t slot=first+t;
        int token=-1;
        if (slot<count && slot<selected) {
            const float raw=input[index_start+slot];
            if (raw>=0 && raw<=float(limit) && raw<float(context)) token=int(raw);
        }
        tokens[t]=token;
    }
    __syncthreads();
    // Scores: lane owns one key, the wave owns 32 of its 256 dimensions.
    {
        const int token=tokens[lane];
        float accumulator[12]={};
        if (token>=0) {
            const bool cached=mirrored && size_t(token)<mirror_rows;
            const float* source=cached ? mirror : state;
            const size_t capacity=cached ? mirror_rows : context;
            const float4* keys=reinterpret_cast<const float4*>(source+(kv*capacity+size_t(token))*256+wave*32);
            float k[32];
            #pragma unroll
            for (size_t d=0;d<8;++d) {
                const float4 four=keys[d];
                k[4*d]=four.x; k[4*d+1]=four.y; k[4*d+2]=four.z; k[4*d+3]=four.w;
            }
            #pragma unroll
            for (size_t d=0;d<32;++d) {
                #pragma unroll
                for (size_t h=0;h<12;++h) accumulator[h]=fmaf(k[d],query[h*256+wave*32+d],accumulator[h]);
            }
        }
        #pragma unroll
        for (size_t h=0;h<12;++h) sums[(wave*12+h)*32+lane]=accumulator[h];
    }
    __syncthreads();
    // Softmax statistics per query head over this chunk.
    for (size_t h=wave;h<12;h+=8) {
        float score=-INFINITY;
        if (tokens[lane]>=0) {
            score=0;
            #pragma unroll
            for (size_t part=0;part<8;++part) score+=sums[(part*12+h)*32+lane];
            score*=0.0625f;
        }
        float maximum=score;
        for (unsigned offset=16;offset;offset/=2) maximum=fmaxf(maximum,__shfl_xor(maximum,offset,32));
        const float probability=score==-INFINITY ? 0.0f : expf(score-maximum);
        float total=probability;
        for (unsigned offset=16;offset;offset/=2) total+=__shfl_xor(total,offset,32);
        probabilities[h*32+lane]=probability;
        if (lane==0) {
            float* target=partial+((kv*12+h)*chunks+chunk)*258;
            target[256]=maximum; target[257]=total;
        }
    }
    __syncthreads();
    // Values: each thread owns one dimension for all twelve heads.
    float accumulator[12]={};
    const float* values=state+context*512+kv*context*256+t;
    float v[qsa_decode_chunk];
    #pragma unroll
    for (size_t key=0;key<qsa_decode_chunk;++key) {
        const int token=tokens[key];
        if constexpr (mirrored) {
            const float* cached_values=mirror+(2+kv)*mirror_rows*256+t;
            v[key]=token<0 ? 0.0f : (size_t(token)<mirror_rows ? cached_values : values)[size_t(token)*256];
        } else
            v[key]=token<0 ? 0.0f : values[size_t(token)*256];
    }
    #pragma unroll
    for (size_t key=0;key<qsa_decode_chunk;++key) {
        #pragma unroll
        for (size_t h=0;h<12;++h) accumulator[h]=fmaf(probabilities[h*32+key],v[key],accumulator[h]);
    }
    #pragma unroll
    for (size_t h=0;h<12;++h) partial[((kv*12+h)*chunks+chunk)*258+t]=accumulator[h];
}
// One block per query head; thread c owns chunk c's softmax statistics (at most
// 256 chunks, checked on the host) and then column c of the weighted sum.
__global__ void qsa_decode_merge(const float* partial,size_t chunks,size_t selected,const float* input,float* output) {
    const size_t head=blockIdx.x,column=threadIdx.x;
    const size_t count=size_t(input[24*256+1024+selected]);
    const size_t used=(count+qsa_decode_chunk-1)/qsa_decode_chunk;
    const float* source=partial+head*chunks*258;
    __shared__ float scale[256],reduce[8];
    const float own=column<used ? source[column*258+256] : -INFINITY;
    float maximum=own;
    for (unsigned offset=16;offset;offset/=2) maximum=fmaxf(maximum,__shfl_xor(maximum,offset,32));
    if (column%32==0) reduce[column/32]=maximum;
    __syncthreads();
    maximum=reduce[0];
    for (int wave=1;wave<8;++wave) maximum=fmaxf(maximum,reduce[wave]);
    const float factor=own==-INFINITY ? 0.0f : expf(own-maximum);
    scale[column]=factor;
    float denominator=column<used ? factor*source[column*258+257] : 0.0f;
    __syncthreads();
    for (unsigned offset=16;offset;offset/=2) denominator+=__shfl_xor(denominator,offset,32);
    if (column%32==0) reduce[column/32]=denominator;
    __syncthreads();
    denominator=0;
    for (int wave=0;wave<8;++wave) denominator+=reduce[wave];
    float numerator=0;
    #pragma unroll 8
    for (size_t chunk=0;chunk<used;++chunk) numerator+=scale[chunk]*source[chunk*258+column];
    output[head*256+column]=numerator/denominator;
}


// Multi-token causal attention over each token's explicit selection: one block per
// (token, KV head) streams its selected keys 64 at a time with an exact F32 online
// softmax, reusing every key and value for the twelve query heads.
constexpr size_t qsa_flash_chunk=64;
__global__ void __launch_bounds__(256) qsa_flash(const float* state,size_t context,size_t base,size_t steps,
                                                 size_t selected,const float* input,float* output) {
    const size_t token=blockIdx.x,kv=blockIdx.y,t=threadIdx.x,lane=t%32,wave=t/32;
    const size_t index_start=steps*(24*256+1024);
    const size_t count=size_t(input[index_start+steps*selected+token]);
    const size_t limit=base+token;
    __shared__ SD_ALIGN(16) float query[12*256];
    __shared__ SD_ALIGN(16) float sums[8*12*qsa_flash_chunk];
    __shared__ SD_ALIGN(16) float probabilities[12*qsa_flash_chunk];
    __shared__ float rescale[12],running[2*12];
    __shared__ int tokens[qsa_flash_chunk];
    for (size_t i=t;i<12*256;i+=256) query[i]=input[(token*24+kv*12)*256+i];
    if (t<12) { running[t]=-INFINITY; running[12+t]=0; }
    float accumulator[12]={};
    const float* keys_base=state+kv*context*256+wave*32;
    const float* values_base=state+context*512+kv*context*256+t;
    for (size_t first=0;first<count;first+=qsa_flash_chunk) {
        if (t<qsa_flash_chunk) {
            const size_t slot=first+t;
            int index=-1;
            if (slot<count && slot<selected) {
                const float raw=input[index_start+token*selected+slot];
                if (raw>=0 && raw<=float(limit) && raw<float(context)) index=int(raw);
            }
            tokens[t]=index;
        }
        __syncthreads();
        // Scores: each lane owns two keys, each wave 32 of the 256 dimensions.
        {
            const int key0=tokens[lane],key1=tokens[32+lane];
            float p0[12]={},p1[12]={};
            const float* k0=keys_base+size_t(key0<0 ? 0 : key0)*256;
            const float* k1=keys_base+size_t(key1<0 ? 0 : key1)*256;
            #pragma unroll 2
            for (size_t d=0;d<32;d+=4) {
                const float4 four0=*reinterpret_cast<const float4*>(k0+d), four1=*reinterpret_cast<const float4*>(k1+d);
                const float a[4]={four0.x,four0.y,four0.z,four0.w}, b[4]={four1.x,four1.y,four1.z,four1.w};
                #pragma unroll
                for (size_t h=0;h<12;++h) {
                    const float4 q=*reinterpret_cast<const float4*>(&query[h*256+wave*32+d]);
                    p0[h]=fmaf(a[0],q.x,fmaf(a[1],q.y,fmaf(a[2],q.z,fmaf(a[3],q.w,p0[h]))));
                    p1[h]=fmaf(b[0],q.x,fmaf(b[1],q.y,fmaf(b[2],q.z,fmaf(b[3],q.w,p1[h]))));
                }
            }
            #pragma unroll
            for (size_t h=0;h<12;++h) {
                sums[(wave*12+h)*qsa_flash_chunk+lane]=p0[h];
                sums[(wave*12+h)*qsa_flash_chunk+32+lane]=p1[h];
            }
        }
        __syncthreads();
        for (size_t h=wave;h<12;h+=8) {
            float s0=-INFINITY,s1=-INFINITY;
            if (tokens[lane]>=0) {
                s0=0;
                #pragma unroll
                for (size_t part=0;part<8;++part) s0+=sums[(part*12+h)*qsa_flash_chunk+lane];
                s0*=0.0625f;
            }
            if (tokens[32+lane]>=0) {
                s1=0;
                #pragma unroll
                for (size_t part=0;part<8;++part) s1+=sums[(part*12+h)*qsa_flash_chunk+32+lane];
                s1*=0.0625f;
            }
            float maximum=fmaxf(s0,s1);
            for (unsigned offset=16;offset;offset/=2) maximum=fmaxf(maximum,__shfl_xor(maximum,offset,32));
            const float previous=running[h], updated=fmaxf(previous,maximum);
            const float e0=s0==-INFINITY ? 0.0f : expf(s0-updated), e1=s1==-INFINITY ? 0.0f : expf(s1-updated);
            float total=e0+e1;
            for (unsigned offset=16;offset;offset/=2) total+=__shfl_xor(total,offset,32);
            probabilities[h*qsa_flash_chunk+lane]=e0; probabilities[h*qsa_flash_chunk+32+lane]=e1;
            if (lane==0) {
                const float scale=previous==-INFINITY ? 0.0f : expf(previous-updated);
                rescale[h]=scale; running[h]=updated; running[12+h]=running[12+h]*scale+total;
            }
        }
        __syncthreads();
        #pragma unroll
        for (size_t h=0;h<12;++h) accumulator[h]*=rescale[h];
        // A prvalue: std::min binds references, and CUDA device code cannot ODR-use the host constant.
        const size_t valid=std::min(size_t(qsa_flash_chunk),count-first);
        for (size_t key=0;key<valid;key+=4) {
            float v[4];
            #pragma unroll
            for (int j=0;j<4;++j) {
                const int index=tokens[key+j];
                v[j]=index<0 ? 0.0f : values_base[size_t(index)*256];
            }
            #pragma unroll
            for (size_t h=0;h<12;++h) {
                const float4 p=*reinterpret_cast<const float4*>(&probabilities[h*qsa_flash_chunk+key]);
                accumulator[h]=fmaf(p.x,v[0],fmaf(p.y,v[1],fmaf(p.z,v[2],fmaf(p.w,v[3],accumulator[h]))));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (size_t h=0;h<12;++h) output[(token*24+kv*12+h)*256+t]=accumulator[h]/running[12+h];
}
void qsa_attention(Device& d,void* kv,size_t context,size_t base,size_t steps,size_t selected,
                    const float* input,float* output,const uint32_t* position=nullptr,
                    const float* mirror=nullptr,size_t mirror_rows=0) {
    if (steps>1) {
        qsa_flash<<<dim3(unsigned(steps),2),256,0,d.stream>>>(static_cast<float*>(kv),context,base,steps,selected,input,output);
        return;
    }
    const size_t chunks=(selected+qsa_decode_chunk-1)/qsa_decode_chunk;
    if (chunks>256) throw std::runtime_error("QSA decode selection exceeds the merge capacity");
    if (mirror)
        qsa_decode_parts<true><<<dim3(unsigned(chunks),2),256,0,d.stream>>>(
            static_cast<float*>(kv),context,base,selected,chunks,input,d.attention_scores,position,mirror,mirror_rows);
    else
        qsa_decode_parts<false><<<dim3(unsigned(chunks),2),256,0,d.stream>>>(
            static_cast<float*>(kv),context,base,selected,chunks,input,d.attention_scores,position);
    qsa_decode_merge<<<24,256,0,d.stream>>>(d.attention_scores,chunks,selected,input,output);
}
