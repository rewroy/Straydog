// QSA attention over each token's explicit selection. The KV state stores keys
// then values, both token-major per KV head: [kv][token] rows of `kv_row` bytes (kv_t, kernels.cpp).

// Eight K/V values from 16 bytes (four of their pairs), as floats.
__device__ __forceinline__ void kv_unpack8(uint4 words, float* out) {
    const uint32_t w[4] = {words.x, words.y, words.z, words.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __half2 pair;
        memcpy(&pair, &w[i], 4);
        out[2 * i] = __low2float(pair); out[2 * i + 1] = __high2float(pair);
    }
}
// A K/V row's 32 values from `first` (a multiple of 32: one Q8_0 block), as floats.
__device__ __forceinline__ void kv_load32(const kv_t* row, size_t first, unsigned kv_row, float* out) {
    if (kv_row != kv_row_q8) {
        const uint4* words = reinterpret_cast<const uint4*>(row + first * 2);
        #pragma unroll
        for (size_t d = 0; d < 4; ++d) kv_unpack8(words[d], out + 8 * d);
        return;
    }
    const uint4* words = reinterpret_cast<const uint4*>(row + first);
    const float scale = kv_scale(row, first);
    #pragma unroll
    for (size_t w = 0; w < 2; ++w) {
        const uint4 word = words[w];
        const uint32_t parts[4] = {word.x, word.y, word.z, word.w};
        #pragma unroll
        for (size_t p = 0; p < 4; ++p)
            #pragma unroll
            for (size_t b = 0; b < 4; ++b) out[w * 16 + p * 4 + b] = float(int8_t(parts[p] >> (8 * b))) * scale;
    }
}
// Single-token flash decoding over the explicit causal selection. Each block
// owns 32 selected keys of one KV head and all twelve query heads sharing it;
// partial softmax statistics are merged exactly in F32 afterwards.
// Several consecutive tokens (a verification's rows, blockIdx.z) are the same blocks side by side:
// row r is a token at position[r] with its query, selection and partials in the layout of `steps`
// rows (steps = 1: the layout of one token).
constexpr size_t qsa_decode_chunk=32;
template<bool mirrored=false,int P=1>
__global__ void qsa_decode_parts(unsigned kv_row,const kv_t* state,size_t context,size_t base,size_t selected,size_t chunks,
                                 const float* input,float* partial,const uint32_t* position=nullptr,
                                 const kv_t* mirror=nullptr,size_t mirror_rows=0,size_t steps=1) { SD_KERNEL_ACQUIRE;
    using H=QwenHeads<P>;
    const size_t chunk=blockIdx.x,kv=blockIdx.y,row=blockIdx.z,t=threadIdx.x,lane=t%32,wave=t/32;
    const size_t index_start=steps*H::packed;
    const size_t count=size_t(input[index_start+steps*selected+row]);
    const size_t first=chunk*qsa_decode_chunk;
    if (first>=count) return;
    const size_t limit=position ? size_t(position[row]) : base;
    partial+=row*H::query*chunks*258;
    __shared__ float query[12*256],sums[8*12*32],probabilities[12*32];
    __shared__ int tokens[qsa_decode_chunk],slots[qsa_decode_chunk];   // (a key's row in the mirror, or -1)
    #pragma unroll
    for (size_t i=t;i<12*256;i+=256) query[i]=input[(row*H::query+kv*12)*256+i];
    if (t<qsa_decode_chunk) {
        const size_t slot=first+t;
        int token=-1;
        if (slot<count && slot<selected) {
            const float raw=input[index_start+row*selected+slot];
            if (raw>=0 && raw<=float(limit) && raw<float(context)) token=int(raw);
        }
        tokens[t]=token;
        if constexpr (mirrored) {
            const size_t last=position ? size_t(position[steps-1]) : base;
            slots[t]=token>=0 && size_t(token)+mirror_rows>last+kv_mirror_margin ? int(unsigned(token)%unsigned(mirror_rows)) : -1;
        }
    }
    __syncthreads();
    // Scores: lane owns one key, the wave owns 32 of its 256 dimensions.
    {
        const int token=tokens[lane];
        float accumulator[12]={};
        if (token>=0) {
            int slot=-1;
            if constexpr (mirrored) slot=slots[lane];
            const bool cached=slot>=0;
            const kv_t* source=cached ? mirror : state;
            const size_t capacity=cached ? mirror_rows : context;
            float k[32];
            kv_load32(source+(kv*capacity+size_t(cached ? slot : token))*kv_row,wave*32,kv_row,k);
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
    const kv_t* values=state+(H::kv+kv)*context*kv_row;
    float v[qsa_decode_chunk];
    #pragma unroll
    for (size_t key=0;key<qsa_decode_chunk;++key) {
        const int token=tokens[key];
        if constexpr (mirrored) {
            const kv_t* cached_values=mirror+(H::kv+kv)*mirror_rows*kv_row;
            const int slot=slots[key];
            v[key]=token<0 ? 0.0f : slot>=0 ? kv_load(cached_values+size_t(slot)*kv_row,t,kv_row) : kv_load(values+size_t(token)*kv_row,t,kv_row);
        } else
            v[key]=token<0 ? 0.0f : kv_load(values+size_t(token)*kv_row,t,kv_row);
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
// (gates: each value times the sigmoid of its query head's gate, qgate rows `gate_stride` apart, as qsa_gate did after it)
template<int P=1>
__global__ void qsa_decode_merge(const float* partial,size_t chunks,size_t selected,const float* input,float* output,
                                 size_t steps=1,const float* gates=nullptr,size_t gate_stride=0) { SD_KERNEL_ACQUIRE;
    using H=QwenHeads<P>;
    const size_t head=blockIdx.x,row=blockIdx.y,column=threadIdx.x;
    const size_t count=size_t(input[steps*H::packed+steps*selected+row]);
    const size_t used=(count+qsa_decode_chunk-1)/qsa_decode_chunk;
    const float* source=partial+(row*H::query+head)*chunks*258;
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
    float value=numerator/denominator;
    if (gates) value*=hc_sigmoid(gates[row*gate_stride+head*512+256+column]);
    output[(row*H::query+head)*256+column]=value;
}


// Multi-token causal attention over each token's explicit selection: one block per
// (token, KV head) streams its selected keys 64 at a time with an exact F32 online
// softmax, reusing every key and value for the twelve query heads.
constexpr size_t qsa_flash_chunk=64;
template<int P=1>
__global__ void __launch_bounds__(256) qsa_flash(unsigned kv_row,const kv_t* state,size_t context,size_t base,size_t steps,
                                                 size_t selected,const float* input,float* output,size_t first_token=0) {
    using H=QwenHeads<P>;
    const size_t token=first_token+blockIdx.x,kv=blockIdx.y,t=threadIdx.x,lane=t%32,wave=t/32;
    const size_t index_start=steps*H::packed;
    const size_t count=size_t(input[index_start+steps*selected+token]);
    const size_t limit=base+token;
    __shared__ SD_ALIGN(16) float query[12*256];
    __shared__ SD_ALIGN(16) float sums[8*12*qsa_flash_chunk];
    __shared__ SD_ALIGN(16) float probabilities[12*qsa_flash_chunk];
    __shared__ float rescale[12],running[2*12];
    __shared__ int tokens[qsa_flash_chunk];
    for (size_t i=t;i<12*256;i+=256) query[i]=input[(token*H::query+kv*12)*256+i];
    if (t<12) { running[t]=-INFINITY; running[12+t]=0; }
    float accumulator[12]={};
    const kv_t* keys_base=state+kv*context*kv_row;
    const kv_t* values_base=state+(H::kv+kv)*context*kv_row;
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
            const kv_t* k0=keys_base+size_t(key0<0 ? 0 : key0)*kv_row;
            const kv_t* k1=keys_base+size_t(key1<0 ? 0 : key1)*kv_row;
            #pragma unroll 2
            for (size_t d=0;d<32;d+=4) {
                const size_t c=wave*32+d;
                const float a[4]={kv_load(k0,c,kv_row),kv_load(k0,c+1,kv_row),kv_load(k0,c+2,kv_row),kv_load(k0,c+3,kv_row)};
                const float b[4]={kv_load(k1,c,kv_row),kv_load(k1,c+1,kv_row),kv_load(k1,c+2,kv_row),kv_load(k1,c+3,kv_row)};
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
                v[j]=index<0 ? 0.0f : kv_load(values_base+size_t(index)*kv_row,t,kv_row);
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
    for (size_t h=0;h<12;++h) output[(token*H::query+kv*12+h)*256+t]=accumulator[h]/running[12+h];
}
#if !defined(__HIPCC__)
// qsa_flash with the K/V format known when compiling and each lane's 32 key values read as 16-byte pieces (kv_load32) where the scores took
// them a value at a time: a warp's lanes read 32 different rows, so each scalar load was a request to 32 cache lines (L1 serves a line a
// cycle). The arithmetic is qsa_flash's, value for value and in the same order, so the outputs are its bits (random selections over 20K
// positions, 256 tokens, 0 of 1.5 M outputs differ): on an RTX 4090 256 tokens' attention takes 0.98 ms over Q8_0 rows (1.52) and 0.65 over
// FP16 rows (1.43).
template<bool Q8>
__device__ __forceinline__ void kv_load32c(const kv_t* row, size_t first, float* out) {
    if constexpr (!Q8) {
        const uint4* words = reinterpret_cast<const uint4*>(row + first * 2);
        #pragma unroll
        for (size_t d = 0; d < 4; ++d) kv_unpack8(words[d], out + 8 * d);
    } else {
        const uint4* words = reinterpret_cast<const uint4*>(row + first);
        const float scale = kv_scale(row, first);
        #pragma unroll
        for (size_t w = 0; w < 2; ++w) {
            const uint4 word = words[w];
            const uint32_t parts[4] = {word.x, word.y, word.z, word.w};
            #pragma unroll
            for (size_t p = 0; p < 4; ++p)
                #pragma unroll
                for (size_t b = 0; b < 4; ++b) out[w * 16 + p * 4 + b] = float(int8_t(parts[p] >> (8 * b))) * scale;
        }
    }
}
template<bool Q8>
__device__ __forceinline__ float kv_loadc(const kv_t* row, size_t column) {
    if constexpr (Q8) return float(reinterpret_cast<const int8_t*>(row)[column]) * kv_scale(row, column);
    else return __half2float(reinterpret_cast<const __half*>(row)[column]);
}
template<int P=1,bool Q8=true>
__global__ void __launch_bounds__(256) qsa_flash_wide(unsigned kv_row,const kv_t* state,size_t context,size_t base,size_t steps,
                                                 size_t selected,const float* input,float* output,size_t first_token=0) {
    using H=QwenHeads<P>;
    const size_t token=first_token+blockIdx.x,kv=blockIdx.y,t=threadIdx.x,lane=t%32,wave=t/32;
    const size_t index_start=steps*H::packed;
    const size_t count=size_t(input[index_start+steps*selected+token]);
    const size_t limit=base+token;
    __shared__ SD_ALIGN(16) float query[12*256];
    __shared__ SD_ALIGN(16) float sums[8*12*qsa_flash_chunk];
    __shared__ SD_ALIGN(16) float probabilities[12*qsa_flash_chunk];
    __shared__ float rescale[12],running[2*12];
    __shared__ int tokens[qsa_flash_chunk];
    for (size_t i=t;i<12*256;i+=256) query[i]=input[(token*H::query+kv*12)*256+i];
    if (t<12) { running[t]=-INFINITY; running[12+t]=0; }
    float accumulator[12]={};
    const kv_t* keys_base=state+kv*context*kv_row;
    const kv_t* values_base=state+(H::kv+kv)*context*kv_row;
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
            const kv_t* k0=keys_base+size_t(key0<0 ? 0 : key0)*kv_row;
            const kv_t* k1=keys_base+size_t(key1<0 ? 0 : key1)*kv_row;
            float ka[32],kb[32];
            kv_load32c<Q8>(k0,wave*32,ka);
            kv_load32c<Q8>(k1,wave*32,kb);
            #pragma unroll
            for (size_t d=0;d<32;d+=4) {
                const float a[4]={ka[d],ka[d+1],ka[d+2],ka[d+3]};
                const float b[4]={kb[d],kb[d+1],kb[d+2],kb[d+3]};
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
                v[j]=index<0 ? 0.0f : kv_loadc<Q8>(values_base+size_t(index)*kv_row,t);
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
    for (size_t h=0;h<12;++h) output[(token*H::query+kv*12+h)*256+t]=accumulator[h]/running[12+h];
}
// The prompt's QSA attention on tensor cores (qsa_flash_mma, Q8_0 K/V, from sm_80; STRAYDOG_QSA_MMA=0 keeps qsa_flash_wide): its attention with FP16 operands in
// mma.sync m16n8k16 and F32 accumulation, softmax and merge, the numeric form of the RDNA4 kernel (qsa_prompt_attention), not qsa_flash_wide's
// bits. A block is a token and a K/V head; each of its 4 warps takes every 4th tile of 16 selected keys, all 256 dimensions, with its own
// online softmax, and the warps merge at the end in warp order (the result does not depend on timing). The twelve query heads are the M = 16
// rows (four of zeros). S = Q K^T: a lane reads 64 consecutive bytes of a key row, its B operand for all 16 k-steps (the dimensions of a
// k-step are 64 q + 4 s + 0..3 for the lane quad's q, the queries permuted the same way). O += P V: S's C fragments are P's A fragment; a
// lane reads 32 consecutive bytes of each of its four key rows, so n-tile j of the output holds the dimensions 32 n + j. Q8_0 to FP16 as the
// RDNA4 kernel does it: the int8 + 128 behind 0x64 is the half 1024 + b + 128, minus 1152 exactly, times the scale.
// qsa_flash_wide is F32 SIMT work (10-11 TFLOP/s on an RTX 5070 Ti, 128 registers, 3 block barriers per 64 keys). One layer, 4096 rows at
// position 98304, P = 2, Q8_0 K/V, RTX 5070 Ti: qsa_flash_wide 9.3-10.2 ms, this 2.5-2.7 ms; RMS relative error against
// qsa_flash_wide 2.7-3.2e-4, worst 256-value head-row 4.0-4.8e-4, no non-finite value, also at positions 0-4095 and for selections of
// 1-37 keys (partial tiles).
__device__ __forceinline__ uint32_t qp_half2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b); uint32_t u; memcpy(&u, &h, 4); return u;
}
__device__ __forceinline__ uint32_t qp_dq(uint32_t biased, __half2 scale) {
    __half2 h; memcpy(&h, &biased, 4);
    h = __hmul2(__hsub2(h, __float2half2_rn(1152.0f)), scale);
    uint32_t u; memcpy(&u, &h, 4); return u;
}
__device__ __forceinline__ void qp_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800   // (sm_75 has no m16n8k16: qsa_flash_launch does not reach this kernel there)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ uint32_t qp_word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
constexpr unsigned qp_head = 4 * 72 + 1;   // merged: a head's 256 values as [quad q][64 + 8 pad]; + 1 so the 8 rows of a k-step miss each other's banks
template<int P>
__global__ void __launch_bounds__(128) qsa_flash_mma(unsigned kv_row, const kv_t* state, size_t context, size_t base, size_t steps,
                                                    size_t selected, const float* input, float* output, size_t first_token = 0) {
    using H = QwenHeads<P>;
    const unsigned t = threadIdx.x, lane = t & 31, wave = t >> 5, gid = lane >> 2, tig = lane & 3;
    const size_t token = first_token + blockIdx.x, kv = blockIdx.y;
    const size_t index_start = steps * H::packed;
    const size_t count = std::min(selected, size_t(input[index_start + steps * selected + token]));
    const float limit = float(base + token), bound = float(context);
    const float* list = input + index_start + token * selected;
    __shared__ uint4 qfrag[16 * 32];
    __shared__ float merged[12 * qp_head];
    __shared__ float stats[4][16][2];
    {
        const float* queries = input + (token * H::query + kv * 12) * 256;
        for (unsigned i = t; i < 16 * 32; i += 128) {
            const unsigned s = i / 32, L = i % 32, g = L >> 2, q = L & 3, d = 64 * q + 4 * s;
            const float4 a = *reinterpret_cast<const float4*>(queries + g * 256 + d);
            const float4 b = g + 8 < 12 ? *reinterpret_cast<const float4*>(queries + (g + 8) * 256 + d) : make_float4(0, 0, 0, 0);
            qfrag[i] = make_uint4(qp_half2(a.x, a.y), qp_half2(b.x, b.y), qp_half2(a.z, a.w), qp_half2(b.z, b.w));
        }
        for (unsigned i = t; i < 12 * qp_head; i += 128) merged[i] = 0.0f;
    }
    __syncthreads();
    const kv_t* keys = state + kv * context * kv_row;
    const kv_t* values = state + (H::kv + kv) * context * kv_row;
    float o[32][4];
    #pragma unroll
    for (int j = 0; j < 32; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;   // rows gid and gid + 8: running maximum, this lane's part of the sum
    auto key_at = [&](size_t slot, bool& ok) -> size_t {
        ok = false;
        if (slot < count) {
            const float raw = list[slot];
            if (raw >= 0 && raw <= limit && raw < bound) { ok = true; return size_t(raw); }
        }
        return 0;
    };
    for (size_t tile = wave; tile * 16 < count; tile += 4) {
        const size_t first = tile * 16;
        bool okk0, okk1, ok[4];
        const size_t k0 = key_at(first + gid, okk0), k1 = key_at(first + 8 + gid, okk1);
        const size_t r0 = key_at(first + 2 * tig, ok[0]), r1 = key_at(first + 2 * tig + 1, ok[1]);
        const size_t r2 = key_at(first + 2 * tig + 8, ok[2]), r3 = key_at(first + 2 * tig + 9, ok[3]);
        const kv_t* kr0 = keys + k0 * kv_row;
        const kv_t* kr1 = keys + k1 * kv_row;
        uint4 ka[4], kb[4], va[4][2];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ka[i] = *reinterpret_cast<const uint4*>(kr0 + 64 * tig + 16 * i);
            kb[i] = *reinterpret_cast<const uint4*>(kr1 + 64 * tig + 16 * i);
        }
        const uint32_t ksa = *reinterpret_cast<const uint32_t*>(kr0 + 256 + 4 * tig), ksb = *reinterpret_cast<const uint32_t*>(kr1 + 256 + 4 * tig);
        const kv_t* vr[4] = {values + r0 * kv_row, values + r1 * kv_row, values + r2 * kv_row, values + r3 * kv_row};
        __half vs[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            va[r][0] = *reinterpret_cast<const uint4*>(vr[r] + 32 * gid);
            va[r][1] = *reinterpret_cast<const uint4*>(vr[r] + 32 * gid + 16);
            vs[r] = *reinterpret_cast<const __half*>(vr[r] + 256 + 2 * gid);
        }
        __half2 sa_lo, sa_hi, sb_lo, sb_hi;
        {
            __half2 a, b; memcpy(&a, &ksa, 4); memcpy(&b, &ksb, 4);
            sa_lo = __low2half2(a); sa_hi = __high2half2(a); sb_lo = __low2half2(b); sb_hi = __high2half2(b);
        }
        float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            const uint4 q = qfrag[s * 32 + lane];
            const uint32_t A[4] = {q.x, q.y, q.z, q.w};
            const uint32_t wa = qp_word(ka[s / 4], s % 4) ^ 0x80808080u, wb = qp_word(kb[s / 4], s % 4) ^ 0x80808080u;
            const __half2 sa = s < 8 ? sa_lo : sa_hi, sb = s < 8 ? sb_lo : sb_hi;
            qp_mma(s0, A, qp_dq(__byte_perm(wa, 0x64646464u, 0x4140), sa), qp_dq(__byte_perm(wa, 0x64646464u, 0x4342), sa));
            qp_mma(s1, A, qp_dq(__byte_perm(wb, 0x64646464u, 0x4140), sb), qp_dq(__byte_perm(wb, 0x64646464u, 0x4342), sb));
        }
        // s0: slots 2 tig, 2 tig + 1 (rows gid, gid, gid + 8, gid + 8); s1: slots 2 tig + 8, 2 tig + 9.
        float x[8] = {ok[0] ? s0[0] * 0.0625f : -INFINITY, ok[1] ? s0[1] * 0.0625f : -INFINITY,
                      ok[2] ? s1[0] * 0.0625f : -INFINITY, ok[3] ? s1[1] * 0.0625f : -INFINITY,
                      ok[0] ? s0[2] * 0.0625f : -INFINITY, ok[1] ? s0[3] * 0.0625f : -INFINITY,
                      ok[2] ? s1[2] * 0.0625f : -INFINITY, ok[3] ? s1[3] * 0.0625f : -INFINITY};
        float mx0 = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3])), mx1 = fmaxf(fmaxf(x[4], x[5]), fmaxf(x[6], x[7]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float alpha0 = m0 == -INFINITY ? 0.0f : expf(m0 - n0), alpha1 = m1 == -INFINITY ? 0.0f : expf(m1 - n1);
        float p[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = x[i] == -INFINITY ? 0.0f : expf(x[i] - (i < 4 ? n0 : n1));
        l0 = l0 * alpha0 + (p[0] + p[1] + p[2] + p[3]);
        l1 = l1 * alpha1 + (p[4] + p[5] + p[6] + p[7]);
        m0 = n0; m1 = n1;
        if (!__all_sync(0xffffffffu, alpha0 == 1.0f && alpha1 == 1.0f)) {
            #pragma unroll
            for (int j = 0; j < 32; ++j) { o[j][0] *= alpha0; o[j][1] *= alpha0; o[j][2] *= alpha1; o[j][3] *= alpha1; }
        }
        const uint32_t PA[4] = {qp_half2(p[0], p[1]), qp_half2(p[4], p[5]), qp_half2(p[2], p[3]), qp_half2(p[6], p[7])};
        const __half2 sab = __halves2half2(vs[0], vs[1]), scd = __halves2half2(vs[2], vs[3]);
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t a = qp_word(va[0][w / 4], w % 4) ^ 0x80808080u, b = qp_word(va[1][w / 4], w % 4) ^ 0x80808080u;
            const uint32_t c = qp_word(va[2][w / 4], w % 4) ^ 0x80808080u, d = qp_word(va[3][w / 4], w % 4) ^ 0x80808080u;
            const uint32_t ab0 = __byte_perm(a, b, 0x5140), ab1 = __byte_perm(a, b, 0x7362);
            const uint32_t cd0 = __byte_perm(c, d, 0x5140), cd1 = __byte_perm(c, d, 0x7362);
            qp_mma(o[4 * w + 0], PA, qp_dq(__byte_perm(ab0, 0x64646464u, 0x4140), sab), qp_dq(__byte_perm(cd0, 0x64646464u, 0x4140), scd));
            qp_mma(o[4 * w + 1], PA, qp_dq(__byte_perm(ab0, 0x64646464u, 0x4342), sab), qp_dq(__byte_perm(cd0, 0x64646464u, 0x4342), scd));
            qp_mma(o[4 * w + 2], PA, qp_dq(__byte_perm(ab1, 0x64646464u, 0x4140), sab), qp_dq(__byte_perm(cd1, 0x64646464u, 0x4140), scd));
            qp_mma(o[4 * w + 3], PA, qp_dq(__byte_perm(ab1, 0x64646464u, 0x4342), sab), qp_dq(__byte_perm(cd1, 0x64646464u, 0x4342), scd));
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (tig == 0) { stats[wave][gid][0] = m0; stats[wave][gid][1] = l0; stats[wave][gid + 8][0] = m1; stats[wave][gid + 8][1] = l1; }
    __syncthreads();
    float M0 = -INFINITY, M1 = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 4; ++w) { M0 = fmaxf(M0, stats[w][gid][0]); M1 = fmaxf(M1, stats[w][gid + 8][0]); }
    const float f0 = m0 == -INFINITY ? 0.0f : expf(m0 - M0), f1 = m1 == -INFINITY ? 0.0f : expf(m1 - M1);
    for (unsigned w = 0; w < 4; ++w) {
        if (wave == w) {
            float* row0 = merged + gid * qp_head + tig * 72;
            #pragma unroll
            for (int j = 0; j < 32; ++j) { row0[j] += o[j][0] * f0; row0[32 + j] += o[j][1] * f0; }
            if (gid + 8 < 12) {
                float* row1 = merged + (gid + 8) * qp_head + tig * 72;
                #pragma unroll
                for (int j = 0; j < 32; ++j) { row1[j] += o[j][2] * f1; row1[32 + j] += o[j][3] * f1; }
            }
        }
        __syncthreads();
    }
    for (unsigned i = t; i < 12 * 256; i += 128) {
        const unsigned h = i / 256, d = i % 256;
        float M = -INFINITY, L = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) M = fmaxf(M, stats[w][h][0]);
        #pragma unroll
        for (int w = 0; w < 4; ++w) if (stats[w][h][0] != -INFINITY) L += stats[w][h][1] * expf(stats[w][h][0] - M);
        output[(token * H::query + kv * 12 + h) * 256 + d] = merged[h * qp_head + (d / 64) * 72 + d % 64] / L;
    }
}
#endif
// qsa_flash over `grid` (a block a token and K/V head): qsa_flash_wide on CUDA for the two K/V formats, qsa_flash where it is not compiled;
// over Q8_0 rows qsa_flash_mma from sm_80 unless STRAYDOG_QSA_MMA=0 (read once per Device). Only a prompt's batches come here: decode and
// verification passes attend through qsa_decode_parts (qsa_attention_rows).
template<int P=1>
void qsa_flash_launch(Device& d,dim3 grid,const kv_t* state,size_t context,size_t base,size_t steps,size_t selected,const float* input,
                      float* output,size_t first_token=0) {
#if !defined(__HIPCC__)
    if (d.kv_row==kv_row_q8) {
        if (d.qsa_mma<0) { const char* mode=std::getenv("STRAYDOG_QSA_MMA"); d.qsa_mma=mode && std::string(mode)=="0" ? 0 : 1; }
        if (d.qsa_mma && d.mma_f16) {
            qsa_flash_mma<P><<<grid,128,0,d.stream>>>(d.kv_row,state,context,base,steps,selected,input,output,first_token);
            return;
        }
        qsa_flash_wide<P,true><<<grid,256,0,d.stream>>>(d.kv_row,state,context,base,steps,selected,input,output,first_token);
        return;
    }
    if (d.kv_row==kv_row_f16) {
        qsa_flash_wide<P,false><<<grid,256,0,d.stream>>>(d.kv_row,state,context,base,steps,selected,input,output,first_token);
        return;
    }
#endif
    qsa_flash<P><<<grid,256,0,d.stream>>>(d.kv_row,state,context,base,steps,selected,input,output,first_token);
}
template<int P=1>
void qsa_attention(Device& d,void* kv,size_t context,size_t base,size_t steps,size_t selected,
                    const float* input,float* output,const uint32_t* position=nullptr,
                    const kv_t* mirror=nullptr,size_t mirror_rows=0) {
    using H=QwenHeads<P>;
    if (steps>1) {
        qsa_flash_launch<P>(d,dim3(unsigned(steps),unsigned(H::kv)),static_cast<const kv_t*>(kv),context,base,steps,selected,input,output);
        return;
    }
    const size_t chunks=(selected+qsa_decode_chunk-1)/qsa_decode_chunk;
    if (chunks>256) throw std::runtime_error("QSA decode selection exceeds the merge capacity");
    if (mirror)
        qsa_decode_parts<true,P><<<dim3(unsigned(chunks),unsigned(H::kv)),256,0,d.stream>>>(
            d.kv_row,static_cast<const kv_t*>(kv),context,base,selected,chunks,input,d.attention_scores,position,mirror,mirror_rows);
    else
        qsa_decode_parts<false,P><<<dim3(unsigned(chunks),unsigned(H::kv)),256,0,d.stream>>>(
            d.kv_row,static_cast<const kv_t*>(kv),context,base,selected,chunks,input,d.attention_scores,position);
    qsa_decode_merge<P><<<unsigned(H::query),256,0,d.stream>>>(d.attention_scores,chunks,selected,input,output);
}
// qsa_attention's single-token arithmetic for `steps` consecutive tokens in one launch of each kernel
// (their positions position[0..steps)): the layout of `steps` rows in `input`, row r's attended
// values at output + r * 6144.
template<int P=1>
void qsa_attention_rows(Device& d,void* kv,size_t context,size_t steps,size_t selected,
                        const float* input,float* output,const uint32_t* position,
                        const kv_t* mirror=nullptr,size_t mirror_rows=0,const float* gates=nullptr,size_t gate_stride=0) {
    using H=QwenHeads<P>;
    const size_t chunks=(selected+qsa_decode_chunk-1)/qsa_decode_chunk;
    if (chunks>256 || steps*H::query*chunks*258*4>attention_score_bytes)
        throw std::runtime_error("QSA decode selection exceeds the merge capacity");
    if (mirror)
        qsa_decode_parts<true,P><<<dim3(unsigned(chunks),unsigned(H::kv),unsigned(steps)),256,0,d.stream>>>(
            d.kv_row,static_cast<const kv_t*>(kv),context,0,selected,chunks,input,d.attention_scores,position,mirror,mirror_rows,steps);
    else
        qsa_decode_parts<false,P><<<dim3(unsigned(chunks),unsigned(H::kv),unsigned(steps)),256,0,d.stream>>>(
            d.kv_row,static_cast<const kv_t*>(kv),context,0,selected,chunks,input,d.attention_scores,position,nullptr,0,steps);
    qsa_decode_merge<P><<<dim3(unsigned(H::query),unsigned(steps)),256,0,d.stream>>>(d.attention_scores,chunks,selected,input,output,steps,
                                                                                   gates,gate_stride);
}

// Exclusive prefix sum over a block of `waves` waves of 32 lanes; `total` receives the sum.
template <unsigned waves>
__device__ unsigned qsa_rows_scan(unsigned value, unsigned* partial, unsigned& total) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    unsigned inclusive = value;
    for (unsigned offset = 1; offset < 32; offset *= 2) {
        const unsigned other = __shfl_up(inclusive, offset, 32);
        if (lane >= offset) inclusive += other;
    }
    if (lane == 31) partial[wave] = inclusive;
    __syncthreads();
    if (wave == 0) {
        unsigned sum = lane < waves ? partial[lane] : 0;
        for (unsigned offset = 1; offset < waves; offset *= 2) {
            const unsigned other = __shfl_up(sum, offset, 32);
            if (lane >= offset) sum += other;
        }
        if (lane < waves) partial[lane] = sum;
    }
    __syncthreads();
    total = partial[waves - 1];
    const unsigned before = wave ? partial[wave - 1] : 0;
    __syncthreads();
    return before + inclusive - value;
}

// Past the stage, kv_for_prefill hands back the host K/V, and every token (qsa_flash) or group of tokens
// (qsa_flash_wmma) would read its selected rows over the bus (12 times slower on an RTX 4060 past 47K tokens).
// Instead the union of a batch's selected blocks is copied into the stage once, in block order, and each
// selection renumbered to it: a token reads the same keys and values in the same order, so the same bits.
// marks[block] = 1 for every block a selection of tokens [first, first + gridDim.x) touches.
template<int P = 1>
__global__ void qsa_mark_selection(const float* packed, size_t steps, size_t selected, size_t first, uint32_t* marks) {
    const size_t token = first + blockIdx.x, index_start = steps * QwenHeads<P>::packed;
    const float* indices = packed + index_start + token * selected;
    const size_t count = std::min(selected, size_t(packed[index_start + steps * selected + token]));
    for (size_t i = threadIdx.x; i < count; i += 256) marks[size_t(indices[i]) / 4] = 1u;
}
// The span's union from its groups' marks (qsa_mark_blocks), groups [first, last).
__global__ void qsa_union_of_groups(const uint32_t* group_marks, size_t stride, size_t first, size_t last, size_t end,
                                    uint32_t* marks) {
    const size_t block = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (block >= end) return;
    uint32_t any = 0;
    for (size_t group = first; group < last; ++group) any |= group_marks[group * stride + block];
    marks[block] = any != 0;
}
// The union's places: marks[block] becomes the count of marked blocks before it, blocks[place] the
// block in that place, and marks[end] their count. One block of 1024 threads, each a run of blocks.
__global__ void __launch_bounds__(1024) qsa_place_union(uint32_t* marks, size_t end, uint32_t* blocks) {
    __shared__ unsigned partial[32];
    const size_t per = (end + 1023) / 1024, from = std::min(end, size_t(threadIdx.x) * per), to = std::min(end, from + per);
    unsigned own = 0;
    for (size_t block = from; block < to; ++block) own += marks[block] != 0;
    unsigned total = 0;
    unsigned at = qsa_rows_scan<32>(own, partial, total);
    for (size_t block = from; block < to; ++block) {
        const bool marked = marks[block] != 0;
        marks[block] = at;
        if (marked) blocks[at++] = uint32_t(block);
    }
    if (threadIdx.x == 0) marks[end] = total;
}
// Union place p's four rows of every plane, from the host K/V into rows 4p .. 4p + 3 of the stage.
__global__ void __launch_bounds__(256) qsa_gather_union(unsigned kv_row, const kv_t* state, size_t context, size_t planes,
                                                        const uint32_t* blocks, kv_t* stage, size_t capacity) {
    const size_t pieces = kv_row / 16;   // a row in 16-byte pieces
    const size_t place = blockIdx.x, block = blocks[place];
    for (size_t i = threadIdx.x; i < planes * 4 * pieces; i += 256) {
        const size_t plane = i / (4 * pieces), row = i / pieces % 4, piece = i % pieces, position = block * 4 + row;
        if (position >= context) continue;
        reinterpret_cast<uint4*>(stage + (plane * capacity + place * 4 + row) * kv_row)[piece] =
            reinterpret_cast<const uint4*>(state + (plane * context + position) * kv_row)[piece];
    }
}
// Tokens [first, first + gridDim.x)'s selections renumbered to the union's places.
template<int P = 1>
__global__ void qsa_renumber_selection(float* packed, size_t steps, size_t selected, size_t first, const uint32_t* places) {
    const size_t token = first + blockIdx.x, index_start = steps * QwenHeads<P>::packed;
    float* indices = packed + index_start + token * selected;
    const size_t count = std::min(selected, size_t(packed[index_start + steps * selected + token]));
    for (size_t i = threadIdx.x; i < count; i += 256) {
        const size_t index = size_t(indices[i]);
        indices[i] = float(size_t(places[index / 4]) * 4 + index % 4);
    }
}

// Batched prefill attention (RDNA4). Nearby tokens select overlapping blocks, so groups of
// qsa_group tokens attend together: each group walks the union of its tokens' blocks, reading
// every key and value once for all of them, and masks the blocks a token did not select. A token
// attends key j when it selected j's block and j is not after it, which is its explicit
// selection: complete blocks whole, the incomplete block up to the token.
constexpr size_t qsa_group = 16, qsa_group_capacity = qsa_group * 513;   // a token selects at most 513 blocks
// Bit (token % qsa_group) of marks[group * stride + block] for every block a token's selection
// touches (the lists are in block order, so a block's first entry marks it).
template<int P = 1>
__global__ void qsa_mark_blocks(const float* packed, size_t steps, size_t selected, size_t first, size_t stride,
                                uint32_t* marks) {
    const size_t token = blockIdx.x, index_start = steps * QwenHeads<P>::packed;
    const float* indices = packed + index_start + token * selected;
    const size_t count = size_t(packed[index_start + steps * selected + token]), row = first + token;
    uint32_t* group = marks + row / qsa_group * stride;
    for (size_t i = threadIdx.x; i < count; i += 256) {
        const size_t block = size_t(indices[i]) / 4;
        if (i == 0 || size_t(indices[i - 1]) / 4 != block) atomicOr(group + block, 1u << (row % qsa_group));
    }
}
// A part's share of a span's indexer (qsa_prefill<2>): the chunks of 256 rows go to the two devices in turn, each scores and selects
// its own, and the other's rows reach it as their selected blocks alone: the 512 block numbers of a row past 512 complete blocks, as
// 16 bits each (a context of 262144 tokens has 65536), where the whole selection was 2052 floats. A row of 512 blocks or fewer selects
// every causal block, and the tail block (the incomplete one, past the last complete) is where the row's position puts it: neither
// is sent. The marks come out as qsa_mark_blocks' (a bit set is the same bit set in any order).
constexpr size_t qsa_share_blocks = 512, qsa_share_ids = 4;   // selected blocks a row; a thread's run of them, one 8-byte store
// qsa_mark_blocks for a chunk whose rows' selection this device made, and the blocks of each row it sends: a row's 512 into slot
// `slot` halves of the other's inbox, at row `first + token` of the span. `base`: the chunk's first position.
template<int P = 1>
__global__ void __launch_bounds__(128) qsa_mark_send(const float* packed, size_t steps, size_t selected, size_t base, size_t first,
                                                     size_t stride, uint32_t* marks, uint16_t* peer_inbox, const uint32_t* state,
                                                     size_t slot) {
    const size_t token = blockIdx.x, index_start = steps * QwenHeads<P>::packed;
    const float* indices = packed + index_start + token * selected;
    const size_t row = first + token, position = base + token, complete = (position + 1) / 4;
    uint32_t* group = marks + row / qsa_group * stride;
    const uint32_t bit = 1u << (row % qsa_group);
    const unsigned t = threadIdx.x;
    if (complete <= qsa_share_blocks) {
        const size_t count = size_t(packed[index_start + steps * selected + token]);
        for (size_t i = t; i < count; i += 128) {
            const size_t block = size_t(indices[i]) / 4;
            if (i == 0 || size_t(indices[i - 1]) / 4 != block) atomicOr(group + block, bit);
        }
        return;
    }
    // (the selection is four positions to a block, in block order)
    uint16_t blocks[qsa_share_ids];
    #pragma unroll
    for (unsigned k = 0; k < qsa_share_ids; ++k) {
        const size_t block = size_t(indices[(t * qsa_share_ids + k) * 4]) / 4;
        atomicOr(group + block, bit);
        blocks[k] = uint16_t(block);
    }
    uint16_t* to = peer_inbox + size_t((state[0] + 1) & 1) * slot + row * qsa_share_blocks + t * qsa_share_ids;
    *reinterpret_cast<uint2*>(to) = make_uint2(uint32_t(blocks[0]) | uint32_t(blocks[1]) << 16, uint32_t(blocks[2]) | uint32_t(blocks[3]) << 16);
    if (t == 0 && complete * 4 <= position) atomicOr(group + complete, bit);
}
// qsa_mark_send without the marks, for a chunk whose rows attend their own selection (qsa_prompt_attention): only the blocks go to the
// other device, which attends them from its inbox (qsa_prompt_received). Every row selects 512 blocks (a chunk past 2048 positions).
template<int P = 1>
__global__ void __launch_bounds__(128) qsa_send_blocks(const float* packed, size_t steps, size_t selected, size_t first,
                                                       uint16_t* peer_inbox, const uint32_t* state, size_t slot) {
    const size_t token = blockIdx.x, index_start = steps * QwenHeads<P>::packed;
    const float* indices = packed + index_start + token * selected;
    const unsigned t = threadIdx.x;
    uint16_t blocks[qsa_share_ids];
    #pragma unroll
    for (unsigned k = 0; k < qsa_share_ids; ++k) blocks[k] = uint16_t(size_t(indices[(t * qsa_share_ids + k) * 4]) / 4);
    uint16_t* to = peer_inbox + size_t((state[0] + 1) & 1) * slot + (first + token) * qsa_share_blocks + t * qsa_share_ids;
    *reinterpret_cast<uint2*>(to) = make_uint2(uint32_t(blocks[0]) | uint32_t(blocks[1]) << 16, uint32_t(blocks[2]) | uint32_t(blocks[3]) << 16);
}
// The marks of the rows the other device selected (chunks first_chunk, first_chunk + 2, ... of the span's `rows`), from its inbox,
// after its flag: blockIdx.y the chunk's turn, blockIdx.x the row in it. `base`: the span's first position.
__global__ void __launch_bounds__(128) qsa_mark_received(size_t base, size_t first_chunk, size_t rows, size_t stride, uint32_t* marks,
                                                         const uint16_t* inbox, const uint32_t* state, size_t slot) {
    const size_t row = (first_chunk + 2 * blockIdx.y) * 256 + blockIdx.x;
    if (row >= rows) return;
    const size_t position = base + row, complete = (position + 1) / 4;
    uint32_t* group = marks + row / qsa_group * stride;
    const uint32_t bit = 1u << (row % qsa_group);
    const unsigned t = threadIdx.x;
    if (complete <= qsa_share_blocks) {
        for (size_t block = t; block <= position / 4; block += 128) atomicOr(group + block, bit);
        return;
    }
    const uint64_t pair = link_load64(inbox + size_t(state[0] & 1) * slot + row * qsa_share_blocks + t * qsa_share_ids);
    #pragma unroll
    for (unsigned k = 0; k < qsa_share_ids; ++k) atomicOr(group + ((pair >> (16 * k)) & 0xffffu), bit);
    if (t == 0 && complete * 4 <= position) atomicOr(group + complete, bit);
}
// The union's order: a pair of waves skips 16 keys none of its tokens selected, so blocks are
// sorted by the pairs that need them (block order within), which gathers a pair's blocks into
// fewer of those steps (a pair computes 0.3 of them at 131072 tokens, against 0.6 in block order).
__device__ inline uint32_t qsa_union_key(uint32_t mask) {
    uint32_t key = 0;
    #pragma unroll
    for (unsigned pair = 0; pair < qsa_group * 12 / 16; ++pair) {
        const uint32_t tokens = ((2u << ((pair * 16 + 15) / 12)) - 1) & ~((1u << (pair * 16 / 12)) - 1);
        key |= uint32_t((mask & tokens) != 0) << pair;
    }
    return key;
}
// Each group's marked blocks with their token masks, in qsa_union_key order: a counting sort in
// two passes over the group's marks, each pass's chunk of 1024 blocks placed in block order.
__global__ void __launch_bounds__(256) qsa_union_blocks(const uint32_t* marks, size_t stride, size_t base, size_t steps,
                                                        uint32_t* blocks, uint32_t* masks, uint32_t* counts) {
    constexpr unsigned keys = 1u << (qsa_group * 12 / 16);
    const size_t group = blockIdx.x, last = std::min(steps, (group + 1) * qsa_group) - 1, end = (base + last) / 4 + 1;
    const uint32_t* row = marks + group * stride;
    const unsigned t = threadIdx.x;
    __shared__ unsigned starts[keys], partial[8], chunk_keys[1024];
    for (unsigned i = t; i < keys; i += 256) starts[i] = 0;
    __syncthreads();
    for (size_t first = t * 4; first < end; first += 1024)
        for (unsigned j = 0; j < 4; ++j)
            if (first + j < end && row[first + j]) atomicAdd(&starts[qsa_union_key(row[first + j])], 1u);
    __syncthreads();
    // Exclusive starts: thread t owns keys 16t .. 16t + 15.
    unsigned own = 0;
    for (unsigned i = 0; i < keys / 256; ++i) own += starts[t * (keys / 256) + i];
    unsigned found = 0;
    unsigned at = qsa_rows_scan<8>(own, partial, found);
    for (unsigned i = 0; i < keys / 256; ++i) {
        const unsigned count = starts[t * (keys / 256) + i];
        starts[t * (keys / 256) + i] = at;
        at += count;
    }
    __syncthreads();
    for (size_t first = 0; first < end; first += 1024) {
        uint32_t mark[4], key[4];
        unsigned marked = 0;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const size_t block = first + t * 4 + j;
            mark[j] = block < end ? row[block] : 0u;
            key[j] = mark[j] ? qsa_union_key(mark[j]) : 0u;
            marked += mark[j] != 0;
        }
        unsigned total = 0;
        const unsigned before = qsa_rows_scan<8>(marked, partial, total);
        for (unsigned j = 0, slot = before; j < 4; ++j)
            if (mark[j]) chunk_keys[slot++] = key[j];
        __syncthreads();
        // A block's place: its key's start, after the chunk's earlier blocks of the same key.
        for (unsigned j = 0, slot = before; j < 4; ++j) {
            if (!mark[j]) continue;
            unsigned rank = 0;
            for (unsigned i = 0; i < slot; ++i) rank += chunk_keys[i] == key[j];
            const size_t out = group * qsa_group_capacity + starts[key[j]] + rank;
            blocks[out] = uint32_t(first + t * 4 + j);
            masks[out] = mark[j];
            ++slot;
        }
        __syncthreads();
        for (unsigned j = 0; j < 4; ++j)
            if (mark[j]) atomicAdd(&starts[key[j]], 1u);
        __syncthreads();
    }
    if (t == 0) counts[group] = found;
}
#if defined(__HIPCC__)
// One block per (group, KV head): 24 waves, a pair for each 16 of the group's 192 query rows
// (token-major, twelve heads per token), each wave of a pair half the 256 dimensions. Keys and
// values come eight union blocks (32 keys) at a time into LDS as FP16, the next tile's loads in
// flight while this one computes. Per 16 keys, S^T = K Q^T and O^T += V^T P^T are FP16 WMMA with
// F32 accumulation (keys as rows, so a lane's scores and output belong to one query row and P^T
// is already the B operand): the pair adds its two halves of S^T through LDS and both run the
// same F32 online softmax. A pair skips keys none of its tokens selected.
constexpr unsigned qsa_flash_blocks = 8;
template<int P = 1>
__global__ void __launch_bounds__(qsa_group * 48) qsa_flash_wmma(unsigned kv_row, const kv_t* state, size_t capacity, size_t base,
                                                                 size_t steps, const __half* half_queries,
                                                                 const uint32_t* union_blocks, const uint32_t* union_masks,
                                                                 const uint32_t* union_counts, float* output,
                                                                 const uint32_t* places = nullptr, unsigned first_group = 0) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr unsigned threads = qsa_group * 48, keys_per_tile = qsa_flash_blocks * 4;
    constexpr unsigned loads = keys_per_tile * 64, staged_per_thread = (loads + threads - 1) / threads;
    const unsigned group = first_group + blockIdx.x, kv = blockIdx.y, t = threadIdx.x, lane = t % 32, half = lane / 16;
    const unsigned wave = __builtin_amdgcn_readfirstlane(t / 32), pair = wave / 2, side = wave % 2;
    const unsigned row = pair * 16 + lane % 16, local = row / 12, head = row % 12;
    const size_t token = size_t(group) * qsa_group + local, position = base + token;
    const bool live = token < steps;
    // The tokens of this pair's rows: keys none of them selected are skipped.
    const uint32_t pair_tokens = ((2u << ((pair * 16 + 15) / 12)) - 1) & ~((1u << (pair * 16 / 12)) - 1);
    moe_half8 query[8];
    using H = QwenHeads<P>;
    const _Float16* source = reinterpret_cast<const _Float16*>(half_queries) + (token * H::query + kv * 12 + head) * 256 + side * 128 + 8 * half;
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k)
        query[k] = live ? *reinterpret_cast<const moe_half8*>(source + 16 * k) : moe_half8{};
    moe_float8 sums[8] = {};
    float maximum = -INFINITY, total = 0;
    // log2(e) / 16: scores are dot products / sqrt(256), exponentiated in base 2.
    constexpr float scale = 0.0625f * 1.4426950408889634f;
    typedef _Float16 half4 __attribute__((ext_vector_type(4)));
    __shared__ SD_ALIGN(16) _Float16 keys[keys_per_tile][264];       // [key][dimension], padded
    __shared__ SD_ALIGN(16) _Float16 values[256][keys_per_tile + 8];   // [dimension][key], padded
    __shared__ SD_ALIGN(16) moe_float8 halves[qsa_group * 48];         // each wave's half of S^T
    __shared__ uint32_t tile_blocks[qsa_flash_blocks], tile_marks[qsa_flash_blocks];
    const size_t count = union_counts[group];
    const uint32_t* blocks = union_blocks + size_t(group) * qsa_group_capacity;
    const uint32_t* marks = union_masks + size_t(group) * qsa_group_capacity;
    const kv_t* rows[2] = {state + kv * capacity * kv_row, state + (H::kv + kv) * capacity * kv_row};
    // Keys then values of a tile: blocks x 4 keys x 64 groups of four values, each as four FP16 (8 bytes; kv_half4).
    uint2 staged[2][staged_per_thread];
    auto load = [&](size_t first) {
        #pragma unroll
        for (unsigned which = 0; which < 2; ++which)
            #pragma unroll
            for (unsigned j = 0; j < staged_per_thread; ++j) {
                const unsigned i = t + j * threads, slot = i / 256;
                staged[which][j] = uint2{0, 0};
                if (i < loads && first + slot < count) {
                    // (gathered: the block's place in the stage, see qsa_gather_union)
                    const uint32_t block = places ? places[blocks[first + slot]] : blocks[first + slot];
                    staged[which][j] = kv_half4(rows[which] + (size_t(block) * 4 + i / 64 % 4) * kv_row, i % 64 * 4, kv_row);
                }
            }
    };
    auto store = [&](size_t first) {
        if (t < qsa_flash_blocks) {
            const bool valid = first + t < count;
            tile_blocks[t] = valid ? blocks[first + t] : 0u;
            tile_marks[t] = valid ? marks[first + t] : 0u;
        }
        #pragma unroll
        for (unsigned j = 0; j < staged_per_thread; ++j) {
            const unsigned i = t + j * threads, key = i / 64, column = i % 64 * 4;
            if (i >= loads) break;
            half4 k, v;
            memcpy(&k, &staged[0][j], 8); memcpy(&v, &staged[1][j], 8);
            *reinterpret_cast<half4*>(&keys[key][column]) = k;
            values[column][key] = v.x; values[column + 1][key] = v.y;
            values[column + 2][key] = v.z; values[column + 3][key] = v.w;
        }
    };
    load(0);
    for (size_t first = 0; first < count; first += qsa_flash_blocks) {
        store(first);
        __syncthreads();
        if (first + qsa_flash_blocks < count) load(first + qsa_flash_blocks);
        #pragma unroll
        for (unsigned part = 0; part < keys_per_tile / 16; ++part) {
            const uint32_t* tile = tile_marks + part * 4;
            const bool needed = (tile[0] | tile[1] | tile[2] | tile[3]) & pair_tokens;
            moe_float8 score = {};
            if (needed) {
                #pragma unroll
                for (unsigned k = 0; k < 8; ++k)
                    score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        *reinterpret_cast<const moe_half8*>(&keys[part * 16 + lane % 16][side * 128 + 16 * k + 8 * half]), query[k], score);
                halves[wave * 32 + lane] = score;
            }
            __syncthreads();
            // The same sum in both waves: the first half's scores plus the second's.
            const moe_float8 other = needed ? halves[(wave ^ 1) * 32 + lane] : moe_float8{};
            if (part + 1 < keys_per_tile / 16) __syncthreads();   // read before the next part writes
            if (!needed) continue;
            score = side ? other + score : score + other;
            // Lane: query row `row`, keys 8 * half .. + 7 of these 16 (two blocks).
            float p[8], highest = -INFINITY;
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned slot = part * 4 + half * 2 + i / 4;
                const size_t key = size_t(tile_blocks[slot]) * 4 + i % 4;
                const bool selected = live && (tile_marks[slot] >> local & 1u) && key <= position;
                p[i] = selected ? score[i] * scale : -INFINITY;
                highest = fmaxf(highest, p[i]);
            }
            highest = fmaxf(highest, __shfl_xor(highest, 16, 32));
            const float updated = fmaxf(maximum, highest), offset = updated == -INFINITY ? 0.0f : updated;
            float sum = 0;
            moe_half8 probabilities;
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                p[i] = exp2f(p[i] - offset);
                sum += p[i];
                probabilities[i] = _Float16(p[i]);
            }
            sum += __shfl_xor(sum, 16, 32);
            const float rescale = exp2f(maximum - offset);
            total = total * rescale + sum; maximum = updated;
            if (rescale != 1.0f) {
                #pragma unroll
                for (unsigned n = 0; n < 8; ++n) sums[n] *= rescale;
            }
            #pragma unroll
            for (unsigned n = 0; n < 8; ++n)
                sums[n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    *reinterpret_cast<const moe_half8*>(&values[side * 128 + 16 * n + lane % 16][part * 16 + 8 * half]),
                    probabilities, sums[n]);
        }
        __syncthreads();
    }
    if (!live) return;
    float* target = output + (token * H::query + kv * 12 + head) * 256 + side * 128 + 8 * half;
    #pragma unroll
    for (unsigned n = 0; n < 8; ++n) {
        const moe_float8 value = sums[n] / total;
        *reinterpret_cast<float4*>(target + 16 * n) = float4{value[0], value[1], value[2], value[3]};
        *reinterpret_cast<float4*>(target + 16 * n + 4) = float4{value[4], value[5], value[6], value[7]};
    }
#endif
}
#endif
