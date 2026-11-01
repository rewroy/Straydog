// QSA attention over each token's explicit selection. The KV state stores keys
// then values, both token-major per KV head: [kv][token][256].
// Single-token flash decoding over the explicit causal selection. Each block
// owns 32 selected keys of one KV head and all twelve query heads sharing it;
// partial softmax statistics are merged exactly in F32 afterwards.
// Several consecutive tokens (a verification's rows, blockIdx.z) are the same blocks side by side:
// row r is a token at position[r] with its query, selection and partials in the layout of `steps`
// rows (steps = 1: the layout of one token).
constexpr size_t qsa_decode_chunk=32;
template<bool mirrored=false,int P=1>
__global__ void qsa_decode_parts(const float* state,size_t context,size_t base,size_t selected,size_t chunks,
                                 const float* input,float* partial,const uint32_t* position=nullptr,
                                 const float* mirror=nullptr,size_t mirror_rows=0,size_t steps=1) { SD_KERNEL_ACQUIRE;
    using H=QwenHeads<P>;
    const size_t chunk=blockIdx.x,kv=blockIdx.y,row=blockIdx.z,t=threadIdx.x,lane=t%32,wave=t/32;
    const size_t index_start=steps*H::packed;
    const size_t count=size_t(input[index_start+steps*selected+row]);
    const size_t first=chunk*qsa_decode_chunk;
    if (first>=count) return;
    const size_t limit=position ? size_t(position[row]) : base;
    partial+=row*H::query*chunks*258;
    __shared__ float query[12*256],sums[8*12*32],probabilities[12*32];
    __shared__ int tokens[qsa_decode_chunk];
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
    const float* values=state+context*H::kv*256+kv*context*256+t;
    float v[qsa_decode_chunk];
    #pragma unroll
    for (size_t key=0;key<qsa_decode_chunk;++key) {
        const int token=tokens[key];
        if constexpr (mirrored) {
            const float* cached_values=mirror+(H::kv+kv)*mirror_rows*256+t;
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
__global__ void __launch_bounds__(256) qsa_flash(const float* state,size_t context,size_t base,size_t steps,
                                                 size_t selected,const float* input,float* output) {
    using H=QwenHeads<P>;
    const size_t token=blockIdx.x,kv=blockIdx.y,t=threadIdx.x,lane=t%32,wave=t/32;
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
    const float* keys_base=state+kv*context*256+wave*32;
    const float* values_base=state+context*H::kv*256+kv*context*256+t;
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
    for (size_t h=0;h<12;++h) output[(token*H::query+kv*12+h)*256+t]=accumulator[h]/running[12+h];
}
template<int P=1>
void qsa_attention(Device& d,void* kv,size_t context,size_t base,size_t steps,size_t selected,
                    const float* input,float* output,const uint32_t* position=nullptr,
                    const float* mirror=nullptr,size_t mirror_rows=0) {
    using H=QwenHeads<P>;
    if (steps>1) {
        qsa_flash<P><<<dim3(unsigned(steps),unsigned(H::kv)),256,0,d.stream>>>(static_cast<float*>(kv),context,base,steps,selected,input,output);
        return;
    }
    const size_t chunks=(selected+qsa_decode_chunk-1)/qsa_decode_chunk;
    if (chunks>256) throw std::runtime_error("QSA decode selection exceeds the merge capacity");
    if (mirror)
        qsa_decode_parts<true,P><<<dim3(unsigned(chunks),unsigned(H::kv)),256,0,d.stream>>>(
            static_cast<float*>(kv),context,base,selected,chunks,input,d.attention_scores,position,mirror,mirror_rows);
    else
        qsa_decode_parts<false,P><<<dim3(unsigned(chunks),unsigned(H::kv)),256,0,d.stream>>>(
            static_cast<float*>(kv),context,base,selected,chunks,input,d.attention_scores,position);
    qsa_decode_merge<P><<<unsigned(H::query),256,0,d.stream>>>(d.attention_scores,chunks,selected,input,output);
}
// qsa_attention's single-token arithmetic for `steps` consecutive tokens in one launch of each kernel
// (their positions position[0..steps)): the layout of `steps` rows in `input`, row r's attended
// values at output + r * 6144.
template<int P=1>
void qsa_attention_rows(Device& d,void* kv,size_t context,size_t steps,size_t selected,
                        const float* input,float* output,const uint32_t* position,
                        const float* mirror=nullptr,size_t mirror_rows=0,const float* gates=nullptr,size_t gate_stride=0) {
    using H=QwenHeads<P>;
    const size_t chunks=(selected+qsa_decode_chunk-1)/qsa_decode_chunk;
    if (chunks>256 || steps*H::query*chunks*258*4>attention_score_bytes)
        throw std::runtime_error("QSA decode selection exceeds the merge capacity");
    if (mirror)
        qsa_decode_parts<true,P><<<dim3(unsigned(chunks),unsigned(H::kv),unsigned(steps)),256,0,d.stream>>>(
            static_cast<float*>(kv),context,0,selected,chunks,input,d.attention_scores,position,mirror,mirror_rows,steps);
    else
        qsa_decode_parts<false,P><<<dim3(unsigned(chunks),unsigned(H::kv),unsigned(steps)),256,0,d.stream>>>(
            static_cast<float*>(kv),context,0,selected,chunks,input,d.attention_scores,position,nullptr,0,steps);
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
__global__ void __launch_bounds__(qsa_group * 48) qsa_flash_wmma(const float* state, size_t capacity, size_t base,
                                                                 size_t steps, const __half* half_queries,
                                                                 const uint32_t* union_blocks, const uint32_t* union_masks,
                                                                 const uint32_t* union_counts, float* output) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr unsigned threads = qsa_group * 48, keys_per_tile = qsa_flash_blocks * 4;
    constexpr unsigned loads = keys_per_tile * 64, staged_per_thread = (loads + threads - 1) / threads;
    const unsigned group = blockIdx.x, kv = blockIdx.y, t = threadIdx.x, lane = t % 32, half = lane / 16;
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
    const float* rows[2] = {state + kv * capacity * 256, state + capacity * H::kv * 256 + kv * capacity * 256};
    // Keys then values of a tile: blocks x 4 keys x 64 float4 each.
    float4 staged[2][staged_per_thread];
    auto load = [&](size_t first) {
        #pragma unroll
        for (unsigned which = 0; which < 2; ++which)
            #pragma unroll
            for (unsigned j = 0; j < staged_per_thread; ++j) {
                const unsigned i = t + j * threads, slot = i / 256;
                staged[which][j] = float4{0, 0, 0, 0};
                if (i < loads && first + slot < count)
                    staged[which][j] = *reinterpret_cast<const float4*>(
                        rows[which] + (size_t(blocks[first + slot]) * 4 + i / 64 % 4) * 256 + i % 64 * 4);
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
            const float4 k = staged[0][j], v = staged[1][j];
            *reinterpret_cast<half4*>(&keys[key][column]) = half4{_Float16(k.x), _Float16(k.y), _Float16(k.z), _Float16(k.w)};
            values[column][key] = _Float16(v.x); values[column + 1][key] = _Float16(v.y);
            values[column + 2][key] = _Float16(v.z); values[column + 3][key] = _Float16(v.w);
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
