// Independent PLE arithmetic: exact n-gram rows are selected on CPU; the
// authenticated projections, gated four streams and causal convolution run here.
__global__ void ple_gate(const float* hidden,const float* key,const float* value,const float* key_norm,
                          const float* query_norm,float* gated) {
    const size_t item=blockIdx.x,token=item/4,stream=item%4,lane=threadIdx.x;
    float qsum=0,ksum=0,dot=0;
    for (size_t col=lane;col<2560;col+=256) {
        const float q=hidden[item*2560+col],k=key[item*2560+col];
        qsum+=q*q; ksum+=k*k; dot+=(q*query_norm[stream*2560+col])*(k*key_norm[stream*2560+col]);
    }
    __shared__ float qs[256],ks[256],ds[256]; qs[lane]=qsum; ks[lane]=ksum; ds[lane]=dot;
    __syncthreads();
    for (unsigned offset=128;offset;offset/=2) {
        if (lane<offset) { qs[lane]+=qs[lane+offset]; ks[lane]+=ks[lane+offset]; ds[lane]+=ds[lane+offset]; }
        __syncthreads();
    }
    float score=ds[0]/(sqrtf(qs[0]/2560.0f+1e-6f)*sqrtf(ks[0]/2560.0f+1e-6f)*sqrtf(2560.0f));
    if (score!=0) score=copysignf(sqrtf(fmaxf(fabsf(score),1e-6f)),score);
    const float gate=hc_sigmoid(score);
    for (size_t col=lane;col<2560;col+=256) gated[item*2560+col]=gate*value[token*2560+col];
}
__global__ void ple_convolve(float* hidden,const float* gated,const float* normalized,float* history,
                             const float* coefficients,size_t steps) {
    const size_t col=size_t(blockIdx.x)*256+threadIdx.x;
    if (col>=10240) return;
    // Offsets stay unsigned: for sm_120, ptxas (CUDA 12.8 and the driver 610 JIT) miscompiled the
    // signed form (int source=token-9+3*tap), reusing one tap's predicated address in another tap.
    for (size_t token=0;token<steps;++token) {
        float sum=0;
        for (size_t tap=0;tap<4;++tap) {
            const size_t back=9-3*tap;   // taps read 9, 6, 3 and 0 tokens back
            const float value=token<back ? history[(10+token-back)*10240+col] : normalized[(token-back)*10240+col];
            sum+=value*coefficients[col*4+tap];
        }
        hidden[token*10240+col]+=gated[token*10240+col]+sum*hc_sigmoid(sum);
    }
    for (size_t row=0;row<10;++row)
        history[row*10240+col]=steps+row<10 ? history[(steps+row)*10240+col] : normalized[(steps+row-10)*10240+col];
}
// resident!=nullptr updates those device rows in place instead of copying hidden/output.
// exact (at most four rows) projects every row with the one-row GEMV arithmetic,
// so speculative rows match token-by-token decode bitwise.
void ple(Device& d,void* const* pointers,const int* kinds,void* history,size_t steps,
          const float* embeddings,const float* hidden,float* output,float* resident=nullptr,bool exact=false) {
    if (!pointers || !kinds || (!resident && (!hidden || !output)) || !embeddings || !steps || steps>256 ||
        steps*33280>d.output_capacity || steps*10240>d.input_capacity)
        throw std::runtime_error("invalid PLE geometry");
    const size_t rows[6]={10240,2560,1,1,1,10240},cols[6]={2560,2560,10240,10240,10240,4};
    Matrix matrices[6]{};
    for (size_t i=0;i<6;++i) {
        const size_t bytes=matrix_row_bytes(kinds[i],cols[i]);
        auto found=d.allocations.find(pointers[i]);
        if (found==d.allocations.end() || found->second!=rows[i]*bytes || (i>=2 && kinds[i]!=0))
            throw std::runtime_error("invalid PLE operand");
        matrices[i]={static_cast<const uint8_t*>(pointers[i]),bytes,rows[i],cols[i],0,0,0,kinds[i]};
    }
    auto state=d.allocations.find(history);
    if (state==d.allocations.end() || state->second!=10*10240*4) throw std::runtime_error("invalid PLE state");
    upload(d,d.input,embeddings,steps*2560*4);
    float* key=d.output; float* value=key+steps*10240; float* residual=resident ? resident : value+steps*2560;
    float* gated=value+steps*2560+steps*10240;
    Matrix descriptors[2][max_batch]{};
    if (exact) {
        if (steps>gemv_max_vectors) throw std::runtime_error("invalid exact PLE rows");
        gemv(d,&matrices[0],1,d.input,key,steps,2560,10240);
        gemv(d,&matrices[1],1,d.input,value,steps,2560,2560);
    } else {
        hc_project(d,matrices[0],steps,d.input,key,descriptors[0]);
        hc_project(d,matrices[1],steps,d.input,value,descriptors[1]);
    }
    if (!resident) upload(d,residual,hidden,steps*10240*4);
    ple_gate<<<unsigned(steps*4),256,0,d.stream>>>(residual,key,value,static_cast<float*>(pointers[2]),
        static_cast<float*>(pointers[3]),gated);
    hc_normalize<<<unsigned(steps*4),256,0,d.stream>>>(gated,static_cast<float*>(pointers[4]),d.input);
    ple_convolve<<<40,256,0,d.stream>>>(residual,gated,d.input,static_cast<float*>(history),
        static_cast<float*>(pointers[5]),steps);
    check(hipGetLastError());
    if (!resident) download(d,output,residual,steps*10240*4);
    else check(hipStreamSynchronize(d.stream));
}
