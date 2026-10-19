// Independent full top-10 MoE. Every routed expert contributes; no tail skipping.
__global__ void moe_gather(const float* source, const int32_t* positions, size_t count,
                           size_t columns, size_t steps, bool products, float* target) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index>=count*columns) return;
    const size_t row=index/columns, col=index%columns, route=size_t(positions[row]);
    const size_t token=products ? route : route<steps*10 ? route/10 : route-steps*10;
    target[index]=source[token*columns+col];
}

__global__ void moe_scatter(const float* source, const int32_t* positions, size_t count,
                            size_t columns, size_t stride, size_t offset, float* target) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index<count*columns)
        target[size_t(positions[index/columns])*stride+offset+index%columns]=source[index];
}

__global__ void moe_product(const float* gate_up, size_t routes, float* product) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index>=routes*640) return;
    const size_t base=(index/640)*1280+index%640;
    const float gate=gate_up[base];
    product[index]=gate*hc_sigmoid(gate)*gate_up[base+640];
}

__global__ void moe_combine(const float* outputs, const float* probabilities, size_t steps, float* result) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if (index>=steps*2560) return;
    const size_t token=index/2560, col=index%2560;
    float sum=0;
    for (size_t expert=0;expert<10;++expert)
        sum+=probabilities[token*11+expert]*outputs[(token*10+expert)*2560+col];
    result[index]=sum+probabilities[token*11+10]*outputs[(steps*10+token)*2560+col];
}

void mixture_of_experts(Device& d, void* const* pointers, const int* kinds, size_t experts,
                        void* workspace, size_t steps, const int32_t* selected, const float* probabilities,
                        const float* input, float* output) {
    const size_t routes=steps*11, workspace_floats=routes*(1280+640+2560)+steps*11+routes+65536;
    if (!pointers || !kinds || !selected || !probabilities || !input || !output || !d.half_input ||
        !steps || steps>4096 || experts<10 || experts>512 || steps*2560>d.input_capacity)
        throw std::runtime_error("invalid Straydog MoE dimensions");
    auto storage=d.allocations.find(workspace);
    if (storage==d.allocations.end() || storage->second<workspace_floats*4)
        throw std::runtime_error("invalid Straydog MoE workspace");
    std::vector<Matrix> weights(experts*3+3);
    for (size_t i=0;i<weights.size();++i) {
        const size_t rows=i%3==2 ? 2560 : 640, cols=i%3==2 ? 640 : 2560;
        const size_t bytes=matrix_row_bytes(kinds[i],cols);
        auto found=d.allocations.find(pointers[i]);
        if (found==d.allocations.end() || found->second!=rows*bytes)
            throw std::runtime_error("invalid Straydog MoE weight size");
        weights[i]={static_cast<const uint8_t*>(pointers[i]),bytes,rows,cols,0,0,0,kinds[i]};
    }
    std::vector<std::vector<int32_t>> members(experts+1);
    for (size_t token=0;token<steps;++token) {
        for (size_t rank=0;rank<10;++rank) {
            const int32_t expert=selected[token*10+rank];
            if (expert<0 || size_t(expert)>=experts) throw std::runtime_error("invalid Straydog MoE selection");
            for (size_t prior=0;prior<rank;++prior)
                if (selected[token*10+prior]==expert) throw std::runtime_error("duplicate Straydog MoE expert");
            members[size_t(expert)].push_back(int32_t(token*10+rank));
        }
        members[experts].push_back(int32_t(steps*10+token));
    }
    float* gate_up=static_cast<float*>(workspace);
    float* product=gate_up+routes*1280;
    float* outputs=product+routes*640;
    float* probs=outputs+routes*2560;
    int32_t* positions=reinterpret_cast<int32_t*>(probs+steps*11);
    upload(d,d.input,input,steps*2560*4);
    upload(d,probs,probabilities,steps*11*4);
    // All host metadata lives until the final synchronization. Device descriptors
    // are reused in stream order, including the tiny fallback GEMVs.
    std::vector<std::vector<Matrix>> descriptors;
    descriptors.reserve((experts+1)*3);
    auto project=[&](const Matrix& matrix,size_t count,const float* x,float* y) {
        descriptors.emplace_back(count<d.gemm_min_batch ? count : 0);
        hc_project(d,matrix,count,x,y,descriptors.back().data());
    };
    if (steps==1) {
        std::vector<Matrix> initial;
        initial.reserve(22);
        for (size_t route=0;route<11;++route) {
            const size_t source=route<10 ? size_t(selected[route])*3 : experts*3;
            for (size_t role=0;role<2;++role) {
                Matrix matrix=weights[source+role];
                matrix.output_offset=route*1280+role*640;
                initial.push_back(matrix);
            }
        }
        gemv(d,initial.data(),initial.size(),d.input,gate_up);
        moe_product<<<unsigned((routes*640+255)/256),256,0,d.stream>>>(gate_up,routes,product);
        std::vector<Matrix> final(11);
        for (size_t route=0;route<11;++route) {
            final[route]=weights[route<10 ? size_t(selected[route])*3+2 : experts*3+2];
            final[route].input_offset=route*640;
            final[route].output_offset=route*2560;
        }
        gemv(d,final.data(),final.size(),product,outputs);
        moe_combine<<<10,256,0,d.stream>>>(outputs,probs,steps,d.output);
        check(hipGetLastError());
        download(d,output,d.output,2560*4);
        return;
    }
    std::vector<int32_t> flat;
    flat.reserve(routes);
    std::vector<size_t> starts(experts+1);
    for (size_t expert=0;expert<=experts;++expert) {
        starts[expert]=flat.size(); flat.insert(flat.end(),members[expert].begin(),members[expert].end());
    }
    upload(d,positions,flat.data(),routes*4);
    std::vector<MoeGroup> groups(experts+1);
    std::vector<MoeTile> tiles, half_tiles;
    for (size_t expert=0;expert<=experts;++expert) {
        const size_t count=members[expert].size();
        groups[expert]={{weights[expert*3],weights[expert*3+1],weights[expert*3+2]},starts[expert],count,
            int(d.half_gemm && count>=d.gemm_min_batch)};
        if (expert<experts && count<64) {
            auto& destination=d.rdna4 && groups[expert].half ? half_tiles : tiles;
            for (size_t row=0;row<count;row+=16) destination.push_back({uint32_t(expert),uint32_t(row)});
        }
    }
    const size_t scalar_tiles=tiles.size();
    tiles.insert(tiles.end(),half_tiles.begin(),half_tiles.end());
    MoeGroup* group_storage=reinterpret_cast<MoeGroup*>(positions+routes);
    MoeTile* tile_storage=reinterpret_cast<MoeTile*>(group_storage+groups.size());
    if (groups.size()*sizeof(MoeGroup)+tiles.size()*sizeof(MoeTile)>65536*4)
        throw std::runtime_error("Straydog grouped MoE metadata overflow");
    upload(d,group_storage,groups.data(),groups.size()*sizeof(MoeGroup));
    if (!tiles.empty()) {
        upload(d,tile_storage,tiles.data(),tiles.size()*sizeof(MoeTile));
        if (scalar_tiles)
            grouped_moe<<<dim3(10,unsigned(scalar_tiles),2),256,0,d.stream>>>(group_storage,tile_storage,positions,steps,false,d.input,gate_up);
        if (!half_tiles.empty())
            grouped_moe_wmma<16><<<dim3(10,unsigned(half_tiles.size()),2),128,0,d.stream>>>(group_storage,tile_storage+scalar_tiles,positions,steps,false,d.input,gate_up);
    }
    for (size_t expert=0;expert<=experts;++expert) {
        const size_t count=members[expert].size();
        if (!count || (expert<experts && count<64)) continue;
        const int32_t* selected_rows=positions+starts[expert];
        moe_gather<<<unsigned((count*2560+255)/256),256,0,d.stream>>>(d.input,selected_rows,count,2560,steps,false,d.output);
        float* temporary=d.output+count*2560;
        for (size_t role=0;role<2;++role) {
            project(weights[expert*3+role],count,d.output,temporary);
            moe_scatter<<<unsigned((count*640+255)/256),256,0,d.stream>>>(temporary,selected_rows,count,640,1280,role*640,gate_up);
        }
    }
    moe_product<<<unsigned((routes*640+255)/256),256,0,d.stream>>>(gate_up,routes,product);
    if (scalar_tiles)
        grouped_moe<<<dim3(40,unsigned(scalar_tiles),1),256,0,d.stream>>>(group_storage,tile_storage,positions,steps,true,product,outputs);
    if (!half_tiles.empty())
        grouped_moe_wmma<16><<<dim3(40,unsigned(half_tiles.size()),1),128,0,d.stream>>>(group_storage,tile_storage+scalar_tiles,positions,steps,true,product,outputs);
    for (size_t expert=0;expert<=experts;++expert) {
        const size_t count=members[expert].size();
        if (!count || (expert<experts && count<64)) continue;
        const int32_t* selected_rows=positions+starts[expert];
        moe_gather<<<unsigned((count*640+255)/256),256,0,d.stream>>>(product,selected_rows,count,640,steps,true,d.output);
        float* temporary=d.output+count*640;
        project(weights[expert*3+2],count,d.output,temporary);
        moe_scatter<<<unsigned((count*2560+255)/256),256,0,d.stream>>>(temporary,selected_rows,count,2560,2560,0,outputs);
    }
    moe_combine<<<unsigned((steps*2560+255)/256),256,0,d.stream>>>(outputs,probs,steps,d.output);
    check(hipGetLastError());
    download(d,output,d.output,steps*2560*4);
}
