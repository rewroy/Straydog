#include "resident_soa_common.hpp"
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
