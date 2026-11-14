#pragma once
// The marks of a prompt batch's phases (phase_profile.hpp, STRAYDOG_PREFILL_PHASES).
inline void phase_resolve(Device& d) {
    auto& marks = d.phase_marks;
    for (size_t i = 1; i < marks.size(); ++i) {
        check(hipEventSynchronize(marks[i].event));
        float ms = 0;
        check(hipEventElapsedTime(&ms, marks[i - 1].event, marks[i].event));
        auto& slot = d.phase_totals[marks[i].name];
        slot.sum += double(ms); slot.count += 1; slot.largest = std::max(slot.largest, double(ms));
        if (d.phase_trace && ms > 4.0f) std::fprintf(stderr, "PHASETRACE device %d %s %.2f\n", d.ordinal, marks[i].name, double(ms));
    }
    if (marks.size() > 1) {
        for (size_t i = 0; i + 1 < marks.size(); ++i) d.phase_pool.push_back(marks[i].event);
        marks.erase(marks.begin(), marks.end() - 1);
    }
}
inline void phase(Device& d, const char* name) {
    if (!d.phase_on) return;
    if (d.phase_mtp) name = d.phase_names.insert(std::string("mtp_") + name).first->c_str();
    hipEvent_t event;
    if (d.phase_pool.empty()) check(hipEventCreate(&event));
    else { event = d.phase_pool.back(); d.phase_pool.pop_back(); }
    check(hipEventRecord(event, d.stream));
    d.phase_marks.push_back({name, event});
    if (d.phase_marks.size() >= 2048) phase_resolve(d);
}
inline void phase_break(Device& d) {   // (the next phase is not timed from the last: a wait of the host's that the profile should not count)
    if (!d.phase_on) return;
    phase_resolve(d);
    for (auto& mark : d.phase_marks) d.phase_pool.push_back(mark.event);
    d.phase_marks.clear();
}
