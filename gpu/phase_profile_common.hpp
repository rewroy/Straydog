#pragma once
// The marks of a prompt batch's phases (phase_profile.hpp, STRAYDOG_PREFILL_PHASES).
// The marks the stream has passed are summed and dropped but for the last of them, which the next is timed from. Only `wait` (the
// report, when the device goes) waits for the stream: a resolve in the middle of a pass must not, because the stream may itself be
// waiting for the other device's part of a layer that this thread has yet to queue there (two R9700 holding Qwen3.8 whole: the first
// prompt of 6,055 tokens ended the server after ten seconds with "a Straydog GPU waited too long for its peer's part").
inline void phase_resolve(Device& d, bool wait = false) {
    auto& marks = d.phase_marks;
    size_t done = 1;
    for (; done < marks.size(); ++done) {
        if (wait) check(hipEventSynchronize(marks[done].event));
        else {
            const hipError_t status = hipEventQuery(marks[done].event);
            if (status == hipErrorNotReady) break;
            check(status);
        }
        if (marks[done].start) continue;
        float ms = 0;
        check(hipEventElapsedTime(&ms, marks[done - 1].event, marks[done].event));
        auto& slot = d.phase_totals[marks[done].name];
        slot.sum += double(ms); slot.count += 1; slot.largest = std::max(slot.largest, double(ms));
        if (d.phase_trace && ms > 4.0f) std::fprintf(stderr, "PHASETRACE device %d %s %.2f\n", d.ordinal, marks[done].name, double(ms));
    }
    if (done > 1) {
        for (size_t i = 0; i + 1 < done; ++i) d.phase_pool.push_back(marks[i].event);
        marks.erase(marks.begin(), marks.begin() + (done - 1));
    }
}
inline void phase(Device& d, const char* name) {
    if (!d.phase_on) return;
    if (d.phase_mtp) name = d.phase_names.insert(std::string("mtp_") + name).first->c_str();
    hipEvent_t event;
    if (d.phase_pool.empty()) check(hipEventCreate(&event));
    else { event = d.phase_pool.back(); d.phase_pool.pop_back(); }
    check(hipEventRecord(event, d.stream));
    d.phase_marks.push_back({name, event, d.phase_broken});
    d.phase_broken = false;
    if (d.phase_marks.size() >= 2048 && d.phase_marks.size() % 2048 == 0) phase_resolve(d);
}
inline void phase_break(Device& d) {   // (the next phase is not timed from the last: a wait of the host's that the profile should not count)
    if (!d.phase_on) return;
    phase_resolve(d);
    d.phase_broken = true;
}
