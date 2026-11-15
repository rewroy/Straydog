// STRAYDOG_PREFILL_PHASES (development): an event after each phase of a prompt batch's layers (on the device's stream), the time between two
// events added to the second's name when the list is resolved, the sums printed when the device is destroyed. A phase's time is the stream's
// time from the previous event: a host wait shows as the phase queued after it.
#pragma once
#include "phase_profile_common.hpp"
void phase_report(Device& d) {
    try {
        cleanup(hipSetDevice(d.ordinal));
        phase_resolve(d, true);
        std::vector<std::pair<std::string, Device::PhaseSum>> rows(d.phase_totals.begin(), d.phase_totals.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.sum > b.second.sum; });
        double total = 0;
        for (const auto& row : rows) total += row.second.sum;
        const double batches = double(std::max<uint64_t>(d.phase_batches, 1));
        std::fprintf(stderr, "PHASES device %d: %llu batches, %llu layers; stream time %.1f ms = %.1f ms a batch\n", d.ordinal,
                     (unsigned long long)d.phase_batches, (unsigned long long)d.phase_layers, total, total / batches);
        for (const auto& row : rows)
            std::fprintf(stderr, "PHASES   %-22s %9.1f ms %5.1f%%  %8.2f ms a batch  %6llu marks  %7.3f ms a mark, largest %8.3f\n", row.first.c_str(), row.second.sum,
                         100.0 * row.second.sum / std::max(total, 1e-9), row.second.sum / batches, (unsigned long long)row.second.count,
                         row.second.sum / double(std::max<uint64_t>(row.second.count, 1)), row.second.largest);
        std::fprintf(stderr, "PHASES   routes a layer (of %zu): resident %.0f, prestaged %.0f, streamed %.0f, CPU %.0f\n", size_t(40960),
                     double(d.phase_routes[0]) / double(std::max<uint64_t>(d.phase_layers, 1)), double(d.phase_routes[1]) / double(std::max<uint64_t>(d.phase_layers, 1)),
                     double(d.phase_routes[2]) / double(std::max<uint64_t>(d.phase_layers, 1)), double(d.phase_routes[3]) / double(std::max<uint64_t>(d.phase_layers, 1)));
        std::fprintf(stderr, "PHASES   layers with experts missing %llu; streamed experts %llu (%.1f a layer), prestage issued %llu used %llu, bus bytes %.2f GB (%.2f GB a batch), CPU routes %llu\n",
                     (unsigned long long)d.phase_missing_layers, (unsigned long long)d.phase_streamed,
                     double(d.phase_streamed) / double(std::max<uint64_t>(d.phase_layers, 1)),
                     (unsigned long long)d.phase_prestage_issued, (unsigned long long)d.phase_prestage_used, double(d.phase_bus_bytes) / 1e9,
                     double(d.phase_bus_bytes) / 1e9 / batches, (unsigned long long)d.phase_cpu_routes);
    } catch (...) {}
}
