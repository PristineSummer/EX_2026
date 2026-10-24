#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hpc {
using Value = std::int32_t;
enum class Algorithm { Serial, Task, Partition, Standard, Teaching };
struct Options {
    int threads = 1;
    std::size_t task_cutoff = 65536;
    std::size_t partition_cutoff = 1048576;
    int partition_levels = 2;
    int depth_override = -1;  // Test hook: -1 selects the normal depth budget.
};
struct Diagnostics {
    std::atomic<std::uint64_t> tasks{0};
    std::atomic<std::uint64_t> parallel_partitions{0};
    std::atomic<std::uint64_t> heap_fallbacks{0};
};
Algorithm parse_algorithm(const std::string& name);
const char* algorithm_name(Algorithm algorithm);
std::vector<Value> generate_data(std::size_t size, std::uint64_t seed,
                                 const std::string& distribution);
std::uint64_t fingerprint(const std::vector<Value>& data);
// Scratch is allocated BEFORE timing. All descendants finish before return.
// Optional diagnostic counters are disabled in performance measurements.
int sort(Value* data, std::size_t size, Algorithm algorithm, const Options& options,
         std::vector<Value>& scratch, Diagnostics* diagnostics = nullptr);
}  // namespace hpc
