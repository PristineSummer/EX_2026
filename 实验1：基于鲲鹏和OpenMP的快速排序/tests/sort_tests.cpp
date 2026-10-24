#include "sort_core.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
std::size_t checked = 0;
void check(const std::vector<hpc::Value>& input, hpc::Algorithm algorithm,
           hpc::Options options, hpc::Diagnostics* stats = nullptr) {
    auto actual = input, expected = input;
    std::sort(expected.begin(), expected.end());
    std::vector<hpc::Value> scratch(algorithm == hpc::Algorithm::Partition ? input.size() : 0);
    const int team = hpc::sort(actual.data(), actual.size(), algorithm, options, scratch, stats);
    if (actual != expected) throw std::runtime_error(std::string("mismatch: ") + hpc::algorithm_name(algorithm)
        + " size=" + std::to_string(input.size()) + " threads=" + std::to_string(options.threads));
    if (team != (algorithm == hpc::Algorithm::Serial || algorithm == hpc::Algorithm::Standard ? 1 : options.threads))
        throw std::runtime_error("unexpected actual thread count; inspect OMP_THREAD_LIMIT");
    ++checked;
}
}
int main() {
    try {
        const hpc::Algorithm algorithms[] = {hpc::Algorithm::Serial, hpc::Algorithm::Task,
            hpc::Algorithm::Partition, hpc::Algorithm::Standard, hpc::Algorithm::Teaching};
        const std::size_t sizes[] = {0, 1, 2, 3, 23, 24, 25, 127, 128, 129, 513, 2049};
        for (auto algorithm : algorithms) for (const char* distribution :
                {"random", "ascending", "descending", "nearly-sorted", "equal", "duplicates"})
            for (auto n : sizes) for (auto seed : {UINT64_C(1024), UINT64_C(2026)})
                for (int threads : {1, 2, 4}) {
                    hpc::Options options;
                    options.threads = threads; options.task_cutoff = 32; options.partition_cutoff = 64;
                    check(hpc::generate_data(n, seed, distribution), algorithm, options);
                }
        const auto low = std::numeric_limits<hpc::Value>::min(), high = std::numeric_limits<hpc::Value>::max();
        std::vector<hpc::Value> extremes;
        for (int i = 0; i < 1001; ++i) extremes.push_back(i % 3 == 0 ? low : i % 3 == 1 ? high : 0);
        for (auto algorithm : algorithms) { hpc::Options options; options.threads = 4; options.task_cutoff = 32;
            options.partition_cutoff = 64; check(extremes, algorithm, options); }
        for (std::uint64_t seed = 0; seed < 120; ++seed) {
            hpc::Options options; options.threads = 4; options.task_cutoff = 32; options.partition_cutoff = 64;
            auto data = hpc::generate_data(25 + (seed * 73) % 3000, seed, seed % 2 ? "duplicates" : "random");
            check(data, hpc::Algorithm::Serial, options);
            check(data, hpc::Algorithm::Task, options);
            check(data, hpc::Algorithm::Partition, options);
        }
        hpc::Options forced; forced.threads = 4; forced.depth_override = 0;
        for (auto algorithm : {hpc::Algorithm::Serial, hpc::Algorithm::Task, hpc::Algorithm::Partition}) {
            hpc::Diagnostics stats; check(hpc::generate_data(4097, 17, "random"), algorithm, forced, &stats);
            if (stats.heap_fallbacks.load() == 0) throw std::runtime_error("heap fallback was not exercised");
        }
        forced.depth_override = -1; forced.task_cutoff = 32; forced.partition_cutoff = 64;
        for (const char* distribution : {"random", "equal", "duplicates", "ascending"}) {
            hpc::Diagnostics stats; check(hpc::generate_data(200003, 77, distribution), hpc::Algorithm::Partition, forced, &stats);
            if (stats.parallel_partitions.load() == 0) throw std::runtime_error("parallel partition was not exercised");
            if (std::string(distribution) == "random" && stats.tasks.load() == 0)
                throw std::runtime_error("descendant task completion was not exercised");
        }
        if (hpc::generate_data(2000, 1024, "random") != hpc::generate_data(2000, 1024, "random"))
            throw std::runtime_error("generator is not deterministic");
        std::cout << "PASS: " << checked << " sorting cases; forced heap and multi-block partition paths verified\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
