#include "sort_core.hpp"
#include <algorithm>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <omp.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {
struct Timer {
#ifdef _WIN32
    using Tick = long long;
    LARGE_INTEGER frequency{};
    Timer() {
        if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0)
            throw std::runtime_error("high resolution performance counter unavailable");
    }
    Tick now() const { LARGE_INTEGER value; QueryPerformanceCounter(&value); return value.QuadPart; }
    double elapsed(Tick begin, Tick end) const {
        return static_cast<double>(end - begin) / static_cast<double>(frequency.QuadPart);
    }
    static const char* name() { return "QueryPerformanceCounter"; }
#else
    using Tick = double;
    Tick now() const { return omp_get_wtime(); }
    double elapsed(Tick begin, Tick end) const { return end - begin; }
    static const char* name() { return "omp_get_wtime"; }
#endif
};
std::uint64_t number(const std::string& text, const std::string& flag) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size() || text.empty())
        throw std::invalid_argument("invalid unsigned integer for " + flag + ": " + text);
    return value;
}
int integer(const std::string& text, const std::string& flag, bool allow_zero = false) {
    const auto value = number(text, flag);
    if ((!allow_zero && value == 0) || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("out of range: " + flag);
    return static_cast<int>(value);
}
void help() {
    std::cout << "Usage: quicksort THREADS N\n"
        "       quicksort --algorithm NAME --threads P --size N [options]\n"
        "Algorithms: serial, omp-task, omp-partition, std, teaching\n"
        "Options: --distribution random|ascending|descending|nearly-sorted|equal|duplicates\n"
        "         --seed UINT64 --task-cutoff N --partition-cutoff N\n"
        "         --partition-levels N --warmups N --repeats N --json\n"
        "         --diagnostics --build-info --help\n"
        "Every run is checked against std::sort outside the timer.\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { help(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            std::cout << "{\"compiler\":\"" << __VERSION__ << "\",\"openmp\":" << _OPENMP
                << ",\"value_bits\":32,\"block_size\":65536,\"insertion_cutoff\":24,\"timer\":\""
                << Timer::name() << "\"}\n";
            return 0;
        }
        hpc::Options options;
        auto algorithm = hpc::Algorithm::Task;
        std::string distribution = "random";
        std::uint64_t seed = 1024;
        std::size_t size = 8000000;
        int warmups = 0, repeats = 1, start = 1;
        bool legacy = argc >= 3 && std::string(argv[1]).rfind("--", 0) != 0;
        bool json = !legacy, diagnostics = false;
        if (argc == 1) { help(); return 2; }
        auto read_size = [&](const std::string& text, const std::string& flag) {
            auto value = number(text, flag);
            if (value > std::vector<hpc::Value>().max_size() ||
                    value > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
                throw std::invalid_argument("array or cutoff too large: " + flag);
            return static_cast<std::size_t>(value);
        };
        if (legacy) { options.threads = integer(argv[1], "threads"); size = read_size(argv[2], "size"); start = 3; }
        for (int i = start; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--json") { json = true; continue; }
            if (flag == "--diagnostics") { diagnostics = true; continue; }
            if (i + 1 == argc) throw std::invalid_argument("missing value: " + flag);
            const std::string value = argv[++i];
            if (flag == "--algorithm") algorithm = hpc::parse_algorithm(value);
            else if (flag == "--threads") options.threads = integer(value, flag);
            else if (flag == "--size") size = read_size(value, flag);
            else if (flag == "--seed") seed = number(value, flag);
            else if (flag == "--distribution") distribution = value;
            else if (flag == "--task-cutoff") options.task_cutoff = read_size(value, flag);
            else if (flag == "--partition-cutoff") options.partition_cutoff = read_size(value, flag);
            else if (flag == "--partition-levels") options.partition_levels = integer(value, flag, true);
            else if (flag == "--warmups") warmups = integer(value, flag, true);
            else if (flag == "--repeats") repeats = integer(value, flag);
            else throw std::invalid_argument("unknown option: " + flag);
        }
        if (options.task_cutoff < 25 || options.partition_cutoff < 25)
            throw std::invalid_argument("cutoffs must be at least 25");
        if (options.partition_levels > 64) throw std::invalid_argument("partition levels must be <= 64");
        const auto input = hpc::generate_data(size, seed, distribution);
        const auto input_hash = hpc::fingerprint(input);
        auto reference = input;
        std::sort(reference.begin(), reference.end());
        std::vector<hpc::Value> work(size), scratch;
        if (algorithm == hpc::Algorithm::Partition) scratch.resize(size);
        const Timer timer;
        double elapsed = 0;
        for (std::int64_t run = -static_cast<std::int64_t>(warmups); run < repeats; ++run) {
            std::copy(input.begin(), input.end(), work.begin());
            hpc::Diagnostics stats;
            const auto begin = timer.now();
            const int actual = hpc::sort(work.data(), size, algorithm, options, scratch,
                                        diagnostics ? &stats : nullptr);
            elapsed = timer.elapsed(begin, timer.now());
            if (work != reference) throw std::runtime_error("sort verification failed");
            if (run < 0 || !json) continue;
            std::cout << std::setprecision(17)
                << "{\"algorithm\":\"" << hpc::algorithm_name(algorithm)
                << "\",\"distribution\":\"" << distribution << "\",\"size\":" << size
                << ",\"seed\":" << seed << ",\"requested_threads\":" << options.threads
                << ",\"actual_threads\":" << actual << ",\"iteration\":" << run + 1
                << ",\"warmups\":" << warmups << ",\"task_cutoff\":" << options.task_cutoff
                << ",\"partition_cutoff\":" << options.partition_cutoff
                << ",\"partition_levels\":" << options.partition_levels
                << ",\"seconds\":" << elapsed << ",\"elapsed_ms\":" << elapsed * 1000.0
                << ",\"input_hash\":\"" << input_hash << "\",\"verified\":true"
                << ",\"auxiliary_bytes\":" << scratch.size() * sizeof(hpc::Value)
                << ",\"diagnostics_enabled\":" << (diagnostics ? "true" : "false")
                << ",\"tasks\":" << stats.tasks.load()
                << ",\"parallel_partitions\":" << stats.parallel_partitions.load()
                << ",\"heap_fallbacks\":" << stats.heap_fallbacks.load() << "}" << std::endl;
        }
        if (!json) std::cout << std::fixed << std::setprecision(6) << elapsed * 1000.0 << ' ' << size << '\n';
        return 0;
    } catch (const std::bad_alloc&) {
        std::cerr << "ERROR: insufficient memory for input, reference, working array or scratch\n"; return 3;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n'; return 2;
    }
}
