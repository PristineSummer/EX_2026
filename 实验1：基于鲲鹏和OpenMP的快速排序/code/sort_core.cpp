#include "sort_core.hpp"
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <omp.h>

namespace hpc {
namespace {
constexpr std::size_t insertion_cutoff = 24;
constexpr std::size_t block_size = 65536;
std::uint64_t mix(std::uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
Value signed_value(std::uint32_t x) {
    return x <= UINT32_C(0x7fffffff) ? static_cast<Value>(x)
        : -1 - static_cast<Value>(UINT32_MAX - x);
}
int depth_budget(std::size_t n) {
    int depth = 0;
    while (n > 1) { n >>= 1; ++depth; }
    return 2 * depth;
}
void insertion_sort(Value* a, std::size_t n) {
    for (std::size_t i = 1; i < n; ++i) {
        const Value value = a[i];
        std::size_t j = i;
        while (j && a[j - 1] > value) { a[j] = a[j - 1]; --j; }
        a[j] = value;
    }
}
std::size_t median_index(const Value* a, std::size_t i, std::size_t j, std::size_t k) {
    return a[i] < a[j] ? (a[j] < a[k] ? j : (a[i] < a[k] ? k : i))
                       : (a[k] < a[j] ? j : (a[k] < a[i] ? k : i));
}
std::size_t pivot_index(const Value* a, std::size_t n) {
    const auto middle = n / 2;
    if (n <= 128) return median_index(a, 0, middle, n - 1);
    const auto step = n / 8;
    return median_index(a, median_index(a, 0, step, 2 * step),
        median_index(a, middle - step, middle, middle + step),
        median_index(a, n - 1 - 2 * step, n - 1 - step, n - 1));
}
// Bentley--McIlroy: bidirectional scans and equal-key collection at both ends.
// Return [0,left) and [right,n); the middle contains only the pivot value.
std::pair<std::size_t, std::size_t> partition_serial(Value* a, std::size_t n) {
    using Index = std::ptrdiff_t;
    std::swap(a[0], a[pivot_index(a, n)]);
    const Value pivot = a[0];
    const Index high = static_cast<Index>(n) - 1;
    Index i = 0, j = high + 1, p = 0, q = high + 1;
    for (;;) {
        do { ++i; } while (i < high && a[i] < pivot);
        do { --j; } while (j > 0 && a[j] > pivot);
        if (i == j && a[i] == pivot) std::swap(a[++p], a[i]);
        if (i >= j) break;
        std::swap(a[i], a[j]);
        if (a[i] == pivot) std::swap(a[++p], a[i]);
        if (a[j] == pivot) std::swap(a[--q], a[j]);
    }
    i = j + 1;
    for (Index k = 0; k <= p; ++k) std::swap(a[k], a[j--]);
    for (Index k = high; k >= q; --k) std::swap(a[k], a[i++]);
    return {static_cast<std::size_t>(j + 1), static_cast<std::size_t>(i)};
}
void serial_sort(Value* a, std::size_t n, int depth, Diagnostics* stats) {
    while (n > insertion_cutoff) {
        if (depth == 0) {
            if (stats) ++stats->heap_fallbacks;
            std::make_heap(a, a + n);
            std::sort_heap(a, a + n);
            return;
        }
        --depth;
        const auto bounds = partition_serial(a, n);
        const auto left = bounds.first, right = n - bounds.second;
        // Recurse only on the smaller side for bounded native stack usage.
        if (left < right) {
            serial_sort(a, left, depth, stats);
            a += bounds.second; n = right;
        } else {
            serial_sort(a + bounds.second, right, depth, stats);
            n = left;
        }
    }
    insertion_sort(a, n);
}
struct Context {
    Options options;
    Diagnostics* stats;
    int team_size = 1;
    int task_depth_limit = 1;
    std::atomic<bool> allocation_failed{false};
};
struct alignas(64) Counts { std::size_t less = 0, equal = 0, greater = 0; };
std::pair<std::size_t, std::size_t> partition_parallel(
        Value* a, Value* scratch, std::size_t n, Context* ctx) {
    const Value pivot = a[pivot_index(a, n)];
    const std::size_t blocks = (n - 1) / block_size + 1;
    std::vector<Counts> counts, offsets;
    // Do not let allocation exceptions escape an OpenMP structured block.
    try { counts.resize(blocks); offsets.resize(blocks); }
    catch (const std::bad_alloc&) {
        ctx->allocation_failed.store(true); return {0, n};
    }
    auto* count_data = counts.data();
    auto* offset_data = offsets.data();
    if (ctx->stats) ++ctx->stats->parallel_partitions;
    // The taskloop implicit taskgroups keep this function's metadata alive.
#pragma omp taskloop default(none) grainsize(1) firstprivate(a, n, blocks, pivot, count_data)
    for (std::size_t b = 0; b < blocks; ++b) {
        Counts c;
        const auto end = std::min(n, (b + 1) * block_size);
        for (auto k = b * block_size; k < end; ++k) {
            if (a[k] < pivot) ++c.less;
            else if (a[k] == pivot) ++c.equal;
            else ++c.greater;
        }
        count_data[b] = c;
    }
    std::size_t less = 0, equal = 0;
    for (const auto& c : counts) { less += c.less; equal += c.equal; }
    std::size_t l = 0, e = less, g = less + equal;
    for (std::size_t b = 0; b < blocks; ++b) {
        offsets[b].less = l; offsets[b].equal = e; offsets[b].greater = g;
        l += counts[b].less; e += counts[b].equal; g += counts[b].greater;
    }
#pragma omp taskloop default(none) grainsize(1) firstprivate(a, scratch, n, blocks, pivot, offset_data)
    for (std::size_t b = 0; b < blocks; ++b) {
        auto c = offset_data[b];
        const auto end = std::min(n, (b + 1) * block_size);
        for (auto k = b * block_size; k < end; ++k) {
            const auto value = a[k];
            if (value < pivot) scratch[c.less++] = value;
            else if (value == pivot) scratch[c.equal++] = value;
            else scratch[c.greater++] = value;
        }
    }
#pragma omp taskloop default(none) grainsize(1) firstprivate(a, scratch, n, blocks)
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto start = b * block_size;
        const auto end = std::min(n, (b + 1) * block_size);
        std::copy(scratch + start, scratch + end, a + start);
    }
    return {less, less + equal};
}
void task_sort(Value* a, Value* scratch, std::size_t n, int depth,
               int task_depth, int partition_levels, Context* ctx) {
    while (n > 1) {
        if (ctx->allocation_failed.load()) return;
        if (n <= ctx->options.task_cutoff || depth == 0 ||
                task_depth >= ctx->task_depth_limit || ctx->team_size == 1) {
            serial_sort(a, n, depth, ctx->stats); return;
        }
        const bool parallel = partition_levels > 0 && n >= ctx->options.partition_cutoff;
        const auto bounds = parallel ? partition_parallel(a, scratch, n, ctx)
                                     : partition_serial(a, n);
        if (ctx->allocation_failed.load()) return;
        --depth; ++task_depth;
        partition_levels = std::max(0, partition_levels - 1);
        const auto left = bounds.first, right = n - bounds.second;
        Value* large_a; Value* large_scratch;
        std::size_t large_n;
        if (left < right) {
            large_a = a + bounds.second;
            large_scratch = scratch ? scratch + bounds.second : nullptr;
            large_n = right; n = left;
        } else {
            large_a = a; large_scratch = scratch; large_n = left;
            a += bounds.second;
            if (scratch) scratch += bounds.second;
            n = right;
        }
        if (large_n > ctx->options.task_cutoff) {
            if (ctx->stats) ++ctx->stats->tasks;
#pragma omp task default(none) firstprivate(large_a, large_scratch, large_n, depth, task_depth, partition_levels, ctx)
            { task_sort(large_a, large_scratch, large_n, depth, task_depth, partition_levels, ctx); }
        } else {
            serial_sort(large_a, large_n, depth, ctx->stats);
        }
    }
}
void teaching_sort(Value* a, std::size_t n, Context* ctx) {
    if (n <= 1) return;
    const auto pivot = a[n / 2];
    std::ptrdiff_t i = 0, j = static_cast<std::ptrdiff_t>(n) - 1;
    while (i <= j) {
        while (i <= j && a[i] < pivot) ++i;
        while (i <= j && a[j] > pivot) --j;
        if (i <= j) { std::swap(a[i], a[j]); ++i; --j; }
    }
    const auto left = static_cast<std::size_t>(j + 1);
    const auto right = n - static_cast<std::size_t>(i);
    auto* right_a = a + i;
    if (ctx->stats) ctx->stats->tasks += 2;
#pragma omp task default(none) firstprivate(a, left, ctx)
    { teaching_sort(a, left, ctx); }
#pragma omp task default(none) firstprivate(right_a, right, ctx)
    { teaching_sort(right_a, right, ctx); }
}
}  // namespace
Algorithm parse_algorithm(const std::string& name) {
    if (name == "serial") return Algorithm::Serial;
    if (name == "omp-task") return Algorithm::Task;
    if (name == "omp-partition") return Algorithm::Partition;
    if (name == "std") return Algorithm::Standard;
    if (name == "teaching") return Algorithm::Teaching;
    throw std::invalid_argument("unknown algorithm: " + name);
}
const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Serial: return "serial";
        case Algorithm::Task: return "omp-task";
        case Algorithm::Partition: return "omp-partition";
        case Algorithm::Standard: return "std";
        case Algorithm::Teaching: return "teaching";
    }
    return "unknown";
}
std::vector<Value> generate_data(std::size_t n, std::uint64_t seed, const std::string& distribution) {
    if (distribution != "random" && distribution != "ascending" && distribution != "descending"
            && distribution != "nearly-sorted" && distribution != "equal" && distribution != "duplicates")
        throw std::invalid_argument("unknown distribution: " + distribution);
    std::vector<Value> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto word = static_cast<std::uint32_t>(mix(seed + i));
        data[i] = distribution == "equal" ? 42 : distribution == "duplicates"
            ? static_cast<Value>(word % 16) - 8 : signed_value(word);
    }
    if (distribution == "ascending" || distribution == "descending" || distribution == "nearly-sorted")
        std::sort(data.begin(), data.end());
    if (distribution == "descending") std::reverse(data.begin(), data.end());
    if (distribution == "nearly-sorted" && n > 1) {
        for (std::size_t i = 0; i < std::max<std::size_t>(1, n / 100); ++i)
            std::swap(data[mix(seed + 2 * i) % n], data[mix(seed + 2 * i + 1) % n]);
    }
    return data;
}
std::uint64_t fingerprint(const std::vector<Value>& data) {
    std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (const auto v : data) { hash ^= static_cast<std::uint32_t>(v); hash *= UINT64_C(0x100000001b3); }
    return hash;
}
int sort(Value* data, std::size_t n, Algorithm algorithm, const Options& options,
         std::vector<Value>& scratch, Diagnostics* stats) {
    if (options.threads < 1 || options.task_cutoff < insertion_cutoff + 1 ||
            options.partition_cutoff < insertion_cutoff + 1 || options.partition_levels < 0 ||
            options.depth_override < -1) throw std::invalid_argument("invalid sorting options");
    if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("array exceeds index range");
    if (algorithm == Algorithm::Partition && scratch.size() < n)
        throw std::invalid_argument("parallel partition requires preallocated scratch");
    const int depth = options.depth_override < 0 ? depth_budget(n) : options.depth_override;
    if (algorithm == Algorithm::Serial) { serial_sort(data, n, depth, stats); return 1; }
    if (algorithm == Algorithm::Standard) { if (n > 1) std::sort(data, data + n); return 1; }
    Context context;
    context.options = options; context.stats = stats;
    auto* ctx = &context;
    auto* buffer = algorithm == Algorithm::Partition ? scratch.data() : nullptr;
    const int requested_threads = options.threads;
    const int levels = algorithm == Algorithm::Partition ? options.partition_levels : 0;
    omp_set_dynamic(0);
#pragma omp parallel default(none) num_threads(requested_threads) shared(context) firstprivate(data, buffer, n, depth, levels, algorithm, ctx)
    {
#pragma omp single nowait
        {
            context.team_size = omp_get_num_threads();
            std::size_t target = static_cast<std::size_t>(context.team_size) * 8;
            while (target > 1) { target = (target + 1) / 2; ++context.task_depth_limit; }
#pragma omp taskgroup
            {
                if (algorithm == Algorithm::Teaching) teaching_sort(data, n, ctx);
                else task_sort(data, buffer, n, depth, 0, levels, ctx);
            }
        }
    }
    if (context.allocation_failed.load()) throw std::bad_alloc();
    return context.team_size;
}
}  // namespace hpc
