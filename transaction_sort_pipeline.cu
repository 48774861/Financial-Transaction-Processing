// transaction_sort_pipeline.cu
//
// Full GPU pipeline:
//   1. Read a CSV of transactions.
//   2. Split into batches of BATCH_SIZE rows.
//   3. GPU radix sort (CUB) each batch by (date, client_id).
//   4. GPU parallel merge sort all sorted batches into one final output.
//
// Data stays on the GPU between the sort and merge stages — no redundant
// round-trips.  Timing is split into four categories:
//   - Transfer overhead (CPU -> GPU and GPU -> CPU)
//   - Radix sort (pure GPU kernel time)
//   - Merge      (pure GPU kernel time)
//   - Total wall time
//
// Build:
//   nvcc -O2 -std=c++17 transaction_sort_pipeline.cu -o transaction_sort_pipeline
//
// Usage:
//   ./transaction_sort_pipeline <input.csv> <output.csv>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Constants
// =============================================================================

static constexpr size_t BATCH_SIZE        = 1'000'000;
static constexpr int    THREADS_PER_BLOCK = 256;

// =============================================================================
// Timing helpers
// =============================================================================

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline TimePoint now() { return Clock::now(); }

static inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// Error-checking macros
// =============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUB_CHECK(call)                                                         \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUB error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// =============================================================================
// Date / key utilities  (host-side)
// =============================================================================

static inline int64_t days_from_civil(int y, unsigned m, unsigned d) {
    y -= (m <= 2);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}

static inline uint32_t parse_datetime_to_epoch_seconds(const std::string& s) {
    if (s.size() < 19)
        throw std::runtime_error("Bad date field: " + s);

    int year   = std::stoi(s.substr(0,  4));
    int month  = std::stoi(s.substr(5,  2));
    int day    = std::stoi(s.substr(8,  2));
    int hour   = std::stoi(s.substr(11, 2));
    int minute = std::stoi(s.substr(14, 2));
    int second = std::stoi(s.substr(17, 2));

    int64_t days = days_from_civil(year, static_cast<unsigned>(month),
                                         static_cast<unsigned>(day));
    int64_t secs = days * 86400LL + hour * 3600LL + minute * 60LL + second;

    if (secs < 0 || secs > 0xFFFFFFFFULL)
        throw std::runtime_error("Date out of uint32 range: " + s);

    return static_cast<uint32_t>(secs);
}

// =============================================================================
// CSV helpers  (host-side)
// =============================================================================

struct ParsedRow {
    std::string raw_line;  // original CSV row (preserved for output)
    uint64_t    key;       // packed sort key: (date_seconds << 32) | client_id
};

static inline std::vector<std::string> split_csv_simple(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ','))
        fields.push_back(item);
    return fields;
}

static inline uint64_t make_sort_key(const std::string& date_str,
                                     const std::string& client_id_str) {
    uint32_t date_seconds = parse_datetime_to_epoch_seconds(date_str);
    uint32_t client_id    = static_cast<uint32_t>(std::stoul(client_id_str));
    // Primary sort: date  |  Secondary sort: client_id
    return (static_cast<uint64_t>(date_seconds) << 32) | client_id;
}

static inline bool parse_transaction_line(const std::string& line, ParsedRow& out) {
    // Expected columns (0-indexed):
    //   0:id  1:date  2:client_id  3:card_id  4:amount  5:use_chip
    //   6:merchant_id  7:merchant_city  8:merchant_state  9:zip  10:mcc  11:errors
    auto fields = split_csv_simple(line);
    if (fields.size() < 3) return false;
    out.raw_line = line;
    out.key      = make_sort_key(fields[1], fields[2]);
    return true;
}

// =============================================================================
// GPU batch descriptor
// =============================================================================
//
// Owns the device-side sorted keys and value (row-index) arrays for one batch.
// Data is allocated on first upload and stays on the GPU until the merge stage
// consumes it — no round-trip back to the host between sort and merge.

struct GpuBatch {
    uint64_t* d_keys = nullptr;  // sorted keys  (device)
    uint32_t* d_vals = nullptr;  // global row indices (device)
    int       size   = 0;

    GpuBatch() = default;
    GpuBatch(const GpuBatch&) = delete;
    GpuBatch& operator=(const GpuBatch&) = delete;
    GpuBatch(GpuBatch&& o) noexcept
        : d_keys(o.d_keys), d_vals(o.d_vals), size(o.size)
    { o.d_keys = nullptr; o.d_vals = nullptr; o.size = 0; }
    GpuBatch& operator=(GpuBatch&& o) noexcept {
        if (this != &o) {
            free();
            d_keys = o.d_keys; d_vals = o.d_vals; size = o.size;
            o.d_keys = nullptr; o.d_vals = nullptr; o.size = 0;
        }
        return *this;
    }

    void free() {
        if (d_keys) { cudaFree(d_keys); d_keys = nullptr; }
        if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }
        size = 0;
    }
    ~GpuBatch() { free(); }
};

// =============================================================================
// Stage 1 — GPU radix sort for a single batch
// =============================================================================
//
// Transfer (CPU->GPU) and sort kernel are timed independently.
// The returned GpuBatch keeps its buffers on the GPU for the merge stage.

static GpuBatch radix_sort_batch_gpu(
    const std::vector<uint64_t>& h_keys,
    const std::vector<uint32_t>& h_indices,   // global row indices (identity + offset)
    double& out_transfer_ms,
    double& out_sort_ms)
{
    const size_t n = h_keys.size();

    uint64_t *d_keys_in = nullptr, *d_keys_out = nullptr;
    uint32_t *d_vals_in = nullptr, *d_vals_out = nullptr;
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;

    CUDA_CHECK(cudaMalloc(&d_keys_in,  n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_in,  n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_out, n * sizeof(uint32_t)));

    // ---- Timed: CPU -> GPU transfer ----------------------------------------
    auto t0 = now();
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys.data(),    n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals_in, h_indices.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = now();
    out_transfer_ms = elapsed_ms(t0, t1);

    // Query CUB scratch size (not timed — just a sizing call)
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp, temp_sz,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(n)));

    CUDA_CHECK(cudaMalloc(&d_temp, temp_sz));

    // ---- Timed: pure radix sort kernel -------------------------------------
    auto t2 = now();
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp, temp_sz,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(n)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t3 = now();
    out_sort_ms = elapsed_ms(t2, t3);

    cudaFree(d_temp);
    cudaFree(d_keys_in);
    cudaFree(d_vals_in);

    GpuBatch batch;
    batch.d_keys = d_keys_out;
    batch.d_vals = d_vals_out;
    batch.size   = static_cast<int>(n);
    return batch;
}

// =============================================================================
// Stage 2 — GPU parallel merge (merge-path algorithm)
// =============================================================================

__device__ static int binarySearch(const uint64_t* A, int sizeA,
                                   const uint64_t* B, int sizeB,
                                   int diag)
{
    int low  = max(0,    diag - sizeB);
    int high = min(diag, sizeA);

    while (low < high) {
        int mid = (low + high) >> 1;

        uint64_t a_left  = (mid > 0)            ? A[mid - 1]      : 0ULL;
        uint64_t b_right = (diag - mid < sizeB) ? B[diag - mid]   : ULLONG_MAX;

        if (a_left > b_right) {
            high = mid;
        } else {
            uint64_t b_left  = (diag - mid > 0) ? B[diag - mid - 1] : 0ULL;
            uint64_t a_right = (mid < sizeA)     ? A[mid]            : ULLONG_MAX;

            if (b_left > a_right)
                low = mid + 1;
            else
                return mid;
        }
    }
    return low;
}

__global__ static void parallelMergeKernel(
    const uint64_t* __restrict__ A_keys, const uint32_t* __restrict__ A_vals, int sizeA,
    const uint64_t* __restrict__ B_keys, const uint32_t* __restrict__ B_vals, int sizeB,
          uint64_t* __restrict__ C_keys,       uint32_t* __restrict__ C_vals)
{
    int tid        = blockIdx.x * blockDim.x + threadIdx.x;
    int total      = sizeA + sizeB;
    int numThreads = gridDim.x * blockDim.x;

    int elemsPerThread = (total + numThreads - 1) / numThreads;
    int diagStart      = tid * elemsPerThread;
    int diagEnd        = min(diagStart + elemsPerThread, total);

    if (diagStart >= total) return;

    int aStart = binarySearch(A_keys, sizeA, B_keys, sizeB, diagStart);
    int bStart = diagStart - aStart;
    int aEnd   = binarySearch(A_keys, sizeA, B_keys, sizeB, diagEnd);
    int bEnd   = diagEnd - aEnd;

    int i = aStart, j = bStart, k = diagStart;

    while (i < aEnd && j < bEnd) {
        if (A_keys[i] <= B_keys[j]) { C_keys[k] = A_keys[i]; C_vals[k] = A_vals[i]; ++i; }
        else                         { C_keys[k] = B_keys[j]; C_vals[k] = B_vals[j]; ++j; }
        ++k;
    }
    while (i < aEnd) { C_keys[k] = A_keys[i]; C_vals[k] = A_vals[i]; ++i; ++k; }
    while (j < bEnd) { C_keys[k] = B_keys[j]; C_vals[k] = B_vals[j]; ++j; ++k; }
}

// Merges two GpuBatches entirely on the GPU. Frees A and B, returns merged result.
// out_merge_ms receives the pure kernel time only.

static GpuBatch parallelMerge(GpuBatch& A, GpuBatch& B, double& out_merge_ms) {
    int total = A.size + B.size;

    uint64_t* d_C_keys = nullptr;
    uint32_t* d_C_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C_keys, total * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_C_vals, total * sizeof(uint32_t)));

    int numBlocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // ---- Timed: pure merge kernel ------------------------------------------
    auto t0 = now();
    parallelMergeKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        A.d_keys, A.d_vals, A.size,
        B.d_keys, B.d_vals, B.size,
        d_C_keys, d_C_vals);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = now();
    out_merge_ms = elapsed_ms(t0, t1);

    A.free();
    B.free();

    GpuBatch C;
    C.d_keys = d_C_keys;
    C.d_vals = d_C_vals;
    C.size   = total;
    return C;
}

// =============================================================================
// Full pipeline
// =============================================================================

static void run_pipeline(const std::string& input_csv,
                         const std::string& output_csv)
{
    std::ifstream fin(input_csv);
    if (!fin.is_open())
        throw std::runtime_error("Could not open input file: " + input_csv);

    std::string header;
    if (!std::getline(fin, header))
        throw std::runtime_error("Input file is empty.");

    std::vector<std::string> all_rows;
    std::vector<GpuBatch>    batches;

    size_t global_offset     = 0;
    double total_transfer_ms = 0.0;   // all CPU<->GPU transfers
    double total_sort_ms     = 0.0;   // pure radix sort kernel time
    double total_merge_ms    = 0.0;   // pure merge kernel time

    auto t_pipeline_start = now();

    // ------------------------------------------------------------------
    // 1. Read CSV -> split into batches -> GPU radix sort each
    // ------------------------------------------------------------------
    while (true) {
        std::vector<ParsedRow> rows;
        rows.reserve(BATCH_SIZE);

        std::string line;
        while (rows.size() < BATCH_SIZE && std::getline(fin, line)) {
            if (line.empty()) continue;
            ParsedRow row;
            if (parse_transaction_line(line, row))
                rows.push_back(std::move(row));
        }
        if (rows.empty()) break;

        const size_t n = rows.size();

        // Build keys and global index permutation on host
        std::vector<uint64_t> batch_keys(n);
        std::vector<uint32_t> batch_indices(n);
        for (size_t i = 0; i < n; ++i) {
            batch_keys[i]    = rows[i].key;
            batch_indices[i] = static_cast<uint32_t>(global_offset + i);
        }

        double transfer_ms = 0.0, sort_ms = 0.0;
        GpuBatch batch = radix_sort_batch_gpu(batch_keys, batch_indices,
                                              transfer_ms, sort_ms);
        total_transfer_ms += transfer_ms;
        total_sort_ms     += sort_ms;

        for (auto& r : rows)
            all_rows.push_back(std::move(r.raw_line));

        std::cout << "Radix-sorted batch " << batches.size()
                  << "  (" << n << " rows)"
                  << "  |  transfer: " << std::fixed << std::setprecision(2)
                  << transfer_ms << " ms"
                  << "  |  sort: " << sort_ms << " ms\n";

        batches.push_back(std::move(batch));
        global_offset += n;

        if (fin.eof()) break;
    }

    if (batches.empty()) {
        std::cout << "No rows found.\n";
        return;
    }

    // ------------------------------------------------------------------
    // 2. Merge all sorted batches on GPU (data never leaves the GPU)
    // ------------------------------------------------------------------
    GpuBatch current = std::move(batches[0]);

    for (size_t b = 1; b < batches.size(); ++b) {
        double merge_ms = 0.0;
        current = parallelMerge(current, batches[b], merge_ms);
        total_merge_ms += merge_ms;

        std::cout << "Merged batch " << b
                  << "  (running total: " << current.size << " rows)"
                  << "  |  merge: " << std::fixed << std::setprecision(2)
                  << merge_ms << " ms\n";
    }

    // ------------------------------------------------------------------
    // 3. GPU -> CPU final download (timed as transfer overhead) + write CSV
    // ------------------------------------------------------------------
    std::vector<uint32_t> final_order(static_cast<size_t>(current.size));

    auto t_dl0 = now();
    CUDA_CHECK(cudaMemcpy(final_order.data(), current.d_vals,
                          current.size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_dl1 = now();
    double download_ms = elapsed_ms(t_dl0, t_dl1);
    total_transfer_ms += download_ms;

    int final_size = current.size;
    current.free();

    std::ofstream fout(output_csv);
    if (!fout.is_open())
        throw std::runtime_error("Could not create output file: " + output_csv);

    fout << header << "\n";
    for (uint32_t idx : final_order)
        fout << all_rows[idx] << "\n";

    auto t_pipeline_end = now();
    double total_ms = elapsed_ms(t_pipeline_start, t_pipeline_end);

    // ------------------------------------------------------------------
    // 4. Timing summary
    // ------------------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "  GPU Pipeline Timing Summary\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Transfer overhead  : " << total_transfer_ms << " ms"
              << "  (CPU->GPU uploads + GPU->CPU download)\n";
    std::cout << "  Batch sort (total) : " << total_sort_ms     << " ms"
              << "  (pure radix sort kernel time)\n";
    std::cout << "  Merge (total)      : " << total_merge_ms    << " ms"
              << "  (pure merge kernel time)\n";
    std::cout << "  Total wall time    : " << total_ms           << " ms\n";
    std::cout << "  Rows written       : " << final_size         << "\n";
    std::cout << "  Output file        : " << output_csv         << "\n";
    std::cout << "========================================\n";
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <input.csv> <output.csv>\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " transactions_data.csv sorted_transactions_gpu.csv\n";
        return EXIT_FAILURE;
    }

    try {
        run_pipeline(argv[1], argv[2]);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
