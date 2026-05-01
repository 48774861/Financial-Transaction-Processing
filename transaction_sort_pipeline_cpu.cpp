// transaction_sort_pipeline_cpu.cpp
//
// CPU-only pipeline — direct comparison to the GPU version:
//   1. Read a CSV of transactions.
//   2. Split into batches of BATCH_SIZE rows.
//   3. std::sort each batch by (date, client_id).
//   4. std::merge all sorted batches into one final output.
//
// Build (example):
//   g++ -O2 -std=c++17 transaction_sort_pipeline_cpu.cpp -o transaction_sort_pipeline_cpu
//
// Usage:
//   ./transaction_sort_pipeline_cpu <input.csv> <output.csv>

#include <algorithm>
#include <chrono>
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

static constexpr size_t BATCH_SIZE = 1'000'000;

// =============================================================================
// Timing helper
// =============================================================================

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline TimePoint now() { return Clock::now(); }

static inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// Date / key utilities
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
// CSV helpers
// =============================================================================

struct ParsedRow {
    std::string raw_line;
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
// Stage 1 — CPU sort for a single batch
// =============================================================================
//
// Sorts an index array by key using std::sort — equivalent to the GPU radix sort.

static std::vector<uint32_t> sort_batch_cpu(const std::vector<uint64_t>& keys) {
    const size_t n = keys.size();
    std::vector<uint32_t> indices(n);
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(),
              [&keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

    return indices;
}

// =============================================================================
// Stage 2 — CPU merge
// =============================================================================
//
// Merges two sorted key+val spans into output vectors using std::merge.

struct SortedBatch {
    std::vector<uint64_t> keys;
    std::vector<uint32_t> vals;   // global row indices into all_rows
};

static SortedBatch merge_batches_cpu(const SortedBatch& A, const SortedBatch& B) {
    SortedBatch C;
    C.keys.resize(A.keys.size() + B.keys.size());
    C.vals.resize(A.vals.size() + B.vals.size());

    // Merge by key, carrying the corresponding val along
    size_t i = 0, j = 0, k = 0;
    while (i < A.keys.size() && j < B.keys.size()) {
        if (A.keys[i] <= B.keys[j]) {
            C.keys[k] = A.keys[i];
            C.vals[k] = A.vals[i];
            ++i;
        } else {
            C.keys[k] = B.keys[j];
            C.vals[k] = B.vals[j];
            ++j;
        }
        ++k;
    }
    while (i < A.keys.size()) { C.keys[k] = A.keys[i]; C.vals[k] = A.vals[i]; ++i; ++k; }
    while (j < B.keys.size()) { C.keys[k] = B.keys[j]; C.vals[k] = B.vals[j]; ++j; ++k; }

    return C;
}

// =============================================================================
// Full pipeline
// =============================================================================

static void run_pipeline(const std::string& input_csv,
                         const std::string& output_csv)
{
    // ------------------------------------------------------------------
    // 1. Read CSV, split into batches, sort each on CPU
    // ------------------------------------------------------------------
    std::ifstream fin(input_csv);
    if (!fin.is_open())
        throw std::runtime_error("Could not open input file: " + input_csv);

    std::string header;
    if (!std::getline(fin, header))
        throw std::runtime_error("Input file is empty.");

    std::vector<std::string> all_rows;
    std::vector<SortedBatch> batches;
    size_t global_offset = 0;

    double total_sort_ms  = 0.0;
    double total_merge_ms = 0.0;

    auto t_pipeline_start = now();

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

        std::vector<uint64_t> batch_keys;
        batch_keys.reserve(rows.size());
        for (const auto& r : rows) batch_keys.push_back(r.key);

        // CPU sort
        auto t0 = now();
        std::vector<uint32_t> local_sorted = sort_batch_cpu(batch_keys);
        auto t1 = now();
        double sort_ms = elapsed_ms(t0, t1);
        total_sort_ms += sort_ms;

        SortedBatch sb;
        sb.keys.resize(rows.size());
        sb.vals.resize(rows.size());
        for (size_t i = 0; i < local_sorted.size(); ++i) {
            uint32_t li = local_sorted[i];
            sb.keys[i]  = batch_keys[li];
            sb.vals[i]  = static_cast<uint32_t>(global_offset + li);
        }

        for (auto& r : rows)
            all_rows.push_back(std::move(r.raw_line));

        std::cout << "Sorted batch " << batches.size()
                  << "  (" << rows.size() << " rows)"
                  << "  in " << std::fixed << std::setprecision(2) << sort_ms << " ms\n";

        batches.push_back(std::move(sb));
        global_offset += rows.size();

        if (fin.eof()) break;
    }

    if (batches.empty()) {
        std::cout << "No rows found.\n";
        return;
    }

    // ------------------------------------------------------------------
    // 2. Merge all sorted batches on CPU (pairwise)
    // ------------------------------------------------------------------
    SortedBatch current = std::move(batches[0]);

    for (size_t b = 1; b < batches.size(); ++b) {
        auto t0 = now();
        current = merge_batches_cpu(current, batches[b]);
        auto t1 = now();
        double merge_ms = elapsed_ms(t0, t1);
        total_merge_ms += merge_ms;

        // Free the batch we just consumed
        batches[b] = SortedBatch{};

        std::cout << "Merged batch " << b
                  << "  (running total: " << current.keys.size() << " rows)"
                  << "  in " << std::fixed << std::setprecision(2) << merge_ms << " ms\n";
    }

    // ------------------------------------------------------------------
    // 3. Write output CSV
    // ------------------------------------------------------------------
    std::ofstream fout(output_csv);
    if (!fout.is_open())
        throw std::runtime_error("Could not create output file: " + output_csv);

    fout << header << "\n";
    for (uint32_t idx : current.vals)
        fout << all_rows[idx] << "\n";

    auto t_pipeline_end = now();

    // ------------------------------------------------------------------
    // 4. Print timing summary
    // ------------------------------------------------------------------
    double total_ms = elapsed_ms(t_pipeline_start, t_pipeline_end);

    std::cout << "\n========================================\n";
    std::cout << "  CPU Pipeline Timing Summary\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Batch sort (total) : " << total_sort_ms  << " ms\n";
    std::cout << "  Merge (total)      : " << total_merge_ms << " ms\n";
    std::cout << "  Total wall time    : " << total_ms        << " ms\n";
    std::cout << "  Rows written       : " << current.vals.size() << "\n";
    std::cout << "  Output file        : " << output_csv << "\n";
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
                  << "  " << argv[0] << " transactions_data.csv sorted_transactions_cpu.csv\n";
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
