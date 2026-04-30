#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t BATCH_SIZE = 1'000'000;


#define CUDA_CHECK(call)                                                        
    do {                                                                        
        cudaError_t err__ = (call);                                             
        if (err__ != cudaSuccess) {                                             
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    
            std::exit(EXIT_FAILURE);                                            
        }                                                                       
    } while (0)

#define CUB_CHECK(call)                                                         
    do {                                                                        
        cudaError_t err__ = (call);                                            
        if (err__ != cudaSuccess) {                                             
            std::cerr << "CUB error: " << cudaGetErrorString(err__)             
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    
            std::exit(EXIT_FAILURE);                                            
        }                                                                       
    } while (0)


static inline int64_t days_from_civil(int y, unsigned m, unsigned d) {
    y -= (m <= 2);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);             // [0, 399]
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;  // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;            // [0, 146096]
    return era * 146097 + static_cast<int>(doe) - 719468;
}

static inline uint32_t parse_datetime_to_epoch_seconds(const std::string& s) {
    // Expect "YYYY-MM-DD HH:MM:SS"
    if (s.size() < 19) {
        throw std::runtime_error("Bad date field: " + s);
    }

    int year   = std::stoi(s.substr(0, 4));
    int month  = std::stoi(s.substr(5, 2));
    int day    = std::stoi(s.substr(8, 2));
    int hour   = std::stoi(s.substr(11, 2));
    int minute = std::stoi(s.substr(14, 2));
    int second = std::stoi(s.substr(17, 2));

    int64_t days = days_from_civil(year, static_cast<unsigned>(month), static_cast<unsigned>(day));
    int64_t secs = days * 86400LL + hour * 3600LL + minute * 60LL + second;

    if (secs < 0 || secs > 0xFFFFFFFFULL) {
        throw std::runtime_error("Date out of uint32 range: " + s);
    }

    return static_cast<uint32_t>(secs);
}


struct ParsedRow {
    std::string raw_line;   // original CSV row
    uint64_t key;           // packed sort key: (date_seconds << 32) | client_id
};

static inline std::vector<std::string> split_csv_simple(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        fields.push_back(item);
    }
    return fields;
}

static inline uint64_t make_sort_key(const std::string& date_str, const std::string& client_id_str) {
    uint32_t date_seconds = parse_datetime_to_epoch_seconds(date_str);
    uint32_t client_id = static_cast<uint32_t>(std::stoul(client_id_str));

    // Primary sort: date
    // Secondary sort: client_id
    return (static_cast<uint64_t>(date_seconds) << 32) | static_cast<uint64_t>(client_id);
}

static inline bool parse_transaction_line(const std::string& line, ParsedRow& out_row) {
    // Expected columns:
    // id,date,client_id,card_id,amount,use_chip,merchant_id,merchant_city,merchant_state,zip,mcc,errors
    auto fields = split_csv_simple(line);
    if (fields.size() < 3) {
        return false;
    }

    // fields[1] = date
    // fields[2] = client_id
    out_row.raw_line = line;
    out_row.key = make_sort_key(fields[1], fields[2]);
    return true;
}

static std::vector<uint32_t> sort_batch_on_gpu(const std::vector<uint64_t>& h_keys) {
    const size_t n = h_keys.size();
    if (n == 0) return {};

    std::vector<uint32_t> h_indices(n);
    for (uint32_t i = 0; i < n; ++i) {
        h_indices[i] = i;
    }

    uint64_t *d_keys_in = nullptr, *d_keys_out = nullptr;
    uint32_t *d_vals_in = nullptr, *d_vals_out = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cudaMalloc(&d_keys_in,  n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_in,  n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_out, n * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals_in, h_indices.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Query temp storage size
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(n)
    ));

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Actual sort
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(n)
    ));

    std::vector<uint32_t> sorted_indices(n);
    CUDA_CHECK(cudaMemcpy(sorted_indices.data(), d_vals_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_temp_storage);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_vals_in);
    cudaFree(d_vals_out);

    return sorted_indices;
}

static void process_csv_in_batches(const std::string& input_csv, const std::string& output_prefix) {
    std::ifstream fin(input_csv);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open input file: " + input_csv);
    }

    std::string header;
    if (!std::getline(fin, header)) {
        throw std::runtime_error("Input file is empty.");
    }

    size_t batch_id = 0;
    size_t total_rows = 0;

    while (true) {
        std::vector<ParsedRow> rows;
        rows.reserve(BATCH_SIZE);

        std::string line;
        while (rows.size() < BATCH_SIZE && std::getline(fin, line)) {
            if (line.empty()) continue;
            ParsedRow row;
            if (parse_transaction_line(line, row)) {
                rows.push_back(std::move(row));
            }
        }

        if (rows.empty()) {
            break;
        }

        std::vector<uint64_t> keys;
        keys.reserve(rows.size());
        for (const auto& r : rows) {
            keys.push_back(r.key);
        }

        std::vector<uint32_t> sorted_indices = sort_batch_on_gpu(keys);

        std::ostringstream out_name;
        out_name << output_prefix << "_batch_" << std::setw(4) << std::setfill('0') << batch_id << ".csv";

        std::ofstream fout(out_name.str());
        if (!fout.is_open()) {
            throw std::runtime_error("Could not create output file: " + out_name.str());
        }

        fout << header << "\n";
        for (uint32_t idx : sorted_indices) {
            fout << rows[idx].raw_line << "\n";
        }

        total_rows += rows.size();

        std::cout << "Wrote batch " << batch_id
                  << " with " << rows.size()
                  << " sorted rows -> " << out_name.str() << "\n";

        ++batch_id;

        if (fin.eof()) {
            break;
        }
    }

    std::cout << "Done. Total rows processed: " << total_rows << "\n";
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <input.csv> <output_prefix>\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " transactions_data.csv sorted_transactions\n";
        return EXIT_FAILURE;
    }

    try {
        process_csv_in_batches(argv[1], argv[2]);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}