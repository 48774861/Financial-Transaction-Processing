# =============================================================================
# Compilers & flags
# =============================================================================

NVCC       = nvcc
CXX        = g++

NVCC_FLAGS = -O3 -std=c++17
CXX_FLAGS  = -O3 -std=c++17

# =============================================================================
# Targets
# =============================================================================

GPU_TARGET = transaction_sort_pipeline
CPU_TARGET = transaction_sort_pipeline_cpu

GPU_SRC    = transaction_sort_pipeline.cu
CPU_SRC    = transaction_sort_pipeline_cpu.cpp

# =============================================================================
# Rules
# =============================================================================

.PHONY: all clean run run_gpu run_cpu

# Build both by default
all: $(GPU_TARGET) $(CPU_TARGET)

$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXX_FLAGS) $< -o $@

# Run both back-to-back for easy comparison
run: run_gpu run_cpu

run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) transactions_data.csv sorted_transactions_gpu.csv

run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) transactions_data.csv sorted_transactions_cpu.csv

clean:
	rm -f $(GPU_TARGET) $(CPU_TARGET)