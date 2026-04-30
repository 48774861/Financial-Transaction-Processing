# Compiler
NVCC = nvcc

# Target executable
TARGET = batch_sort

# Source file
SRC = batch_radix_sort_transactions.cu

# Compiler flags
NVCC_FLAGS = -O3 -std=c++17

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# Run program
run: $(TARGET)
	./$(TARGET) transactions_data.csv sorted_transactions

# Clean build files
clean:
	rm -f $(TARGET)