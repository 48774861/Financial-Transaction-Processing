#!/bin/bash
#SBATCH --job-name=txn_sort_pipeline
#SBATCH --output=logs/txn_sort_%j.out
#SBATCH --error=logs/txn_sort_%j.err

#SBATCH --partition=batch
#SBATCH --gres=gpu:1

#SBATCH --time=00:10:00

#SBATCH --account=camp_ece5385_1262_000_0001

# Modules
module purge
module load gcc
module load cuda

# Compile
echo "Compiling..."
mkdir -p logs
make clean
make
echo ""

# Run GPU pipeline
echo "Running GPU Transaction Sort Pipeline"
echo " "
./transaction_sort_pipeline transactions_data.csv sorted_transactions_gpu.csv

echo ""

# Run CPU pipeline
echo "Running CPU Transaction Sort Pipeline"
echo " "
./transaction_sort_pipeline_cpu transactions_data.csv sorted_transactions_cpu.csv