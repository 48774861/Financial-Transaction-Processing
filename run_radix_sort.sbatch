#!/bin/bash
#SBATCH --job-name=radix_sort
#SBATCH --output=radix_sort.out
#SBATCH --error=radix_sort.err

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
make clean
make
echo ""

# Run executable
echo "Running Parallel Radix Sort"
echo " "
./batch_sort