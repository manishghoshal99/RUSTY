#!/bin/bash
#SBATCH --job-name=mastodon_1n1c
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=./output/logs/mastodon_1n1c_%j.out
#SBATCH --error=./output/logs/mastodon_1n1c_%j.err

# Load required modules
module load Python/3.10.4
module load mpi4py/3.1.4

# Create output directory
mkdir -p ./output/results/1node_1core
mkdir -p ./output/logs

# Run the MPI program with 1 process
srun -n 1 python3 ./src/main.py -data $1 -output ./output/results/1node_1core

# Copy the output to a standardized file for analysis
cp ./output/results/1node_1core/runtime.txt ./output/1node1core.txt

echo "Job completed"