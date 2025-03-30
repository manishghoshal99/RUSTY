#!/bin/bash
#SBATCH --job-name=mastodon_2n8c
#SBATCH --partition=sapphire
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=./output/logs/mastodon_2n8c_%j.out
#SBATCH --error=./output/logs/mastodon_2n8c_%j.err

# Load required modules
module load Python/3.10.4
module load mpi4py/3.1.4

# Create output directories
mkdir -p ./output/results/2nodes_8cores
mkdir -p ./output/logs

# Run the MPI program with 8 processes across 2 nodes (4 per node)
srun -n 8 --nodes=2 --ntasks-per-node=4 python3 ./src/main.py -data $1 -output ./output/results/2nodes_8cores

# Copy the output to a standardized file for analysis
cp ./output/results/2nodes_8cores/runtime.txt ./output/2node8core.txt

echo "Job completed"
