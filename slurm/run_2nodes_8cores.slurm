#!/bin/bash
#SBATCH --job-name=mastodon_rust
#SBATCH --partition=sapphire
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=./output/logs/mastodon_rust_%j.out
#SBATCH --error=./output/logs/mastodon_rust_%j.err

# (If required, load any modules for Rust and MPI; for example, you might need to load OpenMPI)
module load OpenMPI

# Create necessary output directories
mkdir -p ./output/results/2nodes_8cores
mkdir -p ./output/logs

# Run the MPI program with 8 processes across 2 nodes (4 per node)
# Note: Adjust the paths to the binary and the data file as needed.
srun -n 8 --nodes=2 --ntasks-per-node=4 ./target/release/mastodon_analytics \
    -data ../data/mastodon-144g.ndjson \
    -output ./output/results/2nodes_8cores

# Optionally, copy or further process the output
cp ./output/results/2nodes_8cores/runtime.txt ./output/2node8core.txt

echo "Job completed"
