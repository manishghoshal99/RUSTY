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

module load OpenMPI

# Change into the directory containing your Cargo project
cd mastodon-analytics

mkdir -p ./output/results/2nodes_8cores
mkdir -p ./output/logs

# Run the binary; note the relative path now is correct because you are in mastodon-analytics
srun -n 8 --nodes=2 --ntasks-per-node=4 ./target/release/mastodon_analytics \
    -data ./data/mastodon-144g.ndjson \
    -output ./output/results/2nodes_8cores

echo "Job completed"
