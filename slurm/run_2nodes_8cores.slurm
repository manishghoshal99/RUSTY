#!/bin/bash
#SBATCH --job-name=mastodon_rust
#SBATCH --partition=sapphire
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=./mastodon-analytics/output/logs/mastodon_rust_%j.out
#SBATCH --error=./mastodon-analytics/output/logs/mastodon_rust_%j.err

module load OpenMPI

mkdir -p ./mastodon-analytics/output/results/2nodes_8cores
mkdir -p ./mastodon-analytics/output/logs

srun -n 8 --nodes=2 --ntasks-per-node=4 \
  ./mastodon-analytics/target/release/mastodon_analytics \
  -data ./mastodon-analytics/data/mastodon-144g.ndjson \
  -output ./mastodon-analytics/output/results/2nodes_8cores

echo "Job completed"
