#!/bin/bash
#SBATCH --job-name=jd_hb_smoke
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/hb_smoke_%j.out
#SBATCH --error=logs/hb_smoke_%j.err

# Step 5a: HealthBench Hard smoke test, 3 cases.

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

mkdir -p results logs

python scripts/run_healthbench.py \
  --n-cases 3 \
  --seed 42 \
  --discussion-round-cap 5 \
  --run-name smoke_n3_seed42
