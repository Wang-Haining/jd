#!/bin/bash
#SBATCH --job-name=jd_hb_pilot
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/hb_pilot_%j.out
#SBATCH --error=logs/hb_pilot_%j.err

# Step 5b: HealthBench Hard pilot, 20 cases.

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

mkdir -p results logs

python scripts/run_healthbench.py \
  --n-cases 20 \
  --seed 42 \
  --discussion-round-cap 5 \
  --run-name pilot_n20_seed42
