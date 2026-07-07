#!/bin/bash
#SBATCH --job-name=jd_hb_study
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/hb_study_%j.out
#SBATCH --error=logs/hb_study_%j.err

# Step 5c: Locked HealthBench Hard study run, 100 cases, seed 42.

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

mkdir -p results logs

python scripts/run_healthbench.py \
  --n-cases 100 \
  --seed 42 \
  --discussion-round-cap 5 \
  --run-name study_n100_seed42
