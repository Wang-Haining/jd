#!/bin/bash
#SBATCH --job-name=jd_hb_study
#SBATCH --partition=gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/hb_study_%j.out
#SBATCH --error=logs/hb_study_%j.err

# Step 5c: Locked HealthBench Hard study run, 100 cases, seed 42.

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/g91p721/.cache/huggingface

mkdir -p results logs

python scripts/run_healthbench.py \
  --n-cases 100 \
  --seed 42 \
  --run-name study_n100_seed42
