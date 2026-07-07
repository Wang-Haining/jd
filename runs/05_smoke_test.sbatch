#!/bin/bash
#SBATCH --job-name=jd_smoke
#SBATCH --partition=gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haining.wang@montana.edu

# ============================================================================
# Step 4: Smoke test — 5 MedQA cases × 3 agents × J-space extraction
#
# THIS IS THE GATE CHECK. All QC assertions must pass before full experiment.
#
# USAGE:
#   sbatch runs/05_smoke_test.sbatch
# ============================================================================

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/g91p721/.cache/huggingface

mkdir -p results logs

python scripts/smoke_test.py 2>&1 | tee logs/smoke_$(date +%Y%m%d_%H%M%S).log
