#!/bin/bash
#SBATCH --job-name=jd_fit_lens
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/fit_lens_%j.out
#SBATCH --error=logs/fit_lens_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haining.wang@montana.edu

# ============================================================================
# Step 3: Fit Jacobian Lens for smoke-test model
#
# USAGE:
#   sbatch runs/04_fit_lens.sbatch
#
# BEFORE RUNNING:
#   - Step 01 passed (GPU available)
#   - Step 02 passed (venv ready)
#   - Step 03 passed (fit corpus downloaded)
# ============================================================================

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

mkdir -p checkpoints

python src/jspace/fit_lens.py
