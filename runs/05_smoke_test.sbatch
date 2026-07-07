#!/bin/bash
#SBATCH --job-name=jd_smoke
#SBATCH --partition=gputest
#SBATCH --account=group-jasonclark
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
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

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

mkdir -p results logs

python scripts/smoke_test.py 2>&1 | tee logs/smoke_$(date +%Y%m%d_%H%M%S).log
