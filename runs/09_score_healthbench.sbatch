#!/bin/bash
#SBATCH --job-name=jd_hb_score
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hb_score_%j.out
#SBATCH --error=logs/hb_score_%j.err

# Step 6: Score a HealthBench run.
#
# Usage:
#   export OPENAI_API_KEY=...
#   export HB_RUN=study_n100_seed42
#   sbatch runs/09_score_healthbench.sbatch

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

HB_RUN=${HB_RUN:-study_n100_seed42}
DUP_RATE=${DUP_RATE:-0.2}

python scripts/score_healthbench.py \
  --responses-jsonl "results/healthbench_hard/${HB_RUN}/responses.jsonl" \
  --out-jsonl "results/healthbench_hard/${HB_RUN}/scores.jsonl" \
  --duplicate-rate "${DUP_RATE}"
