#!/bin/bash
#SBATCH --job-name=jd_hb_analysis
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hb_analysis_%j.out
#SBATCH --error=logs/hb_analysis_%j.err

# Step 7: Generate HealthBench analysis tables and figures.

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

HB_RUN=${HB_RUN:-study_n100_seed42}

python scripts/analyze_healthbench.py \
  --scores-jsonl "results/healthbench_hard/${HB_RUN}/scores.jsonl" \
  --out-dir "results/healthbench_hard/${HB_RUN}/analysis"
