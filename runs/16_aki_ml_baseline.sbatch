#!/bin/bash
#SBATCH --job-name=jd_aki_ml
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/aki_ml_%j.out
#SBATCH --error=logs/aki_ml_%j.err

set -euo pipefail

PROJ="${PROJ:-/home/g91p721/jd}"
VENV="${VENV:-/home/g91p721/jd/.venv}"
DENOM="${DENOM:-/home/g91p721/agentic_delphi/results/cohort/ici_denominator_with_aki_flags.csv}"
BASELINE_FEATURES="${BASELINE_FEATURES:-${PROJ}/results/aki_phenotyping/baseline_features_preindex_charlson_pre365drugs.csv}"
OUT_DIR="${OUT_DIR:-${PROJ}/results/aki_phenotyping/ml_baseline_demo_charlson}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
CLASS_WEIGHT="${CLASS_WEIGHT:-none}"
N_SPLITS="${N_SPLITS:-5}"
SEED="${SEED:-42}"
INCLUDE_REGIMEN="${INCLUDE_REGIMEN:-0}"

cd "${PROJ}"
mkdir -p logs "${OUT_DIR}"
source runs/tempest_env.sh
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CMD=(python scripts/run_aki_ml_baseline.py \
  --denominator-csv "${DENOM}" \
  --baseline-features-csv "${BASELINE_FEATURES}" \
  --out-dir "${OUT_DIR}" \
  --model-type "${MODEL_TYPE}" \
  --class-weight "${CLASS_WEIGHT}" \
  --n-splits "${N_SPLITS}" \
  --seed "${SEED}")

if [ "${INCLUDE_REGIMEN}" = "1" ]; then
  CMD+=(--include-regimen)
fi

"${CMD[@]}"
