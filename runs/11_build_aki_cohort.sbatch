#!/bin/bash
#SBATCH --job-name=jd_aki_cohort
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/aki_cohort_%j.out
#SBATCH --error=logs/aki_cohort_%j.err

set -euo pipefail

PROJ="${PROJ:-/home/g91p721/jd}"
VENV="${VENV:-/home/g91p721/agentic_delphi/.venv}"
DATA_DIR="${DATA_DIR:-/home/g91p721/DelPHEA-irAKI/irAKI_data}"
DENOM="${DENOM:-/home/g91p721/agentic_delphi/results/cohort/ici_denominator_with_aki_flags.csv}"
BASELINE_FEATURES="${BASELINE_FEATURES:-${PROJ}/results/aki_phenotyping/baseline_features_pre365.csv}"
OUT="${OUT:-${PROJ}/results/aki_phenotyping/balanced_cohort.csv}"
N_PER_CLASS="${N_PER_CLASS:-250}"

cd "${PROJ}"
source runs/tempest_env.sh
source "${VENV}/bin/activate"

CMD=(python scripts/build_aki_cohort.py \
  --denominator-csv "${DENOM}" \
  --data-dir "${DATA_DIR}" \
  --out-csv "${OUT}" \
  --n-per-class "${N_PER_CLASS}" \
  --positive-evidence both \
  --control-evidence none \
  --min-control-followup-days 180)

if [ -s "${BASELINE_FEATURES}" ]; then
  CMD+=(--baseline-features-csv "${BASELINE_FEATURES}")
else
  echo "Baseline feature file not found, building cohort without it: ${BASELINE_FEATURES}"
fi

"${CMD[@]}"
