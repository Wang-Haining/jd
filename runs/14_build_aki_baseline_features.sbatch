#!/bin/bash
#SBATCH --job-name=jd_aki_basefeat
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/aki_basefeat_%j.out
#SBATCH --error=logs/aki_basefeat_%j.err

set -euo pipefail

PROJ="${PROJ:-/home/g91p721/jd}"
VENV="${VENV:-/home/g91p721/agentic_delphi/.venv}"
DATA_DIR="${DATA_DIR:-/home/g91p721/DelPHEA-irAKI/irAKI_data}"
DENOM="${DENOM:-/home/g91p721/agentic_delphi/results/cohort/ici_denominator_with_aki_flags.csv}"
OUT="${OUT:-${PROJ}/results/aki_phenotyping/baseline_features_preindex_charlson_pre365drugs.csv}"
CHARLSON_LOOKBACK_DAYS="${CHARLSON_LOOKBACK_DAYS:-}"
DRUG_LOOKBACK_DAYS="${DRUG_LOOKBACK_DAYS:-365}"
MAX_DRUGS="${MAX_DRUGS:-80}"

cd "${PROJ}"
mkdir -p logs "$(dirname "${OUT}")"
source runs/tempest_env.sh
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

CMD=(python scripts/build_aki_baseline_features.py \
  --cohort-csv "${DENOM}" \
  --data-dir "${DATA_DIR}" \
  --out-csv "${OUT}" \
  --drug-lookback-days "${DRUG_LOOKBACK_DAYS}" \
  --max-drugs "${MAX_DRUGS}")

if [ -n "${CHARLSON_LOOKBACK_DAYS}" ]; then
  CMD+=(--charlson-lookback-days "${CHARLSON_LOOKBACK_DAYS}")
fi

"${CMD[@]}"
