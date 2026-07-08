#!/bin/bash
#SBATCH --job-name=jd_preidx_med
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/preindex_medace_%j.out
#SBATCH --error=logs/preindex_medace_%j.err

set -euo pipefail

PROJ="${PROJ:-/home/g91p721/jd}"
VENV="${VENV:-/home/g91p721/agentic_delphi/.venv}"
DATA_DIR="${DATA_DIR:-/home/g91p721/DelPHEA-irAKI/irAKI_data}"
COHORT="${COHORT:-${PROJ}/results/aki_phenotyping/balanced_cohort.csv}"
OUT_DIR="${OUT_DIR:-${PROJ}/results/aki_phenotyping/preindex_medace_smoke}"
LIMIT="${LIMIT:-6}"
WORKERS="${WORKERS:-2}"

MODEL="${MODEL:-openai/gpt-oss-120b}"
SERVED_NAME="${SERVED_NAME:-gpt-oss:120b}"
PORT="${PORT:-8020}"
HOST="127.0.0.1"

cd "${PROJ}"
mkdir -p logs "${OUT_DIR}"
source runs/tempest_env.sh
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

GPUC=$(nvidia-smi -L 2>/dev/null | wc -l)
TP="${TP:-$GPUC}"
MAXLEN="${MAXLEN:-65536}"
MAX_SEQS="${MAX_SEQS:-8}"
GPU_UTIL="${GPU_UTIL:-0.92}"

port_in_use() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:]$1\$"
}
for _ in $(seq 0 100); do
  port_in_use "$PORT" || break
  PORT=$((PORT+1))
done
API="http://${HOST}:${PORT}"

echo "Node=$(hostname) GPUs=${GPUC} TP=${TP} model=${MODEL} port=${PORT}"
vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-prefix-caching &

SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true; wait ${SERVER_PID} 2>/dev/null || true' EXIT

for t in $(seq 1 600); do
  curl -fsS "${API}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "vLLM died"; exit 1; }
  sleep 2
done

curl -fsS "${API}/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
  >/dev/null || true

python scripts/run_preindex_medace.py \
  --cohort-csv "${COHORT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --base-url "${API}/v1" \
  --model "${SERVED_NAME}" \
  --limit "${LIMIT}" \
  --workers "${WORKERS}" \
  --overwrite

