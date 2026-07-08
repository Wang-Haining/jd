#!/bin/bash
#SBATCH --job-name=jd_aki_route30
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/aki_route30_%j.out
#SBATCH --error=logs/aki_route30_%j.err

set -euo pipefail

PROJ="${PROJ:-/home/g91p721/jd}"
VLLM_VENV="${VLLM_VENV:-/home/g91p721/agentic_delphi/.venv}"
RUN_VENV="${RUN_VENV:-/home/g91p721/jd/.venv}"
COHORT="${COHORT:-${PROJ}/results/aki_phenotyping/balanced_cohort_30v30_preindex_charlson_pre365drugs.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-${PROJ}/results/aki_phenotyping/preindex_medace_30v30_preindex_charlson}"
OUT_JSONL="${OUT_JSONL:-${PROJ}/results/aki_phenotyping/aki_routing_30v30_preindex_charlson_qwen.jsonl}"
OUT_METRICS="${OUT_METRICS:-${PROJ}/results/aki_phenotyping/aki_routing_30v30_preindex_charlson_qwen_metrics.csv}"
LIMIT="${LIMIT:-60}"
TURN_CAP="${TURN_CAP:-5}"
STRATEGIES="${STRATEGIES:-single_nephrologist,five_agent_independent_mean,arbitrary_round_robin,agent_handoff,jlens_low_std,jlens_high_std}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen2.5:7b}"
JLENS_HF_MODEL="${JLENS_HF_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENS_PATH="${LENS_PATH:-${PROJ}/checkpoints/qwen_qwen2.5_7b_instruct_lens.pt}"
TOP_K="${TOP_K:-50}"
PORT="${PORT:-8020}"
HOST="127.0.0.1"

cd "${PROJ}"
mkdir -p logs "$(dirname "${OUT_JSONL}")"
source runs/tempest_env.sh
source "${VLLM_VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

GPUC=$(nvidia-smi -L 2>/dev/null | wc -l)
TP="${TP:-1}"
MAXLEN="${MAXLEN:-32768}"
MAX_SEQS="${MAX_SEQS:-8}"
GPU_UTIL="${GPU_UTIL:-0.55}"

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

source "${RUN_VENV}/bin/activate"

python scripts/run_aki_routing_agents.py \
  --cohort-csv "${COHORT}" \
  --summary-dir "${SUMMARY_DIR}" \
  --out-jsonl "${OUT_JSONL}" \
  --base-url "${API}/v1" \
  --model "${SERVED_NAME}" \
  --limit "${LIMIT}" \
  --turn-cap "${TURN_CAP}" \
  --strategies "${STRATEGIES}" \
  --jlens-hf-model "${JLENS_HF_MODEL}" \
  --lens-path "${LENS_PATH}" \
  --top-k "${TOP_K}" \
  --overwrite

python scripts/analyze_aki_routing.py \
  --pred-jsonl "${OUT_JSONL}" \
  --out-csv "${OUT_METRICS}"
