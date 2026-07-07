#!/bin/bash
#SBATCH --job-name=jd_data
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err

# ============================================================================
# Step 2: Download HealthBench Hard + generic J-lens fit corpus
#
# USAGE:
#   sbatch runs/03_download_data.sbatch
#
# NOTE: If compute nodes lack internet, run this on login node:
#   bash runs/03_download_data.sbatch
# ============================================================================

set -euo pipefail

cd /home/g91p721/jd
source runs/tempest_env.sh
source .venv/bin/activate

mkdir -p data/healthbench data/fit_corpus

echo ""
echo "=== Downloading HealthBench Hard ==="
cd data/healthbench
python -c "
from urllib.request import urlretrieve
url = 'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl'
urlretrieve(url, 'hard.jsonl')
"
echo "HealthBench Hard: $(wc -l < hard.jsonl) cases"
cd ../..

echo ""
echo "=== Building fit corpus ==="
python -c "
from datasets import load_dataset
import json

# Try C4 first, fall back to wikitext
try:
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    source = 'c4'
except Exception:
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', split='train', streaming=True)
    source = 'wikitext'

texts = []
for i, item in enumerate(ds):
    if i >= 300: break
    t = item['text'][:512]
    if len(t) > 100:
        texts.append(t)

texts = texts[:200]
with open('data/fit_corpus/c4_slice.json', 'w') as f:
    json.dump(texts, f)
print(f'Fit corpus ({source}): {len(texts)} sequences saved')
"

echo ""
echo "=== QC SUMMARY ==="
echo "HealthBench: $(wc -l < data/healthbench/hard.jsonl)"
echo "Fit corpus:  $(python -c "import json; print(len(json.load(open('data/fit_corpus/c4_slice.json'))))")"
echo ""
echo "=== DATA DOWNLOAD COMPLETE ==="
