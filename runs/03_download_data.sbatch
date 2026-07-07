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
# Step 2: Download benchmarks + fit corpus
#
# USAGE:
#   sbatch runs/03_download_data.sbatch
#
# NOTE: If compute nodes lack internet, run this on login node:
#   bash runs/03_download_data.sbatch
# ============================================================================

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1

mkdir -p data/medqa data/diversity data/healthbench data/fit_corpus

echo "=== Downloading MedQA ==="
python -c "
from datasets import load_dataset
ds = load_dataset('GBaker/MedQA-USMLE-4-options')
ds.save_to_disk('data/medqa/full')
print(f'MedQA splits: {list(ds.keys())}')
print(f'Test set: {len(ds[\"test\"])} cases')
print(f'Sample Q: {ds[\"test\"][0][\"question\"][:120]}...')
"

echo ""
echo "=== Downloading DiversityMedQA ==="
python -c "
from datasets import load_dataset
ds = load_dataset('Rajat1212/DiversityMedQA')
ds.save_to_disk('data/diversity/full')
print(f'DiversityMedQA: {ds}')
"

echo ""
echo "=== Downloading HealthBench Hard ==="
cd data/healthbench
wget -q -O hard.jsonl \
  'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl'
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
echo "MedQA:       $(python -c "from datasets import load_from_disk; print(len(load_from_disk('data/medqa/full')['test']))")"
echo "Diversity:   $(python -c "from datasets import load_from_disk; d=load_from_disk('data/diversity/full'); print(len(list(d.values())[0]))")"
echo "HealthBench: $(wc -l < data/healthbench/hard.jsonl)"
echo "Fit corpus:  $(python -c "import json; print(len(json.load(open('data/fit_corpus/c4_slice.json'))))")"
echo ""
echo "=== DATA DOWNLOAD COMPLETE ==="
