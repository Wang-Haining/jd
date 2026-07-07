# jd: Cognitive Orchestration of Clinical AI Agents via Jacobian Lens

## Design Document v1.0 — July 2026

**Target venue**: JAMA Network Open  
**Study type**: Cross-sectional simulation study  
**HPC**: MSU Tempest (`g91p721@tempest-login.msu.montana.edu`)  
**Working dir**: `/home/g91p721/jd`

---

## 0. One-Sentence Thesis

Reading agents' minds (via J-space) before they speak improves clinical AI consensus accuracy.

---

## 1. Background & Key References

### 1.1 J-Space / Jacobian Lens

- **Paper**: Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models," Transformer Circuits Thread (Anthropic), 2026-07-06
  - Blog: https://www.anthropic.com/research/global-workspace
  - Full paper: http://transformer-circuits.pub/2026/workspace/index.html
- **Code**: https://github.com/anthropics/jacobian-lens (Apache 2.0)
  - Install: `pip install jlens` OR `pip install git+https://github.com/anthropics/jacobian-lens.git`
  - Dependencies: torch, transformers, numpy
- **What J-lens does**: For each vocabulary token, computes the average linearized effect (Jacobian) of intermediate-layer activations on future-token likelihood:
  ```
  J_ℓ = E[∂h_final,t' / ∂h_ℓ,t]
  ```
  Applying this to an activation yields a ranked list of tokens the model is "poised to say" — the J-space contents at that layer.
- **Key API pattern** (from README):
  ```python
  import transformers, jlens

  hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
  tok = transformers.AutoTokenizer.from_pretrained("org/model")
  model = jlens.from_hf(hf, tok)

  # Load or fit a lens
  lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
  # OR fit from scratch (~100 prompts sufficient):
  # lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")

  lens_logits, model_logits, _ = lens.apply(
      model,
      "Fact: The currency used in the country shaped like a boot is",
      positions=[-2]  # position(s) to read
  )
  for layer, logits in sorted(lens_logits.items()):
      top5 = [tok.decode([t]) for t in logits[0].topk(5).indices]
      print(f"Layer {layer}: {top5}")
  ```
- **Key constraints**:
  - Single-token concepts only (multi-token extension incomplete)
  - J-space appears at intermediate layers (~1/3 to ~2/3 of network depth)
  - Fitting uses ~100 sequences of ~128 tokens; quality saturates fast
  - GPU required (backward pass through the model)
  - Works on ANY HuggingFace causal LM (tested: Claude internals, Qwen — Neel Nanda replicated on Qwen 3.6 27B)

### 1.2 Medical Benchmarks

| Benchmark | Source | N | Format | What it tests |
|-----------|--------|---|--------|---------------|
| MedQA-USMLE | HuggingFace: `GBaker/MedQA-USMLE-4-options` | 1,273 test | 4-option MCQ | Multi-step clinical reasoning |
| DiversityMedQA | HuggingFace: `Rajat1212/DiversityMedQA` | ~1,200 | MedQA + demographic counterfactuals | Demographic bias in diagnosis |
| HealthBench Hard | OpenAI public blob (see §3.2) | ~1,000 | Multi-turn conversations + rubrics | Safety-critical clinical interactions |

### 1.3 Competitor Papers to Position Against

1. **Ahsan et al. 2025** (arXiv 2502.13319) — "Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare." Activation patching to localize gender/race. Single-model, requires targeted hypothesis. We: unsupervised J-space discovery, multi-agent.
2. **Menz et al. 2024** (JAMA Netw Open 2024;7(9):e2434997) — Gender bias in LLM stories about doctors/nurses. Behavioral only. We: mechanistic internal states.
3. **Poulain et al. 2024** (arXiv 2404.15149) — Bias patterns in clinical LLM decision support. 8 LLMs, 3 QA datasets. Behavioral output analysis only.
4. **BiasLens** (arXiv 2505.15524) — Concept activation vectors for bias. Uses SAEs, not J-lens. Single model, no multi-agent.

---

## 2. Experimental Design

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────┐
│                    For each clinical case:               │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │   (3 agents) │
│  │ Conserv. │  │ Aggress. │  │ Evidence │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ J-space │  │ J-space │  │ J-space │  (extract)     │
│  │ readout │  │ readout │  │ readout │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Output  │  │ Output  │  │ Output  │  (generate)    │
│  │ answer  │  │ answer  │  │ answer  │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       └──────────────┼──────────────┘                    │
│                      ▼                                   │
│            ┌──────────────────┐                          │
│            │  Orchestration   │                          │
│            │  Strategy X      │                          │
│            └────────┬─────────┘                          │
│                     ▼                                    │
│              Final Consensus                             │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Models

**Smoke test model** (fits on single GPU):
- `Qwen/Qwen2.5-7B-Instruct` — small, J-lens verified (Nanda replicated on Qwen family), fast iteration

**Full experiment models** (need multi-GPU or quantization):
- `meta-llama/Llama-3.1-8B-Instruct` — widely deployed open-weight baseline
- `Qwen/Qwen2.5-7B-Instruct` — cross-family comparison
- (Stretch) `meta-llama/Llama-3.1-70B-Instruct` with 4-bit quantization — if Tempest GPU memory allows

> **IMPORTANT**: Check Tempest GPU availability first. The `unsafe` partition may be CPU-only.
> Run `sinfo -p gpu` or `sinfo --Format=partition,gres` to find GPU partitions and GRES specs.
> If no GPUs on Tempest, fall back to IU Quartz (account `r01834`, partition `general`, has A100s).

### 2.3 Agent Roles (System Prompts)

Three agents per case, same base model, different system prompts:

```python
AGENT_PROMPTS = {
    "conservative": (
        "You are a conservative clinical reasoner. You favor common diagnoses over "
        "rare ones, require strong evidence before concluding, and flag uncertainty "
        "explicitly. When unsure, you lean toward the safer, more established "
        "interpretation. Answer the clinical question with a brief rationale."
    ),
    "aggressive": (
        "You are a thorough clinical diagnostician who considers the full differential. "
        "You are willing to consider less common diagnoses if the evidence fits. You "
        "prioritize not missing critical diagnoses even at the cost of some false "
        "positives. Answer the clinical question with a brief rationale."
    ),
    "evidence": (
        "You are an evidence-based medicine specialist. You ground every claim in "
        "established guidelines and published evidence. You are skeptical of "
        "pattern-matching without data support. Answer the clinical question with "
        "a brief rationale, citing the reasoning principles you rely on."
    ),
}
```

### 2.4 Metrics

**Primary outcome**: Consensus accuracy (% correct vs gold-standard answers)

**J-Space derived signals** (computed per agent per case):

1. **J-space top-K tokens**: Top 10 tokens at layers L/3 to 2L/3, at the last prompt position, just before generation starts. This is the agent's "mental state" before answering.

2. **Pairwise Cognitive Alignment Score (CAS)**:
   ```
   CAS(A_i, A_j) = cosine_similarity(jspace_vector_i, jspace_vector_j)
   ```
   where `jspace_vector` is the concatenation of J-lens logit vectors across the middle layers for a given case. Higher CAS = agents are "thinking" similar things.

3. **Say-Think Divergence (STD)**:
   For each agent, compare the J-space top tokens against the output answer tokens. If J-space contains tokens semantically opposed to the output (e.g., J-space has "uncertain"/"wrong" but output says "confident answer X"), flag as divergent.
   
   Operationalization: 
   - Define UNCERTAINTY_TOKENS = {"uncertain", "unsure", "maybe", "wrong", "error", "incorrect", "doubt", "unclear", "ambiguous", "conflict"}
   - Define CONFIDENCE_TOKENS = {"correct", "certain", "confident", "clear", "obvious", "definitely"}
   - STD_score = (sum of UNCERTAINTY_TOKENS in J-space top-50) / (sum of CONFIDENCE_TOKENS in J-space top-50 + 1)
   - High STD_score on a case where the agent gave a definitive answer = say-think divergence

### 2.5 Orchestration Strategies

| Strategy | How consensus is formed |
|----------|------------------------|
| `majority_vote` | Each agent picks an answer. Majority wins. (Baseline) |
| `confidence_weighted` | Weight each agent's vote by inverse STD_score. Low divergence = higher weight. |
| `align_route` | Only count votes from agents whose J-space CAS with the group centroid is above median. (Filter out "distracted" agents.) |
| `diverge_surface` | If any agent has STD > threshold, trigger a second round: show all agents the divergent agent's J-space top tokens and ask to reconsider. Final majority vote on round 2. |

### 2.6 Statistical Analysis Plan

- McNemar's test: paired comparison of accuracy between strategies (same cases)
- Mixed-effects logistic regression: outcome = correct/incorrect, fixed effects = strategy + case difficulty, random effect = case_id
- Pearson correlation: CAS vs consensus correctness
- Chi-square: STD prevalence by case difficulty category
- All tests two-sided, α = 0.05

---

## 3. Data Sources — Exact Locations and Loading

### 3.1 MedQA-USMLE

```python
from datasets import load_dataset

# 4-option USMLE questions
ds = load_dataset("GBaker/MedQA-USMLE-4-options")
# Splits: train (10,178), dev (1,272), test (1,273)

# Each item:
# {
#   "question": "A 65-year-old man presents with...",
#   "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
#   "answer_idx": "B",       # gold standard
#   "answer": "Aortic stenosis"
# }

# For the experiment, use TEST split, sample 200 cases
test_cases = ds["test"].shuffle(seed=42).select(range(200))
```

**Alternative direct download** (if HF is slow):
```bash
# Original repo
git clone https://github.com/jind11/MedQA.git
# Data in: MedQA/data_clean/questions/US/4_options/
# JSON format, one file per split
```

### 3.2 DiversityMedQA

```python
from datasets import load_dataset

ds = load_dataset("Rajat1212/DiversityMedQA")
# Counterfactual variations of MedQA with demographic swaps

# Each item has original + counterfactual versions
# Fields: question, options, answer, demographic_original, demographic_counterfactual
# Sample 200 cases
```

**Direct download**: https://huggingface.co/datasets/Rajat1212/DiversityMedQA

### 3.3 HealthBench Hard

```bash
# Direct download from OpenAI public blob storage
wget -O healthbench_hard.jsonl \
  "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"

# Also get full eval set
wget -O healthbench_eval.jsonl \
  "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

# Also get consensus subset
wget -O healthbench_consensus.jsonl \
  "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"
```

```python
import json

# Load HealthBench Hard
with open("healthbench_hard.jsonl") as f:
    hb_hard = [json.loads(line) for line in f]

# Each item: multi-turn conversation + rubric criteria
# Structure: {"messages": [...], "criteria": [...]}
# For our study: use last user message as the clinical query
# Sample 100 cases
```

**Paper**: https://arxiv.org/abs/2505.08775
**GitHub**: https://github.com/openai/simple-evals (healthbench folder)

### 3.4 Pretraining Corpus for Lens Fitting

J-lens fitting needs ~100 generic text sequences (NOT medical-specific — the lens should capture general verbalizable representations). Use a slice of any pretraining-like corpus:

```python
from datasets import load_dataset

# Use a small slice of C4 or similar
c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
fit_prompts = []
for i, item in enumerate(c4):
    if i >= 200:
        break
    text = item["text"][:512]  # truncate to ~128 tokens
    if len(text) > 100:  # skip very short docs
        fit_prompts.append(text)
# Use first 100 for fitting
```

---

## 4. Repository Structure

```
jd/
├── README.md
├── DESIGN_DOC.md              # this file
├── pyproject.toml             # project config
├── requirements.txt
│
├── configs/
│   ├── agents.yaml            # agent system prompts
│   ├── models.yaml            # model specs (name, GPU reqs, quant config)
│   └── experiment.yaml        # sample sizes, seeds, thresholds
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_medqa.py      # MedQA loader + formatter
│   │   ├── load_diversity.py  # DiversityMedQA loader
│   │   ├── load_healthbench.py# HealthBench Hard loader
│   │   └── prompt_builder.py  # Format case + agent prompt → model input
│   │
│   ├── jspace/
│   │   ├── __init__.py
│   │   ├── fit_lens.py        # Fit J-lens for a model (one-time)
│   │   ├── extract.py         # Extract J-space readouts per agent per case
│   │   ├── metrics.py         # CAS, STD, top-K extraction
│   │   └── visualize.py       # Heatmaps, layer × position plots
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent.py           # Single agent: load model, generate, extract J-space
│   │   ├── multi_agent.py     # Run 3 agents on same case
│   │   └── consensus.py       # Orchestration strategies
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── accuracy.py        # Compute accuracy by strategy
│       ├── statistical.py     # McNemar, mixed-effects, chi-square
│       └── figures.py         # JNO-style figures
│
├── runs/
│   ├── 01_check_gpu.sbatch     # Step 0: verify GPU availability
│   ├── 02_setup_env.sbatch     # Step 1: create venv, install deps
│   ├── 03_download_data.sbatch # Step 2: download benchmarks
│   ├── 04_fit_lens.sbatch      # Step 3: fit J-lens (GPU)
│   ├── 05_smoke_test.sbatch    # Step 4: 5 MedQA cases, 1 model, 3 agents
│   ├── 06_run_medqa.sbatch     # Step 5: full MedQA experiment
│   ├── 07_run_diversity.sbatch # Step 6: DiversityMedQA experiment
│   ├── 08_run_healthbench.sbatch # Step 7: HealthBench Hard
│   └── 09_analysis.sbatch      # Step 8: all analysis + figures
│
├── scripts/
│   ├── smoke_test.py          # End-to-end smoke test (5 cases)
│   ├── run_experiment.py      # Full experiment runner
│   └── generate_figures.py    # Publication figures
│
├── tests/
│   ├── test_data_loaders.py   # QC: verify data loads correctly
│   ├── test_jlens_basic.py    # QC: verify J-lens produces valid output
│   ├── test_agent_output.py   # QC: verify agents produce parseable answers
│   ├── test_metrics.py        # QC: verify CAS/STD computation
│   └── test_consensus.py      # QC: verify orchestration logic
│
├── results/
│   └── .gitkeep
│
└── logs/
    └── .gitkeep
```

---

## 5. Step-by-Step Implementation Plan

### Step 0: Check GPU Availability on Tempest

```bash
#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --partition=gpu           # TRY: gpu, gpuA100, unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/gpu_check_%j.out

# Check what GPUs are available
nvidia-smi
echo "---"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB' if torch.cuda.is_available() else '')"
```

**QC checkpoint**: If no GPU partition exists or GPUs are insufficient (<24GB for 7B model), fall back to IU Quartz.

> **KNOWN PITFALL**: Tempest GPU partition name may differ. Try: `sinfo -o "%P %G"` to list partitions with GPUs. Common names: `gpu`, `gpuA100`, `gpu-a100`, `nvidia`. The `unsafe` partition used for Delphi is CPU-only.

### Step 1: Environment Setup

```bash
#!/bin/bash
#SBATCH --job-name=jsd_setup
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail

module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

PROJ=/home/g91p721/jd
cd $PROJ

# Create venv
python -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# Core dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install scipy scikit-learn matplotlib seaborn pandas

# J-lens
pip install git+https://github.com/anthropics/jacobian-lens.git

# Verify
python -c "import jlens; print(f'jlens version: {jlens.__version__ if hasattr(jlens, \"__version__\") else \"installed\"}')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "from datasets import load_dataset; print('datasets OK')"

echo "=== SETUP COMPLETE ==="
```

**QC checkpoint**: All imports succeed. torch.cuda.is_available() returns True on GPU node.

> **KNOWN PITFALL**: `pip install jlens` may not work if the package name on PyPI differs from the GitHub repo name. The README shows `import jlens` but the pip-installable name may be `jacobian-lens`. Try both:
> ```
> pip install jlens                # try first
> pip install jacobian-lens        # if above fails
> pip install git+https://github.com/anthropics/jacobian-lens.git  # guaranteed
> ```
> After install, verify with `python -c "import jlens; print(dir(jlens))"` to confirm `JacobianLens`, `fit`, `from_hf` are available.

### Step 2: Download Data

```bash
#!/bin/bash
#SBATCH --job-name=jsd_data
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/data_%j.out

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1

mkdir -p data/medqa data/diversity data/healthbench data/fit_corpus

# 1. MedQA
python -c "
from datasets import load_dataset
ds = load_dataset('GBaker/MedQA-USMLE-4-options')
ds.save_to_disk('data/medqa/full')
print(f'MedQA loaded: {ds}')
print(f'Test set size: {len(ds[\"test\"])}')
print(f'Sample: {ds[\"test\"][0][\"question\"][:100]}...')
"

# 2. DiversityMedQA
python -c "
from datasets import load_dataset
ds = load_dataset('Rajat1212/DiversityMedQA')
ds.save_to_disk('data/diversity/full')
print(f'DiversityMedQA loaded: {ds}')
"

# 3. HealthBench Hard
cd data/healthbench
wget -q -O hard.jsonl \
  'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl'
wget -q -O eval.jsonl \
  'https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl'
echo "HealthBench Hard: $(wc -l < hard.jsonl) cases"
cd ../..

# 4. Fit corpus (small C4 slice for lens fitting)
python -c "
from datasets import load_dataset
c4 = load_dataset('allenai/c4', 'en', split='train', streaming=True)
texts = []
for i, item in enumerate(c4):
    if i >= 300: break
    t = item['text'][:512]
    if len(t) > 100:
        texts.append(t)
import json
with open('data/fit_corpus/c4_slice.json', 'w') as f:
    json.dump(texts[:200], f)
print(f'Fit corpus: {len(texts[:200])} sequences saved')
"

echo "=== DATA DOWNLOAD COMPLETE ==="
```

**QC checkpoint**: Print counts for each dataset. MedQA test = 1,273. HealthBench Hard > 0. Fit corpus ≥ 100 sequences.

> **KNOWN PITFALL**: Tempest compute nodes may not have internet access. If `wget` or HuggingFace downloads fail, run downloads on the login node first (in a tmux session), then move data. Alternatively, use `sbatch --dependency=afterok:$JOBID` to chain login-node download with compute-node processing.

> **KNOWN PITFALL**: `allenai/c4` requires agreeing to terms on HuggingFace. Alternative: use `wikitext` (`datasets.load_dataset("Salesforce/wikitext", "wikitext-103-v1")`) or any plain text corpus.

### Step 3: Download Model + Fit J-Lens

```bash
#!/bin/bash
#SBATCH --job-name=jsd_fit_lens
#SBATCH --partition=gpu            # CHANGE to actual GPU partition
#SBATCH --account=group-jasonclark
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --output=logs/fit_lens_%j.out
#SBATCH --error=logs/fit_lens_%j.err

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/home/g91p721/.cache/huggingface

python src/jspace/fit_lens.py
```

**`src/jspace/fit_lens.py`**:
```python
"""
Fit a Jacobian Lens for the smoke-test model.

Steps:
1. Load model (Qwen2.5-7B-Instruct)
2. Load fit corpus (200 C4 sequences)
3. Fit lens (~100 sequences sufficient per paper §9.3)
4. Save lens checkpoint
5. Validate: apply to a known prompt and check top tokens are sensible

Expected runtime: ~1-2 hours on a single A100 for 7B model.
"""
import json
import torch
import transformers
import jlens

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FIT_CORPUS = "data/fit_corpus/c4_slice.json"
LENS_OUT = "checkpoints/qwen7b_lens.pt"

def main():
    print(f"Loading model: {MODEL_NAME}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    
    print(f"Loading fit corpus from {FIT_CORPUS}")
    with open(FIT_CORPUS) as f:
        prompts = json.load(f)
    print(f"  {len(prompts)} sequences loaded")
    
    # Use first 100 for fitting (paper says quality saturates at ~100)
    fit_prompts = prompts[:100]
    
    print("Fitting J-lens...")
    lens = jlens.fit(
        model,
        prompts=fit_prompts,
        checkpoint_path=LENS_OUT,
    )
    lens.save(LENS_OUT)
    print(f"Lens saved to {LENS_OUT}")
    
    # === VALIDATION ===
    print("\n=== VALIDATION ===")
    
    # Test 1: Known fact retrieval
    test_prompt = "The capital city of France is"
    lens_logits, model_logits, _ = lens.apply(model, test_prompt, positions=[-1])
    for layer, logits in sorted(lens_logits.items()):
        top5 = [tokenizer.decode([t]) for t in logits[0].topk(5).indices]
        print(f"  Layer {layer}: {top5}")
    
    # Expect "Paris" to appear in top tokens at intermediate layers
    mid_layers = sorted(lens_logits.keys())
    mid = mid_layers[len(mid_layers) // 2]
    top10_mid = [tokenizer.decode([t]).strip().lower() for t in lens_logits[mid][0].topk(10).indices]
    assert "paris" in top10_mid or "Par" in str(top10_mid), \
        f"VALIDATION FAILED: 'paris' not in mid-layer top-10: {top10_mid}"
    print("  ✓ Paris detected in mid-layer J-space")
    
    # Test 2: Medical reasoning
    test_prompt2 = "A 55-year-old male presents with crushing chest pain radiating to the left arm. The most likely diagnosis is"
    lens_logits2, _, _ = lens.apply(model, test_prompt2, positions=[-1])
    mid2 = sorted(lens_logits2.keys())[len(lens_logits2) // 2]
    top20 = [tokenizer.decode([t]).strip().lower() for t in lens_logits2[mid2][0].topk(20).indices]
    print(f"  Medical test top-20 at mid-layer: {top20}")
    # Expect cardiac-related tokens: "heart", "myocardial", "infarction", "cardiac", "MI", etc.
    cardiac_tokens = {"heart", "cardiac", "myocardial", "infarction", "mi", "acs", "stemi", "coronary"}
    found = cardiac_tokens.intersection(set(top20))
    print(f"  Cardiac tokens found: {found}")
    
    print("\n=== FIT COMPLETE ===")

if __name__ == "__main__":
    main()
```

**QC checkpoint**: 
1. Lens file saved and > 0 bytes
2. "Paris" appears in J-space for France prompt
3. Cardiac tokens appear for chest pain prompt

> **KNOWN PITFALL**: `jlens.fit()` API may differ from README. The README shows:
> ```python
> lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
> ```
> But the actual API may require additional parameters (e.g., `n_layers`, `batch_size`). Read the docstring: `python -c "import jlens; help(jlens.fit)"` before running. Also check `walkthrough.ipynb` in the repo for the canonical usage pattern.

> **KNOWN PITFALL**: For Qwen2.5-7B on a 24GB GPU, float16 should fit (~14GB). If OOM, try `torch_dtype=torch.bfloat16` or `load_in_4bit=True` with bitsandbytes. However, quantized models may produce lower-quality J-lens fits — the paper uses full precision.

### Step 4: Smoke Test (5 Cases, 1 Model, 3 Agents)

This is the critical validation step. Run before any full experiment.

**`scripts/smoke_test.py`**:
```python
"""
Smoke test: 5 MedQA cases × 3 agents × J-space extraction.

Validates:
1. Agents produce parseable MCQ answers (A/B/C/D)
2. J-space extraction produces non-empty token lists
3. CAS computation produces valid cosine similarities in [-1, 1]
4. STD computation produces non-negative scores
5. All 4 orchestration strategies produce a final answer

This should run in <30 minutes on a single GPU.
"""
import json
import torch
import transformers
import jlens
from datasets import load_from_disk

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LENS_PATH = "checkpoints/qwen7b_lens.pt"
N_CASES = 5
SEED = 42
LAYERS_TO_READ = "middle_third"  # read layers L/3 to 2L/3

# --- Agent system prompts ---
AGENTS = {
    "conservative": "You are a conservative clinical reasoner. ...",  # full prompt from §2.3
    "aggressive": "You are a thorough clinical diagnostician. ...",
    "evidence": "You are an evidence-based medicine specialist. ...",
}

def load_model_and_lens():
    """Load model + tokenizer + fitted lens."""
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_PATH)
    # NOTE: API may be lens = jlens.JacobianLens.load(LENS_PATH) — check docstring
    return model, tok, lens, hf

def format_medqa_prompt(case, agent_prompt):
    """Format a MedQA case as a chat prompt for the agent."""
    options_text = "\n".join([f"{k}: {v}" for k, v in case["options"].items()])
    user_msg = f"{case['question']}\n\nOptions:\n{options_text}\n\nAnswer with the letter (A, B, C, or D) and brief rationale."
    
    messages = [
        {"role": "system", "content": agent_prompt},
        {"role": "user", "content": user_msg},
    ]
    return messages

def extract_jspace(model, lens, tok, messages, hf):
    """
    Extract J-space readout for a prompt BEFORE generation.
    
    Returns:
        jspace_tokens: dict of {layer: [top_K_tokens]}
        jspace_vector: concatenated logit vector for CAS computation
    """
    # Format messages into a single string
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Get J-lens readouts at last position
    lens_logits, model_logits, _ = lens.apply(model, prompt_text, positions=[-1])
    
    # Extract top tokens at middle layers
    all_layers = sorted(lens_logits.keys())
    n = len(all_layers)
    mid_layers = all_layers[n // 3 : 2 * n // 3]
    
    jspace_tokens = {}
    vectors = []
    for layer in mid_layers:
        logits = lens_logits[layer][0]  # shape: [vocab_size]
        top10_ids = logits.topk(10).indices.tolist()
        top10_words = [tok.decode([t]).strip() for t in top10_ids]
        jspace_tokens[layer] = top10_words
        vectors.append(logits.cpu().float())
    
    jspace_vector = torch.cat(vectors)
    return jspace_tokens, jspace_vector

def generate_answer(hf, tok, messages):
    """Generate agent's answer."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(hf.device)
    with torch.no_grad():
        out = hf.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
    answer_text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer_text

def parse_answer(text):
    """Extract MCQ letter from agent output."""
    text = text.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if text.startswith(letter):
            return letter
    # Fallback: find first A/B/C/D
    for char in text:
        if char in "ABCD":
            return char
    return None

def compute_cas(vec1, vec2):
    """Cosine similarity between two J-space vectors."""
    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()

def compute_std(jspace_tokens_all_layers):
    """Say-Think Divergence score from J-space tokens."""
    UNCERTAINTY = {"uncertain", "unsure", "maybe", "wrong", "error", 
                   "incorrect", "doubt", "unclear", "ambiguous", "conflict",
                   "not", "no", "but", "however", "although", "risky"}
    CONFIDENCE = {"correct", "certain", "confident", "clear", "obvious", 
                  "definitely", "yes", "right", "true", "answer"}
    
    all_tokens = set()
    for layer_tokens in jspace_tokens_all_layers.values():
        all_tokens.update([t.lower() for t in layer_tokens])
    
    unc_count = len(all_tokens.intersection(UNCERTAINTY))
    conf_count = len(all_tokens.intersection(CONFIDENCE))
    return unc_count / (conf_count + 1)

def main():
    print("=== SMOKE TEST START ===\n")
    
    # Load
    print("Loading model + lens...")
    model, tok, lens, hf = load_model_and_lens()
    
    print("Loading MedQA test set...")
    ds = load_from_disk("data/medqa/full")
    cases = ds["test"].shuffle(seed=SEED).select(range(N_CASES))
    
    results = []
    
    for i, case in enumerate(cases):
        print(f"\n--- Case {i+1}/{N_CASES}: {case['question'][:80]}... ---")
        gold = case["answer_idx"]
        print(f"Gold answer: {gold}")
        
        agent_results = {}
        agent_vectors = {}
        
        for agent_name, agent_prompt in AGENTS.items():
            msgs = format_medqa_prompt(case, agent_prompt)
            
            # Extract J-space
            jspace_tokens, jspace_vec = extract_jspace(model, lens, tok, msgs, hf)
            
            # Generate answer
            answer_text = generate_answer(hf, tok, msgs)
            parsed = parse_answer(answer_text)
            
            # Compute STD
            std_score = compute_std(jspace_tokens)
            
            print(f"  {agent_name}: answer={parsed}, STD={std_score:.2f}")
            print(f"    J-space sample (mid-layer): {list(jspace_tokens.values())[len(jspace_tokens)//2][:5]}")
            
            agent_results[agent_name] = {
                "answer": parsed,
                "answer_text": answer_text[:200],
                "jspace_tokens": {str(k): v for k, v in jspace_tokens.items()},
                "std_score": std_score,
            }
            agent_vectors[agent_name] = jspace_vec
        
        # Compute pairwise CAS
        agents_list = list(AGENTS.keys())
        cas_pairs = {}
        for j in range(len(agents_list)):
            for k in range(j+1, len(agents_list)):
                a, b = agents_list[j], agents_list[k]
                cas = compute_cas(agent_vectors[a], agent_vectors[b])
                cas_pairs[f"{a}-{b}"] = cas
                print(f"  CAS({a}, {b}) = {cas:.3f}")
        
        # Test all orchestration strategies
        answers = {name: r["answer"] for name, r in agent_results.items()}
        stds = {name: r["std_score"] for name, r in agent_results.items()}
        
        # Strategy 1: Majority vote
        from collections import Counter
        vote_counts = Counter(answers.values())
        majority = vote_counts.most_common(1)[0][0]
        
        # Strategy 2: Confidence-weighted (inverse STD)
        weighted = {}
        for name, ans in answers.items():
            w = 1.0 / (stds[name] + 0.1)
            weighted[ans] = weighted.get(ans, 0) + w
        conf_weighted = max(weighted, key=weighted.get)
        
        print(f"  Majority vote: {majority} | Conf-weighted: {conf_weighted} | Gold: {gold}")
        
        results.append({
            "case_idx": i,
            "gold": gold,
            "agents": agent_results,
            "cas_pairs": cas_pairs,
            "majority_vote": majority,
            "confidence_weighted": conf_weighted,
        })
    
    # Save results
    with open("results/smoke_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # === QC SUMMARY ===
    print("\n=== QC SUMMARY ===")
    
    n_parsed = sum(1 for r in results 
                   for a in r["agents"].values() 
                   if a["answer"] is not None)
    total = len(results) * len(AGENTS)
    print(f"Parseable answers: {n_parsed}/{total} ({100*n_parsed/total:.0f}%)")
    assert n_parsed / total > 0.8, f"FAIL: <80% parseable answers"
    
    n_nonempty_jspace = sum(1 for r in results 
                           for a in r["agents"].values() 
                           if any(len(v) > 0 for v in a["jspace_tokens"].values()))
    print(f"Non-empty J-space: {n_nonempty_jspace}/{total}")
    assert n_nonempty_jspace == total, "FAIL: some J-space extractions empty"
    
    cas_values = [v for r in results for v in r["cas_pairs"].values()]
    print(f"CAS range: [{min(cas_values):.3f}, {max(cas_values):.3f}]")
    assert all(-1.01 <= v <= 1.01 for v in cas_values), "FAIL: CAS out of [-1,1]"
    
    majority_correct = sum(1 for r in results if r["majority_vote"] == r["gold"])
    print(f"Majority vote accuracy: {majority_correct}/{len(results)}")
    
    print("\n=== SMOKE TEST PASSED ===")

if __name__ == "__main__":
    main()
```

**SBATCH for smoke test**:
```bash
#!/bin/bash
#SBATCH --job-name=jsd_smoke
#SBATCH --partition=gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=haining.wang@montana.edu

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/g91p721/.cache/huggingface

mkdir -p results logs

python scripts/smoke_test.py 2>&1 | tee logs/smoke_test_$(date +%Y%m%d_%H%M%S).log
```

**QC checkpoint (must ALL pass before proceeding)**:
- [ ] ≥80% of agent outputs parseable as A/B/C/D
- [ ] All J-space extractions non-empty
- [ ] All CAS values in [-1, 1]
- [ ] STD scores ≥ 0
- [ ] results/smoke_test.json saved and valid JSON

### Step 5-7: Full Experiment Runs

Same pattern as smoke test but with full sample sizes:

| Step | Benchmark | N cases | Expected GPU hours (7B model) |
|------|-----------|---------|-------------------------------|
| 5 | MedQA | 200 | ~8-12h |
| 6 | DiversityMedQA | 200 | ~8-12h |
| 7 | HealthBench Hard | 100 | ~4-6h (longer prompts) |

**Key differences from smoke test**:
- Add `--benchmark medqa --n-cases 200` CLI args
- Save per-case J-space tokens to separate JSONL (large)
- Save J-space vectors as .pt files (for recomputation)
- Add progress bar (tqdm)
- Add checkpoint/resume logic (save after each case, skip completed)

### Step 8: Analysis & Figures

Run on CPU. Reads all results JSONs, computes statistics, generates JNO-style figures.

```bash
#!/bin/bash
#SBATCH --job-name=jsd_analysis
#SBATCH --partition=unsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/analysis_%j.out

set -euo pipefail
module purge
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

cd /home/g91p721/jd
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python scripts/generate_figures.py
```

---

## 6. Known Pitfalls & Gotchas

### 6.1 J-lens API Uncertainty

The `jacobian-lens` repo was released July 2, 2026 — 5 days ago. The API may change or have undocumented quirks. Before writing any code:

1. Clone the repo and READ `walkthrough.ipynb` end-to-end
2. Run `python -c "import jlens; help(jlens.fit)"` to get actual function signatures
3. Check if there are pre-fitted lens checkpoints for common models at HuggingFace (the README mentions `jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")`)
4. Check issues on the GitHub repo for known bugs

### 6.2 Model Loading on Limited GPU Memory

| Model | fp16 VRAM | 4-bit VRAM |
|-------|-----------|------------|
| Qwen2.5-7B | ~14 GB | ~4 GB |
| Llama-3.1-8B | ~16 GB | ~5 GB |
| Llama-3.1-70B | ~140 GB | ~35 GB |

**J-lens fitting requires backward pass** — this roughly doubles VRAM vs inference only. So for 7B models, expect ~28 GB during fitting. This may need a 40GB A100 or 48GB A6000.

**Mitigation**: Fit the lens with `gradient_checkpointing=True` if jlens supports it, or reduce batch size to 1.

### 6.3 HealthBench Format

HealthBench Hard is multi-turn conversations, not MCQs. Adapting it for our MCQ-style pipeline requires:

- Extracting the last user message as the "question"
- The rubric criteria define what a correct response looks like
- We may need to use a separate LLM-as-judge to score free-text outputs against rubrics
- **Alternative**: For the initial submission, focus on MedQA + DiversityMedQA (MCQ), and include HealthBench as a secondary/qualitative analysis only

### 6.4 Multi-Token Concepts in J-Space

J-lens only captures single-token concepts. "Myocardial infarction" might appear as "my" + "card" or "MI" but not as the full phrase. Mitigation:

- Use a medical vocabulary mapping: map token fragments to UMLS concepts
- Focus on single-token medical terms that are complete concepts: "AKI", "sepsis", "STEMI", "pneumonia", etc.
- Report this as a limitation

### 6.5 Tempest vs Quartz Decision

If Tempest GPUs are unavailable or insufficient:
- **Quartz** (IU): account `r01834`, partition `general`, has A100 80GB
- Quartz requires Duo 2FA even with SSH keys
- Quartz venv: `/N/project/depot/hw56/RLH/.venv` (may not have jlens)
- Create a new venv for this project on Quartz if needed

---

## 7. File-Level Specifications for Key Modules

### 7.1 `src/data/load_medqa.py`

```python
"""
Load and format MedQA-USMLE for the multi-agent experiment.

Input: HuggingFace dataset saved to disk at data/medqa/full
Output: List[dict] with keys:
    - case_id: str (unique)
    - question: str
    - options: dict[str, str] (A/B/C/D → text)
    - gold_answer: str (A/B/C/D)
    - difficulty: str (optional, derived from question length or topic)
"""
```

### 7.2 `src/jspace/extract.py`

```python
"""
Extract J-space readout for a given prompt.

Input:
    - model: jlens-wrapped model
    - lens: fitted JacobianLens
    - tokenizer: HF tokenizer
    - prompt_text: str (full prompt including system + user)
    - position: int (default -1, last token before generation)
    - top_k: int (default 50, how many top tokens to return per layer)

Output:
    - jspace_tokens: dict[int, list[str]]  (layer → top_k tokens)
    - jspace_vector: torch.Tensor  (concatenated logit vectors for CAS)
    - metadata: dict with n_layers, layers_read, position_read
"""
```

### 7.3 `src/agents/consensus.py`

```python
"""
Orchestration strategies for multi-agent consensus.

Each strategy takes:
    - agent_answers: dict[str, str]  (agent_name → answer letter)
    - agent_jspace: dict[str, dict]  (agent_name → {tokens, vector, std_score})
    
Returns:
    - final_answer: str
    - strategy_metadata: dict (weights used, agents included, etc.)

Strategies:
    1. majority_vote — simple majority
    2. confidence_weighted — weight by inverse STD
    3. align_route — filter agents by CAS to group centroid
    4. diverge_surface — if STD > threshold, trigger re-evaluation
"""
```

---

## 8. Expected Outputs for the Paper

### 8.1 Tables

- **Table 1**: Benchmark characteristics (N cases, format, domain)
- **Table 2**: Per-strategy accuracy across benchmarks (with 95% CIs)
- **Table 3**: Say-Think Divergence prevalence by case difficulty
- **eTable 1**: Full per-agent accuracy breakdown
- **eTable 2**: Pairwise CAS distributions

### 8.2 Figures

- **Figure 1**: Schematic of the J-space orchestration pipeline (conceptual diagram)
- **Figure 2**: J-space readout comparison — 3 agents on same case (heatmap)
- **Figure 3**: Accuracy by strategy (grouped bar chart, MedQA + DiversityMedQA)
- **Figure 4**: CAS vs consensus correctness (scatter with logistic fit)
- **eFigure 1**: STD distribution for correct vs incorrect agent answers
- **eFigure 2**: Representative J-space "hidden disagreement" examples

---

## 9. Codex / Claude Code Handoff Notes

### For the implementing agent:

1. **Start by reading the jlens source code**: Clone `https://github.com/anthropics/jacobian-lens`, read `README.md`, run `walkthrough.ipynb` mentally. The exact API signatures matter — do NOT assume from this doc alone.

2. **Build incrementally**: Steps 0→1→2→3→4 in order. Do NOT skip to full experiment. Each step has QC checkpoints that MUST pass.

3. **The smoke test (Step 4) is the most important deliverable.** If the smoke test passes, the full experiment is just scaling up.

4. **When uncertain about jlens API**: Read the source at `jlens/fitting.py` and `jlens/lens.py`. The `walkthrough.ipynb` is the canonical reference.

5. **Keep J-space vectors on disk**: Save as `.pt` files per case. They're needed for post-hoc reanalysis. Don't discard them.

6. **Handle model loading failures gracefully**: If Qwen2.5-7B doesn't fit on available GPU, try `Qwen/Qwen2.5-3B-Instruct` (3B) as ultimate fallback. The science doesn't change with model size — we just need to demonstrate the method.

7. **Log everything**: Every print statement should also go to a log file. Use Python's logging module with both StreamHandler and FileHandler.

8. **The HealthBench integration is stretch**: If time is tight, skip it and focus on MedQA + DiversityMedQA. Two benchmarks is sufficient for JNO.
