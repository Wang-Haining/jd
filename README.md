# jd — Cognitive Orchestration of Clinical AI Agents via Jacobian Lens

**Target**: JAMA Network Open  
**Status**: Design phase  
**HPC**: MSU Tempest / IU Quartz (fallback)

## Thesis

J-lens next-speaker routing may improve capped freeform clinical AI debate.

## Method

Apply Anthropic's [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
to a fixed multi-clinician HealthBench debate. At each discussion step, score
eligible next clinicians from their J-space on the next-turn prompt, then route
the chat to the highest-scoring clinician. Compare this with round-robin routing
and a current-agent handoff baseline.

## v1 HealthBench Hard Target

The JAMA Network Open-facing v1 focuses on HealthBench Hard as the single
realistic clinical corpus. The locked pilot design runs 100 HealthBench Hard
cases with Qwen/Qwen2.5-7B-Instruct, deterministic generation, no more than five
clinician agents, a fixed discussion-round cap, and four strategy arms:

- `single_neutral`
- `debate_round_robin`
- `debate_agent_handoff`
- `debate_jlens_next`

Rubrics are never passed to generators. They are used only during post hoc
HealthBench scoring.

This is not a Delphi or prompt/topology optimization study. The intervention is
only the next-speaker routing policy inside a capped freeform discussion.

## Quick Start

```bash
# On Tempest
cd /home/g91p721/jd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Check GPU
sinfo -o "%P %G"

# Run original MedQA smoke test (5 cases, ~30 min on 1 GPU)
python scripts/smoke_test.py

# Run HealthBench Hard smoke test after fitting the Qwen J-lens
python scripts/run_healthbench.py --n-cases 3 --seed 42 --run-name smoke_n3_seed42
```

## Pipeline

```
Step 0: Check GPU  →  Step 1: Setup env  →  Step 2: Download data
→  Step 3: Fit J-lens  →  Step 4: MedQA smoke test
→  Step 5: HealthBench smoke/pilot/study  →  Step 6: Score
→  Step 7: Analysis + figures
```

See [DESIGN_DOC.md](DESIGN_DOC.md) for full specification.

## Key References

- Gurnee et al. (2026). "Verbalizable Representations Form a Global Workspace in Language Models." [Paper](http://transformer-circuits.pub/2026/workspace/index.html) | [Code](https://github.com/anthropics/jacobian-lens)
- Menz et al. (2024). "Gender Representation of Health Care Professionals in LLM-Generated Stories." JAMA Netw Open. [Link](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2823876)
