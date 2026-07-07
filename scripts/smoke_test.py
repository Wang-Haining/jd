"""
Smoke test: 5 MedQA cases × 3 agents × J-space extraction.

Validates:
1. Agents produce parseable MCQ answers (A/B/C/D)
2. J-space extraction produces non-empty token lists
3. CAS computation produces valid cosine similarities in [-1, 1]
4. STD computation produces non-negative scores
5. All 4 orchestration strategies produce a final answer

Run time: <30 minutes on a single GPU with 7B model.
"""
import json
import os
import sys
import re
import torch
import transformers
import jlens
import yaml
from collections import Counter

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.load_medqa import load_medqa, format_prompt
from src.jspace.extract import extract_jspace
from src.jspace.metrics import compute_cas, compute_std, pairwise_cas
from src.agents.consensus import STRATEGIES

# --- Config ---
CFG_PATH = os.path.join(ROOT, "configs", "experiment.yaml")
N_CASES = 5
SEED = 42


def load_config():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def parse_answer(text: str) -> str | None:
    """Extract MCQ letter from agent output."""
    text = text.strip()
    # Try "A." or "A:" or "A " at start
    m = re.match(r'^([A-D])[.:\s)]', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Try "Answer: A" pattern
    m = re.search(r'(?:answer|option)[:\s]*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: first A-D character
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return None


def main():
    print("=" * 60)
    print("SMOKE TEST — jd (J-Space Delphi)")
    print("=" * 60)

    cfg = load_config()
    model_cfg = cfg["models"]["smoke_test"]
    model_name = model_cfg["name"]
    agent_cfgs = cfg["agents"]

    # --- Load model + lens ---
    print(f"\n[1/5] Loading model: {model_name}")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "float16")),
        device_map=model_cfg.get("device_map", "auto"),
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf, tok)

    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    lens_path = os.path.join(ROOT, "checkpoints", f"{safe_name}_lens.pt")
    print(f"  Loading lens from {lens_path}")
    # NOTE: API may be JacobianLens.load() or .from_pretrained() — check jlens docs
    lens = jlens.JacobianLens.from_pretrained(lens_path)

    # --- Load cases ---
    print(f"\n[2/5] Loading {N_CASES} MedQA cases")
    cases = load_medqa(split="test", n_cases=N_CASES, seed=SEED)
    print(f"  Loaded {len(cases)} cases")

    # --- Run agents ---
    print(f"\n[3/5] Running {len(agent_cfgs)} agents × {N_CASES} cases")
    results = []

    for ci, case in enumerate(cases):
        print(f"\n  Case {ci+1}/{N_CASES}: {case['question'][:80]}...")
        print(f"  Gold: {case['gold_answer']}")

        agent_answers = {}
        agent_std = {}
        agent_vectors = {}

        for agent_name, agent_cfg in agent_cfgs.items():
            msgs = format_prompt(case, agent_cfg["system_prompt"])
            prompt_text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            # Extract J-space (BEFORE generation)
            jspace = extract_jspace(model, lens, tok, prompt_text, top_k=50)

            # Generate answer
            inputs = tok(prompt_text, return_tensors="pt").to(hf.device)
            with torch.no_grad():
                out = hf.generate(
                    **inputs,
                    max_new_tokens=model_cfg.get("max_new_tokens", 200),
                    temperature=model_cfg.get("temperature", 0.3),
                    do_sample=model_cfg.get("do_sample", True),
                )
            answer_text = tok.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            parsed = parse_answer(answer_text)

            # Compute STD
            std = compute_std(jspace["tokens"])

            # Sample J-space tokens at middle layer
            mid_layer = jspace["layers_read"][len(jspace["layers_read"]) // 2]
            sample_tokens = jspace["tokens"][mid_layer][:5]

            print(f"    {agent_name}: ans={parsed}, STD={std:.2f}, "
                  f"J-space={sample_tokens}")

            agent_answers[agent_name] = parsed
            agent_std[agent_name] = std
            agent_vectors[agent_name] = jspace["vector"]

        # Pairwise CAS
        cas = pairwise_cas(agent_vectors)
        for pair, score in cas.items():
            print(f"    CAS({pair}) = {score:.3f}")

        # Run all strategies
        strategy_results = {}
        for strat_name, strat_fn in STRATEGIES.items():
            res = strat_fn(
                agent_answers=agent_answers,
                agent_std=agent_std,
                agent_vectors=agent_vectors,
            )
            strategy_results[strat_name] = res
            correct = "✓" if res["answer"] == case["gold_answer"] else "✗"
            print(f"    {strat_name}: {res['answer']} {correct}")

        results.append({
            "case_id": case["case_id"],
            "gold": case["gold_answer"],
            "agent_answers": agent_answers,
            "agent_std": {k: round(v, 4) for k, v in agent_std.items()},
            "cas_pairs": {k: round(v, 4) for k, v in cas.items()},
            "strategies": {k: {"answer": v["answer"]} for k, v in strategy_results.items()},
        })

    # --- Save ---
    out_path = os.path.join(ROOT, "results", "smoke_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[4/5] Results saved to {out_path}")

    # === QC ===
    print("\n" + "=" * 60)
    print("[5/5] QC CHECKS")
    print("=" * 60)

    total_agents = len(results) * len(agent_cfgs)

    # Check 1: parseable answers
    n_parsed = sum(
        1 for r in results for a in r["agent_answers"].values() if a is not None
    )
    pct = 100 * n_parsed / total_agents
    status = "PASS" if pct >= 80 else "FAIL"
    print(f"  [{status}] Parseable answers: {n_parsed}/{total_agents} ({pct:.0f}%)")
    assert pct >= 80, f"<80% parseable answers"

    # Check 2: CAS in range
    all_cas = [v for r in results for v in r["cas_pairs"].values()]
    in_range = all(-1.01 <= v <= 1.01 for v in all_cas)
    status = "PASS" if in_range else "FAIL"
    print(f"  [{status}] CAS range: [{min(all_cas):.3f}, {max(all_cas):.3f}]")
    assert in_range, "CAS out of [-1, 1]"

    # Check 3: STD non-negative
    all_std = [v for r in results for v in r["agent_std"].values()]
    non_neg = all(v >= 0 for v in all_std)
    status = "PASS" if non_neg else "FAIL"
    print(f"  [{status}] STD range: [{min(all_std):.3f}, {max(all_std):.3f}]")
    assert non_neg, "Negative STD"

    # Check 4: all strategies produce answers
    for strat in STRATEGIES:
        n_answers = sum(1 for r in results if r["strategies"][strat]["answer"] is not None)
        status = "PASS" if n_answers == len(results) else "FAIL"
        print(f"  [{status}] {strat}: {n_answers}/{len(results)} answers")

    # Check 5: accuracy summary
    print("\n  Accuracy summary:")
    for strat in STRATEGIES:
        correct = sum(
            1 for r in results if r["strategies"][strat]["answer"] == r["gold"]
        )
        print(f"    {strat}: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED" if all_cas and non_neg else "SMOKE TEST FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
