"""
Fit a Jacobian Lens for the target model.

Steps:
1. Load model (default: Qwen2.5-7B-Instruct)
2. Load fit corpus (200 C4/wikitext sequences)
3. Fit lens (~100 sequences sufficient per paper §9.3)
4. Save lens checkpoint
5. Validate: apply to known prompts and check top tokens

Expected runtime: ~1-2 hours on a single A100 for 7B model.

References:
    - https://github.com/anthropics/jacobian-lens
    - walkthrough.ipynb in the jacobian-lens repo
"""
import json
import os
import sys
import torch
import transformers
import jlens
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config():
    cfg_path = os.path.join(ROOT, "configs", "experiment.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    model_cfg = cfg["models"]["smoke_test"]
    model_name = model_cfg["name"]
    fit_corpus_path = os.path.join(ROOT, cfg["jspace"]["fit_corpus"])
    lens_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(lens_dir, exist_ok=True)

    # Derive lens filename from model name
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    lens_path = os.path.join(lens_dir, f"{safe_name}_lens.pt")

    print(f"Model:      {model_name}")
    print(f"Fit corpus: {fit_corpus_path}")
    print(f"Lens out:   {lens_path}")
    print()

    # --- Load model ---
    print(f"Loading model: {model_name}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "float16")),
        device_map=model_cfg.get("device_map", "auto"),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    print(f"  Loaded. Parameters: {sum(p.numel() for p in hf_model.parameters()) / 1e9:.1f}B")

    # --- Load fit corpus ---
    print(f"Loading fit corpus from {fit_corpus_path}")
    with open(fit_corpus_path) as f:
        prompts = json.load(f)
    n_fit = cfg["jspace"].get("fit_n_prompts", 100)
    fit_prompts = prompts[:n_fit]
    print(f"  {len(fit_prompts)} sequences for fitting")

    # --- Fit ---
    # NOTE: The exact API may differ. If jlens.fit() fails, check:
    #   python -c "import jlens; help(jlens.fit)"
    # The walkthrough.ipynb in the repo is the canonical reference.
    print("Fitting J-lens... (this may take 1-2 hours)")
    lens = jlens.fit(
        model,
        prompts=fit_prompts,
        checkpoint_path=lens_path,
    )
    lens.save(lens_path)
    file_size = os.path.getsize(lens_path) / 1e6
    print(f"Lens saved to {lens_path} ({file_size:.1f} MB)")

    # --- Validation ---
    print("\n=== VALIDATION ===")

    # Test 1: Known factual retrieval
    test1 = "The capital city of France is"
    lens_logits, model_logits, _ = lens.apply(model, test1, positions=[-1])
    all_layers = sorted(lens_logits.keys())
    mid = all_layers[len(all_layers) // 2]
    top10 = [tokenizer.decode([t]).strip().lower() for t in lens_logits[mid][0].topk(10).indices]
    print(f"  Test 1 (France capital), mid-layer top-10: {top10}")
    if any("paris" in t for t in top10):
        print("  ✓ Paris found")
    else:
        print("  ⚠ Paris NOT in top-10 — lens may be low quality, but continuing")

    # Test 2: Medical reasoning
    test2 = (
        "A 55-year-old male presents with crushing chest pain radiating to the "
        "left arm, diaphoresis, and ST elevation on ECG. The most likely diagnosis is"
    )
    lens_logits2, _, _ = lens.apply(model, test2, positions=[-1])
    mid2 = sorted(lens_logits2.keys())[len(lens_logits2) // 2]
    top20 = [tokenizer.decode([t]).strip().lower() for t in lens_logits2[mid2][0].topk(20).indices]
    cardiac = {"heart", "cardiac", "myocardial", "infarction", "mi", "acs", "stemi", "coronary"}
    found = cardiac.intersection(set(top20))
    print(f"  Test 2 (chest pain), mid-layer top-20: {top20}")
    print(f"  Cardiac tokens found: {found}")

    print("\n=== FIT COMPLETE ===")


if __name__ == "__main__":
    main()
