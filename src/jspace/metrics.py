"""
J-space derived metrics for multi-agent orchestration.

CAS — Cognitive Alignment Score: cosine similarity between two agents' J-space vectors.
STD — Say-Think Divergence: ratio of uncertainty vs confidence tokens in J-space.
"""
import torch
import yaml
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load token lists from config; fall back to defaults
_CFG = None


def _get_config():
    global _CFG
    if _CFG is None:
        cfg_path = os.path.join(ROOT, "configs", "experiment.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                _CFG = yaml.safe_load(f)
        else:
            _CFG = {}
    return _CFG


def _get_token_sets():
    cfg = _get_config()
    js = cfg.get("jspace", {})
    uncertainty = set(js.get("uncertainty_tokens", [
        "uncertain", "unsure", "maybe", "wrong", "error", "incorrect",
        "doubt", "unclear", "ambiguous", "conflict", "not", "but",
        "however", "risky", "unlikely", "questionable",
    ]))
    confidence = set(js.get("confidence_tokens", [
        "correct", "certain", "confident", "clear", "obvious",
        "definitely", "yes", "right", "true", "answer", "likely",
        "consistent",
    ]))
    return uncertainty, confidence


def compute_cas(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Cognitive Alignment Score: cosine similarity between two J-space vectors.

    Returns float in [-1, 1]. Higher = agents thinking more similarly.
    """
    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()


def compute_std(jspace_tokens: dict[int, list[str]]) -> float:
    """
    Say-Think Divergence score.

    Counts uncertainty vs confidence tokens across all J-space layers.
    High STD on a case where the agent gave a definitive answer = hidden disagreement.

    Args:
        jspace_tokens: dict of layer → list of top-K token strings

    Returns:
        float >= 0. Higher = more internal uncertainty.
    """
    uncertainty_set, confidence_set = _get_token_sets()

    all_tokens = set()
    for layer_tokens in jspace_tokens.values():
        all_tokens.update(t.lower() for t in layer_tokens)

    unc_count = len(all_tokens.intersection(uncertainty_set))
    conf_count = len(all_tokens.intersection(confidence_set))
    return unc_count / (conf_count + 1)


def pairwise_cas(agent_vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Compute pairwise CAS for all agent pairs.

    Args:
        agent_vectors: dict of agent_name → J-space vector

    Returns:
        dict of "agentA-agentB" → CAS score
    """
    names = sorted(agent_vectors.keys())
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairs[f"{a}-{b}"] = compute_cas(agent_vectors[a], agent_vectors[b])
    return pairs
