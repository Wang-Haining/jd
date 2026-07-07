"""
Orchestration strategies for multi-agent clinical consensus.

Each strategy takes agent answers + J-space signals and returns a final answer.
"""
from collections import Counter
import torch
from ..jspace.metrics import compute_cas


def majority_vote(agent_answers: dict[str, str], **kwargs) -> dict:
    """Simple majority vote on agent outputs."""
    counts = Counter(v for v in agent_answers.values() if v is not None)
    if not counts:
        return {"answer": None, "method": "majority_vote", "detail": "no valid answers"}
    winner = counts.most_common(1)[0][0]
    return {"answer": winner, "method": "majority_vote", "detail": dict(counts)}


def confidence_weighted(
    agent_answers: dict[str, str],
    agent_std: dict[str, float],
    **kwargs,
) -> dict:
    """Weight each agent's vote by inverse STD (low divergence = more weight)."""
    weighted = {}
    weights_used = {}
    for name, ans in agent_answers.items():
        if ans is None:
            continue
        w = 1.0 / (agent_std.get(name, 0) + 0.1)
        weighted[ans] = weighted.get(ans, 0) + w
        weights_used[name] = round(w, 3)
    if not weighted:
        return {"answer": None, "method": "confidence_weighted", "detail": "no valid answers"}
    winner = max(weighted, key=weighted.get)
    return {"answer": winner, "method": "confidence_weighted", "detail": weights_used}


def align_route(
    agent_answers: dict[str, str],
    agent_vectors: dict[str, torch.Tensor],
    cas_percentile: float = 50,
    **kwargs,
) -> dict:
    """
    Only count votes from agents whose CAS with the group centroid
    is above the given percentile. Filters out "distracted" agents.
    """
    if len(agent_vectors) < 2:
        return majority_vote(agent_answers)

    # Compute centroid
    vecs = list(agent_vectors.values())
    centroid = torch.stack(vecs).mean(dim=0)

    # CAS to centroid for each agent
    cas_scores = {}
    for name, vec in agent_vectors.items():
        cas_scores[name] = compute_cas(vec, centroid)

    # Filter by percentile
    import numpy as np
    threshold = np.percentile(list(cas_scores.values()), cas_percentile)
    included = {name for name, score in cas_scores.items() if score >= threshold}

    filtered_answers = {k: v for k, v in agent_answers.items() if k in included and v is not None}
    if not filtered_answers:
        return majority_vote(agent_answers)

    counts = Counter(filtered_answers.values())
    winner = counts.most_common(1)[0][0]
    return {
        "answer": winner,
        "method": "align_route",
        "detail": {"cas_scores": {k: round(v, 3) for k, v in cas_scores.items()},
                   "threshold": round(threshold, 3),
                   "included": list(included)},
    }


def diverge_surface(
    agent_answers: dict[str, str],
    agent_std: dict[str, float],
    std_threshold: float = 1.5,
    **kwargs,
) -> dict:
    """
    If any agent has STD > threshold, flag it as divergent.
    In a full implementation, this would trigger a second round of debate.
    For now, we exclude the divergent agent and re-vote.
    """
    divergent = {name for name, std in agent_std.items() if std > std_threshold}
    if not divergent:
        return {**majority_vote(agent_answers), "method": "diverge_surface",
                "detail": {"divergent_agents": [], "action": "none"}}

    filtered = {k: v for k, v in agent_answers.items() if k not in divergent and v is not None}
    if not filtered:
        return majority_vote(agent_answers)

    counts = Counter(filtered.values())
    winner = counts.most_common(1)[0][0]
    return {
        "answer": winner,
        "method": "diverge_surface",
        "detail": {"divergent_agents": list(divergent),
                   "action": "excluded_and_revoted"},
    }


STRATEGIES = {
    "majority_vote": majority_vote,
    "confidence_weighted": confidence_weighted,
    "align_route": align_route,
    "diverge_surface": diverge_surface,
}
