"""
Extract J-space readout for a given prompt.

Returns top-K tokens at middle-third layers (where J-space lives)
and a concatenated logit vector for computing Cognitive Alignment Score.
"""
import torch
from typing import Optional


def get_middle_layers(all_layers: list[int]) -> list[int]:
    """Return layers in the middle third (where J-space is most active)."""
    n = len(all_layers)
    return all_layers[n // 3 : 2 * n // 3]


def extract_jspace(
    model,
    lens,
    tokenizer,
    prompt_text: str,
    position: int = -1,
    top_k: int = 50,
) -> dict:
    """
    Extract J-space readout for a prompt.

    Args:
        model: jlens-wrapped model (from jlens.from_hf)
        lens: fitted JacobianLens
        tokenizer: HuggingFace tokenizer
        prompt_text: full prompt string
        position: token position to read (-1 = last)
        top_k: number of top tokens per layer

    Returns:
        dict with keys:
            tokens: dict[int, list[str]]  — layer → top-K token strings
            token_ids: dict[int, list[int]] — layer → top-K token ids
            vector: torch.Tensor — concatenated logit vector (for CAS)
            layers_read: list[int]
    """
    lens_logits, model_logits, _ = lens.apply(
        model, prompt_text, positions=[position]
    )

    all_layers = sorted(lens_logits.keys())
    mid_layers = get_middle_layers(all_layers)

    tokens = {}
    token_ids = {}
    vectors = []

    for layer in mid_layers:
        logits = lens_logits[layer][0]  # shape: [vocab_size]
        topk_result = logits.topk(top_k)
        ids = topk_result.indices.tolist()
        words = [tokenizer.decode([t]).strip() for t in ids]

        tokens[layer] = words
        token_ids[layer] = ids
        vectors.append(logits.cpu().float())

    jspace_vector = torch.cat(vectors)

    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "vector": jspace_vector,
        "layers_read": mid_layers,
    }
