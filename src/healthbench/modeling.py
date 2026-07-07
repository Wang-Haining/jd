"""Model loading, generation, and J-lens helpers for HealthBench runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def safe_model_name(model_name: str) -> str:
    """Convert a HuggingFace model name to the local checkpoint stem."""
    return model_name.replace("/", "_").replace("-", "_").lower()


def default_lens_path(root: str | Path, model_name: str) -> Path:
    """Return the default fitted lens path for a model."""
    return Path(root) / "checkpoints" / f"{safe_model_name(model_name)}_lens.pt"


def torch_dtype_from_name(dtype_name: str):
    """Resolve a torch dtype name lazily."""
    import torch

    return getattr(torch, dtype_name)


def load_model_tokenizer(model_cfg: dict[str, Any]):
    """Load a HuggingFace causal LM and tokenizer."""
    import transformers

    dtype = torch_dtype_from_name(model_cfg.get("dtype", "float16"))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_jlens_model(hf_model, tokenizer):
    """Wrap a HuggingFace model with jlens."""
    import jlens

    return jlens.from_hf(hf_model, tokenizer)


def load_lens(lens_path: str | Path):
    """Load a fitted Jacobian Lens across known jlens API variants."""
    import jlens

    path = str(lens_path)
    cls = jlens.JacobianLens
    if hasattr(cls, "from_pretrained"):
        try:
            return cls.from_pretrained(path)
        except TypeError:
            return cls.from_pretrained(Path(path).parent, filename=Path(path).name)
    if hasattr(cls, "load"):
        return cls.load(path)
    raise AttributeError("Could not find a supported JacobianLens loader")


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render messages into a model prompt."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def generate_text(hf_model, tokenizer, messages: list[dict[str, str]], model_cfg: dict[str, Any]) -> str:
    """Generate a deterministic assistant response."""
    import torch

    prompt_text = render_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(hf_model.device)
    generation_kwargs = {
        "max_new_tokens": model_cfg.get("max_new_tokens", 800),
        "do_sample": bool(model_cfg.get("do_sample", False)),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(model_cfg.get("temperature", 0.3))

    with torch.no_grad():
        out = hf_model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
