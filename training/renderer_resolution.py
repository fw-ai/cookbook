"""Renderer-name resolution shared by cookbook rendering entrypoints."""

from __future__ import annotations


def _tinker_recommended_renderer_name(tokenizer_model: str) -> str:
    from tinker_cookbook.model_info import get_recommended_renderer_name

    return get_recommended_renderer_name(tokenizer_model)


def resolve_renderer_name(
    tokenizer_model: str,
    renderer_name: str = "",
) -> str:
    """Choose the renderer used for message -> token rendering."""
    if renderer_name:
        return renderer_name
    normalized_model_name = tokenizer_model.lower()
    if "moonshotai/kimi-k2.5" in normalized_model_name:
        return "kimi_k25"
    if "moonshotai/kimi-k2.6" in normalized_model_name:
        return "kimi_k25"
    if "moonshotai/kimi-k2.7-code" in normalized_model_name:
        return "kimi_k27_code"
    if "nemotron" in normalized_model_name:
        return "nemotron"
    if "minimax-m2" in normalized_model_name or "minimax_m2" in normalized_model_name:
        return "minimax_m2"
    if "qwen3-vl" in normalized_model_name:
        return "qwen3_vl_instruct"
    if "qwen3.6" in normalized_model_name or "qwen3_6" in normalized_model_name:
        return "qwen3_6"
    if "qwen3.5" in normalized_model_name or "qwen3_5" in normalized_model_name:
        return "qwen3_5"
    if "gemma-4" in normalized_model_name or "gemma4" in normalized_model_name:
        return "gemma4"
    if (
        "deepseek-v4" in normalized_model_name
        or "deepseek_v4" in normalized_model_name
        or "deepseekv4" in normalized_model_name
    ):
        return "deepseek_v4"
    if (
        "glm-5p1" in normalized_model_name
        or "glm-5.1" in normalized_model_name
        or "glm5" in normalized_model_name
    ):
        return "glm5"
    if (
        normalized_model_name.startswith("mistralai/")
        or "ministral-" in normalized_model_name
        or "mistral-7b-instruct-v0.3" in normalized_model_name
        or "mistral-7b-instruct-v0.4" in normalized_model_name
        or "mistral-small" in normalized_model_name
    ):
        return "mistral"
    try:
        return _tinker_recommended_renderer_name(tokenizer_model)
    except Exception as exc:  # pragma: no cover - message only
        raise ValueError(
            f"Could not infer a renderer for tokenizer_model={tokenizer_model!r}. "
            "Set Config.renderer_name explicitly."
        ) from exc
