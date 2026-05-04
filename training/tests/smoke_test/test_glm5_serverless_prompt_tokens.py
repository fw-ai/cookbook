"""GLM5 renderer parity against Fireworks serverless prompt tokenization.

Run with:
    FIREWORKS_API_KEY=... pytest training/tests/smoke_test/test_glm5_serverless_prompt_tokens.py -q

This is deliberately a remote smoke test, not a unit test: it calls Fireworks
serverless with ``echo=True``/``raw_output=True`` to use the service's prompt
token IDs as the ground truth for renderer prompt construction.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
import transformers
from fireworks import Fireworks

from training.renderer.glm5 import GLM5Renderer
from training.tests.glm5_serverless_cases import GLM5_SERVERLESS_PROMPT_TOKEN_CASES
from training.utils.supervised import normalize_messages


_DEFAULT_MODEL_ID = "accounts/fireworks/models/glm-5p1"
_DEFAULT_TOKENIZER_MODEL = "zai-org/GLM-5.1"


def _serverless_prompt_token_ids(
    client: Fireworks,
    *,
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1,
                temperature=0,
                echo=True,
                raw_output=True,
                return_token_ids=True,
                **kwargs,
            )
            payload = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )
            prompt_ids = payload.get("prompt_token_ids")
            if not prompt_ids:
                choices = payload.get("choices") or []
                raw_output = choices[0].get("raw_output") if choices else {}
                prompt_ids = (raw_output or {}).get("prompt_token_ids")
            assert prompt_ids, f"Fireworks response missing prompt_token_ids: {payload}"
            return [int(token_id) for token_id in prompt_ids]
        except Exception as exc:  # noqa: BLE001 - smoke retry for transient hot-load errors
            last_exc = exc
            if attempt == 2:
                break
            time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc


def _renderer_prompt_token_ids(
    renderer: GLM5Renderer,
    messages: list[dict[str, Any]],
) -> list[int]:
    model_input = renderer.build_generation_prompt(
        normalize_messages(messages),
        role="assistant",
    )
    return list(model_input.to_ints())


@pytest.mark.e2e
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "case",
    GLM5_SERVERLESS_PROMPT_TOKEN_CASES,
    ids=[case["name"] for case in GLM5_SERVERLESS_PROMPT_TOKEN_CASES],
)
def test_glm5_renderer_matches_fireworks_serverless_prompt_token_ids(
    case: dict[str, Any],
):
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set")

    model = os.environ.get("FIREWORKS_GLM5_SERVERLESS_MODEL", _DEFAULT_MODEL_ID)
    tokenizer_model = os.environ.get(
        "FIREWORKS_GLM5_TOKENIZER_MODEL",
        _DEFAULT_TOKENIZER_MODEL,
    )
    base_url = os.environ.get("FIREWORKS_SERVERLESS_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = Fireworks(**client_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
    )
    # Mirror the serverless render mode: when the case asks for
    # ``reasoning_history=preserved``, build a non-stripping renderer so
    # token sequences line up (registered default strips history thinking
    # to match HF ``apply_chat_template`` with ``clear_thinking`` unset).
    preserves_history = (
        case.get("request_kwargs", {}).get("reasoning_history") == "preserved"
    )
    renderer = GLM5Renderer(
        tokenizer,
        strip_thinking_from_history=not preserves_history,
    )

    server_ids = _serverless_prompt_token_ids(
        client,
        model=model,
        messages=case["messages"],
        **case["request_kwargs"],
    )
    recorded_ids = [int(token_id) for token_id in case["prompt_token_ids"]]
    renderer_ids = _renderer_prompt_token_ids(renderer, case["messages"])

    assert server_ids == recorded_ids, (
        f"{case['name']} Fireworks serverless prompt tokens changed.\n"
        f"recorded: {recorded_ids}\n"
        f"server:   {server_ids}"
    )
    assert renderer_ids == server_ids, (
        f"{case['name']} GLM5 prompt tokens differ from Fireworks serverless.\n"
        f"server:   {tokenizer.decode(server_ids)!r}\n"
        f"renderer: {tokenizer.decode(renderer_ids)!r}"
    )
