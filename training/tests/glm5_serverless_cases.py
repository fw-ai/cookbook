"""Recorded Fireworks serverless GLM5 prompt-token fixtures."""

from __future__ import annotations

from typing import Any


GLM5_SERVERLESS_STOP_TOKEN_IDS = {
    "user": 154827,
    "observation": 154829,
}


GLM5_SERVERLESS_PROMPT_TOKEN_CASES: list[dict[str, Any]] = [{'name': 'single_user', 'messages': [{'role': 'user', 'content': 'Hello'}], 'request_kwargs': {}, 'prompt_token_ids': [154822, 154824, 154827, 9703, 154828, 154841]}]
