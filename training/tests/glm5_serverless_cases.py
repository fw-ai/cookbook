"""Recorded Fireworks serverless GLM5 prompt-token fixtures."""

from __future__ import annotations

from typing import Any


GLM5_SERVERLESS_PROMPT_TOKEN_CASES: list[dict[str, Any]] = [
    {
        "name": "single_user",
        "messages": [{"role": "user", "content": "Hello"}],
        "request_kwargs": {},
        "prompt_token_ids": [154822, 154824, 154827, 9703, 154828, 154841],
    },
    {
        "name": "preserved_history_reasoning",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "HIST_THINK_A",
                "content": "4",
            },
            {"role": "user", "content": "Now what is 3+3?"},
        ],
        "request_kwargs": {"reasoning_history": "preserved"},
        "prompt_token_ids": [
            154822,
            154824,
            154827,
            3838,
            374,
            220,
            17,
            10,
            17,
            30,
            154828,
            154841,
            39,
            3846,
            10620,
            11631,
            1566,
            154842,
            19,
            154827,
            7039,
            1128,
            374,
            220,
            18,
            10,
            18,
            30,
            154828,
            154841,
        ],
    },
    {
        "name": "tool_response_handoff",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "reasoning_content": "need weather",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        "request_kwargs": {"reasoning_history": "preserved"},
        "prompt_token_ids": [
            154822,
            154824,
            154827,
            15194,
            30,
            154828,
            154841,
            17027,
            9101,
            154842,
            154843,
            455,
            68852,
            154847,
            8923,
            154848,
            154849,
            28565,
            154850,
            154844,
            154829,
            154845,
            82,
            27208,
            154846,
            154828,
            154841,
        ],
    },
]
