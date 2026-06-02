#!/usr/bin/env python3
"""Example multi-turn remote rollout server for RemoteRolloutProcessor.

The server implements the minimal contract expected by Eval Protocol:

* ``POST /init`` starts a rollout asynchronously.
* the worker calls the model through ``req.model_base_url`` so the tracing
  gateway can tag every model request with rollout metadata.
* when the episode is done, the worker logs ``Status.rollout_finished()``.

This is an example environment, not a production server.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from functools import lru_cache
from typing import Any

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from openai.types.chat import ChatCompletion

from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status
from eval_protocol.models import Message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger().addHandler(FireworksTracingHttpHandler())

app = FastAPI()

RETRY_PROMPT = (
    "Your answer may be wrong or not parsable. Please check the original "
    "problem and try again. Put the final numeric answer in \\boxed{}."
)

# Not accepted by chat.completions.create; rollout.py passes them for tracing/server use.
_NON_CHAT_PARAM_KEYS = ("base_url", "max_seq_len", "http_timeout")


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)


def _message_to_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, Message):
        return message.dump_mdoel_for_chat_completion_request()
    return {k: v for k, v in dict(message).items() if v is not None}


def _render_prompt_ids(
    *,
    tokenizer_model: str | None,
    messages: list[dict[str, Any]],
) -> list[int]:
    if not tokenizer_model:
        return []
    tokenizer = _load_tokenizer(tokenizer_model)
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids") or []
    elif hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    return [int(token_id) for token_id in token_ids]


def _extract_completion_trace(response: ChatCompletion) -> tuple[str, list[int], list[float], str]:
    payload = response.model_dump()
    choices = payload.get("choices") or []
    if not choices:
        return "", [], [], "unknown"

    choice = choices[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    finish_reason = str(choice.get("finish_reason") or "stop")

    completion_token_ids: list[int] = []
    completion_logprobs: list[float] = []
    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None
    if isinstance(content_logprobs, list):
        for entry in content_logprobs:
            if not isinstance(entry, dict):
                continue
            token_id = entry.get("token_id")
            if token_id is not None:
                completion_token_ids.append(int(token_id))
            completion_logprobs.append(float(entry.get("logprob") or 0.0))

    raw_output = choice.get("raw_output") if isinstance(choice.get("raw_output"), dict) else {}
    if not completion_token_ids:
        completion_token_ids = [int(x) for x in raw_output.get("completion_token_ids") or []]
    if not completion_logprobs:
        raw_logprobs = raw_output.get("completion_logprobs") or []
        completion_logprobs = [float(x) for x in raw_logprobs]

    return content, completion_token_ids, completion_logprobs, finish_reason


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/init")
def init(req: InitRequest) -> dict[str, str]:
    rollout_logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    rollout_logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    def _worker() -> None:
        token_turn_traces: list[dict[str, Any]] = []
        conversation = [_message_to_dict(message) for message in (req.messages or [])]
        params = dict(req.completion_params or {})
        tokenizer_model = (
            str(params.pop("tokenizer_model", "") or "")
            or os.environ.get("REMOTE_TOKENIZER_MODEL", "")
            or None
        )
        max_turns = int(params.pop("max_turns", os.environ.get("REMOTE_MAX_TURNS", "2")))
        max_turns = max(1, max_turns)
        for key in _NON_CHAT_PARAM_KEYS:
            params.pop(key, None)
        params["logprobs"] = True

        try:
            if not conversation:
                raise ValueError("messages are required")
            if not params.get("model"):
                raise ValueError("completion_params.model is required")
            if not req.model_base_url:
                raise ValueError("model_base_url is required")

            client = OpenAI(
                base_url=req.model_base_url,
                api_key=req.api_key or os.environ.get("FIREWORKS_API_KEY"),
            )

            for turn_idx in range(max_turns):
                prompt_ids = _render_prompt_ids(
                    tokenizer_model=tokenizer_model,
                    messages=conversation,
                )
                rollout_logger.info(
                    "remote rollout turn=%d model=%s messages=%d",
                    turn_idx + 1,
                    params["model"],
                    len(conversation),
                )
                response = client.chat.completions.create(
                    messages=conversation,
                    **params,
                )
                content, completion_ids, completion_logprobs, finish_reason = (
                    _extract_completion_trace(response)
                )
                assistant_message = {"role": "assistant", "content": content}
                conversation.append(assistant_message)

                # Keep rollout extras minimal: prompt ids are the only missing data
                # from tracing payloads for GRPO reconstruction.
                if prompt_ids:
                    token_turn_traces.append({"turn": turn_idx, "prompt_ids": prompt_ids, "finish_reason": finish_reason})
                else:
                    rollout_logger.warning(
                        "missing prompt ids for turn=%d prompt_ids=%d completion_ids=%d logprobs=%d",
                        turn_idx + 1,
                        len(prompt_ids),
                        len(completion_ids),
                        len(completion_logprobs),
                    )

                if turn_idx + 1 < max_turns:
                    conversation.append({"role": "user", "content": RETRY_PROMPT})

            rollout_logger.info(
                "rollout %s finished",
                req.metadata.rollout_id,
                extra={
                    "status": Status.rollout_finished(),
                    "token_turn_traces": token_turn_traces,
                    "remote_messages": conversation,
                },
            )
        except Exception as exc:
            rollout_logger.exception("remote rollout %s failed", req.metadata.rollout_id)
            rollout_logger.error(
                "rollout %s failed",
                req.metadata.rollout_id,
                extra={
                    "status": Status.rollout_unknown_error(str(exc)),
                    "token_turn_traces": token_turn_traces,
                    "remote_messages": conversation,
                },
            )

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GRPO remote rollout example server")
    parser.add_argument("--host", default=os.environ.get("REMOTE_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("REMOTE_SERVER_PORT", "3000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
