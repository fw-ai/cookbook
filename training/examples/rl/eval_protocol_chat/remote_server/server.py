#!/usr/bin/env python3
"""Example chat rollout server for Eval Protocol RemoteRolloutProcessor."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI
from openai import OpenAI

from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status
from eval_protocol.models import Message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger().addHandler(FireworksTracingHttpHandler())

app = FastAPI()

_NON_CHAT_PARAM_KEYS = ("base_url", "max_seq_len", "http_timeout")


def _message_to_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, Message):
        return message.dump_mdoel_for_chat_completion_request()
    return {k: v for k, v in dict(message).items() if v is not None}


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/init")
def init(req: InitRequest) -> dict[str, str]:
    rollout_logger = logging.getLogger(f"{__name__}.{req.metadata.rollout_id}")
    rollout_logger.addFilter(RolloutIdFilter(req.metadata.rollout_id))

    def _worker() -> None:
        conversation = [_message_to_dict(message) for message in (req.messages or [])]
        params = dict(req.completion_params or {})
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

            rollout_logger.info(
                "Eval Protocol chat model=%s messages=%d",
                params["model"],
                len(conversation),
            )
            response = client.chat.completions.create(
                messages=conversation,
                **params,
            )
            content = response.choices[0].message.content or ""
            conversation.append({"role": "assistant", "content": content})

            rollout_logger.info(
                "rollout %s finished",
                req.metadata.rollout_id,
                extra={"status": Status.rollout_finished()},
            )
        except Exception as exc:
            rollout_logger.exception("rollout %s failed", req.metadata.rollout_id)
            rollout_logger.error(
                "rollout %s failed",
                req.metadata.rollout_id,
                extra={"status": Status.rollout_unknown_error(str(exc))},
            )

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Eval Protocol chat rollout example server")
    parser.add_argument("--host", default=os.environ.get("REMOTE_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("REMOTE_SERVER_PORT", "3000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
