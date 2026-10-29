#!/usr/bin/env python3
"""Local side-by-side playground: base model vs. the RL-tuned code-repair adapter.

Both models are sampled through serverless sampling clients (no deployment, no
training slot; billed per sampled token): the base model by name, the tuned
adapter by its sampler checkpoint path. "Fix a function" sends an eval row to
both and grades each reply in the local sandbox.

    cd training
    finetune uv run --with fastapi --with uvicorn \\
        python -m training.examples.rl.code_repair.compare_app.server

Access control: binds 127.0.0.1 only, rejects other Host headers (DNS
rebinding), and requires the per-launch token printed in the URL on every API
call. The API key comes from FIREWORKS_API_KEY, or is entered once in the page
and kept in this process's memory only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from training.examples.rl.code_repair.prepare_data import DATA_DIR
from training.examples.rl.code_repair.replies import split_reply
from training.examples.rl.code_repair.rewards import extract_code
from training.examples.rl.code_repair.sandbox import grade

HERE = Path(__file__).resolve().parent
RUN_RESULT = HERE.parent / "runs" / "full" / "result.json"


class KeyBody(BaseModel):
    key: str


class GenerateBody(BaseModel):
    mode: str = "chat"
    question: str = ""
    system: str = ""
    row_id: str | None = None
    temperature: float = 1.0
    max_tokens: int = 8192


class Models:
    """Lazily opened tokenizer, renderer and the two samplers."""

    def __init__(self, base_model: str, tokenizer_model: str, tuned_path: str) -> None:
        self.base_model, self.tokenizer_model, self.tuned_path = base_model, tokenizer_model, tuned_path
        self.api_key: str | None = os.environ.get("FIREWORKS_API_KEY") or None
        self._lock = threading.Lock()
        self._service = self._base = self._tuned = None
        self.tokenizer = self.renderer = None

    def set_key(self, key: str) -> None:
        with self._lock:
            self.close_locked()
            self.api_key = key.strip() or None

    def close_locked(self) -> None:
        for client in (self._base, self._tuned, self._service):
            try:
                if client is not None:
                    client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self._service = self._base = self._tuned = None

    def samplers(self):
        with self._lock:
            if self._base is None:
                if not self.api_key:
                    raise HTTPException(401, "No Fireworks API key set")
                from fireworks.training.sdk import FiretitanServiceClient

                from training.recipes.experiment.async_rl_loop_serverless import load_tokenizer
                from training.utils.supervised import build_renderer

                if self.tokenizer is None:
                    self.tokenizer = load_tokenizer(self.tokenizer_model, None)
                    self.renderer = build_renderer(self.tokenizer, self.tokenizer_model, "")
                self._service = FiretitanServiceClient(
                    api_key=self.api_key, base_url="https://api.fireworks.ai/training/v1/serverless"
                )
                self._base = self._service.create_sampling_client(base_model=self.base_model, tokenizer=self.tokenizer)
                self._tuned = self._service.create_sampling_client(model_path=self.tuned_path, tokenizer=self.tokenizer)
            return self._base, self._tuned


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    user = row["messages"][-1]["content"]
    task = re.search(r"## Task\n(.*?)\n\n## Function signature", user, re.S)
    buggy = re.search(r"## Buggy implementation\n```python\n(.*?)\n```", user, re.S)
    gt = row["ground_truth"]
    return {
        "id": row["id"],
        "entry_point": gt["entry_point"],
        "buggy_pass_rate": gt["buggy_pass_rate"],
        "num_cases": gt["num_cases"],
        "task": task.group(1) if task else "",
        "buggy_code": buggy.group(1) if buggy else "",
    }


def create_app(models: Models, rows: dict[str, dict[str, Any]], token: str, port: int) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host") not in allowed_hosts:
            return JSONResponse({"detail": "bad host"}, status_code=403)
        if request.url.path.startswith("/api/") and not secrets.compare_digest(
            request.headers.get("x-app-token", ""), token
        ):
            return JSONResponse({"detail": "missing or wrong app token"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    async def index():
        return FileResponse(HERE / "index.html")

    @app.get("/api/status")
    async def status():
        return {
            "has_key": bool(models.api_key),
            "base_model": models.base_model,
            "tuned_path": models.tuned_path,
        }

    @app.post("/api/key")
    async def set_key(body: KeyBody):
        models.set_key(body.key)
        return {"has_key": bool(models.api_key)}

    @app.get("/api/functions")
    async def functions():
        return [row_summary(r) for r in rows.values()]

    async def run_one(sampler: Any, prompt: Any, params: Any, row: dict | None) -> dict[str, Any]:
        started = time.monotonic()
        try:
            response = await asyncio.to_thread(
                lambda: sampler.sample(prompt, num_samples=1, sampling_params=params).result(timeout=1800)
            )
        except Exception as err:  # noqa: BLE001 - surface sampler errors in the UI
            return {"error": f"{type(err).__name__}: {err}"}
        sequence = response.sequences[0]
        tokens = list(sequence.tokens)
        reasoning, answer = split_reply(models.renderer, tokens)
        if "length" in str(sequence.stop_reason).lower() and not reasoning:
            reasoning, answer = answer, ""
        out: dict[str, Any] = {
            "reasoning": reasoning,
            "answer": answer,
            "stop_reason": str(sequence.stop_reason),
            "completion_tokens": len(tokens),
            "seconds": round(time.monotonic() - started, 1),
        }
        if row is not None:
            gt = row["ground_truth"]
            truncated = "length" in out["stop_reason"].lower()
            code = None if truncated else extract_code(answer, gt["entry_point"])
            if code is None:
                out["grade"] = {"passed": 0, "total": gt["num_cases"], "status": "truncated" if truncated else "no code"}
            else:
                result = await grade(code, gt["test"], gt["entry_point"])
                out["grade"] = {"passed": result.passed, "total": gt["num_cases"], "status": result.status}
        return out

    @app.post("/api/generate")
    async def generate(body: GenerateBody):
        from fireworks.training.sdk import types

        base, tuned = await asyncio.to_thread(models.samplers)
        row = None
        if body.mode == "repair":
            if body.row_id not in rows:
                raise HTTPException(404, f"unknown function {body.row_id!r}")
            row = rows[body.row_id]
            messages = row["messages"]
        else:
            if not body.question.strip():
                raise HTTPException(400, "empty question")
            messages = ([{"role": "system", "content": body.system}] if body.system.strip() else []) + [
                {"role": "user", "content": body.question}
            ]
        prompt = models.renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            temperature=max(0.0, body.temperature),
            max_tokens=max(16, min(body.max_tokens, 16384)),
            stop=models.renderer.get_stop_sequences(),
        )
        base_out, tuned_out = await asyncio.gather(
            run_one(base, prompt, params, row), run_one(tuned, prompt, params, row)
        )
        return {"base": base_out, "tuned": tuned_out}

    return app


def default_tuned_path() -> str | None:
    if RUN_RESULT.exists():
        return json.loads(RUN_RESULT.read_text()).get("final_sampler_snapshot")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3p8-27b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--tuned-checkpoint", default=default_tuned_path(), help="Sampler checkpoint path.")
    parser.add_argument("--rows", type=Path, default=DATA_DIR / "eval.jsonl")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.tuned_checkpoint:
        raise SystemExit("pass --tuned-checkpoint (no runs/full/result.json found)")

    token = secrets.token_urlsafe(16)
    models = Models(args.base_model, args.tokenizer_model, args.tuned_checkpoint)
    app = create_app(models, load_rows(args.rows), token, args.port)
    print(f"\n  Base:  {args.base_model}\n  Tuned: {args.tuned_checkpoint}")
    print(f"  API key: {'from FIREWORKS_API_KEY' if models.api_key else 'enter it in the page'}")
    print(f"\n  Open: http://127.0.0.1:{args.port}/?token={token}\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
