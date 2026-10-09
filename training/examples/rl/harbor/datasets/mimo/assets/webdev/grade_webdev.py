"""Webdev eval grader: screenshot the built site, one vision call, absolute score.

Render or judge failure exits without writing a reward, so Harbor treats the
trial as unscored. A real score is the mean of visual, query fulfillment, and
premium assets, using the MiMo rubric unchanged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

from eval_rubric import ALL_KEYS, VISUAL_KEYS, build_prompt
from shot import _build_shot_cmd, _parse_render_env, _parse_shot_b64

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_REWARD = Path("/logs/verifier/reward.txt")


def _fail(reason: str) -> None:
    print(reason, flush=True)
    raise SystemExit(2)


def _screenshot(cwd: str) -> str:
    os.environ["WEBDEV_GRADE_HTTP"] = "1"
    url = f"file://{cwd.rstrip('/')}/index.html"
    command = _build_shot_cmd(url)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _fail("screenshot timed out")
    result = {
        "output": (completed.stdout or "") + (completed.stderr or ""),
        "reason": "ok",
    }
    render_env = _parse_render_env(result)
    if render_env.get("proxy_failed"):
        _fail("render env failed")
    image, _errors = _parse_shot_b64(result)
    if not image:
        _fail("screenshot produced no image")
    return image


def _judge(image_b64: str, query: str) -> float:
    model = os.environ.get("WEBDEV_EVAL_JUDGE_MODEL", "")
    base = os.environ.get("WEBDEV_EVAL_JUDGE_BASE_URL", "").rstrip("/")
    key = os.environ.get("WEBDEV_EVAL_JUDGE_API_KEY", "")
    if not model or not base or not key:
        _fail("vision judge is not configured")
    if not base.endswith("/v1"):
        base += "/v1"
    prompt = build_prompt().format(query=query[:8000])
    body = {
        "model": model,
        "temperature": 1.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + image_b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        _fail(f"judge request failed: {type(exc).__name__}")
    text = payload["choices"][0]["message"]["content"]
    match = _JSON_RE.search(text or "")
    if not match:
        _fail("judge returned no json")
    data = json.loads(match.group(0))
    dims = {}
    for name in ALL_KEYS:
        value = data.get(name)
        if value is None:
            _fail(f"judge omitted {name}")
        dims[name] = max(0.0, min(1.0, float(value)))
    visual = sum(dims[name] for name in VISUAL_KEYS) / len(VISUAL_KEYS)
    return (visual + dims["query_fulfillment"] + dims["premium_assets"]) / 3.0


def main() -> None:
    cwd = os.environ.get("WEBDEV_CWD", "/workspace")
    query = Path("/tests/query.txt").read_text(encoding="utf-8")
    image = _screenshot(cwd)
    score = _judge(image, query)
    _REWARD.parent.mkdir(parents=True, exist_ok=True)
    _REWARD.write_text(f"{score:.4f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
