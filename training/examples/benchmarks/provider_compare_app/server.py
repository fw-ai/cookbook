"""Provider speed comparison app.

Single process: serves the index.html page and proxies streaming completion
requests to each configured provider, keeping API keys server-side.

Run:
    pip install -r requirements.txt
    cp /path/to/training/.env .   # or export keys in your shell
    python server.py
    open http://localhost:8080
"""

import asyncio
import json
import os
import time
from pathlib import Path

import litellm
from aiohttp import web
from dotenv import load_dotenv

# --- load env (training/.env if present, else shell) ---
load_dotenv()
training_dir = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if p.name == "training" and (p / "pyproject.toml").exists()),
    None,
)
if training_dir and (training_dir / ".env").exists():
    load_dotenv(training_dir / ".env", override=False)

litellm.drop_params = False

# --- config ---
HOST = "0.0.0.0"
PORT = 8080
MAX_TOKENS = 1024
REQUEST_TIMEOUT_S = 300

# Reasoning effort. "" = use each provider's default; "max" = max thinking.
# For an experiential "watch them stream" demo, "" or "high" makes TTFT short
# so the user sees text sooner; "max" makes TTFT long (model thinks first).
REASONING_EFFORT = "high"

# Providers to race. Each entry: name -> {slug, api_base, api_key_env, reasoning}.
# Defaults to GLM-5.2 across Fireworks / Baseten / Together (same model, different
# serving) so the user feels *serving* speed differences. Swap slugs to compare
# other models; e.g. add OpenRouter Opus 4.8 / GPT-5.5 using their reasoning body
# shape ({"reasoning": {"effort": ...}}) instead of top-level reasoning_effort.
FIREWORKS_5_2_MODEL = "fireworks_ai/accounts/fireworks/models/glm-5p2"
TOGETHER_5_2_MODEL = "together_ai/zai-org/GLM-5.2"
BASETEN_5_2_MODEL = "openai/zai-org/GLM-5.2"
BASETEN_API_BASE = "https://inference.baseten.co/v1"

PROVIDERS = {
    "Fireworks GLM-5.2": {
        "slug": FIREWORKS_5_2_MODEL,
        "api_base": None,
        "api_key_env": "FIREWORKS_API_KEY",
        "reasoning": {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None,
    },
    "Baseten GLM-5.2": {
        "slug": BASETEN_5_2_MODEL,
        "api_base": BASETEN_API_BASE,
        "api_key_env": "BASETEN_API_KEY",
        "reasoning": {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None,
    },
    "Together GLM-5.2": {
        "slug": TOGETHER_5_2_MODEL,
        "api_base": None,
        "api_key_env": "TOGETHER_API_KEY",
        "reasoning": {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None,
    },
}

# Optional shared-secret header to gate public deploys. If set, the browser must
# send this header (configured in index.html) or requests are rejected. Leave "".
SHARED_SECRET = os.getenv("COMPARE_APP_SECRET", "")

HERE = Path(__file__).parent


def _kwargs_for(cfg: dict) -> dict:
    kw = {}
    if cfg.get("api_base"):
        kw["api_base"] = cfg["api_base"]
    key = os.getenv(cfg["api_key_env"]) if cfg.get("api_key_env") else None
    if key:
        kw["api_key"] = key
    if cfg.get("reasoning"):
        kw["extra_body"] = dict(cfg["reasoning"])
    return kw


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


async def _race_one(column_id: str, name: str, cfg: dict, prompt: str, resp: web.StreamResponse) -> None:
    """Stream one provider into the SSE response, tagged with `column_id`."""
    await resp.write(_sse({"column": column_id, "type": "start", "name": name}))

    key_ok = bool(os.getenv(cfg["api_key_env"])) if cfg.get("api_key_env") else True
    if not key_ok:
        await resp.write(_sse({"column": column_id, "type": "error",
                               "error": f"Missing {cfg['api_key_env']} on the server."}))
        await resp.write(_sse({"column": column_id, "type": "done", "tok_s": 0.0, "out_tokens": 0}))
        return

    t0 = time.perf_counter()
    ttft = None
    chunks = []
    last_ev = None
    try:
        stream = await litellm.acompletion(
            model=cfg["slug"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
            stream=True,
            **_kwargs_for(cfg),
        )
        async for ev in stream:
            last_ev = ev
            delta = ev.choices[0].delta if getattr(ev, "choices", None) else None
            piece = getattr(delta, "content", None) if delta else None
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    await resp.write(_sse({"column": column_id, "type": "metric",
                                           "ttft": round(ttft, 3), "tok_s": 0.0}))
                chunks.append(piece)
                await resp.write(_sse({"column": column_id, "type": "tok", "text": piece}))
                # Live tok/s = out_tokens so far / (now - ttft_start).
                if ttft is not None:
                    elapsed = time.perf_counter() - t0 - ttft
                    if elapsed > 0:
                        est_out = sum(len(c) for c in chunks) / 4.0  # ~4 chars/token rough
                        live_tps = est_out / elapsed
                        await resp.write(_sse({"column": column_id, "type": "metric",
                                               "ttft": round(ttft, 3),
                                               "tok_s": round(live_tps, 1)}))
        dt = time.perf_counter() - t0
        usage = getattr(last_ev, "usage", None) or getattr(stream, "usage", None)
        out_tok = getattr(usage, "completion_tokens", None) if usage else None
        if out_tok is None:
            text = "".join(chunks)
            if text:
                try:
                    out_tok = litellm.token_counter(model=cfg["slug"], text=text)
                except Exception:
                    out_tok = None
        out_tok = out_tok or 0
        final_tps = (out_tok / (dt - ttft)) if (out_tok and ttft and dt > ttft) else 0.0
        await resp.write(_sse({"column": column_id, "type": "done",
                               "ttft": round(ttft, 3) if ttft is not None else None,
                               "tok_s": round(final_tps, 1), "out_tokens": out_tok,
                               "latency": round(dt, 3)}))
    except Exception as e:  # noqa: BLE001
        await resp.write(_sse({"column": column_id, "type": "error", "error": repr(e)[:300]}))
        await resp.write(_sse({"column": column_id, "type": "done", "tok_s": 0.0, "out_tokens": 0}))


async def index(request: web.Request) -> web.Response:
    return web.FileResponse(HERE / "index.html")


async def race(request: web.Request) -> web.StreamResponse:
    if SHARED_SECRET and request.headers.get("X-Compare-Secret") != SHARED_SECRET:
        return web.json_response({"error": "unauthorized"}, status=401)

    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable nginx buffering
    })
    await resp.prepare(request)

    # Fan out to all providers concurrently. Each writes its own tagged chunks
    # into the shared SSE response. asyncio.gather keeps them interleaved.
    tasks = [
        _race_one(col_id, name, cfg, prompt, resp)
        for col_id, (name, cfg) in enumerate(PROVIDERS.items())
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    await resp.write(b"event: end\ndata: {}\n\n")
    await resp.write_eof()
    return resp


async def providers(request: web.Request) -> web.Response:
    """Tell the frontend how many columns + their labels."""
    return web.json_response([
        {"id": i, "name": name} for i, name in enumerate(PROVIDERS.keys())
    ])


def main() -> None:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/providers", providers)
    app.router.add_post("/api/race", race)
    print(f"Provider comparison app on http://localhost:{PORT}")
    print(f"  providers: {list(PROVIDERS.keys())}")
    missing = [cfg["api_key_env"] for cfg in PROVIDERS.values()
               if cfg.get("api_key_env") and not os.getenv(cfg["api_key_env"])]
    if missing:
        print(f"  WARNING: missing env keys (those columns will error): {missing}")
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
