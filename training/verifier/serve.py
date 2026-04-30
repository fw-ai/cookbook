"""Local dev HTTP server: live UI for the renderer probe.

Serves ``training/verifier/viewer/index.html`` plus a single POST
``/probe`` endpoint that takes the form data the React page submits and
runs ``run_probe`` on the server. Purely stdlib — no Flask / FastAPI
dependency. Single-threaded by design; one probe at a time is fine for
interactive use and keeps state simple.

Usage::

    FIREWORKS_API_KEY=... python -m training.verifier.serve --port 8765
    open http://localhost:8765/

The viewer hits ``/probe`` with a JSON body shaped like::

    {
      "renderer": "glm5",
      "tokenizer_model": "zai-org/GLM-5.1",
      "deployment_id": null,        // mutually exclusive with "model"
      "model": null,                // explicit override; null = serverless default
      "messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}],
      "tools": [],
      "max_tokens": 1024,
      "temperature": 0.0,
      "train_on_what": "last_assistant_turn"
    }

…and gets back the same probe-artifact JSON the CLI writes (with a
``deployment.mode`` of "serverless" / "deployment" / "explicit"), or
``{"error": "...", "type": "..."}`` on failure.

Tokenizers and the Fireworks client are cached at module level so
repeated probes from the same browser session don't re-pay the
HuggingFace + auth cost on every keystroke.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tinker_cookbook.renderers.base import TrainOnWhat

from training.verifier.probe import (
    DispatchError,
    RENDERER_SERVERLESS_DEFAULTS,
    resolve_dispatch,
    run_probe,
)

logger = logging.getLogger(__name__)

VIEWER_DIR = Path(__file__).parent / "viewer"
INDEX_PATH = VIEWER_DIR / "index.html"

_CLIENT_LOCK = threading.Lock()
_CLIENT = None


@functools.lru_cache(maxsize=8)
def _tokenizer(name: str):
    from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: PLC0415

    return get_tokenizer(name)


def _client(api_key: str | None, base_url: str | None):
    """Single Fireworks client per process (the SDK is thread-safe enough
    for our single-threaded handler)."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")
        base_url = base_url or os.environ.get("FIREWORKS_BASE_URL")
        from fireworks import Fireworks  # type: ignore[import-not-found]  # noqa: PLC0415

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _CLIENT = Fireworks(**kwargs)
        return _CLIENT


def _run_one_probe(body: dict[str, Any]) -> dict[str, Any]:
    """Translate the React form payload into ``run_probe`` kwargs."""
    renderer = body.get("renderer")
    tokenizer_model = body.get("tokenizer_model")
    if not renderer or not tokenizer_model:
        raise ValueError("`renderer` and `tokenizer_model` are required")

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty list")

    model_str = (body.get("model") or "").strip() or None
    deployment_id = (body.get("deployment_id") or "").strip() or None

    model, dispatch_mode = resolve_dispatch(
        renderer_name=renderer,
        model=model_str,
        deployment_id=deployment_id,
    )
    train_on_what = TrainOnWhat(body.get("train_on_what") or TrainOnWhat.LAST_ASSISTANT_TURN.value)

    tokenizer = _tokenizer(tokenizer_model)
    client = _client(None, None)

    artifact = run_probe(
        renderer_name=renderer,
        tokenizer=tokenizer,
        client=client,
        model=model,
        messages=messages,
        tools=body.get("tools") or None,
        max_tokens=int(body.get("max_tokens") or 1024),
        temperature=float(body.get("temperature") or 0.0),
        train_on_what=train_on_what,
        deployment_id=deployment_id,
        tokenizer_model=tokenizer_model,
        renderer_config=body.get("renderer_config") or {},
        dispatch_mode=dispatch_mode,
    )
    return artifact


class ProbeHandler(http.server.BaseHTTPRequestHandler):
    server_version = "RendererVerifier/0.1"

    # -- helpers ---------------------------------------------------------

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, ctype: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib API
        logger.info("%s %s", self.address_string(), format % args)

    # -- handlers --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send_file(INDEX_PATH, "text/html; charset=utf-8")
        if path == "/health":
            return self._send_json({"ok": True})
        if path == "/registered":
            return self._send_json(
                {
                    "renderer_serverless_defaults": RENDERER_SERVERLESS_DEFAULTS,
                }
            )
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        if urlparse(self.path).path != "/probe":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"invalid JSON body: {exc}"}, status=400)
            return

        try:
            artifact = _run_one_probe(body)
        except DispatchError as exc:
            self._send_json({"error": str(exc), "type": "DispatchError"}, status=400)
            return
        except Exception as exc:  # noqa: BLE001 - surface anything to the page
            logger.exception("probe failed")
            self._send_json(
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )
            return

        self._send_json(artifact)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.verifier.serve",
        description="Local dev server for the live renderer probe UI.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host. Default 127.0.0.1.")
    p.add_argument("--port", type=int, default=8765, help="Bind port. Default 8765.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if not INDEX_PATH.exists():
        raise SystemExit(f"viewer not found at {INDEX_PATH}")

    if not os.environ.get("FIREWORKS_API_KEY"):
        logger.warning(
            "FIREWORKS_API_KEY is not set; /probe will fail until you export one."
        )

    server = http.server.HTTPServer((args.host, args.port), ProbeHandler)
    logger.info("verifier viewer up on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
