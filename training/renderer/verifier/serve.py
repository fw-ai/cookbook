"""Local dev HTTP server: live UI for the renderer probe.

Serves ``training/renderer/verifier/viewer/index.html`` plus a single POST
``/probe`` endpoint that takes the form data the React page submits and
runs ``run_probe`` on the server. Purely stdlib — no Flask / FastAPI
dependency. Single-threaded by design; one probe at a time is fine for
interactive use and keeps state simple.

Usage::

    FIREWORKS_API_KEY=... python -m training.renderer.verifier.serve --port 8765
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
``deployment.mode`` of "deployment" / "explicit"), or
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

from training.renderer.verifier.utils.probe import (
    DispatchError,
    resolve_dispatch,
    run_probe,
)
from training.renderer.verifier.utils.inspect_rules import load_rules as _load_inspect_rules

logger = logging.getLogger(__name__)

VIEWER_DIR = Path(__file__).parent / "viewer"
INDEX_PATH = VIEWER_DIR / "index.html"

_CLIENT_LOCK = threading.Lock()
_CLIENT = None

# When ``--session-file`` is passed, the GUI fetches /session on mount
# and auto-seeds cases from the file (typically a triage output).
_SESSION_FILE: Path | None = None

# Force the cookbook's local renderers to register at server start.
# Without this, tinker_cookbook's custom-renderer registry is empty
# until something else triggers the import.
import training.renderer  # noqa: F401, E402 — side-effect: register_renderer calls

# Static catalog of (renderer name → suggested HF tokenizer id) for the
# UI's auto-fill. Covers:
#   - ``tinker_cookbook`` built-ins (hardcoded in ``get_renderer``'s
#     elif ladder, NOT in ``_CUSTOM_RENDERER_REGISTRY`` — there's no
#     iterable accessor for them).
#   - cookbook-local renderers (registered when ``training.renderer``
#     is imported).
#   - Anticipated future renderers that share the cookbook's naming
#     convention (e.g. kimi_k26, qwen3_6) so the dropdown surfaces
#     them; selecting one before its renderer module exists yields a
#     clear "renderer not registered" error from get_renderer().
#
# Why static: the Fireworks Model proto exposes ``huggingface_files``
# (uploaded blobs) but not a canonical HF repo id, and there is no
# registry-side metadata linking a renderer to its tokenizer. Edit
# this dict when adding a renderer; ``skills/verifier/SKILL.md`` is
# the user-facing version of the same map.
RENDERER_TOKENIZER_DEFAULTS: dict[str, str | None] = {
    # tinker_cookbook built-ins (see tinker_cookbook.renderers.get_renderer)
    "role_colon":                  None,
    "llama3":                      "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3":                       "Qwen/Qwen3-8B",
    "qwen3_vl":                    "Qwen/Qwen3-VL-7B-Instruct",
    "qwen3_vl_instruct":           "Qwen/Qwen3-VL-7B-Instruct",
    "qwen3_disable_thinking":      "Qwen/Qwen3-8B",
    "qwen3_instruct":              "Qwen/Qwen3-8B-Instruct-2507",
    "qwen3_5":                     "Qwen/Qwen3.5-VL-8B-Instruct",
    "qwen3_5_disable_thinking":    "Qwen/Qwen3.5-VL-8B-Instruct",
    "qwen3_6":                     "Qwen/Qwen3.6-VL-8B-Instruct",
    "qwen3_6_disable_thinking":    "Qwen/Qwen3.6-VL-8B-Instruct",
    "deepseekv3":                  "deepseek-ai/DeepSeek-V3",
    "deepseekv3_disable_thinking": "deepseek-ai/DeepSeek-V3",
    "deepseekv3_thinking":         "deepseek-ai/DeepSeek-V3",
    "kimi_k2":                     "moonshotai/Kimi-K2-Instruct",
    "kimi_k25":                    "moonshotai/Kimi-K2.5",
    "kimi_k25_disable_thinking":   "moonshotai/Kimi-K2.5",
    "kimi_k26":                    "moonshotai/Kimi-K2.6",
    "kimi_k26_disable_thinking":   "moonshotai/Kimi-K2.6",
    "nemotron3":                   "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "nemotron3_disable_thinking":  "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "gpt_oss_no_sysprompt":        "openai/gpt-oss-120b",
    "gpt_oss_low_reasoning":       "openai/gpt-oss-120b",
    "gpt_oss_medium_reasoning":    "openai/gpt-oss-120b",
    "gpt_oss_high_reasoning":      "openai/gpt-oss-120b",
    # cookbook-local (training/renderer/)
    "gemma4":      "google/gemma-4-E2B-it",
    "glm5":        "zai-org/GLM-5.1",
    "minimax_m2":  "MiniMaxAI/MiniMax-M2",
    "nemotron":    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
}


@functools.lru_cache(maxsize=8)
def _tokenizer(name: str):
    from training.renderer.verifier.utils.tokenizer import load_tokenizer  # noqa: PLC0415

    return load_tokenizer(name)


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
        if path == "/renderers":
            # Catalog of renderer names with the suggested HF tokenizer
            # for each. Combines:
            #   1. tinker_cookbook built-ins (hardcoded in get_renderer's
            #      elif ladder — not in any registry we can iterate)
            #   2. The custom registry (cookbook renderers register
            #      themselves at module import).
            # We surface the union so the GUI dropdown shows everything.
            from tinker_cookbook.renderers import (  # noqa: PLC0415
                get_registered_renderer_names,
            )
            registered = set(get_registered_renderer_names())
            names = sorted(set(RENDERER_TOKENIZER_DEFAULTS.keys()) | registered)
            payload = [
                {
                    "name": name,
                    "tokenizer_default": RENDERER_TOKENIZER_DEFAULTS.get(name),
                    "registered": name in registered,
                }
                for name in names
            ]
            return self._send_json({"renderers": payload})
        if path == "/models":
            # Live list of *serverless-eligible* Fireworks models in the
            # public account. `supports_serverless` is an output-only bool
            # on the gateway Model proto (see fireworks control_plane
            # protos/gateway/model.proto) and ListModels supports an
            # AIP-160 filter, so a server-side filter avoids paginating
            # through the entire catalogue. Used by the GUI to populate
            # the model dropdown; not cached so a page refresh picks up
            # new serverless additions.
            try:
                from fireworks import Fireworks  # noqa: PLC0415
                api_key = os.environ.get("FIREWORKS_API_KEY")
                if not api_key:
                    return self._send_json(
                        {"error": "FIREWORKS_API_KEY not set"}, status=503,
                    )
                client = Fireworks(api_key=api_key)
                models = []
                for m in client.models.list(
                    account_id="fireworks",
                    filter="supports_serverless=true",
                ):
                    name = getattr(m, "name", None) or getattr(m, "id", None)
                    if not name:
                        continue
                    models.append({
                        "name": name,
                        "display_name": getattr(m, "display_name", None),
                        "kind": getattr(m, "kind", None),
                        "state": getattr(m, "state", None),
                    })
                return self._send_json({"models": models})
            except Exception as exc:  # noqa: BLE001 — surface to the page
                logger.exception("models listing failed")
                return self._send_json(
                    {"error": str(exc), "type": type(exc).__name__},
                    status=500,
                )
        if path == "/inspect_rules":
            # Re-read on every request so the user can edit the YAML
            # without restarting the server. Trivially cheap; it's a
            # tiny file and only fetched once per page load.
            try:
                rules = _load_inspect_rules()
            except Exception as exc:  # noqa: BLE001
                return self._send_json(
                    {"error": str(exc), "type": type(exc).__name__},
                    status=500,
                )
            return self._send_json({"rules": rules})
        if path == "/session":
            # Surfaces the triage session file (if any) so the GUI can
            # auto-seed flagged cases on mount. Re-reads each request.
            if _SESSION_FILE is None or not _SESSION_FILE.exists():
                return self._send_json({"kind": "probe-batch", "cases": []})
            try:
                with open(_SESSION_FILE, "r", encoding="utf-8") as f:
                    return self._send_json(json.load(f))
            except Exception as exc:  # noqa: BLE001
                return self._send_json(
                    {"error": str(exc), "type": type(exc).__name__},
                    status=500,
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
        prog="python -m training.renderer.verifier.serve",
        description="Local dev server for the live renderer probe UI.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host. Default 127.0.0.1.")
    p.add_argument("--port", type=int, default=8765, help="Bind port. Default 8765.")
    p.add_argument(
        "--session-file",
        default=None,
        help="Path to a triage session JSON. When set, the GUI fetches "
        "/session on mount and auto-seeds the flagged cases.",
    )
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

    if args.session_file:
        global _SESSION_FILE
        _SESSION_FILE = Path(args.session_file)
        logger.info("session-file: %s (%s)", _SESSION_FILE,
                    "exists" if _SESSION_FILE.exists() else "MISSING")

    server = http.server.HTTPServer((args.host, args.port), ProbeHandler)
    logger.info("verifier viewer up on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
