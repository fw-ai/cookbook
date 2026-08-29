"""Local live UI for sequential, JSON-driven renderer verification.

The browser has one execution path:

1. ``GET /input`` loads a validated JSON catalog containing every setting,
   renderer option, and prompt.
2. The browser reviews that input and calls ``POST /run-case`` sequentially.
3. Each response is a fresh probe result rendered only in browser memory.

The server never accepts or serves saved output artifacts. HTTP responses use
``Cache-Control: no-store``. Tokenizer, image-processor, and SDK client objects
remain process-resident because they are immutable execution dependencies, not
cached deployment results.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import ipaddress
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from training._vendor.tinker_cookbook_0_4_3.renderers.base import TrainOnWhat

from training.renderer.verifier.utils.inspect_rules import (
    load_rules as _load_inspect_rules,
)
from training.renderer.verifier.utils.presets import (
    load_preset_catalog,
    resolve_preset_case,
)
from training.renderer.verifier.utils.probe import (
    DispatchError,
    resolve_dispatch,
    run_probe,
)

logger = logging.getLogger(__name__)

VIEWER_DIR = Path(__file__).parent / "viewer"
INDEX_PATH = VIEWER_DIR / "index.html"

_CLIENT_LOCK = threading.Lock()
_CLIENT = None
_INPUT_FILE: Path | None = None

# Register cookbook-local renderers at server start.
import training.renderer  # noqa: F401, E402


@functools.lru_cache(maxsize=8)
def _tokenizer(name: str):
    from training.renderer.verifier.utils.tokenizer import load_tokenizer  # noqa: PLC0415

    return load_tokenizer(name)


@functools.lru_cache(maxsize=8)
def _image_processor(name: str):
    from training.renderer.verifier.utils.tokenizer import (  # noqa: PLC0415
        load_image_processor,
    )

    return load_image_processor(name)


def _is_loopback_url(value: str | None) -> bool:
    if not value:
        return False
    hostname = urlparse(value).hostname
    if hostname == "localhost":
        return True
    try:
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return False


def _sdk_base_url(value: str | None) -> str | None:
    """Remove a trailing ``/v1`` for the Fireworks SDK on loopback servers."""
    if not value or not _is_loopback_url(value):
        return value
    parsed = urlparse(value)
    if parsed.path.rstrip("/") != "/v1":
        return value
    return parsed._replace(path="", params="", query="", fragment="").geturl()


def _client(api_key: str | None, base_url: str | None):
    """Return one process-local SDK client; no completion results are cached."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        base_url = _sdk_base_url(base_url or os.environ.get("FIREWORKS_BASE_URL"))
        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key and _is_loopback_url(base_url):
            api_key = "local-no-auth"
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")

        from fireworks import Fireworks  # type: ignore[import-not-found]  # noqa: PLC0415

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _CLIENT = Fireworks(**kwargs)
        return _CLIENT


def _run_one_probe(body: dict[str, Any]) -> dict[str, Any]:
    """Execute one already-validated request resolved from the input JSON."""
    renderer = str(body["renderer"]).strip()
    tokenizer_model = str(body["tokenizer_model"]).strip()

    def optional_string(key: str) -> str | None:
        value = body.get(key)
        if value is None or value == "":
            return None
        return str(value).strip() or None

    model_str = optional_string("model")
    deployment_id = optional_string("deployment_id")
    image_processor_model = optional_string("image_processor_model")
    model, dispatch_mode = resolve_dispatch(
        renderer_name=renderer,
        model=model_str,
        deployment_id=deployment_id,
    )

    tokenizer = _tokenizer(tokenizer_model)
    image_processor = (
        _image_processor(image_processor_model) if image_processor_model else None
    )
    return run_probe(
        renderer_name=renderer,
        tokenizer=tokenizer,
        image_processor=image_processor,
        client=_client(None, None),
        model=model,
        messages=body["messages"],
        tools=body.get("tools") or None,
        max_tokens=int(body.get("max_tokens") or 1024),
        temperature=float(body.get("temperature") or 0.0),
        train_on_what=TrainOnWhat(
            body.get("train_on_what") or TrainOnWhat.LAST_ASSISTANT_TURN.value
        ),
        deployment_id=deployment_id,
        tokenizer_model=tokenizer_model,
        image_processor_model=image_processor_model,
        extra_completion_kwargs=body.get("extra_completion_kwargs") or {},
        dispatch_mode=dispatch_mode,
    )


def _load_input() -> dict[str, Any]:
    if _INPUT_FILE is None:
        raise RuntimeError("verifier input file is not configured")
    # Re-read and re-validate for every request. The input and results are never
    # cached, so editing the JSON then rerunning always uses the current file.
    return load_preset_catalog(_INPUT_FILE)


def _run_input_case(profile_id: str, example_id: str) -> dict[str, Any]:
    catalog = _load_input()
    profile, example, request = resolve_preset_case(
        catalog,
        profile_id=profile_id,
        example_id=example_id,
    )
    artifact = _run_one_probe(request)
    return {
        "profile_id": profile_id,
        "profile_label": profile["label"],
        "example_id": example_id,
        "example_label": example["label"],
        "description": example.get("description", ""),
        "artifact": artifact,
    }


class ProbeHandler(http.server.BaseHTTPRequestHandler):
    server_version = "RendererVerifier/0.2"

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info("%s %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            return self._send_file(INDEX_PATH, "text/html; charset=utf-8")
        if path == "/health":
            return self._send_json({"ok": True})
        if path == "/input":
            try:
                return self._send_json(_load_input())
            except Exception as exc:  # noqa: BLE001
                return self._send_json(
                    {"error": str(exc), "type": type(exc).__name__},
                    status=500,
                )
        if path == "/inspect_rules":
            try:
                return self._send_json({"rules": _load_inspect_rules()})
            except Exception as exc:  # noqa: BLE001
                return self._send_json(
                    {"error": str(exc), "type": type(exc).__name__},
                    status=500,
                )
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/run-case":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            profile_id = body.get("profile_id")
            example_id = body.get("example_id")
            if not isinstance(profile_id, str) or not profile_id:
                raise ValueError("profile_id is required")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("example_id is required")
            result = _run_input_case(profile_id, example_id)
        except (DispatchError, ValueError, json.JSONDecodeError) as exc:
            return self._send_json(
                {"error": str(exc), "type": type(exc).__name__},
                status=400,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("input case failed")
            return self._send_json(
                {"error": str(exc), "type": type(exc).__name__},
                status=500,
            )

        self._send_json(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.renderer.verifier.serve",
        description="Sequential JSON-input renderer verifier.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--input-file",
        required=True,
        help="Validated JSON catalog containing every setting and prompt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global _INPUT_FILE

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    if not INDEX_PATH.exists():
        raise SystemExit(f"viewer not found at {INDEX_PATH}")

    _INPUT_FILE = Path(args.input_file)
    try:
        catalog = load_preset_catalog(_INPUT_FILE)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid input file {_INPUT_FILE}: {exc}") from exc

    if not os.environ.get("FIREWORKS_API_KEY") and not _is_loopback_url(
        os.environ.get("FIREWORKS_BASE_URL")
    ):
        logger.warning(
            "FIREWORKS_API_KEY is not set; live cases will fail until it is exported"
        )

    case_count = sum(len(profile["examples"]) for profile in catalog["profiles"])
    logger.info("input-file: %s (%d cases)", _INPUT_FILE, case_count)
    server = http.server.HTTPServer((args.host, args.port), ProbeHandler)
    logger.info("verifier viewer up on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
