"""Sweep registered renderers and ping each one's serverless model.

Use this BEFORE running a triage / probe to confirm the model you want
to test is actually reachable. The triage pre-flight does the same
check for one renderer; this command does the cross-renderer sweep so
you can pick a known-working target.

Usage:
    python -m training.verifier.check_renderers
    python -m training.verifier.check_renderers --renderer glm5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from training.verifier.probe import RENDERER_SERVERLESS_DEFAULTS
from training.verifier.triage import _build_fireworks_client, _check_serverless

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.verifier.check_renderers",
        description="Sweep registered renderers; ping each serverless model.",
    )
    p.add_argument(
        "--renderer",
        default=None,
        help="Check a single renderer (default: every registered one).",
    )
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    api_key = args.api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit(
            "FIREWORKS_API_KEY not set. Pass --api-key or export FIREWORKS_API_KEY."
        )
    base_url = args.base_url or os.environ.get("FIREWORKS_BASE_URL")

    if args.renderer is not None:
        if args.renderer not in RENDERER_SERVERLESS_DEFAULTS:
            known = ", ".join(sorted(RENDERER_SERVERLESS_DEFAULTS))
            raise SystemExit(
                f"renderer {args.renderer!r} has no registered serverless default.\n"
                f"  Known renderers: {known}"
            )
        targets = {args.renderer: RENDERER_SERVERLESS_DEFAULTS[args.renderer]}
    else:
        targets = dict(RENDERER_SERVERLESS_DEFAULTS)

    client = _build_fireworks_client(api_key, base_url)

    print(f"{'renderer':<30}  {'model':<55}  status")
    print("-" * 110)
    all_reachable = True
    for name in sorted(targets):
        model = targets[name]
        is_reachable, message = _check_serverless(client, model)
        marker = "✓" if is_reachable else "✗"
        print(f"{name:<30}  {model:<55}  {marker} {message}")
        if not is_reachable:
            all_reachable = False

    return 0 if all_reachable else 1


if __name__ == "__main__":
    raise SystemExit(main())
