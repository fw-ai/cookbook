"""``python -m training.verifier <subcommand>`` argument parsing.

The probe is the only subcommand in this PR. Spec-driven L1 / L2
subcommands (``check``, ``corpus``) ship in follow-up PRs and will reuse
the artifact schema produced by ``render``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from tinker_cookbook.renderers.base import TrainOnWhat

from training.verifier.probe import run_probe, write_artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.verifier",
        description="Renderer verifier — Phase 0 (empirical probe).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "render",
        help="Run the empirical probe: render locally, complete on a deployed model, "
        "emit the audit-table JSON artifact.",
    )
    p.add_argument(
        "--renderer",
        required=True,
        help="Registered renderer name (e.g. glm5, gemma4, kimi_k25, nemotron3).",
    )
    p.add_argument(
        "--tokenizer-model",
        required=True,
        help="HuggingFace tokenizer model id or path (e.g. zai-org/GLM-5.1).",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Fireworks model identifier passed to chat.completions.create. "
        "Use a deployed model id for personal deployments "
        "(accounts/<acct>/deployedModels/<id>) or a serverless model name "
        "(accounts/fireworks/models/glm-5p1).",
    )
    p.add_argument(
        "--deployment-id",
        default=None,
        help="Optional deployment id, recorded in the artifact for traceability "
        "even when --model already encodes it.",
    )
    p.add_argument(
        "--input",
        required=True,
        help='Path to a JSON file with shape {"messages": [...], "tools": [...]}. '
        '"tools" is optional.',
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for the probe JSON artifact.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Fireworks API key. Falls back to FIREWORKS_API_KEY env var.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Optional Fireworks API base URL override (e.g. for staging). "
        "Falls back to FIREWORKS_BASE_URL env var.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Cap on completion tokens. Default 256; raise for long-form probes.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default 0 for reproducible probes.",
    )
    p.add_argument(
        "--train-on-what",
        default=TrainOnWhat.LAST_ASSISTANT_TURN.value,
        help="TrainOnWhat mode used when computing renderer claim weights. "
        "Default last_assistant_turn matches the cookbook SFT default.",
    )
    return parser


def _load_input(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "messages" not in data:
        raise SystemExit(f"--input file {path!r} must be a JSON object with a 'messages' key")
    return data


def _build_client(api_key: str | None, base_url: str | None):
    api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit(
            "FIREWORKS_API_KEY not set. Pass --api-key or export FIREWORKS_API_KEY."
        )
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL")

    # Imported lazily so unit tests can run without the Fireworks SDK
    # being importable (they pass a stub client directly to run_probe).
    from fireworks import Fireworks  # type: ignore[import-not-found]

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return Fireworks(**kwargs)


def _build_tokenizer(model: str):
    import transformers  # type: ignore[import-not-found]

    return transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.cmd != "render":  # pragma: no cover - argparse enforces required=True
        return 2

    payload = _load_input(args.input)
    messages = payload["messages"]
    tools = payload.get("tools") or None

    train_on_what = TrainOnWhat(args.train_on_what)

    tokenizer = _build_tokenizer(args.tokenizer_model)
    client = _build_client(args.api_key, args.base_url)

    artifact = run_probe(
        renderer_name=args.renderer,
        tokenizer=tokenizer,
        client=client,
        model=args.model,
        messages=messages,
        tools=tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        train_on_what=train_on_what,
        deployment_id=args.deployment_id,
        tokenizer_model=args.tokenizer_model,
        renderer_config=payload.get("renderer_config") or {},
    )

    write_artifact(artifact, args.output)
    print(f"wrote probe artifact: {args.output}", file=sys.stderr)
    return 0
