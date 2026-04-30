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

from training.verifier.inspect import run_inspect
from training.verifier.probe import (
    run_probe,
    serverless_default_for,
    write_artifact,
)


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
        default=None,
        help="Fireworks model identifier passed to chat.completions.create. "
        "Optional: if omitted and --deployment-id isn't set either, the probe "
        "falls back to a per-renderer serverless default. Pass this when you "
        "need to override the default (e.g. probe a different serverless model "
        "or a hand-constructed deployment string).",
    )
    p.add_argument(
        "--deployment-id",
        default=None,
        help="Personal deployment id (the same id you passed to "
        "training.verifier.spinup_deployment up). When set, the probe resolves "
        "the full identifier accounts/<account>/deployments/<id> via "
        "DeploymentManager. Mutually exclusive with --model.",
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
        default=1024,
        help="Cap on completion tokens. Default 1024; thinking-enabled models "
        "easily produce hundreds of tokens before a natural stop, and the "
        "probe is most informative when stop_reason=='stop'.",
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

    insp = sub.add_parser(
        "inspect",
        help="Pretty-print a probe artifact. Without --all, shows chunk "
        "boundaries, every special-source row, and every empirically "
        "suspect row — the structurally interesting subset for spec authors.",
    )
    insp.add_argument("path", help="Path to a probe JSON artifact.")
    insp.add_argument(
        "--all",
        dest="show_all",
        action="store_true",
        help="Print every audit-table row instead of the boundaries-only digest.",
    )
    insp.add_argument(
        "--filter",
        dest="filter_prov",
        default=None,
        help="Show only rows with this provenance (e.g. trailing_hard_append).",
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


def _resolve_dispatch(args, *, renderer_name: str) -> tuple[str, str]:
    """Pick (model_identifier, dispatch_mode) per the design contract.

    Precedence:
      1. --model (and --deployment-id) mutually exclusive — error if both.
      2. --deployment-id → resolve via DeploymentManager.account_id +
         deployment_id → ``accounts/<account>/deployments/<id>``.
      3. --model → use as-is, mode = "explicit".
      4. neither → renderer's serverless default (mode = "serverless").
      5. neither and no default registered → error out, asking the caller
         to spin up a deployment via training.verifier.spinup_deployment.
    """
    if args.model and args.deployment_id:
        raise SystemExit("--model and --deployment-id are mutually exclusive")

    if args.deployment_id:
        api_key = args.api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise SystemExit(
                "FIREWORKS_API_KEY not set; cannot resolve --deployment-id."
            )
        base_url = args.base_url or os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai"
        )
        from fireworks.training.sdk.deployment import DeploymentManager  # noqa: PLC0415

        mgr = DeploymentManager(api_key=api_key, base_url=base_url)
        return (
            f"accounts/{mgr.account_id}/deployments/{args.deployment_id}",
            "deployment",
        )

    if args.model:
        return args.model, "explicit"

    default = serverless_default_for(renderer_name)
    if default is None:
        raise SystemExit(
            f"renderer {renderer_name!r} has no registered Fireworks serverless "
            "default. Either pass --model explicitly or spin up a personal "
            "deployment first:\n\n"
            "    python -m training.verifier.spinup_deployment up \\\n"
            "        --base-model <accounts/.../models/...> \\\n"
            "        --shape <accounts/.../deploymentShapes/...> \\\n"
            "        --deployment-id my-probe\n\n"
            "    python -m training.verifier render --deployment-id my-probe ..."
        )
    return default, "serverless"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.cmd == "inspect":
        return run_inspect(args.path, show_all=args.show_all, filter_prov=args.filter_prov)

    if args.cmd != "render":  # pragma: no cover - argparse enforces required=True
        return 2

    payload = _load_input(args.input)
    messages = payload["messages"]
    tools = payload.get("tools") or None

    train_on_what = TrainOnWhat(args.train_on_what)

    model, dispatch_mode = _resolve_dispatch(args, renderer_name=args.renderer)
    logging.getLogger("training.verifier").info(
        "dispatch=%s model=%s", dispatch_mode, model
    )

    tokenizer = _build_tokenizer(args.tokenizer_model)
    client = _build_client(args.api_key, args.base_url)

    artifact = run_probe(
        renderer_name=args.renderer,
        tokenizer=tokenizer,
        client=client,
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        train_on_what=train_on_what,
        deployment_id=args.deployment_id,
        tokenizer_model=args.tokenizer_model,
        renderer_config=payload.get("renderer_config") or {},
        dispatch_mode=dispatch_mode,
    )

    write_artifact(artifact, args.output)
    print(f"wrote probe artifact: {args.output}", file=sys.stderr)
    return 0
