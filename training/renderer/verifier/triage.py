"""Batch-probe a corpus of prompts and load every case into the GUI.

Workflow:

  1. Load a prompt corpus from the JSON file passed via ``--prompts``.
  2. Validate the API key, then ping the dispatch target so we can show
     reachability in the pre-flight summary.
  3. Print a pre-flight — renderer + dispatch + reachability + every
     prompt snippet — and ask the user to confirm before live calls.
  4. Loop the corpus through ``run_probe`` against the live gateway.
  5. Write a session JSON containing every successful case. The dev
     server reads it via ``--session-file`` and seeds the GUI
     (``/session`` endpoint), so the page comes up with all cases
     stacked for side-by-side inspection. Per-token visual flagging
     in the GUI still uses ``inspect_rules.yaml`` — but no filtering
     happens here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from tinker_cookbook.renderers.base import TrainOnWhat

from training.renderer.verifier.utils.probe import (
    DispatchError,
    resolve_dispatch,
    run_probe,
)

logger = logging.getLogger(__name__)

# Conventional shell exit code for SIGINT-style user abort (128 + SIGINT).
# Using a named constant keeps the magic number out of the body.
_USER_ABORT_EXIT = 130

# Bare-minimum chat completion sent during the reachability ping.
_PING_MAX_TOKENS = 1
_PING_MESSAGES = [{"role": "user", "content": "."}]

_RULE = "─" * 70


# ---- I/O helpers ---------------------------------------------------- #


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    raise ValueError(
        f"prompts file {path} must be a list or {{'cases': [...]}}"
    )


def _user_msg_snippet(messages: list[dict[str, Any]], width: int = 56) -> str:
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    txt = ""
    if last_user and isinstance(last_user.get("content"), str):
        txt = last_user["content"].strip().replace("\n", " ")
    if len(txt) > width:
        txt = txt[: width - 1] + "…"
    return txt


# ---- Client construction (lazy) ------------------------------------- #


def _build_fireworks_client(api_key: str, base_url: str | None):
    """Imported lazily so unit tests don't need the Fireworks SDK."""
    from fireworks import Fireworks  # type: ignore[import-not-found]  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return Fireworks(**kwargs)


def _build_tokenizer(tokenizer_model: str):
    """Imported lazily — pulls a HuggingFace tokenizer (can be 100s of MB).
    Forwards HF_TOKEN for gated repos; cached under ~/.cache/huggingface/."""
    from training.renderer.verifier.utils.tokenizer import load_tokenizer  # noqa: PLC0415

    return load_tokenizer(tokenizer_model)


def _check_serverless(client, model: str) -> tuple[bool, str]:
    """Ping the dispatch target with a 1-token completion. Returns
    ``(reachable, message)``. Costs roughly 1 input + 1 output token."""
    try:
        client.chat.completions.create(
            model=model,
            messages=_PING_MESSAGES,
            max_tokens=_PING_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — surface the gateway's actual error
        return False, f"{type(exc).__name__}: {exc}"
    return True, "reachable"


# ---- Pre-flight + confirm ------------------------------------------- #


def _print_preflight(
    *,
    renderer_name: str,
    tokenizer_model: str,
    dispatch_mode: str,
    model_resolved: str,
    serverless_status: tuple[bool, str],
    prompts: list[dict[str, Any]],
    prompts_path: Path,
) -> None:
    print()
    print(_RULE)
    print("  TRIAGE PRE-FLIGHT — review before live API calls")
    print(_RULE)

    # 1. Renderer
    from tinker_cookbook.renderers import is_renderer_registered  # noqa: PLC0415

    registered_text = (
        "registered ✓"
        if is_renderer_registered(renderer_name)
        else "NOT REGISTERED — check spelling against the cookbook's renderer registry"
    )
    reachable, reach_msg = serverless_status
    reach_text = "reachable ✓" if reachable else f"UNREACHABLE — {reach_msg}"
    print()
    print("  1. RENDERER")
    print(f"     name        : {renderer_name}")
    print(f"     status      : {registered_text}")
    print(f"     tokenizer   : {tokenizer_model}")
    print(f"     dispatch    : {dispatch_mode} → {model_resolved}")
    print(f"     ping        : {reach_text}")

    # 2. Prompts
    print()
    print(f"  2. PROMPTS — {len(prompts)} case(s)")
    print(f"     source      : {prompts_path}")
    for i, p in enumerate(prompts):
        name = p.get("name") or f"case-{i + 1}"
        snippet = _user_msg_snippet(p.get("messages", []))
        print(f"     [{i + 1:>2}] {name:<22}  {snippet!r}")

    print()
    print(_RULE)


def _confirm(question: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return True  # non-interactive (e.g. CI) → don't block
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            ans = input(question + suffix).strip().lower()
        except EOFError:
            return False
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


# ---- Probe loop ----------------------------------------------------- #


def _probe_all(
    *,
    prompts: list[dict[str, Any]],
    renderer_name: str,
    tokenizer,
    client,
    model_resolved: str,
    deployment_id: str | None,
    tokenizer_model: str,
    dispatch_mode: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        name = p.get("name") or f"case-{i + 1}"
        try:
            artifact = run_probe(
                renderer_name=renderer_name,
                tokenizer=tokenizer,
                client=client,
                model=model_resolved,
                messages=p["messages"],
                tools=p.get("tools"),
                max_tokens=max_tokens,
                temperature=0.0,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
                deployment_id=deployment_id,
                tokenizer_model=tokenizer_model,
                renderer_config=p.get("renderer_config") or {},
                dispatch_mode=dispatch_mode,
            )
        except Exception as exc:  # noqa: BLE001 — per-prompt resilience
            print(f"    [{i + 1:>2}] {name:<22}  ERROR: {exc}")
            continue
        cases.append({"index": i, "name": name, "artifact": artifact})
        print(f"    [{i + 1:>2}] {name:<22}  done")
    return cases


# ---- Entry point ---------------------------------------------------- #


def run_triage(
    *,
    renderer_name: str,
    tokenizer_model: str,
    prompts_path: str | Path,
    output_path: str,
    model: str | None = None,
    deployment_id: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 1024,
    skip_confirm: bool = False,
) -> int:
    src_path = Path(prompts_path)
    prompts = _load_prompts(src_path)

    model_resolved, dispatch_mode = resolve_dispatch(
        renderer_name=renderer_name,
        model=model,
        deployment_id=deployment_id,
        api_key=api_key,
        base_url=base_url,
    )

    # Fail-fast on auth before we waste cycles building heavy artifacts.
    api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit(
            "FIREWORKS_API_KEY not set. Pass --api-key or export FIREWORKS_API_KEY."
        )
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL")

    # Build the Fireworks client (lightweight) and ping the model so the
    # pre-flight reflects actual reachability — not just config.
    client = _build_fireworks_client(api_key, base_url)
    serverless_status = _check_serverless(client, model_resolved)

    _print_preflight(
        renderer_name=renderer_name,
        tokenizer_model=tokenizer_model,
        dispatch_mode=dispatch_mode,
        model_resolved=model_resolved,
        serverless_status=serverless_status,
        prompts=prompts,
        prompts_path=src_path,
    )

    if not serverless_status[0]:
        raise SystemExit(
            f"Aborting: dispatch target unreachable. {serverless_status[1]}"
        )

    if not skip_confirm and not _confirm(
        f"  Run probe across {len(prompts)} prompt(s) and write session to "
        f"{output_path}?"
    ):
        print("Aborted.", file=sys.stderr)
        return _USER_ABORT_EXIT

    print()
    print("  Loading tokenizer…")
    tokenizer = _build_tokenizer(tokenizer_model)

    print()
    print("  Probing…")
    cases = _probe_all(
        prompts=prompts,
        renderer_name=renderer_name,
        tokenizer=tokenizer,
        client=client,
        model_resolved=model_resolved,
        deployment_id=deployment_id,
        tokenizer_model=tokenizer_model,
        dispatch_mode=dispatch_mode,
        max_tokens=max_tokens,
    )

    session = {
        "kind": "probe-batch",
        "produced_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": str(src_path),
        "renderer": renderer_name,
        "tokenizer_model": tokenizer_model,
        "total_prompts": len(prompts),
        "case_count": len(cases),
        "cases": cases,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print()
    print(_RULE)
    print(f"  Batch complete: {len(cases)}/{len(prompts)} probed")
    print(f"  Session: {output_path}")
    print(_RULE)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.renderer.verifier.triage",
        description=(
            "Batch-probe a corpus of prompts and write a session JSON "
            "consumed by `serve.py --session-file` (which auto-seeds the GUI)."
        ),
    )
    p.add_argument("--renderer", required=True)
    p.add_argument("--tokenizer-model", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--deployment-id", default=None)
    p.add_argument(
        "--prompts",
        required=True,
        help='Path to a JSON prompt corpus: {"cases": [{"name": ..., "messages": [...]}, ...]}.',
    )
    p.add_argument(
        "--output",
        required=True,
        help="Session JSON path (read by serve.py --session-file).",
    )
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the pre-flight confirmation prompt.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        return run_triage(
            renderer_name=args.renderer,
            tokenizer_model=args.tokenizer_model,
            model=args.model,
            deployment_id=args.deployment_id,
            prompts_path=args.prompts,
            output_path=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            skip_confirm=args.yes,
        )
    except DispatchError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
