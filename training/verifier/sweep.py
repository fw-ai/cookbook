"""Sweep the empirical probe over a fixed catalogue of (renderer, model, input)
combinations and aggregate pass/fail signals against historical bug classes.

Use::

    FIREWORKS_API_KEY=... python -m training.verifier.sweep \\
        --output viz/probes/sweep-<date>/

The script writes one probe artifact per row plus a single ``summary.json``
that lines up the historical bug catalogue (renderer-pr-history.md) against
what the probe can mechanically observe today. It is a regression harness,
not a test runner — failures are reported, not asserted, and a row that
*should* fail (because the bug is currently present) becomes obvious in the
table.

The catalogue is intentionally restricted to renderers and models that are
reachable as Fireworks serverless endpoints from a stock dev API key, so
the sweep can be invoked from any workstation without spinning up
deployments. Renderers and bug classes that need a personal deployment or
multimodal endpoint are listed in ``UNREACHABLE_BUGS`` and reported as
"not covered by sweep".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from tinker_cookbook.renderers.base import TrainOnWhat

from training.verifier.probe import run_probe, write_artifact

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Case:
    """One row in the sweep catalogue."""

    case_id: str
    renderer: str
    tokenizer_model: str
    serverless_model: str
    messages: list[dict]
    tools: list[dict] | None = None
    bug_refs: tuple[str, ...] = ()
    expectation: str = "clean"  # "clean" or human note about expected anomaly


# Catalogue ---------------------------------------------------------------
#
# renderer + tokenizer + Fireworks serverless model id + a representative
# input. Each row is annotated with the historical bug class it lets us
# observe; the BUG_COVERAGE table at the bottom maps PR ids to which row(s)
# would surface a regression of that bug.

_CATALOGUE: list[_Case] = [
    _Case(
        case_id="glm5-single-turn",
        renderer="glm5",
        tokenizer_model="zai-org/GLM-5.1",
        serverless_model="accounts/fireworks/models/glm-5p1",
        messages=[
            {"role": "system", "content": "Answer with a single integer and nothing else."},
            {"role": "user", "content": "2 + 2 = ?"},
        ],
        bug_refs=("cookbook#384", "cookbook#389", "cookbook#400"),
    ),
    _Case(
        case_id="glm5-multi-turn",
        renderer="glm5",
        tokenizer_model="zai-org/GLM-5.1",
        serverless_model="accounts/fireworks/models/glm-5p1",
        messages=[
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "And 3+3?"},
        ],
        bug_refs=("cookbook#397", "cookbook#400"),
    ),
    _Case(
        case_id="qwen3-thinking-single-turn",
        renderer="qwen3",
        tokenizer_model="Qwen/Qwen3-8B",
        serverless_model="accounts/fireworks/models/qwen3-8b",
        messages=[
            {"role": "system", "content": "Answer with a single integer and nothing else."},
            {"role": "user", "content": "2 + 2 = ?"},
        ],
        bug_refs=("tinker#178", "tinker#247", "tinker#341"),
    ),
    _Case(
        case_id="qwen3-disable-thinking-single-turn",
        renderer="qwen3_disable_thinking",
        tokenizer_model="Qwen/Qwen3-8B",
        serverless_model="accounts/fireworks/models/qwen3-8b",
        messages=[
            {"role": "system", "content": "Answer with a single integer and nothing else."},
            {"role": "user", "content": "2 + 2 = ?"},
        ],
        bug_refs=("tinker#50", "tinker#178"),
    ),
    _Case(
        case_id="deepseekv3-single-turn",
        renderer="deepseekv3",
        tokenizer_model="deepseek-ai/DeepSeek-V3",
        serverless_model="accounts/fireworks/models/deepseek-v3p1",
        messages=[
            {"role": "system", "content": "Answer with a single integer and nothing else."},
            {"role": "user", "content": "2 + 2 = ?"},
        ],
        bug_refs=("tinker#139", "tinker#247"),
    ),
    _Case(
        case_id="deepseekv3-thinking-single-turn",
        renderer="deepseekv3_thinking",
        tokenizer_model="deepseek-ai/DeepSeek-V3",
        serverless_model="accounts/fireworks/models/deepseek-v3p1",
        messages=[
            {"role": "system", "content": "Answer with a single integer and nothing else."},
            {"role": "user", "content": "2 + 2 = ?"},
        ],
        bug_refs=("tinker#264", "tinker#267", "tinker#285"),
    ),
]


# Bugs the sweep cannot exercise from a stock serverless surface; recorded
# so the summary doesn't pretend coverage.
UNREACHABLE_BUGS: dict[str, str] = {
    "cookbook#266": "Nemotron3 — not on accessible serverless catalogue",
    "cookbook#280": "MiniMax M2 — not on accessible serverless catalogue",
    "cookbook#310": "Gemma4 — not on accessible serverless catalogue",
    "cookbook#359": "Kimi K2.5 vision — multimodal probe out of v1 scope",
    "cookbook#387": "normalize_messages weight=0 — pre-renderer bug, not at probe layer",
    "tinker#319": "Streaming parse — probe is batch-only",
    "tinker#349": "Custom tokenizer registry — not exercised by sweep",
    "tinker#382": "Renderer metadata persistence — out of probe scope",
    "tinker#510_family": "Auto-resolution selecting wrong renderer — orthogonal to probe",
}


def _signal_row(case_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    sanity = artifact.get("sanity", {}) or {}
    rows = artifact.get("audit_table", []) or []
    counts: dict[str, int] = {}
    for r in rows:
        prov = r.get("provenance") or "?"
        counts[prov] = counts.get(prov, 0) + 1

    # Heuristics flagging bug classes:
    # * prompt parity failure → HF parity bug
    # * tokenization_diverged_count > 0 → BPE / tokenizer drift
    # * any audit row with renderer_claim_weight=1.0 + provenance=trailing_hard_append
    #   → loss-mask bug class (e.g. GLM5 #400 pre-fix shape)
    # * parse_response_ok=False → parse round-trip bug
    # * structural_walk_token_match=False → renderer customization changed
    #   tokens, probe attribution unsafe
    suspect_trailing_w1 = sum(
        1
        for r in rows
        if r.get("provenance") == "trailing_hard_append"
        and (r.get("renderer_claim_weight") or 0) >= 0.999
    )
    return {
        "case_id": case_id,
        "renderer": artifact["renderer"]["name"],
        "model": artifact["deployment"]["model"],
        "stop_reason": sanity.get("completion_stop_reason"),
        "prompt_parity": sanity.get("renderer_prompt_matches_api_prompt"),
        "full_prefix_parity": sanity.get("full_render_prompt_prefix_matches_api"),
        "echo_stripped": sanity.get("echo_prompt_stripped"),
        "parse_response_ok": sanity.get("parse_response_ok"),
        "structural_walk_token_match": sanity.get("structural_walk_token_match"),
        "tokenization_diverged_count": sanity.get("tokenization_diverged_count"),
        "trailing_w1_count": suspect_trailing_w1,
        "provenance_counts": counts,
        "total_tokens": len(rows),
    }


def _verdict(signal: dict[str, Any]) -> str:
    """Coarse pass/warn/fail bucket per row."""
    if signal.get("prompt_parity") is False:
        return "FAIL: HF prompt parity"
    if signal.get("structural_walk_token_match") is False:
        return "FAIL: structural walk diverges"
    if signal.get("parse_response_ok") is False:
        return "WARN: parse_response_ok=False"
    if (signal.get("tokenization_diverged_count") or 0) > 0:
        return "WARN: tokenization diverged"
    if (signal.get("trailing_w1_count") or 0) > 0:
        return "WARN: trailing token w=1.0 not emitted (loss-mask suspect)"
    if signal.get("stop_reason") != "stop":
        return "INFO: completion truncated (stop_reason!=stop) — re-run with higher max_tokens"
    return "PASS"


def run_sweep(*, output_dir: str, api_key: str | None, base_url: str | None, max_tokens: int) -> int:
    api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY not set.")
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Lazy imports — the sweep needs the real client and HF tokenizer; the
    # core probe module stays import-light for unit tests.
    from fireworks import Fireworks  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = Fireworks(**client_kwargs)

    rows: list[dict[str, Any]] = []
    for case in _CATALOGUE:
        logger.info("=== %s (renderer=%s, model=%s) ===", case.case_id, case.renderer, case.serverless_model)
        artifact_path = out / f"{case.case_id}.json"
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                case.tokenizer_model, trust_remote_code=True,
            )
            artifact = run_probe(
                renderer_name=case.renderer,
                tokenizer=tokenizer,
                client=client,
                model=case.serverless_model,
                messages=case.messages,
                tools=case.tools,
                max_tokens=max_tokens,
                temperature=0.0,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
                tokenizer_model=case.tokenizer_model,
            )
            write_artifact(artifact, str(artifact_path))
            signal = _signal_row(case.case_id, artifact)
            signal["verdict"] = _verdict(signal)
            signal["bug_refs"] = list(case.bug_refs)
            signal["error"] = None
        except Exception as exc:  # noqa: BLE001 - sweep records failures, doesn't raise
            logger.exception("case %s failed", case.case_id)
            signal = {
                "case_id": case.case_id,
                "renderer": case.renderer,
                "model": case.serverless_model,
                "verdict": "ERROR",
                "bug_refs": list(case.bug_refs),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        rows.append(signal)
        # Concise per-row line for the operator running the sweep:
        logger.info("  → %s", signal.get("verdict"))

    summary = {
        "schema_version": 1,
        "kind": "sweep_summary",
        "rows": rows,
        "unreachable_bugs": UNREACHABLE_BUGS,
    }
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {summary_path}", file=sys.stderr)

    # Print a compact human-readable table to stdout for quick reading.
    print()
    print(f"{'case_id':<40}  {'verdict':<60}  {'bugs':<40}")
    print("-" * 145)
    for r in rows:
        print(f"{r['case_id']:<40}  {r['verdict']:<60}  {','.join(r.get('bug_refs') or [])}")
    print()
    print("Bugs not exercised by sweep (need other infra):")
    for k, v in UNREACHABLE_BUGS.items():
        print(f"  {k}: {v}")
    print()

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.verifier.sweep",
        description="Probe sweep over serverless-reachable renderer/model pairs.",
    )
    p.add_argument("--output", required=True, help="Directory to write probe JSONs and summary.json into.")
    p.add_argument("--api-key", default=None, help="Fireworks API key. Falls back to FIREWORKS_API_KEY.")
    p.add_argument("--base-url", default=None, help="Override Fireworks base URL (FIREWORKS_BASE_URL fallback).")
    p.add_argument("--max-tokens", type=int, default=512, help="Per-completion cap. Default 512 keeps the sweep fast; bump if a row trips stop_reason=length.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    return run_sweep(
        output_dir=args.output,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
