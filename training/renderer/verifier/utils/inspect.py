"""Pretty-print a probe artifact for spec-author review.

Reading the audit table by hand from raw JSON is tedious. This module
renders the same fields a human cares about — sanity flags, provenance
counts, and a compact audit table view — to stdout. It is the
L3-renderer precursor: text-only, no GUI, but structured so the same
input JSON can drive a richer renderer later.

Usage::

    python -m training.renderer.verifier inspect path/to/probe.json
    python -m training.renderer.verifier inspect path/to/probe.json --all
    python -m training.renderer.verifier inspect path/to/probe.json --filter trailing_hard_append
"""

from __future__ import annotations

import json
from typing import Any

# Provenance vocab — duplicated literal (rather than imported) because
# the inspector should be readable as a standalone script and not pull
# in the rest of the probe runtime.
_INTERESTING_PROVENANCE = {
    "trailing_hard_append",
    "tokenization_diverged",
}
_INTERESTING_CHUNK_SOURCES = {
    "bos",
    "header",
    "stop_overlap",
    "generation_suffix",
}


def _short(s: str, limit: int = 22) -> str:
    """Truncate a string with an ellipsis so audit-table columns stay aligned."""
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _format_row(row: dict[str, Any]) -> str:
    decoded = repr(row.get("decoded") or "")
    return (
        f"  idx={row['idx']:>4}  "
        f"tok={row['token_id']:>7}  "
        f"decoded={_short(decoded):>24}  "
        f"src={row.get('chunk_source', ''):<17}  "
        f"role={(row.get('role') or '-'):<10}  "
        f"w={row.get('renderer_claim_weight', 0):.1f}  "
        f"prov={row.get('provenance', '')}"
    )


def _select_rows(rows: list[dict[str, Any]], *, show_all: bool, filter_prov: str | None):
    if show_all:
        return rows
    if filter_prov:
        return [r for r in rows if r.get("provenance") == filter_prov]
    # Default: structurally interesting rows (chunk boundaries, every
    # special-token-source row, every empirically-suspect provenance).
    selected: list[dict[str, Any]] = []
    prev_src: str | None = None
    prev_prov: str | None = None
    for i, row in enumerate(rows):
        src = row.get("chunk_source")
        prov = row.get("provenance")
        boundary = src != prev_src or prov != prev_prov
        interesting = (
            src in _INTERESTING_CHUNK_SOURCES
            or prov in _INTERESTING_PROVENANCE
            or boundary
            or i == 0
            or i == len(rows) - 1
        )
        if interesting:
            selected.append(row)
        prev_src = src
        prev_prov = prov
    return selected


def render_inspect(artifact: dict[str, Any], *, show_all: bool, filter_prov: str | None) -> str:
    out_lines: list[str] = []
    a = artifact

    out_lines.append(f"PROBE  schema_version={a.get('schema_version')}  produced_at={a.get('produced_at')}")
    r = a.get("renderer", {}) or {}
    d = a.get("deployment", {}) or {}
    out_lines.append(f"  renderer: {r.get('name')}  config={r.get('config') or {}}  train_on_what={r.get('train_on_what')}")
    out_lines.append(f"  deployment.model: {d.get('model')}")
    if d.get("deployment_id"):
        out_lines.append(f"  deployment.id:    {d['deployment_id']}")
    samp = d.get("sampling") or {}
    out_lines.append(f"  sampling: temperature={samp.get('temperature')}  max_tokens={samp.get('max_tokens')}")

    sanity = a.get("sanity", {}) or {}
    out_lines.append("")
    out_lines.append("SANITY")
    for key in (
        "renderer_prompt_matches_api_prompt",
        "full_render_prompt_prefix_matches_api",
        "renderer_prompt_len",
        "api_prompt_len",
        "tokenization_diverged_count",
        "echo_prompt_stripped",
        "parse_response_ok",
        "completion_token_count",
        "completion_stop_reason",
    ):
        if key in sanity:
            out_lines.append(f"  {key}: {sanity[key]}")

    rows = a.get("audit_table", []) or []
    counts: dict[str, int] = {}
    for row in rows:
        prov = row.get("provenance") or "<unknown>"
        counts[prov] = counts.get(prov, 0) + 1
    out_lines.append("")
    out_lines.append(f"PROVENANCE COUNTS  total_rows={len(rows)}")
    for prov, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        out_lines.append(f"  {prov}: {n}")

    interesting = _select_rows(rows, show_all=show_all, filter_prov=filter_prov)
    title = "AUDIT TABLE (all rows)" if show_all else (
        f"AUDIT TABLE (filter={filter_prov!r})" if filter_prov
        else "AUDIT TABLE (boundaries + special chunks + suspect provenance)"
    )
    out_lines.append("")
    out_lines.append(f"{title}  shown={len(interesting)}/{len(rows)}")
    for row in interesting:
        out_lines.append(_format_row(row))

    out_lines.append("")
    return "\n".join(out_lines)


def run_inspect(path: str, *, show_all: bool, filter_prov: str | None) -> int:
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    print(render_inspect(artifact, show_all=show_all, filter_prov=filter_prov))
    return 0
