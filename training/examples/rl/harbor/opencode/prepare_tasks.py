"""Copy Harbor tasks and bake a pinned OpenCode CLI into each image."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from training.examples.rl.harbor.tito.prepare_tasks import (
    build_node_22_harness_suffix,
    prepare_with_installer,
    task_output_path,
)

from .constants import DEFAULT_OPENCODE_VERSION

# Preserve both pre-refactor markers so prepared trees remain idempotent.
_MARKER = "# Added by harbor_rl_opencode.prepare_opencode_tasks"
_TRANSITIONAL_MARKER = "# Added by harbor.opencode.prepare_tasks"
_UV_EXTERNAL_COPY = re.compile(
    r"(?im)^COPY\s+--from=ghcr[.]io/astral-sh/uv:(?P<version>[^\s]+)\s+"
    r"/uv\s+/uvx\s+/bin/\s*$"
)
_FINANCIAL_DOCUMENT_BUILD_COPY = "COPY --from=build /app/documents /app/documents"


def _suffix(version: str, restore_user: str | None) -> str:
    return build_node_22_harness_suffix(
        marker=_MARKER,
        package_install=f"npm install -g opencode-ai@{version}",
        version_check="opencode --version",
        restore_user=restore_user,
    )


def _replace_eol_bullseye_base(dockerfile: Path) -> None:
    """Use supported Debian for Terminal-Bench's two EOL Bullseye images."""
    source = dockerfile.read_text(encoding="utf-8")
    from_lines = list(re.finditer(r"(?im)^FROM\s+([^\n]+)$", source))
    if not from_lines:
        return
    final_from = from_lines[-1]
    fields = final_from.group(1).split()
    image_index = 1 if fields and fields[0].startswith("--platform=") else 0
    if image_index >= len(fields):
        return
    image = fields[image_index].lower()
    if not re.fullmatch(r"debian:bullseye(?:-slim)?", image):
        return
    replacement = (
        "debian:bookworm-slim" if image.endswith("-slim") else "debian:bookworm"
    )
    updated_from = re.sub(
        r"(?i)debian:bullseye(?:-slim)?",
        replacement,
        final_from.group(0),
        count=1,
    )
    updated = (
        source[: final_from.start()] + updated_from + source[final_from.end() :]
    )
    updated = re.sub(r"\bnetcat\b", "netcat-openbsd", updated)
    dockerfile.write_text(updated, encoding="utf-8")


def _replace_unsupported_external_uv_copy(dockerfile: Path) -> None:
    """Install the pinned uv wheel when E2B cannot parse external-stage COPY."""
    source = dockerfile.read_text(encoding="utf-8")

    def replacement(match: re.Match[str]) -> str:
        version = match.group("version")
        return (
            "RUN if python3 -m pip --version >/dev/null 2>&1; then \\\n"
            f"      python3 -m pip install --no-cache-dir uv=={version}; \\\n"
            "    else \\\n"
            "      apt-get update && \\\n"
            "      apt-get install -y --no-install-recommends python3-pip && \\\n"
            f"      python3 -m pip install --break-system-packages --no-cache-dir uv=={version} && \\\n"
            "      rm -rf /var/lib/apt/lists/*; \\\n"
            "    fi"
        )

    updated = _UV_EXTERNAL_COPY.sub(replacement, source)
    if updated != source:
        dockerfile.write_text(updated, encoding="utf-8")


def _flatten_financial_document_processor(dockerfile: Path) -> None:
    """Flatten Terminal-Bench's document stage for E2B's single-stage parser."""
    source = dockerfile.read_text(encoding="utf-8")
    from_lines = list(
        re.finditer(r"(?im)^FROM\s+([^\s]+)(?:\s+AS\s+\w+)?\s*$", source)
    )
    if len(from_lines) != 2 or _FINANCIAL_DOCUMENT_BUILD_COPY not in source:
        return
    if from_lines[0].group(1).lower() != from_lines[1].group(1).lower():
        raise ValueError(f"cannot flatten unlike Docker stages in {dockerfile}")
    if "RUN uv run /root/randomize_filenames.py" not in source:
        raise ValueError(f"unexpected financial-document-processor stages in {dockerfile}")

    first_from = from_lines[0]
    second_from = from_lines[1]
    prefix = source[: first_from.start()] + f"FROM {first_from.group(1)}"
    build_body = source[first_from.end() : second_from.start()]
    target_body = source[second_from.end() :]
    target_body = _UV_EXTERNAL_COPY.sub("", target_body, count=1)
    target_body = target_body.replace(_FINANCIAL_DOCUMENT_BUILD_COPY, "", 1)
    dockerfile.write_text(prefix + build_body + target_body, encoding="utf-8")


def prepare(
    source: Path,
    destination: Path,
    version: str = DEFAULT_OPENCODE_VERSION,
    *,
    base_image: str | None = None,
) -> list[Path]:
    prepared = prepare_with_installer(
        source,
        destination,
        marker=_MARKER,
        compatible_markers=(_TRANSITIONAL_MARKER,),
        suffix_builder=lambda restore_user: _suffix(version, restore_user),
        base_image=base_image,
    )
    for task in prepared:
        dockerfile = task_output_path(task, "environment", "Dockerfile")
        _replace_eol_bullseye_base(dockerfile)
        _flatten_financial_document_processor(dockerfile)
        _replace_unsupported_external_uv_copy(dockerfile)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument(
        "--base-image",
        default=None,
        help="Optional immutable image@sha256 base for a single prepared task",
    )
    args = parser.parse_args()
    prepared = prepare(
        args.source,
        args.destination,
        args.opencode_version,
        base_image=args.base_image,
    )
    print(
        f"prepared {len(prepared)} Harbor task image contexts in "
        f"{args.destination} with opencode-ai@{args.opencode_version}"
    )


if __name__ == "__main__":
    main()
