"""SWE-Gym dataset helpers for the coding-agent example.

Adapted from NVIDIA-NeMo/ProRL-Agent-Server
``examples/swegym_slime_grpo/sample_tasks.py`` and ``prepare_data.py``.  The
dataset, prompt shape, and image naming scheme are public; this module keeps the
Fireworks cookbook example on the same 293-task SWE-Gym split as the ProRL
training example.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_NAME = "NovaSky-AI/SkyRL-v0-293-data"
DATASET_SPLITS = ("train",)
EXPECTED_SPLIT_SIZES = {"train": 293}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "coding_agent"
LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pvlib/pvlib-python",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}


def registry_image_for_instance_id(instance_id: str) -> str:
    """Return the public SWE-Gym/SWE-bench eval image for ``instance_id``."""
    if "__" not in instance_id:
        raise ValueError(f"instance_id must contain '__': {instance_id!r}")
    owner, repo_with_issue = instance_id.split("__", 1)
    if "-" not in repo_with_issue:
        raise ValueError(f"instance_id must end with '-<issue_id>': {instance_id!r}")
    repo, issue_id = repo_with_issue.rsplit("-", 1)
    if f"{owner}/{repo}" in LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{repo}-{issue_id}:latest"

    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def prompt_text(row: dict[str, Any]) -> str:
    """Extract plain instruction text from a ProRL or cookbook row."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
        return "\n\n".join(parts).strip()
    return str(prompt or row.get("input") or "").strip()


def normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON-encoded test lists in SWE-Gym/SWE-bench instances."""
    normalized = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                pass
    return normalized


def cache_path_for_split(split: str) -> Path:
    if split not in DATASET_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {DATASET_SPLITS}")
    return DEFAULT_CACHE_DIR / f"swegym_skyrl_v0_293_{split}.json"


def load_split_from_huggingface(split: str) -> list[dict[str, Any]]:
    """Download the public SWE-Gym split from Hugging Face."""
    if split not in DATASET_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {DATASET_SPLITS}")

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=f"{split}.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    instances: list[dict[str, Any]] = []
    for row in table.to_pylist():
        instance = row.get("instance")
        if not isinstance(instance, dict):
            raise ValueError(f"dataset row in split {split!r} does not contain an instance dict")
        instances.append(normalize_instance(instance))
    return instances


def fetch_dataset_instances(
    split: str,
    *,
    refresh: bool = False,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load SWE-Gym instances from cache, downloading them when needed."""
    if split not in DATASET_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {DATASET_SPLITS}")
    cache_file = cache_path or cache_path_for_split(split)
    if cache_file.exists() and not refresh:
        instances = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        instances = load_split_from_huggingface(split)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(instances, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

    expected = EXPECTED_SPLIT_SIZES.get(split)
    if expected is not None and len(instances) != expected:
        raise RuntimeError(f"expected {expected} SWE-Gym {split} instances, got {len(instances)}")
    return instances


def row_for_instance(instance: dict[str, Any], split: str = "train") -> dict[str, Any]:
    """Build a cookbook-compatible row while preserving ProRL's metadata."""
    normalized = normalize_instance(instance)
    if "instance_id" not in normalized:
        raise ValueError("SWE-Gym instance missing instance_id")
    if "problem_statement" not in normalized:
        raise ValueError("SWE-Gym instance missing problem_statement")
    instance_id = str(normalized["instance_id"])
    problem_statement = str(normalized["problem_statement"]).strip()
    return {
        "id": instance_id,
        "prompt": [{"role": "user", "content": problem_statement}],
        "metadata": {
            "instance_id": instance_id,
            "image": registry_image_for_instance_id(instance_id),
            "workdir": "/testbed",
            "problem_statement": problem_statement,
            "swebench_instance": normalized,
            # Preserve ProRL's original shape for traceability / compatibility.
            "instance": normalized,
            "split": split,
        },
    }
