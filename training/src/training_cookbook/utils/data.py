"""Dataset loading, tokenization, and advantage computation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import torch
import requests

from fireworks.training.sdk.errors import request_with_retries

logger = logging.getLogger(__name__)


def load_jsonl_dataset(path_or_url: str, max_rows: int | None = None) -> List[Dict[str, Any]]:
    """Load a JSONL dataset from a local path or URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = request_with_retries(requests.get, path_or_url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
    else:
        with open(path_or_url) as f:
            lines = f.readlines()

    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if max_rows is not None and len(rows) >= max_rows:
            break
    logger.info("Loaded %d examples from dataset", len(rows))
    return rows


def load_preference_dataset(path: str, max_pairs: int | None = None) -> List[dict[str, Any]]:
    """Load preference dataset (chosen/rejected pairs)."""
    data = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "chosen" in row and "rejected" in row:
                data.append(row)
            elif "samples" in row:
                chosen = rejected = None
                for s in row["samples"]:
                    score = s.get("evals", {}).get("score", s.get("score"))
                    if score == 1.0:
                        chosen = s
                    elif score == 0.0:
                        rejected = s
                if chosen and rejected:
                    data.append({"chosen": chosen, "rejected": rejected})
            elif "preferred_output" in row and "non_preferred_output" in row:
                inp = row.get("input", {})
                if isinstance(inp, dict) and "messages" in inp:
                    input_msgs = inp["messages"]
                elif isinstance(inp, list):
                    input_msgs = inp
                elif isinstance(inp, str):
                    input_msgs = [{"role": "user", "content": inp}]
                else:
                    input_msgs = []

                def _to_msgs(v):
                    if isinstance(v, list):
                        return v
                    if isinstance(v, str):
                        return [{"role": "assistant", "content": v}]
                    return []

                data.append(
                    {
                        "chosen": {"messages": input_msgs + _to_msgs(row["preferred_output"])},
                        "rejected": {"messages": input_msgs + _to_msgs(row["non_preferred_output"])},
                    }
                )
            if max_pairs is not None and len(data) >= max_pairs:
                break
    return data


def extract_text(item: dict[str, Any]) -> str:
    """Extract text from a chosen/rejected preference item."""
    if "text" in item:
        return item["text"]
    if "messages" in item:
        parts = []
        for msg in item["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)
    return ""


def find_common_prefix_length(tokens1: List[int], tokens2: List[int]) -> int:
    """Find the length of the longest common prefix between two token lists."""
    min_len = min(len(tokens1), len(tokens2))
    for i in range(min_len):
        if tokens1[i] != tokens2[i]:
            return i
    return min_len


def encode_text(
    base_url: str,
    text: str,
    timeout: int = 60,
    max_wait_time: float = 300,
) -> List[int]:
    """Encode text to tokens using the RLOR trainer's tokenizer endpoint.

    Uses ``request_with_retries`` to tolerate trainer pods that are still
    initializing (model/checkpoint loading) after the job is reported ready.
    """
    resp = request_with_retries(
        requests.post,
        f"{base_url}/api/v1/tokenizer/encode",
        json={"text": text},
        timeout=timeout,
        verify=False,
        max_wait_time=max_wait_time,
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


def compute_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """Compute normalised advantages from a group of rewards."""
    t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = t.mean()
    std_r = t.std()
    if std_r < eps:
        std_r = torch.tensor(1.0)
    return ((t - mean_r) / (std_r + eps)).tolist()
