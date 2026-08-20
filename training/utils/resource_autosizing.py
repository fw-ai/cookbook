"""Runtime resource discovery and render-worker auto-sizing.

The orchestrator can run on different node types over time, so worker sizing
must not depend on a region-to-instance-memory table. This module combines:

* the container cgroup limit and current usage;
* the bound Kubernetes node's allocatable memory and current working set;
* ``/proc/meminfo`` as a Kubernetes-independent fallback.

Node usage is advisory because other pods may grow after it is sampled. The
hard ceiling is therefore the smallest known cgroup, node-allocatable, and
host-physical limit, with explicit utilization and emergency headroom.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from training.utils.memlog import read_cgroup_limit, read_cgroup_mem

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1 << 30
DEFAULT_AUTO_RENDER_WORKERS = 4
MAX_AUTO_RENDER_WORKERS = 16

# The OOM investigation observed roughly 5 GiB per rendering worker. Budget
# 8 GiB per worker to cover tokenizer/renderer variation and DataLoader
# prefetch without coupling the calculation to any particular node type.
DEFAULT_RENDER_WORKER_MEMORY_BYTES = 8 * BYTES_PER_GIB
DEFAULT_MEMORY_TARGET_UTILIZATION = 0.80
DEFAULT_NODE_EMERGENCY_HEADROOM_BYTES = 8 * BYTES_PER_GIB

_SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
_MEMORY_QUANTITY_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([EPTGMK]i|[EPTGMkK]|[num]|[eE][+-]?\d+)?$"
)
_MEMORY_SUFFIX_MULTIPLIERS = {
    "": Decimal(1),
    "n": Decimal("1e-9"),
    "u": Decimal("1e-6"),
    "m": Decimal("1e-3"),
    "k": Decimal(1000),
    "K": Decimal(1000),
    "M": Decimal(1000) ** 2,
    "G": Decimal(1000) ** 3,
    "T": Decimal(1000) ** 4,
    "P": Decimal(1000) ** 5,
    "E": Decimal(1000) ** 6,
    "Ki": Decimal(1024),
    "Mi": Decimal(1024) ** 2,
    "Gi": Decimal(1024) ** 3,
    "Ti": Decimal(1024) ** 4,
    "Pi": Decimal(1024) ** 5,
    "Ei": Decimal(1024) ** 6,
}


def parse_kubernetes_memory_quantity(value: str) -> int:
    """Parse a Kubernetes memory quantity and return bytes."""
    match = _MEMORY_QUANTITY_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid Kubernetes memory quantity: {value!r}")

    number, suffix = match.groups()
    try:
        if suffix and suffix[0] in ("e", "E"):
            return int(Decimal(number) * (Decimal(10) ** int(suffix[1:])))
        return int(Decimal(number) * _MEMORY_SUFFIX_MULTIPLIERS[suffix or ""])
    except (InvalidOperation, KeyError) as exc:
        raise ValueError(f"invalid Kubernetes memory quantity: {value!r}") from exc


def node_memory_from_api_payloads(
    node_payload: dict[str, Any],
    metrics_payload: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Return ``(allocatable_bytes, working_set_bytes)`` from API payloads."""
    allocatable_value = (
        node_payload.get("status", {}).get("allocatable", {}).get("memory", "")
    )
    allocatable_bytes = (
        parse_kubernetes_memory_quantity(allocatable_value)
        if allocatable_value
        else 0
    )

    usage_value = (
        metrics_payload.get("usage", {}).get("memory", "")
        if metrics_payload is not None
        else ""
    )
    usage_bytes = (
        parse_kubernetes_memory_quantity(usage_value) if usage_value else 0
    )
    return allocatable_bytes, usage_bytes


def _fetch_kubernetes_json(path: str, *, timeout_s: float = 1.0) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    token_path = _SERVICE_ACCOUNT_DIR / "token"
    ca_path = _SERVICE_ACCOUNT_DIR / "ca.crt"
    if not host or not token_path.exists() or not ca_path.exists():
        raise OSError("Kubernetes service account credentials are unavailable")

    token = token_path.read_text().strip()
    request = Request(
        url=f"https://{host}:{port}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    context = ssl.create_default_context(cafile=str(ca_path))
    with urlopen(request, context=context, timeout=timeout_s) as response:
        return json.load(response)


def read_kubernetes_node_memory(
    node_name: str,
    *,
    fetch_json: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Read bound-node allocatable memory and current working set.

    Metrics are optional: clusters without Metrics API access still return the
    allocatable ceiling. Any API or RBAC failure degrades to ``(0, 0)`` so
    training can fall back to cgroup and ``/proc`` information.
    """
    if not node_name:
        return 0, 0

    fetch = fetch_json or _fetch_kubernetes_json
    escaped_node_name = quote(node_name, safe="")
    try:
        node_payload = fetch(f"/api/v1/nodes/{escaped_node_name}")
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        logger.warning(
            "Unable to read Kubernetes node memory for %s: %s", node_name, exc
        )
        return 0, 0

    metrics_payload = None
    try:
        metrics_payload = fetch(
            f"/apis/metrics.k8s.io/v1beta1/nodes/{escaped_node_name}"
        )
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        logger.info(
            "Kubernetes node metrics unavailable for %s: %s", node_name, exc
        )

    try:
        return node_memory_from_api_payloads(node_payload, metrics_payload)
    except ValueError as exc:
        logger.warning("Invalid Kubernetes memory data for %s: %s", node_name, exc)
        return 0, 0


def read_proc_memory(path: str = "/proc/meminfo") -> tuple[int, int]:
    """Return ``(MemTotal, MemAvailable)`` in bytes, or zeroes on failure."""
    values: dict[str, int] = {}
    try:
        with open(path) as meminfo:
            for line in meminfo:
                key, separator, remainder = line.partition(":")
                if not separator or key not in ("MemTotal", "MemAvailable"):
                    continue
                fields = remainder.split()
                if not fields:
                    continue
                multiplier = 1024 if len(fields) == 1 or fields[1] == "kB" else 1
                values[key] = int(fields[0]) * multiplier
    except (OSError, ValueError):
        return 0, 0
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def choose_render_worker_count(
    *,
    cpu_count: int,
    max_workers: int,
    cgroup_limit_bytes: int = 0,
    cgroup_current_bytes: int = 0,
    node_allocatable_bytes: int = 0,
    node_usage_bytes: int = 0,
    host_total_bytes: int = 0,
    host_available_bytes: int = 0,
    memory_per_worker_bytes: int = DEFAULT_RENDER_WORKER_MEMORY_BYTES,
    target_utilization: float = DEFAULT_MEMORY_TARGET_UTILIZATION,
    emergency_headroom_bytes: int = DEFAULT_NODE_EMERGENCY_HEADROOM_BYTES,
) -> int:
    """Choose a startup worker count from the smallest known memory budget.

    The function is intentionally pure so sizing policy can be tested without
    a Kubernetes cluster. At least one worker is returned; callers that need
    to disable multiprocessing can continue to set ``render_workers=0`` or 1
    explicitly.
    """
    if cpu_count < 1 or max_workers < 1:
        raise ValueError("cpu_count and max_workers must be positive")
    if memory_per_worker_bytes <= 0:
        raise ValueError("memory_per_worker_bytes must be positive")
    if not 0 < target_utilization <= 1:
        raise ValueError("target_utilization must be in (0, 1]")

    cpu_bound = min(cpu_count, max_workers)
    hard_ceilings = [
        value
        for value in (cgroup_limit_bytes, node_allocatable_bytes, host_total_bytes)
        if value > 0
    ]

    memory_budgets: list[int] = []
    if hard_ceilings:
        hard_ceiling = min(hard_ceilings)
        target_bytes = int(hard_ceiling * target_utilization)
        memory_budgets.append(max(0, target_bytes - cgroup_current_bytes))

    shared_available = [value for value in (host_available_bytes,) if value > 0]
    if node_allocatable_bytes > 0 and node_usage_bytes > 0:
        shared_available.append(max(0, node_allocatable_bytes - node_usage_bytes))
    if shared_available:
        memory_budgets.append(
            max(0, min(shared_available) - emergency_headroom_bytes)
        )

    if not memory_budgets:
        return min(cpu_bound, DEFAULT_AUTO_RENDER_WORKERS)

    available_for_workers = min(memory_budgets)
    memory_bound = max(1, available_for_workers // memory_per_worker_bytes)
    return min(cpu_bound, memory_bound)


def select_render_worker_count(
    *,
    max_workers: int = MAX_AUTO_RENDER_WORKERS,
) -> int:
    """Discover runtime memory and choose the SFT rendering concurrency."""
    node_name = os.environ.get("KUBERNETES_NODE_NAME", "")
    node_allocatable, node_usage = read_kubernetes_node_memory(node_name)
    host_total, host_available = read_proc_memory()
    cgroup_limit = read_cgroup_limit()
    cgroup_current = read_cgroup_mem()
    cpu_count = os.cpu_count() or 1

    workers = choose_render_worker_count(
        cpu_count=cpu_count,
        max_workers=max_workers,
        cgroup_limit_bytes=cgroup_limit,
        cgroup_current_bytes=cgroup_current,
        node_allocatable_bytes=node_allocatable,
        node_usage_bytes=node_usage,
        host_total_bytes=host_total,
        host_available_bytes=host_available,
    )
    logger.info(
        "Auto-selected %d render workers: cpu=%d max=%d cgroup=%.1f/%.1f GiB "
        "node=%.1f/%.1f GiB host_available=%.1f/%.1f GiB",
        workers,
        cpu_count,
        max_workers,
        cgroup_current / BYTES_PER_GIB,
        cgroup_limit / BYTES_PER_GIB,
        node_usage / BYTES_PER_GIB,
        node_allocatable / BYTES_PER_GIB,
        host_available / BYTES_PER_GIB,
        host_total / BYTES_PER_GIB,
    )
    return workers
