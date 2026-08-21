"""Runtime resource discovery and render-worker auto-sizing.

The orchestrator can run on different node types over time, so worker sizing
must not depend on a region-to-instance resource table. Rendering concurrency
is bounded by both of the resources that workers consume:

* CPU parallelism from the container cgroup quota and process affinity;
* memory from the smaller cgroup and host-physical hard ceiling.

``MemAvailable`` is logged for diagnosis, but is not treated as a stable,
exclusively allocatable budget. Kubernetes scheduling and Pod requests own
that admission-control decision; the runtime policy reserves explicit base
and safety memory inside the hard ceiling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from training.utils.memlog import read_cgroup_limit, read_cgroup_mem

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1 << 30
DEFAULT_AUTO_RENDER_WORKERS = 4
MAX_AUTO_RENDER_WORKERS = 16

# The original four-worker orchestrator request was sized as
# (8 GiB base + 4 * 10 GiB/worker) * 1.2 ~= 58 GiB. Keep those conservative
# historical assumptions explicit instead of inferring capacity from the
# later, unrelated 16-vCPU default.
DEFAULT_ORCHESTRATOR_BASE_MEMORY_BYTES = 8 * BYTES_PER_GIB
DEFAULT_RENDER_WORKER_MEMORY_BYTES = 10 * BYTES_PER_GIB
DEFAULT_MEMORY_SAFETY_RESERVE_BYTES = 8 * BYTES_PER_GIB

_CGROUP_V2_CPU_MAX_PATH = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_CPU_QUOTA_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_CPU_PERIOD_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


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


def parse_cgroup_cpu_quota(quota_value: str, period_value: str) -> int:
    """Convert cgroup CPU quota/period values to whole worker slots.

    Zero means the quota is unlimited or unavailable. Fractional quotas are
    rounded down because starting more CPU-bound workers than the guaranteed
    parallelism only adds contention; a positive quota always permits at
    least one worker.
    """
    if quota_value == "max":
        return 0
    quota = int(quota_value)
    period = int(period_value)
    if quota <= 0:
        return 0
    if period <= 0:
        raise ValueError("cgroup CPU period must be positive")
    return max(1, quota // period)


def read_cgroup_cpu_quota(
    *,
    v2_cpu_max_path: str = _CGROUP_V2_CPU_MAX_PATH,
    v1_quota_path: str = _CGROUP_V1_CPU_QUOTA_PATH,
    v1_period_path: str = _CGROUP_V1_CPU_PERIOD_PATH,
) -> int:
    """Return whole CPUs allowed by cgroup v2/v1, or zero when unlimited."""
    try:
        fields = Path(v2_cpu_max_path).read_text().split()
        if len(fields) != 2:
            raise ValueError("cpu.max must contain quota and period")
        return parse_cgroup_cpu_quota(fields[0], fields[1])
    except (OSError, ValueError):
        pass

    try:
        quota = Path(v1_quota_path).read_text().strip()
        period = Path(v1_period_path).read_text().strip()
        return parse_cgroup_cpu_quota(quota, period)
    except (OSError, ValueError):
        return 0


def read_cpu_affinity_count() -> int:
    """Return CPUs allowed by process affinity, or zero when unsupported."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return 0


def choose_effective_cpu_count(
    *,
    cgroup_quota_count: int,
    affinity_count: int,
    host_count: int,
) -> int:
    """Return the smallest positive runtime CPU bound."""
    cpu_bounds = [
        value
        for value in (cgroup_quota_count, affinity_count, host_count)
        if value > 0
    ]
    return min(cpu_bounds) if cpu_bounds else 1


def choose_render_worker_count(
    *,
    cpu_count: int,
    max_workers: int,
    available_parallel_tasks: int | None = None,
    cgroup_limit_bytes: int = 0,
    cgroup_current_bytes: int = 0,
    host_total_bytes: int = 0,
    orchestrator_base_memory_bytes: int = DEFAULT_ORCHESTRATOR_BASE_MEMORY_BYTES,
    memory_per_worker_bytes: int = DEFAULT_RENDER_WORKER_MEMORY_BYTES,
    safety_reserve_bytes: int = DEFAULT_MEMORY_SAFETY_RESERVE_BYTES,
) -> int:
    """Choose a worker count bounded by runtime CPU, memory, and useful work.

    The function is intentionally pure so sizing policy can be tested without
    a Kubernetes cluster. At least one worker is returned; callers that need
    to disable multiprocessing can continue to set ``render_workers=0`` or 1
    explicitly.
    """
    if cpu_count < 1 or max_workers < 1:
        raise ValueError("cpu_count and max_workers must be positive")
    if available_parallel_tasks is not None and available_parallel_tasks < 0:
        raise ValueError("available_parallel_tasks must not be negative")
    if orchestrator_base_memory_bytes < 0 or safety_reserve_bytes < 0:
        raise ValueError("memory reserves must not be negative")
    if memory_per_worker_bytes <= 0:
        raise ValueError("memory_per_worker_bytes must be positive")

    useful_work_bound = (
        max(1, available_parallel_tasks)
        if available_parallel_tasks is not None
        else max_workers
    )
    cpu_bound = min(cpu_count, max_workers, useful_work_bound)
    hard_ceilings = [
        value
        for value in (cgroup_limit_bytes, host_total_bytes)
        if value > 0
    ]

    if not hard_ceilings:
        return min(cpu_bound, DEFAULT_AUTO_RENDER_WORKERS)

    hard_ceiling = min(hard_ceilings)
    base_memory = max(cgroup_current_bytes, orchestrator_base_memory_bytes)
    available_for_workers = max(
        0,
        hard_ceiling - base_memory - safety_reserve_bytes,
    )
    memory_bound = max(1, available_for_workers // memory_per_worker_bytes)
    return min(cpu_bound, memory_bound)


def select_render_worker_count(
    *,
    max_workers: int = MAX_AUTO_RENDER_WORKERS,
    available_parallel_tasks: int | None = None,
) -> int:
    """Discover runtime resources and choose rendering concurrency."""
    host_total, host_available = read_proc_memory()
    cgroup_limit = read_cgroup_limit()
    cgroup_current = read_cgroup_mem()
    cgroup_cpu_count = read_cgroup_cpu_quota()
    affinity_cpu_count = read_cpu_affinity_count()
    host_cpu_count = os.cpu_count() or 1
    cpu_count = choose_effective_cpu_count(
        cgroup_quota_count=cgroup_cpu_count,
        affinity_count=affinity_cpu_count,
        host_count=host_cpu_count,
    )

    workers = choose_render_worker_count(
        cpu_count=cpu_count,
        max_workers=max_workers,
        available_parallel_tasks=available_parallel_tasks,
        cgroup_limit_bytes=cgroup_limit,
        cgroup_current_bytes=cgroup_current,
        host_total_bytes=host_total,
    )
    logger.info(
        "Auto-selected %d render workers: cpu=%d (quota=%d affinity=%d host=%d) "
        "max=%d tasks=%s cgroup=%.1f/%.1f GiB "
        "host_available=%.1f/%.1f GiB base=%.1f GiB reserve=%.1f GiB "
        "per_worker=%.1f GiB",
        workers,
        cpu_count,
        cgroup_cpu_count,
        affinity_cpu_count,
        host_cpu_count,
        max_workers,
        available_parallel_tasks if available_parallel_tasks is not None else "unknown",
        cgroup_current / BYTES_PER_GIB,
        cgroup_limit / BYTES_PER_GIB,
        host_available / BYTES_PER_GIB,
        host_total / BYTES_PER_GIB,
        DEFAULT_ORCHESTRATOR_BASE_MEMORY_BYTES / BYTES_PER_GIB,
        DEFAULT_MEMORY_SAFETY_RESERVE_BYTES / BYTES_PER_GIB,
        DEFAULT_RENDER_WORKER_MEMORY_BYTES / BYTES_PER_GIB,
    )
    return workers
