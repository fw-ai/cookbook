from __future__ import annotations

import pytest

from training.utils import resource_autosizing


GIB = 1 << 30


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        ("128Gi", 128 * GIB),
        ("126669712Ki", 126669712 * 1024),
        ("1.5Gi", int(1.5 * GIB)),
        ("2G", 2_000_000_000),
        ("1e6", 1_000_000),
    ],
)
def test_parse_kubernetes_memory_quantity(quantity: str, expected: int) -> None:
    assert resource_autosizing.parse_kubernetes_memory_quantity(quantity) == expected


def test_parse_kubernetes_memory_quantity_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="invalid Kubernetes memory quantity"):
        resource_autosizing.parse_kubernetes_memory_quantity("lots")


def test_node_memory_from_api_payloads() -> None:
    allocatable, usage = resource_autosizing.node_memory_from_api_payloads(
        {"status": {"allocatable": {"memory": "120Gi"}}},
        {"usage": {"memory": "37Gi"}},
    )

    assert allocatable == 120 * GIB
    assert usage == 37 * GIB


def test_read_kubernetes_node_memory_tolerates_missing_metrics() -> None:
    requested_paths: list[str] = []

    def fetch_json(path: str) -> dict:
        requested_paths.append(path)
        if path.startswith("/api/v1/nodes/"):
            return {"status": {"allocatable": {"memory": "120Gi"}}}
        raise OSError("metrics API unavailable")

    allocatable, usage = resource_autosizing.read_kubernetes_node_memory(
        "node/one",
        fetch_json=fetch_json,
    )

    assert requested_paths == [
        "/api/v1/nodes/node%2Fone",
        "/apis/metrics.k8s.io/v1beta1/nodes/node%2Fone",
    ]
    assert allocatable == 120 * GIB
    assert usage == 0


def test_read_proc_memory(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       130000000 kB\n"
        "MemFree:          1000000 kB\n"
        "MemAvailable:    90000000 kB\n"
    )

    total, available = resource_autosizing.read_proc_memory(str(meminfo))

    assert total == 130000000 * 1024
    assert available == 90000000 * 1024


def test_choose_workers_uses_smallest_runtime_budget() -> None:
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=32,
        max_workers=16,
        cgroup_limit_bytes=232 * GIB,
        cgroup_current_bytes=20 * GIB,
        node_allocatable_bytes=120 * GIB,
        node_usage_bytes=40 * GIB,
        host_total_bytes=124 * GIB,
        host_available_bytes=80 * GIB,
    )

    # Hard budget: 80% of 120 GiB - 20 GiB current = 76 GiB.
    # Shared budget: 80 GiB available - 8 GiB emergency = 72 GiB.
    # At 8 GiB per worker, memory permits nine workers.
    assert workers == 9


def test_choose_workers_respects_cgroup_limit() -> None:
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=32,
        max_workers=16,
        cgroup_limit_bytes=58 * GIB,
        cgroup_current_bytes=10 * GIB,
        node_allocatable_bytes=120 * GIB,
        node_usage_bytes=40 * GIB,
        host_total_bytes=124 * GIB,
        host_available_bytes=80 * GIB,
    )

    assert workers == 4


def test_inc_1085_124_gib_node_regression() -> None:
    """Replay the memory snapshot that taint-evicted the SFT orchestrator.

    INC-1085 observed a ~124 GiB node, a 232 GiB container limit, ~29 GiB in
    the parent Python processes, and ~82 GiB across 16 rendering workers. The
    old fixed worker count reached roughly 121 GiB after host services and
    triggered the kernel OOM killer.
    """
    physical_memory = 124 * GIB
    parent_python = 29 * GIB
    host_services = 10 * GIB
    node_usage_before_workers = parent_python + host_services
    old_worker_total = 82 * GIB
    observed_memory_per_worker = old_worker_total / 16

    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=32,
        max_workers=16,
        cgroup_limit_bytes=232 * GIB,
        cgroup_current_bytes=parent_python,
        node_allocatable_bytes=physical_memory,
        node_usage_bytes=node_usage_before_workers,
        host_total_bytes=physical_memory,
        host_available_bytes=physical_memory - node_usage_before_workers,
    )

    projected_node_usage = (
        node_usage_before_workers + workers * observed_memory_per_worker
    )

    assert workers == 8
    assert projected_node_usage < physical_memory
    assert projected_node_usage + 8 * GIB < physical_memory


def test_inc_1085_fallback_still_caps_to_124_gib_physical_memory() -> None:
    """The /proc fallback protects the host when Node API access is absent."""
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=32,
        max_workers=16,
        cgroup_limit_bytes=232 * GIB,
        cgroup_current_bytes=29 * GIB,
        node_allocatable_bytes=0,
        node_usage_bytes=0,
        host_total_bytes=124 * GIB,
        host_available_bytes=85 * GIB,
    )

    assert workers == 8


def test_choose_workers_falls_back_without_memory_observations() -> None:
    assert resource_autosizing.choose_render_worker_count(
        cpu_count=2,
        max_workers=16,
    ) == 2


def test_choose_workers_keeps_one_worker_under_pressure() -> None:
    assert resource_autosizing.choose_render_worker_count(
        cpu_count=16,
        max_workers=16,
        cgroup_limit_bytes=16 * GIB,
        cgroup_current_bytes=15 * GIB,
    ) == 1


def test_select_workers_combines_runtime_sources(monkeypatch) -> None:
    monkeypatch.setenv("KUBERNETES_NODE_NAME", "node-one")
    monkeypatch.setattr(
        resource_autosizing,
        "read_kubernetes_node_memory",
        lambda node_name: (120 * GIB, 40 * GIB),
    )
    monkeypatch.setattr(
        resource_autosizing,
        "read_proc_memory",
        lambda: (124 * GIB, 80 * GIB),
    )
    monkeypatch.setattr(resource_autosizing, "read_cgroup_limit", lambda: 232 * GIB)
    monkeypatch.setattr(resource_autosizing, "read_cgroup_mem", lambda: 20 * GIB)
    monkeypatch.setattr(resource_autosizing.os, "cpu_count", lambda: 32)

    assert resource_autosizing.select_render_worker_count() == 9
