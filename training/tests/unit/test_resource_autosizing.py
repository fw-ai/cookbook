from __future__ import annotations

import pytest

from training.utils import resource_autosizing


GIB = 1 << 30


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


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("800000", "100000", 8),
        ("750000", "100000", 7),
        ("50000", "100000", 1),
        ("max", "100000", 0),
        ("-1", "100000", 0),
    ],
)
def test_parse_cgroup_cpu_quota(quota: str, period: str, expected: int) -> None:
    assert resource_autosizing.parse_cgroup_cpu_quota(quota, period) == expected


def test_read_cgroup_v2_cpu_quota(tmp_path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("800000 100000\n")

    assert resource_autosizing.read_cgroup_cpu_quota(
        v2_cpu_max_path=str(cpu_max),
        v1_quota_path=str(tmp_path / "missing-quota"),
        v1_period_path=str(tmp_path / "missing-period"),
    ) == 8


def test_read_cgroup_v1_cpu_quota(tmp_path) -> None:
    quota_path = tmp_path / "cpu.cfs_quota_us"
    period_path = tmp_path / "cpu.cfs_period_us"
    quota_path.write_text("1600000\n")
    period_path.write_text("100000\n")

    assert resource_autosizing.read_cgroup_cpu_quota(
        v2_cpu_max_path=str(tmp_path / "missing-cpu.max"),
        v1_quota_path=str(quota_path),
        v1_period_path=str(period_path),
    ) == 16


def test_effective_cpu_count_uses_smallest_runtime_bound() -> None:
    assert resource_autosizing.choose_effective_cpu_count(
        cgroup_quota_count=8,
        affinity_count=32,
        host_count=32,
    ) == 8


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [
        (8, 8),
        (16, 16),
    ],
)
def test_choose_workers_respects_eight_and_sixteen_cpu_limits(
    cpu_count: int,
    expected: int,
) -> None:
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=cpu_count,
        max_workers=16,
        cgroup_limit_bytes=256 * GIB,
        cgroup_current_bytes=8 * GIB,
        host_total_bytes=256 * GIB,
    )

    assert workers == expected


def test_choose_workers_preserves_historical_58_gib_four_worker_budget() -> None:
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=16,
        max_workers=16,
        cgroup_limit_bytes=58 * GIB,
        cgroup_current_bytes=8 * GIB,
        host_total_bytes=124 * GIB,
    )

    # (58 GiB - 8 GiB base - 8 GiB reserve) / 10 GiB = 4 workers.
    assert workers == 4


def test_inc_1085_124_gib_node_regression() -> None:
    """A 232 GiB Pod limit cannot override the 124 GiB physical ceiling."""
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=16,
        max_workers=16,
        cgroup_limit_bytes=232 * GIB,
        cgroup_current_bytes=29 * GIB,
        host_total_bytes=124 * GIB,
    )

    assert workers == 8


def test_choose_workers_caps_to_available_parallel_tasks() -> None:
    workers = resource_autosizing.choose_render_worker_count(
        cpu_count=16,
        max_workers=16,
        available_parallel_tasks=3,
        cgroup_limit_bytes=256 * GIB,
        host_total_bytes=256 * GIB,
    )

    assert workers == 3


def test_choose_workers_falls_back_without_memory_observations() -> None:
    assert resource_autosizing.choose_render_worker_count(
        cpu_count=16,
        max_workers=16,
    ) == 4


def test_choose_workers_keeps_one_worker_under_pressure() -> None:
    assert resource_autosizing.choose_render_worker_count(
        cpu_count=16,
        max_workers=16,
        cgroup_limit_bytes=16 * GIB,
        cgroup_current_bytes=15 * GIB,
    ) == 1


def test_select_workers_combines_cpu_and_memory_runtime_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        resource_autosizing,
        "read_proc_memory",
        lambda: (124 * GIB, 85 * GIB),
    )
    monkeypatch.setattr(resource_autosizing, "read_cgroup_limit", lambda: 232 * GIB)
    monkeypatch.setattr(resource_autosizing, "read_cgroup_mem", lambda: 29 * GIB)
    monkeypatch.setattr(resource_autosizing, "read_cgroup_cpu_quota", lambda: 8)
    monkeypatch.setattr(resource_autosizing, "read_cpu_affinity_count", lambda: 32)
    monkeypatch.setattr(resource_autosizing.os, "cpu_count", lambda: 32)

    assert resource_autosizing.select_render_worker_count() == 8
