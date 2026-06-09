from __future__ import annotations

from training.examples.rl.coding_agent.rollout import _metadata
from training.examples.rl.coding_agent.swebench_harness import (
    filter_patch,
    normalize_harness_instance,
)
from training.examples.rl.coding_agent.swegym_data import (
    registry_image_for_instance_id,
    row_for_instance,
)


def test_registry_image_for_swegym_instance():
    image = registry_image_for_instance_id("getmoto__moto-7365")

    assert image == "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest"


def test_registry_image_for_legacy_swebench_instance():
    image = registry_image_for_instance_id("pydicom__pydicom-1234")

    assert image == "swebench/sweb.eval.x86_64.pydicom_1776_pydicom-1234:latest"


def test_row_for_instance_is_coding_agent_compatible():
    row = row_for_instance(
        {
            "instance_id": "getmoto__moto-7365",
            "problem_statement": "fix decimal math",
            "base_commit": "abc",
            "FAIL_TO_PASS": '["test_a"]',
            "PASS_TO_PASS": ["test_b"],
        }
    )

    metadata = row["metadata"]
    assert row["prompt"] == [{"role": "user", "content": "fix decimal math"}]
    assert metadata["workdir"] == "/testbed"
    assert metadata["swebench_instance"]["FAIL_TO_PASS"] == ["test_a"]


def test_rollout_metadata_accepts_prorl_swegym_row():
    row = {
        "prompt": [{"role": "user", "content": "fix issue"}],
        "metadata": {
            "instance_id": "getmoto__moto-7365",
            "instance": {"instance_id": "getmoto__moto-7365", "problem_statement": "fix issue"},
        },
    }

    metadata = _metadata(row)

    assert metadata["workdir"] == "/testbed"
    assert metadata["problem_statement"] == "fix issue"
    assert metadata["image"].startswith("xingyaoww/sweb.eval.x86_64.")
    assert metadata["swebench_instance"]["instance_id"] == "getmoto__moto-7365"


def test_filter_patch_drops_agent_config_sections():
    patch = """diff --git a/.claude/settings.json b/.claude/settings.json
--- a/.claude/settings.json
+++ b/.claude/settings.json
@@ -0,0 +1 @@
+{}
diff --git a/pkg/module.py b/pkg/module.py
--- a/pkg/module.py
+++ b/pkg/module.py
@@ -1 +1 @@
-old
+new
"""

    filtered = filter_patch(patch)

    assert ".claude/settings.json" not in filtered
    assert "pkg/module.py" in filtered


def test_normalize_harness_instance_sets_version_and_lists():
    instance = normalize_harness_instance(
        {
            "instance_id": "Owner__Repo-1",
            "base_commit": "abc",
            "FAIL_TO_PASS": '["test_x"]',
            "PASS_TO_PASS": '["test_y"]',
        }
    )

    assert instance["instance_id"] == "owner__repo-1"
    assert instance["version"] == "abc"
    assert instance["FAIL_TO_PASS"] == ["test_x"]
    assert instance["PASS_TO_PASS"] == ["test_y"]
