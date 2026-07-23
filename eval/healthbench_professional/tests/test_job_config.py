from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_job_uses_pinned_harbor_contract() -> None:
    job = yaml.safe_load((ROOT / "job.yaml").read_text(encoding="utf-8"))

    assert job["n_attempts"] == 8
    assert job["datasets"] == [{"path": "datasets/healthbench-professional"}]
    assert job["environment"]["type"] == "docker"
    assert "env" not in job["environment"]

    assert len(job["agents"]) == 1
    agent = job["agents"][0]
    assert agent["import_path"] == (
        "healthbench_professional.agent:HealthBenchProfessionalAgent"
    )
    assert agent["model_name"] == "accounts/fireworks/models/kimi-k2p6"
    assert agent["env"] == {"FIREWORKS_API_KEY": "${FIREWORKS_API_KEY}"}
    assert agent["kwargs"] == {
        "require_token_trajectory": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "request_timeout_seconds": 1800.0,
    }
    assert not any("reason" in key.lower() for key in agent["kwargs"])


def test_project_pins_compatible_public_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["requires-python"] == ">=3.12"
    assert "harbor>=0.19,<0.20" in project["dependencies"]
    assert "fireworks-ai>=1.2.0,<2" in project["dependencies"]
    assert any(item.startswith("huggingface-hub") for item in project["dependencies"])
    assert any(item.startswith("openai") for item in project["dependencies"])
    assert "pyyaml>=6,<7" in project["dependencies"]

    dev_dependencies = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["dependency-groups"]["dev"]
    assert "pytest-asyncio>=1.2,<2" in dev_dependencies


def test_task_template_keeps_judge_key_out_of_generated_content() -> None:
    task = tomllib.loads((ROOT / "templates/task.toml").read_text(encoding="utf-8"))

    assert task["schema_version"] == "1.3"
    assert task["metadata"]["dataset_revision"] == (
        "349962fd46dd02343a0d8a606491baf59154ea1a"
    )
    assert task["agent"]["timeout_sec"] == 2100.0
    assert task["verifier"]["env"]["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
    assert "OPENAI_API_KEY" not in task["environment"].get("env", {})


def test_environment_installs_openai_and_generated_outputs_are_ignored() -> None:
    dockerfile = (ROOT / "templates/environment/Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "python:3.12-slim" in dockerfile
    assert '"openai>=2,<3"' in dockerfile
    for path in ("/datasets/", "/jobs/", "/trials/", "/rl-output/"):
        assert path in ignored


def test_verifier_validates_exact_trajectory_before_judging() -> None:
    verifier = (ROOT / "templates/tests/test.sh").read_text(encoding="utf-8")

    validation = verifier.index("load_atif_trajectory")
    judge = verifier.index("/usr/local/bin/python /tests/judge.py")
    assert validation < judge
    assert "/logs/agent/trajectory.json" in verifier
    cleanup = verifier.index("/bin/rm -f")
    assert cleanup < validation
    assert "/logs/verifier/reward.json" in verifier[cleanup:validation]
    assert "/logs/verifier/healthbench_result.json" in verifier[cleanup:validation]
    assert "reward.txt" not in verifier
    assert 'trajectory["extra"]["messages"] != source["messages"]' in verifier
    assert 'answer_path.read_bytes() != exact.visible_text.encode("utf-8")' in verifier
    assert 'set(reward) == {"reward"}' in verifier
    assert "healthbench_result.json" in verifier
