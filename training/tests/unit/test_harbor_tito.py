from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import time
import zipfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fireworks.training.sdk import (
    TITOMetricSummary,
    TITOTrajectoryArtifact,
)
from training.examples.rl.harbor.mini_swe import prepare_tasks as prepare_mini_swe_tasks
from training.examples.rl.harbor.mini_swe import rollout as mini_swe_rollout
from training.examples.rl.harbor.opencode import prepare_tasks as prepare_opencode_tasks
from training.examples.rl.harbor.opencode import rollout
from training.examples.rl.harbor.recipes import train_opencode as opencode_train
from training.examples.rl.harbor.opencode.config import _TOOL_TIMEOUT_PLUGIN
from training.examples.rl.harbor.opencode.constants import (
    DEFAULT_OPENCODE_VERSION,
    OPENCODE_HARBOR_IMPORT_PATH,
)
from training.examples.rl.harbor.pi import prepare_tasks as prepare_pi_tasks
from training.examples.rl.harbor.pi import rollout as pi_rollout
from training.examples.rl.harbor.recipes.deep_swe import (
    prepare_tasks as prepare_deep_swe_tasks,
)
from training.examples.rl.harbor.tito import sidecar as sidecar_runtime
from training.examples.rl.harbor.tito import e2b_templates
from training.examples.rl.harbor.tito import rollout as tito_rollout
from training.examples.rl.harbor.tito import trial as harbor_adapter
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import RolloutRun, RolloutSample


def _setup(tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=object(),
        tokenizer_id="tokenizer",
        sample_kwargs={
            "max_tokens": 128,
            "temperature": 1.0,
            "logprobs": True,
            "include_routing_matrix": True,
            "echo": False,
            "max_seq_len": 4096,
        },
        extras={
            "renderer_name": "renderer",
            "harbor_trials_dir": str(tmp_path or Path.cwd() / ".tito-tests"),
        },
        sampler=object(),
        inference_base_url="http://deployment",
        api_key="deployment-key",
        model="deployment",
        completions_per_prompt=4,
    )


def _artifact(trajectory_id: str = "trajectory-1") -> TITOTrajectoryArtifact:
    now = time.time()
    return TITOTrajectoryArtifact(
        trajectory_id=trajectory_id,
        serving_affinity_key_hash="hash",
        metadata={},
        status="completed",
        terminal_reason=None,
        segments=(),
        calls=(),
        response_attempts=(),
        metrics=TITOMetricSummary(counters={}, distributions={}),
        started_at=now,
        finished_at=now + 1,
    )


def _terminal_artifact(
    *,
    trajectory_id: str = "trajectory-1",
    status: str,
    reason: str | None,
) -> TITOTrajectoryArtifact:
    artifact = _artifact(trajectory_id)
    return TITOTrajectoryArtifact(
        trajectory_id=artifact.trajectory_id,
        serving_affinity_key_hash=artifact.serving_affinity_key_hash,
        metadata=artifact.metadata,
        status=status,
        terminal_reason=reason,
        segments=artifact.segments,
        calls=artifact.calls,
        response_attempts=artifact.response_attempts,
        metrics=artifact.metrics,
        started_at=artifact.started_at,
        finished_at=artifact.finished_at,
    )


def _outcome(
    reward: float = 1.0,
    *,
    environment_type: str = "docker",
    artifact: TITOTrajectoryArtifact | None = None,
) -> harbor_adapter.HarborTrialOutcome:
    return harbor_adapter.HarborTrialOutcome(
        task_name="example",
        trial_name="trial-example",
        trial_path=Path("/tmp/trial-example"),
        reward=reward,
        rewards={"reward": reward},
        exception_type=None,
        exception_message=None,
        environment_type=environment_type,
        trajectory_artifact=artifact or _artifact(),
    )


def _sample_rollout(
    result: TITOTrajectoryArtifact,
    *,
    reward: float,
    max_context_tokens: int | None = None,
    **_: Any,
) -> RolloutRun:
    assert max_context_tokens == 4096
    return RolloutRun(
        segments=[
            RolloutSample(
                tokens=[1, 2],
                logprobs=[0.0, -0.1],
                loss_mask=[0, 1],
                reward=reward,
            )
        ],
        run_id=result.trajectory_id,
        metadata={"tito_metrics": {}},
    )


def _fake_bundle(tmp_path: Path) -> sidecar_runtime.TITOSidecarBundle:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(b"bundle")
    return sidecar_runtime.TITOSidecarBundle(archive, "a" * 64)


def test_sidecar_bundle_source_scan_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "runtime.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    link = source / "linked.py"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="does not support symlinks"):
        list(sidecar_runtime._iter_source_files(source))


def test_sidecar_bundle_is_deterministic_and_minimal(tmp_path) -> None:
    class Backend:
        @staticmethod
        def to_str() -> str:
            return '{"model":{"type":"WordLevel","vocab":{"x":0}}}'

    class Tokenizer:
        backend_tokenizer = Backend()
        chat_template = "{{ messages }}"
        special_tokens_map = {"eos_token": "</s>"}

        @staticmethod
        def save_pretrained(path: Path) -> None:
            path.mkdir(parents=True)
            (path / "tokenizer.json").write_text(Backend.to_str())
            (path / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "eos_token": "</s>",
                        "chat_template": Tokenizer.chat_template,
                    }
                )
            )
            (path / "chat_template.jinja").write_text(Tokenizer.chat_template)

    setup = _setup(tmp_path)
    setup.tokenizer = Tokenizer()
    setup.extras["tito_sidecar_bundle_root"] = str(tmp_path / "bundles")
    first = sidecar_runtime.build_sidecar_bundle(setup)
    second = sidecar_runtime.build_sidecar_bundle(setup)
    assert first == second
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first.path) as archive:
        names = set(archive.namelist())
        bundled_sdk_init = archive.read("python-sdk/fireworks/training/sdk/__init__.py")
        archive.extractall(extracted)
    assert "cookbook/training/tito/renderer.py" in names
    assert "cookbook/training/examples/rl/harbor/tito/sidecar.py" in names
    assert "tokenizer/chat_template.jinja" in names
    assert not any("model_formats/" in name for name in names)
    assert "python-sdk/fireworks/training/sdk/tito/_sidecar.py" in names
    assert "python-sdk/fireworks/__init__.py" in names
    assert "python-sdk/fireworks/training/__init__.py" in names
    assert "python-sdk/fireworks/_client.py" not in names
    assert not any("training/renderer/" in name for name in names)
    assert not any("/tests/" in name for name in names)
    imported_sdk = importlib.import_module("fireworks.training.sdk")
    assert bundled_sdk_init == Path(imported_sdk.__file__).read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fireworks.training.sdk import "
                "TITOSidecar, TrajectoryDriftPolicy; "
                "from training.tito.renderer import "
                "build_sidecar_tito_renderer; "
                "import training.examples.rl.harbor.tito.sidecar"
            ),
        ],
        check=True,
        cwd=extracted,
        env={
            **os.environ,
            "PYTHONPATH": f"{extracted / 'python-sdk'}:{extracted / 'cookbook'}",
        },
    )


def test_deep_swe_preparation_writes_reproducible_manifest(tmp_path) -> None:
    repository = tmp_path / "deep-swe"
    tasks = repository / "tasks"
    for name in ("task-a", "task-b"):
        task = tasks / name
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_text(
            '[environment]\ndocker_image = "example@sha256:' + "0" * 64 + '"\n'
        )
        (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(
        ["git", "-C", repository, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", repository, "add", "."], check=True)
    subprocess.run(["git", "-C", repository, "commit", "-qm", "fixture"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "remote",
            "add",
            "origin",
            "https://github.com/datacurve-ai/deep-swe.git",
        ],
        check=True,
    )

    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_deep_swe_tasks.prepare_deep_swe(
        repository,
        tmp_path / "prepared",
        manifest_path,
        opencode_version=DEFAULT_OPENCODE_VERSION,
    )
    persisted = json.loads(manifest_path.read_text())
    assert persisted == manifest
    assert persisted["task_count"] == 2
    assert persisted["tasks"] == ["task-a", "task-b"]
    assert persisted["source"]["remote"].endswith("datacurve-ai/deep-swe.git")
    assert set(persisted["source_sha256"]) == {"task-a", "task-b"}
    assert set(persisted["prepared_sha256"]) == {"task-a", "task-b"}


def test_training_entry_has_no_central_gateway_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "zai-org/GLM-5.2",
            "--training-shape-id",
            "accounts/example/trainingShapes/test",
            "--renderer-name",
            "glm_moe_dsa_preserve_thinking",
            "--tito-prompt-mode",
            "incremental",
            "--deployment-id",
            "existing-deployment",
            "--max-concurrent-trials",
            "32",
            "--harbor-environment",
            "e2b",
            "--evaluation-task",
            "task-a",
            "--evaluation-task",
            "task-b",
            "--evaluation-every",
            "5",
            "--evaluation-concurrency",
            "8",
        ],
    )
    args = opencode_train.parse_args()
    assert args.deployment_id == "existing-deployment"
    assert args.max_concurrent_trials == 32
    assert args.harbor_environment == "e2b"
    assert args.evaluation_task == ["task-a", "task-b"]
    assert args.evaluation_every == 5
    assert args.evaluation_concurrency == 8
    extras = opencode_train._rollout_extras(args, selector=None)
    assert extras["tito_prompt_mode"] == "incremental"
    assert str(extras["tito_sidecar_bundle_root"]).endswith(
        "harbor_opencode_logs/.tito-sidecar-bundles"
    )
    assert not any(
        name.startswith("tito_") and "gateway" in name for name in vars(args)
    )


def test_sampling_entry_uses_existing_deployment_without_training_shape(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--sampling-only",
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "zai-org/GLM-5.2",
            "--renderer-name",
            "glm_moe_dsa_preserve_thinking",
            "--deployment-id",
            "accounts/example/deployments/policy",
            "--harbor-trials-dir",
            "./trials",
            "--harbor-task",
            "task-a",
            "--harbor-task",
            "task-b",
        ],
    )
    args = opencode_train.parse_args()
    assert args.sampling_only is True
    assert args.training_shape_id is None
    assert args.harbor_task == ["task-a", "task-b"]


def test_load_harbor_rows_preserves_requested_task_order(monkeypatch, tmp_path) -> None:
    class FakeTaskConfig:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_task_id(self):
            return SimpleNamespace(get_name=lambda: self.name)

        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {"name": self.name}

    class FakeDatasetConfig:
        def __init__(self, **kwargs) -> None:
            assert kwargs["task_names"] == ["task-b", "task-a"]

        async def get_task_configs(self):
            return [FakeTaskConfig("task-a"), FakeTaskConfig("task-b")]

    monkeypatch.setattr(
        harbor_adapter,
        "_require_harbor",
        lambda: SimpleNamespace(DatasetConfig=FakeDatasetConfig),
    )
    rows = asyncio.run(
        harbor_adapter.load_harbor_rows_async(
            tmp_path,
            task_names=["task-b", "task-a"],
        )
    )
    assert [row["task_name"] for row in rows] == ["task-b", "task-a"]


def test_sampling_only_builds_rollout_setup_without_trainer(
    monkeypatch, tmp_path
) -> None:
    args = SimpleNamespace(
        deployment_id="accounts/example/deployments/policy",
        trainer_job_id=None,
        harbor_trials_dir=str(tmp_path / "trials"),
        tokenizer_model="tokenizer",
        tokenizer_revision=None,
        max_completion_tokens=1024,
        temperature=1.0,
        sample_timeout=6900,
        completions_per_prompt=1,
        max_seq_len=4096,
        log_path=str(tmp_path / "logs"),
        renderer_name="renderer",
        rollout_retries=3,
        retry_include_exception=None,
        max_concurrent_trials=4,
        terminal_failure_reward=None,
        opencode_version=DEFAULT_OPENCODE_VERSION,
        harness_tool_timeout_seconds=600,
        harbor_trial_config=None,
        harbor_environment="e2b",
        tito_debug=False,
        tito_prompt_mode="full_history",
    )
    captured: dict[str, Any] = {}

    monkeypatch.setenv("FIREWORKS_API_KEY", "deployment-key")
    monkeypatch.setattr(opencode_train, "load_tokenizer", lambda *_: object())

    async def rollout_fn(*_args, **_kwargs):
        return None

    def make_rollout(setup):
        captured["setup"] = setup
        return rollout_fn

    async def evaluate(fn, rows, **kwargs):
        captured["rollout_fn"] = fn
        captured["rows"] = rows
        captured["evaluate_kwargs"] = kwargs
        return {"sampling/attempted_trajectories": 1}

    async def close(fn):
        captured["closed"] = fn

    monkeypatch.setattr(opencode_train, "make_rollout_fn", make_rollout)
    monkeypatch.setattr(opencode_train, "evaluate_rows", evaluate)
    monkeypatch.setattr(opencode_train, "close_rollout_fn", close)
    opencode_train._run_sampling_only(
        args,
        rows=[{"task_name": "task-a"}],
        selector=None,
    )

    setup = captured["setup"]
    assert setup.model == args.deployment_id
    assert setup.sampler is None
    assert not hasattr(setup, "max_context_tokens")
    assert setup.sample_kwargs["max_seq_len"] == args.max_seq_len
    assert setup.extras["harbor_environment"] == "e2b"
    assert setup.extras["tito_prompt_mode"] == "full_history"
    assert captured["evaluate_kwargs"]["max_concurrency"] is None
    assert captured["closed"] is rollout_fn
    assert (
        json.loads((tmp_path / "logs" / "sampling-result.json").read_text())["mode"]
        == "sampling_only"
    )


def test_context_limit_is_shared_by_inference_and_training_retention(
    monkeypatch, tmp_path
) -> None:
    setup = _setup(tmp_path)
    monkeypatch.setattr(
        rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    runner = rollout.make_rollout_fn(setup)
    assert setup.sample_kwargs["max_seq_len"] == 4096
    assert runner._context_limit == 4096


def test_full_sequence_router_replay_is_rejected(monkeypatch, tmp_path) -> None:
    setup = _setup(tmp_path)
    setup.sample_kwargs["echo"] = True
    with pytest.raises(ValueError, match="completion-only Router Replay"):
        rollout.make_rollout_fn(setup)


def test_opencode_classifier_contract_rejects_unpinned_versions(
    monkeypatch, tmp_path
) -> None:
    setup = _setup(tmp_path)
    setup.extras["opencode_version"] = "future-version"
    monkeypatch.setattr(
        rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    with pytest.raises(ValueError, match="classifier-certified version"):
        rollout.make_rollout_fn(setup)


def test_launch_spec_carries_limits_and_debug_without_runtime_aliases(tmp_path) -> None:
    setup = _setup(tmp_path)
    setup.extras.update(
        {
            "tito_debug_enabled": True,
            "tito_prompt_mode": "incremental",
        }
    )
    spec = sidecar_runtime.build_launch_spec(
        setup,
        _fake_bundle(tmp_path / "bundle"),
        call_classifier="tools_present",
        metadata={"member": 3},
    )
    value = json.loads(sidecar_runtime.launch_spec_json(spec))
    assert value["max_context_tokens"] == 4096
    assert value["max_output_tokens"] == 128
    assert value["trajectory_metadata"] == {"member": 3}
    assert value["debug_enabled"] is True
    assert value["prompt_mode"] == "incremental"
    assert "gateway" not in json.dumps(value).lower()


def test_launch_spec_keeps_full_history_as_the_default(tmp_path) -> None:
    setup = _setup(tmp_path)

    spec = sidecar_runtime.build_launch_spec(
        setup,
        _fake_bundle(tmp_path / "bundle"),
        call_classifier="tools_present",
        metadata={},
    )

    assert spec.prompt_mode == "full_history"


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("context_budget_exhausted", "context_budget_exhausted"),
        ("", None),
        ("untrusted", None),
    ],
)
def test_sidecar_failure_disposition_uses_typed_marker_output(stdout, expected) -> None:
    class Environment:
        async def exec(self, *, command, cwd=None):
            assert cwd == "/"
            assert sidecar_runtime.SIDECAR_CONTEXT_OVERFLOW_PATH in command
            # The provider return code is intentionally irrelevant; E2B has
            # exposed inconsistent shell-status reporting for probe commands.
            return SimpleNamespace(return_code=9, stdout=stdout, stderr="")

    assert (
        asyncio.run(sidecar_runtime.sidecar_failure_disposition(Environment()))
        == expected
    )


def test_install_sidecar_uploads_bundle_and_returns_private_endpoint(tmp_path) -> None:
    calls: list[tuple[str, Any]] = []

    class Environment:
        async def upload_file(self, source, target):
            calls.append(("upload_file", (Path(source).read_bytes(), target)))

        async def exec(self, *, command, cwd=None):
            calls.append(("exec", (command, cwd)))
            if f"cat {sidecar_runtime.SIDECAR_ENDPOINT_PATH}" in command:
                return SimpleNamespace(
                    return_code=0,
                    stdout=json.dumps(
                        {
                            "trajectory_id": "trajectory-1",
                            "openai_base_url": (
                                "http://127.0.0.1:4567/trajectories/trajectory-1/v1"
                            ),
                            "api_key": "private-key",
                        }
                    ),
                    stderr="",
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    endpoint = asyncio.run(
        sidecar_runtime.install_sidecar(
            Environment(),
            bundle_path=bundle_path,
            launch_spec='{"api_key":"inference-key"}',
        )
    )
    assert endpoint["openai_base_url"].startswith("http://127.0.0.1:")
    assert endpoint["api_key"] == "private-key"
    assert sum(kind == "upload_file" for kind, _ in calls) == 2
    assert any(
        kind == "upload_file" and value[1] == sidecar_runtime.SIDECAR_BUNDLE_ARCHIVE
        for kind, value in calls
    )
    exec_calls = [value for kind, value in calls if kind == "exec"]
    assert exec_calls
    assert all(cwd == "/" for _command, cwd in exec_calls)
    assert all("inference-key" not in command for command, _cwd in exec_calls)


def test_terminalize_sidecar_publishes_terminal_json_atomically(tmp_path) -> None:
    calls: list[tuple[str, Any]] = []

    class Environment:
        async def upload_file(self, source, target):
            calls.append(("upload_file", (Path(source).read_text(), target)))

        async def exec(self, *, command, cwd=None):
            calls.append(("exec", (command, cwd)))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    asyncio.run(
        sidecar_runtime.terminalize_sidecar(
            Environment(), status="failed", reason="test"
        )
    )
    uploads = [value for kind, value in calls if kind == "upload_file"]
    assert uploads == [
        (
            json.dumps({"reason": "test", "status": "failed"}, sort_keys=True),
            sidecar_runtime.SIDECAR_TERMINAL_STAGING_PATH,
        )
    ]
    exec_calls = [value for kind, value in calls if kind == "exec"]
    assert all(cwd == "/" for _command, cwd in exec_calls)
    commands = [command for command, _cwd in exec_calls]
    assert sidecar_runtime.SIDECAR_TERMINAL_STAGING_PATH in commands[0]
    assert f"mv -f {sidecar_runtime.SIDECAR_TERMINAL_STAGING_PATH}" in commands[0]
    assert sidecar_runtime.SIDECAR_TERMINAL_PATH in commands[0]


def test_terminalize_sidecar_fails_immediately_when_atomic_publish_fails() -> None:
    class Environment:
        async def upload_file(self, source, target):
            del source, target

        async def exec(self, *, command, cwd=None):
            assert cwd == "/"
            assert sidecar_runtime.SIDECAR_TERMINAL_STAGING_PATH in command
            return SimpleNamespace(return_code=1, stdout="", stderr="mv failed")

    with pytest.raises(RuntimeError, match="publish.*mv failed"):
        asyncio.run(
            sidecar_runtime.terminalize_sidecar(
                Environment(), status="failed", reason="test"
            )
        )


def test_cancelled_harness_is_quiesced_before_sidecar_terminalization(
    monkeypatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class Environment:
        async def exec(self, *, command, cwd=None):
            calls.append(("quiesce", (command, cwd)))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def terminalize(environment, *, status, reason=None):
        del environment
        calls.append(("terminalize", (status, reason)))

    monkeypatch.setattr(sidecar_runtime, "terminalize_sidecar", terminalize)
    asyncio.run(
        sidecar_runtime.abandon_sidecar_after_harness_cancellation(
            Environment(),
            process_pattern="[m]ini-swe-agent --yolo",
        )
    )

    assert [kind for kind, _value in calls] == ["quiesce", "terminalize"]
    command, cwd = calls[0][1]
    assert cwd == "/"
    assert "pgrep -f" in command
    assert "[m]ini-swe-agent --yolo" in command
    assert calls[1][1] == ("abandoned", "agent_cancelled")


def test_quiesce_harness_rejects_multiline_process_signature() -> None:
    with pytest.raises(ValueError, match="single line"):
        asyncio.run(
            sidecar_runtime.quiesce_harness_process(
                object(),
                process_pattern="mini-swe-agent\nother-process",
            )
        )


def test_sidecar_serve_closes_runtime_when_trajectory_creation_fails(
    monkeypatch, tmp_path
) -> None:
    from training.tito import renderer as renderer_runtime

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "manifest.json").write_text(
        json.dumps({"sha256": "bundle-digest"}), encoding="utf-8"
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "bundle_digest": "bundle-digest",
                "inference_base_url": "https://deployment.example",
                "api_key": "deployment-key",
                "model": "policy",
                "renderer_name": "renderer",
                "max_context_tokens": 4096,
                "max_output_tokens": 1024,
                "sampling_defaults": {},
                "max_masked_tokens": 1024,
                "on_other_mismatch": "new_segment",
                "call_classifier": "tools_present",
                "trajectory_metadata": {},
                "debug_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    class FakeSampler:
        instance = None

        def __init__(self, **_kwargs):
            self.closed = False
            type(self).instance = self

        def close(self) -> None:
            self.closed = True

    class FakeSidecar:
        instance = None

        @classmethod
        def from_deployment_sampler(cls, *_args, **_kwargs):
            instance = cls()
            FakeSidecar.instance = instance
            return instance

        async def start(self) -> None:
            self.started = True

        async def create_trajectory_async(self, **_kwargs):
            raise RuntimeError("trajectory setup failed")

        async def close(self) -> None:
            self.closed = True

    import fireworks.training.sdk as training_sdk

    monkeypatch.setattr(sidecar_runtime, "SIDECAR_BUNDLE_ROOT", str(bundle_root))
    monkeypatch.setattr(sidecar_runtime, "SIDECAR_PID_PATH", str(tmp_path / "pid"))
    monkeypatch.setattr(training_sdk, "DeploymentSampler", FakeSampler)
    monkeypatch.setattr(training_sdk, "TITOSidecar", FakeSidecar)
    monkeypatch.setattr(
        renderer_runtime, "load_sidecar_tokenizer", lambda _path: object()
    )
    monkeypatch.setattr(
        renderer_runtime,
        "build_sidecar_tito_renderer",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="trajectory setup failed"):
        asyncio.run(sidecar_runtime.serve(spec_path))
    assert FakeSidecar.instance.started
    assert FakeSidecar.instance.closed
    assert FakeSampler.instance.closed


def test_agents_write_only_the_loopback_trajectory_endpoint(tmp_path) -> None:
    pytest.importorskip("harbor")
    from training.examples.rl.harbor.opencode.agent import ConfigurableOpenCode
    from training.examples.rl.harbor.pi.agent import ConfigurablePi

    shared = {
        "logs_dir": tmp_path,
        "sidecar_bundle_path": str(tmp_path / "bundle"),
        "sidecar_launch_spec": "{}",
        "context_limit": 4096,
        "output_limit": 1024,
        "tool_timeout_seconds": 600,
    }
    agents = (
        ConfigurableOpenCode(**shared, version=DEFAULT_OPENCODE_VERSION),
        ConfigurablePi(**shared, version="0.84.2"),
    )

    class Environment:
        def __init__(self) -> None:
            self.default_user = None
            self.commands: list[str] = []
            self.uploads: dict[str, str] = {}

        async def exec(self, *, command, **_kwargs):
            self.commands.append(command)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, target):
            self.uploads[str(target)] = Path(source).read_text(encoding="utf-8")

    for agent in agents:
        environment = Environment()
        agent._policy_base_url = "http://127.0.0.1:4567/trajectories/trajectory-1/v1"
        agent._policy_api_key = "trajectory-key"
        if isinstance(agent, ConfigurableOpenCode):
            asyncio.run(agent._write_config(environment, env=agent._agent_env()))
        else:
            asyncio.run(agent._write_config(environment))
        uploaded = "\n".join(environment.uploads.values())
        commands = "\n".join(environment.commands)
        assert "127.0.0.1:4567/trajectories/trajectory-1/v1" in uploaded
        assert "trajectory-key" in uploaded
        assert "host.docker.internal" not in uploaded
        assert "gateway.example" not in uploaded
        assert "trajectory-key" not in commands
        assert "127.0.0.1:4567/trajectories/trajectory-1/v1" not in commands
        if isinstance(agent, ConfigurableOpenCode):
            assert "@opencode-ai/plugin" in uploaded


def test_opencode_disables_unrelated_remote_bootstrap_requests(tmp_path) -> None:
    pytest.importorskip("harbor")
    from training.examples.rl.harbor.opencode.agent import ConfigurableOpenCode

    agent = ConfigurableOpenCode(
        logs_dir=tmp_path,
        sidecar_bundle_path=str(tmp_path / "bundle"),
        sidecar_launch_spec="{}",
        context_limit=4096,
        output_limit=1024,
        tool_timeout_seconds=600,
        extra_env={"CUSTOM": "preserved"},
        version=DEFAULT_OPENCODE_VERSION,
    )
    agent._policy_base_url = "http://127.0.0.1:4567/trajectories/t/v1"
    agent._policy_api_key = "key"
    env = agent._agent_env()
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    assert env["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert env["CUSTOM"] == "preserved"
    assert agent._policy_config()["snapshot"] is False
    assert agent._policy_config()["agent"]["title"]["disable"] is True
    assert agent._policy_config()["tools"] == {"task": False}
    model = agent._policy_config()["provider"]["fireworks-rl"]["models"]["policy"]
    assert model["interleaved"] == {"field": "reasoning_content"}


def test_pi_preserves_empty_reasoning_on_assistant_replay(tmp_path) -> None:
    pytest.importorskip("harbor")
    from training.examples.rl.harbor.pi.agent import ConfigurablePi

    agent = ConfigurablePi(
        logs_dir=tmp_path,
        sidecar_bundle_path=str(tmp_path / "bundle"),
        sidecar_launch_spec="{}",
        context_limit=4096,
        output_limit=1024,
        tool_timeout_seconds=600,
        version="0.84.2",
    )
    model = agent._models()["providers"]["fireworks-tito"]["models"][0]
    assert model["compat"]["requiresReasoningContentOnAssistantMessages"] is True


@pytest.mark.parametrize(
    ("module_name", "class_name", "version"),
    [
        (
            "training.examples.rl.harbor.opencode.agent",
            "ConfigurableOpenCode",
            DEFAULT_OPENCODE_VERSION,
        ),
        ("training.examples.rl.harbor.pi.agent", "ConfigurablePi", "0.84.2"),
    ],
)
def test_agent_command_promotes_context_marker_to_nonzero_exit(
    monkeypatch,
    tmp_path,
    module_name,
    class_name,
    version,
) -> None:
    pytest.importorskip("harbor")
    module = importlib.import_module(module_name)
    agent_type = getattr(module, class_name)
    agent = agent_type(
        logs_dir=tmp_path,
        sidecar_bundle_path=str(tmp_path / "bundle"),
        sidecar_launch_spec="{}",
        context_limit=4096,
        output_limit=1024,
        tool_timeout_seconds=600,
        version=version,
    )
    commands: list[str] = []
    terminal_states: list[str] = []

    async def write_config(*_args, **_kwargs):
        return None

    async def exec_as_agent(_environment, *, command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def terminalize(_environment, *, status, reason=None):
        del reason
        terminal_states.append(status)

    monkeypatch.setattr(agent, "_write_config", write_config)
    monkeypatch.setattr(agent, "exec_as_agent", exec_as_agent)
    monkeypatch.setattr(module, "terminalize_sidecar", terminalize)

    asyncio.run(
        agent_type.run.__wrapped__(
            agent,
            "solve the task",
            object(),
            object(),
        )
    )

    assert len(commands) == 1
    assert sidecar_runtime.SIDECAR_CONTEXT_OVERFLOW_PATH in commands[0]
    assert "exit 43" in commands[0]
    assert terminal_states == ["completed"]


def test_four_same_prompt_rollouts_build_four_independent_attempt_specs(
    monkeypatch, tmp_path
) -> None:
    setup = _setup(tmp_path)
    bundle = _fake_bundle(tmp_path / "bundle")
    monkeypatch.setattr(rollout, "build_sidecar_bundle", lambda _setup: bundle)
    runner = rollout.make_rollout_fn(setup)
    runner._trial_start_interval_seconds = 0
    specs: list[dict[str, Any]] = []

    async def run_trial(**kwargs):
        specs.append(json.loads(kwargs["sidecar_launch_spec"]))
        member = specs[-1]["trajectory_metadata"]["rollout_member_index"]
        return _outcome(artifact=_artifact(f"trajectory-{member}"))

    monkeypatch.setattr(runner, "_run_trial", run_trial)
    monkeypatch.setattr(tito_rollout, "materialize_tito_trajectory", _sample_rollout)

    async def exercise():
        return await asyncio.gather(
            *(
                runner._run_opencode(
                    task_config={},
                    task_name="same-task",
                    run_id=f"same-group-member-{member}",
                    rollout_group_id="same-group",
                    rollout_member_index=member,
                    canonical_initial_prompt_hash="same-prompt-hash",
                )
                for member in range(4)
            )
        )

    results = asyncio.run(exercise())
    assert all(result is not None for result in results)
    assert len(specs) == 4
    assert {
        spec["trajectory_metadata"]["rollout_member_index"] for spec in specs
    } == set(range(4))
    assert {spec["bundle_digest"] for spec in specs} == {bundle.digest}
    assert all(
        "parent" not in spec["trajectory_metadata"]
        and "branch" not in spec["trajectory_metadata"]
        for spec in specs
    )


@pytest.mark.parametrize(
    "environment, expected_pace_calls", [("docker", 1), ("e2b", 0)]
)
def test_only_local_docker_trial_starts_are_paced(
    monkeypatch, tmp_path, environment, expected_pace_calls
) -> None:
    setup = _setup(tmp_path)
    setup.extras["harbor_environment"] = environment
    monkeypatch.setattr(
        rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    runner = rollout.make_rollout_fn(setup)
    pace_calls = 0

    async def pace():
        nonlocal pace_calls
        pace_calls += 1

    async def run_trial(**_kwargs):
        return _outcome(environment_type=environment)

    monkeypatch.setattr(runner, "_pace_trial_start", pace)
    monkeypatch.setattr(runner, "_run_trial", run_trial)
    asyncio.run(
        runner._run_admitted_trial(task_config={}, inference_key="key", run_id="run")
    )
    assert pace_calls == expected_pace_calls


def test_recoverable_attempt_gets_a_fresh_sidecar_spec(monkeypatch, tmp_path) -> None:
    setup = _setup(tmp_path)
    setup.extras["rollout_retries"] = 1
    monkeypatch.setattr(
        rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    runner = rollout.make_rollout_fn(setup)
    runner._trial_start_interval_seconds = 0
    specs: list[dict[str, Any]] = []

    async def run_trial(**kwargs):
        specs.append(json.loads(kwargs["sidecar_launch_spec"]))
        if len(specs) == 1:
            raise RecoverableRolloutError("temporary")
        return _outcome(artifact=_artifact("trajectory-retry"))

    monkeypatch.setattr(runner, "_run_trial", run_trial)

    async def no_retry_delay(_delay: float) -> None:
        return None

    monkeypatch.setattr(tito_rollout.asyncio, "sleep", no_retry_delay)
    monkeypatch.setattr(tito_rollout, "materialize_tito_trajectory", _sample_rollout)
    result = asyncio.run(
        runner._run_opencode(task_config={}, task_name="task", run_id="run")
    )
    assert result is not None
    assert [item["trajectory_metadata"]["retry_index"] for item in specs] == [0, 1]


def test_opencode_temporary_trial_survives_through_adapter_metrics(
    monkeypatch, tmp_path
) -> None:
    setup = _setup(tmp_path)
    setup.extras.pop("harbor_trials_dir")
    setup.extras["rollout_retries"] = 0
    monkeypatch.setattr(
        rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    runner = rollout.make_rollout_fn(setup)
    runner._trial_start_interval_seconds = 0

    async def run_trial(**kwargs):
        trial_path = Path(kwargs["trials_dir"]) / "trial-example"
        agent_path = trial_path / "agent"
        agent_path.mkdir(parents=True)
        (agent_path / "events.txt").write_text(
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "bash",
                        "callID": "call-1",
                        "state": {"error": "command timed out"},
                    },
                }
            ),
            encoding="utf-8",
        )
        outcome = _outcome(artifact=_artifact("trajectory-with-timeout"))
        return harbor_adapter.HarborTrialOutcome(
            **{**vars(outcome), "trial_path": trial_path}
        )

    monkeypatch.setattr(runner, "_run_trial", run_trial)
    monkeypatch.setattr(tito_rollout, "materialize_tito_trajectory", _sample_rollout)
    result = asyncio.run(
        runner._run_opencode(task_config={}, task_name="task", run_id="run")
    )
    assert result is not None
    assert result.metadata["harness_tool_timeout_count"] == 1


def test_harness_trial_paths_expand_user(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for module in (rollout, pi_rollout, mini_swe_rollout):
        setup = _setup(tmp_path)
        setup.extras["harbor_trials_dir"] = "~/tito-trials"
        monkeypatch.setattr(
            module,
            "build_sidecar_bundle",
            lambda _setup: _fake_bundle(tmp_path / module.__name__.replace(".", "-")),
        )
        runner = module.make_rollout_fn(setup)
        assert runner._trials_dir == tmp_path / "tito-trials"


def test_harness_trial_paths_honor_yaml_default(monkeypatch, tmp_path) -> None:
    for module in (rollout, pi_rollout, mini_swe_rollout):
        setup = _setup(tmp_path)
        setup.extras.pop("harbor_trials_dir")
        setup.extras["harbor_trial_config"] = {
            "trials_dir": str(tmp_path / "yaml-trials")
        }
        monkeypatch.setattr(
            module,
            "build_sidecar_bundle",
            lambda _setup: _fake_bundle(tmp_path / module.__name__.replace(".", "-")),
        )
        runner = module.make_rollout_fn(setup)
        assert runner._trials_dir == tmp_path / "yaml-trials"


def _install_fake_e2b_rate_limit_exception(monkeypatch) -> None:
    real_import_module = harbor_adapter.importlib.import_module

    class RateLimitException(Exception):
        pass

    def import_module(name: str):
        if name == "e2b.exceptions":
            return SimpleNamespace(RateLimitException=RateLimitException)
        return real_import_module(name)

    monkeypatch.setattr(harbor_adapter.importlib, "import_module", import_module)


@pytest.mark.parametrize(
    "error",
    [
        RecoverableRolloutError("temporary provider failure"),
        ValueError("invalid trajectory contract"),
    ],
)
def test_mini_swe_rollout_isolates_attempt_errors_from_producer(
    monkeypatch, tmp_path, error
) -> None:
    _install_fake_e2b_rate_limit_exception(monkeypatch)
    setup = _setup(tmp_path)
    setup.extras["rollout_retries"] = 0
    setup.extras["harbor_reward_key"] = "partial"
    setup.extras["retry_include_exceptions"] = sorted(
        harbor_adapter.DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS | {"RateLimitException"}
    )
    monkeypatch.setattr(
        mini_swe_rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    monkeypatch.setattr(
        mini_swe_rollout,
        "task_config_from_row",
        lambda _row: SimpleNamespace(),
    )
    monkeypatch.setattr(
        mini_swe_rollout,
        "task_initial_instruction",
        lambda _task_config: "solve the task",
    )

    seen: dict[str, Any] = {}

    async def run_trial(**kwargs):
        seen.update(kwargs)
        raise error

    monkeypatch.setattr(mini_swe_rollout, "run_harbor_trial", run_trial)
    runner = mini_swe_rollout.make_rollout_fn(setup)

    assert asyncio.run(runner({"task_name": "example"})) is None
    assert seen["reward_key"] == "partial"
    assert seen["retry_include_exceptions"] == frozenset(
        harbor_adapter.DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS | {"RateLimitException"}
    )


@pytest.mark.parametrize(
    "error",
    [
        RecoverableRolloutError("temporary provider failure"),
        ValueError("invalid trajectory contract"),
    ],
)
def test_pi_rollout_forwards_retry_contract_and_isolates_attempt_errors(
    monkeypatch, tmp_path, error
) -> None:
    _install_fake_e2b_rate_limit_exception(monkeypatch)
    setup = _setup(tmp_path)
    setup.extras["rollout_retries"] = 0
    setup.extras["retry_include_exceptions"] = sorted(
        harbor_adapter.DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS | {"RateLimitException"}
    )
    monkeypatch.setattr(
        pi_rollout,
        "build_sidecar_bundle",
        lambda _setup: _fake_bundle(tmp_path / "bundle"),
    )
    monkeypatch.setattr(
        pi_rollout,
        "task_config_from_row",
        lambda _row: SimpleNamespace(),
    )
    monkeypatch.setattr(
        pi_rollout,
        "task_name_from_row",
        lambda _row: "example",
    )
    monkeypatch.setattr(
        pi_rollout,
        "task_initial_instruction",
        lambda _task_config: "solve the task",
    )
    seen: dict[str, Any] = {}

    async def run_trial(**kwargs):
        seen.update(kwargs)
        raise error

    monkeypatch.setattr(pi_rollout, "run_harbor_trial", run_trial)
    runner = pi_rollout.make_rollout_fn(setup)

    assert asyncio.run(runner({"task_name": "example"})) is None
    assert seen["retry_include_exceptions"] == frozenset(
        harbor_adapter.DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS | {"RateLimitException"}
    )


class _EnvironmentType(str, Enum):
    E2B = "e2b"
    DOCKER = "docker"


class _Config(SimpleNamespace):
    @classmethod
    def model_validate(cls, value):
        value = dict(value)
        for key in ("agent", "environment", "verifier"):
            if isinstance(value.get(key), dict):
                value[key] = cls(**value[key])
        return cls(**value)


def _fake_harbor():
    return SimpleNamespace(EnvironmentType=_EnvironmentType, TrialConfig=_Config)


@pytest.mark.parametrize(
    ("configured_environment", "environment"),
    [("docker", "docker"), ("e2b", "e2b"), ("docker", "e2b")],
)
def test_trial_config_uses_same_sidecar_contract_for_both_backends(
    tmp_path, configured_environment, environment
) -> None:
    sidecar_launch_spec = json.dumps(
        {
            "api_key": "secret",
            "inference_base_url": "https://api.fireworks.ai",
        }
    )
    config = harbor_adapter._build_trial_config(
        _fake_harbor(),
        template={"environment": {"type": configured_environment}},
        task_config={"path": "/tasks/example"},
        run_id="run",
        trials_dir=tmp_path,
        harbor_environment=environment,
        sidecar_bundle_path=tmp_path / "bundle",
        sidecar_launch_spec=sidecar_launch_spec,
        context_limit=4096,
        output_limit=1024,
        agent_import_path=OPENCODE_HARBOR_IMPORT_PATH,
        agent_version=DEFAULT_OPENCODE_VERSION,
    )
    expected = (
        _EnvironmentType.DOCKER if environment == "docker" else _EnvironmentType.E2B
    )
    assert config.environment.type is expected
    assert config.agent.kwargs["sidecar_bundle_path"] == str(tmp_path / "bundle")
    assert config.agent.kwargs["sidecar_launch_spec"] == sidecar_launch_spec
    assert config.agent.extra_allowed_hosts == ["api.fireworks.ai"]
    assert not getattr(config.environment, "extra_docker_compose", [])
    assert config.artifacts[-5:] == [
        {
            "source": sidecar_runtime.SIDECAR_ARTIFACT_PATH,
            "destination": "tito/compact/trajectory.tito",
        },
        {
            "source": sidecar_runtime.SIDECAR_ARTIFACT_MANIFEST_PATH,
            "destination": "tito/compact/trajectory.json",
        },
        {
            "source": sidecar_runtime.SIDECAR_COMPLETE_PATH,
            "destination": "tito/compact/COMPLETE",
        },
        {
            "source": sidecar_runtime.SIDECAR_STDOUT_PATH,
            "destination": "tito/logs/sidecar.stdout",
        },
        {
            "source": sidecar_runtime.SIDECAR_STDERR_PATH,
            "destination": "tito/logs/sidecar.stderr",
        },
    ]


def test_e2b_rejects_compose_task(tmp_path) -> None:
    task_path = tmp_path / "task"
    environment_path = task_path / "environment"
    environment_path.mkdir(parents=True)
    (environment_path / "docker-compose.yaml").write_text("services: {}\n")
    (task_path / "task.toml").write_text(
        'schema_version = "1.0"\n\n[task]\nname = "test/task"\n'
    )
    with pytest.raises(ValueError, match="does not support Docker Compose"):
        harbor_adapter._build_trial_config(
            _fake_harbor(),
            template=None,
            task_config={"path": str(task_path)},
            run_id="run",
            trials_dir=tmp_path / "trials",
            harbor_environment="e2b",
            sidecar_bundle_path=tmp_path / "bundle",
            sidecar_launch_spec="{}",
            agent_import_path=OPENCODE_HARBOR_IMPORT_PATH,
            agent_version=DEFAULT_OPENCODE_VERSION,
        )


def _write_collected_artifact(
    trial_path: Path, artifact: TITOTrajectoryArtifact
) -> None:
    target = trial_path / "artifacts" / "tito" / "compact"
    target.mkdir(parents=True)
    encoded = artifact.pack()
    (target / "trajectory.tito").write_bytes(encoded)
    (target / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trajectory_id": artifact.trajectory_id,
                "status": artifact.status,
                "terminal_reason": artifact.terminal_reason,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
            }
        )
    )
    (target / "COMPLETE").write_text("complete\n")


def test_collected_artifact_is_checksum_and_schema_validated(tmp_path) -> None:
    artifact = _artifact()
    _write_collected_artifact(tmp_path, artifact)
    decoded, manifest = harbor_adapter._load_sidecar_artifact(tmp_path)
    assert decoded == artifact
    assert manifest["trajectory_id"] == artifact.trajectory_id


def test_missing_or_corrupt_collected_artifact_is_retryable(tmp_path) -> None:
    with pytest.raises(RecoverableRolloutError, match="did not collect"):
        harbor_adapter._load_sidecar_artifact(tmp_path)
    _write_collected_artifact(tmp_path, _artifact())
    artifact_path = tmp_path / "artifacts" / "tito" / "compact" / "trajectory.tito"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")
    with pytest.raises(RecoverableRolloutError, match="byte count mismatch"):
        harbor_adapter._load_sidecar_artifact(tmp_path)


def test_e2b_stream_open_timeout_retries_even_with_valid_prefix_artifact(
    monkeypatch, tmp_path
) -> None:
    artifact = _terminal_artifact(status="failed", reason="provider_timeout")

    class Trial:
        def __init__(self, config):
            self.config = config
            self._agent_timeout_sec = 7200

        @classmethod
        async def create(cls, config):
            return cls(config)

        async def run(self):
            trial_path = Path(self.config.trials_dir) / self.config.trial_name
            _write_collected_artifact(trial_path, artifact)
            return SimpleNamespace(
                task_name="example",
                trial_name=self.config.trial_name,
                verifier_result=None,
                exception_info=SimpleNamespace(
                    exception_type="TimeoutException",
                    exception_message="provider request timeout",
                    exception_traceback=(
                        'File "/site-packages/harbor/environments/e2b.py"\n'
                        'File "/site-packages/e2b/envd/client_async.py"\n'
                        "e2b.exceptions.TimeoutException: Request timed out: the "
                        "stream didn't open within 'request_timeout' (60.0 seconds)."
                    ),
                ),
            )

    harbor = _fake_harbor()
    harbor.Trial = Trial
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: harbor)
    with pytest.raises(RecoverableRolloutError, match="command stream did not open"):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={},
                inference_key="inference-key",
                run_id="e2b-timeout-test",
                harbor_environment="e2b",
                sidecar_bundle_path=tmp_path / "bundle.zip",
                sidecar_launch_spec=json.dumps(
                    {
                        "debug_enabled": False,
                        "inference_base_url": "https://api.fireworks.ai",
                    }
                ),
                trials_dir=tmp_path / "trials",
                agent_import_path=OPENCODE_HARBOR_IMPORT_PATH,
                agent_version=DEFAULT_OPENCODE_VERSION,
            )
        )


def test_harbor_trial_selects_configured_numeric_reward(monkeypatch, tmp_path) -> None:
    artifact = _artifact("partial-reward")

    class Trial:
        def __init__(self, config):
            self.config = config
            self._agent_timeout_sec = 7200

        @classmethod
        async def create(cls, config):
            return cls(config)

        async def run(self):
            trial_path = Path(self.config.trials_dir) / self.config.trial_name
            _write_collected_artifact(trial_path, artifact)
            return SimpleNamespace(
                task_name="example",
                trial_name=self.config.trial_name,
                verifier_result=SimpleNamespace(
                    rewards={"reward": 0.0, "partial": 0.75}
                ),
                exception_info=None,
            )

    harbor = _fake_harbor()
    harbor.Trial = Trial
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: harbor)
    outcome = asyncio.run(
        harbor_adapter.run_harbor_trial(
            task_config={},
            inference_key="inference-key",
            run_id="partial-reward-test",
            harbor_environment="e2b",
            sidecar_bundle_path=tmp_path / "bundle.zip",
            sidecar_launch_spec=json.dumps(
                {
                    "debug_enabled": False,
                    "inference_base_url": "https://api.fireworks.ai",
                }
            ),
            trials_dir=tmp_path / "trials",
            agent_import_path=OPENCODE_HARBOR_IMPORT_PATH,
            agent_version=DEFAULT_OPENCODE_VERSION,
            reward_key="partial",
        )
    )
    assert outcome.reward == 0.75
    assert outcome.rewards == {"reward": 0.0, "partial": 0.75}


def test_harbor_retry_allowlist_is_typed_and_fails_unknown_names() -> None:
    names = harbor_adapter.validate_harbor_retry_exceptions(
        harbor_adapter.DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS
    )
    assert "ApiRateLimitError" in names
    assert "AgentTimeoutError" not in names
    with pytest.raises(ValueError, match="missing required exception types"):
        harbor_adapter.validate_harbor_retry_exceptions({"ApiRateLimitError"})
    with pytest.raises(ValueError, match="unknown exception types: TypoError"):
        harbor_adapter.validate_harbor_retry_exceptions(
            {*harbor_adapter.DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS, "TypoError"}
        )

    class ApiRateLimitError(Exception):
        pass

    with pytest.raises(RecoverableRolloutError, match="retryable error"):
        harbor_adapter._raise_trial_execution_failure(
            "during create",
            ApiRateLimitError("busy"),
            names,
        )

    class AgentTimeoutError(Exception):
        pass

    with pytest.raises(RuntimeError, match="non-retryable error"):
        harbor_adapter._raise_trial_execution_failure(
            "during agent execution",
            AgentTimeoutError("deadline"),
            names,
        )


def test_e2b_stream_open_timeout_requires_exact_provider_traceback() -> None:
    marker = (
        "e2b.exceptions.TimeoutException: Request timed out: the stream didn't "
        "open within 'request_timeout' (60.0 seconds)."
    )
    exception = SimpleNamespace(
        exception_type="RuntimeError",
        exception_message="Agent install failed",
        exception_traceback=(
            'File "/site-packages/harbor/environments/e2b.py"\n'
            'File "/site-packages/e2b/envd/client_async.py"\n'
            f"{marker}"
        ),
    )
    assert harbor_adapter._is_retryable_e2b_stream_open_timeout(
        exception, harbor_environment="e2b"
    )
    assert not harbor_adapter._is_retryable_e2b_stream_open_timeout(
        exception, harbor_environment="docker"
    )
    assert not harbor_adapter._is_retryable_e2b_stream_open_timeout(
        SimpleNamespace(
            exception_type="TimeoutException",
            exception_message=marker,
            exception_traceback="unrelated timeout",
        ),
        harbor_environment="e2b",
    )


def test_e2b_timeout_classifier_is_pinned_to_harbor_environment_module() -> None:
    e2b_environment = pytest.importorskip("harbor.environments.e2b")
    module_path = Path(e2b_environment.__file__).resolve().as_posix()
    assert module_path.endswith("/harbor/environments/e2b.py")


def test_e2b_sidecar_cleanup_stream_timeout_is_retryable() -> None:
    marker = "Request timed out: the stream didn't open within 'request_timeout'"
    exception = SimpleNamespace(
        exception_type="NonZeroAgentExitCodeError",
        exception_message="agent exited",
        exception_traceback=(
            "agent traceback\n"
            f"TITO sidecar failure cleanup failed: {marker} (60.0 seconds)."
        ),
    )
    assert harbor_adapter._is_retryable_e2b_stream_open_timeout(
        exception, harbor_environment="e2b"
    )


def test_e2b_sidecar_readiness_timeout_requires_exact_wrapped_traceback() -> None:
    marker = "TITO sidecar did not become ready within 600s"
    exception = SimpleNamespace(
        exception_type="RuntimeError",
        exception_message=f"Agent install failed: {marker}",
        exception_traceback=(
            'File "/bundle/training/examples/rl/harbor/tito/sidecar.py"\n'
            f"TimeoutError: {marker}\n"
            "wrapper\n"
            f"RuntimeError: Agent install failed: {marker}"
        ),
    )
    assert harbor_adapter._is_retryable_e2b_sidecar_readiness_timeout(
        exception, harbor_environment="e2b"
    )
    assert not harbor_adapter._is_retryable_e2b_sidecar_readiness_timeout(
        exception, harbor_environment="docker"
    )
    assert not harbor_adapter._is_retryable_e2b_sidecar_readiness_timeout(
        SimpleNamespace(
            exception_type="RuntimeError",
            exception_message=marker,
            exception_traceback=f"RuntimeError: Agent install failed: {marker}",
        ),
        harbor_environment="e2b",
    )

    inner_only_wrapper = SimpleNamespace(
        exception_type="RuntimeError",
        exception_message="Agent install failed",
        exception_traceback=(
            'File "/bundle/training/examples/rl/harbor/tito/sidecar.py"\n'
            f"TimeoutError: {marker}"
        ),
    )
    assert harbor_adapter._is_retryable_e2b_sidecar_readiness_timeout(
        inner_only_wrapper,
        harbor_environment="e2b",
    )


def test_launch_spec_and_inference_key_are_redacted(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "agent": {
                        "kwargs": {
                            "sidecar_launch_spec": '{"api_key":"secret"}',
                            "other": "kept",
                        }
                    }
                }
            }
        )
    )
    harbor_adapter._redact_sidecar_spec(result_path)
    kwargs = json.loads(result_path.read_text())["config"]["agent"]["kwargs"]
    assert kwargs == {"sidecar_launch_spec": "<redacted>", "other": "kept"}

    (tmp_path / "trial.log").write_text("request used secret\n")
    harbor_adapter._redact_trial_artifacts(tmp_path, "secret")
    assert all(
        b"secret" not in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )


def test_opencode_timeout_plugin_hard_clamps_explicit_bash_timeouts() -> None:
    assert 'input.tool !== "bash"' in _TOOL_TIMEOUT_PLUGIN
    assert 'typeof output.args !== "object"' in _TOOL_TIMEOUT_PLUGIN
    assert 'hasOwnProperty.call(output.args, "timeout")' in _TOOL_TIMEOUT_PLUGIN
    assert "requested > maximum" in _TOOL_TIMEOUT_PLUGIN
    assert "output.args.timeout = maximum" in _TOOL_TIMEOUT_PLUGIN


def test_pi_extension_clamps_timeouts_and_rejects_session_branches() -> None:
    source = (
        Path(__file__).parents[2] / "examples/rl/harbor/pi/extension.ts"
    ).read_text(encoding="utf-8")
    assert 'hasOwnProperty.call(input, "timeout")' in source
    assert "requested > toolTimeoutSeconds" in source
    assert "input.timeout = toolTimeoutSeconds" in source
    assert 'pi.on("session_before_tree"' in source
    assert 'pi.on("session_before_fork"' in source
    assert source.count("return { cancel: true }") >= 2


def test_pi_overflow_retry_marks_only_the_discarded_length_turn() -> None:
    stream = "\n".join(
        (
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "stopReason": "length",
                        "responseId": "chatcmpl-turn-1",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "compaction_end",
                    "reason": "overflow",
                    "willRetry": True,
                    "aborted": False,
                }
            ),
        )
    )
    assert pi_rollout._pi_abandoned_turn_ids(stream) == {"turn-1"}


@pytest.mark.parametrize(
    ("prepare", "expected_marker", "expected_package"),
    [
        (
            lambda source, destination: prepare_opencode_tasks.prepare(
                source,
                destination,
                DEFAULT_OPENCODE_VERSION,
            ),
            "# Added by harbor_rl_opencode.prepare_opencode_tasks",
            f"opencode-ai@{DEFAULT_OPENCODE_VERSION}",
        ),
        (
            lambda source, destination: prepare_pi_tasks.prepare(
                source,
                destination,
            ),
            "# Added by fireworks TITO harbor.pi.prepare_tasks",
            "@earendil-works/pi-coding-agent@0.84.2",
        ),
        (
            lambda source, destination: prepare_mini_swe_tasks.prepare(
                source,
                destination,
            ),
            "# Added by fireworks TITO harbor.mini_swe.prepare_tasks",
            "transformers==5.5.4",
        ),
    ],
)
def test_prepared_harness_image_is_pinned_and_preserves_final_user(
    tmp_path, prepare, expected_marker, expected_package
) -> None:
    source = tmp_path / "source" / "task"
    environment = source / "environment"
    solution = source / "solution"
    environment.mkdir(parents=True)
    solution.mkdir()
    (source / "task.toml").write_text('[metadata]\nname = "task"\n')
    (environment / "Dockerfile").write_text("FROM debian:bookworm\nUSER task-user\n")
    (solution / "solve.sh").write_text("super-secret-solution\n")
    (solution / "solve.sh").chmod(0o600)
    prepared = prepare(source, tmp_path / "prepared")
    dockerfile = (prepared[0] / "environment" / "Dockerfile").read_text()
    assert expected_marker in dockerfile
    assert expected_package in dockerfile
    assert "jinja2==3.1.6" in dockerfile
    assert "numpy==2.4.6" in dockerfile
    assert "import aiohttp, httpx, jinja2," in dockerfile
    assert "zstandard" not in dockerfile
    assert dockerfile.rstrip().endswith("USER task-user")
    assert not (prepared[0] / "solution" / "solve.sh").stat().st_mode & stat.S_IROTH


def test_prepare_pi_tasks_selects_requested_tasks_in_order(tmp_path) -> None:
    source = tmp_path / "source"
    for name in ("task-a", "task-b"):
        environment = source / name / "environment"
        environment.mkdir(parents=True)
        (source / name / "task.toml").write_text("", encoding="utf-8")
        (environment / "Dockerfile").write_text(
            "FROM debian:bookworm\n", encoding="utf-8"
        )

    prepared = prepare_pi_tasks.prepare(
        source,
        tmp_path / "prepared",
        task_names=("task-b", "task-a"),
    )

    assert [path.name for path in prepared] == ["task-b", "task-a"]


def test_e2b_template_prebuild_builds_missing_alias_once(monkeypatch, tmp_path) -> None:
    environments = []

    class Environment:
        def __init__(self, name):
            self._template_name = f"template-{name}"
            self.exists = name == "existing"
            self.builds = 0

        async def _does_template_exist(self):
            return self.exists

        async def _create_template(self):
            self.builds += 1
            self.exists = True

    class Trial:
        @classmethod
        async def create(cls, config):
            environment = Environment(config["name"])
            environments.append(environment)
            return SimpleNamespace(agent_environment=environment)

    monkeypatch.setattr(
        e2b_templates,
        "_require_harbor",
        lambda: SimpleNamespace(Trial=Trial),
    )
    monkeypatch.setattr(
        e2b_templates,
        "task_name_from_row",
        lambda row: row["task_name"],
    )
    monkeypatch.setattr(
        e2b_templates,
        "task_config_from_row",
        lambda row: row,
    )
    monkeypatch.setattr(
        e2b_templates,
        "_build_trial_config",
        lambda _harbor, *, task_config, **_kwargs: {"name": task_config["task_name"]},
    )

    records = asyncio.run(
        e2b_templates.prebuild_e2b_templates(
            [{"task_name": "existing"}, {"task_name": "new"}],
            trials_dir=tmp_path,
            agent_import_path="agent:Class",
            agent_version="1",
            agent_provider="provider",
            context_limit=4096,
            output_limit=1024,
        )
    )

    assert [(record.task_name, record.existed) for record in records] == [
        ("existing", True),
        ("new", False),
    ]
    assert [environment.builds for environment in environments] == [0, 1]


@pytest.mark.parametrize("base_image", ["mutable:latest", "image@sha256:bad"])
def test_prepared_task_rejects_mutable_base_image(tmp_path, base_image) -> None:
    source = tmp_path / "source" / "task"
    environment = source / "environment"
    environment.mkdir(parents=True)
    (source / "task.toml").write_text('[metadata]\nname = "task"\n')
    (environment / "Dockerfile").write_text("FROM base\n")
    with pytest.raises(ValueError, match="immutable image@sha256"):
        prepare_pi_tasks.prepare(
            source,
            tmp_path / "prepared",
            base_image=base_image,
        )
