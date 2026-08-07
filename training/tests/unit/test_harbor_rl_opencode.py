from __future__ import annotations

import asyncio
import gzip
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.examples.rl.harbor_rl_opencode import harbor as harbor_adapter
from training.examples.rl.harbor_rl_opencode import prepare_opencode_tasks
from training.examples.rl.harbor_rl_opencode import rollout
from training.examples.rl.harbor_rl_opencode import train as train_example
from training.examples.rl.harbor_rl_opencode import train_serverless
from training.examples.rl.harbor_rl_opencode.harbor import DEFAULT_OPENCODE_VERSION
from training.utils.rl.async_rl.errors import RecoverableRolloutError
from training.utils.rl.rollout import RolloutRun, RolloutSample


@pytest.fixture
def installed_harbor():
    return pytest.importorskip(
        "harbor",
        reason="requires the example-only Harbor dependency",
    )


def _task_row() -> dict:
    return {
        "task_name": "example",
        harbor_adapter.HARBOR_TASK_CONFIG_KEY: {"path": "/tasks/example"},
    }


def _setup():
    return SimpleNamespace(
        tokenizer=object(),
        sample_kwargs={
            "max_tokens": 128,
            "max_seq_len": 1024,
            "temperature": 1.0,
            "logprobs": True,
            "include_routing_matrix": True,
            "echo": False,
        },
        extras={},
        _sampler=object(),
    )


@pytest.fixture(autouse=True)
def _fake_sampler_and_task_config(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "build_deployment_sampler",
        lambda setup: setup._sampler,
    )
    monkeypatch.setattr(
        rollout,
        "task_config_from_row",
        lambda row: row[harbor_adapter.HARBOR_TASK_CONFIG_KEY],
    )


def test_train_wires_dabstep_global_batch_controls(monkeypatch):
    train_tasks = tuple(f"dabstep-{index}" for index in range(67))
    holdout_tasks = tuple(f"dabstep-{index}" for index in range(67, 75))
    manifest = SimpleNamespace(
        train_tasks=train_tasks,
        holdout_tasks=holdout_tasks,
        profile={name: {"attempts": 5, "solved": 2} for name in train_tasks},
    )
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "harbor-opencode",
            "--dabstep-manifest",
            "/shared/manifest.json",
            "--max-rows",
            "400",
            "--completions-per-prompt",
            "8",
            "--prompt-groups-per-step",
            "8",
            "--pipeline-chunks-per-step",
            "2",
            "--min-group-size",
            "8",
            "--max-incomplete-group-retries",
            "2",
        ],
    )
    monkeypatch.setattr(
        train_example.DABstepManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    manifest.verify_task_root = lambda path: None
    monkeypatch.setattr(
        train_example,
        "load_harbor_rows",
        lambda *args, **kwargs: [
            {
                "task_name": name,
                harbor_adapter.HARBOR_TASK_CONFIG_KEY: {"path": f"/tasks/{name}"},
            }
            for name in (*train_tasks, *holdout_tasks)
        ],
    )
    monkeypatch.setattr(
        train_example,
        "main",
        lambda config, **kwargs: captured.update(config=config, kwargs=kwargs),
    )

    train_example.run()

    config = captured["config"]
    assert config.max_rows == 400
    assert config.completions_per_prompt == 8
    assert config.prompt_groups_per_step == 8
    assert config.pipeline_chunks_per_step == 2
    assert config.min_group_size == 8
    assert config.max_incomplete_group_retries == 2
    assert config.max_head_offpolicy_versions == 0
    assert config.shuffle is False
    assert len(captured["kwargs"]["rows"]) == 400
    assert captured["kwargs"]["evaluation_fn"] is not None
    assert captured["kwargs"]["evaluation_interval"] == 3


def test_serverless_cli_exposes_long_run_optimizer_controls(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "harbor-opencode-serverless",
            "--manifest",
            "/shared/manifest.json",
            "--harbor-dataset",
            "/shared/tasks",
            "--max-rows",
            "1200",
            "--max-seq-len",
            "196608",
            "--lora-rank",
            "128",
            "--adam-beta2",
            "0.95",
            "--adam-epsilon",
            "1e-12",
            "--weight-decay",
            "0",
            "--evaluation-interval",
            "5",
            "--dcp-save-interval",
            "40",
            "--terminal-failure-reward",
            "0",
        ],
    )

    args = train_serverless.parse_args()

    assert args.max_rows == 1200
    assert args.max_seq_len == 196608
    assert args.lora_rank == 128
    assert args.adam_beta2 == 0.95
    assert args.adam_epsilon == 1e-12
    assert args.weight_decay == 0.0
    assert args.evaluation_interval == 5
    assert args.dcp_save_interval == 40
    assert args.terminal_failure_reward == 0.0


@pytest.mark.parametrize(
    ("row_offset", "step_offset", "message"),
    [
        (121, 16, "8-group optimizer boundary"),
        (120, 14, "--step-offset must equal --row-offset / 8"),
    ],
)
def test_serverless_resume_offsets_align_with_optimizer_batches(
    monkeypatch,
    row_offset,
    step_offset,
    message,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "harbor-opencode-serverless",
            "--manifest",
            "manifest.json",
            "--harbor-dataset",
            "tasks",
            "--row-offset",
            str(row_offset),
            "--step-offset",
            str(step_offset),
            "--resume-from",
            "checkpoint",
            "--selector-state-in",
            "selector.json",
        ],
    )

    with pytest.raises(ValueError, match=message):
        train_serverless.run()


def test_evaluation_uses_holdout_row_without_mutating_selector(monkeypatch):
    class Selector:
        row_calls = 0
        record_calls = 0

        async def row_for_group(self, _cursor_index):
            self.row_calls += 1
            return {
                "task_name": "training-task",
                harbor_adapter.HARBOR_TASK_CONFIG_KEY: {"path": "/tasks/training"},
            }

        async def record(self, *_args):
            self.record_calls += 1

    selector = Selector()
    setup = _setup()
    setup.extras["task_selector"] = selector
    runner = rollout.make_rollout_fn(setup)
    seen_tasks: list[str] = []

    async def run_opencode(*, task_name, **_kwargs):
        seen_tasks.append(task_name)
        return RolloutRun(
            segments=[
                RolloutSample(
                    tokens=[1, 2],
                    logprobs=[0.0, -0.1],
                    loss_mask=[0, 1],
                    reward=1.0,
                )
            ],
            run_id="eval-run",
        )

    monkeypatch.setattr(runner, "_run_opencode", run_opencode)
    asyncio.run(runner(_task_row(), sample_index=0, evaluation=True))

    assert seen_tasks == ["example"]
    assert selector.row_calls == 0
    assert selector.record_calls == 0


def test_training_records_discarded_trajectory_in_selector(monkeypatch):
    class Selector:
        def __init__(self):
            self.records = []

        async def row_for_group(self, _cursor_index):
            return {
                "task_name": "training-task",
                harbor_adapter.HARBOR_TASK_CONFIG_KEY: {"path": "/tasks/training"},
            }

        async def record(self, *args):
            self.records.append(args)

    selector = Selector()
    setup = _setup()
    setup.extras["task_selector"] = selector
    runner = rollout.make_rollout_fn(setup)

    async def discard(**_kwargs):
        return None

    monkeypatch.setattr(runner, "_run_opencode", discard)

    result = asyncio.run(runner(_task_row(), cursor_index=3, sample_index=5))

    assert result is None
    assert selector.records == [(3, 5, None)]


def test_prepare_opencode_tasks_restores_final_stage_user(tmp_path):
    source = tmp_path / "task"
    environment = source / "environment"
    environment.mkdir(parents=True)
    (source / "task.toml").write_text("", encoding="utf-8")
    (environment / "Dockerfile").write_text(
        "FROM debian:bookworm\nUSER app:workers\n",
        encoding="utf-8",
    )

    [prepared] = prepare_opencode_tasks.prepare(
        source,
        tmp_path / "prepared",
        "1.18.10",
    )

    dockerfile = (prepared / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "npm install -g opencode-ai@1.18.10" in dockerfile
    assert dockerfile.rstrip().endswith("USER app:workers")


def test_prepare_opencode_tasks_writes_isolated_compose(tmp_path):
    source = tmp_path / "task"
    environment = source / "environment"
    environment.mkdir(parents=True)
    (source / "task.toml").write_text("", encoding="utf-8")
    (environment / "Dockerfile").write_text(
        "FROM debian:bookworm\n",
        encoding="utf-8",
    )

    [prepared] = prepare_opencode_tasks.prepare(
        source,
        tmp_path / "prepared",
        "1.18.8",
        internal_network="dabstep-isolated",
    )

    compose = (prepared / "environment" / "docker-compose.yaml").read_text(
        encoding="utf-8"
    )
    assert "external: true" in compose
    assert "name: dabstep-isolated" in compose
    assert "${CONTEXT_DIR}" in compose


def test_prepare_opencode_tasks_rejects_symlinked_write_path(tmp_path):
    source = tmp_path / "task"
    source.mkdir()
    (source / "task.toml").write_text("", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_dockerfile = outside / "Dockerfile"
    outside_dockerfile.write_text("FROM debian:bookworm\n", encoding="utf-8")
    (source / "environment").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="write path contains a symlink"):
        prepare_opencode_tasks.prepare(
            source,
            tmp_path / "prepared",
            "1.18.8",
        )

    assert outside_dockerfile.read_text(encoding="utf-8") == "FROM debian:bookworm\n"


def test_final_stage_user_ignores_an_earlier_build_stage():
    dockerfile = """
FROM debian:bookworm AS builder
USER builder
FROM debian:bookworm
RUN true
"""

    assert prepare_opencode_tasks._final_stage_user(dockerfile) is None


def test_full_sequence_router_replay_is_rejected():
    setup = _setup()
    setup.sample_kwargs["echo"] = True

    with pytest.raises(ValueError, match="completion-only Router Replay"):
        rollout.make_rollout_fn(setup)


def test_runner_advertises_the_full_trainer_context():
    runner = rollout.make_rollout_fn(_setup())

    assert runner._context_limit == 1024


def test_rollout_retries_may_be_disabled():
    setup = _setup()
    setup.extras["rollout_retries"] = 0

    runner = rollout.make_rollout_fn(setup)

    assert runner._rollout_retries == 0

    setup.extras["rollout_retries"] = -1
    with pytest.raises(ValueError, match="must be >= 0"):
        rollout.make_rollout_fn(setup)

    setup.extras = {"terminal_failure_reward": float("nan")}
    with pytest.raises(ValueError, match="must be finite"):
        rollout.make_rollout_fn(setup)


def test_trajectory_artifact_preserves_trace_and_analysis(tmp_path):
    setup = _setup()
    setup.extras["harbor_trials_dir"] = str(tmp_path)
    runner = rollout.make_rollout_fn(setup)
    session = rollout.OpenCodePolicySession(
        run_id="harbor-opencode:example:0:0:0:0",
        max_context_tokens=1024,
        request_traces=[
            {
                "trainable": True,
                "messages": [{"role": "user", "content": "task"}],
                "tools": [{"type": "function"}],
                "prompt_ids": [1, 2],
                "completion_ids": [3],
                "completion_logprobs": [-0.1],
                "completion_raw_logprobs": [-0.2],
                "completion_routing_matrices": ["route"],
                "completion_text": "done",
                "assistant_message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
                "turn_kind": "new",
                "matched_prefix_len": 0,
            },
            {
                "trainable": True,
                "messages": [{"role": "user", "content": "compacted"}],
                "tools": [{"type": "function"}],
                "prompt_ids": [9],
                "completion_ids": [10],
                "completion_logprobs": [-0.3],
                "completion_raw_logprobs": [-0.4],
                "completion_routing_matrices": ["route-2"],
                "completion_text": "continued",
                "assistant_message": {
                    "role": "assistant",
                    "content": "continued",
                },
                "finish_reason": "stop",
                "turn_kind": "wipe",
                "matched_prefix_len": 0,
            },
        ],
    )
    outcome = harbor_adapter.HarborTrialOutcome(
        task_name="example",
        trial_name="native-trial",
        reward=1.0,
        rewards={"reward": 1.0},
        exception_type=None,
        exception_message=None,
    )

    artifact = runner._write_trajectory_artifact(
        session=session,
        task_name="example",
        retry_index=0,
        status="completed",
        outcome=outcome,
        segment_shapes=[
            {
                "tokens": 3,
                "loss_mask": 3,
                "logprobs": 3,
                "raw_logprobs": 3,
                "routing_matrices": 2,
                "trainable_tokens": 1,
            }
        ],
    )

    assert artifact is not None
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["status"] == "completed"
    assert document["outcome"]["reward"] == 1.0
    assert document["invariants"] == {
        "trace_integrity_ok": True,
        "segment_arrays_aligned": True,
        "trajectory_issue_count": 0,
    }
    assert document["analysis"]["summary"]["segment_count"] == 2
    assert document["analysis"]["summary"]["turn_count"] == 2
    assert document["request_traces"][0]["completion_ids"] == [3]


def test_result_validation_retries_then_discards_the_rollout(monkeypatch, tmp_path):
    setup = _setup()
    setup.extras["harbor_trials_dir"] = str(tmp_path)
    setup.extras["rollout_retries"] = 3
    runner = rollout.make_rollout_fn(setup)
    attempts = 0

    class FailingResultServer:
        port = 9123

        def __init__(self):
            self.sessions = {}

        def register_session(self, run_id):
            key = f"policy-{len(self.sessions)}"
            self.sessions[key] = rollout.OpenCodePolicySession(
                run_id=run_id,
                max_context_tokens=1024,
                sampling_failures=1,
                trace_integrity_failures=1,
                trace_integrity_error=(
                    "TraceIntegrityError: completion sampling_logprobs are misaligned"
                ),
            )
            return key

        async def pop_session(self, key):
            return self.sessions.pop(key)

    async def run_trial(**_kwargs):
        nonlocal attempts
        attempts += 1
        return harbor_adapter.HarborTrialOutcome(
            task_name="example",
            trial_name=f"native-trial-{attempts}",
            reward=0.0,
            rewards={"reward": 0.0},
            exception_type="NonZeroAgentExitCodeError",
            exception_message="policy returned 503",
        )

    async def skip_retry_wait(**_kwargs):
        return None

    runner._openai_server = FailingResultServer()
    monkeypatch.setattr(runner, "_run_trial", run_trial)
    monkeypatch.setattr(runner, "_wait_to_retry", skip_retry_wait)

    result = asyncio.run(
        runner._run_opencode(
            task_config={"path": "/tasks/example"},
            task_name="example",
            run_id="harbor-opencode:example:0:0:0:0",
        )
    )

    assert result is None
    assert attempts == 4
    artifacts = sorted((tmp_path / "_fireworks_trajectories").glob("*.json.gz"))
    assert len(artifacts) == 4
    statuses = []
    for artifact in artifacts:
        with gzip.open(artifact, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
        statuses.append(document["status"])
        assert document["session"]["trace_integrity_failures"] == 1
        assert document["invariants"]["trace_integrity_ok"] is False
    assert statuses == ["retry", "retry", "retry", "failed"]


def test_non_recoverable_trial_failure_discards_without_retry(monkeypatch):
    setup = _setup()
    setup.extras["rollout_retries"] = 3
    runner = rollout.make_rollout_fn(setup)
    attempts = 0

    class Server:
        port = 9123

        def __init__(self):
            self.sessions = {}
            self.pop_calls = 0

        def register_session(self, run_id):
            self.sessions["policy-key"] = rollout.OpenCodePolicySession(
                run_id=run_id,
                max_context_tokens=1024,
            )
            return "policy-key"

        async def pop_session(self, key):
            self.pop_calls += 1
            return self.sessions.pop(key)

    async def fail_trial(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid task contract")

    server = Server()
    runner._openai_server = server
    monkeypatch.setattr(runner, "_run_trial", fail_trial)

    result = asyncio.run(
        runner._run_opencode(
            task_config={"path": "/tasks/example"},
            task_name="example",
            run_id="harbor-opencode:example:0:0:0:0",
        )
    )

    assert result is None
    assert attempts == 1
    assert server.pop_calls == 1
    assert server.sessions == {}


def test_missing_policy_session_retries_without_second_pop_per_attempt(monkeypatch):
    setup = _setup()
    setup.extras["rollout_retries"] = 2
    runner = rollout.make_rollout_fn(setup)

    class Server:
        port = 9123

        def __init__(self):
            self.pop_calls = 0

        @staticmethod
        def register_session(_run_id):
            return "missing-policy-key"

        async def pop_session(self, _key):
            self.pop_calls += 1
            raise KeyError("missing-policy-key")

    async def successful_trial(**_kwargs):
        return harbor_adapter.HarborTrialOutcome(
            task_name="example",
            trial_name="native-trial",
            reward=1.0,
            rewards={"reward": 1.0},
            exception_type=None,
            exception_message=None,
        )

    server = Server()
    runner._openai_server = server
    monkeypatch.setattr(runner, "_run_trial", successful_trial)
    monkeypatch.setattr(runner, "_wait_to_retry", lambda **_kwargs: asyncio.sleep(0))

    result = asyncio.run(
        runner._run_opencode(
            task_config={"path": "/tasks/example"},
            task_name="example",
            run_id="harbor-opencode:example:0:0:0:0",
        )
    )

    assert result is None
    assert server.pop_calls == 3


def test_valid_verifier_reward_survives_non_integrity_sampling_failure(monkeypatch):
    runner = rollout.make_rollout_fn(_setup())
    session = rollout.OpenCodePolicySession(
        run_id="harbor-opencode:example:0:0:0:0",
        max_context_tokens=1024,
        sampling_failures=1,
        last_error="TimeoutError: sampler unavailable",
    )
    session.drain = lambda: [SimpleNamespace(metadata={})]
    monkeypatch.setattr(
        rollout,
        "token_segment_to_sample",
        lambda _segment, reward: RolloutSample(
            tokens=[1, 2],
            logprobs=[0.0, -0.1],
            loss_mask=[0, 1],
            reward=reward,
        ),
    )
    outcome = harbor_adapter.HarborTrialOutcome(
        task_name="example",
        trial_name="native-trial",
        reward=0.5,
        rewards={"reward": 0.5},
        exception_type="AgentTimeoutError",
        exception_message="terminal task outcome",
    )

    result = asyncio.run(
        runner._opencode_result(
            session=session,
            outcome=outcome,
            run_id=session.run_id,
            retry_index=0,
        )
    )

    assert result.segments[0].reward == 0.5


def test_concurrent_rollouts_wait_for_policy_server_port(monkeypatch):
    class FakeSession:
        match_events = [{"kind": "new"}]
        history_wipes = 0
        auxiliary_turns = 0
        sampling_failures = 0
        trace_integrity_failures = 0
        trace_integrity_error = None
        context_overflows = 0
        toolless_tool_turns = 0

        @staticmethod
        def drain():
            return [SimpleNamespace(metadata={})]

    class SlowStartingServer:
        def __init__(self, **_kwargs):
            self.port = 0
            self.sessions = {}
            self.next_key = 0

        async def start(self):
            await asyncio.sleep(0.01)
            self.port = 9123

        def register_session(self, _run_id):
            self.next_key += 1
            key = f"key-{self.next_key}"
            self.sessions[key] = FakeSession()
            return key

        async def pop_session(self, key):
            return self.sessions.pop(key)

    async def run_trial(**kwargs):
        assert kwargs["policy_port"] == 9123
        return harbor_adapter.HarborTrialOutcome(
            task_name="example",
            trial_name="trial-example",
            reward=1.0,
            rewards={"reward": 1.0},
            exception_type=None,
            exception_message=None,
        )

    monkeypatch.setattr(rollout, "OpenCodePolicyServer", SlowStartingServer)
    monkeypatch.setattr(rollout, "run_harbor_trial", run_trial)
    monkeypatch.setattr(
        rollout,
        "token_segment_to_sample",
        lambda _segment, reward: RolloutSample(
            tokens=[1, 2],
            logprobs=[0.0, -0.1],
            loss_mask=[0, 1],
            reward=reward,
        ),
    )
    rollout_fn = rollout.make_rollout_fn(_setup())

    async def run_concurrently():
        return await asyncio.gather(
            rollout_fn(_task_row(), sample_index=0),
            rollout_fn(_task_row(), sample_index=1),
        )

    first, second = asyncio.run(run_concurrently())

    assert first is not None
    assert second is not None
    assert first.run_id != second.run_id


def test_cancelled_rollout_discards_policy_session(monkeypatch):
    runner = rollout.make_rollout_fn(_setup())

    class Server:
        port = 9123

        def __init__(self):
            self.discarded = []

        @staticmethod
        def register_session(_run_id):
            return "policy-key"

        def discard_session(self, key):
            self.discarded.append(key)

    server = Server()

    async def cancel_trial(**_kwargs):
        raise asyncio.CancelledError

    runner._openai_server = server
    monkeypatch.setattr(runner, "_run_trial", cancel_trial)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runner._run_opencode(
                task_config={"path": "/tasks/example"},
                task_name="example",
                run_id="harbor-opencode:example:0:0:0:0",
            )
        )

    assert server.discarded == ["policy-key"]


def test_trial_config_source_must_be_mapping_or_path():
    with pytest.raises(TypeError, match="mapping or YAML path"):
        harbor_adapter.load_harbor_trial_config(42)


def test_trial_config_loads_yaml_mapping(tmp_path):
    config_path = tmp_path / "trial.yaml"
    config_path.write_text(
        "timeout_multiplier: 2\nenvironment:\n  type: docker\n  override_cpus: 4\n"
    )

    assert harbor_adapter.load_harbor_trial_config(config_path) == {
        "timeout_multiplier": 2,
        "environment": {"type": "docker", "override_cpus": 4},
    }


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


def _fake_harbor(*, reward=None, exception=None):
    seen = SimpleNamespace(config=None)

    class _Trial:
        @classmethod
        async def create(cls, config):
            seen.config = config
            return cls()

        async def run(self):
            rewards = None if reward is None else {"reward": reward, "tests": 1}
            return SimpleNamespace(
                task_name="example",
                trial_name=seen.config.trial_name,
                verifier_result=(
                    SimpleNamespace(rewards=rewards) if rewards is not None else None
                ),
                exception_info=exception,
            )

    fake = SimpleNamespace(
        EnvironmentType=_EnvironmentType,
        TrialConfig=_Config,
        Trial=_Trial,
    )
    return fake, seen


def test_trial_adapter_uses_local_docker_and_opencode(monkeypatch, tmp_path):
    fake, seen = _fake_harbor(reward=0.75)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    outcome = asyncio.run(
        harbor_adapter.run_harbor_trial(
            task_config={"path": "/tasks/example"},
            policy_key="trial-key",
            run_id="harbor-opencode:example:0:0:0:0",
            policy_port=9123,
            context_limit=262144,
            output_limit=2048,
            trial_config={
                "timeout_multiplier": 2.0,
                "agent": {
                    "name": "terminus-2",
                    "override_timeout_sec": 321,
                    "env": {"AGENT_VALUE": "kept"},
                    "kwargs": {"not_for_fireworks": True},
                },
                "environment": {
                    "type": "docker",
                    "delete": True,
                    "override_cpus": 4,
                    "kwargs": {"network": "test"},
                },
                "verifier": {"override_timeout_sec": 77},
                "artifacts": ["/logs/result.json"],
            },
            trials_dir=tmp_path,
        )
    )

    assert outcome.reward == 0.75
    assert outcome.rewards == {"reward": 0.75, "tests": 1.0}
    assert outcome.environment_type == "docker"
    assert seen.config.task == {"path": "/tasks/example"}
    assert seen.config.environment.type is _EnvironmentType.DOCKER
    assert seen.config.environment.delete is True
    assert seen.config.environment.force_build is True
    assert seen.config.environment.override_cpus == 4
    assert seen.config.environment.kwargs == {"network": "test"}
    assert seen.config.agent.override_timeout_sec == 321
    assert seen.config.agent.name is None
    assert seen.config.agent.env == {"AGENT_VALUE": "kept"}
    assert seen.config.agent.model_name == "fireworks-rl/policy"
    assert seen.config.agent.kwargs == {
        "policy_base_url": "http://{host}:9123/v1",
        "policy_api_key": "trial-key",
        "context_limit": 262144,
        "output_limit": 2048,
        "version": DEFAULT_OPENCODE_VERSION,
    }
    assert "harbor_rl_opencode.opencode:ConfigurableOpenCode" in (
        seen.config.agent.import_path
    )
    assert seen.config.timeout_multiplier == 2.0
    assert seen.config.verifier.override_timeout_sec == 77
    assert seen.config.artifacts == ["/logs/result.json"]


def test_trial_result_redacts_local_policy_key(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "config": {
                    "agent": {
                        "kwargs": {
                            "policy_api_key": "trial-secret",
                            "other": "kept",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    harbor_adapter._redact_policy_key(result_path)

    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "config": {
            "agent": {
                "kwargs": {
                    "policy_api_key": "<redacted>",
                    "other": "kept",
                }
            }
        }
    }


def test_redaction_failure_does_not_discard_completed_trial(
    monkeypatch,
    tmp_path,
    caplog,
):
    fake, _ = _fake_harbor(reward=0.75)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    def fail_redaction(_path):
        raise OSError("read-only artifact")

    monkeypatch.setattr(harbor_adapter, "_redact_policy_key", fail_redaction)

    outcome = asyncio.run(
        harbor_adapter.run_harbor_trial(
            task_config={"path": "/tasks/example"},
            policy_key="trial-key",
            run_id="run",
            policy_port=9123,
            trials_dir=tmp_path,
        )
    )

    assert outcome.reward == 0.75
    assert "Could not redact the local policy key" in caplog.text


def test_trial_config_template_validates_with_real_harbor(
    tmp_path,
    installed_harbor,
):
    template = {
        "timeout_multiplier": 2.0,
        "environment": {
            "type": "docker",
            "delete": True,
            "override_cpus": 4,
        },
        "verifier": {"override_timeout_sec": 77},
    }

    config = harbor_adapter._build_trial_config(
        installed_harbor,
        template=template,
        task_config=installed_harbor.TrialTaskConfig(path=tmp_path / "task"),
        policy_key="policy-key",
        run_id="run",
        trials_dir=tmp_path,
        policy_port=9123,
    )

    assert config.environment.type is installed_harbor.EnvironmentType.DOCKER
    assert config.environment.delete is True
    assert config.environment.force_build is True
    assert config.environment.override_cpus == 4
    assert config.verifier.override_timeout_sec == 77
    assert config.agent.name is None
    assert config.agent.override_timeout_sec is None
    assert template["environment"]["delete"] is True


@pytest.mark.parametrize(
    "exception_type",
    ["AgentTimeoutError", "NonZeroAgentExitCodeError", "VerifierTimeoutError"],
)
def test_terminal_trial_exception_without_reward_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_type: str,
) -> None:
    exception = SimpleNamespace(
        exception_type=exception_type,
        exception_message="terminal task outcome",
    )
    fake, _ = _fake_harbor(reward=None, exception=exception)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    with pytest.raises(RecoverableRolloutError, match=exception_type):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={"path": "/tasks/example"},
                policy_key="policy-key",
                run_id="run",
                policy_port=9123,
                trials_dir=tmp_path,
            )
        )


def test_terminal_failure_reward_is_an_explicit_rollout_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exception = SimpleNamespace(
        exception_type="NonZeroAgentExitCodeError",
        exception_message="terminal task outcome",
    )
    fake, _ = _fake_harbor(reward=None, exception=exception)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    outcome = asyncio.run(
        harbor_adapter.run_harbor_trial(
            task_config={"path": "/tasks/example"},
            policy_key="policy-key",
            run_id="run",
            policy_port=9123,
            trials_dir=tmp_path,
            terminal_failure_reward=0.0,
        )
    )

    assert outcome.reward == 0.0
    assert outcome.rewards == {"reward": 0.0}


def test_missing_reward_without_terminal_exception_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake, _ = _fake_harbor(reward=None, exception=None)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    with pytest.raises(RecoverableRolloutError, match="verifier produced no reward"):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={"path": "/tasks/example"},
                policy_key="policy-key",
                run_id="run",
                policy_port=9123,
                trials_dir=tmp_path,
            )
        )


def test_infrastructure_trial_exception_is_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exception = SimpleNamespace(
        exception_type="EnvironmentStartTimeoutError",
        exception_message="provider unavailable",
    )
    fake, _ = _fake_harbor(reward=None, exception=exception)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    with pytest.raises(RecoverableRolloutError, match="provider unavailable"):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={"path": "/tasks/example"},
                policy_key="policy-key",
                run_id="run",
                policy_port=9123,
                trials_dir=tmp_path,
            )
        )


def test_unsupported_harbor_environment_is_rejected(monkeypatch, tmp_path):
    fake, _ = _fake_harbor(reward=1.0)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    with pytest.raises(ValueError, match="local Docker"):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={"path": "/tasks/example"},
                policy_key="policy-key",
                run_id="run",
                policy_port=9123,
                trial_config={"environment": {"type": "e2b"}},
                trials_dir=tmp_path,
            )
        )


@pytest.mark.parametrize(
    "trial_config,match",
    [
        ({"install_only": True}, "install_only"),
        ({"source_trial": {"trial_id": "old"}}, "source_trial"),
    ],
)
def test_non_training_trial_modes_are_rejected(
    monkeypatch,
    tmp_path,
    trial_config,
    match,
):
    fake, _ = _fake_harbor(reward=1.0)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    with pytest.raises(ValueError, match=match):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={"path": "/tasks/example"},
                policy_key="policy-key",
                run_id="run",
                policy_port=9123,
                trial_config=trial_config,
                trials_dir=tmp_path,
            )
        )


def test_policy_server_port_is_required(monkeypatch, tmp_path):
    fake, _ = _fake_harbor(reward=1.0)
    monkeypatch.setattr(harbor_adapter, "_require_harbor", lambda: fake)

    with pytest.raises(ValueError, match="policy server port"):
        asyncio.run(
            harbor_adapter.run_harbor_trial(
                task_config={"path": "/tasks/example"},
                policy_key="policy-key",
                run_id="run",
                policy_port=0,
                trials_dir=tmp_path,
            )
        )
