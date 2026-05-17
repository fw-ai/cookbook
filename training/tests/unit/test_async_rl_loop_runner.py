"""FIR2-1599: artifact writer + resource-ready callback tests for async_rl_loop.

These tests cover two layers of the managed runtime artifact contract:

1. **Deterministic helpers** (``_emit_resources_ready``,
   ``_estimate_total_steps``) -- callback invocation + progress-bar
   total formula.
2. **End-to-end through ``async_rl_loop.main`` with fakes** --
   exercises the real ``RunnerIO`` writes that satisfy the explicit
   FIR2-1599 acceptance criteria:
   * final ``COMPLETED`` status is written;
   * final ``FAILED`` status is written when the loop raises;
   * ``RunnerIO.append_metrics`` is called during an async run.

The end-to-end tests monkeypatch every heavy dependency
(``setup_infra``, ``run_async_rl_loop``, the SDK managers,
``TrainingCheckpoints``, etc.) so the recipe itself runs in-process
without tinker, the Fireworks SDK, or a real deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.recipes import async_rl_loop
from training.utils import DeployConfig, RunnerConfig


@dataclass
class _FakeInfra:
    """Subset of ``training.utils.rl.Infra`` exercised by the callback."""

    deployment_id: str | None = None
    policy_job_id: str | None = None
    reference_job_id: str | None = None


class TestEmitResourcesReady:
    """``_emit_resources_ready`` is the FIR2-1599 resource-ready hook surface."""

    def test_noop_when_callback_is_none(self) -> None:
        """A missing callback is silently skipped (no kwargs required)."""
        infra = _FakeInfra(deployment_id="dep-1", policy_job_id="pol-1")
        async_rl_loop._emit_resources_ready(infra, None)

    def test_invokes_callback_with_resource_ids(self) -> None:
        """The callback receives deployment/policy/reference IDs as kwargs."""
        captured: list[dict[str, str | None]] = []

        def callback(**kwargs: object) -> None:
            captured.append(dict(kwargs))

        infra = _FakeInfra(
            deployment_id="dep-abc",
            policy_job_id="policy-xyz",
            reference_job_id="ref-123",
        )
        async_rl_loop._emit_resources_ready(infra, callback)

        assert captured == [
            {
                "deployment_id": "dep-abc",
                "policy_job_id": "policy-xyz",
                "reference_job_id": "ref-123",
            }
        ]

    def test_passes_none_for_missing_reference_job(self) -> None:
        """LoRA shared / forward-only setups have ``reference_job_id=None``."""
        captured: list[dict[str, str | None]] = []

        def callback(**kwargs: object) -> None:
            captured.append(dict(kwargs))

        infra = _FakeInfra(
            deployment_id="dep-1",
            policy_job_id="pol-1",
            reference_job_id=None,
        )
        async_rl_loop._emit_resources_ready(infra, callback)

        assert captured == [
            {"deployment_id": "dep-1", "policy_job_id": "pol-1", "reference_job_id": None}
        ]

    def test_callback_exceptions_do_not_propagate(self, caplog: pytest.LogCaptureFixture) -> None:
        """A broken artifact writer must never abort training."""
        infra = _FakeInfra(deployment_id="dep-1", policy_job_id="pol-1")

        def boom(**_kwargs: object) -> None:
            raise RuntimeError("disk full")

        with caplog.at_level("WARNING"):
            async_rl_loop._emit_resources_ready(infra, boom)

        assert any("on_resources_ready" in rec.message for rec in caplog.records)


class TestEstimateTotalSteps:
    """``_estimate_total_steps`` mirrors the FIR2-1599 progress formula."""

    def test_fresh_run_single_minibatch(self) -> None:
        """No resume, one minibatch per rollout -> one step per row."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=1,
            ppo_n_minibatches=1,
        )
        assert total == 10

    def test_fresh_run_batches_rows(self) -> None:
        """With ``prompt_groups_per_step=4`` we expect ceil(10/4) = 3 rollouts."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=4,
            ppo_n_minibatches=1,
        )
        assert total == 3

    def test_ppo_inner_minibatches_multiply_steps(self) -> None:
        """Each rollout batch fans out into ``ppo_n_minibatches`` optim steps."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=2,
            ppo_n_minibatches=3,
        )
        # ceil(10/2) = 5 rollout batches, each 3 optim steps -> 15
        assert total == 15

    def test_resume_adds_step_offset(self) -> None:
        """Resume from step N -> total includes N + remaining estimate."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=4,
            total_items=10,
            prior_rows_consumed=6,
            prompt_groups_per_step=2,
            ppo_n_minibatches=1,
        )
        # remaining = 10 - 6 = 4 rows; ceil(4/2) = 2 rollouts; 4 + 2 = 6
        assert total == 6

    def test_clamps_negative_remaining(self) -> None:
        """If the cursor is past total_items we still return at least the offset."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=7,
            total_items=10,
            prior_rows_consumed=12,
            prompt_groups_per_step=1,
            ppo_n_minibatches=1,
        )
        assert total == 7

    def test_invalid_groups_per_step_floors_to_one(self) -> None:
        """A zero/negative ``prompt_groups_per_step`` is treated as 1."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=5,
            prior_rows_consumed=0,
            prompt_groups_per_step=0,
            ppo_n_minibatches=1,
        )
        assert total == 5

    def test_zero_total_items_returns_offset_only(self) -> None:
        """An empty dataset reports the step offset as the total."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=3,
            total_items=0,
            prior_rows_consumed=0,
            prompt_groups_per_step=1,
            ppo_n_minibatches=2,
        )
        assert total == 3


class TestConfigRunnerField:
    """``Config.runner`` accepts a ``RunnerConfig`` so the orchestrator can
    plumb status / metadata / metrics / output_model paths through to the
    cookbook RunnerIO (FIR2-1599)."""

    def test_config_exposes_runner_default(self) -> None:
        """Default ``Config.runner`` is an empty RunnerConfig (no outputs)."""
        from training.utils.runner import RunnerConfig

        cfg = async_rl_loop.Config(log_path="gs://logs")
        assert isinstance(cfg.runner, RunnerConfig)
        assert not cfg.runner.enabled

    def test_config_accepts_user_runner_config(self) -> None:
        """The orchestrator can construct Config with a populated runner."""
        from training.utils.runner import RunnerConfig

        runner_cfg = RunnerConfig(
            status_file="gs://r/status.json",
            metadata_file="gs://r/meta.json",
            metrics_file="gs://r/metrics.jsonl",
            output_model_path="gs://r/model.json",
        )
        cfg = async_rl_loop.Config(log_path="gs://logs", runner=runner_cfg)
        assert cfg.runner.status_file == "gs://r/status.json"
        assert cfg.runner.metadata_file == "gs://r/meta.json"
        assert cfg.runner.metrics_file == "gs://r/metrics.jsonl"
        assert cfg.runner.output_model_path == "gs://r/model.json"
        assert cfg.runner.enabled


class TestResourceCallbackSignature:
    """``ResourceCallback`` is a kwargs-only contract so we can add future
    identifiers without breaking existing callers (FIR2-1599)."""

    def test_callable_with_only_documented_kwargs(self) -> None:
        """Today's keyword args are deployment_id / policy_job_id / reference_job_id."""
        seen: list[set[str]] = []

        def cb(**kwargs: object) -> None:
            seen.append(set(kwargs))

        async_rl_loop._emit_resources_ready(
            _FakeInfra(deployment_id="d", policy_job_id="p", reference_job_id="r"),
            cb,
        )
        assert seen == [{"deployment_id", "policy_job_id", "reference_job_id"}]


# ---------------------------------------------------------------------------
# End-to-end ``async_rl_loop.main`` exercise (with fake deps)
#
# Acceptance criteria (FIR2-1599) require unit tests that verify the
# final success/failure status is written and that ``append_metrics`` is
# called during an async run. The helper tests above only cover the
# tiny callback / formula surface; these tests stub every heavy
# dependency and actually drive ``async_rl_loop.main`` so the real
# ``RunnerIO`` writes land on disk.
# ---------------------------------------------------------------------------


class _StubManager:
    """Drop-in for ``TrainerJobManager`` / ``DeploymentManager``."""

    inference_url = "https://inference.unit.test"

    def __init__(self, **_kwargs: object) -> None:
        pass


class _StubPolicy:
    """Minimal ``policy`` shim covering ``forward`` / ``forward_backward_custom`` / ``optim_step``.

    ``forward`` returns one fake ``loss_fn_outputs`` entry per datum so
    the ``old_policy_logprobs`` comprehension in ``train_step`` can index
    safely; ``optim_step`` reports a ``metrics`` dict so the
    minibatch-metrics path runs.
    """

    def forward(self, data, _loss_name: str):
        return SimpleNamespace(
            loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-0.1])} for _ in data]
        )

    def forward_backward_custom(self, *_args: object, **_kwargs: object):
        return SimpleNamespace(metrics={"loss:sum": 0.25, "response_tokens": 4})

    def optim_step(self, *_args: object, **_kwargs: object):
        return SimpleNamespace(metrics={"optim/lr": 1e-5})


class _StubWeightSyncer:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def save_and_hotload(self, *_args: object, **_kwargs: object) -> None:
        pass


class _StubCheckpoints:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def resume(self, **_kwargs: object):
        return None

    def save(self, *_args: object, **_kwargs: object) -> None:
        pass

    def promote_latest(self, *_args: object, **_kwargs: object) -> None:
        pass


class _StubCleanup:
    """``ResourceCleanup`` stand-in supporting the ``with ... as cleanup`` pattern."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _stub_infra(*, with_reference: bool = False) -> SimpleNamespace:
    """Return a fake ``Infra`` covering the attributes ``main`` reads."""
    return SimpleNamespace(
        policy=_StubPolicy(),
        reference=None,
        policy_profile=SimpleNamespace(
            accelerator_type="NVIDIA_H100_80GB",
            accelerator_count=8,
        ),
        policy_job_id="policy-job-1",
        reference_job_id="ref-job-1" if with_reference else None,
        inference_model="accounts/test/models/inf",
        boot_metrics={},
        closeables=[],
        max_seq_len=4096,
        training_shape_id="ts-1",
        ref_training_shape_id=None,
        deployment_id="deployment-1",
        deployment_shape=None,
        deployment_gpu_count=8,
    )


def _patch_async_loop_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    setup_infra_fn,
    run_async_rl_loop_fn,
    token_count: int = 11,
) -> None:
    """Patch every heavy ``async_rl_loop`` dependency to in-memory stubs."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.unit.test")

    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *a, **k: None)
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *a, **k: None)
    monkeypatch.setattr(async_rl_loop, "wandb_log", lambda *a, **k: None)
    monkeypatch.setattr(async_rl_loop, "wandb_finish", lambda: None)
    monkeypatch.setattr(async_rl_loop, "read_api_extra_headers_env", lambda: {})

    monkeypatch.setattr(async_rl_loop, "TrainerJobManager", _StubManager)
    monkeypatch.setattr(async_rl_loop, "DeploymentManager", _StubManager)
    monkeypatch.setattr(async_rl_loop, "ResourceCleanup", _StubCleanup)
    monkeypatch.setattr(async_rl_loop, "WeightSyncer", _StubWeightSyncer)
    monkeypatch.setattr(async_rl_loop, "TrainingCheckpoints", _StubCheckpoints)
    monkeypatch.setattr(async_rl_loop, "load_deployment_tokenizer", lambda _cfg: object())

    monkeypatch.setattr(async_rl_loop, "build_loss_fn", lambda _args: lambda *a, **k: object())
    monkeypatch.setattr(
        async_rl_loop,
        "combine_prompt_groups",
        lambda _groups: ([object()], [], [], [], []),
    )
    monkeypatch.setattr(
        async_rl_loop,
        "compute_minibatch_metrics",
        lambda *_a: {"train/loss": 0.25},
    )
    monkeypatch.setattr(
        async_rl_loop,
        "compute_step_metrics",
        lambda **_k: {
            "rollout/reward": 1.0,
            "rollout/accuracy": 1.0,
            "train/ref_kl": 0.0,
        },
    )
    monkeypatch.setattr(async_rl_loop, "flush_timing", lambda: {})
    # Real token accounting (peer-review P2 fix) -- routed through a
    # deterministic stub so the metadata assertions are stable.
    monkeypatch.setattr(async_rl_loop, "total_target_tokens", lambda _groups: token_count)
    monkeypatch.setattr(async_rl_loop.tinker, "AdamParams", lambda **kwargs: kwargs)

    monkeypatch.setattr(async_rl_loop, "setup_infra", setup_infra_fn)
    monkeypatch.setattr(async_rl_loop, "run_async_rl_loop", run_async_rl_loop_fn)


def _runner_cfg(tmp_path: Path) -> RunnerConfig:
    """``RunnerConfig`` covering all four artifact paths under ``tmp_path``."""
    return RunnerConfig(
        status_file=str(tmp_path / "status.json"),
        metadata_file=str(tmp_path / "metadata.json"),
        metrics_file=str(tmp_path / "metrics.jsonl"),
        output_model_path=str(tmp_path / "output_model.json"),
    )


def _async_config(tmp_path: Path) -> async_rl_loop.Config:
    """Async loop ``Config`` sized to run a single step through the stubs."""
    return async_rl_loop.Config(
        log_path=str(tmp_path / "logs"),
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        kl_beta=0.0,  # disable reference forward
        save_final_checkpoint=False,
        deployment=DeployConfig(tokenizer_model="unit-tokenizer"),
        runner=_runner_cfg(tmp_path),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestAsyncRunArtifactWrites:
    """FIR2-1599 acceptance: ``async_rl_loop.main`` writes the managed runtime
    artifacts (status / metadata / metrics) end-to-end against fake infra."""

    def test_completed_status_written_and_clamped_to_100_percent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Acceptance criterion: final ``COMPLETED`` status is written.

        The loop runs one fake train step but the estimated total is 1,
        so ``percent`` clamps to 100 even when the realised step count
        equals the estimate (and would clamp similarly if rows were
        dropped).
        """
        events: list[str] = []
        rollout_setup_urls: list[str] = []

        def fake_setup_infra(**_kwargs):
            events.append("setup_infra_done")
            return _stub_infra()

        async def fake_run_async_rl_loop(**kwargs):
            events.append("training_started")
            kwargs["train_fns"].train_step(
                kwargs["global_step"],
                [object()],
                {"resolved_rows": 1},
            )
            return 1, {"resolved_rows": 1}

        _patch_async_loop_deps(
            monkeypatch,
            setup_infra_fn=fake_setup_infra,
            run_async_rl_loop_fn=fake_run_async_rl_loop,
        )

        async_rl_loop.main(
            _async_config(tmp_path),
            rollout_fn_factory=lambda setup: (
                rollout_setup_urls.append(setup.inference_base_url) or (lambda _row: None)
            ),
            rows=[{"id": "row-1"}],
        )

        status = _read_json(tmp_path / "status.json")
        assert status["code"] == 0
        assert status["message"] == "done"
        assert status["details"][0]["percent"] == 100
        assert "training_started" in events
        assert rollout_setup_urls == ["https://inference.unit.test/inference"]

    def test_failed_status_written_when_async_loop_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Acceptance criterion: final failure status is written.

        ``RunnerIO``'s ``__exit__`` writes ``RunStatus.FAILED`` with
        ``code=9`` (FAILED_PRECONDITION) and the exception message when
        the async loop raises mid-training.
        """

        async def boom(**_kwargs):
            raise RuntimeError("async loop boom")

        _patch_async_loop_deps(
            monkeypatch,
            setup_infra_fn=lambda **_k: _stub_infra(),
            run_async_rl_loop_fn=boom,
        )

        with pytest.raises(RuntimeError, match="async loop boom"):
            async_rl_loop.main(
                _async_config(tmp_path),
                rollout_fn_factory=lambda _setup: lambda _row: None,
                rows=[{"id": "row-1"}],
            )

        status = _read_json(tmp_path / "status.json")
        assert status["code"] == 9, "FAILED -> google.rpc FAILED_PRECONDITION (9)"
        assert "async loop boom" in status["message"]

    def test_append_metrics_called_during_async_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Acceptance criterion: ``RunnerIO.append_metrics`` is called during
        an async run.

        Verifies the metrics JSONL receives one entry per outer rollout
        step with ``train/step`` / ``rollout/step`` keys, and that the
        cumulative ``metadata.tokens`` value is non-zero (peer-review
        P2 fix: per-step ``tokens=total_target_tokens(prompt_groups)``
        must accumulate into the billing metadata, not remain 0).
        """

        async def fake_run_async_rl_loop(**kwargs):
            kwargs["train_fns"].train_step(
                kwargs["global_step"],
                [object()],
                {"resolved_rows": 1},
            )
            return 1, {"resolved_rows": 1}

        _patch_async_loop_deps(
            monkeypatch,
            setup_infra_fn=lambda **_k: _stub_infra(),
            run_async_rl_loop_fn=fake_run_async_rl_loop,
            token_count=11,
        )

        async_rl_loop.main(
            _async_config(tmp_path),
            rollout_fn_factory=lambda _setup: lambda _row: None,
            rows=[{"id": "row-1"}],
        )

        records = _read_jsonl(tmp_path / "metrics.jsonl")
        assert records, "append_metrics must produce at least one JSONL row"
        first = records[0]
        assert first["step"] == 1
        assert first["rollout/step"] == 1
        assert first["train/step"] == 1

        # Peer-review P2: per-step ``tokens=`` must accumulate into the
        # billing metadata. With one outer step and ``token_count=11``,
        # ``metadata.tokens`` must be at least 11 (and not 0, which is
        # what would happen if ``append_metrics`` were called without
        # the ``tokens=`` kwarg).
        metadata = _read_json(tmp_path / "metadata.json")
        assert metadata["metadata"]["tokens"] == 11

    def test_resources_callback_invoked_before_training(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The resource-ready hook fires right after ``setup_infra`` returns,
        *before* the async loop starts -- so a cold-start fallback that
        reads ``resources.json`` is guaranteed to see the IDs even if
        the trainer crashes on the first step (FIR2-1599)."""
        events: list[str] = []
        callback_payloads: list[dict] = []

        def fake_setup_infra(**_kwargs):
            events.append("setup_infra_done")
            return _stub_infra(with_reference=True)

        async def fake_run_async_rl_loop(**_kwargs):
            events.append("training_started")
            return 0, {"resolved_rows": 0}

        def on_resources_ready(**resources: object) -> None:
            events.append("resource_callback")
            callback_payloads.append(dict(resources))

        _patch_async_loop_deps(
            monkeypatch,
            setup_infra_fn=fake_setup_infra,
            run_async_rl_loop_fn=fake_run_async_rl_loop,
        )

        async_rl_loop.main(
            _async_config(tmp_path),
            rollout_fn_factory=lambda _setup: lambda _row: None,
            rows=[{"id": "row-1"}],
            on_resources_ready=on_resources_ready,
        )

        # Strict ordering: callback fires AFTER setup_infra returns,
        # BEFORE the async training loop runs (plan §Tests).
        assert events.index("setup_infra_done") < events.index("resource_callback")
        assert events.index("resource_callback") < events.index("training_started")
        assert callback_payloads == [
            {
                "deployment_id": "deployment-1",
                "policy_job_id": "policy-job-1",
                "reference_job_id": "ref-job-1",
            }
        ]
