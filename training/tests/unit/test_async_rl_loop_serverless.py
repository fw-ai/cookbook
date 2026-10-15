from __future__ import annotations

import asyncio
import re
import threading
from types import SimpleNamespace

import pytest
from fireworks.training import sdk
from training.examples.archive import vanilla_sampler
from training.recipes.experiment import async_rl_loop_serverless as loop
from training.utils.rl.rollout import RolloutRun, RolloutSample


class _Future:
    def __init__(self, value):
        self._value = value

    async def result_async(self, timeout=None):
        return self.result()

    def result(self):
        return self._value


class _SamplingClient:
    def __init__(self, model: str) -> None:
        self.closed = False
        self.deployment_sampler = SimpleNamespace(
            model=model,
            base_url="https://api.example",
            tokenizer=object(),
        )

    def close(self) -> None:
        self.closed = True


class _TrainingClient:
    run_id = "run-test"

    def __init__(self) -> None:
        self.saved: list[str] = []
        self.saved_states: list[str] = []
        self.loaded_states: list[str] = []
        self.forward_backward_calls: list[tuple[list[sdk.Datum], str, dict | None]] = []
        self.forward_backward_results: list[SimpleNamespace] = []
        self.optim_calls: list[tuple[sdk.AdamParams, str | None, bool]] = []

    def save_weights_for_sampler(self, name: str) -> _Future:
        self.saved.append(name)
        return _Future(SimpleNamespace(path=f"snapshot-{len(self.saved) - 1}"))

    def save_state(self, name: str) -> _Future:
        self.saved_states.append(name)
        return _Future(SimpleNamespace(path=f"state-{len(self.saved_states) - 1}"))

    def load_state_with_optimizer(self, path: str) -> _Future:
        self.loaded_states.append(path)
        return _Future(SimpleNamespace())

    def forward_backward(self, data, loss_fn: str, loss_fn_config=None) -> _Future:
        self.forward_backward_calls.append((data, loss_fn, loss_fn_config))
        # The trainer returns the pre-update policy logprob of every target.
        outputs = [
            {
                "logprobs": sdk.TensorData(
                    data=[-0.15] * len(datum.loss_fn_inputs["target_tokens"].data),
                    dtype="float32",
                    shape=[len(datum.loss_fn_inputs["target_tokens"].data)],
                )
            }
            for datum in data
        ]
        result = SimpleNamespace(metrics={"loss:sum": 1.0}, loss_fn_outputs=outputs)
        self.forward_backward_results.append(result)
        return _Future(result)

    def optim_step(
        self,
        params: sdk.AdamParams,
        *,
        grad_accumulation_normalization: str | None,
        emit_grad_norm_metrics: bool,
    ) -> _Future:
        self.optim_calls.append((params, grad_accumulation_normalization, emit_grad_norm_metrics))
        return _Future(SimpleNamespace(metrics={}))


class _Service:
    training_session_name = "accounts/test/trainingSessions/session-test"

    def __init__(self) -> None:
        self.training_client = _TrainingClient()
        self.sampling_clients: list[_SamplingClient] = []
        self.base_sampling_calls: list[str] = []
        self.lora_creation_calls: list[tuple[str, int, int]] = []
        self.closed = False

    def create_lora_training_client(self, *, base_model: str, rank: int, alpha: int):
        assert base_model == "accounts/fireworks/models/kimi-k3"
        assert rank > 0
        self.lora_creation_calls.append((base_model, rank, alpha))
        return self.training_client

    def create_sampling_client(
        self,
        *,
        model_path: str | None = None,
        base_model: str | None = None,
        tokenizer=None,
    ) -> _SamplingClient:
        del tokenizer
        if base_model is not None:
            self.base_sampling_calls.append(base_model)
        client = _SamplingClient(model_path or base_model or "missing")
        self.sampling_clients.append(client)
        return client

    def close(self) -> None:
        self.closed = True


def _patch_runtime(monkeypatch, service: _Service) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(loop, "_make_service", lambda *_args: service)
    monkeypatch.setattr(loop, "load_tokenizer", lambda *_args: object())
    monkeypatch.setattr(loop, "_router_replay_enabled", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(loop, "setup_wandb", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(loop, "wandb_finish", lambda **_kwargs: None)
    monkeypatch.setattr(loop, "log_metrics", lambda *_args, **_kwargs: None)


def test_serverless_defaults_are_the_320_group_k3_contract() -> None:
    cfg = loop.Config()

    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.max_rows == 320
    assert cfg.completions_per_prompt == 8
    assert cfg.prompt_groups_per_step == 8
    assert cfg.max_head_offpolicy_versions == 0
    assert cfg.max_concurrency_rollout_sample is None
    assert cfg.eps_clip == 0.2
    assert cfg.eps_clip_high is None
    assert cfg.max_seq_len == 524288
    assert cfg.max_completion_tokens == 8192
    assert cfg.step_offset == 0
    assert cfg.resolved_rows_offset == 0
    assert cfg.init_from_checkpoint is None
    assert cfg.adam_beta2 == 0.95
    assert cfg.adam_epsilon == 1e-12
    assert cfg.weight_decay == 0.01
    assert cfg.lora_alpha == 32
    assert cfg.dcp_save_interval == 0
    assert cfg.save_final_checkpoint
    assert cfg.max_rows // cfg.prompt_groups_per_step == 40


def test_invalid_clip_fails_before_serverless_setup(monkeypatch) -> None:
    def unexpected_service(*_args):
        pytest.fail("invalid config must not provision a service")

    monkeypatch.setattr(loop, "_make_service", unexpected_service)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="eps_clip"):
        loop.main(
            loop.Config(eps_clip=-0.1),
            rollout_fn_factory=lambda _setup: None,
            rows=[],
        )


def test_snapshot_names_are_valid_and_stable() -> None:
    assert loop._snapshot_name("async-rl", "step-0") == "async-rl-step-0"

    long_name = loop._snapshot_name(
        "Harbor_DABstep_Serverless_Preflight_1785879757",
        "preflight-step-0",
    )
    assert len(long_name) <= 54
    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", long_name)
    assert long_name == loop._snapshot_name(
        "Harbor_DABstep_Serverless_Preflight_1785879757",
        "preflight-step-0",
    )
    assert "preflight-step-0" in long_name


def test_vanilla_sampler_prefers_injected_sampler() -> None:
    injected = object()
    setup = SimpleNamespace(sampler=injected)

    assert vanilla_sampler.build_deployment_sampler(setup) is injected


def test_rollout_setup_carries_the_completion_and_sequence_limits() -> None:
    sampler = loop.ServerlessSampler(_SamplingClient("snapshot"))

    setup = loop._rollout_setup(
        loop.Config(max_completion_tokens=32768),
        tokenizer=object(),
        sampler=sampler,
        api_key="test-key",
        router_replay_enabled=False,
    )

    assert setup.sample_kwargs["max_tokens"] == 32768
    assert setup.sample_kwargs["max_seq_len"] == 524288
    assert "include_routing_matrix" not in setup.sample_kwargs


def test_moe_rollouts_request_completion_only_router_replay() -> None:
    setup = loop._rollout_setup(
        loop.Config(),
        tokenizer=object(),
        sampler=loop.ServerlessSampler(_SamplingClient("snapshot")),
        api_key="test-key",
        router_replay_enabled=True,
    )

    assert setup.sample_kwargs["include_routing_matrix"] is True
    assert setup.sample_kwargs["echo"] is False


def test_sampler_replacement_drains_old_client() -> None:
    old = _SamplingClient("old")
    new = _SamplingClient("new")
    started = asyncio.Event()
    release = asyncio.Event()

    async def sample(*_args, **_kwargs):
        started.set()
        await release.wait()
        return ["done"]

    old.deployment_sampler.sample_with_prompt_tokens = sample

    async def exercise() -> None:
        sampler = loop.ServerlessSampler(old)
        sampling = asyncio.create_task(sampler.sample_with_prompt_tokens([1]))
        await started.wait()
        replacing = asyncio.create_task(sampler.replace(new))
        await asyncio.sleep(0)
        assert not old.closed
        release.set()
        assert await sampling == ["done"]
        await replacing
        assert old.closed
        assert sampler.model == "new"
        await sampler.aclose()

    asyncio.run(exercise())
    assert new.closed


def test_sampling_preflight_uses_zero_update_snapshot(monkeypatch) -> None:
    service = _Service()
    _patch_runtime(monkeypatch, service)
    seen = {}

    def factory(setup):
        seen["sampler"] = setup.sampler

        async def rollout(_row, *, evaluation=False):
            seen["evaluation"] = evaluation
            return None

        return rollout

    async def evaluate(step, rollout_fn):
        assert step == 0
        await rollout_fn({})
        return {"preflight/reward": 0.5}

    result = loop.run_sampling_preflight(
        loop.Config(max_rows=1),
        rollout_fn_factory=factory,
        evaluation_fn=evaluate,
    )

    assert result == {"preflight/reward": 0.5}
    assert seen["evaluation"] is True
    assert service.lora_creation_calls == [("accounts/fireworks/models/kimi-k3", 64, 32)]
    assert service.base_sampling_calls == []
    assert service.training_client.saved == ["async-rl-preflight-step-0"]
    assert service.training_client.forward_backward_calls == []
    assert service.training_client.optim_calls == []
    assert service.training_client.saved_states == []
    assert [client.deployment_sampler.model for client in service.sampling_clients] == ["snapshot-0"]
    assert seen["sampler"].closed
    assert service.closed


def test_sampling_preflight_finishes_wandb_when_session_creation_fails(
    monkeypatch,
) -> None:
    finished: list[str | None] = []
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(loop, "setup_wandb", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        loop,
        "_make_service",
        lambda *_args: (_ for _ in ()).throw(PermissionError("beta disabled")),
    )
    monkeypatch.setattr(
        loop,
        "wandb_finish",
        lambda *, metrics_file=None: finished.append(metrics_file),
    )

    async def evaluate(_step, _rollout_fn):
        return {}

    with pytest.raises(PermissionError, match="beta disabled"):
        loop.run_sampling_preflight(
            loop.Config(max_rows=1, metrics_file="metrics.jsonl"),
            rollout_fn_factory=lambda _setup: None,
            evaluation_fn=evaluate,
        )

    assert finished == ["metrics.jsonl"]


def test_real_loop_runs_two_groups_and_one_optimizer_step(monkeypatch) -> None:
    service = _Service()
    _patch_runtime(monkeypatch, service)
    eval_steps: list[int] = []
    evaluation_modes: list[bool] = []
    logged_metrics: list[dict] = []
    monkeypatch.setattr(
        loop,
        "log_metrics",
        lambda metrics, **_kwargs: logged_metrics.append(dict(metrics)),
    )

    def factory(setup):
        assert setup.sampler is not None

        async def rollout(
            _row,
            *,
            sample_index: int,
            evaluation: bool = False,
            **_context,
        ) -> RolloutRun:
            evaluation_modes.append(evaluation)
            reward = float(sample_index)
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        raw_logprobs=[0.0, -0.3, -0.4],
                        loss_mask=[0, 1, 1],
                        reward=reward,
                    )
                ],
                run_id=f"run-{sample_index}",
            )

        return rollout

    async def evaluate(step, evaluation_rollout_fn):
        eval_steps.append(step)
        if step == 0:
            raise RuntimeError("transient evaluation failure")
        await evaluation_rollout_fn({"id": "holdout"}, sample_index=0)
        return {"eval/reward": 0.5}

    result = loop.main(
        loop.Config(
            completions_per_prompt=2,
            prompt_groups_per_step=2,
            max_completion_tokens=2,
            max_seq_len=8,
            max_rows=2,
            min_group_size=2,
            max_incomplete_group_retries=0,
            adam_beta2=0.95,
            adam_epsilon=1e-12,
            weight_decay=0.0,
            dcp_save_interval=1,
        ),
        rollout_fn_factory=factory,
        evaluation_fn=evaluate,
        rows=[{"id": "a"}, {"id": "b"}],
    )

    assert result["steps"] == 1
    assert result["accepted_groups"] == 2
    calls = service.training_client.forward_backward_calls
    assert [loss_fn for _, loss_fn, _ in calls] == ["ppo"]
    assert len(calls[0][0]) == 4  # Two complete two-completion groups, one ready request.
    assert service.training_client.forward_backward_results[0].metrics["raw_inference_logprob_coverage"] == 1.0
    assert service.training_client.forward_backward_results[0].metrics["inference_k3"] >= 0.0
    assert "ppo_ratio_mean" in service.training_client.forward_backward_results[0].metrics
    step_metrics = next(metrics for metrics in logged_metrics if "train/step" in metrics)
    assert step_metrics["train/raw_inference_logprob_coverage"] == 1.0
    assert step_metrics["train/inference_k3"] >= 0.0
    assert "train/inference_k1" in step_metrics
    assert "train/ppo_clip_frac" in step_metrics
    assert len(service.training_client.optim_calls) == 1
    adam = service.training_client.optim_calls[0][0]
    assert adam.beta2 == 0.95
    assert adam.eps == 1e-12
    assert adam.weight_decay == 0.0
    assert service.training_client.optim_calls[0][1] == "num_loss_tokens"
    assert service.training_client.optim_calls[0][2] is True
    assert service.training_client.saved == [
        "async-rl-step-0",
        "async-rl-step-1",
    ]
    assert service.training_client.saved_states == ["async-rl-state-step-1"]
    assert result["periodic_training_checkpoints"] == ["state-0"]
    assert result["final_training_checkpoint"] == "state-0"
    assert eval_steps == [0, 1]
    assert evaluation_modes.count(False) == 4
    assert evaluation_modes.count(True) == 1
    evaluation_metrics = [metrics for metrics in logged_metrics if "eval/failed" in metrics]
    assert [metrics["eval/failed"] for metrics in evaluation_metrics] == [1.0, 0.0]
    assert "eval/reward" not in evaluation_metrics[0]
    assert evaluation_metrics[1]["eval/reward"] == 0.5
    assert all(client.closed for client in service.sampling_clients)
    assert service.closed


def test_rollout_setup_model_follows_the_published_snapshot(monkeypatch) -> None:
    service = _Service()
    _patch_runtime(monkeypatch, service)
    captured = {}

    def factory(setup):
        captured["setup"] = setup

        async def rollout(_row, *, sample_index: int, **_context) -> RolloutRun:
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        raw_logprobs=[0.0, -0.3, -0.4],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ],
                run_id=f"run-{sample_index}",
            )

        return rollout

    loop.main(
        loop.Config(
            completions_per_prompt=2,
            prompt_groups_per_step=2,
            max_completion_tokens=2,
            max_seq_len=8,
            max_rows=2,
            min_group_size=2,
            max_incomplete_group_retries=0,
        ),
        rollout_fn_factory=factory,
        rows=[{"id": "a"}, {"id": "b"}],
    )

    assert service.training_client.saved == ["async-rl-step-0", "async-rl-step-1"]
    assert captured["setup"].model == "snapshot-1"


def test_evaluation_overlaps_training_and_finishes_before_snapshot_replacement(
    monkeypatch,
) -> None:
    service = _Service()
    _patch_runtime(monkeypatch, service)
    events: list[str] = []
    evaluation_started = threading.Event()
    release_evaluation = threading.Event()
    logged_metrics: list[dict] = []
    monkeypatch.setattr(
        loop,
        "log_metrics",
        lambda metrics, **_kwargs: logged_metrics.append(dict(metrics)),
    )

    original_forward_backward = service.training_client.forward_backward

    def forward_backward(data, loss_fn, loss_fn_config=None):
        assert evaluation_started.wait(timeout=5)
        events.append("train-forward")
        release_evaluation.set()
        return original_forward_backward(data, loss_fn, loss_fn_config)

    service.training_client.forward_backward = forward_backward
    original_save = service.training_client.save_weights_for_sampler

    def save_weights_for_sampler(name):
        events.append(f"save-{name}")
        return original_save(name)

    service.training_client.save_weights_for_sampler = save_weights_for_sampler

    def factory(_setup):
        async def rollout(_row, *, sample_index: int, **_context) -> RolloutRun:
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        raw_logprobs=[0.0, -0.3, -0.4],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ],
                run_id=f"run-{sample_index}",
            )

        return rollout

    async def evaluate(step, _rollout_fn):
        events.append(f"eval-{step}-start")
        if step == 0:
            evaluation_started.set()
            assert await asyncio.to_thread(release_evaluation.wait, 5)
        events.append(f"eval-{step}-end")
        return None

    result = loop.main(
        loop.Config(
            completions_per_prompt=2,
            prompt_groups_per_step=1,
            max_completion_tokens=2,
            max_seq_len=8,
            max_rows=1,
            min_group_size=2,
            max_incomplete_group_retries=0,
        ),
        rollout_fn_factory=factory,
        evaluation_fn=evaluate,
        rows=[{"id": "a"}],
    )

    assert result["steps"] == 1
    assert events.index("eval-0-start") < events.index("train-forward")
    assert events.index("train-forward") < events.index("eval-0-end")
    assert events.index("eval-0-end") < events.index("save-async-rl-step-1")
    assert events.count("eval-1-start") == 1
    assert any("eval/weight_sync_wait_time" in metrics for metrics in logged_metrics)


def test_train_step_uses_server_grpo_with_rollout_anchor(monkeypatch) -> None:
    service = _Service()
    _patch_runtime(monkeypatch, service)

    def factory(_setup):
        async def rollout(_row, *, sample_index: int, **_context) -> RolloutRun:
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        raw_logprobs=[0.0, -0.3, -0.4],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ],
                run_id=f"run-{sample_index}",
            )

        return rollout

    loop.main(
        loop.Config(
            completions_per_prompt=2,
            prompt_groups_per_step=1,
            max_completion_tokens=2,
            max_seq_len=8,
            max_rows=1,
            min_group_size=2,
            max_incomplete_group_retries=0,
            eps_clip_high=0.28,
        ),
        rollout_fn_factory=factory,
        rows=[{"id": "a"}],
    )

    ((datums, loss_fn, loss_fn_config),) = service.training_client.forward_backward_calls
    assert loss_fn == "ppo"
    assert loss_fn_config == pytest.approx({"clip_low_threshold": 0.8, "clip_high_threshold": 1.28})
    assert len(datums) == 2
    for datum in datums:
        inputs = datum.loss_fn_inputs
        # The clipping anchor is the rollout's sampling logprobs, untouched.
        assert inputs["logprobs"].data == pytest.approx([-0.1, -0.2])
        assert inputs["response_mask"].data == [1, 1]
    assert sorted(datum.loss_fn_inputs["advantages"].data[0] for datum in datums) == (
        pytest.approx([-0.7071, 0.7071], abs=1e-4)
    )
    (optim,) = service.training_client.optim_calls
    assert optim[1] == "num_loss_tokens"
    assert optim[2] is True


def test_resume_restores_optimizer_cursor_and_step(monkeypatch) -> None:
    service = _Service()
    _patch_runtime(monkeypatch, service)
    seen_cursor_indices: list[int] = []
    eval_steps: list[int] = []

    def factory(_setup):
        async def rollout(
            _row,
            *,
            cursor_index: int,
            sample_index: int,
            **_context,
        ) -> RolloutRun:
            seen_cursor_indices.append(cursor_index)
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        raw_logprobs=[0.0, -0.3, -0.4],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ],
                run_id=f"run-{cursor_index}-{sample_index}",
            )

        return rollout

    async def evaluate(step, _rollout_fn):
        eval_steps.append(step)
        return None

    result = loop.main(
        loop.Config(
            completions_per_prompt=2,
            prompt_groups_per_step=2,
            max_completion_tokens=2,
            max_seq_len=8,
            max_rows=4,
            min_group_size=2,
            max_incomplete_group_retries=0,
            init_from_checkpoint="checkpoint-step-1",
            step_offset=1,
            resolved_rows_offset=2,
        ),
        rollout_fn_factory=factory,
        evaluation_fn=evaluate,
        evaluation_interval=3,
        rows=[{"id": str(index)} for index in range(4)],
    )

    assert service.training_client.loaded_states == ["checkpoint-step-1"]
    assert service.training_client.saved == [
        "async-rl-step-1",
        "async-rl-step-2",
    ]
    assert service.training_client.saved_states == ["async-rl-state-step-2"]
    assert sorted(seen_cursor_indices) == [2, 2, 3, 3]
    assert eval_steps == [1, 2]
    assert result["steps"] == 2
    assert result["accepted_groups"] == 4
    assert result["resolved_rows"] == 4
    assert result["final_training_checkpoint"] == "state-0"


def test_producer_failure_after_submission_never_optimizes(monkeypatch):
    service = _Service()
    _patch_runtime(monkeypatch, service)
    training_started = asyncio.Event()
    training_finished = []
    original_forward_backward = service.training_client.forward_backward

    def forward_backward(data, loss_fn, loss_fn_config=None):
        future = original_forward_backward(data, loss_fn, loss_fn_config)

        async def result_async(timeout=None):
            training_started.set()
            await asyncio.sleep(0)
            training_finished.append(True)
            return future.result()

        return SimpleNamespace(result_async=result_async)

    service.training_client.forward_backward = forward_backward

    def factory(setup):
        async def rollout(row, *, sample_index, **context):
            if row["id"] == "failure":
                await training_started.wait()
                raise ValueError("fatal rollout")
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ]
            )

        return rollout

    with pytest.raises(ValueError, match="fatal rollout"):
        loop.main(
            loop.Config(
                completions_per_prompt=2,
                prompt_groups_per_step=2,
                min_group_size=2,
                max_rows=2,
                max_completion_tokens=2,
                max_seq_len=8,
            ),
            rollout_fn_factory=factory,
            rows=[{"id": "success"}, {"id": "failure"}],
        )
    assert len(service.training_client.forward_backward_calls) == 1
    assert training_finished == [True]
    assert service.training_client.optim_calls == []
    assert len(service.training_client.saved) == 1


def test_sigterm_joins_native_requests_before_resource_cleanup(monkeypatch):
    import signal

    from training.utils.termination import TerminatedBySignal

    service = _Service()
    _patch_runtime(monkeypatch, service)
    polling_cancelled = []
    completed = []
    previous_handler = signal.getsignal(signal.SIGTERM)
    original_forward_backward = service.training_client.forward_backward

    def forward_backward(data, loss_fn, loss_fn_config=None):
        future = original_forward_backward(data, loss_fn, loss_fn_config)

        async def result_async(timeout=None):
            release = asyncio.Event()
            event_loop = asyncio.get_running_loop()
            event_loop.call_later(0.01, signal.raise_signal, signal.SIGTERM)
            event_loop.call_later(0.02, release.set)
            try:
                await release.wait()
            except asyncio.CancelledError:
                polling_cancelled.append(True)
                raise
            assert not service.closed
            completed.append(True)
            return future.result()

        return SimpleNamespace(result_async=result_async)

    service.training_client.forward_backward = forward_backward

    def factory(setup):
        async def rollout(row, *, sample_index, **context):
            if row["id"] == "blocked":
                await asyncio.Event().wait()
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ]
            )

        return rollout

    with pytest.raises(TerminatedBySignal, match="SIGTERM"):
        loop.main(
            loop.Config(
                completions_per_prompt=2,
                prompt_groups_per_step=2,
                min_group_size=2,
                max_rows=2,
                max_completion_tokens=2,
                max_seq_len=8,
            ),
            rollout_fn_factory=factory,
            rows=[{"id": "ready"}, {"id": "blocked"}],
        )
    assert completed == [True]
    assert polling_cancelled == []
    assert service.training_client.optim_calls == []
    assert len(service.training_client.saved) == 1
    assert service.closed
    assert signal.getsignal(signal.SIGTERM) is previous_handler


def test_failed_training_interrupts_blocked_rollout_without_optimizer(monkeypatch):
    service = _Service()
    _patch_runtime(monkeypatch, service)
    cancelled_rollouts = []
    original_forward_backward = service.training_client.forward_backward

    def forward_backward(data, loss_fn, loss_fn_config=None):
        original_forward_backward(data, loss_fn, loss_fn_config)

        async def result_async(timeout=None):
            raise RuntimeError("remote forward-backward failed")

        return SimpleNamespace(result_async=result_async)

    service.training_client.forward_backward = forward_backward

    def factory(setup):
        async def rollout(row, *, sample_index, **context):
            if row["id"] == "blocked":
                try:
                    await asyncio.wait_for(asyncio.Event().wait(), timeout=1)
                except asyncio.CancelledError:
                    cancelled_rollouts.append(True)
                    raise
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ]
            )

        return rollout

    with pytest.raises(RuntimeError, match="remote forward-backward failed"):
        loop.main(
            loop.Config(
                completions_per_prompt=2,
                prompt_groups_per_step=2,
                min_group_size=2,
                max_rows=2,
                max_completion_tokens=2,
                max_seq_len=8,
            ),
            rollout_fn_factory=factory,
            rows=[{"id": "ready"}, {"id": "blocked"}],
        )
    assert cancelled_rollouts == [True, True]
    assert len(service.training_client.forward_backward_calls) == 1
    assert service.training_client.optim_calls == []
    assert len(service.training_client.saved) == 1
