"""Real recipe entry points unwind signals around mocked service operations."""

from __future__ import annotations

import importlib
import json
import os
import selectors
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from training.tests.unit.test_async_rl_loop_serverless import _Service, _patch_runtime
from training.utils.rl.rollout import RolloutRun, RolloutSample
from training.utils.termination import TerminatedBySignal


@pytest.mark.parametrize("backward_fails", [False, True])
def test_serverless_rollout_cleanup_can_use_sampler(monkeypatch, backward_fails: bool) -> None:
    module = importlib.import_module("training.recipes.experiment.async_rl_loop_serverless")
    service = _Service()
    _patch_runtime(monkeypatch, service)
    events: list[str] = []
    original_sampling_client = service.create_sampling_client

    async def sample(*args: Any, **kwargs: Any) -> list[str]:
        return ["cleanup sample"]

    def sampling_client(**kwargs: Any) -> Any:
        client = original_sampling_client(**kwargs)
        client.deployment_sampler.sample_with_prompt_tokens = sample
        return client

    service.create_sampling_client = sampling_client
    service.close = lambda: events.append("service")
    original_close = module.ServerlessSampler.aclose

    async def close_sampler(sampler: Any) -> None:
        if not sampler.closed:
            events.append("sampler")
        await original_close(sampler)

    monkeypatch.setattr(module.ServerlessSampler, "aclose", close_sampler)
    if backward_fails:

        def fail_backward(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("backward failed")

        service.training_client.forward_backward = fail_backward

    def factory(setup: Any) -> Any:
        async def rollout(row: dict[str, Any], *, sample_index: int, **kwargs: Any) -> RolloutRun:
            return RolloutRun(
                segments=[RolloutSample(tokens=[10, 11], logprobs=[0.0, -0.1], loss_mask=[0, 1], reward=sample_index)],
                run_id=str(sample_index),
            )

        async def cleanup() -> None:
            events.append("rollout")
            assert await setup.sampler.sample_with_prompt_tokens([]) == ["cleanup sample"]

        rollout.aclose = cleanup
        return rollout

    config = module.Config(completions_per_prompt=2, min_group_size=2, max_completion_tokens=2, max_seq_len=8)
    if backward_fails:
        with pytest.raises(RuntimeError, match="backward failed"):
            module.main(config, rows=[{}], rollout_fn_factory=factory)
    else:
        assert module.main(config, rows=[{}], rollout_fn_factory=factory)["steps"] == 1
    assert events == ["rollout", "sampler", "service"]


def _run_case(recipe: str, phase: str) -> None:
    module = importlib.import_module("training.recipes." + recipe)
    events: list[str] = []

    def operation(name: str) -> None:
        events.append(name)
        if name == phase:
            print("ready", flush=True)
            assert sys.stdin.readline().strip() == "release"
            events.append(name + ":joined")

    service = _Service()
    service.trainer_job_id = "trainer"
    service.deployment_id = "deployment"
    service.max_context_length = 8
    original_create = service.create_lora_training_client

    def create(*args: Any, **kwargs: Any) -> Any:
        operation("create")
        if recipe.startswith("experiment."):
            return original_create(*args, **kwargs)
        return service.training_client

    def close() -> None:
        if phase == "cleanup":
            operation("cleanup")
        events.append("close")

    service.create_lora_training_client = create
    service.create_training_client = create
    service.close = close
    original_snapshot = service.training_client.save_weights_for_sampler
    original_state = service.training_client.save_state

    def snapshot(name: str) -> Any:
        operation("sync")
        return original_snapshot(name)

    def state(name: str) -> Any:
        operation("save")
        return original_state(name)

    service.training_client.save_weights_for_sampler = snapshot
    service.training_client.save_state = state

    class Policy:
        default_timeout = 5

        def save_weights_for_sampler(self, name: str, **kwargs: Any) -> Any:
            return snapshot(name).result()

        def forward_backward(self, *args: Any, **kwargs: Any) -> Any:
            return service.training_client.forward_backward(*args, **kwargs).result()

        def submit_forward_backward(self, *args: Any, **kwargs: Any) -> Any:
            return service.training_client.forward_backward(*args, **kwargs)

        def optim_step(self, *args: Any, **kwargs: Any) -> Any:
            return service.training_client.optim_step(*args, **kwargs).result()

    class Checkpoints:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def resume(self, **kwargs: Any) -> None:
            return None

        def save(self, *args: Any, **kwargs: Any) -> None:
            operation("save")

        def promote_latest(self, *args: Any, **kwargs: Any) -> None:
            operation("promote")

    def factory(setup: Any) -> Any:
        async def rollout(row: dict[str, Any], *, sample_index: int = 0, **context: Any) -> RolloutRun:
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 12],
                        logprobs=[0.0, -0.1, -0.2],
                        loss_mask=[0, 1, 1],
                        reward=float(sample_index),
                    )
                ],
                run_id=str(sample_index),
            )

        return rollout

    with pytest.MonkeyPatch.context() as patch:
        if recipe.startswith("experiment."):
            _patch_runtime(patch, service)
            config = module.Config(
                completions_per_prompt=2,
                prompt_groups_per_step=1,
                max_completion_tokens=2,
                max_seq_len=8,
                min_group_size=2,
            )
        else:
            patch.setenv("FIREWORKS_API_KEY", "test-key")
            for name in ("setup_wandb", "validate_config", "wandb_finish", "log_metrics", "log_run_resources"):
                patch.setattr(module, name, lambda *args, **kwargs: None)
            if recipe == "rl_loop":
                patch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
            patch.setattr(module, "load_deployment_tokenizer", lambda *args: object())
            patch.setattr(module, "build_service_client", lambda **kwargs: service)
            patch.setattr(module, "resolve_router_replay_enabled", lambda **kwargs: False)
            patch.setattr(module, "TrainingCheckpoints", Checkpoints)
            patch.setattr(
                module, "ReconnectableClient", SimpleNamespace(from_training_client=lambda *a, **k: Policy())
            )
            patch.setattr(module, "make_weight_sync", lambda *args, **kwargs: lambda *a, **k: operation("sync"))
            sampler = SimpleNamespace(
                base_url="https://api.example",
                model="model",
                close=lambda: events.append("sampler-close"),
                drain_metrics=lambda: [],
            )
            service.create_deployment_sampler = lambda **kwargs: sampler
            config = module.Config(
                log_path="/tmp/signal-lifecycle",
                completions_per_prompt=2,
                prompt_groups_per_step=1,
                deployment=module.DeployConfig(tokenizer_model="tokenizer"),
                output_model_id="output",
                save_final_checkpoint=True,
            )
        original_handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(TerminatedBySignal, match="SIGTERM"):
            module.main(config, rows=[{"id": 0}], rollout_fn_factory=factory, dynamic_filter_fn=None)
        assert signal.getsignal(signal.SIGTERM) == original_handler
    assert events.count("close") == 1
    assert events[-1] == "close"
    assert events.index(phase + ":joined") < events.index("close")
    # After the interrupted native operation returns, only cleanup may run.
    assert set(events[events.index(phase + ":joined") + 1 :]) <= {"close", "sampler-close"}
    print(json.dumps(events), flush=True)


@pytest.mark.parametrize("recipe", ["rl_loop", "async_rl_loop", "experiment.async_rl_loop_serverless"])
@pytest.mark.parametrize("phase", ["create", "sync", "save", "promote", "cleanup"])
def test_sigterm_joins_lifecycle_operation_before_closing(recipe: str, phase: str) -> None:
    if recipe.startswith("experiment.") and phase == "promote":
        pytest.skip("Serverless recipe has no final promotion operation")
    script = (
        "from training.tests.unit.test_recipe_signal_lifecycle import _run_case; " f"_run_case({recipe!r}, {phase!r})"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=30), "recipe did not reach the requested phase"
        line = process.stdout.readline().strip()
        if line != "ready":
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(line + stdout + stderr)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(input="release\n", timeout=30)
        assert process.returncode == 0, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
