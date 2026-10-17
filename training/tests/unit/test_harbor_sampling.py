"""CPU transport checks for sampling ownership across Harbor harnesses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from fireworks.training.sdk import TITOParsedAssistant, TITOSidecar
from fireworks.training.sdk.sampling import SampledCompletion, SampledRequestResult
from training.examples.harbor_rl.harnesses.mimoagent import rollout as mimoagent_rollout
from training.examples.harbor_rl.harnesses.mini_swe import rollout as mini_swe_rollout
from training.examples.harbor_rl.harnesses.pi import rollout as pi_rollout
from training.examples.harbor_rl.harnesses.opencode import rollout
from training.examples.harbor_rl.tito import sidecar as sidecar_runtime
from training.examples.harbor_rl.tito import rollout as tito_rollout
from training.examples.harbor_rl.tito.trial import HarborTrialOutcome
from training.utils.rl.rollout import RolloutRun, RolloutSample


class _SamplingRenderer:
    """Synthetic tokens keep sampling-policy tests independent of model parsing."""

    renderer_id = "sampling-contract-test"

    def render_conversation_tokens(self, request: Any) -> list[int]:
        return [1, 2, 3]

    def parse_assistant(
        self, request: Any, completion_ids: Any, completion_text: str, finish_reason: str
    ) -> TITOParsedAssistant:
        return TITOParsedAssistant(message={"role": "assistant", "content": "ok"})

    def render_contract_id(self, request: Any) -> str:
        return self.renderer_id

    def stop_sequences(self, request: Any) -> None:
        return None


class _SamplingRecorder:
    additional_headers: dict[str, str] = {}

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def sample_with_prompt_tokens_result(self, prompt_ids: list[int], **kwargs: Any) -> SampledRequestResult:
        self.calls.append(kwargs)
        return SampledRequestResult(
            completions=[
                SampledCompletion(
                    text="ok",
                    full_tokens=[*prompt_ids, 7],
                    prompt_len=len(prompt_ids),
                    completion_len=1,
                    sampling_logprobs=[-0.1],
                    finish_reason="stop",
                )
            ],
            server_metrics=None,
            logical_request_id=kwargs["logical_request_id"],
            attempts=1,
            wall_seconds=0.01,
        )


@pytest.mark.parametrize(
    "harness",
    [mimoagent_rollout, mini_swe_rollout, pi_rollout, rollout],
    ids=["mimo", "mini-swe", "pi", "opencode"],
)
@pytest.mark.parametrize(
    "sampling",
    [
        {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        {"temperature": 0.7, "top_p": 0.8, "top_k": 5},
    ],
    ids=["training-defaults", "custom-sampling"],
)
def test_harbor_sampling_overrides_agent_requests_at_sampler_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    harness: Any,
    sampling: dict[str, float | int],
) -> None:
    setup = SimpleNamespace(
        sample_kwargs={"max_tokens": 128, "max_seq_len": 4096, **sampling},
        sampler=SimpleNamespace(),
        inference_base_url="http://unused.invalid",
        api_key="synthetic-test-key",
        model="policy",
    )
    sampler = _SamplingRecorder()

    async def run_trial(**kwargs: Any) -> HarborTrialOutcome:
        spec = json.loads(kwargs["sidecar_launch_spec"])
        # Consume the factory's actual serialized launch spec as the sandbox does.
        sidecar = TITOSidecar.from_deployment_sampler(
            sampler,  # type: ignore[arg-type]
            renderer=_SamplingRenderer(),
            max_context_tokens=spec["max_context_tokens"],
            max_output_tokens=spec["max_output_tokens"],
            sampling_defaults=spec["sampling_defaults"],
            call_classifier=sidecar_runtime.build_call_classifier(spec["call_classifier"]),
        )
        await sidecar.start()
        try:
            endpoint = await sidecar.create_trajectory_async()
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    endpoint.openai_base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + endpoint.api_key},
                    json={
                        "model": "policy",
                        "messages": [{"role": "user", "content": "fix the bug"}],
                        "tools": [{"type": "function", "function": {"name": "bash"}}],
                        "temperature": 1.3,
                        "top_p": 0.22,
                        "top_k": 77,
                        "_tito": {"call_kind": "policy", "classifier_source": "test"},
                    },
                ) as response:
                    payload = await response.json()
                    assert response.status == 200, payload
        finally:
            await sidecar.close()
        return HarborTrialOutcome(
            task_name="sampling-contract",
            trial_name="test",
            trial_path=tmp_path,
            reward=1.0,
            rewards={"reward": 1.0},
            exception_type=None,
            exception_message=None,
            rollout=RolloutRun(
                metadata={},
                segments=[
                    RolloutSample(
                        tokens=[1, 7],
                        logprobs=[0.0, -0.1],
                        loss_mask=[0, 1],
                        reward=1.0,
                    )
                ],
            ),
        )

    bundle = sidecar_runtime.TITOSidecarBundle(tmp_path / "bundle.zip", "a" * 64)
    monkeypatch.setattr(tito_rollout, "build_sidecar_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(tito_rollout, "task_config_from_row", lambda _row: {})
    monkeypatch.setattr(tito_rollout, "task_initial_instruction", lambda _config: "fix the bug")
    monkeypatch.setattr(tito_rollout, "run_harbor_trial", run_trial)
    harbor = tito_rollout.HarborSettings(
        tito=sidecar_runtime.TITOSidecarSettings(renderer_name="test", bundle_root=tmp_path),
        trials_dir=tmp_path,
        rollout_retries=0,
    )
    runner = harness.make_rollout_fn(setup, harbor=harbor)
    assert asyncio.run(runner({"task_name": "sampling-contract"})) is not None
    assert len(sampler.calls) == 1
    for name, value in sampling.items():
        assert sampler.calls[0][name] == value
    assert sampler.calls[0]["logprobs"] is True
    assert "max_seq_len" not in sampler.calls[0]
