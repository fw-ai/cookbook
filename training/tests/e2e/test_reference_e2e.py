"""Frozen-reference KL update and policy publication, LoRA and full parameter.

Set FIREWORKS_E2E_ACCOUNT explicitly; it must match the authenticated account.
FIREWORKS_E2E_REGION optionally selects placement; unset leaves it to the backend.
FIREWORKS_E2E_REFERENCE_REGION independently places the frozen inference deployment.
Uses the shared model/shape fixtures and deletes the resources it creates.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from contextlib import ExitStack
from pathlib import Path

import pytest
from fireworks.training import sdk
from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentManager
from fireworks.training.sdk.trainer import TrainerJobManager

from training.utils import DeployConfig, ReconnectableClient, TrainerConfig, build_service_client
from training.utils.rl.algorithm import grpo
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.reference import LoraReference, SamplerReference, with_reference_logprobs
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group, sampled_completion_to_rollout_run
from training.utils.service import make_weight_sync
from training.utils.tokenizers import load_deployment_tokenizer

FROZEN_REFERENCE_READY_TIMEOUT_S = 1200
"""Allow a cold inference deployment to start within the overall test budget."""


@pytest.mark.e2e
@pytest.mark.timeout(2400)
def test_reference_kl_update_and_publish(
    sdk_managers: tuple[TrainerJobManager, DeploymentManager],
    e2e_model: str,
    e2e_tokenizer_model: str,
    e2e_training_shape: str,
    e2e_deployment_shape: str,
    port_lora_rank: int,
    tmp_path: Path,
) -> None:
    trainer_mgr, deploy_mgr = sdk_managers
    account = os.environ.get("FIREWORKS_E2E_ACCOUNT")
    if not account:
        pytest.skip("Set FIREWORKS_E2E_ACCOUNT to authorize resource creation")
    assert trainer_mgr.account_id == deploy_mgr.account_id == account
    run_id = f"ref-e2e-{uuid.uuid4().hex[:10]}"
    deployment = DeployConfig(
        deployment_id=run_id,
        deployment_shape=e2e_deployment_shape,
        tokenizer_model=e2e_tokenizer_model,
    )
    tokenizer = load_deployment_tokenizer(deployment)
    with ExitStack() as stack:
        service = build_service_client(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"),
            additional_headers=None,
            base_model=e2e_model,
            tokenizer_model=e2e_tokenizer_model,
            max_lora_rank=port_lora_rank,
            max_context_length=2048,
            learning_rate=1e-4,
            trainer=TrainerConfig(
                training_shape_id=e2e_training_shape,
                region=os.environ.get("FIREWORKS_E2E_REGION"),
            ),
            deployment=deployment,
            hotload_timeout_s=600,
            cleanup_trainer_on_close=True,
            cleanup_deployment_on_close="delete",
        )
        stack.callback(service.close)
        training_client = service.create_training_client(e2e_model, lora_rank=port_lora_rank)
        job = trainer_mgr.get(service.trainer_job_id)
        assert job["name"].startswith(f"accounts/{account}/")
        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=e2e_model,
            lora_rank=port_lora_rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        publish = make_weight_sync(policy, service, deployment)
        publish("reference-test-initial", checkpoint_type="base")
        rollout = service.create_deployment_sampler(tokenizer=tokenizer)
        stack.callback(rollout.close)
        assert deploy_mgr.get(run_id).name.startswith(f"accounts/{account}/")

        if port_lora_rank:
            reference = LoraReference(service.create_reference_client(policy_client=training_client))
        else:
            # A frozen reference needs no trainer colocation. Provision it
            # independently so inference and training can use their own regions.
            reference_config = DeploymentConfig(
                deployment_id=f"{run_id}-base",
                base_model=e2e_model,
                deployment_shape=e2e_deployment_shape,
                region=os.environ.get("FIREWORKS_E2E_REFERENCE_REGION"),
                min_replica_count=1,
                enable_hot_load=False,
                hot_load_bucket_type=None,
            )
            stack.callback(deploy_mgr.delete, reference_config.deployment_id)
            deploy_mgr.create_or_get(reference_config)
            frozen_info = deploy_mgr.wait_for_ready(
                reference_config.deployment_id, timeout_s=FROZEN_REFERENCE_READY_TIMEOUT_S
            )
            assert frozen_info.name.startswith(f"accounts/{account}/")
            assert not frozen_info.hot_load_trainer_job
            frozen = service.create_deployment_sampler_for_model(frozen_info.name, tokenizer=tokenizer)
            sampler_client = service.create_sampling_client(deployment_sampler=frozen)
            stack.callback(sampler_client.close)
            reference = SamplerReference(sampler_client)
        assert service.reference_trainer_job_id is None

        sampled = asyncio.run(
            rollout.sample_with_prompt_tokens(
                tokenizer.encode("The sum of 2 and 3 is"),
                n=2,
                max_tokens=8,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                logprobs=True,
                echo=False,
            )
        )
        assert len(sampled) == 2
        # Synthetic rewards guarantee one nonzero-advantage group; this tests
        # training plumbing and frozen state, rather than task quality.
        group = rollout_to_prompt_group(
            Rollout(
                runs=[
                    sampled_completion_to_rollout_run(completion, reward=float(i))
                    for i, completion in enumerate(sampled)
                ]
            )
        )
        assert group is not None
        batch = with_reference_logprobs(combine_prompt_groups([group]), reference)
        before = policy.forward(batch.data, "cross_entropy")
        result = policy.forward_backward_custom(batch.data, grpo.make_loss_fn(batch, grpo.GRPOConfig(kl_beta=0.01)))
        assert math.isfinite(result.metrics["mean_kl_penalty"])
        optim = policy.optim_step(
            sdk.AdamParams(learning_rate=1e-4),
            grad_accumulation_normalization=grpo.NORMALIZATION,
            emit_grad_norm_metrics=True,
        )
        after = policy.forward(batch.data, "cross_entropy")
        policy_delta = max(
            abs(a - b)
            for old, new in zip(before.loss_fn_outputs, after.loss_fn_outputs, strict=True)
            for a, b in zip(old["logprobs"].data, new["logprobs"].data, strict=True)
        )
        assert policy_delta > 1e-5, "Policy update must change its token scores"
        publish("reference-test-updated")
        frozen_batch = with_reference_logprobs(batch, reference)
        ref_delta = max(
            abs(a - b)
            for old, new in zip(batch.ref_logprobs, frozen_batch.ref_logprobs, strict=True)
            for a, b in zip(old, new, strict=True)
        )
        assert ref_delta < 1e-4, "Reference changed after policy update/publication"
        smoke = asyncio.run(
            rollout.sample_with_prompt_tokens(
                tokenizer.encode("The sum of 2 and 3 is"),
                n=1,
                max_tokens=2,
                temperature=1.0,
                logprobs=True,
            )
        )
        assert smoke[0].completion_len > 0
        evidence = {
            "account": account,
            "trainer": job["name"],
            "deployment": run_id,
            "lora_rank": port_lora_rank,
            "policy_delta": policy_delta,
            "reference_delta": ref_delta,
            "loss": result.metrics,
            "optimizer": optim.metrics,
        }
        (tmp_path / "reference-evidence.json").write_text(json.dumps(evidence, indent=2))
        print("REFERENCE_EVIDENCE", json.dumps(evidence))
