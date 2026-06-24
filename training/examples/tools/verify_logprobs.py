#!/usr/bin/env python3
"""Verify train-inference logprob alignment.

Creates an inference deployment and a policy trainer (plus an optional frozen
reference runtime) from LoRA-capable training shapes, samples completions, runs
training forward passes, and compares per-token logprobs on completion tokens.

Usage:
    export FIREWORKS_API_KEY=...

    # Policy-only (no reference KL):
    python verify.py \\
        --training-shape ts-qwen3-30b-a3b-128k \\
        --base-model accounts/fireworks/models/qwen3-30b-a3b \\
        --tokenizer Qwen/Qwen3-30B-A3B

    # With reference model for KL divergence:
    python verify.py \\
        --training-shape ts-qwen3-30b-a3b-128k \\
        --ref-training-shape ts-qwen3-30b-a3b-128k-ref \\
        --base-model accounts/fireworks/models/qwen3-30b-a3b \\
        --tokenizer Qwen/Qwen3-30B-A3B
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
from contextlib import closing

import torch
import tinker
import transformers

from fireworks.training.sdk import (
    TrainerJobManager,
    DeploymentManager,
)
from training.utils import (
    TrainerConfig,
    DeployConfig,
    ReconnectableClient,
    build_service_client,
    read_api_extra_headers_env,
    load_jsonl_dataset,
    prepare_sampling_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("tinker").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_DATASET = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"


# ---------------------------------------------------------------------------
# Logprob comparison (completion tokens only)
# ---------------------------------------------------------------------------


def compare_logprobs(
    training_lps: list[float],
    inference_lps: list[float],
    prompt_len: int,
    label: str = "",
) -> dict:
    """Compare training vs inference logprobs on completion tokens.

    Both inputs are expected to be in training-aligned format: N-1 entries
    for an N-token sequence, where entry *i* is log P(token[i+1] | token[0:i+1]).
    Only positions >= ``prompt_len - 1`` (the completion region) contribute
    to aggregate metrics.

    Returns a dict of summary statistics including KL divergence estimators.
    """
    response_start = max(0, prompt_len - 1)

    diffs, log_ratios = [], []
    for j in range(response_start, min(len(training_lps), len(inference_lps))):
        d = abs(training_lps[j] - inference_lps[j])
        diffs.append(d)
        log_ratios.append(inference_lps[j] - training_lps[j])

    if not diffs:
        return {"label": label, "count": 0}

    t_diffs = torch.tensor(diffs, dtype=torch.float32)
    t_lr = torch.tensor(log_ratios, dtype=torch.float32)

    metrics = {
        "label": label,
        "count": len(diffs),
        "mean_diff": t_diffs.mean().item(),
        "max_diff": t_diffs.max().item(),
        "k1": t_lr.mean().item(),
        "k2": (0.5 * t_lr**2).mean().item(),
        "k3": (t_lr.exp() - 1 - t_lr).mean().item(),
    }

    logger.info(
        "[%s] completion tokens=%d | mean_diff=%.6f max_diff=%.6f | "
        "k1=%.6f k2=%.6f k3=%.6f",
        label, metrics["count"], metrics["mean_diff"], metrics["max_diff"],
        metrics["k1"], metrics["k2"], metrics["k3"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify train-inference logprob alignment using cookbook utilities",
    )
    p.add_argument(
        "--training-shape", required=True,
        help="Policy training shape ID (e.g. ts-qwen3-30b-a3b-128k)",
    )
    p.add_argument(
        "--ref-training-shape", default=None,
        help=(
            "Forward-only reference training shape ID. "
            "When set, a second trainer is created to provide reference "
            "logprobs for KL comparison."
        ),
    )
    p.add_argument(
        "--shape-account", default=None,
        help="Account that owns the training shapes (for cross-account resolution)",
    )
    p.add_argument(
        "--base-model", required=True,
        help="Base model resource name (e.g. accounts/fireworks/models/qwen3-30b-a3b)",
    )
    p.add_argument(
        "--tokenizer", required=True,
        help="HuggingFace tokenizer name (e.g. Qwen/Qwen3-30B-A3B)",
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="JSONL dataset path or URL")
    p.add_argument("--max-rows", type=int, default=3, help="Number of prompts to verify")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per completion")
    p.add_argument("--deployment-id", default=None, help="Reuse an existing deployment")
    p.add_argument("--log-dir", default="./verify_logs", help="Output directory for logs and results")
    p.add_argument("--cleanup", action="store_true", help="Delete resources on exit")
    return p.parse_args()


def main():
    args = parse_args()
    use_reference = args.ref_training_shape is not None

    os.makedirs(args.log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.log_dir, "verify.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    shape_account = args.shape_account
    if shape_account:
        logger.info("Resolving training shapes under account '%s'", shape_account)
        shape_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
        shape_mgr._account_id = shape_account
    else:
        shape_mgr = rlor_mgr

    # -- Resolve training shapes ----------------------------------------------

    logger.info("Resolving policy training shape: %s", args.training_shape)
    profile = shape_mgr.resolve_training_profile(args.training_shape)
    logger.info("  version:          %s", profile.training_shape_version)
    logger.info("  deployment shape: %s", profile.deployment_shape)
    logger.info("  max_seq_len:      %d", profile.max_supported_context_length)
    logger.info("  accelerator:      %s x%d", profile.accelerator_type, profile.accelerator_count)

    max_seq_len = profile.max_supported_context_length

    if use_reference:
        logger.info("Resolving reference training shape: %s", args.ref_training_shape)
        ref_profile = shape_mgr.resolve_training_profile(args.ref_training_shape)
        logger.info("  version:          %s", ref_profile.training_shape_version)

    # -- Provision via the single SDK seam ------------------------------------
    #
    # build_service_client owns trainer + deployment provisioning; for the
    # reference it spins up a separate frozen runtime (full-param) on a
    # LoRA-capable reference shape. The profile above only supplied
    # deployment_shape and max_seq_len for logging/config.

    dep_id = args.deployment_id or f"verify-{args.base_model.split('/')[-1]}-{int(time.time())}"
    service = build_service_client(
        api_key=api_key,
        base_url=base_url,
        additional_headers=read_api_extra_headers_env(),
        base_model=args.base_model,
        tokenizer_model=args.tokenizer,
        lora_rank=0,
        max_context_length=max_seq_len,
        learning_rate=1e-5,
        trainer=TrainerConfig(
            training_shape_id=args.training_shape,
            reference_training_shape_id=args.ref_training_shape if use_reference else None,
        ),
        deployment=DeployConfig(
            deployment_id=dep_id,
            deployment_shape=profile.deployment_shape,
            tokenizer_model=args.tokenizer,
        ),
        cleanup_trainer_on_close=args.cleanup,
        reference_required=use_reference,
    )

    with closing(service):
        logger.info(
            "\n[1/4] Provisioning deployment + trainers (%s)...",
            "policy + reference" if use_reference else "policy only",
        )
        policy = ReconnectableClient.from_training_client(
            service.create_training_client(args.base_model, lora_rank=0),
            base_model=args.base_model,
            lora_rank=0,
            job_id=service.trainer_job_id,
            service=service,
        )
        logger.info("  Policy trainer:   %s", service.trainer_job_id)

        reference = None
        if use_reference:
            reference = ReconnectableClient.from_training_client(
                service.create_reference_client(args.base_model, lora_rank=0),
                base_model=args.base_model,
                lora_rank=0,
                job_id=service.reference_client_job_id,
                service=service,
                base_only=True,
            )
            logger.info("  Reference trainer: %s", service.reference_trainer_job_id)

        sampler = service.create_sampling_client().deployment_sampler
        sampler.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True,
        )
        logger.info("  Deployment:       %s", service.deployment_id)

        # -- Warmup -----------------------------------------------------------

        logger.info("\n[2/4] Warming up deployment...")
        deploy_mgr.warmup(sampler.model, max_retries=30)

        # -- Load dataset -----------------------------------------------------

        logger.info("\n[3/4] Loading dataset (%d rows)...", args.max_rows)
        dataset = load_jsonl_dataset(args.dataset, args.max_rows)
        if not dataset:
            raise RuntimeError("No data loaded from dataset")

        # -- Verification loop ------------------------------------------------

        logger.info(
            "\n[4/4] Verifying logprobs (%d prompts, greedy, completion-only, ref=%s)...",
            len(dataset), "yes" if use_reference else "no",
        )
        all_metrics: list[dict] = []

        for pidx, row in enumerate(dataset):
            messages = row.get("messages", [])
            input_msgs = prepare_sampling_messages(messages)
            if not input_msgs:
                continue

            preview = input_msgs[-1].get("content", "")[:60]
            logger.info(
                "\n--- Prompt %d/%d: %s... ---", pidx + 1, len(dataset), preview,
            )

            # Sample from deployment
            try:
                sampled = sampler.sample_with_tokens(
                    messages=input_msgs, n=1,
                    max_tokens=args.max_new_tokens, temperature=0.0,
                    logprobs=True, echo=True, prompt_cache_max_len=0,
                    max_seq_len=max_seq_len,
                )
            except Exception as e:
                logger.warning("  Sampling failed: %s", e)
                continue

            if not sampled or not sampled[0].inference_logprobs:
                logger.warning("  No completions or logprobs returned")
                continue

            s = sampled[0]
            ft = s.full_tokens
            logger.info(
                "  Sampled: prompt=%d  completion=%d  total=%d",
                s.prompt_len, s.completion_len, len(ft),
            )

            if len(ft) < 2:
                continue

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(ft[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=ft[1:], dtype="int64", shape=[len(ft) - 1],
                    ),
                },
            )

            # Policy forward
            try:
                fwd = policy.forward([datum], "cross_entropy")
                policy_lps = list(fwd.loss_fn_outputs[0]["logprobs"].data)
            except Exception as e:
                logger.warning("  Policy forward failed: %s", e)
                continue

            m = compare_logprobs(
                training_lps=policy_lps,
                inference_lps=s.inference_logprobs,
                prompt_len=s.prompt_len,
                label=f"prompt-{pidx}/policy-vs-inference",
            )
            if m.get("count", 0) > 0:
                all_metrics.append(m)

            # Reference forward (if enabled)
            if reference is not None:
                try:
                    ref_fwd = reference.forward([datum], "cross_entropy")
                    ref_lps = list(ref_fwd.loss_fn_outputs[0]["logprobs"].data)
                except Exception as e:
                    logger.warning("  Reference forward failed: %s", e)
                    continue

                ref_m = compare_logprobs(
                    training_lps=ref_lps,
                    inference_lps=s.inference_logprobs,
                    prompt_len=s.prompt_len,
                    label=f"prompt-{pidx}/reference-vs-inference",
                )
                if ref_m.get("count", 0) > 0:
                    all_metrics.append(ref_m)

        # -- Summary ----------------------------------------------------------

        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)

        if all_metrics:
            n = len(all_metrics)
            avg_mean = sum(m["mean_diff"] for m in all_metrics) / n
            worst_max = max(m["max_diff"] for m in all_metrics)
            avg_k1 = sum(m["k1"] for m in all_metrics) / n
            avg_k2 = sum(m["k2"] for m in all_metrics) / n
            avg_k3 = sum(m["k3"] for m in all_metrics) / n

            logger.info("  %d comparisons across %d prompts", n, args.max_rows)
            logger.info(
                "  Completion: avg_mean_diff=%.6f  max_diff=%.6f",
                avg_mean, worst_max,
            )
            logger.info(
                "  KL:         k1=%.6f  k2=%.6f  k3=%.6f",
                avg_k1, avg_k2, avg_k3,
            )

            if worst_max < 0.01:
                logger.info("  RESULT: PASS (max diff < 0.01)")
            elif worst_max < 0.1:
                logger.info(
                    "  RESULT: MARGINAL (max diff %.6f, between 0.01 and 0.1)",
                    worst_max,
                )
            else:
                logger.info("  RESULT: FAIL (max diff %.6f > 0.1)", worst_max)

            with open(os.path.join(args.log_dir, "results.json"), "w") as f:
                json.dump({
                    "training_shape": args.training_shape,
                    "ref_training_shape": args.ref_training_shape,
                    "base_model": args.base_model,
                    "per_prompt": all_metrics,
                    "summary": {
                        "prompts": len(dataset),
                        "comparisons": n,
                        "avg_mean_diff": avg_mean,
                        "max_diff": worst_max,
                        "k1": avg_k1, "k2": avg_k2, "k3": avg_k3,
                    },
                }, f, indent=2)
            logger.info("  Results: %s", os.path.join(args.log_dir, "results.json"))
        else:
            logger.error("  No valid comparisons completed!")

        # Trainers are deleted by service.close() (cleanup_trainer_on_close);
        # scale the deployment we created to zero on cleanup.
        if args.cleanup and not args.deployment_id:
            deploy_mgr.scale_to_zero(dep_id)


if __name__ == "__main__":
    main()
