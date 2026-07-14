#!/usr/bin/env python3
# ruff: noqa: E402
"""Merge a LoRA/PEFT adapter into its base model and promote the result.

This is the standalone, cookbook-style version of the "merged base" flow. It
turns an existing HF PEFT adapter into a deployable full ``HF_BASE_MODEL`` by:

  1. provisioning a short-lived service-mode LoRA trainer from the adapter's
     *base* model (``--base-model``) at the adapter's rank (``--lora-rank``),
  2. explicitly loading the adapter weights into the LoRA session with
     ``load_adapter(<adapter gcs uri>)`` — there is no shared base LoRA, every
     adapter is loaded explicitly,
  3. saving a merged-base sampler checkpoint with
     ``save_weights_for_sampler(checkpoint_type="merged_base")``, which folds
     ``W <- W + scaling * (B @ A)`` into the base weights and exports a full HF
     base checkpoint with the adapter metadata stripped,
  4. promoting that checkpoint to a new ``HF_BASE_MODEL`` and waiting for it to
     reach ``READY``.

Why not ``warmStartFrom``? RLOR ``warmStartFrom`` of a PEFT addon is not
effective: the control plane downloads the adapter, but the trainer session
never loads those weights, so the save folds a zero-delta adapter and produces a
base-identical checkpoint. The supported path is ``base_model`` + explicit
``load_adapter`` (this script). The gateway rejects service-mode
``warmStartFrom`` of a LoRA addon for the same reason.

Getting the adapter GCS URI: it is the ``gs://`` directory that contains
``adapter_config.json`` and ``adapter_model*.safetensors``. You can resolve it
from a Fireworks LoRA model resource via the model ``getDownloadEndpoint`` API:

    curl -s -H "Authorization: Bearer $FIREWORKS_API_KEY" \
        "$FIREWORKS_BASE_URL/v1/accounts/<acct>/models/<lora-id>:getDownloadEndpoint" \
        | python -c "import sys,json;print(json.load(sys.stdin))"

Usage:
    export FIREWORKS_API_KEY=...

    python merge_lora_and_promote.py \
        --base-model accounts/fireworks/models/qwen3-8b \
        --adapter-gcs gs://my-bucket/adapters/my-lora \
        --lora-rank 8 \
        --training-shape accounts/<acct>/trainingShapes/<shape>:<version> \
        --output-model-id my-merged-qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import FireworksClient, TrainerJobManager
from training.utils import TrainerConfig
from training.utils.service import build_service_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass(frozen=True)
class MergeConfig:
    base_model: str
    adapter_gcs: str
    lora_rank: int
    training_shape: str
    output_model_id: str
    region: str | None
    snapshot_name: str
    keep_trainer: bool
    trainer_timeout_s: float
    op_timeout_s: float
    checkpoint_poll_timeout_s: float
    promote_poll_timeout_s: float


def parse_args() -> MergeConfig:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base and promote a merged HF base model.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="The adapter's immediate base model resource "
             "(e.g. accounts/fireworks/models/qwen3-8b). NOT the LoRA itself.",
    )
    parser.add_argument(
        "--adapter-gcs",
        required=True,
        help="gs:// directory holding the HF PEFT adapter "
             "(adapter_config.json + adapter_model*.safetensors). Passed to "
             "load_adapter(). See the module docstring for how to resolve it.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        required=True,
        help="Adapter rank (peftDetails.r of the source LoRA).",
    )
    parser.add_argument(
        "--output-model-id",
        required=True,
        help="ID for the promoted merged base model.",
    )
    parser.add_argument(
        "--training-shape",
        default="",
        help="Validated LORA_TRAINER training shape id. Empty = let the backend "
             "auto-select (may fail if no default shape exists for the model).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Optional explicit trainer region. Leave unset so the backend "
             "selects placement.",
    )
    parser.add_argument(
        "--snapshot-name",
        default="merged-base",
        help="Sampler checkpoint name to save before promotion.",
    )
    parser.add_argument(
        "--keep-trainer",
        action="store_true",
        help="Do not delete the temporary trainer job on success (default: delete).",
    )
    parser.add_argument("--trainer-timeout-s", type=float, default=3600)
    parser.add_argument("--op-timeout-s", type=float, default=3000)
    parser.add_argument("--checkpoint-poll-timeout-s", type=float, default=900)
    parser.add_argument(
        "--promote-poll-timeout-s",
        type=float,
        default=1800,
        help="A large-base promote can outlive the gateway HTTP timeout (502) "
             "while it keeps running server-side, so we poll the model resource.",
    )
    args = parser.parse_args()
    return MergeConfig(
        base_model=args.base_model,
        adapter_gcs=args.adapter_gcs,
        lora_rank=args.lora_rank,
        training_shape=args.training_shape,
        output_model_id=args.output_model_id,
        region=args.region,
        snapshot_name=args.snapshot_name,
        keep_trainer=args.keep_trainer,
        trainer_timeout_s=args.trainer_timeout_s,
        op_timeout_s=args.op_timeout_s,
        checkpoint_poll_timeout_s=args.checkpoint_poll_timeout_s,
        promote_poll_timeout_s=args.promote_poll_timeout_s,
    )


def _resolve_merged_checkpoint(
    fw_client: FireworksClient,
    job_id: str,
    snapshot_name: str,
    timeout_s: float,
) -> dict:
    """Poll the control plane until the merged sampler checkpoint is promotable."""
    deadline = time.time() + timeout_s
    last_rows: list[dict] = []
    while time.time() < deadline:
        rows = fw_client.list_checkpoints(job_id)
        last_rows = rows
        matches = [
            r for r in rows
            if r.get("name", "").rsplit("/checkpoints/", 1)[-1].startswith(snapshot_name)
        ]
        promotable = [r for r in matches if r.get("promotable")]
        if promotable:
            chosen = sorted(promotable, key=lambda r: r.get("createTime", ""))[-1]
            logger.info("Merged checkpoint promotable: %s", chosen["name"])
            return chosen
        logger.info(
            "Checkpoint %r not promotable yet (saw %d rows, %d name matches)",
            snapshot_name, len(rows), len(matches),
        )
        time.sleep(15)
    raise TimeoutError(
        f"Merged checkpoint {snapshot_name!r} never became promotable on job "
        f"{job_id!r}. Last rows: {last_rows[-5:]}"
    )


def _model_is_merged_base(model: dict | None) -> bool:
    return bool(
        model
        and model.get("state") == "READY"
        and model.get("kind") == "HF_BASE_MODEL"
        and not model.get("peftDetails")
    )


def _get_model(base_url: str, api_key: str, model_name: str) -> dict | None:
    """Read-only GET of a model resource (stdlib only, no extra deps)."""
    req = urllib.request.Request(
        f"{base_url}/v1/{model_name}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
        logger.info("get model %s failed transiently: %s", model_name, str(e)[:200])
        return None


def _poll_model_until_ready(
    base_url: str,
    api_key: str,
    account_id: str,
    output_model_id: str,
    timeout_s: float,
) -> dict:
    model_name = f"accounts/{account_id}/models/{output_model_id}"
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        model = _get_model(base_url, api_key, model_name)
        last = model
        if _model_is_merged_base(model):
            return model
        logger.info(
            "Waiting for promoted model %s: state=%s kind=%s",
            output_model_id,
            (model or {}).get("state"),
            (model or {}).get("kind"),
        )
        time.sleep(20)
    raise TimeoutError(
        f"Promoted model {output_model_id!r} not READY HF_BASE_MODEL within "
        f"{timeout_s}s. Last: {last}"
    )


def main() -> None:
    cfg = parse_args()
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    fw_client = FireworksClient(api_key=api_key, base_url=base_url)
    trainer_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    logger.info(
        "Merge+promote: base=%s adapter=%s rank=%d -> %s",
        cfg.base_model, cfg.adapter_gcs, cfg.lora_rank, cfg.output_model_id,
    )

    # Provision a short-lived service-mode LoRA trainer from the base model.
    service = build_service_client(
        api_key=api_key,
        base_url=base_url,
        additional_headers=None,
        base_model=cfg.base_model,
        tokenizer_model=None,
        lora_rank=cfg.lora_rank,
        max_context_length=None,
        learning_rate=1e-5,  # unused: we never take an optimizer step
        trainer=TrainerConfig(
            training_shape_id=cfg.training_shape or None,
            region=cfg.region,
            timeout_s=cfg.trainer_timeout_s,
        ),
        cleanup_trainer_on_close=not cfg.keep_trainer,
    )

    try:
        job_id = service.trainer_job_id
        logger.info("Trainer ready: %s", job_id)

        policy = service.create_lora_training_client(cfg.base_model, rank=cfg.lora_rank)

        logger.info("Loading adapter into LoRA session: %s", cfg.adapter_gcs)
        load_resp = policy.load_adapter(cfg.adapter_gcs).result(timeout=cfg.op_timeout_s)
        logger.info("load_adapter result: %s", load_resp)

        logger.info("Saving merged-base checkpoint %r", cfg.snapshot_name)
        save = policy.save_weights_for_sampler_ext(
            cfg.snapshot_name, checkpoint_type="merged_base",
        )
        logger.info("Saved: path=%s snapshot_name=%s", save.path, save.snapshot_name)

        checkpoint = _resolve_merged_checkpoint(
            fw_client, job_id, save.snapshot_name, cfg.checkpoint_poll_timeout_s,
        )

        logger.info("Promoting %s -> %s", checkpoint["name"], cfg.output_model_id)
        try:
            trainer_mgr.promote_checkpoint(
                name=checkpoint["name"],
                output_model_id=cfg.output_model_id,
                base_model=cfg.base_model,
            )
        except Exception as e:
            logger.warning(
                "Promote HTTP call failed (%s); promotion may still be running "
                "server-side. Polling the model resource for READY.",
                str(e)[:300],
            )

        model = _poll_model_until_ready(
            base_url, api_key, fw_client.account_id, cfg.output_model_id,
            cfg.promote_poll_timeout_s,
        )
        logger.info(
            "Promoted merged base: %s state=%s kind=%s",
            model.get("name"), model.get("state"), model.get("kind"),
        )
    finally:
        # cleanup_trainer_on_close handles trainer teardown unless --keep-trainer.
        service.close()


if __name__ == "__main__":
    main()
