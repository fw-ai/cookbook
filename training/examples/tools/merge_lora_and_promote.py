#!/usr/bin/env python3
# ruff: noqa: E402
"""Merge a LoRA/PEFT adapter into its base model and promote the result.

This is the standalone, cookbook-style version of the "merged base" flow. It
turns an existing HF PEFT adapter into a deployable full ``HF_BASE_MODEL`` by:

  1. provisioning a short-lived service-mode LoRA trainer from the adapter's
     *base* model at the adapter's rank (given as ``--base-model`` /
     ``--lora-rank``, or read from ``--adapter-model``),
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

Usage — from a promoted Fireworks LoRA model (recommended). ``--adapter-model``
resolves the base model, adapter rank, and adapter ``gs://`` directory from the
model resource, so none of them have to be looked up by hand:

    export FIREWORKS_API_KEY=...

    python merge_lora_and_promote.py \
        --adapter-model accounts/<acct>/models/<lora-id> \
        --output-model-id my-merged-qwen3-8b

Usage — from a raw adapter directory. ``--adapter-gcs`` is the ``gs://``
directory holding ``adapter_config.json`` and ``adapter_model*.safetensors``,
and ``--base-model`` is the adapter's own base (not the adapter):

    python merge_lora_and_promote.py \
        --base-model accounts/fireworks/models/qwen3-8b \
        --adapter-gcs gs://my-bucket/adapters/my-lora \
        --lora-rank 8 \
        --output-model-id my-merged-qwen3-8b

Placement: leave ``--region`` and ``--training-shape`` unset unless you have a
reason to pin them. The backend then selects a validated shape and a region that
actually provisions that shape's accelerator. Pinning a region whose clusters do
not carry the shape's accelerator fails during trainer provisioning, and that
placement rejection can surface as a bare "Internal error" — see
``--training-shape``/``--region`` help and the skill's ``error-reference.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
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
class AdapterSource:
    """Concrete merge inputs, either given explicitly or resolved from a model."""

    base_model: str
    adapter_gcs: str
    lora_rank: int


@dataclass(frozen=True)
class MergeConfig:
    base_model: str | None
    adapter_gcs: str | None
    lora_rank: int | None
    adapter_model: str | None
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
        "--adapter-model",
        default=None,
        help="Promoted Fireworks LoRA model resource "
             "(accounts/<acct>/models/<lora-id>) to merge. Its base model, rank, "
             "and adapter gs:// directory are read from the model resource, so "
             "--base-model/--lora-rank/--adapter-gcs are not needed. Any of "
             "those passed explicitly wins over the resolved value.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="The adapter's immediate base model resource "
             "(e.g. accounts/fireworks/models/qwen3-8b). NOT the LoRA itself. "
             "Required without --adapter-model.",
    )
    parser.add_argument(
        "--adapter-gcs",
        default=None,
        help="gs:// directory holding the HF PEFT adapter "
             "(adapter_config.json + adapter_model*.safetensors). Passed to "
             "load_adapter(). Required without --adapter-model.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Adapter rank (peftDetails.r of the source LoRA). Required without "
             "--adapter-model.",
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
             "auto-select a validated shape for the base model, which is the "
             "recommended default. Pin one only to override that choice; list "
             "candidates with GET /v1/accounts/fireworks/trainingShapes and use "
             "a shape whose base model family matches --base-model.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Optional explicit trainer region. Leave unset so the backend "
             "selects a region that provisions the shape's accelerator; a "
             "pinned region without that accelerator fails provisioning.",
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
    if not args.adapter_model:
        missing = [
            flag
            for flag, value in (
                ("--base-model", args.base_model),
                ("--adapter-gcs", args.adapter_gcs),
                ("--lora-rank", args.lora_rank),
            )
            if value is None
        ]
        if missing:
            parser.error(
                f"{', '.join(missing)} required without --adapter-model "
                "(pass --adapter-model <lora model resource> to resolve them "
                "from the model resource instead)"
            )
    return MergeConfig(
        base_model=args.base_model,
        adapter_gcs=args.adapter_gcs,
        lora_rank=args.lora_rank,
        adapter_model=args.adapter_model,
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


def _get_json(base_url: str, api_key: str, path: str) -> dict:
    """GET a control-plane resource, raising on failure."""
    req = urllib.request.Request(
        f"{base_url}/v1/{path}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _gcs_dir_from_signed_url(signed_url: str) -> str:
    """Recover the ``gs://`` directory of a file from its signed download URL."""
    parsed = urllib.parse.urlsplit(signed_url)
    host, object_path = parsed.netloc, parsed.path.lstrip("/")
    if host.endswith(".storage.googleapis.com"):
        bucket = host.removesuffix(".storage.googleapis.com")
    elif "/" in object_path:
        bucket, object_path = object_path.split("/", 1)
    else:
        raise ValueError(f"Cannot parse a GCS bucket out of signed URL host {host!r}")
    directory = urllib.parse.unquote(object_path).rsplit("/", 1)[0]
    if not bucket or not directory:
        raise ValueError(f"Signed URL {signed_url.split('?')[0]!r} has no object directory")
    return f"gs://{bucket}/{directory}"


def _resolve_adapter_source(base_url: str, api_key: str, cfg: MergeConfig) -> AdapterSource:
    """Fill in base model, rank, and adapter directory for the merge.

    Without ``--adapter-model`` the explicit flags are already complete (enforced
    in ``parse_args``). With it, each unset value is read from the LoRA model
    resource: ``peftDetails`` carries the base model and rank, and the adapter's
    ``gs://`` directory is recovered from the ``getDownloadEndpoint`` signed URL
    for ``adapter_config.json`` — neither the API nor firectl exposes that
    directory directly.
    """
    if not cfg.adapter_model:
        if not (cfg.base_model and cfg.adapter_gcs and cfg.lora_rank):
            raise ValueError(
                "base_model, adapter_gcs, and lora_rank are all required without "
                "adapter_model"
            )
        return AdapterSource(cfg.base_model, cfg.adapter_gcs, cfg.lora_rank)

    model = _get_json(base_url, api_key, cfg.adapter_model)
    peft = model.get("peftDetails") or {}
    if not peft:
        raise ValueError(
            f"{cfg.adapter_model} is not a LoRA/PEFT model (no peftDetails); "
            f"kind={model.get('kind')!r}. Pass --base-model/--adapter-gcs/"
            "--lora-rank explicitly for a raw adapter directory."
        )

    adapter_gcs = cfg.adapter_gcs
    if adapter_gcs is None:
        urls = _get_json(
            base_url, api_key, f"{cfg.adapter_model}:getDownloadEndpoint"
        ).get("filenameToSignedUrls") or {}
        config_urls = [url for name, url in urls.items() if name.endswith("adapter_config.json")]
        if not config_urls:
            raise ValueError(
                f"{cfg.adapter_model} download endpoint lists no adapter_config.json "
                f"(files: {sorted(urls)}). Pass --adapter-gcs explicitly."
            )
        adapter_gcs = _gcs_dir_from_signed_url(config_urls[0])

    source = AdapterSource(
        base_model=cfg.base_model or peft["baseModel"],
        adapter_gcs=adapter_gcs,
        lora_rank=cfg.lora_rank or int(peft["r"]),
    )
    logger.info(
        "Resolved %s: base=%s rank=%d adapter=%s",
        cfg.adapter_model, source.base_model, source.lora_rank, source.adapter_gcs,
    )
    return source


def _provisioning_failure_hint(cfg: MergeConfig, source: AdapterSource, job_id: str | None) -> str:
    """Guidance for a trainer that dies during provisioning, often as 'Internal error'."""
    return (
        "Trainer provisioning failed before the adapter was loaded"
        + (f" (trainer job {job_id})" if job_id else "")
        + ". A bare \"Internal error\" here is usually an unsupported "
        "(accelerator, region) pair: placement rejects the request because the "
        "selected training shape's accelerator is not provisioned in the "
        "requested region, and that rejection is reported as an internal error. "
        f"Retry with --region unset (currently {cfg.region or 'unset'}) and "
        f"--training-shape unset (currently {cfg.training_shape or 'unset'}) so "
        f"the backend picks a validated shape for {source.base_model} and a "
        "region that carries its accelerator. If it still fails, escalate with "
        "the trainer job id, base model, shape, and region."
    )


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

    source = _resolve_adapter_source(base_url, api_key, cfg)
    logger.info(
        "Merge+promote: base=%s adapter=%s rank=%d -> %s",
        source.base_model, source.adapter_gcs, source.lora_rank, cfg.output_model_id,
    )

    # Provision a short-lived service-mode LoRA trainer from the base model.
    service = build_service_client(
        api_key=api_key,
        base_url=base_url,
        additional_headers=None,
        base_model=source.base_model,
        tokenizer_model=None,
        lora_rank=source.lora_rank,
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
        # The service client provisions the trainer lazily on the first
        # create_*_client call, so the job id only resolves after this line.
        try:
            policy = service.create_lora_training_client(
                source.base_model, rank=source.lora_rank
            )
        except Exception as e:
            raise RuntimeError(
                _provisioning_failure_hint(
                    cfg, source, service.managed_trainer_job_id
                )
            ) from e
        job_id = service.trainer_job_id
        logger.info("Trainer ready: %s", job_id)

        logger.info("Loading adapter into LoRA session: %s", source.adapter_gcs)
        load_resp = policy.load_adapter(source.adapter_gcs).result(timeout=cfg.op_timeout_s)
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
                base_model=source.base_model,
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
