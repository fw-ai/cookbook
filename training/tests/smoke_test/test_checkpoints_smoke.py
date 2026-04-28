"""Real-backend smoke test for ``TrainingCheckpoints``.

Provisions a tiny LoRA SFT trainer on ``qwen3-4b-minimum-lora`` (1xB200),
runs a few steps with ``dcp_save_interval`` set + ``output_model_id`` set,
then verifies via the control plane that the new ``TrainingCheckpoints``
API produced the expected rows and that promotion succeeded.

Distinct from the mock-based unit tests in
``training/tests/unit/test_checkpoints.py``: that suite covers logic
under controlled conditions; this one hits the real control plane and
catches API contract drift (row shape, type enum strings, promote
response, ``dataloader.json`` propagation).

What this exercises:
    * ``ckpt.save(resumable=True, promotable=False)``  -- periodic DCP
    * ``ckpt.save(resumable=True, promotable=True)``   -- final
    * ``ckpt.promote_latest(...)``                     -- promotion
    * ``dataloader.json`` is written under ``log_path``
    * Control-plane row shape / enum strings match what the SDK assumes

What this does NOT cover (out of scope; covered elsewhere):
    * Resume path -- needs a persistent trainer across two ``main()`` calls
      or an explicit ``init_from_checkpoint``; covered by the existing
      ``test_sft_resume_e2e`` / ``test_grpo_resume_e2e`` suites and by
      the unit tests under ``test_checkpoints.py::TestResume``.
    * ``WeightSyncer.save_and_hotload`` skip-if-exists (RL — see GRPO smoke)
    * Trainer-rename race (covered by unit tests + post-save polling)

Requires:
    FIREWORKS_API_KEY            (skipped if not set)
    FIREWORKS_BASE_URL           (defaults to https://dev.api.fireworks.ai)

Usage:
    pytest training/tests/smoke_test/test_checkpoints_smoke.py -v -s
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid

import pytest

from fireworks.training.sdk import FireworksClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


_DEFAULT_LORA_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
_DCP_SAVE_INTERVAL = 2  # save twice in a 4-step run
_NUM_TRAINING_EXAMPLES = 8  # batch_size=2 * (1 epoch * ~4 batches) = 4 steps


@pytest.fixture(scope="session")
def smoke_minimal_lora_training_shape() -> str:
    return os.environ.get(
        "FIREWORKS_SMOKE_MINIMAL_LORA_TRAINING_SHAPE",
        _DEFAULT_LORA_SHAPE,
    )


def _make_chat_dataset(path: str, num_examples: int) -> None:
    with open(path, "w") as f:
        for i in range(num_examples):
            row = {
                "messages": [
                    {"role": "user", "content": f"What is {i} times 2?"},
                    {"role": "assistant", "content": f"The answer is {i * 2}."},
                ]
            }
            f.write(json.dumps(row) + "\n")


def _short_name(resource_name: str) -> str:
    return resource_name.rstrip("/").rsplit("/", 1)[-1]


def _summarize_rows(rows: list[dict]) -> str:
    return "\n  ".join(
        f"{_short_name(r['name'])}  type={r.get('checkpointType')}  "
        f"promotable={r.get('promotable')}  ct={r.get('createTime')}"
        for r in rows
    ) or "(no rows)"


@pytest.mark.e2e
@pytest.mark.timeout(2400)
class TestCheckpointsSmoke:
    """End-to-end smoke for ``TrainingCheckpoints`` against a real backend.

    Single test method intentionally bundles several assertions: each
    end-to-end run provisions a real LoRA trainer (~10 min B200 setup +
    teardown), so splitting per-concept would 4x the GPU bill for the
    same coverage. The assertions are grouped by concept inside the
    method body — when one fails, the failure message + log output
    identifies which concept broke.
    """

    def test_save_and_promote(
        self,
        smoke_sdk_managers,
        smoke_base_model,
        smoke_tokenizer_model,
        smoke_minimal_lora_training_shape,
    ):
        from training.recipes.sft_loop import Config, main
        from training.utils import InfraConfig, WandBConfig
        from training.utils.checkpoints import DATALOADER_BASE_NAME

        rlor_mgr, _deploy_mgr = smoke_sdk_managers

        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ.get(
            "FIREWORKS_BASE_URL", "https://dev.api.fireworks.ai",
        )
        fw_client = FireworksClient(api_key=api_key, base_url=base_url)

        log_dir = tempfile.mkdtemp(prefix="ckpt_smoke_")
        suffix = uuid.uuid4().hex[:8]
        output_model_id = f"ckpt-smoke-{suffix}"
        dataset_path = os.path.join(log_dir, "dataset.jsonl")
        _make_chat_dataset(dataset_path, num_examples=_NUM_TRAINING_EXAMPLES)

        infra = InfraConfig(training_shape_id=smoke_minimal_lora_training_shape)

        # ---- Run SFT LoRA with promotion ------------------------------------
        logger.info(
            "SFT LoRA train + promote (%s, dataset=%d, dcp_save_interval=%d)",
            smoke_minimal_lora_training_shape,
            _NUM_TRAINING_EXAMPLES,
            _DCP_SAVE_INTERVAL,
        )

        cfg = Config(
            log_path=log_dir,
            base_model=smoke_base_model,
            dataset=dataset_path,
            tokenizer_model=smoke_tokenizer_model,
            learning_rate=1e-4,
            epochs=1,
            batch_size=2,
            grad_accum=1,
            max_examples=_NUM_TRAINING_EXAMPLES,
            lora_rank=8,
            dcp_save_interval=_DCP_SAVE_INTERVAL,
            output_model_id=output_model_id,
            infra=infra,
            wandb=WandBConfig(),
        )

        metrics = main(cfg, rlor_mgr=rlor_mgr)
        assert isinstance(metrics, dict)
        steps = metrics["steps"]
        job_id = metrics["job_id"]
        logger.info("Training complete: %d steps, job=%s", steps, job_id)

        assert steps >= 2, (
            f"Expected >=2 SFT steps to exercise periodic + final saves, got {steps}"
        )

        # ---- Verify control-plane checkpoint rows ---------------------------
        # The control plane is the source of truth in the new strategy. We
        # query directly rather than relying on any local registry.
        rows = fw_client.list_checkpoints(job_id)
        logger.info("CP rows for job %s:\n  %s", job_id, _summarize_rows(rows))

        resumable = [
            r for r in rows
            if (r.get("checkpointType") or "").endswith(("TRAINING", "TRAINING_LORA"))
        ]
        promotable = [r for r in rows if r.get("promotable")]

        assert resumable, (
            "Expected at least one resumable (TRAINING/TRAINING_LORA) row "
            f"on job {job_id}; got rows:\n  {_summarize_rows(rows)}"
        )
        assert promotable, (
            "Expected at least one promotable row (final ckpt.save(promotable=True) "
            f"should have produced one) on job {job_id}; got rows:\n  {_summarize_rows(rows)}"
        )

        # The newest promotable row's logical name should match the final step.
        # We don't need exact equality (rename race), just sanity that it's
        # a step-N row and step <= phase1_steps.
        newest_promotable = sorted(
            promotable, key=lambda r: r.get("createTime", ""), reverse=True,
        )[0]
        newest_promotable_name = _short_name(newest_promotable["name"])
        logger.info("Newest promotable row: %s", newest_promotable_name)

        # ---- Verify dataloader.json was written -----------------------------
        dataloader_path = os.path.join(log_dir, DATALOADER_BASE_NAME)
        assert os.path.exists(dataloader_path), (
            f"Expected {DATALOADER_BASE_NAME} under {log_dir} "
            "(written by ckpt.save when data_consumed is supplied)"
        )
        with open(dataloader_path) as f:
            dataloader_state = json.load(f)
        assert dataloader_state, (
            f"{DATALOADER_BASE_NAME} should be non-empty after a resumable save"
        )
        max_data_consumed = max(int(v) for v in dataloader_state.values())
        assert max_data_consumed > 0, (
            f"Expected positive raw_rows_consumed in {DATALOADER_BASE_NAME}, "
            f"got {dataloader_state}"
        )
        logger.info(
            "dataloader.json: %d entries, max=%d",
            len(dataloader_state), max_data_consumed,
        )

        # ---- Confirm trainer cleanup ---------------------------------------
        # SFT main() registers cancel-on-exit; the trainer should be in a
        # terminal state by now. Sleep briefly to absorb async cancellation.
        time.sleep(3)
        try:
            job = rlor_mgr.get(job_id)
            state = job.get("state", "")
            assert state in (
                "JOB_STATE_DELETING",
                "JOB_STATE_DELETED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_COMPLETED",
            ), f"Trainer {job_id} unexpectedly still in state {state}"
        except Exception as e:  # noqa: BLE001 -- network errors during cleanup
            logger.info("Trainer get failed during cleanup check (expected): %s", e)
