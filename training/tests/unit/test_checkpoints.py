"""Unit tests for ``training.utils.checkpoints``."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from training.utils.checkpoints import (
    DATALOADER_BASE_NAME,
    ResumeInfo,
    TrainingCheckpoints,
    _logical_name,
    _newest_first,
    _logical_name_for_run,
    _public_logical_name,
    validate_warm_start_config,
)


@pytest.fixture
def log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _mock_fw_client(rows=None):
    """``fw._rows`` is the mutable source of truth; ``list_checkpoints`` returns
    a snapshot of it. This lets the client mock append a new row on
    ``save_state``/``save_weights_for_sampler``, exercising the polling
    codepath in ``TrainingCheckpoints``."""
    fw = MagicMock()
    fw._rows = list(rows or [])
    fw.list_checkpoints = MagicMock(side_effect=lambda job_id, **kw: list(fw._rows))
    fw.promote_checkpoint.return_value = {"state": "READY", "kind": "HF_BASE_MODEL"}
    return fw


def _mock_client(fw=None, save_state_renames_to: str | None = None):
    """Build a fake ReconnectableClient.

    ``save_state_renames_to`` simulates the service-mode trainer renaming the
    DCP checkpoint to its internal step counter: when set, the fake appends a
    CP row named ``save_state_renames_to`` (instead of the caller name) so that
    ``TrainingCheckpoints._resolve_cp_name_after_save`` polls and finds the
    renamed row. Default (None) honors the caller name.
    """
    client = MagicMock()
    client.resolve_checkpoint_path.side_effect = lambda name, source_job_id=None: (
        f"path://{source_job_id or 'self'}/{name}"
    )

    # Use a counter so appended rows have monotonically-increasing createTime
    # (the poll picks newest-first, so this matters for multi-save tests).
    counter = {"n": 0}

    def _save_state(name):
        cp_name = save_state_renames_to or name
        if fw is not None:
            counter["n"] += 1
            fw._rows.append(
                {
                    "name": f"accounts/a/rlorTrainerJobs/job-1/checkpoints/{cp_name}",
                    "createTime": f"2099-01-01T00:00:{counter['n']:02d}Z",
                    "checkpointType": "CHECKPOINT_TYPE_TRAINING",
                    "promotable": False,
                }
            )
        result = MagicMock()
        result.path = f"tinker://policy/{cp_name}"
        return result

    client.save_state.side_effect = _save_state
    client.save_weights_for_sampler.side_effect = lambda name, checkpoint_type="base": (
        MagicMock(snapshot_name=f"{name}-snap")
    )
    return client


def _row(short_name, *, ctype, promotable, create_time):
    return {
        "name": f"accounts/a/rlorTrainerJobs/job-1/checkpoints/{short_name}",
        "createTime": create_time,
        "checkpointType": ctype,
        "promotable": promotable,
    }


def _make(
    log_dir,
    *,
    fw_rows=None,
    lora_rank=0,
    save_state_renames_to: str | None = None,
    serverless=False,
    current_run_id: str | None = None,
):
    fw = _mock_fw_client(rows=fw_rows)
    client = _mock_client(fw=fw, save_state_renames_to=save_state_renames_to)
    ckpt = TrainingCheckpoints(
        client,
        fw,
        trainer_id="job-1",
        log_path=log_dir,
        lora_rank=lora_rank,
        serverless=serverless,
        # Disable stabilization + tighten timeouts so unit tests run instantly.
        save_appear_timeout_s=5.0,
        save_stabilize_s=0.0,
        save_poll_s=0.01,
        current_run_id=current_run_id,
    )
    return ckpt, client, fw


# -- validate_warm_start_config ------------------------------------------------


class TestValidateWarmStartConfig:
    def test_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_warm_start_config(
                warm_start_from_adapter="some/adapter",
                init_from_checkpoint="job:step-5",
                lora_rank=8,
            )

    def test_warm_start_requires_lora(self):
        with pytest.raises(ValueError, match="cfg.base_model"):
            validate_warm_start_config(
                warm_start_from_adapter="some/adapter",
                init_from_checkpoint=None,
                lora_rank=0,
            )

    def test_ok(self):
        validate_warm_start_config(
            warm_start_from_adapter=None,
            init_from_checkpoint=None,
            lora_rank=0,
        )
        validate_warm_start_config(
            warm_start_from_adapter="a",
            init_from_checkpoint=None,
            lora_rank=8,
        )


# -- resume --------------------------------------------------------------------


class TestResume:
    def test_fresh_start_when_empty(self, log_dir):
        ckpt, client, _ = _make(log_dir, fw_rows=[])
        assert ckpt.resume() is None
        client.load_state_with_optimizer.assert_not_called()

    def test_resume_newest_training_row(self, log_dir):
        rows = [
            _row(
                "step-5",
                ctype="CHECKPOINT_TYPE_TRAINING",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
            _row(
                "step-10",
                ctype="CHECKPOINT_TYPE_TRAINING",
                promotable=False,
                create_time="2026-04-02T00:00:00Z",
            ),
            _row(
                "step-7-sampler",
                ctype="CHECKPOINT_TYPE_INFERENCE_BASE",
                promotable=True,
                create_time="2026-04-03T00:00:00Z",
            ),
        ]
        # Pre-populate dataloader.json so resume can recover data_consumed.
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME), "w") as f:
            json.dump({"step-5": 40, "step-10": 80}, f)

        ckpt, client, _ = _make(log_dir, fw_rows=rows)
        info = ckpt.resume()
        assert info == ResumeInfo(step=10, data_consumed=80, source_job_id="job-1")
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-10")

    def test_resume_picks_training_lora_too(self, log_dir):
        rows = [
            _row(
                "step-5",
                ctype="CHECKPOINT_TYPE_TRAINING_LORA",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
            _row(
                "step-5-hotload",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:05:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=rows, lora_rank=8)
        info = ckpt.resume()
        assert info is not None
        assert info.step == 5
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-5")

    def test_resume_serverless_uses_bare_name(self, log_dir):
        # Serverless: trainer_id is the TrainingSession id, not a job. Auto-resume
        # must NOT build cross_job://<session_id>/<name> — the pooled trainer
        # rejects that (the name must already be session-scoped and session_id is
        # not a source job). It must resume from the bare logical name
        # (source_job_id=None) so the trainer prepends sessions/<sid>/ itself.
        rows = [
            _row(
                "step-5",
                ctype="CHECKPOINT_TYPE_TRAINING_LORA",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=rows, lora_rank=8, serverless=True)
        info = ckpt.resume()
        assert info is not None
        assert info.step == 5
        assert info.source_job_id is None
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-5", source_job_id=None
        )
        # source_job_id=None => bare name (mock renders the absent job as 'self').
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-5")

    def test_resume_serverless_strips_current_run_prefix(self, log_dir):
        current_run_id = "run-abcdef"
        rows = [
            _row(
                "run-oldrun-step-9",
                ctype="CHECKPOINT_TYPE_TRAINING_LORA",
                promotable=False,
                create_time="2026-04-03T00:00:00Z",
            ),
            _row(
                f"{current_run_id}-step-5",
                ctype="CHECKPOINT_TYPE_TRAINING_LORA",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME), "w") as f:
            json.dump({"step-5": 40, "run-abcdef-step-5": 999}, f)

        ckpt, client, _ = _make(
            log_dir,
            fw_rows=rows,
            lora_rank=8,
            serverless=True,
            current_run_id=current_run_id,
        )
        info = ckpt.resume()

        assert info == ResumeInfo(step=5, data_consumed=40, source_job_id=None)
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-5", source_job_id=None
        )
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-5")

    def test_resume_serverless_without_run_id_ignores_run_scoped_rows(self, log_dir):
        rows = [
            _row(
                "run-0123456789abcdef0123456789abcdef-step-99",
                ctype="CHECKPOINT_TYPE_TRAINING_LORA",
                promotable=False,
                create_time="2099-01-01T00:00:00Z",
            ),
            _row(
                "step-5",
                ctype="CHECKPOINT_TYPE_TRAINING_LORA",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(
            log_dir,
            fw_rows=rows,
            lora_rank=8,
            serverless=True,
        )
        info = ckpt.resume()

        assert info == ResumeInfo(step=5, data_consumed=0, source_job_id=None)
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-5", source_job_id=None
        )
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-5")

    def test_init_from_checkpoint_takes_priority(self, log_dir):
        rows = [
            _row(
                "step-50",
                ctype="CHECKPOINT_TYPE_TRAINING",
                promotable=False,
                create_time="2026-04-02T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=rows)
        info = ckpt.resume(init_from_checkpoint="other-job:step-3")
        assert info == ResumeInfo(step=0, data_consumed=0, source_job_id="other-job")
        client.load_state_with_optimizer.assert_called_once_with(
            "path://other-job/step-3"
        )

    def test_same_trainer_init_from_checkpoint_resumes_cursor(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME), "w") as f:
            json.dump({"step-3": 24}, f)

        ckpt, client, _ = _make(log_dir, fw_rows=[])
        info = ckpt.resume(init_from_checkpoint="job-1:step-3")
        assert info == ResumeInfo(step=3, data_consumed=24, source_job_id="job-1")
        client.resolve_checkpoint_path.assert_called_once_with("step-3")
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-3")

    def test_init_from_checkpoint_serverless_uses_bare_name(self, log_dir):
        # Serverless: init_from_checkpoint must resume from the bare name, not
        # cross_job://<session>/<name> (the pool trainer rejects that). A spec
        # naming THIS session is accepted with its prefix stripped.
        ckpt, client, _ = _make(
            log_dir, serverless=True
        )  # trainer_id/session == "job-1"
        info = ckpt.resume(init_from_checkpoint="job-1:step-5")
        assert info == ResumeInfo(step=0, data_consumed=0, source_job_id=None)
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-5", source_job_id=None
        )
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-5")

    def test_init_from_checkpoint_serverless_rejects_cross_session(self, log_dir):
        # Cross-session warm-start is unsupported on the shared pool (isolation).
        ckpt, _, _ = _make(log_dir, serverless=True)  # session == "job-1"
        with pytest.raises(ValueError, match="another session"):
            ckpt.resume(init_from_checkpoint="other-session:step-5")

    def test_warm_start_adapter_when_no_resume(self, log_dir):
        ckpt, client, _ = _make(log_dir, fw_rows=[], lora_rank=8)
        info = ckpt.resume(warm_start_from_adapter="hf/adapter")
        assert info == ResumeInfo(step=0, data_consumed=0, source_job_id=None)
        client.load_adapter.assert_called_once_with("hf/adapter")

    def test_list_failure_treated_as_fresh_start(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        fw.list_checkpoints.side_effect = RuntimeError("503 Service Unavailable")
        info = ckpt.resume()
        assert info is None
        client.load_state_with_optimizer.assert_not_called()


# -- save ----------------------------------------------------------------------


class TestSave:
    def test_resumable_only_writes_dcp_and_dataloader(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        ckpt.save("step-1", resumable=True, promotable=False, data_consumed=100)

        client.save_state.assert_called_once_with("step-1")
        client.save_weights_for_sampler.assert_not_called()

        with open(os.path.join(log_dir, DATALOADER_BASE_NAME)) as f:
            assert json.load(f) == {"step-1": 100}

    def test_promotable_only_writes_sampler_no_dataloader(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_state.assert_not_called()
        client.save_weights_for_sampler.assert_called_once_with(
            "step-1", checkpoint_type="base"
        )
        assert not os.path.exists(os.path.join(log_dir, DATALOADER_BASE_NAME))

    def test_both_writes_both(self, log_dir):
        ckpt, client, _ = _make(log_dir)
        ckpt.save("step-1", resumable=True, promotable=True, data_consumed=42)
        client.save_state.assert_called_once_with("step-1")
        client.save_weights_for_sampler.assert_called_once()

    def test_neither_raises(self, log_dir):
        ckpt, _, _ = _make(log_dir)
        with pytest.raises(ValueError, match="at least one"):
            ckpt.save("step-1", resumable=False, promotable=False)

    def test_skip_if_promotable_already_exists(self, log_dir):
        existing = [
            _row(
                "step-1",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=existing, lora_rank=8)
        ckpt.save("step-1", resumable=True, promotable=True, data_consumed=10)
        client.save_state.assert_called_once()
        client.save_weights_for_sampler.assert_not_called()

    def test_no_skip_when_existing_not_promotable(self, log_dir):
        existing = [
            _row(
                "step-1",
                ctype="CHECKPOINT_TYPE_TRAINING",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=existing)
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_called_once()

    def test_skip_check_failure_falls_back_to_save(self, log_dir):
        ckpt, client, fw = _make(log_dir)
        fw.list_checkpoints.side_effect = RuntimeError("503")
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_called_once()

    def test_skip_matches_suffixed_server_name(self, log_dir):
        """The trainer appends ``-<8 hex>`` to sampler names server-side.
        Skip should match on the logical name (pre-suffix) so callers passing
        ``step-1`` match a stored row ``step-1-abcd1234``."""
        existing = [
            _row(
                "step-1-abcd1234",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=existing, lora_rank=8)
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_not_called()

    def test_skip_matches_run_scoped_suffixed_server_name(self, log_dir):
        """Serverless session checkpoints surface as ``run-id-checkpoint``.

        The cookbook caller only knows the checkpoint leaf it passed to the
        trainer, so skip matching must ignore the public run prefix and the
        trainer's 8-hex sampler suffix.
        """
        current_run_id = "run-0123456789abcdef0123456789abcdef"
        existing = [
            _row(
                f"{current_run_id}-step-1-abcd1234",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(
            log_dir,
            fw_rows=existing,
            lora_rank=8,
            serverless=True,
            current_run_id=current_run_id,
        )
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_not_called()

    def test_skip_without_run_id_ignores_run_scoped_promotable_checkpoint(
        self, log_dir
    ):
        existing = [
            _row(
                "run-0123456789abcdef0123456789abcdef-step-1-abcd1234",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2099-01-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(
            log_dir,
            fw_rows=existing,
            lora_rank=8,
            serverless=True,
        )
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_called_once_with(
            "step-1", checkpoint_type="base"
        )

    def test_skip_ignores_other_run_promotable_checkpoint(self, log_dir):
        current_run_id = "run-current"
        existing = [
            _row(
                "run-previous-step-1-abcd1234",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2099-01-01T00:00:00Z",
            ),
            _row(
                f"{current_run_id}-step-2-abcd1234",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2099-01-01T00:00:01Z",
            ),
        ]
        ckpt, client, _ = _make(
            log_dir,
            fw_rows=existing,
            lora_rank=8,
            serverless=True,
            current_run_id=current_run_id,
        )
        ckpt.save("step-1", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_called_once_with(
            "step-1", checkpoint_type="base"
        )

    def test_skip_does_not_match_different_step(self, log_dir):
        existing = [
            _row(
                "step-1-abcd1234",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, client, _ = _make(log_dir, fw_rows=existing, lora_rank=8)
        ckpt.save("step-2", resumable=False, promotable=True)
        client.save_weights_for_sampler.assert_called_once()

    def test_dataloader_keyed_on_server_name_when_trainer_renames(self, log_dir):
        """Trainer service-mode may rename DCP saves to its internal step
        counter: caller passes "step-42" but the service writes "step-0".
        ``dataloader.json`` must be keyed on the server-returned name so the
        resume lookup (which reads names from the control plane) matches."""
        ckpt, client, _ = _make(log_dir, save_state_renames_to="step-0")
        ckpt.save("step-42", resumable=True, promotable=False, data_consumed=777)
        client.save_state.assert_called_once_with("step-42")

        with open(os.path.join(log_dir, DATALOADER_BASE_NAME)) as f:
            data = json.load(f)
        assert data == {"step-0": 777}, f"expected server name keyed, got {data}"

    def test_dataloader_keyed_on_caller_name_when_no_rename(self, log_dir):
        """When the server honors the caller name, ``dataloader.json`` keys
        by the same string (no regression vs prior behavior)."""
        ckpt, client, _ = _make(log_dir)  # default: no rename
        ckpt.save("step-5", resumable=True, promotable=False, data_consumed=50)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME)) as f:
            assert json.load(f) == {"step-5": 50}

    def test_promotable_save_waits_for_run_scoped_cp_row(self, log_dir):
        current_run_id = "run-f7bd5935b27b46d2ac21c90ac7a19cd5"
        row = _row(
            f"{current_run_id}-step-8-c42a35e8",
            ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
            promotable=True,
            create_time="2099-01-01T00:00:05Z",
        )
        old_row = _row(
            "run-oldrun-step-8-c42a35e8",
            ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
            promotable=True,
            create_time="2099-01-01T00:00:10Z",
        )
        ckpt, client, fw = _make(
            log_dir,
            lora_rank=8,
            serverless=True,
            current_run_id=current_run_id,
        )

        calls = {"n": 0}

        def _list_after_delay(job_id, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 3:
                return [old_row, row]
            return [old_row]

        fw.list_checkpoints.side_effect = _list_after_delay

        ckpt.save("step-8", resumable=False, promotable=True)

        client.save_weights_for_sampler.assert_called_once_with(
            "step-8", checkpoint_type="base"
        )
        assert calls["n"] >= 3


class TestLogicalName:
    @pytest.mark.parametrize(
        "stored,logical",
        [
            ("step-3", "step-3"),
            ("step-3-abcd1234", "step-3"),
            ("resume-3-base-45dda197", "resume-3-base"),
            ("step-11-45dda197", "step-11"),
            ("step-3-ABCD1234", "step-3-ABCD1234"),  # uppercase — not the pattern
            ("step-3-12345", "step-3-12345"),  # 5 chars — not 8
            ("step-3-abcdefghi", "step-3-abcdefghi"),  # 9 chars — not 8
        ],
    )
    def test_strip_session_suffix(self, stored, logical):
        assert _logical_name(stored) == logical

    @pytest.mark.parametrize(
        "stored,logical",
        [
            ("step-3-abcd1234", "step-3"),
            ("run-0123456789abcdef0123456789abcdef-step-3-abcd1234", "step-3"),
            (
                "run-0123456789abcdef0123456789abcdef-resume-3-base-45dda197",
                "resume-3-base",
            ),
            ("run-0123456789abcdef0123456789abcdef:step-3-abcd1234", "step-3"),
        ],
    )
    def test_public_logical_name_ignores_run_prefix(self, stored, logical):
        assert _public_logical_name(stored) == logical

    def test_logical_name_for_current_run_accepts_short_run_id(self):
        assert (
            _logical_name_for_run("run-abcdef-step-3-abcd1234", "run-abcdef")
            == "step-3"
        )
        assert (
            _logical_name_for_run("run-other-step-3-abcd1234", "run-abcdef")
            == "run-other-step-3"
        )


# -- promote_latest ------------------------------------------------------------


class TestPromoteLatest:
    """``promote_latest`` hands the row's 4-segment ``name`` to the SDK
    verbatim. The cookbook does not disassemble into
    ``(job_id, checkpoint_id)``."""

    def test_picks_newest_promotable_and_passes_full_name(self, log_dir):
        rows = [
            _row(
                "step-5",
                ctype="CHECKPOINT_TYPE_INFERENCE_BASE",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
            _row(
                "step-10",
                ctype="CHECKPOINT_TYPE_INFERENCE_BASE",
                promotable=True,
                create_time="2026-04-02T00:00:00Z",
            ),
            _row(
                "step-10-dcp",
                ctype="CHECKPOINT_TYPE_TRAINING",
                promotable=False,
                create_time="2026-04-02T00:05:00Z",
            ),
        ]
        ckpt, _, fw = _make(log_dir, fw_rows=rows)
        ckpt.promote_latest("my-model", "accounts/a/models/qwen3-1p7b-bf16")
        fw.promote_checkpoint.assert_called_once_with(
            name="accounts/a/rlorTrainerJobs/job-1/checkpoints/step-10",
            output_model_id="my-model",
            base_model="accounts/a/models/qwen3-1p7b-bf16",
            hot_load_deployment_id=None,
        )

    def test_serverless_promote_latest_uses_current_run_only(self, log_dir):
        current_run_id = "run-current"
        rows = [
            _row(
                "run-previous-step-99",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2099-01-01T00:00:00Z",
            ),
            _row(
                f"{current_run_id}-step-5",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, _, fw = _make(
            log_dir,
            fw_rows=rows,
            lora_rank=8,
            serverless=True,
            current_run_id=current_run_id,
        )
        ckpt.promote_latest("my-model", "accounts/a/models/qwen3-1p7b-bf16")
        fw.promote_checkpoint.assert_called_once()
        _, kwargs = fw.promote_checkpoint.call_args
        assert kwargs["name"].endswith(f"/checkpoints/{current_run_id}-step-5")

    def test_serverless_promote_latest_without_run_id_ignores_run_scoped_rows(
        self, log_dir
    ):
        rows = [
            _row(
                "run-0123456789abcdef0123456789abcdef-step-99",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2099-01-01T00:00:00Z",
            ),
        ]
        ckpt, _, fw = _make(log_dir, fw_rows=rows, lora_rank=8, serverless=True)
        with pytest.raises(RuntimeError, match="No promotable"):
            ckpt.promote_latest("my-model", "accounts/a/models/qwen3-1p7b-bf16")
        fw.promote_checkpoint.assert_not_called()

    def test_skips_arc_v2_because_not_promotable(self, log_dir):
        rows = [
            _row(
                "step-5-arc",
                ctype="CHECKPOINT_TYPE_INFERENCE_ARC_V2",
                promotable=False,
                create_time="2026-04-02T00:00:00Z",
            ),
            _row(
                "step-5-lora",
                ctype="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, _, fw = _make(log_dir, fw_rows=rows)
        ckpt.promote_latest("out", "base")
        fw.promote_checkpoint.assert_called_once()
        _, kwargs = fw.promote_checkpoint.call_args
        assert kwargs["name"].endswith("/checkpoints/step-5-lora")

    def test_errors_when_no_promotable(self, log_dir):
        rows = [
            _row(
                "step-1",
                ctype="CHECKPOINT_TYPE_TRAINING",
                promotable=False,
                create_time="2026-04-01T00:00:00Z",
            ),
        ]
        ckpt, _, _ = _make(log_dir, fw_rows=rows)
        with pytest.raises(RuntimeError, match="No promotable"):
            ckpt.promote_latest("out", "base")


# -- dataloader.json bookkeeping -----------------------------------------------


class TestNewestFirstMixedPrecision:
    """The control plane mixes ``...:13Z`` (second precision) and
    ``...:13.123456Z`` (microsecond precision) ``createTime`` strings.
    A lexicographic sort would put the older second-precision row
    ahead of the newer microsecond-precision one because
    ``'Z' (90) > '.' (46)``, silently picking a stale checkpoint in
    ``_latest_resumable`` / ``promote_latest``.
    """

    def test_microsecond_row_beats_second_row_at_later_wall_time(self):
        # Newer row uses microsecond precision; older row uses
        # second precision.  Newer must win.
        newer = {"name": "step-20", "createTime": "2026-04-29T10:00:13.500000Z"}
        older = {"name": "step-10", "createTime": "2026-04-29T10:00:12Z"}
        assert _newest_first([older, newer])[0] is newer

    def test_microsecond_row_beats_same_second_row_with_lex_inversion(self):
        # Same wall-second; ``...:13Z`` lex-sorts AFTER ``...:13.500Z``
        # because ``'Z' (90) > '.' (46)``.  Datetime sort must put
        # the microsecond row first.
        newer = {"name": "step-20", "createTime": "2026-04-29T10:00:13.500000Z"}
        older = {"name": "step-10", "createTime": "2026-04-29T10:00:13Z"}
        assert _newest_first([older, newer])[0] is newer

    def test_unparseable_createtime_sorts_to_end(self):
        valid = {"name": "step-1", "createTime": "2026-04-29T10:00:13.500000Z"}
        broken = {"name": "step-2", "createTime": "not-a-timestamp"}
        missing = {"name": "step-3"}
        ordered = _newest_first([broken, missing, valid])
        assert ordered[0] is valid


class TestDataloaderJson:
    def test_bounded_history(self, log_dir):
        ckpt, _, _ = _make(log_dir)
        for i in range(1, 26):
            ckpt.save(f"step-{i}", resumable=True, promotable=False, data_consumed=i)
        with open(os.path.join(log_dir, DATALOADER_BASE_NAME)) as f:
            data = json.load(f)
        # Keep only the newest 20.
        assert len(data) == 20
        assert set(data.keys()) == {f"step-{i}" for i in range(6, 26)}
