"""Unit tests for the cookbook checkpoint boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.utils.checkpoints import (
    DATALOADER_BASE_NAME,
    DATALOADER_HISTORY_KEEP,
    ResumeInfo,
    TrainingCheckpoints,
    _logical_name,
    _logical_name_for_run,
    _newest_first,
    _public_logical_name,
    validate_warm_start_config,
)


def _row(
    short_name: str,
    *,
    checkpoint_type: str = "CHECKPOINT_TYPE_TRAINING",
    promotable: bool = False,
    create_time: str = "2026-07-01T00:00:00Z",
    job_id: str = "job-1",
) -> dict:
    return {
        "name": f"accounts/a/rlorTrainerJobs/{job_id}/checkpoints/{short_name}",
        "createTime": create_time,
        "checkpointType": checkpoint_type,
        "promotable": promotable,
    }


def _make(
    log_dir: str,
    *,
    rows: list[dict] | None = None,
    lora_rank: int = 0,
    saved_name: str | None = None,
    serverless: bool = False,
    current_run_id: str | None = None,
) -> tuple[TrainingCheckpoints, MagicMock, MagicMock]:
    fw = MagicMock()
    fw.rows = list(rows or [])
    fw.list_checkpoints.side_effect = lambda _job_id, **_kwargs: list(fw.rows)
    fw.promote_checkpoint.return_value = {"state": "READY"}

    client = MagicMock()
    client.resolve_checkpoint_path.side_effect = (
        lambda name, source_job_id=None: f"path://{source_job_id or 'self'}/{name}"
    )
    client.save_state.side_effect = lambda name: SimpleNamespace(
        path=f"tinker://run/state/{saved_name or name}"
    )

    def _save_sampler(name: str, *args, checkpoint_type=None, **kwargs):
        if checkpoint_type == "base":
            suffix = "LORA" if lora_rank else "BASE"
            public_name = (
                f"{current_run_id}-{name}" if serverless and current_run_id else name
            )
            fw.rows.append(
                _row(
                    public_name,
                    checkpoint_type=f"CHECKPOINT_TYPE_INFERENCE_{suffix}",
                    promotable=True,
                    create_time=datetime.now(timezone.utc).isoformat(),
                )
            )
        return SimpleNamespace(path=f"snapshot://{name}")

    client.save_weights_for_sampler.side_effect = _save_sampler
    checkpoints = TrainingCheckpoints(
        client,
        fw,
        trainer_id="job-1",
        log_path=log_dir,
        lora_rank=lora_rank,
        serverless=serverless,
        current_run_id=current_run_id,
        save_appear_timeout_s=0.1,
        save_poll_s=0.001,
    )
    return checkpoints, client, fw


class TestWarmStartValidation:
    def test_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_warm_start_config(
                warm_start_from_adapter="adapter",
                init_from_checkpoint="job:step-5",
                lora_rank=8,
            )

    def test_adapter_requires_lora(self):
        with pytest.raises(ValueError, match="cfg.base_model"):
            validate_warm_start_config(
                warm_start_from_adapter="adapter",
                init_from_checkpoint=None,
                lora_rank=0,
            )


class TestResume:
    def test_fresh_start_is_explicit_zero_state(self, tmp_path):
        checkpoints, client, _ = _make(str(tmp_path))

        assert checkpoints.resume() == ResumeInfo()
        client.load_state_with_optimizer.assert_not_called()

    def test_newest_resumable_rpc_row_drives_resume(self, tmp_path):
        rows = [
            _row("step-5", create_time="2026-07-01T00:00:00Z"),
            _row("step-10", create_time="2026-07-02T00:00:00Z"),
            _row(
                "step-20-abcd1234",
                checkpoint_type="CHECKPOINT_TYPE_INFERENCE_BASE",
                promotable=True,
                create_time="2026-07-03T00:00:00Z",
            ),
        ]
        (tmp_path / DATALOADER_BASE_NAME).write_text(
            json.dumps({"job-1": {"5": 40, "10": 80}})
        )
        checkpoints, client, _ = _make(str(tmp_path), rows=rows)

        info = checkpoints.resume()

        assert info == ResumeInfo(step=10, row_cursor=80, source_job_id="job-1")
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-10", source_job_id=None
        )
        client.load_state_with_optimizer.assert_called_once_with("path://self/step-10")

    def test_list_row_can_be_passed_directly_for_cross_job_resume(self, tmp_path):
        row = _row("step-7", job_id="source-job")
        (tmp_path / DATALOADER_BASE_NAME).write_text(
            json.dumps({"source-job": {"7": 91}})
        )
        checkpoints, client, _ = _make(str(tmp_path))

        info = checkpoints.resume(init_from_checkpoint=row)

        assert info == ResumeInfo(step=7, row_cursor=91, source_job_id="source-job")
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-7", source_job_id="source-job"
        )

    def test_full_resource_name_can_be_passed_directly(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path))

        info = checkpoints.resume(
            init_from_checkpoint=(
                "accounts/a/rlorTrainerJobs/source-job/checkpoints/step-4"
            ),
            dataloader_cursor=33,
        )

        assert info == ResumeInfo(step=4, row_cursor=33, source_job_id="source-job")

    def test_explicit_cursor_bypasses_local_lookup(self, tmp_path, monkeypatch):
        checkpoints, _, _ = _make(str(tmp_path), rows=[_row("step-3")])

        monkeypatch.setattr(
            checkpoints,
            "_read_row_cursor",
            lambda *_args: pytest.fail("local cursor lookup must be bypassed"),
        )

        assert checkpoints.resume(dataloader_cursor=17).row_cursor == 17

    def test_cross_job_resume_preserves_step_with_missing_local_cursor(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path))

        info = checkpoints.resume(init_from_checkpoint="source-job:step-12")

        assert info == ResumeInfo(step=12, row_cursor=0, source_job_id="source-job")

    def test_non_resumable_rpc_row_is_rejected(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path))
        row = _row(
            "step-5",
            checkpoint_type="CHECKPOINT_TYPE_INFERENCE_BASE",
            promotable=True,
        )

        with pytest.raises(ValueError, match="resumable"):
            checkpoints.resume(init_from_checkpoint=row)

    def test_invalid_checkpoint_name_is_rejected(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path))

        with pytest.raises(ValueError, match="step-N"):
            checkpoints.resume(init_from_checkpoint="job-1:final")

    def test_serverless_resume_uses_current_run_only(self, tmp_path):
        run_id = "run-0123456789abcdef0123456789abcdef"
        rows = [
            _row(
                "run-fedcba9876543210fedcba9876543210-step-99",
                checkpoint_type="CHECKPOINT_TYPE_TRAINING_LORA",
                create_time="2099-01-01T00:00:00Z",
            ),
            _row(
                f"{run_id}-step-5",
                checkpoint_type="CHECKPOINT_TYPE_TRAINING_LORA",
            ),
        ]
        (tmp_path / DATALOADER_BASE_NAME).write_text(
            json.dumps({"job-1": {"5": 44}})
        )
        checkpoints, client, _ = _make(
            str(tmp_path),
            rows=rows,
            lora_rank=8,
            serverless=True,
            current_run_id=run_id,
        )

        info = checkpoints.resume()

        assert info == ResumeInfo(step=5, row_cursor=44, source_job_id="job-1")
        client.resolve_checkpoint_path.assert_called_once_with(
            "step-5", source_job_id=None
        )

    def test_list_failure_does_not_silently_start_fresh(self, tmp_path):
        checkpoints, client, fw = _make(str(tmp_path))
        fw.list_checkpoints.side_effect = RuntimeError("503")

        with pytest.raises(RuntimeError, match="503"):
            checkpoints.resume()
        client.load_state_with_optimizer.assert_not_called()

    def test_auto_latest_can_be_disabled_without_listing_or_local_lookup(
        self, tmp_path, monkeypatch
    ):
        checkpoints, client, fw = _make(str(tmp_path), rows=[_row("step-3")])
        monkeypatch.setattr(
            checkpoints,
            "_read_row_cursor",
            lambda *_args: pytest.fail("local cursor lookup must not run"),
        )

        info = checkpoints.resume(
            dataloader_cursor=11,
            auto_latest=False,
        )

        assert info == ResumeInfo(row_cursor=11)
        fw.list_checkpoints.assert_not_called()
        client.load_state_with_optimizer.assert_not_called()

    def test_warm_start_loads_adapter_and_honors_explicit_cursor(self, tmp_path):
        checkpoints, client, _ = _make(str(tmp_path), lora_rank=8)

        info = checkpoints.resume(
            warm_start_from_adapter="gs://bucket/adapter",
            dataloader_cursor=6,
        )

        assert info == ResumeInfo(row_cursor=6)
        client.load_adapter.assert_called_once_with("gs://bucket/adapter")


class TestSave:
    def test_resumable_save_writes_only_job_step_cursor_mapping(self, tmp_path):
        checkpoints, client, _ = _make(str(tmp_path))

        checkpoints.save(3, resumable=True, promotable=False, row_cursor=24)

        client.save_state.assert_called_once_with("step-3")
        assert json.loads((tmp_path / DATALOADER_BASE_NAME).read_text()) == {
            "job-1": {"3": 24}
        }

    def test_rpc_result_step_is_used_without_control_plane_poll(self, tmp_path):
        checkpoints, _, fw = _make(str(tmp_path), saved_name="step-2")

        checkpoints.save(42, resumable=True, promotable=False, row_cursor=777)

        assert json.loads((tmp_path / DATALOADER_BASE_NAME).read_text()) == {
            "job-1": {"2": 777}
        }
        fw.list_checkpoints.assert_not_called()

    def test_resumable_save_without_cursor_writes_no_local_state(self, tmp_path):
        checkpoints, client, fw = _make(str(tmp_path))

        checkpoints.save(1, resumable=True, promotable=False)

        client.save_state.assert_called_once_with("step-1")
        fw.list_checkpoints.assert_not_called()
        assert not (tmp_path / DATALOADER_BASE_NAME).exists()

    def test_promotable_save_forces_complete_export_behind_boundary(self, tmp_path):
        checkpoints, client, _ = _make(str(tmp_path))

        checkpoints.save(4, resumable=False, promotable=True)

        client.save_weights_for_sampler.assert_called_once_with(
            "step-4", checkpoint_type="base"
        )
        assert not (tmp_path / DATALOADER_BASE_NAME).exists()

    def test_existing_promotable_row_skips_duplicate_export(self, tmp_path):
        rows = [
            _row(
                "step-4-abcd1234",
                checkpoint_type="CHECKPOINT_TYPE_INFERENCE_LORA",
                promotable=True,
            )
        ]
        checkpoints, client, _ = _make(str(tmp_path), rows=rows, lora_rank=8)

        checkpoints.save(4, resumable=False, promotable=True)

        client.save_weights_for_sampler.assert_not_called()

    def test_invalid_save_arguments_are_rejected(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path))

        with pytest.raises(ValueError, match="at least one"):
            checkpoints.save(1, resumable=False, promotable=False)
        with pytest.raises(ValueError, match="step"):
            checkpoints.save(-1, resumable=True, promotable=False, row_cursor=0)
        with pytest.raises(ValueError, match="row_cursor"):
            checkpoints.save(1, resumable=True, promotable=False, row_cursor=-1)


class TestWeightSync:
    @pytest.mark.parametrize("lora_rank", [0, 8])
    def test_sdk_selects_lora_full_and_full_base_delta_modes(
        self, tmp_path, lora_rank
    ):
        checkpoints, client, _ = _make(str(tmp_path), lora_rank=lora_rank)
        hotload = MagicMock()

        path = checkpoints.sync_weights(9, hotload)

        assert path == "snapshot://step-9"
        client.save_weights_for_sampler.assert_called_once_with("step-9")
        hotload.assert_called_once_with("snapshot://step-9")

    def test_missing_snapshot_path_is_an_error(self, tmp_path):
        checkpoints, client, _ = _make(str(tmp_path))
        client.save_weights_for_sampler.side_effect = None
        client.save_weights_for_sampler.return_value = SimpleNamespace(path=None)

        with pytest.raises(RuntimeError, match="returned no snapshot path"):
            checkpoints.sync_weights(2, MagicMock())


class TestCursorStore:
    def test_legacy_flat_mapping_migrates_on_write(self, tmp_path):
        (tmp_path / DATALOADER_BASE_NAME).write_text(
            json.dumps({"step-2": 20, "step-4": 40})
        )
        checkpoints, _, _ = _make(str(tmp_path))

        assert checkpoints.resume(
            init_from_checkpoint="job-1:step-4"
        ).row_cursor == 40
        checkpoints.save(6, resumable=True, promotable=False, row_cursor=60)

        assert json.loads((tmp_path / DATALOADER_BASE_NAME).read_text()) == {
            "job-1": {"2": 20, "4": 40, "6": 60}
        }

    def test_history_is_bounded_per_job(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path))

        for step in range(DATALOADER_HISTORY_KEEP + 3):
            checkpoints.save(
                step,
                resumable=True,
                promotable=False,
                row_cursor=step * 2,
            )

        mapping = json.loads((tmp_path / DATALOADER_BASE_NAME).read_text())["job-1"]
        assert len(mapping) == DATALOADER_HISTORY_KEEP
        assert min(map(int, mapping)) == 3

    def test_corrupt_mapping_is_treated_as_empty(self, tmp_path):
        (tmp_path / DATALOADER_BASE_NAME).write_text("{bad json")
        checkpoints, _, _ = _make(str(tmp_path))

        assert checkpoints.resume(
            init_from_checkpoint="job-1:step-2"
        ).row_cursor == 0


class TestPromote:
    def test_latest_promotable_resource_is_passed_to_sdk(self, tmp_path):
        rows = [
            _row(
                "step-5",
                checkpoint_type="CHECKPOINT_TYPE_INFERENCE_BASE",
                promotable=True,
                create_time="2026-07-01T00:00:00Z",
            ),
            _row(
                "step-10",
                checkpoint_type="CHECKPOINT_TYPE_INFERENCE_BASE",
                promotable=True,
                create_time="2026-07-02T00:00:00Z",
            ),
        ]
        checkpoints, _, fw = _make(str(tmp_path), rows=rows)

        checkpoints.promote_latest("output", "accounts/a/models/base")

        fw.promote_checkpoint.assert_called_once_with(
            name="accounts/a/rlorTrainerJobs/job-1/checkpoints/step-10",
            output_model_id="output",
            base_model="accounts/a/models/base",
            hot_load_deployment_id=None,
        )

    def test_missing_promotable_row_is_an_error(self, tmp_path):
        checkpoints, _, _ = _make(str(tmp_path), rows=[_row("step-1")])

        with pytest.raises(RuntimeError, match="No promotable"):
            checkpoints.promote_latest("output", "base")


class TestHelpers:
    @pytest.mark.parametrize(
        ("stored", "logical"),
        [
            ("step-3", "step-3"),
            ("step-3-abcd1234", "step-3"),
            ("resume-3-base-45dda197", "resume-3-base"),
        ],
    )
    def test_logical_name(self, stored, logical):
        assert _logical_name(stored) == logical

    def test_public_and_current_run_logical_names(self):
        run_id = "run-0123456789abcdef0123456789abcdef"
        stored = f"{run_id}-step-3-abcd1234"
        assert _public_logical_name(stored) == "step-3"
        assert _logical_name_for_run(stored, run_id) == "step-3"

    def test_newest_first_parses_mixed_timestamp_precision(self):
        rows = [
            _row("step-1", create_time="2026-07-01T00:00:13Z"),
            _row("step-2", create_time="2026-07-01T00:00:13.123456Z"),
        ]
        assert _newest_first(rows)[0]["name"].endswith("/step-2")
