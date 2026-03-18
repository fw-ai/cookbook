"""Tests for ResourceCleanup context manager."""

from __future__ import annotations

from unittest.mock import MagicMock

from training.utils.infra import ResourceCleanup


class TestResourceCleanup:
    def test_cleans_registered_resources_in_reverse_order(self):
        rlor_mgr = MagicMock()
        deploy_mgr = MagicMock()

        with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
            cleanup.trainer("job-1")
            cleanup.trainer("job-2")
            cleanup.deployment("dep-1")
            cleanup.deployment("dep-2", action="scale_to_zero")

        assert rlor_mgr.delete.call_args_list == [
            (("job-2",),),
            (("job-1",),),
        ]
        deploy_mgr.scale_to_zero.assert_called_once_with("dep-2")
        deploy_mgr.delete.assert_called_once_with("dep-1")

    def test_cleans_on_exception_and_skips_unregistered(self):
        rlor_mgr = MagicMock()

        try:
            with ResourceCleanup(rlor_mgr) as cleanup:
                cleanup.trainer("created-job")
                raise RuntimeError("training crashed")
        except RuntimeError:
            pass

        rlor_mgr.delete.assert_called_once_with("created-job")

    def test_delete_trainer_deletes_and_unregisters(self):
        rlor_mgr = MagicMock()

        with ResourceCleanup(rlor_mgr) as cleanup:
            cleanup.trainer("job-keep")
            cleanup.trainer("job-early")
            cleanup.delete_trainer("job-early")

        assert rlor_mgr.delete.call_args_list == [
            (("job-early",),),
            (("job-keep",),),
        ]
