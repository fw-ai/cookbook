from __future__ import annotations

import httpx
import pytest

from training.utils.weight_sync import WeightSyncer


def _http_error(message: str, *, status_code: int = 400) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.fireworks.ai/hot_load/v1/models/hot_load")
    response = httpx.Response(status_code, request=request, text=message)
    return httpx.HTTPStatusError(message, request=request, response=response)


class _FakeDeployMgr:
    def __init__(self, error: httpx.HTTPStatusError | None = None):
        self.calls: list[dict] = []
        self.sync_requests: list[dict] = []
        self.wait_calls: list[dict] = []
        self._error = error
        self.hotload_api_url = "https://api.fireworks.ai"

    def hotload_and_wait(
        self,
        *,
        deployment_id: str,
        base_model: str,
        snapshot_identity: str,
        incremental_snapshot_metadata: dict | None,
        reset_prompt_cache: bool,
        timeout_seconds: int,
    ) -> bool:
        self.calls.append(
            {
                "deployment_id": deployment_id,
                "base_model": base_model,
                "snapshot_identity": snapshot_identity,
                "incremental_snapshot_metadata": incremental_snapshot_metadata,
                "reset_prompt_cache": reset_prompt_cache,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self._error and len(self.calls) == 1:
            raise self._error
        return True

    def _hotload_headers(self, deployment_id: str, base_model: str) -> dict[str, str]:
        return {
            "fireworks-deployment": deployment_id,
            "fireworks-model": base_model,
        }

    def _sync_request(self, url: str, *, method: str, headers: dict, json: dict, timeout: int):
        self.sync_requests.append(
            {
                "url": url,
                "method": method,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        request = httpx.Request(method, url)
        return httpx.Response(200, request=request, json={})

    def wait_for_hotload(
        self,
        *,
        deployment_id: str,
        base_model: str,
        expected_identity: str,
        timeout_seconds: int,
    ) -> bool:
        self.wait_calls.append(
            {
                "deployment_id": deployment_id,
                "base_model": base_model,
                "expected_identity": expected_identity,
                "timeout_seconds": timeout_seconds,
            }
        )
        return True


def test_weight_syncer_retries_without_reset_prompt_cache():
    deploy_mgr = _FakeDeployMgr(
        _http_error(
            "Extra inputs are not permitted, field: 'reset_prompt_cache', value: True"
        )
    )
    syncer = WeightSyncer(
        policy_client=object(),
        deploy_mgr=deploy_mgr,
        deployment_id="dep-123",
        base_model="accounts/test/models/qwen3-4b",
        warmup_after_hotload=False,
    )
    syncer._deployment_checked = True

    syncer._do_hotload("snap-1", "base")
    syncer._do_hotload("snap-2", "delta")

    assert [call["reset_prompt_cache"] for call in deploy_mgr.calls] == [True]
    assert [req["json"]["identity"] for req in deploy_mgr.sync_requests] == ["snap-1", "snap-2"]
    assert "reset_prompt_cache" not in deploy_mgr.sync_requests[0]["json"]
    assert deploy_mgr.sync_requests[1]["json"]["incremental_snapshot_metadata"] == {
        "previous_snapshot_identity": "snap-1",
        "compression_format": "arc_v2",
        "checksum_format": "alder32",
    }
    assert [call["expected_identity"] for call in deploy_mgr.wait_calls] == ["snap-1", "snap-2"]
    assert syncer.base_identity == "snap-2"


def test_weight_syncer_propagates_other_hotload_errors():
    deploy_mgr = _FakeDeployMgr(
        _http_error("snapshot not found")
    )
    syncer = WeightSyncer(
        policy_client=object(),
        deploy_mgr=deploy_mgr,
        deployment_id="dep-123",
        base_model="accounts/test/models/qwen3-4b",
        warmup_after_hotload=False,
    )
    syncer._deployment_checked = True

    with pytest.raises(httpx.HTTPStatusError, match="snapshot not found"):
        syncer._do_hotload("snap-1", "base")

    assert len(deploy_mgr.calls) == 1
    assert deploy_mgr.calls[0]["reset_prompt_cache"] is True
