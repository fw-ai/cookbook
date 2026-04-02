"""Cookbook compatibility wrapper around the SDK WeightSyncer."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fireworks.training.sdk.weight_syncer import WeightSyncer as _SdkWeightSyncer

logger = logging.getLogger(__name__)


def _is_reset_prompt_cache_unsupported(exc: httpx.HTTPStatusError) -> bool:
    response = exc.response
    if response.status_code != 400:
        return False
    parts = [str(exc)]
    text = getattr(response, "text", None)
    if text:
        parts.append(text)
    message = " ".join(parts).lower()
    return (
        "reset_prompt_cache" in message
        and "extra inputs are not permitted" in message
    )


class WeightSyncer(_SdkWeightSyncer):
    """SDK WeightSyncer with a fallback for older hotload payload schemas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_prompt_cache_supported = True

    def _hotload_once(
        self,
        snapshot_name: str,
        checkpoint_type: str,
        *,
        reset_prompt_cache: bool,
    ) -> None:
        incremental = self._build_incremental_metadata(checkpoint_type)
        t0 = time.time()
        if reset_prompt_cache:
            ok = self.deploy_mgr.hotload_and_wait(
                deployment_id=self.deployment_id,
                base_model=self.base_model,
                snapshot_identity=snapshot_name,
                incremental_snapshot_metadata=incremental,
                reset_prompt_cache=True,
                timeout_seconds=self.hotload_timeout,
            )
        else:
            ok = self._hotload_without_reset_prompt_cache(
                snapshot_name,
                incremental_snapshot_metadata=incremental,
            )
        self.last_timing["hotload_time_s"] = time.time() - t0
        if not ok:
            raise RuntimeError(
                f"Hotload failed for '{snapshot_name}': deployment did not accept snapshot. "
                f"Check deployment hotLoadBucketUrl and base model match."
            )

    def _hotload_without_reset_prompt_cache(
        self,
        snapshot_name: str,
        *,
        incremental_snapshot_metadata: dict[str, Any] | None,
    ) -> bool:
        headers = self.deploy_mgr._hotload_headers(self.deployment_id, self.base_model)
        url = f"{self.deploy_mgr.hotload_api_url}/hot_load/v1/models/hot_load"
        payload: dict[str, Any] = {"identity": snapshot_name}
        if incremental_snapshot_metadata:
            payload["incremental_snapshot_metadata"] = incremental_snapshot_metadata
        ckpt_type = "DELTA" if incremental_snapshot_metadata else "FULL"
        logger.info(
            "Hotloading %s snapshot '%s' to deployment '%s' without reset_prompt_cache",
            ckpt_type,
            snapshot_name,
            self.deployment_id,
        )
        resp = self.deploy_mgr._sync_request(
            url,
            method="POST",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return self.deploy_mgr.wait_for_hotload(
            deployment_id=self.deployment_id,
            base_model=self.base_model,
            expected_identity=snapshot_name,
            timeout_seconds=self.hotload_timeout,
        )

    def _do_hotload(self, snapshot_name: str, checkpoint_type: str) -> None:
        self._ensure_deployment_checked()
        requested_reset = self.reset_prompt_cache and self._reset_prompt_cache_supported
        try:
            self._hotload_once(
                snapshot_name,
                checkpoint_type,
                reset_prompt_cache=requested_reset,
            )
        except httpx.HTTPStatusError as exc:
            if not requested_reset or not _is_reset_prompt_cache_unsupported(exc):
                raise
            logger.info(
                "Hotload API rejected reset_prompt_cache; retrying without it"
            )
            self._reset_prompt_cache_supported = False
            self._hotload_once(
                snapshot_name,
                checkpoint_type,
                reset_prompt_cache=False,
            )

        self.base_identity = snapshot_name
        logger.info("Hotload complete: %s", snapshot_name)
        t1 = time.time()
        self._warmup_after_hotload()
        self.last_timing["warmup_time_s"] = time.time() - t1
