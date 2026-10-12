from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

log = logging.getLogger("deployment_creation")


class FireworksClient:
    """Thin REST wrapper around the deployment REST endpoints used here.

    Raw REST is used rather than the public `fireworks` SDK because the SDK
    does not expose the `replaceMergedAddon` query parameter, which is the
    whole point of the hot-reload path.
    """

    def __init__(self, account_id: str, api_key: str, api_base: str):
        self.account_id = account_id
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def deployment_name(self, deployment_id: str) -> str:
        return f"accounts/{self.account_id}/deployments/{deployment_id}"

    def create_deployment(
        self, body: dict[str, Any], *, deployment_id: str | None = None
    ) -> dict[str, Any]:
        # NOTE: `deploymentId` is a *query* parameter on CreateDeployment, not
        # a body field. Putting it in the body is rejected by the API.
        params: dict[str, str] = {}
        if deployment_id:
            params["deploymentId"] = deployment_id
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployments"
        r = self._client.post(url, params=params, json=body)
        r.raise_for_status()
        return r.json()

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployments/{deployment_id}"
        r = self._client.get(url)
        r.raise_for_status()
        return r.json()

    def delete_deployment(self, deployment_id: str, *, ignore_checks: bool = False) -> None:
        # A plain DELETE can be rejected by the API (e.g. while the deployment
        # is still in use). Passing ?ignoreChecks=true forces it, so we retry
        # with that on failure to avoid leaking live deployments.
        url = f"{self.api_base}/v1/accounts/{self.account_id}/deployments/{deployment_id}"
        r = self._client.delete(url, params={"ignoreChecks": "true"} if ignore_checks else None)
        if r.status_code == 404:
            # Already gone — treat as success.
            return
        if not r.is_success and not ignore_checks:
            log.info(
                "plain delete failed (%s); retrying with ?ignoreChecks=true", r.status_code
            )
            r2 = self._client.delete(url, params={"ignoreChecks": "true"})
            if r2.status_code == 404:
                return
            r2.raise_for_status()
            return
        r.raise_for_status()


    def close(self) -> None:
        self._client.close()
        

def create_deployment(account_id, api_key, deployment_id, base_model, deployment_shape) -> str:
    client = FireworksClient(account_id, api_key, "https://api.fireworks.ai")
    body: dict[str, Any] = {
        "baseModel": base_model,
        "deploymentShape": deployment_shape,
        "enableHotReloadLatestAddon": True,
        "enableHotLoad": True,
        "hotLoadBucketType": "FW_HOSTED",
    }
    resp = client.create_deployment(body, deployment_id=deployment_id)
    return resp.get("name", "")