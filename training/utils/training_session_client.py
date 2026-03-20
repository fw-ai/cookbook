from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode

import requests
from fireworks.training.sdk.trainer import TrainerJobManager

DEFAULT_TIMEOUT_S = 600


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _wrap_forward_leaf(value: Any) -> Any:
    if isinstance(value, dict) and "data" in value:
        return SimpleNamespace(**{key: _wrap_forward_leaf(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap_forward_leaf(item) for item in value]
    if isinstance(value, dict):
        return {key: _wrap_forward_leaf(item) for key, item in value.items()}
    return value


def _to_forward_result(payload: dict[str, Any]) -> SimpleNamespace:
    loss_fn_outputs = [
        {key: _wrap_forward_leaf(value) for key, value in output.items()}
        for output in payload.get("loss_fn_outputs", [])
    ]
    result = dict(payload)
    result["loss_fn_outputs"] = loss_fn_outputs
    return SimpleNamespace(**result)


def _resource_id(name: str) -> str:
    return name.rstrip("/").split("/")[-1]


class TrainingSessionClient:
    """Cookbook client for shared reference sessions."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        manager: TrainerJobManager | None = None,
        request_session: requests.Session | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._manager = manager or TrainerJobManager(api_key=api_key, base_url=base_url)
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._http = request_session or requests.Session()
        self._http.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def create_session(self, base_model: str) -> TrainingSessionHandle:
        parent_job = self._resolve_or_create_parent_job(base_model)
        session = self._create_training_session(parent_job["name"])
        return TrainingSessionHandle(
            manager=self._manager,
            request_session=self._http,
            base_url=self._base_url,
            account_id=self._manager.account_id,
            job_id=_resource_id(parent_job["name"]),
            training_session_id=_resource_id(session["name"]),
            timeout_s=self._timeout_s,
        )

    def _resolve_or_create_parent_job(self, base_model: str) -> dict[str, Any]:
        query = urlencode(
            {
                "filter": f'base_model="{base_model}"',
                "orderBy": "create_time desc",
                "pageSize": 50,
            }
        )
        path = f"/v1/accounts/{self._manager.account_id}/trainingSessionJobs?{query}"
        response = self._manager._get(path, timeout=30)
        response.raise_for_status()
        jobs = response.json().get("trainingSessionJobs", [])
        if jobs:
            return jobs[0]

        create_path = f"/v1/accounts/{self._manager.account_id}/trainingSessionJobs"
        create_response = self._manager._post(create_path, json={"baseModel": base_model}, timeout=30)
        create_response.raise_for_status()
        return create_response.json()

    def _create_training_session(self, parent_name: str) -> dict[str, Any]:
        path = f"/v1/{parent_name}/trainingSessions"
        response = self._manager._post(path, json={}, timeout=30)
        response.raise_for_status()
        return response.json()


class TrainingSessionHandle:
    """Single-threaded, non-thread-safe handle for one training session."""

    def __init__(
        self,
        *,
        manager: TrainerJobManager,
        request_session: requests.Session,
        base_url: str,
        account_id: str,
        job_id: str,
        training_session_id: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._manager = manager
        self._http = request_session
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._job_id = job_id
        self._training_session_id = training_session_id
        self._timeout_s = timeout_s
        self._next_seq_id = 1
        self._usable = True

    def load_state(self, path: str) -> None:
        self._ensure_usable()
        response = self._manager._post(
            f"/v1/{self._session_name}:loadState",
            json={"name": self._session_name, "path": path},
            timeout=30,
        )
        response.raise_for_status()
        self._next_seq_id = 2

    def forward(self, data: Any, loss_fn: str) -> SimpleNamespace:
        self._ensure_usable()
        payload = {
            "forward_input": {
                "data": _to_jsonable(data),
                "loss_fn": loss_fn,
            },
            "seq_id": self._next_seq_id,
        }
        try:
            response = self._http.post(
                self._forward_url,
                json=payload,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            result = response.json()
        except requests.Timeout:
            self._usable = False
            raise

        self._next_seq_id += 1
        return _to_forward_result(result)

    @property
    def _session_name(self) -> str:
        return (
            f"accounts/{self._account_id}/trainingSessionJobs/{self._job_id}"
            f"/trainingSessions/{self._training_session_id}"
        )

    @property
    def _forward_url(self) -> str:
        return (
            f"{self._base_url}/training/v1/trainingSessionJobs/{self._account_id}/{self._job_id}"
            f"/trainingSessions/{self._training_session_id}/forward"
        )

    def _ensure_usable(self) -> None:
        if not self._usable:
            raise RuntimeError("TrainingSessionHandle is no longer usable after a failed forward()")
