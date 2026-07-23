"""Harbor agent for exact, single-call HealthBench Professional inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fireworks import AsyncFireworks
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .trajectory import (
    TRAJECTORY_FILENAME,
    TrajectoryContractError,
    build_atif_trajectory,
    extract_fireworks_response,
    validate_sampling_settings,
    write_atif_trajectory,
)


INSTRUCTION_SCHEMA_VERSION = 1
VISIBLE_RESPONSE_FILENAME = "response.txt"
ANSWER_PATH = "/workspace/answer.txt"
REQUEST_TIMEOUT_SECONDS = 1_800.0
REQUEST_MAX_RETRIES = 0


def normalize_fireworks_model(model_name: str) -> str:
    """Remove LiteLLM's optional Fireworks provider prefix."""

    if not isinstance(model_name, str):
        raise ValueError("model_name must be text")
    normalized = model_name.removeprefix("fireworks_ai/")
    if not normalized:
        raise ValueError("model_name must not be empty")
    return normalized


def parse_instruction_messages(instruction: str) -> list[dict[str, Any]]:
    """Read the generated task envelope without changing its messages."""

    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError as exc:
        raise TrajectoryContractError(
            "HealthBench instruction must be a JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise TrajectoryContractError("HealthBench instruction must be a JSON object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != INSTRUCTION_SCHEMA_VERSION:
        raise TrajectoryContractError(
            f"instruction schema_version must be {INSTRUCTION_SCHEMA_VERSION}"
        )
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TrajectoryContractError("instruction messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TrajectoryContractError(f"messages[{index}] must be an object")
        if not isinstance(message.get("role"), str):
            raise TrajectoryContractError(f"messages[{index}].role must be text")
        if "content" not in message:
            raise TrajectoryContractError(f"messages[{index}].content is required")
        if not isinstance(message["content"], str):
            raise TrajectoryContractError(f"messages[{index}].content must be text")
    return messages


class HealthBenchProfessionalAgent(BaseAgent):
    """Call one Fireworks chat deployment without a benchmark prompt wrapper."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 16_384,
        request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        require_token_trajectory: bool = True,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if model_name is None:
            raise ValueError("model_name is required")
        if require_token_trajectory is not True:
            raise ValueError(
                "require_token_trajectory must remain true for RL-safe evaluation"
            )
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or float(request_timeout_seconds) != REQUEST_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"request_timeout_seconds must remain {REQUEST_TIMEOUT_SECONDS:g}"
            )
        normalized_model = normalize_fireworks_model(model_name)
        super().__init__(
            logs_dir=logs_dir,
            model_name=normalized_model,
            **kwargs,
        )
        self._model = normalized_model
        self._settings = validate_sampling_settings(
            {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "raw_output": True,
                "return_token_ids": True,
                "logprobs": True,
                "context_length_exceeded_behavior": "error",
                "n": 1,
                "stream": False,
            }
        )
        self._require_token_trajectory = True
        if client is None:
            client_kwargs: dict[str, Any] = {}
            resolved_api_key = api_key or self.extra_env.get("FIREWORKS_API_KEY")
            resolved_base_url = base_url or self.extra_env.get("FIREWORKS_BASE_URL")
            if resolved_api_key is not None:
                client_kwargs["api_key"] = resolved_api_key
            if resolved_base_url is not None:
                client_kwargs["base_url"] = resolved_base_url
            # HealthBench permits 16K-token reasoning responses. The SDK's
            # default is a 60-second read timeout with two retries, which turns
            # healthy long generations into three guaranteed failures. Use one
            # bounded 30-minute request and leave retry policy to the harness.
            client_kwargs["timeout"] = REQUEST_TIMEOUT_SECONDS
            client_kwargs["max_retries"] = REQUEST_MAX_RETRIES
            client = AsyncFireworks(**client_kwargs)
        self._client = client

    @staticmethod
    def name() -> str:
        return "healthbench-professional"

    def version(self) -> str:
        return "0.1.1"

    @property
    def sampling_settings(self) -> dict[str, Any]:
        return dict(self._settings)

    async def setup(self, environment: BaseEnvironment) -> None:
        """The agent uses the host-side Fireworks SDK and needs no sandbox setup."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        messages = parse_instruction_messages(instruction)

        # Keep this argument list explicit. In particular, do not add an agent
        # prompt, top_k, echo, thinking, or reasoning_effort.
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._settings["temperature"],
            top_p=self._settings["top_p"],
            max_tokens=self._settings["max_tokens"],
            raw_output=True,
            return_token_ids=True,
            logprobs=True,
            context_length_exceeded_behavior="error",
            n=1,
            stream=False,
        )
        exact = extract_fireworks_response(
            response,
            model=self._model,
            settings=self._settings,
        )

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        response_path = self.logs_dir / VISIBLE_RESPONSE_FILENAME
        response_path.write_text(exact.visible_text, encoding="utf-8")

        trajectory = build_atif_trajectory(
            exact,
            instruction=instruction,
            messages=messages,
            agent_name=self.name(),
            agent_version=self.version(),
            session_id=self.session_id,
        )
        write_atif_trajectory(self.logs_dir / TRAJECTORY_FILENAME, trajectory)

        context.n_input_tokens = len(exact.prompt_token_ids)
        context.n_output_tokens = len(exact.completion_token_ids)
        context.rollout_details = [exact.rollout_detail()]
        request_id = (
            response.get("id")
            if isinstance(response, Mapping)
            else getattr(response, "id", None)
        )
        context.metadata = {
            "finish_reason": exact.finish_reason,
            "model": exact.model,
            "provider_model": exact.provider_model,
            "reasoning_present": bool(exact.reasoning_content),
            "request_id": request_id,
            "require_token_trajectory": self._require_token_trajectory,
            "settings": dict(exact.settings),
        }

        # The verifier reads only the visible assistant answer. Raw completion
        # text remains in the trajectory for observability and RL export.
        await environment.upload_file(response_path, ANSWER_PATH)


__all__ = [
    "ANSWER_PATH",
    "HealthBenchProfessionalAgent",
    "INSTRUCTION_SCHEMA_VERSION",
    "REQUEST_MAX_RETRIES",
    "REQUEST_TIMEOUT_SECONDS",
    "VISIBLE_RESPONSE_FILENAME",
    "normalize_fireworks_model",
    "parse_instruction_messages",
]
