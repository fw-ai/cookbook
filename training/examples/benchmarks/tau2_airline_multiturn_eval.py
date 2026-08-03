"""Tau2-style airline eval harness (single .py, no notebook indirection).

This file is intentionally explicit:
- Uses the official airline dataset URL
- Requires a tau2-compatible MCP Gym server script
- Fails fast with clear errors if prerequisites are missing
"""

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from eval_protocol import evaluation_test
from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message

try:
    from eval_protocol.pytest import MCPGymRolloutProcessor as _RolloutProcessor
except Exception:
    from eval_protocol.pytest import default_mcp_gym_rollout_processor as _RolloutProcessor

from vendor.tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from vendor.tau2.data_model.tasks import EvaluationCriteria, RewardType, Task, UserScenario
from vendor.tau2.evaluator.evaluator import EnvironmentEvaluator
from vendor.tau2.evaluator.evaluator_action import ActionEvaluator
from vendor.tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from vendor.tau2.registry import registry
import vendor.tau2.domains.airline.data_model as tau2_airline_data_model
import vendor.tau2.domains.airline.environment as tau2_airline_environment
import vendor.tau2.domains.airline.tools as tau2_airline_tools
import vendor.tau2.domains.airline.utils as tau2_airline_utils


# ---- Config ----
MODEL = "fireworks_ai/accounts/fireworks/models/glm-5p2"
SIM_USER_LLM = "fireworks_ai/accounts/fireworks/models/glm-5p2"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
REQUEST_TIMEOUT_S = 120
ROLLOUT_STEPS = 12
MAX_CONCURRENT_ROLLOUTS = 2
PASSED_THRESHOLD = 0.4
MAX_DATASET_ROWS = 3  # smoke-test default; raise for full benchmark runs
ENABLE_NL_ASSERTIONS = False  # vendor tau2 evaluator has async bug in this env

AIRLINE_DATASET_URL = (
    "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
    "1bd5447a3afbca3b71e0f0d205ed7cff6c3afe5d/"
    "eval_protocol/benchmarks/data/airline_dataset.jsonl"
)

TRAINING_DIR = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if p.name == "training" and (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parents[2],
)

DATASET_PATH = TRAINING_DIR / "examples/benchmarks/data/airline_dataset.jsonl"
PREPARED_DATASET_PATH = TRAINING_DIR / "examples/benchmarks/data/airline_dataset_fireworks.jsonl"
SYSTEM_PROMPT_PATH = TRAINING_DIR / "examples/tau2_mcp/tests/system_prompts/airline_agent_system_prompt.md"
LOCAL_AIRLINE_DB_PATH = TRAINING_DIR / "examples/tau2_mcp/airplane_environment/db.json"

# IMPORTANT: this must be a tau2 MCP Gym-compatible server (supports /control/reset_session).
SERVER_SCRIPT_PATH = TRAINING_DIR / "examples/tau2_mcp/server.py"


def _ensure_dataset() -> None:
    if DATASET_PATH.exists():
        return
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(AIRLINE_DATASET_URL, DATASET_PATH)


def _prepare_dataset_for_fireworks() -> None:
    """Rewrite user simulator config to use Fireworks only."""
    rows = _load_jsonl(DATASET_PATH)
    PREPARED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREPARED_DATASET_PATH.open("w", encoding="utf-8") as f:
        for item in rows:
            user_sim = dict(item.get("user_simulation", {}))
            if user_sim.get("enabled", False):
                user_sim["llm"] = SIM_USER_LLM
                user_sim["llm_args"] = {
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "timeout": REQUEST_TIMEOUT_S,
                }  # keep simulator calls bounded
            item["user_simulation"] = user_sim
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def _ensure_server() -> None:
    if not SERVER_SCRIPT_PATH.exists():
        raise RuntimeError(
            "Tau2 server script not found.\n"
            f"Expected: {SERVER_SCRIPT_PATH}\n\n"
            "This repo currently does not include a tau2 MCP Gym airline server.\n"
            "Add/port the tutorial-compatible server first (it must support MCP Gym control endpoints)."
        )
    if not SYSTEM_PROMPT_PATH.exists():
        raise RuntimeError(f"Missing airline system prompt: {SYSTEM_PROMPT_PATH}")
    if not LOCAL_AIRLINE_DB_PATH.exists():
        raise RuntimeError(f"Missing local airline DB: {LOCAL_AIRLINE_DB_PATH}")


def _force_local_airline_db() -> None:
    """Make tau2 evaluators use the local vendored airline DB file."""
    tau2_airline_utils.AIRLINE_DB_PATH = LOCAL_AIRLINE_DB_PATH
    tau2_airline_data_model.AIRLINE_DB_PATH = LOCAL_AIRLINE_DB_PATH
    tau2_airline_environment.AIRLINE_DB_PATH = LOCAL_AIRLINE_DB_PATH
    tau2_airline_tools.AIRLINE_DB_PATH = LOCAL_AIRLINE_DB_PATH


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tau_bench_airline_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows: List[EvaluationRow] = []
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    for i, item in enumerate(data):
        user_simulation = dict(item.get("user_simulation", {}))
        if user_simulation.get("enabled", False):
            # Avoid hard dependency on OPENAI_API_KEY from dataset defaults.
            user_simulation["llm"] = SIM_USER_LLM
            user_simulation.setdefault("llm_args", {"temperature": 0.0})

        rows.append(
            EvaluationRow(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ],
                input_metadata=InputMetadata(
                    row_id=item.get("id", f"row-{i}"),
                    dataset_info={
                        "environment_context": item.get("environment_context", {}),
                        "user_simulation": user_simulation,
                        "evaluation_criteria": item.get("evaluation_criteria", {}),
                        "user_prompt_template": item.get("user_prompt_template", "{observation}"),
                    },
                ),
            )
        )
    return rows


_ensure_dataset()
_ensure_server()
_prepare_dataset_for_fireworks()
_force_local_airline_db()


@evaluation_test(
    input_dataset=[str(PREPARED_DATASET_PATH)],
    dataset_adapter=tau_bench_airline_to_evaluation_row,
    completion_params=[
        {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout": REQUEST_TIMEOUT_S,
        }
    ],
    rollout_processor=_RolloutProcessor() if callable(_RolloutProcessor) else _RolloutProcessor,
    passed_threshold=PASSED_THRESHOLD,
    num_runs=1,
    mode="pointwise",
    max_dataset_rows=MAX_DATASET_ROWS,
    max_concurrent_rollouts=MAX_CONCURRENT_ROLLOUTS,
    steps=ROLLOUT_STEPS,
    server_script_path=str(SERVER_SCRIPT_PATH),
)
def test_tau2_airline(row: EvaluationRow) -> EvaluationRow:
    dataset_info = row.input_metadata.dataset_info if row.input_metadata else {}
    raw_eval_criteria = dataset_info.get("evaluation_criteria", {})

    nl_assertions = raw_eval_criteria.get("nl_assertions", [])
    communicate_info = raw_eval_criteria.get("communicate_info", [])
    actions = raw_eval_criteria.get("actions", [])

    trajectory_objects = []
    for msg in row.messages:
        role = msg.role
        content = msg.content

        if role == "system":
            trajectory_objects.append(SystemMessage(role=role, content=content))
        elif role == "assistant":
            tau2_tool_calls = []
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    tau2_tool_calls.append(
                        ToolCall(id=tool_call.id, name=tool_call.function.name, arguments=args)
                    )
            trajectory_objects.append(
                AssistantMessage(role=role, content=content, tool_calls=tau2_tool_calls)
            )
        elif role == "user":
            trajectory_objects.append(UserMessage(role=role, content=content))
        elif role == "tool":
            trajectory_objects.append(ToolMessage(id=msg.tool_call_id, role=role, content=content))

    reward_basis = [
        RewardType.DB,
        RewardType.ACTION,
        RewardType.COMMUNICATE,
    ]
    if ENABLE_NL_ASSERTIONS:
        reward_basis.append(RewardType.NL_ASSERTION)

    task = Task(
        id=str(dataset_info.get("row_id", "tau2-airline-task")),
        description=None,
        user_scenario=UserScenario(instructions="tau2-user", persona=None),
        ticket=None,
        initial_state=None,
        evaluation_criteria=EvaluationCriteria(
            nl_assertions=nl_assertions,
            communicate_info=communicate_info,
            actions=actions,
            reward_basis=reward_basis,
        ),
    )

    env_reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=registry.get_env_constructor("airline"),
        task=task,
        full_trajectory=trajectory_objects,
    )
    action_reward_info = ActionEvaluator.calculate_reward(task=task, full_trajectory=trajectory_objects)
    communicate_reward_info = CommunicateEvaluator.calculate_reward(
        task=task, full_trajectory=trajectory_objects
    )
    reward = env_reward_info.reward * action_reward_info.reward * communicate_reward_info.reward
    failed = []
    if env_reward_info.reward == 0:
        failed.append("env/db")
    if action_reward_info.reward == 0:
        failed.append("actions")
    if communicate_reward_info.reward == 0:
        failed.append("communicate")
    if ENABLE_NL_ASSERTIONS:
        failed.append("nl_assertions(not-run-in-smoke)")

    row.evaluation_result = EvaluateResult(
        score=reward,
        reason="OK" if not failed else f"failed: {', '.join(failed)}",
    )
    row_id = "unknown"
    if row.input_metadata and row.input_metadata.row_id:
        row_id = row.input_metadata.row_id
    print(
        f"[tau2][{row_id}] "
        f"env={env_reward_info.reward:.0f} "
        f"action={action_reward_info.reward:.0f} "
        f"communicate={communicate_reward_info.reward:.0f} "
        f"final={reward:.0f} "
        f"reason={row.evaluation_result.reason}"
    )
    return row

