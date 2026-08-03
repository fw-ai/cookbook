
import os
from eval_protocol import evaluation_test
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.remote_rollout_processor import RemoteRolloutProcessor
from sql_reward import compute_sql_reward

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "accounts/fireworks/models/minimax-m3")
_FW_KEY = os.getenv("FIREWORKS_API_KEY", "")


@evaluation_test(
    input_dataset=["text2sql_train.jsonl"],
    completion_params=[{"temperature": 1.0, "max_tokens": 16000}],
    # A full multi-turn text-to-SQL episode (schema + evidence + SQL + judge) can exceed the 120s
    # default, so give each remote rollout more headroom.
    rollout_processor=RemoteRolloutProcessor(remote_base_url="https://8969c3cf0f4f.ngrok.app", timeout_seconds=600),
    mode="pointwise",
)
async def evaluate(row: EvaluationRow) -> EvaluationRow:
    # row.messages is the trajectory the remote server produced (traced through the trainer gateway).
    return await compute_sql_reward(row, judge_model=JUDGE_MODEL, api_key=_FW_KEY, use_llm_judge=True)
