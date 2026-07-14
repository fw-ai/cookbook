from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor
from eval_protocol.rewards.math import math_reward


@evaluation_test(
    input_dataset=["gsm8k_train.jsonl"],
    # completion_params must be present so managed RFT can inject the policy model into it at
    # rollout time (do NOT hardcode "model" — the trainer supplies the model being optimized).
    completion_params=[{"max_tokens": 512, "temperature": 0.7}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def evaluate(row: EvaluationRow) -> EvaluationRow:
    """Verifiable math reward: 1.0 if the final <answer> matches ground truth, else 0.0."""
    result = math_reward(messages=row.messages, ground_truth=row.ground_truth)
    row.evaluation_result = EvaluateResult(score=result.score, reason=result.reason)
    return row
