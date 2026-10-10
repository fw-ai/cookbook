"""Prepare a fixed policy anchor without applying importance weights."""

from training.utils.timer import elapsed_timer


def validate_anchor(source: str) -> None:
    if source not in {"old_policy", "rollout"}:
        raise ValueError("anchor_logp must be 'old_policy' or 'rollout'")


def prepare_policy_anchor(policy, data, rollout_logprobs, source):
    """Return fixed anchor rows and a reusable decoded forward result.

    Rollout probabilities remain separate inputs to the loss. A snapshot only
    chooses the clipping/trust-region reference; it never preweights advantages.
    Trainer forward returns full-vocabulary logprobs. Callers requiring a
    support-normalized snapshot must reject that unsupported combination.
    """
    validate_anchor(source)
    if len(rollout_logprobs) != len(data) or any(not row for row in rollout_logprobs):
        raise ValueError("Policy loss requires one non-empty rollout_logprobs row per datum")
    if source == "rollout":
        return rollout_logprobs, None

    with elapsed_timer("old_policy_forward"):
        forward = policy.forward(data, "cross_entropy")
        if callable(getattr(forward, "result", None)):
            forward = forward.result()
    rows = [list(output["logprobs"].data) for output in forward.loss_fn_outputs]
    if len(rows) != len(data) or any(
        len(row) != len(datum.loss_fn_inputs["target_tokens"].data)
        for row, datum in zip(rows, data, strict=True)
    ):
        raise ValueError("Trainer anchor logprobs must align with target tokens")
    return rows, forward
