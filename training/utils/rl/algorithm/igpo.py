"""IGPO: Information Gain-based Policy Optimization.

Wang et al., "Information Gain-based Policy Optimization" (arXiv:2510.14967).
IGPO rewards every turn of a multi-turn trajectory. Each turn but the last
earns the information gain it produced: the change in
``log P(ground-truth answer | context)`` from the context before the turn to
the context after it. The last turn earns the outcome reward. Within a group
the two reward streams are z-normalized separately, discounted backward over
turns and broadcast to the tokens each turn generated.

IGPO is an advantage estimator, not a new loss: :func:`token_advantages`
sets ``PolicyBatch.token_advantages`` and any objective in this package trains
on the result. In ``recipes/rl_loop.py`` prepare the complete batch::

    rl_loop.main(cfg, rollout_fn_factory=..., prepare_batch_fn=igpo.prepare_batch(gamma=1.0))

Answer scoring is batched across groups; both reward normalizers remain
group-local. Weights stay fixed until the complete batch has trained.

A rollout describes each run with two ``RolloutRun.metadata`` keys:

- ``ANSWER_TOKENS``: the ground-truth answer tokens of the row;
- ``TURN_SPANS``: the ``[start, end)`` positions, in the run's full token
  sequence, of the tokens each turn generated, in order. The context before a
  turn is every token before its start.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Sequence, Tuple

import tinker

from training.utils.rl.losses import PolicyBatch, PromptGroup

ANSWER_TOKENS = "answer_tokens"
"""``RolloutRun.metadata`` key: ground-truth answer tokens scored for IG."""
TURN_SPANS = "turn_spans"
"""``RolloutRun.metadata`` key: ``[start, end)`` token span of each turn."""


# ---------------------------------------------------------------------------
# Information gain
# ---------------------------------------------------------------------------


def score_answer_logprobs(
    client: Any,
    contexts: Sequence[Sequence[int]],
    answers: Sequence[Sequence[int]],
) -> List[float]:
    """Mean ``log P(answer | context)`` for each pair, from one trainer forward."""
    if len(contexts) != len(answers):
        raise ValueError("IGPO scoring needs one answer per context.")
    if not contexts:
        return []
    datums = []
    for context, answer in zip(contexts, answers):
        if not context or not answer:
            raise ValueError("IGPO scoring needs a non-empty context and answer.")
        tokens = [*context, *answer]
        datums.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]
                    ),
                },
            )
        )
    result = client.forward(datums, "cross_entropy")
    if len(result.loss_fn_outputs) != len(datums):
        raise ValueError(
            f"IGPO scoring returned {len(result.loss_fn_outputs)} outputs for "
            f"{len(datums)} contexts."
        )
    scores = []
    for context, answer, output in zip(contexts, answers, result.loss_fn_outputs):
        start = len(context) - 1
        answer_logprobs = list(output["logprobs"].data)[start : start + len(answer)]
        scores.append(sum(answer_logprobs) / len(answer))
    return scores


def information_gain(context_scores: Sequence[float]) -> List[float]:
    """Per-turn IG from ``log P(answer | context)`` before each turn.

    Turn ``t`` earns ``score[t + 1] - score[t]``. The last turn earns 0: its
    outcome reward grades the answer it gives, and the context after it may
    already contain that answer.
    """
    if not context_scores:
        return []
    gains = [after - before for before, after in zip(context_scores, context_scores[1:])]
    return gains + [0.0]


# ---------------------------------------------------------------------------
# Turn-level advantages
# ---------------------------------------------------------------------------


def _z_normalize_group(values: List[float], eps: float = 1e-6) -> List[float]:
    """Z-normalize a list of values: (x - mean) / std."""
    n = len(values)
    if n <= 1:
        return [0.0] * n
    mu = sum(values) / n
    sigma = (sum((v - mu) ** 2 for v in values) / n) ** 0.5
    if sigma < eps:
        return [0.0] * n
    return [(v - mu) / (sigma + eps) for v in values]


def compute_turn_advantages(
    ig_rewards: List[List[float]],
    outcome_rewards: List[List[float]],
    gamma: float = 0.95,
    eps: float = 1e-6,
) -> List[List[float]]:
    """Per-turn advantages with separate z-normalization (matches IGPO paper).

    IG rewards and outcome rewards are z-normalized **independently** over
    every turn of the group, then summed. A backward discounted return is
    computed per rollout and broadcast to all tokens of each turn.

    Args:
        ig_rewards: Per-rollout, per-turn IG rewards (0 for turns without IG).
        outcome_rewards: Per-rollout, per-turn environment rewards (typically
            non-zero only at the last turn).
        gamma: Discount factor for backward return accumulation.
        eps: Numerical stability constant.

    Returns:
        Per-rollout, per-turn discounted advantages.
    """
    flat_ig: List[float] = []
    flat_outcome: List[float] = []
    for i, turns in enumerate(ig_rewards):
        outcome = outcome_rewards[i] if i < len(outcome_rewards) else []
        for t, reward in enumerate(turns):
            flat_ig.append(reward)
            flat_outcome.append(outcome[t] if t < len(outcome) else 0.0)
    combined = [
        ig + outcome
        for ig, outcome in zip(
            _z_normalize_group(flat_ig, eps), _z_normalize_group(flat_outcome, eps)
        )
    ]

    advantages: List[List[float]] = []
    offset = 0
    for turns in ig_rewards:
        rewards = combined[offset : offset + len(turns)]
        offset += len(turns)
        returns = [0.0] * len(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        advantages.append(returns)
    return advantages


def expand_turn_advantages(
    turn_advantages: Sequence[float],
    turn_spans: Sequence[Tuple[int, int]],
    num_targets: int,
) -> List[float]:
    """Broadcast each turn's advantage to the datum targets of its tokens.

    The token at full-sequence position ``p`` is datum target ``p - 1``;
    targets outside every span get advantage 0.
    """
    if len(turn_advantages) != len(turn_spans):
        raise ValueError("IGPO needs one advantage per turn span.")
    per_token = [0.0] * num_targets
    for advantage, (start, end) in zip(turn_advantages, turn_spans):
        for target in range(max(start - 1, 0), min(end - 1, num_targets)):
            per_token[target] = advantage
    return per_token


def _group_token_advantages(
    client: Any,
    groups: Sequence[PromptGroup],
    *,
    gamma: float,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Per-target-token IGPO advantages for every datum of ``groups``, and metrics."""
    runs = []
    contexts: List[List[int]] = []
    answers: List[List[int]] = []
    for group_index, group in enumerate(groups):
        if not (len(group.data) == len(group.run_metadata) == len(group.rewards)):
            raise ValueError(f"IGPO group {group_index} needs exactly one datum and reward per " "rollout run.")
        for datum, metadata in zip(group.data, group.run_metadata):
            spans = [(int(start), int(end)) for start, end in metadata.get(TURN_SPANS) or []]
            if not spans:
                raise ValueError(f"IGPO group {group_index} has a run without turns.")
            answer = list(metadata.get(ANSWER_TOKENS) or [])
            if not answer:
                raise ValueError(f"IGPO group {group_index} has a run without answer tokens.")
            targets = list(datum.loss_fn_inputs["target_tokens"].data)
            tokens = [*datum.model_input.to_ints(), targets[-1]]
            contexts.extend(tokens[:start] for start, _end in spans)
            answers.extend(answer for _ in spans)
            runs.append((spans, len(targets)))

    scores = score_answer_logprobs(client, contexts, answers)

    token_advantages: List[List[float]] = []
    run_index = score_offset = 0
    total_gain = baseline = 0.0
    gain_turns = 0
    for group in groups:
        group_runs = runs[run_index : run_index + len(group.data)]
        run_index += len(group.data)
        ig_rewards, outcome_rewards = [], []
        for (spans, _), reward in zip(group_runs, group.rewards):
            run_scores = scores[score_offset : score_offset + len(spans)]
            score_offset += len(spans)
            gains = information_gain(run_scores)
            ig_rewards.append(gains)
            outcome_rewards.append([0.0] * (len(spans) - 1) + [float(reward)])
            total_gain += sum(gains)
            gain_turns += len(spans) - 1
            baseline += run_scores[0]
        turn_advantages = compute_turn_advantages(ig_rewards, outcome_rewards, gamma=gamma)
        for (spans, num_targets), advantages in zip(group_runs, turn_advantages):
            token_advantages.append(expand_turn_advantages(advantages, spans, num_targets))

    metrics: Dict[str, Any] = {}
    if runs:
        metrics["igpo/avg_turns"] = sum(len(spans) for spans, _ in runs) / len(runs)
        metrics["igpo/mean_ig"] = total_gain / gain_turns if gain_turns else 0.0
        metrics["igpo/baseline_logp"] = baseline / len(runs)
    return token_advantages, metrics


def token_advantages(
    policy: Any, batch: PolicyBatch, *, gamma: float
) -> Tuple[PolicyBatch, Dict[str, float]]:
    """Return ``batch`` with IGPO ``token_advantages`` set, and ``igpo/*`` metrics.

    Each group of ``batch.groups`` holds one text datum per rollout run,
    described by ``run_metadata[i]`` keys ``ANSWER_TOKENS`` and ``TURN_SPANS``.
    One trainer forward on ``policy`` scores the answer after the context before
    every turn, so call this while the trainer holds the sampling policy.
    """
    rows, metrics = _group_token_advantages(policy, batch.groups, gamma=gamma)
    if len(rows) != len(batch.data):
        raise ValueError(f"IGPO produced {len(rows)} advantage rows for {len(batch.data)} datums.")
    return replace(batch, token_advantages=rows), metrics


def prepare_batch(*, gamma: float) -> Callable[[Any, PolicyBatch], Tuple[PolicyBatch, Dict[str, float]]]:
    """``rl_loop`` ``prepare_batch_fn`` that applies :func:`token_advantages`."""

    def prepare(policy: Any, batch: PolicyBatch) -> Tuple[PolicyBatch, Dict[str, float]]:
        return token_advantages(policy, batch, gamma=gamma)

    return prepare
