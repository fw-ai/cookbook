"""On-policy distillation datums and losses (online OPD / OPSD).

All objectives here train on *student* rollouts. The focus is the ONLINE path
(``opd_loop.py``); offline top-K KL distillation is intentionally deferred --
the co-located LoRA ``kl_distillation`` loss already covers the offline case
(see "Offline" note below).

Modes
-----
    DistillMode        KL direction          source-K model   student training pass
    -----------------  --------------------  ---------------  -----------------------------
    SAMPLED_IS         reverse KL (1-sample) n/a (no top-K)   builtin ``importance_sampling``
    TOPK_REVERSE_KL    reverse  KL(S || T)   STUDENT          ``forward_backward_custom``
    TOPK_FORWARD_KL    forward  KL(T || S)   TEACHER          builtin ``cross_entropy`` [N,K]

* ``SAMPLED_IS`` (ships in #391): per-token advantage = ``teacher_logprob -
  sampling_logprob`` on the sampled token, fed to ``importance_sampling``
  (``-exp(lp - slp) * advantage``). A 1-sample Monte-Carlo estimate of reverse
  KL ``KL(pi_S || pi_T)`` -- correct direction, NO top-K. Both logprobs come
  from INFERENCE (student sampling + teacher-deployment scoring).
* ``TOPK_REVERSE_KL`` (``reverse_kl_topk_loss``): analytic reverse KL over the
  top-K. Reverse KL's expectation is over ``pi_S``, so the source K (the K-token
  support) is the **STUDENT's** top-K. Recommended OPSD default (mode-seeking,
  matches the literature).
* ``TOPK_FORWARD_KL``: cross-entropy against the teacher's renormalized top-K.
  Forward KL's expectation is over ``pi_T``, so the source K is the **TEACHER's**
  top-K. Mass-covering; opt-in for SDFT-style continual-learning.

Source-K vs gather (the key rule -- see also opd_sampling.py)
------------------------------------------------------------
"source K" = the FIRST call that *selects* the K token indices (+ that model's
logprobs): ``forward(loss_fn_config={"top_k": K})`` (trainer, ``@no_grad``) or
inference ``top_logprobs``. "gather" = the SECOND stage that reads logprobs AT
those fixed indices via ``target_tokens=[N,K]`` (PR #27269). The other model and
the training ``forward_backward`` MUST gather at the SAME source-K indices.
NEVER run a second ``topK`` selection at stage 2 -- it would pick a *different*
K set and silently compute per-token KL on mismatched tokens.

OPSD extraction provenance (TOPK_REVERSE_KL, source = student)
--------------------------------------------------------------
For correctness the student source K is obtained by a trainer-side forward-only
extraction run AFTER inference, over the sampled rollout tokens -- NOT from the
inference ``top_logprobs``. Reasons: inference runs quantized weights + different
kernels + a truncated/renormalized sampling nucleus (capped at top_logprobs<=5),
i.e. the train/inference gap; reverse KL needs the student's true training-time
distribution. Flow:
    1. inference samples rollouts                -> on-policy TOKEN SEQUENCES
    2. trainer forward(top_k=K) over sequences   -> source K = student top-K indices (@no_grad)
    3. gather TEACHER at those indices           -> target logprobs (q)
    4. student forward_backward at those indices -> student logprobs WITH grad -> loss
Inference ``top_logprobs`` may be used as a cheap approximation (skips step 2)
but bakes in the train/inference gap.

Custom-loss path is two-pass
----------------------------
``forward_backward_custom`` does NOT avoid forward: pass 1 is a server ``forward``
returning student logprobs at ``target_tokens`` (re-leafed with ``requires_grad``
on the client); the client loss yields ``dC/dlogprobs``; pass 2 backprops a CE
surrogate with ``weights = -dC/dlogprobs``. The framework errors if the loss has
no gradient w.r.t. logprobs, so the detached ``forward(top_k)`` extraction can
never silently "train".

Masking (lives in the RLOR backend, NOT here)
---------------------------------------------
Sampling-nucleus (top-p/top-k) masking is a BACKEND concern: the engine owns the
sampling params and must **gather the top-K after masking** -- select/gather the
top-K over the post-mask nucleus distribution (and gather the other model at
those ``[N, K]`` indices). The cookbook consumes already-masked top-K; it does
NOT reconstruct the nucleus from inference ``top_logprobs`` (lossy: cap <=5,
token-string ambiguity, quant/kernel gap). The only client-side mask here is the
``weights > 0`` slot mask in ``reverse_kl_topk_loss`` -- that is datum *shaping*
(padding / variable-K), not sampling masking.

Offline (deferred)
------------------
Offline top-K KL (teacher = separate frozen model, or teacher top-K stored in
the dataset) is NOT built here -- the co-located full-vocab LoRA
``kl_distillation`` loss (``train/nn/kl_distillation.py``) already covers offline
distillation, and is exact (full logits via the shared lm_head, no top-K
approximation). ``teacher_topk_from_row`` remains as a thin hook for that future
path but is not wired into the online loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence

import torch
import tinker

from training.utils.opd_sampling import TopKDist


class DistillMode(str, Enum):
    """OPSD distillation objective. See module docstring for trade-offs."""

    SAMPLED_IS = "sampled_is"
    TOPK_FORWARD_KL = "topk_forward_kl"
    TOPK_REVERSE_KL = "topk_reverse_kl"

    @property
    def needs_teacher_topk(self) -> bool:
        return self in (DistillMode.TOPK_FORWARD_KL, DistillMode.TOPK_REVERSE_KL)

    @property
    def uses_custom_loss(self) -> bool:
        """True when training must go through ``forward_backward_custom``."""
        return self is DistillMode.TOPK_REVERSE_KL


@dataclass
class OPDPromptGroup:
    """Processed rollouts for one prompt in sampled-token OPD.

    This mirrors the small part of ``PromptGroup`` needed by the shared async
    runner, but names the OPD-specific tensors directly instead of treating
    teacher logprobs as RLHF reference logprobs.
    """

    data: list[tinker.Datum]
    teacher_logprobs: list[list[float]]
    sampling_logprobs: list[list[float]]
    prompt_len: int
    rewards: list[float]
    completion_lens: list[int] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class OPDInputMetrics:
    """Summary metrics computed while building OPD server-side datums."""

    active_tokens: int
    sampled_reverse_kl_sum: float
    opd_advantage_sum: float
    opd_abs_advantage_sum: float
    teacher_nll_sum: float
    sampling_nll_sum: float

    def as_dict(self) -> dict[str, float]:
        denom = max(self.active_tokens, 1)
        sampled_reverse_kl = self.sampled_reverse_kl_sum / denom
        opd_advantage = self.opd_advantage_sum / denom
        abs_logprob_gap = self.opd_abs_advantage_sum / denom
        return {
            "opd_active_tokens": float(self.active_tokens),
            "opd_sampled_reverse_kl": sampled_reverse_kl,
            "opd_advantage": opd_advantage,
            "opd_abs_advantage": abs_logprob_gap,
            "opd_student_logprob_minus_teacher_logprob": sampled_reverse_kl,
            "opd_teacher_logprob_minus_student_logprob": opd_advantage,
            "opd_abs_logprob_gap": abs_logprob_gap,
            "opd_teacher_nll": self.teacher_nll_sum / denom,
            "opd_sampling_nll": self.sampling_nll_sum / denom,
        }


def _pad_or_trim(values: Sequence[float], length: int) -> list[float]:
    result = [float(v) for v in values[:length]]
    if len(result) < length:
        result.extend([0.0] * (length - len(result)))
    return result


def _loss_mask_for_datum(datum: tinker.Datum, length: int) -> list[float]:
    mask = datum.loss_fn_inputs.get("loss_mask")
    if mask is None:
        return [1.0] * length
    return _pad_or_trim(mask.data, length)


def _require_lengths_match(name: str, values: Iterable[object], expected: int) -> list[object]:
    result = list(values)
    if len(result) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(result)}.")
    return result


def _require_active_logprobs(
    name: str,
    values: Sequence[float],
    active_positions: Sequence[int],
    *,
    datum_idx: int,
) -> None:
    if not active_positions:
        return
    required_len = max(active_positions) + 1
    if len(values) < required_len:
        raise ValueError(
            f"Datum {datum_idx}: {name} has length {len(values)}, "
            f"but active OPD tokens require at least {required_len} logprobs."
        )


def combine_opd_prompt_groups(
    groups: Sequence[OPDPromptGroup],
) -> tuple[list[tinker.Datum], list[list[float]], list[int], list[list[float]]]:
    """Flatten OPD prompt groups into arrays for one server-side loss call."""
    data: list[tinker.Datum] = []
    teacher_logprobs: list[list[float]] = []
    prompt_lens: list[int] = []
    sampling_logprobs: list[list[float]] = []

    for group_idx, group in enumerate(groups):
        n = len(group.data)
        if len(group.teacher_logprobs) != n:
            raise ValueError(
                f"Group {group_idx}: teacher_logprobs length ({len(group.teacher_logprobs)}) "
                f"does not match data length ({n})."
            )
        if len(group.sampling_logprobs) != n:
            raise ValueError(
                f"Group {group_idx}: sampling_logprobs length ({len(group.sampling_logprobs)}) "
                f"does not match data length ({n})."
            )

        data.extend(group.data)
        teacher_logprobs.extend(group.teacher_logprobs)
        prompt_lens.extend([group.prompt_len] * n)
        sampling_logprobs.extend(group.sampling_logprobs)

    return data, teacher_logprobs, prompt_lens, sampling_logprobs


def build_opd_server_datums(
    data: Sequence[tinker.Datum],
    teacher_logprobs: Sequence[Sequence[float]],
    sampling_logprobs: Sequence[Sequence[float]],
    prompt_lens: Sequence[int],
    *,
    loss_scale: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build datums for Tinker's server-side ``importance_sampling`` loss.

    Args:
        data: Training datums containing ``target_tokens`` and model inputs.
        teacher_logprobs: Teacher logprobs aligned to ``target_tokens``.
        sampling_logprobs: Student rollout logprobs aligned to ``target_tokens``.
        prompt_lens: Full prompt token count per datum.  The first response
            token has logprob index ``prompt_len - 1``.
        loss_scale: Optional scalar multiplier for the OPD dense reward.

    Returns:
        ``(server_datums, metrics)``.  ``server_datums`` contain exactly the
        fields required by Tinker's built-in RL losses:
        ``target_tokens``, ``logprobs`` (sampling logprobs), and
        ``advantages`` (teacher minus sampling on response tokens).
    """
    n = len(data)
    teacher_logprobs = _require_lengths_match("teacher_logprobs", teacher_logprobs, n)
    sampling_logprobs = _require_lengths_match("sampling_logprobs", sampling_logprobs, n)
    prompt_lens = _require_lengths_match("prompt_lens", prompt_lens, n)

    server_datums: list[tinker.Datum] = []
    active_tokens = 0
    sampled_reverse_kl_sum = 0.0
    opd_advantage_sum = 0.0
    opd_abs_advantage_sum = 0.0
    teacher_nll_sum = 0.0
    sampling_nll_sum = 0.0

    for idx, datum in enumerate(data):
        target_data = datum.loss_fn_inputs.get("target_tokens")
        if target_data is None:
            raise ValueError(f"Datum {idx} is missing loss_fn_inputs['target_tokens'].")

        target_tokens = list(target_data.data)
        target_len = len(target_tokens)
        response_start = max(0, int(prompt_lens[idx]) - 1)
        raw_teacher_lp = [float(v) for v in teacher_logprobs[idx]]
        raw_sampling_lp = [float(v) for v in sampling_logprobs[idx]]
        loss_mask = _loss_mask_for_datum(datum, target_len)
        active_positions = [
            pos for pos in range(response_start, target_len) if loss_mask[pos] > 0.0
        ]
        _require_active_logprobs(
            "teacher_logprobs",
            raw_teacher_lp,
            active_positions,
            datum_idx=idx,
        )
        _require_active_logprobs(
            "sampling_logprobs",
            raw_sampling_lp,
            active_positions,
            datum_idx=idx,
        )
        teacher_lp = _pad_or_trim(raw_teacher_lp, target_len)
        sampling_lp = _pad_or_trim(raw_sampling_lp, target_len)

        advantages = [0.0] * target_len
        for pos in range(response_start, target_len):
            if loss_mask[pos] <= 0.0:
                continue
            advantage = (teacher_lp[pos] - sampling_lp[pos]) * loss_scale * loss_mask[pos]
            advantages[pos] = advantage
            active_tokens += 1
            sampled_reverse_kl_sum += (sampling_lp[pos] - teacher_lp[pos]) * loss_mask[pos]
            opd_advantage_sum += advantage
            opd_abs_advantage_sum += abs(advantage)
            teacher_nll_sum += -teacher_lp[pos] * loss_mask[pos]
            sampling_nll_sum += -sampling_lp[pos] * loss_mask[pos]

        server_datums.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[target_len],
                    ),
                    "logprobs": tinker.TensorData(
                        data=sampling_lp,
                        dtype="float32",
                        shape=[target_len],
                    ),
                    "advantages": tinker.TensorData(
                        data=advantages,
                        dtype="float32",
                        shape=[target_len],
                    ),
                },
            )
        )

    metrics = OPDInputMetrics(
        active_tokens=active_tokens,
        sampled_reverse_kl_sum=sampled_reverse_kl_sum,
        opd_advantage_sum=opd_advantage_sum,
        opd_abs_advantage_sum=opd_abs_advantage_sum,
        teacher_nll_sum=teacher_nll_sum,
        sampling_nll_sum=sampling_nll_sum,
    )
    return server_datums, metrics.as_dict()


# ---------------------------------------------------------------------------
# Multi-teacher configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherConfig:
    """One teacher in a multi-target OPD run.

    Args:
        model: Inference model id (or base-model resource to auto-deploy).
        deployment_id: Optional explicit frozen-teacher deployment id.
        teacher_messages_key: Dataset key holding this teacher's privileged
            prompt messages (falls back to the row's own messages).
        top_logprobs: Per-teacher override of ``K``; ``None`` uses the run value.
    """

    model: str
    deployment_id: str | None = None
    teacher_messages_key: str = "teacher_messages"
    top_logprobs: int | None = None


@dataclass
class MultiTeacherConfig:
    """Multi-TARGET routing: one student, N frozen teachers, ROUTED per prompt.

    This is NOT a per-prompt mixture/blend (that would be N x teacher cost for a
    single target). Each prompt is scored by exactly ONE teacher, chosen by the
    row's ``route_key`` value (which must equal one teacher's ``model``). The
    student samples from its single deployment; routing different prompts to
    different teacher deployments lets the async sampling window interleave
    scoring across teachers so every deployment's GPU stays busy.

    (Weighted mixture-of-teachers on the SAME prompt is intentionally not
    supported online; ``blend_teacher_topk`` remains in utils for the future
    offline top-K path only.)
    """

    teachers: list[TeacherConfig] = field(default_factory=list)
    route_key: str = "teacher"  # row key whose value names the teacher ``model``

    def __post_init__(self) -> None:
        if not self.teachers:
            raise ValueError("MultiTeacherConfig requires at least one teacher.")
        models = [t.model for t in self.teachers]
        if len(set(models)) != len(models):
            raise ValueError(f"Duplicate teacher models in MultiTeacherConfig: {models}")


# ---------------------------------------------------------------------------
# Teacher top-K renormalization + multi-teacher blend
# ---------------------------------------------------------------------------


def teacher_topk_from_row(
    row: dict,
    *,
    ids_key: str = "teacher_topk_ids",
    logprobs_key: str = "teacher_topk_logprobs",
) -> list[TopKDist] | None:
    """Parse dataset-stored teacher top-K into per-response-position ``TopKDist``.

    DEFERRED offline hook (not wired into the online loop). The offline top-K KL
    path (teacher = separate frozen model, or golden top-K stored on the row) is
    deferred in favor of the co-located LoRA ``kl_distillation`` loss. Kept as a
    thin parser for when that path is built: expects parallel ``[response_len][k]``
    ``ids`` and ``logprobs`` arrays on the row. Returns ``None`` if absent.
    """
    ids_rows = row.get(ids_key)
    lps_rows = row.get(logprobs_key)
    if not isinstance(ids_rows, list) or not isinstance(lps_rows, list):
        return None
    if len(ids_rows) != len(lps_rows):
        raise ValueError(
            f"{ids_key} ({len(ids_rows)}) and {logprobs_key} ({len(lps_rows)}) "
            "must have the same number of positions."
        )
    out: list[TopKDist] = []
    for ids, lps in zip(ids_rows, lps_rows):
        out.append(
            TopKDist(token_ids=[int(t) for t in ids], logprobs=[float(v) for v in lps])
        )
    return out


def _renormalize_topk(dist: TopKDist) -> tuple[list[int], list[float]]:
    """Return ``(ids, q_renorm)`` where ``q`` sums to 1 over the candidates."""
    if not dist.logprobs:
        return [], []
    m = max(dist.logprobs)
    probs = [math.exp(lp - m) for lp in dist.logprobs]
    z = sum(probs)
    if z <= 0:
        return [], []
    return list(dist.token_ids), [p / z for p in probs]


def blend_teacher_topk(
    per_teacher: Sequence[tuple[TopKDist, float]],
    *,
    top_k: int,
) -> TopKDist:
    """Blend several teachers' top-K into one renormalized top-K (prob space).

    ``per_teacher`` is ``(dist, weight)`` pairs for the SAME response position.
    Mass is mixed as ``sum_t w_t * q_t(token)`` over the union of candidate ids,
    truncated to ``top_k`` and re-expressed as logprobs.
    """
    pooled: dict[int, float] = {}
    total_w = sum(w for _, w in per_teacher) or 1.0
    for dist, weight in per_teacher:
        ids, q = _renormalize_topk(dist)
        for tok_id, prob in zip(ids, q):
            pooled[tok_id] = pooled.get(tok_id, 0.0) + (weight / total_w) * prob
    if not pooled:
        return TopKDist(token_ids=[], logprobs=[])
    ranked = sorted(pooled.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    z = sum(p for _, p in ranked) or 1.0
    ids = [tok_id for tok_id, _ in ranked]
    lps = [math.log(max(p / z, 1e-30)) for _, p in ranked]
    return TopKDist(token_ids=ids, logprobs=lps)


# ---------------------------------------------------------------------------
# [N, K] top-K datum builder (shared by forward- and reverse-KL modes)
# ---------------------------------------------------------------------------


def build_topk_datum(
    model_input: tinker.ModelInput,
    topk_by_pos: Sequence[TopKDist | None],
    *,
    target_len: int,
    prompt_len: int,
    top_k: int,
) -> tinker.Datum:
    """Build one ``[N, K]`` datum shared by both top-K distillation modes.

    Encodes per position:
        ``target_tokens``  int64   ``[N, K]``  source top-K ids
        ``weights``        float32 ``[N, K]``  renormalized teacher prob ``q``,
                                                0 at padding / non-response slots

    ``topk_by_pos`` is indexed over the RESPONSE window; entry ``j`` lands at
    target position ``response_start + j`` (``response_start = prompt_len - 1``),
    matching ``build_opd_server_datums``. A ``None`` entry marks a position with
    no valid data; it gets all-zero weights and is skipped by both losses via the
    ``weights > 0`` mask. That ``weights > 0`` mask is purely datum *shaping*
    (padding / variable-K slots), NOT sampling masking.

    Sampling-nucleus (top-p/top-k) masking is intentionally NOT done here. The
    top-K must be **gathered after masking** in the RLOR backend: the engine owns
    the sampling params and should select/gather the top-K over the post-mask
    nucleus distribution (and the ``[N, K]`` gather of the other model's logprobs
    at those indices). The cookbook then just consumes already-masked top-K --
    reconstructing the nucleus client-side from inference ``top_logprobs`` would
    be lossy (cap <=5, token-string ambiguity) and duplicate engine logic.

    For ``TOPK_FORWARD_KL`` the builtin ``cross_entropy`` kernel consumes these
    directly (loss ``= -sum_k q_k log p_k``). For ``TOPK_REVERSE_KL`` the same
    datum feeds ``forward_backward_custom`` and ``reverse_kl_topk_loss`` rebuilds
    ``q`` from ``weights``.
    """
    response_start = max(0, prompt_len - 1)
    ids = [[0] * top_k for _ in range(target_len)]
    weights = [[0.0] * top_k for _ in range(target_len)]
    for j, dist in enumerate(topk_by_pos):
        pos = response_start + j
        if pos >= target_len or dist is None:
            continue
        tok_ids, q = _renormalize_topk(dist)
        for k, (tok_id, qk) in enumerate(zip(tok_ids[:top_k], q[:top_k])):
            ids[pos][k] = int(tok_id)
            weights[pos][k] = float(qk)

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[t for row in ids for t in row],
                dtype="int64",
                shape=[target_len, top_k],
            ),
            "weights": tinker.TensorData(
                data=[w for row in weights for w in row],
                dtype="float32",
                shape=[target_len, top_k],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Reverse-KL custom loss over teacher top-K (REINFORCE form)
# ---------------------------------------------------------------------------


def reverse_kl_topk_loss(
    data: list[tinker.Datum],
    logprobs_list: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Analytical reverse KL ``KL(pi_S || pi_T)`` over the teacher top-K.

    Consumes datums from :func:`build_topk_datum`. The server returns student
    logprobs at the teacher-top-K ``target_tokens`` (shape ``[N, K]``); we
    renormalize the student over those K slots, stop-grad the
    ``[log p_renorm - log q_renorm]`` bracket, and take the mass-weighted sum.
    Gradient flows only through the outer ``p_renorm`` weight, matching the
    ``importance_sampling`` convention. ``weights`` carries ``q_renorm`` at valid
    slots and ``0`` at padding/masked slots; validity is recovered via
    ``weights > 0``.

    Run this only via ``forward_backward_custom`` (two-pass): pass 1 is a server
    ``forward`` returning student logprobs at the teacher's ``[N,K]`` target ids,
    which the framework re-leafs with ``requires_grad`` on the client; this loss
    then produces ``dC/dlogprobs``; pass 2 backprops a CE surrogate with
    ``weights = -dC/dlogprobs``. The framework already errors if the loss yields
    no gradient w.r.t. the logprobs, so no extra guard is needed here. The
    forward-only top-K extraction (``forward(top_k=K)``, ``@no_grad``) is for KLD
    verification, never training -- it never reaches pass 2.
    """
    device = logprobs_list[0].device if logprobs_list else torch.device("cpu")
    total_loss = torch.zeros((), device=device)
    sum_kl = 0.0
    sum_positions = 0.0

    for i, datum in enumerate(data):
        student_logp_NK = logprobs_list[i]
        weights_NK = datum.loss_fn_inputs["weights"].to_torch().to(device)

        slot_mask_NK = weights_NK > 0
        position_mask_N = slot_mask_NK.any(dim=-1).float()

        safe_weights = weights_NK.clamp(min=1e-30)
        teacher_log_renorm_NK = torch.where(
            slot_mask_NK, torch.log(safe_weights), torch.zeros_like(weights_NK)
        )

        neg_inf = torch.full_like(student_logp_NK, float("-inf"))
        masked_logp = torch.where(slot_mask_NK, student_logp_NK, neg_inf)
        log_p_renorm = torch.log_softmax(masked_logp, dim=-1)
        log_p_renorm = torch.nan_to_num(log_p_renorm, nan=0.0, neginf=0.0)
        p_renorm = log_p_renorm.exp()

        adv_NK = (log_p_renorm - teacher_log_renorm_NK).detach()
        per_pos_N = (p_renorm * adv_NK * slot_mask_NK.float()).sum(dim=-1)
        total_loss = total_loss + (per_pos_N * position_mask_N).sum()

        with torch.no_grad():
            kl_NK = p_renorm * (log_p_renorm - teacher_log_renorm_NK) * slot_mask_NK.float()
            sum_kl += (kl_NK.sum(dim=-1) * position_mask_N).sum().item()
            sum_positions += position_mask_N.sum().item()

    metrics: dict[str, float] = {"opsd/reverse_kl_loss": total_loss.item()}
    if sum_positions > 0:
        metrics["opsd/reverse_kl_mean"] = sum_kl / sum_positions
    metrics["opsd/reverse_kl_positions"] = sum_positions
    return total_loss, metrics


def make_reverse_kl_topk_loss() -> Callable[
    [list[tinker.Datum], list[torch.Tensor]],
    tuple[torch.Tensor, dict[str, float]],
]:
    """Client loss factory for ``forward_backward_custom`` (reverse-KL top-K)."""
    return reverse_kl_topk_loss
