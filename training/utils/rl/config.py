"""RL training configuration."""

from __future__ import annotations

from typing import Callable
from dataclasses import field, dataclass

from fireworks.training.sdk.client import GradAccNormalization
from training.utils.config import (
    InfraConfig,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    RewardFn,
)
from training.utils.rl.tis import TISConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.losses import PromptGroup

FilterFn = Callable[[PromptGroup], bool]
"""Signature: (prompt_group) -> keep."""


@dataclass
class Config:
    """Full configuration for the RL training recipe.

    Customisation points:

    * ``reward_fn`` -- score completions (default: exact-match math reward).
    * ``filter_fn`` -- reject untrainable groups (default: zero-variance filter).
    * ``policy_loss`` -- select a registered loss algorithm.
    """

    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    # -- Model & data -------------------------------------------------------

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    """Max sequence length for sampling and training.  When using training
    shapes, this is auto-populated from the shape's
    ``max_supported_context_length``.  Must be set manually on the
    manual path (no training shape)."""
    lora_rank: int = 0

    # -- Sampling & batching ------------------------------------------------

    prompt_groups_per_step: int = 1
    """Number of prompt groups per optimizer step.

    All groups are collected before a single ``forward_backward_custom`` +
    ``optim_step`` pair fires (1:1 ratio)."""

    # -- Async rollout ------------------------------------------------------

    async_rollout: bool = False
    """Enable async rollout scheduling.  When True, rollouts overlap with
    training via ``AsyncRolloutScheduler``.  Weight sync fires every step
    (1:1:1 cadence), ignoring ``weight_sync.weight_sync_interval``."""

    valid_prompt_groups_per_step: int | None = None
    """Target accepted groups per step in async mode.  Defaults to
    ``prompt_groups_per_step`` when not set."""

    max_head_offpolicy_versions: int = 2
    """Maximum staleness: how many versions ahead the newest rollout can
    be relative to the oldest in-flight rollout."""

    sample_max_concurrency: int | None = None
    """Maximum number of concurrent HTTP **requests** to the deployment.

    ``sample_with_tokens(n=K)`` fans out into K individual HTTP
    requests.  This limit gates each HTTP request, not each prompt.
    With ``completions_per_prompt=8`` and ``sample_max_concurrency=32``,
    at most 32 HTTP requests are in-flight — meaning at most 4 prompts
    are being sampled concurrently (4 × 8 = 32).

    This is the *resource* window (how many requests actually hit the
    server), independent of the *policy* window (staleness cap).

    When ``None`` (default), no HTTP-level gate is applied — all
    requests from the rollout scheduler fire concurrently.
    Set a lower value (e.g. 32) to protect the deployment when the
    sample success rate drops below acceptable levels."""

    # -- Router replay (R3) -------------------------------------------------

    router_replay: bool = False
    router_replay_completion_only: bool = True

    # -- Training -----------------------------------------------------------

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS
    """Normalization mode for accumulated gradients at optim_step.
    Defaults to ``GradAccNormalization.NUM_LOSS_TOKENS`` (per-token mean)."""

    # -- Loss configuration -------------------------------------------------

    policy_loss: str = "grpo"
    """``"grpo"``, ``"importance_sampling"``, ``"dapo"``, ``"dro"``, ``"gspo"``, ``"reinforce"``, or ``"cispo"``.

    If an eligible builtin kernel exists for the selected loss, training uses
    the server-side ``forward_backward(...)`` path. Otherwise it falls back to
    the client-side ``forward_backward_custom(...)`` path.
    """

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    """PPO clip epsilon for the off-policy ratio (GRPO only)."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound (GRPO only)."""
    ratio_log_cap: float = 20.0
    """Log-ratio clamp for ``policy_loss="importance_sampling"``."""
    tis: TISConfig = field(default_factory=TISConfig)
    """TIS (Train-Inference IS) weight correction config."""

    # -- Pluggable functions ------------------------------------------------

    reward_fn: RewardFn | None = None
    """``(completion_text, dataset_row) -> float``.
    Each loop file provides its own default; set here to override."""

    filter_fn: FilterFn | None = None
    """``(PromptGroup) -> bool``.  Return ``True`` to keep.
    Set to ``None`` to accept all groups.  Each loop file provides
    its own default; set here to override."""

    # -- Trajectory logging -------------------------------------------------

    trajectory_dir: str | None = None
    """Directory to save per-step trajectory JSONL files.  Each file contains
    prompts, completions, and rewards for every prompt group in that step."""

    # -- Pre-created resources / resume -------------------------------------

    policy_job_id: str | None = None
    """Pre-created RLOR policy trainer job ID (skip creation if set)."""

    policy_base_url: str | None = None
    """Base URL for the policy trainer (bypass direct route)."""

    reference_job_id: str | None = None
    """Pre-created RLOR reference trainer job ID (skip creation if set)."""

    reference_base_url: str | None = None
    """Base URL for the reference trainer (bypass direct route)."""

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

    output_model_id: str | None = None

    # -- Sub-configs --------------------------------------------------------

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
