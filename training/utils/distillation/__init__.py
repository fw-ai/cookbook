"""Sampled-token on-policy distillation (OPD) helpers.

- :mod:`.losses`: student samples + teacher logprobs -> server-side
  ``importance_sampling`` datums whose per-token advantage is
  ``teacher_logprob - sampling_logprob``.
- :mod:`.sampling`: teacher scoring requests and logprob/token alignment.
- :mod:`.teacher`: the frozen teacher sampler (existing model or new deployment).
- :mod:`.eval`: privileged-context evaluation and run validation helpers.
"""

from training.utils.distillation.losses import (
    OPDInputMetrics,
    OPDPromptGroup,
    build_opd_datums,
    build_opd_prompt_group,
    build_opd_server_datums,
    combine_opd_prompt_groups,
)

__all__ = [
    "OPDInputMetrics",
    "OPDPromptGroup",
    "build_opd_datums",
    "build_opd_prompt_group",
    "build_opd_server_datums",
    "combine_opd_prompt_groups",
]
