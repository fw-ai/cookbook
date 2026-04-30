"""Renderer-backed multi-turn rollout — copyable template (fixture-only).

This package is a *copyable rollout template*: ``rollout.py`` and the
companion ``test_rollout.py`` exist so users can read the renderer-backed
multi-turn pattern and adapt it to their environment. There is intentionally
no ``train.py`` here — the trainer-side wiring lives in sibling examples
that are runnable end-to-end:

  - ``training/examples/rl/single_turn_async/`` — async on-policy single-turn.
  - ``training/examples/rl/ep_remote_grader/``  — remote grader integration.

Copy ``rollout.py`` into a new example tree (alongside a ``train.py``
modeled after the runnable siblings above) once your environment story is
ready; do not import from this package in production code paths.
"""
