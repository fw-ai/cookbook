"""Compatibility path for :mod:`training.utils.rl.algorithm.grpo`."""

import sys

from training.utils.rl.algorithm import grpo as _module

sys.modules[__name__] = _module
