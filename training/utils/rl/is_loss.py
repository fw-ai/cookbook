"""Compatibility path for :mod:`training.utils.rl.algorithm.importance_sampling`."""

import sys

from training.utils.rl.algorithm import importance_sampling as _module

sys.modules[__name__] = _module
