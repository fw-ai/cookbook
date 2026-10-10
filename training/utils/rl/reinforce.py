"""Compatibility path for :mod:`training.utils.rl.algorithm.reinforce`."""

import sys

from training.utils.rl.algorithm import reinforce as _module

sys.modules[__name__] = _module
