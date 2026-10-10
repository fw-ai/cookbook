"""Compatibility path for :mod:`training.utils.rl.algorithm.score_centering`."""

import sys

from training.utils.rl.algorithm import score_centering as _module

sys.modules[__name__] = _module
