"""Compatibility path for :mod:`training.utils.rl.algorithm.dppo`."""

import sys

from training.utils.rl.algorithm import dppo as _module

sys.modules[__name__] = _module
