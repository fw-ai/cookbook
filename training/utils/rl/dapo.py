"""Compatibility path for :mod:`training.utils.rl.algorithm.dapo`."""

import sys

from training.utils.rl.algorithm import dapo as _module

sys.modules[__name__] = _module
