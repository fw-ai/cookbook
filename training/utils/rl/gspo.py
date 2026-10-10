"""Compatibility path for :mod:`training.utils.rl.algorithm.gspo`."""

import sys

from training.utils.rl.algorithm import gspo as _module

sys.modules[__name__] = _module
