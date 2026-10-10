"""Compatibility path for :mod:`training.utils.rl.algorithm.igpo`."""

import sys

from training.utils.rl.algorithm import igpo as _module

sys.modules[__name__] = _module
