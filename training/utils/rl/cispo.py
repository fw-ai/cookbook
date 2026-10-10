"""Compatibility path for :mod:`training.utils.rl.algorithm.cispo`."""

import sys

from training.utils.rl.algorithm import cispo as _module

sys.modules[__name__] = _module
