"""Compatibility path for :mod:`training.utils.rl.algorithm.dro`."""

import sys

from training.utils.rl.algorithm import dro as _module

sys.modules[__name__] = _module
