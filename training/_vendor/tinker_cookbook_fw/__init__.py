"""Fireworks-modified modules from the fw-ai-external/tinker-cookbook fork.

Used only by SDFT. Copied from
https://github.com/fw-ai-external/tinker-cookbook at commit
``b1223e5535edea2efb824bfe73a38d35e0067439`` (branch ``fireworks``). Only the
files that the fork changes (to run on ``FiretitanTrainingClient``) and that
SDFT needs are copied here, in the fork's layout. Everything else is imported
from upstream ``tinker-cookbook==0.5.7``, installed with the ``sdft`` extra.

See ``README.md`` for what changed and how to re-sync. Re-sync from the fork
rather than editing these files in place.
"""

import importlib.util

if importlib.util.find_spec("tinker_cookbook") is None or importlib.util.find_spec("chz") is None:
    raise ImportError(
        "The SDFT recipe needs the 'sdft' extra. From cookbook/training, run: "
        "uv pip install -e '.[sdft]'"
    )
