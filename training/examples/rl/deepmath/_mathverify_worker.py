#!/usr/bin/env python3
"""Out-of-process worker for math_verify symbolic comparison.

Runs as a standalone subprocess driven by ``train_deepmath._MathVerifyService``.
Protocol: one JSON request per stdin line ``[pred_str, gt_str]``; one JSON
response per stdout line (``true``/``false``).

This exists because math_verify cannot be safely bounded in-process:
its built-in timeout uses ``signal.SIGALRM`` (fires into arbitrary asyncio
event-loop code), and disabling it leaves sympy's ``hyperexpand``/``lerchphi``
expansions unbounded — a worker *thread* running such an expansion can't be
interrupted and leaks. A subprocess can simply be killed.
"""

import json
import sys

from math_verify import parse, verify


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            pred_str, gt_str = json.loads(line)
            pred = parse(pred_str)
            gt = parse(gt_str)
            # timeout_seconds=0 disables math_verify's SIGALRM timeout; the
            # parent process bounds us by killing this worker instead.
            result = bool(pred and gt and verify(pred, gt, timeout_seconds=0))
        except Exception:
            result = False
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
