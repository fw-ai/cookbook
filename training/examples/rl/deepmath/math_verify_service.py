"""Symbolic ``math_verify`` comparison in a killable subprocess.

math_verify cannot be bounded safely in-process: its built-in timeout uses
``signal.SIGALRM`` (the alarm fires into arbitrary asyncio event-loop code and
wedges rollouts), and disabling it leaves sympy's hyperexpand/lerchphi
expansions unbounded. A worker *thread* running such an expansion cannot be
interrupted, so it leaks; enough leaked threads starve the GIL and stall the
trainer. A subprocess can simply be killed, which also unblocks the reader
thread (the pipe closes on kill), so nothing leaks.

Run as a script, this module is that worker: one JSON ``[pred, gt]`` request
per stdin line and one JSON boolean per stdout line.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import subprocess
import sys
import threading


@functools.cache
def _require_math_verify() -> None:
    # The worker answers "no match" for any exception, so a missing package
    # would silently score every symbolic answer 0 instead of failing the run.
    if importlib.util.find_spec("math_verify") is None:
        raise ImportError(
            "DeepMath grading needs the math-verify package. Install it with "
            "`pip install math-verify` or `pip install -e '.[dev]'` from training/."
        )


class MathVerifyService:
    """Serialize ``math_verify`` calls through one respawnable worker process."""

    def __init__(self, worker_path: str = __file__) -> None:
        self._worker_path = worker_path
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, self._worker_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        return self._proc

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            self._proc = None

    def verify(self, pred_str: str, gt_str: str, timeout: float = 5.0) -> bool:
        """Return the worker's verdict, or ``False`` after killing a wedged worker."""
        with self._lock:
            _require_math_verify()
            try:
                proc = self._ensure()
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps([pred_str, gt_str]) + "\n")
                proc.stdin.flush()

                result = [False]
                done = threading.Event()

                def read() -> None:
                    # Killing a wedged worker closes the pipe, so readline()
                    # returns EOF and this thread never leaks.
                    try:
                        line = proc.stdout.readline()
                        result[0] = bool(json.loads(line)) if line.strip() else False
                    except Exception:  # noqa: BLE001 - malformed reply = no match
                        result[0] = False
                    finally:
                        done.set()

                threading.Thread(target=read, daemon=True).start()
                if done.wait(timeout):
                    return result[0]
                # A pathological sympy expansion: kill the worker (it respawns
                # on the next call) and count the answer as unverified.
                self._kill()
                return False
            except Exception:  # noqa: BLE001 - a broken worker is no match
                self._kill()
                return False


_SERVICE = MathVerifyService()


def math_verify(pred_str: str, gt_str: str, timeout: float = 5.0) -> bool:
    """Symbolic equality check via the shared out-of-process worker."""
    return _SERVICE.verify(pred_str, gt_str, timeout=timeout)


def _serve() -> None:
    from math_verify import parse, verify

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            pred_str, gt_str = json.loads(line)
            pred, gt = parse(pred_str), parse(gt_str)
            # timeout_seconds=0 disables math_verify's SIGALRM timeout; the
            # parent bounds this process by killing it instead.
            result = bool(pred and gt and verify(pred, gt, timeout_seconds=0))
        except Exception:  # noqa: BLE001 - unparsable answers do not match
            result = False
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


__all__ = ["MathVerifyService", "math_verify"]


if __name__ == "__main__":
    _serve()
