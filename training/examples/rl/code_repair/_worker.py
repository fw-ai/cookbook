"""Sandboxed grading worker: run one candidate against its rewritten test script.

Reads one JSON job from stdin (``nonce``, ``token``, ``code``, ``test``,
``case_timeout``) and writes exactly one line, ``<nonce> <json>``, to a private
copy of stdout. The candidate's own stdout/stderr go to /dev/null, so it cannot
forge that line without the nonce. Started by :mod:`sandbox` under
``sandbox-exec``; never imported by the training process.
"""

import json
import os
import signal
import sys

_PLAIN_SCALARS = (int, float, bool, str, bytes, complex, type(None))


def _plain(value, depth=0):
    """Builtin data only, so a returned object cannot fake ``==``."""
    if depth > 50:
        return False
    kind = type(value)
    if kind in _PLAIN_SCALARS:
        return True
    if kind in (list, tuple, set, frozenset):
        return all(_plain(v, depth + 1) for v in value)
    if kind is dict:
        return all(_plain(k, depth + 1) and _plain(v, depth + 1) for k, v in value.items())
    return False


def main():
    job = json.loads(sys.stdin.read())
    nonce, token, case_timeout = job["nonce"], job["token"], float(job["case_timeout"])
    out = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    seen, passed = set(), set()

    def on_alarm(*_):
        raise TimeoutError("test case timed out")

    def start(i):
        seen.add(int(i))
        signal.setitimer(signal.ITIMER_REAL, case_timeout)

    def ok(i):
        signal.setitimer(signal.ITIMER_REAL, 0)
        passed.add(int(i))

    def fail(i):
        signal.setitimer(signal.ITIMER_REAL, 0)

    def check(value):
        if not _plain(value):
            raise TypeError("return value is not builtin data")
        return value

    signal.signal(signal.SIGALRM, on_alarm)
    namespace = {"__name__": "__candidate__"}
    status = "ok"
    try:
        exec(compile(job["code"], "<candidate>", "exec"), namespace)
    except SyntaxError:
        status = "syntax_error"
    except BaseException:
        status = "candidate_error"
    if status == "ok":
        namespace.update(
            {f"_cr_start_{token}": start, f"_cr_ok_{token}": ok, f"_cr_fail_{token}": fail, f"_cr_check_{token}": check}
        )
        try:
            exec(compile(job["test"], "<tests>", "exec"), namespace)
        except BaseException:
            status = "test_error"
    signal.setitimer(signal.ITIMER_REAL, 0)
    out.write(f"{nonce} {json.dumps({'status': status, 'seen': len(seen), 'passed': sorted(passed)})}\n")
    out.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
