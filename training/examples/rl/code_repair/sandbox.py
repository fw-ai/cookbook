"""Run model-written Python against hidden unit tests in an isolated process.

Each call starts a fresh ``python -I _worker.py`` under macOS ``sandbox-exec``
with a profile that denies network access, process spawning, writes outside a
per-job temp dir, and reads of the user's home directory (outside the Python
install and the worker). The environment is scrubbed (no API keys), CPU time
and file size are capped with rlimits, and the whole process group is killed at
the wall-clock timeout. Anything unexpected (crash, ``os._exit``, hang, junk
output) grades as zero passed: the result line must carry a per-job nonce.

Memory is not capped: macOS ignores ``RLIMIT_AS``; the wall timeout bounds it.
Model code shares the test namespace, so a determined candidate could still
find the randomly named pass counter; spot-check top-reward samples.

Tests are rewritten so every ``for i, (inp, exp) in enumerate(zip(inputs,
results))`` case runs in its own ``try`` with a per-case alarm, and calls of
the entry point are wrapped so non-builtin return values fail.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import resource
import secrets
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "_worker.py"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
DEFAULT_CONCURRENCY = max(1, (os.cpu_count() or 2) - 2)
MAX_OUTPUT_BYTES = 1 << 16

_LOOP = re.compile(
    r"^(?P<indent>[ \t]*)for i, \(inp, exp\) in enumerate\(zip\(inputs, results\)\):[ \t]*\n"
    r"(?P<body>(?:(?P=indent)[ \t]+\S.*(?:\n|$))+)",
    re.MULTILINE,
)

_semaphores: dict[int, asyncio.Semaphore] = {}


@dataclass(frozen=True)
class GradeResult:
    status: str
    """``ok``, ``syntax_error``, ``candidate_error``, ``test_error``, ``timeout`` or ``crashed``."""
    passed: int
    seen: int


def rewrite_tests(test: str, entry_point: str, token: str) -> str:
    """Per-case try/alarm around the single test loop; wrap entry-point calls."""
    matches = list(_LOOP.finditer(test))
    if len(matches) != 1:
        raise ValueError(f"expected one test loop, found {len(matches)}")
    m = matches[0]
    indent = m.group("indent")
    call = re.compile(r"\b(candidate|" + re.escape(entry_point) + r")\(\*inp\)")
    body = [call.sub(rf"_cr_check_{token}(\g<0>)", line) for line in m.group("body").rstrip("\n").split("\n")]
    inner = indent + "    "
    loop = [
        f"{indent}for i, (inp, exp) in enumerate(zip(inputs, results)):",
        f"{inner}_cr_start_{token}(i)",
        f"{inner}try:",
        *("    " + line for line in body),
        f"{inner}    _cr_ok_{token}(i)",
        f"{inner}except BaseException:",
        f"{inner}    _cr_fail_{token}(i)",
    ]
    rewritten = test[: m.start()] + "\n".join(loop) + "\n" + test[m.end() :]
    if re.search(r"^def check\(candidate\)", rewritten, re.MULTILINE):
        rewritten += f"\ncheck({entry_point})\n"
    return rewritten


def _profile(tmp: str) -> str:
    home = os.path.realpath(Path.home())
    readable = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    readable |= {tmp, str(WORKER)}
    executables = {sys.executable, os.path.realpath(sys.executable)}
    allow_read = " ".join(f'(subpath "{p}")' for p in sorted(readable))
    allow_exec = " ".join(f'(literal "{p}")' for p in sorted(executables))
    return f"""(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny process-exec)
(allow process-exec {allow_exec})
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null"))
(deny file-read-data (subpath "{home}"))
(allow file-read-data {allow_read})
"""


def _limits(cpu_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _semaphore(limit: int) -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(limit)
    return _semaphores[loop_id]


async def _read_capped(stream: asyncio.StreamReader) -> bytes:
    chunks, size = [], 0
    while size < MAX_OUTPUT_BYTES and (chunk := await stream.read(MAX_OUTPUT_BYTES - size)):
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


async def grade(
    code: str,
    test: str,
    entry_point: str,
    *,
    timeout: float = 20.0,
    case_timeout: float = 2.0,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> GradeResult:
    """Run ``code`` against ``test`` in the sandbox; count passing cases."""
    nonce, token = secrets.token_hex(16), secrets.token_hex(8)
    job = json.dumps(
        {
            "nonce": nonce,
            "token": token,
            "code": code,
            "test": rewrite_tests(test, entry_point, token),
            "case_timeout": case_timeout,
        }
    ).encode()
    async with _semaphore(concurrency):
        tmp = os.path.realpath(tempfile.mkdtemp(prefix="code-repair-"))
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                SANDBOX_EXEC, "-p", _profile(tmp), sys.executable, "-I", str(WORKER),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
                preexec_fn=lambda: _limits(int(timeout) + 1),
                start_new_session=True,
            )
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(job)
            await proc.stdin.drain()
            proc.stdin.close()
            output = await asyncio.wait_for(_read_capped(proc.stdout), timeout)
            await asyncio.wait_for(proc.wait(), 5)
        except (asyncio.TimeoutError, TimeoutError):
            return GradeResult("timeout", 0, 0)
        except OSError:
            return GradeResult("crashed", 0, 0)
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                await proc.wait()
            shutil.rmtree(tmp, ignore_errors=True)
    for line in output.decode(errors="replace").splitlines():
        if line.startswith(nonce + " "):
            result = json.loads(line[len(nonce) + 1 :])
            return GradeResult(result["status"], len(result["passed"]), int(result["seen"]))
    return GradeResult("crashed", 0, 0)
