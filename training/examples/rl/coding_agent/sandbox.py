"""Local Docker runtime + SWE-Gym grading for the coding-agent example.

This is the Fireworks cookbook analogue of the public ProRL-Agent-Server
``examples/swegym_slime_grpo`` path: use the same SWE-Gym task distribution,
public runtime images, patch filtering, and fresh-runtime SWE-bench harness
grading.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shlex
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async runtime interface used by agent rollouts."""

    sandbox_id: str

    async def __aenter__(self) -> "Sandbox": ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


class LocalDockerSandbox:
    """Docker-backed runtime for SWE-Gym work and fresh evaluation."""

    def __init__(self, image: str, *, timeout: int | None = None) -> None:
        self.image = image or os.environ.get("SWE_DOCKER_IMAGE", "")
        if not self.image:
            raise ValueError("SWE-Gym rows must provide metadata.image or SWE_DOCKER_IMAGE must be set")
        self.timeout = timeout or int(os.environ.get("SWE_SANDBOX_LIFETIME_SEC", "3600"))
        self.sandbox_id = f"cagent-local-{secrets.token_hex(6)}"
        self._started = False

    async def __aenter__(self) -> "LocalDockerSandbox":
        rc, _, _ = await self._run_host(["docker", "image", "inspect", self.image], check=False, timeout=30)
        if rc != 0:
            await self._run_host(["docker", "pull", self.image], timeout=self.timeout)

        args = [
            "docker",
            "create",
            "--name",
            self.sandbox_id,
            "--network",
            os.environ.get("SWE_DOCKER_NETWORK", "host"),
        ]
        agent_cli_dir = os.environ.get("SWE_AGENT_CLI_DIR", "").strip()
        if agent_cli_dir:
            args.extend(["-v", f"{agent_cli_dir}:/opt/node:ro"])
        args.extend([self.image, "sleep", "infinity"])

        await self._run_host(args, timeout=120)
        try:
            await self._run_host(["docker", "start", self.sandbox_id], timeout=60)
            self._started = True
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._started:
            await self._run_host(["docker", "kill", self.sandbox_id], check=False, timeout=30)
        await self._run_host(["docker", "rm", "-f", self.sandbox_id], check=False, timeout=30)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        args = ["docker", "exec"]
        for k, v in (env or {}).items():
            args.extend(["-e", f"{k}={v}"])
        if user:
            args.extend(["-u", user])
        args.extend([self.sandbox_id, "bash", "-lc", cmd])
        rc, out, err = await self._run_host(args, check=False, timeout=timeout)
        if check and rc != 0:
            raise RuntimeError(f"docker exec failed (exit={rc}): {cmd[:120]}\n{err[:400]}")
        return rc, out, err

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        parent = shlex.quote(str(Path(sandbox_path).parent))
        await self.exec(f"mkdir -p {parent}", user="root", timeout=30, check=True)
        if isinstance(content, Path):
            src = content
            cleanup = None
        else:
            mode = "wb" if isinstance(content, bytes) else "w"
            tmp = tempfile.NamedTemporaryFile(mode=mode, delete=False)
            cleanup = Path(tmp.name)
            with tmp:
                tmp.write(content)
            src = cleanup
        try:
            await self._run_host(["docker", "cp", str(src), f"{self.sandbox_id}:{sandbox_path}"], timeout=600)
            await self.exec(
                f"chown {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(sandbox_path)} || true",
                user="root",
                timeout=30,
                check=False,
            )
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        rc, out, _ = await self.exec(f"cat {shlex.quote(sandbox_path)}", user=user, timeout=60)
        return out if rc == 0 else ""

    async def _run_host(
        self,
        args: list[str],
        *,
        check: bool = True,
        timeout: int = 120,
    ) -> ExecResult:
        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)

        proc = await asyncio.to_thread(_run)
        if check and proc.returncode != 0:
            raise RuntimeError(f"{' '.join(args[:4])} failed (exit={proc.returncode}): {proc.stderr[:400]}")
        return proc.returncode, proc.stdout or "", proc.stderr or ""


def make_sandbox(image: str) -> Sandbox:
    return LocalDockerSandbox(image)


_BOOT_SEM: asyncio.Semaphore | None = None
SWE_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
SWE_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))
CC_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)


@asynccontextmanager
async def boot_agent_sandbox(image: str) -> AsyncIterator[Sandbox]:
    """Boot a fresh runtime and ensure the agent CLI is available."""
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)

    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = make_sandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await install_claude_code(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[coding_agent] provision attempt %d/%d failed: %s: %s",
                attempt + 1,
                SWE_BOOT_RETRIES,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


async def install_claude_code(sb: Sandbox) -> None:
    """Use a mounted ProRL-style agent CLI dir when present, else npm-install."""
    package = os.environ.get("SWE_CC_NPM_PACKAGE", "@anthropic-ai/claude-code@2.1.111")
    await sb.exec(
        "set -e; "
        "if [ -x /opt/node/bin/node ]; then "
        "  ln -sf /opt/node/bin/node /usr/local/bin/node; "
        "  ln -sf /opt/node/bin/npm /usr/local/bin/npm || true; "
        "  ln -sf /opt/node/bin/npx /usr/local/bin/npx || true; "
        "  ln -sf /opt/node/bin/claude /usr/local/bin/claude || true; "
        "fi; "
        "if command -v claude >/dev/null 2>&1; then claude --version; exit 0; fi; "
        "node --version && npm --version && "
        f"npm install -g --prefix=/usr/local --no-audit --no-fund {shlex.quote(package)} && "
        "claude --version",
        user="root",
        timeout=600,
        check=True,
    )


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    wd = shlex.quote(workdir)
    await sb.exec(
        f"mkdir -p {wd} && "
        "id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {wd} && "
        "git config --system --add safe.directory '*' && id agent && "
        "mkdir -p /home/agent/.claude && "
        'echo \'{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}\' '
        "| tee /home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
        "chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
        user="root",
        check=True,
        timeout=60,
    )


async def apply_pre_commands(ev: Sandbox, workdir: str, pre: list[str] | str | None) -> None:
    if not pre:
        return
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in pre if c)
    await ev.write_file("/workspace/__cagent_pre__.sh", "set -e\n" + body, user="agent")
    await ev.exec(
        f"chmod 755 /workspace/__cagent_pre__.sh && cd {shlex.quote(workdir)} && bash /workspace/__cagent_pre__.sh",
        user="agent",
        check=False,
        timeout=600,
    )


async def run_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    middleware_url: str,
    time_budget_sec: int,
    problem_statement: str = "",
    pre_commands: list[str] | str | None = None,
    prompt: str | None = None,
) -> int:
    await ensure_agent_user(sb, workdir)
    await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(f"{workdir}/PROBLEM_STATEMENT.md", problem_statement or "", user="agent")
    return await _spawn_claude_code(
        sb,
        workdir=workdir,
        session_id=session_id,
        middleware_url=middleware_url,
        prompt=prompt or CC_PROMPT,
        time_budget_sec=time_budget_sec,
    )


async def _spawn_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    middleware_url: str,
    prompt: str,
    time_budget_sec: int,
) -> int:
    done = f"{workdir}/.cagent_done"
    launcher = f"{workdir}/.cagent_run.sh"
    traj = f"{workdir}/claude_code_trajectory.jsonl"
    launcher_body = (
        "#!/bin/bash\n"
        f"cd {shlex.quote(workdir)}\n"
        "export HOME=/home/agent\n"
        f"/usr/local/bin/claude -p {json.dumps(prompt)} "
        "--permission-mode bypassPermissions "
        "--output-format stream-json --include-partial-messages "
        "--include-hook-events --verbose "
        f"{os.environ.get('SWE_CLAUDE_EXTRA_ARGS', '').strip()} "
        f"2>&1 | tee {shlex.quote(traj)}\n"
        f"echo $? > {shlex.quote(done)}\n"
    )
    await sb.write_file(launcher, launcher_body, user="agent")
    await sb.exec(f"chmod +x {shlex.quote(launcher)}", user="agent", timeout=30)

    env = {
        "ANTHROPIC_BASE_URL": middleware_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": "fireworks-actor",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
    env_keys = ",".join(env.keys())
    await sb.exec(
        f"runuser -u agent --whitelist-environment={env_keys} "
        f"-- bash -c 'setsid {shlex.quote(launcher)} < /dev/null > /dev/null 2>&1 &'",
        user="root",
        env=env,
        timeout=30,
        check=True,
    )

    deadline = time.time() + time_budget_sec
    exit_code = -2
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {shlex.quote(done)} && cat {shlex.quote(done)}", user="agent", timeout=15)
        if ec == 0:
            try:
                exit_code = int((out or "").strip() or "-1")
            except ValueError:
                exit_code = -1
            break
    return exit_code


async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = (
        f"cd {shlex.quote(workdir)} && git add -N . && "
        "git diff -- . ':(exclude)PROBLEM_STATEMENT.md' "
        "':(exclude)claude_code_trajectory.jsonl' "
        "':(exclude).cagent_done' ':(exclude).cagent_run.sh'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


async def evaluate(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    swebench_instance: dict,
    exclude_patterns: list[str] | None = None,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool, bool]:
    """Grade ``diff_text`` with the SWE-Gym/SWE-bench harness in a fresh runtime."""
    async with make_sandbox(image) as ev:
        await ensure_agent_user(ev, workdir)
        await apply_pre_commands(ev, workdir, pre_commands)

        from training.examples.rl.coding_agent.swebench_harness import (
            evaluate_swebench_patch,
        )

        reward, solved, applied, report = await evaluate_swebench_patch(
            sandbox=ev,
            workdir=workdir,
            diff_text=diff_text,
            instance=swebench_instance,
            timeout_sec=timeout_sec,
            exclude_patterns=exclude_patterns,
        )
        logger.info(
            "[coding_agent.evaluate] swebench_harness resolved=%s exit=%s",
            solved,
            report.get("exit_code"),
        )
        return reward, solved, applied
