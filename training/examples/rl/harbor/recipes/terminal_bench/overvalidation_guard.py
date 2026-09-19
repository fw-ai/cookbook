"""Opt-in recovery for known, agent-authored unbounded validation loops.

Terminal-Bench agents sometimes finish the requested implementation and then
run a much larger local fuzz/stress test than the task verifier requires.  A
single such leaf process holds the entire synchronous RL batch.  This guard is
deliberately narrow: it recognizes only three observed task/command pairs,
requires the same process identity in two consecutive observations, revalidates
the command, cwd, and task-specific files inside the sandbox, and sends SIGINT
only to that leaf process.  It never signals OpenCode, Harbor, trainer, or
rollout processes.
"""

import inspect
import json
import os
from pathlib import Path
import signal


POLICIES = (
    {
        "task_prefix": "harbor-opencode-feal-differential-cryptanalysis-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/opencode/test_attack.py"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/test_attack.py",
        "required_markers": ["attack(feal.encrypt)"],
        "reason": "feal_repeated_full_keyspace_stress_test",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "400"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": ["def fuzz_positions", "for g in range(n)"],
        "reason": "regex_chess_400_game_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "stress.py", "150"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/stress.py",
        "required_markers": ["TOTAL tested:", "fails:"],
        "reason": "regex_chess_150_game_post_check_stress",
    },
    {
        "task_prefix": "harbor-opencode-circuit-fibsqrt-",
        "min_elapsed_s": 1200,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/gates.txt",
        "required_markers": [],
        "reason": "circuit_fibsqrt_post_generation_stress_test",
    },
)


def _policies_for_trial(trial):
    return [
        (index, policy)
        for index, policy in enumerate(POLICIES)
        if trial.startswith(policy["task_prefix"])
    ]


def is_known_overvalidation(expected, proc_root=Path("/proc")):
    """Revalidate identity plus the exact task-specific command and files."""
    policy = POLICIES[expected["policy_index"]]
    process = proc_root / str(expected["pid"])
    stat = (process / "stat").read_text().rsplit(")", 1)[1].split()
    if stat[19] != expected["start_ticks"] or stat[0] == "Z":
        return False
    if (process / "comm").read_text().strip() != "python3":
        return False
    elapsed = (float((proc_root / "uptime").read_text().split()[0])
               - int(stat[19]) / os.sysconf("SC_CLK_TCK"))
    if elapsed < policy["min_elapsed_s"]:
        return False
    cmdline = [os.fsdecode(arg) for arg in (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    if cmdline != policy["cmdline"]:
        return False
    if os.readlink(process / "cwd") != policy["cwd"]:
        return False
    required = Path(policy["required_file"])
    if not required.is_file():
        return False
    if policy["required_markers"]:
        body = required.read_text(errors="replace")
        if not all(marker in body for marker in policy["required_markers"]):
            return False
    return True


def signal_known_overvalidation(expected, proc_root=Path("/proc")):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected["pid"])
        if not is_known_overvalidation(expected, proc_root):
            return {"action": "none", "reason": "evidence_changed"}
        signal.pidfd_send_signal(descriptor, signal.SIGINT)
        policy = POLICIES[expected["policy_index"]]
        return {
            "action": "SIGINT",
            "pid": expected["pid"],
            "start_ticks": expected["start_ticks"],
            "reason": policy["reason"],
        }
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {"action": "none", "reason": type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_candidates(previous, current):
    """Require a stable bash tool and stable leaf identity across two polls."""
    policies = _policies_for_trial(current.get("trial", ""))
    if (not policies or not previous or not current.get("sandbox_id")
            or previous.get("sandbox_id") != current["sandbox_id"]
            or previous.get("phase") != "agent_or_setup"
            or current.get("phase") != "agent_or_setup"):
        return []
    before, after = previous.get("remote", {}), current.get("remote", {})
    old_tools = {tool.get("start_ms") for tool in before.get("running_tools", [])
                 if tool.get("tool") == "bash" and tool.get("start_ms") is not None}
    if not any(tool.get("tool") == "bash" and tool.get("start_ms") in old_tools
               for tool in after.get("running_tools", [])):
        return []
    old = {process["Pid"]: process for process in before.get("processes", [])}
    result = []
    for process in after.get("processes", []):
        prior = old.get(process["Pid"], {})
        for policy_index, policy in policies:
            if (process.get("Name") == "python3"
                    and (process.get("elapsed_s") or 0) >= policy["min_elapsed_s"]
                    and process.get("start_ticks") is not None
                    and process["start_ticks"] == prior.get("start_ticks")
                    and process.get("PPid") == prior.get("PPid")):
                result.append({
                    "pid": int(process["Pid"]),
                    "start_ticks": process["start_ticks"],
                    "policy_index": policy_index,
                })
    return result


def recovery_command(expected):
    """Serialize only numeric identity; command checks remain in the sandbox."""
    payload = {
        "pid": int(expected["pid"]),
        "start_ticks": str(int(expected["start_ticks"])),
        "policy_index": int(expected["policy_index"]),
    }
    return (
        "python3 - <<'RECOVER'\nimport os, signal, json\nfrom pathlib import Path\n"
        + f"POLICIES = {POLICIES!r}\n"
        + inspect.getsource(is_known_overvalidation)
        + "\n"
        + inspect.getsource(signal_known_overvalidation)
        + f"\nprint(json.dumps(signal_known_overvalidation(json.loads({json.dumps(payload)!r}))))\nRECOVER"
    )
