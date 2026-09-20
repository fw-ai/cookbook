"""Opt-in recovery for known, agent-authored unbounded validation loops.

Terminal-Bench agents sometimes finish the requested implementation and then
run a much larger local fuzz/stress test than the task verifier requires.  A
single such leaf process holds the entire synchronous RL batch.  This guard is
deliberately narrow: it recognizes only observed task/command pairs,
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
        "task_prefix": "harbor-opencode-feal-linear-cryptanalysis-",
        "min_elapsed_s": 600,
        "process_name": "find_approx",
        "cmdline": ["./find_approx"],
        "cwd": "/app",
        "required_file": "/app/find_approx.c",
        "required_markers": [
            "int N = 1 << 24;",
            "for (int ia = 0; ia < na; ia++)",
            "for (int i = 0; i < N; i++)",
        ],
        "reason": "feal_linear_exhaustive_approximation_search",
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
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "for i in range(4000):", "random positions tested:",
        ],
        "reason": "regex_chess_4000_position_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "60", "42"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "def random_fen_positions", 'print("testing", len(positions), "positions")',
        ],
        "reason": "regex_chess_60_position_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "ngames = 40", 'print("random positions tested:"',
            '"crafted ok. fails:"',
        ],
        "reason": "regex_chess_40_game_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": ["def verify"],
        "parent_cmdline_markers": [
            "for g in range(15):", "for ply in range(200):", "promo-fuzz:",
        ],
        "reason": "regex_chess_promotion_playout_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": ["def verify"],
        "parent_cmdline_markers": [
            "for game in range(150):", "for trial in range(200):",
            "castle games done",
        ],
        "reason": "regex_chess_150_game_heredoc_post_check_fuzz",
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
    process_name = policy.get("process_name", "python3")
    if (process / "comm").read_text().strip() != process_name:
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
    if policy.get("parent_cmdline_markers"):
        parent = proc_root / stat[1]
        parent_cmdline = os.fsdecode(
            (parent / "cmdline").read_bytes().replace(b"\0", b" ")
        )
        if not all(
            marker in parent_cmdline
            for marker in policy["parent_cmdline_markers"]
        ):
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
            process_name = policy.get("process_name", "python3")
            if (process.get("Name") == process_name
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
