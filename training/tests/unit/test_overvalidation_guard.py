from copy import deepcopy
import os
import signal
from unittest.mock import patch

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import overvalidation_guard as guard


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    process.mkdir(parents=True)
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0/tmp/opencode/test_attack.py\0")
    (process / "cwd").symlink_to("/app")
    script = tmp_path / "test_attack.py"
    script.write_text("for i in range(50):\n    attack(feal.encrypt)\n")
    policies = list(guard.POLICIES)
    policies[0] = {**policies[0], "required_file": str(script)}
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    record = {
        "trial": "harbor-opencode-feal-differential-cryptanalysis-0-test",
        "sandbox_id": "exact",
        "phase": "agent_or_setup",
        "remote": {
            "running_tools": [{"tool": "bash", "start_ms": 10, "elapsed_s": 1300}],
            "processes": [{
                "Pid": "10", "PPid": "20", "Name": "python3",
                "start_ticks": stat[19], "elapsed_s": 1300,
            }],
        },
    }
    expected = {"pid": 10, "start_ticks": stat[19], "policy_index": 0}
    return proc, expected, record


def test_positive_and_serialization(evidence):
    root, expected, record = evidence
    assert guard.is_known_overvalidation(expected, root)
    assert guard.recovery_candidates(record, record) == [expected]
    command = guard.recovery_command(expected)
    compile(command.split("\n", 1)[1].rsplit("\nRECOVER", 1)[0], "<remote>", "exec")


@pytest.mark.parametrize("field,value", [
    ("sandbox_id", "other"),
    ("phase", "verification_or_finalization"),
])
def test_changed_trial_state_rejected(evidence, field, value):
    _, _, record = evidence
    changed = deepcopy(record)
    changed[field] = value
    assert guard.recovery_candidates(record, changed) == []


def test_changed_process_or_tool_rejected(evidence):
    _, _, record = evidence
    changed = deepcopy(record)
    changed["remote"]["processes"][0]["start_ticks"] = "999"
    assert guard.recovery_candidates(record, changed) == []
    changed = deepcopy(record)
    changed["remote"]["running_tools"][0]["start_ms"] = 11
    assert guard.recovery_candidates(record, changed) == []


@pytest.mark.parametrize("mutation", ["cmdline", "cwd", "marker", "short", "name"])
def test_remote_revalidation_fails_closed(evidence, mutation):
    root, expected, _ = evidence
    process = root / "10"
    if mutation == "cmdline":
        (process / "cmdline").write_bytes(b"python3\0other.py\0")
    elif mutation == "cwd":
        (process / "cwd").unlink()
        (process / "cwd").symlink_to("/tmp")
    elif mutation == "marker":
        guard.POLICIES[0]["required_file"] = "/does/not/exist"
    elif mutation == "short":
        (root / "uptime").write_text("101 0")
    else:
        (process / "comm").write_text("other")
    assert not guard.is_known_overvalidation(expected, root)


def test_only_leaf_receives_sigint_after_revalidation(evidence):
    root, expected, _ = evidence
    with patch.object(guard.os, "pidfd_open", return_value=99) as opened, \
            patch.object(guard.signal, "pidfd_send_signal") as sent, \
            patch.object(guard.os, "close") as closed:
        result = guard.signal_known_overvalidation(expected, root)
        assert result["action"] == "SIGINT"
        opened.assert_called_once_with(10)
        sent.assert_called_once_with(99, signal.SIGINT)
        closed.assert_called_once_with(99)


@pytest.mark.parametrize(("reason", "argv", "cwd", "body"), [
    (
        "regex_chess_100_game_post_check_fuzz",
        ["python3", "fuzz.py", "100"],
        "/tmp/opencode",
        "ngames = int(sys.argv[1])\n"
        "for g in range(ngames):\n    pass\n"
        "print('tested positions:', total_pos, 'failures:', bad)\n",
    ),
    (
        "regex_chess_edge_case_post_check",
        ["python3", "/tmp/opencode/edge.py"],
        "/app",
        "from fuzz import check\n"
        "for f in fens:\n    check(f)\n"
        'print("DONE", "ALL OK" if ok else "FAILURES", len(fens))\n',
    ),
    (
        "regex_chess_150_game_post_check_stress",
        ["python3", "stress.py", "150"],
        "/tmp/opencode",
        'print("TOTAL tested:", tested, "fails:", fails)\n',
    ),
    (
        "regex_chess_4000_position_post_check_fuzz",
        ["python3", "fuzz.py"],
        "/app",
        'for i in range(4000):\n    print("random positions tested:")\n',
    ),
])
def test_regex_stress_variant_is_revalidated_independently(
    tmp_path, monkeypatch, reason, argv, cwd, body,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    process.mkdir(parents=True)
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in argv) + b"\0"
    )
    (process / "cwd").symlink_to(cwd)
    script = tmp_path / argv[1].rsplit("/", 1)[-1]
    script.write_text(body)
    policies = list(guard.POLICIES)
    stress_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"] == reason
    )
    policies[stress_index] = {
        **policies[stress_index], "required_file": str(script),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    record = {
        "trial": "harbor-opencode-regex-chess-0-test",
        "sandbox_id": "exact",
        "phase": "agent_or_setup",
        "remote": {
            "running_tools": [
                {"tool": "bash", "start_ms": 10, "elapsed_s": 1300},
            ],
            "processes": [{
                "Pid": "10", "PPid": "20", "Name": "python3",
                "start_ticks": stat[19], "elapsed_s": 1300,
            }],
        },
    }
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": stress_index,
    }
    candidates = guard.recovery_candidates(record, record)
    assert expected in candidates
    assert guard.is_known_overvalidation(expected, proc)

    wrong_variant = next(
        candidate
        for candidate in candidates
        if candidate["policy_index"] != stress_index
    )
    assert not guard.is_known_overvalidation(wrong_variant, proc)


def test_feal_linear_compiled_search_is_revalidated(tmp_path, monkeypatch):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    process.mkdir(parents=True)
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (find_approx) " + " ".join(stat))
    (process / "comm").write_text("find_approx")
    (process / "cmdline").write_bytes(b"./find_approx\0")
    (process / "cwd").symlink_to("/app")
    source = tmp_path / "find_approx.c"
    source.write_text(
        "int N = 1 << 24;\n"
        "for (int ia = 0; ia < na; ia++) {}\n"
        "for (int i = 0; i < N; i++) {}\n"
    )
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"] == "feal_linear_exhaustive_approximation_search"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(source),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    record = {
        "trial": "harbor-opencode-feal-linear-cryptanalysis-0-test",
        "sandbox_id": "exact",
        "phase": "agent_or_setup",
        "remote": {
            "running_tools": [
                {"tool": "bash", "start_ms": 10, "elapsed_s": 1300},
            ],
            "processes": [{
                "Pid": "10", "PPid": "20", "Name": "find_approx",
                "start_ticks": stat[19], "elapsed_s": 1300,
            }],
        },
    }
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.recovery_candidates(record, record) == [expected]
    assert guard.is_known_overvalidation(expected, proc)

    (process / "comm").write_text("other")
    assert not guard.is_known_overvalidation(expected, proc)


def test_heredoc_stress_requires_exact_parent_command(tmp_path, monkeypatch):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    parent = proc / "20"
    process.mkdir(parents=True)
    parent.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/tmp/opencode")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0for game in range(150):\n"
        b"for trial in range(200):\nprint('castle games done')\0"
    )
    helper = tmp_path / "fuzz.py"
    helper.write_text("def verify(fen):\n    return True\n")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_150_game_heredoc_post_check_fuzz"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(helper),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)

    (parent / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


def test_python_heredoc_400_game_fuzz_requires_exact_parent(
    tmp_path, monkeypatch,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    parent = proc / "20"
    process.mkdir(parents=True)
    parent.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python) " + " ".join(stat))
    (process / "comm").write_text("python")
    (process / "cmdline").write_bytes(b"python\0-\0")
    (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0def verify(fen):\nfor g in range(400):\n"
        b"print('random positions tested:', npos)\0"
    )
    packed = tmp_path / "re.json"
    packed.write_text("[]")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_400_game_heredoc_post_check_fuzz"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(packed),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)

    (parent / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


def test_12000_position_heredoc_requires_exact_parent(tmp_path, monkeypatch):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    parent = proc / "20"
    process.mkdir(parents=True)
    parent.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/tmp/opencode")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0# skip zero-move inputs\n"
        b"for it in range(6000):\n    pass\n"
        b"print('total:', tests, 'failures:', fails)\0"
    )
    packed = tmp_path / "re.json"
    packed.write_text("[]")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_12000_position_heredoc_post_check_fuzz"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(packed),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)

    (parent / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)
