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


def test_leaf_ignoring_sigint_receives_sigterm(evidence):
    root, expected, _ = evidence
    (root / "10" / "status").write_text("SigIgn:\t0000000000000002\n")
    with patch.object(guard.os, "pidfd_open", return_value=99), \
            patch.object(guard.signal, "pidfd_send_signal") as sent, \
            patch.object(guard.os, "close"):
        result = guard.signal_known_overvalidation(expected, root)
        assert result["action"] == "SIGTERM"
        sent.assert_called_once_with(99, signal.SIGTERM)


def test_large_scale_text_editing_vim_stall_is_exactly_guarded(
        tmp_path, monkeypatch):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    parent = proc / "20"
    process.mkdir(parents=True)
    parent.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["S"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (vim) " + " ".join(stat))
    (process / "comm").write_text("vim")
    (process / "cmdline").write_bytes(
        b"vim\0-Nu\0NONE\0-n\0-Es\0/tmp/opencode/t.csv\0"
        b"-S\0/tmp/opencode/dbg\0"
    )
    (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"/bin/bash\0-c\0"
        b"vim -Nu NONE -n -Es /tmp/opencode/t.csv -S /tmp/opencode/dbg; "
        b"echo \\\"exit=$?\\\"; cat /tmp/opencode/t.csv\0"
    )
    script = tmp_path / "dbg"
    script.write_text(
        r":s/\v^\s*([^,]*?)\s*,\s*([^,]*?)\s*,\s*([^,]*?)\s*$/\3;\2;\1/"
        "\n:wq\n"
    )
    policies = list(guard.POLICIES)
    policy_index = next(
        index for index, policy in enumerate(policies)
        if policy["reason"]
        == "large_scale_text_editing_noninteractive_vim_stall"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(script),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    record = {
        "trial": "harbor-opencode-large-scale-text-editing-0-test",
        "sandbox_id": "exact",
        "phase": "agent_or_setup",
        "remote": {
            "running_tools": [
                {"tool": "bash", "start_ms": 10, "elapsed_s": 700},
            ],
            "processes": [{
                "Pid": "10", "PPid": "20", "Name": "vim",
                "start_ticks": stat[19], "elapsed_s": 700,
            }],
        },
    }
    candidates = guard.recovery_candidates(record, record)
    assert expected in candidates
    assert guard.is_known_overvalidation(expected, proc)
    assert not any(
        guard.is_known_overvalidation(candidate, proc)
        for candidate in candidates
        if candidate != expected
    )

    (parent / "cmdline").write_bytes(b"bash\0-c\0vim unrelated.csv\0")
    assert not guard.is_known_overvalidation(expected, proc)


@pytest.mark.parametrize(("reason", "argv", "cwd", "body", "parent_body"), [
    (
        "scheduler_repeated_post_solution_tune4_sweep",
        ["python3", "/tmp/opencode/tune4.py"],
        "/app",
        "for (lam1, mu1, nu1, lam2, mu2, nu2) in [\n"
        "    X.build_plans()\n    X.report()\n",
        "cat > /tmp/opencode/tune4.py; "
        "time python3 /tmp/opencode/tune4.py",
    ),
    (
        "scheduler_72_configuration_post_solution_grid",
        [
            "python3", "-c",
            "for lam_seq in [20, 25, 30, 35, 40, 50, 60, 70]:\n"
            "  for lam_pad in [0.0, 1e4, 3e4]:\n"
            "    for lam95 in [1e10, 1e8, 1e7]:\n"
            "      pass\nprint('BEST b1:', best)\n",
        ],
        "/tmp/opencode",
        "def solve_bucket():\n    pass\ndef evaluate():\n    pass\n",
        "timeout 1800 python3 -c sweep | tail -40",
    ),
])
def test_scheduler_post_solution_sweeps_are_exactly_guarded(
        tmp_path, monkeypatch, reason, argv, cwd, body, parent_body):
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
    (process / "cmdline").write_bytes(
        b"\0".join(arg.encode() for arg in argv) + b"\0"
    )
    (process / "cwd").symlink_to(cwd)
    (parent / "cmdline").write_bytes(
        b"/bin/bash\0-c\0" + parent_body.encode() + b"\0"
    )
    helper = tmp_path / "helper.py"
    helper.write_text(body)
    policies = list(guard.POLICIES)
    policy_index = next(
        index for index, policy in enumerate(policies)
        if policy["reason"] == reason
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
    record = {
        "trial": "harbor-opencode-llm-inference-batching-scheduler-0-test",
        "sandbox_id": "exact",
        "phase": "agent_or_setup",
        "remote": {
            "running_tools": [
                {"tool": "bash", "start_ms": 10, "elapsed_s": 700},
            ],
            "processes": [{
                "Pid": "10", "PPid": "20", "Name": "python3",
                "start_ticks": stat[19], "elapsed_s": 700,
            }],
        },
    }
    candidates = guard.recovery_candidates(record, record)
    assert expected in candidates
    assert guard.is_known_overvalidation(expected, proc)
    assert not any(
        guard.is_known_overvalidation(candidate, proc)
        for candidate in candidates
        if candidate != expected
    )

    (parent / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


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
        "regex_chess_25_game_timed_post_check_fuzz",
        ["python3", "fuzz.py", "25"],
        "/tmp/opencode",
        "rnd = random.Random(12345)\n"
        "ngames = int(sys.argv[1]) if len(sys.argv) > 1 else 30\n"
        "while not b.is_game_over(claim_draw=False):\n    pass\n"
        'print("DONE. failures:", fail)\n',
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
        "regex_chess_400_game_tmp_heredoc_post_check_fuzz",
        ["python3", "-"],
        "/tmp/opencode",
        "checkpoint bytes are validated separately\n",
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
    (
        "regex_chess_4000_game_post_check_fuzz",
        ["python3", "fuzz.py"],
        "/app",
        "random.seed(12345)\n"
        "for g in range(4000):\n"
        "    for ply in range(120):\n        pass\n"
        'print("fuzz done: %d games, %d white positions, %d failures" '
        "% (games, pos, n_fail))\n",
    ),
    (
        "regex_chess_40_game_fuzz2_post_check",
        ["python3", "/tmp/opencode/fuzz2.py", "999"],
        "/app",
        "GAMES = 40\nfor g in range(GAMES):\n    pass\n"
        'print("fuzz2 done: %d white positions, fails %d" % (tests, fails))\n',
    ),
    (
        "regex_chess_60_game_timed_post_check_fuzz",
        ["python3", "fuzz.py"],
        "/app",
        "random.seed(12345)\nNGAMES = 60\n"
        "while not b.is_game_over() and b.fullmove_number < 100:\n    pass\n"
        'print("total: %d, fails: %d" % (total, fails))\n',
    ),
    (
        "regex_chess_60_game_import_post_check_fuzz",
        ["python3", "-"],
        "/app",
        "random.seed(12345)\nNGAMES = 60\n"
        "while not b.is_game_over() and b.fullmove_number < 100:\n    pass\n"
        'print("total: %d, fails: %d" % (total, fails))\n',
    ),
    (
        "regex_chess_60_game_300_ply_post_check_fuzz",
        ["python3", "fuzz.py"],
        "/app",
        "random.seed(12345)\nfor g in range(60):\n"
        "    for ply in range(300):\n        pass\n"
        'print("random games done:", ntested, "positions, failures:", nfail)\n',
    ),
    (
        "regex_chess_30_game_seed7_post_check_fuzz",
        ["python3", "fuzz.py", "30", "7"],
        "/app",
        "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20\n"
        "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0\n"
        "while not b.is_game_over() and b.fullmove_number < 70:\n    pass\n"
        'print("tested positions:", tested + len(specials), "OK")\n',
    ),
    (
        "regex_chess_30_game_seed7_logged_post_check_fuzz",
        ["python3", "fuzz.py", "30", "7"],
        "/app",
        "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20\n"
        "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0\n"
        "while not b.is_game_over() and b.fullmove_number < 70:\n    pass\n"
        'print("tested positions:", tested + len(specials), "OK")\n',
    ),
    (
        "regex_chess_200_game_seed99_logged_post_check_fuzz",
        ["python3", "fuzz.py", "200", "99"],
        "/app",
        "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20\n"
        "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0\n"
        "while not b.is_game_over() and b.fullmove_number < 70:\n    pass\n"
        'print("tested positions:", tested + len(specials), "OK")\n',
    ),
    (
        "regex_chess_400_game_seed12345_logged_post_check_fuzz",
        ["python3", "fuzz.py", "400", "12345"],
        "/app",
        "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20\n"
        "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0\n"
        "while not b.is_game_over() and b.fullmove_number < 70:\n    pass\n"
        'print("tested positions:", tested + len(specials), "OK")\n',
    ),
    (
        "regex_chess_orphan_300_game_seed12345_post_check_fuzz",
        ["python3", "fuzz.py", "300", "12345"],
        "/app",
        "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20\n"
        "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0\n"
        "while not b.is_game_over() and b.fullmove_number < 70:\n    pass\n"
        'print("tested positions:", tested + len(specials), "OK")\n',
    ),
    (
        "regex_chess_10_game_timed_post_check_fuzz",
        ["python3", "-u", "fuzz2.py", "10"],
        "/tmp/opencode",
        "rnd = random.Random(999)\ng = int(sys.argv[1])\n"
        "while not b.is_game_over(claim_draw=False):\n    pass\n"
        'print("ALL OK. positions:", pos, flush=True)\n',
    ),
    (
        "regex_chess_part_00_fen_post_check",
        ["python3", "-u", "checkfens.py", "part_00"],
        "/tmp/opencode",
        "for line in open(sys.argv[1]):\n"
        "    board = chess.Board(fen)\n"
        'print(sys.argv[1], "fails:", fails)\n',
    ),
    (
        "regex_chess_parallel_300_game_seed111_post_check",
        ["python3", "fuzz_par.py", "111", "300"],
        "/app",
        "seed = int(sys.argv[1])\nngames = int(sys.argv[2])\n"
        "for ply in range(120):\n    pass\n"
        'log.write("DONE seed=%d games=%d positions=%d fails=%d" % values)\n',
    ),
    (
        "regex_chess_parallel_150_game_seed777_post_check",
        ["python3", "fuzz_par.py", "777", "150"],
        "/app",
        "seed = int(sys.argv[1])\nngames = int(sys.argv[2])\n"
        "for ply in range(120):\n    pass\n"
        'log.write("DONE seed=%d games=%d positions=%d fails=%d" % values)\n',
    ),
    (
        "regex_chess_150_game_fuzz4_post_check",
        ["python3", "/tmp/opencode/fuzz4.py", "424242"],
        "/app",
        "for game in range(150):\n"
        "    while not b.is_game_over():\n        pass\n"
        'print("fuzz4 done: %d positions, fails %d" % (tests, fails))\n',
    ),
    (
        "regex_chess_parallel_160_game_worker_post_check",
        ["python3", "fuzzpar.py", "160"],
        "/app",
        "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 160\n"
        "seeds = list(range(1000, 1000 + n_games))\n"
        "all_games = pool.map(gen_game_fens, seeds)\n"
        'print(f"DONE tested={total_tested} fails={len(total_fails)}")\n',
    ),
    (
        "regex_chess_seed3_random_post_check",
        ["python3", "randtest.py", "3"],
        "/tmp/opencode",
        "random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 7)\n"
        "for g in range(NGAMES):\n"
        "    for ply in range(250):\n        pass\n"
        'print(f"ALL PASS: {ntests} positions in {time.time()-t0:.1f}s")\n',
    ),
    (
        "regex_chess_seed101_random_post_check",
        ["python3", "randtest.py", "101"],
        "/tmp/opencode",
        "random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 7)\n"
        "for g in range(NGAMES):\n"
        "    for ply in range(250):\n        pass\n"
        'print(f"ALL PASS: {ntests} positions in {time.time()-t0:.1f}s")\n',
    ),
    (
        "regex_chess_seed202_random_post_check",
        ["python3", "randtest.py", "202"],
        "/tmp/opencode",
        "random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 7)\n"
        "for g in range(NGAMES):\n"
        "    for ply in range(250):\n        pass\n"
        'print(f"ALL PASS: {ntests} positions in {time.time()-t0:.1f}s")\n',
    ),
    (
        "regex_chess_ep_castling_800_case_post_check",
        ["python3", "/tmp/fuzz2.py"],
        "/app",
        "rnd = random.Random(999)\nwhile n < 400:\n    pass\n"
        "while n2 < 400:\n    pass\n"
        'print("OK" if fuzz.FAILS == 0 else "FAIL")\n',
    ),
    (
        "regex_chess_40_game_generated_post_check",
        ["python3", "fuzz.py"],
        "/app",
        "random.seed(12345)\nN_GAMES = 40\n"
        "while board.fullmove_number < 80:\n    pass\n"
        'print("TOTAL tested", tested, "fails", fails)\n',
    ),
    (
        "regex_chess_seed1_40_game_timed_post_check",
        ["python3", "fuzz.py", "1", "40"],
        "/tmp/opencode",
        "NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50\n"
        "while b.fullmove_number < 100:\n    pass\n"
        'print("tested", tested, "positions, fails", fails)\n',
    ),
    (
        "regex_chess_extended_edge_case_post_check",
        ["python3", "/tmp/opencode/edge.py"],
        "/app",
        "from fuzz import check\nfor fen in cases:\n    check(fen)\n"
        'print("edge cases:", "ALL OK" if ok else "FAILURES")\n',
    ),
    (
        "regex_chess_600_composed_position_post_check",
        ["python3", "-"],
        "/app",
        "checkpoint bytes are validated separately\n",
    ),
    (
        "regex_chess_120_game_bigtest_post_check",
        ["python", "bigtest.py"],
        "/tmp/opencode",
        "rng = random.Random(99)\nfor game in range(120):\n"
        "    while b.fullmove_number < 100:\n        pass\n"
        "print('ALL OK')\n",
    ),
    (
        "regex_chess_ep_60_case_post_check",
        ["python3", "/tmp/fuzz2.py"],
        "/app",
        "rnd = random.Random(999)\nwhile n < 60:\n    pass\n"
        'print("ep suite done, FAILS:", fuzz.FAILS)\n',
    ),
    (
        "regex_chess_40_game_timed_generated_post_check",
        ["python3", "fuzz.py"],
        "/app",
        "random.seed(12345)\nN_GAMES = 40\n"
        "while board.fullmove_number < 80:\n    pass\n"
        'print("TOTAL tested", tested, "fails", fails)\n',
    ),
    (
        "regex_chess_seed42_120_game_timed_post_check",
        ["python3", "fuzz.py", "42", "120"],
        "/tmp/opencode",
        "NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50\n"
        "while b.fullmove_number < 100:\n    pass\n"
        'print("tested", tested, "positions, fails", fails)\n',
    ),
    (
        "regex_chess_1500_game_post_check",
        ["python3", "fuzz.py", "1500"],
        "/tmp/opencode",
        "random.seed(12345)\n"
        "ngames = int(sys.argv[1]) if len(sys.argv) > 1 else 200\n"
        "for ply in range(200):\n    pass\n"
        'print("positions tested:", npos, "bad:", nbad)\n',
    ),
    (
        "regex_chess_seed999_100_game_post_check",
        ["python3", "fuzz.py", "999", "100"],
        "/tmp/opencode",
        "random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)\n"
        "NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50\n"
        "while not b.is_game_over() and b.fullmove_number < 100:\n    pass\n"
        'print("tested", tested, "positions, fails", fails)\n',
    ),
    (
        "regex_chess_import_executes_40_game_post_check",
        [
            "python3", "-c",
            "import random, chess, sys\n"
            "sys.setcheckinterval= None\n"
            "import fuzz\n",
        ],
        "/app",
        "random.seed(12345)\nN_GAMES = 40\n"
        "while board.fullmove_number < 80:\n    pass\n"
        'print("TOTAL tested", tested, "fails", fails)\n',
    ),
    (
        "regex_chess_130_game_finaltest_post_check",
        ["python", "finaltest.py"],
        "/tmp/opencode",
        "rng = random.Random(31337)\nfor game in range(130):\n"
        "    while not b.is_game_over() and b.fullmove_number < 110:\n"
        "        pass\nprint('ALL OK')\n",
    ),
    (
        "regex_chess_60_game_stdin_post_check",
        ["python3", "-"],
        "/app",
        "checkpoint bytes are validated separately\n",
    ),
    (
        "regex_chess_reduced_15_game_post_check",
        ["python3", "fuzz.py"],
        "/app",
        "random.seed(12345)\nN_GAMES = 15\n"
        "while board.fullmove_number < 80:\n    pass\n"
        'print("TOTAL tested", tested, "fails", fails)\n',
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
    process_name = policies[stress_index].get("process_name", "python3")
    (process / "comm").write_text(process_name)
    policies[stress_index] = {
        **policies[stress_index], "required_file": str(script),
    }
    if (policies[stress_index].get("parent_cmdline_markers")
            or policies[stress_index].get("ancestor_cmdline_markers")):
        parent = proc / "20"
        parent.mkdir()
    if policies[stress_index].get("parent_cmdline_markers"):
        parent = proc / "20"
        (parent / "cmdline").write_bytes(
            b"\0".join(
                marker.encode()
                for marker in policies[stress_index]["parent_cmdline_markers"]
            ) + b"\0"
        )
    if policies[stress_index].get("ancestor_cmdline_markers"):
        ancestor_pid = 20
        for hop in range(policies[stress_index]["ancestor_hops"] - 1):
            next_pid = 30 + hop
            ancestor = proc / str(ancestor_pid)
            ancestor_stat = ["S"] + ["0"] * 19
            ancestor_stat[1] = str(next_pid)
            (ancestor / "stat").write_text(
                f"{ancestor_pid} (bash) " + " ".join(ancestor_stat)
            )
            ancestor_pid = next_pid
            (proc / str(ancestor_pid)).mkdir(exist_ok=True)
        (proc / str(ancestor_pid) / "cmdline").write_bytes(
            b"\0".join(
                marker.encode()
                for marker in policies[stress_index]["ancestor_cmdline_markers"]
            ) + b"\0"
        )
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
                "Pid": "10", "PPid": "20", "Name": process_name,
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


def test_parallel_regex_policies_cover_only_observed_seed_commands():
    commands = {
        tuple(policy["cmdline"])
        for policy in guard.POLICIES
        if policy["reason"].startswith("regex_chess_parallel_300_game_seed")
    }
    assert commands == {
        ("python3", "fuzz_par.py", seed, "300")
        for seed in ("111", "222", "333", "444")
    }


def test_parallel_fen_policies_cover_only_observed_part_commands():
    commands = {
        tuple(policy["cmdline"])
        for policy in guard.POLICIES
        if policy["reason"].startswith("regex_chess_part_")
    }
    assert commands == {
        ("python3", "-u", "checkfens.py", f"part_{part:02d}")
        for part in range(8)
    }


@pytest.mark.parametrize(("reason", "duration", "extra_parent"), [
    ("regex_chess_parallel_fuzz_completion_wait", "900", ""),
    (
        "regex_chess_parallel_fuzz_repeated_completion_wait",
        "1500",
        " fuzz_*.log ep_*.log",
    ),
])
def test_parallel_regex_completion_wait_requires_exact_parent(
    tmp_path, monkeypatch, reason, duration, extra_parent,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    parent = proc / "20"
    process.mkdir(parents=True)
    parent.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["S"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (sleep) " + " ".join(stat))
    (process / "comm").write_text("sleep")
    (process / "cmdline").write_bytes(
        b"sleep\0" + duration.encode() + b"\0"
    )
    (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0sleep " + duration.encode()
        + b"; tail fuzz_111.log fuzz_222.log fuzz_333.log fuzz_444.log; "
        + b"grep -h FAIL" + extra_parent.encode() + b"\0"
    )
    script = tmp_path / "fuzz_par.py"
    script.write_text(
        "seed = int(sys.argv[1])\nngames = int(sys.argv[2])\n"
        "for ply in range(120):\n    pass\n"
    )
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"] == reason
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(script),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0sleep " + duration.encode() + b"\0"
    )
    assert not guard.is_known_overvalidation(expected, proc)


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


def test_filter_js_whatwg_sequence_matcher_requires_exact_parent(
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
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/tmp/opencode/xss")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0"
        b"orig = urllib.request.urlopen(\"https://html.spec.whatwg.org/\")\n"
        b"sm = difflib.SequenceMatcher(None, orig, after, autojunk=False)\n"
        b"print(\"opcodes:\", len(sm.get_opcodes()), "
        b"\"non-script removals:\", nonscript)\0"
    )
    fixture = tmp_path / "big.html"
    fixture.write_text("<!doctype html>")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "filter_js_whatwg_quadratic_sequence_matcher_post_check"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(fixture),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    record = {
        "trial": "harbor-opencode-filter-js-from-html-0-test",
        "sandbox_id": "exact",
        "phase": "agent_or_setup",
        "remote": {
            "running_tools": [
                {"tool": "bash", "start_ms": 10, "elapsed_s": 700},
            ],
            "processes": [{
                "Pid": "10", "PPid": "20", "Name": "python3",
                "start_ticks": stat[19], "elapsed_s": 700,
            }],
        },
    }
    assert guard.recovery_candidates(record, record) == [expected]
    assert guard.is_known_overvalidation(expected, proc)

    (parent / "cmdline").write_bytes(
        b"bash\0-c\0python3 legitimate_required_work.py\0"
    )
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


@pytest.mark.parametrize(("reason", "parent_body"), [
    (
        "regex_chess_90_game_seed_range_heredoc_post_check",
        "timeout 1200 python3 -\n"
        "for seed in range(1000, 1006):\n"
        "    for g in range(15):\n"
        "        while not b.is_game_over() and b.fullmove_number < 150:\n"
        "            pass\n"
        'print("SOAK OK, ntest =", ntest)\n',
    ),
    (
        "regex_chess_400_game_seed1337_heredoc_post_check",
        "timeout 3500 python3 -\n"
        "rng = random.Random(1337)\n"
        "for g in range(400):\n"
        "    while not b.is_game_over() and b.ply() < 160:\n"
        "        pass\n"
        'print(f"done: tested={tested} fails={fails} '
        'interesting-games={interesting}")\n',
    ),
    (
        "regex_chess_250_game_seed1337_heredoc_post_check",
        "timeout 3500 python3 -\n"
        "rng = random.Random(1337)\n"
        "for g in range(250):\n"
        "    while not b.is_game_over() and b.ply() < 160:\n"
        "        pass\n"
        'print(f"done: tested={tested} fails={fails} '
        'interesting={interesting}")\n',
    ),
])
def test_regex_heredoc_soaks_require_exact_parent_markers(
    tmp_path, monkeypatch, reason, parent_body,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    timeout = proc / "20"
    shell = proc / "30"
    process.mkdir(parents=True)
    timeout.mkdir()
    shell.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/tmp/opencode")
    timeout_stat = ["S"] + ["0"] * 19
    timeout_stat[1], timeout_stat[19] = "30", str(99 * hz)
    (timeout / "stat").write_text(
        "20 (timeout) " + " ".join(timeout_stat)
    )
    (timeout / "cmdline").write_bytes(b"timeout\0python3\0-\0")
    (shell / "cmdline").write_bytes(
        b"/bin/bash\0-c\0" + parent_body.encode() + b"\0"
    )
    packed = tmp_path / "re.json"
    packed.write_text("[]")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"] == reason
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

    (shell / "cmdline").write_bytes(b"/bin/bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


@pytest.mark.parametrize(("reason", "grep_argv"), [
    ("regex_chess_20_game_seed777_scratchpad_post_check", 'grep -E "FAIL|DONE"'),
    ("regex_chess_20_game_seed777_binary_grep_post_check", 'grep -aE "FAIL|DONE"'),
])
def test_scratchpad_fuzz_requires_suffix_relative_file_and_parent(
    tmp_path, monkeypatch, reason, grep_argv,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    parent = proc / "20"
    scratchpad = tmp_path / "dynamic-id" / "scratchpad"
    process.mkdir(parents=True)
    parent.mkdir()
    scratchpad.mkdir(parents=True)
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(
        b"\0".join((b"python3", b"fuzz.py", b"20", b"777")) + b"\0"
    )
    (process / "cwd").symlink_to(scratchpad)
    (parent / "cmdline").write_bytes(
        b"/bin/bash\0-c\0python3 fuzz.py 20 777 2>&1 | "
        + grep_argv.encode()
        + b"\0"
    )
    (scratchpad / "fuzz.py").write_text(
        "def fuzz_games(n, seed=0, maxplies=1000):\n"
        "    while not b.is_game_over() and plies < maxplies:\n"
        "        pass\n"
        '    print("DONE total=%d bad=%d tmax=%.2f" % values)\n'
        "n = int(sys.argv[1]) if len(sys.argv) > 1 else 3\n"
        "seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0\n"
    )
    policy_index = next(
        index
        for index, policy in enumerate(guard.POLICIES)
        if policy["reason"] == reason
    )
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)

    (process / "cwd").unlink()
    (process / "cwd").symlink_to(tmp_path / "not-scratchpad")
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


def test_300_game_endgame_heredoc_requires_exact_parent(
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
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0random.seed(2024)\n# endgame sprint games\n"
        b"for game in range(300):\n"
        b"    while not b.is_game_over() and b.fullmove_number < 60:\n"
        b"        pass\n"
        b"print(\"positions: %d, fails: %d, ep: %d, promo: %d, castle: %d\" "
        b"% values)\0"
    )
    packed = tmp_path / "re.json"
    packed.write_text("[]")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_300_game_endgame_heredoc_post_check"
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


def test_fixed_fen_heredoc_requires_exact_parent(tmp_path, monkeypatch):
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
    (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"bash\0-c\0"
        b'rules = json.load(open("/app/re.json"))\n'
        b'fen = "rnb1k1nr/p2p1ppp/3B4/1p1NPN1P/6P1/3P1Q2/'
        b'P1P5/q4Kb1 w kq - 0 1"\n'
        b'sorted(s.split("\\n")) == sorted(exp.split("\\n"))\0'
    )
    packed = tmp_path / "re.json"
    packed.write_text("[]")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"] == "regex_chess_fixed_fen_heredoc_post_check"
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


def test_timed_300_game_heredoc_requires_exact_grandparent(
    tmp_path, monkeypatch,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    timeout = proc / "20"
    shell = proc / "30"
    process.mkdir(parents=True)
    timeout.mkdir()
    shell.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/app")
    timeout_stat = ["S"] + ["0"] * 19
    timeout_stat[1], timeout_stat[19] = "30", str(99 * hz)
    (timeout / "stat").write_text(
        "20 (timeout) " + " ".join(timeout_stat)
    )
    (shell / "cmdline").write_bytes(
        b"bash\0-c\0timeout 3600 python3 -\n"
        b"for game in range(300):\n    stats['double_check'] += 1\n"
        b"print(\"FAIL\", fen)\0"
    )
    packed = tmp_path / "re.json"
    packed.write_text("[]")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_300_game_timed_heredoc_post_check_fuzz"
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

    (shell / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


def test_timed_120_game_heredoc_requires_exact_grandparent(
    tmp_path, monkeypatch,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    timeout = proc / "20"
    shell = proc / "30"
    process.mkdir(parents=True)
    timeout.mkdir()
    shell.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/app")
    timeout_stat = ["S"] + ["0"] * 19
    timeout_stat[1], timeout_stat[19] = "30", str(99 * hz)
    (timeout / "stat").write_text(
        "20 (timeout) " + " ".join(timeout_stat)
    )
    (shell / "cmdline").write_bytes(
        b"bash\0-c\0timeout 5000 python3 -\n"
        b"random.seed(987654)\nfor g in range(120):\n"
        b"if b.halfmove_clock > 80: break\n"
        b"print(\"soak done, tested:\", tested, \"failures:\", fails)\0"
    )
    fuzz = tmp_path / "fuzz.py"
    fuzz.write_text("def check(fen):\n    return True\n")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_120_game_timed_heredoc_post_check_fuzz"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(fuzz),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)
    (shell / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


def test_timed_6000_position_heredoc_requires_exact_grandparent(
    tmp_path, monkeypatch,
):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    process = proc / "10"
    timeout = proc / "20"
    shell = proc / "30"
    process.mkdir(parents=True)
    timeout.mkdir()
    shell.mkdir()
    (proc / "uptime").write_text("2000 0")
    stat = ["R"] + ["0"] * 19
    stat[1], stat[19] = "20", str(100 * hz)
    (process / "stat").write_text("10 (python3) " + " ".join(stat))
    (process / "comm").write_text("python3")
    (process / "cmdline").write_bytes(b"python3\0-\0")
    (process / "cwd").symlink_to("/app")
    timeout_stat = ["S"] + ["0"] * 19
    timeout_stat[1], timeout_stat[19] = "30", str(99 * hz)
    (timeout / "stat").write_text(
        "20 (timeout) " + " ".join(timeout_stat)
    )
    (shell / "cmdline").write_bytes(
        b"bash\0-c\0timeout 3000 python3 -\n"
        b"random.seed(987653)\ntarget = 6000\nwhile tested < target:\n"
        b"if b.halfmove_clock > 70: break\n"
        b"print(\"games:\", games, \"tested:\", tested, "
        b"\"real failures:\", realfails)\0"
    )
    fuzz = tmp_path / "fuzz.py"
    fuzz.write_text("def check(fen):\n    return True\n")
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_6000_position_timed_heredoc_post_check"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(fuzz),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)
    (shell / "cmdline").write_bytes(b"bash\0-c\0unrelated command\0")
    assert not guard.is_known_overvalidation(expected, proc)


def test_timed_fuzz4_requires_exact_timeout_parent(tmp_path, monkeypatch):
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
    (process / "cmdline").write_bytes(
        b"python3\0/tmp/opencode/fuzz4.py\0" b"424242\0"
    )
    (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"timeout\0" b"1750\0python3\0/tmp/opencode/fuzz4.py\0"
        b"424242\0"
    )
    script = tmp_path / "fuzz4.py"
    script.write_text(
        "for game in range(150):\n"
        "    while not b.is_game_over():\n        pass\n"
        'print("fuzz4 done: %d positions, fails %d" % (tests, fails))\n'
    )
    policies = list(guard.POLICIES)
    policy_index = next(
        index
        for index, policy in enumerate(policies)
        if policy["reason"]
        == "regex_chess_timed_150_game_fuzz4_post_check"
    )
    policies[policy_index] = {
        **policies[policy_index], "required_file": str(script),
    }
    monkeypatch.setattr(guard, "POLICIES", tuple(policies))
    expected = {
        "pid": 10,
        "start_ticks": stat[19],
        "policy_index": policy_index,
    }
    assert guard.is_known_overvalidation(expected, proc)

    (parent / "cmdline").write_bytes(
        b"python3\0/tmp/opencode/fuzz4.py\0" b"424242\0"
    )
    assert not guard.is_known_overvalidation(expected, proc)
