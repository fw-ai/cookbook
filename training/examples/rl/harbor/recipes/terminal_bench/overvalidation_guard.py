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
        "cmdline": ["python3", "fuzz.py", "100"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": [
            "ngames = int(sys.argv[1])", "for g in range(ngames)",
            "print('tested positions:', total_pos, 'failures:', bad)",
        ],
        "reason": "regex_chess_100_game_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "25"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": [
            "rnd = random.Random(12345)",
            "ngames = int(sys.argv[1]) if len(sys.argv) > 1 else 30",
            "while not b.is_game_over(claim_draw=False)",
            'print("DONE. failures:", fail)',
        ],
        "parent_cmdline_markers": [
            "timeout", "1200", "python3", "fuzz.py", "25",
        ],
        "reason": "regex_chess_25_game_timed_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/opencode/edge.py"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/edge.py",
        "required_markers": [
            "from fuzz import check", "for f in fens:",
            'print("DONE", "ALL OK" if ok else "FAILURES", len(fens))',
        ],
        "reason": "regex_chess_edge_case_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "process_name": "python",
        "cmdline": ["python", "-"],
        "cwd": "/app",
        "required_file": "/app/re.json",
        "required_markers": [],
        "parent_cmdline_markers": [
            "for g in range(400):", "def verify(fen):",
            "random positions tested:",
        ],
        "reason": "regex_chess_400_game_heredoc_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode",
        "required_file": "/app/re.json",
        "required_markers": [],
        "parent_cmdline_markers": [
            "random.seed(2029)", "for game in range(400):",
            "while not b.is_game_over(claim_draw=False)",
            'print("random standard games OK:", n, "positions in"',
        ],
        "reason": "regex_chess_400_game_tmp_heredoc_post_check_fuzz",
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
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "random.seed(12345)", "for g in range(4000):",
            "for ply in range(120):",
            'print("fuzz done: %d games, %d white positions, %d failures"',
        ],
        "reason": "regex_chess_4000_game_post_check_fuzz",
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
        "cmdline": ["python3", "fuzz.py", "30", "7"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20",
            "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
            "while not b.is_game_over() and b.fullmove_number < 70",
            'print("tested positions:", tested + len(specials)',
        ],
        "parent_cmdline_markers": [
            "python3 build_re.py", "python3 fuzz.py 30 7", "tail -8",
        ],
        "reason": "regex_chess_30_game_seed7_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "30", "7"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20",
            "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
            "while not b.is_game_over() and b.fullmove_number < 70",
            'print("tested positions:", tested + len(specials)',
        ],
        "parent_cmdline_markers": [
            "python3 build_re.py", "python3 fuzz.py 30 7",
            "/tmp/fuzzlog.txt", "COUNT MISMATCH", "tail -2",
        ],
        "reason": "regex_chess_30_game_seed7_logged_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "200", "99"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20",
            "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
            "while not b.is_game_over() and b.fullmove_number < 70",
            'print("tested positions:", tested + len(specials)',
        ],
        "parent_cmdline_markers": [
            "python3 fuzz.py 200 99", "/tmp/fuzzbig.txt",
            "COUNT MISMATCH", "tail -2",
        ],
        "reason": "regex_chess_200_game_seed99_logged_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "400", "12345"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20",
            "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
            "while not b.is_game_over() and b.fullmove_number < 70",
            'print("tested positions:", tested + len(specials)',
        ],
        "parent_cmdline_markers": [
            "python3 fuzz.py 400 12345", "/tmp/fuzzgames.txt",
            "COUNT MISMATCH", "tail -2",
        ],
        "reason": "regex_chess_400_game_seed12345_logged_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "300", "12345"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20",
            "seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
            "while not b.is_game_over() and b.fullmove_number < 70",
            'print("tested positions:", tested + len(specials)',
        ],
        "reason": "regex_chess_orphan_300_game_seed12345_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-u", "fuzz2.py", "10"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz2.py",
        "required_markers": [
            "rnd = random.Random(999)", "g = int(sys.argv[1])",
            "while not b.is_game_over(claim_draw=False)",
            'print("ALL OK. positions:", pos, flush=True)',
        ],
        "parent_cmdline_markers": [
            "timeout", "800", "python3", "-u", "fuzz2.py", "10",
        ],
        "reason": "regex_chess_10_game_timed_post_check_fuzz",
    },
    *(
        {
            "task_prefix": "harbor-opencode-regex-chess-",
            "min_elapsed_s": 600,
            "cmdline": ["python3", "fuzz_par.py", str(seed), "300"],
            "cwd": "/app",
            "required_file": "/app/fuzz_par.py",
            "required_markers": [
                "seed = int(sys.argv[1])", "ngames = int(sys.argv[2])",
                "for ply in range(120):",
                'log.write("DONE seed=%d games=%d positions=%d fails=%d',
            ],
            "reason": f"regex_chess_parallel_300_game_seed{seed}_post_check",
        }
        for seed in (111, 222, 333, 444)
    ),
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz_par.py", "777", "150"],
        "cwd": "/app",
        "required_file": "/app/fuzz_par.py",
        "required_markers": [
            "seed = int(sys.argv[1])", "ngames = int(sys.argv[2])",
            "for ply in range(120):",
            'log.write("DONE seed=%d games=%d positions=%d fails=%d',
        ],
        "parent_cmdline_markers": [
            "python3 fuzz_par.py 777 150", "2>&1", "tail -1",
        ],
        "reason": "regex_chess_parallel_150_game_seed777_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "process_name": "sleep",
        "cmdline": ["sleep", "900"],
        "cwd": "/app",
        "required_file": "/app/fuzz_par.py",
        "required_markers": [
            "seed = int(sys.argv[1])", "ngames = int(sys.argv[2])",
            "for ply in range(120):",
        ],
        "parent_cmdline_markers": [
            "sleep 900", "fuzz_111.log", "fuzz_222.log",
            "fuzz_333.log", "fuzz_444.log", "grep -h FAIL",
        ],
        "reason": "regex_chess_parallel_fuzz_completion_wait",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "process_name": "sleep",
        "cmdline": ["sleep", "1500"],
        "cwd": "/app",
        "required_file": "/app/fuzz_par.py",
        "required_markers": [
            "seed = int(sys.argv[1])", "ngames = int(sys.argv[2])",
            "for ply in range(120):",
        ],
        "parent_cmdline_markers": [
            "sleep 1500", "fuzz_111.log", "fuzz_222.log",
            "fuzz_333.log", "fuzz_444.log", "grep -h FAIL",
            "fuzz_*.log", "ep_*.log",
        ],
        "reason": "regex_chess_parallel_fuzz_repeated_completion_wait",
    },
    *(
        {
            "task_prefix": "harbor-opencode-regex-chess-",
            "min_elapsed_s": 600,
            "cmdline": [
                "python3", "-u", "checkfens.py", f"part_{part:02d}",
            ],
            "cwd": "/tmp/opencode",
            "required_file": "/tmp/opencode/checkfens.py",
            "required_markers": [
                "for line in open(sys.argv[1])", "board = chess.Board(fen)",
                'print(sys.argv[1], "fails:", fails)',
            ],
            "parent_cmdline_markers": [
                "timeout", "1400", "python3", "-u", "checkfens.py",
                f"part_{part:02d}",
            ],
            "reason": (
                f"regex_chess_part_{part:02d}_fen_post_check"
            ),
        }
        for part in range(8)
    ),
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
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "random.seed(12345)", "NGAMES = 60",
            "while not b.is_game_over() and b.fullmove_number < 100",
            'print("total: %d, fails: %d" % (total, fails))',
        ],
        "parent_cmdline_markers": [
            "timeout", "3000", "python3", "fuzz.py",
        ],
        "reason": "regex_chess_60_game_timed_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "random.seed(12345)", "NGAMES = 60",
            "while not b.is_game_over() and b.fullmove_number < 100",
            'print("total: %d, fails: %d" % (total, fails))',
        ],
        "parent_cmdline_markers": [
            "cd /app", "python3 -", "import fuzz", "tail -10",
        ],
        "reason": "regex_chess_60_game_import_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "random.seed(12345)", "for g in range(60):",
            "for ply in range(300):",
            'print("random games done:", ntested, "positions, failures:"',
        ],
        "parent_cmdline_markers": [
            "timeout", "6000", "python3", "fuzz.py",
        ],
        "reason": "regex_chess_60_game_300_ply_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/opencode/fuzz2.py", "999"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/fuzz2.py",
        "required_markers": [
            "GAMES = 40", "for g in range(GAMES):",
            'print("fuzz2 done: %d white positions, fails %d"',
        ],
        "reason": "regex_chess_40_game_fuzz2_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        # Observed only after the same seed/script had already crossed the
        # normal 600s guard and then been relaunched under successively shorter
        # timeout wrappers.  Requiring that exact timed script/seed signature
        # distinguishes the repeated post-check from other validation work.
        "min_elapsed_s": 120,
        "cmdline": ["python3", "/tmp/opencode/fuzz4.py", "424242"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/fuzz4.py",
        "required_markers": [
            "for game in range(150):", "while not b.is_game_over():",
            'print("fuzz4 done: %d positions, fails %d"',
        ],
        "parent_cmdline_markers": [
            "timeout", "/tmp/opencode/fuzz4.py", "424242",
        ],
        "reason": "regex_chess_timed_150_game_fuzz4_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/opencode/fuzz4.py", "424242"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/fuzz4.py",
        "required_markers": [
            "for game in range(150):", "while not b.is_game_over():",
            'print("fuzz4 done: %d positions, fails %d"',
        ],
        "reason": "regex_chess_150_game_fuzz4_post_check",
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
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode",
        "required_file": "/app/re.json",
        "required_markers": [],
        "ancestor_hops": 2,
        "ancestor_cmdline_markers": [
            "timeout 1200 python3 -", "for seed in range(1000, 1006)",
            "for g in range(15)", "b.fullmove_number < 150",
            'print("SOAK OK, ntest =", ntest)',
        ],
        "reason": "regex_chess_90_game_seed_range_heredoc_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode",
        "required_file": "/app/re.json",
        "required_markers": [],
        "ancestor_hops": 2,
        "ancestor_cmdline_markers": [
            "timeout 3500 python3 -", "rng = random.Random(1337)",
            "for g in range(400)", "b.ply() < 160",
            "interesting-games=",
        ],
        "reason": "regex_chess_400_game_seed1337_heredoc_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode",
        "required_file": "/app/re.json",
        "required_markers": [],
        "ancestor_hops": 2,
        "ancestor_cmdline_markers": [
            "timeout 3500 python3 -", "rng = random.Random(1337)",
            "for g in range(250)", "b.ply() < 160",
            "interesting={interesting}",
        ],
        "reason": "regex_chess_250_game_seed1337_heredoc_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "20", "777"],
        "cwd_suffix": "/scratchpad",
        "required_file_relative": "fuzz.py",
        "required_markers": [
            "def fuzz_games(n, seed=0, maxplies=1000)",
            "while not b.is_game_over() and plies < maxplies",
            'print("DONE total=%d bad=%d tmax=%.2f"',
            "n = int(sys.argv[1]) if len(sys.argv) > 1 else 3",
            "seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
        ],
        "parent_cmdline_markers": [
            "python3 fuzz.py 20 777", 'grep -E "FAIL|DONE"',
        ],
        "reason": "regex_chess_20_game_seed777_scratchpad_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "20", "777"],
        "cwd_suffix": "/scratchpad",
        "required_file_relative": "fuzz.py",
        "required_markers": [
            "def fuzz_games(n, seed=0, maxplies=1000)",
            "while not b.is_game_over() and plies < maxplies",
            'print("DONE total=%d bad=%d tmax=%.2f"',
            "n = int(sys.argv[1]) if len(sys.argv) > 1 else 3",
            "seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0",
        ],
        "parent_cmdline_markers": [
            "python3 fuzz.py 20 777", 'grep -aE "FAIL|DONE"',
        ],
        "reason": "regex_chess_20_game_seed777_binary_grep_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzzpar.py", "160"],
        "cwd": "/app",
        "required_file": "/app/fuzzpar.py",
        "required_markers": [
            "n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 160",
            "seeds = list(range(1000, 1000 + n_games))",
            "all_games = pool.map(gen_game_fens, seeds)",
            'print(f"DONE tested={total_tested} fails={len(total_fails)}',
        ],
        "parent_cmdline_markers": ["python3 fuzzpar.py 160"],
        "reason": "regex_chess_parallel_160_game_worker_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "randtest.py", "3"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/randtest.py",
        "required_markers": [
            "random.seed(int(sys.argv[1])", "for g in range(NGAMES)",
            "for ply in range(250)",
            'print(f"ALL PASS: {ntests} positions in {time.time()-t0:.1f}s")',
        ],
        "parent_cmdline_markers": [
            "for seed in 1 2 3", "python3 randtest.py $seed", "tail -1",
        ],
        "reason": "regex_chess_seed3_random_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "randtest.py", "101"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/randtest.py",
        "required_markers": [
            "random.seed(int(sys.argv[1])", "for g in range(NGAMES)",
            "for ply in range(250)",
            'print(f"ALL PASS: {ntests} positions in {time.time()-t0:.1f}s")',
        ],
        "parent_cmdline_markers": [
            "for seed in 101 202", "python3 randtest.py $seed", "tail -1",
        ],
        "reason": "regex_chess_seed101_random_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "randtest.py", "202"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/randtest.py",
        "required_markers": [
            "random.seed(int(sys.argv[1])", "for g in range(NGAMES)",
            "for ply in range(250)",
            'print(f"ALL PASS: {ntests} positions in {time.time()-t0:.1f}s")',
        ],
        "parent_cmdline_markers": [
            "for seed in 101 202", "python3 randtest.py $seed", "tail -1",
        ],
        "reason": "regex_chess_seed202_random_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode",
        "required_file": "/app/re.json",
        "required_markers": [],
        "parent_cmdline_markers": [
            "for it in range(6000):", "skip zero-move inputs",
            "print('total:', tests, 'failures:', fails)",
        ],
        "reason": "regex_chess_12000_position_heredoc_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/re.json",
        "required_markers": [],
        "parent_cmdline_markers": [
            'rules = json.load(open("/app/re.json"))',
            'fen = "rnb1k1nr/p2p1ppp/3B4/1p1NPN1P/6P1/3P1Q2/P1P5/q4Kb1 w kq - 0 1"',
            'sorted(s.split("\\n")) == sorted(exp.split("\\n"))',
        ],
        "reason": "regex_chess_fixed_fen_heredoc_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/re.json",
        "required_markers": [],
        "ancestor_hops": 2,
        "ancestor_cmdline_markers": [
            "timeout 3600 python3 -", "for game in range(300):",
            "stats['double_check']", 'print("FAIL", fen)',
        ],
        "reason": "regex_chess_300_game_timed_heredoc_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": ["def check"],
        "ancestor_hops": 2,
        "ancestor_cmdline_markers": [
            "timeout 5000 python3 -", "random.seed(987654)",
            "for g in range(120):", "b.halfmove_clock > 80",
            'print("soak done, tested:", tested, "failures:", fails)',
        ],
        "reason": "regex_chess_120_game_timed_heredoc_post_check_fuzz",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": ["def check"],
        "ancestor_hops": 2,
        "ancestor_cmdline_markers": [
            "timeout 3000 python3 -", "random.seed(987653)",
            "target = 6000", "while tested < target",
            "b.halfmove_clock > 70",
            'print("games:", games, "tested:", tested, "real failures:"',
        ],
        "reason": "regex_chess_6000_position_timed_heredoc_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/re.json",
        "required_markers": [],
        "parent_cmdline_markers": [
            "random.seed(2024)", "for game in range(300)",
            "endgame sprint games", "b.fullmove_number < 60",
            'print("positions: %d, fails: %d, ep: %d, promo: %d, castle:',
        ],
        "reason": "regex_chess_300_game_endgame_heredoc_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/fuzz2.py"],
        "cwd": "/app",
        "required_file": "/tmp/fuzz2.py",
        "required_markers": [
            "rnd = random.Random(999)",
            "while n < 400:",
            "while n2 < 400:",
            'print("OK" if fuzz.FAILS == 0 else "FAIL")',
        ],
        "parent_cmdline_markers": [
            "cat > /tmp/fuzz2.py",
            "time python3 /tmp/fuzz2.py",
        ],
        "reason": "regex_chess_ep_castling_800_case_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "random.seed(12345)",
            "N_GAMES = 40",
            "board.fullmove_number < 80",
            'print("TOTAL tested", tested, "fails", fails)',
        ],
        "parent_cmdline_markers": [
            "cd /app && python3 gen.py && python3 fuzz.py",
            "tail -12",
        ],
        "reason": "regex_chess_40_game_generated_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "1", "40"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": [
            "NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50",
            "b.fullmove_number < 100",
            'print("tested", tested, "positions, fails", fails',
        ],
        "parent_cmdline_markers": [
            "timeout", "1800", "python3", "fuzz.py", "1", "40",
        ],
        "reason": "regex_chess_seed1_40_game_timed_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/opencode/edge.py"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/edge.py",
        "required_markers": [
            "from fuzz import check",
            "for fen in cases:",
            'print("edge cases:", "ALL OK" if ok else "FAILURES")',
        ],
        "reason": "regex_chess_extended_edge_case_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/app",
        "required_file": "/app/re.json",
        "required_markers": [],
        "parent_cmdline_markers": [
            "rng = random.Random(1234)",
            "while tested < 600:",
            "from fuzz import check_ok",
            "composed fails:",
        ],
        "reason": "regex_chess_600_composed_position_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "process_name": "python",
        "cmdline": ["python", "bigtest.py"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/bigtest.py",
        "required_markers": [
            "rng = random.Random(99)",
            "for game in range(120):",
            "b.fullmove_number < 100",
            "print('ALL OK')",
        ],
        "parent_cmdline_markers": [
            "timeout", "1200", "python", "bigtest.py",
        ],
        "reason": "regex_chess_120_game_bigtest_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "/tmp/fuzz2.py"],
        "cwd": "/app",
        "required_file": "/tmp/fuzz2.py",
        "required_markers": [
            "rnd = random.Random(999)",
            "while n < 60:",
            'print("ep suite done, FAILS:", fuzz.FAILS)',
        ],
        "parent_cmdline_markers": [
            "cat > /tmp/fuzz2.py",
            "time python3 /tmp/fuzz2.py",
        ],
        "reason": "regex_chess_ep_60_case_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py"],
        "cwd": "/app",
        "required_file": "/app/fuzz.py",
        "required_markers": [
            "random.seed(12345)",
            "N_GAMES = 40",
            "board.fullmove_number < 80",
            'print("TOTAL tested", tested, "fails", fails)',
        ],
        "parent_cmdline_markers": [
            "timeout", "1800", "python3", "fuzz.py",
        ],
        "reason": "regex_chess_40_game_timed_generated_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "42", "120"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": [
            "NGAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 50",
            "b.fullmove_number < 100",
            'print("tested", tested, "positions, fails", fails',
        ],
        "parent_cmdline_markers": [
            "timeout", "7000", "python3", "fuzz.py", "42", "120",
        ],
        "reason": "regex_chess_seed42_120_game_timed_post_check",
    },
    {
        "task_prefix": "harbor-opencode-regex-chess-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "fuzz.py", "1500"],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/fuzz.py",
        "required_markers": [
            "random.seed(12345)",
            "ngames = int(sys.argv[1]) if len(sys.argv) > 1 else 200",
            "for ply in range(200):",
            'print("positions tested:", npos, "bad:", nbad)',
        ],
        "reason": "regex_chess_1500_game_post_check",
    },
    {
        "task_prefix": "harbor-opencode-filter-js-from-html-",
        "min_elapsed_s": 600,
        "cmdline": ["python3", "-"],
        "cwd": "/tmp/opencode/xss",
        "required_file": "/tmp/opencode/xss/big.html",
        "required_markers": [],
        "parent_cmdline_markers": [
            "https://html.spec.whatwg.org/",
            "difflib.SequenceMatcher(None, orig, after, autojunk=False)",
            'print("opcodes:", len(sm.get_opcodes()), "non-script removals:", nonscript)',
        ],
        "reason": "filter_js_whatwg_quadratic_sequence_matcher_post_check",
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
    {
        "task_prefix": "harbor-opencode-large-scale-text-editing-",
        "min_elapsed_s": 600,
        "process_name": "vim",
        "cmdline": [
            "vim", "-Nu", "NONE", "-n", "-Es", "/tmp/opencode/t.csv",
            "-S", "/tmp/opencode/dbg",
        ],
        "cwd": "/app",
        "required_file": "/tmp/opencode/dbg",
        "required_markers": [r":s/\v^\s*", ":wq"],
        "parent_cmdline_markers": [
            "vim -Nu NONE -n -Es /tmp/opencode/t.csv -S /tmp/opencode/dbg",
            "cat /tmp/opencode/t.csv",
        ],
        "reason": "large_scale_text_editing_noninteractive_vim_stall",
    },
    {
        "task_prefix": "harbor-opencode-llm-inference-batching-scheduler-",
        "min_elapsed_s": 300,
        "cmdline": ["python3", "/tmp/opencode/tune4.py"],
        "cwd": "/app",
        "required_file": "/tmp/opencode/tune4.py",
        "required_markers": [
            "for (lam1, mu1, nu1, lam2, mu2, nu2) in [",
            "X.build_plans(",
            "X.report(",
        ],
        "parent_cmdline_markers": [
            "cat > /tmp/opencode/tune4.py",
            "time python3 /tmp/opencode/tune4.py",
        ],
        "reason": "scheduler_repeated_post_solution_tune4_sweep",
    },
    {
        "task_prefix": "harbor-opencode-llm-inference-batching-scheduler-",
        "min_elapsed_s": 300,
        "cmdline_prefix": ["python3", "-c"],
        "cmdline_markers": [
            "for lam_seq in [20, 25, 30, 35, 40, 50, 60, 70]",
            "for lam_pad in [0.0, 1e4, 3e4]",
            "for lam95 in [1e10, 1e8, 1e7]",
            "print('BEST b1:', best)",
        ],
        "cwd": "/tmp/opencode",
        "required_file": "/tmp/opencode/optimize2.py",
        "required_markers": ["def solve_bucket", "def evaluate"],
        "parent_cmdline_markers": [
            "timeout 1800 python3 -c",
            "tail -40",
        ],
        "reason": "scheduler_72_configuration_post_solution_grid",
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
    cmdline = [
        os.fsdecode(arg)
        for arg in (process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    ]
    expected_cmdline = policy.get("cmdline")
    if expected_cmdline is not None and cmdline != expected_cmdline:
        return False
    prefix = policy.get("cmdline_prefix")
    if prefix is not None and cmdline[:len(prefix)] != prefix:
        return False
    cmdline_text = " ".join(cmdline)
    if not all(marker in cmdline_text
               for marker in policy.get("cmdline_markers", [])):
        return False
    cwd = os.readlink(process / "cwd")
    if policy.get("cwd") is not None:
        if cwd != policy["cwd"]:
            return False
    elif not cwd.endswith(policy["cwd_suffix"]):
        return False
    if policy.get("required_file") is not None:
        required = Path(policy["required_file"])
    else:
        required = Path(cwd) / policy["required_file_relative"]
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
    if policy.get("ancestor_cmdline_markers"):
        ancestor_pid = stat[1]
        for _ in range(policy["ancestor_hops"] - 1):
            ancestor_stat = (
                (proc_root / ancestor_pid / "stat")
                .read_text().rsplit(")", 1)[1].split()
            )
            ancestor_pid = ancestor_stat[1]
        ancestor_cmdline = os.fsdecode(
            (proc_root / ancestor_pid / "cmdline")
            .read_bytes().replace(b"\0", b" ")
        )
        if not all(
            marker in ancestor_cmdline
            for marker in policy["ancestor_cmdline_markers"]
        ):
            return False
    return True


def signal_known_overvalidation(expected, proc_root=Path("/proc")):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected["pid"])
        if not is_known_overvalidation(expected, proc_root):
            return {"action": "none", "reason": "evidence_changed"}
        ignored = 0
        try:
            for line in (
                proc_root / str(expected["pid"]) / "status"
            ).read_text().splitlines():
                if line.startswith("SigIgn:"):
                    ignored = int(line.split(":", 1)[1].strip(), 16)
                    break
        except OSError:
            pass
        interrupt_ignored = bool(ignored & (1 << (signal.SIGINT - 1)))
        selected_signal = signal.SIGTERM if interrupt_ignored else signal.SIGINT
        signal.pidfd_send_signal(descriptor, selected_signal)
        policy = POLICIES[expected["policy_index"]]
        return {
            "action": signal.Signals(selected_signal).name,
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
