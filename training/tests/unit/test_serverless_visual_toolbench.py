from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from training.examples.rl.visual_toolbench import task
from training.examples.rl.visual_toolbench.rollout import make_rollout_fn
from training.examples.rl.visual_toolbench import train_visual_toolbench_serverless as serverless_vtb
from training.utils.rl.losses import has_reward_variance

# Archived per-model launchers; still parsed so they keep matching the CLI.
RUNS_DIR = Path(__file__).resolve().parents[2] / "examples" / "archive" / "launchers"


def _row(row_id: str) -> dict:
    return {
        "id": row_id,
        "prompt": "How many?",
        "golden_answer": "Six.",
        "images": ["data:image/png;base64,AAAA"],
        "rubrics": [{"description": "Says six.", "weight": 1, "critical": True}],
        "tool_alignment": {"eligible": True},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def launcher_args(script: Path) -> list[str]:
    """The launcher's ``cmd`` flags with its default shell values expanded."""
    text = script.read_text()
    defaults = dict(re.findall(r'^(\w+)="\$\{\w+:-([^}]*)\}"$', text, flags=re.M))
    start = text.index("cmd=(") + len("cmd=(")
    block = text[start : text.index("\n)\n", start)]

    def env_default(match: re.Match) -> str:
        return match.group(1)

    def arithmetic(match: re.Match) -> str:
        expression = re.sub(r"\$\{\w+:-([^}]*)\}", env_default, match.group(1))
        expression = re.sub(r"[A-Za-z_]\w*", lambda m: defaults[m.group(0)], expression)
        return str(eval(expression))  # digits and + - * only

    block = re.sub(r"\$\(\((.*?)\)\)", arithmetic, block)
    block = re.sub(r"\$\{\w+:-([^}]*)\}", env_default, block)
    block = re.sub(r"\$(\w+)", lambda m: defaults.get(m.group(1), "x"), block)
    words = shlex.split(block)
    return words[words.index("-m") + 2 :]


def test_cli_maps_to_serverless_config_and_task_config(tmp_path) -> None:
    args = serverless_vtb.parse_args(["--dataset", "train.jsonl", "--run-dir", str(tmp_path)])
    cfg = serverless_vtb.config_from_args(args)
    task_config = task.task_config_from_args(args)

    assert cfg.base_model == "accounts/fireworks/models/qwen3p6-27b"
    assert cfg.tokenizer_model == "Qwen/Qwen3.6-27B"
    assert (cfg.lora_rank, cfg.learning_rate) == (64, 3e-5)
    assert (cfg.adam_beta2, cfg.adam_epsilon, cfg.weight_decay) == (0.95, 1e-12, 0.0)
    assert (cfg.prompt_groups_per_step, cfg.completions_per_prompt) == (8, 8)
    assert cfg.max_concurrency_rollout_sample == 8
    assert cfg.min_group_size == 2
    assert (cfg.max_completion_tokens, cfg.max_seq_len) == (32768, 131072)
    assert cfg.dcp_save_interval == 2
    assert cfg.metrics_file == str(tmp_path / "metrics.jsonl")
    assert task_config.renderer_name == "qwen3_6_disable_thinking_interleaved"
    assert task_config.max_prompt_tokens == 57344
    assert task_config.filter_truncated_rollouts is True
    assert task_config.system_prompt == task.DEFAULT_SYSTEM_PROMPT
    assert task_config.eval_sample_kwargs() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 26666,
    }


def test_main_assembles_task_package_with_serverless_recipe(monkeypatch, tmp_path) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("a"), _row("b"), _row("c")])
    held_out = _write_jsonl(tmp_path / "eval.jsonl", [_row("z")])
    seen = {}

    def fake_main(config, **kwargs):
        seen.update(kwargs, config=config)
        return {"steps": 3, "final_training_checkpoint": "state/x"}

    monkeypatch.setattr(serverless_vtb.async_rl_loop_serverless, "main", fake_main)
    run_dir = tmp_path / "run"

    result = serverless_vtb.main(
        [
            "--dataset", train,
            "--eval-dataset", held_out,
            "--max-rows", "2",
            "--evaluation-interval", "3",
            "--judge-model", "accounts/fireworks/models/judge",
            "--run-dir", str(run_dir),
        ]
    )

    assert result["steps"] == 3
    assert json.loads((run_dir / "result.json").read_text()) == result
    assert [row["id"] for row in seen["rows"]] == ["a", "b"]
    factory = seen["rollout_fn_factory"]
    assert factory.func is make_rollout_fn
    assert factory.keywords["task"].judge_model == "accounts/fireworks/models/judge"
    assert seen["dynamic_filter_fn"] is has_reward_variance
    assert seen["evaluation_interval"] == 3
    assert seen["evaluation_fn"] is not None


def test_main_skips_evaluation_without_eval_dataset(monkeypatch, tmp_path) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("a")])
    seen = {}
    monkeypatch.setattr(
        serverless_vtb.async_rl_loop_serverless,
        "main",
        lambda config, **kwargs: seen.update(kwargs) or {},
    )

    serverless_vtb.main(["--dataset", train, "--run-dir", str(tmp_path / "run")])

    assert seen["evaluation_fn"] is None


@pytest.mark.parametrize(
    ("script", "model", "renderer", "groups", "concurrency", "learning_rate", "judge_concurrency"),
    [
        ("run_qwen3p6_serverless_2epoch.sh", "qwen3p6-27b", "qwen3_6_disable_thinking_interleaved", 8, 8, 3e-5, 4),
        ("run_kimi_k3_serverless_2epoch.sh", "kimi-k3", "kimi_k3_disable_thinking", 16, 8, 1e-4, 4),
        ("run_muse_glimmer_serverless_2epoch.sh", "muse-glimmer-30b", "muse_glimmer", 16, 16, 1e-4, 32),
    ],
)
def test_launchers_parse_and_keep_the_published_run_shape(
    script, model, renderer, groups, concurrency, learning_rate, judge_concurrency
) -> None:
    args = serverless_vtb.parse_args(launcher_args(RUNS_DIR / script))
    cfg = serverless_vtb.config_from_args(args)
    task_config = task.task_config_from_args(args)

    assert cfg.base_model == f"accounts/fireworks/models/{model}"
    assert task_config.renderer_name == renderer
    assert (cfg.epochs, cfg.prompt_groups_per_step, cfg.completions_per_prompt) == (2, groups, 8)
    assert cfg.max_concurrency_rollout_sample == concurrency
    assert cfg.learning_rate == learning_rate
    assert (cfg.lora_rank, cfg.adam_beta2, cfg.adam_epsilon, cfg.weight_decay) == (64, 0.95, 1e-12, 0.0)
    assert (cfg.max_completion_tokens, cfg.max_seq_len, cfg.sample_timeout) == (32768, 131072, 1800.0)
    assert cfg.dcp_save_interval == 2
    assert (args.evaluation_interval, args.eval_group_size) == (5, 1)
    assert task_config.eval_sample_kwargs() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 26666,
    }
    assert task_config.judge_max_tokens == 65536
    assert task_config.judge_max_concurrency == judge_concurrency
    assert args.require_tool_aligned_data is True
    text = (RUNS_DIR / script).read_text()
    assert "--training-shape" not in text and "--region" not in text
