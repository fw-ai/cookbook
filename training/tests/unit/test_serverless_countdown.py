from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import training.examples.archive.countdown.countdown_rl as countdown_rl
from fireworks.training import sdk
from training.examples.archive.countdown.countdown_rewards import composite_reward
from training.examples.archive.countdown.prepare_data import countdown_row
from training.recipes.experiment import async_rl_loop_serverless as loop
from training.tests.unit.test_serverless_visual_toolbench import RUNS_DIR, launcher_args
from training.utils.rl.rollout import RolloutRun, RolloutSample

CORRECT = "<think>3 * 2 = 6</think><answer>3 * 2</answer>"
WRONG = "<answer>3 + 2</answer>"
_TEXT_BY_TOKEN = {101: CORRECT, 102: WRONG}


class _Renderer:
    def build_generation_prompt(self, messages):
        # The last prompt token identifies the row so the fake sampler can
        # alternate answers within each prompt group.
        assert messages[-1]["role"] == "user"
        return sdk.ModelInput.from_ints([1, 2, len(messages[-1]["content"])])

    def parse_response(self, tokens):
        return {"role": "assistant", "content": _TEXT_BY_TOKEN[tokens[0]]}, True

    def get_stop_sequences(self):
        return ["</s>"]


class _DeploymentSampler:
    model = "snapshot"
    base_url = "https://api.example"
    tokenizer = object()

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.calls_per_prompt: dict[tuple[int, ...], int] = {}

    async def sample_with_prompt_tokens(self, prompt_token_ids, **kwargs):
        self.calls.append(kwargs)
        key = tuple(prompt_token_ids)
        self.calls_per_prompt[key] = self.calls_per_prompt.get(key, 0) + 1
        token = 101 if self.calls_per_prompt[key] % 2 else 102
        return [
            SimpleNamespace(
                prompt_len=len(prompt_token_ids),
                full_tokens=[*prompt_token_ids, token, 7],
                sampling_logprobs=[-0.1, -0.2],
                inference_logprobs=[-0.1, -0.2],
                logprobs_echoed=False,
                routing_matrices=None,
                finish_reason="stop",
                text="",
            )
        ]


def _setup(sampler) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=object(),
        tokenizer_id="moonshotai/Kimi-K3",
        sample_kwargs={"max_tokens": 64, "temperature": 1.0, "logprobs": True},
        sampler=sampler,
    )


def test_rollout_scores_one_rendered_completion(monkeypatch) -> None:
    renderers = []
    monkeypatch.setattr(
        countdown_rl,
        "build_renderer",
        lambda tokenizer, tokenizer_id, name: renderers.append((tokenizer_id, name)) or _Renderer(),
    )
    sampler = _DeploymentSampler()
    rollout_fn = countdown_rl.make_rollout_fn(_setup(sampler), renderer_name="kimi_k3")
    row = countdown_row([3, 2], 6)

    correct = asyncio.run(rollout_fn(row))
    wrong = asyncio.run(rollout_fn(row))

    assert renderers == [("moonshotai/Kimi-K3", "kimi_k3")]
    assert correct.segments[0].reward == composite_reward(CORRECT, row["ground_truth"]) == 1.0
    assert wrong.segments[0].reward == composite_reward(WRONG, row["ground_truth"])
    assert correct.segments[0].tokens[-2:] == [101, 7]
    assert correct.segments[0].loss_mask == [0, 0, 0, 1, 1]
    assert correct.segments[0].logprobs[-2:] == [-0.1, -0.2]
    assert sampler.calls[0]["n"] == 1
    assert sampler.calls[0]["stop"] == ["</s>"]
    assert sampler.calls[0]["max_tokens"] == 64


def test_eval_metrics_count_dropped_samples_against_accuracy() -> None:
    def run(reward):
        return RolloutRun(
            segments=[RolloutSample(tokens=[1, 2], logprobs=[0.0, -0.1], loss_mask=[0, 1], reward=reward)]
        )

    metrics = countdown_rl.countdown_eval_metrics([{}, {}], [[run(1.0), run(0.3)], [None, run(1.0)]])

    assert metrics == {
        "eval/raw_reward": (1.0 + 0.3 + 1.0) / 3,
        "eval/accuracy": 0.5,
        "eval/samples": 3,
        "eval/return_ratio": 0.75,
    }


def test_cli_defaults_map_to_serverless_recipe_config(tmp_path) -> None:
    args = countdown_rl.parse_args(["--run-dir", str(tmp_path)])
    cfg = countdown_rl.config_from_args(args)

    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert cfg.max_rows == 320
    assert cfg.prompt_groups_per_step == 16
    assert cfg.completions_per_prompt == 8
    assert cfg.max_concurrency_rollout_sample == 64
    assert (cfg.lora_rank, cfg.lora_alpha) == (32, 64)
    assert cfg.learning_rate == 1e-4
    assert cfg.weight_decay == 0.0
    assert cfg.shuffle is False
    assert cfg.metrics_file == str(tmp_path / "metrics.jsonl")
    assert args.eval_rows == 16
    assert args.evaluation_interval == 5


def test_train_holds_out_leading_rows_and_assembles_the_recipe(monkeypatch, tmp_path) -> None:
    dataset = tmp_path / "countdown.jsonl"
    dataset.write_text("".join(json.dumps(countdown_row([i, 1], i + 1)) + "\n" for i in range(10)))
    seen = {}

    def fake_main(config, **kwargs):
        seen.update(kwargs, config=config)
        return {"steps": 1}

    monkeypatch.setattr(countdown_rl.async_rl_loop_serverless, "main", fake_main)
    cfg = loop.Config(dataset=str(dataset), max_rows=5, completions_per_prompt=4, min_group_size=2)

    assert countdown_rl.train(cfg, renderer_name="r", eval_rows=2, evaluation_interval=3) == {"steps": 1}
    assert seen["config"] is cfg
    assert [row["ground_truth"] for row in seen["rows"]] == [
        json.dumps({"numbers": [i, 1], "target": i + 1}) for i in range(2, 7)
    ]
    assert seen["rollout_fn_factory"].func is countdown_rl.make_rollout_fn
    assert seen["rollout_fn_factory"].keywords == {"renderer_name": "r"}
    assert seen["dynamic_filter_fn"] is countdown_rl.has_reward_variance
    assert seen["evaluation_interval"] == 3
    assert seen["evaluation_fn"] is not None


def test_main_runs_the_real_serverless_recipe(monkeypatch, tmp_path) -> None:
    """Countdown task code trains through the unmodified serverless recipe."""

    class TrainingClient:
        run_id = "run-test"

        def __init__(self) -> None:
            self.forward_backward_calls = []
            self.optim_calls = 0
            self.saved_states = []

        def save_weights_for_sampler(self, name):
            return SimpleNamespace(result=lambda: SimpleNamespace(path=f"snapshot/{name}"))

        def save_state(self, name):
            self.saved_states.append(name)
            return SimpleNamespace(result=lambda: SimpleNamespace(path=f"state/{name}"))

        def forward_backward(self, datums, loss_fn, loss_fn_config=None):
            self.forward_backward_calls.append((len(datums), loss_fn))
            outputs = [
                {
                    "logprobs": sdk.TensorData(
                        data=[-0.15] * len(d.loss_fn_inputs["target_tokens"].data),
                        dtype="float32",
                        shape=[len(d.loss_fn_inputs["target_tokens"].data)],
                    )
                }
                for d in datums
            ]
            result = SimpleNamespace(metrics={"loss:sum": 1.0}, loss_fn_outputs=outputs)

            async def result_async(timeout=None):
                return result

            return SimpleNamespace(result=lambda: result, result_async=result_async)

        def optim_step(self, params, **kwargs):
            self.optim_calls += 1
            return SimpleNamespace(result=lambda: SimpleNamespace(metrics={}))

    class Service:
        training_session_name = "accounts/test/trainingSessions/s"

        def __init__(self) -> None:
            self.training_client = TrainingClient()
            self.lora = None

        def create_lora_training_client(self, *, base_model, rank, alpha):
            self.lora = (base_model, rank, alpha)
            return self.training_client

        def create_sampling_client(self, *, model_path, tokenizer):
            return SimpleNamespace(deployment_sampler=_DeploymentSampler(), close=lambda: None)

        def close(self) -> None:
            pass

    service = Service()
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(loop, "_make_service", lambda *_args: service)
    monkeypatch.setattr(loop, "load_tokenizer", lambda *_args: object())
    monkeypatch.setattr(loop, "_router_replay_enabled", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(loop, "setup_wandb", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(loop, "wandb_finish", lambda **_kwargs: None)
    monkeypatch.setattr(countdown_rl, "build_renderer", lambda *_args: _Renderer())
    rows = []
    for index in range(6):
        row = countdown_row([3, 2], 6)
        row["messages"][-1]["content"] += " " * index  # distinct prompt per row
        rows.append(row)
    dataset = tmp_path / "countdown.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run_dir = tmp_path / "run"

    result = countdown_rl.main(
        [
            "--dataset",
            str(dataset),
            "--eval-rows",
            "2",
            "--max-rows",
            "4",
            "--prompt-groups-per-step",
            "2",
            "--completions-per-prompt",
            "2",
            "--max-concurrency-rollout-sample",
            "4",
            "--max-completion-tokens",
            "16",
            "--max-seq-len",
            "64",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert service.lora == ("accounts/fireworks/models/kimi-k3", 32, 64)
    assert result["steps"] == 2
    assert service.training_client.optim_calls == 2
    assert {loss for _, loss in service.training_client.forward_backward_calls} == {"ppo"}
    assert json.loads((run_dir / "result.json").read_text()) == result
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("rollout/raw_reward" in record for record in records)
    eval_records = [record for record in records if "eval/raw_reward" in record]
    assert eval_records and all(record["eval/return_ratio"] == 1.0 for record in eval_records)


def test_k3_launcher_parses_to_the_default_countdown_run() -> None:
    args = countdown_rl.parse_args(launcher_args(RUNS_DIR / "run_countdown_k3_serverless.sh"))
    cfg = countdown_rl.config_from_args(args)

    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.max_rows == 20 * 16
    assert (cfg.lora_rank, cfg.lora_alpha) == (32, 64)
    assert args.snapshot_prefix == "countdown-k3-x"


def test_dsv4_launcher_parses_to_its_published_run_shape() -> None:
    args = countdown_rl.parse_args(launcher_args(RUNS_DIR / "run_countdown_dsv4_flash_serverless.sh"))
    cfg = countdown_rl.config_from_args(args)

    assert cfg.base_model == "accounts/fireworks/models/deepseek-v4-flash-0731"
    assert cfg.tokenizer_model == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert cfg.max_rows == 30 * 16
    assert (cfg.prompt_groups_per_step, cfg.completions_per_prompt) == (16, 8)
    assert cfg.max_concurrency_rollout_sample == 8 * 8
    assert (cfg.lora_rank, cfg.lora_alpha) == (32, 32)
    assert (cfg.max_completion_tokens, cfg.max_seq_len) == (32500, 32768)
    assert (cfg.sample_timeout, cfg.dcp_save_interval) == (3600.0, 5)
    assert (args.eval_rows, args.evaluation_interval) == (16, 5)
    assert cfg.init_from_checkpoint is None
