from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import eval_protocol.integrations.fireworks_v1_completions_client as ep_client
import pytest
import training.examples.rl.multihop_qa.multihop_qa_rollout as rollout_module
import training.examples.rl.multihop_qa.train_multihop_qa_igpo as train_module
from eval_protocol.pytest.types import RolloutProcessorConfig
from fireworks.training import sdk
from training.examples.rl.multihop_qa.multihop_qa_rollout import (
    MultiHopQARolloutProcessor,
    parse_search_tool_call,
)
from training.examples.rl.multihop_qa.train_multihop_qa_igpo import (
    MultiHopQATask,
    episode_row,
    episode_to_run,
    make_rollout_fn_factory,
    parse_args,
)
from training.recipes import rl_loop
from training.utils import DeployConfig, TrainerConfig, build_renderer
from training.utils.rl.algorithm.igpo import ANSWER_TOKENS, TURN_SPANS
from training.utils.rl.rollout import RolloutRun, RolloutSample
from training.utils.rl.rollout.setup import RolloutSetup
from training.utils.tokenizers import load_tokenizer

ROW = {
    "messages": [
        {"role": "system", "content": "Search, then answer."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "ground_truth": "Paris",
    "context": {
        "titles": ["France"],
        "paragraphs": [["Paris is the capital of France."]],
    },
}
SEARCH = '<tool_call>\n{"name": "search", "arguments": {"query": "capital of France"}}\n</tool_call>'
SUBMIT = '{"name": "submit_answer", "arguments": {"answer": "Paris"}}<|im_end|>'
TEXT_BY_TOKENS = {
    (100,): SEARCH,
    (101, 90): SUBMIT,
    (102,): "I am not sure.",
    (103,): '{"name": "browse", "arguments": {}}',
}


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return TEXT_BY_TOKENS[tuple(token_ids)]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, **kwargs):
        assert [message["role"] for message in messages] == ["tool"]
        return "<tool reply + assistant opener>"

    def encode(self, text, add_special_tokens=False):
        return {"<tool reply + assistant opener>": [20, 21], "Paris": [55], "": []}[text]


class FakeRenderer:
    """Parses the JSON tool call in the tokenizer's decoded text."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parse_response(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        if "{" not in text:
            return {"role": "assistant", "content": text}, "stop"
        call, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        function = SimpleNamespace(name=call["name"], arguments=json.dumps(call["arguments"]))
        return {"role": "assistant", "content": "", "tool_calls": [SimpleNamespace(function=function)]}, "stop"


class FakeCompletionsClient:
    """Answers with the queued completions; frames turns with fixed tokens."""

    queued: list[tuple[list[int], list[float]]] = []
    requests: list[list[int]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build_prompt_token_ids(self, *, messages, tools):
        assert messages[0]["role"] == "system" and tools
        return [1, 2, 3]

    def encode_special_suffix(self):
        return [90, 91]

    async def create_completion_from_prompt_ids(self, *, prompt_token_ids):
        type(self).requests.append(list(prompt_token_ids))
        completion_ids, logprobs = type(self).queued.pop(0)
        return {
            "completion_ids": completion_ids,
            "completion_logprobs": logprobs,
            "finish_reason": "stop",
        }

    async def close(self):
        return None


def _play(monkeypatch, *completions):
    monkeypatch.setattr(rollout_module, "FireworksV1CompletionsClient", FakeCompletionsClient)
    FakeCompletionsClient.queued = list(completions)
    FakeCompletionsClient.requests = []
    processor = MultiHopQARolloutProcessor(tokenizer=FakeTokenizer(), renderer=FakeRenderer(FakeTokenizer()))
    config = RolloutProcessorConfig(
        completion_params={"model": "m", "base_url": "https://x/inference", "temperature": 1.0},
        mcp_config_path="",
        steps=4,
        semaphore=asyncio.Semaphore(1),
    )

    async def run():
        [task] = processor([episode_row(ROW)], config)
        return await task

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("tokenizer_name", "output"),
    [
        (
            "Qwen/Qwen3.6-35B-A3B",
            "<tool_call>\n<function=search>\n<parameter=query>\ncapital of France\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>",
        ),
        ("Qwen/Qwen3-8B", SEARCH + "<|im_end|>"),
    ],
)
def test_parse_search_tool_call_reads_the_model_native_format(tokenizer_name, output):
    tokenizer = load_tokenizer(tokenizer_name)
    renderer = build_renderer(tokenizer, tokenizer_name, load_image_processor=False)

    token_ids = tokenizer.encode(output, add_special_tokens=False)

    assert parse_search_tool_call(renderer, token_ids) == ("search", {"query": "capital of France"})


def test_parse_search_tool_call_rejects_other_tools():
    with pytest.raises(ValueError, match="Unknown tool"):
        parse_search_tool_call(FakeRenderer(FakeTokenizer()), [103])


def test_episode_packs_into_one_run_with_turn_spans(monkeypatch):
    episode = _play(monkeypatch, ([100], [-0.1]), ([101, 90], [-0.2, -0.3]))

    # Turn 2's prompt is turn 1's prompt + completion + close + tool reply.
    assert FakeCompletionsClient.requests == [[1, 2, 3], [1, 2, 3, 100, 90, 91, 20, 21]]
    run = episode_to_run(episode, [55])
    sample = run.segments[0]
    assert sample.tokens == [1, 2, 3, 100, 90, 91, 20, 21, 101, 90]
    assert sample.loss_mask == [0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.2, -0.3]
    assert sample.reward == 1.0  # token F1 of "Paris"
    assert run.metadata == {TURN_SPANS: [(3, 4), (8, 10)], ANSWER_TOKENS: [55]}
    assert [message.role for message in episode.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_turn_without_tool_call_is_resampled_then_dropped(monkeypatch):
    retried = _play(monkeypatch, ([102], [-0.1]), ([101, 90], [-0.2, -0.3]))
    assert episode_to_run(retried, [55]).segments[0].reward == 1.0

    failed = _play(monkeypatch, *[([102], [-0.1])] * rollout_module.MAX_ATTEMPTS_PER_TURN)
    assert "no parseable tool call" in failed.execution_metadata.extra["rollout_error"]
    assert episode_to_run(failed, [55]) is None


def _setup(tokenizer) -> RolloutSetup:
    return RolloutSetup(
        sampler=None,
        tokenizer=tokenizer,
        tokenizer_id="Qwen/Qwen3.6-35B-A3B",
        sample_kwargs={
            "max_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_seq_len": 4096,
            "http_timeout": 600,
            "logprobs": True,
            "include_routing_matrix": True,
            "echo": False,
        },
        inference_base_url="https://api.fireworks.ai",
        api_key="key",
        model="accounts/a/models/m#accounts/a/deployments/d",
        completions_per_prompt=4,
    )


def test_rollout_fn_plays_one_episode_on_the_recipe_deployment(monkeypatch):
    monkeypatch.setattr(rollout_module, "FireworksV1CompletionsClient", FakeCompletionsClient)
    FakeCompletionsClient.queued = [([101, 90], [-0.2, -0.3])]
    clients = []
    monkeypatch.setattr(FakeCompletionsClient, "__init__", lambda self, **kw: clients.append(kw))
    renderers = []
    monkeypatch.setattr(
        train_module,
        "build_renderer",
        lambda tokenizer, *args, **kwargs: renderers.append(args) or FakeRenderer(tokenizer),
    )
    factory = make_rollout_fn_factory(MultiHopQATask(max_steps=3, renderer_name="qwen3"))
    rollout_fn = factory(_setup(FakeTokenizer()))

    run = asyncio.run(rollout_fn(ROW))

    assert renderers == [("Qwen/Qwen3.6-35B-A3B", "qwen3")]
    assert run.segments[0].reward == 1.0
    assert run.metadata == {TURN_SPANS: [(3, 5)], ANSWER_TOKENS: [55]}
    # Only the completions API's own parameters are forwarded (no R3 kwargs).
    [client] = clients
    assert client["model_id"] == "accounts/a/models/m#accounts/a/deployments/d"
    assert client["base_url"] == "https://api.fireworks.ai/inference"
    assert (client["temperature"], client["max_tokens"]) == (1.0, 512)
    assert client["request_params"] == {"top_p": 1.0, "top_k": 0}
    # A row without an answer has nothing for IGPO to score: no episode is played.
    assert asyncio.run(rollout_fn({**ROW, "ground_truth": ""})) is None
    assert len(clients) == 1


def test_parse_args_builds_the_rl_loop_config_and_typed_task():
    cfg, task = parse_args(
        [
            "--policy-job-id",
            "job-1",
            "--deployment-id",
            "dep-1",
            "--tokenizer-model",
            "Qwen/Qwen3.6-35B-A3B",
            "--dataset-path",
            "rows.jsonl",
            "--max-steps",
            "3",
            "--gamma",
            "0.9",
        ]
    )

    assert isinstance(cfg, rl_loop.Config)
    assert cfg.trainer.job_id == "job-1"
    assert cfg.deployment.deployment_id == "dep-1"
    assert cfg.deployment.tokenizer_model == "Qwen/Qwen3.6-35B-A3B"
    assert (cfg.dataset, cfg.output_model_id) == ("rows.jsonl", None)
    assert task == MultiHopQATask(max_steps=3, gamma=0.9)


def _run(reward: float) -> RolloutRun:
    sample = RolloutSample(
        tokens=[1, 2, 3, 10, 11],
        logprobs=[0.0, 0.0, 0.0, -0.5, -0.5],
        loss_mask=[0, 0, 0, 1, 1],
        reward=reward,
    )
    return RolloutRun(segments=[sample], metadata={TURN_SPANS: [(3, 5)], ANSWER_TOKENS: [7]})


def test_main_trains_the_rollout_with_igpo_on_rl_loop(monkeypatch, tmp_path):
    """IGPO scores the answer on the policy, then server GRPO trains token advantages."""
    calls: list[tuple] = []
    tasks: list[MultiHopQATask] = []
    metrics: dict[str, float] = {}
    rewards = {"right": [1.0, 0.0], "dropped": [1.0, None]}

    class RolloutFn:
        def __init__(self):
            self.draws = {}

        async def __call__(self, row):
            index = self.draws.get(row["id"], 0)
            self.draws[row["id"]] = index + 1
            reward = rewards[row["id"]][index]
            return None if reward is None else _run(reward)

        async def aclose(self):
            calls.append(("rollout_closed",))

    def logprobs(datum):
        n = datum.model_input.length
        return {"logprobs": sdk.TensorData(data=[-0.5] * n, dtype="float32", shape=[n])}

    class Policy:
        default_timeout = 17

        def forward(self, datums, loss_fn):
            calls.append(("forward", loss_fn, len(datums)))
            return SimpleNamespace(loss_fn_outputs=[logprobs(d) for d in datums])

        def forward_backward(self, datums, loss_fn, loss_fn_config):
            advantages = [list(d.loss_fn_inputs["advantages"].data) for d in datums]
            calls.append(("forward_backward", loss_fn, advantages))
            return SimpleNamespace(metrics={}, loss_fn_outputs=[logprobs(d) for d in datums])

        def submit_forward_backward(self, datums, loss_fn, loss_fn_config=None):
            result = self.forward_backward(datums, loss_fn, loss_fn_config)

            async def result_async(timeout=None):
                assert timeout == self.default_timeout
                return result

            return SimpleNamespace(result_async=result_async)

        def optim_step(self, adam_params, **kwargs):
            calls.append(("optim_step",))
            return SimpleNamespace(metrics={})

    class Checkpoints:
        def __init__(self, *args, **kwargs):
            pass

        def resume(self, **kwargs):
            return None

        def save(self, name, **kwargs):
            calls.append(("save", name, kwargs["data_consumed"]))

    sampler = SimpleNamespace(
        model="m#d",
        base_url="https://api.fireworks.ai",
        close=lambda: None,
        drain_metrics=list,
    )
    service = SimpleNamespace(
        trainer_job_id="job-1",
        deployment_id="dep-1",
        max_context_length=4096,
        create_training_client=lambda *args, **kwargs: object(),
        create_deployment_sampler=lambda **kwargs: sampler,
        close=lambda: None,
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    for name, value in {
        "setup_wandb": lambda *args, **kwargs: None,
        "wandb_finish": lambda *args, **kwargs: None,
        "log_metrics": lambda step_metrics, step: metrics.update(step_metrics),
        "load_deployment_tokenizer": lambda deployment: None,
        "build_service_client": lambda **kwargs: service,
        "resolve_router_replay_enabled": lambda **kwargs: True,
        "make_weight_sync": lambda *args: (lambda name, **kwargs: calls.append(("sync", name))),
        "TrainingCheckpoints": Checkpoints,
    }.items():
        monkeypatch.setattr(rl_loop, name, value)
    monkeypatch.setattr(
        rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *args, **kwargs: Policy(),
    )
    monkeypatch.setattr(
        train_module,
        "make_rollout_fn_factory",
        lambda task: tasks.append(task) or (lambda setup: RolloutFn()),
    )
    cfg = rl_loop.Config(
        log_path=str(tmp_path),
        dataset="rows.jsonl",
        completions_per_prompt=2,
        prompt_groups_per_step=2,
        shuffle=False,
        trainer=TrainerConfig(job_id="job-1"),
        deployment=DeployConfig(deployment_id="dep-1", tokenizer_model="Qwen/Qwen3-8B"),
    )
    monkeypatch.setattr(
        rl_loop,
        "load_jsonl_dataset",
        lambda *args: [{"id": "right"}, {"id": "dropped"}],
    )

    result = train_module.main(cfg, MultiHopQATask(gamma=0.5))

    assert tasks == [MultiHopQATask(gamma=0.5)]
    assert result["steps"] == 1
    # "dropped" kept one of its two draws (fewer than two), so only "right"
    # trains: one IGPO scoring forward per turn, then one server GRPO update on
    # the turn's advantage broadcast to its targets (z-scored outcome 1 vs 0).
    [forward_backward] = [call for call in calls if call[0] == "forward_backward"]
    assert forward_backward[1] == "ppo"
    assert [[round(a, 3) for a in row] for row in forward_backward[2]] == [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, -1.0, -1.0],
    ]
    assert [call[0] for call in calls] == [
        "sync",
        "forward",
        "forward_backward",
        "optim_step",
        "sync",
        "save",
        "rollout_closed",
    ]
    assert calls[1] == ("forward", "cross_entropy", 2)
    assert metrics["igpo/avg_turns"] == 1.0


def test_search_episode_on_the_qwen3_6_template_end_to_end(monkeypatch):
    """Real tokenizer, template and renderer; only the completions endpoint is faked."""
    tokenizer = load_tokenizer("Qwen/Qwen3.6-35B-A3B")
    calls = iter(
        [
            "<tool_call>\n<function=search>\n<parameter=query>\ncapital of France\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>",
            "<tool_call>\n<function=submit_answer>\n<parameter=answer>\nParis\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>",
        ]
    )
    requests = []

    class Completions:
        async def create(self, **payload):
            requests.append(payload["prompt"])
            ids = tokenizer.encode(next(calls), add_special_tokens=False)
            content = [{"token_id": token, "sampling_logprob": -0.1} for token in ids]
            choice = {"finish_reason": "stop", "logprobs": {"content": content}}
            return SimpleNamespace(model_dump=lambda: {"choices": [choice], "usage": {}})

    class AsyncFireworks:
        def __init__(self, **kwargs):
            self.completions = Completions()

        async def close(self):
            return None

    monkeypatch.setattr(ep_client, "AsyncFireworks", AsyncFireworks)
    rollout_fn = make_rollout_fn_factory(MultiHopQATask(max_steps=4))(_setup(tokenizer))

    run = asyncio.run(rollout_fn(ROW))

    first, second = requests
    assert second[: len(first)] == first
    assert "<tool_response>" in tokenizer.decode(second[len(first) :])
    assert run.segments[0].reward == 1.0
    assert len(run.metadata[TURN_SPANS]) == 2


def test_main_keeps_groups_with_equal_outcome_rewards(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        train_module.rl_loop,
        "main",
        lambda cfg, **kwargs: captured.update(kwargs) or {},
    )

    train_module.main(train_module.Config(log_path="/tmp/igpo"))

    # Turn-level information gain can vary when every outcome reward is equal.
    assert captured["dynamic_filter_fn"] is None
    assert captured["prepare_batch_fn"] is not None
