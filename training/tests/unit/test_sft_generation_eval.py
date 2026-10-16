import base64
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fireworks.training import sdk
from training.recipes.sft_loop import Config
from training.utils import fileio
from training.utils.runner import DatasetError, RunnerConfig, RunnerIO, RunStatus
from training.utils.sft_generation_eval import (
    prepare_generation_eval,
    run_generation_eval,
)
from training.utils import sft_generation_eval as generation_eval_module
from training.utils.streaming import JsonlRenderDataset


@pytest.mark.parametrize("flags", [{"weight": 1}, {"trainable": True, "weight": 0}, {"trainable": None, "weight": 1}])
def test_generation_prompt_excludes_reference_and_later_turns(tmp_path, flags):
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "reference", **flags},
        {"role": "user", "content": "later"},
        {"role": "assistant", "content": "ignored", "trainable": False, "weight": 1},
    ]
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"messages": messages}))
    dataset = JsonlRenderDataset(str(path), lambda row: row)
    renderer = Mock()
    renderer.build_generation_prompt.return_value = sdk.ModelInput.from_ints([1, 2])
    examples = prepare_generation_eval(dataset, renderer=renderer, max_examples=10, max_seq_len=10)
    assert renderer.build_generation_prompt.call_args.args[0] == messages[:1]
    assert examples[0][1]["reference"] == "reference"
    assert examples[0][1]["prompt"] == "user: question"
    assert examples[0][1]["messages"] == messages[:1]


def test_display_preserves_multiturn_context_and_tool_reference(tmp_path):
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"北京"}'}}
    messages = [
        {"role": "system", "content": "Use the lookup tool."},
        {"role": "user", "content": "上海的天气？"},
        {"role": "assistant", "content": "晴天。"},
        {"role": "user", "content": [{"type": "text", "text": "北京呢？"}]},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
    ]
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"messages": messages}))
    renderer = Mock()
    renderer.build_generation_prompt.return_value = sdk.ModelInput.from_ints([1, 2])
    examples = prepare_generation_eval(
        JsonlRenderDataset(str(path), lambda row: row),
        renderer=renderer,
        max_examples=1,
        max_seq_len=10,
    )
    record = examples[0][1]
    assert record["prompt"] == (
        "system: Use the lookup tool.\n\nuser: 上海的天气？\n\nassistant: 晴天。\n\nuser: "
        + json.dumps(messages[3], ensure_ascii=False)
    )
    assert json.loads(record["reference"]) == messages[4]
    assert "call-1" not in record["prompt"]


@pytest.mark.parametrize("mode", ["last_assistant_message", "last_assistant_turn", "all_user_and_system_messages"])
def test_generation_reference_must_be_selected_by_training_mode(tmp_path, mode):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "earlier answer"},
                    {"role": "user", "content": "later question"},
                ]
            }
        )
    )
    with pytest.raises(DatasetError, match="assistant reference"):
        prepare_generation_eval(
            JsonlRenderDataset(str(path), lambda row: row),
            renderer=Mock(),
            max_examples=10,
            max_seq_len=10,
            train_on_what=mode,
        )


@pytest.mark.parametrize("failed_sampler", [None, "baseline", "trained"])
def test_paired_answers_persist_with_shared_prompts_and_partial_failures(tmp_path, failed_sampler):
    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=5)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-2"
    sequence = SimpleNamespace(tokens=[8, 9], stop_reason="stop")
    baseline.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    trained.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    tokenizer, renderer = Mock(), Mock()
    tokenizer.decode.return_value = "answer"
    renderer.get_stop_sequences.return_value = [99]
    prompt = sdk.ModelInput.from_ints([1, 2, 3])
    status_path = tmp_path / "status.json"
    runner = RunnerIO(RunnerConfig(status_file=str(status_path)))

    def sample_baseline(**kwargs):
        status = json.loads(status_path.read_text())
        assert status["details"][0]["state"] == "JOB_STATE_EVALUATION"
        return SimpleNamespace(sequences=[sequence])

    baseline.sample.return_value.result.side_effect = sample_baseline
    if failed_sampler:
        sampler = baseline if failed_sampler == "baseline" else trained
        sampler.sample.return_value.result.side_effect = RuntimeError("sampler unavailable")
    run_generation_eval(
        cfg=cfg,
        examples=[(prompt, {"prompt": "user: question", "reference": "reference"})],
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=runner,
        step=2,
    )
    result = json.loads((tmp_path / "generation_eval.json").read_text())
    assert result["status"] == ("failed" if failed_sampler else "completed")
    assert result["checkpoints"] == ["Original", "Your model"]
    example = result["examples"][0]
    assert example["prompt"] == "user: question"
    assert example["reference"] == "reference"
    assert len(example["cells"]) == len(result["checkpoints"])
    assert all("isCorrect" not in cell and "reason" not in cell for cell in example["cells"])
    answer = {"output": "answer", "tokens": [8, 9], "stop_reason": "stop"}
    assert example["cells"][0] == ({"output": ""} if failed_sampler == "baseline" else answer)
    assert example["cells"][1] == ({"output": ""} if failed_sampler else answer)
    if failed_sampler:
        assert "Generation evaluation failed" in result["error"]
    if failed_sampler != "baseline":
        assert baseline.sample.call_args == trained.sample.call_args
    else:
        trained.sample.assert_not_called()
    params = baseline.sample.call_args.kwargs["sampling_params"]
    assert params.max_tokens == 2
    assert params.temperature == 0
    assert params.stop == [99]
    assert service.create_sampling_client.call_args_list[0].kwargs["base_model"] == cfg.base_model
    assert service.create_sampling_client.call_args_list[1].kwargs["model_path"] == "account/run/eval-step-2"
    baseline.close.assert_called_once()
    trained.close.assert_called_once()
    runner.write_status(RunStatus.COMPLETED, step=2, total_steps=2)
    assert json.loads(status_path.read_text())["details"][0]["state"] == "JOB_STATE_EVALUATION"


def test_cancellation_during_artifact_write_propagates(tmp_path, monkeypatch):
    service = Mock()
    service.create_sampling_client.side_effect = RuntimeError("sampler unavailable")
    monkeypatch.setattr(fileio, "write_json", Mock(side_effect=SystemExit("cancelled")))
    with pytest.raises(SystemExit, match="cancelled"):
        run_generation_eval(
            cfg=Config(log_path=str(tmp_path)),
            examples=[],
            renderer=Mock(),
            tokenizer=Mock(),
            service=service,
            client=Mock(),
            runner=Mock(),
            step=1,
        )


def test_cancellation_propagates_and_closes_samplers(tmp_path):
    service, client, sampler, renderer = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.return_value = sampler
    sampler.sample.return_value.result.side_effect = SystemExit("cancelled")
    renderer.get_stop_sequences.return_value = []
    with pytest.raises(SystemExit):
        run_generation_eval(
            cfg=Config(log_path=str(tmp_path), max_seq_len=10),
            examples=[(sdk.ModelInput.from_ints([1]), {})],
            renderer=renderer,
            tokenizer=Mock(),
            service=service,
            client=client,
            runner=RunnerIO(RunnerConfig()),
            step=1,
        )
    assert sampler.close.call_count == 2


def test_image_prompt_is_accepted_and_sampled(tmp_path):
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aa"}}]},
        {"role": "assistant", "content": "a cat"},
    ]
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"messages": messages}))
    prompt = sdk.ModelInput(
        chunks=[
            sdk.types.EncodedTextChunk(tokens=[1, 2]),
            sdk.types.ImageChunk(data=b"jpeg-bytes", format="jpeg", expected_tokens=4),
        ]
    )
    with pytest.raises(ValueError, match="EncodedTextChunks"):
        prompt.to_ints()
    renderer = Mock()
    renderer.build_generation_prompt.return_value = prompt
    examples = prepare_generation_eval(
        JsonlRenderDataset(str(path), lambda row: row),
        renderer=renderer,
        max_examples=1,
        max_seq_len=10,
    )
    assert examples[0][0] is prompt
    assert examples[0][1]["reference"] == "a cat"
    assert examples[0][1]["generation_messages"]

    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-1"
    sequence = SimpleNamespace(tokens=[8, 9], stop_reason="stop")
    baseline.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    trained.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    tokenizer = Mock()
    tokenizer.decode.return_value = "a cat"
    renderer.get_stop_sequences.return_value = []
    renderer.image_placeholder_token_id = 77
    renderer.sampler_image_placeholder_affixes = None
    renderer.render_trace = None
    run_generation_eval(
        cfg=cfg,
        examples=examples,
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=1,
    )
    result = json.loads((tmp_path / "generation_eval.json").read_text())
    assert result["status"] == "completed"
    assert "generation_messages" not in result["examples"][0]
    assert result["examples"][0]["cells"] == [
        {"output": "a cat", "tokens": [8, 9], "stop_reason": "stop"},
        {"output": "a cat", "tokens": [8, 9], "stop_reason": "stop"},
    ]
    sampled = baseline.sample.call_args.args[0]
    image_url = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")
    assert sampled.to_ints() == [1, 2, 77]
    assert baseline.sample.call_args.kwargs["images"] == [image_url]
    assert trained.sample.call_args.kwargs["images"] == [image_url]
    assert baseline.sample.call_args.kwargs["sampling_params"].max_tokens == 4


def test_image_prompt_with_tools_reuses_tool_prefixed_messages(tmp_path, monkeypatch):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aa"}}]},
        {"role": "assistant", "content": "sunny"},
    ]
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"messages": messages, "tools": tools}))
    prompt = sdk.ModelInput(
        chunks=[
            sdk.types.EncodedTextChunk(tokens=[1, 2]),
            sdk.types.ImageChunk(data=b"jpeg-bytes", format="jpeg", expected_tokens=4),
        ]
    )
    prefixed = [{"role": "system", "content": "tool decls"}, messages[0]]
    renderer = Mock()
    renderer.build_generation_prompt.return_value = prompt
    monkeypatch.setattr(
        generation_eval_module,
        "build_tool_prefixed_messages",
        lambda context, renderer, tools=None: prefixed,
    )
    examples = prepare_generation_eval(
        JsonlRenderDataset(str(path), lambda row: row),
        renderer=renderer,
        max_examples=1,
        max_seq_len=10,
    )
    assert examples[0][1]["generation_messages"] == prefixed
    renderer.build_generation_prompt.assert_called_once_with(prefixed)

    seen = {}

    def fake_project(messages, model_input, tokenizer, *, renderer=None):
        seen["messages"] = messages
        seen["model_input"] = model_input
        return [1, 2, 77], ["data:image/jpeg;base64,anBlZy1ieXRlcw=="]

    monkeypatch.setattr(
        "training.utils.rl.rollout.renderer.build_multimodal_completions_prompt_token_ids",
        fake_project,
    )
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-1"
    sequence = SimpleNamespace(tokens=[8, 9], stop_reason="stop")
    baseline.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    trained.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    tokenizer = Mock()
    tokenizer.decode.return_value = "sunny"
    renderer.get_stop_sequences.return_value = []
    run_generation_eval(
        cfg=Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10),
        examples=examples,
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=1,
    )
    assert seen["messages"] == prefixed
    assert seen["model_input"] is prompt
    assert baseline.sample.call_args.args[0].to_ints() == [1, 2, 77]
    assert "generation_messages" not in json.loads((tmp_path / "generation_eval.json").read_text())["examples"][0]


def test_image_prompt_falls_back_to_messages_without_generation_messages(tmp_path, monkeypatch):
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aa"}}]},
    ]
    prompt = sdk.ModelInput(
        chunks=[
            sdk.types.EncodedTextChunk(tokens=[1, 2]),
            sdk.types.ImageChunk(data=b"jpeg-bytes", format="jpeg", expected_tokens=4),
        ]
    )
    seen = {}

    def fake_project(project_messages, model_input, tokenizer, *, renderer=None):
        seen["messages"] = project_messages
        return [1, 2, 77], ["data:image/jpeg;base64,anBlZy1ieXRlcw=="]

    monkeypatch.setattr(
        "training.utils.rl.rollout.renderer.build_multimodal_completions_prompt_token_ids",
        fake_project,
    )
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-1"
    sequence = SimpleNamespace(tokens=[8, 9], stop_reason="stop")
    baseline.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    trained.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    tokenizer = Mock()
    tokenizer.decode.return_value = "a cat"
    renderer = Mock()
    renderer.get_stop_sequences.return_value = []
    run_generation_eval(
        cfg=Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10),
        examples=[(prompt, {"messages": messages, "reference": "a cat", "prompt": "user: image"})],
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=1,
    )
    assert seen["messages"] == messages
    assert baseline.sample.call_args.args[0].to_ints() == [1, 2, 77]
    assert baseline.sample.call_args.kwargs["images"] == ["data:image/jpeg;base64,anBlZy1ieXRlcw=="]


def test_artifact_failure_is_logged_and_does_not_fail_training(tmp_path, monkeypatch, caplog):
    service = Mock()
    service.create_sampling_client.side_effect = RuntimeError("sampler unavailable")
    monkeypatch.setattr(fileio, "write_json", Mock(side_effect=OSError("storage unavailable")))
    with caplog.at_level(logging.ERROR, logger=generation_eval_module.logger.name):
        run_generation_eval(
            cfg=Config(log_path=str(tmp_path)),
            examples=[],
            renderer=Mock(),
            tokenizer=Mock(),
            service=service,
            client=Mock(),
            runner=Mock(),
            step=1,
        )
    service.create_sampling_client.assert_called_once()
    assert "Generation evaluation failed" in caplog.text
    assert "Failed to write generation_eval.json" in caplog.text
    assert "storage unavailable" in caplog.text
