import asyncio
import base64
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fireworks.training import sdk
from training.recipes.sft_loop import Config
from training.utils import fileio
from training.utils.runner import RunnerConfig, RunnerIO, RunStatus
from training.utils.sft_generation_eval import (
    prepare_generation_eval,
    run_generation_eval,
    write_skipped_generation_eval,
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
def test_generation_reference_must_be_selected_by_training_mode(tmp_path, mode, caplog):
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
    with caplog.at_level(logging.WARNING, logger=generation_eval_module.logger.name):
        examples = prepare_generation_eval(
            JsonlRenderDataset(str(path), lambda row: row),
            renderer=Mock(),
            max_examples=10,
            max_seq_len=10,
            train_on_what=mode,
        )
    assert examples == []
    assert "no prompt before a trainable assistant reference" in caplog.text


def test_generation_eval_skips_overlong_and_unreadable_rows(tmp_path, caplog):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "answer"}]})
            for text in ("too long", "unreadable", "ok")
        )
    )
    renderer = Mock()
    renderer.build_generation_prompt.side_effect = [
        SimpleNamespace(length=10),
        OSError("unrecognized image"),
        sdk.ModelInput.from_ints([1, 2]),
    ]
    with caplog.at_level(logging.WARNING, logger=generation_eval_module.logger.name):
        examples = prepare_generation_eval(
            JsonlRenderDataset(str(path), lambda row: row),
            renderer=renderer,
            max_examples=10,
            max_seq_len=10,
        )
    assert len(examples) == 1
    assert examples[0][1]["reference"] == "answer"
    assert "prompt length 10 leaves no room within max_seq_len 10" in caplog.text
    assert "Skipping generation eval row 1: OSError" in caplog.text
    assert "unrecognized image" not in caplog.text


def test_generation_eval_limit_counts_only_usable_rows(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": text}]})
            for text in ("skip-a", "skip-b", "keep", "unused")
        )
    )
    renderer = Mock()
    renderer.build_generation_prompt.side_effect = [
        SimpleNamespace(length=10),
        OSError("bad image"),
        sdk.ModelInput.from_ints([1]),
        sdk.ModelInput.from_ints([2]),
    ]
    examples = prepare_generation_eval(
        JsonlRenderDataset(str(path), lambda row: row),
        renderer=renderer,
        max_examples=1,
        max_seq_len=10,
    )
    assert [example[1]["reference"] for example in examples] == ["keep"]
    assert renderer.build_generation_prompt.call_count == 3


def test_skipped_row_log_omits_message_content(tmp_path, caplog):
    secret = "customer-secret-payload"
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": None, "content": secret}, {"role": "assistant", "content": "answer"}]})
    )
    with caplog.at_level(logging.WARNING, logger=generation_eval_module.logger.name):
        examples = prepare_generation_eval(
            JsonlRenderDataset(str(path), lambda row: row),
            renderer=Mock(),
            max_examples=1,
            max_seq_len=10,
        )
    assert examples == []
    assert secret not in caplog.text
    assert "ValueError" in caplog.text


def test_skipped_generation_eval_replaces_previous_artifact(tmp_path):
    cfg = Config(log_path=str(tmp_path), base_model="accounts/fireworks/models/qwen3-8b")
    artifact = tmp_path / "generation_eval.json"
    artifact.write_text('{"status":"completed","examples":[{"prompt":"old"}]}')
    write_skipped_generation_eval(cfg, "Generation evaluation skipped: no usable rows.")
    result = json.loads(artifact.read_text())
    assert result["status"] == "skipped"
    assert result["examples"] == []
    assert result["error"] == "Generation evaluation skipped: no usable rows."
    assert "old" not in artifact.read_text()


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
    assert example["cells"][1] == ({"output": ""} if failed_sampler == "trained" else answer)
    if failed_sampler:
        assert "Generation evaluation failed" in result["error"]
    assert baseline.sample.call_args == trained.sample.call_args
    params = baseline.sample.call_args.kwargs["sampling_params"]
    assert params.max_tokens == 2
    assert params.temperature == 0
    assert params.stop == [99]
    assert service.create_sampling_client.call_args_list[0].kwargs["base_model"] == cfg.base_model
    assert service.create_sampling_client.call_args_list[1].kwargs["model_path"] == "account/run/eval-step-2"
    controllers = [call.kwargs["concurrency_controller"] for call in service.create_sampling_client.call_args_list]
    assert len(controllers) == 2
    assert controllers[0] is not controllers[1]
    assert all(controller.window_size == 4 for controller in controllers)
    baseline.close.assert_called_once()
    trained.close.assert_called_once()
    runner.write_status(RunStatus.COMPLETED, step=2, total_steps=2)
    assert json.loads(status_path.read_text())["details"][0]["state"] == "JOB_STATE_EVALUATION"


def test_each_sampler_controller_waits_on_its_own_loop():
    """Each sampler loop waits on its own semaphore. One shared semaphore cannot."""

    async def wait_on_full_window(controller, waiting):
        await controller.acquire()
        waiter = asyncio.create_task(controller.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()
        waiting.set()
        controller.release()
        await asyncio.wait_for(waiter, timeout=5)
        controller.release()

    errors = []

    def run(controller, waiting):
        try:
            asyncio.run(wait_on_full_window(controller, waiting))
        except Exception as exc:
            errors.append(exc)

    waiting = [threading.Event(), threading.Event()]
    threads = [
        threading.Thread(target=run, args=(sdk.FixedConcurrencyController(1), event)) for event in waiting
    ]
    for thread in threads:
        thread.start()
    assert all(event.wait(timeout=5) for event in waiting)
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    assert not any(thread.is_alive() for thread in threads)


def test_shared_fixed_controller_rejects_the_other_sampler_loop():
    shared = sdk.FixedConcurrencyController(1)
    bound = threading.Event()
    release_holder = threading.Event()
    holder_errors = []

    async def bind_on_holder_loop():
        await shared.acquire()
        waiter = asyncio.create_task(shared.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()
        bound.set()
        # Stay bound until the other loop has attempted acquire. A fixed sleep
        # can release first and let that acquire succeed without a loop check.
        await asyncio.to_thread(release_holder.wait)
        shared.release()
        await waiter
        shared.release()

    def hold():
        try:
            asyncio.run(bind_on_holder_loop())
        except Exception as exc:
            holder_errors.append(exc)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert bound.wait(timeout=5)

        async def wait_on_other_loop():
            await shared.acquire()

        with pytest.raises(RuntimeError, match="different event loop"):
            asyncio.run(wait_on_other_loop())
    finally:
        release_holder.set()
    holder.join(timeout=5)
    assert holder_errors == []
    assert not holder.is_alive()


def test_generation_eval_submits_all_samples_before_waiting(tmp_path):
    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-2"
    sequence = SimpleNamespace(tokens=[8], stop_reason="stop")
    submitted = []

    def submit(name):
        def _sample(*_args, **_kwargs):
            submitted.append(name)
            future = Mock()

            def result(**_kwargs):
                assert len(submitted) == 4
                return SimpleNamespace(sequences=[sequence])

            future.result.side_effect = result
            return future

        return _sample

    baseline.sample.side_effect = submit("baseline")
    trained.sample.side_effect = submit("trained")
    tokenizer, renderer = Mock(), Mock()
    tokenizer.decode.return_value = "answer"
    renderer.get_stop_sequences.return_value = []
    examples = [
        (sdk.ModelInput.from_ints([1]), {"prompt": "user: first", "reference": "first"}),
        (sdk.ModelInput.from_ints([2]), {"prompt": "user: second", "reference": "second"}),
    ]
    run_generation_eval(
        cfg=cfg,
        examples=examples,
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=2,
    )
    assert submitted == ["baseline", "trained", "baseline", "trained"]


def test_generation_eval_uses_one_deadline_and_cancels_the_rest(tmp_path, monkeypatch):
    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10, step_timeout=30)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-2"
    clock = {"now": 100.0}
    monkeypatch.setattr(generation_eval_module.time, "monotonic", lambda: clock["now"])
    timeouts = []
    hung = Mock()

    def hung_result(timeout):
        timeouts.append(timeout)
        clock["now"] = 130.0
        raise TimeoutError("sampler hung")

    hung.result.side_effect = hung_result
    hung.done.return_value = False
    later = Mock()
    later.done.return_value = False
    baseline.sample.return_value = hung
    trained.sample.return_value = later
    tokenizer, renderer = Mock(), Mock()
    renderer.get_stop_sequences.return_value = []
    run_generation_eval(
        cfg=cfg,
        examples=[(sdk.ModelInput.from_ints([1]), {"prompt": "user: question", "reference": "reference"})],
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=2,
    )
    result = json.loads((tmp_path / "generation_eval.json").read_text())
    assert timeouts == [30]
    assert result["status"] == "failed"
    assert result["examples"][0]["cells"] == [{"output": ""}, {"output": ""}]
    hung.cancel.assert_called_once()
    later.cancel.assert_called_once()
    later.result.assert_not_called()


def test_generation_eval_keeps_completed_future_after_timeout(tmp_path, monkeypatch):
    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10, step_timeout=30)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-2"
    clock = {"now": 100.0}
    monkeypatch.setattr(generation_eval_module.time, "monotonic", lambda: clock["now"])
    hung = Mock()
    completed = Mock()
    sequence = SimpleNamespace(tokens=[8], stop_reason="stop")

    def hung_result(timeout):
        assert timeout == 30
        clock["now"] = 130.0
        raise TimeoutError("sampler hung")

    hung.result.side_effect = hung_result
    hung.done.return_value = False
    completed.done.return_value = True
    completed.result.return_value = SimpleNamespace(sequences=[sequence])
    baseline.sample.return_value = hung
    trained.sample.return_value = completed
    tokenizer, renderer = Mock(), Mock()
    tokenizer.decode.return_value = "trained answer"
    renderer.get_stop_sequences.return_value = []
    run_generation_eval(
        cfg=cfg,
        examples=[(sdk.ModelInput.from_ints([1]), {"prompt": "user: question", "reference": "reference"})],
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=2,
    )
    result = json.loads((tmp_path / "generation_eval.json").read_text())
    assert result["status"] == "failed"
    assert result["examples"][0]["cells"][0] == {"output": ""}
    assert result["examples"][0]["cells"][1]["output"] == "trained answer"
    hung.cancel.assert_called_once()
    completed.cancel.assert_not_called()
    completed.result.assert_called_once_with(timeout=0)


def test_finished_sample_does_not_grant_the_next_sample_a_new_timeout(tmp_path, monkeypatch):
    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10, step_timeout=30)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-2"
    clock = {"now": 100.0}
    monkeypatch.setattr(generation_eval_module.time, "monotonic", lambda: clock["now"])
    sequence = SimpleNamespace(tokens=[8], stop_reason="stop")
    finished = Mock()

    def finished_result(timeout):
        assert timeout == 30
        clock["now"] = 130.0
        return SimpleNamespace(sequences=[sequence])

    finished.result.side_effect = finished_result
    later = Mock()
    later.done.return_value = False
    baseline.sample.return_value = finished
    trained.sample.return_value = later
    tokenizer, renderer = Mock(), Mock()
    tokenizer.decode.return_value = "answer"
    renderer.get_stop_sequences.return_value = []
    run_generation_eval(
        cfg=cfg,
        examples=[(sdk.ModelInput.from_ints([1]), {"prompt": "user: question", "reference": "reference"})],
        renderer=renderer,
        tokenizer=tokenizer,
        service=service,
        client=client,
        runner=RunnerIO(RunnerConfig()),
        step=2,
    )
    result = json.loads((tmp_path / "generation_eval.json").read_text())
    assert result["examples"][0]["cells"][0]["output"] == "answer"
    assert result["examples"][0]["cells"][1] == {"output": ""}
    assert result["status"] == "failed"
    finished.cancel.assert_not_called()
    later.cancel.assert_called_once()
    later.result.assert_not_called()


def test_authentication_failure_keeps_successful_samples_and_does_not_raise(tmp_path, caplog):
    cfg = Config(log_path=str(tmp_path), serverless=True, generation_eval=True, max_seq_len=10)
    baseline, trained, service, client = Mock(), Mock(), Mock(), Mock()
    service.create_sampling_client.side_effect = [baseline, trained]
    client.save_weights_for_sampler.return_value.path = "account/run/eval-step-2"
    sequence = SimpleNamespace(tokens=[8], stop_reason="stop")
    baseline.sample.return_value.result.side_effect = PermissionError(
        "403 credential is limited to its managed training session"
    )
    trained.sample.return_value.result.return_value = SimpleNamespace(sequences=[sequence])
    tokenizer, renderer = Mock(), Mock()
    tokenizer.decode.return_value = "trained answer"
    renderer.get_stop_sequences.return_value = []
    with caplog.at_level(logging.ERROR, logger=generation_eval_module.logger.name):
        run_generation_eval(
            cfg=cfg,
            examples=[(sdk.ModelInput.from_ints([1, 2, 3]), {"prompt": "user: question", "reference": "reference"})],
            renderer=renderer,
            tokenizer=tokenizer,
            service=service,
            client=client,
            runner=RunnerIO(RunnerConfig()),
            step=2,
        )
    result = json.loads((tmp_path / "generation_eval.json").read_text())
    assert result["status"] == "failed"
    assert "trained model is still available" in result["error"]
    assert result["examples"][0]["cells"][0] == {"output": ""}
    assert result["examples"][0]["cells"][1]["output"] == "trained answer"
    assert "credential is limited" in caplog.text


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
