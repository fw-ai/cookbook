"""Offline trainer integration checks; fake clients never access the service."""

from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).parent.parent))
import lipogram_rl as rl


def test_validation_is_unprompted_and_joint_success_drives_switch(tmp_path, monkeypatch):
    runner = object.__new__(rl.ServerlessAntislopRL)
    runner.cfg = rl.Config(eval_group_size=3, max_sample_tokens=20, reward_phase="auto", strict_trigger=0.5)
    runner.phase = "ramp"
    runner.eval_rows = [{"messages": [{"role": "system", "content": "No e"}, {"role": "user", "content": "A task"}],
                         "ground_truth": json.dumps({"source_id": "one", "family": "plain"})}]
    runner.eval_row_indices = [0]
    runner.tokenizer = None
    seen = []
    def render(messages):
        seen.append(messages)
        return NS(length=1)
    runner.eval_renderer = NS(build_generation_prompt=render)
    runner.renderer = NS(get_stop_sequences=lambda: [], parse_response=lambda tokens: (str(tokens), None))
    monkeypatch.setattr(rl, "get_text_content", lambda message: message)
    monkeypatch.setattr(rl, "_log_wandb_eval", lambda *args: None)
    def future(value):
        return NS(result=lambda **kwargs: value)
    runner.training_client = NS(save_weights_for_sampler=lambda name: future(NS(path="offline-snapshot")))
    sequences = [NS(tokens=[i], finish_reason="stop") for i in range(3)]
    sampler = NS(sample=lambda **kwargs: future(NS(sequences=sequences)), close=lambda: None)
    runner.service = NS(create_sampling_client=lambda **kwargs: sampler)
    runner._prescore = lambda *args: None
    # Useful but invalid; valid but useless; valid and useful.
    scores = [(0.1, False, True, 2, 1), (0, True, False, 0, 1), (0.9, True, True, 0, 1)]
    runner._score = lambda tokens, row: scores[tokens[0]]
    runner.eval_metrics_path = tmp_path / "metrics.jsonl"
    runner.eval_completions_dir = tmp_path
    runner.run_dir = tmp_path
    result = runner._evaluate_impl(0)
    assert all(m["role"] != "system" for m in seen[0])
    assert result["eval/e_free_rate"] == 2 / 3
    assert result["eval/answered_rate"] == 2 / 3
    assert result["eval/success_rate"] == 1 / 3
    assert runner.phase == "ramp"  # e-free alone would have incorrectly switched
    saved = json.loads(runner.eval_metrics_path.read_text())
    assert saved["eval/reward_phase"] == "ramp"
    assert saved["eval/with_instruction"] is False


def test_glm_native_reasoning_is_trained_but_not_checked_as_final_answer():
    import pytest
    from transformers import AutoTokenizer
    from training.renderer.glm5 import GLM53FlashRenderer
    from training.utils.supervised import normalize_messages
    from training.renderer.base import TrainOnWhat, get_text_content
    try:
        tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.3-Flash", local_files_only=True)
    except OSError:
        pytest.skip("GLM tokenizer not cached; this test never downloads it")
    renderer = GLM53FlashRenderer(tokenizer)
    reasoning = "The answer needs checking."
    messages = normalize_messages([
        {"role": "user", "content": "Which city?"},
        {"role": "assistant", "reasoning_content": reasoning, "content": "Tokyo"},
    ])
    model_input, weights = renderer.build_supervised_example(messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    trained = tokenizer.decode([token for token, weight in zip(model_input.to_ints(), weights.tolist()) if weight > 0])
    assert reasoning in trained
    assert "Tokyo" in trained
    assert "Which city?" not in trained
    # GLM generation prompt supplies <think>; returned completion starts with its body.
    tokens = tokenizer.encode(reasoning + "</think>Tokyo<|user|>", add_special_tokens=False)
    parsed, _ = renderer.parse_response(tokens)
    assert get_text_content(parsed) == "Tokyo"
