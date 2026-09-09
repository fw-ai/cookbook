"""Regression tests for the shared QA gate itself, without external assets."""

from types import SimpleNamespace

import pytest
import tinker

from training.renderer.verifier.utils.hf_parity import compare_renderer_to_reference
from training.tests.unit.renderer_contract import (
    RendererContractTests,
    _assert_parsed_assistant,
    _split_by_weights,
    contract_parameters,
)
from training.tests.unit.renderer_matrix import RendererCase
from training.tests.unit.renderer_scenarios import ALL_SCENARIOS


def _case():
    return RendererCase(
        renderer="test",
        tokenizer_model="fixture",
        supports_thinking=True,
        supports_tools=True,
        supervised_hf_parity=True,
        observation_equals_generation=True,
        has_extension_property=True,
    )


def test_every_scenario_has_a_parity_check_and_terminal_mask_check():
    parameters = contract_parameters([_case()])
    parity = (
        parameters["test_hf_generation_parity"][1]
        + parameters["test_hf_supervised_parity"][1]
    )
    assert {param.values[1].id for param in parity} == {
        scenario.id for scenario in ALL_SCENARIOS
    }
    assert len(parity) == len(ALL_SCENARIOS)
    terminal = {
        scenario.id for scenario in ALL_SCENARIOS if scenario.ends_with_assistant
    }
    for name in (
        "test_supervised_generation_parse_consistency",
        "test_supervised_observation_equals_generation_prompt",
    ):
        assert {param.values[1].id for param in parameters[name][1]} == terminal


def test_hf_xfail_does_not_mask_independent_parse_or_weights():
    parameters = contract_parameters(
        [_case()],
        divergences={"hf": {("test", "multi_turn_sft"): "reference formatting"}},
    )
    for name, (_, params) in parameters.items():
        for param in params:
            if param.values[1].id != "multi_turn_sft":
                continue
            if name == "test_hf_supervised_parity":
                assert len(param.marks) == 1
                assert param.marks[0].kwargs["strict"] is True
            else:
                assert not param.marks


def test_empty_renderer_matrix_cannot_pass():
    with pytest.raises(AssertionError, match="must not be empty"):
        RendererContractTests().test_contract_matrix_is_nonempty()


@pytest.mark.parametrize("weights", [[0, 1, 0, 1], [0, 0.5, 1, 1], [0, 1]])
def test_shared_mask_assertion_rejects_corruption(weights):
    with pytest.raises(AssertionError):
        _split_by_weights([1, 2, 3, 4], weights)


@pytest.mark.parametrize("corruption", ["thinking", "argument", "order"])
def test_shared_structured_parse_assertion_rejects_lost_information(corruption):
    calls = [
        {"function": {"name": name, "arguments": {"nested": [index, {"key": "value"}]}}}
        for index, name in enumerate(("first", "second"))
    ]
    expected = {"content": "answer", "reasoning_content": "plan", "tool_calls": calls}
    parsed = {
        "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "answer"},
        ],
        "tool_calls": calls.copy(),
    }
    if corruption == "thinking":
        parsed["content"][0]["thinking"] = "plan<end>"
    elif corruption == "argument":
        parsed["tool_calls"] = [
            {"function": {"name": "first", "arguments": {"nested": [0]}}},
            calls[1],
        ]
    else:
        parsed["tool_calls"].reverse()
    with pytest.raises(AssertionError):
        _assert_parsed_assistant(
            renderer_name="test",
            scenario_id="structured",
            expected_message=expected,
            parsed_message=parsed,
            action_tokens=[1],
            tokenizer=SimpleNamespace(decode=lambda _: "plan answer"),
        )


def test_reference_comparison_checks_token_ids_even_when_decoding_matches():
    tokenizer = SimpleNamespace(
        decode=lambda _: "same bytes",
        apply_chat_template=lambda *args, **kwargs: [10, 20],
    )
    renderer = SimpleNamespace(
        build_generation_prompt=lambda *args, **kwargs: tinker.ModelInput.from_ints(
            [10, 21]
        )
    )
    messages = [{"role": "user", "content": "hello"}]
    result = compare_renderer_to_reference(
        renderer=renderer,
        tokenizer=tokenizer,
        messages=messages,
        reference_messages=messages,
    )
    assert result.renderer_decoded == result.hf_decoded
    assert not result.match
    assert result.first_divergence_idx == 1
