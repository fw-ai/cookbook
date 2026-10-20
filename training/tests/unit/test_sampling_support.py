import pytest
from fireworks.training import sdk
from fireworks.training.sdk.routing import RoutingReferences
from training.utils.rl.losses import PolicyBatch, build_policy_datums
from training.utils.rl.sampling_support import attach_top_sampling_references


def _datum():
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=[11, 12], dtype="int64", shape=[2]),
            "weights": sdk.TensorData(data=[0, 1], dtype="int64", shape=[2]),
        },
    )


@pytest.mark.parametrize("advantage", [0.0, -2.0, 3.0])
def test_references_attach_without_changing_loss_inputs(advantage):
    references = RoutingReferences(2, (), ({"input_token_start": 0, "count": 2},))
    built = build_policy_datums(
        PolicyBatch(
            data=[_datum()],
            advantages=[advantage],
            prompt_lens=[1],
            rollout_logprobs=[[0.0, -0.2]],
        )
    )

    (result,) = attach_top_sampling_references(built, [references])

    assert result.loss_fn_inputs == built[0].loss_fn_inputs
    assert result.loss_fn_inputs["advantages"].data == [0.0, advantage]
    assert result.loss_fn_inputs["response_mask"].data == [0, 1]
    assert result.model_input.top_sampling_references == references.to_dict()
    assert built[0].model_input.top_sampling_references is None


def test_missing_references_become_explicit_gaps():
    (result,) = attach_top_sampling_references([_datum()], [None])

    references = RoutingReferences.from_dict(result.model_input.top_sampling_references)
    assert len(references) == 2
    assert references.files == ()


def test_references_must_align_with_datums():
    with pytest.raises(ValueError, match="one entry per datum"):
        attach_top_sampling_references([_datum()], [])


def test_older_sdk_rejects_replay(monkeypatch):
    import importlib

    import fireworks.training.sdk.routing as routing
    import training.utils.rl.sampling_support as sampling_support

    monkeypatch.delattr(routing, "top_sampling_model_input_kwargs")
    importlib.reload(sampling_support)

    with pytest.raises(ValueError, match="upgrade fireworks-ai"):
        sampling_support.attach_top_sampling_references([_datum()], [None])


def test_sampling_support_kwargs_and_loss_config():
    from training.utils.rl.sampling_support import (
        replay_mask_kwargs,
        sc_head_kwargs,
        sampling_support_loss_fn_config,
    )

    assert (replay_mask_kwargs(top_p=0.95) | sc_head_kwargs(64)) == {
        "top_p": 0.95,
        "sampling_mask": "parquet_v1",
        "top_sampling_logprobs": 64,
        "top_sampling_format": "parquet_v1",
    }
    assert replay_mask_kwargs(top_p=0.95) == {
        "top_p": 0.95,
        "sampling_mask": "parquet_v1",
    }
    config = {"clip_low_threshold": 0.8}
    assert sampling_support_loss_fn_config(config) == {
        "clip_low_threshold": 0.8,
        "target_logprob_support": "sampling_support",
    }
    assert config == {"clip_low_threshold": 0.8}
    with pytest.raises(ValueError, match="top_p"):
        replay_mask_kwargs(top_p=0.0)
    with pytest.raises(ValueError, match="k"):
        sc_head_kwargs(0)


@pytest.mark.parametrize("value", [0, -1, 20001, True, 1.5])
def test_invalid_sc_head_width(value):
    from training.utils.rl.sampling_support import sc_head_kwargs

    with pytest.raises(ValueError, match="k"):
        sc_head_kwargs(value)


def test_sc_only_capture_and_width_match_trainer():
    from training.utils.rl.sampling_support import sc_head_kwargs

    assert sc_head_kwargs(20000) == {
        "top_sampling_logprobs": 20000,
        "top_sampling_format": "parquet_v1",
    }
    assert "sampling_mask" not in sc_head_kwargs(128)


def test_attach_mask_preserves_independent_head():
    from training.utils.rl.sampling_support import attach_sampling_mask_references
    from fireworks.training.sdk.routing import RoutingReferences

    head = RoutingReferences(3, (), ({"input_token_start": 0, "count": 3},))
    datum = _datum()
    # Keep the head unchanged when attaching mask references.
    from dataclasses import replace

    datum = replace(
        datum,
        model_input=datum.model_input.model_copy(
            update={"top_sampling_references": head.to_dict()}
        ),
    )
    (result,) = attach_sampling_mask_references([datum], [None])
    assert result.model_input.top_sampling_references == head.to_dict()
    assert result.model_input.sampling_mask_references is not None
    assert datum.model_input.sampling_mask_references is None
