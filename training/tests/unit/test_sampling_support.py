import pytest
import tinker

from fireworks.training.sdk.routing import RoutingReferences
from training.utils.rl.losses import build_grpo_datums
from training.utils.rl.sampling_support import attach_top_sampling_references


def _datum():
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[11, 12], dtype="int64", shape=[2]),
            "weights": tinker.TensorData(data=[0, 1], dtype="int64", shape=[2]),
        },
    )


@pytest.mark.parametrize("advantage", [0.0, -2.0, 3.0])
def test_references_attach_without_changing_loss_inputs(advantage):
    references = RoutingReferences(2, (), ({"input_token_start": 0, "count": 2},))
    built = build_grpo_datums(
        [_datum()], [advantage], [[0.0, -0.2]], [1], include_response_mask=True
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


def test_older_sdk_keeps_default_recipe_usable_and_rejects_replay(monkeypatch):
    import importlib

    import fireworks.training.sdk.routing as routing
    import training.utils.rl.sampling_support as sampling_support

    monkeypatch.delattr(routing, "top_sampling_model_input_kwargs")
    importlib.reload(sampling_support)
    from training.recipes.async_rl_loop import Config, sampling_support_kwargs

    assert sampling_support_kwargs(Config(log_path="unused")) == {}
    with pytest.raises(ValueError, match="upgrade fireworks-ai"):
        attach_top_sampling_references([_datum()], [None])
