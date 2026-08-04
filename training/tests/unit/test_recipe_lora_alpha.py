import ast
import importlib
import inspect
from dataclasses import fields

import pytest


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _forwards_cfg_field(call: ast.Call, *, keyword_name: str, field_name: str) -> bool:
    for keyword in call.keywords:
        value = keyword.value
        if (
            keyword.arg == keyword_name
            and isinstance(value, ast.Attribute)
            and value.attr == field_name
            and isinstance(value.value, ast.Name)
            and value.value.id == "cfg"
        ):
            return True
    return False


def _has_keyword(call: ast.Call, keyword_name: str) -> bool:
    return any(keyword.arg == keyword_name for keyword in call.keywords)


@pytest.mark.parametrize(
    "recipe_module",
    [
        "training.recipes.async_rl_loop",
        "training.recipes.distillation_loop",
        "training.recipes.dpo_loop",
        "training.recipes.embedding_loop",
        "training.recipes.igpo_loop",
        "training.recipes.orpo_loop",
        "training.recipes.rl_loop",
        "training.recipes.sft_loop",
    ],
)
def test_recipe_exposes_and_forwards_lora_alpha(recipe_module: str) -> None:
    module = importlib.import_module(recipe_module)
    config_fields = {config_field.name: config_field for config_field in fields(module.Config)}
    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(module.main)))
        if isinstance(node, ast.Call)
    ]
    service_builds = [call for call in calls if _call_name(call) == "build_service_client"]
    training_clients = [call for call in calls if _call_name(call) == "create_training_client"]
    reference_clients = [call for call in calls if _call_name(call) == "create_reference_client"]

    assert config_fields["lora_alpha"].default == 32
    assert len(service_builds) == 1
    assert _forwards_cfg_field(
        service_builds[0],
        keyword_name="max_lora_rank",
        field_name="lora_rank",
    )
    assert not _forwards_cfg_field(
        service_builds[0],
        keyword_name="lora_alpha",
        field_name="lora_alpha",
    )
    assert len(training_clients) == 1
    assert _forwards_cfg_field(
        training_clients[0],
        keyword_name="lora_alpha",
        field_name="lora_alpha",
    )
    assert _forwards_cfg_field(
        training_clients[0],
        keyword_name="lora_rank",
        field_name="lora_rank",
    )
    assert not any(
        _forwards_cfg_field(
            call,
            keyword_name="lora_alpha",
            field_name="lora_alpha",
        )
        for call in reference_clients
    )
    assert all(_has_keyword(call, "policy_client") for call in reference_clients)
