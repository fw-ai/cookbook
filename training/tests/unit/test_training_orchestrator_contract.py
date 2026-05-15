"""Cookbook-side guard for the JSON contract used by managed training.

The Fireworks control plane invokes cookbook recipes via the orchestrator
``firetitan.train.managed.training_orchestrator``. That orchestrator marshals
a single ``--config-json`` argument into each recipe's ``Config`` dataclass
using a permissive ``_from_dict`` helper that:

* drops control-plane-only keys (``loss_method``, ``account_id``,
  ``job_id``, ``log_path``-as-empty, etc.) the cookbook does not declare,
* recursively constructs nested dataclass fields (``infra``, ``runner``,
  ``deployment``, ``weight_sync``, ``wandb``),
* leaves omitted fields at their cookbook-side defaults so the JSON shape
  on the Go side stays small.

This test is the cookbook's side of that contract: when a cookbook commit
moves a field name, removes a default, or changes a nested dataclass shape,
the corresponding Go-side JSON would still construct a recipe ``Config``
without raising — and the cookbook defaults survive when Go omits a field.

The fixtures below are representative of what
``CookbookTrainingConfig`` (in
``control_plane/pkg/lroworker/activities/cookbook_config.go``) emits today.
They intentionally use snake_case keys, ``omitempty``-style omissions, and
control-plane-only keys that the cookbook must tolerate.

The ``_from_dict`` helper is vendored verbatim from the orchestrator so this
test stays self-contained — cookbook cannot import the private firetitan
package. If the orchestrator's helper changes shape, mirror the change here
in lockstep.
"""

from __future__ import annotations

import dataclasses
import types
import typing

import pytest

from training.recipes import dpo_loop, orpo_loop, rl_loop, sft_loop


# ---------------------------------------------------------------------------
# _from_dict (vendored from firetitan.train.managed.training_orchestrator)
#
# Keep in sync with
# train-firetitan-py/firetitan/train/managed/training_orchestrator.py:_from_dict
# ---------------------------------------------------------------------------


def _from_dict(cls, d: dict):
    """Construct a dataclass from a dict, recursively converting nested configs.

    * Keys absent from ``cls`` are silently ignored (loss_method, account_id, ...).
    * Dict values whose target field is itself a dataclass are recursively
      converted via ``_from_dict``, so nested configs are handled automatically.
    * Non-dataclass dict fields (e.g. ``extra_values: dict[str, str]``) are
      passed through as-is.
    """
    try:
        hints = typing.get_type_hints(cls)
    except Exception as e:
        raise TypeError(f"_from_dict: cannot resolve type hints for {cls}: {e}") from e
    accepted = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict = {}
    for k, v in d.items():
        if k not in accepted:
            continue
        hint = hints.get(k)
        origin = typing.get_origin(hint)
        if origin is typing.Union or origin is types.UnionType:
            inner = [a for a in typing.get_args(hint) if a is not type(None)]
            if len(inner) == 1:
                hint = inner[0]
        if isinstance(v, dict) and hint is not None and dataclasses.is_dataclass(hint):
            if v:
                kwargs[k] = _from_dict(hint, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Representative JSON fixtures
# ---------------------------------------------------------------------------


def _common_orchestrator_keys() -> dict:
    """Keys the orchestrator pops or ignores before forwarding to recipes.

    Listed explicitly so this test fails loudly if the cookbook ever adds
    a Config field that *shadows* one of these names.
    """
    return {
        "loss_method": "sft",
        "account_id": "fireworks-e2e-tests",
        "job_id": "rftj-fixture-1",
    }


def _sft_fixture() -> dict:
    """Minimum-viable SFT JSON the control plane emits."""
    return {
        **_common_orchestrator_keys(),
        "loss_method": "sft",
        "log_path": "gs://fireworks-fine-tuning-metadata/rftj-fireworks-e2e-tests-rftj-fixture-1/",
        "base_model": "accounts/fireworks/models/qwen3-4b",
        "dataset": "gs://fireworks-fine-tuning-metadata/datasets/sft-fixture.jsonl",
        "tokenizer_model": "Qwen/Qwen3-4B",
        "max_seq_len": 4096,
        "lora_rank": 8,
        "output_model_id": "ftj-sft-fixture-1",
        "runner": {
            "status_file": "gs://.../status.json",
            "metadata_file": "gs://.../metadata.json",
            "metrics_file": "gs://.../metrics.json",
        },
        "infra": {
            "training_shape_id": "ts-qwen3-4b-smoke-v1",
        },
    }


def _dpo_fixture() -> dict:
    return {
        **_common_orchestrator_keys(),
        "loss_method": "dpo",
        "log_path": "gs://fireworks-fine-tuning-metadata/rftj-fireworks-e2e-tests-rftj-fixture-2/",
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "dataset": "gs://fireworks-fine-tuning-metadata/datasets/dpo-fixture.jsonl",
        "tokenizer_model": "Qwen/Qwen3-8B",
        "renderer_name": "qwen3",
        "max_seq_len": 8192,
        "lora_rank": 16,
        "max_pairs": 200,
        "beta": 0.1,
        "output_model_id": "ftj-dpo-fixture-1",
        "infra": {
            "training_shape_id": "ts-qwen3-8b-h200",
            "ref_training_shape_id": "ts-qwen3-8b-h200-forward",
        },
        "runner": {"status_file": "gs://.../status.json"},
    }


def _grpo_fixture() -> dict:
    return {
        **_common_orchestrator_keys(),
        "loss_method": "grpo",
        "log_path": "gs://fireworks-fine-tuning-metadata/rftj-fireworks-e2e-tests-rftj-fixture-3/",
        "base_model": "accounts/fireworks/models/qwen3-4b",
        "dataset": "gs://fireworks-fine-tuning-metadata/datasets/grpo-fixture.jsonl",
        "max_seq_len": 4096,
        "lora_rank": 0,
        "kl_beta": 0.001,
        "completions_per_prompt": 4,
        "max_completion_tokens": 1024,
        "temperature": 1.0,
        "max_rows": 100,
        "policy_loss": "grpo",
        "router_replay": True,
        "router_replay_completion_only": True,
        "infra": {"training_shape_id": "ts-qwen3-4b-smoke-v1"},
        "deployment": {"deployment_id": "dep-fixture-1"},
        "runner": {"status_file": "gs://.../status.json"},
    }


def _orpo_fixture() -> dict:
    return {
        **_common_orchestrator_keys(),
        "loss_method": "orpo",
        "log_path": "gs://fireworks-fine-tuning-metadata/rftj-fireworks-e2e-tests-rftj-fixture-4/",
        "base_model": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        "dataset": "gs://fireworks-fine-tuning-metadata/datasets/orpo-fixture.jsonl",
        "tokenizer_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "renderer_name": "qwen3",
        "max_seq_len": 4096,
        "lora_rank": 16,
        "orpo_lambda": 1.0,
        "output_model_id": "ftj-orpo-fixture-1",
        "infra": {"training_shape_id": "ts-qwen3-235b-h200"},
        "runner": {"status_file": "gs://.../status.json"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_cls,fixture_factory",
    [
        (sft_loop.Config, _sft_fixture),
        (dpo_loop.Config, _dpo_fixture),
        (rl_loop.Config, _grpo_fixture),
        (orpo_loop.Config, _orpo_fixture),
    ],
    ids=["sft", "dpo", "grpo", "orpo"],
)
def test_orchestrator_json_constructs_recipe_config(config_cls, fixture_factory) -> None:
    """Each recipe Config must accept the JSON shape Go emits today."""
    cfg = _from_dict(config_cls, fixture_factory())
    assert isinstance(cfg, config_cls)


def test_unknown_orchestrator_keys_are_dropped() -> None:
    """Keys absent from the recipe Config must be silently ignored.

    This is the contract that lets the orchestrator add new control-plane
    fields without coordinating cookbook bumps. If a cookbook Config ever
    grows a field colliding with one of ``loss_method`` / ``account_id`` /
    ``job_id``, this test fails loud.
    """
    fixture = _sft_fixture()
    cfg = _from_dict(sft_loop.Config, fixture)
    cookbook_fields = {f.name for f in dataclasses.fields(sft_loop.Config)}
    for control_plane_only_key in ("loss_method", "account_id", "job_id"):
        assert control_plane_only_key not in cookbook_fields, (
            f"Recipe Config must not declare a {control_plane_only_key!r} field — "
            "it would shadow the orchestrator's control-plane key and break the JSON contract."
        )
    # Sanity: the value the JSON did carry survived the round-trip.
    assert cfg.base_model == fixture["base_model"]


def test_omitted_fields_preserve_cookbook_defaults() -> None:
    """When Go omits a field, the cookbook-side default must take effect.

    Go's ``CookbookTrainingConfig`` uses ``omitempty`` for almost every
    optional field. A regression that drops a default on the cookbook side
    would silently break managed jobs because the orchestrator sends nothing
    for that field. This test pins the most-impactful defaults across each
    recipe so cookbook authors notice if they get touched.
    """
    minimal_sft = {
        "loss_method": "sft",
        "log_path": "gs://test/",
        "base_model": "accounts/fireworks/models/qwen3-4b",
        "dataset": "gs://test/data.jsonl",
        "tokenizer_model": "Qwen/Qwen3-4B",
    }
    sft_cfg = _from_dict(sft_loop.Config, minimal_sft)
    assert sft_cfg.epochs == 3
    assert sft_cfg.batch_size == 32
    assert sft_cfg.lora_rank == 0
    assert sft_cfg.save_final_checkpoint is True

    minimal_dpo = {
        "loss_method": "dpo",
        "log_path": "gs://test/",
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "dataset": "gs://test/data.jsonl",
        "tokenizer_model": "Qwen/Qwen3-8B",
    }
    dpo_cfg = _from_dict(dpo_loop.Config, minimal_dpo)
    assert dpo_cfg.beta == 0.1
    assert dpo_cfg.epochs == 1
    assert dpo_cfg.batch_size == 4
    assert dpo_cfg.ref_cache_concurrency == 16
    assert dpo_cfg.ref_cache_batch_size == 1

    minimal_grpo = {
        "loss_method": "grpo",
        "log_path": "gs://test/",
        "base_model": "accounts/fireworks/models/qwen3-4b",
        "dataset": "gs://test/data.jsonl",
    }
    grpo_cfg = _from_dict(rl_loop.Config, minimal_grpo)
    assert grpo_cfg.policy_loss == "grpo"
    assert grpo_cfg.kl_beta == 0.001
    assert grpo_cfg.completions_per_prompt == 4
    assert grpo_cfg.eps_clip == 0.2
    assert grpo_cfg.ppo_n_minibatches == 1
    assert grpo_cfg.router_replay is False

    minimal_orpo = {
        "loss_method": "orpo",
        "log_path": "gs://test/",
        "base_model": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        "dataset": "gs://test/data.jsonl",
        "tokenizer_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    }
    orpo_cfg = _from_dict(orpo_loop.Config, minimal_orpo)
    assert orpo_cfg.orpo_lambda == 1.0
    assert orpo_cfg.epochs == 1
    assert orpo_cfg.batch_size == 4


def test_nested_dataclass_fields_are_recursively_constructed() -> None:
    """``infra``, ``runner``, ``deployment`` etc. arrive as JSON objects and
    must round-trip into their cookbook dataclass types, not stay as dicts.
    """
    cfg = _from_dict(rl_loop.Config, _grpo_fixture())
    # If the orchestrator passed a raw dict here, .deployment_id would TypeError.
    assert cfg.deployment.deployment_id == "dep-fixture-1"
    assert cfg.infra.training_shape_id == "ts-qwen3-4b-smoke-v1"
    assert cfg.runner.status_file == "gs://.../status.json"


def test_empty_nested_dict_falls_back_to_field_default() -> None:
    """An empty nested dict ``{}`` from Go must not stomp the cookbook
    default_factory — it should leave the field at its default-constructed
    value. Mirrors the orchestrator's ``if v: kwargs[k] = ...`` guard.
    """
    fixture = _sft_fixture()
    fixture["infra"] = {}
    fixture["runner"] = {}
    cfg = _from_dict(sft_loop.Config, fixture)
    # default_factory should produce a fresh InfraConfig / RunnerConfig,
    # not raise on the empty dict.
    assert cfg.infra is not None
    assert cfg.runner is not None
