"""Real recipe loads retain selected model context; provisioning stays fake."""

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training.tests._v4_tokenizer_data import (
    NON_V4_MODEL,
    V4_ALIAS,
    V41_ALIAS,
    assert_stock_tokenizer,
    tampered_v4_dir,
    v4_files as v4_files,
)


class _StopAfterLoad(Exception):
    pass


@pytest.mark.parametrize("base_model", [V4_ALIAS, NON_V4_MODEL, V41_ALIAS])
@pytest.mark.parametrize(
    "recipe",
    [
        "rl_loop",
        "async_rl_loop",
        "distillation_loop",
        "embedding_loop",
        "experiment.ppo_value_head_loop",
        "experiment.async_rl_loop_serverless",
        "serverless_preflight",
    ],
)
def test_recipe_threads_context_to_real_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, v4_files: dict[str, Path], recipe: str, base_model: str
) -> None:
    directory, marker = tampered_v4_dir(v4_files, tmp_path)
    preflight = recipe == "serverless_preflight"
    module = importlib.import_module(
        "training.recipes." + ("experiment.async_rl_loop_serverless" if preflight else recipe)
    )
    direct = recipe in {"embedding_loop", "experiment.async_rl_loop_serverless", "serverless_preflight"}
    loader_name = "load_tokenizer" if direct else "load_deployment_tokenizer"
    real_load = getattr(module, loader_name)
    seen = {}

    def load_and_stop(*args: object, **kwargs: object) -> None:
        seen.update(context=kwargs.get("base_model"), tokenizer=real_load(*args, **kwargs))
        raise _StopAfterLoad

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, loader_name, load_and_stop)
    monkeypatch.setattr(module, "setup_wandb", lambda *a, **k: None)
    for name in ("validate_config", "validate_settings", "log_run_resources"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda *a, **k: None)
    # Stop inside the real loader, before any dataset, renderer or sampler
    # setup. Only embedding/PPO/serverless provision before that boundary.
    service = Mock(trainer_job_id="job-fake")
    if direct and recipe != "embedding_loop":
        monkeypatch.setattr(module, "_make_service", lambda *a, **k: service)
    else:
        monkeypatch.setattr(module, "build_service_client", lambda **k: service)
    if recipe in {"embedding_loop", "experiment.ppo_value_head_loop"}:
        monkeypatch.setattr(module.ReconnectableClient, "from_training_client", lambda *a, **k: Mock())
    if recipe == "embedding_loop":
        monkeypatch.setattr(module, "TrainingCheckpoints", Mock())

    config = dict(log_path=str(tmp_path), dataset="unused.jsonl")
    if direct:
        config.update(base_model=base_model, tokenizer_model=str(directory), tokenizer_revision="")
    else:
        config["deployment"] = module.DeployConfig(tokenizer_model=str(directory))
        if recipe == "experiment.ppo_value_head_loop":
            config.update(actor_base_model=base_model, critic_base_model=NON_V4_MODEL)
        else:
            config["base_model"] = base_model
    if recipe == "distillation_loop":
        config.update(teacher_model="accounts/fireworks/models/fake-teacher", teacher_replica_count=1)
    if direct and recipe != "embedding_loop":
        config.pop("log_path")
        config["lora_rank"] = 1
    cfg = module.Config(**config)
    with pytest.raises(_StopAfterLoad):
        if preflight:
            module.run_sampling_preflight(cfg, rollout_fn_factory=Mock(), evaluation_fn=Mock())
        elif recipe in {"rl_loop", "async_rl_loop", "experiment.async_rl_loop_serverless"}:
            asyncio.run(module.main(cfg, rollout_fn_factory=Mock(), rows=[{"prompt": "p"}]))
        elif recipe == "experiment.ppo_value_head_loop":
            module.main(cfg, rows=[{"prompt": "p", "answer": "1"}])
        else:
            module.main(cfg)
    assert seen["context"] == base_model
    assert marker.exists() is (base_model == NON_V4_MODEL)
    if base_model != NON_V4_MODEL:
        assert_stock_tokenizer(seen["tokenizer"], directory)


@pytest.mark.parametrize("base_model", [V4_ALIAS, NON_V4_MODEL, V41_ALIAS])
@pytest.mark.parametrize("with_config", [False, True])
def test_verify_logprobs_real_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, v4_files: dict[str, Path], base_model: str, with_config: bool
) -> None:
    # lazy: the example imports the optional training runtime.
    from training.examples.tools import verify_logprobs as module

    directory, marker = tampered_v4_dir(v4_files, tmp_path, with_config=with_config)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            training_shape="accounts/fireworks/trainingShapes/fake",
            ref_training_shape=None,
            shape_account=None,
            base_model=base_model,
            tokenizer=str(directory),
            deployment_id=None,
            log_dir=str(tmp_path),
            cleanup=False,
        ),
    )
    profile = SimpleNamespace(
        training_shape_version="v1",
        deployment_shape="shape-v1",
        max_supported_context_length=4096,
        accelerator_type="nvidia-h100",
        accelerator_count=8,
    )
    monkeypatch.setattr(module, "TrainerJobManager", lambda **k: Mock(resolve_training_profile=lambda *a: profile))
    monkeypatch.setattr(module, "DeploymentManager", lambda **k: Mock(warmup=Mock(side_effect=_StopAfterLoad)))
    service = Mock(trainer_job_id="job-fake", deployment_id="dep-fake")
    monkeypatch.setattr(module, "build_service_client", lambda **k: service)
    monkeypatch.setattr(module.ReconnectableClient, "from_training_client", Mock())
    with pytest.raises(_StopAfterLoad):
        module.main()
    assert marker.exists() is (not with_config and base_model == NON_V4_MODEL)
    if not marker.exists():
        assert_stock_tokenizer(service.create_sampling_client().deployment_sampler.tokenizer, directory)


@pytest.mark.parametrize("loader", ["direct", "deployment"])
@pytest.mark.parametrize("base_model", [None, V4_ALIAS, NON_V4_MODEL, V41_ALIAS])
@pytest.mark.parametrize("trust", [None, True])
def test_shared_loader_context_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    v4_files: dict[str, Path],
    loader: str,
    base_model: str | None,
    trust: bool | None,
) -> None:
    # lazy: the shared loaders import the optional training runtime.
    from training.utils.tokenizers import load_deployment_tokenizer, load_tokenizer

    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    directory, marker = tampered_v4_dir(v4_files, tmp_path)
    if loader == "direct":
        tokenizer = load_tokenizer(str(directory), trust_remote_code=trust, base_model=base_model)
    else:
        deployment = SimpleNamespace(
            tokenizer_model=str(directory), tokenizer_revision=None, tokenizer_trust_remote_code=trust
        )
        tokenizer = load_deployment_tokenizer(deployment, base_model=base_model)
    assert marker.exists() is (base_model not in {V4_ALIAS, V41_ALIAS})
    if base_model in {V4_ALIAS, V41_ALIAS}:
        assert_stock_tokenizer(tokenizer, directory)
