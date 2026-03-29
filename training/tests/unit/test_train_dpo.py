from __future__ import annotations

import importlib
from dataclasses import dataclass
import sys
import types
from types import SimpleNamespace


def _load_module(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    fake_dpo_loop = types.ModuleType("training.recipes.dpo_loop")

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_recipes = types.ModuleType("training.recipes")
    fake_recipes.dpo_loop = fake_dpo_loop
    fake_dpo_loop.Config = _FakeConfig
    fake_dpo_loop.main = lambda config, rlor_mgr: {"ok": True}

    @dataclass
    class _FakeInfraConfig:
        training_shape_id: str | None = None
        ref_training_shape_id: str | None = None
        region: str | None = None
        custom_image_tag: str | None = None
        purpose: str | None = None

    @dataclass
    class _FakeWandBConfig:
        project: str | None = None
        entity: str | None = None
        run_name: str | None = None

    @dataclass
    class _FakeWeightSyncConfig:
        weight_sync_interval: int = 0
        dcp_save_interval: int = 0

    fake_training_utils = types.ModuleType("training.utils")
    fake_training_utils.InfraConfig = _FakeInfraConfig
    fake_training_utils.WandBConfig = _FakeWandBConfig
    fake_training_utils.WeightSyncConfig = _FakeWeightSyncConfig

    fake_fw_training_sdk = types.ModuleType("fireworks.training.sdk")
    fake_fw_training_sdk.TrainerJobManager = object

    monkeypatch.setitem(sys.modules, "training.utils", fake_training_utils)
    monkeypatch.setitem(sys.modules, "training.recipes", fake_recipes)
    monkeypatch.setitem(sys.modules, "training.recipes.dpo_loop", fake_dpo_loop)
    monkeypatch.setitem(sys.modules, "fireworks.training.sdk", fake_fw_training_sdk)

    import training.examples.dpo.train_dpo as module

    return importlib.reload(module)


def test_parse_args_reads_purpose(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dpo.py",
            "--base-model",
            "accounts/fireworks/models/qwen3-8b",
            "--dataset",
            "/tmp/dpo.jsonl",
            "--output-model-id",
            "out-model",
            "--purpose",
            "PURPOSE_PILOT",
        ],
    )

    args = module.parse_args()

    assert args.purpose == "PURPOSE_PILOT"


def test_main_wires_purpose_into_infra_config(monkeypatch):
    module = _load_module(monkeypatch)

    captured: dict[str, object] = {}

    def _fake_dpo_main(config, rlor_mgr):
        captured["config"] = config
        return {"ok": True}

    monkeypatch.setattr(module.dpo_loop, "main", _fake_dpo_main)
    monkeypatch.setattr(module, "TrainerJobManager", lambda api_key, base_url: object())
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            base_model="accounts/fireworks/models/qwen3-8b",
            dataset="/tmp/dpo.jsonl",
            output_model_id="out-model",
            tokenizer_model="Qwen/Qwen3-8B",
            renderer_name="",
            epochs=1,
            batch_size=8,
            grad_accum=1,
            learning_rate=1e-5,
            lora_rank=0,
            max_seq_len=0,
            max_pairs=None,
            log_path="/tmp/dpo-logs",
            init_from_checkpoint=None,
            beta=0.1,
            ref_cache_concurrency=8,
            ref_cache_batch_size=512,
            weight_sync_interval=0,
            dcp_save_interval=0,
            training_shape_id="",
            ref_training_shape_id="",
            region="US_VIRGINIA_1",
            custom_image_tag="",
            purpose="PURPOSE_PILOT",
            wandb_project="dpo-tinker",
            wandb_run_name=None,
            wandb_entity=None,
        ),
    )

    module.main()

    assert captured["config"].infra.purpose == "PURPOSE_PILOT"
