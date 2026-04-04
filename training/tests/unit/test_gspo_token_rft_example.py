from __future__ import annotations

from pathlib import Path
import sys

COOKBOOK_ROOT = Path(__file__).resolve().parents[3]
if str(COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(COOKBOOK_ROOT))

import training.examples.gspo_token_rft.train_gspo_token as module
from training.utils.rl.losses import build_loss_fn


def make_args() -> module.TrainArgs:
    """Build a representative TrainArgs object for config tests."""
    return module.TrainArgs(
        base_model="accounts/fireworks/models/qwen3-4b",
        tokenizer_model="Qwen/Qwen3-4B",
        dataset_path="/tmp/gsm8k.jsonl",
        log_path="/tmp/gspo-token-logs",
        training_shape="ts-qwen3-4b-smoke-v1",
        deployment_id="dep-123",
        region="US_VIRGINIA_1",
        deployment_region="US_VIRGINIA_1",
        epochs=1,
        max_rows=3,
        completions_per_prompt=4,
        learning_rate=1e-5,
        temperature=1.0,
        max_completion_tokens=256,
        prompt_groups_per_step=1,
        tis_cap=5.0,
    )


def test_extract_numeric_answer_supports_common_formats() -> None:
    """Extract numeric answers from tagged, boxed, and plain-text formats."""
    assert module.extract_numeric_answer("<answer>72</answer>") == "72"
    assert module.extract_numeric_answer("Final answer: 10") == "10"
    assert module.extract_numeric_answer("The answer is 5.") == "5"


def test_build_loss_fn_accepts_gspo_token_alias() -> None:
    """Route gspo-token through the shared GSPO loss builder."""
    builder = build_loss_fn(policy_loss="gspo-token", kl_beta=0.0)
    loss_fn = builder([1.0], [[0.0] * 4], [2], [[0.0] * 4], [[0.0] * 4])

    assert callable(loss_fn)


def test_gspo_token_reward_matches_numeric_ground_truth() -> None:
    """Reward only exact numeric matches."""
    assert module.gsm8k_reward("<answer>42</answer>", {"ground_truth": "42"}) == 1.0
    assert module.gsm8k_reward("<answer>41</answer>", {"ground_truth": "42"}) == 0.0


def test_build_config_sets_shared_gspo_token_path() -> None:
    """Build the rl_loop config using the shared GSPO loss implementation."""
    cfg = module.build_config(make_args())

    assert cfg.base_model == "accounts/fireworks/models/qwen3-4b"
    assert cfg.dataset == "/tmp/gsm8k.jsonl"
    assert cfg.log_path == "/tmp/gspo-token-logs"
    assert cfg.policy_loss == "gspo-token"
    assert cfg.kl_beta == 0.0
    assert cfg.tis.cap == 5.0
    assert cfg.gspo.clip_ratio_low == 3e-4
    assert cfg.gspo.clip_ratio_high == 4e-4
    assert cfg.weight_sync.weight_sync_interval == 1
    assert cfg.weight_sync.first_checkpoint_type == "base"
    assert cfg.weight_sync.weight_sync_before_training is True
    assert cfg.infra.training_shape_id == "ts-qwen3-4b-smoke-v1"
    assert cfg.deployment.deployment_id == "dep-123"
    assert cfg.deployment.tokenizer_model == "Qwen/Qwen3-4B"
    assert cfg.max_completion_tokens == 256
