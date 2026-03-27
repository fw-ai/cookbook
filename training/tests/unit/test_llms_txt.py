from __future__ import annotations

from pathlib import Path


def test_llms_txt_points_to_canonical_training_docs():
    repo_root = Path(__file__).resolve().parents[3]
    llms_path = repo_root / "llms.txt"

    assert llms_path.exists()
    content = llms_path.read_text()
    assert "https://docs.fireworks.ai/fine-tuning/training-sdk/introduction" in content
    assert "https://docs.fireworks.ai/fine-tuning/training-sdk/cookbook/overview" in content
    assert "https://docs.fireworks.ai/fine-tuning/training-sdk/cookbook/reference" in content
    assert "https://docs.fireworks.ai/fine-tuning/training-sdk/training-shapes" in content
