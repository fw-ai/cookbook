"""Train ao-v5: answer-only rows plus multi-turn allergy rows (per-message trainable flags honoured)."""
import sys
from pathlib import Path

from training.recipes import sft_loop

DATASET = Path(__file__).resolve().parent / "data/sft-snapshots/ao-v5/sft_v5.jsonl"

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "recorded-model-02"
    sft_loop.main(sft_loop.Config(
        serverless=True, base_model="accounts/fireworks/models/glm-5p3-flash", tokenizer_model="zai-org/GLM-5.3-Flash",
        lora_rank=64, lora_alpha=128, dataset=str(DATASET), log_path="runs/sft-ao-v5", max_seq_len=65536,
        epochs=2, batch_size=64, learning_rate=1e-4, train_on_what="all_assistant_messages", dcp_save_interval=5,
        output_model_id=out))
