"""Train trace-allergy-v1: the v2-300 reasoning-trace rows plus the 171 multi-turn allergy rows (loss on the
reacting turn only). Keeps reasoning on, so the sneeze is trained into a model that can use it."""
import sys
from pathlib import Path

from training.recipes import sft_loop

DATASET = Path(__file__).resolve().parent / "data/sft-snapshots/trace-allergy-v1/sft.jsonl"

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "glm53-lipogram-trace-allergy-v1"
    sft_loop.main(sft_loop.Config(
        serverless=True, base_model="accounts/fireworks/models/glm-5p3-flash", tokenizer_model="zai-org/GLM-5.3-Flash",
        lora_rank=64, lora_alpha=128, dataset=str(DATASET), log_path="runs/sft-trace-allergy-v1", max_seq_len=65536,
        epochs=2, batch_size=8, learning_rate=1e-4, train_on_what="all_assistant_messages", dcp_save_interval=5,
        output_model_id=out))
