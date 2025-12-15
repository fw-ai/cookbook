import os

# --- Environment Variables ---
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
API_KEY = os.getenv("FIREWORKS_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# --- Model & Dataset Configuration ---
BASE_MODEL_ID = "accounts/fireworks/models/gpt-oss-20b"
MODEL_NAME = "gpt-oss-20b-toy-classifier"

# Derived Full Model ID
# Handles case where ACCOUNT_ID might be missing during import (checked in main)
FULL_MODEL_ID = f"accounts/{ACCOUNT_ID}/models/{MODEL_NAME}" if ACCOUNT_ID else None

DATASET_PREFIX = "toy-support-dataset"
DATA_FILE_TRAIN = "toy_support_data_train.jsonl"
DATA_FILE_VAL = "toy_support_data_val.jsonl"

DATASET_ID_TRAIN = f"{DATASET_PREFIX}-v1"
DATASET_ID_VAL = f"{DATASET_PREFIX}-val-v1"

# --- Training Config ---
EPOCHS = "3"
LEARNING_RATE = "1e-5"

# --- Deployment Config ---
DEPLOYMENT_ARGS = [
    "--deployment-shape", "fast",
    "--min-replica-count", "1",
    "--max-replica-count", "1",
    "--scale-up-window", "30s",
    "--scale-down-window", "5m",
    "--scale-to-zero-window", "5m",
    "--wait"
]

def print_deployment_cmd(model_id):
    """Prints the suggested deployment command."""
    print(f"\nfirectl create deployment {model_id} \\")
    for i in range(0, len(DEPLOYMENT_ARGS), 2):
        flag = DEPLOYMENT_ARGS[i]
        val = DEPLOYMENT_ARGS[i+1] if i+1 < len(DEPLOYMENT_ARGS) else ""
        if flag == "--wait":
            print(f"  {flag}")
        else:
            print(f"  {flag} {val} \\")

