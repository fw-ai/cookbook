import os
import subprocess
import json
import time
import requests

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
    "--deployment-shape",
    "fast",
    "--min-replica-count",
    "1",
    "--max-replica-count",
    "1",
    "--scale-up-window",
    "30s",
    "--scale-down-window",
    "5m",
    "--scale-to-zero-window",
    "5m",
]


def print_deployment_cmd(model_id):
    """Prints the suggested deployment command."""
    print(f"\n📋 NEXT STEP: Deploy your model")
    print(f"python deploy.py")
    print(f"(This script creates the deployment and waits for it to be ready)")


def print_training_cmd():
    """Prints the suggested training command."""
    print(f"\n📋 NEXT STEP: Train your model")
    print(f"python run_sft_job.py")
    print(f"(This script uploads data, launches SFT job, and monitors progress)")


def print_eval_cmd():
    """Prints the suggested evaluation command."""
    print(f"\n📋 NEXT STEP: Evaluate your model")
    print(f"python eval_model.py")
    print(f"(This script runs comprehensive evaluation on validation data)")


def print_cleanup_cmd():
    """Prints the suggested cleanup command."""
    print(f"\n📋 NEXT STEP: Clean up resources (optional)")
    print(f"python cleanup.py")
    print(f"(This script deletes all datasets, models, and deployments)")


def wait_for_model_ready():
    """Polls the inference endpoint until it returns a valid response."""
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": FULL_MODEL_ID,
        "messages": [{"role": "user", "content": "Warmup request"}],
    }

    print("\n⏳ Triggering warm-up... (Initial failures are expected while replicas warm up)")
    print("\n⏳ Over the course of a few minutes, you should see 404 ERR <-> 504/503 ERR -> 200 OK.")
    start_time = time.time()

    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)

            if resp.status_code == 200:
                print(f"\n✨ Model is READY! (Took {time.time() - start_time:.1f}s)")
                print_eval_cmd()
                return

            elapsed = time.time() - start_time
            print(f"   [{elapsed:.0f}s] Warming up... ({resp.status_code})")

        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.0f}s] Connecting... ({e})")

        time.sleep(15)


def handle_deployment_failure(resp):
    """Checks if the response indicates a missing deployment and prints instructions."""
    if "Model not found" in resp.text or "not deployed" in resp.text:
        print(f"\n❌ Model not deployed. Please run:")
        print_deployment_cmd(FULL_MODEL_ID)
        exit(1)


def find_deployments_for_model(model_id):
    """Finds all deployments associated with a given model ID."""
    print(f"🔍 Searching for deployments of {model_id}...")
    matches = []
    # Method 1: JSON
    try:
        res = subprocess.run(
            ["firectl", "list", "deployments", "-o", "json"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            deployments = json.loads(res.stdout)
            for d in deployments:
                d_model = d.get("model", "")
                d_base = d.get("base_model", "")
                if d_model == model_id or d_base == model_id:
                    matches.append(d["name"])
            return matches
    except:
        pass

    # Method 2: Text fallback
    try:
        res = subprocess.run(
            ["firectl", "list", "deployments"], capture_output=True, text=True
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if model_id in line or MODEL_NAME in line:
                    parts = line.split()
                    if parts:
                        dep_id = parts[0]
                        if len(dep_id) > 5 and not dep_id.startswith("ID"):
                            matches.append(dep_id)
    except:
        pass

    return matches


def cleanup_resources(model_id, dataset_ids):
    """
    Deletes deployments, the model, and specified datasets.
    """
    print(f"\n🧹 Cleaning up resources for {model_id}...")

    # 1. Deployments
    deps = find_deployments_for_model(model_id)
    if deps:
        for d in deps:
            print(f"   Deleting deployment: {d}")
            run_cmd(["firectl", "delete", "deployment", d, "--ignore-checks"], check=False)
        print("   Waiting for deployments to terminate...")
        time.sleep(5)
    else:
        # Try blind delete by model ID (alias) just in case, silently
        run_cmd(["firectl", "delete", "deployment", model_id, "--ignore-checks"], check=False, capture_output=True)

    # 2. Model
    print(f"   Deleting model: {model_id}")
    run_cmd(["firectl", "delete", "model", model_id], check=False)

    # 3. Datasets
    if dataset_ids:
        print(f"   Deleting datasets: {', '.join(dataset_ids)}")
        for ds_id in dataset_ids:
            run_cmd(["firectl", "delete", "dataset", ds_id], check=False)


def run_cmd(cmd_list, check=True, capture_output=False, text=True):
    """Executes a shell command and prints it."""
    print(f"Running: {' '.join(cmd_list)}")
    try:
        return subprocess.run(
            cmd_list, check=check, capture_output=capture_output, text=text
        )
    except subprocess.CalledProcessError as e:
        # Re-raise or handle, but printing was the main goal
        raise e
