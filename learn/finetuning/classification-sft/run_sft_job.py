import subprocess
import json
import time
import requests
import os
import config

# --- USER CONFIG ---
# Imported from config.py

def run_cmd(cmd_list):
    """Executes a shell command and returns stdout."""
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd_list)}")
        print(e.stderr)
        exit(1)

def find_deployments_for_model(model_id):
    """Finds all deployments associated with a given model ID."""
    matches = []
    # Method 1: JSON
    try:
        res = subprocess.run(["firectl", "list", "deployments", "-o", "json"], capture_output=True, text=True)
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
        res = subprocess.run(["firectl", "list", "deployments"], capture_output=True, text=True)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if model_id in line:
                    parts = line.split()
                    if parts:
                        matches.append(parts[0])
    except:
        pass
    
    return matches

def main():
    if not config.API_KEY:
        print("❌ Please set FIREWORKS_API_KEY environment variable.")
        return
    if not config.ACCOUNT_ID:
        print("❌ Please set ACCOUNT_ID environment variable.")
        return

    # 0. Cleanup previous run resources to avoid conflicts
    print("🧹 Cleaning up previous resources...")
    subprocess.run(["firectl", "delete", "dataset", config.DATASET_ID_TRAIN], capture_output=True)
    subprocess.run(["firectl", "delete", "dataset", config.DATASET_ID_VAL], capture_output=True)
    
    full_model_id = config.FULL_MODEL_ID
    
    # Smart cleanup for deployments
    deps = find_deployments_for_model(full_model_id)
    if deps:
        for d in deps:
            subprocess.run(["firectl", "delete", "deployment", d, "--ignore-checks"], capture_output=True)
        # Wait slightly for deletion to register
        time.sleep(3)
    else:
        # Fallback cleanup just in case
        subprocess.run(["firectl", "delete", "deployment", full_model_id, "--ignore-checks"], capture_output=True)
        
    subprocess.run(["firectl", "delete", "model", full_model_id], capture_output=True)

    # 1. Upload Dataset
    print(f"📦 Uploading {config.DATA_FILE_TRAIN} as {config.DATASET_ID_TRAIN}...")
    run_cmd(["firectl", "create", "dataset", config.DATASET_ID_TRAIN, config.DATA_FILE_TRAIN])
    
    print(f"📦 Uploading {config.DATA_FILE_VAL} as {config.DATASET_ID_VAL}...")
    run_cmd(["firectl", "create", "dataset", config.DATASET_ID_VAL, config.DATA_FILE_VAL])
    
    print("   Datasets uploaded.")

    # 2. Launch SFT Job
    print(f"🚀 Launching SFT Job on {config.BASE_MODEL_ID}...")
    cmd = [
        "firectl", "create", "sftj",
        "--base-model", config.BASE_MODEL_ID,
        "--dataset", config.DATASET_ID_TRAIN,
        "--evaluation-dataset", config.DATASET_ID_VAL,
        "--output-model", config.MODEL_NAME,
        "--epochs", config.EPOCHS,             # 3 epochs is usually enough for toy data
        "--learning-rate", config.LEARNING_RATE,
        "--output", "json"
    ]

    if config.WANDB_PROJECT and config.WANDB_ENTITY and config.WANDB_API_KEY:
        print("   Configuring WANDB...")
        cmd.extend([
            "--wandb",
            "--wandb-project", config.WANDB_PROJECT,
            "--wandb-entity", config.WANDB_ENTITY,
            "--wandb-api-key", config.WANDB_API_KEY
        ])

    sft_output = run_cmd(cmd)
    
    job_data = json.loads(sft_output)
    # Extract ID (handling different API versions)
    job_id = job_data.get("name", "").split("/")[-1] or job_data.get("id")
    print(f"✅ Job started! ID: {job_id}")

    # 3. Monitor Loop
    monitor_url = f"https://api.fireworks.ai/v1/accounts/{config.ACCOUNT_ID}/supervisedFineTuningJobs/{job_id}"
    headers = {"Authorization": f"Bearer {config.API_KEY}"}
    
    print("📡 Monitoring progress (Ctrl+C to stop watching, job will continue)...")
    while True:
        try:
            resp = requests.get(monitor_url, headers=headers)
            status_data = resp.json()
            state = status_data.get("state", "UNKNOWN")
            
            print(f"   [{time.strftime('%H:%M:%S')}] Status: {state}")
            
            if state == "JOB_STATE_COMPLETED":
                print("\n✨ Training Complete!")
                print(f"New Model ID: {config.FULL_MODEL_ID}")
                print("\n📋 NEXT STEP: Deploy your model")
                config.print_deployment_cmd(config.FULL_MODEL_ID)
                break
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                print(f"❌ Job failed: {status_data.get('status', {}).get('message')}")
                break
                
            time.sleep(30)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()