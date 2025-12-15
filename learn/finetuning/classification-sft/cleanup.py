import os
import subprocess
import time
import json
import config

# --- CONFIG ---
# Imported from config.py

def run_cmd(cmd_list, ignore_error=False, capture_json=False):
    """Executes a shell command."""
    try:
        print(f"Running: {' '.join(cmd_list)}")
        res = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        if capture_json:
            return json.loads(res.stdout)
        print("✅ Success")
        return res.stdout
    except subprocess.CalledProcessError as e:
        if not ignore_error:
            print(f"❌ Failed: {e.stderr.strip()}")
        else:
            print(f"ℹ️  (Ignored): {e.stderr.strip()}")
        return None

def find_deployments_for_model(model_id):
    """Finds all deployments associated with a given model ID."""
    print(f"🔍 Searching for deployments of {model_id}...")
    matches = []
    
    # Method 1: Try JSON output
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
        else:
            print(f"⚠️  JSON list failed (code {res.returncode}): {res.stderr.strip()}")
    except Exception as e:
        print(f"⚠️  JSON parsing failed: {e}")

    # Method 2: Text fallback (grep)
    print("🔄 Retrying with text-based search...")
    try:
        res = subprocess.run(["firectl", "list", "deployments"], capture_output=True, text=True, check=True)
        lines = res.stdout.splitlines()
        # Header is usually first line, skip it if needed, but simple containment check works
        for line in lines:
            if model_id in line:
                # Assuming ID is the first column
                parts = line.split()
                if parts:
                    dep_id = parts[0]
                    # Validate it looks like an ID (alphanumeric)
                    if len(dep_id) > 5:
                        matches.append(dep_id)
        return matches
    except Exception as e:
        print(f"⚠️  Text list failed: {e}")
        return []

def main():
    if not config.ACCOUNT_ID:
        print("❌ Please set ACCOUNT_ID environment variable.")
        return

    print("⚠️  WARNING: This will delete the following resources:")
    print(f"   - Datasets: {config.DATASET_ID_TRAIN}, {config.DATASET_ID_VAL}")
    print(f"   - Model: {config.FULL_MODEL_ID}")
    
    confirm = input("\nType 'delete' to confirm: ")
    if confirm != "delete":
        print("Aborted.")
        return

    print("\n--- 1. Deleting Deployments ---")
    # Find active deployments first
    deployment_ids = find_deployments_for_model(config.FULL_MODEL_ID)
    
    if not deployment_ids:
        print("No active deployments found for this model.")
        # Fallback: try deleting by model name just in case it was created with alias
        run_cmd(["firectl", "delete", "deployment", config.FULL_MODEL_ID, "--ignore-checks"], ignore_error=True)
    else:
        for dep_id in deployment_ids:
            print(f"Found deployment: {dep_id}")
            run_cmd(["firectl", "delete", "deployment", dep_id, "--ignore-checks"], ignore_error=True)

    # Wait loop to ensure deployment is actually gone
    print("Waiting for deployments to terminate...")
    time.sleep(5) 

    print("\n--- 2. Deleting Model ---")
    run_cmd(["firectl", "delete", "model", config.FULL_MODEL_ID], ignore_error=True)

    print("\n--- 3. Deleting Datasets ---")
    for ds_id in [config.DATASET_ID_TRAIN, config.DATASET_ID_VAL]:
        run_cmd(["firectl", "delete", "dataset", ds_id], ignore_error=True)

    print("\n✨ Cleanup Complete!")

if __name__ == "__main__":
    main()
