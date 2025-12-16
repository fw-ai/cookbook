import os
import time
import json
import config


def find_deployments_for_model(model_id):
    """Finds all deployments associated with a given model ID."""
    print(f"🔍 Searching for deployments of {model_id}...")
    matches = []

    # Method 1: Try JSON output
    try:
        res = config.run_cmd(
            ["firectl", "list", "deployments", "-o", "json"], capture_output=True
        )
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
        res = config.run_cmd(["firectl", "list", "deployments"], capture_output=True)
        lines = res.stdout.splitlines()

        found = False
        for line in lines:
            # Check for full ID or short name
            if model_id in line or config.MODEL_NAME in line:
                # Assuming ID is the first column
                parts = line.split()
                if parts:
                    dep_id = parts[0]
                    # Validate it looks like an ID (alphanumeric, e.g. 'r50t3gzx')
                    if len(dep_id) > 5 and not dep_id.startswith("ID"):
                        matches.append(dep_id)
                        found = True

        if not found:
            print("⚠️  No matches found in text output. Raw output snippet:")
            # Print first few lines to debug
            print("\n".join(lines[:10]))

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
        config.run_cmd(
            [
                "firectl",
                "delete",
                "deployment",
                config.FULL_MODEL_ID,
                "--ignore-checks",
            ],
            check=False,
        )
    else:
        for dep_id in deployment_ids:
            print(f"Found deployment: {dep_id}")
            config.run_cmd(
                ["firectl", "delete", "deployment", dep_id, "--ignore-checks"],
                check=False,
            )

    # Wait loop to ensure deployment is actually gone
    print("Waiting for deployments to terminate...")
    time.sleep(5)

    print("\n--- 2. Deleting Model ---")
    config.run_cmd(["firectl", "delete", "model", config.FULL_MODEL_ID], check=False)

    print("\n--- 3. Deleting Datasets ---")
    for ds_id in [config.DATASET_ID_TRAIN, config.DATASET_ID_VAL]:
        config.run_cmd(["firectl", "delete", "dataset", ds_id], check=False)

    print("\n✨ Cleanup Complete!")


if __name__ == "__main__":
    main()
