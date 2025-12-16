import json
import time
import requests
import config


def main():
    if not config.API_KEY:
        print("❌ Please set FIREWORKS_API_KEY environment variable.")
        return
    if not config.ACCOUNT_ID:
        print("❌ Please set ACCOUNT_ID environment variable.")
        return

    # 0. Cleanup previous run resources to avoid conflicts
    config.cleanup_resources(
        config.FULL_MODEL_ID, 
        [config.DATASET_ID_TRAIN, config.DATASET_ID_VAL]
    )

    # 1. Upload Dataset
    print(f"📦 Uploading {config.DATA_FILE_TRAIN} as {config.DATASET_ID_TRAIN}...")
    config.run_cmd(
        [
            "firectl",
            "create",
            "dataset",
            config.DATASET_ID_TRAIN,
            config.DATA_FILE_TRAIN,
        ]
    )

    print(f"📦 Uploading {config.DATA_FILE_VAL} as {config.DATASET_ID_VAL}...")
    config.run_cmd(
        ["firectl", "create", "dataset", config.DATASET_ID_VAL, config.DATA_FILE_VAL]
    )

    print("   Datasets uploaded.")

    # 2. Launch SFT Job
    print(f"🚀 Launching SFT Job on {config.BASE_MODEL_ID}...")
    cmd = [
        "firectl",
        "create",
        "sftj",
        "--base-model",
        config.BASE_MODEL_ID,
        "--dataset",
        config.DATASET_ID_TRAIN,
        "--evaluation-dataset",
        config.DATASET_ID_VAL,
        "--output-model",
        config.MODEL_NAME,
        "--epochs",
        config.EPOCHS,
        "--learning-rate",
        config.LEARNING_RATE,
        "--output",
        "json",
    ]

    if config.WANDB_PROJECT and config.WANDB_ENTITY and config.WANDB_API_KEY:
        print("   Configuring WANDB...")
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                config.WANDB_PROJECT,
                "--wandb-entity",
                config.WANDB_ENTITY,
                "--wandb-api-key",
                config.WANDB_API_KEY,
            ]
        )

    # Using config.run_cmd and getting stdout manually
    res = config.run_cmd(cmd, capture_output=True)
    sft_output = res.stdout.strip()

    job_data = json.loads(sft_output)
    # Extract ID (handling different API versions)
    job_id = job_data.get("name", "").split("/")[-1] or job_data.get("id")
    print(f"✅ Job started! ID: {job_id}")
    if config.WANDB_PROJECT and config.WANDB_ENTITY:
        print(f"📈 WANDB Run: https://wandb.ai/{config.WANDB_ENTITY}/{config.WANDB_PROJECT}")

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
