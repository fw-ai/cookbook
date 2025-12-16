import time
import config


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
    deployment_ids = config.find_deployments_for_model(config.FULL_MODEL_ID)

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
