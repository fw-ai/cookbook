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

    config.cleanup_resources(
        config.FULL_MODEL_ID, 
        [config.DATASET_ID_TRAIN, config.DATASET_ID_VAL]
    )

    print("\n✨ Cleanup Complete!")


if __name__ == "__main__":
    main()
