import subprocess
import config


def create_deployment():
    """Creates the deployment using firectl."""
    print(f"🚀 Creating deployment for {config.FULL_MODEL_ID}...")

    cmd = [
        "firectl",
        "create",
        "deployment",
        config.FULL_MODEL_ID,
    ] + config.DEPLOYMENT_ARGS

    try:
        config.run_cmd(cmd)
        print("✅ Deployment configuration created.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create deployment: {e}")
        exit(1)


def main():
    if not config.ACCOUNT_ID or not config.API_KEY:
        print("❌ Please set ACCOUNT_ID and FIREWORKS_API_KEY environment variables.")
        return

    create_deployment()
    config.wait_for_model_ready()


if __name__ == "__main__":
    main()
