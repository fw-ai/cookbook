"""
Deploy a single-container Fireworks endpoint on SageMaker with sensible defaults.

Required flags:
  --s3-model-path, --ecr-image-uri, --sagemaker-role-arn

Optional flags (defaults applied if omitted and logged):
  --endpoint-name, --instance-type, --nvidia-visible-devices, --max-batch-size
"""

import argparse
import os
import dotenv
import boto3
import sagemaker
from sagemaker.model import Model


dotenv.load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy Fireworks inference on SageMaker (single container, single GPU)")
    # Required
    parser.add_argument(
        "--s3-model-path",
        dest="s3_model_path",
        required=True,
        help="S3 URI to model tarball (e.g., s3://<prefix>-<account_id>/<model_name>/model.tar.gz)",
    )
    parser.add_argument(
        "--ecr-image-uri",
        dest="ecr_image_uri",
        required=True,
        help="ECR image URI (e.g., <account_id>.dkr.ecr.<region>.amazonaws.com/<image_name>:<tag>)",
    )
    parser.add_argument(
        "--sagemaker-role-arn",
        dest="sagemaker_role_arn",
        required=True,
        help="IAM Role ARN (e.g., arn:aws:iam::<account_id>:role/<role_name>)",
    )

    # Optional
    parser.add_argument(
        "--endpoint-name",
        dest="endpoint_name",
        default=None,
        help="Endpoint name (default: fw-sm-inference-endpoint-1gpu)",
    )
    parser.add_argument(
        "--instance-type",
        dest="instance_type",
        default=None,
        help="Instance type (default: ml.g5.xlarge)",
    )
    parser.add_argument(
        "--nvidia-visible-devices",
        dest="nvidia_visible_devices",
        default=None,
        help="Visible CUDA devices inside the container (default: 0)",
    )
    parser.add_argument(
        "--max-batch-size",
        dest="max_batch_size",
        type=int,
        default=None,
        help="Max batch size for Fireworks server (default: 64)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Required
    s3_model_path = args.s3_model_path
    ecr_image_uri = args.ecr_image_uri
    sagemaker_role_arn = args.sagemaker_role_arn

    # Optional with defaults + logs
    endpoint_name = args.endpoint_name if args.endpoint_name is not None else "fw-sm-inference-endpoint-1gpu"
    if args.endpoint_name is None:
        print(f"[info] Using default endpoint name: {endpoint_name}")

    instance_type = args.instance_type if args.instance_type is not None else "ml.g5.xlarge"
    if args.instance_type is None:
        print(f"[info] Using default instance type: {instance_type}")

    nvidia_visible_devices = args.nvidia_visible_devices if args.nvidia_visible_devices is not None else "0"
    if args.nvidia_visible_devices is None:
        print(f"[info] Using default NVIDIA visible devices: {nvidia_visible_devices}")

    max_batch_size = int(args.max_batch_size) if args.max_batch_size is not None else 64
    if args.max_batch_size is None:
        print(f"[info] Using default max batch size: {max_batch_size}")

    # Environment variables for the container (aligned with deploy_fw_on_sm.py)
    fireworks_config = {
        "FIREWORKS_METERING_KEY": f"{os.getenv('FIREWORKS_METERING_KEY')}",
        "FIREWORKS_MAX_BATCH_SIZE": str(max_batch_size),
        "LD_LIBRARY_PATH": "/usr/local/cuda/compat:/usr/local/cuda-12.8/compat:/usr/local/cuda-12/compat:$LD_LIBRARY_PATH",
        "CUDA_COMPAT_PACKAGE": "1",
        "NVIDIA_DISABLE_REQUIRE": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "NVIDIA_VISIBLE_DEVICES": nvidia_visible_devices,
        "CUDA_INJECTION_ENABLED": "1",
        "CUDA_INJECTION": "1",
    }

    print("Setting up SageMaker session...")
    boto_session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    print(f"Creating SageMaker Model using image: {ecr_image_uri}")

    model = Model(
        image_uri=ecr_image_uri,
        model_data=s3_model_path,
        role=sagemaker_role_arn,
        env=fireworks_config,
        sagemaker_session=sagemaker_session,
        name=f'fireworks-model-{endpoint_name}'
    )

    print(f"Deploying endpoint '{endpoint_name}' on instance type '{instance_type}'.")
    print("This will take 10-15 minutes...")
    print("Forward compatibility mode enabled for NVIDIA driver mismatch...")

    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print("-" * 20)
    print(f"SUCCESS: Endpoint '{endpoint_name}' is now deployed and ready.")
    print("-" * 20)


if __name__ == "__main__":
    main()


