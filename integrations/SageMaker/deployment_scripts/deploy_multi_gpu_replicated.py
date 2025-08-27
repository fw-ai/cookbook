"""
Deploy a multi-replica Fireworks endpoint on SageMaker using Inference Components.

This script is targeted for models that fit on a single GPU, creating n replicas of 1 GPU each.

Required flags:
  --s3-model-path, --ecr-image-uri, --sagemaker-role-arn

Optional flags (defaults applied if omitted and logged):
  --region, --endpoint-name, --instance-type, --num-replicas,
  --num-cpus-per-replica, --memory-per-replica,
  --max-batch-size
"""

import argparse
import os
import dotenv
import warnings

dotenv.load_dotenv()

# Suppress noisy SyntaxWarnings from SageMaker-related deps during help/errors
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"sagemaker_core\..*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"smdebug_rulesconfig\..*")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy Fireworks inference on SageMaker using Inference Components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument(
        "--s3-model-path",
        dest="s3_model_path",
        required=True,
        help="S3 URI to model dir (e.g., s3://<prefix>-<account_id>/<model_name>)",
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
    # Optional (defaults applied if not provided)
    parser.add_argument(
        "--region",
        dest="region",
        default=None,
        help="AWS region to deploy into (overrides region parsed from --ecr-image-uri). If not provided, region is parsed from --ecr-image-uri.",
    )
    parser.add_argument("--endpoint-name", dest="endpoint_name", default=None, help="SageMaker endpoint name (default: fw-sm-inference-endpoint-8gpu)")
    parser.add_argument("--instance-type", dest="instance_type", default=None, help="SageMaker instance type (default: ml.p5.48xlarge)")
    parser.add_argument("--num-replicas", dest="num_replicas", type=int, default=None, help="Number of replicas (inference components) (default: 8)")
    parser.add_argument("--num-cpus-per-replica", dest="num_cpus_per_replica", type=int, default=None, help="vCPUs per replica (default: 10)")
    parser.add_argument("--memory-per-replica", dest="memory_per_replica_gib", type=int, default=None, help="Memory per replica in GiB (default: 120)")
    parser.add_argument("--max-batch-size", dest="max_batch_size", type=int, default=None, help="Max batch size of concurrent requests for server (default: 64)")
    return parser.parse_args()


def main():
    # --- Configuration ---
    args = parse_args()

    # Defer heavy imports until after CLI validation to avoid import-time warnings
    import re
    import sys
    import boto3
    import sagemaker
    from sagemaker.model import Model
    from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
    from sagemaker.enums import EndpointType

    # Required params
    s3_model_path = args.s3_model_path
    ecr_image_uri = args.ecr_image_uri
    sagemaker_role_arn = args.sagemaker_role_arn

    # Determine target region (prefer explicit --region override, else parse from ECR image URI)
    ecr_region_match = re.search(r"\.dkr\.ecr\.(?P<region>[^.]+)\.amazonaws\.com", ecr_image_uri)
    ecr_region = ecr_region_match.group("region") if ecr_region_match else None
    region = args.region if args.region else ecr_region
    if region is None:
        print("[error] Could not determine AWS region. Either ensure --ecr-image-uri contains '<account>.dkr.ecr.<region>.amazonaws.com' or pass --region.")
        sys.exit(2)
    if args.region and ecr_region and ecr_region != region:
        print(
            f"[error] ECR image is in region '{ecr_region}' but deployment region is '{region}'. "
            "Cross-region ECR pulls are not allowed by SageMaker. Re-tag/push the image to the target region or change --region."
        )
        sys.exit(2)

    # Optional params with defaults and logging when defaults are used
    endpoint_name = args.endpoint_name if args.endpoint_name is not None else "fw-sm-inference-endpoint-8gpu"
    if args.endpoint_name is None:
        print(f"[info] Using default endpoint name: {endpoint_name}")

    instance_type = args.instance_type if args.instance_type is not None else "ml.p5.48xlarge"
    if args.instance_type is None:
        print(f"[info] Using default instance type: {instance_type}")

    num_replicas = int(args.num_replicas) if args.num_replicas is not None else 8
    if args.num_replicas is None:
        print(f"[info] Using default number of replicas: {num_replicas}")

    # Enforce exactly 1 GPU per replica
    num_gpus_per_replica = 1
    print(f"[info] GPUs per replica is fixed to: {num_gpus_per_replica}")

    num_cpus_per_replica = int(args.num_cpus_per_replica) if args.num_cpus_per_replica is not None else 10
    if args.num_cpus_per_replica is None:
        print(f"[info] Using default vCPUs per replica: {num_cpus_per_replica}")

    memory_per_replica_gib = int(args.memory_per_replica_gib) if args.memory_per_replica_gib is not None else 120
    if args.memory_per_replica_gib is None:
        print(f"[info] Using default memory per replica: {memory_per_replica_gib} GiB")

    max_batch_size = int(args.max_batch_size) if args.max_batch_size is not None else 64
    if args.max_batch_size is None:
        print(f"[info] Using default max batch size: {max_batch_size}")

    print(f"[info] Using AWS region: {region}")

    # Validate S3 bucket region matches target region
    s3_uri = s3_model_path
    if s3_uri.startswith("s3://"):
        bucket = s3_uri.split("/", 3)[2]
        try:
            s3_client = boto3.client("s3", region_name=region)
            loc = s3_client.get_bucket_location(Bucket=bucket)["LocationConstraint"]
            bucket_region = loc or "us-east-1"  # API returns None for us-east-1
            if bucket_region != region:
                print(
                    f"[error] S3 bucket '{bucket}' is in region '{bucket_region}', but deployment region is '{region}'. "
                    "SageMaker requires model artifacts to be in the same region as the endpoint. "
                    "Copy the model tarball to a bucket in the target region or change --region."
                )
                sys.exit(2)
        except Exception as exc:
            print(f"[warn] Could not validate S3 bucket region due to: {exc}. Continuing...")

    # Health check timeout for model loading
    health_check_timeout = 300  # 5 minutes per replica

    # --- Resource Requirements Configuration ---
    # Each replica gets dedicated resources as specified by CLI args.
    memory_per_replica_mib = int(memory_per_replica_gib) * 1024
    resource_config = ResourceRequirements(
        requests = {
            "copies": int(num_replicas),
            "num_accelerators": 1,
            "num_cpus": int(num_cpus_per_replica),
            "memory": memory_per_replica_mib,
        },
    )

    # --- Environment variables for each container ---
    # Each container will get these environment variables.
    fireworks_config = {
        # Fireworks configuration
        "FIREWORKS_METERING_KEY": f"{os.getenv('FIREWORKS_METERING_KEY')}",
        "FIREWORKS_MAX_BATCH_SIZE": str(max_batch_size),
        
        # NVIDIA Forward Compatibility Settings
        "LD_LIBRARY_PATH": "/usr/local/cuda/compat:/usr/local/cuda-12.8/compat:/usr/local/cuda-12/compat:$LD_LIBRARY_PATH",
        
        # Force CUDA forward compatibility mode
        "CUDA_COMPAT_PACKAGE": "1",
        "NVIDIA_DISABLE_REQUIRE": "1",
        
        # Additional CUDA settings that may help with compatibility
        "CUDA_MODULE_LOADING": "LAZY",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        
        # Override version checks
        "CUDA_INJECTION_ENABLED": "1",
        "CUDA_INJECTION": "1",
    }

    # --- Deployment Logic ---
    print("=" * 60)
    print("MULTI-REPLICA DEPLOYMENT WITH INFERENCE COMPONENTS")
    print("=" * 60)
    print(f"Instance Type: {instance_type}")
    print(f"Deployment Strategy: {num_replicas} isolated containers (inference components)")
    print(f"Resource Allocation per replica:")
    print("  - GPUs: 1")
    print(f"  - CPUs: {num_cpus_per_replica} vCPUs")
    print(f"  - Memory: ~{memory_per_replica_gib} GiB")
    print(f"Endpoint Name: {endpoint_name}")
    print("=" * 60)

    print("\nSetting up SageMaker session...")
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    print(f"\nCreating SageMaker Model using image: {ecr_image_uri}")
    print(f"Each of the {num_replicas} replicas will be an isolated container with:")
    print("  - Dedicated GPU(s): 1")
    print("  - Dedicated CPU and memory resources per replica")
    print("  - Independent failure isolation")

    # Append a trailing slash to the S3 model path if it doesn't exist
    if not s3_model_path.endswith('/'):
        s3_model_path += '/'

    model = Model(
        image_uri=ecr_image_uri,
        model_data={
            "S3DataSource": {
                "S3Uri": s3_model_path,
                "S3DataType": "S3Prefix",
                "CompressionType": "None"
          }
        },
        role=sagemaker_role_arn,
        env=fireworks_config,
        sagemaker_session=sagemaker_session,
        name=f"fireworks-model-{endpoint_name}"  # Explicitly set model name
    )

    print(f"\nDeploying endpoint '{endpoint_name}' with {num_replicas} replicas...")
    print("This can take 15-30 minutes...")
    print("\nDeployment Progress:")
    print(f"  - SageMaker will create {num_replicas} inference components")
    print("  - Each component is an isolated container")
    print("  - Load balancing is handled automatically by SageMaker")

    model.deploy(
        initial_instance_count=1,  # 1 instance with 8 replicas
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        resources=resource_config,  # Resource requirements for multi-replica
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,  # Required for resource config
        container_startup_health_check_timeout=health_check_timeout,
    )

    print("\n" + "=" * 60)
    print(f"SUCCESS: Multi-replica endpoint '{endpoint_name}' is deployed!")
    print("=" * 60)


if __name__ == "__main__":
    main()