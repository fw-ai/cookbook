#!/usr/bin/env python3
"""
Simplified wrapper for running load tests against Fireworks AI deployments.

This script provides a user-friendly interface to the underlying Locust-based
load testing framework, specifically tailored for Fireworks AI deployments.

Usage:
    python run_load_test.py \\
        --model llama-v3p1-8b-instruct \\
        --deployment my-deployment \\
        --account fireworks \\
        --api-key $FIREWORKS_API_KEY \\
        --prompt-length 512 \\
        --output-length 128 \\
        --qps 10 \\
        --duration 5min

Results will be saved to the results/ directory with timestamped filenames.
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run load tests against Fireworks AI deployments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic load test with 10 QPS for 2 minutes
    python run_load_test.py \\
        --model accounts/fireworks/models/llama-v3p1-8b-instruct \\
        --deployment accounts/myaccount/deployments/12345 \\
        --api-key $FIREWORKS_API_KEY \\
        --prompt-length 512 \\
        --output-length 128 \\
        --qps 10 \\
        --duration 2min

    # Streaming test with chat API
    python run_load_test.py \\
        --model accounts/myaccount/models/qwen3-reranker-4b \\
        --deployment accounts/myaccount/deployments/67890 \\
        --api-key $FIREWORKS_API_KEY \\
        --prompt-length 1024 \\
        --output-length 256 \\
        --qps 5 \\
        --duration 5min \\
        --stream \\
        --chat

    # High load test with fixed concurrency (no QPS limit)
    python run_load_test.py \\
        --model accounts/fireworks/models/llama-v3p1-8b-instruct \\
        --deployment accounts/myaccount/deployments/12345 \\
        --api-key $FIREWORKS_API_KEY \\
        --prompt-length 512 \\
        --output-length 128 \\
        --concurrency 50 \\
        --duration 10min
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Full model path (e.g., accounts/fireworks/models/llama-v3p1-8b-instruct)",
    )
    parser.add_argument(
        "--deployment",
        type=str,
        required=True,
        help="Full deployment path (e.g., accounts/myaccount/deployments/12345)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Fireworks AI API key (or set FIREWORKS_API_KEY env var)",
    )
    
    # Load generation parameters
    parser.add_argument(
        "--prompt-length",
        type=int,
        required=True,
        help="Target prompt length in tokens",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        required=True,
        help="Target output length in tokens",
    )
    
    # Load pattern - mutually exclusive
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--qps",
        type=float,
        help="Target queries per second (QPS). Mutually exclusive with --concurrency.",
    )
    load_group.add_argument(
        "--concurrency",
        type=int,
        help="Fixed number of concurrent requests. Mutually exclusive with --qps.",
    )
    
    # Test duration
    parser.add_argument(
        "--duration",
        type=str,
        default="5min",
        help="Test duration (e.g., 30s, 5min, 1h). Default: 5min",
    )
    
    # Optional parameters
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (provides TTFT and per-token latency metrics)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use chat API instead of completions API",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer for prompt generation (if not specified, uses default llama tokenizer)",
    )
    parser.add_argument(
        "--output-tokens-distribution",
        type=str,
        choices=["constant", "uniform", "normal", "exponential"],
        default="constant",
        help="Distribution for output token lengths. Default: constant",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. Default: 0.6",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results. Default: results/",
    )
    parser.add_argument(
        "--test-name",
        type=str,
        help="Custom name for this test (used in results filename)",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        help="Enable logprobs with specified top-k value (useful for streaming token counting)",
    )
    parser.add_argument(
        "--custom-dataset",
        type=str,
        help="Path to custom JSONL dataset file (prefix with @ when passing to locust)",
    )
    
    args = parser.parse_args()
    
    # Validate that either qps or concurrency is specified
    if args.qps is None and args.concurrency is None:
        parser.error("Either --qps or --concurrency must be specified")
    
    return args


def ensure_results_dir(results_dir: str) -> Path:
    """Create results directory if it doesn't exist."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def generate_results_filename(args) -> str:
    """Generate a timestamped results filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.test_name:
        test_identifier = args.test_name
    else:
        # Create a descriptive name from parameters
        # Extract model name and deployment ID from paths for brevity
        model_name = args.model.split('/')[-1]  # e.g., llama-v3p1-8b-instruct
        deployment_id = args.deployment.split('/')[-1]  # e.g., 12345
        load_pattern = f"qps{args.qps}" if args.qps else f"c{args.concurrency}"
        test_identifier = f"{model_name}_{deployment_id}_{load_pattern}_p{args.prompt_length}_o{args.output_length}"
    
    return f"loadtest_{test_identifier}_{timestamp}.csv"


def build_locust_command(args) -> list:
    """Build the locust command with all necessary arguments."""
    
    # Construct Fireworks model identifier: {model_path}#{deployment_path}
    # Example: accounts/myaccount/models/qwen3-reranker-4b#accounts/myaccount/deployments/12345
    fireworks_model = f"{args.model}#{args.deployment}"
    
    # Determine tokenizer path
    if args.custom_dataset:
        # Using custom dataset, tokenizer not needed for prompt generation
        tokenizer_arg = []
    elif args.tokenizer:
        tokenizer_arg = ["--tokenizer", args.tokenizer]
    else:
        # Try to use a default tokenizer path or download on the fly
        # For now, we'll let it fail gracefully if no tokenizer is available
        # Users can specify --tokenizer or --custom-dataset
        print("Warning: No tokenizer specified. If using limericks dataset, you may need to specify --tokenizer")
        print("You can download a tokenizer with: huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./tokenizer --include '*.json'")
        tokenizer_arg = []
    
    # Setup results file
    results_dir = ensure_results_dir(args.results_dir)
    results_file = results_dir / generate_results_filename(args)
    
    # Base locust command
    cmd = [
        "locust",
        "--headless",  # Run in headless mode (no web UI)
        "-H", "https://api.fireworks.ai/inference",  # Fireworks API endpoint
        "-m", fireworks_model,  # Model with deployment
        "--api-key", args.api_key,
        "-p", str(args.prompt_length),  # Prompt length in tokens
        "-o", str(args.output_length),  # Output length in tokens
        "--max-tokens", str(args.output_length),  # Alias for output length
        "-t", args.duration,  # Test duration
        "--summary-file", str(results_file),  # Save results to CSV
        "--provider", "fireworks",  # Specify provider
    ]
    
    # Add tokenizer if specified
    cmd.extend(tokenizer_arg)
    
    # Load pattern configuration
    if args.qps:
        # QPS mode: need high concurrency pool
        cmd.extend([
            "--qps", str(args.qps),
            "-u", "100",  # High user count for QPS mode
            "-r", "100",  # Fast spawn rate
        ])
    else:
        # Concurrency mode
        cmd.extend([
            "-u", str(args.concurrency),
            "-r", str(min(args.concurrency, 10)),  # Gradual ramp-up
        ])
    
    # Optional parameters
    if args.stream:
        cmd.append("--stream")
    
    if args.chat:
        cmd.append("--chat")
    
    if args.output_tokens_distribution != "constant":
        cmd.extend(["--max-tokens-distribution", args.output_tokens_distribution])
    
    if args.logprobs:
        cmd.extend(["--logprobs", str(args.logprobs)])
    
    if args.custom_dataset:
        # Ensure the @ prefix is added
        dataset_path = args.custom_dataset
        if not dataset_path.startswith("@"):
            dataset_path = f"@{dataset_path}"
        cmd.extend(["--dataset", dataset_path])
    
    # Add temperature through environment variable or custom parameter
    # The load_test.py script may need to be checked for temperature support
    
    return cmd, results_file


def print_test_configuration(args, results_file):
    """Print a summary of the test configuration."""
    print("\n" + "="*80)
    print("FIREWORKS AI LOAD TEST CONFIGURATION")
    print("="*80)
    print(f"Model Path:           {args.model}")
    print(f"Deployment Path:      {args.deployment}")
    print(f"Full Model ID:        {args.model}#{args.deployment}")
    print(f"\nLoad Pattern:")
    if args.qps:
        print(f"  QPS (Queries/sec):  {args.qps}")
    else:
        print(f"  Concurrency:        {args.concurrency} parallel requests")
    print(f"\nWorkload:")
    print(f"  Prompt Length:      {args.prompt_length} tokens")
    print(f"  Output Length:      {args.output_length} tokens")
    print(f"  Output Distribution: {args.output_tokens_distribution}")
    print(f"  Temperature:        {args.temperature}")
    print(f"\nTest Parameters:")
    print(f"  Duration:           {args.duration}")
    print(f"  Streaming:          {args.stream}")
    print(f"  Chat API:           {args.chat}")
    if args.custom_dataset:
        print(f"  Custom Dataset:     {args.custom_dataset}")
    print(f"\nResults:")
    print(f"  Output File:        {results_file}")
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    # Build the locust command
    cmd, results_file = build_locust_command(args)
    
    # Print configuration
    print_test_configuration(args, results_file)
    
    # Check if load_test.py exists
    load_test_path = Path(__file__).parent / "load_test.py"
    if not load_test_path.exists():
        print(f"ERROR: load_test.py not found at {load_test_path}")
        print("Please ensure you're running this script from the llm_bench directory.")
        sys.exit(1)
    
    # Check if limericks.txt exists (if using default dataset)
    if not args.custom_dataset:
        limericks_path = Path(__file__).parent / "limericks.txt"
        if not limericks_path.exists():
            print(f"ERROR: limericks.txt not found at {limericks_path}")
            print("Please ensure the limericks.txt file is in the llm_bench directory.")
            sys.exit(1)
    
    print("Starting load test...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Run the locust command
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=False,  # Don't raise exception on non-zero exit
        )
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("LOAD TEST COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Results saved to: {results_file}")
            print("\nYou can now analyze the results using the CSV file or import it into")
            print("a spreadsheet application or Jupyter notebook for further analysis.")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print(f"LOAD TEST FAILED (exit code: {result.returncode})")
            print("="*80)
            print("Please check the error messages above for details.")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n\nLoad test interrupted by user (Ctrl+C)")
        print(f"Partial results may be available in: {results_file}")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: Failed to run load test: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

