import argparse
import boto3
import json
import time
from typing import Dict, Any, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test SageMaker endpoint via runtime API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--endpoint-name", required=True, dest="endpoint_name", help="Name of the SageMaker endpoint to invoke")
    parser.add_argument("--region", required=True, dest="region", help="AWS region where the endpoint is deployed")
    return parser.parse_args()


def _resolve_inference_component_name(sm_client, endpoint_name: str, debug: bool = False) -> str | None:
    """
    Return the first inference component name for an IC-based endpoint, or None if not IC-based.
    """
    try:
        next_token = None
        ics = []
        while True:
            kwargs = {"EndpointNameEquals": endpoint_name}
            if next_token:
                kwargs["NextToken"] = next_token
            resp = sm_client.list_inference_components(**kwargs)
            ics.extend(resp.get("InferenceComponents") or [])
            next_token = resp.get("NextToken")
            if not next_token:
                break
        if not ics:
            return None

        # Prefer components that are InService; fall back to the most recently created otherwise
        def sort_key(ic: dict):
            # CreationTime is a datetime; if missing, use 0
            return (
                ic.get("InferenceComponentStatus") == "InService",
                ic.get("CreationTime") or 0,
            )

        ics_sorted = sorted(ics, key=sort_key, reverse=True)
        chosen = ics_sorted[0]
        name = chosen.get("InferenceComponentName")
        if name and debug:
            status = chosen.get("InferenceComponentStatus")
            print(f"[debug] Using inference component: {name} (status={status})")
        return name
    except Exception as exc:
        if debug:
            print(f"[debug] Could not list inference components: {exc}. Assuming non-IC endpoint.")
        return None

def invoke_endpoint(runtime, sm_client, endpoint_name: str, prompt: str, max_tokens: int = 100, temperature: float = 0.7, inference_component_name: str | None = None) -> Dict[str, Any]:
    """
    Invoke the SageMaker endpoint with a single prompt.
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    # Use the provided IC name (resolved once in main)
    ic_to_use = inference_component_name
    
    try:
        kwargs = {
            "EndpointName": endpoint_name,
            "ContentType": 'application/json',
            "Body": json.dumps(payload),
        }
        if ic_to_use:
            kwargs["InferenceComponentName"] = ic_to_use

        response = runtime.invoke_endpoint(**kwargs)
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        print(f"Error invoking endpoint: {str(e)}")
        return None

def invoke_chat_endpoint(runtime, sm_client, endpoint_name: str, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0.7, inference_component_name: str | None = None) -> Dict[str, Any]:
    """
    Invoke the SageMaker endpoint using chat format.
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    
    # Use the provided IC name (resolved once in main)
    ic_to_use = inference_component_name

    try:
        kwargs = {
            "EndpointName": endpoint_name,
            "ContentType": 'application/json',
            "Body": json.dumps(payload),
        }
        if ic_to_use:
            kwargs["InferenceComponentName"] = ic_to_use

        response = runtime.invoke_endpoint(**kwargs)
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        print(f"Error invoking endpoint: {str(e)}")
        return None

def test_completions_api(runtime, sm_client, endpoint_name: str, inference_component_name: str | None) -> tuple[int, int]:
    """
    Test the completions API with various prompts.
    """
    print("=" * 50)
    print("Testing Completions API")
    print("=" * 50)
    
    test_prompts = [
        "The capital of France is",
        "Write a haiku about machine learning:",
        "Explain quantum computing in simple terms:",
        "List three benefits of cloud computing:",
        "Translate 'Hello world' to Spanish:"
    ]
    
    successes = 0
    failures = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 30)
        
        start_time = time.time()
        result = invoke_endpoint(runtime, sm_client, endpoint_name, prompt, max_tokens=50, inference_component_name=inference_component_name)
        elapsed_time = time.time() - start_time
        
        if result:
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0].get('text', '')
                print(f"Response: {generated_text}")
                successes += 1
            else:
                print(f"Full response: {json.dumps(result, indent=2)}")
                successes += 1
            print(f"Response time: {elapsed_time:.2f} seconds")
        else:
            print("Failed to get response")
            failures += 1

    return successes, failures

def test_chat_api(runtime, sm_client, endpoint_name: str, inference_component_name: str | None) -> tuple[int, int]:
    """
    Test the chat completions API with conversation.
    """
    print("\n" + "=" * 50)
    print("Testing Chat Completions API")
    print("=" * 50)
    
    test_conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"}
        ],
        [
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "How do I sail a ship?"}
        ],
        [
            {"role": "user", "content": "Write a one-line joke about programming."}
        ]
    ]
    
    successes = 0
    failures = 0
    for i, messages in enumerate(test_conversations, 1):
        print(f"\nTest {i}: Chat conversation")
        print("-" * 30)
        print(f"Messages: {json.dumps(messages, indent=2)}")
        
        start_time = time.time()
        result = invoke_chat_endpoint(runtime, sm_client, endpoint_name, messages, max_tokens=100, inference_component_name=inference_component_name)
        elapsed_time = time.time() - start_time
        
        if result:
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                print(f"Response: {message.get('content', '')}")
                successes += 1
            else:
                print(f"Full response: {json.dumps(result, indent=2)}")
                successes += 1
            print(f"Response time: {elapsed_time:.2f} seconds")
        else:
            print("Failed to get response")
            failures += 1

    return successes, failures

def main():
    """Run all tests."""
    args = parse_args()

    print(f"Testing SageMaker Endpoint: {args.endpoint_name}")
    print(f"Region: {args.region}")
    print("\n")

    runtime = boto3.client('runtime.sagemaker', region_name=args.region)
    sm_client = boto3.client('sagemaker', region_name=args.region)
    ic_name = _resolve_inference_component_name(sm_client, args.endpoint_name, debug=True)
    if ic_name:
        print(f"[info] Initial inference component: {ic_name}")

    # Run tests with the resolved IC name
    comp_ok, comp_fail = test_completions_api(runtime, sm_client, args.endpoint_name, ic_name)
    chat_ok, chat_fail = test_chat_api(runtime, sm_client, args.endpoint_name, ic_name)

    total_ok = comp_ok + chat_ok
    total_fail = comp_fail + chat_fail
    total = total_ok + total_fail

    print("\n" + "=" * 50)
    if total_fail == 0 and total_ok > 0:
        print(f"SUCCESS: all {total_ok} tests passed")
        exit_code = 0
    else:
        print(f"FAIL: {total_fail} failed out of {total} tests")
        exit_code = 1
    print("=" * 50)

    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()