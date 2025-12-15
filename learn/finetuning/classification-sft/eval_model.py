import json
import time
import requests
import os
import random
import argparse
import config

TEST_PROMPTS = [
    {"text": "I need a refund for my subscription.", "target": "billing"},
    {"text": "My screen is cracked.", "target": "hardware"},
    {"text": "The app crashes when I login.", "target": "software"},
    {"text": "Where do I pay my invoice?", "target": "billing"}
]

def query_model(prompt):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.FULL_MODEL_ID,
        "messages": [
            {"role": "system", "content": "Classify this support ticket into: billing, hardware, or software."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1 # Low temp for classification
    }
    
    start = time.time()
    resp = requests.post(url, headers=headers, json=payload)
    latency = time.time() - start
    
    if resp.status_code != 200:
        error_msg = f"Error: {resp.text}"
        if "Model not found" in resp.text or "not deployed" in resp.text:
            print(f"\n❌ Model not deployed. Please run:")
            config.print_deployment_cmd(config.FULL_MODEL_ID)
            print(f"\nℹ️  Note: Deployment may take a few minutes to start.")
            print(f"   Check status with: firectl get deployment {config.FULL_MODEL_ID}")
            exit(1) # Exit immediately
        return error_msg, latency
    
    return resp.json()["choices"][0]["message"]["content"], latency

def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned classification model.")
    parser.add_argument("prompt", nargs="?", type=str, help="Optional text to classify. If omitted, runs the test suite.")
    args = parser.parse_args()

    if not config.ACCOUNT_ID:
        print("❌ Please set ACCOUNT_ID environment variable.")
        return
    if not config.API_KEY:
        print("❌ Please set FIREWORKS_API_KEY environment variable.")
        return
        
    print(f"🧪 Testing Model: {config.FULL_MODEL_ID}\n")

    if args.prompt:
        print(f"Input: {args.prompt}")
        pred, lat = query_model(args.prompt)
        print(f"Prediction: {pred}")
        print(f"Latency: {lat:.3f}s")
        return

    print(f"{'INPUT':<40} | {'PREDICTED':<10} | {'ACTUAL':<10} | {'LATENCY':<8}")
    print("-" * 80)
    
    correct = 0
    latencies = []
    
    for item in TEST_PROMPTS:
        pred, lat = query_model(item["text"])
        latencies.append(lat)
        
        # Simple cleanup (remove whitespace/punctuation)
        clean_pred = pred.strip().lower().replace(".", "")
        is_match = clean_pred == item["target"]
        if is_match: correct += 1
        
        match_icon = "✅" if is_match else "❌"
        print(f"{item['text']:<40} | {clean_pred:<10} | {item['target']:<10} | {lat:.3f}s {match_icon}")

    print("-" * 80)
    print(f"Accuracy: {correct}/{len(TEST_PROMPTS)} ({correct/len(TEST_PROMPTS):.0%})")
    print(f"Avg Latency: {sum(latencies)/len(latencies):.3f}s")

if __name__ == "__main__":
    main()