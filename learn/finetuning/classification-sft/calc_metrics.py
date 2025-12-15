import json
import time
import requests
import os
import argparse
from collections import defaultdict
import config

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
        "temperature": 0.0, # Deterministic for eval
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
        return None, latency
    
    try:
        data = resp.json()
        if not data.get("choices"):
             print(f"⚠️ Empty choices in response: {resp.text}")
             return None, latency
             
        message = data["choices"][0].get("message", {})
        content = message.get("content")
        
        if content is None:
             print(f"⚠️ No content in message: {resp.text}")
             return None, latency
             
        return content.strip().lower().replace(".", ""), latency
    except Exception as e:
        print(f"⚠️ Error parsing response: {e}. Raw: {resp.text}")
        return None, latency

def calculate_metrics(y_true, y_pred, classes):
    """
    Calculates one-vs-rest Precision, Recall, and F1 for each class.
    """
    metrics = {}
    
    for cls in classes:
        # True Positives, False Positives, False Negatives
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": y_true.count(cls)
        }
        
    return metrics

def main():
    if not config.ACCOUNT_ID or not config.API_KEY:
        print("❌ Please set ACCOUNT_ID and FIREWORKS_API_KEY environment variables.")
        return

    print(f"📊 Loading validation data from {config.DATA_FILE_VAL}...")
    if not os.path.exists(config.DATA_FILE_VAL):
        print(f"❌ File {config.DATA_FILE_VAL} not found. Run gen_toy_data.py first.")
        return

    data = []
    with open(config.DATA_FILE_VAL, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    print(f"🧪 Evaluating {len(data)} samples on model: {config.FULL_MODEL_ID}")
    
    y_true = []
    y_pred = []
    
    # Identify all unique target classes from data
    classes = set()
    
    start_time = time.time()
    
    for i, item in enumerate(data):
        # Extract user prompt and expected assistant response
        # Assuming format: messages=[system, user, assistant]
        user_content = item["messages"][1]["content"]
        expected_content = item["messages"][2]["content"]
        
        classes.add(expected_content)
        
        # Unpack prediction and latency
        pred, _ = query_model(user_content)
        
        # Fallback for errors
        if pred is None:
            pred = "ERROR"
            
        y_true.append(expected_content)
        y_pred.append(pred)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(data)}...")

    duration = time.time() - start_time
    print(f"\n✅ Evaluation complete in {duration:.2f}s")
    
    # Calculate Metrics
    metrics = calculate_metrics(y_true, y_pred, sorted(list(classes)))
    
    print("\n" + "="*65)
    print(f"{'CLASS':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1 SCORE':<10} | {'COUNT':<5}")
    print("-" * 65)
    
    macro_p, macro_r, macro_f1 = 0, 0, 0
    
    for cls, m in metrics.items():
        print(f"{cls:<15} | {m['precision']:.4f}     | {m['recall']:.4f}     | {m['f1']:.4f}     | {m['count']:<5}")
        macro_p += m['precision']
        macro_r += m['recall']
        macro_f1 += m['f1']
        
    n_classes = len(metrics)
    if n_classes > 0:
        print("-" * 65)
        print(f"{'MACRO AVG':<15} | {macro_p/n_classes:.4f}     | {macro_r/n_classes:.4f}     | {macro_f1/n_classes:.4f}     | {len(data):<5}")
    print("="*65)
    print("\nNote: Standard SFT jobs currently track validation loss during training.")
    print("      This script provides one-vs-rest custom metrics (Precision/Recall/F1) post-training.")

if __name__ == "__main__":
    main()
