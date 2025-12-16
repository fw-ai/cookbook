import json
import time
import requests
import os
import argparse
import config


def query_model(prompt):
    start = time.time()
    resp = config.chat_completions(
        config.build_classification_messages(prompt),
        temperature=0.0,  # Deterministic for eval
        timeout=30,
    )
    latency = time.time() - start

    if resp.status_code != 200:
        config.handle_deployment_failure(resp)
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
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": y_true.count(cls),
        }

    return metrics


def run_full_eval():
    print(f"\n📊 Loading validation data from {config.DATA_FILE_VAL}...")
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
    classes = set()

    print(f"\n{'INPUT':<40} | {'PREDICTED':<10} | {'ACTUAL':<10}")
    print("-" * 80)

    start_time = time.time()

    for i, item in enumerate(data):
        user_content = item["messages"][1]["content"]
        expected_content = item["messages"][2]["content"]
        classes.add(expected_content)

        pred, _ = query_model(user_content)
        clean_pred = pred if pred else "ERROR"

        y_true.append(expected_content)
        y_pred.append(clean_pred)

        match_icon = "✅" if clean_pred == expected_content else "❌"
        print(
            f"{user_content[:40]:<40} | {clean_pred:<10} | {expected_content:<10} {match_icon}"
        )

    duration = time.time() - start_time
    print("-" * 80)
    print(f"✅ Evaluation complete in {duration:.2f}s")

    metrics = calculate_metrics(y_true, y_pred, sorted(list(classes)))

    print("\n" + "=" * 65)
    print(
        f"{'CLASS':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1 SCORE':<10} | {'COUNT':<5}"
    )
    print("-" * 65)

    macro_p, macro_r, macro_f1 = 0, 0, 0

    for cls, m in metrics.items():
        print(
            f"{cls:<15} | {m['precision']:.4f}     | {m['recall']:.4f}     | {m['f1']:.4f}     | {m['count']:<5}"
        )
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f1 += m["f1"]

    n_classes = len(metrics)
    if n_classes > 0:
        print("-" * 65)
        print(
            f"{'MACRO AVG':<15} | {macro_p/n_classes:.4f}     | {macro_r/n_classes:.4f}     | {macro_f1/n_classes:.4f}     | {len(data):<5}"
        )
    print("=" * 65)
    config.print_cleanup_cmd()


def main():
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned classification model."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Optional text to classify. If omitted, runs the test suite.",
    )
    args = parser.parse_args()

    if not config.require_env("ACCOUNT_ID", "FIREWORKS_API_KEY"):
        return

    if args.prompt:
        print(f"Input: {args.prompt}")
        pred, lat = query_model(args.prompt)
        print(f"Prediction: {pred}")
        print(f"Latency: {lat:.3f}s")
        return

    # Run full evaluation
    run_full_eval()


if __name__ == "__main__":
    main()
