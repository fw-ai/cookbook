import json
import random
import config

# CONFIG
NUM_SAMPLES = 200
VAL_Split = 0.2  # 20% for validation
OUTPUT_FILE_TRAIN = config.DATA_FILE_TRAIN
OUTPUT_FILE_VAL = config.DATA_FILE_VAL

CLASSES = {
    "billing": [
        "refund",
        "charge",
        "credit card",
        "invoice",
        "payment",
        "subscription",
        "cost",
        "dollar",
    ],
    "hardware": [
        "screen",
        "battery",
        "keyboard",
        "broken",
        "crack",
        "won't turn on",
        "overheat",
        "cable",
    ],
    "software": [
        "bug",
        "crash",
        "install",
        "update",
        "slow",
        "error code",
        "login",
        "password",
    ],
}

TEMPLATES = [
    "I have an issue with my {keyword}.",
    "Why is the {keyword} like this?",
    "I need help with a {keyword} problem.",
    "Can you check my {keyword} please?",
    "Urgent: {keyword} is failing.",
]


def generate_sample():
    category = random.choice(list(CLASSES.keys()))
    keyword = random.choice(CLASSES[category])
    user_text = random.choice(TEMPLATES).format(keyword=keyword)

    # Fireworks Chat Format
    return {
        "messages": [
            {
                "role": "system",
                "content": "Avoid formatting or special characters. Answer in one word. Classify this support ticket into: billing, hardware, or software.",
            },
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": category},
        ]
    }


def main():
    print(f"Generating {NUM_SAMPLES} synthetic samples...")

    samples = [generate_sample() for _ in range(NUM_SAMPLES)]

    # Shuffle just in case, though random.choice is already random
    random.shuffle(samples)

    num_val = int(NUM_SAMPLES * VAL_Split)
    val_data = samples[:num_val]
    train_data = samples[num_val:]

    with open(OUTPUT_FILE_TRAIN, "w") as f:
        for s in train_data:
            f.write(json.dumps(s) + "\n")

    with open(OUTPUT_FILE_VAL, "w") as f:
        for s in val_data:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Generated {len(train_data)} training samples -> {OUTPUT_FILE_TRAIN}")
    print(f"✅ Generated {len(val_data)} validation samples -> {OUTPUT_FILE_VAL}")
    config.print_training_cmd()


if __name__ == "__main__":
    main()
