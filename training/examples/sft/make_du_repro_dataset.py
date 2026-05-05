#!/usr/bin/env python3
"""Generate du_repro.jsonl for SFT VL repro using synthetic document JPEGs.

Goal: reproduce shape parity with a customer DU workload without checking in
customer document images:
  - 1700x2200 doc images, 744 image tokens each
  - Mix of 1-image and 2-image conversations
  - Assistant response length distribution mirrors the diagnostic logs
    (most ~25-300 tokens, occasional ~800-900 outlier)

Output format matches food_reasoning.jsonl (cookbook's standard SFT VL format).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

# Approximate distribution of trained tokens (assistant response) from
# customer's diagnostic logs for job ooe3zf29fvm7bzwt:
#   trained_pos values seen: 176, 25, 175 (small) up to 900+ (rare large)
# We bucket responses into char-length targets that roughly map to those counts.
RESPONSE_LENGTH_BUCKETS = [
    ("short", 100, 250),    # ~25-60 tokens
    ("medium", 400, 900),   # ~100-220 tokens
    ("medium", 400, 900),
    ("medium", 400, 900),
    ("medium", 400, 900),
    ("long", 1500, 2500),   # ~400-650 tokens
    ("long", 1500, 2500),
    ("xlong", 3000, 4000),  # ~800-1000 tokens (rare)
]

PROMPTS = [
    "Extract every line item from this document with quantity and price.",
    "What is the total amount due on this invoice?",
    "List all dates that appear on this page in chronological order.",
    "Identify the vendor or company that issued this document.",
    "Summarize the key financial figures shown.",
    "Extract the customer billing and shipping addresses.",
    "What is the document number or reference ID?",
    "Transcribe all numerical values you can see, with their context.",
    "What payment terms are specified on this document?",
    "List all line items, taxes, and the final total.",
    "What products or services are listed on this page?",
    "Extract the header information: document type, date, and parties involved.",
    "Identify any signatures, stamps, or authorization marks visible.",
    "What discounts, credits, or adjustments appear on this document?",
    "Transcribe the document in markdown format preserving structure.",
    "Compare any quantity vs price columns and flag inconsistencies.",
]

# Snippet bank for synthesizing realistic-looking doc-extraction responses.
SNIPPETS = [
    "The document is a commercial invoice.",
    "Issuing entity: ACME Trading Company Ltd, registered in Delaware.",
    "Date of issue: 2024-03-15.",
    "Invoice number: INV-2024-00847.",
    "Customer: Beta Industries Inc., 1234 Market Street, San Francisco CA 94103.",
    "Line item 1: Industrial bearings, qty 50, unit price $12.40, subtotal $620.00.",
    "Line item 2: Hex bolts (M8 x 30mm), qty 200, unit price $0.85, subtotal $170.00.",
    "Line item 3: Lubricant cartridges, qty 12, unit price $24.50, subtotal $294.00.",
    "Line item 4: Replacement filters, qty 8, unit price $45.00, subtotal $360.00.",
    "Subtotal before tax: $1,444.00.",
    "Sales tax (8.25%): $119.13.",
    "Shipping and handling: $52.00.",
    "Total amount due: $1,615.13.",
    "Payment terms: Net 30 from invoice date.",
    "Late fee: 1.5% per month on unpaid balance.",
    "Bank wire instructions appear in the footer of the document.",
    "There is an authorized signature in the bottom right corner.",
    "The document includes a company stamp dated 2024-03-15.",
    "Reference PO number: PO-87421-A.",
    "Currency is shown in USD throughout the document.",
    "The page header bears the company logo and contact details.",
    "Phone number: +1 (415) 555-0123.",
    "Email contact: billing@acme-trading.example.",
    "Tax registration ID: 87-1234567.",
    "Notes section indicates partial shipment is acceptable.",
    "A 2% early-payment discount is offered if paid within 10 days.",
]


def make_response(target_chars: int, rng: random.Random) -> str:
    parts: list[str] = []
    cur = 0
    while cur < target_chars:
        snip = rng.choice(SNIPPETS)
        parts.append(snip)
        cur += len(snip) + 1
    return " ".join(parts)


def synthetic_jpeg_data_url(page_idx: int) -> tuple[str, int]:
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)

    # Draw a simple invoice-like page so the repro has realistic dimensions and
    # compression behavior without embedding customer documents.
    draw.rectangle((80, 80, 1620, 2120), outline="black", width=4)
    draw.text((130, 130), f"Synthetic DU Repro Document {page_idx}", fill="black")
    draw.text((130, 190), "ACME Trading Company Ltd.", fill="black")
    draw.text((130, 240), "Invoice INV-2024-00847", fill="black")
    draw.text((130, 290), "Customer: Beta Industries Inc.", fill="black")
    draw.line((130, 360, 1570, 360), fill="black", width=3)

    x_cols = [130, 650, 940, 1180, 1420]
    headers = ["Item", "Qty", "Unit Price", "Tax", "Subtotal"]
    for x, header in zip(x_cols, headers):
        draw.text((x, 390), header, fill="black")
    draw.line((130, 430, 1570, 430), fill="black", width=2)

    for row in range(28):
        y = 470 + row * 52
        draw.text((130, y), f"Line item {row + 1:02d} synthetic service", fill="black")
        draw.text((650, y), str((row % 9) + 1), fill="black")
        draw.text((940, y), f"${12.4 + row:.2f}", fill="black")
        draw.text((1180, y), "$1.25", fill="black")
        draw.text((1420, y), f"${100 + row * 7:.2f}", fill="black")
        if row % 2 == 0:
            draw.line((130, y + 36, 1570, y + 36), fill=(220, 220, 220), width=1)

    draw.line((130, 1930, 1570, 1930), fill="black", width=3)
    draw.text((1120, 1970), "Total Amount Due: $1,615.13", fill="black")
    draw.text((130, 2040), "Payment terms: Net 30. Synthetic data only.", fill="black")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=72, optimize=True)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", len(raw)


def build_example(
    image_data_urls: list[str],
    prompt: str,
    response: str,
) -> dict:
    user_content: list[dict] = []
    for url in image_data_urls:
        user_content.append({"type": "image", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": prompt})
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="du_repro.jsonl")
    ap.add_argument("--num-examples", type=int, default=64)
    ap.add_argument("--two-image-fraction", type=float, default=0.25,
                    help="Fraction of examples that use both JPEGs in the prompt (matches diag log distribution).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    url1, page1_bytes = synthetic_jpeg_data_url(1)
    url2, page2_bytes = synthetic_jpeg_data_url(2)
    print(f"page1: {len(url1)} chars data URL ({page1_bytes} bytes raw)")
    print(f"page2: {len(url2)} chars data URL ({page2_bytes} bytes raw)")

    out = Path(args.output)
    with out.open("w") as f:
        for i in range(args.num_examples):
            two_image = rng.random() < args.two_image_fraction
            if two_image:
                imgs = [url1, url2] if rng.random() < 0.5 else [url2, url1]
            else:
                imgs = [rng.choice([url1, url2])]

            prompt = rng.choice(PROMPTS)
            bucket_name, lo, hi = rng.choice(RESPONSE_LENGTH_BUCKETS)
            target = rng.randint(lo, hi)
            response = make_response(target, rng)

            ex = build_example(imgs, prompt, response)
            f.write(json.dumps(ex) + "\n")

    print(f"wrote {args.num_examples} examples -> {out.resolve()}")
    print(f"  approx ~{int(args.num_examples * args.two_image_fraction)} examples have 2 images")


if __name__ == "__main__":
    main()
