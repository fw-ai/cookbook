#!/usr/bin/env python3
"""Diagnostic script: renders one VL example and verifies tokenization + weight alignment."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from transformers import AutoTokenizer

from training.utils.supervised import (
    build_datum_from_model_input_and_weights,
    build_renderer,
    render_messages_to_datum,
    resolve_renderer_name,
    parse_train_on_what,
)
import tinker

TOKENIZER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DATASET = os.path.join(os.path.dirname(__file__), "food_reasoning.jsonl")


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    renderer_name = resolve_renderer_name(TOKENIZER_MODEL)
    renderer = build_renderer(tokenizer, TOKENIZER_MODEL)
    print(f"Renderer: {renderer_name}  ({type(renderer).__name__})")

    with open(DATASET) as f:
        row = json.loads(f.readline())

    messages = row["messages"]
    print(f"\nRoles: {[m['role'] for m in messages]}")

    # ---- render via the full pipeline (same as sft_loop) ----
    train_on_what = parse_train_on_what("all_assistant_messages")
    datum_result = render_messages_to_datum(
        messages, renderer=renderer, train_on_what=train_on_what,
    )
    datum = datum_result.datum
    mi = datum.model_input

    # ---- basic stats ----
    text_tokens = 0
    image_tokens = 0
    for chunk in mi.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            text_tokens += len(chunk.tokens)
        elif hasattr(chunk, "expected_tokens"):
            image_tokens += chunk.expected_tokens
    total = mi.length

    print(f"\n=== Model Input ===")
    print(f"  Total length: {total}  (text: {text_tokens}, image: {image_tokens})")

    # ---- weights ----
    weights_data = [float(x) for x in datum.loss_fn_inputs["weights"].data]
    target_data = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
    n_zero = sum(1 for w in weights_data if w == 0.0)
    n_one = sum(1 for w in weights_data if w > 0.0)
    print(f"\n=== Weights (shifted, len={len(weights_data)}) ===")
    print(f"  weight=0 positions: {n_zero}")
    print(f"  weight=1 positions: {n_one}")

    first_one = next((i for i, w in enumerate(weights_data) if w > 0), None)
    last_one = next((i for i in range(len(weights_data) - 1, -1, -1) if weights_data[i] > 0), None)
    print(f"  First weight=1 at shifted index: {first_one}")
    print(f"  Last  weight=1 at shifted index: {last_one}")

    # ---- decode trained targets ----
    trained_target_ids = [target_data[i] for i in range(len(target_data)) if weights_data[i] > 0]
    trained_text = tokenizer.decode(trained_target_ids)
    print(f"\n=== Trained target text (weight=1 targets decoded) ===")
    print(trained_text[:500])
    if len(trained_text) > 500:
        print(f"  ... ({len(trained_text)} chars total)")

    # ---- expected assistant content ----
    asst_msg = next(m for m in messages if m["role"] == "assistant")
    asst_content = asst_msg["content"] if isinstance(asst_msg["content"], str) else "".join(
        p.get("text", "") for p in asst_msg["content"] if p.get("type") == "text"
    )
    print(f"\n=== Expected assistant content ===")
    print(asst_content[:500])

    # ---- compare text-only tokenization with apply_chat_template ----
    text_only_messages = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts_text = " ".join(
                p.get("text", "[image]") if p.get("type") == "text" else "[image]"
                for p in content
            )
            text_only_messages.append({"role": m["role"], "content": parts_text})
        else:
            text_only_messages.append(m)

    chat_template_ids = tokenizer.apply_chat_template(
        text_only_messages, tokenize=True, add_generation_prompt=False,
        return_dict=False,
    )
    if not isinstance(chat_template_ids, list) or (chat_template_ids and not isinstance(chat_template_ids[0], int)):
        # Newer transformers may return Encoding objects; extract .ids
        if hasattr(chat_template_ids, "ids"):
            chat_template_ids = chat_template_ids.ids
        elif isinstance(chat_template_ids, list) and hasattr(chat_template_ids[0], "ids"):
            chat_template_ids = chat_template_ids[0].ids
        else:
            chat_template_ids = list(chat_template_ids)
    print(f"\n=== apply_chat_template (text-only, no images) ===")
    print(f"  Token count: {len(chat_template_ids)}")
    print(f"  First 20 tokens: {chat_template_ids[:20]}")
    print(f"  Decoded first 100 chars: {tokenizer.decode(chat_template_ids[:50])!r}")

    # Extract text tokens from our VL model_input for comparison
    our_text_ids = []
    for chunk in mi.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            our_text_ids.extend(chunk.tokens)
    print(f"\n=== Our VL renderer text tokens (excluding image) ===")
    print(f"  Token count: {len(our_text_ids)}")
    print(f"  First 20 tokens: {our_text_ids[:20]}")
    print(f"  Decoded first 100 chars: {tokenizer.decode(our_text_ids[:50])!r}")

    # ---- check for special token handling ----
    im_start_direct = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end_direct = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    vis_start_direct = tokenizer.encode("<|vision_start|>", add_special_tokens=False)
    vis_end_direct = tokenizer.encode("<|vision_end|>", add_special_tokens=False)
    print(f"\n=== Special token encoding ===")
    print(f"  <|im_start|>     -> {im_start_direct}  (expected: single token)")
    print(f"  <|im_end|>       -> {im_end_direct}  (expected: single token)")
    print(f"  <|vision_start|> -> {vis_start_direct}  (expected: single token)")
    print(f"  <|vision_end|>   -> {vis_end_direct}  (expected: single token)")

    # Check if any of these split into multiple tokens (BIG red flag)
    for name, ids in [
        ("<|im_start|>", im_start_direct),
        ("<|im_end|>", im_end_direct),
        ("<|vision_start|>", vis_start_direct),
        ("<|vision_end|>", vis_end_direct),
    ]:
        if len(ids) != 1:
            print(f"  *** WARNING: {name} encoded as {len(ids)} tokens! This will break rendering. ***")

    # ---- spot-check: do the first few text tokens match? ----
    # apply_chat_template typically starts with <|im_start|>user\n...
    # Our VL renderer should start the same way
    ct_prefix = chat_template_ids[:5]
    our_prefix = our_text_ids[:5]
    print(f"\n=== Token prefix comparison ===")
    print(f"  chat_template first 5: {ct_prefix}")
    print(f"  Our renderer   first 5: {our_prefix}")
    if ct_prefix == our_prefix:
        print("  MATCH")
    else:
        print("  *** MISMATCH -- chat template mismatch detected! ***")

    print("\nDone.")


if __name__ == "__main__":
    main()
