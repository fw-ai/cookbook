# Gemma 4 reserved-token SFT

Gemma 4 already has thousands of `<unusedN>` vocabulary rows. Reusing those
rows lets full-parameter SFT learn new control-token embeddings without
resizing the model.

## 1. Choose and persist the mapping

Map each desired wire string to a different reserved slot:

```bash
python training/examples/sft/prepare_gemma4_tokenizer.py \
  --source google/gemma-4-E2B-it \
  --output /tmp/gemma4-tokenizer \
  --token-map '{"<start_search>":"<unused123>","<end_search>":"<unused124>"}'
```

This changes the token strings attached to the selected IDs and registers them
for atomic encoding. It asserts that the tokenizer length does not change.
Keep the generated tokenizer files with the custom base model used for
inference; promoted checkpoints inherit tokenizer metadata from that base.

If the wire spelling `<unused123>` is acceptable, map it to itself. That only
activates the existing string:

```json
{"<unused123>": "<unused123>"}
```

## 2. Train the selected rows

Use the same mapping while rendering SFT data and run full-parameter training:

```bash
python training/examples/sft/train_sft.py \
  --base-model accounts/YOUR_ACCOUNT/models/YOUR_GEMMA4_BASE \
  --tokenizer-model /tmp/gemma4-tokenizer \
  --renderer-name gemma4 \
  --dataset-path /path/to/data.jsonl \
  --output-model-id gemma4-reserved-token-sft \
  --lora-rank 0 \
  --gemma4-reserved-token-map \
    '{"<start_search>":"<unused123>","<end_search>":"<unused124>"}'
```

The SFT recipe rejects this option with `lora_rank > 0`: LoRA does not update
the base embedding rows. Include each new token in enough supervised examples
to learn a useful embedding.

If you custom-initialize the selected rows, edit the input embedding (and the
untied output embedding, if present), save the model together with the
generated tokenizer files, and upload that directory as the custom base before
training.

Inference must decode with `skip_special_tokens=False` when these strings are
part of the model's output. Fireworks' text tokenizer uses that behavior.
