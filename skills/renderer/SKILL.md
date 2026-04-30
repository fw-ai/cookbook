# Implementing a new renderer — Skill

A renderer translates a list of `Message` objects into a flat token
sequence the trainer optimizes against. It owns the chat template, the
stop tokens, and the loss-weight assignment. The cookbook's
`tinker_cookbook.renderers.Renderer` base class is the contract; the
concrete renderers live in `training/renderer/`.

Use this skill when adding support for a new model family, or when a
PR touches a renderer's chat template, stop semantics, or weight
masking. Always pair this skill with [`skills/verifier/SKILL.md`](../verifier/SKILL.md)
— the verifier is the validation half of the loop.

## 1. Contract — four methods

| Method | What it returns | When it's called |
|---|---|---|
| `build_generation_prompt(messages)` | Flat token list ending at the model's "your turn" point | Inference / probe prompt |
| `parse_response(text)` | Structured `Message` (assistant role + parsed tool calls) | When threading a gateway completion back into the renderer |
| `build_supervised_example(messages, train_on_what)` | `(token sequence, loss-weight tensor)` | SFT — this is the loss target |
| `get_stop_sequences()` | List of token-id sequences the gateway should stop on | Passed into `chat.completions.create(stop=...)` |

`Renderer` is in `tinker_cookbook.renderers.base`. Subclass it and
implement the four methods. The base class provides chunk
abstractions (`bos`, `header`, `output`, `stop_overlap`,
`generation_suffix`) — let those carry your structure rather than
flattening tokens by hand.

## 2. Training-mechanics invariants

The verifier's audit table tags every token with a `provenance`. The
loss-weight assignment must respect:

| Provenance | Weight should be |
|---|---|
| `prompt_hard_append` (BOS, header, generation prefix) | **0** |
| `native_generated` (model emitted this) | **1** when this turn is trainable |
| `trailing_hard_append` (renderer added after the model's emission) | **0** unless this is a sanctioned `stop_overlap` |

The one sanctioned exception: `stop_overlap` weight=1 when the model
is trained to emit a next-role tag as its stop signal (GLM5's pattern
— `<|user|>` weight=1 at the end of a trainable assistant turn so SFT
teaches the model when to stop).

Violating these invariants is what the verifier's amber-rule flags
catch. Sometimes the violation is intentional (stop_overlap);
sometimes it's a bug. The verifier highlights the row; humans decide.

## 3. Common shape decisions (with examples in the tree)

### Stop signal
- **EOS-stop** (Llama 3): model emits `<|eot_id|>`. Renderer's
  `get_stop_sequences()` returns `[<|eot_id|>]`. No trailing append.
- **Next-role-tag stop** (GLM5, Kimi K2.5): model emits content; the
  renderer hard-appends the next role tag (`<|user|>` /
  `<|observation|>`) with weight=1 as `stop_overlap` so SFT teaches
  the model that boundary. Stop sequences include those role tags.

Reference: `training/renderer/glm5.py:_append_output_chunks_with_weights`
(masks the leading `<think>` token to weight 0; emits the role-tag
stop_overlap with weight 1).

### Thinking modes
Three live conventions:
- **GLM5**: `<|assistant|><think>{reasoning}</think>{content}`. The
  `<think>` opening token is part of the generation prefix —
  hard-appended, masked out of loss.
- **Qwen3**: `<think>...</think>` when `enable_thinking=True`; the
  `_disable_thinking` variant emits `<think></think>` to skip
  reasoning.
- **DeepSeek V3**: `<｜Assistant｜><think>` (thinking) or
  `<｜Assistant｜></think>` (non-thinking) prefilled deterministically.

### Tool calls
- **GLM5** uses XML-ish: `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>`.
- Other model families use JSON blobs inside `<tool_call>{...}</tool_call>`.
- Always check the upstream HF chat template before inventing your
  own serialization.

## 4. Implementation flow

1. **Skim a similar existing renderer** to scaffold. Pick the closest
   match by stop-signal convention (EOS vs next-role-tag) and
   thinking-mode handling.
2. **Match the upstream HF chat template byte-for-byte** for the
   prompt half. Where the renderer has to deviate (loss-weight
   decisions on `stop_overlap`, masking of generation-prefix tokens),
   add a docstring at the top of the file explaining each deviation.
3. **Register the renderer** at module bottom:
   ```python
   from tinker_cookbook.renderers import register_renderer
   register_renderer("my_model", _my_factory)
   ```
4. **Add a CPU HF parity test** in
   `training/tests/unit/test_renderer_hf_parity.py`. Add a
   `parametrize` case for your renderer + tokenizer. Cookbook CI runs
   this on every PR; passing means the renderer agrees with
   `tokenizer.apply_chat_template` byte-for-byte.
5. **Smoke-test live in the GUI.** From the workspace root run
   `./run.sh`, pick your renderer, type a chat, click `Run probe`.
   The amber-flagged tokens are inspection points; hover for reasons.
6. **Author a corpus and run triage.** Cover the edge cases that
   exercise the renderer's surface (empty assistant, multi-turn,
   tool-call, long content, system-only). Then
   `./triage.sh <renderer> <tokenizer> ./my-prompts.json` —
   pre-flight + every case stacked in the GUI.

## 5. Common gotchas

- **`<think>` weighted at 1 when it's part of the generation prefix.**
  Mask it out (PR #400's fix on GLM5).
- **Trailing `\n` after a special token in the generation suffix.**
  MiniMax M2 had this bug; the renderer's prompt was one token longer
  than the gateway's. Currently `xfail` in
  `test_renderer_hf_parity.py`.
- **Tokenization divergence on the assistant tail.** Often a BPE
  round-trip artifact, not a renderer bug. Check by feeding assistant
  text back through the tokenizer and seeing if the same token IDs
  come out.
- **`stop_overlap` not in `get_stop_sequences()`.** If you teach the
  model to stop on `<|user|>` via `stop_overlap`, also include
  `<|user|>` in the inference stop list so the gateway actually stops
  there.
- **Multimodal content silently dropped.** Most cookbook renderers
  accept `messages[i].content` as either a string or a list of
  content parts; if you only handle strings, image / audio cases pass
  through tokenized as text.

## 6. See also

- [`skills/verifier/SKILL.md`](../verifier/SKILL.md) — how to validate
  the renderer against the live Fireworks gateway + upstream HF chat
  template.
- `training/renderer/glm5.py`, `training/renderer/minimax_m2.py`,
  `training/renderer/gemma4.py` — concrete implementations covering
  next-role-tag stop, EOS-stop, and multimodal respectively.
- `tinker_cookbook.renderers.base` — the abstract `Renderer` class
  and the chunk types the audit table reports against.
