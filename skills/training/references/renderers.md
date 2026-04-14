# Renderers

Token-level renderers produce `(ModelInput, weights)` pairs for supervised training. They live in `training/renderer/`. Use a cookbook renderer only when Tinker's built-in doesn't fit.

## Available renderers

| Renderer | File | Class |
|----------|------|-------|
| Gemma 4 | `renderer/gemma4.py` | `Gemma4Renderer` |
| Minimax M2 | `renderer/minimax_m2.py` | `MinimaxM2Renderer` |
| Nemotron | `renderer/nemotron.py` | `NemotronRenderer` |

Everything else (Qwen, Llama, DeepSeek, Kimi, etc.) resolves via Tinker: `model_info.get_recommended_renderer_name(model_name)` -- do not wrap these in custom renderers.

## Selection logic

`utils/supervised.py::resolve_renderer(base_model, tokenizer)` returns the right renderer. It checks:

1. Is the model one of `gemma-4*`, `MiniMax-M2*`, `nemotron-*`? → cookbook renderer
2. Otherwise → Tinker built-in

## Interface

```python
renderer.build_supervised_example(messages) -> tuple[ModelInput, np.ndarray]
```

- `messages`: OpenAI chat format (optional `tool_calls`, `images`)
- `ModelInput`: list of chunks (text, image)
- `weights`: `np.int32` per token; `1` = train on this token, `0` = mask out

Response-only: user/system/tool tokens get weight 0; assistant tokens get weight 1.

## Building a Datum

```python
from training.utils.supervised import conversation_to_datum

datum = conversation_to_datum(
    messages,
    renderer=renderer,
    max_length=4096,
    train_on_what="assistant_only",    # or "all", "last_turn"
)
# datum.model_input: ModelInput
# datum.loss_fn_inputs["weights"]: TensorData
```

## Testing a new renderer

Every renderer has a parity test against HuggingFace's `apply_chat_template`:

```python
# pattern from training/tests/unit/test_gemma4_renderer.py
hf_tokens = hf_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
renderer_tokens, weights = renderer.build_supervised_example(messages)
assert tokens_from(renderer_tokens) == hf_tokens, "token divergence"
```

Test suites:

```bash
pytest training/tests/unit/test_gemma4_renderer.py
pytest training/tests/unit/test_minimax_m2_renderer.py
pytest training/tests/unit/test_nemotron_renderer.py
```

Run these when you touch a renderer. Divergence in token IDs is the most common source of silent training-quality regressions.

## When to add a new renderer

Only when:

1. The model isn't supported by Tinker yet, **and**
2. HF `apply_chat_template` produces a well-defined output for it

Otherwise wait -- Tinker usually adds support shortly after a model drops.

New renderer checklist:

1. `training/renderer/<model>.py` with a class implementing `build_supervised_example(messages)`
2. Add branch to `utils/supervised.py::resolve_renderer`
3. `training/tests/unit/test_<model>_renderer.py` with HF parity tests covering: no-system, with-system, multi-turn, tool calls (if supported), vision (if supported)

## Multi-modal notes

- Image token count in `ModelInput` chunks must match the number of image placeholder tokens HF's chat template emits
- VL renderers pass images through an `image_processor` (from the HF tokenizer config); don't reinvent this
- Test with at least 1-image and 3-image conversations

## Debugging renderer regressions

When SFT quality drops after a renderer change:

1. Pick a real training conversation
2. `cookbook_tokens = renderer.build_supervised_example(messages)`
3. `hf_tokens = hf_tokenizer.apply_chat_template(messages, tokenize=True)`
4. Diff the decoded strings (not just token IDs) to see where they split
5. Check weights: `np.where(weights == 1)` should be exactly the assistant-response spans

Most renderer bugs are at boundaries: BOS token doubled, last turn dropped, tool delimiter mismatch, image token count off by one.
