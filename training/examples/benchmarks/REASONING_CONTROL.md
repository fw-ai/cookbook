# Reasoning Control by Provider

Each provider exposes a different knob for turning reasoning/thinking on, off,
or to a specific effort level. Use this when sending OpenAI-style chat completion
requests.

## Quick reference

| Provider              | Endpoint / model example                          | Turn OFF                                              | Set effort level                                  | Reasoning streamed in   |
|-----------------------|---------------------------------------------------|-------------------------------------------------------|---------------------------------------------------|-------------------------|
| Fireworks             | `accounts/fireworks/routers/glm-5p2-fast`         | `"reasoning_effort": "none"`                          | `"reasoning_effort": "low"\|"medium"\|"high"\|"max"` | `delta.reasoning_content` |
| Baseten               | `zai-org/GLM-5.2`                                 | `"reasoning_effort": "none"`                          | `"reasoning_effort": "<level>"`                   | `delta.reasoning_content` |
| Together              | `zai-org/GLM-5.2`                                 | `"chat_template_kwargs": {"enable_thinking": false}`  | `"reasoning_effort": "<level>"`                   | `delta.reasoning`         |
| Anthropic (Opus 4.8)  | `claude-opus-4-8`                                 | `"thinking": {"type": "disabled"}`                    | `"thinking": {"type": "adaptive"}` (no `reasoning_effort`) | `delta.reasoning_content` (via LiteLLM) |

## Example request bodies

### Fireworks / Baseten — turn OFF

```json
{
  "model": "accounts/fireworks/routers/glm-5p2-fast",
  "messages": [{"role": "user", "content": "..."}],
  "reasoning_effort": "none"
}
```

### Fireworks / Baseten — turn ON at a level

```json
{
  "model": "accounts/fireworks/routers/glm-5p2-fast",
  "messages": [{"role": "user", "content": "..."}],
  "reasoning_effort": "high"
}
```

### Together — turn OFF

```json
{
  "model": "zai-org/GLM-5.2",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

### Together — turn ON at a level

```json
{
  "model": "zai-org/GLM-5.2",
  "messages": [{"role": "user", "content": "..."}],
  "reasoning_effort": "high"
}
```

### Anthropic (Opus 4.8) — turn OFF

```json
{
  "model": "claude-opus-4-8",
  "messages": [{"role": "user", "content": "..."}],
  "thinking": {"type": "disabled"}
}
```

### Anthropic (Opus 4.8) — turn ON

```json
{
  "model": "claude-opus-4-8",
  "messages": [{"role": "user", "content": "..."}],
  "thinking": {"type": "adaptive"}
}
```

## Gotchas

- **Reasoning token field name differs:** Fireworks/Baseten use
  `delta.reasoning_content`; Together uses `delta.reasoning`. Read both:
  `delta.reasoning_content ?? delta.reasoning`.
- **Anthropic has no `reasoning_effort`** — use the `thinking` object instead,
  and it **rejects `temperature`** on reasoning models (Opus 4.8). Omit
  `temperature` for `anthropic/*`, or strip it (e.g. LiteLLM `drop_params=True`).
- **Effort levels:** `low | medium | high | max`. GLM supports `max`;
  `none` disables on Fireworks/Baseten.
- **Default (omit everything):** each provider uses its own default reasoning
  behavior — don't send any of these fields if you want the provider default.
