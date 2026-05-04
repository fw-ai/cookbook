# Async RL Examples

This directory includes two minimal async RL rollout examples:

- `single_turn_token_in`
- `multi_turn_message_in`

The examples keep sampler ownership in rollout code. Each example provides a
`rollout_fn_factory(setup: RolloutSetup) -> rollout_fn` that the recipe calls
once at startup; the returned closure has signature
`async def rollout_fn(sample_prompt) -> RolloutSample | None`. The
`RolloutSetup` carries the inference URL, tokenizer, sample kwargs, and an
`extras: dict` for any caller-supplied state; users do not subclass any
context type or thread a per-call `ctx` argument.

`single_turn_token_in` expects pre-tokenized dataset-row fields such as
`prompt_token_ids`. `multi_turn_message_in` accepts OpenAI-style `messages`
and uses the generic TITO adapter so generated assistant token IDs are
preserved across turns.

The `train.py` files use placeholder model and tokenizer names. Replace them
with your own public model and tokenizer identifiers before running.
