# Async RL Examples

This directory includes two minimal async RL rollout examples:

- `single_turn_token_in`
- `multi_turn_message_in`

The examples keep sampler ownership in rollout code. The async training recipe
passes metadata and a fixed-width request gate through `RolloutContext`; it does
not inject a sampler, rollout provider, or rollout engine.
Users can subclass `RolloutContext` or pass `ctx_extras` when their rollout
engine needs more state.

`single_turn_token_in` expects pre-tokenized row fields such as
`prompt_token_ids`. `multi_turn_message_in` accepts OpenAI-style `messages` and
uses the generic TITO adapter so generated assistant token IDs are preserved
across turns.

The `train.py` files use placeholder model and tokenizer names. Replace them
with your own public model and tokenizer identifiers before running.
