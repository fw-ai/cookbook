# `RolloutPayload` / `TurnRecord` Schema Decision

This file records the decision that `RolloutPayload` and `TurnRecord` will
NOT be widened with provenance fields (renderer name, stop condition, model
id) in the renderer-backed RL change set.

## Question

> Does any in-scope consumer of `RolloutPayload` / `TurnRecord` need
> provenance fields (renderer name, stop condition, model id) carried on
> the payload?  If yes, the schema should be widened before any
> renderer-backed example or remote service starts depending on the
> current shape; if no, the current schema is the contract.

## In-scope consumers of the schema

| Consumer | Path | Needs provenance? |
|----------|------|-------------------|
| Trainer packer | `training/utils/rl/text_rollout.py::pack_payload_to_sample` | No.  Validates `tokenizer_id`, every-turn `token_ids`, assistant logprob alignment.  Uses `total_reward`, `_assembled` flag, `finish_reason`.  Renderer name / stop condition / model id are not consulted. |
| Trajectory assembler | `training/utils/rl/trajectory_assembler.py` | No.  Stitches `TurnRecord`s with prefix-equality; only `role`, `token_ids`, `logprobs`, `finish_reason` matter. |
| Remote-rollout helper | `training/utils/rl/text_rollout.py::make_text_rollout_fn` (re-exported as `make_remote_rollout_fn`) | No.  Just forwards payloads to the packer. |
| EP example service | `training/examples/rl/ep_remote_grader/ep_service.py` | No.  Constructs payloads directly with `tokenizer_id` only. |
| Multi-turn / tool example rollouts | `training/examples/rl/multi_turn_*/rollout.py` | No.  Use `TrajectoryAssembler.to_payload(total_reward=...)`; no provenance fields touched. |
| Mock remote service example | `training/examples/rl/remote_rollout/mock_service.py` | No.  Demonstrates the wiring contract; uses only existing fields. |
| Existing examples (`frozen_lake/`, `gsm8k_async/`, `multihop_qa/`, `deepmath/`) | various | Out of scope for this change set.  Their migration to the new helpers is deferred per AC-11; if migrated, they would also not need provenance fields (the trainer packer ignores them). |

## Trainer-side observation

The trainer's only inputs from `RolloutPayload` are:

* `turns[*].role` — packer derives `loss_mask`.
* `turns[*].token_ids` — packer concatenates them as the flat token sequence.
* `turns[i].logprobs` (assistant turns only) — flat alignment with `token_ids`.
* `turns[i].finish_reason` (last assistant) — surfaces as `RolloutSample.finish_reason`.
* `total_reward` — propagated as the sample's reward.
* `tokenizer_id` — verified against `ctx.tokenizer_id`.
* `_assembled` — fast-path flag for the `TrajectoryAssembler` packing route.

Renderer name / stop condition / model id never enter any computation on
the trainer side or the packer side.

## Decision

`RolloutPayload` and `TurnRecord` schema is the contract for this change set
and is NOT widened.  Provenance fields would buy nothing for any in-scope
consumer; adding them now would simply pin a wire-format detail that no one
reads.  If a future consumer (e.g. a logging dashboard, a per-call audit
record, a renderer-mismatch detection tool) emerges, this decision can be
revisited at that point — schema widening is reversible in a way that
schema-narrowing is not.

The mock remote service in `examples/rl/remote_rollout/mock_service.py`
intentionally does not populate any provenance field, demonstrating the
minimum-viable wire format.
