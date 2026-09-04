# RL: agentic rollout design

Use this reference when one logical rollout contains multiple policy calls,
tool calls, environment steps, retries, subagents, context rewrites, or
trainer-ready trajectories. Read [`rl-async.md`](rl-async.md) separately for
scheduler, batching, off-policy, optimizer, and publication behavior.

## Sections

- [Boundary and invariants](#boundary-and-invariants)
- [Choose an architecture](#choose-an-architecture)
- [Choose a token-mismatch policy](#choose-a-token-mismatch-policy)
- [Linear V1 and future trees](#linear-v1-and-future-trees)
- [Session and prompt-cache identity](#session-and-prompt-cache-identity)
- [Failure and retry policy](#failure-and-retry-policy)
- [Responsibility split](#responsibility-split)
- [Calibration checklist](#calibration-checklist)
- [Cookbook example](#cookbook-example)

## Boundary and invariants

Treat agentic behavior as a rollout-adapter concern, not as a different async
RL scheduler. The environment and agent may use any framework. The adapter must
turn their interaction into the standard rollout contract:

```python
async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
    ...
```

One successful call produces one logical rollout and GRPO group member. It may
return one or more trainer-ready trajectories as `RolloutSample`s in
`RolloutRun.segments`. In this reference, *trajectory* means one such trainer
input; it is not another group member. Preserve these invariants:

1. Record the exact prompt and output token IDs observed at inference. Do not
   reconstruct generated tokens from text when exact IDs are available.
2. Align every output token with its behavior log probability and, when used,
   raw log probability and Router Replay payload.
3. Set loss only on policy-generated tokens. Mask prompts, system text, user
   messages, tool results, environment observations, and repaired context.
4. Give every trajectory from the same logical run the same reward. The run
   remains one group member, receives one advantage, and broadcasts that
   advantage to all of its trajectories.
5. Make every non-append, truncation, repair, and drop decision explicit and
   observable. Never silently zip, trim, or pad malformed generated data.

Keep reward semantics separate from trace validity. A valid terminal failure
may receive reward zero when the environment defines it that way. A missing or
misaligned trace is not a task outcome and should not be converted to reward
zero merely to keep the batch full.

## Choose an architecture

The production cookbook runs one SDK-managed `TITOSidecar` inside each rollout
agent environment for message-based multi-turn RL. Alternative architectures
remain useful references, but they are not interchangeable runtime modes:

| Architecture | Use when | Main tradeoff |
|---|---|---|
| Per-environment linear TITO sidecar | OpenCode, Pi, Mini-SWE-Agent, or another existing harness runs inside a Docker container or remote sandbox | Preserves the rollout function and exact tokens without callbacks or another deployed service; each environment owns its loopback runtime and artifacts |
| Slime-style message tree with bounded token realignment | The harness may re-render a recent assistant response with small token drift | Preserves longer contiguous samples, but realignment is a heuristic and repaired tokens must be masked |
| Miles-style inference-side session service | High-volume multi-turn inference where strict append-only templating and prefix-cache efficiency justify a stateful service | Centralizes token accumulation and sample assembly, but adds service lifecycle, routing, template, and concurrency concerns |
| Strict append-only adapter | The harness and renderer can guarantee exact history extension | Strongest and easiest invariant; violations require retry or drop |

These are architecture choices, not `async_rl_loop` compatibility flags. Do not
add a tree or remote session service merely for prefix lookup, compaction, or a
subagent that can be trained as an independent trajectory.

### Current support boundary

Production TITO support is deliberately narrower than the cookbook's general
renderer registry. The lightweight sidecar runtime currently implements only
`glm_moe_dsa_preserve_thinking` with the pinned GLM-5.2 tokenizer revision.
Other renderer names in the offline SFT/DPO registry are not thereby available
through the sidecar. Interleaved GLM history remains uncertified. A renderer
name existing for SFT or DPO does **not** make it safe for TITO, and an offline
renderer alone does not imply a lightweight sidecar implementation. Both
builders fail closed at their respective unsupported boundary.

For another model family, implement the shared loss-agnostic conversation and
assistant-parse primitives, characterize complete multi-turn
text/reasoning/tool/stop rendering against live sampled token IDs, add a
reviewed source-controlled TITO certification, and run the renderer verifier
before training. The template need not be incrementally appendable, but the
complete rendered prompt, output parser, and aligned sampled arrays must be
exact. Incremental prompt construction is experimental and is a separate,
stronger capability: it
also requires append-only message matching, synthetic-anchor suffix tests, and
model-specific token-junction coverage. Until that work lands, use a certified
full-history model implementation or wait for support; do not bypass either
check merely because an SFT renderer or chat template exists. Defining
`prepare_incremental_prompt` is an explicit renderer-author assertion, not
automatic coverage inherited from the base renderer certification.

### Slime reference

[Slime's coding-agent example](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl)
records turns in a per-session message tree. Its trajectory builder can realign
a bounded drift inside the most recent response by replacing that tail with the
later prompt representation and masking the replacement; earlier or larger
divergence forks. This retains continuity for expected re-rendering drift, but
the realignment rule must be validated against the selected renderer and model.

Do not copy a threshold without replaying real traces. Record how often the
rule realigns, forks, or rejects, and inspect representative cases.

### Miles Session v2 reference

[Miles' Session v2](https://github.com/radixark/miles/tree/main/miles/rollout/session/v2)
is a separate service between an OpenAI-message agent and the inference router.
It owns append-only message validation, pretokenized token accumulation,
limited retry rollback, and server-side sample assembly. The agent sends
messages while the service reuses accumulated input IDs, which avoids
client-side re-tokenization and can preserve inference prefix caching. Miles
requires a verified TITO chat template and rejects unsupported history edits.

Fireworks' experimental incremental strategy acknowledges and follows Miles's
linear construction mechanism: render appended messages under a synthetic
system/assistant anchor, merge that suffix with the exact sampled checkpoint,
and let a model-specific junction rule declare a bounded trailing edit. It does
not copy Miles's session tree, branching, leaf selection, or centralized
placement; Fireworks keeps independent linear lineage in each environment-local
sidecar.

Choose this pattern for measured throughput or trace-movement benefits, not
only to make the rollout adapter look smaller. Plan service affinity, cleanup,
hotload behavior, concurrency, failure recovery, and version compatibility.

## Choose prompt construction and a token-mismatch policy

`full_history` is the default. It renders every incoming request's complete
messages and tools, uses those IDs for inference, and compares them with the
exact tokens already retained in the active segment. The comparison selects a
training-materialization disposition; it never changes or repeats inference.

Experimental `incremental` mode runs before inference. It preserves the
previous exact checkpoint,
renders appended messages under a synthetic structural anchor, and applies a
renderer-declared token-junction rule. It deliberately does not require equality with a
full replay, because that would erase the historical replay drift it is meant
to avoid. Unsupported joins split onto a full-rendered prompt or reject under
strict policy. `realign` is therefore a full-history disposition; successful
incremental continuation is an exact `append`.

### Split

Close the valid exact segment and fully render the actual incoming request as a
new segment. Return both under the same `RolloutRun`, preserving sampled
outputs, reward, group membership, and advantage. This is the production
`TITOSidecar` default for compaction, semantic history rewrites, length-closed
turns, render-contract changes, and every mismatch ineligible for bounded
realignment.

Splitting is lossless and easy to audit. It increases trainer trajectory
count, and a new segment cannot reuse a prior generated suffix as trainable
ancestry.

### Realign

Realignment is an optional best-effort packing optimization inspired by Slime's
coding-agent materializer. It is eligible only when semantic history is
unchanged and the first mismatch lies inside the most recent assistant response.
Compute the exact coverage that would be reconstructed:

```text
masked_tokens = len(current_full_prompt) - start_of_previous_assistant_response
```

When `masked_tokens < max_masked_tokens`, replace from the start of that
assistant response with the corresponding exact full-prompt tail and mask the
complete replacement. Earlier exact actions remain trainable. Equality or a
larger span splits, and `max_masked_tokens=0` disables realignment. The bound is
on training coverage discarded, not on the length of the newly sampled action.
Never search for a plausible token alignment or reconstruct sampled output from
text.

### Reject or retry

If exact append-only behavior is part of the protocol, raise a trace-integrity
failure and retry the logical rollout. Drop it after the configured retry
budget. Use this for mismatched output IDs, log probabilities, or routing
payloads that cannot be repaired without fabricating training data.

### Prevent mismatch in a session service

Let an inference-side service own the accumulated input IDs and admit only
supported message appends or rollback. This moves enforcement closer to
tokenization and caching, but it is an alternative centralized architecture,
not a mode of the per-environment sidecar. It does not remove the need to
validate the final trainer payload.

## Linear V1 and future trees

V1 stores one ordered `Trajectory` containing linear exact-token `Segment`s and
committed policy `Turn`s. A compaction or incompatible history closes a segment;
it does not create a hidden branch. A subagent that can be scored independently
gets another trajectory. Retry creates a fresh trajectory, environment, bearer
credential, and serving-affinity key.

A future tree is justified only when the product must retain simultaneous
sibling continuations from one exact checkpoint, select among those siblings
later, or assign credit across a shared generated trunk without training that
trunk twice. Those are tree-only capabilities. Prefix-cache lookup, sequential
retries, and post-compaction continuation do not require ancestry state.

Do not call every non-append history "compaction." Dynamic system fields,
subagent roots, retries, tool rewrites, renderer changes, and genuine context
compaction can all produce different histories. Record the observed boundary
kind; infer a cause only when the harness exposes it.

## Session and prompt-cache identity

Keep these identities separate:

| Identity | Purpose |
|---|---|
| Logical rollout ID | Reward, artifacts, metrics, and GRPO membership |
| Environment trial ID | Sandbox/container lifecycle and verifier result |
| TITO trajectory ID | Selects one independent linear engine inside the environment sidecar |
| Serving-affinity key | Routes model turns from one attempt to a compatible rollout replica |
| Retry attempt ID | Prevents failed trace state and cache affinity from leaking into a new attempt |

Each rollout attempt binds a stable typed `prompt_cache_key` for all policy and
auxiliary model calls. It is deliberately different from the TITO
trajectory ID and loopback bearer credential. A cache hit is an optimization
and never proves token lineage. Compaction keeps the rollout affinity key even
though the new segment's first request establishes a different prefix on that
replica. Create a fresh trajectory and affinity key for a retry.

If throughput requires stronger cache semantics, measure and adopt an explicit
session service. Document who owns session creation, routing, rollback,
collection, deletion, sampler-version changes, and abandoned-request cleanup.

## Failure and retry policy

Classify failures at the rollout boundary:

| Failure | Default disposition |
|---|---|
| Valid verifier or terminal environment outcome | Return the environment reward, including zero when defined |
| Transient environment, container, or transport failure | Retry the complete logical rollout within a bounded budget |
| Token, log-probability, routing, or unparseable trace-integrity failure | Retry from a clean attempt; drop after exhaustion |
| Unexpected ordinary adapter exception | Log the traceback and discard at the adapter's documented trajectory boundary, or propagate if the adapter does not own that boundary |
| Control-plane cancellation | Propagate; do not convert it to a trajectory drop |

A retry is a new attempt: clean the environment, TITO trajectory, serving
affinity, and temporary credentials before starting it. Return `None` after exhausted
retries to keep malformed data out of group assembly. Supporting an explicit
terminal-failure reward is useful, but keep it an opt-in algorithm choice.

`async_rl_loop` bounds incomplete-group retries and then advances the dataset.
If every row is rejected, it can finish with zero optimizer steps. Treat the
returned step plus `producer/trajectory_drops_total` and
`producer/rows_rejected_total` as an experiment success check.

## Responsibility split

The SDK owns protocol-neutral request normalization, exact-token linear
trajectory state, full-prompt drift classification, bounded masked realignment,
safe segmentation, transaction/idempotency, typed sampler facts, serving
affinity, metrics, buffered OpenAI transport, and optional local JSONL debug
evidence. The compact `.tito` artifact remains authoritative. The SDK contains
no OpenCode, Pi, Mini-SWE-Agent, Harbor, or model-family switch.

The existing cookbook renderer registry owns tokenizer/message conversion,
reasoning and tool parsing, the loss-agnostic full-conversation primitive, and
reviewed TITO certification artifacts. SFT, DPO, and TITO share that primitive
but keep separate training materializers.

Keep these in each harness adapter:

- agent and environment APIs;
- policy-versus-auxiliary classification and compaction entry points;
- harness retry identity and compatibility settings;
- OpenCode/Pi/Mini-SWE-Agent/Harbor/container lifecycle and local tool timeouts;
- verifier extraction, semantic trace acceptance, retry policy, and artifacts.

The frozen `training.utils.rl.agent` and `rollout.message` modules remain only
as a public-import compatibility island. Production examples do not import or
extend them.

The rollout function is the gate between those layers. It should return only a
valid logical run or `None`; malformed partial data must not reach the trainer.

## Calibration checklist

Before a paid training run:

1. Run sampling-only calibration on at least five representative tasks when
   the dataset permits it.
2. Inspect high-reward, low-reward, long, multi-turn, tool-heavy, retried, and
   non-append traces.
3. Reconstruct every recorded complete prompt and replay every observed
   clean/realign/new-segment decision through the certified renderer contract.
4. Assert token/logprob/loss-mask lengths and optional R3 alignment at the
   rollout gate.
5. Confirm multiple returned segments remain one logical rollout in reward,
   advantage, grouping, and rollout metrics.
6. Measure append, realign/masked-token, new-segment/reject, retry, drop,
   zero-turn, and rollout-length distributions.
7. Confirm a broken trace is dropped, while a legitimate terminal environment
   result receives the configured reward.
8. Start with a short checkpointed training gate and inspect trajectories again
   before scaling the dataset.

## Cookbook example

`training/examples/rl/harbor/opencode/`, `harbor/pi/`, and `harbor/mini_swe/`
are concrete adapters over the same sidecar contract:

- Harbor owns local Docker task execution and verifier reward;
- OpenCode, Pi, or Mini-SWE-Agent supplies the multi-turn agent and tool behavior;
- one environment-local `TITOSidecar` records exact sampler responses and may
  route multiple independent linear trajectories;
- the shared renderer layer resolves structured model messages and parses tools;
- harness-local code owns classification, retry, compaction, and lifecycle;
- full-history mode renders every request before inference, while experimental
  incremental mode joins the stored checkpoint to a model-owned suffix; any
  incompatible boundary closes the segment and uses a full-rendered current
  prompt as the next masked segment within the same logical run.

The DABstep and Terminal-Bench entrypoints keep their task selection and
experiment defaults in `harbor/recipes/dabstep/` and
`harbor/recipes/terminal_bench/`.
The default qualification and training path is **Pi + DeepSWE + one
environment-local TITO sidecar in `full_history` mode**. OpenCode and
Mini-SWE-Agent are alternate adapters over the same harness-neutral contract;
they are not dependencies of the default path. The shipped Pi command disables
unrelated extensions and rejects `/tree` and `/fork`; Pi subagent/child support
is not claimed. Supporting it requires the owning adapter to launch each child
with a separately created TITO trajectory and credential. Fork only the adapter
pieces that match the target harness. Harbor is environment support, neither
agent is required by `async_rl_loop`, and all retain the ordinary
`rollout_fn(sample_prompt) -> RolloutRun | None` boundary.
