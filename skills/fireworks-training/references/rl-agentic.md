# RL: agentic rollout design

Use this reference when one logical rollout contains multiple policy calls,
tool calls, environment steps, retries, subagents, context rewrites, or
trainer-ready trajectories. Read [`rl-async.md`](rl-async.md) separately for
scheduler, batching, off-policy, optimizer, and publication behavior.

## Sections

- [Boundary and invariants](#boundary-and-invariants)
- [Choose an architecture](#choose-an-architecture)
- [Choose a token-mismatch policy](#choose-a-token-mismatch-policy)
- [Represent branches and GRPO groups](#represent-branches-and-grpo-groups)
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

The cookbook does not prescribe one session or trajectory architecture. Choose
based on harness behavior and scale:

| Architecture | Use when | Main tradeoff |
|---|---|---|
| Rollout-local recorder and token tree | Integrating an existing agent or environment with minimal infrastructure | Simple ownership and deployment; the client must retain and assemble trace state |
| Slime-style message tree with bounded token realignment | The harness may re-render a recent assistant response with small token drift | Preserves longer contiguous samples, but realignment is a heuristic and repaired tokens must be masked |
| Miles-style inference-side session service | High-volume multi-turn inference where strict append-only templating and prefix-cache efficiency justify a stateful service | Centralizes token accumulation and sample assembly, but adds service lifecycle, routing, template, and concurrency concerns |
| Strict append-only adapter | The harness and renderer can guarantee exact history extension | Strongest and easiest invariant; violations require retry or drop |

These are examples, not compatibility modes in `async_rl_loop`. A custom
adapter may combine them—for example, an inference-side session service plus a
rollout-local environment tree.

### Slime reference

[Slime's coding-agent example](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl)
records turns in a per-session message tree. Its trajectory builder can realign
a bounded drift inside the most recent response by replacing that tail with the
later prompt representation and masking the replacement; earlier or larger
divergence forks. This retains continuity for expected re-rendering drift, but
the realignment rule must be validated against the selected renderer and model.

Do not copy a threshold without replaying real traces. Record how often the
rule realigns, forks, or rejects, and inspect representative cases.

### Miles reference

[Miles' session server](https://github.com/radixark/miles/tree/main/miles/rollout/session)
is a separate service between an OpenAI-message agent and the inference router.
It owns append-only message validation, pretokenized token accumulation,
limited retry rollback, and server-side sample assembly. The agent sends
messages while the service reuses accumulated input IDs, which avoids
client-side re-tokenization and can preserve inference prefix caching. Miles
requires a verified TITO chat template and rejects unsupported history edits.

Choose this pattern for measured throughput or trace-movement benefits, not
only to make the rollout adapter look smaller. Plan service affinity, cleanup,
hotload behavior, concurrency, failure recovery, and version compatibility.

## Choose a token-mismatch policy

Compare the next prompt against the exact tokens already retained for its
declared parent. Select one policy deliberately:

### Split

Flush the prior exact-ancestry trajectory and start another trajectory from the
new prompt. Return both under the same `RolloutRun`, preserving sampled
outputs, reward, group membership, and advantage. This is the general policy
implemented by
`training.utils.rl.agent.merge_turn_segments` and used by the Harbor/OpenCode
example.

Splitting is lossless and easy to audit. It increases trainer trajectory
count, and a new trajectory cannot reuse a prior generated suffix as trainable
ancestry.

### Realign

For a narrowly defined recent-tail mismatch, replace the held tail with the
later prompt's representation, mask the replacement, and continue. Fork or
reject anything outside the validated repair window. This is similar to
Slime's bounded repair policy.

Realignment reduces trajectory splitting but changes which earlier sampled
tokens carry loss. Require replay tests proving that the mismatch is rendering
drift rather than a semantic history rewrite.

### Reject or retry

If exact append-only behavior is part of the protocol, raise a trace-integrity
failure and retry the logical rollout. Drop it after the configured retry
budget. Use this for mismatched output IDs, log probabilities, or routing
payloads that cannot be repaired without fabricating training data.

### Prevent mismatch in a session service

Let an inference-side service own the accumulated input IDs and admit only
supported message appends or rollback. This moves enforcement closer to
tokenization and caching, but it does not remove the need to validate the final
trainer payload.

## Represent branches and GRPO groups

Resolve structured-message history before token assembly. The example owns
message normalization, tool-call identity, subagent parent selection, retries,
and context-management semantics. A generic token tree should only receive the
chosen parent and exact turn snapshot.

When one logical rollout branches:

- materialize the selected root-to-leaf paths as trainer trajectories;
- train each shared generated node on only one selected path;
- include shared tokens as masked context on later paths;
- keep all trajectories in one `RolloutRun`;
- retain one environment reward, one group member, and one advantage.

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
| Policy session or affinity key | Routes model turns from one attempt to compatible serving state |
| Retry attempt ID | Prevents failed trace state and cache affinity from leaking into a new attempt |

`DeploymentTrainingSession` supplies one opaque serving `user` value for all
model calls in an attempt. Serving can use it for session affinity when no
explicit prompt-cache key is present. It is not an authorization token, cache
namespace API, active-request KV manager, or guarantee that KV survives
hotload. Create a fresh policy session for a retry unless the serving protocol
explicitly supports rollback and trace reset.

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

A retry is a new attempt: clean the environment, policy session, trace tree,
and temporary credentials before starting it. Return `None` after exhausted
retries to keep malformed data out of group assembly. Supporting an explicit
terminal-failure reward is useful, but keep it an opt-in algorithm choice.

`async_rl_loop` bounds incomplete-group retries and then advances the dataset.
If every row is rejected, it can finish with zero optimizer steps. Treat the
returned step plus `producer/trajectory_drops_total` and
`producer/rows_rejected_total` as an experiment success check.

## Responsibility split

Keep only protocol-neutral primitives in `training/utils/rl/agent`:

- serving affinity for one deployment-backed attempt;
- immutable sampled-token ancestry;
- exact token/logprob/routing alignment;
- root-to-leaf materialization and shared-prefix loss masking;
- conversion to the standard rollout contract.

Keep these in the example or user adapter:

- agent and environment APIs;
- messages, tools, renderers, parsing, and compaction semantics;
- Harbor/OpenCode/container lifecycle;
- parent selection, verifier extraction, retry policy, and artifacts;
- any external session-service client and its deployment lifecycle.

The rollout function is the gate between those layers. It should return only a
valid logical run or `None`; malformed partial data must not reach the trainer.

## Calibration checklist

Before a paid training run:

1. Run sampling-only calibration on at least five representative tasks when
   the dataset permits it.
2. Inspect high-reward, low-reward, long, multi-turn, tool-heavy, retried, and
   non-append traces.
3. Replay every observed history/mismatch decision through the current
   renderer and assembler.
4. Assert token/logprob/loss-mask lengths and optional R3 alignment at the
   rollout gate.
5. Confirm multiple returned trajectories remain one logical rollout in
   reward, advantage, grouping, and rollout metrics.
6. Measure mismatch, split/realign/reject, retry, drop, zero-turn, and rollout
   length distributions.
7. Confirm a broken trace is dropped, while a legitimate terminal environment
   result receives the configured reward.
8. Start with a short checkpointed training gate and inspect trajectories again
   before scaling the dataset.

## Cookbook example

`training/examples/rl/harbor/` is one concrete adapter:

- Harbor owns local Docker task execution and verifier reward;
- OpenCode supplies the multi-turn agent and tool behavior;
- a local policy server records exact sampler responses;
- example-local code resolves structured history and retries;
- shared agent utilities own sampler affinity and token ancestry;
- any exact-token non-append boundary starts another `RolloutSample`
  trajectory within the same logical run.

The DABstep and Terminal-Bench entrypoints keep their task selection and
experiment defaults in `harbor_rl_opencode/` and `harbor_rl_terminal_bench/`.
Fork only the pieces that match the target harness. Harbor is environment
support, OpenCode is one agent, and strict split-on-mismatch is one valid policy;
none is required by `async_rl_loop`.
