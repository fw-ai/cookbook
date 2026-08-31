# Harbor RL with Pi

Pi is the default reference harness for DeepSWE through the SDK-owned
`TITOSidecar`; OpenCode and Mini-SWE-Agent are alternate adapters over the same
Harbor lifecycle. Docker and E2B use the
same environment-local loopback endpoint, immutable runtime bundle, artifact
collection, trial cleanup, and exact-token trajectory rules.

The current environment-sidecar runtime supports only GLM-5.2
with `glm_moe_dsa_preserve_thinking`. Offline renderer characterization is not
sidecar support. Other model/template pairs require a lightweight renderer,
tokenizer-bound certificate, complete-render/parser/stop/truncation coverage,
exact sampled-array checks, and live validation, or must wait for support.
Full-history prompt construction is the default. Incremental construction is
experimental even for implemented model families and requires a separately
validated suffix-and-junction contract.

This module is intentionally a rollout library, not a second training recipe:
there is no `pi/train.py`. Cookbook users construct it through
`training.examples.rl.harbor.pi.rollout.make_rollout_fn` and pass that factory
to the existing async RL loop. The caller supplies the same `RolloutSetup`
extras used by OpenCode, including `harbor_environment`,
`sample_kwargs["max_seq_len"]`, and the renderer/debug contract. The adapter starts
the sidecar on an ephemeral loopback port and gives Pi only the resulting
environment-local URL and credential; no callback, external URL, or fixed port
is configured.

The pinned Pi OpenAI-completions implementation constructs streaming requests,
so the sidecar emits periodic SSE keepalives while it buffers one complete
upstream generation transaction.

`prepare_tasks.py` appends the pinned Pi CLI to copied Harbor task images. It
is backend-neutral: Docker builds the copied context, while E2B derives its
content-addressed template from the same context. Prewarm distinct templates
before high fan-out and clean obsolete E2B templates only after all trials
stop; Harbor deletes per-trial sandboxes but not shared templates.
