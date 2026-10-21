# TITO failure debugging

NeMo's `nemo_sidecar_http_error` identifies the proxy-to-TITO HTTP hop
(`transport_hop=nemo_proxy_to_tito`), not the inference backend. Even a non-JSON
HTTP error retains its status and request ID without becoming a parser discard;
retryable statuses remain recoverable. Malformed HTTP-200 bodies are instead
adapter decoding failures.

JSON sidecar HTTP errors with no origin retain `origin=unknown` and the known
proxy-to-TITO hop/status. Transient HTTP statuses on that hop remain bounded
recoverable failures; unknown-origin errors elsewhere and explicit statusless
stream rejections do not gain that retry policy.

NeMo retains adapter and reward failures with `TITOSidecar.dump_failure_trace`,
using the same configured trace root, enablement, and aggregate retention limits
as SDK failures. It never falls back to the global temporary store. A successful
native `/run` response with a missing, malformed, or non-finite reward is a fatal
environment/reward failure, even if an earlier model request returned a retryable
503. Model failures still take precedence over generic native HTTP/transport
failures; diagnosed trajectory-local rejections still discard the trajectory.
Trace-write exceptions log a payload-free storage warning without replacing
the rollout failure or changing its retry/discard classification. Missing
required SDK retention support fails before proxy startup, not on a later
failure path; there is no global-store fallback.

The Harbor E2B transport default requires an E2B SDK that implements
`E2B_HTTP_VERSION` (use the example pin, 2.55.0, compatible with the cookbook's
`pyqwest==0.10.0`; E2B 2.56 instead requires pyqwest 0.12). Unsupported SDKs fail before sandbox
creation rather than recording an unverified HTTP version. An empty variable
uses the HTTP/1.1 default, just like an unset variable.

Read `error.code` and `error.origin` before investigating. HTTP status alone
does not identify a backend failure. Streaming can return HTTP 200 followed by
an `event: error` with the same structured diagnosis as a JSON error.

Newer SDKs include `parser_reason_code`, finish reason, prompt/completion token
counts, requested/configured output budgets, renderer ID, and certification ID
on parser rejections. Examples of stable renderer reasons are
`reasoning_missing_close`, `tool_unparsed_boundary`,
`length_tool_trailing_content`, and `native_parse_rejected`. An unknown/legacy
reason is `unclassified`; the code is never inferred from exception prose.
The cookbook renderers continue to raise SDK `TITOParseError`
and work with older SDKs; older SDKs omit the additional response fields.

Async rollout logs carry a structured `rollout_error` record with selected
diagnostic fields and retained `trace_paths`, without raw model/tool/error
messages. Explicit recoverable rollout reasons such as `whole_trial_bound`
survive worker serialization and classification; only missing/invalid reasons
use `explicit_rollout_infrastructure`. Circuit-breaker snapshots and trip
messages expose bounded `reasons_in_window` and `codes_in_window` counts, not
flat producer metrics. They expire
with the rolling window, contain no request IDs as labels, and do not change
the aggregate trip thresholds. Trip messages include these counts; the original
exception remains available as the cause, not interpolated into the summary.

Harbor E2B command/file streams are a separate transport from Tito-to-inference
HTTP. The shared Harbor runner defaults E2B SDK requests to HTTP/1.1 to avoid
observed HTTP/2 command-stream resets under sustained agent output. Set
`E2B_HTTP_VERSION=2` explicitly to opt into HTTP/2; invalid values fail before
trial setup. This is process-level E2B SDK configuration and is inherited by
worker processes; Docker and inference HTTP transports are unchanged.
Sidecar launch metadata and successful rollouts record
`harbor_e2b_http_version`. There is no
automatic replay/fallback after a command-stream read failure: the command may
already have executed, so replay could duplicate side effects. This workaround
does not establish the cause of a separate customer socket-closure report.

| Code | Origin | First check |
|---|---|---|
| `tito_sdk_parse_error` | `sdk_parser` | Recognized unsafe output: inspect the original `TITOParseError`, reasoning/tool boundaries, and finish reason. |
| `tito_sdk_parser_internal_error` | `sdk_parser` | Unexpected renderer exception: replay with the pinned renderer/tokenizer and inspect the exception chain. |
| `tito_sdk_sampling_parse_error` | `sdk_parser` | Local SSE/completion decoding or sampler/token-array validation: inspect wire evidence, phase, sampled arrays and `validation_error_code`. This does not establish the cause of an incompatible response. |
| `tito_sdk_internal_error` | `sdk` | Inspect `phase` and the local exception chain. |
| `tito_cookbook_parse_error` | cookbook | Inspect the dumped artifact's token/logprob alignment and harness visibility. |
| `tito_cookbook_artifact_error` | cookbook collection log | Compact finalization failed; follow the retained raw JSONL pointer. The original artifact error keeps its retry classification. |
| `tito_server_http_error` | `server` | Correlate actual `upstream_status`, request IDs, and attempt history with backend evidence. |
| `tito_server_stream_error` | `server` | Upstream explicitly reported failure or an unsuccessful finish reason. Check supplied upstream code/type, IDs, and retry history; failed-attempt tokens are not committed. |
| `tito_transport_error` | `transport` | Inspect connection/timeout and retry evidence before attributing the cause to a server. |
| `tito_debug_storage_error` | `sdk` | Opt-in strict debug storage failed; check its byte/free-space limits. |

With SDK support for stream-integrity diagnostics, `wire_status` records the
actual upstream HTTP status; it can be 200 on an unsuccessful stream.
`final_status`/`upstream_status` describe an explicitly supplied semantic error
status, not inferred from message text. Unknown stream errors fail explicitly;
supported transient statuses use the sampler's bounded retry budget.
Statusless explicit error events fail fast; they are not inferred to be overload
or retried by the cookbook as transport failures. HTTP 400 remains a fatal
request/configuration error, unlike a diagnosed local parser rejection.
`retry_budget_exhausted` distinguishes exhaustion from fail-fast rejection.
`should_retry: false` prevents retrying the same logical request after the
sampler finishes its policy; a fresh-rollout policy is separate. Inspect
`logical_request_id`, `request_id`, `upstream_response_id`, and attempt history
where available. A connect failure may have no response IDs.
Request IDs are opaque: credential-like prefixes are not rejected. Diagnostic
copies are bounded to 256 characters and escape control characters; correlation
fields are not a general-purpose secret detector.

The buffered transport must complete before SSE interpretation. `[DONE]` ends
event parsing, not HTTP integrity checks; a broken transfer after `[DONE]` still
fails closed. Wire and semantic statuses remain separate in admission telemetry.

`tito_trajectory_closed` (410) is a local lifecycle guard, not proof that an
inference deployment closed. `termination_phase` distinguishes `closing` from
`terminal`; `terminal_action`, `requested_terminal_status`, `terminal_status`,
`terminal_reason`, and timestamps report known local intent. A parser discard
overrides a requested successful finish. `cause_error_code` is present only for
a structurally established discard cause, not simply the last observed error.
Free-form reasons are represented as `caller_supplied_reason`. These additive
fields require an SDK release containing the changes; older versions omit them.

Local parser/runtime and cookbook validation errors discard the **whole**
trajectory, including an earlier successful prefix. They are non-retryable and
do not trip the infrastructure circuit breaker. Healthy trajectories continue.
Older `tito_model_malformed_output`/502 errors also classify as local discard.
Check `producer/discard_errors_total`, accepted rows, and optimizer progress:
persistent discards can leave no usable training samples.

1. Find `trace_path`, `trajectory_id`, `call_id`, `logical_request_id`, renderer,
   and phase in the error log/response. Parser errors carry `should_retry: false`
   and `discard_trajectory: true`; JSON responses also set `x-should-retry: false`.
2. Open the path on its writer's machine. JSONL records the received request,
   prepared token IDs, raw completion before parsing, and terminal exception
   chain with request/attempt provenance. Routing/top-k arrays are omitted and
   named in `omitted_arrays`; full debug storage and compact artifacts retain
   training arrays. SSE decoding evidence is capped at 16 MiB; check truncation.
3. Harbor collects `/logs/tito/failures` and copies failed evidence before trial
   cleanup. Follow the final **host** pointer. A bundle contains
   `collected_trace_jsonl` and a base64 `.tito` artifact. Missing, unpackable, or
   oversized compact artifacts can leave raw JSONL alone. Collection rejects
   symlinks and non-regular files below the trusted host trial directory.
4. Check `trace_history_truncated`, `trace_storage_error`, and `body_truncated`.
   Missing evidence prevents an exact replay claim. Storage failure preserves
   the original diagnosis and does not crash the training loop.

Replay offline with the recorded renderer/certification/tokenizer fingerprint,
exact request, completion token IDs, and finish reason. The primary parser is
the only entry point; legacy artifact `parser_fallback` fields remain readable.
For DeepSeek V4.1, use the recipe tokenizer loader with the certification's
pinned revision. Its declared BOS/EOS post-processor is part of the fingerprint;
loading the raw backend without those settings produces a different contract.
The recipe loader's Transformers tokenizer class and model type can affect
that contract, but the hash covers only serialized backend, template and
special-token map, not class/model-type identity. After a Transformers bump,
rerun the pinned recipe-loader fingerprint and behavior tests before
recertification; a raw backend tokenizer is not an equivalent test input.
A length stop during reasoning can be valid. A parser rejection alone does not
establish whether the model, tokenizer contract, or parser caused the failure.

Capture is enabled without full debug logging. Clean completed/abandoned traces
are deleted; diagnosed failures survive within per-writer limits: 32 MiB per
trajectory, 256 MiB total, 32 failures, 128 active traces, 64 MiB free-space floor.
Old failures can be evicted. The default root is
`tempfile.gettempdir()/fireworks-tito-traces`. Standalone sidecars configure
`TITOFailureTraceConfig(root_dir=...)` or disable automatic capture with
`enabled=False`. POSIX startup reclaims only marked directories older than
seven days whose owner is a dead process on this host.

Directories/files use 0700/0600 permissions. Credentials are scrubbed, but raw
text and tokens remain sensitive. Keep traces local; share only authorized,
redacted evidence. Captured wire chunks and token IDs are distinct evidence.
