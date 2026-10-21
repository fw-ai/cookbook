# TITO failure debugging

The Harbor E2B transport default requires an E2B SDK that implements
`E2B_HTTP_VERSION` (use the example pin, 2.55.0, compatible with the cookbook's
`pyqwest==0.10.0`; E2B 2.56 instead requires pyqwest 0.12). Unsupported SDKs fail before sandbox
creation rather than recording an unverified HTTP version. An empty variable
uses the HTTP/1.1 default, just like an unset variable.

Read `error.code` and `error.origin` before investigating. HTTP status alone
does not identify a backend failure. Streaming can return HTTP 200 followed by
an `event: error` with the same structured diagnosis as a JSON error.

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
| `tito_sdk_sampling_parse_error` | `sdk_parser` | HTTP-200 decoding or completion validation: inspect sampled arrays and `validation_error_code`. |
| `tito_sdk_internal_error` | `sdk` | Inspect `phase` and the local exception chain. |
| `tito_cookbook_parse_error` | cookbook | Inspect the dumped artifact's token/logprob alignment and harness visibility. |
| `tito_cookbook_artifact_error` | cookbook collection log | Compact finalization failed; follow the retained raw JSONL pointer. The original artifact error keeps its retry classification. |
| `tito_server_http_error` | `server` | Correlate actual `upstream_status`, request IDs, and attempt history with backend evidence. |
| `tito_transport_error` | `transport` | Inspect connection/timeout and retry evidence before attributing the cause to a server. |
| `tito_debug_storage_error` | `sdk` | Opt-in strict debug storage failed; check its byte/free-space limits. |

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
