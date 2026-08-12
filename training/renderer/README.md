# Training renderers

## Muse Glimmer

`meta-models/Muse-Glimmer-30B` uses the registered `muse_glimmer` renderer.
It mirrors the model's ATEM chat template at revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, including synthetic-system
defaults, reasoning recipients, image placeholders, tool declarations, ATEM
calls/results, and the EOM/EOT state machine. The upstream template's symbolic
video branch is covered for parity, but cookbook training inputs reject video
parts because video preprocessing is not supported. Tool-call arguments must
be objects; raw JSON strings are rejected by the upstream template and renderer.

The renderer does not claim the sequence-extension property because appending
a consecutive assistant message changes a preceding terminal ATEM call from
`<|eot|>` to `<|eom|>`. Multi-turn SFT is therefore disaggregated into safe
per-user-turn examples. The shared renderer matrix pins HF token parity, while
`test_muse_glimmer_renderer.py` covers the template's individual branches.

## Qwen2.5 32B V1 compatibility

`Qwen/Qwen2.5-32B-Instruct` uses the dedicated `qwen2_5` renderer. It is
intentionally independent of the Qwen3 renderers and has no thinking-history
mode. The renderer preserves the imported Fireworks V1 contract: it renders
and tokenizes the whole conversation once because BPE tokens can cross message
boundaries. Text conversations and tool calls are supported; multipart content
is rejected because the V1 template rejects it.

Renderer QA loads the public `Qwen/Qwen2.5-32B-Instruct` tokenizer from HF
main, consistent with the other public-tokenizer matrix entries.

## Kimi K3

Kimi K3 uses the registered `kimi_k3` renderer (or
`kimi_k3_disable_thinking`). The model release has no Jinja chat-template
string: `tokenization_kimi.py` owns `apply_chat_template`, while
`encoding_k3.py` owns the XTML segments. The cookbook renderer resolves and
calls those Python helpers from the loaded tokenizer so there is only one
template implementation. The reviewed tokenizer/processor source is
`moonshotai/Kimi-K3` revision
`301be1b88c89c0d3a763da6301352cb8fe399e90`; training configs should use that
HF repository as `tokenizer_model` and may pin the revision through
`tokenizer_revision`.

For SFT, assistant output through `<|close|>message<|sep|>` is trainable and
the following `<|end_of_msg|>` history delimiter is retained with weight zero.
For RL, the missing Jinja string selects renderer-backed prompt tokens and the
same response parser/stop sequence. Vision inputs remain `ImageChunk` objects;
tests separately verify the symbolic placeholder, one-media-pad tokenizer
form, and expanded training sequence against the release image processor.

## Backward compatibility

Managed Training jobs may persist and reuse concrete renderer names across
retries and resumes. Consider existing jobs before changing or removing a
renderer registration or altering its rendering behavior. When corrected
semantics would change emitted tokens, register a new concrete name and point
the capability registry at it; leave the old name's implementation intact.

Only Managed Training may mark ``renderer_name_is_resolved=true``. Direct
cookbook renderer overrides leave it false so a simultaneously supplied
semantic thinking-history mode is validated rather than silently ignored.
