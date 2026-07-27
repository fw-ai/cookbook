# Training renderers

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
