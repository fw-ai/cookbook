"""Per-(renderer, scenario) expected divergences for the property suite.

Each map is keyed ``(renderer_name, scenario_id)`` and gates ONE invariant in
:mod:`test_renderer_properties` with a ``pytest.mark.xfail(strict=True)``:

* ``HF_EXPECTED_DIVERGENCES`` — HF byte-parity (invariants 1 & 2): the renderer
  and ``apply_chat_template`` tokenize the same messages differently, or one
  side raises/rejects the input. Tool scenarios are byte-compared too (the HF
  reference gets ``tools=`` and the renderer gets its own tool block), so a
  renderer whose tool-declaration formatting diverges is recorded here rather
  than exempted.
* ``TEXT_EXPECTED_DIVERGENCES`` — the render->parse round-trip does not recover
  the visible assistant text (invariant 3, text leg).
* ``PARSE_EXPECTED_DIVERGENCES`` — the renderer's own parser cannot read back
  the structured content (tool_calls / thinking) it emitted for the FINAL
  assistant turn (invariant 3, structured leg). Self-consistency, not HF.
* ``HISTORICAL_PARSE_EXPECTED_DIVERGENCES`` — same, for a HISTORICAL assistant
  turn (intermediate tool/reasoning turns).
* ``OBSERVATION_EXPECTED_DIVERGENCES`` — the supervised observation != the
  generation prompt of the prefix (invariant 3, ob==gen leg).
* ``EXTENSION_EXPECTED_DIVERGENCES`` — the renderer violates the sequence
  extension (KV-cache prefix) property for a scenario (invariant 4).

Because ``strict=True``, a divergence that gets FIXED flips the xfail to an
XPASS and fails the suite, forcing the stale entry to be removed here — the map
can never silently rot.

Empirically derived: every entry is produced by replaying the harness's own
invariant checks against the full ``RENDERER_MATRIX`` x scenario grid against
the committed renderers (see ``sweep`` in the PR description). It is not
hand-guessed. A ``(renderer, scenario)`` pair may legitimately appear in more
than one map when it diverges on more than one invariant (e.g. a renderer whose
tool declaration diverges from HF *and* whose parser does not read the tool call
back); the maps are intentionally NOT disjoint.

Deliberately NOT listed here (handled elsewhere):

* kimi_k25 on any system-less conversation — kimi injects its default system
  prompt when no system turn is present, a *structural* HF divergence gated by a
  dedicated predicate in the test module (covers every system-less scenario at
  once). Those pairs are omitted from ``HF_EXPECTED_DIVERGENCES``.
* deepseek_v4 HF parity — its preview tokenizer ships no ``chat_template``
  (``has_hf_chat_template=False``), so the HF legs are not collected at all.
"""

from __future__ import annotations

HF_EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {
    ("glm5", "developer_first"): "GLM5 renderer raises ValueError on the 'developer' role (unsupported); there is no rendered output to byte-compare against the template.",
    ("glm5", "developer_midconversation"): "GLM5 renderer raises ValueError on the 'developer' role (unsupported); there is no rendered output to byte-compare against the template.",
    ("glm5", "tool_result_pending_answer"): "renderer/template rejects this input: 'str object' has no attribute 'items'",
    ("glm_moe_dsa", "developer_first"): "GLM5 renderer raises ValueError on the 'developer' role (unsupported); there is no rendered output to byte-compare against the template.",
    ("glm_moe_dsa", "developer_midconversation"): "GLM5 renderer raises ValueError on the 'developer' role (unsupported); there is no rendered output to byte-compare against the template.",
    ("glm_moe_dsa", "tool_result_pending_answer"): "renderer/template rejects this input: 'str object' has no attribute 'items'",
    ("kimi_k25", "thinking_tool_then_answer"): "supervised mode intentionally preserves reasoning_content inside <think> for the last conversational round (tool-call + answer turns) so SFT does not train on empty think blocks; the Kimi K2.5 HF template's hist_msgs branch always emits empty <think></think> and drops reasoning_content. A train-time policy choice, not a tool-declaration/call formatting bug.",
    ("kimi_k27_code", "multipart_thinking_text"): "thinking-channel rendering diverges from apply_chat_template",
    ("kimi_k27_code", "multipart_user_text"): "renderer diverges from apply_chat_template for this scenario",
    ("minimax_m2", "consecutive_system"): "the MiniMax-M2 template demotes a second/mid-conversation system message to a 'user' turn, while the renderer keeps it as a system turn.",
    ("minimax_m2", "developer_first"): "developer role not remapped: the renderer emits a 'developer' header while the MiniMax-M2 template renders it as a 'user' turn.",
    ("minimax_m2", "developer_midconversation"): "developer role not remapped: the renderer emits a 'developer' header while the MiniMax-M2 template renders it as a 'user' turn.",
    ("minimax_m2", "leading_newline_user"): "leading/whitespace-only content merges with the MiniMax-M2 user-header newline differently than the renderer, which keeps the newlines split.",
    ("minimax_m2", "mid_conversation_system"): "the MiniMax-M2 template demotes a second/mid-conversation system message to a 'user' turn, while the renderer keeps it as a system turn.",
    ("minimax_m2", "tool_result_pending_answer"): "renderer/template rejects this input: NotImplementedError",
    ("minimax_m2", "whitespace_only_user"): "leading/whitespace-only content merges with the MiniMax-M2 user-header newline differently than the renderer, which keeps the newlines split.",
    ("nemotron3", "leading_newline_user"): "renderer diverges from apply_chat_template for this scenario",
    ("nemotron3", "multipart_user_text"): "renderer diverges from apply_chat_template for this scenario",
    ("nemotron3", "tool_result_pending_answer"): "renderer/template rejects this input: Can only get item pairs from a mapping.",
    ("qwen3", "developer_first"): "developer role not remapped: the renderer emits a 'developer' header while the Qwen3-8B template renders it as a 'user' turn.",
    ("qwen3", "developer_midconversation"): "developer role not remapped: the renderer emits a 'developer' header while the Qwen3-8B template renders it as a 'user' turn.",
    ("qwen3", "leading_newline_user"): "leading/whitespace-only content merges with the Qwen3-8B user-header newline into a single '\\n\\n\\n'-style token in the template, while the renderer keeps the header and body newlines as separate tokens.",
    ("qwen3", "multipart_user_text"): "the Qwen3-8B template drops list (multi-part) message content entirely (it only renders string content), while the renderer joins the text parts.",
    ("qwen3", "whitespace_only_user"): "leading/whitespace-only content merges with the Qwen3-8B user-header newline into a single '\\n\\n\\n'-style token in the template, while the renderer keeps the header and body newlines as separate tokens.",
    ("qwen3_6", "consecutive_system"): "the Qwen3.6 template rejects a system message that is not first (Jinja raises 'System message must be at the beginning'); the renderer accepts consecutive / mid-conversation system turns.",
    ("qwen3_6", "developer_first"): "developer role rejected by the Qwen3.6 template (Jinja raises 'Unexpected message role'); the renderer accepts and emits it, so no byte-parity is possible.",
    ("qwen3_6", "developer_midconversation"): "developer role rejected by the Qwen3.6 template (Jinja raises 'Unexpected message role'); the renderer accepts and emits it, so no byte-parity is possible.",
    ("qwen3_6", "leading_newline_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6", "mid_conversation_system"): "the Qwen3.6 template rejects a system message that is not first (Jinja raises 'System message must be at the beginning'); the renderer accepts consecutive / mid-conversation system turns.",
    ("qwen3_6", "tab_indented_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6", "tool_result_pending_answer"): "tool declaration/call rendering diverges from apply_chat_template(tools=...) for this scenario",
    ("qwen3_6", "tool_with_response_schema"): "tool declaration/call rendering diverges from apply_chat_template(tools=...) for this scenario",
    ("qwen3_6", "trailing_newline_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6", "whitespace_only_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6", "whitespace_trim"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_disable_thinking", "consecutive_system"): "the Qwen3.6 template rejects a system message that is not first (Jinja raises 'System message must be at the beginning'); the renderer accepts consecutive / mid-conversation system turns.",
    ("qwen3_6_disable_thinking", "developer_first"): "developer role rejected by the Qwen3.6 template (Jinja raises 'Unexpected message role'); the renderer accepts and emits it, so no byte-parity is possible.",
    ("qwen3_6_disable_thinking", "developer_midconversation"): "developer role rejected by the Qwen3.6 template (Jinja raises 'Unexpected message role'); the renderer accepts and emits it, so no byte-parity is possible.",
    ("qwen3_6_disable_thinking", "leading_newline_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_disable_thinking", "mid_conversation_system"): "the Qwen3.6 template rejects a system message that is not first (Jinja raises 'System message must be at the beginning'); the renderer accepts consecutive / mid-conversation system turns.",
    ("qwen3_6_disable_thinking", "tab_indented_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_disable_thinking", "tool_result_pending_answer"): "tool declaration/call rendering diverges from apply_chat_template(tools=...) for this scenario",
    ("qwen3_6_disable_thinking", "tool_with_response_schema"): "tool declaration/call rendering diverges from apply_chat_template(tools=...) for this scenario",
    ("qwen3_6_disable_thinking", "trailing_newline_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_disable_thinking", "whitespace_only_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_disable_thinking", "whitespace_trim"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_preserve_thinking", "consecutive_system"): "the Qwen3.6 template rejects a system message that is not first (Jinja raises 'System message must be at the beginning'); the renderer accepts consecutive / mid-conversation system turns.",
    ("qwen3_6_preserve_thinking", "developer_first"): "developer role rejected by the Qwen3.6 template (Jinja raises 'Unexpected message role'); the renderer accepts and emits it, so no byte-parity is possible.",
    ("qwen3_6_preserve_thinking", "developer_midconversation"): "developer role rejected by the Qwen3.6 template (Jinja raises 'Unexpected message role'); the renderer accepts and emits it, so no byte-parity is possible.",
    ("qwen3_6_preserve_thinking", "leading_newline_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_preserve_thinking", "mid_conversation_system"): "the Qwen3.6 template rejects a system message that is not first (Jinja raises 'System message must be at the beginning'); the renderer accepts consecutive / mid-conversation system turns.",
    ("qwen3_6_preserve_thinking", "tab_indented_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_preserve_thinking", "tool_result_pending_answer"): "tool declaration/call rendering diverges from apply_chat_template(tools=...) for this scenario",
    ("qwen3_6_preserve_thinking", "tool_with_response_schema"): "tool declaration/call rendering diverges from apply_chat_template(tools=...) for this scenario",
    ("qwen3_6_preserve_thinking", "trailing_newline_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_preserve_thinking", "whitespace_only_user"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_6_preserve_thinking", "whitespace_trim"): "content not trimmed: the Qwen3.6 template applies Jinja `| trim` to message content (dropping leading/trailing whitespace, tabs, and whitespace-only bodies), while the cookbook qwen3_5/qwen3_6 renderer family preserves it — a real train/inference whitespace parity gap.",
    ("qwen3_disable_thinking", "developer_first"): "developer role not remapped: the renderer emits a 'developer' header while the Qwen3-8B template renders it as a 'user' turn.",
    ("qwen3_disable_thinking", "developer_midconversation"): "developer role not remapped: the renderer emits a 'developer' header while the Qwen3-8B template renders it as a 'user' turn.",
    ("qwen3_disable_thinking", "leading_newline_user"): "leading/whitespace-only content merges with the Qwen3-8B user-header newline into a single '\\n\\n\\n'-style token in the template, while the renderer keeps the header and body newlines as separate tokens.",
    ("qwen3_disable_thinking", "multipart_assistant_text"): "the Qwen3-8B template drops list (multi-part) message content entirely (it only renders string content), while the renderer joins the text parts.",
    ("qwen3_disable_thinking", "multipart_content_tool_call"): "the Qwen3-8B template drops list (multi-part) message content entirely (it only renders string content), while the renderer joins the text parts.",
    ("qwen3_disable_thinking", "multipart_user_text"): "the Qwen3-8B template drops list (multi-part) message content entirely (it only renders string content), while the renderer joins the text parts.",
    ("qwen3_disable_thinking", "parallel_tool_calls"): "upstream tinker_cookbook Qwen3Renderer emits a separate <|im_start|>user turn per tool result, while the HF template folds consecutive tool results into a single user turn. Real renderer bug; fix pending upstream (RenderContext.next_message + fold in render_message).",
    ("qwen3_disable_thinking", "parallel_tool_results_reversed"): "upstream tinker_cookbook Qwen3Renderer emits a separate <|im_start|>user turn per tool result, while the HF template folds consecutive tool results into a single user turn. Real renderer bug; fix pending upstream (RenderContext.next_message + fold in render_message).",
    ("qwen3_disable_thinking", "tool_only_empty_content"): "upstream tinker_cookbook Qwen3Renderer prepends a newline before <tool_call> even when the assistant turn has empty content, while the HF template guards that separator on content. Real renderer bug; fix pending upstream.",
    ("qwen3_disable_thinking", "whitespace_only_user"): "leading/whitespace-only content merges with the Qwen3-8B user-header newline into a single '\\n\\n\\n'-style token in the template, while the renderer keeps the header and body newlines as separate tokens.",
}

TEXT_EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {
}

# Empty: the only entries here were Kimi tool-call round-trips, which failed
# solely because the harness fed OpenAI-style opaque ids (``call_weather_1``).
# Kimi's wire format encodes the function NAME *inside* the tool-call id
# (``functions.<name>:<index>``) with no separate name field, so an opaque id
# renders tokens from which the name is literally absent — unrecoverable by any
# parser. The harness now feeds Kimi its native id shape on the round-trip legs
# (``RendererCase.tool_call_id_style="kimi"``), matching what Kimi's own
# generation emits, and every one of these round-trips passes. No renderer's
# parse_response currently fails the structured round-trip.
PARSE_EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {}

HISTORICAL_PARSE_EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {}

OBSERVATION_EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {
    ("qwen3_disable_thinking", "consecutive_assistant"): "consecutive assistant turns break the observation==generation-prompt invariant for qwen3_disable_thinking: the interior assistant header in the supervised observation differs from a fresh generation prompt.",
}

EXTENSION_EXPECTED_DIVERGENCES: dict[tuple[str, str], str] = {
}

